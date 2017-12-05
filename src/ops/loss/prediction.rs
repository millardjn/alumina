use graph::{GraphDef, NodeID, OpID, PassID, DataID, Storage, GraphShapes, Result, ErrorKind};
use ops::{standard_op_name, Op, OpInstance, Pass};
use shape::{NodeShape, NodeDim};
use ndarray::{Dimension, IxDyn};
use std::any::Any;
use smallvec::SmallVec;
use std::f32;


/// `Prediction` is a non differentiable classification loss
///
/// This operation does not provide gradients but rather just returns a 1 or 0 depending if the max value in input group is paired with a `1.0` in the target group.
/// The output shape is the same as the input/target shape with the axes provided for grouping removed.
/// If `keep_dims(true)` then the provided axes arent removed but are instead replaced with `1`.
#[must_use]
#[derive(Clone, Debug)]
pub struct Prediction {
	name: Option<String>,
	input_id: NodeID,
	target_id: NodeID,
	output_id: NodeID,
	axes: SmallVec<[isize; 6]>,
	keep_dims: bool,
}

impl Prediction {
	pub fn new(input_id: &NodeID, target_id: &NodeID, output_id: &NodeID) -> Self {
		Prediction{
			name: None,
			input_id: input_id.clone(),
			target_id: target_id.clone(),
			output_id: output_id.clone(),
			axes: SmallVec::new(),
			keep_dims: false,
		}
	}

	/// The axes supplied will be grouped together when finding the maximum of the input,
	/// with the operation repeated across the axes not supplied.
	///
	/// `axes` can be in the range [-input.ndims(), input.ndims());
	/// If no axes are supplied then all dimensions with Known size will be grouped.
	pub fn axes(mut self, axes: &[isize]) -> Self {
		self.axes = axes.iter().cloned().collect();
		self
	}

	/// If `true` the grouped axes still appear in the output with size 1, otherwise they are removed.
	///
	/// Default: `false`
	pub fn keep_dims(mut self, keep_dims: bool) -> Self {
		self.keep_dims = keep_dims;
		self
	}
}

impl Op for Prediction {
	type InstanceType = PredictionInstance;

	fn type_name(&self) -> &'static str {
		"Prediction"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(mut self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		let name = standard_op_name(&self, &self.name, graph, &[self.input_id.clone()], &[self.output_id.clone()]);

		{
			let input_shape = graph.node_shape(&self.input_id)?;
			if self.axes.len() == 0 {
				for i in 0..input_shape.ndim() {
					if matches!(input_shape.dimensions()[i], NodeDim::Known(_)) {
						self.axes.push(i as isize);
					}
				}
			}
		}

		Ok(PredictionInstance{
			name: name,
			input_id: self.input_id.clone(),
			target_id: self.target_id.clone(),
			output_id: self.output_id.clone(),
			axes: self.axes.clone(),
			keep_dims: self.keep_dims,
			forward_id:graph.add_pass(PredictionForward::new(
				self.input_id.clone(),
				self.target_id.clone(),
				self.output_id.clone(),
				self.axes.clone(),
				self.keep_dims,
			)),
		})
	}
}

#[derive(Debug, Clone)]
pub struct PredictionInstance {
	name: String,
	input_id: NodeID,
	target_id: NodeID,
	output_id: NodeID,
	axes: SmallVec<[isize; 6]>,
	keep_dims: bool,
	forward_id: PassID,
}

impl OpInstance for PredictionInstance {
	fn instance_name(&self) -> &str {&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		(
			vec![self.input_id.clone(), self.target_id.clone()],
			vec![self.output_id.clone()]
		)
	}

	fn inner_passes(&self) -> Vec<PassID> {
		vec![self.forward_id.clone()]
	}

	fn inner_ops(&self) -> Vec<OpID> {vec![]}

	fn inner_nodes(&self) -> Vec<NodeID> {vec![]}

	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{

		let input_shape = shapes.get_shape(&self.input_id).to_data_shape()?;
		let target_shape = shapes.get_shape(&self.target_id).to_data_shape()?;
		ensure!(target_shape == input_shape, "input shape doesnt match target shape");

		let output_shape: NodeShape = calc_output_shape(input_shape.slice(), &self.axes, self.keep_dims).into();

		shapes.merge_with(&self.output_id, &output_shape)?;
		Ok(())
	}
}

fn calc_output_shape(input_shape: &[usize], axes: &[isize], keep_dims: bool) -> SmallVec<[usize; 6]> {
	let group_mask = group_mask(input_shape.len(), &axes);
	if keep_dims {
		input_shape.iter().zip(&group_mask).map(|(&dim, &group)| {
				if group {1} else {dim}
			}).collect()
	} else {
		input_shape.iter().zip(&group_mask).filter_map(|(&dim, &group)| {
				if group {None} else {Some(dim)}
			}).collect()
	}
}

fn group_mask(len: usize, axes: &[isize]) -> SmallVec<[bool; 6]> {
	let mut group = SmallVec::with_capacity(len);
	for _ in 0..len {
		group.push(false);
	}
	for axis in axes {
		group[(axis + len as isize) as usize % len] = true;
	}
	group
}

#[derive(Debug, Clone)]
struct PredictionForward {
	input_id: NodeID,
	target_id: NodeID,
	output_id: NodeID,
	axes: SmallVec<[isize; 6]>,
	keep_dims: bool,
}

impl PredictionForward {
	pub fn new(input_id: NodeID, target_id: NodeID, output_id: NodeID, axes: SmallVec<[isize; 6]>, keep_dims: bool) -> Self{
		PredictionForward  {
			input_id,
			target_id,
			output_id,
			axes,
			keep_dims,
		}
	}
}

impl Pass for PredictionForward {
	fn type_name(&self) -> &'static str {"PredictionForward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(
			vec![self.input_id.value_id(), self.target_id.value_id()],
			vec![self.output_id.value_id()]
		)
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input = data.get(&self.input_id.value_id())?;
		let target = data.get(&self.target_id.value_id())?;
		let mut output = data.get_mut(&self.output_id.value_id())?;

		let input_shape: SmallVec<[usize; 6]> = input.shape().iter().cloned().collect();
		let output_shape: SmallVec<[usize; 6]> = output.shape().iter().cloned().collect();

		let group_mask = group_mask(input_shape.len(), &self.axes);

		let output_shape_actual = calc_output_shape(&input_shape, &self.axes, self.keep_dims);
		//let output_shape_keep_dims = calc_output_shape(&input_shape, &self.axes, true);

		ensure!(output_shape_actual.as_slice() == output_shape.as_slice(), "Output shape {:?} does not match reduced input shape {:?}", output_shape.as_slice(), output_shape_actual.as_slice());
		ensure!(input.shape() == target.shape(),ErrorKind::PassError(self.instance_name(data.graph()), format!("input shape: {:?} did not match target shape: {:?}", input.shape(), target.shape())));

		let group_shape: Vec<usize> = input.shape().iter().enumerate().map(|(i, dim)| if group_mask[i] {*dim} else {1}).collect();

		let output_len = output.len();

		let iter = input.exact_chunks(group_shape.as_slice()).into_iter()
			.zip(target.exact_chunks(group_shape.as_slice()))
			.zip(output.iter_mut());
		let mut count = 0;
		for ((input_chunk, mut target_chunk), output_element) in iter {
			let (max_index, _max) = input_chunk.indexed_iter()
				.fold( (IxDyn::zeros(input_chunk.ndim()), f32::NEG_INFINITY), |(max_index, max), (index, &v)| {
						if v > max {(index, v)} else {(max_index, max)}
					});
			
			if target_chunk[max_index] != 1.0 {
				*output_element += 1.0;
			}
			count +=1;
		}
		assert_eq!(count, output_len);
		Ok(Box::new(()))
	}
}


#[test]
fn test_prediction(){
	_prediction().unwrap();
}

fn _prediction() -> Result<()>{
	use graph::GraphDef;
	use ndarray::ArrayD;

	let mut g = GraphDef::new();

	let input = g.new_node(shape![7, 9, 5], "input", tag![])?;
	let target = g.new_node(shape![7, 9, 5], "target", tag![])?;

	let output = g.new_node(shape![9], "output", tag![])?;

	let _o1 = g.new_op(Prediction::new(&input, &target, &output).axes(&[0, -1]), tag![])?;

	let mut subgraph = g.subgraph(&[input.value_id(), target.value_id()], &[output.value_id()])?;

	let coords = vec![
		((2,3), (2,3), 0.0),
		((0,0), (0,0), 0.0),
		((4,1), (6,4), 1.0),
		((2,4), (0,3), 1.0),
		((5,3), (5,3), 0.0),
		((2,0), (1,1), 1.0),
		((3,4), (3,4), 0.0),
		((6,4), (6,4), 0.0),
		((2,2), (6,2), 1.0),
	];
	
	let mut input_arr = ArrayD::zeros(IxDyn(&[7, 9, 5]));
	let mut target_arr = ArrayD::zeros(IxDyn(&[7, 9, 5]));

	for (i, coord) in coords.iter().enumerate() {
		input_arr[IxDyn(&[(coord.0).0, i, (coord.0).1])] = 0.1;
		target_arr[IxDyn(&[(coord.1).0, i, (coord.1).1])] = 1.0;
	}

	let storage = subgraph.execute(vec![input_arr, target_arr])?;
	let out = storage.get_mut(&output.value_id())?;

	let expect: Vec<f32> = coords.iter().map(|coords| coords.2).collect();

	assert_eq!(out.as_slice().unwrap(), expect.as_slice());

	Ok(())
}