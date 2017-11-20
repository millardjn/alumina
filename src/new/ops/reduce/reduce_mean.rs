use new::graph::{GraphDef, NodeID, OpID, PassID, DataID, Storage, GraphShapes, Result};
use new::ops::{standard_op_name, Op, OpInstance, Pass};
use new::shape::NodeShape;
use ndarray::Dimension;
use std::any::Any;
use smallvec::SmallVec;


/// ReduceMean
#[derive(Clone)] 
pub struct ReduceMean {
	name: Option<String>,
	input_id: NodeID,
	output_id: NodeID,
	axes: SmallVec<[isize; 6]>,
	keep_dims: bool,
}

impl ReduceMean {

	pub fn new(input_id: &NodeID, output_id: &NodeID) -> Self{
		ReduceMean {
			name: None,
			input_id: input_id.clone(),
			output_id: output_id.clone(),
			axes: SmallVec::new(),
			keep_dims: false
		}
	}
	
	/// Supply which axes are to be reduced across.
	///
	/// If axes is empty, all axes are reduced.
	/// Each element of `axes` can be in the range [-input.ndims(), input.ndims()).
	///
	/// Default: empty
	pub fn axes(mut self, axes: &[isize]) -> Self {
		self.axes = axes.iter().cloned().collect();
		self
	}

	/// If `true` the reduced axes still appear in the output with size 1, otherwise they are removed.
	///
	/// Default: `false`
	pub fn keep_dims(mut self, keep_dims: bool) -> Self {
		self.keep_dims = keep_dims;
		self
	}
}

impl Op for ReduceMean {
	type InstanceType = ReduceMeanInstance;

	fn type_name(&self) -> &'static str {
		"ReduceMean"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		let name = standard_op_name(&self, &self.name, graph, &[self.input_id.clone()], &[self.output_id.clone()]);

		Ok(ReduceMeanInstance{
			name: name,
			input_id: self.input_id.clone(),
			output_id: self.output_id.clone(),
			axes: self.axes.clone(),
			keep_dims: self.keep_dims,
			forward_id:graph.add_pass(ReduceMeanForward::new(
				self.input_id.clone(),
				self.output_id.clone(),
				self.axes.clone(),
				self.keep_dims,
			)),
			backward_id:graph.add_pass(ReduceMeanBackward::new(
				self.input_id.clone(),
				self.output_id.clone(),
				self.axes.clone(),
				self.keep_dims,
			)),
		})
	}
}

#[derive(Debug, Clone)]
pub struct ReduceMeanInstance {
	name: String,
	input_id: NodeID,
	output_id: NodeID,
	axes: SmallVec<[isize; 6]>,
	keep_dims: bool,
	forward_id: PassID,
	backward_id: PassID,
}

impl OpInstance for ReduceMeanInstance {
	fn instance_name(&self) -> &str {&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		(
			vec![self.input_id.clone()],
			vec![self.output_id.clone()]
		)
	}

	fn inner_passes(&self) -> Vec<PassID> {
		vec![self.forward_id.clone(), self.backward_id.clone()]
	}

	fn inner_ops(&self) -> Vec<OpID> {vec![]}

	fn inner_nodes(&self) -> Vec<NodeID> {vec![]}

	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{

		let input_shape = shapes.get_shape(&self.input_id).to_data_shape()?;

		let output_shape: NodeShape = calc_output_shape(input_shape.slice(), &self.axes, self.keep_dims).into();

		shapes.merge_with(&self.output_id, &output_shape)?;
		Ok(())
	}
}

fn calc_output_shape(input_shape: &[usize], axes: &[isize], keep_dims: bool) -> SmallVec<[usize; 6]> {
	let reduce_mask = reduction_mask(input_shape.len(), &axes);
	if keep_dims {
		input_shape.iter().zip(&reduce_mask).map(|(&dim, &reduce)| {
				if reduce {1} else {dim}
			}).collect()
	} else {
		input_shape.iter().zip(&reduce_mask).filter_map(|(&dim, &reduce)| {
				if reduce {None} else {Some(dim)}
			}).collect()
	}
}

/// Returns a mask indicating whether an axis should be reduced based on the axes list
/// If axes is empty this returns all true,
/// else only the axis provided are marked true.
fn reduction_mask(len: usize, axes: &[isize]) -> SmallVec<[bool; 6]> {
	let mut reduce = SmallVec::with_capacity(len);
	if axes.len() == 0 {
		for _ in 0..len {
			reduce.push(true);
		}
	} else {
		for _ in 0..len {
			reduce.push(false);
		}
		for axis in axes {
			reduce[(axis + len as isize) as usize % len] = true;
		}
	}
	reduce
}


#[derive(Debug, Clone)]
pub struct ReduceMeanForward {
	input_id: NodeID,
	output_id: NodeID,
	axes: SmallVec<[isize; 6]>,
	keep_dims: bool,
}

impl ReduceMeanForward {
	pub fn new(input_id: NodeID, output_id: NodeID, axes: SmallVec<[isize; 6]>, keep_dims: bool) -> Self{
		ReduceMeanForward {
			input_id,
			output_id,
			axes,
			keep_dims,
		}
	}
}

impl Pass for ReduceMeanForward {
	fn type_name(&self) -> &'static str {"ReduceMeanForward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input_id.value_id()],
		vec![self.output_id.value_id()])
	}

	fn run(&self, data: &Storage) -> Result<Box<Any>> {
		let input = data.get(&self.input_id.value_id())?;
		let output = data.get_mut(&self.output_id.value_id())?;

		let input_shape: SmallVec<[usize; 6]> = input.shape().iter().cloned().collect();
		let output_shape: SmallVec<[usize; 6]> = output.shape().iter().cloned().collect();

		let divisor: usize = input_shape.iter().zip(reduction_mask(input_shape.len(), &self.axes)).filter_map(|(dim, reduce)| if reduce{Some(dim)} else {None}).product();

		let output_shape_actual = calc_output_shape(&input_shape, &self.axes, self.keep_dims);
		let output_shape_keep_dims = calc_output_shape(&input_shape, &self.axes, true);

		ensure!(output_shape_actual.as_slice() == output_shape.as_slice(), "Output shape {:?} does not match reduced input shape {:?}", output_shape.as_slice(), output_shape_actual.as_slice());

		let mut output = output.into_shape(output_shape_keep_dims.as_slice()).expect("This should have been caught on the line above");
		for in_chunk in input.exact_chunks(output_shape_keep_dims.as_slice()) {
			output.scaled_add(1.0/divisor as f32, &in_chunk);
		}

		Ok(Box::new(()))
	}
}


#[derive(Debug, Clone)]
pub struct ReduceMeanBackward {
	input_id: NodeID,
	output_id: NodeID,
	axes: SmallVec<[isize; 6]>,
	keep_dims: bool,
}

impl ReduceMeanBackward {
	pub fn new(input_id: NodeID, output_id: NodeID, axes: SmallVec<[isize; 6]>, keep_dims: bool) -> Self{
		ReduceMeanBackward {
			input_id,
			output_id,
			axes,
			keep_dims,
		}
	}
}

impl Pass for ReduceMeanBackward {
	fn type_name(&self) -> &'static str {"ReduceMeanBackward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.output_id.gradient_id()],
		vec![self.input_id.gradient_id()])
	}

	fn run(&self, data: &Storage) -> Result<Box<Any>> {
		let mut input_grad = data.get_mut(&self.input_id.gradient_id())?;
		let output_grad = data.get(&self.output_id.gradient_id())?;

		let input_shape: SmallVec<[usize; 6]> = input_grad.shape().iter().cloned().collect();
		let output_shape: SmallVec<[usize; 6]> = output_grad.shape().iter().cloned().collect();

		let divisor: usize = input_shape.iter().zip(reduction_mask(input_shape.len(), &self.axes)).filter_map(|(dim, reduce)| if reduce{Some(dim)} else {None}).product();

		let output_shape_actual: SmallVec<[usize; 6]> = calc_output_shape(&input_shape, &self.axes[..], self.keep_dims);
		let output_shape_keep_dims: SmallVec<[usize; 6]> = calc_output_shape(&input_shape, &self.axes[..], true);

		ensure!(output_shape_actual.as_slice() == output_shape.as_slice(), "Output shape {:?} does not match reduced input shape {:?}", output_shape.as_slice(), output_shape_actual.as_slice());

		let output_grad = output_grad.into_shape(output_shape_keep_dims.as_slice()).expect("This should have been caught on the line above");
		for mut in_grad_chunk in input_grad.exact_chunks_mut(output_shape_keep_dims.as_slice()) {
			in_grad_chunk.scaled_add(1.0/divisor as f32, &output_grad);
		}

		Ok(Box::new(()))
	}
}


#[test]
fn test_reduce_mean_backprop(){
	_reduce_mean_backprop().unwrap();
}

fn _reduce_mean_backprop() -> Result<()>{
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use new::ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 2, 11, 3, 5], "input", tag![])?;
	let node2 = g.new_node(shape![7, 11, 5], "output", tag![])?;
	let node3 = g.new_node(shape![7, 11, 5], "target", tag![])?;

	let _o1 = g.new_op(ReduceMean::new(&node1, &node2).axes(&[-2, 1]), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.002;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}


#[test]
fn test_reduce_mean_keep_dims_backprop(){
	_reduce_mean_keep_dims_backprop().unwrap();
}

fn _reduce_mean_keep_dims_backprop() -> Result<()>{
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use new::ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 2, 11, 3, 5], "input", tag![])?;
	let node2 = g.new_node(shape![7, 1, 11, 1, 5], "output", tag![])?;
	let node3 = g.new_node(shape![7, 1, 11, 1, 5], "target", tag![])?;

	let _o1 = g.new_op(ReduceMean::new(&node1, &node2).axes(&[-2, 1]).keep_dims(true), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.002;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}