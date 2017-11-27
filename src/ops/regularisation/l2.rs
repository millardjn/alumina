use graph::{GraphDef, NodeID, OpID, PassID, DataID, Storage, GraphShapes, Result};
use ops::{standard_op_name, Op, OpInstance, Pass};
use ops::loss::LossType;
use shape::NodeShape;
use smallvec::SmallVec;
use ndarray::{Dimension, Zip};
use std::any::Any;

/// An `Op` which implements a loss equal to the L2 norm
///
/// By default this `Op` has no output and will generate loss and gradients.
///
/// If `output()` is set, the L2 loss will be written to that Node,
/// and instead of generating gradients this loss function will backprop gradients from the output node.
pub struct L2 {
	input_id: NodeID,
	output: Option<NodeID>,
	mean_axes: SmallVec<[isize; 6]>,
	keep_dims: bool,
	multiplier: f32,
	name: Option<String>,
}

impl L2 {
	pub fn new(input: &NodeID) -> Self {
		L2 {
			input_id: input.clone(),
			output: None,
			mean_axes: SmallVec::new(),
			keep_dims: false,
			multiplier: 1.0,
			name: None,
		}
	}

	/// If set this `Op` will output to the supplied node, any rely no other use ops to generate loss and gradients
	/// The output node must have the same size as the input node unless reductions are applied using `.mean_axes()`.
	///
	/// Default: None.
	pub fn output(mut self, output: &NodeID) -> Self {
		self.output = Some(output.clone());
		self
	}

	/// The axes supplied will be grouped when finding the mean,
	/// with the operation repeated across the axes not supplied.
	///
	/// `axes` can be in the range [-input.ndims(), input.ndims());
	/// If no axes are supplied then no mean operation is applied.
	pub fn mean_axes(mut self, mean_axes: &[isize]) -> Self {
		self.mean_axes = mean_axes.iter().cloned().collect();
		self
	}

	/// If `true` the reduced axes still appear in the output with size 1, otherwise they are removed.
	///
	/// Default: `false`
	pub fn keep_dims(mut self, keep_dims: bool) -> Self {
		self.keep_dims = keep_dims;
		self
	}

	/// Applies a multiplier to the output or to the loss generated.
	pub fn multiplier(mut self, multiplier: f32) -> Self {
		self.multiplier = multiplier;
		self
	}
}


impl Op for L2 {
	type InstanceType = L2Instance;

	fn type_name(&self) -> &'static str {
		"L2"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {

		let name =  if let Some(ref output_id) = self.output {
			standard_op_name(&self, &self.name, graph, &[self.input_id.clone()], &[output_id.clone()])
		} else {
			standard_op_name(&self, &self.name, graph, &[self.input_id.clone()], &[])
		};

		let loss_type = if let Some(output_id) = self.output {
			LossType::Output{
				output_id: output_id.clone(),
				forward_id: graph.add_pass(L2Forward::new(
					self.multiplier,
					self.input_id.clone(),
					output_id.clone(),
					self.mean_axes.clone(),
					self.keep_dims)),
				backward_id: graph.add_pass(L2Backward::new(
					self.multiplier,
					self.input_id.clone(),
					output_id.clone(),
					self.mean_axes.clone(),
					self.keep_dims)),
			}
		} else {
			LossType::Joint{
				pass_id: graph.add_pass(L2JointPass::new(
					self.multiplier,
					self.input_id.clone(),
					self.mean_axes.clone()))
			}
		};

		Ok(L2Instance{
			name: name,
			multiplier: self.multiplier,
			input_id: self.input_id.clone(),
			loss_type: loss_type,
			mean_axes: self.mean_axes,
			keep_dims: self.keep_dims,
		})
	}
}


#[derive(Clone, Debug)] 
pub struct L2Instance {
	name: String,
	multiplier: f32,
	input_id: NodeID,
	loss_type: LossType,
	mean_axes: SmallVec<[isize; 6]>,
	keep_dims: bool,
}

impl OpInstance for L2Instance {

	fn instance_name(&self) -> &str {&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		match &self.loss_type {
			&LossType::Joint{..} => (vec![self.input_id.clone()], vec![]),
			&LossType::Output{ref output_id, ..} => (vec![self.input_id.clone()], vec![output_id.clone()]),
		}
	}

	fn inner_passes(&self) -> Vec<PassID> {
		match &self.loss_type {
			&LossType::Joint{ref pass_id} => vec![pass_id.clone()],
			&LossType::Output{ref forward_id, ref backward_id, ..} => vec![forward_id.clone(), backward_id.clone()],
		}
	}

	fn inner_ops(&self) -> Vec<OpID> {
		vec![]
	}

	fn inner_nodes(&self) -> Vec<NodeID> {
		vec![]
	}

	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{
		if let &LossType::Output{ref output_id, ..} = &self.loss_type {
			let input1_shape = shapes.get_shape(&self.input_id).to_data_shape()?;
			let output_shape: NodeShape = calc_output_shape(input1_shape.slice(), &self.mean_axes, self.keep_dims).into();
			shapes.merge_with(output_id, &output_shape)
		} else {
			Ok(())
		}
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
fn reduction_mask(len: usize, axes: &[isize]) -> SmallVec<[bool; 6]> {
	let mut reduce = SmallVec::with_capacity(len);
	for _ in 0..len {
		reduce.push(false);
	}
	for axis in axes {
		reduce[(axis + len as isize) as usize % len] = true;
	}
	reduce
}

#[derive(Clone, Debug)]
struct L2JointPass {
	multiplier: f32,
	input_id: NodeID,
	mean_axes: SmallVec<[isize; 6]>,
}

impl L2JointPass {
	pub fn new(multiplier: f32, input_id: NodeID, mean_axes: SmallVec<[isize; 6]>) -> Self {
		L2JointPass {
			multiplier,
			input_id,
			mean_axes,
		}
	}
}

impl Pass for L2JointPass {
	fn type_name(&self) -> &'static str {"L2JointPass"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input_id.value_id()],
		vec![self.input_id.gradient_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input = data.get(&self.input_id.value_id())?;

		let input_shape: SmallVec<[usize; 6]> = input.shape().iter().cloned().collect();

		let divisor: usize = input_shape.iter().zip(reduction_mask(input_shape.len(), &self.mean_axes)).filter_map(|(dim, reduce)| if reduce{Some(dim)} else {None}).product();
		let multiplier = self.multiplier/divisor as f32;

		let output_shape_keep_dims = calc_output_shape(&input_shape, &self.mean_axes, true);

		let mut error = 0.0;

		let mut input_grad = data.get_mut(&self.input_id.gradient_id())?;

		let iter1 = input.exact_chunks(output_shape_keep_dims.as_slice()).into_iter();
		let iter2 = input_grad.exact_chunks_mut(output_shape_keep_dims.as_slice()).into_iter();

		for (input_chunk, mut input_grad_chunk) in iter1.zip(iter2) {
			Zip::from(&input_chunk) 
			.and(&mut input_grad_chunk) 
			.apply(|input, input1_grad| { 
				error += input*input*multiplier;
				*input1_grad +=  2.0*input*multiplier;
			});
		}

		data.loss_add(error);

		Ok(Box::new(()))
	}
}


#[derive(Clone, Debug)]
struct L2Forward {
	multiplier: f32,
	input_id: NodeID,
	output_id: NodeID,
	mean_axes: SmallVec<[isize; 6]>,
	keep_dims: bool,
}

impl L2Forward {
	pub fn new(multiplier: f32, input_id: NodeID, output_id: NodeID, mean_axes: SmallVec<[isize; 6]>, keep_dims: bool) -> Self {
		L2Forward {
			multiplier,
			input_id,
			output_id,
			mean_axes,
			keep_dims,
		}
	}
}

impl Pass for L2Forward {
	fn type_name(&self) -> &'static str {"L2Forward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input_id.value_id()],
		vec![self.output_id.value_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input = data.get(&self.input_id.value_id())?;
		let output = data.get_mut(&self.output_id.value_id())?;

		let input_shape: SmallVec<[usize; 6]> = input.shape().iter().cloned().collect();
		let output_shape: SmallVec<[usize; 6]> = output.shape().iter().cloned().collect();

		let divisor: usize = input_shape.iter().zip(reduction_mask(input_shape.len(), &self.mean_axes)).filter_map(|(dim, reduce)| if reduce{Some(dim)} else {None}).product();
		let multiplier = self.multiplier/divisor as f32;

		let output_shape_actual = calc_output_shape(&input_shape, &self.mean_axes, self.keep_dims);
		let output_shape_keep_dims = calc_output_shape(&input_shape, &self.mean_axes, true);

		ensure!(output_shape_actual.as_slice() == output_shape.as_slice(), "Output shape {:?} does not match reduced input shape {:?}", output_shape.as_slice(), output_shape_actual.as_slice());

		let iter = input.exact_chunks(output_shape_keep_dims.as_slice()).into_iter();

		let mut output = output.into_shape(&output_shape_keep_dims[..]).expect("This should have been caught by the ensure above");;

		for input_chunk in iter {
			Zip::from(&mut output) 
			.and(&input_chunk) 
			.apply(|output, input| { 
				*output += input*input * multiplier;
			}); 
		}

		Ok(Box::new(()))
	}
}

#[derive(Clone, Debug)]
struct L2Backward {
	multiplier: f32,
	input_id: NodeID,
	output_id: NodeID,
	mean_axes: SmallVec<[isize; 6]>,
	keep_dims: bool,
}

impl L2Backward {
	pub fn new(multiplier: f32, input_id: NodeID, output_id: NodeID, mean_axes: SmallVec<[isize; 6]>, keep_dims: bool) -> Self {
		L2Backward {
			multiplier,
			input_id,
			output_id,
			mean_axes,
			keep_dims,
		}
	}
}

impl Pass for L2Backward {
	fn type_name(&self) -> &'static str {"L2Backward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input_id.value_id(), self.output_id.gradient_id()],
		vec![self.input_id.gradient_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input = data.get(&self.input_id.value_id())?;
		let output_grad = data.get(&self.output_id.gradient_id())?;

		let input_shape: SmallVec<[usize; 6]> = input.shape().iter().cloned().collect();
		let output_shape: SmallVec<[usize; 6]> = output_grad.shape().iter().cloned().collect();

		let divisor: usize = input_shape.iter().zip(reduction_mask(input_shape.len(), &self.mean_axes)).filter_map(|(dim, reduce)| if reduce{Some(dim)} else {None}).product();
		let multiplier = self.multiplier/divisor as f32;

		let output_shape_actual = calc_output_shape(&input_shape, &self.mean_axes, self.keep_dims);
		let output_shape_keep_dims = calc_output_shape(&input_shape, &self.mean_axes, true);

		ensure!(output_shape_actual.as_slice() == output_shape.as_slice(), "Output shape {:?} does not match reduced input shape {:?}", output_shape.as_slice(), output_shape_actual.as_slice());

		let output_grad = output_grad.into_shape(&output_shape_keep_dims[..]).expect("This should have been caught by the ensure above");;


		let mut input_grad = data.get_mut(&self.input_id.gradient_id())?;

		let iter1 = input.exact_chunks(output_shape_keep_dims.as_slice()).into_iter();
		let iter2 = input_grad.exact_chunks_mut(output_shape_keep_dims.as_slice()).into_iter();

		for (input_chunk, mut input_grad_chunk) in iter1.zip(iter2) {
			Zip::from(&output_grad) 
			.and(&input_chunk) 
			.and(&mut input_grad_chunk) 
			.apply(|output_grad, input, input_grad| { 
				*input_grad +=  2.0*input*multiplier*output_grad;
			});
		}

		Ok(Box::new(()))
	}
}

#[test]
fn test_l2_backprop(){
	_l2_backprop().unwrap();
}

fn _l2_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;

	let _o1 = g.new_op(L2::new(&node1), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}

#[test]
fn test_l2_output_backprop(){
	_l2_output_backprop().unwrap();
}

fn _l2_output_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ordermap::OrderMap;
	use ops::loss::proportional::Proportional;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;
	let node2 = g.new_node(shape![5], "output", tag![])?;

	let _o1 = g.new_op(L2::new(&node1).mean_axes(&[0, -1]).output(&node2), tag![])?;
	let _o2 = g.new_op(Proportional::new(&node2), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}