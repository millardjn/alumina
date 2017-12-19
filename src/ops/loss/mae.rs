use graph::{GraphDef, GraphShapes, ErrorKind, Result};
use id::{NodeID, DataID, OpID, PassID};
use storage::Storage;
use ops::{standard_op_name, Op, OpInstance, Pass};
use ops::loss::LossType;
use shape::NodeShape;
use smallvec::SmallVec;
use ndarray::{Dimension, Zip};
use std::any::Any;

/// An `Op` which implements the Mean Absolute Error
///
/// By default this `Op` has no output and will generate loss and gradients.
///
/// If `output()` is set, the Mae loss will be written to that Node,
/// and instead of generating gradients this loss function will backprop gradients from the output node.
#[must_use]
#[derive(Clone, Debug)]
pub struct Mae {
	input1_id: NodeID,
	input2_id: NodeID,
	output: Option<NodeID>,
	mean_axes: SmallVec<[isize; 6]>,
	keep_dims: bool,
	multiplier: f32,
	name: Option<String>,
}

impl Mae {
	pub fn new(input1: &NodeID, input2: &NodeID) -> Self {
		Mae {
			input1_id: input1.clone(),
			input2_id: input2.clone(),
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


impl Op for Mae {
	type InstanceType = MaeInstance;

	fn type_name(&self) -> &'static str {
		"Mae"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef) -> Result<Self::InstanceType> {

		let name =  if let Some(ref output_id) = self.output {
			standard_op_name(&self, &self.name, graph, &[self.input1_id.clone(), self.input2_id.clone()], &[output_id.clone()])
		} else {
			standard_op_name(&self, &self.name, graph, &[self.input1_id.clone(), self.input2_id.clone()], &[])
		};

		let loss_type = if let Some(output_id) = self.output {
			LossType::Output{
				output_id: output_id.clone(),
				forward_id: graph.add_pass(MaeForward::new(
					self.multiplier,
					self.input1_id.clone(),
					self.input2_id.clone(),
					output_id.clone(),
					self.mean_axes.clone(),
					self.keep_dims)),
				backward_id: graph.add_pass(MaeBackward::new(
					self.multiplier,
					self.input1_id.clone(),
					self.input2_id.clone(),
					output_id.clone(),
					self.mean_axes.clone(),
					self.keep_dims)),
			}
		} else {
			LossType::Joint{
				pass_id: graph.add_pass(MaeJointPass::new(
					self.multiplier,
					self.input1_id.clone(),
					self.input2_id.clone(),
					self.mean_axes.clone()))
			}
		};

		Ok(MaeInstance{
			name: name,
			multiplier: self.multiplier,
			input1_id: self.input1_id.clone(),
			input2_id: self.input2_id.clone(),
			loss_type: loss_type,
			mean_axes: self.mean_axes,
			keep_dims: self.keep_dims,
		})
	}
}


#[derive(Clone, Debug)] 
pub struct MaeInstance {
	name: String,
	multiplier: f32,
	input1_id: NodeID,
	input2_id: NodeID,
	loss_type: LossType,
	mean_axes: SmallVec<[isize; 6]>,
	keep_dims: bool,
}

impl OpInstance for MaeInstance {

	fn name(&self) -> &str {&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		match &self.loss_type {
			&LossType::Joint{..} => (vec![self.input1_id.clone(), self.input2_id.clone()], vec![]),
			&LossType::Output{ref output_id, ..} => (vec![self.input1_id.clone(), self.input2_id.clone()], vec![output_id.clone()]),
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
			let input1_shape = shapes.get_shape(&self.input1_id).to_data_shape()?;
			let input2_shape = shapes.get_shape(&self.input2_id).to_data_shape()?;
			ensure!(input1_shape == input2_shape, "Shape of input1 did not match shape of input2");
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
struct MaeJointPass {
	multiplier: f32,
	input1_id: NodeID,
	input2_id: NodeID,
	mean_axes: SmallVec<[isize; 6]>,
}

impl MaeJointPass {
	pub fn new(multiplier: f32, input1_id: NodeID, input2_id: NodeID, mean_axes: SmallVec<[isize; 6]>) -> Self {
		MaeJointPass {
			multiplier,
			input1_id,
			input2_id,
			mean_axes,
		}
	}
}

impl Pass for MaeJointPass {
	fn type_name(&self) -> &'static str {"MaeJointPass"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input1_id.value_id(), self.input2_id.value_id()],
		vec![self.input1_id.gradient_id(), self.input2_id.gradient_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input1 = data.get(&self.input1_id.value_id())?;
		let input2 = data.get(&self.input2_id.value_id())?;

		ensure!(
			input2.shape() == input1.shape(),
			ErrorKind::PassError(self.name(), format!("input1 shape: {:?} did not match input2 shape: {:?}", input2.shape(), input1.shape()))
		);

		let input_shape: SmallVec<[usize; 6]> = input1.shape().iter().cloned().collect();

		let divisor: usize = input_shape.iter().zip(reduction_mask(input_shape.len(), &self.mean_axes)).filter_map(|(dim, reduce)| if reduce{Some(dim)} else {None}).product();
		let multiplier = self.multiplier/divisor as f32;

		//let output_shape_actual = calc_output_shape(&input_shape, &self.axes, self.keep_dims);
		let output_shape_keep_dims = calc_output_shape(&input_shape, &self.mean_axes, true);

		let mut error = 0.0;

		if data.is_required(&self.input1_id.gradient_id()) && data.is_required(&self.input2_id.gradient_id()) {
			let mut input1_grad = data.get_mut(&self.input1_id.gradient_id())?;
			let mut input2_grad = data.get_mut(&self.input2_id.gradient_id())?;

			let iter1 = input1.exact_chunks(output_shape_keep_dims.as_slice()).into_iter()
				.zip(input2.exact_chunks(output_shape_keep_dims.as_slice()));
			let iter2 = input1_grad.exact_chunks_mut(output_shape_keep_dims.as_slice()).into_iter()
				.zip(input2_grad.exact_chunks_mut(output_shape_keep_dims.as_slice()));

			for ((input1_chunk, input2_chunk), (mut input1_grad_chunk, mut input2_grad_chunk)) in iter1.zip(iter2) {
				Zip::from(&input1_chunk) 
				.and(&input2_chunk)
				.and(&mut input1_grad_chunk) 
				.and(&mut input2_grad_chunk) 
				.apply(|input1, input2, input1_grad, input2_grad| { 
					let diff = input1-input2;
					error += diff.abs()*multiplier;
					*input1_grad +=  diff.signum()*multiplier;
					*input2_grad += -diff.signum()*multiplier;
				});
			}
		} else if data.is_required(&self.input1_id.gradient_id()) {
			let mut input1_grad = data.get_mut(&self.input1_id.gradient_id())?;

			let iter1 = input1.exact_chunks(output_shape_keep_dims.as_slice()).into_iter()
				.zip(input2.exact_chunks(output_shape_keep_dims.as_slice()));
			let iter2 = input1_grad.exact_chunks_mut(output_shape_keep_dims.as_slice()).into_iter();

			for ((input1_chunk, input2_chunk), mut input1_grad_chunk) in iter1.zip(iter2) {
				Zip::from(&input1_chunk) 
				.and(&input2_chunk)
				.and(&mut input1_grad_chunk) 
				.apply(|input1, input2, input1_grad| { 
					let diff = input1-input2;
					error += diff.abs()*multiplier;
					*input1_grad +=  diff.signum()*multiplier;
				});
			}
		} else if data.is_required(&self.input2_id.gradient_id()) {
			let mut input2_grad = data.get_mut(&self.input2_id.gradient_id())?;

			let iter1 = input1.exact_chunks(output_shape_keep_dims.as_slice()).into_iter()
				.zip(input2.exact_chunks(output_shape_keep_dims.as_slice()));
			let iter2 = input2_grad.exact_chunks_mut(output_shape_keep_dims.as_slice()).into_iter();

			for ((input1_chunk, input2_chunk), mut input2_grad_chunk) in iter1.zip(iter2) {
				Zip::from(&input1_chunk) 
				.and(&input2_chunk)
				.and(&mut input2_grad_chunk) 
				.apply(|input1, input2, input2_grad| { 
					let diff = input1-input2;
					error += diff.abs()*multiplier;
					*input2_grad += -diff.signum()*multiplier;
				});
			}
		}

		data.loss_add(error);

		Ok(Box::new(()))
	}
}


#[derive(Clone, Debug)]
struct MaeForward {
	multiplier: f32,
	input1_id: NodeID,
	input2_id: NodeID,
	output_id: NodeID,
	mean_axes: SmallVec<[isize; 6]>,
	keep_dims: bool,
}

impl MaeForward {
	pub fn new(multiplier: f32, input1_id: NodeID, input2_id: NodeID, output_id: NodeID, mean_axes: SmallVec<[isize; 6]>, keep_dims: bool) -> Self {
		MaeForward {
			multiplier,
			input1_id,
			input2_id,
			output_id,
			mean_axes,
			keep_dims,
		}
	}
}

impl Pass for MaeForward {
	fn type_name(&self) -> &'static str {"MaeForward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input1_id.value_id(), self.input2_id.value_id()],
		vec![self.output_id.value_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input1 = data.get(&self.input1_id.value_id())?;
		let input2 = data.get(&self.input2_id.value_id())?;
		let output = data.get_mut(&self.output_id.value_id())?;

		ensure!(
			input2.shape() == input1.shape(),
			ErrorKind::PassError(self.name(), format!("input1 shape: {:?} did not match input2 shape: {:?}", input2.shape(), input1.shape()))
		);

		let input_shape: SmallVec<[usize; 6]> = input1.shape().iter().cloned().collect();
		let output_shape: SmallVec<[usize; 6]> = output.shape().iter().cloned().collect();

		let divisor: usize = input_shape.iter().zip(reduction_mask(input_shape.len(), &self.mean_axes)).filter_map(|(dim, reduce)| if reduce{Some(dim)} else {None}).product();
		let multiplier = self.multiplier/divisor as f32;

		let output_shape_actual = calc_output_shape(&input_shape, &self.mean_axes, self.keep_dims);
		let output_shape_keep_dims = calc_output_shape(&input_shape, &self.mean_axes, true);

		ensure!(output_shape_actual.as_slice() == output_shape.as_slice(), "Output shape {:?} does not match reduced input shape {:?}", output_shape.as_slice(), output_shape_actual.as_slice());

		let iter = input1.exact_chunks(output_shape_keep_dims.as_slice()).into_iter()
			.zip(input2.exact_chunks(output_shape_keep_dims.as_slice()));

		let mut output = output.into_shape(&output_shape_keep_dims[..]).expect("This should have been caught by the ensure above");;

		for (input1_chunk, input2_chunk) in iter {
			Zip::from(&mut output) 
			.and(&input1_chunk) 
			.and(&input2_chunk) 
			.apply(|output, input1, input2| { 
				let diff = input1 - input2;
				*output += diff.abs() * multiplier;
			}); 
		}

		Ok(Box::new(()))
	}
}

#[derive(Clone, Debug)]
struct MaeBackward {
	multiplier: f32,
	input1_id: NodeID,
	input2_id: NodeID,
	output_id: NodeID,
	mean_axes: SmallVec<[isize; 6]>,
	keep_dims: bool,
}

impl MaeBackward {
	pub fn new(multiplier: f32, input1_id: NodeID, input2_id: NodeID, output_id: NodeID, mean_axes: SmallVec<[isize; 6]>, keep_dims: bool) -> Self {
		MaeBackward {
			multiplier,
			input1_id,
			input2_id,
			output_id,
			mean_axes,
			keep_dims,
		}
	}
}

impl Pass for MaeBackward {
	fn type_name(&self) -> &'static str {"MaeBackward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input1_id.value_id(), self.input2_id.value_id(), self.output_id.gradient_id()],
		vec![self.input1_id.gradient_id(), self.input2_id.gradient_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input1 = data.get(&self.input1_id.value_id())?;
		let input2 = data.get(&self.input2_id.value_id())?;
		let output_grad = data.get(&self.output_id.gradient_id())?;

		ensure!(
			input2.shape() == input1.shape(),
			ErrorKind::PassError(self.name(), format!("input1 shape: {:?} did not match input2 shape: {:?}", input2.shape(), input1.shape()))
		);

		let input_shape: SmallVec<[usize; 6]> = input1.shape().iter().cloned().collect();
		let output_shape: SmallVec<[usize; 6]> = output_grad.shape().iter().cloned().collect();

		let divisor: usize = input_shape.iter().zip(reduction_mask(input_shape.len(), &self.mean_axes)).filter_map(|(dim, reduce)| if reduce{Some(dim)} else {None}).product();
		let multiplier = self.multiplier/divisor as f32;

		let output_shape_actual = calc_output_shape(&input_shape, &self.mean_axes, self.keep_dims);
		let output_shape_keep_dims = calc_output_shape(&input_shape, &self.mean_axes, true);

		ensure!(output_shape_actual.as_slice() == output_shape.as_slice(), "Output shape {:?} does not match reduced input shape {:?}", output_shape.as_slice(), output_shape_actual.as_slice());

		let output_grad = output_grad.into_shape(&output_shape_keep_dims[..]).expect("This should have been caught by the ensure above");;

		if data.is_required(&self.input1_id.gradient_id()) && data.is_required(&self.input2_id.gradient_id()) {
			let mut input1_grad = data.get_mut(&self.input1_id.gradient_id())?;
			let mut input2_grad = data.get_mut(&self.input2_id.gradient_id())?;

			let iter1 = input1.exact_chunks(output_shape_keep_dims.as_slice()).into_iter()
				.zip(input2.exact_chunks(output_shape_keep_dims.as_slice()));
			let iter2 = input1_grad.exact_chunks_mut(output_shape_keep_dims.as_slice()).into_iter()
				.zip(input2_grad.exact_chunks_mut(output_shape_keep_dims.as_slice()));

			for ((input1_chunk, input2_chunk), (mut input1_grad_chunk, mut input2_grad_chunk)) in iter1.zip(iter2) {
				Zip::from(&output_grad) 
				.and(&input1_chunk) 
				.and(&input2_chunk)
				.and(&mut input1_grad_chunk) 
				.and(&mut input2_grad_chunk) 
				.apply(|output_grad, input1, input2, input1_grad, input2_grad| { 
					let diff = input1-input2;
					*input1_grad +=  diff.signum()*multiplier*output_grad;
					*input2_grad += -diff.signum()*multiplier*output_grad;
				});
			}

		} else if data.is_required(&self.input1_id.gradient_id()) {
			let mut input1_grad = data.get_mut(&self.input1_id.gradient_id())?;

			let iter1 = input1.exact_chunks(output_shape_keep_dims.as_slice()).into_iter()
				.zip(input2.exact_chunks(output_shape_keep_dims.as_slice()));
			let iter2 = input1_grad.exact_chunks_mut(output_shape_keep_dims.as_slice()).into_iter();

			for ((input1_chunk, input2_chunk), mut input1_grad_chunk) in iter1.zip(iter2) {
				Zip::from(&output_grad) 
				.and(&input1_chunk) 
				.and(&input2_chunk)
				.and(&mut input1_grad_chunk) 
				.apply(|output_grad, input1, input2, input1_grad| { 
					let diff = input1-input2;
					*input1_grad +=  diff.signum()*multiplier*output_grad;
				});
			}
		} else if data.is_required(&self.input2_id.gradient_id()) {
			let mut input2_grad = data.get_mut(&self.input2_id.gradient_id())?;

			let iter1 = input1.exact_chunks(output_shape_keep_dims.as_slice()).into_iter()
				.zip(input2.exact_chunks(output_shape_keep_dims.as_slice()));
			let iter2 = input2_grad.exact_chunks_mut(output_shape_keep_dims.as_slice()).into_iter();

			for ((input1_chunk, input2_chunk), mut input2_grad_chunk) in iter1.zip(iter2) {
				Zip::from(&output_grad) 
				.and(&input1_chunk) 
				.and(&input2_chunk)
				.and(&mut input2_grad_chunk) 
				.apply(|output_grad, input1, input2, input2_grad| { 
					let diff = input1-input2;
					*input2_grad += -diff.signum()*multiplier*output_grad;
				});
			}
		}


		Ok(Box::new(()))
	}
}

#[test]
fn test_mae_backprop(){
	_mae_backprop().unwrap();
}

fn _mae_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "input2", tag![])?;

	let _o1 = g.new_op(Mae::new(&node1, &node2), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.005;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}

#[test]
fn test_mae_output_backprop(){
	_mae_output_backprop().unwrap();
}

fn _mae_output_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ordermap::OrderMap;
	use ops::loss::proportional::Proportional;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "input2", tag![])?;
	let node3 = g.new_node(shape![5], "output", tag![])?;

	let _o1 = g.new_op(Mae::new(&node1, &node2).mean_axes(&[0, -1]).output(&node3), tag![])?;
	let _o2 = g.new_op(Proportional::new(&node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.005;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}