use graph::{GraphDef, NodeID, DataID, OpID, PassID, Storage, GraphShapes, ErrorKind, Result};
use ops::{standard_op_name, Op, OpInstance, Pass};
use shape::{NodeShape, NodeDim};
use ndarray::{ArrayViewMutD, ArrayViewD, Dimension};
use smallvec::SmallVec;
use std::any::Any;

#[must_use]
#[derive(Clone, Debug)]
pub struct Add {
	output: NodeID,
	input: NodeID,
	name: Option<String>,
	extra_axes: SmallVec<[isize; 6]>,
}

impl Add {
	pub fn new(input: &NodeID, output: &NodeID) -> Self {
		Add {
			input: input.clone(),
			output: output.clone(),
			name: None,
			extra_axes: SmallVec::new(),
		}
	}

	/// Adds additional unit sized dimensions to the effective shape when broadcasting.
	///
	/// This is primarily to allow broadcasting to an output of higher rank.
	/// An argument of &[1, 2] and an input shape of [3, 4] would produce an effective shape of [3, 1, 1, 4] when broadcasting to the output.
	/// Each element of `axes` can be in the range [-input.ndims(), input.ndims()). After converting to positive axes, there must be no duplicates.
	///
	/// Default: empty
	pub fn extra_axes(mut self, extra_axes: &[isize]) -> Self {
		self.extra_axes = extra_axes.iter().cloned().collect();
		self
	}
}

impl Op for Add {
	type InstanceType = AddInstance;

	fn type_name(&self) -> &'static str {
		"Add"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		let name = standard_op_name(&self, &self.name, graph, &[self.input.clone()], &[self.output.clone()]);

		Ok(AddInstance{
			name: name,
			input_id: self.input.clone(),
			output_id: self.output.clone(),
			extra_axes: self.extra_axes.clone(),
			forward_id: graph.add_pass(AddForward::new(
					self.input.clone(),
					self.output.clone(),
					self.extra_axes.clone())),
			backward_id: graph.add_pass(AddBackward::new(
					self.input.clone(),
					self.output.clone(),
					self.extra_axes.clone())),
		})
	}
}

fn effective_shape(input_shape: &[usize], extra_axes: &[isize], output_len: usize) -> Result<SmallVec<[usize; 6]>> {
	let extra_axes: SmallVec<[usize; 6]> = extra_axes.iter().map(|dim| (dim + output_len as isize) as usize % output_len).collect();
	for i in 0..extra_axes.len() {
		if extra_axes[i] >= output_len {
			bail!("Extra_axes contained value {} which is out of bounds for output shape rank {}", extra_axes[i], output_len);
		}
		for j in i+1..extra_axes.len() {
			if extra_axes[i] == extra_axes[j] {
				bail!("Extra_axes contained duplicate values {} at {} and {}", extra_axes[i], i, j);
			}
		}
	}

	if input_shape.len() + extra_axes.len() != output_len {
		bail!("Input shape rank {} + extra_axes.len() {} did not equal output shape rank {}", input_shape.len(), extra_axes.len(), output_len);
	}

	let mut effective_shape: SmallVec<[usize; 6]> = (0..output_len).map(|_| 1).collect();

	let mut extra_count = 0;
	let mut input_count = 0;
	for i in 0..output_len {
		if extra_count < extra_axes.len() && i == extra_axes[extra_count] {
			extra_count += 1;
		} else {
			effective_shape[i] = input_shape[input_count];
			input_count +=1;
		}
	}
	debug_assert!(input_count == input_shape.len());

	Ok(effective_shape)
}


/// Add Op, the value of the input is added to 
#[derive(Clone, Debug)] 
pub struct AddInstance{
	name: String,
	input_id: NodeID,
	output_id: NodeID,
	extra_axes: SmallVec<[isize; 6]>,
	forward_id: PassID,
	backward_id: PassID,
}

impl OpInstance for AddInstance {

	fn instance_name(&self) -> &str{&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){(vec![self.input_id.clone()], vec![self.output_id.clone()])}

	fn inner_passes(&self) -> Vec<PassID>{vec![self.forward_id.clone(), self.backward_id.clone()]}

	fn inner_ops(&self) -> Vec<OpID>{vec![]}

	fn inner_nodes(&self) -> Vec<NodeID>{vec![]}

	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{

		let output_len = shapes.get_output_shape(&self.output_id).ndim();
		let input_shape = shapes.get_shape(&self.input_id).to_data_shape()?;

		let effective_shape = effective_shape(input_shape.slice(), &self.extra_axes, output_len)?;

		let output_shape: NodeShape = effective_shape.into_iter().map(|dim|{
			match dim {
				1 => NodeDim::Unknown,
				x => NodeDim::Known(x),
			}
		}).into();
		shapes.merge_with(&self.output_id, &output_shape)
	}
}

#[derive(Clone, Debug)]
struct AddForward {
	input_id: NodeID,
	output_id: NodeID,
	extra_axes: SmallVec<[isize; 6]>,
}

impl AddForward{
	pub fn new(input_id: NodeID, output_id: NodeID, extra_axes: SmallVec<[isize; 6]>) -> Self {
		AddForward {
			input_id,
			output_id,
			extra_axes,
		}
	}
}

impl Pass for AddForward {
	fn type_name(&self) -> &'static str {"AddBackward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(
			vec![self.input_id.value_id()],
			vec![self.output_id.value_id()]
		)
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input: ArrayViewD<f32> = data.get(&self.input_id.value_id())?;
		let mut output: ArrayViewMutD<f32> = data.get_mut(&self.output_id.value_id())?;

		let effective_shape = effective_shape(input.shape(), &self.extra_axes, output.ndim())?;
		let input_effective = input.into_shape(effective_shape.as_slice()).expect("must be a bug in effective_shape()");

		let input_broadcast = if let Some(view) = input_effective.broadcast(output.shape()) {
			view
		} else {
			bail!(ErrorKind::PassError(self.instance_name(data.graph()), format!("Could not broadcast input shape: {:?} to output shape: {:?}", effective_shape.as_slice(), output.shape())));
		};

		output += &input_broadcast;

		Ok(Box::new(()))
	}
}

#[derive(Clone, Debug)]
struct AddBackward {
	input_id: NodeID,
	output_id: NodeID,
	extra_axes: SmallVec<[isize; 6]>,
}

impl AddBackward {
	pub fn new(input_id: NodeID, output_id: NodeID, extra_axes: SmallVec<[isize; 6]>) -> Self {
		AddBackward {
			input_id,
			output_id,
			extra_axes,
		}
	}
}

impl Pass for AddBackward {
	fn type_name(&self) -> &'static str {"AddBackward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(
			vec![self.output_id.gradient_id()],
			vec![self.input_id.gradient_id()]
		)
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input_grad = data.get_mut(&self.input_id.gradient_id())?;
		let output_grad = data.get(&self.output_id.gradient_id())?;
		

		let effective_shape = effective_shape(input_grad.shape(), &self.extra_axes, output_grad.ndim())?;
		let mut input_grad_effective = input_grad.into_shape(effective_shape.as_slice()).expect("must be a bug in effective_shape()");

		ensure!(
			input_grad_effective.broadcast(output_grad.shape()).is_some(), 
			ErrorKind::PassError(self.instance_name(data.graph()), format!("Could not broadcast input shape: {:?} to output shape: {:?}", input_grad_effective.shape(), output_grad.shape()))
		);

		for chunk in output_grad.exact_chunks(input_grad_effective.shape()){
			input_grad_effective += &chunk;
		}

		Ok(Box::new(()))
	}
}

#[test]
fn test_add_backprop(){
	_add_backprop().unwrap();
}

fn _add_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![1, 1, 16], "broadcast", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "broadcasted", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "target", tag![])?;


	let _o1 = g.new_op(Add::new(&node1, &node2), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}

#[test]
fn test_add_broadcast_backprop(){
	_add_broadcast_backprop().unwrap();
}

fn _add_broadcast_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![5], "broadcast", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "broadcasted", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "target", tag![])?;


	let _o1 = g.new_op(Add::new(&node1, &node2).extra_axes(&[0, 2]), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}