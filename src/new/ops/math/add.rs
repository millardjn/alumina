use new::graph::{GraphDef, NodeID, DataID, OpID, PassID, Storage, GraphShapes, ErrorKind, Result};
use new::ops::{standard_op_name, Op, OpInstance, Pass};
use new::shape::{NodeShape, NodeDim};
use ndarray::{ArrayViewMutD, ArrayViewD};
use std::any::Any;

#[derive(Clone, Debug)] 
pub struct Add {
	output: NodeID,
	input: NodeID,
	name: Option<String>,
}

impl Add {
	pub fn new(input: &NodeID, output: &NodeID) -> Self {
		Add {
			input: input.clone(),
			output: output.clone(),
			name: None,
		}
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
			forward_id: graph.add_pass(AddForward::new(
					self.input.clone(),
					self.output.clone())),
			backward_id: graph.add_pass(AddBackward::new(
					self.input.clone(),
					self.output.clone())),
		})
	}
}




/// Add Op, the value of the input is added to 
#[derive(Clone, Debug)] 
pub struct AddInstance{
	name: String,
	input_id: NodeID,
	output_id: NodeID,
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
		let output_shape: NodeShape = shapes.get_shape(&self.input_id).dimensions().iter().map(|dim|{
			match dim {
				&NodeDim::Known(1) => NodeDim::Unknown,
				&NodeDim::Known(x) => NodeDim::Known(x),
				_ => unreachable!(),
			}
		}).into();
		shapes.merge_with(&self.output_id, &output_shape)
	}
}

#[derive(Clone, Debug)]
struct AddForward {
	input_id: NodeID,
	output_id: NodeID,
}

impl AddForward{
	pub fn new(input_id: NodeID, output_id: NodeID) -> Self {
		AddForward {
			input_id,
			output_id,
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

		let input_broadcast = if let Some(view) = input.broadcast(output.shape()) {
			view
		} else {
			bail!(ErrorKind::ForwardPassError(format!("'{}' could not broadcast input to output shape", self.instance_name(data.graph()))));
		};

		output += &input_broadcast;

		Ok(Box::new(()))
	}
}

#[derive(Clone, Debug)]
struct AddBackward {
	input_id: NodeID,
	output_id: NodeID,
}

impl AddBackward {
	pub fn new(input_id: NodeID, output_id: NodeID) -> Self {
		AddBackward {
			input_id,
			output_id,
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
		let mut input_grad = data.get_mut(&self.input_id.gradient_id())?;
		let output_grad = data.get(&self.output_id.gradient_id())?;
		
		ensure!(
			input_grad.broadcast(output_grad.shape()).is_some(), 
			ErrorKind::BackwardPassError(format!("'{}' could not broadcast input to output shape", self.instance_name(data.graph())))
		);

		// TODO check that the match shapes before rather than panic
		for chunk in output_grad.exact_chunks(input_grad.shape()){
			input_grad += &chunk;
		}

		Ok(Box::new(()))
	}
}

#[test]
fn test_add_backprop(){
	_add_backprop().unwrap();
}

fn _add_backprop() -> Result<()>{
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use new::ops::loss::mse::Mse;
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