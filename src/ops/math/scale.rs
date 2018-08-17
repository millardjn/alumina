use graph::{GraphDef, GraphShapes, ErrorKind, Result};
use id::{NodeID, DataID, OpID, PassID};
use storage::Storage;
use ops::{standard_op_name, Op, OpInstance, Pass};
use shape::{NodeShape, NodeDim};
use ndarray::{ArrayViewMutD, ArrayViewD};
use std::any::Any;

#[must_use]
#[derive(Clone, Debug)]
pub struct Scale {
	output: NodeID,
	input: NodeID,
	multiplier: f32,
	name: Option<String>,
}

impl Scale {
	pub fn new(input: &NodeID, output: &NodeID, multiplier: f32,) -> Self {
		Scale {
			input: input.clone(),
			output: output.clone(),
			multiplier: multiplier,
			name: None,
		}
	}
}

impl Op for Scale {
	type InstanceType = ScaleInstance;

	fn type_name(&self) -> &'static str {
		"Scale"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef) -> Result<Self::InstanceType> {
		let name = standard_op_name(&self, &self.name, graph, &[self.input.clone()], &[self.output.clone()]);

		Ok(ScaleInstance{
			name: name,
			input_id: self.input.clone(),
			output_id: self.output.clone(),
			multiplier: self.multiplier,
			forward_id: graph.add_pass(ScaleForward::new(
					self.input.clone(),
					self.output.clone(),
					self.multiplier)),
			backward_id: graph.add_pass(ScaleBackward::new(
					self.input.clone(),
					self.output.clone(),
					self.multiplier)),
		})
	}
}




#[derive(Clone, Debug)] 
pub struct ScaleInstance{
	name: String,
	input_id: NodeID,
	output_id: NodeID,
	multiplier: f32,
	forward_id: PassID,
	backward_id: PassID,
}

impl OpInstance for ScaleInstance {

	fn name(&self) -> &str{&self.name}

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
struct ScaleForward {
	input_id: NodeID,
	output_id: NodeID,
	multiplier: f32,
}

impl ScaleForward{
	pub fn new(input_id: NodeID, output_id: NodeID, multiplier: f32) -> Self {
		ScaleForward {
			input_id,
			output_id,
			multiplier,
		}
	}
}

impl Pass for ScaleForward {
	fn type_name(&self) -> &'static str {"ScaleBackward"}

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
			bail!(ErrorKind::PassError(self.name(), format!("Could not broadcast input shape: {:?} to output shape: {:?}", input.shape(), output.shape())));
		};

		output += &input_broadcast;

		Ok(Box::new(()))
	}
}

#[derive(Clone, Debug)]
struct ScaleBackward {
	input_id: NodeID,
	output_id: NodeID,
	multiplier: f32,
}

impl ScaleBackward {
	pub fn new(input_id: NodeID, output_id: NodeID, multiplier: f32) -> Self {
		ScaleBackward {
			input_id,
			output_id,
			multiplier,
		}
	}
}

impl Pass for ScaleBackward {
	fn type_name(&self) -> &'static str {"ScaleBackward"}

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
			ErrorKind::PassError(self.name(), format!("Could not broadcast input shape: {:?} to output shape: {:?}", input_grad.shape(), output_grad.shape()))
		);

		for chunk in output_grad.exact_chunks(input_grad.shape()){
			input_grad += &chunk;
		}

		Ok(Box::new(()))
	}
}

#[test]
fn test_scale_backprop(){
	_scale_backprop().unwrap();
}

fn _scale_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::mse::Mse;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![1, 1, 16], "broadcast", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "broadcasted", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "target", tag![])?;


	let _o1 = g.new_op(Scale::new(&node1, &node2, 3.14), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut indexmap![])?;

	Ok(())
}