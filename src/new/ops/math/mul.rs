use new::graph::{GraphDef, NodeID, DataID, OpID, PassID, Storage, GraphShapes, ErrorKind, Result};
use new::ops::{standard_op_name, Op, OpInstance, Pass};
use new::shape::{NodeShape, NodeDim};
use ndarray::{ArrayViewMutD, ArrayViewD};
use std::any::Any;

/// Mul Op
///
/// The value of input2 is broadcast to the shape of input1, elementwise multiplied, then added to the output
pub struct Mul {
	input1: NodeID,
	input2: NodeID,
	output: NodeID,
	name: Option<String>,
}

impl Mul {
	pub fn new(input1: &NodeID, input2: &NodeID, output: &NodeID) -> Self {
		Mul {
			input1: input1.clone(),
			input2: input2.clone(),
			output: output.clone(),
			name: None,
		}
	}
}

impl Op for Mul {
	type InstanceType = MulInstance;

	fn type_name(&self) -> &'static str {
		"Mul"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		let name = standard_op_name(&self, &self.name, graph, &[self.input1.clone(), self.input2.clone()], &[self.output.clone()]);

		Ok(MulInstance{
			name: name,
			input1_id: self.input1.clone(),
			input2_id: self.input2.clone(),
			output_id: self.output.clone(),
			forward_id: graph.add_pass(MulForward::new(
				self.input1.clone(),
				self.input2.clone(),
				self.output.clone())),
			backward_id: graph.add_pass(MulBackward::new(
				self.input1.clone(),
				self.input2.clone(),
				self.output.clone())),
		})
	}
}


/// Mul OpInstance
///
/// the value of input2 is broadcast to the shape of input1, elementwise multiplied, then added to the output
#[derive(Clone, Debug)] 
pub struct MulInstance{
	name: String,
	input1_id: NodeID,
	input2_id: NodeID,
	output_id: NodeID,
	forward_id: PassID,
	backward_id: PassID,
}

impl OpInstance for MulInstance {

	fn instance_name(&self) -> &str{&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){(vec![self.input1_id.clone(),self.input2_id.clone()], vec![self.output_id.clone()])}

	fn inner_passes(&self) -> Vec<PassID>{vec![self.forward_id.clone(), self.backward_id.clone()]}

	fn inner_ops(&self) -> Vec<OpID>{vec![]}

	fn inner_nodes(&self) -> Vec<NodeID>{vec![]}

	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{
		let output_shape: NodeShape = shapes.get_shape(&self.input2_id).dimensions().iter().map(|dim|{
			match dim {
				&NodeDim::Known(1) => NodeDim::Unknown,
				&NodeDim::Known(x) => NodeDim::Known(x),
				_ => unreachable!(),
			}
		}).into();
		output_shape.merge(shapes.get_shape(&self.input1_id))?;
		shapes.merge_with(&self.output_id, &output_shape)
	}

}


#[derive(Clone, Debug)]
struct MulForward {
	input1_id: NodeID,
	input2_id: NodeID,
	output_id: NodeID,
}

impl MulForward {
	pub fn new(input1_id: NodeID, input2_id: NodeID, output_id: NodeID) -> Self {
		MulForward {
			input1_id,
			input2_id,
			output_id,
		}
	}
}

impl Pass for MulForward {
	fn type_name(&self) -> &'static str {"MulForward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(
			vec![self.input1_id.value_id(), self.input2_id.value_id()],
			vec![self.output_id.value_id()]
		)
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input1: ArrayViewD<f32> = data.get(&self.input1_id.value_id())?;
		let input2: ArrayViewD<f32> = data.get(&self.input2_id.value_id())?;
		let mut output: ArrayViewMutD<f32> = data.get_mut(&self.output_id.value_id())?;

		ensure!(
			input1.shape() == output.shape(),
			ErrorKind::ForwardPassError(format!("'{}' input1 shape did not match output shape", self.instance_name(data.graph())))
		);
		ensure!(
			input2.broadcast(input1.shape()).is_some(),
			ErrorKind::ForwardPassError(format!("'{}' could not broadcast input2 to input1 shape", self.instance_name(data.graph())))
		);
		ensure!(
			input2.broadcast(output.shape()).is_some(), 
			ErrorKind::ForwardPassError(format!("'{}' could not broadcast input2 to output shape", self.instance_name(data.graph())))
		);

		let iter = input1.exact_chunks(input2.shape()).into_iter()
			.zip(output.exact_chunks_mut(input2.shape()));
		for (in1_chunk, mut out_chunk) in iter {
			out_chunk += &(&in1_chunk * &input2);
		}

		Ok(Box::new(()))
	}
}


#[derive(Clone, Debug)]
struct MulBackward {
	input1_id: NodeID,
	input2_id: NodeID,
	output_id: NodeID,
}

impl MulBackward {
	pub fn new(input1_id: NodeID, input2_id: NodeID, output_id: NodeID) -> Self {
		MulBackward {
			input1_id,
			input2_id,
			output_id,
		}
	}
}

impl Pass for MulBackward {
	fn type_name(&self) -> &'static str {"MulBackward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(
			vec![self.input1_id.value_id(), self.input2_id.value_id(), self.output_id.gradient_id()],
			vec![self.input1_id.gradient_id(),self.input2_id.gradient_id()]
		)
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input1: ArrayViewD<f32> = data.get(&self.input1_id.value_id())?;
		let input2: ArrayViewD<f32> = data.get(&self.input2_id.value_id())?;
		let output_grad = data.get(&self.output_id.gradient_id())?;
		
		ensure!(
			input1.shape() == output_grad.shape(),
			ErrorKind::BackwardPassError(format!("'{}' input1 shape did not match output shape", self.instance_name(data.graph())))
		);
		ensure!(
			input2.broadcast(input1.shape()).is_some(),
			ErrorKind::BackwardPassError(format!("'{}' could not broadcast input2 to input1 shape", self.instance_name(data.graph())))
		);
		ensure!(
			input2.broadcast(output_grad.shape()).is_some(), 
			ErrorKind::BackwardPassError(format!("'{}' could not broadcast input2 to output shape", self.instance_name(data.graph())))
		);


		if data.is_required(&self.input1_id.gradient_id()) && data.is_required(&self.input2_id.gradient_id()) {
			let mut input1_grad = data.get_mut(&self.input1_id.gradient_id())?;
			let mut input2_grad = data.get_mut(&self.input2_id.gradient_id())?;

			let iter = input1.exact_chunks(input2.shape()).into_iter()
				.zip(input1_grad.exact_chunks_mut(input2.shape()))
				.zip(output_grad.exact_chunks(input2.shape()));

			for ((input1_chunk, mut input1_grad_chunk) , out_grad_chunk) in iter {
				input1_grad_chunk += &(&input2 * &out_grad_chunk);
				input2_grad += &(&input1_chunk * &out_grad_chunk);
			}

		} else if data.is_required(&self.input1_id.gradient_id()) {
			let mut input1_grad = data.get_mut(&self.input1_id.gradient_id())?;

			let iter = input1_grad.exact_chunks_mut(input2.shape()).into_iter()
				.zip(output_grad.exact_chunks(input2.shape()));

			for (mut input1_grad_chunk, out_grad_chunk) in iter {
				input1_grad_chunk += &(&input2 * &out_grad_chunk);
			}
		} else if data.is_required(&self.input2_id.gradient_id()) {
			let mut input2_grad = data.get_mut(&self.input2_id.gradient_id())?;

			let iter = input1.exact_chunks(input2.shape()).into_iter()
				.zip(output_grad.exact_chunks(input2.shape()));

			for (input1_chunk, out_grad_chunk) in iter {
				input2_grad += &(&input1_chunk * &out_grad_chunk);
			}
		}

		Ok(Box::new(()))
	}
}


#[test]
fn test_mul_backprop(){
	_mul_backprop().unwrap();
}

fn _mul_backprop() -> Result<()>{
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use new::ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;
	let node2 = g.new_node(shape![1, 1, 16], "input2", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "output", tag![])?;
	let node4 = g.new_node(shape![7, 5, 16], "target", tag![])?;

	let _o1 = g.new_op(Mul::new(&node1, &node2, &node3), tag![])?;
	let _o2 = g.new_op(Mse::new(&node3, &node4), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}