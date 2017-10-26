use new::graph::{GraphDef, NodeID, DataID, OpID, PassID, Storage, GraphShapes, ErrorKind, Result};
use new::ops::{standard_op_name, Op, OpInstance, Pass};
use std::any::Any;
use std::fmt::Debug;



pub fn elementwise_build<O: Op, F: ActivationFunc>(graph: &mut GraphDef, op: &O, name: &Option<String>, input: &NodeID, output: &NodeID, func: F) -> Result<ElementwiseInstance<F>> {
	let name = standard_op_name(op, name, graph, &[input.clone()], &[output.clone()]);

	Ok(ElementwiseInstance{
		name: name,
		input_id: input.clone(),
		output_id: output.clone(),
		func: func.clone(),
		forward_id: graph.add_pass(ElementwiseForward::new(
				input.clone(),
				output.clone(),
				func.clone())),
		backward_id: graph.add_pass(ElementwiseBackward::new(
				input.clone(),
				output.clone(),
				func.clone())),
	})
}


/// Used to define graph op with no parameters, where the effect of the input on the output is entirely seperable.
pub trait ActivationFunc: Clone + Debug + 'static {
	#[inline(always)]
	/// For a given input x, what is the output y, and the derivative dy/dx
	fn value(&self, input: f32) -> f32;

	fn gradient(&self, input: f32, output_grad: f32) -> f32;

	fn backprop_requires_input_value() -> bool;
}

#[derive(Clone, Debug)]
pub struct ElementwiseInstance<F: ActivationFunc> {
	name: String,
	input_id: NodeID,
	output_id: NodeID,
	func: F,
	forward_id: PassID,
	backward_id: PassID,
}

impl<F: ActivationFunc> OpInstance for ElementwiseInstance<F> {

	fn instance_name(&self) -> &str{&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){(vec![self.input_id.clone()], vec![self.output_id.clone()])}

	fn inner_passes(&self) -> Vec<PassID>{vec![self.forward_id.clone(), self.backward_id.clone()]}

	fn inner_ops(&self) -> Vec<OpID>{vec![]}

	fn inner_nodes(&self) -> Vec<NodeID>{vec![]}

	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{
		let input_shape = shapes.get_shape(&self.input_id).clone();
		shapes.merge_with(&self.output_id, &input_shape)
	}
}


#[derive(Clone, Debug)]
pub struct ElementwiseForward<F: ActivationFunc> {
	input_id: NodeID,
	output_id: NodeID,
	func: F,
}

impl<F: ActivationFunc> ElementwiseForward<F> {
	pub fn new(input_id: NodeID, output_id: NodeID, func: F) -> Self {
		ElementwiseForward {
			input_id,
			output_id,
			func,
		}
	}
}

impl<F: ActivationFunc> Pass for ElementwiseForward<F> {
	fn type_name(&self) -> &'static str {"ElementwiseForward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(
			vec![self.input_id.value_id()],
			vec![self.output_id.value_id()]
		)
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input = data.get(&self.input_id.value_id())?;
		let mut output = data.get_mut(&self.output_id.value_id())?;

		ensure!(
			input.shape() == output.shape(),
			ErrorKind::PassError(self.instance_name(data.graph()), format!("input shape: {:?} did not match output shape: {:?}", input.shape(), output.shape()))
		);

		let input = input.as_slice().unwrap();
		let output = output.as_slice_mut().unwrap();

		let len = input.len();
		
		let inp = &input[..len];
		let out = &mut output[..len];

		for i in 0..len{
			out[i] += self.func.value(inp[i]);
		}

		Ok(Box::new(()))
	}
}


#[derive(Clone, Debug)]
pub struct ElementwiseBackward<F: ActivationFunc> {
	input_id: NodeID,
	output_id: NodeID,
	func: F,
}

impl<F: ActivationFunc> ElementwiseBackward<F> {
	pub fn new(input_id: NodeID, output_id: NodeID, func: F) -> Self {
		ElementwiseBackward {
			input_id,
			output_id,
			func,
		}
	}
}

impl<F: ActivationFunc> Pass for ElementwiseBackward<F> {
	fn type_name(&self) -> &'static str {"ElementwiseBackward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		if F::backprop_requires_input_value() {
			(
				vec![self.input_id.value_id(), self.output_id.gradient_id()],
				vec![self.input_id.gradient_id()]
			)
		} else {
			(
				vec![self.output_id.gradient_id()],
				vec![self.input_id.gradient_id()]
			)
		}
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		
		let output_grad = data.get(&self.output_id.gradient_id())?;
		let mut input_grad = data.get_mut(&self.input_id.gradient_id())?;
		
		ensure!(
			input_grad.shape() == output_grad.shape(),
			ErrorKind::PassError(self.instance_name(data.graph()), format!("input shape: {:?} did not match output shape: {:?}", input_grad.shape(), output_grad.shape()))
		);

		let output_grad = output_grad.as_slice().unwrap();
		let input_grad = input_grad.as_slice_mut().unwrap();

		let len = input_grad.len();

		if F::backprop_requires_input_value() {
			let input = data.get(&self.input_id.value_id())?;
			let input = input.as_slice().unwrap();

			let inp = &input[..len];
			let outd = &output_grad[..len];
			let inpd = &mut input_grad[..len];

			for i in 0..len{
				inpd[i] += self.func.gradient(inp[i], outd[i]);
			}
		} else {

			let outd = &output_grad[..len];
			let inpd = &mut input_grad[..len];

			for i in 0..len{
				inpd[i] += self.func.gradient(0.0, outd[i]);
			}
		}

		Ok(Box::new(()))
	}
}