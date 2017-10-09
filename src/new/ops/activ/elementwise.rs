use new::graph::{NodeID, DataID, Storage, GraphShapes, ErrorKind, Result};
use new::ops:: Pass;
use std::any::Any;
use std::marker::PhantomData;
use std::fmt::Debug;

/// Used to define graph op with no parameters, where the effect of the input on the output is entirely seperable.
pub trait ActivationFunc: Clone + Debug + 'static {
	#[inline(always)]
	/// For a given input x, what is the output y, and the derivative dy/dx
	fn value(input: f32) -> f32;

	fn gradient(input: f32, output_grad: f32) -> f32;

	fn backprop_requires_input_value() -> bool;
}


#[derive(Clone, Debug)]
struct ElementwiseForward<F: ActivationFunc> {
	input_id: NodeID,
	output_id: NodeID,
	func: PhantomData<F>,
}

impl<F: ActivationFunc> ElementwiseForward<F> {
	pub fn new(input_id: NodeID, output_id: NodeID) -> Self {
		ElementwiseForward {
			input_id,
			output_id,
			func: PhantomData,
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
			ErrorKind::BackwardPassError(format!("'{}' input shape did not match output shape", self.instance_name(data.graph())))
		);

		let input = input.as_slice().unwrap();
		let output = output.as_slice_mut().unwrap();

		let len = input.len();
		
		let inp = &input[..len];
		let out = &mut output[..len];

		for i in 0..len{
			out[i] += F::value(inp[i]);
		}

		Ok(Box::new(()))
	}
}


#[derive(Clone, Debug)]
struct ElementwiseBackward<F: ActivationFunc> {
	input_id: NodeID,
	output_id: NodeID,
	func: PhantomData<F>,
}

impl<F: ActivationFunc> ElementwiseBackward<F> {
	pub fn new(input_id: NodeID, output_id: NodeID) -> Self {
		ElementwiseBackward {
			input_id,
			output_id,
			func: PhantomData,
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
			ErrorKind::BackwardPassError(format!("'{}' input shape did not match output shape", self.instance_name(data.graph())))
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
				inpd[i] += F::gradient(inp[i], outd[i]);
			}
		} else {

			let outd = &output_grad[..len];
			let inpd = &mut input_grad[..len];

			for i in 0..len{
				inpd[i] += F::gradient(0.0, outd[i]);
			}
		}

		Ok(Box::new(()))
	}
}