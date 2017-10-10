use new::graph::{GraphDef, NodeID, OpID, Result};
use new::ops::Op;
use new::ops::activ::elementwise::{ActivationFunc, ElementwiseInstance, elementwise_build};

#[derive(Clone, Debug)] 
pub struct LeakyReLUFunc{
	alpha: f32,
}

impl ActivationFunc for LeakyReLUFunc {
	fn value(&self, input: f32) -> f32{
		 (input + input.abs())*0.5 + (input - input.abs())*0.5*self.alpha
	}

	fn gradient(&self, input: f32, output_grad: f32) -> f32{
		let sign = input.signum();
		output_grad*0.5*((sign + 1.0) + (sign - 1.0)*self.alpha)
	}

	fn backprop_requires_input_value() -> bool {true}
}

#[derive(Clone, Debug)] 
pub struct LeakyReLU {
	output: NodeID,
	input: NodeID,
	name: Option<String>,
	alpha: f32,
}

impl LeakyReLU {
	pub fn new(input: &NodeID, output: &NodeID) -> Self {
		LeakyReLU {
			input: input.clone(),
			output: output.clone(),
			name: None,
			alpha: 0.3,
		}
	}

	pub fn alpha(mut self, alpha: f32) -> Self{
		self.alpha = alpha;
		self
	}
}

impl Op for LeakyReLU {
	type InstanceType = ElementwiseInstance<LeakyReLUFunc>;

	fn type_name(&self) -> &'static str {
		"LeakyReLU"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		elementwise_build(graph, &self, &self.name, &self.input, &self.output, LeakyReLUFunc{alpha: self.alpha})
	}
}