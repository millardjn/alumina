use new::graph::{GraphDef, NodeID, DataID, OpID, PassID, Storage, GraphShapes, ErrorKind, Result};
use new::ops::{standard_op_name, Op, OpInstance, Pass};
use std::any::Any;
use std::fmt::Debug;

use new::ops::activ::elementwise::{ActivationFunc, ElementwiseInstance, elementwise_build};

#[derive(Clone, Debug)] 
pub struct ReLUFunc{}

impl ActivationFunc for ReLUFunc {
	fn value(&self, input: f32) -> f32{
		(input.abs() + input)*0.5 // vectorises, but pretty questionable
	}

	fn gradient(&self, input: f32, output_grad: f32) -> f32{
		let sign = input.signum();
		output_grad * (sign+sign.abs())*0.5 //x.signum().max(0.0); <- this should be better but doesnt compile to maxps,
	}

	fn backprop_requires_input_value() -> bool {true}
}

#[derive(Clone, Debug)] 
pub struct ReLU {
	output: NodeID,
	input: NodeID,
	name: Option<String>,
}

impl ReLU {
	pub fn new(input: &NodeID, output: &NodeID) -> Self {
		ReLU {
			input: input.clone(),
			output: output.clone(),
			name: None,
		}
	}
}

impl Op for ReLU {
	type InstanceType = ElementwiseInstance<ReLUFunc>;

	fn type_name(&self) -> &'static str {
		"ReLU"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		elementwise_build(graph, &self, &self.name, &self.input, &self.output, ReLUFunc{})
	}
}