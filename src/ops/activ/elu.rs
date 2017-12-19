use graph::{GraphDef, Result};
use id::NodeID;
use ops::Op;
use ops::activ::elementwise::{ActivationFunc, ElementwiseInstance, elementwise_build};

#[derive(Clone, Debug)] 
pub struct ELUFunc{}

impl ActivationFunc for ELUFunc {
	fn value(&self, input: f32) -> f32{
		if input >= 0.0 {
			input
		} else {
			input.exp() - 1.0
		}
	}

	fn gradient(&self, input: f32, output_grad: f32) -> f32{
		if input >= 0.0 {
			output_grad
		} else {
			output_grad*input.exp()
		}
	}

	fn backprop_requires_input_value() -> bool {true}
}

#[must_use]
#[derive(Clone, Debug)] 
pub struct ELU {
	output: NodeID,
	input: NodeID,
	name: Option<String>,
}

impl ELU {
	pub fn new(input: &NodeID, output: &NodeID) -> Self {
		ELU {
			input: input.clone(),
			output: output.clone(),
			name: None,
		}
	}
}

impl Op for ELU {
	type InstanceType = ElementwiseInstance<ELUFunc>;

	fn type_name(&self) -> &'static str {
		"ELU"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef) -> Result<Self::InstanceType> {
		elementwise_build(graph, &self, &self.name, &self.input, &self.output, ELUFunc{})
	}
}


#[test]
fn test_elu_backprop(){
	_elu_backprop().unwrap();
}

fn _elu_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "output", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "target", tag![])?;


	let _o1 = g.new_op(ELU::new(&node1, &node2), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.002;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}