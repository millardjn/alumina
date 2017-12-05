use graph::{GraphDef, NodeID, OpID, Result};
use ops::Op;
use ops::activ::elementwise::{ActivationFunc, ElementwiseInstance, elementwise_build};

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

#[must_use]
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


#[test]
fn test_relu_backprop(){
	_relu_backprop().unwrap();
}

fn _relu_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "output", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "target", tag![])?;


	let _o1 = g.new_op(ReLU::new(&node1, &node2), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.002;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}