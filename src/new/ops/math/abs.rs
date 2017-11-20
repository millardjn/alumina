use new::graph::{GraphDef, NodeID, OpID, Result};
use new::ops::Op;
use new::ops::activ::elementwise::{ActivationFunc, ElementwiseInstance, elementwise_build};

#[derive(Clone, Debug)] 
pub struct AbsFunc{}

impl ActivationFunc for AbsFunc {
	fn value(&self, input: f32) -> f32{
		input.abs()
	}

	fn gradient(&self, input: f32, output_grad: f32) -> f32{
		output_grad * input.signum()
	}

	fn backprop_requires_input_value() -> bool {true}
}

#[derive(Clone, Debug)] 
pub struct Abs {
	output: NodeID,
	input: NodeID,
	name: Option<String>,
}

impl Abs {
	pub fn new(input: &NodeID, output: &NodeID) -> Self {
		Abs {
			input: input.clone(),
			output: output.clone(),
			name: None,
		}
	}
}

impl Op for Abs {
	type InstanceType = ElementwiseInstance<AbsFunc>;

	fn type_name(&self) -> &'static str {
		"Abs"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		elementwise_build(graph, &self, &self.name, &self.input, &self.output, AbsFunc{})
	}
}


#[test]
fn test_abs_backprop(){
	_abs_backprop().unwrap();
}

fn _abs_backprop() -> Result<()>{
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use new::ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "output", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "target", tag![])?;


	let _o1 = g.new_op(Abs::new(&node1, &node2), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.002;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}