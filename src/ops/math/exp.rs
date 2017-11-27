use graph::{GraphDef, NodeID, OpID, Result};
use ops::Op;
use ops::activ::elementwise::{ActivationFunc, ElementwiseInstance, elementwise_build};

#[derive(Clone, Debug)] 
pub struct ExpFunc{}

impl ActivationFunc for ExpFunc {
	fn value(&self, input: f32) -> f32{
		input.exp()
	}

	fn gradient(&self, input: f32, output_grad: f32) -> f32{
		output_grad * input.exp()
	}

	fn backprop_requires_input_value() -> bool {true}
}

#[derive(Clone, Debug)] 
pub struct Exp {
	output: NodeID,
	input: NodeID,
	name: Option<String>,
}

impl Exp {
	pub fn new(input: &NodeID, output: &NodeID) -> Self {
		Exp {
			input: input.clone(),
			output: output.clone(),
			name: None,
		}
	}
}

impl Op for Exp {
	type InstanceType = ElementwiseInstance<ExpFunc>;

	fn type_name(&self) -> &'static str {
		"Exp"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		elementwise_build(graph, &self, &self.name, &self.input, &self.output, ExpFunc{})
	}
}


#[test]
fn test_exp_backprop(){
	_exp_backprop().unwrap();
}

fn _exp_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "output", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "target", tag![])?;


	let _o1 = g.new_op(Exp::new(&node1, &node2), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.002;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}