use graph::{GraphDef, NodeID, OpID, Result};
use ops::Op;
use ops::activ::elementwise::{ActivationFunc, ElementwiseInstance, elementwise_build};

#[derive(Clone, Debug)] 
pub struct SqrtFunc{}

impl ActivationFunc for SqrtFunc {
	fn value(&self, input: f32) -> f32{
		input.sqrt()
	}

	fn gradient(&self, input: f32, output_grad: f32) -> f32{
		output_grad * (0.5/input.sqrt())
	}

	fn backprop_requires_input_value() -> bool {true}
}

#[derive(Clone, Debug)] 
pub struct Sqrt {
	output: NodeID,
	input: NodeID,
	name: Option<String>,
}

impl Sqrt {
	pub fn new(input: &NodeID, output: &NodeID) -> Self {
		Sqrt {
			input: input.clone(),
			output: output.clone(),
			name: None,
		}
	}
}

impl Op for Sqrt {
	type InstanceType = ElementwiseInstance<SqrtFunc>;

	fn type_name(&self) -> &'static str {
		"Sqrt"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		elementwise_build(graph, &self, &self.name, &self.input, &self.output, SqrtFunc{})
	}
}


#[test]
fn test_sqrt_backprop(){
	_sqrt_backprop().unwrap();
}

fn _sqrt_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::mse::Mse;
	use ops::math::square::Square;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "square", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "output", tag![])?;
	let node4 = g.new_node(shape![7, 5, 16], "target", tag![])?;


	let _o1 = g.new_op(Square::new(&node1, &node2), tag![])?;
	let _o2 = g.new_op(Sqrt::new(&node2, &node3), tag![])?;
	let _o3 = g.new_op(Mse::new(&node3, &node4), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.002;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}