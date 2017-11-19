use new::graph::{GraphDef, NodeID, OpID, Result};
use new::ops::Op;
use new::ops::activ::elementwise::{ActivationFunc, ElementwiseInstance, elementwise_build};

#[derive(Clone, Debug)] 
pub struct SquareFunc{}

impl ActivationFunc for SquareFunc {
	fn value(&self, input: f32) -> f32{
		input * input
	}

	fn gradient(&self, input: f32, output_grad: f32) -> f32{
		output_grad * 2.0 * input
	}

	fn backprop_requires_input_value() -> bool {true}
}

#[derive(Clone, Debug)] 
pub struct Square {
	output: NodeID,
	input: NodeID,
	name: Option<String>,
}

impl Square {
	pub fn new(input: &NodeID, output: &NodeID) -> Self {
		Square {
			input: input.clone(),
			output: output.clone(),
			name: None,
		}
	}
}

impl Op for Square {
	type InstanceType = ElementwiseInstance<SquareFunc>;

	fn type_name(&self) -> &'static str {
		"Square"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		elementwise_build(graph, &self, &self.name, &self.input, &self.output, SquareFunc{})
	}
}


#[test]
fn test_square_backprop(){
	_square_backprop().unwrap();
}

fn _square_backprop() -> Result<()>{
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use new::ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "output", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "target", tag![])?;


	let _o1 = g.new_op(Square::new(&node1, &node2), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.002;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}