use graph::{GraphDef, Result};
use id::NodeID;
use ops::Op;
use ops::activ::elementwise::{ActivationFunc, ElementwiseInstance, elementwise_build};

#[derive(Clone, Debug)] 
pub struct LogisticFunc{}

impl ActivationFunc for LogisticFunc {
	fn value(&self, input: f32) -> f32{
		let exp = input.exp();
		1.0/(1.0 + 1.0/exp)
	}

	fn gradient(&self, input: f32, output_grad: f32) -> f32{
		let exp = input.exp();
		output_grad * exp/((exp+1.0)*(exp+1.0))
	}

	fn backprop_requires_input_value() -> bool {true}
}

#[must_use]
#[derive(Clone, Debug)] 
pub struct Logistic {
	output: NodeID,
	input: NodeID,
	name: Option<String>,
}

impl Logistic {
	pub fn new(input: &NodeID, output: &NodeID) -> Self {
		Logistic {
			input: input.clone(),
			output: output.clone(),
			name: None,
		}
	}
}

impl Op for Logistic {
	type InstanceType = ElementwiseInstance<LogisticFunc>;

	fn type_name(&self) -> &'static str {
		"Logistic"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef) -> Result<Self::InstanceType> {
		elementwise_build(graph, &self, &self.name, &self.input, &self.output, LogisticFunc{})
	}
}


#[test]
fn test_logistic_backprop(){
	_logistic_backprop().unwrap();
}

fn _logistic_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "output", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "target", tag![])?;


	let _o1 = g.new_op(Logistic::new(&node1, &node2), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}