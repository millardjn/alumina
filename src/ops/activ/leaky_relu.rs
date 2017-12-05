use graph::{GraphDef, NodeID, OpID, Result};
use ops::Op;
use ops::activ::elementwise::{ActivationFunc, ElementwiseInstance, elementwise_build};

#[derive(Clone, Debug)] 
pub struct LeakyReLUFunc{
	alpha: f32,
}

impl ActivationFunc for LeakyReLUFunc {
	fn value(&self, input: f32) -> f32{
		 (input + input.abs())*0.5 + (input - input.abs())*(0.5*self.alpha)
	}

	fn gradient(&self, input: f32, output_grad: f32) -> f32{
		let sign = input.signum();
		//output_grad*0.5*((sign + 1.0) - (sign - 1.0)*self.alpha) // 3 mults and 3 adds
		output_grad* (sign*(0.5 - 0.5*self.alpha) + (0.5 + 0.5*self.alpha)) // after optimisation this should have 2 mults and 1 add
	}

	fn backprop_requires_input_value() -> bool {true}
}

#[must_use]
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


#[test]
fn test_leaky_relu_backprop(){
	_leaky_relu_backprop().unwrap();
}

fn _leaky_relu_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "output", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "target", tag![])?;


	let _o1 = g.new_op(LeakyReLU::new(&node1, &node2), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.002;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}