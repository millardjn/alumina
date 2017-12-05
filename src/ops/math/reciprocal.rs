use graph::{GraphDef, NodeID, OpID, Result};
use ops::Op;
use ops::activ::elementwise::{ActivationFunc, ElementwiseInstance, elementwise_build};

#[derive(Clone, Debug)] 
pub struct ReciprocalFunc{}

impl ActivationFunc for ReciprocalFunc {
	fn value(&self, input: f32) -> f32{
		1.0/input
	}

	fn gradient(&self, input: f32, output_grad: f32) -> f32{
		output_grad/(-input*input)
	}

	fn backprop_requires_input_value() -> bool {true}
}

#[must_use]
#[derive(Clone, Debug)]
pub struct Reciprocal {
	output: NodeID,
	input: NodeID,
	name: Option<String>,
}

impl Reciprocal {
	pub fn new(input: &NodeID, output: &NodeID) -> Self {
		Reciprocal {
			input: input.clone(),
			output: output.clone(),
			name: None,
		}
	}
}

impl Op for Reciprocal {
	type InstanceType = ElementwiseInstance<ReciprocalFunc>;

	fn type_name(&self) -> &'static str {
		"Reciprocal"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		elementwise_build(graph, &self, &self.name, &self.input, &self.output, ReciprocalFunc{})
	}
}


#[test]
fn test_reciprocal_backprop(){
	_reciprocal_backprop().unwrap();
}

fn _reciprocal_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::mse::Mse;
	use ordermap::OrderMap;
	use rand::thread_rng;
	use rand::distributions::{Sample, Range};

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "output", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "target", tag![])?;


	let _o1 = g.new_op(Reciprocal::new(&node1, &node2), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.002;
	let step_size = 1E-2;
	let default_variance = 1.0;

	let sample: Box<::std::ops::FnMut() -> f64 + 'static> = Box::new(|| {
		let rng = &mut thread_rng();
		let mut range = Range::new(0.2, 10.0);
		range.sample(rng)
	});
	let mut override_dist = OrderMap::new();
	override_dist.insert(node1.clone(), sample);

	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut override_dist)?;

	Ok(())
}