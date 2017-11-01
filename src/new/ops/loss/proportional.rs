use new::graph::{GraphDef, NodeID, OpID, PassID, DataID, Storage, GraphShapes, Result};
use new::ops::{standard_op_name, Op, OpInstance, Pass};
use std::any::Any;

/// This `Op` applies a Proportional loss to every element of the input.
pub struct Proportional {
	input_id: NodeID,
	multiplier: f32,
	name: Option<String>,
}

impl Proportional {
	pub fn new(input_id: &NodeID) -> Self {
		Proportional {
			input_id: input_id.clone(),
			multiplier: 1.0,
			name: None,
		}
	}

	/// Loss is of the form `multiplier * sum(x)`
	pub fn multiplier(mut self, multiplier: f32) -> Self {
		self.multiplier = multiplier;
		self
	}
}

impl Op for Proportional {
	type InstanceType = ProportionalInstance;

	fn type_name(&self) -> &'static str {
		"Proportional"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		let name = standard_op_name(&self, &self.name, graph, &[self.input_id.clone()], &[]);

		Ok(ProportionalInstance{
			name: name,
			input_id: self.input_id.clone(),
			multiplier: self.multiplier,
			pass_id: graph.add_pass(ProportionalBackward::new(
				self.multiplier,
				self.input_id.clone())),
		})
	}
}


#[derive(Clone, Debug)] 
pub struct ProportionalInstance{
	name: String,
	multiplier: f32,
	input_id: NodeID,
	pass_id: PassID,
}

impl OpInstance for ProportionalInstance {

	fn instance_name(&self) -> &str {&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){(vec![self.input_id.clone()], vec![])}

	fn inner_passes(&self) -> Vec<PassID> {vec![self.pass_id.clone()]}

	fn inner_ops(&self) -> Vec<OpID> {vec![]}

	fn inner_nodes(&self) -> Vec<NodeID> {vec![]}

	fn propagate_shape_constraints(&self, _shapes: &mut GraphShapes) -> Result<()>{Ok(())}

}


#[derive(Clone, Debug)]
struct ProportionalBackward {
	multiplier: f32,
	input_id: NodeID,
}

impl ProportionalBackward {
	pub fn new(multiplier: f32, input_id: NodeID) -> Self {
		ProportionalBackward {
			multiplier,
			input_id,
		}
	}
}

impl Pass for ProportionalBackward {
	fn type_name(&self) -> &'static str {"ProportionalBackward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input_id.value_id()],
		vec![self.input_id.gradient_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input_val = data.get(&self.input_id.value_id())?;
		let mut input_grad = data.get_mut(&self.input_id.gradient_id())?;
		let input_val = input_val.as_slice().unwrap();
		let input_grad = input_grad.as_slice_mut().unwrap();

		let n = input_val.len();

		assert!(input_grad.len() == n);

		let multiplier = self.multiplier/n as f32;
		const SIMD: usize = 16;
		let mut error = 0.0;
		let mut errs = [0.;SIMD];
		
		for i in 0..n/SIMD {
			let input1_val = &input_val[i*SIMD..][..SIMD];
			let input1_grad = &mut input_grad[i*SIMD..][..SIMD];

			for j in 0..SIMD{
				errs[j] += input1_val[j]*multiplier;
				input1_grad[j] += multiplier;
			}
		}

		for j in (n/SIMD)*SIMD..n {
			error += input_val[j]*multiplier;
			input_grad[j] += multiplier;
		}


		for e in errs.iter() {
			error += *e;
		}

		data.loss_add(error);

		Ok(Box::new(()))
	}
}


#[test]
fn test_proportional_backprop(){
	_proportional_backprop().unwrap();
}

fn _proportional_backprop() -> Result<()>{
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;

	let _o1 = g.new_op(Proportional::new(&node1).multiplier(3.14), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-3;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}