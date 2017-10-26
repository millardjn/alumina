use new::graph::{GraphDef, NodeID, OpID, PassID, DataID, Storage, GraphShapes, ErrorKind, Result};
use new::ops::{standard_op_name, Op, OpInstance, Pass};
// use generic_array::GenericArray;
// use typenum::{Unsigned, U16};
// use typenum_loops::Loop;
use std::any::Any;


pub struct L2 {
	input1: NodeID,
	input2: NodeID,
	multiplier: f32,
	name: Option<String>,
}

impl L2 {

	pub fn new(input1: &NodeID, input2: &NodeID) -> Self {
		L2 {
			input1: input1.clone(),
			input2: input2.clone(),
			multiplier: 1.0,
			name: None,
		}
	}

	/// Loss is of the form `multiplier * x.dot(x)`
	pub fn multiplier(mut self, multiplier: f32) -> Self {
		self.multiplier = multiplier;
		self
	}
}


impl Op for L2 {
	type InstanceType = L2Instance;

	fn type_name(&self) -> &'static str {
		"L2"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		let name = standard_op_name(&self, &self.name, graph, &[self.input1.clone(), self.input2.clone()], &[]);

		Ok(L2Instance{
			name: name,
			input1_id: self.input1.clone(),
			input2_id: self.input2.clone(),
			multiplier: self.multiplier,
			pass_id: graph.add_pass(L2Pass::new(
				self.multiplier, 
				self.input1.clone(), 
				self.input2.clone())),
		})
	}
}


/// Broadcast Op, the value of the input is added to 
#[derive(Clone, Debug)] 
pub struct L2Instance{
	name: String,
	multiplier: f32,
	input1_id: NodeID,
	input2_id: NodeID,
	pass_id: PassID,
}


impl OpInstance for L2Instance {

	fn instance_name(&self) -> &str {&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){(vec![self.input1_id.clone(), self.input2_id.clone()], vec![])}

	fn inner_passes(&self) -> Vec<PassID> {vec![self.pass_id.clone()]}

	fn inner_ops(&self) -> Vec<OpID> {vec![]}

	fn inner_nodes(&self) -> Vec<NodeID> {vec![]}

	fn propagate_shape_constraints(&self, _shapes: &mut GraphShapes) -> Result<()>{Ok(())}

}


#[derive(Clone, Debug)]
struct L2Pass{
	multiplier: f32,
	input1_id: NodeID,
	input2_id: NodeID,
}

impl L2Pass {
	pub fn new(multiplier: f32, input1_id: NodeID, input2_id: NodeID) -> Self {
		L2Pass {
			multiplier,
			input1_id,
			input2_id,
		}
	}
}

impl Pass for L2Pass {
	fn type_name(&self) -> &'static str {"L2Pass"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input1_id.value_id(), self.input2_id.value_id()],
		vec![self.input1_id.gradient_id(), self.input2_id.gradient_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input1_val = data.get(&self.input1_id.value_id())?;
		let input2_val = data.get(&self.input2_id.value_id())?;
		
		ensure!(
			input2_val.shape() == input1_val.shape(),
			ErrorKind::PassError(self.instance_name(data.graph()), format!("input1 shape: {:?} did not match input2 shape: {:?}", input2_val.shape(), input1_val.shape()))
			);

		let input2_val = input2_val.as_slice().unwrap();
		let input1_val = input1_val.as_slice().unwrap();

		let n = input1_val.len();
		assert!(input2_val.len() == n);

		let multiplier = self.multiplier as f32;
		const SIMD: usize = 16;
		let mut error = 0.0;
		let mut errs = [0.;SIMD];
		
		// type SIMD = U16;
		// let mut errs = <GenericArray<f32, SIMD>>::default();
		// let mut iv1 = <GenericArray<f32, SIMD>>::default();
		// let mut iv2 = <GenericArray<f32, SIMD>>::default();
		// let mut diff = <GenericArray<f32, SIMD>>::default();

		if data.is_required(&self.input1_id.gradient_id()) && data.is_required(&self.input2_id.gradient_id()) {
			let mut input1_grad = data.get_mut(&self.input1_id.gradient_id())?;
			let input1_grad = input1_grad.as_slice_mut().unwrap();
			let mut input2_grad = data.get_mut(&self.input2_id.gradient_id())?;
			let input2_grad = input2_grad.as_slice_mut().unwrap();
			assert!(input1_grad.len() == n);
			assert!(input2_grad.len() == n);

			// SIMD::partial_unroll(n, |i, j|{
			// 	unsafe{
			// 		iv1[j] = *odds::get_unchecked(input1_val, i);
			// 		iv2[j] = *odds::get_unchecked(input2_val, i);
			// 		diff[j] = iv1[j] - iv2[j];
			// 		errs[j] += diff[j]*diff[j]*multiplier;
			// 		*odds::get_unchecked_mut(input1_grad, i) +=  2.0*diff[j]*multiplier;
			// 		*odds::get_unchecked_mut(input2_grad, i) += -2.0*diff[j]*multiplier;
			// 	}
			// });

			for i in 0..n/SIMD {
				let input1_val = &input1_val[i*SIMD..][..SIMD];
				let input2_val = &input2_val[i*SIMD..][..SIMD];
				let input1_grad = &mut input1_grad[i*SIMD..][..SIMD];
				let input2_grad = &mut input2_grad[i*SIMD..][..SIMD];

				for j in 0..SIMD{
					let diff = input1_val[j]-input2_val[j];
					errs[j] += diff*diff*multiplier;
					input1_grad[j] +=  2.0*diff*multiplier;
					input2_grad[j] += -2.0*diff*multiplier;
				}
			}

			for j in (n/SIMD)*SIMD..n {
				let diff = input1_val[j]-input2_val[j];
				error += diff*diff*multiplier;
				input1_grad[j] +=  2.0*diff*multiplier;
				input2_grad[j] += -2.0*diff*multiplier;
			}
		} else if data.is_required(&self.input1_id.gradient_id()) {
			let mut input1_grad = data.get_mut(&self.input1_id.gradient_id())?;
			let input1_grad = input1_grad.as_slice_mut().unwrap();
			assert!(input1_grad.len() == n);

			// SIMD::partial_unroll(n, |i, j|{
			// 	unsafe{
			// 		let diff = *odds::get_unchecked(input1_val, i) - *odds::get_unchecked(input2_val, i);
			// 		errs[j] += diff*diff*multiplier;
			// 		*odds::get_unchecked_mut(input1_grad, i) +=  2.0*diff*multiplier;
			// 	}
			// });

			for i in 0..n/SIMD {
				let input1_val = &input1_val[i*SIMD..][..SIMD];
				let input2_val = &input2_val[i*SIMD..][..SIMD];
				let input1_grad = &mut input1_grad[i*SIMD..][..SIMD];

				for j in 0..SIMD{
					let diff = input1_val[j]-input2_val[j];
					errs[j] += diff*diff*multiplier;
					input1_grad[j] += 2.0*diff*multiplier;
				}
			}

			for j in (n/SIMD)*SIMD..n {
				let diff = input1_val[j]-input2_val[j];
				error += diff*diff*multiplier;
				input1_grad[j] += 2.0*diff*multiplier;
			}
		} else if data.is_required(&self.input2_id.gradient_id()) {
			let mut input2_grad = data.get_mut(&self.input2_id.gradient_id())?;
			let input2_grad = input2_grad.as_slice_mut().unwrap();
			assert!(input2_grad.len() == n);

			// SIMD::partial_unroll(n, |i, j|{
			// 	unsafe{
			// 		let diff = *odds::get_unchecked(input1_val, i) - *odds::get_unchecked(input2_val, i);
			// 		errs[j] += diff*diff*multiplier;
			// 		*odds::get_unchecked_mut(input2_grad, i) += -2.0*diff*multiplier;
			// 	}
			// });

			for i in 0..n/SIMD {
				let input1_val = &input1_val[i*SIMD..][..SIMD];
				let input2_val = &input2_val[i*SIMD..][..SIMD];
				let input2_grad = &mut input2_grad[i*SIMD..][..SIMD];

				for j in 0..SIMD{
					let diff = input1_val[j]-input2_val[j];
					errs[j] += diff*diff*multiplier;
					input2_grad[j] += -2.0*diff*multiplier;
				}
			}

			for j in (n/SIMD)*SIMD..n {
				let diff = input1_val[j]-input2_val[j];
				error += diff*diff*multiplier;
				input2_grad[j] += -2.0*diff*multiplier;
			}
		}

		for e in errs.iter() {
			error += *e;
		}

		data.loss_add(error);

		Ok(Box::new(()))
	}
}


#[test]
fn test_l2_backprop(){
	_l2_backprop().unwrap();
}

fn _l2_backprop() -> Result<()>{
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "input2", tag![])?;

	let _o1 = g.new_op(L2::new(&node1, &node2), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E0;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}