use new::graph::{GraphDef, NodeID, OpID, PassID, DataID, Storage, GraphShapes, ErrorKind, Result};
use new::ops::{standard_op_name, standard_inner_node_name, Op, OpInstance, Pass};
use new::ops::loss::proportional::Proportional;
// use generic_array::GenericArray;
// use typenum::{Unsigned, U16};
// use typenum_loops::Loop;
use std::any::Any;

/// An `Op` which implements the Mean Squared Error
///
/// The outer dimension of the inputs is taken to be the batch size and is excluded from the denominator of the Mean,
/// i.e a larger batch size corresponds to a larger error.
/// By default this `Op` has no output and will generate gradients.
///
/// If `output()` is set, the Mse will be written every element of that Node, and the gradient will be backprop'd from the output node.
///
/// If `separate_loss()` is set a scalar node will be added to the graph, and a `Loss` Op attached to it.
pub struct Mse {
	input1: NodeID,
	input2: NodeID,
	separate_loss: bool,
	output: Option<NodeID>,
	multiplier: f32,
	name: Option<String>,
}

impl Mse {
	pub fn new(input1: &NodeID, input2: &NodeID) -> Self {
		Mse {
			input1: input1.clone(),
			input2: input2.clone(),
			separate_loss: false,
			output: None,
			multiplier: 1.0,
			name: None,
		}
	}

	/// If true (and output is None) a scalar output node is created along with a `Loss` Op. This allows the loss from this Op to be queries separately while still
	/// Default: false
	pub fn separate_loss(mut self, separate_loss: bool) -> Self {
		self.separate_loss = separate_loss;
		self
	}

	/// If set this `Op` will output to a 
	/// Default: None.
	pub fn output(mut self, output: &NodeID) -> Self {
		self.output = Some(output.clone());
		self
	}

	/// Loss is of the form `multiplier * x.dot(x)`
	pub fn multiplier(mut self, multiplier: f32) -> Self {
		self.multiplier = multiplier;
		self
	}
}


impl Op for Mse {
	type InstanceType = MseInstance;

	fn type_name(&self) -> &'static str {
		"Mse"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		// TODO check broadcast at graph define time?
		let name = standard_op_name(&self, &self.name, graph, &[self.input1.clone(), self.input2.clone()], &[]);

		let loss_type = if let Some(output_id) = self.output {
			MseType::Output{
				output_id: output_id.clone(),
				forward_id: graph.add_pass(MseForward::new(
					self.multiplier,
					self.input1.clone(),
					self.input2.clone(),
					output_id.clone())),
				backward_id: graph.add_pass(MseBackward::new(
					self.multiplier,
					self.input1.clone(),
					self.input2.clone(),
					output_id.clone())),
			}
		} else if self.separate_loss {
			let output_name = standard_inner_node_name(&name, graph);
			let output_id = graph.new_node(shape![1], output_name, tag![])?;
			let loss_id = graph.new_op(Proportional::new(&output_id), tag![])?;

			MseType::Separate{
				output_id: output_id.clone(),
				loss_id: loss_id,
				forward_id: graph.add_pass(MseForward::new(
					self.multiplier,
					self.input1.clone(),
					self.input2.clone(),
					output_id.clone())),
				backward_id: graph.add_pass(MseBackward::new(
					self.multiplier,
					self.input1.clone(),
					self.input2.clone(),
					output_id.clone())),
			}
		} else {
			MseType::Joint{
				pass_id: graph.add_pass(MseJointPass::new(
					self.multiplier,
					self.input1.clone(),
					self.input2.clone()))
			}
		};

		Ok(MseInstance{
			name: name,
			multiplier: self.multiplier,
			input1_id: self.input1.clone(),
			input2_id: self.input2.clone(),
			loss_type: loss_type,
		})
	}
}

#[derive(Clone, Debug)] 
enum MseType {
	Joint { // No output node, losses are applied to the graph
		pass_id: PassID
	},
	Output {
		output_id: NodeID,
		forward_id: PassID,
		backward_id: PassID
	},
	Separate {
		output_id: NodeID,
		loss_id: OpID,
		forward_id: PassID,
		backward_id: PassID
	},
}

#[derive(Clone, Debug)] 
pub struct MseInstance {
	name: String,
	multiplier: f32,
	input1_id: NodeID,
	input2_id: NodeID,
	loss_type: MseType,
}

impl OpInstance for MseInstance {

	fn instance_name(&self) -> &str {&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		match &self.loss_type {
			&MseType::Joint{..} | &MseType::Separate{..} => (vec![self.input1_id.clone(), self.input2_id.clone()], vec![]),
			&MseType::Output{ref output_id, ..} => (vec![self.input1_id.clone(), self.input2_id.clone()], vec![output_id.clone()]),
		}
	}

	fn inner_passes(&self) -> Vec<PassID> {
		match &self.loss_type {
			&MseType::Joint{ref pass_id} => vec![pass_id.clone()],
			&MseType::Separate{ref forward_id, ref backward_id, ..} | &MseType::Output{ref forward_id, ref backward_id, ..} => vec![forward_id.clone(), backward_id.clone()],
		}
	}

	fn inner_ops(&self) -> Vec<OpID> {
		match &self.loss_type {
			&MseType::Joint{..} | &MseType::Output{..} => vec![],
			&MseType::Separate{ref loss_id, ..} => (vec![loss_id.clone()]),
		}
	}

	fn inner_nodes(&self) -> Vec<NodeID> {
		match &self.loss_type {
			&MseType::Joint{..} | &MseType::Output{..} => vec![],
			&MseType::Separate{ref output_id, ..} => (vec![output_id.clone()]),
		}
	}

	fn propagate_shape_constraints(&self, _shapes: &mut GraphShapes) -> Result<()>{Ok(())}

}


#[derive(Clone, Debug)]
struct MseJointPass {
	multiplier: f32,
	input1_id: NodeID,
	input2_id: NodeID,
}

impl MseJointPass {
	pub fn new(multiplier: f32, input1_id: NodeID, input2_id: NodeID) -> Self {
		MseJointPass {
			multiplier,
			input1_id,
			input2_id,
		}
	}
}

impl Pass for MseJointPass {
	fn type_name(&self) -> &'static str {"MseJointPass"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input1_id.value_id(), self.input2_id.value_id()],
		vec![self.input1_id.gradient_id(), self.input2_id.gradient_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input1_val = data.get(&self.input1_id.value_id())?;
		let input2_val = data.get(&self.input2_id.value_id())?;

		ensure!(input2_val.shape() == input1_val.shape(), ErrorKind::ForwardPassError("TODO".to_string()));

		let avg_denom: usize = input1_val.shape()[1..].iter().product();

		let input1_val = input1_val.as_slice().unwrap();
		let input2_val = input2_val.as_slice().unwrap();

		let n = input1_val.len();
		
		let multiplier = self.multiplier/avg_denom as f32;
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


#[derive(Clone, Debug)]
struct MseForward {
	multiplier: f32,
	input1_id: NodeID,
	input2_id: NodeID,
	output_id: NodeID,
}

impl MseForward {
	pub fn new(multiplier: f32, input1_id: NodeID, input2_id: NodeID, output_id: NodeID,) -> Self {
		MseForward {
			multiplier,
			input1_id,
			input2_id,
			output_id,
		}
	}
}

impl Pass for MseForward {
	fn type_name(&self) -> &'static str {"MseForward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input1_id.value_id(), self.input2_id.value_id()],
		vec![self.output_id.value_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input1_val = data.get(&self.input1_id.value_id())?;
		let input2_val = data.get(&self.input2_id.value_id())?;
		let mut output_val = data.get_mut(&self.output_id.value_id())?;

		ensure!(input2_val.shape() == input1_val.shape(), ErrorKind::ForwardPassError("TODO".to_string()));
		ensure!(output_val.len() == 1, ErrorKind::ForwardPassError("Mse Output must be scalar".to_string()));

		let avg_denom: usize = input1_val.shape()[1..].iter().product();

		let input1_val = input1_val.as_slice().unwrap();
		let input2_val = input2_val.as_slice().unwrap();
		let output_val = output_val.as_slice_mut().unwrap();

		let n = input1_val.len();

		let multiplier = self.multiplier/avg_denom as f32;
		const SIMD: usize = 16;
		let mut error = 0.0;
		let mut errs = [0.;SIMD];
		
		// type SIMD = U16;
		// let mut errs = <GenericArray<f32, SIMD>>::default();
		// let mut iv1 = <GenericArray<f32, SIMD>>::default();
		// let mut iv2 = <GenericArray<f32, SIMD>>::default();
		// let mut diff = <GenericArray<f32, SIMD>>::default();


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

			for j in 0..SIMD{
				let diff = input1_val[j]-input2_val[j];
				errs[j] += diff*diff;
			}
		}

		for j in (n/SIMD)*SIMD..n {
			let diff = input1_val[j]-input2_val[j];
			error += diff*diff;
		}
		

		for e in errs.iter() {
			error += *e;
		}

		output_val[0] = error*multiplier;

		Ok(Box::new(()))
	}
}

#[derive(Clone, Debug)]
struct MseBackward {
	multiplier: f32,
	input1_id: NodeID,
	input2_id: NodeID,
	output_id: NodeID,
}

impl MseBackward {
	pub fn new(multiplier: f32, input1_id: NodeID, input2_id: NodeID, output_id: NodeID) -> Self {
		MseBackward {
			multiplier,
			input1_id,
			input2_id,
			output_id,
		}
	}
}

impl Pass for MseBackward {
	fn type_name(&self) -> &'static str {"MseBackward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input1_id.value_id(), self.input2_id.value_id(), self.output_id.gradient_id()],
		vec![self.input1_id.gradient_id(), self.input2_id.gradient_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input1_val = data.get(&self.input1_id.value_id())?;
		let input2_val = data.get(&self.input2_id.value_id())?;
		let output_grad = data.get(&self.output_id.gradient_id())?;

		ensure!(input2_val.shape() == input1_val.shape(), ErrorKind::BackwardPassError("TODO".to_string()));
		ensure!(output_grad.len() == 1, ErrorKind::BackwardPassError("Mse Output must be scalar".to_string()));

		let avg_denom: usize = input1_val.shape()[1..].iter().product();

		let input1_val = input1_val.as_slice().unwrap();
		let input2_val = input2_val.as_slice().unwrap();
		let output_grad = output_grad.as_slice().unwrap()[0];


		let n = input1_val.len();
		let multiplier = output_grad * self.multiplier/avg_denom as f32;

		
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

			for j in 0..n {
				let diff = input1_val[j]-input2_val[j];
				input1_grad[j] +=  2.0*diff*multiplier;
				input2_grad[j] += -2.0*diff*multiplier;
			}
		} else if data.is_required(&self.input1_id.gradient_id()) {
			let mut input1_grad = data.get_mut(&self.input1_id.gradient_id())?;
			let input1_grad = input1_grad.as_slice_mut().unwrap();
			assert!(input1_grad.len() == n);

			for j in 0..n {
				let diff = input1_val[j]-input2_val[j];
				input1_grad[j] += 2.0*diff*multiplier;
			}
		} else if data.is_required(&self.input2_id.gradient_id()) {
			let mut input2_grad = data.get_mut(&self.input2_id.gradient_id())?;
			let input2_grad = input2_grad.as_slice_mut().unwrap();
			assert!(input2_grad.len() == n);

			for j in 0..n {
				let diff = input1_val[j]-input2_val[j];
				input2_grad[j] += -2.0*diff*multiplier;
			}
		}


		Ok(Box::new(()))
	}
}

#[test]
fn test_mse_backprop(){
	_mse_backprop().unwrap();
}

fn _mse_backprop() -> Result<()>{
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "input2", tag![])?;

	let _o1 = g.new_op(Mse::new(&node1, &node2), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}

#[test]
fn test_mse_separate_backprop(){
	_mse_separate_backprop().unwrap();
}

fn _mse_separate_backprop() -> Result<()>{
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "input2", tag![])?;

	let _o1 = g.new_op(Mse::new(&node1, &node2).separate_loss(true), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}