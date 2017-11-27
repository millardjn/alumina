use graph::{GraphDef, NodeID, OpID, PassID, DataID, Storage, GraphShapes, Result};
use ops::{standard_op_name, Op, OpInstance, Pass};
use shape::NodeShape;
use ndarray::{ArrayD, Dimension, IxDyn};
use std::any::Any;
use std::cmp::min;
use smallvec::SmallVec;


/// Average Pooling operation
///
/// Decrease size of dimensions by given factors.
/// Output values are the average of windows of the input with the size of factors
#[derive(Clone)] 
pub struct AvgPool {
 	name: Option<String>,
 	input_id: NodeID,
	output_id: NodeID,
	factors: Vec<usize>,
}
 	
impl AvgPool {
	pub fn new(input_id: &NodeID, output_id: &NodeID, factors: &[usize]) -> Self{
		AvgPool {
			name: None,
			input_id: input_id.clone(),
			output_id: output_id.clone(),
			factors: factors.to_vec(),
		}
	}
}

impl Op for AvgPool {
	type InstanceType = AvgPoolInstance;

	fn type_name(&self) -> &'static str {
		"AvgPool"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		let name =standard_op_name(&self, &self.name, graph, &[self.input_id.clone()], &[self.output_id.clone()]);

		Ok(AvgPoolInstance{
			name: name,
			input_id: self.input_id.clone(),
			output_id: self.output_id.clone(),
			factors: self.factors.clone(),
			forward_id:graph.add_pass(AvgPoolForward::new(
				self.input_id.clone(),
				self.output_id.clone(),
				self.factors.clone(),
			)),
			backward_id:graph.add_pass(AvgPoolBackward::new(
				self.input_id.clone(),
				self.output_id.clone(),
				self.factors.clone(),
			)),
		})
	}
}

#[derive(Debug, Clone)]
pub struct AvgPoolInstance {
	name: String,
	input_id: NodeID,
	output_id: NodeID,
	factors: Vec<usize>,
	forward_id: PassID,
	backward_id: PassID,
}

impl OpInstance for AvgPoolInstance {
	fn instance_name(&self) -> &str {&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		(
			vec![self.input_id.clone()],
			vec![self.output_id.clone()]
		)
	}

	fn inner_passes(&self) -> Vec<PassID> {
		vec![self.forward_id.clone(), self.backward_id.clone()]
	}

	fn inner_ops(&self) -> Vec<OpID> {vec![]}

	fn inner_nodes(&self) -> Vec<NodeID> {vec![]}

	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{

		let input_shape = shapes.get_shape(&self.input_id).to_data_shape()?;

		ensure!(input_shape.ndim() == self.factors.len(), "pooling factors must be the same length as input shape");

		let output_shape: NodeShape = input_shape.slice().iter().zip(&self.factors).map(|(i, f)| (i + f - 1)/f).into();

		shapes.merge_with(&self.output_id, &output_shape)?;
		Ok(())
	}

}


#[derive(Debug, Clone)]
pub struct AvgPoolForward {
	input_id: NodeID,
	output_id: NodeID,
	factors: Vec<usize>,
}

impl AvgPoolForward {
	pub fn new(input_id: NodeID, output_id: NodeID, factors: Vec<usize>) -> Self{
		AvgPoolForward {
			input_id,
			output_id,
			factors,
		}
	}
}

impl Pass for AvgPoolForward {
	fn type_name(&self) -> &'static str {"AvgPoolForward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input_id.value_id()],
		vec![self.output_id.value_id()])
	}

	fn run(&self, data: &Storage) -> Result<Box<Any>> {
		let input = data.get(&self.input_id.value_id())?;
		let mut output = data.get_mut(&self.output_id.value_id())?;

		let input_shape = input.shape();
		let output_shape = output.shape().to_vec();

		ensure!(input_shape.len() == output_shape.len(), "Input ndims does not match output ndims");
		ensure!(input_shape.len() == self.factors.len(), "pooling factors must be the same length as input shape");
		ensure!(input_shape.iter().zip(&self.factors).map(|(i, f)| (i + f - 1)/f).eq(output_shape.iter().cloned()), "input shape and factors incompatible with output shape");

		let input = input.as_slice().unwrap();
		let output = output.as_slice_mut().unwrap();

		let input_strides = strides(input_shape);
		let output_strides = strides(&output_shape);

		let scale = 1.0;

		let axis = 0;

		// starting from the innermost dims, skip dimensions that match then find the product of the 
		let outer_dims = input_shape.iter().zip(&output_shape).rev().skip_while(|&(i, o)| i == o).map(|(_i, o)| o).count();
		let n = output_shape[..outer_dims].iter().product();
		let ind_stride = output.len()/n;

		for i in 0..n {
			let output_ind = i * ind_stride;
			let ox = output_ind/output_strides[axis];
			pool_recurse_forward(input, output,
									input_shape, &output_shape, &self.factors,
									&input_strides, &output_strides,
									ox,
									axis, output_ind, scale)
		}

		Ok(Box::new(()))
	}
}


#[derive(Debug, Clone)]
pub struct AvgPoolBackward {
	input_id: NodeID,
	output_id: NodeID,
	factors: Vec<usize>,
}

impl AvgPoolBackward {
	pub fn new(input_id: NodeID, output_id: NodeID, factors: Vec<usize>) -> Self{
		AvgPoolBackward {
			input_id,
			output_id,
			factors,
		}
	}
}

impl Pass for AvgPoolBackward {
	fn type_name(&self) -> &'static str {"AvgPoolBackward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.output_id.gradient_id()],
		vec![self.input_id.gradient_id()])
	}

	fn run(&self, data: &Storage) -> Result<Box<Any>> {
		let mut input_grad = data.get_mut(&self.input_id.gradient_id())?;
		let output_grad = data.get(&self.output_id.gradient_id())?;

		let input_shape = input_grad.shape().to_vec();
		let output_shape = output_grad.shape();

		ensure!(input_shape.len() == output_shape.len(), "Input ndims does not match output ndims");
		ensure!(input_shape.len() == self.factors.len(), "pooling factors must be the same length as input dimensions");
		ensure!(input_shape.iter().zip(&self.factors).map(|(i, f)| (i + f - 1)/f).eq(output_shape.iter().cloned()), "input shape and factors incompatible with output shape");

		let input_grad = input_grad.as_slice_mut().unwrap();
		let output_grad = output_grad.as_slice().unwrap();

		let input_strides = strides(&input_shape);
		let output_strides = strides(output_shape);

		let scale = 1.0;
		let axis = 0;

		// starting from the innermost dims, skip dimensions that match then find the product of the 
		let outer_dims = input_shape.iter().zip(output_shape).rev().skip_while(|&(i, o)| i == o).map(|(_i, o)| o).count();
		let n = output_shape[..outer_dims].iter().product();
		let ind_stride = output_grad.len()/n;

		// TODO consider lifting outer unit factor dimensions into an outer loop like in linterp
		for i in 0..n {
			let output_ind = i * ind_stride;
			let ox = output_ind/output_strides[axis];
			pool_recurse_backward(input_grad, output_grad,
									&input_shape, output_shape, &self.factors,
									&input_strides, &output_strides,
									ox,
									axis, output_ind, scale)
		}

		Ok(Box::new(()))
	}
}

fn strides(shape: &[usize]) -> SmallVec<[usize;6]>{
	let mut strides = shape.iter().rev().scan(1, |state, &i| {
		let res = Some(*state);
		*state *= i;
		res
	}).collect::<SmallVec<[usize;6]>>();
	strides.reverse();
	strides
}

#[allow(dead_code)]
fn pool_recurse_forward(input: &[f32], output: &mut [f32],
						input_shape: &[usize], output_shape: &[usize], factors:&[usize],
						input_strides: &[usize], output_strides: &[usize],
						ox: usize,
						axis: usize, output_ind: usize, scale: f32){

	if input.len() != output.len() {
		let start = ox * factors[axis];
		let end = min(start + factors[axis], input_shape[axis]);

		let scale = scale/(end-start) as f32;

		let new_axis = axis+1;
		let new_output = &mut output[output_strides[axis]*ox..output_strides[axis]*(ox+1)];
		let new_output_ind = output_ind - ox*output_strides[axis];
		let new_ox = new_output_ind/output_strides[new_axis];
		for ix in start..end{
			let new_input = &input[input_strides[axis]*ix..input_strides[axis]*(ix+1)];
			pool_recurse_forward(new_input, new_output, input_shape, output_shape, factors, input_strides, output_strides, new_ox, new_axis, new_output_ind, scale);
		}
	} else {
		for i in 0..input.len(){
			output[i] += input[i] * scale;
		}
	}
}

fn pool_recurse_backward(input_grad: &mut [f32], output_grad: &[f32],
						input_shape: &[usize], output_shape: &[usize], factors:&[usize],
						input_strides: &[usize], output_strides: &[usize],
						ox: usize,
						axis: usize, output_ind: usize, scale: f32){

	if input_grad.len() != output_grad.len() {
		let start = ox * factors[axis];
		let end = min(start + factors[axis], input_shape[axis]);

		let scale = scale/(end-start) as f32;

		let new_axis = axis+1;
		let new_output_grad = &output_grad[output_strides[axis]*ox..output_strides[axis]*(ox+1)];
		let new_output_ind = output_ind - ox*output_strides[axis];
		let new_ox = new_output_ind/output_strides[new_axis];
		for ix in start..end{
			let new_input_grad = &mut input_grad[input_strides[axis]*ix..input_strides[axis]*(ix+1)];
			pool_recurse_backward(new_input_grad, new_output_grad, input_shape, output_shape, factors, input_strides, output_strides, new_ox, new_axis, new_output_ind, scale);
		}
	} else {
		for i in 0..input_grad.len(){
			input_grad[i] += output_grad[i] * scale;
		}
	}
}



#[test]
fn avg_pool_backprop(){
	_avg_pool_backprop().unwrap();
}

fn _avg_pool_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();
	
	let node1 = g.new_node(shape![3, 7, 9, 13], "input", tag![])?;
	let node2 = g.new_node(shape![Unknown, Unknown, Unknown, 13], "conv", tag![])?;
	let node3 = g.new_node(shape![3, 4, 9, 13], "target", tag![])?;
		
	let _o1 = g.new_op(AvgPool::new(&node1, &node2, &[1, 2, 1, 1]), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.01;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;
	
	Ok(())
}

#[test]
fn avg_pool(){
	_avg_pool();
}

fn _avg_pool() {
		let factors = vec![1, 2, 1, 3, 1];
		let input = ArrayD::from_elem(IxDyn(&[3, 7, 9, 8, 13]), 3.14);
		let mut output = ArrayD::from_elem(IxDyn(&[3, 4, 9, 3, 13]), -0.5);

		let input_shape = input.shape();
		let output_shape = output.shape().to_vec();

		let input = input.as_slice().unwrap();
		let output = output.as_slice_mut().unwrap();

		let input_strides = strides(input_shape);
		let output_strides = strides(&output_shape);

		let scale = 1.0;

		let axis = 0;

		// starting from the innermost dims, skip dimensions that match then find the product of the 
		let n = input_shape.iter().zip(&output_shape).rev().skip_while(|&(i, o)| i == o).map(|(_i, o)| o).product();
		let ind_stride = output.len()/n;

		for i in 0..n {
			let output_ind = i * ind_stride;
			let ox = output_ind/output_strides[axis];
			pool_recurse_forward(input, output,
									input_shape, &output_shape, &factors,
									&input_strides, &output_strides,
									ox,
									axis, output_ind, scale)
		}

		assert!(output.iter().all(|e| (e-2.64).abs() < 0.001), "{:?}", output);
}

#[test]
fn avg_pool_unit_factors(){
	_avg_pool_unit_factors();
}

fn _avg_pool_unit_factors() {
		let factors = vec![1, 1, 1, 1, 1];
		let input = ArrayD::from_elem(IxDyn(&[3, 7, 9, 8, 13]), 3.14);
		let mut output = ArrayD::from_elem(IxDyn(&[3, 7, 9, 8, 13]), -0.5);

		let input_shape = input.shape();
		let output_shape = output.shape().to_vec();

		let input = input.as_slice().unwrap();
		let output = output.as_slice_mut().unwrap();

		let input_strides = strides(input_shape);
		let output_strides = strides(&output_shape);

		let scale = 1.0;

		let axis = 0;

		// starting from the innermost dims, skip dimensions that match then find the product of the 
		let n = input_shape.iter().zip(&output_shape).rev().skip_while(|&(i, o)| i == o).map(|(_i, o)| o).product();
		let ind_stride = output.len()/n;

		for i in 0..n {
			let output_ind = i * ind_stride;
			let ox = output_ind/output_strides[axis];
			pool_recurse_forward(input, output,
									input_shape, &output_shape, &factors,
									&input_strides, &output_strides,
									ox,
									axis, output_ind, scale)
		}

		assert!(output.iter().all(|e| (e-2.64).abs() < 0.001), "{:?}", output);
}