use new::graph::{GraphDef, NodeID, DataID, OpID, PassID, Storage, GraphShapes, ErrorKind, Result};
use new::ops::{standard_op_name, Op, OpInstance, Pass};
use new::shape::NodeDim;
use ndarray::{ArrayViewMutD, ArrayViewD};
use std::any::Any;
use smallvec::SmallVec;
use std::f32;

/// Softmax Activation Op
///
/// The Softmax operation squeezes all activations into the (0,1) range, and ensures that all activations in a group will sum to one.
/// By default all dimensions from the inner most outward until the first non-`Known` dimension will be grouped, and the Softmax operation repeated over the remaining dimensions.
/// 
/// This can be overridden using `inner()`, `outer()`, or most generally `mask()`.
#[derive(Clone, Debug)]
pub struct Softmax {
	input_id: NodeID,
	output_id: NodeID,
	axes: SmallVec<[isize; 6]>,
	name: Option<String>,
}

impl Softmax {
	pub fn new(input: &NodeID, output: &NodeID) -> Self {
		Softmax {
			input_id: input.clone(),
			output_id: output.clone(),
			//grouping: Grouping::Auto,
			axes: SmallVec::new(),
			name: None,
		}
	}

	/// The axes supplied will be grouped for the Softmax operation,
	/// with the operation repeated across the axes not supplied.
	///
	/// `axes` can be in the range [-input.ndims(), input.ndims());
	/// If no axes are supplied then all dimensions with Known size will be grouped.
	pub fn axes(mut self, axes: &[isize]) -> Self {
		self.axes = axes.iter().cloned().collect();
		self
	}
}

impl Op for Softmax {
	type InstanceType = SoftmaxInstance;

	fn type_name(&self) -> &'static str {
		"Softmax"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(mut self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		let name = standard_op_name(&self, &self.name, graph, &[self.input_id.clone()], &[self.output_id.clone()]);

		let mask = {
			let input_shape = graph.node_shape(&self.input_id)?;
			if self.axes.len() == 0 {
				for i in 0..input_shape.ndim() {
					if matches!(input_shape.dimensions()[i], NodeDim::Known(_)) {
						self.axes.push(i as isize)
					}
				}
			}
			group_mask(input_shape.ndims(), &self.axes)
		};

		Ok(SoftmaxInstance{
			name: name,
			input_id: self.input_id.clone(),
			output_id: self.output_id.clone(),
			mask: mask.clone(),
			forward_id: graph.add_pass(SoftmaxForward::new(
					self.input_id.clone(),
					self.output_id.clone(),
					mask.clone()
				)),
			backward_id: graph.add_pass(SoftmaxBackward::new(
					self.input_id.clone(),
					self.output_id.clone(),
					mask.clone()
				)),
		})
	}
}


#[derive(Clone, Debug)] 
pub struct SoftmaxInstance {
	name: String,
	input_id: NodeID,
	output_id: NodeID,
	mask: SmallVec<[bool; 6]>,
	forward_id: PassID,
	backward_id: PassID,
}

impl OpInstance for SoftmaxInstance {
	
	fn instance_name(&self) -> &str{&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){(vec![self.input_id.clone()], vec![self.output_id.clone()])}

	fn inner_passes(&self) -> Vec<PassID>{vec![self.forward_id.clone(), self.backward_id.clone()]}

	fn inner_ops(&self) -> Vec<OpID>{vec![]}

	fn inner_nodes(&self) -> Vec<NodeID>{vec![]}

	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{
		let input_shape = shapes.get_shape(&self.input_id).clone();
		shapes.merge_with(&self.output_id, &input_shape)
	}
}


/// Returns a mask indicating whether an axis should be reduced based on the axes list
/// If axes is empty this returns all true,
/// else only the axis provided are marked true.
fn group_mask(len: usize, axes: &[isize]) -> SmallVec<[bool; 6]> {
	let mut group = SmallVec::with_capacity(len);
	for _ in 0..len {
		group.push(false);
	}
	for axis in axes {
		group[(axis + len as isize) as usize % len] = true;
	}
	group
}


#[derive(Clone, Debug)]
struct SoftmaxForward {
	input_id: NodeID,
	output_id: NodeID,
	mask: SmallVec<[bool; 6]>,
}

impl SoftmaxForward {
	pub fn new(input_id: NodeID, output_id: NodeID, mask: SmallVec<[bool; 6]>) -> Self {
		SoftmaxForward {
			input_id,
			output_id,
			mask,
		}
	}
}

impl Pass for SoftmaxForward {
	fn type_name(&self) -> &'static str {"SoftmaxForward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(
			vec![self.input_id.value_id()],
			vec![self.output_id.value_id()]
		)
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input: ArrayViewD<f32> = data.get(&self.input_id.value_id())?;
		let mut output: ArrayViewMutD<f32> = data.get_mut(&self.output_id.value_id())?;

		let group_shape: Vec<usize> = input.shape().iter().enumerate().map(|(i, dim)| if self.mask[i] {*dim} else {1}).collect();

		ensure!(
			input.shape() == output.shape(),
			ErrorKind::PassError(self.instance_name(data.graph()), format!("input shape: {:?} did not match output shape: {:?}", input.shape(), output.shape()))
		);

		let iter = input.exact_chunks(group_shape.as_slice()).into_iter()
			.zip(output.exact_chunks_mut(group_shape.as_slice()));
		for (in_chunk, mut out_chunk) in iter {
			let max = in_chunk.iter().fold(f32::NEG_INFINITY, |max, &v| v.max(max));
			let sum = in_chunk.iter().fold(0., |sum, &v| sum + (v-max).exp());	
			
			out_chunk.zip_mut_with(&in_chunk, |o, i| *o += (*i-max).exp()/sum);
		}

		Ok(Box::new(()))
	}
}


#[derive(Clone, Debug)]
struct SoftmaxBackward {
	input_id: NodeID,
	output_id: NodeID,
	mask: SmallVec<[bool; 6]>,
}

impl SoftmaxBackward {
	pub fn new(input_id: NodeID, output_id: NodeID, mask: SmallVec<[bool; 6]>) -> Self {
		SoftmaxBackward {
			input_id,
			output_id,
			mask,
		}
	}
}

impl Pass for SoftmaxBackward {
	fn type_name(&self) -> &'static str {"SoftmaxBackward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(
			vec![self.input_id.value_id(), self.output_id.gradient_id()],
			vec![self.input_id.gradient_id()]
		)
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input: ArrayViewD<f32> = data.get(&self.input_id.value_id())?;
		let mut input_grad: ArrayViewMutD<f32> = data.get_mut(&self.input_id.gradient_id())?;
		let output_grad: ArrayViewD<f32> = data.get(&self.output_id.gradient_id())?;

		let group_shape: Vec<usize> = input.shape().iter().enumerate().map(|(i, dim)| if self.mask[i] {*dim} else {1}).collect();

		ensure!(
			input.shape() == output_grad.shape(),
			ErrorKind::PassError(self.instance_name(data.graph()), format!("input shape: {:?} did not match output shape: {:?}", input.shape(), output_grad.shape()))
		);


		let iter = input.exact_chunks(group_shape.as_slice()).into_iter()
			.zip(input_grad.exact_chunks_mut(group_shape.as_slice()))
			.zip(output_grad.exact_chunks(group_shape.as_slice()));
		for ((in_chunk, mut in_grad_chunk), out_grad_chunk) in iter {

			let max = in_chunk.iter().fold(f32::NEG_INFINITY, |max, &v| v.max(max));
			let sum = in_chunk.iter().fold(0., |sum, &v| sum + (v-max).exp());

			for (dim, og) in out_grad_chunk.indexed_iter() {
				if og.abs() > 0. {// hopefully output gradients are sparse, eg from cross entropy loss
					let a = in_chunk[&dim] - max;
					let denom = sum*sum;

					in_grad_chunk.zip_mut_with(&in_chunk, |ig, i|{
						let b = i - max;
						*ig += -(a + b).exp()*og/denom;
					});

					in_grad_chunk[&dim] += a.exp() * sum*og/denom;
				}
			}
		}

		Ok(Box::new(()))
	}
}



#[test]
fn test_softmax_backprop(){
	_softmax_backprop().unwrap();
}

fn _softmax_backprop() -> Result<()>{
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use new::ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![5, 16], "input", tag![])?;
	let node2 = g.new_node(shape![5, 16], "output", tag![])?;
	let node3 = g.new_node(shape![5, 16], "target", tag![])?;


	let _o1 = g.new_op(Softmax::new(&node1, &node2), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.002;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}

#[test]
fn test_softmax_mask_backprop(){
	_softmax_mask_backprop().unwrap();
}

fn _softmax_mask_backprop() -> Result<()>{
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use new::ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![3, 16, 5], "input", tag![])?;
	let node2 = g.new_node(shape![3, 16, 5], "output", tag![])?;
	let node3 = g.new_node(shape![3, 16, 5], "target", tag![])?;


	let _o1 = g.new_op(Softmax::new(&node1, &node2).axes(&[0, -1]), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.002;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}