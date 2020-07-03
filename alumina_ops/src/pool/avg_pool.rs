use alumina_core::{
	base_ops::{OpBuilder, OpInstance},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{Node, NodeInner},
	shape::{NodeAxis, NodeShape},
	shape_prop::ShapePropContext,
};
use indexmap::{indexset, IndexSet};

use ndarray::Dimension;

use smallvec::SmallVec;
use std::cmp::min;

pub fn avg_pool<I>(input: I, factors: &[usize]) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();

	let output_shape: NodeShape = input
		.shape()
		.slice()
		.iter()
		.zip(factors)
		.map(|(i, f)| match i {
			NodeAxis::Known { val } => NodeAxis::known(val.saturating_add(f - 1) / f),
			NodeAxis::Interval { lower, upper } => {
				NodeAxis::interval(lower.saturating_add(f - 1) / f, upper.saturating_add(f - 1) / f)
			}
		})
		.into();

	let output = input
		.graph()
		.new_node(output_shape)
		.set_name_unique(&format!("avg_pool({})", input));

	AvgPool::new(input, output.clone(), factors).build()?;

	Ok(output)
}

/// Average Pooling operation
///
/// Decrease size of dimensions by given factors.
/// Output values are the average of windows of the input with the size of factors
#[must_use]
#[derive(Clone, Debug)]
pub struct AvgPool {
	name: Option<String>,
	input: Node,
	output: Node,
	factors: Vec<usize>,
}

impl AvgPool {
	pub fn new<I, O>(input: I, output: O, factors: &[usize]) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		let input = input.into();
		let output = output.into();
		AvgPool {
			name: None,
			input,
			output,
			factors: factors.to_vec(),
		}
	}
}

impl OpBuilder for AvgPool {
	type InstanceType = AvgPoolInstance;

	fn type_name(&self) -> &'static str {
		"AvgPool"
	}

	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input.clone()]
	}

	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		if self.input.shape().len() != self.factors.len() {
			return Err(format!(
				"pooling factors length ({}) must be the same as input shape length ({})",
				self.factors.len(),
				self.input.shape().len(),
			)
			.into());
		}

		if self.input.shape().len() != self.output.shape().len() {
			return Err(format!(
				"output shape length ({}) must be the same as input shape length ({})",
				self.output.shape().len(),
				self.input.shape().len(),
			)
			.into());
		}

		if self.factors.iter().any(|&f| f == 0) {
			return Err(format!("all factors ({}) must be greater than 0", self.factors.len(),).into());
		}

		// TODO check for shape problems early
		Ok(AvgPoolInstance {
			input: self.input.inner().clone(),
			output: self.output.inner().clone(),
			factors: self.factors.clone(),
		})
	}
}

#[derive(Debug, Clone)]
pub struct AvgPoolInstance {
	input: NodeInner,
	output: NodeInner,
	factors: Vec<usize>,
}

impl OpInstance for AvgPoolInstance {
	fn type_name(&self) -> &'static str {
		"AvgPool"
	}

	fn inputs(&self) -> IndexSet<NodeInner> {
		indexset![self.input.clone()]
	}

	fn outputs(&self) -> IndexSet<NodeInner> {
		indexset![self.output.clone()]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		AvgPoolBack::new(ctx.grad_of(&self.output), ctx.grad_of(&self.input), &self.factors).build()?;

		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let input_shape = ctx.input_shape(&self.input);

		debug_assert_eq!(input_shape.ndim(), self.factors.len()); // This should be caught in the builder

		let output_shape: NodeShape = input_shape
			.slice()
			.iter()
			.zip(&self.factors)
			.map(|(i, f)| i.saturating_add(f - 1) / f)
			.into();

		ctx.merge_output_shape(&self.output, &output_shape)?;
		Ok(())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let input = ctx.get_input_standard(&self.input);
		let mut output = ctx.get_output_standard(&self.output);

		let input_shape = input.shape();
		let output_shape = output.shape().to_vec();

		debug_assert_eq!(input_shape.len(), self.factors.len()); // should be caught in builder
		debug_assert_eq!(input_shape.len(), output_shape.len()); // should be caught in builder
		debug_assert!(input_shape
			.iter()
			.zip(&self.factors)
			.map(|(i, f)| i.saturating_add(f - 1) / f)
			.eq(output_shape.iter().cloned())); // should be caught in shape prop

		let input = input.as_slice().unwrap(); // these unwraps are ok as the _standard array accessors are used
		let output = output.as_slice_mut().unwrap();

		let input_strides = strides(input_shape);
		let output_strides = strides(&output_shape);

		let scale = 1.0;

		let axis = 0;

		// starting from the innermost dims, skip dimensions that match then find the product of the
		let outer_dims = input_shape
			.iter()
			.zip(&output_shape)
			.rev()
			.skip_while(|&(i, o)| i == o)
			.map(|(_i, o)| o)
			.count();
		let n = output_shape[..outer_dims].iter().product();
		let ind_stride = output.len() / n;

		for i in 0..n {
			let output_ind = i * ind_stride;
			let ox = output_ind / output_strides[axis];
			pool_recurse_forward(
				input,
				output,
				input_shape,
				&output_shape,
				&self.factors,
				&input_strides,
				&output_strides,
				ox,
				axis,
				output_ind,
				scale,
			)
		}

		Ok(())
	}
}

#[derive(Debug, Clone)]
pub struct AvgPoolBack {
	output_grad: Node,
	input_grad: Node,
	factors: Vec<usize>,
}

impl AvgPoolBack {
	pub fn new<I, O>(output_grad: O, input_grad: I, factors: &[usize]) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		let input_grad = input_grad.into();
		let output_grad = output_grad.into();
		AvgPoolBack {
			output_grad,
			input_grad,
			factors: factors.to_vec(),
		}
	}
}

impl OpBuilder for AvgPoolBack {
	type InstanceType = AvgPoolBackInstance;

	fn type_name(&self) -> &'static str {
		"AvgPoolBack"
	}

	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.output_grad.clone()]
	}

	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.input_grad.clone()]
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		if self.output_grad.shape().len() != self.factors.len() {
			return Err(format!(
				"pooling factors length ({}) must be the same as output_grad shape length ({})",
				self.factors.len(),
				self.output_grad.shape().len(),
			)
			.into());
		}

		if self.output_grad.shape().len() != self.input_grad.shape().len() {
			return Err(format!(
				"input_grad shape length ({}) must be the same as output_grad shape length ({})",
				self.input_grad.shape().len(),
				self.output_grad.shape().len(),
			)
			.into());
		}

		if self.factors.iter().any(|&f| f == 0) {
			return Err(format!("all factors ({}) must be greater than 0", self.factors.len(),).into());
		}

		// TODO check for shape problems early
		Ok(AvgPoolBackInstance {
			output_grad: self.output_grad.inner().clone(),
			input_grad: self.input_grad.inner().clone(),
			factors: self.factors.clone(),
		})
	}
}

#[derive(Debug, Clone)]
pub struct AvgPoolBackInstance {
	output_grad: NodeInner,
	input_grad: NodeInner,
	factors: Vec<usize>,
}

impl OpInstance for AvgPoolBackInstance {
	fn type_name(&self) -> &'static str {
		"AvgPoolBack"
	}

	fn inputs(&self) -> IndexSet<NodeInner> {
		indexset![self.output_grad.clone()]
	}

	fn outputs(&self) -> IndexSet<NodeInner> {
		indexset![self.input_grad.clone()]
	}

	fn gradient(&self, _ctx: &mut GradientContext) -> Result<(), GradientError> {
		Err(GradientError::Unimplemented)
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let output_shape = ctx.input_shape(&self.output_grad);

		debug_assert_eq!(output_shape.ndim(), self.factors.len()); // This should be caught in the builder

		let input_grad_shape: NodeShape = output_shape
			.slice()
			.iter()
			.zip(&self.factors)
			.map(|(o, f)| {
				let upper = f * o;
				let lower = upper.saturating_sub(f - 1);
				(lower, upper)
			})
			.into();

		ctx.merge_output_shape(&self.input_grad, &input_grad_shape)?;

		Ok(())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let output_grad = ctx.get_input_standard(&self.output_grad);
		let mut input_grad = ctx.get_output_standard(&self.input_grad);

		let input_shape = input_grad.shape().to_vec();
		let output_shape = output_grad.shape();

		debug_assert_eq!(input_shape.len(), self.factors.len()); // should be caught in builder
		debug_assert_eq!(input_shape.len(), output_shape.len()); // should be caught in builder
		debug_assert!(input_shape
			.iter()
			.zip(&self.factors)
			.map(|(i, f)| i.saturating_add(f - 1) / f)
			.eq(output_shape.iter().cloned())); // should be caught in shape prop

		let input_grad = input_grad.as_slice_mut().unwrap();
		let output_grad = output_grad.as_slice().unwrap();

		let input_strides = strides(&input_shape);
		let output_strides = strides(output_shape);

		let scale = 1.0;
		let axis = 0;

		// starting from the innermost dims, skip dimensions that match then find the product of the
		let outer_dims = input_shape
			.iter()
			.zip(output_shape)
			.rev()
			.skip_while(|&(i, o)| i == o)
			.map(|(_i, o)| o)
			.count();
		let n = output_shape[..outer_dims].iter().product();
		let ind_stride = output_grad.len() / n;

		// TODO consider lifting outer unit factor dimensions into an outer loop like in linterp
		for i in 0..n {
			let output_ind = i * ind_stride;
			let ox = output_ind / output_strides[axis];
			pool_recurse_backward(
				input_grad,
				output_grad,
				&input_shape,
				output_shape,
				&self.factors,
				&input_strides,
				&output_strides,
				ox,
				axis,
				output_ind,
				scale,
			)
		}

		Ok(())
	}
}

fn strides(shape: &[usize]) -> SmallVec<[usize; 6]> {
	let mut strides = shape
		.iter()
		.rev()
		.scan(1, |state, &i| {
			let res = Some(*state);
			*state *= i;
			res
		})
		.collect::<SmallVec<[usize; 6]>>();
	strides.reverse();
	strides
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn pool_recurse_forward(
	input: &[f32],
	output: &mut [f32],
	input_shape: &[usize],
	output_shape: &[usize],
	factors: &[usize],
	input_strides: &[usize],
	output_strides: &[usize],
	ox: usize,
	axis: usize,
	output_ind: usize,
	scale: f32,
) {
	if input.len() != output.len() {
		let start = ox * factors[axis];
		let end = min(start + factors[axis], input_shape[axis]);

		let scale = scale / (end - start) as f32;

		let new_axis = axis + 1;
		let new_output = &mut output[output_strides[axis] * ox..output_strides[axis] * (ox + 1)];
		let new_output_ind = output_ind - ox * output_strides[axis];
		let new_ox = new_output_ind / output_strides[new_axis];
		for ix in start..end {
			let new_input = &input[input_strides[axis] * ix..input_strides[axis] * (ix + 1)];
			pool_recurse_forward(
				new_input,
				new_output,
				input_shape,
				output_shape,
				factors,
				input_strides,
				output_strides,
				new_ox,
				new_axis,
				new_output_ind,
				scale,
			);
		}
	} else {
		for i in 0..input.len() {
			output[i] += input[i] * scale;
		}
	}
}

#[allow(clippy::too_many_arguments)]
fn pool_recurse_backward(
	input_grad: &mut [f32],
	output_grad: &[f32],
	input_shape: &[usize],
	output_shape: &[usize],
	factors: &[usize],
	input_strides: &[usize],
	output_strides: &[usize],
	ox: usize,
	axis: usize,
	output_ind: usize,
	scale: f32,
) {
	if input_grad.len() != output_grad.len() {
		let start = ox * factors[axis];
		let end = min(start + factors[axis], input_shape[axis]);

		let scale = scale / (end - start) as f32;

		let new_axis = axis + 1;
		let new_output_grad = &output_grad[output_strides[axis] * ox..output_strides[axis] * (ox + 1)];
		let new_output_ind = output_ind - ox * output_strides[axis];
		let new_ox = new_output_ind / output_strides[new_axis];
		for ix in start..end {
			let new_input_grad = &mut input_grad[input_strides[axis] * ix..input_strides[axis] * (ix + 1)];
			pool_recurse_backward(
				new_input_grad,
				new_output_grad,
				input_shape,
				output_shape,
				factors,
				input_strides,
				output_strides,
				new_ox,
				new_axis,
				new_output_ind,
				scale,
			);
		}
	} else {
		for i in 0..input_grad.len() {
			input_grad[i] += output_grad[i] * scale;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::avg_pool;
	use alumina_core::graph::Node;
	use alumina_test::grad_numeric_test::GradNumericTest;
	use indexmap::indexset;
	use ndarray::{ArrayD, IxDyn};

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[3, 7, 9, 13]).set_name("input");

		let output = avg_pool(&input, &[1, 2, 1, 1]).unwrap().set_name("output");

		GradNumericTest::new(&output, &indexset![&input]).run();
	}

	#[test]
	fn pooling_function_test() {
		let factors = vec![1, 2, 1, 3, 1];
		let input = ArrayD::from_elem(IxDyn(&[3, 7, 9, 8, 13]), ::std::f32::consts::PI);
		let mut output = ArrayD::from_elem(IxDyn(&[3, 4, 9, 3, 13]), -0.5);

		let input_shape = input.shape();
		let output_shape = output.shape().to_vec();

		let input = input.as_slice().unwrap();
		let output = output.as_slice_mut().unwrap();

		let input_strides = super::strides(input_shape);
		let output_strides = super::strides(&output_shape);

		let scale = 1.0;

		let axis = 0;

		// starting from the innermost dims, skip dimensions that match then find the product of the
		let n = input_shape
			.iter()
			.zip(&output_shape)
			.rev()
			.skip_while(|&(i, o)| i == o)
			.map(|(_i, o)| o)
			.product();
		let ind_stride = output.len() / n;

		for i in 0..n {
			let output_ind = i * ind_stride;
			let ox = output_ind / output_strides[axis];
			super::pool_recurse_forward(
				input,
				output,
				input_shape,
				&output_shape,
				&factors,
				&input_strides,
				&output_strides,
				ox,
				axis,
				output_ind,
				scale,
			)
		}

		assert!(output.iter().all(|e| (e - 2.6415).abs() < 0.001), "{:?}", output);
	}

	#[test]
	fn pooling_function_unit_test() {
		let factors = vec![1, 1, 1, 1, 1];
		let input = ArrayD::from_elem(IxDyn(&[3, 7, 9, 8, 13]), ::std::f32::consts::PI);
		let mut output = ArrayD::from_elem(IxDyn(&[3, 7, 9, 8, 13]), -0.5);

		let input_shape = input.shape();
		let output_shape = output.shape().to_vec();

		let input = input.as_slice().unwrap();
		let output = output.as_slice_mut().unwrap();

		let input_strides = super::strides(input_shape);
		let output_strides = super::strides(&output_shape);

		let scale = 1.0;

		let axis = 0;

		// starting from the innermost dims, skip dimensions that match then find the product of the
		let n = input_shape
			.iter()
			.zip(&output_shape)
			.rev()
			.skip_while(|&(i, o)| i == o)
			.map(|(_i, o)| o)
			.product();
		let ind_stride = output.len() / n;

		for i in 0..n {
			let output_ind = i * ind_stride;
			let ox = output_ind / output_strides[axis];
			super::pool_recurse_forward(
				input,
				output,
				input_shape,
				&output_shape,
				&factors,
				&input_strides,
				&output_strides,
				ox,
				axis,
				output_ind,
				scale,
			)
		}

		assert!(output.iter().all(|e| (e - 2.6415).abs() < 0.001), "{:?}", output);
	}
}
