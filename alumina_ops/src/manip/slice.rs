

use crate::{
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{Node, NodeInner},
	ops::{ExecutionError, GradientError, OpBuildError, OpBuilder, OpInstance, ShapePropError},
	shape::{NodeAxis, NodeShape},
	shape_prop::ShapePropContext,
};
use indexmap::IndexSet;
use ndarray::{IxDyn, Dimension, SliceInfo, SliceOrIndex};
use smallvec::SmallVec;
use std::cmp::min;
use std::sync::Arc;
use std::fmt::Debug;

// SliceInfoFn exists for Into conversions from the following types:
// |shape|
struct SliceInfoFn {
	func: Arc<Fn(&[usize]) -> SliceInfo<Vec<SliceOrIndex>, IxDyn>>
}

impl Debug for SliceInfoFn {
	fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
		write!(f, "SliceFn {{ .. }}")
	}
}

impl Clone for SliceInfoFn {
	fn clone(&self) -> Self {
		SliceInfoFn{
			func: self.func.clone(),
		}
	}
}

// impl<N, F> From<F> for SliceInfoFn where N: Into<SliceInfoFn>, F: Fn(&[usize]) -> N {
// 	fn from(f: F) -> Self {
// 		SliceInfoFn {
// 			func: Arc::new(move |shape| {
// 				let x = f(shape).into();
// 				x(shape)
// 			})
// 		}
// 	}
// }

impl From<&SliceInfo<[SliceOrIndex], IxDyn>> for SliceInfoFn {
	fn from(index: &SliceInfo<[SliceOrIndex], IxDyn>) -> Self {
		let index : SliceInfo<Vec<SliceOrIndex>, IxDyn> = SliceInfo::new(index.as_ref().to_vec()).expect("Error converting SliceInfo<[_], _> to SliceInfo<Vec<_>, _>");
		SliceInfoFn {
			func: Arc::new(move |_| index.clone())
		}
	}
}

impl From<SliceInfo<Vec<SliceOrIndex>, IxDyn>> for SliceInfoFn {
	fn from(index: SliceInfo<Vec<SliceOrIndex>, IxDyn>) -> Self {
		SliceInfoFn {
			func: Arc::new(move |_| index.clone())
		}
	}
}

impl From<&[isize]> for SliceInfoFn {
	fn from(index: &[isize]) -> Self {
		let index: SliceInfo<Vec<SliceOrIndex>, IxDyn> = SliceInfo::new(index.iter().map(|i| SliceOrIndex::Slice{start: 0, end: Some(i), step: 1}).collect()).expect("Error converting &[usize] to SliceInfo");

		SliceInfoFn {
			func: Arc::new(move |_| index.clone())
		}
	}
}

impl From<&[(isize, isize)]> for SliceInfoFn {
	fn from(index: &[(isize, isize)]) -> Self {
		let index: SliceInfo<Vec<SliceOrIndex>, IxDyn> = SliceInfo::new(index.iter().map(|&(s, e)| SliceOrIndex::Slice{start: s, end: Some(e), step: 1}).collect()).expect("Error converting &[usize] to SliceInfo");

		SliceInfoFn {
			func: Arc::new(move |_| index.clone())
		}
	}
}

pub fn slice<I, N>(input: I, info: N) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
	N: Into<SliceInfoFn>
{
	// let input = input.into();

	// let output_shape: NodeShape = input
	// 	.shape()
	// 	.slice()
	// 	.iter()
	// 	.zip(factors)
	// 	.map(|(i, f)| match i {
	// 		NodeAxis::Known { val } => NodeAxis::known(val.saturating_add(f - 1) / f),
	// 		NodeAxis::Interval { lower, upper } => {
	// 			NodeAxis::interval(lower.saturating_add(f - 1) / f, upper.saturating_add(f - 1) / f)
	// 		},
	// 	})
	// 	.into();

	// let output = input
	// 	.graph()
	// 	.new_node(output_shape)
	// 	.set_name_unique(&format!("avg_pool({})", input));

	// AvgPool::new(input, output.clone(), factors).build()?;

	// Ok(output)
	unimplemented!()
}

/// Average Pooling operation
///
/// Decrease size of dimensions by given factors.
/// Output values are the average of windows of the input with the size of factors
#[must_use]
#[derive(Clone, Debug)]
pub struct Slice {
	name: Option<String>,
	input: Node,
	output: Node,
	index_fn: SliceFn,
}

impl Slice {
	pub fn new<I, O>(input: I, output: O, index_fn: SliceFn) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		let input = input.into();
		let output = output.into();
		Slice {
			name: None,
			input,
			output,
			index_fn,
		}
	}
}

impl OpBuilder for Slice {
	type InstanceType = SliceInstance;

	fn type_name(&self) -> &'static str {
		"Slice"
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
		Ok(SliceInstance {
			input: self.input.inner().clone(),
			output: self.output.inner().clone(),
			factors: self.factors.clone(),
		})
	}
}

#[derive(Debug, Clone)]
pub struct SliceInstance {
	input: NodeInner,
	output: NodeInner,
	factors: Vec<usize>,
}

impl OpInstance for SliceInstance {
	fn type_name(&self) -> &'static str {
		"Slice"
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












/// Average Pooling operation
///
/// Decrease size of dimensions by given factors.
/// Output values are the average of windows of the input with the size of factors
#[must_use]
#[derive(Clone, Debug)]
pub struct SliceBack {
	name: Option<String>,
	input: Node,
	output: Node,
	factors: Vec<usize>,
}

impl SliceBack {
	pub fn new<I, O>(input: I, output: O, factors: &[usize]) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		let input = input.into();
		let output = output.into();
		SliceBack {
			name: None,
			input,
			output,
			factors: factors.to_vec(),
		}
	}
}

impl OpBuilder for SliceBack {
	type InstanceType = SliceBackInstance;

	fn type_name(&self) -> &'static str {
		"SliceBack"
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
		Ok(SliceBackInstance {
			input: self.input.inner().clone(),
			output: self.output.inner().clone(),
			factors: self.factors.clone(),
		})
	}
}

#[derive(Debug, Clone)]
pub struct SliceBackInstance {
	input: NodeInner,
	output: NodeInner,
	factors: Vec<usize>,
}

impl OpInstance for SliceBackInstance {
	fn type_name(&self) -> &'static str {
		"SliceBack"
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