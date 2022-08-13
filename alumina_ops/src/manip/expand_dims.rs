//! Adds additional unit sized dimensions to the shape of the input.
//!
//! This is primarily to allow more flexible broadcasting to an output of higher rank.
//!
//! After converting to positive axes duplicates will be removed.
//! The length of the output shape is the sum of the input shape length and the deduplicated axes.
//! Each element of `axes` can be in the range [-output.len(), output.len()).
//!
//! # Examples
//!
//! ```
//! # use alumina_core::graph::Node;
//! # use alumina_core::shape::{NodeShape};
//! # use alumina_core::errors::OpBuildError;
//! # use alumina_ops::manip::expand_dims::expand_dims;
//! # fn main() -> Result<(), OpBuildError> {
//! let input = Node::new(&[5, 6, 7]);
//! let output = expand_dims(&input, &[0, 1, 4, 6])?;
//!
//! let expected: NodeShape = (&[1, 1, 5, 6, 1, 7, 1]).into();
//!
//! assert_eq!(expected, output.shape());
//! # Ok(())
//! # }
//! ```

use crate::manip::remove_dims::RemoveDims;
use alumina_core::{
	base_ops::{OpInstance, OpSpecification},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{Graph, Node, NodeID},
	shape::{NodeAxis, NodeShape},
	shape_prop::ShapePropContext,
	util::wrap_dim,
};
use indexmap::{indexset, IndexMap, IndexSet};
use ndarray::{ArrayViewD, ArrayViewMutD, Dimension};
use smallvec::SmallVec;
use std::any::Any;

/// Insert unit axes into a nodes shape.
///
/// Unlike with the `OpBuilder` the `extra_axes` argument must be `usize` rather than `isize` because reverse indexing
/// isn't possible before the output shape is known.
///
/// # Panics
/// Panics if an axis is not in the range [0, output.len()).
pub fn expand_dims<I>(input: I, extra_axes: &[usize]) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();

	let mut extra_axes: SmallVec<[usize; 4]> = extra_axes.iter().cloned().collect();
	extra_axes.sort_unstable();
	extra_axes.dedup();

	let mut output_shape: NodeShape = (0..input.shape().len() + extra_axes.len())
		.map(|_| NodeAxis::known(1))
		.into();

	for &dim in &extra_axes {
		assert!(
			dim < output_shape.len(),
			"Arg Error: expand_dims() produced an output with length {} but had an axis argument {}",
			output_shape.len(),
			dim
		);
	}

	let mut extra_count = 0;
	let mut input_count = 0;
	for i in 0..output_shape.len() {
		if extra_count < extra_axes.len() && i == extra_axes[extra_count] {
			extra_count += 1;
		} else {
			output_shape.slice_mut()[i] = input.shape().slice()[input_count].clone();
			input_count += 1;
		}
	}
	debug_assert!(input_count == input.shape().len());
	debug_assert!(extra_count == extra_axes.len());

	let output = input
		.graph()
		.new_node(output_shape)
		.set_name_unique(&format!("expand_dims({})", input));

	let _op = ExpandDims {
		input,
		output: output.clone(),
		extra_axes,
	}
	.build()?;

	Ok(output)
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct ExpandDims {
	output: Node,
	input: Node,
	extra_axes: SmallVec<[usize; 4]>,
}

impl ExpandDims {
	pub fn new<I, O>(input: I, output: O) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		ExpandDims {
			input: input.into(),
			output: output.into(),
			extra_axes: SmallVec::new(),
		}
	}

	/// Adds additional unit sized dimensions to the shape of the input.
	///
	/// An argument of &[1, 4, 5] and an input shape of [4, 5, 6, 7] would produce an output shape of [4, 1, 5, 6, 1,
	/// 1, 7]. Each element of `axes` can be in the range [-output.len(),
	/// output.len()). After converting to positive axes duplicates will be
	/// removed.
	///
	/// Default: empty
	///
	/// # Panics
	/// Panics if an axis is not in the range [-output.len(), output.len()).
	/// Panics if the sum of the input shape length and the number of deduplicated axes is not equal to the output
	/// shape length.
	pub fn extra_axes(mut self, extra_axes: &[isize]) -> Self {
		let output_len = self.output.shape().len();

		self.extra_axes = extra_axes.iter().map(|&dim| wrap_dim(dim, output_len)).collect();
		self.extra_axes.sort_unstable();
		self.extra_axes.dedup();
		assert!(
			self.extra_axes.len() + self.input.shape().ndim() == output_len,
			"Arg Error: input.shape().ndim() + dedup(extra_axes).len() does not equal output.shape().ndim(), ({}) ({}) ({}) respectively",
			self.input.shape().ndim(),
			self.extra_axes.len(),
			output_len
		);
		self
	}
}

impl OpSpecification for ExpandDims {
	type InstanceType = ExpandDimsInstance;

	fn op_type(&self) -> &'static str {
		"ExpandDims"
	}

	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input.clone()]
	}

	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			input: mapping.get(&self.input).unwrap_or(&self.input).clone(),
			output: mapping.get(&self.output).unwrap_or(&self.output).clone(),
			extra_axes: self.extra_axes.clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(ExpandDimsInstance {
			input: self.input.id(),
			output: self.output.id(),
			extra_axes: self.extra_axes,
		})
	}
}

fn expanded_shape(input_shape: &[usize], extra_axes: &[usize]) -> SmallVec<[usize; 8]> {
	let mut effective_shape: SmallVec<[usize; 8]> = (0..input_shape.len() + extra_axes.len()).map(|_| 1).collect();

	let mut extra_count = 0;
	let mut input_count = 0;
	// for i in 0..effective_shape.len() {
	for (i, effective_shape_i) in effective_shape.iter_mut().enumerate() {
		if extra_count < extra_axes.len() && i == extra_axes[extra_count] {
			extra_count += 1;
		} else {
			*effective_shape_i = input_shape[input_count];
			input_count += 1;
		}
	}
	debug_assert!(input_count == input_shape.len());

	effective_shape
}
// TODO test

/// ExpandDims OpInstance
#[derive(Clone, Debug)]
pub struct ExpandDimsInstance {
	input: NodeID,
	output: NodeID,
	extra_axes: SmallVec<[usize; 4]>,
}

impl OpInstance for ExpandDimsInstance {
	fn op_type(&self) -> &'static str {
		"ExpandDims"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(ExpandDims {
			input: graph.node_from_id(self.input),
			output: graph.node_from_id(self.output),
			extra_axes: self.extra_axes.clone(),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		let axes: SmallVec<[isize; 8]> = self.extra_axes.iter().map(|&axis| axis as isize).collect();
		RemoveDims::new(ctx.grad_of(&self.output), ctx.grad_of(&self.input))
			.axes(&axes)
			.build()?;
		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let output_shape = expanded_shape(ctx.input_shape(&self.input).slice(), &self.extra_axes)
			.iter()
			.into();

		ctx.merge_output_shape(&self.output, &output_shape)
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let output_shape = expanded_shape(ctx.shape(&self.input), &self.extra_axes);

		if ctx.can_take(&self.input) && ctx.can_set(&self.output) {
			// if output can be set using the input array, do that.
			let input = ctx
				.take(&self.input)
				.into_shape(output_shape.as_slice())
				.unwrap_or_else(|err| {
					panic!(
						"Alumina Bug: ExpandDim ({}) reshape failed:\n{:#?}",
						ctx.current_op(),
						err
					)
				});
			ctx.set(&self.output, input);
		} else {
			let input: ArrayViewD<f32> = ctx.get_input(&self.input);
			let mut output: ArrayViewMutD<f32> = ctx.get_output(&self.output);

			let input = input.into_shape(output_shape.as_slice()).unwrap_or_else(|err| {
				panic!(
					"Alumina Bug: ExpandDim ({}) reshape failed:\n{:#?}",
					ctx.current_op(),
					err
				)
			});
			output += &input;
		}

		Ok(())
	}
}

// TODO tests!
