//! Removes unit sized dimensions from the shape of the input.
//!
//! This is primarily to allow more flexible reduction to an input of lower rank.
//!
//! After converting to positive axes duplicates will be removed.
//! The length of the output shape is the input shape length minus the deduplicated axes.
//! Each element of `axes` can be in the range [-input.len(), input.len()).
//!
//! # Examples
//!
//! ```
//! # use alumina_core::graph::Node;
//! # use alumina_core::shape::{NodeShape};
//! # use alumina_core::errors::OpBuildError;
//! # use alumina_ops::manip::remove_dims::remove_dims;
//! # fn main() -> Result<(), OpBuildError> {
//! let input = Node::new(&[1, 1, 5, 6, 1, 7, 1]);
//! let output = remove_dims(&input, &[0, 1, 4, 6])?;
//!
//! let expected: NodeShape = (&[5, 6, 7]).into();
//!
//! assert_eq!(expected, output.shape());
//! # Ok(())
//! # }
//! ```

use crate::manip::expand_dims::ExpandDims;
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

/// Removes unit axes from a nodes shape.
///
/// # Panics
/// Panics if an axis is not in the range [-input.len(), input.len()).
/// Panics if an axis is not known to be 1.
pub fn remove_dims<I>(input: I, axes: &[isize]) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let input_len = input.shape().ndim();

	let mut axes: SmallVec<[usize; 4]> = axes.iter().map(|&dim| wrap_dim(dim, input_len)).collect();
	axes.sort_unstable();
	axes.dedup();

	for (i, &dim) in axes.iter().enumerate() {
		match input.shape().slice()[dim] {
			NodeAxis::Known { val: 1 } => {},
			_ => panic!("Arg Error: non-unit axis ({}) removed at position ({})", dim, i),
		}
	}

	let mut axes_i = 0;
	let output_shape: NodeShape = input
		.shape()
		.into_iter()
		.enumerate()
		.filter_map(|(i, na)| {
			if axes_i < axes.len() && i == axes[axes_i] {
				axes_i += 1;
				None
			} else {
				Some(na)
			}
		})
		.into();

	let output = input
		.graph()
		.new_node(output_shape)
		.set_name_unique(&format!("remove_dims({})", input));

	let _op = RemoveDims {
		input,
		output: output.clone(),
		axes,
	}
	.build()
	.expect("Error building RemoveDims Op");

	Ok(output)
}

/// `RemoveDims` `OpBuilder`
#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct RemoveDims {
	output: Node,
	input: Node,
	axes: SmallVec<[usize; 4]>,
}

impl RemoveDims {
	pub fn new<I, O>(input: I, output: O) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		// TODO move checks from methods above to here
		RemoveDims {
			input: input.into(),
			output: output.into(),
			axes: SmallVec::new(),
		}
	}

	/// Removes unit sized dimensions from the shape of the input.
	///
	/// This is primarily to allow more flexible broadcasting to an output of higher rank.
	/// An argument of &[1, 4, 5] and an input shape of [4, 1, 5, 6, 1, 1, 7] would produce an output shape of [4, 5,
	/// 6, 7]. Each element of `axes` can be in the range [-input.shape().len(),
	/// input.shape().len()). After converting to positive axes duplicates
	/// will be removed.
	///
	/// Default: empty
	///
	/// # Panics
	/// Panics if an axis is not in the range [-input.len(), input.len()).
	/// Panics if the input shape length minus the number of deduplicated axes is not equal to the output shape length.
	pub fn axes(mut self, axes: &[isize]) -> Self {
		let input_len = self.input.shape().ndim();

		self.axes = axes.iter().map(|&dim| wrap_dim(dim, input_len)).collect();
		self.axes.sort_unstable();
		self.axes.dedup();
		assert!(
			input_len - self.axes.len() == self.output.shape().ndim(),
			"Arg Error: input.shape().ndim() + dedup(axes).len() does not equal output.shape().ndim(), ({}) ({}) ({}) respectively",
			input_len,
			self.axes.len(),
			self.output.shape().ndim()
		);
		for (i, &dim) in self.axes.iter().enumerate() {
			match self.input.shape().slice()[dim] {
				NodeAxis::Known { val: 1 } => {},
				_ => panic!("Arg Error: non-unit axis ({}) removed at position ({})", dim, i),
			}
		}

		self
	}
}

impl OpSpecification for RemoveDims {
	type InstanceType = RemoveDimsInstance;

	fn type_name(&self) -> &'static str {
		"RemoveDims"
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
			axes: self.axes.clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(RemoveDimsInstance {
			input: self.input.id(),
			output: self.output.id(),
			axes: self.axes,
		})
	}
}

/// RemoveDims OpInstance
#[derive(Clone, Debug)]
pub struct RemoveDimsInstance {
	input: NodeID,
	output: NodeID,
	axes: SmallVec<[usize; 4]>, // must be sorted
}

impl OpInstance for RemoveDimsInstance {
	fn type_name(&self) -> &'static str {
		"RemoveDims"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(RemoveDims {
			input: graph.node_from_id(self.input),
			output: graph.node_from_id(self.output),
			axes: self.axes.clone(),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		let extra_axes: SmallVec<[isize; 8]> = self.axes.iter().map(|&axis| axis as isize).collect();
		ExpandDims::new(ctx.grad_of(&self.output), ctx.grad_of(&self.input))
			.extra_axes(&extra_axes)
			.build()?;
		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let mut axes_i = 0;
		let output_shape: NodeShape = ctx
			.input_shape(&self.input)
			.slice()
			.iter()
			.enumerate()
			.filter_map(|(i, na)| {
				if axes_i < self.axes.len() && i == self.axes[axes_i] {
					axes_i += 1;
					None
				} else {
					Some(na)
				}
			})
			.into();

		ctx.merge_output_shape(&self.output, &output_shape)
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let mut axes_i = 0;
		let output_shape: SmallVec<[usize; 8]> = ctx
			.shape(&self.input)
			.iter()
			.enumerate()
			.filter_map(|(i, &na)| {
				if axes_i < self.axes.len() && i == self.axes[axes_i] {
					assert_eq!(1, na); // self.axes[axes_i]
					axes_i += 1;
					None
				} else {
					Some(na)
				}
			})
			.collect();

		if ctx.can_take(&self.input) && ctx.can_set(&self.output) {
			// if output can be set using the input array, do that.
			let input = ctx
				.take(&self.input)
				.into_shape(output_shape.as_slice())
				.unwrap_or_else(|err| {
					panic!(
						"Alumina Bug: RemoveDim ({}) reshape failed:\n{:#?}",
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
					"Alumina Bug: RemoveDim ({}) reshape failed:\n{:#?}",
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
