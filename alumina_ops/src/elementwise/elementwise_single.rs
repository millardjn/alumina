//! Contains traits for ease of constructing new elementwise operations which produce a single output.
//!
//! Using the unary (single input) family as an example:
//!  * UnaryFunc is the trait which is unique to each implemented Op, defining the forward operation, and any relevant
//!    gradients.
//!  * UnaryElementwiseInstance<T: UnaryFunc> is the generic OpInstance
//!  * UnaryElementwise<T: UnaryFunc> implements the
//!
//! This optimised implementations are also available for Nullary, Binary, and Ternary, along with a less efficient
//! N-ary family for any input number up to 64. All Ops constructed this way have a single output.
//!
//! See the Scale Op for a simple example of how this is used.

use alumina_core::{
	base_ops::{OpInstance, OpSpecification},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{Graph, Node, NodeID},
	shape::NodeShape,
	shape_prop::ShapePropContext,
};
use indexmap::{indexset, IndexMap, IndexSet};

use ndarray::{Dimension, Zip};
use rayon::prelude::*;
use std::any::Any;
use std::fmt;

pub trait NullaryFunc: Send + Sync + Clone + fmt::Debug + 'static {
	fn calc(&self) -> f32;

	fn type_name(&self) -> &'static str;
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct NullaryElementwise<F: NullaryFunc> {
	output: Node,
	f: F,
}

impl<F: NullaryFunc> NullaryElementwise<F> {
	pub fn new<O>(output: O, f: F) -> Self
	where
		O: Into<Node>,
	{
		let output = output.into();
		NullaryElementwise { output, f }
	}

	pub fn new_default<O>(output: O) -> Self
	where
		O: Into<Node>,
		F: Default,
	{
		Self::new(output, F::default())
	}
}

impl<F: NullaryFunc> OpSpecification for NullaryElementwise<F> {
	type InstanceType = NullaryElementwiseInstance<F>;

	fn op_type(&self) -> &'static str {
		self.f.type_name()
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node> {
		indexset![]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			output: mapping.get(&self.output).unwrap_or(&self.output).clone(),
			f: self.f.clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(NullaryElementwiseInstance {
			output: self.output.id(),
			f: self.f,
		})
	}
}

/// Elementwise Op, the value of the input is added to
#[derive(Clone, Debug)]
pub struct NullaryElementwiseInstance<F: NullaryFunc> {
	output: NodeID,
	f: F,
}

impl<F: NullaryFunc> OpInstance for NullaryElementwiseInstance<F> {
	fn op_type(&self) -> &'static str {
		self.f.type_name()
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(NullaryElementwise {
			output: graph.node_from_id(self.output),
			f: self.f.clone(),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output]
	}

	fn gradient(&self, _ctx: &mut GradientContext) -> Result<(), GradientError> {
		Ok(())
	}

	fn propagate_shapes(&self, _ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		Ok(())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		Zip::from(ctx.get_output(&self.output)).par_for_each(|output| {
			*output += self.f.calc();
		});
		Ok(())
	}
}

pub trait UnaryFunc: Send + Sync + Clone + fmt::Debug + 'static {
	fn calc(&self, input: f32) -> f32;

	fn type_name(&self) -> &'static str;

	fn grad(&self, ctx: &mut GradientContext, input: &NodeID, output: &NodeID) -> Result<(), GradientError>;
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct UnaryElementwise<F: UnaryFunc> {
	output: Node,
	input: Node,
	f: F,
}

impl<F: UnaryFunc> UnaryElementwise<F> {
	pub fn new<I, O>(input: I, output: O, f: F) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		let input = input.into();
		let output = output.into();
		UnaryElementwise { output, input, f }
	}

	pub fn new_default<I, O>(input: I, output: O) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
		F: Default,
	{
		Self::new(input, output, F::default())
	}
}

impl<F: UnaryFunc> OpSpecification for UnaryElementwise<F> {
	type InstanceType = UnaryElementwiseInstance<F>;

	fn op_type(&self) -> &'static str {
		self.f.type_name()
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input.clone()]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			output: mapping.get(&self.output).unwrap_or(&self.output).clone(),
			input: mapping.get(&self.output).unwrap_or(&self.input).clone(),
			f: self.f.clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(UnaryElementwiseInstance {
			input: self.input.id(),
			output: self.output.id(),
			f: self.f,
		})
	}
}

/// Elementwise Op, the value of the input is added to
#[derive(Clone, Debug)]
pub struct UnaryElementwiseInstance<F: UnaryFunc> {
	input: NodeID,
	output: NodeID,
	f: F,
}

impl<F: UnaryFunc> OpInstance for UnaryElementwiseInstance<F> {
	fn op_type(&self) -> &'static str {
		self.f.type_name()
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(UnaryElementwise {
			input: graph.node_from_id(self.input),
			output: graph.node_from_id(self.output),
			f: self.f.clone(),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		self.f.grad(ctx, &self.input, &self.output)
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let input_shape: NodeShape = ctx.input_shape(&self.input).slice().iter().into();
		ctx.merge_output_shape(&self.output, &input_shape)
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		assert_eq!(
			ctx.shape(&self.input),
			ctx.shape(&self.output),
			"Alumina Bug: input shape: {:?} did not match output shape: {:?}",
			ctx.shape(&self.input),
			ctx.shape(&self.output)
		);

		if ctx.can_take(&self.input) && ctx.can_set(&self.output) {
			// if output can be set using the input array, update inplace and do that.
			let mut input = ctx.take(&self.input);
			input.par_map_inplace(|x| *x = self.f.calc(*x));

			// Zip::from(&mut input).par_apply(|el| {
			// 	*el += self.f.calc(*el);
			// });
			ctx.set(&self.output, input);
		} else {
			Zip::from(ctx.get_output(&self.output))
				.and(ctx.get_input(&self.input))
				.par_for_each(|output, &input| {
					*output += self.f.calc(input);
				});
		}

		Ok(())
	}
}

pub trait BinaryFunc: Send + Sync + Clone + fmt::Debug + 'static {
	fn calc(&self, input1: f32, input2: f32) -> f32;

	fn type_name(&self) -> &'static str;

	fn grad(
		&self,
		ctx: &mut GradientContext,
		input1: &NodeID,
		input2: &NodeID,
		output: &NodeID,
	) -> Result<(), GradientError>;
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct BinaryElementwise<F: BinaryFunc> {
	output: Node,
	input1: Node,
	input2: Node,
	f: F,
}

impl<F: BinaryFunc> BinaryElementwise<F> {
	pub fn new<I1, I2, O>(input1: I1, input2: I2, output: O, f: F) -> Self
	where
		I1: Into<Node>,
		I2: Into<Node>,
		O: Into<Node>,
	{
		let input1 = input1.into();
		let input2 = input2.into();
		let output = output.into();
		BinaryElementwise {
			output,
			input1,
			input2,
			f,
		}
	}

	pub fn new_default<I1, I2, O>(input1: I1, input2: I2, output: O) -> Self
	where
		I1: Into<Node>,
		I2: Into<Node>,
		O: Into<Node>,
		F: Default,
	{
		Self::new(input1, input2, output, F::default())
	}
}

impl<F: BinaryFunc> OpSpecification for BinaryElementwise<F> {
	type InstanceType = BinaryElementwiseInstance<F>;

	fn op_type(&self) -> &'static str {
		self.f.type_name()
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input1.clone(), self.input2.clone()]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			output: mapping.get(&self.output).unwrap_or(&self.output).clone(),
			input1: mapping.get(&self.output).unwrap_or(&self.input1).clone(),
			input2: mapping.get(&self.output).unwrap_or(&self.input2).clone(),
			f: self.f.clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(BinaryElementwiseInstance {
			input1: self.input1.id(),
			input2: self.input2.id(),
			output: self.output.id(),
			f: self.f,
		})
	}
}

/// Elementwise Op, the value of the input is added to
#[derive(Clone, Debug)]
pub struct BinaryElementwiseInstance<F: BinaryFunc> {
	input1: NodeID,
	input2: NodeID,
	output: NodeID,
	f: F,
}

impl<F: BinaryFunc> OpInstance for BinaryElementwiseInstance<F> {
	fn op_type(&self) -> &'static str {
		self.f.type_name()
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(BinaryElementwise {
			input1: graph.node_from_id(self.input1),
			input2: graph.node_from_id(self.input2),
			output: graph.node_from_id(self.output),
			f: self.f.clone(),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input1, self.input2]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		self.f.grad(ctx, &self.input1, &self.input2, &self.output)
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let input_shape1: NodeShape = ctx.input_shape(&self.input1).slice().iter().into();
		let input_shape2: NodeShape = ctx.input_shape(&self.input2).slice().iter().into();
		ctx.merge_output_shape(&self.output, &input_shape1.merge(&input_shape2)?)
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		assert_eq!(
			ctx.shape(&self.input1),
			ctx.shape(&self.output),
			"Alumina Bug: input shape: {:?} did not match output shape: {:?}",
			ctx.shape(&self.input1),
			ctx.shape(&self.output)
		);

		assert_eq!(
			ctx.shape(&self.input2),
			ctx.shape(&self.output),
			"Alumina Bug: input shape: {:?} did not match output shape: {:?}",
			ctx.shape(&self.input2),
			ctx.shape(&self.output)
		);

		if ctx.can_take(&self.input1) && ctx.can_set(&self.output) {
			// if output can be set using the input array, update inplace and do that.
			let mut input1 = ctx.take(&self.input1);
			if self.input1 == self.input2 {
				Zip::from(&mut input1).par_for_each(|in1| {
					*in1 = self.f.calc(*in1, *in1);
				});
			} else {
				let input2 = ctx.get_input(&self.input2);
				Zip::from(&mut input1).and(input2).par_for_each(|in1, &in2| {
					*in1 = self.f.calc(*in1, in2);
				});
			}
			ctx.set(&self.output, input1);
		} else if ctx.can_take(&self.input2) && ctx.can_set(&self.output) {
			let mut input2 = ctx.take(&self.input2);
			if self.input1 == self.input2 {
				Zip::from(&mut input2).par_for_each(|in2| {
					*in2 = self.f.calc(*in2, *in2);
				});
			} else {
				let input1 = ctx.get_input(&self.input1);
				Zip::from(&mut input2).and(input1).par_for_each(|in2, &in1| {
					*in2 = self.f.calc(in1, *in2);
				});
			}
			ctx.set(&self.output, input2);
		} else {
			Zip::from(ctx.get_output(&self.output))
				.and(ctx.get_input(&self.input1))
				.and(ctx.get_input(&self.input2))
				.par_for_each(|output, &input1, &input2| {
					*output += self.f.calc(input1, input2);
				});
		}

		Ok(())
	}
}

pub trait TernaryFunc: Send + Sync + Clone + fmt::Debug + 'static {
	fn calc(&self, input1: f32, input2: f32, input3: f32) -> f32;

	fn type_name(&self) -> &'static str;

	fn grad(
		&self,
		ctx: &mut GradientContext,
		input1: &NodeID,
		input2: &NodeID,
		input3: &NodeID,
		output: &NodeID,
	) -> Result<(), GradientError>;
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct TernaryElementwise<F: TernaryFunc> {
	output: Node,
	input1: Node,
	input2: Node,
	input3: Node,
	f: F,
}

impl<F: TernaryFunc> TernaryElementwise<F> {
	pub fn new<I1, I2, I3, O>(input1: I1, input2: I2, input3: I3, output: O, f: F) -> Self
	where
		I1: Into<Node>,
		I2: Into<Node>,
		I3: Into<Node>,
		O: Into<Node>,
	{
		let input1 = input1.into();
		let input2 = input2.into();
		let input3 = input3.into();
		let output = output.into();

		TernaryElementwise {
			output,
			input1,
			input2,
			input3,
			f,
		}
	}

	pub fn new_default<I1, I2, I3, O>(input1: I1, input2: I2, input3: I3, output: O) -> Self
	where
		I1: Into<Node>,
		I2: Into<Node>,
		I3: Into<Node>,
		O: Into<Node>,
		F: Default,
	{
		Self::new(input1, input2, input3, output, F::default())
	}
}

impl<F: TernaryFunc> OpSpecification for TernaryElementwise<F> {
	type InstanceType = TernaryElementwiseInstance<F>;

	fn op_type(&self) -> &'static str {
		self.f.type_name()
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input1.clone(), self.input2.clone()]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			output: mapping.get(&self.output).unwrap_or(&self.output).clone(),
			input1: mapping.get(&self.output).unwrap_or(&self.input1).clone(),
			input2: mapping.get(&self.output).unwrap_or(&self.input2).clone(),
			input3: mapping.get(&self.output).unwrap_or(&self.input3).clone(),
			f: self.f.clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(TernaryElementwiseInstance {
			input1: self.input1.id(),
			input2: self.input2.id(),
			input3: self.input3.id(),
			output: self.output.id(),
			f: self.f,
		})
	}
}

/// Elementwise Op, the value of the input is added to
#[derive(Clone, Debug)]
pub struct TernaryElementwiseInstance<F: TernaryFunc> {
	input1: NodeID,
	input2: NodeID,
	input3: NodeID,
	output: NodeID,
	f: F,
}

impl<F: TernaryFunc> OpInstance for TernaryElementwiseInstance<F> {
	fn op_type(&self) -> &'static str {
		self.f.type_name()
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(TernaryElementwise {
			input1: graph.node_from_id(self.input1),
			input2: graph.node_from_id(self.input2),
			input3: graph.node_from_id(self.input3),
			output: graph.node_from_id(self.output),
			f: self.f.clone(),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input1, self.input2, self.input3]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		self.f.grad(ctx, &self.input1, &self.input2, &self.input3, &self.output)
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let input_shape1: NodeShape = ctx.input_shape(&self.input1).slice().iter().into();
		let input_shape2: NodeShape = ctx.input_shape(&self.input2).slice().iter().into();
		let input_shape3: NodeShape = ctx.input_shape(&self.input3).slice().iter().into();
		ctx.merge_output_shape(&self.output, &input_shape1.merge(&input_shape2.merge(&input_shape3)?)?)
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		assert_eq!(
			ctx.shape(&self.input1),
			ctx.shape(&self.output),
			"Alumina Bug: input1 shape: {:?} did not match output shape: {:?}",
			ctx.shape(&self.input1),
			ctx.shape(&self.output)
		);

		assert_eq!(
			ctx.shape(&self.input2),
			ctx.shape(&self.output),
			"Alumina Bug: input2 shape: {:?} did not match output shape: {:?}",
			ctx.shape(&self.input2),
			ctx.shape(&self.output)
		);

		assert_eq!(
			ctx.shape(&self.input3),
			ctx.shape(&self.output),
			"Alumina Bug: input3 shape: {:?} did not match output shape: {:?}",
			ctx.shape(&self.input3),
			ctx.shape(&self.output)
		);

		if ctx.can_take(&self.input1) && ctx.can_set(&self.output) {
			// if output can be set using the input array, update inplace and do that.
			let mut input1 = ctx.take(&self.input1);
			if self.input1 == self.input2 && self.input1 == self.input3 {
				Zip::from(&mut input1).par_for_each(|in1| {
					*in1 = self.f.calc(*in1, *in1, *in1);
				});
			} else if self.input1 == self.input2 {
				let input3 = ctx.get_input(&self.input3);
				Zip::from(&mut input1).and(input3).par_for_each(|in1, in3| {
					*in1 = self.f.calc(*in1, *in1, *in3);
				});
			} else if self.input1 == self.input3 {
				let input2 = ctx.get_input(&self.input2);
				Zip::from(&mut input1).and(input2).par_for_each(|in1, in2| {
					*in1 = self.f.calc(*in1, *in2, *in1);
				});
			} else {
				let input2 = ctx.get_input(&self.input2);
				let input3 = ctx.get_input(&self.input3);
				Zip::from(&mut input1)
					.and(input2)
					.and(input3)
					.par_for_each(|in1, in2, in3| {
						*in1 = self.f.calc(*in1, *in2, *in3);
					});
			}
			ctx.set(&self.output, input1);
		} else if ctx.can_take(&self.input2) && ctx.can_set(&self.output) {
			let mut input2 = ctx.take(&self.input2);
			if self.input2 == self.input1 && self.input2 == self.input3 {
				Zip::from(&mut input2).par_for_each(|in2| {
					*in2 = self.f.calc(*in2, *in2, *in2);
				});
			} else if self.input2 == self.input1 {
				let input3 = ctx.get_input(&self.input3);
				Zip::from(&mut input2).and(input3).par_for_each(|in2, in3| {
					*in2 = self.f.calc(*in2, *in2, *in3);
				});
			} else if self.input2 == self.input3 {
				let input1 = ctx.get_input(&self.input1);
				Zip::from(&mut input2).and(input1).par_for_each(|in2, in1| {
					*in2 = self.f.calc(*in1, *in2, *in2);
				});
			} else {
				let input1 = ctx.get_input(&self.input1);
				let input3 = ctx.get_input(&self.input3);
				Zip::from(&mut input2)
					.and(input1)
					.and(input3)
					.par_for_each(|in2, in1, in3| {
						*in2 = self.f.calc(*in1, *in2, *in3);
					});
			}
			ctx.set(&self.output, input2);
		} else if ctx.can_take(&self.input3) && ctx.can_set(&self.output) {
			let mut input3 = ctx.take(&self.input3);
			if self.input3 == self.input1 && self.input3 == self.input2 {
				Zip::from(&mut input3).par_for_each(|in3| {
					*in3 = self.f.calc(*in3, *in3, *in3);
				});
			} else if self.input3 == self.input1 {
				let input2 = ctx.get_input(&self.input2);
				Zip::from(&mut input3).and(input2).par_for_each(|in3, in2| {
					*in3 = self.f.calc(*in3, *in2, *in3);
				});
			} else if self.input3 == self.input2 {
				let input1 = ctx.get_input(&self.input1);
				Zip::from(&mut input3).and(input1).par_for_each(|in3, in1| {
					*in3 = self.f.calc(*in1, *in3, *in3);
				});
			} else {
				let input1 = ctx.get_input(&self.input1);
				let input2 = ctx.get_input(&self.input2);
				Zip::from(&mut input3)
					.and(input1)
					.and(input2)
					.par_for_each(|in3, in1, in2| {
						*in3 = self.f.calc(*in1, *in2, *in3);
					});
			}
			ctx.set(&self.output, input3);
		} else {
			Zip::from(ctx.get_output(&self.output))
				.and(ctx.get_input(&self.input1))
				.and(ctx.get_input(&self.input2))
				.and(ctx.get_input(&self.input3))
				.par_for_each(|output, in1, in2, in3| {
					*output += self.f.calc(*in1, *in2, *in3);
				});
		}
		Ok(())
	}
}

pub trait NaryFunc: Send + Sync + Clone + fmt::Debug + 'static {
	fn calc(&self, input: &[f32]) -> f32;

	fn type_name(&self) -> &'static str;

	fn grad(&self, ctx: &mut GradientContext, inputs: &[NodeID], output: &NodeID) -> Result<(), GradientError>;
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct NaryElementwise<F: NaryFunc> {
	output: Node,
	inputs: Vec<Node>,
	f: F,
}

impl<F: NaryFunc> NaryElementwise<F> {
	pub fn new<I, O, T: IntoIterator<Item = I>>(inputs: T, output: O, f: F) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		let inputs: Vec<Node> = inputs.into_iter().map(Into::into).collect();
		let output = output.into();
		assert!(
			inputs.len() <= 64,
			"Nary Ops can only be constructed with up to 64 inputs"
		);
		NaryElementwise { output, inputs, f }
	}

	pub fn new_default<I, T: IntoIterator<Item = I>, O>(inputs: T, output: O) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
		F: Default,
	{
		Self::new(inputs, output, F::default())
	}
}

impl<F: NaryFunc> OpSpecification for NaryElementwise<F> {
	type InstanceType = NaryElementwiseInstance<F>;

	fn op_type(&self) -> &'static str {
		self.f.type_name()
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node> {
		self.inputs.iter().cloned().collect()
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			output: mapping.get(&self.output).unwrap_or(&self.output).clone(),
			inputs: self
				.inputs
				.iter()
				.map(|i| mapping.get(i).unwrap_or(i))
				.cloned()
				.collect(),
			f: self.f.clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(NaryElementwiseInstance {
			inputs: self.inputs.iter().map(|n| n.id()).collect(),
			output: self.output.id(),
			f: self.f,
		})
	}
}

/// Elementwise Op, the value of the input is added to
#[derive(Clone, Debug)]
pub struct NaryElementwiseInstance<F: NaryFunc> {
	inputs: Vec<NodeID>,
	output: NodeID,
	f: F,
}

impl<F: NaryFunc> OpInstance for NaryElementwiseInstance<F> {
	fn op_type(&self) -> &'static str {
		self.f.type_name()
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(NaryElementwise {
			inputs: self.inputs.iter().map(|&i| graph.node_from_id(i)).collect(),
			output: graph.node_from_id(self.output),
			f: self.f.clone(),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		self.inputs.iter().cloned().collect()
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		self.f.grad(ctx, &self.inputs, &self.output)
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		if !self.inputs.is_empty() {
			let mut input_shape: NodeShape = ctx.input_shape(&self.inputs[0]).slice().iter().into();
			for input in &self.inputs[1..] {
				let next_shape: NodeShape = ctx.input_shape(input).slice().iter().into();
				input_shape = input_shape.merge(&next_shape)?;
			}
			ctx.merge_output_shape(&self.output, &input_shape)
		} else {
			Ok(())
		}
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		// No input shortcut
		if self.inputs.is_empty() {
			let mut output = ctx.get_output_standard(&self.output);

			let len = output.len();

			struct Ptrs {
				output: *mut f32,
			}
			unsafe impl Sync for Ptrs {}

			let ptrs = Ptrs {
				output: output.as_slice_mut().unwrap().as_mut_ptr(),
			};

			(0..len).into_par_iter().for_each(|j| unsafe {
				let arr = [0.0; 0];
				*ptrs.output.add(j) += self.f.calc(&arr[..]);
			});
			return Ok(());
		}

		for (i, input) in self.inputs.iter().enumerate() {
			assert_eq!(
				ctx.shape(input),
				ctx.shape(&self.output),
				"Alumina Bug: input {} shape: {:?} did not match output shape: {:?}",
				i,
				ctx.shape(input),
				ctx.shape(&self.output)
			);
		}

		if ctx.can_set(&self.output) {
			for (i, input) in self.inputs.iter().enumerate() {
				if ctx.can_take(input) && self.inputs.iter().filter(|&n| n == input).count() == 1 {
					// Inplace version

					let input_arr = ctx.take(input);

					let len = input_arr.len();

					let inputs: Vec<*mut f32> = (0..self.inputs.len())
						.map(|j| {
							if i == j {
								input_arr.as_slice().unwrap().as_ptr() as *mut f32
							} else {
								let arr = ctx.get_input_standard(&self.inputs[j]);
								let slice = arr.as_slice().unwrap();
								debug_assert_eq!(len, slice.len());
								slice.as_ptr() as *mut f32
							}
						})
						.collect();

					struct Ptrs {
						inputs: Vec<*mut f32>,
					}
					unsafe impl Sync for Ptrs {}

					let ptrs = Ptrs { inputs };

					// get input slices into a vec

					if self.inputs.len() <= 8 {
						// par_iter + unsafe + array
						(0..len).into_par_iter().for_each(|j| {
							unsafe {
								let mut arr = [0.0; 8]; // hopefully this array gets optimised out
								for k in 0..ptrs.inputs.len() {
									*arr.get_unchecked_mut(k) = *ptrs.inputs.get_unchecked(k).offset(j as isize);
								}
								*ptrs.inputs.get_unchecked(i).offset(j as isize) =
									self.f.calc(&arr[0..ptrs.inputs.len()]);
							}
						});
					} else if self.inputs.len() <= 64 {
						(0..len).into_par_iter().for_each(|j| {
							unsafe {
								let mut arr = [0.0; 64]; // hopefully this array gets optimised out
								for k in 0..ptrs.inputs.len() {
									*arr.get_unchecked_mut(k) = *ptrs.inputs.get_unchecked(k).offset(j as isize);
								}
								*ptrs.inputs.get_unchecked(i).offset(j as isize) =
									self.f.calc(&arr[0..ptrs.inputs.len()]);
							}
						});
					} else {
						return Err("NaryElementwise does not currently support more than 64 inputs"
							.to_string()
							.into());
					}
					ctx.set(&self.output, input_arr);
					return Ok(());
				}
			}
		}

		{
			// Non-inplace version
			let mut output = ctx.get_output_standard(&self.output);

			let len = output.len();

			let inputs: Vec<*mut f32> = (0..self.inputs.len())
				.map(|j| {
					let arr = ctx.get_input_standard(&self.inputs[j]);
					let slice = arr.as_slice().unwrap();
					debug_assert_eq!(len, slice.len());
					slice.as_ptr() as *mut f32
				})
				.collect();

			struct Ptrs {
				inputs: Vec<*mut f32>,
				output: *mut f32,
			}
			unsafe impl Sync for Ptrs {}

			let ptrs = Ptrs {
				inputs,
				output: output.as_slice_mut().unwrap().as_mut_ptr(),
			};

			// get input slices into a vec

			if ptrs.inputs.len() <= 8 {
				// par_iter + unsafe + array
				(0..len).into_par_iter().for_each(|j| {
					unsafe {
						let mut arr = [0.0; 8]; // hopefully this array gets optimised out
						for k in 0..ptrs.inputs.len() {
							*arr.get_unchecked_mut(k) = *ptrs.inputs.get_unchecked(k).offset(j as isize);
						}
						*ptrs.output.add(j) += self.f.calc(&arr[0..ptrs.inputs.len()]);
					}
				});
			} else if ptrs.inputs.len() <= 64 {
				(0..len).into_par_iter().for_each(|j| {
					unsafe {
						let mut arr = [0.0; 64]; // hopefully this array gets optimised out
						for k in 0..ptrs.inputs.len() {
							*arr.get_unchecked_mut(k) = *ptrs.inputs.get_unchecked(k).offset(j as isize);
						}
						*ptrs.output.add(j) += self.f.calc(&arr[0..ptrs.inputs.len()]);
					}
				});
			} else {
				return Err("NaryElementwise does not currently support more than 64 inputs"
					.to_string()
					.into());
			}
		}

		Ok(())
	}
}
