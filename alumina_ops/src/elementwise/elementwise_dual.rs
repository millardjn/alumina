//! Contains traits for ease of constructing new elementwise operations which produce two outputs.
//!
//! Using the unary (single input) family as an example:
//!  * UnaryDualFunc is the trait which is unique to each implemented Op, defining the forward operation, and any
//!    relevant gradients.
//!  * UnaryElementwiseInstance<T: UnaryDualFunc> is the generic OpInstance
//!  * UnaryElementwise<T: UnaryDualFunc> implements the
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

pub trait NullaryDualFunc: Send + Sync + Clone + fmt::Debug + 'static {
	fn calc(&self) -> (f32, f32);

	fn type_name(&self) -> &'static str;
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct NullaryElementwiseDual<F: NullaryDualFunc> {
	output1: Node,
	output2: Node,
	f: F,
}

impl<F: NullaryDualFunc> NullaryElementwiseDual<F> {
	pub fn new<O1, O2>(output1: O1, output2: O2, f: F) -> Self
	where
		O1: Into<Node>,
		O2: Into<Node>,
	{
		let output1 = output1.into();
		let output2 = output2.into();
		NullaryElementwiseDual { output1, output2, f }
	}

	pub fn new_default<O1, O2>(output1: O1, output2: O2) -> Self
	where
		O1: Into<Node>,
		O2: Into<Node>,
		F: Default,
	{
		Self::new(output1, output2, F::default())
	}
}

impl<F: NullaryDualFunc> OpSpecification for NullaryElementwiseDual<F> {
	type InstanceType = NullaryElementwiseDualInstance<F>;

	fn type_name(&self) -> &'static str {
		self.f.type_name()
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node> {
		indexset![]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output1.clone(), self.output2.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			output1: mapping.get(&self.output1).unwrap_or(&self.output1).clone(),
			output2: mapping.get(&self.output2).unwrap_or(&self.output2).clone(),
			f: self.f.clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(NullaryElementwiseDualInstance {
			output1: self.output1.id(),
			output2: self.output2.id(),
			f: self.f,
		})
	}
}

/// ElementwiseDual Op, the value of the input is added to
#[derive(Clone, Debug)]
pub struct NullaryElementwiseDualInstance<F: NullaryDualFunc> {
	output1: NodeID,
	output2: NodeID,
	f: F,
}

impl<F: NullaryDualFunc> OpInstance for NullaryElementwiseDualInstance<F> {
	fn type_name(&self) -> &'static str {
		self.f.type_name()
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(NullaryElementwiseDual {
			output1: graph.node_from_id(self.output1),
			output2: graph.node_from_id(self.output2),
			f: self.f.clone(),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output1, self.output2]
	}

	fn gradient(&self, _ctx: &mut GradientContext) -> Result<(), GradientError> {
		Ok(())
	}

	fn propagate_shapes(&self, _ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		Ok(())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		if self.output1 == self.output2 {
			Zip::from(ctx.get_output(&self.output1)).par_apply(|output1| {
				let (o1, o2) = self.f.calc();
				*output1 += o1 + o2;
			});
		} else {
			Zip::from(ctx.get_output(&self.output1))
				.and(ctx.get_output(&self.output2))
				.par_apply(|output1, output2| {
					let (o1, o2) = self.f.calc();
					*output1 += o1;
					*output2 += o2;
				});
		}

		Ok(())
	}
}

pub trait UnaryDualFunc: Send + Sync + Clone + fmt::Debug + 'static {
	fn calc(&self, input: f32) -> (f32, f32);

	fn type_name(&self) -> &'static str;

	fn grad(
		&self,
		ctx: &mut GradientContext,
		input: &NodeID,
		output1: &NodeID,
		output2: &NodeID,
	) -> Result<(), GradientError>;
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct UnaryElementwiseDual<F: UnaryDualFunc> {
	output1: Node,
	output2: Node,
	input: Node,
	f: F,
}

impl<F: UnaryDualFunc> UnaryElementwiseDual<F> {
	pub fn new<I, O1, O2>(input: I, output1: O1, output2: O2, f: F) -> Self
	where
		I: Into<Node>,
		O1: Into<Node>,
		O2: Into<Node>,
	{
		let input = input.into();
		let output1 = output1.into();
		let output2 = output2.into();
		UnaryElementwiseDual {
			output1,
			output2,
			input,
			f,
		}
	}

	pub fn new_default<I, O1, O2>(input: I, output1: O1, output2: O2) -> Self
	where
		I: Into<Node>,
		O1: Into<Node>,
		O2: Into<Node>,
		F: Default,
	{
		Self::new(input, output1, output2, F::default())
	}
}

impl<F: UnaryDualFunc> OpSpecification for UnaryElementwiseDual<F> {
	type InstanceType = UnaryElementwiseDualInstance<F>;

	fn type_name(&self) -> &'static str {
		self.f.type_name()
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input.clone()]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output1.clone(), self.output2.clone()]
	}
	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			output1: mapping.get(&self.output1).unwrap_or(&self.output1).clone(),
			output2: mapping.get(&self.output2).unwrap_or(&self.output2).clone(),
			input: mapping.get(&self.output2).unwrap_or(&self.input).clone(),
			f: self.f.clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(UnaryElementwiseDualInstance {
			input: self.input.id(),
			output1: self.output1.id(),
			output2: self.output2.id(),
			f: self.f,
		})
	}
}

/// ElementwiseDual Op, the value of the input is added to
#[derive(Clone, Debug)]
pub struct UnaryElementwiseDualInstance<F: UnaryDualFunc> {
	input: NodeID,
	output1: NodeID,
	output2: NodeID,
	f: F,
}

impl<F: UnaryDualFunc> OpInstance for UnaryElementwiseDualInstance<F> {
	fn type_name(&self) -> &'static str {
		self.f.type_name()
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(UnaryElementwiseDual {
			input: graph.node_from_id(self.input),
			output1: graph.node_from_id(self.output1),
			output2: graph.node_from_id(self.output2),
			f: self.f.clone(),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output1, self.output2]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		self.f.grad(ctx, &self.input, &self.output1, &self.output2)
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let input_shape: NodeShape = ctx.input_shape(&self.input).slice().iter().into();
		ctx.merge_output_shape(&self.output1, &input_shape)?;
		ctx.merge_output_shape(&self.output2, &input_shape)
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		assert_eq!(
			ctx.shape(&self.input),
			ctx.shape(&self.output1),
			"Alumina Bug: input shape: {:?} did not match output1 shape: {:?}",
			ctx.shape(&self.input),
			ctx.shape(&self.output1)
		);

		assert_eq!(
			ctx.shape(&self.input),
			ctx.shape(&self.output2),
			"Alumina Bug: input shape: {:?} did not match output2 shape: {:?}",
			ctx.shape(&self.input),
			ctx.shape(&self.output2)
		);

		if self.output1 == self.output2 {
			if ctx.can_take(&self.input) && ctx.can_set(&self.output1) {
				// if output can be set using the input array, update inplace and do that.
				let mut input = ctx.take(&self.input); // input.par_map_inplace(|x| *x = self.f.calc(*x));

				Zip::from(&mut input).par_apply(|input| {
					let (o1, o2) = self.f.calc(*input);
					*input = o1 + o2;
				});
				ctx.set(&self.output1, input);
			} else {
				Zip::from(ctx.get_output(&self.output1))
					.and(ctx.get_input(&self.input))
					.par_apply(|output1, &input| {
						let (o1, o2) = self.f.calc(input);
						*output1 += o1 + o2;
					});
			}
		} else if ctx.can_take(&self.input) && ctx.can_set(&self.output1) {
			// if output can be set using the input array, update inplace and do that.
			let mut input = ctx.take(&self.input); // input.par_map_inplace(|x| *x = self.f.calc(*x));

			Zip::from(&mut input)
				.and(ctx.get_output(&self.output2))
				.par_apply(|input, output2| {
					let (o1, o2) = self.f.calc(*input);
					*input = o1;
					*output2 += o2;
				});
			ctx.set(&self.output1, input);
		} else if ctx.can_take(&self.input) && ctx.can_set(&self.output2) {
			// if output can be set using the input array, update inplace and do that.
			let mut input = ctx.take(&self.input); // input.par_map_inplace(|x| *x = self.f.calc(*x));

			Zip::from(&mut input)
				.and(ctx.get_output(&self.output1))
				.par_apply(|input, output1| {
					let (o1, o2) = self.f.calc(*input);
					*input = o2;
					*output1 += o1;
				});
			ctx.set(&self.output2, input);
		} else {
			Zip::from(ctx.get_output(&self.output1))
				.and(ctx.get_output(&self.output2))
				.and(ctx.get_input(&self.input))
				.par_apply(|output1, output2, &input| {
					let (o1, o2) = self.f.calc(input);
					*output1 += o1;
					*output2 += o2;
				});
		}

		Ok(())
	}
}

pub trait BinaryDualFunc: Send + Sync + Clone + fmt::Debug + 'static {
	fn calc(&self, input1: f32, input2: f32) -> (f32, f32);

	fn type_name(&self) -> &'static str;

	fn grad(
		&self,
		ctx: &mut GradientContext,
		input1: &NodeID,
		input2: &NodeID,
		output1: &NodeID,
		output2: &NodeID,
	) -> Result<(), GradientError>;
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct BinaryElementwiseDual<F: BinaryDualFunc> {
	output1: Node,
	output2: Node,
	input1: Node,
	input2: Node,
	f: F,
}

impl<F: BinaryDualFunc> BinaryElementwiseDual<F> {
	pub fn new<I1, I2, O1, O2>(input1: I1, input2: I2, output1: O1, output2: O2, f: F) -> Self
	where
		I1: Into<Node>,
		I2: Into<Node>,
		O1: Into<Node>,
		O2: Into<Node>,
	{
		let input1 = input1.into();
		let input2 = input2.into();
		let output1 = output1.into();
		let output2 = output2.into();
		BinaryElementwiseDual {
			output1,
			output2,
			input1,
			input2,
			f,
		}
	}

	pub fn new_default<I1, I2, O1, O2>(input1: I1, input2: I2, output1: O1, output2: O2) -> Self
	where
		I1: Into<Node>,
		I2: Into<Node>,
		O1: Into<Node>,
		O2: Into<Node>,
		F: Default,
	{
		Self::new(input1, input2, output1, output2, F::default())
	}
}

impl<F: BinaryDualFunc> OpSpecification for BinaryElementwiseDual<F> {
	type InstanceType = BinaryElementwiseDualInstance<F>;

	fn type_name(&self) -> &'static str {
		self.f.type_name()
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input1.clone(), self.input2.clone()]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output1.clone(), self.output2.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			output1: mapping.get(&self.output1).unwrap_or(&self.output1).clone(),
			output2: mapping.get(&self.output2).unwrap_or(&self.output2).clone(),
			input1: mapping.get(&self.output1).unwrap_or(&self.input1).clone(),
			input2: mapping.get(&self.output2).unwrap_or(&self.input2).clone(),
			f: self.f.clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(BinaryElementwiseDualInstance {
			input1: self.input1.id(),
			input2: self.input2.id(),
			output1: self.output1.id(),
			output2: self.output2.id(),
			f: self.f,
		})
	}
}

/// ElementwiseDual Op, the value of the input is added to
#[derive(Clone, Debug)]
pub struct BinaryElementwiseDualInstance<F: BinaryDualFunc> {
	input1: NodeID,
	input2: NodeID,
	output1: NodeID,
	output2: NodeID,
	f: F,
}

impl<F: BinaryDualFunc> OpInstance for BinaryElementwiseDualInstance<F> {
	fn type_name(&self) -> &'static str {
		self.f.type_name()
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(BinaryElementwiseDual {
			input1: graph.node_from_id(self.input1),
			input2: graph.node_from_id(self.input2),
			output1: graph.node_from_id(self.output1),
			output2: graph.node_from_id(self.output2),
			f: self.f.clone(),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input1, self.input2]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output1, self.output2]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		self.f
			.grad(ctx, &self.input1, &self.input2, &self.output1, &self.output2)
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let input_shape1: NodeShape = ctx.input_shape(&self.input1).slice().iter().into();
		let input_shape2: NodeShape = ctx.input_shape(&self.input2).slice().iter().into();
		let input_shape = input_shape1.merge(&input_shape2)?;
		ctx.merge_output_shape(&self.output1, &input_shape)?;
		ctx.merge_output_shape(&self.output2, &input_shape)
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		assert_eq!(
			ctx.shape(&self.input1),
			ctx.shape(&self.input2),
			"Alumina Bug: input1 shape: {:?} did not match input2 shape: {:?}",
			ctx.shape(&self.input1),
			ctx.shape(&self.input2)
		);

		assert_eq!(
			ctx.shape(&self.input1),
			ctx.shape(&self.output1),
			"Alumina Bug: input1 shape: {:?} did not match output1 shape: {:?}",
			ctx.shape(&self.input1),
			ctx.shape(&self.output1)
		);

		assert_eq!(
			ctx.shape(&self.input1),
			ctx.shape(&self.output2),
			"Alumina Bug: input1 shape: {:?} did not match output2 shape: {:?}",
			ctx.shape(&self.input1),
			ctx.shape(&self.output2)
		);

		// Are the inputs the same node?
		// Are the outputs the same node?
		// What are the combinations of can_take and can_set?
		//
		// This handles each output near optimally
		match (self.input1 == self.input2, self.output1 == self.output2) {
			(true, true) => {
				// one input and one output
				if ctx.can_take(&self.input1) && ctx.can_set(&self.output1) {
					let mut input1 = ctx.take(&self.input1);
					Zip::from(&mut input1).par_apply(|in1| {
						let (o1, o2) = self.f.calc(*in1, *in1);
						*in1 = o1 + o2;
					});
					ctx.set(&self.output1, input1);
				} else {
					Zip::from(ctx.get_input(&self.input1))
						.and(ctx.get_output(&self.output1))
						.par_apply(|in1, output| {
							let (o1, o2) = self.f.calc(*in1, *in1);
							*output += o1 + o2;
						});
				}
			}
			(true, false) => {
				// one input and two outputs
				if ctx.can_take(&self.input1) && ctx.can_set(&self.output1) {
					let mut input1 = ctx.take(&self.input1);
					Zip::from(&mut input1)
						.and(ctx.get_output(&self.output2))
						.par_apply(|in1, out2| {
							let (o1, o2) = self.f.calc(*in1, *in1);
							*in1 = o1;
							*out2 += o2;
						});
					ctx.set(&self.output1, input1);
				} else if ctx.can_take(&self.input1) && ctx.can_set(&self.output2) {
					let mut input1 = ctx.take(&self.input1);
					Zip::from(&mut input1)
						.and(ctx.get_output(&self.output1))
						.par_apply(|in1, out1| {
							let (o1, o2) = self.f.calc(*in1, *in1);
							*in1 = o2;
							*out1 += o1;
						});
					ctx.set(&self.output2, input1);
				} else {
					Zip::from(ctx.get_input(&self.input1))
						.and(ctx.get_output(&self.output1))
						.and(ctx.get_output(&self.output2))
						.par_apply(|in1, output1, output2| {
							let (o1, o2) = self.f.calc(*in1, *in1);
							*output1 += o1;
							*output2 += o2;
						});
				}
			}
			(false, true) => {
				// two inputs and one output
				if ctx.can_take(&self.input1) && ctx.can_set(&self.output1) {
					let mut input1 = ctx.take(&self.input1);
					Zip::from(&mut input1)
						.and(ctx.get_input(&self.input2))
						.par_apply(|in1, in2| {
							let (o1, o2) = self.f.calc(*in1, *in2);
							*in1 = o1 + o2;
						});
					ctx.set(&self.output1, input1);
				} else if ctx.can_take(&self.input2) && ctx.can_set(&self.output1) {
					let mut input2 = ctx.take(&self.input2);
					Zip::from(ctx.get_input(&self.input1))
						.and(&mut input2)
						.par_apply(|in1, in2| {
							let (o1, o2) = self.f.calc(*in1, *in2);
							*in2 = o1 + o2;
						});
					ctx.set(&self.output1, input2);
				} else {
					Zip::from(ctx.get_input(&self.input1))
						.and(ctx.get_input(&self.input2))
						.and(ctx.get_output(&self.output1))
						.par_apply(|in1, in2, out1| {
							let (o1, o2) = self.f.calc(*in1, *in2);
							*out1 += o1 + o2;
						});
				}
			}
			(false, false) => {
				// two inputs and two outputs
				if ctx.can_set(&self.output1)
					&& ctx.can_set(&self.output2)
					&& ctx.can_take(&self.input1)
					&& ctx.can_take(&self.input2)
				{
					let mut input1 = ctx.take(&self.input1);
					let mut input2 = ctx.take(&self.input2);
					Zip::from(&mut input1).and(&mut input2).par_apply(|in1, in2| {
						let (o1, o2) = self.f.calc(*in1, *in2);
						*in1 = o1;
						*in2 = o2;
					});
					ctx.set(&self.output1, input1);
					ctx.set(&self.output2, input2);
				} else if ctx.can_set(&self.output1) && ctx.can_take(&self.input1) {
					let mut input1 = ctx.take(&self.input1);
					Zip::from(&mut input1)
						.and(ctx.get_input(&self.input2))
						.and(ctx.get_output(&self.output2))
						.par_apply(|in1, in2, out2| {
							let (o1, o2) = self.f.calc(*in1, *in2);
							*in1 = o1;
							*out2 += o2;
						});
					ctx.set(&self.output1, input1);
				} else if ctx.can_set(&self.output1) && ctx.can_take(&self.input2) {
					let mut input2 = ctx.take(&self.input2);
					Zip::from(ctx.get_input(&self.input1))
						.and(&mut input2)
						.and(ctx.get_output(&self.output2))
						.par_apply(|in1, in2, out2| {
							let (o1, o2) = self.f.calc(*in1, *in2);
							*in2 = o1;
							*out2 += o2;
						});
					ctx.set(&self.output1, input2);
				} else if ctx.can_set(&self.output2) && ctx.can_take(&self.input1) {
					let mut input1 = ctx.take(&self.input1);
					Zip::from(&mut input1)
						.and(ctx.get_input(&self.input2))
						.and(ctx.get_output(&self.output1))
						.par_apply(|in1, in2, out1| {
							let (o1, o2) = self.f.calc(*in1, *in2);
							*in1 = o2;
							*out1 += o1;
						});
					ctx.set(&self.output2, input1);
				} else if ctx.can_set(&self.output2) && ctx.can_take(&self.input2) {
					let mut input2 = ctx.take(&self.input2);
					Zip::from(ctx.get_input(&self.input1))
						.and(&mut input2)
						.and(ctx.get_output(&self.output1))
						.par_apply(|in1, in2, out1| {
							let (o1, o2) = self.f.calc(*in1, *in2);
							*in2 = o2;
							*out1 += o1;
						});
					ctx.set(&self.output2, input2);
				} else {
					Zip::from(ctx.get_input(&self.input1))
						.and(ctx.get_input(&self.input2))
						.and(ctx.get_output(&self.output1))
						.and(ctx.get_output(&self.output2))
						.par_apply(|in1, in2, out1, out2| {
							let (o1, o2) = self.f.calc(*in1, *in2);
							*out1 += o1;
							*out2 += o2;
						});
				}
			}
		}

		Ok(())
	}
}

pub trait NaryDualFunc: Send + Sync + Clone + fmt::Debug + 'static {
	fn calc(&self, input: &[f32]) -> (f32, f32);

	fn type_name(&self) -> &'static str;

	fn grad(
		&self,
		ctx: &mut GradientContext,
		inputs: &[NodeID],
		output1: &NodeID,
		output2: &NodeID,
	) -> Result<(), GradientError>;
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct NaryElementwiseDual<F: NaryDualFunc> {
	output1: Node,
	output2: Node,
	inputs: Vec<Node>,
	f: F,
}

impl<F: NaryDualFunc> NaryElementwiseDual<F> {
	pub fn new<I, T: IntoIterator<Item = I>, O1, O2>(inputs: T, output1: O1, output2: O2, f: F) -> Self
	where
		I: Into<Node>,
		O1: Into<Node>,
		O2: Into<Node>,
	{
		let inputs: Vec<Node> = inputs.into_iter().map(Into::into).collect();
		let output1 = output1.into();
		let output2 = output2.into();
		assert!(
			inputs.len() <= 64,
			"Nary Ops can only be constructed with up to 64 inputs"
		);
		NaryElementwiseDual {
			output1,
			output2,
			inputs,
			f,
		}
	}

	pub fn new_default<I, T: IntoIterator<Item = I>, O1, O2>(inputs: T, output1: O1, output2: O2) -> Self
	where
		I: Into<Node>,
		O1: Into<Node>,
		O2: Into<Node>,
		F: Default,
	{
		Self::new(inputs, output1, output2, F::default())
	}
}

impl<F: NaryDualFunc> OpSpecification for NaryElementwiseDual<F> {
	type InstanceType = NaryElementwiseDualInstance<F>;

	fn type_name(&self) -> &'static str {
		self.f.type_name()
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node> {
		self.inputs.iter().cloned().collect()
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output1.clone(), self.output2.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			output1: mapping.get(&self.output1).unwrap_or(&self.output1).clone(),
			output2: mapping.get(&self.output2).unwrap_or(&self.output2).clone(),
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
		Ok(NaryElementwiseDualInstance {
			inputs: self.inputs.iter().map(|n| n.id()).collect(),
			output1: self.output1.id(),
			output2: self.output2.id(),
			f: self.f,
		})
	}
}

/// ElementwiseDual Op, the value of the input is added to
#[derive(Clone, Debug)]
pub struct NaryElementwiseDualInstance<F: NaryDualFunc> {
	inputs: Vec<NodeID>,
	output1: NodeID,
	output2: NodeID,
	f: F,
}

impl<F: NaryDualFunc> OpInstance for NaryElementwiseDualInstance<F> {
	fn type_name(&self) -> &'static str {
		self.f.type_name()
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(NaryElementwiseDual {
			inputs: self.inputs.iter().map(|&i| graph.node_from_id(i)).collect(),
			output1: graph.node_from_id(self.output1),
			output2: graph.node_from_id(self.output2),
			f: self.f.clone(),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		self.inputs.iter().cloned().collect()
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output1, self.output2]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		self.f.grad(ctx, &self.inputs, &self.output1, &self.output2)
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		if !self.inputs.is_empty() {
			let mut input_shape: NodeShape = ctx.input_shape(&self.inputs[0]).slice().iter().into();
			for input in &self.inputs[1..] {
				let next_shape: NodeShape = ctx.input_shape(input).slice().iter().into();
				input_shape = input_shape.merge(&next_shape)?;
			}
			ctx.merge_output_shape(&self.output1, &input_shape)?;
			ctx.merge_output_shape(&self.output2, &input_shape)
		} else {
			Ok(())
		}
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		assert_eq!(
			ctx.shape(&self.output1),
			ctx.shape(&self.output2),
			"Alumina Bug: output1 shape: {:?} did not match output2 shape: {:?}",
			ctx.shape(&self.output1),
			ctx.shape(&self.output2)
		);

		// No input shortcut
		if self.inputs.is_empty() {
			let mut output1 = ctx.get_output_standard(&self.output1);
			let mut output2 = ctx.get_output_standard(&self.output2);

			let len = output1.len();

			struct Ptrs {
				output1: *mut f32,
				output2: *mut f32,
			}
			unsafe impl Sync for Ptrs {}

			let ptrs = Ptrs {
				output1: output1.as_slice_mut().unwrap().as_mut_ptr(),
				output2: output2.as_slice_mut().unwrap().as_mut_ptr(),
			};

			(0..len).into_par_iter().for_each(|j| unsafe {
				let arr = [0.0; 0];
				let (o1, o2) = self.f.calc(&arr[..]);
				*ptrs.output1.add(j) += o1;
				*ptrs.output2.add(j) += o2;
			});
			return Ok(());
		}

		for (i, input) in self.inputs.iter().enumerate() {
			assert_eq!(
				ctx.shape(input),
				ctx.shape(&self.output1),
				"Alumina Bug: input {} shape: {:?} did not match output1 shape: {:?}",
				i,
				ctx.shape(&input),
				ctx.shape(&self.output1)
			);
		}

		{
			// Non-inplace version
			let mut output1 = ctx.get_output_standard(&self.output1);
			let mut output2 = ctx.get_output_standard(&self.output2);

			let len = output1.len();

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
				output1: *mut f32,
				output2: *mut f32,
			}
			unsafe impl Sync for Ptrs {}

			let ptrs = Ptrs {
				inputs,
				output1: output1.as_slice_mut().unwrap().as_mut_ptr(),
				output2: output2.as_slice_mut().unwrap().as_mut_ptr(),
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
						let (o1, o2) = self.f.calc(&arr[0..ptrs.inputs.len()]);
						*ptrs.output1.add(j) += o1;
						*ptrs.output2.add(j) += o2;
					}
				});
			} else if ptrs.inputs.len() <= 64 {
				(0..len).into_par_iter().for_each(|j| {
					unsafe {
						let mut arr = [0.0; 64]; // hopefully this array gets optimised out
						for k in 0..ptrs.inputs.len() {
							*arr.get_unchecked_mut(k) = *ptrs.inputs.get_unchecked(k).offset(j as isize);
						}
						let (o1, o2) = self.f.calc(&arr[0..ptrs.inputs.len()]);
						*ptrs.output1.add(j) += o1;
						*ptrs.output2.add(j) += o2;
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
