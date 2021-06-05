use crate::{
	elementwise::identity::{identity, Identity},
	manip::remove_dims::RemoveDims,
	reduce::reduce_sum::{reduce_sum, ReduceSum},
};
use alumina_core::{
	base_ops::{shape_constraint::same_shape, OpInstance, OpSpecification},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{merge_graphs, Graph, Node, NodeID, NodeTag, Op},
	init::duplicate,
	shape::{NodeAxis, NodeShape},
	shape_prop::ShapePropContext,
	util::wrap_dim,
};
use indexmap::{indexset, IndexMap, IndexSet};
use ndarray::{ArrayViewD, ArrayViewMutD, Dimension, Zip};
use smallvec::SmallVec;
use std::any::Any;
use std::iter::repeat;

/// broadcast the values of value_input to the shape of shape_input and return the result
pub fn broadcast<I1, I2>(shape_input: I1, value_input: I2) -> Result<Node, OpBuildError>
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	let shape_input = shape_input.into();
	let value_input = value_input.into();
	let graph = merge_graphs(&[shape_input.graph(), value_input.graph()]);

	let broadcast = graph
		.new_node(shape_input.shape().clone())
		.set_name_unique(&format!("broadcast({},{})", shape_input, value_input));

	let _op = same_shape(shape_input, broadcast.clone())?;
	let _op = Broadcast::new(value_input, broadcast.clone()).build()?;

	Ok(broadcast)
}

/// broadcast the values of input to the existing output and return the Op
pub fn broadcast_into<I, O>(input: I, output: O) -> Result<Op, OpBuildError>
where
	I: Into<Node>,
	O: Into<Node>,
{
	let input = input.into();
	let output = output.into();

	Broadcast::new(input, output).build()
}

/// broadcast input2 to the shape of input1, and call f with the inputs
pub fn broadcast_fn<F, I1, I2>(f: F, input1: I1, input2: I2) -> Result<Node, OpBuildError>
where
	F: FnOnce(Node, Node) -> Result<Node, OpBuildError>,
	I1: Into<Node>,
	I2: Into<Node>,
{
	let input1 = input1.into();
	let input2 = input2.into();
	let graph = merge_graphs(&[input1.graph(), input2.graph()]);

	let broadcast = graph
		.new_node(input1.shape().clone())
		.set_name_unique(&format!("broadcast({},{})", input1, input2));
	let _op = same_shape(input1.clone(), broadcast.clone())?;
	let _op = Broadcast::new(input2, broadcast.clone()).build()?;

	f(input1, broadcast)
}

/// broadcast input1 to the shape of input2, and call f with the inputs
pub fn broadcast_rev_fn<F, I1, I2>(f: F, input1: I1, input2: I2) -> Result<Node, OpBuildError>
where
	F: FnOnce(Node, Node) -> Result<Node, OpBuildError>,
	I1: Into<Node>,
	I2: Into<Node>,
{
	let input1 = input1.into();
	let input2 = input2.into();
	let graph = merge_graphs(&[input1.graph(), input2.graph()]);

	let broadcast = graph
		.new_node(input2.shape().clone())
		.set_name_unique(&format!("broadcast({},{})", input2, input1));
	let _op = same_shape(input2.clone(), broadcast.clone())?;
	let _op = Broadcast::new(input1, broadcast.clone()).build()?;

	f(broadcast, input2)
}

/// In-place Bias. Returns the input node after adding a broadcasted bias.
///
/// This function is odd in that it returns the input node rather than a newly created output.
///
/// `axes` determines which will have unique bias values and wont be broadcast. These axes must have a fixed shape or an
/// error is returned.
pub fn ibias<I>(input: I, axes: &[isize]) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();

	let mut bias_shape: NodeShape = repeat(NodeAxis::known(1)).take(input.shape().len()).into();

	for &axis in axes {
		let axis = wrap_dim(axis, input.shape().len());
		if !input.shape().slice()[axis].is_known() {
			return Err(format!("error in biased(..), axis {} of input ({}) shape {} did not have a fixed size and therefore cannot be used in the bias parameter. remove this axis from axes or ensure that size of the axis is fixed.", axis, input, input.shape()).into());
		}
		bias_shape.slice_mut()[axis] = input.shape().slice()[axis].clone();
	}

	let bias = input
		.graph()
		.new_node(bias_shape)
		.set_init(duplicate(0.0))
		.add_tag(NodeTag::Parameter)
		.set_name_unique(&format!("ibias({})_bias", input));

	let _op = Broadcast::new(bias, input.clone()).build()?;
	Ok(input)
}

/// Returns a new output node which is the addition of the input node and a broadcasted bias.
///
/// `axes` determines which will have unique bias values and wont be broadcast. These axes must have a fixed shape or an
/// error is returned.
pub fn bias<I>(input: I, axes: &[isize]) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = identity(&input)?.set_name_unique(&format!("bias({})", input));

	let mut bias_shape: NodeShape = repeat(NodeAxis::known(1)).take(input.shape().len()).into();

	for &axis in axes {
		let axis = wrap_dim(axis, input.shape().len());
		if !input.shape().slice()[axis].is_known() {
			return Err(format!("error in biased(..), axis {} of input ({}) shape {} did not have a fixed size and therefore cannot be used in the bias parameter. remove this axis from axes or ensure that size of the axis is fixed.", axis, input, input.shape()).into());
		}
		bias_shape.slice_mut()[axis] = input.shape().slice()[axis].clone();
	}

	let bias = input
		.graph()
		.new_node(bias_shape)
		.set_init(duplicate(0.0))
		.add_tag(NodeTag::Parameter)
		.set_name_unique(&format!("bias({})_bias", input));

	let _op = Broadcast::new(bias, output.clone()).build()?;
	Ok(output)
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct Broadcast {
	output: Node,
	input: Node,
}

impl Broadcast {
	pub fn new<I, O>(input: I, output: O) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		// TODO error if input cant broadcast to output
		Broadcast {
			input: input.into(),
			output: output.into(),
		}
	}
}

impl OpSpecification for Broadcast {
	type InstanceType = BroadcastInstance;

	fn type_name(&self) -> &'static str {
		"Broadcast"
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
			input: mapping.get(&self.input).unwrap_or(&self.input).clone(),
			output: mapping.get(&self.output).unwrap_or(&self.output).clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(BroadcastInstance {
			input: self.input.id().clone(),
			output: self.output.id().clone(),
		})
	}
}

/// Broadcast Op, the value of the input is added to
#[derive(Clone, Debug)]
pub struct BroadcastInstance {
	input: NodeID,
	output: NodeID,
}

impl OpInstance for BroadcastInstance {
	fn type_name(&self) -> &'static str {
		"Broadcast"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(Broadcast {
			input: graph.node_from_id(self.input),
			output: graph.node_from_id(self.output),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input.clone()]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output.clone()]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		let leading_ones = ctx.node(&self.output).shape().len() - ctx.node(&self.input).shape().len();

		if leading_ones == 0 {
			let broadcast_axes: SmallVec<[isize; 8]> = ctx
				.node(&self.input)
				.shape()
				.into_iter()
				.enumerate()
				.filter_map(|(i, na)| match na {
					NodeAxis::Known { val: 1 } => Some(i as isize),
					_ => None,
				})
				.collect();

			if broadcast_axes.is_empty() {
				Identity::new_default(ctx.grad_of(&self.output), ctx.grad_of(&self.input)).build()?;
			} else {
				ReduceSum::new(ctx.grad_of(&self.output), ctx.grad_of(&self.input))
					.keep_dims(true)
					.axes(broadcast_axes.as_slice())
					.build()?;
			}
		} else {
			let broadcast_axes: SmallVec<[isize; 8]> = ctx
				.node(&self.input)
				.shape()
				.into_iter()
				.enumerate()
				.filter_map(|(i, na)| match na {
					NodeAxis::Known { val: 1 } => Some((i + leading_ones) as isize),
					_ => None,
				})
				.chain(0..leading_ones as isize)
				.collect();

			let intermediate = if broadcast_axes.is_empty() {
				ctx.grad_of(&self.output)
			} else {
				reduce_sum(ctx.grad_of(&self.output), &broadcast_axes, true)?
			};

			// strip leading dims
			let leading_axes: SmallVec<[isize; 8]> = (0..leading_ones as isize).collect();

			RemoveDims::new(intermediate, ctx.grad_of(&self.input))
				.axes(&leading_axes)
				.build()?;
		}

		// context.
		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let input_shape: NodeShape = ctx.input_shape(&self.input).slice().iter().into();
		ctx.broadcast_merge_output_shape(&self.output, &input_shape)
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		if ctx.shape(&self.input) == ctx.shape(&self.output) && ctx.can_take(&self.input) && ctx.can_set(&self.output) {
			// if output can be set using the input array, do that.
			ctx.set(&self.output, ctx.take(&self.input));
		} else {
			let input: ArrayViewD<f32> = ctx.get_input(&self.input);
			let output: ArrayViewMutD<f32> = ctx.get_output(&self.output);

			Zip::from(output).and_broadcast(input).par_apply(|output, input| {
				*output += input;
			});

			// let input_broadcast = input.broadcast(output.shape()).ok_or_else(|| {
			// 	format!(
			// 		"Could not broadcast input({}) shape {:?} to output({}) shape {:?}",
			// 		&self.input,
			// 		input.shape(),
			// 		&self.output,
			// 		output.shape()
			// 	)
			// })?;

			// output += &input_broadcast;
		}

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::{bias, broadcast_fn, Broadcast};
	use crate::elementwise::identity;
	use alumina_core::{base_ops::OpSpecification, graph::Node};
	use alumina_test::grad_numeric_test::GradNumericTest;

	use ndarray::{arr0, ArrayD, IxDyn};

	#[test]
	fn forward_test() {
		let input1 = Node::new(&[13, 33]).set_value(arr0(1.25)).set_name("input1");
		let input2 = Node::new(&[33]).set_value(arr0(2.5)).set_name("input2");

		let output = broadcast_fn(identity::add, input1, input2).unwrap();

		let expected = ArrayD::from_elem(IxDyn(&[13, 33]), 3.75);
		assert_eq!(expected, output.calc().unwrap());
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[13, 33]).set_name("input");
		let output = Node::new(&[13, 33]).set_name("output");

		let _op = Broadcast::new(&input, &output).build().unwrap();

		GradNumericTest::new(output, &[input]).run();
	}

	#[test]
	fn grad_numeric_test_broadcast() {
		let input = Node::new(&[13, 1, 33]).set_name("input");
		let output = Node::new(&[5, 13, 7, 33]).set_name("output");

		let _op = Broadcast::new(&input, &output).build().unwrap();

		GradNumericTest::new(output, &[input]).run();
	}

	#[test]
	fn grad_numeric_bias_test() {
		let input = Node::new(&[13, 33]).set_name("input");
		let output = bias(&input, &[]).unwrap();

		GradNumericTest::new(output, &[input]).run();
	}
}
