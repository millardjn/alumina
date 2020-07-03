use crate::elementwise::elementwise_single::{UnaryElementwise, UnaryFunc};
use alumina_core::{
	base_ops::OpBuilder,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{merge_graphs, merge_node_graphs, Node, NodeInner},
	shape::SCALAR,
};

use smallvec::SmallVec;

/// Returns the same value (identity) as the input.
///
/// Inputs must have the same shape.
///
/// The output node has the same shape as the input.
pub fn identity<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("identity({})", input));
	let _op = Identity::new_default(input, output.clone()).build()?;
	Ok(output)
}

// Elementwise addition of the values in the inputs
pub fn add<I1, I2>(input1: I1, input2: I2) -> Result<Node, OpBuildError>
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	let input1 = input1.into();
	let input2 = input2.into();
	let graph = merge_graphs(&[input1.graph(), input2.graph()]);

	let output = graph
		.new_node(input1.shape().clone())
		.set_name_unique(&format!("add({},{})", input1, input2));
	let _op = Identity::new_default(input1, output.clone()).build()?;
	let _op = Identity::new_default(input2, output.clone()).build()?;

	Ok(output)
}

/// Elementwise addition of the values in the inputs.
///
/// Does not perform broadcasting.
///
/// If no inputs are supplied a scalar output without inputs is returned.
pub fn add_n<I, T>(inputs: T) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
	T: IntoIterator<Item = I>,
{
	let inputs: SmallVec<[Node; 8]> = inputs.into_iter().map(Into::into).collect();
	let output = if !inputs.is_empty() {
		let graph = merge_node_graphs(&inputs);

		let mut output_name = "add_n(".to_string();
		for (i, input) in inputs.iter().enumerate() {
			if i > 0 {
				output_name.push_str(", ");
			}
			output_name.push_str(&input.name());
		}
		output_name.push_str(")");

		let output = graph.new_node(inputs[0].shape().clone()).set_name_unique(&output_name);

		for input in inputs {
			let _op = Identity::new_default(input, output.clone()).build()?;
		}
		output
	} else {
		Node::new(SCALAR).set_name_unique("add_n()")
	};

	Ok(output)
}

pub type Identity = UnaryElementwise<IdentityFunc>;

#[derive(Clone, Debug, Default)]
pub struct IdentityFunc {}

impl UnaryFunc for IdentityFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		input
	}

	fn type_name(&self) -> &'static str {
		"Identity"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeInner, output: &NodeInner) -> Result<(), GradientError> {
		Identity::new_default(ctx.grad_of(output), ctx.grad_of(input)).build()?;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::identity;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = identity(&input).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.25), ::std::f32::EPSILON));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(-0.8), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[13, 33]).set_name("input");
		let output = identity(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}
}
