use crate::elementwise::elementwise_single::{UnaryElementwise, UnaryFunc};
use alumina_core::{
	base_ops::OpBuilder,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeInner},
};

/// Returns the negative of each element of the input.
///
/// The output node has the same shape as the input.
pub fn negative<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("negative({})", input));
	let _op = Negative::new_default(input, output.clone()).build()?;
	Ok(output)
}

pub type Negative = UnaryElementwise<NegativeFunc>;

#[derive(Clone, Debug, Default)]
pub struct NegativeFunc {}

impl UnaryFunc for NegativeFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		-input
	}

	fn type_name(&self) -> &'static str {
		"Negative"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeInner, output: &NodeInner) -> Result<(), GradientError> {
		Negative::new_default(ctx.grad_of(output), ctx.grad_of(input)).build()?;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::negative;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = negative(&input).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(-1.25), ::std::f32::EPSILON));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.8), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[13, 33]).set_name("input");
		let output = negative(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}
}
