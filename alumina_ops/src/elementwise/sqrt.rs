use crate::{
	elementwise::div::Div,
	elementwise::elementwise_single::{UnaryElementwise, UnaryFunc},
	elementwise::scale::scale,
};
use alumina_core::{
	base_ops::OpSpecification,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeID},
};

/// Returns the square (sqrt) of the input.
///
/// The output node has the same shape as the input.
pub fn sqrt<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape())
		.set_name_unique(&format!("sqrt({})", input));
	let _op = Sqrt::new_default(&input, &output).build()?;
	Ok(output)
}

pub type Sqrt = UnaryElementwise<SqrtFunc>;

#[derive(Clone, Debug, Default)]
pub struct SqrtFunc {}

impl UnaryFunc for SqrtFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		input.sqrt()
	}

	fn type_name(&self) -> &'static str {
		"Sqrt"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeID, output: &NodeID) -> Result<(), GradientError> {
		let _op = Div::new_default(
			ctx.grad_of(output),
			scale(sqrt(ctx.node(input))?, 2.0)?,
			ctx.grad_of(input),
		)
		.build()?;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::sqrt;
	use alumina_core::{graph::Node, init::uniform};
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = sqrt(&input).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.118_034), ::std::f32::EPSILON));

		input.set_value(arr0(0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.894_427_2), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[37, 33]).set_name("input").set_init(uniform(0.2, 3.0));
		let output = sqrt(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).tolerance(1e-3).run();
	}
}
