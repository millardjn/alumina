use crate::{
	elementwise::div::Div,
	elementwise::elementwise_single::{UnaryElementwise, UnaryFunc},
};
use alumina_core::{
	base_ops::OpSpecification,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeID},
};

/// Returns the natural logarithm (ln) of the input.
///
/// The output node has the same shape as the input.
pub fn ln<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape())
		.set_name_unique(&format!("ln({})", input));
	let _op = Ln::new_default(input, output.clone()).build()?;
	Ok(output)
}

pub type Ln = UnaryElementwise<LnFunc>;

#[derive(Clone, Debug, Default)]
pub struct LnFunc {}

impl UnaryFunc for LnFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		input.ln()
	}

	fn type_name(&self) -> &'static str {
		"Ln"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeID, output: &NodeID) -> Result<(), GradientError> {
		let _op = Div::new_default(ctx.grad_of(output), ctx.node(input), ctx.grad_of(input)).build()?;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::ln;
	use alumina_core::{graph::Node, init::uniform};
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = ln(&input).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.223_143_55), ::std::f32::EPSILON));

		input.set_value(arr0(0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(-0.223_143_55), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[37, 33]).set_name("input").set_init(uniform(0.1, 5.0));
		let output = ln(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).tolerance(1e-3).run();
	}
}
