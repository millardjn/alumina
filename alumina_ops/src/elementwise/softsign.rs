// y = x / (abs(x) + 1)
// y' = 1 / (abs(x) + 1)^2

use crate::{
	elementwise::elementwise_single::{UnaryElementwise, UnaryFunc},
	elementwise::{abs::abs, div::Div, offset::offset, sqr::sqr},
};
use alumina_core::{
	base_ops::OpBuilder,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeInner},
};

/// Returns the softsign (y = x / (x.abs() + 1.0)) of the input element-wise.
///
/// The output node has the same shape as the input.
pub fn softsign<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("softsign({})", input));
	let _op = Softsign::new_default(input, output.clone()).build()?;
	Ok(output)
}

pub type Softsign = UnaryElementwise<SoftsignFunc>;

#[derive(Clone, Debug, Default)]
pub struct SoftsignFunc {}

impl UnaryFunc for SoftsignFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		input / (input.abs() + 1.0)
	}

	fn type_name(&self) -> &'static str {
		"Softsign"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeInner, output: &NodeInner) -> Result<(), GradientError> {
		let abs = abs(ctx.node(input))?;
		let abs_p1 = offset(abs.clone(), 1.0)?;
		let sqr_abs_p1 = sqr(abs_p1)?;

		let _op = Div::new_default(ctx.grad_of(output), sqr_abs_p1, ctx.grad_of(input)).build()?;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::softsign;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = softsign(&input).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.555_555_6), ::std::f32::EPSILON));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(-0.444_444_45), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[13, 33]).set_name("input");
		let output = softsign(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}
}
