// y = ln(exp(x)+1)
// y' = ln(exp(x)+1)

use crate::{
	elementwise::elementwise_single::{UnaryElementwise, UnaryFunc},
	elementwise::exp::exp,
	elementwise::{div::div, mul::Mul, offset::offset},
};
use alumina_core::{
	base_ops::OpSpecification,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeID},
};

/// Returns the softplus (y = (x.exp() + 1.0).ln()) of the input element-wise.
///
/// The output node has the same shape as the input.
pub fn softplus<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape())
		.set_name_unique(&format!("softplus({})", input));
	let _op = Softplus::new_default(input, output.clone()).build()?;
	Ok(output)
}

pub type Softplus = UnaryElementwise<SoftplusFunc>;

#[derive(Clone, Debug, Default)]
pub struct SoftplusFunc {}

impl UnaryFunc for SoftplusFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		(input.exp() + 1.0).ln()
	}

	fn type_name(&self) -> &'static str {
		"Softplus"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeID, output: &NodeID) -> Result<(), GradientError> {
		let exp = exp(ctx.node(input))?;
		let exp_p1 = offset(exp.clone(), 1.0)?;
		let do_di = div(exp, exp_p1)?;

		let _op = Mul::new_default(do_di, ctx.grad_of(output), ctx.grad_of(input)).build()?;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::softplus;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = softplus(&input).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.501_929), ::std::f32::EPSILON));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.371_100_66), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[13, 33]).set_name("input");
		let output = softplus(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}
}
