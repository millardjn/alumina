use crate::{
	elementwise::elementwise_single::{UnaryElementwise, UnaryFunc},
	elementwise::mul::Mul,
};
use alumina_core::{
	base_ops::OpSpecification,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeID},
};

/// Returns the natural exponent (exp) of the input.
///
/// The output node has the same shape as the input.
pub fn exp<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape())
		.set_name_unique(&format!("exp({})", input));
	let _op = Exp::new_default(input, output.clone()).build()?;
	Ok(output)
}

pub type Exp = UnaryElementwise<ExpFunc>;

#[derive(Clone, Debug, Default)]
pub struct ExpFunc {}

impl UnaryFunc for ExpFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		input.exp()
	}

	fn type_name(&self) -> &'static str {
		"Exp"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeID, output: &NodeID) -> Result<(), GradientError> {
		let _op = Mul::new_default(ctx.grad_of(output), exp(ctx.node(input))?, ctx.grad_of(input)).build()?;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::exp;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = exp(&input).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(3.490_342_9), ::std::f32::EPSILON));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.449_328_96), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[37, 33]).set_name("input");
		let output = exp(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).tolerance(1e-3).run();
	}
}
