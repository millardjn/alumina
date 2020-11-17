use crate::{
	elementwise::elementwise_single::{UnaryElementwise, UnaryFunc},
	elementwise::{mul::Mul, scale::scale},
};
use alumina_core::{
	base_ops::OpSpecification,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeID},
};

/// Returns the square (sqr) of the input.
///
/// The output node has the same shape as the input.
pub fn sqr<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("sqr({})", input));
	let _op = Sqr::new_default(input, output.clone()).build()?;
	Ok(output)
}

pub type Sqr = UnaryElementwise<SqrFunc>;

#[derive(Clone, Debug, Default)]
pub struct SqrFunc {}

impl UnaryFunc for SqrFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		input * input
	}

	fn type_name(&self) -> &'static str {
		"Sqr"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeID, output: &NodeID) -> Result<(), GradientError> {
		let _op = Mul::new_default(ctx.grad_of(output), scale(ctx.node(input), 2.0)?, ctx.grad_of(input)).build()?;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::sqr;
	use alumina_core::{graph::Node, init::uniform};
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = sqr(&input).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.5625), ::std::f32::EPSILON));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.64), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[37, 33]).set_name("input").set_init(uniform(-2.0, 2.0));
		let output = sqr(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).tolerance(1e-3).run();
	}
}
