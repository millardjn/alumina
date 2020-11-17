use crate::{
	elementwise::elementwise_single::{UnaryElementwise, UnaryFunc},
	elementwise::{mul::Mul, sign::sign},
};
use alumina_core::{
	base_ops::OpSpecification,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeID},
};

/// Returns the absolute (abs) of the input.
///
/// The output node has the same shape as the input.
pub fn abs<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("abs({})", input));
	let _op = Abs::new_default(input, output.clone()).build()?;
	Ok(output)
}

pub type Abs = UnaryElementwise<AbsFunc>;

#[derive(Clone, Debug, Default)]
pub struct AbsFunc {}

impl UnaryFunc for AbsFunc {
	fn calc(&self, input: f32) -> f32 {
		input.abs()
	}

	fn type_name(&self) -> &'static str {
		"Abs"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeID, output: &NodeID) -> Result<(), GradientError> {
		let _op = Mul::new_default(ctx.grad_of(output), sign(ctx.node(input))?, ctx.grad_of(input)).build()?;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::abs;
	use alumina_core::{graph::Node, init::uniform};
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = abs(&input).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.25), ::std::f32::EPSILON));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.8), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[37, 33]).set_name("input").set_init(uniform(-2.0, 2.0));
		let output = abs(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).tolerance(2e-3).run();
	}
}
