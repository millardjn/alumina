use crate::{
	elementwise::div::Div,
	elementwise::elementwise_single::{UnaryElementwise, UnaryFunc},
	elementwise::{negative::negative, sqr::sqr},
};
use alumina_core::{
	base_ops::OpBuilder,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeInner},
};

/// Returns the reciprocal of the input.
///
/// The output node has the same shape as the input.
pub fn reciprocal<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("reciprocal({})", input));
	let _op = Reciprocal::new_default(input, output.clone()).build()?;
	Ok(output)
}

pub type Reciprocal = UnaryElementwise<ReciprocalFunc>;

#[derive(Clone, Debug, Default)]
pub struct ReciprocalFunc {}

impl UnaryFunc for ReciprocalFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		1.0 / input
	}

	fn type_name(&self) -> &'static str {
		"Reciprocal"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeInner, output: &NodeInner) -> Result<(), GradientError> {
		let _op = Div::new_default(
			ctx.grad_of(output),
			negative(sqr(ctx.node(input))?)?,
			ctx.grad_of(input),
		)
		.build()?;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::reciprocal;
	use alumina_core::{graph::Node, init::uniform};
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = reciprocal(&input).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.8), ::std::f32::EPSILON));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(-1.25), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[37, 33]).set_name("input").set_init(uniform(0.2, 3.0));
		let output = reciprocal(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).tolerance(1e-3).run();
	}
}
