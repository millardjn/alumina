use crate::elementwise::elementwise_single::{UnaryElementwise, UnaryFunc};
use alumina_core::{
	base_ops::OpBuilder,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeInner},
};

/// Returns the multiplication of the input and a fixed number.
///
/// The output node has the same shape as the input.
pub fn scale<I>(input: I, scale: f32) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("scale({})", input));
	let _op = Scale::new(input, output.clone(), ScaleFunc { scale }).build()?;
	Ok(output)
}

pub type Scale = UnaryElementwise<ScaleFunc>;

#[derive(Clone, Debug)]
pub struct ScaleFunc {
	scale: f32,
}

impl UnaryFunc for ScaleFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		input * self.scale
	}

	fn type_name(&self) -> &'static str {
		"Scale"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeInner, output: &NodeInner) -> Result<(), GradientError> {
		let _op = Scale::new(ctx.grad_of(output), ctx.grad_of(input), ScaleFunc { scale: self.scale }).build()?;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::scale;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = scale(&input, 2.5).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(3.125), ::std::f32::EPSILON));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(-2.0), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[13, 33]).set_name("input");
		let output = scale(&input, 2.5).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}
}
