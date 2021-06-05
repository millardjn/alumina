use crate::elementwise::elementwise_single::{BinaryElementwise, BinaryFunc, UnaryElementwise, UnaryFunc};
use alumina_core::{
	base_ops::OpSpecification,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeID},
};

/// Returns the hyperbolic tangent (tanh) of the input.
///
/// The output node has the same shape as the input.
pub fn tanh<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape())
		.set_name_unique(&format!("tanh({})", input));
	let _op = Tanh::new_default(input, output.clone()).build()?;
	Ok(output)
}

pub type Tanh = UnaryElementwise<TanhFunc>;

pub type TanhBack = BinaryElementwise<TanhBackFunc>;

#[derive(Clone, Debug, Default)]
pub struct TanhFunc {}

impl UnaryFunc for TanhFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		input.tanh()
	}

	fn type_name(&self) -> &'static str {
		"Tanh"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeID, output: &NodeID) -> Result<(), GradientError> {
		TanhBack::new_default(ctx.node(input), ctx.grad_of(output), ctx.grad_of(input)).build()?;
		Ok(())
	}
}

/// input1 = input of tanh
/// input2 = grad of output of tanh
#[derive(Clone, Debug, Default)]
pub struct TanhBackFunc {}

impl BinaryFunc for TanhBackFunc {
	#[inline]
	fn calc(&self, input1: f32, input2: f32) -> f32 {
		let s = input1.cosh();
		input2 / (s * s)
	}

	fn type_name(&self) -> &'static str {
		"TanhBackward"
	}

	fn grad(
		&self,
		_ctx: &mut GradientContext,
		_input1: &NodeID,
		_input2: &NodeID,
		_output: &NodeID,
	) -> Result<(), GradientError> {
		Err(GradientError::Unimplemented)
	}
}

#[cfg(test)]
mod tests {
	use super::tanh;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = tanh(&input).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.848_283_65), ::std::f32::EPSILON));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(-0.664_036_75), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[13, 33]).set_name("input");
		let output = tanh(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}
}
