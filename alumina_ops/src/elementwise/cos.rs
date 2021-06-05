use crate::elementwise::elementwise_single::{BinaryElementwise, BinaryFunc, UnaryElementwise, UnaryFunc};
use alumina_core::{
	base_ops::OpSpecification,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeID},
};

/// Returns the cosine (cos) of the input.
///
/// The output node has the same shape as the input.
pub fn cos<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape())
		.set_name_unique(&format!("cos({})", input));
	let _op = Cos::new_default(input, output.clone()).build()?;
	Ok(output)
}

pub type Cos = UnaryElementwise<CosFunc>;

pub type CosBack = BinaryElementwise<CosBackFunc>;

#[derive(Clone, Debug, Default)]
pub struct CosFunc {}

impl UnaryFunc for CosFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		input.cos()
	}

	fn type_name(&self) -> &'static str {
		"Cos"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeID, output: &NodeID) -> Result<(), GradientError> {
		CosBack::new_default(ctx.node(input), ctx.grad_of(output), ctx.grad_of(input)).build()?;
		Ok(())
	}
}

/// input1 = input of cos
/// input2 = grad of output of cos
#[derive(Clone, Debug, Default)]
pub struct CosBackFunc {}

impl BinaryFunc for CosBackFunc {
	#[inline]
	fn calc(&self, input1: f32, input2: f32) -> f32 {
		-(input2 * input1.sin())
	}

	fn type_name(&self) -> &'static str {
		"CosBackward"
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
	use super::cos;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = cos(&input).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.315_322_37), ::std::f32::EPSILON));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.696_706_7), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[13, 33]).set_name("input");
		let output = cos(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}
}
