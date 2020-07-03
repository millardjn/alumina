use crate::elementwise::elementwise_single::{UnaryElementwise, UnaryFunc};
use alumina_core::{
	base_ops::OpBuilder,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeInner},
};

/// Returns the sign of the input.
///
/// The output node has the same shape as the input.
pub fn sign<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("sign({})", input));
	let _op = Sign::new_default(input, output.clone()).build()?;
	Ok(output)
}

pub type Sign = UnaryElementwise<SignFunc>;

#[derive(Clone, Debug, Default)]
pub struct SignFunc {}

impl UnaryFunc for SignFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		if input.is_nan() || input == 0.0 {
			0.0
		} else {
			input.signum()
		}
	}

	fn type_name(&self) -> &'static str {
		"Sign"
	}

	fn grad(&self, _ctx: &mut GradientContext, _input: &NodeInner, _output: &NodeInner) -> Result<(), GradientError> {
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::sign;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = sign(&input).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.0), ::std::f32::EPSILON));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(-1.0), ::std::f32::EPSILON));

		input.set_value(arr0(-0.0));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.0), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[13, 33]).set_name("input");
		let output = sign(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input])
			.expect_zero(&input, ::std::f32::EPSILON)
			.run();
	}
}
