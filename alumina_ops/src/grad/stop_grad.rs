use alumina_core::{
	base_ops::OpBuilder,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeID},
};

use crate::elementwise::elementwise_single::{UnaryElementwise, UnaryFunc};

/// Returns the same value (stop_grad) as the input.
///
/// The output node has the same shape as the input.
pub fn stop_grad<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("stop_grad({})", input));
	let _op = StopGrad::new_default(input, output.clone()).build()?;
	Ok(output)
}

pub type StopGrad = UnaryElementwise<StopGradFunc>;

#[derive(Clone, Debug, Default)]
pub struct StopGradFunc {}

impl UnaryFunc for StopGradFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		input
	}

	fn type_name(&self) -> &'static str {
		"StopGrad"
	}

	fn grad(&self, _ctx: &mut GradientContext, _input: &NodeID, _output: &NodeID) -> Result<(), GradientError> {
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::stop_grad;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};
	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = stop_grad(&input).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.25), ::std::f32::EPSILON));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(-0.8), ::std::f32::EPSILON));
	}

	#[test]
	#[should_panic]
	fn grad_numeric_test() {
		let input = Node::new(&[13, 33]).set_name("input");
		let output = stop_grad(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}
}
