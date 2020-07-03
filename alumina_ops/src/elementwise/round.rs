use crate::elementwise::elementwise_single::{UnaryElementwise, UnaryFunc};
use alumina_core::{
	base_ops::OpBuilder,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeInner},
};

/// Returns the closest integer (round) to the input.
///
/// The output node has the same shape as the input.
pub fn round<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("round({})", input));
	let _op = Round::new_default(input, output.clone()).build()?;
	Ok(output)
}

pub type Round = UnaryElementwise<RoundFunc>;

#[derive(Clone, Debug, Default)]
pub struct RoundFunc {}

impl UnaryFunc for RoundFunc {
	fn calc(&self, input: f32) -> f32 {
		input.round()
	}

	fn type_name(&self) -> &'static str {
		"Round"
	}

	fn grad(&self, _ctx: &mut GradientContext, _input: &NodeInner, _output: &NodeInner) -> Result<(), GradientError> {
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::round;
	use alumina_core::graph::Node;
	use alumina_test::relatively_close::RelClose;

	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = round(&input).unwrap();

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

		input.set_value(arr0(1.0));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.00), ::std::f32::EPSILON));
	}
}
