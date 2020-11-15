use crate::elementwise::elementwise_single::{UnaryElementwise, UnaryFunc};
use alumina_core::{
	base_ops::OpBuilder,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeID},
};

/// Returns the closest integer less than or equal to (floor) the input.
///
/// The output node has the same shape as the input.
pub fn floor<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("floor({})", input));
	let _op = Floor::new_default(input, output.clone()).build()?;
	Ok(output)
}

pub type Floor = UnaryElementwise<FloorFunc>;

#[derive(Clone, Debug, Default)]
pub struct FloorFunc {}

impl UnaryFunc for FloorFunc {
	fn calc(&self, input: f32) -> f32 {
		input.floor()
	}

	fn type_name(&self) -> &'static str {
		"Floor"
	}

	fn grad(&self, _ctx: &mut GradientContext, _input: &NodeID, _output: &NodeID) -> Result<(), GradientError> {
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::floor;
	use alumina_core::{graph::Node, init::uniform};
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = floor(&input).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.00), ::std::f32::EPSILON));

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

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[37, 33]).set_name("input").set_init(uniform(-2.0, 2.0));
		let output = floor(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input])
			.tolerance(2e-3)
			.expect_zero(&input, ::std::f32::EPSILON)
			.run();
	}
}
