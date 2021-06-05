use crate::{
	elementwise::elementwise_single::{BinaryElementwise, BinaryFunc},
	elementwise::identity::Identity,
	elementwise::negative::Negative,
};
use alumina_core::{
	base_ops::OpSpecification,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{merge_graphs, Node, NodeID},
};

/// Calculates the elementwise subtractision (subtract) of input1 over input2.
///
/// The output node has the same shape as the inputs.
pub fn subtract<I1, I2>(input1: I1, input2: I2) -> Result<Node, OpBuildError>
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	let input1 = input1.into();
	let input2 = input2.into();
	merge_graphs(&[input1.graph(), input2.graph()]);
	let output = input1
		.graph()
		.new_node(input1.shape())
		.set_name_unique(&format!("subtract({},{})", input1, input2));
	let _op = Subtract::new_default(input1, input2, output.clone()).build()?;
	Ok(output)
}

pub type Subtract = BinaryElementwise<SubtractFunc>;

#[derive(Clone, Debug, Default)]
pub struct SubtractFunc {}

impl BinaryFunc for SubtractFunc {
	#[inline]
	fn calc(&self, input1: f32, input2: f32) -> f32 {
		input1 - input2
	}

	fn type_name(&self) -> &'static str {
		"Subtract"
	}

	fn grad(
		&self,
		ctx: &mut GradientContext,
		input1: &NodeID,
		input2: &NodeID,
		output: &NodeID,
	) -> Result<(), GradientError> {
		let _op = Identity::new_default(&ctx.grad_of(output), &ctx.grad_of(input1)).build()?;
		let _op = Negative::new_default(&ctx.grad_of(output), &ctx.grad_of(input2)).build()?;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::subtract;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input1 = Node::new(&[13, 33]).set_name("input1");
		let input2 = Node::new(&[13, 33]).set_name("input2");

		let output = subtract(&input1, &input2).unwrap();

		input1.set_value(arr0(1.25));
		input2.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.0), ::std::f32::EPSILON));

		input1.set_value(arr0(1.25));
		input2.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(2.05), ::std::f32::EPSILON));

		input1.set_value(arr0(-0.8));
		input2.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(-2.05), ::std::f32::EPSILON));

		input1.set_value(arr0(-0.8));
		input2.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.0), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input1 = Node::new(&[13, 33]).set_name("input1");
		let input2 = Node::new(&[13, 33]).set_name("input2");

		let output = subtract(&input1, &input2).unwrap();

		GradNumericTest::new(&output, &indexset![&input1, &input2])
			.step_size(1e-3)
			.tolerance(1e-3)
			.run();
	}

	#[test]
	fn grad_numeric_shared_input_test() {
		let input1 = Node::new(&[13, 33]).set_name("input1");

		let output = subtract(&input1, &input1).unwrap();

		GradNumericTest::new(&output, &indexset![&input1])
			.expect_zero(&input1, ::std::f32::EPSILON)
			.step_size(1e-3)
			.run();
	}
}
