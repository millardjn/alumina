use crate::elementwise::elementwise_single::{BinaryElementwise, BinaryFunc};
use alumina_core::{
	base_ops::OpSpecification,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{merge_graphs, Node, NodeID},
};

/// Calculates the elementwise multiplication (mul) of input1 and input2.
///
/// The output node has the same shape as the inputs.
pub fn mul<I1, I2>(input1: I1, input2: I2) -> Result<Node, OpBuildError>
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	let input1 = input1.into();
	let input2 = input2.into();
	merge_graphs(&[input1.graph(), input2.graph()]);
	let output = input1
		.graph()
		.new_node(input1.shape().clone())
		.set_name_unique(&format!("mul({},{})", input1, input2));
	let _op = Mul::new_default(input1, input2, output.clone())
		.build()
		.expect("Error building Mul Op");
	Ok(output)
}

pub type Mul = BinaryElementwise<MulFunc>;

#[derive(Clone, Debug, Default)]
pub struct MulFunc {}

impl BinaryFunc for MulFunc {
	#[inline]
	fn calc(&self, input1: f32, input2: f32) -> f32 {
		input1 * input2
	}

	fn type_name(&self) -> &'static str {
		"Mul"
	}

	fn grad(
		&self,
		ctx: &mut GradientContext,
		input1: &NodeID,
		input2: &NodeID,
		output: &NodeID,
	) -> Result<(), GradientError> {
		let _op = Mul::new_default(ctx.grad_of(output), ctx.node(input2), ctx.grad_of(input1)).build()?;
		let _op = Mul::new_default(ctx.grad_of(output), ctx.node(input1), ctx.grad_of(input2)).build()?;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::mul;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input1 = Node::new(&[13, 33]).set_name("input1");
		let input2 = Node::new(&[13, 33]).set_name("input2");

		let output = mul(&input1, &input2).unwrap();

		input1.set_value(arr0(1.25));
		input2.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.5625), ::std::f32::EPSILON));

		input1.set_value(arr0(1.25));
		input2.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(-1.0), ::std::f32::EPSILON));

		input1.set_value(arr0(-0.8));
		input2.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(-1.0), ::std::f32::EPSILON));

		input1.set_value(arr0(-0.8));
		input2.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.64), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input1 = Node::new(&[13, 33]).set_name("input1");
		let input2 = Node::new(&[13, 33]).set_name("input2");

		let output = mul(&input1, &input2).unwrap();

		GradNumericTest::new(&output, &indexset![&input1, &input2]).run();
	}

	#[test]
	fn grad_numeric_shared_input_test() {
		let input1 = Node::new(&[13, 33]).set_name("input1");

		let output = mul(&input1, &input1).unwrap();

		GradNumericTest::new(&output, &indexset![&input1]).run();
	}
}
