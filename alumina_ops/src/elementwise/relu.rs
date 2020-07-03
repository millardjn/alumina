use crate::elementwise::elementwise_single::{BinaryElementwise, BinaryFunc, UnaryElementwise, UnaryFunc};
use alumina_core::{
	base_ops::OpBuilder,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeInner},
};

/// Returns the rectified linear unit activation (relu) of the input.
///
/// The output node has the same shape as the input.
pub fn relu<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("relu({})", input));
	let _op = Relu::new_default(input, output.clone()).build()?;
	Ok(output)
}

pub type Relu = UnaryElementwise<ReluFunc>;

pub type ReluBack = BinaryElementwise<ReluBackFunc>;

#[derive(Clone, Debug, Default)]
pub struct ReluFunc {}

impl UnaryFunc for ReluFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		// input.max(0.0)
		// if input > 0.0 {
		// 	input
		// } else {
		// 	0.0
		// }
		(input.abs() + input) * 0.5 // vectorises, but pretty questionable
	}

	fn type_name(&self) -> &'static str {
		"Relu"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeInner, output: &NodeInner) -> Result<(), GradientError> {
		ReluBack::new_default(ctx.node(input), ctx.grad_of(output), ctx.grad_of(input)).build()?;
		Ok(())
	}
}

/// input1 = input of relu
/// input2 = grad of output of relu
#[derive(Clone, Debug, Default)]
pub struct ReluBackFunc {}

impl BinaryFunc for ReluBackFunc {
	#[inline]
	fn calc(&self, input1: f32, input2: f32) -> f32 {
		let sign = input1.signum();
		input2 * (sign + sign.abs()) * 0.5 // x.signum().max(0.0); <- this should be better but doesnt compile to maxps,
	}

	fn type_name(&self) -> &'static str {
		"ReluBackward"
	}

	fn grad(
		&self,
		_ctx: &mut GradientContext,
		_input1: &NodeInner,
		_input2: &NodeInner,
		_output: &NodeInner,
	) -> Result<(), GradientError> {
		Err(GradientError::Unimplemented)
	}
}

#[cfg(test)]
mod tests {
	use super::relu;
	use alumina_core::{graph::Node, init::uniform};
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = relu(&input).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.25), ::std::f32::EPSILON));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.0), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[37, 33]).set_name("input").set_init(uniform(-2.0, 2.0));
		let output = relu(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).tolerance(2e-3).run();
	}
}
