use crate::elementwise::elementwise_single::{BinaryElementwise, BinaryFunc, UnaryElementwise, UnaryFunc};
use alumina_core::{
	base_ops::OpBuilder,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeInner},
};

/// Applies the logistic function to each element of the input.
///
/// The output node has the same shape as the input.
pub fn logistic<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("logistic({})", input));
	let _op = Logistic::new_default(input, output.clone())
		.build()
		.expect("Error building Logistic Op");
	Ok(output)
}

pub type Logistic = UnaryElementwise<LogisticFunc>;

pub type LogisticBack = BinaryElementwise<LogisticBackFunc>;

#[derive(Clone, Debug, Default)]
pub struct LogisticFunc {}

impl UnaryFunc for LogisticFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		1.0 / (1.0 + (-input).exp())
	}

	fn type_name(&self) -> &'static str {
		"Logistic"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeInner, output: &NodeInner) -> Result<(), GradientError> {
		LogisticBack::new_default(ctx.node(input), ctx.grad_of(output), ctx.grad_of(input)).build()?;
		Ok(())
	}
}

/// input1 = input of logistic
/// input2 = grad of output of logistic
#[derive(Clone, Debug, Default)]
pub struct LogisticBackFunc {}

impl BinaryFunc for LogisticBackFunc {
	#[inline]
	fn calc(&self, input1: f32, input2: f32) -> f32 {
		let exp = input1.exp();
		input2 * exp / ((exp + 1.0) * (exp + 1.0))
	}

	fn type_name(&self) -> &'static str {
		"LogisticBackward"
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
	use super::logistic;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = logistic(&input).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.777_299_9), ::std::f32::EPSILON));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.310_025_5), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[13, 33]).set_name("input");
		let output = logistic(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}
}
