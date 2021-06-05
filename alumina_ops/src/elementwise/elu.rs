use crate::elementwise::elementwise_single::{BinaryElementwise, BinaryFunc, UnaryElementwise, UnaryFunc};
use alumina_core::{
	base_ops::OpSpecification,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeID},
};

/// Returns the exponential linear unit activation (elu) of the input.
///
/// The output node has the same shape as the input.
pub fn elu<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape())
		.set_name_unique(&format!("elu({})", input));
	let _op = ELU::new_default(input, output.clone()).build()?;
	Ok(output)
}

pub type ELU = UnaryElementwise<ELUFunc>;

pub type ELUBack = BinaryElementwise<ELUBackFunc>;

#[derive(Clone, Debug, Default)]
pub struct ELUFunc {}

impl UnaryFunc for ELUFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		if input >= 0.0 {
			input
		} else {
			input.exp() - 1.0
		}
	}

	fn type_name(&self) -> &'static str {
		"ELU"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeID, output: &NodeID) -> Result<(), GradientError> {
		ELUBack::new_default(&ctx.node(input), &ctx.grad_of(output), &ctx.grad_of(input)).build()?;
		Ok(())
	}
}

/// input1 = input of elu
/// input2 = grad of output of elu
#[derive(Clone, Debug, Default)]
pub struct ELUBackFunc {}

impl BinaryFunc for ELUBackFunc {
	#[inline]
	fn calc(&self, input1: f32, input2: f32) -> f32 {
		if input1 >= 0.0 {
			input2
		} else {
			input2 * input1.exp()
		}
	}

	fn type_name(&self) -> &'static str {
		"ELUBackward"
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
	use super::elu;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = elu(&input).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.25), ::std::f32::EPSILON));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(-0.550_671_04), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[13, 33]).set_name("input");
		let output = elu(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}
}
