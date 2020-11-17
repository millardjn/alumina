use crate::{
	elementwise::elementwise_single::{UnaryElementwise, UnaryFunc},
	elementwise::identity::Identity,
};
use alumina_core::{
	base_ops::OpSpecification,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeID},
};

/// Returns the addition of the input and a fixed number.
///
/// The output node has the same shape as the input.
pub fn offset<I>(input: I, offset: f32) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("offset({})", input));
	let _op = Offset::new(input, output.clone(), OffsetFunc { offset }).build()?;
	Ok(output)
}

pub type Offset = UnaryElementwise<OffsetFunc>;

#[derive(Clone, Debug)]
pub struct OffsetFunc {
	offset: f32,
}

impl UnaryFunc for OffsetFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		input + self.offset
	}

	fn type_name(&self) -> &'static str {
		"Offset"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeID, output: &NodeID) -> Result<(), GradientError> {
		let _op = Identity::new_default(ctx.grad_of(output), ctx.grad_of(input)).build()?;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::offset;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = offset(&input, 2.5).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(3.75), ::std::f32::EPSILON));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.7), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[13, 33]).set_name("input");
		let output = offset(&input, 2.5).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}
}
