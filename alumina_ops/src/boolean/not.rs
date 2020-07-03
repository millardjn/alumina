use crate::elementwise::elementwise_single::{UnaryElementwise, UnaryFunc};
use alumina_core::{
	base_ops::OpBuilder,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeInner},
};

/// Calculates the elementwise negation of the input, returning 1.0 if it is 0.0 and 0.0 otherwise.
///
/// The output node has the same shape as the input.
pub fn not<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();

	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("not({})", input));
	let _op = Not::new_default(input, output.clone())
		.build()
		.expect("Error building Not Op");
	Ok(output)
}

pub type Not = UnaryElementwise<NotFunc>;

#[derive(Clone, Debug, Default)]
pub struct NotFunc {}

impl UnaryFunc for NotFunc {
	#[inline]
	fn calc(&self, input1: f32) -> f32 {
		#[allow(clippy::float_cmp)]
		{
			if input1 == 0.0 {
				1.0
			} else {
				0.0
			}
		}
	}

	fn type_name(&self) -> &'static str {
		"Not"
	}

	fn grad(&self, _ctx: &mut GradientContext, _input: &NodeInner, _output: &NodeInner) -> Result<(), GradientError> {
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::not;
	use alumina_core::graph::Node;
	use alumina_test::relatively_close::RelClose;

	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input1 = Node::new(&[13, 33]).set_name("input1");

		let output = not(&input1).unwrap();

		input1.set_value(arr0(0.0));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.0), ::std::f32::EPSILON));

		input1.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.0), ::std::f32::EPSILON));

		input1.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.0), ::std::f32::EPSILON));
	}
}
