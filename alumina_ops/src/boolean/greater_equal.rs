use crate::elementwise::elementwise_single::{BinaryElementwise, BinaryFunc};
use alumina_core::{
	base_ops::OpBuilder,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{merge_graphs, Node, NodeInner},
};

/// Returns the truth value (1.0 or 0.0) of (input1 >= input2) element-wise.
///
/// No automatic broadcasting. Inputs must be the same shape.
///
/// The output node has the same shape as the input.
pub fn greater_equal<I1, I2>(input1: I1, input2: I2) -> Result<Node, OpBuildError>
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
		.set_name_unique(&format!("greater_equal({},{})", input1, input2));
	let _op = GreaterEqual::new_default(input1, input2, output.clone())
		.build()
		.expect("Error building GreaterEqual Op");
	Ok(output)
}

pub type GreaterEqual = BinaryElementwise<GreaterEqualFunc>;

#[derive(Clone, Debug, Default)]
pub struct GreaterEqualFunc {}

impl BinaryFunc for GreaterEqualFunc {
	#[inline]
	fn calc(&self, input1: f32, input2: f32) -> f32 {
		{
			if input1 >= input2 {
				1.0
			} else {
				0.0
			}
		}
	}

	fn type_name(&self) -> &'static str {
		"GreaterEqual"
	}

	fn grad(
		&self,
		_ctx: &mut GradientContext,
		_input1: &NodeInner,
		_input2: &NodeInner,
		_output: &NodeInner,
	) -> Result<(), GradientError> {
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::greater_equal;
	use alumina_core::graph::Node;
	use alumina_test::relatively_close::RelClose;

	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input1 = Node::new(&[13, 33]).set_name("input1");
		let input2 = Node::new(&[13, 33]).set_name("input2");

		let output = greater_equal(&input1, &input2).unwrap();

		input1.set_value(arr0(1.25));
		input2.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.0), ::std::f32::EPSILON));

		input1.set_value(arr0(1.25));
		input2.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.0), ::std::f32::EPSILON));

		input1.set_value(arr0(-0.8));
		input2.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(0.0), ::std::f32::EPSILON));

		input1.set_value(arr0(-0.8));
		input2.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.0), ::std::f32::EPSILON));
	}
}
