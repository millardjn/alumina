use crate::elementwise::elementwise_single::{BinaryElementwise, BinaryFunc, UnaryElementwise, UnaryFunc};
use alumina_core::{
	base_ops::OpSpecification,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeID},
};

/// Returns the leaky rectified linear unit activation (leaky relu) of the input.
///
/// The output node has the same shape as the input.
pub fn leaky_relu<I>(input: I, alpha: f32) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("leaky_relu({})", input));
	let _op = LeakyRelu::new(input, output.clone(), LeakyReluFunc { alpha }).build()?;
	Ok(output)
}

pub type LeakyRelu = UnaryElementwise<LeakyReluFunc>;

pub type LeakyReluBack = BinaryElementwise<LeakyReluBackFunc>;

#[derive(Clone, Debug)]
pub struct LeakyReluFunc {
	alpha: f32,
}

impl Default for LeakyReluFunc {
	fn default() -> Self {
		Self { alpha: 0.1 }
	}
}

impl UnaryFunc for LeakyReluFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		// input.max(0.0)
		// if input > 0.0 {
		// 	input
		// } else {
		// 	alpha * input
		// }

		let half_grad_change_at_zero = (1.0 - self.alpha) * 0.5;

		input.abs() * half_grad_change_at_zero + input * (1.0 - half_grad_change_at_zero)
		// TODO does this vectorize?
	}

	fn type_name(&self) -> &'static str {
		"LeakyRelu"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeID, output: &NodeID) -> Result<(), GradientError> {
		LeakyReluBack::new(
			ctx.node(input),
			ctx.grad_of(output),
			ctx.grad_of(input),
			LeakyReluBackFunc { alpha: self.alpha },
		)
		.build()?;
		Ok(())
	}
}

/// input1 = input of leaky_relu
/// input2 = grad of output of leaky_relu
#[derive(Clone, Debug)]
pub struct LeakyReluBackFunc {
	alpha: f32,
}

impl Default for LeakyReluBackFunc {
	fn default() -> Self {
		Self { alpha: 0.1 }
	}
}

impl BinaryFunc for LeakyReluBackFunc {
	#[inline]
	fn calc(&self, input1: f32, input2: f32) -> f32 {
		let sign = input1.signum();
		input2 * ((1.0 + self.alpha) + sign * (1.0 - self.alpha)) * 0.5 // x.signum().max(0.0); <- this should be better but doesnt compile to maxps,
	}

	fn type_name(&self) -> &'static str {
		"LeakyReluBackward"
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
	use super::leaky_relu;
	use alumina_core::{graph::Node, init::uniform};
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[13, 33]).set_name("input");

		let output = leaky_relu(&input, 0.2).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.25), ::std::f32::EPSILON));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(-0.16), ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[37, 33]).set_name("input").set_init(uniform(-2.0, 2.0));
		let output = leaky_relu(&input, 0.2).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).tolerance(1e-3).run();
	}
}
