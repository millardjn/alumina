use crate::elementwise::elementwise_single::{BinaryElementwise, BinaryFunc, UnaryElementwise, UnaryFunc};
use alumina_core::{
	base_ops::OpBuilder,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeID},
};

/// Converts the input from sRGB(0.0-1.0) to Linear RGB(0.0-1.0).
///
/// The output node has the same shape as the input.
pub fn srgb_to_linear<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("srgb_to_linear({})", input));
	let _op = SrgbToLinear::new_default(input, output.clone()).build()?;
	Ok(output)
}

/// Converts the input from Linear RGB(0.0-1.0) to sRGB(0.0-1.0).
///
/// The output node has the same shape as the input.
pub fn linear_to_srgb<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("linear_to_srgb({})", input));
	let _op = SrgbToLinear::new_default(input, output.clone()).build()?;
	Ok(output)
}

/// Converts the input from sRGB(0.0-1.0) to Linear RGB(0.0-1.0).
///
/// The output node has the same shape as the input.
pub fn srgb_to_linear_slow<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("srgb_to_linear({})", input));
	let _op = SrgbToLinearSlow::new_default(input, output.clone()).build()?;
	Ok(output)
}

/// Converts the input from Linear RGB(0.0-1.0) to sRGB(0.0-1.0).
///
/// The output node has the same shape as the input.
pub fn linear_to_srgb_slow<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("linear_to_srgb({})", input));
	let _op = SrgbToLinearSlow::new_default(input, output.clone()).build()?;
	Ok(output)
}

pub type SrgbToLinear = UnaryElementwise<SrgbToLinearFunc>;

pub type SrgbToLinearBack = BinaryElementwise<SrgbToLinearBackFunc>;

#[derive(Clone, Debug, Default)]
pub struct SrgbToLinearFunc {}

impl UnaryFunc for SrgbToLinearFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		if input <= 0.040_448_237 {
			input / 12.92
		} else {
			0.001_522_305 + 0.012_475_774 * input + 0.662_456_8 * input * input + 0.326_793_97 * input * input * input
		}
	}

	fn type_name(&self) -> &'static str {
		"SrgbToLinear"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeID, output: &NodeID) -> Result<(), GradientError> {
		SrgbToLinearBack::new_default(ctx.node(input), ctx.grad_of(output), ctx.grad_of(input)).build()?;
		Ok(())
	}
}

/// input1 = input of SrgbToLinear
/// input2 = grad of output of SrgbToLinear
#[derive(Clone, Debug, Default)]
pub struct SrgbToLinearBackFunc {}

impl BinaryFunc for SrgbToLinearBackFunc {
	#[inline]
	fn calc(&self, input1: f32, input2: f32) -> f32 {
		if input1 <= 0.040_448_237 {
			input2 / 12.92
		} else {
			input2 * (0.012_475_774 + (2.0 * 0.662_456_8) * input1 + (3.0 * 0.326_793_97) * input1 * input1)
		}
	}

	fn type_name(&self) -> &'static str {
		"SrgbToLinearBackward"
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

pub type LinearToSrgb = UnaryElementwise<LinearToSrgbFunc>;

pub type LinearToSrgbBack = BinaryElementwise<LinearToSrgbBackFunc>;

#[derive(Clone, Debug, Default)]
pub struct LinearToSrgbFunc {}

impl UnaryFunc for LinearToSrgbFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		if input <= 0.003_130_668_5 {
			input * 12.92
		} else {
			let s1 = input.sqrt();
			let s2 = s1.sqrt();
			-0.074_312_54 + 0.852_548_2 * s1 + 0.284_336_3 * s2 - 0.063_628_644 * input
		}
	}

	fn type_name(&self) -> &'static str {
		"LinearToSrgb"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeID, output: &NodeID) -> Result<(), GradientError> {
		LinearToSrgbBack::new_default(ctx.node(input), ctx.grad_of(output), ctx.grad_of(input)).build()?;
		Ok(())
	}
}

/// input1 = input of LinearToSrgb
/// input2 = grad of output of LinearToSrgb
#[derive(Clone, Debug, Default)]
pub struct LinearToSrgbBackFunc {}

impl BinaryFunc for LinearToSrgbBackFunc {
	#[inline]
	fn calc(&self, input1: f32, input2: f32) -> f32 {
		if input1 <= 0.003_130_668_5 {
			input2 * 12.92
		} else {
			let s1 = input1.sqrt();
			let s2 = s1.sqrt();
			input2 * ((0.5 * 0.852_548_2) / s1 + (0.25 * 0.284_336_3) / (s1 * s2) - 0.063_628_644)
		}
	}

	fn type_name(&self) -> &'static str {
		"LinearToSrgbBackward"
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

pub type SrgbToLinearSlow = UnaryElementwise<SrgbToLinearSlowFunc>;

pub type SrgbToLinearSlowBack = BinaryElementwise<SrgbToLinearSlowBackFunc>;

#[derive(Clone, Debug, Default)]
pub struct SrgbToLinearSlowFunc {}

impl UnaryFunc for SrgbToLinearSlowFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		if input <= 0.040_448_237 {
			input / 12.92
		} else {
			((input + 0.055) / 1.055).powf(2.4)
		}
	}

	fn type_name(&self) -> &'static str {
		"SrgbToLinearSlow"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeID, output: &NodeID) -> Result<(), GradientError> {
		SrgbToLinearSlowBack::new_default(ctx.node(input), ctx.grad_of(output), ctx.grad_of(input)).build()?;
		Ok(())
	}
}

/// input1 = input of SrgbToLinear
/// input2 = grad of output of SrgbToLinear
#[derive(Clone, Debug, Default)]
pub struct SrgbToLinearSlowBackFunc {}

impl BinaryFunc for SrgbToLinearSlowBackFunc {
	#[inline]
	fn calc(&self, input1: f32, input2: f32) -> f32 {
		if input1 <= 0.040_448_237 {
			input2 / 12.92
		} else {
			input2 * 0.001_267_54 * (200.0 * input1 + 11.0).powf(1.4)
		}
	}

	fn type_name(&self) -> &'static str {
		"SrgbToLinearSlowBackward"
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

pub type LinearToSrgbSlow = UnaryElementwise<LinearToSrgbSlowFunc>;

pub type LinearToSrgbSlowBack = BinaryElementwise<LinearToSrgbSlowBackFunc>;

#[derive(Clone, Debug, Default)]
pub struct LinearToSrgbSlowFunc {}

impl UnaryFunc for LinearToSrgbSlowFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		if input <= 0.003_130_668_5 {
			input * 12.92
		} else {
			-1.055 * (0.052_132_7 - input.powf(0.416_666_66))
		}
	}

	fn type_name(&self) -> &'static str {
		"LinearToSrgbSlow"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeID, output: &NodeID) -> Result<(), GradientError> {
		LinearToSrgbSlowBack::new_default(ctx.node(input), ctx.grad_of(output), ctx.grad_of(input)).build()?;
		Ok(())
	}
}

/// input1 = input of LinearToSrgb
/// input2 = grad of output of LinearToSrgb
#[derive(Clone, Debug, Default)]
pub struct LinearToSrgbSlowBackFunc {}

impl BinaryFunc for LinearToSrgbSlowBackFunc {
	#[inline]
	fn calc(&self, input1: f32, input2: f32) -> f32 {
		if input1 <= 0.003_130_668_5 {
			12.92 * input2
		} else {
			0.439_583 * input1.powf(-0.583_333_3) * input2
		}
	}

	fn type_name(&self) -> &'static str {
		"LinearToSrgbSlowBackward"
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
	use super::{linear_to_srgb, linear_to_srgb_slow, srgb_to_linear, srgb_to_linear_slow};
	use alumina_core::graph::Node;
	use alumina_test::grad_numeric_test::GradNumericTest;

	use indexmap::indexset;

	// TODO

	// #[test]
	// fn forward_test() {
	// 	let input = Node::new(&[13, 33]).set_name("input");

	// 	let output = tanh(&input);

	// 	input.set_value(arr0(1.25));
	// 	assert!(output
	// 		.calc()
	// 		.unwrap()
	// 		.all_relatively_close(&arr0(0.84828363995), ::std::f32::EPSILON));

	// 	input.set_value(arr0(-0.8));
	// 	assert!(output
	// 		.calc()
	// 		.unwrap()
	// 		.all_relatively_close(&arr0(-0.66403677026), ::std::f32::EPSILON));
	// }

	#[test]
	fn grad_numeric_srgb_to_linear_test() {
		let input = Node::new(&[13, 33]).set_name("input");
		let output = srgb_to_linear(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}

	#[test]
	fn grad_numeric_linear_to_srgb_test() {
		let input = Node::new(&[13, 33]).set_name("input");
		let output = linear_to_srgb(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}

	#[test]
	fn grad_numeric_srgb_to_linear_slow_test() {
		let input = Node::new(&[13, 33]).set_name("input");
		let output = srgb_to_linear_slow(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}

	#[test]
	fn grad_numeric_linear_to_srgb_slow_test() {
		let input = Node::new(&[13, 33]).set_name("input");
		let output = linear_to_srgb_slow(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}
}
