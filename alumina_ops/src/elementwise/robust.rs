use crate::elementwise::elementwise_single::{BinaryElementwise, BinaryFunc, UnaryElementwise, UnaryFunc};
use alumina_core::{
	base_ops::OpBuilder,
	errors::{GradientError, OpBuildError},
	grad::GradientContext,
	graph::{Node, NodeInner},
};

/// An `Op` which implements a range of robust loss functions.
///
/// Based on the paper: A More General Robust Loss Function https://arxiv.org/pdf/1701.03077.pdf Eq.13 & Eq.14
/// Note that:
///
/// when power(α) == 2, this is the L2 loss
///
/// when power(α) == 1, this is the pseudo-Huber/Charbonnier loss (smooth L1 loss)
///
/// when power(α) == 0, this is the Cauchy/Lorentzian loss
///
/// when power(α) == -2, this is the Geman-McClure loss
///
/// when power(α) == -∞, this is the Welsch/Leclerc loss
///
/// The scale(c) is the range of values either size of zero for which the loss will closely approximate the L2 loss.
/// A small scale value will mean that small inputs will result in larger outputs.
/// See paper for futher details.
///
/// ρ(x,α,c) =
/// if α == 0 : log(0.5*(x/c)^2+ 1)
/// if α == -∞: 1 - exp(-0.5 *(x/c)^2)
/// else      : z(α)/α * (((x/c)^2/z(α) + 1)^(α/2) − 1)
/// where z(α) = max(1, 2 - α)
pub fn robust<I>(input: I, scale: f32, power: f32) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(input.shape().clone())
		.set_name_unique(&format!("robust({})", input));
	let _op = Robust::new(input, output.clone(), RobustFunc { scale, power }).build()?;
	Ok(output)
}

pub type Robust = UnaryElementwise<RobustFunc>;

pub type RobustBack = BinaryElementwise<RobustBackFunc>;

#[derive(Clone, Debug)]
pub struct RobustFunc {
	scale: f32,
	power: f32,
}

impl UnaryFunc for RobustFunc {
	#[inline]
	fn calc(&self, input: f32) -> f32 {
		let c = self.scale; // use notation from paper
		let a = self.power;
		let x = input;
		#[allow(clippy::float_cmp)] // comparing to a user value not a computed value, stfu clippy
		{
			if a == 0.0 {
				(0.5 * (x / c) * (x / c)).ln_1p()
			} else if a == ::std::f32::NEG_INFINITY {
				-(-0.5 * (x / c) * (x / c)).exp_m1()
			} else if a == 1.0 {
				// (((x/c)*(x/c) + 1.0).sqrt() - 1.0)

				// This is the numerically stable version of above as per https://stackoverflow.com/questions/32444817/numerically-stable-evaluation-of-sqrtxa-sqrtx
				let z = (x / c) * (x / c);
				z / ((z + 1.0).sqrt() + 1.0)
			} else if a == 2.0 {
				((x / c) * (x / c)) / a
			} else {
				let za = 1.0f32.max(2.0 - a);
				za / a * (((x / c) * (x / c) / za + 1.0).powf(0.5 * a) - 1.0)
			}
		}
	}

	fn type_name(&self) -> &'static str {
		"Robust"
	}

	fn grad(&self, ctx: &mut GradientContext, input: &NodeInner, output: &NodeInner) -> Result<(), GradientError> {
		RobustBack::new(
			ctx.node(input),
			ctx.grad_of(output),
			ctx.grad_of(input),
			RobustBackFunc {
				scale: self.scale,
				power: self.power,
			},
		)
		.build()?;
		Ok(())
	}
}

/// input1 = input of robust
/// input2 = grad of output of robust
#[derive(Clone, Debug)]
pub struct RobustBackFunc {
	scale: f32,
	power: f32,
}

impl BinaryFunc for RobustBackFunc {
	#[inline]
	fn calc(&self, input1: f32, input2: f32) -> f32 {
		let c = self.scale; // use notation from paper
		let a = self.power;
		let x = input1;
		let grad = input2;
		#[allow(clippy::float_cmp)] // comparing to a user value not a computed value, stfu clippy
		{
			if a == 0.0 {
				grad * 2.0 * x / (x * x + 2.0 * c * c)
			} else if a == ::std::f32::NEG_INFINITY {
				grad * x / (c * c) * (-0.5 * (x / c) * (x / c)).exp()
			} else if a == 1.0 {
				grad * x / ((c * c) * ((x / c) * (x / c) + 1.0).sqrt())
			} else if a == 2.0 {
				grad * x / (c * c)
			} else {
				let za = 1.0f32.max(2.0 - a);
				grad * x / (c * c) * ((x / c) * (x / c) / za + 1.0).powf(0.5 * a - 1.0)
			}
		}
	}

	fn type_name(&self) -> &'static str {
		"RobustBackward"
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
	use super::{Robust, RobustFunc};
	use alumina_core::{base_ops::OpBuilder, graph::Node};
	use alumina_test::grad_numeric_test::GradNumericTest;

	use rand::thread_rng;
	use rand_distr::{Distribution, Uniform};

	#[test]
	fn grad_numeric_robust_zero_test() {
		let mut rng = thread_rng();
		let range = Uniform::new(0.1, 1.0);

		let power = 0.0;
		let scale = 1.0 + range.sample(&mut rng);

		let input = Node::new(&[13, 33]).set_name("input");
		let output = Node::new(&[13, 33]).set_name("output");
		let _op = Robust::new(input.clone(), output.clone(), RobustFunc { scale, power })
			.build()
			.unwrap();

		GradNumericTest::new(output, &[input]).run();
	}

	#[test]
	fn grad_numeric_robust_one_test() {
		let mut rng = thread_rng();
		let range = Uniform::new(0.1, 1.0);

		let power = 1.0;
		let scale = 1.0 + range.sample(&mut rng);

		let input = Node::new(&[13, 33]).set_name("input");
		let output = Node::new(&[13, 33]).set_name("output");
		let _op = Robust::new(input.clone(), output.clone(), RobustFunc { scale, power })
			.build()
			.unwrap();

		GradNumericTest::new(output, &[input]).run();
	}

	#[test]
	fn grad_numeric_robust_two_test() {
		let mut rng = thread_rng();
		let range = Uniform::new(0.1, 1.0);

		let power = 2.0;
		let scale = 1.0 + range.sample(&mut rng);

		let input = Node::new(&[13, 33]).set_name("input");
		let output = Node::new(&[13, 33]).set_name("output");
		let _op = Robust::new(input.clone(), output.clone(), RobustFunc { scale, power })
			.build()
			.unwrap();

		GradNumericTest::new(output, &[input]).run();
	}

	#[test]
	fn grad_numeric_robust_neg_inf_test() {
		let mut rng = thread_rng();
		let range = Uniform::new(0.1, 1.0);

		let power = ::std::f32::NEG_INFINITY;
		let scale = 1.0 + range.sample(&mut rng);

		let input = Node::new(&[13, 33]).set_name("input");
		let output = Node::new(&[13, 33]).set_name("output");
		let _op = Robust::new(input.clone(), output.clone(), RobustFunc { scale, power })
			.build()
			.unwrap();

		GradNumericTest::new(output, &[input]).run();
	}

	#[test]
	fn grad_numeric_robust_rand_test() {
		let mut rng = thread_rng();
		let range = Uniform::new(0.1, 1.0);
		let power_range = Uniform::new(-3.0, 3.0);

		for _ in 0..20 {
			let power = power_range.sample(&mut rng);
			let scale = 1.0 + range.sample(&mut rng);

			let input = Node::new(&[13, 33]).set_name("input");
			let output = Node::new(&[13, 33]).set_name("output");
			let _op = Robust::new(input.clone(), output.clone(), RobustFunc { scale, power })
				.build()
				.unwrap();

			GradNumericTest::new(output, &[input]).tolerance(2e-4).run();
		}
	}
}