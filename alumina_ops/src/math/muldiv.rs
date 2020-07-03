use alumina_core::{
	base_ops::{OpBuilder, OpInstance},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{Node, NodeInner},
	shape_prop::ShapePropContext,
};
use indexmap::{indexset, IndexSet};

use ndarray::{Axis, Dimension, Zip};

use unchecked_index as ui;

/// An activation function based on complex multiplication and division.
///
/// This Op breaks up the inner most axis into groups of 4,
/// interprets them as two complex numbers w = (a + ib), x = (c + id),
/// and outputs the multiplication result of w * x, and division of w/x.
///
/// If the innermost axis has a remainder after group into 4s, these values are passed through without modification.
pub fn muldiv<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input.graph().new_node(input.shape().clone());

	MulDiv::new(input, output.clone()).build()?;

	Ok(output)
}

#[derive(Clone, Debug)]
pub struct MulDiv {
	input: Node,
	output: Node,
	epsilon: f32,
}

impl MulDiv {
	fn new<I, O>(input: I, output: O) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		let input = input.into();
		let output = output.into();

		MulDiv {
			input,
			output,
			epsilon: 0.5,
		}
	}

	// epsilon for divisor preventing division by zero
	//
	// Default: 1e-4
	pub fn epsilon(mut self, epsilon: f32) -> Self {
		self.epsilon = epsilon;
		self
	}
}

impl OpBuilder for MulDiv {
	type InstanceType = MulDivInstance;

	fn type_name(&self) -> &'static str {
		"MulDiv"
	}

	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input.clone()]
	}

	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	// Create a new OpInstance with nodes switched out
	// fn clone_with_nodes_changed(&self, mapping: IndexMap<Node, Node>) -> Result<Self, CloneError> {
	// 	Ok(Add {
	// 		input: mapping.get(&self.input).unwrap_or_else(|| &self.input).clone(),
	// 		output: mapping.get(&self.output).unwrap_or_else(|| &self.output).clone(),
	// 		//extra_axes: self.extra_axes,
	// 	})
	// }

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(MulDivInstance {
			input: self.input.inner().clone(),
			output: self.output.inner().clone(),
			epsilon: self.epsilon,
		})
	}
}

#[derive(Clone, Debug)]
pub struct MulDivInstance {
	input: NodeInner,
	output: NodeInner,
	epsilon: f32,
}

impl OpInstance for MulDivInstance {
	fn type_name(&self) -> &'static str {
		"MulDiv"
	}

	// fn clone_with_nodes_changed(&self, mapping: IndexMap<NodeInner, NodeInner>) -> Result<Box<OpInstance>,
	// CloneError> { 	Ok(Box::new(ExpandDimsInstance {
	// 		input: mapping.get(&self.input).unwrap_or_else(|| &self.input).clone(),
	// 		output: mapping.get(&self.output).unwrap_or_else(|| &self.output).clone(),
	// 		//extra_axes: self.extra_axes.clone(),
	// 	}))
	// }

	fn inputs(&self) -> IndexSet<NodeInner> {
		indexset![self.input.clone()]
	}

	fn outputs(&self) -> IndexSet<NodeInner> {
		indexset![self.output.clone()]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		MulDivBack::new(
			ctx.node(&self.input),
			ctx.grad_of(&self.input),
			ctx.grad_of(&self.output),
		)
		.epsilon(self.epsilon)
		.build()?;
		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		ctx.merge_output_shape(&self.output, &ctx.input_shape(&self.input).slice().into())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let input = ctx.get_input_standard(&self.input);
		let mut output = ctx.get_output_standard(&self.output);
		assert_eq!(input.shape(), output.shape());

		let epsilon = self.epsilon;
		let ndim = input.ndim();

		Zip::from(input.lanes(Axis(ndim - 1)))
			.and(output.lanes_mut(Axis(ndim - 1)))
			.par_apply(|input, mut output| {
				let len = input.len();
				debug_assert_eq!(input.len(), output.len());

				let groups = len / 4;
				let remainder = len - groups * 4;

				unsafe {
					let input = input.as_slice().unwrap();
					let output = output.as_slice_mut().unwrap();

					for i in 0..groups {
						let a = ui::get_unchecked(input, i * 4);
						let b = ui::get_unchecked(input, i * 4 + 1);
						let c = ui::get_unchecked(input, i * 4 + 2);
						let d = ui::get_unchecked(input, i * 4 + 3);

						// complex multiplication
						*ui::get_unchecked_mut(output, i * 4) += a * c - b * d;
						*ui::get_unchecked_mut(output, i * 4 + 1) += a * d + b * c;

						// complex division
						*ui::get_unchecked_mut(output, i * 4 + 2) += (a * c + b * d) / (epsilon + c * c + d * d);
						*ui::get_unchecked_mut(output, i * 4 + 3) += (b * c - a * d) / (epsilon + c * c + d * d);
					}

					for i in 0..remainder {
						*ui::get_unchecked_mut(output, groups * 4 + i) += *ui::get_unchecked(input, groups * 4 + i);
					}
				}
			});

		Ok(())
	}
}

#[derive(Clone, Debug)]
pub struct MulDivBack {
	input: Node,
	input_grad: Node,
	output_grad: Node,
	epsilon: f32,
}

impl MulDivBack {
	fn new<I1, I2, O>(input: I1, input_grad: O, output_grad: I2) -> Self
	where
		I1: Into<Node>,
		I2: Into<Node>,
		O: Into<Node>,
	{
		let input = input.into();
		let input_grad = input_grad.into();
		let output_grad = output_grad.into();

		MulDivBack {
			input,
			input_grad,
			output_grad,
			epsilon: 0.5,
		}
	}

	// epsilon for divisor preventing division by zero
	//
	// Default: 1e-4
	pub fn epsilon(mut self, epsilon: f32) -> Self {
		self.epsilon = epsilon;
		self
	}
}

impl OpBuilder for MulDivBack {
	type InstanceType = MulDivBackInstance;

	fn type_name(&self) -> &'static str {
		"MulDivBack"
	}

	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input.clone(), self.output_grad.clone()]
	}

	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.input_grad.clone()]
	}

	// Create a new OpInstance with nodes switched out
	// fn clone_with_nodes_changed(&self, mapping: IndexMap<Node, Node>) -> Result<Self, CloneError> {
	// 	Ok(Add {
	// 		input: mapping.get(&self.input).unwrap_or_else(|| &self.input).clone(),
	// 		output: mapping.get(&self.output).unwrap_or_else(|| &self.output).clone(),
	// 		//extra_axes: self.extra_axes,
	// 	})
	// }

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(MulDivBackInstance {
			input: self.input.inner().clone(),
			input_grad: self.input_grad.inner().clone(),
			output_grad: self.output_grad.inner().clone(),
			epsilon: self.epsilon,
		})
	}
}

#[derive(Clone, Debug)]
pub struct MulDivBackInstance {
	input: NodeInner,
	input_grad: NodeInner,
	output_grad: NodeInner,
	epsilon: f32,
}

impl OpInstance for MulDivBackInstance {
	fn type_name(&self) -> &'static str {
		"MulDivBack"
	}

	// fn clone_with_nodes_changed(&self, mapping: IndexMap<NodeInner, NodeInner>) -> Result<Box<OpInstance>,
	// CloneError> { 	Ok(Box::new(ExpandDimsInstance {
	// 		input: mapping.get(&self.input).unwrap_or_else(|| &self.input).clone(),
	// 		output: mapping.get(&self.output).unwrap_or_else(|| &self.output).clone(),
	// 		//extra_axes: self.extra_axes.clone(),
	// 	}))
	// }

	fn inputs(&self) -> IndexSet<NodeInner> {
		indexset![self.input.clone(), self.output_grad.clone()]
	}

	fn outputs(&self) -> IndexSet<NodeInner> {
		indexset![self.input_grad.clone()]
	}

	fn gradient(&self, _ctx: &mut GradientContext) -> Result<(), GradientError> {
		Err(GradientError::Unimplemented)
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let input_shape = ctx.input_shape(&self.input).clone();
		let output_grad_shape = ctx.input_shape(&self.output_grad).clone();

		if output_grad_shape != input_shape {
			return Err(format!(
				"MulDivBack requires the output grad to have the shape of the input: input:{:?} output_grad:{:?}",
				input_shape.slice(),
				output_grad_shape.slice()
			)
			.into());
		}

		ctx.merge_output_shape(&self.input_grad, &input_shape.slice().into())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let input = ctx.get_input_standard(&self.input);
		let mut input_grad = ctx.get_output_standard(&self.input_grad);
		let output_grad = ctx.get_input_standard(&self.output_grad);
		assert_eq!(input.shape(), output_grad.shape());
		assert_eq!(input.shape(), input_grad.shape());

		let epsilon = self.epsilon;
		let ndim = input.ndim();

		Zip::from(input_grad.lanes_mut(Axis(ndim - 1)))
			.and(input.lanes(Axis(ndim - 1)))
			.and(output_grad.lanes(Axis(ndim - 1)))
			.par_apply(|mut input_grad, input, output_grad| {
				let len = input.len();
				debug_assert_eq!(input.len(), output_grad.len());
				debug_assert_eq!(input.len(), input_grad.len());

				let groups = len / 4;
				let remainder = len - groups * 4;

				unsafe {
					let input = input.as_slice().unwrap();
					let input_grad = input_grad.as_slice_mut().unwrap();
					let output_grad = output_grad.as_slice().unwrap();

					for i in 0..groups {
						let a = ui::get_unchecked(input, i * 4);
						let b = ui::get_unchecked(input, i * 4 + 1);
						let c = ui::get_unchecked(input, i * 4 + 2);
						let d = ui::get_unchecked(input, i * 4 + 3);

						let wg = ui::get_unchecked(input_grad, i * 4);
						let xg = ui::get_unchecked(input_grad, i * 4 + 1);
						let yg = ui::get_unchecked(input_grad, i * 4 + 2);
						let zg = ui::get_unchecked(input_grad, i * 4 + 3);

						let c2d2e = c * c + d * d + epsilon;
						let c2d2e_2 = c2d2e * c2d2e;

						// gradients from multiplication
						// let agm = c*wg + d*xg;
						// let bgm = -d*wg +c*xg;
						// let cgm = a*wg + b*xg;
						// let dgm = -b*wg + a*xg;

						//gradients from division
						// a/c2d2e - 2.0*c*(a*c+b*d)/c2d2e_2 // dydc
						// b/c2d2e - 2.0*d*(a*c+b*d)/c2d2e_2 // dydd
						// b/c2d2e - 2.0*c*(b*c-a*d)/c2d2e_2 // dzdc
						// -a/c2d2e- 2.0*d*(b*c-a*d)/c2d2e_2 // dzdd

						// let agd = c*wg/c2d2e - d*xg/c2d2e;
						// let bgd = d*wg/c2d2e + c*xg/c2d2e;
						// let cgd = (a/c2d2e - 2.0*c*(a*c+b*d)/c2d2e_2)*yg + (b/c2d2e - 2.0*c*(b*c-a*d)/c2d2e_2)*zg;
						// let dgd = (b/c2d2e - 2.0*d*(a*c+b*d)/c2d2e_2)*yg + (-a/c2d2e- 2.0*d*(b*c-a*d)/c2d2e_2)*zg;

						// combined gradients
						// hopefully this vectorises
						let ag = wg * c + xg * d + yg * (c / c2d2e) + zg * -(d / c2d2e);
						let bg = wg * -d + xg * c + yg * (d / c2d2e) + zg * (c / c2d2e);
						let cg = wg * a
							+ xg * b + yg * (a / c2d2e - (a * c + b * d) * (c * 2.0 / c2d2e_2))
							+ zg * (b / c2d2e - (b * c - a * d) * (c * 2.0 / c2d2e_2));
						let dg = wg * -b
							+ xg * a + yg * (b / c2d2e - (a * c + b * d) * (d * 2.0 / c2d2e_2))
							+ zg * (-a / c2d2e - (b * c - a * d) * (d * 2.0 / c2d2e_2));

						*ui::get_unchecked_mut(input_grad, i * 4) += ag;
						*ui::get_unchecked_mut(input_grad, i * 4 + 1) += bg;
						*ui::get_unchecked_mut(input_grad, i * 4 + 2) += cg;
						*ui::get_unchecked_mut(input_grad, i * 4 + 3) += dg;
					}

					for i in 0..remainder {
						*ui::get_unchecked_mut(input_grad, groups * 4 + i) +=
							*ui::get_unchecked(output_grad, groups * 4 + i);
					}
				}
			});

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::{muldiv, MulDiv};
	use alumina_core::{base_ops::OpBuilder, graph::Node};
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr2;

	#[test]
	fn forward_test() {
		let input = Node::new(&[2, 9])
			.set_value(arr2(&[
				[0.2, 0.4, 0.6, 0.8, 2.2, 2.4, 2.6, 2.8, 4.7],
				[1.2, 1.4, 1.6, 1.8, 3.2, 3.4, 3.6, 3.8, 3.2],
			]))
			.set_name("input");

		let output = muldiv(&input).unwrap();

		assert!(output.calc().unwrap().all_relatively_close(
			&arr2(&[
				[
					-0.2,
					0.4,
					0.439_956,
					0.079_992_004,
					-1.0,
					12.4,
					0.852_048_93,
					0.005_479_415,
					4.7
				],
				[
					-0.6,
					4.4,
					0.765_504_06,
					0.013_792_866,
					-1.4,
					24.4,
					0.891_967_54,
					0.002_919_697,
					3.2
				],
			]),
			1e-5
		));
	}

	#[test]
	fn forward_epse_one_test() {
		let input = Node::new(&[2, 9])
			.set_value(arr2(&[
				[0.2, 0.4, 0.6, 0.8, 2.2, 2.4, 2.6, 2.8, 4.7],
				[1.2, 1.4, 1.6, 1.8, 3.2, 3.4, 3.6, 3.8, 3.2],
			]))
			.set_name("input");
		let output = Node::new(&[2, 9]).set_name("output");

		MulDiv::new(&input, &output).epsilon(1.0).build().unwrap();

		assert!(output.calc().unwrap().all_relatively_close(
			&arr2(&[
				[-0.2, 0.4, 0.22, 0.04, -1.0, 12.4, 0.797_435_9, 0.005_128_205, 4.7],
				[
					-0.6,
					4.4,
					0.652_941_17,
					0.011_764_706,
					-1.4,
					24.4,
					0.860_563_4,
					0.002_816_901,
					3.2
				],
			]),
			1e-5
		));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[13, 43]).set_name("input");

		let output = muldiv(&input).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}

	#[test]
	fn grad_numeric_eps_one_test() {
		let input = Node::new(&[13, 43]).set_name("input");
		let output = Node::new(&[13, 43]).set_name("output");

		MulDiv::new(&input, &output).epsilon(1.0).build().unwrap();

		GradNumericTest::new(&output, &indexset![&input]).tolerance(2e-5).run();
	}
}
