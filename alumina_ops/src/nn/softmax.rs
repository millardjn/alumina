use alumina_core::{
	base_ops::{OpInstance, OpSpecification},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{Graph, Node, NodeID},
	shape_prop::ShapePropContext,
	util::wrap_dim,
};
use indexmap::{indexset, IndexMap, IndexSet};
use ndarray::{Axis, Dimension, Zip};
use std::any::Any;

/// Calculates the combined Softmax norm of the input nodes.
///
/// Axis determines the grouping direction.
pub fn softmax<I>(logits: I, axis: isize) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let logits = logits.into();
	let axis = wrap_dim(axis, logits.shape().len());

	let output = logits.graph().new_node(logits.shape().clone());

	Softmax::new(logits, output.clone(), axis).build()?;

	Ok(output)
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct Softmax {
	logits: Node,
	output: Node,
	axis: usize,
}

impl Softmax {
	pub fn new<I, O>(logits: I, output: O, axis: usize) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		let logits = logits.into();
		let output = output.into();
		assert!(
			logits.shape().len() == output.shape().len(),
			"output and logits must have the same shape"
		);
		assert!(
			axis < logits.shape().len(),
			"axis {} must be less than logits.shape().len() {}",
			axis,
			logits.shape().len()
		);
		Softmax {
			logits: logits.clone(),
			output: output.clone(),
			axis,
		}
	}
}

impl OpSpecification for Softmax {
	type InstanceType = SoftmaxInstance;

	fn type_name(&self) -> &'static str {
		"Softmax"
	}

	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.logits.clone()]
	}

	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			logits: mapping.get(&self.logits).unwrap_or(&self.logits).clone(),
			output: mapping.get(&self.output).unwrap_or(&self.output).clone(),
			axis: self.axis,
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(SoftmaxInstance {
			logits: self.logits.id().clone(),
			output: self.output.id().clone(),
			axis: self.axis,
		})
	}
}

/// Softmax OpInstance
#[derive(Clone, Debug)]
pub struct SoftmaxInstance {
	logits: NodeID,
	output: NodeID,
	axis: usize,
}

impl OpInstance for SoftmaxInstance {
	fn type_name(&self) -> &'static str {
		"Softmax"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(Softmax {
			logits: graph.node_from_id(self.logits),
			output: graph.node_from_id(self.output),
			axis: self.axis,
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.logits.clone()]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output.clone()]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		SoftmaxBack::new(
			ctx.node(&self.logits),
			ctx.grad_of(&self.logits),
			ctx.grad_of(&self.output),
			self.axis,
		)
		.build()?;
		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		ctx.merge_output_shape(&self.output, &ctx.input_shape(&self.logits).slice().into())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		Zip::from(ctx.get_input(&self.logits).lanes(Axis(self.axis)))
			.and(ctx.get_output(&self.output).lanes_mut(Axis(self.axis)))
			.par_apply(|logits, outputs| {
				let max = logits.iter().fold(::std::f32::NEG_INFINITY, |max, &v| v.max(max));
				let exp_sum = logits.iter().fold(0.0, |sum, &v| sum + (v - max).exp());

				Zip::from(logits).and(outputs).apply(|logit, output| {
					*output += (logit - max).exp() / exp_sum;
				});
			});

		Ok(())
	}
}

/// Optimised Backward pass for Softmax Op.
///
/// Input/Output naming convention matches Softmax Input/Outputs, i.e. output_grad is an input to this Op.
///
/// All inputs and grads must be unique.
#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct SoftmaxBack {
	logits: Node,
	logits_grad: Node,
	output_grad: Node,
	axis: usize,
}

impl SoftmaxBack {
	pub fn new<I1, I2, O>(logits: I1, logits_grad: O, output_grad: I2, axis: usize) -> Self
	where
		I1: Into<Node>,
		I2: Into<Node>,
		O: Into<Node>,
	{
		let logits = logits.into();
		let logits_grad = logits_grad.into();
		let output_grad = output_grad.into();
		assert!(logits.shape().len() == logits_grad.shape().len());
		assert!(logits.shape().len() == output_grad.shape().len());
		assert!(
			axis < logits.shape().len(),
			"axis {} must be less than logits.shape().len() {}",
			axis,
			logits.shape().len()
		);
		SoftmaxBack {
			logits,
			logits_grad,
			output_grad,
			axis,
		}
	}
}

impl OpSpecification for SoftmaxBack {
	type InstanceType = SoftmaxBackInstance;

	fn type_name(&self) -> &'static str {
		"SoftmaxBack"
	}

	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.logits.clone(), self.output_grad.clone()]
	}

	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.logits_grad.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			logits: mapping.get(&self.logits).unwrap_or(&self.logits).clone(),
			output_grad: mapping.get(&self.output_grad).unwrap_or(&self.output_grad).clone(),
			logits_grad: mapping.get(&self.logits_grad).unwrap_or(&self.logits_grad).clone(),
			axis: self.axis,
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(SoftmaxBackInstance {
			logits: self.logits.id().clone(),
			logits_grad: self.logits_grad.id().clone(),
			output_grad: self.output_grad.id().clone(),
			axis: self.axis,
		})
	}
}

/// SoftmaxBack OpInstance
#[derive(Clone, Debug)]
pub struct SoftmaxBackInstance {
	logits: NodeID,
	logits_grad: NodeID,
	output_grad: NodeID,
	axis: usize,
}

impl OpInstance for SoftmaxBackInstance {
	fn type_name(&self) -> &'static str {
		"SoftmaxBack"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(SoftmaxBack {
			logits: graph.node_from_id(self.logits),
			logits_grad: graph.node_from_id(self.logits_grad),
			output_grad: graph.node_from_id(self.output_grad),
			axis: self.axis,
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.logits.clone(), self.output_grad.clone()]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.logits_grad.clone()]
	}

	fn gradient(&self, _ctx: &mut GradientContext) -> Result<(), GradientError> {
		Err(GradientError::Unimplemented)
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let logits_shape = ctx.input_shape(&self.logits).clone();
		let output_grad_shape = ctx.input_shape(&self.output_grad).clone();

		if logits_shape.ndim() == 0 {
			return Err(format!(
				"Softmax requires logit and label shapes to have at least one axis: {:?}",
				logits_shape.slice(),
			)
			.into());
		}

		if output_grad_shape != logits_shape {
			return Err(format!("SoftmaxBack requires the output grad to have the shape of the logits: logits:{:?} output_grad:{:?}, axis: {}", logits_shape.slice(), output_grad_shape.slice(), self.axis).into());
		}

		ctx.merge_output_shape(&self.logits_grad, &logits_shape.slice().into())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		Zip::from(ctx.get_output(&self.logits_grad).lanes_mut(Axis(self.axis)))
			.and(ctx.get_input(&self.logits).lanes(Axis(self.axis)))
			.and(ctx.get_input(&self.output_grad).lanes(Axis(self.axis)))
			.par_apply(|mut logits_grad, logits, output_grad| {
				let len = logits.len();

				let max = logits.iter().fold(::std::f32::NEG_INFINITY, |max, &v| v.max(max));
				let exp_sum = logits.iter().fold(0., |sum, &v| sum + (v - max).exp());
				// let exp_sum_ln = exp_sum.ln();

				for (i, grad) in output_grad.iter().enumerate() {
					if grad.abs() > 0.0 {
						// hopefully output gradients are sparse, eg from cross entropy loss

						let a = logits[i] - max;
						// let x = (a - exp_sum_ln).exp();
						let x = a.exp() / exp_sum;
						let g_x = grad * x;

						let mut other_sum = 0.0;

						for j in 0..i {
							let b = logits[j] - max;
							// logits_grad[j] -= g_x * (b - exp_sum_ln).exp();
							logits_grad[j] -= g_x * b.exp() / exp_sum;
							other_sum += b.exp() / exp_sum;
						}

						// logits_grad[i] += g_x - g_x * x;

						// inpd_n[i] += - inp_n.iter().enumerate().fold(0., |sum, (ind, v)| sum + if ind != i
						// {(v-max).exp()} else {0.0})*(mult/exp_sum);

						for j in i + 1..len {
							let b = logits[j] - max;
							// logits_grad[j] -= g_x * (b- exp_sum_ln).exp();
							logits_grad[j] -= g_x * b.exp() / exp_sum;
							other_sum += b.exp() / exp_sum;
						}

						logits_grad[i] += g_x * other_sum;
					}
				}
			});

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::softmax;
	use crate::elementwise::mul::mul;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};
	use indexmap::indexset;
	use ndarray::arr2;

	#[test]
	fn forward_test() {
		let logits = Node::new(&[4, 4])
			.set_value(arr2(&[
				[0.2, 0.4, 0.6, 0.8],
				[1.2, 1.4, 1.6, 1.8],
				[2.2, 2.4, 2.6, 2.8],
				[3.2, 3.4, 3.6, 3.8],
			]))
			.set_name("logits");

		let hor_groups = softmax(&logits, -1).unwrap();
		let vert_groups = softmax(&logits, 0).unwrap();

		assert!(hor_groups
			.calc()
			.unwrap()
			.all_relatively_close(&arr2(&[[0.180_657_18, 0.220_655_17, 0.269_508_84, 0.329_178_84]]), 1e-4));

		assert!(vert_groups.calc().unwrap().all_relatively_close(
			&arr2(&[[0.032_058_604], [0.087_144_32f32], [0.236_882_82], [0.643_914_3]]),
			1e-4
		));
	}

	#[test]
	fn grad_numeric_rand_test() {
		let logits = Node::new(&[13, 33]).set_name("logits");
		let rand = Node::new(&[13, 33]).set_name("rand"); // multiply output by random amounts to prevent gradient cancellation

		let output = mul(&softmax(&logits, -1).unwrap(), &rand).unwrap();

		GradNumericTest::new(&output, &indexset![&logits, &rand])
			.step_size(1e-3)
			.tolerance(4e-3)
			.run();
	}

	#[test]
	fn grad_numeric_test() {
		let logits = Node::new(&[13, 33]).set_name("logits");

		let output = softmax(&logits, -1).unwrap();

		GradNumericTest::new(&output, &indexset![&logits])
			.expect_zero(&logits, 20.0 * ::std::f32::EPSILON) // under a uniform output gradient the gradient of the logits should cancel out to zero
			.step_size(1e-3)
			.tolerance(4e-3)
			.run();
	}
}
