use alumina_core::{
	base_ops::{OpSpecification, OpInstance},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{merge_graphs, Node, NodeID, Graph},
	shape_prop::ShapePropContext,
	util::wrap_dim,
};
use indexmap::{indexset, IndexSet};
use ndarray::{Axis, Dimension, Zip};
use std::any::Any;

/// Calculates the Softmax of the logits across the select axis followed by CrossEntropy of that result with the
/// supplied labels.
///
/// These operations are combined for numerical stability and performance improvement.
///
/// `let output = reduce_sum(mul(labels, negative(ln(softmax(logits, axis)))), &[axis])`
///
/// The output node has the shape of the logits and labels, but with the axis removed.
pub fn softmax_cross_entropy<I1, I2>(logits: I1, labels: I2, axis: isize) -> Result<Node, OpBuildError>
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	let logits = logits.into();
	let labels = labels.into();
	let axis = wrap_dim(axis, logits.shape().len());

	let graph = merge_graphs(&[logits.graph(), labels.graph()]);

	let output_shape = logits
		.shape()
		.iter()
		.enumerate()
		.filter_map(|(i, x)| if i == axis { None } else { Some(x) })
		.into();

	let output = graph
		.new_node(output_shape)
		.set_name_unique(&format!("softmax_cross_entropy({},{})", logits, labels));

	SoftmaxCrossEntropy::new(logits, labels, output.clone(), axis).build()?;

	Ok(output)
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct SoftmaxCrossEntropy {
	logits: Node,
	labels: Node,
	output: Node,
	axis: usize,
}

impl SoftmaxCrossEntropy {
	pub fn new<I1, I2, O>(logits: I1, labels: I2, output: O, axis: usize) -> Self
	where
		I1: Into<Node>,
		I2: Into<Node>,
		O: Into<Node>,
	{
		let logits = logits.into();
		let labels = labels.into();
		let output = output.into();

		assert!(logits.shape().len() == labels.shape().len());
		assert!(logits.shape().len() == output.shape().len() + 1);
		assert!(
			axis < logits.shape().len(),
			"axis {} must be less than logits.shape().len() {}",
			axis,
			logits.shape().len()
		);
		SoftmaxCrossEntropy {
			logits: logits.clone(),
			labels: labels.clone(),
			output: output.clone(),
			axis,
		}
	}
}

impl OpSpecification for SoftmaxCrossEntropy {
	type InstanceType = SoftmaxCrossEntropyInstance;

	fn type_name(&self) -> &'static str {
		"SoftmaxCrossEntropy"
	}

	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.logits.clone(), self.labels.clone()]
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
		Ok(SoftmaxCrossEntropyInstance {
			logits: self.logits.id().clone(),
			labels: self.labels.id().clone(),
			output: self.output.id().clone(),
			axis: self.axis,
		})
	}
}

/// SoftmaxCrossEntropy OpInstance
#[derive(Clone, Debug)]
pub struct SoftmaxCrossEntropyInstance {
	logits: NodeID,
	labels: NodeID,
	output: NodeID,
	axis: usize,
}

impl OpInstance for SoftmaxCrossEntropyInstance {
	fn type_name(&self) -> &'static str {
		"SoftmaxCrossEntropy"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(SoftmaxCrossEntropy {
			logits: graph.node_from_id(self.logits),
			labels: graph.node_from_id(self.labels),
			output: graph.node_from_id(self.output),
			axis: self.axis,
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.logits.clone(), self.labels.clone()]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output.clone()]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		SoftmaxCrossEntropyBack::new(
			ctx.node(&self.logits),
			ctx.grad_of(&self.logits),
			ctx.node(&self.labels),
			ctx.grad_of(&self.labels),
			ctx.grad_of(&self.output),
			self.axis,
		)
		.build()?;
		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let logits_shape = ctx.input_shape(&self.logits);
		let labels_shape = ctx.input_shape(&self.labels);

		if logits_shape.ndim() == 0 || labels_shape.ndim() == 0 {
			return Err(format!(
				"SoftmaxCrossEntropy requires logit and label shapes to have at least one axis: {:?} {:?}",
				logits_shape.slice(),
				labels_shape.slice()
			)
			.into());
		}

		if logits_shape != labels_shape {
			return Err(format!(
				"SoftmaxCrossEntropy requires logit and label shapes to be the same: {:?} {:?}",
				logits_shape.slice(),
				labels_shape.slice()
			)
			.into());
		}

		let output_shape = &logits_shape
			.slice()
			.iter()
			.enumerate()
			.filter_map(|(i, axis)| if i == self.axis { None } else { Some(axis) })
			.into();

		ctx.merge_output_shape(&self.output, output_shape)
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		Zip::from(ctx.get_output(&self.output))
			.and(ctx.get_input(&self.logits).lanes(Axis(self.axis)))
			.and(ctx.get_input(&self.labels).lanes(Axis(self.axis)))
			.par_apply(|output, logits, labels| {
				let max = logits.iter().fold(::std::f32::NEG_INFINITY, |max, &v| v.max(max));
				let exp_sum = logits.iter().fold(0., |sum, &v| sum + (v - max).exp());

				Zip::from(logits).and(labels).apply(|logit, label| {
					*output += label * (exp_sum.ln() - (logit - max));
				});
			});

		Ok(())
	}
}

/// Optimised Backward pass for SoftmaxCrossEntropy Op.
///
/// Input/Output naming convention matches SoftmaxCrossEntropy Input/Outputs, i.e. output_grad is an input to this Op.
///
/// All inputs and grads must be unique.
#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct SoftmaxCrossEntropyBack {
	logits: Node,
	logits_grad: Node,
	labels: Node,
	labels_grad: Node,
	output_grad: Node,
	axis: usize,
}

impl SoftmaxCrossEntropyBack {
	pub fn new<I1, I2, I3, O1, O2>(
		logits: I1,
		logits_grad: O1,
		labels: I2,
		labels_grad: O2,
		output_grad: I3,
		axis: usize,
	) -> Self
	where
		I1: Into<Node>,
		I2: Into<Node>,
		I3: Into<Node>,
		O1: Into<Node>,
		O2: Into<Node>,
	{
		let logits = logits.into();
		let logits_grad = logits_grad.into();
		let labels = labels.into();
		let labels_grad = labels_grad.into();
		let output_grad = output_grad.into();
		assert!(logits.shape().len() == labels.shape().len());
		assert!(logits.shape().len() == logits_grad.shape().len());
		assert!(logits.shape().len() == labels_grad.shape().len());
		assert!(logits.shape().len() == output_grad.shape().len() + 1);
		assert!(
			axis < logits.shape().len(),
			"axis {} must be less than logits.shape().len() {}",
			axis,
			logits.shape().len()
		);
		SoftmaxCrossEntropyBack {
			logits,
			logits_grad,
			labels,
			labels_grad,
			output_grad,
			axis,
		}
	}
}

impl OpSpecification for SoftmaxCrossEntropyBack {
	type InstanceType = SoftmaxCrossEntropyBackInstance;

	fn type_name(&self) -> &'static str {
		"SoftmaxCrossEntropyBack"
	}

	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.logits.clone(), self.labels.clone(), self.output_grad.clone()]
	}

	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.logits_grad.clone(), self.labels_grad.clone()]
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
		Ok(SoftmaxCrossEntropyBackInstance {
			logits: self.logits.id().clone(),
			logits_grad: self.logits_grad.id().clone(),
			labels: self.labels.id().clone(),
			labels_grad: self.labels_grad.id().clone(),
			output_grad: self.output_grad.id().clone(),
			axis: self.axis,
		})
	}
}

/// SoftmaxCrossEntropyBack OpInstance
#[derive(Clone, Debug)]
pub struct SoftmaxCrossEntropyBackInstance {
	logits: NodeID,
	logits_grad: NodeID,
	labels: NodeID,
	labels_grad: NodeID,
	output_grad: NodeID,
	axis: usize,
}

impl OpInstance for SoftmaxCrossEntropyBackInstance {
	fn type_name(&self) -> &'static str {
		"SoftmaxCrossEntropyBack"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(SoftmaxCrossEntropyBack {
			logits: graph.node_from_id(self.logits),
			logits_grad: graph.node_from_id(self.logits_grad),
			labels: graph.node_from_id(self.labels),
			labels_grad: graph.node_from_id(self.labels_grad),
			output_grad: graph.node_from_id(self.output_grad),
			axis: self.axis,
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.logits.clone(), self.labels.clone(), self.output_grad.clone()]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.logits_grad.clone(), self.labels_grad.clone()]
	}

	fn gradient(&self, _ctx: &mut GradientContext) -> Result<(), GradientError> {
		Err(GradientError::Unimplemented)
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let logits_shape = ctx.input_shape(&self.logits).clone();
		let labels_shape = ctx.input_shape(&self.labels).clone();
		let output_grad_shape = ctx.input_shape(&self.output_grad).clone();

		if logits_shape.ndim() == 0 || labels_shape.ndim() == 0 {
			return Err(format!(
				"SoftmaxCrossEntropy requires logit and label shapes to have at least one axis: {:?} {:?}",
				logits_shape.slice(),
				labels_shape.slice()
			)
			.into());
		}

		if logits_shape != labels_shape {
			return Err(format!(
				"SoftmaxCrossEntropy requires logit and label shapes to be the same: {:?} {:?}",
				logits_shape.slice(),
				labels_shape.slice()
			)
			.into());
		}

		if output_grad_shape.slice().len() + 1 != logits_shape.slice().len()
			|| logits_shape
				.slice()
				.iter()
				.enumerate()
				.filter_map(|(i, &axis)| if i == self.axis { None } else { Some(axis) })
				.zip(output_grad_shape.slice())
				.any(|(logit_axis, &out_grad_axis)| logit_axis != out_grad_axis)
		{
			return Err(format!("SoftmaxCrossEntropyBack requires the output grad to have the shape of the logits with the selected axis removed: logits:{:?} output_grad:{:?}, axis: {}", logits_shape.slice(), output_grad_shape.slice(), self.axis).into());
		}

		ctx.merge_output_shape(&self.logits_grad, &logits_shape.slice().into())?;
		ctx.merge_output_shape(&self.labels_grad, &logits_shape.slice().into())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		if !ctx.is_required_output(&self.labels_grad) {
			Zip::from(ctx.get_output(&self.logits_grad).lanes_mut(Axis(self.axis)))
				.and(ctx.get_input(&self.logits).lanes(Axis(self.axis)))
				.and(ctx.get_input(&self.labels).lanes(Axis(self.axis)))
				.and(ctx.get_input(&self.output_grad))
				.par_apply(|mut logits_grad, logits, labels, output_grad| {
					let len = logits.len();

					let max = logits.iter().fold(::std::f32::NEG_INFINITY, |max, &v| v.max(max));
					let exp_sum = logits.iter().fold(0., |sum, &v| sum + (v - max).exp());

					for (i, &label) in labels.iter().enumerate() {
						if label != 0.0 {
							let strength = label * output_grad;

							let a = logits[i] - max;

							for j in 0..i {
								logits_grad[j] += (logits[j] - max).exp() * (strength / exp_sum);
							}

							logits_grad[i] += (a.exp() - exp_sum) * (strength / exp_sum);
							// inpd_n[i] += - inp_n.iter().enumerate().fold(0., |sum, (ind, v)| sum + if ind != i
							// {(v-max).exp()} else {0.0})*(mult/exp_sum);

							for j in i + 1..len {
								logits_grad[j] += (logits[j] - max).exp() * (strength / exp_sum);
							}
						}
					}
				});
		} else if !ctx.is_required_output(&self.logits_grad) {
			Zip::from(ctx.get_input(&self.logits).lanes(Axis(self.axis)))
				.and(ctx.get_output(&self.labels_grad).lanes_mut(Axis(self.axis)))
				.and(ctx.get_input(&self.output_grad))
				.par_apply(|logits, mut label_grad, output_grad| {
					let max = logits.iter().fold(::std::f32::NEG_INFINITY, |max, &v| v.max(max));
					let exp_sum = logits.iter().fold(0., |sum, &v| sum + (v - max).exp());

					for (i, &logit) in logits.iter().enumerate() {
						label_grad[i] += *output_grad * (exp_sum.ln() - (logit - max));
					}
				});
		} else {
			// both

			Zip::from(ctx.get_output(&self.logits_grad).lanes_mut(Axis(self.axis)))
				.and(ctx.get_input(&self.logits).lanes(Axis(self.axis)))
				.and(ctx.get_output(&self.labels_grad).lanes_mut(Axis(self.axis)))
				.and(ctx.get_input(&self.labels).lanes(Axis(self.axis)))
				.and(ctx.get_input(&self.output_grad))
				.par_apply(|mut logits_grad, logits, mut label_grad, labels, output_grad| {
					let len = logits.len();

					let max = logits.iter().fold(::std::f32::NEG_INFINITY, |max, &v| v.max(max));
					let exp_sum = logits.iter().fold(0., |sum, &v| sum + (v - max).exp());

					for (i, &logit) in logits.iter().enumerate() {
						label_grad[i] += *output_grad * (exp_sum.ln() - (logit - max));
					}

					for (i, &label) in labels.iter().enumerate() {
						if label != 0.0 {
							let strength = label * output_grad;

							let a = logits[i] - max;

							for j in 0..i {
								logits_grad[j] += (logits[j] - max).exp() * (strength / exp_sum);
							}

							logits_grad[i] += (a.exp() - exp_sum) * (strength / exp_sum);
							// inpd_n[i] += - inp_n.iter().enumerate().fold(0., |sum, (ind, v)| sum + if ind != i
							// {(v-max).exp()} else {0.0})*(mult/exp_sum);

							for j in i + 1..len {
								logits_grad[j] += (logits[j] - max).exp() * (strength / exp_sum);
							}
						}
					}
				});
		}

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::softmax_cross_entropy;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};
	use indexmap::indexset;

	use ndarray::{arr1, arr2};
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

		let labels = Node::new(&[4, 4])
			.set_value(arr2(&[
				[0.0, 1.0, 0.0, 0.0],
				[0.0, 0.0, 1.0, 0.0],
				[1.0, 0.0, 0.0, 0.0],
				[0.0, 0.0, 0.0, 1.0],
			]))
			.set_name("labels");

		let hor_groups = softmax_cross_entropy(&logits, &labels, -1).unwrap();
		let vert_groups = softmax_cross_entropy(&logits, &labels, 0).unwrap();

		assert!(hor_groups
			.calc()
			.unwrap()
			.all_relatively_close(&arr1(&[1.511_154, 1.311_154, 1.711_154, 1.111_154]), 1e-4));

		assert!(vert_groups
			.calc()
			.unwrap()
			.all_relatively_close(&arr1(&[1.44019, 3.44019, 2.44019, 0.44019]), 1e-4));
	}

	#[test]
	fn grad_numeric_test() {
		let input1 = Node::new(&[13, 33]).set_name("input1");
		let input2 = Node::new(&[13, 33]).set_name("input2");

		let output = softmax_cross_entropy(&input1, &input2, -1).unwrap();

		GradNumericTest::new(&output, &indexset![&input1, &input2])
			.step_size(1e-3)
			.tolerance(4e-3)
			.run();
	}
}
