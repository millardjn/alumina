use std::any::Any;

use alumina_core::{
	errors::{OpBuildError, GradientError, ShapePropError, ExecutionError},
	graph::{Node, Graph, NodeID, merge_graphs}, base_ops::{OpSpecification, OpInstance}, grad::GradientContext, shape_prop::ShapePropContext, exec::ExecutionContext
};
use indexmap::{indexset, IndexMap, IndexSet};
use ndarray::{Zip, Dimension};

use crate::{elementwise::{mul::mul, ln::ln, offset::offset, negative::negative, subtract::subtract}};

/// Calculates the logistic function of the logits followed by BinaryCrossEntropy of that result with the
/// supplied labels.
///
/// `let output = labels * -ln(logistic(logits)) + (1-labels) * -ln(1-logistic(logits))`
///
/// The output node has the shape of the logits and labels.
pub fn cross_entropy_with_logits<I1, I2>(logits: I1, labels: I2) -> Result<Node, OpBuildError>
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	let logits = logits.into();
	let labels = labels.into();

	let graph = merge_graphs(&[logits.graph(), labels.graph()]);
	// let name = format!("cross_entropy_with_logits({},{})", logits, labels);

	// let probability = logistic(logits)?;

	// let true_loss = mul(&labels, ln(&probability)?)?;

	// let false_loss = mul(offset(labels, -1.0)?, ln(negative(offset(probability, -1.0)?)?)?)?;

	// let output = subtract(false_loss, true_loss)?.set_name_unique(&name);

	let output = graph
		.new_node(logits.shape())
		.set_name_unique(&format!("cross_entropy_with_logits({},{})", logits, labels));

	CrossEntropyWithLogits::new(logits, labels, output.clone()).build()?;

	Ok(output)
}



/// Calculates the logistic function of the logits followed by BinaryCrossEntropy of that result with the
/// supplied labels.
///
/// `let output = mul(labels, negative(ln(logistic(logits))))`
///
/// The output node has the shape of the logits and labels.
pub fn cross_entropy<I1, I2>(probability: I1, labels: I2) -> Result<Node, OpBuildError>
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	let probability = probability.into();
	let labels = labels.into();
	let name = format!("cross_entropy({},{})", probability, labels);

	let true_loss = mul(&labels, ln(&probability)?)?;

	let false_loss = mul(offset(labels, -1.0)?, ln(negative(offset(probability, -1.0)?)?)?)?;

	let output = subtract(false_loss, true_loss)?.set_name_unique(&name);

	Ok(output)
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct CrossEntropyWithLogits {
	logits: Node,
	labels: Node,
	output: Node,
}

impl CrossEntropyWithLogits {
	pub fn new<I1, I2, O>(logits: I1, labels: I2, output: O) -> Self
	where
		I1: Into<Node>,
		I2: Into<Node>,
		O: Into<Node>,
	{
		let logits = logits.into();
		let labels = labels.into();
		let output = output.into();

		assert!(logits.shape().len() == labels.shape().len());
		assert!(logits.shape().len() == output.shape().len());

		CrossEntropyWithLogits {
			logits,
			labels,
			output,
		}
	}
}

impl OpSpecification for CrossEntropyWithLogits {
	type InstanceType = CrossEntropyWithLogitsInstance;

	fn type_name(&self) -> &'static str {
		"CrossEntropyWithLogits"
	}

	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.logits.clone(), self.labels.clone()]
	}

	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			logits: mapping.get(&self.logits).unwrap_or(&self.logits).clone(),
			labels: mapping.get(&self.labels).unwrap_or(&self.labels).clone(),
			output: mapping.get(&self.output).unwrap_or(&self.output).clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(CrossEntropyWithLogitsInstance {
			logits: self.logits.id(),
			labels: self.labels.id(),
			output: self.output.id(),
		})
	}
}

/// CrossEntropyWithLogits OpInstance
#[derive(Clone, Debug)]
pub struct CrossEntropyWithLogitsInstance {
	logits: NodeID,
	labels: NodeID,
	output: NodeID,
}

impl OpInstance for CrossEntropyWithLogitsInstance {
	fn type_name(&self) -> &'static str {
		"CrossEntropyWithLogits"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(CrossEntropyWithLogits {
			logits: graph.node_from_id(self.logits),
			labels: graph.node_from_id(self.labels),
			output: graph.node_from_id(self.output),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.logits, self.labels]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		CrossEntropyWithLogitsBack::new(
			ctx.node(&self.logits),
			ctx.grad_of(&self.logits),
			ctx.node(&self.labels),
			ctx.grad_of(&self.labels),
			ctx.grad_of(&self.output),
		)
		.build()?;
		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let logits_shape = ctx.input_shape(&self.logits);
		let labels_shape = ctx.input_shape(&self.labels);

		if logits_shape != labels_shape {
			return Err(format!(
				"CrossEntropyWithLogits requires logit and label shapes to be the same: {:?} {:?}",
				logits_shape.slice(),
				labels_shape.slice()
			)
			.into());
		}

		let output_shape = logits_shape.slice().into();

		ctx.merge_output_shape(&self.output, &output_shape)
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		Zip::from(ctx.get_output(&self.output))
			.and(ctx.get_input(&self.logits))
			.and(ctx.get_input(&self.labels))
			.par_for_each(|output, &logit, &label| {
				*output += logit.max(0.0) - logit*label + ((-logit.abs()).exp()).ln_1p();
			});

		Ok(())
	}
}

/// Optimised Backward pass for CrossEntropyWithLogits Op.
///
/// Input/Output naming convention matches CrossEntropyWithLogits Input/Outputs, i.e. output_grad is an input to this Op.
///
/// All inputs and grads must be unique.
#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct CrossEntropyWithLogitsBack {
	logits: Node,
	labels: Node,
	output_grad: Node,
	logits_grad: Node,
	labels_grad: Node,
}

impl CrossEntropyWithLogitsBack {
	pub fn new<I1, I2, I3, O1, O2>(
		logits: I1,
		logits_grad: O1,
		labels: I2,
		labels_grad: O2,
		output_grad: I3,
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
		assert!(logits.shape().len() == output_grad.shape().len());

		CrossEntropyWithLogitsBack {
			logits,
			labels,
			output_grad,
			logits_grad,
			labels_grad,
		}
	}
}

impl OpSpecification for CrossEntropyWithLogitsBack {
	type InstanceType = CrossEntropyWithLogitsBackInstance;

	fn type_name(&self) -> &'static str {
		"CrossEntropyWithLogitsBack"
	}

	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.logits.clone(), self.labels.clone(), self.output_grad.clone()]
	}

	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.logits_grad.clone(), self.labels_grad.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			logits: mapping.get(&self.logits).unwrap_or(&self.logits).clone(),
			labels: mapping.get(&self.labels).unwrap_or(&self.labels).clone(),
			output_grad: mapping.get(&self.output_grad).unwrap_or(&self.output_grad).clone(),
			logits_grad: mapping.get(&self.logits_grad).unwrap_or(&self.logits_grad).clone(),
			labels_grad: mapping.get(&self.labels_grad).unwrap_or(&self.labels_grad).clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(CrossEntropyWithLogitsBackInstance {
			logits: self.logits.id(),
			logits_grad: self.logits_grad.id(),
			labels: self.labels.id(),
			labels_grad: self.labels_grad.id(),
			output_grad: self.output_grad.id(),
		})
	}
}

/// CrossEntropyWithLogitsBack OpInstance
#[derive(Clone, Debug)]
pub struct CrossEntropyWithLogitsBackInstance {
	logits: NodeID,
	logits_grad: NodeID,
	labels: NodeID,
	labels_grad: NodeID,
	output_grad: NodeID,
}

impl OpInstance for CrossEntropyWithLogitsBackInstance {
	fn type_name(&self) -> &'static str {
		"CrossEntropyWithLogitsBack"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(CrossEntropyWithLogitsBack {
			logits: graph.node_from_id(self.logits),
			logits_grad: graph.node_from_id(self.logits_grad),
			labels: graph.node_from_id(self.labels),
			labels_grad: graph.node_from_id(self.labels_grad),
			output_grad: graph.node_from_id(self.output_grad),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.logits, self.labels, self.output_grad]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.logits_grad, self.labels_grad]
	}

	fn gradient(&self, _ctx: &mut GradientContext) -> Result<(), GradientError> {
		Err(GradientError::Unimplemented)
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let logits_shape = ctx.input_shape(&self.logits).clone();
		let labels_shape = ctx.input_shape(&self.labels).clone();
		let output_grad_shape = ctx.input_shape(&self.output_grad).clone();

		if logits_shape != labels_shape {
			return Err(format!(
				"CrossEntropyWithLogitsBack requires logit and label shapes to be the same: {:?} {:?}",
				logits_shape.slice(),
				labels_shape.slice()
			)
			.into());
		}

		if output_grad_shape.slice().len() != logits_shape.slice().len()
			|| logits_shape
				.slice()
				.iter()
				.zip(output_grad_shape.slice())
				.any(|(&logit_axis, &out_grad_axis)| logit_axis != out_grad_axis)
		{
			return Err(format!("CrossEntropyWithLogitsBack requires the output grad to have the shape of the logits: logits:{:?} output_grad:{:?}", logits_shape.slice(), output_grad_shape.slice()).into());
		}

		ctx.merge_output_shape(&self.logits_grad, &logits_shape.slice().into())?;
		ctx.merge_output_shape(&self.labels_grad, &logits_shape.slice().into())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		if !ctx.is_required_output(&self.labels_grad) {
			Zip::from(ctx.get_output(&self.logits_grad))
				.and(ctx.get_input(&self.logits))
				.and(ctx.get_input(&self.labels))
				.and(ctx.get_input(&self.output_grad))
				.par_for_each(|logits_grad, logit, label, output_grad| {
					//*logits_grad += output_grad * -((label-1.0) * logit.exp() + label)/(logit.exp() + 1.0);
					*logits_grad += output_grad * -(label/(logit.exp() + 1.0) + (label-1.0)/((-logit).exp() + 1.0));
				});
		} else if !ctx.is_required_output(&self.logits_grad) {
			Zip::from(ctx.get_input(&self.logits))
				.and(ctx.get_output(&self.labels_grad))
				.and(ctx.get_input(&self.output_grad))
				.par_for_each(|logit, label_grad, output_grad| {
					*label_grad += output_grad *-logit;
				});
		} else {
			// both
			Zip::from(ctx.get_output(&self.logits_grad))
				.and(ctx.get_input(&self.logits))
				.and(ctx.get_output(&self.labels_grad))
				.and(ctx.get_input(&self.labels))
				.and(ctx.get_input(&self.output_grad))
				.par_for_each(|logits_grad, logit, label_grad, label, output_grad| {
					*logits_grad += output_grad * -(label/(logit.exp() + 1.0) + (label-1.0)/((-logit).exp() + 1.0));
					*label_grad += output_grad *-logit;					
				});
		}

		Ok(())
	}
}



#[cfg(test)]
mod tests {
	use super::cross_entropy_with_logits;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};
	use indexmap::indexset;

	use ndarray::arr1;
	#[test]
	fn forward_test() {
		let logits = Node::new(&[8])
			.set_value(arr1(&
				[0.2, 0.4, 0.6, 0.8, -1.2, -1.4, -1.6, -1.8],
			))
			.set_name("logits");

		let labels = Node::new(&[8])
			.set_value(arr1(&
				[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
			))
			.set_name("labels");

		let hor_groups = cross_entropy_with_logits(&logits, &labels).unwrap();

		assert!(hor_groups
			.calc()
			.unwrap()
			.all_relatively_close(&arr1(&[0.798_138_86, 0.513_015_32, 1.037_488, 0.371_100_66, 0.263_282_48, 1.620_417_4, 1.783_900_7, 0.152_977_62 ]), 1e-5));
	}

	#[test]
	fn grad_numeric_test() {
		let input1 = Node::new(&[13, 33]).set_name("input1");
		let input2 = Node::new(&[13, 33]).set_name("input2");

		let output = cross_entropy_with_logits(&input1, &input2).unwrap();

		GradNumericTest::new(&output, &indexset![&input1, &input2])
			.step_size(1e-3)
			.tolerance(1e-4)
			.run();
	}
}
