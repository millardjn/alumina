use graph::{GraphDef, GraphShapes, ErrorKind, Result};
use id::{NodeID, DataID, OpID, PassID};
use storage::Storage;
use ops::{standard_op_name, Op, OpInstance, Pass};
use ops::loss::LossType;
use std::any::Any;


/// An `Op` which implements the Cross entropy Loss
///
/// This op expects one input tensor of values in the range (0, 1), and a second of 0 or 1 labels.
///
/// By default this `Op` has no output and will generate loss and gradients.
///
/// If `output()` is set to a node of size 1, the Cross Entropy will be written to that Node, and the gradient will be backprop'd from the output node.
///
/// If `separate_loss()` is set a scalar node will be added to the graph, and a `Loss` Op attached to it.
#[must_use]
#[derive(Clone, Debug)]
pub struct CrossEntropy {
	logits_id: NodeID,
	labels_id: NodeID,
	output: Option<NodeID>,
	multiplier: f32,
	name: Option<String>,
}

impl CrossEntropy {
	pub fn new(logits_id: &NodeID, labels_id: &NodeID) -> Self{
		CrossEntropy{
			logits_id: logits_id.clone(),
			labels_id: labels_id.clone(),
			output: None,
			multiplier: 1.0,
			name: None,
		}
	}
	
	/// If set this `Op` will output to the supplied node, and will no longer generate loss and gradients.
	///
	/// The output node must have size 1.
	/// Default: None.
	pub fn output(mut self, output: &NodeID) -> Self {
		self.output = Some(output.clone());
		self
	}

	/// Loss is of the form `multiplier * x.dot(x)`
	pub fn multiplier(mut self, multiplier: f32) -> Self {
		self.multiplier = multiplier;
		self
	}
}

impl Op for CrossEntropy {
	type InstanceType = CrossEntropyInstance;

	fn type_name(&self) -> &'static str {
		"CrossEntropy"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef) -> Result<Self::InstanceType> {

		let name =  if let Some(ref output_id) = self.output {
			standard_op_name(&self, &self.name, graph, &[self.logits_id.clone(), self.labels_id.clone()], &[output_id.clone()])
		} else {
			standard_op_name(&self, &self.name, graph, &[self.logits_id.clone(), self.labels_id.clone()], &[])
		};

		let loss_type = if let Some(output_id) = self.output {
			LossType::Output{
				output_id: output_id.clone(),
				forward_id: graph.add_pass(CrossEntropyForward::new(
					self.multiplier,
					self.logits_id.clone(),
					self.labels_id.clone(),
					output_id.clone())),
				backward_id: graph.add_pass(CrossEntropyBackward::new(
					self.multiplier,
					self.logits_id.clone(),
					self.labels_id.clone(),
					output_id.clone())),
			}
		} else {
			LossType::Joint{
				pass_id: graph.add_pass(CrossEntropyJointPass::new(
					self.multiplier,
					self.logits_id.clone(),
					self.labels_id.clone()))
			}
		};

		Ok(CrossEntropyInstance{
			name: name,
			multiplier: self.multiplier,
			logits_id: self.logits_id.clone(),
			labels_id: self.labels_id.clone(),
			loss_type: loss_type,
		})
	}
}


#[derive(Clone, Debug)] 
pub struct CrossEntropyInstance {
	name: String,
	multiplier: f32,
	logits_id: NodeID,
	labels_id: NodeID,
	loss_type: LossType,
}

impl OpInstance for CrossEntropyInstance {
	fn name(&self) -> &str {&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		match &self.loss_type {
			&LossType::Joint{..} => (vec![self.logits_id.clone(), self.labels_id.clone()], vec![]),
			&LossType::Output{ref output_id, ..} => (vec![self.logits_id.clone(), self.labels_id.clone()], vec![output_id.clone()]),
		}
	}

	fn inner_passes(&self) -> Vec<PassID> {
		match &self.loss_type {
			&LossType::Joint{ref pass_id} => vec![pass_id.clone()],
			&LossType::Output{ref forward_id, ref backward_id, ..} => vec![forward_id.clone(), backward_id.clone()],
		}
	}

	fn inner_ops(&self) -> Vec<OpID> {
		vec![]
	}

	fn inner_nodes(&self) -> Vec<NodeID> {
		vec![]
	}

	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{
		if let &LossType::Output{ref output_id, ..} = &self.loss_type {
			let logits_shape = shapes.get_shape(&self.logits_id).clone();
			{
				let labels_shape = shapes.get_shape(&self.labels_id);
				ensure!(&logits_shape == labels_shape, "Shape of logits did not match shape of labels");
			}

			shapes.merge_with(output_id, &logits_shape)
		} else {
			Ok(())
		}
	}
}


#[derive(Clone, Debug)]
struct CrossEntropyJointPass {
	multiplier: f32,
	logits_id: NodeID,
	labels_id: NodeID,
}

impl CrossEntropyJointPass {
	pub fn new(multiplier: f32, logits_id: NodeID, labels_id: NodeID) -> Self {
		CrossEntropyJointPass {
			multiplier,
			logits_id,
			labels_id,
		}
	}
}

impl Pass for CrossEntropyJointPass {
	fn type_name(&self) -> &'static str {"CrossEntropyJointPass"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.logits_id.value_id(), self.labels_id.value_id()],
		vec![self.logits_id.gradient_id(), self.labels_id.gradient_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let logits_val = data.get(&self.logits_id.value_id())?;
		let labels_val = data.get(&self.labels_id.value_id())?;

		ensure!(
			labels_val.shape() == logits_val.shape(),
			ErrorKind::PassError(self.name(), format!("labels shape: {:?} did not match logits shape: {:?}", labels_val.shape(), logits_val.shape()))
			);


		let logits_val = logits_val.as_slice().unwrap();
		let labels_val = labels_val.as_slice().unwrap();

		let n = logits_val.len();
		assert!(labels_val.len() == n);
		
		let multiplier = self.multiplier;

		let mut error = 0.0;

		if data.is_required(&self.logits_id.gradient_id()) && data.is_required(&self.labels_id.gradient_id()) {
			let mut logits_grad = data.get_mut(&self.logits_id.gradient_id())?;
			let logits_grad = logits_grad.as_slice_mut().unwrap();
			let mut labels_grad = data.get_mut(&self.labels_id.gradient_id())?;
			let labels_grad = labels_grad.as_slice_mut().unwrap();
			assert!(logits_grad.len() == n);
			assert!(labels_grad.len() == n);

			for i in 0..n {
				error += -labels_val[i] * logits_val[i].ln() * multiplier;
				logits_grad[i] += -labels_val[i] * multiplier / logits_val[i];
				labels_grad[i] += - logits_val[i].ln() * multiplier;
			}

		} else if data.is_required(&self.logits_id.gradient_id()) {
			let mut logits_grad = data.get_mut(&self.logits_id.gradient_id())?;
			let logits_grad = logits_grad.as_slice_mut().unwrap();
			assert!(logits_grad.len() == n);


			for i in 0..n {
				error += -labels_val[i] * logits_val[i].ln() * multiplier;
				logits_grad[i] += -labels_val[i] * multiplier / logits_val[i];
			}

		} else if data.is_required(&self.labels_id.gradient_id()) {
			let mut labels_grad = data.get_mut(&self.labels_id.gradient_id())?;
			let labels_grad = labels_grad.as_slice_mut().unwrap();
			assert!(labels_grad.len() == n);

			for i in 0..n {
				error += -labels_val[i] * logits_val[i].ln() * multiplier;
				labels_grad[i] += - logits_val[i].ln() * multiplier;
			}
		}

		data.loss_add(error);

		Ok(Box::new(()))
	}
}


#[derive(Clone, Debug)]
struct CrossEntropyForward {
	multiplier: f32,
	logits_id: NodeID,
	labels_id: NodeID,
	output_id: NodeID,
}

impl CrossEntropyForward {
	pub fn new(multiplier: f32, logits_id: NodeID, labels_id: NodeID, output_id: NodeID) -> Self {
		CrossEntropyForward {
			multiplier,
			logits_id,
			labels_id,
			output_id,
		}
	}
}

impl Pass for CrossEntropyForward {
	fn type_name(&self) -> &'static str {"CrossEntropyForward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.logits_id.value_id(), self.labels_id.value_id()],
		vec![self.output_id.value_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let logits_val = data.get(&self.logits_id.value_id())?;
		let labels_val = data.get(&self.labels_id.value_id())?;
		let mut output_val = data.get_mut(&self.output_id.value_id())?;

		ensure!(
			labels_val.shape() == logits_val.shape(),
			ErrorKind::PassError(self.name(), format!("labels shape: {:?} did not match logits shape: {:?}", labels_val.shape(), logits_val.shape()))
			);

		let logits_val = logits_val.as_slice().unwrap();
		let labels_val = labels_val.as_slice().unwrap();
		let output_val = output_val.as_slice_mut().unwrap();

		let n = logits_val.len();
		assert!(labels_val.len() == n);
		assert!(output_val.len() == n);

		let multiplier = self.multiplier;

		for i in 0..n {
			output_val[i] += -labels_val[i] * logits_val[i].ln() * multiplier;
		}

		Ok(Box::new(()))
	}
}

#[derive(Clone, Debug)]
struct CrossEntropyBackward {
	multiplier: f32,
	logits_id: NodeID,
	labels_id: NodeID,
	output_id: NodeID,
}

impl CrossEntropyBackward {
	pub fn new(multiplier: f32, logits_id: NodeID, labels_id: NodeID, output_id: NodeID) -> Self {
		CrossEntropyBackward {
			multiplier,
			logits_id,
			labels_id,
			output_id,
		}
	}
}

impl Pass for CrossEntropyBackward {
	fn type_name(&self) -> &'static str {"CrossEntropyBackward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.logits_id.value_id(), self.labels_id.value_id(), self.output_id.gradient_id()],
		vec![self.logits_id.gradient_id(), self.labels_id.gradient_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let logits_val = data.get(&self.logits_id.value_id())?;
		let labels_val = data.get(&self.labels_id.value_id())?;
		let output_grad = data.get(&self.output_id.gradient_id())?;

		ensure!(
			labels_val.shape() == logits_val.shape(),
			ErrorKind::PassError(self.name(), format!("labels shape: {:?} did not match logits shape: {:?}", labels_val.shape(), logits_val.shape()))
			);

		let logits_val = logits_val.as_slice().unwrap();
		let labels_val = labels_val.as_slice().unwrap();
		let output_grad = output_grad.as_slice().unwrap();

		let n = logits_val.len();
		assert!(labels_val.len() == n);
		
		let multiplier = self.multiplier;

		if data.is_required(&self.logits_id.gradient_id()) && data.is_required(&self.labels_id.gradient_id()) {
			let mut logits_grad = data.get_mut(&self.logits_id.gradient_id())?;
			let logits_grad = logits_grad.as_slice_mut().unwrap();
			let mut labels_grad = data.get_mut(&self.labels_id.gradient_id())?;
			let labels_grad = labels_grad.as_slice_mut().unwrap();
			assert!(logits_grad.len() == n);
			assert!(labels_grad.len() == n);
			assert!(output_grad.len() == n);

			for i in 0..n {
				logits_grad[i] += -labels_val[i] * multiplier / logits_val[i] * output_grad[i];
				labels_grad[i] += - logits_val[i].ln() * multiplier * output_grad[i];
			}

		} else if data.is_required(&self.logits_id.gradient_id()) {
			let mut logits_grad = data.get_mut(&self.logits_id.gradient_id())?;
			let logits_grad = logits_grad.as_slice_mut().unwrap();
			assert!(logits_grad.len() == n);
			assert!(output_grad.len() == n);

			for i in 0..n {
				logits_grad[i] += -labels_val[i] * multiplier / logits_val[i] * output_grad[i];
			}

		} else if data.is_required(&self.labels_id.gradient_id()) {
			let mut labels_grad = data.get_mut(&self.labels_id.gradient_id())?;
			let labels_grad = labels_grad.as_slice_mut().unwrap();
			assert!(labels_grad.len() == n);
			assert!(output_grad.len() == n);

			for i in 0..n {
				labels_grad[i] += - logits_val[i].ln() * multiplier * output_grad[i];
			}
		}

		Ok(Box::new(()))
	}
}


#[test]
fn test_cross_entropy_backprop(){
	_cross_entropy_backprop().unwrap();
}

fn _cross_entropy_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ordermap::OrderMap;
	use ops::activ::logistic::Logistic;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "logistic", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "input2", tag![])?;

	let _o1 = g.new_op(Logistic::new(&node1, &node2), tag![])?;
	let _o2 = g.new_op(CrossEntropy::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 2;
	let tolerance = 0.005;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}

#[test]
fn test_cross_entropy_output_backprop(){
	_cross_entropy_output_backprop().unwrap();
}

fn _cross_entropy_output_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ordermap::OrderMap;
	use ops::activ::logistic::Logistic;
	use ops::loss::proportional::Proportional;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "logits", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "labels", tag![])?;

	let node4 = g.new_node(shape![7, 5, 16], "output", tag![])?;

	let _o1 = g.new_op(Logistic::new(&node1, &node2), tag![])?;
	let _o2 = g.new_op(CrossEntropy::new(&node2, &node3).output(&node4), tag![])?;
	let _o3 = g.new_op(Proportional::new(&node4), tag![])?;

	let iters = 100;
	let failures = 2;
	let tolerance = 0.005;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}