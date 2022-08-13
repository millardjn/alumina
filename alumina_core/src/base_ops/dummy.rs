use crate::{
	base_ops::{OpInstance, OpSpecification},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{Graph, Node, NodeID},
	shape_prop::ShapePropContext,
};
use indexmap::{IndexMap, IndexSet};
use std::any::Any;

#[derive(Default)]
pub struct DummyOp {
	inputs: IndexSet<Node>,
	outputs: IndexSet<Node>,
}

impl DummyOp {
	pub fn new() -> Self {
		DummyOp {
			inputs: IndexSet::new(),
			outputs: IndexSet::new(),
		}
	}

	pub fn input<I>(mut self, node: I) -> Self
	where
		I: Into<Node>,
	{
		let node = node.into();
		self.inputs.insert(node);
		self
	}

	pub fn output<O>(mut self, node: O) -> Self
	where
		O: Into<Node>,
	{
		let node = node.into();
		self.outputs.insert(node);
		self
	}
}

impl OpSpecification for DummyOp {
	type InstanceType = DummyOpInstance;

	fn op_type(&self) -> &'static str {
		"DummyOp"
	}

	fn inputs(&self) -> IndexSet<Node> {
		self.inputs.clone()
	}

	fn outputs(&self) -> IndexSet<Node> {
		self.outputs.clone()
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		DummyOp {
			inputs: self
				.inputs
				.iter()
				.map(|node| mapping.get(node).unwrap_or(node))
				.cloned()
				.collect(),
			outputs: self
				.outputs
				.iter()
				.map(|node| mapping.get(node).unwrap_or(node))
				.cloned()
				.collect(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(DummyOpInstance {
			inputs: self.inputs.iter().map(|node| node.id()).collect(),
			outputs: self.outputs.iter().map(|node| node.id()).collect(),
		})
	}
}

#[derive(Clone, Debug)]
pub struct DummyOpInstance {
	inputs: IndexSet<NodeID>,
	outputs: IndexSet<NodeID>,
}

impl OpInstance for DummyOpInstance {
	fn op_type(&self) -> &'static str {
		"DummyOp"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(DummyOp {
			inputs: self.inputs.iter().map(|&i| graph.node_from_id(i)).collect(),
			outputs: self.outputs.iter().map(|&o| graph.node_from_id(o)).collect(),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		self.inputs.clone()
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		self.outputs.clone()
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		// TODO turn NodeInner into a type and AsRef node as NodeInner for builders
		let grad_dummy = self
			.outputs
			.iter()
			.fold(DummyOp::new(), |dummy, node| dummy.input(ctx.grad_of(node)));
		let grad_dummy = self
			.inputs
			.iter()
			.fold(grad_dummy, |dummy, node| dummy.output(ctx.grad_of(node)));
		grad_dummy.build().expect("TODO");
		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		for node in &self.inputs() {
			ctx.input_shape(node);
		}

		for node in &self.outputs() {
			ctx.output_shape(node);
		}

		Ok(())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		for node in &self.inputs() {
			ctx.get_input(node);
		}

		for node in &self.outputs() {
			if ctx.is_required_output(node) {
				ctx.get_output(node);
			}
		}

		Ok(())
	}
}
