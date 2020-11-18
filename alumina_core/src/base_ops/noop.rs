use crate::{
	base_ops::{OpSpecification, OpInstance},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{Node, NodeID, Graph},
	shape_prop::ShapePropContext,
};
use indexmap::{IndexSet, IndexMap};
use std::any::Any;

struct NoOp {

}

impl OpSpecification for NoOp {
	type InstanceType = NoOpInstance;

	fn type_name(&self) -> &'static str {
		"NoOp"
	}

	fn inputs(&self) -> IndexSet<Node> {
		IndexSet::new()
	}

	fn outputs(&self) -> IndexSet<Node> {
		IndexSet::new()
	}

	fn clone_with_nodes_changed(&self, _mapping: &IndexMap<Node, Node>) -> Self {
		NoOp {}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(NoOpInstance {})
	}
}

/// An OpInstance which does nothing
#[derive(Clone, Debug)]
pub struct NoOpInstance {}

impl OpInstance for NoOpInstance {
	fn type_name(&self) -> &'static str {
		"NoOp"
	}

	fn as_specification(&self, _graph: &Graph) -> Box<dyn Any> {
		Box::new(NoOp {})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		IndexSet::new()
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		IndexSet::new()
	}

	fn gradient(&self, _ctx: &mut GradientContext) -> Result<(), GradientError> {
		Ok(())
	}

	fn propagate_shapes(&self, _ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		// shapes: &mut Shapes
		Ok(())
	}

	fn execute(&self, _ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		Ok(())
	}
}
