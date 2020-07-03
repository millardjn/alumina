use crate::{
	base_ops::OpInstance,
	errors::{ExecutionError, GradientError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::NodeInner,
	shape_prop::ShapePropContext,
};

use indexmap::IndexSet;

/// An OpInstance which does nothing
#[derive(Clone, Debug)]
pub struct NoOpInstance {}

impl OpInstance for NoOpInstance {
	fn type_name(&self) -> &'static str {
		"NoOp"
	}

	// fn clone_with_nodes_changed(
	// 	&self,
	// 	_mapping: IndexMap<NodeInner, NodeInner>,
	// ) -> Result<Box<OpInstance>, CloneError> {
	// 	Ok(Box::new(self.clone()))
	// }

	fn inputs(&self) -> IndexSet<NodeInner> {
		IndexSet::new()
	}

	fn outputs(&self) -> IndexSet<NodeInner> {
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
