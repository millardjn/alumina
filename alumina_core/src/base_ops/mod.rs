pub mod apply;
pub mod dummy;
pub mod fill;
pub mod noop;
pub mod shape_constraint;

use crate::errors::{ExecutionError, GradientError, ShapePropError};
use crate::graph::NodeID;
use crate::{
	errors::OpBuildError,
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{merge_node_graphs, Graph, Node, Op},
	shape_prop::ShapePropContext,
};
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use std::any::Any;
use std::fmt;
use std::sync::Arc;

/// Generated default unique names for `Op`s
///
/// If a name isn't set on an `OpBuilder` a default name will be generated using the `type_name()` and the names of
/// input and output nodes. Similar for to: `format!("{}({},{}=>{},{}){}" type_name(), i, node1_name, node2_name,
/// node3_name, node4_name)` e.g. `Dummy0(node1,node2=>node3,node4)` where i is incremented until a unique/unused name
/// is found.
pub fn standard_op_name<O: OpSpecification>(graph: &Graph, op: &O) -> String {
	standard_op_name_inner(graph, op.type_name().to_string(), op.inputs(), op.outputs())
}

/// Split into inner to avoid monomorphising
fn standard_op_name_inner(
	graph: &Graph,
	mut new_name: String,
	inputs: IndexSet<Node>,
	outputs: IndexSet<Node>,
) -> String {
	new_name.push('(');

	let input_names = Itertools::intersperse(inputs.iter().map(|node| node.name()), ",".to_string());
	new_name.extend(input_names);

	new_name.push_str("=>");

	let output_names = Itertools::intersperse(outputs.iter().map(|node| node.name()), ",".to_string());
	new_name.extend(output_names);

	new_name.push(')');

	if graph.ops_named(&new_name).is_empty() {
		return new_name;
	}

	let mut i = 1;
	loop {
		let next_op_name = format!("{}{}", new_name, i);
		let list = graph.ops_named(&next_op_name);
		if list.is_empty() {
			return next_op_name;
		}
		i += 1
	}
}

pub trait OpSpecification: Any + Sized {
	type InstanceType: OpInstance;

	fn type_name(&self) -> &'static str;

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node>;

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<Node>;

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self; //, CloneError>;

	/// Construct the instance of this op
	///
	/// This method is not responsible for merging the graphs of input and output nodes.
	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError>;

	fn build(self) -> Result<Op, OpBuildError> {
		let graph = merge_node_graphs(self.inputs().into_iter().chain(self.outputs()));
		let name = standard_op_name(&graph, &self);

		let instance = self.build_instance()?;

		Ok(graph.new_op(Arc::new(instance)).set_name(name))
	}
}

/// An OpInstance should not behave as though it contains internal state, i.e. state as as an optimisation only.
///
/// No `OpInstance` should retain any reference to its containing graph as this will prevent deallocation.
/// This includes `Node`s, instead `InnerRef<NodeInner>` should be used to identify specific inputs and outputs.
pub trait OpInstance: fmt::Debug + Any + Send + Sync {
	///
	fn type_name(&self) -> &'static str;

	// Create a new OpInstance with nodes switched out
	fn as_specification(&self, graph: &Graph) -> Box<dyn Any>;

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<NodeID>;

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<NodeID>;

	/// Test
	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError>;

	/// This method is called to allow the `Op` to impose constraints on the shape of outputs based on the shape of
	/// inputs.
	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError>;

	/// Executes the operation, updating the outputs states
	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError>;
}

// /// Cloneable trait object workaround from DK : http://stackoverflow.com/questions/30353462/how-to-clone-a-struct-storing-a-trait-object
// pub trait OpClone {
// 	fn clone_box(&self) -> Box<dyn OpInstance>;
// }

// impl<T> OpClone for T
// where
// 	T: 'static + OpInstance + Clone,
// {
// 	fn clone_box(&self) -> Box<dyn OpInstance> {
// 		Box::new(self.clone())
// 	}
// }

// // We can now implement Clone manually by forwarding to clone_box.
// impl Clone for Box<dyn OpInstance> {
// 	fn clone(&self) -> Box<dyn OpInstance> {
// 		self.clone_box()
// 	}
// }
