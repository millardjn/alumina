use crate::{
	base_ops::{OpBuilder, OpInstance},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{Node, NodeInner},
	shape::NodeShape,
	shape_prop::ShapePropContext,
};
use indexmap::{indexset, IndexSet};
use ndarray::Zip;

/// Produces and output node with the given shape, and an op to fill it elementwise with the provided value.
///
/// The output node has the same shape as the input.
pub fn fill<S: Into<NodeShape>>(value: f32, shape: S) -> Result<Node, OpBuildError> {
	let output = Node::new(shape).set_name_unique("fill()");
	let _op = Fill::new(&output, value).build()?;
	Ok(output)
}

/// Fillls the provided output elementwise with the provided value, then returns the same output Node.
///
/// The output node has the same shape as the input.
pub fn fill_into<O: Into<Node>>(value: f32, output: O) -> Result<Node, OpBuildError> {
	let output = output.into();
	let _op = Fill::new(output.clone(), value).build()?;
	Ok(output)
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct Fill {
	output: Node,
	value: f32,
}

impl Fill {
	pub fn new<O>(output: O, value: f32) -> Self
	where
		O: Into<Node>,
	{
		let output = output.into();
		Fill { output, value }
	}
}

impl OpBuilder for Fill {
	type InstanceType = FillInstance;

	fn type_name(&self) -> &'static str {
		"Fill"
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node> {
		indexset![]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
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
		Ok(FillInstance {
			output: self.output.inner().clone(),
			value: self.value,
		})
	}
}

/// Elementwise Op, the value of the input is added to
#[derive(Clone, Debug)]
pub struct FillInstance {
	output: NodeInner,
	value: f32,
}

impl OpInstance for FillInstance {
	fn type_name(&self) -> &'static str {
		"Fill"
	}

	// fn clone_with_nodes_changed(&self, mapping: IndexMap<NodeInner, NodeInner>) -> Result<Box<OpInstance>,
	// CloneError> { 	Ok(Box::new(AddInstance {
	// 		input: mapping.get(&self.input).unwrap_or_else(|| &self.input).clone(),
	// 		output: mapping.get(&self.output).unwrap_or_else(|| &self.output).clone(),
	// 		//extra_axes: self.extra_axes.clone(),
	// 	}))
	// }

	fn inputs(&self) -> IndexSet<NodeInner> {
		indexset![]
	}

	fn outputs(&self) -> IndexSet<NodeInner> {
		indexset![self.output.clone()]
	}

	fn gradient(&self, _ctx: &mut GradientContext) -> Result<(), GradientError> {
		Ok(())
	}

	fn propagate_shapes(&self, _ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		Ok(())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		Zip::from(&mut ctx.get_output(&self.output)).par_apply(|output| {
			*output += self.value;
		});
		Ok(())
	}
}

// #[cfg(test)]
// mod tests {
// 	use crate::{base_ops::fill::fill, util::relatively_close::RelClose};
// 	use ndarray::arr0;

// 	#[test]
// 	fn forward_test() {
// 		let output = fill(1.25, &[13, 33]).unwrap();

// 		assert!(output
// 			.calc()
// 			.unwrap()
// 			.all_relatively_close(&arr0(1.25), ::std::f32::EPSILON));
// 	}
// }
