use alumina_core::{
	base_ops::{OpInstance, OpSpecification},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{Graph, Node, NodeID},
	shape_prop::ShapePropContext,
};
use indexmap::{indexset, IndexMap, IndexSet};
use ndarray::Dimension;
use std::any::Any;

/// Returns the integer location of the maximum for each lane in the provided axis.
///
/// The output node has the shape of the input, but with the axis removed.
pub fn shape_of<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let output = input
		.graph()
		.new_node(::std::iter::once(input.shape().len()).into())
		.set_name_unique(&format!("shape_of({})", input));

	let _op = ShapeOf::new(input, output.clone()).build()?;

	Ok(output)
}

/// `ShapeOf` `OpBuilder`
#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct ShapeOf {
	input: Node,
	output: Node,
}

impl ShapeOf {
	pub fn new<I, O>(input: I, output: O) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		let input = input.into();
		let output = output.into();
		ShapeOf { input, output }
	}
}

impl OpSpecification for ShapeOf {
	type InstanceType = ShapeOfInstance;

	fn type_name(&self) -> &'static str {
		"ShapeOf"
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input.clone()]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			input: mapping.get(&self.input).unwrap_or(&self.input).clone(),
			output: mapping.get(&self.output).unwrap_or(&self.output).clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(ShapeOfInstance {
			input: self.input.id().clone(),
			output: self.output.id().clone(),
		})
	}
}

/// ShapeOf OpInstance,
#[derive(Clone, Debug)]
pub struct ShapeOfInstance {
	input: NodeID,
	output: NodeID,
}

impl OpInstance for ShapeOfInstance {
	fn type_name(&self) -> &'static str {
		"ShapeOf"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(ShapeOf {
			input: graph.node_from_id(self.input),
			output: graph.node_from_id(self.output),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input.clone()]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output.clone()]
	}

	fn gradient(&self, _ctx: &mut GradientContext) -> Result<(), GradientError> {
		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		ctx.merge_output_shape(
			&self.output,
			&::std::iter::once(&ctx.input_shape(&self.input).ndim()).into(),
		)
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let input = ctx.get_input(&self.input);
		let output = ctx.get_output(&self.output);

		for (&i, o) in input.shape().iter().zip(output) {
			*o += i as f32;
		}

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::shape_of;
	use alumina_core::graph::Node;
	use alumina_test::relatively_close::RelClose;
	use ndarray::{arr0, arr1};

	#[test]
	fn shape_of_test() {
		let input = Node::new(&[5, 7, 13, 31]).set_name("input").set_value(arr0(0.0));

		let output = shape_of(&input).unwrap();

		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr1(&[5.0, 7.0, 13.0, 31.0]), ::std::f32::EPSILON));
	}
}
