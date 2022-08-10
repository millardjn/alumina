use alumina_core::{
	base_ops::{shape_constraint::same_shape, OpInstance, OpSpecification},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{merge_graphs, Graph, Node, NodeID, Op},
	shape::NodeAxis,
	shape_prop::ShapePropContext,
};
use indexmap::{indexset, IndexMap, IndexSet};
use ndarray::{ArrayViewD, ArrayViewMutD, Dimension, Zip};
use std::any::Any;

/// breshape the values of value_input to the shape of shape_input and return the result
pub fn reshape<I1, I2>(shape_input: I1, value_input: I2) -> Result<Node, OpBuildError>
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	let shape_input = shape_input.into();
	let value_input = value_input.into();
	let graph = merge_graphs(&[shape_input.graph(), value_input.graph()]);

	let output = graph
		.new_node(shape_input.shape())
		.set_name_unique(&format!("reshape({},{})", shape_input, value_input));

	let _op = same_shape(shape_input, output.clone())?;
	let _op = Reshape::new(value_input, output.clone()).build()?;

	Ok(output)
}

/// reshape the values of input to the existing output and return the Op
pub fn reshape_into<I, O>(input: I, output: O) -> Result<Op, OpBuildError>
where
	I: Into<Node>,
	O: Into<Node>,
{
	let input = input.into();
	let output = output.into();

	Reshape::new(input, output).build()
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct Reshape {
	output: Node,
	input: Node,
}

impl Reshape {
	pub fn new<I, O>(input: I, output: O) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		Reshape {
			input: input.into(),
			output: output.into(),
		}
	}
}

impl OpSpecification for Reshape {
	type InstanceType = ReshapeInstance;

	fn type_name(&self) -> &'static str {
		"Reshape"
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
		Ok(ReshapeInstance {
			input: self.input.id(),
			output: self.output.id(),
		})
	}
}

/// Broadcast Op, the value of the input is added to
#[derive(Clone, Debug)]
pub struct ReshapeInstance {
	input: NodeID,
	output: NodeID,
}

impl OpInstance for ReshapeInstance {
	fn type_name(&self) -> &'static str {
		"Reshape"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(Reshape {
			input: graph.node_from_id(self.input),
			output: graph.node_from_id(self.output),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		Reshape::new(ctx.grad_of(&self.output), ctx.grad_of(&self.input)).build()?;
		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		//let input_shape: NodeShape = ctx.input_shape(&self.input).slice().to_vec();

		let input_len: usize = ctx.input_shape(&self.input).slice().iter().product();

		let output_shape = ctx.output_shape(&self.output);
		let unknowns = output_shape.iter().filter(|e| !e.is_known()).count();
		if unknowns > 1 {
			return Err(format!(
				"Output Node '{}' had a shape ({}) with more than one unknown dimensions",
				ctx.node(&self.output),
				output_shape
			)
			.into());
		} else if unknowns == 0 {
			if output_shape.flat_size().as_known().unwrap() != input_len {
				return Err(format!("Output node '{}' had a shape ({}) with a different number of elements to the input node '{}' shape ({:?})", ctx.node(&self.output), output_shape, ctx.node(&self.input), ctx.input_shape(&self.input).slice()).into());
			} else {
				Ok(())
			}
		} else {
			let output_known_product: usize = output_shape.iter().filter_map(NodeAxis::as_known).product();
			let unknown = input_len / output_known_product;
			if unknown * output_known_product != input_len {
				return Err(format!("Output node '{}' had a shape ({}) with an unknown that could not be reconciled with the number of elements in the input node '{}' shape ({:?})", ctx.node(&self.output), output_shape, ctx.node(&self.input), ctx.input_shape(&self.input).slice()).into());
			}

			let mut output_shape = output_shape;
			for e in output_shape.slice_mut() {
				if !e.is_known() {
					*e = NodeAxis::known(unknown)
				}
			}

			ctx.broadcast_merge_output_shape(&self.output, &output_shape)
		}

		//ctx.broadcast_merge_output_shape(&self.output, &input_shape)
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		if ctx.shape(&self.input).iter().product::<usize>() != ctx.shape(&self.output).iter().product() {
			return Err(format!(
				"Input '{:?}' shape ({:?}) does not have the same number of elements as the output '{:?}' shape ({:?})",
				self.input,
				ctx.shape(&self.input),
				self.output,
				ctx.shape(&self.output)
			)
			.into());
		}

		if ctx.can_take(&self.input) && ctx.can_set(&self.output) {
			let array = ctx.take(&self.input);
			ctx.set(
				&self.output,
				array
					.as_standard_layout()
					.into_shape(ctx.shape(&self.output))
					.unwrap()
					.to_shared(),
			);
		} else {
			let input: ArrayViewD<f32> = ctx.get_input_standard(&self.input);
			let input = input.into_shape(ctx.shape(&self.output)).unwrap();
			let output: ArrayViewMutD<f32> = ctx.get_output(&self.output);

			Zip::from(output).and(input).par_for_each(|output, input| {
				*output += input;
			});
		}

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::{reshape, reshape_into};
	use alumina_core::{
		errors::{ExecError, ShapesError},
		graph::Node,
	};
	use alumina_test::grad_numeric_test::GradNumericTest;

	use ndarray::{arr0, arr1, arr2, arr3};

	#[test]
	fn forward_reshape_test_fail_large() {
		let input1 = Node::from(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).set_name("input1");
		let output = Node::new(&[-1, 49]).set_name("output");

		let op = reshape_into(input1, &output).unwrap();

		match output.calc() {
			Err(ExecError::Shape {
				error: ShapesError::ShapePropError { op: err_op, .. },
			}) if err_op == op => {},
			_ => panic!("wrong error kind"),
		}
	}

	#[test]
	fn forward_reshape_test_fail_small() {
		let input1 = Node::from(arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])).set_name("input1");
		let output = Node::new(&[2, -1]).set_name("output");

		let op = reshape_into(input1, &output).unwrap();

		match output.calc() {
			Err(ExecError::Shape {
				error: ShapesError::ShapePropError { op: err_op, .. },
			}) if err_op == op => {},
			_ => panic!("wrong error kind"),
		}
	}

	#[test]
	fn forward_test_reshape_fail_multi_unknown() {
		let input1 = Node::from(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).set_name("input1");
		let output = Node::new(&[-1, 2, -1]).set_name("output");

		let op = reshape_into(input1, &output).unwrap();

		match output.calc() {
			Err(ExecError::Shape {
				error: ShapesError::ShapePropError { op: err_op, .. },
			}) if err_op == op => {},
			_ => panic!("wrong error kind"),
		}
	}

	#[test]
	fn forward_reshape_test_flip() {
		let input1 = Node::from(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).set_name("input1");
		let output = Node::new(&[3, 2]).set_name("output");

		let _ = reshape_into(input1, &output).unwrap();

		let expected = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).into_dyn().to_shared();
		assert_eq!(expected, output.calc().unwrap());
	}

	#[test]
	fn forward_reshape_test_combine() {
		let input1 = Node::from(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).set_name("input1");
		let output = Node::new(&[6]).set_name("output");

		let _ = reshape_into(input1, &output).unwrap();

		let expected = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).into_dyn().to_shared();
		assert_eq!(expected, output.calc().unwrap());
	}

	#[test]
	fn forward_test_reshape_unknown() {
		let input1 = Node::from(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).set_name("input1");
		let output = Node::new(&[3, -1]).set_name("output");

		let _ = reshape_into(input1, &output).unwrap();

		let expected = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).into_dyn().to_shared();
		assert_eq!(expected, output.calc().unwrap());
	}

	#[test]
	fn forward_test_reshape_unknown2() {
		let input1 = Node::from(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).set_name("input1");
		let output = Node::new(&[-1, 3, 1]).set_name("output");

		let _ = reshape_into(input1, &output).unwrap();

		let expected = arr3(&[[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
			.into_dyn()
			.to_shared();
		assert_eq!(expected, output.calc().unwrap());
	}

	#[test]
	fn grad_numeric_test_reshape_into() {
		let input = Node::new(&[13, 33]).set_name("input");
		let output = Node::new(&[33, 1, 13]).set_name("output");

		let _ = reshape_into(&input, &output).unwrap();

		GradNumericTest::new(output, &[input]).run();
	}

	#[test]
	fn grad_numeric_test_reshape() {
		let input = Node::new(&[13, 1, 33]).set_name("input");
		let target = Node::new(&[1, 33, 13]).set_value(arr0(0.0)).set_name("target");

		let output = reshape(target, &input).unwrap();

		GradNumericTest::new(output, &[input]).run();
	}
}
