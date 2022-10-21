use alumina_core::{
	base_ops::{OpInstance, OpSpecification},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{merge_node_graphs, Graph, Node, NodeID},
	shape::{NodeShape, NodeAxis},
	shape_prop::ShapePropContext,
};
use indexmap::{indexset, IndexMap, IndexSet};
use ndarray::{ArrayViewMutD, Axis, Dimension, Slice};
use smallvec::SmallVec;
use std::any::Any;
use std::iter::once;

/// Concatenates the list of input nodes along the chosen axis
pub fn concat<I, T: IntoIterator<Item = I>>(inputs: T, axis: usize) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let inputs: SmallVec<[Node; 16]> = inputs.into_iter().map(Into::into).collect();
	merge_node_graphs(&inputs);

	let out_shape = if inputs.is_empty() {
		NodeShape::from(vec![0; axis])
	} else {
		let dims = inputs[0].shape().len();
		assert!(inputs.iter().all(|i| i.shape().len() == dims));

		let mut shape = NodeShape::from(vec![-1; dims]);
		shape.slice_mut()[axis] = NodeAxis::Known{val: 0};
		for i in &inputs {
			for (j, a) in i.shape().iter().enumerate() {
				if j == axis {
					shape.slice_mut()[j] = shape.slice()[j].add(a);
				} else {
					shape.slice_mut()[j] = shape.slice()[j]
						.merge(a)
						.map_err(|e| format!("Could not concat due to incompatible axes at index {}: {}", j, e))?;
				}
			}
		}
		shape
	};

	let output = Node::new(out_shape).set_name_unique(&format!(
		"concat({})",
		inputs.iter().map(|n| n.name()).collect::<Vec<_>>().join(",")
	));

	let mut op = Concat::new(output.clone(), axis);
	for input in inputs {
		op = op.input(input);
	}
	let _op = op.build()?;

	Ok(output)
}

pub struct Concat {
	inputs: Vec<Node>,
	output: Node,
	axis: usize,
}

impl Concat {
	pub fn new<O>(output: O, axis: usize) -> Self
	where
		O: Into<Node>,
	{
		let output = output.into();
		Concat {
			inputs: vec![],
			output,
			axis,
		}
	}

	/// Add another input node
	pub fn input<I>(mut self, input: I) -> Self
	where
		I: Into<Node>,
	{
		let input = input.into();
		self.inputs.push(input);
		self
	}
}

impl OpSpecification for Concat {
	type InstanceType = ConcatInstance;

	fn op_type(&self) -> &'static str {
		"Concat"
	}

	fn inputs(&self) -> IndexSet<Node> {
		self.inputs.iter().cloned().collect()
	}

	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			inputs: self
				.inputs
				.iter()
				.map(|i| mapping.get(i).unwrap_or(i).clone())
				.collect(),
			output: mapping.get(&self.output).unwrap_or(&self.output).clone(),
			axis: self.axis,
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(ConcatInstance {
			inputs: self.inputs.iter().map(Node::id).collect(),
			output: self.output.id(),
			axis: self.axis,
		})
	}
}

/// Concat OpInstance
#[derive(Clone, Debug)]
pub struct ConcatInstance {
	inputs: Vec<NodeID>,
	output: NodeID,
	axis: usize,
}

impl OpInstance for ConcatInstance {
	fn op_type(&self) -> &'static str {
		"Concat"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(Concat {
			inputs: self.inputs.iter().map(|&i| graph.node_from_id(i)).collect(),
			output: graph.node_from_id(self.output),
			axis: self.axis,
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		self.inputs.iter().cloned().collect()
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		self.inputs
			.iter()
			.fold(ConcatBack::new(ctx.grad_of(&self.output), self.axis), |op, input| {
				op.input_and_grad(ctx.node(input), ctx.grad_of(input))
			})
			.build()?;

		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		if self.inputs.is_empty() {
			ctx.merge_output_shape(&self.output, &NodeShape::from(vec![0; self.axis]))
		} else {
			let input_shapes: Vec<_> = self.inputs.iter().map(|input| ctx.input_shape(input).clone()).collect();
			let axis_sum = input_shapes.iter().map(|shape| shape[self.axis]).sum();
			for mut shape in input_shapes {
				shape[self.axis] = axis_sum;
				ctx.merge_output_shape(&self.output, &shape.slice().into())?;
			}
			Ok(())
		}
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let mut output: ArrayViewMutD<f32> = ctx.get_output(&self.output);

		let mut start = 0;
		for input in &self.inputs {
			let values = ctx.get_input(input);
			let axis_size = values.shape()[self.axis];
			let mut output_slice = output.slice_axis_mut(
				Axis(self.axis),
				Slice::new(start as isize, Some((start + axis_size) as isize), 1),
			);
			output_slice += &values;
			start += axis_size;
		}

		Ok(())
	}
}

/// Optimised Backward pass for Concat Op.
///
/// Input/Output naming convention matches Concat Input/Outputs, i.e. output_grad is an input to this Op.
///
/// All inputs and grads must be unique.
#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct ConcatBack {
	input_and_grads: Vec<(Node, Node)>,
	output_grad: Node,
	axis: usize,
}

impl ConcatBack {
	pub fn new<I>(output_grad: I, axis: usize) -> Self
	where
		I: Into<Node>,
	{
		let output_grad = output_grad.into();
		ConcatBack {
			input_and_grads: vec![],
			output_grad,
			axis,
		}
	}

	/// input is the input to the L1 Op and grad is the relevant grad node
	pub fn input_and_grad<I, G>(mut self, input: I, grad: G) -> Self
	where
		G: Into<Node>,
		I: Into<Node>,
	{
		self.input_and_grads.push((input.into(), grad.into()));
		self
	}
}

impl OpSpecification for ConcatBack {
	type InstanceType = ConcatBackInstance;

	fn op_type(&self) -> &'static str {
		"ConcatBack"
	}

	fn inputs(&self) -> IndexSet<Node> {
		self.input_and_grads
			.iter()
			.map(|(i, _)| i.clone())
			.chain(once(self.output_grad.clone()))
			.collect()
	}

	fn outputs(&self) -> IndexSet<Node> {
		self.input_and_grads.iter().map(|(_, g)| g.clone()).collect()
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			input_and_grads: self
				.input_and_grads
				.iter()
				.map(|(i, g)| (mapping.get(i).unwrap_or(i).clone(), mapping.get(g).unwrap_or(g).clone()))
				.collect(),
			output_grad: mapping.get(&self.output_grad).unwrap_or(&self.output_grad).clone(),
			axis: self.axis,
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(ConcatBackInstance {
			input_and_grads: self.input_and_grads.iter().map(|(i, g)| (i.id(), g.id())).collect(),
			output_grad: self.output_grad.id(),
			axis: self.axis,
		})
	}
}

/// ConcatBack OpInstance
#[derive(Clone, Debug)]
pub struct ConcatBackInstance {
	input_and_grads: Vec<(NodeID, NodeID)>,
	output_grad: NodeID,
	axis: usize,
}

impl OpInstance for ConcatBackInstance {
	fn op_type(&self) -> &'static str {
		"ConcatBack"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(ConcatBack {
			input_and_grads: self
				.input_and_grads
				.iter()
				.map(|&(i, g)| (graph.node_from_id(i), graph.node_from_id(g)))
				.collect(),
			output_grad: graph.node_from_id(self.output_grad),
			axis: self.axis,
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		self.input_and_grads
			.iter()
			.map(|(i, _)| *i)
			.chain(once(self.output_grad))
			.collect()
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		self.input_and_grads.iter().map(|(_, g)| *g).collect()
	}

	fn gradient(&self, _ctx: &mut GradientContext) -> Result<(), GradientError> {
		Err(GradientError::Unimplemented)
	}

	fn propagate_shapes(&self, _ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		Ok(())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let output_grad = ctx.get_input(&self.output_grad);

		let mut start = 0;
		let mut grad_map = IndexMap::new(); // store output references in a map to avoid potentially calling get_output() twice for the same node
		for (input, input_grad) in &self.input_and_grads {
			if ctx.is_required_output(input_grad) {
				let grad = grad_map.entry(input_grad).or_insert_with(|| ctx.get_output(input_grad));
				let axis_size = grad.shape()[self.axis];
				*grad += &output_grad.slice_axis(
					Axis(self.axis),
					Slice::new(start as isize, Some((start + axis_size) as isize), 1),
				);
				start += axis_size;
			} else {
				let axis_size = ctx.shape(input)[self.axis];
				start += axis_size;
			}
		}

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::concat;
	use alumina_core::{graph::Node, init::uniform};
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};
	use indexmap::indexset;

	use ndarray::arr3;

	#[test]
	fn forward_test() {
		let input1 = Node::new(&[2, -1, 3]).set_name("input1").set_value(arr3(&[
			[[6.0, 4.0, 0.0], [7.0, 7.0, 7.0], [2.0, 10.0, 3.0], [1.0, 2.0, 5.0]],
			[[1.0, 7.0, 4.0], [1.0, 10.0, 1.0], [0.0, 2.0, 9.0], [5.0, 9.0, 0.0]],
		]));

		let input2 = Node::new(&[2, 4, -1]).set_name("input2").set_value(arr3(&[
			[[4.0, 5.0, 5.0], [3.0, 0.0, 2.0], [8.0, 5.0, 8.0], [0.0, 1.0, 3.0]],
			[[9.0, 5.0, 5.0], [5.0, 5.0, 3.0], [2.0, 9.0, 5.0], [1.0, 5.0, 3.0]],
		]));

		let input3 = Node::new(&[2, 4, 3]).set_name("input3").set_value(arr3(&[
			[[8.0, 1.0, 9.0], [7.0, 1.0, 0.0], [0.0, 1.0, 2.0], [2.0, 1.0, 10.0]],
			[[2.0, 7.0, 3.0], [0.0, 8.0, 10.0], [9.0, 1.0, 5.0], [6.0, 1.0, 4.0]],
		]));

		let output1 = concat(vec![&input1, &input2, &input3, &input1], 0).unwrap();
		let output2 = concat(vec![&input1, &input2, &input3, &input1], 1).unwrap();

		let expected1 = arr3(&[
			[[6.0, 4.0, 0.0], [7.0, 7.0, 7.0], [2.0, 10.0, 3.0], [1.0, 2.0, 5.0]],
			[[1.0, 7.0, 4.0], [1.0, 10.0, 1.0], [0.0, 2.0, 9.0], [5.0, 9.0, 0.0]],
			[[4.0, 5.0, 5.0], [3.0, 0.0, 2.0], [8.0, 5.0, 8.0], [0.0, 1.0, 3.0]],
			[[9.0, 5.0, 5.0], [5.0, 5.0, 3.0], [2.0, 9.0, 5.0], [1.0, 5.0, 3.0]],
			[[8.0, 1.0, 9.0], [7.0, 1.0, 0.0], [0.0, 1.0, 2.0], [2.0, 1.0, 10.0]],
			[[2.0, 7.0, 3.0], [0.0, 8.0, 10.0], [9.0, 1.0, 5.0], [6.0, 1.0, 4.0]],
			[[6.0, 4.0, 0.0], [7.0, 7.0, 7.0], [2.0, 10.0, 3.0], [1.0, 2.0, 5.0]],
			[[1.0, 7.0, 4.0], [1.0, 10.0, 1.0], [0.0, 2.0, 9.0], [5.0, 9.0, 0.0]],
		]);

		let expected2 = arr3(&[
			[
				[6.0, 4.0, 0.0],
				[7.0, 7.0, 7.0],
				[2.0, 10.0, 3.0],
				[1.0, 2.0, 5.0],
				[4.0, 5.0, 5.0],
				[3.0, 0.0, 2.0],
				[8.0, 5.0, 8.0],
				[0.0, 1.0, 3.0],
				[8.0, 1.0, 9.0],
				[7.0, 1.0, 0.0],
				[0.0, 1.0, 2.0],
				[2.0, 1.0, 10.0],
				[6.0, 4.0, 0.0],
				[7.0, 7.0, 7.0],
				[2.0, 10.0, 3.0],
				[1.0, 2.0, 5.0],
			],
			[
				[1.0, 7.0, 4.0],
				[1.0, 10.0, 1.0],
				[0.0, 2.0, 9.0],
				[5.0, 9.0, 0.0],
				[9.0, 5.0, 5.0],
				[5.0, 5.0, 3.0],
				[2.0, 9.0, 5.0],
				[1.0, 5.0, 3.0],
				[2.0, 7.0, 3.0],
				[0.0, 8.0, 10.0],
				[9.0, 1.0, 5.0],
				[6.0, 1.0, 4.0],
				[1.0, 7.0, 4.0],
				[1.0, 10.0, 1.0],
				[0.0, 2.0, 9.0],
				[5.0, 9.0, 0.0],
			],
		]);

		assert!(output1
			.calc()
			.unwrap()
			.all_relatively_close(&expected1, ::std::f32::EPSILON));

		assert!(output2
			.calc()
			.unwrap()
			.all_relatively_close(&expected2, ::std::f32::EPSILON));
	}

	#[test]
	fn grad_numeric_test() {
		let input1 = Node::new(&[13, 7]).set_name("input1").set_init(uniform(-1.0, 1.0));
		let input2 = Node::new(&[13, 7]).set_name("input2").set_init(uniform(-1.0, 1.0));

		let output = concat(vec![&input1, &input2, &input1], 1).unwrap();

		GradNumericTest::new(&output, &indexset![&input1, &input2])
			.step_size(1e-3)
			.tolerance(1e-3)
			.run();
	}
}
