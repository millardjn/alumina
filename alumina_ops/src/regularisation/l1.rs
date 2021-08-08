use alumina_core::{
	base_ops::{OpInstance, OpSpecification},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{merge_node_graphs, Graph, Node, NodeID},
	shape::SCALAR,
	shape_prop::ShapePropContext,
};
use indexmap::{indexmap, indexset, IndexMap, IndexSet};
use ndarray::{ArrayViewD, ArrayViewMutD, Zip};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::any::Any;

/// Calculates the combined L1 norm of the input nodes, returning a scalar node.
pub fn l1<I, T: IntoIterator<Item = I>>(inputs: T) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let inputs: SmallVec<[Node; 16]> = inputs.into_iter().map(Into::into).collect();
	merge_node_graphs(&inputs);

	let output = Node::new(SCALAR).set_name_unique(&format!(
		"l1({})",
		inputs.iter().map(|n| n.name()).collect::<Vec<_>>().join(",")
	));

	let mut op = L1::new(output.clone());
	for input in inputs {
		op = op.input(input);
	}
	let _op = op.build()?;

	Ok(output)
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct L1 {
	inputs: IndexSet<Node>,
	output: Node,
}

impl L1 {
	pub fn new<O>(output: O) -> Self
	where
		O: Into<Node>,
	{
		let output = output.into();
		L1 {
			inputs: indexset![],
			output,
		}
	}

	pub fn input<I>(mut self, input: I) -> Self
	where
		I: Into<Node>,
	{
		let input = input.into();
		self.inputs.insert(input);
		self
	}
}

impl OpSpecification for L1 {
	type InstanceType = L1Instance;

	fn type_name(&self) -> &'static str {
		"L1"
	}

	fn inputs(&self) -> IndexSet<Node> {
		self.inputs.clone()
	}

	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			inputs: self
				.inputs
				.iter()
				.map(|i| mapping.get(i).unwrap_or(i))
				.cloned()
				.collect(),
			output: mapping.get(&self.output).unwrap_or(&self.output).clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(L1Instance {
			inputs: self.inputs.iter().map(Node::id).collect(),
			output: self.output.id(),
		})
	}
}

/// L1 OpInstance
#[derive(Clone, Debug)]
pub struct L1Instance {
	inputs: IndexSet<NodeID>,
	output: NodeID,
}

impl OpInstance for L1Instance {
	fn type_name(&self) -> &'static str {
		"L1"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(L1 {
			inputs: self.inputs.iter().map(|&i| graph.node_from_id(i)).collect(),
			output: graph.node_from_id(self.output),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		self.inputs.clone()
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		self.inputs()
			.iter()
			.fold(L1Back::new(ctx.grad_of(&self.output)), |op, input| {
				op.input_and_grad(ctx.node(input), ctx.grad_of(input))
			})
			.build()?;

		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		ctx.merge_output_shape(&self.output, &SCALAR.into())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let mut output: ArrayViewMutD<f32> = ctx.get_output(&self.output);
		assert_eq!(
			1,
			output.len(),
			"Alumina Bug: L1 ({}) output had shape greater than 1: {}",
			ctx.current_op(),
			output.len()
		);

		let inputs: SmallVec<[ArrayViewD<f32>; 16]> = self.inputs.iter().map(|input| ctx.get_input(input)).collect();

		output[[]] += inputs
			.par_iter()
			.with_max_len(1)
			.map(|input| input.view().into_par_iter().map(|&i| i.abs()).sum::<f32>())
			.sum::<f32>();

		Ok(())
	}
}

/// Optimised Backward pass for L1 Op.
///
/// Input/Output naming convention matches L1 Input/Outputs, i.e. output_grad is an input to this Op.
///
/// All inputs and grads must be unique.
#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct L1Back {
	input_and_grads: IndexMap<Node, Node>,
	output_grad: Node,
}

impl L1Back {
	pub fn new<I>(output_grad: I) -> Self
	where
		I: Into<Node>,
	{
		let output_grad = output_grad.into();
		L1Back {
			input_and_grads: indexmap![],
			output_grad,
		}
	}

	/// input is the input to the L1 Op and grad is the relevant grad node
	pub fn input_and_grad<I, G>(mut self, input: I, input_grad: G) -> Self
	where
		I: Into<Node>,
		G: Into<Node>,
	{
		let input = input.into();
		let input_grad = input_grad.into();
		self.input_and_grads.insert(input, input_grad);
		self
	}
}

impl OpSpecification for L1Back {
	type InstanceType = L1BackInstance;

	fn type_name(&self) -> &'static str {
		"L1Back"
	}

	fn inputs(&self) -> IndexSet<Node> {
		let mut inputs: IndexSet<Node> = self.input_and_grads.keys().cloned().collect();
		inputs.insert(self.output_grad.clone());
		inputs
	}

	fn outputs(&self) -> IndexSet<Node> {
		self.input_and_grads.values().cloned().collect()
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			output_grad: mapping.get(&self.output_grad).unwrap_or(&self.output_grad).clone(),
			input_and_grads: self
				.input_and_grads
				.iter()
				.map(|(i, g)| (mapping.get(i).unwrap_or(i).clone(), mapping.get(g).unwrap_or(g).clone()))
				.collect(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(L1BackInstance {
			input_and_grads: self.input_and_grads.iter().map(|(n1, n2)| (n1.id(), n2.id())).collect(),
			output_grad: self.output_grad.id(),
		})
	}
}

/// L1Back OpInstance
#[derive(Clone, Debug)]
pub struct L1BackInstance {
	input_and_grads: IndexMap<NodeID, NodeID>,
	output_grad: NodeID,
}

impl OpInstance for L1BackInstance {
	fn type_name(&self) -> &'static str {
		"L1Back"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(L1Back {
			input_and_grads: self
				.input_and_grads
				.iter()
				.map(|(&i, &g)| (graph.node_from_id(i), graph.node_from_id(g)))
				.collect(),
			output_grad: graph.node_from_id(self.output_grad),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		let mut inputs: IndexSet<NodeID> = self.input_and_grads.keys().cloned().collect();
		inputs.insert(self.output_grad);
		inputs
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		self.input_and_grads.values().cloned().collect()
	}

	fn gradient(&self, _ctx: &mut GradientContext) -> Result<(), GradientError> {
		Err(GradientError::Unimplemented)
	}

	fn propagate_shapes(&self, _ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		Ok(())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let output_grad = ctx.get_input(&self.output_grad);
		assert_eq!(
			1,
			output_grad.len(),
			"Alumina Bug: L1Back ({}) output_grad had shape greater than 1: {}",
			ctx.current_op(),
			output_grad.len()
		);
		let double_output_grad = output_grad[[]];

		for (input, input_grad) in &self.input_and_grads {
			if ctx.is_required_output(input_grad) {
				if ctx.can_take(input) && ctx.can_set(input_grad) {
					let mut input_arr = ctx.take(input);
					input_arr
						.par_iter_mut()
						.for_each(|input| *input = input.signum() * double_output_grad);
					ctx.set(input_grad, input_arr);
				} else {
					Zip::from(&mut ctx.get_output(input_grad))
						.and(&ctx.get_input(input))
						.par_for_each(|output, input| {
							*output += input.signum() * double_output_grad;
						});
				}
			}
		}

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::l1;
	use alumina_core::{graph::Node, init::uniform};
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};
	use indexmap::indexset;

	use ndarray::arr0;
	#[test]
	fn forward_test() {
		let input1 = Node::new(&[13, 33]).set_name("input1");
		let input2 = Node::new(&[13, 33]).set_name("input2");

		let output = l1(vec![&input1, &input2]).unwrap();

		input1.set_value(arr0(1.25));
		input2.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1072.5), ::std::f32::EPSILON * 10.0));

		input1.set_value(arr0(1.25));
		input2.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(879.45), ::std::f32::EPSILON * 10.0));

		input1.set_value(arr0(-0.8));
		input2.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(879.45), ::std::f32::EPSILON * 10.0));

		input1.set_value(arr0(-0.8));
		input2.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(686.4), ::std::f32::EPSILON * 10.0));
	}

	#[test]
	fn grad_numeric_test() {
		let input1 = Node::new(&[13, 33]).set_name("input1").set_init(uniform(-1.0, 1.0));
		let input2 = Node::new(&[13, 33]).set_name("input2").set_init(uniform(-1.0, 1.0));

		let output = l1(vec![&input1, &input2]).unwrap();

		GradNumericTest::new(&output, &indexset![&input1, &input2])
			.step_size(1e-3)
			.tolerance(4e-3)
			.run();
	}
}
