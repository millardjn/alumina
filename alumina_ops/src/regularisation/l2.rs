use alumina_core::{
	base_ops::{OpBuilder, OpInstance},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{merge_node_graphs, Node, NodeID},
	shape::SCALAR,
	shape_prop::ShapePropContext,
};
use indexmap::{indexmap, indexset, IndexMap, IndexSet};

use ndarray::{ArrayViewD, ArrayViewMutD, Zip};
use rayon::prelude::*;
use smallvec::SmallVec;

/// Calculates the combined L2 norm of the input nodes, returning a scalar node.
pub fn l2<I, T: IntoIterator<Item = I>>(inputs: T) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let inputs: SmallVec<[Node; 16]> = inputs.into_iter().map(Into::into).collect();
	merge_node_graphs(&inputs);

	let output = Node::new(SCALAR).set_name_unique(&format!(
		"l2({})", inputs.iter().map(|n|n.name()).collect::<Vec<_>>().join(",")
	));

	let mut op = L2::new(output.clone());
	for input in inputs {
		op = op.input(input);
	}
	let _op = op.build()?;

	Ok(output)
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct L2 {
	inputs: IndexSet<Node>,
	output: Node,
}

impl L2 {
	pub fn new<O>(output: O) -> Self
	where
		O: Into<Node>,
	{
		let output = output.into();
		L2 {
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

impl OpBuilder for L2 {
	type InstanceType = L2Instance;

	fn type_name(&self) -> &'static str {
		"L2"
	}

	fn inputs(&self) -> IndexSet<Node> {
		self.inputs.clone()
	}

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
		Ok(L2Instance {
			inputs: self.inputs.iter().map(Node::id).collect(),
			output: self.output.id().clone(),
		})
	}
}

/// L2 OpInstance
#[derive(Clone, Debug)]
pub struct L2Instance {
	inputs: IndexSet<NodeID>,
	output: NodeID,
}

impl OpInstance for L2Instance {
	fn type_name(&self) -> &'static str {
		"L2"
	}

	// fn clone_with_nodes_changed(&self, mapping: IndexMap<NodeInner, NodeInner>) -> Result<Box<OpInstance>,
	// CloneError> { 	Ok(Box::new(ExpandDimsInstance {
	// 		input: mapping.get(&self.input).unwrap_or_else(|| &self.input).clone(),
	// 		output: mapping.get(&self.output).unwrap_or_else(|| &self.output).clone(),
	// 		//extra_axes: self.extra_axes.clone(),
	// 	}))
	// }

	fn inputs(&self) -> IndexSet<NodeID> {
		self.inputs.clone()
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output.clone()]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		self.inputs()
			.iter()
			.fold(L2Back::new(ctx.grad_of(&self.output)), |op, input| {
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
			"Alumina Bug: L2 ({}) output had shape greater than 1: {}",
			ctx.current_op(),
			output.len()
		);

		let inputs: SmallVec<[ArrayViewD<f32>; 16]> = self.inputs.iter().map(|input| ctx.get_input(input)).collect();

		output[[]] += inputs
			.par_iter()
			.with_max_len(1)
			.map(|input| input.view().into_par_iter().map(|&i| i * i).sum::<f32>())
			.sum::<f32>();

		Ok(())
	}
}

/// Optimised Backward pass for L2 Op.
///
/// Input/Output naming convention matches L2 Input/Outputs, i.e. output_grad is an input to this Op.
///
/// All inputs and grads must be unique.
#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct L2Back {
	input_and_grads: IndexMap<Node, Node>,
	output_grad: Node,
}

impl L2Back {
	pub fn new<I>(output_grad: I) -> Self
	where
		I: Into<Node>,
	{
		let output_grad = output_grad.into();
		L2Back {
			input_and_grads: indexmap![],
			output_grad,
		}
	}

	/// input is the input to the L2 Op and grad is the relevant grad node
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

impl OpBuilder for L2Back {
	type InstanceType = L2BackInstance;

	fn type_name(&self) -> &'static str {
		"L2Back"
	}

	fn inputs(&self) -> IndexSet<Node> {
		let mut inputs: IndexSet<Node> = self.input_and_grads.keys().cloned().collect();
		inputs.insert(self.output_grad.clone());
		inputs
	}

	fn outputs(&self) -> IndexSet<Node> {
		self.input_and_grads.values().cloned().collect()
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
		Ok(L2BackInstance {
			input_and_grads: self
				.input_and_grads
				.iter()
				.map(|(n1, n2)| (n1.id().clone(), n2.id().clone()))
				.collect(),
			output_grad: self.output_grad.id().clone(),
		})
	}
}

/// L2Back OpInstance
#[derive(Clone, Debug)]
pub struct L2BackInstance {
	input_and_grads: IndexMap<NodeID, NodeID>,
	output_grad: NodeID,
}

impl OpInstance for L2BackInstance {
	fn type_name(&self) -> &'static str {
		"L2Back"
	}

	// fn clone_with_nodes_changed(&self, mapping: IndexMap<NodeInner, NodeInner>) -> Result<Box<OpInstance>,
	// CloneError> { 	Ok(Box::new(ExpandDimsInstance {
	// 		input: mapping.get(&self.input).unwrap_or_else(|| &self.input).clone(),
	// 		output: mapping.get(&self.output).unwrap_or_else(|| &self.output).clone(),
	// 		//extra_axes: self.extra_axes.clone(),
	// 	}))
	// }

	fn inputs(&self) -> IndexSet<NodeID> {
		let mut inputs: IndexSet<NodeID> = self.input_and_grads.keys().cloned().collect();
		inputs.insert(self.output_grad.clone());
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
			"Alumina Bug: L2Back ({}) output_grad had shape greater than 1: {}",
			ctx.current_op(),
			output_grad.len()
		);
		let double_output_grad = 2.0 * output_grad[[]];

		for (input, input_grad) in &self.input_and_grads {
			if ctx.is_required_output(input_grad) {
				if ctx.can_take(input) && ctx.can_set(input_grad) {
					let mut input_arr = ctx.take(input);
					input_arr.par_iter_mut().for_each(|input| *input *= double_output_grad);
					ctx.set(input_grad, input_arr);
				} else {
					Zip::from(&mut ctx.get_output(input_grad))
						.and(&ctx.get_input(input))
						.par_apply(|output, &input| {
							*output += input * double_output_grad;
						});
				}
			}
		}

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::l2;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};
	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input1 = Node::new(&[13, 33]).set_name("input1");
		let input2 = Node::new(&[13, 33]).set_name("input2");

		let output = l2(vec![&input1, &input2]).unwrap();

		input1.set_value(arr0(1.25));
		input2.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1340.625), ::std::f32::EPSILON * 2.0));

		input1.set_value(arr0(1.25));
		input2.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(944.8725), ::std::f32::EPSILON * 2.0));

		input1.set_value(arr0(-0.8));
		input2.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(944.8725), ::std::f32::EPSILON * 2.0));

		input1.set_value(arr0(-0.8));
		input2.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(549.12), ::std::f32::EPSILON * 4.0));
	}

	#[test]
	fn grad_numeric_test() {
		let input1 = Node::new(&[13, 33]).set_name("input1");
		let input2 = Node::new(&[13, 33]).set_name("input2");

		let output = l2(vec![&input1, &input2]).unwrap();

		GradNumericTest::new(&output, &indexset![&input1, &input2])
			.tolerance(2e-4)
			.run();
	}
}
