use alumina_core::{
	base_ops::{OpSpecification, OpInstance},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{Node, NodeID, Graph},
	shape_prop::ShapePropContext,
};
use indexmap::{indexset, IndexSet};
use std::any::Any;

/// Calculates from the input a result where the axes of the ndarray have been rearranged (permuted), returning an
/// output with the same number of axes.
///
/// An input with the shape [3, 5, 7] combined with a permutation of [1, 2, 0] results in a output shape of [5, 7, 3].
///
/// # Error
///  * Err if permutation length is not equal to input length
///  * Err if permutation contains duplicate values
pub fn permute_axes<I>(input: I, permutation: &[usize]) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let input_shape = input.shape();

	let output = input
		.graph()
		.new_node(permutation.iter().map(|&i| &input_shape.slice()[i]).into())
		.set_name_unique(&format!("permute_axes({})", input));

	let _op = PermuteAxes::new(&input, &output, permutation).build()?;

	Ok(output)
}

/// Calculates from the input a result where the axes of the ndarray have been reversed (transposed), returning an
/// output with the same number of axes.
///
/// An input with the shape [3, 5, 7] results in a output shape of [7, 5, 3].
pub fn transpose<I>(input: I) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let input_shape = input.shape();

	let permutation: Vec<usize> = (0..input.shape().len()).rev().collect();

	let output = input
		.graph()
		.new_node(permutation.iter().map(|&i| &input_shape.slice()[i]).into())
		.set_name_unique(&format!("transpose({})", input));

	let _op = PermuteAxes::new(&input, &output, &permutation).build()?;

	Ok(output)
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct PermuteAxes {
	input: Node,
	output: Node,
	permutation: Vec<usize>,
}

impl PermuteAxes {
	pub fn new<I, O>(input: I, output: O, permutation: &[usize]) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		PermuteAxes {
			input: input.into(),
			output: output.into(),
			permutation: permutation.to_vec(),
		}
	}
}

impl OpSpecification for PermuteAxes {
	type InstanceType = PermuteAxesInstance;

	fn type_name(&self) -> &'static str {
		"PermuteAxes"
	}

	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input.clone()]
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
		if self.permutation.len() != self.input.shape().len() {
			return Err(format!(
				"input shape length ({}), output shape length ({}), permutation length ({}) must all be equal",
				self.input.shape().len(),
				self.output.shape().len(),
				self.permutation.len()
			)
			.into());
		}

		let mut used = vec![false; self.input.shape().len()];
		for &p in &self.permutation {
			if p > used.len() {
				return Err(format!(
					"permutation contains value larger than input shape size ({} > {})",
					p,
					self.input.shape().len(),
				)
				.into());
			}
			if used[p] {
				return Err(format!("permutation contains duplicate value {}", p,).into());
			}
			used[p] = true;
		}

		Ok(PermuteAxesInstance {
			input: self.input.id().clone(),
			output: self.output.id().clone(),
			permutation: self.permutation,
		})
	}
}

/// PermuteAxes OpInstance
#[derive(Clone, Debug)]
pub struct PermuteAxesInstance {
	input: NodeID,
	output: NodeID,
	permutation: Vec<usize>,
}

impl OpInstance for PermuteAxesInstance {
	fn type_name(&self) -> &'static str {
		"PermuteAxes"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(PermuteAxes {
			input: graph.node_from_id(self.input),
			output: graph.node_from_id(self.output),
			permutation: self.permutation.clone(),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input.clone()]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output.clone()]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		let mut inverse_permutation = vec![0; self.permutation.len()];
		for i in 0..self.permutation.len() {
			inverse_permutation[self.permutation[i]] = i;
		}

		let _op = PermuteAxes::new(
			ctx.grad_of(&self.output),
			ctx.grad_of(&self.input),
			&inverse_permutation,
		)
		.build()?;

		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let input_shape = ctx.input_shape(&self.input);

		let output_shape = &self.permutation.iter().map(|&i| input_shape[i]).into();

		ctx.merge_output_shape(&self.output, output_shape)
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		if ctx.can_take(&self.input) && ctx.can_set(&self.output) {
			let input = ctx.take(&self.input);
			ctx.set(&self.output, input.permuted_axes(self.permutation.as_slice()));
		} else {
			let mut output = ctx.get_output(&self.output);
			let input = ctx.get_input(&self.input);
			output += &input.permuted_axes(self.permutation.as_slice());
		}

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::{permute_axes, transpose};
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr0;

	#[test]
	fn forward_test() {
		let input = Node::new(&[3, 5, 7]).set_name("input");

		let output = permute_axes(&input, &[1, 2, 0]).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.25), ::std::f32::EPSILON * 1.0));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(-0.8), ::std::f32::EPSILON * 1.0));

		assert_eq!(output.calc().unwrap().shape(), &[5, 7, 3]);
	}

	#[test]
	fn forward_transpose_test() {
		let input = Node::new(&[3, 5, 7]).set_name("input");

		let output = transpose(&input).unwrap();

		input.set_value(arr0(1.25));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(1.25), ::std::f32::EPSILON * 1.0));

		input.set_value(arr0(-0.8));
		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr0(-0.8), ::std::f32::EPSILON * 1.0));

		assert_eq!(output.calc().unwrap().shape(), &[7, 5, 3]);
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[3, 5, 7]).set_name("input");

		let output = permute_axes(&input, &[1, 2, 0]).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}
}
