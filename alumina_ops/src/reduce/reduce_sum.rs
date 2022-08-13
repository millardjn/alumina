use crate::{
	elementwise::{div::div, mul::mul},
	manip::expand_dims::expand_dims,
	math::broadcast::{broadcast_fn, Broadcast},
	reduce::reduce_prod::reduce_prod,
	shape::shape_of::shape_of,
};
use alumina_core::{
	base_ops::{OpInstance, OpSpecification},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{Graph, Node, NodeID},
	shape::{NodeAxis, NodeShape},
	shape_prop::ShapePropContext,
};
use indexmap::{indexset, IndexMap, IndexSet};
use ndarray::{Dimension, Zip};
use smallvec::SmallVec;
use std::any::Any;

pub fn reduce_sum<I>(input: I, axes: &[isize], keep_dims: bool) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let usize_axes = regularise_axes(axes, input.shape().len());

	let output_shape: NodeShape = calc_output_shape(&input.shape(), &usize_axes, keep_dims);

	let output = input
		.graph()
		.new_node(output_shape)
		.set_name_unique(&format!("reduce_sum({})", input));

	let _op = ReduceSum::new(input, output.clone())
		.axes(axes)
		.keep_dims(keep_dims)
		.build()?;

	Ok(output)
}

pub fn reduce_mean<I>(input: I, axes: &[isize], keep_dims: bool) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();

	let sum_output = reduce_sum(input.clone(), axes, keep_dims)?;

	let scale = div(
		reduce_prod(shape_of(sum_output.clone())?, &[], false)?
			.set_name_unique(&format!("reduce_mean({})_output_size", input)),
		reduce_prod(shape_of(input.clone())?, &[], false)?
			.set_name_unique(&format!("reduce_mean({})_input_size", input)),
	)?
	.set_name_unique(&format!("reduce_mean({})_size_ratio", input));

	let output = broadcast_fn(mul, sum_output, scale)?.set_name_unique(&format!("reduce_mean({})", input));

	Ok(output)
}

/// `ReduceSum` `OpBuilder`
#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct ReduceSum {
	input: Node,
	output: Node,
	axes: Vec<usize>,
	keep_dims: bool,
}

impl ReduceSum {
	pub fn new<I, O>(input: I, output: O) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		let input = input.into();
		let output = output.into();
		ReduceSum {
			input,
			output,
			axes: vec![],
			keep_dims: false,
		}
	}

	/// Supply which axes are to be reduced across.
	///
	/// If axes is empty, all axes are reduced.
	/// Each element of `axes` can be in the range [-input.len(), input.len()).
	///
	/// Default: empty
	///
	/// # Panics
	/// Panics if axes are outside of the valid range.
	pub fn axes(mut self, axes: &[isize]) -> Self {
		self.axes = regularise_axes(axes, self.input.shape().len());
		self
	}

	/// If `true` the reduced axes still appear in the output with size 1, otherwise they are removed.
	///
	/// Default: `false`
	pub fn keep_dims(mut self, keep_dims: bool) -> Self {
		self.keep_dims = keep_dims;
		self
	}
}

impl OpSpecification for ReduceSum {
	type InstanceType = ReduceSumInstance;

	fn op_type(&self) -> &'static str {
		"ReduceSum"
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
			axes: self.axes.clone(),
			keep_dims: self.keep_dims,
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(ReduceSumInstance {
			input: self.input.id(),
			output: self.output.id(),
			axes: self.axes.clone(),
			keep_dims: self.keep_dims,
		})
	}
}

/// ReduceSum OpInstance,
#[derive(Clone, Debug)]
pub struct ReduceSumInstance {
	input: NodeID,
	output: NodeID,
	axes: Vec<usize>,
	keep_dims: bool,
}

impl OpInstance for ReduceSumInstance {
	fn op_type(&self) -> &'static str {
		"ReduceSum"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(ReduceSum {
			input: graph.node_from_id(self.input),
			output: graph.node_from_id(self.output),
			axes: self.axes.clone(),
			keep_dims: self.keep_dims,
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		// if self.input.shape().len() == self.output.shape().len() {
		// 	Broadcast::new(ctx.grad_of(&self.output), ctx.grad_of(&self.input)).build()?;
		// } else {
		// 	let intermediate = expand_dims(ctx.grad_of(&self.output), &self.axes)?;

		// }
		let expand_grad = if ctx.node(&self.input).shape().len() == ctx.node(&self.output).shape().len() {
			ctx.grad_of(&self.output)
		} else {
			expand_dims(ctx.grad_of(&self.output), &self.axes)?
		};
		Broadcast::new(expand_grad, ctx.grad_of(&self.input)).build()?;
		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let output_shape: NodeShape =
			calc_output_shape(&ctx.input_shape(&self.input).slice().into(), &self.axes, self.keep_dims);
		ctx.merge_output_shape(&self.output, &output_shape)
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		if ctx.shape(&self.input) == ctx.shape(&self.output) && ctx.can_take(&self.input) && ctx.can_set(&self.output) {
			// if output can be set using the input array, do that.
			ctx.set(&self.output, ctx.take(&self.input));
		} else {
			let input = ctx.get_input(&self.input);
			let output = ctx.get_output_standard(&self.output);

			if self.axes.iter().all(|axis| input.shape()[*axis] != 0) {
				// reshape as though keep_dims is true
				let output_shape = calc_output_shape(&input.shape().into(), &self.axes, true)
					.into_iter()
					.map(NodeAxis::lower)
					.collect::<SmallVec<[usize; 8]>>();
				let mut output = output.into_shape(output_shape.as_slice()).expect("Alumina Bug: ReduceSum should be guaranteed that the reshape is valid by shape_prop and that the output is contiguous");

				Zip::from(&input)
					.and_broadcast(output.cell_view())
					.for_each(|&input, output| {
						let new = output.get() + input;
						output.set(new);
					});
			}
		}

		Ok(())
	}
}

/// convert from wrapping isize axis numbering, to sorted, direct, deduplicated usize numbering
fn regularise_axes(axes: &[isize], input_len: usize) -> Vec<usize> {
	if axes.is_empty() {
		return (0..input_len).collect();
	}

	for &dim in axes {
		assert!(
			dim < input_len as isize,
			" axes ({}) must be less than input.shape().len() ({})",
			dim,
			input_len
		);
		assert!(
			dim >= -(input_len as isize),
			" axes must be greater or equal to -input.shape().len()"
		);
	}
	let mut axes: Vec<_> = axes
		.iter()
		.map(|&dim| (dim + input_len as isize) as usize % input_len)
		.collect();
	axes.sort_unstable();
	axes.dedup();
	axes
}

fn calc_output_shape(input_shape: &NodeShape, axes: &[usize], keep_dims: bool) -> NodeShape {
	let output_len = input_shape.len() - if keep_dims { 0 } else { axes.len() };
	let mut output_shape: SmallVec<[NodeAxis; 8]> = (0..output_len).map(|_| NodeAxis::known(1)).collect();
	let mut axes_i = 0;
	let mut output_i = 0;

	for (i, in_dim) in input_shape.into_iter().enumerate() {
		if axes_i < axes.len() && i == axes[axes_i] {
			if keep_dims {
				output_i += 1;
			}
			axes_i += 1;
		} else {
			output_shape[output_i] = in_dim.clone();
			output_i += 1;
		}
	}

	output_shape.into()
}

#[cfg(test)]
mod tests {
	use super::{reduce_mean, reduce_sum};
	use alumina_core::graph::Node;
	use alumina_test::grad_numeric_test::GradNumericTest;
	use indexmap::indexset;
	use ndarray::{arr2, arr3};

	#[test]
	fn forward_sum_test() {
		let input = Node::new(&[2, 3, 5])
			.set_value(arr3(&[
				[
					[1.0, 2.0, 3.0, 4.0, 5.0],
					[6.0, 7.0, 8.0, 9.0, 10.0],
					[11.0, 12.0, 13.0, 14.0, 15.0],
				],
				[
					[16.0, 17.0, 18.0, 19.0, 20.0],
					[21.0, 22.0, 23.0, 24.0, 25.0],
					[26.0, 27.0, 28.0, 29.0, 30.0],
				],
			]))
			.set_name("input");

		let output1 = reduce_sum(&input, &[0], false).unwrap().set_name("output2");
		let output2 = reduce_sum(&input, &[1], false).unwrap().set_name("output1");

		let expected1 = arr2(&[
			[17.0, 19.0, 21.0, 23.0, 25.0],
			[27.0, 29.0, 31.0, 33.0, 35.0],
			[37.0, 39.0, 41.0, 43.0, 45.0],
		])
		.into_dyn();

		let expected2 = arr2(&[[18.0, 21.0, 24.0, 27.0, 30.0], [63.0, 66.0, 69.0, 72.0, 75.0]]).into_dyn();

		assert_eq!(expected1, output1.calc().unwrap());
		assert_eq!(expected2, output2.calc().unwrap());
	}

	#[test]
	fn forward_mean_test() {
		let input = Node::new(&[2, 3, 5])
			.set_value(arr3(&[
				[
					[1.0, 2.0, 3.0, 4.0, 5.0],
					[6.0, 7.0, 8.0, 9.0, 10.0],
					[11.0, 12.0, 13.0, 14.0, 15.0],
				],
				[
					[16.0, 17.0, 18.0, 19.0, 20.0],
					[21.0, 22.0, 23.0, 24.0, 25.0],
					[26.0, 27.0, 28.0, 29.0, 30.0],
				],
			]))
			.set_name("input");

		let output1 = reduce_mean(&input, &[0], false).unwrap().set_name("output2");
		let output2 = reduce_mean(&input, &[1], false).unwrap().set_name("output1");

		let expected1 = arr2(&[
			[8.5, 9.5, 10.5, 11.5, 12.5],
			[13.5, 14.5, 15.5, 16.5, 17.5],
			[18.5, 19.5, 20.5, 21.5, 22.5],
			/* [17.0, 19.0, 21.0, 23.0, 25.0],
			 * [27.0, 29.0, 31.0, 33.0, 35.0],
			 * [37.0, 39.0, 41.0, 43.0, 45.0], */
		])
		.into_dyn();

		let expected2 = arr2(&[[6.0, 7.0, 8.0, 9.0, 10.0], [21.0, 22.0, 23.0, 24.0, 25.0]]).into_dyn();

		assert_eq!(expected1, output1.calc().unwrap());
		assert_eq!(expected2, output2.calc().unwrap());
	}

	#[test]
	fn grad_numeric_sum_test() {
		let input = Node::new(&[13, 7, 33]).set_name("input");

		let output = reduce_sum(&input, &[1], false).unwrap().set_name("output");

		GradNumericTest::new(&output, &indexset![&input]).run();
	}

	#[test]
	fn grad_numeric_sum_test_keep() {
		let input = Node::new(&[13, 7, 33]).set_name("input");

		let output = reduce_sum(&input, &[1], true).unwrap().set_name("output");

		GradNumericTest::new(&output, &indexset![&input]).run();
	}

	#[test]
	fn grad_numeric_mean_test() {
		let input = Node::new(&[13, 7, 33]).set_name("input");

		let output = reduce_mean(&input, &[1], false).unwrap().set_name("output");

		GradNumericTest::new(&output, &indexset![&input]).run();
	}
}
