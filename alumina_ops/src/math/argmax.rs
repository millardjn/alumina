use alumina_core::{
	base_ops::{OpBuilder, OpInstance},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{Node, NodeInner},
	shape::{NodeAxis, NodeShape},
	shape_prop::ShapePropContext,
	util::wrap_dim,
};
use indexmap::{indexset, IndexSet};

use ndarray::{Axis, Dimension, Zip};

use smallvec::SmallVec;

/// Returns the integer location of the maximum for each lane in the provided axis.
///
/// The output node has the shape of the input, but with the axis removed.
pub fn argmax<I>(input: I, axis: isize) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();
	let axis = wrap_dim(axis, input.shape().len());

	let output_shape: NodeShape = calc_output_shape(input.shape(), axis, false);

	let output = input
		.graph()
		.new_node(output_shape)
		.set_name_unique(&format!("argmax({})", input));

	let _op = ArgMax::new(input, output.clone(), axis as isize).build()?;

	Ok(output)
}

/// `ArgMax` `OpBuilder`
#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct ArgMax {
	input: Node,
	output: Node,
	axis: usize,
}

impl ArgMax {
	pub fn new<I, O>(input: I, output: O, axis: isize) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		let input = input.into();
		let output = output.into();
		let axis = wrap_dim(axis, input.shape().len());
		ArgMax { input, output, axis }
	}
}

impl OpBuilder for ArgMax {
	type InstanceType = ArgMaxInstance;

	fn type_name(&self) -> &'static str {
		"ArgMax"
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input.clone()]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	// Create a new OpInstance with nodes switched out
	// fn clone_with_nodes_changed(&self, mapping: IndexMap<Node, Node>) -> Result<Self, CloneError> {
	// 	Ok(ArgMax {
	// 		input: mapping.get(&self.input).unwrap_or_else(|| &self.input).clone(),
	// 		output: mapping.get(&self.output).unwrap_or_else(|| &self.output).clone(),
	// 		//extra_axes: self.extra_axes,
	// 	})
	// }

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(ArgMaxInstance {
			input: self.input.inner().clone(),
			output: self.output.inner().clone(),
			axis: self.axis,
		})
	}
}

/// ArgMax OpInstance,
#[derive(Clone, Debug)]
pub struct ArgMaxInstance {
	input: NodeInner,
	output: NodeInner,
	axis: usize,
}

impl OpInstance for ArgMaxInstance {
	fn type_name(&self) -> &'static str {
		"ArgMax"
	}

	// fn clone_with_nodes_changed(&self, mapping: IndexMap<NodeInner, NodeInner>) -> Result<Box<OpInstance>,
	// CloneError> { 	Ok(Box::new(ArgMaxInstance {
	// 		input: mapping.get(&self.input).unwrap_or_else(|| &self.input).clone(),
	// 		output: mapping.get(&self.output).unwrap_or_else(|| &self.output).clone(),
	// 		//extra_axes: self.extra_axes.clone(),
	// 	}))
	// }

	fn inputs(&self) -> IndexSet<NodeInner> {
		indexset![self.input.clone()]
	}

	fn outputs(&self) -> IndexSet<NodeInner> {
		indexset![self.output.clone()]
	}

	fn gradient(&self, _ctx: &mut GradientContext) -> Result<(), GradientError> {
		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let output_shape: NodeShape = calc_output_shape(&ctx.input_shape(&self.input).slice().into(), self.axis, false);
		ctx.merge_output_shape(&self.output, &output_shape)
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let input = ctx.get_input(&self.input);
		let mut output = ctx.get_output(&self.output);

		Zip::from(input.lanes(Axis(self.axis)))
			.and(&mut output)
			.apply(|input, output| {
				let mut iter = input.iter().enumerate();
				if let Some((mut max_i, mut max)) = iter.next() {
					for (i, x) in iter {
						if x > max {
							max = x;
							max_i = i;
						}
					}
					*output += max_i as f32;
				}
			});

		Ok(())
	}
}

#[allow(unused)]
fn calc_output_shape_slice(input_shape: &[usize], axis: usize, keep_dims: bool) -> SmallVec<[usize; 8]> {
	input_shape
		.iter()
		.enumerate()
		.filter_map(|(i, &node_axis)| {
			if i != axis {
				Some(node_axis)
			} else if keep_dims {
				Some(1)
			} else {
				None
			}
		})
		.collect()
}

fn calc_output_shape(input_shape: &NodeShape, axis: usize, keep_dims: bool) -> NodeShape {
	input_shape
		.iter()
		.enumerate()
		.filter_map(|(i, node_axis)| {
			if i != axis {
				Some(node_axis.clone())
			} else if keep_dims {
				Some(NodeAxis::known(1))
			} else {
				None
			}
		})
		.into()
}

#[cfg(test)]
mod tests {
	use super::argmax;
	use alumina_core::graph::Node;
	use alumina_test::relatively_close::RelClose;

	use ndarray::{arr1, arr2};

	#[test]
	fn argmax_test() {
		let input = Node::new(&[5, 7]).set_name("input").set_value(arr2(&[
			[18.0, 3.0, 25.0, 0.0, 6.0, 35.0, 9.2],
			[28.0, 14.0, 33.0, 22.0, 20.0, 8.0, 41.0],
			[13.0, 30.0, 21.0, 19.0, 7.0, 9.0, 18.0],
			[16.0, 1.0, 26.0, 32.0, 2.0, 29.0, 17.0],
			[17.0, 12.0, 5.0, 11.0, 10.0, 15.0, 3.0],
		]));

		let output = argmax(&input, -1).unwrap();

		assert!(output
			.calc()
			.unwrap()
			.all_relatively_close(&arr1(&[5.0, 6.0, 1.0, 3.0, 0.0]), ::std::f32::EPSILON));
	}
}
