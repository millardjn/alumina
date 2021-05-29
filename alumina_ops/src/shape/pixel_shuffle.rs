use alumina_core::{
	base_ops::{shape_constraint::ShapeConstraint, OpSpecification, OpInstance},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{Node, NodeID, Graph},
	shape::{NodeAxis, NodeShape},
	shape_prop::ShapePropContext,
};
use indexmap::{indexset, IndexSet, IndexMap};
use ndarray::Dimension;
use std::{cmp::min, iter::once, any::Any};

/// Collapse outer dimensions, shuffling entries into the channel dimension
///
/// Decrease size of higher dimensions by given factors by mapping from each spaxel to chunks of the channel dimension.
/// Output channel dimension is increased by the product of the collapse factors. Inverse operation of Expand.
pub fn collapse<I>(input: I, factors: &[usize]) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();

	let output_channels =
		input.shape().slice()[input.shape().len() - 1].multiply(&NodeAxis::known(factors.iter().product()));
	let output_shape = input.shape().slice()[0..input.shape().len() - 1]
		.iter()
		.zip(factors)
		.map(|(axis, &f)| match axis {
			NodeAxis::Known { val } => (val.saturating_add(f - 1) / f).into(),
			NodeAxis::Interval { lower, upper } => {
				(lower.saturating_add(f - 1) / f, upper.saturating_add(f - 1) / f).into()
			}
		})
		.chain(once(output_channels))
		.into();

	let output = input
		.graph()
		.new_node(output_shape)
		.set_name_unique(&format!("collapse({})", input));

	let _op = Collapse::new(input, output.clone(), factors).build()?;

	Ok(output)
}

/// Expand outer dimensions, shuffling entries out of the channel dimension
///
/// Increase size of higher dimensions by given factors by mapping from chunks of the channel dimension to each new
/// spaxel. Output channel dimension is reduced by the product of the expansion factors. Inverse operation of Collapse.
/// Used in sub-pixel convolution.
pub fn expand<I>(input: I, factors: &[usize]) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();

	let expansion_factor: usize = factors.iter().product();
	let output_channels = match input.shape().slice()[input.shape().len() - 1] {
		NodeAxis::Known { val } => {
			if val % expansion_factor == 0 {
				(val / expansion_factor).into()
			} else {
				return Err(format!("The channel axis (last axis) of the input shape ({}) must be evenly divisible by the product of the factors ({:?})", input.shape(), factors).into());
			}
		}
		NodeAxis::Interval { lower, upper } => {
			let min = ((lower + expansion_factor - 1) / expansion_factor) * expansion_factor;
			let max = (upper / expansion_factor) * expansion_factor;

			if min > max {
				return Err(format!("The channel axis (last axis) of the input shape ({}) must be evenly divisible by the product of the factors ({:?})", input.shape(), factors).into());
			} else {
				NodeAxis::interval(min, max)
			}
		}
	};
	let output_shape = input.shape().slice()[0..input.shape().len() - 1]
		.iter()
		.zip(factors)
		.map(|(axis, &f)| axis.multiply(&NodeAxis::known(f)))
		.chain(once(output_channels))
		.into();

	let output = input
		.graph()
		.new_node(output_shape)
		.set_name_unique(&format!("expand({})", input));

	let _op = Expand::new(input.clone(), output.clone(), factors).build()?;

	let factors = factors.to_vec();
	let _op = ShapeConstraint::new(input, output.clone())
		.joint(move |axes| {
			axes.iter()
				.zip(&factors)
				.map(|(axis, f)| (axis * f).into())
				.chain(once(NodeAxis::unknown()))
				.into()
		})
		.build()?;

	Ok(output)
}

/// Collapse outer dimensions, shuffling entries into the channel dimension
///
/// Decrease size of higher dimensions by given factors by mapping from each spaxel to chunks of the channel dimension.
/// Output channel dimension is increased by the product of the collapse factors. Inverse operation of Expand.
#[must_use]
#[derive(Clone, Debug)]
pub struct Collapse {
	input: Node,
	output: Node,
	factors: Vec<usize>,
}

impl Collapse {
	pub fn new<I, O>(input: I, output: O, factors: &[usize]) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		let input = input.into();
		let output = output.into();

		Collapse {
			input,
			output,
			factors: factors.to_vec(),
		}
	}
}

impl OpSpecification for Collapse {
	type InstanceType = CollapseInstance;

	fn type_name(&self) -> &'static str {
		"Collapse"
	}

	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input.clone()]
	}

	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}
	
	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			input: mapping.get(&self.input).unwrap_or(&self.input).clone(),
			output: mapping.get(&self.output).unwrap_or(&self.output).clone(),
			factors: self.factors.clone()
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		if self.input.shape().len() != self.factors.len() + 1 {
			return Err(format!(
				"The input shape ({}) must be 1 axis larger than the number of factors ({:?})",
				self.input.shape(),
				self.factors
			)
			.into());
		}

		if self.factors.iter().any(|&f| f == 0) {
			return Err(format!("All factors ({:?}) must be greater than zero.", self.factors).into());
		}

		Ok(CollapseInstance::new(
			self.input.id().clone(),
			self.output.id().clone(),
			self.factors,
		))
	}
}

#[derive(Debug, Clone)]
pub struct CollapseInstance {
	input: NodeID,
	output: NodeID,
	factors: Vec<usize>,
	batch_end: usize,
}

impl CollapseInstance {
	fn new(input: NodeID, output: NodeID, factors: Vec<usize>) -> Self {
		let batch_end = factors.iter().take_while(|&&i| i == 1).count();
		CollapseInstance {
			input,
			output,
			factors,
			batch_end,
		}
	}
}

impl OpInstance for CollapseInstance {
	fn type_name(&self) -> &'static str {
		"Collapse"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(Collapse {
			input: graph.node_from_id(self.input),
			output: graph.node_from_id(self.output),
			factors: self.factors.clone(),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input.clone()]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output.clone()]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		let _op = Expand::new(ctx.grad_of(&self.output), ctx.grad_of(&self.input), &self.factors).build()?;
		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let input_shape = ctx.input_shape(&self.input);

		debug_assert_eq!(
			input_shape.ndim(),
			self.factors.len() + 1,
			"collapse factors must be the same length as input shape length minus 1"
		);

		let out_channels: usize = input_shape[input_shape.ndim() - 1] * self.factors.iter().product::<usize>();
		let output_shape: NodeShape = input_shape
			.slice()
			.iter()
			.zip(&self.factors)
			.map(|(dim, f)| dim.saturating_add(f - 1) / f)
			.chain(once(out_channels))
			.into();

		ctx.merge_output_shape(&self.output, &output_shape)?;
		Ok(())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let input = ctx.get_input_standard(&self.input);
		let mut output = ctx.get_output_standard(&self.output);

		let input_shape = input.shape();
		let output_shape = output.shape().to_vec();

		let input_channels = input_shape[input_shape.len() - 1];
		let output_channels = output_shape[output_shape.len() - 1];

		let input_spatial = &input_shape[self.batch_end..input_shape.len() - 1];
		let output_spatial = &output_shape[self.batch_end..output_shape.len() - 1];
		let factors_spatial = &self.factors[self.batch_end..];

		debug_assert_eq!(
			input_shape.len(),
			output_shape.len(),
			"Input ndims does not match output ndims"
		);
		debug_assert_eq!(
			input_shape.len(),
			self.factors.len() + 1,
			"factors must be the same length as input dimensions minus 1"
		);
		debug_assert!(
			input_shape
				.iter()
				.zip(&self.factors)
				.map(|(dim, f)| dim.saturating_add(f - 1) / f)
				.eq(output_shape[..output_shape.len() - 1].iter().cloned()),
			"input shape and factors incompatible with output shape"
		);
		debug_assert_eq!(
			output_channels,
			input_channels * self.factors.iter().product::<usize>(),
			"The ratio of output channels over input channels must equal the product of all factors"
		);

		let input = input.as_slice().unwrap();
		let output = output.as_slice_mut().unwrap();

		let batches = input_shape[..self.batch_end].iter().product();

		let in_size = input.len() / batches;
		let out_size = output.len() / batches;

		for b_ind in 0..batches {
			let out_batch = &mut output[b_ind * out_size..][..out_size];
			let in_batch = &input[b_ind * in_size..][..in_size];

			for (i, patch) in out_batch.chunks_mut(output_channels).enumerate() {
				let output_ind = i * output_channels;
				collapse_recurse(
					patch,
					&in_batch,
					&factors_spatial,
					input_channels,
					&input_spatial,
					&output_spatial,
					0,
					output_ind,
					out_size,
				);
			}
		}

		Ok(())
	}
}

/// Expand outer dimensions, shuffling entries out of the channel dimension
///
/// Increase size of higher dimensions by given factors by mapping from chunks of the channel dimension to each new
/// spaxel. Output channel dimension is reduced by the product of the expansion factors. Inverse operation of Collapse.
/// Used in sub-pixel convolution.
#[must_use]
#[derive(Clone, Debug)]
pub struct Expand {
	factors: Vec<usize>,
	input: Node,
	output: Node,
}

impl Expand {
	pub fn new<I, O>(input: I, output: O, factors: &[usize]) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		let input = input.into();
		let output = output.into();
		Expand {
			input,
			output,
			factors: factors.to_vec(),
		}
	}
}

impl OpSpecification for Expand {
	type InstanceType = ExpandInstance;

	fn type_name(&self) -> &'static str {
		"Expand"
	}

	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input.clone()]
	}

	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			input: mapping.get(&self.input).unwrap_or(&self.input).clone(),
			output: mapping.get(&self.output).unwrap_or(&self.output).clone(),
			factors: self.factors.clone()
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		if self.input.shape().len() != self.factors.len() + 1 {
			return Err(format!(
				"The input shape ({}) must be 1 axis larger than the number of factors ({:?})",
				self.input.shape(),
				self.factors
			)
			.into());
		}

		if self.factors.iter().any(|&f| f == 0) {
			return Err(format!("All factors ({:?}) must be greater than zero.", self.factors).into());
		}

		assert_eq!(
			self.input.shape().len(),
			self.factors.len() + 1,
			"collapse factors must be the same length as input shape length minus 1"
		);

		Ok(ExpandInstance::new(
			self.input.id().clone(),
			self.output.id().clone(),
			self.factors,
		))
	}
}

#[derive(Debug, Clone)]
pub struct ExpandInstance {
	input: NodeID,
	output: NodeID,
	factors: Vec<usize>,
	batch_end: usize,
}

impl ExpandInstance {
	fn new(input: NodeID, output: NodeID, factors: Vec<usize>) -> Self {
		let batch_end = factors.iter().take_while(|&&i| i == 1).count();
		ExpandInstance {
			input,
			output,
			factors,
			batch_end,
		}
	}
}

impl OpInstance for ExpandInstance {
	fn type_name(&self) -> &'static str {
		"Expand"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(Expand {
			input: graph.node_from_id(self.input),
			output: graph.node_from_id(self.output),
			factors: self.factors.clone(),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input.clone()]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output.clone()]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		let _op = Collapse::new(ctx.grad_of(&self.output), ctx.grad_of(&self.input), &self.factors).build()?;
		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let input_shape = ctx.input_shape(&self.input);

		debug_assert_eq!(
			input_shape.ndim(),
			self.factors.len() + 1,
			"Expansion factors must be the same length as input shape length minus 1"
		);

		if input_shape[input_shape.ndim() - 1] % self.factors.iter().product::<usize>() != 0 {
			return Err(format!("Input shape ({:?}) channel axis (last axis) must be evenly divisible by the product of all factors ({:?})", input_shape, self.factors).into());
		}

		let out_channels: usize = input_shape[input_shape.ndim() - 1] / self.factors.iter().product::<usize>();
		let output_shape: NodeShape = input_shape
			.slice()
			.iter()
			.zip(&self.factors)
			.map(|(dim, f)| ((dim - 1) * f + 1, dim * f))
			.chain(once((out_channels, out_channels)))
			.into();

		ctx.merge_output_shape(&self.output, &output_shape)?;
		Ok(())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let input = ctx.get_input_standard(&self.input);
		let mut output = ctx.get_output_standard(&self.output);

		let input_shape = input.shape();
		let output_shape = output.shape().to_vec();

		let input_channels = input_shape[input_shape.len() - 1];
		let output_channels = output_shape[output_shape.len() - 1];

		let input_spatial = &input_shape[self.batch_end..input_shape.len() - 1];
		let output_spatial = &output_shape[self.batch_end..output_shape.len() - 1];
		let factors_spatial = &self.factors[self.batch_end..];

		debug_assert_eq!(
			input_shape.len(),
			output_shape.len(),
			"Input ndims does not match output ndims"
		); // caught in shape prop
		debug_assert_eq!(
			input_shape.len(),
			self.factors.len() + 1,
			"factors must be the same length as input dimensions minus 1"
		); // caught in shape prop
		debug_assert!(
			output_shape
				.iter()
				.zip(&self.factors)
				.map(|(dim, f)| dim.saturating_add(f - 1) / f)
				.eq(input_shape[..input_shape.len() - 1].iter().cloned()),
			"input shape and factors incompatible with output shape"
		); // caught in shape prop
		debug_assert_eq!(
			input_channels,
			output_channels * self.factors.iter().product::<usize>(),
			"The ratio of input channels over output channels must equal the product of all factors"
		); // caught in shape prop

		let input = input.as_slice().unwrap();
		let output = output.as_slice_mut().unwrap();

		let batches = input_shape[..self.batch_end].iter().product();

		let in_size = input.len() / batches;
		let out_size = output.len() / batches;

		for b_ind in 0..batches {
			let out_batch = &mut output[b_ind * out_size..][..out_size];
			let in_batch = &input[b_ind * in_size..][..in_size];

			for (i, patch) in in_batch.chunks(input_channels).enumerate() {
				let input_ind = i * input_channels;
				expand_recurse(
					patch,
					out_batch,
					&factors_spatial,
					output_channels,
					&output_spatial,
					&input_spatial,
					0,
					input_ind,
					in_size,
				)
			}
		}

		Ok(())
	}
}

#[allow(clippy::too_many_arguments)]
fn collapse_recurse(
	patch: &mut [f32],
	input: &[f32],
	factors_spatial: &[usize],
	in_channels: usize,
	input_spatial: &[usize],
	output_spatial: &[usize],
	axis: usize,
	output_ind: usize,
	old_out_stride: usize,
) {
	// stride in array index, not spaxel index
	let out_stride = old_out_stride / output_spatial[axis];
	let in_stride = input.len() / input_spatial[axis];
	let patch_stride = patch.len() / factors_spatial[axis];

	// coordinates of the centre spaxel of the kernel, for the current axis
	let ox = (output_ind % old_out_stride) / out_stride;

	// valid range of input coordinates the current axis for the current patch
	let start = ox * factors_spatial[axis];
	let end = min(start + factors_spatial[axis], input_spatial[axis]);

	if axis < factors_spatial.len() - 1 {
		for i in start..end {
			let new_input = &input[in_stride * i..in_stride * (i + 1)];

			let i_patch = i - start;
			let new_patch = &mut patch[i_patch * patch_stride..(i_patch + 1) * patch_stride];
			let new_axis = axis + 1;

			collapse_recurse(
				new_patch,
				new_input,
				factors_spatial,
				in_channels,
				input_spatial,
				output_spatial,
				new_axis,
				output_ind,
				out_stride,
			);
		}
	} else {
		let offset = start * in_channels;
		let len = (end - start) * in_channels;
		let input_crop = &input[offset..][..len];
		let patch_crop = &mut patch[..len];

		for i in 0..len {
			patch_crop[i] += input_crop[i];
		}
	}
}

#[allow(clippy::too_many_arguments)]
fn expand_recurse(
	patch: &[f32],
	output: &mut [f32],
	factors_spatial: &[usize],
	out_channels: usize,
	output_spatial: &[usize],
	input_spatial: &[usize],
	axis: usize,
	input_ind: usize,
	old_in_stride: usize,
) {
	// stride in array index, not spaxel index
	let in_stride = old_in_stride / input_spatial[axis];
	let out_stride = output.len() / output_spatial[axis];
	let patch_stride = patch.len() / factors_spatial[axis];

	// coordinates of the centre spaxel of the kernel, for the current axis
	let ix = (input_ind % old_in_stride) / in_stride;

	// valid range of input coordinates the current axis for the current patch
	let start = ix * factors_spatial[axis];
	let end = min(start + factors_spatial[axis], output_spatial[axis]);

	if axis < factors_spatial.len() - 1 {
		for i in start..end {
			let new_output = &mut output[out_stride * i..out_stride * (i + 1)];

			let i_patch = i - start;
			let new_patch = &patch[i_patch * patch_stride..(i_patch + 1) * patch_stride];
			let new_axis = axis + 1;

			expand_recurse(
				new_patch,
				new_output,
				factors_spatial,
				out_channels,
				output_spatial,
				input_spatial,
				new_axis,
				input_ind,
				in_stride,
			);
		}
	} else {
		let offset = start * out_channels;
		let len = (end - start) * out_channels;
		let output_crop = &mut output[offset..][..len];
		let patch_crop = &patch[..len];

		for i in 0..len {
			output_crop[i] += patch_crop[i];
		}
	}
}

#[cfg(test)]
mod tests {
	use super::{collapse, collapse_recurse, expand, expand_recurse};
	use alumina_core::graph::Node;
	use alumina_test::grad_numeric_test::GradNumericTest;
	use ndarray::Dimension;

	#[test]
	fn grad_numeric_collapse_test() {
		let input = Node::new(&[2, 20, 9, 5]).set_name("input");

		let output = collapse(&input, &[1, 5, 3]).unwrap();

		assert_eq!(output.shape().to_data_shape().unwrap().slice(), &[2, 4, 3, 75]);

		GradNumericTest::new(output, &[input]).run();
	}

	#[test]
	fn grad_numeric_collapse_indivisible_test() {
		let input = Node::new(&[2, 21, 10, 5]).set_name("input");

		let output = collapse(&input, &[1, 5, 3]).unwrap();

		assert_eq!(output.shape().to_data_shape().unwrap().slice(), &[2, 5, 4, 75]);

		GradNumericTest::new(output, &[input]).run();
	}

	#[test]
	fn grad_numeric_expand_test() {
		let input = Node::new(&[2, 4, 3, 75]).set_name("input");

		let output = expand(&input, &[1, 5, 3]).unwrap();

		assert_eq!(output.shape().to_data_shape().unwrap().slice(), &[2, 20, 9, 5]);

		GradNumericTest::new(output, &[input]).run();
	}

	#[test]
	fn test_collapse_recurse() {
		_test_collapse_recurse();
	}

	fn _test_collapse_recurse() {
		let factors = vec![5, 3];
		let input_shape: Vec<usize> = vec![7, 5];
		let output_shape: Vec<usize> = input_shape
			.iter()
			.zip(factors.iter())
			.map(|(&i, &f)| i.saturating_add(f - 1) / f)
			.collect();
		let in_channels = 2;
		let out_channels = factors.iter().fold(in_channels, |p, v| p * v);

		let input_spaxel_count: usize = input_shape.iter().product();
		let output_spaxel_count: usize = output_shape.iter().product();

		let input_size = input_spaxel_count * in_channels;
		let output_size = output_spaxel_count * out_channels;
		let patch_size = factors.iter().fold(in_channels, |p, v| p * v);

		let input: Vec<f32> = (0..input_size).map(|x| x as f32).collect();

		let axis = 0;

		let mut output = vec![0.0; patch_size * output_spaxel_count];

		for (i, patch) in output.chunks_mut(patch_size).enumerate() {
			let output_ind = i * out_channels;
			collapse_recurse(
				patch,
				&input,
				&factors,
				in_channels,
				&input_shape,
				&output_shape,
				axis,
				output_ind,
				output_size,
			);
		}

		let target = vec![
			00., 1., 2., 3., 4., 5., 10., 11., 12., 13., 14., 15., 20., 21., 22., 23., 24., 25., 30., 31., 32., 33.,
			34., 35., 40., 41., 42., 43., 44., 45., 06., 7., 8., 9., 0., 0., 16., 17., 18., 19., 0., 0., 26., 27., 28.,
			29., 0., 0., 36., 37., 38., 39., 0., 0., 46., 47., 48., 49., 0., 0., 50., 51., 52., 53., 54., 55., 60.,
			61., 62., 63., 64., 65., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 56., 57.,
			58., 59., 0., 0., 66., 67., 68., 69., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
			0., 0., 0.,
		];

		assert_eq!(output, target);
	}

	#[test]
	fn test_expand_recurse() {
		let factors = vec![5, 3];
		let input_shape = vec![2, 2];
		let output_shape = vec![7, 5];
		let out_channels = 2;
		let in_channels = factors.iter().fold(out_channels, |p, v| p * v);

		let input_spaxel_count: usize = input_shape.iter().product();
		let output_spaxel_count: usize = output_shape.iter().product();

		let input_size = input_spaxel_count * in_channels;
		let output_size = output_spaxel_count * out_channels;

		let input = vec![
			00., 1., 2., 3., 4., 5., 10., 11., 12., 13., 14., 15., 20., 21., 22., 23., 24., 25., 30., 31., 32., 33.,
			34., 35., 40., 41., 42., 43., 44., 45., 06., 7., 8., 9., 0., 0., 16., 17., 18., 19., 0., 0., 26., 27., 28.,
			29., 0., 0., 36., 37., 38., 39., 0., 0., 46., 47., 48., 49., 0., 0., 50., 51., 52., 53., 54., 55., 60.,
			61., 62., 63., 64., 65., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 56., 57.,
			58., 59., 0., 0., 66., 67., 68., 69., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
			0., 0., 0.,
		];

		let axis = 0;

		let mut output = vec![0.0; output_size];

		for (i, patch) in input.chunks(in_channels).enumerate() {
			let input_ind = i * in_channels;

			expand_recurse(
				patch,
				&mut output,
				&factors,
				out_channels,
				&output_shape,
				&input_shape,
				axis,
				input_ind,
				input_size,
			)
		}

		let target: Vec<f32> = (0..output_size).map(|x| x as f32).collect();

		assert_eq!(output, target);
	}

	#[test]
	fn test_expand_recurse_full() {
		let factors = vec![5, 3];
		let input_shape = vec![2, 2];
		let output_shape: Vec<usize> = input_shape.iter().zip(factors.iter()).map(|(i, f)| i * f).collect();
		let out_channels = 2;
		let in_channels = factors.iter().fold(out_channels, |p, v| p * v);

		let input_spaxel_count: usize = input_shape.iter().product();
		let output_spaxel_count: usize = output_shape.iter().product();

		let input_size = input_spaxel_count * in_channels;
		let output_size = output_spaxel_count * out_channels;

		let input = vec![
			0., 1., 2., 3., 4., 5., 12., 13., 14., 15., 16., 17., 24., 25., 26., 27., 28., 29., 36., 37., 38., 39.,
			40., 41., 48., 49., 50., 51., 52., 53., 6., 7., 8., 9., 10., 11., 18., 19., 20., 21., 22., 23., 30., 31.,
			32., 33., 34., 35., 42., 43., 44., 45., 46., 47., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64.,
			65., 72., 73., 74., 75., 76., 77., 84., 85., 86., 87., 88., 89., 96., 97., 98., 99., 100., 101., 108.,
			109., 110., 111., 112., 113., 66., 67., 68., 69., 70., 71., 78., 79., 80., 81., 82., 83., 90., 91., 92.,
			93., 94., 95., 102., 103., 104., 105., 106., 107., 114., 115., 116., 117., 118., 119.,
		];

		let axis = 0;

		let mut output = vec![0.0; output_size];

		for (i, patch) in input.chunks(in_channels).enumerate() {
			let input_ind = i * in_channels;

			expand_recurse(
				patch,
				&mut output,
				&factors,
				out_channels,
				&output_shape,
				&input_shape,
				axis,
				input_ind,
				input_size,
			)
		}

		let target: Vec<f32> = (0..output_size).map(|x| x as f32).collect();

		assert!(output == target, "target{:?} \n output{:?}", target, output);
	}
}
