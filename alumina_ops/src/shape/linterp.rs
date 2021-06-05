// use graph::{GraphDef, GraphShapes, Result};
// use id::{NodeID, DataID, OpID, PassID};
// use storage::Storage;
// use ops::{standard_op_name, Op, OpInstance, Pass};
// use shape::NodeShape;

// use crate::{
// 	exec::ExecutionContext,
// 	grad::GradientContext,
// 	graph::{Node, NodeInner},
// 	ops::{
// 		shape::shape_constraint::ShapeConstraint, ExecutionError, GradientError, OpBuildError, OpBuilder, OpInstance,
// 		ShapePropError,
// 	},
// 	shape::{NodeAxis, NodeShape},
// 	shape_prop::ShapePropContext,
// };
// use indexmap::IndexSet;

use alumina_core::{
	base_ops::{shape_constraint::ShapeConstraint, OpInstance, OpSpecification},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{Graph, Node, NodeID},
	shape::{NodeAxis, NodeShape},
	shape_prop::ShapePropContext,
};
use indexmap::{indexset, IndexMap, IndexSet};

use ndarray::Dimension;
use smallvec::SmallVec;
use std::{
	any::Any,
	cmp::{max, min},
	ops::Range,
};

pub fn linterp<I>(input: I, factors: &[usize]) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();

	let output_shape: NodeShape = input
		.shape()
		.iter()
		.zip(factors)
		.map(|(axis, &f)| axis.multiply(&NodeAxis::known(f)))
		.into();

	let output = input
		.graph()
		.new_node(output_shape)
		.set_name_unique(&format!("linterp({})", input));

	if input
		.shape()
		.iter()
		.zip(factors)
		.any(|(axis, &f)| f != 1 && !axis.is_known())
	{
		let factors = factors.to_vec();
		let _op = ShapeConstraint::new(input.clone(), output.clone())
			.joint(move |shape| shape.iter().zip(&factors).map(|(axis, f)| axis * f).into())
			.build()?;
	}
	let _op = Linterp::new(input, output.clone(), factors).build()?;

	Ok(output)
}

/// Linterp implements linear interpolation upscaling
///
/// Increase size of each spatial dimension by given a factor by linear interpolating between spaxels in the input
#[must_use]
#[derive(Clone, Debug)]
pub struct Linterp {
	factors: Vec<usize>,
	input: Node,
	output: Node,
}

impl Linterp {
	pub fn new<I, O>(input: I, output: O, factors: &[usize]) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		let input = input.into();
		let output = output.into();
		Linterp {
			input,
			output,
			factors: factors.to_vec(),
		}
	}
}

impl OpSpecification for Linterp {
	type InstanceType = LinterpInstance;

	fn type_name(&self) -> &'static str {
		"Linterp"
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
			factors: self.factors.clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		if self.factors.iter().any(|&f| f == 0) {
			return Err(format!("expansion factors ({:?}) must all be greater than 0", self.factors,).into());
		}
		if self.input.shape().len() != self.factors.len() {
			return Err(format!(
				"expansion factors ({:?}) and input shape ({}) must have the same length",
				self.factors,
				self.input.shape(),
			)
			.into());
		}
		if self.input.shape().len() != self.output.shape().len() {
			return Err(format!(
				"input shape({}) and output shape ({}) must have the same length",
				self.input.shape(),
				self.output.shape(),
			)
			.into());
		}

		Ok(LinterpInstance::new(self.input.id(), self.output.id(), self.factors))
	}
}

// #[derive(Debug, Clone)]
// pub struct LinterpInstance {
// 	name: String,
// 	input_id: NodeID,
// 	output_id: NodeID,
// 	factors: Vec<usize>,
// 	forward_id: PassID,
// 	backward_id: PassID,
// }

// impl OpInstance for LinterpInstance {
// 	fn name(&self) -> &str {&self.name}

// 	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
// 		(
// 			vec![self.input_id.clone()],
// 			vec![self.output_id.clone()]
// 		)
// 	}

// 	fn inner_passes(&self) -> Vec<PassID> {
// 		vec![self.forward_id.clone(), self.backward_id.clone()]
// 	}

// 	fn inner_ops(&self) -> Vec<OpID> {vec![]}

// 	fn inner_nodes(&self) -> Vec<NodeID> {vec![]}

// fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{

// 	let input_shape = shapes.get_shape(&self.input_id).to_data_shape()?;

// 	ensure!(input_shape.ndim() == self.factors.len(), "expansion factors must be the same length as input shape");

// 	let output_shape: NodeShape = input_shape.slice().iter().zip(&self.factors).map(|(dim, f)| {
// 			((dim-1)*f + 1, dim * f)
// 		}).into();

// 	shapes.merge_with(&self.output_id, &output_shape)?;
// 	Ok(())
// }
// }

#[derive(Debug, Clone)]
pub struct LinterpInstance {
	input: NodeID,
	output: NodeID,
	factors: Vec<usize>,

	///
	central_range: Range<usize>, /* This is the minimum range within the upscaling factors which contains all
	                              * non-unit entries. */

	///
	upscale_matrix: Vec<f32>,
}

impl LinterpInstance {
	pub fn new(input: NodeID, output: NodeID, factors: Vec<usize>) -> Self {
		let range_start = factors.iter().take_while(|&&i| i == 1).count();
		let range_end = factors.len() - factors.iter().rev().take_while(|&&i| i == 1).count();
		let central_range = range_start..range_end;
		let upscale_matrix = upscale_matrix(&factors[central_range.clone()]);
		LinterpInstance {
			input,
			output,
			factors,
			central_range,
			upscale_matrix,
		}
	}
}

impl OpInstance for LinterpInstance {
	fn type_name(&self) -> &'static str {
		"Linterp"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(Linterp {
			input: graph.node_from_id(self.input),
			output: graph.node_from_id(self.output),
			factors: self.factors.clone(),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		// let expand_grad = if self.input.shape().len() == self.output.shape().len() {
		// 	ctx.grad_of(&self.output)
		// } else {
		// 	expand_dims(ctx.grad_of(&self.output), &self.axes)?
		// };
		// Broadcast::new(expand_grad, ctx.grad_of(&self.input)).build()?;

		LinterpBack::new(ctx.grad_of(&self.output), ctx.grad_of(&self.input), &self.factors).build()?;
		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let input_shape = ctx.input_shape(&self.input);

		debug_assert_eq!(input_shape.slice().len(), self.factors.len());

		let output_shape: NodeShape = input_shape
			.slice()
			.iter()
			.zip(&self.factors)
			.map(
				|(&dim, &f)| {
					if dim == 0 {
						(0, 0)
					} else {
						((dim - 1) * f + 1, dim * f)
					}
				},
			)
			.into();

		ctx.merge_output_shape(&self.output, &output_shape)?;
		Ok(())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let input = ctx.get_input_standard(&self.input);
		let mut output = ctx.get_output_standard(&self.output);

		let input_shape = input.shape();
		let output_shape = output.shape().to_vec();

		let input_spatial = &input_shape[self.central_range.clone()];
		let output_spatial = &output_shape[self.central_range.clone()];
		let factors_spatial = &self.factors[self.central_range.clone()];

		debug_assert_eq!(
			input_shape.len(),
			self.factors.len(),
			"Expansion factors must be the same length as input dimensions"
		); // caught by builder
		debug_assert_eq!(
			input_shape.len(),
			output_shape.len(),
			"Input ndims does not match output ndims"
		); // caught by propagate shapes
		debug_assert!(
			input_shape
				.iter()
				.zip(&output_shape)
				.map(|(i, o)| (o + i - 1) / i)
				.eq(self.factors.iter().cloned()),
			"input shape and factors incompatible with output shape"
		); // caught by propagate shapes

		let input = input.as_slice().unwrap();
		let output = output.as_slice_mut().unwrap();

		let batches = input_shape[..self.central_range.start].iter().product();
		let n_channels = input_shape[self.central_range.end..].iter().product();

		let in_size = input.len() / batches;
		let out_size = output.len() / batches;

		let patch_strides = patch_strides(&input_spatial);
		let n_patches = patch_strides[0] * (input_spatial[0] + 1); // TODO use ensure to guard against zero length shapes

		let k = 2usize.pow(factors_spatial.len() as u32);
		let m = factors_spatial.iter().product();
		let n = n_channels * n_patches;

		let mut hires_matrix = Vec::with_capacity(m * n);
		unsafe {
			hires_matrix.set_len(m * n);
		}

		let mut lores_matrix = Vec::with_capacity(k * n);
		unsafe {
			lores_matrix.set_len(k * n);
		}

		for b_ind in 0..batches {
			let out_batch = &mut output[b_ind * out_size..][..out_size];
			let in_batch = &input[b_ind * in_size..][..in_size];

			for i in 0..n_patches {
				pack_lowres_patch(
					in_batch,
					&input_spatial,
					n_channels,
					i,
					i,
					&patch_strides,
					&mut lores_matrix,
					0,
				);
			}

			unsafe {
				matrixmultiply_mt::sgemm(
					m,
					k,
					n,
					1.0,
					self.upscale_matrix.as_ptr(),
					1,
					m as isize, // A is upscale matrix, col major
					lores_matrix.as_ptr(),
					n as isize,
					1, // B, low res image in patches, row major
					0.0,
					hires_matrix.as_mut_ptr(),
					n as isize,
					1,
				); // C, hires matrix values in patches, row major
			}

			for i in 0..n_patches {
				unpack_hires_patch(
					out_batch,
					&output_spatial,
					n_channels,
					i,
					i,
					&patch_strides,
					&hires_matrix,
					&factors_spatial,
					0,
				)
			}
		}

		Ok(())
	}
}

pub struct LinterpBack {
	input_grad: Node,
	output_grad: Node,
	factors: Vec<usize>,
}

impl LinterpBack {
	pub fn new<I, O>(output_grad: O, input_grad: I, factors: &[usize]) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		let input_grad = input_grad.into();
		let output_grad = output_grad.into();
		LinterpBack {
			input_grad,
			output_grad,
			factors: factors.to_vec(),
		}
	}
}

impl OpSpecification for LinterpBack {
	type InstanceType = LinterpBackInstance;

	fn type_name(&self) -> &'static str {
		"LinterpBack"
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.output_grad.clone()]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.input_grad.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			input_grad: mapping.get(&self.input_grad).unwrap_or(&self.input_grad).clone(),
			output_grad: mapping.get(&self.output_grad).unwrap_or(&self.output_grad).clone(),
			factors: self.factors.clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		if self.factors.iter().any(|&f| f == 0) {
			return Err(format!("expansion factors ({:?}) must all be greater than 0", self.factors,).into());
		}
		if self.output_grad.shape().len() != self.factors.len() {
			return Err(format!(
				"expansion factors ({:?}) and output_grad shape ({}) must have the same length",
				self.factors,
				self.output_grad.shape(),
			)
			.into());
		}
		if self.output_grad.shape().len() != self.input_grad.shape().len() {
			return Err(format!(
				"output_grad shape({}) and input_grad shape ({}) must have the same length",
				self.output_grad.shape(),
				self.input_grad.shape(),
			)
			.into());
		}
		Ok(LinterpBackInstance::new(
			self.input_grad.id(),
			self.output_grad.id(),
			self.factors,
		))
	}
}

#[derive(Debug, Clone)]
pub struct LinterpBackInstance {
	input_grad: NodeID,
	output_grad: NodeID,
	factors: Vec<usize>,
	central_range: Range<usize>, /* This is the minimum range within the upscaling factors which contains all
	                              * non-unit entries. */
	upscale_matrix: Vec<f32>,
}

impl LinterpBackInstance {
	pub fn new(input_grad: NodeID, output_grad: NodeID, factors: Vec<usize>) -> Self {
		let range_start = factors.iter().take_while(|&&i| i == 1).count();
		let range_end = factors.len() - factors.iter().rev().take_while(|&&i| i == 1).count();
		let central_range = range_start..range_end;
		let upscale_matrix = upscale_matrix(&factors[central_range.clone()]);
		LinterpBackInstance {
			input_grad,
			output_grad,
			factors,
			central_range,
			upscale_matrix,
		}
	}
}

impl OpInstance for LinterpBackInstance {
	fn type_name(&self) -> &'static str {
		"LinterpBack"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(LinterpBack {
			input_grad: graph.node_from_id(self.input_grad),
			output_grad: graph.node_from_id(self.output_grad),
			factors: self.factors.clone(),
		})
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.output_grad]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.input_grad]
	}

	fn gradient(&self, _ctx: &mut GradientContext) -> Result<(), GradientError> {
		Err(GradientError::Unimplemented)
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let output_shape = ctx.input_shape(&self.output_grad);

		debug_assert_eq!(output_shape.slice().len(), self.factors.len()); // ensured by builder

		let input_shape: NodeShape = output_shape
			.slice()
			.iter()
			.zip(&self.factors)
			.map(|(dim, f)| (dim + f - 1) / f)
			.into();

		ctx.merge_output_shape(&self.input_grad, &input_shape)?;
		Ok(())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let mut input_grad = ctx.get_output_standard(&self.input_grad);
		let output_grad = ctx.get_input_standard(&self.output_grad);

		let input_shape = input_grad.shape().to_vec();
		let output_shape = output_grad.shape();

		let input_spatial = &input_shape[self.central_range.clone()];
		let output_spatial = &output_shape[self.central_range.clone()];
		let factors_spatial = &self.factors[self.central_range.clone()];

		debug_assert_eq!(
			input_shape.len(),
			self.factors.len(),
			"Expansion factors must be the same length as input dimensions"
		); // caught by builder
		debug_assert_eq!(
			input_shape.len(),
			output_shape.len(),
			"Input ndims does not match output ndims"
		); // caught by propagate shapes
		debug_assert!(
			input_shape
				.iter()
				.zip(output_shape)
				.map(|(i, o)| (o + i - 1) / i)
				.eq(self.factors.iter().cloned()),
			"input shape and factors incompatible with output shape"
		); // caught by propagate shapes

		let input_grad = input_grad.as_slice_mut().unwrap();
		let output_grad = output_grad.as_slice().unwrap();

		let batches = input_shape[..self.central_range.start].iter().product();
		let n_channels = input_shape[self.central_range.end..].iter().product();

		let in_size = input_grad.len() / batches;
		let out_size = output_grad.len() / batches;

		let factor_strides = factor_strides(factors_spatial);

		let patch_strides = patch_strides(&input_spatial);
		let n_patches = patch_strides[0] * (input_spatial[0] + 1);

		let k: usize = factors_spatial.iter().product(); // k and m are swapped vs forward pass
		let m = 2usize.pow(factors_spatial.len() as u32);
		let n = n_channels * n_patches;

		let mut hires_matrix = Vec::with_capacity(k * n);
		unsafe {
			hires_matrix.set_len(k * n);
		}

		let mut lores_matrix = Vec::with_capacity(m * n);
		unsafe {
			lores_matrix.set_len(m * n);
		}

		for b_ind in 0..batches {
			let out_grad_batch = &output_grad[b_ind * out_size..][..out_size];
			let in_grad_batch = &mut input_grad[b_ind * in_size..][..in_size];

			for i in 0..n_patches {
				pack_hires_patch(
					out_grad_batch,
					&output_spatial,
					n_channels,
					i,
					i,
					&patch_strides,
					&mut hires_matrix,
					&factors_spatial,
					&factor_strides,
					0,
				)
			}

			unsafe {
				matrixmultiply_mt::sgemm(
					m,
					k,
					n,
					1.0,
					self.upscale_matrix.as_ptr(),
					k as isize,
					1, // A is upscale matrix, row major
					hires_matrix.as_ptr(),
					n as isize,
					1, // B, hires image in patches, row major
					0.0,
					lores_matrix.as_mut_ptr(),
					n as isize,
					1,
				); // C, lores matrix values in patches, row major
			}

			for i in 0..n_patches {
				unpack_lowres_patch(
					in_grad_batch,
					&input_spatial,
					n_channels,
					i,
					i,
					&patch_strides,
					&lores_matrix,
					0,
				);
			}
		}

		Ok(())
	}
}

/// "batch" dimensions are all outer dimensions which have a factor equal to 1
/// "channel" dimensions are all inner dimensions which have a factor equal to 1
/// "spatial" dimensions are all central channels inclusive of the first and last non-1 factor

/// Keeps a running product of spatial dimensions + 1, returning the stride of the patches
/// [7, 3, 5, 2] returns [72, 18, 3, 1]
fn patch_strides(input_shape: &[usize]) -> SmallVec<[usize; 6]> {
	let mut strides = input_shape
		.iter()
		.rev()
		.scan(1, |state, &i| {
			let res = Some(*state);
			*state *= i + 1;
			res
		})
		.collect::<SmallVec<[usize; 6]>>();
	strides.reverse();
	strides
}

fn factor_strides(factors: &[usize]) -> SmallVec<[usize; 6]> {
	let mut strides = factors
		.iter()
		.rev()
		.scan(1, |state, &i| {
			let res = Some(*state);
			*state *= i;
			res
		})
		.collect::<SmallVec<[usize; 6]>>();
	strides.reverse();
	strides
}

/// Packs values from lowres tensor/node into the B matrix for C = A x B upscaling
#[allow(clippy::too_many_arguments)]
fn pack_lowres_patch(
	input: &[f32],
	input_spatial: &[usize],
	channels: usize,
	patch_index: usize,
	patch_index_rem: usize,
	patch_strides: &[usize],
	matrix: &mut [f32],
	axis: usize,
) {
	let len = matrix.len();
	debug_assert_eq!(0, len % 2);

	let in_stride = input.len() / input_spatial[axis];
	let patch_x = patch_index_rem / patch_strides[axis];
	let in_x = patch_x as isize - 1; // start of the image patch in the current recurse.

	for (i, new_matrix) in matrix.chunks_mut(len / 2).enumerate() {
		let in_x = min(input_spatial[axis] - 1, max(0, in_x + i as isize) as usize); // handle image boundaries by repeating samples inside the valid range.

		let new_input = &input[in_x * in_stride..(in_x + 1) * in_stride];
		if axis < input_spatial.len() - 1 {
			let new_patch_index_rem = patch_index_rem - patch_x * patch_strides[axis];
			let new_axis = axis + 1;
			pack_lowres_patch(
				new_input,
				input_spatial,
				channels,
				patch_index,
				new_patch_index_rem,
				patch_strides,
				new_matrix,
				new_axis,
			);
		} else {
			debug_assert_eq!(new_input.len(), channels);
			debug_assert_eq!(0, new_matrix.len() % channels);
			debug_assert_eq!(patch_strides[axis], 1);

			let new_input = &new_input[0..channels];

			let m = &mut new_matrix[patch_index * channels..(patch_index + 1) * channels];
			m.copy_from_slice(new_input);
		}
	}
}

#[allow(clippy::too_many_arguments)]
fn unpack_lowres_patch(
	input_grad: &mut [f32],
	input_spatial: &[usize],
	channels: usize,
	patch_index: usize,
	patch_index_rem: usize,
	patch_strides: &[usize],
	matrix: &[f32],
	axis: usize,
) {
	let len = matrix.len();
	debug_assert_eq!(0, len % 2);

	let in_stride = input_grad.len() / input_spatial[axis];
	let patch_x = patch_index_rem / patch_strides[axis];
	let in_x = patch_x as isize - 1; // start of the image patch in the current recurse.

	for (i, new_matrix) in matrix.chunks(len / 2).enumerate() {
		let in_x = min(input_spatial[axis] - 1, max(0, in_x + i as isize) as usize); // handle image boundaries by repeating samples inside the valid range.

		let new_input_grad = &mut input_grad[in_x * in_stride..(in_x + 1) * in_stride];
		if axis < input_spatial.len() - 1 {
			let new_patch_index_rem = patch_index_rem - patch_x * patch_strides[axis];
			let new_axis = axis + 1;
			unpack_lowres_patch(
				new_input_grad,
				input_spatial,
				channels,
				patch_index,
				new_patch_index_rem,
				patch_strides,
				new_matrix,
				new_axis,
			);
		} else {
			debug_assert_eq!(new_input_grad.len(), channels);
			debug_assert_eq!(0, new_matrix.len() % channels);
			debug_assert_eq!(patch_strides[axis], 1);

			let new_input_grad = &mut new_input_grad[0..channels];

			let m = &new_matrix[patch_index * channels..(patch_index + 1) * channels];

			for j in 0..channels {
				new_input_grad[j] += m[j];
			}
		}
	}
}

/// reads from C matrix into tensor/node after C= A x B upscaling
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn unpack_hires_patch(
	output: &mut [f32],
	output_spatial: &[usize],
	n_channels: usize,
	patch_index: usize,
	patch_index_rem: usize,
	patch_strides: &[usize],
	matrix: &[f32],
	factors: &[usize],
	axis: usize,
) {
	let len = matrix.len();
	debug_assert_eq!(0, len % factors[axis]);

	let out_stride = output.len() / output_spatial[axis];
	let patch_x = patch_index_rem / patch_strides[axis];
	let out_x = (patch_x as isize - 1) * factors[axis] as isize + factors[axis] as isize / 2; // start of the image patch in the current recurse.

	// println!("{} {}",output.len(), output_shape[axis]);
	for (i, new_matrix) in matrix.chunks(len / factors[axis]).enumerate() {
		let out_x = out_x + i as isize;
		let out_x = if out_x < 0 || out_x >= output_spatial[axis] as isize {
			// handle image boundaries by skipping them
			continue;
		} else {
			out_x as usize
		};

		// println!("{}", out_x);
		let new_output = &mut output[out_x * out_stride..(out_x + 1) * out_stride];
		if axis < output_spatial.len() - 1 {
			let new_patch_index_rem = patch_index_rem - patch_x * patch_strides[axis];
			let new_axis = axis + 1;
			unpack_hires_patch(
				new_output,
				output_spatial,
				n_channels,
				patch_index,
				new_patch_index_rem,
				patch_strides,
				new_matrix,
				factors,
				new_axis,
			);
		} else {
			debug_assert_eq!(new_output.len(), n_channels);
			debug_assert_eq!(0, new_matrix.len() % n_channels);
			debug_assert_eq!(patch_strides[axis], 1);

			let new_output = &mut new_output[0..n_channels];

			let m = &new_matrix[patch_index * n_channels..][..n_channels];

			for j in 0..n_channels {
				new_output[j] += m[j];
			}
		}
	}
}

/// reads from C matrix into tensor/node after C= A x B upscaling
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn pack_hires_patch(
	output_grad: &[f32],
	output_spatial: &[usize],
	n_channels: usize,
	patch_index: usize,
	patch_index_rem: usize,
	patch_strides: &[usize],
	matrix: &mut [f32],
	factors: &[usize],
	factor_strides: &[usize],
	axis: usize,
) {
	let len = matrix.len();
	debug_assert_eq!(0, len % factors[axis]);

	let out_stride = output_grad.len() / output_spatial[axis];
	let patch_x = patch_index_rem / patch_strides[axis];
	let out_x = (patch_x as isize - 1) * factors[axis] as isize + factors[axis] as isize / 2; // start of the image patch in the current recurse.

	for (i, new_matrix) in matrix.chunks_mut(len / factors[axis]).enumerate() {
		let out_x = out_x + i as isize;
		let out_x = if out_x < 0 || out_x >= output_spatial[axis] as isize {
			// handle image boundaries of this axis by zeroing/skipping them

			// unlike unpack_hiris_patch(), we need to zero out mismatches between patches and the output, else
			// gradients from a previous patch may remain.
			let new_len = new_matrix.len();
			for m in new_matrix.chunks_mut(new_len / factor_strides[axis]) {
				for i in 0..n_channels {
					m[patch_index * n_channels + i] = 0.0;
				}
			}
			continue;
		} else {
			out_x as usize
		};

		let new_output_grad = &output_grad[out_x * out_stride..(out_x + 1) * out_stride];
		if axis < output_spatial.len() - 1 {
			let new_patch_index_rem = patch_index_rem - patch_x * patch_strides[axis];
			let new_axis = axis + 1;
			pack_hires_patch(
				new_output_grad,
				output_spatial,
				n_channels,
				patch_index,
				new_patch_index_rem,
				patch_strides,
				new_matrix,
				factors,
				factor_strides,
				new_axis,
			);
		} else {
			debug_assert_eq!(new_output_grad.len(), n_channels);
			debug_assert_eq!(0, new_matrix.len() % n_channels);
			debug_assert_eq!(patch_strides[axis], 1);

			let new_output_grad = &new_output_grad[0..n_channels];

			let m = &mut new_matrix[patch_index * n_channels..][..n_channels];
			m.copy_from_slice(new_output_grad);
		}
	}
}

/// Returns the A matrix for C = A x B
/// column major
/// shape is : W.H x (2^D)
/// where D is the number of spatial dimensions
#[allow(dead_code)]
fn upscale_matrix(factors: &[usize]) -> Vec<f32> {
	let cols = 2usize.pow(factors.len() as u32);
	let rows: usize = factors.iter().product();
	let mut out = vec![0.0; cols * rows];
	out[0] = 1.0;

	for axis in 0..factors.len() {
		fill_next_axis(factors, axis, rows, &mut out);
	}

	out
}

fn fill_next_axis(factors: &[usize], axis: usize, col_stride: usize, matrix: &mut [f32]) {
	let f = factors[axis];
	let step = 1.0 / f as f32;
	let start = 1.0 - ((f + 1) % 2) as f32 * 0.5 * step;

	// size of already filled patch;
	let cols = 2usize.pow(axis as u32);
	let rows = factors[0..axis].iter().product();
	let col_offset = col_stride * cols;

	for i in (0..f).rev() {
		// must do blocks in reverse order so that data in top left of matrix isnt overwritten too early
		let mult1 = start - step * i as f32;
		let mult2 = 1.0 - mult1;

		let row_offset = rows * i;

		for col in 0..cols {
			for row in 0..rows {
				let ind = row + col * col_stride;
				let val = matrix[ind];
				matrix[ind + row_offset] = mult1 * val;
				matrix[ind + row_offset + col_offset] = mult2 * val;
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::{linterp, pack_lowres_patch, patch_strides, unpack_hires_patch, upscale_matrix, Linterp};
	use alumina_core::{base_ops::OpSpecification, graph::Node};
	use alumina_test::grad_numeric_test::GradNumericTest;
	use indexmap::indexset;
	use ndarray::arr1;

	#[test]
	fn grad_numeric_test1() {
		let input = Node::new(&[2, 4, 3, 5]).set_name("input");

		let output = linterp(&input, &[1, 2, 2, 1]).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[3, 5, 1, 7, 13]).set_name("input");

		let output = linterp(&input, &[1, 3, 1, 3, 1]).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}

	#[test]
	fn forward_test() {
		{
			let input = Node::new(&[1, 2, 1, 2, 1]).set_name("input").set_value(
				arr1(&[0.0, 1.0, 1.0, 0.0])
					.into_shape([1, 2, 1, 2, 1])
					.unwrap()
					.into_dyn(),
			);
			let output = Node::new(&[1, 4, 1, 4, 1]).set_name("output");
			let _o1 = Linterp::new(&input, &output, &[1, 2, 1, 2, 1]).build();

			let out = output.calc().unwrap();
			let out_slice = out.as_slice().unwrap();

			let expected = vec![
				0.0, 0.25, 0.75, 1.0, 0.25, 0.375, 0.625, 0.75, 0.75, 0.625, 0.375, 0.25, 1.0, 0.75, 0.25, 0.0,
			];

			let diff = out_slice
				.iter()
				.zip(expected.iter())
				.fold(0.0, |acc, (&o, &e)| acc + (o - e) * (o - e));
			assert!(diff < 1e-6, "{:?} {:?}", out_slice, expected);
		}

		{
			let input = Node::new(&[1, 3, 1, 3, 1]).set_name("input").set_value(
				arr1(&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
					.into_shape([1, 3, 1, 3, 1])
					.unwrap()
					.into_dyn(),
			);
			let output = Node::new(&[1, 9, 1, 9, 1]).set_name("output");
			let _o1 = Linterp::new(&input, &output, &[1, 3, 1, 3, 1]).build();

			let out = output.calc().unwrap();
			let out_slice = out.as_slice().unwrap();

			let expected = vec![
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				1. / 9.,
				2. / 9.,
				3. / 9.,
				2. / 9.,
				1. / 9.,
				0.0,
				0.0,
				0.0,
				0.0,
				2. / 9.,
				4. / 9.,
				6. / 9.,
				4. / 9.,
				2. / 9.,
				0.0,
				0.0,
				0.0,
				0.0,
				3. / 9.,
				6. / 9.,
				1.0,
				6. / 9.,
				3. / 9.,
				0.0,
				0.0,
				0.0,
				0.0,
				2. / 9.,
				4. / 9.,
				6. / 9.,
				4. / 9.,
				2. / 9.,
				0.0,
				0.0,
				0.0,
				0.0,
				1. / 9.,
				2. / 9.,
				3. / 9.,
				2. / 9.,
				1. / 9.,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
			];

			let diff = out_slice
				.iter()
				.zip(expected.iter())
				.fold(0.0, |acc, (&o, &e)| acc + (o - e) * (o - e));
			assert!(diff < 1e-6, "/n{:?} /n{:?}/n", out_slice, expected);
		}
	}

	#[test]
	fn test_linear_interp_matrix() {
		_test_linear_interp_matrix();
	}

	fn _test_linear_interp_matrix() {
		let factors_list = vec![
			vec![3],
			vec![2],
			vec![3, 2],
			vec![1, 1, 1],
			vec![13, 31, 5],
			vec![16, 4, 8, 2],
		];

		for factors in factors_list {
			let matrix = upscale_matrix(&factors[..]);
			let cols = 2usize.pow(factors.len() as u32);
			let rows: usize = factors.iter().product();

			for row in 0..rows {
				let mut sum = 0.0;
				for col in 0..cols {
					sum += matrix[row + col * rows]
				}

				assert!((1.0 - sum).abs() < 1e-6, "sum: {} matrix:\n{:?}", sum, matrix);
			}
		}

		{
			let matrix = upscale_matrix(&[3]);
			let expected = vec![
				// col major
				1.0,
				2. / 3.,
				1. / 3.,
				0.0,
				1. / 3.,
				2. / 3.,
			];

			let diff = matrix
				.iter()
				.zip(expected.iter())
				.fold(0.0, |acc, (&m, &e)| acc + (m - e) * (m - e));
			assert!(diff < 1e-6, "{:?} {:?}", matrix, expected);
		}

		{
			let matrix = upscale_matrix(&[2]);
			let expected = vec![
				// col major
				0.75, 0.25, 0.25, 0.75,
			];

			let diff = matrix
				.iter()
				.zip(expected.iter())
				.fold(0.0, |acc, (&m, &e)| acc + (m - e) * (m - e));
			assert!(diff < 1e-6, "{:?} {:?}", matrix, expected);
		}
	}

	#[test]
	fn test_pack_lowres_recurse() {
		_test_pack_lowres_recurse();
	}

	fn _test_pack_lowres_recurse() {
		let input_spatial = vec![2, 3];
		let n_channels = 5usize;

		let input_size = input_spatial.iter().fold(n_channels, |acc, v| acc * v);
		let input = (0..input_size).map(|x| x as f32).collect::<Vec<_>>();

		let n_patches = input_spatial.iter().fold(1, |acc, v| acc * (v + 1));
		let axis = 0;

		let rows = 2usize.pow(input_spatial.len() as u32);
		let mut matrix = vec![0.0; rows * n_channels * n_patches];
		let strides = patch_strides(&input_spatial);

		for index in 0..n_patches {
			pack_lowres_patch(
				&input,
				&input_spatial,
				n_channels,
				index,
				index,
				&strides,
				&mut matrix,
				axis,
			);
		}

		let expected = vec![
			0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8,
			9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 0,
			1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
			12, 13, 14, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 25, 26, 27, 28,
			29, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 15, 16, 17, 18,
			19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 15, 16, 17, 18, 19, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
			26, 27, 28, 29, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
			20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 25, 26, 27, 28, 29, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
			27, 28, 29, 25, 26, 27, 28, 29,
		]
		.iter()
		.map(|x: &usize| *x as f32)
		.collect::<Vec<_>>();

		assert_eq!(expected, matrix);
	}

	#[test]
	fn test_unpack_hires_matrix() {
		_test_unpack_hires_matrix();
	}

	fn _test_unpack_hires_matrix() {
		let input_spatial = vec![3, 2];
		let factors = vec![2, 2];
		let output_spatial = input_spatial
			.iter()
			.zip(factors.iter())
			.map(|(x, f)| x * f)
			.collect::<Vec<usize>>();
		let n_channels = 1usize;

		let output_size = output_spatial.iter().fold(n_channels, |acc, v| acc * v);
		let mut output = vec![0.0; output_size];

		let n_patches: usize = input_spatial.iter().fold(1, |acc, v| acc * (v + 1));
		let axis = 0;

		let rows: usize = factors.iter().product(); // 2usize.pow(input_shape.len()as u32);
		let matrix = (0..rows * n_channels * n_patches)
			.map(|x: usize| (x / n_patches + (x % n_patches) * rows) as f32)
			.collect::<Vec<f32>>();
		let strides = patch_strides(&input_spatial);

		for index in 0..n_patches {
			unpack_hires_patch(
				&mut output,
				&output_spatial,
				n_channels,
				index,
				index,
				&strides,
				&matrix,
				&factors,
				axis,
			);
		}

		let expected = vec![
			3, 6, 7, 10, 13, 16, 17, 20, 15, 18, 19, 22, 25, 28, 29, 32, 27, 30, 31, 34, 37, 40, 41, 44,
		]
		.iter()
		.map(|x: &usize| *x as f32)
		.collect::<Vec<_>>();

		assert_eq!(expected, output);
	}
}
