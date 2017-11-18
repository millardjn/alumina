use new::graph::{GraphDef, NodeID, OpID, PassID, DataID, Storage, GraphShapes, Result};
use new::ops::{standard_op_name, Op, OpInstance, Pass};
use new::shape::NodeShape;
use ndarray::{ArrayD, Dimension, IxDyn};
use std::any::Any;
use std::cmp::{min, max};
use std::ops::Range;
use smallvec::SmallVec;
use matrixmultiply;


/// Linterp implements linear interpolation upscaling
///
/// Increase size of each spatial dimension by given a factor by linear interpolating between spaxels in the input
#[derive(Clone)] 
pub struct Linterp {
	name: Option<String>,
	factors: Vec<usize>,
	input_id: NodeID,
	output_id: NodeID,
}

impl Linterp {
	pub fn new(input_id: &NodeID, output_id: &NodeID, factors: &[usize]) -> Self{
		Linterp {
			name: None,
			input_id: input_id.clone(),
			output_id: output_id.clone(),
			factors: factors.to_vec(),
		}
	}
}

impl Op for Linterp {
	type InstanceType = LinterpInstance;

	fn type_name(&self) -> &'static str {
		"Linterp"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		let name =standard_op_name(&self, &self.name, graph, &[self.input_id.clone()], &[self.output_id.clone()]);

		Ok(LinterpInstance{
			name: name,
			input_id: self.input_id.clone(),
			output_id: self.output_id.clone(),
			factors: self.factors.clone(),
			forward_id:graph.add_pass(LinterpForward::new(
				self.input_id.clone(),
				self.output_id.clone(),
				self.factors.clone(),
			)),
			backward_id:graph.add_pass(LinterpBackward::new(
				self.input_id.clone(),
				self.output_id.clone(),
				self.factors.clone(),
			)),
		})
	}
}

#[derive(Debug, Clone)]
pub struct LinterpInstance {
	name: String,
	input_id: NodeID,
	output_id: NodeID,
	factors: Vec<usize>,
	forward_id: PassID,
	backward_id: PassID,
}

impl OpInstance for LinterpInstance {
	fn instance_name(&self) -> &str {&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		(
			vec![self.input_id.clone()],
			vec![self.output_id.clone()]
		)
	}

	fn inner_passes(&self) -> Vec<PassID> {
		vec![self.forward_id.clone(), self.backward_id.clone()]
	}

	fn inner_ops(&self) -> Vec<OpID> {vec![]}

	fn inner_nodes(&self) -> Vec<NodeID> {vec![]}

	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{

		let input_shape = shapes.get_shape(&self.input_id).to_data_shape()?;

		ensure!(input_shape.ndim() == self.factors.len(), "expansion factors must be the same length as input shape");

		let output_shape: NodeShape = input_shape.slice().iter().zip(&self.factors).map(|(dim, f)| {
				((dim-1)*f + 1, dim * f)
			}).into();

		shapes.merge_with(&self.output_id, &output_shape)?;
		Ok(())
	}

}

#[derive(Debug, Clone)]
pub struct LinterpForward {
	input_id: NodeID,
	output_id: NodeID,
	factors: Vec<usize>,
	central_range: Range<usize>, // This is the minimum range within the upscaling factors which contains all non-unit entries.
	upscale_matrix: Vec<f32>,
}

impl LinterpForward {
	pub fn new(input_id: NodeID, output_id: NodeID, factors: Vec<usize>) -> Self{
		let range_start = factors.iter().take_while(|&&i| i == 1).count();
		let range_end = factors.len() - factors.iter().rev().take_while(|&&i| i == 1).count();
		let central_range = range_start..range_end;
		let upscale_matrix = upscale_matrix(&factors[central_range.clone()]);
		LinterpForward {
			input_id,
			output_id,
			factors,
			central_range,
			upscale_matrix,
		}
	}
}

impl Pass for LinterpForward {
	fn type_name(&self) -> &'static str {"LinterpForward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input_id.value_id()],
		vec![self.output_id.value_id()])
	}

	fn run(&self, data: &Storage) -> Result<Box<Any>> {
		let input = data.get(&self.input_id.value_id())?;
		let mut output = data.get_mut(&self.output_id.value_id())?;

		let input_shape = input.shape();
		let output_shape = output.shape().to_vec();
		
		let input_spatial = &input_shape[self.central_range.clone()];
		let output_spatial = &output_shape[self.central_range.clone()];
		let factors_spatial = &self.factors[self.central_range.clone()];
		
		ensure!(input_shape.len() == output_shape.len(), "Input ndims does not match output ndims");
		ensure!(input_shape.len() == self.factors.len(), "Expansion factors must be the same length as input dimensions");
		ensure!(input_shape.iter().zip(&output_shape).map(|(i, o)| (o + i - 1)/i).eq(self.factors.iter().cloned()), "input shape and factors incompatible with output shape");

		let input = input.as_slice().unwrap();
		let output = output.as_slice_mut().unwrap();

		let batches = input_shape[..self.central_range.start].iter().product();
		let n_channels = input_shape[self.central_range.end..].iter().product();

		let in_size = input.len()/batches;
		let out_size = output.len()/batches;

		let patch_strides = patch_strides(&input_spatial);
		let n_patches = patch_strides[0] * (input_spatial[0] + 1);

		let k = 2usize.pow(factors_spatial.len() as u32);
		let m = factors_spatial.iter().product();
		let n = n_channels * n_patches;

		let mut hires_matrix = Vec::with_capacity(m * n);
		unsafe{
			hires_matrix.set_len(m * n);
		}

		let mut lores_matrix = Vec::with_capacity(k * n);
		unsafe{
			lores_matrix.set_len(k * n);
		}

		for b_ind in 0..batches{
			let out_batch = &mut output[b_ind*out_size..][..out_size];
			let in_batch = &input[b_ind*in_size..][..in_size];

			for i in 0..n_patches{
				pack_lowres_patch(in_batch, &input_spatial, n_channels, i, i, &patch_strides, &mut lores_matrix, 0);
			}
			
			unsafe{
				matrixmultiply::sgemm(m, k, n,
					1.0,
					self.upscale_matrix.as_ptr(), 1, m as isize, // A is upscale matrix, col major
					lores_matrix.as_ptr(), n as isize, 1, // B, low res image in patches, row major
					0.0,
					hires_matrix.as_mut_ptr(), n as isize, 1); // C, hires matrix values in patches, row major
			}

			for i in 0..n_patches{
				unpack_hires_patch(out_batch, &output_spatial, n_channels, i, i, &patch_strides, &mut hires_matrix, &factors_spatial, 0)
			}
		}

		Ok(Box::new(()))
	}
}


#[derive(Debug, Clone)]
pub struct LinterpBackward {
	input_id: NodeID,
	output_id: NodeID,
	factors: Vec<usize>,
	central_range: Range<usize>, // This is the minimum range within the upscaling factors which contains all non-unit entries.
	upscale_matrix: Vec<f32>,
}

impl LinterpBackward {
	pub fn new(input_id: NodeID, output_id: NodeID, factors: Vec<usize>) -> Self{
		let range_start = factors.iter().take_while(|&&i| i == 1).count();
		let range_end = factors.len() - factors.iter().rev().take_while(|&&i| i == 1).count();
		let central_range = range_start..range_end;
		let upscale_matrix = upscale_matrix(&factors[central_range.clone()]);
		LinterpBackward {
			input_id,
			output_id,
			factors,
			central_range,
			upscale_matrix,
		}
	}
}

impl Pass for LinterpBackward {
	fn type_name(&self) -> &'static str {"LinterpBackward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.output_id.gradient_id()],
		vec![self.input_id.gradient_id()])
	}

	fn run(&self, data: &Storage) -> Result<Box<Any>> {
		let mut input_grad = data.get_mut(&self.input_id.gradient_id())?;
		let output_grad = data.get(&self.output_id.gradient_id())?;

		let input_shape = input_grad.shape().to_vec();
		let output_shape = output_grad.shape();
		
		let input_spatial = &input_shape[self.central_range.clone()];
		let output_spatial = &output_shape[self.central_range.clone()];
		let factors_spatial = &self.factors[self.central_range.clone()];
		
		ensure!(input_shape.len() == output_shape.len(), "Input ndims does not match output ndims");
		ensure!(input_shape.len() == self.factors.len(), "Expansion factors must be the same length as input dimensions");
		ensure!(input_shape.iter().zip(output_shape).map(|(i, o)| (o + i - 1)/i).eq(self.factors.iter().cloned()), "input shape and factors incompatible with output shape");

		let input_grad = input_grad.as_slice_mut().unwrap();
		let output_grad = output_grad.as_slice().unwrap();

		let batches = input_shape[..self.central_range.start].iter().product();
		let n_channels = input_shape[self.central_range.end..].iter().product();

		let in_size = input_grad.len()/batches;
		let out_size = output_grad.len()/batches;

		let factor_strides = factor_strides(factors_spatial);

		let patch_strides = patch_strides(&input_spatial);
		let n_patches = patch_strides[0] * (input_spatial[0] + 1);

		let k: usize = factors_spatial.iter().product(); // k and m are swapped vs forward pass
		let m = 2usize.pow(factors_spatial.len() as u32);
		let n = n_channels * n_patches;

		let mut hires_matrix = Vec::with_capacity(k * n);
		unsafe{
			hires_matrix.set_len(k * n);
		}

		let mut lores_matrix = Vec::with_capacity(m * n);
		unsafe{
			lores_matrix.set_len(m * n);
		}

		for b_ind in 0..batches{
			let out_grad_batch = &output_grad[b_ind*out_size..][..out_size];
			let in_grad_batch = &mut input_grad[b_ind*in_size..][..in_size];

			for i in 0..n_patches{
				pack_hires_patch(out_grad_batch, &output_spatial, n_channels, i, i, &patch_strides, &mut hires_matrix, &factors_spatial, &factor_strides, 0)
			}

			unsafe{
				matrixmultiply::sgemm(m, k, n,
					1.0,
					self.upscale_matrix.as_ptr(), k as isize, 1, // A is upscale matrix, row major
					hires_matrix.as_ptr(), n as isize, 1, // B, hires image in patches, row major
					0.0,
					lores_matrix.as_mut_ptr(), n as isize, 1); // C, lores matrix values in patches, row major
			}

			for i in 0..n_patches{
				unpack_lowres_patch(in_grad_batch, &input_spatial, n_channels, i, i, &patch_strides, &mut lores_matrix, 0);
			}
		}

		Ok(Box::new(()))
	}
}


/// "batch" dimensions are all outer dimensions which have a factor equal to 1
/// "channel" dimensions are all inner dimensions which have a factor equal to 1
/// "spatial" dimensions are all central channels inclusive of the first and last non-1 factor



/// Keeps a running product of spatial dimensions + 1, returning the stride of the patches
/// [7, 3, 5, 2] returns [72, 18, 3, 1]
fn patch_strides(input_shape: &[usize]) -> SmallVec<[usize;6]>{
	let mut strides = input_shape.iter().rev().scan(1, |state, &i| {
		let res = Some(*state);
		*state *= i + 1;
		res
	}).collect::<SmallVec<[usize;6]>>();
	strides.reverse();
	strides
}

fn factor_strides(factors: &[usize]) -> SmallVec<[usize;6]>{
	let mut strides = factors.iter().rev().scan(1, |state, &i| {
		let res = Some(*state);
		*state *= i;
		res
	}).collect::<SmallVec<[usize;6]>>();
	strides.reverse();
	strides
}

/// Packs values from lowres tensor/node into the B matrix for C = A x B upscaling
fn pack_lowres_patch(input: &[f32], input_spatial: &[usize], channels: usize, patch_index: usize, patch_index_rem: usize, patch_strides: &[usize], matrix: &mut [f32], axis: usize){
	let len = matrix.len();
	debug_assert_eq!(0, len%2);


	let in_stride = input.len()/input_spatial[axis];
	let patch_x = patch_index_rem/patch_strides[axis];
	let in_x = patch_x as isize - 1; // start of the image patch in the current recurse.

	
	for (i, new_matrix) in matrix.chunks_mut(len/2).enumerate(){
		let in_x = min(input_spatial[axis] - 1 ,max(0, in_x+i as isize)as usize); // handle image boundaries by repeating samples inside the valid range.

		let new_input = &input[in_x*in_stride..(in_x+1)*in_stride];
		if axis < input_spatial.len() - 1 {
			let new_patch_index_rem = patch_index_rem - patch_x * patch_strides[axis];
			let new_axis = axis + 1;			
			pack_lowres_patch(new_input, input_spatial, channels, patch_index, new_patch_index_rem, patch_strides, new_matrix, new_axis);
		} else {
			debug_assert_eq!(new_input.len(), channels);
			debug_assert_eq!(0, new_matrix.len() % channels);
			debug_assert_eq!(patch_strides[axis], 1);

			let new_input = &new_input[0..channels];
			
			let m = &mut new_matrix[patch_index*channels..(patch_index+1)*channels];
			m.copy_from_slice(new_input);

		}
	}
}


fn unpack_lowres_patch(input_grad: &mut [f32], input_spatial: &[usize], channels: usize, patch_index: usize, patch_index_rem: usize, patch_strides: &[usize], matrix: &[f32], axis: usize){
	let len = matrix.len();
	debug_assert_eq!(0, len%2);


	let in_stride = input_grad.len()/input_spatial[axis];
	let patch_x = patch_index_rem/patch_strides[axis];
	let in_x = patch_x as isize - 1; // start of the image patch in the current recurse.

	for (i, new_matrix) in matrix.chunks(len/2).enumerate(){
		let in_x = min(input_spatial[axis] - 1 ,max(0, in_x+i as isize)as usize); // handle image boundaries by repeating samples inside the valid range.

		let new_input_grad = &mut input_grad[in_x*in_stride..(in_x+1)*in_stride];
		if axis < input_spatial.len() - 1 {
			let new_patch_index_rem = patch_index_rem - patch_x * patch_strides[axis];
			let new_axis = axis + 1;			
			unpack_lowres_patch(new_input_grad, input_spatial, channels, patch_index, new_patch_index_rem, patch_strides, new_matrix, new_axis);
		} else {
			debug_assert_eq!(new_input_grad.len(), channels);
			debug_assert_eq!(0, new_matrix.len() % channels);
			debug_assert_eq!(patch_strides[axis], 1);

			let new_input_grad = &mut new_input_grad[0..channels];
			
			let m = &new_matrix[patch_index*channels..(patch_index+1)*channels];

			for j in 0..channels{
				new_input_grad[j] += m[j];
			}
		}
	}
}

/// reads from C matrix into tensor/node after C= A x B upscaling
#[allow(dead_code)]
fn unpack_hires_patch(output: &mut [f32], output_spatial: &[usize], n_channels: usize, patch_index: usize, patch_index_rem: usize, patch_strides: &[usize], matrix: &[f32], factors: &[usize], axis: usize){
	let len = matrix.len();
	debug_assert_eq!(0, len%factors[axis]);


	let out_stride = output.len()/output_spatial[axis];
	let patch_x = patch_index_rem/patch_strides[axis];
	let out_x = (patch_x as isize-1)*factors[axis]as isize + factors[axis]as isize/2 ; // start of the image patch in the current recurse.

	//println!("{} {}",output.len(), output_shape[axis]);
	for (i, new_matrix) in matrix.chunks(len/factors[axis]).enumerate(){
		let out_x = out_x + i as isize;
		let out_x = if out_x < 0 || out_x >= output_spatial[axis] as isize { // handle image boundaries by skipping them
			continue;
		} else {
			out_x as usize
		};

		//println!("{}", out_x);
		let new_output = &mut output[out_x*out_stride..(out_x+1)*out_stride];
		if axis < output_spatial.len() - 1 {
			let new_patch_index_rem = patch_index_rem - patch_x * patch_strides[axis];
			let new_axis = axis + 1;
			unpack_hires_patch(new_output, output_spatial, n_channels, patch_index, new_patch_index_rem, patch_strides, new_matrix, factors, new_axis);
		} else {
			debug_assert_eq!(new_output.len(), n_channels);
			debug_assert_eq!(0, new_matrix.len() % n_channels);
			debug_assert_eq!(patch_strides[axis], 1);

			let new_output = &mut new_output[0..n_channels];
			
			let m = &new_matrix[patch_index*n_channels..][..n_channels];

			for j in 0..n_channels{
				new_output[j] += m[j];
			}
		}
	}
}

/// reads from C matrix into tensor/node after C= A x B upscaling
#[allow(dead_code)]
fn pack_hires_patch(output_grad: &[f32], output_spatial: &[usize], n_channels: usize, patch_index: usize, patch_index_rem: usize, patch_strides: &[usize], matrix: &mut [f32], factors: &[usize], factor_strides: &[usize], axis: usize){
	let len = matrix.len();
	debug_assert_eq!(0, len%factors[axis]);


	let out_stride = output_grad.len()/output_spatial[axis];
	let patch_x = patch_index_rem/patch_strides[axis];
	let out_x = (patch_x as isize-1)*factors[axis]as isize + factors[axis]as isize/2 ; // start of the image patch in the current recurse.


	for (i, new_matrix) in matrix.chunks_mut(len/factors[axis]).enumerate(){
		let out_x = out_x + i as isize;
		let out_x = if out_x < 0 || out_x >= output_spatial[axis] as isize { // handle image boundaries of this axis by zeroing/skipping them

			// unlike unpack_hiris_patch(), we need to zero out mismatches between patches and the output, else gradients from a previous patch may remain.
			let new_len = new_matrix.len();
			for m in new_matrix.chunks_mut(new_len/factor_strides[axis]) {
				for i in 0..n_channels {
					m[patch_index*n_channels + i] = 0.0;
				}
			}
			continue;
		} else {
			out_x as usize
		};

		let new_output_grad = &output_grad[out_x*out_stride..(out_x+1)*out_stride];
		if axis < output_spatial.len() - 1 {
			let new_patch_index_rem = patch_index_rem - patch_x * patch_strides[axis];
			let new_axis = axis + 1;
			pack_hires_patch(new_output_grad, output_spatial, n_channels, patch_index, new_patch_index_rem, patch_strides, new_matrix, factors, factor_strides, new_axis);
		} else {
			debug_assert_eq!(new_output_grad.len(), n_channels);
			debug_assert_eq!(0, new_matrix.len() % n_channels);
			debug_assert_eq!(patch_strides[axis], 1);

			let new_output_grad = &new_output_grad[0..n_channels];
			
			let m = &mut new_matrix[patch_index*n_channels..][..n_channels];
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
	let mut out = vec![0.0; cols*rows];
	out[0] = 1.0;

	for axis in 0..factors.len(){
		fill_next_axis(factors, axis, rows, &mut out);
	}

	out
}

fn fill_next_axis(factors: &[usize], axis: usize, col_stride: usize, matrix: &mut [f32]){
	
	let f = factors[axis];
	let step = 1.0/f as f32;
	let start = 1.0-((f+1)%2)as f32*0.5*step;
	
	// size of already filled patch;
	let cols = 2usize.pow(axis as u32);
	let rows = factors[0..axis].iter().product();
	let col_offset = col_stride*cols;
	
	for i in (0..f).rev() { // must do blocks in reverse order so that data in top left of matrix isnt overwritten too early
		let mult1 = start-step*i as f32;
		let mult2 = 1.0 - mult1;
		
		let row_offset = rows*i;

		for col in 0..cols {
			for row in 0..rows{
				let ind = row+col*col_stride;
				let val = matrix[ind];
				matrix[ind + row_offset] = mult1*val;
				matrix[ind + row_offset + col_offset] = mult2*val;

			}
		}

	}
}


#[test]
fn linterp_backprop1(){
	_linterp_backprop1().unwrap();
}

fn _linterp_backprop1() -> Result<()> {
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use new::ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();
	
	let node1 = g.new_node(shape![2, 4, 3, 5], "input", tag![])?;
	let node2 = g.new_node(shape![2, 8, 6, 5], "conv", tag![])?;
	let node3 = g.new_node(shape![2, 8, 6, 5], "target", tag![])?;
		
	let _o1 = g.new_op(Linterp::new(&node1, &node2, &[1, 2, 2, 1]), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.01;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}

#[test]
fn linterp_backprop(){
	_linterp_backprop().unwrap();
}

fn _linterp_backprop() -> Result<()>{
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use new::ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();
	
	let node1 = g.new_node(shape![3, 5, 1, 7, 13], "input", tag![])?;
	let node2 = g.new_node(shape![3, 10, 1, 14, 13], "conv", tag![])?;
	let node3 = g.new_node(shape![3, 10, 1, 14, 13], "target", tag![])?;
		
	let _o1 = g.new_op(Linterp::new(&node1, &node2, &[1, 2, 1, 2, 1]), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.01;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;
	
	Ok(())
}

#[test]
fn test_linterp(){
	_test_linterp().unwrap();
}

fn _test_linterp() -> Result<()>{
	{
		let mut g = GraphDef::new();
		let input = g.new_node(shape![1, 2, 1, 2, 1], "input", tag![])?;
		let output = g.new_node(shape![1, 4, 1, 4, 1], "output", tag![])?;
		let _o1 = g.new_op(Linterp::new(&input, &output, &[1, 2, 1, 2, 1]), tag![])?;

		let mut data_in = ArrayD::zeros(IxDyn(&[1, 2, 1, 2, 1]));
		{
			let data_in_slice = data_in.as_slice_mut().unwrap();
			data_in_slice[1] = 1.0;
			data_in_slice[2] = 1.0;
		}


		let mut subgraph = g.subgraph(&[input.value_id()], &[output.value_id()])?;

		let storage = subgraph.execute(vec![data_in])?;

		let out = storage.get_mut(&output.value_id())?;
		let out_slice = out.as_slice().unwrap();

		let expected = vec![
			0.0,  0.25,  0.75,  1.0,
			0.25, 0.375, 0.625, 0.75,
			0.75, 0.625, 0.375, 0.25,
			1.0,  0.75,  0.25,  0.0,
		];

		let diff = out_slice.iter().zip(expected.iter()).fold(0.0, |acc, (&o, &e)| acc + (o-e)*(o-e));
		assert!(diff < 1e-6, "{:?} {:?}", out_slice, expected);
	}



	{
		let mut g = GraphDef::new();
		let input = g.new_node(shape![1, 3, 1, 3, 1], "input", tag![])?;
		let output = g.new_node(shape![1, 9, 1, 9, 1], "output", tag![])?;
		let _o1 = g.new_op(Linterp::new(&input, &output, &[1, 3, 1, 3, 1]), tag![])?;

		let mut data_in = ArrayD::zeros(IxDyn(&[1, 3, 1, 3, 1]));
		{
			let data_in_slice = data_in.as_slice_mut().unwrap();
			data_in_slice[4] = 1.0;
		}

		let mut subgraph = g.subgraph(&[input.value_id()], &[output.value_id()])?;

		let storage = subgraph.execute(vec![data_in])?;

		let out = storage.get_mut(&output.value_id())?;
		let out_slice = out.as_slice().unwrap();

		let expected = vec![
			0.0, 0.0,   0.0,   0.0,   0.0,   0.0,   0.0, 0.0, 0.0,
			0.0, 0.0,   0.0,   0.0,   0.0,   0.0,   0.0, 0.0, 0.0,
			0.0, 0.0, 1./9., 2./9., 3./9., 2./9., 1./9., 0.0, 0.0, 
			0.0, 0.0, 2./9., 4./9., 6./9., 4./9., 2./9., 0.0, 0.0, 
			0.0, 0.0, 3./9., 6./9.,	  1.0, 6./9., 3./9., 0.0, 0.0, 
			0.0, 0.0, 2./9., 4./9., 6./9., 4./9., 2./9., 0.0, 0.0, 
			0.0, 0.0, 1./9., 2./9., 3./9., 2./9., 1./9., 0.0, 0.0, 
			0.0, 0.0,   0.0,   0.0,   0.0,   0.0,   0.0, 0.0, 0.0,
			0.0, 0.0,   0.0,   0.0,   0.0,   0.0,   0.0, 0.0, 0.0,
		];

		let diff = out_slice.iter().zip(expected.iter()).fold(0.0, |acc, (&o, &e)| acc + (o-e)*(o-e));
		assert!(diff < 1e-6, "/n{:?} /n{:?}/n", out_slice, expected);
		
	}

	Ok(())
}


#[test]
fn test_linear_interp_matrix(){
	_test_linear_interp_matrix();
}

fn _test_linear_interp_matrix(){


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

		for row in 0..rows{
			let mut sum = 0.0;
			for col in 0..cols {
				sum += matrix[row + col*rows]
			}
			
			assert!( (1.0 - sum).abs() < 1e-6, "sum: {} matrix:\n{:?}", sum, matrix);
		}

	}
	
	{
		let matrix = upscale_matrix(&vec![3]);
		let expected = vec![ // col major
			1.0, 2./3., 1./3.,
			0.0, 1./3., 2./3.,
		];

		let diff = matrix.iter().zip(expected.iter()).fold(0.0, |acc, (&m, &e)| acc+(m-e)*(m-e));
		assert!(diff < 1e-6, "{:?} {:?}", matrix, expected);

	}

	{
		let matrix = upscale_matrix(&vec![2]);
		let expected = vec![ // col major
			0.75, 0.25,
			0.25, 0.75,
		];

		let diff = matrix.iter().zip(expected.iter()).fold(0.0, |acc, (&m, &e)| acc+(m-e)*(m-e));
		assert!(diff < 1e-6, "{:?} {:?}", matrix, expected);

	}

}


#[test]
fn test_pack_lowres_recurse(){
	_test_pack_lowres_recurse();
}

fn _test_pack_lowres_recurse(){

	let input_spatial = vec![2, 3];
	let n_channels = 5usize;

	let input_size = input_spatial.iter().fold(n_channels, |acc, v| acc*v);
	let mut input = (0..input_size).map(|x| x as f32).collect::<Vec<_>>();

	let n_patches = input_spatial.iter().fold(1, |acc, v| acc * (v+1));
	let axis = 0;

	let rows = 2usize.pow(input_spatial.len()as u32);
	let mut matrix = vec![0.0; rows*n_channels*n_patches];
	let strides = patch_strides(&input_spatial);

	for index in 0..n_patches{
		pack_lowres_patch(&mut input, &input_spatial, n_channels, index, index, &strides, &mut matrix, axis);
	}

	let expected = vec![
		0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 25, 26, 27, 28, 29,
		0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 15, 16, 17, 18, 19, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 25, 26, 27, 28, 29, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 25, 26, 27, 28, 29
		].iter().map(|x: &usize| *x as f32).collect::<Vec<_>>();
	
	assert_eq!(expected, matrix);

}

#[test]
fn test_unpack_hires_matrix(){
	_test_unpack_hires_matrix();
}

fn _test_unpack_hires_matrix(){
	let input_spatial = vec![3, 2];
	let factors = vec![2, 2];
	let output_spatial = input_spatial.iter().zip(factors.iter()).map(|(x, f)| x*f).collect::<Vec<usize>>();
	let n_channels = 1usize;

	let output_size = output_spatial.iter().fold(n_channels, |acc, v| acc*v);
	let mut output = vec![0.0; output_size];
	
	

	let n_patches: usize = input_spatial.iter().fold(1, |acc, v| acc * (v+1));
	let axis = 0;

	let rows: usize = factors.iter().product();//2usize.pow(input_shape.len()as u32);
	let mut matrix = (0..rows*n_channels*n_patches).map(|x: usize| (x/n_patches + (x%n_patches)*rows) as f32).collect::<Vec<f32>>();
	let strides = patch_strides(&input_spatial);

	for index in 0..n_patches{
		unpack_hires_patch(&mut output, &output_spatial, n_channels, index, index, &strides, &mut matrix, &factors, axis);
	}

	let expected = vec![
		 3,  6,  7, 10,
		13, 16, 17, 20,
		15, 18, 19, 22,
		25, 28, 29, 32,
		27, 30, 31, 34,
		37, 40, 41, 44,
	].iter().map(|x: &usize| *x as f32).collect::<Vec<_>>();

	assert_eq!(expected, output);
}