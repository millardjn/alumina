use graph::*;
use std::cell::RefCell;
use ops::math;
use std::cmp::{min, max};
use shape::*;
use ops::Operation;
use std::sync::Arc;
use std::iter;
use matrixmultiply;
use odds;
use num_cpus;
use std::sync::Mutex;
use std::sync::mpsc::sync_channel;
use scoped_threadpool::Pool;

/// Threadpool for offloading lowering/packing operations
lazy_static! {
	static ref NUM_CPUS: usize = num_cpus::get();
	static ref THREAD_POOL: Mutex<Pool> = Mutex::new(Pool::new(*NUM_CPUS as u32));
}

#[derive(Clone)]
pub enum Padding {
	Full,
	Same,
	Valid,
	Padded(usize), // extra padding. 0 is equivalent to same
	PaddedDiff(Vec<usize>), // extra padding per dimension. 0 is equivalent to same
}

#[derive(Clone)] 	
pub struct Convolution {
 	name: String,
 	kernel_shape: Vec<usize>, // kernel shape
 	padding: Padding,
 	input_id: NodeID,
	output_id: NodeID,
	input_channels: usize,
	output_channels: usize,
	num_params: usize,
	init_func: Arc<Fn(&Convolution, &mut [f32])>,
	lowering_memory: usize, // when lowering the input for convolution try to use less memory than this, in bytes

	shuffled_kernel:  Vec<f32>, // scratch space backprop kernel shuffles
	shuffled_derivatives: Vec<f32>,
}
 	
/// Convolution    - PSISOD - Standard convolution operation, various padding options
/// parameters are a row major matrix Cin.W.H.Cout
impl Convolution {
	pub fn new(input_id: &NodeID, output_id: &NodeID, kernel_shape: &[usize], padding: Padding, name: &str, init_func: Arc<Fn(&Convolution, &mut [f32])>) -> Box<Convolution>{
		assert!(input_id.shape.rank() == output_id.shape.rank());
		let num_params = kernel_shape.iter().fold(input_id.shape.channels * output_id.shape.channels, |p, v| p*v);
		Box::new(Convolution{
			name: name.to_string(),
			kernel_shape: kernel_shape.to_vec(),
			padding: padding,
			input_id: input_id.clone(),
			output_id: output_id.clone(),
			input_channels: input_id.shape.channels,
			output_channels: output_id.shape.channels,
			num_params: num_params,
			init_func: init_func,
			lowering_memory: 1024*1024*1, // by default perform lowering on the cpu cache, not the whole image

			shuffled_kernel: vec![0.0; num_params],
			shuffled_derivatives: vec![0.0; num_params],			
		})
	}
	
	pub fn new_default(input_id: &NodeID, output_id: &NodeID, ks: &[usize]) -> Box<Convolution>{
		Convolution::new(input_id, output_id, ks, Padding::Same, "Convolution", Convolution::init_xavier())
	}
	
	pub fn init_xavier() -> Arc<Fn(&Convolution, &mut [f32])> {
		Convolution::init_msra(1.0)
	}
	
	pub fn init_msra(sd_multiplier: f32) -> Arc<Fn(&Convolution, &mut [f32])> {
		Arc::new(
			move |op: &Convolution, params: &mut [f32]| {
				let variance = 1.0/op.kernel_shape.iter().fold(op.input_channels,|p, v| p*v) as f32;
				math::random_vector::normal_fill(params, 0.0, sd_multiplier*variance.sqrt());
			}
		)
	}

}


impl Operation for Convolution {

	fn name(&self) -> &str{&self.name}
	
	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_id.ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Input node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_id.ind].name, self.name));
		
		let required_shape = {
			let dims = shapes[self.input_id.ind].spatial_dimensions.iter().map(|dim| match dim {
				&Dimension::Fixed(v) => v, 
				_ => unreachable!(),
			});
			
			NodeShape{
				channels: shapes[self.output_id.ind].channels,
				spatial_dimensions: match self.padding {
						Padding::Full => dims.zip(&self.kernel_shape).map(|(dim, k_dim)| Dimension::Fixed(dim + k_dim - 1)).collect(),
						Padding::Same => shapes[self.input_id.ind].spatial_dimensions.clone(),
						Padding::Valid => dims.zip(&self.kernel_shape).map(|(dim, k_dim)| Dimension::Fixed(dim - k_dim + 1)).collect(),
						Padding::Padded(ref size) => dims.map(|dim| Dimension::Fixed(dim +size)).collect(),
						Padding::PaddedDiff(ref vec) => dims.zip(vec).map(|(dim, vec_dim)| Dimension::Fixed(dim + vec_dim)).collect(),
					},
			}
		};

		shapes[self.output_id.ind] = required_shape.merge(&shapes[self.output_id.ind])
			.expect(&format!("Error: Operation '{}' error could not merge required output shape with existing shape for Node '{}'.", self.name, nodes[self.output_id.ind].name));
			//.expect(&format!("Error: Operation '{}' error could not merge required output shape with existing shape for Node '{}'. old shape: {:?}, new shape: {:?}", self.name, nodes[self.output_id.ind].name, shapes[self.output_id.ind], required_shape));
		
	}
	
	fn input_node_IDs(&self) -> Vec<NodeID>{vec![self.input_id.clone()]}
	
	fn output_node_IDs(&self) -> Vec<NodeID>{vec![self.output_id.clone()]}
	
	fn init_params(&mut self, params: &mut [f32]){
		assert!(self.num_params() == params.len());
		self.init_func.as_ref()(&self, params);
	}
	
	fn num_params(&self) -> usize {self.num_params}
	
	fn forward(&mut self, data: &mut [RefCell<NodeData>], params: &[f32]){
		let input = &*{data[self.input_id.ind].borrow_mut()};
		let output = &mut *{data[self.output_id.ind].borrow_mut()};

		let mut pool = THREAD_POOL.lock().expect("Could not lock conv threadpool");
		pool.scoped(|scope|{
			


			let &mut NodeData{shape: ref output_shape, values: ref mut output_values, ..} = output;

			let in_size = input.shape.flat_size_single();
			let out_size = output.shape.flat_size_single();
			
			//let in_spaxels = in_size/input.shape.channel_dimension;
			let out_spaxels = out_size/output.shape.channels;
			
			// These checks shouldnt be necessary unless code in the graph doesnt correctly resolve compatible shapes.
			// should check shape compatability under padding etc assert!(input.shape.n == output.shape.n);
			let n = input.shape.n;
			debug_assert_eq!(n, output.shape.n);
			debug_assert_eq!(input.shape.channels, self.input_channels);
			debug_assert_eq!(output.shape.channels, self.output_channels);
			debug_assert_eq!(input.shape.spatial_dimensions.len(), output.shape.spatial_dimensions.len());
			debug_assert_eq!(input.shape.spatial_dimensions.len(), self.kernel_shape.len());
			debug_assert_eq!(params.len()/self.output_channels, self.kernel_shape.iter().fold(self.input_channels, |p, v| p*v));

			let patch_size = params.len()/self.output_channels;

			let max_spaxels = min(max(1, self.lowering_memory/(patch_size*4)), out_spaxels*n); // number of spaxels to combine in one sgemm
			let n_batches = (out_spaxels*n + max_spaxels -1)/max_spaxels;

			// let mut patches = Vec::with_capacity(patch_size * max_spaxels); 
			// unsafe{
			// 	patches.set_len(patch_size * max_spaxels);
			// }

			let kernel_strides = stride_vec(self.input_channels, &self.kernel_shape);
			let input_strides = stride_vec(self.input_channels, &input.shape.spatial_dimensions);
			let output_strides = stride_vec(self.output_channels, &output.shape.spatial_dimensions);

			let kernel_shape = &self.kernel_shape;
			let input_channels = self.input_channels;
			let output_channels = self.output_channels;

			let (tx, rx) = sync_channel(1);
			let (tx2, rx2) = sync_channel(1);
			let mut spare_patches = Vec::with_capacity(patch_size * max_spaxels); 
			unsafe{spare_patches.set_len(patch_size * max_spaxels);}
			let mut spare_patches_opt = Some(spare_patches);
			scope.execute(move|| {
				let mut patches = Vec::with_capacity(patch_size * max_spaxels); 
				unsafe{patches.set_len(patch_size * max_spaxels);}
				let mut patches_opt = Some(patches);

				for batch in 0..n_batches {
					let spaxel_ind = batch*max_spaxels;
					
					let batch_spaxels = min(out_spaxels*n - spaxel_ind, max_spaxels);
					{
						let patches = &mut patches_opt.as_mut().expect("conv patches missing")[..batch_spaxels*patch_size];
						for (i, patch) in patches.chunks_mut(patch_size).enumerate() {
							debug_assert_eq!(patch_size, patch.len());
							let n_ind = (spaxel_ind+i)/out_spaxels;

							let in_n = &input.values[n_ind*in_size..][..in_size];	

							let output_ind = (spaxel_ind+i)%out_spaxels*output_channels;
							unsafe_pack_patch_outer(patch, in_n, input_channels, output_ind, kernel_shape, &input.shape.spatial_dimensions, &output_shape.spatial_dimensions, &kernel_strides, &input_strides, &output_strides);
							//pack_patch_recurse(patch, in_n, &kernel_shape, input_channels, &input.shape.spatial_dimensions, &output_shape.spatial_dimensions, kernel_shape.len()-1, output_ind, out_size);
						}
					}
					tx.send(Some((patches_opt.take().expect("conv patches missing"), spaxel_ind, batch_spaxels))).expect("conv patch send err");
					patches_opt = Some(rx2.recv().expect("conv patches missing"));
				}
				tx.send(None).expect("conv patch send err");
			});

			while let Some((patches, spaxel_ind, batch_spaxels)) = rx.recv().expect("Convolution channel receive error") {
				tx2.send(spare_patches_opt.take().expect("conv patches missing")).expect("conv patch send err");
				let out_b = &mut output_values[spaxel_ind*self.output_channels..][..batch_spaxels*self.output_channels];

				let m = self.output_channels;
				let n = batch_spaxels;
				let k = patch_size;

				unsafe{
					matrixmultiply::sgemm(m, k, n,
						1.0,
						params.as_ptr(), k as isize, 1, // A is params, row major
						patches.as_ptr(), 1, k as isize, // B, input patches column major
						1.0,
						out_b.as_mut_ptr(), 1, m as isize); // C output values column major
				}
				spare_patches_opt = Some(patches);
			}
		});
	}
	
	fn backward (&mut self, data: &mut [RefCell<NodeData>], params: &[f32], param_deriv: &mut [f32], _error: &mut f32){
		let input = &mut *{data[self.input_id.ind].borrow_mut()};
		let output = &*{data[self.output_id.ind].borrow_mut()};
	
		let mut pool = THREAD_POOL.lock().expect("Could not lock conv threadpool");
		pool.scoped(|scope|{

			let &mut NodeData{shape: ref input_shape, values: ref input_values, derivatives: ref mut input_derivatives} = input;

			let in_size = input.shape.flat_size_single();
			let out_size = output.shape.flat_size_single();
			
			let in_spaxels = in_size/input.shape.channels;
			//let out_spaxels = out_size/output.shape.channel_dimension;
			
			// These checks shouldnt be necessary unless code in the graph doesnt correctly resolve compatible shapes.
			// should check shape compatability under padding etc
			let n = input.shape.n;
			debug_assert_eq!(n, output.shape.n);
			debug_assert_eq!(input.shape.channels, self.input_channels);
			debug_assert_eq!(output.shape.channels, self.output_channels);
			debug_assert_eq!(input.shape.spatial_dimensions.len(), output.shape.spatial_dimensions.len());
			debug_assert_eq!(input.shape.spatial_dimensions.len(), self.kernel_shape.len());
			debug_assert_eq!(params.len()/self.input_channels, self.kernel_shape.iter().fold(self.output_channels, |p, v| p*v));
			debug_assert_eq!(params.len(), param_deriv.len());
			
			let patch_size = params.len()/self.input_channels;

			let max_spaxels = min(max(1, self.lowering_memory/(patch_size*4)), in_spaxels*n); // number of spaxels to combine in one sgemm
			let n_batches = (in_spaxels*n + max_spaxels -1)/max_spaxels;

			// let mut patches = Vec::with_capacity(patch_size * max_spaxels); 
			// unsafe{
			// 	patches.set_len(patch_size * max_spaxels);
			// }

			col_flip_transpose_kernel_overwrite(&params, self.input_channels, self.output_channels, &mut self.shuffled_kernel);
			col_flip_transpose_kernel_overwrite(&param_deriv, self.input_channels, self.output_channels, &mut self.shuffled_derivatives);

			let kernel_strides = stride_vec(self.output_channels, &self.kernel_shape);
			let input_strides = stride_vec(self.input_channels, &input.shape.spatial_dimensions);
			let output_strides = stride_vec(self.output_channels, &output.shape.spatial_dimensions);
			
			let kernel_shape = &self.kernel_shape;
			let input_channels = self.input_channels;
			let output_channels = self.output_channels;

			let (tx, rx) = sync_channel(1);
			let (tx2, rx2) = sync_channel(1);
			let mut spare_patches = Vec::with_capacity(patch_size * max_spaxels); 
			unsafe{spare_patches.set_len(patch_size * max_spaxels);}
			let mut spare_patches_opt = Some(spare_patches);
			scope.execute(move|| {
				let mut patches = Vec::with_capacity(patch_size * max_spaxels); 
				unsafe{patches.set_len(patch_size * max_spaxels);}
				let mut patches_opt = Some(patches);
				for batch in 0..n_batches {

					let spaxel_ind = batch*max_spaxels;
					
					let batch_spaxels = min(in_spaxels*n - spaxel_ind, max_spaxels);

					//let patches = &mut patches[..batch_spaxels*patch_size];
					{
						let patches = &mut patches_opt.as_mut().expect("conv patches missing")[..batch_spaxels*patch_size];
						for (i, patch) in patches.chunks_mut(patch_size).enumerate() {
							debug_assert_eq!(patch_size, patch.len());
							let n_ind = (spaxel_ind+i)/in_spaxels;

							let outd_n = &output.derivatives[n_ind*out_size..][..out_size];

							let input_ind = (spaxel_ind+i)%in_spaxels*input_channels;
							unsafe_pack_patch_outer(patch, outd_n, output_channels, input_ind, &kernel_shape, &output.shape.spatial_dimensions, &input_shape.spatial_dimensions, &kernel_strides, &output_strides, &input_strides);
							//pack_patch_recurse(patch, outd_n, &kernel_shape, output_channels, &output.shape.spatial_dimensions, &input_shape.spatial_dimensions, kernel_shape.len()-1, input_ind, in_size);
						}
					}
					tx.send(Some((patches_opt.take().expect("conv patches missing"), spaxel_ind, batch_spaxels))).expect("conv patch send err");
					patches_opt = Some(rx2.recv().expect("conv patches missing"));
				}
				tx.send(None).expect("conv patch send err");
			});
			
			while let Some((patches, spaxel_ind, batch_spaxels)) = rx.recv().expect("Convolution channel receive error") {
				tx2.send(spare_patches_opt.take().expect("conv patches missing")).expect("conv patch send err");

				let ind_b = &mut input_derivatives[spaxel_ind*self.input_channels..][..batch_spaxels*self.input_channels];
				let in_b = &input_values[spaxel_ind*self.input_channels..][..batch_spaxels*self.input_channels];

				let m1 = self.input_channels;
				let n1 = batch_spaxels;
				let k1 = patch_size;	
				
				let m2 = self.input_channels;
				let n2 = patch_size;
				let k2 = batch_spaxels;	

				unsafe{
					// input derivatives
					matrixmultiply::sgemm(m1, k1, n1,
						1.0,
						//self.shuffled_kernel.as_ptr(), k1 as isize, 1, // A is params, row major
						self.shuffled_kernel.as_ptr(), 1, m1 as isize, // A is params, col major
						patches.as_ptr(), 1, k1 as isize, // B, input values, column major
						1.0,
						ind_b.as_mut_ptr(), 1, m1 as isize // C output values, column major
					); 
				
					// // parameter derivatives
					matrixmultiply::sgemm(m2, k2, n2,
						1.0,
						in_b.as_ptr(), 1, m2 as isize, // A is input image, col major
						patches.as_ptr(), n2 as isize, 1, // B, derivative patches, row major
						1.0,
						//self.shuffled_derivatives.as_mut_ptr(), n2 as isize, 1 // C shuffled parameter derivatives, row major
						self.shuffled_derivatives.as_mut_ptr(), 1, m2 as isize // C shuffled parameter derivatives, col major
					);
				}
				spare_patches_opt = Some(patches);
			}
			//flip_transpose_kernel_overwrite(&self.shuffled_derivatives, self.output_channels, self.input_channels, param_deriv);
			rev_col_flip_transpose_kernel_overwrite(&self.shuffled_derivatives, self.input_channels, self.output_channels, param_deriv);
		});
	}		
}




/// A recursive N-dimensional im2col like function.
/// Packs data from a rectangular region of 'input' into 'patch'.
/// Inner recursions deal with lower dimensional slices of the 'input' and the 'patch' recursing until it is reduced to a 1D memcpy.
///
/// # Arguments
/// * `patch` - a rectangular region of input, of shape Cin.ks[0]...ks[axis]; Each recursion removes the outermost spatial dimension.
/// * `input` - input 'image' of shape Cin.W.H
/// * `patch_shape` - Spatial dimensions of the patch, in spaxels
/// * `n_channels` - The number of channels in the 'input' and 'patch' i.e. Cin
/// * `input_shape` - Spatial dimensions of the input, in spaxels
/// * `output_shape` - Spatial dimensions of the output, in spaxels
/// * `axis` - current axis being iterated over. This should be ks.len() - 1 for root call. Reduces by 1 each recursion.
/// * `output_ind` - Index of output spaxel on which the patch is centred. Note: index is the slice index not spaxel index (factor of Cin difference)
/// * `old_out_stride` - Slice stride of output for the layer bove the current iteration. used for interpreting `output_ind`. Root call should be output.len()
#[allow(unused)]
fn pack_patch_recurse(patch: &mut [f32], input: &[f32], patch_shape:&[usize], n_channels: usize,
	input_shape: &[usize], output_shape: &[usize], axis: usize, output_ind: usize, old_out_stride: usize){
	
	
	// stride in array index, not spaxel index
	let out_stride = old_out_stride/output_shape[axis];
	let in_stride = input.len()/input_shape[axis];
	let ks_stride = patch.len()/patch_shape[axis];
	
	// coordinates of the centre spaxel of the kernel, for the current axis, for both the output and the kernel itself
	let ox = (output_ind % old_out_stride)/out_stride;
	let ix = ox as isize + (input_shape[axis] as isize - output_shape[axis] as isize)/2;
	
	// valid range of the kernels in the current axis
	let (start, end) = kernel_range(ix, input_shape[axis], patch_shape[axis]);

	for i in 0..start*ks_stride{
		patch[i] = 0.0;// fill zeros
	}
				
	if axis > 0 {
		
		for i in start..end{
			let ix = (ix + i as isize - (patch_shape[axis]/2) as isize) as  usize; // shadow ix is the coordinate for the current iteration, rather than the centre of the kernel.

			let new_input = &input[in_stride*ix..in_stride*(ix+1)];
			let new_patch = &mut patch[i*ks_stride..(i+1)*ks_stride];
			let new_axis = axis-1;

			pack_patch_recurse(new_patch, new_input, patch_shape, n_channels, input_shape, output_shape, new_axis, output_ind, out_stride);
		}

	} else {	

		let offset = ((ix-patch_shape[axis] as isize/2)*n_channels as isize + (start*n_channels) as isize) as usize;
		let len = (end - start)*n_channels;
		let input_crop = &input[offset..][..len];
		let mut patch_crop = &mut patch[(start*n_channels)..][..len];
		
		patch_crop.copy_from_slice(input_crop);		
	}

	for i in (end*ks_stride)..(patch_shape[axis]*ks_stride){
		patch[i] = 0.0;// fill zero
	}		
}


/// returns a vector with the array stride of each dimension. output[0] == channel.
fn stride_vec(channels: usize, shape: &[usize]) -> Vec<usize>{
	iter::once(&channels).chain(shape.iter()).scan(1, |state, &i| {*state *= i; Some(*state)}).collect::<Vec<usize>>()
}

//#[inline(never)]
fn unsafe_pack_patch_outer(patch: &mut [f32], input: &[f32], channels: usize, output_ind: usize,
	kernel_shape: &[usize], input_shape: &[usize], output_shape: &[usize],
	kernel_strides: &[usize], input_strides: &[usize], output_strides: &[usize]){
	let axis = kernel_shape.len() - 1;
	let o_stride = output_strides[axis];

	let ox = output_ind/o_stride;
	let ix = ox as isize + (input_shape[axis] as isize - output_shape[axis] as isize)/2;
	let (start, end) = kernel_range(ix, input_shape[axis], kernel_shape[axis]);

	unsafe {unsafe_pack_patch_recurse(patch, input, channels, axis, output_ind, ox, ix, start, end,
	kernel_shape, input_shape, output_shape, kernel_strides, input_strides, output_strides)};
}

//#[inline(always)]
unsafe fn unsafe_pack_patch_recurse(patch: &mut [f32], input: &[f32], channels: usize, axis: usize, output_ind: usize,
	ox: usize, ix: isize,
	start: usize, end: usize, // valid range of the kernels in the current axis
	kernel_shape: &[usize], input_shape: &[usize], output_shape: &[usize],
	kernel_strides: &[usize], input_strides: &[usize], output_strides: &[usize]){
	

	let i_stride = *odds::get_unchecked(input_strides, axis);//input_strides[axis];
	let o_stride = *odds::get_unchecked(output_strides, axis);//output_strides[axis];
	let k_stride = *odds::get_unchecked(kernel_strides, axis);//kernel_strides[axis];
	// coordinates of the centre spaxel of the kernel, for the current axis, for both the output and the kernel itself

	
	
	
	for i in 0..start*k_stride{
		*odds::get_unchecked_mut(patch, i) = 0.0; // fill zero
	}
				
	if axis > 0 {
		for i in start..end{
			let temp_ix = (ix + i as isize - (*odds::get_unchecked(kernel_shape, axis)/2) as isize) as  usize; // temp_ix is the coordinate for the current iteration, rather than the centre of the kernel.
			debug_assert!(ix + i as isize - (kernel_shape[axis]/2) as isize >= 0);
			
			//let new_input = &input[i_stride*temp_ix..i_stride*(temp_ix+1)];
			let new_input = odds::slice_unchecked(input, i_stride*temp_ix, i_stride*(temp_ix+1));

			//let new_patch = &mut patch[i*k_stride..(i+1)*k_stride];
			let new_patch = odds::slice_unchecked_mut(patch, i*k_stride, (i+1)*k_stride);

			let new_axis = axis-1;
			let new_output_ind = output_ind - ox*o_stride;
			let new_ox = new_output_ind/ *odds::get_unchecked(output_strides, new_axis);//output_strides[new_axis];
			let new_ix = new_ox as isize + (*odds::get_unchecked(input_shape, new_axis) as isize - *odds::get_unchecked(input_shape, new_axis) as isize)/2;
			let (new_start, new_end) = kernel_range(new_ix, *odds::get_unchecked(input_shape, new_axis), *odds::get_unchecked(kernel_shape, new_axis));// input_shape[new_axis]    kernel_shape[new_axis]);

			unsafe_pack_patch_recurse(new_patch, new_input, channels, axis - 1, new_output_ind, new_ox, new_ix,
			new_start, new_end, kernel_shape, input_shape, output_shape, kernel_strides, input_strides, output_strides)
		}

	} else {	
		let offset = ((ix-*odds::get_unchecked(kernel_shape, axis) as isize/2)*channels as isize + (start*channels) as isize) as usize;
		let len = (end - start)*channels;

		//let input_crop = &input[offset..][..len];
		let input_crop = odds::slice_unchecked(input, offset, offset + len);
		
		//let mut patch_crop = &mut patch[(start*channels)..][..len];
		let mut patch_crop = odds::slice_unchecked_mut(patch, start*channels, start*channels + len);
		
		//patch_crop.copy_from_slice(input_crop);
		for i in 0..len{
			*odds::get_unchecked_mut(patch_crop, i) = *odds::get_unchecked(input_crop, i);
		}
	}

	for i in (end*k_stride)..(*odds::get_unchecked(kernel_shape, axis)*k_stride){
		*odds::get_unchecked_mut(patch, i) = 0.0; // fill zero
	}
}

/// returns the [start, end) range indicies of the kernel which overlap with the 'image'
/// returns [start, end) of the kernel range which overlaps with the image, given the width of the image and the position of the center of the kernel
/// only odd kernels are valid.
fn kernel_range(center: isize, width: usize, kernel_width: usize) -> (usize, usize){
 	debug_assert!(kernel_width % 2 == 1);
	(
		min(kernel_width as isize, max(0, kernel_width as isize/2 - center)) as usize,
		max(0, kernel_width as isize - max(0, center + kernel_width as isize/2 + 1- width as isize)) as usize
	)
}


#[cfg(test)]
/// take a kernel ordered as Cinner.W.H.Couter and convert to a same shaped kernel which has spatial dimension (W.H) orders reversed.
/// Also known as ROT180() in some libraries
fn flip_kernel_overwrite(kernel: &[f32], in_channels: usize, out_channels: usize, out: &mut [f32]){
	debug_assert_eq!(kernel.len(), out.len());
	debug_assert_eq!(kernel.len()%(in_channels * out_channels), 0);
	let num_blocks = kernel.len()/(in_channels * out_channels);
	let in_stride = kernel.len()/out_channels;
	
	for b in 0..num_blocks{
		let x_in = b * in_channels;
		let x_out = (num_blocks - b - 1) * in_channels;
		
		for out_ind in 0.. out_channels{
			let x_in = x_in + out_ind*in_stride;
			let x_out = x_out + out_ind*in_stride;		
			for in_ind in 0..in_channels{
				let x_in = x_in + in_ind;
				let x_out = x_out + in_ind;

				out[x_out] = kernel[x_in];
			}
		}
		
	}	
}

#[cfg(test)]
/// take a kernel ordered as Cinner.W.H.Couter and shuffle data to Couter.W.H.Cinner, transposing the channels of each spaxel.
fn transpose_kernel_overwrite(kernel: &[f32], in_channels: usize, out_channels: usize, out: &mut [f32]){
	debug_assert_eq!(kernel.len(), out.len());
	debug_assert_eq!(kernel.len()%(in_channels * out_channels), 0);
	let num_blocks = kernel.len()/(in_channels * out_channels);
	let in_stride = kernel.len()/out_channels;
	let out_stride = kernel.len()/in_channels;
	
	for b in 0..num_blocks{
		let x_in = b * in_channels;
		let x_out = b * out_channels;
		
		for in_ind in 0..in_channels{
			let x_in = x_in + in_ind;
			let x_out = x_out + in_ind*out_stride;
			
			for out_ind in 0.. out_channels{
				let x_in = x_in + out_ind*in_stride;
				let x_out = x_out + out_ind;

				out[x_out] = kernel[x_in];
			}
		}
		
	}
}

/// Combine kernel Flip and Transpose functions
///  takes row major, outputs row major
#[cfg(test)]
fn flip_transpose_kernel_overwrite(kernel: &[f32], in_channels: usize, out_channels: usize, out: &mut [f32]){
	debug_assert_eq!(kernel.len(), out.len());
	debug_assert_eq!(kernel.len()%(in_channels * out_channels), 0);
	let num_blocks = kernel.len()/(in_channels * out_channels);
	let in_stride = kernel.len()/out_channels;
	let out_stride = kernel.len()/in_channels;
	
	for b in 0..num_blocks{
		let x_in = b * in_channels;
		let x_out = (num_blocks - b - 1) * out_channels;
		
		for in_ind in 0..in_channels{
			let x_in = x_in + in_ind;
			let x_out = x_out + in_ind*out_stride;
			for out_ind in 0.. out_channels{
				let x_in = x_in + out_ind*in_stride;
				let x_out = x_out + out_ind;
				//out[x_out] = kernel[x_in];
				unsafe{
					*out.get_unchecked_mut(x_out) = *kernel.get_unchecked(x_in);
				}
			}
		}
		
	}
}

/// Combine kernel Flip and Transpose functions

/// takes row major, outputs col major
#[inline(never)]
fn col_flip_transpose_kernel_overwrite(kernel: &[f32], in_channels: usize, out_channels: usize, out: &mut [f32]){
	debug_assert_eq!(kernel.len(), out.len());
	debug_assert_eq!(kernel.len()%(in_channels * out_channels), 0);
	let num_blocks = kernel.len()/(in_channels * out_channels);
	let in_stride = kernel.len()/out_channels;
	//let out_stride = kernel.len()/in_channels;

	for b in 0..num_blocks{
		let kernel = &kernel[b * in_channels..];
		let out = &mut out[(num_blocks - b - 1) * (in_channels * out_channels)..];
		
		for out_ind in 0.. out_channels{
			let kernel = &kernel[out_ind*in_stride..][..in_channels];
			let out = &mut out[out_ind*in_channels..][..in_channels];

			for i in 0..in_channels{
				out[i] = kernel[i];
			}
		}
	}
}

#[inline(never)]
fn rev_col_flip_transpose_kernel_overwrite(kernel: &[f32], in_channels: usize, out_channels: usize, out: &mut [f32]){
	debug_assert_eq!(kernel.len(), out.len());
	debug_assert_eq!(kernel.len()%(in_channels * out_channels), 0);
	let num_blocks = kernel.len()/(in_channels * out_channels);
	let in_stride = kernel.len()/out_channels;
	//let out_stride = kernel.len()/in_channels;
	
	for b in 0..num_blocks{
		let out = &mut out[b * in_channels..];
		let kernel = &kernel[(num_blocks - b - 1) * (in_channels * out_channels)..];
		
		for out_ind in 0.. out_channels{
			let out = &mut out[out_ind*in_stride..][..in_channels];
			let kernel = &kernel[out_ind*in_channels..][..in_channels];

			for i in 0..in_channels{
				out[i] = kernel[i];
			}
		}
	}

}

#[cfg(test)]
mod test {
	use super::*; 	
	use ops::loss::MseLoss;
	
	#[test]
	fn test_conv_backprop(){
		for _ in 1..20{
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_sized(5, &[9, 11], "nodein"));
			let n2 = graph.add_output_node(Node::new_sized(7, &[9, 11], "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_sized(7, &[9, 11], "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				Convolution::new_default(&n1, &n2, &[3, 5]),
				MseLoss::new_default(&n2, &n3),
			];
			graph.add_operations(ops);
			graph.init_params();
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-1);
		}
	}	
	
	#[test]
	fn test_kernel_shuffles(){
		
		let num_blocks = 3;
		let in_channels = 2;
		let out_channels = 5;

		let kernel = (0..num_blocks*in_channels*out_channels).map(|x| x as f32).collect::<Vec<_>>();

		let mut flip = vec![0.0; kernel.len()];
		super::flip_kernel_overwrite(&kernel, in_channels, out_channels, &mut flip);	
		assert_eq!(flip, vec![
			 4.,  5.,  2.,  3.,  0.,  1.,
			10., 11.,  8.,  9.,  6.,  7.,
			16., 17., 14., 15., 12., 13.,
			22., 23., 20., 21., 18., 19.,
			28., 29., 26., 27., 24., 25.,
			]);
		
		let mut trans = vec![0.0; kernel.len()];
		super::transpose_kernel_overwrite(&kernel, in_channels, out_channels, &mut trans);
		assert_eq!(trans, vec![
			0.,  6., 12., 18., 24.,
			2.,  8., 14., 20., 26.,	
			4., 10., 16., 22., 28.,
			1.,  7., 13., 19., 25., 
			3.,  9., 15., 21., 27.,
			5., 11., 17., 23., 29.,
			]);

		let mut flip_trans = vec![0.0; kernel.len()];
		super::flip_transpose_kernel_overwrite(&kernel, in_channels, out_channels, &mut flip_trans);
		
		let mut flip_trans2 = vec![0.0; kernel.len()];
		super::flip_kernel_overwrite(&trans, out_channels, in_channels, &mut flip_trans2);
		
		let mut flip_trans3 = vec![0.0; kernel.len()];
		super::transpose_kernel_overwrite(&flip, in_channels, out_channels, &mut flip_trans3);		
		
		assert_eq!(flip_trans, flip_trans2);
		assert_eq!(flip_trans, flip_trans3);
	}	


	#[test]
	fn test_kernel_range(){
		assert!((0, 1) == super::kernel_range(0, 1, 1));
		assert!((1, 2) == super::kernel_range(0, 1, 3));
		assert!((3, 4) == super::kernel_range(0, 1, 7));

		assert!((3, 3) == super::kernel_range(-3, 7, 3));
		assert!((3, 3) == super::kernel_range(-2, 7, 3));
		assert!((2, 3) == super::kernel_range(-1, 7, 3));		
		assert!((1, 3) == super::kernel_range(0, 7, 3));
		assert!((0, 3) == super::kernel_range(1, 7, 3));
		assert!((0, 3) == super::kernel_range(2, 7, 3));
		
		assert!((0, 3) == super::kernel_range(5, 7, 3));
		assert!((0, 2) == super::kernel_range(6, 7, 3));
		assert!((0, 1) == super::kernel_range(7, 7, 3));
		assert!((0, 0) == super::kernel_range(8, 7, 3));
		assert!((0, 0) == super::kernel_range(9, 7, 3));
		

	}
	

	
	#[test]
	fn test_pack(){
		
		

		let ks = vec![3, 5];
		let input_shape = vec![5, 7];
		let output_shape = vec![7, 9];
		let in_channels = 3;
		let out_channels = 5;
		
		let input_spaxel_count = input_shape.iter().fold(1, |p, v| p*v);
		let output_spaxel_count = output_shape.iter().fold(1, |p, v| p*v);
		
		let input_size = input_spaxel_count*in_channels;
		let output_size = output_spaxel_count*out_channels;
		let patch_size = ks.iter().fold(in_channels, |p, v| p*v);


		let input: Vec<f32> = (0..input_size).map(|x| x as f32).collect();		
				
		
		let axis = ks.len()-1;
		
		let mut patches = vec![-0.5; patch_size*output_spaxel_count];
		
		for (i, patch) in patches.chunks_mut(patch_size).enumerate(){
			let output_ind = i*out_channels;
			super::pack_patch_recurse
		(patch, &input, &ks, in_channels, &input_shape, &output_shape, axis, output_ind, output_size);
		}
		
		//TODO: test output
		
//		for patch in patches.chunks(kernel_size){
//			println!("patch:{:?}", patch);
//		}

		//panic!();
	}
	

}
 
 