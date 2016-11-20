// Expand - depth to higher dimensions
// Collapse - higher dimensions to depth

// Linterp  - NSISOD - Integer upscaling of higher dimensions, interpolating between columns.
// MaxPool  - NSISOD - each pooling windows passes through the maximum value
// AvgPool  - NSISOD - each pooling window passes through the 
// RMSPool  - NSISOD - RMS of a given window size
// SRMSPool - NSISOD - signed RMS pooling

// FullMaxPool  - NSIHOF - A window spanning all higher dimensions is pooled as per MaxPool, producing a single column. Useful for FC clasifying layers where
// FullAvgPool  - NSIHOF - A window spanning all higher dimensions is pooled as per AvgPool,
// FullRMSPool  - NSIHOF - A window spanning all higher dimensions is pooled as per RMSPool,
// FullSRMSPool - NSIHOF - A window spanning all higher dimensions is pooled as per SRMSPool,

// Trim - trim spatial dimensions by fixed amount, start end, or both.
// TrimTo - trim spatial dimensions to given size, optional offset.

// resizecolumns

// TransferShape - no values or derivatives transfered, input shape is enforced on output shapes


use graph::*;
use std::cell::RefCell;
use std::cmp::{min, max};
use shape::*;
use ops::Operation;
use std::sync::Arc;
use matrixmultiply;


/// `LinearInterp` - NSISOD - Increase size of each spatial dimension by given a factor by linear interpolating between spaxels in the input
/// Cout = Cin
#[derive(Clone)] 
pub struct LinearInterp {
	name: String,
	factors: Vec<usize>,
	input_ind: NodeIndex,
	output_ind: NodeIndex,
	// input_channels: usize,
	// output_channels: usize,
	upscale_matrix: Vec<f32>,
}
	
impl LinearInterp {
	pub fn new(&(input, ref input_shape): &(NodeIndex, NodeShape), &(output, ref output_shape): &(NodeIndex, NodeShape), factors: Vec<usize>, name: &str) -> Box<LinearInterp>{
		assert_eq!(input_shape.rank(), output_shape.rank());
		assert_eq!(input_shape.rank(), output_shape.rank());
		assert_eq!(input_shape.rank(), factors.len() + 1);
		assert_eq!(input_shape.channels, output_shape.channels);
		let matrix = upscale_matrix(&factors);
		Box::new(LinearInterp{
			name: name.to_string(),
			factors: factors,
			input_ind: input,
			output_ind: output,
			// input_channels: input_shape.channels,
			// output_channels: output_shape.channels,
			upscale_matrix: matrix,
		})
	}


}

/// Packs values from lowres tensor/node into the B matrix for C = A x B upscaling
#[allow(dead_code)]
fn pack_lowres_matrix(_input: &[f32], _matrix: &mut [f32], _shape: &[usize]){

}

// Note: Due to the requirement of handling boundaries the number of patches required across  a given dimension is D+1,

// input must be the input_shape, not output_shape. Returns the stride over which a given axis must move
fn shape_to_strides(shape: &[usize]) -> Vec<usize> {
	(0..shape.len()+1).map(|i|{
		shape.iter().take(i).map(|i| i + 1).product()
	}).collect()
}

/// 
#[allow(dead_code)]
fn pack_lowres_recurse(input: &mut [f32], input_shape: &[usize], matrix: &mut [f32], axis: usize, n_channels: usize, patch_index: usize, patch_strides: &[usize]){
	let len = matrix.len();
	debug_assert_eq!(0, len%2);

	//let patch_stride = old_patch_stride/(input_shape[axis]+1);// spaxel stride

	let in_stride = input.len()/input_shape[axis]; // array stride (spaxelstride*n_channels)
	let patch_stride = patch_strides[axis];
	let patch_x = (patch_index%patch_strides[axis+1])/patch_stride;//(patch_index % old_patch_stride)/patch_stride;
	let in_x = patch_x as isize - 1; // start of the image patch in the current recurse.

	
	for (i, m) in matrix.chunks_mut(len/2).enumerate(){
		let in_x = min(input_shape[axis] - 1 ,max(0, in_x+i as isize)as usize); // handle image boundaries by repeating samples inside the valid range.

		let new_input = &mut input[in_x*in_stride..(in_x+1)*in_stride];
		if axis > 0 {

			let new_axis = axis - 1;			
			pack_lowres_recurse(new_input, input_shape, m, new_axis, n_channels, patch_index, patch_strides);
		} else {
			debug_assert_eq!(new_input.len(), n_channels);
			debug_assert_eq!(0, m.len() % n_channels);
			debug_assert_eq!(patch_stride, 1);

			let new_input = &new_input[0..n_channels];
			
			let m = &mut m[patch_index*n_channels..(patch_index+1)*n_channels];
			m.copy_from_slice(new_input);

		}
	}


}


/// reads from C matrix into tensor/node after C= A x B upscaling
#[allow(dead_code)]
fn unpack_hires_matrix(output: &mut [f32], output_shape: &[usize], matrix: &mut [f32], factors: &[usize], axis: usize, n_channels: usize, patch_index: usize, patch_strides: &[usize]){
	let len = matrix.len();
	debug_assert_eq!(0, len%factors[axis]);


	let out_stride = output.len()/output_shape[axis]; // array stride (spaxelstride*n_channels)
	let patch_stride = patch_strides[axis];
	let patch_x = (patch_index%patch_strides[axis+1])/patch_stride;//(patch_index % old_patch_stride)/patch_stride;
	let out_x = (patch_x as isize-1)*factors[axis]as isize + factors[axis]as isize/2 ; // start of the image patch in the current recurse.

	//println!("{} {}",output.len(), output_shape[axis]);
	for (i, m) in matrix.chunks_mut(len/factors[axis]).enumerate(){
		let out_x = out_x + i as isize;
		let out_x = if out_x < 0 || out_x >= output_shape[axis] as isize { // handle image boundaries by skipping them
			continue;
		} else {
			out_x as usize
		};

		//println!("{}", out_x);
		let new_output = &mut output[out_x*out_stride..(out_x+1)*out_stride];
		if axis > 0 {

			let new_axis = axis - 1;
			
			unpack_hires_matrix(new_output, output_shape, m, factors, new_axis, n_channels, patch_index, patch_strides);
		} else {
			debug_assert_eq!(new_output.len(), n_channels);
			debug_assert_eq!(0, m.len() % n_channels);
			debug_assert_eq!(patch_stride, 1);

			let new_output = &mut new_output[0..n_channels];
			
			let m = &m[patch_index*n_channels..][..n_channels];

			for j in 0..n_channels{
				new_output[j] += m[j];
			}
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
	
	let x = factors[axis];
	let step = 1.0/x as f32;
	let start = 1.0-((x+1)%2)as f32*0.5*step;
	
	// size of already filled patch;
	let cols = 2usize.pow(axis as u32);
	let rows = factors[0..axis].iter().product();
	let col_offset = col_stride*cols;
	
	for ri in 0..x {
		let i = x-ri-1; // must do blocks in reverse order so that data in top left of matrix isnt overwritten too early
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

impl Operation for LinearInterp {

	fn name(&self) -> &str{&self.name}
	
	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_ind].name, self.name));
		

		let required_shape = {
			let dims = shapes[self.input_ind].spatial_dimensions.iter().map(|dim| match dim {
				&Dimension::Fixed(v) => v,
				_ => unreachable!(),
			});
			
			NodeShape{
				channels: shapes[self.output_ind].channels,
				spatial_dimensions: dims.zip(&self.factors).map(|(dim, f)| Dimension::Range{lower: (dim-1)*f + 1, upper: dim * f}).collect(),
			}
		};

		shapes[self.output_ind] = required_shape.merge(&shapes[self.output_ind])
			.expect(&format!("Error: Operation '{}' could not merge required output shape with existing shape for Node '{}'", self.name, nodes[self.output_ind].name));

	}
	
	fn input_node_ind(&self) -> Vec<NodeIndex>{vec![self.input_ind]}
	
	fn output_node_ind(&self) -> Vec<NodeIndex>{vec![self.output_ind]}
		
	fn num_params(&self) -> usize {0}
	
	fn forward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32]){
		let input = &mut *{data[self.input_ind].borrow_mut()};
		let output = &mut *{data[self.output_ind].borrow_mut()};
		let in_size = input.shape.flat_size_single();
		let out_size = output.shape.flat_size_single();


		// These checks shouldnt be necessary unless code in the graph doesnt correctly resolve compatible shapes.
		// should check shape compatability under padding etc assert!(input.shape.n == output.shape.n);
		assert_eq!(input.shape.n, output.shape.n);
		assert_eq!(input.shape.channels, output.shape.channels);//,"{:?} {:?}", input, output);
		assert_eq!(input.shape.spatial_dimensions.len(), output.shape.spatial_dimensions.len());
		assert_eq!(input.shape.spatial_dimensions.len(), self.factors.len());

		let n_channels = input.shape.channels;
		let patch_strides = shape_to_strides(&input.shape.spatial_dimensions);
		let n_patches =patch_strides[patch_strides.len()-1];
		let k = 2usize.pow(self.factors.len() as u32);
		let m = self.factors.iter().product();
		let n = n_channels * n_patches;

		let mut hires_matrix =  Vec::with_capacity(m * n);
		unsafe{
			hires_matrix.set_len(m * n);
		}

		let mut lores_matrix =  Vec::with_capacity(k * n);
		unsafe{
			lores_matrix.set_len(k * n);
		}		

		for n_ind in 0..input.shape.n{
			let out_n = &mut output.values[n_ind*out_size..][..out_size];
			let in_n = &mut input.values[n_ind*in_size..][..in_size];

			for i in 0..n_patches{
				pack_lowres_recurse(in_n, &input.shape.spatial_dimensions, &mut lores_matrix, self.factors.len()-1, n_channels, i, &patch_strides);
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
				unpack_hires_matrix(out_n, &output.shape.spatial_dimensions, &mut hires_matrix, &self.factors, self.factors.len()-1, n_channels, i, &patch_strides)

			}
		}

		
	}
	
	fn backward (&mut self, _data: &mut [RefCell<NodeData>], _params: &[f32], _param_deriv: &mut [f32], _error: &mut f32){
		// let input = &mut *{data[self.input_ind].borrow_mut()};
		// let output = &*{data[self.output_ind].borrow_mut()};
		// let in_size = input.shape.flat_size_single();
		// let out_size = output.shape.flat_size_single();
				
		// // These checks shouldnt be necessary unless code in the graph doesnt correctly resolve compatible shapes.
		// // should check shape compatability under padding etc assert!(input.shape.n == output.shape.n);
		// assert_eq!(input.shape.n, output.shape.n);
		// assert_eq!(input.shape.channels, self.input_channels);
		// assert_eq!(output.shape.channels, self.output_channels);
		// assert_eq!(input.shape.spatial_dimensions.len(), output.shape.spatial_dimensions.len());
		// assert_eq!(input.shape.spatial_dimensions.len(), self.factors.len());

		// TODO:
		// unimplemented!();
	}

}


/// `ShapeConstraint` - NSISOD - Propagates custom shape constraints
#[derive(Clone)] 
pub struct ShapeConstraint {
 	name: String,
	rules: Vec<Arc<Fn(usize) -> usize>>,
 	input_ind: NodeIndex,
	output_ind: NodeIndex,
}
 	
impl ShapeConstraint {
	pub fn new(&(input, ref input_shape): &(NodeIndex, NodeShape), &(output, ref output_shape): &(NodeIndex, NodeShape), rules: Vec<Arc<Fn(usize) -> usize>>, name: &str) -> Box<ShapeConstraint>{
		assert_eq!(input_shape.rank(), output_shape.rank());
		assert_eq!(input_shape.rank(), rules.len() + 1);

		Box::new(ShapeConstraint{
			name: name.to_string(),
			rules: rules,
			input_ind: input,
			output_ind: output,
		})
	}
}

impl Operation for ShapeConstraint {

	fn name(&self) -> &str{&self.name}
	
	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Input node '{}' could not be Pooling to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_ind].name, self.name));
		

		let required_shape = {
			let dims = shapes[self.input_ind].spatial_dimensions.iter().map(|dim| match dim {
				&Dimension::Fixed(v) => v,
				_ => unreachable!(),
			});
			
			NodeShape{
				channels: shapes[self.output_ind].channels,
				spatial_dimensions: dims.zip(&self.rules).map(|(dim, rule)| Dimension::Fixed(rule(dim)) ).collect(),
			}
		};
		
		shapes[self.output_ind] = required_shape.merge(&shapes[self.output_ind])
			.expect(&format!("Error: Operation '{}' could not merge required output shape with existing shape for Node '{}'", self.name, nodes[self.output_ind].name));
		
	}
	
	fn input_node_ind(&self) -> Vec<NodeIndex>{vec![self.input_ind]}
	
	fn output_node_ind(&self) -> Vec<NodeIndex>{vec![self.output_ind]}
		
	fn num_params(&self) -> usize {0}
	
	fn forward (&mut self, _data: &mut [RefCell<NodeData>], _params: &[f32]){}
	
	fn backward (&mut self, _data: &mut [RefCell<NodeData>], _params: &[f32], _param_deriv: &mut [f32], _error: &mut f32){}		
}


/// Pooling - NSISOD - Decrease size of higher dimensions by given factors by mapping from each spaxel to chunks of the channel dimension
/// Cout = Cin.F1.F2.F3
#[derive(Clone)] 
pub struct Pooling {
 	name: String,
 	factors: Vec<usize>,
 	input_ind: NodeIndex,
	output_ind: NodeIndex,
	input_channels: usize,
	output_channels: usize,
}
 	
impl Pooling {
	pub fn new(&(input, ref input_shape): &(NodeIndex, NodeShape), &(output, ref output_shape): &(NodeIndex, NodeShape), factors: Vec<usize>, name: &str) -> Box<Pooling>{
		assert_eq!(input_shape.rank(), output_shape.rank());
		assert_eq!(input_shape.rank(), factors.len() + 1);
		assert_eq!(output_shape.channels, input_shape.channels,
			"Error: Operation '{}' output node shape channel dimension must be equal to the input node channel dimension", name);

		Box::new(Pooling{
			name: name.to_string(),
			factors: factors,
			input_ind: input,
			output_ind: output,
			input_channels: input_shape.channels,
			output_channels: output_shape.channels,
		})
	}
}

impl Operation for Pooling {

	fn name(&self) -> &str{&self.name}
	
	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Input node '{}' could not be Pooling to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_ind].name, self.name));
		

		let required_shape = {
			let dims = shapes[self.input_ind].spatial_dimensions.iter().map(|dim| match dim {
				&Dimension::Fixed(v) => v,
				_ => unreachable!(),
			});
			
			NodeShape{
				channels: shapes[self.input_ind].channels,
				spatial_dimensions: dims.zip(&self.factors).map(|(dim, f)| Dimension::Fixed((dim + f-1)/f)).collect(),
			}
		};
		
		shapes[self.output_ind] = required_shape.merge(&shapes[self.output_ind])
			.expect(&format!("Error: Operation '{}' could not merge required output shape with existing shape for Node '{}'", self.name, nodes[self.output_ind].name));
		
	}
	
	fn input_node_ind(&self) -> Vec<NodeIndex>{vec![self.input_ind]}
	
	fn output_node_ind(&self) -> Vec<NodeIndex>{vec![self.output_ind]}
		
	fn num_params(&self) -> usize {0}
	
	fn forward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32]){
		let input = &*{data[self.input_ind].borrow_mut()};
		let output = &mut *{data[self.output_ind].borrow_mut()};
		let in_size = input.shape.flat_size_single();
		let out_size = output.shape.flat_size_single();

		// These checks shouldnt be necessary unless code in the graph doesnt correctly resolve compatible shapes.
		// should check shape compatability under padding etc assert!(input.shape.n == output.shape.n);
		assert_eq!(input.shape.n, output.shape.n);
		assert_eq!(input.shape.channels, self.input_channels);
		assert_eq!(output.shape.channels, self.output_channels);
		assert_eq!(input.shape.spatial_dimensions.len(), output.shape.spatial_dimensions.len());
		assert_eq!(input.shape.spatial_dimensions.len(), self.factors.len());

		let scale = 1.0/self.factors.iter().fold(1,|p, v| p* v) as f32;

		for n_ind in  0..input.shape.n{
			let out_n = &mut output.values[n_ind*out_size..][..out_size];
			let in_n = &input.values[n_ind*in_size..][..in_size];

			for (i, patch) in out_n.chunks_mut(self.output_channels).enumerate(){
				let output_ind = i*self.output_channels;
				pool_recurse_forward(patch, &in_n, &self.factors, self.input_channels, &input.shape.spatial_dimensions, &output.shape.spatial_dimensions, self.factors.len() - 1, output_ind, out_size, scale);
			}
		}
	}
	
	fn backward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32], _param_deriv: &mut [f32], _error: &mut f32){
		let input = &mut *{data[self.input_ind].borrow_mut()};
		let output = &*{data[self.output_ind].borrow_mut()};
		let in_size = input.shape.flat_size_single();
		let out_size = output.shape.flat_size_single();
				
		// These checks shouldnt be necessary unless code in the graph doesnt correctly resolve compatible shapes.
		// should check shape compatability under padding etc assert!(input.shape.n == output.shape.n);
		assert_eq!(input.shape.n, output.shape.n);
		assert_eq!(input.shape.channels, self.input_channels);
		assert_eq!(output.shape.channels, self.output_channels);
		assert_eq!(input.shape.spatial_dimensions.len(), output.shape.spatial_dimensions.len());
		assert_eq!(input.shape.spatial_dimensions.len(), self.factors.len());

		let scale = 1.0/self.factors.iter().fold(1,|p, v| p* v) as f32;

		for n_ind in  0..input.shape.n{
			let outd_n = &output.derivatives[n_ind*out_size..][..out_size];
			let ind_n = &mut input.derivatives[n_ind*in_size..][..in_size];
			
			for (i, patch) in outd_n.chunks(self.output_channels).enumerate(){
				let output_ind = i*self.output_channels;
				// input derivs are the output of this functions so all names are reversed from Pooling, and then reversed again
				pool_recurse_backward(patch, ind_n, &self.factors, self.input_channels, &input.shape.spatial_dimensions, &output.shape.spatial_dimensions, self.factors.len() - 1, output_ind, out_size, scale) ;
			}			
		}			
	}		
}


/// Collapse - NSISOD - Decrease size of higher dimensions by given factors by mapping from each spaxel to chunks of the channel dimension
/// Cout = Cin.F1.F2.F3
#[derive(Clone)] 
pub struct Collapse {
 	name: String,
 	factors: Vec<usize>,
 	input_ind: NodeIndex,
	output_ind: NodeIndex,
	input_channels: usize,
	output_channels: usize,
}
 	
impl Collapse {
	pub fn new(&(input, ref input_shape): &(NodeIndex, NodeShape), &(output, ref output_shape): &(NodeIndex, NodeShape), factors: Vec<usize>, name: &str) -> Box<Collapse>{
		assert_eq!(input_shape.rank(), output_shape.rank());
		assert_eq!(input_shape.rank(), factors.len() + 1);
		assert_eq!(output_shape.channels, factors.iter().fold(input_shape.channels, |p,v| p*v),
			"Error: Operation '{}' output node shape channel dimension must be equal to the product of the input node channel dimensions and each of the factors", name);

		Box::new(Collapse{
			name: name.to_string(),
			factors: factors,
			input_ind: input,
			output_ind: output,
			input_channels: input_shape.channels,
			output_channels: output_shape.channels,
		})
	}
}


impl Operation for Collapse {

	fn name(&self) -> &str{&self.name}
	
	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Input node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_ind].name, self.name));
		

		let required_shape = {
			let dims = shapes[self.input_ind].spatial_dimensions.iter().map(|dim| match dim {
				&Dimension::Fixed(v) => v,
				_ => unreachable!(),
			});
			
			NodeShape{
				channels: shapes[self.output_ind].channels,
				spatial_dimensions: dims.zip(&self.factors).map(|(dim, f)| Dimension::Fixed((dim + f-1)/f)).collect(),
			}
		};
		
		shapes[self.output_ind] = required_shape.merge(&shapes[self.output_ind])
			.expect(&format!("Error: Operation '{}' could not merge required output shape with existing shape for Node '{}'", self.name, nodes[self.output_ind].name));
		
	}
	
	fn input_node_ind(&self) -> Vec<NodeIndex>{vec![self.input_ind]}
	
	fn output_node_ind(&self) -> Vec<NodeIndex>{vec![self.output_ind]}
		
	fn num_params(&self) -> usize {0}
	
	fn forward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32]){
		let input = &*{data[self.input_ind].borrow_mut()};
		let output = &mut *{data[self.output_ind].borrow_mut()};
		let in_size = input.shape.flat_size_single();
		let out_size = output.shape.flat_size_single();

		// These checks shouldnt be necessary unless code in the graph doesnt correctly resolve compatible shapes.
		// should check shape compatability under padding etc assert!(input.shape.n == output.shape.n);
		assert_eq!(input.shape.n, output.shape.n);
		assert_eq!(input.shape.channels, self.input_channels);
		assert_eq!(output.shape.channels, self.output_channels);
		assert_eq!(input.shape.spatial_dimensions.len(), output.shape.spatial_dimensions.len());
		assert_eq!(input.shape.spatial_dimensions.len(), self.factors.len());

		for n_ind in  0..input.shape.n{
			let out_n = &mut output.values[n_ind*out_size..][..out_size];
			let in_n = &input.values[n_ind*in_size..][..in_size];

			for (i, patch) in out_n.chunks_mut(self.output_channels).enumerate(){
				let output_ind = i*self.output_channels;
				collapse_recurse(patch, &in_n, &self.factors, self.input_channels, &input.shape.spatial_dimensions, &output.shape.spatial_dimensions, self.factors.len() - 1, output_ind, out_size);
			}
		}
	}
	
	fn backward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32], _param_deriv: &mut [f32], _error: &mut f32){
		let input = &mut *{data[self.input_ind].borrow_mut()};
		let output = &*{data[self.output_ind].borrow_mut()};
		let in_size = input.shape.flat_size_single();
		let out_size = output.shape.flat_size_single();
				
		// These checks shouldnt be necessary unless code in the graph doesnt correctly resolve compatible shapes.
		// should check shape compatability under padding etc assert!(input.shape.n == output.shape.n);
		assert_eq!(input.shape.n, output.shape.n);
		assert_eq!(input.shape.channels, self.input_channels);
		assert_eq!(output.shape.channels, self.output_channels);
		assert_eq!(input.shape.spatial_dimensions.len(), output.shape.spatial_dimensions.len());
		assert_eq!(input.shape.spatial_dimensions.len(), self.factors.len());

		for n_ind in  0..input.shape.n{
			let outd_n = &output.derivatives[n_ind*out_size..][..out_size];
			let ind_n = &mut input.derivatives[n_ind*in_size..][..in_size];
			
			for (i, patch) in outd_n.chunks(self.output_channels).enumerate(){
				let output_ind = i*self.output_channels;
				//println!("out ind{}", output_ind);
				// input derivs are the output of this functions so all names are reversed from collapse, and then reversed again
				expand_recurse(patch, ind_n, &self.factors, self.input_channels, &input.shape.spatial_dimensions, &output.shape.spatial_dimensions, self.factors.len() - 1, output_ind, out_size) ;
			}			
		}			
	}		
}


/// Expand - NSISOD - Increase size of higher dimensions by given factors by mapping from chunks of the channel dimension to each new spaxel
/// Cout = Cin/(F1.F2.F3)
#[derive(Clone)] 
pub struct Expand {
	name: String,
	factors: Vec<usize>,
	input_ind: NodeIndex,
	output_ind: NodeIndex,
	input_channels: usize,
	output_channels: usize,
}
	
impl Expand {
	pub fn new(&(input, ref input_shape): &(NodeIndex, NodeShape), &(output, ref output_shape): &(NodeIndex, NodeShape), factors: Vec<usize>, name: &str) -> Box<Expand>{
		assert_eq!(input_shape.rank(), output_shape.rank());
		assert_eq!(input_shape.rank(), output_shape.rank());
		assert_eq!(input_shape.rank(), factors.len() + 1);
		assert_eq!(input_shape.channels, factors.iter().fold(output_shape.channels, |p,v| p*v),
			"Error: Operation '{}' output node shape channel dimensions must be equal to the input node channel dimensions divided by each of the factors without remainder", name);
		Box::new(Expand{
			name: name.to_string(),
			factors: factors,
			input_ind: input,
			output_ind: output,
			input_channels: input_shape.channels,
			output_channels: output_shape.channels,
		})
	}
}

impl Operation for Expand {

	fn name(&self) -> &str{&self.name}
	
	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Input node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_ind].name, self.name));
		

		let required_shape = {
			let dims = shapes[self.input_ind].spatial_dimensions.iter().map(|dim| match dim {
				&Dimension::Fixed(v) => v,
				_ => unreachable!(),
			});
			
			NodeShape{
				channels: shapes[self.output_ind].channels,
				spatial_dimensions: dims.zip(&self.factors).map(|(dim, f)| Dimension::Range{lower: (dim-1)*f + 1, upper: dim * f}).collect(),
			}
		};
		
		shapes[self.output_ind] = required_shape.merge(&shapes[self.output_ind])
			.expect(&format!("Error: Operation '{}' could not merge required output shape with existing shape for Node '{}'", self.name, nodes[self.output_ind].name));
		
	}
	
	fn input_node_ind(&self) -> Vec<NodeIndex>{vec![self.input_ind]}
	
	fn output_node_ind(&self) -> Vec<NodeIndex>{vec![self.output_ind]}
		
	fn num_params(&self) -> usize {0}
	
	fn forward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32]){
		let input = &*{data[self.input_ind].borrow_mut()};
		let output = &mut *{data[self.output_ind].borrow_mut()};
		let in_size = input.shape.flat_size_single();
		let out_size = output.shape.flat_size_single();

		// These checks shouldnt be necessary unless code in the graph doesnt correctly resolve compatible shapes.
		// should check shape compatability under padding etc assert!(input.shape.n == output.shape.n);
		assert_eq!(input.shape.n, output.shape.n);
		assert_eq!(input.shape.channels, self.input_channels);
		assert_eq!(output.shape.channels, self.output_channels);
		assert_eq!(input.shape.spatial_dimensions.len(), output.shape.spatial_dimensions.len());
		assert_eq!(input.shape.spatial_dimensions.len(), self.factors.len());

		for n_ind in  0..input.shape.n{
			let out_n = &mut output.values[n_ind*out_size..][..out_size];
			let in_n = &input.values[n_ind*in_size..][..in_size];
			
			for (i, patch) in in_n.chunks(self.input_channels).enumerate(){
				let input_ind = i*self.input_channels;

				expand_recurse(patch, out_n, &self.factors, self.output_channels, &output.shape.spatial_dimensions, &input.shape.spatial_dimensions, self.factors.len() - 1, input_ind, in_size)
			}

			// for (i, patch) in out_n.chunks_mut(self.output_channels).enumerate(){
			// 	let output_ind = i*self.output_channels;
			// 	collapse_recurse(patch, &in_n, &self.factors, self.input_channels, &input.shape.higher_dimensions, &output.shape.higher_dimensions, self.factors.len() - 1, output_ind, out_size);
			// }
		}
	}
	
	fn backward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32], _param_deriv: &mut [f32], _error: &mut f32){
		let input = &mut *{data[self.input_ind].borrow_mut()};
		let output = &*{data[self.output_ind].borrow_mut()};
		let in_size = input.shape.flat_size_single();
		let out_size = output.shape.flat_size_single();
				
		// These checks shouldnt be necessary unless code in the graph doesnt correctly resolve compatible shapes.
		// should check shape compatability under padding etc assert!(input.shape.n == output.shape.n);
		assert_eq!(input.shape.n, output.shape.n);
		assert_eq!(input.shape.channels, self.input_channels);
		assert_eq!(output.shape.channels, self.output_channels);
		assert_eq!(input.shape.spatial_dimensions.len(), output.shape.spatial_dimensions.len());
		assert_eq!(input.shape.spatial_dimensions.len(), self.factors.len());

		for n_ind in  0..input.shape.n{
			let outd_n = &output.derivatives[n_ind*out_size..][..out_size];
			let ind_n = &mut input.derivatives[n_ind*in_size..][..in_size];
			
			for (i, patch) in ind_n.chunks_mut(self.input_channels).enumerate(){
				let input_ind = i*self.input_channels;
				collapse_recurse(patch, outd_n, &self.factors, self.output_channels, &output.shape.spatial_dimensions, &input.shape.spatial_dimensions, self.factors.len() - 1, input_ind, in_size);
			}

			// for (i, patch) in outd_n.chunks(self.output_channels).enumerate(){
			// 	let output_ind = i*self.output_channels;
			// 	//println!("out ind{}", output_ind);
			// 	// input derivs are the output of this functions so all names are reversed from collapse, and then reversed again
			// 	expand_recurse(patch, ind_n, &self.factors, self.input_channels, &input.shape.higher_dimensions, &output.shape.higher_dimensions, self.factors.len() - 1, output_ind, out_size) ;
			// }			
		}			
	}		
}

#[allow(dead_code)]
fn pool_recurse_forward(patch: &mut [f32], input: &[f32], factors:&[usize], in_channels: usize, input_shape: &[usize], output_shape: &[usize], axis: usize, output_ind: usize, old_out_stride: usize, scale: f32){
	debug_assert_eq!(scale, 1.0/factors.iter().fold(1,|p, v| p* v) as f32);

	// stride in array index, not spaxel index
	let out_stride = old_out_stride/output_shape[axis];
	let in_stride = input.len()/input_shape[axis];
	
	// coordinates of the centre spaxel of the kernel, for the current axis
	let ox = (output_ind % old_out_stride)/out_stride;

	// valid range of input coordinates the current axis for the current patch
	let start = ox * factors[axis];
	let end = min(start + factors[axis], input_shape[axis]);
		
	if axis > 0 {
		
		for i in start..end{
			let new_input = &input[in_stride*i..in_stride*(i+1)];

			let new_axis = axis-1;
			pool_recurse_forward(patch, new_input, factors, in_channels, input_shape, output_shape, new_axis, output_ind, out_stride, scale);
		}

	} else {	

		let offset = start * in_channels;
		let len = (end - start)*in_channels;
		let input = &input[offset..][..len];
		
		
		for i in 0.. input.len()/patch.len(){
			let input = &input[i*patch.len()..][..patch.len()];
			for j in 0..patch.len(){
				patch[j] += input[j] * scale;
			}
		}
	}	
}

#[allow(dead_code)]
fn pool_recurse_backward(patch: &[f32], output: &mut [f32], factors:&[usize], out_channels: usize, output_shape: &[usize], input_shape: &[usize], axis: usize, input_ind: usize, old_in_stride: usize, scale: f32){
	debug_assert_eq!(scale, 1.0/factors.iter().fold(1,|p, v| p* v) as f32);
	
	// stride in array index, not spaxel index
	let in_stride = old_in_stride/input_shape[axis];
	let out_stride = output.len()/output_shape[axis];

	
	// coordinates of the centre spaxel of the kernel, for the current axis
	let ix = (input_ind % old_in_stride)/in_stride;
 
	// valid range of input coordinates the current axis for the current patch
	let start = ix * factors[axis];
	let end = min(start + factors[axis], output_shape[axis]);


	
	if axis > 0 {
		
		for i in start..end{
			let new_output = &mut output[out_stride*i..out_stride*(i+1)];
			
			let new_axis = axis-1;

			pool_recurse_backward(patch, new_output, factors, out_channels, output_shape, input_shape, new_axis, input_ind, in_stride, scale);
		}

	} else {	
		
		let offset = start * out_channels;
		let len = (end - start)*out_channels;
		let output = &mut output[offset..][..len];
		
		
		for i in 0.. output.len()/patch.len(){
			let output = &mut output[i*patch.len()..][..patch.len()];
			for j in 0..patch.len(){
				output[j] += patch[j] * scale;
			}
		}


	}	
	
}


fn collapse_recurse(patch: &mut [f32], input: &[f32], factors:&[usize], in_channels: usize, input_shape: &[usize], output_shape: &[usize], axis: usize, output_ind: usize, old_out_stride: usize){
	
	// stride in array index, not spaxel index
	let out_stride = old_out_stride/output_shape[axis];
	let in_stride = input.len()/input_shape[axis];
	let patch_stride = patch.len()/factors[axis];
	
	// coordinates of the centre spaxel of the kernel, for the current axis
	let ox = (output_ind % old_out_stride)/out_stride;

	// valid range of input coordinates the current axis for the current patch
	let start = ox * factors[axis];
	let end = min(start + factors[axis], input_shape[axis]);
		
	if axis > 0 {
		
		for i in start..end{
			let new_input = &input[in_stride*i..in_stride*(i+1)];
			
			let i_patch = i - start;
			let new_patch = &mut patch[i_patch*patch_stride..(i_patch+1)*patch_stride];
			let new_axis = axis-1;

			collapse_recurse(new_patch, new_input, factors, in_channels, input_shape, output_shape, new_axis, output_ind, out_stride);
		}

	} else {	
		
		let offset = start * in_channels;
		let len = (end - start)*in_channels;
		let input_crop = &input[offset..][..len];
		let patch_crop = &mut patch[..len];
		
		for i in 0..len{
			patch_crop[i] += input_crop[i];
		}
	}	
		
}


fn expand_recurse(patch: &[f32], output: &mut [f32], factors:&[usize], out_channels: usize, output_shape: &[usize], input_shape: &[usize], axis: usize, input_ind: usize, old_in_stride: usize){
	
	// stride in array index, not spaxel index
	let in_stride = old_in_stride/input_shape[axis];
	let out_stride = output.len()/output_shape[axis];
	let patch_stride = patch.len()/factors[axis];
	
	// coordinates of the centre spaxel of the kernel, for the current axis
	let ix = (input_ind % old_in_stride)/in_stride;

	// valid range of input coordinates the current axis for the current patch
	let start = ix * factors[axis];
	let end = min(start + factors[axis], output_shape[axis]);


	
	if axis > 0 {
		
		for i in start..end{
			let new_output = &mut output[out_stride*i..out_stride*(i+1)];
			
			let i_patch = i - start;
			let new_patch = &patch[i_patch*patch_stride..(i_patch+1)*patch_stride];
			let new_axis = axis-1;

			expand_recurse(new_patch, new_output, factors, out_channels, output_shape, input_shape, new_axis, input_ind, in_stride);
		}

	} else {	
		
		let offset = start * out_channels;
		let len = (end - start)*out_channels;
		let output_crop = &mut output[offset..][..len];
		let patch_crop = &patch[..len];
		
		for i in 0..len{
			output_crop[i] += patch_crop[i];
		}
	}	
		
}


#[cfg(test)]
mod test {
	use super::*; 	
	use graph::*;
	use ops::loss::MseLoss;
	use ops::math::*;
	use ops::*;
	use shape::DataShape;

	#[test]
	fn test_linear_interp(){
		{
			let mut g = Graph::new();
			let input =g.add_input_node(Node::new_sized(1, vec![2,2], "input"));
			let output =g.add_output_node(Node::new_sized(1, vec![4,4], "output"));
			g.add_operation(LinearInterp::new(&input, &output, vec![2, 2], "linterp"));
			//g.add_operation(ShapeConstraint::new(&input, &output, vec![Arc::new(move|d| d*2), Arc::new(move|d| d*2)], "exact size constraint"));	

			let mut data_in = NodeData::new_blank(DataShape::new(1, vec![2,2], 1));
			data_in.values[1] = 1.0;
			data_in.values[2] = 1.0;

			let out = g.forward(1, vec![data_in], &vec![]).remove(0);

			let expected = vec![
				0.0,  0.25,  0.75,  1.0,
				0.25, 0.375, 0.625, 0.75,
				0.75, 0.625, 0.375, 0.25,
				1.0,  0.75,  0.25,  0.0,
			];

			let diff = out.values.iter().zip(expected.iter()).fold(0.0, |acc, (&o, &e)| (o-e)*(o-e));
			assert!(diff < 1e-6, "{:?} {:?}", out.values, expected);
		}



		{
			let mut g = Graph::new();
			let input =g.add_input_node(Node::new_sized(1, vec![3,3], "input"));
			let output =g.add_output_node(Node::new_sized(1, vec![9,9], "output"));
			g.add_operation(LinearInterp::new(&input, &output, vec![3, 3], "linterp"));
			//g.add_operation(ShapeConstraint::new(&input, &output, vec![Arc::new(move|d| d*3), Arc::new(move|d| d*3)], "exact size constraint"));

			let mut data_in = NodeData::new_blank(DataShape::new(1, vec![3,3], 1));
			data_in.values[4] = 1.0;

			let out = g.forward(1, vec![data_in], &vec![]).remove(0);

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

			let diff = out.values.iter().zip(expected.iter()).fold(0.0, |acc, (&o, &e)| acc+(o-e)*(o-e));
			assert!(diff < 1e-6, "{:?} {:?}", out.values, expected);
		}
	}


	#[test]
	fn test_linear_interp_matrix(){


		let factors_list = vec![
			vec![3],
			vec![2],
			vec![3, 2],
			vec![1, 1, 1],
			vec![13, 31, 5],
			vec![16, 4, 8, 2],
		];

		for factors in factors_list {
			
			let matrix = reshape::upscale_matrix(&factors[..]);
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
			let matrix = reshape::upscale_matrix(&vec![3]);
			let expected = vec![ // col major
				1.0, 2./3., 1./3.,
				0.0, 1./3., 2./3.,
			];

			let diff = matrix.iter().zip(expected.iter()).fold(0.0, |acc, (&m, &e)| acc+(m-e)*(m-e));
			assert!(diff < 1e-6, "{:?} {:?}", matrix, expected);

		}

		{
			let matrix = reshape::upscale_matrix(&vec![2]);
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

		let input_shape = vec![3, 2];
		let n_channels = 5usize;

		let input_size = input_shape.iter().fold(n_channels, |acc, v| acc*v);
		let mut input = (0..input_size).map(|x| x as f32).collect::<Vec<_>>();

		let n_patches = input_shape.iter().fold(1, |acc, v| acc * (v+1));
		let axis = input_shape.len() - 1;

		let rows = 2usize.pow(input_shape.len()as u32);
		let mut matrix = vec![0.0; rows*n_channels*n_patches];
		let strides = reshape::shape_to_strides(&input_shape);
		for index in 0..n_patches{
			reshape::pack_lowres_recurse(&mut input, &input_shape, &mut matrix, axis, n_channels, index, &strides);
			
			// for y in 0..rows{
			// 	let start = y*n_channels*n_patches + index*n_channels;
			// 	let slice = &matrix[start..start+n_channels];
			// 	println!("{:?}", slice);
			// }
			// println!();
		}
		
		// for row in matrix.chunks(n_channels*n_patches){
		// 	println!("{:?}", row);
		// }

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
		let input_shape = vec![2, 3];
		let factors = vec![2, 2];
		let output_shape = input_shape.iter().zip(factors.iter()).map(|(x, f)| x*f).collect::<Vec<usize>>();
		let n_channels = 1usize;

		let output_size = output_shape.iter().fold(n_channels, |acc, v| acc*v);
		let mut output = vec![0.0; output_size];
		
		

		let n_patches: usize = input_shape.iter().fold(1, |acc, v| acc * (v+1));
		let axis = input_shape.len() - 1;

		let rows: usize = factors.iter().product();//2usize.pow(input_shape.len()as u32);
		let mut matrix = (0..rows*n_channels*n_patches).map(|x: usize| (x/n_patches + (x%n_patches)*rows) as f32).collect::<Vec<f32>>();
		let strides = reshape::shape_to_strides(&input_shape);

		for index in 0..n_patches{
			reshape::unpack_hires_matrix(&mut output, &output_shape, &mut matrix, &factors, axis, n_channels, index, &strides);
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


	#[test]
	fn test_pooling_backprop(){
		for _ in 1..100{
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_sized(5, vec![13, 17], "nodein"));
			let n2 = graph.add_output_node(Node::new_sized(5, vec![5, 4], "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_sized(5, vec![5, 4], "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				Pooling::new(&n1, &n2, vec![3, 5], "Pooling"),
				MseLoss::new_default(&n2, &n3),
			];
			graph.add_operations(ops);
			graph.init_params();
			
			test_numeric(graph, 1.0, 1e-1);
		}
	}

	#[test]
	fn test_collapse_backprop(){
		for _ in 1..100{
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_sized(5, vec![13, 17], "nodein"));
			let n2 = graph.add_output_node(Node::new_sized(5 * 3 * 5, vec![5, 4], "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_sized(5 * 3 * 5, vec![5, 4], "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				Collapse::new(&n1, &n2, vec![3, 5], "Collapse"),
				MseLoss::new_default(&n2, &n3),
			];
			graph.add_operations(ops);
			graph.init_params();
			
			test_numeric(graph, 1.0, 1e-1);
		}
	}
	
	#[test]
	#[should_panic]
	fn test_collapse_shape_checks(){
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_sized(5, vec![13, 17], "nodein"));
			let n2 = graph.add_output_node(Node::new_sized(6, vec![13, 17], "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_sized(7, vec![13, 17], "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				Collapse::new(&n1, &n2, vec![3, 5], "Collapse"),
				MseLoss::new_default(&n2, &n3),
			];
			graph.add_operations(ops);
			graph.init_params();
			
			test_numeric(graph, 1.0, 1e-4);
	}

	#[test]
	fn test_expand_backprop(){
		for _ in 1..100{
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_sized(5 * 3 * 5, vec![5, 4], "nodein"));
			let n2 = graph.add_output_node(Node::new_sized(5, vec![13, 17], "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_sized(5, vec![13, 17], "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				Expand::new(&n1, &n2, vec![3, 5], "Expand"),
				MseLoss::new_default(&n2, &n3),
			];
			graph.add_operations(ops);
			graph.init_params();
			
			test_numeric(graph, 1.0, 1e-1);
		}
	}
	
	#[test]
	#[should_panic]
	fn test_expand_shape_checks(){
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_sized(5, vec![13, 17], "nodein"));
			let n2 = graph.add_output_node(Node::new_sized(6, vec![13, 17], "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_sized(7, vec![13, 17], "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				Expand::new(&n1, &n2, vec![3, 5],  "Expand"),
				MseLoss::new_default(&n2, &n3),
			];
			graph.add_operations(ops);
			graph.init_params();
			
			test_numeric(graph, 1.0, 1e-4);
	}

	#[test]
	fn test_collapse_recurse(){
		
		let factors = vec![3, 5];
		let input_shape = vec![5, 7];
		let output_shape: Vec<usize> = input_shape.iter().zip(factors.iter()).map(|(i, f)| (i+f-1)/f).collect();
		let in_channels = 2;
		let out_channels = factors.iter().fold(in_channels, |p, v| p*v);
		
		let input_spaxel_count = input_shape.iter().fold(1, |p, v| p*v);
		let output_spaxel_count = output_shape.iter().fold(1, |p, v| p*v);
		
		let input_size = input_spaxel_count*in_channels;
		let output_size = output_spaxel_count*out_channels;
		let patch_size = factors.iter().fold(in_channels, |p, v| p*v);


		let input: Vec<f32> = (0..input_size).map(|x| x as f32).collect();		
				
		
		let axis = factors.len()-1;
		
		let mut output = vec![0.0; patch_size*output_spaxel_count];
		
		for (i, patch) in output.chunks_mut(patch_size).enumerate(){
			let output_ind = i*out_channels;
			super::collapse_recurse(patch, &input, &factors, in_channels, &input_shape, &output_shape, axis, output_ind, output_size);
		}
		
		let target = vec![
			00.,  1.,  2.,  3.,  4.,  5.,  10., 11., 12., 13., 14., 15.,  20., 21., 22., 23., 24., 25.,  30., 31., 32., 33., 34., 35.,  40., 41., 42., 43., 44., 45.,
			06.,  7.,  8.,  9.,  0.,  0.,  16., 17., 18., 19.,  0.,  0.,  26., 27., 28., 29.,  0.,  0.,  36., 37., 38., 39.,  0.,  0.,  46., 47., 48., 49.,  0.,  0.,
			50., 51., 52., 53., 54., 55.,  60., 61., 62., 63., 64., 65.,   0.,  0.,  0.,  0.,  0.,  0.,   0.,  0.,  0.,  0.,  0.,  0.,   0.,  0.,  0.,  0.,  0.,  0.,
			56., 57., 58., 59.,  0.,  0.,  66., 67., 68., 69.,  0.,  0.,   0.,  0.,  0.,  0.,  0.,  0.,   0.,  0.,  0.,  0.,  0.,  0.,   0.,  0.,  0.,  0.,  0.,  0.,
		];


		assert_eq!(output, target);
	}
	
	#[test]
	fn test_expand_recurse(){
		
		let factors = vec![3, 5];
		let input_shape = vec![2, 2];
		let output_shape = vec![5, 7];
		let out_channels = 2;
		let in_channels = factors.iter().fold(out_channels, |p, v| p*v);
		
		let input_spaxel_count = input_shape.iter().fold(1, |p, v| p*v);
		let output_spaxel_count = output_shape.iter().fold(1, |p, v| p*v);
		
		let input_size = input_spaxel_count*in_channels;
		let output_size = output_spaxel_count*out_channels;


		let input = vec![
			00.,  1.,  2.,  3.,  4.,  5.,  10., 11., 12., 13., 14., 15.,  20., 21., 22., 23., 24., 25.,  30., 31., 32., 33., 34., 35.,  40., 41., 42., 43., 44., 45.,
			06.,  7.,  8.,  9.,  0.,  0.,  16., 17., 18., 19.,  0.,  0.,  26., 27., 28., 29.,  0.,  0.,  36., 37., 38., 39.,  0.,  0.,  46., 47., 48., 49.,  0.,  0.,
			50., 51., 52., 53., 54., 55.,  60., 61., 62., 63., 64., 65.,   0.,  0.,  0.,  0.,  0.,  0.,   0.,  0.,  0.,  0.,  0.,  0.,   0.,  0.,  0.,  0.,  0.,  0.,
			56., 57., 58., 59.,  0.,  0.,  66., 67., 68., 69.,  0.,  0.,   0.,  0.,  0.,  0.,  0.,  0.,   0.,  0.,  0.,  0.,  0.,  0.,   0.,  0.,  0.,  0.,  0.,  0.,

		];				
				
		
		let axis = factors.len()-1;
		
		let mut output = vec![0.0; output_size];
		
		
		for (i, patch) in input.chunks(in_channels).enumerate(){
			let input_ind = i*in_channels;

			super::expand_recurse(patch, &mut output, &factors, out_channels, &output_shape, &input_shape, axis, input_ind, input_size)
		}
		
		let target: Vec<f32> = (0..output_size).map(|x| x as f32).collect();

		assert_eq!(output, target);
	}
	
	#[test]
	fn test_expand_recurse_full(){
		
		

		let factors = vec![3, 5];
		let input_shape = vec![2, 2];
		let output_shape: Vec<usize> = input_shape.iter().zip(factors.iter()).map(|(i, f)| i*f).collect();
		let out_channels = 2;
		let in_channels = factors.iter().fold(out_channels, |p, v| p*v);
		
		let input_spaxel_count = input_shape.iter().fold(1, |p, v| p*v);
		let output_spaxel_count = output_shape.iter().fold(1, |p, v| p*v);
		
		let input_size = input_spaxel_count*in_channels;
		let output_size = output_spaxel_count*out_channels;


		let input = vec![
			  0.,   1.,   2.,   3.,   4.,   5.,   12.,  13.,  14.,  15.,  16.,  17.,   24.,  25.,  26.,  27.,  28.,  29.,   36.,  37.,  38.,  39.,  40.,  41.,   48.,  49.,  50.,  51.,  52.,  53.,
			  6.,   7.,   8.,   9.,  10.,  11.,   18.,  19.,  20.,  21.,  22.,  23.,   30.,  31.,  32.,  33.,  34.,  35.,   42.,  43.,  44.,  45.,  46.,  47.,   54.,  55.,  56.,  57.,  58.,  59.,
			 60.,  61.,  62.,  63.,  64.,  65.,   72.,  73.,  74.,  75.,  76.,  77.,   84.,  85.,  86.,  87.,  88.,  89.,   96.,  97.,  98.,  99., 100., 101.,  108., 109., 110., 111., 112., 113.,
			 66.,  67.,  68.,  69.,  70.,  71.,   78.,  79.,  80.,  81.,  82.,  83.,   90.,  91.,  92.,  93.,  94.,  95.,  102., 103., 104., 105., 106., 107.,  114., 115., 116., 117., 118., 119.,

		];				
				
		
		let axis = factors.len()-1;
		
		let mut output = vec![0.0; output_size];
		
		
		for (i, patch) in input.chunks(in_channels).enumerate(){
			let input_ind = i*in_channels;

			super::expand_recurse(patch, &mut output, &factors, out_channels, &output_shape, &input_shape, axis, input_ind, input_size)
		}
		
		let target: Vec<f32> = (0..output_size).map(|x| x as f32).collect();

		assert!(output == target, format!("target{:?} \n output{:?}", target, output));
	}
}