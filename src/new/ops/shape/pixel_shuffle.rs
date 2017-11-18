use new::graph::{GraphDef, NodeID, OpID, PassID, DataID, Storage, GraphShapes, Result};
use new::ops::{standard_op_name, Op, OpInstance, Pass};
use new::shape::NodeShape;
use ndarray::Dimension;
use std::any::Any;
use std::cmp::min;
use std::iter;

/// Collapse outer dimensions, shuffling entries into the channel dimension
///
/// Decrease size of higher dimensions by given factors by mapping from each spaxel to chunks of the channel dimension.
/// Output channel dimension is increased by the product of the collapse factors. Inverse operation of Expand.
#[derive(Clone)] 
pub struct Collapse {
	name: Option<String>,
	factors: Vec<usize>,
	input_id: NodeID,
	output_id: NodeID,
}

impl Collapse {
	pub fn new(input_id: &NodeID, output_id: &NodeID, factors: &[usize]) -> Self{
		Collapse {
			name: None,
			input_id: input_id.clone(),
			output_id: output_id.clone(),
			factors: factors.to_vec(),
		}
	}
}

impl Op for Collapse {
	type InstanceType = CollapseInstance;

	fn type_name(&self) -> &'static str {
		"Collapse"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		let name = standard_op_name(&self, &self.name, graph, &[self.input_id.clone()], &[self.output_id.clone()]);

		Ok(CollapseInstance{
			name: name,
			factors: self.factors.clone(),
			input_id: self.input_id.clone(),
			output_id: self.output_id.clone(),
			forward_id:graph.add_pass(CollapsePass::new(
				self.input_id.value_id(),
				self.output_id.value_id(),
				self.factors.clone(),
			)),
			backward_id:graph.add_pass(ExpandPass::new(
				self.output_id.gradient_id(),
				self.input_id.gradient_id(),
				self.factors.clone(),
			)),
		})
	}
}

#[derive(Debug, Clone)]
pub struct CollapseInstance {
	name: String,
	factors: Vec<usize>,
	input_id: NodeID,
	output_id: NodeID,
	forward_id: PassID,
	backward_id: PassID,
}

impl OpInstance for CollapseInstance {

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

		ensure!(input_shape.ndim() == self.factors.len() + 1, "collapse factors must be the same length as input shape length minus 1");

		let out_channels: usize = input_shape[input_shape.ndim()-1] * self.factors.iter().product::<usize>();
		let output_shape: NodeShape = input_shape.slice().iter().zip(&self.factors).map(|(dim, f)| {
				(dim + f-1)/f
			}).chain(iter::once(out_channels)).into();

		shapes.merge_with(&self.output_id, &output_shape)?;
		Ok(())
	}
}


/// Expand outer dimensions, shuffling entries out of the channel dimension
///
/// Increase size of higher dimensions by given factors by mapping from chunks of the channel dimension to each new spaxel. 
/// Output channel dimension is reduced by the product of the expansion factors. Inverse operation of Collapse.
/// Used in sub-pixel convolution.
#[derive(Clone)] 
pub struct Expand {
	name: Option<String>,
	factors: Vec<usize>,
	input_id: NodeID,
	output_id: NodeID,
}

impl Expand {
	pub fn new(input_id: &NodeID, output_id: &NodeID, factors: &[usize]) -> Self{
		Expand {
			name: None,
			input_id: input_id.clone(),
			output_id: output_id.clone(),
			factors: factors.to_vec(),
		}
	}
}

impl Op for Expand {
	type InstanceType = ExpandInstance;

	fn type_name(&self) -> &'static str {
		"Expand"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		let name = standard_op_name(&self, &self.name, graph, &[self.input_id.clone()], &[self.output_id.clone()]);

		Ok(ExpandInstance{
			name: name,
			factors: self.factors.clone(),
			input_id: self.input_id.clone(),
			output_id: self.output_id.clone(),
			forward_id:graph.add_pass(ExpandPass::new(
				self.input_id.value_id(),
				self.output_id.value_id(),
				self.factors.clone(),
			)),
			backward_id:graph.add_pass(CollapsePass::new(
				self.output_id.gradient_id(),
				self.input_id.gradient_id(),
				self.factors.clone(),
			)),
		})
	}
}

#[derive(Debug, Clone)]
pub struct ExpandInstance {
	name: String,
	factors: Vec<usize>,
	input_id: NodeID,
	output_id: NodeID,
	forward_id: PassID,
	backward_id: PassID,
}

impl OpInstance for ExpandInstance {
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

		ensure!(input_shape.ndim() == self.factors.len() + 1, "Expansion factors must be the same length as input shape length minus 1");
		ensure!(input_shape[input_shape.ndim()-1]%self.factors.iter().product::<usize>() == 0, "Input channel dimension must be evenly divisible by the product of all factors");
		
		let out_channels: usize = input_shape[input_shape.ndim()-1]/self.factors.iter().product::<usize>();
		let output_shape: NodeShape = input_shape.slice().iter().zip(&self.factors).map(|(dim, f)| {
				((dim-1)*f + 1, dim * f)
			}).chain(iter::once((out_channels, out_channels))).into();

		shapes.merge_with(&self.output_id, &output_shape)?;
		Ok(())
	}
}

#[derive(Debug, Clone)]
struct CollapsePass {
	input_id: DataID,
	output_id: DataID,
	factors: Vec<usize>,
	batch_end: usize,
}

impl CollapsePass {
	pub fn new(input_id: DataID, output_id: DataID, factors: Vec<usize>) -> Self{
		let batch_end = factors.iter().take_while(|&&i| i == 1).count();
		CollapsePass {
			input_id,
			output_id,
			factors,
			batch_end,
		}
	}
}

impl Pass for CollapsePass {
	fn type_name(&self) -> &'static str {"CollapsePass"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input_id.clone()],
		vec![self.output_id.clone()])
	}

	fn run(&self, data: &Storage) -> Result<Box<Any>> {
		let input = data.get(&self.input_id)?;
		let mut output = data.get_mut(&self.output_id)?;

		let input_shape = input.shape();
		let output_shape = output.shape().to_vec();
		
		let input_channels = input_shape[input_shape.len()-1];
		let output_channels = output_shape[output_shape.len()-1];

		let input_spatial = &input_shape[self.batch_end..input_shape.len()-1];
		let output_spatial = &output_shape[self.batch_end..output_shape.len()-1];
		let factors_spatial = &self.factors[self.batch_end..];

		ensure!(input_shape.len() == output_shape.len(), "Input ndims does not match output ndims");
		ensure!(input_shape.len() == self.factors.len() + 1, "factors must be the same length as input dimensions minus 1");
		ensure!(input_shape.iter().zip(&self.factors).map(|(dim, f)| (dim + f-1)/f).eq(output_shape[..output_shape.len()-1].iter().cloned()), "input shape and factors incompatible with output shape");
		ensure!(output_channels == input_channels * self.factors.iter().product::<usize>(), "The ratio of output channels over input channels must equal the product of all factors");

		let input = input.as_slice().unwrap();
		let output = output.as_slice_mut().unwrap();

		let batches = input_shape[..self.batch_end].iter().product();

		let in_size = input.len()/batches;
		let out_size = output.len()/batches;

		for b_ind in  0..batches{
			let out_batch = &mut output[b_ind*out_size..][..out_size];
			let in_batch = &input[b_ind*in_size..][..in_size];

			for (i, patch) in out_batch.chunks_mut(output_channels).enumerate(){
				let output_ind = i*output_channels;
				collapse_recurse(patch, &in_batch, &factors_spatial, input_channels, &input_spatial, &output_spatial, 0, output_ind, out_size);
			}
		}

		Ok(Box::new(()))
	}
}

#[derive(Debug, Clone)]
struct ExpandPass {
	input_id: DataID,
	output_id: DataID,
	factors: Vec<usize>,
	batch_end: usize,
}

impl ExpandPass {
	pub fn new(input_id: DataID, output_id: DataID, factors: Vec<usize>) -> Self{
		let batch_end = factors.iter().take_while(|&&i| i == 1).count();
		ExpandPass {
			input_id,
			output_id,
			factors,
			batch_end,
		}
	}
}

impl Pass for ExpandPass {
	fn type_name(&self) -> &'static str {"ExpandPass"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input_id.clone()],
		vec![self.output_id.clone()])
	}

	fn run(&self, data: &Storage) -> Result<Box<Any>> {
		let input = data.get(&self.input_id)?;
		let mut output = data.get_mut(&self.output_id)?;

		let input_shape = input.shape();
		let output_shape = output.shape().to_vec();
		
		let input_channels = input_shape[input_shape.len()-1];
		let output_channels = output_shape[output_shape.len()-1];

		let input_spatial = &input_shape[self.batch_end..input_shape.len()-1];
		let output_spatial = &output_shape[self.batch_end..output_shape.len()-1];
		let factors_spatial = &self.factors[self.batch_end..];

		ensure!(input_shape.len() == output_shape.len(), "Input ndims does not match output ndims");
		ensure!(input_shape.len() == self.factors.len() + 1, "factors must be the same length as input dimensions minus 1");
		ensure!(output_shape.iter().zip(&self.factors).map(|(dim, f)| (dim + f-1)/f).eq(input_shape[..input_shape.len()-1].iter().cloned()), "input shape and factors incompatible with output shape");
		ensure!(input_channels == output_channels * self.factors.iter().product::<usize>(), "The ratio of input channels over output channels must equal the product of all factors");

		let input = input.as_slice().unwrap();
		let output = output.as_slice_mut().unwrap();

		let batches = input_shape[..self.batch_end].iter().product();

		let in_size = input.len()/batches;
		let out_size = output.len()/batches;

		for b_ind in  0..batches {
			let out_batch = &mut output[b_ind*out_size..][..out_size];
			let in_batch = &input[b_ind*in_size..][..in_size];
			
			for (i, patch) in in_batch.chunks(input_channels).enumerate(){
				let input_ind = i*input_channels;
				expand_recurse(patch, out_batch, &factors_spatial, output_channels, &output_spatial, &input_spatial, 0, input_ind, in_size)
			}
		}

		Ok(Box::new(()))
	}
}


fn collapse_recurse(patch: &mut [f32], input: &[f32], factors_spatial:&[usize], in_channels: usize, input_spatial: &[usize], output_spatial: &[usize], axis: usize, output_ind: usize, old_out_stride: usize){
	
	// stride in array index, not spaxel index
	let out_stride = old_out_stride/output_spatial[axis];
	let in_stride = input.len()/input_spatial[axis];
	let patch_stride = patch.len()/factors_spatial[axis];
	
	// coordinates of the centre spaxel of the kernel, for the current axis
	let ox = (output_ind % old_out_stride)/out_stride;

	// valid range of input coordinates the current axis for the current patch
	let start = ox * factors_spatial[axis];
	let end = min(start + factors_spatial[axis], input_spatial[axis]);
	
	if axis < factors_spatial.len() - 1 {
		
		for i in start..end{
			let new_input = &input[in_stride*i..in_stride*(i+1)];
			
			let i_patch = i - start;
			let new_patch = &mut patch[i_patch*patch_stride..(i_patch+1)*patch_stride];
			let new_axis = axis+1;

			collapse_recurse(new_patch, new_input, factors_spatial, in_channels, input_spatial, output_spatial, new_axis, output_ind, out_stride);
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


fn expand_recurse(patch: &[f32], output: &mut [f32], factors_spatial:&[usize], out_channels: usize, output_spatial: &[usize], input_spatial: &[usize], axis: usize, input_ind: usize, old_in_stride: usize){
	
	// stride in array index, not spaxel index
	let in_stride = old_in_stride/input_spatial[axis];
	let out_stride = output.len()/output_spatial[axis];
	let patch_stride = patch.len()/factors_spatial[axis];
	
	// coordinates of the centre spaxel of the kernel, for the current axis
	let ix = (input_ind % old_in_stride)/in_stride;

	// valid range of input coordinates the current axis for the current patch
	let start = ix * factors_spatial[axis];
	let end = min(start + factors_spatial[axis], output_spatial[axis]);
	
	if axis < factors_spatial.len() - 1 {
		
		for i in start..end{
			let new_output = &mut output[out_stride*i..out_stride*(i+1)];
			
			let i_patch = i - start;
			let new_patch = &patch[i_patch*patch_stride..(i_patch+1)*patch_stride];
			let new_axis = axis+1;

			expand_recurse(new_patch, new_output, factors_spatial, out_channels, output_spatial, input_spatial, new_axis, input_ind, in_stride);
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




#[test]
fn test_collapse_backprop(){
	_test_collapse_backprop().unwrap();
}

fn _test_collapse_backprop() -> Result<()> {
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use new::ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![2, 20, 9, 5], "input", tag![])?;
	let node2 = g.new_node(shape![Unknown, Unknown, Unknown, 75], "collapse", tag![])?;
	let node3 = g.new_node(shape![2, 4, 3, 75], "target", tag![])?;
		
	let _o1 = g.new_op(Collapse::new(&node1, &node2, &[1, 5, 3]), tag![])?;
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
fn test_expand_backprop(){
	_test_expand_backprop().unwrap();
}

fn _test_expand_backprop() -> Result<()> {
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use new::ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![2, 4, 3, 75], "input", tag![])?;
	let node2 = g.new_node(shape![Unknown, Unknown, Unknown, Unknown], "expand", tag![])?;
	let node3 = g.new_node(shape![2, 16, 7, 5], "target", tag![])?;

	// let node2 = g.new_node(shape![2, 20, 9, 5], "expand", tag![])?;
	// let node3 = g.new_node(shape![2, 20, 9, 5], "target", tag![])?;
		
	let _o1 = g.new_op(Expand::new(&node1, &node2, &[1, 5, 3]), tag![])?;
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
fn test_collapse_recurse(){
	_test_collapse_recurse();
}

fn _test_collapse_recurse(){
	
	let factors = vec![5, 3];
	let input_shape = vec![7, 5];
	let output_shape: Vec<usize> = input_shape.iter().zip(factors.iter()).map(|(i, f)| (i+f-1)/f).collect();
	let in_channels = 2;
	let out_channels = factors.iter().fold(in_channels, |p, v| p*v);
	
	let input_spaxel_count = input_shape.iter().fold(1, |p, v| p*v);
	let output_spaxel_count = output_shape.iter().fold(1, |p, v| p*v);
	
	let input_size = input_spaxel_count*in_channels;
	let output_size = output_spaxel_count*out_channels;
	let patch_size = factors.iter().fold(in_channels, |p, v| p*v);


	let input: Vec<f32> = (0..input_size).map(|x| x as f32).collect();		
			
	
	let axis = 0;
	
	let mut output = vec![0.0; patch_size*output_spaxel_count];
	
	for (i, patch) in output.chunks_mut(patch_size).enumerate(){
		let output_ind = i*out_channels;
		collapse_recurse(patch, &input, &factors, in_channels, &input_shape, &output_shape, axis, output_ind, output_size);
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
	_test_expand_recurse();
}

fn _test_expand_recurse(){
	
	let factors = vec![5, 3];
	let input_shape = vec![2, 2];
	let output_shape = vec![7, 5];
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
			
	
	let axis = 0;
	
	let mut output = vec![0.0; output_size];
	
	
	for (i, patch) in input.chunks(in_channels).enumerate(){
		let input_ind = i*in_channels;

		expand_recurse(patch, &mut output, &factors, out_channels, &output_shape, &input_shape, axis, input_ind, input_size)
	}
	
	let target: Vec<f32> = (0..output_size).map(|x| x as f32).collect();

	assert_eq!(output, target);
}


#[test]
fn test_expand_recurse_full(){
	_test_expand_recurse_full();
}

fn _test_expand_recurse_full(){
	
	

	let factors = vec![5, 3];
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
			
	
	let axis = 0;
	
	let mut output = vec![0.0; output_size];
	
	
	for (i, patch) in input.chunks(in_channels).enumerate(){
		let input_ind = i*in_channels;

		expand_recurse(patch, &mut output, &factors, out_channels, &output_shape, &input_shape, axis, input_ind, input_size)
	}
	
	let target: Vec<f32> = (0..output_size).map(|x| x as f32).collect();

	assert!(output == target, format!("target{:?} \n output{:?}", target, output));
}