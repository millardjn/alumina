use graph::{GraphDef, GraphShapes, ErrorKind, Result};
use id::{NodeID, DataID, OpID, PassID};
use storage::Storage;
use ops::{standard_op_name, Op, OpInstance, Pass};
use shape::NodeDim;
use smallvec::SmallVec;
use std::any::Any;
use odds;
/// An `Op` which fills the output with the coordinates of the selected dimensions
///
/// The op writes coordinates of selected higher dimensions to the channel dimensions (last axis).
/// The number of output channels is equal to two times the number of coord_axes selected.
/// Each coord axis produces one channel which increases from 0 to 1 with spaxel position in that axis and another which decreases from 1 to 0
#[must_use]
#[derive(Clone, Debug)] 
pub struct Coord {
	output_id: NodeID,
	input: Option<NodeID>,
	coord_axes: SmallVec<[isize; 6]>,
	name: Option<String>,
}

impl Coord {
	pub fn new(output_id: &NodeID, coord_axes: &[isize]) -> Self {
		Coord {
			output_id: output_id.clone(),
			input: None,
			coord_axes: coord_axes.iter().cloned().collect(),
			name: None,
		}
	}

	/// If set this `Op` will propagate all dimensions, except the last, from the input to the output.
	///
	/// Default: None.
	pub fn input(mut self, input: &NodeID) -> Self {
		self.input = Some(input.clone());
		self
	}
}

impl Op for Coord {
	type InstanceType = CoordInstance;

	fn type_name(&self) -> &'static str {
		"Coord"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef) -> Result<Self::InstanceType> {

		let name = if let Some(ref input_id) = self.input {
			standard_op_name(&self, &self.name, graph, &[input_id.clone()], &[self.output_id.clone()])
		} else {
			standard_op_name(&self, &self.name, graph, &[], &[self.output_id.clone()])
		};

		Ok(CoordInstance{
			name: name,
			output_id: self.output_id.clone(),
			input: self.input,
			forward_id: graph.add_pass(CoordForward::new(
				self.output_id.clone(),
				self.coord_axes.clone())),
			coord_axes: self.coord_axes.clone(),
		})
	}
}


#[derive(Clone, Debug)] 
pub struct CoordInstance {
	name: String,
	output_id: NodeID,
	input: Option<NodeID>,
	forward_id: PassID,
	coord_axes: SmallVec<[isize; 6]>,
}

impl OpInstance for CoordInstance {

	fn name(&self) -> &str {&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		if let Some(ref input_id) = self.input {
			(vec![input_id.clone()], vec![self.output_id.clone()])
		} else {
			(vec![], vec![self.output_id.clone()])
		}
	}

	fn inner_passes(&self) -> Vec<PassID> {
		vec![self.forward_id.clone()]
	}

	fn inner_ops(&self) -> Vec<OpID> {
		vec![]
	}

	fn inner_nodes(&self) -> Vec<NodeID> {
		vec![]
	}

	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{
		if let Some(ref input_id) = self.input {
			let mut input_shape = shapes.get_shape(input_id).clone();
			let mask = coord_mask(input_shape.ndim(), self.coord_axes.as_slice());
			let output_channels = mask.iter().filter(|&&b|b).count() * 2;
			let channel_axis = input_shape.ndim()-1;
			input_shape.dimensions_mut()[channel_axis] = NodeDim::Known(output_channels);
			shapes.merge_with(&self.output_id, &input_shape)?;
		} else {
			let mut output_shape = shapes.get_output_shape(&self.output_id).clone();
			let mask = coord_mask(output_shape.ndim(), self.coord_axes.as_slice());
			let output_channels = mask.iter().filter(|&&b|b).count() * 2;
			let channel_axis = output_shape.ndim()-1;
			output_shape.dimensions_mut()[channel_axis] = NodeDim::Known(output_channels);
			shapes.merge_with(&self.output_id, &output_shape)?;
		}
		Ok(())
	}
}

/// Returns a mask indicating whether an axis should be reduced based on the axes list
fn coord_mask(len: usize, axes: &[isize]) -> SmallVec<[bool; 6]> {
	let mut reduce = SmallVec::with_capacity(len);
	for _ in 0..len {
		reduce.push(false);
	}
	for axis in axes {
		reduce[(axis + len as isize) as usize % len] = true;
	}
	reduce[len - 1] = false;
	reduce
}

#[derive(Clone, Debug)]
struct CoordForward {
	output_id: NodeID,
	coord_axes: SmallVec<[isize; 6]>,
}

impl CoordForward {
	pub fn new(output_id: NodeID, coord_axes: SmallVec<[isize; 6]>) -> Self {
		CoordForward {
			output_id,
			coord_axes,
		}
	}
}

impl Pass for CoordForward {
	fn type_name(&self) -> &'static str {"CoordForward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![], vec![self.output_id.value_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let mut output = data.get_mut(&self.output_id.value_id())?;

		let output_shape: SmallVec<[usize; 6]> = output.shape().iter().cloned().collect();

		let mask = coord_mask(output_shape.len(), self.coord_axes.as_slice());
		let output_channels = mask.iter().filter(|&&b|b).count() * 2;

		ensure!(
			output_shape[output_shape.len()-1] == output_channels,
			ErrorKind::PassError(self.name(), format!("output shape channel(last) dimension: {:?} did not match {:?}", output_shape.as_slice(), output_channels))
		);

		let output_slice = output.as_slice_mut().unwrap();

		let mut channels: SmallVec<[f32; 6]> = (0..output_channels).map(|_| 0.0).collect();

		unsafe{fill_coord(0, &mask, output_shape.as_slice(), 0, channels.as_mut_slice(), output_slice)};

		Ok(Box::new(()))
	}
}


unsafe fn fill_coord(axis: usize, mask: &[bool], shape: &[usize], channel_ind: usize, channels: &mut[f32], slice: &mut [f32]) {

	if channel_ind >= channels.len() {
		//println!("{}   {}", slice.len(), channels.len());
		for i in 0..slice.len()/channels.len() {
			for j in 0..channels.len() {
				*odds::get_unchecked_mut(slice, i*channels.len() + j) += *odds::get_unchecked(channels, j);
			}
		}
		return;
	}

	if mask[axis] && shape[axis] > 0 {
		let mut val = 0.0;
		let inc = 1.0/(shape[axis] - 1) as f32;
		for i in 0..shape[axis]{
			//println!("true  a{} i{}", axis, i);
			channels[channel_ind] = val;
			channels[channel_ind + 1] = 1.0 - val;

			let stride = slice.len()/shape[axis];
			let new_slice = odds::slice_unchecked_mut(slice, i*stride, (i+1)*stride);
			fill_coord(axis + 1, mask, shape, channel_ind + 2, channels, new_slice);
			val += inc;
		}
	} else if shape[axis] > 0 {
		for i in 0..shape[axis]{
			//println!("false a{} i{}", axis, i);
			let stride = slice.len()/shape[axis];
			let new_slice = odds::slice_unchecked_mut(slice, i*stride, (i+1)*stride);
			fill_coord(axis + 1, mask, shape, channel_ind, channels, new_slice);
		}
	}
}



#[test]
fn test_coord_input(){
	_coord_input().unwrap();
}

fn _coord_input() -> Result<()>{
	use graph::{GraphDef, Dependencies};
	use ndarray::{ArrayD, Axis, IxDyn};

	let mut g = GraphDef::new();

	let input = g.new_node(shape![7, 5, 13], "input", tag![])?;
	let output = g.new_node(shape![Unknown, Unknown, 4], "output", tag![])?;
	let _o1 = g.new_op(Coord::new(&output, &[0, 1]).input(&input), tag![])?;

	let deps = Dependencies::new(&g);
	println!("{:?}", deps);

	let mut subgraph = g.subgraph(&[input.value_id()], &[output.value_id()])?;

	let storage = subgraph.execute(vec![ArrayD::zeros(IxDyn(&[7, 5, 13]))])?;

	let out = storage.get_mut(&output.value_id())?;

	for x in out.axis_iter(Axis(2)) {
		println!("{:?}", x);
	}

	// let out_slice = out.as_slice().unwrap();

	// let expected = vec![
	// 	0.0, 0.0,   0.0,   0.0,   0.0,   0.0,   0.0, 0.0, 0.0,
	// 	0.0, 0.0,   0.0,   0.0,   0.0,   0.0,   0.0, 0.0, 0.0,
	// 	0.0, 0.0, 1./9., 2./9., 3./9., 2./9., 1./9., 0.0, 0.0, 
	// 	0.0, 0.0, 2./9., 4./9., 6./9., 4./9., 2./9., 0.0, 0.0, 
	// 	0.0, 0.0, 3./9., 6./9.,	  1.0, 6./9., 3./9., 0.0, 0.0, 
	// 	0.0, 0.0, 2./9., 4./9., 6./9., 4./9., 2./9., 0.0, 0.0, 
	// 	0.0, 0.0, 1./9., 2./9., 3./9., 2./9., 1./9., 0.0, 0.0, 
	// 	0.0, 0.0,   0.0,   0.0,   0.0,   0.0,   0.0, 0.0, 0.0,
	// 	0.0, 0.0,   0.0,   0.0,   0.0,   0.0,   0.0, 0.0, 0.0,
	// ];

	Ok(())
}

#[test]
fn test_coord(){
	_coord().unwrap();
}

fn _coord() -> Result<()>{
	use graph::{GraphDef, Dependencies};
	use ndarray::{Axis};

	let mut g = GraphDef::new();

	let output = g.new_node(shape![Unknown, Unknown, 4], "output", tag![])?;
	let _o1 = g.new_op(Coord::new(&output, &[0, 1]), tag![])?;

	let deps = Dependencies::new(&g);
	println!("{:?}", deps);

	let mut subgraph = g.subgraph(&[], &[output.value_id()])?;

	let storage = subgraph.execute(vec![])?;

	let out = storage.get_mut(&output.value_id())?;

	for x in out.axis_iter(Axis(2)) {
		println!("{:?}", x);
	}

	// let out_slice = out.as_slice().unwrap();

	// let expected = vec![
	// 	0.0, 0.0,   0.0,   0.0,   0.0,   0.0,   0.0, 0.0, 0.0,
	// 	0.0, 0.0,   0.0,   0.0,   0.0,   0.0,   0.0, 0.0, 0.0,
	// 	0.0, 0.0, 1./9., 2./9., 3./9., 2./9., 1./9., 0.0, 0.0, 
	// 	0.0, 0.0, 2./9., 4./9., 6./9., 4./9., 2./9., 0.0, 0.0, 
	// 	0.0, 0.0, 3./9., 6./9.,	  1.0, 6./9., 3./9., 0.0, 0.0, 
	// 	0.0, 0.0, 2./9., 4./9., 6./9., 4./9., 2./9., 0.0, 0.0, 
	// 	0.0, 0.0, 1./9., 2./9., 3./9., 2./9., 1./9., 0.0, 0.0, 
	// 	0.0, 0.0,   0.0,   0.0,   0.0,   0.0,   0.0, 0.0, 0.0,
	// 	0.0, 0.0,   0.0,   0.0,   0.0,   0.0,   0.0, 0.0, 0.0,
	// ];

	Ok(())
}