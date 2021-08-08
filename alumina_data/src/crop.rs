use crate::DataSet;
use indexmap::{indexmap, IndexMap};
use ndarray::{ArcArray, IxDyn, Slice, SliceInfo, SliceInfoElem};
use rand::{thread_rng, Rng};
use smallvec::SmallVec;
use std::{convert::TryFrom, mem};

#[derive(Copy, Clone)]
pub enum Cropping {
	Centre,
	Random,
}

/// For one component in each element of the dataset: apply a function.
///
/// Renaming the component is optional.
pub struct Crop<S: DataSet> {
	set: S,
	fill: IndexMap<usize, f32>,
	crops: IndexMap<usize, (Vec<usize>, Cropping)>,
}

impl<S: DataSet> Crop<S> {
	/// Crop the given component to the given shape. `shape` must have the same dimensionality as the component.
	pub fn new(set: S, component: usize, shape: &[usize], cropping: Cropping) -> Self {
		let mut crops = indexmap![];
		crops.insert(component, (shape.to_vec(), cropping));
		Crop {
			set,
			fill: indexmap![],
			crops,
		}
	}

	/// Crop another component
	pub fn and_crop(mut self, component: usize, shape: &[usize], cropping: Cropping) -> Self {
		self.crops.insert(component, (shape.to_vec(), cropping));
		self
	}

	/// Set what should be used to fill areas where the crop dimension is larger than the input dimension.
	///
	/// Default: 0.0
	pub fn fill(mut self, component: usize, fill: f32) -> Self {
		self.fill.insert(component, fill);
		self
	}

	/// Borrows the wrapped dataset.
	pub fn inner(&self) -> &S {
		&self.set
	}

	/// Returns the wrapped dataset.
	pub fn into_inner(self) -> S {
		let Self { set, .. } = self;
		set
	}
}

impl<S: DataSet> DataSet for Crop<S> {
	fn get(&mut self, i: usize) -> Vec<ArcArray<f32, IxDyn>> {
		let mut data = self.set.get(i);

		for (&component, &(ref shape, ref cropping)) in self.crops.iter() {
			let arr = mem::replace(&mut data[component], ArcArray::zeros(IxDyn(&[])));
			let fill = self.fill.get(&component).cloned().unwrap_or(0.0);
			data[component] = crop(arr, shape, *cropping, fill);
			//mem::replace(&mut data[component], crop(arr, shape, *cropping, fill));
		}

		data
	}

	fn length(&self) -> usize {
		self.set.length()
	}

	fn width(&self) -> usize {
		self.set.width()
	}

	fn components(&self) -> Vec<String> {
		self.set.components()
	}
}

fn crop(arr: ArcArray<f32, IxDyn>, crop_shape: &[usize], cropping: Cropping, fill: f32) -> ArcArray<f32, IxDyn> {
	assert_eq!(crop_shape.len(), arr.ndim());

	let mut out_arr = ArcArray::from_elem(IxDyn(crop_shape), fill);

	let mut input_slice_arg: SmallVec<[SliceInfoElem; 6]> = SmallVec::new();
	let mut output_slice_arg: SmallVec<[SliceInfoElem; 6]> = SmallVec::new();
	for (&input_width, &output_width) in arr.shape().iter().zip(crop_shape) {
		let (in_si, out_si) = range(cropping, input_width as isize, output_width as isize);
		input_slice_arg.push(in_si.into());
		output_slice_arg.push(out_si.into());
	}

	{
		let in_si: SliceInfo<&[SliceInfoElem], IxDyn, IxDyn> = SliceInfo::try_from(input_slice_arg.as_slice()).unwrap();
		let out_si: SliceInfo<&[SliceInfoElem], IxDyn, IxDyn> =
			SliceInfo::try_from(output_slice_arg.as_slice()).unwrap();

		let in_slice = arr.slice(in_si);
		let mut out_slice = out_arr.slice_mut(out_si);
		out_slice.assign(&in_slice);
	}

	out_arr
}

// returns Si for input and output
fn range(cropping: Cropping, input_width: isize, output_width: isize) -> (Slice, Slice) {
	match cropping {
		Cropping::Centre { .. } => {
			if input_width < output_width {
				let width = input_width;
				let output_start = (input_width - output_width) / 2;
				(
					Slice::new(0, Some(width), 1),
					Slice::new(output_start, Some(output_start + width), 1),
				)
			} else {
				let width = output_width;
				let input_start = (output_width - input_width) / 2;
				(
					Slice::new(input_start, Some(input_start + width), 1),
					Slice::new(0, Some(width), 1),
				)
			}
		}

		Cropping::Random { .. } => {
			if input_width < output_width {
				let width = input_width;
				let output_start = thread_rng().gen_range(0..output_width - input_width + 1);
				(
					Slice::new(0, Some(width), 1),
					Slice::new(output_start, Some(output_start + width), 1),
				)
			} else {
				let width = output_width;
				let input_start = thread_rng().gen_range(0..input_width - output_width + 1);
				(
					Slice::new(input_start, Some(input_start + width), 1),
					Slice::new(0, Some(width), 1),
				)
			}
		}
	}
}
