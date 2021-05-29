use crate::{ConcatComponents, DataSet};
use byteorder::{BigEndian, ByteOrder};
use ndarray::{IxDyn, ArcArray};
use std::{fs::File, io::Read, path::Path};

/// A `DataSet` created from both mnist binary format image and label files.
///
/// Each element contains two components, a (typically) 28x28 image and a one-hot label encoding the category out of 10
pub struct Mnist {
	data: ConcatComponents<MnistImages, MnistLabels>,
}

impl Mnist {
	pub fn new(image_file: File, label_file: File) -> Self {
		let joined = MnistImages::new(image_file).concat_components(MnistLabels::new(label_file));
		Mnist { data: joined }
	}

	pub fn training<P: AsRef<Path>>(folder_path: P) -> Self {
		let folder_path = folder_path.as_ref();
		let err = "Pass a folder containing 'train-images.idx3-ubyte' and 'train-labels.idx1-ubyte'";
		assert!(folder_path.is_dir(), "{}", err);
		let label_file = File::open(folder_path.join("train-labels.idx1-ubyte").as_path()).expect(err);
		let image_file = File::open(folder_path.join("train-images.idx3-ubyte").as_path()).expect(err);
		Mnist::new(image_file, label_file)
	}

	pub fn testing<P: AsRef<Path>>(folder_path: P) -> Self {
		let folder_path = folder_path.as_ref();
		let err = "Pass a folder containing 't10k-images.idx3-ubyte' and 't10k-labels.idx1-ubyte'";
		assert!(folder_path.is_dir(),"{}", err);
		let label_file = File::open(folder_path.join("t10k-labels.idx1-ubyte").as_path()).expect(err);
		let image_file = File::open(folder_path.join("t10k-images.idx3-ubyte").as_path()).expect(err);
		Mnist::new(image_file, label_file)
	}
}

impl DataSet for Mnist {
	fn get(&mut self, i: usize) -> Vec<ArcArray<f32, IxDyn>> {
		self.data.get(i)
	}

	fn length(&self) -> usize {
		self.data.length()
	}

	fn width(&self) -> usize {
		self.data.width()
	}

	fn components(&self) -> Vec<String> {
		self.data.components()
	}
}

/// A `DataSet` created from a mnist binary format image file.
///
/// Each element contains only one component, a typically [28, 28, 1] array with values in the range (0,1)
pub struct MnistImages {
	shape: Vec<usize>,
	data: Vec<Vec<u8>>,
}

impl MnistImages {
	pub fn new(image_file: File) -> Self {
		let (shape, data) = read_image_file(image_file);
		MnistImages { shape, data }
	}
}

impl DataSet for MnistImages {
	fn get(&mut self, i: usize) -> Vec<ArcArray<f32, IxDyn>> {
		let image = ArcArray::<f32, IxDyn>::from_shape_vec(
			self.shape.as_slice(),
			self.data[i].iter().map(|&v| f32::from(v) / 255.0).collect(),
		)
		.unwrap();
		vec![image]
	}

	fn length(&self) -> usize {
		self.data.len()
	}

	fn width(&self) -> usize {
		1
	}

	fn components(&self) -> Vec<String> {
		vec!["Images".to_string()]
	}
}

fn read_image_file(mut file: File) -> (Vec<usize>, Vec<Vec<u8>>) {
	let mut buf: Vec<u8> = vec![0; file.metadata().unwrap().len() as usize];
	file.read_exact(&mut buf).unwrap();
	let mut i: usize = 0;

	assert!(BigEndian::read_u32(&buf[i..]) == 2051);
	i += 4;

	let n = BigEndian::read_u32(&buf[i..]) as usize;
	i += 4;
	let h = BigEndian::read_u32(&buf[i..]) as usize;
	i += 4;
	let w = BigEndian::read_u32(&buf[i..]) as usize;
	i += 4;

	let mut v1 = Vec::with_capacity(n);
	for _ in 0..n {
		let mut v2 = Vec::with_capacity(h * w);
		for _ in 0..h * w {
			v2.push(buf[i]);
			i += 1;
		}
		v1.push(v2);
	}

	(vec![w, h, 1], v1)
}

/// A `DataSet` created from a mnist binary format label file.
///
/// Each element contains only one component, a typically [10] one-hot array encoding the category of the associated
/// image element
pub struct MnistLabels {
	labels: Vec<u8>,
}

impl MnistLabels {
	pub fn new(label_file: File) -> Self {
		MnistLabels {
			labels: read_label_file(label_file),
		}
	}
}

impl DataSet for MnistLabels {
	fn get(&mut self, i: usize) -> Vec<ArcArray<f32, IxDyn>> {
		let mut one_hot_label = ArcArray::zeros(IxDyn(&[10]));
		one_hot_label[self.labels[i] as usize] = 1.0;
		vec![one_hot_label]
	}

	fn length(&self) -> usize {
		self.labels.len()
	}

	fn width(&self) -> usize {
		1
	}

	fn components(&self) -> Vec<String> {
		vec!["Labels".to_string()]
	}
}

fn read_label_file(mut file: File) -> Vec<u8> {
	let mut buf: Vec<u8> = vec![0; file.metadata().unwrap().len() as usize];
	file.read_exact(&mut buf).unwrap();
	let mut i: usize = 0;

	assert!(BigEndian::read_u32(&buf[i..]) == 2049);
	i += 4;

	let n = BigEndian::read_u32(&buf[i..]) as usize;
	i += 4;

	let mut v1 = Vec::with_capacity(n);
	for _ in 0..n {
		v1.push(buf[i]);
		i += 1;
	}

	v1
}
