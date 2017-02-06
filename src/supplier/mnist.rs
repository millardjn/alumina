extern crate byteorder;
extern crate rand;

use self::byteorder::*;

use std::fs::*;
use std::path::Path;
use std::io::Read;

use shape::*;
use graph::*;
use supplier::Supplier;
use supplier::Selector;

pub struct MnistSupplier<S: Selector>{
	count: u64,
	labels: Vec<u8>,
	images: Vec<Vec<u8>>,
	shape: NodeShape,
	order: S,
}

impl<S: Selector> Supplier for MnistSupplier<S>{
	
	fn next_n(&mut self, n: usize) -> (Vec<NodeData>, Vec<NodeData>){
		assert!(n > 0, "n must be larger than 0");
		
		let (mut input, mut train) = self.get();
		
		for _ in 1..n{
			let (input2, train2) = self.get();
			input = input.join(input2);
			train = train.join(train2);
		}
		(vec![input], vec![train])
	}
	
	fn epoch_size(&self) -> usize{
		self.labels.len()
	}
	fn samples_taken(&self) -> u64{
		self.count
	}
	fn reset(&mut self){
		self.order.reset();
		self.count = 0;
	}
}

impl<S: Selector> MnistSupplier<S>{
	
	
	fn get(&mut self) -> (NodeData, NodeData){
		
		let ind = self.order.next();

		let mut one_hot_label = vec![0.0; 10];
		one_hot_label[self.labels[ind] as usize] = 1.0;
		self.count += 1;
		let n = 1;
		(
			NodeData::new(self.shape.to_data_shape(n).unwrap(), self.images[ind].iter().map(|&i| i as f32/255.0).collect()), 
			NodeData::new(DataShape::new_flat(10, n), one_hot_label)
		)

	}
	
	pub fn training(folder_path: &Path) -> MnistSupplier<S> {
		let err = "Pass a folder containing 'train-images.idx3-ubyte' and 'train-labels.idx1-ubyte'";
		assert!(folder_path.is_dir(), err);
		let label_file = File::open(folder_path.join("train-labels.idx1-ubyte").as_path()).expect(err);
		let image_file = File::open(folder_path.join("train-images.idx3-ubyte").as_path()).expect(err);
		MnistSupplier::create(image_file, label_file)
	}
	
	pub fn testing(folder_path: &Path) -> MnistSupplier<S> {
		let err = "Pass a folder containing 't10k-images.idx3-ubyte' and 't10k-labels.idx1-ubyte'";
		assert!(folder_path.is_dir(), err);
		let label_file = File::open(folder_path.join("t10k-labels.idx1-ubyte").as_path()).expect(err);
		let image_file = File::open(folder_path.join("t10k-images.idx3-ubyte").as_path()).expect(err);
		MnistSupplier::create(image_file, label_file)
	}
	
	fn create(image_file: File, label_file: File) -> MnistSupplier<S>  {
		let labels = Self::read_label_file(label_file);
		let n = labels.len();
		let (shape, images) = Self::read_image_file(image_file);
		assert!(images.len() == labels.len());

		MnistSupplier{
			count: 0,
			labels: labels,
			images: images,
			shape: shape,
			order: S::new(n),
		}
	}
	
	fn read_label_file(mut file: File) -> Vec<u8> {
		let mut buf: Vec<u8> = vec![0; file.metadata().unwrap().len() as usize];
		file.read_exact(&mut buf).unwrap();
		let mut i: usize = 0;
		
		assert!(BigEndian::read_u32(&buf[i..]) == 2049); i += 4;
		
		let n = BigEndian::read_u32(&buf[i..]) as usize; i += 4;
		
		let mut v1 = Vec::with_capacity(n);
		for _ in 0..n {
			v1.push(buf[i]); i += 1;
		}
		
		v1
	}
	
	fn read_image_file(mut file: File) -> (NodeShape, Vec<Vec<u8>>){
		let mut buf: Vec<u8> = vec![0; file.metadata().unwrap().len() as usize];
		file.read_exact(&mut buf).unwrap();
		let mut i: usize = 0;
		
		assert!(BigEndian::read_u32(&buf[i..]) == 2051); i += 4;
		
		let n = BigEndian::read_u32(&buf[i..]) as usize; i += 4;
		let h = BigEndian::read_u32(&buf[i..]) as usize; i += 4;
		let w = BigEndian::read_u32(&buf[i..]) as usize; i += 4;
		
		let mut v1 = Vec::with_capacity(n);
		for _ in 0..n {
			let mut v2 = Vec::with_capacity(h*w);
			for _ in 0..h*w {
				v2.push(buf[i]); i += 1;
			}
			v1.push(v2);
		}
		
		(NodeShape::new(1, &[w, h]), v1)
	}
}