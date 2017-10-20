use ndarray::{ArrayD, IxDyn};
use std::fs::File;
use std::path::Path;
use std::io::Read;
use new::data::DataSet;

/// A `DataSet` created from cifar10 binary format files.
///
/// Each element contains two components, a (typically) 32x32x3 RGB image and a one-hot label encoding the category out of 10
pub struct Cifar10 {
	images: Vec<Vec<u8>>,
	labels: Vec<u8>,
}

impl Cifar10 {
	pub fn new(files: Vec<File>) -> Self {
		let mut labels = vec![];
		let mut images = vec![];
		for file in files {
			let (next_labels, next_images) = read_cifar10_file(file);
			labels.append(&mut next_labels);
			images.append(&mut next_images);
		}
		Cifar10{labels, images}
	}

	pub fn training(folder_path: &Path) -> Self {
		let err = "Pass a folder containing 'data_batch_1.bin', ..., 'data_batch_5.bin'";
		assert!(folder_path.is_dir(), err);
		let training_file1 = File::open(folder_path.join("data_batch_1.bin").as_path()).expect(err);
		let training_file2 = File::open(folder_path.join("data_batch_2.bin").as_path()).expect(err);
		let training_file3 = File::open(folder_path.join("data_batch_3.bin").as_path()).expect(err);
		let training_file4 = File::open(folder_path.join("data_batch_4.bin").as_path()).expect(err);
		let training_file5 = File::open(folder_path.join("data_batch_5.bin").as_path()).expect(err);
		Cifar10::new(vec![training_file1, training_file2, training_file3, training_file4, training_file5])
	}

	pub fn testing(folder_path: &Path) -> Self {
		let err = "Pass a folder containing 'test_batch.bin'";
		assert!(folder_path.is_dir(), err);
		let test_file = File::open(folder_path.join("test_batch.bin").as_path()).expect(err);
		Cifar10::new(vec![test_file])
	}

	pub fn categories() -> &'static[&'static str] {
		&CIFAR_10[..]
	}
}

impl DataSet for Cifar10 {
	fn get(&mut self, i: usize) -> Vec<ArrayD<f32>> {
		let bytes = self.images[i];
		let mut image_vec = vec![0.0; 3072];
		for i in 0..1024{
			image_vec[i*3 + 0] = bytes[i + 0*1024] as f32/255.0;
			image_vec[i*3 + 1] = bytes[i + 1*1024] as f32/255.0;
			image_vec[i*3 + 2] = bytes[i + 2*1024] as f32/255.0;
		}
		let image = ArrayD::from_shape_vec(&[32, 32, 3][..], image_vec).unwrap();
		let mut one_hot_label = ArrayD::zeros(IxDyn(&[10]));
		one_hot_label[self.labels[i] as usize] = 1.0;
		vec![image, one_hot_label]
	}

	fn length(&self) -> usize {
		self.images.len()
	}

	fn width(&self) -> usize {
		2
	}

	fn components(&self) -> Vec<String> {
		vec!["Images".to_string(), "Labels".to_string()]
	}
}


fn read_cifar10_file(mut file: File) -> (Vec<u8>, Vec<Vec<u8>>){
	let mut buf: Vec<u8> = vec![0; file.metadata().unwrap().len() as usize];
	file.read_exact(&mut buf).unwrap();
	let mut i: usize = 0;
	
	assert!(buf.len() % (1 + 3072));
	let n = buf.len() / (1 + 3072);

	let mut labels = Vec::with_capacity(n);
	let mut images = Vec::with_capacity(n);
	for _ in 0..n {
		labels.push(buf[i]); i += 1;
		let pixels = vec![3072];
		for _ in 0..3072 {
			pixels.push(buf[i]); i += 1;
		}
		images.push(pixels)
	}
	
	(labels, images)
}



/// A `DataSet` created from a cifar100 binary format file.
///
/// Each element contains three components, a (typically) 32x32x3 RGB image and two one-hot label encoding
/// the coarse category out of 20 and the fine category out of 100;
pub struct Cifar100 {
	images: Vec<Vec<u8>>,
	coarse_labels: Vec<u8>,
	fine_labels: Vec<u8>,
}

impl Cifar100 {
	pub fn new(file: File) -> Self {
		let (coarse_labels, fine_labels, images) = read_cifar100_file(file);
		Cifar100{coarse_labels, fine_labels, images}
	}

	pub fn training(folder_path: &Path) -> Self {
		let err = "Pass a folder containing 'train.bin'";
		assert!(folder_path.is_dir(), err);
		let training_file = File::open(folder_path.join("train.bin").as_path()).expect(err);
		Cifar100::new(training_file)
	}

	pub fn testing(folder_path: &Path) -> Self {
		let err = "Pass a folder containing 'test.bin'";
		assert!(folder_path.is_dir(), err);
		let test_file = File::open(folder_path.join("test.bin").as_path()).expect(err);
		Cifar100::new(test_file)
	}

	pub fn coarse_categories() -> &'static[&'static str] {
		&CIFAR_100_COARSE[..]
	}

	pub fn fine_categories() -> &'static[&'static str] {
		&CIFAR_100_FINE[..]
	}
}

impl DataSet for Cifar100 {
	fn get(&mut self, i: usize) -> Vec<ArrayD<f32>>{
		let bytes = self.images[i];
		let mut image_vec = vec![0.0; 3072];
		for i in 0..1024{
			image_vec[i*3 + 0] = bytes[i + 0*1024] as f32/255.0;
			image_vec[i*3 + 1] = bytes[i + 1*1024] as f32/255.0;
			image_vec[i*3 + 2] = bytes[i + 2*1024] as f32/255.0;
		}
		let image = ArrayD::from_shape_vec(&[32, 32, 3][..], image_vec).unwrap();
		let mut coarse_one_hot_label = ArrayD::zeros(IxDyn(&[20]));
		coarse_one_hot_label[self.coarse_labels[i] as usize] = 1.0;
		let mut fine_one_hot_label = ArrayD::zeros(IxDyn(&[100]));
		fine_one_hot_label[self.fine_labels[i] as usize] = 1.0;
		vec![image, coarse_one_hot_label, fine_one_hot_label]
	}

	fn length(&self) -> usize{
		self.images.len()
	}

	fn width(&self) -> usize{
		3
	}

	fn components(&self) -> Vec<String> {
		vec!["Images".to_string(), "Coarse Labels".to_string(), "Fine Labels".to_string()]
	}
}

fn read_cifar100_file(mut file: File) -> (Vec<u8>, Vec<u8>, Vec<Vec<u8>>) {
	let mut buf: Vec<u8> = vec![0; file.metadata().unwrap().len() as usize];
	file.read_exact(&mut buf).unwrap();
	let mut i: usize = 0;
	
	assert!(buf.len() % (2 + 3072));
	let n = buf.len() / (2 + 3072);

	let mut coarse_labels = Vec::with_capacity(n);
	let mut fine_labels = Vec::with_capacity(n);
	let mut images = Vec::with_capacity(n);
	for _ in 0..n {
		coarse_labels.push(buf[i]); i += 1;
		fine_labels.push(buf[i]); i += 1;

		let pixels = vec![3072];
		for _ in 0..3072 {
			pixels.push(buf[i]); i += 1;
		}
		images.push(pixels)
	}
	
	(coarse_labels, fine_labels, images)
}


static CIFAR_10: [&str; 10] = [
	"airplane",
	"automobile",
	"bird",
	"cat",
	"deer",
	"dog",
	"frog",
	"horse",
	"ship",
	"truck",
];

static CIFAR_100_COARSE: [&str; 20] = [
	"aquatic_mammals",
	"fish",
	"flowers",
	"food_containers",
	"fruit_and_vegetables",
	"household_electrical_devices",
	"household_furniture",
	"insects",
	"large_carnivores",
	"large_man-made_outdoor_things",
	"large_natural_outdoor_scenes",
	"large_omnivores_and_herbivores",
	"medium_mammals",
	"non-insect_invertebrates",
	"people",
	"reptiles",
	"small_mammals",
	"trees",
	"vehicles_1",
	"vehicles_2",
];

static CIFAR_100_FINE: [&str; 100] = [
	"apple",
	"aquarium_fish",
	"baby",
	"bear",
	"beaver",
	"bed",
	"bee",
	"beetle",
	"bicycle",
	"bottle",
	"bowl",
	"boy",
	"bridge",
	"bus",
	"butterfly",
	"camel",
	"can",
	"castle",
	"caterpillar",
	"cattle",
	"chair",
	"chimpanzee",
	"clock",
	"cloud",
	"cockroach",
	"couch",
	"crab",
	"crocodile",
	"cup",
	"dinosaur",
	"dolphin",
	"elephant",
	"flatfish",
	"forest",
	"fox",
	"girl",
	"hamster",
	"house",
	"kangaroo",
	"keyboard",
	"lamp",
	"lawn_mower",
	"leopard",
	"lion",
	"lizard",
	"lobster",
	"man",
	"maple_tree",
	"motorcycle",
	"mountain",
	"mouse",
	"mushroom",
	"oak_tree",
	"orange",
	"orchid",
	"otter",
	"palm_tree",
	"pear",
	"pickup_truck",
	"pine_tree",
	"plain",
	"plate",
	"poppy",
	"porcupine",
	"possum",
	"rabbit",
	"raccoon",
	"ray",
	"road",
	"rocket",
	"rose",
	"sea",
	"seal",
	"shark",
	"shrew",
	"skunk",
	"skyscraper",
	"snail",
	"snake",
	"spider",
	"squirrel",
	"streetcar",
	"sunflower",
	"sweet_pepper",
	"table",
	"tank",
	"telephone",
	"television",
	"tiger",
	"tractor",
	"train",
	"trout",
	"tulip",
	"turtle",
	"wardrobe",
	"whale",
	"willow_tree",
	"wolf",
	"woman",
	"worm",
];