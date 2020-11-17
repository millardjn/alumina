use ndarray::{ArrayD, IxDyn};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use data::{ConcatComponents, DataSet};
use byteorder::{BigEndian, ByteOrder};

/// A `DataSet` created from both mnist binary format image and label files.
///
/// Each element contains two components, a (typically) 28x28 image and a one-hot label encoding the category out of 10
pub struct Mnist {
	data: ConcatComponents<MnistImages, MnistLabels>
}

impl Mnist {
	pub fn new(image_file: File, label_file: File) -> Self {
		let joined = MnistImages::new(image_file).concat_components(MnistLabels::new(label_file));
		Mnist{data: joined}
	}

	pub fn training<P: AsRef<Path>>(folder_path: P) -> Self {
		let folder_path = folder_path.as_ref();
		let err = "Pass a folder containing 'train-images-idx3-ubyte' and 'train-labels-idx1-ubyte'";
		assert!(folder_path.is_dir(), err);
		let label_file = File::open(folder_path.join("train-labels-idx1-ubyte").as_path()).expect(err);
		let image_file = File::open(folder_path.join("train-images-idx3-ubyte").as_path()).expect(err);
		Mnist::new(image_file, label_file)
	}

	pub fn testing<P: AsRef<Path>>(folder_path: P) -> Self {
		let folder_path = folder_path.as_ref();
		let err = "Pass a folder containing 't10k-images-idx3-ubyte' and 't10k-labels-idx1-ubyte'";
		assert!(folder_path.is_dir(), err);
		let label_file = File::open(folder_path.join("t10k-labels-idx1-ubyte").as_path()).expect(err);
		let image_file = File::open(folder_path.join("t10k-images-idx3-ubyte").as_path()).expect(err);
		Mnist::new(image_file, label_file)
	}
}

impl DataSet for Mnist {
	fn get(&mut self, i: usize) -> Vec<ArrayD<f32>>{
		self.data.get(i)
	}

	fn length(&self) -> usize{
		self.data.length()
	}

	fn width(&self) -> usize{
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
	pub fn new(image_file: File) -> Self{
		let (shape, data) = read_image_file(image_file);
		MnistImages {
			shape,
			data,
		}
	}
}

impl DataSet for MnistImages {
	fn get(&mut self, i: usize) -> Vec<ArrayD<f32>>{
		let image = ArrayD::from_shape_vec(self.shape.as_slice(), self.data[i].iter().map(|&v| v as f32/255.0).collect()).unwrap();
		vec![image]
	}

	fn length(&self) -> usize{
		self.data.len()
	}

	fn width(&self) -> usize{
		1
	}

	fn components(&self) -> Vec<String> {
		vec!["Images".to_string()]
	}
}

fn read_image_file(mut file: File) -> (Vec<usize>, Vec<Vec<u8>>){
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
	
	(vec![w, h, 1], v1)
}

/// A `DataSet` created from a mnist binary format label file.
///
/// Each element contains only one component, a typically [10] one-hot array encoding the category of the associated image element
pub struct MnistLabels {
	labels: Vec<u8>,
}

impl MnistLabels {
	pub fn new(label_file: File) -> Self{
		MnistLabels{
			labels: read_label_file(label_file),
		}
	}
}

impl DataSet for MnistLabels {
	fn get(&mut self, i: usize) -> Vec<ArrayD<f32>>{
		let mut one_hot_label = ArrayD::zeros(IxDyn(&[10]));
		one_hot_label[self.labels[i] as usize] = 1.0;
		vec![one_hot_label]
	}

	fn length(&self) -> usize{
		self.labels.len()
	}

	fn width(&self) -> usize{
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
	
	assert!(BigEndian::read_u32(&buf[i..]) == 2049); i += 4;
	
	let n = BigEndian::read_u32(&buf[i..]) as usize; i += 4;
	
	let mut v1 = Vec::with_capacity(n);
	for _ in 0..n {
		v1.push(buf[i]); i += 1;
	}
	
	v1
}

#[cfg(feature = "download_mnist")]
pub fn download() -> Result<PathBuf, String> {
	Ok(mnist_download::download_and_extract_all()?)
}

#[cfg(feature = "download_mnist")]
mod mnist_download {
	extern crate flate2;
	extern crate reqwest;

	use std::io::{Read, Write};
	use std::path::{Path, PathBuf};
	use std::{fs, io};

	const MNIST_BASE_URL: &str = "http://yann.lecun.com/exdb/mnist";

	const MNIST_ARCHIVE_TRAIN_IMAGES: &str = "train-images-idx3-ubyte.gz";
	const MNIST_ARCHIVE_TRAIN_LABELS: &str = "train-labels-idx1-ubyte.gz";
	const MNIST_ARCHIVE_TEST_IMAGES: &str = "t10k-images-idx3-ubyte.gz";
	const MNIST_ARCHIVE_TEST_LABELS: &str = "t10k-labels-idx1-ubyte.gz";

	const ARCHIVES_TO_DOWNLOAD: &[&str] = &[
		MNIST_ARCHIVE_TRAIN_IMAGES,
		MNIST_ARCHIVE_TRAIN_LABELS,
		MNIST_ARCHIVE_TEST_IMAGES,
		MNIST_ARCHIVE_TEST_LABELS,
	];

	pub(super) fn download_and_extract_all() -> Result<PathBuf, String> {
		let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
		let download_dir = crate_dir.join("target").join("mnist");
		if !download_dir.exists() {
			println!(
				"Download directory {} does not exists. Creating....",
				download_dir.display()
			);
			fs::create_dir(&download_dir).expect("Failed to create directory");
		}
		for archive in ARCHIVES_TO_DOWNLOAD {
			println!("Attempting to download and extract {}...", archive);
			download(&archive, &download_dir)?;
			extract(&archive, &download_dir)?;
		}

		Ok(download_dir)
	}

	fn download(archive: &str, download_dir: &Path) -> Result<(), String> {
		let url = format!("{}/{}", MNIST_BASE_URL, archive);

		let file_name = download_dir.join(&archive);

		if file_name.exists() {
			println!(
				"  File {:?} already exists, skipping downloading.",
				file_name
			);
		} else {
			println!("  Downloading {} to {:?}...", url, download_dir);

			let f = fs::File::create(&file_name)
				.or_else(|e| Err(format!("Failed to create file {:?}: {:?}", file_name, e)))?;

			let mut writer = io::BufWriter::new(f);

			let mut response = reqwest::get(&url)
				.or_else(|e| Err(format!("Failed to download {:?}: {:?}", url, e)))?;

			let _ = io::copy(&mut response, &mut writer).or_else(|e| {
				Err(format!(
					"Failed to to write to file {:?}: {:?}",
					file_name, e
				))
			})?;

			println!("  Downloading or {} to {:?} done!", archive, download_dir);
		}

		Ok(())
	}

	fn extract(archive_name: &str, download_dir: &Path) -> Result<(), String> {
		let archive = download_dir.join(&archive_name);
		let extract_to = download_dir.join(&archive_name.replace(".gz", ""));
		if extract_to.exists() {
			println!(
				"  Extracted file {:?} already exists, skipping extraction.",
				extract_to
			);
		} else {
			println!("Extracting archive {:?} to {:?}...", archive, extract_to);
			let file_in = fs::File::open(&archive)
				.or_else(|e| Err(format!("Failed to open archive {:?}: {:?}", archive, e)))?;
			let file_in = io::BufReader::new(file_in);

			let file_out = fs::File::create(&extract_to).or_else(|e| {
				Err(format!(
					"  Failed to create extracted file {:?}: {:?}",
					archive, e
				))
			})?;
			let mut file_out = io::BufWriter::new(file_out);

			let mut gz = flate2::bufread::GzDecoder::new(file_in);
			let mut v: Vec<u8> = Vec::with_capacity(10 * 1024 * 1024);
			gz.read_to_end(&mut v)
				.or_else(|e| Err(format!("Failed to extract archive {:?}: {:?}", archive, e)))?;

			file_out.write_all(&v).or_else(|e| {
				Err(format!(
					"Failed to write extracted data to {:?}: {:?}",
					archive, e
				))
			})?;
		}

		Ok(())
	}
}
