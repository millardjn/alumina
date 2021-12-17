use alumina_data::DataSet;
use image::{DynamicImage, ImageBuffer, Pixel};
use ndarray::{ArcArray, ArrayD, ArrayViewD, IxDyn};
use std::{
	path::{Path, PathBuf},
	usize,
};
use walkdir::WalkDir;

pub const CHANNELS: usize = 3;

/// An DataSet formed from the images in the supplied folder.
///
/// Images are not loaded into memory immediately, but are read from disk as necessary.
///
/// Image order is defined by lexical order of their Paths.
///
/// If there is an error loading an image, a 1x1 pixel image is return instead allowing training to continue.
pub struct ImageFolder {
	paths: Vec<PathBuf>,
}

impl ImageFolder {
	pub fn new<P: AsRef<Path>>(root_path: P, subfolders: bool) -> ImageFolder {
		let root_path = root_path.as_ref();

		let walker = WalkDir::new(root_path)
			.sort_by_file_name()
			.max_depth(if subfolders { usize::MAX } else { 1 })
			.into_iter();
		let mut paths = walker
			.filter_map(|e| e.ok())
			.filter_map(|e| {
				let path = e.path();

				if path.is_file() {
					if let Some(extension) = path.extension() {
						let extension = extension.to_string_lossy().to_lowercase();
						if ["jpg", "jpeg", "png", "bmp", "tiff"]
							.iter()
							.any(|&ext| ext == extension)
						{
							return Some(path.to_path_buf());
						}
					}
				}
				None
			})
			.collect::<Vec<_>>();

		paths.sort();
		ImageFolder { paths }
	}
}

impl DataSet for ImageFolder {
	fn get(&mut self, i: usize) -> Vec<ArcArray<f32, IxDyn>> {
		let image = match image::open(&self.paths[i]) {
			Ok(ref dyn_image) => image_to_data(dyn_image).to_shared(),
			Err(err) => {
				eprintln!("Image load error '{}' {}", self.paths[i].to_string_lossy(), err);
				ArcArray::zeros(IxDyn(&[1, 1, CHANNELS][..]))
			}
		};

		vec![image]
	}

	fn length(&self) -> usize {
		self.paths.len()
	}

	fn width(&self) -> usize {
		1
	}

	fn components(&self) -> Vec<String> {
		vec!["Images".to_string()]
	}
}

pub fn data_to_image(image_data: ArrayViewD<f32>) -> DynamicImage {
	assert_eq!(image_data.shape()[2], CHANNELS);
	let data = &image_data.as_slice().unwrap();
	let width = image_data.shape()[1] as u32;
	let height = image_data.shape()[0] as u32;

	let mut img = ImageBuffer::new(width, height);

	const MAX: f32 = u16::MAX as f32;

	for y in 0..height {
		for x in 0..width {
			let data = &data[(x + y * width) as usize * CHANNELS..][..CHANNELS];
			img.put_pixel(
				x,
				y,
				image::Rgb::from_channels(
					(data[0] * MAX + 0.5).min(MAX).max(0.0) as u16,
					(data[1] * MAX + 0.5).min(MAX).max(0.0) as u16,
					(data[2] * MAX + 0.5).min(MAX).max(0.0) as u16,
					0,
				),
			);
		}
	}
	DynamicImage::ImageRgb16(img)
}

pub fn image_to_data(image: &DynamicImage) -> ArrayD<f32> {
	let image = image.to_rgb16();
	let (width, height) = image.dimensions();

	let mut data = ArrayD::zeros(IxDyn(&[height as usize, width as usize, CHANNELS][..]));
	{
		let data_slice = data.as_slice_mut().unwrap();
		for (x, y, pixel) in image.enumerate_pixels() {
			let channels = pixel.channels();
			let data_slice = &mut data_slice[(x + y * width) as usize * CHANNELS..][..CHANNELS];
			for i in 0..CHANNELS {
				data_slice[i] = channels[i] as f32 / u16::MAX as f32;
			}
		}
	}

	data
}

// #[test]
// fn image_crop_test() {
// 	_image_crop_test()
// }

// fn _image_crop_test() {
// 	use alumina_data::{Cropping, DataStream};

// 	let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
// 	d.push("res");

// 	let mut images = ImageFolder::new(d, true)
// 		.crop(0, &[25, 25, 3], Cropping::Random)
// 		.sequential();

// 	assert_eq!(&[25, 25, 3], images.next()[0].shape());
// }
