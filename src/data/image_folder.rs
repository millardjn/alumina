extern crate image;

use walkdir::WalkDir;
use self::image::{GenericImage, DynamicImage, Pixel};
use ndarray::{ArrayD, ArrayViewD, IxDyn};
use std::path::{PathBuf, Path};
use data::DataSet;
use std::usize;
use std::io::*;

pub const CHANNELS: usize = 3;

pub struct ImageFolder {
	paths: Vec<PathBuf>,
}

impl ImageFolder {
	
	pub fn new<P: AsRef<Path>>(root_path: P, subfolders: bool) -> ImageFolder {
		let root_path = root_path.as_ref();

		print!("Loading paths for {} ... ", root_path.to_string_lossy());
		stdout().flush().ok();

		let walker = WalkDir::new(root_path).max_depth(if subfolders {usize::MAX} else {1}).into_iter();
		let paths = walker.filter_map(|e| e.ok()).filter_map(|e| {
			let path = e.path();
			
			if path.is_file() {
				if let Some(extension) = path.extension() {
					let extension = extension.to_string_lossy().to_lowercase();
					if ["jpg", "jpeg", "png", "bmp", "tiff"].iter().any(|&ext| ext == extension) {
						Some(path.to_path_buf())
					} else {
						None
					}
				} else {
					None
				}
			} else {
				None
			}
		}).collect::<Vec<_>>();
		println!("loaded {} paths.", paths.len());

		ImageFolder{
			paths: paths,
		}
	}
}

impl DataSet for ImageFolder {
	fn get(&mut self, i: usize) -> Vec<ArrayD<f32>> {

		let image = match image::open(&self.paths[i]) {
			Ok(ref dyn_image) => image_to_data(dyn_image),
			Err(err) => {
					eprintln!("Image load error '{}' {}", self.paths[i].to_string_lossy(), err);
					ArrayD::zeros(IxDyn(&[1, 1, CHANNELS][..]))
				},
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

		let mut img = DynamicImage::new_rgba8(width, height);

		for y in 0..height {
			for x in 0..width {
				let data = &data[(x + y*width) as usize*CHANNELS..][..CHANNELS];
				img.put_pixel(x, y, image::Rgba::from_channels((data[0]*255.0 + 0.5).min(255.0).max(0.0) as u8, (data[1]*255.0 + 0.5).min(255.0).max(0.0) as u8, (data[2]*255.0 + 0.5).min(255.0).max(0.0) as u8, 255u8));	
			}
		}
		// for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
		// 		let data = &mut data[(x + y*width) as usize*CHANNELS..][..CHANNELS];
		// 		*pixel = image::Rgba::from_channels((data[0]*255.0).min(255.0) as u8, (data[1]*255.0).min(255.0) as u8, (data[2]*255.0).min(255.0) as u8, 0u8);		
		// }
		img

}

pub fn image_to_data(image: &DynamicImage) -> ArrayD<f32> {
		let width = image.dimensions().0;
		let height = image.dimensions().1;

		let mut data = unsafe{
				ArrayD::uninitialized(IxDyn(&[height as usize, width as usize, CHANNELS][..]))
			};
		{
			let data_slice = data.as_slice_mut().unwrap();
			for y in 0..height {
				for x in 0..width {
					let pixel = image.get_pixel(x, y);
					let channels = pixel.channels();
					let data_slice = &mut data_slice[(x + y*width) as usize*CHANNELS..][..CHANNELS];
					for i in 0..CHANNELS {
						data_slice[i] = channels[i] as f32/255.0;
					}
				}
			}
		}

		data
}


#[test]
fn image_crop_test() {
	_image_crop_test()
}

fn _image_crop_test() {
	use data::{Cropping, DataStream};

	let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("res");

	let mut images = ImageFolder::new(d, true)
		.crop(0, &[25, 25, 3], Cropping::Random)
		.sequential();

	assert_eq!(&[25, 25, 3], images.next()[0].shape());
}