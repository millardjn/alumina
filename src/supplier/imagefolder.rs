extern crate image;

use self::image::{GenericImage, DynamicImage, Pixel};
use supplier::*;
use std::path::{PathBuf, Path};
use graph::*;	
use shape::*;
use rand::*;

use std::io::*;

pub const CHANNELS: usize = 3;

#[derive(Copy, Clone)]
pub enum Cropping {
	None,
	Centre{width:u32, height:u32},
	Random{width:u32, height:u32},
}

pub struct ImageFolderSupplier<S: Selector>{
	count: u64,
	epoch: usize,
	paths: Vec<PathBuf>,
	order: S,
	crop: Cropping,
}

impl<S: Selector> ImageFolderSupplier<S>{
		
	pub fn new(folder_path: &Path, subfolders: bool, crop: Cropping) -> ImageFolderSupplier<S> {
		
		print!("Loading paths for {} ... ", folder_path.to_string_lossy());
		stdout().flush().ok();
		let mut paths = vec![];
		file_paths(&mut paths, folder_path, subfolders);
		println!("loaded {} paths.", paths.len());

		let n = paths.len();
		ImageFolderSupplier{
			count: 0,
			epoch: n,
			paths: paths,
			order: S::new(n),
			crop: crop,
		}
	}
}

fn file_paths(mut paths: &mut Vec<PathBuf>, folder_path: &Path, subfolders: bool){
	let dir = folder_path.read_dir().expect(&format!("Could not read folder: {}", folder_path.to_string_lossy()));

	for e in dir.filter_map(|e| e.ok()) {

		let path = e.path();
		if path.is_file(){
			paths.push(path);
		} else if subfolders && path.is_dir(){
			file_paths(paths, &path, subfolders);
		}
	}
}

impl<S: Selector> Supplier for ImageFolderSupplier<S>{

	
	fn next_n(&mut self, n: usize) -> (Vec<NodeData>, Vec<NodeData>){
		assert!(n > 0, "n must be larger than 0");
		match self.crop {
			Cropping::None =>{
				assert_eq!(n, 1, "If cropping isnt specified images but be loaded one at a time. Specifiy cropping for this supplier, or restrict evaluation batching to 1");

				let data = load_full(&mut self.order, &self.paths);
				self.count += n as u64;

				(vec![data], vec![])
			},
			Cropping::Centre {width, height}| Cropping::Random{width, height} => {
				let mut data = NodeData::new_blank(DataShape::new(CHANNELS, &[width as usize, height as usize], n));
				let data_len = data.values.len();

				for data in data.values.chunks_mut(data_len/n){
					load_crop(&mut self.order, &self.paths, data, &self.crop);
				}

				self.count += n as u64;
				(vec![data], vec![])
			}
		}


	}
	
	fn epoch_size(&self) -> usize{
		self.epoch
	}

	fn samples_taken(&self) -> u64{
		self.count
	}

	fn reset(&mut self){
		self.order.reset();
		self.count = 0;
	}

}







pub fn load_crop<S: Selector>(order: &mut S, paths: &[PathBuf], data: &mut [f32], &cropping: &Cropping ){
	let (width, height) = match cropping {
		Cropping::Centre {width, height}| Cropping::Random{width, height} => (width, height),
		_ => panic!(""),
	};

	let mut iter = 0;
	let mut result = None;
	let mut last_path = paths[0].as_path();

	while iter < 100 && result.is_none() {
		let path_index = order.next();
		last_path = paths[path_index].as_path();
		result = image::open(last_path).ok();
		iter += 1;
	}

	let image = result.expect(&format!("100 consecutive attempts at opening images failed. Last path was: {}", last_path.to_string_lossy()));

	let (out_width, img_x_off, data_x_off) = range(&cropping, image.dimensions().0, width);
	let (out_height, img_y_off, data_y_off) = range(&cropping, image.dimensions().1, height);

	
	for y in 0..out_height {
		for x in 0..out_width {
			let pixel = image.get_pixel(x + img_x_off, y + img_y_off);
			let channels = pixel.channels();
			let data = &mut data[(x + data_x_off + (y+data_y_off)*width) as usize*CHANNELS..][..CHANNELS];
			for i in 0..CHANNELS {
				data[i] = channels[i] as f32/255.0;
			}
		}
	}
}

pub fn data_to_img(image_node: NodeData) -> DynamicImage {
		assert_eq!(image_node.shape.channels, CHANNELS);
		let data = &image_node.values;
		let width = image_node.shape.spatial_dimensions[0] as u32;
		let height = image_node.shape.spatial_dimensions[1] as u32;

		// let mut imgbuf = image::ImageBuffer::new(width, height);

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

pub fn img_to_data(data: &mut[f32], image: &DynamicImage){
		let width = image.dimensions().0;
		let height = image.dimensions().1;
		for y in 0..height {
			for x in 0..width {
				let pixel = image.get_pixel(x, y);
				let channels = pixel.channels();
				let data = &mut data[(x + y*width) as usize*CHANNELS..][..CHANNELS];
				for i in 0..CHANNELS {
					data[i] = channels[i] as f32/255.0;
				}
			}
		}
}

fn open_randomised_img<S: Selector>(order: &mut S, paths: &[PathBuf]) -> DynamicImage{
	let mut iter = 0;
	let mut result = None;
	let mut last_path = paths[0].as_path();

	while iter < 100 && result.is_none() {
		let path_index = order.next();
		last_path = paths[path_index].as_path();
		match image::open(last_path) {
			Ok(val) => result = Some(val),
			Err(err) => println!("Image load error '{}' {}", last_path.to_string_lossy(), err),
		}
		iter += 1;
	}

	let image = result.expect(&format!("100 consecutive attempts at opening images failed. Last path was: {}", last_path.to_string_lossy()));
	image
}

pub fn load_full<S: Selector>(order: &mut S, paths: &[PathBuf]) -> NodeData{
	
	let image = open_randomised_img(order, paths);
	let mut node_data = NodeData::new_blank(DataShape::new(CHANNELS, &[image.dimensions().0 as usize, image.dimensions().1 as usize], 1));

	img_to_data(&mut node_data.values, &image);

	node_data
}



// returns iterated width, img_x_off, and data_x_off
fn range(cropping: &Cropping, image_x: u32, data_x: u32) -> (u32, u32, u32) {
	match cropping {
		&Cropping::Centre{..} => {
			if image_x < data_x {
				(image_x, 0, (data_x - image_x)/2)
			} else {
				(data_x, (image_x - data_x)/2, 0)
			}
		},

		&Cropping::Random{..} => {
			if image_x < data_x {
				(image_x, 0, thread_rng().gen_range(0, data_x - image_x + 1))
			} else {
				(data_x, thread_rng().gen_range(0, image_x - data_x + 1), 0)
			}
		},
		_ => panic!(),
	}
}

