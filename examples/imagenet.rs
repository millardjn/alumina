extern crate image;
extern crate rand;
extern crate alumina;
extern crate bytevec;

use bytevec::{ByteEncodable, ByteDecodable};
use image::{GenericImage, DynamicImage, Pixel};
use rand::*;

use alumina::ops::activ::*;
use alumina::ops::basic::*;	
use alumina::ops::loss::*;
use alumina::ops::conv::*;
use alumina::ops::reshape::*;
use alumina::opt::*;
use alumina::opt::asgd::*;
use alumina::ops::*;

use alumina::graph::*;	
use alumina::vec_math::*;
use alumina::shape::*;

use alumina::opt::supplier::*;
use alumina::opt::supplier::imagenet::ImagenetSupplier;
use alumina::opt::supplier::imagefolder::{ImageFolderSupplier, Cropping, CHANNELS, img_to_data, data_to_img};


use std::io::{self, Write, Read};
use std::sync::Arc;
use std::path::{PathBuf, Path};
use std::fs::File;

fn main(){

	let args = std::env::args().collect::<Vec<_>>();

	if args.len() <= 1 {
		println!("Argument should either, be a path of an image to upscale followed by a path for the parameter file, or -train followed by the path of a folder containing the imagenet dataset")
	} else if args[1].to_lowercase() == "-train" {
		upscaler_train();
	} else if args[1].to_lowercase() == "-linear" {
		let image = image::open(Path::new(&args[2])).expect("Error opening image");
		let mut input = NodeData::new_blank(DataShape::new(CHANNELS, &[image.dimensions().0 as usize, image.dimensions().1 as usize], 1));
		img_to_data(&mut input.values, &image);

		let mut g = Graph::new();
		let inp = g.add_input_node(Node::new_shaped(CHANNELS, 2, "input"));
		let lin = g.add_node(Node::new_shaped(CHANNELS, 2, "linear"));
		let upscale = g.add_node(Node::new_shaped(CHANNELS, 2, "upscale"));
		let out = g.add_output_node(Node::new_shaped(CHANNELS, 2, "output"));
		g.add_operation(SrgbToLinear::new(&inp, &lin, "lin"));
		g.add_operation(LinearInterp::new(&lin, &upscale, &[3, 3], "linterp"));
		g.add_operation(LinearToSrgb::new(&upscale, &out, "srgb"));
		g.add_operation(ShapeConstraint::new(&inp, &upscale, &[Arc::new(|d| d*3), Arc::new(|d| d*3)], "triple"));

		let output = g.forward(1, vec![input], &vec![]).remove(0);


		let mut out_path: PathBuf = Path::new(&args[2]).to_path_buf();
		let st = out_path.file_stem().unwrap().to_string_lossy().into_owned();
		out_path.set_file_name(format!("{}{}", st, "upscaled"));
		out_path.set_extension("png");
		let mut out_file = File::create(&out_path).expect("Could not save output file");
		
		data_to_img(output).save(&mut out_file, image::ImageFormat::PNG);


	} else if args.len() > 2 {
		let image = image::open(Path::new(&args[1])).expect("Error opening image");
		let mut param_file = File::open(Path::new(&args[2])).expect("Error opening parameter file");

		let mut data = Vec::new();
		param_file.read_to_end(&mut data);
		let params = <Vec<f32>>::decode::<u32>(&data).unwrap();

		let activ_func: Box<Fn(&NodeID, &NodeID) -> Box<Operation>>= Box::new(|input, output| BeLU::new(input, output, ParamSharing::Spatial, "generic activ", BeLU::init_porque_no_los_dos()));
		let mut g = upscaler_net(3, false, 0.0, activ_func.as_ref());

		let mut input = NodeData::new_blank(DataShape::new(CHANNELS, &[image.dimensions().0 as usize, image.dimensions().1 as usize], 1));

		img_to_data(&mut input.values, &image);

		let output = g.forward(1, vec![input], &params).remove(0);
		// let mut out_name: String = args[1].clone();
		// out_name.push_str("_upscaled");

		let mut out_path: PathBuf = Path::new(&args[1]).to_path_buf();
		let st = out_path.file_stem().unwrap().to_string_lossy().into_owned();
		out_path.set_file_name(format!("{}{}", st, "upscaled"));
		out_path.set_extension("png");
		let mut out_file = File::create(&out_path).expect("Could not save output file");
		
		data_to_img(output).save(&mut out_file, image::ImageFormat::PNG);

	}
	


	//upscaler_linear_test();
	
	
}

fn upscaler_train(){

	// type inference seems to crap out unless the type is made explicit
	let activ_func: Box<Fn(&NodeID, &NodeID) -> Box<Operation>> = Box::new(|input, output| BeLU::new(input, output, ParamSharing::Spatial, "generic activ", BeLU::init_porque_no_los_dos()));
	
	let mut g = upscaler_net(3, true, 1e-5, activ_func.as_ref());

	let mut training_set = ImagenetSupplier::<Random>::new(Path::new("D:/ML/ImageNet"), Cropping::Random{width:120, height:120});

	//let mut training_set = ImageFolderSupplier::new(Path::new("C:/anime"),Some((120, 120,Cropping::Random)));

	let args = std::env::args().collect::<Vec<_>>();
	let start_params = if args.len() > 2 {
		let mut param_file = File::open(Path::new(&args[2])).expect("Error opening parameter file");
		let mut data = Vec::new();
		param_file.read_to_end(&mut data);
		<Vec<f32>>::decode::<u32>(&data).unwrap()
	} else {
		g.init_params()
	};

	//let start_params = g.init_params();
	let mut opt_params = start_params.clone();

	let mut solver = Asgd2::new(&mut g);
	

	{ // Add occasional test set evaluation as solver callback
		let mut g2 = upscaler_net(3, true, 0.0, activ_func.as_ref());
		let mut test_set = ImageFolderSupplier::<Random>::new(Path::new("C:/Set14"), Cropping::None);
		let n = test_set.epoch_size() as usize;

		solver.add_step_callback(move |_err, step, _evaluations, _graph, params|{

			if step % 100 == 0 {
				print!("Test errors:\t");
				for i in 0..n {

					let (input, mut training_input) = test_set.next_n(1);
					training_input.push(NodeData::new_blank(DataShape::new_flat(1000, 1)));

					let (batch_err, _, _) = g2.backprop(1, input, training_input, params);
					
					print!("{}\t", batch_err)
				}
				println!("");
			}

			if step % 100 == 0 || step == 1{
				let bytes = params.encode::<u32>().unwrap();
				let string = format!("D:/ML/{}.par", step);
				let out_path = Path::new(&string);
				let mut out = File::create(out_path).unwrap();

				out.write_all(&bytes).unwrap();
			}


			true
		});
	}

	//solver.set_min_batch_size(2.);
	solver.set_max_evals(10_000_000);//(training_set.epoch_size()*10) as u64
	opt_params = solver.optimise_from(&mut training_set, opt_params.clone());	

	println!("Params moved:{}", opt_params.add_scaled(&start_params, -1.0).norm2());

}

fn upscaler_net6(factor: usize, training: bool, regularisation: f32, activation_func: &Fn(&NodeID, &NodeID) -> Box<Operation>) -> Graph {
	let mut g = Graph::new();

	//-- Input/Output
	let (input, output) = if training {
		(g.add_node(Node::new_shaped(CHANNELS, 2, "input")), // in training a high resolution image will be the input node, added at the bottom.
		g.add_node(Node::new_shaped(CHANNELS, 2, "output")))
	} else {
		(g.add_input_node(Node::new_shaped(CHANNELS, 2, "input")),
		g.add_output_node(Node::new_shaped(CHANNELS, 2, "output")))
	};

	g.add_operation(LinearInterp::new(&input, &output, &[factor, factor], "linterp"));



	let mut ops: Vec<Box<Operation>> = vec![];

	let n = 32;
	let l1_conv = g.add_node(Node::new_shaped(n, 2, "l1_conv"));
	let l1_activ = g.add_node(Node::new_shaped(n, 2, "l1_activ"));
	let l2_conv = g.add_node(Node::new_shaped(n, 2, "l2_conv"));
	let l2_activ = g.add_node(Node::new_shaped(n, 2, "l2_activ"));
	let l3_conv = g.add_node(Node::new_shaped(n, 2, "l3_conv"));
	let l3_activ = g.add_node(Node::new_shaped(n, 2, "l3_activ"));

	let n_expand = 8;
	let expand = g.add_node(Node::new_shaped(n_expand*factor*factor, 2, "expand"));
	let expand_activ = g.add_node(Node::new_shaped(n_expand*factor*factor, 2, "expand_activ"));
	let expand_out = g.add_node(Node::new_shaped(n_expand, 2, "expand_out"));
	

	//-- Low Res DenseNet-like Convolution Connections

	ops.push(Bias::new(&l1_conv, ParamSharing::Spatial, "l1_bias", init_fill(0.0)));
	ops.push(Bias::new(&l2_conv, ParamSharing::Spatial, "l2_bias", init_fill(0.0)));
	ops.push(Bias::new(&l3_conv, ParamSharing::Spatial, "l3_bias", init_fill(0.0)));

	ops.push(activation_func(&l1_conv, &l1_activ));
	ops.push(activation_func(&l2_conv, &l2_activ));
	ops.push(activation_func(&l3_conv, &l3_activ));

	ops.push(Convolution::new(&input, &l1_conv, &[5, 5], Padding::Same, "conv1", Convolution::init_msra(0.1)));
	ops.push(Convolution::new(&input, &l2_conv, &[5, 5], Padding::Same, "conv2", Convolution::init_msra(0.1)));
	ops.push(Convolution::new(&input, &l3_conv, &[5, 5], Padding::Same, "conv3", Convolution::init_msra(0.1)));
	ops.push(Convolution::new(&input, &expand, &[1, 1], Padding::Same, "conv4", Convolution::init_msra(0.01)));

	ops.push(Convolution::new(&l1_activ, &l2_conv, &[3, 3], Padding::Same, "conv5", Convolution::init_msra(0.1)));
	ops.push(Convolution::new(&l1_activ, &l3_conv, &[3, 3], Padding::Same, "conv6", Convolution::init_msra(0.1)));
	ops.push(Convolution::new(&l1_activ, &expand, &[1, 1], Padding::Same, "conv7", Convolution::init_msra(0.1)));
	ops.push(Convolution::new(&l2_activ, &l3_conv, &[3, 3], Padding::Same, "conv8", Convolution::init_msra(0.1)));
	ops.push(Convolution::new(&l2_activ, &expand, &[1, 1], Padding::Same, "conv9", Convolution::init_msra(0.1)));
	ops.push(Convolution::new(&l3_activ, &expand, &[1, 1], Padding::Same, "conv10", Convolution::init_msra(0.1)));


	// -- Output
	ops.push(Bias::new(&expand, ParamSharing::Spatial, "expand_bias", init_fill(0.0)));
	ops.push(activation_func(&expand, &expand_activ));
	ops.push(Expand::new(&expand_activ, &expand_out, &[factor, factor], "expand"));
	ops.push(Convolution::new(&expand_out, &output, &[5, 5], Padding::Same, "conv_out", Convolution::init_msra(0.1)));
	ops.push(Bias::new(&output, ParamSharing::Spatial, "output_bias", init_fill(0.0)));	



	let op_ids = g.add_operations(ops);

	if training && regularisation != 0.0 {
		for op_id in &op_ids {
			if op_id.num_params == 0 {continue};
			g.add_secondary_operation(L2Regularisation::new(op_id, regularisation, "L2"), op_id);
		}
	}

	if training {
		//let _dummy_training_node = g.add_training_input_node(Node::new_flat(1000, "label"));
		let input_hr = g.add_input_node(Node::new_shaped(CHANNELS, 2, "input_hr"));
		let input_hr_lin = g.add_node(Node::new_shaped(CHANNELS, 2, "input_hr_lin"));
		let input_pool = g.add_node(Node::new_shaped(CHANNELS, 2, "input_pool"));
		g.add_operation(SrgbToLinear::new(&input_hr, &input_hr_lin,"srgb2lin"));
		g.add_operation(Pooling::new(&input_hr_lin, &input_pool, &[factor, factor], "input_pooling"));
		g.add_operation(LinearToSrgb::new(&input_pool, &input, "lin2srgb"));

		//g.add_operation(Pooling::new(&input_hr, &input, &[factor, factor], "input_pooling"));

		g.add_operation(MseLoss::new(&output, &input_hr, 100.0, "loss"));

		g.add_operation(ShapeConstraint::new(&input_hr, &output, &[Arc::new(|d| d), Arc::new(|d| d)], "output_shape"));
		g.add_operation(ShapeConstraint::new(&input_hr, &expand_out, &[Arc::new(|d| d), Arc::new(|d| d)], "expand_shape"));	
	} else {
		g.add_operation(ShapeConstraint::new(&input, &output, &[Arc::new(move|d| d*factor), Arc::new(move|d| d*factor)], "output_shape"));	
		g.add_operation(ShapeConstraint::new(&input, &expand_out, &[Arc::new(move|d| d*factor), Arc::new(move|d| d*factor)], "expand_shape"));
	}

	g

}

fn upscaler_net(factor: usize, training: bool, regularisation: f32, activation_func: &Fn(&NodeID, &NodeID) -> Box<Operation>) -> Graph {
	let mut g = Graph::new();

	//-- Input/Output
	let (input, output) = if training {
		(g.add_node(Node::new_shaped(CHANNELS, 2, "input")), // in training a high resolution image will be the input node, added at the bottom.
		g.add_node(Node::new_shaped(CHANNELS, 2, "output")))
	} else {
		(g.add_input_node(Node::new_shaped(CHANNELS, 2, "input")),
		g.add_output_node(Node::new_shaped(CHANNELS, 2, "output")))
	};

	

	g.add_operation(LinearInterp::new(&input, &output, &[factor, factor], "linterp"));

	let mut ops: Vec<Box<Operation>> = vec![];

	let f_conv = g.add_node(Node::new_shaped(32, 2, "l1_conv"));
	let f_activ = g.add_node(Node::new_shaped(32, 2, "l1_activ"));
	ops.push(Convolution::new(&input, &f_conv, &[3, 3], Padding::Same, "conv0", Convolution::init_msra(1.0)));
	ops.push(Bias::new(&f_conv, ParamSharing::Spatial, "l1_bias", init_fill(0.0)));
	ops.push(activation_func(&f_conv, &f_activ));
	

	let expand = g.add_node(Node::new_shaped(CHANNELS*factor*factor, 2, "expand"));
	ops.push(Bias::new(&expand, ParamSharing::Spatial, "expand_bias", init_fill(0.0)));
	ops.push(Expand::new(&expand, &output, &[factor, factor], "expand"));

	
	//g.add_operation(Convolution::new(&f_activ, &expand, &[1, 1], Padding::Same, "conv1", Convolution::init_msra(0.1)));


	//-- Low Res DenseNet-like Convolution Connections
	for _ in 0..1 {
		let n = 32;
		let l1_conv = g.add_node(Node::new_shaped(n, 2, "l1_conv"));
		let l1_activ = g.add_node(Node::new_shaped(n, 2, "l1_activ"));
		let l2_conv = g.add_node(Node::new_shaped(n, 2, "l2_conv"));
		let l2_activ = g.add_node(Node::new_shaped(n, 2, "l2_activ"));
		let l3_conv = g.add_node(Node::new_shaped(n, 2, "l3_conv"));
		let l3_activ = g.add_node(Node::new_shaped(n, 2, "l3_activ"));


		ops.push(Bias::new(&l1_conv, ParamSharing::Spatial, "l1_bias", init_fill(0.0)));
		ops.push(Bias::new(&l2_conv, ParamSharing::Spatial, "l2_bias", init_fill(0.0)));
		ops.push(Bias::new(&l3_conv, ParamSharing::Spatial, "l3_bias", init_fill(0.0)));

		ops.push(activation_func(&l1_conv, &l1_activ));
		ops.push(activation_func(&l2_conv, &l2_activ));
		ops.push(activation_func(&l3_conv, &l3_activ));

		ops.push(Convolution::new(&f_activ, &l1_conv, &[5, 5], Padding::Same, "conv1", Convolution::init_msra(0.1)));
		ops.push(Convolution::new(&f_activ, &l2_conv, &[5, 5], Padding::Same, "conv2", Convolution::init_msra(0.1)));
		ops.push(Convolution::new(&f_activ, &l3_conv, &[5, 5], Padding::Same, "conv3", Convolution::init_msra(0.1)));
		//ops.push(Convolution::new(&f_activ, &expand, vec![3, 3], Padding::Same, "conv3", Convolution::init_msra(0.1)));

		ops.push(Convolution::new(&l1_activ, &l2_conv, &[3, 3], Padding::Same, "conv5", Convolution::init_msra(0.1)));
		ops.push(Convolution::new(&l1_activ, &l3_conv, &[3, 3], Padding::Same, "conv6", Convolution::init_msra(0.1)));
		ops.push(Convolution::new(&l1_activ, &expand, &[3, 3], Padding::Same, "conv8", Convolution::init_msra(0.1)));

		ops.push(Convolution::new(&l2_activ, &l3_conv, &[3, 3], Padding::Same, "conv9", Convolution::init_msra(0.1)));
		ops.push(Convolution::new(&l2_activ, &expand, &[3, 3], Padding::Same, "conv11", Convolution::init_msra(0.1)));

		ops.push(Convolution::new(&l3_activ, &expand, &[3, 3], Padding::Same, "conv13", Convolution::init_msra(0.1)));

	}

	let op_ids = g.add_operations(ops);

	if training && regularisation != 0.0 {
		for op_id in &op_ids {
			if op_id.num_params == 0 {continue};
			g.add_secondary_operation(L2Regularisation::new(op_id, regularisation, "L2"), op_id);
		}
	}


	if training {
		let _dummy_training_node = g.add_training_input_node(Node::new_flat(1000, "label"));
		let input_hr = g.add_input_node(Node::new_shaped(CHANNELS, 2, "input_hr"));
		let input_hr_lin = g.add_node(Node::new_shaped(CHANNELS, 2, "input_hr_lin"));
		let input_pool = g.add_node(Node::new_shaped(CHANNELS, 2, "input_pool"));
		g.add_operation(SrgbToLinear::new(&input_hr, &input_hr_lin,"srgb2lin"));
		g.add_operation(Pooling::new(&input_hr_lin, &input_pool, &[factor, factor], "input_pooling"));
		g.add_operation(LinearToSrgb::new(&input_pool, &input, "lin2srgb"));

		//g.add_operation(Pooling::new(&input_hr, &input, &[factor, factor], "input_pooling"));

		g.add_operation(MseLoss::new(&output, &input_hr, 100.0, "loss"));

		g.add_operation(ShapeConstraint::new(&input_hr, &output, &[Arc::new(|d| d), Arc::new(|d| d)], "output_shape"));
		//g.add_operation(ShapeConstraint::new(&input_hr, &expand_out, &[Arc::new(|d| d), Arc::new(|d| d)], "expand_shape"));	
	} else {
		g.add_operation(ShapeConstraint::new(&input, &output, &[Arc::new(move|d| d*factor), Arc::new(move|d| d*factor)], "output_shape"));	
		//g.add_operation(ShapeConstraint::new(&input, &expand_out, &[Arc::new(move|d| d*factor), Arc::new(move|d| d*factor)], "expand_shape"));
	}

	g

}



fn upscaler_net4(factor: usize, training_net: bool, activation_func: &Fn(&NodeID, &NodeID) -> Box<Operation>) -> Graph {
	let mut g = Graph::new();


	//-- Input/Output
	let (input, output) = if training_net {
		(g.add_node(Node::new_shaped(CHANNELS, 2, "input")), // in training a high resolution image will be the input node, added at the bottom.
		g.add_node(Node::new_shaped(CHANNELS, 2, "output")))
	} else {
		(g.add_input_node(Node::new_shaped(CHANNELS, 2, "input")),
		g.add_output_node(Node::new_shaped(CHANNELS, 2, "output")))
	};
	

	//-- Linear Bypass
	let input_lin = g.add_node(Node::new_shaped(CHANNELS, 2, "input_lin"));
	let upscale_lin = g.add_node(Node::new_shaped(CHANNELS, 2, "upscale_lin"));

	g.add_operation(SrgbToLinear::new(&input, &input_lin,"srgb2lin"));
	g.add_operation(LinearInterp::new(&input_lin, &upscale_lin, &[factor, factor], "linterp"));
	g.add_operation(LinearToSrgb::new(&upscale_lin, &output, "lin2srgb"));
	



	//-- Tower Exit Path
	let exit_channels = 16;
	let exit_conv   = g.add_node(Node::new_shaped(exit_channels*factor*factor, 2, "exit_conv"));
	let exit_activ  = g.add_node(Node::new_shaped(exit_channels*factor*factor, 2, "exit_activ"));
	let exit_expand = g.add_node(Node::new_shaped(exit_channels, 2, "exit_expand"));

	
	g.add_operation(Bias::new(&exit_conv, ParamSharing::Spatial, "exit_conv_bias", init_fill(0.0)));
	g.add_operation(activation_func(&exit_conv, &exit_activ));
	g.add_operation(Expand::new(&exit_activ, &exit_expand, &[factor, factor], "exit_expand"));
	

	g.add_operation(Convolution::new(&exit_expand, &output, &[5, 5], Padding::Same, "output_conv", Convolution::init_msra(1.0)));
	g.add_operation(Bias::new(&output, ParamSharing::Spatial, "output_bias", init_fill(0.0)));
	
	// let exit_conv = g.add_node(Node::new_shaped(CHANNELS*factor*factor, 2, "exit_conv"));
	// g.add_operation(Expand::new(&exit_conv, &output, &[factor, factor], "exit_expand"));
	// g.add_operation(Bias::new(&output, ParamSharing::Spatial, "output_bias", init_fill(0.0)));

	// more training input stuff
	if training_net {
		let _dummy_training_node = g.add_training_input_node(Node::new_flat(1000, "label"));
		let input_hr = g.add_input_node(Node::new_shaped(CHANNELS, 2, "input_hr"));
		let input_hr_lin = g.add_node(Node::new_shaped(CHANNELS, 2, "input_hr_lin"));
		let input_pool = g.add_node(Node::new_shaped(CHANNELS, 2, "input_pool"));
		g.add_operation(SrgbToLinear::new(&input_hr, &input_hr_lin,"srgb2lin"));
		g.add_operation(Pooling::new(&input_hr_lin, &input_pool, &[factor, factor], "input_pooling"));
		g.add_operation(LinearToSrgb::new(&input_pool, &input, "lin2srgb"));
		//g.add_operation(Pooling::new(&input_hr, &input, &[factor, factor], "input_pooling"));
		g.add_operation(MseLoss::new(&output, &input_hr, 100.0, "loss"));

		g.add_operation(ShapeConstraint::new(&input_hr, &exit_expand, &[Arc::new(|d| d), Arc::new(|d| d)], "output_shape"));	
		g.add_operation(ShapeConstraint::new(&input_hr, &upscale_lin, &[Arc::new(|d| d), Arc::new(|d| d)], "output_shape"));	
	} else {
		g.add_operation(ShapeConstraint::new(&input, &exit_expand, &[Arc::new(move|d| d*factor), Arc::new(move|d| d*factor)], "output_shape"));
		g.add_operation(ShapeConstraint::new(&input, &upscale_lin, &[Arc::new(move|d| d*factor), Arc::new(move|d| d*factor)], "output_shape"));		
	}



	//-- Tower entry exit points
	let entry = input;
	let exit = exit_conv;

	let (entry, exit) = add_single_layer(&mut g, &entry, &exit, 128, 64, 3, 9, activation_func);
	//let (entry, exit) = add_single_layer(&mut g, &entry, &exit, 64, 32, 3, 5, activation_func);
	//let (entry, exit) = add_single_layer(&mut g, &entry, &exit, 64, 32, 3, 3, activation_func);

	g.add_operation(Convolution::new(&entry, &exit, &[5, 5], Padding::Same, "cap_conv", Convolution::init_msra(1.0)));

	g
}

fn add_single_layer(g: &mut Graph, entry: &NodeID, exit: &NodeID,
	up_channels: usize, down_channels: usize, factor: usize, up_conv_size: usize,
	activation_func: &Fn(&NodeID, &NodeID) -> Box<Operation>
	) -> (NodeID, NodeID){

	// Up rail
	let up_conv   = g.add_node(Node::new_shaped(up_channels, 2, "up_conv"));
	let up_activ  = g.add_node(Node::new_shaped(up_channels, 2, "up_activ"));
	
	g.add_operation(Bias::new(&up_conv, ParamSharing::Spatial, "up_conv_bias", init_fill(0.0)));
	g.add_operation(Convolution::new(&entry, &up_conv, &[up_conv_size, up_conv_size], Padding::Same, "up_conv", Convolution::init_msra(1.0)));
	g.add_operation(activation_func(&up_conv, &up_activ));

	
	// Down rail
	let down_conv   = g.add_node(Node::new_shaped(down_channels*factor*factor, 2, "down_conv"));
	let down_activ  = g.add_node(Node::new_shaped(down_channels*factor*factor, 2, "down_activ"));
	
	g.add_operation(Bias::new(&down_conv, ParamSharing::Spatial, "down_conv_bias", init_fill(0.0)));
	g.add_operation(activation_func(&down_conv, &down_activ));
	let down_expand = if factor == 1 {
		down_activ
	} else {
		let down_expand   = g.add_node(Node::new_shaped(down_channels, 2, "down_expand"));
		g.add_operation(Expand::new(&down_activ, &down_expand, &[factor, factor], "down_expand"));
		down_expand
	};
	g.add_operation(Convolution::new(&down_expand, &exit, &[3, 3], Padding::Same, "down_conv", Convolution::init_msra(1.0)));



	// Cross rail
	let cross_conv = g.add_node(Node::new_shaped(down_channels, 2, "cross_conv"));
	g.add_operation(Bias::new(&cross_conv, ParamSharing::Spatial, "cross_conv_bias", init_fill(0.0)));
	g.add_operation(Convolution::new(&up_activ, &cross_conv, &[3, 3], Padding::Same, "cross_conv", Convolution::init_msra(0.1)));
	g.add_operation(activation_func(&cross_conv, &down_expand));


	let up_pool = if factor == 1 {
		up_activ
	} else {
		let up_pool = g.add_node(Node::new_shaped(up_channels, 2, "up_pool"));
		g.add_operation(Pooling::new(&up_activ, &up_pool, &[factor, factor], "up_pool"));
		up_pool
	};

	(up_pool, down_conv)
}



fn upscaler_linear_test(){

	let mut g = upscaler_linear_net(2);
	

	let mut training_set = ImagenetSupplier::<Random>::new(Path::new("D:/ML/ImageNet"), Cropping::Centre{width:120, height:120});

	let start_params = g.init_params();
	let mut opt_params = start_params.clone();

	let mut solver = Asgd::new(&mut g);
	solver.set_max_evals((training_set.epoch_size()*10) as u64);
	
	solver.add_step_callback(|err, step, evaluations, _graph, _params|{
		if step == 1 {println!("");println!("step\tevals\terr");}
		println!("{}\t{}\t{:.5}", step, evaluations, err);
		true
	});

	{ // Add occasional test set evaluation as solver callback

		let mut g2 = upscaler_linear_net(2);
		let mut test_set = ImageFolderSupplier::<Random>::new(Path::new("D:/ML/Set14"), Cropping::None);
		let n = test_set.epoch_size() as usize;

		solver.add_step_callback(move |_err, step, _evaluations, _graph, params|{

			if step % 1000 == 0 {
				print!("Test errors:\t");
				for i in 0..n {

					let (input, mut training_input) = test_set.next_n(1);
					training_input.push(NodeData::new_blank(DataShape::new_flat(1000, 1)));

					let (batch_err, _, _) = g2.backprop(1, input, training_input, params);
					
					print!("{}\t", batch_err)
				}
				println!("");
			}
			if step % 100 == 0 || step == 1{
				let bytes = params.encode::<u32>().unwrap();
				let string = format!("D:/ML/linear{}.par", step);
				let out_path = Path::new(&string);
				let mut out = File::create(out_path).unwrap();

				out.write_all(&bytes).unwrap();
			}
			true
		});

	}


	opt_params = solver.optimise_from(&mut training_set, opt_params.clone());	

	println!("Params moved:{}", opt_params.add_scaled(&start_params, -1.0).norm2());

}

fn upscaler_linear_net(factor: usize) -> Graph{
	let mut g = Graph::new();

	let _dummy_training_node = g.add_training_input_node(Node::new_flat(1000, "label"));
	let input = g.add_input_node(Node::new_shaped(3, 2, "input"));

	let input_linear = g.add_node(Node::new_shaped(3, 2, "input_linear"));
	let input_pool = g.add_node(Node::new_shaped(3, 2, "input_pool"));
	let input_lr = g.add_node(Node::new_shaped(3, 2, "input_lr"));


	let conv1 = g.add_node(Node::new_shaped(3*factor*factor, 2, "conv"));
	let output = g.add_node(Node::new_shaped(3, 2, "output"));
	//let output_srgb = g.add_node(Node::new_shaped(3, 2, "output")); 

	let ops: Vec<Box<Operation>> = vec![
		
		// Downscale
		SrgbToLinear::new(&input, &input_linear, "srgb2lin"),
		Pooling::new(&input_linear, &input_pool, &[factor, factor], "pooling1"),
		LinearToSrgb::new(&input_pool, &input_lr, "lin2srgb"),

		// Upscale
		Convolution::new(&input_lr, &conv1, &[3, 3], Padding::Same, "conv1", init_fill(1.0/27.0)),
		Expand::new(&conv1, &output, &[factor, factor], "expand1"),
		ShapeConstraint::new(&input, &output, &[Arc::new(|d| d), Arc::new(|d| d)], "shape1"),
	

		MseLoss::new(&output, &input, 100.0, "loss"),
	];
	let _op_inds = g.add_operations(ops);

	g
}





