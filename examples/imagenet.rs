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
use alumina::opt::cain::*;
use alumina::ops::*;

use alumina::graph::*;	
use alumina::vec_math::*;
use alumina::shape::*;

use alumina::supplier::*;
use alumina::supplier::imagenet::ImagenetSupplier;
use alumina::supplier::imagefolder::{ImageFolderSupplier, Cropping, CHANNELS, img_to_data, data_to_img};


use std::io::{self, Write, Read};
use std::sync::Arc;
use std::path::{PathBuf, Path};
use std::fs::File;

fn main(){

	train();

}

fn train(){

	// type inference seems to crap out unless the type is made explicit
	let activ_func: Box<Fn(&NodeID, &NodeID) -> Box<Operation>> = Box::new(|input, output| BeLU::new(input, output, ParamSharing::Auto, "generic activ", BeLU::init_porque_no_los_dos()));
	
	let mut g = small_alexnet(true, 1e-6, activ_func.as_ref());

	let mut training_set = ImagenetSupplier::<Random>::new(Path::new("D:/ML/ImageNet"), Cropping::Random{width:227, height:227});

	let args = std::env::args().collect::<Vec<_>>();
	let start_params = if args.len() > 1 {
		let mut param_file = File::open(Path::new(&args[2])).expect("Error opening parameter file");
		let mut data = Vec::new();
		param_file.read_to_end(&mut data);
		<Vec<f32>>::decode::<u32>(&data).unwrap()
	} else {
		g.init_params()
	};

	//let start_params = g.init_params();
	let mut opt_params = start_params.clone();

	let mut solver = Cain::new(&mut g)
		.initial_learning_rate(1e-3)
		.finish();
	

	// { // Add occasional test set evaluation as solver callback
	// 	let mut g2 = upscaler_net(3, true, 0.0, activ_func.as_ref());
	// 	let mut test_set = ImageFolderSupplier::<Random>::new(Path::new("C:/Set14"), Cropping::None);
	// 	let n = test_set.epoch_size() as usize;

	// 	solver.add_step_callback(move |_err, step, _evaluations, _graph, params|{

	// 		if step % 100 == 0 {
	// 			print!("Test errors:\t");
	// 			for i in 0..n {

	// 				let (input, mut training_input) = test_set.next_n(1);
	// 				training_input.push(NodeData::new_blank(DataShape::new_flat(1000, 1)));

	// 				let (batch_err, _, _) = g2.backprop(1, input, training_input, params);
					
	// 				print!("{}\t", batch_err)
	// 			}
	// 			println!("");
	// 		}

	// 		if step % 100 == 0 || step == 1{
	// 			let bytes = params.encode::<u32>().unwrap();
	// 			let string = format!("D:/ML/{}.par", step);
	// 			let out_path = Path::new(&string);
	// 			let mut out = File::create(out_path).unwrap();

	// 			out.write_all(&bytes).unwrap();
	// 		}


	// 		true
	// 	});
	// }

	//solver.set_min_batch_size(2.);
	solver.add_boxed_step_callback(max_evals(10 * training_set.epoch_size()));
	opt_params = solver.optimise_from(&mut training_set, opt_params.clone());	

	println!("Params moved:{}", opt_params.add_scaled(&start_params, -1.0).norm2());

}

// fn darknet(training: bool, regularisation: f32, activation_func: &Fn(&NodeID, &NodeID) -> Box<Operation>) -> Graph {
// 	let mut g = Graph::new();

// 	let mut ops: Vec<Box<Operation>> = vec![];

// 	let target = g.add_training_input_node(Node::new_flat(1000, "target"));
// 	let input = g.add_input_node(Node::new_shaped(CHANNELS, 2, "input")); // 227x227

// 	// Layers 0 + 1
// 	let n0 = 16;
// 	let l0_conv = g.add_node(Node::new_shaped(n1, 2, "l1a_conv")); // 217x217 Valid padding
// 	let l0_activ = g.add_node(Node::new_shaped(n1, 2, "l1a_activ")); // 217x217
// 	let l0_pool= g.add_node(Node::new_sized(n1, &[55, 55], "l1a_pool")); // 55x55
// 	ops.push(Convolution::new(&input, &l1a_conv, &[3, 3], Padding::Valid, "l1a_conv", Convolution::init_msra(1.0)));
// 	ops.push(Bias::new(&l1a_conv, ParamSharing::Spatial, "l1a_bias", init_fill(0.0)));
// 	ops.push(activation_func(&l1a_conv, &l1a_activ));
// 	ops.push(Pooling::new(&l1a_activ, &l1a_pool, &[2, 2], "l1a_pool"));

// 	// Layers 2 + 3
// 	let n2 = 32;
// 	let l2_conv = g.add_node(Node::new_shaped(n2, 2, "l2a_conv")); //51x51 Valid padding
// 	let l2_activ = g.add_node(Node::new_shaped(n2, 2, "l2a_activ")); //51x51
// 	let l2_pool = g.add_node(Node::new_sized(n2, &[26,26], "l2a_pool")); // 26x26
// 	ops.push(Convolution::new(&l1a_pool, &l2a_conv, &[3, 3], Padding::Valid, "l2a_conv", Convolution::init_msra(1.0)));
// 	ops.push(Bias::new(&l2a_conv, ParamSharing::Spatial, "l2a_bias", init_fill(0.0)));
// 	ops.push(activation_func(&l2a_conv, &l2a_activ));
// 	ops.push(Pooling::new(&l2a_activ, &l2a_pool, &[2, 2], "l2a_pool"));

// 	// Layer 4
// 	let n4 = 16;
// 	let l4_conv = g.add_node(Node::new_shaped(n4, 2, "l4_conv")); //26x26 Same padding
// 	let l4_activ = g.add_node(Node::new_shaped(n4, 2, "l4_activ")); //26x26
// 	ops.push(Convolution::new(&l2a_pool, &l3a_conv, &[3, 3], Padding::Same, "l4_conv", Convolution::init_msra(0.5)));
// 	ops.push(Bias::new(&l3a_conv, ParamSharing::Spatial, "l4_bias", init_fill(0.0)));
// 	ops.push(activation_func(&l4_conv, &l4_activ));

// 	// Layer 5
// 	let n4 = 16;
// 	let l4a_conv = g.add_node(Node::new_shaped(n3, 2, "l3a_conv")); //26x26 Same padding
// 	let l4a_activ = g.add_node(Node::new_shaped(n3, 2, "l3a_activ")); //26x26
// 	ops.push(Convolution::new(&l2a_pool, &l3a_conv, &[3, 3], Padding::Same, "l3a_conv", Convolution::init_msra(0.5)));
// 	ops.push(Bias::new(&l3a_conv, ParamSharing::Spatial, "l3a_bias", init_fill(0.0)));
// 	ops.push(activation_func(&l3a_conv, &l3a_activ));








// 	let n6 = 4096;
// 	let l6_fc = g.add_node(Node::new_flat(n6, "l6_fc"));
// 	let l6_activ = g.add_node(Node::new_flat(n6, "l6_activ"));
// 	ops.push(LinearMap::new(&l5a_activ, &l6_fc, "l6_fca", LinearMap::init_msra(0.5)));
// 	ops.push(LinearMap::new(&l5b_activ, &l6_fc, "l6_fcb", LinearMap::init_msra(0.5)));
// 	ops.push(Bias::new(&l6_fc, ParamSharing::None, "l6_bias", init_fill(0.0)));
// 	ops.push(activation_func(&l6_fc, &l6_activ));

// 	let n7 = 4096;
// 	let l7_fc = g.add_node(Node::new_flat(n7, "l7_fc"));
// 	let l7_activ = g.add_node(Node::new_flat(n7, "l7_activ"));
// 	ops.push(LinearMap::new(&l6_activ, &l7_fc, "l7_fca", LinearMap::init_msra(1.0)));
// 	ops.push(Bias::new(&l7_fc, ParamSharing::None, "l7_bias", init_fill(0.0)));
// 	ops.push(activation_func(&l7_fc, &l7_activ));

// 	let output_fc = g.add_node(Node::new_flat(1000, "output_fc"));
// 	ops.push(LinearMap::new(&l7_activ, &output_fc, "output_fc", LinearMap::init_msra(1.0)));
// 	ops.push(Bias::new(&output_fc, ParamSharing::None, "output_bias", init_fill(0.0)));

// 	if training {
// 		ops.push(SoftMaxCrossEntLoss::new(&output_fc, &target, 1.0, ""));
// 	} else {
// 		let output = g.add_output_node(Node::new_flat(1000, "output"));
// 		ops.push(SoftMax::new(&output_fc, &output, "softmax"));
// 	}

// 	let op_ids = g.add_operations(ops);


// 	if regularisation != 0.0 {
// 		for op_id in &op_ids {
// 			if op_id.num_params == 0 {continue};
// 			g.add_secondary_operation(L2Regularisation::new(op_id, regularisation, "L2"), op_id);
// 		}
// 	}


// 	g

// }

fn small_alexnet(training: bool, regularisation: f32, activation_func: &Fn(&NodeID, &NodeID) -> Box<Operation>) -> Graph {
	let mut g = Graph::new();

	let mut ops: Vec<Box<Operation>> = vec![];

	
	let input = g.add_input_node(Node::new_shaped(CHANNELS, 2, "input")); // 227x227
	let input_lr = g.add_node(Node::new_shaped(CHANNELS, 2, "input_lr"));
	ops.push(Pooling::new(&input, &input_lr, &[2,2], "downscale")); // my computer pukes at the memory requirements

	let target = g.add_training_input_node(Node::new_flat(1000, "target"));

	let n1 = 24; //48
	let l1a_conv = g.add_node(Node::new_shaped(n1, 2, "l1a_conv")); // 217x217 Valid padding
	let l1a_activ = g.add_node(Node::new_shaped(n1, 2, "l1a_activ")); // 217x217
	let l1a_pool= g.add_node(Node::new_shaped(n1, 2, "l1a_pool")); // 55x55
	let l1b_conv = g.add_node(Node::new_shaped(n1, 2, "l1b_conv"));
	let l1b_activ = g.add_node(Node::new_shaped(n1, 2, "l1b_activ"));
	let l1b_pool= g.add_node(Node::new_shaped(n1, 2, "l1b_pool")); 
	ops.push(Convolution::new(&input_lr, &l1a_conv, &[11, 11], Padding::Valid, "l1a_conv", Convolution::init_msra(1.0)));
	ops.push(Convolution::new(&input_lr, &l1b_conv, &[11, 11], Padding::Valid, "l1b_conv", Convolution::init_msra(1.0)));
	ops.push(Bias::new(&l1a_conv, ParamSharing::Spatial, "l1a_bias", init_fill(0.0)));
	ops.push(Bias::new(&l1b_conv, ParamSharing::Spatial, "l1b_bias", init_fill(0.0)));
	ops.push(activation_func(&l1a_conv, &l1a_activ));
	ops.push(activation_func(&l1b_conv, &l1b_activ));
	ops.push(Pooling::new(&l1a_activ, &l1a_pool, &[4, 4], "l1a_pool"));
	ops.push(Pooling::new(&l1b_activ, &l1b_pool, &[4, 4], "l1b_pool"));

	let n2 = 64;//128
	let l2a_conv = g.add_node(Node::new_shaped(n2, 2, "l2a_conv")); //51x51 Valid padding
	let l2a_activ = g.add_node(Node::new_shaped(n2, 2, "l2a_activ")); //51x51
	let l2a_pool = g.add_node(Node::new_shaped(n2, 2, "l2a_pool")); // 26x26
	let l2b_conv = g.add_node(Node::new_shaped(n2, 2, "l2b_conv"));
	let l2b_activ = g.add_node(Node::new_shaped(n2, 2, "l2b_activ"));
	let l2b_pool = g.add_node(Node::new_shaped(n2, 2, "l2b_pool"));
	ops.push(Convolution::new(&l1a_pool, &l2a_conv, &[5, 5], Padding::Same, "l2a_conv", Convolution::init_msra(1.0)));
	ops.push(Convolution::new(&l1b_pool, &l2b_conv, &[5, 5], Padding::Same, "l2b_conv", Convolution::init_msra(1.0)));
	ops.push(Bias::new(&l2a_conv, ParamSharing::Spatial, "l2a_bias", init_fill(0.0)));
	ops.push(Bias::new(&l2b_conv, ParamSharing::Spatial, "l2b_bias", init_fill(0.0)));
	ops.push(activation_func(&l2a_conv, &l2a_activ));
	ops.push(activation_func(&l2b_conv, &l2b_activ));
	ops.push(Pooling::new(&l2a_activ, &l2a_pool, &[2, 2], "l2a_pool"));
	ops.push(Pooling::new(&l2b_activ, &l2b_pool, &[2, 2], "l2b_pool"));

	let n3 = 96; //192
	let l3a_conv = g.add_node(Node::new_shaped(n3, 2, "l3a_conv")); //26x26 Same padding
	let l3a_activ = g.add_node(Node::new_shaped(n3, 2, "l3a_activ")); //26x26
	//let l3a_pool = g.add_node(Node::new_sized(n3, &[13, 13], "l3a_pool")); //13x13
	let l3b_conv = g.add_node(Node::new_shaped(n3, 2, "l3b_conv"));
	let l3b_activ = g.add_node(Node::new_shaped(n3, 2, "l3b_activ"));
	//let l3b_pool = g.add_node(Node::new_sized(n3, &[13, 13], "l3b_pool"));
	ops.push(Convolution::new(&l2a_pool, &l3a_conv, &[3, 3], Padding::Same, "l3a_conv", Convolution::init_msra(0.5)));
	ops.push(Convolution::new(&l2b_pool, &l3b_conv, &[3, 3], Padding::Same, "l3b_conv", Convolution::init_msra(0.5)));
	ops.push(Convolution::new(&l2b_pool, &l3a_conv, &[3, 3], Padding::Same, "l3a_conv", Convolution::init_msra(0.5)));
	ops.push(Convolution::new(&l2a_pool, &l3b_conv, &[3, 3], Padding::Same, "l3b_conv", Convolution::init_msra(0.5)));
	ops.push(Bias::new(&l3a_conv, ParamSharing::Spatial, "l3a_bias", init_fill(0.0)));
	ops.push(Bias::new(&l3b_conv, ParamSharing::Spatial, "l3b_bias", init_fill(0.0)));
	ops.push(activation_func(&l3a_conv, &l3a_activ));
	ops.push(activation_func(&l3b_conv, &l3b_activ));
	//ops.push(Pooling::new(&l3a_activ, &l3a_pool, &[2, 2], "l3a_pool"));
	//ops.push(Pooling::new(&l3b_activ, &l3b_pool, &[2, 2], "l3b_pool"));

	let n4 = 96; //192
	let l4a_conv = g.add_node(Node::new_shaped(n4, 2, "l4a_conv")); //13x13
	let l4a_activ = g.add_node(Node::new_shaped(n4, 2, "l4a_activ")); //13x13
	let l4b_conv = g.add_node(Node::new_shaped(n4, 2, "l4b_conv"));
	let l4b_activ = g.add_node(Node::new_shaped(n4, 2, "l4b_activ"));
	ops.push(Convolution::new(&l3a_activ, &l4a_conv, &[3, 3], Padding::Same, "l4a_conv", Convolution::init_msra(1.0)));
	ops.push(Convolution::new(&l3b_activ, &l4b_conv, &[3, 3], Padding::Same, "l4b_conv", Convolution::init_msra(1.0)));
	ops.push(Bias::new(&l4a_conv, ParamSharing::Spatial, "l4a_bias", init_fill(0.0)));
	ops.push(Bias::new(&l4b_conv, ParamSharing::Spatial, "l4b_bias", init_fill(0.0)));
	ops.push(activation_func(&l4a_conv, &l4a_activ));
	ops.push(activation_func(&l4b_conv, &l4b_activ));

	let n5 = 64; //128
	let l5a_conv = g.add_node(Node::new_shaped(n5, 2, "l5a_conv")); //13x13
	let l5a_activ = g.add_node(Node::new_sized(n5, &[13,13], "l5a_activ")); //13x13
	let l5b_conv = g.add_node(Node::new_shaped(n5, 2, "l5b_conv"));
	let l5b_activ = g.add_node(Node::new_sized(n5, &[13,13], "l5b_activ"));
	ops.push(Convolution::new(&l4a_activ, &l5a_conv, &[3, 3], Padding::Same, "l5a_conv", Convolution::init_msra(1.0)));
	ops.push(Convolution::new(&l4b_activ, &l5b_conv, &[3, 3], Padding::Same, "l5b_conv", Convolution::init_msra(1.0)));
	ops.push(Bias::new(&l5a_conv, ParamSharing::Spatial, "l5a_bias", init_fill(0.0)));
	ops.push(Bias::new(&l5b_conv, ParamSharing::Spatial, "l5b_bias", init_fill(0.0)));
	ops.push(activation_func(&l5a_conv, &l5a_activ));
	ops.push(activation_func(&l5b_conv, &l5b_activ));

	let n6 = 2048; //4096
	let l6_fc = g.add_node(Node::new_flat(n6, "l6_fc"));
	let l6_activ = g.add_node(Node::new_flat(n6, "l6_activ"));
	ops.push(LinearMap::new(&l5a_activ, &l6_fc, "l6_fca", LinearMap::init_msra(0.5)));
	ops.push(LinearMap::new(&l5b_activ, &l6_fc, "l6_fcb", LinearMap::init_msra(0.5)));
	ops.push(Bias::new(&l6_fc, ParamSharing::None, "l6_bias", init_fill(0.0)));
	ops.push(activation_func(&l6_fc, &l6_activ));

	let n7 = 2048; //4096
	let l7_fc = g.add_node(Node::new_flat(n7, "l7_fc"));
	let l7_activ = g.add_node(Node::new_flat(n7, "l7_activ"));
	ops.push(LinearMap::new(&l6_activ, &l7_fc, "l7_fca", LinearMap::init_msra(1.0)));
	ops.push(Bias::new(&l7_fc, ParamSharing::None, "l7_bias", init_fill(0.0)));
	ops.push(activation_func(&l7_fc, &l7_activ));

	let output_fc = g.add_node(Node::new_flat(1000, "output_fc"));
	ops.push(LinearMap::new(&l7_activ, &output_fc, "output_fc", LinearMap::init_msra(1.0)));
	ops.push(Bias::new(&output_fc, ParamSharing::None, "output_bias", init_fill(0.0)));

	if training {
		ops.push(SoftMaxCrossEntLoss::new(&output_fc, &target, 1.0, "Loss"));
	} else {
		let output = g.add_output_node(Node::new_flat(1000, "output"));
		ops.push(SoftMax::new(&output_fc, &output, "softmax"));
	}

	let op_ids = g.add_operations(ops);


	if regularisation != 0.0 {
		for op_id in &op_ids {
			if op_id.num_params == 0 {continue};
			g.add_secondary_operation(L2Regularisation::new(op_id, regularisation, "L2"), op_id);
		}
	}


	g

}




