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

	//let args = std::env::args().collect::<Vec<_>>();
	train();
	
	
}

fn train(){

	// type inference seems to crap out unless the type is made explicit
	let activ_func: Box<Fn(&NodeID, &NodeID) -> Box<Operation>> = Box::new(|input, output| BeLU::new(input, output, ParamSharing::Auto, "generic activ", BeLU::init_porque_no_los_dos()));
	
	let mut g = alexnet(true, 1e-6, activ_func.as_ref());

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

	let mut solver = Asgd2::new(&mut g);
	

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
	solver.set_max_evals((training_set.epoch_size()*10) as u64);//(training_set.epoch_size()*10) as u64
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

fn alexnet(training: bool, regularisation: f32, activation_func: &Fn(&NodeID, &NodeID) -> Box<Operation>) -> Graph {
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

// fn upscaler_net(factor: usize, training: bool, regularisation: f32, activation_func: &Fn(&NodeID, &NodeID) -> Box<Operation>) -> Graph {
// 	let mut g = Graph::new();

// 	//-- Input/Output
// 	let (input, output) = if training {
// 		(g.add_node(Node::new_shaped(CHANNELS, 2, "input")), // in training a high resolution image will be the input node, added at the bottom.
// 		g.add_node(Node::new_shaped(CHANNELS, 2, "output")))
// 	} else {
// 		(g.add_input_node(Node::new_shaped(CHANNELS, 2, "input")),
// 		g.add_output_node(Node::new_shaped(CHANNELS, 2, "output")))
// 	};

	

// 	g.add_operation(LinearInterp::new(&input, &output, &[factor, factor], "linterp"));

// 	let mut ops: Vec<Box<Operation>> = vec![];

// 	let f_conv = g.add_node(Node::new_shaped(32, 2, "l1_conv"));
// 	let f_activ = g.add_node(Node::new_shaped(32, 2, "l1_activ"));
// 	ops.push(Convolution::new(&input, &f_conv, &[3, 3], Padding::Same, "conv0", Convolution::init_msra(1.0)));
// 	ops.push(Bias::new(&f_conv, ParamSharing::Spatial, "l1_bias", init_fill(0.0)));
// 	ops.push(activation_func(&f_conv, &f_activ));
	

// 	let expand = g.add_node(Node::new_shaped(CHANNELS*factor*factor, 2, "expand"));
// 	ops.push(Bias::new(&expand, ParamSharing::Spatial, "expand_bias", init_fill(0.0)));
// 	ops.push(Expand::new(&expand, &output, &[factor, factor], "expand"));

	
// 	//g.add_operation(Convolution::new(&f_activ, &expand, &[1, 1], Padding::Same, "conv1", Convolution::init_msra(0.1)));


// 	//-- Low Res DenseNet-like Convolution Connections
// 	for _ in 0..1 {
// 		let n = 32;
// 		let l1_conv = g.add_node(Node::new_shaped(n, 2, "l1_conv"));
// 		let l1_activ = g.add_node(Node::new_shaped(n, 2, "l1_activ"));
// 		let l2_conv = g.add_node(Node::new_shaped(n, 2, "l2_conv"));
// 		let l2_activ = g.add_node(Node::new_shaped(n, 2, "l2_activ"));
// 		let l3_conv = g.add_node(Node::new_shaped(n, 2, "l3_conv"));
// 		let l3_activ = g.add_node(Node::new_shaped(n, 2, "l3_activ"));


// 		ops.push(Bias::new(&l1_conv, ParamSharing::Spatial, "l1_bias", init_fill(0.0)));
// 		ops.push(Bias::new(&l2_conv, ParamSharing::Spatial, "l2_bias", init_fill(0.0)));
// 		ops.push(Bias::new(&l3_conv, ParamSharing::Spatial, "l3_bias", init_fill(0.0)));

// 		ops.push(activation_func(&l1_conv, &l1_activ));
// 		ops.push(activation_func(&l2_conv, &l2_activ));
// 		ops.push(activation_func(&l3_conv, &l3_activ));

// 		ops.push(Convolution::new(&f_activ, &l1_conv, &[5, 5], Padding::Same, "conv1", Convolution::init_msra(0.1)));
// 		ops.push(Convolution::new(&f_activ, &l2_conv, &[5, 5], Padding::Same, "conv2", Convolution::init_msra(0.1)));
// 		ops.push(Convolution::new(&f_activ, &l3_conv, &[5, 5], Padding::Same, "conv3", Convolution::init_msra(0.1)));
// 		//ops.push(Convolution::new(&f_activ, &expand, vec![3, 3], Padding::Same, "conv3", Convolution::init_msra(0.1)));

// 		ops.push(Convolution::new(&l1_activ, &l2_conv, &[3, 3], Padding::Same, "conv5", Convolution::init_msra(0.1)));
// 		ops.push(Convolution::new(&l1_activ, &l3_conv, &[3, 3], Padding::Same, "conv6", Convolution::init_msra(0.1)));
// 		ops.push(Convolution::new(&l1_activ, &expand, &[3, 3], Padding::Same, "conv8", Convolution::init_msra(0.1)));

// 		ops.push(Convolution::new(&l2_activ, &l3_conv, &[3, 3], Padding::Same, "conv9", Convolution::init_msra(0.1)));
// 		ops.push(Convolution::new(&l2_activ, &expand, &[3, 3], Padding::Same, "conv11", Convolution::init_msra(0.1)));

// 		ops.push(Convolution::new(&l3_activ, &expand, &[3, 3], Padding::Same, "conv13", Convolution::init_msra(0.1)));

// 	}

// 	let op_ids = g.add_operations(ops);

// 	if training && regularisation != 0.0 {
// 		for op_id in &op_ids {
// 			if op_id.num_params == 0 {continue};
// 			g.add_secondary_operation(L2Regularisation::new(op_id, regularisation, "L2"), op_id);
// 		}
// 	}


// 	if training {
// 		let _dummy_training_node = g.add_training_input_node(Node::new_flat(1000, "label"));
// 		let input_hr = g.add_input_node(Node::new_shaped(CHANNELS, 2, "input_hr"));
// 		let input_hr_lin = g.add_node(Node::new_shaped(CHANNELS, 2, "input_hr_lin"));
// 		let input_pool = g.add_node(Node::new_shaped(CHANNELS, 2, "input_pool"));
// 		g.add_operation(SrgbToLinear::new(&input_hr, &input_hr_lin,"srgb2lin"));
// 		g.add_operation(Pooling::new(&input_hr_lin, &input_pool, &[factor, factor], "input_pooling"));
// 		g.add_operation(LinearToSrgb::new(&input_pool, &input, "lin2srgb"));

// 		//g.add_operation(Pooling::new(&input_hr, &input, &[factor, factor], "input_pooling"));

// 		g.add_operation(MseLoss::new(&output, &input_hr, 100.0, "loss"));

// 		g.add_operation(ShapeConstraint::new(&input_hr, &output, &[Arc::new(|d| d), Arc::new(|d| d)], "output_shape"));
// 		//g.add_operation(ShapeConstraint::new(&input_hr, &expand_out, &[Arc::new(|d| d), Arc::new(|d| d)], "expand_shape"));	
// 	} else {
// 		g.add_operation(ShapeConstraint::new(&input, &output, &[Arc::new(move|d| d*factor), Arc::new(move|d| d*factor)], "output_shape"));	
// 		//g.add_operation(ShapeConstraint::new(&input, &expand_out, &[Arc::new(move|d| d*factor), Arc::new(move|d| d*factor)], "expand_shape"));
// 	}

// 	g

// }



// fn upscaler_net4(factor: usize, training_net: bool, activation_func: &Fn(&NodeID, &NodeID) -> Box<Operation>) -> Graph {
// 	let mut g = Graph::new();


// 	//-- Input/Output
// 	let (input, output) = if training_net {
// 		(g.add_node(Node::new_shaped(CHANNELS, 2, "input")), // in training a high resolution image will be the input node, added at the bottom.
// 		g.add_node(Node::new_shaped(CHANNELS, 2, "output")))
// 	} else {
// 		(g.add_input_node(Node::new_shaped(CHANNELS, 2, "input")),
// 		g.add_output_node(Node::new_shaped(CHANNELS, 2, "output")))
// 	};
	

// 	//-- Linear Bypass
// 	let input_lin = g.add_node(Node::new_shaped(CHANNELS, 2, "input_lin"));
// 	let upscale_lin = g.add_node(Node::new_shaped(CHANNELS, 2, "upscale_lin"));

// 	g.add_operation(SrgbToLinear::new(&input, &input_lin,"srgb2lin"));
// 	g.add_operation(LinearInterp::new(&input_lin, &upscale_lin, &[factor, factor], "linterp"));
// 	g.add_operation(LinearToSrgb::new(&upscale_lin, &output, "lin2srgb"));
	



// 	//-- Tower Exit Path
// 	let exit_channels = 16;
// 	let exit_conv   = g.add_node(Node::new_shaped(exit_channels*factor*factor, 2, "exit_conv"));
// 	let exit_activ  = g.add_node(Node::new_shaped(exit_channels*factor*factor, 2, "exit_activ"));
// 	let exit_expand = g.add_node(Node::new_shaped(exit_channels, 2, "exit_expand"));

	
// 	g.add_operation(Bias::new(&exit_conv, ParamSharing::Spatial, "exit_conv_bias", init_fill(0.0)));
// 	g.add_operation(activation_func(&exit_conv, &exit_activ));
// 	g.add_operation(Expand::new(&exit_activ, &exit_expand, &[factor, factor], "exit_expand"));
	

// 	g.add_operation(Convolution::new(&exit_expand, &output, &[5, 5], Padding::Same, "output_conv", Convolution::init_msra(1.0)));
// 	g.add_operation(Bias::new(&output, ParamSharing::Spatial, "output_bias", init_fill(0.0)));
	
// 	// let exit_conv = g.add_node(Node::new_shaped(CHANNELS*factor*factor, 2, "exit_conv"));
// 	// g.add_operation(Expand::new(&exit_conv, &output, &[factor, factor], "exit_expand"));
// 	// g.add_operation(Bias::new(&output, ParamSharing::Spatial, "output_bias", init_fill(0.0)));

// 	// more training input stuff
// 	if training_net {
// 		let _dummy_training_node = g.add_training_input_node(Node::new_flat(1000, "label"));
// 		let input_hr = g.add_input_node(Node::new_shaped(CHANNELS, 2, "input_hr"));
// 		let input_hr_lin = g.add_node(Node::new_shaped(CHANNELS, 2, "input_hr_lin"));
// 		let input_pool = g.add_node(Node::new_shaped(CHANNELS, 2, "input_pool"));
// 		g.add_operation(SrgbToLinear::new(&input_hr, &input_hr_lin,"srgb2lin"));
// 		g.add_operation(Pooling::new(&input_hr_lin, &input_pool, &[factor, factor], "input_pooling"));
// 		g.add_operation(LinearToSrgb::new(&input_pool, &input, "lin2srgb"));
// 		//g.add_operation(Pooling::new(&input_hr, &input, &[factor, factor], "input_pooling"));
// 		g.add_operation(MseLoss::new(&output, &input_hr, 100.0, "loss"));

// 		g.add_operation(ShapeConstraint::new(&input_hr, &exit_expand, &[Arc::new(|d| d), Arc::new(|d| d)], "output_shape"));	
// 		g.add_operation(ShapeConstraint::new(&input_hr, &upscale_lin, &[Arc::new(|d| d), Arc::new(|d| d)], "output_shape"));	
// 	} else {
// 		g.add_operation(ShapeConstraint::new(&input, &exit_expand, &[Arc::new(move|d| d*factor), Arc::new(move|d| d*factor)], "output_shape"));
// 		g.add_operation(ShapeConstraint::new(&input, &upscale_lin, &[Arc::new(move|d| d*factor), Arc::new(move|d| d*factor)], "output_shape"));		
// 	}



// 	//-- Tower entry exit points
// 	let entry = input;
// 	let exit = exit_conv;

// 	let (entry, exit) = add_single_layer(&mut g, &entry, &exit, 128, 64, 3, 9, activation_func);
// 	//let (entry, exit) = add_single_layer(&mut g, &entry, &exit, 64, 32, 3, 5, activation_func);
// 	//let (entry, exit) = add_single_layer(&mut g, &entry, &exit, 64, 32, 3, 3, activation_func);

// 	g.add_operation(Convolution::new(&entry, &exit, &[5, 5], Padding::Same, "cap_conv", Convolution::init_msra(1.0)));

// 	g
// }

// fn add_single_layer(g: &mut Graph, entry: &NodeID, exit: &NodeID,
// 	up_channels: usize, down_channels: usize, factor: usize, up_conv_size: usize,
// 	activation_func: &Fn(&NodeID, &NodeID) -> Box<Operation>
// 	) -> (NodeID, NodeID){

// 	// Up rail
// 	let up_conv   = g.add_node(Node::new_shaped(up_channels, 2, "up_conv"));
// 	let up_activ  = g.add_node(Node::new_shaped(up_channels, 2, "up_activ"));
	
// 	g.add_operation(Bias::new(&up_conv, ParamSharing::Spatial, "up_conv_bias", init_fill(0.0)));
// 	g.add_operation(Convolution::new(&entry, &up_conv, &[up_conv_size, up_conv_size], Padding::Same, "up_conv", Convolution::init_msra(1.0)));
// 	g.add_operation(activation_func(&up_conv, &up_activ));

	
// 	// Down rail
// 	let down_conv   = g.add_node(Node::new_shaped(down_channels*factor*factor, 2, "down_conv"));
// 	let down_activ  = g.add_node(Node::new_shaped(down_channels*factor*factor, 2, "down_activ"));
	
// 	g.add_operation(Bias::new(&down_conv, ParamSharing::Spatial, "down_conv_bias", init_fill(0.0)));
// 	g.add_operation(activation_func(&down_conv, &down_activ));
// 	let down_expand = if factor == 1 {
// 		down_activ
// 	} else {
// 		let down_expand   = g.add_node(Node::new_shaped(down_channels, 2, "down_expand"));
// 		g.add_operation(Expand::new(&down_activ, &down_expand, &[factor, factor], "down_expand"));
// 		down_expand
// 	};
// 	g.add_operation(Convolution::new(&down_expand, &exit, &[3, 3], Padding::Same, "down_conv", Convolution::init_msra(1.0)));



// 	// Cross rail
// 	let cross_conv = g.add_node(Node::new_shaped(down_channels, 2, "cross_conv"));
// 	g.add_operation(Bias::new(&cross_conv, ParamSharing::Spatial, "cross_conv_bias", init_fill(0.0)));
// 	g.add_operation(Convolution::new(&up_activ, &cross_conv, &[3, 3], Padding::Same, "cross_conv", Convolution::init_msra(0.1)));
// 	g.add_operation(activation_func(&cross_conv, &down_expand));


// 	let up_pool = if factor == 1 {
// 		up_activ
// 	} else {
// 		let up_pool = g.add_node(Node::new_shaped(up_channels, 2, "up_pool"));
// 		g.add_operation(Pooling::new(&up_activ, &up_pool, &[factor, factor], "up_pool"));
// 		up_pool
// 	};

// 	(up_pool, down_conv)
// }



// fn upscaler_linear_test(){

// 	let mut g = upscaler_linear_net(2);
	

// 	let mut training_set = ImagenetSupplier::<Random>::new(Path::new("D:/ML/ImageNet"), Cropping::Centre{width:120, height:120});

// 	let start_params = g.init_params();
// 	let mut opt_params = start_params.clone();

// 	let mut solver = Asgd::new(&mut g);
// 	solver.set_max_evals((training_set.epoch_size()*10) as u64);
	
// 	solver.add_step_callback(|err, step, evaluations, _graph, _params|{
// 		if step == 1 {println!("");println!("step\tevals\terr");}
// 		println!("{}\t{}\t{:.5}", step, evaluations, err);
// 		true
// 	});

// 	{ // Add occasional test set evaluation as solver callback

// 		let mut g2 = upscaler_linear_net(2);
// 		let mut test_set = ImageFolderSupplier::<Random>::new(Path::new("D:/ML/Set14"), Cropping::None);
// 		let n = test_set.epoch_size() as usize;

// 		solver.add_step_callback(move |_err, step, _evaluations, _graph, params|{

// 			if step % 1000 == 0 {
// 				print!("Test errors:\t");
// 				for i in 0..n {

// 					let (input, mut training_input) = test_set.next_n(1);
// 					training_input.push(NodeData::new_blank(DataShape::new_flat(1000, 1)));

// 					let (batch_err, _, _) = g2.backprop(1, input, training_input, params);
					
// 					print!("{}\t", batch_err)
// 				}
// 				println!("");
// 			}
// 			if step % 100 == 0 || step == 1{
// 				let bytes = params.encode::<u32>().unwrap();
// 				let string = format!("D:/ML/linear{}.par", step);
// 				let out_path = Path::new(&string);
// 				let mut out = File::create(out_path).unwrap();

// 				out.write_all(&bytes).unwrap();
// 			}
// 			true
// 		});

// 	}


// 	opt_params = solver.optimise_from(&mut training_set, opt_params.clone());	

// 	println!("Params moved:{}", opt_params.add_scaled(&start_params, -1.0).norm2());

// }

// fn upscaler_linear_net(factor: usize) -> Graph{
// 	let mut g = Graph::new();

// 	let _dummy_training_node = g.add_training_input_node(Node::new_flat(1000, "label"));
// 	let input = g.add_input_node(Node::new_shaped(3, 2, "input"));

// 	let input_linear = g.add_node(Node::new_shaped(3, 2, "input_linear"));
// 	let input_pool = g.add_node(Node::new_shaped(3, 2, "input_pool"));
// 	let input_lr = g.add_node(Node::new_shaped(3, 2, "input_lr"));


// 	let conv1 = g.add_node(Node::new_shaped(3*factor*factor, 2, "conv"));
// 	let output = g.add_node(Node::new_shaped(3, 2, "output"));
// 	//let output_srgb = g.add_node(Node::new_shaped(3, 2, "output")); 

// 	let ops: Vec<Box<Operation>> = vec![
		
// 		// Downscale
// 		SrgbToLinear::new(&input, &input_linear, "srgb2lin"),
// 		Pooling::new(&input_linear, &input_pool, &[factor, factor], "pooling1"),
// 		LinearToSrgb::new(&input_pool, &input_lr, "lin2srgb"),

// 		// Upscale
// 		Convolution::new(&input_lr, &conv1, &[3, 3], Padding::Same, "conv1", init_fill(1.0/27.0)),
// 		Expand::new(&conv1, &output, &[factor, factor], "expand1"),
// 		ShapeConstraint::new(&input, &output, &[Arc::new(|d| d), Arc::new(|d| d)], "shape1"),
	

// 		MseLoss::new(&output, &input, 100.0, "loss"),
// 	];
// 	let _op_inds = g.add_operations(ops);

// 	g
// }





