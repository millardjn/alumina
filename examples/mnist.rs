
extern crate alumina;

use std::path::Path;

// This would normally be a single glob import, but broken out for clarity.
use alumina::ops;
use alumina::ops::Operation;
use alumina::ops::activ;
use alumina::ops::basic;	
use alumina::ops::loss;
use alumina::ops::conv;
use alumina::ops::reshape;


use alumina::opt;
use alumina::opt::{Optimiser, CallbackData, CallbackSignal, max_evals};
use alumina::supplier::{Supplier, ShuffleRandom, Sequential};
use alumina::supplier::mnist::MnistSupplier;
use alumina::vec_math::*;
use alumina::graph::*;


const MNIST_PATH: &'static str = "D:/ML/Mnist"; // This folder should have the mnist binary data files for training and testing.

fn main(){

	let (mut g, pred, label) = mnist_lenet(1.0e-3);
	let train_loss = loss::SoftMaxCrossEntLoss::new_default(&pred, &label);
	g.add_operation(train_loss);

	let mut training_set = MnistSupplier::<ShuffleRandom>::training(Path::new(MNIST_PATH));

	let start_params = g.init_params();
	// let mut solver = opt::cain::Cain::new(&mut g)
	// 	.num_subbatches(8)
	// 	//.target_err(0.75)
	// 	.target_err(0.9)
	// 	.subbatch_increase_damping(0.15)
	// 	.subbatch_decrease_damping(0.15)
	// 	.aggression(0.75)
	// 	.momentum(0.9)
	// 	.initial_learning_rate(3e-3)
	// 	.finish();

	let mut solver = opt::adam::Adam::new(&mut g)
	.learning_rate(1e-3)
	.beta1(0.9)
	.beta2(0.999)
	.batch_size(32)
	.finish();

	println!("Total num params: {}", start_params.len());

	solver.add_boxed_step_callback(validation(&start_params, training_set.epoch_size()));
	solver.add_boxed_step_callback(max_evals(50 * training_set.epoch_size()));
	
	let _opt_params = solver.optimise_from(&mut training_set, start_params);


}


fn validation(start_params: &[f32], training_set_size: usize) -> Box<FnMut(&CallbackData)->CallbackSignal>{
	let mut val_set = MnistSupplier::<Sequential>::testing(Path::new(MNIST_PATH)); // I mean, who doesnt validate on the test set /s
	let (mut g2, pred2, label2) = mnist_lenet(0.0);
	let val_loss = loss::PredictionLoss::new_default(&pred2, &label2);
	g2.add_operation(val_loss);

	let start_params = start_params.to_vec();

	opt::every_n_evals(training_set_size, Box::new(move |data: &CallbackData|{
		println!("Params moved:{}", data.params.add_scaled(&start_params, -1.0).norm2());
		
		let mut n = val_set.epoch_size();
		let count = n/256;
		let mut err = 0.0;
		
		for i in 0..count {
			let batch_size = n/(count - i);

			let (input, training_input) = val_set.next_n(batch_size);
			let (batch_err, _, _) = g2.backprop(batch_size, input, training_input, data.params);
			err += batch_err;
			n -= batch_size;
		}

		println!("Validation error is: {}%", 100.0*err/val_set.epoch_size() as f32);
		CallbackSignal::Continue
	}))

}


/// Based on LeNet variant as descripted at http://luizgh.github.io/libraries/2015/12/08/getting-started-with-lasagne/
/// Activation not specified so using BeLU
fn mnist_lenet2(regularise: f32) -> (Graph, NodeID, NodeID){
	let mut g = Graph::new();

	let input = g.add_input_node(Node::new_sized(1, &[28,28], "input"));

	let ch1 = 6;
	let layer1 = g.add_node(Node::new_shaped(ch1, 2, "layer1"));
	let layer1_activ = g.add_node(Node::new_shaped(ch1, 2, "layer1_activ"));
	let layer1_pool = g.add_node(Node::new_sized(ch1, &[14, 14], "layer1_pool"));

	let ch2 = 10;
	let layer2 = g.add_node(Node::new_shaped(ch2, 2, "layer2"));
	let layer2_activ = g.add_node(Node::new_shaped(ch2, 2, "layer2_activ"));
	let layer2_pool = g.add_node(Node::new_sized(ch2, &[7, 7],"layer2_pool")); // changed from [7,7] to [5,5] to reduce parameters/overfitting

	let ch3 = 32; // changed from 100 to 64 to reduce parameters/overfitting
	let layer3 = g.add_node(Node::new_flat(ch3, "layer3"));
	let layer3_activ = g.add_node(Node::new_flat(ch3, "layer3_activ"));	


	let pred = g.add_node(Node::new_flat(10, "prediction"));
	let label = g.add_training_input_node(Node::new_flat(10, "training_label"));
	

	let ops: Vec<Box<Operation>> = vec![
		
		conv::Convolution::new(&input, &layer1, &[5, 5], conv::Padding::Same, "conv1", conv::Convolution::init_msra(1.0)),
		basic::Bias::new(&layer1, ops::ParamSharing::Spatial, "bias1", ops::init_fill(0.0)),
		activ::BeLU::new(&layer1, &layer1_activ, ops::ParamSharing::Spatial, "activation1", activ::BeLU::init_porque_no_los_dos()),
		reshape::Pooling::new(&layer1_activ, &layer1_pool, &[2, 2], "pooling1"),

		conv::Convolution::new(&layer1_pool, &layer2, &[5, 5], conv::Padding::Same, "conv2", conv::Convolution::init_msra(1.0)),
		basic::Bias::new(&layer2, ops::ParamSharing::Spatial, "bias2", ops::init_fill(0.0)),
		activ::BeLU::new(&layer2, &layer2_activ, ops::ParamSharing::Spatial, "activation2", activ::BeLU::init_porque_no_los_dos()),
		reshape::Pooling::new(&layer2_activ, &layer2_pool, &[2, 2], "pooling2"), // downscale by [3,3] instead of [2,2] to reduce parameters/overfitting

		basic::LinearMap::new(&layer2_pool, &layer3, "dense1", basic::LinearMap::init_msra(1.0)),
		basic::Bias::new(&layer3, ops::ParamSharing::None, "bias_dense1", ops::init_fill(0.0)),
		activ::BeLU::new(&layer3, &layer3_activ, ops::ParamSharing::None, "activation2", activ::BeLU::init_porque_no_los_dos()),

		basic::LinearMap::new(&layer3_activ, &pred, "dense2", basic::LinearMap::init_msra(1.0)),
		basic::Bias::new(&pred, ops::ParamSharing::None, "bias_dense2", ops::init_fill(0.0)),
	];
	let op_inds = g.add_operations(ops);

	if regularise != 0.0 {
		for op_ind in &op_inds {
			g.add_secondary_operation(basic::L2Regularisation::new(op_ind, regularise, "L2"), op_ind);
		}
	}

	(g, pred, label)
}

/// Based on LeNet variant as descripted at http://luizgh.github.io/libraries/2015/12/08/getting-started-with-lasagne/
/// Activation not specified so using BeLU
fn mnist_lenetd(regularise: f32) -> (Graph, NodeID, NodeID){
	let mut g = Graph::new();

	let input = g.add_input_node(Node::new_sized(1, &[28,28], "input"));

	let ch1 = 16;
	let layer1 = g.add_node(Node::new_shaped(ch1, 2, "layer1"));
	let layer1_activ = g.add_node(Node::new_shaped(ch1, 2, "layer1_activ"));
	let layer1_pool = g.add_node(Node::new_sized(ch1, &[14, 14], "layer1_pool"));

	let ch2 = 16;
	let layer2a = g.add_node(Node::new_shaped(ch2, 2, "layer2a"));
	let layer2a_activ = g.add_node(Node::new_shaped(ch2, 2, "layer2a_activ"));
	let layer2b = g.add_node(Node::new_shaped(ch2, 2, "layer2b"));
	let layer2b_activ = g.add_node(Node::new_shaped(ch2, 2, "layer2b_activ"));
	let layer2_pool = g.add_node(Node::new_sized(ch2, &[7, 7],"layer2_pool"));

	let ch3 = 8;
	let layer3a = g.add_node(Node::new_shaped(ch3, 2, "layer3"));
	let layer3a_activ = g.add_node(Node::new_shaped(ch3, 2, "layer3_activ"));
	let layer3b = g.add_node(Node::new_shaped(ch3, 2, "layer3"));
	let layer3b_activ = g.add_node(Node::new_shaped(ch3, 2, "layer3_activ"));
	let layer3_pool = g.add_node(Node::new_sized(ch3, &[4, 4],"layer3_pool"));

	let pred = g.add_node(Node::new_flat(10, "prediction"));
	let label = g.add_training_input_node(Node::new_flat(10, "training_label"));
	

	let ops: Vec<Box<Operation>> = vec![
		
		conv::Convolution::new(&input, &layer1, &[5, 5], conv::Padding::Same, "conv1", conv::Convolution::init_msra(2.0)),
		basic::Bias::new(&layer1, ops::ParamSharing::Spatial, "bias1", ops::init_fill(0.0)),
		activ::BeLU::new(&layer1, &layer1_activ, ops::ParamSharing::Spatial, "activation1", activ::BeLU::init_porque_no_los_dos()),
		reshape::Pooling::new(&layer1_activ, &layer1_pool, &[2, 2], "pooling1"),


		conv::Convolution::new(&layer1_pool, &layer2a, &[3, 3], conv::Padding::Same, "conv2", conv::Convolution::init_msra(2.0)),
		basic::Bias::new(&layer2a, ops::ParamSharing::Spatial, "bias2", ops::init_fill(0.0)),
		activ::BeLU::new(&layer2a, &layer2a_activ, ops::ParamSharing::Spatial, "activation2", activ::BeLU::init_porque_no_los_dos()),
		
		conv::Convolution::new(&layer2a, &layer2b, &[3, 3], conv::Padding::Same, "conv2", conv::Convolution::init_msra(1.0)),
		conv::Convolution::new(&layer1_pool, &layer2b, &[3, 3], conv::Padding::Same, "conv2", conv::Convolution::init_msra(1.0)),
		basic::Bias::new(&layer2b, ops::ParamSharing::Spatial, "bias2", ops::init_fill(0.0)),
		activ::BeLU::new(&layer2b, &layer2b_activ, ops::ParamSharing::Spatial, "activation2", activ::BeLU::init_porque_no_los_dos()),

		reshape::Pooling::new(&layer2a_activ, &layer2_pool, &[2, 2], "pooling2a"),
		reshape::Pooling::new(&layer2b_activ, &layer2_pool, &[2, 2], "pooling2b"),


		conv::Convolution::new(&layer2_pool, &layer3a, &[3, 3], conv::Padding::Same, "conv3", conv::Convolution::init_msra(2.0)),
		basic::Bias::new(&layer3a, ops::ParamSharing::Spatial, "bias3", ops::init_fill(0.0)),
		activ::BeLU::new(&layer3a, &layer3a_activ, ops::ParamSharing::Spatial, "activation3", activ::BeLU::init_porque_no_los_dos()),

		conv::Convolution::new(&layer3a, &layer3b, &[3, 3], conv::Padding::Same, "conv3", conv::Convolution::init_msra(1.0)),
		conv::Convolution::new(&layer2_pool, &layer3b, &[3, 3], conv::Padding::Same, "conv3", conv::Convolution::init_msra(1.0)),
		basic::Bias::new(&layer3b, ops::ParamSharing::Spatial, "bias3", ops::init_fill(0.0)),
		activ::BeLU::new(&layer3b, &layer3b_activ, ops::ParamSharing::Spatial, "activation3", activ::BeLU::init_porque_no_los_dos()),

		reshape::Pooling::new(&layer3a_activ, &layer3_pool, &[2, 2], "pooling3a"),
		reshape::Pooling::new(&layer3b_activ, &layer3_pool, &[2, 2], "pooling3b"),
		

		basic::LinearMap::new(&layer3_pool, &pred, "dense2", basic::LinearMap::init_msra(1.0)),
		basic::Bias::new(&pred, ops::ParamSharing::None, "bias_dense2", ops::init_fill(0.0)),
	];
	let op_inds = g.add_operations(ops);

	if regularise != 0.0 {
		for op_ind in &op_inds {
			g.add_secondary_operation(basic::L2Regularisation::new(op_ind, regularise, "L2"), op_ind);
		}
	}

	(g, pred, label)
}

/// Based on LeNet variant as descripted at http://luizgh.github.io/libraries/2015/12/08/getting-started-with-lasagne/
/// Activation not specified so using BeLU
fn mnist_lenet(regularise: f32) -> (Graph, NodeID, NodeID){
	let mut g = Graph::new();

	let input = g.add_input_node(Node::new_sized(1, &[28,28], "input"));

	let ch1 = 16;
	let layer1 = g.add_node(Node::new_shaped(ch1, 2, "layer1"));
	let layer1_activ = g.add_node(Node::new_shaped(ch1, 2, "layer1_activ"));
	let layer1_pool = g.add_node(Node::new_sized(ch1, &[14, 14], "layer1_pool"));

	let ch2 = 32;
	let layer2 = g.add_node(Node::new_shaped(ch2, 2, "layer2"));
	let layer2_activ = g.add_node(Node::new_shaped(ch2, 2, "layer2_activ"));
	let layer2_pool = g.add_node(Node::new_sized(ch2, &[7, 7],"layer2_pool"));

	let ch3 = 16;
	let layer3 = g.add_node(Node::new_shaped(ch3, 2, "layer3"));
	let layer3_activ = g.add_node(Node::new_shaped(ch3, 2, "layer3_activ"));
	let layer3_pool = g.add_node(Node::new_sized(ch3, &[4, 4],"layer3_pool"));

	let pred = g.add_node(Node::new_flat(10, "prediction"));
	let label = g.add_training_input_node(Node::new_flat(10, "training_label"));
	

	let mut ops: Vec<Box<Operation>> = vec![];
	let mut bias_ops: Vec<Box<Operation>> = vec![];
	ops.push(conv::Convolution::new(&input, &layer1, &[7, 7], conv::Padding::Same, "conv1", conv::Convolution::init_msra(2.0)));
	bias_ops.push(basic::Bias::new(&layer1, ops::ParamSharing::Spatial, "bias1", ops::init_fill(0.0)));
	ops.push(activ::BeLU::new(&layer1, &layer1_activ, ops::ParamSharing::Spatial, "activation1", activ::BeLU::init_porque_no_los_dos()));
	ops.push(reshape::Pooling::new(&layer1_activ, &layer1_pool, &[2, 2], "pooling1"));

	ops.push(conv::Convolution::new(&layer1_pool, &layer2, &[3, 3], conv::Padding::Same, "conv2", conv::Convolution::init_msra(2.0)));
	bias_ops.push(basic::Bias::new(&layer2, ops::ParamSharing::Spatial, "bias2", ops::init_fill(0.0)));
	ops.push(activ::BeLU::new(&layer2, &layer2_activ, ops::ParamSharing::Spatial, "activation2", activ::BeLU::init_porque_no_los_dos()));
	ops.push(reshape::Pooling::new(&layer2_activ, &layer2_pool, &[2, 2], "pooling2"));

	ops.push(conv::Convolution::new(&layer2_pool, &layer3, &[3, 3], conv::Padding::Same, "conv3", conv::Convolution::init_msra(2.0)));
	bias_ops.push(basic::Bias::new(&layer3, ops::ParamSharing::Spatial, "bias3", ops::init_fill(0.0)));
	ops.push(activ::BeLU::new(&layer3, &layer3_activ, ops::ParamSharing::Spatial, "activation3", activ::BeLU::init_porque_no_los_dos()));
	ops.push(reshape::Pooling::new(&layer3_activ, &layer3_pool, &[2, 2], "pooling3"));

	ops.push(basic::LinearMap::new(&layer3_pool, &pred, "dense2", basic::LinearMap::init_msra(1.0)));
	bias_ops.push(basic::Bias::new(&pred, ops::ParamSharing::None, "bias_dense2", ops::init_fill(0.0)));
	
	let op_inds = g.add_operations(ops);
	let bias_op_inds = g.add_operations(bias_ops);

	if regularise != 0.0 {
		for op_ind in &op_inds {
			g.add_secondary_operation(basic::L2Regularisation::new(op_ind, regularise, "L2"), op_ind);
		}
		for op_ind in &bias_op_inds {
			g.add_secondary_operation(basic::L2Regularisation::new(op_ind, regularise*0.001, "L2"), op_ind);
		}
	}

	(g, pred, label)
}

/// Mnist network used for benchmarking optimisers in the ADAM paper: https://arxiv.org/pdf/1412.6980v8.pdf
fn mnist_adam(regularise: f32) -> (Graph, NodeID, NodeID){
	let mut g = Graph::new();

	let input = g.add_input_node(Node::new_sized(1, &[28,28], "input"));

	let layer1 = g.add_node(Node::new_flat(1000, "layer1"));
	let layer1_activ = g.add_node(Node::new_flat(1000, "layer1_activ"));

	let layer2 = g.add_node(Node::new_flat(1000, "layer2"));
	let layer2_activ = g.add_node(Node::new_flat(1000, "layer2_activ"));

	let pred = g.add_node(Node::new_flat(10, "prediction"));
	let label = g.add_training_input_node(Node::new_flat(10, "training_label"));
	

	let ops: Vec<Box<Operation>> = vec![
		

		basic::LinearMap::new(&input, &layer1, "dense1", basic::LinearMap::init_msra(0.1)),
		basic::Bias::new(&layer1, ops::ParamSharing::None, "bias1", ops::init_fill(0.0)),
		//activ::LeakyReLU::new(&layer1, &layer1_activ, 0.01, "activation1"),
		activ::BeLU::new(&layer1, &layer1_activ, ops::ParamSharing::Spatial, "activation1", activ::BeLU::init_porque_no_los_dos()),

		basic::LinearMap::new(&layer1_activ, &layer2, "dense2", basic::LinearMap::init_msra(0.1)),
		basic::Bias::new(&layer2, ops::ParamSharing::None, "bias2", ops::init_fill(0.0)),
		//activ::LeakyReLU::new(&layer2, &layer2_activ, 0.01, "activation2"),
		activ::BeLU::new(&layer2, &layer2_activ, ops::ParamSharing::Spatial, "activation1", activ::BeLU::init_porque_no_los_dos()),

		basic::LinearMap::new(&layer2_activ, &pred, "dense3", basic::LinearMap::init_msra(0.1)),
		basic::Bias::new(&pred, ops::ParamSharing::None, "bias3", ops::init_fill(0.0)),
	];
	let op_inds = g.add_operations(ops);

	if regularise != 0.0 {
		for op_ind in &op_inds {
			g.add_secondary_operation(basic::L2Regularisation::new(op_ind, regularise, "L2"), op_ind);
		}
	}

	(g, pred, label)
}

/// A common mnist network with two hidden layers of 800 units and tanh activation functions
#[allow(unused)]
fn mnist_tanh_800(regularise: f32) -> (Graph, NodeID, NodeID){
	let mut g = Graph::new();
	
	let input = g.add_input_node(Node::new_flat(28*28, "input"));
	let label = g.add_training_input_node(Node::new_flat(10, "training_label"));
		
	let layer1 = g.add_node(Node::new_flat(800, "layer1"));
	let layer1_activ = g.add_node(Node::new_flat(800, "layer1_activ"));
	
	let layer2 = g.add_node(Node::new_flat(800, "layer2"));
	let layer2_activ = g.add_node(Node::new_flat(800, "layer2_activ"));
		
	let pred = g.add_node(Node::new_flat(10, "prediction"));

	let ops: Vec<Box<Operation>> = vec![
		basic::LinearMap::new(&input, &layer1, "dense1", basic::LinearMap::init_msra(1.0)),
		basic::Bias::new(&layer1, ops::ParamSharing::Auto, "bias1", ops::init_fill(0.0)),
		activ::Tanh::new(&layer1, &layer1_activ, "activation1"),
		
		basic::LinearMap::new(&layer1_activ, &layer2, "dense2", basic::LinearMap::init_msra(1.0)),
		basic::Bias::new(&layer2, ops::ParamSharing::Auto, "bias2", ops::init_fill(0.0)),
		activ::Tanh::new(&layer2, &layer2_activ, "activation1"),
		
		basic::LinearMap::new(&layer2_activ, &pred, "dense5", basic::LinearMap::init_msra(1.0)),
	];
	let op_inds = g.add_operations(ops);

	if regularise != 0.0 {
		for op_ind in &op_inds {
			g.add_secondary_operation(basic::L2Regularisation::new(op_ind, regularise, "L2"), op_ind);
		}
	}

	(g, pred, label) 
}


/// A non convolutional densenet like graph, but rather than connecting to all previous layers only connecting each layer to layers that are a power of 2 away.
#[allow(unused)]
fn mnist_lognet(regularise: f32) -> (Graph, NodeID, NodeID){
	let mut g = Graph::new();
	
	let mut linear_nodes = vec![];
	let mut active_nodes = vec![];
	let mut ops: Vec<Box<Operation>> = vec![];

	active_nodes.push(g.add_input_node(Node::new_flat(28*28, "input")));
	
	let hidden_layers = 6; // 2^x-1 if the prediction layer should connect directly to the input
	let hidden_layer_size = 64;

	
	for i in 0..hidden_layers+1{

		let layer_size = if i < hidden_layers {hidden_layer_size} else {10};

		let new_linear_node = g.add_node(Node::new_flat(layer_size, "base_node"));

		// connect each layer (hidden and output) to previous layers which are a power of 2 from it.
		let mut jump = 1;
		while jump <= active_nodes.len(){
			ops.push(basic::LinearMap::new(&active_nodes[active_nodes.len() - jump], &new_linear_node, "dense", basic::LinearMap::init_msra(0.5/jump as f32)));
			jump *= 2;
		}
		g.add_operation(basic::Bias::new(&new_linear_node, ops::ParamSharing::Auto, "bias", ops::init_fill(0.0)));
		//ops.push(basic::Bias::new(&new_linear_node, ops::ParamSharing::Auto, "bias", ops::init_fill(0.0)));
		
		// add activation only for hidden layers
		if i < hidden_layers{
			let new_active_node = g.add_node(Node::new_flat(layer_size, "active_node"));
			//ops.push(activ::Tanh::new(&new_linear_node, &new_active_node, "activation"));
			ops.push(activ::BeLU::new(&new_linear_node, &new_active_node, ops::ParamSharing::None, "activation", activ::BeLU::init_porque_no_los_dos()));
			active_nodes.push(new_active_node);
		}

		linear_nodes.push(new_linear_node);
		
	}

	let op_inds = g.add_operations(ops);

	if regularise != 0.0 {
		for op_ind in &op_inds {
			g.add_secondary_operation(basic::L2Regularisation::new(op_ind, regularise, "L2"), op_ind);
		}
	}

	let pred = linear_nodes[linear_nodes.len()-1].clone();
	let label = g.add_training_input_node(Node::new_flat(10, "training_label"));
	(g, pred, label) 
}
