
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
use alumina::opt::Optimiser;
use alumina::opt::supplier::{Supplier, ShuffleRandom, Sequential};
use alumina::opt::supplier::mnist::MnistSupplier;
use alumina::vec_math::*;
use alumina::graph::*;


const MNIST_PATH: &'static str = "D:/ML/Mnist"; // This folder should have the mnist binary data files for training and testing.

fn main(){

	let (mut g, pred, label) = mnist_lenet(1e-4);
	let train_loss = loss::SoftMaxCrossEntLoss::new_default(&pred, &label);
	g.add_operation(train_loss);

	let mut training_set = MnistSupplier::<ShuffleRandom>::training(Path::new(MNIST_PATH));

	let start_params = g.init_params();
	let mut solver = opt::asgd::Asgd2::new(&mut g);
	

	{ // Add occasional test set evaluation as solver callback	
		let mut test_set = MnistSupplier::<Sequential>::testing(Path::new(MNIST_PATH));
		let (mut g2, pred2, label2) = mnist_lenet(0.0);
		let test_loss = loss::PredictionLoss::new_default(&pred2, &label2);
		g2.add_operation(test_loss);

		let start_params = start_params.clone();
		let mut prev_evals = 0;
		let s = training_set.epoch_size() as u64;

		solver.add_step_callback(move |_err, _step, evaluations, _graph, params|{
			
			if evaluations/s > prev_evals/s {
				prev_evals = evaluations;
				println!("Params moved:{}", params.add_scaled(&start_params, -1.0).norm2());
				
				let mut n = test_set.epoch_size();
				let count = n/128;
				let mut err = 0.0;
				
				for i in 0..count {
					let batch_size = n/(count - i);

					let (input, training_input) = test_set.next_n(batch_size);
					let (batch_err, _, _) = g2.backprop(batch_size, input, training_input, params);
					err += batch_err;
					n -= batch_size;
				}

				println!("Test error was: {}", err/test_set.epoch_size() as f32);
			}


			true
		});
	}

	solver.set_max_evals(25 * training_set.epoch_size() as u64);
	let _opt_params = solver.optimise_from(&mut training_set, start_params);


}

/// Based on LeNet variant as descripted at http://luizgh.github.io/libraries/2015/12/08/getting-started-with-lasagne/
/// Activation not specified so using BeLU
fn mnist_lenet(regularise: f32) -> (Graph, NodeID, NodeID){
	let mut g = Graph::new();

	let input = g.add_input_node(Node::new_sized(1, &[28,28], "input"));

	let ch1 = 6;
	let layer1 = g.add_node(Node::new_shaped(ch1, 2, "layer1"));
	let layer1_activ = g.add_node(Node::new_shaped(ch1, 2, "layer1_activ"));
	let layer1_pool = g.add_node(Node::new_sized(ch1, &[14, 14], "layer1_pool"));

	let ch2 = 10;
	let layer2 = g.add_node(Node::new_shaped(ch2, 2, "layer2"));
	let layer2_activ = g.add_node(Node::new_shaped(ch2, 2, "layer2_activ"));
	let layer2_pool = g.add_node(Node::new_sized(ch2, &[5, 5],"layer2_pool")); // changed from [7,7] to [5,5] to reduce parameters/overfitting


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
		reshape::Pooling::new(&layer2_activ, &layer2_pool, &[3, 3], "pooling2"), // downscale by [3,3] instead of [2,2] to reduce parameters/overfitting

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
		

		basic::LinearMap::new(&input, &layer1, "dense1", basic::LinearMap::init_msra(1.0)),
		basic::Bias::new(&layer1, ops::ParamSharing::None, "bias1", ops::init_fill(0.0)),
		//activ::LeakyReLU::new(&layer1, &layer1_activ, 0.01, "activation1"),
		activ::BeLU::new(&layer1, &layer1_activ, ops::ParamSharing::Spatial, "activation1", ops::init_fill(1.0)),

		basic::LinearMap::new(&layer1_activ, &layer2, "dense2", basic::LinearMap::init_msra(1.0)),
		basic::Bias::new(&layer2, ops::ParamSharing::None, "bias2", ops::init_fill(0.0)),
		//activ::LeakyReLU::new(&layer2, &layer2_activ, 0.01, "activation2"),
		activ::BeLU::new(&layer2, &layer2_activ, ops::ParamSharing::Spatial, "activation1", ops::init_fill(1.0)),

		basic::LinearMap::new(&layer2_activ, &pred, "dense3", basic::LinearMap::init_msra(1.0)),
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

