#[macro_use]
extern crate alumina as al;

use std::path::Path;

use al::new::graph::{GraphDef, Result};
use al::new::ops::nn::linear::Linear;
use al::new::ops::nn::bias::Bias;
use al::new::ops::activ::tanh::Tanh;
use al::new::ops::activ::softmax::Softmax;
use al::new::ops::loss::cross_entropy::CrossEntropy;
use al::new::opt::{Opt, CallbackSignal, max_steps, every_n_steps};

use al::new::opt::sgd::Sgd;

use al::new::data::mnist::Mnist;
use al::new::data::*;


fn main(){
	learn_mnist().unwrap();
}

fn learn_mnist() -> Result<()> {
	let g = mnist_tanh_800(1.0e-3)?;

	let batch_size = 16;

	let data = Mnist::training(Path::new("D:/ML/Mnist"));
	let epoch = data.length();
	let mut data_stream = data.shuffle_random()
		.batch(batch_size);
		//.buffered(32);

	let mut solver = Sgd::new(&g)?
		.rate(1e-3)
		.momentum(0.9);

	let start_params = g.initialise_nodes(solver.parameters())?;

	//solver.add_boxed_callback(validation(&start_params, training_set.epoch_size()));
	solver.add_boxed_callback(max_steps(100 * epoch/batch_size));
	solver.add_boxed_callback(every_n_steps(100, Box::new(|data| {println!("err: {}", data.err); CallbackSignal::Continue})));

	let _params = solver.optimise_from(&mut data_stream, start_params).unwrap();

	// println!("Total num params: {}", start_params.len());

	// let _opt_params = solver.optimise_from(&mut training_set, start_params);

	Ok(())
}

// fn validation(start_params: &[f32], training_set_size: usize) -> Box<FnMut(&CallbackData)->CallbackSignal>{
// 	let mut val_set = MnistSupplier::<Sequential>::testing(Path::new(MNIST_PATH)); // I mean, who doesnt validate on the test set /s
// 	let (mut g2, pred2, label2) = mnist_adam(0.0);
// 	let val_loss = loss::PredictionLoss::new_default(&pred2, &label2);
// 	g2.add_operation(val_loss);

// 	let start_params = start_params.to_vec();

// 	opt::every_n_evals(training_set_size, Box::new(move |data: &CallbackData|{
// 		println!("Params moved:{}", data.params.add_scaled(&start_params, -1.0).norm2());
		
// 		let mut n = val_set.epoch_size();
// 		let count = n/256;
// 		let mut err = 0.0;
		
// 		for i in 0..count {
// 			let batch_size = n/(count - i);

// 			let (input, training_input) = val_set.next_n(batch_size);
// 			let (batch_err, _, _) = g2.backprop(batch_size, input, training_input, data.params);
// 			err += batch_err;
// 			n -= batch_size;
// 		}

// 		println!("Validation error is: {}%", 100.0*err/val_set.epoch_size() as f32);
// 		CallbackSignal::Continue
// 	}))
// }

/// A common mnist network with two hidden layers of 800 units and tanh activation functions
#[allow(unused)]
fn mnist_tanh_800(regularise: f32) -> Result<GraphDef> {
	let mut g = GraphDef::new();

	let input = g.new_node(shape![Unknown, 28, 28, 1], "input", tag![])?;
	let labels = g.new_node(shape![Unknown, 10], "labels", tag![])?;

	let layer1 = g.new_node(shape![Unknown, 800], "layer1", tag![])?;
	let layer1_activ = g.new_node(shape![Unknown, 800], "layer1_activ", tag![])?;

	let layer2 = g.new_node(shape![Unknown, 800], "layer2", tag![])?;
	let layer2_activ = g.new_node(shape![Unknown, 800], "layer2_activ", tag![])?;

	let prediction = g.new_node(shape![Unknown, 10], "prediction", tag![])?;
	let softmax = g.new_node(shape![Unknown, 10], "softmax", tag![])?;

	g.new_op(Linear::new(&input, &layer1).init(Linear::msra(2.0)), tag![])?;
	g.new_op(Bias::new(&layer1), tag![])?;
	g.new_op(Tanh::new(&layer1, &layer1_activ), tag![])?;

	g.new_op(Linear::new(&layer1_activ, &layer2).init(Linear::msra(2.0)), tag![])?;
	g.new_op(Bias::new(&layer2), tag![])?;
	g.new_op(Tanh::new(&layer2, &layer2_activ), tag![])?;

	g.new_op(Linear::new(&layer2_activ, &prediction).init(Linear::msra(1.0)), tag![])?;
	g.new_op(Softmax::new(&prediction, &softmax), tag![])?;
	g.new_op(CrossEntropy::new(&softmax, &labels), tag![])?;

	Ok(g)
}

// /// Based on LeNet variant as descripted at http://luizgh.github.io/libraries/2015/12/08/getting-started-with-lasagne/
// /// Activation not specified so using BeLU
// fn mnist_lenet(regularise: f32) -> (Graph, NodeID, NodeID){
	
// 	let mut g = Graph::new();

// 	let input = g.add_input_node(Node::new_sized(1, &[28,28], "input"));

// 	let ch1 = 6;
// 	let layer1 = g.add_node(Node::new_shaped(ch1, 2, "layer1"));
// 	let layer1_activ = g.add_node(Node::new_shaped(ch1, 2, "layer1_activ"));
// 	let layer1_pool = g.add_node(Node::new_sized(ch1, &[14, 14], "layer1_pool"));

// 	let ch2 = 10;
// 	let layer2 = g.add_node(Node::new_shaped(ch2, 2, "layer2"));
// 	let layer2_activ = g.add_node(Node::new_shaped(ch2, 2, "layer2_activ"));
// 	let layer2_pool = g.add_node(Node::new_sized(ch2, &[7, 7],"layer2_pool")); // changed from [7,7] to [5,5] to reduce parameters/overfitting

// 	let ch3 = 32; // changed from 100 to 64 to reduce parameters/overfitting
// 	let layer3 = g.add_node(Node::new_flat(ch3, "layer3"));
// 	let layer3_activ = g.add_node(Node::new_flat(ch3, "layer3_activ"));	


// 	let pred = g.add_node(Node::new_flat(10, "prediction"));
// 	let label = g.add_training_input_node(Node::new_flat(10, "training_label"));
	

// 	let ops: Vec<Box<Operation>> = vec![
		
// 		conv::Convolution::new(&input, &layer1, &[5, 5], conv::Padding::Same, "conv1", conv::Convolution::init_msra(1.0)),
// 		basic::Bias::new(&layer1, ops::ParamSharing::Spatial, "bias1", ops::init_fill(0.0)),
// 		activ::BeLU::new(&layer1, &layer1_activ, ops::ParamSharing::Spatial, "activation1", activ::BeLU::init_porque_no_los_dos()),
// 		reshape::Pooling::new(&layer1_activ, &layer1_pool, &[2, 2], "pooling1"),

// 		conv::Convolution::new(&layer1_pool, &layer2, &[5, 5], conv::Padding::Same, "conv2", conv::Convolution::init_msra(1.0)),
// 		basic::Bias::new(&layer2, ops::ParamSharing::Spatial, "bias2", ops::init_fill(0.0)),
// 		activ::BeLU::new(&layer2, &layer2_activ, ops::ParamSharing::Spatial, "activation2", activ::BeLU::init_porque_no_los_dos()),
// 		reshape::Pooling::new(&layer2_activ, &layer2_pool, &[2, 2], "pooling2"), // downscale by [3,3] instead of [2,2] to reduce parameters/overfitting

// 		basic::LinearMap::new(&layer2_pool, &layer3, "dense1", basic::LinearMap::init_msra(1.0)),
// 		basic::Bias::new(&layer3, ops::ParamSharing::None, "bias_dense1", ops::init_fill(0.0)),
// 		activ::BeLU::new(&layer3, &layer3_activ, ops::ParamSharing::None, "activation2", activ::BeLU::init_porque_no_los_dos()),

// 		basic::LinearMap::new(&layer3_activ, &pred, "dense2", basic::LinearMap::init_msra(1.0)),
// 		basic::Bias::new(&pred, ops::ParamSharing::None, "bias_dense2", ops::init_fill(0.0)),
// 	];
// 	let op_inds = g.add_operations(ops);

// 	if regularise != 0.0 {
// 		for op_ind in &op_inds {
// 			g.add_secondary_operation(basic::L2Regularisation::new(op_ind, regularise, "L2"), op_ind);
// 		}
// 	}

// 	(g, pred, label)
// }


