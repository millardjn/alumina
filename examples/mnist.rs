#[macro_use]
extern crate alumina as al;
extern crate ndarray;

use std::path::Path;
use std::sync::Arc;
use std::cell::Cell;
use al::graph::{GraphDef, NodeTag, Result};
use al::ops::nn::linear::Linear;
use al::ops::shape::avg_pool::AvgPool;
use al::ops::nn::bias::Bias;
use al::ops::nn::conv::Conv;
use al::ops::activ::tanh::Tanh;
use al::ops::activ::spline::Spline;
use al::ops::activ::softmax::Softmax;
use al::ops::loss::cross_entropy::CrossEntropy;
use al::ops::loss::prediction::Prediction;
use al::opt::{Opt, UnboxedCallbacks, CallbackSignal, max_steps, every_n_steps};

#[allow(unused_imports)]
use al::opt::sgd::Sgd;
use al::opt::adam::Adam;

use al::data::mnist::Mnist;
use al::data::*;
use ndarray::ArrayD;


fn main(){
	learn_mnist().unwrap();
}

fn learn_mnist() -> Result<()> {
	//let g = mnist_tanh_800(1.0e-3)?;
	let g = mnist_lenet(1.0e-3)?;

	let batch_size = 16;

	let data = Mnist::training("D:/ML/Mnist");
	let epoch = data.length();
	let mut data_stream = data
		.shuffle_random()
		.batch(batch_size)
		.buffered(32);


	let mut params = None;
	let avg_err = Arc::new(Cell::new(2.3));

	for &lr in &[1e-2, 1e-4, 1e-5] {
		// let mut solver = Sgd::new(&g)?
		// 	.rate(lr)
		// 	.momentum(0.9);

		let mut solver = Adam::new(&g)?
			.rate(lr)
			.beta1(0.9)
			.beta2(0.995);

		params = if params.is_some() {params} else {Some(g.initialise_nodes(solver.parameters())?)};
		
		let mut validation = validation(&g)?;
		solver.add_boxed_callback(every_n_steps(epoch/batch_size, Box::new(move |data| {
			validation(data.params);
			CallbackSignal::Continue
		})));
		let mut i = 0;
		solver.add_boxed_callback(every_n_steps(epoch/batch_size, Box::new(move |_| {i += 1; println!("epoch:{}", i); CallbackSignal::Continue})));
		solver.add_boxed_callback(max_steps(3 * epoch/batch_size));
		let mut avg_err1 = avg_err.clone();
		solver.add_callback(move |data| {
			let new_avg_err = 0.95 * avg_err1.get() + 0.05 * data.err/batch_size as f32;
			avg_err1.set(new_avg_err);
			CallbackSignal::Continue
		});
		let mut avg_err2 = avg_err.clone();
		solver.add_boxed_callback(every_n_steps(100, Box::new(move |_data| {
			println!("err: {}", avg_err2.get());
			CallbackSignal::Continue
		})));

		params = Some(solver.optimise_from(&mut data_stream, params.unwrap()).unwrap());
	}

	Ok(())
}

fn validation(g: &GraphDef) -> Result<Box<FnMut(&[ArrayD<f32>])>>{
	let data = Mnist::testing(Path::new("D:/ML/Mnist")); // I mean, who doesnt validate on the test set!
	let epoch = data.length();
	let batch_size = 100;
	let mut data_stream = data
		.sequential()
		.batch(batch_size)
		.buffered(32);

	let inputs: Vec<_> = [g.node_id("input")?, g.node_id("labels")?].iter()
		.chain(g.node_ids(NodeTag::Parameter).keys())
		.map(|node_id| node_id.value_id()).collect();
	let prediction_loss = g.node_id("prediction_loss")?;
	let mut subgraph = g.subgraph(&inputs, &[prediction_loss.value_id()])?;
	
	Ok(Box::new(move |parameters: &[ArrayD<f32>]|{
		//println!("Params moved:{}", data.params.add_scaled(&start_params, -1.0).norm2());
		
		let mut err = 0.0;
		
		for _ in 0..epoch/batch_size {
			let mut inputs = data_stream.next();
			inputs.extend(parameters.iter().cloned());
			let storage = subgraph.execute(inputs).expect("Could not execute validation");
			let err_vec = storage.get(&prediction_loss.value_id()).unwrap();
			err += err_vec.scalar_sum();
		}

		println!("Validation error is: {}%", 100.0*err/epoch as f32);
	}))
}

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

	let prediction_loss = g.new_node(shape![Unknown], "prediction_loss", tag![])?;

	g.new_op(Linear::new(&input, &layer1).init(Linear::msra(1.0)), tag![])?;
	g.new_op(Bias::new(&layer1), tag![])?;
	g.new_op(Tanh::new(&layer1, &layer1_activ), tag![])?;

	g.new_op(Linear::new(&layer1_activ, &layer2).init(Linear::msra(1.0)), tag![])?;
	g.new_op(Bias::new(&layer2), tag![])?;
	g.new_op(Tanh::new(&layer2, &layer2_activ), tag![])?;

	g.new_op(Linear::new(&layer2_activ, &prediction).init(Linear::msra(1.0)), tag![])?;
	g.new_op(Softmax::new(&prediction, &softmax), tag![])?;
	g.new_op(CrossEntropy::new(&softmax, &labels), tag![])?;

	g.new_op(Prediction::new(&prediction, &labels, &prediction_loss).axes(&[-1]), tag![])?;

	Ok(g)
}

/// Based on LeNet variant as descripted at http://luizgh.github.io/libraries/2015/12/08/getting-started-with-lasagne/
/// Activation used is the non-traditional Spline
#[allow(unused)]
fn mnist_lenet(regularise: f32) -> Result<GraphDef> {
	let mut g = GraphDef::new();

	let input = g.new_node(shape![Unknown, 28, 28, 1], "input", tag![])?;
	let labels = g.new_node(shape![Unknown, 10], "labels", tag![])?;

	let c1 = 6;
	let layer1 = g.new_node(shape![Unknown, Unknown, Unknown, c1], "layer1", tag![])?;
	let layer1_activ = g.new_node(shape![Unknown, Unknown, Unknown, c1], "layer1_activ", tag![])?;
	let layer1_pool = g.new_node(shape![Unknown, Unknown, Unknown, c1], "layer1_pool", tag![])?;

	let c2 = 10;
	let layer2 = g.new_node(shape![Unknown, Unknown, Unknown, c2], "layer2", tag![])?;
	let layer2_activ = g.new_node(shape![Unknown, Unknown, Unknown, c2], "layer2_activ", tag![])?;
	let layer2_pool = g.new_node(shape![Unknown, 7, 7, c2], "layer2_pool", tag![])?;

	let c3 = 32;
	let layer3 = g.new_node(shape![Unknown, c3], "layer3", tag![])?;
	let layer3_activ = g.new_node(shape![Unknown, c3], "layer3_activ", tag![])?;

	let prediction = g.new_node(shape![Unknown, 10], "prediction", tag![])?;
	let softmax = g.new_node(shape![Unknown, 10], "softmax", tag![])?;

	let prediction_loss = g.new_node(shape![Unknown], "prediction_loss", tag![])?;

	g.new_op(Conv::new(&input, &layer1, &[5, 5]).init(Conv::msra(1.0)), tag![])?;
	g.new_op(Bias::new(&layer1), tag![])?;
	g.new_op(Spline::new(&layer1, &layer1_activ).init(Spline::swan()), tag![])?;
	g.new_op(AvgPool::new(&layer1_activ, &layer1_pool, &[1, 2, 2, 1]), tag![])?;

	g.new_op(Conv::new(&layer1_pool, &layer2, &[5, 5]).init(Conv::msra(1.0)), tag![])?;
	g.new_op(Bias::new(&layer2), tag![])?;
	g.new_op(Spline::new(&layer2, &layer2_activ).init(Spline::swan()), tag![])?;
	g.new_op(AvgPool::new(&layer2_activ, &layer2_pool, &[1, 2, 2, 1]), tag![])?;

	g.new_op(Linear::new(&layer2_pool, &layer3).init(Linear::msra(1.0)), tag![])?;
	g.new_op(Bias::new(&layer3), tag![])?;
	g.new_op(Spline::new(&layer3, &layer3_activ).init(Spline::swan()), tag![])?;

	g.new_op(Linear::new(&layer3_activ, &prediction).init(Linear::msra(1.0)), tag![])?;
	g.new_op(Softmax::new(&prediction, &softmax), tag![])?;
	g.new_op(CrossEntropy::new(&softmax, &labels), tag![])?;

	g.new_op(Prediction::new(&prediction, &labels, &prediction_loss).axes(&[-1]), tag![])?;

	Ok(g)
}
