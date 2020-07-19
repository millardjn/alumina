use alumina::{
	core::exec::{exec, ExecConfig},
	core::graph::{Node, NodeTag},
	core::init::msra,
	data::{cifar::Cifar10, DataSet, DataStream},
	ops::{
		nn::conv::Padding,
		panicking::{
			add, add_n, affine, argmax, avg_pool, conv, elu, equal, ibias, l2, linear, reduce_mean, reduce_sum, scale,
			softmax_cross_entropy,
		},
	},
	opt::{adam::Adam, every_n_steps, max_steps, print_step_data, soop3::Soop, GradientOptimiser, GradientStepper},
};
use ndarray::{ArcArray, IxDyn};
use failure::Error;
use std::time::Instant;
use std::iter::empty;
use indexmap::IndexMap;

fn main() -> Result<(), Error> {
	// 1. Build a neural net graph - 76 @ 5 epochs     76 @ 10 epochs
	let input = Node::new(&[-1, 32, 32, 3]).set_name("input");
	let labels = Node::new(&[-1, 10]).set_name("labels");

	let layer1 = elu(ibias(conv(&input, 64, &[3, 3], Padding::Same), &[1, 2])).set_name("layer1");
	let layer2 = elu(ibias(conv(layer1, 64, &[3, 3], Padding::Same), &[1, 2])).set_name("layer2");
	let pool1 = avg_pool(&layer2, &[1, 2, 2, 1]).set_name("pool1");

	let layer3 = elu(ibias(conv(pool1, 128, &[3, 3], Padding::Same), &[1, 2])).set_name("layer3");
	let layer4 = elu(ibias(conv(layer3, 128, &[3, 3], Padding::Same), &[1, 2])).set_name("layer4");
	let pool2 = avg_pool(&layer4, &[1, 2, 2, 1]).set_name("pool2");

	let layer5 = elu(ibias(conv(pool2, 256, &[3, 3], Padding::Same), &[1, 2])).set_name("layer5");
	let layer6 = elu(ibias(conv(layer5, 256, &[3, 3], Padding::Same), &[1, 2])).set_name("layer6");
	let layer7 = elu(ibias(conv(layer6, 256, &[3, 3], Padding::Same), &[1, 2])).set_name("layer7");
	let layer8 = elu(ibias(conv(layer7, 256, &[3, 3], Padding::Same), &[1, 2])).set_name("layer8");
	let pool3 = avg_pool(&layer8, &[1, 2, 2, 1]).set_name("pool3");

	let layer9 = elu(ibias(conv(pool3, 512, &[3, 3], Padding::Same), &[1, 2])).set_name("layer9");
	let layer10 = elu(ibias(conv(layer9, 512, &[3, 3], Padding::Same), &[1, 2])).set_name("layer10");
	let layer11 = elu(ibias(conv(layer10, 512, &[3, 3], Padding::Same), &[1, 2])).set_name("layer11");
	let layer12 = elu(ibias(conv(layer11, 512, &[3, 3], Padding::Same), &[1, 2])).set_name("layer12");
	let pool4 = avg_pool(&layer12, &[1, 2, 2, 1]).set_name("pool4");

	let layer13 = elu(ibias(conv(pool4, 512, &[3, 3], Padding::Same), &[1, 2])).set_name("layer13");
	let layer14 = elu(ibias(conv(layer13, 512, &[3, 3], Padding::Same), &[1, 2])).set_name("layer14");
	let layer15 = elu(ibias(conv(layer14, 512, &[3, 3], Padding::Same), &[1, 2])).set_name("layer15");
	let layer16 = elu(ibias(conv(layer15, 512, &[3, 3], Padding::Same), &[1, 2])).set_name("layer16");
	let pool5 = avg_pool(&layer16, &[1, 2, 2, 1]).set_name("pool5");

	let logits = linear(&pool5, 10, msra(1.0)).set_name("logits");

	//let pool3 = avg_pool(layer10, &[1, 2, 2, 1]).set_name("pool3");

	//let layer11 = elu(affine(pool3, 512, msra(1.0))).set_name("layer7");
	//let logits = linear(&layer11, 10, msra(1.0)).set_name("logits");

	// let logits = ibias(add(
	// 	linear(reduce_mean(layer7, &[1, 2], false), 10, msra(1.0)),
	// 	linear(reduce_mean(layer10, &[1, 2], false), 10, msra(1.0))
	// ), &[]).set_name("logits");

	//let logits = reduce_mean(layer10, &[1, 2], false).set_name("logits");

	// // 1. Build a neural net graph - 76 @ 5 epochs     76 @ 10 epochs
	// let input = Node::new(&[-1, 32, 32, 3]).set_name("input");
	// let labels = Node::new(&[-1, 10]).set_name("labels");

	// let layer1 = elu(ibias(conv(&input, 32, &[3, 3], Padding::Valid), &[1, 2])).set_name("layer1");
	// let layer2 = elu(ibias(conv(layer1, 64, &[3, 3], Padding::Valid), &[1, 2])).set_name("layer2");
	// let pool1 = avg_pool(layer2, &[1, 2, 2, 1]).set_name("pool1");

	// let layer3 = elu(ibias(conv(pool1, 64, &[3, 3], Padding::Same), &[1, 2])).set_name("layer3");
	// let layer4 = elu(ibias(conv(layer3, 128, &[3, 3], Padding::Same), &[1, 2])).set_name("layer4");
	// let pool2 = avg_pool(layer4, &[1, 2, 2, 1]).set_name("pool2");

	// let layer5 = elu(ibias(conv(pool2, 128, &[3, 3], Padding::Same), &[1, 2])).set_name("layer5");
	// let layer6 = elu(ibias(conv(layer5, 256, &[3, 3], Padding::Same), &[1, 2])).set_name("layer6");
	// let pool3 = avg_pool(layer6, &[1, 2, 2, 1]).set_name("pool3");

	// let layer7 = elu(affine(pool3, 512, msra(1.0))).set_name("layer7");
	// let logits = linear(&layer7, 10, msra(1.0)).set_name("logits");

	let training_loss = add(
		reduce_sum(softmax_cross_entropy(&logits, &labels, -1), &[], false).set_name("loss"),
		scale(l2(logits.graph().nodes_tagged(NodeTag::Parameter)).set_name("l2"), 1e-3).set_name("l2_regularisation"),
	)
	.set_name("training_loss");
	let accuracy = equal(argmax(&logits, -1), argmax(&labels, -1)).set_name("accuracy");

	// 2. Print shapes
	println!("\n Values:");
	for node in accuracy
		.graph()
		.nodes()
		.difference(&accuracy.graph().nodes_tagged(NodeTag::Parameter))
	{
		println!("{:>28}  {}", node.shape(), node);
	}
	println!("\n Parameters:");
	for node in accuracy.graph().nodes_tagged(NodeTag::Parameter) {
		println!("{:>28}  {}", node.shape(), node);
	}

	// 3. Set up CIFAR10 training DataSet and DataStream
	let data_set = Cifar10::training("D:/ML/CIFAR10");
	let epoch = data_set.length();
	let batch_size = 80;
	let mut data_stream = data_set.shuffle_random().batch(batch_size).buffered(1);

	// 4. Set up validation procedure
	let mut val = validation(&input, &labels, &accuracy, &training_loss);

	// 5. Set up optimiser to run for 5 epochs and check validations every 250 steps
	let mut opt = GradientOptimiser::new(
		&training_loss,
		&[&input, &labels],
		//Soop::new(0.997).early_steps(EarlySteps::Sgd { rate: 1e-3 }).clone()
		Soop::new(),
		//Adam::new(1e-2, 0.9, 0.995)
	);
	opt.callback(max_steps(10 * epoch / batch_size));
	opt.callback(every_n_steps(50, print_step_data(batch_size as f32)));
	opt.callback(every_n_steps(250, |s: &mut Soop, data| {
		val(&mut empty());
		val(&mut s.best_estimate(&mut data.opt_inner.parameters_and_grads.keys()).into_iter());
	}));

	// 6. Train (optimise) the neural net until the max_steps callback returns Signal::Stop
	let start = Instant::now();
	opt.optimise(&mut data_stream)?;
	println!("training finished in: {:?}", start.elapsed());

	// 7. Call validation one last time
	opt.finalise();
	validation(&input, &labels, &accuracy, &training_loss)(&mut empty());

	Ok(())
}

/// Returns a closure suitable for use as a step callback which checks accuracy on the validation set
fn validation<'a>(input: &'a Node, labels: &'a Node, accuracy: &'a Node, loss: &'a Node) -> impl 'a + FnMut(&mut dyn Iterator<Item = (Node, ArcArray<f32, IxDyn>)>) {
	// Set up CIFAR10 validation DataSet and DataStream
	let val_data_set = Cifar10::testing("D:/ML/CIFAR10");
	let val_epoch = val_data_set.length();
	let val_batch = 400;
	let mut val_data_stream = val_data_set.sequential().batch(val_batch).buffered(1);

	// Define validation callback
	move |values: &mut dyn Iterator<Item = (Node, ArcArray<f32, IxDyn>)>| {
		let values: IndexMap<_, _> = values.into_iter().collect();
		let (acc_sum, loss_sum) = (0..val_epoch / val_batch).fold((0.0, 0.0), |(acc_sum, loss_sum), _| {
			let outputs = exec(
				values.clone().into_iter().chain(DataStream::next_with(&mut val_data_stream, &[input, labels])),
				&[accuracy, loss],
				&mut ExecConfig::default(),
			)
			.expect("validation execution failed");
			(acc_sum + outputs[accuracy].sum(), loss_sum + outputs[loss].sum())
		});
		let avg_accuracy = acc_sum / val_epoch as f32;
		let avg_loss = loss_sum / val_epoch as f32;
		println!("validation accuracy: {} loss: {}", avg_accuracy, avg_loss);
	}
}
