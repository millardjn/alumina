use alumina::{
	core::exec::{exec, ExecConfig},
	core::graph::{Node, NodeTag},
	core::init::msra,
	data::{cifar::Cifar100, DataSet, DataStream},
	ops::{
		nn::conv::Padding,
		nn::spline::swan,
		panicking::{
			add, argmax, avg_pool, conv, equal, ibias, l2, linear, reduce_mean, reduce_sum, scale,
			softmax_cross_entropy, spline,
		},
	},
	opt::{adam::Adam, every_n_steps, max_steps, nth_step, print_step_data, GradientOptimiser},
};
use alumina_core::graph::Graph;
use failure::Error;
use indexmap::IndexMap;
use ndarray::{ArcArray, IxDyn};
use std::iter::empty;
use std::time::Instant;

fn main() -> Result<(), Error> {
	// 1. Build a neural net graph
	let input = Node::new(&[-1, 32, 32, 3]).set_name("input");
	let fine_labels = Node::new(&[-1, 100]).set_name("labels");

	// 54% @ 25 epochs
	let layer1 = spline(ibias(conv(&input, 32, &[3, 3], Padding::Valid), &[]), &[1, 2], swan()).set_name("layer1");
	let layer2 = spline(ibias(conv(layer1, 64, &[3, 3], Padding::Valid), &[]), &[1, 2], swan()).set_name("layer2");
	let layer3 = spline(ibias(conv(layer2, 64, &[3, 3], Padding::Valid), &[]), &[1, 2], swan()).set_name("layer3");
	let layer4 = spline(ibias(conv(layer3, 96, &[3, 3], Padding::Valid), &[]), &[1, 2], swan()).set_name("layer4");
	let pool1 = avg_pool(layer4, &[1, 2, 2, 1]).set_name("pool1");

	let layer5 = spline(ibias(conv(pool1, 96, &[3, 3], Padding::Same), &[]), &[1, 2], swan()).set_name("layer5");
	let layer6 = spline(ibias(conv(layer5, 96, &[3, 3], Padding::Same), &[]), &[1, 2], swan()).set_name("layer6");
	let layer7 = spline(ibias(conv(layer6, 192, &[3, 3], Padding::Same), &[]), &[1, 2], swan()).set_name("layer7");
	let pool2 = avg_pool(&layer7, &[1, 2, 2, 1]).set_name("pool2");

	let layer8 = spline(ibias(conv(pool2, 192, &[3, 3], Padding::Same), &[]), &[1, 2], swan()).set_name("layer8");
	let layer9 = spline(ibias(conv(layer8, 192, &[3, 3], Padding::Same), &[]), &[1, 2], swan()).set_name("layer9");
	let layer10 = spline(ibias(conv(layer9, 192, &[3, 3], Padding::Same), &[]), &[1, 2], swan()).set_name("layer10");

	//let m2 = linear(reduce_mean(layer7, &[1, 2], false), 100, msra(1.0)).set_name("m2");
	//let m3 = affine(reduce_mean(layer10, &[1, 2], false), 100, msra(1.0)).set_name("m3");
	let logits = ibias(
		//add(
		//linear(reduce_mean(layer7, &[1, 2], false), 100, msra(1.0)),
		linear(reduce_mean(layer10, &[1, 2], false), 100, msra(1.0)), //)
		&[],
	)
	.set_name("logits");

	let training_loss = add(
		reduce_sum(softmax_cross_entropy(&logits, &fine_labels, -1), &[], false).set_name("loss"),
		scale(l2(logits.graph().nodes_tagged(NodeTag::Parameter)).set_name("l2"), 1e-3).set_name("l2_regularisation"),
	)
	.set_name("training_loss");
	let accuracy = equal(argmax(&logits, -1), argmax(&fine_labels, -1)).set_name("accuracy");

	// 2. Print shapes
	print_node_shapes(accuracy.graph());

	// 3. Set up CIFAR10 training DataSet and DataStream
	let data_set = Cifar100::training("D:/ML/CIFAR100").reorder_components(&[0, 2]);
	let epoch = data_set.length();
	let batch_size = 80;
	let mut data_stream = data_set.shuffle_random().batch(batch_size).buffered(1);

	// 4. Set up validation procedure
	let mut val = validation(&input, &fine_labels, &accuracy, &training_loss);

	// 5. Set up optimiser to run for 5 epochs and check validations every 75 steps
	let mut opt = GradientOptimiser::new(&training_loss, &[&input, &fine_labels], Adam::new(3e-3, 0.9, 0.995));

	opt.callback(max_steps(25 * epoch / batch_size));
	opt.callback(every_n_steps(50, print_step_data(batch_size as f32)));
	opt.callback(every_n_steps(250, move |_s: &mut Adam, _data| {
		val(&mut empty());
	}));
	opt.callback(nth_step(10 * epoch / batch_size, |s: &mut Adam, _data| {
		s.rate(3.3e-4);
	}));

	// 6. Train (optimise) the neural net until the max_steps callback returns Signal::Stop
	let start = Instant::now();
	opt.optimise(&mut data_stream)?;
	println!("soop training finished in: {:?}", start.elapsed());

	// 7. Call validation one last time
	opt.finalise();
	validation(&input, &fine_labels, &accuracy, &training_loss)(&mut empty());

	Ok(())
}

/// Constructs and returns a closure suitable for use as a step callback which checks accuracy and loss on the validation set
fn validation<'a>(
	input: &'a Node,
	labels: &'a Node,
	accuracy: &'a Node,
	loss: &'a Node,
) -> impl 'a + FnMut(&mut dyn Iterator<Item = (Node, ArcArray<f32, IxDyn>)>) {
	let val_data_set = Cifar100::testing("D:/ML/CIFAR100").reorder_components(&[0, 2]);
	let val_epoch = val_data_set.length();
	let val_batch = val_epoch;
	let mut val_data_stream = val_data_set.sequential().batch(val_batch).buffered(0);

	// Define validation callback
	move |values: &mut dyn Iterator<Item = (Node, ArcArray<f32, IxDyn>)>| {
		let values: IndexMap<_, _> = values.into_iter().collect();
		let (acc_sum, loss_sum) = (0..val_epoch / val_batch).fold((0.0, 0.0), |(acc_sum, loss_sum), _| {
			let outputs = exec(
				values
					.clone()
					.into_iter()
					.chain(<dyn DataStream>::next_with(&mut val_data_stream, &[input, labels])),
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

fn print_node_shapes(graph: &Graph) {
	println!("\n Values:");
	for node in graph.nodes().difference(&graph.nodes_tagged(NodeTag::Parameter)) {
		println!("{:>28}  {}", node.shape(), node);
	}
	println!("\n Parameters:");
	for node in graph.nodes_tagged(NodeTag::Parameter) {
		println!("{:>28}  {}", node.shape(), node);
	}
}
