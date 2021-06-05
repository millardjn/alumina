use alumina::{
	core::exec::{exec, ExecConfig},
	core::graph::{Node, NodeTag},
	core::init::msra,
	data::{mnist::Mnist, DataSet, DataStream},
	ops::panicking::{add, affine, argmax, elu, equal, l2, linear, reduce_sum, scale, softmax_cross_entropy},
	opt::{adam::Adam, every_n_steps, max_steps, nth_step, print_step_data, GradientOptimiser, GradientStepper},
};
use failure::Error;
use indexmap::IndexMap;
use ndarray::{ArcArray, IxDyn};
use std::iter::empty;
use std::time::Instant;

fn main() -> Result<(), Error> {
	// 1. Build a neural net graph - 98% @ 10 epochs
	let input = Node::new(&[-1, 28, 28, 1]).set_name("input");
	let labels = Node::new(&[-1, 10]).set_name("labels");

	let layer1 = elu(affine(&input, 256, msra(1.0))).set_name("layer1");
	let layer2 = elu(affine(&layer1, 256, msra(1.0))).set_name("layer2");
	let logits = linear(&layer2, 10, msra(1.0)).set_name("logits");

	let training_loss = add(
		reduce_sum(softmax_cross_entropy(&logits, &labels, -1), &[], false).set_name("loss"),
		scale(l2(logits.graph().nodes_tagged(NodeTag::Parameter)), 1e-3).set_name("regularisation"),
	)
	.set_name("training_loss");
	let accuracy = equal(argmax(&logits, -1), argmax(&labels, -1)).set_name("accuracy");

	// 2. Print shapes
	print_node_shapes(accuracy.graph());

	// 3. Set up MNIST training DataSet and DataStream
	let data_set = Mnist::training("D:/ML/Mnist");
	let epoch = data_set.length();
	let batch_size = 80;
	let mut data_stream = data_set.shuffle_random().batch(batch_size).buffered(1);

	// 4. Set up validation procedure
	let mut val = validation(&input, &labels, &accuracy, &training_loss);

	// 5. Set up optimiser to run for 25 epochs and check validation every 300 steps
	let mut opt = GradientOptimiser::new(&training_loss, &[&input, &labels], Adam::new(3.3e-3, 0.9, 0.995));
	opt.callback(max_steps(15 * epoch / batch_size));
	opt.callback(every_n_steps(50, print_step_data(batch_size as f32)));
	opt.callback(nth_step(10 * epoch / batch_size, |s: &mut Adam, _data| {
		s.rate(3.3e-4);
	}));
	opt.callback(every_n_steps(300, move |s: &mut Adam, data| {
		val(&mut empty());
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

/// Constructs and returns a closure suitable for use as a step callback which checks accuracy and loss on the validation set
fn validation<'a>(
	input: &'a Node,
	labels: &'a Node,
	accuracy: &'a Node,
	loss: &'a Node,
) -> impl 'a + FnMut(&mut dyn Iterator<Item = (Node, ArcArray<f32, IxDyn>)>) {
	let val_data_set = Mnist::testing("D:/ML/Mnist");
	let val_epoch = val_data_set.length();
	let val_batch = 400;
	let mut val_data_stream = val_data_set.sequential().batch(val_batch).buffered(0);

	// Define validation callback
	move |values: &mut dyn Iterator<Item = (Node, ArcArray<f32, IxDyn>)>| {
		let values: IndexMap<_, _> = values.into_iter().collect();
		let (acc_sum, loss_sum) = (0..val_epoch / val_batch).fold((0.0, 0.0), |(acc_sum, loss_sum), _| {
			let outputs = exec(
				values
					.clone()
					.into_iter()
					.chain(DataStream::next_with(&mut val_data_stream, &[input, labels])),
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


fn print_node_shapes(graph: &Graph){
	println!("\n Values:");
	for node in graph
		.nodes()
		.difference(&graph.nodes_tagged(NodeTag::Parameter))
	{
		println!("{:>28}  {}", node.shape(), node);
	}
	println!("\n Parameters:");
	for node in graph.nodes_tagged(NodeTag::Parameter) {
		println!("{:>28}  {}", node.shape(), node);
	}
}