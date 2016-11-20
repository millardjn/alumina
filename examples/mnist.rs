extern crate byteorder;
extern crate rand;
extern crate alumina;


use byteorder::*;
use rand::*;

use alumina::ops::activ::*;
use alumina::ops::basic::*;	
use alumina::ops::loss::*;
use alumina::ops::conv::*;
use alumina::ops::reshape::*;
use alumina::opt::*;
use alumina::opt::asgd::*;
//use alumina::opt::sgd::*;
use alumina::ops::*;

use alumina::graph::*;	
use alumina::vec_math::*;

use alumina::shape::*;

use std::fs::*;
use std::path::Path;
use std::io::Read;
use std::iter;



fn main(){
	//let (mut g, pred, label) = mnist_graph(false);
	let (mut g, pred, label) = mnist_conv_graph(0.001);
	let train_loss = SoftMaxCrossEntLoss::new_default(&pred, &label);
	g.add_operation(train_loss);

	//let (mut g2, pred2, label2) = mnist_graph(false);
	let (mut g2, pred2, label2) = mnist_conv_graph(0.0);
	let test_loss = PredictionLoss::new_default(&pred2, &label2);
	g2.add_operation(test_loss);


	let mut training_set = MnistSupplier::create_training(Path::new("D:/ML/Mnist"));
	let mut test_set = MnistSupplier::create_testing(Path::new("D:/ML/Mnist"));


	let start_params = g.init_params();
	let mut opt_params = start_params.clone();

	let mut solver = Asgd2::new(&mut g);
	solver.set_max_evals(training_set.epoch_size() as u64);

	for _ in 0..200 {



		solver.reset_eval_steps();
		//solver2.set_min_batch_size(16.0);
		opt_params = solver.optimise_from(&mut training_set, opt_params.clone());	

		println!("Params moved:{}", opt_params.add_scaled(&start_params, -1.0).norm2());



		let mut n = test_set.epoch_size() as usize;
		let count = n/128;
		let mut err = 0.0;

		for i in 0..count {
			let batch_size = n/(count - i);

			let (input, training_input) = test_set.next_n(batch_size);
			let (batch_err, _, _) = g2.backprop(batch_size, input, training_input, &opt_params);
			err += batch_err;
			n -= batch_size;
		}

		println!("Test error was: {}", err/test_set.epoch_size() as f32);

	}

}

fn mnist_conv_graph(regularise: f32) -> (Graph, (NodeIndex, NodeShape), (NodeIndex, NodeShape)){
	let mut g = Graph::new();

	let input = g.add_input_node(Node::new_sized(1, vec![28,28], "input"));

	let ch1 = 32;
	let layer1 = g.add_node(Node::new_shaped(ch1, 2, "layer1"));
	let layer1_activ = g.add_node(Node::new_sized(ch1, vec![28,28], "layer1_activ"));
	let layer1_pool = g.add_node(Node::new_sized(ch1, vec![10, 10], "layer1_pool"));

	let ch2 = 16;
	let layer2 = g.add_node(Node::new_shaped(ch2, 2, "layer2"));
	let layer2_activ = g.add_node(Node::new_shaped(ch2, 2, "layer2_activ"));
	let layer2_pool = g.add_node(Node::new_sized(ch2, vec![4, 4],"layer2_pool"));

	let pred = g.add_node(Node::new_flat(10, "prediction"));
	let label = g.add_training_input_node(Node::new_flat(10, "training_label"));
	

	let ops: Vec<Box<Operation>> = vec![
		
		Convolution::new(&input, &layer1, vec![5, 5], Padding::Same, "conv1", Convolution::init_msra(1.0)),
		Bias::new(&layer1, ParamSharing::Auto, "bias1", init_fill(0.0)),
		BeLU::new(&layer1, &layer1_activ, ParamSharing::Auto, "activation1", BeLU::init_porque_no_los_dos()),
		//LeakyReLU::new(&g, layer1, layer1_activ, 0.05, "activation1"),
		Pooling::new(&layer1_activ, &layer1_pool, vec![3, 3], "pooling1"),

		Convolution::new(&layer1_pool, &layer2, vec![3, 3], Padding::Same, "conv2", Convolution::init_msra(1.0)),
		Bias::new(&layer2, ParamSharing::Auto, "bias2", init_fill(0.0)),
		BeLU::new(&layer2, &layer2_activ, ParamSharing::Auto, "activation2", BeLU::init_porque_no_los_dos()),
		//LeakyReLU::new(&g, layer2, layer2_activ, 0.05, "activation2"),
		Pooling::new(&layer2_activ, &layer2_pool, vec![3, 3], "pooling2"),


		LinearMap::new(&layer2_pool, &pred, "dense1", LinearMap::init_msra(1.0)),
		Bias::new(&pred, ParamSharing::Auto, "bias_dense1", init_fill(0.0)),
		
		//MseLoss::new_default(&g, pred, label),
		//SoftMaxCrossEntLoss::new_default(&g, pred, label),
	];
	let op_inds = g.add_operations(ops);

	if regularise != 0.0 {
		for (op_ind, num_params) in op_inds {
			g.add_secondary_operation(L2Regularisation::new((op_ind, num_params), regularise, "L2"), op_ind);
		}
		//g.add_secondary_operation(L2Regularisation::new(skip, 0.05, "L2"), skip.0);
	}

	(g, pred, label)
}

fn mnist_conv_graph2(regularise: f32) -> (Graph, (NodeIndex, NodeShape), (NodeIndex, NodeShape)){
	let mut g = Graph::new();

	let input = g.add_input_node(Node::new_sized(1, vec![28,28], "input"));

	let ch1 = 32;
	let layer1 = g.add_node(Node::new_shaped(ch1, 2, "layer1"));
	let layer1_activ = g.add_node(Node::new_sized(ch1, vec![28,28], "layer1_activ"));
	let layer1_pool = g.add_node(Node::new_sized(ch1, vec![14, 14], "layer1_pool"));

	let ch2 = 16;
	let layer2 = g.add_node(Node::new_shaped(ch2, 2, "layer2"));
	let layer2_activ = g.add_node(Node::new_shaped(ch2, 2, "layer2_activ"));
	let layer2_pool = g.add_node(Node::new_shaped(ch2, 2, "layer2_pool"));

	let ch3 = 8;
	let layer3 = g.add_node(Node::new_shaped(ch3, 2, "layer3"));
	let layer3_activ = g.add_node(Node::new_sized(ch3, vec![7, 7], "layer3_activ"));	
	//let layer3_pool = g.add_node(Node::new_sized(ch3, vec![4, 4], "layer3_pool"));

	let pred = g.add_node(Node::new_flat(10, "prediction"));
	let label = g.add_training_input_node(Node::new_flat(10, "training_label"));
	

	let ops: Vec<Box<Operation>> = vec![
		
		Convolution::new(&input, &layer1, vec![5, 5], Padding::Same, "conv1", Convolution::init_msra(1.0)),
		Bias::new(&layer1, ParamSharing::Auto, "bias1", init_fill(0.0)),
		BeLU::new(&layer1, &layer1_activ, ParamSharing::Auto, "activation1", BeLU::init_porque_no_los_dos()),
		//LeakyReLU::new(&g, layer1, layer1_activ, 0.05, "activation1"),
		Pooling::new(&layer1_activ, &layer1_pool, vec![2, 2], "pooling1"),

		Convolution::new(&layer1_pool, &layer2, vec![5, 5], Padding::Same, "conv2", Convolution::init_msra(1.0)),
		Bias::new(&layer2, ParamSharing::Auto, "bias2", init_fill(0.0)),
		BeLU::new(&layer2, &layer2_activ, ParamSharing::Auto, "activation2", BeLU::init_porque_no_los_dos()),
		//LeakyReLU::new(&g, layer2, layer2_activ, 0.05, "activation2"),
		Pooling::new(&layer2_activ, &layer2_pool, vec![2, 2], "pooling2"),

		Convolution::new(&layer2_pool, &layer3, vec![5, 5], Padding::Same, "conv3", Convolution::init_msra(1.0)),
		Bias::new(&layer3, ParamSharing::Auto, "bias3", init_fill(0.0)),
		BeLU::new(&layer3, &layer3_activ, ParamSharing::Auto, "activation3", BeLU::init_porque_no_los_dos()),
		//LeakyReLU::new(&g, layer3, layer3_activ, 0.05, "activation3"),
		//Pooling::new(&g, vec![2, 2], layer3_activ, layer3_pool, "pooling3"),

		LinearMap::new(&layer3_activ, &pred, "dense1", LinearMap::init_msra(1.0)),
		Bias::new(&pred, ParamSharing::Auto, "bias_dense1", init_fill(0.0)),
		
		
		//MseLoss::new_default(&g, pred, label),
		//SoftMaxCrossEntLoss::new_default(&g, pred, label),
	];
	let op_inds = g.add_operations(ops);
	//let skip = g.add_operation(LinearMap::new(&layer1_activ, &pred, "skip_dense", LinearMap::init_msra(1.0)));

	if regularise != 0.0 {
		for (op_ind, num_params) in op_inds {
			g.add_secondary_operation(L2Regularisation::new((op_ind, num_params), regularise, "L2"), op_ind);
		}
		//g.add_secondary_operation(L2Regularisation::new(skip, 0.05, "L2"), skip.0);
	}

	(g, pred, label)
}


fn mnist_graph(regularise: bool) -> (Graph, (NodeIndex, NodeShape), (NodeIndex, NodeShape)){
	let mut g = Graph::new();
	
	let input = g.add_input_node(Node::new_flat(28*28, "input"));
	let label = g.add_training_input_node(Node::new_flat(10, "training_label"));
		
	let layer1 = g.add_node(Node::new_flat(800, "layer1"));
	let layer1_activ = g.add_node(Node::new_flat(800, "layer1_activ"));
	
	let layer2 = g.add_node(Node::new_flat(800, "layer2"));
	let layer2_activ = g.add_node(Node::new_flat(800, "layer2_activ"));
		
	let pred = g.add_node(Node::new_flat(10, "prediction"));

	
	let ops: Vec<Box<Operation>> = vec![
		LinearMap::new(&input, &layer1, "dense1", LinearMap::init_msra(1.0)),
		Bias::new(&layer1, ParamSharing::Auto, "bias1", init_fill(0.0)),
		//Tanh::new(&layer1, &layer1_activ, "activation1"),
		BeLU::new(&layer1, &layer1_activ, ParamSharing::Auto, "activation1", BeLU::init_elu_like()),
		//SoftExp::new(&g, layer1, layer1_activ, "activation1", SoftExp::init_fill(0.0)),
		//LeakyReLU::new(&g, layer1, layer1_activ, 0.05, "activation1"),
		
		LinearMap::new(&layer1_activ, &layer2, "dense2", LinearMap::init_msra(1.0)),
		Bias::new(&layer2, ParamSharing::Auto, "bias2", init_fill(0.0)),
		//Tanh::new(&layer2, &layer2_activ, "activation1"),
		BeLU::new(&layer2, &layer2_activ, ParamSharing::Auto, "activation2", BeLU::init_elu_like()),
		//SoftExp::new(&g, layer2, layer2_activ, "activation2", SoftExp::init_fill(0.0)),
		//LeakyReLU::new(&g, layer2, layer2_activ, 0.05, "activation2"),

		
		LinearMap::new(&layer2_activ, &pred, "dense5", LinearMap::init_msra(1.0)),
		//LinearMap::new(&g, unit, pred, "bias5", LinearMap::init_zero_fill()),
		
		//MseLoss::new_default(&g, pred, label),
		//SoftMaxCrossEntLoss::new_default(&g, pred, label),
	];
	let op_inds = g.add_operations(ops);

	if regularise {
		for (op_ind, num_params) in op_inds {
			g.add_secondary_operation(L2Regularisation::new((op_ind, num_params), 0.001, "L2"), op_ind);
		}
	}

	(g, pred, label) 
}



pub struct MnistSupplier{
	count: u64,
	labels: Vec<u8>,
	images: Vec<Vec<u8>>,
	shape: NodeShape,
	order: Box<Iterator<Item=usize>>,
}

impl Supplier for MnistSupplier{
	// fn next(&mut self) -> (Vec<NodeData>, Vec<NodeData>){
	// 	let (input, train) = self.get();
	// 	(vec![input], vec![train])
	// }
	
	fn next_n(&mut self, n: usize) -> (Vec<NodeData>, Vec<NodeData>){
		if n < 1 {panic!("")}
		
		let (mut input, mut train) = self.get();
		
		for _ in 1..n{
			let (input2, train2) = self.get();
			input = input.join(input2);
			train = train.join(train2);
		}
		(vec![input], vec![train])
	}
	
	fn epoch_size(&self) -> usize{
		self.labels.len()
	}
	fn samples_taken(&self) -> u64{
		self.count
	}
	fn reset(&mut self){
		self.order = Box::new(iter::empty());
		self.count = 0;
	}
	fn once(mut self) -> Vec<(Vec<NodeData>, Vec<NodeData>)> {
		self.order = self.random_order();
		(0..self.labels.len())
			.map(|_| self.get())
			.map(|(input, train)| (vec![input], vec![train])).collect()
	}
}

impl MnistSupplier{
	
	fn random_order(&self) -> Box<Iterator<Item=usize>>{
		
		let mut v: Vec<usize> = (0..self.labels.len()).collect();
		let mut rng = rand::thread_rng();
		rng.shuffle(&mut v);
		
		Box::new(v.into_iter())
	}
	
	fn get(&mut self) -> (NodeData, NodeData){
		
		match self.order.next() {
			Some(i) => {
				let mut one_hot_label = vec![0.0; 10];
				one_hot_label[self.labels[i] as usize] = 1.0;
				self.count += 1;
				let n = 1;
				(
					NodeData::new(self.shape.to_data_shape(n).unwrap(), self.images[i].iter().map(|&i| i as f32/255.0).collect()), 
					NodeData::new(DataShape::new_flat(10, n), one_hot_label)
				)
			},
			None => {
				self.order = self.random_order();
				self.get()
			},
		}

	}
	
	pub fn create_training(folder_path: &Path) -> MnistSupplier {
		let err = "Pass a folder containing 'train-images.idx3-ubyte' and 'train-labels.idx1-ubyte'";
		assert!(folder_path.is_dir(), err);
		let label_file = File::open(folder_path.join("train-labels.idx1-ubyte").as_path()).expect(err);
		let image_file = File::open(folder_path.join("train-images.idx3-ubyte").as_path()).expect(err);
		MnistSupplier::create(image_file, label_file)
	}
	
	pub fn create_testing(folder_path: &Path) -> MnistSupplier {
		let err = "Pass a folder containing 't10k-images.idx3-ubyte' and 't10k-labels.idx1-ubyte'";
		assert!(folder_path.is_dir(), err);
		let label_file = File::open(folder_path.join("t10k-labels.idx1-ubyte").as_path()).expect(err);
		let image_file = File::open(folder_path.join("t10k-images.idx3-ubyte").as_path()).expect(err);
		MnistSupplier::create(image_file, label_file)
	}
	
	fn create(image_file: File, label_file: File) -> MnistSupplier  {
		let labels = MnistSupplier::read_label_file(label_file);
		let (shape, images) = MnistSupplier::read_image_file(image_file);
		assert!(images.len() == labels.len());

		MnistSupplier{
			count: 0,
			labels: labels,
			images: images,
			shape: shape,
			order: Box::new(iter::empty()),
		}
	}
	
	fn read_label_file(mut file: File) -> Vec<u8> {
		let mut buf: Vec<u8> = vec![0; file.metadata().unwrap().len() as usize];
		file.read_exact(&mut buf).unwrap();
		let mut i: usize = 0;
		
		assert!(BigEndian::read_u32(&buf[i..]) == 2049); i += 4;
		
		let n = BigEndian::read_u32(&buf[i..]) as usize; i += 4;
		
		let mut v1 = Vec::with_capacity(n);
		for _ in 0..n {
			v1.push(buf[i]); i += 1;
		}
		
		v1
	}
	
	fn read_image_file(mut file: File) -> (NodeShape, Vec<Vec<u8>>){
		let mut buf: Vec<u8> = vec![0; file.metadata().unwrap().len() as usize];
		file.read_exact(&mut buf).unwrap();
		let mut i: usize = 0;
		
		assert!(BigEndian::read_u32(&buf[i..]) == 2051); i += 4;
		
		let n = BigEndian::read_u32(&buf[i..]) as usize; i += 4;
		let h = BigEndian::read_u32(&buf[i..]) as usize; i += 4;
		let w = BigEndian::read_u32(&buf[i..]) as usize; i += 4;
		
		let mut v1 = Vec::with_capacity(n);
		for _ in 0..n {
			let mut v2 = Vec::with_capacity(h*w);
			for _ in 0..h*w {
				v2.push(buf[i]); i += 1;
			}
			v1.push(v2);
		}
		
		(NodeShape::new(1, vec![w, h]), v1)
	}
}