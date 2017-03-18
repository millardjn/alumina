#![feature(test)]

extern crate test;
extern crate alumina;

use test::Bencher;
use alumina::graph::*;
use alumina::ops::loss::*;
use alumina::ops::conv::Convolution;
use alumina::ops::math;
use alumina::ops::Operation;

#[bench]
fn conv_bench_forward(bench: &mut Bencher){
	conv2D_bench(bench, 32, (128, 128), (3,3), 16, 16, true);
}

#[bench]
fn conv_bench_backward(bench: &mut Bencher){
	conv2D_bench(bench, 32, (128, 128), (3,3), 16, 16, false);
}

fn conv2D_bench(bench: &mut Bencher, n: usize, img: (usize, usize), filter: (usize, usize), ch_in: usize, ch_out: usize, forward: bool){
	let mut graph = Graph::new();

	let n1 = graph.add_input_node(Node::new_sized(ch_in, &[img.0, img.1], "nodein"));
	let n2 = graph.add_output_node(Node::new_sized(ch_out, &[img.0, img.1], "nodeout"));

	let ops: Vec<Box<Operation>> = vec![
		Convolution::new_default(&n1, &n2, &[3, 5]),
	];
	graph.add_operations(ops);

	let input_0 = graph.input_nodes().iter().map(|node| {
		let data_shape = node.shape.to_data_shape(n).unwrap();
		let size = data_shape.flat_size_all();
		NodeData::new(
			data_shape,
			vec![0.99; size]
			//math::random_vector::normal(size, 0.0, 1.0)
		)
	}).collect::<Vec<NodeData>>();
	let params_0 = graph.init_params();
	
	if forward {
		bench.iter(|| {
			let node_data = graph.forward(n, input_0.clone(), &params_0);
		});
	} else {
		bench.iter(|| {
			let (_loss, grad_0, node_data) = graph.backprop(n, input_0.clone(), vec![], &params_0);
		});
	}
}


