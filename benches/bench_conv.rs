#![feature(test)]

extern crate test;
#[macro_use]
extern crate alumina;
extern crate ndarray;

use test::Bencher;
use alumina::graph::{GraphDef, Result};
use alumina::ops::Op;
use alumina::ops::nn::conv::Conv;
use alumina::ops::regularisation::l2::L2;

use ndarray::{ArrayD, IxDyn};

#[bench]
fn conv_bench_128x128_5x5_3_3_forward(bench: &mut Bencher){
	conv_2d_bench(bench, 8, (128, 128), (5,5), 3, 3, true).unwrap();
}

#[bench]
fn conv_bench_128x128_5x5_3_3_backward(bench: &mut Bencher){
	conv_2d_bench(bench, 8, (128, 128), (5,5), 3, 3, false).unwrap();
}

#[bench]
fn conv_bench_64x64_3x3_16_16_forward(bench: &mut Bencher){
	conv_2d_bench(bench, 8, (64, 64), (3,3), 16, 16, true).unwrap();
}

#[bench]
fn conv_bench_64x64_3x3_16_16_backward(bench: &mut Bencher){
	conv_2d_bench(bench, 8, (64, 64), (3,3), 16, 16, false).unwrap();
}

#[bench]
fn conv_bench_64x64_3x3_64_64_forward(bench: &mut Bencher){
	conv_2d_bench(bench, 8, (64, 64), (3,3), 64, 64, true).unwrap();
}

#[bench]
fn conv_bench_64x64_3x3_64_64_backward(bench: &mut Bencher){
	conv_2d_bench(bench, 8, (64, 64), (3,3), 64, 64, false).unwrap();
}

fn conv_2d_bench(bench: &mut Bencher, n: usize, img: (usize, usize), filter: (usize, usize), ch_in: usize, ch_out: usize, forward: bool) -> Result<()>{

	let mut g = GraphDef::new();
	
	let node1 = g.new_node(shape![n, img.0, img.1, ch_in], "input", tag![])?;
	let node2 = g.new_node(shape![Unknown, Unknown, Unknown, ch_out], "conv", tag![])?;

	let o1 = Conv::new(&node1, &node2, &[filter.0, filter.1]).init(Conv::msra(1.0)).add_to(&mut g, tag![])?;
	let _o2 = L2::new(&node2).add_to(&mut g, tag![])?;

	let mut subgraph = if forward {
		g.subgraph(&[node1.value_id(), o1.instance().inner_nodes()[0].value_id()], &[node2.value_id()])?
	} else {
		g.subgraph(&[node1.value_id(), o1.instance().inner_nodes()[0].value_id()], &[node1.gradient_id(), o1.instance().inner_nodes()[0].gradient_id()])?
	};

	let input = ArrayD::zeros(IxDyn(&[n, img.0, img.1, ch_in]));

	let params = g.initialise_nodes(&o1.instance().inner_nodes())?;
	let mut input_vec = vec![input];
	input_vec.extend(params);

	bench.iter(|| {
		let _result = subgraph.execute(input_vec.clone()).unwrap();
	});
	Ok(())
}



