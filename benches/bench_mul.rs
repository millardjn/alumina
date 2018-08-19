#![feature(test)]

extern crate test;
#[macro_use]
extern crate alumina;
extern crate ndarray;

use test::Bencher;
use alumina::graph::{GraphDef, Result};
use alumina::ops::Op;
use alumina::ops::nn::conv::Conv;
use alumina::ops::math::mul::Mul;
use alumina::ops::regularisation::l2::L2;

use ndarray::{ArrayD, IxDyn};


#[bench]
fn mul_outer_bench_32_forward(bench: &mut Bencher){
	mul_outer(bench, 8, 32, true).unwrap();
}

#[bench]
fn mul_outer_bench_32_backward(bench: &mut Bencher){
	mul_outer(bench, 8, 32, false).unwrap();
}

#[bench]
fn mul_inner_bench_32_forward(bench: &mut Bencher){
	mul_inner(bench, 8, 32, true).unwrap();
}

#[bench]
fn mul_inner_bench_32_backward(bench: &mut Bencher){
	mul_inner(bench, 8, 32, false).unwrap();
}



#[bench]
fn mul_outer_bench_64_forward(bench: &mut Bencher){
	mul_outer(bench, 8, 64, true).unwrap();
}

#[bench]
fn mul_outer_bench_64_backward(bench: &mut Bencher){
	mul_outer(bench, 8, 64, false).unwrap();
}

#[bench]
fn mul_inner_bench_64_forward(bench: &mut Bencher){
	mul_inner(bench, 8, 64, true).unwrap();
}

#[bench]
fn mul_inner_bench_64_backward(bench: &mut Bencher){
	mul_inner(bench, 8, 64, false).unwrap();
}


fn mul_inner(bench: &mut Bencher, n: usize, ch: usize, forward: bool) -> Result<()>{

	let mut g = GraphDef::new();
	
	let node1 = g.new_node(shape![n, 128, 128, ch], "input1", tag![])?;
	let node2 = g.new_node(shape![n, 1, 1, ch], "input2", tag![])?;

	let node3 = g.new_node(shape![n, 128, 128, ch], "output", tag![])?;

	let o1 = Mul::new(&node1, &node2, &node3).add_to(&mut g, tag![])?;
	let _o2 = L2::new(&node3).add_to(&mut g, tag![])?;

	let mut subgraph = if forward {
		g.subgraph(&[node1.value_id(), node2.value_id()], &[node3.value_id()])?
	} else {
		g.subgraph(&[node1.value_id(), node2.value_id()], &[node1.gradient_id(), node2.gradient_id()])?
	};

	let input1 = ArrayD::zeros(IxDyn(&[n, 128, 128, ch]));
	let input2 = ArrayD::zeros(IxDyn(&[n, 1, 1, ch]));

	let params = g.initialise_nodes(&o1.instance().inner_nodes())?;
	let mut input_vec = vec![input1, input2];
	input_vec.extend(params);

	bench.iter(|| {
		let _result = subgraph.execute(input_vec.clone()).unwrap();
	});
	Ok(())
}

fn mul_outer(bench: &mut Bencher, n: usize, ch: usize, forward: bool) -> Result<()>{

	let mut g = GraphDef::new();
	
	let node1 = g.new_node(shape![n, ch, 128, 128], "input1", tag![])?;
	let node2 = g.new_node(shape![n, ch, 1, 1], "input2", tag![])?;

	let node3 = g.new_node(shape![n, ch, 128, 128], "output", tag![])?;

	let o1 = Mul::new(&node1, &node2, &node3).add_to(&mut g, tag![])?;
	let _o2 = L2::new(&node3).add_to(&mut g, tag![])?;

	let mut subgraph = if forward {
		g.subgraph(&[node1.value_id(), node2.value_id()], &[node3.value_id()])?
	} else {
		g.subgraph(&[node1.value_id(), node2.value_id()], &[node1.gradient_id(), node2.gradient_id()])?
	};

	let input1 = ArrayD::zeros(IxDyn(&[n, ch, 128, 128]));
	let input2 = ArrayD::zeros(IxDyn(&[n, ch, 1, 1]));

	let params = g.initialise_nodes(&o1.instance().inner_nodes())?;
	let mut input_vec = vec![input1, input2];
	input_vec.extend(params);

	bench.iter(|| {
		let _result = subgraph.execute(input_vec.clone()).unwrap();
	});
	Ok(())
}