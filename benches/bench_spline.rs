#![feature(test)]

extern crate test;
#[macro_use]
extern crate alumina;
extern crate ndarray;

use test::Bencher;
use alumina::graph::{GraphDef, Result};
use alumina::ops::Op;
use alumina::ops::activ::spline::Spline;
use alumina::ops::regularisation::l2::L2;

use ndarray::{ArrayD, IxDyn};


#[bench]
fn spline_outer_bench_32_forward(bench: &mut Bencher){
	spline_outer(bench, 8, 32, true).unwrap();
}

#[bench]
fn spline_outer_bench_32_backward(bench: &mut Bencher){
	spline_outer(bench, 8, 32, false).unwrap();
}

#[bench]
fn spline_inner_bench_32_forward(bench: &mut Bencher){
	spline_inner(bench, 8, 32, true).unwrap();
}

#[bench]
fn spline_inner_bench_32_backward(bench: &mut Bencher){
	spline_inner(bench, 8, 32, false).unwrap();
}



#[bench]
fn spline_outer_bench_64_forward(bench: &mut Bencher){
	spline_outer(bench, 8, 64, true).unwrap();
}

#[bench]
fn spline_outer_bench_64_backward(bench: &mut Bencher){
	spline_outer(bench, 8, 64, false).unwrap();
}

#[bench]
fn spline_inner_bench_64_forward(bench: &mut Bencher){
	spline_inner(bench, 8, 64, true).unwrap();
}

#[bench]
fn spline_inner_bench_64_backward(bench: &mut Bencher){
	spline_inner(bench, 8, 64, false).unwrap();
}


fn spline_inner(bench: &mut Bencher, n: usize, ch: usize, forward: bool) -> Result<()>{

	let mut g = GraphDef::new();
	
	let node1 = g.new_node(shape![n, 128, 128, ch], "input1", tag![])?;

	let node3 = g.new_node(shape![n, 128, 128, ch], "output", tag![])?;

	let o1 = Spline::new(&node1, &node3).shared_axes(&[0, 1]).add_to(&mut g, tag![])?;
	let _o2 = L2::new(&node3).add_to(&mut g, tag![])?;

	let node2 = o1.instance().inner_nodes().remove(0);

	let mut subgraph = if forward {
		g.subgraph(&[node1.value_id(), node2.value_id()], &[node3.value_id()])?
	} else {
		g.subgraph(&[node1.value_id(), node2.value_id()], &[node1.gradient_id(), node2.gradient_id()])?
	};

	let input1 = ArrayD::zeros(IxDyn(&[n, 128, 128, ch]));

	let params = g.initialise_nodes(&o1.instance().inner_nodes())?;
	let mut input_vec = vec![input1];
	input_vec.extend(params);

	bench.iter(|| {
		let _result = subgraph.execute(input_vec.clone()).unwrap();
	});
	Ok(())
}

fn spline_outer(bench: &mut Bencher, n: usize, ch: usize, forward: bool) -> Result<()>{

	let mut g = GraphDef::new();
	
	let node1 = g.new_node(shape![n, 128, 128, ch], "input1", tag![])?;

	let node3 = g.new_node(shape![n, 128, 128, ch], "output", tag![])?;

	let o1 = Spline::new(&node1, &node3).shared_axes(&[2, 3]).add_to(&mut g, tag![])?;
	let _o2 = L2::new(&node3).add_to(&mut g, tag![])?;

	let node2 = o1.instance().inner_nodes().remove(0);

	let mut subgraph = if forward {
		g.subgraph(&[node1.value_id(), node2.value_id()], &[node3.value_id()])?
	} else {
		g.subgraph(&[node1.value_id(), node2.value_id()], &[node1.gradient_id(), node2.gradient_id()])?
	};

	let input1 = ArrayD::zeros(IxDyn(&[n, 128, 128, ch]));

	let params = g.initialise_nodes(&o1.instance().inner_nodes())?;
	let mut input_vec = vec![input1];
	input_vec.extend(params);

	bench.iter(|| {
		let _result = subgraph.execute(input_vec.clone()).unwrap();
	});
	Ok(())
}