use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use indexmap::{indexset, IndexMap, IndexSet};

use alumina::{
	core::base_ops::{dummy::DummyOp, OpSpecification},
	core::exec::{exec, ExecConfig},
	core::grad::grad,
	core::graph::{Node, NodeTag},
	core::init::gaussian,
	core::subgraph::{execution_subgraph, SubGraph},
	ops::elementwise::relu::relu,
};

fn elementwise_benchmark(c: &mut Criterion) {
	c.bench_function("forward_dummy_small", dummy_small_bench);
	c.bench_function("forward_dummy", dummy_bench);
	c.bench_function("forward_relu", relu_bench);

	// c.bench_function("backward_dummy_small", dummy_small_bench);
	// c.bench_function("backward_dummy", dummy_bench);
	c.bench_function("backward_relu", backward_relu_bench);
}

/// Set value of all inputs using initialisers
///
/// calculate exec subgraph
fn setup(output: &Node, inputs: IndexSet<&Node>) -> SubGraph {
	for input in output
		.graph()
		.nodes_tagged(NodeTag::Parameter)
		.into_iter()
		.chain(inputs.into_iter().cloned())
	{
		input.init_value();
	}

	execution_subgraph(&[] as &[&Node], &[output], true).unwrap()
}

/// Set value of all inputs using initialisers
///
/// Take grad
///
/// calculate exec subgraph
fn setup_backward(output: &Node, inputs: IndexSet<&Node>) -> (IndexSet<Node>, SubGraph) {
	let inputs: IndexSet<Node> = output
		.graph()
		.nodes_tagged(NodeTag::Parameter)
		.into_iter()
		.chain(inputs.into_iter().cloned())
		.collect();
	for input in &inputs {
		input.init_value();
	}

	let grads: IndexSet<_> = grad(&output, inputs.clone()).unwrap().keys().cloned().collect();
	let subgraph = execution_subgraph(&[] as &[&Node], &grads, true).unwrap();
	(grads, subgraph)
}

/// exec overhead
fn dummy_small_bench(b: &mut Bencher<'_>) {
	let output = Node::new(&[13, 11]).set_name("output");
	DummyOp::new().output(&output).build().unwrap();

	let exec_subgraph = setup(&output, indexset![]);
	b.iter(|| {
		exec(
			IndexMap::<Node, _>::new(),
			indexset![output.clone()],
			&mut ExecConfig::default().subgraph(Some(&exec_subgraph)),
		)
		.unwrap()
	})
}

/// exec and allocation overhead
fn dummy_bench(b: &mut Bencher<'_>) {
	let output = Node::new(&[1031, 1033]).set_name("output");
	DummyOp::new().output(&output).build().unwrap();

	let exec_subgraph = setup(&output, indexset![]);
	b.iter(|| {
		exec(
			IndexMap::<Node, _>::new(),
			indexset![output.clone()],
			&mut ExecConfig::default().subgraph(Some(&exec_subgraph)),
		)
		.unwrap()
	})
}

fn relu_bench(b: &mut Bencher<'_>) {
	let input = Node::new(&[1031, 1033]).set_name("input").set_init(gaussian(0.0, 1.0));
	let output = relu(&input).unwrap();

	let exec_subgraph = setup(&output, indexset![&input]);
	b.iter(|| {
		exec(
			IndexMap::<Node, _>::new(),
			indexset![output.clone()],
			&mut ExecConfig::default().subgraph(Some(&exec_subgraph)),
		)
		.unwrap()
	})
}

fn backward_relu_bench(b: &mut Bencher<'_>) {
	let input = Node::new(&[1031, 1033]).set_name("input").set_init(gaussian(0.0, 1.0));
	let output = relu(&input).unwrap();

	let (grads, exec_subgraph) = setup_backward(&output, indexset![&input]);
	b.iter(|| {
		exec(
			IndexMap::<Node, _>::new(),
			grads.clone(),
			&mut ExecConfig::default().subgraph(Some(&exec_subgraph)),
		)
		.unwrap()
	})
}

criterion_group!(benches, elementwise_benchmark);
criterion_main!(benches);
