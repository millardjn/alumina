
use graph::*;
use ops::activ::ReLU;
use ops::basic::LinearMap;
use ops::loss::MseLoss;
// use ops::math;
use shape::*;
use ops::*;
#[test]
pub fn build_circular_graph() {
	
	
	let mut g = Graph::new();
	
	let layer0 = g.add_input_node(Node::new_flat(10, "start"));
	let layer1 = g.add_node(Node::new_flat(100, "hidden1"));
	let layer2 = g.add_node(Node::new_flat(100, "hidden2"));
	let layer3 = g.add_node(Node::new_flat(100, "hidden3"));
	let layer4 = g.add_output_node(Node::new_flat(10, "end"));
	
	
	let ops: Vec<Box<Operation>> = vec![
		LinearMap::new_default(&layer0, &layer1),
		ReLU::new(&layer1, &layer2, "ReLU"),
		ReLU::new(&layer2, &layer3, "ReLU"),
		ReLU::new(&layer3, &layer1, "ReLU"),
		LinearMap::new_default(&layer3, &layer4),
	];
	g.add_operations(ops);
	
	let params = math::random_vector::normal(g.num_params(), 0.0, 1.0);// vec![0.0; g.num_params()];
	println!["{}", params.len()];
	let n = 2;
	g.forward(n, vec![NodeData::new(DataShape::new_flat(10, n), math::random_vector::normal(10*n, 0.0, 1.0))], &params);
	
	println!["{:?}", g.evaluation_order()];
}

#[test]
pub fn build_graph() {
	
	
	let mut g = Graph::new();
	
	let layer0 = g.add_input_node(Node::new_flat(10, "start"));
	let layer1 = g.add_node(Node::new_flat(100, "hidden1"));
	let layer2 = g.add_node(Node::new_flat(100, "hidden2"));
	let layer3 = g.add_node(Node::new_flat(10, "hidden3"));
	let layer4 = g.add_output_node(Node::new_flat(10, "end"));
	
	
	let ops: Vec<Box<Operation>> = vec![
		LinearMap::new_default(&layer0, &layer1),
		ReLU::new(&layer1, &layer2, "ReLU"),
		LinearMap::new_default(&layer2, &layer3),
		ReLU::new(&layer3, &layer4, "ReLU"),
		MseLoss::new_default(&layer4, &layer0),
	];
	g.add_operations(ops);

	
	let params = math::random_vector::normal(g.num_params(), 0.0, 1.0);// vec![0.0; g.num_params()];
	println!["{}", params.len()];
	let n = 2;
	g.forward(n, vec![NodeData::new(DataShape::new_flat(10, n), math::random_vector::normal(10*n, 0.0, 1.0))], &params);
	let (loss, _grad, _node_data) = g.backprop(n, vec![NodeData::new(DataShape::new_flat(10, n), math::random_vector::normal(10*n, 0.0, 1.0))], vec![], &params);
	println!("loss: {}", loss);
	println!["{:?}", g.evaluation_order()];
	
}


