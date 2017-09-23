use new::graph::{GraphDef, SubGraph, DataID, NodeID, NodeTag, Result, Dependencies};
use ndarray::ArrayD;
use rand::thread_rng;
use rand::distributions::{Normal, Sample};
use ordermap::OrderMap;

pub fn normal_fill(v: &mut [f32], mean: f32, std_dev: f32){
	let rng = &mut thread_rng();
	let mut norm = Normal::new(mean as f64, std_dev as f64);
	

	for x in v {
		*x = norm.sample(rng) as f32;
	}

}

pub fn func_fill(v: &mut [f32], func: &mut FnMut()->f64){
	for x in v {
		*x = func() as f32;
	}
}

pub fn generate_input_data(graph: &GraphDef, node_ids:&[NodeID], default_variance: f32, override_distributions: &mut OrderMap<NodeID, Box<FnMut()->f64>>) -> Result<Vec<ArrayD<f32>>> {
	let mut input_data: Vec<ArrayD<f32>> = vec!{};
	for node_id in node_ids {
		let shape = graph.node_shape(node_id)?.to_data_shape()?;

		let mut data = ArrayD::zeros(shape);

		if let Some(func) = override_distributions.get_mut(node_id) {
			func_fill(data.as_slice_mut().unwrap(), &mut **func);
		} else {
			normal_fill(data.as_slice_mut().unwrap(), 0.0, default_variance);
		}

		input_data.push(data);
	}
	Ok(input_data)
}

/// Take a stepsize
pub fn step(step_size: f32, node_ids: &[NodeID], data: &[ArrayD<f32>], results: &OrderMap<DataID, ArrayD<f32>>) -> (Vec<ArrayD<f32>>, Vec<ArrayD<f32>>) {	
		let input_1 = data.iter().cloned().zip(node_ids)
			.map(|(mut data, node_id)|{
				let grad = results.get(&node_id.gradient_id()).unwrap();
				data.scaled_add(-step_size/grad.iter().fold(0.0, |acc, d| acc + d*d), grad);
				data
			}).collect();

		let input_2 = data.iter().cloned().zip(node_ids)
			.map(|(mut data, node_id)|{
				let grad = results.get(&node_id.gradient_id()).unwrap();
				data.scaled_add(step_size/grad.iter().fold(0.0, |acc, d| acc + d*d), grad);
				data
			}).collect();

		(input_1, input_2)
}


/// Returns the relative error of the derivatives with respect to parameters and inputs
///
/// (param_err, input_err)
pub fn test_numeric (graph: GraphDef, step_size: f32, default_variance: f32, override_distributions: &mut OrderMap<NodeID, Box<FnMut()->f64>>) -> Result<(f32, f32)> {
	let dependencies = Dependencies::new(&graph);

	let input_ids: Vec<NodeID> = graph.nodes().iter().filter(|node_id| dependencies.data_inputs(node_id.value_id()).len() == 0 && !graph.is_node_tagged(*node_id, NodeTag::Parameter)).cloned().collect();
	let parameter_ids: Vec<NodeID> = graph.nodes().iter().filter(|node_id| dependencies.data_inputs(node_id.value_id()).len() == 0 && graph.is_node_tagged(*node_id, NodeTag::Parameter)).cloned().collect();

	let inputs_0 = generate_input_data(&graph, &input_ids, default_variance, override_distributions)?;
	let params_0 = generate_input_data(&graph, &parameter_ids, default_variance, override_distributions)?;

	let mut subgraph = graph.subgraph(
		&input_ids.iter().chain(&parameter_ids).map(|node_id| node_id.value_id()).collect::<Vec<_>>(),
		&input_ids.iter().chain(&parameter_ids).map(|node_id| node_id.gradient_id()).collect::<Vec<_>>())?;

	let data_0 = inputs_0.iter().chain(&params_0).cloned().collect();
	let output_0 = subgraph.execute(data_0)?.into_map();
	


	// A small step along along the parameter gradient should produce a predictable change in error.
	let mut param_err = 0.0;
	if parameter_ids.len() > 0 {
		let (params_1, params_2) = step(step_size, &parameter_ids, &params_0, &output_0);

		let data_1 = inputs_0.iter().chain(&params_1).cloned().collect();
		let loss_1 = *subgraph.execute(data_1)?.loss();

		let data_2 = inputs_0.iter().chain(&params_2).cloned().collect();
		let loss_2 = *subgraph.execute(data_2)?.loss();

		let expected_diff = 2.0*step_size;
		let diff = loss_2 - loss_1;
		let err = expected_diff - diff;
		param_err = err.abs()/diff.abs().max(expected_diff.abs());
	}
	
	// A small step along along the input gradient should produce a predictable change in error.
	let mut input_err = 0.0;
	if input_ids.len() > 0 {
		let (inputs_1, inputs_2) = step(step_size, &input_ids, &inputs_0, &output_0);

		let data_1 = inputs_1.iter().chain(&params_0).cloned().collect();
		let loss_1 = *subgraph.execute(data_1)?.loss();

		let data_2 = inputs_2.iter().chain(&params_0).cloned().collect();
		let loss_2 = *subgraph.execute(data_2)?.loss();

		let expected_diff = 2.0*step_size;
		let diff = loss_2 - loss_1;
		let err = expected_diff - diff;
		input_err = err.abs()/diff.abs().max(expected_diff.abs());
	}
	
	Ok((param_err, input_err))

	// TODO A small step in a random direction should produce a change in error proportional to the projection onto the gradient.
	// repeat for parameters and inputs	
}