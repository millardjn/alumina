
use graph::*;
use vec_math::VecMath;



pub fn test_numeric (mut graph: Graph, input_variance: f32, step_size: f32){
	assert!(graph.nodes().iter().all(|n| n.shape.is_fixed()), "Must use fixed size nodes in graph when doing numerical testing.");
	let n = 13;
	
	
	let training = graph.training_input_nodes().iter().map(|node| {
		let data_shape = node.shape.to_data_shape(n).unwrap();
		let size = data_shape.flat_size_all();
		NodeData::new(
			data_shape,
			random_vector::normal(size, 0.0, input_variance)
		)
	}).collect::<Vec<NodeData>>();
	
	let input_0 = graph.input_nodes().iter().map(|node| {
		let data_shape = node.shape.to_data_shape(n).unwrap();
		let size = data_shape.flat_size_all();
		NodeData::new(
			data_shape,
			random_vector::normal(size, 0.0, input_variance)
		)
	}).collect::<Vec<NodeData>>();
	
	let params_0 = graph.init_params();
	
	let (_loss_0, grad_0, node_data) = graph.backprop(n, input_0.clone(), training.clone(), &params_0);
	
	
	let tolerance = 0.001; //should be near zero but some functions are less stable.
	
	
	// A small step along along the returned gradient should produce a correspending change in error.
	// repeat for parameters and inputs
	if params_0.len() > 0 {

		let params_1 = params_0.add_scaled(&grad_0, -step_size);
		let params_2 = params_0.add_scaled(&grad_0, step_size);
		let (loss_1, _, _) = graph.backprop(n, input_0.clone(), training.clone(), &params_1);
		let (loss_2, _, _) = graph.backprop(n, input_0.clone(), training.clone(), &params_2);

		let expected_diff = 2.0*step_size*grad_0.dot(&grad_0);
		let diff = loss_2 - loss_1;
		let err = expected_diff - diff;
		let rel_err = err.abs()/diff.abs().max(expected_diff.abs());
		
		println!("err_1 {}", rel_err);
		assert!(rel_err < tolerance);
	}
	
	
	if graph.input_node_IDs().len() > 0{// change inputs in direction of error gradients and check for correct change in error
		
		let input_norm_sqr = graph.input_node_IDs().iter()
			.map(|id| &node_data[id.ind])
			.map(|node| node.derivatives.dot(&node.derivatives))
			.fold(0.0, |acc, n| acc + n);
		

		let input_1 = graph.input_node_IDs().iter()
			.map(|id| &node_data[id.ind])
			.map(|node| NodeData::new(node.shape.clone(), node.values.add_scaled(&node.derivatives, -step_size)))
			.collect();
			
		let input_2 = graph.input_node_IDs().iter()
			.map(|id| &node_data[id.ind])
			.map(|node| NodeData::new(node.shape.clone(), node.values.add_scaled(&node.derivatives, step_size)))
			.collect();

		let (loss_1, _, _) = graph.backprop(n, input_1, training.clone(), &params_0);
		let (loss_2, _, _) = graph.backprop(n, input_2, training.clone(), &params_0);
		
		let expected_diff = 2.0*step_size*input_norm_sqr;
		let diff = loss_2 - loss_1;
		let err = expected_diff - diff;
		let rel_err = err.abs()/diff.abs().max(expected_diff.abs());

		
		println!("err_2 {}", rel_err);
		assert!(rel_err < tolerance);
	}
	
	
	// A small step in a random direction should produce a change in error proportional to the projection onto the gradient.
	// repeat for parameters and inputs

	
}



pub mod random_vector{
	
	use rand::*;
	use rand::distributions::*;

	pub fn randomize_signs(vector: &mut [f32]){
		let mut rng = thread_rng();
		for n in vector{
			*n = *n * if rng.next_u32() & 1 == 0 {1.0} else {-1.0};
		}
	}

	pub fn lognormal (len: usize, mean: f32, std_dev: f32) -> Vec<f32> {
		let mut v = vec![0.0; len];
		lognormal_fill(&mut v, mean, std_dev);
		v
	}
	
	pub fn normal (len: usize, mean: f32, std_dev: f32) -> Vec<f32> {
		let mut v = vec![0.0; len];
		normal_fill(&mut v, mean, std_dev);
		v
	}

	pub fn lognormal_fill(v: &mut [f32], mean: f32, std_dev: f32){
		let rng = &mut thread_rng();
		let mut lognorm = LogNormal::new(mean as f64, std_dev as f64);
		
		
		for x in v {
			*x = lognorm.sample(rng) as f32;
		}
	}
	
	pub fn normal_fill(v: &mut [f32], mean: f32, std_dev: f32){
		let rng = &mut thread_rng();
		let mut norm = Normal::new(mean as f64, std_dev as f64);
		

		for x in v {
			*x = norm.sample(rng) as f32;
		}

	}
}



#[inline(never)]
pub fn ssd_error(input: &[f32], target: &[f32], scale: f32, derivs: &mut[f32], error: &mut f32){ // Currently doesnt vectorise to to sequential err additions
	debug_assert_eq!(input.len(), target.len());
	debug_assert_eq!(input.len(), derivs.len());
	
	let n = input.len();
	let input = &input[..n];
	let target = &target[..n];
	let derivs = &mut derivs[..n];

	const SIMD: usize = 16;
	
	if n >= SIMD {
		let mut errs = [0.;SIMD];

		for i in 0..n/SIMD {

			let input = &input[i*SIMD..][..SIMD];
			let target = &target[i*SIMD..][..SIMD];
			let derivs = &mut derivs[i*SIMD..][..SIMD];

			for j in 0..SIMD{
				let diff = input[j]-target[j];
				errs[j] += diff*diff;
				derivs[j] += 2.0*diff*scale;		
			}
		}

		for i in 0..SIMD {
			*error += errs[i]*scale;
		}
	}

	for j in (n/SIMD)*SIMD..n {
		let diff = input[j]-target[j];
		*error += diff*diff*scale;
		derivs[j] += 2.0*diff*scale;
	}

}












