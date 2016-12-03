use opt::*;
use graph::*;
use vec_math::{VecMath, VecMathMut, VecMathMove};
use std::f32;
use opt::supplier::Supplier;


const NUM_BINS: u32 = 8;
const PWR: f32 = 1.0;
/// Adapt-all-the-things SGD
pub struct Asgd<'a>{
	max_evals: Option<u64>,
	min_loss: Option<f32>,
	min_loss_delta: Option<f32>,
	eval_count: u64,
	step_count: u64,
	graph: &'a mut Graph,
	curvature_est: Vec<f32>,
	decay_const: f32,
	rate: f32,
	batch_size: f32,
	min_batch_size: f32,
	previous_derivs: Vec<f32>,
	previous_change: Vec<f32>,
	previous_err: f32,
	step_callback: Vec<Box<FnMut(f32, u64, u64, &mut Graph, &[f32])->bool>>,
}

impl <'a> Asgd<'a> {
	pub fn new(graph: &'a mut Graph) -> Asgd<'a>{
		let num_params = graph.num_params();
		Asgd{
			max_evals: None,
			min_loss: None,
			min_loss_delta: None,
			eval_count: 0,
			step_count: 0,
			graph: graph,
			curvature_est: vec![0.0; num_params],
			decay_const: 0.9,
			rate: 0.00001,
			batch_size: 1.0,
			min_batch_size: 1.0,
			previous_derivs: vec![0.0; num_params],
			previous_change: vec![0.0; num_params],
			previous_err: 0.,
			step_callback: vec![],
		}
	}
	
	pub fn set_min_batch_size(&mut self, min: f32){
		self.min_batch_size = min;
		self.batch_size = min;
	}
	
	pub fn reset_eval_steps(&mut self){
		self.eval_count = 0;
		self.step_count = 0;
	}

	/// Returns error and error derivatives
	fn part_step(&mut self, training_set: &mut Supplier, params: &[f32], batch_size: u64) -> (f32, Vec<f32>){

			let (input, training_input) = training_set.next_n(batch_size as usize);
			let (mut err, mut param_derivs, _data) = self.graph.backprop(batch_size as usize, input, training_input, &params);
			
			err /= batch_size as f32;
			param_derivs.scale_mut(1.0/batch_size as f32);
			
			self.eval_count += batch_size;
			(err, param_derivs)
			
	}
}

impl<'a> Optimiser<'a> for Asgd<'a>{
	fn set_max_evals(&mut self, max_evals: u64){
		self.max_evals = Some(max_evals);
	}
	fn set_min_loss(&mut self, min_loss: f32){
		self.min_loss = Some(min_loss);
	}
	fn set_min_epoch_loss_delta(&mut self, delta: f32){
		self.min_loss_delta = Some(delta);
	}
	fn get_max_evals(&self) -> Option<u64>{
		self.max_evals
	}
	fn get_min_loss(&self) -> Option<f32>{
		self.min_loss
	}
	fn get_min_epoch_loss_delta(&self) -> Option<f32>{
		self.min_loss_delta
	}
	fn add_step_callback<F>(&mut self, func: F) where F: FnMut(f32, u64, u64, &mut Graph, &[f32])->bool + 'static{ // err, step, evaluations, graph, params
		self.step_callback.push(Box::new(func));

	}
	// fn add_evaluation_callback<F>(&mut self, func: F) where F: FnMut(f32, u64, u64, &mut Graph, &[f32])->bool{ // err, step, evaluations, graph, params{
	// 	self.eval_callback.push(Box::new(func));
	// }
	fn get_graph(&mut self) -> &mut Graph{
		&mut self.graph
	}	
	fn optimise_from(&mut self, training_set: &mut Supplier,  mut params: Vec<f32>) -> Vec<f32>{
		let mut cont = true;
		while self.max_evals.map_or(true, |max| self.eval_count < max) & cont{
			let (err, new_params) = self.step(training_set, params);
			params = new_params;

			
			for func in self.step_callback.iter_mut(){
				cont &= func(err, self.step_count, self.eval_count, self.graph, &params);
			}
		}
		
		params
	}

	
	fn step(&mut self, training_set: &mut Supplier, params: Vec<f32>) -> (f32, Vec<f32>){

			// Take multiple measurements of error and gradient, then find L2 variance in derivative vectors
			
			let batch_ceil = self.batch_size.ceil() as u64;
			let results = (0..NUM_BINS).map(|_| self.part_step(training_set, &params, batch_ceil)).collect::<Vec<_>>();
			
			let err: f32 = results.iter().fold(0.0f32, |acc, &(err, _)| acc + err)/NUM_BINS as f32;
			let mean: Vec<f32> = results.iter().fold(vec![0.0f32;params.len()], |acc, &(_, ref derivs)| acc.add_move(&derivs)).scale_move(1.0/NUM_BINS as f32);


			// let var = results.iter()
			// 	.map(|&(_, ref derivs)| derivs.add_scaled(&mean, -1.0))
			// 	.fold(0.0, |acc, diff| acc + diff.dot(&diff))/(NUM_BINS - 1) as f32;
			// let std_err_var = var/NUM_BINS as f32;
			// let rel_std_err = (std_err_var/mean.dot(&mean)).sqrt();			


			let mean_norm = mean.normalise();
			let var2 = results.iter()
				.map(|&(_, ref derivs)| derivs.add_scaled(&mean, -1.0).dot(&mean_norm))
				.fold(0.0, |acc, diff| acc + diff*diff)/(NUM_BINS - 1) as f32;
			let std_err_var2 = var2/NUM_BINS as f32;
			let rel_std_err = (std_err_var2/mean.dot(&mean)).sqrt();		

			
			
			
			self.curvature_est.scale_mut(self.decay_const);			
			
			// New ADAM curvature
			for &(_, ref derivs) in results.iter() {
				self.curvature_est.add_scaled_mut(&derivs.elementwise_mul(&derivs), (1.0 - self.decay_const)/NUM_BINS as f32);				
			}
			
			// Normal ADAM curvature		
			//self.curvature_est.add_scaled_mut(&mean.elementwise_mul(&mean), 1.0 - self.decay_const);
			
			let corrected_curvature = self.curvature_est.scale(1.0/(1.0 - self.decay_const.powi(self.step_count as i32 + 1)));
			let cond_derivs: Vec<f32> = mean.iter().zip(&corrected_curvature).map(|(m,c)| m/(c.sqrt() + 1e-8).powf(PWR)).collect();


			//ADAMAX
			// self.curvature_est = self.curvature_est.iter().zip(mean.iter()).map(|(&c, &m)| m.abs().max(c * self.decay_const)).collect();
			// let cond_derivs: Vec<f32> = mean.iter().zip(self.curvature_est.iter()).map(|(m,c)| m/(c + 1e-6)).collect();
			
			//let cond_prev_derivs: Vec<f32> = self.previous_derivs.iter().zip(&corrected_curvature).map(|(m,c)| m/(c.sqrt() + 1e-8).powf(PWR)).collect();
	
			//let sim = mean.cos_similarity(&self.previous_derivs);	// Adapt learning rate based on inter-step cosine similarity
			let sim = if self.step_count == 0 {
				0.0
			} else {
				//mean.cos_similarity(&self.previous_derivs)
				(mean.dot(&self.previous_derivs)/self.previous_derivs.dot(&self.previous_derivs)).max(-1.0).min(1.0) // clipping prevents unexpected noise results from blowing up step size adaption.
			};
			let new_rate = self.rate*1.33f32.powf(sim+0.0625);//*1.25; +0.0625
			
			//let biased_sim = sim - 0.25;
			//let repeat = 1.0 - 1.0/(1.0-biased_sim.min(0.5)); // use a parabolic approximation to move to the minima, in the case of sim > 0 limit extrapolation to 1*previous step
			//let change = cond_derivs.scale(-new_rate).add_scaled(&cond_prev_derivs, repeat*self.rate +biased_sim*new_rate); // since we use the previous parabolic esimate, also cancel out change in the direction of the previous step (sim*new_rate).



			let change = cond_derivs.scale(-new_rate);

			// if self.step_count != 0 {
			// 	// let norm_prev_change = self.previous_change.normalise();
			// 	// let a = self.previous_derivs.dot(&norm_prev_change); // -ve
			// 	// let b = mean.dot(&norm_prev_change); // -/+ ve
			// 	// let c = a/(a-b) - 1.0; // range -1 to + 1

			// 	// let d = cond_derivs.dot(&norm_prev_change);
			// 	//println!("a:{} b:{} c:{} d:{}", a, b, c, d);
			// 	//change = change.add_scaled(&self.previous_change, -c).add_scaled(&norm_prev_change, d*new_rate);
			// 	change = change.add_scaled(&self.previous_change, 0.6666666);
			// }


			

			// print progress (this should be moved to a callback lambda)
			if self.step_count == 0 {println!("");println!("count\terr\trel_err\tbatchSize\tcos_sim\trate\tmovement");}
			println!("{}\t{:.4}\t{:.4}\t{}x{}\t{:.4}\t{:.4e}\t{:.4e}", training_set.samples_taken(), err, rel_std_err, NUM_BINS, self.batch_size.ceil(), sim, self.rate, change.norm2());

			
			
			// Adapt batch size based on derivative relative err vs target relative variance
			let target_err = 0.5;//0.5f32;//f32.sqrt().sqrt();
			
			self.batch_size *= if rel_std_err > target_err {
					(rel_std_err/target_err).powf(0.25) // increase batch size
				} else {
					(rel_std_err/target_err).powf(0.25) // decrease batch size
				};
			self.batch_size = self.batch_size.max(self.min_batch_size);			
	
			

			let new_params = params.add(&change);

			self.rate = new_rate;
			self.previous_derivs = mean;
			self.previous_change = change;
			self.previous_err = err;
			self.step_count += 1;
			
			(err, new_params)
	}
		
}





pub struct Asgd2<'a>{
	max_evals: Option<u64>,
	min_loss: Option<f32>,
	min_loss_delta: Option<f32>,
	eval_count: u64,
	step_count: u64,
	graph: &'a mut Graph,
	curvature_est: Vec<f32>,
	decay_const: f32,
	rate: f32,
	batch_size: f32,
	min_batch_size: usize,
	averaged_derivs: Vec<f32>,
	prev_derivs: Vec<f32>,
	step_callback: Vec<Box<FnMut(f32, u64, u64, &mut Graph, &[f32])->bool>>,
	cos_sim_var: f32,
}

impl <'a> Asgd2<'a> {
	pub fn new(graph: &'a mut Graph) -> Asgd2<'a>{
		let num_params = graph.num_params();
		Asgd2{
			max_evals: None,
			min_loss: None,
			min_loss_delta: None,
			eval_count: 0,
			step_count: 0,
			graph: graph,
			curvature_est: vec![0.0; num_params],
			decay_const: 0.9,
			rate: 0.001,
			batch_size: 1.0,
			min_batch_size: 1,
			averaged_derivs: vec![0.0; num_params],
			prev_derivs: vec![0.0; num_params],
			step_callback: vec![],
			cos_sim_var: 0.0
		}
	}
	
	pub fn set_min_batch_size(&mut self, min: usize){
		self.min_batch_size = min;
		self.batch_size = min as f32;
	}
	
	pub fn reset_eval_steps(&mut self){
		self.eval_count = 0;
		self.step_count = 0;
	}

	/// Returns error and error derivatives
	fn part_step(&mut self, training_set: &mut Supplier, params: &[f32], batch_size: u64) -> (f32, Vec<f32>){

			let (input, training_input) = training_set.next_n(batch_size as usize);
			let (mut err, mut param_derivs, _data) = self.graph.backprop(batch_size as usize, input, training_input, &params);
			
			err /= batch_size as f32;
			param_derivs.scale_mut(1.0/batch_size as f32);
			
			self.eval_count += batch_size;
			(err, param_derivs)
			
	}

	/// mutably updates batch_size and returns relative error measure
	fn update_batch_size(&mut self, mean: &[f32], results: &[(f32, Vec<f32>)]) -> f32{

		//-- Vector Dispersion measures
		let mean_norm = mean.norm2();
		let (max, avg) = results.iter()
			.map(|&(_, ref derivs)| derivs.add_scaled(&mean, -1.0))
			.fold((0.0f32, 0.0f32), |(max, avg), diff| {
				let len = diff.norm2()/((NUM_BINS - 1) as f32 * mean_norm);
				(max.max(len), avg + len/NUM_BINS as f32)
			} );
		// max being below 1.0 shows that there isn't one vector that is orders of magnitude larger than the others and is swamping the mean
		let max_target = 0.66;

		// avg being less than 1.0 shows that the vectors arent just orthogonal vectors of equal length (random vector in high dimensions are likely to be orthogonal), therefore there is some signal in the noise
		let avg_target = 0.66;

		let rel_err = (max/max_target).max(avg/avg_target).max(0.125);
		
		//print!("{} {} ", avg, max);
				
		//-- Vector Variance
		// let var = results.iter()
		// 	.map(|&(_, ref derivs)| derivs.add_scaled(&mean, -1.0))
		// 	.fold(0.0, |acc, diff| acc + diff.dot(&diff))/(NUM_BINS - 1) as f32;
		// let std_err_var = var/NUM_BINS as f32;
		// let rel_err = (std_err_var/mean.dot(&mean)).sqrt();			
		// let target_err = 0.9;
		
		
		//-- Variance when projected onto mean vector
		// let mean_norm = mean.normalise();
		// let var2 = results.iter()
		// 	.map(|&(_, ref derivs)| derivs.add_scaled(&mean, -1.0).dot(&mean_norm))
		// 	.fold(0.0, |acc, diff| acc + diff*diff)/(NUM_BINS - 1) as f32;
		// let std_err_var2 = var2/NUM_BINS as f32;
		// let rel_err = (std_err_var2/mean.dot(&mean)).sqrt();		
		// let target_err = 0.25;//
		
		
		// Adapt batch size based on derivative relative err vs target relative variance			
		self.batch_size *= if rel_err > 1.0 {
				rel_err.powf(0.25) // increase batch size
			} else {
				rel_err.powf(0.125) // decrease batch size
			};
		self.batch_size = self.batch_size.max(self.min_batch_size as f32);
		rel_err
	}
}

impl<'a> Optimiser<'a> for Asgd2<'a>{
	fn set_max_evals(&mut self, max_evals: u64){
		self.max_evals = Some(max_evals);
	}
	fn set_min_loss(&mut self, min_loss: f32){
		self.min_loss = Some(min_loss);
	}
	fn set_min_epoch_loss_delta(&mut self, delta: f32){
		self.min_loss_delta = Some(delta);
	}
	fn get_max_evals(&self) -> Option<u64>{
		self.max_evals
	}
	fn get_min_loss(&self) -> Option<f32>{
		self.min_loss
	}
	fn get_min_epoch_loss_delta(&self) -> Option<f32>{
		self.min_loss_delta
	}
	fn add_step_callback<F>(&mut self, func: F) where F: FnMut(f32, u64, u64, &mut Graph, &[f32])->bool + 'static{ // err, step, evaluations, graph, params
		self.step_callback.push(Box::new(func));

	}
	fn get_graph(&mut self) -> &mut Graph{
		&mut self.graph
	}	
	fn optimise_from(&mut self, training_set: &mut Supplier,  mut params: Vec<f32>) -> Vec<f32>{
		let mut cont = true;
		while self.max_evals.map_or(true, |max| self.eval_count < max) & cont{
			let (err, new_params) = self.step(training_set, params);
			params = new_params;

			
			for func in self.step_callback.iter_mut(){
				cont &= func(err, self.step_count, self.eval_count, self.graph, &params);
			}
		}
		
		params
	}

	
	fn step(&mut self, training_set: &mut Supplier, params: Vec<f32>) -> (f32, Vec<f32>){

		// Take multiple measurements of error and gradient, then find L2 variance in derivative vectors
		
		let batch_ceil = self.batch_size.floor() as u64;
		let results = (0..NUM_BINS).map(|_| self.part_step(training_set, &params, batch_ceil)).collect::<Vec<_>>();
		
		let err: f32 = results.iter().fold(0.0f32, |acc, &(err, _)| acc + err)/NUM_BINS as f32;
		let mean: Vec<f32> = results.iter().fold(vec![0.0f32;params.len()], |acc, &(_, ref derivs)| acc.add_move(&derivs)).scale_move(1.0/NUM_BINS as f32);


	// {
	// 	let avg_norm_sqr = self.averaged_derivs.dot(&self.averaged_derivs);
	// 	let mean_dot = mean.dot(&self.averaged_derivs)/avg_norm_sqr;
	// 	let prev_dot = self.prev_derivs.dot(&self.averaged_derivs)/avg_norm_sqr;


	// 	let mean_reg = mean.add_scaled(&self.averaged_derivs, -mean_dot);
	// 	let prev_reg = self.prev_derivs.add_scaled(&self.averaged_derivs, -prev_dot);
	// 	let mut sim = mean_reg.cos_similarity(&prev_reg);
	// 	//if sim > 0.0 {(sim = sim/2.0);}

	// 	let rate: f32 = 1.05;
	// 	let mut change = rate.powf(sim-0.25);
	// 	//if change > 1.0 {change = change.powf(0.25);}
	// 	self.decay_const = self.decay_const.powf(change).max(0.25).min(0.9999);
	// 	println!("sim: {} momen: {} chan: {}", sim, self.decay_const, change);

	// }

		
		let curv_decay = self.decay_const.powf(0.09539).max(0.9);
		self.curvature_est.scale_mut(curv_decay);			
		
		// New ADAM curvature
		for &(_, ref derivs) in results.iter() {
			let var = derivs.add_scaled(&self.averaged_derivs, -1.0);
			self.curvature_est.add_scaled_mut(&var.elementwise_mul(&var), (1.0 - curv_decay)/NUM_BINS as f32);	
			//self.curvature_est.add_scaled_mut(&derivs.elementwise_mul(&derivs), (1.0 - curv_decay)/NUM_BINS as f32);				
		}
		// let var = mean.add_scaled(&self.averaged_derivs, -1.0);
		// self.curvature_est.add_scaled_mut(&var.elementwise_mul(&var), (1.0 - curv_decay) as f32);


		let corrected_curvature = self.curvature_est.scale(1.0/(1.0 - curv_decay.powi(self.step_count as i32 + 1)));
		


		//ADAMAX
		// self.curvature_est = self.curvature_est.iter().zip(mean.iter()).map(|(&c, &m)| m.abs().max(c * curv_decay)).collect();
		// let cond_derivs: Vec<f32> = mean.iter().zip(self.curvature_est.iter()).map(|(m,c)| m/(c + 1e-6)).collect();
		


		let mut sim = mean.cos_similarity(&self.averaged_derivs); // Adapt learning rate based on inter-step cosine similarity
		//(mean.dot(&self.averaged_derivs)/self.averaged_derivs.dot(&self.averaged_derivs)).max(-4.0).min(2.0) // clipping prevents unexpected noise results from blowing up step size adaption.

		//if sim < 0.0 {(sim = sim/2.0);}
		// sim = if sim > 0.0 {
		// 	sim.min(0.2)
		// } else {
		// 	(sim*0.5).max(-0.2)
		// };

		sim = (sim+0.0625).min(0.5).max(-0.5);
		let new_rate = self.rate*2.0f32.sqrt().powf(sim);//*1.25; +0.0625
		

		let mut new_average_derivs = self.averaged_derivs.scale(self.decay_const);
		new_average_derivs.add_scaled_mut(&mean, 1.0 - self.decay_const);

		let cond_derivs: Vec<f32> = new_average_derivs.iter().zip(&corrected_curvature).map(|(m,c)| m/(c.sqrt() + 1e-8).powf(PWR)).collect();
		let change = cond_derivs.scale(-new_rate);


		let rel_err = self.update_batch_size(&mean, &results);

		// print progress (this should be moved to a callback lambda)
		if self.step_count == 0 {println!("");println!("count\terr\trel_err\tbatchSize\tcos_sim\trate\tmovement");}
		println!("{}\t{}\t{:.4}\t{}x{}\t{:.4}\t{:.4e}\t{:.4e}", training_set.samples_taken(), err, rel_err, NUM_BINS, self.batch_size.floor(), sim, self.rate, change.norm2());

		//:.4
		


		

		let new_params = params.add(&change);

		self.rate = new_rate;
		self.averaged_derivs = new_average_derivs;
		self.step_count += 1;
		self.prev_derivs = mean;
		(err, new_params)
	}






		
}