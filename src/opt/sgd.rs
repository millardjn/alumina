use opt::*;
use graph::*;
use vec_math::{VecMathMut, VecMathMove};
use std::f32;
use opt::supplier::Supplier;

pub struct Sgd<'a>{
	max_evals: Option<u64>,
	min_loss: Option<f32>,
	min_loss_delta: Option<f32>,
	eval_count: u64,
	graph: &'a mut Graph,
	rate: f32,
	batch_size: u32,
	_momentum: Option<f32>,
	_momentum_vec: Vec<f32>,
	step_callback: Vec<Box<FnMut(f32, u64, u64, &mut Graph, &[f32])->bool>>,
}

impl <'a> Sgd<'a> {
	pub fn new(graph: &'a mut Graph, rate: f32, batch_size: u32, momentum: Option<f32>) -> Sgd<'a>{
		let num_params = graph.num_params();
		Sgd{
			max_evals: None,
			min_loss: None,
			min_loss_delta: None,
			eval_count: 0,
			graph: graph,
			rate: rate,
			batch_size: batch_size,
			_momentum: None,
			_momentum_vec: if momentum.is_some() {vec![0.0; num_params]} else {vec![]},
			step_callback: vec![],
		}
	}
}

impl<'a> Optimiser<'a> for Sgd<'a>{
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
	fn get_graph(&mut self) -> &mut Graph{
		&mut self.graph
	}
	
	fn add_step_callback<F>(&mut self, func: F) where F: FnMut(f32, u64, u64, &mut Graph, &[f32])->bool + 'static{ // err, step, evaluations, graph, params
		self.step_callback.push(Box::new(func));

	}
	// fn add_evaluation_callback<F>(&mut self, func: F) where F: FnMut(f32, u64, u64, &mut Graph, &[f32])->bool{
	// 	self.eval_callback.push(Box::new(func));
	// }
	
	fn optimise_from(&mut self, training_set: &mut Supplier,  mut params: Vec<f32>) -> Vec<f32>{ 
		
		
		while self.max_evals.map_or(true, |max| self.eval_count < max){
			let (_err, new_params) = self.step(training_set, params);
			params = new_params;
		}
		
		
		params
	}
	
	fn step(&mut self, training_set: &mut Supplier, params: Vec<f32>) -> (f32, Vec<f32>){
			
			
			let (input, training_input) = training_set.next_n(self.batch_size as usize);
			let (mut err, mut param_derivs, _data) = self.graph.backprop(self.batch_size as usize, input, training_input, &params);
			
			err /= self.batch_size as f32;

			param_derivs.scale_mut(1.0/self.batch_size as f32);
			
			println!("err:{}", err);
			self.eval_count += self.batch_size as u64;
			(err, params.add_scaled_move(&param_derivs, -self.rate))
			
			
			
			
			
			
			// let mut param_derivs = vec![0.0; params.len()];
			// let mut err_sum = 0.0;
			// for _ in 0..self.batch_size {
				
			// 	let (input, training_input) = training_set.next();
			// 	let _data = self.graph.backprop_mut(1, input, training_input, &params, &mut err_sum, &mut param_derivs);

			// 	self.eval_count += 1;
			// }
			// err_sum /= self.batch_size as f32;
			// //println!("err:{}", err_sum);
			// println!("{}\t{}", training_set.samples_taken(), err_sum);
			// param_derivs.scale_mut(1.0/(self.batch_size as f32));
			
			// params.add_scaled_move(&param_derivs, -self.rate)
			
	}
		
}






