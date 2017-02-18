use opt::*;
use graph::*;
use vec_math::{VecMathMut, VecMathMove};
use std::f32;
use supplier::Supplier;



pub struct Sgd<'a>{
	eval_count: usize,
	step_count: usize,
	graph: &'a mut Graph,
	rate: f32,
	batch_size: u32,
	_momentum: Option<f32>,
	_momentum_vec: Vec<f32>,
	step_callback: Vec<Box<FnMut(&CallbackData)->CallbackSignal>>,
}

impl <'a> Sgd<'a> {
	pub fn new(graph: &'a mut Graph, rate: f32, batch_size: u32, momentum: Option<f32>) -> Sgd<'a>{
		let num_params = graph.num_params();
		Sgd{
			eval_count: 0,
			step_count: 0,
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

	fn get_graph(&mut self) -> &mut Graph{
		&mut self.graph
	}
	
	fn add_boxed_step_callback(&mut self, func: Box<FnMut(&CallbackData)->CallbackSignal>){ // err, step, evaluations, graph, params
		self.step_callback.push(func);
	}
	
	fn get_step_callbacks(&mut self) -> &mut [Box<FnMut(&CallbackData)->CallbackSignal>]{
		&mut self.step_callback[..]
	}
	
	fn get_step_count(&self) -> usize{
		self.step_count
	}

	fn get_eval_count(&self) -> usize{
		self.eval_count
	}

	fn step(&mut self, training_set: &mut Supplier, params: Vec<f32>) -> (f32, Vec<f32>){
			
			
			let (input, training_input) = training_set.next_n(self.batch_size as usize);
			let (mut err, mut param_derivs, _data) = self.graph.backprop(self.batch_size as usize, input, training_input, &params);
			
			err /= self.batch_size as f32;

			param_derivs.scale_mut(1.0/self.batch_size as f32);
			self.step_count +=1;

			self.eval_count += self.batch_size as usize;
			(err, params.add_scaled_move(&param_derivs, -self.rate))
			
	}
		
}






