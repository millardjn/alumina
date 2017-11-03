use opt::*;
use graph::*;
use vec_math::{VecMath, VecMathMut, VecMathMove};

use supplier::Supplier;
use std::f32;
use std::usize;


pub struct AdamBuilder<'a>{
	graph: &'a mut Graph,
	learning_rate: f32,
	batch_size: usize,
	beta1: f32,
	beta2: f32,
	epsilon: f32,
}

impl<'a> AdamBuilder<'a> {

	pub fn learning_rate(mut self, val: f32) -> Self{
		self.learning_rate = val;
		self
	}

	pub fn batch_size(mut self, val: usize) -> Self{
		self.batch_size = val;
		self
	}

	pub fn beta1(mut self, val: f32) -> Self{
		self.beta1 = val;
		self
	}

	pub fn beta2(mut self, val: f32) -> Self{
		self.beta2 = val;
		self
	}

	pub fn epsilon(mut self, val: f32) -> Self{
		self.epsilon = val;
		self
	}


	pub fn finish(self) -> Adam<'a>{
		let num_params = self.graph.num_params();
		Adam{
			graph: self.graph,
			learning_rate: self.learning_rate,
			batch_size: self.batch_size,
			beta1: self.beta1,
			beta2: self.beta2,
			epsilon: self.epsilon,

			eval_count: 0,
			step_count: 0,
			
			m: vec![0.0; num_params],
			v: vec![0.0; num_params],

			step_callback: vec![],
		}
	}
}


pub struct Adam <'a>{
	graph: &'a mut Graph,
	learning_rate: f32,
	batch_size: usize,
	beta1: f32,
	beta2: f32,
	epsilon: f32,

	eval_count: usize,
	step_count: usize,
	
	m: Vec<f32>,
	v: Vec<f32>,

	step_callback: Vec<Box<FnMut(&CallbackData)->CallbackSignal>>,
}

impl <'a> Adam <'a> {
	pub fn new <'b>(graph: &'b mut Graph) -> AdamBuilder<'b>{
		AdamBuilder{
			graph: graph,
			learning_rate: 1e-3,
			batch_size: 16,
			beta1: 0.9,
			beta2: 0.99,
			epsilon: 1e-08,
		}
	}
}

impl<'a> Optimiser<'a> for Adam<'a>{

	fn add_boxed_step_callback(&mut self, func: Box<FnMut(&CallbackData)->CallbackSignal>){
		self.step_callback.push(func);
	}

	fn get_graph(&mut self) -> &mut Graph{
		&mut self.graph
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
		let (mut err, mut param_derivs, _) = self.graph.backprop(self.batch_size as usize, input, training_input, &params);
		
		err /= self.batch_size as f32;
		param_derivs.scale_mut(1.0/self.batch_size as f32);
		
		self.eval_count += self.batch_size;


		self.m.scale_mut(self.beta1);
		self.m.add_scaled_mut(&param_derivs, 1.0 - self.beta1);

		self.v.scale_mut(self.beta2);
		self.v.add_scaled_mut(&param_derivs.elementwise_mul(&param_derivs), 1.0 - self.beta2);



		let m_correction = if self.step_count < 1_000_000{1.0/(1.0 - self.beta1.powi(self.step_count as i32 + 1))} else {1.0};
		let v_correction = if self.step_count < 1_000_000{1.0/(1.0 - self.beta2.powi(self.step_count as i32 + 1))} else {1.0};

		let change: Vec<f32> = self.m.iter().zip(&self.v).map(|(m,v)| -self.learning_rate * m * m_correction/((v*v_correction).sqrt() + self.epsilon)).collect();
		

		// print progress (TODO: this should be moved to a callback lambda)
		if self.step_count == 0 {println!("");println!("count\terr\tmovement");}
		println!("{}\t{:.4}\t{:.4e}", training_set.samples_taken(), err, change.norm2());

		let new_params = change.add_move(&params);

		self.step_count += 1;
		(err, new_params)
	}

		
}