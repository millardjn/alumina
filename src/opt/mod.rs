pub mod sgd;
pub mod asgd;


use graph::*;
use rand::*;
use std::iter;



pub trait Optimiser<'a> {
	
	fn set_max_evals(&mut self, max_evals: u64);
	fn set_min_loss(&mut self, min_loss: f32);
	fn set_min_epoch_loss_delta(&mut self, delta: f32);
	fn get_max_evals(&self) -> Option<u64>;
	fn get_min_loss(&self) -> Option<f32>;
	fn get_min_epoch_loss_delta(&self) -> Option<f32>;
	fn get_graph(& mut self) -> & mut Graph;
	/// err, step, evaluations, graph, params -> returns whether the optimisation loop should continue
	fn add_step_callback<F>(&mut self, func: F) where F: FnMut(f32, u64, u64, &mut Graph, &[f32])->bool+'static;
	//fn add_evaluation_callback<F>(&mut self, func: F) where F: FnMut(f32, u64, u64, &mut Graph, &[f32])->bool; 
	fn optimise(&mut self, training_set: &mut Supplier) -> Vec<f32>{
		let params = self.get_graph().init_params();
		self.optimise_from(training_set, params)
	}
	fn optimise_from(&mut self, training_set: &mut Supplier, params: Vec<f32>) -> Vec<f32>;
	fn step(&mut self, training_set: &mut Supplier, params: Vec<f32>) -> (f32, Vec<f32>);
}

pub trait Supplier {
	fn next_n(&mut self, u: usize) -> (Vec<NodeData>, Vec<NodeData>);
	fn epoch_size(&self) -> usize;
	fn samples_taken(&self) -> u64;
	fn reset(&mut self);
	fn once(self)-> Vec<(Vec<NodeData>, Vec<NodeData>)> ;
}

pub struct ShuffleRandomiser {
	n: usize,
	order: Box<Iterator<Item=usize>>,
}

impl ShuffleRandomiser {
	pub fn new(n: usize) -> ShuffleRandomiser{
		ShuffleRandomiser{
			n: n,
			order: Box::new(iter::empty()),
			}
	}

	pub fn next(&mut self) -> usize{
		match self.order.next() {
			Some(i) => i,
			None => {
				assert!(self.n > 0, "Cant generate indices over a zero size set.");
				let mut v: Vec<usize> = (0..self.n).collect();
				let mut rng = thread_rng();
				rng.shuffle(&mut v);
				
				self.order = Box::new(v.into_iter());
				
				self.next()
			},
		}

	}

	pub fn reset(&mut self){
		self.order = Box::new(iter::empty());
	}
}

pub struct Randomiser {
	n: usize,
}

impl Randomiser {
	pub fn new(n: usize) -> Randomiser{
		Randomiser{
			n: n,
			}
	}

	pub fn next(&self) -> usize{
		thread_rng().gen_range(0, self.n)
	}

	pub fn reset(&mut self){}
}