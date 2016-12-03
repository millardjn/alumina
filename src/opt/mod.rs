pub mod sgd;
pub mod asgd;
pub mod supplier;

use graph::*;

use self::supplier::Supplier;


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

