pub mod sgd;
pub mod cain;

use graph::*;

use supplier::Supplier;


pub enum CallbackSignal{
	Stop,
	Continue,
}

pub struct CallbackData<'a>{
	pub err: f32,
	pub step_count: u64,
	pub eval_count: u64,
	pub graph: &'a Graph,
	pub params: &'a [f32],
}

pub fn print_step_data() -> Box<FnMut(CallbackData)->CallbackSignal>{
	Box::new(move |data|{
		println!("err:{}\tstep_count:{}\teval_count{}", data.err, data.step_count, data.eval_count);
		CallbackSignal::Continue
	})
}

pub fn max_evals(max: u64) -> Box<FnMut(CallbackData)->CallbackSignal>{
	Box::new(move |data|{
		if data.eval_count < max {
			CallbackSignal::Continue
		} else {
			CallbackSignal::Stop
		}
	})
}

pub fn max_steps(max: u64) -> Box<FnMut(CallbackData)->CallbackSignal>{
	Box::new(move |data|{
		if data.step_count < max {
			CallbackSignal::Continue
		} else {
			CallbackSignal::Stop
		}
	})
}

pub fn min_err(min: f32) -> Box<FnMut(CallbackData)->CallbackSignal>{
	Box::new(move |data|{
		if data.err > min {
			CallbackSignal::Continue
		} else {
			CallbackSignal::Stop
		}
	})
}

pub trait Optimiser<'a> {

	fn get_graph(& mut self) -> & mut Graph;
	/// err, step, evaluations, graph, params -> returns whether the optimisation loop should continue
	fn add_boxed_step_callback(&mut self, func: Box<FnMut(CallbackData)->CallbackSignal>);
	fn add_step_callback<F: 'static + FnMut(CallbackData)->CallbackSignal>(&mut self, func: F){
		self.add_boxed_step_callback(Box::new(func));
	}
	//fn add_evaluation_callback<F>(&mut self, func: F) where F: FnMut(f32, u64, u64, &mut Graph, &[f32])->bool; 
	fn optimise(&mut self, training_set: &mut Supplier) -> Vec<f32>{
		let params = self.get_graph().init_params();
		self.optimise_from(training_set, params)
	}
	fn optimise_from(&mut self, training_set: &mut Supplier, params: Vec<f32>) -> Vec<f32>;
	fn step(&mut self, training_set: &mut Supplier, params: Vec<f32>) -> (f32, Vec<f32>);
}

