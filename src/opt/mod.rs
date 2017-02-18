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
	pub step_count: usize,
	pub eval_count: usize,
	pub params: &'a [f32],
}

pub fn print_step_data() -> Box<FnMut(&CallbackData)->CallbackSignal>{
	Box::new(move |data|{
		println!("err:{}\tstep_count:{}\teval_count{}", data.err, data.step_count, data.eval_count);
		CallbackSignal::Continue
	})
}

pub fn max_evals(max: usize) -> Box<FnMut(&CallbackData)->CallbackSignal>{
	Box::new(move |data|{
		if data.eval_count < max {
			CallbackSignal::Continue
		} else {
			CallbackSignal::Stop
		}
	})
}

pub fn max_steps(max: usize) -> Box<FnMut(&CallbackData)->CallbackSignal>{
	Box::new(move |data|{
		if data.step_count < max {
			CallbackSignal::Continue
		} else {
			CallbackSignal::Stop
		}
	})
}

pub fn min_err(min: f32) -> Box<FnMut(&CallbackData)->CallbackSignal>{
	Box::new(move |data|{
		if data.err > min {
			CallbackSignal::Continue
		} else {
			CallbackSignal::Stop
		}
	})
}

pub fn every_n_steps(n: usize, mut func: Box<FnMut(&CallbackData)->CallbackSignal>) -> Box<FnMut(&CallbackData)->CallbackSignal>{

	let mut prev_steps = 0;
	Box::new(move |data|{
		if data.step_count/n > prev_steps/n {
			prev_steps = data.step_count;
			func(data)
		} else {
			CallbackSignal::Continue
		}
	})
}

pub fn every_n_evals(n: usize, mut func: Box<FnMut(&CallbackData)->CallbackSignal>) -> Box<FnMut(&CallbackData)->CallbackSignal>{

	let mut prev_evals = 0;
	Box::new(move |data|{
		if data.eval_count/n > prev_evals/n {
			prev_evals = data.eval_count;
			func(data)
		} else {
			CallbackSignal::Continue
		}
		
	})
}


pub trait Optimiser<'a> {

	fn get_graph(&mut self) -> & mut Graph;

	fn add_boxed_step_callback(&mut self, func: Box<FnMut(&CallbackData)->CallbackSignal>);

	fn add_step_callback<F: 'static + FnMut(&CallbackData)->CallbackSignal>(&mut self, func: F){
		self.add_boxed_step_callback(Box::new(func));
	}

	fn get_step_callbacks(&mut self) -> &mut [Box<FnMut(&CallbackData)->CallbackSignal>];

	fn get_step_count(&self) -> usize;

	fn get_eval_count(&self) -> usize;

	fn step(&mut self, training_set: &mut Supplier, params: Vec<f32>) -> (f32, Vec<f32>);

	fn optimise(&mut self, training_set: &mut Supplier) -> Vec<f32>{
		let params = self.get_graph().init_params();
		self.optimise_from(training_set, params)
	}

	fn optimise_from(&mut self, training_set: &mut Supplier,  mut params: Vec<f32>) -> Vec<f32>{
		'outer: loop {
			let (err, new_params) = self.step(training_set, params);
			params = new_params;

			let data = CallbackData{err: err, step_count: self.get_step_count(), eval_count: self.get_eval_count(), params: &params};
			for func in self.get_step_callbacks().iter_mut(){
				
				match func(&data){
					CallbackSignal::Stop => {break 'outer},
					CallbackSignal::Continue =>{},
				}
			}
		}
		params
	}
	
}

