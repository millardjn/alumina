pub mod sgd;
pub mod adam;

use graph::{Subgraph, NodeID, DataID, Result};
use data::DataStream;
use ndarray::ArrayD;

pub enum CallbackSignal{
	Stop,
	Continue,
}

pub struct CallbackData<'a>{
	pub err: f32,
	pub params: &'a [ArrayD<f32>],
	pub stream: &'a DataStream,
}

pub trait Opt {

	/// Borrows subgraph
	fn subgraph(&self) -> &Subgraph;

	/// This is the list of inputs to the subgraph which will be fed from the supplied `DataStream`.
	fn inputs(&self) -> &[DataID];

	/// This is the list of value `NodeID`s which correspond to nodes marked `Parameter`,
	/// excluding `NodeID`s which overlap with DataIDs included in `inputs()`
	fn parameters(&self) -> &[NodeID];

	/// Returns the error and the new parameters
	fn step(&mut self, inputs: Vec<ArrayD<f32>>, parameters: Vec<ArrayD<f32>>) -> Result<(f32, Vec<ArrayD<f32>>)>;

	fn callbacks(&mut self) -> &mut [Box<FnMut(&CallbackData)->CallbackSignal>];

	fn add_boxed_callback(&mut self, func: Box<FnMut(&CallbackData)->CallbackSignal>);

	fn add_callback<F: 'static + FnMut(&CallbackData)->CallbackSignal>(&mut self, func: F){
		self.add_boxed_callback(Box::new(func));
	}

	fn optimise<S: DataStream>(&mut self, training_stream: &mut S) -> Result<Vec<ArrayD<f32>>>{
		let params = self.subgraph().graph().initialise_nodes(self.parameters())?;
		self.optimise_from(training_stream, params)
	}

	fn optimise_from<S: DataStream>(&mut self, training_stream: &mut S, mut params: Vec<ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>>{
		let mut stop = false;
		while !stop {
			let (err, new_params) = self.step(training_stream.next(), params)?;
			params = new_params;

			let data = CallbackData{err: err, params: &params, stream: training_stream};
			for func in self.callbacks().iter_mut(){
				stop = stop | matches!(func(&data), CallbackSignal::Stop);
			}
		}
		Ok(params)
	}
}


pub fn print_step_data() -> Box<FnMut(&CallbackData)->CallbackSignal>{
	let mut step = 0;
	Box::new(move |data|{
		println!("step:{}\terr:{}", step, data.err);
		step += 1;
		CallbackSignal::Continue
	})
}

pub fn max_steps(max: usize) -> Box<FnMut(&CallbackData)->CallbackSignal>{
	let mut step = 0;
	Box::new(move |_data|{
		if step < max {
			step += 1;
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
	let mut step = 0;
	Box::new(move |data|{
		if step % n == 0 {
			step += 1;
			func(data)
		} else {
			step += 1;
			CallbackSignal::Continue
		}
	})
}