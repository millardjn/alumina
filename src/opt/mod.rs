pub mod sgd;
pub mod adam;

use graph::{GraphDef, Subgraph, Result};
use id::{NodeID, DataID};
use data::DataStream;
use ndarray::ArrayD;

pub enum CallbackSignal{
	Stop,
	Continue,
}

pub struct CallbackData<'a>{
	pub err: f32,
	pub step: usize,
	pub change_norm: f32,
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

	/// Returns the error, step number, l2 norm of param change, and the new parameters
	fn step(&mut self, inputs: Vec<ArrayD<f32>>, parameters: Vec<ArrayD<f32>>) -> Result<(f32, usize, f32, Vec<ArrayD<f32>>)>;

	fn callbacks(&mut self) -> &mut [Box<FnMut(&CallbackData)->CallbackSignal>];

	fn add_boxed_callback(&mut self, func: Box<FnMut(&CallbackData)->CallbackSignal>);

	fn optimise(&mut self, training_stream: &mut DataStream, graph: &GraphDef) -> Result<Vec<ArrayD<f32>>>{
		let params = graph.initialise_nodes(self.parameters())?;
		self.optimise_from(training_stream, params)
	}

	fn optimise_from(&mut self, training_stream: &mut DataStream, mut params: Vec<ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>>{
		let mut stop = false;
		while !stop {
			let (err, step, change_norm, new_params) = self.step(training_stream.next(), params)?;
			params = new_params;

			let data = CallbackData{err: err, step: step, change_norm: change_norm, params: &params, stream: training_stream};
			for func in self.callbacks().iter_mut(){
				stop = stop | matches!(func(&data), CallbackSignal::Stop);
			}
		}
		Ok(params)
	}
}

pub trait UnboxedCallbacks: Opt {
	fn add_callback<F: 'static + FnMut(&CallbackData)->CallbackSignal>(&mut self, func: F){
		self.add_boxed_callback(Box::new(func));
	}
}

impl<O: Opt> UnboxedCallbacks for O {}

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