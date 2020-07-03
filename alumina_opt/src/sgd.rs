use crate::{calc_change_sqr, GradientStepper};
use alumina_core::{errors::ExecError, graph::Node};
use indexmap::{indexmap, IndexMap};
use ndarray::{ArrayD, Zip};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Sgd {
	step_count: usize,

	rate: f32,
	momentum: Option<f32>,
	momentums: IndexMap<Node, ArrayD<f32>>,
}

impl Sgd {
	/// Create an optimisation problem assuming that all nodes marked `Parameter` should be optimised.
	pub fn new(rate: f32, momentum: Option<f32>) -> Self {
		Sgd {
			rate,
			momentum,
			momentums: indexmap![],
			step_count: 0,
		}
	}

	/// Learning rate, α, which defines the proportionality between parameter gradients and the size of the update for each step.
	///
	///
	/// θ = θ - α ∇f(θ)
	///
	/// Default: 1e-3
	pub fn rate(&mut self, rate: f32) -> &mut Self {
		self.rate = rate;
		self
	}

	/// Momentum coefficient, β
	///
	/// If not `None`, the following update is used:
	/// m = β m + ∇f(θ)
	/// θ = θ - α m
	///
	/// Default: None
	pub fn momentum<O: Into<Option<f32>>>(&mut self, momentum: O) -> &mut Self {
		self.momentum = momentum.into();
		self
	}
}

impl GradientStepper for Sgd {
	fn step_count(&self) -> usize {
		self.step_count
	}

	fn step(
		&mut self,
		mut parameters_and_grad_values: IndexMap<Node, ArrayD<f32>>,
		//parameters_and_grads: &IndexMap<Node, Node>,
		//mut results: IndexMap<Node, ArrayD<f32>>,
		calc_change: bool,
		// inputs: IndexMap<I, ArrayD<f32>>,
		// parameters_and_grads: &IndexMap<Node, Node>,
		// options: StepOptions,
	) -> Result<f32, ExecError>
//where
		//I: Borrow<Node> + Hash + Eq,
	{
		// let (mut results, loss) = if let Some(loss) = options.loss {
		// 	let results = exec(
		// 		inputs,
		// 		parameters_and_grads.values().chain(once(loss)),
		// 		ExecConfig::default().subgraph(options.subgraph),
		// 	)?;
		// 	let loss = results.get(loss).map(ArrayBase::sum).unwrap();
		// 	(results, loss)
		// } else {
		// 	let results = exec(
		// 		inputs,
		// 		parameters_and_grads.values(),
		// 		ExecConfig::default().subgraph(options.subgraph),
		// 	)?;
		// 	(results, 0.0)
		// };

		let rate = self.rate;
		//let calc_change = options.calc_change;

		let change_sqr: f32 = if let Some(momentum) = self.momentum {
			for param in parameters_and_grad_values.keys() {
				self.momentums.entry(param.clone()).or_insert_with(|| {
					ArrayD::zeros(
						param
							.shape()
							.to_data_shape()
							.expect("Parameters must have fixed shapes"),
					)
				});
			}

			// write new parameter array into grad array
			self.momentums
				.iter_mut()
				.filter_map(|(param, momentum_arr)| {
					parameters_and_grad_values
						.swap_remove(param)
						.map(|grad_arr| (param, momentum_arr, grad_arr))
				})
				.par_bridge()
				.map(|(param, momentum_arr, mut grad_arr)| {
					let param_arr = param.value().unwrap();

					Zip::from(&param_arr)
						.and(momentum_arr)
						.and(grad_arr.view_mut())
						.par_apply(|param, grad_momentum, grad| {
							*grad_momentum = (*grad_momentum) * momentum + *grad;
							let change = -rate * (*grad_momentum);
							*grad = *param + change;
						});

					let change_sqr = if calc_change {
						calc_change_sqr(param_arr.view(), grad_arr.view())
					} else {
						0.0
					};
					param.set_value(grad_arr);
					change_sqr
				})
				.sum()
		} else {
			// write new parameter array into grad array
			parameters_and_grad_values
				.into_iter()
				.par_bridge()
				.map(|(param, mut grad_arr)| {
					let param_arr = param.value().unwrap();

					Zip::from(&param_arr).and(grad_arr.view_mut()).par_apply(|param, grad| {
						let change = -rate * *grad;
						*grad = *param + change;
					});

					let change_sqr = if calc_change {
						calc_change_sqr(param_arr.view(), grad_arr.view())
					} else {
						0.0
					};
					param.set_value(grad_arr);
					change_sqr
				})
				.sum()
		};

		self.step_count += 1;

		//Ok(StepData {
		//	loss,
		//	step: self.step_count,
		Ok(change_sqr.sqrt())
		//)
	}
}
