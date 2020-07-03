use crate::{calc_change_sqr, GradientStepper};
use alumina_core::{errors::ExecError, graph::Node};
use indexmap::{indexmap, IndexMap};
use ndarray::{ArcArray, IxDyn, ArrayD, ArrayViewD, ArrayViewMutD, Zip};
use rand::{thread_rng, Rng, SeedableRng};
use rand_distr::{Exp1, StandardNormal};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use std::f32::EPSILON;
use std::sync::Arc;

// /// Scale for mixture of gaussians
// pub enum MogVarScale {
// 	/// Mixture of gaussians with a unit scale on variance results in plain gaussian sampling for each parameter
// 	/// element.
// 	Unit,
// 	/// Mixture of gaussians with a exponential variate scale on variance results in sampling from the Laplace
// 	/// distribution for each parameter element.
// 	///
// 	/// This has fatter tails than the gaussian distribution (excess kurtosis of 3).
// 	Exp1,
// 	/// Mixture of gaussians with a exponential variate scale on standard deviation results in a fatter tailed
// 	/// distribution than the Laplace distribution.
// 	///
// 	/// (Excess kurtosis of 15)
// 	Exp1sqr,
// }

// impl MogVarScale {
// 	fn sd_scale(&self) -> f32 {
// 		match self {
// 			MogVarScale::Unit => 1.0,
// 			MogVarScale::Exp1sqr => ::std::f32::consts::FRAC_1_SQRT_2 * thread_rng().sample::<f32, _>(Exp1),
// 			MogVarScale::Exp1 => thread_rng().sample::<f32, _>(Exp1).sqrt(),
// 		}
// 	}

// 	#[allow(unused)]
// 	fn var_scale(&self) -> f32 {
// 		let x = self.sd_scale();
// 		x * x
// 	}
// }

// /// Determines how samples for the early steps are produced (before the regression has seen enough data to be useful).
// /// This strategy is only used while step_count < 1.0/(1.0-lambda), i.e. if lambda is 0.99 then the first 100 steps;
// #[derive(Clone, Debug)]
// pub enum EarlySteps {
// 	/// A simple SGD is used to update the parameters, without momentum.
// 	/// The rate does not have to be tuned well, anywhere from the optimal rate for standalone SGD to a factor of 100
// 	/// less is sufficient.
// 	Sgd { rate: f32 },

// 	/// Adds a gaussian variate to the previous value.
// 	RandomWalk { rel_std_dev: f32 },
// }

#[derive(Clone)]
pub struct Soop {
	init_lambda: f32,
	min_lambda: f32,
	max_lambda: f32,
	rls: IndexMap<Node, ArrayD<Rls>>,
	step_count: usize,

	/// Oh boy
	update_fn: Arc<dyn Fn(Pcg64Mcg, ArrayViewD<Rls>, ArrayViewD<f32>, ArrayViewMutD<f32>, f32) + Sync + Send>,

	param_weight: f32,
	param_var_weight: f32,
	tensor_var_weight: f32,
	static_var: f32,
}

impl Soop {
	/// Create an optimisation problem assuming that all nodes marked `Parameter` should be optimised.
	pub fn new() -> Self {
		let mut opt = Soop {
			min_lambda: 0.9,
			init_lambda: 0.9,
			max_lambda: 0.99999,
			rls: indexmap![],
			step_count: 0,

			update_fn: Arc::new(|_, _, _, _, _| {}),

			param_weight: 0.98,
			param_var_weight: 0.01,
			tensor_var_weight: 0.01,
			static_var: EPSILON * EPSILON,
		};
		opt.variance_addition(|p, g, lp| {
			EPSILON*EPSILON // permanent noise at small fixed scale
			// + (1.0-lp)*1e-6*p*p // permanent noise at 0.1% of parameter scale
			// + (1.0-lp)*1e-6*g*g // permanent noise at 0.1% of gradient scale
		});
		opt
	}

	// pub fn auto(batch_size: usize, epoch_size: usize) -> Self {
	// 	Soop {
	// 		min_std_dev: 10.0 * ::std::f32::EPSILON,
	// 		min_lambda: 0.9,
	// 		init_lambda: 0.9,
	// 		max_lambda: 0.99999,
	// 		rls: indexmap![],
	// 		step_count: 0,
	// 		early_steps: EarlySteps::RandomWalk { rel_std_dev: 0.1 },
	// 		overshoot: 1.25,
	// 	}
	// }

	/// Set weights used to determine the initial sampling variance for each parameter
	///
	/// Typically these should sum to 1.0 to avoid changing the variance of linear functions.
	///
	/// param_weight
	pub fn init_weights(&mut self, param_weight: f32, param_var_weight: f32, tensor_var_weight: f32) -> &mut Self {
		self.param_weight = param_weight;
		self.param_var_weight = param_var_weight;
		self.tensor_var_weight = tensor_var_weight;
		self
	}

	/// Takes a function which determines the additional sampling variance
	///
	/// Function arguments in order are:
	///  * parameter
	///  * gradient
	///  * lambda_product(product of lamda at each step, asymptotically decreasing from 1 toward 0)
	///
	/// The default function applies a temporary amount of additional variance base on
	/// the gradient magnitued to unstick parameters with otherwise small variance:
	///
	/// ```rust
	/// |p, g, lp| {
	/// 	EPSILON*EPSILON // permanent noise at small fixed scale
	/// 	+ (1.0-lp)*1e-6*p*p // permanent noise at 0.1% of parameter scale
	/// 	+ (1.0-lp)*1e-6*g*g // temporary noise at 0.1% of gradient scale
	/// }
	/// ```
	///
	pub fn variance_addition<F>(&mut self, f: F) -> &mut Self
	where
		F: 'static + Sync + Send + Fn(f32, f32, f32) -> f32,
	{
		self.update_fn = Arc::new(move |mut rng, rls_arr, param_arr, grad_arr, lambda_prod| {
			Zip::from(rls_arr)
				.and(param_arr)
				.and(grad_arr)
				.apply(|rls, param, grad| {
					let added_var = f(*param**param, *grad**grad, lambda_prod);
					*grad = standard_update(rls, &mut rng, added_var, lambda_prod); //, *grad, scale
				});
		});
		self
	}

	/// Minimum rls lambda value
	///
	/// The weight of each previously observed datapoint decays by the lambda factor on each new observation.
	/// As lambda adapts, it is capped to at least this minimum value.
	/// Cannot be set to a value greater than max_lambda.
	///
	/// Default: 0.9
	pub fn min_lambda(&mut self, min_lambda: f32) -> &mut Self {
		self.min_lambda = min_lambda.min(self.max_lambda);
		self.init_lambda = self.init_lambda.max(self.min_lambda).min(self.max_lambda);
		self
	}

	/// Maximum rls lambda value
	///
	/// The weight of each previously observed datapoint decays by the lambda factor on each new observation
	/// As lambda adapts, it is capped to at most this maximum value.
	/// Cannot be set to a value less than min_lambda.
	///
	/// Default: 0.99999
	pub fn max_lambda(&mut self, max_lambda: f32) -> &mut Self {
		self.max_lambda = max_lambda.max(self.min_lambda);
		self.init_lambda = self.init_lambda.max(self.min_lambda).min(self.max_lambda);
		self
	}

	/// Initial rls lambda value
	///
	/// The weight of each previously observed datapoint decays by the lambda factor on each new observation.
	/// This value is used during the early steps of the optimisation, until a sufficient history of datapoints has built up.
	/// If less than min_lambda or greater than max_lambda, init_lambda will be clipped.
	///
	/// Default: 0.9
	pub fn init_lambda(&mut self, init_lambda: f32) -> &mut Self {
		self.init_lambda = init_lambda.max(self.min_lambda).min(self.max_lambda);
		self
	}
}

impl GradientStepper for Soop {
	fn step_count(&self) -> usize {
		self.step_count
	}

	fn step(
		&mut self,
		mut parameters_and_grad_values: IndexMap<Node, ArrayD<f32>>,
		calc_change: bool,
	) -> Result<f32, ExecError> {
		let min_lambda = self.min_lambda;
		let max_lambda = self.max_lambda;
		let init_lambda = self.init_lambda;
		let step = self.step_count as f32;

		let param_mul = self.param_weight.sqrt();
		let param_var_weight = self.param_var_weight;
		let tensor_var_weight = self.tensor_var_weight;
		let static_var = self.static_var;

		let update_fn = &*self.update_fn;

		for param in parameters_and_grad_values.keys() {
			self.rls.entry(param.clone()).or_insert_with(|| {
				let arr = param.value().expect("param must be initialised");
				let mean = arr.mean().unwrap_or(0.0);

				let var = arr.iter().fold(0.0, |acc, e| acc + (e - mean) * (e - mean)) / arr.len() as f32;
				arr.map(|e| {
					let mut rls = Rls::new(init_lambda);
					rls.x_bar = *e * param_mul;
					rls.x_var = static_var + *e * *e * param_var_weight + var * tensor_var_weight;
					rls
				})
			});
		}

		let scale = thread_rng().sample::<f32, _>(Exp1);
		let change_sqr: f32 = self
			.rls
			.iter_mut()
			.filter_map(|(param, rls_arr)| {
				parameters_and_grad_values
					.swap_remove(param)
					.map(|grad_arr| (param, rls_arr, grad_arr))
			})
			.collect::<Vec<_>>()
			.into_par_iter()
			.with_max_len(1)
			.map(|(param, rls_arr, mut grad_arr)| {
				let param_arr = param.value().unwrap();

				//let lambda_prod;
				// if step < -(0.25f32.ln()) / (1.0 - init_lambda) {
				let lambda_prod = init_lambda.powi(step as i32 + 1);
				let correction = 1.0 - lambda_prod;
				// 	Zip::from(&param_arr)
				// 		.and(rls_arr.view_mut())
				// 		.and(grad_arr.view_mut())
				// 		.par_apply(|&param, rls, grad| rls.add(param, *grad, RlsUpdate::FixLambda { correction }));

						// Zip::from(rls_arr.outer_iter())
						// .and(param_arr.outer_iter())
						// .and(grad_arr.outer_iter_mut())
						// .par_apply(|rls_arr, param_arr, grad_arr| {
						// 	let mut rng = Pcg64Mcg::from_rng(rand::thread_rng()).unwrap();
						// 	Zip::from(rls_arr)
						// 	.and(param_arr)
						// 	.and(grad_arr)
						// 	.apply(|rls, param, grad| {
						// 		*grad = rls.x_bar + rls.x_var * rng.sample::<f32, _>(StandardNormal);
						// 	});
						// });
				//} else {
					//lambda_prod = 0.0;
					Zip::from(&param_arr)
						.and(rls_arr.view_mut())
						.and(grad_arr.view_mut())
						.par_apply(|&param, rls, grad| {
							rls.add(param, *grad, min_lambda, max_lambda, correction);
						});
				
				//}

				// Rather than par_apply over whole array, only parallelise over outer dim so that
				Zip::from(rls_arr.outer_iter())
				.and(param_arr.outer_iter())
				.and(grad_arr.outer_iter_mut())
				.par_apply(|rls_arr, param_arr, grad_arr| {
					let rng = Pcg64Mcg::from_rng(rand::thread_rng()).unwrap();
					update_fn(rng, rls_arr, param_arr, grad_arr, lambda_prod);
				});

				let change_sqr = if calc_change {
					calc_change_sqr(param_arr.view(), grad_arr.view())
				} else {
					0.0
				};
				param.set_value(grad_arr);
				change_sqr
			})
			.sum();

		// let change_sqr: f32 = self
		// 	.rls
		// 	.iter_mut()
		// 	.filter_map(|(param, rls_arr)| parameters_and_grad_values.swap_remove(param).map(|grad_arr|(param, rls_arr, grad_arr)))
		// 	.par_bridge()
		// 	.map(|(param, rls_arr, mut grad_arr)| {
		// 		let param_arr = param.value().unwrap();

		// 		if step < -(0.5f32.ln()) / (1.0 - init_lambda) {

		// 			match early_steps {
		// 				EarlySteps::Sgd { rate } => {
		// 					let correction = 1.0 - init_lambda.powi(step as i32 + 1);
		// 					Zip::from(&param_arr)
		// 						.and(rls_arr.view_mut())
		// 						.and(grad_arr.view_mut())
		// 						.par_apply(|&param, rls, grad| {
		// 							//let temp = if step == 0 {param} else {rls.x_bar};
		// 							rls.add(param, *grad, RlsUpdate::FixLambda{correction});
		// 							//rls.x_bar = temp;
		// 						});

		// 					Zip::from(param_arr.outer_iter())
		// 						.and(rls_arr.outer_iter())
		// 						.and(grad_arr.outer_iter_mut())
		// 						.par_apply(|param_arr, rls_arr, grad_arr| {
		// 							let mut rng = Pcg64Mcg::from_rng(rand::thread_rng()).unwrap();
		// 							Zip::from(param_arr)
		// 								.and(rls_arr)
		// 								.and(grad_arr)
		// 								.apply(|param, _rls, grad| {
		// 									// let (mean, var) = Self::sampling_distribution(rls);
		// 									// let sd = scale * var.sqrt().max(mean.abs().max(1.0) * min_std_dev);

		// 									// *grad = mean + sd * rng.sample(StandardNormal) as f32 - rate * *grad;

		// 									let sd = param.abs().max(1.0) * min_std_dev;
		// 									*grad = param - rate * *grad + sd * rng.sample::<f32, _>(StandardNormal);
		// 								});
		// 						});
		// 				},
		// 				EarlySteps::RandomWalk { .. } => {
		// 					//let correction = 1.0 - init_lambda.powi(step as i32 + 1);
		// 					let correction = 1.0;

		// 					//let rel_std_dev = rel_std_dev * (1.0 - init_lambda).sqrt();
		// 					Zip::from(&param_arr)
		// 						.and(rls_arr.view_mut())
		// 						.and(grad_arr.view_mut())
		// 						.par_apply(|&param, rls, grad| {
		// 							//let old_x_bar = if step == 0 {param} else {rls.x_bar};
		// 							rls.add(param, *grad, RlsUpdate::FixLambda{correction});
		// 							//rls.x_bar = old_x_bar; //rls.x_bar * correction + (1.0-correction)*
		// 						});

		// 					Zip::from(rls_arr.outer_iter_mut())
		// 						.and(grad_arr.outer_iter_mut())
		// 						.par_apply(|rls_arr, grad_arr| {
		// 							let mut rng = Pcg64Mcg::from_rng(rand::thread_rng()).unwrap();
		// 							Zip::from(rls_arr)
		// 								.and(grad_arr)
		// 								.apply(|rls, grad| {
		// 									*grad = standard_update(rls, &mut rng, min_std_dev)
		// 								});
		// 						});
		// 				},
		// 			}
		// 		} else {
		// 			Zip::from(&param_arr)
		// 				.and(rls_arr.view_mut())
		// 				.and(grad_arr.view_mut())
		// 				.par_apply(|&param, rls, grad| {
		// 					rls.add(param, *grad, RlsUpdate::UpdateLambda{min_lambda, max_lambda});
		// 				});

		// 			// Rather than par_apply over whole array, only parallelise over outer dim so that
		// 			Zip::from(rls_arr.outer_iter_mut())
		// 				.and(grad_arr.outer_iter_mut())
		// 				.par_apply(|rls_arr, grad_arr| {
		// 					let mut rng = Pcg64Mcg::from_rng(rand::thread_rng()).unwrap();
		// 					Zip::from(rls_arr)
		// 						.and(grad_arr)
		// 						.apply(|rls, grad| {
		// 							*grad = standard_update(rls, &mut rng, min_std_dev)
		// 						});
		// 				});

		// 			// // Rather than par_apply over whole array, only parallelise over outer dim so that
		// 			// Zip::from(rls_arr.outer_iter())
		// 			// 	.and(grad_arr.outer_iter_mut())
		// 			// 	.par_apply(|rls_arr, grad_arr| {
		// 			// 		let mut rng = Pcg64Mcg::from_rng(rand::thread_rng()).unwrap();
		// 			// 		Zip::from(rls_arr)
		// 			// 			.and(grad_arr)
		// 			// 			.apply(|rls, grad| {
		// 			// 				let (mean, var) = Self::sampling_distribution(rls);
		// 			// 				// let sd = scale * var.sqrt().max(mean.abs().max(1.0) * min_std_dev);
		// 			// 				// *grad = rls.x_bar + overshoot * mean+ sd * rng.sample(StandardNormal) as f32;

		// 			// 				// // Dist shape changes, bernoulli component is always scaled to x_var
		// 			// 				// let split = 0.5;
		// 			// 				// let sd1 = (split * var).sqrt().max(mean.abs().max(1.0) * min_std_dev);
		// 			// 				// let sd2 = ((1.0 - split) * rls.x_var).sqrt();
		// 			// 				// *grad = rls.x_bar + overshoot * mean - rls.y_bar.signum() * (scale * sd1 * rng.sample(StandardNormal) as f32 + sd2 * rng.sample(StandardNormal).signum() as f32).abs();

		// 			// 				// // dist always has the shape of double gaussian at the sparrow criterion
		// 			// 				// let x_var_split = 0.5;
		// 			// 				// let var = (1.0-x_var_split) * var + x_var_split* rls.x_var;
		// 			// 				// let bernoulli_var_split = 0.5; // this controls the distribution shape. 0.5 is exactly the sparrow criterion
		// 			// 				// let sample = scale * ((1.0 - bernoulli_var_split) * var).sqrt() * rng.sample(StandardNormal) as f32 + (bernoulli_var_split*var).sqrt() * rng.sample(StandardNormal).signum() as f32;
		// 			// 				// *grad = rls.x_bar + overshoot * mean - rls.y_bar.signum() * sample.abs();

		// 			// 				// Just Normal distribution
		// 			// 				let x_var_split = 0.5;
		// 			// 				let sd = ((1.0-x_var_split) * var + x_var_split* rls.x_var).sqrt().max(mean.abs().max(1.0) * min_std_dev);
		// 			// 				*grad = rls.x_bar + overshoot * mean - rls.y_bar.signum() * (scale * sd * rng.sample::<f32, _>(StandardNormal)).abs();
		// 			// 			});
		// 			// 	});
		// 		}

		// 		let change_sqr = if calc_change {
		// 			calc_change_sqr(param_arr.view(), grad_arr.view())
		// 		} else {
		// 			0.0
		// 		};
		// 		param.set_value(grad_arr);
		// 		change_sqr
		// 	})
		// 	.sum();

		self.step_count += 1;

		Ok(change_sqr.sqrt())
	}

	fn best_estimate(&self, params: &mut dyn Iterator<Item = &Node>) -> IndexMap<Node, ArcArray<f32, IxDyn>> {
		params
		.map(|param| {
			(param.clone(),
			self.rls.get_full(param).map(|(_, param, rls_arr)| rls_arr.map(|rls| rls.x_bar).to_shared())
			.unwrap_or_else(|| {
				let mut shape = param.shape().clone();
				shape.collapse_dimensions_to_minimum();
				ArcArray::zeros(shape.to_data_shape().unwrap())
			}))		
		}).collect()
	}

	// fn finalise(&self, parameters_and_grads: &IndexMap<Node, Node>) {
	// 	self.rls
	// 		.iter()
	// 		.filter(|(param, _rls)| parameters_and_grads.contains_key(*param))
	// 		.par_bridge()
	// 		.for_each(|(param, rls_arr)| {
	// 			let mut param_arr = param
	// 				.take_value()
	// 				.map(ArrayBase::into_owned)
	// 				.unwrap_or_else(|| ArrayD::zeros(rls_arr.shape()));

	// 			Zip::from(&mut param_arr).and(rls_arr).par_apply(|param, rls| {
	// 				*param = rls.x_bar;
	// 			});
	// 			param.set_value(param_arr);
	// 		});
	// }
}

fn standard_update<R: Rng>(
	rls: &Rls,
	rng: &mut R,
	additional_var: f32,
	lambda_prod: f32,
	// grad: f32,
	// scale: f32,
) -> f32 {
	let effective_n = rls.effective_n();
	let (mean, var) = sampling_distribution(rls);

	let total_y_var = rls.y_error_variance + rls.x_var * (rls.m * rls.m);

	let split1 = total_y_var / (total_y_var + 2.0*effective_n * rls.y_bar * rls.y_bar);
	//let split1 = 1.0 - (1.0 - split1)*(1.0-lambda_prod); 
	//let split2 = total_y_var/(total_y_var + effective_n*rls.y_bar*rls.y_bar*0.5);

	let mean_split = mean * mean / (mean * mean + 2.0*rls.x_var);
	let x_var_split = rls.y_error_variance / (rls.y_error_variance + 2.0*rls.x_var * (rls.m * rls.m));

	let x = (1.0 - mean_split) * x_var_split;
	//let x = 1.0 - (1.0 - x)*(1.0-lambda_prod); // force x to be 1 early
	let z = 1.0 - (1.0 - x)*split1;
	

	let s = rng.sample::<f32, _>(StandardNormal);
	//let s = rng.sample::<f32, _>(Exp1); //* ::std::f32::consts::FRAC_1_SQRT_2;
	//let s = 1.0;// 0.8404 on cifar10!! very good
	let step_squared = z * rls.x_var*s*s  + 2.0*(mean * mean) * (1.0 - x);

	rls.x_bar - rls.y_bar.signum() * step_squared.sqrt()
		+ (2.0*var * (1.0 - z) + additional_var).sqrt() // + (mean * mean) * (1.0 - x)
			* rng.sample::<f32, _>(StandardNormal)
}


// #[allow(unused)]
// fn standard_update<R: Rng>(rls: &Rls, rng: &mut R, additional_var: f32, lambda_prod: f32, grad: f32, scale: f32) -> f32{
// 	let effective_n = rls.effective_n();
// 	let (mean, var) = sampling_distribution(rls);

// 	let y_excess = effective_n*rls.y_bar*rls.y_bar/((rls.y_error_variance + rls.x_var * (rls.m * rls.m)));
// 	//let y_excess = grad*grad/((rls.y_error_variance + rls.x_var * (rls.m * rls.m)));

// 	// Good on variance initialisation, testing sgd now
// 	let split = (-y_excess*2.0).exp();
// 	let step_squared =  (1.0-split)*rls.x_var*0.5 + 2.0*mean*mean;

// 	rls.x_bar
// 	- rls.y_bar.signum() * step_squared.sqrt()
// 	+ (split*var + rls.x_var*0.5 + additional_var).sqrt() * rng.sample::<f32, _>(StandardNormal)
// }

/// Returns sampling distribution mean (distance relative to rls.x_bar) and variance.
#[inline(always)]
fn sampling_distribution(rls: &Rls) -> (f32, f32) {
	// let prior_mean = rls.x_bar - rls.y_bar.signum() * rls.x_var.sqrt() * 0.5;
	let effective_n = rls.effective_n();

	let numer_mean = rls.y_bar;
	let fuzz = 1e-4 * (numer_mean * numer_mean) + 1e-5 * (rls.x_var * rls.m * rls.m);
	let numer_var = rls.y_error_variance / effective_n + fuzz;

	// when the sd of m is larger than m things start to break down. better to steepen m when uncertainty is high as
	// this reduces effective step size
	let denom_var = numer_var / rls.x_var;

	// when the sd of m is larger than m things start to break down. better to steepen m when uncertainty is high as
	// this reduces effective step size
	let denom_mean = rls.m.abs().max(denom_var.sqrt());

	let (mean, var) = gaussian_approx_posterior(rls.x_var, numer_mean, numer_var, denom_mean, denom_var);
	let clip_limit = 3.0; // stop steps that are larger than `clip_limit * x_var.sqrt()`
	let mean = mean
		.min(clip_limit * rls.x_var.sqrt())
		.max(-clip_limit * rls.x_var.sqrt());
	let var = var.min(rls.x_var);

	(mean, var)
}

#[derive(Clone, Debug)]
struct Rls {
	/// exponentially weighted average x
	x_bar: f32,

	/// exponentially weighted average y
	y_bar: f32,

	/// current gradient estimate
	m: f32,

	/// how strongly does regression error respond to changes in slope
	///
	/// exponentially weighted average of (x_variance * lambda * lambda)
	x_var: f32,

	/// exponentially weighted y prediction error variance
	y_error_variance: f32,

	lambda: f32,

	// where relative to the new x_bar did the new regression line cross the old regression line
	change_breakeven: f32,

	// was the change in the regression line positive to the right of the break even
	change_right_positive: bool,
}

// enum RlsUpdate {
// 	FixLambda { correction: f32 },
// 	UpdateLambda { min_lambda: f32, max_lambda: f32 },
// }

impl Rls {
	fn new(lambda: f32) -> Self {
		Rls {
			// required for regression
			x_bar: 0.0,
			y_bar: 0.0,
			m: 0.0,
			x_var: EPSILON * EPSILON,

			y_error_variance: 0.0,

			lambda,

			// where relative to the new x_bar did the new regression line cross the old regression line
			change_breakeven: 0.0,

			// was the change in the regression line positive to the right of the break even
			change_right_positive: false,
		}
	}

	fn effective_n(&self) -> f32 {
		2.0 / (1.0 - self.lambda) - 1.0 // maybe this should be redone  as 2.0 * lambda_prod.ln / lambda.ln - 1
	}

	fn update_lambda(&mut self, decrease_lambda: bool, min_lambda: f32, max_lambda: f32, correction: f32) {
		let scale = 2.0; //::std::f32::consts::FRAC_1_SQRT_2; // numbers lower than 1 slow down the changes to lambda
		let bias = ::std::f32::consts::FRAC_1_SQRT_2; // make lambda decreases slightly smaller than increases, so lambda increases slowly when results are random
		if decrease_lambda {
			let x = 1.0 - self.lambda;
			self.lambda = 1.0 - x / (1.0 - x * scale * bias);
		} else {
			self.lambda = (self.lambda + (scale - scale * self.lambda)) / (1.0 + (scale - scale * self.lambda))
		}
		self.lambda = self.lambda.max(min_lambda).min(max_lambda);
	}

	/// add new exponentially weighted (x, y) data point
	#[inline(always)]
	fn add(&mut self, x: f32, y: f32, min_lambda: f32, max_lambda: f32, correction: f32) {
		
		let m0 = self.m;
		let x_bar0 = self.x_bar;
		let y_bar0 = self.y_bar;

		let x_diff = x - self.x_bar;
		let y_diff = y - self.y_bar;
		let pred_err = y_diff - x_diff * self.m; // pred_err = (y - (y_bar + (x - x_bar)*m))

		// if the following are true then decrease lambda, because it would have moved the regression in the correct direction.
		let decrease_lambda = (x > self.x_bar + self.change_breakeven) // is new x to the right of breakeven point?
							^ (pred_err > 0.0) // is y greater than regression prediction?
							^ self.change_right_positive; // was change in regression to right side of breakeven point positive?

		// let correction = match update_type {
		// 	RlsUpdate::FixLambda { correction } => correction,
		// 	RlsUpdate::UpdateLambda { min_lambda, max_lambda } => {
		// 		self.update_lambda(decrease_lambda, min_lambda, max_lambda);
		// 		1.0
		// 	}
		// };
		self.update_lambda(decrease_lambda, min_lambda, max_lambda, correction);
		

		self.y_error_variance =
			self.y_error_variance * self.lambda + pred_err * pred_err * ((1.0 - self.lambda) / correction);

		let correction = 1.0;
		

		self.x_bar += x_diff * ((1.0 - self.lambda) / correction);
		self.y_bar += y_diff * ((1.0 - self.lambda) / correction);

		

		// m update
		let x_diff2 = x - self.x_bar;
		let y_diff2 = y - self.y_bar;

		let new_m_stiffness = ((1.0 - self.lambda) / correction * x_diff2) * x_diff2 / self.lambda;
		self.x_var *= self.lambda;

		// combine stiffness and new_m calculation to avoid division by zero
		let new_m_by_new_m_stiffness = ((1.0 - self.lambda) / correction * x_diff2) * y_diff2 / self.lambda; // = new_m * new_m_stiffness where let new_m = y_diff2 / x_diff2;
		self.m = (self.m * self.x_var + new_m_by_new_m_stiffness) / (self.x_var + new_m_stiffness);
		self.x_var += new_m_stiffness;

		// remember point that the regression rotates around when updated.
		self.change_breakeven = ((self.x_bar - x_bar0) * m0 - (self.y_bar - y_bar0)) / (self.m - m0);
		self.change_right_positive = self.m > m0;
	}
}

// returns (mean, var) of a normal distribution approximating the posterior
#[inline(always)]
fn gaussian_approx_posterior(
	prior_var: f32,
	numer_mean: f32,
	numer_var: f32,
	denom_mean: f32,
	denom_var: f32,
) -> (f32, f32) {
	// let numer_mean = -numer_mean;
	let denom_mean = denom_mean;

	let lower = (-numer_mean / denom_mean).min(0.0);
	let upper = (-numer_mean / denom_mean).max(0.0);

	let (x, _slope, curv) = find_root2(lower, upper, |x| {
		let (_, first_deriv, second_deriv) =
			ratio_posterior(x, prior_var, numer_mean, numer_var, denom_mean, denom_var);
		(first_deriv, second_deriv)
	});

	(x, gaussian_approx_var(curv))
}

#[inline(always)]
fn gaussian_approx_var(log_pdf_second_deriv: f32) -> f32 {
	(0.5 / log_pdf_second_deriv).abs()
}

/// find the root of a monotonic decreasing function
#[inline(always)]
fn find_root2<F: Fn(f32) -> (f32, f32)>(x_min: f32, x_max: f32, f: F) -> (f32, f32, f32) {
	let mut x_min = x_min;
	let mut x_max = x_max;

	let (y_min, _) = f(x_min);
	let (y_max, _) = f(x_max);

	debug_assert!(y_min >= 0.0);
	debug_assert!(y_max <= 0.0);

	let iters = 150;
	for i in 0..iters {
		let x_new = (x_max + x_min) * 0.5;

		let (y_new, y_grad_new) = f(x_new);

		let delta = y_new / y_grad_new; // how close to root based on most recent point
		let length_scale = gaussian_approx_var(y_grad_new).sqrt() * 0.01;

		if i + 1 == iters || length_scale > delta {
			return (x_new - y_new / y_grad_new, y_new, y_grad_new);
		}

		if y_new > 0.0 {
			x_min = x_new;
		} else {
			x_max = x_new;
		}
	}

	debug_assert!(false);
	return (0.0, 0.0, 0.0);
}

// /// find the root of a monotonic decreasing function
// #[inline(always)]
// fn find_root<F: Fn(f32) -> (f32, f32)>(x_min: f32, x_max: f32, f: F) -> (f32, f32, f32) {
// 	let mut x_min = x_min;
// 	let mut x_max = x_max;

// 	let (mut y_min, mut y_grad_min) = f(x_min);
// 	let (mut y_max, mut y_grad_max) = f(x_max);

// 	debug_assert!(y_min >= 0.0);
// 	debug_assert!(y_max <= 0.0);

// 	let iters = 10;
// 	for i in 0..iters {
// 		let x_new = if y_min.abs() > y_max.abs() {
// 			let (lower_bound, upper_bound) = (0.5*(x_min + x_max), x_max);
// 			let x_new_min = (x_min - y_min/y_grad_min).max(lower_bound).min(upper_bound);
// 			let x_new_max = (x_max - y_max/y_grad_max).max(lower_bound).min(upper_bound);
// 			0.5*(x_new_min + x_new_max) - (x_new_min - x_new_max).abs()
// 		} else {
// 			let (lower_bound, upper_bound) = (x_min, 0.5*(x_min + x_max));
// 			let x_new_min = (x_min - y_min/y_grad_min).max(lower_bound).min(upper_bound);
// 			let x_new_max = (x_max - y_max/y_grad_max).max(lower_bound).min(upper_bound);
// 			0.5*(x_new_min + x_new_max) + (x_new_min - x_new_max).abs()
// 		};

// 		let (y_new, y_grad_new) = f(x_new);

// 		if y_new > 0.0 {
// 			x_min = x_new;
// 			y_min = y_new;
// 			y_grad_min = y_grad_new;
// 		} else {
// 			x_max = x_new;
// 			y_max = y_new;
// 			y_grad_max = y_grad_new
// 		}

// 		if i + 1 == iters {
// 			return (x_new, y_new, y_grad_new);//y_grad_min.min(y_grad_max));
// 		}
// 	}

// 	unreachable!()
// }

#[inline(always)]
fn ratio_posterior(
	x: f32,
	prior_var: f32,
	numer_mean: f32,
	numer_var: f32,
	denom_mean: f32,
	denom_var: f32,
) -> (f32, f32, f32) {
	// where a = denom_mean, b=numer_mean, c=denom_sd, d= numer_sd, e = prior_sd
	// post log-pdf = -0.5*((a*x-b)^2/(c^2*x^2 + d^2) + x^2/e^2)
	// deriv = (-a^2 d^2 x - a b c^2 x^2 + a b d^2 + b^2 c^2 x)/(c^2 x^2 + d^2)^2 - x/e^2
	// second deriv = -a^2/(c^2 x^2 + d^2) + (4 a c^2 x (a x - b))/(c^2 x^2 + d^2)^2 - ((b - a x)^2 (3 c^4 x^2 - c^2
	// d^2))/(c^2 x^2 + d^2)^3 - 1/e^2

	// prior log-pdf = -0.5 * x^2/prior_var
	// log_likelihood = -0.5*(denom_mean*x - numer_mean)^2/(denom_var*x^2 + numer_var) + C
	// posterior log-pdf = -0.5*((denom_mean*x - numer_mean)^2/(denom_var*x^2 + numer_var) + x^2/prior_var) + C
	// posterior log-pdf grad = ((numer_mean^2*denom_var-denom_mean^2*numer_var)*x - denom_mean*numer_mean*denom_var*x^2
	// + denom_mean*numer_mean*numer_var)/(denom_var x^2 + numer_var)^2 - x/prior_var

	let e2 = prior_var;
	let c2 = denom_var;
	let d2 = numer_var;
	let a = denom_mean;
	// this negative is because the calcs below are for a ratio distribution, but this is a negative ratio "-c/m" given
	// "y = mx+c"
	let b = -numer_mean;
	let a2 = a * a;
	let b2 = b * b;
	let x2 = x * x;

	let axmb = a * x - b;
	let axmb2 = axmb * axmb;

	let c2x2pd2 = c2 * x2 + d2;
	let c2x2pd22 = c2x2pd2 * c2x2pd2;

	let log_pdf = -0.5 * (axmb2 / c2x2pd2 + x2 / e2);
	let log_pdf_deriv = ((b2 * c2 - a2 * d2) * x + a * b * (d2 - c2 * x2)) / c2x2pd22 - x / e2;
	let log_pdf_second_deriv = -a2 / c2x2pd2 + (4.0 * a * c2 * x * axmb) / c2x2pd22
		- (axmb2 * c2 * (3.0 * c2 * x2 - d2)) / (c2x2pd2 * c2x2pd22)
		- 1.0 / e2;

	(log_pdf, log_pdf_deriv, log_pdf_second_deriv)
}
