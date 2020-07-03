use crate::{calc_change_sqr, GradientStepper};
use alumina_core::{errors::ExecError, graph::Node};
use indexmap::{indexmap, IndexMap};
use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, Zip, ArcArray, IxDyn};
use rand::{thread_rng, Rng, SeedableRng};
use rand_distr::{Exp1, StandardNormal};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use std::f32::EPSILON;
use std::sync::Arc;

#[derive(Clone)]
pub struct Soop {
	lambda: f32,
	lambda_prod: f32,
	rls: IndexMap<Node, ArrayD<Rls>>,
	step_count: usize,

	/// Oh boy
	update_fn: Arc<
		dyn Fn(Pcg64Mcg, ArrayViewMutD<Rls>, ArrayViewD<f32>, ArrayViewMutD<f32>, f32, f32, f32, f32) + Sync + Send,
	>,

	param_weight: f32,
	param_var_weight: f32,
	tensor_var_weight: f32,
	static_var: f32,
}

impl Soop {
	/// Create an optimisation problem assuming that all nodes marked `Parameter` should be optimised.
	pub fn new(lambda: f32) -> Self {
		let mut opt = Soop {
			lambda,
			lambda_prod: 1.0,
			rls: indexmap![],
			step_count: 0,

			update_fn: Arc::new(|_, _, _, _, _, _, _, _| {}),

			param_weight: 0.99,
			param_var_weight: 0.01,
			tensor_var_weight: 0.0001,
			static_var: EPSILON * EPSILON,
		};
		opt.variance_addition(|_p2, _g2, _lp| {
			EPSILON*EPSILON // permanent noise at small fixed scale
			//+ (1.0-lp)*1e-6*p2 // permanent noise at 0.1% of parameter scale
			//+ (1.0-lp)*1e-6*g2 // permanent noise at 0.1% of gradient scale
		});
		opt
	}

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
	///  * parameter_sqr
	///  * gradient_sqr
	///  * lambda_product(product of lamda at each step, asymptotically decreasing from 1 toward 0)
	///
	/// The default function applies a temporary amount of additional variance base on
	/// the gradient magnitued to unstick parameters with otherwise small variance:
	///
	/// ```rust
	/// |p2, g2, lp| {
	/// 	EPSILON*EPSILON // permanent noise at small fixed scale
	/// 	+ (1.0-lp)*1e-6*p2 // permanent noise at 0.1% of parameter scale
	/// 	+ (1.0-lp)*1e-6*g2 // temporary noise at 0.1% of gradient scale
	/// }
	/// ```
	///
	pub fn variance_addition<F>(&mut self, f: F) -> &mut Self
	where
		F: 'static + Sync + Send + Fn(f32, f32, f32) -> f32,
	{
		self.update_fn = Arc::new(
			move |mut rng, rls_arr, param_arr, grad_arr, lambda, lambda_prod, effective_n, scale| {
				Zip::from(rls_arr)
					.and(param_arr)
					.and(grad_arr)
					.apply(|rls, param, grad| {
						let pred_error = rls.add(*param, *grad, lambda, 1.0);
						let added_var = f(param*param, *grad**grad, lambda_prod);
						*grad = standard_update(
							rls,
							&mut rng,
							effective_n,
							added_var,
							lambda_prod,
							*grad,
							scale,
							pred_error,
						);
					});
			},
		);
		self
	}

	/// Rls lambda value
	///
	/// The weight of each previously observed datapoint decays by this factor on each new observation
	///
	/// Default: 0.99
	pub fn lambda(&mut self, lambda: f32) -> &mut Self {
		let current_effective_datapoints = (self.lambda_prod.ln() / self.lambda.ln()).min(1.0 / (1.0 - self.lambda));

		self.lambda = lambda;
		self.lambda_prod = lambda.powf(current_effective_datapoints);

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
		let lambda = self.lambda;
		//let one_minus_lambda = 1.0 - lambda;
		self.lambda_prod *= lambda;
		let lambda_prod = self.lambda_prod;
		//let correction = 1.0;// - self.lambda_prod;
		let effective_n = Rls::effective_n(lambda); // this should be redone  as 2.0 * lambda_prod.ln / lambda.ln - 1

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
					let mut rls = Rls::new();
					rls.x_bar = *e * param_mul;
					rls.x_var = static_var + *e * *e * param_var_weight + var * tensor_var_weight;
					rls
				})
			});
		}

		// sample new parameters and
		// write new parameter array into grad array
		//let scale = MogVarScale::Exp1.sd_scale();
		//let scale = thread_rng().sample::<f32, _>(StandardNormal) + 1.0;
		//let scale = thread_rng().sample::<f32, _>(Exp1) * ::std::f32::consts::FRAC_1_SQRT_2;
		let scale = thread_rng().sample::<f32, _>(Exp1);
		//let scale = thread_rng().sample::<f32, _>(StandardNormal).abs();
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
				// Zip::from(&param_arr)
				// 	.and(rls_arr.view_mut())
				// 	.and(grad_arr.view_mut())
				// 	.par_apply(|&param, rls, grad| {
				// 		rls.add(param, *grad, lambda, correction);
				// 	});
				// Rather than par_apply over whole array, only parallelise over outer dim so that
				Zip::from(rls_arr.outer_iter_mut())
					.and(param_arr.outer_iter())
					.and(grad_arr.outer_iter_mut())
					.par_apply(|rls_arr, param_arr, grad_arr| {
						let rng = Pcg64Mcg::from_rng(rand::thread_rng()).unwrap();
						update_fn(
							rng,
							rls_arr,
							param_arr,
							grad_arr,
							lambda,
							lambda_prod,
							effective_n,
							scale,
						);
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

		self.step_count += 1;

		Ok(change_sqr.sqrt())
	}

	fn best_estimate(&self, params: &mut dyn Iterator<Item = &Node>) -> IndexMap<Node, ArcArray<f32, IxDyn>> {
		params
		.map(|param| {
			(param.clone(),
			self.rls.get_full(param).map(|(_i, _param, rls_arr)| rls_arr.map(|rls| rls.x_bar).to_shared())
			.unwrap_or_else(|| {
				let mut shape = param.shape().clone();
				shape.collapse_dimensions_to_minimum();
				ArcArray::zeros(shape.to_data_shape().unwrap())
			}))		
		}).collect()
	}
}




fn standard_update<R: Rng>(
	rls: &Rls,
	rng: &mut R,
	effective_n: f32,
	additional_var: f32,
	lambda_prod: f32,
	grad: f32,
	scale: f32,
	_prediction_error: f32,
) -> f32 {
	let (mean, var) = sampling_distribution(rls, effective_n);

	let total_y_var = rls.y_error_variance + rls.x_var * (rls.m * rls.m);

	let split1 = total_y_var / (total_y_var + 2.0*effective_n * rls.y_bar * rls.y_bar);
	//let split1 = 1.0 - (1.0 - split1)*(1.0-lambda_prod); 
	//let split2 = total_y_var/(total_y_var + effective_n*rls.y_bar*rls.y_bar*0.5);

	let mean_split = mean * mean / (mean * mean + 2.0*rls.x_var);
	let x_var_split = rls.y_error_variance / (rls.y_error_variance + 2.0*rls.x_var * (rls.m * rls.m));

	let x = (1.0 - mean_split) * x_var_split;
	//let x = 1.0 - (1.0 - x)*(1.0-lambda_prod); // force x to be 1 early
	let z = 1.0 - (1.0 - x)*split1;
	

	//let s = rng.sample::<f32, _>(StandardNormal);
	//let s = rng.sample::<f32, _>(Exp1); //* ::std::f32::consts::FRAC_1_SQRT_2;
	let s = 1.0;// 0.8404 on cifar10!! very good
	let step_squared = z * rls.x_var*s  + 2.0*(mean * mean) * (1.0 - x);

	rls.x_bar - rls.y_bar.signum() * step_squared.sqrt()
		+ (2.0*var * (1.0 - x) + additional_var).sqrt() // + (mean * mean) * (1.0 - x)
			* rng.sample::<f32, _>(StandardNormal)
}





// excellent without added variance, 0.8383 0.5232
// step:6250       validation accuracy: 0.8301 loss: 0.5278609
// step:7250       validation accuracy: 0.8349 loss: 0.5636797
// step:15600      loss:0.48490906 change:31.242823
// step:30000      loss:1.8442583  change:222.77081 validation accuracy: 0.7324 loss: 0.8368162

// excellent with added variance (1e-6 both)
// step:6250       validation accuracy: 0.8324 loss: 0.52011263
// fn standard_update<R: Rng>(rls: &Rls, rng: &mut R, effective_n: f32, additional_var: f32, lambda_prod: f32, grad: f32, scale: f32, _prediction_error: f32) -> f32{
// 	let (mean, var) = sampling_distribution(rls, effective_n);

// 	let total_y_var = rls.y_error_variance + rls.x_var * (rls.m * rls.m);

// 	let split1 = total_y_var/(total_y_var + effective_n*rls.y_bar*rls.y_bar);
// 	let split2 = total_y_var/(total_y_var + effective_n*rls.y_bar*rls.y_bar*0.5);

// 	let x_var_split = rls.y_error_variance/total_y_var;

// 	let step_squared = 2.0*x_var_split*rls.x_var*(1.0-split2) + (1.0 - x_var_split)*(2.0*mean*mean);

// 	rls.x_bar
// 	- rls.y_bar.signum() * step_squared.sqrt()
// 	+ (x_var_split*rls.x_var*split1 + var*(1.0-x_var_split) + additional_var).sqrt() * rng.sample::<f32, _>(StandardNormal)
// }

// excellent without added variance
// step:6250       validation accuracy: 0.8311 loss: 0.5227267
// step:10000      validation accuracy: 0.8417 loss: 0.5806931
// step:15600      loss:0.7345025  change:60.273537
//
// excellent with added variance (1e-6 both)
// step:5750       validation accuracy: 0.8291 loss: 0.5239293
// step:7500       validation accuracy: 0.8394 loss: 0.5300053
// step:15600      loss:1.0422609  change:61.793705
// fn standard_update<R: Rng>(rls: &Rls, rng: &mut R, effective_n: f32, additional_var: f32, lambda_prod: f32, grad: f32, scale: f32, _prediction_error: f32) -> f32{
// 	let (mean, var) = sampling_distribution(rls, effective_n);

// 	let total_y_var = rls.y_error_variance + rls.x_var * (rls.m * rls.m);

// 	let split1 = total_y_var/(total_y_var + effective_n*rls.y_bar*rls.y_bar);
// 	let split2 = total_y_var/(total_y_var + effective_n*rls.y_bar*rls.y_bar*0.5);

// 	let x_var_split = rls.y_error_variance/total_y_var;

// 	let step_squared = 2.0*x_var_split*(rls.x_var-var)*(1.0-split2) + (1.0 - x_var_split)*(2.0*mean*mean); //2.0*mean*mean*(1.0-x_var_split); // rls.x_var*x_var_split

// 	rls.x_bar
// 	- rls.y_bar.signum() * step_squared.sqrt()
// 	+ (x_var_split*rls.x_var*split2 + var*(1.0-x_var_split) + additional_var).sqrt() * rng.sample::<f32, _>(StandardNormal) // + (rls.x_var-var)*x_var_split       (var)*(1.0-x_var_split) + 1.0*(var)*x_var_split*split1
// }

// retest with lamdaprod
// excellent without added variance,
// step:7000       validation accuracy: 0.8321 loss: 0.5244251
// step:8500       validation accuracy: 0.8356 loss: 0.5504822
// step:14700      loss:0.48072785 change:36.674397
// _ with added variance (1e-6 both)
// fn standard_update<R: Rng>(rls: &Rls, rng: &mut R, effective_n: f32, additional_var: f32, lambda_prod: f32, grad: f32, scale: f32, _prediction_error: f32) -> f32{
// 	let (mean, var) = sampling_distribution(rls, effective_n);

// 	let total_y_var = rls.y_error_variance + rls.x_var * (rls.m * rls.m);

// 	let split1 = total_y_var/(total_y_var + effective_n*rls.y_bar*rls.y_bar);
// 	let split2 = total_y_var/(total_y_var + effective_n*rls.y_bar*rls.y_bar*0.5);

// 	let x_var_split = (1.0-lambda_prod)*rls.y_error_variance/total_y_var;

// 	let step_squared = 2.0*x_var_split*(rls.x_var-var)*(1.0-split2) + (1.0 - x_var_split)*(2.0*mean*mean); //2.0*mean*mean*(1.0-x_var_split); // rls.x_var*x_var_split

// 	rls.x_bar
// 	- rls.y_bar.signum() * step_squared.sqrt()
// 	+ (x_var_split*rls.x_var*split2 + var*(1.0-x_var_split) + additional_var).sqrt() * rng.sample::<f32, _>(StandardNormal) // + (rls.x_var-var)*x_var_split       (var)*(1.0-x_var_split) + 1.0*(var)*x_var_split*split1
// }

// fn standard_update<R: Rng>(rls: &Rls, rng: &mut R, effective_n: f32, additional_var: f32, lambda_prod: f32, grad: f32, scale: f32, _prediction_error: f32) -> f32{
// 	let (mean, var) = sampling_distribution(rls, effective_n);

// 	let total_y_var = rls.y_error_variance + rls.x_var * (rls.m * rls.m);

// 	let split1 = total_y_var/(total_y_var + effective_n*rls.y_bar*rls.y_bar);
// 	let split2 = total_y_var/(total_y_var + effective_n*rls.y_bar*rls.y_bar*0.5);

// 	let x_var_split = (1.0-lambda_prod)*rls.y_error_variance/total_y_var;

// 	let x = 0.75*x_var_split*split1;
// 	let z = var/rls.x_var;

// 	let step_squared = 2.0*x_var_split*(rls.x_var)*(1.0-split2) + (2.0*mean*mean); //2.0*mean*mean*(1.0-x_var_split); // rls.x_var*x_var_split

// 	rls.x_bar
// 	- rls.y_bar.signum() * step_squared.sqrt()
// 	+ (rls.x_var*x + var*(1.0-x) + additional_var).sqrt() * rng.sample::<f32, _>(StandardNormal) // + (rls.x_var-var)*x_var_split       (var)*(1.0-x_var_split) + 1.0*(var)*x_var_split*split1
// }

// //untested eh
// fn standard_update<R: Rng>(rls: &Rls, rng: &mut R, effective_n: f32, additional_var: f32, lambda_prod: f32, grad: f32, scale: f32, _prediction_error: f32) -> f32{
// 	let (mean, var) = sampling_distribution(rls, effective_n);

// 	let total_y_var = rls.y_error_variance + rls.x_var * (rls.m * rls.m);

// 	let split1 = total_y_var/(total_y_var + effective_n*rls.y_bar*rls.y_bar);
// 	let split2 = total_y_var/(total_y_var + effective_n*rls.y_bar*rls.y_bar*0.5);

// 	let x_var_split = (1.0 - lambda_prod)*rls.y_error_variance/total_y_var;

// 	//let z = split2*split2;//(y_excess/(y_excess+1.0))*(y_excess/(y_excess + 2.0)); //(1.0+split1-split2*2.0).max(0.0)

// 	let step_squared = 2.0*rls.x_var*(1.0 - split2);

// 	rls.x_bar
// 	- rls.y_bar.signum() * step_squared.sqrt()
// 	+ (x_var_split*rls.x_var*split1 + var*(1.0-x_var_split) + additional_var).sqrt() * rng.sample::<f32, _>(StandardNormal)
// }

// // doesnt expand enough
// fn standard_update<R: Rng>(rls: &Rls, rng: &mut R, effective_n: f32, additional_var: f32, lambda_prod: f32, grad: f32, scale: f32, _prediction_error: f32) -> f32{
// 	let (mean, var) = sampling_distribution(rls, effective_n);

// 	let total_y_var = rls.y_error_variance + rls.x_var * (rls.m * rls.m);

// 	let split1 = total_y_var/(total_y_var + effective_n*rls.y_bar*rls.y_bar);
// 	//let split2 = total_y_var/(total_y_var + effective_n*rls.y_bar*rls.y_bar*0.5);

// 	let x_var_split = rls.y_error_variance/total_y_var;

// 	let step_squared = x_var_split*(rls.x_var-var)*(1.0-split1) + (1.0 - x_var_split)*(2.0*mean*mean); //2.0*mean*mean*(1.0-x_var_split); // rls.x_var*x_var_split

// 	rls.x_bar
// 	- rls.y_bar.signum() * step_squared.sqrt()
// 	+ (x_var_split*rls.x_var*split1 + var*(1.0-x_var_split) + additional_var).sqrt() * rng.sample::<f32, _>(StandardNormal) // + (rls.x_var-var)*x_var_split       (var)*(1.0-x_var_split) + 1.0*(var)*x_var_split*split1
// }

// // works really well
// #[allow(unused)]
// fn standard_update<R: Rng>(rls: &Rls, rng: &mut R, effective_n: f32, additional_var: f32, lambda_prod: f32, grad: f32, scale: f32, _prediction_error: f32) -> f32{
// 	let (mean, var) = sampling_distribution(rls, effective_n);
// 	//assert!(var <= rls.x_var);

// 	let y_excess = effective_n*rls.y_bar*rls.y_bar/((rls.y_error_variance + rls.x_var * (rls.m * rls.m)));
// 	let exploration_ratio = (rls.x_var * (rls.m * rls.m))/rls.y_error_variance;

// 	//let y_excess = effective_n*rls.y_error_bar*rls.y_error_bar/rls.y_error_variance;

// 	//let x_var_split = (-exploration_ratio*2.0).exp();
// 	//let split = 0.5 + 0.5*(-y_excess).exp();

// 	let split = 1.0/(1.0 + y_excess*0.5);

// 	let x_var_split = 1.0/(1.0 + exploration_ratio*2.0);

// 	let x = var*(1.0-x_var_split) + rls.x_var*x_var_split;

// 	let step_squared = x*(1.0 - split) + mean*mean; // rls.x_var*x_var_split

// 	rls.x_bar
// 	- rls.y_bar.signum() * step_squared.sqrt()
// 	+ (x*split + additional_var).sqrt() * rng.sample::<f32, _>(StandardNormal)
// }

// // works really well
// #[allow(unused)]
// fn standard_update<R: Rng>(rls: &Rls, rng: &mut R, effective_n: f32, additional_var: f32, lambda_prod: f32, grad: f32, scale: f32, _prediction_error: f32) -> f32{
// 	let (mean, var) = sampling_distribution(rls, effective_n);

// 	let y_excess = effective_n*rls.y_bar*rls.y_bar/((rls.y_error_variance + rls.x_var * (rls.m * rls.m)));
// 	let exploration_ratio = (rls.x_var * (rls.m * rls.m))/rls.y_error_variance;

// 	let x_var_split = (-exploration_ratio*2.0).exp();
// 	let split = 0.5 + 0.5*(-y_excess).exp();

// 	let x = var*(1.0-x_var_split) + rls.x_var*x_var_split;

// 	let step_squared =  (1.0-split)*x + mean*mean;

// 	rls.x_bar
// 	- rls.y_bar.signum() * step_squared.sqrt()
// 	+ (split*x).sqrt() * rng.sample::<f32, _>(StandardNormal)
// }

/// Returns sampling distribution mean (distance relative to rls.x_bar) and variance.
#[inline(always)]
fn sampling_distribution(rls: &Rls, effective_n: f32) -> (f32, f32) {
	let numer_mean = rls.y_bar;
	let fuzz = 1e-4 * (numer_mean * numer_mean) + 1e-5 * (rls.x_var * rls.m * rls.m);
	let numer_var = rls.y_error_variance / effective_n + fuzz;

	let denom_var = numer_var / rls.x_var;
	// if rls.m < 0.0 {
	// 	denom_var = denom_var.max(rls.m*rls.m);
	// }
	// when the sd of m is larger than m things start to break down. better to steepen m when uncertainty is high as
	// this reduces effective step size
	let denom_mean = rls.m.abs().max(denom_var.sqrt());

	let (mean, var) = gaussian_approx_posterior(rls.x_var, numer_mean, numer_var, denom_mean, denom_var);
	let clip_limit = 3.0; // stop steps that are larger than `clip_limit * x_var.sqrt()`
	let mean = mean
		.min(clip_limit * rls.x_var.sqrt())
		.max(-clip_limit * rls.x_var.sqrt());
	let var = var.min(rls.x_var);

	// if rls.m < 0.0 {
	// 	return (mean, rls.x_var);
	// }

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

	y_error_bar: f32,
}

impl Rls {
	fn new() -> Self {
		Rls {
			// required for regression
			x_bar: 0.0,
			y_bar: 0.0,
			m: 0.0,
			x_var: EPSILON * EPSILON,

			y_error_variance: 0.0,
			y_error_bar: 0.0,
		}
	}

	fn effective_n(lambda: f32) -> f32 {
		2.0 / (1.0 - lambda)
	}

	/// Produce the distribution descripting where the root could be for N(mean, variance).
	///
	/// This is the approximately the prediction interval of the regression, sliced by the x axis.
	#[inline(always)]
	#[allow(unused)]
	fn root_confidence_distribution(&self, effective_n: f32) -> (f32, f32) {
		// let effective_n = 2.0/(1.0-lambda)-1.0;
		// let dof = effective_n - 2.0;
		let m = self.m.abs(); // to avoid converging on maxima always take m to be positive

		let root_mean = self.x_bar - self.y_bar / m; // x_intercept of current regression

		let x_diff = self.y_bar / m;
		let confidence_var_at_root =
			self.y_error_variance * (1.0 / effective_n + x_diff * x_diff / (self.x_var * effective_n));

		let root_variance = confidence_var_at_root / m;

		(root_mean, root_variance)
	}

	/// Produce the distribution descripting where the root could be for N(mean, variance).
	///
	/// This is the approximately the prediction interval distribution of the regression, sliced by the x axis.
	#[allow(unused)]
	#[inline(always)]
	fn root_prediction_distribution(&self, effective_n: f32) -> (f32, f32) {
		// let effective_n = 2.0/(1.0-lambda)-1.0;
		// let dof = effective_n - 2.0;
		let m = self.m.abs(); // to avoid converging on maxima always take m to be positive

		let root_mean = self.x_bar - self.y_bar / m; // x_intercept of current regression

		let x_diff = self.y_bar / m;
		let prediction_var_at_root =
			self.y_error_variance * (1.0 + 1.0 / effective_n + x_diff * x_diff / (self.x_var * effective_n));

		let root_variance = prediction_var_at_root / m;

		(root_mean, root_variance)
	}

	/// add new exponentially weighted (x, y) data point
	#[inline(always)]
	fn add(&mut self, x: f32, y: f32, lambda: f32, correction: f32) -> f32 {
		let one_minus_lambda = 1.0 - lambda;

		// Update x_bar, y_bar and y_error_variance
		let x_diff = x - self.x_bar;
		let y_diff = y - self.y_bar;

		let pred_err = y_diff - x_diff * self.m; // pred_err = (y - (y_bar + (x - x_bar)*m))
		self.y_error_variance = self.y_error_variance * lambda + pred_err * pred_err * (one_minus_lambda / correction);
		self.y_error_bar = self.y_error_bar * lambda + pred_err * (one_minus_lambda / correction);

		

		self.x_bar += x_diff * (one_minus_lambda / correction);
		self.y_bar += y_diff * (one_minus_lambda / correction);
		

		// update self.m to a weighted average of the slope to the new point and the existing slope
		// where the weighting is how sensitive the regression error is to changes in m for the new point and the old
		// points
		let x_diff2 = x - self.x_bar;
		let y_diff2 = y - self.y_bar;
		let new_m_stiffness = (one_minus_lambda / correction * x_diff2) * x_diff2 / lambda;
		self.x_var *= lambda;

		// combine stiffness and new_m calculation to avoid division by zero
		let new_m_by_new_m_stiffness = (one_minus_lambda / correction * x_diff2) * y_diff2 / lambda; // = new_m * new_m_stiffness where let new_m = y_diff2 / x_diff2;
		self.m = (self.m * self.x_var + new_m_by_new_m_stiffness) / (self.x_var + new_m_stiffness);
		self.x_var += new_m_stiffness;
		pred_err
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
		let length_scale = gaussian_approx_var(y_grad_new) * 0.001;

		if i + 1 == iters || length_scale > delta*delta {
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
