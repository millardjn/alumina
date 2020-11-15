//! Types and tools for the initialisation of node values.
use ndarray::{ArrayD, ArrayViewMutD, IxDyn};
use rand::thread_rng;
use rand_distr::{Distribution, Normal, Uniform};
use std::{
	fmt,
	ops::DerefMut,
	sync::{Arc, Mutex},
};

/// Wrapper for initialiser closures that implements `Clone` and `Debug`
#[derive(Clone)]
pub struct Initialiser {
	name: String,
	func: Arc<Mutex<dyn FnMut(ArrayViewMutD<f32>) + Send>>,
}

impl Initialiser {
	pub fn new<F: 'static + FnMut(ArrayViewMutD<f32>) + Send>(name: String, func: F) -> Self {
		Initialiser {
			name,
			func: Arc::new(Mutex::new(func)),
		}
	}

	// 	pub fn wrap(name: String, func: Arc<Mutex<FnMut(ArrayViewMutD<f32>, Option<&OpInstance>)>>) -> Self {
	// 		Initialiser {
	// 			name,
	// 			func,
	// 			op_id: None,
	// 		}
	// 	}

	// Panics if node has non-known dimensions
	pub fn array(&mut self, shape: IxDyn) -> ArrayD<f32> {
		let mut arr = ArrayD::zeros(shape);
		self.call(arr.view_mut());
		arr
	}

	pub fn call(&mut self, arr: ArrayViewMutD<f32>) {
		let mut guard = self
			.func
			.lock()
			.unwrap_or_else(|_| panic!("Could not acquire lock on initialiser: {:?}", self));
		guard.deref_mut()(arr);
	}
}

impl fmt::Debug for Initialiser {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "Initialiser {{ name: {}, .. }}", self.name)
	}
}

/// Gaussian initialisation
///
/// This initialises with gaussian values drawn from N(mean, std_dev^2).
pub fn gaussian(mean: f32, std_dev: f32) -> Initialiser {
	Initialiser::new(
		format!("Gaussian Initialiser{{mean: {}, std_dev :{}}}", mean, std_dev),
		move |mut arr: ArrayViewMutD<f32>| {
			let mut rng = thread_rng();
			let norm = Normal::new(f64::from(mean), f64::from(std_dev)).expect("Could not create normal distribution");
			for e in arr.iter_mut() {
				*e = norm.sample(&mut rng) as f32;
			}
		},
	)
}

/// MSRA initialisation
///
/// This initialises with gaussian values drawn from N(0, multiplier*shape[-1]/shape_product).
///
/// For use with conv and linear Ops.
pub fn msra(multiplier: f32) -> Initialiser {
	Initialiser::new(
		format!("MSRA Initialiser{{multiplier: {}}}", multiplier),
		move |mut arr: ArrayViewMutD<f32>| {
			let mut rng = thread_rng();
			//let inputs_per_output = *arr.shape().get(0).unwrap_or(&1); // output channels is the last axis
			let inputs_per_output = arr.shape()[0..arr.ndim() -1].iter().product::<usize>();
			let norm = Normal::new(0.0, f64::from((multiplier / inputs_per_output as f32).sqrt()))
				.expect("Could not create normal distribution");
			for e in arr.iter_mut() {
				*e = norm.sample(&mut rng) as f32;
			}
		},
	)
}

/// Uniform initialisation
///
/// This initialises uniform values drawn from [low, high).
pub fn uniform(low: f32, high: f32) -> Initialiser {
	Initialiser::new(
		format!("Uniform Initialiser{{low: {}, high :{}}}", low, high),
		move |mut arr: ArrayViewMutD<f32>| {
			let mut rng = thread_rng();
			let rang = Uniform::new(low, high);
			for e in arr.iter_mut() {
				*e = rang.sample(&mut rng) as f32;
			}
		},
	)
}

/// Duplicate initialisation
///
/// Sets all elements to the supplied value
pub fn duplicate(val: f32) -> Initialiser {
	Initialiser::new(
		format!("Duplicate Initialiser{{val: {}}}", val),
		move |mut arr: ArrayViewMutD<f32>| {
			for e in arr.iter_mut() {
				*e = val;
			}
		},
	)
}
