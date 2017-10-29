use std::sync::{Mutex, Arc};
use std::fmt;
use std::ops::DerefMut;
use new::ops::{OpInstance};
use new::graph::OpID;
use ndarray::ArrayD;
use rand::{thread_rng, Isaac64Rng, Rng};
use rand::distributions::{Sample, Normal, Range};

/// Wrapper for initialiser closures that implements `Clone` and `Debug`
#[derive(Clone)]
pub struct Initialiser {
	name: String,
	func: Arc<Mutex<FnMut(&mut ArrayD<f32>, Option<&OpInstance>)>>,
	op_id: Option<OpID>,
}

impl Initialiser {
	pub fn new<F: 'static + FnMut(&mut ArrayD<f32>, Option<&OpInstance>)>(name: String, func: F) -> Self {
		Initialiser {
			name: name,
			func: Arc::new(Mutex::new(func)),
			op_id: None,
		}
	}

	pub fn wrap(name: String, func: Arc<Mutex<FnMut(&mut ArrayD<f32>, Option<&OpInstance>)>>) -> Self {
		Initialiser {
			name: name,
			func: func,
			op_id: None,
		}
	}

	/// Guassian initialisation
	///
	/// This initialises with gaussian values drawn from N(mean, std_dev^2).
	pub fn guassian(mean: f32, std_dev: f32) -> Initialiser {
		Initialiser::new("Gaussian Initialiser".to_string(), move |arr: &mut ArrayD<f32>, _instance: Option<&OpInstance>|{
			let mut rng = thread_rng().gen::<Isaac64Rng>();
			let mut norm = Normal::new(mean as f64, std_dev as f64);
			for e in arr.iter_mut() {
				*e = norm.sample(&mut rng) as f32;
			}
		})
	}

	/// Uniform initialisation
	///
	/// This initialises uniform values drawn from [low, high).
	pub fn uniform(low: f32, high: f32) -> Initialiser {
		Initialiser::new("Uniform Initialiser".to_string(), move |arr: &mut ArrayD<f32>, _instance: Option<&OpInstance>|{
			let mut rng = thread_rng().gen::<Isaac64Rng>();
			let mut rang = Range::new(low, high);
			for e in arr.iter_mut() {
				*e = rang.sample(&mut rng) as f32;
			}
		})
	}

	/// Fill initialisation
	///
	/// Sets all elements to the supplied value
	pub fn fill(val: f32) -> Initialiser {
		Initialiser::new("Fill Initialiser".to_string(), move |arr: &mut ArrayD<f32>, _instance: Option<&OpInstance>|{
			for e in arr.iter_mut() {
				*e = val;
			}
		})
	}

	pub fn call(&self, arr: &mut ArrayD<f32>, op: Option<&OpInstance>) {
		let mut guard = self.func.lock().expect(&format!("Could not acquire lock on initialiser: {:?}", self));
		guard.deref_mut()(arr, op);
	}

	pub fn set_op_id(mut self, op_id: OpID) -> Self {
		self.op_id = Some(op_id);
		self
	}

	pub fn clear_op_id(mut self) -> Self {
		self.op_id = None;
		self
	}

	/// The OpID of the associated operation
	///
	/// if None then None will be passed to call()
	pub fn op_id(&self) -> Option<OpID> {
		self.op_id.clone()
	}
}

impl fmt::Debug for Initialiser {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "Initialiser {{ name: {}, .. }}", self.name)
	}
}