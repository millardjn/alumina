use alumina_core::{
	errors::ExecError,
	exec::{ExecutionPlan, OpPerf},
	grad::Grad,
	graph::{Node, NodeTag, Op},
	subgraph::{execution_subgraph, SubGraph},
};
use alumina_data::DataStream;
use indexmap::{IndexMap, IndexSet};
use ndarray::{ArcArray, IxDyn};
use ndarray::{ArrayViewD, Zip};
use std::{borrow::Borrow, iter::once};
use unchecked_index as ui;

pub mod adam;
pub mod sgd;

/// Calculates the L2 norm of the difference between two arrays
pub fn calc_change_sqr(arr1: ArrayViewD<f32>, arr2: ArrayViewD<f32>) -> f32 {
	debug_assert_eq!(arr1.shape(), arr2.shape());
	let mut change_sqr: f32 = 0.0;
	let outer = arr1.shape().get(0).cloned().unwrap_or(1);
	let inner = arr1.len() / outer;

	if let (Some(slice1), Some(slice2), true, true) = (arr1.as_slice(), arr2.as_slice(), outer > 16, inner > 64) {
		unsafe {
			let mut sum = vec![0.0; inner];
			for i in 0..outer {
				for j in 0..inner {
					let diff = *ui::get_unchecked(slice1, i * inner + j) - *ui::get_unchecked(slice2, i * inner + j);
					*ui::get_unchecked_mut(sum.as_mut_slice(), j) += diff * diff;
				}
			}
			change_sqr += sum.iter().sum::<f32>();
		}
	} else {
		Zip::from(arr1).and(arr2).for_each(|&a1, &a2| {
			change_sqr += (a1 - a2) * (a1 - a2);
		});
	}
	change_sqr
}

pub struct StepData<'a> {
	pub loss: f32,

	/// Note that `step` is incremented prior to calling callbacks, and will equal 1 after the first step.
	pub step: usize,
	pub change_norm: f32,
	//pub gradient_optimiser: &'a GradientOptimiser,
	pub opt_inner: &'a OptInner,
}

// pub struct StepOptions<'a> {
// 	/// If a subgraph is supplied it will be used for executing the
// 	pub subgraph: Option<&'a SubGraph>,

// 	/// If a loss node is supplied
// 	pub loss: Option<&'a Node>,
// 	pub calc_change: bool,
// }

// impl<'a> Default for StepOptions<'a> {
// 	fn default() -> Self {
// 		StepOptions {
// 			subgraph: None,
// 			loss: None,
// 			calc_change: true,
// 		}
// 	}
// }

pub trait GradientStepper {
	/// Complete one optimisation step.
	///
	/// Parameters are updated in place.
	/// Returns the l2 norm of the parameter changes
	// error, step number, l2 norm of param change, and the new parameters
	fn step(
		&mut self,
		parameters_and_grad_values: IndexMap<Node, ArcArray<f32, IxDyn>>,
		calc_change: bool,
	) -> Result<f32, ExecError>;

	fn step_count(&self) -> usize;

	/// Return best estimate for each node
	fn best_estimate(&self, params: &mut dyn Iterator<Item = &Node>) -> IndexMap<Node, ArcArray<f32, IxDyn>> {
		params
			.into_iter()
			.map(|n| {
				(
					n.borrow().clone(),
					n.borrow().value().unwrap_or_else(|| {
						let mut shape = n.borrow().shape();
						shape.collapse_dimensions_to_minimum();
						ArcArray::zeros(shape.to_data_shape().unwrap())
					}),
				)
			})
			.collect()
	}

	/// If required complete any final step required after optimisation e.g. parameter snapshot averaging
	fn finalise(&self, params: &mut dyn Iterator<Item = &Node>) {
		self.best_estimate(params).iter().for_each(|(param, arr)| {
			param.set_value(arr);
		});
	}
}

pub enum Signal {
	Stop,
	Continue,
}

impl From<()> for Signal {
	fn from(_val: ()) -> Self {
		Signal::Continue
	}
}

pub struct OptInner {
	pub loss: Node,
	pub inputs: IndexSet<Node>,
	pub parameters_and_grads: IndexMap<Node, Node>,
	pub subgraph: SubGraph,
}

pub struct GradientOptimiser<'a, S>
where
	S: GradientStepper,
{
	inner: OptInner,
	callbacks: Vec<Box<dyn 'a + FnMut(&mut S, &StepData) -> Signal>>,
	perf_records: Option<&'a mut IndexMap<Op, OpPerf>>,
	grad_step: S,
	calc_change: bool,
	calc_loss: bool,
}

impl<'a, S> GradientOptimiser<'a, S>
where
	S: GradientStepper,
{
	pub fn new<I1, I2, T>(loss: I1, inputs: T, grad_step: S) -> Self
	where
		I1: Into<Node>,
		I2: Into<Node>,
		T: IntoIterator<Item = I2>,
	{
		let loss = loss.into();
		let mut count = 0;
		let inputs: IndexSet<Node> = inputs.into_iter().map(Into::into).inspect(|_| count += 1).collect();
		assert_eq!(inputs.len(), count, "inputs contained duplicates");

		// TODO only parameters required to calculate loss should be included, no more
		let parameters = loss
			.graph()
			.nodes_tagged(NodeTag::Parameter)
			.into_iter()
			.filter(|p| !inputs.contains(p));

		let parameters_and_grads = Grad::of(&loss)
			.wrt(parameters)
			.include_intermediate(false)
			.build()
			.expect("Construction of Grad failed in GradientOptimiser");

		// init parameters
		for param in parameters_and_grads.keys() {
			assert!(
				param.shape().is_known(),
				"Parameter shapes must be fully known. Parameter {} has shape {}",
				param,
				param.shape()
			);

			if let Some(arr) = param.value() {
				if param.shape() != arr.shape().iter().into() {
					let x = arr.broadcast(param.shape().to_data_shape().unwrap()).unwrap();
					param.set_value(x);
				}
			} else if let Some(array) = param.init_array() {
				param.set_value(array);
			} else {
				panic!("Parameter {} does not have an initialiser or value set", param)
			}
		}

		let subgraph = execution_subgraph(inputs.iter(), parameters_and_grads.values().chain(once(&loss)), false)
			.expect("Call to execution_subgraph(..) failed");

		GradientOptimiser {
			inner: OptInner {
				loss,
				inputs,
				parameters_and_grads,
				subgraph,
			},
			callbacks: vec![],
			perf_records: None,
			grad_step,
			calc_change: true,
			calc_loss: true,
		}
	}

	// Unlike new, this method does not initialise parameters which can result in an InsuffiicientInputs error when the
	// graph is executed.
	pub fn new_with<I1, I2, T>(
		loss: I1,
		inputs: T,
		grad_step: S,
		parameters_and_grads: IndexMap<Node, Node>,
		subgraph: SubGraph,
	) -> Self
	where
		I1: Into<Node>,
		I2: Into<Node>,
		T: IntoIterator<Item = I2>,
	{
		let loss = loss.into();
		let mut count = 0;
		let inputs: IndexSet<Node> = inputs.into_iter().map(Into::into).inspect(|_| count += 1).collect();
		assert_eq!(inputs.len(), count, "inputs contained duplicates");

		assert!(
			subgraph.nodes.contains(&loss),
			"Loss {} must be contained by the subgraph, but it is not.",
			loss
		);
		for (param, param_grad) in parameters_and_grads.iter() {
			assert!(
				subgraph.nodes.contains(param),
				"Parameters must be contained by the subgraph. Parameter {} is not.",
				param
			);
			assert!(
				!inputs.contains(param),
				"Inputs cannot overlap with parameters. Parameter {} is also an input.",
				param
			);
			assert!(
				param.shape().is_known(),
				"Parameter shapes must be fully known. Parameter {} has shape {}",
				param,
				param.shape()
			);
			assert!(
				param.shape() == param_grad.shape(),
				"Parameter shapes must equal grad shapes. Parameter {} has shape {}, grad {} has shape {}",
				param,
				param.shape(),
				param_grad,
				param_grad.shape()
			);
		}

		GradientOptimiser {
			inner: OptInner {
				loss,
				inputs,
				parameters_and_grads,
				subgraph,
			},
			callbacks: vec![],
			perf_records: None,
			grad_step,
			calc_change: true,
			calc_loss: true,
		}
	}

	/// Should the value of the loss node be calculated in each step.
	///
	/// Default: true
	pub fn calc_loss(&mut self, calc_loss: bool) -> &mut Self {
		self.calc_loss = calc_loss;
		self
	}

	/// Should the L2 norm of the change to the parameters nodes be calculated each step.
	///
	/// This introduces a bit of overhead to each step.
	///
	/// Default: true
	pub fn calc_change(&mut self, calc_change: bool) -> &mut Self {
		self.calc_change = calc_change;
		self
	}

	/// If Some the OpPerf for each Op will be updated
	///
	/// Default: None
	pub fn perf_records(&mut self, perf_records: Option<&'a mut IndexMap<Op, OpPerf>>) -> &mut Self {
		self.perf_records = perf_records;
		self
	}
}

impl<'a, S> GradientOptimiser<'a, S>
where
	S: GradientStepper,
{
	pub fn inner(&self) -> &OptInner {
		&self.inner
	}

	pub fn inner_mut(&mut self) -> &mut OptInner {
		&mut self.inner
	}

	pub fn into_inner(self) -> OptInner {
		self.inner
	}

	pub fn boxed_callback(&mut self, func: Box<dyn 'a + FnMut(&mut S, &StepData) -> Signal>) {
		self.callbacks.push(func);
	}

	/// Complete one optimisation step then run each callback.
	///
	/// If any callback returns `Signal::Stop` then `Stop` is returned here.
	fn step_with_callbacks_impl<'b, I, T1>(
		inner: &'b OptInner,
		grad_stepper: &mut S,
		callbacks: &mut [Box<dyn 'a + FnMut(&mut S, &StepData) -> Signal>],
		perf_records: Option<&'b mut IndexMap<Op, OpPerf>>,
		inputs: T1,
		calc_loss: bool,
		calc_change: bool,
	) -> Result<(StepData<'b>, Signal), ExecError>
	where
		I: Into<Node>,
		T1: IntoIterator<Item = (I, ArcArray<f32, IxDyn>)>,
	{
		let mut signal = Signal::Continue;

		let (mut results, loss) = if calc_loss {
			let mut results = ExecutionPlan::new(inputs, inner.parameters_and_grads.values().chain(once(&inner.loss)))
				.perf_records(perf_records)
				.subgraph(Some(&inner.subgraph))
				.execute()?;
			let loss = results.swap_remove(&inner.loss).unwrap().sum();
			(results, loss)
		} else {
			let results = ExecutionPlan::new(inputs, inner.parameters_and_grads.values())
				.perf_records(perf_records)
				.subgraph(Some(&inner.subgraph))
				.execute()?;
			(results, 0.0)
		};

		let params_and_grad_arrs = inner
			.parameters_and_grads
			.iter()
			.filter_map(|(p, g)| results.swap_remove(g).map(|ga| (p.clone(), ga)))
			.collect();

		let change_norm = grad_stepper.step(params_and_grad_arrs, calc_change)?;

		let data = StepData {
			loss,
			step: grad_stepper.step_count(),
			change_norm,
			opt_inner: inner,
		};

		{
			for func in callbacks.iter_mut() {
				if let Signal::Stop = func(grad_stepper, &data) {
					signal = Signal::Stop;
				}
			}
		}

		Ok((data, signal))
	}

	pub fn callback<I: Into<Signal>, F: 'a + FnMut(&mut S, &StepData) -> I>(&mut self, mut func: F) {
		self.boxed_callback(Box::new(move |s, d| func(s, d).into()));
	}

	/// The loss being optimised.
	pub fn loss(&self) -> &Node {
		&self.inner().loss
	}

	/// This is the list of inputs to the subgraph which will be fed from the supplied `DataStream`.
	pub fn inputs(&self) -> &IndexSet<Node> {
		&self.inner().inputs
	}

	/// This is the list of value `NodeID`s which correspond to nodes marked `Parameter`,
	/// excluding `NodeID`s which overlap with DataIDs included in `inputs()`
	pub fn parameters_and_grads(&self) -> &IndexMap<Node, Node> {
		&self.inner().parameters_and_grads
	}

	pub fn subgraph(&self) -> &SubGraph {
		&self.inner().subgraph
	}

	pub fn callbacks(&mut self) -> &mut [Box<dyn 'a + FnMut(&mut S, &StepData) -> Signal>] {
		&mut self.callbacks
	}

	/// Continually call `step_with_callbacks(..)` with while drawing inputs from the `DataStream` until a
	/// `Signal::Stop` is generated.
	pub fn optimise<D: DataStream>(&mut self, data_stream: &mut D) -> Result<(), ExecError> {
		let mut perf_records = self.perf_records.take();
		let opt_result = loop {
			let step_result = Self::step_with_callbacks_impl(
				&self.inner,
				&mut self.grad_step,
				&mut self.callbacks,
				perf_records.as_deref_mut(),
				//self.perf_records.as_mut().map(|i| *i).unwrap_or(&mut x),
				self.inner.inputs.iter().zip(data_stream.next()),
				self.calc_loss,
				self.calc_change,
			);

			match step_result {
				Err(e) => break Err(e),
				Ok((_, Signal::Stop)) => break Ok(()),
				_ => {}
			}
		};

		self.perf_records = perf_records;

		opt_result
	}

	pub fn finalise(&mut self) {
		self.grad_step.finalise(&mut self.inner.parameters_and_grads.keys());
	}
}

/// print step number, loss, and L2 norm of variable change
pub fn print_step_data<S: GradientStepper>(batch_size: f32) -> impl FnMut(&mut S, &StepData) -> Signal {
	move |_step, data| {
		println!(
			"step:{}\tloss:{}\tchange:{}",
			data.step,
			data.loss / batch_size,
			data.change_norm
		);
		Signal::Continue
	}
}

/// Produce an optimiser callback which will stop optimisation once a maximum number of steps is reached.
pub fn max_steps<S: GradientStepper>(max: usize) -> impl FnMut(&mut S, &StepData) -> Signal {
	move |_step, data| {
		if data.step < max {
			Signal::Continue
		} else {
			Signal::Stop
		}
	}
}

/// Produce an optimiser callback which will stop optimisation once a minimum loss threshold is reached.
pub fn min_err<S: GradientStepper>(min: f32) -> impl FnMut(&mut S, &StepData) -> Signal {
	move |_step, data| {
		if data.loss > min {
			Signal::Continue
		} else {
			Signal::Stop
		}
	}
}

/// Wrap an optimiser callback in a new callback which will only execute on every n steps.
pub fn every_n_steps<'a, S: GradientStepper, I: Into<Signal>, F: 'a + FnMut(&mut S, &StepData) -> I>(
	n: usize,
	mut func: F,
) -> impl 'a + FnMut(&mut S, &StepData) -> Signal {
	move |step, data| {
		if data.step % n == 0 {
			func(step, data).into()
		} else {
			Signal::Continue
		}
	}
}

/// Wrap an optimiser callback in a new callback which will only execute on the nth step.
pub fn nth_step<'a, S: GradientStepper, I: Into<Signal>, F: 'a + FnMut(&mut S, &StepData) -> I>(
	n: usize,
	mut func: F,
) -> impl 'a + FnMut(&mut S, &StepData) -> Signal {
	move |step, data| {
		if data.step == n {
			func(step, data).into()
		} else {
			Signal::Continue
		}
	}
}
