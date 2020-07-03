use alumina_core::{
	exec::{exec, ExecConfig},
	grad::grad,
	graph::{Node, NodeTag},
	util::display::IterDisplay,
};
use indexmap::{indexmap, indexset, IndexMap, IndexSet};
use ndarray::{ArrayD, Dimension, Zip, ArcArray, IxDyn};
use rand::{seq::SliceRandom, thread_rng, Rng};
use rand_distr::{Distribution, Normal};

/// A builder for `numeric_test_iters()` with sane defaults.
#[derive(Clone)]
#[must_use]
pub struct GradNumericTest {
	loss: Node,
	inputs: IndexSet<Node>,
	step_size: f32,
	variance: f32,
	params_as_inputs: bool,
	isolate_inputs: bool,
	iters: usize,
	failures: usize,
	tolerance: f32,
	// rel_tolerance: bool,
	expect_zero: IndexMap<Node, f32>,
}

impl GradNumericTest {
	pub fn new<L, I, T>(loss: L, inputs: T) -> GradNumericTest
	where
		L: Into<Node>,
		I: Into<Node>,
		T: IntoIterator<Item = I>,
	{
		let loss = loss.into();
		let inputs = inputs.into_iter().map(Into::into).collect();

		GradNumericTest {
			loss,
			inputs,
			step_size: 1e-2,
			variance: 1.0,
			params_as_inputs: true,
			isolate_inputs: true,
			iters: 100,
			failures: 1,
			tolerance: 1e-4,
			// rel_tolerance: true,
			expect_zero: indexmap![],
		}
	}

	/// The size of the step in each direction of the gradient, taken for central difference gradient approximation.
	///
	/// Default: 1e-2
	pub fn step_size(mut self, step_size: f32) -> Self {
		self.step_size = step_size;
		self
	}

	/// The default variance used to initialise any input nodes that do not have an initialiser.
	///
	/// Default: 1.0
	pub fn variance(mut self, variance: f32) -> Self {
		self.variance = variance;
		self
	}

	/// Determines whether nodes tagged as parameters should also be treated as inputs in the testing.
	///
	/// Only applies to parameters in the graph of `loss`.
	///
	/// Default: true
	pub fn params_as_inputs(mut self, params_as_inputs: bool) -> Self {
		self.params_as_inputs = params_as_inputs;
		self
	}

	/// Determines whether (when there are more than one inputs), each input will be tested individually, in addition
	/// to the standard joint test.
	///
	/// Default: true
	pub fn isolate_inputs(mut self, isolate_inputs: bool) -> Self {
		self.isolate_inputs = isolate_inputs;
		self
	}

	/// How many times the test proceedure is repeated.
	///
	/// Default: 100
	pub fn iters(mut self, iters: usize) -> Self {
		self.iters = iters;
		self
	}

	/// How many times the test proceedure is allowed to fail, without raising a panic.
	///
	/// Default: 1
	pub fn failures(mut self, failures: usize) -> Self {
		self.failures = failures;
		self
	}

	/// Error of the gradient allowed before treating an iteration as a failure.
	///
	/// By default this is relative.
	///
	/// Default: 1e-4
	pub fn tolerance(mut self, tolerance: f32) -> Self {
		self.tolerance = tolerance;
		self
	}

	// /// Switch between relative error and error when checking tolerance.
	// ///
	// /// Default: true
	// pub fn rel_tolerance(mut self, rel_tolerance: bool) -> Self {
	// 	self.rel_tolerance = rel_tolerance;
	// 	self
	// }

	/// For a specific node, instead of testing that the numerical grad matched the computed gradient, instead check
	/// that both are within a tolerance of zero.
	pub fn expect_zero<N: Into<Node>>(mut self, node: N, tolerance: f32) -> Self
	where
		N: Into<Node>,
	{
		self.expect_zero.insert(node.into(), tolerance);
		self
	}

	pub fn run(self) {
		grad_numeric_test_iters(&self)
	}
}

fn random_permutation<R: Rng>(len: usize, rng: &mut R) -> (Vec<usize>, Vec<usize>) {
	let mut a: Vec<usize> = (0..len).collect();
	a.shuffle(rng);
	let mut b = vec![0; len];
	for i in 0..len {
		b[a[i]] = i;
	}
	(a, b)
}

/// The number of failures must be less than or equal to failures.
pub fn grad_numeric_test_iters(config: &GradNumericTest) {
	let mut failure_count = 0;
	let mut failure_errs = vec![];
	let mut failure_nodes = vec![];
	let mut ok_errs = vec![];

	for _ in 0..config.iters {
		let (rel_input_err, rel_nodes) = grad_numeric_test(config);

		if rel_input_err > config.tolerance || rel_input_err.is_nan() {
			failure_count += 1;
			failure_errs.push(rel_input_err);
			failure_nodes.push(IterDisplay { inner: rel_nodes });
		} else {
			ok_errs.push(rel_input_err);
		};
	}

	assert!(
		failure_count <= config.failures,
		"failures: {}/{}\nok values:{:?}\nfailure values:{:?}\nfailure node sets: {}",
		failure_count,
		config.iters,
		ok_errs,
		failure_errs,
		IterDisplay { inner: failure_nodes }
	);
}

// TODO second test that takes two points and calculates the dot of the difference between them and the average
// gradient and checks against expected loss.

/// Numerical test of gradients.
///
/// Calculates the (error, relative error) of the gradient buy numerical means for the simultaneous combination of all
/// inputs, and optionally repeating the test over each input individually returning the worst error
///
/// # Panics
/// Panics if input nodes aren't of fixed shape
pub fn grad_numeric_test(config: &GradNumericTest) -> (f32, IndexSet<Node>) {
	// Add parameters to inputs
	let mut inputs: IndexSet<Node> = config.inputs.clone();

	if config.params_as_inputs {
		inputs.extend(config.loss.graph().nodes_tagged(NodeTag::Parameter));
	}

	// instantiate inputs and params
	let input_values: IndexMap<Node, ArcArray<f32, IxDyn>> = inputs
		.iter()
		.map(|node| {
			let rng = &mut thread_rng();
			let norm =
				Normal::new(0.0, f64::from(config.variance.sqrt())).expect("Could not create normal distribution");

			let val = node.init_array().unwrap_or_else(|| {
				let shape = node
					.shape()
					.to_data_shape()
					.expect("All inputs to numeric test must have a fixed shape (all axes must be known).");

				// Permute the data representation to test Ops that rely on standard ordering
				let (permute, inv_permute) = random_permutation(shape.ndim(), rng);
				let shape_permute: Vec<usize> = permute.iter().map(|&i| shape[i]).collect();
				let mut arr = ArrayD::zeros(shape_permute);

				for x in &mut arr {
					*x = norm.sample(rng) as f32
				}

				arr.permuted_axes(inv_permute)
			});

			(node.clone(), val.to_shared())
		})
		.collect();

	let grads = grad(&config.loss, inputs.iter()).expect("Call to grad() failed in numeric test.");

	let mut rel_worst = 0.0f32;
	let mut rel_worst_input = indexset![];

	if config.isolate_inputs && inputs.len() > 1 {
		// loop over inputs and test gradient of each individually
		for i in &inputs {
			let (diff, expected_diff) = grad_numeric_test_inner(
				&config.loss,
				&input_values,
				indexset![i.clone()],
				&grads,
				config.step_size,
			);
			let error = expected_diff - diff;
			let rel_error = (error.abs() / diff.abs().max(expected_diff.abs())) as f32;
			if let Some(&tolerance) = config.expect_zero.get(i) {
				if expected_diff > f64::from(tolerance) {
					panic!(
						"{} Grad test failed as grad*step is greater than tolerance for expect_zero input: {} > {}",
						i,
						expected_diff,
						f64::from(tolerance)
					)
				} else if diff > f64::from(tolerance) {
					panic!("{} Grad test failed as numerical difference in loss is greater than tolerance for expect_zero input: {} > {}", i, diff, f64::from(tolerance))
				}
			} else if expected_diff < f64::from(::std::f32::EPSILON) {
				panic!(
					"{} Grad test failed as grad*step was near zero and expect_zero was not set: {} < {}",
					i,
					expected_diff,
					f64::from(::std::f32::EPSILON)
				)
			} else if rel_error > rel_worst {
				rel_worst = rel_error;
				rel_worst_input = indexset![i.clone()];
			}
		}
	}

	// grad test all simultaneously
	let (diff, expected_diff) =
		grad_numeric_test_inner(&config.loss, &input_values, inputs.clone(), &grads, config.step_size);
	let error = expected_diff - diff;
	let rel_error = (error.abs() / diff.abs().max(expected_diff.abs())) as f32;
	if let Some(tolerance) = inputs.iter().fold(Some(0.0f32), |max, i| {
		max.and_then(|max| config.expect_zero.get(i).map(|x| x.max(max)))
	}) {
		if expected_diff > f64::from(tolerance) {
			panic!("Grad test failed as expected difference in loss is greater than tolerance for expect_zero inputs: {} > {}", expected_diff, f64::from(tolerance))
		} else if diff > f64::from(tolerance) {
			panic!(
				"Grad test failed as difference in loss is greater than tolerance for expect_zero inputs: {} > {}",
				diff,
				f64::from(tolerance)
			)
		}
	} else if expected_diff < f64::from(::std::f32::EPSILON) {
		panic!(
			"Grad test failed as expected difference was near zero and expect_zero was not set: {} < {}",
			expected_diff,
			f64::from(::std::f32::EPSILON)
		)
	} else if rel_error > rel_worst {
		rel_worst = rel_error;
		rel_worst_input = inputs;
	}

	(rel_worst, rel_worst_input)
}

/// Returns the (diff, expected_diff) when testing the combination of all tested_grads
fn grad_numeric_test_inner(
	loss: &Node,
	input_values: &IndexMap<Node, ArcArray<f32, IxDyn>>,
	tested_inputs: IndexSet<Node>,
	grads: &IndexMap<Node, Node>,
	step_size: f32,
) -> (f64, f64) {
	// first call with grads and y as outputs
	let outputs = tested_inputs.iter().map(|n| &grads[n]).chain(::std::iter::once(loss));
	let results = exec(input_values.clone(), outputs, &mut ExecConfig::default())
		.unwrap_or_else(|err| panic!("Call to exec() failed in numeric test.\n{:#?}", err));

	// adjust in each direction
	let grad_dot: f64 = tested_inputs
		.iter()
		.map(|tested_input| {
			results[&grads[tested_input]]
				.iter()
				.fold(0.0f64, |acc, &d| acc + f64::from(d) * f64::from(d))
		})
		.sum();

	let scale = step_size / (grad_dot.sqrt() as f32);

	// first call with just y output
	let input1_values = input_values.iter().map(|(node, value)| {
		if tested_inputs.contains(node) {
			unsafe {
				let mut new = ArrayD::<f32>::uninitialized(value.shape());
				Zip::from(&mut new)
					.and(value)
					.and(&results[&grads[node]])
					.apply(|new, &value, &grad| {
						*new = value - scale * grad;
					});
				(node.clone(), new.to_shared())
			}
		} else {
			(node.clone(), value.clone())
		}
	});

	// second call with just y output TODO consider pre-extracting a subgraph
	let input2_values = input_values.iter().map(|(node, value)| {
		if tested_inputs.contains(node) {
			unsafe {
				let mut new = ArrayD::<f32>::uninitialized(value.shape());
				Zip::from(&mut new)
					.and(value)
					.and(&results[&grads[node]])
					.apply(|new, &value, &grad| {
						*new = value + scale * grad;
					});
				(node.clone(), new.to_shared())
			}
		} else {
			(node.clone(), value.clone())
		}
	});

	// let loss1 = exec(input1_values, indexset![loss], true)
	// 	.unwrap_or_else(|err| panic!("Call to exec() failed in numeric test.\n{:#?}", err))[loss]
	// 	.iter()
	// 	.fold(0.0f64, |acc, &val| acc + f64::from(val));
	// let loss2 = exec(input2_values, indexset![loss], true)
	// 	.unwrap_or_else(|err| panic!("Call to exec() failed in numeric test.\n{:#?}", err))[loss]
	// 	.iter()
	// 	.fold(0.0f64, |acc, &val| acc + f64::from(val));
	// let diff = loss2 - loss1;

	let exec_vals1 = exec(input1_values, indexset![loss], &mut ExecConfig::default())
		.unwrap_or_else(|err| panic!("Call to exec() failed in numeric test.\n{:#?}", err));

	let exec_vals2 = exec(input2_values, indexset![loss], &mut ExecConfig::default())
		.unwrap_or_else(|err| panic!("Call to exec() failed in numeric test.\n{:#?}", err));

	assert_eq!(
		exec_vals1[loss].shape(),
		exec_vals2[loss].shape(),
		"loss should not change shape: {:?} {:?}",
		exec_vals1[loss].shape(),
		exec_vals2[loss].shape()
	);
	let diff = exec_vals1[loss]
		.iter()
		.zip(exec_vals2[loss].iter())
		.fold(0.0f64, |acc, (&val1, &val2)| acc + (f64::from(val2) - f64::from(val1)));

	let expected_diff = 2.0 * grad_dot.sqrt() * f64::from(step_size);

	// let err = expected_diff - diff;

	//(err as f32, (err.abs() / diff.abs().max(expected_diff.abs())) as f32)
	(diff, expected_diff)
}

// TODO test grad_numeric_test when input and loss arnt connected
