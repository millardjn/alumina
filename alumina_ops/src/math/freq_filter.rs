use alumina_core::{
	base_ops::{OpInstance, OpSpecification},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{Graph, Node, NodeID},
	shape::NodeShape,
	shape_prop::ShapePropContext,
};
use indexmap::{indexset, IndexMap, IndexSet};
use ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};
use ndarray::{ArrayD, ArrayViewD, Dimension};
use ndarray::{Axis, Zip};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{FftDirection, FftPlanner};
use smallvec::SmallVec;
use std::{any::Any, fmt::Debug, sync::Arc};

/// Apply a multiplicative filter in the freqency domain
///
/// The supplied closure must accept frequency bin indexes from -N/2 to (N-1)/2 for each axis
pub fn freq_filter<I, F>(input: I, axes: &[usize], f: F) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
	F: Fn(&[isize]) -> f32 + 'static + Sync + Send,
{
	let input = input.into();

	let output_shape: NodeShape = input.shape();

	let output = input
		.graph()
		.new_node(output_shape)
		.set_name_unique(&format!("fft({})", input));

	FreqFilter::new(input, output.clone(), Filter::new(axes, f)).build()?;

	Ok(output)
}

#[derive(Clone)]
pub struct Filter {
	axes: Vec<usize>,
	#[allow(clippy::type_complexity)]
	f: Arc<dyn Fn(ArrayViewD<f32>, &[usize]) -> ArrayD<f32> + 'static + Sync + Send>,
}

impl Debug for Filter {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("Filter")
			.field("axes", &self.axes)
			.field("f", &"processing function")
			.finish()
	}
}

impl Filter {
	pub fn new<F: Fn(&[isize]) -> f32 + 'static + Sync + Send>(axes: &[usize], f: F) -> Self {
		Self {
			axes: axes.to_vec(),
			f: Arc::new(move |array: ArrayViewD<f32>, axes: &[usize]| {
				let mut planner = FftPlanner::new();

				let mut c_array: ArrayD<Complex<f32>> = array.map(|&v| Complex::new(v, 0.0));

				// Apply ND-FFT
				for &axis in axes {
					let fft = planner.plan_fft(c_array.shape()[axis], FftDirection::Forward);

					Zip::from(c_array.lanes_mut(Axis(axis))).into_par_iter().for_each_init(
						|| {
							(
								vec![Zero::zero(); fft.len()],
								vec![Zero::zero(); fft.get_inplace_scratch_len()],
							)
						},
						|(temp, scratch), mut col| {
							//((_, _), ArrayViewMut1<f32>)
							debug_assert_eq!(col.0.len(), temp.len());
							unsafe {
								if let Some(slice) = col.0.as_slice_mut() {
									fft.process_with_scratch(slice, scratch);
								} else {
									for (i, col) in col.0.iter_mut().enumerate() {
										*temp.get_unchecked_mut(i) = *col;
									}
									fft.process_with_scratch(temp, scratch);

									for (i, col) in col.0.iter_mut().enumerate() {
										*col = *temp.get_unchecked(i);
									}
								}
							}
						},
					);
				}

				// Apply filter
				assert!(axes.iter().all(|&a| a < c_array.shape().len()));
				let shape = c_array.shape().to_vec();
				c_array.indexed_iter_mut().for_each(|(i, v)| {
					let i: SmallVec<[isize; 8]> = axes
						.iter()
						.map(|&a| {
							let s = shape[a] as isize;
							(unsafe { *i.slice().get_unchecked(a) } as isize + s / 2) % s - s / 2
						})
						.collect();
					*v = v.scale(f(&i));
				});

				// Apply ND-IFFT and normalise
				for &axis in axes {
					let fft = planner.plan_fft(c_array.shape()[axis], FftDirection::Inverse);

					let normalisation = 1.0 / c_array.shape()[axis] as f32; // normalisation for both forward and reverse

					Zip::from(c_array.lanes_mut(Axis(axis))).into_par_iter().for_each_init(
						|| {
							(
								vec![Zero::zero(); fft.len()],
								vec![Zero::zero(); fft.get_inplace_scratch_len()],
							)
						},
						|(temp, scratch), mut col| {
							debug_assert_eq!(col.0.len(), temp.len());
							unsafe {
								if let Some(slice) = col.0.as_slice_mut() {
									fft.process_with_scratch(slice, scratch);
									for col in col.0.iter_mut() {
										*col = col.scale(normalisation);
									}
								} else {
									for (i, col) in col.0.iter_mut().enumerate() {
										*temp.get_unchecked_mut(i) = *col;
									}
									fft.process_with_scratch(temp, scratch);

									for (i, col) in col.0.iter_mut().enumerate() {
										*col = temp.get_unchecked(i).scale(normalisation);
									}
								}
							}
						},
					);
				}

				c_array.map(|c| c.re)
			}),
		}
	}

	pub fn apply(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
		(self.f)(input, &self.axes)
	}
}

/// Average Pooling operation
///
/// Decrease size of dimensions by given factors.
/// Output values are the average of windows of the input with the size of factors
#[must_use]
#[derive(Clone, Debug)]
pub struct FreqFilter {
	input: Node,
	output: Node,
	filter: Filter,
}

impl FreqFilter {
	pub fn new<I, O>(input: I, output: O, filter: Filter) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		let input = input.into();
		let output = output.into();
		FreqFilter { input, output, filter }
	}
}

impl OpSpecification for FreqFilter {
	type InstanceType = FreqFilterInstance;

	fn type_name(&self) -> &'static str {
		"FreqFilter"
	}

	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input.clone()]
	}

	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			input: mapping.get(&self.input).unwrap_or(&self.input).clone(),
			output: mapping.get(&self.output).unwrap_or(&self.output).clone(),
			filter: self.filter.clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		for &axis in &self.filter.axes {
			if axis >= self.input.shape().len() {
				return Err(format!(
					"Fft axis ({}) out of bounds for input shape size ({})",
					axis,
					self.input.shape().len(),
				)
				.into());
			}
		}

		if self.input.shape().len() != self.output.shape().len() {
			return Err(format!(
				"FreqFilter input shape ({}) must be the same length as Fft output shape ({})",
				self.input.shape().len(),
				self.output.shape().len()
			)
			.into());
		}

		Ok(FreqFilterInstance {
			input: self.input.id(),
			output: self.output.id(),
			filter: self.filter,
		})
	}
}

#[derive(Debug, Clone)]
pub struct FreqFilterInstance {
	input: NodeID,
	output: NodeID,
	filter: Filter,
}

impl OpInstance for FreqFilterInstance {
	fn type_name(&self) -> &'static str {
		"FreqFilter"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(FreqFilter {
			input: graph.node_from_id(self.input),
			output: graph.node_from_id(self.output),
			filter: self.filter.clone(),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		FreqFilter::new(ctx.grad_of(&self.output), ctx.grad_of(&self.input), self.filter.clone()).build()?;
		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		ctx.merge_output_shape(&self.output, &ctx.input_shape(&self.input).slice().into())?;
		Ok(())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let input = ctx.get_input_standard(&self.input);

		let out = self.filter.apply(input);

		if ctx.can_set(&self.output) {
			ctx.set(&self.output, out.to_shared())
		} else {
			let mut output = ctx.get_output_standard(&self.output);
			output += &out;
		}

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use crate::math::freq_filter::{Filter, FreqFilter};

	use super::freq_filter;
	use alumina_core::{base_ops::OpSpecification, graph::Node};
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::arr2;

	#[test]
	#[allow(clippy::excessive_precision)]
	fn forward_test() {
		let input = Node::new(&[2, 9])
			.set_value(arr2(&[
				[0.2, 0.4, 0.6, 0.8, 2.2, 2.4, 2.6, 2.8, 4.7],
				[1.2, 1.4, 1.6, 1.8, 3.2, 3.4, 3.6, 3.8, 3.2],
			]))
			.set_name("input");

		let output = freq_filter(input, &[0, 1], |index| if index == [0, 0] { 1.0 } else { 0.5 }).unwrap();

		assert!(output.calc().unwrap().all_relatively_close(
			&arr2(&[
				[
					1.208333333,
					1.308333333,
					1.408333333,
					1.508333333,
					2.208333333,
					2.308333333,
					2.408333333,
					2.508333333,
					3.458333333
				],
				[
					1.708333333,
					1.808333333,
					1.908333333,
					2.008333333,
					2.708333333,
					2.808333333,
					2.908333333,
					3.008333333,
					2.708333333
				],
			]),
			1e-5
		));
	}

	#[test]
	#[allow(clippy::excessive_precision)]
	fn forward_single_axis_test() {
		let input = Node::new(&[2, 9])
			.set_value(arr2(&[
				[0.2, 0.4, 0.6, 0.8, 2.2, 2.4, 2.6, 2.8, 4.7],
				[1.2, 1.4, 1.6, 1.8, 3.2, 3.4, 3.6, 3.8, 3.2],
			]))
			.set_name("input");

		let output = freq_filter(input, &[1], |index| if index == [0] { 1.0 } else { 0.5 }).unwrap();

		assert!(output.calc().unwrap().all_relatively_close(
			&arr2(&[
				[
					1.027777778,
					1.127777778,
					1.227777778,
					1.327777778,
					2.027777778,
					2.127777778,
					2.227777778,
					2.327777778,
					3.277777778,
				],
				[
					1.888888889,
					1.988888889,
					2.088888889,
					2.188888889,
					2.888888889,
					2.988888889,
					3.088888889,
					3.188888889,
					2.888888889,
				],
			]),
			1e-5
		));
	}

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[13, 43]).set_name("input");

		let output = freq_filter(&input, &[0, 1], |index| if index == [0, 0] { 1.0 } else { 0.5 }).unwrap();

		GradNumericTest::new(&output, &indexset![&input]).run();
	}

	#[test]
	fn grad_numeric_brown_noise_test() {
		let input = Node::new(&[13, 43]).set_name("input");
		let output = Node::new(&[13, 43]).set_name("output");

		FreqFilter::new(
			&input,
			&output,
			Filter::new(&[0, 1], |index| {
				if index == [0, 0] {
					1.0
				} else {
					1.0 / (index[0] * index[0] + index[1] * index[1]) as f32
				}
			}),
		)
		.build()
		.unwrap();

		GradNumericTest::new(&output, &indexset![&input]).tolerance(1e-5).run();
	}
}
