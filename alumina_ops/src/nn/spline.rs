use alumina_core::{
	base_ops::{OpInstance, OpSpecification},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{merge_graphs, Graph, Node, NodeID, NodeTag, Op},
	init::Initialiser,
	shape::{NodeAxis, NodeShape},
	shape_prop::ShapePropContext,
};
use indexmap::{indexset, IndexMap, IndexSet};
use ndarray::{ArrayViewMutD, Dimension, Zip};
use std::any::Any;
use std::iter::once;

fn _custom(name: &'static str, left_slope: f32, centre_slope: f32, right_slope: f32) -> Initialiser {
	Initialiser::new(name.to_string(), move |mut arr: ArrayViewMutD<f32>| {
		if arr.shape()[0] == 3 {
			let mut weights_iter = arr.outer_iter_mut();
			weights_iter.next().unwrap().fill(left_slope);
			weights_iter.next().unwrap().fill(centre_slope);
			weights_iter.next().unwrap().fill(right_slope);
		} else {
			eprintln!(
				"{} could not be executed because outermost dimension was not equal to 3",
				name
			);
		}
	})
}

pub fn custom(left_slope: f32, centre_slope: f32, right_slope: f32) -> Initialiser {
	_custom(
		"Custom Initialiser for Spline Op",
		left_slope,
		centre_slope,
		right_slope,
	)
}

pub fn elu_esque() -> Initialiser {
	_custom("ELU-esque Initialiser for Spline Op", 0.01, 1.0, 1.0)
}

pub fn tanh_esque() -> Initialiser {
	_custom("Tanh-esque Initialiser for Spline Op", 0.01, 1.0, 0.01)
}

pub fn parabola_esque() -> Initialiser {
	_custom("Parabola-esque Initialiser for Spline Op", -1.0, 0.0, 1.0)
}

pub fn swan() -> Initialiser {
	_custom("Swan Initialiser for Spline Op", 0.01, 1.0, 0.25)
}

/// A parameterised activation function that is smooth and continuous, consisting of linear components jointed by a
/// central cubic spline region.
///
/// Defined as a cubic function in the domain (-1, 1) which passes through 0,0. Three learnable parameters control the
/// gradients at x=-1, x=0 and x=1. Linear extensions are used outside the central region.
///
///
/// Supply axes which learnable weights should be shared over.
///
/// By default all axes with Known size are assigned unique weights, and sharing via broadcasting is used for non-Known
/// axes. Setting an axis as shared will prevent unique weights being used, and enforce sharing, even if the size is
/// Known. Each element of `axes` can be in the range [-input.ndims(), input.ndims()).
///
/// Default: empty
pub fn spline<I>(input: I, axes: &[isize], init: Initialiser) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();

	let mut weights_shape: NodeShape = once(3)
		.chain(input.shape().slice().iter().map(|axis| match axis {
			NodeAxis::Known { val } => *val,
			NodeAxis::Interval { .. } => 1,
		}))
		.into();

	if !axes.is_empty() {
		for axis in axes {
			let axis = (axis + input.shape().len() as isize) as usize % input.shape().len();
			weights_shape.slice_mut()[axis + 1] = 1.into();
		}
	}

	let weights = input
		.graph()
		.new_node(weights_shape)
		.set_init(init)
		.add_tag(NodeTag::Parameter)
		.set_name_unique(&format!("spline({})_weights", input));

	let output = input
		.graph()
		.new_node(input.shape())
		.set_name_unique(&format!("spline({})", input));

	let _op = Spline::new(input, weights, output.clone()).build()?;

	Ok(output)
}

/// A parameterised activation function that is smooth and continuous, consisting of linear components jointed by a
/// central cubic spline region.
pub fn spline_with<I, W>(input: I, weights: W) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
	W: Into<Node>,
{
	let input = input.into();
	let weights = weights.into();
	merge_graphs(&[input.graph(), weights.graph()]);

	let output = input
		.graph()
		.new_node(input.shape())
		.set_name_unique(&format!("spline({})", input));

	let _op = Spline::new(input, weights, output.clone()).build()?;

	Ok(output)
}

/// A parameterised activation function that is smooth and continuous, consisting of linear components jointed by a
/// central cubic spline region.
pub fn spline_into<I, O>(input: I, axes: &[isize], init: Initialiser, output: O) -> Result<Op, OpBuildError>
where
	I: Into<Node>,
	O: Into<Node>,
{
	let input = input.into();
	let output = output.into();
	merge_graphs(&[input.graph(), output.graph()]);

	let mut weights_shape: NodeShape = once(3)
		.chain(input.shape().slice().iter().map(|axis| match axis {
			NodeAxis::Known { val } => *val,
			NodeAxis::Interval { .. } => 1,
		}))
		.into();

	if !axes.is_empty() {
		for axis in axes {
			let axis = (axis + input.shape().len() as isize) as usize % input.shape().len();
			weights_shape.slice_mut()[axis + 1] = 1.into();
		}
	}

	let weights = input
		.graph()
		.new_node(weights_shape)
		.set_init(init)
		.add_tag(NodeTag::Parameter)
		.set_name_unique(&format!("spline({})_weights", input));

	let op = Spline::new(input, weights, output).build()?;

	Ok(op)
}

/// A parameterised activation function that is smooth and continuous, consisting of linear components jointed by a
/// central cubic spline region.
pub fn spline_with_into<I, W, O>(input: I, weights: W, output: O) -> Result<Op, OpBuildError>
where
	I: Into<Node>,
	W: Into<Node>,
	O: Into<Node>,
{
	let input = input.into();
	let weights = weights.into();
	let output = output.into();
	Spline::new(input, weights, output).build()
}

/// `Spline` A smooth continuous function consisting of linear components jointed by a central cubic region.
///
/// Defined as a cubic function in the domain (-1, 1) which passes through 0,0. Three learnable parameters control the
/// gradients at x=-1, x=0 and x=1. Linear extensions are used outside the central region.
#[must_use]
#[derive(Clone, Debug)]
pub struct Spline {
	input: Node,
	weights: Node,
	output: Node,
}

impl Spline {
	pub fn new<I, W, O>(input: I, weights: W, output: O) -> Self
	where
		I: Into<Node>,
		W: Into<Node>,
		O: Into<Node>,
	{
		let input = input.into();
		let weights = weights.into();
		let output = output.into();
		Spline { input, weights, output }
	}
}

impl OpSpecification for Spline {
	type InstanceType = SplineInstance;

	fn op_type(&self) -> &'static str {
		"Spline"
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input.clone(), self.weights.clone()]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			input: mapping.get(&self.input).unwrap_or(&self.input).clone(),
			weights: mapping.get(&self.weights).unwrap_or(&self.weights).clone(),
			output: mapping.get(&self.output).unwrap_or(&self.output).clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		if self
			.weights
			.shape()
			.slice()
			.get(0)
			.and_then(|axis| axis.merge(&NodeAxis::known(3)).ok())
			.is_none()
		{
			return Err(format!(
				"weights shape ({}) must have a length > 1 and a first (outermost) axis of size 3",
				self.weights.shape(),
			)
			.into());
		}

		if self.input.shape().merge(&self.output.shape()).is_err() {
			return Err(format!(
				"It must be possible to merge input shape ({}) with output shape ({}) to ensure they can be set to be equal during shape propagation, i.e. at op construction they must have the same length and overlapping axis size ranges",
				self.output.shape(),
				self.input.shape(),
			)
			.into());
		}

		Ok(SplineInstance {
			input: self.input.id(),
			weights: self.weights.id(),
			output: self.output.id(),
		})
	}
}

#[derive(Debug, Clone)]
pub struct SplineInstance {
	input: NodeID,
	weights: NodeID,
	output: NodeID,
}

impl OpInstance for SplineInstance {
	fn op_type(&self) -> &'static str {
		"Spline"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(Spline {
			input: graph.node_from_id(self.input),
			weights: graph.node_from_id(self.weights),
			output: graph.node_from_id(self.output),
		})
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input, self.weights]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		SplineBack::new(
			ctx.node(&self.input),
			ctx.node(&self.weights),
			ctx.grad_of(&self.output),
			ctx.grad_of(&self.input),
			ctx.grad_of(&self.weights),
		)
		.build()?;
		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		ctx.merge_output_shape(&self.output, &ctx.input_shape(&self.input).slice().into())?;

		let weights_shape = ctx.input_shape(&self.weights).slice();
		if 3 != weights_shape[0] {
			return Err(format!(
				"weights shape ({}) must have a first (outermost) axis of size 3",
				ctx.node(&self.weights).shape(),
			)
			.into());
		}

		let output_shape = &weights_shape[1..].into();

		ctx.broadcast_merge_output_shape(&self.output, output_shape)
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let input = ctx.get_input(&self.input);
		let weights = ctx.get_input(&self.weights);
		let output = ctx.get_output(&self.output);

		let input_shape = input.shape();
		let output_shape = output.shape();
		let weights_outer_shape = weights.shape();
		let weights_shape = &weights_outer_shape[1..];

		let mut weights_iter = weights.outer_iter();
		let weights0 = weights_iter.next().unwrap();
		let weights1 = weights_iter.next().unwrap();
		let weights2 = weights_iter.next().unwrap();

		debug_assert_eq!(
			input_shape, output_shape,
			"input shape: {:?} did not match output shape: {:?}",
			input_shape, output_shape
		);
		debug_assert!(
			weights0.broadcast(output_shape).is_some(),
			"Could not broadcast weights_shape[1..]: {:?} to input/output shape: {:?}",
			weights_shape,
			input_shape
		);

		Zip::from(output)
			.and(&input)
			.and_broadcast(&weights0)
			.and_broadcast(&weights1)
			.and_broadcast(&weights2)
			.par_for_each(|output, &input, &left, &centre, &right| {
				let x = input;
				if x <= -1.0 {
					*output += (-2.0 / 3.0) * (centre - 1.5 * left * x - 0.875 * left - 0.125 * right);
				// linear segment to the left of x=-1
				} else if x >= 1.0 {
					*output += (2.0 / 3.0) * (centre + 1.5 * right * x - 0.125 * left - 0.875 * right);
				// linear segment to the right of x=1
				} else {
					let x2 = x * x;
					let x3 = x * x * x;
					*output += (-1.0 / 3.0)
						* (centre * x3 - 3.0 * centre * x - 0.5 * (left + right) * x3 + 0.75 * (left - right) * x2);
					// cubic spline passing through 0,0 connecting left and right
				}
			});

		Ok(())
	}
}

#[derive(Debug, Clone)]
pub struct SplineBack {
	input: Node,
	weights: Node,
	output_grad: Node,
	input_grad: Node,
	weights_grad: Node,
}

impl SplineBack {
	pub fn new<I, W, OG, IG, WG>(input: I, weights: W, output_grad: OG, input_grad: IG, weights_grad: WG) -> Self
	where
		I: Into<Node>,
		W: Into<Node>,
		OG: Into<Node>,
		IG: Into<Node>,
		WG: Into<Node>,
	{
		let input = input.into();
		let weights = weights.into();
		let output_grad = output_grad.into();
		let input_grad = input_grad.into();
		let weights_grad = weights_grad.into();

		SplineBack {
			input,
			weights,
			output_grad,
			input_grad,
			weights_grad,
		}
	}
}

impl OpSpecification for SplineBack {
	type InstanceType = SplineBackInstance;

	fn op_type(&self) -> &'static str {
		"SplineBack"
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input.clone(), self.weights.clone(), self.output_grad.clone()]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.input_grad.clone(), self.weights_grad.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			input: mapping.get(&self.input).unwrap_or(&self.input).clone(),
			weights: mapping.get(&self.weights).unwrap_or(&self.weights).clone(),
			output_grad: mapping.get(&self.output_grad).unwrap_or(&self.output_grad).clone(),
			input_grad: mapping.get(&self.input_grad).unwrap_or(&self.input_grad).clone(),
			weights_grad: mapping.get(&self.weights_grad).unwrap_or(&self.weights_grad).clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		if self
			.weights
			.shape()
			.slice()
			.get(0)
			.and_then(|axis| axis.merge(&NodeAxis::known(3)).ok())
			.is_none()
		{
			return Err(format!(
				"weights shape ({}) must have a length > 1 and a first (outermost) axis of size 3",
				self.weights.shape(),
			)
			.into());
		}

		if self.input.shape().merge(&self.output_grad.shape()).is_err() {
			return Err(format!(
				"It must be possible to merge input shape ({}) with output_grad shape ({}) to ensure they can be set to be equal during shape propagation, i.e. at op construction they must have the same length and overlapping axis size ranges",
				self.output_grad.shape(),
				self.input.shape(),
			)
			.into());
		}

		if self.input.shape().merge(&self.input_grad.shape()).is_err() {
			return Err(format!(
				"It must be possible to merge input shape ({}) with input_grad shape ({}) to ensure they can be set to be equal during shape propagation, i.e. at op construction they must have the same length and overlapping axis size ranges",
				self.input_grad.shape(),
				self.input.shape(),
			)
			.into());
		}

		if self.weights.shape().merge(&self.weights_grad.shape()).is_err() {
			return Err(format!(
				"It must be possible to merge input shape ({}) with weights_grad shape ({}) to ensure they can be set to be equal during shape propagation, i.e. at op construction they must have the same length and overlapping axis size ranges",
				self.weights_grad.shape(),
				self.input.shape(),
			)
			.into());
		}

		Ok(SplineBackInstance {
			input: self.input.id(),
			weights: self.weights.id(),
			output_grad: self.output_grad.id(),
			input_grad: self.input_grad.id(),
			weights_grad: self.weights_grad.id(),
		})
	}
}

#[derive(Debug, Clone)]
pub struct SplineBackInstance {
	input: NodeID,
	weights: NodeID,
	output_grad: NodeID,
	input_grad: NodeID,
	weights_grad: NodeID,
}

impl OpInstance for SplineBackInstance {
	fn op_type(&self) -> &'static str {
		"SplineBack"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(SplineBack {
			input: graph.node_from_id(self.input),
			weights: graph.node_from_id(self.weights),
			output_grad: graph.node_from_id(self.output_grad),
			input_grad: graph.node_from_id(self.input_grad),
			weights_grad: graph.node_from_id(self.weights_grad),
		})
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input, self.weights, self.output_grad]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.input_grad, self.weights_grad]
	}

	fn gradient(&self, _ctx: &mut GradientContext) -> Result<(), GradientError> {
		Err(GradientError::Unimplemented)
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		if ctx.input_shape(&self.input) != ctx.input_shape(&self.output_grad) {
			return Err(format!(
				"input shape ({:?}) must equal output_grad shape ({:?})",
				ctx.input_shape(&self.input).slice(),
				ctx.input_shape(&self.output_grad).slice(),
			)
			.into());
		}

		ctx.merge_output_shape(&self.input_grad, &ctx.input_shape(&self.input).slice().into())?;

		ctx.merge_output_shape(&self.weights_grad, &ctx.input_shape(&self.weights).slice().into())?;
		let weights_shape = ctx.input_shape(&self.weights).slice();
		if 3 != weights_shape[0] {
			return Err(format!(
				"weights shape ({:?}) must have a first (outermost) axis of size 3",
				weights_shape,
			)
			.into());
		}

		let output_shape = &weights_shape[1..].into();

		ctx.broadcast_merge_output_shape(&self.input_grad, output_shape)
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let input = ctx.get_input(&self.input);
		let weights = ctx.get_input(&self.weights);
		let output_grad = ctx.get_input(&self.output_grad);

		let input_shape = input.shape();
		let output_shape = output_grad.shape();
		let weights_outer_shape = weights.shape();
		let weights_shape = &weights_outer_shape[1..];

		let mut weights_iter = weights.outer_iter();
		let weights0 = weights_iter.next().unwrap();
		let weights1 = weights_iter.next().unwrap();
		let weights2 = weights_iter.next().unwrap();

		debug_assert_eq!(
			input_shape, output_shape,
			"input shape: {:?} did not match output shape: {:?}",
			input_shape, output_shape
		);
		debug_assert_eq!(
			weights_outer_shape[0], 3,
			"The outermost dimension of the weights shape must be 3 not {}",
			weights_outer_shape[0]
		);
		debug_assert!(
			weights0.broadcast(output_shape).is_some(),
			"Could not broadcast weights_shape[1..]: {:?} to input/output shape: {:?}",
			weights_shape,
			input_shape
		);

		if ctx.is_required_output(&self.input_grad) {
			let input_grad = ctx.get_output(&self.input_grad);

			Zip::from(&output_grad)
				.and(&input)
				.and_broadcast(&weights0)
				.and_broadcast(&weights1)
				.and_broadcast(&weights2)
				.and(input_grad)
				.par_for_each(|&output_grad, &input, &left, &centre, &right, input_grad| {
					let x = input;
					if x <= -1.0 {
						*input_grad += output_grad * left;
					} else if x >= 1.0 {
						*input_grad += output_grad * right;
					} else {
						let x2 = x * x;
						// let x3 = x*x*x;
						*input_grad += output_grad
							* (centre * (1.0 - x2) + x * (left * (0.5 * x - 0.5) + right * (0.5 * x + 0.5)));
					}
				});
		}

		if ctx.is_required_output(&self.weights_grad) {
			let mut weights_grad = ctx.get_output(&self.weights_grad);
			let mut weights_grad_iter = weights_grad.outer_iter_mut();
			let mut weights_grad0 = weights_grad_iter.next().unwrap();
			let mut weights_grad1 = weights_grad_iter.next().unwrap();
			let mut weights_grad2 = weights_grad_iter.next().unwrap();

			Zip::from(&output_grad)
				.and(&input)
				.and_broadcast(weights_grad0.cell_view())
				.and_broadcast(weights_grad1.cell_view())
				.and_broadcast(weights_grad2.cell_view())
				.for_each(|&output_grad, &input, left_grad, centre_grad, right_grad| {
					let x = input;
					if x <= -1.0 {
						left_grad.set(left_grad.get() + output_grad * (x + 7.0 / 12.0));
						centre_grad.set(centre_grad.get() + output_grad * (-2.0 / 3.0));
						right_grad.set(right_grad.get() + output_grad * (1.0 / 12.0));
					} else if x >= 1.0 {
						left_grad.set(left_grad.get() + output_grad * (-1.0 / 12.0));
						centre_grad.set(centre_grad.get() + output_grad * (2.0 / 3.0));
						right_grad.set(right_grad.get() + output_grad * (x - 7.0 / 12.0));
					} else {
						let x2 = x * x;
						let x3 = x * x * x;
						left_grad.set(left_grad.get() + output_grad * (x * (1.0 / 6.0) - 0.25) * x2);
						centre_grad.set(centre_grad.get() + output_grad * (x - x3 * (1.0 / 3.0)));
						right_grad.set(right_grad.get() + output_grad * (x * (1.0 / 6.0) + 0.25) * x2);
					}
				});
		}

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::{spline, tanh_esque};
	use alumina_core::graph::Node;
	use alumina_test::grad_numeric_test::GradNumericTest;

	#[test]
	fn grad_numeric_test() {
		let input = Node::new(&[7, 5, 16]).set_name("input");

		let output = spline(&input, &[], tanh_esque()).unwrap();

		GradNumericTest::new(output, &[input]).run();
	}

	#[test]
	fn grad_numeric_shared_test() {
		let input = Node::new(&[7, 5, 16]).set_name("input");

		let output = spline(&input, &[1], tanh_esque()).unwrap();

		GradNumericTest::new(output, &[input]).run();
	}
}
