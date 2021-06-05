use crate::{
	boolean::equal,
	build_or_pretty_panic,
	elementwise::{
		abs, ceil, cos, div, elu, exp, floor, identity, leaky_relu, ln, logistic, max, min, mul, negative, reciprocal,
		relu, robust, round, scale, sign, sin, sqr, sqrt, srgb, subtract, tanh,
	},
	grad::stop_grad,
	manip::{expand_dims, permute_axes, remove_dims},
	math::{argmax, broadcast, muldiv},
	nn::{
		conv::{self, ConvData, Padding},
		matmul, softmax, softmax_cross_entropy, spline,
	},
	pool::avg_pool,
	reduce::{reduce_prod, reduce_sum},
	regularisation::{hoyer_squared, l1, l2},
	shape::{linterp, shape_of},
};
use alumina_core::{
	base_ops::{fill, shape_constraint},
	graph::{HeavyNode, Node, Op},
	init::Initialiser,
	shape::NodeShape,
};
/// Calculates the elementwise equality of input1 and input2, returning 1.0 if they are equal and 0.0 otherwise.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn equal<I1, I2>(input1: I1, input2: I2) -> Node
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	build_or_pretty_panic(equal::equal(input1, input2), "Equal")
}

/// Returns the absolute (abs) of the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn abs<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(abs::abs(input), "Abs")
}

/// Returns the closest integer greater than or equal to (ceil) the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn ceil<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(ceil::ceil(input), "Ceil Op")
}

/// Returns the cosine (cos) of the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn cos<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(cos::cos(input), "Cos Op")
}

/// Calculates the elementwise division (div) of input1 over input2.
///
/// The output node has the same shape as the inputs.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn div<I1, I2>(input1: I1, input2: I2) -> Node
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	build_or_pretty_panic(div::div(input1, input2), "Div")
}

/// Returns the exponential linear unit activation (elu) of the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn elu<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(elu::elu(input), "ELU ")
}

/// Returns the natural exponent (exp) of the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn exp<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(exp::exp(input), "Exp")
}

/// Produces and output node with the given shape, and an op to fill it elementwise with the provided value.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn fill<S: Into<NodeShape>>(value: f32, shape: S) -> Node {
	build_or_pretty_panic(fill::fill(value, shape), "Fill")
}

/// Returns the closest integer less than or equal to (floor) the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn floor<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(floor::floor(input), "Floor")
}

/// Returns the same value (identity) as the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn identity<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(identity::identity(input), "Identity")
}

/// Returns the elementwise addition of the two inputs.
///
/// Inputs must have the same shape.
///
/// The output node has the same shape as the inputs.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn add<I1, I2>(input1: I1, input2: I2) -> Node
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	build_or_pretty_panic(identity::add(input1, input2), "Identity")
}

/// Elementwise addition of the values in the inputs.
///
/// Does not perform broadcasting.
///
/// If no inputs are supplied a scalar output without inputs is returned.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn add_n<I, T>(inputs: T) -> Node
where
	I: Into<Node>,
	T: IntoIterator<Item = I>,
{
	build_or_pretty_panic(identity::add_n(inputs), "Identity")
}

/// Returns the leaky rectified linear unit activation (leaky relu) of the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn leaky_relu<I>(input: I, alpha: f32) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(leaky_relu::leaky_relu(input, alpha), "LeakyRelu")
}

/// Returns the natural logarithm (ln) of the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn ln<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(ln::ln(input), "Ln")
}

/// Applies the logistic function to each element of the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn logistic<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(logistic::logistic(input), "Logistic")
}

/// Calculates the elementwise maximum (max) of input1 and input2.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn max<I1, I2>(input1: I1, input2: I2) -> Node
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	build_or_pretty_panic(max::max(input1, input2), "Max")
}

/// Calculates the elementwise minimum (min) of input1 and input2.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn min<I1, I2>(input1: I1, input2: I2) -> Node
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	build_or_pretty_panic(min::min(input1, input2), "Min")
}

/// Calculates the elementwise multiplication (mul) of input1 and input2.
///
/// The output node has the same shape as the inputs.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn mul<I1, I2>(input1: I1, input2: I2) -> Node
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	build_or_pretty_panic(mul::mul(input1, input2), "Mul")
}

/// Returns the negative of each element of the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn negative<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(negative::negative(input), "Negative")
}

/// Returns the reciprocal of the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn reciprocal<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(reciprocal::reciprocal(input), "Reciprocal")
}

/// Returns the rectified linear unit activation (relu) of the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn relu<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(relu::relu(input), "Relu")
}

/// An `Op` which implements a range of robust loss functions.
///
/// Based on the paper: A More General Robust Loss Function https://arxiv.org/pdf/1701.03077.pdf Eq.13 & Eq.14
/// Note that:
///
/// when power(α) == 2, this is the L2 loss
///
/// when power(α) == 1, this is the pseudo-Huber/Charbonnier loss (smooth L1 loss)
///
/// when power(α) == 0, this is the Cauchy/Lorentzian loss
///
/// when power(α) == -2, this is the Geman-McClure loss
///
/// when power(α) == -∞, this is the Welsch/Leclerc loss
///
/// The scale(c) is the range of values either size of zero for which the loss will closely approximate the L2 loss.
/// A small scale value will mean that small inputs will result in larger outputs.
/// See paper for futher details.
///
/// ρ(x,α,c) =
/// if α == 0 : log(0.5*(x/c)^2+ 1)
/// if α == -∞: 1 - exp(-0.5 *(x/c)^2)
/// else      : z(α)/α * (((x/c)^2/z(α) + 1)^(α/2) − 1)
/// where z(α) = max(1, 2 - α)
pub fn robust<I>(input: I, scale: f32, power: f32) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(robust::robust(input, scale, power), "robust")
}

/// Returns the closest integer (round) to the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn round<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(round::round(input), "Round")
}

/// Returns the scaled value of the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn scale<I>(input: I, scale: f32) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(scale::scale(input, scale), "Scale")
}

/// Returns the sign of the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn sign<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(sign::sign(input), "Sign")
}

/// Returns the sine (sin) of the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn sin<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(sin::sin(input), "Sin")
}

/// Returns the square (sqr) of the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn sqr<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(sqr::sqr(input), "Sqr")
}

/// Returns the square root (sqrt) of the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn sqrt<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(sqrt::sqrt(input), "Sqrt")
}

/// Converts the input from sRGB(0.0-1.0) to Linear RGB(0.0-1.0).
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn srgb_to_linear<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(srgb::srgb_to_linear(input), "SrgbToLinear")
}

/// Converts the input from Linear RGB(0.0-1.0) to sRGB(0.0-1.0).
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn linear_to_srgb<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(srgb::linear_to_srgb(input), "LinearToSrgb")
}

/// Converts the input from sRGB(0.0-1.0) to Linear RGB(0.0-1.0).
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn srgb_to_linear_slow<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(srgb::srgb_to_linear_slow(input), "SrgbToLinear")
}

/// Converts the input from Linear RGB(0.0-1.0) to sRGB(0.0-1.0).
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn linear_to_srgb_slow<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(srgb::linear_to_srgb_slow(input), "LinearToSrgb")
}

/// Calculates the elementwise subtractision (subtract) of input1 over input2.
///
/// The output node has the same shape as the inputs.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn subtract<I1, I2>(input1: I1, input2: I2) -> Node
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	build_or_pretty_panic(subtract::subtract(input1, input2), "Subtract")
}

/// Returns the hyperbolic tangent (tanh) of the input.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn tanh<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(tanh::tanh(input), "Tanh")
}

/// Returns the same value (stop_grad) as the input but does not produce a gradient.
///
/// The output node has the same shape as the input.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn stop_grad<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(stop_grad::stop_grad(input), "StopGrad")
}

/// Insert unit axes into a nodes shape.
///
/// Unlike with the `OpBuilder` the `extra_axes` argument must be `usize` rather than `isize` because reverse indexing
/// isn't possible before the output shape is known.
///
/// # Panics
/// Panics if an axis is not in the range [0, output.len()).
/// Panics if building the underlying Op panics.
pub fn expand_dims<I>(input: I, extra_axes: &[usize]) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(expand_dims::expand_dims(input, extra_axes), "ExpandDims")
}

/// Calculates from the input a result where the axes of the ndarray have been rearranged (permuted), returning an
/// output with the same number of axes.
///
/// An input with the shape [3, 5, 7] combined with a permutation of [1, 2, 0] results in a output shape of [5, 7, 3].
///
/// # Panics
///  * panics if building the underlying Op panics.
pub fn permute_axes<I>(input: I, permutation: &[usize]) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(permute_axes::permute_axes(input, permutation), "PermuteAxes")
}

/// Calculates from the input a result where the axes of the ndarray have been reversed (transposed), returning an
/// output with the same number of axes.
///
/// An input with the shape [3, 5, 7] results in a output shape of [7, 5, 3].
///
/// # Panics
///  * panics if building the underlying Op panics.
pub fn transpose<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(permute_axes::transpose(input), "PermuteAxes")
}

/// Removes unit axes from a nodes shape.
///
/// # Panics
/// Panics if an axis is not in the range [-input.len(), input.len()).
/// Panics if an axis is not known to be 1.
pub fn remove_dims<I>(input: I, axes: &[isize]) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(remove_dims::remove_dims(input, axes), "PermuteAxes")
}

/// Returns the integer location of the maximum for each lane in the provided axis.
///
/// The output node has the shape of the input, but with the axis removed.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn argmax<I>(input: I, axis: isize) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(argmax::argmax(input, axis), "ArgMax")
}

/// broadcast the values of value_input to the shape of shape_input and return the result
pub fn broadcast<I1, I2>(shape_input: I1, value_input: I2) -> Node
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	build_or_pretty_panic(broadcast::broadcast(shape_input, value_input), "Broadcast")
}

/// broadcast the values of input to the existing output and return the Op
pub fn broadcast_into<I, O>(input: I, output: O) -> Op
where
	I: Into<Node>,
	O: Into<Node>,
{
	build_or_pretty_panic(broadcast::broadcast_into(input, output), "Broadcast")
}

/// broadcast input2 to the shape of input1, and call f with the inputs
pub fn broadcast_fn<F, I1, I2>(f: F, input1: I1, input2: I2) -> Node
where
	F: FnOnce(Node, Node) -> Node,
	I1: Into<Node>,
	I2: Into<Node>,
{
	build_or_pretty_panic(
		broadcast::broadcast_fn(|n1, n2| Ok(f(n1, n2)), input1, input2),
		"Broadcast",
	)
}

/// broadcast input1 to the shape of input2, and call f with the inputs
pub fn broadcast_rev_fn<F, I1, I2>(f: F, input1: I1, input2: I2) -> Node
where
	F: FnOnce(Node, Node) -> Node,
	I1: Into<Node>,
	I2: Into<Node>,
{
	build_or_pretty_panic(
		broadcast::broadcast_rev_fn(|n1, n2| Ok(f(n1, n2)), input1, input2),
		"Broadcast",
	)
}

/// In-place Bias. Returns the input node after adding a broadcasted bias.
///
/// This function is odd in that it returns the input node rather than a newly created output.
///
/// axes determines which will have unique bias values and wont be broadcast. These axes must have a fixed shape or an
/// error is returned.
pub fn ibias<I>(input: I, axes: &[isize]) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(broadcast::ibias(input, axes), "Broadcast")
}

/// Returns a new output node which is the addition of the input node and a broadcasted bias.
///
/// axes determines which will have unique bias values and wont be broadcast. These axes must have a fixed shape or an
/// error is returned.
pub fn bias<I>(input: I, axes: &[isize]) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(broadcast::bias(input, axes), "Broadcast")
}

/// An activation function based on complex multiplication and division.
///
/// This Op breaks up the inner most axis into groups of 4,
/// interprets them as two complex numbers w = (a + ib), x = (c + id),
/// and outputs the multiplication result of w * x, and division of w/x.
///
/// If the innermost axis has a remainder after group into 4s, these values are passed through without modification.
pub fn muldiv<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(muldiv::muldiv(input), "MulDiv")
}

// TODO
pub fn conv<I>(input: I, output_channels: usize, filter_shape: &[usize], padding: conv::Padding) -> HeavyNode<ConvData>
where
	I: Into<Node>,
{
	build_or_pretty_panic(conv::conv(input, output_channels, filter_shape, padding), "Conv")
}

// TODO
pub fn conv_with<I, F>(input: I, filter: F, padding: conv::Padding) -> Node
where
	I: Into<Node>,
	F: Into<Node>,
{
	build_or_pretty_panic(conv::conv_with(input, filter, padding), "Conv")
}

// TODO
pub fn conv_into<I, O>(input: I, output: O, filter_shape: &[usize], padding: Padding) -> ConvData
where
	I: Into<Node>,
	O: Into<Node>,
{
	build_or_pretty_panic(conv::conv_into(input, output, filter_shape, padding), "Conv")
}

// TODO
pub fn conv_with_into<I, F, O>(input: I, filter: F, output: O, padding: conv::Padding) -> Op
where
	I: Into<Node>,
	F: Into<Node>,
	O: Into<Node>,
{
	build_or_pretty_panic(conv::conv_with_into(input, filter, output, padding), "Conv")
}

/// Matrix multiply by a new node
pub fn linear<I>(input: I, output_channels: usize, init: Initialiser) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(matmul::linear(input, output_channels, init), "MatMul")
}

/// Matrix multiply by a parameters node plus a bias parameter on the output.
pub fn affine<I>(input: I, output_channels: usize, init: Initialiser) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(matmul::affine(input, output_channels, init), "MatMul or Add")
}

pub fn matmul<I1, I2>(input1: I1, input2: I2) -> Node
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	build_or_pretty_panic(matmul::matmul(input1, input2), "MatMul")
}

pub fn matmul_into<I1, I2, O>(input1: I1, input2: I2, output: O) -> Op
where
	I1: Into<Node>,
	I2: Into<Node>,
	O: Into<Node>,
{
	build_or_pretty_panic(matmul::matmul_into(input1, input2, output), "MatMul")
}

/// Calculates the Softmax of the logits across the select axis followed by CrossEntropy of that result with the
/// supplied labels.
///
/// These operations are combined for numerical stability and performance improvement.
///
/// `let output = mul(labels, negative(ln(softmax(logits, axis))))`
///
/// The output node has the shape of the logits and labels, but with the axis removed.
///
/// # Panics
///  * Panics if building the underlying Op panics.
///  * Panics if logits and labels have shapes of different lengths.
pub fn softmax_cross_entropy<I1, I2>(logits: I1, labels: I2, axis: isize) -> Node
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	build_or_pretty_panic(
		softmax_cross_entropy::softmax_cross_entropy(logits, labels, axis),
		"SoftmaxCrossEntropy",
	)
}

/// Calculates the combined Softmax of the input nodes.
///
/// Axis determines the grouping direction.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn softmax<I>(logits: I, axis: isize) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(softmax::softmax(logits, axis), "Softmax")
}

/// A parameterised activation function that is smooth and continuous, consisting of linear components jointed by a
/// central cubic spline region.
pub fn spline<I>(input: I, axes: &[isize], init: Initialiser) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(spline::spline(input, axes, init), "Spline")
}

/// A parameterised activation function that is smooth and continuous, consisting of linear components jointed by a
/// central cubic spline region.
pub fn spline_with<I, W>(input: I, weights: W) -> Node
where
	I: Into<Node>,
	W: Into<Node>,
{
	build_or_pretty_panic(spline::spline_with(input, weights), "Spline")
}

/// A parameterised activation function that is smooth and continuous, consisting of linear components jointed by a
/// central cubic spline region.
pub fn spline_into<I, O>(input: I, axes: &[isize], init: Initialiser, output: O) -> Op
where
	I: Into<Node>,
	O: Into<Node>,
{
	build_or_pretty_panic(spline::spline_into(input, axes, init, output), "Spline")
}

/// A parameterised activation function that is smooth and continuous, consisting of linear components jointed by a
/// central cubic spline region.
pub fn spline_with_into<I, W, O>(input: I, weights: W, output: O) -> Op
where
	I: Into<Node>,
	W: Into<Node>,
	O: Into<Node>,
{
	build_or_pretty_panic(spline::spline_with_into(input, weights, output), "Spline")
}

pub fn avg_pool<I>(input: I, factors: &[usize]) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(avg_pool::avg_pool(input, factors), "ReduceSum")
}

/// # Panics
/// Panics if building the underlying Op panics.
pub fn reduce_sum<I>(input: I, axes: &[isize], keep_dims: bool) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(reduce_sum::reduce_sum(input, axes, keep_dims), "ReduceSum")
}

/// # Panics
/// Panics if building the underlying Op panics.
pub fn reduce_mean<I>(input: I, axes: &[isize], keep_dims: bool) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(reduce_sum::reduce_mean(input, axes, keep_dims), "ReduceMean")
}

/// # Panics
/// Panics if building the underlying Op panics.
pub fn reduce_prod<I>(input: I, axes: &[isize], keep_dims: bool) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(reduce_prod::reduce_prod(input, axes, keep_dims), "ReduceSum")
}

/// Calculates the combined L1 norm of the input nodes, returning a scalar node.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn l1<I, T: IntoIterator<Item = I>>(inputs: T) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(l1::l1(inputs), "L1")
}

/// Calculates the combined L2 norm of the input nodes, returning a scalar node.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn l2<I, T: IntoIterator<Item = I>>(inputs: T) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(l2::l2(inputs), "L2")
}

pub fn hoyer_squared<I, T: IntoIterator<Item = I>>(inputs: T) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(hoyer_squared::hoyer_squared(inputs), "hoyer_squared")
}

pub fn linterp<I>(input: I, factors: &[usize]) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(linterp::linterp(input, factors), "Linterp")
}

/// Applies a ShapeConstraint  which enforces propagation of the runtime input shape to the output shape at runtime.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn same_shape<I, O>(input: I, output: O) -> Op
where
	I: Into<Node>,
	O: Into<Node>,
{
	build_or_pretty_panic(shape_constraint::same_shape(input, output), "ShapeConstraint")
}

/// Returns the integer location of the maximum for each lane in the provided axis.
///
/// The output node has the shape of the input, but with the axis removed.
///
/// # Panics
/// Panics if building the underlying Op panics.
pub fn shape_of<I>(input: I) -> Node
where
	I: Into<Node>,
{
	build_or_pretty_panic(shape_of::shape_of(input), "ShapeOf")
}
