use crate::math::broadcast::Broadcast;
use alumina_core::{
	base_ops::{OpSpecification, OpInstance},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{merge_graphs, Node, NodeID, NodeTag, Op, Graph},
	init::{duplicate, Initialiser},
	shape::{NodeAxis, NodeShape},
	shape_prop::ShapePropContext,
};
use indexmap::{indexset, IndexSet};
use ndarray::Dimension;
use matrixmultiply_mt;
use std::any::Any;

fn get_inner_outer(shape: &NodeShape) -> (usize, NodeAxis) {
	let mut iter = shape.slice().iter().enumerate().rev();

	let mut inner = 1;
	let mut outer = NodeAxis::known(1);
	for (i, axis) in &mut iter {
		if let (NodeAxis::Known { val }, true) = (axis, i > 0) {
			inner *= val;
		} else {
			outer = outer.multiply(axis);
			break;
		}
	}

	for (_i, axis) in &mut iter {
		outer = outer.multiply(axis);
	}

	(inner, outer)
	// shape.slice()[1..]
	// 	.iter()
	// 	.rev()
	// 	.take_while(|dim| dim.is_known())
	// 	.fold(1, |prod, dim| match dim {
	// 		&NodeAxis::Known { val: x } => prod * x,
	// 		_ => unreachable!(),
	// 	})
}

pub fn linear<I>(input: I, output_channels: usize, init: Initialiser) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();

	let n = output_channels;
	let (k, outer) = get_inner_outer(&input.shape());

	let weights = input
		.graph()
		.new_node([k, n].iter().into())
		.set_init(init)
		.add_tag(NodeTag::Parameter)
		.set_name_unique(&format!("linear({})_weights", input));

	let output = input
		.graph()
		.new_node([outer, NodeAxis::known(output_channels)].iter().into())
		.set_name_unique(&format!("linear({})", input));

	let _op = MatMul::new(input, weights, output.clone())
		.k(Some(k))
		.n(Some(n))
		.build()?;

	Ok(output)
}

pub fn linear_into<I, O>(input: I, output: O, init: Initialiser) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
	O: Into<Node>,
{
	let input = input.into();
	let output = output.into();

	let (n, _outer) = get_inner_outer(&output.shape());
	let (k, _outer) = get_inner_outer(&input.shape());

	let weights = input
		.graph()
		.new_node([k, n].iter().into())
		.set_init(init)
		.add_tag(NodeTag::Parameter)
		.set_name_unique(&format!("linear({})_weights", input));

	// let output = input
	// 	.graph()
	// 	.new_node([-1, output_channels as isize].iter().into())
	// 	.set_name_unique(&format!("linear({})", input));

	let _op = MatMul::new(input, weights, output.clone())
		.k(Some(k))
		.n(Some(n))
		.build()?;

	Ok(output)
}

/// Matrix multiply by a parameters node plus a bias parameter on the output.
pub fn affine<I>(input: I, output_channels: usize, init: Initialiser) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();

	let n = output_channels;
	let (k, _) = get_inner_outer(&input.shape());

	let weights = input
		.graph()
		.new_node([k, n].iter().into())
		.set_init(init)
		.add_tag(NodeTag::Parameter)
		.set_name_unique(&format!("affine({})_weights", input));

	let bias = input
		.graph()
		.new_node([1, n].iter().into())
		.set_init(duplicate(0.0))
		.add_tag(NodeTag::Parameter)
		.set_name_unique(&format!("affine({})_bias", input));

	let output = input
		.graph()
		.new_node([-1, output_channels as isize].iter().into())
		.set_name_unique(&format!("affine({})", input));

	let _op = MatMul::new(input, weights.clone(), output.clone())
		.k(Some(k))
		.n(Some(n))
		.build()?;

	let _op = Broadcast::new(&bias, &output).build()?;

	Ok(output)
}

pub fn matmul<I1, I2>(input1: I1, input2: I2) -> Result<Node, OpBuildError>
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	let input1 = input1.into();
	let input2 = input2.into();
	merge_graphs(&[input1.graph(), input2.graph()]);
	let output = input1
		.graph()
		.new_node([-1, -1].iter().into())
		.set_name_unique(&format!("matmul({},{})", input1, input2));

	let _op = MatMul::new(input1, input2, output.clone()).build()?;

	Ok(output)
}

pub fn matmul_into<I1, I2, O>(input1: I1, input2: I2, output: O) -> Result<Op, OpBuildError>
where
	I1: Into<Node>,
	I2: Into<Node>,
	O: Into<Node>,
{
	let input1 = input1.into();
	let input2 = input2.into();
	let output = output.into();
	let op = MatMul::new(input1, input2, output).build()?;
	Ok(op)
}

/// Calculate C += α A B
///
/// If one or more of M, N, or K are known, the others can be found at run time.
/// If none of M, N or K are known, a guess will be made based on any arguments having 2 dimensions.
/// Shapes may be n-dimensional, however they must all split cleanly along an axis into M, N, K shapes.
///
/// An error will be returned if any inconsistencies are found or the M, N, K sizes cannot be uniquely inferred.
#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct MatMul {
	matrix_a: Node,
	matrix_b: Node,
	matrix_c: Node,
	a_trans: bool,
	b_trans: bool,
	c_trans: bool,
	m: Option<usize>,
	n: Option<usize>,
	k: Option<usize>,
	alpha: f32,
}

impl MatMul {
	pub fn new<A, B, C>(matrix_a: A, matrix_b: B, matrix_c: C) -> Self
	where
		A: Into<Node>,
		B: Into<Node>,
		C: Into<Node>,
	{
		MatMul {
			matrix_a: matrix_a.into(),
			matrix_b: matrix_b.into(),
			matrix_c: matrix_c.into(),
			a_trans: false,
			b_trans: false,
			c_trans: false,
			m: None,
			n: None,
			k: None,
			alpha: 1.0,
		}
	}

	pub fn alpha(mut self, alpha: f32) -> Self {
		self.alpha = alpha;
		self
	}

	pub fn a_trans(mut self, trans: bool) -> Self {
		self.a_trans = trans;
		self
	}

	pub fn b_trans(mut self, trans: bool) -> Self {
		self.b_trans = trans;
		self
	}

	pub fn c_trans(mut self, trans: bool) -> Self {
		self.c_trans = trans;
		self
	}

	pub fn m(mut self, m: Option<usize>) -> Self {
		self.m = m;
		self
	}

	pub fn n(mut self, n: Option<usize>) -> Self {
		self.n = n;
		self
	}

	pub fn k(mut self, k: Option<usize>) -> Self {
		self.k = k;
		self
	}
}

impl OpSpecification for MatMul {
	type InstanceType = MatMulInstance;

	fn type_name(&self) -> &'static str {
		"MatMul"
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.matrix_a.clone(), self.matrix_b.clone()]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.matrix_c.clone()]
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(MatMulInstance {
			matrix_a: self.matrix_a.id().clone(),
			matrix_b: self.matrix_b.id().clone(),
			matrix_c: self.matrix_c.id().clone(),
			a_trans: self.a_trans,
			b_trans: self.b_trans,
			c_trans: self.c_trans,
			m: self.m,
			n: self.n,
			k: self.k,
			alpha: self.alpha,
		})
	}
}

#[derive(Debug, Clone)]
pub struct MatMulInstance {
	matrix_a: NodeID,
	matrix_b: NodeID,
	matrix_c: NodeID,
	a_trans: bool,
	b_trans: bool,
	c_trans: bool,
	m: Option<usize>,
	n: Option<usize>,
	k: Option<usize>,
	alpha: f32,
}

impl OpInstance for MatMulInstance {
	fn type_name(&self) -> &'static str {
		"ReduceSum"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(MatMul {
			matrix_a: graph.node_from_id(self.matrix_a),
			matrix_b: graph.node_from_id(self.matrix_b),
			matrix_c: graph.node_from_id(self.matrix_c),
			a_trans: self.a_trans,
			b_trans: self.b_trans,
			c_trans: self.c_trans,
			m: self.m,
			n: self.n,
			k: self.k,
			alpha: self.alpha,
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.matrix_a.clone(), self.matrix_b.clone()]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.matrix_c.clone()]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		let _op = MatMul::new(
			ctx.node(&self.matrix_a),
			ctx.grad_of(&self.matrix_c),
			ctx.grad_of(&self.matrix_b),
		)
		.a_trans(!self.a_trans)
		.b_trans(self.c_trans)
		.c_trans(self.b_trans)
		.m(self.k)
		.n(self.n)
		.k(self.m)
		.alpha(self.alpha)
		.build()?;

		let _op = MatMul::new(
			ctx.grad_of(&self.matrix_c),
			ctx.node(&self.matrix_b),
			ctx.grad_of(&self.matrix_a),
		)
		.a_trans(self.c_trans)
		.b_trans(!self.b_trans)
		.c_trans(self.a_trans)
		.m(self.m)
		.n(self.k)
		.k(self.n)
		.alpha(self.alpha)
		.build()?;

		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		// Use the shape of A and B to try and inffer a single unknown dimension of C
		// if shape of C has more than 1 unknown then throw error

		// no free dims -> exit, passes will check for errors
		// one free dim -> calculate or throw error
		// two adjacent free dims -> calculate or throw error
		// else -> error

		let a_shape = ctx.input_shape(&self.matrix_a);
		let b_shape = ctx.input_shape(&self.matrix_b);
		let mut c_shape = ctx.output_shape(&self.matrix_c).clone();

		let (m, n, _k) = if c_shape.is_known() {
			let c_shape = c_shape.to_data_shape().unwrap();
			let (m, n, _k) = find_mnk(
				(a_shape.slice(), self.a_trans),
				(b_shape.slice(), self.b_trans),
				Some((c_shape.slice(), self.c_trans)),
				self.m,
				self.n,
				self.k,
			)?;

			let err = || {
				format!(
					"MatMul Op ({}) Error: \
					 The M({}) and N({}) dimensions of the matrix multiplication, \
					 are not divisible into the output shape ({:?}, transpose:{}). \
					 These must be divisible to allow inference of the unknowns",
					ctx.current_op(),
					m,
					n,
					c_shape,
					self.c_trans
				)
				.into()
			};

			let check_divisibility = |iter: &mut dyn Iterator<Item = &usize>| {
				let mut m = m;
				let mut n = n;

				for i in iter {
					if m > 1 && m % i == 0 {
						m /= i;
					} else if m == 1 && n % i == 0 {
						n /= i;
					} else {
						return Err(err());
					}
				}
				if m == 1 && n == 1 {
					Ok(())
				} else {
					Err(err())
				}
			};

			if self.c_trans {
				return check_divisibility(&mut c_shape.slice().iter().rev());
			} else {
				return check_divisibility(&mut c_shape.slice().iter());
			};
		} else {
			find_mnk(
				(a_shape.slice(), self.a_trans),
				(b_shape.slice(), self.b_trans),
				None,
				self.m,
				self.n,
				self.k,
			)?
		};

		let (outer_mat_dim, inner_mat_dim) = if self.c_trans { (n, m) } else { (m, n) };

		// Locate the unknowns in the shape of C, and the strides (product of dims) before and after them
		let mut outer_stride = 1;
		let mut inner_stride = 1;

		let mut outer_ind = 0; // this should be the index of the outermost unknown
		let mut inner_ind = c_shape.len(); // this should be the index AFTER the innermost unknown

		for dim in c_shape.iter() {
			match dim {
				NodeAxis::Known { val } => {
					outer_stride *= val;
					outer_ind += 1;
				}
				_ => break,
			}
		}

		for dim in c_shape.iter().rev() {
			match dim {
				NodeAxis::Known { val } => {
					inner_stride *= val;
					inner_ind -= 1;
				}
				_ => break,
			}
		}

		let outer_val = outer_mat_dim / outer_stride;
		let inner_val = inner_mat_dim / inner_stride;

		if outer_mat_dim - outer_val * outer_stride != 0 || inner_mat_dim - inner_val * inner_stride != 0 {
			return Err(format!(
				"MatMul Op ({}) Error: \
				 Could not infer shape of output. \
				 The M({}) and N({}) dimensions of the matrix multiplication, \
				 are not divisible by the product of the output dimensions before({}) and after({}) \
				 the unknown dimensions in the output shape ({:?}, transpose:{}). \
				 These must be divisible to allow inference of the unknowns",
				ctx.current_op(),
				m,
				n,
				outer_stride,
				inner_stride,
				c_shape,
				self.c_trans
			)
			.into());
		}

		if outer_ind + 1 == inner_ind {
			// one unknown
			if ::std::cmp::min(inner_val, outer_val) != 1 {
				return Err(format!(
					"MatMul Op ({}) Error: \
					 Could not infer shape of output. \
					 One of either the M({}) or N({}) dimensions of the matrix multiplication, \
					 must be exactly equal to the product of the output dimensions before({}) and after({}) \
					 the unknown dimension in the output shape ({:?}, transpose:{}).",
					ctx.current_op(),
					m,
					n,
					outer_stride,
					inner_stride,
					c_shape,
					self.c_trans
				)
				.into());
			}
			c_shape.slice_mut()[outer_ind] = NodeAxis::known(::std::cmp::max(outer_val, inner_val));
			ctx.merge_output_shape(&self.matrix_c, &c_shape)?;
		} else if outer_ind + 2 == inner_ind {
			// two unknowns, adjacent
			c_shape.slice_mut()[outer_ind] = NodeAxis::known(outer_val);
			c_shape.slice_mut()[inner_ind - 1] = NodeAxis::known(inner_val);
			ctx.merge_output_shape(&self.matrix_c, &c_shape)?;
		} else {
			return Err(format!(
				"MatMul Op ({}) Error: \
				 could not infer shape of output. \
				 The output shape contains non-adjacent unknown dimensions. This creates ambiguity.",
				ctx.current_op()
			)
			.into());
		}

		Ok(())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let matrix_a = ctx.get_input_standard(&self.matrix_a);
		let matrix_b = ctx.get_input_standard(&self.matrix_b);
		let mut matrix_c = ctx.get_output_standard(&self.matrix_c);

		let (m, n, k) = find_mnk(
			(matrix_a.shape(), self.a_trans),
			(matrix_b.shape(), self.b_trans),
			Some((matrix_c.shape(), self.c_trans)),
			self.m,
			self.n,
			self.k,
		)?;

		let matrix_a = matrix_a.as_slice().unwrap(); // these unwraps are ok as the _standard array accessors are used
		let matrix_b = matrix_b.as_slice().unwrap();
		let matrix_c = matrix_c.as_slice_mut().unwrap();

		unsafe {
			let (rsa, csa) = if self.a_trans { (1, m) } else { (k, 1) };
			let (rsb, csb) = if self.b_trans { (1, k) } else { (n, 1) };
			let (rsc, csc) = if self.c_trans { (1, m) } else { (n, 1) };

			matrixmultiply_mt::sgemm(
				m,
				k,
				n,
				self.alpha,
				matrix_a.as_ptr(),
				rsa as isize,
				csa as isize,
				matrix_b.as_ptr(),
				rsb as isize,
				csb as isize,
				1.0,
				matrix_c.as_mut_ptr(),
				rsc as isize,
				csc as isize,
			);
		}

		Ok(())
	}
}

/// If one or more of M, N, or K are known find the others
/// Shapes may be multidimensional, however they must split cleanly along an axis into M, N, K shapes.
/// An error will be returned if any inconsistencies are found.
fn find_mnk(
	a_shape: (&[usize], bool),
	b_shape: (&[usize], bool),
	c_shape: Option<(&[usize], bool)>,
	old_m: Option<usize>,
	old_n: Option<usize>,
	old_k: Option<usize>,
) -> ::std::result::Result<(usize, usize, usize), String> {
	let (a_shape, a_trans) = a_shape;
	let (b_shape, b_trans) = b_shape;

	let mut m = old_m;
	let mut n = old_n;
	let mut k = old_k;

	// If no matrix dimensions are known, guess based on any of the inputs having 2 dimensions.
	if let (None, None, None) = (m, n, k) {
		if a_shape.len() == 2 {
			if a_trans {
				k = Some(a_shape[0]);
				m = Some(a_shape[1]);
			} else {
				m = Some(a_shape[0]);
				k = Some(a_shape[1]);
			}
		} else if b_shape.len() == 2 {
			if b_trans {
				n = Some(b_shape[0]);
				k = Some(b_shape[1]);
			} else {
				k = Some(b_shape[0]);
				n = Some(b_shape[1]);
			}
		} else if let Some((c_shape, c_trans)) = c_shape {
			if c_shape.len() == 2 {
				if c_trans {
					n = Some(c_shape[0]);
					m = Some(c_shape[1]);
				} else {
					m = Some(c_shape[0]);
					n = Some(c_shape[1]);
				}
			} else {
				return Err("MatMulPass could not infer matrix shapes. M, N, K are all None and no inputs or outputs have 2 dimensions.".to_string());
			}
		} else {
			return Err(
				"MatMulPass could not infer matrix shapes. M, N, K are all None and no inputs have 2 dimensions."
					.to_string(),
			);
		}
	}

	let (m, n, k) = if let Some(m) = m {
		let k = get_inner(m, k, a_shape, a_trans)?;
		let n = get_inner(k, n, b_shape, b_trans)?;
		(m, n, k)
	} else if let Some(n) = n {
		let k = get_outer(n, k, b_shape, b_trans)?;
		let m = get_outer(k, m, a_shape, a_trans)?;
		(m, n, k)
	} else if let Some(k) = k {
		let m = get_outer(k, m, a_shape, a_trans)?;
		let n = get_inner(k, n, b_shape, b_trans)?;
		(m, n, k)
	} else {
		unreachable!()
	};

	let err = || {
		format!(
			"MNK inferred from shapes \n\
			 A shape ({:?}), B shape ({:?}), C shape ({:?}) \n\
			 M ({:?}), N ({:?}), K ({:?})\n\
			 does not match MNK provided\n\
			 M ({}), N ({}), K ({})",
			a_shape, b_shape, c_shape, old_m, old_n, old_k, m, n, k
		)
	};

	match (old_m, old_n, old_k) {
		(Some(old_m), _, _) if old_m != m => Err(err()),
		(_, Some(old_n), _) if old_n != n => Err(err()),
		(_, _, Some(old_k)) if old_k != k => Err(err()),
		_ => Ok((m, n, k)),
	}
}

fn get_inner(outer: usize, inner: Option<usize>, shape: &[usize], trans: bool) -> ::std::result::Result<usize, String> {
	if trans {
		return get_outer(outer, inner, shape, false);
	};

	let (i, o) = shape.iter().fold((1, 1), |(mut i, mut o), &dim| {
		if o == outer {
			i *= dim;
		} else {
			o *= dim;
		}
		(i, o)
	});

	if o != outer {
		return Err(format!("Could not determine inner (outer if transposed) dimension of matrix. Outer matrix dimension: '{}' did not equal the product of outer dimensions in shape{:?}", outer, shape));
	}

	match inner {
		None => Ok(i),
		Some(existing) if existing == i => Ok(i),
		_ => Err(format!("Could not determine inner (outer if transposed) dimension of matrix. The found inner dimension: '{}' conflicts with the inner dimension hint: '{:?}'", i, inner))
	}
}

fn get_outer(inner: usize, outer: Option<usize>, shape: &[usize], trans: bool) -> ::std::result::Result<usize, String> {
	if trans {
		return get_inner(inner, outer, shape, false);
	};

	let (i, o) = shape.iter().rev().fold((1, 1), |(mut i, mut o), &dim| {
		if i == inner {
			o *= dim;
		} else {
			i *= dim;
		}
		(i, o)
	});

	if i != inner {
		return Err(format!("Could not determine outer (inner if transposed) dimension of matrix. Inner matrix dimension: '{}' did not equal the product of inner dimensions in shape{:?}", inner, shape));
	}

	match outer {
		None => Ok(o),
		Some(existing) if existing == o => Ok(o),
		_ => Err(format!("Could not determine outer (inner if transposed) dimension of matrix. The found outer dimension: '{}' conflicts with the outer dimension hint: '{:?}'", o, outer))
	}
}

#[cfg(test)]
mod tests {
	use super::MatMul;
	use crate::elementwise::identity::Identity;
	use alumina_core::{base_ops::OpSpecification, graph::Node};
	use alumina_test::grad_numeric_test::GradNumericTest;
	use indexmap::indexset;

	#[test]
	fn grad_numeric_test() {
		let input1 = Node::new(&[7, 5]).set_name("input1");
		let input2 = Node::new(&[5, 16]).set_name("input2");
		let output = Node::new(&[7, 16]).set_name("output");

		let _o1 = MatMul::new(&input1, &input2, &output).build().unwrap();

		GradNumericTest::new(&output, &indexset![&input1, &input2]).run();
	}

	#[test]
	fn grad_numeric_a_trans_test() {
		let input1 = Node::new(&[5, 7]).set_name("input1");
		let input2 = Node::new(&[5, 16]).set_name("input2");
		let output = Node::new(&[7, 16]).set_name("output");

		let _o1 = MatMul::new(&input1, &input2, &output).a_trans(true).build().unwrap();

		GradNumericTest::new(&output, &indexset![&input1, &input2]).run();
	}

	#[test]
	fn grad_numeric_c_trans_test() {
		let input1 = Node::new(&[7, 5]).set_name("input1");
		let input2 = Node::new(&[5, 16]).set_name("input2");
		let output = Node::new(&[16, 7]).set_name("output");

		let _o1 = MatMul::new(&input1, &input2, &output).c_trans(true).build().unwrap();

		GradNumericTest::new(&output, &indexset![&input1, &input2]).run();
	}

	#[test]
	fn grad_numeric_inference_outer_test() {
		let input1 = Node::new(&[7, 5]).set_name("input1");
		let input2 = Node::new(&[5, 16]).set_name("input2");
		let output1 = Node::new(&[-1, 16]).set_name("output1");
		let output2 = Node::new(&[7, 16]).set_name("output2");

		let _o1 = MatMul::new(&input1, &input2, &output1).build().unwrap();
		let _o1 = Identity::new_default(&output1, &output2).build().unwrap();

		GradNumericTest::new(&output2, &indexset![&input1, &input2]).run();
	}

	#[test]
	fn grad_numeric_inference_inner_test() {
		let input1 = Node::new(&[3, 5, 2]).set_name("input1");
		let input2 = Node::new(&[2, 16]).set_name("input2");
		let output1 = Node::new(&[15, -1]).set_name("output1");
		let output2 = Node::new(&[15, 16]).set_name("output2");

		let _o1 = MatMul::new(&input1, &input2, &output1).build().unwrap();
		let _o1 = Identity::new_default(&output1, &output2).build().unwrap();

		GradNumericTest::new(&output2, &indexset![&input1, &input2]).run();
	}

	#[test]
	fn grad_numeric_inference_dual_test() {
		let input1 = Node::new(&[7, 3]).set_name("input1");
		let input2 = Node::new(&[3, 4, 4]).set_name("input2");
		let output1 = Node::new(&[-1, -1]).set_name("output1");
		let output2 = Node::new(&[16, 7]).set_name("output2");

		let _o1 = MatMul::new(&input1, &input2, &output1).c_trans(true).build().unwrap();
		let _o1 = Identity::new_default(&output1, &output2).build().unwrap();

		GradNumericTest::new(&output2, &indexset![&input1, &input2]).run();
	}
}
