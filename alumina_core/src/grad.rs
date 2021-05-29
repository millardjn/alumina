//! Types and tools for constructing symbolic gradients.
use crate::{
	base_ops::{fill::fill_into, shape_constraint::same_shape},
	errors::GradError,
	graph::{Node, NodeID},
	subgraph::{backward_subgraph_from, forward_subgraph_from, SubGraph},
	util::display::{Iter2Display, IterDisplay},
};
use indexmap::{IndexMap, IndexSet};
use std::borrow::Borrow;

/// Returns a map from the `xs` nodes to the gradients of `y` w.r.t. them, excluding any intermediate nodes.
///
/// ## Validity
/// The gradients returned are only valid for the graph at the time of the call. If further forward pass operations are
/// added after calling `grad(..)`, the respective gradiant operations will be missing.
///
/// # Example
/// ```
/// # use alumina::graph::Node;
/// # use alumina::ops::elementwise::{mul::mul, identity::identity};
/// # use alumina::grad::grad_strict;
/// # use ndarray::{arr0, arr1};
/// # use indexmap::indexset;
/// # use failure::Error;
/// # fn main() -> Result<(), Error> {
/// let x = Node::new(&[2]).set_name("x").set_value(arr0(3.0));
/// let y = Node::new(&[2]).set_name("y").set_value(arr0(2.1));
///
/// let intermediate = mul(&x, &y)?;
/// let loss = identity(&intermediate)?;
///
/// let grads = grad_strict(&loss, &[&x, &y])?;
///
/// assert_eq!(grads[&x].calc()?, arr1(&[2.1, 2.1]).into_dyn());
/// assert_eq!(grads[&y].calc()?, arr1(&[3.0, 3.0]).into_dyn());
///
/// assert!(!grads.contains_key(&intermediate));
/// # Ok(())
/// # }
/// ```
pub fn grad_strict<N1, N2, T>(y: N1, xs: T) -> Result<IndexMap<Node, Node>, GradError>
where
	N1: Into<Node>,
	N2: Into<Node>,
	T: IntoIterator<Item = N2>,
{
	let xs: IndexSet<Node> = xs.into_iter().map(Into::into).collect();
	Ok(grad(y, &xs)?.into_iter().filter(|(n, _ng)| xs.contains(n)).collect())
}

/// Returns a map from the `xs` nodes to the gradients of `y` w.r.t. them, along with any other gradients required to
/// calculate them.
///
/// ## Validity
/// The gradients returned are only valid for the graph at the time of the call. If further forward pass operations are
/// added after calling `grad(..)`, the respective gradiant operations will be missing.
///
/// # Example
/// ```
/// # use alumina::graph::Node;
/// # use alumina::ops::elementwise::{mul::mul, identity::identity};
/// # use alumina::grad::grad;
/// # use ndarray::{arr0, arr1};
/// # use indexmap::indexset;
/// # use failure::Error;
/// # fn main() -> Result<(), Error> {
/// let x = Node::new(&[2]).set_name("x").set_value(arr0(3.0));
/// let y = Node::new(&[2]).set_name("y").set_value(arr0(2.1));
///
/// let intermediate = mul(&x, &y)?;
/// let loss = identity(&intermediate)?;
///
/// let grads = grad(&loss, &[&x, &y])?;
///
/// assert_eq!(grads[&x].calc()?, arr1(&[2.1, 2.1]).into_dyn());
/// assert_eq!(grads[&y].calc()?, arr1(&[3.0, 3.0]).into_dyn());
///
/// assert_eq!(grads[&intermediate].calc()?, arr1(&[1.0, 1.0]).into_dyn());
/// # Ok(())
/// # }
/// ```
pub fn grad<N1, N2, T>(y: N1, xs: T) -> Result<IndexMap<Node, Node>, GradError>
where
	N1: Into<Node>,
	N2: Into<Node>,
	T: IntoIterator<Item = N2>,
{
	let xs: IndexSet<Node> = xs.into_iter().map(Into::into).collect();
	let y = y.into();

	let SubGraph { ops, nodes } = grad_subgraph(&y, xs.clone());

	let mut context = GradientContext::new(y, nodes);

	// make sure all xs have a grad node
	for x in &xs {
		let _ = context.grad_of(&x.borrow().id());
	}

	// take the gradient of each op, and collect any errors
	let errors = ops.iter().fold(IndexMap::new(), |mut errors, op| {
		op.instance().gradient(&mut context).unwrap_or_else(|e| {
			errors.insert(op.clone(), e);
		});
		errors
	});

	let GradientContext {
		node_to_grad,
		mut nodes,
		y,
		..
	} = context;

	// Set them to fill zero if there are no inputs to the grad
	for (node_inner, grad) in &node_to_grad {
		if y.id() != *node_inner && grad.parent_ops().is_empty() {
			fill_into(0.0, grad).unwrap_or_else(|err| {
				panic!(
					"Alumina Bug: Error building fill op for gradient of ({}).\n{:#?}",
					nodes.get(node_inner).unwrap(), err
				)
			});
			// grad.set_value(arr0(0.0));
		}
		if !grad.shape().is_known() {
			same_shape(nodes.get(node_inner).unwrap(), grad).unwrap_or_else(|err| {
				panic!(
					"Alumina Bug: Error building shape constraint for gradient of ({}).\n{:#?}",
					nodes.get(node_inner).unwrap(), err
				)
			});
		}
	}

	let result_map = node_to_grad
		.into_iter()
		.map(|(inner, grad)| (nodes.swap_take(&inner).unwrap(), grad))
		.collect();

	if errors.is_empty() {
		Ok(result_map)
	} else {
		Err(GradError {
			errors: Iter2Display { inner: errors },
			partial: result_map,
		})
	}
}

/// The subgraph should be the intersection of everything that the xs could affect, and everything that could affect y.
/// With the addition of the xs regardless of whether they can affect y.
fn grad_subgraph(y: &Node, xs: IndexSet<Node>) -> SubGraph {
	// First work forward and find all nodes that could be affected by the values of xs
	let forward_subgraph = forward_subgraph_from(
		xs.clone(),
		|node| (false, node == y),
		|_op| (false, false),
		false,
		false,
	);

	// Second work backward from the y and only include nodes that appeared in the first subgraph
	let mut subgraph = backward_subgraph_from(
		&[&y],
		|node| {
			let exclude = !forward_subgraph.nodes.contains(node);
			(false, exclude)
		},
		|op| {
			let exclude = !forward_subgraph.ops.contains(op);
			(exclude, exclude)
		},
		false,
		false,
	);

	for x in xs {
		subgraph.nodes.insert(x);
	}

	subgraph
}

/// This context is provided when calling `gradient()` on each `Op`.
///
/// This is mostly used inside of a `grad()` call, and is used to lazily generate or retrieve the gradient nodes
/// required by an `Op`.
pub struct GradientContext {
	y: Node,
	node_to_grad: IndexMap<NodeID, Node>,
	nodes: IndexSet<Node>,
}

impl GradientContext {
	fn new(y: Node, subgraph_nodes: IndexSet<Node>) -> Self {
		let mut context = GradientContext {
			y: y.clone(),
			nodes: subgraph_nodes,
			node_to_grad: IndexMap::new(),
		};

		
		fill_into(1.0, context.grad_of(&y.id())).unwrap_or_else(|err| {
			panic!(
				"Alumina Bug: Error building fill op for gradient of output ({}).\n{:#?}",
				y,
				err
			)
		});

		context
	}

	/// Returns the numerator of the gradient, that is `y` in `dy/dx`.
	///
	/// Useful for automatically constructing meaningful names.
	pub fn y(&self) -> Node {
		self.y.clone()
	}

	/// This lazily instantiates and returns gradient nodes corresponding to a non-gradient inner.
	pub fn grad_of(&mut self, inner: &NodeID) -> Node {
		let &mut GradientContext {
			ref y,
			ref mut node_to_grad,
			ref nodes,
		} = self;

		let node = nodes.get(inner).unwrap_or_else(|| {
			panic!(
				"Op Bug: Node (id:{}) was accessed but is not part of {}",
				inner.id(),
				IterDisplay {
					inner: nodes.clone()
				}
			)
		});

		node_to_grad
			.entry(inner.clone())
			.or_insert_with(|| {
				node.graph()
					.new_node(node.shape())
					.set_name(format!("d({})/d({})", y.name(), node.name()))
			})
			.clone()
	}

	/// Returns the full node for an inner.
	pub fn node(&self, inner: &NodeID) -> Node {
		self.nodes.get(inner).cloned().unwrap_or_else(|| {
			panic!(
				"Op Bug: Node (id:{}) was accessed but is not part of {}",
				inner.id(),
				IterDisplay {
					inner: self.nodes.clone()
				}
			)
		})
	}
}

#[cfg(test)]
mod tests {
	use crate::{grad::grad, graph::Node};
	use ndarray::arr2;

	#[test]
	fn grad_build() {
		let x = Node::new(&[2, 1]).set_name("x");
		let y = Node::new(&[3, 2]).set_name("y");

		let grads = grad(&y, &[&x, &y]).unwrap();

		let dydy = grads.get(&y).unwrap();
		let dydx = grads.get(&x).unwrap();

		assert_eq!(dydy.shape(), y.shape());
		assert_eq!(dydx.shape(), x.shape());

		assert_eq!(dydy.graph(), y.graph());
		assert_eq!(dydx.graph(), x.graph());

		assert_ne!(dydy.graph(), dydx.graph());
	}

	#[test]
	fn grad_names() {
		let x = Node::new(&[2, 1]).set_name("x");
		let y = Node::new(&[3, 2]).set_name("y");

		let grads = grad(&y, &[&x, &y]).unwrap();

		let dydy = grads.get(&y).unwrap();
		let dydx = grads.get(&x).unwrap();

		assert_eq!(&dydy.name(), "d(y)/d(y)");
		assert_eq!(&dydx.name(), "d(y)/d(x)");
	}

	#[test]
	fn self_grad_is_one() {
		// grad() w.r.t. itself should be an array of all ones.
		// Additionally no input value should be required for this.
		let x = Node::new(&[2, 1]).set_name("x");
		let y = Node::new(&[3, 2]).set_name("y");

		let grads = grad(&y, &[&x, &y]).unwrap();

		let dydy = grads.get(&y).unwrap();
		let _dydx = grads.get(&x).unwrap();

		assert_eq!(
			dydy.calc().unwrap(),
			arr2(&[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]).into_dyn()
		);
	}

	#[test]
	fn no_grad_is_zero() {
		// grad() w.r.t. a node that isnt connected in any way should be all zero.
		// Additionally no input value should be required for this.
		let x = Node::new(&[2, 1]).set_name("x");
		let y = Node::new(&[3, 2]).set_name("y");

		let grads = grad(&y, &[&x, &y]).unwrap();

		let _dydy = grads.get(&y).unwrap();
		let dydx = grads.get(&x).unwrap();

		assert_eq!(dydx.calc().unwrap(), arr2(&[[0.0], [0.0]]).into_dyn());
	}
}
