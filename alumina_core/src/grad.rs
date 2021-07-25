//! Types and tools for constructing symbolic gradients.
use crate::{base_ops::{OpSpecification, apply::{Apply}, fill::fill_into, shape_constraint::same_shape}, errors::GradError, graph::{IntoNodeValue, Node, NodeID}, subgraph::{backward_subgraph_from, forward_subgraph_from, SubGraph}, util::display::{Iter2Display, IterDisplay}};
use indexmap::{IndexMap, IndexSet, indexset, indexmap};
use ndarray::{ArcArray, ArrayViewMutD, IxDyn};
use std::{borrow::Borrow, sync::Arc};



enum GradValue {
	Value(ArcArray<f32, IxDyn>),
	Fn(Arc<dyn Fn(ArrayViewMutD<f32>) + Send + Sync + 'static>),
}

/// Builds a gradient constructor that returns a map from the `x` nodes their gradients.
/// The gradient is of the form of `y` w.r.t. `x` returning `dy/dx`, for each. If multiple `y` nodes are used, then the gradient is the sum of the gradient from each.
///
/// ## Validity
/// The gradients returned are only valid for the graph at the time of the call. If further forward pass operations are
/// added after calling `Grad::build(..)`, the respective gradiant operations will be missing.
///
/// # Example
/// ```
/// # use alumina_core::grad::Grad;
/// # use alumina_core::graph::Node;
/// # use ndarray::{arr0, arr1};
/// # use indexmap::indexset;
/// # use failure::Error;
/// # fn main() -> Result<(), Error> {
/// let x = Node::new(&[2]).set_name("x").set_value(arr0(3.0));
/// let y = Node::new(&[3]).set_name("y").set_value(arr0(2.1));
/// let z = Node::new(&[5]).set_name("y").set_value(arr0(4.2));
///
/// let dyd = Grad::of(&y).wrt(&[&x, &y]).build()?;
///
/// assert_eq!(dyd[&x].calc()?, arr1(&[0.0, 0.0]).into_dyn());
/// assert_eq!(dyd[&y].calc()?, arr1(&[1.0, 1.0, 1.0]).into_dyn());
///
/// assert_eq!(dyd.get(&z), None);
/// # Ok(())
/// # }
/// ```
#[must_use]
pub struct Grad {
	ys: IndexSet<Node>,
	xs: IndexSet<Node>,
	grad_values: IndexMap<NodeID, GradValue>,
	include_intermediate: bool,
}

impl Grad {
	/// Begin building the gradient constructor by selective a single dependant value node, i.e. the `y` in `dy/dx`.
	pub fn of(node: impl Into<Node>) -> Self{
		Self {
			ys: indexset![node.into()],
			xs: indexset![],
			grad_values: indexmap![],
			include_intermediate: false,
		}
	}

	/// Begin building the gradient constructor by selective multiple dependant value nodes, i.e. the `y`s in `d y1/dx + dy2/dx + dy3/dx`.
	/// If multiple `y`s are used, the calculated gradients for each x are the summation of the contribution from each y.
	/// Avoid using multiple `y`s where one is a dependant variable of another.
	pub fn of_multi<I, T>(nodes: T) -> Self 
	where
	I: Into<Node>,
	T: IntoIterator<Item = I>,
	{
		Self {
			ys: nodes.into_iter().map(Into::into).collect(),
			xs: indexset![],
			grad_values: indexmap![],
			include_intermediate: false,
		}
	}

	/// Select the independant variables over which the dependant variables will be differentiated.
	/// These will be included in the resuling output map.
	pub fn wrt<I, T>(mut self, nodes: T) -> Self
	where
	I: Into<Node>,
	T: IntoIterator<Item = I>,
	{
		self.xs = nodes.into_iter().map(Into::into).collect();
		self
	}
	
	/// Overwrite the default `1.0` value for the initial gradient of a chosen dependant value node.
	/// The given value will be broadcast to the shape of the node at evaluation time.
	pub fn grad_value(mut self, node: Node, value: impl IntoNodeValue) -> Self {
		assert!(self.ys.contains(&node));
		self.grad_values.insert(node.id(), GradValue::Value(value.into_value()));
		self
	}

	/// Overwrite the default `1.0` value for the initial gradient of a chosen dependant value node.
	/// The given closure is applied to the node at evaluation time.
	pub fn grad_fn<F: Fn(ArrayViewMutD<f32>) + Send + Sync + 'static>(mut self, node: Node, f: F) -> Self {
		assert!(self.ys.contains(&node));
		self.grad_values.insert(node.id(), GradValue::Fn(Arc::new(f)));
		self
	}
	
	/// If `false` the map returned from build only includest the gradients of the w.r.t nodes. If `true` then all gradient nodes added to the graph are returned.
	///
	/// Default: `false`
	pub fn include_intermediate(mut self, value: bool) -> Self {
		self.include_intermediate = value;
		self
	}

	/// Returns a result containing a map from existing nodes to there respective gradient nodes.
	pub fn build(self) -> Result<IndexMap<Node, Node>, GradError> {
		let Grad {
			ys,
			xs,
			grad_values,
			include_intermediate,
		} = self;

		let SubGraph { ops, nodes } = grad_subgraph(ys.clone(), xs.clone());
		let y_ids: IndexSet<NodeID> = ys.iter().map(Node::id).collect();
		let mut context = GradientContext::new(ys, nodes);

		for y_id in &y_ids {
			match grad_values.get(y_id) {
				None => {
					context.grad_of(y_id).set_value(1.0f32.into_value());
				}
				Some(GradValue::Value(v)) => {
					// Set Value if fixed value
					context.grad_of(y_id).set_value(v);
				},
				Some(GradValue::Fn(f)) => {
					Apply::new_boxed(context.grad_of(y_id), f.clone()).build().expect("Failed to add apply op to graph during gradient construction");
				}
			};
		}

	
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
			..
		} = context;
	
		// Set them to fill zero if there are no inputs to the grad
		for (node_inner, grad) in &node_to_grad {
			//if y.id() != *node_inner && grad.parent_ops().is_empty() {
			if !y_ids.contains(node_inner) && grad.parent_ops().is_empty() {
				fill_into(0.0, grad).unwrap_or_else(|err| {
					panic!(
						"Alumina Bug: Error building fill op for gradient of ({}).\n{:#?}",
						nodes.get(node_inner).unwrap(),
						err
					)
				});


			}
			if !grad.shape().is_known() {
				same_shape(nodes.get(node_inner).unwrap(), grad).unwrap_or_else(|err| {
					panic!(
						"Alumina Bug: Error building shape constraint for gradient of ({}).\n{:#?}",
						nodes.get(node_inner).unwrap(),
						err
					)
				});
			}
		}
	
		let result_map = node_to_grad
			.into_iter()
			.map(|(inner, grad)| (nodes.swap_take(&inner).unwrap(), grad))
			.filter(|(n, _ng)| include_intermediate || xs.contains(n))
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
}




/// The subgraph should be the intersection of everything that the xs could affect, and everything that could affect y.
/// With the addition of the xs regardless of whether they can affect y.
fn grad_subgraph(ys: IndexSet<Node>, xs: IndexSet<Node>) -> SubGraph {
	// First work forward and find all nodes that could be affected by the values of xs
	let forward_subgraph = forward_subgraph_from(
		xs.clone(),
		|node| (false, ys.contains(node)),
		|_op| (false, false),
		false,
		false,
	);

	// Second work backward from the y and only include nodes that appeared in the first subgraph
	let mut subgraph = backward_subgraph_from(
		&ys,
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
	ys: IndexSet<Node>,
	y_names: String,
	node_to_grad: IndexMap<NodeID, Node>,
	nodes: IndexSet<Node>,
}

impl GradientContext {
	/// #Panics
	/// panics if the set of y nodes is empty
	fn new(ys: IndexSet<Node>, subgraph_nodes: IndexSet<Node>) -> Self {
		assert!(!ys.is_empty());

		let mut y_names;
		if ys.len() > 1 {
			y_names = "".to_string();
			for (i, y) in ys.iter().enumerate() {
				if i > 0 {
					y_names.push_str(" + ");
				}
				y_names.push_str(&y.name());
			}
		} else {
			y_names = ys.get_index(0).unwrap().name()
		};


		GradientContext {
			ys,
			y_names,
			nodes: subgraph_nodes,
			node_to_grad: IndexMap::new(),
		}
	}

	/// Returns the numerator of the gradient, that is `y` in `dy/dx`.
	///
	/// Useful for automatically constructing meaningful names.
	pub fn ys(&self) -> &IndexSet<Node> {
		&self.ys
	}

	/// This lazily instantiates and returns gradient nodes corresponding to a non-gradient inner.
	pub fn grad_of(&mut self, inner: &NodeID) -> Node {
		let &mut GradientContext {
			ref mut node_to_grad,
			ref nodes,
			ref y_names,
			..
		} = self;

		let node = nodes.get(inner).unwrap_or_else(|| {
			panic!(
				"Op Bug: Node (id:{}) was accessed but is not part of {}",
				inner.id(),
				IterDisplay { inner: nodes.clone() }
			)
		});

		node_to_grad
			.entry(*inner)
			.or_insert_with(|| {
				node.graph()
					.new_node(node.shape())
					.set_name(format!("d({})/d({})", y_names, node.name()))
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
	use crate::{grad::Grad, graph::Node};
	use ndarray::arr2;

	#[test]
	fn grad_build() {
		let x = Node::new(&[2, 1]).set_name("x");
		let y = Node::new(&[3, 2]).set_name("y");

		let grads = Grad::of(&y).wrt(&[&x, &y]).build().unwrap();

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

		let grads = Grad::of(&y).wrt(&[&x, &y]).build().unwrap();

		let dydy = grads.get(&y).unwrap();
		let dydx = grads.get(&x).unwrap();

		assert_eq!(&dydy.name(), "d(y)/d(y)");
		assert_eq!(&dydx.name(), "d(y)/d(x)");
	}

	#[test]
	fn grad_names_multi() {
		let x = Node::new(&[2, 1]).set_name("x");
		let y1 = Node::new(&[3, 2]).set_name("y1");
		let y2 = Node::new(&[3, 2]).set_name("y2");

		let grads = Grad::of_multi(&[&y1, &y2]).wrt(&[&x, &y1, &y2]).build().unwrap();

		let dydy1 = grads.get(&y1).unwrap();
		let dydy2 = grads.get(&y2).unwrap();
		let dydx = grads.get(&x).unwrap();

		assert_eq!(&dydy1.name(), "d(y1 + y2)/d(y1)");
		assert_eq!(&dydy2.name(), "d(y1 + y2)/d(y2)");
		assert_eq!(&dydx.name(), "d(y1 + y2)/d(x)");
	}

	#[test]
	fn self_grad_is_one() {
		// grad() w.r.t. itself should be an array of all ones.
		// Additionally no input value should be required for this.
		let x = Node::new(&[2, 1]).set_name("x");
		let y = Node::new(&[3, 2]).set_name("y");

		let grads = Grad::of(&y).wrt(&[&x, &y]).build().unwrap();

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

		let grads = Grad::of(&y).wrt(&[&x, &y]).build().unwrap();

		let _dydy = grads.get(&y).unwrap();
		let dydx = grads.get(&x).unwrap();

		assert_eq!(dydx.calc().unwrap(), arr2(&[[0.0], [0.0]]).into_dyn());
	}
}
