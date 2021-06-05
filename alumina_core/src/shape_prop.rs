//! Types and tools for shape inference in the graph.
use crate::{
	errors::ShapePropError,
	errors::{ShapeError, ShapesError},
	graph::{Node, NodeID, Op, OpID},
	shape::NodeShape,
	subgraph::SubGraph,
	util::display::{Iter2Display, IterDisplay},
};
use failure::ResultExt;
use indexmap::{IndexMap, IndexSet};
use lru::LruCache;
use ndarray::{Dimension, IxDyn};
use std::cell::RefCell;

/// Computes the shapes of the `Node`s in a `SubGraph`, using the `Op`s to propagate from the supplied inputs.
pub fn shapes(
	execution_subgraph: &SubGraph,
	inputs: IndexMap<Node, IxDyn>,
	use_node_values: bool,
) -> Result<IndexMap<Node, IxDyn>, ShapesError> {
	let mut inner_map = shapes_inner(
		execution_subgraph,
		&inputs.into_iter().map(|(node, shape)| (node.id(), shape)).collect(),
		use_node_values,
	)?;

	Ok(execution_subgraph
		.nodes
		.iter()
		.map(|node| (node.clone(), inner_map.swap_remove(&node.id()).unwrap()))
		.collect())
}

/// Computes the shapes of the `Node`s in a `SubGraph`, using the `Op`s to propagate from the supplied inputs.
///
/// # Contract
/// Execution subgraph must be topologically sorted
pub fn shapes_inner(
	execution_subgraph: &SubGraph,
	inputs: &IndexMap<NodeID, IxDyn>,
	use_node_values: bool,
) -> Result<IndexMap<NodeID, IxDyn>, ShapesError> {
	// Set up initial map of nodes to shapes from node shapes, the inputs, and optionally the shape of node valued
	let (input_errors, mut map) = input_update(&execution_subgraph, inputs, use_node_values);

	if !input_errors.is_empty() {
		return Err(ShapesError::InputCantMerge {
			input_errors: Iter2Display { inner: input_errors },
			partial: map
				.iter()
				.map(|(k, v)| (execution_subgraph.nodes.get(k).unwrap().clone(), v.clone()))
				.collect(),
		});
	}

	let mut map_completed = IndexMap::with_capacity(execution_subgraph.nodes.len());

	op_update(&execution_subgraph, &mut map, &mut map_completed)?;

	Ok(map
		.into_iter()
		.map(|(node, mut shape)| {
			shape.collapse_dimensions_to_minimum();
			(node, shape.to_data_shape().unwrap())
		})
		.collect())
}

/// for each node, merge the input shape with the base node shape, and optionally merge to shape of the node value.
fn input_update(
	execution_subgraph: &SubGraph,
	inputs: &IndexMap<NodeID, IxDyn>,
	use_node_values: bool,
) -> (IndexMap<Node, ShapeError>, IndexMap<NodeID, NodeShape>) {
	let mut map = IndexMap::with_capacity(execution_subgraph.nodes.len());

	let mut input_errors = IndexMap::new();
	for node in &execution_subgraph.nodes {
		let mut shape = node.shape().clone();

		if let Some(s) = inputs.get(&node.id()) {
			match shape.broadcast_merge(&s.slice().into()) {
				// TODO should inputs be broadcast?
				Ok(s) => shape = s,
				Err(e) => {
					input_errors.insert(node.clone(), e);
				}
			}
		} else if use_node_values {
			if let Some(s) = node.value_shape() {
				let s: NodeShape = s.slice().into();
				shape = shape.broadcast_merge(&s).unwrap_or_else(|err| {
					panic!("Alumina Bug: Node `{}` has a value with a shape ({}) that cannot be broadcast_merged into its shape ({}).\n{:#?}", node.name(), s, shape, err)
				});
			}
		}
		map.insert(node.id(), shape);
	}
	(input_errors, map)
}

/// If there are any ops, then propagate shapes and collect errors
fn op_update<'a>(
	execution_subgraph: &SubGraph,
	map: &'a mut IndexMap<NodeID, NodeShape>,
	map_completed: &'a mut IndexMap<NodeID, IxDyn>,
) -> Result<(), ShapesError> {
	for op in execution_subgraph.ops.iter() {
		{
			let mut context = ShapePropContext {
				subgraph: &execution_subgraph,
				map,
				map_completed,

				current_op: op.clone(),
				current_inputs: op.parent_nodes(),
				current_outputs: op.child_nodes(),
			};

			context.set_next_op(op.clone())?;
			op.instance().propagate_shapes(&mut context)
		}
		.map_err(|e| ShapesError::ShapePropError {
			op: op.clone(),
			error: e,
			partial: map
				.iter()
				.map(|(k, v)| (execution_subgraph.nodes.get(k).unwrap().clone(), v.clone()))
				.collect(),
		})?;
	}

	Ok(())
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct ShapeCacheKey {
	subgraph_nodes: Vec<NodeID>,
	subgraph_ops: Vec<OpID>,
	input_shapes: Vec<(NodeID, IxDyn)>,
}

impl ShapeCacheKey {
	fn new(subgraph: &SubGraph, inputs: &IndexMap<NodeID, IxDyn>, use_node_values: bool) -> Self {
		let mut key = ShapeCacheKey {
			subgraph_nodes: subgraph.nodes.iter().map(|node| node.id()).collect(),
			subgraph_ops: subgraph.ops.iter().map(|op| op.id()).collect(),
			input_shapes: if use_node_values {
				inputs
					.iter()
					.map(|(node, shape)| (*node, shape.clone()))
					.chain(
						subgraph
							.nodes
							.iter()
							.filter(|node| !inputs.contains_key(&node.id()))
							.filter_map(|node| node.value_shape().map(|shape| (node.id(), shape))),
					)
					.collect()
			} else {
				inputs.iter().map(|(node, shape)| (*node, shape.clone())).collect()
			},
		};
		key.subgraph_nodes.sort_unstable_by_key(|a| a.id());
		key.input_shapes.sort_unstable_by(|a, b| a.0.id().cmp(&b.0.id()));
		key
	}
}

thread_local! {
	static NODE_SHAPE_CACHE: RefCell<LruCache<ShapeCacheKey, IndexMap<NodeID, IxDyn>>> = RefCell::new(LruCache::new(32));
}

/// Computes the shapes of the `Node`s in a `SubGraph`, using the `Op`s to propagate from the supplied inputs.
///
/// #Contract
/// Execution subgraph must be topologically sorted
pub(crate) fn cached_shapes_inner(
	subgraph: &SubGraph,
	inputs: &IndexMap<NodeID, IxDyn>,
	use_node_values: bool,
) -> Result<IndexMap<NodeID, IxDyn>, ShapesError> {
	let result = NODE_SHAPE_CACHE.with(|cache_cell| {
		let mut cache = cache_cell.borrow_mut();

		// Generate a key which has a unique result
		let key = ShapeCacheKey::new(subgraph, inputs, use_node_values);

		// Get a copy of counts from the cache, or insert a new one for this subgraph
		let result: Result<IndexMap<NodeID, IxDyn>, ShapesError> =
			cache.get(&key).cloned().map(Ok).unwrap_or_else(|| {
				let shape_map = shapes_inner(&subgraph, &inputs, use_node_values)?;

				cache.put(key.clone(), shape_map.clone());

				Ok(shape_map)
			});

		// Return
		result
	});

	debug_assert!(
		match (&result, &shapes_inner(&subgraph, &inputs, use_node_values)) {
			(&Err(_), &Err(_)) => true,
			(&Ok(ref m1), &Ok(ref m2)) => m1 == m2,
			(_, _) => false,
		},
		"cached shape did not match shapes_inner result"
	);

	result
}

/// This context is supplied to `Op`s during shape propagation.
///
/// Only worry about this when implementing new `Op`s.
pub struct ShapePropContext<'a> {
	subgraph: &'a SubGraph,
	map: &'a mut IndexMap<NodeID, NodeShape>,
	map_completed: &'a mut IndexMap<NodeID, IxDyn>,

	current_op: Op,
	current_inputs: IndexSet<Node>,
	current_outputs: IndexSet<Node>,
}

impl<'a> ShapePropContext<'a> {
	/// The subgraph overwhich shape propagation is being run.
	pub fn subgraph(&self) -> &SubGraph {
		self.subgraph
	}

	/// Returns the full node for an inner.
	pub fn node(&self, inner: &NodeID) -> Node {
		self.subgraph.nodes.get(inner).cloned().unwrap_or_else(|| {
			panic!(
				"Op Bug: Node (id:{}) was accessed but is not part of {}",
				inner.id(),
				IterDisplay {
					inner: self.subgraph.nodes.clone()
				}
			)
		})
	}

	/// Get the current output shape
	///
	/// # Panics
	/// If the `Node` which shape is accessed isn't listed as an output to the `Op`, or otherwise isn't included in the
	/// subgraph.
	/// If the output is not part of the SubGraph for any other reason then this will also panic.
	pub fn output_shape<'b, 'c>(&'a self, node_id: &'b NodeID) -> NodeShape
	where
		'a: 'c,
		'b: 'c,
	{
		if let Some((_, node)) = self.current_outputs.get_full(node_id) {
			self.map.get(node_id).cloned().unwrap_or_else(|| node.shape())
		} else {
			panic!(
				"Op Bug: Op `{}` attempted to retrieve shape of Node (id:{}) but it is not listed as an output.",
				self.current_op().name(),
				node_id.id()
			);
		}
	}

	/// Get the shape for the given input node. Type is `IxDyn` as it is now a fixed shape.
	///
	/// # Panics
	/// If the `Node` which shape is accessed isn't listed as an input by the `Op`.
	pub fn input_shape(&self, node: &NodeID) -> &IxDyn {
		assert!(
			self.current_inputs.contains(node),
			"Op Bug: Op `{}` attempted to retrieve shape of Node (id:{}) but it is not listed as an input.",
			self.current_op().name(),
			node.id()
		);

		self.map_completed.get(node).unwrap_or_else(|| {
			panic!(
				"Op Bug: Op `{}` attempted to retrieve shape of Node (id:{}) but it is not part of the subgraph.",
				self.current_op().name(),
				node.id()
			)
		})
	}

	/// If output shape is not part of the graph, this does nothing.
	///
	/// # Panics
	/// Panics if the node isn't listed as an output by the `Op`.
	pub fn merge_output_shape(&mut self, node: &NodeID, shape: &NodeShape) -> Result<(), ShapePropError> {
		assert!(
			self.current_outputs.contains(node),
			"Op Bug: Op `{}` attempted to update shape of Node (id:{}) but it is not listed as an output.",
			self.current_op().name(),
			node.id()
		);

		let op = self.current_op();
		let map = &mut self.map;
		let subgraph = &self.subgraph;
		if let Some((_, node, existing_shape)) = map.get_full_mut(node) {
			let new_shape = existing_shape.merge(shape).with_context(|_e| {
				format!(
					"Op `{}` could not merge calculated shape ({}) with existing shape ({}) for Node `{}`.",
					op,
					shape,
					existing_shape,
					subgraph
						.nodes
						.get(node)
						.expect("all nodes in map must be in subgraph")
						.name(),
				)
			})?;
			*existing_shape = new_shape;
		}
		Ok(())
	}

	/// If output shape is not part of the graph, this does nothing.
	///
	/// # Panics
	/// Panics if the node isn't listed as an output by the `Op`.
	pub fn broadcast_merge_output_shape(&mut self, node: &NodeID, shape: &NodeShape) -> Result<(), ShapePropError> {
		assert!(
			self.current_outputs.contains(node),
			"Op Bug: Op `{}` attempted to update shape of Node (id:{}) but it is not listed as an output.",
			self.current_op().name(),
			node.id()
		);

		let op = self.current_op();
		let map = &mut self.map;
		let subgraph = &self.subgraph;
		if let Some((_, node, existing_shape)) = map.get_full_mut(node) {
			let new_shape = existing_shape.broadcast_merge(shape).with_context(|_e| {
				// TODO proper error
				format!(
					"Op `{}` could not merge calculated shape ({}) with existing shape ({}) for Node `{}`.",
					op,
					shape,
					existing_shape,
					subgraph
						.nodes
						.get(node)
						.expect("all nodes in map must be in subgraph")
						.name(),
				)
			})?;
			*existing_shape = new_shape;
		}
		Ok(())
	}

	/// Set the shape of an output.
	///
	/// Should generally be avoided, and used only as a last resort, as it may potentially break shape inference for
	/// another `Op`.
	///
	/// If output is not part of the subgraph, this does nothing.
	pub fn set_output_shape(&mut self, node: &NodeID, shape: NodeShape) {
		assert!(
			self.current_outputs.contains(node),
			"Op Bug: Op `{}` attempted to set shape of Node (id:{}) but it is not listed as an output.",
			self.current_op().name(),
			node.id()
		);

		if let Some((_, node)) = self.current_outputs.get_full(node) {
			self.map.insert(node.id(), shape);
		}
	}

	fn set_next_op(&mut self, op: Op) -> Result<(), ShapesError> {
		{
			self.current_inputs = op.parent_nodes();
			self.current_outputs = op.child_nodes();
		}
		self.current_op = op;

		for node in &self.current_inputs {
			// return error if not all inputs are in the subgraph
			if !self.map.contains_key(node) {
				return Err(ShapesError::OpInputNotInSubgraph {
					op: self.current_op.clone(),
					input_node: node.clone(),
				});
			}

			// add all inputs to map_completed
			if !self.map_completed.contains_key(node) {
				let shape = self.map.get_mut(node).unwrap();
				shape.collapse_dimensions_to_minimum();
				self.map_completed.insert(node.id(), shape.to_data_shape().unwrap());
			}
		}

		// loop over outputs and confirm that they are not already completed
		for node in &self.current_outputs {
			if self.map_completed.contains_key(node) {
				return Err(ShapesError::SubGraphNotExecutable { node: node.clone() });
			}
		}

		Ok(())
	}

	pub fn current_op(&self) -> Op {
		self.current_op.clone()
	}
}
