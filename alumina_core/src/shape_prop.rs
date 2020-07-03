//! Types and tools for shape inference in the graph.
use crate::{
	errors::ShapePropError,
	errors::{ShapeError, ShapesError},
	graph::{Node, NodeInner, Op, WeakNodeInner, WeakOpInner},
	shape::NodeShape,
	subgraph::SubGraph,
	util::display::Iter2Display,
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
		&inputs
			.into_iter()
			.map(|(node, shape)| (node.inner().clone(), shape))
			.collect(),
		use_node_values,
	)?;

	Ok(execution_subgraph
		.nodes
		.iter()
		.map(|node| (node.clone(), inner_map.swap_remove(node.inner()).unwrap()))
		.collect())
}

/// Computes the shapes of the `Node`s in a `SubGraph`, using the `Op`s to propagate from the supplied inputs.
///
/// #Contract
/// Execution subgraph must be topologically sorted
pub fn shapes_inner(
	execution_subgraph: &SubGraph,
	inputs: &IndexMap<NodeInner, IxDyn>,
	use_node_values: bool,
) -> Result<IndexMap<NodeInner, IxDyn>, ShapesError> {
	// Set up initial map of nodes to shapes from node shapes, the inputs, and optionally the shape of node valued
	let (input_errors, mut map) = input_update(&execution_subgraph, inputs, use_node_values);

	if !input_errors.is_empty() {
		return Err(ShapesError::InputCantMerge {
			input_errors: Iter2Display { inner: input_errors },
			partial: map,
		});
	}

	let mut map_completed = IndexMap::with_capacity(execution_subgraph.nodes.len());

	op_update(&execution_subgraph, &mut map, &mut map_completed)?;

	Ok(map
		.into_iter()
		.map(|(node, mut shape)| {
			shape.collapse_dimensions_to_minimum();
			(node.clone(), shape.to_data_shape().unwrap())
		})
		.collect())
}

/// for each node, merge the input shape with the base node shape, and optionally merge to shape of the node value.
fn input_update<'a>(
	execution_subgraph: &'a SubGraph,
	inputs: &IndexMap<NodeInner, IxDyn>,
	use_node_values: bool,
) -> (IndexMap<Node, ShapeError>, IndexMap<NodeInner, NodeShape>) {
	let mut map = IndexMap::with_capacity(execution_subgraph.nodes.len());

	let mut input_errors = IndexMap::new();
	for node in &execution_subgraph.nodes {
		let mut shape = node.shape().clone();

		if let Some(s) = inputs.get(node.inner()) {
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
		map.insert(node.inner().clone(), shape);
	}
	(input_errors, map)
}

/// If there are any ops, then propagate shapes and collect errors
fn op_update<'a>(
	execution_subgraph: &SubGraph,
	map: &'a mut IndexMap<NodeInner, NodeShape>,
	map_completed: &'a mut IndexMap<NodeInner, IxDyn>,
) -> Result<(), ShapesError> {
	for op in execution_subgraph.ops.iter() {
		{
			let mut context = ShapePropContext {
				subgraph: &execution_subgraph,
				map,
				map_completed,

				current_op: op.clone(),
				current_inputs: op.instance().inputs(),
				current_outputs: op.instance().outputs(),
			};

			context.set_next_op(op.clone())?;
			op.instance().propagate_shapes(&mut context)
		}
		.map_err(|e| ShapesError::ShapePropError {
			op: op.clone(),
			error: e,
			partial: map.clone(),
		})?;
	}

	Ok(())
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct ShapeCacheKey {
	subgraph_nodes: Vec<WeakNodeInner>,
	subgraph_ops: Vec<WeakOpInner>,
	input_shapes: Vec<(WeakNodeInner, IxDyn)>,
}

impl ShapeCacheKey {
	fn new(subgraph: &SubGraph, inputs: &IndexMap<NodeInner, IxDyn>, use_node_values: bool) -> Self {
		let mut key = ShapeCacheKey {
			subgraph_nodes: subgraph.nodes.iter().map(|node| node.inner().into()).collect(),
			subgraph_ops: subgraph.ops.iter().map(|op| op.inner().into()).collect(),
			input_shapes: if use_node_values {
				inputs
					.iter()
					.map(|(node, shape)| (node.into(), shape.clone()))
					.chain(
						subgraph
							.nodes
							.iter()
							.filter(|node| !inputs.contains_key(node.inner()))
							.filter_map(|node| node.value_shape().map(|shape| (node.inner().into(), shape))),
					)
					.collect()
			} else {
				inputs
					.iter()
					.map(|(node, shape)| (node.into(), shape.clone()))
					.collect()
			},
		};
		key.subgraph_nodes.sort_unstable_by(|a, b| a.id().cmp(&b.id()));
		key.input_shapes.sort_unstable_by(|a, b| a.0.id().cmp(&b.0.id()));
		key
	}
}

thread_local! {
	static NODE_SHAPE_CACHE: RefCell<LruCache<ShapeCacheKey, IndexMap<NodeInner, IxDyn>>> = RefCell::new(LruCache::new(32));
}

/// Computes the shapes of the `Node`s in a `SubGraph`, using the `Op`s to propagate from the supplied inputs.
///
/// #Contract
/// Execution subgraph must be topologically sorted
pub(crate) fn cached_shapes_inner(
	subgraph: &SubGraph,
	inputs: &IndexMap<NodeInner, IxDyn>,
	use_node_values: bool,
) -> Result<IndexMap<NodeInner, IxDyn>, ShapesError> {
	let result = NODE_SHAPE_CACHE.with(|cache_cell| {
		let mut cache = cache_cell.borrow_mut();

		// Generate a key which has a unique result
		let key = ShapeCacheKey::new(subgraph, inputs, use_node_values);

		// Get a copy of counts from the cache, or insert a new one for this subgraph
		let result: Result<IndexMap<NodeInner, IxDyn>, ShapesError> =
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
	map: &'a mut IndexMap<NodeInner, NodeShape>,
	map_completed: &'a mut IndexMap<NodeInner, IxDyn>,

	current_op: Op,
	current_inputs: IndexSet<NodeInner>,
	current_outputs: IndexSet<NodeInner>,
}

impl<'a> ShapePropContext<'a> {
	/// The subgraph overwhich shape propagation is being run.
	pub fn subgraph(&self) -> &SubGraph {
		self.subgraph
	}

	/// Get the current output shape
	///
	/// If the output is not part of the SubGraph then this will just return the initial shape from the node.
	///
	/// # Panics
	/// If the `Node` which shape is accessed isn't listed as an output to the `Op`, or otherwise isn't included in the
	/// subgraph.
	pub fn output_shape<'b, 'c>(&'a self, node: &'b NodeInner) -> &'c NodeShape
	where
		'a: 'c,
		'b: 'c,
	{
		assert!(
			self.current_outputs.contains(node),
			"Op Bug: Op `{}` attempted to retrieve shape of Node `{}` but it is not listed as an output.",
			self.current_op().name(),
			node.name()
		);

		self.map.get(node).unwrap_or_else(|| node.shape())
	}

	/// Get the shape for the given input node. Type is `IxDyn` as it is now a fixed shape.
	///
	/// # Panics
	/// If the `Node` which shape is accessed isn't listed as an input by the `Op`.
	pub fn input_shape(&self, node: &NodeInner) -> &IxDyn {
		assert!(
			self.current_inputs.contains(node),
			"Op Bug: Op `{}` attempted to retrieve shape of Node `{}` but it is not listed as an input.",
			self.current_op().name(),
			node.name()
		);

		self.map_completed.get(node).unwrap()
	}

	/// If output shape is not part of the graph, this does nothing.
	///
	/// # Panics
	/// Panics if the node isn't listed as an output by the `Op`.
	pub fn merge_output_shape(&mut self, node: &NodeInner, shape: &NodeShape) -> Result<(), ShapePropError> {
		assert!(
			self.current_outputs.contains(node),
			"Op Bug: Op `{}` attempted to update shape of Node `{}` but it is not listed as an output.",
			self.current_op().name(),
			node.name()
		);

		let op = self.current_op();
		if let Some(existing_shape) = self.map.get_mut(node) {
			let new_shape = existing_shape.merge(shape).with_context(|_e| {
				format!(
					"Op `{}` could not merge calculated shape ({}) with existing shape ({}) for Node `{}`.",
					op,
					shape,
					existing_shape,
					node.name(),
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
	pub fn broadcast_merge_output_shape(&mut self, node: &NodeInner, shape: &NodeShape) -> Result<(), ShapePropError> {
		assert!(
			self.current_outputs.contains(node),
			"Op Bug: Op `{}` attempted to update shape of Node `{}` but it is not listed as an output.",
			self.current_op().name(),
			node.name()
		);

		let op = self.current_op();
		if let Some(existing_shape) = self.map.get_mut(node) {
			let new_shape = existing_shape.broadcast_merge(shape).with_context(|_e| {
				// TODO proper error
				format!(
					"Op `{}` could not merge calculated shape ({}) with existing shape ({}) for Node `{}`.",
					op,
					shape,
					existing_shape,
					node.name()
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
	pub fn set_output_shape(&mut self, node: &NodeInner, shape: NodeShape) {
		assert!(
			self.current_outputs.contains(node),
			"Op Bug: Op `{}` attempted to set shape of Node `{}` but it is not listed as an output.",
			self.current_op().name(),
			node.name()
		);

		if self.current_outputs.contains(node) {
			self.map.insert(node.clone(), shape);
		}
	}

	fn set_next_op(&mut self, op: Op) -> Result<(), ShapesError> {
		{
			let instance = op.instance();
			self.current_inputs = instance.inputs();
			self.current_outputs = instance.outputs();
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
				self.map_completed.insert(node.clone(), shape.to_data_shape().unwrap());
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
