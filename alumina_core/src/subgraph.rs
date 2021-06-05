//! Types and tools extracting a portion of a graph.
//!
//! This is used to minimise work in execution and symbolic differentiation.

use indexmap::{IndexMap, IndexSet};
use std::{
	borrow::Borrow,
	collections::{BinaryHeap, VecDeque},
	fmt::{Debug, Display, Formatter},
	hash::{Hash, Hasher},
};

use crate::{
	errors::{CyclicGraphError, ExecutionSubgraphError},
	graph::{Graph, Node, Op},
	util::display::{IterDebug, IterDisplay},
};

/// A collection of `Node`s and `Op`s which may come from a single or multiple `Graph`s.
///
/// This exposes some methods for sorting/reordering items and counting inputs/outputs for each item.
///
/// Typically this type is constructed via the methods in the same module.
#[derive(Clone)]
pub struct SubGraph {
	pub nodes: IndexSet<Node>,
	pub ops: IndexSet<Op>,
}

impl SubGraph {
	pub fn new<N, O>(nodes: IndexSet<N>, ops: IndexSet<O>) -> Self
	where
		N: Borrow<Node> + Hash + Eq,
		O: Borrow<Op> + Hash + Eq,
	{
		SubGraph {
			nodes: nodes.into_iter().map(|n| n.borrow().clone()).collect(),
			ops: ops.into_iter().map(|o| o.borrow().clone()).collect(),
		}
	}

	/// Return the set of `Graphs` which the `SubGraph`s `Op`s and `Node`s are members of.
	pub fn graphs(&self) -> IndexSet<Graph> {
		self.nodes
			.iter()
			.map(Node::graph)
			.chain(self.ops.iter().map(Op::graph))
			.cloned()
			.collect()
	}

	/// Create a deep cloned graph of the nodes and ops referenced by the `SubGraph`.
	pub fn into_graph() -> Graph {
		unimplemented!()
	}

	/// Return a `SubGraph` with `Op`s and `Node`s in same order that they were added to their current `Graph`.
	///
	/// In the case of multiple graphs, items are only strictly ordered with respect to items of the same `Graph`.
	pub fn graph_order(&self) -> SubGraph {
		let mut node_index: IndexSet<Node> = self.nodes.iter().cloned().collect();
		let mut op_index: IndexSet<Op> = self.ops.iter().cloned().collect();
		node_index.sort();
		op_index.sort();

		SubGraph {
			nodes: node_index,
			ops: op_index,
		}
	}

	/// Return the order which is closest the graph order while still topologically sorted (necessary for execution).
	///
	/// In the case of multiple graphs, items are only strictly ordered with respect to items of the same `Graph`.
	///
	/// That is `Op`s and `Node`s are ordered as in the `Graph` except where one must be delayed until all its inputs
	/// are ready.
	pub fn execution_order(&self) -> Result<SubGraph, CyclicGraphError> {
		let (mut node_remaining, mut op_remaining) = self.input_counts();

		let mut op_queue: BinaryHeap<Op> = op_remaining
			.iter()
			.filter_map(|(op, i)| if *i == 0 { Some((*op).clone()) } else { None })
			.collect();
		let mut node_queue: BinaryHeap<Node> = node_remaining
			.iter()
			.filter_map(|(node, i)| if *i == 0 { Some((*node).clone()) } else { None })
			.collect();

		let mut node_order = IndexSet::with_capacity(self.nodes.len());
		let mut op_order = IndexSet::with_capacity(self.ops.len());

		loop {
			let mut did_something = false;

			// Process as many nodes as possible
			while let Some(node) = node_queue.pop() {
				if node_order.contains(&node) {
					panic!("should only be added by the last parent op?")
				}

				for child_op in node.child_ops() {
					if let Some(count) = op_remaining.get_mut(&child_op) {
						*count -= 1;

						if *count == 0 {
							op_queue.push(child_op);
						}
					}
				}
				node_order.insert(node);
				did_something = true;
			}

			// Only process a single op before reattempting more nodes
			if let Some(op) = op_queue.pop() {
				if op_order.contains(&op) {
					panic!("should only be added by the last parent node?")
				}

				for child_node in op.child_nodes() {
					if let Some(count) = node_remaining.get_mut(&child_node) {
						*count -= 1;

						if *count == 0 {
							node_queue.push(child_node);
						}
					}
				}
				op_order.insert(op);
				did_something = true;
			}

			if !did_something {
				break;
			}
		}

		if node_remaining.values().any(|&i| i > 0) && op_remaining.values().any(|&i| i > 0) {
			return Err(CyclicGraphError {
				unsorted_nodes: IterDisplay {
					inner: node_remaining
						.into_iter()
						.filter(|&(_, count)| count > 0)
						.map(|(node, _)| node.clone())
						.collect(),
				},
				unsorted_ops: IterDisplay {
					inner: op_remaining
						.into_iter()
						.filter(|&(_, count)| count > 0)
						.map(|(op, _)| op.clone())
						.collect(),
				},
			});
		}

		debug_assert_eq!(self.nodes.len(), node_order.len());
		debug_assert_eq!(self.ops.len(), op_order.len());

		Ok(SubGraph {
			nodes: node_order,
			ops: op_order,
		})
	}

	/// Returns the number of inputs to each `Op` and `Node`, taking only members of the `SubGraph` into account.
	pub fn input_counts(&self) -> (IndexMap<&Node, usize>, IndexMap<&Op, usize>) {
		let mut node_remaining: IndexMap<_, _> = self.nodes.iter().map(|n| (n, 0)).collect();
		let mut op_remaining: IndexMap<_, _> = self.ops.iter().map(|o| (o, 0)).collect();

		for node in &self.nodes {
			for child in node.child_ops() {
				if let Some(count) = op_remaining.get_mut(&child) {
					*count += 1;
				}
			}
		}

		for op in &self.ops {
			for child in op.child_nodes() {
				if let Some(count) = node_remaining.get_mut(&child) {
					*count += 1;
				}
			}
		}

		(node_remaining, op_remaining)
	}

	/// Returns the number of outputs from each `Op` and `Node`, taking only members of the `SubGraph` into account.
	pub fn output_counts(&self) -> (IndexMap<&Node, usize>, IndexMap<&Op, usize>) {
		let mut node_remaining: IndexMap<_, _> = self.nodes.iter().map(|n| (n, 0)).collect();
		let mut op_remaining: IndexMap<_, _> = self.ops.iter().map(|o| (o, 0)).collect();

		for node in &self.nodes {
			for parent in node.parent_ops() {
				if let Some(count) = op_remaining.get_mut(&parent) {
					*count += 1;
				}
			}
		}

		for op in &self.ops {
			for parent in op.parent_nodes() {
				if let Some(count) = node_remaining.get_mut(&parent) {
					*count += 1;
				}
			}
		}

		(node_remaining, op_remaining)
	}
}

impl Display for SubGraph {
	fn fmt(&self, f: &mut Formatter) -> ::std::fmt::Result {
		write!(f, "SubGraph {{ nodes: [")?;
		let mut nodes = self.nodes.iter();
		if let Some(d) = nodes.next() {
			write!(f, "{}", d)?;
			for d in nodes {
				write!(f, ", {}", d)?;
			}
		}
		write!(f, "], ops: [")?;
		let mut ops = self.ops.iter();
		if let Some(d) = ops.next() {
			write!(f, "{}", d)?;
			for d in ops {
				write!(f, ", {}", d)?;
			}
		}
		write!(f, "] }}")?;
		Ok(())
	}
}

impl Debug for SubGraph {
	fn fmt(&self, fmt: &mut Formatter) -> ::std::fmt::Result {
		fmt.debug_struct("Graph")
			.field(
				"nodes",
				&IterDebug {
					inner: self.nodes.clone(),
				},
			)
			.field(
				"ops",
				&IterDebug {
					inner: self.ops.clone(),
				},
			)
			.finish()
	}
}

/// Implemented so as to be order dependent for both `Op`s and `Node`s
impl PartialEq for SubGraph {
	fn eq(&self, other: &SubGraph) -> bool {
		self.nodes.len() == other.nodes.len()
			&& self.ops.len() == other.ops.len()
			&& self.nodes.iter().zip(&other.nodes).all(|(a, b)| a.id() == b.id())
			&& self.ops.iter().zip(&other.ops).all(|(a, b)| a.id() == b.id())
	}
}
impl Eq for SubGraph {}

/// Implemented as to be order dependent for both `Op`s and `Node`s
impl Hash for SubGraph {
	fn hash<H: Hasher>(&self, state: &mut H) {
		for node in &self.nodes {
			node.id().hash(state)
		}
		for op in &self.ops {
			op.id().hash(state)
		}
	}
}

/// Extract `SubGraph` suitable for execution of outputs from inputs.
///
/// Works backward from outputs to inputs fanning out.
/// Returns of the set of nodes and the set of Ops that are ancestors to the `outputs` set
/// stopping at nodes included in `inputs` (and optionally, at nodes which have a value set).
///
/// The returned `SubGraph` is topologically sorted into execution order.
///
/// Returns an error if the subgraph contains nodes that cannot be calculated as they
/// are not inputs, and do not have parents.
///
///  * `outputs` - The set of `Node`s which must be computed by the `SubGraph`.
///
///  * `inputs` - The set of available input `Node`s.
///
///  * `use_node_values` - If true then `Node`s with values are treated as inputs.
pub fn execution_subgraph<I, O, T1, T2>(
	inputs: T1,
	outputs: T2,
	use_node_values: bool,
) -> Result<SubGraph, ExecutionSubgraphError>
where
	I: Into<Node>,
	O: Into<Node>,
	T1: IntoIterator<Item = I>,
	T2: IntoIterator<Item = O>,
{
	let inputs: IndexSet<_> = inputs.into_iter().map(Into::into).collect();
	let mut parentless_nodes = vec![];

	let subgraph = backward_subgraph_from(
		outputs,
		|node: &Node| {
			// Exclude the parents of nodes which are inputs
			// Record if a node has no parents and is not an input
			if inputs.contains(node) || (use_node_values && node.has_value()) {
				(false, true) // exclude parents of node
			} else if node.parent_ops().is_empty() {
				parentless_nodes.push(node.clone());
				(false, true) // exclude parents of node
			} else {
				(false, false) // exclude nothing
			}
		},
		|_op| (false, false),
		false,
		false,
	);

	if !parentless_nodes.is_empty() {
		Err(ExecutionSubgraphError::InsufficientInputs {
			parentless_nodes: IterDisplay {
				inner: parentless_nodes,
			},
			subgraph_nodes: IterDisplay { inner: subgraph.nodes },
			subgraph_ops: IterDisplay { inner: subgraph.ops },
		})
	} else {
		let ordered_subgraph = subgraph
			.execution_order()
			.map_err(|cycle| ExecutionSubgraphError::Cycle {
				error: cycle,
				subgraph_nodes: IterDisplay { inner: subgraph.nodes },
				subgraph_ops: IterDisplay { inner: subgraph.ops },
			})?;
		Ok(ordered_subgraph)
	}
}

/// Extract a `SubGraph` by stepping through the directed graph in the forward direction.
///
/// The returned `SubGraph` may have any order.
///
///  * `start` - Initial set of nodes queued for traversal from.
///
///  * `node_stop_cond` - Evaluated for each `Node`. If it returns `(true, _)` then the `Node` is excluded. If it
/// returns `(_, true)`, then no attempt is made to include the children of the `Node`.
///
///  * `op_stop_cond` - Evaluated
/// for each `Op`. If it returns `(true, _)` then the `Op` is excluded. If it returns `(_, true)`, then no attempt is
/// made to include the children of the `Op`.
///
///  * `exclude_partial_nodes` - If true, all the parents of candidate
/// `Node`s must be included otherwise the `Node` is not processed.
///
///  * `exclude_partial_ops` - If true, all the parents
/// of candidate `Op`s must be included otherwise the `Op` is not processed.
pub fn forward_subgraph_from<N, T, F1: FnMut(&Node) -> (bool, bool), F2: FnMut(&Op) -> (bool, bool)>(
	start: T,
	mut node_stop_cond: F1,
	mut op_stop_cond: F2,
	exclude_partial_nodes: bool,
	exclude_partial_ops: bool,
) -> SubGraph
where
	N: Into<Node>,
	T: IntoIterator<Item = N>,
{
	let mut nodes = IndexSet::new();
	let mut ops = IndexSet::new();

	let mut node_queue = VecDeque::new();
	node_queue.extend(start.into_iter().map(Into::into));

	while let Some(node) = node_queue.pop_front() {
		if nodes.contains(&node) || (exclude_partial_nodes && node.parent_ops().iter().any(|op| !ops.contains(op))) {
			continue;
		}

		// process only if it hasn't been already
		let (exclude_node, exclude_node_children) = node_stop_cond(&node);
		if !exclude_node_children {
			for op in node.child_ops() {
				if ops.contains(&op)
					|| (exclude_partial_ops && op.parent_nodes().iter().any(|node| !nodes.contains(node)))
				{
					continue;
				}

				let (exclude_op, exclude_op_children) = op_stop_cond(&op);
				if !exclude_op_children {
					node_queue.extend(op.child_nodes());
				}
				if !exclude_op {
					ops.insert(op);
				}
			}
		}
		if !exclude_node {
			nodes.insert(node);
		}
	}
	SubGraph { nodes, ops }
}

/// Extract a `SubGraph` by stepping through the directed graph in the backward direction.
///
/// The returned `SubGraph` may have any order.
///
///  * `start` - Initial set of nodes queued for traversal from.
///
///  * `node_stop_cond` - Evaluated for each `Node`. If it returns `(true, _)` then the `Node` is excluded. If it
/// returns `(_, true)`, then no attempt is made to include the parents of the `Node`.
///
///  * `op_stop_cond` - Evaluated for each `Op`. If it returns `(true, _)` then the `Op` is excluded. If it returns
/// `(_, true)`, then no attempt is made to include the parents of the `Op`.
///
///  * `exclude_partial_nodes` - If true, all the children of candidate `Node`s must be included otherwise the `Node`
/// is not processed.
///
///  * `exclude_partial_ops` - If true, all the children of candidate `Op`s must be included
/// otherwise the `Op` is not processed.
pub fn backward_subgraph_from<N, T, F1: FnMut(&Node) -> (bool, bool), F2: FnMut(&Op) -> (bool, bool)>(
	start: T,
	mut node_stop_cond: F1,
	mut op_stop_cond: F2,
	exclude_partial_nodes: bool,
	exclude_partial_ops: bool,
) -> SubGraph
where
	N: Into<Node>,
	T: IntoIterator<Item = N>,
{
	let mut nodes = IndexSet::new();
	let mut ops = IndexSet::new();

	let mut node_queue = VecDeque::<Node>::new();
	node_queue.extend(start.into_iter().map(Into::into));

	while let Some(node) = node_queue.pop_front() {
		if nodes.contains(&node) || (exclude_partial_nodes && node.child_ops().iter().any(|op| !ops.contains(op))) {
			continue;
		}

		// process only if it hasn't been already
		let (exclude_node, exclude_node_parents) = node_stop_cond(&node);
		if !exclude_node_parents {
			for op in node.parent_ops() {
				if ops.contains(&op)
					|| (exclude_partial_ops && op.child_nodes().iter().any(|node| !nodes.contains(node)))
				{
					continue;
				}

				let (exclude_op, exclude_op_parents) = op_stop_cond(&op);
				if !exclude_op_parents {
					node_queue.extend(op.parent_nodes());
				}
				if !exclude_op {
					ops.insert(op);
				}
			}
		}
		if !exclude_node {
			nodes.insert(node);
		}
	}
	SubGraph { nodes, ops }
}
