//! `Graph`, `Node`, and `Op` form the computational graph.
//!
//! The primary datastructure of this library is a bipartite graph of alternating tensors and operations, or
//! alternately a hypergraph where operations are hyperedges. The `Node` type is a handle to a tensor in the graph, and
//! the `Op` type is a handle to an operation. Each `Op` and `Node` can have multiple parents and children, with `Op`s
//! reading from their parents and writing to their children. A Node with multiple Ops writing to it will have a value
//! equal to the addition of their outputs.
//!
//! The `Graph` type is a mostly behind-the-scenes record of what connects to what, and what is tagged or named what.
//! When Nodes are created with Node::new(..) they exist in their own Graph until
//! an Op is created and the graphs of all the Nodes it uses are merged.
//!
//!
//! ```rust
//! # use alumina::graph::Node;
//! # use std::hash::{Hash, Hasher, SipHasher};
//!
//! let fixed_2d_node = Node::new(&[4, 3]);
//! let variable_2d_node = Node::new(&[-1, 3]);
//!
//! assert_ne!(fixed_2d_node.graph(), variable_2d_node.graph());
//! assert_ne!(hash(fixed_2d_node.graph()), hash(variable_2d_node.graph()));
//!
//! fixed_2d_node.graph().merge(variable_2d_node.graph());
//!
//! assert_eq!(fixed_2d_node.graph(), variable_2d_node.graph());
//! assert_eq!(hash(fixed_2d_node.graph()), hash(variable_2d_node.graph()));
//!
//! # fn hash<T: Hash>(val: T) -> u64 {
//! #     let mut hasher = SipHasher::new();
//! #     val.hash(&mut hasher);
//! #     return hasher.finish()
//! # }
//! ```

use crate::{
	base_ops::OpInstance,
	errors::ExecError,
	exec::{exec, ExecConfig},
	init::Initialiser,
	shape::NodeShape,
	util::display::IterDebug,
};
use indexmap::{Equivalent, IndexMap, IndexSet};
use ndarray::{arr0, arr1, ArcArray, ArrayBase, ArrayD, Data, Dimension, IxDyn, OwnedArcRepr, OwnedRepr, ViewRepr};
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
	borrow::Borrow,
	collections::BTreeMap,
	fmt::{self, Debug, Display},
	hash::{Hash, Hasher},
	ops::Deref,
	sync::{atomic::AtomicU64, atomic::Ordering, Arc},
};

// TODO write explanation for all the ways of creating a Node
pub trait IntoNodeValue {
	fn into_value(self) -> ArcArray<f32, IxDyn>;
}

impl IntoNodeValue for f32 {
	fn into_value(self) -> ArcArray<f32, IxDyn> {
		arr0(self).into_shared().into_dyn()
	}
}

impl IntoNodeValue for &[f32] {
	fn into_value(self) -> ArcArray<f32, IxDyn> {
		arr1(self).into_shared().into_dyn()
	}
}

impl IntoNodeValue for Vec<f32> {
	fn into_value(self) -> ArcArray<f32, IxDyn> {
		ArcArray::from(self).into_dyn()
	}
}

impl<D: Dimension> IntoNodeValue for ArrayBase<ViewRepr<&f32>, D> {
	fn into_value(self) -> ArcArray<f32, IxDyn> {
		self.into_owned().into_shared().into_dyn()
	}
}

impl<D: Dimension> IntoNodeValue for ArrayBase<ViewRepr<&mut f32>, D> {
	fn into_value(self) -> ArcArray<f32, IxDyn> {
		self.into_owned().into_shared().into_dyn()
	}
}

impl<D: Dimension> IntoNodeValue for ArrayBase<OwnedRepr<f32>, D> {
	fn into_value(self) -> ArcArray<f32, IxDyn> {
		self.into_shared().into_dyn()
	}
}

impl<D: Dimension> IntoNodeValue for ArrayBase<OwnedArcRepr<f32>, D> {
	fn into_value(self) -> ArcArray<f32, IxDyn> {
		self.into_dyn()
	}
}

impl<S: Data<Elem = f32>, D: Dimension> IntoNodeValue for &ArrayBase<S, D> {
	fn into_value(self) -> ArcArray<f32, IxDyn> {
		self.to_owned().into_shared().into_dyn()
	}
}

/// `Node` is an Arc-like handle to a tensor in the computational graph.
///
/// The `Node` may or may not have:
///  * a unique name
///  * an assigned value
///  * marked with a `NodeTag::Parameter`
///
/// This type allows access to node specifics and the Graph
#[derive(Clone)]
pub struct Node {
	graph: Graph,
	data: Arc<Mutex<NodeInnerData>>,
	id: NodeID,
}

impl Node {
	/// Remove this `Node` from its `Graph`
	///
	/// # Panics
	/// Panics if any other `Node`s ists for this.
	/// If the Node is still referenced by Ops in the graph then this will panic in debug mode, and may panic in Release.
	pub fn remove(self) -> NodeInnerData {
		let Node { graph, id, data } = self;
		drop(data);
		graph.remove_node(id)
	}

	/// Create a node with no associated value.
	///
	/// This node belongs to its own graph until joined by operations.
	/// ```
	/// # use alumina::graph::Node;
	/// # use alumina::shape::NodeAxis;
	///
	/// let fixed_2d_node = Node::new(&[4, 3]);
	/// let variable_2d_node = Node::new(&[-1, 3]);
	/// let verbose_node = Node::new(&[NodeAxis::unknown(), NodeAxis::known(3)]);
	/// let copy_shape_node = Node::new(verbose_node.shape());
	/// ```
	pub fn new<S: Into<NodeShape>>(shape: S) -> Self {
		let g = Graph::new();
		g.new_node(shape.into())
	}

	/// Create a new node with a value and fixed shape determined buy the provided value.
	///
	/// ```
	/// # extern crate ndarray;
	/// # extern crate alumina;
	/// # use ndarray::{arr0, arr2};
	/// # use alumina::graph::Node;
	///
	/// let value_node1 = Node::from_value(arr0(5.0));
	/// let value_node2 = Node::from_value(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
	/// ```
	pub fn from_value<V: IntoNodeValue>(value: V) -> Self {
		let value = value.into_value();
		Node::new(value.shape()).set_value(value)
	}

	/// Call `exec()` for this node only.
	pub fn calc(&self) -> Result<ArrayD<f32>, ExecError> {
		Ok(exec(IndexMap::<Node, _>::new(), &[&self], &mut ExecConfig::default())?
			.remove(self)
			.unwrap())
	}

	/// Call 'calc()' and use the result to update this node.
	pub fn calc_value(&self) -> Result<Self, ExecError> {
		self.set_value(self.calc()?);
		Ok(self.clone())
	}

	/// Sets the name of the node, replaces existing names.
	///
	/// Calling after node has been passed into an `Op` may result in derived names for `Op`s and `Node`s becoming out
	/// of sync.
	pub fn set_name<S: Into<String>>(&self, new_name: S) -> Self {
		let new_name = new_name.into();
		self.graph.with_root_inner_mut(|_, inner| {
			let mut data = self.data.lock();
			inner.associations.unassociate_node_name(self.id, &data.name);
			inner.associations.associate_node_name(self.id, &new_name);
			data.name = new_name;
		});

		self.clone()
	}

	/// Sets the name of the node, replaces existing name. The name root is extended with an integer, counting up from 0
	/// until a unique name is found.
	///
	/// Calling after node has been passed into an `Op` may result in derived names for `Op`s and `Node`s becoming out
	/// of sync.
	pub fn set_name_unique(&self, new_name_root: &str) -> Self {
		self.graph.with_root_inner_mut(|_, inner| {
			let new_name = inner.unique_node_name(new_name_root);
			let mut data = self.data.lock();
			inner.associations.unassociate_node_name(self.id, &data.name);
			inner.associations.associate_node_name(self.id, &new_name);
			data.name = new_name;
		});

		self.clone()
	}

	/// Associates a new tag with the node.
	pub fn add_tag<T: Into<NodeTag>>(&self, tag: T) -> Self {
		let tag = tag.into();
		self.graph.with_root_inner_mut(|_, inner| {
			let mut data = self.data.lock();
			inner.associations.associate_node_tag(self.id, &tag);
			data.tags.insert(tag);
		});

		self.clone()
	}

	/// Associates new tags with the node.
	pub fn add_tags<I: IntoIterator<Item = NodeTag>>(&self, tags: I) -> Self {
		self.graph.with_root_inner_mut(|_, inner| {
			let mut data = self.data.lock();
			for tag in tags {
				inner.associations.associate_node_tag(self.id, &tag);
				data.tags.insert(tag);
			}
		});

		self.clone()
	}

	/// Remove an assosiated tag with the node.
	///
	/// Can arbitrarily change the order of the tags.
	pub fn remove_tag<T: Into<NodeTag>>(&self, tag: T) -> Self {
		let tag = tag.into();
		self.graph.with_root_inner_mut(|_, inner| {
			let mut data = self.data.lock();
			inner.associations.unassociate_node_tag(self.id, &tag);
			data.tags.swap_remove(&tag);
		});

		self.clone()
	}

	/// Set a persistant value for the node.
	///
	/// This is used for calls to `calc()` and `eval()`, except when overridden with an input value.
	///
	/// ```
	/// # extern crate ndarray;
	/// # extern crate alumina;
	/// # use ndarray::{arr0, arr2};
	/// # use alumina::graph::Node;
	///
	/// let value_node1 = Node::new(&[0; 0]).set_value(arr0(5.0));
	/// let value_node2 = Node::new(&[2, 3]).set_value(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
	/// ```
	pub fn set_value<V: IntoNodeValue>(&self, new_value: V) -> Self {
		let mut data = self.data.lock();
		data.value = Some(new_value.into_value());

		self.clone()
	}

	/// Clear the persistant value for the node.
	pub fn clear_value(&self) -> Self {
		let mut data = self.data.lock();
		data.value = None;

		self.clone()
	}

	/// Sets the initialiser for the node.
	///
	/// # Panics
	///
	/// Panics if node shape is not fully determined.
	pub fn set_init<I: Into<Initialiser>>(&self, new_init: I) -> Self {
		let mut data = self.data.lock();
		data.init = Some(new_init.into());

		self.clone()
	}

	/// Calls initialiser if one has been set and assigns the output as the node value.
	pub fn init_value(&self) -> Self {
		self.init_array().map(|v| self.set_value(v));

		self.clone()
	}

	/// Returns the set of `Op`s that output to this `Node`.
	pub fn parent_ops(&self) -> IndexSet<Op> {
		self.graph.with_root_inner_mut(|graph, inner| {
			inner
				.node_parents(self.id)
				.iter()
				.map(|&op_inner| inner.op_from_inner(graph, op_inner)) // TODO check efficiency impact of repeat graph locking
				.collect()
		})
	}

	/// Returns the `Op` that output to this `Node`.
	///
	/// # Panics
	///
	/// Panics if number of input ops is not equal to 1.
	pub fn parent_op(&self) -> Op {
		self.graph.with_root_inner_mut(|graph, inner| {
			let parent = inner.node_parent(self.id);
			inner.op_from_inner(graph, parent)
		})
	}

	/// Returns the set of `Op`s that use this `Node` as an input.
	pub fn child_ops(&self) -> IndexSet<Op> {
		self.graph.with_root_inner_mut(|graph, inner| {
			inner
				.node_children(self.id)
				.iter()
				.map(move |&op_inner| inner.op_from_inner(graph, op_inner))// TODO check efficiency impact of repeat graph locking
				.collect()
		})
	}

	/// Returns the `Graph` of which this node is a member.
	pub fn graph(&self) -> &Graph {
		&self.graph
	}

	/// Returns a shared reference to the `NodeInner` which identifies this `Node`.
	pub fn id(&self) -> NodeID {
		self.id
	}

	/// Returns a copy of the current `Node` name.
	///
	/// Default: Unnamed_Node
	pub fn name(&self) -> String {
		let data = self.data.lock();
		data.name.clone()
	}

	/// Returns a copy of the current `Node` tags.
	pub fn tags(&self) -> IndexSet<NodeTag> {
		let data = self.data.lock();
		data.tags.clone()
	}

	/// Returns the `Initialiser` if one is set.
	pub fn init(&self) -> Option<Initialiser> {
		let data = self.data.lock();
		data.init.clone()
	}

	/// Returns the nodes shape
	pub fn shape(&self) -> NodeShape {
		let data = self.data.lock();
		data.shape.clone()
	}

	/// Returns copy-on-write value of the `Node`, if one has been set.
	pub fn value(&self) -> Option<ArcArray<f32, IxDyn>> {
		let data = self.data.lock();
		data.value.clone()
	}

	/// Takes and returns value of the `Node`, if one has been set, leaving None in its place.
	pub fn take_value(&self) -> Option<ArcArray<f32, IxDyn>> {
		let mut data = self.data.lock();
		data.value.take()
	}

	/// Returns the shape of the value if one is set
	pub fn value_shape(&self) -> Option<IxDyn> {
		let data = self.data.lock();
		data.value.as_ref().map(|v| IxDyn(v.shape()))
	}

	pub fn has_value(&self) -> bool {
		let data = self.data.lock();
		data.value.is_some()
	}

	/// Returns the result of calling the initialiser.
	///
	/// # Panics
	/// Panics if the node is not a fixed shape (all known axes).
	pub fn init_array(&self) -> Option<ArrayD<f32>> {
		let mut data = self.data.lock();
		let NodeInnerData {
			ref shape,
			ref mut init,
			..
		} = &mut *data;
		init.as_mut().map(|i| i.array(shape.to_data_shape().unwrap()))
	}
}

impl PartialEq for Node {
	fn eq(&self, other: &Node) -> bool {
		self.id == other.id
	}
}

impl Eq for Node {}

impl PartialOrd for Node {
	fn partial_cmp(&self, other: &Self) -> std::option::Option<std::cmp::Ordering> {
		self.id.partial_cmp(&other.id)
	}
}

impl Ord for Node {
	fn cmp(&self, other: &Self) -> std::cmp::Ordering {
		self.id.cmp(&other.id)
	}
}

impl Hash for Node {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.id.hash(state)
	}
}

impl Display for Node {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> ::std::fmt::Result {
		Display::fmt(&self.name(), fmt)
	}
}

impl Debug for Node {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> ::std::fmt::Result {
		fmt.debug_struct("Node")
			.field("name", &self.name())
			.field("shape", &format!("{}", self.shape()))
			.finish()
	}
}

impl Deref for Node {
	type Target = NodeID;

	fn deref(&self) -> &Self::Target {
		&self.id
	}
}

impl From<&Node> for Node {
	fn from(s: &Self) -> Self {
		s.clone()
	}
}

impl From<&&Node> for Node {
	fn from(s: &&Self) -> Self {
		(*s).clone()
	}
}

impl From<&&&Node> for Node {
	fn from(s: &&&Self) -> Self {
		(**s).clone()
	}
}

impl<V: IntoNodeValue> From<V> for Node {
	fn from(v: V) -> Node {
		let v = v.into_value();
		Node::new(v.shape()).set_value(v)
	}
}

impl<V: IntoNodeValue, S: Into<String>> From<(V, S)> for Node {
	fn from((v, s): (V, S)) -> Node {
		let v = v.into_value();
		Node::new(v.shape()).set_value(v).set_name(s)
	}
}

impl<T> From<HeavyNode<T>> for Node {
	fn from(hn: HeavyNode<T>) -> Node {
		hn.into_node()
	}
}

static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(0);

/// A reference to a node without reference to the `Graph`.
///
/// To avoid circular references this type is used inside `Graph` and `OpInstance`s, however most users should prefer
/// to use `Node` unless extending the library.
///
/// Implements methods which are possible without the graph.
#[derive(Copy, Debug)]
pub struct NodeID {
	id: u64,
}

impl NodeID {
	/// Returns the memory address of the node data for use as a unique identifer.
	///
	/// Hash and Eq are implemented based on this value.
	pub fn id(&self) -> u64 {
		self.id
	}

	/// Generate a new, unique NodeID  
	#[allow(clippy::new_without_default)] // Default is misleading because each new is unique
	pub fn new() -> Self {
		Self {
			id: NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed),
		}
	}
}

impl Clone for NodeID {
	fn clone(&self) -> Self {
		NodeID {
			//data: self.data.clone(),
			id: self.id,
		}
	}
}

impl PartialEq for NodeID {
	fn eq(&self, other: &NodeID) -> bool {
		self.id() == other.id()
	}
}

impl Eq for NodeID {}

impl PartialOrd for NodeID {
	fn partial_cmp(&self, other: &Self) -> std::option::Option<std::cmp::Ordering> {
		self.id.partial_cmp(&other.id)
	}
}

impl Ord for NodeID {
	fn cmp(&self, other: &Self) -> std::cmp::Ordering {
		self.id.cmp(&other.id)
	}
}

impl Hash for NodeID {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.id().hash(state)
	}
}

impl Equivalent<Node> for NodeID {
	fn equivalent(&self, key: &Node) -> bool {
		key.id().id == self.id
	}
}

impl Equivalent<NodeID> for Node {
	fn equivalent(&self, key: &NodeID) -> bool {
		key.id == self.id().id
	}
}

pub struct NodeInnerData {
	pub shape: NodeShape,
	pub name: String,
	pub tags: IndexSet<NodeTag>,
	pub value: Option<ArcArray<f32, IxDyn>>,
	pub init: Option<Initialiser>,
}

/// A Node, plus an additional data component
///
/// This is typically returned by builder functions, dereferences as a `Node` (usually the output), but also carries
/// meta-information specific to that function, e.g. what operations were created, what parameter nodes were created
/// etc.
///
/// This allows the HeavyNode to be fed directly into anything that accepts Into<Node>, such as further builder
/// functions.
pub struct HeavyNode<T> {
	node: Node,
	data: T,
}

impl<T> HeavyNode<T> {
	pub fn new(node: Node, data: T) -> Self {
		HeavyNode { node, data }
	}

	pub fn node(&self) -> &Node {
		&self.node
	}

	pub fn data(&self) -> &T {
		&self.data
	}

	/// Takes a closure and runs it providing mutable access to data component of the Heavy Node
	///
	/// Used for keeping changes inline, e.g. setting filter initialiser after calling `conv(..)`.
	pub fn with<F>(mut self, f: F) -> Self
	where
		F: FnOnce(&mut T),
	{
		f(&mut self.data);
		self
	}

	pub fn into_node(self) -> Node {
		self.node
	}

	pub fn into_data(self) -> T {
		self.data
	}
}

impl<T> Deref for HeavyNode<T> {
	type Target = Node;

	fn deref(&self) -> &Self::Target {
		&self.node
	}
}

impl<T> From<(Node, T)> for HeavyNode<T> {
	fn from((n, t): (Node, T)) -> Self {
		HeavyNode { node: n, data: t }
	}
}

/// A handle to the operation, cloning this does not duplicate the operation itself.
#[derive(Clone)]
pub struct Op {
	graph: Graph,
	id: OpID,
	data: Arc<Mutex<OpInnerData>>,
	instance: Arc<dyn OpInstance>,
}

impl Op {
	/// Remove this `Op` from its `Graph`
	///
	/// # Panics
	/// Panics if any other `Op`s still exists for this.
	pub fn remove(self) -> OpInnerData {
		let Op {
			graph,
			id,
			data,
			instance,
		} = self;
		drop(data);
		drop(instance);
		graph.remove_op(id)
	}

	/// Set the name of the Op, replaces existing name.
	pub fn set_name<S: Into<String>>(&self, new_name: S) -> Self {
		let new_name: String = new_name.into();
		self.graph.with_root_inner_mut(|_, inner| {
			let mut data = self.data.lock();
			inner.associations.unassociate_op_name(self.id, &data.name);
			inner.associations.associate_op_name(self.id, &new_name);
			data.name = new_name;
		});

		self.clone()
	}

	/// Sets the name of the Op, replaces existing name. The name root is extended with an integer, counting up from 0
	/// until a unique name is found.
	pub fn set_name_unique(&self, new_name_root: &str) -> Self {
		self.graph.with_root_inner_mut(|_, inner| {
			let new_name = inner.unique_op_name(new_name_root);
			let mut data = self.data.lock();
			inner.associations.unassociate_op_name(self.id, &data.name);
			inner.associations.associate_op_name(self.id, &new_name);
			data.name = new_name;
		});

		self.clone()
	}

	/// Associates a new tag with the op.
	pub fn add_tag<T: Into<OpTag>>(&self, tag: T) -> Self {
		let tag = tag.into();
		self.graph.with_root_inner_mut(|_, inner| {
			let mut data = self.data.lock();
			inner.associations.associate_op_tag(self.id, &tag);
			data.tags.insert(tag);
		});

		self.clone()
	}

	/// Associates a new tags with the op.
	pub fn add_tags<I: IntoIterator<Item = OpTag>>(&self, tags: I) -> Self {
		self.graph.with_root_inner_mut(|_, inner| {
			let mut data = self.data.lock();

			for tag in tags {
				inner.associations.associate_op_tag(self.id, &tag);
				data.tags.insert(tag);
			}
		});

		self.clone()
	}

	/// Remove an assosiated tag with the node.
	///
	/// Can arbitrarily change the order of the tags.
	pub fn remove_tag<T: Into<OpTag>>(&self, tag: T) -> Self {
		let tag = tag.into();
		self.graph.with_root_inner_mut(|_, inner| {
			let mut data = self.data.lock();
			inner.associations.unassociate_op_tag(self.id, &tag);
			data.tags.swap_remove(&tag);
		});

		self.clone()
	}

	/// Returns the set of `Node`s this `Op` uses as inputs in the `Graph`.
	pub fn parent_nodes(&self) -> IndexSet<Node> {
		self.graph.with_root_inner_mut(|graph, inner| {
			inner
				.op_parents(&self.id)
				.iter()
				.map(|&node_inner| inner.node_from_inner(graph, node_inner))// TODO check efficiency impact of repeat graph locking
				.collect()
		})
	}

	/// Returns the set of `Node`s this `Op` outputs to in the `Graph`.
	pub fn child_nodes(&self) -> IndexSet<Node> {
		self.graph.with_root_inner_mut(|graph, inner| {
			inner
				.op_children(&self.id)
				.iter()
				.map(|&node_inner| inner.node_from_inner(graph, node_inner))// TODO check efficiency impact of repeat graph locking
				.collect()
		})
	}

	/// Returns the `Graph` of which this `Op` is a member.
	pub fn graph(&self) -> &Graph {
		&self.graph
	}

	/// Returns a shared reference to the `OpInner` which identifies this `Op`.
	pub fn id(&self) -> OpID {
		self.id
	}

	/// Returns a copy of the current `Op` name.
	///
	/// Default: Unnamed_Op
	///
	/// Usually this default is overwritten by a name derived from parent and child `Node`s.
	pub fn name(&self) -> String {
		let data = self.data.lock();
		data.name.clone()
	}

	pub fn type_name(&self) -> &'static str {
		let data = self.data.lock();
		data.instance.type_name()
	}

	/// Returns a copy of the current `Node` tags.
	pub fn tags(&self) -> IndexSet<OpTag> {
		let data = self.data.lock();
		data.tags.clone()
	}

	pub fn instance(&self) -> Arc<dyn OpInstance> {
		self.instance.clone()
	}
}

impl PartialEq for Op {
	fn eq(&self, other: &Op) -> bool {
		self.id == other.id
	}
}

impl Eq for Op {}

impl PartialOrd for Op {
	fn partial_cmp(&self, other: &Self) -> std::option::Option<std::cmp::Ordering> {
		self.id.partial_cmp(&other.id)
	}
}

impl Ord for Op {
	fn cmp(&self, other: &Self) -> std::cmp::Ordering {
		self.id.cmp(&other.id)
	}
}

impl Hash for Op {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.id.hash(state)
	}
}

impl Display for Op {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> ::std::fmt::Result {
		Display::fmt(&self.name(), fmt)
	}
}

impl Debug for Op {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> ::std::fmt::Result {
		fmt.debug_struct("Op")
			.field("name", &self.name())
			.field("type", &self.type_name())
			.finish()
	}
}

impl Deref for Op {
	type Target = OpID;

	fn deref(&self) -> &Self::Target {
		&self.id
	}
}

static NEXT_OP_ID: AtomicU64 = AtomicU64::new(0);

/// A reference to an op without reference to the `Graph`.
///
/// To avoid circular references this type is used inside `Graph` and `OpInstance`s, however most users should prefer
/// to use `Op` unless extending the library.
///
/// Implements methods which are possible without the graph.
#[derive(Copy, Debug)]
pub struct OpID {
	id: u64,
}

impl OpID {
	/// Returns the memory address of the op data for use as a unique identifer.
	///
	/// Hash and Eq are implemented based on this value.
	pub fn id(&self) -> u64 {
		self.id
	}

	/// Generate a new, unique OpID  
	#[allow(clippy::new_without_default)] // Default is misleading because each new is unique
	pub fn new() -> Self {
		Self {
			id: NEXT_OP_ID.fetch_add(1, Ordering::Relaxed),
		}
	}
}

impl Clone for OpID {
	fn clone(&self) -> Self {
		OpID { id: self.id }
	}
}

impl PartialEq for OpID {
	fn eq(&self, other: &OpID) -> bool {
		self.id() == other.id()
	}
}

impl Eq for OpID {}

impl Hash for OpID {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.id().hash(state)
	}
}

impl PartialOrd for OpID {
	fn partial_cmp(&self, other: &Self) -> std::option::Option<std::cmp::Ordering> {
		self.id.partial_cmp(&other.id)
	}
}

impl Ord for OpID {
	fn cmp(&self, other: &Self) -> std::cmp::Ordering {
		self.id.cmp(&other.id)
	}
}

impl Equivalent<Op> for OpID {
	fn equivalent(&self, key: &Op) -> bool {
		key.id().id == self.id
	}
}

pub struct OpInnerData {
	pub name: String,
	pub tags: IndexSet<OpTag>,
	pub instance: Arc<dyn OpInstance>,
}

impl Default for Graph {
	fn default() -> Self {
		Self::new()
	}
}

/// An Arc-like handle to the computational graph.
pub struct Graph {
	link: Arc<Mutex<GraphLink>>,
}

/// Implementation detail: When cloning a best effort attempt is made to create a `Graph` with
/// a direct link to the `GraphInner`, along with updating this `Graph` to be at most one jump away.
impl Clone for Graph {
	fn clone(&self) -> Self {
		// If chasing the root would block, then just return at the current position in the tree.
		let mut self_link = match self.link.try_lock() {
			Some(guard) => guard,
			None => {
				return Graph {
					link: self.link.clone(),
				};
			}
		};

		// get root by either being root, or cloning
		let root = match &*self_link {
			GraphLink::Root(_) => {
				return Graph {
					link: self.link.clone(),
				};
			}
			GraphLink::MergedInto(ref next_graph) => next_graph.clone(),
		};

		// If we aren't the root, then update so we are at most one jump away from the root
		*self_link = GraphLink::MergedInto(Graph {
			link: root.link.clone(),
		});

		root
	}
}

impl Hash for Graph {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.with_root_link_mut(|graph, _link| graph.link_addr().hash(state))
	}
}

impl Display for Graph {
	fn fmt(&self, f: &mut fmt::Formatter) -> ::std::fmt::Result {
		self.with_root_inner_mut(|_graph, inner| {
			write!(f, "Graph {{ nodes: [")?;
			let mut iter = inner.nodes.iter();
			if let Some((_node, node_data)) = iter.next() {
				write!(f, "{}", node_data.lock().name)?;
				for (_node, node_data) in iter {
					write!(f, ", {}", node_data.lock().name)?;
				}
			}
			write!(f, "], ops: [")?;
			let mut iter = inner.ops.iter();
			if let Some((_op, op_data)) = iter.next() {
				write!(f, "{}", op_data.lock().name)?;
				for (_op, op_data) in iter {
					write!(f, ", {}", op_data.lock().name)?;
				}
			}
			write!(f, "] }}")?;
			Ok(())
		})
	}
}

impl Debug for Graph {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> ::std::fmt::Result {
		self.with_root_inner_mut(|_graph, inner| {
			fmt.debug_struct("Graph")
				.field(
					"nodes",
					&IterDebug {
						inner: inner
							.nodes
							.iter()
							.map(|(_, data)| data.lock().name.clone())
							.collect::<Vec<_>>(),
					},
				)
				.field(
					"ops",
					&IterDebug {
						inner: inner
							.ops
							.iter()
							.map(|(_, data)| data.lock().name.clone())
							.collect::<Vec<_>>(),
					},
				)
				.finish()
		})
		// self.with_nodes_ops(|nodes, ops| {
		// 	fmt.debug_struct("Graph")
		// 		.field("nodes", &IterDebug { inner: nodes.iter().map(|(_, data)| data.name.clone()).collect::<Vec<_>>() })
		// 		.field("ops", &IterDebug { inner: ops.iter().map(|(_, data)| data.name.clone()).collect::<Vec<_>>() })
		// 		.finish()
		// })
	}
}

impl PartialEq for Graph {
	fn eq(&self, other: &Self) -> bool {
		// If they have the same root then they are equal
		self.with_locked_roots(other, |_, _, _, _| false, |_, _| true)
	}
}

impl Eq for Graph {}

impl Graph {
	/// Create an empty `Graph`
	pub fn new() -> Self {
		Graph {
			link: Arc::new(Mutex::new(GraphLink::Root(GraphInner::default()))),
		}
	}

	/// Create a new node within the Graph
	pub fn new_node(&self, shape: NodeShape) -> Node {
		let node_inner = NodeID::new();
		let node_data = NodeInnerData {
			shape,
			name: "Unnamed_Node".to_string(),
			tags: IndexSet::new(),
			value: None,
			init: None,
		};

		self.with_root_inner_mut(|graph, inner| {
			inner.add_node(node_inner, Arc::new(Mutex::new(node_data)));
			inner.node_from_inner(graph, node_inner)
		})
	}

	/// Create a new `Op` within the Graph
	pub fn new_op(&self, instance: Arc<dyn OpInstance>) -> Op {
		debug_assert!(
			self.with_root_inner_mut(|_, inner|{
				instance.inputs().iter().chain(&instance.outputs()).all(|node| inner.nodes.contains_key(node))
			}),
			"New OpInstance lists input or output nodes that are not part of this Graph. Prior to constructing an OpInstance, Graphs must be merged."
		);

		let op_inner = OpID::new();
		let op_data = OpInnerData {
			name: "Unnamed_Op".to_string(),
			tags: IndexSet::new(),
			instance,
		};

		self.with_root_inner_mut(|graph, inner| {
			inner.add_op(op_inner, Arc::new(Mutex::new(op_data)));
			inner.op_from_inner(graph, op_inner)
		})
	}

	/// Remove from the graph the selected `NodeID`
	///
	/// # Panics
	/// Panics if the `NodeID` is not a member of this graph.
	/// Panics if any `Node`s still exists for this `NodeID`.
	/// If the Node is still referenced by Ops in the graph then this will panic in debug mode, and may panic in Release.
	pub fn remove_node(&self, node_id: NodeID) -> NodeInnerData {
		self.with_root_inner_mut(|_graph, graph_inner| graph_inner.remove_node(node_id))
	}

	/// Remove from the graph the selected `OpID`
	///
	/// # Panics
	/// Panics if the `OpID` is not a member of this graph.
	/// Panics if any `Op`s still exists for this `OpID`.
	pub fn remove_op(&self, op_id: OpID) -> OpInnerData {
		self.with_root_inner_mut(|_graph, graph_inner| graph_inner.remove_op(op_id))
	}

	/// Create a `Node` from a `NodeInner`
	///
	/// # Locking
	/// This locks the underlying graph. If the graph is already locked then GraphInner should be used to avoid deadlocks.
	/// # Panics
	/// Panics if the NodeInner is not a member of this graph.
	pub fn node_from_id(&self, inner: NodeID) -> Node {
		self.with_root_inner_mut(|graph, graph_inner| graph_inner.node_from_inner(graph, inner))
	}

	/// Create an `Op` from an `OpInner`
	///
	/// # Locking
	/// This locks the underlying graph. If the graph is already locked then GraphInner should be used to avoid deadlocks.
	/// # Panics
	/// Panics if the OpInner is not a member of this graph.
	pub fn op_from_id(&self, inner: OpID) -> Op {
		self.with_root_inner_mut(|graph, graph_inner| graph_inner.op_from_inner(graph, inner))
	}

	// /// Create a `Node` from a `NodeInner`
	// fn node_from_inner_unchecked(&self, inner: NodeInner) -> Node {
	// 	Node {
	// 		graph: self.clone(),
	// 		inner,
	// 	}
	// }

	// /// Create an `Op` from an `OpInner`
	// fn op_from_inner_unchecked(&self, inner: OpInner) -> Op {
	// 	let instance = 		self.with_root_inner_mut(|_graph, graph_inner| {
	// 		graph_inner.ops.get(&inner).unwrap().instance.clone()
	// 	});

	// 	Op {
	// 		graph: self.clone(),
	// 		inner,
	// 		instance
	// 	}
	// }

	/// Return the set of `Node`s associated with the given tag.
	pub fn nodes_tagged<T: Into<NodeTag>>(&self, tag: T) -> IndexSet<Node> {
		self.with_root_inner_mut(|graph, inner| {
			inner
				.nodes_tagged(&tag.into())
				.iter()
				.map(|node_inner| inner.node_from_inner(graph, *node_inner))
				.collect()
		})
	}

	/// Return the set of `Node`s associated with the given name.
	pub fn nodes_named(&self, name: &str) -> IndexSet<Node> {
		self.with_root_inner_mut(|graph, inner| {
			inner
				.nodes_named(name)
				.iter()
				.map(|node_inner| inner.node_from_inner(graph, *node_inner))
				.collect()
		})
	}

	/// Return the `Node` associated with the given name.
	///
	/// # Panics
	///
	/// Panics if number of `Node`s found is not equal to 1.
	pub fn node_named(&self, name: &str) -> Node {
		self.with_root_inner_mut(|graph, inner| {
			let node = inner.node_named(name);
			inner.node_from_inner(graph, node)
		})
	}

	/// Return the set of `Op`s associated with the given tag.
	pub fn ops_tagged<T: Into<OpTag>>(&self, tag: T) -> IndexSet<Op> {
		self.with_root_inner_mut(|graph, inner| {
			inner
				.ops_tagged(&tag.into())
				.iter()
				.map(|op_inner| inner.op_from_inner(graph, *op_inner))
				.collect()
		})
	}

	/// Return the set of `Op`s associated with the given name.
	pub fn ops_named(&self, name: &str) -> IndexSet<Op> {
		self.with_root_inner_mut(|graph, inner| {
			inner
				.ops_named(name)
				.iter()
				.map(|op_inner| inner.op_from_inner(graph, *op_inner))
				.collect()
		})
	}

	/// Return the `Op` associated with the given name.
	///
	/// # Panics
	///
	/// Panics if number of `Op`s found is not equal to 1.
	pub fn op_named(&self, name: &str) -> Op {
		self.with_root_inner_mut(|graph, inner| {
			let op = inner.op_named(name);
			inner.op_from_inner(graph, op)
		})
	}

	/// Returns a clone of the current set of `Node`s in the graph.
	///
	/// Does not update with modifications to the graph.
	pub fn nodes(&self) -> IndexSet<Node> {
		self.with_root_inner_mut(|graph, inner| {
			inner
				.nodes
				.keys()
				.map(|node_inner| inner.node_from_inner(graph, *node_inner))
				.collect()
		})
	}

	/// Returns a clone of the current set of `Op`s in the graph
	///
	/// Does not update with modifications to the graph
	pub fn ops(&self) -> IndexSet<Op> {
		self.with_root_inner_mut(|graph, inner| {
			inner
				.ops
				.keys()
				.map(|op_inner| inner.op_from_inner(graph, *op_inner))
				.collect()
		})
	}

	pub fn node_count(&self) -> usize {
		self.with_root_inner_mut(|_graph, inner| inner.nodes.len())
	}

	pub fn op_count(&self) -> usize {
		self.with_root_inner_mut(|_graph, inner| inner.ops.len())
	}

	// pub fn node_index(&self, node: NodeID) -> usize {
	// 	self.with_root_inner_mut(|_graph, inner| {
	// 		inner.nodes.get_full(&node).unwrap().0
	// 	})
	// }

	// /// Returns the index of the op
	// ///
	// /// Ops are indicies are in order of addition to the `Graph`
	// /// Graph merges produce
	// /// # Panics
	// /// Panics if the OpInner is not a member of the Graph
	// pub fn op_index(&self, node: OpID) -> usize {
	// 	self.with_root_inner_mut(|_graph, inner| {
	// 		inner.ops.get_full(&node).unwrap().0
	// 	})
	// }

	/// Merges `g2` into self.
	pub fn merge(&self, g2: &Graph) {
		if self.link_addr() != g2.link_addr() {
			self.with_locked_roots(
				g2,
				|g1: &Graph, gl1: &mut GraphLink, _g2: &Graph, gl2: &mut GraphLink| {
					let g2_inner = gl2.take_inner(g1).unwrap();

					let g1_inner: &mut GraphInner = gl1.inner_mut().unwrap();

					for (node, node_data) in g2_inner.nodes.into_iter() {
						g1_inner.add_node(node, node_data)
					}

					for (op, op_data) in g2_inner.ops.into_iter() {
						g1_inner.add_op(op, op_data)
					}
				},
				|_, _| {},
			);
		}
	}

	// fn lock_graph_link(&self) -> MutexGuard<GraphLink> {
	// 	self.link
	// 		.lock()
	// 		.expect("Another thread has poisoned (panicked while holding) the GraphLink Mutex")
	// }

	/// Finds the graph root, locks it, then runs the provided closure
	fn with_root_link_mut<T, F: FnOnce(&Graph, &mut GraphLink) -> T>(&self, f: F) -> T {
		let mut graph = self.clone();
		loop {
			{
				let mut graph_link = graph.link.lock();
				if let x @ &mut GraphLink::Root(_) = &mut *graph_link {
					return f(&graph, x);
				}
			}
			// Cloning acquires a root reference.
			// If it doesnt, then we are already going to deadlock when we lock, or it will get closer in future
			// iterations
			graph = graph.clone();
		}
	}

	/// Finds the graph root, locks it, then runs the provided closure
	fn with_root_inner_mut<T, F: FnOnce(&Graph, &mut GraphInner) -> T>(&self, f: F) -> T {
		self.with_root_link_mut(|graph, link| {
			f(graph, link.inner_mut().unwrap()) // this is ok as we are guaranteed to be at the root node
		})
	}

	// /// Run a closure taking the sets of `Node`s and `Op`s in the graph.
	// ///
	// /// # Locking
	// /// Useful to avoid cloning the sets of ops and nodes in the graph by running the closure while the graph internals
	// /// are locked. However, caution is advised as nesting multiple calls, or triggering any other lock of the graph
	// /// will deadlock.
	// pub fn with_nodes_ops<T, F: FnOnce(&IndexMap<NodeInner, NodeInnerData>, &IndexMap<OpInner, OpInnerData>) -> T>(&self, f: F) -> T {
	// 	self.with_root_inner_mut(|_, inner| f(&inner.nodes, &inner.ops))
	// }

	/// Find and lock the roo(s). Run either of two functions depending on whether the two graphs share a root.
	fn with_locked_roots<T, F, G>(&self, g2: &Graph, different_root: F, same_root: G) -> T
	where
		F: FnOnce(&Graph, &mut GraphLink, &Graph, &mut GraphLink) -> T,
		G: FnOnce(&Graph, &mut GraphLink) -> T,
	{
		// Clone to get the current graph roots. Because we cant assume other threads arent merging graphs and these dont lock,
		// they might not still be the roots.
		let mut g1 = self.clone();
		let mut g2 = g2.clone();

		loop {
			// Lock graph with lower memory address first to avoid dining philosophers problem.
			match g1.link_addr().cmp(&g2.link_addr()) {
				std::cmp::Ordering::Less => {
					let mut g1_link = g1.link.lock();
					if let gl1 @ &mut GraphLink::Root(_) = &mut *g1_link {
						let mut g2_link = g2.link.lock();
						if let gl2 @ &mut GraphLink::Root(_) = &mut *g2_link {
							return different_root(&g1, gl1, &g2, gl2);
						}
					}
				}
				std::cmp::Ordering::Equal => {
					let mut g1_link = g1.link.lock();
					match &mut *g1_link {
						x @ &mut GraphLink::Root(_) => return same_root(&g1, x),
						&mut GraphLink::MergedInto(_) => continue,
					}
				}
				std::cmp::Ordering::Greater => {
					let mut g2_link = g2.link.lock();
					if let gl2 @ &mut GraphLink::Root(_) = &mut *g2_link {
						let mut g1_link = g1.link.lock();
						if let gl1 @ &mut GraphLink::Root(_) = &mut *g1_link {
							return different_root(&g1, gl1, &g2, gl2);
						}
					}
				}
			}

			// If locking g1 and g2 suceeded and they were both roots, then we already have returned.
			// Cloning acquires a root reference, so we clone to reattempt locking the root.
			// Under concurrent modification these may not longer point to root once locking is attemped so iteration is required.
			g1 = g1.clone();
			g2 = g2.clone();
		}
	}

	fn link_addr(&self) -> usize {
		&*self.link as *const Mutex<GraphLink> as usize
	}
}

/// Most ops should use this when merging the graphs of nodes.
///
/// Merges into the graph with the most nodes, calling merge_graphs() under the hood.
///
/// # Panics
/// Panics if slice is empty
pub fn merge_node_graphs<N: Borrow<Node>, I: IntoIterator<Item = N>>(node_iter: I) -> Graph {
	let nodes: SmallVec<[N; 8]> = node_iter.into_iter().collect();
	let graphs: SmallVec<[&Graph; 8]> = nodes.iter().map(|n| n.borrow().graph()).collect();
	merge_graphs(&graphs)
}

/// Merges into the graph with the most nodes.
///
/// # Panics
/// Panics if slice is empty
pub fn merge_graphs<G: Borrow<Graph>>(graphs: &[G]) -> Graph {
	assert!(!graphs.is_empty(), "Merge of graphs failed, slice was empty.");

	if graphs.len() == 1 {
		return graphs[0].borrow().clone();
	}

	let (max_i, _) = graphs.iter().enumerate().fold((0, 0), |(max_i, max_count), (i, g)| {
		let graph_i = g.borrow();
		if graph_i.node_count() > max_count {
			(i, graph_i.node_count())
		} else {
			(max_i, max_count)
		}
	});

	let max_graph = graphs[max_i].borrow().clone();
	for graph_i in graphs {
		max_graph.merge(graph_i.borrow());
	}

	max_graph
}

/// `GraphLinks` form a tree, lengthened when graph merges occur.
/// There is only one Root in each tree, which contains the only GraphInner
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum GraphLink {
	MergedInto(Graph),
	Root(GraphInner), // TODO consider a variant optimised for graphs with a single op
}

impl GraphLink {
	/// Returns a `&mut GraphInner` if this is a `Root` variant, otherwise `None`
	fn inner_mut(&mut self) -> Option<&mut GraphInner> {
		match self {
			GraphLink::Root(inner) => Some(inner),
			GraphLink::MergedInto(_) => None,
		}
	}

	fn take_inner(&mut self, merged_into: &Graph) -> Option<GraphInner> {
		if let GraphLink::MergedInto(_) = self {
			return None;
		}

		let mut x = GraphLink::MergedInto(merged_into.clone());
		::std::mem::swap(&mut x, self);

		match x {
			GraphLink::Root(inner) => Some(inner),
			GraphLink::MergedInto(_) => unreachable!(),
		}
	}
}

/// Guts of the graph, non public type
///
/// store lists of nodes and ops
/// optionally caches computational information such as shapes based on inputs and necessary subgraphs for output
/// computations
#[derive(Default)]
struct GraphInner {
	nodes: BTreeMap<NodeID, Arc<Mutex<NodeInnerData>>>,
	ops: BTreeMap<OpID, Arc<Mutex<OpInnerData>>>,

	// De-normalised data only
	// NodeInner and OpInner contain the primary record
	associations: Associations,
	relations: Relations,
}

impl GraphInner {
	/// Create a `Node` from a `NodeInner`
	///
	/// # Panics
	/// Panics if the NodeInner is not a member of this graph.
	pub fn node_from_inner(&self, graph: &Graph, inner: NodeID) -> Node {
		let data = self.nodes.get(&inner);

		Node {
			graph: graph.clone(),
			data: match data {
				Some(x) => x.clone(),
				None => panic!("NodeInner (id:{}) is not a part of this Graph.", inner.id()),
			},
			id: inner,
		}
	}

	/// Create an `Op` from an `OpInner`
	///
	/// # Panics
	/// Panics if the OpInner is not a member of this graph.
	pub fn op_from_inner(&self, graph: &Graph, inner: OpID) -> Op {
		let data = self.ops.get(&inner);
		match data {
			Some(x) => Op {
				graph: graph.clone(),
				id: inner,
				data: x.clone(),
				instance: x.lock().instance.clone(),
			},
			None => panic!("OpInner (id:{}) is not a part of this Graph.", inner.id()),
		}
	}

	fn node_children(&mut self, node: NodeID) -> IndexSet<OpID> {
		self.relations
			.get_or_instantiate_relations(&self.nodes, &self.ops)
			.node_children(node)
	}

	fn node_parents(&mut self, node: NodeID) -> IndexSet<OpID> {
		self.relations
			.get_or_instantiate_relations(&self.nodes, &self.ops)
			.node_parents(node)
	}

	fn node_parent(&mut self, node: NodeID) -> OpID {
		let set = self.node_parents(node);
		let mut iter = set.iter();

		let op = iter
			.next()
			.unwrap_or_else(|| panic!("No parent Op for Node: {}", node.id()));
		assert!(
			iter.next().is_none(),
			"More than one parent Op for Node : {}",
			self.nodes.get(&node).unwrap().lock().name
		);
		*op
	}

	fn nodes_tagged(&mut self, tag: &NodeTag) -> IndexSet<NodeID> {
		self.associations
			.get_or_instantiate_association(&self.nodes, &self.ops)
			.nodes_tagged(tag)
	}

	fn nodes_named(&mut self, name: &str) -> IndexSet<NodeID> {
		self.associations
			.get_or_instantiate_association(&self.nodes, &self.ops)
			.nodes_named(name)
	}

	fn node_named(&mut self, name: &str) -> NodeID {
		let set = self.nodes_named(name);
		let mut iter = set.iter();

		let node = iter
			.next()
			.unwrap_or_else(|| panic!("No Node associated with name: {}", name));
		assert!(iter.next().is_none(), "More than one Node named: {}", name);
		*node
	}

	fn op_children(&mut self, op: &OpID) -> IndexSet<NodeID> {
		self.relations
			.get_or_instantiate_relations(&self.nodes, &self.ops)
			.op_children(op)
	}

	fn op_parents(&mut self, op: &OpID) -> IndexSet<NodeID> {
		self.relations
			.get_or_instantiate_relations(&self.nodes, &self.ops)
			.op_parents(op)
	}

	fn ops_tagged(&mut self, tag: &OpTag) -> IndexSet<OpID> {
		self.associations
			.get_or_instantiate_association(&self.nodes, &self.ops)
			.ops_tagged(tag)
	}

	fn ops_named(&mut self, name: &str) -> IndexSet<OpID> {
		self.associations
			.get_or_instantiate_association(&self.nodes, &self.ops)
			.ops_named(name)
	}

	fn op_named(&mut self, name: &str) -> OpID {
		let set = self.ops_named(name);
		let mut iter = set.iter();

		let op = iter
			.next()
			.unwrap_or_else(|| panic!("No Node associated with name: {}", name));
		assert!(iter.next().is_none(), "More than one Op named: {}", name);
		*op
	}

	/// Add an existing node to the graph.
	/// Updates membership in the graph and denormalised data for names and tags.
	/// Does not add information for relationships to ops.
	fn add_node(&mut self, node_id: NodeID, node_data: Arc<Mutex<NodeInnerData>>) {
		assert!(
			!self.nodes.contains_key(&node_id),
			"Cannot add node that is already in graph: {}",
			node_data.lock().name
		);
		let node_data = self.nodes.entry(node_id).or_insert(node_data);

		self.relations.add_node(node_id);

		self.associations.associate_node_name(node_id, &node_data.lock().name);

		for tag in node_data.lock().tags.iter() {
			self.associations.associate_node_tag(node_id, tag);
		}
	}

	/// Remove from the graph the selected `NodeID`
	///
	/// # Panics
	/// Panics if the `NodeID` is not a member of this graph.
	/// Panics if any `Node`s still exists for this `NodeID`.
	/// If the Node is still referenced by Ops in the graph then this will panic in debug mode, and may panic in Release.
	fn remove_node(&mut self, node_id: NodeID) -> NodeInnerData {
		let node_data = match self.nodes.remove(&node_id) {
			Some(x) => x,
			None => panic!("NodeID (id:{}) is not a part of this Graph.", node_id.id()),
		};

		let node_data = match Arc::try_unwrap(node_data) {
			Ok(x) => x.into_inner(),
			Err(_) => panic!(
				"NodeID (id:{}) could not be removed from the graph as Node handles still exist",
				node_id.id()
			),
		};
		if cfg!(debug_assertions) {
			// if debug, instantiate so that remove node can panic if parents or children still exist
			let _ = self.relations.get_or_instantiate_relations(&self.nodes, &self.ops);
		}
		self.relations.remove_node(node_id);

		self.associations.unassociate_node_name(node_id, &node_data.name);
		for tag in node_data.tags.iter() {
			self.associations.unassociate_node_tag(node_id, tag);
		}

		node_data
	}

	/// Add an existing `Op` to the graph
	///
	/// # Panics
	/// May panic if input and output nodes haven't already been added.
	fn add_op(&mut self, op_id: OpID, op_data: Arc<Mutex<OpInnerData>>) {
		assert!(
			!self.ops.contains_key(&op_id),
			"Cannot add node that is already in graph: {}",
			op_data.lock().name
		);
		let op_data = self.ops.entry(op_id).or_insert(op_data);

		self.relations.add_op(op_id, op_data);

		self.associations.associate_op_name(op_id, &op_data.lock().name);

		for tag in op_data.lock().tags.iter() {
			self.associations.associate_op_tag(op_id, tag);
		}
	}

	/// Remove from the graph the selected `OpID`
	///
	/// # Panics
	/// Panics if the `OpID` is not a member of this graph.
	/// Panics if any `Op`s still exists for this `OpID`.
	fn remove_op(&mut self, op_id: OpID) -> OpInnerData {
		let op_data = match self.ops.remove(&op_id) {
			Some(x) => x,
			None => panic!("OpID (id:{}) is not a part of this Graph.", op_id.id()),
		};

		let op_data = match Arc::try_unwrap(op_data) {
			Ok(x) => x.into_inner(),
			Err(_) => panic!(
				"OpID (id:{}) could not be removed from the graph as Op handles still exist",
				op_id.id()
			),
		};

		self.relations
			.remove_op(op_id, op_data.instance.inputs(), op_data.instance.outputs());

		self.associations.unassociate_op_name(op_id, &op_data.name);
		for tag in op_data.tags.iter() {
			self.associations.unassociate_op_tag(op_id, tag);
		}

		op_data
	}

	fn unique_node_name(&mut self, new_name_root: &str) -> String {
		if self.nodes_named(&new_name_root).is_empty() {
			return new_name_root.to_string();
		}

		for i in 1.. {
			let next_node_name = format!("{}{}", new_name_root, i);
			if self.nodes_named(&next_node_name).is_empty() {
				return next_node_name;
			}
		}
		new_name_root.to_string()
	}

	fn unique_op_name(&mut self, new_name_root: &str) -> String {
		if self.ops_named(&new_name_root).is_empty() {
			return new_name_root.to_string();
		}

		for i in 1.. {
			let next_op_name = format!("{}{}", new_name_root, i);
			if self.ops_named(&next_op_name).is_empty() {
				return next_op_name;
			}
		}
		new_name_root.to_string()
	}
}

impl Display for GraphInner {
	fn fmt(&self, f: &mut fmt::Formatter) -> ::std::fmt::Result {
		write!(f, "GraphInner {{ nodes: [")?;
		let mut iter = self.nodes.values();
		if let Some(node) = iter.next() {
			write!(f, "{}", node.lock().name)?;
			for node in iter {
				write!(f, ", {}", node.lock().name)?;
			}
		}
		write!(f, "], ops: [")?;
		let mut iter = self.ops.values();
		if let Some(op) = iter.next() {
			write!(f, "{}", op.lock().name)?;
			for op in iter {
				write!(f, ", {}", op.lock().name)?;
			}
		}
		write!(f, "] }}")?;
		Ok(())
	}
}

impl Debug for GraphInner {
	fn fmt(&self, f: &mut fmt::Formatter) -> ::std::fmt::Result {
		Display::fmt(self, f)
	}
}

// impl Drop for GraphInner {
// 	fn drop(&mut self) {
// 		trace!(
// 			"Dropping graph with {} nodes and {} ops",
// 			self.nodes.len(),
// 			self.ops.len()
// 		);
// 	}
// }

#[derive(Default)]
struct Relations {
	inner: Option<RelationsInner>,
}

impl Relations {
	fn get_or_instantiate_relations(
		&mut self,
		nodes: &BTreeMap<NodeID, Arc<Mutex<NodeInnerData>>>,
		ops: &BTreeMap<OpID, Arc<Mutex<OpInnerData>>>,
	) -> &mut RelationsInner {
		if self.inner.is_none() {
			self.inner = Some(RelationsInner::from_graph(nodes, ops))
		}
		self.inner.as_mut().unwrap()
	}

	fn add_node(&mut self, node_id: NodeID) {
		if let Some(ref mut relations) = self.inner {
			if relations.node_parents.insert(node_id, IndexSet::new()).is_some() {
				panic!(
					"Parent relations already existed Node (id:{}) was already a member of the graph",
					node_id.id()
				)
			}

			if relations.node_children.insert(node_id, IndexSet::new()).is_some() {
				panic!(
					"Child relations already existed Node (id:{}) was already a member of the graph",
					node_id.id()
				)
			};
		}
	}

	fn remove_node(&mut self, node_id: NodeID) {
		if let Some(ref mut relations) = self.inner {
			if let Some(v) = relations.node_parents.swap_remove(&node_id) {
				assert!(
					v.is_empty(),
					"Node (id:{}) was removed while it still had remaining parents",
					node_id.id()
				)
			};
			if let Some(v) = relations.node_children.swap_remove(&node_id) {
				assert!(
					v.is_empty(),
					"Node (id:{}) was removed while it still had remaining children",
					node_id.id()
				)
			};
		}
	}

	fn add_op(&mut self, op_id: OpID, op_data: &Mutex<OpInnerData>) {
		if let Some(ref mut relations) = self.inner {
			let instance = &op_data.lock().instance;
			let parents = instance.inputs();
			for node in &parents {
				relations
					.node_children
					.get_mut(node)
					.unwrap_or_else(|| {
						panic!(
							"Bug in Op (id:{}), input Node (id:{}) has not been added as a member of the graph",
							op_id.id(),
							node.id()
						)
					})
					.insert(op_id);
			}
			relations.op_parents.insert(op_id, parents);
			let children = instance.outputs();
			for node in &children {
				relations
					.node_parents
					.get_mut(node)
					.unwrap_or_else(|| {
						panic!(
							"Bug in Op (id:{}), output Node (id:{}) has not been added as a member of the graph",
							op_id.id(),
							node.id()
						)
					})
					.insert(op_id);
			}
			relations.op_children.insert(op_id, children);
		}
	}

	fn remove_op(&mut self, op_id: OpID, parents: IndexSet<NodeID>, children: IndexSet<NodeID>) {
		if let Some(ref mut relations) = self.inner {
			for node in &parents {
				relations
					.node_children
					.get_mut(node)
					.unwrap_or_else(|| {
						panic!(
							"Bug in Op (id:{}), input Node (id:{}) has not been added as a member of the graph",
							op_id.id(),
							node.id()
						)
					})
					.swap_remove(&op_id);
			}
			relations.op_parents.swap_remove(&op_id);

			for node in &children {
				relations
					.node_parents
					.get_mut(node)
					.unwrap_or_else(|| {
						panic!(
							"Bug in Op (id:{}), output Node (id:{}) has not been added as a member of the graph",
							op_id.id(),
							node.id()
						)
					})
					.swap_remove(&op_id);
			}
			relations.op_children.swap_remove(&op_id);
		}
	}
}

#[derive(Default)]
struct RelationsInner {
	node_parents: IndexMap<NodeID, IndexSet<OpID>>,
	node_children: IndexMap<NodeID, IndexSet<OpID>>,
	op_parents: IndexMap<OpID, IndexSet<NodeID>>,
	op_children: IndexMap<OpID, IndexSet<NodeID>>,
}

impl RelationsInner {
	fn from_graph(
		nodes: &BTreeMap<NodeID, Arc<Mutex<NodeInnerData>>>,
		ops: &BTreeMap<OpID, Arc<Mutex<OpInnerData>>>,
	) -> Self {
		let mut relations = RelationsInner { ..Default::default() };

		for &node in nodes.keys() {
			relations.node_parents.insert(node, IndexSet::new());
			relations.node_children.insert(node, IndexSet::new());
		}

		for (&op, op_data) in ops {
			let inputs = op_data.lock().instance.inputs();
			for node in &inputs {
				relations.node_children.get_mut(node).unwrap().insert(op);
			}
			relations.op_parents.insert(op, inputs);

			let outputs = op_data.lock().instance.outputs();
			for node in &outputs {
				relations.node_parents.get_mut(node).unwrap().insert(op);
			}
			relations.op_children.insert(op, outputs);
		}

		relations
	}

	fn node_children(&mut self, node: NodeID) -> IndexSet<OpID> {
		self.node_children.get(&node).map_or_else(IndexSet::new, |r| r.clone())
	}

	fn node_parents(&mut self, node: NodeID) -> IndexSet<OpID> {
		self.node_parents.get(&node).map_or_else(IndexSet::new, |r| r.clone())
	}

	fn op_children(&mut self, op: &OpID) -> IndexSet<NodeID> {
		self.op_children.get(op).map_or_else(IndexSet::new, |r| r.clone())
	}

	fn op_parents(&mut self, op: &OpID) -> IndexSet<NodeID> {
		self.op_parents.get(op).map_or_else(IndexSet::new, |r| r.clone())
	}
}

#[derive(Default)]
struct Associations {
	inner: Option<AssociationsInner>,
}

impl Associations {
	fn get_or_instantiate_association(
		&mut self,
		nodes: &BTreeMap<NodeID, Arc<Mutex<NodeInnerData>>>,
		ops: &BTreeMap<OpID, Arc<Mutex<OpInnerData>>>,
	) -> &mut AssociationsInner {
		if self.inner.is_none() {
			self.inner = Some(AssociationsInner::from_graph(nodes, ops))
		}
		self.inner.as_mut().unwrap()
	}

	fn associate_node_name(&mut self, node: NodeID, name: &str) {
		if let Some(ref mut assoc) = self.inner {
			assoc
				.name_to_nodes
				.entry(name.to_string())
				.or_insert_with(IndexSet::new)
				.insert(node);
		}
	}

	fn associate_node_tag(&mut self, node: NodeID, tag: &NodeTag) {
		if let Some(ref mut assoc) = self.inner {
			assoc
				.tag_to_nodes
				.entry(tag.clone())
				.or_insert_with(IndexSet::new)
				.insert(node);
		}
	}

	fn associate_op_name(&mut self, op: OpID, name: &str) {
		if let Some(ref mut assoc) = self.inner {
			assoc
				.name_to_ops
				.entry(name.to_string())
				.or_insert_with(IndexSet::new)
				.insert(op);
		}
	}

	fn associate_op_tag(&mut self, op: OpID, tag: &OpTag) {
		if let Some(ref mut assoc) = self.inner {
			assoc
				.tag_to_ops
				.entry(tag.clone())
				.or_insert_with(IndexSet::new)
				.insert(op);
		}
	}

	fn unassociate_node_name(&mut self, node: NodeID, name: &str) {
		if let Some(ref mut assoc) = self.inner {
			assoc.name_to_nodes.get_mut(name).map(|set| set.swap_remove(&node));
		}
	}

	fn unassociate_node_tag(&mut self, node: NodeID, tag: &NodeTag) {
		if let Some(ref mut assoc) = self.inner {
			assoc.tag_to_nodes.get_mut(tag).map(|set| set.swap_remove(&node));
		}
	}

	fn unassociate_op_name(&mut self, op: OpID, name: &str) {
		if let Some(ref mut assoc) = self.inner {
			assoc.name_to_ops.get_mut(name).map(|set| set.swap_remove(&op));
		}
	}

	fn unassociate_op_tag(&mut self, op: OpID, tag: &OpTag) {
		if let Some(ref mut assoc) = self.inner {
			assoc.tag_to_ops.get_mut(tag).map(|set| set.swap_remove(&op));
		}
	}
}

/// denormalised for reverse lookups
#[derive(Default)]
struct AssociationsInner {
	name_to_nodes: IndexMap<String, IndexSet<NodeID>>,
	name_to_ops: IndexMap<String, IndexSet<OpID>>,
	tag_to_nodes: IndexMap<NodeTag, IndexSet<NodeID>>,
	tag_to_ops: IndexMap<OpTag, IndexSet<OpID>>,
}

impl AssociationsInner {
	fn from_graph(
		nodes: &BTreeMap<NodeID, Arc<Mutex<NodeInnerData>>>,
		ops: &BTreeMap<OpID, Arc<Mutex<OpInnerData>>>,
	) -> Self {
		let mut associations = Self { ..Default::default() };

		for (&node, data) in nodes {
			let NodeInnerData { ref name, ref tags, .. } = &*data.lock();
			associations
				.name_to_nodes
				.entry(name.clone())
				.or_insert_with(IndexSet::new)
				.insert(node);

			for tag in tags {
				associations
					.tag_to_nodes
					.entry(tag.clone())
					.or_insert_with(IndexSet::new)
					.insert(node);
			}
		}

		for (&op, data) in ops {
			let OpInnerData { ref name, ref tags, .. } = &*data.lock();
			associations
				.name_to_ops
				.entry(name.clone())
				.or_insert_with(IndexSet::new)
				.insert(op);

			for tag in tags {
				associations
					.tag_to_ops
					.entry(tag.clone())
					.or_insert_with(IndexSet::new)
					.insert(op);
			}
		}

		associations
	}

	fn nodes_tagged(&mut self, tag: &NodeTag) -> IndexSet<NodeID> {
		self.tag_to_nodes.get(tag).map_or_else(IndexSet::new, |r| r.clone())
	}

	fn nodes_named(&mut self, name: &str) -> IndexSet<NodeID> {
		self.name_to_nodes.get(name).map_or_else(IndexSet::new, |r| r.clone())
	}

	fn ops_tagged(&mut self, tag: &OpTag) -> IndexSet<OpID> {
		self.tag_to_ops.get(tag).map_or_else(IndexSet::new, |r| r.clone())
	}

	fn ops_named(&mut self, name: &str) -> IndexSet<OpID> {
		self.name_to_ops.get(name).map_or_else(IndexSet::new, |r| r.clone())
	}
}

/// A type used to mark nodes as parameters, or for easy retrival from a graph.
///
/// When calling `new_node()` consider using the `tag![]` macro to supply tags.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum NodeTag {
	/// Marks a node as a Parameter, for use when seeking. This should only be set on `Node`s with a fixed shape.
	Parameter,
	/// A customisable `NodeTag` which impl `From<usize>`.
	Int(usize),
	/// A customisable `NodeTag` which impl `From<String>` and `From<&str>`.
	Str(Arc<str>),
}

impl fmt::Display for NodeTag {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			NodeTag::Parameter => write!(f, "NodeTag::Parameter"),
			NodeTag::Int(int) => write!(f, "NodeTag::Int({})", int),
			NodeTag::Str(ref string) => write!(f, "NodeTag::Str({})", string),
		}
	}
}

impl From<usize> for NodeTag {
	fn from(i: usize) -> NodeTag {
		NodeTag::Int(i)
	}
}

impl<'a> From<&'a str> for NodeTag {
	fn from(i: &str) -> NodeTag {
		NodeTag::Str(i.into())
	}
}

impl From<String> for NodeTag {
	fn from(i: String) -> NodeTag {
		NodeTag::Str(i.into())
	}
}

/// A type used to mark Ops for easy retrival from a graph.
///
/// When calling `new_op()` consider using the `tag![]` macro to supply tags.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum OpTag {
	/// A customisable `Tag` which impl `From<usize>`.
	Int(usize),
	/// A customisable `Tag` which impl `From<String>` and `From<&str>`.
	Str(Arc<str>),
}

impl fmt::Display for OpTag {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			OpTag::Int(int) => write!(f, "OpTag::Int({})", int),
			OpTag::Str(ref string) => write!(f, "OpTag::Str({})", string),
		}
	}
}

impl From<usize> for OpTag {
	fn from(i: usize) -> OpTag {
		OpTag::Int(i)
	}
}

impl<'a> From<&'a str> for OpTag {
	fn from(i: &str) -> OpTag {
		OpTag::Str(i.into())
	}
}

impl From<String> for OpTag {
	fn from(i: String) -> OpTag {
		OpTag::Str(i.into())
	}
}

#[cfg(test)]
mod tests {
	use crate::{
		base_ops::noop::NoOpInstance,
		graph::{Graph, Node, NodeTag},
	};
	use std::sync::Arc;

	#[test]
	fn display() {
		let g = Graph::new();

		let n1 = g.new_node((&[1]).into()).set_name("n1");
		let n2 = g.new_node((&[1]).into()).set_name("n2");

		let o1 = g.new_op(Arc::new(NoOpInstance {})).set_name("o1");

		assert_eq!(&format!("{}", n2), "n2");
		assert_eq!(format!("{}", n1), "n1".to_string());
		assert_eq!(format!("{}", o1), "o1".to_string());
		assert_eq!(format!("{}", g), "Graph { nodes: [n1, n2], ops: [o1] }".to_string());
	}

	// Graph Tests

	#[test]
	fn graph_new() {
		let g = Graph::new();
		assert!(g.nodes().is_empty());
		assert!(g.ops().is_empty());
	}

	#[test]
	fn graph_locking() {
		let g = Graph::new();
		g.with_root_inner_mut(|_, _| {});

		let g2 = g.clone();

		g.with_locked_roots(&g2, |_, _, _, _| panic!(), |_, _| ());

		let g3 = Graph::new();

		g.with_locked_roots(&g3, |_, _, _, _| (), |_, _| panic!());
	}

	// Node Tests

	#[test]
	fn node_name() {
		let test_name = "test_name";
		let node = Node::new(&[-1, 1]).set_name(test_name);

		for node in node.graph().nodes().iter() {
			assert_eq!(node.name(), test_name);
		}

		let lookup_node = node.graph().node_named(test_name);

		assert_eq!(lookup_node, node);
	}

	#[test]
	fn node_name_duplicates_1() {
		let test_name = "test_name";
		let node1 = Node::new(&[-1, 1]).set_name(test_name);
		let node2 = Node::new(&[1, -1]).set_name(test_name);

		node1.graph().merge(node2.graph());

		for node in node1.graph().nodes().iter() {
			assert_eq!(node.name(), test_name);
		}

		let lookup_nodes = node1.graph().nodes_named(test_name);
		let mut iter = lookup_nodes.iter();

		assert_eq!(iter.next(), Some(&node1));
		assert_eq!(iter.next(), Some(&node2));
		assert_eq!(iter.next(), None);
	}

	#[test]
	#[should_panic]
	fn node_name_duplicates_2() {
		let test_name = "test_name";
		let node1 = Node::new(&[-1, 1]).set_name(test_name);
		let node2 = Node::new(&[1, -1]).set_name(test_name);

		node1.graph().merge(node2.graph());

		for node in node1.graph().nodes().iter() {
			assert_eq!(node.name(), test_name);
		}

		let _lookup_nodes: Node = node1.graph().node_named(test_name);
	}

	#[test]
	fn node_tag() {
		let test_tag = "test_tag";
		let node1 = Node::new(&[-1, 1]).set_name("1").add_tag(test_tag);
		let node2 = Node::new(&[1, -1]).set_name("2").add_tag(test_tag);

		assert_ne!(node1.graph(), node2.graph());
		node1.graph().merge(node2.graph());
		assert_eq!(node1.graph(), node2.graph());

		for node in node1.graph().nodes().iter() {
			assert!(node.tags().contains(&NodeTag::from(test_tag)));
		}

		let lookup_nodes = node1.graph().nodes_tagged(test_tag);
		let mut iter = lookup_nodes.iter();

		assert_eq!(iter.next(), Some(&node1));
		assert_eq!(iter.next(), Some(&node2));
		assert_eq!(iter.next(), None);
	}

	// Op Tests
}

// create new

// set name

// set tag

// set tags

// lookup by name

// lookup by tag

// TODO set of ops named x is empty after renaming x to y
