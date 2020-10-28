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
//! # 	let mut hasher = SipHasher::new();
//! # 	val.hash(&mut hasher);
//! # 	return hasher.finish()
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
use indexmap::{IndexMap, IndexSet};
use ndarray::{arr0, arr1, ArcArray, ArrayBase, ArrayD, Data, Dimension, IxDyn, OwnedArcRepr, OwnedRepr, ViewRepr};
use smallvec::SmallVec;
use std::{
	borrow::{Borrow, Cow},
	fmt::{self, Debug, Display},
	hash::{Hash, Hasher},
	ops::Deref,
	sync::{Arc, Mutex, MutexGuard, TryLockError, Weak},
};

// TODO write explanation for all the ways of creating a Node
pub trait ToNodeValue {
	fn to_value(self) -> ArcArray<f32, IxDyn>;
}

impl ToNodeValue for f32 {
	fn to_value(self) -> ArcArray<f32, IxDyn> {
		arr0(self).into_shared().into_dyn()
	}
}

impl ToNodeValue for &[f32] {
	fn to_value(self) -> ArcArray<f32, IxDyn> {
		arr1(self).into_shared().into_dyn()
	}
}

impl ToNodeValue for Vec<f32> {
	fn to_value(self) -> ArcArray<f32, IxDyn> {
		ArcArray::from(self).into_dyn()
	}
}

impl<D: Dimension> ToNodeValue for ArrayBase<ViewRepr<&f32>, D> {
	fn to_value(self) -> ArcArray<f32, IxDyn> {
		self.into_owned().into_shared().into_dyn()
	}
}

impl<D: Dimension> ToNodeValue for ArrayBase<ViewRepr<&mut f32>, D> {
	fn to_value(self) -> ArcArray<f32, IxDyn> {
		self.into_owned().into_shared().into_dyn()
	}
}

impl<D: Dimension> ToNodeValue for ArrayBase<OwnedRepr<f32>, D> {
	fn to_value(self) -> ArcArray<f32, IxDyn> {
		self.into_shared().into_dyn()
	}
}

impl<D: Dimension> ToNodeValue for ArrayBase<OwnedArcRepr<f32>, D> {
	fn to_value(self) -> ArcArray<f32, IxDyn> {
		self.into_dyn()
	}
}

impl<S: Data<Elem = f32>, D: Dimension> ToNodeValue for &ArrayBase<S, D> {
	fn to_value(self) -> ArcArray<f32, IxDyn> {
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
	inner: NodeInner,
}

impl Node {
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
	pub fn from_value<V: ToNodeValue>(value: V) -> Self {
		let value = value.to_value();
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
			let mut name = self
				.inner
				.data
				.name
				.lock()
				.expect("Mutex lock error when setting Node name");
			inner.associate_node_name(self.inner.clone(), &new_name, Some(&name));
			*name = new_name;
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
			let mut name = self
				.inner
				.data
				.name
				.lock()
				.expect("Mutex lock error when setting Node name");
			inner.associate_node_name(self.inner.clone(), &new_name, Some(&name));
			*name = new_name;
		});

		self.clone()
	}

	/// Associates a new tag with the node.
	pub fn add_tag<T: Into<NodeTag>>(&self, tag: T) -> Self {
		let tag = tag.into();
		self.graph.with_root_inner_mut(|_, inner| {
			let mut tags = self
				.inner
				.data
				.tags
				.lock()
				.expect("Mutex lock error when setting NodeTag");
			inner.associate_node_tag(self.inner.clone(), &tag);
			tags.insert(tag);
		});

		self.clone()
	}

	/// Associates new tags with the node.
	pub fn add_tags<I: IntoIterator<Item = NodeTag>>(&self, tags: I) -> Self {
		self.graph.with_root_inner_mut(|_, inner| {
			let mut inner_tags = self
				.inner
				.data
				.tags
				.lock()
				.expect("Mutex lock error when setting NodeTags");
			for tag in tags {
				inner.associate_node_tag(self.inner.clone(), &tag);
				inner_tags.insert(tag);
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
			let mut tags = self
				.inner
				.data
				.tags
				.lock()
				.expect("Mutex lock error when removing NodeTag");
			inner.associations = None;
			tags.swap_remove(&tag);
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
	pub fn set_value<V: ToNodeValue>(&self, new_value: V) -> Self {
		let new_value = new_value.to_value();
		let mut value = self
			.inner
			.data
			.value
			.lock()
			.unwrap_or_else(|_| panic!("Mutex lock error when setting value for Node: {}", self.name()));
		*value = Some(new_value);
		self.clone()
	}

	/// Clear the persistant value for the node.
	pub fn clear_value(&self) -> Self {
		let mut value = self
			.inner
			.data
			.value
			.lock()
			.unwrap_or_else(|_| panic!("Mutex lock error when setting value for Node: {}", self.name()));
		*value = None;
		self.clone()
	}

	/// Sets the initialiser for the node.
	///
	/// # Panics
	///
	/// Panics if node shape is not fully determined.
	pub fn set_init<I: Into<Initialiser>>(&self, new_init: I) -> Self {
		let mut init = self
			.inner
			.data
			.init
			.lock()
			.unwrap_or_else(|_| panic!("Mutex lock error when setting init for Node: {}", self.name()));
		*init = Some(new_init.into());
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
				.node_parents(&self.inner)
				.iter()
				.map(|inner| graph.op_from_inner_unchecked(inner.clone()))
				.collect()
		})
	}

	/// Returns the `Op` that output to this `Node`.
	///
	/// # Panics
	///
	/// Panics if number of input ops is not equal to 1.
	pub fn parent_op(&self) -> Op {
		self.graph
			.with_root_inner_mut(|graph, inner| graph.op_from_inner_unchecked(inner.node_parent(&self.inner)))
	}

	/// Returns the set of `Op`s that use this `Node` as an input.
	pub fn child_ops(&self) -> IndexSet<Op> {
		self.graph.with_root_inner_mut(|graph, inner| {
			inner
				.node_children(&self.inner)
				.iter()
				.map(move |inner| graph.op_from_inner_unchecked(inner.clone()))
				.collect()
		})
	}

	/// Returns the `Graph` of which this node is a member.
	pub fn graph(&self) -> &Graph {
		&self.graph
	}

	/// Returns a shared reference to the `NodeInner` which identifies this `Node`.
	pub fn inner(&self) -> &NodeInner {
		&self.inner
	}
}

impl PartialEq for Node {
	fn eq(&self, other: &Node) -> bool {
		self.inner == other.inner
	}
}

impl Eq for Node {}

impl Hash for Node {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.inner.hash(state)
	}
}

impl Display for Node {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> ::std::fmt::Result {
		Display::fmt(&self.inner(), fmt)
	}
}

impl Debug for Node {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> ::std::fmt::Result {
		fmt.debug_struct("Node")
			.field("name", &self.inner.data.name.lock().unwrap())
			.field("shape", &format!("{}", self.inner.data.shape))
			.finish()
	}
}

impl Deref for Node {
	type Target = NodeInner;

	fn deref(&self) -> &Self::Target {
		&self.inner
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

impl<V: ToNodeValue> From<V> for Node {
	fn from(v: V) -> Node {
		let v = v.to_value();
		Node::new(v.shape()).set_value(v)
	}
}

impl<V: ToNodeValue, S: Into<String>> From<(V, S)> for Node {
	fn from((v, s): (V, S)) -> Node {
		let v = v.to_value();
		Node::new(v.shape()).set_value(v).set_name(s)
	}
}

impl<T> From<HeavyNode<T>> for Node {
	fn from(hn: HeavyNode<T>) -> Node {
		hn.into_node()
	}
}

/// A reference to a node without reference to the `Graph`.
///
/// To avoid circular references this type is used inside `Graph` and `OpInstance`s, however most users should prefer
/// to use `Node` unless extending the library.
///
/// Implements methods which are possible without the graph.
pub struct NodeInner {
	/// This data should never be moved, its address is used as a unique id.
	data: Arc<NodeInnerData>,
}

impl NodeInner {
	/// Returns the memory address of the node data for use as a unique identifer.
	///
	/// Hash and Eq are implemented based on this value.
	pub fn id(&self) -> usize {
		&*self.data as *const NodeInnerData as usize
	}

	/// Returns a copy of the current `Node` name.
	///
	/// Default: Unnamed_Node
	pub fn name(&self) -> String {
		self.data
			.name
			.lock()
			.expect("Mutex lock error when getting Node name")
			.clone()
	}

	/// Returns a copy of the `Node` shape.
	pub fn shape(&self) -> &NodeShape {
		&self.data.shape
	}

	/// Returns a copy of the current `Node` tags.
	pub fn tags(&self) -> IndexSet<NodeTag> {
		self.data
			.tags
			.lock()
			.unwrap_or_else(|_| panic!("Mutex lock error when getting tags for Node: {}", self.name()))
			.clone()
	}

	pub fn has_value(&self) -> bool {
		self.data
			.value
			.lock()
			.unwrap_or_else(|_| panic!("Mutex lock error when setting value for Node: {}", self.name()))
			.is_some()
	}

	/// Returns copy-on-write value of the `Node`, if one has been set.
	///
	/// This avoids cloning the
	pub fn value(&self) -> Option<ArcArray<f32, IxDyn>> {
		self.data
			.value
			.lock()
			.unwrap_or_else(|_| panic!("Mutex lock error when setting value for Node: {}", self.name()))
			.clone()
	}

	/// Takes and returns value of the `Node`, if one has been set, leaving None in its place.
	pub fn take_value(&self) -> Option<ArcArray<f32, IxDyn>> {
		self.data
			.value
			.lock()
			.unwrap_or_else(|_| panic!("Mutex lock error when setting value for Node: {}", self.name()))
			.take()
	}

	/// Returns shape of the of the `Node`, if one has been set.
	pub fn value_shape(&self) -> Option<IxDyn> {
		self.data
			.value
			.lock()
			.unwrap_or_else(|_| panic!("Mutex lock error when setting value for Node: {}", self.name()))
			.as_ref()
			.map(|v| IxDyn(v.shape()))
	}

	/// Returns the `Initialiser` if one is set.
	pub fn init(&self) -> Option<Initialiser> {
		self.data
			.init
			.lock()
			.unwrap_or_else(|_| panic!("Mutex lock error when setting value for Node: {}", self.name()))
			.clone()
	}

	/// Returns the result of calling the initialiser.
	///
	/// # Panics
	/// Panics if the node is not a fixed shape (all known axes).
	pub fn init_array(&self) -> Option<ArrayD<f32>> {
		self.init().map(|mut init| init.array(self))
	}
}

impl Clone for NodeInner {
	fn clone(&self) -> Self {
		NodeInner {
			data: self.data.clone(),
		}
	}
}

impl PartialEq for NodeInner {
	fn eq(&self, other: &NodeInner) -> bool {
		self.id() == other.id()
	}
}

impl Eq for NodeInner {}

impl Hash for NodeInner {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.id().hash(state)
	}
}

impl Display for NodeInner {
	fn fmt(&self, f: &mut fmt::Formatter) -> ::std::fmt::Result {
		Display::fmt(&self.name(), f)
	}
}

impl Debug for NodeInner {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> ::std::fmt::Result {
		fmt.debug_struct("NodeInner")
			.field("name", &self.data.name.lock().unwrap())
			.field("shape", &format!("{}", self.data.shape))
			.finish()
	}
}

/// A weak reference that doesn't prevent deallocation
///
/// Used for implementing LRU caches.
pub struct WeakNodeInner {
	inner: Weak<NodeInnerData>,
	ptr_val: usize,
}

impl WeakNodeInner {
	/// returns the memory address of the InnerRef for use as a unique identifer
	///
	/// Hash and Eq are implemented based on this value
	pub fn id(&self) -> usize {
		self.ptr_val
	}
}

impl<'a> From<&'a NodeInner> for WeakNodeInner {
	fn from(val: &'a NodeInner) -> Self {
		WeakNodeInner {
			inner: Arc::downgrade(&val.data),
			ptr_val: val.id(),
		}
	}
}

impl Clone for WeakNodeInner {
	fn clone(&self) -> Self {
		WeakNodeInner {
			inner: self.inner.clone(),
			ptr_val: self.ptr_val,
		}
	}
}

impl PartialEq for WeakNodeInner {
	fn eq(&self, other: &WeakNodeInner) -> bool {
		if self.ptr_val != other.ptr_val {
			return false;
		}
		match (self.inner.upgrade(), other.inner.upgrade()) {
			(Some(_), Some(_)) | (None, None) => true,
			_ => false,
		}
	}
}

impl Eq for WeakNodeInner {}

impl Hash for WeakNodeInner {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.ptr_val.hash(state)
	}
}

struct NodeInnerData {
	shape: NodeShape,
	name: Mutex<String>,
	tags: Mutex<IndexSet<NodeTag>>,
	value: Mutex<Option<ArcArray<f32, IxDyn>>>,
	init: Mutex<Option<Initialiser>>,
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
	inner: OpInner,
}

impl Op {
	/// Returns a copy of the current `Op` name.
	///
	/// Default: Unnamed_Op
	///
	/// Usually this default is overwritten by a name derived from parent and child `Node`s.
	pub fn name(&self) -> String {
		self.inner.name()
	}

	/// Returns a copy of the current `Node` tags.
	pub fn tags(&self) -> IndexSet<OpTag> {
		self.inner.tags()
	}

	/// Set the name of the Op, replaces existing name.
	pub fn set_name<S: Into<String>>(&self, new_name: S) -> Self {
		let new_name = new_name.into();
		self.graph.with_root_inner_mut(|_, inner| {
			let mut name = self
				.inner
				.data
				.name
				.lock()
				.expect("Mutex lock error when setting Op name");
			inner.associate_op_name(self.inner.clone(), &new_name, Some(&name));
			*name = new_name;
		});

		self.clone()
	}

	/// Sets the name of the Op, replaces existing name. The name root is extended with an integer, counting up from 0
	/// until a unique name is found.
	pub fn set_name_unique(&self, new_name_root: &str) -> Self {
		self.graph.with_root_inner_mut(|_, inner| {
			let new_name = inner.unique_op_name(new_name_root);
			let mut name = self
				.inner
				.data
				.name
				.lock()
				.expect("Mutex lock error when setting Op name");
			inner.associate_op_name(self.inner.clone(), &new_name, Some(&name));
			*name = new_name;
		});

		self.clone()
	}

	/// Associates a new tag with the op.
	pub fn add_tag<T: Into<OpTag>>(&self, tag: T) -> Self {
		let tag = tag.into();
		self.graph.with_root_inner_mut(|_, inner| {
			let mut tags = self
				.inner
				.data
				.tags
				.lock()
				.expect("Mutex lock error when setting OpTag");
			inner.associate_op_tag(self.inner.clone(), &tag);
			tags.insert(tag);
		});

		self.clone()
	}

	/// Associates a new tags with the op.
	pub fn add_tags<I: IntoIterator<Item = OpTag>>(&self, tags: I) -> Self {
		self.graph.with_root_inner_mut(|_, inner| {
			let mut inner_tags = self
				.inner
				.data
				.tags
				.lock()
				.expect("Mutex lock error when setting OpTags");
			for tag in tags {
				inner.associate_op_tag(self.inner.clone(), &tag);
				inner_tags.insert(tag);
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
			let mut tags = self
				.inner
				.data
				.tags
				.lock()
				.expect("Mutex lock error when removing OpTag");
			inner.associations = None;
			tags.swap_remove(&tag);
		});

		self.clone()
	}

	pub fn instance(&self) -> &dyn OpInstance {
		&*self.inner.instance()
	}

	// TODO this doesn't uphold the invariant that an OpInstance is always in the same graph as its inputs/outputs
	// Need to use conversion to OpBuilder and replace nodes
	// pub fn clone_instance(&self) -> Op {
	// 	let graph = Graph::new();
	// 	graph
	// 		.new_op(self.inner.data.instance.clone())
	// 		.set_name(self.inner.data.name.lock().unwrap().clone())
	// 		.add_tags(self.inner.data.tags.lock().unwrap().clone())
	// }

	/// Returns the set of `Node`s this `Op` uses as inputs in the `Graph`.
	pub fn parent_nodes(&self) -> IndexSet<Node> {
		self.graph.with_root_inner_mut(|graph, inner| {
			inner
				.op_parents(&self.inner)
				.iter()
				.map(|inner| graph.node_from_inner_unchecked(inner.clone()))
				.collect()
		})
	}

	/// Returns the set of `Node`s this `Op` outputs to in the `Graph`.
	pub fn child_nodes(&self) -> IndexSet<Node> {
		self.graph.with_root_inner_mut(|graph, inner| {
			inner
				.op_children(&self.inner)
				.iter()
				.map(|inner| graph.node_from_inner_unchecked(inner.clone()))
				.collect()
		})
	}

	/// Returns the `Graph` of which this `Op` is a member.
	pub fn graph(&self) -> &Graph {
		&self.graph
	}

	/// Returns a shared reference to the `OpInner` which identifies this `Op`.
	pub fn inner(&self) -> &OpInner {
		&self.inner
	}
}

impl PartialEq for Op {
	fn eq(&self, other: &Op) -> bool {
		self.inner == other.inner
	}
}

impl Eq for Op {}

impl Hash for Op {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.inner.hash(state)
	}
}

impl Display for Op {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> ::std::fmt::Result {
		Display::fmt(&self.inner(), fmt)
	}
}

impl Debug for Op {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> ::std::fmt::Result {
		fmt.debug_struct("Op")
			.field("name", &self.data.name.lock().unwrap())
			.field("type", &self.data.instance.type_name())
			.finish()
	}
}

impl Deref for Op {
	type Target = OpInner;

	fn deref(&self) -> &Self::Target {
		&self.inner
	}
}

/// A reference to an op without reference to the `Graph`.
///
/// To avoid circular references this type is used inside `Graph` and `OpInstance`s, however most users should prefer
/// to use `Op` unless extending the library.
///
/// Implements methods which are possible without the graph.
pub struct OpInner {
	/// This data should never be moved, its address is used as a unique id.
	data: Arc<OpInnerData>,
}

impl OpInner {
	/// Returns the memory address of the op data for use as a unique identifer.
	///
	/// Hash and Eq are implemented based on this value.
	pub fn id(&self) -> usize {
		&*self.data as *const OpInnerData as usize
	}

	/// Returns a copy of the current `Op` name.
	///
	/// Default: Unnamed_Op
	///
	/// Usually this default is overwritten by a name derived from parent and child `Node`s.
	pub fn name(&self) -> String {
		self.data
			.name
			.lock()
			.expect("Mutex lock error when getting Op name")
			.clone()
	}

	/// Returns a copy of the current `Node` tags.
	pub fn tags(&self) -> IndexSet<OpTag> {
		self.data
			.tags
			.lock()
			.unwrap_or_else(|_| panic!("Mutex lock error when getting tags for op: {}", self.name()))
			.clone()
	}

	pub fn instance(&self) -> &dyn OpInstance {
		&*self.data.instance
	}
}

impl Clone for OpInner {
	fn clone(&self) -> Self {
		OpInner {
			data: self.data.clone(),
		}
	}
}

impl PartialEq for OpInner {
	fn eq(&self, other: &OpInner) -> bool {
		self.id() == other.id()
	}
}

impl Eq for OpInner {}

impl Hash for OpInner {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.id().hash(state)
	}
}

impl Display for OpInner {
	fn fmt(&self, f: &mut fmt::Formatter) -> ::std::fmt::Result {
		Display::fmt(&self.name(), f)
	}
}

impl Debug for OpInner {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> ::std::fmt::Result {
		fmt.debug_struct("OpInner")
			.field("name", &self.data.name.lock().unwrap())
			.field("type", &self.data.instance.type_name())
			.finish()
	}
}

/// A weak reference that doesn't prevent deallocation
///
/// Used for implementing LRU caches.
pub struct WeakOpInner {
	inner: Weak<OpInnerData>,
	ptr_val: usize,
}

impl WeakOpInner {
	/// returns the memory address of the InnerRef for use as a unique identifer
	///
	/// Hash and Eq are implemented based on this value
	pub fn id(&self) -> usize {
		self.ptr_val
	}
}

impl<'a> From<&'a OpInner> for WeakOpInner {
	fn from(val: &'a OpInner) -> Self {
		WeakOpInner {
			inner: Arc::downgrade(&val.data),
			ptr_val: val.id(),
		}
	}
}

impl Clone for WeakOpInner {
	fn clone(&self) -> Self {
		WeakOpInner {
			inner: self.inner.clone(),
			ptr_val: self.ptr_val,
		}
	}
}

impl PartialEq for WeakOpInner {
	fn eq(&self, other: &WeakOpInner) -> bool {
		if self.ptr_val != other.ptr_val {
			return false;
		}
		match (self.inner.upgrade(), other.inner.upgrade()) {
			(Some(_), Some(_)) | (None, None) => true,
			_ => false,
		}
	}
}

impl Eq for WeakOpInner {}

impl Hash for WeakOpInner {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.ptr_val.hash(state)
	}
}

struct OpInnerData {
	name: Mutex<String>,
	tags: Mutex<IndexSet<OpTag>>,
	instance: Box<dyn OpInstance>,
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
		let mut self_link: MutexGuard<GraphLink> = match self.link.try_lock() {
			Ok(guard) => guard,
			Err(TryLockError::WouldBlock) => {
				return Graph {
					link: self.link.clone(),
				};
			}
			x @ Err(_) => x.expect("Graph mutex is poisoned"), // If poisoned, just crash and burn
		};

		// get root by either being root, or cloning
		let root = match &mut *self_link {
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
		self.with_root_link_mut(|graph, _link| (&*graph.link as *const Mutex<GraphLink> as usize).hash(state))
	}
}

impl Display for Graph {
	fn fmt(&self, f: &mut fmt::Formatter) -> ::std::fmt::Result {
		// self.with_nodes_ops(|nodes, ops| {
		// 	write!(f, "Graph {{ nodes: {}, ops: {} }}", IterDisplay{inner: nodes.clone()}, IterDisplay{inner:
		// ops.clone()})?; 	Ok(())
		// })

		self.with_nodes_ops(|nodes, ops| {
			write!(f, "Graph {{ nodes: [")?;
			let mut iter = nodes.iter();
			if let Some(node) = iter.next() {
				write!(f, "{}", node)?;
				for node in iter {
					write!(f, ", {}", node)?;
				}
			}
			write!(f, "], ops: [")?;
			let mut iter = ops.iter();
			if let Some(op) = iter.next() {
				write!(f, "{}", op)?;
				for op in iter {
					write!(f, ", {}", op)?;
				}
			}
			write!(f, "] }}")?;
			Ok(())
		})
	}
}

impl Debug for Graph {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> ::std::fmt::Result {
		self.with_nodes_ops(|nodes, ops| {
			fmt.debug_struct("Graph")
				.field("nodes", &IterDebug { inner: nodes.clone() })
				.field("ops", &IterDebug { inner: ops.clone() })
				.finish()
		})
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
		let node_inner = NodeInner {
			data: Arc::new(NodeInnerData {
				shape,
				name: Mutex::new("Unnamed_Node".to_string()),
				tags: Mutex::new(IndexSet::new()),
				value: Mutex::new(None),
				init: Mutex::new(None),
			}),
		};

		self.with_root_inner_mut(|graph, inner| {
			inner.add_node(node_inner.clone());
			graph.node_from_inner_unchecked(node_inner)
		})
	}

	/// Create a new `Op` within the Graph
	pub fn new_op(&self, instance: Box<dyn OpInstance>) -> Op {
		#[allow(clippy::block_in_if_condition_stmt)]
		{
			debug_assert!(
				self.with_root_inner_mut(|_, inner|{
					instance.inputs().iter().chain(&instance.outputs()).all(|node| inner.nodes.contains(node))
				}),
				"New OpInstance lists input or output nodes that are not part of this Graph. Prior to constructing an OpInstance, Graphs must be merged."
			);
		}

		let op_inner = OpInner {
			data: Arc::new(OpInnerData {
				name: Mutex::new("Unnamed_Op".to_string()),
				tags: Mutex::new(IndexSet::new()),
				instance,
			}),
		};

		self.with_root_inner_mut(|graph, inner| {
			inner.add_op(op_inner.clone());
			graph.op_from_inner_unchecked(op_inner)
		})
	}

	/// Create a `Node` from a `NodeInner`
	///
	/// # Panics
	/// Panics if the NodeInner is not a member of this graph.
	pub fn node_from(&self, inner: NodeInner) -> Node {
		assert!(
			self.with_nodes_ops(|nodes, _| nodes.contains(&inner)),
			"NodeInner {} is not a part of this Graph.",
			inner
		);
		Node {
			graph: self.clone(),
			inner,
		}
	}

	/// Create an `Op` from an `OpInner`
	///
	/// # Panics
	/// Panics if the OpInner is not a member of this graph.
	pub fn op_from_inner(&self, inner: OpInner) -> Op {
		assert!(
			self.with_nodes_ops(|_, ops| ops.contains(&inner)),
			"OpInner {} is not a part of this Graph.",
			inner
		);
		Op {
			graph: self.clone(),
			inner,
		}
	}

	/// Create a `Node` from a `NodeInner`
	fn node_from_inner_unchecked(&self, inner: NodeInner) -> Node {
		Node {
			graph: self.clone(),
			inner,
		}
	}

	/// Create an `Op` from an `OpInner`
	fn op_from_inner_unchecked(&self, inner: OpInner) -> Op {
		Op {
			graph: self.clone(),
			inner,
		}
	}

	/// Return the set of `Node`s associated with the given tag.
	pub fn nodes_tagged<T: Into<NodeTag>>(&self, tag: T) -> IndexSet<Node> {
		self.with_root_inner_mut(|graph, inner| {
			inner
				.nodes_tagged(&tag.into())
				.iter()
				.map(|inner| graph.node_from_inner_unchecked(inner.clone()))
				.collect()
		})
	}

	/// Return the set of `Node`s associated with the given name.
	pub fn nodes_named(&self, name: &str) -> IndexSet<Node> {
		self.with_root_inner_mut(|graph, inner| {
			inner
				.nodes_named(name)
				.iter()
				.map(|inner| graph.node_from_inner_unchecked(inner.clone()))
				.collect()
		})
	}

	/// Return the `Node` associated with the given name.
	///
	/// # Panics
	///
	/// Panics if number of `Node`s found is not equal to 1.
	pub fn node_named(&self, name: &str) -> Node {
		self.with_root_inner_mut(|graph, inner| graph.node_from_inner_unchecked(inner.node_named(name)))
	}

	/// Return the set of `Op`s associated with the given tag.
	pub fn ops_tagged<T: Into<OpTag>>(&self, tag: T) -> IndexSet<Op> {
		self.with_root_inner_mut(|graph, inner| {
			inner
				.ops_tagged(&tag.into())
				.iter()
				.map(|inner| graph.op_from_inner_unchecked(inner.clone()))
				.collect()
		})
	}

	/// Return the set of `Op`s associated with the given name.
	pub fn ops_named(&self, name: &str) -> IndexSet<Op> {
		self.with_root_inner_mut(|graph, inner| {
			inner
				.ops_named(name)
				.iter()
				.map(|inner| graph.op_from_inner_unchecked(inner.clone()))
				.collect()
		})
	}

	/// Return the `Op` associated with the given name.
	///
	/// # Panics
	///
	/// Panics if number of `Op`s found is not equal to 1.
	pub fn op_named(&self, name: &str) -> Op {
		self.with_root_inner_mut(|graph, inner| graph.op_from_inner_unchecked(inner.op_named(name)))
	}

	/// Returns a clone of the current set of `Node`s in the graph.
	///
	/// Does not update with modifications to the graph.
	pub fn nodes(&self) -> IndexSet<Node> {
		self.with_root_inner_mut(|graph, inner| {
			inner
				.nodes
				.iter()
				.map(|inner| graph.node_from_inner_unchecked(inner.clone()))
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
				.iter()
				.map(|inner| graph.op_from_inner_unchecked(inner.clone()))
				.collect()
		})
	}

	pub fn node_count(&self) -> usize {
		self.with_nodes_ops(|nodes, _ops| nodes.len())
	}

	pub fn op_count(&self) -> usize {
		self.with_nodes_ops(|_nodes, ops| ops.len())
	}

	/// Merges `g2` into self.
	pub fn merge(&self, g2: &Graph) {
		if self.link_addr() != g2.link_addr() {
			self.with_locked_roots(
				g2,
				|g1: &Graph, gl1: &mut GraphLink, _g2: &Graph, gl2: &mut GraphLink| {
					let mut g2_inner = gl2.take_inner(g1).unwrap();

					let g1_inner: &mut GraphInner = gl1.inner_mut().unwrap();

					for node in g2_inner.nodes.drain(..) {
						g1_inner.add_node(node)
					}

					for op in g2_inner.ops.drain(..) {
						g1_inner.add_op(op)
					}
				},
				|_, _| {},
			);
		}
	}

	fn lock_graph_link(&self) -> MutexGuard<GraphLink> {
		self.link
			.lock()
			.expect("Another thread has poisoned (panicked while holding) the GraphLink Mutex")
	}

	/// Finds the graph root, locks it, then runs the provided closure
	fn with_root_link_mut<T, F: FnOnce(&Graph, &mut GraphLink) -> T>(&self, f: F) -> T {
		let mut graph = self.clone();
		loop {
			{
				let mut graph_link: MutexGuard<GraphLink> = graph.lock_graph_link();
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

	/// Run a closure taking the sets of `Node`s and `Op`s in the graph.
	///
	/// Useful to avoid cloning the sets of ops and nodes in the graph by running the closure while the graph internals
	/// are locked. However, caution is advised as nesting multiple calls, or triggering any other lock of the graph
	/// will deadlock.
	pub fn with_nodes_ops<T, F: FnOnce(&IndexSet<NodeInner>, &IndexSet<OpInner>) -> T>(&self, f: F) -> T {
		self.with_root_inner_mut(|_, inner| f(&inner.nodes, &inner.ops))
	}

	/// Find and lock the roo(s). Run either of two functions depending on whether the two graphs share a root.
	fn with_locked_roots<T, F, G>(&self, g2: &Graph, different_root: F, same_root: G) -> T
	where
		F: FnOnce(&Graph, &mut GraphLink, &Graph, &mut GraphLink) -> T,
		G: FnOnce(&Graph, &mut GraphLink) -> T,
	{
		// Get the current graph roots. Because we cant assume other threads arent merging graphs and these dont lock,
		// they might not still be the roots.
		let mut g1 = self.clone();
		let mut g2 = g2.clone();

		loop {
			// Lock graph with lower memory address first to avoid dining philosophers problem.
			if g1.link_addr() < g2.link_addr() {
				let mut g1_link: MutexGuard<GraphLink> = g1.lock_graph_link();
				if let gl1 @ &mut GraphLink::Root(_) = &mut *g1_link {
					let mut g2_link: MutexGuard<GraphLink> = g2.lock_graph_link();
					if let gl2 @ &mut GraphLink::Root(_) = &mut *g2_link {
						return different_root(&g1, gl1, &g2, gl2);
					}
				}
			} else if g1.link_addr() > g2.link_addr() {
				let mut g2_link: MutexGuard<GraphLink> = g2.lock_graph_link();
				if let gl2 @ &mut GraphLink::Root(_) = &mut *g2_link {
					let mut g1_link: MutexGuard<GraphLink> = g1.lock_graph_link();
					if let gl1 @ &mut GraphLink::Root(_) = &mut *g1_link {
						return different_root(&g1, gl1, &g2, gl2);
					}
				}
			} else {
				let mut g1_link: MutexGuard<GraphLink> = g1.lock_graph_link();
				match &mut *g1_link {
					x @ &mut GraphLink::Root(_) => return same_root(&g1, x),
					&mut GraphLink::MergedInto(_) => continue,
				}
			}

			// Cloning acquires a root reference, unless we would block
			// If it doesnt, then we are already going to block when we lock
			// it will get closer in future iterations
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
	nodes: IndexSet<NodeInner>,
	ops: IndexSet<OpInner>,

	// De-normalised data only
	// NodeInner and OpInner contain the primary record
	associations: Option<Associations>,
	relations: Option<Relations>,
	/* Caches for execution. These do not form part of the graph definition.
	 * shape_cache: ShapeCache,
	 * execution_order_cache: ExecutionOrderCache, */
}

impl GraphInner {
	fn node_children(&mut self, node: &NodeInner) -> Cow<IndexSet<OpInner>> {
		self.get_or_instantiate_relations()
			.node_children
			.get(node)
			.map_or_else(|| Cow::Owned(IndexSet::new()), |r| Cow::Borrowed(r))
	}

	fn node_parents(&mut self, node: &NodeInner) -> Cow<IndexSet<OpInner>> {
		self.get_or_instantiate_relations()
			.node_parents
			.get(node)
			.map_or_else(|| Cow::Owned(IndexSet::new()), |r| Cow::Borrowed(r))
	}

	fn node_parent(&mut self, node: &NodeInner) -> OpInner {
		let mut iter = self
			.get_or_instantiate_relations()
			.node_parents
			.get(node)
			.unwrap_or_else(|| panic!("Node: {} is not a member of this graph", node.name()))
			.iter();

		let op = iter
			.next()
			.unwrap_or_else(|| panic!("No parent Op for Node: {}", node.name()));
		assert!(
			iter.next().is_none(),
			"More than one parent Op for Node : {}",
			node.name()
		);
		op.clone()
	}

	fn op_children(&mut self, op: &OpInner) -> Cow<IndexSet<NodeInner>> {
		self.get_or_instantiate_relations()
			.op_children
			.get(op)
			.map_or_else(|| Cow::Owned(IndexSet::new()), |r| Cow::Borrowed(r))
	}

	fn op_parents(&mut self, op: &OpInner) -> Cow<IndexSet<NodeInner>> {
		self.get_or_instantiate_relations()
			.op_parents
			.get(op)
			.map_or_else(|| Cow::Owned(IndexSet::new()), |r| Cow::Borrowed(r))
	}

	fn nodes_tagged(&mut self, tag: &NodeTag) -> Cow<IndexSet<NodeInner>> {
		self.get_or_instantiate_association()
			.tag_to_nodes
			.get(tag)
			.map_or_else(|| Cow::Owned(IndexSet::new()), |r| Cow::Borrowed(r))
	}

	fn nodes_named(&mut self, name: &str) -> Cow<IndexSet<NodeInner>> {
		self.get_or_instantiate_association()
			.name_to_nodes
			.get(name)
			.map_or_else(|| Cow::Owned(IndexSet::new()), |r| Cow::Borrowed(r))
	}

	fn node_named(&mut self, name: &str) -> NodeInner {
		let iter_opt = self
			.get_or_instantiate_association()
			.name_to_nodes
			.get(name)
			.map(IndexSet::iter);

		if let Some(mut iter) = iter_opt {
			let node = iter
				.next()
				.unwrap_or_else(|| panic!("No Node associated with name: {}", name));
			assert!(
				iter.next().is_none(),
				"More than one parent Op for Node : {}",
				node.name()
			);
			node.clone()
		} else {
			panic!("No Node associated with name: {}", name)
		}
	}

	fn ops_tagged<'a>(&'a mut self, tag: &OpTag) -> Cow<'a, IndexSet<OpInner>> {
		self.get_or_instantiate_association()
			.tag_to_ops
			.get(tag)
			.map_or_else(|| Cow::Owned(IndexSet::new()), |r| Cow::Borrowed(r))
	}

	fn ops_named(&mut self, name: &str) -> Cow<IndexSet<OpInner>> {
		self.get_or_instantiate_association()
			.name_to_ops
			.get(name)
			.map_or_else(|| Cow::Owned(IndexSet::new()), |r| Cow::Borrowed(r))
	}

	fn op_named(&mut self, name: &str) -> OpInner {
		let iter_opt = self
			.get_or_instantiate_association()
			.name_to_ops
			.get(name)
			.map(IndexSet::iter);

		if let Some(mut iter) = iter_opt {
			let op = iter
				.next()
				.unwrap_or_else(|| panic!("No Node associated with name: {}", name));
			assert!(
				iter.next().is_none(),
				"More than one parent Op for Node : {}",
				op.name()
			);
			op.clone()
		} else {
			panic!("No Node associated with name: {}", name)
		}
	}

	fn get_or_instantiate_relations(&mut self) -> &mut Relations {
		if self.relations.is_none() {
			self.relations = Some(Relations::from_graph(self))
		}
		self.relations.as_mut().unwrap()
	}

	fn get_or_instantiate_association(&mut self) -> &mut Associations {
		if self.associations.is_none() {
			self.associations = Some(Associations::from_graph(self))
		}
		self.associations.as_mut().unwrap()
	}

	/// Add an existing node to the graph.
	/// Updates membership in the graph and denormalised data for names and tags.
	/// Does not add information for relationships to ops.
	fn add_node(&mut self, node: NodeInner) {
		self.nodes.insert(node.clone());

		let node_name = node.data.name.lock().expect("Mutex lock error when getting Node name");
		self.associate_node_name(node.clone(), &node_name, None);

		let node_tags = node
			.data
			.tags
			.lock()
			.unwrap_or_else(|_| panic!("Mutex lock error when getting tags for Node: {}", node.name()));

		for tag in node_tags.iter() {
			self.associate_node_tag(node.clone(), tag);
		}

		if let Some(ref mut relations) = self.relations {
			relations.node_parents.insert(node.clone(), IndexSet::new());
			relations.node_children.insert(node.clone(), IndexSet::new());
		}
	}

	/// Add an existing `Op` to the graph
	///
	/// # Panics
	/// May panic if input and output nodes haven't already been added.
	fn add_op(&mut self, op: OpInner) {
		self.ops.insert(op.clone());

		let op_name = op.data.name.lock().expect("Mutex lock error when getting Op name");
		self.associate_op_name(op.clone(), &*op_name, None);

		let op_tags = op
			.data
			.tags
			.lock()
			.unwrap_or_else(|_| panic!("Mutex lock error when getting tags for Op: {}", op.name()));

		for tag in op_tags.iter() {
			self.associate_op_tag(op.clone(), tag);
		}

		if let Some(ref mut relations) = self.relations {
			let inputs = op.instance().inputs();
			for node in &inputs {
				relations
					.node_children
					.get_mut(node)
					.unwrap_or_else(|| {
						panic!(
							"Bug in Op: {}, input Node: {} has not been added as a member of the graph",
							op.name(),
							node.name()
						)
					})
					.insert(op.clone());
			}
			relations.op_parents.insert(op.clone(), inputs);

			let outputs = op.instance().outputs();
			for node in &outputs {
				relations
					.node_parents
					.get_mut(node)
					.unwrap_or_else(|| {
						panic!(
							"Bug in Op: {}, output Node: {} has not been added as a member of the graph",
							op.name(),
							node.name()
						)
					})
					.insert(op.clone());
			}
			relations.op_children.insert(op.clone(), outputs);
		}
	}

	fn associate_node_name(&mut self, node: NodeInner, name: &str, old_name: Option<&str>) {
		if let Some(ref mut assoc) = self.associations {
			// remove from previous name group
			if let Some(old_name) = old_name {
				assoc.name_to_nodes.get_mut(old_name).map(|set| set.swap_remove(&node));
			}

			// add to new name group
			assoc
				.name_to_nodes
				.entry(name.to_string())
				.or_insert_with(IndexSet::new)
				.insert(node);
		}
	}

	fn associate_node_tag(&mut self, node: NodeInner, tag: &NodeTag) {
		if let Some(ref mut assoc) = self.associations {
			// add to new tag group
			assoc
				.tag_to_nodes
				.entry(tag.clone())
				.or_insert_with(IndexSet::new)
				.insert(node);
		}
	}

	fn associate_op_name(&mut self, op: OpInner, name: &str, old_name: Option<&str>) {
		if let Some(ref mut assoc) = self.associations {
			// remove from previous name group
			if let Some(old_name) = old_name {
				assoc.name_to_ops.get_mut(old_name).map(|set| set.swap_remove(&op));
			}

			// add to new name group
			assoc
				.name_to_ops
				.entry(name.to_string())
				.or_insert_with(IndexSet::new)
				.insert(op);
		}
	}

	fn associate_op_tag(&mut self, op: OpInner, tag: &OpTag) {
		if let Some(ref mut assoc) = self.associations {
			assoc
				.tag_to_ops
				.entry(tag.clone())
				.or_insert_with(IndexSet::new)
				.insert(op);
		}
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
		return new_name_root.to_string();
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
		return new_name_root.to_string();
	}
}

impl Display for GraphInner {
	fn fmt(&self, f: &mut fmt::Formatter) -> ::std::fmt::Result {
		write!(f, "GraphInner {{ nodes: [")?;
		let mut iter = self.nodes.iter();
		if let Some(node) = iter.next() {
			write!(f, "{}", node)?;
			for node in iter {
				write!(f, ", {}", node)?;
			}
		}
		write!(f, "], ops: [")?;
		let mut iter = self.ops.iter();
		if let Some(op) = iter.next() {
			write!(f, "{}", op)?;
			for op in iter {
				write!(f, ", {}", op)?;
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

#[derive(Debug, Default)]
struct Relations {
	node_parents: IndexMap<NodeInner, IndexSet<OpInner>>,
	node_children: IndexMap<NodeInner, IndexSet<OpInner>>,
	op_parents: IndexMap<OpInner, IndexSet<NodeInner>>,
	op_children: IndexMap<OpInner, IndexSet<NodeInner>>,
}

impl Relations {
	fn from_graph(graph: &GraphInner) -> Self {
		let mut relations = Relations { ..Default::default() };

		for node in &graph.nodes {
			relations.node_parents.insert(node.clone(), IndexSet::new());
			relations.node_children.insert(node.clone(), IndexSet::new());
		}

		for op in &graph.ops {
			let inputs = op.instance().inputs();
			for node in &inputs {
				relations.node_children.get_mut(node).unwrap().insert(op.clone());
			}
			relations.op_parents.insert(op.clone(), inputs);

			let outputs = op.instance().outputs();
			for node in &outputs {
				relations.node_parents.get_mut(node).unwrap().insert(op.clone());
			}
			relations.op_children.insert(op.clone(), outputs);
		}

		relations
	}
}

/// denormalised for reverse lookups
#[derive(Debug, Default)]
struct Associations {
	name_to_nodes: IndexMap<String, IndexSet<NodeInner>>,
	name_to_ops: IndexMap<String, IndexSet<OpInner>>,
	tag_to_nodes: IndexMap<NodeTag, IndexSet<NodeInner>>,
	tag_to_ops: IndexMap<OpTag, IndexSet<OpInner>>,
}

impl Associations {
	fn from_graph(graph: &GraphInner) -> Self {
		let mut associations = Associations { ..Default::default() };

		for node in &graph.nodes {
			associations
				.name_to_nodes
				.entry(node.name())
				.or_insert_with(IndexSet::new)
				.insert(node.clone());

			for tag in node.tags() {
				associations
					.tag_to_nodes
					.entry(tag)
					.or_insert_with(IndexSet::new)
					.insert(node.clone());
			}
		}

		for op in &graph.ops {
			associations
				.name_to_ops
				.entry(op.name())
				.or_insert_with(IndexSet::new)
				.insert(op.clone());

			for tag in op.tags() {
				associations
					.tag_to_ops
					.entry(tag)
					.or_insert_with(IndexSet::new)
					.insert(op.clone());
			}
		}

		associations
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

	#[test]
	fn display() {
		let g = Graph::new();

		let n1 = g.new_node((&[1]).into()).set_name("n1");
		let n2 = g.new_node((&[1]).into()).set_name("n2");

		let o1 = g.new_op(Box::new(NoOpInstance {})).set_name("o1");

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