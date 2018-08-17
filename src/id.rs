use shape::NodeShape;
use indexmap::IndexSet;
use ops::*;
use std::borrow::Borrow;
use std::sync::Arc;
use std::hash::{Hash, Hasher};
use std::fmt;

//#[derive(PartialEq, Eq, Ord, PartialOrd, Hash, Clone, Debug)]
#[derive(Clone, Debug)]
struct NodeDesc {
	name: String,
	shape: NodeShape,
	tags: IndexSet<NodeTag>,
}


trait OpDescTrait: fmt::Debug + Send + Sync{
	fn instance(&self) -> &OpInstance;

	fn tags(&self) -> &IndexSet<OpTag>;
}
impl<O: OpInstance> OpDescTrait for OpDesc<O> {
	fn instance(&self) -> &OpInstance {
		&self.instance
	}

	fn tags(&self) -> &IndexSet<OpTag> {
		&self.tags
	}
}

#[derive(Clone, Debug)]
struct OpDesc<O: OpInstance> {
	instance: O,
	tags: IndexSet<OpTag>,
}

/// A unique identifier for a node in the computational graph
#[derive(Clone, Debug)]
pub struct NodeID {
	id: usize,
	desc: Arc<NodeDesc>,
	//shape: Arc<NodeShape>, // op builders should be able to just get it from the graphbuilders
}

impl NodeID {
	pub fn new(id: usize, name: String, shape: NodeShape, tags: IndexSet<NodeTag>) -> Self {
		NodeID {
			id,
			desc: Arc::new(NodeDesc{
				name,
				shape,
				tags,
			}),
		}
	}

	pub fn value_id(&self) -> DataID {
		DataID{id: self.id * 2, desc: self.desc.clone()}//, shape: self.shape.clone()}
	}

	pub fn gradient_id(&self) -> DataID {
		DataID{id: self.id * 2 + 1, desc: self.desc.clone()}//, shape: self.shape.clone()}
	}

	pub fn name(&self) -> &str {
		&self.desc.name
	}

	/// Returns the associated NodeShape
	pub fn shape(&self) -> &NodeShape {
		&self.desc.shape
	}

	/// Returns the tags of the associated node
	pub fn tags(&self) -> &IndexSet<NodeTag> {
		&self.desc.tags
	}
}

impl Hash for NodeID {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.id.hash(state);
	}
}

impl PartialEq for NodeID {
	fn eq(&self, other: &NodeID) -> bool {
		self.id == other.id
	}
}
impl Eq for NodeID {}

impl fmt::Display for NodeID {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{}", self.name())
	}
}


/// A unique identifier for a tensor (values or gradients) of a node.
#[derive(Clone, Debug)]
pub struct DataID {
	id: usize,
	desc: Arc<NodeDesc>,
	//shape: Arc<NodeShape>,
}

impl DataID {

	pub fn is_value(&self) -> bool {
		self.id % 2 == 0
	}

	pub fn is_gradient(&self) -> bool {
		self.id % 2 == 1
	}

	pub fn node_id(&self) -> NodeID {
		NodeID{id: self.id / 2, desc: self.desc.clone()}//, shape: self.shape.clone()}
	}

	/// Returns the associated node name with '_value' or '_gradient' appended as appropriate.
	pub fn name(&self) -> String {
		format!("{}_{}", self.desc.name, if self.is_value() {"value"} else {"gradient"})
	}

	/// Returns the associated NodeShape
	pub fn shape(&self) -> &NodeShape {
		&self.desc.shape
	}

	/// Returns the tags of the associated node
	pub fn tags(&self) -> &IndexSet<NodeTag> {
		&self.desc.tags
	}
}

impl Hash for DataID {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.id.hash(state);
	}
}

impl PartialEq for DataID {
	fn eq(&self, other: &DataID) -> bool {
		self.id == other.id
	}
}

impl Eq for DataID {}

impl fmt::Display for DataID {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{}", self.name())
	}
}


/// A unique identifier for a graph op.
#[derive(Clone, Debug)]
pub struct OpID {
	id: usize,
	desc: Arc<OpDescTrait>,
}

impl OpID {
	pub fn new<O: OpInstance>(id: usize, op: O, tags: IndexSet<OpTag>) -> Self {
		OpID{
			id: id,
			desc: Arc::new(OpDesc{
				instance: op,
				tags: tags,
			}),
		}
	}

	pub fn name(&self) -> &str {
		self.desc.instance().name()
	}

	pub fn tags(&self) -> &IndexSet<OpTag> {
		self.desc.tags()
	}

	pub fn instance(&self) -> &OpInstance {
		self.desc.instance()
	}
}

impl Hash for OpID {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.id.hash(state);
	}
}

impl PartialEq for OpID {
	fn eq(&self, other: &OpID) -> bool {
		self.id == other.id
	}
}

impl Eq for OpID {}

impl fmt::Display for OpID {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{}", self.name())
	}
}


/// A unique identifier for the (forward or backward) pass of an operator.
#[derive(Clone, Debug)]
pub struct PassID {
	id: usize,
	instance: Arc<Pass>,
}

impl PassID {
	pub fn new<P: Pass>(id: usize, pass: P) -> Self {
		PassID{id: id, instance: Arc::new(pass)}
	}
	// pub fn is_forward(&self) -> bool {
	// 	self.forward
	// }

	// pub fn is_backward(&self) -> bool {
	// 	!self.forward
	// }

	// pub fn op_id(&self) -> OpID {
	// 	self.op_id.clone()
	// 	//OpID{index: self.index / 2}
	// }

	pub fn name(&self) -> String {
		self.instance.name()
	}

	pub fn instance(&self) -> &Pass {
		self.instance.as_ref()
	}
}

impl Hash for PassID {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.id.hash(state);
	}
}

impl PartialEq for PassID {
	fn eq(&self, other: &PassID) -> bool {
		self.id == other.id
	}
}

impl Eq for PassID {}

impl fmt::Display for PassID {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{}", self.name())
	}
}

/// A type used to mark nodes as parameters, or for easy retrival from a graph.
///
/// When calling `new_node()` consider using the `tag![]` macro to supply tags.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum NodeTag{
	/// Marks a node as a `Parameter`.
	Parameter,
	/// A `Tag` which impl `From<NodeID>`. Will only match one node.
	Id(NodeID),
	/// A customisable `Tag` which impl `From<usize>`.
	Int(usize),
	/// A customisable `Tag` which impl `From<String>` and `From<&str>`.
	Str(String),
}

impl fmt::Display for NodeTag {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			&NodeTag::Parameter => write!(f, "NodeTag::Parameter"),
			&NodeTag::Id(ref id) => write!(f, "NodeTag::Id({})", id.name()),
			&NodeTag::Int(int) => write!(f, "NodeTag::Int({})", int),
			&NodeTag::Str(ref string) => write!(f, "NodeTag::Str({})", string),
		}
	}
}

impl<T: Borrow<NodeID>> From<T> for NodeTag{
	fn from(i: T) -> NodeTag {
		NodeTag::Id(i.borrow().clone())
	}
}

impl From<usize> for NodeTag{
	fn from(i: usize) -> NodeTag {
		NodeTag::Int(i)
	}
}

impl<'a> From<&'a str> for NodeTag{
	fn from(i: &str) -> NodeTag {
		NodeTag::Str(i.to_string())
	}
}

impl From<String> for NodeTag{
	fn from(i: String) -> NodeTag {
		NodeTag::Str(i)
	}
}

/// A type used to mark Ops for easy retrival from a graph.
///
/// When calling `new_op()` consider using the `tag![]` macro to supply tags.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum OpTag{
	/// A `Tag` which impl `From<OpID>`. Will only match one Op.
	Id(OpID),
	/// A customisable `Tag` which impl `From<usize>`.
	Int(usize),
	/// A customisable `Tag` which impl `From<String>` and `From<&str>`.
	Str(String),
}

impl fmt::Display for OpTag {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			&OpTag::Id(ref id) => write!(f, "OpTag::Id({})", id.name()),
			&OpTag::Int(int) => write!(f, "OpTag::Int({})", int),
			&OpTag::Str(ref string) => write!(f, "OpTag::Str({})", string),
		}
	}
}

impl<T: Borrow<OpID>> From<T> for OpTag{
	fn from(i: T) -> OpTag {
		OpTag::Id(i.borrow().clone())
	}
}

impl From<usize> for OpTag{
	fn from(i: usize) -> OpTag {
		OpTag::Int(i)
	}
}

impl<'a> From<&'a str> for OpTag{
	fn from(i: &str) -> OpTag {
		OpTag::Str(i.to_string())
	}
}

impl From<String> for OpTag{
	fn from(i: String) -> OpTag {
		OpTag::Str(i)
	}
}