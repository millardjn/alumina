
use std::collections::{HashMap, HashSet};
use ndarray::ArrayD;
use ndarray::prelude::*;
use smallvec::SmallVec;
use ops::Operation;
use std::cmp;
use std::iter::repeat;
use std::iter::FromIterator;
use std::sync::Arc;
use std::iter;
use ndarray;
use new::shape;
use new::shape::{NodeShape, NodeDim};
use std::cell::{Cell, UnsafeCell};
use std::mem;

error_chain!{
	errors {
		OperationAccessedDeallocatedNode
		OperationMutablyAccessedABorrowedNode
		OperationAccessedAMutablyBorrowedNode
	}

	links {
		Utils(shape::Error, shape::ErrorKind);
	}
}

// TODO change types to 'Node' and 'SharedNode'

pub enum NodeType {

	ParameterNode(ParameterType),

	VariableNode,

	XVariableNode,

	ViewNode(NodeID),

	ReifiedViewNode,
}

pub enum ParameterType {
	Locked(ArrayD<f32>),
	Init(Arc<Fn(&mut [f32])>),
}

/// A unique identifier for a node in the computational graph
#[derive(Clone)]
pub struct NodeID {
	index: usize,
	shape: Arc<NodeShape>,
}

impl NodeID {
	// pub fn index(&self) -> usize {
	// 	self.index
	// }

	pub fn shape(&self) -> &[NodeDim] {
		&self.shape.dimensions[..]
	}

	pub fn value_id(&self) -> DataID {
		DataID{index: self.index * 2, shape: self.shape.clone()}
	}

	pub fn gradient_id(&self) -> DataID {
		DataID{index: self.index * 2 + 1, shape: self.shape.clone()}
	}
}

/// A unique identifier for a tensor (values or gradients) of a node
pub struct DataID {
	index: usize,
	shape: Arc<NodeShape>,
}

impl DataID {
	// pub fn index(&self) -> usize {
	// 	self.index
	// }
	
	pub fn shape(&self) -> &[NodeDim] {
		&self.shape.dimensions[..]
	}

	pub fn node_id(&self) -> NodeID {
		NodeID{index: self.index / 2, shape: self.shape.clone()}
	}
}

/// A unique identifier for a graph operation
pub struct OpID {
	index: usize,
}

impl OpID {
	// pub fn index(&self) -> usize {
	// 	self.index
	// }

	pub fn value_id(&self) -> PassID {
		PassID{index: self.index * 2}
	}

	pub fn gradient_id(&self) -> PassID {
		PassID{index: self.index * 2 + 1}
	}
}

/// A unique identifier for the (forward or backward) pass of an operator
pub struct PassID {
	index: usize,
}

impl PassID {
	// fn index(&self) -> usize {
	// 	self.index
	// }

	pub fn op_id(&self) -> OpID {
		OpID{index: self.index / 2}
	}
}

#[derive(PartialEq, Eq, Hash)]
pub enum NodeTag{
	Parameter,
	Id(usize),
	Int(usize),
	Str(String),
}

impl From<NodeID> for NodeTag{
	fn from(i: NodeID) -> NodeTag {
		NodeTag::Id(i.index)
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

#[derive(PartialEq, Eq, Hash)]
pub enum OpTag{
	Id(usize),
	Int(usize),
	Str(String),
}

impl From<OpID> for OpTag{
	fn from(i: OpID) -> OpTag {
		OpTag::Int(i.index)
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

pub struct Builder {

	node_type: Vec<NodeType>,
	nodes: Vec<NodeID>,
	operations: Vec<Box<Operation>>,

	node_tags: HashMap<NodeTag, HashSet<usize>>, // tag, index
	operation_tags: HashMap<OpTag, HashSet<usize>> // tag, index
}

impl Builder {
	
	pub fn new() -> Builder{
		unimplemented!()
	}

	fn new_node<I: Into<NodeTag>, T: IntoIterator<Item=I>>(&mut self, shape: NodeShape, tag_iter: T, nodetype: NodeType) -> NodeID{
		let id = NodeID{index: self.nodes.len(), shape: Arc::new(shape)};
		self.nodes.push(id.clone());
		self.node_type.push(nodetype);
		
		let mut tag_iter = tag_iter.into_iter();
		while let Some(into_tag) = tag_iter.next(){
			self.tag(id.clone(), into_tag);
		}

		id
	}

	fn tag<T: Into<NodeTag>>(&mut self, node: NodeID, tag: T){
		let tag = tag.into();
		match tag {
			NodeTag::Id(_) => {},
			NodeTag::Int(_) | NodeTag::Str(_) | NodeTag::Parameter => {
				let set = self.node_tags.entry(tag).or_insert(HashSet::new());
				set.insert(node.index);
			}
		}	
	}

	pub fn set_initial_value(){}

	pub fn new_variable<S: Into<NodeShape>, I: Into<NodeTag>, T: IntoIterator<Item=I>>(&mut self, shape: S, tag_iter: T) -> NodeID {
		self.new_node(shape.into(), tag_iter, NodeType::VariableNode)
	}

	/// Creates a node which 
	// pub fn new_parameter<S: Into<NodeShape>, I: Into<NodeTag>, T: Into<Option<I>>>(&mut self, shape: S, init: Arc<Fn(&mut [f32])>, tag: T) -> NodeID {
	// 	let shape = shape.into();
	// 	assert!(shape.is_fixed(), "Parameter nodes must be of a fixed shape");
	// 	self.new_node(shape, tag, NodeType::ParameterNode(ParameterType::Init(init)))
	// }

	// pub fn new_locked_parameter<S: Into<NodeShape>, I: Into<NodeTag>, T: Into<Option<I>>>(&mut self, data: ArrayD<f32>, tag: T) -> NodeID {

	// 	self.new_node(data.shape().into(), tag, NodeType::ParameterNode(ParameterType::Locked(data)))
	// }

	/// Create a node which acts as a subview of another node.
	/// Contiguous views will be free from time and memory overhead.
	/// Non-contigues views will incurr a memory and time overhead during runtime.
	/// Returns NodeID of the new node.
	pub fn new_view<S: Into<NodeShape>, T: Into<NodeTag>>(&mut self, shape: S, tag: T) -> NodeID {
		unimplemented!()
	}

	// Returns the NodeID which was tagged with 'tag'. returns none if zero or more than one NodeIDs are associated with the tag.
	pub fn get_node<T: Into<NodeTag>>(&self, tag: T) -> Option<NodeID> {
		let tag = tag.into();
		match tag {
			NodeTag::Id(ind) => Some(self.nodes[ind].clone()),
			NodeTag::Int(_) | NodeTag::Str(_) | NodeTag::Parameter => {
				self.node_tags.get(&tag).and_then(|set| if set.len() == 1 {Some(self.nodes[*set.iter().next().unwrap()].clone())} else {None})
			}
		}
	}

	pub fn get_nodes<'a, T: Into<NodeTag>>(&'a self, tag: T) -> Box<Iterator<Item=NodeID> + 'a> {
		let tag = tag.into();
		match tag {
			NodeTag::Id(ind) => Box::new(iter::once(self.nodes[ind].clone())),
			NodeTag::Int(_) | NodeTag::Str(_) | NodeTag::Parameter  => {
				match self.node_tags.get(&tag){
					Some(set) => Box::new(set.iter().map(move |&ind| self.nodes[ind].clone())),
					None => Box::new(iter::empty::<NodeID>()),
				}
			}
		}
	}

	pub fn get_parameters<'a>(&'a self) -> Box<Iterator<Item=NodeID> + 'a> {
		self.get_nodes(NodeTag::Parameter)
	}

	pub fn get_op<T: Into<OpTag>>(&self, tag: T) -> Option<OpID> {
		let tag = tag.into();
		match tag {
			OpTag::Id(ind) => Some(OpID{index: ind}),
			OpTag::Int(_) | OpTag::Str(_) => {
				self.operation_tags.get(&tag).and_then(|set| if set.len() == 1 {Some(OpID{index: *set.iter().next().unwrap()})} else {None})
			}
		}
	}

	pub fn get_ops<'a, T: Into<OpTag>>(&'a self, tag: T) -> Box<Iterator<Item=OpID> + 'a> {
		let tag = tag.into();
		match tag {
			OpTag::Id(ind) => Box::new(iter::once(OpID{index: ind})),
			OpTag::Int(_) | OpTag::Str(_)  => {
				match self.operation_tags.get(&tag){
					Some(set) => Box::new(set.iter().map(|&ind| OpID{index: ind})),
					None => Box::new(iter::empty::<OpID>()),
				}
			}
		}
	}
}

// struct Graph{
// 	pass_inputs: Vec<Vec<NodeID>>,
// 	pass_outputs: Vec<Vec<NodeID>>,

// 	// Passes which input into the data
// 	data_inputs: Vec<Vec<PassID>>,

// 	// Passes which rely on the data
// 	data_outputs: Vec<Vec<PassID>>,

// 	pass order:

// 	available_inputs:
// 	required_outputs:
// }


enum DataState<T>{
	Unallocated,
	Allocated(T),
	Deallocated,
}

/// A structure which allows for runtime checked borrowing, similar to a RefCell for a Collection of Arrays,
/// but with some limitations.
/// Each element can only be borrowed either once mutably or many times immutably, however,
/// once borrowed as such it is stuck until DataBorrow is dropped, or by calling reset_all().
pub struct GraphData {
	n: usize,
	shapes: Vec<NodeShape>,
	data: Vec<DataState<ArrayD<f32>>>,
	borrow_flags: Vec<Cell<usize>>,
}

const UNUSED: usize = 0;
const WRITING: usize = !0;
impl GraphData {

	pub fn new(n: usize, shapes: Vec<NodeShape>) -> GraphData {
		let num_nodes = shapes.len();
		GraphData{
			n: n,
			shapes: shapes,
			data: (0..num_nodes * 2).map(|_i| DataState::Unallocated).collect(),
			borrow_flags: vec![Cell::new(UNUSED); num_nodes * 2],
		}
	}

	/// Deallocates the data specified by DataID
	pub fn deallocate(&mut self, id: DataID){
		mem::replace(&mut self.data[id.index], DataState::Deallocated);
	}

	/// This resets runtime checks, allowing new borrowing patterns
	/// By taking `self` this forces return of all prior borrows
	pub fn clear_borrow_flags(mut self) -> Self{
		for e in &mut self.borrow_flags{
			e.set(UNUSED);
		}
		self
	}

	/// Should never be called if a &mut borrow could possibly exist.
	unsafe fn get_or_init(&self, id: &DataID) -> Result<*mut ArrayD<f32>>{
		let ptr = &self.data[id.index] as *const _ as *mut _;
		match *ptr {
			DataState::Deallocated => bail!(ErrorKind::OperationAccessedDeallocatedNode),
			DataState::Unallocated => {
					*ptr = DataState::Allocated(ArrayD::zeros(self.shapes[id.node_id().index].to_data_shape(self.n)?));
					if let DataState::Allocated(ref mut data) = *ptr{
						Ok(data as *mut ArrayD<f32>)
					} else {
						unreachable!()
					}
				},
			DataState::Allocated(ref mut data) => Ok(data as *mut ArrayD<f32>),
		}
	}

	/// - `n` the number of input samples in the batch
	pub fn n(&self) -> usize{
		self.n
	}

	/// Immutably borrows data element associated with the given ID
	/// Will panic if data element is already borrowed mutably
	pub fn get<'a>(&'a self, id: DataID) -> Result<ArrayViewD<f32>> {
		if self.borrow_flags[id.index].get() != WRITING {
				let ptr = unsafe{self.get_or_init(&id)?};
				self.borrow_flags[id.index].set(self.borrow_flags[id.index].get() + 1);
				let array: &'a ArrayD<f32> = unsafe{&*ptr};
				Ok(array.view())
		} else {
			bail!(ErrorKind::OperationAccessedAMutablyBorrowedNode)
		}
	}

	/// Mutably borrows data element associated with the given ID
	/// Will panic if data element is already mutably or immutably borrowed 
	/// The borrow will stick until
	pub fn get_mut<'a>(&'a self, id: DataID) -> Result<ArrayViewMutD<f32>> {
		match self.borrow_flags[id.index].get() {
			UNUSED => {
				let ptr = unsafe{self.get_or_init(&id)?};
				self.borrow_flags[id.index].set(WRITING);
				let array: &'a mut ArrayD<f32> = unsafe{&mut *ptr};
				Ok(array.view_mut())
			},
			_ => bail!(ErrorKind::OperationMutablyAccessedABorrowedNode),
		}
	}
}