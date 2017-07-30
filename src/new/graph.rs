
use std::collections::{HashMap, HashSet};
use ndarray::ArrayD;
use ndarray::prelude::*;
use smallvec::SmallVec;
//use ops::Operation;
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
use std::collections::VecDeque;
use ordermap::OrderMap;
use new::ops::*;

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

	/// This node has an independent values for each input the input batch.
	/// The outermost dimension of the tensor is the batch size `n`
	IndependentNode,

	/// This node's shape does not depend on batch size.
	/// Examples are optimisable parameters, or an average across a mini batch
	SharedNode,
}

//pub enum ParameterType {
//	Locked(ArrayD<f32>),
//	Init(Arc<Fn(&mut [f32])>),
//}

/// A unique identifier for a node in the computational graph
#[derive(Clone, Debug)]
pub struct NodeID {
	index: usize,
	//shape: Arc<NodeShape>, // op builders should be able to just get it from the graphbuilders
}

impl NodeID {

	// pub fn shape(&self) -> &[NodeDim] {
	// 	&self.shape.dimensions[..]
	// }

	pub fn value_id(&self) -> DataID {
		DataID{index: self.index * 2}//, shape: self.shape.clone()}
	}

	pub fn gradient_id(&self) -> DataID {
		DataID{index: self.index * 2 + 1}//, shape: self.shape.clone()}
	}
}

/// A unique identifier for a tensor (values or gradients) of a node
#[derive(Clone, Debug)]
pub struct DataID {
	index: usize,
	//shape: Arc<NodeShape>,
}

impl DataID {
	pub fn is_value(&self) -> bool {
		self.index % 2 == 0
	}

	pub fn is_gradient(&self) -> bool {
		self.index % 2 == 1
	}

	// pub fn shape(&self) -> &[NodeDim] {
	// 	&self.shape.dimensions[..]
	// }

	pub fn node_id(&self) -> NodeID {
		NodeID{index: self.index / 2}//, shape: self.shape.clone()}
	}
}

/// A unique identifier for a graph operation
#[derive(Clone, Debug)]
pub struct OpID {
	index: usize,
}

impl OpID {
	pub fn forward_id(&self) -> PassID {
		PassID{index: self.index * 2}
	}

	pub fn backward_id(&self) -> PassID {
		PassID{index: self.index * 2 + 1}
	}
}

/// A unique identifier for the (forward or backward) pass of an operator
#[derive(Clone, Debug)]
pub struct PassID {
	index: usize,
}

impl PassID {
	pub fn is_forward(&self) -> bool {
		self.index % 2 == 0
	}

	pub fn is_backward(&self) -> bool {
		self.index % 2 == 1
	}

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

/// Used to construct the definition of the computational hypergraph.
/// This cannot be executed, an executable `Graph` can be built using 
pub struct Builder {

	node_type: Vec<NodeType>,
	nodes: Vec<NodeID>,
	operations: Vec<Box<Operation>>,

	node_tags: OrderMap<NodeTag, OrderMap<usize, ()>>, // tag, index
	operation_tags: OrderMap<OpTag, OrderMap<usize, ()>>, // tag, index
	initial_value: Vec<Option<ArrayD<f32>>>,
}

impl Builder {
	
	pub fn new() -> Builder {
		unimplemented!()
	}

	///
	/// * inputs - this must be an in order slice of the nodes which will be supplied by the data stream used when evaluating the graph.
	pub fn build(&self, inputs: &[DataID], requested_outputs: &[DataID]) -> Graph {

		let mut is_input = vec![false; self.nodes.len()*2];
		for data_id in inputs {
			is_input[data_id.index] = true;
		}

		let (pass_inputs, pass_outputs, data_inputs, data_outputs) = self.collect_dependencies();
		
		// Find the minimum set of operations and nodes required to calculate the `required_outputs`
		let mut required_data: Vec<bool> = vec![false; self.nodes.len()*2];
		let mut required_passes: Vec<bool> = vec![false; self.operations.len()*2];
		{
			let mut pass_queue: VecDeque<PassID> = VecDeque::new();
			let mut data_queue: VecDeque<DataID> = VecDeque::new(); 


			// start with the data requested by the user
			for data_id in requested_outputs {
				required_data[data_id.index] == true;
				data_queue.push_back(data_id.clone());
			}

			// For each item queued also queue its inputs if they havent already been marked as required.
			while !(pass_queue.is_empty() && data_queue.is_empty()) {
				if let Some(data_id) = data_queue.pop_front() {

					if !is_input[data_id.index] {
						for pass_id in &data_inputs[data_id.index] {
							if required_passes[pass_id.index] == false {
								required_passes[pass_id.index] = true;
								pass_queue.push_back(pass_id.clone());
							}
						}
					}

				}
				if let Some(pass_id) = pass_queue.pop_front() {
					for data_id in &pass_inputs[pass_id.index] {
						if required_data[data_id.index] == false {
							required_data[data_id.index] == true;
							data_queue.push_back(data_id.clone());
						}
					}
				}
			}
		}

		// Todo find operation order, and 

		let graph = Graph{
			pass_inputs: pass_inputs,
			pass_outputs: pass_outputs,

			data_inputs: data_inputs,
			data_outputs: data_outputs,

			available_inputs: inputs.to_vec(),
			requested_outputs: requested_outputs.to_vec(),

			pass_order: unimplemented!(),
		};

		graph
	}

	fn collect_dependencies(&self) -> (Vec<Vec<DataID>>, Vec<Vec<DataID>>, Vec<Vec<PassID>>, Vec<Vec<PassID>>) {
		let mut pass_inputs: Vec<Vec<DataID>> = (0..self.operations.len()*2).map(|_| vec![]).collect();
		let mut pass_outputs: Vec<Vec<DataID>> = (0..self.operations.len()*2).map(|_| vec![]).collect();

		let mut data_inputs: Vec<Vec<PassID>> = (0..self.nodes.len()*2).map(|_| vec![]).collect();
		let mut data_outputs: Vec<Vec<PassID>> = (0..self.nodes.len()*2).map(|_| vec![]).collect();

		for op_id in (0..self.operations.len()).map(|i| OpID{index:i}) {
			let operation = &*self.operations[op_id.index];

			let forward_id = op_id.forward_id();
			let (forward_inputs, forward_outputs) = operation.forward_dependencies();
			pass_inputs[forward_id.index] = forward_inputs;
			pass_outputs[forward_id.index] = forward_outputs;

			let backward_id = op_id.backward_id();
			let (backward_inputs, backward_outputs) = operation.backward_dependencies();
			pass_inputs[backward_id.index] = backward_inputs;
			pass_outputs[backward_id.index] = backward_outputs;
		}

		for pass_id in (0..self.operations.len()*2).map(|i| PassID{index:i}) {
			for data_id in &pass_inputs[pass_id.index] {
				data_outputs[data_id.index].push(pass_id.clone());
			}
			for data_id in &pass_outputs[pass_id.index] {
				data_inputs[data_id.index].push(pass_id.clone())
			}
		}

		(pass_inputs, pass_outputs, data_inputs, data_outputs)
	}

	/// Node values are initialised to be zero filled by default
	/// If a NdArray value is supplied to this method that can be broadcast to this node, it will be used to set the initial value of the node
	/// This can be used to supply fixed inputs, which replace parameters, to operations when further operations is to be avoided.
	pub fn set_initial_value(&mut self, id: NodeID, value: ArrayD<f32>){
		self.initial_value[id.index] = Some(value);
	}

	pub fn clear_initial_value(&mut self, id: NodeID){
		self.initial_value[id.index] = None;
	}

	fn add_operation<B: OperationBuilder>(&mut self, builder: B) -> OpID {
		unimplemented!();
	}

	fn add_simple_operation<B: SimpleOperationBuilder>(&mut self, mut builder: B) -> (OpID, NodeID) {
		let node_id = self.new_node(builder.required_output_shape(), iter::empty::<NodeTag>(), NodeType::IndependentNode);
		builder.set_output(&node_id);
		let op_id = self.add_operation(builder);
		(op_id, node_id)
	}

	fn new_node<I: Into<NodeTag>, T: IntoIterator<Item=I>>(&mut self, shape: NodeShape, tag_iter: T, nodetype: NodeType) -> NodeID{
		let id = NodeID{index: self.nodes.len()};//, shape: Arc::new(shape)};
		self.nodes.push(id.clone());
		self.node_type.push(nodetype);
		
		let mut tag_iter = tag_iter.into_iter();
		while let Some(into_tag) = tag_iter.next(){
			self.tag_node(id.clone(), into_tag);
		}

		id
	}

	fn tag_node<T: Into<NodeTag>>(&mut self, node: NodeID, tag: T){
		let tag = tag.into();
		match tag {
			NodeTag::Id(_) => {},
			NodeTag::Int(_) | NodeTag::Str(_) | NodeTag::Parameter => {
				let set = self.node_tags.entry(tag).or_insert(OrderMap::new());
				set.insert(node.index, ());
			}
		}	
	}

	pub fn new_shared_node<S: Into<NodeShape>, I: Into<NodeTag>, T: IntoIterator<Item=I>>(&mut self, shape: S, tag_iter: T) -> NodeID {
		self.new_node(shape.into(), tag_iter, NodeType::SharedNode)
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
				self.node_tags.get(&tag).and_then(|set| if set.len() == 1 {Some(self.nodes[*set.keys().next().unwrap()].clone())} else {None})
			}
		}
	}

	pub fn get_nodes<'a, T: Into<NodeTag>>(&'a self, tag: T) -> Box<Iterator<Item=NodeID> + 'a> {
		let tag = tag.into();
		match tag {
			NodeTag::Id(ind) => Box::new(iter::once(self.nodes[ind].clone())),
			NodeTag::Int(_) | NodeTag::Str(_) | NodeTag::Parameter  => {
				match self.node_tags.get(&tag){
					Some(set) => Box::new(set.keys().map(move |&ind| self.nodes[ind].clone())),
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
				self.operation_tags.get(&tag).and_then(|set| if set.len() == 1 {Some(OpID{index: *set.keys().next().unwrap()})} else {None})
			}
		}
	}

	pub fn get_ops<'a, T: Into<OpTag>>(&'a self, tag: T) -> Box<Iterator<Item=OpID> + 'a> {
		let tag = tag.into();
		match tag {
			OpTag::Id(ind) => Box::new(iter::once(OpID{index: ind})),
			OpTag::Int(_) | OpTag::Str(_)  => {
				match self.operation_tags.get(&tag){
					Some(set) => Box::new(set.keys().map(|&ind| OpID{index: ind})),
					None => Box::new(iter::empty::<OpID>()),
				}
			}
		}
	}
}

pub struct Graph{
	pass_inputs: Vec<Vec<DataID>>,
	pass_outputs: Vec<Vec<DataID>>,

	// Passes which input into the data
	data_inputs: Vec<Vec<PassID>>,

	// Passes which rely on the data
	data_outputs: Vec<Vec<PassID>>,

	available_inputs: Vec<DataID>,
	requested_outputs: Vec<DataID>,

	pass_order: Vec<PassID>,
}


pub struct GraphShapes{
	shapes: Vec<NodeShape>,
}

impl GraphShapes {
	pub fn merge_with(&mut self, id: NodeID, shape: NodeShape) -> Result<()>{
		self.shapes[id.index] = self.shapes[id.index].merge(&shape)?;
		Ok(())
	}
}

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
	initial_value: Vec<Option<ArrayD<f32>>>,
}

const UNUSED: usize = 0;
const WRITING: usize = !0;
impl GraphData {

	pub fn new(n: usize, shapes: Vec<NodeShape>, initial_value: Vec<Option<ArrayD<f32>>>,) -> GraphData {
		let num_nodes = shapes.len();
		GraphData{
			n: n,
			shapes: shapes,
			data: (0..num_nodes * 2).map(|_i| DataState::Unallocated).collect(),
			borrow_flags: vec![Cell::new(UNUSED); num_nodes * 2],
			initial_value: initial_value,
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
					let mut data = ArrayD::zeros(self.shapes[id.node_id().index].to_data_shape(self.n)?);
					if id.is_value() {
						if let Some(ref init_value) = self.initial_value[id.node_id().index]{
							if let Some(broadcasted_init) = init_value.broadcast(data.shape()){
								data = data + broadcasted_init;
							} else {
								bail!(format!("Broadcast of initial value failed for node {:?} as shape {:?} could not be broadcast to shape: {:?}", id.node_id(), init_value.shape(), data.shape()));
							}
						}
					}
					*ptr = DataState::Allocated(data);

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