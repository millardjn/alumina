#![allow(unused_imports)]
use std::collections::{HashMap, HashSet};
use ndarray::ArrayD;
use ndarray::ArrayBase;
use ndarray::prelude::*;
use smallvec::SmallVec;
use ndarray::Ix;
use std::cmp;
use std::iter;
use std::iter::repeat;
use std::iter::FromIterator;
use std::sync::Arc;
use ndarray;
use new::ops;
use new::shape;
use new::shape::{NodeShape, NodeDim};
use std::cell::{Cell, UnsafeCell};
use std::mem;
use std::collections::VecDeque;
use ordermap::OrderMap;
use new::ops::*;
use std::borrow::Borrow;
use std::rc::Rc;
use std::any::Any;
use std::sync::Mutex;
use std::fmt;
use std::ops::DerefMut;

error_chain!{
	errors {
		/// This errorkind indicates that the data requested was never allocated as it was not a required component of the subgraph.
		StorageDataMarkedNotRequired{} //TODO give name of node
		/// This errorkind indicates that the data requested has already been deallocated.
		/// Ensure that it was included in the output used to define the subgraph being executed.
		StorageDataDeallocated{}
		/// This errorkind indicates that data requested cannot be mutably borrowed as it has already been immutably borrowed.
		/// Borrows are not reset until after the pass has completed.
		StorageDataAlreadyBorrowed{}
		/// This errorkind indicates that data requested cannot be borrowed as it has already been mutably borrowed.
		/// Borrows are not reset until after the pass has completed.
		StorageDataAlreadyMutablyBorrowed{}
		/// This errorkind indicates that two identical node names have been supplied.
		/// Node names must be unique.
		NodeNameConflict(name: String){
			display("There is a conflict between node names, ensure they are unique. Duplicate name: {}", name)
		}
		/// This errorkind indicates that the node name supplied is identical to a node tag.
		/// Node names must be unique.
		NodeTagNameConflict(name: String){
			display("There is a conflict between tags and node names, ensure names are unique. Duplicate name: {}", name)
		}
		/// This errorkind indicates that two identical op names have been supplied.
		/// Op names must be unique.
		OpNameConflict(name: String){
			display("There is a conflict between op names, ensure they are unique. Duplicate name: {}", name)
		}
		/// This errorkind indicates that the op name supplied is identical to an op tag.
		/// Op names must be unique.
		OpTagNameConflict(name: String){
			display("There is a conflict between tags and op names, ensure names are unique. Duplicate name: {}", name)
		}
		/// This errorkind indicates that a node was marked as a `Parameter` did not have a Known size.
		/// Parameter node shapes cannot contain any `Unknown` or `Interval` dimensions.
		ParameterNodesMustHaveKnownSize(name: String, shape: NodeShape){
			display("Parameter node shapes cannot contain any `Unknown` or `Interval` dimensions: {} shape: {:?}", name, shape)
		}
		/// Could not find any nodes matching the tag supplied
		ZeroNodesMatchTag(tag: NodeTag){
			display("Could not find any nodes matching the tag supplied: {:?}", tag)
		} // TODO include tag
		/// Found more than one nodes matching the tag supplied, use the method for multiple NodeIDs.
		MultipleNodesMatchTag(tag: NodeTag){
			display("Found more than one nodes matching the tag supplied, use the method for multiple NodeIDs. {:?}", tag)
		}
		/// Could not find any ops matching the tag supplied
		ZeroOpsMatchTag(tag: OpTag){
			display("Could not find any ops matching the tag supplied: {:?}", tag)
		}
		/// Found more than one ops matching the tag supplied, use the method for multiple OpIDs.
		MultipleOpsMatchTag(tag: OpTag){
			display("Found more than one ops matching the tag supplied, use the method for multiple OpIDs. {:?}", tag)
		}
		/// A topological sort could not be completed, due to circular dependencies.
		GraphContainsCircularOps(deferred_ops: Vec<(OpID, Vec<NodeID>)>){
			display("The following ops were required, but could not be included in the execution order: {:?}", deferred_ops) //TODO change to print op and node names
		}
		/// A topological sort could not be completed, due to circular dependencies.
		GraphContainsCircularPasses(deferred_passes: Vec<(PassID, Vec<DataID>)>){
			display("The following passes were required, but could not be included in the execution order: {:?}", deferred_passes) //TODO change to print op and node names
		}
		/// The outputs of the subgraph are not computable from the inputs
		SubgraphInsufficientInputsForOutputs(unavailable_data: Vec<String>){
			display("The following data were required, but could not be computed from the inputs: {:?}", unavailable_data)
		}
		/// Some `NodeShapes` could not be inferred
		SubgraphInsufficientInputsForShapeInference(unavailable_nodes: Vec<String>){
			display("The following node shapes were required, but could not be inferred from the inputs: {:?}", &unavailable_nodes)
		}
		InputSizeError{}
		StaticInputBroadcastFailure(id: NodeID, s1: Vec<Ix>, s2: Vec<Ix>){
			display("Broadcast of initial value failed for node {:?} as shape {:?} could not be broadcast to shape: {:?}", id, s1, s2)
		}

		/// Occurs when a pass immutably accesses data at a data_ID not listed as an input or output dependency
		StorageImmutableBorrowError(pass_name: String, data_name: String){
			display("Pass '{}' attemped to access '{}' but did not have it listed as an input or output dependency", pass_name, data_name)
		}

		/// Occurs when a pass mutably accesses data at a data_ID not listed as an output dependency
		StorageMutableBorrowError(pass_name: String, data_name: String){
			display("Pass '{}' attemped to mutably access '{}' but did not have it listed as an output dependency", pass_name, data_name)
		}

		// Op Errors
		ShapePropagationError(op_instance_name: String, message: String){
			display("OpInstance: '{}' returned error message: {}", op_instance_name, message)
		}

		/// Generic error to be returned from the `run()` method of a `Pass`
		PassError(pass_name: String, message: String){
			display("Pass: '{}' returned error message: {}", pass_name,	message)
		}
	}

	links {
		ShapeError(shape::Error, shape::ErrorKind);
	}
}

/// A unique identifier for a node in the computational graph
#[derive(PartialEq, Eq, Ord, PartialOrd, Hash, Clone, Debug)]
pub struct NodeID {
	index: usize,
	//shape: Arc<NodeShape>, // op builders should be able to just get it from the graphbuilders
}

impl NodeID {

	pub fn value_id(&self) -> DataID {
		DataID{index: self.index * 2}//, shape: self.shape.clone()}
	}

	pub fn gradient_id(&self) -> DataID {
		DataID{index: self.index * 2 + 1}//, shape: self.shape.clone()}
	}
}

/// A unique identifier for a tensor (values or gradients) of a node.
#[derive(PartialEq, Eq, Ord, PartialOrd, Hash, Clone, Debug)]
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

	pub fn node_id(&self) -> NodeID {
		NodeID{index: self.index / 2}//, shape: self.shape.clone()}
	}
}

/// A unique identifier for a graph op.
#[derive(PartialEq, Eq, Ord, PartialOrd, Hash, Clone, Debug)]
pub struct OpID {
	index: usize,
}

impl OpID {
	// pub fn forward_id(&self) -> PassID {
	// 	PassID{index: self.index * 2}
	// }

	// pub fn backward_id(&self) -> PassID {
	// 	PassID{index: self.index * 2 + 1}
	// }
}

/// A unique identifier for the (forward or backward) pass of an operator.
#[derive(PartialEq, Eq, Ord, PartialOrd, Hash, Clone, Debug)]
pub struct PassID {
	index: usize,
}

impl PassID {
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
}

/// A type used to mark nodes as parameters, or for easy retrival from a graph.
///
/// When calling `new_node()` consider using the `tag![]` macro to supply tags.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum NodeTag{
	/// Marks a node as a `Parameter`.
	Parameter,
	/// A `Tag` which impl `From<NodeID>`. Will only match one node.
	Id(usize),
	/// A customisable `Tag` which impl `From<usize>`.
	Int(usize),
	/// A customisable `Tag` which impl `From<String>` and `From<&str>`.
	Str(String),
}

impl<T: Borrow<NodeID>> From<T> for NodeTag{
	fn from(i: T) -> NodeTag {
		NodeTag::Id(i.borrow().index)
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
		NodeTag::Str(i.to_string())
	}
}

/// A type used to mark Ops for easy retrival from a graph.
///
/// When calling `new_op()` consider using the `tag![]` macro to supply tags.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum OpTag{
	/// A `Tag` which impl `From<OpID>`. Will only match one Op.
	Id(usize),
	/// A customisable `Tag` which impl `From<usize>`.
	Int(usize),
	/// A customisable `Tag` which impl `From<String>` and `From<&str>`.
	Str(String),
}

impl<T: Borrow<OpID>> From<T> for OpTag{
	fn from(i: T) -> OpTag {
		OpTag::Id(i.borrow().index)
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

/// Wrapper for initialiser closures that implements `Clone` and `Debug`
#[derive(Clone)]
struct Initialiser {
	name: String,
	func: Arc<Mutex<FnMut(&mut ArrayD<f32>)>>
}

impl Initialiser {
	pub fn new<F: 'static + FnMut(&mut ArrayD<f32>)>(name: String, func: F) -> Self {
		Initialiser {
			name: name,
			func: Arc::new(Mutex::new(func)),
		}
	}

	pub fn call(&self, arr: &mut ArrayD<f32>) {
		let mut guard = self.func.lock().expect(&format!("Could not unwrap initialiser: {:?}", self));
		(guard.deref_mut())(arr);
	}
}

impl fmt::Debug for Initialiser {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "Initialiser {{ name: {}, .. }}", self.name)
	}
}

/// Used to construct the definition of the computational hypergraph.
/// This cannot be executed, an executable `Graph` can be built using
#[derive(Clone, Debug)]
pub struct GraphDef {

	node_ids: Vec<NodeID>,
	node_shapes: Vec<NodeShape>,
	node_names: OrderMap<String, NodeID>,
	node_tags: OrderMap<NodeTag, OrderMap<NodeID, ()>>,

	pass_ids: Vec<PassID>,
	passes: Vec<Box<Pass>>,

	op_ids: Vec<OpID>,
	ops: Vec<Rc<OpInstance>>,
	op_names: OrderMap<String, OpID>,
	op_tags: OrderMap<OpTag, OrderMap<OpID, ()>>,

	static_inputs: OrderMap<DataID, ArrayD<f32>>,

	initialisers: OrderMap<NodeID, Initialiser>,
}

impl GraphDef {
	
	pub fn new() -> GraphDef {
		GraphDef {
			node_ids: vec![],
			node_shapes: vec![],
			node_names: OrderMap::new(),
			node_tags: OrderMap::new(),

			pass_ids: vec![],
			passes: vec![],

			op_ids: vec![],
			ops: vec![],
			op_names: OrderMap::new(),
			op_tags: OrderMap::new(),

			static_inputs: OrderMap::new(),

			initialisers: OrderMap::new(),
		}
	}

	/// Extracts a subgraph which can be executed, for training or inference purposes.
	///
	/// * inputs - this must be an in order slice of the nodes which will be supplied by the data stream used when evaluating the graph.
	/// * outputs - the order of the output DataIDs does not currently matter
	pub fn subgraph(&self, inputs: &[DataID], outputs: &[DataID]) -> Result<Subgraph> {
		Subgraph::new(&self, inputs, outputs)
	}

	/// The default subgraph is typicaly suitable for training.
	///
	/// All nodes with no input ops are taken to be subgraph inputs,
	/// and all parameters values and parameter gradients are taken to be outputs.
	/// The ordering of the inputs follows the order of node creation,
	/// with the additional constraint that non-parameter nodes are strictly before parameter nodes.
	///
	/// See `subgraph()`.
	pub fn default_subgraph(&self) -> Result<Subgraph> {
		let dependencies = Dependencies::new(self);
		let input_ids: Vec<NodeID> = self.nodes().iter().filter(|node_id| dependencies.data_inputs(&node_id.value_id()).len() == 0 && !self.is_node_tagged(*node_id, NodeTag::Parameter)).cloned().collect();
		let parameter_ids: Vec<NodeID> = self.nodes().iter().filter(|node_id| dependencies.data_inputs(&node_id.value_id()).len() == 0 && self.is_node_tagged(*node_id, NodeTag::Parameter)).cloned().collect();
		
		self.subgraph(
			&input_ids.iter().chain(&parameter_ids).map(|node_id| node_id.value_id()).collect::<Vec<_>>(),
			&parameter_ids.iter().map(|node_id| node_id.value_id()).chain(parameter_ids.iter().map(|node_id| node_id.gradient_id())).collect::<Vec<_>>()
		)
	}

	/// Node values are initialised to be zero filled by default.
	/// If a NdArray value is supplied to this method that can be broadcast to this node, it will be used to set the initial value of the node
	/// This can be used to supply fixed inputs to Ops in place of parameters
	pub fn set_static_input(&mut self, id: DataID, value: ArrayD<f32>){
		self.static_inputs.insert(id, value);
	}

	pub fn clear_static_input(&mut self, id: DataID){
		self.static_inputs.remove(&id);
	}

	pub fn set_initialiser<F: 'static + FnMut(&mut ArrayD<f32>)>(&mut self, node_id: &NodeID, name: String, func: F) {
		self.initialisers.insert(node_id.clone(), Initialiser::new(name, func));
	}

	pub fn clear_initialiser(&mut self, node_id: &NodeID) {
		self.initialisers.remove(node_id);
	}

	pub fn initialise_nodes(&self, nodes: &[NodeID]) -> Result<Vec<ArrayD<f32>>>{
		let mut vec = Vec::with_capacity(nodes.len());
		for node in nodes {
			let shape = self.node_shapes[node.index].to_data_shape()?;
			let mut arr = ArrayD::zeros(shape);
			if let Some(init_func) = self.initialisers.get(node) {
				init_func.call(&mut arr);
			}
			vec.push(arr);
		}
		Ok(vec)
	}

	/// Create a new node in the graph
	pub fn new_node<I: Into<String>>(&mut self, shape: NodeShape, name: I, tags: Vec<NodeTag>) -> Result<NodeID>{
		
		let name = name.into();

		// ensure names are unique w.r.t other names and tags
		ensure!(!self.node_names.contains_key(&name), ErrorKind::NodeNameConflict(name));
		ensure!(!self.node_tags.contains_key(&name.as_str().into()), ErrorKind::NodeTagNameConflict(name));
		for tag in &tags{
			// ensure that tags don't overlap with existing names
			match tag {
				&NodeTag::Str(ref tag_str) => {
					ensure!(&name != tag_str, ErrorKind::NodeTagNameConflict(tag_str.to_string()));
					ensure!(!self.node_names.contains_key(tag_str), ErrorKind::NodeTagNameConflict(tag_str.to_string()));
				},
				_ => {},
			}
		}

		for tag in &tags{
			if matches!(tag, &NodeTag::Parameter){
				ensure!(shape.is_known(), ErrorKind::ParameterNodesMustHaveKnownSize(name, shape));
			}
		}

		// all good, so add node
		let node_id = NodeID{index: self.node_ids.len()};
		self.node_names.insert(name, node_id.clone());
		self.node_shapes.push(shape);
		self.node_ids.push(node_id.clone());
		
		for tag in tags{
			match tag {
				NodeTag::Id(_) => {},
				NodeTag::Int(_) | NodeTag::Str(_) | NodeTag::Parameter => {
					self.node_tags.entry(tag).or_insert(OrderMap::new()).insert(node_id.clone(), ());
				},
			}
		}


		Ok(node_id)
	}



	pub fn new_op<O: Op>(&mut self, op: O, tags: Vec<OpTag>) -> Result<OpID> {
		
		// A noop is pushed in place of the real op to reserve a place before
		let op_id = OpID{index: self.ops.len()};
		let noop = Rc::new(NoOp::new().build(self, &op_id)?);
		self.ops.push(noop);


		let op = op.build(self, &op_id)?;
		
		let name = op.instance_name().to_string();

		// ensure names are unique w.r.t other names and tags
		ensure!(!self.op_names.contains_key(&name), ErrorKind::OpNameConflict(name));
		ensure!(!self.op_tags.contains_key(&name.as_str().into()), ErrorKind::OpTagNameConflict(name));
		for tag in &tags{
			// ensure that tags don't overlap with existing names
			match tag {
				&OpTag::Str(ref tag_str) => {
					ensure!(&name != tag_str, ErrorKind::OpTagNameConflict(tag_str.to_string()));
					ensure!(!self.op_names.contains_key(tag_str), ErrorKind::OpTagNameConflict(tag_str.to_string()));
				},
				_ => {},
			}
		}

		// all good, so add op
		self.op_names.insert(name, op_id.clone());
		self.ops[op_id.index] = Rc::new(op);
		self.op_ids.push(op_id.clone());

		for tag in tags{
			match tag {
				OpTag::Id(_) => {},
				OpTag::Int(_) => {
					self.op_tags.entry(tag).or_insert(OrderMap::new()).insert(op_id.clone(), ());
				},
				OpTag::Str(_) => {
					self.op_tags.entry(tag).or_insert(OrderMap::new()).insert(op_id.clone(), ());
				},
			}
		}

		Ok(op_id)
	}

	pub fn add_pass<P: Pass>(&mut self, pass: P) -> PassID {
		let pass_id = PassID{index: self.passes.len()};
		self.passes.push(Box::new(pass));
		self.pass_ids.push(pass_id.clone());
		pass_id
	}

	// fn new_layer(SimpleOpBuilder and output NodeShape){}
	// fn new_simple_op<B: SimpleOpBuilder>(&mut self, mut builder: B, shape: NodeShape) -> (OpID, NodeID) {
	// 	let name = unimplemented!();
	// 	let node_id = self.new_node(builder.required_output_shape(), name, &[]);
	// 	builder.set_output(&node_id);
	// 	let op_id = self.new_op(builder);
	// 	(op_id, node_id)
	// }

	

	/// Create a node which acts as a subview of another node.
	/// Contiguous views will be free from time and memory overhead, recording just a view.
	/// Non-contigues views will incurr a memory and time overhead during runtime.
	/// Returns NodeID of the new node.
	pub fn new_read_view<I: Into<String>>(&mut self, _name: I, _shape: NodeShape, _tags: Vec<NodeTag>) -> Result<NodeID>{
		unimplemented!()
	}

	pub fn new_write_view<I: Into<String>>(&mut self, _name: I, _shape: NodeShape, _tags: Vec<NodeTag>) -> Result<NodeID>{
		unimplemented!()
	}

	pub fn nodes(&self) -> &[NodeID] {
		&self.node_ids
	}

	pub fn ops(&self) -> &[OpID] {
		&self.op_ids
	}

	pub fn node_name(&self, node_id: &NodeID) -> &str{
		for (k, v) in self.node_names.iter(){
			if v == node_id {
				return k;
			}
		}
		panic!("Is this a NodeID from a different GraphDef?")
	}

	/// Returns the associated node name with '_value' or '_gradient' appended as appropriate.
	pub fn data_name(&self, data_id: &DataID) -> String {
		format!("{}_{}", self.node_name(&data_id.node_id()), if data_id.is_value() {"value"} else {"gradient"})
	}

	pub fn op_name(&self, op_id: &OpID) -> &str{
		for (k, v) in self.op_names.iter(){
			if v == op_id {
				return k;
			}
		}
		panic!("Is this a OpID from a different GraphDef?")
	}

	pub fn pass_name(&self, pass_id: &PassID) -> String {
		self.passes[pass_id.index].instance_name(self)
	}

	pub fn node_shape<T: Into<NodeTag>>(&self, tag: T) -> Result<&NodeShape> {
		self.node_id(tag).map(|id| &self.node_shapes[id.index])
	}

	pub fn parameter_ids<'a>(&'a self) -> OrderMap<NodeID, ()> {
		self.node_ids(NodeTag::Parameter)
	}

	pub fn op<T: Into<OpTag>>(&self, tag: T) -> Result<&OpInstance> {
		self.op_id(tag).map(|id| &*self.ops[id.index])
	}

	// TODO check names
	pub fn is_node_tagged<T: Into<NodeTag>>(&self, node_id: &NodeID, tag: T) -> bool {
		let tag = tag.into();
		match tag {
			NodeTag::Id(ind) => {ind == node_id.index},
			NodeTag::Int(_) | NodeTag::Str(_) | NodeTag::Parameter => {
				match self.node_tags.get(&tag){
					Some(set) => set.contains_key(node_id),
					None => false,
				}
			}
		}
	}

	// TODO check names
	pub fn is_op_tagged<T: Into<OpTag>>(&self, op_id: &OpID, tag: T) -> bool {
		let tag = tag.into();
		match tag {
			OpTag::Id(ind) => {ind == op_id.index},
			OpTag::Int(_) | OpTag::Str(_) => {
				match self.op_tags.get(&tag){
					Some(set) => set.contains_key(op_id),
					None => false,
				}
			}
		}
	}

	// Returns the NodeID which was tagged with 'tag'. returns none if zero or more than one NodeIDs are associated with the tag.
	pub fn node_id<T: Into<NodeTag>>(&self, tag: T) -> Result<NodeID> {
		let tag = tag.into();
		match tag {
			NodeTag::Id(ind) => Ok(self.node_ids[ind].clone()),
			NodeTag::Str(ref string) => {
				// Check names first, then other string tags
				if let Some(node_id) = self.node_names.get(string) {
					Ok(node_id.clone())
				} else {
					let set = match self.node_tags.get(&tag){
						Some(set) => set,
						None => bail!(ErrorKind::ZeroNodesMatchTag(tag.clone())),
					};
					match set.len() {
						0 => bail!(ErrorKind::ZeroNodesMatchTag(tag.clone())),
						1 => Ok(set.keys().next().unwrap().clone()),
						_ => bail!(ErrorKind::MultipleNodesMatchTag(tag.clone())),
					}
				}
			},
			NodeTag::Int(_) | NodeTag::Parameter => {
				let set = match self.node_tags.get(&tag){
					Some(set) => set,
					None => bail!(ErrorKind::ZeroNodesMatchTag(tag)),
				};
				match set.len() {
					0 => bail!(ErrorKind::ZeroNodesMatchTag(tag)),
					1 => Ok(set.keys().next().unwrap().clone()),
					_ => bail!(ErrorKind::MultipleNodesMatchTag(tag)),
				}
			}
		}
	}

	pub fn node_ids<'a, T: Into<NodeTag>>(&'a self, tag: T) -> OrderMap<NodeID, ()> {
		let tag = tag.into();
		match tag {
			NodeTag::Id(ind) => iter::once((self.node_ids[ind].clone(), ())).collect(),
			NodeTag::Str(ref string) => {
				// Check names first, then other string tags
				if let Some(node_id) = self.node_names.get(string) {
					iter::once((node_id.clone(), ())).collect()
				} else {
					match self.node_tags.get(&tag){
						Some(set) => set.clone(),
						None => OrderMap::new(),
					}
				}
			},
			NodeTag::Int(_) | NodeTag::Parameter  => {
				match self.node_tags.get(&tag){
					Some(set) => set.clone(),
					None => OrderMap::new(),
				}
			}
		}
	}

	/// Returns a single OpID matching a op tag
	/// Returns an error if multiple or zero ops are associated with the tag
	pub fn op_id<T: Into<OpTag>>(&self, tag: T) -> Result<OpID> {
		let tag = tag.into();
		match tag {
			OpTag::Id(ind) => Ok(OpID{index: ind}),
			OpTag::Str(ref string) => {
				//self.op_tags.get(&tag).and_then(|set| if set.len() == 1 {Some(set.keys().next().unwrap().clone())} else {None})
				if let Some(op_id) = self.op_names.get(string) {
					Ok(op_id.clone())
				} else {
					let set = match self.op_tags.get(&tag){
						Some(set) => set,
						None => bail!(ErrorKind::ZeroOpsMatchTag(tag.clone())),
					};
					match set.len() {
						0 => bail!(ErrorKind::ZeroOpsMatchTag(tag.clone())),
						1 => Ok(set.keys().next().unwrap().clone()),
						_ => bail!(ErrorKind::MultipleOpsMatchTag(tag.clone())),
					}
				}
			},
			OpTag::Int(_) => {
				let set = match self.op_tags.get(&tag){
					Some(set) => set,
					None => bail!(ErrorKind::ZeroOpsMatchTag(tag)),
				};
				match set.len() {
					0 => bail!(ErrorKind::ZeroOpsMatchTag(tag)),
					1 => Ok(set.keys().next().unwrap().clone()),
					_ => bail!(ErrorKind::MultipleOpsMatchTag(tag)),
				}
			}
		}
	}

	pub fn op_ids<'a, T: Into<OpTag>>(&'a self, tag: T) -> Box<Iterator<Item=OpID> + 'a> {
		let tag = tag.into();
		match tag {
			OpTag::Id(ind) => Box::new(iter::once(OpID{index: ind})),
			OpTag::Int(_) | OpTag::Str(_)  => {
				match self.op_tags.get(&tag){
					Some(set) => Box::new(set.keys().cloned()),
					None => Box::new(iter::empty::<OpID>()),
				}
			}
		}
	}

	/// Returns the number of nodes in the graph.
	pub fn num_nodes(&self) -> usize{
		self.node_ids.len()
	}

	/// Returns the number of tensors in the graph.
	///
	/// Currently this is twice the number of nodes (values and gradients).
	pub fn num_data(&self) -> usize{
		self.node_ids.len()*2
	}

	/// Returns the number of ops in the graph.
	pub fn num_ops(&self) -> usize{
		self.op_ids.len()
	}

	/// Returns the number of passes in the graph.
	///
	/// Currently this is twice the number of ops (forward pass and backwards pass).
	pub fn num_passes(&self) -> usize{
		self.passes.len()
	}
}

#[derive(Clone, Debug)]
pub struct Dependencies {
	pass_inputs: Vec<Vec<DataID>>,
	pass_outputs: Vec<Vec<DataID>>,
	pass_is_forward: Vec<bool>,
	data_inputs: Vec<Vec<PassID>>,
	data_outputs: Vec<Vec<PassID>>,

	op_inputs: Vec<Vec<NodeID>>,
	op_outputs: Vec<Vec<NodeID>>,
	op_shape_outputs: Vec<Vec<NodeID>>,
	node_inputs: Vec<Vec<OpID>>,
	node_shape_inputs: Vec<Vec<OpID>>,
	node_outputs: Vec<Vec<OpID>>,
}

impl Dependencies {
	pub fn new(graph: &GraphDef) -> Dependencies {
		let mut pass_inputs = vec![];
		let mut pass_outputs = vec![];
		let mut pass_is_forward = vec![];

		for pass_id in &graph.pass_ids {
			let (inputs, outputs) = graph.passes[pass_id.index].dependencies();
			let is_forward = inputs.iter().chain(outputs.iter()).all(|data_id| data_id.is_value());
			pass_inputs.push(inputs);
			pass_outputs.push(outputs);
			pass_is_forward.push(is_forward);			
		}

		let mut data_inputs: Vec<Vec<PassID>> = (0..graph.num_data()).map(|_| vec![]).collect();
		let mut data_outputs: Vec<Vec<PassID>> = (0..graph.num_data()).map(|_| vec![]).collect();

		for pass_id in &graph.pass_ids {
			for data_id in &pass_inputs[pass_id.index] {
				data_outputs[data_id.index].push(pass_id.clone());
			}
			for data_id in &pass_outputs[pass_id.index] {
				data_inputs[data_id.index].push(pass_id.clone())
			}
		}


		let mut op_inputs = vec![];
		let mut op_outputs = vec![];
		let mut op_shape_outputs = vec![];

		for op_id in &graph.op_ids {
			let (inputs, outputs) = graph.ops[op_id.index].dependencies();
			let mut shape_outputs = graph.ops[op_id.index].inner_nodes();
			shape_outputs.extend_from_slice(&outputs);
			op_inputs.push(inputs);
			op_outputs.push(outputs);
			op_shape_outputs.push(shape_outputs)
		}

		let mut node_inputs: Vec<Vec<OpID>> = (0..graph.num_nodes()).map(|_| vec![]).collect();
		let mut node_shape_inputs: Vec<Vec<OpID>> = (0..graph.num_nodes()).map(|_| vec![]).collect();
		let mut node_outputs: Vec<Vec<OpID>> = (0..graph.num_nodes()).map(|_| vec![]).collect();

		for op_id in &graph.op_ids {
			for node_id in &op_inputs[op_id.index] {
				node_outputs[node_id.index].push(op_id.clone());
			}
			for node_id in &op_outputs[op_id.index] {
				node_inputs[node_id.index].push(op_id.clone())
			}
			for node_id in &op_shape_outputs[op_id.index] {
				node_shape_inputs[node_id.index].push(op_id.clone())
			}
		}

		Dependencies{pass_inputs, pass_outputs, pass_is_forward, data_inputs, data_outputs, op_inputs, op_outputs, op_shape_outputs, node_inputs, node_shape_inputs, node_outputs}
	}

	pub fn data_inputs(&self, data_id: &DataID) -> &[PassID] {
		&self.data_inputs[data_id.index]
	}

	pub fn data_outputs(&self, data_id: &DataID) -> &[PassID] {
		&self.data_outputs[data_id.index]
	}

	pub fn pass_inputs(&self, pass_id: &PassID) -> &[DataID] {
		&self.pass_inputs[pass_id.index]
	}
	
	pub fn pass_is_forward(&self, pass_id: &PassID) -> bool {
		self.pass_is_forward[pass_id.index]
	}

	pub fn pass_outputs(&self, pass_id: &PassID) -> &[DataID] {
		&self.pass_outputs[pass_id.index]
	}

	pub fn node_inputs(&self, node_id: &NodeID) -> &[OpID] {
		&self.node_inputs[node_id.index]
	}

	pub fn node_outputs(&self, node_id: &NodeID) -> &[OpID] {
		&self.node_outputs[node_id.index]
	}

	pub fn op_inputs(&self, op_id: &OpID) -> &[NodeID] {
		&self.op_inputs[op_id.index]
	}
	
	pub fn op_outputs(&self, op_id: &OpID) -> &[NodeID] {
		&self.op_outputs[op_id.index]
	}

	pub fn op_shape_outputs(&self, op_id: &OpID) -> &[NodeID] {
		&self.op_shape_outputs[op_id.index]
	}
}


// enum to record the compute status of each data
#[derive(Clone, Debug)]
enum DataStatus {
	Input,
	Compute,
	NotIncluded,
}

/// enum to record the shape status of each node
#[derive(Clone, Debug)]
enum NodeStatus {
	Input,
	StaticInput,
	Known,
	Infer,
	NotIncluded,
}

#[derive(Clone, Debug)]
pub struct Subgraph {
	graph: GraphDef,
	dependencies: Dependencies,

	subgraph_inputs: Vec<DataID>,
	subgraph_outputs: Vec<DataID>,

	// Only keep static inputs that dont conflict with a subgraph input
	filtered_static_inputs: OrderMap<DataID, ArrayD<f32>>,

	// Ops and nodes included in the subgraph, used to perform shape inference
	included_nodes: Vec<NodeStatus>,
	included_ops: Vec<bool>,
	op_order: Vec<OpID>,
	shapes: Vec<IxDyn>,

	// passes and data inclded in the subgraph, used to perform graph execution
	included_data: Vec<DataStatus>,
	included_passes: Vec<bool>,
	pass_order: Vec<PassID>,
	passes_before_dealloc: Vec<usize>,

}

impl Subgraph {
	/// An executable subgraph derived from a `GraphDef`.
	///
	/// todo
	fn new(graph: &GraphDef, inputs: &[DataID], outputs: &[DataID]) -> Result<Subgraph> {

		let dependencies = Dependencies::new(graph);
		
		// Find the minimum set of data, passes, nodes and ops required to perform shape inference and calculate the `outputs` of the subgraph
		let (included_data, included_passes, included_nodes, included_ops) = find_included(&graph, inputs, &graph.static_inputs, outputs, &dependencies);

		// Find the order of ops
		let op_order = find_op_order(&graph, &included_nodes, &included_ops, &dependencies)?;
		let pass_order = find_pass_order(&graph, &included_data, &included_passes, &dependencies)?;

		// remove overlap between static_inpts and inputs
		let filtered_static_inputs = graph.static_inputs.iter()
			.filter(|&(k, _v)| !inputs.contains(k))
			.map(|(k, v)| (k.clone(), v.clone())).collect();

		// for each data_id count the number of passes deemed required which depend on it, then add 1 if it is a requested output
		let mut passes_before_dealloc: Vec<usize> = dependencies.data_outputs.iter().map(|passes| passes.iter().filter(|pass| included_passes[pass.index]).count()).collect();
		for data_id in outputs {
			passes_before_dealloc[data_id.index] += 1;
		}

		let graph = Subgraph{
			graph: graph.clone(),
			dependencies: dependencies,

			filtered_static_inputs: filtered_static_inputs,

			included_nodes: included_nodes,
			included_ops: included_ops,
			op_order: op_order,
			shapes: vec![],

			included_data: included_data,
			included_passes: included_passes,
			pass_order: pass_order,
			passes_before_dealloc: passes_before_dealloc,

			subgraph_inputs: inputs.to_vec(),
			subgraph_outputs: outputs.to_vec(),
		};

		Ok(graph)
	}

	/// Calling this executes the subgraph and returns a Storage which contains the outputs of the subgraph.
	///
	/// todo
	pub fn execute(&mut self, inputs: Vec<ArrayD<f32>>) -> Result<Storage>{
		assert_eq!(inputs.len(), self.subgraph_inputs.len());

		// if shapes is empty, or doesnt match the new inputs, recalculate all shapes.
		if self.shapes.len() != inputs.len()
		|| inputs.iter().enumerate().any(|(i, input)|{input.shape() != self.shapes[self.subgraph_inputs[i].node_id().index].slice()}) {
			self.shapes = find_shapes(&self, &self.op_order, &self.subgraph_inputs, &inputs, &self.filtered_static_inputs)?;
		}

		let mut storage = Storage::new(&self.included_data, &self.dependencies, &self.filtered_static_inputs, &self.subgraph_inputs, inputs, &self.shapes, &self.graph);

		let mut passes_before_dealloc = self.passes_before_dealloc.clone();

		for pass_id in &self.pass_order {
			storage.set_current_pass(Some(pass_id.clone()));
			let pass_data = self.graph.passes[pass_id.index].run(&mut storage)?;
			storage.set_pass_data(pass_id, pass_data);

			for data_id in &self.dependencies.pass_inputs[pass_id.index] {
				debug_assert!(passes_before_dealloc[data_id.index] > 0);
				passes_before_dealloc[data_id.index] -= 1;
				if passes_before_dealloc[data_id.index] == 0 {
					storage.deallocate(data_id);
				}
			}
			
			storage = storage.clear_borrow_flags();
		}
		storage.set_current_pass(None);

		Ok(storage)
	}

	pub fn graph(&self) -> &GraphDef {
		&self.graph
	}

	pub fn inputs(&self) -> &[DataID]{
		&self.subgraph_inputs
	}

	pub fn outputs(&self) -> &[DataID]{
		&self.subgraph_outputs
	}
}


/// Work backwards from the requested output data marking data, passes, nodes, and ops as required.
fn find_included(graph: &GraphDef, inputs: &[DataID], static_inputs: &OrderMap<DataID, ArrayD<f32>>, outputs: &[DataID], dependencies: &Dependencies) -> (Vec<DataStatus>, Vec<bool>, Vec<NodeStatus>, Vec<bool>){
		
	let mut included_data: Vec<DataStatus> = vec![DataStatus::NotIncluded; graph.num_data()];
	let mut included_passes: Vec<bool> = vec![false; graph.num_passes()];


	for data_id in inputs.iter().chain(static_inputs.keys()) {
		included_data[data_id.index] = DataStatus::Input;
	}

	// queues contain locations which should be visited and marked required
	// if a node is already marked when removed from the queue then nothing happens, otherwise it is marked and its dependencies get added to the queue
	let mut pass_queue: VecDeque<PassID> = VecDeque::new();
	let mut data_queue: VecDeque<DataID> = VecDeque::new(); 

	// start with the data requested by the user
	for data_id in outputs {
		data_queue.push_back(data_id.clone());
	}

	// Continue propagating to dependencies, stopping at inputs to graph.
	// This is robust to circular graphs, as locations already marked as required will be passed over if visited a second time
	while !(pass_queue.is_empty() && data_queue.is_empty()) {

		if let Some(data_id) = data_queue.pop_front() {
			let data_status = &mut included_data[data_id.index];
			match data_status {
				&mut DataStatus::Input | &mut DataStatus::Compute => {},
				&mut DataStatus::NotIncluded => {
					*data_status = DataStatus::Compute;
					for pass_id in &dependencies.data_inputs[data_id.index] {
						pass_queue.push_back(pass_id.clone());
					}
				}
				
			}
		}

		if let Some(pass_id) = pass_queue.pop_front() {
			if !included_passes[pass_id.index] {
				included_passes[pass_id.index] = true;
				for data_id in &dependencies.pass_inputs[pass_id.index] {
					data_queue.push_back(data_id.clone());
				}
			}
		}

	}

	let mut included_nodes = vec![NodeStatus::NotIncluded; graph.num_nodes()];
	let mut included_ops = vec![false; graph.num_ops()];

	for data_id in static_inputs.keys() {
		included_nodes[data_id.node_id().index] = NodeStatus::StaticInput;
	}

	// Input overwrites StaticInput
	for data_id in inputs.iter() {
		included_nodes[data_id.node_id().index] = NodeStatus::Input;
	}

	for (i, data) in included_data.iter().enumerate() {
		let data_id = DataID{index: i};
		match data {
			&DataStatus::Compute => {
				let node_status = &mut included_nodes[data_id.node_id().index];
				match node_status{
					&mut NodeStatus::NotIncluded => {
						if graph.node_shapes[data_id.node_id().index].is_known() {
							*node_status = NodeStatus::Known;
						} else {
							*node_status = NodeStatus::Infer;
						}
					},
					_ => {}
				};

			},
			_ => {},
		};
	}

	for op_id in &graph.op_ids {
		let shape_inputs = &dependencies.op_inputs[op_id.index];
		let shape_outputs = &dependencies.op_shape_outputs[op_id.index];

		// if outputs are 'Infer' include
		// if outputs are `StaticInputs` AND no inputs are `NotIncluded` then include
		if shape_outputs.iter().any(|node_id| matches!(included_nodes[node_id.index], NodeStatus::Infer)) {
			included_ops[op_id.index] = true
		} else if shape_outputs.iter().any(|node_id| matches!(included_nodes[node_id.index], NodeStatus::Infer))
		&& !shape_inputs.iter().any(|node_id| matches!(included_nodes[node_id.index], NodeStatus::NotIncluded)){
			included_ops[op_id.index] = true
		}
	}


	(included_data, included_passes, included_nodes, included_ops)
}

/// Returns the order in which passes should be called such that dependencies are respected.
/// By default this will order passes in the order that they were added to the graph, and only perform the minimal rearrangement required to ensure dependencies are met.
/// out of order dependeancies can cause quadratic slow down (this can probably be removed using priority queues)
fn find_pass_order(graph: &GraphDef, included_data: &[DataStatus], included_passes: &[bool], dependencies: &Dependencies) -> Result<Vec<PassID>>{
	debug_assert_eq!(graph.num_passes(), dependencies.pass_inputs.len());
	debug_assert_eq!(graph.num_passes(), dependencies.pass_outputs.len());
	debug_assert_eq!(graph.num_passes(), included_passes.len());

	debug_assert_eq!(graph.num_data(), dependencies.data_inputs.len());
	debug_assert_eq!(graph.num_data(), dependencies.data_outputs.len());
	debug_assert_eq!(graph.num_data(), included_data.len());


	#[derive(Clone)]
	enum DataState {
		Input,
		Ready, // Data should be marked as ready if 1) it is a graph input, or 2) when the last input pass to it is sucessfully retired as ready
		Pending(usize), // Indicates the number of remaining input passes before the data can be marked as ready
		Unavailable, // propagated if any input pass is unavailable, but does not overwrite Ready (to preserve inputs to the graph as ready).
	};

	// states start as pending, if all inputs to a pass or data are 'ready' then it is ready, if any inputs to a pass or node are unavailable then it is unavailable.
	#[derive(Clone)]
	enum PassState {
		Ready, // Indicates that a pass has been retired as ready
		Pending(usize), // Indicates the number of remaining input data before the pass can be marked as ready
		PendingUnavailable, // an input to the 
		Unavailable, // propagated if any input for data or a pass is unavailable, but does not overwrite Ready.
	};

	/// Attempts to retire a pass as Ready or Unavailable, return true if sucessful false otherwise
	/// If it returns true this method should never be called again for that pass_id.
	fn try_retire_pass(pass_id: &PassID, pass_order: &mut Vec<PassID>, data_state: &mut[DataState], passes_ready: &mut [PassState], dependencies: &Dependencies) -> bool{
		if matches!(passes_ready[pass_id.index] , PassState::Ready | PassState:: Unavailable) {
			panic!("pass has already been retired, try_retire_pass() should not be called")
		} else if matches!(passes_ready[pass_id.index] , PassState::Pending(0)) {
			// add to pass order and update output data locations readiness
			pass_order.push(pass_id.clone());
			passes_ready[pass_id.index] = PassState::Ready;
			for data_id in &dependencies.pass_outputs[pass_id.index] {
				// If a data outptu of a pass is pending decrement
				// If that data output can now be marked ready
				match data_state[data_id.index] {
					DataState::Unavailable | DataState::Input => {},
					DataState::Pending(rem) if rem == 1 => {
						mark_data_ready(data_id, data_state, passes_ready, &dependencies)
					},
					DataState::Pending(rem) if rem > 1 => {data_state[data_id.index] = DataState::Pending(rem - 1)},
					DataState::Pending(_) => panic!("Data with zero inputs should have already been marked Unavailable or Input"),
					DataState::Ready => panic!("data marked ready before last input pass was processed. graph likely contains a requires pass which writes to a input tensor"), //TODO: create test to confirm this is caused by fan-out ops writing to a subgraph input
				}
			}
			true
		} else if matches!(passes_ready[pass_id.index] , PassState::PendingUnavailable) {
			passes_ready[pass_id.index] = PassState::Unavailable;
			for data_id in &dependencies.pass_outputs[pass_id.index] {
				mark_data_unavailable(data_id, data_state, passes_ready, &dependencies)
			}
			true
		} else {
			false
		}
	}

	/// Marks data as ready, and decreases pending count of dependent passes
	/// Only legal to call this if is_input[]==true or as the last input pass is retired
	fn mark_data_input(data_id: &DataID, data_state: &mut[DataState], passes_ready: &mut [PassState], dependencies: &Dependencies){
		data_state[data_id.index] = DataState::Input;
		for pass_id in &dependencies.data_outputs[data_id.index] {
			match passes_ready[pass_id.index] {
				PassState::Pending(rem) if rem > 0 => {passes_ready[pass_id.index] = PassState::Pending(rem - 1)},
				PassState::Unavailable | PassState::PendingUnavailable =>{},
				PassState::Pending(_) | PassState::Ready => panic!("Something has happened out of order. pass_id: {:?}", pass_id),
			}
		}
	}

	/// Marks data as ready, and decreases pending count of dependent passes
	/// Only legal to call this if is_input[]==true or as the last input pass is retired
	fn mark_data_ready(data_id: &DataID, data_state: &mut[DataState], passes_ready: &mut [PassState], dependencies: &Dependencies){
		data_state[data_id.index] = DataState::Ready;
		for pass_id in &dependencies.data_outputs[data_id.index] {
			match passes_ready[pass_id.index] {
				PassState::Pending(rem) if rem > 0 => {passes_ready[pass_id.index] = PassState::Pending(rem - 1)},
				PassState::Unavailable | PassState::PendingUnavailable =>{},
				PassState::Pending(_) | PassState::Ready => panic!("Something has happened out of order. pass_id: {:?}", pass_id),
			}
		}
	}

	/// Can be called on data in any state, but will only mark data and dependent passes as unavailable if the current data state is Pending
	fn mark_data_unavailable(data_id: &DataID, data_state: &mut[DataState], passes_ready: &mut [PassState], dependencies: &Dependencies){
		if matches!(data_state[data_id.index], DataState::Ready | DataState::Unavailable){return} 
		data_state[data_id.index] = DataState::Unavailable;
		for pass_id in &dependencies.data_outputs[data_id.index] {
			match passes_ready[pass_id.index] {
				PassState::Pending(rem) if rem > 0 => {passes_ready[pass_id.index] = PassState::PendingUnavailable},
				PassState::Unavailable | PassState::PendingUnavailable =>{},
				PassState::Pending(_) | PassState::Ready => panic!("Something has happened out of order. pass_id: {:?}", pass_id),
			}
		}
	}



	let mut pass_order: Vec<PassID> = vec![];
	let mut deferred_passes: VecDeque<PassID> = VecDeque::new();
	
	// Setup states
	let mut passes_state: Vec<PassState> = (0..graph.num_passes()).map(|i| PassState::Pending(dependencies.pass_inputs[i].len())).collect();
	let mut data_state: Vec<DataState> = (0..graph.num_data()).map(|i| DataState::Pending(dependencies.data_inputs[i].len())).collect();
	for (i, data_status) in included_data.iter().enumerate() {
		match data_status {
			&DataStatus::Input => {
				mark_data_input(&DataID{index: i}, &mut data_state, &mut passes_state, &dependencies);
			},
			&DataStatus::Compute => {
				if dependencies.data_inputs[i].len() == 0  {
					mark_data_unavailable(&DataID{index: i}, &mut data_state, &mut passes_state, &dependencies);
				}
			}
			&DataStatus::NotIncluded => {
				mark_data_unavailable(&DataID{index: i}, &mut data_state, &mut passes_state, &dependencies);
			}
		}
	}




	// iterate over all required passes,
	// add to pass order where possible (inputs are ready), otherwise add to deferred queue
	// the resulting pass order should be as close to the users order while still not being out of order
	// let forward_required_passes = (0..graph.num_passes()).map(|i| PassID{index:i}).filter(|id| id.is_forward() && required_passes[id.index]);
	// let backward_required_passes = (0..graph.num_passes()).map(|i| PassID{index:i}).filter(|id| id.is_backward() && required_passes[id.index]);

	let forward_required_passes = graph.pass_ids.iter().filter(|id| dependencies.pass_is_forward[id.index] && included_passes[id.index]);
	let backward_required_passes = graph.pass_ids.iter().filter(|id| !dependencies.pass_is_forward[id.index] && included_passes[id.index]);

	let default_pass_order = forward_required_passes.chain(backward_required_passes.rev());
	for pass_id in default_pass_order {

		let success = try_retire_pass(&pass_id, &mut pass_order, &mut data_state, &mut passes_state, &dependencies);
		if !success {
			deferred_passes.push_back(pass_id.clone());
			continue;
		}

		// Attempt to empty deferred queue
		// always try to add deferred passes in order
		let mut i = 0;
		while i < deferred_passes.len(){
			let success = try_retire_pass(&deferred_passes[i], &mut pass_order, &mut data_state, &mut passes_state, &dependencies);
			if success {
				deferred_passes.remove(i);
				i = 0; // keep trying from the start again
			} else {
				i += 1;
			}
		}
	}
	
	let unavailable_data: Vec<String> = (0..graph.num_data())
		.filter(|&i| !matches!(included_data[i], DataStatus::NotIncluded) && matches!(data_state[i], DataState::Unavailable))
		.map(|i| graph.data_name(&DataID{index: i})).collect();
	if unavailable_data.len() > 0 {
		bail!(ErrorKind::SubgraphInsufficientInputsForOutputs(unavailable_data))
	}

	if deferred_passes.len() > 0 {
		bail!(ErrorKind::GraphContainsCircularPasses(deferred_passes.into_iter().map(|pass| (pass.clone(), dependencies.pass_inputs[pass.index].iter().filter(|data_id| !matches!(data_state[data_id.index], DataState::Ready)).cloned().collect())).collect()))
	}

	Ok(pass_order)
}

/// Returns the order in which op should be called such that dependencies are respected.
/// By default this will order ops in the order that they were added to the graph, and only perform the minimal rearrangement required to ensure dependencies are met.
/// out of order dependeancies can cause quadratic slow down (this can probably be removed using priority queues)
fn find_op_order(graph: &GraphDef, included_nodes: &[NodeStatus], included_ops: &[bool], dependencies: &Dependencies) -> Result<Vec<OpID>>{
	debug_assert_eq!(graph.num_ops(), dependencies.op_inputs.len());
	debug_assert_eq!(graph.num_ops(), dependencies.op_shape_outputs.len());
	debug_assert_eq!(graph.num_ops(), included_ops.len());

	debug_assert_eq!(graph.num_nodes(), dependencies.node_shape_inputs.len());
	debug_assert_eq!(graph.num_nodes(), dependencies.node_outputs.len());
	debug_assert_eq!(graph.num_nodes(), included_nodes.len());


	#[derive(Clone)]
	enum NodeState {
		Input,
		Ready, // Node should be marked as ready if 1) it is a graph input, or 2) when the last input op to it is sucessfully retired as ready
		Pending(usize), // Indicates the number of remaining input opes before the node can be marked as ready
		Unavailable, // propagated if any input op is unavailable, but does not overwrite Ready (to preserve inputs to the graph as ready).
	};

	// states start as pending, if all inputs to a op or node are 'ready' then it is ready, if any inputs to a op or node are unavailable then it is unavailable.
	#[derive(Clone)]
	enum OpState {
		Ready, // Indicates that a op has been retired as ready
		Pending(usize), // Indicates the number of remaining input node before the op can be marked as ready
		PendingUnavailable, // an input to the op has marked it as unavailable, but the op has not propagated unavailable to its outputs yet
		Unavailable, // propagated if any input for node or a op is unavailable, but does not overwrite Ready.
	};

	/// Attempts to retire a op as Ready or Unavailable, return true if sucessful false otherwise
	/// If it returns true this method should never be called again for that op_id.
	fn try_retire_op(graph: &GraphDef, op_id: &OpID, op_order: &mut Vec<OpID>, node_state: &mut[NodeState], ops_ready: &mut [OpState], dependencies: &Dependencies) -> bool{
		if matches!(ops_ready[op_id.index] , OpState::Ready | OpState:: Unavailable) {
			panic!("op has already been retired, try_retire_op() should not be called")
		} else if matches!(ops_ready[op_id.index] , OpState::Pending(0)) {
			// add to op order and update output node locations readiness
			op_order.push(op_id.clone());
			ops_ready[op_id.index] = OpState::Ready;
			for node_id in &dependencies.op_shape_outputs[op_id.index] {
				// If a node outptu of a op is pending decrement
				// If that node output can now be marked ready
				match node_state[node_id.index] {
					NodeState::Unavailable | NodeState::Input => {},
					NodeState::Pending(rem) if rem == 1 => {
						mark_node_ready(graph, node_id, node_state, ops_ready, &dependencies)
					},
					NodeState::Pending(rem) if rem > 1 => {node_state[node_id.index] = NodeState::Pending(rem - 1)},
					NodeState::Pending(_) => panic!("node with zero inputs should have already been marked Unavailable or Input"),
					NodeState::Ready => panic!("node marked ready before last input op was processed. graph likely contains a requires op which writes to a input tensor"), //TODO: create test to confirm this is caused by fan-out ops writing to a subgraph input
				}
			}
			true
		} else if matches!(ops_ready[op_id.index] , OpState::PendingUnavailable) {
			ops_ready[op_id.index] = OpState::Unavailable;
			for node_id in &dependencies.op_shape_outputs[op_id.index] {
				mark_node_unavailable(graph, node_id, node_state, ops_ready, &dependencies)
			}
			true
		} else {
			false
		}
	}

	/// Marks node as ready, and decreases pending count of dependent ops
	/// Only legal to call this if is_input[]==true or as the last input op is retired
	fn mark_node_input(graph: &GraphDef, node_id: &NodeID, node_state: &mut[NodeState], ops_ready: &mut [OpState], dependencies: &Dependencies){
		node_state[node_id.index] = NodeState::Input;
		for op_id in &dependencies.node_outputs[node_id.index] {
			match ops_ready[op_id.index] {
				OpState::Pending(rem) if rem > 0 => {ops_ready[op_id.index] = OpState::Pending(rem - 1)},
				OpState::Unavailable | OpState::PendingUnavailable =>{},
				OpState::Pending(_) | OpState::Ready => panic!("Something has happened out of order. node_id: {:?} op_id: {:?} op_name: {}", node_id, op_id, graph.op_name(op_id)),
			}
		}
	}

	/// Marks node as ready, and decreases pending count of dependent ops
	/// Only legal to call this if is_input[]==true or as the last input op is retired
	fn mark_node_ready(graph: &GraphDef, node_id: &NodeID, node_state: &mut[NodeState], ops_ready: &mut [OpState], dependencies: &Dependencies){
		node_state[node_id.index] = NodeState::Ready;
		for op_id in dependencies.node_outputs(node_id) {
			match ops_ready[op_id.index] {
				OpState::Pending(rem) if rem > 0 => {ops_ready[op_id.index] = OpState::Pending(rem - 1)},
				OpState::Unavailable | OpState::PendingUnavailable =>{},
				OpState::Pending(_) | OpState::Ready => panic!("Something has happened out of order. node_id: {:?} op_id: {:?} op_name: {}", node_id, op_id, graph.op_name(op_id)),
			}
		}
	}

	/// Can be called on node in any state, but will only mark node and dependent ops as unavailable if the current node state is Pending
	fn mark_node_unavailable(graph: &GraphDef, node_id: &NodeID, node_state: &mut[NodeState], ops_ready: &mut [OpState], dependencies: &Dependencies){
		if matches!(node_state[node_id.index], NodeState::Ready | NodeState::Unavailable){return} 
		node_state[node_id.index] = NodeState::Unavailable;
		for op_id in &dependencies.node_outputs[node_id.index] {
			match ops_ready[op_id.index] {
				OpState::Pending(rem) if rem > 0 => {ops_ready[op_id.index] = OpState::PendingUnavailable},
				OpState::Unavailable | OpState::PendingUnavailable =>{},
				OpState::Pending(_) | OpState::Ready => panic!("Something has happened out of order. node_id: {:?} op_id: {:?} op_name: {}", node_id, op_id, graph.op_name(op_id)),
			}
		}
	}


	let mut op_order: Vec<OpID> = vec![];
	let mut deferred_ops: VecDeque<OpID> = VecDeque::new();
	
	// Setup states
	let mut op_state: Vec<OpState> = (0..graph.num_ops()).map(|i| OpState::Pending(dependencies.op_inputs[i].len())).collect();
	let mut node_state: Vec<NodeState> = (0..graph.num_nodes()).map(|i| NodeState::Pending(dependencies.node_shape_inputs[i].len())).collect();
	for (i, node_status) in included_nodes.iter().enumerate() {
		match node_status{
			&NodeStatus::Input | &NodeStatus::Known => {
				mark_node_input(graph, &NodeID{index: i}, &mut node_state, &mut op_state, &dependencies)
			},
			&NodeStatus::StaticInput => {
				if !dependencies.node_shape_inputs[i].iter().all(|op_id| included_ops[op_id.index]) {
					mark_node_input(graph, &NodeID{index: i}, &mut node_state, &mut op_state, &dependencies)
				}
			},
			&NodeStatus::Infer => {
				if dependencies.node_shape_inputs[i].len() == 0 {
					mark_node_unavailable(graph, &NodeID{index: i}, &mut node_state, &mut op_state, &dependencies)
				}
			},
			&NodeStatus::NotIncluded => {
				mark_node_unavailable(graph, &NodeID{index: i}, &mut node_state, &mut op_state, &dependencies)
			},
		}
	}




	// iterate over all required ops,
	// add to op order where possible (inputs are ready), otherwise add to deferred queue
	// the resulting op order should be as close to the users order while still not being out of order
	// let forward_required_ops = (0..graph.num_ops()).map(|i| opID{index:i}).filter(|id| id.is_forward() && required_ops[id.index]);
	// let backward_required_ops = (0..graph.num_ops()).map(|i| opID{index:i}).filter(|id| id.is_backward() && required_ops[id.index]);

	let required_ops = graph.op_ids.iter().filter(|id| included_ops[id.index]);

	for op_id in required_ops {

		let success = try_retire_op(graph, &op_id, &mut op_order, &mut node_state, &mut op_state, &dependencies);
		if !success {
			deferred_ops.push_back(op_id.clone());
			continue;
		}

		// Attempt to empty deferred queue
		// always try to add deferred ops in order
		let mut i = 0;
		while i < deferred_ops.len(){
			let success = try_retire_op(graph, &deferred_ops[i], &mut op_order, &mut node_state, &mut op_state, &dependencies);
			if success {
				deferred_ops.remove(i);
				i = 0; // keep trying from the start again
			} else {
				i += 1;
			}
		}
	}

	let unavailable_nodes: Vec<String> = (0..graph.num_nodes())
		.filter(|&i| !matches!(included_nodes[i], NodeStatus::NotIncluded) && matches!(node_state[i], NodeState::Unavailable))
		.map(|i| graph.node_name(&NodeID{index: i}).to_string()).collect();
	if unavailable_nodes.len() > 0 {
		bail!(ErrorKind::SubgraphInsufficientInputsForShapeInference(unavailable_nodes))
	}

	if deferred_ops.len() > 0 {
		bail!(ErrorKind::GraphContainsCircularOps(deferred_ops.into_iter().map(|op_id| (op_id.clone(), dependencies.op_inputs[op_id.index].iter().filter(|node_id| !matches!(node_state[node_id.index], NodeState::Ready)).cloned().collect())).collect()))
	}

	Ok(op_order)
}


fn find_shapes(subgraph: &Subgraph, op_order: &[OpID], inputs: &[DataID], input_data: &[ArrayD<f32>], static_inputs: &OrderMap<DataID, ArrayD<f32>>) -> Result<Vec<IxDyn>> {
	// if inputs are present along with static_inputs the inputs should add

	let mut shapes = GraphShapes::new(subgraph);

	// for all inputs, merge data shape into existing graph shape
	ensure!(inputs.len() == input_data.len(), ErrorKind::InputSizeError);
	for (input, input_data) in inputs.iter().zip(input_data) {
		shapes.merge_input(&input, input_data.shape())?;
	}

	// for all static inputs, if not in inputs, merge into graph shape
	// because static_inputs can be broadcast, resolving the dimension will be harder
	// iterate from the lowest dimension up, if the static_input dimension is not 1 then enforce it in the shape
	for (static_input, static_input_data) in static_inputs.iter() {
		if !inputs.contains(static_input) {
			shapes.merge_static_input(&static_input, static_input_data.shape())?;
		}
	}

	// for all op forward passes that are scheduled. call the relevant shape propagation
	//let op_ids = passes.iter().filter(|pass| pass.is_forward()).map(|pass_id| pass_id.op_id());
	for op_id in op_order {
		shapes.set_current_op(Some(op_id.clone()));
		subgraph.graph.ops[op_id.index].propagate_shape_constraints(&mut shapes)?;
	}
	shapes.set_current_op(None);

	shapes.shapes.iter_mut().map(|shape| {
		shape.collapse_dimensions_to_minimum();
		shape.to_data_shape().map_err(|e| e.into())
	}).collect()
}



/// The interface through which ops can perform shape propagation.
///
/// Immediately prior to each graph execution, the propagation of shape constraints from inputs through the graph takes place.
/// Each Op can read the shape of its inputs, and new constraints can be applied/merged with the shapes of its outputs.
#[derive(Debug)]
pub struct GraphShapes<'a> {
	shapes: Vec<NodeShape>,
	subgraph: &'a Subgraph,
	current_op_instance: Option<OpID>,
}

impl<'a> GraphShapes<'a> {
	fn new(subgraph: &Subgraph) -> GraphShapes {
		GraphShapes{
			shapes: subgraph.graph.node_shapes.clone(),
			subgraph: subgraph,
			current_op_instance: None,
		}
	}

	pub fn graph(&self) -> &GraphDef {
		&self.subgraph.graph
	}

	fn set_current_op(&mut self, op_id: Option<OpID>){
		self.current_op_instance = op_id;
	}

	pub fn current_op_instance(&self) -> &Option<OpID>{
		&self.current_op_instance
	}

	fn merge_input(&mut self, data_id: &DataID, shape: &[Ix]) -> Result<()>{
		self.shapes[data_id.node_id().index] = self.shapes[data_id.node_id().index].merge(&shape.iter().cloned().into())?;
		Ok(())
	}

	fn merge_static_input(&mut self, data_id: &DataID, shape: &[Ix]) -> Result<()> {
		let shape: NodeShape = shape.iter().map(|&ix| if ix == 1 {NodeDim::Unknown} else {NodeDim::Known(ix)}).into();
		self.shapes[data_id.node_id().index] = self.shapes[data_id.node_id().index].merge(&shape)?;
		Ok(())
	}

	// TODO only allow getting inputs
	pub fn get_shape(&mut self, id: &NodeID) -> &NodeShape{
		self.shapes[id.index].collapse_dimensions_to_minimum();
		debug_assert!(self.shapes[id.index].dimensions().iter().all(|dim| matches!(dim, &NodeDim::Known(_))));
		&self.shapes[id.index]
	}

	// TODO only allow getting outputs
	pub fn get_output_shape(&mut self, id: &NodeID) -> &NodeShape{
		// self.shapes[id.index].collapse_dimensions_to_minimum();
		// debug_assert!(self.shapes[id.index].dimensions().iter().all(|dim| matches!(dim, &NodeDim::Known(_))));
		&self.shapes[id.index]
	}

	// TODO only allow merging to outputs
	pub fn merge_with(&mut self, id: &NodeID, shape: &NodeShape) -> Result<()>{
		self.shapes[id.index] = self.shapes[id.index].merge(shape)?;
		Ok(())
	}
}






enum DataState<T>{
	NotRequired,
	Unallocated,
	UnallocatedInput(usize), //index in input_data
	UnallocatedStaticInput,
	Allocated(T),
	Deallocated,
}

/// This type allows Ops to access the values and gradients of nodes at execution time.
///
/// To achieve safe mutable access to multiple nodes this structure uses runtime checked borrowing,
/// similar to a RefCell for a Collection of Arrays, but with some limitations.
/// Each element can only be borrowed either once mutably or many times immutably, however,
/// once borrowed as such it is stuck until `clear_borrow_flags()` is called (typically after each pass is completed).
pub struct Storage<'a> {
	shapes: &'a [IxDyn],
	input_data: Vec<ArrayD<f32>>,
	loss: Cell<f32>,
	data: Vec<DataState<ArrayD<f32>>>,
	borrow_flags: Vec<Cell<usize>>,
	static_inputs: &'a OrderMap<DataID, ArrayD<f32>>,
	dependencies: &'a Dependencies,
	current_pass: Option<PassID>,
	pass_data: Vec<Option<Box<Any>>>,
	graph: &'a GraphDef,
}

const UNUSED: usize = 0;
const WRITING: usize = !0;
impl<'a> Storage<'a> {

	fn new(included_data: &[DataStatus], dependencies: &'a Dependencies, static_inputs: &'a OrderMap<DataID, ArrayD<f32>>, supplied_inputs: &[DataID], input_data: Vec<ArrayD<f32>>, shapes: &'a[IxDyn], graph: &'a GraphDef) -> Storage<'a> {
		debug_assert_eq!(supplied_inputs.len(), input_data.len());

		let num_nodes = dependencies.node_inputs.len();
		let num_data = dependencies.data_inputs.len();
		let num_passes = dependencies.pass_inputs.len();
		
		debug_assert_eq!(num_nodes, shapes.len());
		debug_assert_eq!(num_data, included_data.len());

		let mut data: Vec<DataState<ArrayD<f32>>> = included_data.iter().map(|r| if matches!(r, &DataStatus::NotIncluded) {DataState::NotRequired} else {DataState::Unallocated}).collect();

		for (i, (data_id, input_data)) in supplied_inputs.iter().zip(&input_data).enumerate() {
			debug_assert!(shapes[data_id.node_id().index].slice() == input_data.shape());
			data[data_id.index] = DataState::UnallocatedInput(i);
		}

		for (data_id, _data) in static_inputs.iter() {
			debug_assert!(!matches!(data[data_id.index], DataState::UnallocatedInput(_)));
			data[data_id.index] = DataState::UnallocatedStaticInput;
		}

		Storage{
			shapes: shapes,
			input_data: input_data,
			loss: Cell::new(0.0),
			data: data,
			borrow_flags: vec![Cell::new(UNUSED); num_data],
			static_inputs,
			dependencies,
			current_pass: None,
			pass_data: (0..num_passes).map(|_| None).collect(),
			graph: graph,
		}
	}

	pub fn graph(&self) -> &GraphDef {
		self.graph
	}

	fn set_pass_data(&mut self, pass_id: &PassID, pass_data: Box<Any>){
		self.pass_data[pass_id.index] = Some(pass_data);
	}

	pub fn get_pass_data(&self, pass_id: &PassID) -> Option<&Any>{
		self.pass_data[pass_id.index].as_ref().map(|x| &**x)
	}

	/// If this value is not `None`, all subsequent accesses will be checked against the dependency list for the Pass.
	/// This can be useful to ensure that passes dont access anything they havent listed as and input or output.
	fn set_current_pass(&mut self, pass_id: Option<PassID>){
		self.current_pass = pass_id;
	}

	pub fn get_current_pass(&self) -> &Option<PassID>{
		&self.current_pass
	}

	/// Deallocates the data specified by DataID.
	fn deallocate(&mut self, data_id: &DataID){
		mem::replace(&mut self.data[data_id.index], DataState::Deallocated);
	}

	/// This resets runtime borrow checks, allowing for a new round of borrowing patterns.
	/// By taking `self` this forces return of all prior borrows.
	pub fn clear_borrow_flags(mut self) -> Self{
		for e in &mut self.borrow_flags{
			e.set(UNUSED);
		}
		self
	}	
		
	/// Should never be called if a &mut borrow could possibly already exist.
	unsafe fn get_or_init(&self, id: &DataID) -> Result<*mut ArrayD<f32>>{
		let ptr = &self.data[id.index] as *const _ as *mut _;
		match *ptr {
			DataState::NotRequired => bail!(ErrorKind::StorageDataMarkedNotRequired),
			DataState::Deallocated => bail!(ErrorKind::StorageDataDeallocated),
			DataState::Unallocated => {
				*ptr = DataState::Allocated(ArrayD::zeros(self.shapes[id.node_id().index].clone()));
			},
			DataState::UnallocatedInput(ind) =>{
				*ptr = DataState::Allocated(self.input_data[ind].clone())
			},
			DataState::UnallocatedStaticInput => {
				let shape = self.shapes[id.node_id().index].clone();
				if let Some(ref static_data) = self.static_inputs.get(id){
					if let Some(broadcasted_view) = static_data.broadcast(shape){
						*ptr = DataState::Allocated(broadcasted_view.to_owned())
					} else {
						bail!(ErrorKind::StaticInputBroadcastFailure(id.node_id(), static_data.shape().to_owned(), self.shapes[id.node_id().index].slice().to_owned()))
					}
				} else {
					unreachable!();
				}
			},
			DataState::Allocated(_) => {},
		}

		// return pointer to allocated data
		if let DataState::Allocated(ref mut data) = *ptr{
			Ok(data as *mut ArrayD<f32>)
		} else {
			unreachable!()
		}
	}

	/// Access the loss variable.
	pub fn loss(&self) -> f32 {
		self.loss.get()
	}

	/// Access the loss variable.
	/// Loss should only be added to in the backwards passes of ops.
	pub fn loss_add(&self, additional_loss: f32){
		unsafe{*self.loss.as_ptr() += additional_loss;}
	}

	/// Immutably borrows data element associated with the given ID
	/// Will panic if data element is already mutably borrowed.
	/// The borrow will stick until `clear_borrow_flags()` is called.
	pub fn get<'b>(&'b self, data_id: &DataID) -> Result<ArrayViewD<f32>> {
		if let Some(ref pass_id) = self.current_pass {
			ensure!(self.dependencies.pass_inputs[pass_id.index].contains(data_id)||self.dependencies.pass_outputs[pass_id.index].contains(data_id), ErrorKind::StorageImmutableBorrowError(self.graph.pass_name(pass_id), self.graph.data_name(data_id)));
		}

		if self.borrow_flags[data_id.index].get() != WRITING {
				let ptr = unsafe{self.get_or_init(data_id)?};
				self.borrow_flags[data_id.index].set(self.borrow_flags[data_id.index].get() + 1);
				let array: &'b ArrayD<f32> = unsafe{&*ptr};
				Ok(array.view())
		} else {
			bail!(ErrorKind::StorageDataAlreadyMutablyBorrowed)
		}
	}

	/// Mutably borrows data element associated with the given ID.
	/// Will panic if data element is already mutably or immutably borrowed.
	/// The borrow will stick until `clear_borrow_flags()` is called.
	pub fn get_mut<'b>(&'b self, data_id: &DataID) -> Result<ArrayViewMutD<f32>> {
		if let Some(ref pass_id) = self.current_pass {
			ensure!(self.dependencies.pass_outputs[pass_id.index].contains(data_id), ErrorKind::StorageMutableBorrowError(self.graph.pass_name(pass_id), self.graph.data_name(data_id)));
		}
		match self.borrow_flags[data_id.index].get() {
			UNUSED => {
				let ptr = unsafe{self.get_or_init(data_id)?};
				self.borrow_flags[data_id.index].set(WRITING);
				let array: &'b mut ArrayD<f32> = unsafe{&mut *ptr};
				Ok(array.view_mut())
			},
			WRITING => bail!(ErrorKind::StorageDataAlreadyMutablyBorrowed),
			_ => bail!(ErrorKind::StorageDataAlreadyBorrowed),
		}
	}

	/// Returns true if a `DataID` is a required component of the subgraph.
	///
	/// Checking this is only required for the outputs of a pass, and only if a pass has multiple outputs.
	/// If false, no attempt should be made to write to that data_id using 'get_mut()'.
	pub fn is_required(&self, data_id: &DataID) -> bool {
		!matches!(self.data[data_id.index], DataState::NotRequired) //TODO this doesnt perfectly match the required_data vector from graph
	}

	/// Consume the Storage and converts it into a OrderMap.
	///
	/// Intended for use after storage is returned from `execute()`.
	pub fn into_map(self) -> OrderMap<DataID, ArrayD<f32>> {
		self.data.into_iter().enumerate().filter_map(|(i, entry)|{
			match entry {
				DataState::Allocated(arr) => Some((DataID{index: i}, arr)),
				_ => None,
			}
		}).collect()
	}
}














#[test]
fn test_build(){
	_test_build().unwrap();
}

fn _test_build() -> Result<()>{
	use new::ops::dummy::Dummy;
	use new::graph::GraphDef;
	use new::shape;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![Unknown, 5, 16], "node1", tag!["input"])?;
	let node2 = g.new_node(shape![Unknown, 5, 16], "node2", tag![])?;
	g.new_op(Dummy::new().name("first op").input(&node1).output(&node2), tag![])?;

	let mut prev_node = node2.clone();
	for i in 3..10 {
		let next_node = g.new_node(shape![Unknown, 5, 16], format!("node{}", i), tag![i])?;
		g.new_op(Dummy::new().name(format!("op{}", i)).input(&prev_node).output(&next_node), tag![])?;
		prev_node = next_node;
	}

	g.new_op(Dummy::new().name("last op").input(&prev_node), tag![])?;

	let sg1 = g.subgraph(&[node2.value_id()], &[prev_node.value_id()])?;
	let sg2 = g.subgraph(&[node1.value_id()], &[node2.gradient_id()])?;

	assert!(sg1.pass_order.len() > 0);
	assert!(sg2.pass_order.len() > 0);

	Ok(())
}


#[test]
fn test_execute(){
	_test_execute().unwrap();
}

fn _test_execute() -> Result<()>{
	use new::ops::dummy::Dummy;
	use new::graph::GraphDef;
	use new::shape;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![4, 5, 16], "node1", tag!["input"])?;
	let node2 = g.new_node(shape![4, 5, 16], "node2", tag![])?;
	g.new_op(Dummy::new().name("first op").input(&node1).output(&node2).touch_data(true), tag![])?;

	let mut prev_node = node2.clone();
	for i in 3..10 {
		let next_node = g.new_node(shape![4, 5, 16], format!("node{}", i), tag![i])?;
		g.new_op(Dummy::new().name(format!("op{}", i)).input(&prev_node).output(&next_node).touch_data(true), tag![])?;
		prev_node = next_node;
	}

	g.new_op(Dummy::new().name("last op").input(&prev_node).touch_data(true), tag![])?;

	let mut sg1 = g.subgraph(&[node2.value_id()], &[prev_node.value_id()])?;
	let mut sg2 = g.subgraph(&[node1.value_id()], &[node2.gradient_id()])?;


	sg1.execute(vec![ArrayD::zeros(&[4, 5, 16][..])])?;

	sg2.execute(vec![ArrayD::zeros(&[4, 5, 16][..])])?;

	Ok(())
}


#[test]
fn test_execute_deallocation(){
	_test_execute_deallocation().unwrap();
}

fn _test_execute_deallocation() -> Result<()>{
	use new::ops::dummy::Dummy;
	use new::graph::GraphDef;
	use new::shape;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![Unknown, 5, 16], "node1", tag!["input"])?;
	let node2 = g.new_node(shape![4, 5, 16], "node2", tag![])?;
	g.new_op(Dummy::new().name("first op").input(&node1).output(&node2).touch_data(true), tag![])?;

	let mut prev_node = node2.clone();
	for i in 3..10 {
		let next_node = g.new_node(shape![4, 5, 16], format!("node{}", i), tag![i])?;
		g.new_op(Dummy::new().name(format!("op{}", i)).input(&prev_node).output(&next_node).touch_data(true), tag![])?;
		prev_node = next_node;
	}

	g.new_op(Dummy::new().name("last op").input(&prev_node).touch_data(true), tag![])?;

	let mut sg1 = g.subgraph(&[node2.value_id()], &[prev_node.value_id()])?;
	let mut sg2 = g.subgraph(&[node1.value_id()], &[node2.gradient_id()])?;

	// Make sure we get the right errors when accessing nodes that should be deallocated or never have been allocated
	let s1 = sg1.execute(vec![ArrayBase::zeros(&[4, 5, 16][..])])?;
	s1.get(&prev_node.value_id()).unwrap();
	assert!(matches!(s1.get(&node2.value_id()), Err(Error(ErrorKind::StorageDataDeallocated, _))));
	assert!(matches!(s1.get(&node2.gradient_id()), Err(Error(ErrorKind::StorageDataMarkedNotRequired, _))));

	let s2 = sg2.execute(vec![ArrayBase::zeros(&[9, 5, 16][..])])?;
	s2.get(&node2.gradient_id()).unwrap();
	assert!(matches!(s2.get(&prev_node.value_id()), Err(Error(ErrorKind::StorageDataDeallocated, _))));
	assert!(matches!(s2.get(&node1.gradient_id()), Err(Error(ErrorKind::StorageDataMarkedNotRequired, _))));

	Ok(())
}

#[test]
fn test_pass_reordering(){
	_test_pass_reordering().unwrap();
}

fn _test_pass_reordering() -> Result<()>{
	use new::ops::dummy::Dummy;
	use new::graph::GraphDef;
	use new::shape;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![Unknown, 5, 16], "node1", tag!["input"])?;
	let node2a = g.new_node(shape![Unknown, 5, 16], "node2a", tag![])?;
	let node2b = g.new_node(shape![Unknown, 5, 16], "node2b", tag![])?;
	let node3 = g.new_node(shape![Unknown, 5, 16], "node3", tag![])?;
	let node4 = g.new_node(shape![Unknown, 5, 16], "node4", tag!["output"])?;

	let o4 = g.new_op(Dummy::new().input(&node2a).output(&node3), tag![])?;
	let o2 = g.new_op(Dummy::new().input(&node2b).output(&node3), tag![])?;
	let o1 = g.new_op(Dummy::new().input(&node1).output(&node2b), tag![])?;
	let o3 = g.new_op(Dummy::new().input(&node1).output(&node2a), tag![])?;
	let o5 = g.new_op(Dummy::new().input(&node3).output(&node4), tag![])?;
	let o6 = g.new_op(Dummy::new().input(&node4), tag![])?;


	let sg_forward = g.subgraph(&[node1.value_id()], &[node4.value_id()])?;
	let expected_order: Vec<OpID> = [&o1, &o2, &o3, &o4, &o5].iter().map(|&op_id| op_id.clone()).collect();
	assert_eq!(&sg_forward.op_order, &expected_order);
	let expected_order: Vec<PassID> = [&o1, &o2, &o3, &o4, &o5].iter().map(|&op_id| g.op(op_id)).collect::<Result<Vec<_>>>()?
		.iter().map(|op| op.inner_passes()[0].clone()).collect();
	assert_eq!(&sg_forward.pass_order, &expected_order);

	let sg_forward_backward = g.subgraph(&[node1.value_id()], &[node1.gradient_id()])?;
	let expected_order: Vec<OpID> = [&o1, &o2, &o3, &o4, &o5].iter().map(|&op_id| op_id.clone()).collect();
	assert_eq!(&sg_forward_backward.op_order, &expected_order);
	// backward pass prefers to run in the opposite order to the order of ops being added
	// this results in o2 and o1 firing before o4 and o3
	// if you read the op order above bottom to top, you can see this is correct.
	let expected_order: Vec<PassID> = [&o1, &o2, &o3, &o4, &o5].iter().map(|&op_id| g.op(op_id)).collect::<Result<Vec<_>>>()?
		.iter().map(|op| op.inner_passes()[0].clone())
		.chain([&o6, &o5, &o2, &o1, &o4, &o3].iter().map(|&op_id| g.op(op_id)).collect::<Result<Vec<_>>>()?
		.iter().map(|op| op.inner_passes()[1].clone())).collect();
	assert_eq!(&sg_forward_backward.pass_order, &expected_order);

	Ok(())
}


#[test]
fn test_circular_detection(){
	_test_circular_detection().unwrap();
}

fn _test_circular_detection() -> Result<()>{
	use new::ops::dummy::Dummy;
	use new::graph::GraphDef;
	use new::shape;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![Unknown, 5, 16], "node1", tag!["input"])?;
	let node2 = g.new_node(shape![Unknown, 5, 16], "node2", tag![])?;
	let node3 = g.new_node(shape![Unknown, 5, 16], "node3", tag![])?;
	let node4 = g.new_node(shape![Unknown, 5, 16], "node4", tag![])?;
	let node5 = g.new_node(shape![Unknown, 5, 16], "node5", tag!["output"])?;


	let _o1 = g.new_op(Dummy::new().input(&node1).output(&node2), tag![])?;
	let _o2 = g.new_op(Dummy::new().input(&node2).output(&node3), tag![])?;
	let _o3 = g.new_op(Dummy::new().input(&node3).output(&node4), tag![])?;
	let _o4 = g.new_op(Dummy::new().input(&node4).output(&node2), tag![])?; // circular link
	let _o5 = g.new_op(Dummy::new().input(&node4).output(&node5), tag![])?;
	let _o6 = g.new_op(Dummy::new().input(&node5), tag![])?;


	// Check that the circular link raises an error
	let sg_forward = g.subgraph(&[node1.value_id()], &[node5.value_id()]);
	assert!(matches!(sg_forward, Err(Error(ErrorKind::GraphContainsCircularOps(_), _))), "{:?}", sg_forward);

	let sg_forward_backward = g.subgraph(&[node1.value_id()], &[node1.gradient_id()]);
	assert!(matches!(sg_forward_backward, Err(Error(ErrorKind::GraphContainsCircularOps(_), _))), "{:?}", sg_forward_backward);


	// Check that setting node2 as an input means that _o4 is not required, resulting in a non circular graph
	let sg_forward = g.subgraph(&[node2.value_id()], &[node5.value_id()]);
	assert!(matches!(sg_forward, Ok(_)), "{:?}", sg_forward);

	// However the circular dependencies in the backward passes still exists
	let sg_forward_backward = g.subgraph(&[node2.value_id()], &[node2.gradient_id()]);
	assert!(matches!(sg_forward_backward, Err(Error(ErrorKind::GraphContainsCircularPasses(_), _))), "{:?}", sg_forward_backward);

	// TODO check Graph ContainsCircularPass

	Ok(())
}


#[test]
fn test_insufficient_input_detection(){
	_test_insufficient_input_detection().unwrap();
}

fn _test_insufficient_input_detection() -> Result<()>{
	use new::ops::dummy::Dummy;
	use new::graph::GraphDef;
	use new::shape;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![Unknown, 5, 16], "node1", tag!["input"])?;
	let node2 = g.new_node(shape![Unknown, 5, 16], "node2", tag![])?;
	let node3 = g.new_node(shape![Unknown, 5, 16], "node3", tag!["output"])?;


	let _o1 = g.new_op(Dummy::new().input(&node1).output(&node3), tag![])?;
	let _o2 = g.new_op(Dummy::new().input(&node2).output(&node3), tag![])?;


	let sg_forward = g.subgraph(&[node1.value_id()], &[node3.value_id()]);
	assert!(matches!(sg_forward, Err(Error(ErrorKind::SubgraphInsufficientInputsForShapeInference(_), _))), "{:?}", sg_forward);

	let sg_forward_backward = g.subgraph(&[node1.value_id()], &[node1.gradient_id()]);
	assert!(matches!(sg_forward_backward, Err(Error(ErrorKind::SubgraphInsufficientInputsForShapeInference(_), _))), "{:?}", sg_forward_backward);

	Ok(())
}



// TODO detect required ops which want to write to input data

// TODO detect that name conflict detection works

// TODO detect problems with shape propagation

// TODO detect problems with static_input broadcasting