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

error_chain!{
	errors {
		/// This errorkind indicates that the data requested was never allocated as it was not a required component of the subgraph.
		StorageDataMarkedNotRequired{} //TODO give name of node
		/// This errorkind indicates that the data requested has already been deallocated.
		/// Ensure that it was included in the output used to define the subgraph being executed.
		StorageDataDeallocated{}
		/// This errorkind indicates that data requested cannot be mutably borrowed as it has already been immutably borrowed.
		/// Borrows are not reset until after the operation pass has completed.
		StorageDataAlreadyBorrowed{}
		/// This errorkind indicates that data requested cannot be borrowed as it has already been mutably borrowed.
		/// Borrows are not reset until after the operation pass has completed.
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
		GraphContainsCircularDependencies(deferred_passes: Vec<(PassID, Vec<DataID>)>){
			display("The following passes were required, but could not be included in the execution order: {:?}", deferred_passes) //TODO change to print op and node names
		}
		InputsInsufficientForRequestedOutputs{}
		InputSizeError{}
		StaticInputBroadcastFailure(id: NodeID, s1: Vec<Ix>, s2: Vec<Ix>){
			display("Broadcast of initial value failed for node {:?} as shape {:?} could not be broadcast to shape: {:?}", id, s1, s2)
		}
		PassAttemptedToAccessDataNotListedAsInputOrOutput{}
		PassAttemptedToMutablyAccessDataNotListedAsOutput{}

		// Operation Errors
		ShapePropagationError(message: String){
			display("{}", message)
		}
		ForwardPassError(message: String){
			display("{}", message)
		}
		BackwardPassError(message: String){
			display("{}", message)
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

/// A unique identifier for a graph operation.
#[derive(PartialEq, Eq, Ord, PartialOrd, Hash, Clone, Debug)]
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

/// A unique identifier for the (forward or backward) pass of an operator.
#[derive(PartialEq, Eq, Ord, PartialOrd, Hash, Clone, Debug)]
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
/// When calling `add_operation()` consider using the `tag![]` macro to supply tags.
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
		OpTag::Int(i.borrow().index)
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

/// Used to construct the definition of the computational hypergraph.
/// This cannot be executed, an executable `Graph` can be built using 
pub struct Builder {

	nodes: Vec<NodeID>,
	node_shapes: Vec<NodeShape>,
	node_names: OrderMap<String, NodeID>,
	node_tags: OrderMap<NodeTag, OrderMap<NodeID, ()>>,

	ops: Vec<Box<Operation>>,
	op_names: OrderMap<String, OpID>,
	op_tags: OrderMap<OpTag, OrderMap<OpID, ()>>,

	static_inputs: OrderMap<DataID, ArrayD<f32>>,
}

impl Builder {
	
	pub fn new() -> Builder {
		Builder {
			nodes: vec![],
			node_shapes: vec![],
			node_names: OrderMap::new(),
			node_tags: OrderMap::new(),

			ops: vec![],
			op_names: OrderMap::new(),
			op_tags: OrderMap::new(),

			static_inputs: OrderMap::new(),
		}
	}

	///
	/// * inputs - this must be an in order slice of the nodes which will be supplied by the data stream used when evaluating the graph.
	pub fn build(&self, inputs: &[DataID], requested_outputs: &[DataID]) -> Result<Graph> {
		Graph::new(&self, inputs, requested_outputs)
	}

	/// Node values are initialised to be zero filled by default
	/// If a NdArray value is supplied to this method that can be broadcast to this node, it will be used to set the initial value of the node
	/// This can be used to supply fixed inputs, which replace parameters, to operations when further operations is to be avoided.
	pub fn set_static_input(&mut self, id: DataID, value: ArrayD<f32>){
		self.static_inputs.insert(id, value);
	}

	pub fn clear_static_input(&mut self, id: DataID){
		self.static_inputs.remove(&id);
	}

	pub fn add_operation<B: OperationBuilder>(&mut self, builder: B, tags: Vec<OpTag>) -> Result<OpID> {
		
		let op = Box::new(builder.build(self)?);
		
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
		let op_id = OpID{index: self.ops.len()};
		self.op_names.insert(name, op_id.clone());
		self.ops.push(op);

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

	// fn add_layer(SimpleOperationBuilder and output NodeShape){}
	// fn add_simple_operation<B: SimpleOperationBuilder>(&mut self, mut builder: B, shape: NodeShape) -> (OpID, NodeID) {
	// 	let name = unimplemented!();
	// 	let node_id = self.new_node(builder.required_output_shape(), name, &[]);
	// 	builder.set_output(&node_id);
	// 	let op_id = self.add_operation(builder);
	// 	(op_id, node_id)
	// }

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
		let node_id = NodeID{index: self.nodes.len()};
		self.node_names.insert(name, node_id.clone());
		self.node_shapes.push(shape);
		self.nodes.push(node_id.clone());
		
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

	pub fn node_name(&self, node_id: &NodeID) -> &str{
		for (k, v) in self.node_names.iter(){
			if v == node_id {
				return k;
			}
		}
		unreachable!()
	}

	// TODO check names
	pub fn is_node_tagged<T: Into<NodeTag>>(&self, node_id: NodeID, tag: T) -> bool {
		let tag = tag.into();
		match tag {
			NodeTag::Id(ind) => {ind == node_id.index},
			NodeTag::Int(_) | NodeTag::Str(_) | NodeTag::Parameter => {
				match self.node_tags.get(&tag){
					Some(set) => set.contains_key(&node_id),
					None => false,
				}
			}
		}
	}

	// TODO check names
	pub fn is_op_tagged<T: Into<OpTag>>(&self, op_id: OpID, tag: T) -> bool {
		let tag = tag.into();
		match tag {
			OpTag::Id(ind) => {ind == op_id.index},
			OpTag::Int(_) | OpTag::Str(_) => {
				match self.op_tags.get(&tag){
					Some(set) => set.contains_key(&op_id),
					None => false,
				}
			}
		}
	}

	// Returns the NodeID which was tagged with 'tag'. returns none if zero or more than one NodeIDs are associated with the tag.
	pub fn get_node_id<T: Into<NodeTag>>(&self, tag: T) -> Result<NodeID> {
		let tag = tag.into();
		match tag {
			NodeTag::Id(ind) => Ok(self.nodes[ind].clone()),
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

	pub fn get_node_ids<'a, T: Into<NodeTag>>(&'a self, tag: T) -> OrderMap<NodeID, ()> {
		let tag = tag.into();
		match tag {
			NodeTag::Id(ind) => iter::once((self.nodes[ind].clone(), ())).collect(),
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

	pub fn get_node_shape<T: Into<NodeTag>>(&self, tag: T) -> Result<&NodeShape> {
		self.get_node_id(tag).map(|id| &self.node_shapes[id.index])
	}

	pub fn get_parameter_ids<'a>(&'a self) -> OrderMap<NodeID, ()> {
		self.get_node_ids(NodeTag::Parameter)
	}

	pub fn get_op<T: Into<OpTag>>(&self, tag: T) -> Result<&Operation> {
		self.get_op_id(tag).map(|id| &*self.ops[id.index])
	}

	/// Returns a single OpID matching a op tag
	/// Returns an error if multiple or zero ops are associated with the tag
	pub fn get_op_id<T: Into<OpTag>>(&self, tag: T) -> Result<OpID> {
		let tag = tag.into();
		match tag {
			OpTag::Id(ind) => Ok(OpID{index: ind}),
			OpTag::Str(ref string) => {
				//self.operation_tags.get(&tag).and_then(|set| if set.len() == 1 {Some(set.keys().next().unwrap().clone())} else {None})
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

	pub fn get_op_ids<'a, T: Into<OpTag>>(&'a self, tag: T) -> Box<Iterator<Item=OpID> + 'a> {
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

	/// Returns the number of tensors in the graph.
	///
	/// Currently this is twice the number of nodes (values and gradients).
	pub fn num_data(&self) -> usize{
		self.nodes.len()*2
	}

	/// Returns the number of passes in the graph.
	///
	/// Currently this is twice the number of ops (forward pass and backwards pass).
	pub fn num_passes(&self) -> usize{
		self.ops.len()*2
	}
}

#[derive(Clone, Debug)]
pub struct Dependencies {
	pass_inputs: Vec<Vec<DataID>>,
	pass_outputs: Vec<Vec<DataID>>,
	data_inputs: Vec<Vec<PassID>>,
	data_outputs: Vec<Vec<PassID>>,
}

impl Dependencies {
	fn new(builder: &Builder) -> Dependencies {
		let mut pass_inputs: Vec<Vec<DataID>> = (0..builder.num_passes()).map(|_| vec![]).collect();
		let mut pass_outputs: Vec<Vec<DataID>> = (0..builder.num_passes()).map(|_| vec![]).collect();

		for op_id in (0..builder.ops.len()).map(|i| OpID{index:i}) {
			let op = &*builder.ops[op_id.index];

			let forward_id = op_id.forward_id();
			let (forward_inputs, forward_outputs) = op.forward_dependencies();
			pass_inputs[forward_id.index] = forward_inputs;
			pass_outputs[forward_id.index] = forward_outputs;

			let backward_id = op_id.backward_id();
			let (backward_inputs, backward_outputs) = op.backward_dependencies();
			pass_inputs[backward_id.index] = backward_inputs;
			pass_outputs[backward_id.index] = backward_outputs;
		}

		let mut data_inputs: Vec<Vec<PassID>> = (0..builder.num_data()).map(|_| vec![]).collect();
		let mut data_outputs: Vec<Vec<PassID>> = (0..builder.num_data()).map(|_| vec![]).collect();

		for pass_id in (0..builder.ops.len()*2).map(|i| PassID{index:i}) {
			for data_id in &pass_inputs[pass_id.index] {
				data_outputs[data_id.index].push(pass_id.clone());
			}
			for data_id in &pass_outputs[pass_id.index] {
				data_inputs[data_id.index].push(pass_id.clone())
			}
		}

		Dependencies{pass_inputs, pass_outputs, data_inputs, data_outputs}
	}
}

/// 
#[derive(Clone, Debug)]
pub struct Graph{
	dependencies: Dependencies,

	nodes: Vec<NodeID>,
	node_shapes: Vec<NodeShape>,
	node_names: OrderMap<String, NodeID>,
	ops: Vec<Box<Operation>>,

	filtered_static_inputs: OrderMap<DataID, ArrayD<f32>>,

	required_data: Vec<bool>,
	required_passes: Vec<bool>,
	pass_order: Vec<PassID>,

	supplied_inputs: Vec<DataID>,
	requested_outputs: Vec<DataID>,

	passes_before_dealloc: Vec<usize>,

	shapes: Vec<IxDyn>,
}

impl Graph {
	fn new(builder: &Builder, inputs: &[DataID], requested_outputs: &[DataID]) -> Result<Graph> {

		let is_input = find_inputs(builder.num_data(), inputs, &builder.static_inputs);

		let dependencies = Dependencies::new(builder);
		
		// Find the minimum set of operations and nodes required to calculate the `required_outputs`
		let (required_data, required_passes) = find_required(builder.num_passes(), builder.num_data(), &is_input, requested_outputs, &dependencies);

		// Find the order of operations
		let pass_order = find_pass_order(&builder, &is_input, &required_data, &required_passes, &dependencies)?;

		// remove overlap between static_inpts and inputs
		let filtered_static_inputs = builder.static_inputs.iter()
			.filter(|&(k, _v)| !inputs.contains(k))
			.map(|(k, v)| (k.clone(), v.clone())).collect();

		// for each data_id count the number of passes deemed required which depend on it, then add 1 if it is a requested output
		let mut passes_before_dealloc: Vec<usize> = dependencies.data_outputs.iter().map(|passes| passes.iter().filter(|pass| required_passes[pass.index]).count()).collect();
		for data_id in requested_outputs {
			passes_before_dealloc[data_id.index] += 1;
		}

		let graph = Graph{
			dependencies: dependencies,

			nodes: builder.nodes.clone(),
			node_shapes: builder.node_shapes.clone(),
			node_names: builder.node_names.clone(),

			ops: builder.ops.clone(),

			filtered_static_inputs: filtered_static_inputs,

			required_data: required_data,
			required_passes: required_passes,
			pass_order: pass_order,

			supplied_inputs: inputs.to_vec(),
			requested_outputs: requested_outputs.to_vec(),

			passes_before_dealloc: passes_before_dealloc,

			shapes: vec![],
		};

		Ok(graph)
	}

	/// Calling this executes the 
	///
	/// 
	pub fn execute(&mut self, inputs: Vec<ArrayD<f32>>) -> Result<Storage>{
		assert_eq!(inputs.len(), self.supplied_inputs.len());

		// if shapes is empty, or doesnt match the new inputs, recalculate all shapes.
		if self.shapes.len() != inputs.len()
		|| inputs.iter().enumerate().any(|(i, input)|{input.shape() != self.shapes[self.supplied_inputs[i].index].slice()}) {
			self.shapes = find_shapes(&self, &self.pass_order, &self.supplied_inputs, &inputs, &self.filtered_static_inputs)?;
		}

		let mut storage = Storage::new(&self.required_data, &self.dependencies, &self.filtered_static_inputs, &self.supplied_inputs, inputs, &self.shapes);

		let mut passes_before_dealloc = self.passes_before_dealloc.clone();

		for pass in &self.pass_order {
			storage.set_next_pass_debug(Some(pass));
			if pass.is_forward() {
				self.ops[pass.op_id().index].forward(&mut storage)?;
			}
			if pass.is_backward() {
				self.ops[pass.op_id().index].backward(&mut storage)?;
			}

			for data_id in &self.dependencies.pass_inputs[pass.index] {
				debug_assert!(passes_before_dealloc[data_id.index] > 0);
				passes_before_dealloc[data_id.index] -= 1;
				if passes_before_dealloc[data_id.index] == 0 {
					storage.deallocate(data_id);
				}
			}
			
			storage = storage.clear_borrow_flags();
		}
		storage.set_next_pass_debug(None);

		Ok(storage)
	}

	pub fn inputs(&self) -> &[DataID]{
		&self.supplied_inputs
	}

	pub fn outputs(&self) -> &[DataID]{
		&self.requested_outputs
	}
}

/// Returns a Vec<bool> indicating whether the array value of a DataID is available as an input or static_input
fn find_inputs(n_data: usize, inputs: &[DataID], static_inputs: &OrderMap<DataID, ArrayD<f32>>) -> Vec<bool>{
	let mut is_input = vec![false; n_data];
	for data_id in inputs.iter().chain(static_inputs.keys()) {
		is_input[data_id.index] = true;
	}
	is_input
}


/// Work backwards from the requested output data marking data and operation passes as required.
fn find_required(n_passes: usize, n_data: usize, is_input: &[bool], requested_outputs: &[DataID], dependencies: &Dependencies) -> (Vec<bool>, Vec<bool>){
	assert_eq!(n_passes, dependencies.pass_inputs.len());
	assert_eq!(n_data, dependencies.data_inputs.len());
	assert_eq!(n_data, is_input.len());
	
	let mut required_data: Vec<bool> = vec![false; n_data];
	let mut required_passes: Vec<bool> = vec![false; n_passes];
	
	// queues contain locations which should be visited and marked required
	// if a node is already marked when removed from the queue then nothing happens, otherwise it is marked and its dependencies get added to the queue
	let mut pass_queue: VecDeque<PassID> = VecDeque::new();
	let mut data_queue: VecDeque<DataID> = VecDeque::new(); 

	// start with the data requested by the user
	for data_id in requested_outputs {
		data_queue.push_back(data_id.clone());
	}

	// Continue propagating to dependencies, stopping at inputs to graph.
	// This is robust to circular graphs, as locations already marked as required will be passed over if visited a second time
	while !(pass_queue.is_empty() && data_queue.is_empty()) {

		if let Some(data_id) = data_queue.pop_front() {
			if !required_data[data_id.index] {
				required_data[data_id.index] = true;
				if !is_input[data_id.index]{
					for pass_id in &dependencies.data_inputs[data_id.index] {
						pass_queue.push_back(pass_id.clone());
					}
				}
			}
		}

		if let Some(pass_id) = pass_queue.pop_front() {
			if !required_passes[pass_id.index] {
				required_passes[pass_id.index] = true;
				for data_id in &dependencies.pass_inputs[pass_id.index] {
					data_queue.push_back(data_id.clone());
				}
			}
		}

	}
	
	(required_data, required_passes)
}

/// Returns the order in which passes should be called such that dependencies are respected.
/// By default this will order passes in the order that they were added to the graph, and only perform the minimal rearrangement required to ensure dependencies are met.
/// out of order dependeancies can cause quadratic slow down (this can probably be removed using priority queues)
fn find_pass_order(builder: &Builder, is_input: &[bool], required_data: &[bool], required_passes: &[bool], dependencies: &Dependencies) -> Result<Vec<PassID>>{
	assert_eq!(builder.num_passes(), dependencies.pass_inputs.len());
	assert_eq!(builder.num_passes(), dependencies.pass_outputs.len());
	assert_eq!(builder.num_passes(), required_passes.len());

	assert_eq!(builder.num_data(), dependencies.data_inputs.len());
	assert_eq!(builder.num_data(), dependencies.data_outputs.len());
	assert_eq!(builder.num_data(), is_input.len());


	#[derive(Clone)]
	enum DataState {
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
	fn try_retire_pass(pass_id: &PassID, pass_order: &mut Vec<PassID>, data_ready: &mut[DataState], passes_ready: &mut [PassState], dependencies: &Dependencies) -> bool{
		if matches!(passes_ready[pass_id.index] , PassState::Ready | PassState:: Unavailable) {
			panic!("pass has already been retired, try_retire_pass() should not be called")
		} else if matches!(passes_ready[pass_id.index] , PassState::Pending(0)) {
			// add to pass order and update output data locations readiness
			pass_order.push(pass_id.clone());
			passes_ready[pass_id.index] = PassState::Ready;
			for data_id in &dependencies.pass_outputs[pass_id.index] {
				// If a data outptu of a pass is pending decrement
				// If that data output can now be marked ready
				match data_ready[data_id.index] {
					DataState::Unavailable => {},
					DataState::Pending(rem) if rem == 1 => {
						mark_data_ready(data_id, data_ready, passes_ready, &dependencies)
					},
					DataState::Pending(rem) if rem > 1 => {data_ready[data_id.index] = DataState::Pending(rem - 1)},
					DataState::Pending(_) => panic!("Data with zero inputs should have already been marked Unavailable or Ready"),
					DataState::Ready => panic!("data marked ready before last input pass was processed. graph likely contains a requires pass which writes to a input tensor"), //TODO: create test to confirm this is caused by fan-out operations writing to a subgraph input
				}
			}
			true
		} else if matches!(passes_ready[pass_id.index] , PassState::PendingUnavailable) {
			passes_ready[pass_id.index] = PassState::Unavailable;
			for data_id in &dependencies.pass_outputs[pass_id.index] {
				mark_data_unavailable(data_id, data_ready, passes_ready, &dependencies)
			}
			true
		} else {
			false
		}
	}

	/// Marks data as ready, and decreases pending count of dependent passes
	/// Only legal to call this if is_input[]==true or as the last input pass is retired
	fn mark_data_ready(data_id: &DataID, data_ready: &mut[DataState], passes_ready: &mut [PassState], dependencies: &Dependencies){
		//debug_assert!(is_input[data_id.index] || matches!(data_ready[data_id.index], State::Pending(rem) if rem == 1));
		data_ready[data_id.index] = DataState::Ready;
		for pass_id in &dependencies.data_outputs[data_id.index] {
			match passes_ready[pass_id.index] {
				PassState::Pending(rem) if rem > 0 => {passes_ready[pass_id.index] = PassState::Pending(rem - 1)},
				PassState::Unavailable | PassState::PendingUnavailable =>{},
				PassState::Pending(_) | PassState::Ready => panic!("Something has happened out of order"),
			}
		}
	}

	/// Can be called on data in any state, but will only mark data and dependent passes as unavailable if the current data state is Pending
	fn mark_data_unavailable(data_id: &DataID, data_ready: &mut[DataState], passes_ready: &mut [PassState], dependencies: &Dependencies){
		if matches!(data_ready[data_id.index], DataState::Ready | DataState::Unavailable){return} 
		data_ready[data_id.index] = DataState::Unavailable;
		for pass_id in &dependencies.data_outputs[data_id.index] {
			match passes_ready[pass_id.index] {
				PassState::Pending(rem) if rem > 0 => {passes_ready[pass_id.index] = PassState::PendingUnavailable},
				PassState::Unavailable | PassState::PendingUnavailable =>{},
				PassState::Pending(_) | PassState::Ready => panic!("Something has happened out of order"),
			}
		}
	}



	let mut pass_order: Vec<PassID> = vec![];
	let mut deferred_passes: VecDeque<PassID> = VecDeque::new();
	
	// Setup states
	let mut passes_ready: Vec<PassState> = (0..builder.num_passes()).map(|i| PassState::Pending(dependencies.pass_inputs[i].len())).collect();
	let mut data_ready: Vec<DataState> = (0..builder.num_data()).map(|i| DataState::Pending(dependencies.data_inputs[i].len())).collect();
	for (i, &is_input) in is_input.iter().enumerate() {
		if is_input {
			mark_data_ready(&DataID{index: i}, &mut data_ready, &mut passes_ready, &dependencies)
		} else if dependencies.data_inputs[i].len() == 0 {
			mark_data_unavailable(&DataID{index: i}, &mut data_ready, &mut passes_ready, &dependencies)
		}
	}




	// iterate over all required passes,
	// add to pass order where possible (inputs are ready), otherwise add to deferred queue
	// the resulting pass order should be as close to the users order while still not being out of order
	let forward_required_passes = (0..builder.num_passes()).map(|i| PassID{index:i}).filter(|id| id.is_forward() && required_passes[id.index]);
	let backward_required_passes = (0..builder.num_passes()).map(|i| PassID{index:i}).filter(|id| id.is_backward() && required_passes[id.index]);
	let default_pass_order = forward_required_passes.chain(backward_required_passes.rev());
	for pass_id in default_pass_order {

		let success = try_retire_pass(&pass_id, &mut pass_order, &mut data_ready, &mut passes_ready, &dependencies);
		if !success {
			deferred_passes.push_back(pass_id.clone());
			continue;
		}

		// Attempt to empty deferred queue
		// always try to add deferred passes in order
		let mut i = 0;
		while i < deferred_passes.len(){
			let success = try_retire_pass(&deferred_passes[i], &mut pass_order, &mut data_ready, &mut passes_ready, &dependencies);
			if success {
				deferred_passes.remove(i);
				i = 0; // keep trying from the start again
			} else {
				i += 1;
			}
		}
	}
	
	if (0..builder.num_data()).filter(|&i| required_data[i]).any(|i| matches!(data_ready[i], DataState::Unavailable)) {
		bail!(ErrorKind::InputsInsufficientForRequestedOutputs)
	}

	if deferred_passes.len() > 0 {
		bail!(ErrorKind::GraphContainsCircularDependencies(deferred_passes.into_iter().map(|pass| (pass.clone(), dependencies.pass_inputs[pass.index].iter().filter(|data_id| !matches!(data_ready[data_id.index], DataState::Ready)).cloned().collect())).collect()))
	}

	Ok(pass_order)
}



fn find_shapes(graph: &Graph, passes: &[PassID], inputs: &[DataID], input_data: &[ArrayD<f32>], static_inputs: &OrderMap<DataID, ArrayD<f32>>) -> Result<Vec<IxDyn>> {
	// if inputs are present along with static_inputs the inputs should add

	let mut shapes = GraphShapes::new(graph);

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

	// for all operation forward passes that are scheduled. call the relevant shape propagation
	let op_ids = passes.iter().filter(|pass| pass.is_forward()).map(|pass_id| pass_id.op_id());
	for op_id in op_ids {
		graph.ops[op_id.index].propagate_shape_constraints(&mut shapes)?;
	}

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
pub struct GraphShapes{
	shapes: Vec<NodeShape>,
}

impl GraphShapes {
	fn new(graph: &Graph) -> GraphShapes {
		GraphShapes{
			shapes: graph.node_shapes.clone(),
		}
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
	error: f32,
	data: Vec<DataState<ArrayD<f32>>>,
	borrow_flags: Vec<Cell<usize>>,
	static_inputs: &'a OrderMap<DataID, ArrayD<f32>>,
	dependencies: &'a Dependencies,
	next_pass: Option<PassID>
}

const UNUSED: usize = 0;
const WRITING: usize = !0;
impl<'a> Storage<'a> {

	fn new(required_data: &[bool], dependencies: &'a Dependencies, static_inputs: &'a OrderMap<DataID, ArrayD<f32>>, supplied_inputs: &[DataID], input_data: Vec<ArrayD<f32>>, shapes: &'a[IxDyn]) -> Storage<'a> {
		debug_assert_eq!(supplied_inputs.len(), input_data.len());

		let num_nodes = shapes.len();

		let mut data: Vec<DataState<ArrayD<f32>>> = required_data.iter().map(|&r| if r {DataState::Unallocated} else {DataState::NotRequired}).collect();

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
			error: 0.0,
			data: data,
			borrow_flags: vec![Cell::new(UNUSED); num_nodes * 2],
			static_inputs,
			dependencies,
			next_pass: None,
		}
	}

	/// If this value is set, all subsequent accesses will be checked against the dependency list for the Pass.
	/// This can be useful to ensure that passes dont access anything they havent listed as and input or output.
	fn set_next_pass_debug(&mut self, pass_id: Option<&PassID>){
		self.next_pass = pass_id.cloned();
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

	/// Access the error variable.
	/// Error should only be added to in the backwards passes of operations.
	pub fn error(&mut self) -> &mut f32 {
		&mut self.error
	}

	/// Immutably borrows data element associated with the given ID
	/// Will panic if data element is already mutably borrowed.
	/// The borrow will stick until `clear_borrow_flags()` is called.
	pub fn get<'b>(&'b self, data_id: &DataID) -> Result<ArrayViewD<f32>> {
		if let Some(ref pass_id) = self.next_pass {
			ensure!(self.dependencies.pass_inputs[pass_id.index].contains(data_id)||self.dependencies.pass_outputs[pass_id.index].contains(data_id), ErrorKind::PassAttemptedToAccessDataNotListedAsInputOrOutput);
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
		if let Some(ref pass_id) = self.next_pass {
			ensure!(self.dependencies.pass_outputs[pass_id.index].contains(data_id), ErrorKind::PassAttemptedToMutablyAccessDataNotListedAsOutput);
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
	use new::ops::dummy;
	use new::graph;
	use new::shape;

	let mut b = graph::Builder::new();

	let node1 = b.new_node(shape![Unknown, 5, 16], "node1", tag!["input"])?;
	let node2 = b.new_node(shape![Unknown, 5, 16], "node2", tag![])?;
	b.add_operation(dummy::Builder::new().name("first op").input(&node1).output(&node2), tag![])?;

	let mut prev_node = node2.clone();
	for i in 3..10 {
		let next_node = b.new_node(shape![Unknown, 5, 16], format!("node{}", i), tag![i])?;
		b.add_operation(dummy::Builder::new().name(format!("op{}", i)).input(&prev_node).output(&next_node), tag![])?;
		prev_node = next_node;
	}

	b.add_operation(dummy::Builder::new().name("last op").input(&prev_node), tag![])?;

	let g1 = Graph::new(&b, &[node2.value_id()], &[prev_node.value_id()])?;
	let g2 = Graph::new(&b, &[node1.value_id()], &[node2.gradient_id()])?;

	assert!(g1.pass_order.len() > 0);
	assert!(g2.pass_order.len() > 0);

	Ok(())
}


#[test]
fn test_execute(){
	_test_execute().unwrap();
}

fn _test_execute() -> Result<()>{
	use new::ops::dummy;
	use new::graph;
	use new::shape;

	let mut b = graph::Builder::new();

	let node1 = b.new_node(shape![4, 5, 16], "node1", tag!["input"])?;
	let node2 = b.new_node(shape![4, 5, 16], "node2", tag![])?;
	b.add_operation(dummy::Builder::new().name("first op").input(&node1).output(&node2).touch_data(true), tag![])?;

	let mut prev_node = node2.clone();
	for i in 3..10 {
		let next_node = b.new_node(shape![4, 5, 16], format!("node{}", i), tag![i])?;
		b.add_operation(dummy::Builder::new().name(format!("op{}", i)).input(&prev_node).output(&next_node).touch_data(true), tag![])?;
		prev_node = next_node;
	}

	b.add_operation(dummy::Builder::new().name("last op").input(&prev_node).touch_data(true), tag![])?;

	let mut g1 = Graph::new(&b, &[node2.value_id()], &[prev_node.value_id()])?;
	let mut g2 = Graph::new(&b, &[node1.value_id()], &[node2.gradient_id()])?;


	g1.execute(vec![ArrayD::zeros(&[4, 5, 16][..])])?;

	g2.execute(vec![ArrayD::zeros(&[4, 5, 16][..])])?;

	Ok(())
}


#[test]
fn test_execute_deallocation(){
	_test_execute_deallocation().unwrap();
}

fn _test_execute_deallocation() -> Result<()>{
	use new::ops::dummy;
	use new::graph;
	use new::shape;

	let mut b = graph::Builder::new();

	let node1 = b.new_node(shape![Unknown, 5, 16], "node1", tag!["input"])?;
	let node2 = b.new_node(shape![4, 5, 16], "node2", tag![])?;
	b.add_operation(dummy::Builder::new().name("first op").input(&node1).output(&node2).touch_data(true), tag![])?;

	let mut prev_node = node2.clone();
	for i in 3..10 {
		let next_node = b.new_node(shape![4, 5, 16], format!("node{}", i), tag![i])?;
		b.add_operation(dummy::Builder::new().name(format!("op{}", i)).input(&prev_node).output(&next_node).touch_data(true), tag![])?;
		prev_node = next_node;
	}

	b.add_operation(dummy::Builder::new().name("last op").input(&prev_node).touch_data(true), tag![])?;

	let mut g1 = Graph::new(&b, &[node2.value_id()], &[prev_node.value_id()])?;
	let mut g2 = Graph::new(&b, &[node1.value_id()], &[node2.gradient_id()])?;

	// Make sure we get the right errors when accessing nodes that should be deallocated or never have been allocated
	let s1 = g1.execute(vec![ArrayBase::zeros(&[4, 5, 16][..])])?;
	s1.get(&prev_node.value_id()).unwrap();
	assert!(matches!(s1.get(&node2.value_id()), Err(Error(ErrorKind::StorageDataDeallocated, _))));
	assert!(matches!(s1.get(&node2.gradient_id()), Err(Error(ErrorKind::StorageDataMarkedNotRequired, _))));

	let s2 = g2.execute(vec![ArrayBase::zeros(&[9, 5, 16][..])])?;
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
	use new::ops::dummy;
	use new::graph;
	use new::shape;

	let mut b = graph::Builder::new();

	let node1 = b.new_node(shape![Unknown, 5, 16], "node1", tag!["input"])?;
	let node2a = b.new_node(shape![Unknown, 5, 16], "node2a", tag![])?;
	let node2b = b.new_node(shape![Unknown, 5, 16], "node2b", tag![])?;
	let node3 = b.new_node(shape![Unknown, 5, 16], "node3", tag![])?;
	let node4 = b.new_node(shape![Unknown, 5, 16], "node4", tag!["output"])?;

	let o4 = b.add_operation(dummy::Builder::new().input(&node2a).output(&node3), tag![])?;
	let o2 = b.add_operation(dummy::Builder::new().input(&node2b).output(&node3), tag![])?;
	let o1 = b.add_operation(dummy::Builder::new().input(&node1).output(&node2b), tag![])?;
	let o3 = b.add_operation(dummy::Builder::new().input(&node1).output(&node2a), tag![])?;
	let o5 = b.add_operation(dummy::Builder::new().input(&node3).output(&node4), tag![])?;
	let o6 = b.add_operation(dummy::Builder::new().input(&node4), tag![])?;


	let g_forward = Graph::new(&b, &[node1.value_id()], &[node4.value_id()])?;
	let expected_order: Vec<PassID> = [&o1, &o2, &o3, &o4, &o5].iter().map(|op| op.forward_id()).collect();
	assert_eq!(&g_forward.pass_order, &expected_order);

	let g_forward_backward = Graph::new(&b, &[node1.value_id()], &[node1.gradient_id()])?;
	// backward pass prefers to run in the opposite order to the order of ops being added
	// this results in o2 and o1 firing before o4 and o3
	// if you read the op order above bottom to top, you can see this is correct.
	let expected_order: Vec<PassID> = [&o1, &o2, &o3, &o4, &o5].iter().map(|op| op.forward_id()).chain(
				[&o6, &o5, &o2, &o1, &o4, &o3].iter().map(|op| op.backward_id())).collect();
	assert_eq!(&g_forward_backward.pass_order, &expected_order);

	Ok(())
}


#[test]
fn test_circular_detection(){
	_test_circular_detection().unwrap();
}

fn _test_circular_detection() -> Result<()>{
	use new::ops::dummy;
	use new::graph;
	use new::shape;

	let mut b = graph::Builder::new();

	let node1 = b.new_node(shape![Unknown, 5, 16], "node1", tag!["input"])?;
	let node2 = b.new_node(shape![Unknown, 5, 16], "node2", tag![])?;
	let node3 = b.new_node(shape![Unknown, 5, 16], "node3", tag![])?;
	let node4 = b.new_node(shape![Unknown, 5, 16], "node4", tag![])?;
	let node5 = b.new_node(shape![Unknown, 5, 16], "node5", tag!["output"])?;


	let _o1 = b.add_operation(dummy::Builder::new().input(&node1).output(&node2), tag![]);
	let _o2 = b.add_operation(dummy::Builder::new().input(&node2).output(&node3), tag![]);
	let _o3 = b.add_operation(dummy::Builder::new().input(&node3).output(&node4), tag![]);
	let _o4 = b.add_operation(dummy::Builder::new().input(&node4).output(&node2), tag![]); // circular link
	let _o5 = b.add_operation(dummy::Builder::new().input(&node4).output(&node5), tag![]);
	let _o6 = b.add_operation(dummy::Builder::new().input(&node5), tag![]);


	// Check that the circular link raises an error
	let g_forward = Graph::new(&b, &[node1.value_id()], &[node5.value_id()]);
	assert!(matches!(g_forward, Err(Error(ErrorKind::GraphContainsCircularDependencies(_), _))));

	let g_forward_backward = Graph::new(&b, &[node1.value_id()], &[node1.gradient_id()]);
	assert!(matches!(g_forward_backward, Err(Error(ErrorKind::GraphContainsCircularDependencies(_), _))));


	// Check that setting node2 as an input means that _o4 is not required, resulting in a non circular graph
	let g_forward = Graph::new(&b, &[node2.value_id()], &[node5.value_id()]);
	assert!(matches!(g_forward, Ok(_)));

	// However the circular dependencies in the backward passes still exists
	let g_forward_backward = Graph::new(&b, &[node2.value_id()], &[node2.gradient_id()]);
	assert!(matches!(g_forward_backward, Err(Error(ErrorKind::GraphContainsCircularDependencies(_), _))));

	Ok(())
}


#[test]
fn test_insufficient_input_detection(){
	_test_insufficient_input_detection().unwrap();
}

fn _test_insufficient_input_detection() -> Result<()>{
	use new::ops::dummy;
	use new::graph;
	use new::shape;

	let mut b = graph::Builder::new();

	let node1 = b.new_node(shape![Unknown, 5, 16], "node1", tag!["input"])?;
	let node2 = b.new_node(shape![Unknown, 5, 16], "node2", tag![])?;
	let node3 = b.new_node(shape![Unknown, 5, 16], "node3", tag!["output"])?;


	let _o1 = b.add_operation(dummy::Builder::new().input(&node1).output(&node3), tag![]);
	let _o2 = b.add_operation(dummy::Builder::new().input(&node2).output(&node3), tag![]);


	let g_forward = Graph::new(&b, &[node1.value_id()], &[node3.value_id()]);
	assert!(matches!(g_forward, Err(Error(ErrorKind::InputsInsufficientForRequestedOutputs, _))));

	let g_forward_backward = Graph::new(&b, &[node1.value_id()], &[node1.gradient_id()]);
	assert!(matches!(g_forward_backward, Err(Error(ErrorKind::InputsInsufficientForRequestedOutputs, _))));

	Ok(())
}

// TODO detect required ops which want to write to input data

// TODO detect that name conflict detection works

// TODO detect problems with shape propagation

// TODO detect problems with static_input broadcasting