#![allow(unused_imports)]
use std::collections::{HashMap, HashSet};
use ndarray::ArrayD;
use ndarray::prelude::*;
use smallvec::SmallVec;
use ndarray::Ix;
//use ops::Operation;
use std::cmp;
use std::iter;
use std::iter::repeat;
use std::iter::FromIterator;
use std::sync::Arc;
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
		NodeNameMatchesExistingTag
		NodeNameMatchesExistingName
		NodeTagMatchesExistingName
		ParameterNodesMustHaveKnownSize
		ZeroNodesMatchTag
		MultipleNodesMatchTag
		GraphContainsCircularDependencies
		InputsInsufficientForRequestedOutputs
		InputSizeError
		StaticInputBroadcastFailure(id: NodeID, s1: Vec<Ix>, s2: Vec<Ix>){
			display("Broadcast of initial value failed for node {:?} as shape {:?} could not be broadcast to shape: {:?}", id, s1, s2)
		}
		PassAttemptedToAccessDataNotListedAsInputOrOutput
		PassAttemptedToMutablyAccessDataNotListedAsOutput
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

/// A unique identifier for a tensor (values or gradients) of a node
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

/// A unique identifier for a graph operation
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

/// A unique identifier for the (forward or backward) pass of an operator
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

#[macro_export]
macro_rules! tag(

    (@parse Parameter) => {
        NodeTag::Parameter,
    };
    
    (@parse $v:expr) => {
        ($v).into(),
    };
    
    ( $( $x:tt ),* ) => {
        vec![
            $(
                tag!(@parse $x)
            )*
        ]
    };
    
);

#[derive(PartialEq, Eq, Hash, Clone)]
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

impl From<String> for NodeTag{
	fn from(i: String) -> NodeTag {
		NodeTag::Str(i.to_string())
	}
}

#[derive(PartialEq, Eq, Hash, Clone)]
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

	operations: Vec<Box<Operation>>,
	operation_tags: OrderMap<OpTag, OrderMap<OpID, ()>>,

	static_inputs: OrderMap<DataID, ArrayD<f32>>,
}

impl Builder {
	
	pub fn new() -> Builder {
		unimplemented!()
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

	fn add_operation<B: OperationBuilder>(&mut self, mut builder: B) -> OpID {
		let op_id = OpID{index: self.operations.len()};
		let op = Box::new(builder.build(self));
		self.operations.push(op);
		op_id
	}

	// fn add_layer(SimpleOperationBuilder and output NodeShape){}
	// fn add_simple_operation<B: SimpleOperationBuilder>(&mut self, mut builder: B, shape: NodeShape) -> (OpID, NodeID) {
	// 	let name = unimplemented!();
	// 	let node_id = self.new_node(builder.required_output_shape(), name, &[]);
	// 	builder.set_output(&node_id);
	// 	let op_id = self.add_operation(builder);
	// 	(op_id, node_id)
	// }

	fn new_node<I: Into<String>>(&mut self, shape: NodeShape, name: I, tags: Vec<NodeTag>) -> Result<NodeID>{
		
		let name = name.into();

		// ensure names are unique w.r.t other names and tags
		ensure!(!self.node_names.contains_key(&name), ErrorKind::NodeNameMatchesExistingName);
		ensure!(!self.node_tags.contains_key(&name.as_str().into()), ErrorKind::NodeNameMatchesExistingTag);

		let node_id = NodeID{index: self.nodes.len()};
		self.node_names.insert(name, node_id.clone());
		self.nodes.push(node_id.clone());
		
		for tag in tags{
			// ensure that tags don't overlap with node names
			match tag {
				NodeTag::Str(ref tag_str) => {
					ensure!(!self.node_names.contains_key(tag_str), ErrorKind::NodeTagMatchesExistingName);
				},
				_ => {},
			}

			match tag {
				NodeTag::Id(_) => {},
				NodeTag::Int(_) => {
					self.node_tags.entry(tag).or_insert(OrderMap::new()).insert(node_id.clone(), ());
				},
				NodeTag::Str(_) => {
					self.node_tags.entry(tag).or_insert(OrderMap::new()).insert(node_id.clone(), ());
				},
				NodeTag::Parameter => {
					ensure!(shape.is_known(), ErrorKind::ParameterNodesMustHaveKnownSize);
					self.node_tags.entry(tag).or_insert(OrderMap::new()).insert(node_id.clone(), ());
				}
			}
		}

		// push shape after tags to avoid deep copy
		self.node_shapes.push(shape);

		Ok(node_id)
	}

	/// Create a node which acts as a subview of another node.
	/// Contiguous views will be free from time and memory overhead, recording just a view.
	/// Non-contigues views will incurr a memory and time overhead during runtime.
	/// Returns NodeID of the new node.
	fn new_read_view<I: Into<String>>(&mut self, name: I, shape: NodeShape, tags: Vec<NodeTag>) -> Result<NodeID>{
		unimplemented!()
	}

	fn new_write_view<I: Into<String>>(&mut self, name: I, shape: NodeShape, tags: Vec<NodeTag>) -> Result<NodeID>{
		unimplemented!()
	}

	pub fn get_node_name(&self, node_id: &NodeID) -> &str{
		for (k, v) in self.node_names.iter(){
			if v == node_id {
				return k;
			}
		}
		unreachable!()
	}

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

	pub fn is_op_tagged<T: Into<OpTag>>(&self, op_id: OpID, tag: T) -> bool {
		let tag = tag.into();
		match tag {
			OpTag::Id(ind) => {ind == op_id.index},
			OpTag::Int(_) | OpTag::Str(_) => {
				match self.operation_tags.get(&tag){
					Some(set) => set.contains_key(&op_id),
					None => false,
				}
			}
		}
	}

	// Returns the NodeID which was tagged with 'tag'. returns none if zero or more than one NodeIDs are associated with the tag.
	pub fn get_node<T: Into<NodeTag>>(&self, tag: T) -> Result<NodeID> {
		let tag = tag.into();
		match tag {
			NodeTag::Id(ind) => Ok(self.nodes[ind].clone()),
			NodeTag::Str(_) => {
				unimplemented!()
			},
			NodeTag::Int(_) | NodeTag::Parameter => {
				let set = match self.node_tags.get(&tag){
					Some(set) => set,
					None => bail!(ErrorKind::ZeroNodesMatchTag),
				};
				match set.len() {
					0 => bail!(ErrorKind::ZeroNodesMatchTag),
					1 => Ok(set.keys().next().unwrap().clone()),
					_ => bail!(ErrorKind::MultipleNodesMatchTag),
				}
			}
		}
	}

	pub fn get_nodes<'a, T: Into<NodeTag>>(&'a self, tag: T) -> OrderMap<NodeID, ()> {
		let tag = tag.into();
		match tag {
			NodeTag::Id(ind) => iter::once((self.nodes[ind].clone(), ())).collect(),
			NodeTag::Str(_) => {
				unimplemented!()
			},
			NodeTag::Int(_) | NodeTag::Parameter  => {
				match self.node_tags.get(&tag){
					Some(set) => set.clone(),//.map(move |&ind| self.nodes[ind].0.clone()).collect(),
					None => OrderMap::new(),
				}
			}
		}
	}

	pub fn get_parameters<'a>(&'a self) -> OrderMap<NodeID, ()> {
		self.get_nodes(NodeTag::Parameter)
	}

	pub fn get_op<T: Into<OpTag>>(&self, tag: T) -> Option<OpID> {
		let tag = tag.into();
		match tag {
			OpTag::Id(ind) => Some(OpID{index: ind}),
			OpTag::Int(_) | OpTag::Str(_) => {
				self.operation_tags.get(&tag).and_then(|set| if set.len() == 1 {Some(set.keys().next().unwrap().clone())} else {None})
			}
		}
	}

	pub fn get_ops<'a, T: Into<OpTag>>(&'a self, tag: T) -> Box<Iterator<Item=OpID> + 'a> {
		let tag = tag.into();
		match tag {
			OpTag::Id(ind) => Box::new(iter::once(OpID{index: ind})),
			OpTag::Int(_) | OpTag::Str(_)  => {
				match self.operation_tags.get(&tag){
					Some(set) => Box::new(set.keys().cloned()),
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

	nodes: Vec<NodeID>,
	node_shapes: Vec<NodeShape>,
	node_names: OrderMap<String, NodeID>,
	operations: Vec<Box<Operation>>,
	static_inputs: OrderMap<DataID, ArrayD<f32>>,

	required_data: Vec<bool>,
	required_passes: Vec<bool>,
	pass_order: Vec<PassID>,

	supplied_inputs: Vec<DataID>,
	requested_outputs: Vec<DataID>,
}

impl Graph {
	fn new(builder: &Builder, inputs: &[DataID], requested_outputs: &[DataID]) -> Result<Graph> {
		let n_passes = builder.operations.len()*2;
		let n_data = builder.nodes.len()*2;


		let is_input = find_inputs(n_data, inputs, &builder.static_inputs);

		let (pass_inputs, pass_outputs, data_inputs, data_outputs) = find_dependencies(builder);
		
		// Find the minimum set of operations and nodes required to calculate the `required_outputs`
		let (required_data, required_passes) = find_required(n_passes, n_data, &is_input, requested_outputs, &pass_inputs, &data_inputs);

		// Find the order of operations
		let pass_order = find_pass_order(n_passes, n_data, &is_input, &required_data, &required_passes, &pass_inputs, &pass_outputs, &data_inputs, &data_outputs)?;

		let filtered_static_inputs = builder.static_inputs.iter()
			.filter(|&(k, v)| !inputs.contains(k))
			.map(|(k, v)| (k.clone(), v.clone())).collect();

		let graph = Graph{
			pass_inputs: pass_inputs,
			pass_outputs: pass_outputs,
			data_inputs: data_inputs,
			data_outputs: data_outputs,

			nodes: builder.nodes.clone(),
			node_shapes: builder.node_shapes.clone(),
			node_names: builder.node_names.clone(),

			operations: builder.operations.clone(),
			static_inputs: filtered_static_inputs,

			required_data: required_data,
			required_passes: required_passes,
			pass_order: pass_order,

			supplied_inputs: inputs.to_vec(),
			requested_outputs: requested_outputs.to_vec(),
		};

		Ok(graph)
	}

	pub fn execute(&mut self, input: Vec<ArrayD<f32>>, parameters: Vec<ArrayD<f32>>) -> Storage{
		unimplemented!()
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

fn find_dependencies(builder: &Builder) -> (Vec<Vec<DataID>>, Vec<Vec<DataID>>, Vec<Vec<PassID>>, Vec<Vec<PassID>>) {
	let mut pass_inputs: Vec<Vec<DataID>> = (0..builder.operations.len()*2).map(|_| vec![]).collect();
	let mut pass_outputs: Vec<Vec<DataID>> = (0..builder.operations.len()*2).map(|_| vec![]).collect();

	for op_id in (0..builder.operations.len()).map(|i| OpID{index:i}) {
		let operation = &*builder.operations[op_id.index];

		let forward_id = op_id.forward_id();
		let (forward_inputs, forward_outputs) = operation.forward_dependencies();
		pass_inputs[forward_id.index] = forward_inputs;
		pass_outputs[forward_id.index] = forward_outputs;

		let backward_id = op_id.backward_id();
		let (backward_inputs, backward_outputs) = operation.backward_dependencies();
		pass_inputs[backward_id.index] = backward_inputs;
		pass_outputs[backward_id.index] = backward_outputs;
	}

	let mut data_inputs: Vec<Vec<PassID>> = (0..builder.nodes.len()*2).map(|_| vec![]).collect();
	let mut data_outputs: Vec<Vec<PassID>> = (0..builder.nodes.len()*2).map(|_| vec![]).collect();

	for pass_id in (0..builder.operations.len()*2).map(|i| PassID{index:i}) {
		for data_id in &pass_inputs[pass_id.index] {
			data_outputs[data_id.index].push(pass_id.clone());
		}
		for data_id in &pass_outputs[pass_id.index] {
			data_inputs[data_id.index].push(pass_id.clone())
		}
	}

	(pass_inputs, pass_outputs, data_inputs, data_outputs)
}

/// Work backwards from the requested output data marking data and operation passes as required.
fn find_required(n_passes: usize, n_data: usize, is_input: &[bool], requested_outputs: &[DataID], pass_inputs: &[Vec<DataID>], data_inputs: &[Vec<PassID>]) -> (Vec<bool>, Vec<bool>){
	assert_eq!(n_passes, pass_inputs.len());
	assert_eq!(n_data, data_inputs.len());
	assert_eq!(n_data, is_input.len());
	
	let mut required_data: Vec<bool> = vec![false; n_data];
	let mut required_passes: Vec<bool> = vec![false; n_passes];
	{
		// queues contain locations which should be visited and marked required
		// if a node is already marked when removed from the queue then nothing happens, otherwise it is marked and its dependencies get added to the queue
		let mut pass_queue: VecDeque<PassID> = VecDeque::new();
		let mut data_queue: VecDeque<DataID> = VecDeque::new(); 

		// start with the data requested by the user
		for data_id in requested_outputs {
			data_queue.push_back(data_id.clone());
		}

		// This is robust to circular graphs, as locations already marked as required will be passed over if visited a second time
		while !(pass_queue.is_empty() && data_queue.is_empty()) {

			if let Some(data_id) = data_queue.pop_front() {
				if !required_data[data_id.index] {
					required_data[data_id.index] = true;
					if !is_input[data_id.index]{
						for pass_id in &data_inputs[data_id.index] {
							pass_queue.push_back(pass_id.clone());
						}
					}
				}
			}

			if let Some(pass_id) = pass_queue.pop_front() {
				if !required_passes[pass_id.index] {
					required_passes[pass_id.index] == true;
					for data_id in &pass_inputs[pass_id.index] {
						data_queue.push_back(data_id.clone());
					}
				}
			}
		}
	}
	(required_data, required_passes)
}

/// returns the order in which passes should be called such that dependencies are respected.
/// By default this will order passes in the otder that they were added to the graph, and only perform the minimal rearragement required to ensure dependencies are met.
/// out of order dependeancies can cause quadratic slow down (this can probably be removed using priority queues)
fn find_pass_order(n_passes: usize, n_data: usize, is_input: &[bool], required_data: &[bool], required_passes: &[bool], pass_inputs: &[Vec<DataID>], pass_outputs: &[Vec<DataID>], data_inputs: &[Vec<PassID>], data_outputs: &[Vec<PassID>]) -> Result<Vec<PassID>>{
	assert_eq!(n_passes, pass_inputs.len());
	assert_eq!(n_passes, pass_outputs.len());
	assert_eq!(n_passes, required_passes.len());

	assert_eq!(n_data, data_inputs.len());
	assert_eq!(n_data, data_outputs.len());
	assert_eq!(n_data, is_input.len());


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

	// Attempts to retire a pass as Ready or Unavailable, return true if sucessful false otherwise
	// If it returns true this method should never be called again for that pass_id.
	fn try_retire_pass(pass_id: &PassID, pass_order: &mut Vec<PassID>, data_ready: &mut[DataState], passes_ready: &mut [PassState], data_outputs: &[Vec<PassID>], pass_outputs: &[Vec<DataID>]) -> bool{
		if matches!(passes_ready[pass_id.index] , PassState::Ready | PassState:: Unavailable) {
			panic!("pass has already been retired, try_retire_pass() should not be called")
		} else if matches!(passes_ready[pass_id.index] , PassState::Pending(0)) {
			// add to pass order and update output data locations readiness
			pass_order.push(pass_id.clone());
			passes_ready[pass_id.index] = PassState::Ready;
			for data_id in &pass_outputs[pass_id.index] {
				// If a data outptu of a pass is pending decrement
				// If that data output can now be marked ready
				match data_ready[data_id.index] {
					DataState::Unavailable => {},
					DataState::Pending(rem) if rem == 1 => {
						mark_data_ready(data_id, data_ready, passes_ready, data_outputs)
					},
					DataState::Pending(rem) if rem > 1 => {data_ready[data_id.index] = DataState::Pending(rem - 1)},
					DataState::Pending(_) => panic!("Data with zero inputs should have already been marked Unavailable or Ready"),
					DataState::Ready => panic!("data marked ready before last input pass was processed."),
				}
			}
			true
		} else if matches!(passes_ready[pass_id.index] , PassState::PendingUnavailable) {
			passes_ready[pass_id.index] = PassState::Unavailable;
			for data_id in &pass_outputs[pass_id.index] {
				mark_data_unavailable(data_id, data_ready, passes_ready, data_outputs)
			}
			true
		} else {
			false
		}
	}

	/// Marks data as ready, and decreases pending count of dependent passes
	/// Only legal to call this if is_input[]==true or as the last input pass is retired
	fn mark_data_ready(data_id: &DataID, data_ready: &mut[DataState], passes_ready: &mut [PassState], data_outputs: &[Vec<PassID>]){
		//debug_assert!(is_input[data_id.index] || matches!(data_ready[data_id.index], State::Pending(rem) if rem == 1));
		data_ready[data_id.index] = DataState::Ready;
		for pass_id in &data_outputs[data_id.index] {
			match passes_ready[pass_id.index] {
				PassState::Pending(rem) if rem > 0 => {passes_ready[pass_id.index] = PassState::Pending(rem - 1)},
				PassState::Unavailable | PassState::PendingUnavailable =>{},
				PassState::Pending(_) | PassState::Ready => panic!("Something has happened out of order"),
			}
		}
	}

	/// Can be called on data in any state, but will only mark data and dependent passes as unavailable if the current data state is Pending
	fn mark_data_unavailable(data_id: &DataID, data_ready: &mut[DataState], passes_ready: &mut [PassState], data_outputs: &[Vec<PassID>]){
		if matches!(data_ready[data_id.index], DataState::Ready | DataState::Unavailable){return} 
		data_ready[data_id.index] = DataState::Unavailable;
		for pass_id in &data_outputs[data_id.index] {
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
	let mut passes_ready: Vec<PassState> = (0..n_passes).map(|i| PassState::Pending(pass_inputs[i].len())).collect();
	let mut data_ready: Vec<DataState> = (0..n_data).map(|i| DataState::Pending(data_inputs[i].len())).collect();
	for (i, &is_input) in is_input.iter().enumerate() {
		if is_input {
			mark_data_ready(&DataID{index: i}, &mut data_ready, &mut passes_ready, data_outputs)
		} else if data_inputs[i].len() == 0 {
			mark_data_unavailable(&DataID{index: i}, &mut data_ready, &mut passes_ready, data_outputs)
		}
	}




	// iterate over all required passes,
	// add to pass order where possible (inputs are ready), otherwise add to deferred queue
	// the resulting pass order should be as close to the users order while still not being out of order
	let forward_required_passes = (0..n_passes).map(|i| PassID{index:i}).filter(|id| id.is_forward() && required_passes[id.index]);
	let backward_required_passes = (0..n_passes).map(|i| PassID{index:i}).filter(|id| id.is_backward() && required_passes[id.index]);
	let default_pass_order = forward_required_passes.chain(backward_required_passes.rev());
	for pass_id in default_pass_order {

		let success = try_retire_pass(&pass_id, &mut pass_order, &mut data_ready, &mut passes_ready, &data_outputs, &pass_outputs);
		if !success {
			deferred_passes.push_back(pass_id.clone());
			continue;
		}

		// Attempt to empty deferred queue
		// always try to add deferred passes in order
		let mut i = 0;
		while i < deferred_passes.len(){
			let success = try_retire_pass(&deferred_passes[i], &mut pass_order, &mut data_ready, &mut passes_ready, &data_outputs, &pass_outputs);
			if success {
				deferred_passes.remove(i);
				i = 0; // keep trying from the start again
			} else {
				i += 1;
			}
		}
	}
	
	if (0..n_data).filter(|&i| required_data[i]).any(|i| matches!(data_ready[i], DataState::Unavailable)) {
		bail!(ErrorKind::InputsInsufficientForRequestedOutputs)
	}

	if deferred_passes.len() > 0 {
		bail!(ErrorKind::GraphContainsCircularDependencies)
	}

	Ok(pass_order)
}



fn find_shapes(graph: &Graph, passes: &[PassID], inputs: &[DataID], input_data: &[ArrayD<f32>], static_inputs: &OrderMap<DataID, ArrayD<f32>>) -> Result<Vec<IxDyn>> {
	// if inputs are present along with static_inputs the inputs should add

	let mut shapes = GraphShapes::new(graph);

	// for all inputs, merge data shape into existing graph shape
	ensure!(inputs.len() == input_data.len(), ErrorKind::InputSizeError);
	for (input, input_data) in inputs.iter().zip(input_data) {
		shapes.merge_input(&input, input_data.shape());
	}

	// for all static inputs, if not in inputs, merge into graph shape
	// because static_inputs can be broadcast, resolving the dimension will be harder
	// iterate from the lowest dimension up, if the static_input dimension is not 1 then enforce it in the shape
	for (static_input, static_input_data) in static_inputs.iter() {
		if !inputs.contains(static_input) {
			shapes.merge_static_input(&static_input, static_input_data.shape());
		}
	}

	// for all operation forward passes that are scheduled. call the relevant shape propagation
	let op_ids = passes.iter().filter(|pass| pass.is_forward()).map(|pass_id| pass_id.op_id());
	for op_id in op_ids {
		graph.operations[op_id.index].propagate_shape_constraints(&mut shapes);
	}

	shapes.shapes.iter().map(|shape| shape.to_data_shape().map_err(|e| e.into())).collect()
}












enum DataState<T>{
	Unallocated,
	UnallocatedInput(usize), //index in input_data
	UnallocatedStaticInput,
	Allocated(T),
	Deallocated,
}

/// A structure which allows for runtime checked borrowing, similar to a RefCell for a Collection of Arrays,
/// but with some limitations.
/// Each element can only be borrowed either once mutably or many times immutably, however,
/// once borrowed as such it is stuck until DataBorrow is dropped, or by calling reset_all().
pub struct Storage<'a> {
	shapes: Vec<IxDyn>,
	input_data: &'a [ArrayD<f32>],
	data: Vec<DataState<ArrayD<f32>>>,
	borrow_flags: Vec<Cell<usize>>,
	graph: &'a Graph,
	next_pass: Option<PassID>
}

const UNUSED: usize = 0;
const WRITING: usize = !0;
impl<'a> Storage<'a> {

	pub fn new(input_data: &'a [ArrayD<f32>], shapes: Vec<IxDyn>, graph: &'a Graph) -> Storage<'a> {
		let num_nodes = shapes.len();

		let mut data: Vec<DataState<ArrayD<f32>>> = (0..num_nodes * 2).map(|_i| DataState::Unallocated).collect();

		for (i, data_id) in graph.supplied_inputs.iter().enumerate() {
			debug_assert!(shapes[data_id.node_id().index].slice() == input_data[data_id.index].shape());
			data[data_id.index] = DataState::UnallocatedInput(i);
		}
		
		for (data_id, _data) in graph.static_inputs.iter() {
			data[data_id.index] = DataState::UnallocatedStaticInput;
		}

		Storage{
			shapes: shapes,
			input_data: input_data,
			data: data,
			borrow_flags: vec![Cell::new(UNUSED); num_nodes * 2],
			graph: graph,
			next_pass: None,
		}
	}

	/// If this value is set, all subsequent accesses will be checked against the dependency list for the Pass
	/// This can be useful to ensure that passes dont access anything they shouldn't.
	fn set_next_pass_debug(&mut self, pass_id: &PassID){
		self.next_pass = Some(pass_id.clone());
	}

	/// Deallocates the data specified by DataID
	fn deallocate(&mut self, id: DataID){
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
				*ptr = DataState::Allocated(ArrayD::zeros(self.shapes[id.node_id().index].clone()));
			},
			DataState::UnallocatedInput(ind) =>{
				*ptr = DataState::Allocated(self.input_data[ind].clone())
			},
			DataState::UnallocatedStaticInput => {
				let shape = self.shapes[id.node_id().index].clone();
				if let Some(ref static_data) = self.graph.static_inputs.get(id){
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

	/// Immutably borrows data element associated with the given ID
	/// Will panic if data element is already borrowed mutably
	pub fn get<'b>(&'b self, id: DataID) -> Result<ArrayViewD<f32>> {
		if let Some(ref pass_id) = self.next_pass {
			ensure!(self.graph.pass_inputs[pass_id.index].contains(&id)||self.graph.pass_outputs[pass_id.index].contains(&id), ErrorKind::PassAttemptedToAccessDataNotListedAsInputOrOutput);
		}

		if self.borrow_flags[id.index].get() != WRITING {
				let ptr = unsafe{self.get_or_init(&id)?};
				self.borrow_flags[id.index].set(self.borrow_flags[id.index].get() + 1);
				let array: &'b ArrayD<f32> = unsafe{&*ptr};
				Ok(array.view())
		} else {
			bail!(ErrorKind::OperationAccessedAMutablyBorrowedNode)
		}
	}

	/// Mutably borrows data element associated with the given ID
	/// Will panic if data element is already mutably or immutably borrowed 
	/// The borrow will stick until
	pub fn get_mut<'b>(&'b self, id: DataID) -> Result<ArrayViewMutD<f32>> {
		if let Some(ref pass_id) = self.next_pass {
			ensure!(self.graph.pass_outputs[pass_id.index].contains(&id), ErrorKind::PassAttemptedToMutablyAccessDataNotListedAsOutput);
		}
		match self.borrow_flags[id.index].get() {
			UNUSED => {
				let ptr = unsafe{self.get_or_init(&id)?};
				self.borrow_flags[id.index].set(WRITING);
				let array: &'b mut ArrayD<f32> = unsafe{&mut *ptr};
				Ok(array.view_mut())
			},
			_ => bail!(ErrorKind::OperationMutablyAccessedABorrowedNode),
		}
	}
}




























pub struct GraphShapes{
	shapes: Vec<NodeShape>,
}

impl GraphShapes {
	pub fn new(graph: &Graph) -> GraphShapes {
		GraphShapes{
			shapes: graph.node_shapes.clone(),
		}
	}

	fn merge_input(&mut self, data_id: &DataID, shape: &[Ix]) -> Result<()>{
		unimplemented!()
	}

	fn merge_static_input(&mut self, data_id: &DataID, shape: &[Ix]) -> Result<()> {
		unimplemented!()
	}

	pub fn merge_with(&mut self, id: NodeID, shape: NodeShape) -> Result<()>{
		self.shapes[id.index] = self.shapes[id.index].merge(&shape)?;
		Ok(())
	}
}






#[test]
fn test_builders(){
	let g = graph::Config();

	let node1 = g.new_node("node1", shape![None, 5, 16], tag!["input"]);
	let node2 = g.new_node("node2", shape![None, 5, 16], tag![]);

	for i in 3..10 {
		let node2 = g.new_node(&format!("node{}", i), shape![None, 5, 16], tag![i]);
	}
}