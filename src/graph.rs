
use ndarray::ArrayD;
use ndarray::ArrayBase;
use ndarray::prelude::*;
use ndarray::Ix;
use shape;
use shape::{NodeShape, NodeDim};
use init::Initialiser;
use std::collections::VecDeque;
use ordermap::{OrderMap, OrderSet};
use ops::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use id::*;
use storage::Storage;

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
		}
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


lazy_static! {
	static ref NODE_COUNT: AtomicUsize = {AtomicUsize::new(0)};
	static ref OP_COUNT: AtomicUsize = {AtomicUsize::new(0)};
	static ref PASS_COUNT: AtomicUsize = {AtomicUsize::new(0)};
}

/// Used to construct the definition of the computational hypergraph.
/// This cannot be executed, an executable `Subgraph` can be built by calling `subgraph()`
#[derive(Clone, Debug)]
pub struct GraphDef {
	// The key information
	node_ids: Vec<NodeID>,
	op_ids: Vec<OpID>,
	pass_ids: Vec<PassID>,
	
	// Extra information pertaining to nodes
	static_inputs: OrderMap<DataID, ArrayD<f32>>,
	initialisers: OrderMap<NodeID, Initialiser>,

	// These are used to quickly look op names and tags
	// Just duplicates data from node_ids/op_ids
	node_names: OrderMap<String, NodeID>,
	node_tags: OrderMap<NodeTag, OrderSet<NodeID>>,
	op_names: OrderMap<String, OpID>,
	op_tags: OrderMap<OpTag, OrderSet<OpID>>,

	// Used to track potentially recursive build calls inside add_op()
	// and ensure each Initialiser added in set_initialiser gets tagged with the Op that created it
	// Should normally be empty, except when add_op is called.
	in_flight_op_builds: Vec<usize>,
	deferred_initialisers: Vec<(usize, (NodeID, Initialiser))>,
}

impl GraphDef {
	
	pub fn new() -> GraphDef {
		GraphDef {
			node_ids: Vec::new(),
			op_ids: Vec::new(),
			pass_ids: Vec::new(),
			

			static_inputs: OrderMap::new(),
			initialisers: OrderMap::new(),

			node_names: OrderMap::new(),
			node_tags: OrderMap::new(),
			op_names: OrderMap::new(),
			op_tags: OrderMap::new(),

			in_flight_op_builds: Vec::new(),
			deferred_initialisers: Vec::new(),
		}
	}

	/// Returns a usize guarenteed to be unique amongst nodes in the graph
	fn next_node_id(&self) -> usize {
		NODE_COUNT.fetch_add(1, Ordering::Relaxed)
	}

	/// Returns a usize guarenteed to be unique amongst ops in the graph
	fn next_op_id(&self) -> usize {
		OP_COUNT.fetch_add(1, Ordering::Relaxed)
	}

	/// Returns a usize guarenteed to be unique amongst passes in the graph
	fn next_pass_id(&self) -> usize {
		PASS_COUNT.fetch_add(1, Ordering::Relaxed)
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
		let input_ids: Vec<NodeID> = self.get_nodes().iter().filter(|node_id| !self.static_inputs.contains_key(&node_id.value_id()) && dependencies.data_inputs(&node_id.value_id()).len() == 0 && !node_id.tags().contains(&NodeTag::Parameter)).cloned().collect();
		let parameter_ids: Vec<NodeID> = self.get_nodes().iter().filter(|node_id| dependencies.data_inputs(&node_id.value_id()).len() == 0 && node_id.tags().contains(&NodeTag::Parameter)).cloned().collect();
		
		self.subgraph(
			&input_ids.iter().chain(&parameter_ids).map(|node_id| node_id.value_id()).collect::<Vec<_>>(),
			&parameter_ids.iter().map(|node_id| node_id.value_id()).chain(parameter_ids.iter().map(|node_id| node_id.gradient_id())).collect::<Vec<_>>()
		)
	}

	/// Node values are initialised to be zero filled by default.
	/// The ArrayD value supplied to this method will be used to set the initial value of the node, and this data must be able to broadcast to this node.
	/// This can be used to supply fixed inputs to Ops in place of parameters
	pub fn set_static_input(&mut self, id: DataID, value: ArrayD<f32>){
		self.static_inputs.insert(id, value);
	}

	pub fn clear_static_input(&mut self, id: DataID){
		self.static_inputs.remove(&id);
	}

	pub fn set_initialiser(&mut self, node_id: &NodeID, init: Initialiser) {
		if self.in_flight_op_builds.len() == 0 {
			self.initialisers.insert(node_id.clone(), init);
		} else {
			self.deferred_initialisers.push((self.in_flight_op_builds[self.in_flight_op_builds.len() - 1] , (node_id.clone(), init)))
		}
	}

	pub fn clear_initialiser(&mut self, node_id: &NodeID) {
		self.initialisers.remove(node_id);
	}

	/// Creates values for the requested nodes according to the initialisers of each node.
	///
	/// This should only be called on nodes with a fully known shape.
	pub fn initialise_nodes(&self, nodes: &[NodeID]) -> Result<Vec<ArrayD<f32>>>{
		let mut vec = Vec::with_capacity(nodes.len());
		for node in nodes {
			let shape = node.shape().to_data_shape()?;
			let mut arr = ArrayD::zeros(shape);
			if let Some(initialiser) = self.initialisers.get(node) {
				let op_id = initialiser.op_id();
				let op = op_id.as_ref().map(|id| id.instance());
				initialiser.call(arr.view_mut(), op);
			}
			vec.push(arr);
		}
		Ok(vec)
	}

	fn new_node_checks(&self, name: &str, tags: &[NodeTag], shape: &NodeShape) -> Result<()> {
		// ensure names are unique w.r.t other names and tags
		ensure!(!self.node_names.contains_key(name), ErrorKind::NodeNameConflict(name.to_string()));
		ensure!(!self.node_tags.contains_key(&NodeTag::from(name)), ErrorKind::NodeTagNameConflict(name.to_string()));

		for tag in tags{
			// ensure that tags don't overlap with existing names
			match tag {
				&NodeTag::Str(ref tag_str) => {
					ensure!(&name != tag_str, ErrorKind::NodeTagNameConflict(tag_str.to_string()));
					ensure!(!self.node_names.contains_key(tag_str), ErrorKind::NodeTagNameConflict(tag_str.to_string()));
				},
				_ => {},
			}
		}

		for tag in tags{
			if matches!(tag, &NodeTag::Parameter){
				ensure!(shape.is_known(), ErrorKind::ParameterNodesMustHaveKnownSize(name.to_string(), shape.clone()));
			}
		}
		Ok(())
	}

	/// Create a new node in the graph
	pub fn new_node<I: Into<String>>(&mut self, shape: NodeShape, name: I, tags: Vec<NodeTag>) -> Result<NodeID>{
		
		let name = name.into();

		self.new_node_checks(&name, &tags, &shape)?;

		// all good, so add node
		let node_id = NodeID::new(self.next_node_id(), name.clone(), shape, tags.iter().cloned().collect());
		self.node_ids.push(node_id.clone());
		
		// update lookup maps
		self.node_names.insert(name, node_id.clone());
		for tag in tags{
			match tag {
				NodeTag::Id(_) => {},
				NodeTag::Int(_) | NodeTag::Str(_) | NodeTag::Parameter => {
					self.node_tags.entry(tag).or_insert_with(OrderSet::new).insert(node_id.clone());
				},
			}
		}

		Ok(node_id)
	}

	pub fn new_op<O: Op>(&mut self, op: O, tags: Vec<OpTag>) -> Result<OpID> {
		let next_id = self.next_op_id();

		self.in_flight_op_builds.push(next_id);

		let result = self.new_op_impl(op, tags, next_id);

		self.in_flight_op_builds.pop().unwrap();

		if let Ok(ref op_id) = result {
			let initialisers = &mut self.initialisers;
			self.deferred_initialisers.retain(|&(id, (ref node, ref init))|{
				if id == next_id {
					initialisers.insert(node.clone(), init.clone().set_op_id(op_id.clone()));
					false
				} else {
					true
				}
			});
		} else {
			self.deferred_initialisers.retain(|&(id, (ref _node, ref _init))| id != next_id );
		}

		result
	}

	fn new_op_impl<O: Op>(&mut self, op: O, tags: Vec<OpTag>, next_id: usize) -> Result<OpID> {
		
		
		let op = op.build(self)?;
		
		let name = op.name().to_string();

		// ensure names are unique w.r.t other names and tags
		ensure!(!self.op_names.contains_key(&name), ErrorKind::OpNameConflict(name));
		ensure!(!self.op_tags.contains_key(&OpTag::from(name.as_str())), ErrorKind::OpTagNameConflict(name));
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
		let op_id = OpID::new(next_id, op, tags.iter().cloned().collect());
		self.op_ids.push(op_id.clone());

		// update lookup maps
		self.op_names.insert(name, op_id.clone());
		for tag in tags{
			match tag {
				OpTag::Id(_) => {},
				OpTag::Int(_) => {
					self.op_tags.entry(tag).or_insert_with(OrderSet::new).insert(op_id.clone());
				},
				OpTag::Str(_) => {
					self.op_tags.entry(tag).or_insert_with(OrderSet::new).insert(op_id.clone());
				},
			}
		}

		Ok(op_id)
	}

	pub fn add_pass<P: Pass>(&mut self, pass: P) -> PassID {
		let pass_id = PassID::new(self.next_pass_id(), pass);
		self.pass_ids.push(pass_id.clone());
		pass_id
	}

	/// Create a node which acts as a subview of another node.
	/// Contiguous views will be free from time and memory overhead, recording just a view.
	/// Non-contigues views will incurr a memory and time overhead during runtime.
	/// Returns NodeID of the new node.
	// pub fn new_read_view<I: Into<String>>(&mut self, _name: I, _shape: NodeShape, _tags: Vec<NodeTag>) -> Result<NodeID>{
	// 	unimplemented!()
	// }

	// pub fn new_write_view<I: Into<String>>(&mut self, _name: I, _shape: NodeShape, _tags: Vec<NodeTag>) -> Result<NodeID>{
	// 	unimplemented!()
	// }

	pub fn get_nodes(&self) -> &[NodeID] {
		&self.node_ids
	}

	pub fn get_ops(&self) -> &[OpID] {
		&self.op_ids
	}

	pub fn get_passes(&self) -> &[PassID] {
		&self.pass_ids
	}

	pub fn parameter_ids<'a>(&'a self) -> Vec<NodeID> {
		self.node_ids(NodeTag::Parameter)
	}

	/// Returns the NodeID which was tagged with 'tag'.
	/// 
	/// #Panics
	/// If multiple or zero nodes are associated with the tag
	pub fn node_id<T: Into<NodeTag>>(&self, tag: T) -> NodeID {
		let tag = tag.into();
		let mut ids = self.node_ids(tag.clone());
		if ids.len() > 1 {
			panic!("Multiple nodes match tag: {}", tag);
		} else if ids.len() < 1 {
			panic!("Zero nodes match tag: {}", tag);
		} else {
			ids.remove(0)
		}
	}

	pub fn node_ids<'a, T: Into<NodeTag>>(&'a self, tag: T) -> Vec<NodeID> {
		let tag = tag.into();
		match tag {
			NodeTag::Id(id) => vec![id.clone()],
			NodeTag::Str(ref string) => {
				// Check names first, then other string tags
				if let Some(node_id) = self.node_names.get(string) {
					vec![node_id.clone()]
				} else {
					match self.node_tags.get(&tag){
						Some(set) => set.iter().cloned().collect(),
						None => Vec::new(),
					}
				}
			},
			NodeTag::Int(_) | NodeTag::Parameter  => {
				match self.node_tags.get(&tag){
					Some(set) => set.iter().cloned().collect(),
					None => Vec::new(),
				}
			}
		}
	}

	/// Returns a single OpID matching a op tag
	/// 
	/// #Panics
	/// If multiple or zero ops are associated with the tag.
	pub fn op_id<T: Into<OpTag>>(&self, tag: T) -> OpID {
		let tag = tag.into();
		let mut ids = self.op_ids(tag.clone());
		if ids.len() > 1 {
			panic!("Multiple ops match tag: {}", tag);
		} else if ids.len() < 1 {
			panic!("Zero ops match tag: {}", tag);
		} else {
			ids.remove(0)
		}
	}

	pub fn op_ids<'a, T: Into<OpTag>>(&'a self, tag: T) -> Vec<OpID> {
		let tag = tag.into();
		match tag {
			OpTag::Id(id) => vec![id],
			OpTag::Str(ref string) => {
				if let Some(op_id) = self.op_names.get(string) {
					vec![op_id.clone()]
				} else {
					match self.op_tags.get(&tag){
						Some(set) => set.iter().cloned().collect(),
						None => Vec::new(),
					}
				}
			},
			OpTag::Int(_) => {
				match self.op_tags.get(&tag){
					Some(set) => set.iter().cloned().collect(),
					None => Vec::new(),
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
	/// This number is independent of the number of ops
	pub fn num_passes(&self) -> usize{
		self.pass_ids.len()
	}
}

/// Denormalised data about dependencies
#[derive(Clone, Debug)]
pub struct Dependencies {
	pass_inputs: OrderMap<PassID, OrderSet<DataID>>,
	pass_outputs: OrderMap<PassID, OrderSet<DataID>>,
	pass_is_forward: OrderMap<PassID, bool>,

	data_inputs: OrderMap<DataID, OrderSet<PassID>>,
	data_outputs: OrderMap<DataID, OrderSet<PassID>>,

	op_inputs: OrderMap<OpID, OrderSet<NodeID>>,
	op_outputs: OrderMap<OpID, OrderSet<NodeID>>,
	op_shape_outputs: OrderMap<OpID, OrderSet<NodeID>>,

	node_inputs: OrderMap<NodeID, OrderSet<OpID>>,
	node_shape_inputs: OrderMap<NodeID, OrderSet<OpID>>,
	node_outputs: OrderMap<NodeID, OrderSet<OpID>>,
}

impl Dependencies {
	pub fn new(graph: &GraphDef) -> Dependencies {

		let mut data_inputs = OrderMap::new();
		let mut data_outputs = OrderMap::new();

		let mut node_inputs = OrderMap::new();
		let mut node_shape_inputs = OrderMap::new();
		let mut node_outputs = OrderMap::new();

		for node_id in graph.get_nodes() {
			data_inputs.insert(node_id.value_id(), OrderSet::new());
			data_inputs.insert(node_id.gradient_id(), OrderSet::new());
			data_outputs.insert(node_id.value_id(), OrderSet::new());
			data_outputs.insert(node_id.gradient_id(), OrderSet::new());

			node_inputs.insert(node_id.clone(), OrderSet::new());
			node_shape_inputs.insert(node_id.clone(), OrderSet::new());
			node_outputs.insert(node_id.clone(), OrderSet::new());
		}

		let mut pass_inputs: OrderMap<PassID, OrderSet<DataID>> = OrderMap::new();
		let mut pass_outputs: OrderMap<PassID, OrderSet<DataID>> = OrderMap::new();
		let mut pass_is_forward = OrderMap::new();

		for pass_id in graph.get_passes() {
			let (inputs, outputs) = pass_id.instance().dependencies();

			for data_id in &inputs {
				data_outputs.get_mut(data_id).unwrap().insert(pass_id.clone());
			}
			for data_id in &outputs {
				data_inputs.get_mut(data_id).unwrap().insert(pass_id.clone());
			}

			let is_forward = inputs.iter().chain(outputs.iter()).all(|data_id| data_id.is_value());
			pass_inputs.insert(pass_id.clone(), inputs.into_iter().collect());
			pass_outputs.insert(pass_id.clone(), outputs.into_iter().collect());
			pass_is_forward.insert(pass_id.clone(), is_forward);
		}


		let mut op_inputs: OrderMap<OpID, OrderSet<NodeID>> = OrderMap::new();
		let mut op_outputs: OrderMap<OpID, OrderSet<NodeID>> = OrderMap::new();
		let mut op_shape_outputs: OrderMap<OpID, OrderSet<NodeID>> = OrderMap::new();

		for op_id in graph.get_ops() {
			let (inputs, outputs) = op_id.instance().dependencies();
			let mut shape_outputs = op_id.instance().inner_nodes();
			shape_outputs.extend_from_slice(&outputs);

			for node_id in &inputs {
				node_outputs.get_mut(node_id).unwrap().insert(op_id.clone());
			}
			for node_id in &outputs {
				node_inputs.get_mut(node_id).unwrap().insert(op_id.clone());
			}
			for node_id in &shape_outputs {
				node_shape_inputs.get_mut(node_id).unwrap().insert(op_id.clone());
			}

			op_inputs.insert(op_id.clone(), inputs.into_iter().collect());
			op_outputs.insert(op_id.clone(), outputs.into_iter().collect());
			op_shape_outputs.insert(op_id.clone(), shape_outputs.into_iter().collect());
		}

		Dependencies{pass_inputs, pass_outputs, pass_is_forward, data_inputs, data_outputs, op_inputs, op_outputs, op_shape_outputs, node_inputs, node_shape_inputs, node_outputs}
	}

	pub fn contains_data(&self, data_id: &DataID) -> bool {
		self.data_inputs.get(data_id).is_some()
	}

	pub fn contains_pass(&self, pass_id: &PassID) -> bool {
		self.pass_inputs.get(pass_id).is_some()
	}
	
	pub fn contains_node(&self, node_id: &NodeID) -> bool {
		self.node_inputs.get(node_id).is_some()
	}

	pub fn contains_op(&self, op_id: &OpID) -> bool {
		self.op_inputs.get(op_id).is_some()
	}

	pub fn data_inputs(&self, data_id: &DataID) -> &OrderSet<PassID> {
		self.data_inputs.get(data_id).unwrap()
	}

	pub fn data_outputs(&self, data_id: &DataID) -> &OrderSet<PassID> {
		self.data_outputs.get(data_id).unwrap()
	}

	pub fn pass_inputs(&self, pass_id: &PassID) -> &OrderSet<DataID> {
		self.pass_inputs.get(pass_id).unwrap()
	}
	
	pub fn pass_is_forward(&self, pass_id: &PassID) -> bool {
		*self.pass_is_forward.get(pass_id).unwrap()
	}

	pub fn pass_outputs(&self, pass_id: &PassID) -> &OrderSet<DataID> {
		self.pass_outputs.get(pass_id).unwrap()
	}

	pub fn node_inputs(&self, node_id: &NodeID) -> &OrderSet<OpID> {
		&self.node_inputs.get(node_id).unwrap()
	}

	pub fn node_shape_inputs(&self, node_id: &NodeID) -> &OrderSet<OpID> {
		&self.node_shape_inputs.get(node_id).unwrap()
	}

	pub fn node_outputs(&self, node_id: &NodeID) -> &OrderSet<OpID> {
		&self.node_outputs.get(node_id).unwrap()
	}

	/// Inputs of the Op
	pub fn op_inputs(&self, op_id: &OpID) -> &OrderSet<NodeID> {
		&self.op_inputs.get(op_id).unwrap()
	}
	
	/// Outputs of the Op, excluding inner nodes
	pub fn op_outputs(&self, op_id: &OpID) -> &OrderSet<NodeID> {
		&self.op_outputs.get(op_id).unwrap()
	}

	/// Outputs of the Op, including inner nodes
	pub fn op_shape_outputs(&self, op_id: &OpID) -> &OrderSet<NodeID> {
		&self.op_shape_outputs.get(op_id).unwrap()
	}
}


// enum to record the compute status of each data
#[derive(Clone, Debug)]
pub (crate) enum DataStatus {
	Input,
	Compute,
}

/// enum to record the shape status of each node
#[derive(Clone, Debug)]
enum NodeStatus {
	InputOrKnown,
	StaticInput,
	Infer,
}

#[derive(Clone, Debug)]
pub struct Subgraph {
	//graph: GraphDef,
	dependencies: Dependencies,

	subgraph_inputs: Vec<DataID>,
	subgraph_outputs: Vec<DataID>,

	// Only keep static inputs that dont conflict with a subgraph input
	filtered_static_inputs: OrderMap<DataID, ArrayD<f32>>,

	// Ops and nodes included in the subgraph, used to perform shape inference
	included_nodes: OrderMap<NodeID, NodeStatus>,
	included_ops: OrderSet<OpID>,
	op_order: Vec<OpID>,
	shapes: OrderMap<NodeID, IxDyn>,

	// passes and data inclded in the subgraph, used to perform graph execution
	included_data: OrderMap<DataID, DataStatus>,
	included_passes: OrderSet<PassID>,
	pass_order: Vec<PassID>,
	passes_before_dealloc: OrderMap<DataID, usize>,

	// To what degree should ops drag in upstream ops
	strict_op_inclusion: bool,
}

impl Subgraph {
	/// An executable subgraph derived from a `GraphDef`.
	///
	/// todo
	fn new(graph: &GraphDef, inputs: &[DataID], outputs: &[DataID]) -> Result<Subgraph> {

		let input_set: OrderSet<_> = inputs.iter().cloned().collect();
		let output_set: OrderSet<_> = outputs.iter().cloned().collect();
		assert_eq!(inputs.len(), input_set.len(), "Inputs contains duplicates");
		assert_eq!(outputs.len(), output_set.len(), "Outputs contains duplicates");

		let dependencies = Dependencies::new(graph);

		assert!(inputs.iter().all(|id| dependencies.contains_data(id)), "Inputs contained DataIDs from another graph");
		assert!(outputs.iter().all(|id| dependencies.contains_data(id)), "Outputs contained DataIDs from another graph");

		let strict_op_inclusion = true;
		// Find the minimum set of data, passes, nodes and ops required to perform shape inference and calculate the `outputs` of the subgraph
		let (included_data, included_passes, included_nodes, included_ops) = find_included(&graph, inputs, &graph.static_inputs, outputs, &dependencies, strict_op_inclusion);

		let op_order = find_op_order(&included_nodes, &included_ops, &dependencies)?;
		let pass_order = find_pass_order(&included_data, &included_passes, &dependencies)?;

		// remove overlap between static_inpts and inputs
		let filtered_static_inputs = graph.static_inputs.iter()
			.filter(|&(k, _v)| !inputs.contains(k))
			.map(|(k, v)| (k.clone(), v.clone())).collect();

		// for each data_id count the number of passes deemed required which depend on it, then add 1 if it is a requested output
		let mut passes_before_dealloc: OrderMap<DataID, usize> = dependencies.data_outputs.iter().map(|(id, passes)| (id.clone(), passes.iter().filter(|&pass| included_passes.contains(pass)).count())).collect();
		for data_id in outputs {
			*passes_before_dealloc.get_mut(data_id).unwrap() += 1;
		}

		let graph = Subgraph{
			//graph: graph.clone(),
			dependencies: dependencies,

			filtered_static_inputs: filtered_static_inputs,

			included_nodes: included_nodes,
			included_ops: included_ops,
			op_order: op_order,
			shapes: OrderMap::new(),

			included_data: included_data,
			included_passes: included_passes,
			pass_order: pass_order,
			passes_before_dealloc: passes_before_dealloc,

			subgraph_inputs: inputs.to_vec(),
			subgraph_outputs: outputs.to_vec(),

			strict_op_inclusion: strict_op_inclusion,
		};

		Ok(graph)
	}

	/// Calling this executes the subgraph and returns a Storage which contains the outputs of the subgraph.
	///
	/// todo
	pub fn execute(&mut self, inputs: Vec<ArrayD<f32>>) -> Result<Storage>{
		ensure!(inputs.len() == self.subgraph_inputs.len(), "The number of inputs provided ({}) did not match the number of expected inputs ({})", inputs.len(), self.subgraph_inputs.len());

		let input_data: OrderMap<DataID, ArrayD<f32>> = self.subgraph_inputs.iter().cloned().zip(inputs).collect();

		// if shapes is empty, or doesnt match the new inputs, recalculate all shapes.
		if self.shapes.len() != self.included_nodes.len()
		|| input_data.iter().any(|(id, input_data)|{input_data.shape() != self.shapes.get(&id.node_id()).unwrap().slice()}) {
			self.shapes = find_shapes(&self, &self.op_order, &input_data, &self.filtered_static_inputs)?;
		}

		let mut storage = Storage::new(&self.included_data, &self.dependencies, &self.filtered_static_inputs, input_data, &self.shapes);

		let mut passes_before_dealloc = self.passes_before_dealloc.clone();

		for pass_id in &self.pass_order {
			storage.set_current_pass(Some(pass_id.clone()));
			let pass_data = pass_id.instance().run(&mut storage)?;
			storage.set_pass_data(pass_id, pass_data);

			for data_id in self.dependencies.pass_inputs.get(pass_id).unwrap() {
				let pbd = passes_before_dealloc.get_mut(data_id).unwrap();
				*pbd -= 1;
				if *pbd == 0 {
					storage.deallocate(data_id);
				}
			}
			
			storage = storage.clear_borrow_flags();
		}
		storage.set_current_pass(None);

		Ok(storage)
	}

	/// Determines the degree to which ops are marked as included for shape inference.
	/// 
	/// In strict mode (true) operations will be added recursively starting from all nodes associated with data included in the subgraph, until sufficient inputs/known shapes are found or an error is generated.
	/// In non-strict mode, only operations with inputs which are already included due to association with data will be included, which can result in poorly defined shapes.
	/// 
	/// Default: true
	pub fn strict_op_inclusion(&mut self, strict: bool) -> Result<()> {
		if self.strict_op_inclusion != strict {
			self.strict_op_inclusion = strict;
			self.op_order = find_op_order(&self.included_nodes, &self.included_ops, &self.dependencies)?;
		}
		Ok(())
	}

	/// Returns a slice containings all the inputs required to execute this subgraph.
	pub fn inputs(&self) -> &[DataID]{
		&self.subgraph_inputs
	}

	/// Returns a slice containing all the outputs which will be available after the subgraph is executed
	pub fn outputs(&self) -> &[DataID]{
		&self.subgraph_outputs
	}
}


/// Work backwards from the requested output data marking data, passes, nodes, and ops as required.
fn find_included(graph: &GraphDef, inputs: &[DataID], static_inputs: &OrderMap<DataID, ArrayD<f32>>, outputs: &[DataID], dependencies: &Dependencies, strict_op_inclusion: bool) -> (OrderMap<DataID, DataStatus>, OrderSet<PassID>, OrderMap<NodeID, NodeStatus>, OrderSet<OpID>){
		
	let mut included_data: OrderMap<DataID, DataStatus> = OrderMap::new();
	let mut included_passes = OrderSet::new();

	for data_id in inputs.iter().chain(static_inputs.keys()) {
		included_data.insert(data_id.clone(), DataStatus::Input);
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
			if !included_data.contains_key(&data_id) {
				for pass_id in dependencies.data_inputs(&data_id) {
					pass_queue.push_back(pass_id.clone());
				}
				included_data.insert(data_id, DataStatus::Compute);
			}
		}

		if let Some(pass_id) = pass_queue.pop_front() {
			if !included_passes.contains(&pass_id) {
				for data_id in dependencies.pass_inputs(&pass_id) {
					data_queue.push_back(data_id.clone());
				}
				included_passes.insert(pass_id);
			}
		}
	}



	let mut included_nodes = OrderMap::new();
	let mut included_ops = OrderSet::new();

	for data_id in static_inputs.keys() {
		included_nodes.insert(data_id.node_id(), NodeStatus::StaticInput);
	}
	// Input overwrites StaticInput
	for data_id in inputs.iter() {
		included_nodes.insert(data_id.node_id(), NodeStatus::InputOrKnown);
	}

	for node_id in graph.get_nodes() {
		if !included_nodes.contains_key(node_id) && node_id.shape().is_known() {
			included_nodes.insert(node_id.clone(), NodeStatus::InputOrKnown);
		}
	}

	let mut op_queue: VecDeque<OpID> = VecDeque::new();
	let mut node_queue: VecDeque<NodeID> = VecDeque::new(); 

	for (data_id, _status) in &included_data {
		let node_id = data_id.node_id();
		if !included_nodes.contains_key(&node_id) {
			node_queue.push_back(node_id);
		}
	}

	while !(op_queue.is_empty() && node_queue.is_empty()) {
		if let Some(node_id) = node_queue.pop_front() {
			if !included_nodes.contains_key(&node_id) {
				for op_id in dependencies.node_inputs(&node_id) {
					op_queue.push_back(op_id.clone());
				}
				included_nodes.insert(node_id, NodeStatus::Infer);
			}
		
		} else if let Some(op_id) = op_queue.pop_front() { // non strict mode requires all nodes included before ops are tried
			if !included_ops.contains(&op_id) {
				if strict_op_inclusion {
					for node_id in dependencies.op_inputs(&op_id) {
						node_queue.push_back(node_id.clone());
					}
					included_ops.insert(op_id);
				} else if dependencies.op_inputs(&op_id).iter().all(|node_id| included_nodes.contains_key(node_id)) {
					included_ops.insert(op_id); // only include ops which have all inputs already included. Don't queue any further nodes.
				}
			}
		}
	}


	// // if outputs are 'Infer' include
	// // if outputs are `StaticInputs` AND no inputs are `NotIncluded` then include
	// if shape_outputs.iter().any(|node_id| matches!(included_nodes.get(node_id).unwrap(), &NodeStatus::Infer)) {
	// 	included_ops.insert(op_id.clone());
	// } else if shape_outputs.iter().any(|node_id| matches!(included_nodes.get(node_id).unwrap(), &NodeStatus::StaticInput))
	// && !shape_inputs.iter().any(|node_id| included_nodes.get(node_id).is_none()){
	// 	included_ops.insert(op_id.clone());
	// }


	// Sort all this back into graph order
	let included_data = graph.get_nodes().iter().flat_map(|node_id| vec![node_id.value_id(), node_id.gradient_id()]).filter_map(|data_id|{
		included_data.get(&data_id).map(|status| (data_id, status.clone()))
	}).collect();

	let included_passes: OrderSet<PassID> = graph.get_passes().iter().filter(|&pass_id| included_passes.contains(pass_id)).cloned().collect();

	let included_nodes = graph.get_nodes().iter().filter_map(|node_id|{
		included_nodes.get(node_id).map(|status| (node_id.clone(), status.clone()))
	}).collect();

	let included_ops: OrderSet<OpID> = graph.get_ops().iter().filter(|&op_id| included_ops.contains(op_id)).cloned().collect();

	(included_data, included_passes, included_nodes, included_ops)
}

/// Returns the order in which passes should be called such that dependencies are respected.
/// By default this will order passes in the order that they were added to the graph, and only perform the minimal rearrangement required to ensure dependencies are met.
/// out of order dependeancies can cause quadratic slow down (this can probably be removed using priority queues)
fn find_pass_order(included_data: &OrderMap<DataID, DataStatus>, included_passes: &OrderSet<PassID>, dependencies: &Dependencies) -> Result<Vec<PassID>>{

	#[derive(Clone, Debug)]
	enum DataState {
		Input,
		Ready, // Data should be marked as ready if 1) it is a graph input, or 2) when the last input pass to it is sucessfully retired as ready
		Pending(usize), // Indicates the number of remaining input passes before the data can be marked as ready
	};

	// states start as pending, if all inputs to a pass or data are 'ready' then it is ready, if any inputs to a pass or node are unavailable then it is unavailable.
	#[derive(Clone, Debug)]
	enum PassState {
		Ready, // Indicates that a pass has been retired as ready
		Pending(usize), // Indicates the number of remaining input data before the pass can be marked as ready
	};

	/// Attempts to retire a pass as Ready or Unavailable, return true if sucessful false otherwise
	/// If it returns true this method should never be called again for that pass_id.
	fn try_retire_pass(pass_id: &PassID, pass_order: &mut Vec<PassID>, data_states: &mut OrderMap<DataID, DataState>, pass_states: &mut OrderMap<PassID, PassState>, dependencies: &Dependencies) -> bool{

		if matches!(pass_states.get(pass_id).unwrap(), &PassState::Ready) {
			panic!("pass has already been retired, try_retire_pass() should not be called")
		} else if matches!(pass_states.get(pass_id).unwrap(), &PassState::Pending(0)) {
			// add to pass order and update output data locations readiness
			pass_order.push(pass_id.clone());
			pass_states.insert(pass_id.clone(), PassState::Ready);

			for data_id in dependencies.pass_outputs(pass_id) {
				// If a data output of a pass is pending decrement
				// If that data output can now be marked ready
				if !data_states.contains_key(data_id) {continue;}
				match data_states.get(data_id).unwrap() {
					&DataState::Input => {},
					&DataState::Pending(rem) => {
						if rem == 1 {
							mark_data_ready(data_id, data_states, pass_states, &dependencies)
						} else if rem > 1 {
							data_states.insert(data_id.clone(), DataState::Pending(rem - 1));
						} else {
							panic!("Data with zero inputs should have already been marked Unavailable or Input")
						}
					},
					&DataState::Ready => panic!("data marked ready before last input pass was processed. graph likely contains a requires pass which writes to a input tensor"), //TODO: create test to confirm this is caused by fan-out ops writing to a subgraph input
				}
			}
			true
		} else {
			false
		}
	}

	/// Marks data as ready, and decreases pending count of dependent passes
	/// Only legal to call this if is_input[]==true or as the last input pass is retired
	fn mark_data_input(data_id: &DataID, data_states: &mut OrderMap<DataID, DataState>, pass_states: &mut OrderMap<PassID, PassState>, dependencies: &Dependencies){
		data_states.insert(data_id.clone(), DataState::Input);
		for pass_id in dependencies.data_outputs(data_id) {
			if let Some(pass_state) = pass_states.get_mut(pass_id) {
				match pass_state {
					&mut PassState::Pending(rem) if rem > 0 => *pass_state = PassState::Pending(rem - 1),
					&mut PassState::Pending(_) | &mut PassState::Ready => panic!("Something has happened out of order. pass_id: {}", pass_id),
				}
			}
		}
	}

	/// Marks data as ready, and decreases pending count of dependent passes
	/// Only legal to call this if is_input[]==true or as the last input pass is retired
	fn mark_data_ready(data_id: &DataID, data_states: &mut OrderMap<DataID, DataState>, pass_states: &mut OrderMap<PassID, PassState>, dependencies: &Dependencies){
		data_states.insert(data_id.clone(), DataState::Ready);
		for pass_id in dependencies.data_outputs(data_id) {
			if let Some(pass_state) = pass_states.get_mut(pass_id) {
				match pass_state {
					&mut PassState::Pending(rem) if rem > 0 => *pass_state = PassState::Pending(rem - 1),
					&mut PassState::Pending(_) | &mut PassState::Ready => panic!("Something has happened out of order. pass_id: {}", pass_id),
				}
			}
		}
	}



	let mut pass_order: Vec<PassID> = vec![];
	let mut deferred_passes: VecDeque<PassID> = VecDeque::new();
	
	// Setup states
	let mut pass_states = included_passes.iter().map(|id| (id.clone(), PassState::Pending(dependencies.pass_inputs(id).len()))).collect();
	let mut data_states = included_data.keys().map(|id| (id.clone(), DataState::Pending(dependencies.data_inputs(id).len()))).collect();
	let mut unavailable_data = vec![];
	for (data_id, data_status) in included_data.iter() {
		match data_status {
			&DataStatus::Input => {
				mark_data_input(data_id, &mut data_states, &mut pass_states, &dependencies);
			},
			&DataStatus::Compute => {
				if dependencies.data_inputs(data_id).len() == 0 { // if not an input, and has no input passes, it is unavailable
					unavailable_data.push(data_id.clone());
				}
			}
		}
	}

	if unavailable_data.len() > 0 {
		let unavailable_names: Vec<String> = unavailable_data.iter().map(|id| id.name()).collect();
		bail!(ErrorKind::SubgraphInsufficientInputsForOutputs(unavailable_names))
	}


	// iterate over all required passes,
	// add to pass order where possible (inputs are ready), otherwise add to deferred queue
	let forward_required_passes = included_passes.iter().filter(|id| dependencies.pass_is_forward(id));
	let backward_required_passes = included_passes.iter().filter(|id| !dependencies.pass_is_forward(id));
	let default_pass_order = forward_required_passes.chain(backward_required_passes.rev());

	for pass_id in default_pass_order {
		
		let success = try_retire_pass(pass_id, &mut pass_order, &mut data_states, &mut pass_states, &dependencies);
		if !success {
			deferred_passes.push_back(pass_id.clone());
			continue;
		}

		// Attempt to empty deferred queue
		// always try to add deferred passes in order
		let mut i = 0;
		while i < deferred_passes.len(){
			let success = try_retire_pass(&deferred_passes[i], &mut pass_order, &mut data_states, &mut pass_states, &dependencies);
			if success {
				deferred_passes.remove(i);
				i = 0; // keep trying from the start again
			} else {
				i += 1;
			}
		}
	}


	if deferred_passes.len() > 0 {
		bail!(ErrorKind::GraphContainsCircularPasses(deferred_passes.into_iter().map(|pass| (pass.clone(), dependencies.pass_inputs(&pass).iter().filter(|&data_id| !matches!(data_states.get(data_id), Some(&DataState::Ready))).cloned().collect())).collect()))
	}

	Ok(pass_order)
}

/// Returns the order in which op should be called such that dependencies are respected.
/// By default this will order ops in the order that they were added to the graph, and only perform the minimal rearrangement required to ensure dependencies are met.
/// out of order dependeancies can cause quadratic slow down (this can probably be removed using priority queues)
fn find_op_order(included_nodes: &OrderMap<NodeID, NodeStatus>, included_ops: &OrderSet<OpID>, dependencies: &Dependencies) -> Result<Vec<OpID>>{

	#[derive(Clone, Debug)]
	enum NodeState {
		Input,
		Ready, // Node should be marked as ready if 1) it is a graph input, or 2) when the last input op to it is sucessfully retired as ready
		Pending(usize), // Indicates the number of remaining input opes before the node can be marked as ready
	};

	// states start as pending, if all inputs to a op or node are 'ready' then it is ready, if any inputs to a op or node are unavailable then it is unavailable.
	#[derive(Clone, Debug)]
	enum OpState {
		Ready, // Indicates that a op has been retired as ready
		Pending(usize), // Indicates the number of remaining input node before the op can be marked as ready
	};

	/// Attempts to retire a op as Ready or Unavailable, return true if sucessful false otherwise
	/// If it returns true this method should never be called again for that op_id.
	fn try_retire_op(op_id: &OpID, op_order: &mut Vec<OpID>, node_states: &mut OrderMap<NodeID, NodeState>, op_states: &mut OrderMap<OpID, OpState>, dependencies: &Dependencies) -> bool{

		if matches!(op_states.get(op_id).unwrap(), &OpState::Ready) {
			panic!("op has already been retired, try_retire_op() should not be called")
		} else if matches!(op_states.get(op_id).unwrap(), &OpState::Pending(0)) {
			// add to op order and update output node locations readiness
			op_order.push(op_id.clone());
			op_states.insert(op_id.clone(), OpState::Ready);
			for node_id in dependencies.op_shape_outputs(op_id) {
				// If a node output of a op is pending decrement
				// If that node output can now be marked ready
				if !node_states.contains_key(node_id) {continue;}
				match node_states.get(node_id).unwrap() {
					&NodeState::Input => {},
					&NodeState::Pending(rem) if rem == 1 => {
						mark_node_ready(node_id, node_states, op_states, &dependencies)
					},
					&NodeState::Pending(rem) if rem > 1 => {node_states.insert(node_id.clone(), NodeState::Pending(rem - 1));},
					&NodeState::Pending(_) => panic!("node with zero inputs should have already been marked Unavailable or Input"),
					&NodeState::Ready => panic!("node marked ready before last input op was processed. graph likely contains a requires op which writes to a input tensor"), //TODO: create test to confirm this is caused by fan-out ops writing to a subgraph input
				}
			}
			true
		} else {
			false
		}
	}

	/// Marks node as ready, and decreases pending count of dependent ops
	/// Only legal to call this if is_input[]==true or as the last input op is retired
	fn mark_node_input(node_id: &NodeID, node_states: &mut OrderMap<NodeID, NodeState>, op_states: &mut OrderMap<OpID, OpState>, dependencies: &Dependencies){
		node_states.insert(node_id.clone(), NodeState::Input);
		for op_id in dependencies.node_outputs(node_id) {
			if let Some(op_state) = op_states.get_mut(op_id){
				match op_state {
					&mut OpState::Pending(rem) if rem > 0 => {*op_state = OpState::Pending(rem - 1)},
					&mut OpState::Pending(_) | &mut OpState::Ready => panic!("Something has happened out of order. node_id: {} op_id: {}", node_id, op_id),
				}
			}
		}
	}

	/// Marks node as ready, and decreases pending count of dependent ops
	/// Only legal to call this if is_input[]==true or as the last input op is retired
	fn mark_node_ready(node_id: &NodeID, node_states: &mut OrderMap<NodeID, NodeState>, op_states: &mut OrderMap<OpID, OpState>, dependencies: &Dependencies){
		node_states.insert(node_id.clone(), NodeState::Ready);
		for op_id in dependencies.node_outputs(node_id) {
			if let Some(op_state) = op_states.get_mut(op_id){
				match op_state {
					&mut OpState::Pending(rem) if rem > 0 => {*op_state = OpState::Pending(rem - 1)},
					&mut OpState::Pending(_) | &mut OpState::Ready => panic!("Something has happened out of order. node_id: {} op_id: {}", node_id, op_id),
				}
			}
		}
	}


	let mut op_order: Vec<OpID> = vec![];
	let mut deferred_ops: VecDeque<OpID> = VecDeque::new();
	
	// Setup states
	let mut op_states = included_ops.iter().map(|id| (id.clone(), OpState::Pending(dependencies.op_inputs(id).len()))).collect();
	let mut node_states = included_nodes.keys().map(|id| (id.clone(), NodeState::Pending(dependencies.node_shape_inputs(id).iter().filter(|&id| included_ops.contains(id)).count()))).collect(); // due to non-strict mode for op inclusion, not all input ops are nessesarily included.
	let mut unavailable_nodes = vec![];
	for (node_id, node_status) in included_nodes.iter() {
		match node_status {
			&NodeStatus::InputOrKnown => {
				mark_node_input(node_id, &mut node_states, &mut op_states, &dependencies)
			},
			&NodeStatus::StaticInput => {
				if !dependencies.node_shape_inputs(node_id).iter().any(|op_id| included_ops.contains(op_id)) {
					mark_node_input(node_id, &mut node_states, &mut op_states, &dependencies)
				}
			},
			&NodeStatus::Infer => {
				if dependencies.node_shape_inputs(node_id).len() == 0 {
					unavailable_nodes.push(node_id.clone())
				}
			},
		}
	}

	if unavailable_nodes.len() > 0 {
		let unavailable_names: Vec<String> = unavailable_nodes.iter().map(|id| id.name().to_string()).collect();
		bail!(ErrorKind::SubgraphInsufficientInputsForShapeInference(unavailable_names))
	}

	// iterate over all required ops,
	// add to op order where possible (inputs are ready), otherwise add to deferred queue
	for op_id in included_ops {

		let success = try_retire_op(&op_id, &mut op_order, &mut node_states, &mut op_states, &dependencies);
		if !success {
			deferred_ops.push_back(op_id.clone());
			continue;
		}

		// Attempt to empty deferred queue
		// always try to add deferred ops in order
		let mut i = 0;
		while i < deferred_ops.len(){
			let success = try_retire_op(&deferred_ops[i], &mut op_order, &mut node_states, &mut op_states, &dependencies);
			if success {
				deferred_ops.remove(i);
				i = 0; // keep trying from the start again
			} else {
				i += 1;
			}
		}
	}

	if deferred_ops.len() > 0 {
		bail!(ErrorKind::GraphContainsCircularOps(deferred_ops.into_iter().map(|op_id| (op_id.clone(), dependencies.op_inputs(&op_id).iter().filter(|&node_id| !matches!(node_states.get(node_id), Some(&NodeState::Ready))).cloned().collect())).collect()))
	}

	Ok(op_order)
}


fn find_shapes(subgraph: &Subgraph, op_order: &[OpID], inputs: &OrderMap<DataID, ArrayD<f32>>, static_inputs: &OrderMap<DataID, ArrayD<f32>>) -> Result<OrderMap<NodeID, IxDyn>> {
	// if inputs are present along with static_inputs the inputs should add

	let mut shapes = GraphShapes::new(subgraph);

	// for all inputs, merge data shape into existing graph shape
	//ensure!(inputs.len() == input_data.len(), ErrorKind::InputSizeError);
	for (input_id, input_data) in inputs {
		shapes.merge_input(input_id, input_data.shape()).chain_err(|| format!("Could not merge input value supplied to {}", input_id))?;
	}

	// for all static inputs, if not in inputs, merge into graph shape
	// because static_inputs can be broadcast, resolving the dimension will be harder
	// iterate from the lowest dimension up, if the static_input dimension is not 1 then enforce it in the shape
	for (static_input_id, static_input_data) in static_inputs.iter() {
		if !inputs.contains_key(static_input_id) {
			shapes.merge_static_input(&static_input_id, static_input_data.shape()).chain_err(|| format!("Could not merge static input for {}", static_input_id))?;
		}
	}

	// for all ops that are scheduled, call the relevant shape propagation
	for op_id in op_order {
		shapes.set_current_op(Some(op_id.clone()));
		op_id.instance().propagate_shape_constraints(&mut shapes).chain_err(|| format!("Could not complete shape inference for {}", op_id))?;
	}
	shapes.set_current_op(None);

	let temp_vec: Result<Vec<(NodeID, IxDyn)>> = shapes.shapes.iter_mut().map(|(id, shape)| {
		shape.collapse_dimensions_to_minimum();
		shape.to_data_shape().map_err(|e| e.into()).map(|shape| (id.clone(), shape))
	}).collect();

	Ok(temp_vec?.into_iter().collect())
}



/// The interface through which ops can perform shape propagation.
///
/// Immediately prior to each graph execution, the propagation of shape constraints from inputs through the graph takes place.
/// Each Op can read the shape of its inputs, and new constraints can be applied/merged with the shapes of its outputs.
#[derive(Debug)]
pub struct GraphShapes<'a> {
	shapes: OrderMap<NodeID, NodeShape>,
	subgraph: &'a Subgraph,
	current_op_instance: Option<OpID>,
}

impl<'a> GraphShapes<'a> {
	fn new(subgraph: &Subgraph) -> GraphShapes {
		GraphShapes{
			shapes: subgraph.included_nodes.keys().map(|id| (id.clone(), id.shape().clone())).collect(),
			subgraph: subgraph,
			current_op_instance: None,
		}
	}

	fn set_current_op(&mut self, op_id: Option<OpID>){
		self.current_op_instance = op_id;
	}

	pub fn current_op_instance(&self) -> &Option<OpID>{
		&self.current_op_instance
	}

	fn merge_input(&mut self, data_id: &DataID, shape: &[Ix]) -> Result<()>{
		let new_shape = self.shapes.get(&data_id.node_id()).unwrap().merge(&shape.iter().cloned().into())?;
		self.shapes.insert(data_id.node_id(), new_shape);
		Ok(())
	}

	fn merge_static_input(&mut self, data_id: &DataID, shape: &[Ix]) -> Result<()> {
		let shape: NodeShape = shape.iter().map(|&ix| if ix == 1 {NodeDim::Unknown} else {NodeDim::Known(ix)}).into();
		let new_shape = self.shapes.get(&data_id.node_id()).unwrap().merge(&shape)?;
		self.shapes.insert(data_id.node_id(), new_shape);
		Ok(())
	}

	/// Causes the shape to be collapsed to the smallest allowed known shape. Should only be called for op inputs.
	// TODO only allow getting inputs
	pub fn get_shape(&mut self, id: &NodeID) -> &NodeShape{
		self.shapes.get_mut(id).unwrap().collapse_dimensions_to_minimum();
		debug_assert!(self.shapes.get(id).unwrap().dimensions().iter().all(|dim| matches!(dim, &NodeDim::Known(_))));
		self.shapes.get(id).unwrap()
	}

	// TODO only allow getting outputs
	pub fn get_output_shape(&self, id: &NodeID) -> &NodeShape{
		// self.shapes[id.index].collapse_dimensions_to_minimum();
		// debug_assert!(self.shapes[id.index].dimensions().iter().all(|dim| matches!(dim, &NodeDim::Known(_))));
		self.shapes.get(id).unwrap()
	}

	// TODO only allow merging to outputs
	pub fn merge_with(&mut self, id: &NodeID, shape: &NodeShape) -> Result<()>{
		let new_shape = self.shapes.get(id).unwrap().merge(shape)?;
		self.shapes.insert(id.clone(), new_shape);
		Ok(())
	}
}




















#[test]
fn test_build(){
	_test_build().unwrap();
}

fn _test_build() -> Result<()>{
	use ops::dummy::Dummy;
	use graph::GraphDef;

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
	use ops::dummy::Dummy;
	use graph::GraphDef;

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
	use ops::dummy::Dummy;
	use graph::GraphDef;

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
	use ops::dummy::Dummy;
	use graph::GraphDef;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![Unknown, 5, 16], "node1", tag!["input"])?;
	let node2a = g.new_node(shape![Unknown, 5, 16], "node2a", tag![])?;
	let node2b = g.new_node(shape![Unknown, 5, 16], "node2b", tag![])?;
	let node3 = g.new_node(shape![Unknown, 5, 16], "node3", tag![])?;
	let node4 = g.new_node(shape![Unknown, 5, 16], "node4", tag!["output"])?;

	let o4 = g.new_op(Dummy::new().input(&node2a).output(&node3), tag![])?; //70
	let o2 = g.new_op(Dummy::new().input(&node2b).output(&node3), tag![])?; //72
	let o1 = g.new_op(Dummy::new().input(&node1).output(&node2b), tag![])?; //74
	let o3 = g.new_op(Dummy::new().input(&node1).output(&node2a), tag![])?; //76
	let o5 = g.new_op(Dummy::new().input(&node3).output(&node4), tag![])?; //78
	let o6 = g.new_op(Dummy::new().input(&node4), tag![])?; //80


	let sg_forward = g.subgraph(&[node1.value_id()], &[node4.value_id()])?;
	let expected_order: Vec<OpID> = [&o1, &o2, &o3, &o4, &o5].iter().map(|&op_id| op_id.clone()).collect();
	assert_eq!(&sg_forward.op_order, &expected_order);
	let expected_order: Vec<PassID> = [&o1, &o2, &o3, &o4, &o5].into_iter().map(|id| id.instance()).collect::<Vec<_>>()
		.iter().map(|op| op.inner_passes()[0].clone()).collect();
	assert_eq!(&sg_forward.pass_order, &expected_order);

	let sg_forward_backward = g.subgraph(&[node1.value_id()], &[node1.gradient_id()])?;
	let expected_order: Vec<OpID> = [&o1, &o2, &o3, &o4, &o5].iter().map(|&op_id| op_id.clone()).collect();
	assert_eq!(&sg_forward_backward.op_order, &expected_order);
	// backward pass prefers to run in the opposite order to the order of ops being added
	// this results in o2 and o1 firing before o4 and o3
	// if you read the op order above bottom to top, you can see this is correct.
	let expected_order: Vec<PassID> = [&o1, &o2, &o3, &o4, &o5].into_iter().map(|id| id.instance()).collect::<Vec<_>>()
		.iter().map(|op| op.inner_passes()[0].clone())
		.chain([&o6, &o5, &o2, &o1, &o4, &o3].into_iter().map(|id| id.instance()).collect::<Vec<_>>()
		.iter().map(|op| op.inner_passes()[1].clone())).collect();
	assert_eq!(&sg_forward_backward.pass_order, &expected_order);

	Ok(())
}


#[test]
fn test_circular_detection(){
	_test_circular_detection().unwrap();
}

fn _test_circular_detection() -> Result<()>{
	use ops::dummy::Dummy;
	use graph::GraphDef;

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

	Ok(())
}


#[test]
fn test_insufficient_input_detection(){
	_test_insufficient_input_detection().unwrap();
}

fn _test_insufficient_input_detection() -> Result<()>{
	use ops::dummy::Dummy;
	use graph::GraphDef;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![Unknown, 5, 16], "node1", tag!["input"])?;
	let node2 = g.new_node(shape![5, 5, 16], "node2", tag![])?;
	let node3 = g.new_node(shape![Unknown, 5, 16], "node3", tag!["output"])?;


	let _o1 = g.new_op(Dummy::new().input(&node1).output(&node3), tag![])?;
	let _o2 = g.new_op(Dummy::new().input(&node2).output(&node3), tag![])?;


	let sg_forward = g.subgraph(&[node1.value_id()], &[node3.value_id()]);
	assert!(matches!(sg_forward, Err(Error(ErrorKind::SubgraphInsufficientInputsForOutputs(_), _))), "{:?}", sg_forward);

	let sg_forward_backward = g.subgraph(&[node1.value_id()], &[node1.gradient_id()]);
	assert!(matches!(sg_forward_backward, Err(Error(ErrorKind::SubgraphInsufficientInputsForOutputs(_), _))), "{:?}", sg_forward_backward);

	Ok(())
}



// TODO detect required ops which want to write to input data

// TODO detect that name conflict detection works

// TODO detect problems with shape propagation

// TODO detect problems with static_input broadcasting