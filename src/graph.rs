use std::cell::RefCell;
use std::mem;
use std::ops::Range;
use self::indexing::*;
use shape::*;
use ops::Operation;

#[derive(Debug, Clone)]
pub enum ParamRange {
	Primary(Range<usize>),
	Secondary(Range<usize>),
}

impl ParamRange{
	pub fn get_range(&self) -> Range<usize> {
		match self {
			&ParamRange::Primary(ref range) => range.clone(),
			&ParamRange::Secondary(ref range) => range.clone(),			
		}
	}

	pub fn get_len(&self) -> usize {
		match self {
			&ParamRange::Primary(ref range)|&ParamRange::Secondary(ref range) => range.len(),		
		}
	}

	pub fn get_primary_len(&self) -> usize {
		match self {
			&ParamRange::Primary(ref range) => range.len(),
			&ParamRange::Secondary(_) => 0,			
		}
	}
	
	pub fn is_primary(&self) -> bool {
		match self {
			&ParamRange::Primary(_) => true,
			&ParamRange::Secondary(_) => false,			
		}
	}
}

//TODO:
// 1. add way to specify "locked" parameters, zero out their error gradients in both evalutation and regularisation
// 2. add way to specify parameter reuse across operations //DONE
// 3. add way to do regularisation functions, including subgradients
// 4. add way to add default parameter generaters to ops.
// 5. add interface for unrolling into RNN
// 1. add way to add a subgraph node
#[derive(Clone)]
pub struct Graph{
	input_indices: Vec<NodeIndex>,
	training_input_indices: Vec<NodeIndex>,
	output_indices: Vec<NodeIndex>,
	nodes: Vec<Node>,
	operations: Vec<Box<Operation>>,
	training_order: Vec<GraphIndex>,
	evaluation_order: Vec<GraphIndex>,
	param_ranges: Vec<ParamRange>, // parameter ranges corresponding to operations vector
	
	is_ordered: bool,
}


impl Graph{
	pub fn new() -> Graph{
		Graph{
			nodes: vec![Node::new_flat(1, "Unit")],
			operations: vec![],
			input_indices: vec![],
			training_input_indices: vec![],
			output_indices: vec![],
			training_order: vec![],
			evaluation_order: vec![],
			is_ordered: true,
			param_ranges: vec![],
		}
	}

	pub fn operations(&self) -> &[Box<Operation>]{
		&self.operations
	}

	pub fn nodes(&self) -> &[Node]{
		&self.nodes
	}
	
	pub fn input_node_indices(&self) -> &[NodeIndex]{
		&self.input_indices
	}

	pub fn input_nodes(&self) -> Vec<Node>{
		self.input_indices.iter().map(|&i| self.nodes[..][i].clone()).collect()
	}

	pub fn training_input_node_indices(&self) -> &[NodeIndex]{
		&self.training_input_indices
	}

	pub fn training_input_nodes(&self) -> Vec<Node>{
		self.training_input_indices.iter().map(|&i| self.nodes[..][i].clone()).collect()
	}
	
	pub fn ops(&self) -> &[Box<Operation>]{
		&self.operations
	}
	
	pub fn evaluation_order(&self) -> &[GraphIndex]{
		&self.evaluation_order
	}

	pub fn num_params(&self) -> usize{
		self.param_ranges.iter().fold(0, |acc, r| acc + r.get_primary_len()) 
	}
	

	pub fn unit_node_index(&self) -> NodeIndex{
		NodeIndex(0)
	}
	
	pub fn add_input_node(&mut self, node: Node) -> (NodeIndex, NodeShape){
		self.is_ordered = false;
		let index = self.nodes.len().into();
		self.input_indices.push(index);
		self.nodes.push(node);
		(index, self.nodes[index.0].shape.clone())
	}
	
	pub fn add_training_input_node(&mut self, node: Node) -> (NodeIndex, NodeShape){
		self.is_ordered = false;
		let index = self.nodes.len().into();
		self.training_input_indices.push(index);
		self.nodes.push(node);
		(index, self.nodes[index.0].shape.clone())
	}
	
	pub fn add_output_node(&mut self, node: Node) -> (NodeIndex, NodeShape){
		self.is_ordered = false;
		let index: NodeIndex = self.nodes.len().into();
		self.output_indices.push(index);
		self.nodes.push(node);
		(index, self.nodes[index.0].shape.clone())
	}
	
	pub fn add_node(&mut self, node_in: Node) -> (NodeIndex, NodeShape) {
		self.is_ordered = false;
		let index = self.nodes.len().into();
		self.nodes.push(node_in);
		(index, self.nodes[index.0].shape.clone())
	}
	
	
	pub fn add_operation(&mut self, op: Box<Operation>) -> (OpIndex, usize){
		self.is_ordered = false;
		
		if op.output_node_ind().contains(&self.unit_node_index())
		|| self.input_indices.iter().any(|ind| op.output_node_ind().contains(ind))
		|| self.training_input_indices.iter().any(|ind| op.output_node_ind().contains(ind)) {
			panic!("Operation '{}' attempted to use a graph input as an output node.", op.name());
		}
		
		let size = self.num_params();
		self.param_ranges.push(ParamRange::Primary(Range{start: size, end: size + op.num_params()}));
		let index = self.operations.len().into();
		let num_params = op.num_params();
		self.operations.push(op);
		(index, num_params)
	}

	
	/// add new operation between nodes, but share the existing parameters from an existing operation
	/// must have same number of parameters as primary operation
	pub fn add_secondary_operation(&mut self, op: Box<Operation>, index: OpIndex) -> (OpIndex, usize){
		self.is_ordered = false;
		
		if op.output_node_ind().contains(&self.unit_node_index())
		|| self.input_indices.iter().any(|ind| op.output_node_ind().contains(ind))
		|| self.training_input_indices.iter().any(|ind| op.output_node_ind().contains(ind)) {
			panic!("Operation '{}' attempted to use a graph input as an output node.", op.name());
		}
		
		
		let range = self.param_ranges[index.0].get_range();
		
		assert!(range.end - range.start == op.num_params(), "Incompatible parameter sizes for '{}({})' reusing parameters from '{}({})'", 
			op.name(), op.num_params(), self.operations[index].name(), self.operations[index].num_params());
		
		self.param_ranges.push(ParamRange::Secondary(range));
		let index = self.operations.len().into();
		let num_params = op.num_params();
		self.operations.push(op);
		(index, num_params)
	}
	
	pub fn add_operations(&mut self, ops: Vec<Box<Operation>>) -> Vec<(OpIndex, usize)>{
		let mut indices = Vec::with_capacity(ops.len());
		for op in ops {
			indices.push(self.add_operation(op));
		}
		indices
	}
		
	pub fn init_params(&mut self) -> Vec<f32>{
		let mut params = vec![0.0; self.num_params()];
		
		for (pr, op) in self.param_ranges.iter().zip(self.operations.iter_mut()).filter(|&(pr, _)| pr.is_primary()){
			op.init_params(&mut params[pr.get_range()]);
		}
		
		params
	}
	
	/// Runs the graph in the forward direction only	
	pub fn forward(&mut self, n: usize, input: Vec<NodeData>, params: &[f32]) -> Vec<NodeData> { // returns output node values
		let mut data = self.init_data(n, input, None, true);
		let mut data = &mut data[..];
		
		//forward prop each operation
		for index in self.evaluation_order.iter() {
			let i = match index {
				&GraphIndex::O(op_index) => op_index,
				_ => continue,
			};
			
			self.operations[i].forward(&mut data, &params[self.param_ranges[i.0].get_range()]);
		}
		
		self.output_indices.iter()
			.map(|&index| {
				mem::replace(&mut *data[index].borrow_mut(), NodeData::dummy())
			}).collect::<Vec<NodeData>>()
	}
	
	// pub fn regularise(params: &[f32]) -> (f32, Vec<f32>) {
	// 	(0.0, vec![0.0;params.len()]) ///////////////////////////////////////////////////////TODO !!!
	// }
	
	//TODO: param_derives passed in from outside but error isnt? could lead to problems if param derives are accumulated by being passed in multiple times.
	/// Performs the forward and backward passes through the graph, and updates parameters. 
	pub fn backprop_mut(&mut self, n: usize, input: Vec<NodeData>, training_input: Vec<NodeData>, params: &[f32], err: &mut f32, param_derivs: &mut [f32]) -> Vec<NodeData>{
		let mut data = self.init_data(n, input, Some(training_input), false);
		let mut data = &mut data[..];
		
		//forward prop each operation
		for index in self.training_order.iter() {
			let i = match index {
				&GraphIndex::O(op_index) => op_index,
				_ => continue,
			};
			
			self.operations[i].forward(&mut data, &params[self.param_ranges[i.0].get_range()]);
		}
		
		//backprop eash operation in reverse order
		for index in self.training_order.iter().rev() {
			let i = match index {
				&GraphIndex::O(op_index) => op_index,
				_ => continue,
			};
			
			self.operations[i].backward(&mut data, &params[self.param_ranges[i.0].get_range()], &mut param_derivs[self.param_ranges[i.0].get_range()], err);
		}
		

		data.iter().map(|refcell| {
				mem::replace(&mut *refcell.borrow_mut(), NodeData::dummy())
			}).collect::<Vec<NodeData>>()
	}

	pub fn backprop(&mut self, n: usize, input: Vec<NodeData>, training_input: Vec<NodeData>, params: &[f32]) -> (f32, Vec<f32>, Vec<NodeData>){ // error, error derivatives over parameters, data
		let mut param_derivs = vec![0.0; params.len()];
		let mut err = 0.0;
		let data = self.backprop_mut(n, input, training_input, params, &mut err, &mut param_derivs);
		(err, param_derivs, data)
	}
	
	fn check_order(&mut self){
		if !self.is_ordered {
			self.training_order = self.generate_order(true);
			self.evaluation_order = self.generate_order(false);
			self.is_ordered = true;
		}
	}
	
	fn generate_order(&self, training_order: bool) -> Vec<GraphIndex>{

		let mut order = vec![];
		let mut ready = vec![false; self.nodes.len()];
		
		//Unit and start nodes are ready a priori
		ready[self.unit_node_index()] = true;
		for &start_ind in self.input_indices.iter() {
			ready[start_ind] = true;
		}
		// If generative a training order set those inputs to ready
		if training_order {
			for &train_ind in self.training_input_indices.iter(){
				ready[train_ind] = true;
			}
		}


		let mut finished = false;
		while !finished {

			// 1. gather output nodes of operations which currently done have satisfied inputs.
			let input_ops_not_ready = self.operations.iter()
				.filter(|&op| !op.input_node_ind().iter().all(|&ind| ready[ind]))
				.flat_map(|op| op.output_node_ind())
				.collect::<Vec<_>>();

			// 2. set output nodes of all operations that have satisfied inputs to true
			for ind in order.iter()
			.filter_map(|g| match g {&GraphIndex::O(op_ind) => Some(op_ind), _ => None})
			.flat_map(|ind| self.operations[ind].output_node_ind()) {
				ready[ind] = true
			}

			// 3. set output nodes of all operations gathered in 1 to false;
			for ind in input_ops_not_ready {
				ready[ind] = false;
			}
			
			//add new nodes which now have all incoming operations ready
			let mut new_satisfied_nodes = ready.iter().enumerate()
				.filter(|&(_, &b)| b)
				.filter(|&(i, _)| !order.contains(&GraphIndex::node(i)))
				.map(|(i, _)| GraphIndex::node(i))
				.collect::<Vec<_>>();
			

			//find new operations that have all input nodes available
			let mut new_satisfied_ops = self.operations.iter().enumerate()
				.filter(|&(i, _)| !order.contains(&GraphIndex::op(i)))
				.filter(|&(_, op)| op.input_node_ind().iter().all(|&ind| ready[ind]))
				.map(|(i, _)| GraphIndex::op(i))
				.collect::<Vec<_>>();
			
			// if no new operations become available, then process is complete or at a dead end. Finished.
			finished = 0 == new_satisfied_nodes.len() + new_satisfied_ops.len();
			
			order.append(&mut new_satisfied_nodes);
			order.append(&mut new_satisfied_ops);
		}
		
		if training_order {
			for (i, _) in ready.iter().enumerate().filter(|&(_, b)| !b) {
				println!("Warning: Node #{} '{}' can not be included in the computation order. Graph may be circular, or have a node with no inputs.", i, self.nodes[i].name);
			}
			
			for (i, op) in self.operations.iter().enumerate().filter(|&(_, op)| !op.input_node_ind().iter().all(|&ind| ready[ind])){
				println!("Warning: An input node to Operation #{} '{}' is not available. The operation has been removed.", i, op.name());
			}
		} else {
			for (i, _) in ready.iter().enumerate().filter(|&(_, b)| !b).filter(|&(i, _)| self.output_indices.contains(&i.into())) {
				println!("Warning: Output Node #{} '{}' can not be included in the computation order. Check warnings for training order to see if other nodes are affected.", i, self.nodes[i].name);
			}
		}

		order
	}
	

	
	fn init_data(&mut self, n: usize, inputs: Vec<NodeData>, training_inputs: Option<Vec<NodeData>>, no_derivs: bool) -> Vec<RefCell<NodeData>>{
		assert!(inputs.len() == self.input_indices.len() && if let Some(t_inputs) = training_inputs.as_ref() {t_inputs.len() == self.training_input_indices.len()}else{true},
			format!("Input NodeData did not match graph, expected '{}' received '{}'", self.input_indices.len(), inputs.len()));
		self.check_order();
		

		let shapes = self.determine_shapes(n, &inputs, training_inputs.as_ref().map(|vec| &vec[..]));
		
		// Set up node data, with defined unit value and input values.
		let mut opt_data: Vec<Option<NodeData>> = vec![None; shapes.len()];
		
		let unit_index = self.unit_node_index().0;
		
		let mut unit = NodeData::new_blank(shapes[unit_index].to_data_shape(n).expect("Error: Unit node isnt fixed size?"));
		for v in unit.values.iter_mut(){
			*v = 1.0;
		}
		opt_data[unit_index] = Some(unit);

		
		for (i, inp) in self.input_indices.iter().zip(inputs.into_iter()) {
			opt_data[i.0] = Some(inp);
		}
		
		if let Some(t_inputs) = training_inputs {
			for (i, inp) in self.training_input_indices.iter().zip(t_inputs.into_iter()) {
				opt_data[i.0] = Some(inp);
			}
		}
		
		//////////Rather than just filling everything with dummies if it doesnt unwrap we should iterate over order and fill holes. panicing if error occurs, then iterator over all and fill holes with dummies.
		opt_data.into_iter().enumerate().map(|(i, od)| {
			let mut d = match od {
				Some(d) => d,
				None =>  NodeData::new_blank(shapes[i].to_data_shape(n).unwrap_or(DataShape::new_flat(0, n))),
			};
			if no_derivs {d.derivatives = vec![];}
			RefCell::new(d)
		}).collect()
		
	}
	
	fn determine_shapes(&self, n: usize, inputs: &[NodeData], training_inputs: Option<&[NodeData]>) -> Vec<NodeShape>{
		// 1. generate shape array from nodes
		let mut shapes: Vec<NodeShape> = self.nodes.iter().map(|n| n.shape.clone()).collect();

		// 2. Check inputs shapes against node shapes
		for (i, node_data) in inputs.iter().enumerate() {
			let index: usize = self.input_indices[i].0;
			
			assert!(node_data.shape.n == n, "Input NodeData.shape.n did not match n of batch");
			
			shapes[index] = shapes[index].merge(&node_data.shape.to_node_shape()).expect("Input data not compatible with input node shape");
		}
		
		// 3. Check training_inputs shapes against node shapes
		if let Some(t_inputs) = training_inputs.as_ref() {
			for (i, node_data) in t_inputs.iter().enumerate() {
				let index: usize = self.training_input_indices[i].0;

				assert!(node_data.shape.n == n, "Training input NodeData.shape.n did not match n of batch");
				
				shapes[index] = shapes[index].merge(&node_data.shape.to_node_shape()).expect("Training input data not compatible with training input node shape")
			}
		}
		
		// 4. Propagate shapes, enforcing output shapes of each active operation
		for graph_index in if training_inputs.is_some() {&self.training_order}else{&self.evaluation_order}.iter() {
			let index = match graph_index {
				&GraphIndex::O(op_index) => op_index,
				_ => continue,
			};
			
			self.operations[index].propagate_shape_constraints(&self.nodes, &mut shapes);
		}
		
		// 5. Collapse any remaining shapes underdetermined to their minimum size
		for s in shapes.iter_mut(){
			s.collapse_ranges_to_minimum().unwrap();
		}


		shapes
	}
}


#[derive(Clone)]
pub struct Node {
	pub name: String,
	pub shape: NodeShape, // (columnn depth,  each dimension size). -1 = input dependant dimension.
}

impl Node{
	
	pub fn new_shaped(channels: usize, num_higher_dims: usize, name: &str) -> Node {
		Node{	
			shape: NodeShape::new_flex(channels, num_higher_dims),
			name: name.to_string(),
		}
	}
	
	pub fn new_sized(channels: usize, higher_dimensions: Vec<usize>, name: &str) -> Node {
		Node{	
			shape: NodeShape::new(channels, higher_dimensions),//(channels, higher_dimensions.iter().map(|&x| Some(x)).collect()),
			name: name.to_string(),
		}
	}
	
	pub fn new_flat(size: usize, name: &str) -> Node {
		Node{	
			shape: NodeShape::new_flat(size),
			name: name.to_string(),
		}
	}
	
}

#[derive(Clone, Debug)]
pub struct NodeData{ // structured, from the bottom up, in column depth, then each dimension in order
	pub shape: DataShape,
	pub values: Vec<f32>,
	pub derivatives: Vec<f32>,
}

impl NodeData{
	
	pub fn new(shape: DataShape, values: Vec<f32>) -> NodeData{
		let size = shape.flat_size_all();
		assert!(size == values.len());
		NodeData{
			shape: shape,
			values: values,
			derivatives: vec![0.0; size],
		}
	}
	
	pub fn new_blank(shape: DataShape) -> NodeData{
		let size = shape.flat_size_all();
		NodeData{
			shape: shape,
			values: vec![0.0; size],
			derivatives: vec![0.0; size],
		}
	}
	
	fn dummy() -> NodeData{
			NodeData{
			shape: DataShape::new_flat(0, 0),
			values: vec![0.0; 0],
			derivatives: vec![0.0; 0],
		}
	}
	
	pub fn join(mut self, mut other: NodeData) -> NodeData {
		if self.shape.channels == other.shape.channels
		&& self.shape.spatial_dimensions == other.shape.spatial_dimensions{
			self.values.append(&mut other.values);
			self.derivatives.append(&mut other.derivatives);
			self.shape.n += other.shape.n;
			self
		} else {
			panic!("cant join NodeData with different shapes");
		}
	}
	
	pub fn split(mut self, n: usize) -> (NodeData, NodeData){
		if n < self.shape.n {
			(
				NodeData::new(self.shape.clone(), self.values.split_off(n*self.shape.flat_size_single())),
				self
			)
		} else {
			panic!("cant split NodeData at {}, shape.n is only {}", n, self.shape.n);
		}
	}
	
}

#[derive(PartialEq, Debug, Copy, Clone)]
pub struct OpIndex(pub usize);

#[derive(PartialEq, Debug, Copy, Clone)]
pub struct NodeIndex(pub usize);

pub mod indexing {
	use std::cell::RefCell;
	use std::ops::{Index, IndexMut};
	use graph::*;
	use ops::Operation;
	use shape::NodeShape;
	

	
	impl From<usize> for NodeIndex{
		fn from(val: usize) -> NodeIndex {NodeIndex(val)}
	}
	
	impl From<usize> for OpIndex{
		fn from(val: usize) -> OpIndex {OpIndex(val)}
	}
	
	#[derive(Clone, PartialEq, Debug)]
	pub enum GraphIndex {
		O(OpIndex),
		N(NodeIndex),
	}
	
	impl GraphIndex {
		pub fn node(val: usize) -> GraphIndex {
			GraphIndex::N(NodeIndex(val))
		}
		pub fn op(val: usize) -> GraphIndex {
			GraphIndex::O(OpIndex(val))
		}
	}
	
	impl Index<NodeIndex> for [NodeShape]{
		type Output = NodeShape;
		fn index(&self, index: NodeIndex) -> &NodeShape {
			&self[index.0]
		}
	}
	
	impl IndexMut<NodeIndex> for [NodeShape]{
		fn index_mut(&mut self, index: NodeIndex) -> &mut NodeShape {
			&mut self[index.0]
		}
	}
	
	impl Index<NodeIndex> for [Node]{
		type Output = Node;
		fn index(&self, index: NodeIndex) -> &Node {
			&self[index.0]
		}
	}
	
	impl Index<NodeIndex> for [RefCell<NodeData>]{
		type Output = RefCell<NodeData>;
		fn index(&self, index: NodeIndex) -> &RefCell<NodeData> {
			&self[index.0]
		}
	}
	
	impl IndexMut<NodeIndex> for [RefCell<NodeData>]{
		fn index_mut(&mut self, index: NodeIndex) -> &mut RefCell<NodeData> {
			&mut self[index.0]
		}
	}
	
	impl Index<NodeIndex> for [NodeData] {
		type Output = NodeData;
		fn index(&self, index: NodeIndex) -> &NodeData {
			&self[index.0]
		}
	}
	
	impl IndexMut<NodeIndex> for [NodeData] {
		fn index_mut(&mut self, index: NodeIndex) -> &mut NodeData {
			&mut self[index.0]
		}
	}
	
	impl Index<NodeIndex> for Vec<bool>{
		type Output = bool;
		fn index(&self, index: NodeIndex) -> &bool {
			&self[index.0]
		}
	}
	impl IndexMut<NodeIndex> for Vec<bool>{
		fn index_mut(&mut self, index: NodeIndex) -> &mut bool {
			&mut self[index.0]
		}
	}
	
	impl Index<OpIndex> for Vec<Box<Operation>>{
		type Output = Box<Operation>;
		fn index(&self, index: OpIndex) -> &Box<Operation> {
			&self[index.0]
		}
	}

	impl IndexMut<OpIndex> for Vec<Box<Operation>>{
		fn index_mut(&mut self, index: OpIndex) -> &mut Box<Operation> {
			&mut self[index.0]
		}
	}
}

