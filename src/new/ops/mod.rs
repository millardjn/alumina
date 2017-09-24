pub mod dummy;
pub mod broadcast;
pub mod bias;
pub mod numeric_check;
pub mod loss;

use new::graph::{GraphDef, NodeID, DataID, Storage, GraphShapes, Error, ErrorKind, Result};
use new::shape::NodeShape;
use std::any::Any;
use std::fmt::Debug;

/// Generated default names for `Op`s
///
/// If a name isn't set on an `OpBuilder` a default name will be generated using the `type_name()` and the names of input and output nodes.
/// Similar for to: `format!("{}({},{}=>{},{}){}" type_name(), i, node1_name, node2_name, node3_name, node4_name)`
/// e.g. `Dummy0(node1,node2=>node3,node4)` where i is incremented until an unused name is found.
pub fn standard_op_name<B: OpBuilder>(builder: &B, graph: &mut GraphDef, inputs: &[NodeID], outputs: &[NodeID]) -> String {

	let mut node_string = "(".to_string();
	let mut input_names = inputs.iter().map(|id| graph.node_name(id));
	if let Some(name) = input_names.next(){
		node_string.push_str(name);
		for name in input_names {
			node_string.push_str(",");
			node_string.push_str(name);
		}
	}
	node_string.push_str("=>");
	let mut output_names = outputs.iter().map(|id| graph.node_name(id));
	if let Some(name) = output_names.next(){
		node_string.push_str(name);
		for name in output_names {
			node_string.push_str(",");
			node_string.push_str(name);
		}
	}
	node_string.push_str(")");


	let mut i = 0;
	loop {
		let next_op_name = format!("{}{}{}", builder.type_name(), i, node_string);
		let result = graph.op_id(&*next_op_name);
		i += 1;
		if matches!(result, Err(Error(ErrorKind::ZeroOpsMatchTag(_), _))) {
			return next_op_name;
		}
	}
}

/// Generated default names for parameter nodes created by `OpBuilder`s
///
/// Names are generated from the name of the `Op` returned by the `OpBuilder`, using the following: `format!("P{}_{}", i, builder_name)`
/// e.g. `P0_Dummy0(node1,node2=>node3,node4)`, where `i` is incremented until an unused name is found.
pub fn standard_parameter_names(n: usize, builder_name: &str, graph: &mut GraphDef) -> Vec<String> {

	let mut names = vec![];

	let mut i = 0;
	while names.len() < n {
		let next_param_name = format!("P{}_{}", i, builder_name);
		let result = graph.node_id(&*next_param_name);
		i += 1;
		if matches!(result, Err(Error(ErrorKind::ZeroNodesMatchTag(_), _))) {
			names.push(next_param_name);
		}
	}

	names
}

pub trait OpBuilder: Any {
	type OpType: Op;

	/// The type name of an `OpBuilder` is used to construct the default instance name of the `Op` returned by `build()`.
	///
	/// This should be the same as the part of the type name that comes before "Builder".
	/// This name may not match the `type_name()` of the `Op` returned by `build()`
	/// or ops added to the graph by `build()`.
	fn type_name(&self) -> &'static str;

	/// Supply a name for this instance of the Op which will be returned by `build()`.
	///
	/// If a name isn't set, a default name will be generated using `standard_op_name()`
	fn name<T: Into<String>>(self, name: T) -> Self;

	/// Called by `GraphDef` to construct the `Op` instance.
	///
	/// Arbitrary graph modification may occur allowing builders to implement high level effects by composing multiple low level `Op`s.
	/// Also used to let Ops create parameter nodes as necessary.
	fn build(self, &mut GraphDef) -> Result<Self::OpType>;
}

//TODO: remove mutability of Ops in favor of state objects that implement Any
pub trait Op: OpClone + Any + Debug{
	/// The name of the `Op` type
	fn type_name(&self) -> &'static str;

	/// The name of this instance of the Op
	fn instance_name(&self) -> &str;

	/// TODO
	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>;
	
	// TODO sort out initilisation
	// fn num_params(&self) -> usize;

	// fn init_params(&mut self, params: &mut [f32]){
	// 	if self.num_params() == 0 {
	// 		assert_eq!(0, params.len(), "init_params passed non-zero length slice for an Op with no parameters");
	// 	} else {
	// 		unimplemented!();
	// 	}
	// }
	
	/// Returns the meta data
	//fn get_meta(&self) -> &OpMetaData;
	
	/// Returns the nodeIDs (inputs, outputs) of the nodes
	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>);

	/// Returns the DataIDs of the (inputs, outputs) of the forward pass
	/// where the inputs are a subset of the Op input nodes values
	/// and where the outputs are a subset of the Ops output node values
	fn forward_dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		// the default implementation returns the largest possible set of DataIDs based on the Op dependancies
		// if these arent used in the pass then overriding this will allow for earlier deallocation of the tensors
		let (op_inputs, op_outputs) = self.dependencies();
		(
			op_inputs.iter().map(|nodeid| nodeid.value_id()).collect(),
			op_outputs.iter().map(|nodeid| nodeid.value_id()).collect()
		)
	}

	/// Returns the DataIDs of the (inputs, outputs) of the backward pass
	/// where the inputs are a subset of the Op input nodes values, and output nodes gradients
	/// and where the outputs are a subset of the Ops input nodes gradients
	fn backward_dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		// the default implementation returns the largest possible set of DataIDs based on the Op dependancies
		// if these arent used in the pass then overriding this will allow for earlier deallocation of the tensors
		let (op_inputs, op_outputs) = self.dependencies();
		(
			op_inputs.iter().map(|in_node| in_node.value_id())
			.chain(op_outputs.iter().map(|out_node| out_node.gradient_id())).collect(),
			op_inputs.iter().map(|in_node| in_node.gradient_id()).collect()
		)
	}

	/// should update output node values based on input node values. Must use += when writing to output node.
	fn forward (&mut self, data: &mut Storage) -> Result<()>;
	
	/// Should calculate error gradient contribution of Op to the input node and parameters based on the output node derivatives.
	/// Each Op will be passed its relevant slice for params and param_derivs
	/// Note: all calculations should use += as to not overwrite other Ops contributions,
	/// and in the case of data shape n>1 the sum of parameter gradients from all individual examples should be accumulated in param_deriv and error
	/// the graph will later divide by n to get the mean error and error derivatives.
	fn backward (&mut self, data: &mut Storage) -> Result<()>;
}


/// Cloneable trait object workaround from DK : http://stackoverflow.com/questions/30353462/how-to-clone-a-struct-storing-a-trait-object
pub trait OpClone {
	fn clone_box(&self) -> Box<Op>;
}

impl<T> OpClone for T where T: 'static + Op + Clone {
	fn clone_box(&self) -> Box<Op> {
		Box::new(self.clone())
	}
}

impl Clone for Box<Op> {
	fn clone(&self) -> Box<Op> {
		self.clone_box()
	}
}


pub trait SimpleOpBuilder: OpBuilder {
	fn set_output(&mut self, id: &NodeID);
	fn required_output_shape(&self) -> NodeShape;
	//fn build_with_output() -> Self::OpType; TODO
}

pub trait SimpleOp: Op {

}


// based on metadata from: https://github.com/Metadiff/gir
// #[derive(Debug, Clone, PartialEq, Eq)]
// pub struct OpMetaData {
// 	pub name: &'static str,
// 	//pub arity: Arity,
// 	pub num_outputs: usize,
// 	pub differential_parents: usize,
// 	pub ordered_parents: bool,
// 	pub elementwise: bool,
// 	pub type_preserving: bool,
// 	pub reduction: bool,
// 	pub differentiable: bool,
// 	pub scalar_output: bool,
// 	pub shape_operator: bool,
// 	//pub fixed_output_type: Option<FundamentalType>,
// }




/// An Op which does nothing
/// Can be returned as an 
#[derive(Clone, Debug)]
pub struct NullOp {
	name: String
}

impl Op for NullOp {
	fn type_name(&self) -> &'static str {
		"Null"
	}

	fn instance_name(&self) -> &str {
		&self.name
	}

	fn propagate_shape_constraints(&self, _shapes: &mut GraphShapes) -> Result<()>{Ok(())}
			
	
	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		(vec![], vec![])
	}

	fn forward (&mut self, _data: &mut Storage) -> Result<()> {Ok(())}
	
	fn backward (&mut self, _data: &mut Storage) -> Result<()> {Ok(())}
}


#[test]
fn test_name_generation(){
	_test_name_generation().unwrap();
}

fn _test_name_generation() -> Result<()>{
	use new::ops::dummy;
	use new::graph::GraphDef;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![Unknown, 5, 16], "node1", tag![])?;
	let node2 = g.new_node(shape![Unknown, 5, 16], "node2", tag![])?;
	let node3 = g.new_node(shape![Unknown, 5, 16], "node3", tag![])?;
	let node4 = g.new_node(shape![Unknown, 5, 16], "node4", tag![])?;


	let o1 = g.new_op(dummy::Builder::new().input(&node1).input(&node2).output(&node3).output(&node4), tag![])?;

	assert_eq!("Dummy0(node1,node2=>node3,node4)", g.op_name(&o1));

	Ok(())
}