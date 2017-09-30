pub mod dummy;
pub mod numeric_check;
pub mod loss;
// pub mod nn;
// pub mod math;

use new::graph::{GraphDef, NodeID, DataID, OpID, PassID, Storage, GraphShapes, Error, ErrorKind, Result};
use std::any::Any;
use std::fmt::Debug;


/// Generated default names for `Op`s
///
/// If a name isn't set on an `OpBuilder` a default name will be generated using the `type_name()` and the names of input and output nodes.
/// Similar for to: `format!("{}({},{}=>{},{}){}" type_name(), i, node1_name, node2_name, node3_name, node4_name)`
/// e.g. `Dummy0(node1,node2=>node3,node4)` where i is incremented until an unused name is found.
pub fn standard_op_name<O: Op>(op: &O, name: &Option<String>, graph: &mut GraphDef, inputs: &[NodeID], outputs: &[NodeID]) -> String {

	if let Some(name) = name.as_ref() {
		return name.clone();
	}

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
		let next_op_name = format!("{}{}{}", op.type_name(), i, node_string);
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


pub trait Op: Any {
	type InstanceType: OpInstance;

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
	fn build(self, graph: &mut GraphDef, op_id: &OpID) -> Result<Self::InstanceType>;
}


pub trait OpInstance: Any + OpClone + Debug{
	/// The name of the `Op` type
	fn type_name(&self) -> &'static str;

	/// The name of this instance of the Op
	fn instance_name(&self) -> &str;

	// TODO consider whether non standard order ops may be useful
	//fn requires_standard_order_inputs(&self) -> bool {true}
	
	/// Returns the nodeIDs (inputs, outputs) of the nodes
	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>);

	fn inner_passes(&self) -> Vec<PassID>;

	fn inner_ops(&self) -> Vec<OpID>;

	fn inner_nodes(&self) -> Vec<NodeID>;

	/// TODO
	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>;
}


pub trait Pass: Any + PassClone + Debug {
	
	// TODO consider mutable access so the graph con modify passes:
	// this would force inputs and outputs to be stored sequentially for all ops. maybe a mut iter is better?
	// fn dependencies_mut(&mut self) -> (&mut[DataID], &mut[DataID]);

	/// Returns the DataIDs of the (inputs, outputs) of the pass
	///
	/// where the inputs are a subset of the Op input nodes values, or internal nodes values
	/// and where the outputs are a subset of the Ops output node values
	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>);

	/// Performs the computation and runtime checks
	///
	/// This should either:
	///
	/// * Update output node values based on input node values, or
	///
	/// * Calculate error gradient contribution of Op to the input nodes based on the output node derivatives.
	///
	/// Note: all calculations should output using += as to not overwrite other `Pass`'s contributions.
	///
	/// The return of Box<Any> can be used to store arbitrary data for retreival from `Storage` by another pass, e.g. a dropout mask.
	/// Most passes will simply return nothing: `Ok(Box::new(()))`.
	fn run (&self, data: &mut Storage) -> Result<Box<Any>>;
}


/// Cloneable trait object workaround from DK : http://stackoverflow.com/questions/30353462/how-to-clone-a-struct-storing-a-trait-object
pub trait OpClone {
	fn clone_box(&self) -> Box<OpInstance>;
}

impl<T> OpClone for T where T: 'static + OpInstance + Clone {
	fn clone_box(&self) -> Box<OpInstance> {
		Box::new(self.clone())
	}
}

impl Clone for Box<OpInstance> {
	fn clone(&self) -> Box<OpInstance> {
		self.clone_box()
	}
}


/// Cloneable trait object workaround from DK : http://stackoverflow.com/questions/30353462/how-to-clone-a-struct-storing-a-trait-object
pub trait PassClone {
	fn clone_box(&self) -> Box<Pass>;
}

impl<T> PassClone for T where T: 'static + Pass + Clone {
	fn clone_box(&self) -> Box<Pass> {
		Box::new(self.clone())
	}
}

impl Clone for Box<Pass> {
	fn clone(&self) -> Box<Pass> {
		self.clone_box()
	}
}




pub struct NoOp {
	name: Option<String>
}

impl Op for NoOp {
	type InstanceType = NoOpInstance;

	fn type_name(&self) -> &'static str {"NoOp"}

	fn name<T: Into<String>>(mut self, name: T) -> Self {
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, op_id: &OpID) -> Result<Self::InstanceType> {
		let name = standard_op_name(&self, &self.name, graph, &[], &[]);
		Ok(NoOpInstance{name})
	}
}

/// An OpInstance which does nothing
#[derive(Clone, Debug)]
pub struct NoOpInstance {
	pub name: String
}

impl OpInstance for NoOpInstance {

	fn type_name(&self) -> &'static str {"NoOp"}
	
	fn instance_name(&self) -> &str {&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){(vec![], vec![])}

	fn inner_passes(&self) -> Vec<PassID> {vec![]}

	fn inner_ops(&self) -> Vec<OpID> {vec![]}

	fn inner_nodes(&self) -> Vec<NodeID> {vec![]}

	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{Ok(())}
}


#[test]
fn test_name_generation(){
	_test_name_generation().unwrap();
}

fn _test_name_generation() -> Result<()>{
	use new::ops::dummy::Dummy;
	use new::graph::GraphDef;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![Unknown, 5, 16], "node1", tag![])?;
	let node2 = g.new_node(shape![Unknown, 5, 16], "node2", tag![])?;
	let node3 = g.new_node(shape![Unknown, 5, 16], "node3", tag![])?;
	let node4 = g.new_node(shape![Unknown, 5, 16], "node4", tag![])?;


	let o1 = g.new_op(Dummy::new().input(&node1).input(&node2).output(&node3).output(&node4), tag![])?;

	assert_eq!("Dummy0(node1,node2=>node3,node4)", g.op_name(&o1));

	Ok(())
}