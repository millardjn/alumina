pub mod dummy;
pub mod numeric_check;
pub mod loss;
pub mod nn;
pub mod math;
pub mod activ;
pub mod shape;
pub mod reduce;
pub mod regularisation;
pub mod fill;

use graph::{GraphDef, NodeID, DataID, OpID, PassID, OpTag, Storage, GraphShapes, Error, ErrorKind, Result};
use std::any::Any;
use std::fmt::Debug;


/// Generated default unique names for `Op`s
///
/// If a name isn't set on an `OpBuilder` a default name will be generated using the `type_name()` and the names of input and output nodes.
/// Similar for to: `format!("{}({},{}=>{},{}){}" type_name(), i, node1_name, node2_name, node3_name, node4_name)`
/// e.g. `Dummy0(node1,node2=>node3,node4)` where i is incremented until a unique/unused name is found.
pub fn standard_op_name<O: Op>(op: &O, name: &Option<String>, graph: &GraphDef, inputs: &[NodeID], outputs: &[NodeID]) -> String {

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

/// Generated default names for `Pass`s
///
/// Names for passes may not be unique.
/// A default name will be generated using the `type_name()` and the names of input and output data.
/// Similar for to: `format!("{}({},{}=>{},{}){}" type_name(), data1_name, data2_name, data3_name, data4_name)`
pub fn standard_pass_name(pass: &Pass, graph: &GraphDef, inputs: &[DataID], outputs: &[DataID]) -> String {

	let mut name_string = pass.type_name().to_string();
	name_string.push_str("(");
	let mut input_names = inputs.iter().map(|id| graph.data_name(id));
	if let Some(name) = input_names.next(){
		name_string.push_str(&name);
		for name in input_names {
			name_string.push_str(",");
			name_string.push_str(&name);
		}
	}
	name_string.push_str("=>");
	let mut output_names = outputs.iter().map(|id| graph.data_name(id));
	if let Some(name) = output_names.next(){
		name_string.push_str(&name);
		for name in output_names {
			name_string.push_str(",");
			name_string.push_str(&name);
		}
	}
	name_string.push_str(")");
	name_string
}

/// Generated default names for parameter nodes created by `OpBuilder`s
///
/// Names are generated from the name of the `Op` returned by the `OpBuilder`, using the following: `format!("P{}_{}", i, builder_name)`
/// e.g. `P0_Dummy0(node1,node2=>node3,node4)`, where `i` is incremented until an unused name is found.
pub fn standard_inner_parameter_name(builder_name: &str, graph: &mut GraphDef) -> String {
	let mut i = 0;
	loop {
		let next_param_name = format!("P{}_{}", i, builder_name);
		let result = graph.node_id(&*next_param_name);
		i += 1;
		if matches!(result, Err(Error(ErrorKind::ZeroNodesMatchTag(_), _))) {
			return next_param_name;
		}
	}

}

/// Generated default names for non-parameter nodes created by `OpBuilder`s
///
/// Names are generated from the name of the `Op` returned by the `OpBuilder`, using the following: `format!("N{}_{}", i, builder_name)`
/// e.g. `N0_Dummy0(node1,node2=>node3,node4)`, where `i` is incremented until an unused name is found.
pub fn standard_inner_node_name(builder_name: &str, graph: &mut GraphDef) -> String {
	let mut i = 0;
	loop {
		let next_param_name = format!("N{}_{}", i, builder_name);
		let result = graph.node_id(&*next_param_name);
		i += 1;
		if matches!(result, Err(Error(ErrorKind::ZeroNodesMatchTag(_), _))) {
			return next_param_name;
		}
	}

}

/// The `Op` trait provides a builder interface for differentible operations to be added to a `GraphDef`.
///
/// The `Op` Trait is responsible for:
/// * providing a builder interface
/// * producing an `OpInstance` which provides shape inference
/// * modifying the `GraphDef` by adding `Pass`s, `Op`s, and `Node`s to ensure the correct value and gradient calculations take place
pub trait Op: Any {
	type InstanceType: OpInstance;

	/// The type name of an `Op` is used to construct the default instance name of the `OpInstance` returned by `build()`.
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
	/// Also used to let an `Op` create parameter nodes as necessary.
	fn build(self, graph: &mut GraphDef, op_id: &OpID) -> Result<Self::InstanceType>;

	/// A convenience method which just calls GraphDef::new_op(..)
	fn add_to(self, graph: &mut GraphDef, tags: Vec<OpTag>) -> Result<OpID> where Self: Sized{
		graph.new_op(self, tags)
	}
}

/// The `OpInstance` trait is used to record each `Op` that has been added to a `GraphDef`.
///
/// An OpInstance is produced when `build()` is called on an Op
pub trait OpInstance: Any + OpClone + OpAny + Debug{

	/// The name of this instance of the Op
	fn instance_name(&self) -> &str;
	
	/// Returns the (input, output) nodeIDs of the nodes use when creating this Op. This does not include nodes created by this Op which are returns in `inner_nodes()`.
	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>);

	/// Returns the `Pass`s used directly to implement this Op
	fn inner_passes(&self) -> Vec<PassID>;

	/// Returns the Ops used to implement this Op.
	///
	/// Passes created by these ops are not returned by `inner_passes()`,
	/// nodes created by these ops are not returned by `inner_nodes()`.
	fn inner_ops(&self) -> Vec<OpID>;

	/// Returns the Nodes created when this Op was built
	fn inner_nodes(&self) -> Vec<NodeID>;

	/// TODO
	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>;
}


pub trait Pass: Any + PassClone + Debug {
	/// The name of the `Pass` type
	fn type_name(&self) -> &'static str;

	/// This name may not be unique, unlike node and op names
	fn instance_name(&self, graph: &GraphDef) -> String where Self:Pass{
		let (inputs, outputs) = self.dependencies();

		let mut name_string = self.type_name().to_string();
		name_string.push_str("(");
		let mut input_names = inputs.iter().map(|id| graph.data_name(id));
		if let Some(name) = input_names.next(){
			name_string.push_str(&name);
			for name in input_names {
				name_string.push_str(",");
				name_string.push_str(&name);
			}
		}
		name_string.push_str("=>");
		let mut output_names = outputs.iter().map(|id| graph.data_name(id));
		if let Some(name) = output_names.next(){
			name_string.push_str(&name);
			for name in output_names {
				name_string.push_str(",");
				name_string.push_str(&name);
			}
		}
		name_string.push_str(")");
		name_string
	}

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
	fn run (&self, data: &Storage) -> Result<Box<Any>>;
}


pub trait OpAny {
	fn as_any(&self) -> &Any;
}

impl<T> OpAny for T where T: 'static + OpInstance {
	fn as_any (&self) -> &Any {
		self
	}
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


#[must_use]
#[derive(Clone, Debug)]
pub struct NoOp {
	name: Option<String>
}

impl NoOp {
	pub fn new() -> Self {
		NoOp{name: None}
	}
}

impl Op for NoOp {
	type InstanceType = NoOpInstance;

	fn type_name(&self) -> &'static str {"NoOp"}

	fn name<T: Into<String>>(mut self, name: T) -> Self {
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
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

	fn instance_name(&self) -> &str {&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){(vec![], vec![])}

	fn inner_passes(&self) -> Vec<PassID> {vec![]}

	fn inner_ops(&self) -> Vec<OpID> {vec![]}

	fn inner_nodes(&self) -> Vec<NodeID> {vec![]}

	fn propagate_shape_constraints(&self, _shapes: &mut GraphShapes) -> Result<()>{Ok(())}
}

#[derive(Clone, Debug)]
pub struct NoOpPass {}

impl NoOpPass {
	pub fn new() -> Self {
		NoOpPass {}
	}
}

impl Pass for NoOpPass {
	fn type_name(&self) -> &'static str {"NoOpPass"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![], vec![])
	}

	fn run (&self, _data: &Storage) -> Result<Box<Any>>{
		Ok(Box::new(()))
	}
}

#[test]
fn test_name_generation(){
	_test_name_generation().unwrap();
}

fn _test_name_generation() -> Result<()>{
	use ops::dummy::Dummy;
	use graph::GraphDef;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![Unknown, 5, 16], "node1", tag![])?;
	let node2 = g.new_node(shape![Unknown, 5, 16], "node2", tag![])?;
	let node3 = g.new_node(shape![Unknown, 5, 16], "node3", tag![])?;
	let node4 = g.new_node(shape![Unknown, 5, 16], "node4", tag![])?;

	let o1 = g.new_op(Dummy::new().input(&node1).input(&node2).output(&node3).output(&node4), tag![])?;

	assert_eq!("Dummy0(node1,node2=>node3,node4)", g.op_name(&o1));

	Ok(())
}