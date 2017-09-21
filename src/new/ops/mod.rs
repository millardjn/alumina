pub mod dummy;
pub mod broadcast;
pub mod bias;

use new::graph::{NodeID, DataID, Storage, GraphShapes, Error, ErrorKind, Result};
use new::graph;
use new::shape::NodeShape;
use std::any::Any;
use std::fmt::Debug;


pub fn op_name_gen(builder: &mut graph::Builder, op_type_name: &str, inputs: &[NodeID], outputs: &[NodeID]) -> String {
	let mut op_name = op_type_name.to_string();
	op_name.push_str("(");

	let mut input_names = inputs.iter().map(|id| builder.node_name(id));
	if let Some(name) = input_names.next(){
		op_name.push_str(name);
		for name in input_names {
			op_name.push_str(",");
			op_name.push_str(name);
		}
	}
	op_name.push_str("=>");
	let mut output_names = outputs.iter().map(|id| builder.node_name(id));
	if let Some(name) = output_names.next(){
		op_name.push_str(name);
		for name in output_names {
			op_name.push_str(",");
			op_name.push_str(name);
		}
	}
	op_name.push_str(")");

	// If name already exists (should be rare), append an integer. Try in ascending order
	let mut result = builder.get_op_id(&*op_name);
	let mut i = 1;
	while matches!(result, Err(Error(ErrorKind::StorageDataMarkedNotRequired, _))){
		let next_op_name = format!("{}{}", op_name, i);
		result = builder.get_op_id(&*next_op_name);
		i += 1;
	}
	
	op_name
}


//TODO: remove mutability of operations in favor of state objects that implement Any
pub trait OperationBuilder: Any {
	type OperationType: Operation;

	/// Supply name for operation
	fn name<T: Into<String>>(self, name: T) -> Self;

	/// Called by graph::Builder to construct the operation instance
	/// Arbitrary graph modification occur allowing 
	/// Used to let Operations create parameter nodes as necessary,
	/// or to implement operations with are compositions of smaller operations
	fn build(self, &mut graph::Builder) -> Result<Self::OperationType>;
}

pub trait Operation: OperationClone + Any + Debug{

	fn instance_name(&self) -> &str;
	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>;
	
	// TODO sort out initilisation
	// fn num_params(&self) -> usize;

	// fn init_params(&mut self, params: &mut [f32]){
	// 	if self.num_params() == 0 {
	// 		assert_eq!(0, params.len(), "init_params passed non-zero length slice for an operation with no parameters");
	// 	} else {
	// 		unimplemented!();
	// 	}
	// }
	
	/// Returns the meta data
	fn get_meta(&self) -> &OperationMetaData;
	
	/// Returns the nodeIDs (inputs, outputs) of the nodes
	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>);

	/// Returns the DataIDs of the (inputs, outputs) of the forward pass
	/// where the inputs are a subset of the operation input nodes values
	/// and where the outputs are a subset of the operations output node values
	fn forward_dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		// the default implementation returns the largest possible set of DataIDs based on the operation dependancies
		// if these arent used in the pass then overriding this will allow for earlier deallocation of the tensors
		let (op_inputs, op_outputs) = self.dependencies();
		(
			op_inputs.iter().map(|nodeid| nodeid.value_id()).collect(),
			op_outputs.iter().map(|nodeid| nodeid.value_id()).collect()
		)
	}

	/// Returns the DataIDs of the (inputs, outputs) of the backward pass
	/// where the inputs are a subset of the operation input nodes values, and output nodes gradients
	/// and where the outputs are a subset of the operations input nodes gradients
	fn backward_dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		// the default implementation returns the largest possible set of DataIDs based on the operation dependancies
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
	
	/// Should calculate error gradient contribution of operation to the input node and parameters based on the output node derivatives.
	/// Each operation will be passed its relevant slice for params and param_derivs
	/// Note: all calculations should use += as to not overwrite other operations contributions,
	/// and in the case of data shape n>1 the sum of parameter gradients from all individual examples should be accumulated in param_deriv and error
	/// the graph will later divide by n to get the mean error and error derivatives.
	fn backward (&mut self, data: &mut Storage) -> Result<()>;
}

pub trait SimpleOperationBuilder: OperationBuilder {
	fn set_output(&mut self, id: &NodeID);
	fn required_output_shape(&self) -> NodeShape;
	//fn build_with_output() -> Self::OperationType;
}

pub trait SimpleOperation: Operation {

}


// based on metadata from: https://github.com/Metadiff/gir
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationMetaData {
	pub name: &'static str,
	//pub arity: Arity,
	pub num_outputs: usize,
	pub differential_parents: usize,
	pub ordered_parents: bool,
	pub elementwise: bool,
	pub type_preserving: bool,
	pub reduction: bool,
	pub differentiable: bool,
	pub scalar_output: bool,
	pub shape_operator: bool,
	//pub fixed_output_type: Option<FundamentalType>,
}

// Cloneable trait object workaround from DK : http://stackoverflow.com/questions/30353462/how-to-clone-a-struct-storing-a-trait-object
pub trait OperationClone {
	fn clone_box(&self) -> Box<Operation>;
}

impl<T> OperationClone for T where T: 'static + Operation + Clone {
	fn clone_box(&self) -> Box<Operation> {
		Box::new(self.clone())
	}
}

impl Clone for Box<Operation> {
	fn clone(&self) -> Box<Operation> {
		self.clone_box()
	}
}


/// An operation which does nothing
/// Can be returned as an 
#[derive(Clone, Debug)]
struct NullOperation {
	
}

impl Operation for NullOperation {
	fn instance_name(&self) -> &str {
		"Null Operation"
	}

	fn propagate_shape_constraints(&self, _shapes: &mut GraphShapes) -> Result<()>{Ok(())}
			
	fn get_meta(&self) -> &OperationMetaData{
		unimplemented!()
	}
	
	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		(vec![], vec![])
	}

	fn forward (&mut self, _data: &mut Storage) -> Result<()> {Ok(())}
	
	fn backward (&mut self, _data: &mut Storage) -> Result<()> {Ok(())}
}


// mod test {
// 	use new::graph::{NodeID, Storage, GraphShapes};
// 	use new::graph;
// 	use super::*;


// }