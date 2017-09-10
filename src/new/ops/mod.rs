pub mod dummy;

use new::graph::{NodeID, DataID, Storage, GraphShapes};
use new::graph;
use new::shape::NodeShape;
use std::any::Any;
use std::fmt::Debug;

pub trait OperationBuilder: Any + Default {
	type OperationType: Operation;

	/// Supply name for operation
	fn name<T: Into<String>>(self, name: T) -> Self;

	/// Called by graph::Builder to construct the operation instance
	/// Arbitrary graph modification occur allowing 
	/// Used to let Operations create parameter nodes as necessary,
	/// or to implement operations with are compositions of smaller operations
	fn build(self, &mut graph::Builder) -> Self::OperationType;
}

pub trait Operation: OperationClone + Any + Debug{

	fn instance_name(&self) -> &str;
	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes);
	
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
	fn get_meta(&self) -> &OperatorMetaData;
	
	/// Returns the nodeIDs (inputs, outputs) of the nodes
	fn operation_dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>);

	/// Returns the DataIDs of the (inputs, outputs) of the forward pass
	/// where the inputs are a subset of the operation input nodes values
	/// and where the outputs are a subset of the operations output node values
	fn forward_dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		// the default implementation returns the largest possible set of DataIDs based on the operation dependancies
		// if these arent used in the pass then overriding this will allow for earlier deallocation of the tensors
		let (op_inputs, op_outputs) = self.operation_dependencies();
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
		let (op_inputs, op_outputs) = self.operation_dependencies();
		(
			op_inputs.iter().map(|in_node| in_node.value_id()).chain(op_outputs.iter().map(|out_node| out_node.gradient_id())).collect(),
			op_inputs.iter().map(|in_node| in_node.gradient_id()).collect()
		)
	}

	/// should update output node values based on input node values. Must use += when writing to output node.
	fn forward (&mut self, data: &mut Storage);
	
	/// Should calculate error gradient contribution of operation to the input node and parameters based on the output node derivatives.
	/// Each operation will be passed its relevant slice for params and param_derivs
	/// Note: all calculations should use += as to not overwrite other operations contributions,
	/// and in the case of data shape n>1 the sum of parameter gradients from all individual examples should be accumulated in param_deriv and error
	/// the graph will later divide by n to get the mean error and error derivatives.
	fn backward (&mut self, data: &mut Storage);
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
pub struct OperatorMetaData {
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

	fn propagate_shape_constraints(&self, _shapes: &mut GraphShapes){}
			
	fn get_meta(&self) -> &OperatorMetaData{
		unimplemented!()
	}
	
	fn operation_dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		(vec![], vec![])
	}

	fn forward (&mut self, _data: &mut Storage){}
	
	fn backward (&mut self, _data: &mut Storage){}
}


// mod test {
// 	use new::graph::{NodeID, Storage, GraphShapes};
// 	use new::graph;
// 	use super::*;


// }