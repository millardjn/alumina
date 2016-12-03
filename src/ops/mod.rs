
#[macro_use]pub mod math;

pub mod activ;
pub mod conv;
pub mod reshape;
pub mod basic;
pub mod loss;


// shapes
// shapes have a mandatory 'depth' width correspons to the size of the columns in nodes
// shapes also have a fixed number of higher dimensions
// A MLP layer would have a fixed depth and no higher dimensions
// and NxM RGB image would be an shape of depth 3 and higher dimensions N and M
// a convoluton operation can be defined with a single


//###### Op Signatures - [N|P][HI|SI][HO|SO][D|F|U]
// N - nonparameterised
// P - parameterised
// HI - hard shape input at graph construction (all dimensions fixed)
// SI - soft shape input at graph construction (nonfixed dimensions)
// HO - hard shape output at graph construction (all dimensions fixed)
// SO - soft shape output at graph construction (nonfixed dimensions)
// D - will impose a number of dimenions and associated sizes at evaluation time
// F - will impose only a required flat size at evaluation time
// U - will impose only a required depth at evaluation time


// PseudoBatchNorm     - PHIHOF - imposes new error gradients which push the avtications toward a normalised gaussian distribution.
// ConvPsuedoBatchNorm - PSISOD - as above but has only enough parameters for shape depth, and resuses them by scanning across higher dimensions

use graph::*;
use shape::*;
use std::cell::RefCell;
use std::sync::Arc;

#[allow(non_snake_case)]
pub trait Operation: OperationClone {
	fn name(&self) -> &str;
	//fn get_initial_params(&self, params: &mut[f32]);
	
	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]);
	fn num_params(&self) -> usize;
	fn input_node_IDs(&self) -> Vec<NodeID>;
	fn output_node_IDs(&self) -> Vec<NodeID>; 
	fn init_params(&mut self, params: &mut [f32]){
		if self.num_params() == 0 {
			if params.len() != 0 {
				panic!("init_params passed non-zero length slice for an operation with no parameters");
			}
		} else {
			unimplemented!();
		}
	}
	
	/// should update output node values based on input node values. Must use += when writing to output node.
	fn forward (&mut self, data: &mut [RefCell<NodeData>], params: &[f32]);
	
	/// Should calculate error gradient contribution of operation to the input node and parameters based on the output node derivatives.
	/// Each operation will be passed its relevant slice for params and param_derivs
	/// Note: all calculations should use += as to not overwrite other operations contributions,
	/// and in the case of data shape n>1 the sum of parameter gradients from all individual examples should be accumulated in param_deriv and error
	/// the graph will later divide by n to get the mean error and error derivatives.
	
	fn backward (&mut self, data: &mut [RefCell<NodeData>], params: &[f32], param_deriv: &mut [f32], error: &mut f32);
	
}

pub fn init_fill<O: Operation>(filler: f32) -> Arc<Fn(&O, &mut [f32])> {
	Arc::new(
		move |_op: &O, params: &mut [f32]| {
			for x in params {
				*x = filler;
			}
		}
	)
}

pub fn init_dummy<O: Operation>() -> Arc<Fn(&O, &mut [f32])> {
	Arc::new(move |_op, _params| {})
}

#[derive(Clone)]
pub enum ParamSharing {
	/// Let the operation choose something compatible if possible
	Auto,

	/// Parameters are shared across all spatial dimensions and channels
	Full,

	/// Number of params is proportional to channels, but shared across spatial dimensions
	Spatial,
	
	// Shared across some spatial dimensions, true corresponds to dimensions with sharing.
	// PartialSpatial(Vec<bool>), TODO. would be annoying to support on many functions

	/// No sharing. Number of params is proportional to the flat size
	None,
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
