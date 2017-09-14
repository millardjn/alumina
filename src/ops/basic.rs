

/// Operations that ignore shape


use graph::*;
use std::cell::RefCell;
use shape::NodeShape;
use matrixmultiply;
use ops::*;
use std::sync::Arc;

/// `LinearMap`      - PHIHOF - all to all, typical in MLP network
#[derive(Clone)] 
pub struct LinearMap {
	name: String,
	input_id: NodeID,
	output_id: NodeID,
	input_size: usize,
	output_size: usize,
	init_func: Arc<Fn(&LinearMap, &mut [f32])>,
}

const FC_ERR: &'static str = "Full Connection requires fully determined shapes for both input and output nodes";

impl LinearMap {
	pub fn new(input_id: &NodeID, output_id: &NodeID, name: &str, init_func: Arc<Fn(&LinearMap, &mut [f32])>) -> Box<LinearMap>{
		Box::new(LinearMap{
			name: name.to_string(),
			input_id: input_id.clone(),
			output_id: output_id.clone(),
			input_size: input_id.shape.force_flat_size().expect(FC_ERR),
			output_size: output_id.shape.force_flat_size().expect(FC_ERR),
			init_func: init_func
		})
	}
	
	pub fn new_default(input_id: &NodeID, output_id: &NodeID) -> Box<LinearMap>{
		LinearMap::new(input_id, output_id, "LinearMap", LinearMap::init_xavier())
	}
	
	pub fn init_xavier() -> Arc<Fn(&LinearMap, &mut [f32])> {
		LinearMap::init_msra(1.0)
	}
	
	pub fn init_msra(sd_multiplier: f32) -> Arc<Fn(&LinearMap, &mut [f32])> {
		Arc::new(
			move |op: &LinearMap, params: &mut [f32]| {
				let variance = 1.0/op.input_size as f32;
				math::random_vector::normal_fill(params, 0.0, sd_multiplier*variance.sqrt());
			}
		)
	}

	pub fn init_fill(filler: f32) -> Arc<Fn(&LinearMap, &mut [f32])> {
		Arc::new(
			move |_op: &LinearMap, params: &mut [f32]| {
				for x in params {
					*x = filler;
				}
			}
		)
	}
	
	pub fn init_zero_fill() -> Arc<Fn(&LinearMap, &mut [f32])> {
		LinearMap::init_fill(0.0)
	}
	
	pub fn init_unit_fill() -> Arc<Fn(&LinearMap, &mut [f32])> {
		LinearMap::init_fill(1.0)
	}

}

/// parameters are a row major matrix
impl Operation for LinearMap {

	fn name(&self) -> &str{&self.name}
	
	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_id.ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Input node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_id.ind].name, self.name));
		
		let in_err_msg = format!("Error: Operation '{}' error input Node '{}' size has changed since graph construction.", self.name, nodes[self.input_id.ind].name);
		let out_err_msg = format!("Error: Operation '{}' error output Node '{}' size has changed since graph construction.", self.name, nodes[self.output_id.ind].name);
		
		shapes[self.input_id.ind] = NodeShape::new_flat(self.input_size).merge(&shapes[self.input_id.ind])
			.expect(&in_err_msg);
			
		shapes[self.output_id.ind] = NodeShape::new_flat(self.output_size).merge(&shapes[self.output_id.ind])
			.expect(&out_err_msg);

	}
	
	fn input_node_IDs(&self) -> Vec<NodeID>{vec![self.input_id.clone()]}
	
	fn output_node_IDs(&self) -> Vec<NodeID>{vec![self.output_id.clone()]}
	
	fn init_params(&mut self, params: &mut [f32]){
		assert!(self.num_params() == params.len());
		self.init_func.as_ref()(&self, params);
	}
	
	fn num_params(&self) -> usize {self.input_size * self.output_size}
	
	fn forward (&mut self, data: &mut [RefCell<NodeData>], params: &[f32]){
		let input = &*{data[self.input_id.ind].borrow_mut()};
		let output = &mut *{data[self.output_id.ind].borrow_mut()};
		let in_size = input.shape.flat_size_single();
		let out_size = output.shape.flat_size_single();
		// These checks shouldnt be necessary unless code in the graph doesnt correctly resolve compatible shapes.
		assert!(input.shape.n == output.shape.n);
		assert!(in_size == self.input_size);
		assert!(out_size == self.output_size);
		
		
		let m = out_size;
		let n = input.shape.n;
		let k = in_size;
		
		unsafe{
			matrixmultiply::sgemm(m, k, n,
				1.0,
				params.as_ptr(), k as isize, 1, // A is params, row major
				input.values.as_ptr(), 1, k as isize, // B, input values column major
				1.0,
				output.values.as_mut_ptr(), 1, m as isize); // C output values volumn major
		}


	}
	
	fn backward (&mut self, data: &mut [RefCell<NodeData>], params: &[f32], param_deriv: &mut [f32], _error: &mut f32){
		let input = &mut *{data[self.input_id.ind].borrow_mut()};
		let output = &*{data[self.output_id.ind].borrow_mut()};
		let in_size = input.shape.flat_size_single();
		let out_size = output.shape.flat_size_single();
		// These checks shouldnt be necessary unless code in the graph doesnt correctly resolve compatible shapes.
		assert!(input.shape.n == output.shape.n);
		assert!(in_size == self.input_size);
		assert!(out_size == self.output_size);	
				

		// A B C  and M N K are defines off the forward pass.		
		let m = out_size;
		let n = input.shape.n;
		let k = in_size;
		
		unsafe{
			let m1 = k;
			let n1 = n;
			let k1 = m;
			// input derivatives
			matrixmultiply::sgemm(m1, k1, n1,
				1.0,

				params.as_ptr(), 1, k as isize, // At is params, transposed ( treat as column major
				output.derivatives.as_ptr(), 1, m as isize, // C' output derives, column major
				1.0,
				input.derivatives.as_mut_ptr(), 1, k as isize); // B' input derives, colum major
			
			let k2 = n;
			let m2 = m;
			let n2 = k;
			// parameter derivatives
			matrixmultiply::sgemm(m2, k2, n2,
				1.0,
				output.derivatives.as_ptr(), 1, m as isize, // C' output derives, column major
				input.values.as_ptr(), k as isize, 1, // Bt, input values, transposed (treat as row major)
				1.0,
				param_deriv.as_mut_ptr(), k as isize, 1); // A' parameter derivatives, row major
		}
					
		
	}	
}

// /// Bias      - PHIHOF - all to all, typical in MLP network
#[derive(Clone)] 
pub struct Bias {
	name: String,
	output_id: NodeID,
	sharing: ParamSharing,
	num_params: usize,
	init_func: Arc<Fn(&Bias, &mut [f32])>,
}

impl Bias {
	pub fn new(output_id: &NodeID, sharing: ParamSharing, name: &str, init_func: Arc<Fn(&Bias, &mut [f32])>) -> Box<Bias>{
		
		let (sharing, num_params) = match sharing {
			ParamSharing::Auto => {
				(if output_id.shape.rank() == 1 {
					ParamSharing::None
				} else {
					ParamSharing::Spatial
				}, output_id.shape.channels)
			},
			ParamSharing::None => (ParamSharing::None, output_id.shape.force_flat_size().expect("Bias with 'None' parameter sharing requires a fully determined shape for the input node")),
			ParamSharing::Full => (ParamSharing::Full, 1),
			ParamSharing::Spatial => (ParamSharing::Spatial, output_id.shape.channels),
		};
		
		Box::new(Bias{
			name: name.to_string(),
			output_id: output_id.clone(),
			num_params: num_params,
			sharing: sharing,
			init_func: init_func,
		})
	}

	pub fn new_default(output_id: &NodeID,) -> Box<Bias>{
		Bias::new(output_id, ParamSharing::Auto, "Bias", init_fill(0.0))
	}

}

impl Operation for Bias {
	
	fn name(&self) -> &str{ &self.name }

	fn propagate_shape_constraints(&self, _nodes: &[Node], _shapes: &mut [NodeShape]){
		
	}
	
	fn num_params(&self) -> usize{ self.num_params }
	
	fn input_node_IDs(&self) -> Vec<NodeID>{ vec![] }
	
	fn output_node_IDs(&self) -> Vec<NodeID>{ vec![self.output_id.clone()] }
	
	fn init_params(&mut self, params: &mut [f32]){
		assert!(self.num_params() == params.len());
		
		self.init_func.as_ref()(&self, params);
	}
	
	fn forward (&mut self, data: &mut [RefCell<NodeData>], params: &[f32]){
		let output = &mut *{data[self.output_id.ind].borrow_mut()};

		let len = output.shape.flat_size_all();

		let stride = match self.sharing {
			ParamSharing::None => output.shape.flat_size_single(),
			ParamSharing::Spatial => output.shape.channels,
			ParamSharing::Full => 1,
			_ => unreachable!(),
		};

		if stride == 1 {
			let out_n = &mut output.values[..len];
			let param = params[0];
			
			for i in 0..len {
				out_n[i] += param;
			}				
		} else {

			for n_ind in 0..len/stride{
				let out_n = &mut output.values[n_ind*stride..][..stride];
				let params = &params[..stride];
				
				for i in 0..stride {
					out_n[i] += params[i];
				}			
				
			}

		}
		
	}
	
	fn backward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32], param_deriv: &mut [f32], _error: &mut f32){
		let output = &*{data[self.output_id.ind].borrow_mut()};

		let len = output.shape.flat_size_all();

		let stride = match self.sharing {
			ParamSharing::None => output.shape.flat_size_single(),
			ParamSharing::Spatial => output.shape.channels,
			ParamSharing::Full => 1,
			_ => unreachable!(),
		};


		if stride == 1 {
			let outd_n = &output.derivatives[..len];
			let param_deriv = &mut param_deriv[0];	
				
			for i in 0..len {
				*param_deriv += outd_n[i];
			}
			
		} else {
			for n_ind in 0..len/stride{
				let outd_n = &output.derivatives[n_ind*stride..][..stride];
				let param_deriv = &mut param_deriv[..stride];	
					
				for i in 0..stride {
					param_deriv[i] += outd_n[i];
				}
			}
		}

	}
}

/// `L2Regularisation`   - NSISOF - This operation does not provide gradients but rather just contributes a 1 or 0 to error depending if the max value in input matches the max value in target
#[derive(Clone)] 
pub struct L2Regularisation {
	name: String,
	num_params: usize,
	target_values: Option<Vec<f32>>,
	strength: f32,
}

impl L2Regularisation {
	pub fn new(op_id: &OpID, strength: f32, name: &str) -> Box<L2Regularisation>{
		Box::new(L2Regularisation{
			name: name.to_string(),
			num_params: op_id.num_params,
			target_values: None,
			strength: strength,
		})
	}

	pub fn new_default(op_id: &OpID) -> Box<L2Regularisation>{
		L2Regularisation::new(op_id, 1.0, "L2Regularisation")
	}
	
}

impl Operation for L2Regularisation {
	
	fn name(&self) -> &str{ &self.name }

	fn propagate_shape_constraints(&self, _nodes: &[Node], _shapes: &mut [NodeShape]){}
	
	fn num_params(&self) -> usize{self.num_params}
	
	fn input_node_IDs(&self) -> Vec<NodeID>{ vec![] }
	
	fn output_node_IDs(&self) -> Vec<NodeID>{ vec![] }
		
	fn init_params(&mut self, _params: &mut [f32]){
		panic!("L2Regularisation is not able to initialise parameters, and therefore should only be added to a graph as a secondary operation.");
	}

	fn forward (&mut self, _data: &mut [RefCell<NodeData>], _params: &[f32]){}
	
	fn backward (&mut self, _data: &mut [RefCell<NodeData>], params: &[f32], param_deriv: &mut [f32], error: &mut f32){

		if let Some(ref values) = self.target_values {
			debug_assert_eq!(values.len(), params.len());
			math::ssd_error(params, values, self.strength, param_deriv, error);
		} else {
			math::ssd_error(params, &vec![0.0; params.len()], self.strength, param_deriv, error);
		}

	}
}

// TODO update constructors and include some way to determine the level of parameter sharing.
/// Scale  - PSISOF - all elements are scaled by the same parameter
#[derive(Clone)] 
pub struct Scale {
	name: String,
	input_id: NodeID,
	output_id: NodeID,
	sharing: ParamSharing,
	num_params: usize,
	init_func: Arc<Fn(&Scale, &mut [f32])>,
}

impl Scale {
	pub fn new(input_id: &NodeID, output_id: &NodeID, sharing: ParamSharing, name: &str, init_func: Arc<Fn(&Scale, &mut [f32])>) -> Box<Scale>{
		let (sharing, num_params) = match sharing {
			ParamSharing::Auto => {
				if let Ok(size) = input_id.shape.force_flat_size() {
					(ParamSharing::None, size)
				} else {
					assert_eq!(input_id.shape.rank(), output_id.shape.rank());
					(ParamSharing::Spatial, input_id.shape.channels)
				}
			},
			ParamSharing::None => (ParamSharing::None, input_id.shape.force_flat_size().expect("Scale Operation with 'None' parameter sharing requires a fully determined shape for the input node")),
			ParamSharing::Full => (ParamSharing::Full, 1),
			ParamSharing::Spatial => (ParamSharing::Spatial, input_id.shape.channels),
		};
		
		Box::new(Scale{
			name: name.to_string(),
			input_id: input_id.clone(),
			output_id: output_id.clone(),
			sharing: sharing,
			num_params: num_params,
			init_func: init_func,
		})
	}
	
	pub fn new_default(input_id: &NodeID, output_id: &NodeID,) -> Box<Scale>{
		Scale::new(input_id, output_id, ParamSharing::Auto, "Scale", init_fill(1.0))
	}
	
}

impl Operation for Scale {

	fn name(&self) -> &str{&self.name}
	
	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_id.ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Input node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_id.ind].name, self.name));
		
		shapes[self.output_id.ind] = shapes[self.input_id.ind].merge(&shapes[self.output_id.ind])
			.expect(&format!("Error: Operation '{}' error could not merge input shape with existing shape for output Node '{}'", self.name, nodes[self.output_id.ind].name));
	}
	
	fn input_node_IDs(&self) -> Vec<NodeID>{vec![self.input_id.clone()]}
	
	fn output_node_IDs(&self) -> Vec<NodeID>{vec![self.output_id.clone()]}
	
	fn num_params(&self) -> usize {self.num_params}
	
	fn init_params(&mut self, params: &mut [f32]){
		assert!(self.num_params() == params.len());
		self.init_func.as_ref()(&self, params);
	}
	
	fn forward (&mut self, data: &mut [RefCell<NodeData>], params: &[f32]){
		let input = &*{data[self.input_id.ind].borrow()};
		let output = &mut *{data[self.output_id.ind].borrow_mut()};
		
		// These checks shouldnt be necessary unless code in the graph/propagate_shapes doesnt correctly resolve compatible shapes.
		debug_assert_eq!(input.shape.n, output.shape.n);
		debug_assert_eq!(input.shape.flat_size_single(), output.shape.flat_size_single());

		let len = input.shape.flat_size_all();

		let stride = match self.sharing {
			ParamSharing::None => input.shape.flat_size_single(),
			ParamSharing::Spatial => input.shape.channels,
			ParamSharing::Full => 1,
			_ => unreachable!(),
		};

		if stride == 1 {
			debug_assert_eq!(params.len(), 1);
			let out_n = &mut output.values[..len];
			let inp_n = &input.values[..len];
			let scale = &params[0];
			
			for i in 0..len {
				let v = inp_n[i];
				out_n[i] += v*scale;
			}				
		} else {

			for n_ind in 0..len/stride{
				let out_n = &mut output.values[n_ind*stride..][..stride];
				let inp_n = &input.values[n_ind*stride..][..stride];
				let params = &params[..stride];
				
				
				for i in 0..stride {
					let v = inp_n[i];
					out_n[i] += v*params[i];
				}			
				
			}

		}

	}
	
	fn backward (&mut self, data: &mut [RefCell<NodeData>], params: &[f32], param_deriv: &mut [f32], _error: &mut f32){
		let input = &mut *{data[self.input_id.ind].borrow_mut()};
		let output = &*{data[self.output_id.ind].borrow()};

		// These checks shouldnt be necessary unless code in the graph/propagate_shapes doesnt correctly resolve compatible shapes.
		debug_assert_eq!(input.shape.n, output.shape.n);
		debug_assert_eq!(input.shape.flat_size_single(), output.shape.flat_size_single());
		

		let len = input.shape.flat_size_all();

		let stride = match self.sharing {
			ParamSharing::None => input.shape.flat_size_single(),
			ParamSharing::Spatial => input.shape.channels,
			ParamSharing::Full => 1,
			_ => unreachable!(),
		};


		if stride == 1 {
			debug_assert_eq!(params.len(), 1);
			let inp_n = &input.values[..len];
			let outd_n = &output.derivatives[..len];
			let inpd_n = &mut input.derivatives[..len];
			let scale = &params[0];
			let param_deriv = &mut param_deriv[0];	
				
			for i in 0..len {
				inpd_n[i] += outd_n[i] * scale;
				*param_deriv += outd_n[i] * inp_n[i];
			}
			
		} else {

			for n_ind in 0..len/stride{

				let inp_n = &input.values[n_ind*stride..][..stride];
				let outd_n = &output.derivatives[n_ind*stride..][..stride];
				let inpd_n = &mut input.derivatives[n_ind*stride..][..stride];
				let params = &params[..stride];
				let param_deriv = &mut param_deriv[..stride];	
					
				for i in 0..stride {
					inpd_n[i] += outd_n[i] * params[i];
					param_deriv[i] += outd_n[i] * inp_n[i];
				}
	
				
			}

		}
		
	}	
}



#[cfg(test)]
mod tests {
	use ops::loss::MseLoss;
	use super::*;
	
	#[test]
	fn test_linear_map_backprop(){
		for _ in 1..100{
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(10, "nodein"));
			let n2 = graph.add_output_node(Node::new_flat(10, "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_flat(10, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				LinearMap::new_default(&n1, &n2),
				MseLoss::new_default(&n2, &n3),
			];
			graph.add_operations(ops);
			graph.init_params();
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-1);
		}
	}
	
	#[test]
	fn test_bias1_backprop(){
		for _ in 1..100{
			let mut graph = Graph::new();
		
			let n2 = graph.add_output_node(Node::new_flat(10, "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_flat(10, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				Bias::new_default(&n2),
				MseLoss::new_default(&n2, &n3),
			];
			graph.add_operations(ops);
			graph.init_params();
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-2);
		}
	}

	#[test]
	fn test_bias2_backprop(){
		for _ in 1..100{
			let mut graph = Graph::new();
		
			let n2 = graph.add_output_node(Node::new_sized(10, &[5, 7], "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_sized(10, &[5, 7], "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				Bias::new(&n2, ParamSharing::Spatial, "Bias", init_fill(0.0)),
				MseLoss::new_default(&n2, &n3),
			];
			graph.add_operations(ops);
			graph.init_params();
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-1);
		}
	}

	#[test]
	fn test_scale_backprop(){
		for _ in 1..100{
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(10, "nodein"));
			let n2 = graph.add_output_node(Node::new_flat(10, "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_flat(10, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				Scale::new(&n1, &n2, ParamSharing::None, "ScaleNoSharing", init_fill(1.0)),
				MseLoss::new_default(&n2, &n3),
			];
		
			graph.add_operations(ops);
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-2);
		}
	}

	#[test]
	fn test_scale_sharing_full_backprop(){
		for _ in 1..100{
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(10, "nodein"));
			let n2 = graph.add_output_node(Node::new_flat(10, "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_flat(10, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				Scale::new(&n1, &n2, ParamSharing::Full, "ScaleFullSharing", init_fill(1.0)),
				MseLoss::new_default(&n2, &n3),
			];
		
			graph.add_operations(ops);
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-1);
		}
	}

	#[test]
	fn test_scale_sharing_spatial_backprop(){
		for _ in 1..100{
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_sized(10, &[5, 5], "nodein"));
			let n2 = graph.add_output_node(Node::new_sized(10, &[5, 5], "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_sized(10, &[5, 5], "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				Scale::new(&n1, &n2, ParamSharing::Spatial, "ScaleSpatialSharing", init_fill(1.0)),
				MseLoss::new_default(&n2, &n3),
			];
		
			graph.add_operations(ops);
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-1);
		}
	}


//TODO
	// #[test]
	// fn test_scale_backprop(){
	// 	for _ in 1..100{		
	// 		let mut graph = Graph::new();
		
	// 		let n1 = graph.add_input_node(Node::new_flat(10, "nodein"));
	// 		let n2 = graph.add_output_node(Node::new_flat(10, "nodeout"));
	// 		let n3 = graph.add_training_input_node(Node::new_flat(10, "nodetrain"));
			
	// 		let ops: Vec<Box<Operation>> = vec![
	// 			Scale::new_default(n1, n2),
	// 			MseLoss::new_default(n2, n3),
	// 		];
	// 		graph.add_operations(ops);
	// 		graph.init_params();
			
	// 		use ops::math::*;
	// 		test_numeric(graph, 0.1, 1e-1);
	// 	}
	// }
}

// ScaleElementwise          - PHIHOF - scales each element by a parameter

// TriangularSparseLinearMap, triangular number offsets are connected
// ExponentialSparseLinearMap,  each 2^nth element is connected

// Hadamard       - NSISOF - element by element multiplication using two inputs, one output
// Max            - NSISOD - Takes many inputs, and outputs the max value for each element.
// DampMax        - NSISOD - Takes many inputs, and outputs the max value for each element. Smooth at intersections.

// InfNorm
// LpNorm
// LPNormStab
// Reciprocal - output = 1/input


// InvertGradient = negate gradients during backprop, DOES NOT IMPLMENT CHAIN RULE CORRECTLY, useful for training a GAN and discriminator together
