

use graph::*;
use std::cell::RefCell;
use shape::*;
use ops::*;
use std::sync::Arc;
use std::marker::PhantomData;


//###### Activation Ops

#[derive(Clone)]
pub struct IdentityFunc{}
impl ActivationFunc for IdentityFunc {
	fn activ (x: f32) -> (f32, f32){
		(x, 1.0)
	}
}

/// Identity - NSISOF - output equals input
pub type Identity = GenericActivation<IdentityFunc>;


#[derive(Clone)]
pub struct LogisticFunc{}
impl ActivationFunc for LogisticFunc {
	fn activ (x: f32) -> (f32, f32){
		let exp = x.exp();
		(1.0/(1.0 + (-x).exp()), exp/((exp+1.0)*(exp+1.0)))
	}
}

/// Logistic - NSISOF - sigmoid that compresses input and outputs between 0 and 1
pub type Logistic = GenericActivation<LogisticFunc>;


#[derive(Clone)]
pub struct TanhFunc{}
impl ActivationFunc for TanhFunc {
	fn activ (x: f32) -> (f32, f32){
		let s = x.cosh();
		(x.tanh(), 1.0/(s*s))
	}
}

/// Tanh - NSISOF - sigmoid that compresses input and outputs between -1 and 1
pub type Tanh = GenericActivation<TanhFunc>;


#[derive(Clone)]
pub struct ReLUFunc{}
impl ActivationFunc for ReLUFunc {
	fn activ (x: f32) -> (f32, f32){
		let sign = x.signum();
		(
			(x.abs() + x)*0.5, // vectorises, but pretty questionable
			(sign+sign.abs())*0.5 //x.signum().max(0.0); <- this should be better but doesnt compile to maxps,
		)
	}
}

/// `ReLU` - NSISOF - above 0 grad is 1, below 0 grad is 0
pub type ReLU = GenericActivation<ReLUFunc>;


/// `LeakyReLU` - NSISOF - left side slope is a fixed small number, default 0.01
#[derive(Clone)] 
pub struct LeakyReLU {
	name: String,
	left_slope: f32,
	input_id: NodeID,
	output_id: NodeID,
}

impl LeakyReLU {
	
	pub fn new(input_id: &NodeID, output_id: &NodeID, left_slope: f32, name: &str) -> Box<LeakyReLU>{
		Box::new(LeakyReLU{
			name: name.to_string(),
			input_id: input_id.clone(),
			output_id: output_id.clone(),
			left_slope: left_slope,
		})
	}
	pub fn new_default(input_id: &NodeID, output_id: &NodeID,) -> Box<LeakyReLU>{
		LeakyReLU::new(input_id, output_id, 0.01, "LeakyReLU")
	}
}

impl Operation for LeakyReLU {
	
	fn name(&self) -> &str{ &self.name }

	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_id.ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Input node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_id.ind].name, self.name));
		
		shapes[self.output_id.ind] = shapes[self.input_id.ind].merge(&shapes[self.output_id.ind])
			.expect(&format!("Error: Operation '{}' error could not merge input shape with existing shape for output Node '{}'", self.name, nodes[self.output_id.ind].name));
	}
	
	fn num_params(&self) -> usize{ 0 }
	
	fn input_node_IDs(&self) -> Vec<NodeID>{ vec![self.input_id.clone()] }
	
	fn output_node_IDs(&self) -> Vec<NodeID>{ vec![self.output_id.clone()] }
	
	fn forward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32]){
		let input = &*{data[self.input_id.ind].borrow()};
		let output = &mut*{data[self.output_id.ind].borrow_mut()};
			
		// These checks shouldnt be necessary unless code in the graph doesnt correctly resolve compatible shapes.
		debug_assert_eq!(input.shape.n, output.shape.n);
		debug_assert_eq!(input.shape.flat_size_single(), output.shape.flat_size_single());

		let len = input.shape.flat_size_all();
		
		let inp = &input.values[..len];
		let out = &mut output.values[..len];


		// for i in 0..len{
		// 	if inp[i] > 0.0 {
		// 		out[i] += inp[i];
		// 	} else {
		// 		out[i] += inp[i] * self.left_grad;
		// 	}
		// }

		// Hopefully vectorise better?
		for i in 0..len{
			let v = (inp[i] + inp[i].abs())*0.5;
			let v2 = (inp[i] - inp[i].abs())*0.5;
			//let v = inp[i].max(0.0);
			out[i] += v + v2 * self.left_slope;
		}	

	}
	
	fn backward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32], _param_deriv: &mut [f32], _error: &mut f32){
		let input = &mut *{data[self.input_id.ind].borrow_mut()};
		let output = &*{data[self.output_id.ind].borrow()};
		
		// These checks shouldnt be necessary unless code in the graph doesnt correctly resolve compatible shapes.
		debug_assert_eq!(input.shape.n, output.shape.n);
		debug_assert_eq!(input.shape.flat_size_single(), output.shape.flat_size_single());

		let len = input.shape.flat_size_all();
		
		let inp = &input.values[..len];
		let outd = &output.derivatives[..len];
		let inpd = &mut input.derivatives[..len];

		// for i in 0..len{
		// 	if inp[i] > 0.0 {
		// 		inpd[i] += outd[i];
		// 	} else {
		// 		inpd[i] += outd[i] * self.left_slope;
		// 	}
		// }

		for i in 0..len{
			let sign = inp[i].signum();
			let switch = (sign+sign.abs())*0.5;
			let switch2 = (sign-sign.abs())*0.5;
			//let switch = inp[i].signum().max(0.0); // doesnt compile to maxps
			inpd[i] += outd[i] * switch + outd[i] * switch2 * self.left_slope;
		}
	}
}



#[derive(Clone)]
pub struct ELUFunc{}
impl ActivationFunc for ELUFunc {
	fn activ (x: f32) -> (f32, f32){
		if x >= 0.0 {
			(x, 1.0)
		} else {
			(x.exp() - 1.0, x.exp())
		}
	}
}

/// ELU - NSISOF - left side grad is a fixed small number, default 0.01
pub type ELU = GenericActivation<ELUFunc>;



// #[derive(Clone)] 
// pub struct ELU {
// 	name: String,
// 	input_ind: NodeIndex,
// 	output_ind: NodeIndex,
// }

// impl ELU {
// 	pub fn new(&(input, ref _input_shape): &(NodeIndex, NodeShape), &(output, ref _output_shape): &(NodeIndex, NodeShape), name: &str) -> Box<ELU>{
// 		Box::new(ELU{
// 			name: name.to_string(),
// 			input_ind: input,
// 			output_ind: output,
// 		})
// 	}
	
// 	pub fn new_default(input: &(NodeIndex, NodeShape), output: &(NodeIndex, NodeShape)) -> Box<ELU>{
// 		ELU::new(input, output, "ELU")
// 	}
// }

// impl Operation for ELU {
	
// 	fn name(&self) -> &str{ &self.name }

// 	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
// 		shapes[self.input_ind].collapse_ranges_to_minimum()
// 			.expect(&format!("Error: Input node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_ind].name, self.name));
		
// 		shapes[self.output_ind] = shapes[self.input_ind].merge(&shapes[self.output_ind])
// 			.expect(&format!("Error: Operation '{}' error could not merge input shape with existing shape for output Node '{}'", self.name, nodes[self.output_ind].name));
// 	}
	
// 	fn num_params(&self) -> usize{ 0 }
	
// 	fn input_node_ind(&self) -> Vec<NodeIndex>{ vec![self.input_ind] }
	
// 	fn output_node_ind(&self) -> Vec<NodeIndex>{ vec![self.output_ind] }
	
// 	fn forward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32]){
// 		let input_size = data[self.input_ind].borrow().shape.flat_size_all();
// 		let output_size = data[self.output_ind].borrow().shape.flat_size_all();
// 		assert!(input_size == output_size, format!("Error: Operation '{}' input and output node sizes were not equal during evaluation", self.name));
		
// 		let input_values  = &data[self.input_ind].borrow().values;
// 		let out_values = &mut {data[self.output_ind].borrow_mut()}.values;
		
// 		for (i, v) in input_values.iter().enumerate() {
// 			if *v >= 0.0 {
// 				out_values[i] += *v;
// 			} else {
// 				out_values[i] += v.exp() - 1.0;
// 			}
// 		}
		
// 	}
	
// 	fn backward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32], _param_deriv: &mut [f32], _error: &mut f32){
// 		let input_size = data[self.input_ind].borrow().shape.flat_size_all();
// 		let output_size = data[self.output_ind].borrow().shape.flat_size_all();
// 		assert!(input_size == output_size, format!("Error: Operation '{}' input and output node sizes were not equal during evaluation", self.name));

// 		let input_node = &mut *{data[self.input_ind].borrow_mut()};
// 		let input_deriv: &mut [f32] = &mut input_node.derivatives;
// 		let input_values: &[f32] = &input_node.values;
		
// //		let input_values  = &data[self.input_ind].borrow().values;
// //		let input_deriv  = &mut {data[self.input_ind].borrow_mut()}.derivatives;
// 		let out_deriv = &data[self.output_ind].borrow().derivatives;
		
// 		for (i, od) in out_deriv.iter().enumerate() {
// 			if input_values[i] >= 0.0 {
// 				input_deriv[i] += *od;
// 			} else {
// 				input_deriv[i] += od * input_values[i].exp();
// 			}
// 		}
// 	}
// }


#[derive(Clone)]
pub struct SignedLn1pFunc{}
impl ActivationFunc for SignedLn1pFunc {
	fn activ (x: f32) -> (f32, f32){
		if x >= 0.0 {
			((x+1.0).ln(), 1.0/(x + 1.0))
		} else {
			(-(-x+1.0).ln(), 1.0/(-x + 1.0))
		}
	}
}

/// `SignedLn1p` - NSISOF - Ln1p(x) above 0, -Ln1p(-x) below
pub type SignedLn1p = GenericActivation<SignedLn1pFunc>;




// #[derive(Clone)] 
// pub struct SignedLn1p {
// 	name: String,
// 	input_ind: NodeIndex,
// 	output_ind: NodeIndex,
// }

// impl SignedLn1p {
// 	pub fn new(&(input, ref _input_shape): &(NodeIndex, NodeShape), &(output, ref _output_shape): &(NodeIndex, NodeShape), name: &str) -> Box<SignedLn1p>{
// 		Box::new(SignedLn1p{
// 			name: name.to_string(),
// 			input_ind: input,
// 			output_ind: output,
// 		})
// 	}
	
// 	pub fn new_default(input: &(NodeIndex, NodeShape), output: &(NodeIndex, NodeShape)) -> Box<SignedLn1p>{
// 		SignedLn1p::new(input, output, "SignedLn1p")
// 	}
// }

// impl Operation for SignedLn1p {
	
// 	fn name(&self) -> &str{ &self.name }

// 	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
// 		shapes[self.input_ind].collapse_ranges_to_minimum()
// 			.expect(&format!("Error: Input node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_ind].name, self.name));
		
// 		shapes[self.output_ind] = shapes[self.input_ind].merge(&shapes[self.output_ind])
// 			.expect(&format!("Error: Operation '{}' error could not merge required output shape with existing shape for Node '{}'", self.name, nodes[self.output_ind].name));
// 	}
	
// 	fn num_params(&self) -> usize{ 0 }
	
// 	fn input_node_ind(&self) -> Vec<NodeIndex>{ vec![self.input_ind] }
	
// 	fn output_node_ind(&self) -> Vec<NodeIndex>{ vec![self.output_ind] }
	
// 	fn forward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32]){
// 		let input_size = data[self.input_ind].borrow().shape.flat_size_all();
// 		let output_size = data[self.output_ind].borrow().shape.flat_size_all();
// 		assert!(input_size == output_size, format!("Error: Operation '{}' input and output node sizes were not equal during evaluation", self.name));
		
// 		let input_values  = &data[self.input_ind].borrow().values;
// 		let out_values = &mut {data[self.output_ind].borrow_mut()}.values;
		
// 		for (i, v) in input_values.iter().enumerate() {
// 			if *v >= 0.0 {
// 				out_values[i] += (v+1.0).ln();
// 			} else {
// 				out_values[i] += -(-v+1.0).ln();
// 			}
// 		}
		
// 	}
	
// 	fn backward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32], _param_deriv: &mut [f32], _error: &mut f32){
// 		let input_size = data[self.input_ind].borrow().shape.flat_size_all();
// 		let output_size = data[self.output_ind].borrow().shape.flat_size_all();
// 		assert!(input_size == output_size, format!("Error: Operation '{}' input and output node sizes were not equal during evaluation", self.name));
		
// 		let input_node = &mut *{data[self.input_ind].borrow_mut()};
// 		let input_deriv: &mut [f32] = &mut input_node.derivatives;
// 		let input_values: &[f32] = &input_node.values;

// 		let out_deriv = &data[self.output_ind].borrow().derivatives;
		
// 		for (i, od) in out_deriv.iter().enumerate() {
// 			if input_values[i] >= 0.0 {
// 				input_deriv[i] += od / (input_values[i] + 1.0);
// 			} else {
// 				input_deriv[i] += od / (-input_values[i] + 1.0);
// 			}
// 		}
// 	}
// }


// PReLU      - PSISOF - left size grad is an active parameter, a parameter for each element



/// `BeLU`     - PSISOF - Bent Linear Unit - huber loss function superimposed with a parameterised (learnable) linear activation
#[derive(Clone)] 
pub struct BeLU {
	name: String,
	input_id: NodeID,
	output_id: NodeID,
	sharing: ParamSharing,
	num_params: usize,
	init_func: Arc<Fn(&BeLU, &mut [f32])>,
	offset: f32,
}

impl BeLU {
	pub fn new(input_id: &NodeID, output_id: &NodeID, sharing: ParamSharing, name: &str, init_func: Arc<Fn(&BeLU, &mut [f32])>) -> Box<BeLU>{
		
		let (sharing, num_params) = match sharing {
			ParamSharing::Auto => {
				if let Ok(size) = input_id.shape.force_flat_size() {
					(ParamSharing::None, size)
				} else {
					assert_eq!(input_id.shape.rank(), output_id.shape.rank());
					(ParamSharing::Spatial, input_id.shape.channels)
				}
			},
			ParamSharing::None => (ParamSharing::None, input_id.shape.force_flat_size().expect("BeLU Operation with 'None' parameter sharing requires a fully determined shape for the input node")),
			ParamSharing::Full => (ParamSharing::Full, 1),
			ParamSharing::Spatial => (ParamSharing::Spatial, input_id.shape.channels),
		};
		
		Box::new(BeLU{
			name: name.to_string(),
			input_id: input_id.clone(),
			output_id: output_id.clone(),
			num_params: num_params,
			sharing: sharing,
			init_func: init_func,
			offset: 0.0,
		})
	}

	pub fn new_default(input_id: &NodeID, output_id: &NodeID) -> Box<BeLU>{
		BeLU::new(input_id, output_id, ParamSharing::Auto, "BeLU", BeLU::init_elu_like())
	}

	pub fn init_elu_like() -> Arc<Fn(&BeLU, &mut [f32])>{
		init_fill(1.0)

	}
	pub fn init_half() -> Arc<Fn(&BeLU, &mut [f32])>{
		init_fill(0.5)
	}	
	pub fn init_parabola_like() -> Arc<Fn(&BeLU, &mut [f32])>{
		init_fill(0.0)		
	}
	
	pub fn init_porque_no_los_dos() -> Arc<Fn(&BeLU, &mut [f32])>{
		Arc::new(
			|_op: &BeLU, params: &mut [f32]| {
				for (i, x) in params.iter_mut().enumerate(){
					if i%2 == 0 {
						*x = 1.0;
					} else {
						*x = 0.0;
					}
					
				}
			}
		)

	}
}

impl Operation for BeLU {
	
	fn name(&self) -> &str{ &self.name }

	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_id.ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Input node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_id.ind].name, self.name));
		

		shapes[self.output_id.ind] = shapes[self.input_id.ind].merge(&shapes[self.output_id.ind])
			.expect(&format!("Error: Operation '{}' error could not merge required output shape with existing shape for Node '{}'", self.name, nodes[self.output_id.ind].name));
	}
	
	fn num_params(&self) -> usize{ self.num_params }
	
	fn input_node_IDs(&self) -> Vec<NodeID>{ vec![self.input_id.clone()] }
	
	fn output_node_IDs(&self) -> Vec<NodeID>{ vec![self.output_id.clone()] }
	
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
			let param = &params[0]+self.offset;
			
			for i in 0..len {
				let v = inp_n[i];
				out_n[i] += (v*v+1.0).sqrt() - 1.0 + v*param;
			}				
		} else {

			for n_ind in 0..len/stride{
				let out_n = &mut output.values[n_ind*stride..][..stride];
				let inp_n = &input.values[n_ind*stride..][..stride];
				let params = &params[..stride];
				
				
				for i in 0..stride {
					let v = inp_n[i];
					out_n[i] += (v*v+1.0).sqrt() - 1.0 + v*(params[i]+self.offset);
				}			
				
				// Leaving this hear as an example of what not to do. Huge vectorisation improvement not using enumerate.
				//for (i, v) in inp_n.iter().enumerate() {
				//	out_n[i] += (v*v+1.0).sqrt() - 1.0 + v*params[i];
				//}
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
			let param = &params[0]+self.offset;
			let param_deriv = &mut param_deriv[0];	
				
			for i in 0..len {
				let od = outd_n[i];
				let iv = inp_n[i];
				inpd_n[i] += od * (param + iv/ (iv*iv + 1.0).sqrt());
				*param_deriv += od * inp_n[i];
			}
			
		} else {

			for n_ind in 0..len/stride{

				let inp_n = &input.values[n_ind*stride..][..stride];
				let outd_n = &output.derivatives[n_ind*stride..][..stride];
				let inpd_n = &mut input.derivatives[n_ind*stride..][..stride];
				let params = &params[..stride];
				let param_deriv = &mut param_deriv[..stride];	
					
				for i in 0..stride {
					let od = outd_n[i];
					let iv = inp_n[i];
					inpd_n[i] += od * (params[i]+ self.offset + iv/ (iv*iv + 1.0).sqrt());
					param_deriv[i] += od * inp_n[i];
				}
	
				
			}

		}
		
	}
}


#[derive(Clone)]
pub struct SrgbToLinearFunc{}
impl ActivationFunc for SrgbToLinearFunc {
	fn activ (x: f32) -> (f32, f32){
		if x <= 0.0404482362771082{
			(x/12.92, 1.0/12.92)
		} else {
			(((x+0.055)/1.055).powf(2.4), 0.00126754*(200.0*x + 11.0).powf(1.4)) //2.1106*(x+0.055).powf(1.4)
		}
	}
}
pub type SrgbToLinear = GenericActivation<SrgbToLinearFunc>;


#[derive(Clone)]
pub struct LinearToSrgbFunc{}
impl ActivationFunc for LinearToSrgbFunc {
	fn activ (x: f32) -> (f32, f32){	
			if x <= 0.00313066844250063{
				(x*12.92, 12.92)
			} else {
				(-1.055*(0.0521327-x.powf(0.4166666666666667)) , 0.439583*x.powf(-0.5833333333333333) )
			}
	}
}
pub type LinearToSrgb = GenericActivation<LinearToSrgbFunc>;






/// Used to define graph operation where the effect of the input on the output is entirely seperable.
pub trait ActivationFunc: Clone {
	#[inline(always)]
	/// For a given input x, what is the output y, and the derivative dy/dx
	fn activ(input: f32) -> (f32, f32);
}

/// `GenericActivation` - NSISOF
/// Generic over a function which, given an input value, returns an output value and derivative of the output w.r.t the input.
#[derive(Clone)]
pub struct GenericActivation<F: ActivationFunc> {
	name: String,
	input_id: NodeID,
	output_id: NodeID,
	func: PhantomData<F>,
}

impl<F: ActivationFunc> GenericActivation<F> {
	pub fn new(input_id: &NodeID, output_id: &NodeID, name: &str) -> Box<GenericActivation<F>>{
		Box::new(GenericActivation{
			name: name.to_string(),
			input_id: input_id.clone(),
			output_id: output_id.clone(),
			func: PhantomData,
		})
	}
}


impl<F: ActivationFunc + 'static> Operation for GenericActivation<F> {

	fn name(&self) -> &str{&self.name}
	
	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_id.ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Input node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_id.ind].name, self.name));
		
		shapes[self.output_id.ind] = shapes[self.input_id.ind].merge(&shapes[self.output_id.ind])
			.expect(&format!("Error: Operation '{}' error could not merge input shape with existing shape for output Node '{}'", self.name, nodes[self.output_id.ind].name));
	}
	
	fn input_node_IDs(&self) -> Vec<NodeID>{vec![self.input_id.clone()]}
	
	fn output_node_IDs(&self) -> Vec<NodeID>{vec![self.output_id.clone()]}
	
	fn num_params(&self) -> usize {0}
	
	fn forward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32]){
		let input = &*{data[self.input_id.ind].borrow()};
		let output = &mut*{data[self.output_id.ind].borrow_mut()};
			
		// These checks shouldnt be necessary unless code in the graph doesnt correctly resolve compatible shapes.
		debug_assert_eq!(input.shape.n, output.shape.n);
		debug_assert_eq!(input.shape.flat_size_single(), output.shape.flat_size_single());

		let len = input.shape.flat_size_all();
		
		let inp = &input.values[..len];
		let out = &mut output.values[..len];

		for i in 0..len{
			let (output, _deriv) = F::activ(inp[i]);
			out[i] += output;
		}
	}
	
	fn backward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32], _param_deriv: &mut [f32], _error: &mut f32){
		let input = &mut *{data[self.input_id.ind].borrow_mut()};
		let output = &*{data[self.output_id.ind].borrow()};
			
		// These checks shouldnt be necessary unless code in the graph doesnt correctly resolve compatible shapes.
		debug_assert_eq!(input.shape.n, output.shape.n);
		debug_assert_eq!(input.shape.flat_size_single(), output.shape.flat_size_single());

		let len = input.shape.flat_size_all();
		
		let inp = &input.values[..len];
		let outd = &output.derivatives[..len];
		let inpd = &mut input.derivatives[..len];

		for i in 0..len{
			let (_output, deriv) = F::activ(inp[i]);
			inpd[i] += outd[i] * deriv;
		}
		
	}	
}



//
///// SoftExp     - PSISOF - activation function that interpolates between exp, ln and identity.
///// Can perform multiplication/division when 2 layers deep
///// http://arxiv.org/abs/1602.01321
//pub struct SoftExp {
//	name: String,
//	input_ind: NodeIndex,
//	output_ind: NodeIndex,
//	num_params: usize,
//	init_func: Box<Fn(&SoftExp, &mut [f32])>,
//}
//
//impl SoftExp {
//	pub fn new(graph: &Graph, input: NodeIndex, output: NodeIndex, name: &str, init_func: Box<Fn(&SoftExp, &mut [f32])>) -> Box<SoftExp>{
//		Box::new(SoftExp{
//			name: name.to_string(),
//			input_ind: input,
//			output_ind: output,
//			num_params: graph.get_node(input).shape.force_flat_size().expect("SoftExp requires a fully determined shape for the input node"),
//			init_func: init_func,
//		})
//	}
//	
//	pub fn new_default(graph: &Graph, input: NodeIndex, output: NodeIndex) -> Box<SoftExp>{
//		SoftExp::new(graph, input, output, "SoftExp", SoftExp::init_fill(0.0))
//	}
//
//	pub fn init_fill(filler: f32) -> Box<Fn(&SoftExp, &mut [f32])> {
//		Box::new(
//			move |_op: &SoftExp, params: &mut [f32]| {
//				for x in params {
//					*x = filler;
//				}
//			}
//		)
//	}
//
//}

//impl Operation for SoftExp {
//	
//	fn name(&self) -> &str{ &self.name }
//
//	fn enforce_output_shapes(&self, nodes: &[Node], shapes: &mut [NodeShape]){
//		shapes[self.output_ind] = shapes[self.input_ind].merge(&shapes[self.output_ind])
//			.expect(&format!("Error: Operation '{}' error could not merge required output shape with existing shape for Node '{}'", self.name, nodes[self.output_ind].name));
//	}
//	
//	fn num_params(&self) -> usize{ self.num_params }
//	
//	fn input_node_ind(&self) -> Vec<NodeIndex>{ vec![self.input_ind] }
//	
//	fn output_node_ind(&self) -> Vec<NodeIndex>{ vec![self.output_ind] }
//	
//	fn init_params(&mut self, params: &mut [f32]){
//		assert!(self.num_params() == params.len());
//		
//		self.init_func.as_ref()(&self, params);
//	}
//	
//	fn forward (&self, _graph: &Graph, data: &mut [RefCell<NodeData>], params: &[f32]){
//		let input = &*{data[self.input_ind].borrow_mut()};
//		let output = &mut *{data[self.output_ind].borrow_mut()};
//		assert!(input.shape.flat_size_all() == output.shape.flat_size_all(),
//			format!("Error: Operation '{}' input and output node sizes were not equal during evaluation", self.name));
//
//
//		let size = input.shape.flat_size_single();
//		for n_ind in 0..input.shape.n{
//
//			let inp_n = &input.values[n_ind*size..][..size];
//			let out_n = &mut output.values[n_ind*size..][..size];
//							
//			for (i, v) in inp_n.iter().enumerate() {
//				let alpha = params[i];
//				out_n[i] += if alpha > 0.0 {
//					((alpha*v).exp() -1.0)/alpha + alpha
//				} else if alpha < 0.0 {
//					-(1.0 - alpha*(v+alpha)).ln()/alpha
//				} else {
//					 *v
//				}
//			}
//		}
//		
//	}
//	
//	fn backward (&self, _graph: &Graph, data: &mut [RefCell<NodeData>], params: &[f32], param_deriv: &mut [f32], _error: &mut f32){
//		let input = &mut *{data[self.input_ind].borrow_mut()};
//		let output = &*{data[self.output_ind].borrow_mut()};
//		assert!(input.shape.flat_size_all() == output.shape.flat_size_all(),
//			format!("Error: Operation '{}' input and output node sizes were not equal during evaluation", self.name));
//		
//		let size = input.shape.flat_size_single();
//		for n_ind in 0..input.shape.n{
//
//			let inp_n = &input.values[n_ind*size..][..size];
//			let outd_n = &output.derivatives[n_ind*size..][..size];
//			let inpd_n = &mut input.derivatives[n_ind*size..][..size];
//						
//								
//			for (i, od) in outd_n.iter().enumerate() {
//				
//				let a = params[i];
//				let x = inp_n[i];
//				let ax = a * x;
//				let eax = ax.exp();
//				let a2 = a*a;				
//				
//				if a > 0.0 {
//					inpd_n[i] += od * eax;
//					param_deriv[i] += od * (a2 + (ax -1.0) * eax + 1.0)/a2;
//				} else if a < 0.0 {
//					inpd_n[i] += od/(1.0-a*(a+x));
//					param_deriv[i] += od * ( (1.0-(a2+ax)).ln() - (2.0*a2+ax)/(a2+ax-1.0) )/a2;
//				} else {
//					inpd_n[i] += *od;
//					param_deriv[i] += od * (x*x/2.0 + 1.0);
//				}
//			}
//		}
//		
//	}
//}

/// `SoftMax`     - NSISOF - 
#[derive(Clone)] 
pub struct SoftMax {
	name: String,
	input_id: NodeID,
	output_id: NodeID,
}

impl SoftMax {
	pub fn new(input_id: &NodeID, output_id: &NodeID, name: &str) -> Box<SoftMax>{
		Box::new(SoftMax{
			name: name.to_string(),
			input_id: input_id.clone(),
			output_id: output_id.clone(),
		})
	}

	pub fn new_default(input_id: &NodeID, output_id: &NodeID) -> Box<SoftMax>{
		SoftMax::new(input_id, output_id, "SoftMax")
	}
	
}

impl Operation for SoftMax {
	
	fn name(&self) -> &str{ &self.name }

	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_id.ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Input node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_id.ind].name, self.name));
		
		shapes[self.output_id.ind] = shapes[self.input_id.ind].merge(&shapes[self.output_id.ind])
			.expect(&format!("Error: Operation '{}' error could not merge required output shape with existing shape for Node '{}'", self.name, nodes[self.output_id.ind].name));
	}
	
	fn num_params(&self) -> usize{ 0 }
	
	fn input_node_IDs(&self) -> Vec<NodeID>{ vec![self.input_id.clone()] }
	
	fn output_node_IDs(&self) -> Vec<NodeID>{ vec![self.output_id.clone()] }
		
	fn forward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32]){
		let input = &*{data[self.input_id.ind].borrow_mut()};
		let output = &mut *{data[self.output_id.ind].borrow_mut()};
		assert!(input.shape.flat_size_all() == output.shape.flat_size_all(),
			format!("Error: Operation '{}' input and output node sizes were not equal during evaluation", self.name));


		let size = input.shape.flat_size_single();
		for n_ind in 0..input.shape.n{

			let inp_n = &input.values[n_ind*size..][..size];
			let out_n = &mut output.values[n_ind*size..][..size];
			
			let max = inp_n.iter().fold(inp_n[0], |max, &v| v.max(max));
			let sum = inp_n.iter().fold(0., |sum, &v| sum + (v-max).exp());	
				
			for (i, v) in inp_n.iter().enumerate() {
				out_n[i] += (v-max).exp()/sum;
			}
		}
		
		
	}
	
	fn backward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32], _param_deriv: &mut [f32], _error: &mut f32){
		let input = &mut *{data[self.input_id.ind].borrow_mut()};
		let output = &*{data[self.output_id.ind].borrow_mut()};
		assert!(input.shape.flat_size_all() == output.shape.flat_size_all(),
			format!("Error: Operation '{}' input and output node sizes were not equal during evaluation", self.name));
		
		let size = input.shape.flat_size_single();
		for n_ind in 0..input.shape.n{

			let inp_n = &input.values[n_ind*size..][..size];
			let outd_n = &output.derivatives[n_ind*size..][..size];
			let inpd_n = &mut input.derivatives[n_ind*size..][..size];
						
			let max = inp_n.iter().fold(inp_n[0], |max, &v| v.max(max));
			let sum = inp_n.iter().fold(0., |sum, &v| sum + (v-max).exp());	
								
			for (i, od) in outd_n.iter().enumerate() {
				if od.abs() > 0. {
					
					let a = inp_n[i] - max;
					let denom = sum*sum;
					for (j, b) in inp_n.iter().enumerate(){
						let b = b - max;
						inpd_n[j] += if i == j {
							//a.exp() * inp_n.iter().enumerate().fold(0.0, |sum, (ind, v)| sum + if ind != i {(v-max).exp()} else {0.0})
							a.exp() * (sum - a.exp()) // TODO check if cancellation causes instability.
						} else {
							-(a + b).exp()
						}*od/denom;
					}
				}
				

			}
		}
		
		
	}
}



// columnwise PReLU - a parameter for each element in a column
// columnwise BeLU  - a parameter for each element in a column



#[cfg(test)]
mod tests {
	use ops::loss::MseLoss;
	use super::*;

	#[test]
	fn test_srgb2lin_backprop(){
		for _ in 1..100{		
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(40, "nodein"));
			let n2 = graph.add_output_node(Node::new_flat(40, "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_flat(40, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				SrgbToLinear::new(&n1, &n2, "linear"),
				MseLoss::new_default(&n2, &n3),
			];
		
			graph.add_operations(ops);
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-3);
		}
	}

	#[test]
	fn test_lin2srgb_backprop(){
		for _ in 1..100{		
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(40, "nodein"));
			let n2 = graph.add_output_node(Node::new_flat(40, "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_flat(40, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				LinearToSrgb::new(&n1, &n2, "srgb"),
				MseLoss::new_default(&n2, &n3),
			];
		
			graph.add_operations(ops);
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-3);
		}
	}


	#[test]
	fn test_logistic_backprop(){
		for _ in 1..100{		
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(10, "nodein"));
			let n2 = graph.add_output_node(Node::new_flat(10, "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_flat(10, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				Logistic::new(&n1, &n2, "Logistic"),
				MseLoss::new_default(&n2, &n3),
			];
		
			graph.add_operations(ops);
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-2);
		}
	}

	#[test]
	fn test_tanh_backprop(){
		for _ in 1..100{
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(10, "nodein"));
			let n2 = graph.add_output_node(Node::new_flat(10, "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_flat(10, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				Tanh::new(&n1, &n2, "Tanh"),
				MseLoss::new_default(&n2, &n3),
			];
		
			graph.add_operations(ops);
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-2);
		}	
	}

	#[test]
	fn test_relu_backprop(){
		for _ in 1..100{		
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(1000, "nodein"));
			let n2 = graph.add_output_node(Node::new_flat(1000, "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_flat(1000, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				ReLU::new(&n1, &n2, "ReLU"),
				MseLoss::new_default(&n2, &n3),
			];
		
			graph.add_operations(ops);
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-1);
		}
	}


	#[test]
	fn test_leaky_relu_backprop(){
		for _ in 1..100{		
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(1000, "nodein"));
			let n2 = graph.add_output_node(Node::new_flat(1000, "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_flat(1000, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				LeakyReLU::new_default(&n1, &n2),
				MseLoss::new_default(&n2, &n3),
			];
		
			graph.add_operations(ops);
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-1);
		}
	}
	
		#[test]
	fn test_elu_backprop(){
		for _ in 1..100{		
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(10, "nodein"));
			let n2 = graph.add_output_node(Node::new_flat(10, "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_flat(10, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				ELU::new(&n1, &n2, "ELU"),
				MseLoss::new_default(&n2, &n3),
			];
		
			graph.add_operations(ops);
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-2);
		}
	}
	
	#[test]
	fn test_signed_ln1p_backprop(){
		for _ in 1..100{
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(10, "nodein"));
			let n2 = graph.add_output_node(Node::new_flat(10, "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_flat(10, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				SignedLn1p::new(&n1, &n2, "SignedLn1p"),
				MseLoss::new_default(&n2, &n3),
			];
		
			graph.add_operations(ops);
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-2);
		}
	}
	
	#[test]
	#[allow(non_snake_case)]
	fn test_BeLU_backprop(){
		for _ in 1..100{
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(10, "nodein"));
			let n2 = graph.add_output_node(Node::new_flat(10, "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_flat(10, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				BeLU::new(&n1, &n2, ParamSharing::None, "BeluNoSharing", init_fill(1.0)),
				MseLoss::new_default(&n2, &n3),
			];
		
			graph.add_operations(ops);
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-2);
		}
	}

	#[test]
	#[allow(non_snake_case)]
	fn test_BeLU_sharing_full_backprop(){
		for _ in 1..100{
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(10, "nodein"));
			let n2 = graph.add_output_node(Node::new_flat(10, "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_flat(10, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				BeLU::new(&n1, &n2, ParamSharing::Full, "BeluFullSharing", init_fill(1.0)),
				MseLoss::new_default(&n2, &n3),
			];
		
			graph.add_operations(ops);
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-2);
		}
	}

	#[test]
	#[allow(non_snake_case)]
	fn test_BeLU_sharing_spatial_backprop(){
		for _ in 1..100{
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_sized(10, &[5, 5], "nodein"));
			let n2 = graph.add_output_node(Node::new_sized(10, &[5, 5], "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_sized(10, &[5, 5], "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				BeLU::new(&n1, &n2, ParamSharing::Spatial, "BeluFullSharing", init_fill(1.0)),
				MseLoss::new_default(&n2, &n3),
			];
		
			graph.add_operations(ops);
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-2);
		}
	}

//	#[test]
//	fn test_soft_exp(){
//		for _ in 1..100{		
//			let mut graph = Graph::new();
//		
//			let n1 = graph.add_input_node(Node::new_flat(10, "nodein"));
//			let n2 = graph.add_output_node(Node::new_flat(10, "nodeout"));
//			let n3 = graph.add_training_input_node(Node::new_flat(10, "nodetrain"));
//			
//			let ops: Vec<Box<Operation>> = vec![
//				SoftExp::new_default(&graph, n1, n2),
//				MseLoss::new_default(&graph, n2, n3),
//			];
//		
//			graph.add_operations(ops);
//			
//			use ops::math::*;
//			test_numeric(graph, 0.1, 1e-1);
//		}
//	}	
	
	#[test]
	fn test_soft_max(){
		for _ in 1..100{		
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(10, "nodein"));
			let n2 = graph.add_output_node(Node::new_flat(10, "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_flat(10, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				SoftMax::new(&n1, &n2, "softmax"),
				MseLoss::new_default(&n2, &n3),
			];
		
			graph.add_operations(ops);
			
			use ops::math::*;
			test_numeric(graph, 0.1, 1e-1);
		}
	}	
	
		
	#[test]
	fn test_identity_backprop(){
		for _ in 1..100{
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(10, "nodein"));
			let n2 = graph.add_output_node(Node::new_flat(10, "nodeout"));
			let n3 = graph.add_training_input_node(Node::new_flat(10, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				Identity::new(&n1, &n2, "identity"),
				MseLoss::new_default(&n2, &n3),
			];
			graph.add_operations(ops);
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-3);
		}
	}
}
