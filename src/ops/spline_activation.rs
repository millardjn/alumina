use graph::*;
use std::cell::RefCell;
use shape::*;
use ops::*;
use std::sync::Arc;

/// `Spline`     - PSISOF - A smooth continuous function consisting of linear comonents jointed by a central cubic region.
/// Defined as a cubic function in the domain (-1, 1) which passes through 0,0. 3 learnable parameters control the gradients at x=-1, x=0 and x=1.
/// Linear extensions are used outside the central region.

#[derive(Clone)] 
pub struct Spline {
	name: String,
	input_id: NodeID,
	output_id: NodeID,
	sharing: ParamSharing,
	num_params: usize,
	init_func: Arc<Fn(&Spline, &mut [f32])>,
}

impl Spline {
	pub fn new(input_id: &NodeID, output_id: &NodeID, sharing: ParamSharing, name: &str, init_func: Arc<Fn(&Spline, &mut [f32])>) -> Box<Spline>{
		
		let (sharing, num_splines) = match sharing {
			ParamSharing::Auto => {
				if let Ok(size) = input_id.shape.force_flat_size() {
					(ParamSharing::None, size)
				} else {
					assert_eq!(input_id.shape.rank(), output_id.shape.rank());
					(ParamSharing::Spatial, input_id.shape.channels)
				}
			},
			ParamSharing::None => (ParamSharing::None, input_id.shape.force_flat_size().expect("Spline Operation with 'None' parameter sharing requires a fully determined shape for the input node")),
			ParamSharing::Full => (ParamSharing::Full, 1),
			ParamSharing::Spatial => (ParamSharing::Spatial, input_id.shape.channels),
		};
		
		Box::new(Spline{
			name: name.to_string(),
			input_id: input_id.clone(),
			output_id: output_id.clone(),
			num_params: num_splines * 3,
			sharing: sharing,
			init_func: init_func,
		})
	}

	pub fn new_default(input_id: &NodeID, output_id: &NodeID) -> Box<Spline>{
		Spline::new(input_id, output_id, ParamSharing::Auto, "Spline", Spline::init_elu_like())
	}

	pub fn init_custom(left_slope: f32, centre_slope: f32, right_slope: f32) -> Arc<Fn(&Spline, &mut [f32])>{
		Arc::new(
			move |_op: &Spline, params: &mut [f32]| {
				for chunk in params.chunks_mut(3){
					chunk[0] = left_slope;
					chunk[1] = centre_slope;
					chunk[2] = right_slope;
				}
			}
		)
	}

	pub fn init_elu_like() -> Arc<Fn(&Spline, &mut [f32])>{
		Arc::new(
			|_op: &Spline, params: &mut [f32]| {
				for chunk in params.chunks_mut(3){
					chunk[0] = 0.01;
					chunk[1] = 1.0;
					chunk[2] = 1.0;
				}
			}
		)
	}

	pub fn init_tan_like() -> Arc<Fn(&Spline, &mut [f32])>{
		Arc::new(
			|_op: &Spline, params: &mut [f32]| {
				for chunk in params.chunks_mut(3){
					chunk[0] = 0.1;
					chunk[1] = 1.0;
					chunk[2] = 0.1;
				}
			}
		)
	}
	
	pub fn init_parabola_like() -> Arc<Fn(&Spline, &mut [f32])>{
		Arc::new(
			|_op: &Spline, params: &mut [f32]| {
				for chunk in params.chunks_mut(3){
					chunk[0] = -1.0;
					chunk[1] = 0.0;
					chunk[2] = 1.0;
				}
			}
		)
	}

	pub fn init_swan() -> Arc<Fn(&Spline, &mut [f32])>{
		Arc::new(
			|_op: &Spline, params: &mut [f32]| {
				for chunk in params.chunks_mut(3){
					chunk[0] = 0.01;
					chunk[1] = 1.0;
					chunk[2] = 0.25;
				}
			}
		)
	}
	
}

impl Operation for Spline {
	
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

		assert_eq!(stride * 3, params.len());
		assert_eq!(len % stride, 0);
		
		if stride == 1 {
			let out_n = &mut output.values[..len];
			let inp_n = &input.values[..len];
			let left = params[0];
			let centre = params[1];
			let right = params[2];
			
			for i in 0..len {
				let x = inp_n[i];
				let x2 = x*x;
				let x3 = x*x*x;
				
				if x <= -1.0 {
					out_n[i] += (-2.0/3.0)*(centre-1.5*left*x-0.875*left-0.125*right)
				} else if x >= 1.0 {
					out_n[i] += (2.0/3.0)*(centre-0.125*left+1.5*right*x-0.875*right)
				} else {
					out_n[i] += (-1.0/3.0)*(centre*x3-3.0*centre*x-0.5*left*x3+0.75*left*x2-0.5*right*x3-0.75*right*x2)
				}
			}				
		} else {

			for n_ind in 0..len/stride{
				let out_n = &mut output.values[n_ind*stride..][..stride];
				let inp_n = &input.values[n_ind*stride..][..stride];
				let params = &params[..stride*3];
				
				
				for i in 0..stride {
					let left = params[i*3+0];
					let centre = params[i*3+1];
					let right = params[i*3+2];
					let x = inp_n[i];

					if x <= -1.0 {
						out_n[i] += (-2.0/3.0)*(centre-1.5*left*x-0.875*left-0.125*right); // linear segment to the left of x=-1
					} else if x >= 1.0 {
						out_n[i] += (2.0/3.0)*(centre-0.125*left+1.5*right*x-0.875*right); // linear segment to the right of x=1
					} else {
						let x2 = x*x;
						let x3 = x*x*x;
						out_n[i] += (-1.0/3.0)*(centre*x3-3.0*centre*x-0.5*left*x3+0.75*left*x2-0.5*right*x3-0.75*right*x2); // cubic spline passing through 0,0 connecting left and right
					}
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

		assert_eq!(stride * 3, params.len());
		assert_eq!(len % stride, 0);

		if stride == 1 {
			let inp_n = &input.values[..len];
			let outd_n = &output.derivatives[..len];
			let inpd_n = &mut input.derivatives[..len];

			let left = params[0];
			let centre = params[1];
			let right = params[2];

			for i in 0..len {
				let od = outd_n[i];
				let x = inp_n[i];
				if x <= -1.0 {
					inpd_n[i] += od * left;
					param_deriv[0] += od * (x + 7.0/12.0);
					param_deriv[1] += od * (-2.0/3.0);
					param_deriv[2] += od * (1.0/12.0);
				} else if x >= 1.0 {
					inpd_n[i] += od * right;
					param_deriv[0] += od * (-1.0/12.0);
					param_deriv[1] += od * (2.0/3.0);
					param_deriv[2] += od * (x - 7.0/12.0);
				} else {
					let x2 = x*x;
					let x3 = x*x*x;
					inpd_n[i] += od * (centre*(1.0-x2) + x*(left*(0.5*x-0.5) + right*(0.5*x+0.5)));
					param_deriv[0] += od * (x*(1.0/6.0) - 0.25)*x2;
					param_deriv[1] += od * (x - x3*(1.0/3.0));
					param_deriv[2] += od * (x*(1.0/6.0) + 0.25)*x2;
				}
			}
			
		} else {

			for n_ind in 0..len/stride{
				let inp_n = &input.values[n_ind*stride..][..stride];
				let outd_n = &output.derivatives[n_ind*stride..][..stride];
				let inpd_n = &mut input.derivatives[n_ind*stride..][..stride];
				let params = &params[..stride*3];
				let param_deriv = &mut param_deriv[..stride*3];	
					
				for i in 0..stride {
					let od = outd_n[i];
					let x = inp_n[i];
					let left = params[i*3+0];
					let centre = params[i*3+1];
					let right = params[i*3+2];

					if x <= -1.0 {
						inpd_n[i] += od * left;
						param_deriv[i*3+0] += od * (x + 7.0/12.0);
						param_deriv[i*3+1] += od * (-2.0/3.0);
						param_deriv[i*3+2] += od * (1.0/12.0);
					} else if x >= 1.0 {
						inpd_n[i] += od * right;
						param_deriv[i*3+0] += od * (-1.0/12.0);
						param_deriv[i*3+1] += od * (2.0/3.0);
						param_deriv[i*3+2] += od * (x - 7.0/12.0);
					} else {
						let x2 = x*x;
						let x3 = x*x*x;
						inpd_n[i] += od * (centre*(1.0-x2) + x*(left*(0.5*x-0.5) + right*(0.5*x+0.5)));
						param_deriv[i*3+0] += od * (x*(1.0/6.0) - 0.25)*x2;
						param_deriv[i*3+1] += od * (x - x3*(1.0/3.0));
						param_deriv[i*3+2] += od * (x*(1.0/6.0) + 0.25)*x2;
					}
				}
	
				
			}

		}
		
	}
}




#[test]
#[allow(non_snake_case)]
fn test_Spline_backprop(){
	use ops::loss::MseLoss;
	for _ in 1..100{
		let mut graph = Graph::new();
	
		let n1 = graph.add_input_node(Node::new_flat(100, "nodein"));
		let n2 = graph.add_output_node(Node::new_flat(100, "nodeout"));
		let n3 = graph.add_training_input_node(Node::new_flat(100, "nodetrain"));
		
		let ops: Vec<Box<Operation>> = vec![
			Spline::new(&n1, &n2, ParamSharing::None, "SplineNoSharing", init_fill(1.0)),
			MseLoss::new_default(&n2, &n3),
		];
	
		graph.add_operations(ops);
		
		use ops::math::*;
		test_numeric(graph, 1.0, 1e-2);
	}
}

#[test]
#[allow(non_snake_case)]
fn test_Spline_sharing_full_backprop(){
	use ops::loss::MseLoss;
	for _ in 1..100{
		let mut graph = Graph::new();
	
		let n1 = graph.add_input_node(Node::new_flat(100, "nodein"));
		let n2 = graph.add_output_node(Node::new_flat(100, "nodeout"));
		let n3 = graph.add_training_input_node(Node::new_flat(100, "nodetrain"));
		
		let ops: Vec<Box<Operation>> = vec![
			Spline::new(&n1, &n2, ParamSharing::Full, "SplineFullSharing", init_fill(1.0)),
			MseLoss::new_default(&n2, &n3),
		];
	
		graph.add_operations(ops);
		
		use ops::math::*;
		test_numeric(graph, 1.0, 1e-2);
	}
}

#[test]
#[allow(non_snake_case)]
fn test_Spline_sharing_spatial_backprop(){
	use ops::loss::MseLoss;
	for _ in 1..100{
		let mut graph = Graph::new();
	
		let n1 = graph.add_input_node(Node::new_sized(100, &[5, 5], "nodein"));
		let n2 = graph.add_output_node(Node::new_sized(100, &[5, 5], "nodeout"));
		let n3 = graph.add_training_input_node(Node::new_sized(100, &[5, 5], "nodetrain"));
		
		let ops: Vec<Box<Operation>> = vec![
			Spline::new(&n1, &n2, ParamSharing::Spatial, "SplineFullSharing", init_fill(1.0)),
			MseLoss::new_default(&n2, &n3),
		];
	
		graph.add_operations(ops);
		
		use ops::math::*;
		test_numeric(graph, 1.0, 1e-2);
	}
}