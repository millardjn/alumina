/// `LinearMap`      - PHIHOF - all to all, typical in MLP network
#[derive(Clone)] 
pub struct MatMul {
	name: String,
	inputA_id: NodeID,
	inputB_id: NodeID,
	outputC_id: NodeID,
	M: usize,
	N: usize,
	// K can change at runtime
}

const FC_ERR: &'static str = "Full Connection requires fully determined shapes for both input and output nodes";

impl MatMul {
	pub fn new(input_id: &NodeID, output_id: &NodeID, name: &str, init_func: Arc<Fn(&LinearMap, &mut [f32])>) -> Box<LinearMap>{
		Box::new(LinearMap{
			name: name.to_string(),
			input_id: input_id.clone(),
			output_id: output_id.clone(),
			input_size: input_id.shape.force_flat_size().expect(FC_ERR),
			output_size: output_id.shape.force_flat_size().expect(FC_ERR),
		})
	}
	
}

/// parameters are a row major matrix
impl OpInstance for MatMul {

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