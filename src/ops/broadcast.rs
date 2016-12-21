/// Broadcast - Adds the input node to each spaxel in the output node.
/// Panics if input does not have a fixed size equal to the output channel dimension
#[derive(Clone)] 
pub struct Broadcast {
 	name: String,
 	input_id: NodeID,
	output_id: NodeID,
	output_channels: usize,
}
 	
impl Broadcast {
	pub fn new(input_id: &NodeID, output_id: &NodeID, factors: &[usize], name: &str) -> Box<Broadcast>{
		assert_eq!(output_id.shape.channels, input_id.shape.force_flat_size()
			.expect(&format!("Error: Pooling Operation '{}' requires an input node with a fixed shape.", self.name)),
			"Error: Pooling Operation '{}' output node shape channel dimension must be equal to the input node flat size", name);

		Box::new(Broadcast{
			name: name.to_string(),
			input_id: input_id.clone(),
			output_id: output_id.clone(),
			output_channels: output_id.shape.channels,
		})
	}
}

impl Operation for Broadcast {

	fn name(&self) -> &str{&self.name}
	
	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_id.ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Input node '{}' could not be Pooling to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_id.ind].name, self.name));
		

		let in_err_msg = format!("Error: Operation '{}' error input Node '{}' size has changed since graph construction.", self.name, nodes[self.input_id.ind].name);
		let out_err_msg = format!("Error: Operation '{}' error output Node '{}' size has changed since graph construction.", self.name, nodes[self.output_id.ind].name);
		
		assert_eq!(self.output_channels, input_id.shape.force_flat_size()
			.expect(&format!("Error: Pooling Operation '{}' requires an input node with a fixed shape.", self.name)),
			"Error: Pooling Operation '{}' input node flat size has changed", name);
			
		assert_eq!(output_id.shape.channels, self.output_channels,
			"Error: Pooling Operation '{}' output node shape channel has changed", name);
		
	}
	
	fn input_node_IDs(&self) -> Vec<NodeID>{vec![self.input_id.clone()]}
	
	fn output_node_IDs(&self) -> Vec<NodeID>{vec![self.output_id.clone()]}
		
	fn num_params(&self) -> usize {0}
	
	fn forward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32]){
		let input = &*{data[self.input_id.ind].borrow_mut()};
		let output = &mut *{data[self.output_id.ind].borrow_mut()};
		let in_size = input.shape.flat_size_single();
		let out_size = output.shape.flat_size_single();

		// These checks shouldnt be necessary unless code in the graph doesnt correctly resolve compatible shapes.
		// should check shape compatability under padding etc assert!(input.shape.n == output.shape.n);
		assert_eq!(input.shape.n, output.shape.n);
		assert_eq!(input.shape.channels, self.input_channels);
		assert_eq!(output.shape.channels, self.output_channels);
		assert_eq!(input.shape.spatial_dimensions.len(), output.shape.spatial_dimensions.len());
		assert_eq!(input.shape.spatial_dimensions.len(), self.factors.len());

		let scale = 1.0/self.factors.iter().fold(1,|p, v| p* v) as f32;

		for n_ind in  0..input.shape.n{
			let out_n = &mut output.values[n_ind*out_size..][..out_size];
			let in_n = &input.values[n_ind*in_size..][..in_size];

			for (i, patch) in out_n.chunks_mut(self.output_channels).enumerate(){

			}
		}
	}
	
	fn backward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32], _param_deriv: &mut [f32], _error: &mut f32){
		let input = &mut *{data[self.input_id.ind].borrow_mut()};
		let output = &*{data[self.output_id.ind].borrow_mut()};
		let in_size = input.shape.flat_size_single();
		let out_size = output.shape.flat_size_single();
				
		// These checks shouldnt be necessary unless code in the graph doesnt correctly resolve compatible shapes.
		// should check shape compatability under padding etc assert!(input.shape.n == output.shape.n);
		assert_eq!(input.shape.n, output.shape.n);
		assert_eq!(input.shape.channels, self.input_channels);
		assert_eq!(output.shape.channels, self.output_channels);
		assert_eq!(input.shape.spatial_dimensions.len(), output.shape.spatial_dimensions.len());
		assert_eq!(input.shape.spatial_dimensions.len(), self.factors.len());

		let scale = 1.0/self.factors.iter().fold(1,|p, v| p* v) as f32;

		for n_ind in  0..input.shape.n{
			let outd_n = &output.derivatives[n_ind*out_size..][..out_size];
			let ind_n = &mut input.derivatives[n_ind*in_size..][..in_size];
			
			for (i, patch) in outd_n.chunks(self.output_channels).enumerate(){

			}			
		}			
	}		
}