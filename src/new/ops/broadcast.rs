use new::graph::{GraphDef, NodeID, DataID, Storage, GraphShapes, Result};
use new::ops::{standard_op_name, Op, OpBuilder};
use new::shape::{NodeShape, NodeDim};
use ndarray::{ArrayViewMutD, ArrayViewD};


pub struct Builder {
	output: NodeID,
	input: NodeID,
	name: Option<String>,
}

impl Builder {

	pub fn new(input: &NodeID, output: &NodeID) -> Self {
		Builder {
			input: input.clone(),
			output: output.clone(),
			name: None,
		}
	}

}

impl OpBuilder for Builder {
	type OpType = Broadcast;

	fn type_name(&self) -> &'static str {
		"Broadcast"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef) -> Result<Self::OpType> {
		// TODO check broadcast at graph define time?
		let name = if let Some(name) = self.name {
			name
		} else {
			standard_op_name(&self, graph, &[self.input.clone()], &[self.output.clone()])
		};

		Ok(Broadcast{
			name: name,
			input_id: self.input,
			output_id: self.output,
		})
	}
}




/// Broadcast Op, the value of the input is added to 
#[derive(Clone, Debug)] 
pub struct Broadcast{
	pub(crate) name: String,
	pub(crate) input_id: NodeID,
	pub(crate) output_id: NodeID,
}

impl Op for Broadcast {
	
	fn type_name(&self) -> &'static str {
		"Broadcast"
	}

	fn instance_name(&self) -> &str{ &self.name }

	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{
		let output_shape: NodeShape = shapes.get_shape(&self.input_id).dimensions().iter().map(|dim|{
			match dim {
				&NodeDim::Known(1) => NodeDim::Unknown,
				&NodeDim::Known(x) => NodeDim::Known(x),
				_ => unreachable!(),
			}
		}).into();
		shapes.merge_with(&self.output_id, &output_shape)
	}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		(vec![self.input_id.clone()], vec![self.output_id.clone()])
	}

	// Overwrite default backward_deps because input node values arent needed for backprop
	fn backward_dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		let (op_inputs, op_outputs) = self.dependencies();
		(
			op_outputs.iter().map(|out_node| out_node.gradient_id()).collect(),
			op_inputs.iter().map(|in_node| in_node.gradient_id()).collect()
		)
	}

	fn forward (&mut self, data: &mut Storage) -> Result<()>{
		let input: ArrayViewD<f32> = data.get(&self.input_id.value_id())?;
		let mut output: ArrayViewMutD<f32> = data.get_mut(&self.output_id.value_id())?;

		let input_broadcast = input.broadcast(output.shape()).expect("TODO Handle op errors rather than panic");

		output += &input_broadcast;

		Ok(())

	}
	
	fn backward (&mut self, data: &mut Storage) -> Result<()>{
		let mut input_grad = data.get_mut(&self.input_id.gradient_id())?;
		let output_grad = data.get(&self.output_id.gradient_id())?;

		// TODO check that the match shapes before rather than panic
		for chunk in output_grad.exact_chunks(input_grad.shape()){
			input_grad += &chunk;
		}

		Ok(())
	}
}