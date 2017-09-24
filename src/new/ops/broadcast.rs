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

#[test]
fn test_broadcast(){
	_broadcast().unwrap();
}

fn _broadcast() -> Result<()>{
	use new::ops::dummy;
	use new::graph::GraphDef;
	use new::shape;
	use new::ops::numeric_check::numeric_test;
	use new::ops::loss::mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![1, 1, 16], "broadcast", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "broadcasted", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "target", tag![])?;


	let _o1 = g.new_op(Builder::new(&node1, &node2), tag![])?;
	let _o2 = g.new_op(mse::Builder::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 2;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}