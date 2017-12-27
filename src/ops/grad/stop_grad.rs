use graph::{GraphDef, GraphShapes, ErrorKind, Result};
use id::{NodeID, DataID, OpID, PassID};
use storage::Storage;
use ops::{standard_op_name, Op, OpInstance, Pass};
use std::any::Any;
use rayon::prelude::*;

#[must_use]
#[derive(Clone, Debug)]
pub struct StopGrad {
	output: NodeID,
	input: NodeID,
	name: Option<String>,
}

impl StopGrad {
	pub fn new(input: &NodeID, output: &NodeID) -> Self {
		StopGrad {
			input: input.clone(),
			output: output.clone(),
			name: None,
		}
	}
}

impl Op for StopGrad {
	type InstanceType = StopGradInstance;

	fn type_name(&self) -> &'static str {
		"StopGrad"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef) -> Result<Self::InstanceType> {
		let name = standard_op_name(&self, &self.name, graph, &[self.input.clone()], &[self.output.clone()]);

		Ok(StopGradInstance{
			name: name,
			input_id: self.input.clone(),
			output_id: self.output.clone(),
			forward_id: graph.add_pass(StopGradForward::new(
					self.input.clone(),
					self.output.clone())),
		})
	}
}

#[derive(Clone, Debug)]
pub struct StopGradInstance {
	name: String,
	input_id: NodeID,
	output_id: NodeID,
	forward_id: PassID,
}

impl OpInstance for StopGradInstance {

	fn name(&self) -> &str{&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){(vec![self.input_id.clone()], vec![self.output_id.clone()])}

	fn inner_passes(&self) -> Vec<PassID>{vec![self.forward_id.clone()]}

	fn inner_ops(&self) -> Vec<OpID>{vec![]}

	fn inner_nodes(&self) -> Vec<NodeID>{vec![]}

	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{
		let input_shape = shapes.get_shape(&self.input_id).clone();
		shapes.merge_with(&self.output_id, &input_shape)
	}
}




#[derive(Clone, Debug)]
pub struct StopGradForward {
	input_id: NodeID,
	output_id: NodeID,
}

impl StopGradForward {
	pub fn new(input_id: NodeID, output_id: NodeID) -> Self {
		StopGradForward {
			input_id,
			output_id,
		}
	}
}

impl Pass for StopGradForward {
	fn type_name(&self) -> &'static str {"StopGradForward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(
			vec![self.input_id.value_id()],
			vec![self.output_id.value_id()]
		)
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input = data.get(&self.input_id.value_id())?;
		let mut output = data.get_mut(&self.output_id.value_id())?;

		ensure!(
			input.shape() == output.shape(),
			ErrorKind::PassError(self.name(), format!("input shape: {:?} did not match output shape: {:?}", input.shape(), output.shape()))
		);

		let input = input.as_slice().unwrap();
		let output = output.as_slice_mut().unwrap();

		let len = input.len();
		
		let inp = &input[..len];
		let out = &mut output[..len];

		inp.par_iter().zip(out.par_iter_mut()).for_each(|(inp, out)|{
			*out += *inp;
		});

		Ok(Box::new(()))
	}
}