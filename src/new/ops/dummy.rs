use new::graph::{NodeID, Storage, GraphShapes};
use new::graph;
use super::*;

#[derive(Clone, Debug)]
pub struct DummyOperation {
	name: String,
	inputs: Vec<NodeID>,
	outputs: Vec<NodeID>
}

impl Operation for DummyOperation {
	fn instance_name(&self) -> &str {
		&self.name
	}

	fn propagate_shape_constraints(&self, _shapes: &mut GraphShapes){
		// Nothing
	}
			
	fn get_meta(&self) -> &OperatorMetaData{
		unimplemented!()
	}
	
	fn operation_dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		(self.inputs.clone(), self.outputs.clone())
	}

	fn forward (&mut self, _data: &mut Storage){
		// Nothing
	}
	
	fn backward (&mut self, _data: &mut Storage){
		// Nothing
	}
}

pub struct Builder{
	name: Option<String>,
	inputs: Vec<NodeID>,
	outputs: Vec<NodeID>
}

impl Builder {
	pub fn new() -> Builder {
		Builder {
			name: None,
			inputs: vec![],
			outputs: vec![],
		}
	}

	pub fn input(mut self, node_id: &NodeID) -> Self{
		self.inputs.push(node_id.clone());
		self
	}

	pub fn output(mut self, node_id: &NodeID) -> Self{
		self.outputs.push(node_id.clone());
		self
	}

	pub fn inputs(mut self, ids: &[NodeID]) -> Self{
		for node_id in ids {
			self.inputs.push(node_id.clone());
		}
		self
	}

	pub fn outputs(mut self, ids: &[NodeID]) -> Self{
		for node_id in ids {
			self.outputs.push(node_id.clone());
		}
		self
	}
}

impl Default for Builder {
	fn default() -> Self {
		Builder {
			name: None,
			inputs: vec![],
			outputs: vec![],
		}
	}
}

impl OperationBuilder for Builder {
	type OperationType = DummyOperation;

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	/// Called by graph::Builder to construct the operation instance
	fn build(self, graph: &mut graph::Builder) -> Self::OperationType{
		DummyOperation{
			name: "".to_string(),
			inputs: self.inputs.clone(),
			outputs: self.outputs.clone(),
		}.into()
	}
}