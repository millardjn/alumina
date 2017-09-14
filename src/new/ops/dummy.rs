use new::graph::{NodeID, Storage, GraphShapes};
use new::graph;
use super::*;

#[derive(Clone, Debug)]
pub struct DummyOperation {
	name: String,
	inputs: Vec<NodeID>,
	outputs: Vec<NodeID>,
	touch_data: bool,
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

	fn forward (&mut self, data: &mut Storage){
		if self.touch_data {
			for id in &self.inputs {
				let _x = data.get(&id.value_id());
			}
			for id in &self.outputs {
				let _x = data.get_mut(&id.value_id());
			}
		}
	}
	
	fn backward (&mut self, data: &mut Storage){
		if self.touch_data {
			for id in &self.inputs {
				let _x = data.get(&id.value_id());
				let _x = data.get_mut(&id.gradient_id());
			}
			for id in &self.outputs {
				let _x = data.get(&id.gradient_id());
			}
		}
	}
}

pub struct Builder{
	name: Option<String>,
	inputs: Vec<NodeID>,
	outputs: Vec<NodeID>,
	touch_data: bool,
}

impl Builder {
	pub fn new() -> Builder {
		Builder {
			name: None,
			inputs: vec![],
			outputs: vec![],
			touch_data: false,
		}
	}

	/// Access the input and output node data from Storage
	pub fn touch_data(mut self, touch_data: bool) -> Self{
		self.touch_data = touch_data;
		self
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
		Builder::new()
	}
}

impl OperationBuilder for Builder {
	type OperationType = DummyOperation;

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	/// Called by graph::Builder to construct the operation instance
	fn build(self, _graph: &mut graph::Builder) -> Self::OperationType{
		DummyOperation{
			name: "".to_string(),
			inputs: self.inputs.clone(),
			outputs: self.outputs.clone(),
			touch_data: self. touch_data,
		}.into()
	}
}