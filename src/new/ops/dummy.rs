use new::graph::{GraphDef, NodeID, Storage, GraphShapes, Result};
use new::ops::*;

pub struct Dummy{
	name: Option<String>,
	inputs: Vec<NodeID>,
	outputs: Vec<NodeID>,
	touch_data: bool,
}

impl Dummy {
	pub fn new() -> Dummy {
		Dummy {
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

impl Op for Dummy {
	type InstanceType = DummyInstance;

	fn type_name(&self) -> &'static str {
		"Dummy"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	/// Called by GraphDef::new_op to construct the op instance
	fn build(self, graph: &mut GraphDef) -> Result<Self::InstanceType> {
		let name = if let Some(name) = self.name {
			name
		} else {
			standard_op_name(&self, graph, &self.inputs, &self.outputs)
		};
		Ok(DummyInstance{
			name: name,
			inputs: self.inputs.clone(),
			outputs: self.outputs.clone(),
			touch_data: self. touch_data,
		})
	}
}


#[derive(Clone, Debug)]
pub struct DummyInstance {
	name: String,
	inputs: Vec<NodeID>,
	outputs: Vec<NodeID>,
	touch_data: bool,
}

impl OpInstance for DummyInstance {
	fn type_name(&self) -> &'static str {
		"Dummy"
	}

	fn instance_name(&self) -> &str {
		&self.name
	}

	fn propagate_shape_constraints(&self, _shapes: &mut GraphShapes) -> Result<()>{
		Ok(())
	}
				
	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		(self.inputs.clone(), self.outputs.clone())
	}

	fn forward (&mut self, data: &mut Storage) -> Result<()>{
		if self.touch_data {
			for id in &self.inputs {
				let _x = data.get(&id.value_id());
			}
			for id in &self.outputs {
				let _x = data.get_mut(&id.value_id());
			}
		}
		Ok(())
	}
	
	fn backward (&mut self, data: &mut Storage) -> Result<()>{
		if self.touch_data {
			for id in &self.inputs {
				let _x = data.get(&id.value_id());
				let _x = data.get_mut(&id.gradient_id());
			}
			for id in &self.outputs {
				let _x = data.get(&id.gradient_id());
			}
		}
		Ok(())
	}
}