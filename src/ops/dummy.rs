use graph::{GraphDef, NodeID, Storage, GraphShapes, Result};
use ops::*;

#[must_use]
#[derive(Clone, Debug)] 
pub struct Dummy {
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
	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		let name = standard_op_name(&self, &self.name, graph, &self.inputs, &self.outputs);

		
		Ok(DummyInstance{
			name: name,
			inputs: self.inputs.clone(),
			outputs: self.outputs.clone(),
			touch_data: self.touch_data,
			forward_id: graph.add_pass(DummyForward::new(
					self.inputs.clone(),
					self.outputs.clone(),
					self.touch_data)),
			backward_id: graph.add_pass(DummyBackward::new(
					self.inputs.clone(),
					self.outputs.clone(),
					self.touch_data)),
		})
	}
}


#[derive(Clone, Debug)]
pub struct DummyInstance {
	name: String,
	inputs: Vec<NodeID>,
	outputs: Vec<NodeID>,
	touch_data: bool,
	forward_id: PassID,
	backward_id: PassID,
}

impl OpInstance for DummyInstance {

	fn instance_name(&self) -> &str {&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){(self.inputs.clone(), self.outputs.clone())}

	fn inner_passes(&self) -> Vec<PassID> {vec![self.forward_id.clone(), self.backward_id.clone()]}

	fn inner_ops(&self) -> Vec<OpID> {vec![]}

	fn inner_nodes(&self) -> Vec<NodeID> {vec![]}

	fn propagate_shape_constraints(&self, _shapes: &mut GraphShapes) -> Result<()>{Ok(())}
}

#[derive(Clone, Debug)]
pub struct DummyForward {
	inputs: Vec<NodeID>,
	outputs: Vec<NodeID>,
	touch_data: bool,
}

impl DummyForward {
	pub fn new(inputs: Vec<NodeID>, outputs: Vec<NodeID>, touch_data: bool) -> Self {
		DummyForward {
			inputs,
			outputs,
			touch_data
		}
	}
}

/// Backward pass of the Dummy Op
///
/// Note that the inputs of this pass are the inputs of the Dummy Op, not the input dependencies of the pass.
#[derive(Clone, Debug)]
pub struct DummyBackward {
	inputs: Vec<NodeID>,
	outputs: Vec<NodeID>,
	touch_data: bool,
}

impl DummyBackward {
	pub fn new(inputs: Vec<NodeID>, outputs: Vec<NodeID>, touch_data: bool) -> Self {
		DummyBackward {
			inputs,
			outputs,
			touch_data
		}
	}
}

impl Pass for DummyForward {
	fn type_name(&self) -> &'static str {"DummyForward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(self.inputs.iter().map(|node_id| node_id.value_id()).collect(),
		self.outputs.iter().map(|node_id| node_id.value_id()).collect())
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		if self.touch_data {
			for id in &self.inputs {
				let _x = data.get(&id.value_id());
			}
			for id in &self.outputs {
				let _x = data.get_mut(&id.value_id());
			}
		}
		Ok(Box::new(()))
	}
}

impl Pass for DummyBackward {
	fn type_name(&self) -> &'static str {"DummyBackwards"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(self.inputs.iter().map(|node_id| node_id.value_id())
		.chain(self.outputs.iter().map(|node_id| node_id.gradient_id())).collect(),
		self.inputs.iter().map(|node_id| node_id.gradient_id()).collect())
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>> {
		if self.touch_data {
			for id in &self.inputs {
				let _x = data.get(&id.value_id());
				let _x = data.get_mut(&id.gradient_id());
			}
			for id in &self.outputs {
				let _x = data.get(&id.gradient_id());
			}
		}
		Ok(Box::new(()))
	}
}