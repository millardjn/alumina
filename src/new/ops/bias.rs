use new::graph::{NodeID, Result};
use new::graph;
use new::ops::broadcast::Broadcast;
use new::ops::*;

pub struct Builder {
	output: NodeID,
	input: Option<NodeID>,
	param_shape: Option<NodeShape>,
	name: Option<String>,
}

impl Builder {

	pub fn new(output: &NodeID) -> Self {
		Builder {
			output: output.clone(),
			input: None,
			param_shape: None,
			name: None,
		}
	}

	/// Provide a node to be broadcast as 
	/// If input is `None` a new `Parameter` node will be created to use as input
	/// Default value: None
	pub fn input(mut self, node_id: Option<&NodeID>) -> Self {
		self.input = node_id.cloned();
		self
	}

	/// If 'input()' is not set this can be used to control the shape of the parameter node which will be created.
	/// If both this and `input()` are `None` then the created parameter will use the largest dimension sizes that can be guarenteed to broadcast the output node
	pub fn parameter_shape(mut self, shape: Option<NodeShape>) -> Self {
		self.param_shape = shape;
		self
	}
}

impl OperationBuilder for Builder {
	type OperationType = Broadcast;

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, builder: &mut graph::Builder) -> Result<Self::OperationType> {

		let input = if let Some(input) = self.input {
			input
		} else {
			let shape = if let Some(shape) = self.param_shape {
				shape
			} else {
				builder.get_node_shape(&self.output)?.collapse_nonfixed_dimensions()
			};
			
			builder.new_node(shape, "TODO", tag![])?
			// make new node
		};

		let name = if let Some(name) = self.name {
			name
		} else {
			op_name_gen(builder, "Bias", &[input.clone()], &[self.output.clone()])
		};


		Ok(Broadcast{
			name: name,
			input_id: input,
			output_id: self.output,
		})
	}
}
