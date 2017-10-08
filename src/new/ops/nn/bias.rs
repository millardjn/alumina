use new::graph::{GraphDef, NodeID, OpID, PassID, NodeTag, Result};
use new::ops::math::add::{AddInstance, Add};
use new::ops::{standard_op_name, standard_inner_parameter_name, Op, OpInstance, Pass};
use new::shape::{NodeShape, NodeDim};

pub struct Bias {
	output: NodeID,
	input: Option<NodeID>,
	param_shape: Option<NodeShape>,
	name: Option<String>,
}

impl Bias {
	/// Creates an Op which implements the Bias component of typical neural nets
	///
	/// Intended to provide the Bias component associated with convolutions and fully connected layers in neural nets.
	pub fn new(output: &NodeID) -> Self {
		Bias {
			output: output.clone(),
			input: None,
			param_shape: None,
			name: None,
		}
	}

	/// Provide a node in place of the bias parameter
	///
	/// This node will be added to the output, with broadcasting.
	/// Any value other than `None` prevents the automatic creation of a `Parameter` node.
	/// Default value: `None`
	pub fn input(mut self, node_id: Option<&NodeID>) -> Self {
		self.input = node_id.cloned();
		self
	}

	/// If 'input()' is not set this can be used to control the shape of the parameter node which will be created.
	/// If both this and `input()` are `None` then the created parameter will use the largest dimension sizes that can be guarenteed to broadcast the output node.
	pub fn parameter_shape(mut self, shape: Option<NodeShape>) -> Self {
		self.param_shape = shape;
		self
	}
}

impl Op for Bias {
	type InstanceType = AddInstance;

	fn type_name(&self) -> &'static str {
		"Bias"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, op_id: &OpID) -> Result<Self::InstanceType> {

		let name = if let Some(ref input) = self.input {
			standard_op_name(&self, &self.name, graph, &[input.clone()], &[self.output.clone()])
		} else {
			standard_op_name(&self, &self.name, graph, &[], &[self.output.clone()])
		};

		let input = if let Some(input) = self.input {
			input
		} else {
			let shape = if let Some(shape) = self.param_shape {
				shape
			} else {
				graph.node_shape(&self.output)?.collapse_to_broadcastable_dimension()
			};
			
			let param_name = standard_inner_parameter_name(&name, graph);
			graph.new_node(shape, param_name, tag![NodeTag::Parameter])?
		};

		Add::new(&input, &self.output).name(name).build(graph, op_id)
	}
}
