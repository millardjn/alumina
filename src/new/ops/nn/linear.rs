use new::graph::{GraphDef, NodeID, OpID,  NodeTag, Result};
use new::ops::{standard_op_name, standard_inner_parameter_name, Op};
use new::shape::{NodeShape, NodeDim};
use new::ops::math::matmul::{MatMul, MatMulInstance};

/// The Linear portion of a fully connected layer
///
/// Creates an Op which implements the differentiable matrix multiplication component of typical neural nets.
/// Calculates C += A B, where B is a parameter matrix, A is the input node, and C is the output node.
/// Does not include bias.
pub struct Linear {
	input_id: NodeID,
	output_id: NodeID,
	parameter_id: Option<NodeID>,
	name: Option<String>,
}

impl Linear {
	/// Constructs a new `Linear` Op.
	pub fn new(input: &NodeID, output: &NodeID) -> Self {
		Linear {
			input_id: input.clone(),
			output_id: output.clone(),
			parameter_id: None,
			name: None,
		}
	}

	/// Provide a node to replace the parameter matrix, B
	///
	/// Any value other than `None` prevents the automatic creation of a `Parameter` node.
	/// Default value: `None`
	pub fn parameter(mut self, node_id: Option<&NodeID>) -> Self {
		self.parameter_id = node_id.cloned();
		self
	}
}

impl Op for Linear {
	type InstanceType = MatMulInstance;

	fn type_name(&self) -> &'static str {
		"Linear"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, op_id: &OpID) -> Result<Self::InstanceType> {

		let name = if let Some(ref param) = self.parameter_id {
			standard_op_name(&self, &self.name, graph, &[self.input_id.clone(), param.clone()], &[self.output_id.clone()])
		} else {
			standard_op_name(&self, &self.name, graph, &[self.input_id.clone()], &[self.output_id.clone()])
		};

		// the inner dimension of the matrix excludes the outermost dimension,
		// and includes the largest number of inner dimensions possible (they have to be Known).
		let get_inner = |shape: &NodeShape|{
			shape.dimensions()[1..].iter().rev()
			.take_while(|dim| matches!(dim, &&NodeDim::Known(_))).fold(1, |prod, dim|{
				match dim {
					&NodeDim::Known(x) => prod * x,
					_ => unreachable!(),
				}
			})
		};


		if let Some(param) = self.parameter_id {
			// TODO check that dimensions of param works
			// currently any errors will be picked up at graph execution time.
			MatMul::new(&self.input_id, &param, &self.output_id).name(name).build(graph, op_id)
		} else {
			let n = get_inner(graph.node_shape(&self.output_id)?);
			let k = get_inner(graph.node_shape(&self.input_id)?);
			
			let param_name = standard_inner_parameter_name(&name, graph);
			let param = graph.new_node(shape![k, n], param_name, tag![NodeTag::Parameter])?;
			MatMul::new(&self.input_id, &param, &self.output_id).n(n).k(k).name(name).build(graph, op_id)
		}
	}
}

// TODO currently no shape inference is performed by MatMul. This should be implemented there or implemented here.