use new::graph::{GraphDef, PassID, NodeID, OpID, GraphShapes, Result};
use new::init::Initialiser;
use new::ops::math::add::Add;
use new::ops::{standard_op_name, standard_inner_parameter_name, Op, OpInstance};
use new::shape::{NodeShape};

pub struct Bias {
	output_id: NodeID,
	parameter_id: Option<NodeID>,
	param_shape: Option<NodeShape>,
	name: Option<String>,
	initialiser: Option<Initialiser>,
}

impl Bias {
	/// Creates an Op which implements the Bias component of typical neural nets
	///
	/// Intended to provide the Bias component associated with convolutions and fully connected layers in neural nets.
	pub fn new(output: &NodeID) -> Self {
		Bias {
			output_id: output.clone(),
			parameter_id: None,
			param_shape: None,
			name: None,
			initialiser: None,
		}
	}

	/// Provide a node in place of the bias parameter
	///
	/// This node will be added to the output, with broadcasting.
	/// Any value other than `None` prevents the automatic creation of a `Parameter` node.
	/// Default value: `None`
	pub fn parameter(mut self, node_id: Option<&NodeID>) -> Self {
		self.parameter_id = node_id.cloned();
		self
	}

	/// If 'input()' is not set this can be used to control the shape of the parameter node which will be created.
	/// If both this and `input()` are `None` then the created parameter will use the largest dimension sizes that can be guarenteed to broadcast the output node.
	pub fn parameter_shape(mut self, shape: Option<NodeShape>) -> Self {
		self.param_shape = shape;
		self
	}

	pub fn init (mut self, initialiser: Initialiser) -> Self {
		self.initialiser = Some(initialiser);
		self
	}
}

impl Op for Bias {
	type InstanceType = BiasInstance;

	fn type_name(&self) -> &'static str {
		"Bias"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, op_id: &OpID) -> Result<Self::InstanceType> {

		let (name, parameter_is_inner) = if let Some(ref parameter_id) = self.parameter_id {
			(standard_op_name(&self, &self.name, graph, &[parameter_id.clone()], &[self.output_id.clone()]), false)
		} else {
			(standard_op_name(&self, &self.name, graph, &[], &[self.output_id.clone()]), true)
		};

		let parameter_id = if let Some(parameter_id) = self.parameter_id {
			parameter_id
		} else {
			let shape = {
				let output_shape = graph.node_shape(&self.output_id)?;
				self.param_shape.unwrap_or_else(||{output_shape.collapse_to_broadcastable_dimension()})
				};

			let param_name = standard_inner_parameter_name(&name, graph);
			graph.new_node(shape, param_name, tag![Parameter])?
		};

		if let Some(initialiser) = self.initialiser {
			graph.set_initialiser(&parameter_id, initialiser.set_op_id(op_id.clone()));
		}

		let add_id = graph.new_op(Add::new(&parameter_id, &self.output_id.clone()), tag![])?;

		Ok(BiasInstance{
			name: name,
			parameter_id: parameter_id,
			output_id: self.output_id,
			parameter_is_inner: parameter_is_inner,
			add_id: add_id,
		})
	}
}



/// Bias Op, a parameter array is broadcast to the output
#[derive(Clone, Debug)] 
pub struct BiasInstance{
	name: String,
	output_id: NodeID,
	parameter_id: NodeID,
	parameter_is_inner: bool,
	add_id: OpID,
}

impl OpInstance for BiasInstance {

	fn instance_name(&self) -> &str{&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		(
			if self.parameter_is_inner {
				vec![]
			} else {
				vec![self.parameter_id.clone()]
			},
			vec![self.output_id.clone()]
		)
	}

	fn inner_passes(&self) -> Vec<PassID>{vec![]}

	fn inner_ops(&self) -> Vec<OpID>{vec![self.add_id.clone()]}

	fn inner_nodes(&self) -> Vec<NodeID>{
			if self.parameter_is_inner {
				vec![self.parameter_id.clone()]
			} else {
				vec![]
			}
	}

	fn propagate_shape_constraints(&self, _shapes: &mut GraphShapes) -> Result<()>{Ok(())}
}


#[test]
fn test_bias_init_backprop(){
	_bias_init_backprop().unwrap();
}

fn _bias_init_backprop() -> Result<()>{
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use new::ops::loss::mse::Mse;
	use ordermap::OrderMap;
	use new::init::Initialiser;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 16], "input1", tag![])?;
	let node2 = g.new_node(shape![7, 16], "output", tag![])?;

	let o1 = g.new_op(Bias::new(&node1).parameter_shape(Some(shape![1, 16])), tag![])?;
	let _o2 = g.new_op(Mse::new(&node1, &node2), tag![])?;

	let parameter_node = &g.op(&o1)?.inner_nodes()[0];
	g.set_initialiser(parameter_node, Initialiser::fill(1.0));

	let init_values = g.initialise_nodes(&g.op(o1)?.inner_nodes())?;

	assert_eq!(init_values.len(), 1);
	assert_ne!(init_values[0].scalar_sum(), 0.0);


	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}