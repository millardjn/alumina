use graph::{GraphDef, GraphShapes, Result};
use id::{NodeID, OpID, PassID};
use init::Initialiser;
use ops::math::add::Add;
use ops::{standard_op_name, standard_inner_parameter_name, Op, OpInstance};
use shape::NodeDim;
use smallvec::SmallVec;

#[must_use]
#[derive(Clone, Debug)]
pub struct Bias {
	output_id: NodeID,
	weights_id: Option<NodeID>,
	shared_axes: SmallVec<[isize; 6]>,
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
			weights_id: None,
			shared_axes: SmallVec::new(),
			name: None,
			initialiser: None,
		}
	}

	/// Provide a node in place of the bias weights
	///
	/// This node will be added to the output, with broadcasting.
	/// Any value other than `None` prevents the automatic creation of a `Parameter` node.
	/// Default value: `None`
	pub fn weights(mut self, node_id: Option<&NodeID>) -> Self {
		self.weights_id = node_id.cloned();
		self
	}

	/// Supply axes which learnable weights should be shared over.
	///
	/// By default all axes with Known size are assigned unique weights, and sharing via broadcasting is used for non-Known axes.
	/// Setting an axis as shared will prevent unique weights being used, and enforce sharing, even if the size is Known.
	/// Each element of `axes` can be in the range [-input.ndims(), input.ndims()).
	///
	/// Default: empty
	pub fn shared_axes(mut self, shared_axes: &[isize]) -> Self {
		self.shared_axes = shared_axes.iter().cloned().collect();
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

	fn build(self, graph: &mut GraphDef) -> Result<Self::InstanceType> {

		let (name, weights_are_inner) = if let Some(ref weights_id) = self.weights_id {
			(standard_op_name(&self, &self.name, graph, &[weights_id.clone()], &[self.output_id.clone()]), false)
		} else {
			(standard_op_name(&self, &self.name, graph, &[], &[self.output_id.clone()]), true)
		};

		let weights_id = if let Some(weights_id) = self.weights_id {
			weights_id
		} else {
			let weights_shape = {
				let output_shape = self.output_id.shape();
				let mut weights_shape = vec![1; output_shape.ndim()];

				for axis in 0..output_shape.ndim() {
					if let NodeDim::Known(dim) = output_shape.dimensions()[axis] {
						weights_shape[axis] = dim;
					}
				}

				for shared_axis in &self.shared_axes {
					let shared_axis = (shared_axis + output_shape.ndim() as isize) as usize % output_shape.ndim();
					weights_shape[shared_axis] = 1;
				}

				weights_shape
			};
			let weights_name = standard_inner_parameter_name(&name, graph);
			graph.new_node(weights_shape.into(), weights_name, tag![Parameter])?
		};

		if let Some(initialiser) = self.initialiser {
			graph.set_initialiser(&weights_id, initialiser);
		}

		let add_id = graph.new_op(Add::new(&weights_id, &self.output_id.clone()), tag![])?;

		Ok(BiasInstance{
			name: name,
			weights_id: weights_id,
			output_id: self.output_id,
			weights_are_inner: weights_are_inner,
			add_id: add_id,
		})
	}
}



/// Bias Op, a parameter array is broadcast to the output
#[derive(Clone, Debug)] 
pub struct BiasInstance{
	name: String,
	output_id: NodeID,
	weights_id: NodeID,
	weights_are_inner: bool,
	add_id: OpID,
}

impl OpInstance for BiasInstance {

	fn name(&self) -> &str{&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		(
			if self.weights_are_inner {
				vec![]
			} else {
				vec![self.weights_id.clone()]
			},
			vec![self.output_id.clone()]
		)
	}

	fn inner_passes(&self) -> Vec<PassID>{vec![]}

	fn inner_ops(&self) -> Vec<OpID>{vec![self.add_id.clone()]}

	fn inner_nodes(&self) -> Vec<NodeID>{
			if self.weights_are_inner {
				vec![self.weights_id.clone()]
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
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::mse::Mse;
	use ordermap::OrderMap;
	use init::Initialiser;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 16], "output", tag![])?;
	let node2 = g.new_node(shape![7, 16], "target", tag![])?;

	let o1 = g.new_op(Bias::new(&node1).shared_axes(&[0]), tag![])?;
	let _o2 = g.new_op(Mse::new(&node1, &node2), tag![])?;

	let parameter_node = &o1.instance().inner_nodes()[0];
	g.set_initialiser(parameter_node, Initialiser::fill(1.0));

	let init_values = g.initialise_nodes(&o1.instance().inner_nodes())?;

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