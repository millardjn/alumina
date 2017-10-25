use new::graph::{GraphDef, Subgraph, Dependencies, NodeTag, NodeID, DataID, Result};
use new::opt::{Opt, CallbackData, CallbackSignal};
use ndarray::ArrayD;

pub struct Sgd {
	subgraph: Subgraph,
	inputs: Vec<DataID>,
	parameters: Vec<NodeID>,
	callbacks: Vec<Box<FnMut(&CallbackData)->CallbackSignal>>,
	rate: f32,
	momentum: Option<f32>,
	momentum_vec: Vec<ArrayD<f32>>,
}


impl Sgd {

	/// Create an optimisation problem assuming that all nodes marked `Parameter` should be optimised, and all other leaf nodes are batch inputs.
	pub fn new(graph: &GraphDef) -> Result<Self> {
		let dependencies = Dependencies::new(&graph);
		let input_ids: Vec<NodeID> = graph.nodes().iter().filter(|node_id| dependencies.data_inputs(&node_id.value_id()).len() == 0 && !graph.is_node_tagged(*node_id, NodeTag::Parameter)).cloned().collect();
		let parameter_ids: Vec<NodeID> = graph.nodes().iter().filter(|node_id| dependencies.data_inputs(&node_id.value_id()).len() == 0 && graph.is_node_tagged(*node_id, NodeTag::Parameter)).cloned().collect();
		
		let subgraph = graph.subgraph(
			&input_ids.iter().chain(&parameter_ids).map(|node_id| node_id.value_id()).collect::<Vec<_>>(),
			&parameter_ids.iter().map(|node_id| node_id.value_id()).chain(parameter_ids.iter().map(|node_id| node_id.gradient_id())).collect::<Vec<_>>()
		)?;

		Ok(Sgd {
			inputs: input_ids.iter().map(|node_id| node_id.value_id()).collect::<Vec<_>>(),
			parameters: parameter_ids,
			subgraph: subgraph,
			callbacks: vec![],
			rate: 1e-3,
			momentum: None,
			momentum_vec: vec![],
		})
	}

	/// Define a custom optimisation problem by supplying a subgraph and a list of parameters to optimise.
	///
	/// The subgraph must meet the following:
	/// - subgraph inputs are ordered with general inputs (values or gradients) followed by parameter values.
	/// - subgraph outputs must include all parameters values and gradients.
	///
	/// Note: All leaf nodes not listed as parameters are assumed to be batch inputs.
	pub fn with_subgraph(subgraph: Subgraph, parameter_ids: Vec<NodeID>) -> Self {

		let n_inputs = subgraph.inputs().len() - parameter_ids.len();
		let maybe_inputs = subgraph.inputs()[0..n_inputs].to_vec();
		
		assert!(subgraph.inputs()[n_inputs..].iter().cloned().eq(parameter_ids.iter().map(|id| id.value_id())), "The final inputs to the subgraph must be the values of the optimiser parameter nodes");

		assert!(parameter_ids.iter().all(|id| subgraph.outputs().contains(&id.value_id())), "Subgraph outputs must contain all parameter values");
		assert!(parameter_ids.iter().all(|id| subgraph.outputs().contains(&id.gradient_id())), "Subgraph outputs must contain all parameter gradients");

		Sgd {
			inputs: maybe_inputs,
			parameters: parameter_ids,
			subgraph: subgraph,
			callbacks: vec![],
			rate: 1e-3,
			momentum: None,
			momentum_vec: vec![],
		}
	}

	/// Learning rate, α
	///
	///
	/// θ = θ - α ∇f(θ)
	///
	/// Default: 1e-3
	pub fn rate(mut self, rate: f32) -> Self{
		self.rate = rate;
		self
	}

	/// Momentum coefficient, β
	///
	/// If not `None`, the following update is used:
	/// m = β m + ∇f(θ)
	/// θ = θ - α m
	///
	/// Default: None
	pub fn momentum<O: Into<Option<f32>>>(mut self, momentum: O) -> Self{
		self.momentum = momentum.into();
		self
	}
}

impl Opt for Sgd {

	fn subgraph(&self) -> &Subgraph {
		&self.subgraph
	}

	fn inputs(&self) -> &[DataID]{
		&self.inputs
	}

	fn parameters(&self) -> &[NodeID]{
		&self.parameters
	}

	fn step(&mut self, mut inputs: Vec<ArrayD<f32>>, mut parameters: Vec<ArrayD<f32>>) -> Result<(f32, Vec<ArrayD<f32>>)>{
		assert_eq!(inputs.len(), self.inputs().len(), "Incorrect number of inputs supplied to optimiser.step()");
		assert_eq!(parameters.len(), self.parameters().len(), "Incorrect number of prameters supplied to optimiser.step()");

		inputs.append(&mut parameters);
		
		assert_eq!(self.subgraph.inputs().len(), inputs.len());

		let storage = self.subgraph.execute(inputs)?;
		let loss = storage.loss();
		let mut map = storage.into_map();

		let mut params: Vec<_> = self.parameters.iter().map(|p| map.remove(&p.value_id()).expect("Subgraph must have parameter values as outputs.")).collect();

		if let Some(momentum) = self.momentum {
			if self.momentum_vec.len() != self.parameters.len() {
				self.momentum_vec = params.iter().map(|param| ArrayD::zeros(param.shape())).collect();
			}

			for (i, grad) in self.parameters.iter().map(|p| map.remove(&p.gradient_id()).expect("Subgraph must have parameter gradients as outputs.")).enumerate() {
				self.momentum_vec[i] *= momentum;
				self.momentum_vec[i] += &grad;
				params[i].scaled_add(self.rate, &self.momentum_vec[i]);
			}

		} else {
			for (i, grad) in self.parameters.iter().map(|p| map.remove(&p.gradient_id()).expect("Subgraph must have parameter gradients as outputs.")).enumerate() {
				params[i].scaled_add(self.rate, &grad);
			}
		};

		Ok((loss, params))
	}

	fn callbacks(&mut self) -> &mut [Box<FnMut(&CallbackData)->CallbackSignal>]{
		&mut self.callbacks
	}

	fn add_boxed_callback(&mut self, func: Box<FnMut(&CallbackData)->CallbackSignal>){
		self.callbacks.push(func)
	}
}