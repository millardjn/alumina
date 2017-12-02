use graph::{GraphDef, Subgraph, NodeTag, NodeID, DataID, Result};
use opt::{Opt, CallbackData, CallbackSignal};
use ndarray::{ArrayD, Zip};

/// Adam Optimiser
///
/// t = t + 1
/// m = β1 m + (1 - β1) ∇f(θ)
/// v = β2 v + (1 - β2) ∇f(θ) ∇f(θ)
/// m_c = m / (1 - β1^t)
/// v_c = v / (1 - β2^t)
/// θ = θ - α m_c / sqrt(v_c + eps)
///
pub struct Adam {
	subgraph: Subgraph,
	inputs: Vec<DataID>,
	parameters: Vec<NodeID>,
	callbacks: Vec<Box<FnMut(&CallbackData)->CallbackSignal>>,
	rate: f32,
	beta1: f32,
	beta2: f32,
	epsilon: f32,
	bias_correct: bool,
	momentum_vec: Vec<ArrayD<f32>>,
	curvature_vec: Vec<ArrayD<f32>>,
	step_count: usize,
}


impl Adam {

	/// Create an optimisation problem assuming that all nodes marked `Parameter` should be optimised, and all other leaf nodes are batch inputs.
	pub fn new(graph: &GraphDef) -> Result<Self> {

		let subgraph = graph.default_subgraph()?;

		Ok(Adam {
			inputs: subgraph.inputs().iter().filter(|data_id| !graph.is_node_tagged(&data_id.node_id(), NodeTag::Parameter)).cloned().collect(),
			parameters: subgraph.inputs().iter().filter_map(|data_id| if graph.is_node_tagged(&data_id.node_id(), NodeTag::Parameter) {Some(data_id.node_id())} else {None}).collect(),
			subgraph: subgraph,
			callbacks: vec![],
			rate: 1e-3,
			beta1: 0.9,
			beta2: 0.995,
			epsilon: 1e-8,
			bias_correct: true,
			momentum_vec: vec![],
			curvature_vec: vec![],
			step_count: 0,
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

		Adam {
			inputs: maybe_inputs,
			parameters: parameter_ids,
			subgraph: subgraph,
			callbacks: vec![],
			rate: 1e-3,
			beta1: 0.9,
			beta2: 0.995,
			epsilon: 1e-7,
			bias_correct: true,
			momentum_vec: vec![],
			curvature_vec: vec![],
			step_count: 0,
		}
	}

	/// Learning rate, α
	pub fn rate(mut self, rate: f32) -> Self{
		self.rate = rate;
		self
	}

	/// Momentum coefficient, β1
	///
	/// Default: 0.9
	pub fn beta1(mut self, beta1: f32) -> Self{
		self.beta1 = beta1;
		self
	}

	/// Momentum coefficient, β2
	///
	/// Default: 0.995
	pub fn beta2(mut self, beta2: f32) -> Self{
		self.beta2 = beta2;
		self
	}

	/// Fuzz Factor, eps
	///
	/// Sometimes worth increasing, according to google.
	/// Default: 1e-8
	pub fn epsilon(mut self, epsilon: f32) -> Self{
		self.epsilon = epsilon;
		self
	}

	/// Should bias correction be performed.
	///
	///Default: true
	pub fn bias_correct(mut self, bias_correct: bool) -> Self {
		self.bias_correct = bias_correct;
		self
	}
}

impl Opt for Adam {

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

		if self.momentum_vec.len() != self.parameters.len() {
			self.momentum_vec = params.iter().map(|param| ArrayD::zeros(param.shape())).collect();
		}
		if self.curvature_vec.len() != self.parameters.len() {
			self.curvature_vec = params.iter().map(|param| ArrayD::zeros(param.shape())).collect();
		}

		let rate = self.rate;
		let beta1 = self.beta1;
		let beta2 = self.beta2;
		let epsilon = self.epsilon;
		let momentum_correction = if self.step_count < 1_000_000{1.0/(1.0 - self.beta1.powi(self.step_count as i32 + 1))} else {1.0}; 
		let curv_correction = if self.step_count < 1_000_000{1.0/(1.0 - self.beta2.powi(self.step_count as i32 + 1))} else {1.0}; 

		for (i, param_grad) in self.parameters.iter().map(|p| map.remove(&p.gradient_id()).expect("Subgraph must have parameter gradients as outputs.")).enumerate() {
			if self.bias_correct {
				Zip::from(&mut params[i])
					.and(&mut self.momentum_vec[i])
					.and(&mut self.curvature_vec[i])
					.and(&param_grad)
					.apply(|param, momentum, curv, param_grad| {
						*momentum = *momentum * beta1 + (1.0-beta1)*param_grad;
						*curv = *curv * beta2 + (1.0-beta1)*param_grad*param_grad;

						*param += -rate * (*momentum) * momentum_correction/((*curv*curv_correction).sqrt() + epsilon);
					});
			} else {
				Zip::from(&mut params[i])
					.and(&mut self.momentum_vec[i])
					.and(&mut self.curvature_vec[i])
					.and(&param_grad)
					.apply(|param, momentum, curv, param_grad| {
						*momentum = *momentum * beta1 + (1.0-beta1)*param_grad;
						*curv = *curv * beta2 + (1.0-beta1)*param_grad*param_grad;

						*param += -rate * (*momentum) /(curv.sqrt() + epsilon);
					});
			}
		}

		self.step_count += 1;

		Ok((loss, params))
	}

	fn callbacks(&mut self) -> &mut [Box<FnMut(&CallbackData)->CallbackSignal>]{
		&mut self.callbacks
	}

	fn add_boxed_callback(&mut self, func: Box<FnMut(&CallbackData)->CallbackSignal>){
		self.callbacks.push(func)
	}
}