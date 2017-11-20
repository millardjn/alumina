use new::graph::{GraphDef, NodeID, OpID, PassID, GraphShapes, Result};
use new::init::Initialiser;
use new::ops::{standard_op_name, standard_inner_parameter_name, Op, OpInstance};
use new::shape::{NodeShape, NodeDim};
use new::ops::math::matmul::{MatMul, MatMulInstance};
use rand::{thread_rng, Isaac64Rng, Rng};
use rand::distributions::{Sample, Normal};
use ndarray::ArrayD;

/// The Linear portion of a fully connected layer
///
/// Creates an Op which implements the differentiable matrix multiplication component of typical neural nets.
/// Calculates C += A B, where B is a weights matrix, A is the input node, and C is the output node.
/// Does not include bias.
pub struct Linear {
	input_id: NodeID,
	output_id: NodeID,
	weights_id: Option<NodeID>,
	k: Option<usize>,
	n: Option<usize>,
	name: Option<String>,
	initialiser: Option<Initialiser>,
}

impl Linear {
	/// Constructs a new `Linear` Op.
	pub fn new(input: &NodeID, output: &NodeID) -> Self {
		Linear {
			input_id: input.clone(),
			output_id: output.clone(),
			weights_id: None,
			k: None,
			n: None,
			name: None,
			initialiser: None,
		}
	}

	/// The number of columns in A, the input matrix
	///
	/// If not set, this will be inferred to be the product of all known dimensions of the input
	/// from the innermost outward, stopping before including the first unknown dimension or the outermost dimension
	pub fn k(mut self, k: usize) -> Self {
		self.k = Some(k);
		self
	}

	/// The number of columns in C, the output matrix
	///
	/// If not set, this will be inferred to be the product of all known dimensions of the ouput
	/// from the innermost outward, stopping before including the first unknown dimension or the outermost dimension
	pub fn n(mut self, n: usize) -> Self {
		self.n = Some(n);
		self
	}

	/// Provide a node to replace the weights matrix, B
	///
	/// If setting weights() and init(), ensure that k() is set.
	/// If left as `None` a suitable `Parameter` node will be automatically created.
	///
	/// Default value: `None`
	pub fn weights(mut self, node_id: Option<&NodeID>) -> Self {
		self.weights_id = node_id.cloned();
		self
	}

	/// Provide an Initialiser for the weights node
	pub fn init (mut self, initialiser: Initialiser) -> Self {
		self.initialiser = Some(initialiser);
		self
	}


	/// MSRA/He initialisation
	///
	/// This initialises the parameter matrix with gaussian values drawn from N(0, multiplier/K).
	/// If K of the MatMulInstance is not known, the outermost dimension of the parameter shape will be used.
	/// For typical use, the variance multiplier should cancel out the variance modifying
	/// effect of the nonlinearity, e.g. use 2.0 with ReLU.
	pub fn msra(multiplier: f32) -> Initialiser {
		Initialiser::new("MSRA Initialiser for Linear Op".to_string(), move |arr: &mut ArrayD<f32>, instance: Option<&OpInstance>|{
			let k = instance
				.and_then(|i| i.as_any().downcast_ref::<MatMulInstance>())
				.and_then(|matmul_instance| matmul_instance.K)
				.unwrap_or(arr.shape()[0]); //TODO use ensure to guard against zero length shapes

			let mut rng = thread_rng().gen::<Isaac64Rng>();
			let mut norm = Normal::new(0.0, (multiplier as f64 / k as f64).sqrt());
			for e in arr.iter_mut() {
				*e = norm.sample(&mut rng) as f32;
			}
		})
	}

	/// Xavier initialisation
	///
	/// This is just msra with a multiplier of 1.0
	pub fn xavier() -> Initialiser {
		Linear::msra(1.0)
	}

}

impl Op for Linear {
	type InstanceType = LinearInstance;

	fn type_name(&self) -> &'static str {
		"Linear"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, op_id: &OpID) -> Result<Self::InstanceType> {

		let (name, weights_are_inner) = if let Some(ref weights) = self.weights_id {
			(standard_op_name(&self, &self.name, graph, &[self.input_id.clone(), weights.clone()], &[self.output_id.clone()]), false)
		} else {
			(standard_op_name(&self, &self.name, graph, &[self.input_id.clone()], &[self.output_id.clone()]), true)
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

		let mut n = self.n;
		let mut k = self.k;

		let weights = if let Some(weights) = self.weights_id {
			// TODO check that dimensions of param works
			weights
		} else {
			if n.is_none() {n = Some(get_inner(graph.node_shape(&self.output_id)?));}
			if k.is_none() {k = Some(get_inner(graph.node_shape(&self.input_id)?));}

			let weights_name = standard_inner_parameter_name(&name, graph);
			graph.new_node(shape![k.unwrap(), n.unwrap()], weights_name, tag![Parameter])?
		};

		if let Some(initialiser) = self.initialiser {
			graph.set_initialiser(&weights, initialiser.set_op_id(op_id.clone()));
		}

		let mut mat_mul = MatMul::new(&self.input_id.clone(), &weights, &self.output_id.clone());
		if let Some(n) = self.n {mat_mul = mat_mul.n(n)}
		if let Some(k) = self.k {mat_mul = mat_mul.k(k)}
		let matmul_id = graph.new_op(mat_mul, tag![])?;


		Ok(LinearInstance{
			name: name,
			input_id: self.input_id,
			output_id: self.output_id,
			weights_id: weights,
			weights_are_inner: weights_are_inner,
			matmul_id: matmul_id,
		})
	}
}


/// Linear Op
#[derive(Clone, Debug)] 
pub struct LinearInstance{
	name: String,
	input_id: NodeID,
	output_id: NodeID,
	weights_id: NodeID,
	weights_are_inner: bool,
	matmul_id: OpID,
}

impl OpInstance for LinearInstance {

	fn instance_name(&self) -> &str{&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		(
			if self.weights_are_inner {
				vec![self.input_id.clone()]
			} else {
				vec![self.input_id.clone(), self.weights_id.clone()]
			},
			vec![self.output_id.clone()]
		)
	}

	fn inner_passes(&self) -> Vec<PassID>{vec![]}

	fn inner_ops(&self) -> Vec<OpID>{vec![self.matmul_id.clone()]}

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
fn test_linear_init_backprop(){
	_linear_init_backprop().unwrap();
}

fn _linear_init_backprop() -> Result<()>{
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use new::ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5], "input1", tag![])?;
	let node2 = g.new_node(shape![7, 16], "output", tag![])?;
	let node3 = g.new_node(shape![7, 16], "target", tag![])?;

	let o1 = g.new_op(Linear::new(&node1, &node2).init(Linear::msra(2.0)).k(5), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

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