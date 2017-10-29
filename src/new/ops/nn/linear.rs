use new::graph::{GraphDef, NodeID, OpID, NodeTag, Result};
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
/// Calculates C += A B, where B is a parameter matrix, A is the input node, and C is the output node.
/// Does not include bias.
pub struct Linear {
	input_id: NodeID,
	output_id: NodeID,
	parameter_id: Option<NodeID>,
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
			parameter_id: None,
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

	/// Provide a node to replace the parameter matrix, B
	///
	/// If setting parameter() and init(), ensure that k() is set.
	/// Any value other than `None` prevents the automatic creation of a `Parameter` node.
	/// Default value: `None`
	pub fn parameter(mut self, node_id: Option<&NodeID>) -> Self {
		self.parameter_id = node_id.cloned();
		self
	}

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
				.unwrap_or(arr.shape()[0]);

			let mut rng = thread_rng().gen::<Isaac64Rng>();
			let mut norm = Normal::new(0.0, (multiplier as f64 / k as f64));
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
			if let Some(initialiser) = self.initialiser {
				graph.set_initialiser(&param, initialiser.set_op_id(op_id.clone()));
			}
			let mut mat_mul = MatMul::new(&self.input_id, &param, &self.output_id).name(name);
			if let Some(n) = self.n {mat_mul = mat_mul.n(n)}
			if let Some(k) = self.k {mat_mul = mat_mul.k(k)}
			mat_mul.build(graph, op_id)
		} else {
			let n = if let Some(n) = self.n {n} else {get_inner(graph.node_shape(&self.output_id)?)};
			let k = if let Some(k) = self.k {k} else {get_inner(graph.node_shape(&self.input_id)?)};
			let param_name = standard_inner_parameter_name(&name, graph);
			let param = graph.new_node(shape![k, n], param_name, tag![NodeTag::Parameter])?;
			if let Some(initialiser) = self.initialiser {
				graph.set_initialiser(&param, initialiser.set_op_id(op_id.clone()));
			}
			MatMul::new(&self.input_id, &param, &self.output_id).n(n).k(k).name(name).build(graph, op_id)
		}
	}
}



#[test]
fn test_linear_init(){
	_linear_init().unwrap();
}

fn _linear_init() -> Result<()>{
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use new::ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5], "input1", tag![])?;
	let node2 = g.new_node(shape![7, 16], "output", tag![])?;
	let node3 = g.new_node(shape![7, 16], "target", tag![])?;

	let o1 = g.new_op(Linear::new(&node1, &node2).init(Linear::msra(2.0)), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let init_values = g.initialise_nodes(&g.op(o1)?.inner_nodes());

	// let iters = 100;
	// let failures = 1;
	// let tolerance = 0.001;
	// let step_size = 1E-2;
	// let default_variance = 1.0;
	// numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}