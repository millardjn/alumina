use graph::{GraphDef, NodeID, DataID, OpID, PassID, Storage, GraphShapes, ErrorKind, Result};
use ops::{standard_op_name, standard_inner_parameter_name, Op, OpInstance, Pass};
use shape::NodeDim;
use ndarray::{ArrayViewMutD, Zip};
use std::any::Any;
use smallvec::SmallVec;
use init::Initialiser;
use arrayvec::ArrayVec;

/// `Spline` A smooth continuous function consisting of linear components jointed by a central cubic region.
///
/// Defined as a cubic function in the domain (-1, 1) which passes through 0,0. Three learnable parameters control the gradients at x=-1, x=0 and x=1.
/// Linear extensions are used outside the central region.
#[must_use]
#[derive(Clone, Debug)] 
pub struct Spline {
	name: Option<String>,
	input_id: NodeID,
	weights_id: Option<NodeID>,
	output_id: NodeID,
	shared_axes: SmallVec<[isize; 6]>,
	initialiser: Option<Initialiser>,
}

impl Spline {
	pub fn new(input_id: &NodeID, output_id: &NodeID) -> Self {
		Spline{
			name: None,
			input_id: input_id.clone(),
			weights_id: None,
			output_id: output_id.clone(),
			shared_axes: SmallVec::new(),
			initialiser: None,
		}
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

	/// Provide a node to act as the weights
	///
	/// The shape of the weights must be a 3 prepended to a shape which can broadcast to the input shape.
	/// If left as `None` a suitable `Parameter` node will be automatically created.
	///
	/// Default value: `None`
	pub fn weights(mut self, node_id: Option<&NodeID>) -> Self {
		self.weights_id = node_id.cloned();
		self
	}

	/// Provide an Initialiser for the weights node
	pub fn init(mut self, initialiser: Initialiser) -> Self {
		self.initialiser = Some(initialiser);
		self
	}

	fn _custom(name: &'static str, left_slope: f32, centre_slope: f32, right_slope: f32) -> Initialiser{
		Initialiser::new(name.to_string(), move |mut arr: ArrayViewMutD<f32>, _instance: Option<&OpInstance>|{
			if arr.shape()[0] == 3 {
				let mut weights_iter = arr.outer_iter_mut();
				weights_iter.next().unwrap().fill(left_slope);
				weights_iter.next().unwrap().fill(centre_slope);
				weights_iter.next().unwrap().fill(right_slope);
			} else {
				eprintln!("{} could not be executed because outermost dimension was not equal to 3", name);
			}
		})
	}

	pub fn custom(left_slope: f32, centre_slope: f32, right_slope: f32) -> Initialiser{
		Spline::_custom("Custom Initialiser for Spline Op", left_slope, centre_slope, right_slope)
	}

	pub fn elu_esque() -> Initialiser{
		Spline::_custom("ELU-esque Initialiser for Spline Op", 0.01, 1.0, 1.0)
	}

	pub fn tanh_esque() -> Initialiser{
		Spline::_custom("Tanh-esque Initialiser for Spline Op", 0.01, 1.0, 0.01)
	}

	pub fn parabola_esque() -> Initialiser{
		Spline::_custom("Parabola-esque Initialiser for Spline Op", -1.0, 0.0, 1.0)
	}

	pub fn swan() -> Initialiser{
		Spline::_custom("Swan Initialiser for Spline Op", 0.01, 1.0, 0.25)
	}
}

impl Op for Spline {
	type InstanceType = SplineInstance;

	fn type_name(&self) -> &'static str {
		"Spline"
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

		let weights = if let Some(weights) = self.weights_id {
			// TODO check that dimensions of param works
			weights
		} else {
			let weights_shape = {
				let input_shape = graph.node_shape(&self.input_id)?;
				let mut weights_shape = vec![1; input_shape.ndim() + 1];
				weights_shape[0] = 3;
				for axis in 0..input_shape.ndim() {
					if let NodeDim::Known(dim) = input_shape.dimensions()[axis] {
						weights_shape[axis + 1] = dim;
					}
				}

				for shared_axis in &self.shared_axes {
					let shared_axis = (shared_axis + input_shape.ndim() as isize) as usize % input_shape.ndim();
					weights_shape[shared_axis + 1] = 1;
				}

				weights_shape
			};

			let weights_name = standard_inner_parameter_name(&name, graph);
			graph.new_node(weights_shape.into(), weights_name, tag![Parameter])?
		};

		if let Some(initialiser) = self.initialiser {
			graph.set_initialiser(&weights, initialiser.set_op_id(op_id.clone()));
		}

		Ok(SplineInstance{
			name: name,
			input_id: self.input_id.clone(),
			weights_id: weights.clone(),
			output_id: self.output_id.clone(),
			shared_axes: self.shared_axes.clone(),
			weights_are_inner: weights_are_inner,
			forward_id:graph.add_pass(SplineForward::new(
				self.input_id.clone(),
				weights.clone(),
				self.output_id.clone(),
			)),
			backward_id:graph.add_pass(SplineBackward::new(
				self.input_id.clone(),
				weights.clone(),
				self.output_id.clone(),
			)),
		})
	}
}


#[derive(Clone, Debug)] 
pub struct SplineInstance {
	name: String,
	input_id: NodeID,
	weights_id: NodeID,
	output_id: NodeID,
	shared_axes: SmallVec<[isize; 6]>,
	weights_are_inner: bool,
	forward_id: PassID,
	backward_id: PassID,
}

impl OpInstance for SplineInstance {
	
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

	fn inner_passes(&self) -> Vec<PassID>{vec![self.forward_id.clone(), self.backward_id.clone()]}

	fn inner_ops(&self) -> Vec<OpID>{vec![]}

	fn inner_nodes(&self) -> Vec<NodeID>{
		if self.weights_are_inner {
			vec![self.weights_id.clone()]
		} else {
			vec![]
		}
	}

	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{
		let input_shape = shapes.get_shape(&self.input_id).clone();
		shapes.merge_with(&self.output_id, &input_shape)
	}
}



#[derive(Debug, Clone)]
pub struct SplineForward {
	input_id: NodeID,
	weights_id: NodeID,
	output_id: NodeID,
}

impl SplineForward {
	pub fn new(input_id: NodeID, weights_id: NodeID, output_id: NodeID) -> Self {
		SplineForward {
			input_id,
			weights_id,
			output_id,
		}
	}
}

impl Pass for SplineForward {
	fn type_name(&self) -> &'static str {"SplineForward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input_id.value_id(), self.weights_id.value_id()],
		vec![self.output_id.value_id()])
	}

	fn run(&self, data: &Storage) -> Result<Box<Any>> {
		let input = data.get(&self.input_id.value_id())?;
		let weights = data.get(&self.weights_id.value_id())?;
		let mut output = data.get_mut(&self.output_id.value_id())?;

		let input_shape = input.shape();
		let output_shape = output.shape().to_vec();
		let weights_outer_shape = weights.shape();
		let weights_shape = &weights_outer_shape[1..];

		let weights: ArrayVec<[_;3]> = weights.outer_iter().collect();
		
		ensure!(
			input_shape == &output_shape[..],
			ErrorKind::PassError(self.instance_name(data.graph()), format!("input shape: {:?} did not match output shape: {:?}", input_shape, output_shape))
		);
		ensure!(
			weights[0].broadcast(output_shape).is_some(), 
			ErrorKind::PassError(self.instance_name(data.graph()), format!("Could not broadcast weights_shape[1..]: {:?} to input/output shape: {:?}", weights_shape, input_shape))
		);

		let iter = output.exact_chunks_mut(weights_shape).into_iter()
			.zip(input.exact_chunks(weights_shape));
		
		for (mut output, input) in iter {
			Zip::from(&mut output)
				.and(&input)
				.and(&weights[0])
				.and(&weights[1])
				.and(&weights[2])
				.apply(|output, &input, &left, &centre, &right| {
					let x = input;
					if x <= -1.0 {
						*output += (-2.0/3.0)*(centre - 1.5*left*x - 0.875*left - 0.125*right); // linear segment to the left of x=-1
					} else if x >= 1.0 {
						*output += (2.0/3.0)*(centre + 1.5*right*x - 0.125*left - 0.875*right); // linear segment to the right of x=1
					} else {
						let x2 = x*x;
						let x3 = x*x*x;
						*output += (-1.0/3.0)*(
								centre*x3
								-3.0*centre*x
								-0.5*(left+right)*x3
								+0.75*(left-right)*x2
							); // cubic spline passing through 0,0 connecting left and right
					}
				});
		}

		Ok(Box::new(()))
	}
}


#[derive(Debug, Clone)]
pub struct SplineBackward {
	input_id: NodeID,
	weights_id: NodeID,
	output_id: NodeID,
}

impl SplineBackward {
	pub fn new(input_id: NodeID, weights_id: NodeID, output_id: NodeID) -> Self {
		SplineBackward {
			input_id,
			weights_id,
			output_id,
		}
	}
}

impl Pass for SplineBackward {
	fn type_name(&self) -> &'static str {"SplineBackward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input_id.value_id(), self.weights_id.value_id(), self.output_id.gradient_id()],
		vec![self.input_id.gradient_id(), self.weights_id.gradient_id()])
	}

	fn run(&self, data: &Storage) -> Result<Box<Any>> {
		let input = data.get(&self.input_id.value_id())?;
		let weights = data.get(&self.weights_id.value_id())?;
		let output_grad = data.get(&self.output_id.gradient_id())?;

		let input_shape = input.shape();
		let output_shape = output_grad.shape();
		let weights_outer_shape = weights.shape();
		let weights_shape = &weights_outer_shape[1..];

		let weights: ArrayVec<[_;3]> = weights.outer_iter().collect();

		ensure!(
			input_shape == &output_shape[..],
			ErrorKind::PassError(self.instance_name(data.graph()), format!("input shape: {:?} did not match output shape: {:?}", input_shape, output_shape))
		);
		ensure!(
			weights_outer_shape[0] == 3,
			ErrorKind::PassError(self.instance_name(data.graph()), format!("The outermost dimension of the weights shape must be 3 not {}", weights_outer_shape[0]))
		);
		ensure!(
			weights[0].broadcast(output_shape).is_some(), 
			ErrorKind::PassError(self.instance_name(data.graph()), format!("Could not broadcast weights_shape[1..]: {:?} to input/output shape: {:?}", weights_shape, input_shape))
		);


		
		if data.is_required(&self.input_id.gradient_id()) {
			let mut input_grad = data.get_mut(&self.input_id.gradient_id())?;
			let iter = output_grad.exact_chunks(weights_shape).into_iter()
				.zip(input.exact_chunks(weights_shape))
				.zip(input_grad.exact_chunks_mut(weights_shape));

			for ((output_grad, input), mut input_grad) in iter {

				Zip::from(&output_grad)
					.and(&input)
					.and(&weights[0])
					.and(&weights[1])
					.and(&weights[2])
					.and(&mut input_grad)
					.apply(|&output_grad, &input, &left, &centre, &right, input_grad| {
						let x = input;
						if x <= -1.0 {
							*input_grad += output_grad * left;
						} else if x >= 1.0 {
							*input_grad += output_grad * right;
						} else {
							let x2 = x*x;
							//let x3 = x*x*x;
							*input_grad += output_grad * (centre*(1.0-x2) + x*(left*(0.5*x-0.5) + right*(0.5*x+0.5)));
						}
					});

			}
		}

		if data.is_required(&self.weights_id.gradient_id()) {
			let mut weights_grad = data.get_mut(&self.weights_id.gradient_id())?;
			let mut weights_grad_iter = weights_grad.outer_iter_mut();
			let mut weights_grad0 = weights_grad_iter.next().unwrap();
			let mut weights_grad1 = weights_grad_iter.next().unwrap();
			let mut weights_grad2 = weights_grad_iter.next().unwrap();

			let iter = output_grad.exact_chunks(weights_shape).into_iter()
				.zip(input.exact_chunks(weights_shape));

			for (output_grad, input) in iter {
				Zip::from(&output_grad)
					.and(&input)
					.and(&mut weights_grad0)
					.and(&mut weights_grad1)
					.and(&mut weights_grad2)
					.apply(|&output_grad, &input, left_grad, centre_grad, right_grad| {
						let x = input;
						if x <= -1.0 {
							*left_grad   += output_grad * (x + 7.0/12.0);
							*centre_grad += output_grad * (-2.0/3.0);
							*right_grad  += output_grad * (1.0/12.0);
						} else if x >= 1.0 {
							*left_grad   += output_grad * (-1.0/12.0);
							*centre_grad += output_grad * (2.0/3.0);
							*right_grad  += output_grad * (x - 7.0/12.0);
						} else {
							let x2 = x*x;
							let x3 = x*x*x;
							*left_grad   += output_grad * (x*(1.0/6.0) - 0.25)*x2;
							*centre_grad += output_grad * (x - x3*(1.0/3.0));
							*right_grad  += output_grad * (x*(1.0/6.0) + 0.25)*x2;
						}
					});
			}
		}


		Ok(Box::new(()))
	}
}

#[test]
fn test_spline_backprop(){
	_spline_backprop().unwrap();
}

fn _spline_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "output", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "target", tag![])?;


	let _o1 = g.new_op(Spline::new(&node1, &node2).init(Spline::elu_esque()), tag![])?;
	let _o2 = g.new_op(Spline::new(&node1, &node2).init(Spline::tanh_esque()), tag![])?;
	let _o3 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.002;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}


#[test]
fn test_spline_shared_backprop(){
	_spline_shared_backprop().unwrap();
}

fn _spline_shared_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "output", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "target", tag![])?;


	let _o1 = g.new_op(Spline::new(&node1, &node2).shared_axes(&[0, -1]).init(Spline::elu_esque()), tag![])?;
	let _o2 = g.new_op(Spline::new(&node1, &node2).shared_axes(&[1]).init(Spline::tanh_esque()), tag![])?;
	let _o3 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.002;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;

	Ok(())
}