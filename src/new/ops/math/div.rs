use new::graph::{GraphDef, NodeID, DataID, OpID, PassID, Storage, GraphShapes, ErrorKind, Result};
use new::ops::{standard_op_name, Op, OpInstance, Pass};
use new::shape::{NodeShape, NodeDim};
use ndarray::{ArrayViewMutD, ArrayViewD, Zip};
use std::any::Any;

/// Div Op
///
/// Computed the elementwise division, which is then added to the output;
/// By default the value of denominator is broadcast to the shape of the numerator, this can be overridden.
pub struct Div {
	numerator_id: NodeID,
	denominator_id: NodeID,
	output_id: NodeID,
	broadcast_numerator: bool,
	name: Option<String>,
}

impl Div {
	pub fn new(numerator: &NodeID, denominator: &NodeID, output: &NodeID) -> Self {
		Div {
			numerator_id: numerator.clone(),
			denominator_id: denominator.clone(),
			output_id: output.clone(),
			broadcast_numerator: false,
			name: None,
		}
	}

	/// If true, broadcast the numerator to the denominators shape and use the denominator for shape inference.
	///
	/// Default: false
	pub fn broadcast_numerator(mut self, broadcast: bool) -> Self {
		self.broadcast_numerator = broadcast;
		self
	}
}

impl Op for Div {
	type InstanceType = DivInstance;

	fn type_name(&self) -> &'static str {
		"Div"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, _op_id: &OpID) -> Result<Self::InstanceType> {
		let name = standard_op_name(&self, &self.name, graph, &[self.numerator_id.clone(), self.denominator_id.clone()], &[self.output_id.clone()]);

		Ok(DivInstance{
			name: name,
			numerator_id: self.numerator_id.clone(),
			denominator_id: self.denominator_id.clone(),
			output_id: self.output_id.clone(),
			broadcast_numerator: self.broadcast_numerator,
			forward_id: graph.add_pass(DivForward::new(
				self.numerator_id.clone(),
				self.denominator_id.clone(),
				self.output_id.clone(),
				self.broadcast_numerator)),
			backward_id: graph.add_pass(DivBackward::new(
				self.numerator_id.clone(),
				self.denominator_id.clone(),
				self.output_id.clone(),
				self.broadcast_numerator)),
		})
	}
}


#[derive(Clone, Debug)] 
pub struct DivInstance{
	name: String,
	numerator_id: NodeID,
	denominator_id: NodeID,
	output_id: NodeID,
	broadcast_numerator: bool,
	forward_id: PassID,
	backward_id: PassID,
}

impl OpInstance for DivInstance {

	fn instance_name(&self) -> &str{&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){(vec![self.numerator_id.clone(),self.denominator_id.clone()], vec![self.output_id.clone()])}

	fn inner_passes(&self) -> Vec<PassID>{vec![self.forward_id.clone(), self.backward_id.clone()]}

	fn inner_ops(&self) -> Vec<OpID>{vec![]}

	fn inner_nodes(&self) -> Vec<NodeID>{vec![]}

	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{
		if self.broadcast_numerator {
			let output_shape: NodeShape = shapes.get_shape(&self.numerator_id).dimensions().iter().map(|dim|{
				match dim {
					&NodeDim::Known(1) => NodeDim::Unknown,
					&NodeDim::Known(x) => NodeDim::Known(x),
					_ => unreachable!(),
				}
			}).into();
			output_shape.merge(shapes.get_shape(&self.denominator_id))?;
			shapes.merge_with(&self.output_id, &output_shape)
		} else {
			let output_shape: NodeShape = shapes.get_shape(&self.denominator_id).dimensions().iter().map(|dim|{
				match dim {
					&NodeDim::Known(1) => NodeDim::Unknown,
					&NodeDim::Known(x) => NodeDim::Known(x),
					_ => unreachable!(),
				}
			}).into();
			output_shape.merge(shapes.get_shape(&self.numerator_id))?;
			shapes.merge_with(&self.output_id, &output_shape)
		}


	}

}


#[derive(Clone, Debug)]
struct DivForward {
	numerator_id: NodeID,
	denominator_id: NodeID,
	output_id: NodeID,
	broadcast_numerator: bool,
}

impl DivForward {
	pub fn new(numerator_id: NodeID, denominator_id: NodeID, output_id: NodeID, broadcast_numerator: bool) -> Self {
		DivForward {
			numerator_id,
			denominator_id,
			output_id,
			broadcast_numerator,
		}
	}
}

impl Pass for DivForward {
	fn type_name(&self) -> &'static str {"DivForward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(
			vec![self.numerator_id.value_id(), self.denominator_id.value_id()],
			vec![self.output_id.value_id()]
		)
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let numerator: ArrayViewD<f32> = data.get(&self.numerator_id.value_id())?;
		let denominator: ArrayViewD<f32> = data.get(&self.denominator_id.value_id())?;
		let mut output: ArrayViewMutD<f32> = data.get_mut(&self.output_id.value_id())?;


		if self.broadcast_numerator {
			ensure!(
				denominator.shape() == output.shape(),
				ErrorKind::PassError(self.instance_name(data.graph()), format!("denominator shape: {:?} did not match output shape: {:?}", denominator.shape(), output.shape()))
			);
			ensure!(
				numerator.broadcast(output.shape()).is_some(), 
				ErrorKind::PassError(self.instance_name(data.graph()), format!("Could not broadcast numerator shape: {:?} to output shape: {:?}", numerator.shape(), output.shape()))
			);
			let iter = denominator.exact_chunks(numerator.shape()).into_iter()
				.zip(output.exact_chunks_mut(numerator.shape()));
			for (denominator_chunk, mut out_chunk) in iter {
				Zip::from(&mut out_chunk)
					.and(&numerator)
					.and(&denominator_chunk)
					.apply(|output, numerator, denominator| {
						*output += numerator / denominator;
					});
			}
		} else {
			ensure!(
				numerator.shape() == output.shape(),
				ErrorKind::PassError(self.instance_name(data.graph()), format!("numerator shape: {:?} did not match output shape: {:?}", numerator.shape(), output.shape()))
			);
			ensure!(
				denominator.broadcast(output.shape()).is_some(), 
				ErrorKind::PassError(self.instance_name(data.graph()), format!("Could not broadcast denominator shape: {:?} to output shape: {:?}", denominator.shape(), output.shape()))
			);
			let iter = numerator.exact_chunks(denominator.shape()).into_iter()
				.zip(output.exact_chunks_mut(denominator.shape()));
			for (numerator_chunk, mut out_chunk) in iter {
				Zip::from(&mut out_chunk)
					.and(&numerator_chunk)
					.and(&denominator)
					.apply(|output, numerator, denominator| {
						*output += numerator / denominator;
					});
			}
		}



		Ok(Box::new(()))
	}
}


#[derive(Clone, Debug)]
struct DivBackward {
	numerator_id: NodeID,
	denominator_id: NodeID,
	output_id: NodeID,
	broadcast_numerator: bool,
}

impl DivBackward {
	pub fn new(numerator_id: NodeID, denominator_id: NodeID, output_id: NodeID, broadcast_numerator: bool) -> Self {
		DivBackward {
			numerator_id,
			denominator_id,
			output_id,
			broadcast_numerator,
		}
	}
}

impl Pass for DivBackward {
	fn type_name(&self) -> &'static str {"DivBackward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(
			vec![self.numerator_id.value_id(), self.denominator_id.value_id(), self.output_id.gradient_id()],
			vec![self.numerator_id.gradient_id(),self.denominator_id.gradient_id()]
		)
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let numerator: ArrayViewD<f32> = data.get(&self.numerator_id.value_id())?;
		let denominator: ArrayViewD<f32> = data.get(&self.denominator_id.value_id())?;
		let output_grad = data.get(&self.output_id.gradient_id())?;
		
		if self.broadcast_numerator {
				ensure!(
					denominator.shape() == output_grad.shape(),
					ErrorKind::PassError(self.instance_name(data.graph()), format!("denominator shape: {:?} did not match output shape: {:?}", denominator.shape(), output_grad.shape()))
				);
				ensure!(
					numerator.broadcast(output_grad.shape()).is_some(), 
					ErrorKind::PassError(self.instance_name(data.graph()), format!("Could not broadcast numerator shape: {:?} to output shape: {:?}", numerator.shape(), output_grad.shape()))
				);
			if data.is_required(&self.numerator_id.gradient_id()) && data.is_required(&self.denominator_id.gradient_id()) {
				let mut numerator_grad = data.get_mut(&self.numerator_id.gradient_id())?;
				let mut denominator_grad = data.get_mut(&self.denominator_id.gradient_id())?;

				let iter = denominator.exact_chunks(numerator.shape()).into_iter()
					.zip(denominator_grad.exact_chunks_mut(numerator.shape()))
					.zip(output_grad.exact_chunks(numerator.shape()));

				for ((denominator_chunk, mut denominator_grad_chunk) , out_grad_chunk) in iter {

					Zip::from(&mut numerator_grad)
						.and(&denominator_chunk)
						.and(&out_grad_chunk)
						.apply(|numerator_grad, denominator, out_grad| {
							*numerator_grad += out_grad/denominator;
						});

					Zip::from(&mut denominator_grad_chunk)
						.and(&numerator)
						.and(&denominator_chunk)
						.and(&out_grad_chunk)
						.apply(|denominator_grad, numerator, denominator, out_grad| {
							*denominator_grad += -numerator * out_grad / (denominator * denominator);
						});

				}

			} else if data.is_required(&self.numerator_id.gradient_id()) {
				let mut numerator_grad = data.get_mut(&self.numerator_id.gradient_id())?;

				let iter = denominator.exact_chunks(numerator.shape()).into_iter()
					.zip(output_grad.exact_chunks(numerator.shape()));

				for (mut denominator_chunk, out_grad_chunk) in iter {

					Zip::from(&mut numerator_grad)
						.and(&denominator_chunk)
						.and(&out_grad_chunk)
						.apply(|numerator_grad, denominator, out_grad| {
							*numerator_grad += out_grad/denominator;
						});
				}
			} else if data.is_required(&self.denominator_id.gradient_id()) {
				let mut denominator_grad = data.get_mut(&self.denominator_id.gradient_id())?;

				let iter = denominator.exact_chunks(numerator.shape()).into_iter()
					.zip(denominator_grad.exact_chunks_mut(numerator.shape()))
					.zip(output_grad.exact_chunks(numerator.shape()));

				for ((denominator_chunk, mut denominator_grad_chunk) , out_grad_chunk) in iter {

					Zip::from(&mut denominator_grad_chunk)
						.and(&numerator)
						.and(&denominator_chunk)
						.and(&out_grad_chunk)
						.apply(|denominator_grad, numerator, denominator, out_grad| {
							*denominator_grad += -numerator * out_grad / (denominator * denominator);
						});
				}
			}
		} else {
				ensure!(
					numerator.shape() == output_grad.shape(),
					ErrorKind::PassError(self.instance_name(data.graph()), format!("numerator shape: {:?} did not match output shape: {:?}", numerator.shape(), output_grad.shape()))
				);
						ensure!(
					denominator.broadcast(output_grad.shape()).is_some(), 
					ErrorKind::PassError(self.instance_name(data.graph()), format!("Could not broadcast denominator shape: {:?} to output shape: {:?}", denominator.shape(), output_grad.shape()))
				);

				if data.is_required(&self.numerator_id.gradient_id()) && data.is_required(&self.denominator_id.gradient_id()) {
				let mut numerator_grad = data.get_mut(&self.numerator_id.gradient_id())?;
				let mut denominator_grad = data.get_mut(&self.denominator_id.gradient_id())?;

				let iter = numerator.exact_chunks(denominator.shape()).into_iter()
					.zip(numerator_grad.exact_chunks_mut(denominator.shape()))
					.zip(output_grad.exact_chunks(denominator.shape()));

				for ((numerator_chunk, mut numerator_grad_chunk) , out_grad_chunk) in iter {

					Zip::from(&mut numerator_grad_chunk)
						.and(&denominator)
						.and(&out_grad_chunk)
						.apply(|numerator_grad, denominator, out_grad| {
							*numerator_grad += out_grad/denominator;
						});

					Zip::from(&mut denominator_grad)
						.and(&numerator_chunk)
						.and(&denominator)
						.and(&out_grad_chunk)
						.apply(|denominator_grad, numerator, denominator, out_grad| {
							*denominator_grad += -numerator * out_grad / (denominator * denominator);
						});

				}

			} else if data.is_required(&self.numerator_id.gradient_id()) {
				let mut numerator_grad = data.get_mut(&self.numerator_id.gradient_id())?;

				let iter = numerator_grad.exact_chunks_mut(denominator.shape()).into_iter()
					.zip(output_grad.exact_chunks(denominator.shape()));

				for (mut numerator_grad_chunk, out_grad_chunk) in iter {

					Zip::from(&mut numerator_grad_chunk)
						.and(&denominator)
						.and(&out_grad_chunk)
						.apply(|numerator_grad, denominator, out_grad| {
							*numerator_grad += out_grad/denominator;
						});
				}
			} else if data.is_required(&self.denominator_id.gradient_id()) {
				let mut denominator_grad = data.get_mut(&self.denominator_id.gradient_id())?;

				let iter = numerator.exact_chunks(denominator.shape()).into_iter()
					.zip(output_grad.exact_chunks(denominator.shape()));

				for (numerator_chunk, out_grad_chunk) in iter {

					Zip::from(&mut denominator_grad)
						.and(&numerator_chunk)
						.and(&denominator)
						.and(&out_grad_chunk)
						.apply(|denominator_grad, numerator, denominator, out_grad| {
							*denominator_grad += -numerator * out_grad / (denominator * denominator);
						});
				}
			}
		}




		Ok(Box::new(()))
	}
}


#[test]
fn test_div_backprop(){
	_div_backprop().unwrap();
}

fn _div_backprop() -> Result<()>{
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use new::ops::loss::mse::Mse;
	use ordermap::OrderMap;
	use rand::thread_rng;
	use rand::distributions::{Sample, Range};

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "numerator", tag![])?;
	let node2 = g.new_node(shape![1, 1, 16], "denominator", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "output", tag![])?;
	let node4 = g.new_node(shape![7, 5, 16], "target", tag![])?;

	let _o1 = g.new_op(Div::new(&node1, &node2, &node3), tag![])?;
	let _o2 = g.new_op(Mse::new(&node3, &node4), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;

	let sample: Box<::std::ops::FnMut() -> f64 + 'static> = Box::new(|| {
		let rng = &mut thread_rng();
		let mut range = Range::new(0.1, 10.0);
		range.sample(rng)
	});
	let mut override_dist = OrderMap::new();
	override_dist.insert(node2.clone(), sample);

	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut override_dist)?;

	Ok(())
}

#[test]
fn test_div_numerator_broadcast_backprop(){
	_div_numerator_broadcast_backprop().unwrap();
}

fn _div_numerator_broadcast_backprop() -> Result<()>{
	use new::graph::GraphDef;
	use new::ops::numeric_check::numeric_test;
	use new::ops::loss::mse::Mse;
	use ordermap::OrderMap;
	use rand::thread_rng;
	use rand::distributions::{Sample, Range};

	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![1, 1, 16], "numerator", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "denominator", tag![])?;
	let node3 = g.new_node(shape![7, 5, 16], "output", tag![])?;
	let node4 = g.new_node(shape![7, 5, 16], "target", tag![])?;

	let _o1 = g.new_op(Div::new(&node1, &node2, &node3).broadcast_numerator(true), tag![])?;
	let _o2 = g.new_op(Mse::new(&node3, &node4), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;

	let sample: Box<::std::ops::FnMut() -> f64 + 'static> = Box::new(|| {
		let rng = &mut thread_rng();
		let mut range = Range::new(0.1, 10.0);
		range.sample(rng)
	});
	let mut override_dist = OrderMap::new();
	override_dist.insert(node2.clone(), sample);

	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut override_dist)?;

	Ok(())
}