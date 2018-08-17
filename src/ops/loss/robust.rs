use graph::{GraphDef, GraphShapes, ErrorKind, Result};
use id::{NodeID, DataID, OpID, PassID};
use storage::Storage;
use ops::{standard_op_name, Op, OpInstance, Pass};
use ops::loss::LossType;
use shape::NodeShape;
use smallvec::SmallVec;
use ndarray::{Dimension, Zip};
use std::any::Any;
use std::f32;
use std::num;


/// An `Op` which implements a range of robust loss functions.
///
/// By default this `Op` has no output and will generate loss and gradients.
///
/// If `output()` is set, the Mse loss will be written to that Node,
/// and instead of generating gradients this loss function will backprop gradients from the output node.
///
/// Based on the paper: A More General Robust Loss Function https://arxiv.org/pdf/1701.03077.pdf Eq.13 & Eq.14
/// Note that:
///
/// when power(α) == 2, this is the L2 loss
///
/// when power(α) == 1, this is the pseudo-Huber/Charbonnier loss (smooth L1 loss)
///
/// when power(α) == 0, this is the Cauchy/Lorentzian loss
///
/// when power(α) == -2, this is the Geman-McClure loss
///
/// when power(α) == -∞, this is the Welsch/Leclerc loss
///
/// The scale(c) is the range of values either size of zero for which the loss will closely approximate the L2 loss.
/// A small scale value will mean that small errors get treated as larger errors.
/// See paper for futher details.
///
/// ρ(x,α,c) = 
/// if α == 0 : log(0.5*(x/c)^2+ 1)
/// if α == -∞: 1 - exp(-0.5 *(x/c)^2)
/// else      : z(α)/α * (((x/c)^2/z(α) + 1)^(α/2) − 1)
/// where z(α) = max(1, 2 - α)
#[must_use]
#[derive(Clone, Debug)]
pub struct Robust {
	input1_id: NodeID,
	input2_id: NodeID,
	output: Option<NodeID>,
	mean_axes: SmallVec<[isize; 6]>,
	keep_dims: bool,
	scale: f32,
	power: f32,
	multiplier: f32,
	name: Option<String>,
}

impl Robust {
	pub fn new(input1: &NodeID, input2: &NodeID, scale: f32, power: f32) -> Self {
		Robust {
			input1_id: input1.clone(),
			input2_id: input2.clone(),
			output: None,
			mean_axes: SmallVec::new(),
			keep_dims: false,
			multiplier: 1.0,
			scale: scale,
			power: power,
			name: None,
		}
	}

	/// If set this `Op` will output to the supplied node, any rely no other use ops to generate loss and gradients
	/// The output node must have the same size as the input node unless reductions are applied using `.mean_axes()`.
	///
	/// Default: None.
	pub fn output(mut self, output: &NodeID) -> Self {
		self.output = Some(output.clone());
		self
	}

	/// The axes supplied will be grouped when finding the mean,
	/// with the operation repeated across the axes not supplied.
	///
	/// `axes` can be in the range [-input.ndims(), input.ndims());
	/// If no axes are supplied then no mean operation is applied.
	pub fn mean_axes(mut self, mean_axes: &[isize]) -> Self {
		self.mean_axes = mean_axes.iter().cloned().collect();
		self
	}

	/// If `true` the reduced mean_axes still appear in the output with size 1, otherwise they are removed.
	///
	/// Default: `false`
	pub fn keep_dims(mut self, keep_dims: bool) -> Self {
		self.keep_dims = keep_dims;
		self
	}

	/// Applies a multiplier to the output or to the loss generated.
	pub fn multiplier(mut self, multiplier: f32) -> Self {
		self.multiplier = multiplier;
		self
	}
}


impl Op for Robust {
	type InstanceType = RobustInstance;

	fn type_name(&self) -> &'static str {
		"Robust"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef) -> Result<Self::InstanceType> {

		let name =  if let Some(ref output_id) = self.output {
			standard_op_name(&self, &self.name, graph, &[self.input1_id.clone(), self.input2_id.clone()], &[output_id.clone()])
		} else {
			standard_op_name(&self, &self.name, graph, &[self.input1_id.clone(), self.input2_id.clone()], &[])
		};

		let loss_type = if let Some(output_id) = self.output {
			LossType::Output{
				output_id: output_id.clone(),
				forward_id: graph.add_pass(RobustForward::new(
					self.multiplier,
					self.scale,
					self.power,
					self.input1_id.clone(),
					self.input2_id.clone(),
					output_id.clone(),
					self.mean_axes.clone(),
					self.keep_dims)),
				backward_id: graph.add_pass(RobustBackward::new(
					self.multiplier,
					self.scale,
					self.power,
					self.input1_id.clone(),
					self.input2_id.clone(),
					output_id.clone(),
					self.mean_axes.clone(),
					self.keep_dims)),
			}
		} else {
			LossType::Joint{
				pass_id: graph.add_pass(RobustJointPass::new(
					self.multiplier,
					self.scale,
					self.power,
					self.input1_id.clone(),
					self.input2_id.clone(),
					self.mean_axes.clone()))
			}
		};

		Ok(RobustInstance{
			name: name,
			multiplier: self.multiplier,
			scale: self.scale,
			power: self.power,
			input1_id: self.input1_id.clone(),
			input2_id: self.input2_id.clone(),
			loss_type: loss_type,
			mean_axes: self.mean_axes,
			keep_dims: self.keep_dims,
		})
	}
}


#[derive(Clone, Debug)] 
pub struct RobustInstance {
	name: String,
	multiplier: f32,
	scale: f32,
	power: f32,
	input1_id: NodeID,
	input2_id: NodeID,
	loss_type: LossType,
	mean_axes: SmallVec<[isize; 6]>,
	keep_dims: bool,
}

impl OpInstance for RobustInstance {

	fn name(&self) -> &str {&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		match &self.loss_type {
			&LossType::Joint{..} => (vec![self.input1_id.clone(), self.input2_id.clone()], vec![]),
			&LossType::Output{ref output_id, ..} => (vec![self.input1_id.clone(), self.input2_id.clone()], vec![output_id.clone()]),
		}
	}

	fn inner_passes(&self) -> Vec<PassID> {
		match &self.loss_type {
			&LossType::Joint{ref pass_id} => vec![pass_id.clone()],
			&LossType::Output{ref forward_id, ref backward_id, ..} => vec![forward_id.clone(), backward_id.clone()],
		}
	}

	fn inner_ops(&self) -> Vec<OpID> {
		vec![]
	}

	fn inner_nodes(&self) -> Vec<NodeID> {
		vec![]
	}

	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{
		if let &LossType::Output{ref output_id, ..} = &self.loss_type {
			let input1_shape = shapes.get_shape(&self.input1_id).to_data_shape()?;
			let input2_shape = shapes.get_shape(&self.input2_id).to_data_shape()?;
			ensure!(input1_shape == input2_shape, "Shape of input1 did not match shape of input2");
			let output_shape: NodeShape = calc_output_shape(input1_shape.slice(), &self.mean_axes, self.keep_dims).into();
			shapes.merge_with(output_id, &output_shape)
		} else {
			Ok(())
		}
	}

}

fn calc_output_shape(input_shape: &[usize], axes: &[isize], keep_dims: bool) -> SmallVec<[usize; 6]> {
	let reduce_mask = reduction_mask(input_shape.len(), &axes);
	if keep_dims {
		input_shape.iter().zip(&reduce_mask).map(|(&dim, &reduce)| {
				if reduce {1} else {dim}
			}).collect()
	} else {
		input_shape.iter().zip(&reduce_mask).filter_map(|(&dim, &reduce)| {
				if reduce {None} else {Some(dim)}
			}).collect()
	}
}

/// Returns a mask indicating whether an axis should be reduced based on the axes list
fn reduction_mask(len: usize, axes: &[isize]) -> SmallVec<[bool; 6]> {
	let mut reduce = SmallVec::with_capacity(len);
	for _ in 0..len {
		reduce.push(false);
	}
	for axis in axes {
		reduce[(axis + len as isize) as usize % len] = true;
	}
	reduce
}

#[derive(Clone, Debug)]
struct RobustJointPass {
	multiplier: f32,
	scale: f32,
	power: f32,
	input1_id: NodeID,
	input2_id: NodeID,
	mean_axes: SmallVec<[isize; 6]>,
}

impl RobustJointPass {
	pub fn new(multiplier: f32, scale: f32, power: f32, input1_id: NodeID, input2_id: NodeID, mean_axes: SmallVec<[isize; 6]>) -> Self {
		RobustJointPass {
			multiplier,
			scale,
			power,
			input1_id,
			input2_id,
			mean_axes,
		}
	}
}

impl Pass for RobustJointPass {
	fn type_name(&self) -> &'static str {"RobustJointPass"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input1_id.value_id(), self.input2_id.value_id()],
		vec![self.input1_id.gradient_id(), self.input2_id.gradient_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input1 = data.get(&self.input1_id.value_id())?;
		let input2 = data.get(&self.input2_id.value_id())?;

		ensure!(
			input2.shape() == input1.shape(),
			ErrorKind::PassError(self.name(), format!("input1 shape: {:?} did not match input2 shape: {:?}", input2.shape(), input1.shape()))
		);

		let input_shape: SmallVec<[usize; 6]> = input1.shape().iter().cloned().collect();

		let divisor: usize = input_shape.iter().zip(reduction_mask(input_shape.len(), &self.mean_axes)).filter_map(|(dim, reduce)| if reduce{Some(dim)} else {None}).product();
		let multiplier = self.multiplier/divisor as f32;

		//let output_shape_actual = calc_output_shape(&input_shape, &self.axes, self.keep_dims);
		//let output_shape_keep_dims = calc_output_shape(&input_shape, &self.mean_axes, true);

		let mut error = 0.0;

		if data.is_required(&self.input1_id.gradient_id()) && data.is_required(&self.input2_id.gradient_id()) {
			let mut input1_grad = data.get_mut(&self.input1_id.gradient_id())?;
			let mut input2_grad = data.get_mut(&self.input2_id.gradient_id())?;

			// let iter1 = input1.into_iter().zip(input2);
			// let iter2 = input1_grad.into_iter().zip(input2_grad);
		
			let c = self.scale; // use notation from paper
			let a = self.power;
			if a == 0.0 {
				Zip::from(&input1) 
					.and(&input2)
					.and(&mut input1_grad) 
					.and(&mut input2_grad) 
					.apply(|input1, input2, input1_grad, input2_grad| { 
						let x = input1-input2;
						error += multiplier * (0.5*(x/c)*(x/c)).ln_1p();
						*input1_grad +=  multiplier * 2.0 * x / (x*x + 2.0*c*c);
						*input2_grad += -multiplier * 2.0 * x / (x*x + 2.0*c*c);
					});
			} else if a == f32::NEG_INFINITY {
				Zip::from(input1) 
					.and(input2)
					.and(&mut input1_grad) 
					.and(&mut input2_grad) 
					.apply(|input1, input2, input1_grad, input2_grad| { 
						let x = input1-input2;
						error += -multiplier * (-0.5*(x/c)*(x/c)).exp_m1();
						*input1_grad +=  multiplier * x/(c*c) * (-0.5*(x/c)*(x/c)).exp();
						*input2_grad += -multiplier * x/(c*c) * (-0.5*(x/c)*(x/c)).exp();
					});
			} else if a == 1.0 {
				Zip::from(&input1) 
					.and(&input2)
					.and(&mut input1_grad) 
					.and(&mut input2_grad) 
					.apply(|input1, input2, input1_grad, input2_grad| { 
						let x = input1-input2;
						error += multiplier * (((x/c)*(x/c) + 1.0).sqrt() - 1.0); //TODO change to numerically stable version https://stackoverflow.com/questions/32444817/numerically-stable-evaluation-of-sqrtxa-sqrtx
						*input1_grad +=  multiplier * x/((c*c) * ((x/c)*(x/c) + 1.0).sqrt());
						*input2_grad += -multiplier * x/((c*c) * ((x/c)*(x/c) + 1.0).sqrt());
					});
			} else if a == 2.0 {
				Zip::from(&input1) 
					.and(&input2)
					.and(&mut input1_grad) 
					.and(&mut input2_grad) 
					.apply(|input1, input2, input1_grad, input2_grad| { 
						let x = input1-input2;
						error += multiplier * ((x/c)*(x/c))/a;;
						*input1_grad +=  multiplier * x/(c*c);
						*input2_grad += -multiplier * x/(c*c);
					});
			} else {
				let za = 1.0f32.max(2.0-a);
				Zip::from(&input1) 
					.and(&input2)
					.and(&mut input1_grad) 
					.and(&mut input2_grad) 
					.apply(|input1, input2, input1_grad, input2_grad| { 
						let x = input1-input2;
						error += multiplier * za / a *(((x/c)*(x/c)/za + 1.0).powf(0.5 * a) - 1.0);
						*input1_grad +=  multiplier * x/(c*c) * ((x/c)*(x/c)/za + 1.0).powf(0.5*a - 1.0);
						*input2_grad += -multiplier * x/(c*c) * ((x/c)*(x/c)/za + 1.0).powf(0.5*a - 1.0);
					});
			}
			

		} else if data.is_required(&self.input1_id.gradient_id()) {
			let mut input1_grad = data.get_mut(&self.input1_id.gradient_id())?;

			// let iter1 = input1.into_iter().zip(input2);
			// let iter2 = input1_grad.into_iter();

			let c = self.scale; // use notation from paper
			let a = self.power;
			if a == 0.0 {
				Zip::from(&input1) 
					.and(&input2)
					.and(&mut input1_grad) 
					.apply(|input1, input2, input1_grad| { 
						let x = input1-input2;
						error += multiplier * (0.5*(x/c)*(x/c)).ln_1p();
						*input1_grad +=  multiplier * 2.0 * x / (x*x + 2.0*c*c);
					});
			} else if a == f32::NEG_INFINITY {
				Zip::from(&input1) 
					.and(&input2)
					.and(&mut input1_grad) 
					.apply(|input1, input2, input1_grad| { 
						let x = input1-input2;
						error += multiplier * (-0.5*(x/c)*(x/c)).exp_m1();
						*input1_grad +=  multiplier * x/(c*c) * (-0.5*(x/c)*(x/c)).exp();
					});
			} else if a == 1.0 {
				Zip::from(&input1) 
					.and(&input2)
					.and(&mut input1_grad) 
					.apply(|input1, input2, input1_grad| { 
						let x = input1-input2;
						error += multiplier * (((x/c)*(x/c) + 1.0).sqrt() - 1.0); //TODO change to numerically stable version https://stackoverflow.com/questions/32444817/numerically-stable-evaluation-of-sqrtxa-sqrtx
						*input1_grad +=  multiplier * x/((c*c) * ((x/c)*(x/c) + 1.0).sqrt());
					});
			} else if a == 2.0 {
				Zip::from(&input1) 
					.and(&input2)
					.and(&mut input1_grad) 
					.apply(|input1, input2, input1_grad| { 
						let x = input1-input2;
						error += multiplier * ((x/c)*(x/c))/a;;
						*input1_grad +=  multiplier * x/(c*c);
					});
			} else {
				let za = 1.0f32.max(2.0-a);
				Zip::from(&input1) 
					.and(&input2)
					.and(&mut input1_grad) 
					.apply(|input1, input2, input1_grad| { 
						let x = input1-input2;
						error += multiplier * za / a *(((x/c)*(x/c)/za + 1.0).powf(0.5 * a) - 1.0);
						*input1_grad +=  multiplier * x/(c*c) * ((x/c)*(x/c)/za + 1.0).powf(0.5*a - 1.0);
					});
			}

		} else if data.is_required(&self.input2_id.gradient_id()) {
			let mut input2_grad = data.get_mut(&self.input2_id.gradient_id())?;

			// let iter1 = input1.exact_chunks(output_shape_keep_dims.as_slice()).into_iter()
			// 	.zip(input2.exact_chunks(output_shape_keep_dims.as_slice()));
			// let iter2 = input2_grad.exact_chunks_mut(output_shape_keep_dims.as_slice()).into_iter();

			
			let c = self.scale; // use notation from paper
			let a = self.power;
			if a == 0.0 {
				Zip::from(&input1) 
					.and(&input2)
					.and(&mut input2_grad) 
					.apply(|input1, input2, input2_grad| { 
						let x = input1-input2;
						error += multiplier * (0.5*(x/c)*(x/c)).ln_1p();
						*input2_grad += -multiplier * 2.0 * x / (x*x + 2.0*c*c);
					});
			} else if a == f32::NEG_INFINITY {
				Zip::from(&input1) 
					.and(&input2)
					.and(&mut input2_grad) 
					.apply(|input1, input2, input2_grad| { 
						let x = input1-input2;
						error += multiplier * (-0.5*(x/c)*(x/c)).exp_m1();
						*input2_grad += -multiplier * x/(c*c) * (-0.5*(x/c)*(x/c)).exp();
					});
			} else if a == 1.0 {
				Zip::from(&input1) 
					.and(&input2)
					.and(&mut input2_grad) 
					.apply(|input1, input2, input2_grad| { 
						let x = input1-input2;
						error += multiplier * (((x/c)*(x/c) + 1.0).sqrt() - 1.0); //TODO change to numerically stable version https://stackoverflow.com/questions/32444817/numerically-stable-evaluation-of-sqrtxa-sqrtx
						*input2_grad += -multiplier * x/((c*c) * ((x/c)*(x/c) + 1.0).sqrt());
					});
			} else if a == 2.0 {
				Zip::from(&input1) 
					.and(&input2)
					.and(&mut input2_grad) 
					.apply(|input1, input2, input2_grad| { 
						let x = input1-input2;
						error += multiplier * ((x/c)*(x/c))/a;;
						*input2_grad += -multiplier * x/(c*c);
					});
			} else {
				let za = 1.0f32.max(2.0-a);
				Zip::from(&input1) 
					.and(&input2)
					.and(&mut input2_grad) 
					.apply(|input1, input2, input2_grad| { 
						let x = input1-input2;
						error += multiplier * za / a *(((x/c)*(x/c)/za + 1.0).powf(0.5 * a) - 1.0);
						*input2_grad += -multiplier * x/(c*c) * ((x/c)*(x/c)/za + 1.0).powf(0.5*a - 1.0);
					});
			}
			
		}

		data.loss_add(error);

		Ok(Box::new(()))
	}
}


#[derive(Clone, Debug)]
struct RobustForward {
	multiplier: f32,
	scale: f32,
	power: f32,
	input1_id: NodeID,
	input2_id: NodeID,
	output_id: NodeID,
	mean_axes: SmallVec<[isize; 6]>,
	keep_dims: bool,
}

impl RobustForward {
	pub fn new(multiplier: f32, scale: f32, power: f32, input1_id: NodeID, input2_id: NodeID, output_id: NodeID, mean_axes: SmallVec<[isize; 6]>, keep_dims: bool) -> Self {
		RobustForward {
			multiplier,
			scale,
			power,
			input1_id,
			input2_id,
			output_id,
			mean_axes,
			keep_dims,
		}
	}
}

impl Pass for RobustForward {
	fn type_name(&self) -> &'static str {"RobustForward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input1_id.value_id(), self.input2_id.value_id()],
		vec![self.output_id.value_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input1 = data.get(&self.input1_id.value_id())?;
		let input2 = data.get(&self.input2_id.value_id())?;
		let output = data.get_mut(&self.output_id.value_id())?;

		ensure!(
			input2.shape() == input1.shape(),
			ErrorKind::PassError(self.name(), format!("input1 shape: {:?} did not match input2 shape: {:?}", input2.shape(), input1.shape()))
		);

		let input_shape: SmallVec<[usize; 6]> = input1.shape().iter().cloned().collect();
		let output_shape: SmallVec<[usize; 6]> = output.shape().iter().cloned().collect();

		let divisor: usize = input_shape.iter().zip(reduction_mask(input_shape.len(), &self.mean_axes)).filter_map(|(dim, reduce)| if reduce{Some(dim)} else {None}).product();
		let multiplier = self.multiplier/divisor as f32;

		let output_shape_actual = calc_output_shape(&input_shape, &self.mean_axes, self.keep_dims);
		let output_shape_keep_dims = calc_output_shape(&input_shape, &self.mean_axes, true);

		ensure!(output_shape_actual.as_slice() == output_shape.as_slice(), "Output shape {:?} does not match reduced input shape {:?}", output_shape.as_slice(), output_shape_actual.as_slice());

		let iter = input1.exact_chunks(output_shape_keep_dims.as_slice()).into_iter()
			.zip(input2.exact_chunks(output_shape_keep_dims.as_slice()));

		let mut output = output.into_shape(&output_shape_keep_dims[..]).expect("This should have been caught by the ensure above");;

		for (input1_chunk, input2_chunk) in iter {
			let c = self.scale; // use notation from paper
			let a = self.power;
			if a == 0.0 {
				Zip::from(&mut output) 
					.and(&input1_chunk) 
					.and(&input2_chunk) 
					.apply(|output, input1, input2| { 
						let x = input1 - input2;
						*output += multiplier * (0.5*(x/c)*(x/c)).ln_1p();
					});
			} else if a == f32::NEG_INFINITY {
				Zip::from(&mut output) 
					.and(&input1_chunk) 
					.and(&input2_chunk) 
					.apply(|output, input1, input2| { 
						let x = input1 - input2;
						*output += -multiplier * (-0.5*(x/c)*(x/c)).exp_m1();
					});
			} else if a == 1.0 {
				Zip::from(&mut output) 
					.and(&input1_chunk) 
					.and(&input2_chunk) 
					.apply(|output, input1, input2| { 
						let x = input1 - input2;
						*output += multiplier * (((x/c)*(x/c) + 1.0).sqrt() - 1.0); //TODO change to numerically stable version https://stackoverflow.com/questions/32444817/numerically-stable-evaluation-of-sqrtxa-sqrtx
					});
			} else if a == 2.0 {
				Zip::from(&mut output) 
					.and(&input1_chunk) 
					.and(&input2_chunk) 
					.apply(|output, input1, input2| { 
						let x = input1 - input2;
						*output += multiplier * ((x/c)*(x/c))/a;
					});
			} else {
				let za = 1.0f32.max(2.0-a);
				Zip::from(&mut output) 
					.and(&input1_chunk) 
					.and(&input2_chunk) 
					.apply(|output, input1, input2| { 
						let x = input1 - input2;
						*output += multiplier * za / a *(((x/c)*(x/c)/za + 1.0).powf(0.5 * a) - 1.0);
					});
			}
		}

		Ok(Box::new(()))
	}
}

#[derive(Clone, Debug)]
struct RobustBackward {
	multiplier: f32,
	scale: f32,
	power: f32,
	input1_id: NodeID,
	input2_id: NodeID,
	output_id: NodeID,
	mean_axes: SmallVec<[isize; 6]>,
	keep_dims: bool,
}

impl RobustBackward {
	pub fn new(multiplier: f32, scale: f32, power: f32, input1_id: NodeID, input2_id: NodeID, output_id: NodeID, mean_axes: SmallVec<[isize; 6]>, keep_dims: bool) -> Self {
		RobustBackward {
			multiplier,
			scale,
			power,
			input1_id,
			input2_id,
			output_id,
			mean_axes,
			keep_dims,
		}
	}
}

impl Pass for RobustBackward {
	fn type_name(&self) -> &'static str {"RobustBackward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input1_id.value_id(), self.input2_id.value_id(), self.output_id.gradient_id()],
		vec![self.input1_id.gradient_id(), self.input2_id.gradient_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>>{
		let input1 = data.get(&self.input1_id.value_id())?;
		let input2 = data.get(&self.input2_id.value_id())?;
		let output_grad = data.get(&self.output_id.gradient_id())?;

		ensure!(
			input2.shape() == input1.shape(),
			ErrorKind::PassError(self.name(), format!("input1 shape: {:?} did not match input2 shape: {:?}", input2.shape(), input1.shape()))
		);

		let input_shape: SmallVec<[usize; 6]> = input1.shape().iter().cloned().collect();
		let output_shape: SmallVec<[usize; 6]> = output_grad.shape().iter().cloned().collect();

		let divisor: usize = input_shape.iter().zip(reduction_mask(input_shape.len(), &self.mean_axes)).filter_map(|(dim, reduce)| if reduce{Some(dim)} else {None}).product();
		let multiplier = self.multiplier/divisor as f32;

		let output_shape_actual = calc_output_shape(&input_shape, &self.mean_axes, self.keep_dims);
		let output_shape_keep_dims = calc_output_shape(&input_shape, &self.mean_axes, true);

		ensure!(output_shape_actual.as_slice() == output_shape.as_slice(), "Output shape {:?} does not match reduced input shape {:?}", output_shape.as_slice(), output_shape_actual.as_slice());

		let output_grad = output_grad.into_shape(&output_shape_keep_dims[..]).expect("This should have been caught by the ensure above");;

		if data.is_required(&self.input1_id.gradient_id()) && data.is_required(&self.input2_id.gradient_id()) {
			let mut input1_grad = data.get_mut(&self.input1_id.gradient_id())?;
			let mut input2_grad = data.get_mut(&self.input2_id.gradient_id())?;

			let iter1 = input1.exact_chunks(output_shape_keep_dims.as_slice()).into_iter()
				.zip(input2.exact_chunks(output_shape_keep_dims.as_slice()));
			let iter2 = input1_grad.exact_chunks_mut(output_shape_keep_dims.as_slice()).into_iter()
				.zip(input2_grad.exact_chunks_mut(output_shape_keep_dims.as_slice()));

			for ((input1_chunk, input2_chunk), (mut input1_grad_chunk, mut input2_grad_chunk)) in iter1.zip(iter2) {
				let c = self.scale; // use notation from paper
				let a = self.power;
				if a == 0.0 {
					Zip::from(&output_grad) 
						.and(&input1_chunk) 
						.and(&input2_chunk)
						.and(&mut input1_grad_chunk) 
						.and(&mut input2_grad_chunk) 
						.apply(|output_grad, input1, input2, input1_grad, input2_grad| { 
							let x = input1-input2;
							*input1_grad +=  multiplier*output_grad * 2.0 * x / (x*x + 2.0*c*c);
							*input2_grad += -multiplier*output_grad * 2.0 * x / (x*x + 2.0*c*c);
						});
				} else if a == f32::NEG_INFINITY {
					Zip::from(&output_grad) 
						.and(&input1_chunk) 
						.and(&input2_chunk)
						.and(&mut input1_grad_chunk) 
						.and(&mut input2_grad_chunk) 
						.apply(|output_grad, input1, input2, input1_grad, input2_grad| { 
							let x = input1-input2;
							*input1_grad +=  multiplier*output_grad * x/(c*c) * (-0.5*(x/c)*(x/c)).exp();
							*input2_grad += -multiplier*output_grad * x/(c*c) * (-0.5*(x/c)*(x/c)).exp();
						});
				} else if a == 1.0 {
					Zip::from(&output_grad) 
						.and(&input1_chunk) 
						.and(&input2_chunk)
						.and(&mut input1_grad_chunk) 
						.and(&mut input2_grad_chunk) 
						.apply(|output_grad, input1, input2, input1_grad, input2_grad| { 
							let x = input1-input2;
							*input1_grad +=  multiplier*output_grad * x/((c*c) * ((x/c)*(x/c) + 1.0).sqrt());
							*input2_grad += -multiplier*output_grad * x/((c*c) * ((x/c)*(x/c) + 1.0).sqrt());
						});
				} else if a == 2.0 {
					Zip::from(&output_grad) 
						.and(&input1_chunk) 
						.and(&input2_chunk)
						.and(&mut input1_grad_chunk) 
						.and(&mut input2_grad_chunk) 
						.apply(|output_grad, input1, input2, input1_grad, input2_grad| { 
							let x = input1-input2;
							*input1_grad +=  multiplier*output_grad * x/(c*c);
							*input2_grad += -multiplier*output_grad * x/(c*c);
						});
				} else {
					let za = 1.0f32.max(2.0-a);
					Zip::from(&output_grad) 
						.and(&input1_chunk) 
						.and(&input2_chunk)
						.and(&mut input1_grad_chunk) 
						.and(&mut input2_grad_chunk) 
						.apply(|output_grad, input1, input2, input1_grad, input2_grad| { 
							let x = input1-input2;
							*input1_grad +=  multiplier*output_grad * x/(c*c) * ((x/c)*(x/c)/za + 1.0).powf(0.5*a - 1.0);
							*input2_grad += -multiplier*output_grad * x/(c*c) * ((x/c)*(x/c)/za + 1.0).powf(0.5*a - 1.0);
						});
				}
			}

		} else if data.is_required(&self.input1_id.gradient_id()) {
			let mut input1_grad = data.get_mut(&self.input1_id.gradient_id())?;

			let iter1 = input1.exact_chunks(output_shape_keep_dims.as_slice()).into_iter()
				.zip(input2.exact_chunks(output_shape_keep_dims.as_slice()));
			let iter2 = input1_grad.exact_chunks_mut(output_shape_keep_dims.as_slice()).into_iter();

			for ((input1_chunk, input2_chunk), mut input1_grad_chunk) in iter1.zip(iter2) {
				let c = self.scale; // use notation from paper
				let a = self.power;
				if a.classify() == num::FpCategory::Zero {
					Zip::from(&output_grad) 
						.and(&input1_chunk) 
						.and(&input2_chunk)
						.and(&mut input1_grad_chunk) 
						.apply(|output_grad, input1, input2, input1_grad| { 
							let x = input1-input2;
							*input1_grad +=  multiplier*output_grad * 2.0 * x / (x*x + 2.0*c*c);
						});
				} else if a == f32::NEG_INFINITY {
					Zip::from(&output_grad) 
						.and(&input1_chunk) 
						.and(&input2_chunk)
						.and(&mut input1_grad_chunk) 
						.apply(|output_grad, input1, input2, input1_grad| { 
							let x = input1-input2;
							*input1_grad +=  multiplier*output_grad * x/(c*c) * (-0.5*(x/c)*(x/c)).exp();
						});
				} else if a == 1.0 {
					Zip::from(&output_grad) 
						.and(&input1_chunk) 
						.and(&input2_chunk)
						.and(&mut input1_grad_chunk) 
						.apply(|output_grad, input1, input2, input1_grad| { 
							let x = input1-input2;
							*input1_grad +=  multiplier*output_grad * x/((c*c) * ((x/c)*(x/c) + 1.0).sqrt());
						});
				} else if a == 2.0 {
					Zip::from(&output_grad) 
						.and(&input1_chunk) 
						.and(&input2_chunk)
						.and(&mut input1_grad_chunk) 
						.apply(|output_grad, input1, input2, input1_grad| { 
							let x = input1-input2;
							*input1_grad +=  multiplier*output_grad * x/(c*c);
						});
				} else {
					let za = 1.0f32.max(2.0-a);
					Zip::from(&output_grad) 
						.and(&input1_chunk) 
						.and(&input2_chunk)
						.and(&mut input1_grad_chunk) 
						.apply(|output_grad, input1, input2, input1_grad| { 
							let x = input1-input2;
							*input1_grad +=  multiplier*output_grad * x/(c*c) * ((x/c)*(x/c)/za + 1.0).powf(0.5*a - 1.0);
						});
				}
			}
		} else if data.is_required(&self.input2_id.gradient_id()) {
			let mut input2_grad = data.get_mut(&self.input2_id.gradient_id())?;

			let iter1 = input1.exact_chunks(output_shape_keep_dims.as_slice()).into_iter()
				.zip(input2.exact_chunks(output_shape_keep_dims.as_slice()));
			let iter2 = input2_grad.exact_chunks_mut(output_shape_keep_dims.as_slice()).into_iter();

			for ((input1_chunk, input2_chunk), mut input2_grad_chunk) in iter1.zip(iter2) {
				let c = self.scale; // use notation from paper
				let a = self.power;
				if a.classify() == num::FpCategory::Zero {
					Zip::from(&output_grad) 
						.and(&input1_chunk) 
						.and(&input2_chunk)
						.and(&mut input2_grad_chunk)
						.apply(|output_grad, input1, input2, input2_grad| { 
							let x = input1-input2;
							*input2_grad += -multiplier*output_grad * 2.0 * x / (x*x + 2.0*c*c);
						});
				} else if a == f32::NEG_INFINITY {
					Zip::from(&output_grad) 
						.and(&input1_chunk) 
						.and(&input2_chunk)
						.and(&mut input2_grad_chunk)
						.apply(|output_grad, input1, input2, input2_grad| { 
							let x = input1-input2;
							*input2_grad += -multiplier*output_grad * x/(c*c) * (-0.5*(x/c)*(x/c)).exp();
						});
				} else if a == 1.0 {
					Zip::from(&output_grad) 
						.and(&input1_chunk) 
						.and(&input2_chunk)
						.and(&mut input2_grad_chunk)
						.apply(|output_grad, input1, input2, input2_grad| { 
							let x = input1-input2;
							*input2_grad += -multiplier*output_grad * x/((c*c) * ((x/c)*(x/c) + 1.0).sqrt());
						});
				} else if a == 2.0 {
					Zip::from(&output_grad) 
						.and(&input1_chunk) 
						.and(&input2_chunk)
						.and(&mut input2_grad_chunk)
						.apply(|output_grad, input1, input2, input2_grad| { 
							let x = input1-input2;
							*input2_grad += -multiplier*output_grad * x/(c*c);
						});
				} else {
					let za = 1.0f32.max(2.0-a);
					Zip::from(&output_grad) 
						.and(&input1_chunk) 
						.and(&input2_chunk)
						.and(&mut input2_grad_chunk) 
						.apply(|output_grad, input1, input2, input2_grad| { 
							let x = input1-input2;
							*input2_grad += -multiplier*output_grad * x/(c*c) * ((x/c)*(x/c)/za + 1.0).powf(0.5*a - 1.0);
						});
				}
			}
		}


		Ok(Box::new(()))
	}
}



#[test]
fn test_robust_zero_backprop(){
	_robust_zero_backprop().unwrap();
}

fn _robust_zero_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use rand;
	use rand::distributions::{Range, Distribution};

	let mut rng = rand::thread_rng();
	let range = Range::new(0.1, 1.0);

	let power = 0.0;
	let scale = 1.0 + range.sample(&mut rng);
	let mult = range.sample(&mut rng);
	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "input2", tag![])?;

	let _o1 = g.new_op(Robust::new(&node1, &node2, scale, power).multiplier(mult), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut indexmap![])?;

	Ok(())
}


#[test]
fn test_robust_one_backprop(){
	_robust_one_backprop().unwrap();
}

fn _robust_one_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use rand;
	use rand::distributions::{Range, Distribution};

	let mut rng = rand::thread_rng();
	let range = Range::new(0.1, 1.0);

	let power = 1.0;
	let scale = 1.0 + range.sample(&mut rng);
	let mult = range.sample(&mut rng);
	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "input2", tag![])?;

	let _o1 = g.new_op(Robust::new(&node1, &node2, scale, power).multiplier(mult), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut indexmap![])?;

	Ok(())
}


#[test]
fn test_robust_two_backprop(){
	_robust_two_backprop().unwrap();
}

fn _robust_two_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use rand;
	use rand::distributions::{Range, Distribution};

	let mut rng = rand::thread_rng();
	let range = Range::new(0.1, 1.0);

	let power = 2.0;
	let scale = 1.0 + range.sample(&mut rng);
	let mult = range.sample(&mut rng);
	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "input2", tag![])?;

	let _o1 = g.new_op(Robust::new(&node1, &node2, scale, power).multiplier(mult), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut indexmap![])?;

	Ok(())
}


#[test]
fn test_robust_neg_inf_backprop(){
	_robust_neg_inf_backprop().unwrap();
}

fn _robust_neg_inf_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use rand;
	use rand::distributions::{Range, Distribution};

	let mut rng = rand::thread_rng();
	let range = Range::new(0.1, 1.0);

	let power = f32::NEG_INFINITY;
	let scale = 1.0 + range.sample(&mut rng);
	let mult = range.sample(&mut rng);
	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "input2", tag![])?;

	let _o1 = g.new_op(Robust::new(&node1, &node2, scale, power).multiplier(mult), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut indexmap![])?;

	Ok(())
}


#[test]
fn test_robust_rand_backprop(){
	_robust_rand_backprop().unwrap();
}

fn _robust_rand_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use rand;
	use rand::distributions::{Range, Distribution};

	let mut rng = rand::thread_rng();
	let range = Range::new(0.1, 1.0);
	let power_range = Range::new(-2.0, 2.0);

	for _ in 0..10 {
		let power = power_range.sample(&mut rng);
		let scale = 1.0 + range.sample(&mut rng);
		let mult = range.sample(&mut rng);
		let mut g = GraphDef::new();

		let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;
		let node2 = g.new_node(shape![7, 5, 16], "input2", tag![])?;

		let _o1 = g.new_op(Robust::new(&node1, &node2, scale, power).multiplier(mult), tag![])?;

		let iters = 10;
		let failures = 1;
		let tolerance = 0.002;
		let step_size = 1E-2;
		let default_variance = 1.0;
		numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut indexmap![])?;
	}

	Ok(())
}




#[test]
fn test_robust_zero_output_backprop(){
	_robust_zero_output_backprop().unwrap();
}

fn _robust_zero_output_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::proportional::Proportional;
	use rand;
	use rand::distributions::{Range, Distribution};

	let mut rng = rand::thread_rng();
	let range = Range::new(0.1, 1.0);

	let power = 0.0;
	let scale = 1.0 + range.sample(&mut rng);
	let mult = range.sample(&mut rng);
	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "input2", tag![])?;
	let node3 = g.new_node(shape![5], "output", tag![])?;

	let _o1 = g.new_op(Robust::new(&node1, &node2, scale, power).multiplier(mult).mean_axes(&[0, -1]).output(&node3), tag![])?;
	let _o2 = g.new_op(Proportional::new(&node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut indexmap![])?;

	Ok(())
}


#[test]
fn test_robust_one_output_backprop(){
	_robust_one_output_backprop().unwrap();
}

fn _robust_one_output_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::proportional::Proportional;
	use rand;
	use rand::distributions::{Range, Distribution};

	let mut rng = rand::thread_rng();
	let range = Range::new(0.1, 1.0);

	let power = 1.0;
	let scale = 1.0 + range.sample(&mut rng);
	let mult = range.sample(&mut rng);
	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "input2", tag![])?;
	let node3 = g.new_node(shape![5], "output", tag![])?;

	let _o1 = g.new_op(Robust::new(&node1, &node2, scale, power).multiplier(mult).mean_axes(&[0, -1]).output(&node3), tag![])?;
	let _o2 = g.new_op(Proportional::new(&node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut indexmap![])?;

	Ok(())
}


#[test]
fn test_robust_two_output_backprop(){
	_robust_two_output_backprop().unwrap();
}

fn _robust_two_output_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::proportional::Proportional;
	use rand;
	use rand::distributions::{Range, Distribution};

	let mut rng = rand::thread_rng();
	let range = Range::new(0.1, 1.0);

	let power = 2.0;
	let scale = 1.0 + range.sample(&mut rng);
	let mult = range.sample(&mut rng);
	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "input2", tag![])?;
	let node3 = g.new_node(shape![5], "output", tag![])?;

	let _o1 = g.new_op(Robust::new(&node1, &node2, scale, power).multiplier(mult).mean_axes(&[0, -1]).output(&node3), tag![])?;
	let _o2 = g.new_op(Proportional::new(&node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut indexmap![])?;

	Ok(())
}


#[test]
fn test_robust_neg_inf_output_backprop(){
	_robust_neg_inf_output_backprop().unwrap();
}

fn _robust_neg_inf_output_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::proportional::Proportional;
	use rand;
	use rand::distributions::{Range, Distribution};

	let mut rng = rand::thread_rng();
	let range = Range::new(0.1, 1.0);

	let power = f32::NEG_INFINITY;
	let scale = 1.0 + range.sample(&mut rng);
	let mult = range.sample(&mut rng);
	let mut g = GraphDef::new();

	let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;
	let node2 = g.new_node(shape![7, 5, 16], "input2", tag![])?;
	let node3 = g.new_node(shape![5], "output", tag![])?;

	let _o1 = g.new_op(Robust::new(&node1, &node2, scale, power).multiplier(mult).mean_axes(&[0, -1]).output(&node3), tag![])?;
	let _o2 = g.new_op(Proportional::new(&node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.001;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut indexmap![])?;

	Ok(())
}


#[test]
fn test_robust_rand_output_backprop(){
	_robust_rand_output_backprop().unwrap();
}

fn _robust_rand_output_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::proportional::Proportional;
	use rand;
	use rand::distributions::{Range, Distribution};

	let mut rng = rand::thread_rng();
	let range = Range::new(0.1, 1.0);
	let power_range = Range::new(-2.0, 2.0);

	for _ in 0..10 {
		let power = power_range.sample(&mut rng);
		let scale = 1.0 + range.sample(&mut rng);
		let mult = range.sample(&mut rng);
		let mut g = GraphDef::new();

		let node1 = g.new_node(shape![7, 5, 16], "input1", tag![])?;
		let node2 = g.new_node(shape![7, 5, 16], "input2", tag![])?;
		let node3 = g.new_node(shape![5], "output", tag![])?;

		let _o1 = g.new_op(Robust::new(&node1, &node2, scale, power).multiplier(mult).mean_axes(&[0, -1]).output(&node3), tag![])?;
		let _o2 = g.new_op(Proportional::new(&node3), tag![])?;

		let iters = 10;
		let failures = 1;
		let tolerance = 0.002;
		let step_size = 1E-2;
		let default_variance = 1.0;
		numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut indexmap![])?;
	}

	Ok(())
}








// ///`GeneralLoss` - A general implementation of a range of robust loss functions.
// /// A More General Robust Loss Function https://arxiv.org/pdf/1701.03077.pdf Eq.13 & Eq.14
// /// when power == 2, this is the L2 loss
// /// when power == 1, this is the charbonnier loss (smooth L1 loss)
// /// when power == 0, this is the Cauchy/Lorentzian loss
// /// the scale is the range of values either size of zero for which the loss will closely approximate the L2 loss
// /// a small scale value means small errors get treated as large errors.
// /// see paper for futher losses
// #[derive(Clone)] 
// pub struct GeneralLoss {
// 	name: String,
// 	input_id: NodeID,
// 	target_id: NodeID,
// 	scale: f32,
// 	power: f32,
// 	strength: f32,
// }

// impl GeneralLoss {
// 	pub fn new(input_id: &NodeID, target_id: &NodeID, strength: f32, scale: f32, power: f32, name: &str) -> Box<GeneralLoss>{
// 		Box::new(GeneralLoss{
// 			name: name.to_string(),
// 			input_id: input_id.clone(),
// 			target_id: target_id.clone(),
// 			scale: scale,
// 			power: power,
// 			strength: strength,
// 		})
// 	}
	
// 	pub fn new_default(input_id: &NodeID, target_id: &NodeID,) -> Box<GeneralLoss>{
// 		GeneralLoss::new(input_id, target_id, 1.0, 1.0, 1.0, "GeneralLoss")
// 	}
// }

// impl Operation for GeneralLoss {

// 	fn name(&self) -> &str{&self.name}
	
// 	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
// 		shapes[self.input_id.ind].collapse_ranges_to_minimum()
// 			.expect(&format!("Error: Input node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_id.ind].name, self.name));
// 	}
	
// 	fn input_node_IDs(&self) -> Vec<NodeID>{vec![self.input_id.clone(), self.target_id.clone()]}
	
// 	fn output_node_IDs(&self) -> Vec<NodeID>{vec![]}
	
// 	fn num_params(&self) -> usize {0}
	
// 	fn forward (&mut self, _data: &mut [RefCell<NodeData>], _params: &[f32]){}// No Output
	
// 	fn backward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32], _param_deriv: &mut [f32], error: &mut f32){
// 		let input = &mut *{data[self.input_id.ind].borrow_mut()};
// 		let target = &*{data[self.target_id.ind].borrow()};
// 		let input_size = input.shape.flat_size_single();
// 		let target_size = target.shape.flat_size_single();
		
				
// 		assert_eq!(input_size, target_size, "Error: Operation '{}' input and target node sizes were not equal during evaluation", self.name);
// 		assert_eq!(input.shape.n, target.shape.n, "Error: Operation '{}' input and target node 'n' were not equal during evaluation", self.name);

// 		let n = input.shape.flat_size_all();
// 		let input_deriv: &mut [f32] = &mut input.derivatives[..n];
// 		let input_values: &[f32] = &input.values[..n];
// 		let target_values: &[f32] =  &target.values[..n];
		
// 		let strength = self.strength/input_size as f32;
// 		let c = self.scale; // use notation from paper
// 		let a = self.power;
// 		if a.classify() == num::FpCategory::Zero {
// 			for i in 0..n{
// 				let x = input_values[i]-target_values[i];
// 				*error += strength * (0.5*(x/c)*(x/c)).ln_1p();
// 				input_deriv[i] += strength * 2.0 * x / (x*x + 2.0*c*c);
// 			}
// 		} else if a == f32::NEG_INFINITY {
// 			for i in 0..n{
// 				let x = input_values[i]-target_values[i];
// 				*error += -strength * (-0.5*(x/c)*(x/c)).exp_m1();
// 				input_deriv[i] += strength * x/(c*c) * (-0.5*(x/c)*(x/c)).exp();
// 			}
// 		} else if a == 1.0 {
// 			for i in 0..n{
// 				let x = input_values[i]-target_values[i];
// 				*error += strength *(((x/c)*(x/c) + 1.0).sqrt() - 1.0); //TODO change to numerically stable version https://stackoverflow.com/questions/32444817/numerically-stable-evaluation-of-sqrtxa-sqrtx
// 				input_deriv[i] += strength * x/((c*c) * ((x/c)*(x/c) + 1.0).sqrt());
// 			}
// 		} else if a == 2.0 {
// 			for i in 0..n{
// 				let x = input_values[i]-target_values[i];
// 				*error += strength * ((x/c)*(x/c))/a;
// 				input_deriv[i] += strength * x/(c*c);
// 			}
// 		} else {
// 			let za = 1.0f32.max(2.0-a);
// 			for i in 0..n{
// 				let x = input_values[i]-target_values[i];
// 				*error += strength * za / a *(((x/c)*(x/c)/za + 1.0).powf(0.5 * a) - 1.0);
// 				input_deriv[i] += strength * x/(c*c) * ((x/c)*(x/c)/za + 1.0).powf(0.5*a - 1.0);
// 			}
// 		}
// 	}
// }








