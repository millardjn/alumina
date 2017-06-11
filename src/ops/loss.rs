use ops::math;
use graph::*;
use std::cell::RefCell;
use std::num::FpCategory;
use shape::*;
use ops::Operation;
use std::f32;
use std::num;
//evaluation operations. no output

///`MseLoss` - imposes an error on the input node values proportional to the average distance squared of each element from the target node values.
#[derive(Clone)] 
pub struct MseLoss {
	name: String,
	input_id: NodeID,
	target_id: NodeID,
	strength: f32,
}

impl MseLoss {
	pub fn new(input_id: &NodeID, target_id: &NodeID, strength: f32, name: &str) -> Box<MseLoss>{
		Box::new(MseLoss{
			name: name.to_string(),
			input_id: input_id.clone(),
			target_id: target_id.clone(),
			strength: strength,
		})
	}
	
	pub fn new_default(input_id: &NodeID, target_id: &NodeID,) -> Box<MseLoss>{
		MseLoss::new(input_id, target_id, 1.0, "MseLoss")
	}
}

impl Operation for MseLoss {

	fn name(&self) -> &str{&self.name}
	
	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_id.ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Input node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_id.ind].name, self.name));
	}
	
	fn input_node_IDs(&self) -> Vec<NodeID>{vec![self.input_id.clone(), self.target_id.clone()]}
	
	fn output_node_IDs(&self) -> Vec<NodeID>{vec![]}
	
	fn num_params(&self) -> usize {0}
	
	fn forward (&mut self, _data: &mut [RefCell<NodeData>], _params: &[f32]){}// No Output
	
	fn backward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32], _param_deriv: &mut [f32], error: &mut f32){
		let input = &mut *{data[self.input_id.ind].borrow_mut()};
		let target = &*{data[self.target_id.ind].borrow()};
		let input_size = input.shape.flat_size_single();
		let target_size = target.shape.flat_size_single();
		
				
		assert_eq!(input_size, target_size, "Error: Operation '{}' input and target node sizes were not equal during evaluation", self.name);
		assert_eq!(input.shape.n, target.shape.n, "Error: Operation '{}' input and target node 'n' were not equal during evaluation", self.name);


		let input_deriv: &mut [f32] = &mut input.derivatives;
		let input_values: &[f32] = &input.values;
		let target_values: &[f32] =  &target.values;

		math::ssd_error(input_values, target_values, self.strength/input_size as f32, input_deriv, error);	
		
	}	
}

///`MaeLoss` - imposes an error on the input node values proportional to the distance of each element from the target node values.
#[derive(Clone)] 
pub struct MaeLoss {
	name: String,
	input_id: NodeID,
	target_id: NodeID,
	strength: f32,
}

impl MaeLoss {
	pub fn new(input_id: &NodeID, target_id: &NodeID, strength: f32, name: &str) -> Box<MaeLoss>{
		Box::new(MaeLoss{
			name: name.to_string(),
			input_id: input_id.clone(),
			target_id: target_id.clone(),
			strength: strength,
		})
	}
	
	pub fn new_default(input_id: &NodeID, target_id: &NodeID,) -> Box<MaeLoss>{
		MaeLoss::new(input_id, target_id, 1.0, "MaeLoss")
	}
}

impl Operation for MaeLoss {

	fn name(&self) -> &str{&self.name}
	
	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_id.ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Input node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_id.ind].name, self.name));
	}
	
	fn input_node_IDs(&self) -> Vec<NodeID>{vec![self.input_id.clone(), self.target_id.clone()]}
	
	fn output_node_IDs(&self) -> Vec<NodeID>{vec![]}
	
	fn num_params(&self) -> usize {0}
	
	fn forward (&mut self, _data: &mut [RefCell<NodeData>], _params: &[f32]){}// No Output
	
	fn backward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32], _param_deriv: &mut [f32], error: &mut f32){
		let input = &mut *{data[self.input_id.ind].borrow_mut()};
		let target = &*{data[self.target_id.ind].borrow()};
		let input_size = input.shape.flat_size_single();
		let target_size = target.shape.flat_size_single();
		
				
		assert_eq!(input_size, target_size, "Error: Operation '{}' input and target node sizes were not equal during evaluation", self.name);
		assert_eq!(input.shape.n, target.shape.n, "Error: Operation '{}' input and target node 'n' were not equal during evaluation", self.name);

		let n = input.shape.flat_size_all();
		let input_deriv: &mut [f32] = &mut input.derivatives[..n];
		let input_values: &[f32] = &input.values[..n];
		let target_values: &[f32] =  &target.values[..n];
		
		let scale = self.strength/input_size as f32;
		for i in 0..n{
			*error += (input_values[i]-target_values[i]).abs()*scale;
			input_deriv[i] += (input_values[i]-target_values[i]).signum()*scale;
		}

	}
}


///`GeneralLoss` - A general implementation of a range of robust loss functions.
/// A More General Robust Loss Function https://arxiv.org/pdf/1701.03077.pdf Eq.13 & Eq.14
/// when power == 2, this is the L2 loss
/// when power == 1, this is the charbonnier loss (smooth L1 loss)
/// when power == 0, this is the Cauchy/Lorentzian loss
/// the scale is the range of values either size of zero for which the loss will closely approximate the L2 loss
/// a small scale value means small errors get treated as large errors.
/// see paper for futher losses
#[derive(Clone)] 
pub struct GeneralLoss {
	name: String,
	input_id: NodeID,
	target_id: NodeID,
	scale: f32,
	power: f32,
	strength: f32,
}

impl GeneralLoss {
	pub fn new(input_id: &NodeID, target_id: &NodeID, strength: f32, scale: f32, power: f32, name: &str) -> Box<GeneralLoss>{
		Box::new(GeneralLoss{
			name: name.to_string(),
			input_id: input_id.clone(),
			target_id: target_id.clone(),
			scale: scale,
			power: power,
			strength: strength,
		})
	}
	
	pub fn new_default(input_id: &NodeID, target_id: &NodeID,) -> Box<GeneralLoss>{
		GeneralLoss::new(input_id, target_id, 1.0, 1.0, 1.0, "GeneralLoss")
	}
}

impl Operation for GeneralLoss {

	fn name(&self) -> &str{&self.name}
	
	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_id.ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Input node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_id.ind].name, self.name));
	}
	
	fn input_node_IDs(&self) -> Vec<NodeID>{vec![self.input_id.clone(), self.target_id.clone()]}
	
	fn output_node_IDs(&self) -> Vec<NodeID>{vec![]}
	
	fn num_params(&self) -> usize {0}
	
	fn forward (&mut self, _data: &mut [RefCell<NodeData>], _params: &[f32]){}// No Output
	
	fn backward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32], _param_deriv: &mut [f32], error: &mut f32){
		let input = &mut *{data[self.input_id.ind].borrow_mut()};
		let target = &*{data[self.target_id.ind].borrow()};
		let input_size = input.shape.flat_size_single();
		let target_size = target.shape.flat_size_single();
		
				
		assert_eq!(input_size, target_size, "Error: Operation '{}' input and target node sizes were not equal during evaluation", self.name);
		assert_eq!(input.shape.n, target.shape.n, "Error: Operation '{}' input and target node 'n' were not equal during evaluation", self.name);

		let n = input.shape.flat_size_all();
		let input_deriv: &mut [f32] = &mut input.derivatives[..n];
		let input_values: &[f32] = &input.values[..n];
		let target_values: &[f32] =  &target.values[..n];
		
		let strength = self.strength/input_size as f32;
		let c = self.scale; // use notation from paper
		let a = self.power;
		if a.classify() == num::FpCategory::Zero {
			for i in 0..n{
				let x = input_values[i]-target_values[i];
				*error += strength * (0.5*(x/c)*(x/c)).ln_1p();
				input_deriv[i] += strength * 2.0 * x / (x*x + 2.0*c*c);
			}
		} else if a == f32::NEG_INFINITY {
			for i in 0..n{
				let x = input_values[i]-target_values[i];
				*error += -strength * (-0.5*(x/c)*(x/c)).exp_m1();
				input_deriv[i] += strength * x/(c*c) * (-0.5*(x/c)*(x/c)).exp();
			}
		} else if a == 1.0 {
			for i in 0..n{
				let x = input_values[i]-target_values[i];
				*error += strength *(((x/c)*(x/c) + 1.0).sqrt() - 1.0);
				input_deriv[i] += strength * x/((c*c) * ((x/c)*(x/c) + 1.0).sqrt());
			}
		} else if a == 2.0 {
			for i in 0..n{
				let x = input_values[i]-target_values[i];
				*error += strength * (((x/c)*(x/c) + 1.0) - 1.0)/a;
				input_deriv[i] += strength * x/(c*c);
			}
		} else {
			let za = 1.0f32.max(2.0-a);
			for i in 0..n{
				let x = input_values[i]-target_values[i];
				*error += strength * za / a *(((x/c)*(x/c)/za + 1.0).powf(0.5 * a) - 1.0);
				input_deriv[i] += strength * x/(c*c) * ((x/c)*(x/c)/za + 1.0).powf(0.5*a - 1.0);
			}
		}
	}
}


///`CrossEntLoss` - imposes an error on the input node values proportional to the distance of each element from the target node values.
#[derive(Clone)] 
pub struct CrossEntLoss {
	name: String,
	input_id: NodeID,
	target_id: NodeID,
	strength: f32,
}

impl CrossEntLoss {
	pub fn new(input_id: &NodeID, target_id: &NodeID, strength: f32, name: &str) -> Box<CrossEntLoss>{
		Box::new(CrossEntLoss{
			name: name.to_string(),
			input_id: input_id.clone(),
			target_id: target_id.clone(),
			strength: strength,
		})
	}
	
	pub fn new_default(input_id: &NodeID, target_id: &NodeID,) -> Box<CrossEntLoss>{
		CrossEntLoss::new(input_id, target_id, 1.0, "CrossEntError")
	}
}

impl Operation for CrossEntLoss {

	fn name(&self) -> &str{&self.name}
	
	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_id.ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Input node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_id.ind].name, self.name));
	}
	
	fn input_node_IDs(&self) -> Vec<NodeID>{vec![self.input_id.clone(), self.target_id.clone()]}
	
	fn output_node_IDs(&self) -> Vec<NodeID>{vec![]}
	
	fn num_params(&self) -> usize {0}
	
	fn forward (&mut self, _data: &mut [RefCell<NodeData>], _params: &[f32]){}// No Output
	
	fn backward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32], _param_deriv: &mut [f32], error: &mut f32){
		let input_size = data[self.input_id.ind].borrow().shape.flat_size_all();
		let target_size = data[self.target_id.ind].borrow().shape.flat_size_all();
		assert!(input_size == target_size, format!("Error: Operation '{}' input and target node sizes were not equal during evaluation", self.name));
		
		let input_node = &mut *{data[self.input_id.ind].borrow_mut()};
		let input_deriv: &mut [f32] = &mut input_node.derivatives;
		let input_values: &[f32] = &input_node.values;
		let target_values: &[f32] =  &data[self.target_id.ind].borrow().values;

		for (i, &t) in target_values.iter().enumerate() {
			
			let x = input_values[i];
			assert!(t >= 0. && t <= 1., "Cross Entropy inputs and targets must be between 0 and 1");
			assert!(x >= 0. && x <= 1., "Cross Entropy inputs and targets must be between 0 and 1");
			*error += -t*x.ln() * self.strength;
			input_deriv[i] += -t/x * self.strength;
		}

		
	}	
}

/// `SoftMaxCrossEntLoss`     - NSISOF - 
#[derive(Clone)] 
pub struct SoftMaxCrossEntLoss {
	name: String,
	input_id: NodeID,
	target_id: NodeID,
	strength: f32,
}

impl SoftMaxCrossEntLoss {
	pub fn new(input_id: &NodeID, target_id: &NodeID, strength: f32, name: &str) -> Box<SoftMaxCrossEntLoss>{
		Box::new(SoftMaxCrossEntLoss{
			name: name.to_string(),
			input_id: input_id.clone(),
			target_id: target_id.clone(),
			strength: strength,
		})
	}

	pub fn new_default(input_id: &NodeID, target_id: &NodeID) -> Box<SoftMaxCrossEntLoss>{
		SoftMaxCrossEntLoss::new(input_id, target_id, 1.0, "SoftMaxCrossEntLoss")
	}
	
}

impl Operation for SoftMaxCrossEntLoss {
	
	fn name(&self) -> &str{ &self.name }

	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_id.ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Input node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_id.ind].name, self.name));
	}
	
	fn num_params(&self) -> usize{0}
	
	fn input_node_IDs(&self) -> Vec<NodeID>{ vec![self.input_id.clone(), self.target_id.clone()] }
	
	fn output_node_IDs(&self) -> Vec<NodeID>{ vec![] }
		
	fn forward (&mut self, _data: &mut [RefCell<NodeData>], _params: &[f32]){}
	
	fn backward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32], _param_deriv: &mut [f32], error: &mut f32){
		let input = &mut *{data[self.input_id.ind].borrow_mut()};
		let target = &*{data[self.target_id.ind].borrow_mut()};
		assert!(input.shape.flat_size_all() == target.shape.flat_size_all(),
			format!("Error: Operation '{}' input and target node sizes were not equal during evaluation", self.name));
		
		let size = input.shape.flat_size_single();
		for n_ind in 0..input.shape.n{

			let inp_n = &input.values[n_ind*size..][..size];
			let tar_n = &target.values[n_ind*size..][..size];
			let inpd_n = &mut input.derivatives[n_ind*size..][..size];
						
			let max = inp_n.iter().fold(inp_n[0], |max, &v| v.max(max));
			let exp_sum = inp_n.iter().fold(0., |sum, &v| sum + (v-max).exp());
								
			for (i, &t) in tar_n.iter().enumerate() {
				
				let mult = t * self.strength;
				if mult != 0.0 {
					
					let a = inp_n[i] - max;
					
					for j in 0..i {
						inpd_n[j] += (inp_n[j]-max).exp()*(mult/exp_sum);
					}

					inpd_n[i] += (a.exp() - exp_sum)*(mult/exp_sum);
					//inpd_n[i] += - inp_n.iter().enumerate().fold(0., |sum, (ind, v)| sum + if ind != i {(v-max).exp()} else {0.0})*(mult/exp_sum);

					for j in i+1..size{
						inpd_n[j] += (inp_n[j]-max).exp()*(mult/exp_sum);
					}

					
					*error += mult * (exp_sum.ln() - a);
					
				}
				

			}
		}
		
		
	}
}

/// `SoftMaxDampedCrossEntLoss`     - NSISOF - 
#[derive(Clone)] 
pub struct SoftMaxDampedCrossEntLoss {
	name: String,
	input_id: NodeID,
	target_id: NodeID,
	strength: f32,
}

impl SoftMaxDampedCrossEntLoss {
	pub fn new(input_id: &NodeID, target_id: &NodeID, strength: f32, name: &str) -> Box<SoftMaxDampedCrossEntLoss>{
		Box::new(SoftMaxDampedCrossEntLoss{
			name: name.to_string(),
			input_id: input_id.clone(),
			target_id: target_id.clone(),
			strength: strength,
		})
	}

	pub fn new_default(input_id: &NodeID, target_id: &NodeID) -> Box<SoftMaxDampedCrossEntLoss>{
		SoftMaxDampedCrossEntLoss::new(input_id, target_id, 1.0, "SoftMaxDampedCrossEntLoss")
	}
	
}

impl Operation for SoftMaxDampedCrossEntLoss {
	
	fn name(&self) -> &str{ &self.name }

	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_id.ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Input node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_id.ind].name, self.name));
	}
	
	fn num_params(&self) -> usize{0}
	
	fn input_node_IDs(&self) -> Vec<NodeID>{ vec![self.input_id.clone(), self.target_id.clone()] }
	
	fn output_node_IDs(&self) -> Vec<NodeID>{ vec![] }
		
	fn forward (&mut self, _data: &mut [RefCell<NodeData>], _params: &[f32]){}
	
	fn backward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32], _param_deriv: &mut [f32], error: &mut f32){
		let input = &mut *{data[self.input_id.ind].borrow_mut()};
		let target = &*{data[self.target_id.ind].borrow_mut()};
		assert!(input.shape.flat_size_all() == target.shape.flat_size_all(),
			format!("Error: Operation '{}' input and target node sizes were not equal during evaluation", self.name));
		
		let size = input.shape.flat_size_single();
		for n_ind in 0..input.shape.n{

			let inp_n = &input.values[n_ind*size..][..size];
			let tar_n = &target.values[n_ind*size..][..size];
			let inpd_n = &mut input.derivatives[n_ind*size..][..size];
						
			let max = inp_n.iter().fold(inp_n[0], |max, &v| v.max(max));
			let exp_sum = inp_n.iter().fold(0., |sum, &v| sum + (v-max).exp());
								
			for (i, &t) in tar_n.iter().enumerate() {
				
				let mult = t * self.strength;
				if mult.classify() != FpCategory::Zero {
					
					let a = inp_n[i] - max;
					let a_exp = a.exp();
					let div = a_exp/exp_sum;
					let div_ln = div.ln();
					let exp_sum_sqr = exp_sum*exp_sum;
					let exp_sum_m_a = inp_n.iter().enumerate().fold(0., |sum, (ind, v)| sum + if ind != i {(v-max).exp()} else {0.0});
					//let exp_sum_m_a = exp_sum - a_exp;

					let mut deriv_sum = 0.0;

					for j in 0..i {
						let d = (inp_n[j]-max).exp() * (exp_sum_m_a- a_exp*div_ln) * (mult/exp_sum_sqr);
						inpd_n[j] += d;
						deriv_sum += d;

					}
					

					for j in i+1..size{
						let d = (inp_n[j]-max).exp() * (exp_sum_m_a- a_exp*div_ln) * (mult/exp_sum_sqr);
						inpd_n[j] += d;
						deriv_sum += d;
					}

					inpd_n[i] += -deriv_sum;
					
					*error += mult * (exp_sum.ln() - a) * (1.0 - div);
					
				}
				

			}
		}
		
		
	}
}

/// `PredictionLoss`     - NSISOF - This operation does not provide gradients but rather just contributes a 1 or 0 to error depending if the max value in input matches the max value in target
#[derive(Clone)] 
pub struct PredictionLoss {
	name: String,
	input_id: NodeID,
	target_id: NodeID,
	strength: f32,
}

impl PredictionLoss {
	pub fn new(input_id: &NodeID, target_id: &NodeID, strength: f32, name: &str) -> Box<PredictionLoss>{
		Box::new(PredictionLoss{
			name: name.to_string(),
			input_id: input_id.clone(),
			target_id: target_id.clone(),
			strength: strength,
		})
	}

	pub fn new_default(input_id: &NodeID, target_id: &NodeID) -> Box<PredictionLoss>{
		PredictionLoss::new(input_id, target_id, 1.0, "PredictionLoss")
	}
	
}

impl Operation for PredictionLoss {
	
	fn name(&self) -> &str{ &self.name }

	fn propagate_shape_constraints(&self, nodes: &[Node], shapes: &mut [NodeShape]){
		shapes[self.input_id.ind].collapse_ranges_to_minimum()
			.expect(&format!("Error: Input node '{}' could not be collapsed to a fixed shape prior to being used by Operation '{}'. Provide dimensions or stronger constraints.", nodes[self.input_id.ind].name, self.name));
	}
	
	fn num_params(&self) -> usize{0}
	
	fn input_node_IDs(&self) -> Vec<NodeID>{ vec![self.input_id.clone(), self.target_id.clone()] }
	
	fn output_node_IDs(&self) -> Vec<NodeID>{ vec![] }
		
	fn forward (&mut self, _data: &mut [RefCell<NodeData>], _params: &[f32]){}
	
	fn backward (&mut self, data: &mut [RefCell<NodeData>], _params: &[f32], _param_deriv: &mut [f32], error: &mut f32){
		let input = &mut *{data[self.input_id.ind].borrow_mut()};
		let target = &*{data[self.target_id.ind].borrow_mut()};
		assert!(input.shape.flat_size_all() == target.shape.flat_size_all(),
			format!("Error: Operation '{}' input and target node sizes were not equal during evaluation", self.name));
		
		let size = input.shape.flat_size_single();
		for n_ind in 0..input.shape.n{

			let inp_n = &input.values[n_ind*size..][..size];
			let tar_n = &target.values[n_ind*size..][..size];
			
			
			let (in_ind, _) = inp_n.iter().enumerate().fold((0, f32::NEG_INFINITY), |(max_ind, max), (i, &v)| if v >= max {(i, v)} else {(max_ind, max)});
			let (tar_ind, _) = tar_n.iter().enumerate().fold((0, f32::NEG_INFINITY), |(max_ind, max), (i, &v)| if v >= max {(i, v)} else {(max_ind, max)});
			if in_ind != tar_ind {
				*error += self.strength;
			}
		}
		
	}
}

/// Sum     - NSISOF - Exists mainly for testing. Error is a simple sum over all input elements


#[cfg(test)]
mod tests {
	use super::*;
	
	#[test]
	fn test_mse_backprop(){
		for _ in 1..100{		
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(10, "nodein"));
			let n3 = graph.add_training_input_node(Node::new_flat(10, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				MseLoss::new_default(&n1, &n3),
			];
			graph.add_operations(ops);
			graph.init_params();
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-3);
		}
	}

	#[test]
	fn test_mae_loss_backprop(){
		for _ in 1..100{		
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(1000, "nodein"));
			let n3 = graph.add_training_input_node(Node::new_flat(1000, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				MaeLoss::new_default(&n1, &n3),
			];
			graph.add_operations(ops);
			graph.init_params();
			
			use ops::math::*;
			test_numeric(graph, 1.0, 1e-2);
		}
	}

	#[test]
	fn test_general_loss_zero_backprop(){
		use rand;
		use rand::distributions::*;
		let power = 0.0;
		let mut rng = rand::thread_rng();
		let range = Range::new(0.1, 1.0);
		for _ in 1..100{		
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(100, "nodein"));
			let n3 = graph.add_training_input_node(Node::new_flat(100, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				GeneralLoss::new(&n1, &n3, range.ind_sample(&mut rng), range.ind_sample(&mut rng), power, "generalloss"),
			];
			graph.add_operations(ops);
			graph.init_params();
			
			use ops::math::*;
			test_numeric(graph, 1.0, 0.1);
		}
	}

	#[test]
	fn test_general_loss_one_backprop(){
		use rand;
		use rand::distributions::*;
		let power = 1.0;
		let mut rng = rand::thread_rng();
		let range = Range::new(0.1, 1.0);
		for _ in 1..100{		
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(100, "nodein"));
			let n3 = graph.add_training_input_node(Node::new_flat(100, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				GeneralLoss::new(&n1, &n3, range.ind_sample(&mut rng), range.ind_sample(&mut rng), power, "generalloss"),
			];
			graph.add_operations(ops);
			graph.init_params();
			
			use ops::math::*;
			test_numeric(graph, 1.0, 0.01);
		}
	}

	#[test]
	fn test_general_loss_two_backprop(){
		use rand;
		use rand::distributions::*;
		let power = 2.0;
		let mut rng = rand::thread_rng();
		let range = Range::new(0.1, 1.0);
		for _ in 1..100{		
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(100, "nodein"));
			let n3 = graph.add_training_input_node(Node::new_flat(100, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				GeneralLoss::new(&n1, &n3, range.ind_sample(&mut rng), range.ind_sample(&mut rng), power, "generalloss"),
			];
			graph.add_operations(ops);
			graph.init_params();
			
			use ops::math::*;
			test_numeric(graph, 1.0, 0.01);
		}
	}

	#[test]
	fn test_general_loss_neg_inf_backprop(){
		use rand;
		use rand::distributions::*;
		let power = f32::NEG_INFINITY;
		let mut rng = rand::thread_rng();
		let range = Range::new(0.1, 1.0);
		for _ in 1..100{		
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(100, "nodein"));
			let n3 = graph.add_training_input_node(Node::new_flat(100, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				GeneralLoss::new(&n1, &n3, range.ind_sample(&mut rng), range.ind_sample(&mut rng), power, "generalloss"),
			];
			graph.add_operations(ops);
			graph.init_params();
			
			use ops::math::*;
			test_numeric(graph, 1.0, 0.01);
		}
	}

	#[test]
	fn test_general_loss_rand_backprop(){
		use rand;
		use rand::distributions::*;
		let mut rng = rand::thread_rng();
		let range = Range::new(0.1, 1.0);
		let power_range = Range::new(-2.0, 2.0);
		for _ in 1..100{		
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(100, "nodein"));
			let n3 = graph.add_training_input_node(Node::new_flat(100, "nodetrain"));
			
			let ops: Vec<Box<Operation>> = vec![
				GeneralLoss::new(&n1, &n3, range.ind_sample(&mut rng), range.ind_sample(&mut rng), power_range.ind_sample(&mut rng), "generalloss"),
			];
			graph.add_operations(ops);
			graph.init_params();
			
			use ops::math::*;
			test_numeric(graph, 1.0, 0.1);
		}
	}

//	#[test]
//	fn test_cross_ent_backprop(){
//		for _ in 1..100{
//			let mut graph = Graph::new();
//		
//			let n1 = graph.add_input_node(Node::new_flat(10, "nodein"));
//			let n2 = graph.add_node(Node::new_flat(10, "softmax_nodein"));
//			let n3 = graph.add_training_input_node(Node::new_flat(10, "nodetrain"));
//			let n4 = graph.add_node(Node::new_flat(10, "softmax_nodetrain"));
//			
//			let ops: Vec<Box<Operation>> = vec![
//				SoftMax::new_default(&graph, n1, n2),
//				SoftMax::new_default(&graph, n3, n4),
//				CrossEntLoss::new_default(&graph, n2, n4),
//			];
//			graph.add_operations(ops);
//			graph.init_params();
//			
//			use ops::math::*;
//			test_numeric(graph, 0.01, 1e-3);
//		}
//	}
	
	#[test]
	fn test_soft_max_cross_ent_backprop(){
		for _ in 1..100{
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(30, "nodein"));
			let n2 = graph.add_training_input_node(Node::new_flat(30, "nodetrain"));
			
			
			let ops: Vec<Box<Operation>> = vec![
				SoftMaxCrossEntLoss::new_default(&n1, &n2),
			];
			graph.add_operations(ops);
			graph.init_params();
			
			use ops::math::*;
			test_numeric(graph, 0.05, 1e-0);
		}
	}

	#[test]
	fn test_soft_max_damp_cross_ent_backprop(){
		for _ in 1..100{
			let mut graph = Graph::new();
		
			let n1 = graph.add_input_node(Node::new_flat(30, "nodein"));
			let n2 = graph.add_training_input_node(Node::new_flat(30, "nodetrain"));
			
			
			let ops: Vec<Box<Operation>> = vec![
				SoftMaxDampedCrossEntLoss::new_default(&n1, &n2),
			];
			graph.add_operations(ops);
			graph.init_params();
			
			use ops::math::*;
			test_numeric(graph, 0.3, 1e-3);
		}
	}
}


// SAD-error

// Normalised cross-correlation