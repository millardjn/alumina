pub mod proportional;
pub mod mse;
pub mod mae;
pub mod cross_entropy;
pub mod prediction;
pub mod robust_loss;


use graph::{NodeID, PassID};

#[derive(Clone, Debug)] 
pub(crate) enum LossType {
	Joint { // No output node, losses are applied to the graph
		pass_id: PassID
	},
	Output {
		output_id: NodeID,
		forward_id: PassID,
		backward_id: PassID
	},
}