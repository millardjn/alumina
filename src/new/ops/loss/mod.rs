pub mod proportional;
pub mod mse;
pub mod mae;
pub mod l2;
pub mod cross_entropy;
pub mod prediction;


use new::graph::{NodeID, OpID, PassID};

#[derive(Clone, Debug)] 
enum LossType {
	Joint { // No output node, losses are applied to the graph
		pass_id: PassID
	},
	Output {
		output_id: NodeID,
		forward_id: PassID,
		backward_id: PassID
	},
	Separate {
		output_id: NodeID,
		loss_id: OpID,
		forward_id: PassID,
		backward_id: PassID
	},
}