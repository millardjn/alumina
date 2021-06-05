use crate::elementwise::{div::div, sqr::sqr};
use crate::regularisation::{l1::l1, l2::l2};
use alumina_core::{errors::OpBuildError, graph::Node};

use smallvec::SmallVec;

/// Calculates an L0 approximating regularisation of the input nodes, returning a scalar node.
/// This regularisation is:
/// * differentiable (almost everywhere)
/// * L0-eqsue (same minima structure on each axis)
/// * scale invariant (unlike L1)
///
/// This is calculated as the ratio of the L1^2/L2 norms.
///
/// Based on DeepHoyer by Huanrui Yang, Wei Wen, Hai Li
pub fn hoyer_squared<I, T: IntoIterator<Item = I>>(inputs: T) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
{
	let inputs: SmallVec<[Node; 16]> = inputs.into_iter().map(Into::into).collect();

	let hoyer_squared = div(sqr(l1(&inputs)?)?, l2(&inputs)?)?;
	hoyer_squared.set_name_unique(&format!(
		"hoyer_squared({})",
		inputs.iter().map(|n| n.name()).collect::<Vec<_>>().join(",")
	));

	Ok(hoyer_squared)
}
