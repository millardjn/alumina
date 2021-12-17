use crate::graph::{Node, Op};
use crate::shape::{NodeAxis, NodeShape};
use crate::util::display::{Iter2Display, IterDisplay};
use failure::{Context, Error, Fail};

use indexmap::{IndexMap, IndexSet};

#[derive(Debug, Fail)]
#[fail(display = "OpBuildError cause: {}", cause)]
pub struct OpBuildError {
	#[cause]
	cause: Context<String>,
}

impl<I: Into<String>> From<I> for OpBuildError {
	fn from(context: I) -> Self {
		Self {
			cause: Context::from(context.into()),
		}
	}
}

#[derive(Debug, Fail)]
#[fail(display = "GradientError cause: {}", cause)]
pub enum GradientError {
	#[fail(display = "The gradient for this operation is not yet implemented")]
	Unimplemented,

	#[fail(display = "There was an error building an Op required from the gradient: {}", error)]
	OpBuild { error: OpBuildError },

	#[fail(display = "There was an error constructing the gradient: {}.", desc)]
	Other { desc: String },
}

impl From<OpBuildError> for GradientError {
	fn from(error: OpBuildError) -> GradientError {
		GradientError::OpBuild { error }
	}
}

impl<I: Into<String>> From<I> for GradientError {
	fn from(desc: I) -> GradientError {
		GradientError::Other { desc: desc.into() }
	}
}

#[derive(Debug, Fail)]
#[fail(display = "CloneError cause: {}", cause)]
pub struct CloneError {
	#[cause]
	cause: Context<Error>,
}

#[derive(Debug, Fail)]
#[fail(display = "ExecutionError cause: {}", cause)]
pub struct ExecutionError {
	#[cause]
	cause: Context<String>,
}

impl From<String> for ExecutionError {
	fn from(context: String) -> Self {
		Self {
			cause: Context::from(context),
		}
	}
}

impl From<Context<String>> for ExecutionError {
	fn from(context: Context<String>) -> Self {
		Self { cause: context }
	}
}

#[derive(Debug, Fail)]
#[fail(display = "ShapePropError cause: {}", cause)]
pub struct ShapePropError {
	#[cause]
	cause: Context<String>,
}

impl From<String> for ShapePropError {
	fn from(context: String) -> ShapePropError {
		ShapePropError {
			cause: Context::from(context),
		}
	}
}

impl From<Context<String>> for ShapePropError {
	fn from(context: Context<String>) -> ShapePropError {
		ShapePropError { cause: context }
	}
}

impl From<ShapeError> for ShapePropError {
	fn from(err: ShapeError) -> ShapePropError {
		ShapePropError {
			cause: err.context("Shape error".to_string()),
		}
	}
}

/// Returned from `Grad::build(..)` when one or more `Op`s produced an error when constructing the gradient call.
#[derive(Debug, Fail)]
#[fail(display = "The following ops errored when producing their gradient {}.", errors)]
pub struct GradError {
	pub(crate) errors: Iter2Display<Op, GradientError, IndexMap<Op, GradientError>>,
	pub(crate) partial: IndexMap<Node, Node>,
}

impl GradError {
	pub fn into_partial(self) -> IndexMap<Node, Node> {
		self.partial
	}
}

// impl ::std::fmt::Display for GradError {
// 	fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
// 		write!(f, "The following ops errored when producing their gradient: [\n")?;
// 		let mut iter = self.errors.iter();
// 		if let Some((ref op, ref error)) = iter.next() {
// 			write!(f, "{}: {}", op, error)?;
// 			for (ref op, ref error) in iter {
// 				write!(f, "{} {}\n", op, error)?;
// 			}
// 		}
// 		write!(f, "]")?;
// 		Ok(())
// 	}
// }

/// A `Fail` type returned from `shapes(..)` when one or more inputs can't be merged with their respective `Node`s
/// shapes, or, one or more `Op`s produced an error when propagating shapes.
#[derive(Fail)]
pub enum ShapesError {
	///
	#[fail(
		display = "The following nodes could not merge with the shapes of the supplied inputs:\n{}.",
		input_errors
	)]
	InputCantMerge {
		input_errors: Iter2Display<Node, ShapeError, IndexMap<Node, ShapeError>>,
		partial: IndexMap<Node, NodeShape>,
	},

	///
	#[fail(
		display = "The Op '{}' errored when propagating shapes:\n{}.\n resolved shapes were: {:#?}",
		op, error, partial
	)]
	ShapePropError {
		op: Op,
		error: ShapePropError,
		partial: IndexMap<Node, NodeShape>,
	},

	/// Returned when the `SubGraph` contains an `Op` which has inputs outside the `SubGraph`.
	#[fail(
		display = "An input node ({}) of op ({}) is missing from the SubGraph",
		op, input_node
	)]
	OpInputNotInSubgraph { op: Op, input_node: Node },

	/// Returned due to problems with order of `Op`s in the `SubGraph`.
	///
	/// When a node is used as an input to an Op, and then as an output, this indicates that the subgraph order is not
	/// topologically sorted (executable).
	#[fail(
		display = "The node ({}) is used as an output after being used as an input, indicating the Subgraph is not topologically sorted.",
		node
	)]
	SubGraphNotExecutable { node: Node },
}

impl ::std::fmt::Debug for ShapesError {
	fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
		write!(f, "{}", self)
	}
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Fail)]
pub enum ShapeError {
	#[fail(
		display = "Cannot produce a final array shape, as shape contains to dimensions that aren't known: {}.",
		shape
	)]
	UnderDeterminedDimensions { shape: NodeShape },

	#[fail(
		display = "Cannot merge shapes as they have a different number of dimensions ({} vs {}).",
		shape1, shape2
	)]
	IncompatibleRanks { shape1: NodeShape, shape2: NodeShape },

	#[fail(
		display = "Cannot merge shapes due to incompatible dimensions at index {} in shapes ({} vs {}).",
		index, shape1, shape2
	)]
	IncompatibleDimensionAt {
		shape1: NodeShape,
		shape2: NodeShape,
		index: usize,
		#[cause]
		cause: DimError,
	},

	#[fail(
		display = "Cannot merge shapes due to incompatible flat size (total number of elements). Shapes: ({} vs {}).",
		shape1, shape2
	)]
	IncompatibleFlatSize { shape1: NodeShape, shape2: NodeShape },
}

#[derive(Debug, Fail)]
pub enum DimError {
	#[fail(display = "Cannot merge incompatible dimensions ({} vs {}).", dim1, dim2)]
	IncompatibleDimension { dim1: NodeAxis, dim2: NodeAxis },
}

/// Fail type returned when executiong a graph with `exec()`.
#[derive(Debug, Fail)]
pub enum ExecError {
	/// Returned when the execution of a particular `Op` returns an error.
	#[fail(display = "ExecError::Op Executing Op {} returned error: {}", op, error)]
	Op { error: ExecutionError, op: Op },

	/// Returned when the `SubGraph` contains an `Op` which has inputs outside the `SubGraph`.
	#[fail(
		display = "ExecError::OpInputNotInSubgraph An input node ({}) of op ({}) is missing from the SubGraph",
		op, node
	)]
	OpInputNotInSubgraph { op: Op, node: Node },

	/// This is returned when execution subgraphs are extracted with insufficient inputs to calculate the required
	/// outputs.
	#[fail(
		display = "ExecError::InsufficientInputs The execution Subgraph contained a node that is not an input and is not written to by any Ops: node {} was read by {} ",
		node, op
	)]
	InsufficientInputs { node: Node, op: Op },

	/// Returned from `exec()` if errors arise when extracting the default execution `SubGraph`.
	#[fail(
		display = "ExecError::Subgraph Error when extracting the default execution SubGraph: {}",
		error
	)]
	Subgraph { error: ExecutionSubgraphError },

	/// Returned when an output has no parent `Op`s in the `SubGraph` and is not an input.
	#[fail(
		display = "ExecError::OutputNotComputable Output node ({}) cannot be calculated (Not an input and has no parent ops).",
		node
	)]
	OutputNotComputable { node: Node },

	///
	#[fail(display = "ExecError::Shape Calculating node shapes returned error:\n{}", error)]
	Shape { error: ShapesError },

	/// Returned when one of the outputs is not in the `SubGraph`.
	#[fail(
		display = "ExecError::OutputsNotInSubgraph The output node ({}) is not in the execution subgraph",
		node
	)]
	OutputsNotInSubgraph { node: Node },

	/// Returned due to problems with order of `Op`s in the `SubGraph`.
	///
	/// When a node is used as an input to an Op, and then as an output, this indicates that the subgraph order is not
	/// topologically sorted (executable), and if returned from `exec_sub()` that it could be cyclic.
	#[fail(
		display = "ExecError::SubGraphNotExecutable The node ({}) is used as an output after being used as an input, indicating the Subgraph is not topologically sorted.",
		node
	)]
	SubGraphNotExecutable { node: Node },
}

/// Fail type returned when extraction of an execution subgraph
#[derive(Debug, Fail)]
pub enum ExecutionSubgraphError {
	/// This is returned when execution subgraphs are extracted with insufficient inputs to calculate the required
	/// outputs.
	#[fail(
		display = "ExecutionSubgraphError::InsufficientInputs Subgraph extracted contained nodes that cannot be calculated (Not an input and no parents):\n{}",
		parentless_nodes
	)]
	InsufficientInputs {
		parentless_nodes: IterDisplay<Node, Vec<Node>>,
		subgraph_nodes: IterDisplay<Node, IndexSet<Node>>,
		subgraph_ops: IterDisplay<Op, IndexSet<Op>>,
	},

	/// Returned when the execution subgraph cannot be ordered topologically due to a cycle.
	#[fail(
		display = "ExecutionSubgraphError::CyclicGraphError Could not order the execution subgraph due to a cycle:\n{}",
		error
	)]
	Cycle {
		error: CyclicGraphError,
		subgraph_nodes: IterDisplay<Node, IndexSet<Node>>,
		subgraph_ops: IterDisplay<Op, IndexSet<Op>>,
	},
}

/// `Fail` type returned when topologically sorting a `SubGraph`, for execution purposes, and the process cannot
/// continue due to cyclic dependencies.
#[derive(Debug, Fail)]
#[fail(
	display = "Subgraph extracted contained ops or nodes that cannot be put into topological order. Check for cycles.:\nnodes: {}\nops: {}\n",
	unsorted_nodes, unsorted_ops
)]
pub struct CyclicGraphError {
	pub unsorted_nodes: IterDisplay<Node, Vec<Node>>,
	pub unsorted_ops: IterDisplay<Op, Vec<Op>>,
}
