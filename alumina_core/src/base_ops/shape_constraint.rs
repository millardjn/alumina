use crate::{
	base_ops::{OpInstance, OpSpecification},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{Graph, Node, NodeID, Op},
	shape::{NodeAxis, NodeShape},
	shape_prop::ShapePropContext,
};
use indexmap::{indexset, IndexMap, IndexSet};
use ndarray::Dimension;
use std::{
	any::Any,
	convert::Into,
	fmt::{Debug, Formatter},
	sync::Arc,
};

/// Applies a ShapeConstraint  which enforces propagation of the runtime input shape to the output shape at runtime.
pub fn same_shape<I, O>(input: I, output: O) -> Result<Op, OpBuildError>
where
	I: Into<Node>,
	O: Into<Node>,
{
	let input = input.into();
	let output = output.into();
	ShapeConstraint::new(input, output).joint(|x| x.into()).build()
}

#[derive(Clone)]
enum Rules {
	#[allow(clippy::type_complexity)]
	Individual(Vec<Option<Arc<dyn Fn(usize) -> NodeAxis + Sync + Send>>>),
	Joint(Arc<dyn Fn(&[usize]) -> NodeShape + Sync + Send>),
}

impl Debug for Rules {
	fn fmt(&self, f: &mut Formatter<'_>) -> ::std::fmt::Result {
		match self {
			Rules::Individual(_) => write!(f, "Rules::Individual(..)"),
			Rules::Joint(_) => write!(f, "Rules::Joint(..)"),
		}
	}
}

/// An `Op` with no computation. Used purely to allow for customisable shape inference.
#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct ShapeConstraint {
	rules: Rules,
	input: Node,
	output: Node,
}

impl ShapeConstraint {
	pub fn new<I, O>(input: I, output: O) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		let input = input.into();
		let output = output.into();
		ShapeConstraint {
			rules: Rules::Individual(vec![]),
			input,
			output,
		}
	}

	/// For a single axis, apply a constraint that the input dimension is the same as the output dimension.
	///
	/// This method will overwrite any previous `joint()` rules, or `single()` rules which apply to this axis.
	/// Any axis without a supplied rule with simply be unconstrained.
	pub fn same(
		self,
		axis: usize,
	) -> Self {
		self.single(axis, std::convert::identity)
	}

	/// For a single axis, apply a rule which takes the input dimension and produces a constraint on the corresponding
	/// output dimension.
	///
	/// This method will overwrite any previous `joint()` rules, or `single()` rules which apply to this axis.
	/// Any axis without a supplied rule with simply be unconstrained.
	pub fn single<D: Into<NodeAxis>, F: 'static + Fn(usize) -> D + Sync + Send>(
		mut self,
		axis: usize,
		rule: F,
	) -> Self {
		self.rules = match self.rules {
			Rules::Individual(mut vec) => {
				if axis + 1 > vec.len() {
					vec.resize(axis + 1, None);
				}
				vec[axis] = Some(Arc::new(move |dim| rule(dim).into()));
				Rules::Individual(vec)
			}
			Rules::Joint(_) => {
				#[allow(clippy::type_complexity)]
				let mut vec: Vec<Option<Arc<dyn Fn(usize) -> NodeAxis + Sync + Send>>> = vec![None; axis + 1];
				vec[axis] = Some(Arc::new(move |dim| rule(dim).into()));
				Rules::Individual(vec)
			}
		};
		self
	}

	/// Apply a rule for mapping from a known input shape to a constraint on the output shape.
	///
	/// This method will overwrite any previous rules for generating constraints.
	pub fn joint<F: 'static + Fn(&[usize]) -> NodeShape + Sync + Send>(mut self, rule: F) -> Self {
		self.rules = Rules::Joint(Arc::new(rule));
		self
	}

	/// Special case joint constraint where every axis is multiplied by a shared factor.
	pub fn multiple(self, factor: usize) -> Self {
		self.joint(move |shape| shape.iter().map(|x| x * factor).into())
	}
}

impl OpSpecification for ShapeConstraint {
	type InstanceType = ShapeConstraintInstance;

	fn type_name(&self) -> &'static str {
		"ShapeConstraint"
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input.clone()]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		ShapeConstraint {
			rules: self.rules.clone(),
			input: mapping.get(&self.input).unwrap_or(&self.input).clone(),
			output: mapping.get(&self.output).unwrap_or(&self.output).clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(ShapeConstraintInstance {
			rules: self.rules,
			input: self.input.id(),
			output: self.output.id(),
		})
	}
}

#[derive(Clone, Debug)]
pub struct ShapeConstraintInstance {
	input: NodeID,
	output: NodeID,
	rules: Rules,
}

impl OpInstance for ShapeConstraintInstance {
	fn type_name(&self) -> &'static str {
		"ShapeConstraint"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(ShapeConstraint {
			rules: self.rules.clone(),
			input: graph.node_from_id(self.input),
			output: graph.node_from_id(self.output),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output]
	}

	fn gradient(&self, _context: &mut GradientContext) -> Result<(), GradientError> {
		Ok(())
	}

	fn propagate_shapes(&self, context: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let new_output_shape = {
			let input_shape = context.input_shape(&self.input);
			let output_rank = context.output_shape(&self.output).len();
			match &self.rules {
				Rules::Individual(ref rules) => {
					let mut vec = vec![NodeAxis::unknown(); output_rank];
					for (i, rule) in rules.iter().enumerate() {
						if let Some(ref rule) = rule {
							vec[i] = rule(input_shape.slice()[i]);
						} else {
							vec[i] = NodeAxis::unknown();
						}
					}
					vec.into()
				}
				Rules::Joint(ref rule) => rule(input_shape.slice()),
			}
		};

		context.merge_output_shape(&self.output, &new_output_shape)
	}

	fn execute(&self, _context: &ExecutionContext) -> Result<(), ExecutionError> {
		Ok(())
	}
}
