use graph::{GraphDef, GraphShapes, Result};
use id::{NodeID, OpID, PassID};
use ops::{standard_op_name, Op, OpInstance};
use shape::{NodeDim, NodeShape};
use ndarray::Dimension;
use std::sync::Arc;
use std::fmt;

#[derive(Clone)]
enum Rules {
	Individual(Vec<Option<Arc<Fn(usize) -> NodeDim + Sync + Send>>>),
	Joint(Arc<Fn(&[usize]) -> NodeShape + Sync + Send>),
}

impl fmt::Debug for Rules {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "Rules(...)")
	}
}

/// An Op with no passes or computation. Used purely to allow for customisable shape inference.
#[must_use]
#[derive(Clone, Debug)]
pub struct ShapeConstraint {
 	name: Option<String>,
	rules: Rules,
 	input_id: NodeID,
	output_id: NodeID,
}
 	
impl ShapeConstraint {
	pub fn new(input_id: &NodeID, output_id: &NodeID) -> Self {
		ShapeConstraint{
			name: None,
			rules: Rules::Individual(vec![]),
			input_id: input_id.clone(),
			output_id: output_id.clone(),
		}
	}

	/// For a single axis, apply a rule which takes the input dimension and produces a constraint on the corresponding output dimension.
	///
	/// This method will overwrite any previous `joint()` rules, or `single()` rules which apply to this axis.
	/// Any axis without a supplied rule with simply be unconstrained.
	pub fn single<D: Into<NodeDim>, F: 'static + Fn(usize) -> D + Sync + Send>(mut self, axis: usize, rule: F) -> Self {
		self.rules = match self.rules {
			Rules::Individual(mut vec) => {
				if axis + 1 > vec.len() {
					vec.resize(axis + 1, None);
				}
				vec[axis] = Some(Arc::new(move |dim| rule(dim).into()));
				Rules::Individual(vec)
			},
			Rules::Joint(_) => {
				let mut vec: Vec<Option<Arc<Fn(usize) -> NodeDim + Sync + Send>>> = vec![None; axis + 1];
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
}

impl Op for ShapeConstraint {
	type InstanceType = ShapeConstraintInstance;

	fn type_name(&self) -> &'static str {
		"ShapeConstraint"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef) -> Result<Self::InstanceType> {
		let name = standard_op_name(&self, &self.name, graph, &[self.input_id.clone()], &[self.output_id.clone()]);

		Ok(ShapeConstraintInstance{
			name: name,
			rules: self.rules,
			input_id: self.input_id.clone(),
			output_id: self.output_id.clone(),
		})
	}
}

#[derive(Debug, Clone)]
pub struct ShapeConstraintInstance {
	name: String,
	rules: Rules,
	input_id: NodeID,
	output_id: NodeID,
}

impl OpInstance for ShapeConstraintInstance {
	fn name(&self) -> &str {&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		(
			vec![self.input_id.clone()],
			vec![self.output_id.clone()]
		)
	}

	fn inner_passes(&self) -> Vec<PassID> {vec![]}

	fn inner_ops(&self) -> Vec<OpID> {vec![]}

	fn inner_nodes(&self) -> Vec<NodeID> {vec![]}

	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{

		let input_shape = shapes.get_shape(&self.input_id).to_data_shape()?;
		let output_rank = shapes.get_shape(&self.input_id).ndims();

		let output_shape = match &self.rules {
			&Rules::Individual(ref rules) => {
				let mut vec = vec![NodeDim::Unknown; output_rank];
				for (i, rule) in rules.iter().enumerate() {
					if let &Some(ref rule) = rule {
						vec[i] = rule(input_shape.slice()[i]);
					} else {
						vec[i] = NodeDim::Unknown;
					}
				}
				vec.into()
			},
			&Rules::Joint(ref rule) => {
				rule(input_shape.slice())
			}
		};
		shapes.merge_with(&self.output_id, &output_shape)?;
		Ok(())
	}
}