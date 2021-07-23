use crate::{
	base_ops::{OpInstance, OpSpecification},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{Graph, Node, NodeID},
	shape::NodeShape,
	shape_prop::ShapePropContext,
};
use indexmap::{indexset, IndexMap, IndexSet};
use ndarray::ArrayViewMutD;
use std::{any::Any, fmt, sync::Arc};


/// Wrap to implement debug and clone
#[derive(Clone)]
struct ApplyWrap {
    f: Arc<dyn Fn(ArrayViewMutD<f32>) + Send + Sync + 'static>
}

impl fmt::Debug for ApplyWrap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Apply Op Closure {{ .. }}")
    }
}


/// Produces and output node with the given shape, and an op to fill the array with the provided closure.
///
/// The output node has the same shape as the input.
pub fn apply<F: Fn(ArrayViewMutD<f32>) + Send + Sync + 'static, S: Into<NodeShape>>(f: F, shape: S) -> Result<Node, OpBuildError> {
	let output = Node::new(shape).set_name_unique("apply()");
	let _op = Apply::new(&output, f).build()?;


	Ok(output)
}

/// Fills the provided output with the provided closure, then returns the same output Node.
///
/// The output node has the same shape as the input.
pub fn apply_into<F: Fn(ArrayViewMutD<f32>) + Send + Sync + 'static, O: Into<Node>>(f: F, output: O) -> Result<Node, OpBuildError> {
	let output = output.into();
	let _op = Apply::new(output.clone(), f).build()?;
	Ok(output)
}

#[must_use = "Op builder not used, call .build()"]
#[derive(Debug)]
pub struct Apply {
	output: Node,
    f: ApplyWrap,
}

/// Note: If cloned, any state in the closure will be cloned as is including pseudo random generators.
impl Apply {
	pub fn new<O, F: Fn(ArrayViewMutD<f32>) + Send + Sync + 'static>(output: O, f: F) -> Self
	where
		O: Into<Node>,
	{
		let output = output.into();
		Apply { output, f: ApplyWrap{f: Arc::new(f)} }
	}

    pub fn new_boxed<O>(output: O, f: Arc<dyn Fn(ArrayViewMutD<f32>) + Send + Sync + 'static>) -> Self
	where
		O: Into<Node>,
	{
		let output = output.into();
		Apply { output, f: ApplyWrap{f} }
	}
}

impl OpSpecification for Apply {
	type InstanceType = ApplyInstance;

	fn type_name(&self) -> &'static str {
		"Apply"
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node> {
		indexset![]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Apply {
			f: self.f.clone(),
			output: mapping.get(&self.output).unwrap_or(&self.output).clone(),
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		Ok(ApplyInstance {
			output: self.output.id(),
			f: self.f,
		})
	}
}

/// Elementwise Op, the value of the input is added to
#[derive(Debug)]
pub struct ApplyInstance {
	output: NodeID,
	f: ApplyWrap,
}

impl OpInstance for ApplyInstance {
	fn type_name(&self) -> &'static str {
		"Apply"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(Apply {
			output: graph.node_from_id(self.output),
			f: self.f.clone(),
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output]
	}

	fn gradient(&self, _ctx: &mut GradientContext) -> Result<(), GradientError> {
		Ok(())
	}

	fn propagate_shapes(&self, _ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		Ok(())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
        (self.f.f)(ctx.get_output(&self.output));
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use crate::{base_ops::apply::{apply, apply_into}, graph::Node};

	#[test]
	fn forward_fill() {
        let output = apply(|mut x| x.fill(1.25), &[13, 33]).unwrap();

		assert!(output
			.calc()
			.unwrap().iter().all(|&e| (e - 1.25).abs() < f32::EPSILON));
	}

	#[test]
	fn forward_into_count() {
		let output = Node::new(&[13, 33]);
        
        apply_into(|mut x| {x.iter_mut().enumerate().for_each(|(i, x)| *x = i as f32)}, &output).unwrap();

		assert!(output
			.calc()
			.unwrap().iter().enumerate().all(|(i, &e)| (e - i as f32).abs() < f32::EPSILON));
	}


    #[test]
	fn shape() {
		let output = apply(|mut x| x.fill(1.25), &[13, 33]).unwrap();

		assert!(output
			.calc().unwrap().shape() == [13, 33]);
	}
}
