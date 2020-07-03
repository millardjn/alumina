






PaddingShape {
	shape: Vec<(isize, isize)>
}



/// Collapse outer dimensions, shuffling entries into the channel dimension
///
/// Decrease size of higher dimensions by given factors by mapping from each spaxel to chunks of the channel dimension.
/// Output channel dimension is increased by the product of the collapse factors. Inverse operation of Expand.
#[must_use]
#[derive(Clone, Debug)]
pub struct Pad {
	input: Node,
	output: Node,
	shape: PaddingShape,
}

impl Pad {
	pub fn new<I, O>(input: I, output: O, factors: &[usize]) -> Self
	where
		I: Into<Node>,
		O: Into<Node>,
	{
		let input = input.into();
		let output = output.into();

		Pad {
			input,
			output,
			factors: factors.to_vec(),
		}
	}
}

impl OpBuilder for Pad {
	type InstanceType = PadInstance;

	fn type_name(&self) -> &'static str {
		"Pad"
	}

	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input.clone()]
	}

	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		if self.input.shape().len() != self.factors.len() + 1 {
			return Err(format!(
				"The input shape ({}) must be 1 axis larger than the number of factors ({:?})",
				self.input.shape(),
				self.factors
			)
			.into());
		}

		if self.factors.iter().any(|&f| f == 0) {
			return Err(format!("All factors ({:?}) must be greater than zero.", self.factors).into());
		}

		Ok(CollapseInstance::new(
			self.input.inner().clone(),
			self.output.inner().clone(),
			self.factors,
		))
	}
}