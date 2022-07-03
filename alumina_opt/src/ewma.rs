use alumina_core::graph::Node;
use indexmap::IndexMap;
use ndarray::{ArrayBase, ArrayD, Zip};

// Exponential weighted average of node values
pub struct Ewma {
	values: IndexMap<Node, ArrayD<f32>>,
	beta: f32,
	update_count: i32,
}

impl Ewma {
	pub fn new<I: Into<Node>, N: IntoIterator<Item = I>>(nodes: N, beta: f32) -> Self {
		assert!(beta > 0.0);
		assert!(beta < 1.0);
		let mut values = IndexMap::new();
		for node in nodes {
			let node = node.into();
			let value = ArrayBase::zeros(
				node.value_shape()
					.expect("All emwa nodes must have values be initialised"),
			);
			values.insert(node, value);
		}
		Self {
			values,
			beta,
			update_count: 0,
		}
	}

	// Return the latest EMWA for each node
	pub fn get(&self) -> IndexMap<Node, ArrayD<f32>> {
		let correction = 1.0 / (self.beta.powi(self.update_count)).max(1.0);
		self.values
			.iter()
			.map(|(node, v)| (node.clone(), v * correction))
			.collect()
	}

	// Update the EMWA from the current node values
	pub fn update(&mut self) {
		let beta = self.beta;
		for (node, value) in &mut self.values {
			value.map_inplace(|e| *e *= beta);
			Zip::from(value)
				.and(&node.value().expect("Emwa node did not have a value set"))
				.for_each(|emwa, next| {
					*emwa = (1.0 - beta) * next;
				});
		}
	}

	// Update the EMWA from custom node values
	pub fn update_custom(&mut self, values: IndexMap<Node, ArrayD<f32>>) {
		let beta = self.beta;
		for (node, value) in &mut self.values {
			value.map_inplace(|e| *e *= beta);
			Zip::from(value)
				.and(
					values
						.get(node)
						.expect("Custom emwa values did not contain all required nodes"),
				)
				.for_each(|emwa, next| {
					*emwa = (1.0 - beta) * next;
				});
		}
	}
}
