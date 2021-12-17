use alumina_core::{
	errors::OpBuildError,
	graph::Node
};

use crate::{elementwise::{logistic::logistic, mul::mul, ln::ln, offset::offset, negative::negative, subtract::subtract}};

/// Calculates the logistic function of the logits followed by BinaryCrossEntropy of that result with the
/// supplied labels.
///
/// `let output = mul(labels, negative(ln(softmax(logits))))`
///
/// The output node has the shape of the logits and labels.
pub fn logistic_cross_entropy<I1, I2>(logits: I1, labels: I2) -> Result<Node, OpBuildError>
where
	I1: Into<Node>,
	I2: Into<Node>,
{
	let logits = logits.into();
	let labels = labels.into();
	let name = format!("softmax_cross_entropy({},{})", logits, labels);

	let probability = logistic(logits)?;

	let true_loss = mul(&labels, ln(&probability)?)?;

	let false_loss = mul(offset(labels, -1.0)?, ln(negative(offset(probability, -1.0)?)?)?)?;

	let output = subtract(false_loss, true_loss)?.set_name_unique(&name);

	Ok(output)
}


#[cfg(test)]
mod tests {
	use super::logistic_cross_entropy;
	use alumina_core::graph::Node;
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};
	use indexmap::indexset;

	use ndarray::arr1;
	#[test]
	fn forward_test() {
		let logits = Node::new(&[8])
			.set_value(arr1(&
				[0.2, 0.4, 0.6, 0.8, -1.2, -1.4, -1.6, -1.8],
			))
			.set_name("logits");

		let labels = Node::new(&[8])
			.set_value(arr1(&
				[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
			))
			.set_name("labels");

		let hor_groups = logistic_cross_entropy(&logits, &labels).unwrap();

		assert!(hor_groups
			.calc()
			.unwrap()
			.all_relatively_close(&arr1(&[0.798_138_86, 0.513_015_32, 1.037_488, 0.371_100_66, 0.263_282_48, 1.620_417_4, 1.783_900_7, 0.152_977_62 ]), 1e-5));
	}

	#[test]
	fn grad_numeric_test() {
		let input1 = Node::new(&[13, 33]).set_name("input1");
		let input2 = Node::new(&[13, 33]).set_name("input2");

		let output = logistic_cross_entropy(&input1, &input2).unwrap();

		GradNumericTest::new(&output, &indexset![&input1, &input2])
			.step_size(1e-3)
			.tolerance(1e-4)
			.run();
	}
}
