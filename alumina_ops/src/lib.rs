//! Types (`OpBuilder`, `OpInstance`) for constructing and defining an `Op` within the graph.

pub mod boolean;
pub mod elementwise;
pub mod grad;
pub mod manip;
pub mod math;
pub mod nn;
pub mod panicking;
pub mod pool;
pub mod reduce;
pub mod regularisation;
mod sgemm;
pub mod shape;

use alumina_core::errors::OpBuildError;

/// Unwraps the `Result` but prints the `Err` using `Display` rather than `Debug`
pub fn build_or_pretty_panic<T>(result: Result<T, OpBuildError>, op_name: &str) -> T {
	match result {
		Ok(x) => x,
		Err(err) => panic!("Error building {} Op \n{}\n", op_name, err),
	}
}

// /// Appends an underscore and number, starting with "_0", to the supplied name to ensure uniqueness
// pub fn unique_node_name(base: String) -> String {
// }

// /// Appends a number, starting with 0, to the supplied name to ensure uniqueness
// pub fn unique_op_name(base: String) -> String {
// }

// pub trait OpAny {
// 	fn as_any(&self) -> &Any;
// }

// impl<T> OpAny for T
// where
// 	T: 'static + OpInstance,
// {
// 	fn as_any(&self) -> &Any {
// 		self
// 	}
// }
