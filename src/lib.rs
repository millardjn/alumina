#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]
#![recursion_limit = "1024"]

#[macro_use] 
extern crate error_chain;
extern crate ndarray;
extern crate odds;
extern crate walkdir;
extern crate smallvec;
extern crate rand;
extern crate matrixmultiply_mt as matrixmultiply;
extern crate byteorder;

extern crate num_cpus;
extern crate threadpool;
#[macro_use] extern crate lazy_static;
extern crate typenum;
extern crate typenum_loops;
extern crate generic_array;
extern crate scoped_threadpool;
extern crate ordermap;
#[macro_use] extern crate matches;
extern crate arrayvec;
extern crate image;
extern crate rayon;
extern crate divide_range;

//#![deny(missing_docs)]

/// NodeTag / OpTag constructor for nodes and operations
///
/// Returns a value of the type `Vec<NodeTag>` or `Vec<OpTag>`, relying on inference from surrounding code.
///
/// Three or four types of values can be entered as a list into the macro,
/// with the following conversions taking place:
///
/// | Value | | Output |
/// |---|---|---|
/// | Parameter  | => | NodeTag::Parameter`                |
/// | 5          | => | NodeTag::Int(5)`                   |
/// | "input"    | => | NodeTag::Str("input".to_string())` |
/// | id_binding | => | NodeTag::Id(id_binding)`           |
///
/// All of these conversions are also possible for `OpTag` with the exception of Parameter values
///
/// #Example
/// ```
/// #[macro_use]
/// extern crate alumina;
/// use alumina::id::NodeTag;
///
/// fn main() {
///		let s1: Vec<NodeTag> = tag![Parameter, 5, "input"];
///		let s2: Vec<NodeTag> = vec![NodeTag::Parameter, NodeTag::Int(5), NodeTag::Str("input".to_string())];
///		assert_eq!(s1, s2);
/// }
/// ```
#[macro_export]
macro_rules! tag(
    (@parse $v:expr) => {
        ($v).into()
    };

    ( $($x:expr),* ) => {
        {
			#[allow(unused_imports)]
			use $crate::id::NodeTag::Parameter;
			vec![
				$(
					tag![@parse $x],
				)*
			]
		}
    };
);


/// NodeShape constructor.
///
/// Returns a value of the type `NodeShape`
///
/// Three types of values can be entered as a list into the macro,
/// with the following conversions taking place:
///
/// | Value | | Output |
/// |---|---|---|
/// | Unknown | => | NodeDim::Unknown` |
/// | 5       | => | NodeDim::Known(5)` |
/// | (4, 7)  | => | NodeDim::Interval{lower:4, upper:7}` |
///
/// #Example
/// ```
/// #[macro_use]
/// extern crate alumina;
/// use alumina::shape::{NodeDim, NodeShape};
///
/// fn main() {
///		let s1: NodeShape = shape![Unknown, 5, (3, 9), 7];
///		let s2: NodeShape = NodeShape::from(&[NodeDim::Unknown, NodeDim::Known(5), NodeDim::Interval{lower:3, upper:9}, NodeDim::Known(7)]);
///		assert_eq!(s1, s2);
/// }
/// ```
#[macro_export]
macro_rules! shape(

	(@parse $x:expr) => {
		($x).into()
	};

	[ $($x:expr),+ ] => {
		{
			#[allow(unused_imports)]
			use $crate::shape::NodeDim::Unknown;
			let slice: &[$crate::shape::NodeDim] = &[
				$(
					shape![@parse $x],
				)*
			];
			$crate::shape::NodeShape::from(slice)
		}
	};

);



pub mod shape;
pub mod graph;
pub mod ops;
pub mod opt;
pub mod data;
pub mod init;
pub mod id;
pub mod storage;