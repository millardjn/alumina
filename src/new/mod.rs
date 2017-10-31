//#![deny(missing_docs)]

/// NodeTag / OpTag constructor for nodes and operations
///
/// Returns a value of the type `Vec<NodeTag>` or `Vec<OpTag>`, relying on inference from surrounding code.
///
/// Three or four types of values can be entered as a list into the macro,
/// with the following conversions taking place:
///
///     Parameter values:     Parameter  => NodeTag::Parameter
///     usize values:         5          => NodeTag::Int(5)
///     str|String values:    "input"    => NodeTag::Str("input".to_string())
///     NodeID values         id_binding => NodeTag::Id(id_binding)
///
/// All of these conversions are also possible for `OpTag` with the exception of Parameter values
///
/// #Example
/// ```
/// #[macro_use]
/// extern crate alumina;
/// use alumina::new::graph::NodeTag;
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
			use $crate::new::graph::NodeTag::Parameter;
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
///     Unknown values:   Unknown => NodeDim::Unknown
///     usize values:     5       => NodeDim::Known(5)
///     usize tuples:     (4, 7)  => NodeDim::Interval{lower:4, upper:7}
///
/// #Example
/// ```
/// #[macro_use]
/// extern crate alumina;
/// use alumina::new::shape::{NodeDim, NodeShape};
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
			use $crate::new::shape::NodeDim::Unknown;
			let slice: &[$crate::new::shape::NodeDim] = &[
				$(
					shape![@parse $x],
				)*
			];
			$crate::new::shape::NodeShape::from(slice)
		}
	};

);



pub mod shape;
pub mod graph;
pub mod ops;
pub mod opt;
pub mod data;
pub mod init;