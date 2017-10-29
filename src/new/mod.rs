//#![deny(missing_docs)]

/// tag constructor for nodes and operations
#[macro_export]
macro_rules! tag(

    (@parse Parameter) => {
        $crate::new::graph::NodeTag::Parameter
    };
    
    (@parse $v:expr) => {
        ($v).into()
    };
    
    ( $( $x:expr ),* ) => {
        vec![
            $(
                tag!(@parse $x),
            )*
        ]
    };
    
);

/// NodeShape constructor.
///
/// Returns a value of the type `&[NodeDim]`
///
/// Three types of vales can be entered as a list into the macro,
/// with the following conversions taking place:
/// Unknown values:   Unknown => NodeDim::Unknown
/// usize values:     5       => NodeDim::Known(5)
/// usize tuples:     (4, 7)  => NodeDim::Interval{lower:4, upper:7}
///
/// note: 
///
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

	(@parse Unknown) => {
		$crate::new::shape::NodeDim::Unknown
	};
	
	(@parse ($l:expr, $u:expr)) => {
		$crate::new::shape::NodeDim::Interval{lower: $l, upper: $u}
	};
	
	(@parse $v:expr) => {
		$crate::new::shape::NodeDim::Known($v)
	};
	
	
	( $( $x:tt ),* ) => {
		{let slice: &[$crate::new::shape::NodeDim] = &[
			$(
				shape!(@parse $x),
			)*
		];
		$crate::new::shape::NodeShape::from(slice)}
	};
	
);

pub mod shape;
pub mod graph;
pub mod ops;
pub mod opt;
pub mod data;
pub mod init;