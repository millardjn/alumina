use smallvec::SmallVec;
use self::NodeDim::*;
use ndarray::ArrayD;
use ndarray;
use ndarray::prelude::*;
use std::cmp;


error_chain!{
	errors {
		/// Cannot convert a NodeShape into a Data Shape when some higher dimensions are still Unknown after propagating constraints from all prior operations.
		IncompleteNodeShape{}
		
		/// Cannot merge shapes which have a Known but different total number of elements.
		MergeIncompatibleFlatSize{}
		
		/// Cannot merge shapes which have a different number of channels
		MergeIncompatibleChannelDimension{}
		
		/// Cant merge shapes which have different numbers of higher dimensions, unless one has no higher dimensions
		MergeIncompatibleRank{}

		MergeIncompatibleHigherDimension{}

		UnderDeterminedFlatSize{}
	}
}


/// NodeShape argument constructor.
///
/// Returns a value of the type `&[NodeDim]`
///
/// Three types of vales can be entered as a list into the macro,
/// with the following conversions taking place:
/// Unknown values:   Unknown => NodeDim::Unknown
/// usize values:     5       => NodeDim::Known(5)
/// usize tuples:     (4, 7)  => NodeDim::IncInterval{lower:4, upper:7}
///
///
/// ```
/// #[macro_use]
/// extern crate alumina;
/// use alumina::new::shape::{NodeDim, NodeShape};
///
/// fn main() {
///		let s1: NodeShape = shape![Unknown, 5, (3, 9), 7];
///		let s2: NodeShape = NodeShape::from(&[NodeDim::Unknown, NodeDim::Known(5), NodeDim::IncInterval{lower:3, upper:9}, NodeDim::Known(7)]);
///		assert_eq!(s1, s2);
/// }
/// ```
#[macro_export]
macro_rules! shape(

	(@parse Unknown) => {
		$crate::new::shape::NodeDim::Unknown
	};
	
	(@parse ($l:expr, $u:expr)) => {
		$crate::new::shape::NodeDim::IncInterval{lower: $l, upper: $u}
	};
	
	(@parse $v:expr) => {
		$crate::new::shape::NodeDim::Known($v)
	};
	
	
	( $( $x:tt ),* ) => {
		{let slice: &[NodeDim] = &[
			$(
				shape!(@parse $x),
			)*
		];
		$crate::new::shape::NodeShape::from(slice)}
	};
	
);

#[derive(Clone, Debug, PartialEq)]
pub enum NodeDim {

	/// A dimension which does not yet have any constraints
	Unknown,

	/// A fully constrained dimension
	Known(usize),

	/// Inclusive interval of possible sizes for a given dimension
	IncInterval{lower: usize, upper: usize},
}

impl<'a> From<&'a NodeDim> for NodeDim{
	fn from(s: &NodeDim) -> NodeDim {
		s.clone()
	}
}

impl From<usize> for NodeDim{
	fn from(s: usize) -> NodeDim {
		NodeDim::Known(s)
	}
}

impl<'a> From<&'a usize> for NodeDim{
	fn from(s: &usize) -> NodeDim {
		NodeDim::Known(*s)
	}
}

impl From<(usize, usize)> for NodeDim{
	fn from((lower, upper): (usize, usize)) -> NodeDim {
		NodeDim::IncInterval{lower, upper}
	}
}

impl<'a> From<(&'a usize, &'a usize)> for NodeDim{
	fn from((lower, upper): (&usize, &usize)) -> NodeDim {
		NodeDim::IncInterval{lower: *lower, upper: *upper}
	}
}

impl NodeDim {
	fn multiply (&self, other: &NodeDim) -> NodeDim{
		match (self, other) {
			(&Unknown, _) | (_, &Unknown) => Unknown,
			(&Known(x), &Known(y)) => Known(x*y),
			(&Known(x), &IncInterval{upper, lower}) | (&IncInterval{upper, lower}, &Known(x)) => IncInterval{upper: upper*x, lower: lower*x},
			(&IncInterval{upper: upper1, lower: lower1}, &IncInterval{upper: upper2, lower: lower2}) => IncInterval{upper: upper1*upper2, lower: lower1*lower2},
		}
	}

	fn merge(&self, other: &NodeDim) -> Result<NodeDim>{
		match (self, other) {
			(&Unknown, x) | (x, &Unknown) => Ok(x.clone()),	
			(&Known(x), &Known(y)) => if x == y {Ok(Known(x))} else {bail!(ErrorKind::MergeIncompatibleHigherDimension)},
			(&Known(v), &IncInterval{upper, lower}) | (&IncInterval{upper, lower}, &Known(v)) => if v >= lower && v <= upper {Ok(Known(v))} else {bail!(ErrorKind::MergeIncompatibleHigherDimension)},
			(&IncInterval{upper: upper1, lower: lower1}, &IncInterval{upper: upper2, lower: lower2}) =>  {
				let upper = cmp::min(upper1, upper2);
				let lower = cmp::max(lower1, lower2);
				if lower == upper {
					Ok(Known(lower))
				} else if lower < upper {
					Ok(IncInterval{upper: upper, lower: lower})
				} else {
					bail!(ErrorKind::MergeIncompatibleHigherDimension)
				}
			},	
		}
		
	}

}



#[derive(Clone, Debug, PartialEq)]
pub struct NodeShape{
	pub dimensions: SmallVec<[NodeDim;6]>,
}

impl <T: Into<NodeDim> + Clone, I: IntoIterator<Item=T>> From<I> for NodeShape {
	fn from(i: I) -> NodeShape {
		let dimensions: SmallVec<[NodeDim; 6]> = i.into_iter().map(|e| e.clone().into()).collect();
		if let NodeDim::Known(_) = dimensions[dimensions.len() - 1] {} else {panic!("Final dimension in node shape (channels) must be of Known size")}
		NodeShape{dimensions}
	}
}

impl NodeShape{
	
	pub fn channels(&self) -> usize {
		match self.dimensions[self.dimensions.len() - 1] {
			NodeDim::Known(dim) => dim,
			_ => unreachable!(),
		}
	}
	
	/// Should be called and only called by operations prior to propagating shape constraints
	/// The higher dimension ranges are collapsed to the lower bound, and any Unknown entries will result in an IncompleteNodeShape Error
	pub fn collapse_dimensions_to_minimum(&mut self) -> Result<()>{
		
		for i in 0.. self.dimensions.len(){
			match &self.dimensions[i] {
				&Unknown => bail!(ErrorKind::IncompleteNodeShape),
				&Known(_) => {},
				&IncInterval{lower, ..} => self.dimensions[i] = Known(lower),
			};
			
		}
		Ok(())
	}
	
	pub fn flat_size(&self) -> NodeDim {
		self.dimensions.iter().fold(1.into(), |prev, item| prev.multiply(item))
	}
	
	pub fn force_flat_size(&self) -> Result<usize>{
		let mut size = 1;

		for dim in self.dimensions.iter() {
			match dim {
				&Known(v) => size *= v,
				&IncInterval{upper, lower} if upper == lower => size *= lower,
				_ => bail!(ErrorKind::UnderDeterminedFlatSize),
			}
		}

		Ok(size)
	}
	
	/// If all dimension values are `Known` this will 
	/// This should generally only be called after `collapse_dimensions_to_minimum`
	pub fn to_data_shape(&self) -> Result<IxDyn> {
		let mut dims = Vec::with_capacity(self.dimensions.len());

		//dims.push(n);
		for dim in self.dimensions.iter() {
			match dim {
				 &Known(v) => dims.push(v),
				_ => bail!(ErrorKind::IncompleteNodeShape),
			}
		}
		Ok(ndarray::IxDyn(&dims))
	}
	
	pub fn is_known(&self) -> bool {
		self.dimensions.iter().all(|x|{
			match x {
				&Known(_) => true,
				_ => false,
			}
		})
	}
	
	pub fn ndim(&self) -> usize {
		self.dimensions.len()
	}
	
	pub fn merge(&self, other: &NodeShape) -> Result<NodeShape>{

		if self.is_known() && other.is_known() {
			self.merge_known(other)	
		} else if self.channels() != other.channels() {
			bail!(ErrorKind::MergeIncompatibleChannelDimension)
		} else if self.ndim() != other.ndim() {
			bail!(ErrorKind::MergeIncompatibleRank)
		} else {
			let mut vec = SmallVec::new();
			for (s, o) in self.dimensions.iter().zip(&other.dimensions){
				vec.push(s.merge(o)?);
			}
			Ok(NodeShape{dimensions: vec})
		}
	}
	
	#[inline(always)]
	fn merge_known(&self, other: &NodeShape) -> Result<NodeShape>{

		match (self.flat_size(), other.flat_size()){
			(Known(x), Known(y)) if x == y => {},
			_ => bail!(ErrorKind::MergeIncompatibleFlatSize),
		}

		if self.ndim() == 1 {
			Ok(other.clone())
		} else if other.ndim() == 1 {
			Ok(self.clone())
		} else if self.ndim() != other.ndim() {
			bail!(ErrorKind::MergeIncompatibleRank)
		} else if self.channels() != other.channels() {	
			bail!(ErrorKind::MergeIncompatibleChannelDimension)
		} else {
			let mut vec = SmallVec::new();
			for (s, o) in self.dimensions.iter().zip(&other.dimensions){
				vec.push(s.merge(o)?);
			}
			Ok(NodeShape{dimensions: vec})
		}	
	}
}
