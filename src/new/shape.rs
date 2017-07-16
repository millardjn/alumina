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
		
		/// Cannot merge shapes which have a fixed but different total number of elements.
		MergeIncompatibleFlatSize{}
		
		/// Cannot merge shapes which have a different number of elements
		MergeIncompatibleChannelDimension{}
		
		/// Cant merge shapes which have different numbers of higher dimensions, unless one has no higher dimensions
		MergeIncompatibleRank{}
		
		
		MergeIncompatibleHigherDimension{}
		
		
		UnderDeterminedFlatSize{}
	}
}




#[derive(Clone, Debug, PartialEq)]
pub enum NodeDim {
	Unknown,
	Fixed(usize),
	/// Inclusive range of possible sizes for a given dimension
	Range{lower: usize, upper: usize},
}

impl From<usize> for NodeDim{
	fn from(s: usize) -> NodeDim {
		NodeDim::Fixed(s)
	}
}

impl<'a> From<&'a usize> for NodeDim{
	fn from(s: &usize) -> NodeDim {
		NodeDim::Fixed(*s)
	}
}

impl From<(usize, usize)> for NodeDim{
	fn from((lower, upper): (usize, usize)) -> NodeDim {
		NodeDim::Range{lower, upper}
	}
}

impl<'a> From<(&'a usize, &'a usize)> for NodeDim{
	fn from((lower, upper): (&usize, &usize)) -> NodeDim {
		NodeDim::Range{lower: *lower, upper: *upper}
	}
}

impl NodeDim {
	fn multiply (&self, other: &NodeDim) -> NodeDim{
		match (self, other) {
			(&Unknown, _) | (_, &Unknown) => Unknown,
			(&Fixed(x), &Fixed(y)) => Fixed(x*y),
			(&Fixed(x), &Range{upper, lower}) | (&Range{upper, lower}, &Fixed(x)) => Range{upper: upper*x, lower: lower*x},
			(&Range{upper: upper1, lower: lower1}, &Range{upper: upper2, lower: lower2}) => Range{upper: upper1*upper2, lower: lower1*lower2},
		}
	}

	fn merge(&self, other: &NodeDim) -> Result<NodeDim>{
		match (self, other) {
			(&Unknown, x) | (x, &Unknown) => Ok(x.clone()),	
			(&Fixed(x), &Fixed(y)) => if x == y {Ok(Fixed(x))} else {bail!(ErrorKind::MergeIncompatibleHigherDimension)},
			(&Fixed(v), &Range{upper, lower}) | (&Range{upper, lower}, &Fixed(v)) => if v >= lower && v <= upper {Ok(Fixed(v))} else {bail!(ErrorKind::MergeIncompatibleHigherDimension)},
			(&Range{upper: upper1, lower: lower1}, &Range{upper: upper2, lower: lower2}) =>  {
				let upper = cmp::min(upper1, upper2);
				let lower = cmp::max(lower1, lower2);
				if lower == upper {
					Ok(Fixed(lower))
				} else if lower < upper {
					Ok(Range{upper: upper, lower: lower})
				} else {
					bail!(ErrorKind::MergeIncompatibleHigherDimension)
				}
			},	
		}
		
	}

}



#[derive(Clone, PartialEq)]
pub struct NodeShape{
	pub dimensions: SmallVec<[NodeDim;6]>, // None indicates Runtime Determined, Range indicates acceptible range for fixed size
}

impl <T: Into<NodeDim> + Clone, I: IntoIterator<Item=T>> From<I> for NodeShape {
	fn from(i: I) -> NodeShape {
		let dimensions: SmallVec<[NodeDim; 6]> = i.into_iter().map(|e| e.clone().into()).collect();
		if let NodeDim::Fixed(_) = dimensions[dimensions.len() - 1] {} else {panic!("Final dimension in node shape (channels) must be of fixed size")}
		NodeShape{dimensions}
	}
}

impl NodeShape{
	
	pub fn channels(&self) -> usize {
		match self.dimensions[self.dimensions.len() - 1] {
			NodeDim::Fixed(dim) => dim,
			_ => unreachable!(),
		}
	}

	pub fn spatial_dimensions(&self) -> &[NodeDim]{
		&self.dimensions[1..]
	}
	
	/// Should be called and only called by operations prior to propagating shape constraints
	/// The higher dimension ranges are collapsed to the lower bound, and all None entries are replaced with the range 0:0
	pub fn collapse_dimensions_to_minimum(&mut self) -> Result<()>{
		
		for i in 0.. self.dimensions.len(){
			match &self.dimensions[i] {
				&Unknown => bail!(ErrorKind::IncompleteNodeShape),
				&Fixed(_) => {},
				&Range{lower, ..} => self.dimensions[i] = Fixed(lower),
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
				&Fixed(v) => size *= v,
				&Range{upper, lower} if upper == lower => size *= lower,
				_ => bail!(ErrorKind::UnderDeterminedFlatSize),
			}
		}

		Ok(size)
	}
	
	/// If range upper != lower, lowe will be used.
	pub fn to_data_shape(&self, n: usize) -> Result<IxDyn> {
		let mut dims: SmallVec<[usize; 6]> = SmallVec::new();
		dims.push(n);
		for dim in self.spatial_dimensions().iter() {
			match dim {
				 &Fixed(v) => dims.push(v),
				_ => bail!(ErrorKind::IncompleteNodeShape),
			}
		}
		Ok(ndarray::IxDyn(&dims))
	}
	
	pub fn is_fixed(&self) -> bool {
		self.dimensions.iter().all(|x|{
			match x {
				&Fixed(_) => true,
				_ => false,
			}
		})
	}
	
	pub fn ndim(&self) -> usize {
		self.dimensions.len()
	}
	
	pub fn merge(&self, other: &NodeShape) -> Result<NodeShape>{
		
		if self.is_fixed() && other.is_fixed() {
			self.merge_fixed(other)	
		} else if self.channels() != other.channels() {
			bail!(ErrorKind::MergeIncompatibleChannelDimension)
		} else if self.ndim() != other.ndim() {
			bail!(ErrorKind::MergeIncompatibleRank)
		} else {
			let mut vec = SmallVec::new();
			for (s, o) in self.spatial_dimensions().iter().zip(other.spatial_dimensions()){
				vec.push(s.merge(o)?);
			}
			vec.push(self.channels().into());
			Ok(NodeShape{dimensions: vec})
		}
	}
	
	#[inline(always)]
	fn merge_fixed(&self, other: &NodeShape) -> Result<NodeShape>{

		match (self.flat_size(), other.flat_size()){
			(Fixed(x), Fixed(y)) if x == y => {},
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
			for (s, o) in self.spatial_dimensions().iter().zip(other.spatial_dimensions()){
				vec.push(s.merge(o)?);
			}
			vec.push(self.channels().into());
			Ok(NodeShape{dimensions: vec})
		}	
	}
}
