use std::cmp::*;

// #[derive(Clone, PartialEq, Debug)]
// pub struct InclusiveRange {
// 	lower: usize,
// 	upper: usize,
// }

// impl InclusiveRange{
// 	pub fn new(lower: usize, upper: usize)-> InclusiveRange{
// 		InclusiveRange{
// 			lower: lower,
// 			upper: upper,
// 		}
// 	}
	
// 	pub fn new_fixed(val: usize) -> InclusiveRange{
// 		InclusiveRange{
// 			lower: val,
// 			upper: val,
// 		}
// 	}
	
// 	/// panics if range if upper != lower, otherwise returns lower
// 	pub fn get_fixed_val(&self) -> usize{
// 		assert!(self.upper == self.lower, "Range not collapsed");
// 		self.lower
// 	}
// }

#[derive(Debug)]
pub enum ShapeError{
	/// Cannot convert a NodeShape into a Data Shape when some higher dimensions are still Unknown after propagating constraints from all prior operations.
	IncompleteNodeShape,
	
	/// Cannot merge shapes which have a fixed but different total number of elements.
	MergeIncompatibleFlatSize,
	
	/// Cannot merge shapes which have a different number of elements
	MergeIncompatibleChannelDimension,
	
	/// Cant merge shapes which have different numbers of higher dimensions, unless one has no higher dimensions
	MergeIncompatibleRank,
	
	
	MergeIncompatibleHigherDimension,
	
	
	UnderDeterminedFlatSize,
}


use self::Dimension::*;

#[derive(Clone, Debug)]
pub enum Dimension {
	Unknown,
	Fixed(usize),
	Range{upper: usize, lower: usize},
}

impl Dimension {
	fn multiply (&self, other: &Dimension) -> Dimension{
		match (self, other) {
			(&Unknown, _) | (_, &Unknown) => Unknown,
			(&Fixed(x), &Fixed(y)) => Fixed(x*y),
			(&Fixed(x), &Range{upper, lower}) | (&Range{upper, lower}, &Fixed(x)) => Range{upper: upper*x, lower: lower*x},
			(&Range{upper: upper1, lower: lower1}, &Range{upper: upper2, lower: lower2}) => Range{upper: upper1*upper2, lower: lower1*lower2},
		}
	}

	fn merge(&self, other: &Dimension) -> Result<Dimension, ShapeError>{
		match (self, other) {
			(&Unknown, x) | (x, &Unknown) => Ok(x.clone()),	
			(&Fixed(x), &Fixed(y)) => if x == y {Ok(Fixed(x))} else {Err(ShapeError::MergeIncompatibleHigherDimension)},
			(&Fixed(v), &Range{upper, lower}) | (&Range{upper, lower}, &Fixed(v)) => if v >= lower && v <= upper {Ok(Fixed(v))} else {Err(ShapeError::MergeIncompatibleHigherDimension)},
			(&Range{upper: upper1, lower: lower1}, &Range{upper: upper2, lower: lower2}) =>  {
				let upper = min(upper1, upper2);
				let lower = max(lower1, lower2);
				if lower == upper {
					Ok(Fixed(lower))
				} else if lower < upper {
					Ok(Range{upper: upper, lower: lower})
				} else {
					Err(ShapeError::MergeIncompatibleHigherDimension)
				}
			},	
		}
		
	}

}

#[derive(Clone, PartialEq, Debug)]
pub struct DataShape{
	
	pub channels: usize,
	
	pub spatial_dimensions: Vec<usize>,
	/// The number of training examples being processed in parallel
	pub n: usize,
}

impl DataShape{

	
	pub fn flat_size_single(&self) -> usize {
		self.spatial_dimensions.iter().fold(self.channels, |prev, &item| prev*item)
	}
	
	pub fn flat_size_all(&self) -> usize {
		self.spatial_dimensions.iter().fold(self.channels * self.n, |prev, &item| prev*item)
	}
	
	/// Rank excluding 'n'
	pub fn rank(&self) -> usize {
		self.spatial_dimensions.len() + 1
	}
	
	pub fn new(channels: usize, higher_dims: Vec<usize>, n: usize) -> DataShape{
		DataShape{
			channels: channels, 
			spatial_dimensions: higher_dims, 
			n: n
		}
	}
	
	pub fn new_flat(size: usize, n: usize) -> DataShape{
		DataShape{
			channels: size, 
			spatial_dimensions: vec![], 
			n: n
		}
	}
	
	pub fn to_node_shape(&self) -> NodeShape {
		NodeShape::new(self.channels, self.spatial_dimensions.clone())
	}
}


#[derive(Clone, Debug)]
pub struct NodeShape{
	pub channels: usize,
	pub spatial_dimensions: Vec<Dimension>, // None indicates Runtime Determined, Range indicates acceptible range for fixed size
} // depth, dimensions

impl NodeShape{
	
	pub fn new(channels: usize, higher_dims: Vec<usize>) -> NodeShape{
		NodeShape{
			channels: channels, 
			spatial_dimensions: higher_dims.iter().map(|&x| Dimension::Fixed(x)).collect(), 
		}
	}
	
	/// Creates a new NodeShape, with higher timensions to be determined at runtime
	pub fn new_flex(channels: usize, num_higher_dims: usize) -> NodeShape{
		NodeShape{
			channels: channels, 
			spatial_dimensions: vec![Dimension::Unknown; num_higher_dims], 
		}
	}
	
	pub fn new_flat(size: usize) -> NodeShape{
		NodeShape{
			channels: size, 
			spatial_dimensions: vec![], 
		}
	}
	
	
	/// Should be called and only called by operations prior to propagating shape constraints
	/// The higher dimension ranges are collapsed to the lower bound, and all None entries are replaced with the range 0:0
	pub fn collapse_ranges_to_minimum(&mut self) -> Result<(), ShapeError>{
		
		for i in 0.. self.spatial_dimensions.len(){
			//self.spatial_dimensions [i] =
			match &self.spatial_dimensions[i] {
				&Unknown => return Err(ShapeError::IncompleteNodeShape),
				&Fixed(_) => {},
				&Range{lower, ..} => self.spatial_dimensions[i] = Fixed(lower),

				// None => return Err(ShapeError::IncompleteNodeShape),
				// Some(ref range) => InclusiveRange{upper: range.lower, lower: range.lower},
			};
			
		}
		Ok(())
	}
	
	pub fn flat_size(&self) -> Dimension {
		self.spatial_dimensions.iter().fold(Fixed(self.channels), |prev, item| prev.multiply(item))
	}
	
	pub fn force_flat_size(&self) -> Result<usize, ShapeError>{
		let mut size = self.channels;

		for dim in self.spatial_dimensions.iter() {
			match dim {
				&Fixed(v) => size *= v,
				&Range{upper, lower} if upper == lower => size *= lower,
				_ => return Err(ShapeError::UnderDeterminedFlatSize),
			}
		}

		Ok(size)
	}
	
	/// If range upper != lower, lowe will be used.
	pub fn to_data_shape(&self, n: usize) -> Result<DataShape, ShapeError> {

		let mut dims = vec![];

		for dim in self.spatial_dimensions.iter() {
			match dim {
				 &Fixed(v) => dims.push(v),
				_ => return Err(ShapeError::IncompleteNodeShape),
			}
		}

		Ok(DataShape{
			channels: self.channels,
			spatial_dimensions: dims,
			n: n,
		})

	}
	
	pub fn is_fixed(&self) -> bool {
		self.spatial_dimensions.iter().all(|x|{
			match x {
				&Fixed(_) => true,
				_ => false,
			}
		})
	}
	
	pub fn rank(&self) -> usize {
		self.spatial_dimensions.len() + 1
	}
	
	pub fn merge(&self, other: &NodeShape) -> Result<NodeShape, ShapeError>{
		
		if self.is_fixed() && other.is_fixed() {
			self.merge_fixed(other)	
		} else if self.channels != other.channels {
			Err(ShapeError::MergeIncompatibleChannelDimension)
		} else if self.rank() != other.rank() {
			Err(ShapeError::MergeIncompatibleRank)
		} else {
			
			let mut new = NodeShape::new_flat(self.channels);
			
			for i in 0..self.spatial_dimensions.len(){
				match self.spatial_dimensions[i].merge(&other.spatial_dimensions[i]) {
					Err(x) => return Err(x),
					Ok(range) => new.spatial_dimensions.push(range),
				}
				
			}
			
			Ok(new)
			
		}
			
		
	}
	
	#[inline(always)]
	fn merge_fixed(&self, other: &NodeShape) -> Result<NodeShape, ShapeError>{

		debug_assert!(match (self.flat_size(), self.flat_size()){
			(Fixed(x), Fixed(y)) => x == y,
			_ => false,
		});

		if self.rank() == 1 {
			Ok(other.clone())
		} else if other.rank() == 1 {
			Ok(self.clone())
		} else if self.rank() != other.rank() {
			Err(ShapeError::MergeIncompatibleRank)
		} else if self.channels != other.channels {	
			Err(ShapeError::MergeIncompatibleChannelDimension)
		} else {
		
			let mut new = NodeShape::new_flat(self.channels);
			
			for i in 0..self.spatial_dimensions.len(){
				match self.spatial_dimensions[i].merge(&other.spatial_dimensions[i]) {
					Err(x) => return Err(x),
					Ok(range) => new.spatial_dimensions.push(range),
				}
				
			}
			
			Ok(new)
			
		}	
	}
	
	
//	// How can we end up calling merge not fixed? maybe some operation that can broadcast its channels accross higher dimensions?
//	fn merge_not_fixed(&self, other: &NodeShape) -> Result<NodeShape, ShapeError>{
//		if self.0 != other.0 {
//			Err("To merge two non-fixed shapes column depth must be the same.")
//		} else if self.1.len() != other.1.len() {
//			Err("To merge two non-fixed shapes the number of dimensions must be the same")
//		} else {
//			Ok(Shape(self.0, vec![None; self.1.len()]))
//		}
//	}
		

}

