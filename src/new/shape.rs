use smallvec::SmallVec;
use self::NodeDim::*;
use ndarray;
use ndarray::prelude::*;
use std::cmp;


error_chain!{
	errors {
		/// Cannot convert a NodeShape into a Data Shape when some higher dimensions are still Unknown after propagating constraints from all prior operations.
		IncompleteNodeShape(shape: NodeShape){
			display("Nodeshape is incomplete: {:?}", shape)
		}
		
		/// Cannot merge shapes which have a Known but different total number of elements.
		MergeIncompatibleFlatSize(x: usize, y: usize){
			display("Shapes could not be merged as the had different flat sizes. Size1:{:?} Size2:{:?}", x, y)
		}
		
		/// Cannot merge shapes which have a different number of channels
		MergeIncompatibleChannelDimension{}
		
		/// Cant merge shapes which have different numbers of higher dimensions, unless one has no higher dimensions
		MergeIncompatibleRank{}

		MergeIncompatibleHigherDimension(x: NodeDim, y: NodeDim){
			display("Node dimensions could not be merged. Dim1:{:?} Dim2:{:?}", x, y)
		}

		UnderDeterminedFlatSize{}
	}
}




#[derive(Clone, Debug, PartialEq)]
pub enum NodeDim {

	/// A dimension which does not yet have any constraints
	Unknown,

	/// A fully constrained dimension
	Known(usize),

	/// Inclusive interval of possible sizes for a given dimension
	Interval{lower: usize, upper: usize},
}

impl NodeDim {
	pub fn unknown() -> NodeDim {
		NodeDim::Unknown
	}

	pub fn known(x: usize) -> NodeDim {
		NodeDim::Known(x)
	}

	pub fn interval(lower: usize, upper: usize) -> NodeDim {
		if upper < lower {
			panic!("NodeDim::Interval cannot be constructed with lower({}) > upper({})", lower, upper);
		} else if upper == lower {
			NodeDim::Known(lower)
		} else {
			NodeDim::Interval{lower, upper}
		}
		
	}
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
		NodeDim::Interval{lower, upper}
	}
}

impl<'a> From<(&'a usize, &'a usize)> for NodeDim{
	fn from((lower, upper): (&usize, &usize)) -> NodeDim {
		NodeDim::Interval{lower: *lower, upper: *upper}
	}
}

impl NodeDim {
	fn multiply (&self, other: &NodeDim) -> NodeDim {
		match (self, other) {
			(&Unknown, _) | (_, &Unknown) => Unknown,
			(&Known(x), &Known(y)) => Known(x*y),
			(&Known(x), &Interval{upper, lower}) | (&Interval{upper, lower}, &Known(x)) => Interval{upper: upper*x, lower: lower*x},
			(&Interval{upper: upper1, lower: lower1}, &Interval{upper: upper2, lower: lower2}) => {
				let upper = upper1* upper2;
				let lower = lower1* lower2;
				if lower == upper {
					Known(lower)
				} else if lower < upper {
					Interval{upper: upper, lower: lower}
				} else {
					panic!("NodeDim::Interval cannot be constructed with lower({}) > upper({})", lower, upper);
				}
			},
		}
	}

	fn merge(&self, other: &NodeDim) -> Result<NodeDim> {
		match (self, other) {
			(&Unknown, x) | (x, &Unknown) => Ok(x.clone()),	
			(&Known(x), &Known(y)) => if x == y {Ok(Known(x))} else {bail!(ErrorKind::MergeIncompatibleHigherDimension(self.clone(), other.clone()))},
			(&Known(v), &Interval{upper, lower}) | (&Interval{upper, lower}, &Known(v)) => if v >= lower && v <= upper {Ok(Known(v))} else {bail!(ErrorKind::MergeIncompatibleHigherDimension(self.clone(), other.clone()))},
			(&Interval{upper: upper1, lower: lower1}, &Interval{upper: upper2, lower: lower2}) =>  {
				let upper = cmp::min(upper1, upper2);
				let lower = cmp::max(lower1, lower2);
				if lower == upper {
					Ok(Known(lower))
				} else if lower < upper {
					Ok(Interval{upper: upper, lower: lower})
				} else {
					panic!("NodeDim::Interval cannot be constructed with lower({}) > upper({})", lower, upper);
				}
			},	
		}
		
	}

}



#[derive(Clone, Debug, PartialEq)]
pub struct NodeShape{
	dimensions: SmallVec<[NodeDim;6]>,
}

impl <T: Into<NodeDim> + Clone, I: IntoIterator<Item=T>> From<I> for NodeShape {
	fn from(i: I) -> NodeShape {
		let dimensions: SmallVec<[NodeDim; 6]> = i.into_iter()
			.map(|dim| dim.clone().into())
			.inspect(|dim| {
				match dim {
					&NodeDim::Interval{lower, upper} if upper < lower => {
						panic!("NodeDim::Interval cannot be constructed with lower({}) > upper({})", lower, upper);
					},
					_ => {},
				}
			})
			.collect();
		if dimensions.len() == 0 {panic!("Node shape must have at least 1 dimension.")}
		// TODO this may not be necessary or true:
		if let NodeDim::Known(_) = dimensions[dimensions.len() - 1] {} else {panic!("Final dimension in node shape (channels) must be of Known size")}
		NodeShape{dimensions}
	}
}

impl NodeShape{
	pub fn dimensions(&self) -> &[NodeDim] {
		&self.dimensions
	}


	// TODO consider removing
	pub fn channels(&self) -> usize {
		match self.dimensions[self.dimensions.len() - 1] {
			NodeDim::Known(dim) => dim,
			_ => unreachable!(),
		}
	}
	
	/// Should be called and only called by operations prior to propagating shape constraints
	/// The NodeDim::Interval0 are collapsed to the lower bound, and any NodeDim::Unknown entries will be replaced with 0
	pub fn collapse_dimensions_to_minimum(&mut self) {
		for i in 0.. self.dimensions.len(){
			match &self.dimensions[i] {
				&Unknown => self.dimensions[i] = Known(0),
				&Known(_) => {},
				&Interval{lower, ..} => self.dimensions[i] = Known(lower),
			};
			
		}
	}
	
	/// Returns a copy of the shape with all dimensions that arent Known(_) set to 1.
	pub fn collapse_to_broadcastable_dimension(&self) -> NodeShape {
		let mut shape = self.clone();

		for i in 0..shape.dimensions.len(){
			match &shape.dimensions[i] {
				&Known(_) => {},
				&Interval{lower, upper} if lower == upper => shape.dimensions[i] = Known(lower),
				_ => {shape.dimensions[i] = Known(1)},
			};
		}

		shape
	}

	pub fn flat_size(&self) -> NodeDim {
		self.dimensions.iter().fold(1.into(), |prev, item| prev.multiply(item))
	}
	
	pub fn force_flat_size(&self) -> Result<usize>{
		let mut size = 1;

		for dim in self.dimensions.iter() {
			match dim {
				&Known(v) => size *= v,
				&Interval{upper, lower} if upper == lower => size *= lower,
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
				_ => bail!(ErrorKind::IncompleteNodeShape(self.clone())),
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
			(Known(x), Known(y))  => {if x != y {bail!(ErrorKind::MergeIncompatibleFlatSize(x, y))}},
			_ => panic!(),
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
