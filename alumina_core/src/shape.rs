//! Types for describing the shapes of `Node`s.
use crate::errors::{DimError, ShapeError};
use ndarray::IxDyn;
use smallvec::SmallVec;
use std::{
	cmp::{self, Ordering},
	fmt::{self, Debug, Display, Write},
};

/// Easy `Into<NodeShape>` value for scalar nodes.
///
/// ```
/// # use alumina_core::graph::Node;
/// # use alumina_core::shape::SCALAR;
/// # use std::sync::Arc;
/// let scalar_node = Node::new(SCALAR);
/// assert_eq!(0, scalar_node.shape().len());
/// assert_eq!(1, scalar_node.shape().known_flat_size().unwrap());
/// ```
pub const SCALAR: &[NodeAxis] = &[];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeAxis {
	/// A fully constrained dimension
	Known { val: usize },

	/// Inclusive interval of possible sizes for a given dimension
	Interval { lower: usize, upper: usize },
}

impl NodeAxis {
	/// Create a `NodeAxis` contrained only by usize::MAX.
	pub fn unknown() -> NodeAxis {
		NodeAxis::interval(0, usize::max_value())
	}

	/// Create a `NodeAxis` with a known size.
	pub fn known(x: usize) -> NodeAxis {
		NodeAxis::Known { val: x }
	}

	/// Create a `NodeAxis` with a limited range of flexibility.
	///
	/// # Panics
	/// Panics if lower > upper.
	pub fn interval(lower: usize, upper: usize) -> NodeAxis {
		match lower.cmp(&upper) {
			Ordering::Less => NodeAxis::Interval { lower, upper },
			Ordering::Equal => NodeAxis::Known { val: lower },
			Ordering::Greater => panic!(
				"NodeAxis::Interval cannot be constructed with lower({}) > upper({})",
				lower, upper
			),
		}
	}

	pub fn is_known(&self) -> bool {
		match self {
			NodeAxis::Known { .. } => true,
			NodeAxis::Interval { .. } => false,
		}
	}

	pub fn as_known(&self) -> Option<usize> {
		match self {
			NodeAxis::Known { val } => Some(*val),
			NodeAxis::Interval { .. } => None,
		}
	}

	pub fn as_interval(&self) -> (usize, usize) {
		match self {
			NodeAxis::Known { val } => (*val, *val),
			NodeAxis::Interval { lower, upper } => (*lower, *upper),
		}
	}

	pub fn lower(&self) -> usize {
		match self {
			NodeAxis::Known { val } => *val,
			NodeAxis::Interval { lower, .. } => *lower,
		}
	}

	pub fn upper(&self) -> usize {
		match self {
			NodeAxis::Known { val } => *val,
			NodeAxis::Interval { upper, .. } => *upper,
		}
	}
}

impl<'a> From<&'a NodeAxis> for NodeAxis {
	fn from(s: &NodeAxis) -> NodeAxis {
		s.clone()
	}
}

impl From<usize> for NodeAxis {
	fn from(s: usize) -> NodeAxis {
		NodeAxis::known(s)
	}
}

impl<'a> From<&'a usize> for NodeAxis {
	fn from(s: &usize) -> NodeAxis {
		NodeAxis::known(*s)
	}
}

impl From<isize> for NodeAxis {
	fn from(s: isize) -> NodeAxis {
		if s < 0 {
			NodeAxis::unknown()
		} else {
			NodeAxis::known(s as usize)
		}
	}
}

impl<'a> From<&'a isize> for NodeAxis {
	fn from(s: &isize) -> NodeAxis {
		if *s < 0 {
			NodeAxis::unknown()
		} else {
			NodeAxis::known(*s as usize)
		}
	}
}

impl From<u32> for NodeAxis {
	fn from(s: u32) -> NodeAxis {
		NodeAxis::known(s as usize)
	}
}

impl<'a> From<&'a u32> for NodeAxis {
	fn from(s: &u32) -> NodeAxis {
		NodeAxis::known(*s as usize)
	}
}

impl From<i32> for NodeAxis {
	fn from(s: i32) -> NodeAxis {
		if s < 0 {
			NodeAxis::unknown()
		} else {
			NodeAxis::known(s as usize)
		}
	}
}

impl<'a> From<&'a i32> for NodeAxis {
	fn from(s: &i32) -> NodeAxis {
		if *s < 0 {
			NodeAxis::unknown()
		} else {
			NodeAxis::known(*s as usize)
		}
	}
}

impl From<u64> for NodeAxis {
	fn from(s: u64) -> NodeAxis {
		NodeAxis::known(s as usize)
	}
}

impl<'a> From<&'a u64> for NodeAxis {
	fn from(s: &u64) -> NodeAxis {
		NodeAxis::known(*s as usize)
	}
}

impl From<i64> for NodeAxis {
	fn from(s: i64) -> NodeAxis {
		if s < 0 {
			NodeAxis::unknown()
		} else {
			NodeAxis::known(s as usize)
		}
	}
}

impl<'a> From<&'a i64> for NodeAxis {
	fn from(s: &i64) -> NodeAxis {
		if *s < 0 {
			NodeAxis::unknown()
		} else {
			NodeAxis::known(*s as usize)
		}
	}
}

impl From<(usize, usize)> for NodeAxis {
	fn from((lower, upper): (usize, usize)) -> NodeAxis {
		NodeAxis::Interval { lower, upper }
	}
}

impl<'a> From<(&'a usize, &'a usize)> for NodeAxis {
	fn from((lower, upper): (&usize, &usize)) -> NodeAxis {
		NodeAxis::Interval {
			lower: *lower,
			upper: *upper,
		}
	}
}

impl NodeAxis {
	pub fn add(&self, other: &NodeAxis) -> NodeAxis {
		match (self, other) {
			(NodeAxis::Known { val: x }, NodeAxis::Known { val: y }) => NodeAxis::Known {
				val: y.checked_add(*x).expect("NodeAxis add overflowed"),
			},
			(NodeAxis::Known { val: x }, NodeAxis::Interval { upper, lower })
			| (NodeAxis::Interval { upper, lower }, NodeAxis::Known { val: x }) => NodeAxis::interval(
				lower.checked_add(*x).expect("NodeAxis add overflowed"),
				upper.saturating_add(*x),
			),
			(
				NodeAxis::Interval {
					upper: upper1,
					lower: lower1,
				},
				NodeAxis::Interval {
					upper: upper2,
					lower: lower2,
				},
			) => NodeAxis::interval(
				lower1.checked_add(*lower2).expect("NodeAxis add overflowed"),
				upper1.saturating_add(*upper2),
			),
		}
	}

	pub fn multiply(&self, other: &NodeAxis) -> NodeAxis {
		match (self, other) {
			(NodeAxis::Known { val: x }, NodeAxis::Known { val: y }) => NodeAxis::Known {
				val: y.checked_mul(*x).expect("NodeAxis multiply overflowed"),
			},
			(NodeAxis::Known { val: x }, NodeAxis::Interval { upper, lower })
			| (NodeAxis::Interval { upper, lower }, NodeAxis::Known { val: x }) => NodeAxis::interval(
				lower.checked_mul(*x).expect("NodeAxis multiply overflowed"),
				upper.saturating_mul(*x),
			),
			(
				NodeAxis::Interval {
					upper: upper1,
					lower: lower1,
				},
				NodeAxis::Interval {
					upper: upper2,
					lower: lower2,
				},
			) => NodeAxis::interval(
				lower1.checked_mul(*lower2).expect("NodeAxis multiply overflowed"),
				upper1.saturating_mul(*upper2),
			),
		}
	}

	/// Broadcast the constraints of other
	pub fn broadcast_merge(&self, other: &NodeAxis) -> Result<NodeAxis, DimError> {
		if let NodeAxis::Known { val: 1 } = other {
			self.merge(&NodeAxis::interval(1, usize::MAX))
		} else {
			self.merge(other)
		}
	}

	pub fn merge(&self, other: &NodeAxis) -> Result<NodeAxis, DimError> {
		match (self, other) {
			(NodeAxis::Known { val: x }, NodeAxis::Known { val: y }) => {
				if x == y {
					Ok(NodeAxis::known(*x))
				} else {
					Err(DimError::IncompatibleDimension {
						dim1: self.clone(),
						dim2: other.clone(),
					})
				}
			},
			(NodeAxis::Known { val }, NodeAxis::Interval { upper, lower })
			| (NodeAxis::Interval { upper, lower }, NodeAxis::Known { val }) => {
				if val >= lower && val <= upper {
					Ok(NodeAxis::known(*val))
				} else {
					Err(DimError::IncompatibleDimension {
						dim1: self.clone(),
						dim2: other.clone(),
					})
				}
			},
			(
				NodeAxis::Interval {
					upper: upper1,
					lower: lower1,
				},
				NodeAxis::Interval {
					upper: upper2,
					lower: lower2,
				},
			) => {
				let upper = *cmp::min(upper1, upper2);
				let lower = *cmp::max(lower1, lower2);
				Ok(NodeAxis::interval(lower, upper))
			},
		}
	}
}

impl Display for NodeAxis {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> ::std::fmt::Result {
		match self {
			NodeAxis::Known { val } => fmt.pad(&format!("{}", val)),
			NodeAxis::Interval { lower, upper } if *lower == 0 && *upper == ::std::usize::MAX => fmt.pad("-1"),
			NodeAxis::Interval { lower, upper } if *upper == ::std::usize::MAX => {
				fmt.pad(&format!("({} — MAX)", lower))
			},
			NodeAxis::Interval { lower, upper } => fmt.pad(&format!("({} — {})", lower, upper)),
		}
	}
}

/// Represents the range of shapes a tensor can take on.
#[derive(Clone, PartialEq, Eq)]
pub struct NodeShape {
	dimensions: SmallVec<[NodeAxis; 4]>,
}

impl Display for NodeShape {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> ::std::fmt::Result {
		let mut string = String::new();
		let mut iter = self.dimensions.iter();
		write!(string, "[")?;
		if let Some(axis) = iter.next() {
			write!(string, "{:>4}", axis)?;
			for axis in iter {
				write!(string, ",{:>4}", axis)?;
			}
		}
		write!(string, "]")?;

		fmt.pad(&string)
	}
}

impl Debug for NodeShape {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> ::std::fmt::Result {
		Display::fmt(self, fmt)
	}
}

impl<'a> IntoIterator for &'a NodeShape {
	type IntoIter = ::std::slice::Iter<'a, NodeAxis>;
	type Item = &'a NodeAxis;

	fn into_iter(self) -> Self::IntoIter {
		self.dimensions.iter()
	}
}

impl<T: Into<NodeAxis>, I: IntoIterator<Item = T>> From<I> for NodeShape {
	fn from(i: I) -> NodeShape {
		let dimensions: SmallVec<[NodeAxis; 4]> = i
			.into_iter()
			.map(Into::into)
			.inspect(|dim| match dim {
				NodeAxis::Interval { lower, upper } if upper < lower => {
					panic!(
						"NodeAxis::Interval cannot be constructed with lower({}) > upper({})",
						lower, upper
					);
				},
				_ => {},
			})
			.collect();
		NodeShape { dimensions }
	}
}

impl NodeShape {
	/// Return the number of axes in the shape
	pub fn len(&self) -> usize {
		self.dimensions.len()
	}

	pub fn is_empty(&self) -> bool {
		self.dimensions.is_empty()
	}

	pub fn slice(&self) -> &[NodeAxis] {
		&self.dimensions
	}

	pub fn slice_mut(&mut self) -> &mut [NodeAxis] {
		&mut self.dimensions
	}

	pub fn iter(&self) -> impl Iterator<Item = &NodeAxis> + DoubleEndedIterator {
		self.dimensions.iter()
	}

	pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut NodeAxis> + DoubleEndedIterator {
		self.dimensions.iter_mut()
	}

	// Should be called and only called by operations prior to propagating shape constraints
	/// The NodeAxis::Interval are collapsed to the lower bound
	pub fn collapse_dimensions_to_minimum(&mut self) {
		for i in 0..self.dimensions.len() {
			match self.dimensions[i] {
				NodeAxis::Known { .. } => {},
				NodeAxis::Interval { lower, .. } => self.dimensions[i] = NodeAxis::known(lower),
			};
		}
	}

	/// Returns a copy of the shape with all dimensions that arent Known(_) set to 1.
	pub fn collapse_to_broadcastable_dimension(&self) -> NodeShape {
		let mut shape = self.clone();

		for i in 0..shape.dimensions.len() {
			match shape.dimensions[i] {
				NodeAxis::Known { .. } => {},
				NodeAxis::Interval { lower, upper } if lower == upper => shape.dimensions[i] = NodeAxis::known(lower),
				_ => shape.dimensions[i] = NodeAxis::known(1),
			};
		}

		shape
	}

	pub fn flat_size(&self) -> NodeAxis {
		self.dimensions.iter().fold(1.into(), |prev, item| prev.multiply(item))
	}

	/// Return the product of known NodeAxis, or an error if any non-Known axis is present.
	///
	/// #Errors
	/// Returns `ShapeError::UnderDeterminedDimensions` if not all `NodeAxis` are known.
	///
	/// # Panics
	/// Panics on overflow.
	pub fn known_flat_size(&self) -> Result<usize, ShapeError> {
		let mut size = 1usize;

		for dim in self.dimensions.iter() {
			match dim {
				NodeAxis::Known { val } => size = size.checked_mul(*val).expect("NodeAxis multiply overflowed"),
				_ => return Err(ShapeError::UnderDeterminedDimensions { shape: self.clone() }),
			}
		}

		Ok(size)
	}

	/// If all dimension values are `Known` this will
	/// This should generally only be called after `collapse_dimensions_to_minimum`
	pub fn to_data_shape(&self) -> Result<IxDyn, ShapeError> {
		let mut dims = Vec::with_capacity(self.dimensions.len());

		// dims.push(n);
		for dim in self.dimensions.iter() {
			match dim {
				NodeAxis::Known { val } => dims.push(*val),
				_ => return Err(ShapeError::UnderDeterminedDimensions { shape: self.clone() }),
			}
		}
		Ok(IxDyn(&dims))
	}

	/// Returns true if all NodeAxes have a Known size.
	pub fn is_known(&self) -> bool {
		self.dimensions.iter().all(|x| matches!(x, NodeAxis::Known { .. }))
	}

	pub fn ndim(&self) -> usize {
		self.dimensions.len()
	}

	pub fn merge(&self, other: &NodeShape) -> Result<NodeShape, ShapeError> {
		if self.ndim() != other.ndim() {
			return Err(ShapeError::IncompatibleRanks {
				shape1: self.clone(),
				shape2: other.clone(),
			});
		}

		let mut vec = SmallVec::new();
		for (i, (s, o)) in self.dimensions.iter().zip(&other.dimensions).enumerate() {
			match s.merge(o) {
				Ok(x) => vec.push(x),
				Err(x) => {
					return Err(ShapeError::IncompatibleDimensionAt {
						shape1: self.clone(),
						shape2: other.clone(),
						index: i,
						cause: x,
					});
				},
			}
		}
		Ok(NodeShape { dimensions: vec })
	}

	/// Broadcast
	pub fn broadcast_merge(&self, other: &NodeShape) -> Result<NodeShape, ShapeError> {
		if self.ndim() < other.ndim() {
			return Err(ShapeError::IncompatibleRanks {
				shape1: self.clone(),
				shape2: other.clone(),
			});
		}

		let mut vec = SmallVec::new();
		let unit_dim = 1usize.into();
		let iter = self
			.dimensions
			.iter()
			.rev()
			.zip(other.dimensions.iter().rev().chain(::std::iter::repeat(&unit_dim)));

		for (i, (s, o)) in iter.enumerate() {
			match s.broadcast_merge(o) {
				Ok(x) => vec.push(x),
				Err(x) => {
					return Err(ShapeError::IncompatibleDimensionAt {
						shape1: self.clone(),
						shape2: other.clone(),
						index: i,
						cause: x,
					});
				},
			}
		}
		vec.reverse();
		Ok(NodeShape { dimensions: vec })
	}
}
