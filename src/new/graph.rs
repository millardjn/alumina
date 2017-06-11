
use std::collections::{HashMap, HashSet};
use ndarray::ArrayD;
use ndarray::prelude::*;
use smallvec::SmallVec;
use ops::Operation;
use std::cmp;
use std::iter::repeat;
use std::iter::FromIterator;
use std::sync::Arc;
use ndarray;

enum NodeType {

	ParameterNode(ParameterType),

	VariableNode,

	XVariableNode,

	ViewNode(NodeID),

	ReifiedViewNode,
}

enum ParameterType {
	Locked(ArrayD<f32>),
	Init(Arc<Fn(&mut [f32])>),
}


#[derive(Clone)]
pub struct NodeID {
	index: usize,
	shape: NodeShape,
}

impl NodeID {


	fn value_id(&self) -> DataID {
		DataID{index: self.index * 2, shape: self.shape.clone()}
	}

	fn gradient_id(&self) -> DataID {
		DataID{index: self.index * 2 + 1, shape: self.shape.clone()}
	}
}

struct DataID {
	index: usize,
	shape: NodeShape,
}

impl DataID {
	fn node_id(&self) -> NodeID {
		NodeID{index: self.index / 2, shape: self.shape.clone()}
	}
}


pub struct OpID {
	index: usize,
}

impl OpID {
	fn new (index: usize) -> OpID {
		OpID{index}
	}

	fn value_id(&self) -> PassID {
		PassID{index: self.index * 2}
	}

	fn gradient_id(&self) -> PassID {
		PassID{index: self.index * 2 + 1}
	}
}

struct PassID {
	index: usize,
}

impl PassID {
	fn op_id(&self) -> OpID {
		OpID{index: self.index / 2}
	}
}


// 	data: Vec<ArrayD<f32>>




/// A structure which allows for runtime checked borrowing, similar to a RefCell for a Collection of Arrays,
/// but with some limitations.
/// Each element can only be borrowed either once mutably or many times immutably, however,
/// once borrowed as such it is stuck until DataBorrow is dropped, or by calling reset_all().
struct DataBorrow <'a> {
	data: &'a mut [ArrayD<f32>],
	borrow_flags: Vec<usize>,
}

const UNUSED: usize = 0;
const WRITING: usize = !0;
impl<'a> DataBorrow <'a> {


	fn new(slice: &mut [ArrayD<f32>]) -> DataBorrow {
		let len = slice.len();
		DataBorrow{
			data: slice,
			borrow_flags: vec![UNUSED; len],
		}
	}

	/// Forces return of all borrows, and resets runtime checks, allowing new borrowing patterns
	fn reset_all(mut self) -> Self{
		for e in &mut self.borrow_flags{
			*e = UNUSED;
		}
		self
	}

	/// Immutably borrows data element associated with the given ID
	/// Will panic if data element is already borrowed mutably
	fn get(&mut self, id: DataID) -> ArrayViewD<f32> {
		if self.borrow_flags[id.index] != WRITING {
				self.borrow_flags[id.index] += 1;
				let ptr = &self.data[id.index] as *const ArrayD<f32>;
				unsafe{(&*ptr).view()}
		} else {
			panic!("already mutably borrowed")
		}
	}

	/// Mutably borrows data element associated with the given ID
	/// Will panic if data element is already mutably or immutably borrowed 
	/// The borrow will stick until
	fn get_mut(&mut self, id: DataID) -> ArrayViewMutD<f32> {
		match self.borrow_flags[id.index] {
			UNUSED => {
				self.borrow_flags[id.index] = WRITING;
				let ptr = &mut self.data[id.index] as *const ArrayD<f32> as *mut ArrayD<f32>;
				unsafe{(&mut *ptr).view_mut()}
			},
			WRITING => panic!("already mutably borrowed"),
			_ => panic!("already immutably borrowed"),
		}
	}
}

#[derive(PartialEq, Eq, Hash)]
pub enum NodeTag{
	Id(usize),
	Int(usize),
	Str(String),
}

impl From<NodeID> for NodeTag{
	fn from(i: NodeID) -> NodeTag {
		NodeTag::Id(i.index)
	}
}

impl From<usize> for NodeTag{
	fn from(i: usize) -> NodeTag {
		NodeTag::Int(i)
	}
}

impl<'a> From<&'a str> for NodeTag{
	fn from(i: &str) -> NodeTag {
		NodeTag::Str(i.to_string())
	}
}

#[derive(PartialEq, Eq, Hash)]
pub enum OpTag{
	Id(usize),
	Int(usize),
	Str(String),
}

impl From<OpID> for OpTag{
	fn from(i: OpID) -> OpTag {
		OpTag::Int(i.index)
	}
}

impl From<usize> for OpTag{
	fn from(i: usize) -> OpTag {
		OpTag::Int(i)
	}
}

impl<'a> From<&'a str> for OpTag{
	fn from(i: &str) -> OpTag {
		OpTag::Str(i.to_string())
	}
}

pub struct GraphBuilder {

	node_type: Vec<NodeType>,
	nodes: Vec<NodeID>,
	operations: Vec<Box<Operation>>,

	node_tags: HashMap<NodeTag, HashSet<usize>>, // tag, index
	operation_tags: HashMap<OpTag, usize> // tag, index
}

impl GraphBuilder {
	fn tag<T: Into<NodeTag>>(&mut self, node: NodeID, tag: T){
			let tag = tag.into();
			match tag {
				NodeTag::Id(_) => {},
				NodeTag::Int(_) | NodeTag::Str(_) => {
					let set = self.node_tags.entry(tag).or_insert(HashSet::new());
					set.insert(node.index);
				}
			}	
	}
	
	fn new_node<I: Into<NodeTag>, T: Into<Option<I>>>(&mut self, shape: NodeShape, tag: T, nodetype: NodeType) -> NodeID{
		let id = NodeID{index: self.nodes.len(), shape: shape};
		self.nodes.push(id.clone());
		self.node_type.push(nodetype);
		
		if let Some(into_tag) = tag.into() {
			self.tag(id.clone(), into_tag);
		}

		id
	}

	pub fn new_variable<S: Into<NodeShape>, I: Into<NodeTag>, T: Into<Option<I>>>(&mut self, shape: S, tag: T) -> NodeID {
		self.new_node(shape.into(), tag, NodeType::VariableNode)
	}

	/// Creates a node which 
	pub fn new_parameter<S: Into<NodeShape>, I: Into<NodeTag>, T: Into<Option<I>>>(&mut self, shape: S, init: Arc<Fn(&mut [f32])>, tag: T) -> NodeID {
		let shape = shape.into();
		assert!(shape.is_fixed(), "Parameter nodes must be of a fixed shape");
		self.new_node(shape, tag, NodeType::ParameterNode(ParameterType::Init(init)))
	}

	pub fn new_locked_parameter<S: Into<NodeShape>, I: Into<NodeTag>, T: Into<Option<I>>>(&mut self, data: ArrayD<f32>, tag: T) -> NodeID {

		self.new_node(data.shape().into(), tag, NodeType::ParameterNode(ParameterType::Locked(data)))
	}

	/// Create a node which acts as a subview of another node.
	/// Contiguous views will be free from time and memory overhead.
	/// Non-contigues views will incurr a memory and time overhead during runtime.
	/// Returns NodeID of the new node.
	pub fn new_view<S: Into<NodeShape>, T: Into<NodeTag>>(&mut self, shape: S, tag: T) -> NodeID {
		unimplemented!()
	}

	// Returns the NodeID which was tagged with 'tag'. returns none if zero or more than one NodeIDs are associated with the tag.
	pub fn get_node<T: Into<NodeTag>>(&self, tag: T) -> Option<NodeID> {
		let tag = tag.into();
		match tag {
			NodeTag::Id(ind) => Some(self.nodes[ind].clone()),
			NodeTag::Int(_) | NodeTag::Str(_) => {
				self.node_tags.get(&tag).and_then(|set| if set.len() == 1 {Some(self.nodes[*set.iter().next().unwrap()].clone())} else {None})
			}
		}
	}

	fn get_nodes<T: Into<NodeTag>>(&self, tag: T) {
		unimplemented!()
	}

	fn get_parameter<T: Into<OpTag>>(&self, tag: T) -> NodeID {
		unimplemented!()
	}

	fn get_op<T: Into<OpTag>>(&self, tag: T) -> &Operation {
		unimplemented!()
	}

	fn get_ops<T: Into<OpTag>>(&self, tag: T) -> &Operation {
		unimplemented!()
	}



}


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



use self::NodeDim::*;
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

	fn merge(&self, other: &NodeDim) -> Result<NodeDim, ShapeError>{
		match (self, other) {
			(&Unknown, x) | (x, &Unknown) => Ok(x.clone()),	
			(&Fixed(x), &Fixed(y)) => if x == y {Ok(Fixed(x))} else {Err(ShapeError::MergeIncompatibleHigherDimension)},
			(&Fixed(v), &Range{upper, lower}) | (&Range{upper, lower}, &Fixed(v)) => if v >= lower && v <= upper {Ok(Fixed(v))} else {Err(ShapeError::MergeIncompatibleHigherDimension)},
			(&Range{upper: upper1, lower: lower1}, &Range{upper: upper2, lower: lower2}) =>  {
				let upper = cmp::min(upper1, upper2);
				let lower = cmp::max(lower1, lower2);
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
	pub fn collapse_dimensions_to_minimum(&mut self) -> Result<(), ShapeError>{
		
		for i in 0.. self.dimensions.len(){
			match &self.dimensions[i] {
				&Unknown => return Err(ShapeError::IncompleteNodeShape),
				&Fixed(_) => {},
				&Range{lower, ..} => self.dimensions[i] = Fixed(lower),
			};
			
		}
		Ok(())
	}
	
	pub fn flat_size(&self) -> NodeDim {
		self.dimensions.iter().fold(1.into(), |prev, item| prev.multiply(item))
	}
	
	pub fn force_flat_size(&self) -> Result<usize, ShapeError>{
		let mut size = 1;

		for dim in self.dimensions.iter() {
			match dim {
				&Fixed(v) => size *= v,
				&Range{upper, lower} if upper == lower => size *= lower,
				_ => return Err(ShapeError::UnderDeterminedFlatSize),
			}
		}

		Ok(size)
	}
	
	/// If range upper != lower, lowe will be used.
	pub fn to_data_shape(&self, n: usize) -> Result<IxDyn, ShapeError> {
		let mut dims: SmallVec<[usize; 6]> = SmallVec::new();
		dims.push(n);
		for dim in self.spatial_dimensions().iter() {
			match dim {
				 &Fixed(v) => dims.push(v),
				_ => return Err(ShapeError::IncompleteNodeShape),
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
	
	pub fn merge(&self, other: &NodeShape) -> Result<NodeShape, ShapeError>{
		
		if self.is_fixed() && other.is_fixed() {
			self.merge_fixed(other)	
		} else if self.channels() != other.channels() {
			Err(ShapeError::MergeIncompatibleChannelDimension)
		} else if self.ndim() != other.ndim() {
			Err(ShapeError::MergeIncompatibleRank)
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
	fn merge_fixed(&self, other: &NodeShape) -> Result<NodeShape, ShapeError>{

		match (self.flat_size(), other.flat_size()){
			(Fixed(x), Fixed(y)) if x == y => {},
			_ => return Err(ShapeError::MergeIncompatibleFlatSize),
		}

		if self.ndim() == 1 {
			Ok(other.clone())
		} else if other.ndim() == 1 {
			Ok(self.clone())
		} else if self.ndim() != other.ndim() {
			Err(ShapeError::MergeIncompatibleRank)
		} else if self.channels() != other.channels() {	
			Err(ShapeError::MergeIncompatibleChannelDimension)
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
