use ndarray::ArrayD;
use ndarray::prelude::*;
use std::cell::Cell;
use std::mem;
use indexmap::IndexMap;
use std::any::Any;

use id::*;
use graph::{Dependencies, DataStatus, ErrorKind, Result};

enum DataState<T>{
	Unallocated,
	UnallocatedStaticInput,
	Allocated(T),
	Deallocated,
}

/// This type allows a `Pass` to access the values and gradients of nodes at execution time.
///
/// To achieve safe mutable access to multiple nodes this structure uses runtime checked borrowing,
/// similar to a RefCell for a Collection of Arrays, but with some limitations.
/// Each element can only be borrowed either once mutably or many times immutably, however, borrows are not reset until the end of the Pass
pub struct Storage<'a> {
	shapes: &'a IndexMap<NodeID, IxDyn>,
	static_inputs: &'a IndexMap<DataID, ArrayD<f32>>,
	dependencies: &'a Dependencies,

	loss: Cell<f32>,
	data: IndexMap<DataID, DataState<ArrayD<f32>>>,
	borrow_flags: IndexMap<DataID, Cell<usize>>,
	current_pass: Option<PassID>,
	pass_data: IndexMap<PassID, Box<Any>>,
}

const UNUSED: usize = 0;
const WRITING: usize = !0;
impl<'a> Storage<'a> {

	pub (crate) fn new(included_data: &IndexMap<DataID, DataStatus>, dependencies: &'a Dependencies, static_inputs: &'a IndexMap<DataID, ArrayD<f32>>, input_data: IndexMap<DataID, ArrayD<f32>>, shapes: &'a IndexMap<NodeID, IxDyn>) -> Storage<'a> { //, graph: &'a GraphDef

		// let num_nodes = dependencies.node_inputs().len();
		// let num_data = dependencies.data_inputs().len();
		// let num_passes = dependencies.pass_inputs().len();
		
		//debug_assert_eq!(num_nodes, shapes.len());
		//debug_assert_eq!(num_data, included_data.len());

		let mut data: IndexMap<DataID, DataState<ArrayD<f32>>> = included_data.iter().map(|(id, _state)| (id.clone(), DataState::Unallocated)).collect();
		let borrow_flags = included_data.iter().map(|(id, _state)| (id.clone(), Cell::new(UNUSED))).collect();

		for (data_id, input_data) in input_data.into_iter() {
			debug_assert!(shapes.get(&data_id.node_id()).unwrap().slice() == input_data.shape());
			data.insert(data_id.clone(), DataState::Allocated(input_data));
		}

		for (data_id, _data) in static_inputs.iter() {
			debug_assert!(!matches!(data.get(data_id).unwrap(), &DataState::Allocated(_))); // static inputs should have already been filtered to not collide with inputs
			data.insert(data_id.clone(), DataState::UnallocatedStaticInput);
		}

		Storage{
			shapes: shapes,
			static_inputs,
			dependencies,

			loss: Cell::new(0.0),
			data: data,
			borrow_flags: borrow_flags,
			current_pass: None,
			pass_data: indexmap![],
		}
	}

	pub (crate) fn set_pass_data(&mut self, pass_id: &PassID, pass_data: Box<Any>){
		self.pass_data.insert(pass_id.clone(), pass_data);
	}

	pub fn get_pass_data(&self, pass_id: &PassID) -> Option<&Any>{
		self.pass_data.get(pass_id).map(|x| &**x)
	}

	/// If this value is not `None`, all subsequent accesses will be checked against the dependency list for the Pass.
	/// This can be useful to ensure that passes dont access anything they havent listed as and input or output.
	pub (crate) fn set_current_pass(&mut self, pass_id: Option<PassID>){
		self.current_pass = pass_id;
	}

	pub fn get_current_pass(&self) -> &Option<PassID>{
		&self.current_pass
	}

	/// Deallocates the data specified by DataID.
	pub (crate) fn deallocate(&mut self, data_id: &DataID){
		mem::replace(self.data.get_mut(data_id).unwrap(), DataState::Deallocated);
	}

	/// This resets runtime borrow checks, allowing for a new round of borrowing patterns.
	/// By taking `self` this forces return of all prior borrows.
	pub fn clear_borrow_flags(mut self) -> Self{
		for (_id, e) in self.borrow_flags.iter_mut(){
			e.set(UNUSED);
		}
		self
	}	
		
	/// Should never be called if a &mut borrow could possibly already exist.
	unsafe fn get_or_init(&self, id: &DataID) -> Result<*mut ArrayD<f32>>{
		
		let ptr = if let Some(reference) = self.data.get(id) {
			reference as *const _ as *mut _
		} else {
			bail!(ErrorKind::StorageDataMarkedNotRequired)
		};

		match *ptr {
			DataState::Deallocated => bail!(ErrorKind::StorageDataDeallocated),
			DataState::Unallocated => {
				*ptr = DataState::Allocated(ArrayD::zeros(self.shapes.get(&id.node_id()).unwrap().clone()));
			},
			// DataState::UnallocatedInput(ind) =>{
			// 	*ptr = DataState::Allocated(self.input_data[ind].clone())
			// },
			DataState::UnallocatedStaticInput => {
				let shape = self.shapes.get(&id.node_id()).unwrap().clone();
				if let Some(ref static_data) = self.static_inputs.get(id){
					if let Some(broadcasted_view) = static_data.broadcast(shape){
						*ptr = DataState::Allocated(broadcasted_view.to_owned())
					} else {
						bail!(ErrorKind::StaticInputBroadcastFailure(id.node_id(), static_data.shape().to_owned(), self.shapes.get(&id.node_id()).unwrap().slice().to_owned()))
					}
				} else {
					unreachable!();
				}
			},
			DataState::Allocated(_) => {},
		}

		// return pointer to allocated data
		if let DataState::Allocated(ref mut data) = *ptr{
			Ok(data as *mut ArrayD<f32>)
		} else {
			unreachable!()
		}
	}

	/// Access the loss variable.
	pub fn loss(&self) -> f32 {
		self.loss.get()
	}

	/// Access the loss variable.
	/// Loss should only be added to in the backwards passes of ops.
	pub fn loss_add(&self, additional_loss: f32){
		unsafe{*self.loss.as_ptr() += additional_loss;}
	}

	/// Immutably borrows data element associated with the given ID.
	/// 
	/// A Pass may only borrow data which is listed as a input or output dependency.
	/// Will panic if data element is already mutably borrowed.
	/// The borrow will stick until `clear_borrow_flags()` is called.
	pub fn get<'b>(&'b self, data_id: &DataID) -> Result<ArrayViewD<f32>> {
		if let Some(ref pass_id) = self.current_pass {
			ensure!(self.dependencies.pass_inputs(pass_id).contains(data_id)||self.dependencies.pass_outputs(pass_id).contains(data_id), ErrorKind::StorageImmutableBorrowError(pass_id.name(), data_id.name()));
		}

		let flag = if let Some(reference) = self.borrow_flags.get(data_id) {
			reference
		} else {
			bail!(ErrorKind::StorageDataMarkedNotRequired)
		};
		if flag.get() != WRITING {
				let ptr = unsafe{self.get_or_init(data_id)?};
				flag.set(flag.get() + 1);
				let array: &'b ArrayD<f32> = unsafe{&*ptr};
				Ok(array.view())
		} else {
			bail!(ErrorKind::StorageDataAlreadyMutablyBorrowed)
		}
	}

	/// Mutably borrows data element associated with the given ID.
	/// Will panic if data element is already mutably or immutably borrowed.
	/// The borrow will stick until `clear_borrow_flags()` is called.
	pub fn get_mut<'b>(&'b self, data_id: &DataID) -> Result<ArrayViewMutD<f32>> {
		if let Some(ref pass_id) = self.current_pass {
			ensure!(self.dependencies.pass_outputs(pass_id).contains(data_id), ErrorKind::StorageMutableBorrowError(pass_id.name(), data_id.name()));
		}

		let flag = if let Some(reference) = self.borrow_flags.get(data_id) {
			reference
		} else {
			bail!(ErrorKind::StorageDataMarkedNotRequired)
		};
		match flag.get() {
			UNUSED => {
				let ptr = unsafe{self.get_or_init(data_id)?};
				flag.set(WRITING);
				let array: &'b mut ArrayD<f32> = unsafe{&mut *ptr};
				Ok(array.view_mut())
			},
			WRITING => bail!(ErrorKind::StorageDataAlreadyMutablyBorrowed),
			_ => bail!(ErrorKind::StorageDataAlreadyBorrowed),
		}
	}

	/// Returns true if a `DataID` is a required component of the subgraph.
	///
	/// Checking this is only required for the outputs of a pass, and only if a pass has multiple outputs.
	/// If false, no attempt should be made to write to that data_id using 'get_mut()'.
	pub fn is_required(&self, data_id: &DataID) -> bool {
		//!matches!(self.data[data_id.index], DataState::NotRequired) //TODO this doesnt perfectly match the required_data vector from graph
		self.data.contains_key(data_id)
	}

	/// Consume the Storage and converts it into a IndexMap.
	///
	/// Intended for use after storage is returned from `execute()`.
	pub fn into_map(self) -> IndexMap<DataID, ArrayD<f32>> {
		self.data.into_iter().filter_map(|(id, entry)|{
			match entry {
				DataState::Allocated(arr) => Some((id, arr)),
				_ => None,
			}
		}).collect()
	}
}