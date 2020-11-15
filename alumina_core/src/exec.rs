//! Types and tools for executing a graph.
use crate::{
	errors::ExecError,
	graph::{Node, NodeID, Op, OpID},
	//shape_prop::cached_shapes_inner,
	shape_prop::cached_shapes_inner,
	subgraph::{execution_subgraph, SubGraph},
};
use indexmap::{IndexMap, IndexSet};
use lru::LruCache;
use ndarray::{ArcArray, ArrayD, ArrayViewD, ArrayViewMutD, Dimension, IxDyn};
use std::{
	borrow::Borrow,
	cell::{RefCell, UnsafeCell},
	hash::Hash,
};
use std::time::{Instant};
use sysinfo::{ProcessorExt, SystemExt, RefreshKind};


enum DataState<T> {
	Unallocated {
		writers_remaining: usize,
		readers_remaining: usize,
	},
	Writable {
		writers_remaining: usize,
		readers_remaining: usize,
		data: ArcArray<T, IxDyn>,
	}, // Not all inputs have been input
	Readable {
		readers_remaining: usize,
		data: ArcArray<T, IxDyn>,
	}, // All input ops have now run
	Deallocated,

	Input {
		data: ArcArray<T, IxDyn>,
		readers_remaining: usize,
	},
	/// This variant marks an input that required broadcasting. It exists to delay broadcasting until required.
	BroadcastInput {
		data: ArcArray<T, IxDyn>,
		readers_remaining: usize,
	},
}

impl<T> DataState<T> {
	fn readers_remaining(&self) -> usize {
		match self {
			DataState::Unallocated { readers_remaining, .. }
			| DataState::Writable { readers_remaining, .. }
			| DataState::Readable { readers_remaining, .. }
			| DataState::Input { readers_remaining, .. }
			| DataState::BroadcastInput { readers_remaining, .. } => *readers_remaining,
			DataState::Deallocated => 0,
		}
	}

	fn writers_remaining(&self) -> usize {
		match self {
			DataState::Unallocated { writers_remaining, .. } | DataState::Writable { writers_remaining, .. } => {
				*writers_remaining
			}
			DataState::Readable { .. }
			| DataState::Deallocated
			| DataState::Input { .. }
			| DataState::BroadcastInput { .. } => 0,
		}
	}

	fn deallocatable(&self) -> bool {
		match self {
			DataState::Unallocated {
				readers_remaining,
				writers_remaining,
			} => *readers_remaining == 0 && *writers_remaining == 0,
			DataState::Writable {
				readers_remaining,
				writers_remaining,
				..
			} => *readers_remaining == 0 && *writers_remaining == 0,
			DataState::Readable { readers_remaining, .. } => *readers_remaining == 0,
			DataState::Deallocated => false,

			DataState::Input { readers_remaining, .. } => *readers_remaining == 0,

			DataState::BroadcastInput { readers_remaining, .. } => *readers_remaining == 0,
		}
	}
}

/// Provides access to input and output values during Op execution.
///
/// Primary responsibility is to provide a safe interface to `OpInstance` implementations which prevents multiple
/// mutable borrows.
pub struct ExecutionContext {
	value_map: UnsafeCell<IndexMap<Node, DataState<f32>>>,
	shape_map: IndexMap<NodeID, IxDyn>,

	/// None = not borrowed
	/// false = shared borrow
	/// true = mutable borrow
	borrows: UnsafeCell<IndexMap<NodeID, bool>>,

	current_op: Option<Op>,
	current_inputs: IndexSet<Node>,
	current_outputs: IndexSet<Node>,
}

impl ExecutionContext {
	fn new(value_map: IndexMap<Node, DataState<f32>>, shape_map: IndexMap<NodeID, IxDyn>) -> ExecutionContext {
		ExecutionContext {
			value_map: UnsafeCell::new(value_map),
			shape_map,

			/// None = not borrowed
			/// false = shared borrow
			/// true = mutable borrow
			borrows: UnsafeCell::new(IndexMap::with_capacity(8)),

			current_op: None,
			current_inputs: IndexSet::new(),
			current_outputs: IndexSet::new(),
		}
	}

	/// Returns the `OpInner` to an `OpInstance` inside its `execute()` method
	pub fn current_op(&self) -> &Op {
		self.current_op
			.as_ref()
			.expect("Alumina Bug: current_op() called without op set")
	}

	/// Must be called with an `Op` before calling execute on the `OpInstance`.
	///
	/// Clears borrows.
	/// Sets current_op for execute call.
	/// Pre flight checks for inputs and outputs.
	///
	/// returns self and whether the op should be skipped
	///
	///  * Errors if an output is not either writable or unallocated with writers remaining > 0 (SubGraphNotExecutable)
	///  * Errors if an input is unallocated (InsufficientInputs)
	///  * Errors if an input has 0 readers_remaining or if there are writers_remaining (SubGraphNotExecutable)
	///  * Errors if an input is not in the subgraph (OpInputNotInSubgraph)
	fn set_next_op(mut self, op: &Op) -> Result<(Self, bool), ExecError> {
		self.finalise_current_op();

		// These references must not escape the current method.
		let value_map = unsafe { &*self.value_map.get() };
		let borrows = unsafe { &mut *self.borrows.get() };

		borrows.clear();
		self.current_op = Some(op.clone());
		self.current_inputs = op.parent_nodes(); // TODO consider getting all op inputs and outputs in one pass to avoid relocking graph repeatedly
		self.current_outputs = op.child_nodes();

		let mut output_required = false;
		for node in &self.current_outputs {
			if self.is_required_output(node) {
				output_required = true; // ensure at least one output is required

				// check that outputs can be promoted to writable
				if let Some(ref value) = value_map.get(node) {
					match value {
						DataState::Unallocated { writers_remaining, .. }
						| DataState::Writable { writers_remaining, .. }
							if *writers_remaining > 0 => {} // Ok
						DataState::Deallocated => {
							panic!("Alumina Bug: An output ({}) of op ({}) has been deallocated", node, op) // unreachable?
						}
						_ => {
							return Err(ExecError::SubGraphNotExecutable { node: node.clone() });
						}
					}
				}
			}
		}

		// check that preconditions for allocate_readable_or_input hold
		for node in &self.current_inputs {
			if let Some(ref value) = value_map.get(node) {
				match value {
					DataState::Unallocated {
						writers_remaining,
						readers_remaining,
					} => {
						if *readers_remaining == 0 || *writers_remaining > 0 {
							return Err(ExecError::SubGraphNotExecutable { node: node.clone() });
						}
						if *writers_remaining == 0 {
							return Err(ExecError::InsufficientInputs {
								node: node.clone(),
								op: self.current_op().clone(),
							});
						}
					}
					DataState::Writable {
						writers_remaining,
						readers_remaining,
						..
					} => {
						if *readers_remaining == 0 || *writers_remaining > 0 {
							return Err(ExecError::SubGraphNotExecutable { node: node.clone() });
						}
					}
					DataState::Readable { readers_remaining, .. } => {
						if *readers_remaining == 0 {
							return Err(ExecError::SubGraphNotExecutable { node: node.clone() });
						}
					}

					DataState::Deallocated => panic!(
						"Alumina Bug: An input ({}) to op ({}) has been deallocated",
						node,
						self.current_op()
					),

					DataState::Input { readers_remaining, .. }
					| DataState::BroadcastInput { readers_remaining, .. } => {
						if *readers_remaining == 0 {
							return Err(ExecError::SubGraphNotExecutable { node: node.clone() });
						}
					}
				}
			} else {
				return Err(ExecError::OpInputNotInSubgraph {
					op: self.current_op().clone(),
					node: node.clone(),
				});
			}
		}

		Ok((self, !output_required)) // skip the op if no outputs are required
	}

	fn finalise_current_op(&mut self) {
		if self.current_op.is_some() {
			// This reference must not escape the current method.
			let value_map = unsafe { &mut *self.value_map.get() };

			for node in &self.current_inputs {
				let value = &mut value_map[node];
				match value {
					DataState::Unallocated { readers_remaining, .. }
					| DataState::Writable { readers_remaining, .. }
					| DataState::Readable { readers_remaining, .. }
					| DataState::Input { readers_remaining, .. }
					| DataState::BroadcastInput { readers_remaining, .. } => *readers_remaining -= 1,
					DataState::Deallocated => {}
				}
				if value.deallocatable() {
					::std::mem::swap(value, &mut DataState::Deallocated);
				}
			}

			for node in &self.current_outputs {
				if let Some(value) = value_map.get_mut(node) {
					match value {
						DataState::Unallocated { writers_remaining, .. }
						| DataState::Writable { writers_remaining, .. } => *writers_remaining -= 1,
						DataState::Readable { .. }
						| DataState::Deallocated
						| DataState::Input { .. }
						| DataState::BroadcastInput { .. } => {}
					}
					if value.deallocatable() {
						::std::mem::swap(value, &mut DataState::Deallocated);
					}
				}
			}
			self.current_op = None;
		}
	}

	unsafe fn convert_to_standard(&self, node: &NodeID) {
		let value_map = &mut *self.value_map.get();

		match &mut value_map[node] {
			DataState::Deallocated | DataState::Unallocated { .. } => unreachable!(),
			DataState::Writable { ref mut data, .. }
			| DataState::Readable { ref mut data, .. }
			| DataState::Input { ref mut data, .. }
			| DataState::BroadcastInput { ref mut data, .. } => {
				let mut arr_std = ArcArray::<f32, IxDyn>::uninitialized(data.shape());
				arr_std.assign(&data);
				::std::mem::swap(data, &mut arr_std)
			}
		}
	}

	/// Get an immutable view of an input.
	///
	/// # Panics
	///  * Panics if the node requested is not an input of the `Op`.
	pub fn get_input<'b>(&'b self, node: &NodeID) -> ArrayViewD<'b, f32> {
		let (_, node) = self.current_inputs.get_full(node).unwrap_or_else(||{
			panic!(
				"Op Bug: Op ({}) requested node (id:{}) as an input during execution, but does not list it as an input.",
				self.current_op(),
				node.id(),
			);
		});

		unsafe {
			// This reference must not escape the current method.
			let borrows = &mut *self.borrows.get();

			if let Some(&mutable) = borrows.get(&node.id()) {
				if mutable {
					panic!("Op Bug: Op ({}) requested an input (immutable) borrow of node ({}) which it had already borrowed mutably.", self.current_op(), node);
				}
			} else {
				borrows.insert(node.id(), false);
			}

			self.allocate_readable_or_input(node)
		}
	}

	/// Get an immutable view of an input which is guaranteed to be in contiguous standard layout ( C order).
	///
	/// # Panics
	///  * Panics if the node requested is not an input of the `Op`.
	pub fn get_input_standard<'b>(&'b self, node: &NodeID) -> ArrayViewD<'b, f32> {
		let (_, node) = self.current_inputs.get_full(node).unwrap_or_else(||{
			panic!(
				"Op Bug: Op ({}) requested node (id:{}) as an input during execution, but does not list it as an input.",
				self.current_op(),
				node.id(),
			);
		});

		unsafe {
			// This reference must not escape the current method.
			let borrows = &mut *self.borrows.get();

			if let Some(&mutable) = borrows.get(&node.id()) {
				if mutable {
					panic!("Op Bug: Op ({}) requested an input (immutable) borrow of node ({}) which it had already borrowed mutably.", self.current_op(), node);
				}
			} else {
				borrows.insert(node.id(), false);
			}

			let arr = self.allocate_readable_or_input(node);

			if arr.is_standard_layout() {
				arr
			} else {
				self.convert_to_standard(node);
				let arr_std = self.allocate_readable_or_input(node);
				debug_assert!(arr_std.is_standard_layout());
				arr_std
			}
		}
	}

	/// Get a mutable view of an output. Must check `is_required_output()` if Op has more than one output.
	///
	/// # Panics
	///  * Panics if the node is not `is_required_output()`.
	///  * Panics if the node has already been mutable borrowed by the `Op`.
	///  * Panics if the node requested is not an output of the `Op`.
	pub fn get_output<'b>(&'b self, node: &NodeID) -> ArrayViewMutD<'b, f32> {
		let (_, node) = self.current_outputs.get_full(node).unwrap_or_else(||{
			panic!(
				"Op Bug: Op ({}) requested node (id:{}) as an output during execution, but does not list it as an output.",
				self.current_op(),
				node.id()
			);
		});

		if !self.is_required_output(node) {
			panic!("Op Bug: Op ({}) requested node ({}) as an output during execution, but it is not required for the execution of the subgraph. Op should use is_required_output() when it has more than output", self.current_op(), node);
		}

		unsafe {
			// This reference must not escape the current method.
			let borrows = &mut *self.borrows.get();

			if let Some(&mutable) = borrows.get(&node.id()) {
				if mutable {
					panic!("Op Bug: Op ({}) requested an output (mutable) borrow of node ({}) which it had already borrowed mutably.", self.current_op(), node);
				} else {
					panic!("Op Bug: Op ({}) requested an output (mutable) borrow of node ({}) which it had already borrowed immutably.", self.current_op(), node);
				}
			} else {
				borrows.insert(node.id(), true);
			}

			self.allocate_writeable(node)
		}
	}

	/// Get a mutable view of an output. Must check `is_required_output()` if Op has more than one output which is
	/// guaranteed to be in contiguous standard layout ( C order).
	pub fn get_output_standard<'b>(&'b self, node: &NodeID) -> ArrayViewMutD<'b, f32> {
		let (_, node) = self.current_outputs.get_full(node).unwrap_or_else(||{
			panic!(
				"Op Bug: Op ({}) requested node (id:{}) as an output during execution, but does not list it as an output.",
				self.current_op(),
				node.id()
			);
		});

		if !self.is_required_output(node) {
			panic!("Op Bug: Op ({}) requested node ({}) as an output during execution, but it is not required for the execution of the subgraph. Op should use is_required_output() when it has more than output", self.current_op(), node);
		}

		unsafe {
			// This reference must not escape the current method.
			let borrows = &mut *self.borrows.get();

			if let Some(&mutable) = borrows.get(&node.id()) {
				if mutable {
					panic!("Op Bug: Op ({}) requested an output (mutable) borrow of node ({}) which it had already borrowed mutably.", self.current_op(), node);
				} else {
					panic!("Op Bug: Op ({}) requested an output (mutable) borrow of node ({}) which it had already borrowed immutably.", self.current_op(), node);
				}
			} else {
				borrows.insert(node.id(), true);
			}

			let arr = self.allocate_writeable(node);

			if arr.is_standard_layout() {
				arr
			} else {
				self.convert_to_standard(node);
				let arr_std = self.allocate_writeable(node);
				debug_assert!(arr_std.is_standard_layout());
				arr_std
			}
		}
	}

	/// Check whether a particular output is required from an `Op`.
	///
	/// Not necessary to check for ops which only have one output.
	pub fn is_required_output(&self, node: &NodeID) -> bool {
		unsafe {
			// This reference must not escape the current method.
			let value_map = &mut *self.value_map.get();

			value_map
				.get(node)
				.map(|value| value.readers_remaining() > 0 && value.writers_remaining() > 0)
				.unwrap_or(false)
		}
	}

	/// Must check borrows to ensure the node hasnt been borrowed in any way before calling to avoid creating a
	/// duplicate mutable reference.
	///
	/// # Panics
	/// if data has already been deallocated
	/// if data is already readable
	/// if shape is not compatible with the available input values
	unsafe fn allocate_writeable(&self, node: &NodeID) -> ArrayViewMutD<f32> {
		// This reference must not escape the current method.
		let value_map = &mut *self.value_map.get();

		match &mut value_map[node] {
			x @ &mut DataState::Unallocated { .. } => {
				// upgrade to writable

				let data = ArcArray::<f32, IxDyn>::zeros(self.shape_map[node].slice()).clone();

				let mut new_value = DataState::Writable {
					writers_remaining: x.writers_remaining(),
					readers_remaining: x.readers_remaining(),
					data,
				};

				::std::mem::swap(&mut new_value, x);

				if let DataState::Writable { ref mut data, .. } = x {
					data.view_mut()
				} else {
					unreachable!()
				}
			}
			&mut DataState::Writable { ref mut data, .. } => data.view_mut(),
			&mut DataState::Readable { .. } => {
				panic!("Alumina Bug: data already promoted to readable cannot be allocated as writable")
			}
			&mut DataState::Deallocated => panic!("Alumina Bug: valid input has been deallocated prematurely"),

			&mut DataState::Input { .. } | &mut DataState::BroadcastInput { .. } => {
				panic!("Alumina Bug: data already allocated as an input cannot be allocated as writable")
			}
		}
	}

	/// Must check borrows to ensure the node hasnt been borrowed mutably before calling to avoid creating an illegal
	/// reference.
	unsafe fn allocate_readable_or_input(&self, node: &Node) -> ArrayViewD<f32> {
		// This reference must not escape the current method.
		let value_map = &mut *self.value_map.get();

		match &mut value_map[node] {
			&mut DataState::Unallocated { .. } => {
				panic!("Alumina Bug: Attempting to directly allocate node as readable indicates that an InsufficientInputs error should have been thrown: node {}", node)
			},
			x @ &mut DataState::Writable { .. } => {
				// upgrade to readable

				debug_assert_eq!(
					x.writers_remaining(),
					0,
					"allocating as readable when writers still remain: {}",
					node
				);

				let data = if let DataState::Writable { ref mut data, .. } = x {
					let mut data2 = ArcArray::<f32, IxDyn>::uninitialized(IxDyn(&[0])); // 1. deallocated at 2
					::std::mem::swap(data, &mut data2);
					data2
				} else {
					unreachable!()
				};

				let mut new_value = DataState::Readable {
					readers_remaining: x.readers_remaining(),
					data: data.into_shared(),
				};

				::std::mem::swap(&mut new_value, x); // 2. deallocate uninit value here, its ok because f32 doesnt have a destructor

				if let DataState::Readable { ref mut data, .. } = x {
					data.view()
				} else {
					unreachable!()
				}
			},
			&mut DataState::Readable { ref data, .. } => data.view(),
			&mut DataState::Deallocated => {
				panic!("Alumina Bug: valid input has been deallocated prematurely: {}", node)
			},

			&mut DataState::Input { ref data, .. } => data.view(),

			x @ &mut DataState::BroadcastInput { .. } => {
				// upgrade to Input

				let data = if let DataState::BroadcastInput { ref mut data, .. } = x {
					let mut data2 = ArcArray::uninitialized(IxDyn(&[0])); // 1. deallocated at 2
					::std::mem::swap(data, &mut data2);
					data2
				} else {
					unreachable!()
				};

				let data = if self.shape_map[&node.id()].slice() == data.shape() {
					data
				} else {
					data.broadcast(self.shape_map[&node.id()].slice())
						.expect("Alumina Bug: an incorrect shape snuck through somehow")
						.to_owned()
						.into_shared()
				};

				let mut new_value = DataState::Input {
					readers_remaining: x.readers_remaining(),
					data,
				};

				::std::mem::swap(&mut new_value, x); // 2. deallocate uninit value here, its ok because f32 doesnt have a destructor

				if let DataState::Input { ref mut data, .. } = x {
					data.view()
				} else {
					unreachable!()
				}
			},
		}
	}

	/// Returns true if `set()` will succeed.
	///
	/// This guarantee is invalidated if the input is then borrowed before `set()` is called.
	///
	/// Generally this checks that the array for an output node is not yet allocated and can be set rather than
	/// accumulating into it.
	///
	/// # Panics
	/// * Panics if `node` is not an output.
	pub fn can_set(&self, node: &NodeID) -> bool {
		assert!(
			self.current_outputs.contains(node),
			"Op Bug: Op ({}) called can_set with node (id:{}) during execution, but does not list it as an output.",
			self.current_op(),
			node.id()
		);
		unsafe {
			// These references must not escape the current method.
			let value_map = &mut *self.value_map.get();
			let borrows = &mut *self.borrows.get();

			!borrows.contains_key(node)
				&& self.is_required_output(node)
				&& match value_map[node] {
					DataState::Unallocated { .. } => true,
					DataState::Deallocated
					| DataState::Readable { .. }
					| DataState::Writable { .. }
					| DataState::Input { .. }
					| DataState::BroadcastInput { .. } => false,
				}
		}
	}

	/// Set an output directly with an owned array.
	///
	/// Can be used with `take()` to implement in-place optimisation for operations.
	/// Must be proceeded by a call to `can_set()` to avoid panics.
	///
	/// # Panics
	/// * Panics if `!array.is_standard_layout()`.
	/// * May panic if `can_set()` would panic.
	/// * May panic if `can_set()` would have returned false.
	/// * Panics if `node` is not an output.
	pub fn set(&self, node: &NodeID, array: ArcArray<f32, IxDyn>) {
		assert!(
			self.can_set(node),
			"Op Bug: Op ({}) called set on node (id:{}), but can_set returned false.",
			self.current_op(),
			node.id()
		);
		// assert!(array.is_standard_layout(), "Op Bug: Op ({}) called set, but array does not have a standard layout,
		// array has shape ({:?}) and strides ({:?})", self.current_op(), array.shape(), array.strides());
		unsafe {
			// This reference must not escape the current method.
			let value_map = &mut *self.value_map.get();

			let (readers_remaining, writers_remaining) = match &mut value_map[node] {
				&mut DataState::Unallocated {
					readers_remaining,
					writers_remaining,
				} => (readers_remaining, writers_remaining),
				&mut DataState::Deallocated
				| &mut DataState::Readable { .. }
				| &mut DataState::Writable { .. }
				| &mut DataState::Input { .. }
				| DataState::BroadcastInput { .. } => unreachable!(),
			};

			let mut data = DataState::Writable {
				readers_remaining,
				writers_remaining,
				data: array,
			};
			::std::mem::swap(&mut data, &mut value_map[node]);
		}
	}

	/// Returns true if `take()` will succeed.
	///
	/// This guarantee is invalidated if the input is then borrowed.
	///
	/// Generally this checks that the array for an input node has no remaining readers other than the current op and
	/// can be taken rather then immutably borrowed.
	///
	/// # Panics
	/// * Panics if `node` is not an input.
	pub fn can_take(&self, node: &NodeID) -> bool {
		assert!(
			self.current_inputs.contains(node),
			"Op Bug: Op ({}) called can_take with node (id:{}) during execution, but does not list it as an output.",
			self.current_op(),
			node.id()
		);
		unsafe {
			// These references must not escape the current method.
			let value_map = &mut *self.value_map.get();
			let borrows = &mut *self.borrows.get();

			!borrows.contains_key(node)
			&& match value_map[node] {
				DataState::Unallocated{..} => panic!("Alumina Bug: Attempting to directly allocate node as readable indicates that an InsufficientInputs error should have been thrown: node (id:{})", node.id()),
				DataState::Readable{readers_remaining, ..} |
				DataState::Writable{readers_remaining, ..} | DataState::Input {readers_remaining, ..} | DataState::BroadcastInput {readers_remaining, ..} => readers_remaining == 1,
				DataState::Deallocated => false,
			}
		}
	}

	/// Take the value of a node as an owned array.
	///
	/// When paired with `set()` this can be used for in-place modification.
	/// Must be proceeded by a call to `can_take()` to avoid panics.
	///
	/// After the value is taken, the context will act as though the node is mutably borrowed preventing reborrowing.
	///
	/// # Panics
	/// * May panic if `can_take()` would panic.
	/// * May panic if `can_take()` would have returned false.
	/// * Panics if `node` is not an input.
	pub fn take(&self, node: &NodeID) -> ArcArray<f32, IxDyn> {
		assert!(
			self.can_take(node),
			"Op Bug: Op ({}) called take on node (id:{}), but can_take returned false.",
			self.current_op(),
			node.id()
		);

		unsafe {
			// These references must not escape the current method.
			let value_map = &mut *self.value_map.get();
			let borrows = &mut *self.borrows.get();

			borrows.insert(node.clone(), true);

			let mut data = DataState::Deallocated;
			::std::mem::swap(&mut data, &mut value_map[node]);

			match data {
				DataState::Unallocated { .. } | DataState::Deallocated => unreachable!(),
				DataState::Writable { data, .. } | DataState::Readable { data, .. } => data,
				DataState::Input { data, .. } => data,
				DataState::BroadcastInput { data, .. } => data
					.broadcast(self.shape_map[node].slice())
					.expect("Alumina Bug: an incorrect shape snuck through somehow")
					.to_shared(), // TODO do broadcast here data.into_owned(),
			}
		}
	}

	/// Take the value of a node as an owned array, which is guaranteed to be in standard layout ( C order).
	pub fn take_standard(&self, node: &NodeID) -> ArcArray<f32, IxDyn> {
		let arr = self.take(node);
		if arr.is_standard_layout() {
			arr
		} else {
			unsafe {
				let mut arr_std = ArcArray::<f32, IxDyn>::uninitialized(arr.shape());
				debug_assert!(arr_std.is_standard_layout());
				arr_std.assign(&arr);
				arr_std
			}
		}
	}

	pub fn shape(&self, node: &NodeID) -> &[usize] {
		self.shape_map
			.get(node)
			.unwrap_or_else(|| {
				panic!(
					"Op Bug: Op ({}) called shape on node (id:{}), but it is not part of the subgraph",
					self.current_op(),
					node.id()
				)
			})
			.slice()
	}
}

fn form_output_map(
	outputs: IndexSet<Node>,
	mut value_map: IndexMap<Node, DataState<f32>>,
	shape_map: IndexMap<NodeID, IxDyn>,
) -> Result<IndexMap<Node, ArrayD<f32>>, ExecError> {
	// if output was not calculated return error
	let mut map = IndexMap::with_capacity(outputs.len());
	for node in outputs {
		if map.contains_key(&node) {
			continue;
		}
		match value_map.swap_remove(&node.id()) {
			Some(DataState::Writable { data, .. }) | Some(DataState::Readable { data, .. }) => {
				map.insert(node, data.into_owned());
			}
			Some(DataState::Input { data, .. }) => {
				map.insert(node, data.into_owned());
			}
			Some(DataState::BroadcastInput { data, .. }) => {
				let arr = data
					.broadcast(shape_map[&node.id()].slice())
					.expect("Alumina Bug: an incorrect shape snuck through somehow")
					.into_owned();
				map.insert(node, arr);
			}
			Some(DataState::Unallocated { .. }) | None => {
				return Err(ExecError::OutputNotComputable { node });
			}
			Some(DataState::Deallocated) => panic!("Alumina bug: output was deallocated: {}", node),
		}
	}

	Ok(map)
}

#[derive(Default, Debug)]
pub struct OpPerf {
	pub invocation_count: usize,
	pub cumulative_usage: f32,
	pub cumulative_time: f32,
}

pub struct ExecConfig<'a> {
	pub use_node_values: bool,
	pub subgraph: Option<&'a SubGraph>,
	pub perf_records: Option<&'a mut IndexMap<Op, OpPerf>>,
}

impl<'a> ExecConfig<'a> {
	/// Default: true
	pub fn use_node_values(mut self, use_node_values: bool) -> Self {
		self.use_node_values = use_node_values;
		self
	}

	/// Default: None
	pub fn subgraph(mut self, subgraph: Option<&'a SubGraph>) -> Self {
		self.subgraph = subgraph;
		self
	}

	/// If Some the OpPerf for each Op will be updated
	///
	/// Default: None
	pub fn perf_records(mut self, perf_records: Option<&'a mut IndexMap<Op, OpPerf>>) -> Self {
		self.perf_records = perf_records;
		self
	}
}

impl<'a> Default for ExecConfig<'a> {
	fn default() -> Self {
		ExecConfig {
			use_node_values: true,
			subgraph: None,
			perf_records: None,
		}
	}
}

/// Execution with a custom subgraph
///
/// Ops are executed in the order contained in the subgraph, if this order is not topological 
/// then a `SubGraphNotExecutable` Error is returned (i.e. Any Op that reads from a node must come after all Ops that write to that node).
///
/// The order of nodes in the result map is the same as the outputs argument with duplicates skipped.
pub fn exec<I, O, T1, T2>(
	inputs: T1,
	outputs: T2,
	config: &mut ExecConfig,
) -> Result<IndexMap<Node, ArrayD<f32>>, ExecError>
where
	I: Borrow<Node> + Hash + Eq,
	O: Into<Node>,
	T1: IntoIterator<Item = (I, ArcArray<f32, IxDyn>)>,
	T2: IntoIterator<Item = O>,
{
	let inputs: IndexMap<_, _> = inputs.into_iter().collect();
	let outputs: IndexSet<Node> = outputs.into_iter().map(Into::into).collect();
	
	let mut system = sysinfo::System::new_with_specifics(RefreshKind::new().with_cpu());
	
	let new_subgraph;
	let subgraph = if let Some(subgraph) = config.subgraph.as_ref() {
		subgraph
	} else {
		new_subgraph = execution_subgraph(
			inputs.keys().map(|n| n.borrow().clone()),
			outputs.clone(),
			config.use_node_values,
		)
		.map_err(|e| ExecError::Subgraph { error: e })?;

		&new_subgraph
	};

	// Check that all outputs are in the subgraph
	// Excess inputs are deduplicated
	for node in &outputs {
		if !subgraph.nodes.contains(node) {
			return Err(ExecError::OutputsNotInSubgraph { node: node.clone() });
		}
	}

	let (writers_remaining, readers_remaining) = cached_node_input_output_count(subgraph, &outputs);

	let mut inputs: IndexMap<NodeID, ArcArray<f32, IxDyn>> = inputs
		.into_iter()
		.map(|(node, data)| (node.borrow().id().clone(), data))
		.collect();

	// put values into inputs now to avoid possible race condition if a value gets removed partway through the execution
	if config.use_node_values {
		for node in &subgraph.nodes {
			if let Some(val) = node.value() {
				inputs.entry(node.id().clone()).or_insert_with(|| val);
			}
		}
	}

	let shape_map = cached_shapes_inner(
		subgraph,
		&inputs
			.iter()
			.map(|(node, value)| (node.clone(), IxDyn(value.shape())))
			.collect(),
		false
	)
	.map_err(|e| ExecError::Shape { error: e })?;

	let value_map: IndexMap<Node, DataState<f32>> = subgraph
		.nodes
		.iter()
		.map(|node| {
			(
				node.clone(),
				inputs
					.swap_remove(&node.id())
					.map(|data| {
						if data.shape() == shape_map[&node.id()].slice() {
							DataState::Input {
								readers_remaining: readers_remaining[&node.id()],
								data,
							}
						} else {
							DataState::BroadcastInput {
								readers_remaining: readers_remaining[&node.id()],
								data,
							}
						}
					})
					.unwrap_or_else(|| DataState::Unallocated {
						writers_remaining: writers_remaining[&node.id()],
						readers_remaining: readers_remaining[&node.id()],
					}),
			)
		})
		.collect();


	// let mut perf_map = OP_PERF_DATA.lock().unwrap();

	// Fold over ops executing those that arent skipped. No permanent references handed out
	let mut context = subgraph
		.ops
		.iter()
		.fold(Ok(ExecutionContext::new(value_map, shape_map)), |result, op| {
			result.and_then(|ctx| {
				let (ctx, skip) = ctx.set_next_op(op)?;

				if !skip {
					//assert!(config.perf_records.as_mut().and_then(|pr|pr.get_mut(op)).is_some());
					if let Some(record) = &mut config.perf_records.as_mut().and_then(|pr|pr.get_mut(op)) {
						system.refresh_cpu();
						//let _ = system.get_processors().iter().map(|p|p.get_cpu_usage()).sum::<f32>();
						//let _ = system.get_global_processor_info().get_cpu_usage() as f32;
						let start = Instant::now();
						op.instance().execute(&ctx).map_err(|e| ExecError::Op {
							error: e,
							op: op.clone(),
						})?;
						system.refresh_cpu();
						record.invocation_count += 1;
						record.cumulative_usage +=  system.get_processors().iter().map(|p|p.get_cpu_usage()).sum::<f32>()/system.get_processors().len() as f32;//system.get_global_processor_info().get_cpu_usage() as f32;
						record.cumulative_time += start.elapsed().as_micros() as f32;
					} else {
						op.instance().execute(&ctx).map_err(|e| ExecError::Op {
							error: e,
							op: op.clone(),
						})?;
					}
				}

				Ok(ctx)
			})
		})?;

	// This gets called by set_next_op for all ops except the last one.
	context.finalise_current_op();

	let ExecutionContext {
		value_map, shape_map, ..
	} = context;

	form_output_map(outputs, value_map.into_inner(), shape_map)
}



#[derive(Hash, PartialEq, Eq, Clone)]
struct InputOutputCountCacheKey {
	subgraph_nodes: Vec<NodeID>,
	subgraph_ops: Vec<OpID>,
	outputs: Vec<NodeID>,
}

thread_local! {
	#[allow(clippy::type_complexity)]
	static NODE_INPUT_OUTPUT_COUNT_CACHE: RefCell<LruCache<InputOutputCountCacheKey, (IndexMap<NodeID, usize>, IndexMap<NodeID, usize>)>> = RefCell::new(LruCache::new(32));
}

/// returns a tuple `(writers_remaining, readers_remaining)` of counts of inputs to a node and outputs to a node
///
/// results for a given subgraph and
fn cached_node_input_output_count<O>(
	subgraph: &SubGraph,
	outputs: &IndexSet<O>,
) -> (IndexMap<NodeID, usize>, IndexMap<NodeID, usize>)
where
	O: Borrow<Node> + Hash + Eq,
{
	NODE_INPUT_OUTPUT_COUNT_CACHE.with(|cache_cell| {
		let mut cache = cache_cell.borrow_mut();

		// Generate a key which has a unique result
		let mut key = InputOutputCountCacheKey {
			subgraph_nodes: subgraph.nodes.iter().map(|node| node.id().into()).collect(),
			subgraph_ops: subgraph.ops.iter().map(|op| op.id()).collect(),
			outputs: outputs.iter().map(|node| node.borrow().id().into()).collect(),
		};
		key.subgraph_nodes.sort_unstable_by(|a, b| a.id().cmp(&b.id()));
		key.subgraph_ops.sort_unstable_by(|a, b| a.id().cmp(&b.id()));
		key.outputs.sort_unstable_by(|a, b| a.id().cmp(&b.id()));

		// Get a copy of counts from the cache, or insert a new one for this subgraph
		let (writers_remaining, readers_remaining) = cache.get(&key).cloned().unwrap_or_else(|| {
			let writers_remaining: IndexMap<NodeID, usize> = subgraph
				.input_counts()
				.0
				.into_iter()
				.map(|(node, count)| (node.id().clone(), count))
				.collect();
			let mut readers_remaining: IndexMap<NodeID, usize> = subgraph
				.output_counts()
				.0
				.into_iter()
				.map(|(node, count)| (node.id().clone(), count))
				.collect();

			// Increase remaining readers to avoid deallocation
			for output in outputs {
				*readers_remaining.get_mut(&output.borrow().id()).unwrap() += 1;
			}

			cache.put(key.clone(), (writers_remaining.clone(), readers_remaining.clone()));

			(writers_remaining, readers_remaining)
		});

		// Return
		(writers_remaining, readers_remaining)
	})
}

// /// Call execute and use the results to update the node values
// pub fn exec_into_values(
// 	inputs: IndexMap<Node, ArrayD<f32>>,
// 	outputs: IndexSet<Node>,
// 	use_node_values: bool,
// ) -> Result<(), ExecError> {
// 	let map = exec(inputs, outputs, use_node_values)?;
// 	for (mut n, v) in map {
// 		n.set_value(v);
// 	}
// 	Ok(())
// }

#[cfg(test)]
mod tests {
	#![allow(non_snake_case)]

	use crate::{
		base_ops::{dummy::DummyOp, shape_constraint::same_shape, OpBuilder},
		errors::{ExecutionSubgraphError, ShapesError},
		exec::{exec, ExecConfig, ExecError},
		graph::Node,
		subgraph::SubGraph,
	};
	use indexmap::indexset;
	use indexmap::IndexMap;
	use ndarray::arr0;

	#[test]
	fn exec_error_OpInputNotInSubgraph() {
		let x = Node::new(&[2, 1]).set_name("x");
		let y = Node::new(&[3, 2]).set_name("y");

		let op = DummyOp::new().input(&x).output(&y).build().unwrap();

		let subgraph = SubGraph::new(indexset![&y], indexset![&op]);

		match exec(
			IndexMap::<Node, _>::new(),
			&[y],
			&mut ExecConfig::default().subgraph(Some(&subgraph)).use_node_values(false),
		) {
			Err(ExecError::OpInputNotInSubgraph { .. }) => {}
			Err(ExecError::Shape {
				error: ShapesError::OpInputNotInSubgraph { .. },
			}) => {}
			Err(x) => panic!("{}", x),
			Ok(_) => panic!("No Error"),
		}
	}

	#[test]
	fn exec_error_Subgraph_Cycle() {
		let x = Node::new(&[2, 1]).set_name("x");
		let y = Node::new(&[3, 2]).set_name("y");

		let _op = DummyOp::new().input(&x).output(&y).build().unwrap();
		let _op = DummyOp::new().input(&y).output(&x).build().unwrap();

		match exec(
			IndexMap::<Node, _>::new(),
			&[y],
			&mut ExecConfig::default().use_node_values(false),
		) {
			Err(ExecError::Subgraph {
				error: ExecutionSubgraphError::Cycle { .. },
			}) => {}
			Err(x) => panic!("{}", x),
			Ok(_) => panic!("No Error"),
		}
	}

	#[test]
	fn exec_error_Subgraph_InsufficientInputs() {
		let x = Node::new(&[2, 1]).set_name("x");
		let y = Node::new(&[3, 2]).set_name("y");

		let _op = DummyOp::new().input(&x).output(&y).build().unwrap();

		match exec(
			IndexMap::<Node, _>::new(),
			&[&y],
			&mut ExecConfig::default().use_node_values(false),
		) {
			Err(ExecError::Subgraph {
				error: ExecutionSubgraphError::InsufficientInputs { .. },
			}) => {}
			Err(x) => panic!("{}", x),
			Ok(_) => panic!("No Error"),
		}
	}

	#[test]
	fn exec_error_OutputNotComputable() {
		let x = Node::new(&[2, 1]).set_name("x");
		let y = Node::new(&[3, 2]).set_name("y");
		let z = Node::new(&[3, 3]).set_name("z");

		let op = DummyOp::new().input(&x).output(&y).build().unwrap();

		let subgraph = SubGraph::new(indexset![&x, &y, &z], indexset![&op]);

		match exec(
			vec![(x, arr0(1.0).into_dyn().into_shared())],
			&[&z],
			&mut ExecConfig::default().subgraph(Some(&subgraph)).use_node_values(false),
		) {
			Err(ExecError::OutputNotComputable { .. }) => {}
			Err(x) => panic!("{}", x),
			Ok(_) => panic!("No Error"),
		}
	}

	#[test]
	fn exec_error_Shape() {
		let x = Node::new(&[2, 1]).set_name("x");
		let y = Node::new(&[3, 2]).set_name("y");

		same_shape(&x, &y).unwrap();

		match exec(
			vec![(x, arr0(1.0).into_dyn().into_shared())],
			&[&y],
			&mut ExecConfig::default().use_node_values(false),
		) {
			Err(ExecError::Shape { .. }) => {}
			Err(x) => panic!("{}", x),
			Ok(_) => panic!("No Error"),
		}
	}

	#[test]
	fn exec_error_OutputsNotInSubgraph() {
		let x = Node::new(&[2, 1]).set_name("x");
		let y = Node::new(&[3, 2]).set_name("y");

		let op = DummyOp::new().input(&x).output(&y).build().unwrap();

		let subgraph = SubGraph::new(indexset![&x], indexset![&op]);

		match exec(
			vec![(x, arr0(1.0).into_dyn().into_shared())],
			&[y],
			&mut ExecConfig::default().subgraph(Some(&subgraph)).use_node_values(false),
		) {
			Err(ExecError::OutputsNotInSubgraph { .. }) => {}
			Err(x) => panic!("{}", x),
			Ok(_) => panic!("No Error"),
		}
	}

	#[test]
	fn exec_error_SubGraphNotExecutable_order() {
		let x = Node::new(&[2, 1]).set_name("x");
		let y = Node::new(&[3, 2]).set_name("y");
		let z = Node::new(&[3, 3]).set_name("z");

		let op1 = DummyOp::new().input(&y).output(&z).build().unwrap();
		let op2 = DummyOp::new().input(&x).output(&y).build().unwrap();

		let subgraph = SubGraph::new(indexset![&x, &y], indexset![&op1, &op2]);

		match exec(
			vec![(x, arr0(1.0).into_dyn().into_shared())],
			&[y],
			&mut ExecConfig::default().subgraph(Some(&subgraph)).use_node_values(false),
		) {
			Err(ExecError::Shape {
				error: ShapesError::SubGraphNotExecutable { .. },
			}) => {}
			Err(ExecError::SubGraphNotExecutable { .. }) => {}
			Err(x) => panic!("{}", x),
			Ok(_) => panic!("No Error"),
		}
	}

	#[test]
	fn exec_error_SubGraphNotExecutable_cyclic() {
		let x = Node::new(&[2, 1]).set_name("x");
		let y = Node::new(&[3, 2]).set_name("y");

		let op1 = DummyOp::new().input(&x).output(&y).build().unwrap();
		let op2 = DummyOp::new().input(&y).output(&x).build().unwrap();

		let subgraph = SubGraph::new(indexset![&x, &y], indexset![&op1, &op2]);

		match exec(
			vec![(x, arr0(1.0).into_dyn().into_shared())],
			&[y],
			&mut ExecConfig::default().subgraph(Some(&subgraph)).use_node_values(false),
		) {
			Err(ExecError::Shape {
				error: ShapesError::SubGraphNotExecutable { .. },
			}) => {}
			Err(ExecError::SubGraphNotExecutable { .. }) => {}
			Err(x) => panic!("{}", x),
			Ok(_) => panic!("No Error"),
		}
	}
}
