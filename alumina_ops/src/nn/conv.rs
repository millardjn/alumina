use alumina_core::{
	base_ops::{OpInstance, OpSpecification},
	errors::{ExecutionError, GradientError, OpBuildError, ShapePropError},
	exec::ExecutionContext,
	grad::GradientContext,
	graph::{merge_graphs, Graph, HeavyNode, Node, NodeID, NodeTag, Op},
	init::msra,
	shape::{NodeAxis, NodeShape},
	shape_prop::ShapePropContext,
};
use indexmap::{indexset, IndexMap, IndexSet};
use lazy_static::lazy_static;

use ndarray::{ArrayD, Dimension, IxDyn};

use smallvec::SmallVec;
use threadpool::ThreadPool;
use threadpool_scope::scope_with;

use std::{
	any::Any,
	cmp::{max, min},
	iter::once,
	mem::size_of,
	ops::Sub,
	sync::{
		atomic::{AtomicUsize, Ordering},
		Mutex,
	},
};
use typenum::{
	bit::*,
	consts::{U1, U2, U3},
	marker_traits::Unsigned,
	operator_aliases::Sub1,
	UInt, UTerm,
};
use unchecked_index as ui;

use crate::sgemm::sgemm;

// Threadpool for offloading lowering/packing operations
lazy_static! {
	static ref NUM_CPUS: usize = num_cpus::get();
	static ref THREAD_POOL: Mutex<ThreadPool> = Mutex::new(ThreadPool::new(*NUM_CPUS));
}

#[derive(Debug, Clone)]
pub enum Padding {
	Full,
	Same,
	Valid,
	/// extra padding. 0 is equivalent to same
	Padded(usize),
	/// extra padding per dimension. 0 is equivalent to same
	PaddedDiff(Vec<usize>),
}

impl Padding {
	fn output_node_shape(&self, input_shape: &NodeShape, filter_shape: &[usize]) -> NodeShape {
		let batch_size = &input_shape.slice()[0];
		let out_channels = &filter_shape[filter_shape.len() - 1];
		let input_spatial = &input_shape.slice()[1..input_shape.len() - 1];
		let filter_spatial = &filter_shape[0..filter_shape.len() - 2];

		once(batch_size.into())
			.chain(
				input_spatial
					.iter()
					.zip(filter_spatial)
					.enumerate()
					.map(|(i, (input, f))| match (input, self) {
						(NodeAxis::Known { val }, Padding::Full) => (val + f - 1).into(),
						(NodeAxis::Known { val }, Padding::Same) => val.into(),
						(NodeAxis::Known { val }, Padding::Valid) => (val - f + 1).into(),
						(NodeAxis::Known { val }, Padding::Padded(size)) => (val + size).into(),
						(NodeAxis::Known { val }, Padding::PaddedDiff(ref vec)) => (val + vec[i]).into(),
						(_, _) => NodeAxis::unknown(),
					}),
			)
			.chain(once(out_channels.into()))
			.into()
	}

	fn output_shape(&self, input_shape: &[usize], filter_shape: &[usize]) -> NodeShape {
		let batch_size = input_shape[0];
		let out_channels = filter_shape[filter_shape.len() - 1];
		let input_spatial = input_shape[1..input_shape.len() - 1].iter();
		let filter_spatial = filter_shape[0..filter_shape.len() - 2].iter();

		match self {
			Padding::Full => once(batch_size)
				.chain(input_spatial.zip(filter_spatial).map(|(dim, k_dim)| dim + k_dim - 1))
				.chain(once(out_channels))
				.into(),
			Padding::Same => once(batch_size)
				.chain(input_spatial.cloned())
				.chain(once(out_channels))
				.into(),
			Padding::Valid => once(batch_size)
				.chain(input_spatial.zip(filter_spatial).map(|(dim, k_dim)| dim - k_dim + 1))
				.chain(once(out_channels))
				.into(),
			Padding::Padded(ref size) => once(batch_size)
				.chain(input_spatial.map(|dim| dim + size))
				.chain(once(out_channels))
				.into(),
			Padding::PaddedDiff(ref vec) => once(batch_size)
				.chain(input_spatial.zip(vec).map(|(dim, vec_dim)| dim + vec_dim))
				.chain(once(out_channels))
				.into(),
		}
	}
}

pub struct ConvData {
	pub filter: Node,
	pub conv: Op,
}

/// Default initialisation is conv_msra(1.0)
/// #Errors
/// * TODO
pub fn conv<I>(
	input: I,
	output_channels: usize,
	filter_shape: &[usize],
	padding: Padding,
) -> Result<HeavyNode<ConvData>, OpBuildError>
where
	I: Into<Node>,
{
	let input = input.into();

	let input_channels = if let NodeAxis::Known { val } = input.shape().slice()[input.shape().len() - 1] {
		val
	} else {
		return Err(format!(
			"Input shape must have a known number of channels (innermost/last axis): {}",
			input.shape()
		)
		.into());
	};
	let filter_shape: Vec<usize> = filter_shape
		.iter()
		.chain(once(&input_channels))
		.chain(once(&output_channels))
		.cloned()
		.collect();
	let output_shape = padding.output_node_shape(&input.shape(), &filter_shape);

	let filter = input
		.graph()
		.new_node(filter_shape.into())
		.set_init(msra(1.0))
		.add_tag(NodeTag::Parameter)
		.set_name_unique(&format!("conv({})_weights", input));

	let output = input
		.graph()
		.new_node(output_shape)
		.set_name_unique(&format!("conv({})", input));

	let conv = Conv::new(input, filter.clone(), output.clone())
		.padding(padding)
		.build()?;

	Ok(HeavyNode::new(output, ConvData { filter, conv }))
}

/// TODO: Are non-known filter shapes ok here?
pub fn conv_with<I, F>(input: I, filter: F, padding: Padding) -> Result<Node, OpBuildError>
where
	I: Into<Node>,
	F: Into<Node>,
{
	let input = input.into();
	let filter = filter.into();
	merge_graphs(&[input.graph(), filter.graph()]);

	let filter_shape = filter
		.shape()
		.to_data_shape()
		.map_err(|_err| format!("Filter shape must be known.: {}", filter.shape()))?;

	let output_shape = padding.output_node_shape(&input.shape(), filter_shape.slice());

	let output = input
		.graph()
		.new_node(output_shape)
		.set_name_unique(&format!("conv_with({},{})", input, filter));

	let _op = Conv::new(input, filter, output.clone()).padding(padding).build()?;

	Ok(output)
}

pub fn conv_into<I, O>(input: I, output: O, filter_shape: &[usize], padding: Padding) -> Result<ConvData, OpBuildError>
where
	I: Into<Node>,
	O: Into<Node>,
{
	let input = input.into();
	let output = output.into();

	let input_channels = if let NodeAxis::Known { val } = input.shape().slice()[input.shape().len() - 1] {
		val
	} else {
		return Err(format!(
			"Input shape must have a known number of channels (innermost/last axis): {}",
			input.shape()
		)
		.into());
	};

	let output_channels = if let NodeAxis::Known { val } = output.shape().slice()[input.shape().len() - 1] {
		val
	} else {
		return Err(format!(
			"Output shape must have a known number of channels (innermost/last axis): {}",
			output.shape()
		)
		.into());
	};

	let filter_shape: Vec<usize> = filter_shape
		.iter()
		.chain(once(&input_channels))
		.chain(once(&output_channels))
		.cloned()
		.collect();

	let filter = input
		.graph()
		.new_node(filter_shape.into())
		.set_init(msra(1.0))
		.add_tag(NodeTag::Parameter)
		.set_name_unique(&format!("conv({})_weights", input));

	let op = Conv::new(input, filter.clone(), output).padding(padding).build()?;

	Ok(ConvData { conv: op, filter })
}

/// TODO: Are non-known filter shapes ok here?
pub fn conv_with_into<I, F, O>(input: I, filter: F, output: O, padding: Padding) -> Result<Op, OpBuildError>
where
	I: Into<Node>,
	F: Into<Node>,
	O: Into<Node>,
{
	let input = input.into();
	let filter = filter.into();
	let output = output.into();

	Conv::new(input, filter, output).padding(padding).build()
}

/// The convolution operation used in convolutional neural nets
///
/// Borrowing the tensorflow description:
///
/// Computes sums of N-D convolutions (actually cross-correlation).
///
/// The input and output of the Op are both rank (N+2) Tensors of shape:
///
/// `[num_batches, input_spatial_shape[0], ..., input_spatial_shape[N-1], num_input_channels]`
///
/// and the filters are a rank (N+2) Tensor of shape
///
/// `[spatial_filter_shape[0], ..., spatial_filter_shape[N-1], num_input_channels, num_output_channels]`
#[must_use = "Op builder not used, call .build()"]
#[derive(Clone, Debug)]
pub struct Conv {
	padding: Padding,
	input: Node,
	output: Node,
	filter: Node,

	lowering_memory: usize,
}

impl Conv {
	pub fn new<I, F, O>(input: I, filter: F, output: O) -> Self
	where
		I: Into<Node>,
		F: Into<Node>,
		O: Into<Node>,
	{
		Conv {
			padding: Padding::Same,
			input: input.into(),
			output: output.into(),
			filter: filter.into(),
			lowering_memory: 1024 * 384,
		}
	}

	/// Padding determines the shape of the output with respect to the input
	///
	/// Default: `Padding::Same`
	pub fn padding(mut self, padding: Padding) -> Self {
		self.padding = padding;
		self
	}

	/// Sets the limit on how large a lowering operation block can be before splitting it.
	///
	/// Convolution is implemented by converting a section of the input into a matrix (lowering).
	/// Generally this should fit within the per core data cache for good performance.
	///
	/// This may be exceeded if there is no smaller way to perform the lowering.
	pub fn lowering_memory(mut self, lowering_memory: usize) -> Self {
		self.lowering_memory = lowering_memory;
		self
	}
}

impl OpSpecification for Conv {
	type InstanceType = ConvInstance;

	fn op_type(&self) -> &'static str {
		"Conv"
	}

	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input.clone(), self.filter.clone()]
	}

	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.output.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			padding: self.padding.clone(),
			output: mapping.get(&self.output).unwrap_or(&self.output).clone(),
			input: mapping.get(&self.input).unwrap_or(&self.input).clone(),
			filter: mapping.get(&self.filter).unwrap_or(&self.filter).clone(),
			lowering_memory: self.lowering_memory,
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		if self.input.shape().len() < 3 {
			return Err(format!(
				"Node shapes should be of length 3 or greater (minimum 1 spatial dimension), input: {}",
				self.input.shape().len()
			)
			.into());
		}
		if self.input.shape().len() != self.filter.shape().len() {
			return Err(format!(
				"Node shapes should all have the same length: input: {}, filter: {}",
				self.input.shape().len(),
				self.filter.shape().len()
			)
			.into());
		}
		if self.input.shape().len() != self.output.shape().len() {
			return Err(format!(
				"Node shapes should all have the same length: input: {}, output: {}",
				self.input.shape().len(),
				self.output.shape().len()
			)
			.into());
		}

		Ok(ConvInstance {
			padding: self.padding,

			input: self.input.id(),
			output: self.output.id(),
			filter: self.filter.id(),

			lowering_memory: self.lowering_memory,
		})
	}
}

#[derive(Debug, Clone)]
pub struct ConvInstance {
	padding: Padding,
	input: NodeID,
	output: NodeID,
	filter: NodeID,
	lowering_memory: usize,
}

impl OpInstance for ConvInstance {
	fn op_type(&self) -> &'static str {
		"Conv"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(Conv {
			padding: self.padding.clone(),
			input: graph.node_from_id(self.input),
			output: graph.node_from_id(self.output),
			filter: graph.node_from_id(self.filter),
			lowering_memory: self.lowering_memory,
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input, self.filter]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.output]
	}

	fn gradient(&self, ctx: &mut GradientContext) -> Result<(), GradientError> {
		ConvBack::new(
			ctx.node(&self.input),
			ctx.node(&self.filter),
			ctx.grad_of(&self.output),
			ctx.grad_of(&self.input),
			ctx.grad_of(&self.filter),
		)
		.padding(self.padding.clone())
		.lowering_memory(self.lowering_memory)
		.build()?;

		Ok(())
	}

	fn propagate_shapes(&self, ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		let input_shape = ctx.input_shape(&self.input);
		let filter_shape = ctx.input_shape(&self.filter);
		let input_shape = input_shape.slice();
		let filter_shape = filter_shape.slice();

		let in_channels = input_shape[input_shape.len() - 1];

		if in_channels != filter_shape[filter_shape.len() - 2] {
			return Err(format!(
				"input channels dimension {} does not match final filter dimension {}",
				in_channels,
				filter_shape[filter_shape.len() - 2]
			)
			.into());
		}

		let input_spatial = input_shape[1..input_shape.len() - 1].iter();
		let filter_spatial = filter_shape[0..filter_shape.len() - 2].iter();
		if input_spatial.len() != filter_spatial.len() {
			return Err(
				format!("input shape and filter shape do not have the same number of spatial dimensions. input spatial: {:?}, filter spatial: {:?}", input_spatial, filter_spatial).into(),
			);
		}

		let output_shape = self.padding.output_shape(input_shape, filter_shape);

		ctx.merge_output_shape(&self.output, &output_shape)?;

		Ok(())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let input = ctx.get_input_standard(&self.input);
		let filter = ctx.get_input_standard(&self.filter);
		let mut output = ctx.get_output_standard(&self.output);

		let n = input.shape()[0]; // TODO use ensure to guard against zero length shapes
		let in_size: usize = input.shape()[1..].iter().product();
		let _out_size: usize = output.shape()[1..].iter().product();
		let patch_size = filter.shape()[0..filter.ndim() - 1].iter().product();

		let input_channels = input.shape()[input.ndim() - 1];
		let output_channels = output.shape()[output.ndim() - 1];

		let input_spatial = &input.shape()[1..input.ndim() - 1];
		let filter_spatial = &filter.shape()[0..filter.ndim() - 2];
		let output_spatial = output.shape()[1..output.ndim() - 1].to_vec();

		let in_spaxels: usize = input_spatial.iter().product();
		let out_spaxels: usize = output_spatial.iter().product();

		// use mat mul if possible
		if filter_spatial.iter().product::<usize>() == 1 && in_spaxels == out_spaxels {
			let m = output_channels;
			let n = n * out_spaxels;
			let k = patch_size;
			unsafe {
				sgemm(
					m,
					k,
					n,
					1.0,
					filter.as_ptr(),
					1,
					m as isize, // A is params, col major
					input.as_ptr(),
					1,
					k as isize, // B, input patches column major
					1.0,
					output.as_mut_ptr(),
					1,
					m as isize,
				); // C output values column major
			}
			return Ok(());
		}

		// Checks
		assert!(input.ndim() == output.ndim(), "Input len does not match output len");
		assert!(input.ndim() == filter.ndim(), "Filter len does not match input len");

		assert!(
			input.shape()[0] == output.shape()[0],
			"Batch size of input does not match batch size of output"
		);
		assert!(
			input_channels == filter.shape()[filter.ndim() - 2],
			"input channels dimension does not match final filter dimension"
		);
		assert!(
			output_channels == filter.shape()[filter.ndim() - 1],
			"output channels dimension does not match first filter dimension"
		);

		let input = input.as_slice().unwrap();
		let filter = filter.as_slice().unwrap();
		let output = output.as_slice().unwrap();
		debug_assert!(!filter.iter().cloned().any(f32::is_nan), "{:?}", filter);

		let filter_strides = stride_vec(input_channels, filter_spatial);
		let input_strides = stride_vec(input_channels, input_spatial);
		let output_strides = stride_vec(output_channels, &output_spatial);

		let n_threads = *NUM_CPUS;
		let cache_limit = self.lowering_memory / (patch_size * size_of::<f32>());
		let thread_division = (out_spaxels * n + n_threads - 1) / n_threads;
		let max_batch_spaxels = min(max(4, min(cache_limit, thread_division)), out_spaxels * n); // number of spaxels to combine in one sgemm (the last batch can have fewer)
		let n_batches = (out_spaxels * n + max_batch_spaxels - 1) / max_batch_spaxels;
		let batch_atomic = AtomicUsize::new(0);

		let pool = THREAD_POOL.lock().expect("Could not lock conv threadpool");
		scope_with(&pool, |scope| {
			for _ in 0..n_threads {
				let filter_strides = &filter_strides;
				let input_strides = &input_strides;
				let output_strides = &output_strides;
				let output_spatial = &output_spatial;
				let batch_atomic = &batch_atomic;

				scope.execute(move || {
					let mut patches_alloc = vec![0.0; patch_size * max_batch_spaxels];

					loop {
						let batch = batch_atomic.fetch_add(1, Ordering::SeqCst);
						if batch >= n_batches {
							break;
						}

						let spaxel_ind = batch * max_batch_spaxels;
						let batch_spaxels = min(out_spaxels * n - spaxel_ind, max_batch_spaxels);

						let patches = &mut patches_alloc[..batch_spaxels * patch_size];
						for (i, patch) in patches.chunks_mut(patch_size).enumerate() {
							debug_assert_eq!(patch_size, patch.len());
							let n_ind = (spaxel_ind + i) / out_spaxels;

							let in_n = &input[n_ind * in_size..][..in_size];

							let output_ind = (spaxel_ind + i) % out_spaxels * output_channels;
							match filter_spatial.len() {
								1 => unsafe_pack_specialised::<U1>(
									patch,
									in_n,
									input_channels,
									output_ind,
									filter_spatial,
									input_spatial,
									output_spatial,
									filter_strides,
									input_strides,
									output_strides,
								),
								2 => unsafe_pack_specialised::<U2>(
									patch,
									in_n,
									input_channels,
									output_ind,
									filter_spatial,
									input_spatial,
									output_spatial,
									filter_strides,
									input_strides,
									output_strides,
								),
								3 => unsafe_pack_specialised::<U3>(
									patch,
									in_n,
									input_channels,
									output_ind,
									filter_spatial,
									input_spatial,
									output_spatial,
									filter_strides,
									input_strides,
									output_strides,
								),
								_ => unsafe_pack(
									patch,
									in_n,
									input_channels,
									output_ind,
									filter_spatial,
									input_spatial,
									output_spatial,
									filter_strides,
									input_strides,
									output_strides,
								),
							}
						}

						unsafe {
							// ptr guaranteed to at least m*n away from other threads mutating the output
							let out_batch_ptr = output.as_ptr().add(spaxel_ind * output_channels) as *mut f32;

							let m = output_channels;
							let n = batch_spaxels;
							let k = patch_size;
							debug_assert_eq!(filter.len(), k * m);
							debug_assert!(patches.len() >= n * k);
							sgemm(
								m,
								k,
								n,
								1.0,
								filter.as_ptr(),
								// k as isize,
								// 1, // A is params, row major
								1,
								m as isize, // A is params, col major
								patches.as_ptr(),
								1,
								k as isize, // B, input patches column major
								1.0,
								out_batch_ptr,
								1,
								m as isize,
							); // C output values column major
						}
					}
				});
			}
		});
		Ok(())
	}
}

#[derive(Debug, Clone)]
pub struct ConvBack {
	padding: Padding,
	input: Node,
	filter: Node,
	output_grad: Node,

	input_grad: Node,
	filter_grad: Node,

	lowering_memory: usize,
}

impl ConvBack {
	pub fn new<I1, I2, I3, O1, O2>(input: I1, filter: I2, output_grad: I3, input_grad: O1, filter_grad: O2) -> Self
	where
		I1: Into<Node>,
		I2: Into<Node>,
		I3: Into<Node>,
		O1: Into<Node>,
		O2: Into<Node>,
	{
		let input = input.into();
		let filter = filter.into();
		let output_grad = output_grad.into();
		let input_grad = input_grad.into();
		let filter_grad = filter_grad.into();
		ConvBack {
			padding: Padding::Same,
			input,
			filter,
			output_grad,
			input_grad,
			filter_grad,
			lowering_memory: 384 * 1024,
		}
	}

	/// Padding determines the shape of the output with respect to the input
	///
	/// Default: `Padding::Same`
	pub fn padding(mut self, padding: Padding) -> Self {
		self.padding = padding;
		self
	}

	/// Sets the limit on how large a lowering operation block can be before splitting it.
	///
	/// Convolution is implemented by converting a section of the input into a matrix (lowering).
	/// Generally this should fit within the per core data cache for good performance.
	///
	/// This may be exceeded if there is no smaller way to perform the lowering.
	pub fn lowering_memory(mut self, lowering_memory: usize) -> Self {
		self.lowering_memory = lowering_memory;
		self
	}
}

impl OpSpecification for ConvBack {
	type InstanceType = ConvBackInstance;

	fn op_type(&self) -> &'static str {
		"ConvBackward"
	}

	/// Returns a list of `Node`s this `Op` may need to read when executed
	fn inputs(&self) -> IndexSet<Node> {
		indexset![self.input.clone(), self.filter.clone(), self.output_grad.clone()]
	}

	/// Returns a list of `Node`s this `Op` may need to write to when executed
	fn outputs(&self) -> IndexSet<Node> {
		indexset![self.input_grad.clone(), self.filter_grad.clone()]
	}

	fn clone_with_nodes_changed(&self, mapping: &IndexMap<Node, Node>) -> Self {
		Self {
			padding: self.padding.clone(),
			input: mapping.get(&self.input).unwrap_or(&self.input).clone(),
			filter: mapping.get(&self.filter).unwrap_or(&self.filter).clone(),
			output_grad: mapping.get(&self.output_grad).unwrap_or(&self.output_grad).clone(),
			input_grad: mapping.get(&self.input_grad).unwrap_or(&self.input_grad).clone(),
			filter_grad: mapping.get(&self.filter_grad).unwrap_or(&self.filter_grad).clone(),

			lowering_memory: self.lowering_memory,
		}
	}

	fn build_instance(self) -> Result<Self::InstanceType, OpBuildError> {
		// TODO check shape lengths
		if self.input.shape().len() < 3 {
			return Err(format!(
				"Node shapes should be of length 3 or greater (minimum 1 spatial dimension), input: {}",
				self.input.shape().len()
			)
			.into());
		}
		if self.input.shape().len() != self.filter.shape().len() {
			return Err(format!(
				"Node shapes should all have the same length: input: {}, filter: {}",
				self.input.shape().len(),
				self.filter.shape().len()
			)
			.into());
		}
		if self.input.shape().len() != self.output_grad.shape().len() {
			return Err(format!(
				"Node shapes should all have the same length: input: {}, output_grad: {}",
				self.input.shape().len(),
				self.output_grad.shape().len()
			)
			.into());
		}
		if self.input.shape().len() != self.input_grad.shape().len() {
			return Err(format!(
				"Node shapes should all have the same length: input: {}, input_grad: {}",
				self.input.shape().len(),
				self.input_grad.shape().len()
			)
			.into());
		}
		if self.input.shape().len() != self.filter_grad.shape().len() {
			return Err(format!(
				"Node shapes should all have the same length: input: {}, filter_grad: {}",
				self.input.shape().len(),
				self.filter_grad.shape().len()
			)
			.into());
		}
		Ok(ConvBackInstance {
			padding: self.padding,

			input: self.input.id(),
			filter: self.filter.id(),
			output_grad: self.output_grad.id(),

			input_grad: self.input_grad.id(),
			filter_grad: self.filter_grad.id(),

			lowering_memory: self.lowering_memory,
		})
	}
}

#[derive(Debug, Clone)]
pub struct ConvBackInstance {
	padding: Padding,
	input: NodeID,
	filter: NodeID,
	output_grad: NodeID,

	input_grad: NodeID,
	filter_grad: NodeID,

	lowering_memory: usize,
}

impl OpInstance for ConvBackInstance {
	fn op_type(&self) -> &'static str {
		"ConvBackward"
	}

	fn as_specification(&self, graph: &Graph) -> Box<dyn Any> {
		Box::new(ConvBack {
			padding: self.padding.clone(),
			input: graph.node_from_id(self.input),
			filter: graph.node_from_id(self.filter),
			output_grad: graph.node_from_id(self.output_grad),
			input_grad: graph.node_from_id(self.input_grad),
			filter_grad: graph.node_from_id(self.filter_grad),
			lowering_memory: self.lowering_memory,
		})
	}

	fn inputs(&self) -> IndexSet<NodeID> {
		indexset![self.input, self.filter, self.output_grad]
	}

	fn outputs(&self) -> IndexSet<NodeID> {
		indexset![self.input_grad, self.filter_grad]
	}

	fn gradient(&self, _ctx: &mut GradientContext) -> Result<(), GradientError> {
		Err(GradientError::Unimplemented)
	}

	fn propagate_shapes(&self, _ctx: &mut ShapePropContext) -> Result<(), ShapePropError> {
		// TODO checks?
		Ok(())
	}

	fn execute(&self, ctx: &ExecutionContext) -> Result<(), ExecutionError> {
		let input = ctx.get_input_standard(&self.input);
		let filter = ctx.get_input_standard(&self.filter);
		let output_grad = ctx.get_input_standard(&self.output_grad);

		let n = input.shape()[0]; // TODO use ensure to guard against zero length shapes
		let _in_size: usize = input.shape()[1..].iter().product();
		let out_size: usize = output_grad.shape()[1..].iter().product();

		let input_spatial = &input.shape()[1..input.ndim() - 1];
		let filter_spatial = &filter.shape()[0..filter.ndim() - 2];
		let output_spatial = output_grad.shape()[1..output_grad.ndim() - 1].to_vec();

		let input_channels = input.shape()[input.ndim() - 1];
		let output_channels = output_grad.shape()[output_grad.ndim() - 1];
		let patch_size = filter_spatial.iter().product::<usize>() * output_channels;

		let in_spaxels: usize = input_spatial.iter().product();
		let _out_spaxels: usize = output_spatial.iter().product();

		// Checks for things that should have already been caught
		assert!(
			input.ndim() == output_grad.ndim(),
			"Input len does not match output len"
		);
		assert!(input.ndim() == filter.ndim(), "Filter len does not match input len");

		assert!(
			input.shape()[0] == output_grad.shape()[0],
			"Batch size of input does not match batch size of output"
		);
		assert!(
			input_channels == filter.shape()[filter.ndim() - 2],
			"input channels dimension does not match final filter dimension"
		);
		assert!(
			output_channels == filter.shape()[filter.ndim() - 1],
			"output channels dimension does not match first filter dimension"
		);

		let input = input.as_slice().unwrap();
		let output_grad = output_grad.as_slice().unwrap();

		let require_filter_gradients = ctx.is_required_output(&self.filter_grad);
		let input_grad = if ctx.is_required_output(&self.input_grad) {
			Some(ctx.get_output_standard(&self.input_grad))
		} else {
			None
		};
		let filter_grad = if ctx.is_required_output(&self.filter_grad) {
			Some(ctx.get_output_standard(&self.filter_grad))
		} else {
			None
		};

		let input_grad_slice = input_grad.as_ref().map(|ig| ig.as_slice().unwrap());

		let filter_strides = stride_vec(output_channels, filter_spatial);
		let input_strides = stride_vec(input_channels, input_spatial);
		let output_strides = stride_vec(output_channels, &output_spatial);

		let mut inverted_filter;
		unsafe {
			inverted_filter = ArrayD::zeros(
				filter_spatial
					.iter()
					.cloned()
					.chain(once(output_channels).chain(once(input_channels)))
					.collect::<Vec<_>>(),
			);
			rot180_assign(
				filter.as_slice().unwrap(),
				inverted_filter.as_slice_mut().unwrap(),
				input_channels,
				output_channels,
			);
		}
		let inverted_filter_slice = inverted_filter.as_slice().unwrap();

		let n_threads = *NUM_CPUS;
		let mut inverted_filter_grads = vec![None; n_threads];

		let cache_limit = self.lowering_memory / (patch_size * size_of::<f32>());
		let thread_division = (in_spaxels * n + n_threads - 1) / n_threads;
		let spaxels_per_batch = min(max(4, min(cache_limit, thread_division)), in_spaxels * n); // number of spaxels to combine in one sgemm (the last batch can have fewer)
		let n_batches = (in_spaxels * n + spaxels_per_batch - 1) / spaxels_per_batch;
		let batch_atomic = AtomicUsize::new(0);

		let n_blocks = filter_spatial.iter().product::<usize>();

		let pool = THREAD_POOL.lock().expect("Could not lock conv threadpool");
		scope_with(&pool, |scope| {
			for inverted_filter_grad in inverted_filter_grads.iter_mut() {
				let inverted_filter_shape = inverted_filter.shape();
				let filter_strides = &filter_strides;
				let input_strides = &input_strides;
				let output_strides = &output_strides;
				let output_spatial = &output_spatial;
				let batch_atomic = &batch_atomic;

				scope.execute(move || {
					let mut patches_alloc = vec![0.0; patch_size * spaxels_per_batch];

					let mut inverted_filter_grad_temp: ArrayD<f32> = if require_filter_gradients {
						ArrayD::zeros(inverted_filter_shape)
					} else {
						ArrayD::default(IxDyn(&[]))
					};
					let inverted_filter_grad_slice = inverted_filter_grad_temp.as_slice_mut().unwrap();

					loop {
						let batch = batch_atomic.fetch_add(1, Ordering::Relaxed);
						if batch >= n_batches {
							break;
						}

						let spaxel_ind = batch * spaxels_per_batch;
						let batch_spaxels = min(in_spaxels * n - spaxel_ind, spaxels_per_batch);

						let patches = &mut patches_alloc[..batch_spaxels * patch_size];
						for (i, patch) in patches.chunks_mut(patch_size).enumerate() {
							debug_assert_eq!(patch_size, patch.len());
							let n_ind = (spaxel_ind + i) / in_spaxels;

							let outg_n = &output_grad[n_ind * out_size..][..out_size];

							let input_ind = (spaxel_ind + i) % in_spaxels * input_channels;
							match filter_spatial.len() {
								1 => unsafe_pack_specialised::<U1>(
									patch,
									outg_n,
									output_channels,
									input_ind,
									filter_spatial,
									output_spatial,
									input_spatial,
									filter_strides,
									output_strides,
									input_strides,
								),
								2 => unsafe_pack_specialised::<U2>(
									patch,
									outg_n,
									output_channels,
									input_ind,
									filter_spatial,
									output_spatial,
									input_spatial,
									filter_strides,
									output_strides,
									input_strides,
								),
								3 => unsafe_pack_specialised::<U3>(
									patch,
									outg_n,
									output_channels,
									input_ind,
									filter_spatial,
									output_spatial,
									input_spatial,
									filter_strides,
									output_strides,
									input_strides,
								),
								_ => unsafe_pack(
									patch,
									outg_n,
									output_channels,
									input_ind,
									filter_spatial,
									output_spatial,
									input_spatial,
									filter_strides,
									output_strides,
									input_strides,
								),
							}

							// pack_patch_recurse(patch, outd_n, &kernel_shape, output_channels,
							// &output.shape.spatial_dimensions, &input_shape.spatial_dimensions, kernel_shape.len()-1,
							// input_ind, in_size);
						}

						// mult
						let in_b = &input[spaxel_ind * input_channels..][..batch_spaxels * input_channels];

						if let Some(input_grad_slice) = input_grad_slice {
							// let mut input_grad_slice = input_grad.as_slice_mut().unwrap();
							let m1 = input_channels;
							let n1 = batch_spaxels;
							let k1 = patch_size;

							let ind_b =
								&input_grad_slice[spaxel_ind * input_channels..][..batch_spaxels * input_channels];
							// let inverted_filter_slice = inverted_filter.as_slice().unwrap();
							debug_assert_eq!(inverted_filter_slice.len(), k1 * m1);
							debug_assert!(patches.len() >= n1 * k1);
							debug_assert_eq!(ind_b.len(), n1 * m1);
							unsafe {
								// input derivatives
								sgemm(
									m1,
									k1,
									n1,
									1.0,
									inverted_filter_slice.as_ptr(),
									// k1 as isize,
									// 1, // A is params, row major
									1,
									m1 as isize, // A is params, col major
									patches.as_ptr(),
									1,
									k1 as isize, // B, input values, column major
									1.0,
									ind_b.as_ptr() as *mut f32,
									1,
									m1 as isize, // C output values, column major
								);
							}
						}

						if require_filter_gradients {
							let m2 = input_channels;
							let n2 = patch_size;
							let k2 = batch_spaxels;

							debug_assert_eq!(in_b.len(), k2 * m2);
							debug_assert!(patches.len() >= n2 * k2);
							debug_assert_eq!(inverted_filter_grad_slice.len(), n2 * m2);
							unsafe {
								// parameter derivatives
								sgemm(
									m2,
									k2,
									n2,
									1.0,
									in_b.as_ptr(),
									1,
									m2 as isize, // A is input image, col major
									patches.as_ptr(),
									n2 as isize,
									1, // B, derivative patches, row major
									1.0,
									inverted_filter_grad_slice.as_mut_ptr(),
									// n2 as isize,
									// 1, // C shuffled parameter derivatives, row major
									1,
									m2 as isize, // C shuffled parameter derivatives, col major
								);
							}
						}
					}
					if require_filter_gradients {
						*inverted_filter_grad = Some(inverted_filter_grad_temp);
					}
				});
			}
		});

		if let Some(filter_grad) = filter_grad {
			let ch2 = input_channels;
			let ch1 = output_channels;
			let block_size = ch1 * ch2;
			debug_assert!(filter_grad.is_standard_layout());
			debug_assert!(filter_grad.len() % (block_size) == 0);
			let block_atomic = AtomicUsize::new(0);
			let output = filter_grad.as_slice().unwrap(); // first inverted grad;

			let (first_ifg, other_ifg) = inverted_filter_grads.as_slice().split_first().unwrap();
			let first_ifg = first_ifg.as_ref().unwrap();

			scope_with(&pool, |scope| {
				scope.join_all();
				// write and invert first block to output block

				for _ in 0..min(n_blocks, n_threads) {
					let block_atomic = &block_atomic;

					debug_assert!(first_ifg.is_standard_layout());
					debug_assert!(first_ifg.len() % (block_size) == 0);

					scope.execute(move || {
						loop {
							let block_num = block_atomic.fetch_add(1, Ordering::Relaxed);
							if block_num >= n_blocks {
								break;
							}
							let input_offset = (n_blocks - block_num - 1) * block_size;
							let output_offset = block_num * block_size;

							unsafe {
								// add all blocks into first
								let inverted_grad_sum = first_ifg.as_ptr().add(input_offset) as *mut f32;
								for ifg in other_ifg {
									let inverted_grad_i = ifg.as_ref().unwrap().as_ptr().add(input_offset);

									debug_assert!(ifg.as_ref().unwrap().is_standard_layout());
									for j in 0..block_size {
										*inverted_grad_sum.add(j) += *inverted_grad_i.add(j);
									}
								}

								// write into output with inversion
								let output = output.as_ptr().add(output_offset) as *mut f32;
								for c2 in 0..ch2 {
									for c1 in 0..ch1 {
										*output.add(c1 + c2 * ch1) = *inverted_grad_sum.add(c2 + c1 * ch2);
									}
								}
							}
						}
					})
				}
			});
		}
		Ok(())
	}
}

/// Converts from [H, W, C1, C2] to [-H, -W, C2, C1] where negatives represent an axis reversal
///
/// Overwrites values of output (they can be uninitialized).
///
/// Unsafe as checks are only performed during debug builds.
unsafe fn rot180_assign(input: &[f32], output: &mut [f32], ch1: usize, ch2: usize) {
	let block_size = ch1 * ch2;
	let num_blocks = input.len() / (block_size);
	debug_assert_eq!(input.len(), output.len());
	debug_assert!(input.len() % (block_size) == 0);

	for b in 0..num_blocks {
		let input_offset = (num_blocks - b - 1) * block_size;
		let output_offset = b * block_size;
		for c2 in 0..ch2 {
			for c1 in 0..ch1 {
				//output[c1 + c2*ch1] = input[c2 + c1*ch2];
				*ui::get_unchecked_mut(output, output_offset + c1 + c2 * ch1) =
					*ui::get_unchecked(input, input_offset + c2 + c1 * ch2);
			}
		}
	}
}

/// A recursive N-dimensional im2col like function.
/// Packs data from a rectangular region of 'input' into 'patch'.
/// Inner recursions deal with lower dimensional slices of the 'input' and the 'patch' recursing until it is reduced to
/// a 1D memcpy.
///
/// # Arguments
/// * `patch` - a rectangular region of input, of shape Cin.ks[0]...ks[axis]; Each recursion removes the outermost
///   spatial dimension.
/// * `input` - input 'image' of shape Cin.W.H
/// * `patch_shape` - Spatial dimensions of the patch, in spaxels
/// * `n_channels` - The number of channels in the 'input' and 'patch' i.e. Cin
/// * `input_shape` - Spatial dimensions of the input, in spaxels
/// * `output_shape` - Spatial dimensions of the output, in spaxels
/// * `axis` - current axis being iterated over. This should be ks.len() - 1 for root call. Reduces by 1 each recursion.
/// * `output_ind` - Index of output spaxel on which the patch is centred. Note: index is the slice index not spaxel
///   index (factor of Cin difference)
/// * `old_out_stride` - Slice stride of output for the layer bove the current iteration. used for interpreting
///   `output_ind`. Root call should be output.len()
#[allow(unused, clippy::too_many_arguments)]
fn pack_patch_recurse(
	patch: &mut [f32],
	input: &[f32],
	patch_shape: &[usize],
	n_channels: usize,
	input_shape: &[usize],
	output_shape: &[usize],
	axis: usize,
	output_ind: usize,
	old_out_stride: usize,
) {
	// stride in array index, not spaxel index
	let out_stride = old_out_stride / output_shape[axis];
	let in_stride = input.len() / input_shape[axis];
	let ks_stride = patch.len() / patch_shape[axis];

	// coordinates of the centre spaxel of the kernel, for the current axis, for both the output and the kernel itself
	let ox = (output_ind % old_out_stride) / out_stride;
	let ix = ox as isize + (input_shape[axis] as isize - output_shape[axis] as isize) / 2;

	// valid range of the kernels in the current axis
	let (start, end) = kernel_range(ix, input_shape[axis], patch_shape[axis]);

	for el in &mut patch[0..start * ks_stride] {
		*el = 0.0; // fill zeros
	}

	if axis > 0 {
		for i in start..end {
			let ix = (ix + i as isize - (patch_shape[axis] / 2) as isize) as usize; // shadow ix is the coordinate for the current iteration, rather than the centre of the kernel.

			let new_input = &input[in_stride * ix..in_stride * (ix + 1)];
			let new_patch = &mut patch[i * ks_stride..(i + 1) * ks_stride];
			let new_axis = axis - 1;

			pack_patch_recurse(
				new_patch,
				new_input,
				patch_shape,
				n_channels,
				input_shape,
				output_shape,
				new_axis,
				output_ind,
				out_stride,
			);
		}
	} else {
		let offset =
			((ix - patch_shape[axis] as isize / 2) * n_channels as isize + (start * n_channels) as isize) as usize;
		let len = (end - start) * n_channels;
		let input_crop = &input[offset..][..len];
		let mut patch_crop = &mut patch[(start * n_channels)..][..len];

		patch_crop.copy_from_slice(input_crop);
	}

	for el in &mut patch[(end * ks_stride)..(patch_shape[axis] * ks_stride)] {
		*el = 0.0; // fill zero
	}
}

// /// returns a vector with the array stride of each dimension. output[0] == channel.
// fn stride_vec(channels: usize, shape: &[usize]) -> Vec<usize>{
// 	once(&channels).chain(shape.iter()).scan(1, |state, &i| {*state *= i; Some(*state)}).collect::<Vec<usize>>()
// }

/// returns a vector with the array stride of each dimension. output[n] == channel.
fn stride_vec(channels: usize, shape: &[usize]) -> SmallVec<[usize; 6]> {
	let mut strides = once(&channels)
		.chain(shape.iter().rev())
		.scan(1, |state, &i| {
			let res = Some(*state);
			*state *= i;
			res
		})
		.collect::<SmallVec<[usize; 6]>>();
	strides.reverse();
	strides
}

#[allow(clippy::too_many_arguments)]
fn unsafe_pack_specialised<Axes: PackSpecialised + Unsigned>(
	patch: &mut [f32],
	input: &[f32],
	channels: usize,
	output_ind: usize,
	kernel_shape: &[usize],
	input_shape: &[usize],
	output_shape: &[usize],
	kernel_strides: &[usize],
	input_strides: &[usize],
	output_strides: &[usize],
) {
	debug_assert_eq!(Axes::to_usize(), kernel_shape.len());

	let axis = 0;
	let ox = output_ind / output_strides[axis];
	let ix = ox as isize + (input_shape[axis] as isize - output_shape[axis] as isize) / 2;
	let (start, end) = kernel_range(ix, input_shape[axis], kernel_shape[axis]);

	unsafe {
		Axes::pack(
			patch,
			input,
			channels,
			output_ind,
			ox,
			ix,
			start,
			end,
			kernel_shape,
			input_shape,
			output_shape,
			kernel_strides,
			input_strides,
			output_strides,
		)
	};
}

trait PackSpecialised {
	#[allow(clippy::too_many_arguments)]
	unsafe fn pack(
		patch: &mut [f32],
		input: &[f32],
		channels: usize,
		output_ind: usize,
		ox: usize,
		ix: isize,
		start: usize,
		end: usize, // valid range of the kernels in the current axis
		kernel_shape: &[usize],
		input_shape: &[usize],
		output_shape: &[usize],
		kernel_strides: &[usize],
		input_strides: &[usize],
		output_strides: &[usize],
	);
}

impl<U: Unsigned, B: Bit, C: Bit> PackSpecialised for UInt<UInt<U, B>, C>
where
	UInt<UInt<U, B>, C>: Sub<B1>,
	Sub1<UInt<UInt<U, B>, C>>: PackSpecialised,
{
	#[inline(always)]
	unsafe fn pack(
		patch: &mut [f32],
		input: &[f32],
		channels: usize,
		output_ind: usize,
		ox: usize,
		ix: isize,
		start: usize,
		end: usize, // valid range of the kernels in the current axis
		kernel_shape: &[usize],
		input_shape: &[usize],
		output_shape: &[usize],
		kernel_strides: &[usize],
		input_strides: &[usize],
		output_strides: &[usize],
	) {
		let axis = kernel_shape.len() - Self::to_usize();

		let i_stride = *ui::get_unchecked(input_strides, axis);
		let o_stride = *ui::get_unchecked(output_strides, axis);
		let k_stride = *ui::get_unchecked(kernel_strides, axis);
		// coordinates of the centre spaxel of the kernel, for the current axis, for both the output and the kernel
		// itself

		for i in 0..start * k_stride {
			*ui::get_unchecked_mut(patch, i) = 0.0; // fill zero
		}

		debug_assert!(axis < kernel_shape.len() - 1);
		for i in start..end {
			let temp_ix = (ix + i as isize - (*ui::get_unchecked(kernel_shape, axis) / 2) as isize) as usize; // temp_ix is the coordinate for the current iteration, rather than the centre of the kernel.
			debug_assert!(ix + i as isize - (kernel_shape[axis] / 2) as isize >= 0);

			let new_input = ui::get_unchecked(input, i_stride * temp_ix..i_stride * (temp_ix + 1));

			let new_patch = ui::get_unchecked_mut(patch, i * k_stride..(i + 1) * k_stride);

			let new_axis = axis + 1;
			let new_output_ind = output_ind - ox * o_stride;
			let new_ox = new_output_ind / *ui::get_unchecked(output_strides, new_axis);
			let new_ix = new_ox as isize
				+ (*ui::get_unchecked(input_shape, new_axis) as isize
					- *ui::get_unchecked(output_shape, new_axis) as isize)
					/ 2;
			let (new_start, new_end) = kernel_range(
				new_ix,
				*ui::get_unchecked(input_shape, new_axis),
				*ui::get_unchecked(kernel_shape, new_axis),
			); // input_shape[new_axis]    kernel_shape[new_axis]);

			<Sub1<Self>>::pack(
				new_patch,
				new_input,
				channels,
				new_output_ind,
				new_ox,
				new_ix,
				new_start,
				new_end,
				kernel_shape,
				input_shape,
				output_shape,
				kernel_strides,
				input_strides,
				output_strides,
			)
		}

		for i in (end * k_stride)..(*ui::get_unchecked(kernel_shape, axis) * k_stride) {
			*ui::get_unchecked_mut(patch, i) = 0.0; // fill zero
		}
	}
}

impl PackSpecialised for UInt<UTerm, B1> {
	#[inline(always)]
	unsafe fn pack(
		patch: &mut [f32],
		input: &[f32],
		channels: usize,
		_output_ind: usize,
		_ox: usize,
		ix: isize,
		start: usize,
		end: usize, // valid range of the kernels in the current axis
		kernel_shape: &[usize],
		_input_shape: &[usize],
		_output_shape: &[usize],
		kernel_strides: &[usize],
		_input_strides: &[usize],
		_output_strides: &[usize],
	) {
		let axis = kernel_shape.len() - Self::to_usize();

		let k_stride = *ui::get_unchecked(kernel_strides, axis);
		// coordinates of the centre spaxel of the kernel, for the current axis, for both the output and the kernel
		// itself

		for i in 0..start * k_stride {
			*ui::get_unchecked_mut(patch, i) = 0.0; // fill zero
		}

		if end > start {
			let offset = ((ix - *ui::get_unchecked(kernel_shape, axis) as isize / 2) * channels as isize
				+ (start * channels) as isize) as usize;
			let len = (end - start) * channels;

			// let input_crop = &input[offset..][..len];
			// let mut patch_crop = &mut patch[(start*channels)..][..len];
			// patch_crop.copy_from_slice(input_crop);
			let input_crop = ui::get_unchecked(input, offset..offset + len);
			let patch_crop = ui::get_unchecked_mut(patch, start * channels..start * channels + len);
			for i in 0..len {
				*ui::get_unchecked_mut(patch_crop, i) = *ui::get_unchecked(input_crop, i);
			}
		}

		for i in (end * k_stride)..(*ui::get_unchecked(kernel_shape, axis) * k_stride) {
			*ui::get_unchecked_mut(patch, i) = 0.0; // fill zero
		}
	}
}

#[allow(clippy::too_many_arguments)]
fn unsafe_pack(
	patch: &mut [f32],
	input: &[f32],
	channels: usize,
	output_ind: usize,
	kernel_shape: &[usize],
	input_shape: &[usize],
	output_shape: &[usize],
	kernel_strides: &[usize],
	input_strides: &[usize],
	output_strides: &[usize],
) {
	let axis = 0;

	let ox = output_ind / output_strides[axis];
	let ix = ox as isize + (input_shape[axis] as isize - output_shape[axis] as isize) / 2;
	let (start, end) = kernel_range(ix, input_shape[axis], kernel_shape[axis]);

	unsafe {
		_unsafe_pack_impl(
			patch,
			input,
			channels,
			axis,
			output_ind,
			ox,
			ix,
			start,
			end,
			kernel_shape,
			input_shape,
			output_shape,
			kernel_strides,
			input_strides,
			output_strides,
		)
	};
}

#[allow(clippy::too_many_arguments)]
unsafe fn _unsafe_pack_impl(
	patch: &mut [f32],
	input: &[f32],
	channels: usize,
	axis: usize,
	output_ind: usize,
	ox: usize,
	ix: isize,
	start: usize,
	end: usize, // valid range of the kernels in the current axis
	kernel_shape: &[usize],
	input_shape: &[usize],
	output_shape: &[usize],
	kernel_strides: &[usize],
	input_strides: &[usize],
	output_strides: &[usize],
) {
	// println!("a:{} s:{} e:{} ind: {} ox:{}, ix:{}", axis, start, end, output_ind, ox, ix);

	let i_stride = *ui::get_unchecked(input_strides, axis);
	let o_stride = *ui::get_unchecked(output_strides, axis);
	let k_stride = *ui::get_unchecked(kernel_strides, axis);
	// coordinates of the centre spaxel of the kernel, for the current axis, for both the output and the kernel itself

	for i in 0..start * k_stride {
		*ui::get_unchecked_mut(patch, i) = 0.0; // fill zero
	}

	if axis < kernel_shape.len() - 1 {
		for i in start..end {
			let temp_ix = (ix + i as isize - (*ui::get_unchecked(kernel_shape, axis) / 2) as isize) as usize; // temp_ix is the coordinate for the current iteration, rather than the centre of the kernel.
			debug_assert!(ix + i as isize - (kernel_shape[axis] / 2) as isize >= 0);

			let new_input = ui::get_unchecked(input, i_stride * temp_ix..i_stride * (temp_ix + 1));

			let new_patch = ui::get_unchecked_mut(patch, i * k_stride..(i + 1) * k_stride);

			let new_axis = axis + 1;
			let new_output_ind = output_ind - ox * o_stride;
			let new_ox = new_output_ind / *ui::get_unchecked(output_strides, new_axis);
			let new_ix = new_ox as isize
				+ (*ui::get_unchecked(input_shape, new_axis) as isize
					- *ui::get_unchecked(output_shape, new_axis) as isize)
					/ 2;
			let (new_start, new_end) = kernel_range(
				new_ix,
				*ui::get_unchecked(input_shape, new_axis),
				*ui::get_unchecked(kernel_shape, new_axis),
			); // input_shape[new_axis]    kernel_shape[new_axis]);

			_unsafe_pack_impl(
				new_patch,
				new_input,
				channels,
				new_axis,
				new_output_ind,
				new_ox,
				new_ix,
				new_start,
				new_end,
				kernel_shape,
				input_shape,
				output_shape,
				kernel_strides,
				input_strides,
				output_strides,
			)
		}
	} else if end > start {
		let offset = ((ix - *ui::get_unchecked(kernel_shape, axis) as isize / 2) * channels as isize
			+ (start * channels) as isize) as usize;
		let len = (end - start) * channels;

		// let input_crop = &input[offset..][..len];
		// let mut patch_crop = &mut patch[(start*channels)..][..len];
		// patch_crop.copy_from_slice(input_crop);
		let input_crop = ui::get_unchecked(input, offset..offset + len);
		let patch_crop = ui::get_unchecked_mut(patch, start * channels..start * channels + len);
		for i in 0..len {
			*ui::get_unchecked_mut(patch_crop, i) = *ui::get_unchecked(input_crop, i);
		}
	}

	for i in (end * k_stride)..(*ui::get_unchecked(kernel_shape, axis) * k_stride) {
		*ui::get_unchecked_mut(patch, i) = 0.0; // fill zero
	}
}

/// returns the [start, end) range indicies of the kernel which overlap with the 'image'
/// returns [start, end) of the kernel range which overlaps with the image, given the width of the image and the
/// position of the center of the kernel only odd kernels are valid.
fn kernel_range(center: isize, width: usize, kernel_width: usize) -> (usize, usize) {
	debug_assert!(kernel_width % 2 == 1);
	(
		min(kernel_width as isize, max(0, kernel_width as isize / 2 - center)) as usize,
		max(
			0,
			kernel_width as isize - max(0, center + kernel_width as isize / 2 + 1 - width as isize),
		) as usize,
	)
}

#[cfg(test)]
mod tests {
	use super::{conv, conv_with, kernel_range, stride_vec, unsafe_pack, unsafe_pack_specialised, Padding};

	use alumina_core::{graph::Node, init::msra};
	use alumina_test::{grad_numeric_test::GradNumericTest, relatively_close::RelClose};

	use indexmap::indexset;
	use ndarray::{arr2, arr3, s, ArrayD};
	use typenum::U2;

	#[test]
	fn test_forward() {
		let input = Node::new(&[1, 8, 10, 3]).set_name("input").set_value(arr3(&[
			[
				[6.0, 4.0, 0.0],
				[7.0, 7.0, 7.0],
				[2.0, 10.0, 3.0],
				[1.0, 2.0, 5.0],
				[7.0, 10.0, 3.0],
				[6.0, 6.0, 6.0],
				[10.0, 4.0, 6.0],
				[1.0, 6.0, 0.0],
				[7.0, 2.0, 0.0],
				[1.0, 9.0, 6.0],
			],
			[
				[1.0, 7.0, 4.0],
				[1.0, 10.0, 1.0],
				[0.0, 2.0, 9.0],
				[5.0, 9.0, 0.0],
				[0.0, 2.0, 4.0],
				[10.0, 7.0, 5.0],
				[9.0, 8.0, 1.0],
				[6.0, 1.0, 4.0],
				[7.0, 3.0, 8.0],
				[4.0, 7.0, 3.0],
			],
			[
				[4.0, 5.0, 5.0],
				[3.0, 0.0, 2.0],
				[8.0, 5.0, 8.0],
				[0.0, 1.0, 3.0],
				[6.0, 5.0, 8.0],
				[7.0, 0.0, 7.0],
				[10.0, 3.0, 8.0],
				[6.0, 7.0, 6.0],
				[1.0, 5.0, 5.0],
				[0.0, 3.0, 6.0],
			],
			[
				[9.0, 5.0, 5.0],
				[5.0, 5.0, 3.0],
				[2.0, 9.0, 5.0],
				[1.0, 5.0, 3.0],
				[9.0, 0.0, 0.0],
				[9.0, 6.0, 6.0],
				[7.0, 7.0, 6.0],
				[0.0, 6.0, 1.0],
				[5.0, 6.0, 6.0],
				[1.0, 0.0, 10.0],
			],
			[
				[8.0, 1.0, 9.0],
				[7.0, 1.0, 0.0],
				[0.0, 1.0, 2.0],
				[2.0, 1.0, 10.0],
				[4.0, 6.0, 10.0],
				[2.0, 0.0, 0.0],
				[4.0, 1.0, 1.0],
				[4.0, 8.0, 4.0],
				[6.0, 5.0, 8.0],
				[2.0, 6.0, 4.0],
			],
			[
				[2.0, 7.0, 3.0],
				[0.0, 8.0, 10.0],
				[9.0, 1.0, 5.0],
				[6.0, 1.0, 4.0],
				[1.0, 9.0, 1.0],
				[8.0, 2.0, 2.0],
				[4.0, 9.0, 3.0],
				[10.0, 4.0, 9.0],
				[8.0, 2.0, 6.0],
				[6.0, 3.0, 4.0],
			],
			[
				[3.0, 5.0, 3.0],
				[1.0, 6.0, 7.0],
				[3.0, 10.0, 1.0],
				[8.0, 6.0, 5.0],
				[6.0, 0.0, 7.0],
				[0.0, 5.0, 3.0],
				[3.0, 7.0, 1.0],
				[8.0, 2.0, 6.0],
				[1.0, 7.0, 8.0],
				[6.0, 2.0, 1.0],
			],
			[
				[10.0, 10.0, 10.0],
				[1.0, 1.0, 0.0],
				[1.0, 2.0, 5.0],
				[1.0, 3.0, 9.0],
				[9.0, 8.0, 7.0],
				[7.0, 4.0, 7.0],
				[9.0, 10.0, 2.0],
				[1.0, 0.0, 1.0],
				[5.0, 2.0, 10.0],
				[2.0, 5.0, 7.0],
			],
		]));

		let filter = Node::new(&[3, 5, 3, 1]).set_name("filter").set_value(
			arr3(&[
				[
					[3.0, 2.0, 3.0],
					[4.0, 4.0, 9.0],
					[2.0, 6.0, 0.0],
					[8.0, 2.0, 8.0],
					[0.0, 6.0, 2.0],
				],
				[
					[5.0, 10.0, 6.0],
					[0.0, 0.0, 5.0],
					[2.0, 5.0, 1.0],
					[7.0, 3.0, 7.0],
					[2.0, 4.0, 5.0],
				],
				[
					[6.0, 3.0, 0.0],
					[4.0, 5.0, 7.0],
					[3.0, 10.0, 1.0],
					[1.0, 1.0, 5.0],
					[9.0, 0.0, 10.0],
				],
			])
			.into_shape([3, 5, 3, 1])
			.unwrap(),
		);

		let expected_full: ArrayD<f32> = arr2(&[
			[
				54.0, 143.0, 155.0, 228.0, 390.0, 316.0, 407.0, 248.0, 314.0, 297.0, 187.0, 161.0, 139.0, 33.0,
			],
			[
				77.0, 178.0, 393.0, 419.0, 450.0, 721.0, 616.0, 584.0, 670.0, 623.0, 446.0, 341.0, 208.0, 176.0,
			],
			[
				160.0, 296.0, 598.0, 536.0, 837.0, 939.0, 956.0, 965.0, 793.0, 1001.0, 703.0, 414.0, 325.0, 156.0,
			],
			[
				234.0, 324.0, 434.0, 678.0, 765.0, 843.0, 935.0, 747.0, 1044.0, 985.0, 685.0, 499.0, 359.0, 107.0,
			],
			[
				265.0, 361.0, 379.0, 673.0, 725.0, 732.0, 851.0, 836.0, 789.0, 902.0, 736.0, 468.0, 382.0, 119.0,
			],
			[
				153.0, 422.0, 550.0, 673.0, 730.0, 708.0, 739.0, 849.0, 956.0, 900.0, 634.0, 525.0, 408.0, 172.0,
			],
			[
				128.0, 384.0, 414.0, 651.0, 926.0, 693.0, 693.0, 863.0, 859.0, 772.0, 704.0, 589.0, 304.0, 156.0,
			],
			[
				279.0, 319.0, 518.0, 716.0, 971.0, 773.0, 903.0, 777.0, 1086.0, 816.0, 693.0, 517.0, 364.0, 119.0,
			],
			[
				146.0, 284.0, 299.0, 359.0, 706.0, 556.0, 566.0, 517.0, 662.0, 496.0, 494.0, 271.0, 222.0, 127.0,
			],
			[
				80.0, 186.0, 112.0, 266.0, 250.0, 264.0, 369.0, 315.0, 297.0, 314.0, 170.0, 158.0, 140.0, 37.0,
			],
		])
		.into_shape([1, 10, 14, 1])
		.unwrap()
		.into_dyn();

		#[allow(clippy::deref_addrof)] // cant fix s!
		let expected_same = expected_full.slice(s![.., 1..9, 2..12, ..]);

		#[allow(clippy::deref_addrof)] // cant fix s!
		let expected_valid = expected_full.slice(s![.., 2..8, 4..10, ..]);

		let output_full = conv_with(&input, &filter, Padding::Full)
			.unwrap()
			.set_name("output_full");
		let output_same = conv_with(&input, &filter, Padding::Same)
			.unwrap()
			.set_name("output_same");
		let output_valid = conv_with(&input, &filter, Padding::Valid)
			.unwrap()
			.set_name("output_valid");

		assert!(output_full
			.calc()
			.unwrap()
			.all_relatively_close(&expected_full, ::std::f32::EPSILON));

		assert!(output_same
			.calc()
			.unwrap()
			.all_relatively_close(&expected_same, ::std::f32::EPSILON));

		assert!(output_valid
			.calc()
			.unwrap()
			.all_relatively_close(&expected_valid, ::std::f32::EPSILON));
	}

	#[test]
	fn test_shapes() {
		let input = Node::new(&[3, 5, 7, 13]).set_name("input");

		let output_full = conv(&input, 15, &[3, 5], Padding::Full)
			.unwrap()
			.set_name("output_full");
		let output_same = conv(&input, 15, &[3, 5], Padding::Same)
			.unwrap()
			.set_name("output_same");
		let output_valid = conv(&input, 15, &[3, 5], Padding::Valid)
			.unwrap()
			.set_name("output_valid");

		assert_eq!(output_full.shape(), [3, 7, 11, 15].iter().into());
		assert_eq!(output_same.shape(), [3, 5, 7, 15].iter().into());
		assert_eq!(output_valid.shape(), [3, 3, 3, 15].iter().into());
	}

	#[test]
	fn test_shapes_unknown() {
		let input = Node::new(&[3, -1, -1, 13]).set_name("input");

		let output_full = conv(&input, 15, &[3, 5], Padding::Full)
			.unwrap()
			.set_name("output_full");
		let output_same = conv(&input, 15, &[3, 5], Padding::Same)
			.unwrap()
			.set_name("output_same");
		let output_valid = conv(&input, 15, &[3, 5], Padding::Valid)
			.unwrap()
			.set_name("output_valid");

		assert_eq!(output_full.shape(), [3, -1, -1, 15].iter().into());
		assert_eq!(output_same.shape(), [3, -1, -1, 15].iter().into());
		assert_eq!(output_valid.shape(), [3, -1, -1, 15].iter().into());
	}

	#[test]
	fn test_shapes_unknown_batch() {
		let input = Node::new(&[-1, -1, -1, 13]).set_name("input");

		let output_full = conv(&input, 15, &[3, 5], Padding::Full)
			.unwrap()
			.set_name("output_full");

		assert_eq!(output_full.shape(), [-1, -1, -1, 15].iter().into());
	}

	#[test]
	fn grad_numeric_matmul_fallthrough_test() {
		let input = Node::new(&[2, 5, 7, 13]).set_name("input");
		let filter = Node::new(&[1, 1, 13, 11]).set_name("filter").set_init(msra(1.0));

		let output = conv_with(&input, &filter, Padding::Same).unwrap().set_name("output");

		GradNumericTest::new(&output, &indexset![&input, &filter]).run();
	}

	#[test]
	fn grad_numeric_same_test() {
		let input = Node::new(&[2, 5, 7, 13]).set_name("input");
		let filter = Node::new(&[3, 5, 13, 11]).set_name("filter").set_init(msra(1.0));

		let output = conv_with(&input, &filter, Padding::Same).unwrap().set_name("output");

		GradNumericTest::new(&output, &indexset![&input, &filter]).run();
	}

	#[test]
	fn grad_numeric_valid_test() {
		let input = Node::new(&[2, 5, 7, 13]).set_name("input");
		let filter = Node::new(&[3, 5, 13, 11]).set_name("filter").set_init(msra(1.0));

		let output = conv_with(&input, &filter, Padding::Valid).unwrap().set_name("output");

		GradNumericTest::new(&output, &indexset![&input, &filter]).run();
	}

	#[test]
	fn grad_numeric_full_test() {
		let input = Node::new(&[2, 5, 7, 13]).set_name("input");
		let filter = Node::new(&[3, 5, 13, 11]).set_name("filter").set_init(msra(1.0));

		let output = conv_with(&input, &filter, Padding::Full).unwrap().set_name("output");

		GradNumericTest::new(&output, &indexset![&input, &filter]).run();
	}

	#[test]
	fn test_kernel_range() {
		assert!((0, 1) == kernel_range(0, 1, 1));
		assert!((1, 2) == kernel_range(0, 1, 3));
		assert!((3, 4) == kernel_range(0, 1, 7));

		assert!((3, 3) == kernel_range(-3, 7, 3));
		assert!((3, 3) == kernel_range(-2, 7, 3));
		assert!((2, 3) == kernel_range(-1, 7, 3));
		assert!((1, 3) == kernel_range(0, 7, 3));
		assert!((0, 3) == kernel_range(1, 7, 3));
		assert!((0, 3) == kernel_range(2, 7, 3));

		assert!((0, 3) == kernel_range(5, 7, 3));
		assert!((0, 2) == kernel_range(6, 7, 3));
		assert!((0, 1) == kernel_range(7, 7, 3));
		assert!((0, 0) == kernel_range(8, 7, 3));
		assert!((0, 0) == kernel_range(9, 7, 3));
	}

	#[test]
	fn test_pack() {
		let filter_spatial = vec![3, 5];
		let input_spatial = vec![5, 7];
		let output_spatial = vec![7, 9];
		let input_channels = 3;
		let output_channels = 5;

		let input_spaxel_count: usize = input_spatial.iter().product();
		let output_spaxel_count: usize = output_spatial.iter().product();

		let input_size = input_spaxel_count * input_channels;
		let _output_size = output_spaxel_count * output_channels;
		let patch_size = filter_spatial.iter().fold(input_channels, |p, v| p * v);

		let input: Vec<f32> = (0..input_size).map(|x| x as f32 + 1.0).collect();

		let filter_strides = stride_vec(input_channels, &filter_spatial);
		let input_strides = stride_vec(input_channels, &input_spatial);
		let output_strides = stride_vec(output_channels, &output_spatial);

		let mut patches = vec![-0.5; patch_size * output_spaxel_count];

		for (i, patch) in patches.chunks_mut(patch_size).enumerate() {
			let output_ind = i * output_channels;
			unsafe_pack(
				patch,
				&input,
				input_channels,
				output_ind,
				&filter_spatial,
				&input_spatial,
				&output_spatial,
				&filter_strides,
				&input_strides,
				&output_strides,
			);
		}

		debug_assert!(
			!patches.iter().cloned().any(|x| x.is_nan() || {
				#[allow(clippy::float_cmp)]
				{
					x == -0.5
				}
			}),
			"test1: {:?}",
			patches
		);

		let mut patches = vec![-0.5; patch_size * output_spaxel_count];

		for (i, patch) in patches.chunks_mut(patch_size).enumerate() {
			let output_ind = i * output_channels;
			unsafe_pack_specialised::<U2>(
				patch,
				&input,
				input_channels,
				output_ind,
				&filter_spatial,
				&input_spatial,
				&output_spatial,
				&filter_strides,
				&input_strides,
				&output_strides,
			);
		}

		debug_assert!(
			!patches.iter().cloned().any(|x| x.is_nan() || {
				#[allow(clippy::float_cmp)]
				{
					x == -0.5
				}
			}),
			"test2: {:?}",
			patches
		);

		// for (i, patch) in patches.chunks(patch_size).enumerate(){
		// 	let (y, x) = (i/output_spatial[1], i%output_spatial[1]);
		// 	println!("y:{} x:{} patch:{:?}", y, x, patch);
		// }
		// panic!();
	}
}
