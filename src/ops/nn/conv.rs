use graph::{GraphDef, NodeID, OpID, PassID, DataID, Storage, GraphShapes, Result};
use ops::{standard_op_name, standard_inner_parameter_name, Op, OpInstance, Pass};
use shape::{NodeDim, NodeShape};
use ndarray::{ArrayViewMutD, ArrayD, Dimension, Axis, IxDyn};
use std::any::Any;
use std::iter;
use std::sync::Mutex;
use std::sync::mpsc::sync_channel;
use std::cmp::{min, max};
use std::mem::size_of;
use scoped_threadpool::Pool;
use odds;
use num_cpus;
use matrixmultiply;
use init::Initialiser;
use rand::{thread_rng, Isaac64Rng, Rng};
use rand::distributions::{Sample, Normal};
use smallvec::SmallVec;

/// Threadpool for offloading lowering/packing operations
lazy_static! {
	static ref NUM_CPUS: usize = num_cpus::get();
	static ref THREAD_POOL: Mutex<Pool> = Mutex::new(Pool::new(*NUM_CPUS as u32));
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
/// `[num_output_channels, spatial_filter_shape[0], ..., spatial_filter_shape[N-1], num_input_channels]`
#[must_use]
#[derive(Clone, Debug)]
pub struct Conv {
	name: Option<String>,
	kernel_shape: Vec<usize>,
	padding: Padding,
	input_id: NodeID,
	output_id: NodeID,
	filter_id: Option<NodeID>,
	initialiser: Option<Initialiser>,
	lowering_memory: usize,
}

impl Conv {
	pub fn new(input_id: &NodeID, output_id: &NodeID, kernel_shape: &[usize]) -> Self{
		Conv {
			name: None,
			kernel_shape: kernel_shape.to_vec(),
			padding: Padding::Same,
			input_id: input_id.clone(),
			output_id: output_id.clone(),
			filter_id: None,
			initialiser: None,
			lowering_memory: 1024*1024*2,
		}
	}

	/// Padding determines the shape of the output with respect to the input
	///
	/// Default: `Padding::Same`
	pub fn padding(mut self, padding: Padding) -> Self {
		self.padding = padding;
		self
	}

	/// Provide a node to replace the filter tensor
	///
	/// The expected shape is `Cout.H.W.Cin`
	/// If left as `None` a suitable `Parameter` node will be automatically created.
	///
	/// Default value: `None`
	pub fn filter(mut self, node_id: Option<&NodeID>) -> Self {
		self.filter_id = node_id.cloned();
		self
	}

	pub fn init (mut self, initialiser: Initialiser) -> Self {
		self.initialiser = Some(initialiser);
		self
	}


	/// MSRA/He initialisation
	///
	/// This initialises the parameter filter with gaussian values drawn from N(0, multiplier/K).
	/// Where K is the number of incoming neurons to each outgoing neuron.
	/// For typical use, the variance multiplier should cancel out the variance modifying
	/// effect of the nonlinearity, e.g. use 2.0 with ReLU, and 1.0 with Tanh.
	pub fn msra(multiplier: f32) -> Initialiser {
		Initialiser::new("MSRA Initialiser for Linear Op".to_string(), move |mut arr: ArrayViewMutD<f32>, _instance: Option<&OpInstance>|{
			let k = arr.len()/arr.shape()[0];

			let mut rng = thread_rng().gen::<Isaac64Rng>();
			let mut norm = Normal::new(0.0, (multiplier as f64 / k as f64).sqrt());
			for e in arr.iter_mut() {
				*e = norm.sample(&mut rng) as f32;
			}
		})
	}

	/// Xavier initialisation
	///
	/// This is just msra with a multiplier of 1.0
	pub fn xavier() -> Initialiser {
		Conv::msra(1.0)
	}
}


impl Op for Conv {
	type InstanceType = ConvInstance;

	fn type_name(&self) -> &'static str {
		"Conv"
	}

	fn name<T: Into<String>>(mut self, name: T) -> Self{
		self.name = Some(name.into());
		self
	}

	fn build(self, graph: &mut GraphDef, op_id: &OpID) -> Result<Self::InstanceType> {
		let (name, filter_is_inner) = if let Some(ref filter) = self.filter_id {
			(standard_op_name(&self, &self.name, graph, &[self.input_id.clone(), filter.clone()], &[self.output_id.clone()]), false)
		} else {
			(standard_op_name(&self, &self.name, graph, &[self.input_id.clone()], &[self.output_id.clone()]), true)
		};
		




		let filter = if let Some(filter) = self.filter_id {
			let filter_shape = graph.node_shape(&filter)?.to_data_shape()?;
			let filter_shape = filter_shape.slice();
			//TODO also check in_shape and out_shape
			ensure!(&self.kernel_shape[..] == &filter_shape[1..filter_shape.len()-1], "If a filter node is supplied, it must have a fixed shape");
			filter
		} else {
			let filter_name = standard_inner_parameter_name(&name, graph);
			let shape: NodeShape = {
				let in_shape = graph.node_shape(&self.input_id)?;
				let out_shape = graph.node_shape(&self.output_id)?;

				let c_in = &in_shape.dimensions()[in_shape.ndims()-1];
				let c_out = &out_shape.dimensions()[out_shape.ndims()-1];
				if let (&NodeDim::Known(c_in), &NodeDim::Known(c_out)) = (c_in, c_out) {
					iter::once(c_out).chain(self.kernel_shape.iter().cloned()).chain(iter::once(c_in)).into()
				} else {
					bail!(format!("The channel dimensions (innermost dimensions) of both the input and output must be known so that a fixed sized parameter node can be inferred."));
				}
			};
			graph.new_node(shape, filter_name, tag![Parameter])?
		};


		if let Some(initialiser) = self.initialiser {
			graph.set_initialiser(&filter, initialiser.set_op_id(op_id.clone()));
		};

		Ok(ConvInstance{
			name: name,
			padding: self.padding,
			//kernel_shape: self.kernel_shape.clone(),
			input_id: self.input_id.clone(),
			output_id: self.output_id.clone(),
			filter_id: filter.clone(),
			filter_is_inner: filter_is_inner,
			forward_id: graph.add_pass(ConvForward::new(
				self.input_id.clone(),
				self.output_id.clone(),
				filter.clone(),
				self.lowering_memory,
				//self.kernel_shape.clone(),
			)),
			backward_id: graph.add_pass(ConvBackward::new(
				self.input_id.clone(),
				self.output_id.clone(),
				filter.clone(),
				self.lowering_memory,
				//self.kernel_shape.clone(),
			)),
		})
	}
}


#[derive(Debug, Clone)]
pub struct ConvInstance {
	name: String,
	padding: Padding,
	input_id: NodeID,
	output_id: NodeID,
	filter_id: NodeID,
	filter_is_inner: bool,
	forward_id: PassID,
	backward_id: PassID,
}

impl OpInstance for ConvInstance {
	fn instance_name(&self) -> &str {&self.name}

	fn dependencies(&self) -> (Vec<NodeID>, Vec<NodeID>){
		(
			if self.filter_is_inner {
				vec![self.input_id.clone()]
			} else {
				vec![self.input_id.clone(), self.filter_id.clone()]
			},
			vec![self.output_id.clone()]
		)
	}

	fn inner_passes(&self) -> Vec<PassID> {
		vec![self.forward_id.clone(), self.backward_id.clone()]
	}

	fn inner_ops(&self) -> Vec<OpID> {vec![]}

	fn inner_nodes(&self) -> Vec<NodeID> {
		if self.filter_is_inner {
			vec![self.filter_id.clone()]
		} else {
			vec![]
		}
	}

	fn propagate_shape_constraints(&self, shapes: &mut GraphShapes) -> Result<()>{

		let input_shape = shapes.get_shape(&self.input_id).to_data_shape()?;
		let filter_shape = shapes.get_shape(&self.filter_id).to_data_shape()?;
		let input_shape = input_shape.slice();
		let filter_shape = filter_shape.slice();

		let batch_size = input_shape[0]; //TODO use ensure to guard against zero length shapes
		let out_channels = filter_shape[0];  //TODO use ensure to guard against zero length shapes
		let in_channels = input_shape[input_shape.len()-1];
		ensure!(in_channels == filter_shape[filter_shape.len()-1], format!("input channels dimension {} does not match final filter dimension {}", in_channels, filter_shape[filter_shape.len()-1]));

		let input_spatial = input_shape[1..input_shape.len()-1].iter();
		let filter_spatial = filter_shape[1..filter_shape.len()-1].iter();
		ensure!(input_spatial.len() == filter_spatial.len(), "input shape and filter shape do not hav ethe same number of spatial dimensions");
		


		let output_shape: NodeShape = match self.padding {
			Padding::Full => {
					iter::once(batch_size)
					.chain(input_spatial.zip(filter_spatial).map(|(dim, k_dim)| dim + k_dim - 1))
					.chain(iter::once(out_channels)).into()
				},
			Padding::Same => {
					iter::once(batch_size)
					.chain(input_spatial.cloned())
					.chain(iter::once(out_channels)).into()
				},
			Padding::Valid => {
					iter::once(batch_size)
					.chain(input_spatial.zip(filter_spatial).map(|(dim, k_dim)| dim - k_dim + 1))
					.chain(iter::once(out_channels)).into()
				},
			Padding::Padded(ref size) => {
					iter::once(batch_size)
					.chain(input_spatial.map(|dim| dim +size))
					.chain(iter::once(out_channels)).into()
				},
			Padding::PaddedDiff(ref vec) => {
					iter::once(batch_size)
					.chain(input_spatial.zip(vec).map(|(dim, vec_dim)| dim + vec_dim))
					.chain(iter::once(out_channels)).into()
				},
		};


		shapes.merge_with(&self.output_id, &output_shape)?;

		// shapes[self.output_id.ind] = required_shape.merge(&shapes[self.output_id.ind])
		// 	.expect(&format!("Error: Operation '{}' error could not merge required output shape with existing shape for Node '{}'.", self.name, nodes[self.output_id.ind].name));
		// 	//.expect(&format!("Error: Operation '{}' error could not merge required output shape with existing shape for Node '{}'. old shape: {:?}, new shape: {:?}", self.name, nodes[self.output_id.ind].name, shapes[self.output_id.ind], required_shape));
		
		Ok(())
	}
}


#[derive(Debug, Clone)]
pub struct ConvForward {
	input_id: NodeID,
	output_id: NodeID,
	filter_id: NodeID,
	lowering_memory: usize,
}

impl ConvForward {
	pub fn new(input_id: NodeID, output_id: NodeID, filter_id: NodeID, lowering_memory: usize) -> Self{
		ConvForward {
			input_id,
			output_id,
			filter_id,
			lowering_memory,
		}
	}
}

impl Pass for ConvForward {
	fn type_name(&self) -> &'static str {"ConvForward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input_id.value_id(), self.filter_id.value_id()],
		vec![self.output_id.value_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>> {
		let input = data.get(&self.input_id.value_id())?;
		let filter = data.get(&self.filter_id.value_id())?;
		let mut output = data.get_mut(&self.output_id.value_id())?;

		let n = input.shape()[0]; //TODO use ensure to guard against zero length shapes
		let in_size: usize = input.shape()[1..].iter().product();
		let _out_size: usize = output.shape()[1..].iter().product();
		let patch_size = filter.shape()[1..].iter().product();

		let input_channels = input.shape()[input.shape().len()-1];
		let output_channels = output.shape()[output.shape().len()-1];

		let input_spatial = &input.shape()[1..input.shape().len()-1];
		let filter_spatial = &filter.shape()[1..filter.shape().len()-1];
		let output_spatial = output.shape()[1..output.shape().len()-1].to_vec();

		let _in_spaxels: usize = input_spatial.iter().product();
		let out_spaxels: usize = output_spatial.iter().product();

		// Checks
		ensure!(input.shape().len() == output.shape().len(), "Input ndims does not match output ndims");
		ensure!(input.shape().len() == filter.shape().len(), "Filter ndims does not match input ndims");

		ensure!(input.shape()[0] == output.shape()[0], "Batch size of input does not match batch size of output");
		ensure!(input_channels == filter.shape()[filter.shape().len()-1], "input channels dimension does not match final filter dimension");
		ensure!(output_channels == filter.shape()[0], "output channels dimension does not match first filter dimension");



		let input = input.as_slice().unwrap();
		let filter = filter.as_slice().unwrap();
		let output = output.as_slice_mut().unwrap();
		debug_assert!(!filter.iter().cloned().any(f32::is_nan), "{:?}", filter);


		let mut pool = THREAD_POOL.lock().expect("Could not lock conv threadpool");
		pool.scoped(|scope|{

			let max_spaxels = min(max(1, self.lowering_memory/(patch_size*size_of::<f32>())), out_spaxels*n); // number of spaxels to combine in one sgemm
			let n_batches = (out_spaxels*n + max_spaxels -1)/max_spaxels;


			let filter_strides = stride_vec2(input_channels, &filter_spatial);
			let input_strides = stride_vec2(input_channels, &input_spatial);
			let output_strides = stride_vec2(output_channels, &output_spatial);

			let (tx, rx) = sync_channel(1);
			let (tx2, rx2) = sync_channel(1);
			let mut spare_patches = Vec::with_capacity(patch_size * max_spaxels); 
			unsafe{spare_patches.set_len(patch_size * max_spaxels);}
			let mut spare_patches_opt = Some(spare_patches);
			scope.execute(move|| {
				let mut patches = Vec::with_capacity(patch_size * max_spaxels); 
				unsafe{patches.set_len(patch_size * max_spaxels);}
				let mut patches_opt = Some(patches);

				for batch in 0..n_batches {
					let spaxel_ind = batch*max_spaxels;
					
					let batch_spaxels = min(out_spaxels*n - spaxel_ind, max_spaxels);
					{
						let patches = &mut patches_opt.as_mut().expect("conv patches missing")[..batch_spaxels*patch_size];
						for (i, patch) in patches.chunks_mut(patch_size).enumerate() {
							debug_assert_eq!(patch_size, patch.len());
							let n_ind = (spaxel_ind+i)/out_spaxels;

							let in_n = &input[n_ind*in_size..][..in_size];	

							let output_ind = (spaxel_ind+i)%out_spaxels*output_channels;
							unsafe_pack_patch_outer(patch, in_n, input_channels, output_ind, &filter_spatial, &input_spatial, &output_spatial, &filter_strides, &input_strides, &output_strides);
							//pack_patch_recurse(patch, in_n, &kernel_shape, input_channels, &input.shape.spatial_dimensions, &output_shape.spatial_dimensions, kernel_shape.len()-1, output_ind, out_size);
						}
					}
					tx.send(Some((patches_opt.take().expect("conv patches missing"), spaxel_ind, batch_spaxels))).expect("conv patch send err");
					patches_opt = Some(rx2.recv().expect("conv patches missing"));
				}
				tx.send(None).expect("conv patch send err");
			});

			while let Some((patches, spaxel_ind, batch_spaxels)) = rx.recv().expect("Convolution channel receive error") {
				tx2.send(spare_patches_opt.take().expect("conv patches missing")).expect("conv patch send err");
				let out_batch = &mut output[spaxel_ind*output_channels..][..batch_spaxels*output_channels];

				let m = output_channels;
				let n = batch_spaxels;
				let k = patch_size;
				debug_assert_eq!(filter.len(), k*m);
				debug_assert!(patches.len() >= n*k);
				debug_assert_eq!(out_batch.len(), n*m);
				unsafe{
					matrixmultiply::sgemm(m, k, n,
						1.0,
						filter.as_ptr(), k as isize, 1, // A is params, row major
						patches.as_ptr(), 1, k as isize, // B, input patches column major
						1.0,
						out_batch.as_mut_ptr(), 1, m as isize); // C output values column major
				}
				spare_patches_opt = Some(patches);
			}
		});
		Ok(Box::new(()))
	}
}

#[derive(Debug, Clone)]
struct ConvBackward {
	input_id: NodeID,
	output_id: NodeID,
	filter_id: NodeID,
	lowering_memory: usize,
}

impl ConvBackward {
	pub fn new(input_id: NodeID, output_id: NodeID, filter_id: NodeID, lowering_memory: usize) -> Self {
		ConvBackward {
			input_id,
			output_id,
			filter_id,
			lowering_memory,
		}
	}
}

impl Pass for ConvBackward {
	fn type_name(&self) -> &'static str {"ConvBackward"}

	fn dependencies(&self) -> (Vec<DataID>, Vec<DataID>){
		(vec![self.input_id.value_id(), self.filter_id.value_id(), self.output_id.gradient_id()],
		vec![self.input_id.gradient_id(), self.filter_id.gradient_id()])
	}

	fn run (&self, data: &Storage) -> Result<Box<Any>> {

		let input = data.get(&self.input_id.value_id())?;
		let filter = data.get(&self.filter_id.value_id())?;
		let output_grad = data.get(&self.output_id.gradient_id())?;

		let n = input.shape()[0]; //TODO use ensure to guard against zero length shapes
		let _in_size: usize = input.shape()[1..].iter().product();
		let out_size: usize = output_grad.shape()[1..].iter().product();
		let patch_size = filter.shape()[..filter.ndim()-1].iter().product();

		let input_channels = input.shape()[input.shape().len()-1];
		let output_channels = output_grad.shape()[output_grad.shape().len()-1];

		let input_spatial = &input.shape()[1..input.shape().len()-1];
		let filter_spatial = &filter.shape()[1..filter.shape().len()-1];
		let output_spatial = output_grad.shape()[1..output_grad.shape().len()-1].to_vec();

		let in_spaxels: usize = input_spatial.iter().product();
		let _out_spaxels: usize = output_spatial.iter().product();

		// Checks
		ensure!(input.shape().len() == output_grad.shape().len(), "Input ndims does not match output ndims");
		ensure!(input.shape().len() == filter.shape().len(), "Filter ndims does not match input ndims");

		ensure!(input.shape()[0] == output_grad.shape()[0], "Batch size of input does not match batch size of output");
		ensure!(input_channels == filter.shape()[filter.shape().len()-1], "input channels dimension does not match final filter dimension");
		ensure!(output_channels == filter.shape()[0], "output channels dimension does not match first filter dimension");


		let input = input.as_slice().unwrap();
		let output_grad = output_grad.as_slice().unwrap();

		let mut input_grad = if data.is_required(&self.input_id.gradient_id()) {
			Some(data.get_mut(&self.input_id.gradient_id())?)
		} else {
			None
		};
		let mut filter_grad = if data.is_required(&self.filter_id.gradient_id()) {
			Some(data.get_mut(&self.filter_id.gradient_id())?)
		} else {
			None
		};


		let mut pool = THREAD_POOL.lock().expect("Could not lock conv threadpool");
		pool.scoped(|scope|{

			let max_spaxels = min(max(1, self.lowering_memory/(patch_size*4)), in_spaxels*n); // number of spaxels to combine in one sgemm
			let n_batches = (in_spaxels*n + max_spaxels -1)/max_spaxels;


			// Rot180, or filter inversion
			// Convert filter from [C_out, H, W, C_in] to [C_in, -H, -W, C_out]
			// where negative dimensions indicate the dimension has been inverted
			let mut inverted_filter_view = filter.view();
			inverted_filter_view.swap_axes(0, filter.ndim()-1);
			for axis in (1..filter.ndim()-1).map(Axis) {
				inverted_filter_view.invert_axis(axis);
			}
			
			let mut inverted_filter = unsafe{ArrayD::uninitialized(inverted_filter_view.shape())};
			inverted_filter.assign(&inverted_filter_view);

			debug_assert!(inverted_filter.is_standard_layout());
			let inverted_filter_slice = inverted_filter.as_slice().unwrap();


			let mut inverted_filter_grad: ArrayD<f32> = if filter_grad.is_some() {
				ArrayD::zeros(inverted_filter.shape())
			} else {
				ArrayD::default(IxDyn(&[]))
			};
			
			let filter_strides = stride_vec2(output_channels, &filter_spatial);
			let input_strides = stride_vec2(input_channels, &input_spatial);
			let output_strides = stride_vec2(output_channels, &output_spatial);
			
			let (tx, rx) = sync_channel(1);
			let (tx2, rx2) = sync_channel(1);
			let mut spare_patches = Vec::with_capacity(patch_size * max_spaxels); 
			unsafe{spare_patches.set_len(patch_size * max_spaxels);}
			let mut spare_patches_opt = Some(spare_patches);
			scope.execute(move|| {
				let mut patches = Vec::with_capacity(patch_size * max_spaxels); 
				unsafe{patches.set_len(patch_size * max_spaxels);}
				let mut patches_opt = Some(patches);
				for batch in 0..n_batches {
					let spaxel_ind = batch*max_spaxels;
					let batch_spaxels = min(in_spaxels*n - spaxel_ind, max_spaxels);

					{
						let patches = &mut patches_opt.as_mut().expect("conv patches missing")[..batch_spaxels*patch_size];
						for (i, patch) in patches.chunks_mut(patch_size).enumerate() {
							debug_assert_eq!(patch_size, patch.len());
							let n_ind = (spaxel_ind+i)/in_spaxels;

							let outg_n = &output_grad[n_ind*out_size..][..out_size];

							let input_ind = (spaxel_ind+i)%in_spaxels*input_channels;
							unsafe_pack_patch_outer(patch, outg_n, output_channels, input_ind, &filter_spatial, &output_spatial, &input_spatial, &filter_strides, &output_strides, &input_strides);
							//pack_patch_recurse(patch, outd_n, &kernel_shape, output_channels, &output.shape.spatial_dimensions, &input_shape.spatial_dimensions, kernel_shape.len()-1, input_ind, in_size);
						}
					}
					tx.send(Some((patches_opt.take().expect("conv patches missing"), spaxel_ind, batch_spaxels))).expect("conv patch send err");
					patches_opt = Some(rx2.recv().expect("conv patches missing"));
				}
				tx.send(None).expect("conv patch send err");
			});
			

			while let Some((patches, spaxel_ind, batch_spaxels)) = rx.recv().expect("Convolution channel receive error") {
				tx2.send(spare_patches_opt.take().expect("conv patches missing")).expect("conv patch send err");

				let in_b = &input[spaxel_ind*input_channels..][..batch_spaxels*input_channels];

				if let Some(ref mut input_grad) = input_grad {
					let mut input_grad_slice = input_grad.as_slice_mut().unwrap();
					let m1 = input_channels;
					let n1 = batch_spaxels;
					let k1 = patch_size;
					let ind_b = &mut input_grad_slice[spaxel_ind*input_channels..][..batch_spaxels*input_channels];
					debug_assert_eq!(inverted_filter_slice.len(), k1*m1);
					debug_assert!(patches.len() >= n1*k1);
					debug_assert_eq!(ind_b.len(), n1*m1);
					unsafe{
						// input derivatives
						matrixmultiply::sgemm(m1, k1, n1,
							1.0,
							inverted_filter_slice.as_ptr(), k1 as isize, 1, // A is params, row major
							patches.as_ptr(), 1, k1 as isize, // B, input values, column major
							1.0,
							ind_b.as_mut_ptr(), 1, m1 as isize // C output values, column major
						); 
					}
				}

				if let Some(ref mut _filter_grad) = filter_grad {
					let inverted_filter_grad_slice = inverted_filter_grad.as_slice_mut().unwrap();
					let m2 = input_channels;
					let n2 = patch_size;
					let k2 = batch_spaxels;
					debug_assert_eq!(in_b.len(), k2*m2);
					debug_assert!(patches.len() >= n2*k2);
					debug_assert_eq!(inverted_filter_grad_slice.len(), n2*m2);
					unsafe{
						matrixmultiply::sgemm(m2, k2, n2,
							1.0,
							in_b.as_ptr(), 1, m2 as isize, // A is input image, col major
							patches.as_ptr(), n2 as isize, 1, // B, derivative patches, row major
							1.0,
							inverted_filter_grad_slice.as_mut_ptr(), n2 as isize, 1 // C shuffled parameter derivatives, row major
						);
					}
				}

				spare_patches_opt = Some(patches);
			}

			// Write accumulated gradients back to the original (non-ROT180) format
			if let Some(ref mut filter_grad) = filter_grad {
				let mut inverted_filter_grad_actual = filter_grad.view_mut();
				inverted_filter_grad_actual.swap_axes(0, filter.ndim()-1);
				for axis in (1..filter.ndim()-1).map(Axis) {
					inverted_filter_grad_actual.invert_axis(axis);
				}
				inverted_filter_grad_actual += &inverted_filter_grad;
			}

		});

		Ok(Box::new(()))
	}		
}




/// A recursive N-dimensional im2col like function.
/// Packs data from a rectangular region of 'input' into 'patch'.
/// Inner recursions deal with lower dimensional slices of the 'input' and the 'patch' recursing until it is reduced to a 1D memcpy.
///
/// # Arguments
/// * `patch` - a rectangular region of input, of shape Cin.ks[0]...ks[axis]; Each recursion removes the outermost spatial dimension.
/// * `input` - input 'image' of shape Cin.W.H
/// * `patch_shape` - Spatial dimensions of the patch, in spaxels
/// * `n_channels` - The number of channels in the 'input' and 'patch' i.e. Cin
/// * `input_shape` - Spatial dimensions of the input, in spaxels
/// * `output_shape` - Spatial dimensions of the output, in spaxels
/// * `axis` - current axis being iterated over. This should be ks.len() - 1 for root call. Reduces by 1 each recursion.
/// * `output_ind` - Index of output spaxel on which the patch is centred. Note: index is the slice index not spaxel index (factor of Cin difference)
/// * `old_out_stride` - Slice stride of output for the layer bove the current iteration. used for interpreting `output_ind`. Root call should be output.len()
#[allow(unused)]
fn pack_patch_recurse(patch: &mut [f32], input: &[f32], patch_shape:&[usize], n_channels: usize,
	input_shape: &[usize], output_shape: &[usize], axis: usize, output_ind: usize, old_out_stride: usize){
	
	
	// stride in array index, not spaxel index
	let out_stride = old_out_stride/output_shape[axis];
	let in_stride = input.len()/input_shape[axis];
	let ks_stride = patch.len()/patch_shape[axis];
	
	// coordinates of the centre spaxel of the kernel, for the current axis, for both the output and the kernel itself
	let ox = (output_ind % old_out_stride)/out_stride;
	let ix = ox as isize + (input_shape[axis] as isize - output_shape[axis] as isize)/2;
	
	// valid range of the kernels in the current axis
	let (start, end) = kernel_range(ix, input_shape[axis], patch_shape[axis]);

	for i in 0..start*ks_stride{
		patch[i] = 0.0;// fill zeros
	}
				
	if axis > 0 {
		
		for i in start..end{
			let ix = (ix + i as isize - (patch_shape[axis]/2) as isize) as  usize; // shadow ix is the coordinate for the current iteration, rather than the centre of the kernel.

			let new_input = &input[in_stride*ix..in_stride*(ix+1)];
			let new_patch = &mut patch[i*ks_stride..(i+1)*ks_stride];
			let new_axis = axis-1;

			pack_patch_recurse(new_patch, new_input, patch_shape, n_channels, input_shape, output_shape, new_axis, output_ind, out_stride);
		}

	} else {	

		let offset = ((ix-patch_shape[axis] as isize/2)*n_channels as isize + (start*n_channels) as isize) as usize;
		let len = (end - start)*n_channels;
		let input_crop = &input[offset..][..len];
		let mut patch_crop = &mut patch[(start*n_channels)..][..len];
		
		patch_crop.copy_from_slice(input_crop);		
	}

	for i in (end*ks_stride)..(patch_shape[axis]*ks_stride){
		patch[i] = 0.0;// fill zero
	}		
}


// /// returns a vector with the array stride of each dimension. output[0] == channel.
// fn stride_vec(channels: usize, shape: &[usize]) -> Vec<usize>{
// 	iter::once(&channels).chain(shape.iter()).scan(1, |state, &i| {*state *= i; Some(*state)}).collect::<Vec<usize>>()
// }

/// returns a vector with the array stride of each dimension. output[n] == channel.
fn stride_vec2(channels: usize, shape: &[usize]) -> SmallVec<[usize;6]>{
	let mut strides = iter::once(&channels).chain(shape.iter().rev()).scan(1, |state, &i| {
		let res = Some(*state);
		*state *= i;
		res
	}).collect::<SmallVec<[usize;6]>>();
	strides.reverse();
	strides
}

//#[inline(never)]
fn unsafe_pack_patch_outer(patch: &mut [f32], input: &[f32], channels: usize, output_ind: usize,
	kernel_shape: &[usize], input_shape: &[usize], output_shape: &[usize],
	kernel_strides: &[usize], input_strides: &[usize], output_strides: &[usize]){
	let axis = 0;

	let ox = output_ind/output_strides[axis];
	let ix = ox as isize + (input_shape[axis] as isize - output_shape[axis] as isize)/2;
	let (start, end) = kernel_range(ix, input_shape[axis], kernel_shape[axis]);

	unsafe {unsafe_pack_patch_recurse(patch, input, channels, axis, output_ind, ox, ix, start, end,
	kernel_shape, input_shape, output_shape, kernel_strides, input_strides, output_strides)};
}

//#[inline(always)]
unsafe fn unsafe_pack_patch_recurse(patch: &mut [f32], input: &[f32], channels: usize, axis: usize, output_ind: usize,
	ox: usize, ix: isize,
	start: usize, end: usize, // valid range of the kernels in the current axis
	kernel_shape: &[usize], input_shape: &[usize], output_shape: &[usize],
	kernel_strides: &[usize], input_strides: &[usize], output_strides: &[usize]){
	
	//println!("a:{} s:{} e:{} ind: {} ox:{}, ix:{}", axis, start, end, output_ind, ox, ix);

	let i_stride = *odds::get_unchecked(input_strides, axis);
	let o_stride = *odds::get_unchecked(output_strides, axis);
	let k_stride = *odds::get_unchecked(kernel_strides, axis);
	// coordinates of the centre spaxel of the kernel, for the current axis, for both the output and the kernel itself
	
	for i in 0..start*k_stride {
		*odds::get_unchecked_mut(patch, i) = 0.0; // fill zero
	}
				
	if axis < kernel_shape.len() - 1 {
		for i in start..end{
			let temp_ix = (ix + i as isize - (*odds::get_unchecked(kernel_shape, axis)/2) as isize) as usize; // temp_ix is the coordinate for the current iteration, rather than the centre of the kernel.
			debug_assert!(ix + i as isize - (kernel_shape[axis]/2) as isize >= 0);
			
			let new_input = odds::slice_unchecked(input, i_stride*temp_ix, i_stride*(temp_ix+1));

			let new_patch = odds::slice_unchecked_mut(patch, i*k_stride, (i+1)*k_stride);

			let new_axis = axis+1;
			let new_output_ind = output_ind - ox*o_stride;
			let new_ox = new_output_ind/ *odds::get_unchecked(output_strides, new_axis);
			let new_ix = new_ox as isize + (*odds::get_unchecked(input_shape, new_axis) as isize - *odds::get_unchecked(output_shape, new_axis) as isize)/2;
			let (new_start, new_end) = kernel_range(new_ix, *odds::get_unchecked(input_shape, new_axis), *odds::get_unchecked(kernel_shape, new_axis));// input_shape[new_axis]    kernel_shape[new_axis]);

			unsafe_pack_patch_recurse(new_patch, new_input, channels, new_axis, new_output_ind, new_ox, new_ix,
			new_start, new_end, kernel_shape, input_shape, output_shape, kernel_strides, input_strides, output_strides)
		}

	} else if end > start {
		let offset = ((ix-*odds::get_unchecked(kernel_shape, axis) as isize/2)*channels as isize + (start*channels) as isize) as usize;
		let len = (end - start)*channels;

		//let input_crop = &input[offset..][..len];
		let input_crop = odds::slice_unchecked(input, offset, offset + len);
		
		//let mut patch_crop = &mut patch[(start*channels)..][..len];
		let patch_crop = odds::slice_unchecked_mut(patch, start*channels, start*channels + len);
		
		//patch_crop.copy_from_slice(input_crop);
		for i in 0..len{
			*odds::get_unchecked_mut(patch_crop, i) = *odds::get_unchecked(input_crop, i);
		}
	}

	for i in (end*k_stride)..(*odds::get_unchecked(kernel_shape, axis)*k_stride){
		*odds::get_unchecked_mut(patch, i) = 0.0; // fill zero
	}
}

/// returns the [start, end) range indicies of the kernel which overlap with the 'image'
/// returns [start, end) of the kernel range which overlaps with the image, given the width of the image and the position of the center of the kernel
/// only odd kernels are valid.
fn kernel_range(center: isize, width: usize, kernel_width: usize) -> (usize, usize){
 	debug_assert!(kernel_width % 2 == 1);
	(
		min(kernel_width as isize, max(0, kernel_width as isize/2 - center)) as usize,
		max(0, kernel_width as isize - max(0, center + kernel_width as isize/2 + 1- width as isize)) as usize
	)
}


#[cfg(test)]
/// take a kernel ordered as Cinner.W.H.Couter and convert to a same shaped kernel which has spatial dimension (W.H) orders reversed.
/// Also known as ROT180() in some libraries
fn flip_kernel_overwrite(kernel: &[f32], in_channels: usize, out_channels: usize, out: &mut [f32]){
	debug_assert_eq!(kernel.len(), out.len());
	debug_assert_eq!(kernel.len()%(in_channels * out_channels), 0);
	let num_blocks = kernel.len()/(in_channels * out_channels);
	let in_stride = kernel.len()/out_channels;
	
	for b in 0..num_blocks{
		let x_in = b * in_channels;
		let x_out = (num_blocks - b - 1) * in_channels;
		
		for out_ind in 0.. out_channels{
			let x_in = x_in + out_ind*in_stride;
			let x_out = x_out + out_ind*in_stride;		
			for in_ind in 0..in_channels{
				let x_in = x_in + in_ind;
				let x_out = x_out + in_ind;

				out[x_out] = kernel[x_in];
			}
		}
		
	}	
}

#[cfg(test)]
/// take a kernel ordered as Cinner.W.H.Couter and shuffle data to Couter.W.H.Cinner, transposing the channels of each spaxel.
fn transpose_kernel_overwrite(kernel: &[f32], in_channels: usize, out_channels: usize, out: &mut [f32]){
	debug_assert_eq!(kernel.len(), out.len());
	debug_assert_eq!(kernel.len()%(in_channels * out_channels), 0);
	let num_blocks = kernel.len()/(in_channels * out_channels);
	let in_stride = kernel.len()/out_channels;
	let out_stride = kernel.len()/in_channels;
	
	for b in 0..num_blocks{
		let x_in = b * in_channels;
		let x_out = b * out_channels;
		
		for in_ind in 0..in_channels{
			let x_in = x_in + in_ind;
			let x_out = x_out + in_ind*out_stride;
			
			for out_ind in 0.. out_channels{
				let x_in = x_in + out_ind*in_stride;
				let x_out = x_out + out_ind;

				out[x_out] = kernel[x_in];
			}
		}
		
	}
}

/// Combine kernel Flip and Transpose functions
///  takes row major, outputs row major
#[cfg(test)]
fn flip_transpose_kernel_overwrite(kernel: &[f32], in_channels: usize, out_channels: usize, out: &mut [f32]){
	debug_assert_eq!(kernel.len(), out.len());
	debug_assert_eq!(kernel.len()%(in_channels * out_channels), 0);
	let num_blocks = kernel.len()/(in_channels * out_channels);
	let in_stride = kernel.len()/out_channels;
	let out_stride = kernel.len()/in_channels;
	
	for b in 0..num_blocks{
		let x_in = b * in_channels;
		let x_out = (num_blocks - b - 1) * out_channels;
		
		for in_ind in 0..in_channels{
			let x_in = x_in + in_ind;
			let x_out = x_out + in_ind*out_stride;
			for out_ind in 0.. out_channels{
				let x_in = x_in + out_ind*in_stride;
				let x_out = x_out + out_ind;
				//out[x_out] = kernel[x_in];
				unsafe{
					*out.get_unchecked_mut(x_out) = *kernel.get_unchecked(x_in);
				}
			}
		}
		
	}
}

/// Combine kernel Flip and Transpose functions

/// takes row major, outputs col major
#[cfg(test)]
#[allow(unused)]
fn col_flip_transpose_kernel_overwrite(kernel: &[f32], in_channels: usize, out_channels: usize, out: &mut [f32]){
	debug_assert_eq!(kernel.len(), out.len());
	debug_assert_eq!(kernel.len()%(in_channels * out_channels), 0);
	let num_blocks = kernel.len()/(in_channels * out_channels);
	let in_stride = kernel.len()/out_channels;
	//let out_stride = kernel.len()/in_channels;

	for b in 0..num_blocks{
		let kernel = &kernel[b * in_channels..];
		let out = &mut out[(num_blocks - b - 1) * (in_channels * out_channels)..];
		
		for out_ind in 0.. out_channels{
			let kernel = &kernel[out_ind*in_stride..][..in_channels];
			let out = &mut out[out_ind*in_channels..][..in_channels];

			for i in 0..in_channels{
				out[i] = kernel[i];
			}
		}
	}
}

#[cfg(test)]
#[allow(unused)]
fn rev_col_flip_transpose_kernel_overwrite(kernel: &[f32], in_channels: usize, out_channels: usize, out: &mut [f32]){
	debug_assert_eq!(kernel.len(), out.len());
	debug_assert_eq!(kernel.len()%(in_channels * out_channels), 0);
	let num_blocks = kernel.len()/(in_channels * out_channels);
	let in_stride = kernel.len()/out_channels;
	//let out_stride = kernel.len()/in_channels;
	
	for b in 0..num_blocks{
		let out = &mut out[b * in_channels..];
		let kernel = &kernel[(num_blocks - b - 1) * (in_channels * out_channels)..];
		
		for out_ind in 0.. out_channels{
			let out = &mut out[out_ind*in_stride..][..in_channels];
			let kernel = &kernel[out_ind*in_channels..][..in_channels];

			for i in 0..in_channels{
				out[i] = kernel[i];
			}
		}
	}
}



#[test]
fn conv_backprop(){
	_conv_backprop().unwrap();
}

fn _conv_backprop() -> Result<()>{
	use graph::GraphDef;
	use ops::numeric_check::numeric_test;
	use ops::loss::mse::Mse;
	use ordermap::OrderMap;

	let mut g = GraphDef::new();
	
	let node1 = g.new_node(shape![3, 5, 7, 13], "input", tag![])?;
	let node2 = g.new_node(shape![Unknown, Unknown, Unknown, 11], "conv", tag![])?;
	let node3 = g.new_node(shape![3, 5, 7, 11], "target", tag![])?;
		
	let _o1 = g.new_op(Conv::new(&node1, &node2, &[3, 5]), tag![])?;
	let _o2 = g.new_op(Mse::new(&node2, &node3), tag![])?;

	let iters = 100;
	let failures = 1;
	let tolerance = 0.01;
	let step_size = 1E-2;
	let default_variance = 1.0;
	numeric_test(iters, failures, tolerance, &g, step_size, default_variance, &mut OrderMap::new())?;
	
	Ok(())
}


#[test]
fn test_kernel_shuffles(){
	
	let num_blocks = 3;
	let in_channels = 2;
	let out_channels = 5;

	let kernel = (0..num_blocks*in_channels*out_channels).map(|x| x as f32).collect::<Vec<_>>();

	let mut flip = vec![0.0; kernel.len()];
	flip_kernel_overwrite(&kernel, in_channels, out_channels, &mut flip);	
	assert_eq!(flip, vec![
			4.,  5.,  2.,  3.,  0.,  1.,
		10., 11.,  8.,  9.,  6.,  7.,
		16., 17., 14., 15., 12., 13.,
		22., 23., 20., 21., 18., 19.,
		28., 29., 26., 27., 24., 25.,
		]);
	
	let mut trans = vec![0.0; kernel.len()];
	transpose_kernel_overwrite(&kernel, in_channels, out_channels, &mut trans);
	assert_eq!(trans, vec![
		0.,  6., 12., 18., 24.,
		2.,  8., 14., 20., 26.,	
		4., 10., 16., 22., 28.,
		1.,  7., 13., 19., 25., 
		3.,  9., 15., 21., 27.,
		5., 11., 17., 23., 29.,
		]);

	let mut flip_trans = vec![0.0; kernel.len()];
	flip_transpose_kernel_overwrite(&kernel, in_channels, out_channels, &mut flip_trans);
	
	let mut flip_trans2 = vec![0.0; kernel.len()];
	flip_kernel_overwrite(&trans, out_channels, in_channels, &mut flip_trans2);
	
	let mut flip_trans3 = vec![0.0; kernel.len()];
	transpose_kernel_overwrite(&flip, in_channels, out_channels, &mut flip_trans3);		
	
	assert_eq!(flip_trans, flip_trans2);
	assert_eq!(flip_trans, flip_trans3);
}	


#[test]
fn test_kernel_range(){
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
fn test_pack(){
	
	let filter_spatial = vec![3, 5];
	let input_spatial = vec![5, 7];
	let output_spatial = vec![7, 9];
	let input_channels = 3;
	let output_channels = 5;
	
	let input_spaxel_count = input_spatial.iter().fold(1, |p, v| p*v);
	let output_spaxel_count = output_spatial.iter().fold(1, |p, v| p*v);
	
	let input_size = input_spaxel_count*input_channels;
	let _output_size = output_spaxel_count*output_channels;
	let patch_size = filter_spatial.iter().fold(input_channels, |p, v| p*v);


	let input: Vec<f32> = (0..input_size).map(|x| x as f32 + 1.0).collect();		
			
	let filter_strides = stride_vec2(input_channels, &filter_spatial);
	let input_strides = stride_vec2(input_channels, &input_spatial);
	let output_strides = stride_vec2(output_channels, &output_spatial);

	let mut patches = vec![-0.5; patch_size*output_spaxel_count];
	
	for (i, patch) in patches.chunks_mut(patch_size).enumerate(){
		let output_ind = i*output_channels;
		unsafe_pack_patch_outer(patch, &input, input_channels, output_ind, &filter_spatial, &input_spatial, &output_spatial, &filter_strides, &input_strides, &output_strides);
		//pack_patch_recurse(patch, &input, &ks, in_channels, &input_shape, &output_shape, axis, output_ind, output_size);
	}
	
	debug_assert!(!patches.iter().cloned().any(|x| x.is_nan() || x == -0.5), "{:?}", patches);



	// for (i, patch) in patches.chunks(patch_size).enumerate(){
	// 	let (y, x) = (i/output_spatial[1], i%output_spatial[1]);
	// 	println!("y:{} x:{} patch:{:?}", y, x, patch);
	// }
	// panic!();
}