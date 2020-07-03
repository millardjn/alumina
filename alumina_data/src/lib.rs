pub mod cifar;
pub mod crop;
pub mod mnist;

pub use crate::crop::{Crop, Cropping};
use alumina_core::graph::Node;
use indexmap::IndexMap;
use ndarray::{Axis, IxDyn, ArcArray};
use rand::{seq::SliceRandom, thread_rng, Rng, RngCore, SeedableRng};
use rand_pcg::Pcg64Mcg;
use smallvec::SmallVec;
use std::{
	iter, mem,
	sync::{
		mpsc::{sync_channel, Receiver, TrySendError},
		Arc, Mutex, MutexGuard,
	},
	thread,
	time::Duration,
};

/// An indexable data set.
/// To use tensorflow terminology a dataset is made up of elements(`Vec<ArrayD>>`s), each of which can contain multiple
/// components (`ArrayD`s)
pub trait DataSet {
	/// Returns the `i`th element of the `Dataset`
	fn get(&mut self, i: usize) -> Vec<ArcArray<f32, IxDyn>>;

	/// Returns the number of elements in the `Dataset`
	fn length(&self) -> usize;

	/// Returns the number of components
	fn width(&self) -> usize;

	/// Returns the names of components
	fn components(&self) -> Vec<String>;

	fn iter<'a>(&'a mut self) -> Box<dyn Iterator<Item = Vec<ArcArray<f32, IxDyn>>> + 'a> {
		let iter = (0..self.length()).map(move |i| self.get(i));
		Box::new(iter)
	}

	fn boxed(self) -> BoxedSet
	where
		Self: 'static + Sized + Send,
	{
		// Box::new(self)
		BoxedSet { inner: Box::new(self) }
	}

	fn reorder_components(self, order: &[usize]) -> ReorderComponents<Self>
	where
		Self: Sized,
	{
		ReorderComponents::new(self, order)
	}

	fn reorder_elements(self, order: &[usize]) -> ReorderElements<Self>
	where
		Self: Sized,
	{
		ReorderElements::new(self, order)
	}

	fn concat_components<S: DataSet>(self, set: S) -> ConcatComponents<Self, S>
	where
		Self: Sized,
	{
		ConcatComponents::new(self, set)
	}

	fn concat_elements<S: DataSet>(self, set: S) -> ConcatElements<Self, S>
	where
		Self: Sized,
	{
		ConcatElements::new(self, set)
	}

	fn map_all<F: FnMut(usize, Vec<ArcArray<f32, IxDyn>>) -> Vec<ArcArray<f32, IxDyn>>>(
		self,
		func: F,
		names: Option<Vec<String>>,
	) -> MapAll<Self, F>
	where
		Self: Sized,
	{
		MapAll::new(self, func, names)
	}

	fn map_one<F: FnMut(usize, ArcArray<f32, IxDyn>) -> ArcArray<f32, IxDyn>>(self, func: F, component: usize) -> MapOne<Self, F>
	where
		Self: Sized,
	{
		MapOne::new(self, func, component)
	}

	fn crop(self, component: usize, shape: &[usize], cropping: Cropping) -> Crop<Self>
	where
		Self: Sized,
	{
		Crop::new(self, component, shape, cropping)
	}

	fn sequential(self) -> Sequential<Self>
	where
		Self: Sized,
	{
		Sequential::new(self)
	}

	fn random(self) -> Random<Self>
	where
		Self: Sized,
	{
		Random::new(self)
	}

	fn shuffle_random(self) -> ShuffleRandom<Self>
	where
		Self: Sized,
	{
		ShuffleRandom::new(self)
	}
}

/// Its just a box, but the newtype avoids deref not Sized nonsense
pub struct BoxedSet {
	inner: Box<dyn DataSet + Send>,
}

impl DataSet for BoxedSet {
	fn get(&mut self, i: usize) -> Vec<ArcArray<f32, IxDyn>> {
		self.inner.get(i)
	}

	fn length(&self) -> usize {
		self.inner.length()
	}

	fn width(&self) -> usize {
		self.inner.width()
	}

	fn components(&self) -> Vec<String> {
		self.inner.components()
	}
}

/// Reorder the components within each element of the set
///
/// The new order of components will be defined by a list of indices, the length of the list will define the new number
/// of components (dataset.width()), and the value of each index will be used to sample the old component order.
///
/// E.g. a set with component order [A, B, C] when reordered with the indices [2, 1, 3, 1] will produce an a set with
/// component order [B, A, C, A]
pub struct ReorderComponents<S: DataSet> {
	set: S,
	order: Vec<usize>,
}

impl<S: DataSet> ReorderComponents<S> {
	pub fn new(s: S, order: &[usize]) -> Self {
		assert!(
			order.iter().all(|&i| i < s.width()),
			"An element of order ({:?}) exceeded the width of the dataset ({})",
			order,
			s.width()
		);
		ReorderComponents {
			set: s,
			order: order.to_vec(),
		}
	}

	/// Borrows the wrapped dataset.
	pub fn inner(&self) -> &S {
		&self.set
	}

	/// Returns the wrapped dataset.
	pub fn into_inner(self) -> S {
		let Self { set, .. } = self;
		set
	}
}

impl<S: DataSet> DataSet for ReorderComponents<S> {
	fn get(&mut self, i: usize) -> Vec<ArcArray<f32, IxDyn>> {
		enum Comp {
			Present(ArcArray<f32, IxDyn>),
			Moved(usize),
		}
		let mut components: Vec<_> = self.set.get(i).into_iter().map(Comp::Present).collect();

		let mut out: Vec<ArcArray<f32, IxDyn>> = Vec::with_capacity(self.order.len());
		for (new, &old) in self.order.iter().enumerate() {
			match &mut components[old] {
				x @ &mut Comp::Present(_) => {
					let x = mem::replace(x, Comp::Moved(new));
					if let Comp::Present(data) = x {
						out.push(data);
					}
				}
				&mut Comp::Moved(ind) => {
					let data = out[ind].clone();
					out.push(data);
				}
			}
		}

		out
	}

	fn length(&self) -> usize {
		self.set.length()
	}

	fn width(&self) -> usize {
		self.order.len()
	}

	fn components(&self) -> Vec<String> {
		let names = self.set.components();
		self.order.iter().map(|&i| names[i].clone()).collect()
	}
}

/// Reorder the element of the set
///
/// The new order of element will be defined by a list of indices, the length of the list will define the new number of
/// elements (dataset.size()), and the value of each index will be used to sample the old element order.
///
/// E.g. a set with element order [A, B, C] when reordered with the indices [2, 1, 3, 1] will produce an a set with
/// element order [B, A, C, A]
pub struct ReorderElements<S: DataSet> {
	set: S,
	order: Vec<usize>,
}

impl<S: DataSet> ReorderElements<S> {
	pub fn new(set: S, order: &[usize]) -> Self {
		ReorderElements {
			set,
			order: order.to_vec(),
		}
	}

	/// Borrows the wrapped dataset.
	pub fn inner(&self) -> &S {
		&self.set
	}

	/// Returns the wrapped dataset.
	pub fn into_inner(self) -> S {
		let Self { set, .. } = self;
		set
	}
}

impl<S: DataSet> DataSet for ReorderElements<S> {
	fn get(&mut self, i: usize) -> Vec<ArcArray<f32, IxDyn>> {
		let j = self.order[i];
		self.set.get(j)
	}

	fn length(&self) -> usize {
		self.order.len()
	}

	fn width(&self) -> usize {
		self.set.width()
	}

	fn components(&self) -> Vec<String> {
		self.set.components()
	}
}

/// Concatenate two `DataSet`s in the width direction
///
/// The new dataset has the same number of elements (length) as the two inputs,
/// but the number of components in each element increases.
pub struct ConcatComponents<S1: DataSet, S2: DataSet> {
	set1: S1,
	set2: S2,
}

impl<S1: DataSet, S2: DataSet> ConcatComponents<S1, S2> {
	pub fn new(s1: S1, s2: S2) -> Self {
		assert_eq!(s1.length(), s2.length());
		ConcatComponents { set1: s1, set2: s2 }
	}

	/// Borrows the first wrapped dataset
	pub fn inner1(&self) -> &S1 {
		&self.set1
	}

	/// Borrows the second wrapped dataset
	pub fn inner2(&self) -> &S2 {
		&self.set2
	}

	/// Returns the wrapped datasets.
	pub fn into_inner(self) -> (S1, S2) {
		let Self { set1, set2 } = self;
		(set1, set2)
	}
}

impl<S1: DataSet, S2: DataSet> DataSet for ConcatComponents<S1, S2> {
	fn get(&mut self, i: usize) -> Vec<ArcArray<f32, IxDyn>> {
		let mut data1 = self.set1.get(i);
		let mut data2 = self.set2.get(i);
		data1.append(&mut data2);
		data1
	}

	fn length(&self) -> usize {
		self.set1.length()
	}

	fn width(&self) -> usize {
		self.set1.width() + self.set2.width()
	}

	fn components(&self) -> Vec<String> {
		let mut names1 = self.set1.components();
		let mut names2 = self.set2.components();
		names1.append(&mut names2);
		names1
	}
}

/// Concatenate two `DataSet`s in the length direction
///
/// The new dataset has the same number of components(width) in each element as the two inputs,
/// but the number of elements increases.
pub struct ConcatElements<S1: DataSet, S2: DataSet> {
	set1: S1,
	set2: S2,
}

impl<S1: DataSet, S2: DataSet> ConcatElements<S1, S2> {
	pub fn new(s1: S1, s2: S2) -> Self {
		assert_eq!(s1.width(), s2.width());
		ConcatElements { set1: s1, set2: s2 }
	}

	/// Borrows the first wrapped dataset
	pub fn inner1(&self) -> &S1 {
		&self.set1
	}

	/// Borrows the second wrapped dataset
	pub fn inner2(&self) -> &S2 {
		&self.set2
	}

	/// Returns the wrapped datasets.
	pub fn into_inner(self) -> (S1, S2) {
		let Self { set1, set2 } = self;
		(set1, set2)
	}
}

impl<S1: DataSet, S2: DataSet> DataSet for ConcatElements<S1, S2> {
	fn get(&mut self, i: usize) -> Vec<ArcArray<f32, IxDyn>> {
		if i < self.set1.length() {
			self.set1.get(i)
		} else {
			self.set2.get(i - self.set1.length())
		}
	}

	fn length(&self) -> usize {
		self.set1.length() + self.set2.length()
	}

	fn width(&self) -> usize {
		self.set1.width()
	}

	fn components(&self) -> Vec<String> {
		self.set1.components()
	}
}

/// For each element, map the components to a completely new vec of components.
///
/// If names for the new components arent provided the old names will be used.
/// If the map changes the number of components, new names must be provided.
pub struct MapAll<S: DataSet, F: FnMut(usize, Vec<ArcArray<f32, IxDyn>>) -> Vec<ArcArray<f32, IxDyn>>> {
	func: F,
	set: S,
	names: Option<Vec<String>>,
}

impl<S: DataSet, F: FnMut(usize, Vec<ArcArray<f32, IxDyn>>) -> Vec<ArcArray<f32, IxDyn>>> MapAll<S, F> {
	pub fn new(set: S, func: F, names: Option<Vec<String>>) -> Self {
		MapAll { func, set, names }
	}

	/// Borrows the wrapped dataset.
	pub fn inner(&self) -> &S {
		&self.set
	}

	/// Returns the wrapped dataset.
	pub fn into_inner(self) -> S {
		let Self { set, .. } = self;
		set
	}
}

impl<S: DataSet, F: FnMut(usize, Vec<ArcArray<f32, IxDyn>>) -> Vec<ArcArray<f32, IxDyn>>> DataSet for MapAll<S, F> {
	fn get(&mut self, i: usize) -> Vec<ArcArray<f32, IxDyn>> {
		let data = self.set.get(i);
		let new_data = (self.func)(i, data); // that is some weird syntax

		assert_eq!(new_data.len(), self.width());
		new_data
	}

	fn length(&self) -> usize {
		self.set.length()
	}

	fn width(&self) -> usize {
		if let Some(ref names) = self.names {
			names.len()
		} else {
			self.set.width()
		}
	}

	fn components(&self) -> Vec<String> {
		if let Some(ref names) = self.names {
			names.clone()
		} else {
			self.set.components()
		}
	}
}

/// For one component in each element of the dataset: apply a function.
///
/// Renaming the component is optional.
pub struct MapOne<S: DataSet, F: FnMut(usize, ArcArray<f32, IxDyn>) -> ArcArray<f32, IxDyn>> {
	func: F,
	set: S,
	component: usize,
	component_name: Option<String>,
}

impl<S: DataSet, F: FnMut(usize, ArcArray<f32, IxDyn>) -> ArcArray<f32, IxDyn>> MapOne<S, F> {
	pub fn new(set: S, func: F, component: usize) -> Self {
		MapOne {
			func,
			set,
			component,
			component_name: None,
		}
	}

	pub fn component_name(mut self, name: String) -> Self {
		self.component_name = Some(name);
		self
	}

	/// Borrows the wrapped dataset.
	pub fn inner(&self) -> &S {
		&self.set
	}

	/// Returns the wrapped dataset.
	pub fn into_inner(self) -> S {
		let Self { set, .. } = self;
		set
	}
}

impl<S: DataSet, F: FnMut(usize, ArcArray<f32, IxDyn>) -> ArcArray<f32, IxDyn>> DataSet for MapOne<S, F> {
	fn get(&mut self, i: usize) -> Vec<ArcArray<f32, IxDyn>> {
		let mut data = self.set.get(i);
		let arr = mem::replace(&mut data[self.component], ArcArray::zeros(IxDyn(&[])));
		mem::replace(&mut data[self.component], (self.func)(i, arr));
		data
	}

	fn length(&self) -> usize {
		self.set.length()
	}

	fn width(&self) -> usize {
		self.set.width()
	}

	fn components(&self) -> Vec<String> {
		let mut names = self.set.components();
		if let Some(ref name) = self.component_name {
			mem::replace(&mut names[self.component], name.clone());
		}
		names
	}
}

pub struct Sequential<S: DataSet> {
	set: S,
	next_i: usize,
}

impl<S: DataSet> Sequential<S> {
	pub fn new(set: S) -> Self {
		Sequential { set, next_i: 0 }
	}

	/// Borrows the wrapped dataset.
	pub fn inner(&self) -> &S {
		&self.set
	}

	/// Returns the wrapped dataset.
	pub fn into_inner(self) -> S {
		let Self { set, .. } = self;
		set
	}
}

impl<S: DataSet> DataStream for Sequential<S> {
	fn next(&mut self) -> Vec<ArcArray<f32, IxDyn>> {
		let out = self.set.get(self.next_i);
		self.next_i = (self.next_i + 1) % self.set.length();
		out
	}
}

pub struct Random<S: DataSet> {
	set: S,
	rng: Box<dyn RngCore + Send>,
}

impl<S: DataSet> Random<S> {
	pub fn new(set: S) -> Self {
		Random {
			set,
			rng: Box::new(Pcg64Mcg::from_rng(thread_rng()).unwrap()),
		}
	}

	pub fn rng<R: RngCore + 'static + Send>(mut self, rng: R) -> Self {
		self.rng = Box::new(rng);
		self
	}

	/// Borrows the wrapped dataset.
	pub fn inner(&self) -> &S {
		&self.set
	}

	/// Returns the wrapped dataset.
	pub fn into_inner(self) -> S {
		let Self { set, .. } = self;
		set
	}
}

impl<S: DataSet> DataStream for Random<S> {
	fn next(&mut self) -> Vec<ArcArray<f32, IxDyn>> {
		let set_len = self.set.length();
		self.set.get(self.rng.gen_range(0, set_len))
	}
}

pub struct ShuffleRandom<S: DataSet> {
	set: S,
	rng: Box<dyn RngCore + Send>,
	order: Vec<usize>,
	next_i: usize,
}

impl<S: DataSet> ShuffleRandom<S> {
	pub fn new(set: S) -> Self {
		let set_len = set.length();
		ShuffleRandom {
			set,
			rng: Box::new(Pcg64Mcg::from_rng(thread_rng()).unwrap()),
			order: (0..set_len).collect(),
			next_i: set_len,
		}
	}

	pub fn rng<R: RngCore + 'static + Send>(mut self, rng: R) -> Self {
		self.rng = Box::new(rng);
		self
	}

	/// Borrows the wrapped dataset.
	pub fn inner(&self) -> &S {
		&self.set
	}

	/// Returns the wrapped dataset.
	pub fn into_inner(self) -> S {
		let Self { set, .. } = self;
		set
	}
}

impl<S: DataSet> DataStream for ShuffleRandom<S> {
	fn next(&mut self) -> Vec<ArcArray<f32, IxDyn>> {
		if self.next_i >= self.order.len() {
			self.order.shuffle(&mut self.rng);
			self.next_i = 0;
		}

		let val = self.set.get(self.order[self.next_i]);
		self.next_i += 1;
		val
	}
}

pub trait DataStream {
	fn next(&mut self) -> Vec<ArcArray<f32, IxDyn>>;

	fn boxed(self) -> Box<Self>
	where
		Self: Sized,
	{
		Box::new(self)
	}

	fn batch(self, batch_size: usize) -> Batch<Self>
	where
		Self: Sized,
	{
		Batch::new(self, batch_size)
	}

	fn buffered(self, buffer_size: usize) -> Buffered<Self>
	where
		Self: Sized + Send,
	{
		Buffered::new(self, buffer_size)
	}

	fn zip<S: DataStream>(self, stream: S) -> Zip<Self, S>
	where
		Self: Sized,
	{
		Zip::new(self, stream)
	}

	fn interleave<S: DataStream + 'static>(self, stream: S) -> Interleave
	where
		Self: 'static + Sized,
	{
		Interleave::new(self, stream)
	}

	fn count<S: DataStream>(self) -> Count<Self>
	where
		Self: Sized,
	{
		Count::new(self)
	}

	fn map_all<F: FnMut(Vec<ArcArray<f32, IxDyn>>) -> Vec<ArcArray<f32, IxDyn>>>(self, func: F) -> StreamMapAll<Self, F>
	where
		Self: Sized,
	{
		StreamMapAll::new(self, func)
	}

	fn map_one<F: FnMut(ArcArray<f32, IxDyn>) -> ArcArray<f32, IxDyn>>(self, func: F, component: usize) -> StreamMapOne<Self, F>
	where
		Self: Sized,
	{
		StreamMapOne::new(self, func, component)
	}
}

impl dyn DataStream {
	/// Conviniece method for associating the result of self.next() with given nodes.
	///
	/// # Panics
	/// Panics if the length of `nodes` does not match the number of components in the DataStream (`self.next().len()`).
	pub fn next_with<I: Into<Node>, T: IntoIterator<Item = I>>(&mut self, nodes: T) -> IndexMap<Node, ArcArray<f32, IxDyn>> {
		let next = self.next();
		let len = next.len();

		let mut node_iter = nodes.into_iter().map(Into::into);
		let map: IndexMap<Node, _> = (&mut node_iter).zip(next).collect();
		let extra = node_iter.count();

		assert_eq!(
			map.len(),
			len,
			"The length of the nodes argument ({}) was less than the length of DataStream::next(..) ({})",
			map.len(),
			len
		);

		assert!(
			extra == 0,
			"The nodes argument contained {} more items than DataStream::next(..) ",
			extra
		);
		map
	}
}

/// Augment a stream with a fixed sized buffer fed by a new thread.
///
/// Up to buffer_size + 1 results may be in memory at one
pub struct Buffered<S: DataStream + Send + 'static> {
	stream: Arc<Mutex<S>>,
	rx: Receiver<Vec<ArcArray<f32, IxDyn>>>,
}

impl<S: DataStream + Send + 'static> Buffered<S> {
	pub fn new(stream: S, buffer_size: usize) -> Self {
		let (tx, rx) = sync_channel(buffer_size);

		let stream = Arc::new(Mutex::new(stream));
		let stream_clone = stream.clone();

		thread::spawn(move || {
			let mut prev = None; // Failed attempts to send store a value here

			// use two loops so that the mutex lock can be dropped when the buffer is full
			loop {
				let mut locked_stream = stream_clone.lock().unwrap();
				// keep lock while values are still being accepted by the buffer
				let err = loop {
					match tx.try_send(prev.take().unwrap_or_else(|| locked_stream.next())) {
						Ok(()) => {}
						Err(e) => break e,
					}
				};
				match err {
					TrySendError::Full(val) => prev = Some(val), // save the value and let the lock drop
					TrySendError::Disconnected(_val) => {
						return;
					} // let the thread die
				}
				thread::sleep(Duration::from_millis(10));
			}
		});

		Buffered { stream, rx }
	}

	/// Borrows the wrapped datastream
	///
	/// Blocks until the buffer is full and the lock can be acquired, holding the guard blocks the buffer thread
	pub fn inner(&self) -> MutexGuard<'_, S> {
		self.stream
			.lock()
			.expect("Buffer internal thread has poisoned the mutex")
	}

	/// Returns the wrapped datastream
	///
	/// Spinwaits until the buffer thread disconnects, if the bufferthread is blocked this can hang.
	pub fn into_inner(self) -> S {
		let Buffered { stream, rx } = self;
		drop(rx);
		let mut result = Arc::try_unwrap(stream);
		loop {
			match result {
				Ok(lock) => {
					return lock
						.into_inner()
						.expect("Buffer internal thread has poisoned the mutex");
				}
				Err(stream) => {
					thread::yield_now();
					result = Arc::try_unwrap(stream);
				}
			}
		}
	}
}

impl<S: DataStream + Send + 'static> DataStream for Buffered<S> {
	fn next(&mut self) -> Vec<ArcArray<f32, IxDyn>> {
		self.rx.recv().expect("Buffer internal thread has died")
	}
}

/// Concatenates the components of two `DataStream`s
pub struct Zip<S1: DataStream, S2: DataStream> {
	stream1: S1,
	stream2: S2,
}

impl<S1: DataStream, S2: DataStream> Zip<S1, S2> {
	pub fn new(stream1: S1, stream2: S2) -> Self {
		Zip { stream1, stream2 }
	}

	/// Borrows the first wrapped datastream
	pub fn inner1(&self) -> &S1 {
		&self.stream1
	}

	/// Borrows the second wrapped datastream
	pub fn inner2(&self) -> &S2 {
		&self.stream2
	}

	/// Returns the wrapped datastreams
	pub fn into_inner(self) -> (S1, S2) {
		let Self { stream1, stream2 } = self;
		(stream1, stream2)
	}
}

impl<S1: DataStream, S2: DataStream> DataStream for Zip<S1, S2> {
	fn next(&mut self) -> Vec<ArcArray<f32, IxDyn>> {
		let mut data = self.stream1.next();
		data.append(&mut self.stream2.next());
		data
	}
}

/// Alternates returning from each provided `DataStream`s
pub struct Interleave {
	streams: Vec<Box<dyn DataStream>>,
	next: usize,
}

impl Interleave {
	pub fn new<S1: DataStream + 'static, S2: DataStream + 'static>(stream1: S1, stream2: S2) -> Self {
		Interleave {
			streams: vec![],
			next: 0,
		}
		.and(stream1)
		.and(stream2)
	}

	pub fn and<S: DataStream + 'static>(mut self, stream: S) -> Self {
		self.streams.push(stream.boxed());
		self
	}

	/// Borrows the first wrapped datastream
	pub fn inner(&self) -> &[Box<dyn DataStream>] {
		&self.streams
	}

	/// Returns the wrapped datastreams
	pub fn into_inner(self) -> Vec<Box<dyn DataStream>> {
		self.streams
	}
}

impl DataStream for Interleave {
	fn next(&mut self) -> Vec<ArcArray<f32, IxDyn>> {
		let data = self.streams[self.next].next();
		self.next = (self.next + 1) % self.streams.len();
		data
	}
}

/// Keeps a count of how many times next is called().
pub struct Count<S: DataStream> {
	stream: S,
	count: usize,
}

impl<S: DataStream> Count<S> {
	pub fn new(stream: S) -> Self {
		Count { stream, count: 0 }
	}

	/// Returns a count of how many times next has been called.
	pub fn count(&self) -> usize {
		self.count
	}

	/// Borrows the wrapped datastreams.
	pub fn inner(&self) -> &S {
		&self.stream
	}

	/// Returns the wrapped datastreams.
	pub fn into_inner(self) -> S {
		let Self { stream, .. } = self;
		stream
	}
}

impl<S: DataStream> DataStream for Count<S> {
	fn next(&mut self) -> Vec<ArcArray<f32, IxDyn>> {
		self.count += 1;
		self.stream.next()
	}
}

/// Adds an outer batch dimension to each component by combining multiple elements.
///
/// Batch size must be greater than 0.
pub struct Batch<S: DataStream> {
	stream: S,
	batch_size: usize,
}

impl<S: DataStream> Batch<S> {
	pub fn new(stream: S, batch_size: usize) -> Self {
		assert!(batch_size > 0);
		Batch { stream, batch_size }
	}

	/// Borrows the wrapped datastreams.
	pub fn inner(&self) -> &S {
		&self.stream
	}

	/// Returns the wrapped datastreams.
	pub fn into_inner(self) -> S {
		let Self { stream, .. } = self;
		stream
	}
}

impl<S: DataStream> DataStream for Batch<S> {
	fn next(&mut self) -> Vec<ArcArray<f32, IxDyn>> {
		let mut batch_data: Vec<_> = self
			.stream
			.next()
			.into_iter()
			.map(|arr| {
				let batch_shape = iter::once(&self.batch_size)
					.chain(arr.shape())
					.cloned()
					.collect::<SmallVec<[usize; 6]>>();
				let mut batch_arr = unsafe { ArcArray::uninitialized(IxDyn(&batch_shape)) };
				batch_arr
					.index_axis_mut(Axis(0), 0)
					.as_slice_mut()
					.unwrap()
					.copy_from_slice(arr.as_slice().unwrap());
				batch_arr
			})
			.collect();

		for i in 1..self.batch_size {
			let input_vec = self.stream.next();
			assert_eq!(input_vec.len(), batch_data.len());
			for (input_arr, batch_arr) in input_vec.into_iter().zip(&mut batch_data) {
				let mut batch_view = (batch_arr as &mut ArcArray<f32, IxDyn>).index_axis_mut(Axis(0), i);
				assert_eq!(
					input_arr.shape(),
					batch_view.shape(),
					"Cannot batch arrays of different shapes."
				);
				batch_view
					.as_slice_mut()
					.unwrap()
					.copy_from_slice(input_arr.as_slice().unwrap());
			}
		}

		batch_data
	}
}

/// For each element, map the components to a completely new vec of components.
///
/// If names for the new components arent provided the old names will be used.
/// If the map changes the number of components, new names must be provided.
pub struct StreamMapAll<S: DataStream, F: FnMut(Vec<ArcArray<f32, IxDyn>>) -> Vec<ArcArray<f32, IxDyn>>> {
	func: F,
	stream: S,
}

impl<S: DataStream, F: FnMut(Vec<ArcArray<f32, IxDyn>>) -> Vec<ArcArray<f32, IxDyn>>> StreamMapAll<S, F> {
	pub fn new(stream: S, func: F) -> Self {
		StreamMapAll { func, stream }
	}

	/// Borrows the wrapped datastream.
	pub fn inner(&self) -> &S {
		&self.stream
	}

	/// Returns the wrapped datastream.
	pub fn into_inner(self) -> S {
		let Self { stream, .. } = self;
		stream
	}
}

impl<S: DataStream, F: FnMut(Vec<ArcArray<f32, IxDyn>>) -> Vec<ArcArray<f32, IxDyn>>> DataStream for StreamMapAll<S, F> {
	fn next(&mut self) -> Vec<ArcArray<f32, IxDyn>> {
		let data = self.stream.next();
		(self.func)(data) // that is some weird syntax
	}
}

/// For one component in each element of the datastream: apply a function.
///
/// Renaming the component is optional.
pub struct StreamMapOne<S: DataStream, F: FnMut(ArcArray<f32, IxDyn>) -> ArcArray<f32, IxDyn>> {
	func: F,
	stream: S,
	component: usize,
}

impl<S: DataStream, F: FnMut(ArcArray<f32, IxDyn>) -> ArcArray<f32, IxDyn>> StreamMapOne<S, F> {
	pub fn new(stream: S, func: F, component: usize) -> Self {
		StreamMapOne {
			func,
			stream,
			component,
		}
	}

	/// Borrows the wrapped datastream.
	pub fn inner(&self) -> &S {
		&self.stream
	}

	/// Returns the wrapped datastream.
	pub fn into_inner(self) -> S {
		let Self { stream, .. } = self;
		stream
	}
}

impl<S: DataStream, F: FnMut(ArcArray<f32, IxDyn>) -> ArcArray<f32, IxDyn>> DataStream for StreamMapOne<S, F> {
	fn next(&mut self) -> Vec<ArcArray<f32, IxDyn>> {
		let mut data = self.stream.next();
		let arr = mem::replace(&mut data[self.component], ArcArray::zeros(IxDyn(&[])));
		mem::replace(&mut data[self.component], (self.func)(arr));
		data
	}
}
