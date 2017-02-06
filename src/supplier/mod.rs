#[cfg(feature = "mnist")]
pub mod mnist;
#[cfg(feature = "images")]
pub mod imagenet;
#[cfg(feature = "images")]
pub mod imagefolder;


use std::iter;
use graph::*;
use rand::*;
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread;

pub trait Supplier {
	fn next_n(&mut self, n: usize) -> (Vec<NodeData>, Vec<NodeData>);
	fn epoch_size(&self) -> usize;
	fn samples_taken(&self) -> u64;
	fn reset(&mut self);
}

pub struct Buffer {
	rx: Receiver<(Vec<NodeData>, Vec<NodeData>)>,
	epoch_size: usize,
	count: u64,
}

impl Buffer {
	pub fn new<S: Supplier + Send + 'static> (mut s: S, size: usize) -> Buffer{
		let (tx, rx) = sync_channel(size);
		let epoch_size = s.epoch_size();
		let count = s.samples_taken();

		thread::spawn(move|| {
			
			'l1: loop {
				match tx.send(s.next_n(1)) {
					Ok(_) => (),
					Err(_) => break 'l1,
				}
			}
			
		});

		Buffer{
			rx: rx,
			epoch_size: epoch_size,
			count: count,
		}
	}
}

impl Supplier for Buffer{
	fn next_n(&mut self, n: usize) -> (Vec<NodeData>, Vec<NodeData>){
		assert!(n > 0, "n must be larger than 0");

		let (mut input, mut train_input) = self.rx.recv().expect("Buffer internal thread has died");
		
		for nd in input.iter_mut() {
			nd.reserve_exact(n-1);
		}

		for nd in train_input.iter_mut() {
			nd.reserve_exact(n-1);
		}

		for _ in 1..n {
			let (input2, train_input2) = self.rx.recv().expect("Buffer internal thread has died");

			for (i, nd) in input2.into_iter().enumerate() {
				input[i].join_mut(nd);
			}

			for (i, nd) in train_input2.into_iter().enumerate() {
				train_input[i].join_mut(nd);
			}

		}

		// for nd in input.iter_mut() {
		// 	nd.shrink_to_fit();
		// }

		// for nd in train_input.iter_mut() {
		// 	nd.shrink_to_fit();
		// }

		self.count += n as u64;
		(input, train_input)
	}

	fn epoch_size(&self) -> usize{
		self.epoch_size
	}

	fn samples_taken(&self) -> u64{
		self.count
	}

	fn reset(&mut self){
		unimplemented!();
	}
}

pub trait Selector {
	fn new(n: usize) -> Self;
	fn next(&mut self) -> usize;
	fn reset(&mut self);
}


pub struct Sequential {
	n: usize,
	count: usize,
}

impl Selector for Sequential{
	fn new(n: usize) -> Sequential{
		assert!(n > 0, "Cant generate indices over a zero size set.");
		Sequential{
			n: n,
			count: 0,
			}
	}

	fn next(&mut self) -> usize{
		let ind = self.count;
		self.count = (self.count + 1)%self.n;
		ind
	}

	fn reset(&mut self){self.count = 0;}
}

/// 
pub struct ShuffleRandom {
	n: usize,
	order: Box<Iterator<Item=usize> + Send>,
}

impl Selector for ShuffleRandom {
	fn new(n: usize) -> ShuffleRandom{
		assert!(n > 0, "Cant generate indices over a zero size set.");
		ShuffleRandom{
			n: n,
			order: Box::new(iter::empty()),
			}
	}

	fn next(&mut self) -> usize{
		match self.order.next() {
			Some(i) => i,
			None => {
				let mut v: Vec<usize> = (0..self.n).collect();
				let mut rng = thread_rng();
				rng.shuffle(&mut v);
				
				self.order = Box::new(v.into_iter());
				
				self.next()
			},
		}

	}

	fn reset(&mut self){
		self.order = Box::new(iter::empty());
	}
}

pub struct Random {
	n: usize,
}

impl Selector for Random {
	fn new(n: usize) -> Random{
		assert!(n > 0, "Cant generate indices over a zero size set.");
		Random{
			n: n,
			}
	}

	fn next(&mut self) -> usize{
		thread_rng().gen_range(0, self.n)
	}

	fn reset(&mut self){}
}