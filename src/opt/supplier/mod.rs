#[cfg(feature = "mnist")]
pub mod mnist;
#[cfg(feature = "images")]
pub mod imagenet;
#[cfg(feature = "images")]
pub mod imagefolder;


use std::iter;
use graph::*;
use rand::*;

pub trait Supplier {
	fn next_n(&mut self, u: usize) -> (Vec<NodeData>, Vec<NodeData>);
	fn epoch_size(&self) -> usize;
	fn samples_taken(&self) -> u64;
	fn reset(&mut self);
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
	order: Box<Iterator<Item=usize>>,
}

impl Selector for ShuffleRandom {
	fn new(n: usize) -> ShuffleRandom{
		ShuffleRandom{
			n: n,
			order: Box::new(iter::empty()),
			}
	}

	fn next(&mut self) -> usize{
		match self.order.next() {
			Some(i) => i,
			None => {
				assert!(self.n > 0, "Cant generate indices over a zero size set.");
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
		Random{
			n: n,
			}
	}

	fn next(&mut self) -> usize{
		thread_rng().gen_range(0, self.n)
	}

	fn reset(&mut self){}
}