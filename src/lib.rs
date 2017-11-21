#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]
#![recursion_limit = "1024"]

#[macro_use] extern crate error_chain;
extern crate ndarray;
extern crate odds;
extern crate walkdir;
extern crate smallvec;
extern crate rand;
extern crate matrixmultiply_mt as matrixmultiply;
extern crate byteorder;

extern crate num_cpus;
extern crate threadpool;
#[macro_use] extern crate lazy_static;
extern crate typenum;
extern crate typenum_loops;
extern crate generic_array;
extern crate scoped_threadpool;
extern crate ordermap;
#[macro_use] extern crate matches;
extern crate arrayvec;

pub mod graph;
pub mod ops;
pub mod opt;
pub mod supplier;
pub mod shape;
pub mod vec_math;
pub mod new;
#[cfg(test)]mod test;


// /// error_chain 
// mod errors {
// 	error_chain!{
// 		links {
// 			Another(other_error::Error, other_error::ErrorKind) #[cfg(unix)];
// 		}
// 	}
// }