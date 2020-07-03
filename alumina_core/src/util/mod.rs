pub mod display;

pub fn wrap_dim(index: isize, len: usize) -> usize {
	(index % len as isize + len as isize) as usize % len
}
