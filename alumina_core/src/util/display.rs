use std::fmt::{Debug, Display, Formatter, Result};

/// Wrapper type to impl display for iterable types for use with Failure
///
/// `vec![1, 2, 3]` should display as `[1, 2, 3]`.
pub struct IterDisplay<T, I>
where
	for<'a> &'a I: IntoIterator<Item = &'a T>,
	T: Display,
{
	pub inner: I,
}

impl<T, I> Display for IterDisplay<T, I>
where
	for<'a> &'a I: IntoIterator<Item = &'a T>,
	T: Display,
{
	fn fmt(&self, f: &mut Formatter) -> Result {
		let mut inner = self.inner.into_iter();
		write!(f, "[")?;
		if let Some(d) = inner.next() {
			write!(f, "{}", d)?;
			for d in inner {
				write!(f, ", {}", d)?;
			}
		}
		write!(f, "]")?;
		Ok(())
	}
}
impl<T, I> Debug for IterDisplay<T, I>
where
	for<'a> &'a I: IntoIterator<Item = &'a T>,
	T: Display,
{
	fn fmt(&self, fmt: &mut Formatter) -> Result {
		Display::fmt(self, fmt)
	}
}

pub struct IterDebug<T, I>
where
	for<'a> &'a I: IntoIterator<Item = &'a T>,
	T: Debug,
{
	pub inner: I,
}

impl<T, I> Debug for IterDebug<T, I>
where
	for<'a> &'a I: IntoIterator<Item = &'a T>,
	T: Debug,
{
	fn fmt(&self, fmt: &mut Formatter) -> Result {
		let mut list = fmt.debug_list();
		for i in self.inner.into_iter() {
			list.entry(i);
		}
		list.finish()
	}
}
impl<T, I> Display for IterDebug<T, I>
where
	for<'a> &'a I: IntoIterator<Item = &'a T>,
	T: Debug,
{
	fn fmt(&self, fmt: &mut Formatter) -> Result {
		Debug::fmt(self, fmt)
	}
}

/// Wrapper type to impl display for iterable types for use with Failure
///
/// `indexmap!["a"=>1, "b"=>2, "c"=>3]` should display as `[a: 1, b: 2, c: 3]`.
pub struct Iter2Display<T1, T2, I>
where
	for<'a> &'a I: IntoIterator<Item = (&'a T1, &'a T2)>,
	T1: Display,
	T2: Display,
{
	pub inner: I,
}
impl<T1, T2, I> Display for Iter2Display<T1, T2, I>
where
	for<'a> &'a I: IntoIterator<Item = (&'a T1, &'a T2)>,
	T1: Display,
	T2: Display,
{
	fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
		let mut iter = self.inner.into_iter();
		if let Some((ref op, ref error)) = iter.next() {
			write!(f, "{}: {}", op, error)?;
			for (ref op, ref error) in iter {
				write!(f, ", {}: {}", op, error)?;
			}
		}
		write!(f, "]")?;
		Ok(())
	}
}
impl<T1, T2, I> Debug for Iter2Display<T1, T2, I>
where
	for<'a> &'a I: IntoIterator<Item = (&'a T1, &'a T2)>,
	T1: Display,
	T2: Display,
{
	fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
		write!(f, "{}", self)?;
		Ok(())
	}
}
