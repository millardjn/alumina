use ndarray::{prelude::*, Data, FoldWhile, Zip};
use num_traits::Float;
use std::fmt::Debug;

pub trait RelClose<A: Float + Debug> {
	/// Similar to `all_close()` for ArrayBase, however it tests
	/// `(*x - *y).abs() <= tol * y.abs()`
	/// rather than
	/// `(*x - *y).abs() <= tol`
	fn all_relatively_close<S2, E2>(&self, rhs: &ArrayBase<S2, E2>, tol: A) -> bool
	where
		A: Float,
		S2: Data<Elem = A>,
		E2: Dimension;
}

impl<A: Float + Debug, S1: Data<Elem = A>, E1: Dimension> RelClose<A> for ArrayBase<S1, E1> {
	fn all_relatively_close<S2, E2>(&self, rhs: &ArrayBase<S2, E2>, tol: A) -> bool
	where
		S2: Data<Elem = A>,
		E2: Dimension,
	{
		!Zip::from(self)
			.and(rhs.broadcast(self.raw_dim()).expect("Broadcast failed"))
			.fold_while((), |_, &x, &y| {
				if (x - y).abs() <= tol * y.abs().max(A::one()) {
					FoldWhile::Continue(())
				} else {
					eprintln!("tolerance failed, found: {:?}, expect within {:?} of {:?}", x, tol, y);
					FoldWhile::Done(())
				}
			})
			.is_done()
	}
}
