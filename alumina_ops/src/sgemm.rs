#[cfg(feature = "cblas")]
use std::convert::TryInto;
#[cfg(feature = "cblas")]
extern crate cblas_sys;
#[cfg(feature = "cblas")]
use cblas_sys::{cblas_sgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};

/// General matrix multiplication (f32)
///
/// C ← α A B + β C
///
/// + m, k, n: dimensions
/// + a, b, c: pointer to the first element in the matrix
/// + A: m by k matrix
/// + B: k by n matrix
/// + C: m by n matrix
/// + rs<em>x</em>: row stride of *x*
/// + cs<em>x</em>: col stride of *x*
///
/// Strides for A and B may be arbitrary. Strides for C must not result in
/// elements that alias each other, for example they can not be zero.
/// Stride naming is such that row strides (rsx) are the memory stride to the next row (along a column),
/// and col strides (csx) are the memory stride to the next column (along a row).
///
/// If β is zero, then C does not need to be initialized.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn sgemm(
	m: usize,
	k: usize,
	n: usize,
	alpha: f32,
	a: *const f32,
	rsa: isize,
	csa: isize,
	b: *const f32,
	rsb: isize,
	csb: isize,
	beta: f32,
	c: *mut f32,
	rsc: isize,
	csc: isize,
) {
	#[cfg(feature = "cblas")]
	{
		if let Ok(()) = try_sgemm_cblas(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc) {
			return;
		}
		panic!();
	}

	matrixmultiply_mt::sgemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc);
}

#[cfg(feature = "cblas")]
#[allow(clippy::too_many_arguments)]
unsafe fn try_sgemm_cblas(
	m: usize,
	k: usize,
	n: usize,
	alpha: f32,
	a: *const f32,
	rsa: isize,
	csa: isize,
	b: *const f32,
	rsb: isize,
	csb: isize,
	beta: f32,
	c: *mut f32,
	rsc: isize,
	csc: isize,
) -> Result<(), ()> {
	if !(((rsa == 1 && csa > 1) || (rsa > 1 && csa == 1))
		&& ((rsb == 1 && csb > 1) || (rsb > 1 && csb == 1))
		&& ((rsc == 1 && csc > 1) || (rsc > 1 && csc == 1)))
	{
		return Err(());
	}

	let m: i32 = m.try_into().unwrap();
	let n: i32 = n.try_into().unwrap();
	let k: i32 = k.try_into().unwrap();

	let rsa: i32 = rsa.try_into().unwrap();
	let csa: i32 = csa.try_into().unwrap();
	let rsb: i32 = rsb.try_into().unwrap();
	let csb: i32 = csb.try_into().unwrap();
	let rsc: i32 = rsc.try_into().unwrap();
	let csc: i32 = csc.try_into().unwrap();

	let (layouta, lda) = if csa == 1 && rsa == k {
		(CBLAS_LAYOUT::CblasRowMajor, k)
	} else if rsa == 1 && csa == m {
		(CBLAS_LAYOUT::CblasColMajor, m)
	} else {
		return Err(());
	};

	let (layoutb, ldb) = if csb == 1 && rsb == n {
		(CBLAS_LAYOUT::CblasRowMajor, n)
	} else if rsb == 1 && csb == k {
		(CBLAS_LAYOUT::CblasColMajor, k)
	} else {
		return Err(());
	};

	let (layoutc, ldc) = if csc == 1 && rsc == n {
		(CBLAS_LAYOUT::CblasRowMajor, n)
	} else if rsc == 1 && csc == m {
		(CBLAS_LAYOUT::CblasColMajor, m)
	} else {
		return Err(());
	};

	let transa = match (layouta, layoutc) {
		(CBLAS_LAYOUT::CblasRowMajor, CBLAS_LAYOUT::CblasRowMajor)
		| (CBLAS_LAYOUT::CblasColMajor, CBLAS_LAYOUT::CblasColMajor) => CBLAS_TRANSPOSE::CblasNoTrans,
		_ => CBLAS_TRANSPOSE::CblasTrans,
	};
	let transb = match (layoutb, layoutc) {
		(CBLAS_LAYOUT::CblasRowMajor, CBLAS_LAYOUT::CblasRowMajor)
		| (CBLAS_LAYOUT::CblasColMajor, CBLAS_LAYOUT::CblasColMajor) => CBLAS_TRANSPOSE::CblasNoTrans,
		_ => CBLAS_TRANSPOSE::CblasTrans,
	};

	cblas_sgemm(layoutc, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

	Ok(())
}
