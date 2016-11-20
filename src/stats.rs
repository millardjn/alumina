
	
pub fn mean(values: &[f32]) -> f32{		
	values.iter().fold(0.0, |acc, &x| acc + x)/values.len() as f32
}

pub fn variance_population (v: &[f32]) -> f32{
	let mean = mean(v);
	v.iter().fold(0.0, |acc, &x| acc + (x - mean).powi(2)) /v.len() as f32
}

pub fn variance_sample (v: &[f32]) -> f32{
	let n = v.len() as f32;
	variance_population(v)*n/(n-1.0)
}

pub fn variance_of_mean_pop(v: &[f32]) -> f32{
	let n = v.len() as f32;
	variance_population(v)/n
}

pub fn variance_of_mean_sample(v: &[f32]) -> f32{
	let n = v.len() as f32;
	variance_sample(v)/n
}

pub fn t_test_one_sample(v: &[f32]) -> f32{
	mean(v)/variance_of_mean_sample(v).sqrt()
}

pub fn t_test_two_samples(v1: &[f32], v2: &[f32]) -> f32{
	(mean(v1) - mean(v2))/(variance_of_mean_sample(v1) + variance_of_mean_sample(v2)).sqrt()
}

pub fn dof_one_sample(v: &[f32]) -> f32 {
	(v.len() - 1) as f32
}

pub fn dof_two_samples(v1: &[f32], v2: &[f32]) -> f32 {
	let n1 = v1.len() as f32;
	let n2 = v2.len() as f32;
	let vms1 = variance_of_mean_sample(v1);
	let vms2 = variance_of_mean_sample(v2);
	
	(vms1 + vms2).powi(2) / (vms1.powi(2)/(n1 -1.0) + vms2.powi(2)/(n2 - 1.0))
}


pub fn erf(x: f32) -> f32 {
	1.0 - erfc(x)
}

/// from numerical recipies
pub fn erfc(x: f32) -> f32{

	let z: f64 = x.abs() as f64;
	let t: f64 = 2./(2.+z);
	
	let ans = (-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+ t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+ t*(-0.82215223+t*0.17087277))))))))).exp() * t;
	
	(if x >= 0.0 {
		ans
	} else {
		2.0 - ans
	}) as f32
	
}

	
/// CDF for normal dist, p (0,1)
pub fn norm_cfd (x: f32) -> f32{
	0.5 * (1.0 + erf(x/2f32.sqrt()))
}

// inv_CDF for normal dist, returns z for a given probability (capped at +- 20, because seriously...)
pub fn probit(p: f32) -> f32 {
	let mut lz = -20.0;
	let mut rz = 20.0;
	let mut lp = norm_cfd(lz);
	let mut rp = norm_cfd(rz);
	
	if p < lp {
		return lz;
	} else if p > rp {
		return rz;
	}
	
	let mut i = 0;
	while rz-lz > 1e-6  && i < 10000{
		let r = (p - lp)/(rp-lp);
		let new = lz + r*(rz-lz);
		let new_p = norm_cfd(new);
		
		if (p-new_p).abs() < 1e-6 {
			return new;
		} else if new_p < p {
			lz = new;
			lp = new_p;
		} else {
			rz = new;
			rp = new_p;
		}
		i += 1;
	}
	
	(rz+lz)/2.0
}

// Using Cornish-Fisher approach from Shaw, W. - "NEW METHODS FOR SIMULATING THE STUDENT T-DISTRIBUTION"
pub fn t_dist_inv_cfd(p: f32, n: f32) -> f32 {
	
	let z = probit(p) as f64;
	if n > 1e6 {
		z as f32
	} else {
		let n: f64 = n as f64;
		
		let t = z
		+ z*(z.powi(2) + 1.0)/(4.*n)
		+ z*(5.0*z.powi(4) + 16.0*z.powi(2) + 3.0)/(96.0*n.powi(2))
		+ z*(3.0*z.powi(7) + 19.0*z.powi(5) + 17.0*z.powi(3) - 15.0*z)/(384.0*n.powi(3))
		+ z*(79.0*z.powi(9) + 776.0*z.powi(7) + 1482.0*z.powi(5) - 1920.0*z.powi(3) - 945.0*z)/(92160.0*n.powi(4));
		
		t as f32
	}
}

#[test]
fn check_t_dist() {
	
	// bad at low n and small/large p
	assert!((-2.015048373 - t_dist_inv_cfd(0.05, 5.0)).abs() < 1e-1);
	
	//ok
	assert!((-0.816496581 - t_dist_inv_cfd(0.25, 2.0)).abs() < 1e-2);
	assert!((-0.764892328 - t_dist_inv_cfd(0.25, 3.0)).abs() < 1e-2);
	assert!((-0.726686844 - t_dist_inv_cfd(0.25, 5.0)).abs() < 1e-2);
	assert!((0.437501264 - t_dist_inv_cfd(0.66, 5.0)).abs() < 1e-2);
	assert!((-1.812461123 - t_dist_inv_cfd(0.05, 10.0)).abs() < 1e-2);
	
	//good
	assert!((-1.724718243 - t_dist_inv_cfd(0.05, 20.0)).abs() < 1e-3);
	assert!((-2.364217366 - t_dist_inv_cfd(0.01, 100.0)).abs() < 1e-3);
}
	
//	#[test]
//	fn check_erf() {
//
//		let x = -1.0;
//		let erf = erf(x);
//		println!("erf:{} x:{}", erf, x);
//	}


pub mod gamma_func{
	// http://codereview.stackexchange.com/questions/116850/gamma-function-in-rust
	// initial code in rust by Dair, cleaned up by Shepmaster
	
	const TAYLOR_COEFFICIENTS: [f64; 29] = [
	    -0.00000000000000000023,  0.00000000000000000141,  0.00000000000000000119,
	    -0.00000000000000011813,  0.00000000000000122678, -0.00000000000000534812,
	    -0.00000000000002058326,  0.00000000000051003703, -0.00000000000369680562,
	     0.00000000000778226344,  0.00000000010434267117, -0.00000000118127457049,
	     0.00000000500200764447,  0.00000000611609510448, -0.00000020563384169776,
	     0.00000113302723198170, -0.00000125049348214267, -0.00002013485478078824,
	     0.00012805028238811619, -0.00021524167411495097, -0.00116516759185906511,
	     0.00721894324666309954, -0.00962197152787697356, -0.04219773455554433675,
	     0.16653861138229148950, -0.04200263503409523553, -0.65587807152025388108,
	     0.57721566490153286061,  1.00000000000000000000,
	];
	
	const INITIAL_SUM: f64 = 0.00000000000000000002;
	
	pub fn gamma(x: f64) -> f64 {
	    TAYLOR_COEFFICIENTS.iter().fold(INITIAL_SUM, |sum, coefficient| {
	        sum * (x - 1.0) + coefficient
	    }).recip()
	}
}