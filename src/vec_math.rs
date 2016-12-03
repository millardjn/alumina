pub trait VecMath {
	fn scale(&self, scale: f32) -> Vec<f32>;	
	fn add(&self, other: &[f32]) -> Vec<f32>;	
	fn add_scaled(&self, other: &[f32], scale: f32) -> Vec<f32>;	
	fn normalise(&self) -> Vec<f32>;
	fn dot(&self, other: &[f32]) -> f32;
	fn cos_similarity(&self, other: &[f32]) -> f32;
	fn norm2(&self) -> f32;
	fn elementwise_sqrt(&self) -> Vec<f32>;
	fn elementwise_mul(&self, other: &[f32]) -> Vec<f32>;
	fn elementwise_div(&self, other: &[f32]) -> Vec<f32>;
}

pub trait VecMathMut {
	fn scale_mut(&mut self, scale: f32) -> &mut[f32];
	fn add_mut(&mut self, other: &[f32]) -> &mut[f32];	
	fn add_scaled_mut(&mut self, other: &[f32], scale: f32) -> &mut[f32];	
	fn normalise_mut(&mut self) -> &mut[f32];
}

pub trait VecMathMove {
	fn scale_move(self, scale: f32) -> Vec<f32>;
	fn add_move(self, other: &[f32]) -> Vec<f32>;	
	fn add_scaled_move(self, other: &[f32], scale: f32) -> Vec<f32>;	
	fn normalise_move(self) -> Vec<f32>;
}

impl VecMathMove for Vec<f32> {
	fn scale_move(mut self, scale: f32) -> Vec<f32>{
		self.iter_mut().map(|s| *s *= scale).last();
		self
	}
	fn add_move(mut self, other: &[f32]) -> Vec<f32>{
		self.iter_mut().zip(other.iter()).map(|(s, o)| *s += *o).last();
		self
	}
	fn add_scaled_move(mut self, other: &[f32], scale: f32) -> Vec<f32>{
		self.iter_mut().zip(other.iter()).map(|(s, o)| *s += scale*o).last();
		self
	}
	fn normalise_move(mut self) -> Vec<f32> {
		let norm2 = self.norm2();
		self.iter_mut().map(|s| *s /= norm2).last();
		self
	}
}



impl VecMathMut for Vec<f32> {
	fn scale_mut(&mut self, scale: f32) -> &mut[f32]{
		self.iter_mut().map(|s| *s *= scale).last();
		self
	}
	fn add_mut(&mut self, other: &[f32]) -> &mut[f32]{
		self.iter_mut().zip(other.iter()).map(|(s, o)| *s += *o).last();
		self
	}
	fn add_scaled_mut(&mut self, other: &[f32], scale: f32) -> &mut[f32]{
		self.iter_mut().zip(other.iter()).map(|(s, o)| *s += scale*o).last();
		self
	}
	fn normalise_mut(&mut self) -> &mut[f32] {
		let norm2 = self.norm2();
		self.iter_mut().map(|s| *s /= norm2).last();
		self
	}
}

impl VecMathMut for [f32] {
	fn scale_mut(&mut self, scale: f32) -> &mut[f32]{
		self.iter_mut().map(|s| *s *= scale).last();
		self
	}
	fn add_mut(&mut self, other: &[f32]) -> &mut[f32]{
		self.iter_mut().zip(other.iter()).map(|(s, o)| *s += *o).last();
		self
	}
	fn add_scaled_mut(&mut self, other: &[f32], scale: f32) -> &mut[f32]{
		self.iter_mut().zip(other.iter()).map(|(s, o)| *s += scale*o).last();
		self
	}
	fn normalise_mut(&mut self) -> &mut[f32] {
		let norm2 = self.norm2();
		self.iter_mut().map(|s| *s /= norm2).last();
		self
	}
}

impl VecMath for Vec<f32> {
	
	fn scale(&self, scale: f32) -> Vec<f32>{
		self.iter().map(|s| s*scale).collect()
	}
	
	fn add(&self, other: &[f32]) -> Vec<f32>{
		self.iter().zip(other.iter()).map(|(s, o)| s + o).collect()
	}
	
	fn add_scaled(&self, other: &[f32], scale: f32) -> Vec<f32>{
		self.iter().zip(other.iter()).map(|(s, o)| s + scale*o).collect()
	}
	fn normalise(&self) -> Vec<f32>{
		let norm2 = self.norm2();
		self.iter().map(|s| *s/norm2).collect()
	}
	fn dot(&self, other: &[f32]) -> f32{
		dot(&other, &self)
	}
	
	fn cos_similarity(&self, other: &[f32]) -> f32 {
		cosine_similarity(&self, other)
	}
	
	fn norm2(&self) -> f32{
		self.dot(&self).sqrt()
	}
	
	fn elementwise_sqrt(&self) -> Vec<f32> {
		self.iter().map(|x| x.sqrt()).collect()
	}
	
	fn elementwise_mul(&self, other: &[f32]) -> Vec<f32> {
		self.iter().zip(other).map(|(s, o)| s*o).collect()
	}
	
	fn elementwise_div(&self, other: &[f32]) -> Vec<f32> {
		self.iter().zip(other).map(|(s, o)| s/o).collect()
	}
}

impl VecMath for [f32] {
	
	fn scale(&self, scale: f32) -> Vec<f32>{
		self.iter().map(|s| s*scale).collect()
	}
	
	fn add(&self, other: &[f32]) -> Vec<f32>{
		self.iter().zip(other.iter()).map(|(s, o)| s + o).collect()
	}
	
	fn add_scaled(&self, other: &[f32], scale: f32) -> Vec<f32>{
		self.iter().zip(other.iter()).map(|(s, o)| s + scale*o).collect()
	}
	fn normalise(&self) -> Vec<f32>{
		let norm2 = self.norm2();
		self.iter().map(|s| *s/norm2).collect()
	}
	fn dot(&self, other: &[f32]) -> f32{
		dot(&other, &self)
	}
	
	fn cos_similarity(&self, other: &[f32]) -> f32 {
		cosine_similarity(&self, other)
	}
	
	fn norm2(&self) -> f32{
		self.dot(&self).sqrt()
	}
	
	fn elementwise_sqrt(&self) -> Vec<f32> {
		self.iter().map(|x| x.sqrt()).collect()
	}
	
	fn elementwise_mul(&self, other: &[f32]) -> Vec<f32> {
		self.iter().zip(other).map(|(s, o)| s*o).collect()
	}
	
	fn elementwise_div(&self, other: &[f32]) -> Vec<f32> {
		self.iter().zip(other).map(|(s, o)| s/o).collect()
	}
}



pub const SIMD_WIDTH: usize = 8;

macro_rules! loop4 {
    ($i:ident, $e:expr) => {{
        let $i = 0; $e;
        let $i = 1; $e;
        let $i = 2; $e;
        let $i = 3; $e;
    }}
}

macro_rules! loop8 {
    ($i:ident, $e:expr) => {{
        let $i = 0; $e;
        let $i = 1; $e;
        let $i = 2; $e;
        let $i = 3; $e;
        let $i = 4; $e;
        let $i = 5; $e;
        let $i = 6; $e;
        let $i = 7; $e;
    }}
}


//#[inline(never)]
pub fn sum_acc(acc: &mut [f32; SIMD_WIDTH]) -> f32 {
	let mut half = SIMD_WIDTH/2;
	while half > 0 {
		for i in 0..half{
			acc[i] += acc[i+half];
		}
		half /=2;
	}
	acc[0]
}

pub fn dot(xs: &[f32], ys: &[f32]) -> f32 {
	let mut acc = [0.; SIMD_WIDTH];
	//dot_acc(xs, ys, &mut acc);
	dot_vec(xs, ys, &mut acc);
	sum_acc(&mut acc)
}

use std::cmp;
//#[inline(never)]
fn dot_vec(xs: &[f32], ys: &[f32], acc_: &mut [f32; SIMD_WIDTH]){
    let n = cmp::min(ys.len(), xs.len());
	let mut v1 = &xs[..n];
	let mut v2 = &ys[..n];
	let mut acc = *acc_;

	//assert!(v1.len() == v2.len()); Assert doesnt provide enough of a guarantee to enable vectorization?

	while v1.len() >= SIMD_WIDTH {
	
		loop8!(i, acc[i] += v1[i] * v2[i]);
//		acc[0] += v1[0] * v2[0];
//		acc[1] += v1[1] * v2[1];
//		acc[2] += v1[2] * v2[2];
//		acc[3] += v1[3] * v2[3];
//		acc[4] += v1[4] * v2[4];
//		acc[5] += v1[5] * v2[5];
//		acc[6] += v1[6] * v2[6];
//		acc[7] += v1[7] * v2[7];
		
		v1 = &v1[SIMD_WIDTH..];
		v2 = &v2[SIMD_WIDTH..];
	}

    *acc_ = acc;
    
    for i in 0..v1.len() {
  		acc_[i] += v1[i]*v2[i];
  	}
}


// fn cosine_similarity(xs: &[f32], ys: &[f32]) -> f32 {
// 	let mut xn = 0.;
// 	let mut yn = 0.;
// 	let mut dn = 0.;
	
// 	let n = xs.len();
// 	let ys = &ys[..n];

//     for i in 0..xs.len() {
//   		xn += xs[i] * xs[i];
// 		yn += ys[i] * ys[i];
// 		dn += xs[i] * ys[i];
//   	}
	
// 	if xn*yn == 0.0 {
// 		0.0
// 	} else {
// 		dn/(xn*yn).sqrt()
// 	}
	
// }


fn cosine_similarity(xs: &[f32], ys: &[f32]) -> f32 {

	let xn = dot(xs, xs);
	let yn = dot(ys, ys);
	let dn = dot(xs, ys);
	
	let res = dn/(xn*yn).sqrt();
	if res.is_nan(){
		0.0
	} else {
		res
	}
	
}