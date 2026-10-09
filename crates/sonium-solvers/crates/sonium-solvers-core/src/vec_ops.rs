//! Slice/Vec helpers that replace the ndarray method surface used internally
//! by the solvers crate.
//!
//! These are intentionally simple — they exist to keep the port off ndarray
//! mechanical. Hot paths should go through `sonium_compute::cpu::blas1` via
//! the `ComplexField::vec_*` methods.

use crate::traits::ComplexField;
use num_traits::Zero;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

pub trait SliceDot<T> {
    /// Unconjugated bilinear sum `Σ self[i] * rhs[i]`.
    ///
    /// This intentionally does NOT conjugate: the bound is generic over any
    /// scalar, and every in-repo caller is real-valued (identity for reals).
    /// Complex callers needing the Hermitian inner product must use
    /// [`ComplexField::vec_dot`] (`Σ conj(x) * y`) instead.
    fn dot(&self, rhs: &[T]) -> T;
}

impl<T> SliceDot<T> for [T]
where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
{
    #[inline]
    fn dot(&self, rhs: &[T]) -> T {
        assert_eq!(self.len(), rhs.len(), "vector length mismatch");
        self.iter()
            .zip(rhs.iter())
            .fold(T::zero(), |acc, (&a, &b)| acc + a * b)
    }
}

/// `out = a + b` (element-wise).
#[inline]
pub fn vec_add<T>(a: &[T], b: &[T]) -> Vec<T>
where
    T: Copy + Add<Output = T>,
{
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect()
}

/// `out = a - b` (element-wise).
#[inline]
pub fn vec_sub<T>(a: &[T], b: &[T]) -> Vec<T>
where
    T: Copy + Sub<Output = T>,
{
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| *x - *y).collect()
}

/// `out = -a` (element-wise).
#[inline]
pub fn vec_neg<T>(a: &[T]) -> Vec<T>
where
    T: Copy + Neg<Output = T>,
{
    a.iter().map(|x| -*x).collect()
}

/// `dst += src`.
#[inline]
pub fn vec_add_assign<T>(dst: &mut [T], src: &[T])
where
    T: Copy + AddAssign,
{
    assert_eq!(dst.len(), src.len(), "vector length mismatch");
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

/// `dst -= src`.
#[inline]
pub fn vec_sub_assign<T>(dst: &mut [T], src: &[T])
where
    T: Copy + SubAssign,
{
    assert_eq!(dst.len(), src.len(), "vector length mismatch");
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d -= *s;
    }
}

/// `out = a * alpha` (element-wise).
#[inline]
pub fn vec_scale<T>(a: &[T], alpha: T) -> Vec<T>
where
    T: Copy + Mul<Output = T>,
{
    a.iter().map(|&x| x * alpha).collect()
}

/// `out = a + alpha * b` (element-wise).
#[inline]
pub fn vec_add_scaled<T>(a: &[T], b: &[T], alpha: T) -> Vec<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| ai + alpha * bi)
        .collect()
}

/// Apply a function to each element, returning a new `Vec<T>`.
#[inline]
pub fn vec_mapv<T: Clone, U, F: FnMut(T) -> U>(a: &[T], mut f: F) -> Vec<U> {
    a.iter().cloned().map(&mut f).collect()
}

/// In-place map.
#[inline]
pub fn vec_mapv_inplace<T, F: FnMut(T) -> T>(a: &mut [T], mut f: F)
where
    T: Copy,
{
    for v in a.iter_mut() {
        *v = f(*v);
    }
}

/// Copy `src` into `dst` (length must match). Wrapper around `copy_from_slice`.
#[inline]
pub fn vec_assign<T: Copy>(dst: &mut [T], src: &[T]) {
    assert_eq!(dst.len(), src.len(), "vector length mismatch");
    dst.copy_from_slice(src);
}

/// `||x||_∞ = max(|x_i|)`.
#[inline]
pub fn vec_inf_norm<T: ComplexField>(x: &[T]) -> T::Real {
    use num_traits::Zero;
    let mut m = T::Real::zero();
    for v in x.iter() {
        let n = v.norm();
        if n > m {
            m = n;
        }
    }
    m
}
