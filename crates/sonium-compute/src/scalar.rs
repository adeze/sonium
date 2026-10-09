//! Scalar traits for device-resident data.
//!
//! Two traits split the responsibilities:
//!
//! - [`BufferElement`]: anything that can live in a device buffer. Includes
//!   storage-only low-precision types and index types like `i32` (used for CSR
//!   row/column pointers).
//! - [`DeviceScalar`]: a numeric scalar that BLAS operations are defined
//!   over (`f32`, `f64`, `Complex32`, `Complex64`). Subtype of
//!   `BufferElement`.
//!
//! This split lets backends store CSR index buffers (`Buffer<i32>`) without
//! also requiring `Float`-like arithmetic on `i32`.

use num_complex::{Complex32, Complex64};
use num_traits::{Float, FromPrimitive, Zero};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Marker for any plain-old-data type that can live in a device buffer.
///
/// `Default` is required so that backends can allocate host-staging
/// buffers (`vec![T::default(); n]`) without needing to know the concrete
/// scalar type. All current impls derive `Default` to zero, so this is a
/// no-op on the supported set.
pub trait BufferElement: Copy + Default + Send + Sync + Debug + 'static {}

impl BufferElement for f32 {}
impl BufferElement for f64 {}
impl BufferElement for Complex32 {}
impl BufferElement for Complex64 {}
impl BufferElement for half::bf16 {}
impl BufferElement for half::f16 {}
impl BufferElement for i32 {}
impl BufferElement for u32 {}
impl BufferElement for usize {}

/// A numeric scalar over which BLAS operations are defined.
///
/// Implemented for `f32`, `f64`, `Complex32`, `Complex64`. The f16/bf16 types
/// are [`BufferElement`] storage types only; backend BLAS does not operate on
/// them directly.
pub trait DeviceScalar:
    BufferElement
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Real-valued result of reductions like `nrm2`.
    type Real: Float + FromPrimitive + Copy + Send + Sync + Debug + 'static;

    /// Construct from a real value (imaginary part zero for complex scalars).
    fn from_real(r: Self::Real) -> Self;

    /// Multiplicatively neutral element (`1`).
    fn one() -> Self;

    /// Additively neutral element (`0`).
    fn zero() -> Self;

    /// `Σ conj(x_i) * y_i`. Hermitian inner product; for real types this is
    /// just the dot product.
    fn conj_mul(self, other: Self) -> Self;

    /// Squared magnitude `|self|²`.
    fn norm_sqr(&self) -> Self::Real;

    /// Magnitude `|self|`, evaluated without squaring first.
    ///
    /// This matters for norm reductions: a finite component can have an
    /// overflowing squared magnitude (for example `f64::MAX`).
    fn magnitude(&self) -> Self::Real {
        self.norm_sqr().sqrt()
    }

    /// Whether every component of the scalar is finite.
    ///
    /// The default is suitable for ordinary magnitudes. Implementations for
    /// the built-in scalars override it so very large finite values are not
    /// rejected merely because squaring them overflows.
    fn is_finite(&self) -> bool {
        self.norm_sqr().is_finite()
    }
}

/// A `DeviceScalar` that has a paired lower-precision variant for
/// mixed-precision iterative refinement (Carson-Higham 2018):
///
/// - `f64` ↔ `f32`
/// - `Complex64` ↔ `Complex32`
///
/// The outer Krylov runs at `Self`; the inner Krylov runs at
/// `Self::Lower`. The cast functions are required to round-trip
/// values without changing semantics beyond representational
/// precision loss (real → real cast, complex → complex cast that
/// preserves the (re, im) pair).
pub trait LowerPrecision: DeviceScalar {
    /// The matching lower-precision scalar.
    type Lower: DeviceScalar;

    /// Cast `self` to its lower-precision representation.
    fn cast_to_lower(self) -> Self::Lower;

    /// Cast a lower-precision value back to `Self`.
    fn cast_from_lower(low: Self::Lower) -> Self;

    /// Cast a real-valued scalar of `Self::Real` to the lower
    /// precision's real type. Used by preconditioners that need to
    /// pass eigenvalue bounds, damping coefficients, etc. between
    /// the two precision levels.
    fn cast_real_to_lower(r: Self::Real) -> <Self::Lower as DeviceScalar>::Real;
}

impl LowerPrecision for f64 {
    type Lower = f32;
    #[inline]
    fn cast_to_lower(self) -> f32 {
        self as f32
    }
    #[inline]
    fn cast_from_lower(low: f32) -> Self {
        low as f64
    }
    #[inline]
    fn cast_real_to_lower(r: f64) -> f32 {
        r as f32
    }
}

impl LowerPrecision for Complex64 {
    type Lower = Complex32;
    #[inline]
    fn cast_to_lower(self) -> Complex32 {
        Complex32::new(self.re as f32, self.im as f32)
    }
    #[inline]
    fn cast_from_lower(low: Complex32) -> Self {
        Complex64::new(low.re as f64, low.im as f64)
    }
    #[inline]
    fn cast_real_to_lower(r: f64) -> f32 {
        r as f32
    }
}

impl DeviceScalar for f32 {
    type Real = f32;
    #[inline]
    fn from_real(r: f32) -> Self {
        r
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn conj_mul(self, other: Self) -> Self {
        self * other
    }
    #[inline]
    fn norm_sqr(&self) -> Self::Real {
        *self * *self
    }

    #[inline]
    fn magnitude(&self) -> Self::Real {
        self.abs()
    }
    #[inline]
    fn is_finite(&self) -> bool {
        f32::is_finite(*self)
    }
}

impl DeviceScalar for f64 {
    type Real = f64;
    #[inline]
    fn from_real(r: f64) -> Self {
        r
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn conj_mul(self, other: Self) -> Self {
        self * other
    }
    #[inline]
    fn norm_sqr(&self) -> Self::Real {
        *self * *self
    }

    #[inline]
    fn magnitude(&self) -> Self::Real {
        self.abs()
    }
    #[inline]
    fn is_finite(&self) -> bool {
        f64::is_finite(*self)
    }
}

impl DeviceScalar for Complex32 {
    type Real = f32;
    #[inline]
    fn from_real(r: f32) -> Self {
        Complex32::new(r, 0.0)
    }
    #[inline]
    fn one() -> Self {
        Complex32::new(1.0, 0.0)
    }
    #[inline]
    fn zero() -> Self {
        Complex32::new(0.0, 0.0)
    }
    #[inline]
    fn conj_mul(self, other: Self) -> Self {
        self.conj() * other
    }
    #[inline]
    fn norm_sqr(&self) -> Self::Real {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    fn magnitude(&self) -> Self::Real {
        self.norm()
    }
    #[inline]
    fn is_finite(&self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

impl DeviceScalar for Complex64 {
    type Real = f64;
    #[inline]
    fn from_real(r: f64) -> Self {
        Complex64::new(r, 0.0)
    }
    #[inline]
    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }
    #[inline]
    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }
    #[inline]
    fn conj_mul(self, other: Self) -> Self {
        self.conj() * other
    }
    #[inline]
    fn norm_sqr(&self) -> Self::Real {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    fn magnitude(&self) -> Self::Real {
        self.norm()
    }
    #[inline]
    fn is_finite(&self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

/// Compute an L2 norm with scaled accumulation.
///
/// The usual `sum(value.norm_sqr())` reduction can turn a finite result into
/// infinity before the square root. Keeping the largest magnitude as a scale
/// avoids that intermediate overflow while retaining a deterministic input
/// order for the accumulation.
pub(crate) fn stable_nrm2<T: DeviceScalar>(values: &[T]) -> T::Real {
    let mut scale = T::Real::zero();
    for value in values {
        let magnitude = value.magnitude();
        if magnitude.is_nan() {
            return magnitude;
        }
        if magnitude.is_infinite() {
            return magnitude;
        }
        if magnitude > scale {
            scale = magnitude;
        }
    }

    if scale == T::Real::zero() {
        return scale;
    }

    let mut scaled_sum = T::Real::zero();
    for value in values {
        let ratio = value.magnitude() / scale;
        scaled_sum = scaled_sum + ratio * ratio;
    }
    scale * scaled_sum.sqrt()
}
