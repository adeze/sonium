//! [`FftScalar`] — the real-valued scalar over which FFTs are defined.
//!
//! Implemented for `f32` and `f64`. The buffer element type is always
//! `num_complex::Complex<T: FftScalar>`.

use num_traits::{Float, FromPrimitive};

/// Real-valued scalar accepted by [`crate::FftPlan`]. The associated
/// complex element is `num_complex::Complex<T>` (which `rustfft` calls
/// `FftNum` — same idea).
pub trait FftScalar:
    Float + FromPrimitive + Send + Sync + Default + Copy + std::fmt::Debug + 'static
{
}

impl FftScalar for f32 {}
impl FftScalar for f64 {}
