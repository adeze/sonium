use crate::traits::ComplexField;
use half::f16;
use num_complex::{Complex32, Complex64};

/// Trait for casting between precision levels.
///
/// Enables conversion of scalar values between high-precision (e.g. f64)
/// and low-precision (e.g. f32) representations.
pub trait PrecisionCast<TLow: ComplexField>: ComplexField {
    /// Cast a high-precision value down to low precision
    fn cast_down(val: Self) -> TLow;

    /// Cast a low-precision value back up to high precision
    fn cast_up(val: TLow) -> Self;

    /// Target inf-norm for vectors cast down to `TLow`, or `None` when
    /// no rescaling is needed.
    fn down_cast_max_abs() -> Option<Self::Real> {
        None
    }
}

impl PrecisionCast<f32> for f64 {
    #[inline]
    fn cast_down(val: f64) -> f32 {
        val as f32
    }

    #[inline]
    fn cast_up(val: f32) -> f64 {
        val as f64
    }
}

impl PrecisionCast<Complex32> for Complex64 {
    #[inline]
    fn cast_down(val: Complex64) -> Complex32 {
        Complex32::new(val.re as f32, val.im as f32)
    }

    #[inline]
    fn cast_up(val: Complex32) -> Complex64 {
        Complex64::new(val.re as f64, val.im as f64)
    }
}

/// Inf-norm target for f16 cast-down: 2^13 leaves three bits of headroom
/// up to f16::MAX (65504) for amplification inside the low-precision
/// apply, while f16 subnormals (~6e-8) still cover ~11 orders of
/// dynamic range below it.
pub const F16_SCALE_TARGET: f64 = 8192.0;

impl PrecisionCast<f16> for f64 {
    #[inline]
    fn cast_down(val: f64) -> f16 {
        f16::from_f64(val)
    }

    #[inline]
    fn cast_up(val: f16) -> f64 {
        f64::from(val)
    }

    #[inline]
    fn down_cast_max_abs() -> Option<f64> {
        Some(F16_SCALE_TARGET)
    }
}

impl PrecisionCast<f16> for f32 {
    #[inline]
    fn cast_down(val: f32) -> f16 {
        f16::from_f32(val)
    }

    #[inline]
    fn cast_up(val: f16) -> f32 {
        f32::from(val)
    }

    #[inline]
    fn down_cast_max_abs() -> Option<f32> {
        Some(F16_SCALE_TARGET as f32)
    }
}
