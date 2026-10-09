//! Explicit-scale low-precision storage helpers.
//!
//! These primitives are for solver/assembly data that is stored at f16/bf16
//! precision but accumulated elsewhere, such as far-field or low-rank blocks.
//! They deliberately do not make f16/bf16 [`crate::DeviceScalar`] types:
//! backend BLAS still runs on f32/f64/complex scalars.

use half::{bf16, f16};
use num_complex::Complex32;

/// Metadata needed to reconstruct a scaled low-precision block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaledStorageMeta {
    pub len: usize,
    pub scale: f32,
    pub max_abs: f32,
}

/// Element type suitable for explicitly scaled f32 storage.
pub trait ScaledStorageElement: crate::BufferElement {
    const NAME: &'static str;

    fn max_finite_f32() -> f32;
    fn from_f32(value: f32) -> Self;
    fn to_f32(self) -> f32;
}

impl ScaledStorageElement for f16 {
    const NAME: &'static str = "f16";

    #[inline]
    fn max_finite_f32() -> f32 {
        f16::MAX.to_f32()
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        f16::from_f32(value)
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
}

impl ScaledStorageElement for bf16 {
    const NAME: &'static str = "bf16";

    #[inline]
    fn max_finite_f32() -> f32 {
        bf16::MAX.to_f32()
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        bf16::from_f32(value)
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
}

/// Quantize real f32 values into explicitly scaled low-precision storage.
pub fn quantize_scaled_f32<E: ScaledStorageElement>(src: &[f32]) -> (Vec<E>, ScaledStorageMeta) {
    let max_abs = src.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let scale = storage_scale::<E>(max_abs);
    let inv_scale = 1.0 / scale;
    let storage = src
        .iter()
        .map(|&value| E::from_f32(value * inv_scale))
        .collect();
    (
        storage,
        ScaledStorageMeta {
            len: src.len(),
            scale,
            max_abs,
        },
    )
}

/// Reconstruct real f32 values from explicitly scaled low-precision storage.
pub fn dequantize_scaled_f32<E: ScaledStorageElement>(
    storage: &[E],
    meta: ScaledStorageMeta,
) -> Vec<f32> {
    assert_eq!(
        storage.len(),
        meta.len,
        "scaled storage length does not match metadata"
    );
    storage
        .iter()
        .map(|&value| value.to_f32() * meta.scale)
        .collect()
}

/// Quantize complex values into real/imag interleaved low-precision storage.
pub fn quantize_scaled_complex32<E: ScaledStorageElement>(
    src: &[Complex32],
) -> (Vec<E>, ScaledStorageMeta) {
    let max_abs = src
        .iter()
        .map(|value| value.re.abs().max(value.im.abs()))
        .fold(0.0_f32, f32::max);
    let scale = storage_scale::<E>(max_abs);
    let inv_scale = 1.0 / scale;
    let mut storage = Vec::with_capacity(src.len() * 2);
    for &value in src {
        storage.push(E::from_f32(value.re * inv_scale));
        storage.push(E::from_f32(value.im * inv_scale));
    }
    (
        storage,
        ScaledStorageMeta {
            len: src.len() * 2,
            scale,
            max_abs,
        },
    )
}

/// Reconstruct complex values from real/imag interleaved storage.
pub fn dequantize_scaled_complex32<E: ScaledStorageElement>(
    storage: &[E],
    meta: ScaledStorageMeta,
) -> Vec<Complex32> {
    assert_eq!(
        storage.len(),
        meta.len,
        "scaled complex storage length does not match metadata"
    );
    assert_eq!(
        storage.len() % 2,
        0,
        "scaled complex storage must be real/imag interleaved"
    );
    let (chunks, remainder) = storage.as_chunks::<2>();
    debug_assert!(remainder.is_empty());
    chunks
        .iter()
        .map(|chunk| {
            Complex32::new(
                chunk[0].to_f32() * meta.scale,
                chunk[1].to_f32() * meta.scale,
            )
        })
        .collect()
}

fn storage_scale<E: ScaledStorageElement>(max_abs: f32) -> f32 {
    if max_abs == 0.0 || !max_abs.is_finite() {
        return 1.0;
    }
    max_abs / (E::max_finite_f32() * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_scaled_storage_round_trips_real_values() {
        let values = [-42.0_f32, -1.25, 0.0, 3.5, 19.75];
        let (storage, meta) = quantize_scaled_f32::<f16>(&values);

        assert_eq!(storage.len(), values.len());
        assert_eq!(meta.len, values.len());
        assert_eq!(meta.max_abs, 42.0);
        assert!(meta.scale > 0.0);

        let decoded = dequantize_scaled_f32(&storage, meta);
        for (got, want) in decoded.iter().zip(values.iter()) {
            assert!(
                (got - want).abs() <= 0.02,
                "got {got}, want {want}, meta {meta:?}"
            );
        }
    }

    #[test]
    fn bf16_scaled_storage_round_trips_complex_values() {
        let values = [
            Complex32::new(1.0, -2.0),
            Complex32::new(-32.0, 0.5),
            Complex32::new(0.125, 16.0),
        ];
        let (storage, meta) = quantize_scaled_complex32::<bf16>(&values);

        assert_eq!(storage.len(), values.len() * 2);
        assert_eq!(meta.len, values.len() * 2);
        assert_eq!(meta.max_abs, 32.0);

        let decoded = dequantize_scaled_complex32(&storage, meta);
        for (got, want) in decoded.iter().zip(values.iter()) {
            assert!(
                (got.re - want.re).abs() <= 0.25,
                "real got {got:?}, want {want:?}, meta {meta:?}"
            );
            assert!(
                (got.im - want.im).abs() <= 0.25,
                "imag got {got:?}, want {want:?}, meta {meta:?}"
            );
        }
    }

    #[test]
    fn zero_scaled_storage_uses_unit_scale() {
        let values = [0.0_f32; 4];
        let (storage, meta) = quantize_scaled_f32::<f16>(&values);

        assert_eq!(meta.scale, 1.0);
        assert_eq!(dequantize_scaled_f32(&storage, meta), values);
    }
}
