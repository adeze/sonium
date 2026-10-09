//! Small host-side elementwise primitives shared by FFT and compute callers.

use crate::DeviceScalar;
use std::fmt;
use std::ops::Mul;

/// Shape errors for checked elementwise operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementwiseError {
    /// The input and output slices do not have the same length.
    LengthMismatch {
        left: usize,
        right: usize,
        output: usize,
    },
}

impl fmt::Display for ElementwiseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                left,
                right,
                output,
            } => write!(
                formatter,
                "elementwise lengths differ: left={left}, right={right}, output={output}"
            ),
        }
    }
}

impl std::error::Error for ElementwiseError {}

/// Compute `output[i] = left[i] * right[i]`.
pub fn pointwise_mul<T: Copy + Mul<Output = T>>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) -> Result<(), ElementwiseError> {
    if left.len() != right.len() || left.len() != output.len() {
        return Err(ElementwiseError::LengthMismatch {
            left: left.len(),
            right: right.len(),
            output: output.len(),
        });
    }
    for ((destination, &lhs), &rhs) in output.iter_mut().zip(left).zip(right) {
        *destination = lhs * rhs;
    }
    Ok(())
}

/// Scale a real or complex vector by a real scalar in place.
pub fn scale_real<T: DeviceScalar>(values: &mut [T], alpha: T::Real) {
    let factor = T::from_real(alpha);
    for value in values {
        *value = *value * factor;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn pointwise_mul_handles_real_and_complex_values() {
        let mut real = [0.0; 3];
        pointwise_mul(&[2.0, -3.0, 4.0], &[0.5, 2.0, -1.5], &mut real).unwrap();
        assert_eq!(real, [1.0, -6.0, -6.0]);

        let mut complex = [Complex64::default(); 2];
        pointwise_mul(
            &[Complex64::new(1.0, 2.0), Complex64::new(-1.0, 0.5)],
            &[Complex64::new(2.0, -1.0), Complex64::new(0.5, 3.0)],
            &mut complex,
        )
        .unwrap();
        assert_eq!(
            complex,
            [Complex64::new(4.0, 3.0), Complex64::new(-2.0, -2.75)]
        );
    }

    #[test]
    fn pointwise_mul_rejects_shape_mismatch_and_scale_real_is_exact() {
        let mut output = [0.0; 2];
        assert_eq!(
            pointwise_mul(&[1.0], &[2.0, 3.0], &mut output),
            Err(ElementwiseError::LengthMismatch {
                left: 1,
                right: 2,
                output: 2,
            })
        );
        scale_real(&mut output, -2.0);
        assert_eq!(output, [0.0, 0.0]);
    }
}
