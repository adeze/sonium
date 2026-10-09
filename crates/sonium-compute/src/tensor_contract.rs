//! Deterministic batched local tensor contractions.

use crate::DeviceScalar;
use thiserror::Error;

/// Validation failures for a batched local matrix contraction.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LocalContractError {
    #[error("{operand} length overflow for contraction shape")]
    ShapeOverflow { operand: &'static str },
    #[error("{operand} length {actual}, expected {expected}")]
    LengthMismatch {
        operand: &'static str,
        actual: usize,
        expected: usize,
    },
}

fn checked_product(
    operand: &'static str,
    dimensions: &[usize],
) -> Result<usize, LocalContractError> {
    dimensions.iter().try_fold(1usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or(LocalContractError::ShapeOverflow { operand })
    })
}

/// Compute `output[b,r,c] = Σ_k lhs[b,r,k] * rhs[b,k,c]`.
///
/// All tensors use contiguous row-major storage. The accumulation order is
/// fixed in `k`, and the output is fully overwritten, including zero-sized or
/// empty batches. This is the local contraction used by element and boundary
/// kernels before a backend-specific batched GEMM is selected.
pub fn batched_local_contract<T: DeviceScalar>(
    lhs: &[T],
    rhs: &[T],
    output: &mut [T],
    batches: usize,
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<(), LocalContractError> {
    let lhs_len = checked_product("lhs", &[batches, rows, inner])?;
    let rhs_len = checked_product("rhs", &[batches, inner, cols])?;
    let output_len = checked_product("output", &[batches, rows, cols])?;
    for (operand, actual, expected) in [
        ("lhs", lhs.len(), lhs_len),
        ("rhs", rhs.len(), rhs_len),
        ("output", output.len(), output_len),
    ] {
        if actual != expected {
            return Err(LocalContractError::LengthMismatch {
                operand,
                actual,
                expected,
            });
        }
    }

    for batch in 0..batches {
        let lhs_batch = batch * rows * inner;
        let rhs_batch = batch * inner * cols;
        let output_batch = batch * rows * cols;
        for row in 0..rows {
            for col in 0..cols {
                let mut sum = T::zero();
                for k in 0..inner {
                    sum = sum + lhs[lhs_batch + row * inner + k] * rhs[rhs_batch + k * cols + col];
                }
                output[output_batch + row * cols + col] = sum;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn contracts_multiple_local_matrices_in_fixed_order() {
        let lhs = [1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 1.0, 3.0];
        let rhs = [5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0];
        let mut output = [99.0; 8];
        batched_local_contract(&lhs, &rhs, &mut output, 2, 2, 2, 2)
            .expect("valid local contraction");
        assert_eq!(output, [19.0, 22.0, 43.0, 50.0, 2.0, 4.0, 10.0, 14.0]);
    }

    #[test]
    fn contracts_complex_local_values() {
        let lhs = [Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)];
        let rhs = [Complex64::new(3.0, 2.0), Complex64::new(-1.0, 4.0)];
        let mut output = [Complex64::default()];
        batched_local_contract(&lhs, &rhs, &mut output, 1, 1, 2, 1)
            .expect("valid complex contraction");
        assert_eq!(output[0], lhs[0] * rhs[0] + lhs[1] * rhs[1]);
    }

    #[test]
    fn rejects_contraction_shape_mismatch() {
        let mut output = [0.0; 1];
        let error = batched_local_contract(&[1.0], &[1.0], &mut output, 1, 1, 2, 1)
            .expect_err("lhs shape must be checked");
        assert!(matches!(
            error,
            LocalContractError::LengthMismatch { operand: "lhs", .. }
        ));
    }
}
