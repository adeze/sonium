//! Invariant-subspace comparison for clustered and repeated modes.
//!
//! Eigenvector-by-eigenvector comparison is misleading for degenerate or
//! tightly clustered spectra: any orthonormal basis of the invariant subspace
//! is a valid answer, and signs are arbitrary. These helpers compare
//! subspaces through their orthogonal projectors instead of pairing
//! individual vectors.

// Rust guideline compliant 2026-02-21

use sonium_solvers_core::traits::SolverError;

/// Orthonormalize `columns` with twice-repeated modified Gram-Schmidt.
///
/// Directions whose residual norm falls at or below `drop_tolerance` are
/// dropped as numerically dependent.
///
/// # Errors
///
/// Returns [`SolverError`] when the input is empty or ragged, contains
/// non-finite entries, uses an invalid tolerance, or is entirely
/// rank-deficient below the tolerance.
pub fn orthonormal_basis(
    columns: &[Vec<f64>],
    drop_tolerance: f64,
) -> Result<Vec<Vec<f64>>, SolverError> {
    if columns.is_empty() {
        return Err(SolverError::OperationFailed {
            operation: "subspace orthonormalization",
            reason: "at least one column is required".to_string(),
        });
    }
    if !drop_tolerance.is_finite() || drop_tolerance < 0.0 {
        return Err(SolverError::OperationFailed {
            operation: "subspace orthonormalization",
            reason: "drop tolerance must be finite and non-negative".to_string(),
        });
    }
    let dimension = columns[0].len();
    if dimension == 0 {
        return Err(SolverError::OperationFailed {
            operation: "subspace orthonormalization",
            reason: "columns must be non-empty".to_string(),
        });
    }
    for (index, column) in columns.iter().enumerate() {
        if column.len() != dimension {
            return Err(SolverError::DimensionMismatch {
                expected: dimension,
                got: column.len(),
            });
        }
        if !column.iter().all(|value| value.is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "subspace orthonormalization",
                reason: format!("column {index} contains non-finite entries"),
            });
        }
    }

    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(columns.len());
    for column in columns {
        let mut residual = column.clone();
        for _ in 0..2 {
            for kept in &basis {
                let projection: f64 = kept.iter().zip(residual.iter()).map(|(&q, &r)| q * r).sum();
                for (r, &q) in residual.iter_mut().zip(kept.iter()) {
                    *r -= projection * q;
                }
            }
        }
        let norm_squared: f64 = residual.iter().map(|value| value * value).sum();
        let norm = norm_squared.sqrt();
        if norm <= drop_tolerance {
            continue;
        }
        for value in residual.iter_mut() {
            *value /= norm;
        }
        basis.push(residual);
    }
    if basis.is_empty() {
        return Err(SolverError::OperationFailed {
            operation: "subspace orthonormalization",
            reason: "all columns are numerically dependent below the drop tolerance".to_string(),
        });
    }
    Ok(basis)
}

/// Root-mean-square sine of the principal angles between two subspaces.
///
/// Both column sets are orthonormalized internally, so raw eigenvector
/// output (arbitrary scale, sign, and in-subspace rotation) is accepted
/// directly. The distance is 0 for identical subspaces and 1 when every
/// principal angle is 90 degrees.
///
/// # Errors
///
/// Returns [`SolverError`] when the two sets span different ambient
/// dimensions, hold different numbers of vectors, or are numerically
/// rank-deficient.
pub fn subspace_distance(
    a_columns: &[Vec<f64>],
    b_columns: &[Vec<f64>],
) -> Result<f64, SolverError> {
    if a_columns.len() != b_columns.len() {
        return Err(SolverError::DimensionMismatch {
            expected: a_columns.len(),
            got: b_columns.len(),
        });
    }
    let basis_a = orthonormal_basis(a_columns, 1.0e-12)?;
    let basis_b = orthonormal_basis(b_columns, 1.0e-12)?;
    if basis_a[0].len() != basis_b[0].len() {
        return Err(SolverError::DimensionMismatch {
            expected: basis_a[0].len(),
            got: basis_b[0].len(),
        });
    }
    if basis_a.len() != a_columns.len() || basis_b.len() != b_columns.len() {
        return Err(SolverError::OperationFailed {
            operation: "subspace distance",
            reason: "inputs are numerically rank-deficient".to_string(),
        });
    }
    let count = basis_a.len();
    // For orthonormal k-frames, ||A A^T - B B^T||_F^2 = 2(k - ||A^T B||_F^2),
    // so sqrt((k - ||A^T B||_F^2)/k) is the RMS sine of principal angles.
    let mut gram_frobenius_squared = 0.0;
    for column_a in &basis_a {
        for column_b in &basis_b {
            let inner: f64 = column_a
                .iter()
                .zip(column_b.iter())
                .map(|(&x, &y)| x * y)
                .sum();
            gram_frobenius_squared += inner * inner;
        }
    }
    let gap = (count as f64 - gram_frobenius_squared).max(0.0);
    Ok((gap / count as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotated_basis_of_same_subspace_has_zero_distance() {
        let angle = 0.7f64;
        let (sin, cos) = angle.sin_cos();
        let canonical = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let rotated = [vec![cos, sin, 0.0], vec![-sin, cos, 0.0]];
        let distance = subspace_distance(&canonical, &rotated).expect("valid");
        assert!(distance < 1.0e-14, "distance = {distance}");
        // Scaling and in-subspace mixing must not matter either.
        let scaled = [vec![3.0, 0.0, 0.0], vec![1.0, -2.0, 0.0]];
        let distance = subspace_distance(&canonical, &scaled).expect("valid");
        assert!(distance < 1.0e-14, "distance = {distance}");
    }

    #[test]
    fn distance_matches_sine_of_principal_angle() {
        for degrees in [0.0f64, 15.0, 30.0, 45.0, 60.0, 90.0] {
            let theta = degrees.to_radians();
            let distance = subspace_distance(&[vec![1.0, 0.0]], &[vec![theta.cos(), theta.sin()]])
                .expect("valid");
            assert!(
                (distance - theta.sin()).abs() < 1.0e-12,
                "degrees = {degrees}, distance = {distance}"
            );
        }
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert!(subspace_distance(&[], &[]).is_err());
        assert!(subspace_distance(&[vec![1.0]], &[vec![1.0], vec![0.0]]).is_err());
        assert!(subspace_distance(&[vec![1.0, 0.0]], &[vec![f64::NAN, 0.0]]).is_err());
        assert!(subspace_distance(&[vec![1.0, 0.0]], &[vec![1.0]]).is_err());
        assert!(orthonormal_basis(&[vec![1.0, 0.0], vec![2.0, 0.0]], 1.0e-12).is_ok());
        assert!(subspace_distance(&[vec![1.0, 0.0]], &[vec![1.0, 0.0], vec![0.0, 1.0]]).is_err());
    }
}
