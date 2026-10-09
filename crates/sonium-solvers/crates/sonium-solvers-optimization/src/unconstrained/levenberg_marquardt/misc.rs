/// Project `x` onto the box `[lb, ub]`.
pub(super) fn project(x: &[f64], bounds: &[(f64, f64)]) -> Vec<f64> {
    x.iter()
        .zip(bounds.iter())
        .map(|(&xi, &(lo, hi))| xi.clamp(lo, hi))
        .collect::<Vec<_>>()
}

/// Weighted sum of squares: sum(w_i * r_i^2).
pub(super) fn weighted_sos(r: &[f64], w: &[f64]) -> f64 {
    r.iter().zip(w.iter()).map(|(&ri, &wi)| wi * ri * ri).sum()
}

/// Finite-difference step size scaled to the parameter magnitude.
///
/// Uses the standard relative rule `eps * (1 + |x|)` with a lower bound of
/// `eps`, so coordinates near zero still receive a resolvable central
/// difference step.
pub(super) fn jacobian_step(eps: f64, x: f64) -> f64 {
    (eps * (1.0 + x.abs())).max(eps)
}

/// Solve a dense symmetric positive-definite system Ax = b via Gaussian
/// elimination with partial pivoting. Returns `None` if the matrix is singular.
///
/// The singularity test is scale-aware: the pivot threshold is
/// `1e-12 * max(1.0, max|A_ij|)`, so a fixed absolute cutoff neither rejects
/// legitimate tiny-scale systems below unit magnitude nor mistakes pure
/// roundoff for a usable pivot in large-scale systems.
pub(super) fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    debug_assert_eq!(a.len(), n);
    debug_assert_eq!(a.first().map_or(0, |row| row.len()), n);

    // Scale-aware pivot threshold derived from the largest entry of `A`.
    let mut anorm: f64 = 0.0;
    for row in a.iter() {
        for v in row.iter() {
            anorm = anorm.max(v.abs());
        }
    }
    let pivot_tol = 1e-12 * anorm.max(1.0);

    // Augmented matrix [A | b]
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for (offset, row) in aug.iter().enumerate().skip(col + 1) {
            let val = row[col].abs();
            if val > max_val {
                max_val = val;
                max_row = offset;
            }
        }

        if max_val < pivot_tol {
            return None; // Singular
        }

        // Swap rows
        if max_row != col {
            aug.swap(col, max_row);
        }

        // Eliminate below
        let pivot = aug[col][col];
        let (top, bottom) = aug.split_at_mut(col + 1);
        let pivot_row = &top[col];
        for row in bottom.iter_mut() {
            let factor = row[col] / pivot;
            for (entry, &pivot_entry) in row.iter_mut().zip(pivot_row.iter()).skip(col) {
                *entry -= factor * pivot_entry;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let mut sum = aug[col][n];
        for j in (col + 1)..n {
            sum -= aug[col][j] * x[j];
        }
        x[col] = sum / aug[col][col];
    }

    // NaN check
    if x.iter().any(|v| !v.is_finite()) {
        return None;
    }

    Some(x)
}
