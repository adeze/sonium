//! Adversarial coverage for direct factorizations: matrices that require row
//! exchanges (pivoting), carry extreme row scaling, or are exactly rank
//! deficient. Dense LU must solve the first two accurately; both dense and
//! sparse paths must report rank deficiency as a clean error, never a panic,
//! hang, or silently wrong result.

use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_direct::{LuError, SparseLuError, SparseLuFactorization, lu_factorize};
use sonium_solvers_sparse::CsrMatrix;

fn dense_residual(matrix: &DenseMatrix<f64>, x: &[f64], rhs: &[f64]) -> f64 {
    let n = matrix.nrows();
    let error = (0..n)
        .map(|row| {
            let ax: f64 = (0..n).map(|column| matrix[[row, column]] * x[column]).sum();
            (ax - rhs[row]).abs()
        })
        .fold(0.0_f64, f64::max);
    let scale = rhs.iter().map(|value| value.abs()).fold(0.0, f64::max);
    error / scale.max(f64::MIN_POSITIVE)
}

#[test]
fn anti_identity_permutation_requires_full_row_exchange() {
    // Every diagonal entry is zero: factorization without pivoting cannot
    // even start; the solution must still be exact.
    let n = 8;
    let matrix = DenseMatrix::from_fn(
        n,
        n,
        |row, column| {
            if column + row + 1 == n { 2.5 } else { 0.0 }
        },
    );
    let expected: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.5).collect();
    let mut rhs = vec![0.0; n];
    for row in 0..n {
        rhs[row] = matrix[[row, n - 1 - row]] * expected[n - 1 - row];
    }
    let factor = lu_factorize(&matrix).expect("permutation matrix must factorize");
    let solution = factor.solve(&rhs).expect("solve must succeed");
    assert!(dense_residual(&matrix, &solution, &rhs) < 1.0e-14);
    for (got, want) in solution.iter().zip(&expected) {
        assert!((got - want).abs() < 1.0e-12);
    }
}

#[test]
fn zero_leading_pivot_with_nonsingular_matrix_solves() {
    // Zero (0,0) entry in an otherwise well-conditioned nonsingular matrix.
    let matrix = DenseMatrix::from_row_major_vec(
        4,
        4,
        vec![
            0.0, 2.0, 3.0, 1.0, //
            1.0, 4.0, 0.0, 2.0, //
            2.0, 1.0, 5.0, 0.0, //
            1.0, 0.0, 2.0, 6.0, //
        ],
    );
    let rhs = vec![1.0, 2.0, 3.0, 4.0];
    let factor = lu_factorize(&matrix).expect("nonsingular matrix must factorize");
    let solution = factor.solve(&rhs).expect("solve must succeed");
    assert!(dense_residual(&matrix, &solution, &rhs) < 1.0e-12);
}

#[test]
fn extreme_row_scaling_stays_accurate() {
    // Row scales spanning 16 orders of magnitude: partial pivoting must not
    // silently lose the small rows.
    let scales = [1.0e-8, 1.0e-4, 1.0, 1.0e4, 1.0e8];
    let n = scales.len();
    let base = DenseMatrix::from_fn(n, n, |row, column| {
        let value = (((row * 37 + column * 17 + 11) % 101) as f64 - 50.0) / 101.0;
        if row == column {
            value + n as f64
        } else {
            value
        }
    });
    let matrix = DenseMatrix::from_fn(n, n, |row, column| scales[row] * base[[row, column]]);
    let expected: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.25).collect();
    let mut rhs = vec![0.0; n];
    for row in 0..n {
        rhs[row] = (0..n)
            .map(|column| matrix[[row, column]] * expected[column])
            .sum();
    }
    let factor = lu_factorize(&matrix).expect("scaled nonsingular matrix must factorize");
    let solution = factor.solve(&rhs).expect("solve must succeed");
    assert!(
        dense_residual(&matrix, &solution, &rhs) < 1.0e-8,
        "row scaling must not destroy accuracy"
    );
}

#[test]
fn dense_rank_deficiency_reports_singular_matrix() {
    // Duplicate rows: exactly singular; must be a clean error.
    let matrix =
        DenseMatrix::from_row_major_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    assert!(matches!(
        lu_factorize(&matrix),
        Err(LuError::SingularMatrix)
    ));

    // Zero matrix: singular at the very first pivot.
    let zero = DenseMatrix::from_row_major_vec(4, 4, vec![0.0; 16]);
    assert!(matches!(lu_factorize(&zero), Err(LuError::SingularMatrix)));
}

fn csr_from_dense(rows: &[Vec<f64>]) -> CsrMatrix<f64> {
    let n = rows.len();
    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_ptrs.push(0);
    for row in rows {
        for (column, &value) in row.iter().enumerate() {
            if value != 0.0 {
                col_indices.push(column);
                values.push(value);
            }
        }
        row_ptrs.push(col_indices.len());
    }
    CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values)
}

#[test]
fn sparse_rank_deficiency_reports_zero_pivot() {
    // Duplicate rows in CSR form: every sparse path must report the zero
    // pivot instead of dividing by zero or hanging.
    let matrix = csr_from_dense(&[
        vec![2.0, 1.0, 0.0],
        vec![1.0, 2.0, 1.0],
        vec![2.0, 1.0, 0.0],
    ]);
    assert!(matches!(
        SparseLuFactorization::factorize(&matrix),
        Err(SparseLuError::SingularMatrix(_))
    ));
    assert!(matches!(
        SparseLuFactorization::factorize_supernodal(&matrix),
        Err(SparseLuError::SingularMatrix(_))
    ));
}

#[test]
fn sparse_zero_diagonal_nonsingular_pattern_solves() {
    // Zero diagonal with a nonsingular pattern: symbolic analysis must keep
    // the entries and numeric factorization must pivot or reorder.
    let matrix = csr_from_dense(&[
        vec![0.0, 1.0, 0.0, 2.0],
        vec![1.0, 0.0, 3.0, 0.0],
        vec![0.0, 4.0, 0.0, 5.0],
        vec![6.0, 0.0, 7.0, 0.0],
    ]);
    let factor = SparseLuFactorization::factorize(&matrix)
        .expect("nonsingular pattern with zero diagonal must factorize");
    let rhs = vec![1.0, 2.0, 3.0, 4.0];
    let solution = factor.solve(&rhs).expect("solve must succeed");
    let residual = matrix
        .matvec(&solution)
        .iter()
        .zip(&rhs)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    assert!(residual < 1.0e-10, "residual {residual:e} too large");
}
