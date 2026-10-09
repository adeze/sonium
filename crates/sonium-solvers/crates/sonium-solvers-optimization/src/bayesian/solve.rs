use sonium_solvers_core::dense::DenseMatrix;

use crate::util::dense_linalg::{solve_lower_triangular, solve_lower_triangular_transpose};

pub(super) fn solve_lower(l: &DenseMatrix<f64>, b: &[f64]) -> Vec<f64> {
    solve_lower_triangular(l, b)
}

pub(super) fn solve_upper_from_lower(l: &DenseMatrix<f64>, b: &[f64]) -> Vec<f64> {
    solve_lower_triangular_transpose(l, b)
}

pub(super) fn solve_lower_matrix(l: &DenseMatrix<f64>, b: &DenseMatrix<f64>) -> DenseMatrix<f64> {
    let (m, n) = (b.nrows(), b.ncols());
    let mut out = DenseMatrix::zeros(m, n);
    for col in 0..n {
        let rhs: Vec<f64> = (0..m).map(|row| b[(row, col)]).collect();
        let solved = solve_lower_triangular(l, &rhs);
        for row in 0..m {
            out[(row, col)] = solved[row];
        }
    }
    out
}
