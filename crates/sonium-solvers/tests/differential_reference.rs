//! Differential solver check with a deliberately independent reference.

use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{
    DenseDirectConfig, DenseDirectMethod, DirectFactorization, MatrixClass, Ordering,
    SparseDirectConfig, SparseDirectMethod, factorize_dense, factorize_sparse,
};
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::{GmresConfig, LinearOperator, SolverResultContract, try_gmres};

#[derive(Clone)]
struct MatrixOperator(Vec<Vec<f64>>);

impl LinearOperator<f64> for MatrixOperator {
    fn num_rows(&self) -> usize {
        self.0.len()
    }

    fn num_cols(&self) -> usize {
        self.0.first().map_or(0, Vec::len)
    }

    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        for (row, output) in self.0.iter().zip(y.iter_mut()) {
            *output = row.iter().zip(x).map(|(a, x)| a * x).sum();
        }
    }

    fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
        y.fill(0.0);
        for (row, x) in self.0.iter().zip(x) {
            for (column, value) in row.iter().enumerate() {
                y[column] += value * x;
            }
        }
    }
}

fn matvec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(x).map(|(a, x)| a * x).sum())
        .collect()
}

fn reference_solve(matrix: &[Vec<f64>], rhs: &[f64]) -> Vec<f64> {
    let n = matrix.len();
    let mut a = matrix.to_vec();
    let mut b = rhs.to_vec();
    for pivot in 0..n {
        let pivot_row = (pivot..n)
            .max_by(|&left, &right| a[left][pivot].abs().total_cmp(&a[right][pivot].abs()))
            .expect("pivot row");
        assert!(a[pivot_row][pivot].abs() > 1.0e-14);
        if pivot_row != pivot {
            a.swap(pivot, pivot_row);
            b.swap(pivot, pivot_row);
        }
        for row in (pivot + 1)..n {
            let scale = a[row][pivot] / a[pivot][pivot];
            a[row][pivot] = 0.0;
            for column in (pivot + 1)..n {
                a[row][column] -= scale * a[pivot][column];
            }
            b[row] -= scale * b[pivot];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        x[row] = (b[row]
            - ((row + 1)..n)
                .map(|column| a[row][column] * x[column])
                .sum::<f64>())
            / a[row][row];
    }
    x
}

fn max_error(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

#[test]
fn dense_sparse_and_gmres_match_independent_reference() {
    let matrix = vec![
        vec![7.0, 1.0, 0.5, 0.0],
        vec![2.0, 8.0, 1.0, 0.2],
        vec![0.0, 1.0, 6.0, 1.0],
        vec![0.3, 0.0, 2.0, 9.0],
    ];
    let expected = [1.0, -2.0, 0.5, 3.0];
    let rhs = matvec(&matrix, &expected);
    let reference = reference_solve(&matrix, &rhs);
    let dense = DenseMatrix::from_row_major_vec(4, 4, matrix.iter().flatten().copied().collect());
    let dense_factor = factorize_dense(
        &dense,
        &DenseDirectConfig {
            matrix_class: MatrixClass::General,
            method: DenseDirectMethod::Lu,
            ..Default::default()
        },
    )
    .expect("dense factorization");
    let dense_solution = dense_factor.solve(&rhs).expect("dense solve");

    let triplets = matrix
        .iter()
        .enumerate()
        .flat_map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .filter_map(move |(column, &value)| (value != 0.0).then_some((row, column, value)))
        })
        .collect();
    let sparse = CsrMatrix::from_triplets(4, 4, triplets);
    let sparse_factor = factorize_sparse(
        &sparse,
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .expect("sparse factorization");
    let sparse_solution = sparse_factor.solve(&rhs).expect("sparse solve");

    let gmres = try_gmres(
        &MatrixOperator(matrix.clone()),
        &rhs,
        &GmresConfig {
            restart: 4,
            max_iterations: 40,
            tolerance: 1.0e-12,
            ..Default::default()
        },
    )
    .expect("GMRES solve");
    gmres.validate_contract().expect("GMRES contract");

    assert!(max_error(&dense_solution, &reference) <= 1.0e-11);
    assert!(max_error(&sparse_solution, &reference) <= 1.0e-11);
    assert!(max_error(&gmres.x, &reference) <= 1.0e-9);
    assert!(max_error(&matvec(&matrix, &gmres.x), &rhs) <= 1.0e-10);
}
