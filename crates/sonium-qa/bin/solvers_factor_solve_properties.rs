//! Promoted factor/solve property packet for mixed precision and permutations.

use num_complex::{Complex32, Complex64};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{
    DenseDirectConfig, DenseDirectMethod, DirectFactorization, MatrixClass, Ordering,
    SparseDirectConfig, SparseDirectMethod, SparseLuFactorization, factorize_dense,
    factorize_sparse,
};
use sonium_solvers::sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.factor-solve-properties.v1";
const CHECKSUM: &str = "deterministic-fixture:factor-solve-properties-v1";

fn max_error<T: Copy + Into<f64>>(left: &[T], right: &[T]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (Into::<f64>::into(*left) - Into::<f64>::into(*right)).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let original = CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![1.0_f32, 0.2, 0.2, 1.1, 0.3, 0.3, 0.9],
    );
    let updated = CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![1.001_f32, 0.2, 0.2, 1.099, 0.3, 0.3, 0.901],
    );
    let expected = [1.0_f32, -2.0, 0.5];
    let rhs = updated.matvec(&expected);
    let factor = SparseLuFactorization::factorize(&original)
        .map_err(|error| format!("f32 factorization: {error}"))?;
    let refinement = factor
        .solve_refined(&updated, &rhs, 1.0e-6, 6)
        .map_err(|error| format!("f32 refinement: {error}"))?;
    if !refinement.initial_relative_residual.is_finite()
        || !refinement.relative_residual.is_finite()
        || refinement.relative_residual > refinement.initial_relative_residual + 1.0e-6
        || refinement.relative_residual > 1.0e-5
    {
        return Err(format!(
            "f32 refinement residual did not improve: initial={} final={}",
            refinement.initial_relative_residual, refinement.relative_residual
        ));
    }

    let matrix = vec![
        vec![4.0_f64, 1.0, 0.0],
        vec![1.0, 3.0, 1.0],
        vec![0.0, 1.0, 2.0],
    ];
    let expected = [1.0_f64, -2.0, 0.75];
    let rhs = matrix
        .iter()
        .map(|row| row.iter().zip(expected).map(|(a, x)| a * x).sum())
        .collect::<Vec<_>>();
    let permutation = [2_usize, 0, 1];
    let permuted = permutation
        .iter()
        .map(|&row| {
            permutation
                .iter()
                .map(|&column| matrix[row][column])
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    let permuted_rhs = permutation.iter().map(|&row| rhs[row]).collect::<Vec<_>>();
    let triplets = permuted
        .iter()
        .enumerate()
        .flat_map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .filter_map(move |(column, &value)| (value != 0.0).then_some((row, column, value)))
        })
        .collect();
    let permuted_csr = CsrMatrix::from_triplets(3, 3, triplets);
    let permuted_factor = factorize_sparse(
        &permuted_csr,
        &SparseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            method: SparseDirectMethod::SimplicialCholesky,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .map_err(|error| format!("permuted factorization: {error}"))?;
    let permuted_solution = permuted_factor
        .solve(&permuted_rhs)
        .map_err(|error| format!("permuted solve: {error}"))?;
    let expected_permuted = permutation
        .iter()
        .map(|&row| expected[row])
        .collect::<Vec<_>>();
    let permutation_error = max_error(&permuted_solution, &expected_permuted);
    if permutation_error > 1.0e-11 {
        return Err(format!("permutation solution error={permutation_error}"));
    }

    let complex_matrix = vec![
        vec![
            Complex64::new(4.0, 0.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(1.0, 0.5),
        ],
        vec![
            Complex64::new(2.0, 2.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(-0.75, -0.75),
        ],
        vec![
            Complex64::new(1.0, -0.5),
            Complex64::new(-0.75, 0.75),
            Complex64::new(3.375, 0.0),
        ],
    ];
    let complex_expected = [
        Complex64::new(0.5, -1.0),
        Complex64::new(1.25, 0.25),
        Complex64::new(-0.75, 0.5),
    ];
    let complex_matvec = |x: &[Complex64]| {
        complex_matrix
            .iter()
            .map(|row| {
                row.iter()
                    .zip(x)
                    .map(|(value, entry)| *value * *entry)
                    .sum::<Complex64>()
            })
            .collect::<Vec<_>>()
    };
    let complex_error = |actual: &[Complex64], expected: &[Complex64]| {
        actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0, f64::max)
    };
    let complex_rhs = complex_matvec(&complex_expected);
    let complex_dense =
        DenseMatrix::from_row_major_vec(3, 3, complex_matrix.iter().flatten().copied().collect());
    let complex_dense_factor = factorize_dense(
        &complex_dense,
        &DenseDirectConfig {
            matrix_class: MatrixClass::HermitianPositiveDefinite,
            method: DenseDirectMethod::Cholesky,
            ..Default::default()
        },
    )
    .map_err(|error| format!("complex HPD dense factorization: {error}"))?;
    let complex_dense_solution = complex_dense_factor
        .solve(&complex_rhs)
        .map_err(|error| format!("complex HPD dense solve: {error}"))?;
    let complex_triplets = complex_matrix
        .iter()
        .enumerate()
        .flat_map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .filter_map(move |(column, &value)| {
                    (value.norm() > 0.0).then_some((row, column, value))
                })
        })
        .collect();
    let complex_csr = CsrMatrix::from_triplets(3, 3, complex_triplets);
    let complex_sparse_factor = factorize_sparse(
        &complex_csr,
        &SparseDirectConfig {
            matrix_class: MatrixClass::HermitianPositiveDefinite,
            method: SparseDirectMethod::SimplicialCholesky,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .map_err(|error| format!("complex HPD sparse factorization: {error}"))?;
    let complex_sparse_solution = complex_sparse_factor
        .solve(&complex_rhs)
        .map_err(|error| format!("complex HPD sparse solve: {error}"))?;
    let complex_dense_error = complex_error(&complex_dense_solution, &complex_expected);
    let complex_sparse_error = complex_error(&complex_sparse_solution, &complex_expected);
    let complex_dense_sparse_error =
        complex_error(&complex_dense_solution, &complex_sparse_solution);
    let complex_residual = complex_error(&complex_matvec(&complex_sparse_solution), &complex_rhs);
    let second_expected = [
        Complex64::new(-1.0, 0.25),
        Complex64::new(0.25, -0.5),
        Complex64::new(0.75, 1.0),
    ];
    let second_rhs = complex_matvec(&second_expected);
    let complex_block_rhs: Vec<_> = (0..3)
        .flat_map(|row| [complex_rhs[row], second_rhs[row]])
        .collect();
    let mut complex_block_solution = vec![Complex64::new(f64::NAN, f64::NAN); 6];
    complex_dense_factor
        .solve_many_into(&complex_block_rhs, &mut complex_block_solution, 2)
        .map_err(|error| format!("complex HPD dense block solve: {error}"))?;
    let complex_expected_block: Vec<_> = (0..3)
        .flat_map(|row| [complex_expected[row], second_expected[row]])
        .collect();
    let complex_block_error = complex_error(&complex_block_solution, &complex_expected_block);
    for (name, value) in [
        ("complex_hpd_dense_error", complex_dense_error),
        ("complex_hpd_sparse_error", complex_sparse_error),
        ("complex_hpd_dense_sparse_error", complex_dense_sparse_error),
        ("complex_hpd_residual", complex_residual),
        ("complex_hpd_block_error", complex_block_error),
    ] {
        if !value.is_finite() || value > 1.0e-12 {
            return Err(format!("{name}={value} exceeds complex HPD tolerance"));
        }
    }

    let complex_indefinite_matrix = vec![
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 1.0)],
        vec![Complex64::new(1.0, -1.0), Complex64::new(0.0, 0.0)],
    ];
    let complex_indefinite_expected = [Complex64::new(0.75, -0.25), Complex64::new(-1.0, 0.5)];
    let complex_indefinite_matvec = |x: &[Complex64]| {
        complex_indefinite_matrix
            .iter()
            .map(|row| {
                row.iter()
                    .zip(x)
                    .map(|(value, entry)| *value * *entry)
                    .sum::<Complex64>()
            })
            .collect::<Vec<_>>()
    };
    let complex_indefinite_rhs = complex_indefinite_matvec(&complex_indefinite_expected);
    let complex_indefinite_dense = DenseMatrix::from_row_major_vec(
        2,
        2,
        complex_indefinite_matrix
            .iter()
            .flatten()
            .copied()
            .collect(),
    );
    let complex_indefinite_dense_factor = factorize_dense(
        &complex_indefinite_dense,
        &DenseDirectConfig {
            matrix_class: MatrixClass::HermitianIndefinite,
            method: DenseDirectMethod::Ldlt,
            ..Default::default()
        },
    )
    .map_err(|error| format!("complex indefinite dense factorization: {error}"))?;
    let complex_indefinite_dense_solution = complex_indefinite_dense_factor
        .solve(&complex_indefinite_rhs)
        .map_err(|error| format!("complex indefinite dense solve: {error}"))?;
    let complex_indefinite_triplets = complex_indefinite_matrix
        .iter()
        .enumerate()
        .flat_map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .filter_map(move |(column, &value)| {
                    (value.norm() > 0.0).then_some((row, column, value))
                })
        })
        .collect();
    let complex_indefinite_csr = CsrMatrix::from_triplets(2, 2, complex_indefinite_triplets);
    let complex_indefinite_sparse_factor = factorize_sparse(
        &complex_indefinite_csr,
        &SparseDirectConfig {
            matrix_class: MatrixClass::HermitianIndefinite,
            method: SparseDirectMethod::MultifrontalLdlt,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .map_err(|error| format!("complex indefinite sparse factorization: {error}"))?;
    let complex_indefinite_sparse_solution = complex_indefinite_sparse_factor
        .solve(&complex_indefinite_rhs)
        .map_err(|error| format!("complex indefinite sparse solve: {error}"))?;
    let complex_indefinite_second_expected =
        [Complex64::new(-0.25, 0.5), Complex64::new(0.5, -1.0)];
    let complex_indefinite_second_rhs =
        complex_indefinite_matvec(&complex_indefinite_second_expected);
    let complex_indefinite_block_rhs: Vec<_> = (0..2)
        .flat_map(|row| {
            [
                complex_indefinite_rhs[row],
                complex_indefinite_second_rhs[row],
            ]
        })
        .collect();
    let mut complex_indefinite_block_solution = vec![Complex64::new(f64::NAN, f64::NAN); 4];
    complex_indefinite_sparse_factor
        .solve_many_into(
            &complex_indefinite_block_rhs,
            &mut complex_indefinite_block_solution,
            2,
        )
        .map_err(|error| format!("complex indefinite sparse block solve: {error}"))?;
    let complex_indefinite_expected_block: Vec<_> = (0..2)
        .flat_map(|row| {
            [
                complex_indefinite_expected[row],
                complex_indefinite_second_expected[row],
            ]
        })
        .collect();
    let complex_indefinite_block_error = complex_error(
        &complex_indefinite_block_solution,
        &complex_indefinite_expected_block,
    );
    let complex_indefinite_dense_error = complex_error(
        &complex_indefinite_dense_solution,
        &complex_indefinite_expected,
    );
    let complex_indefinite_sparse_error = complex_error(
        &complex_indefinite_sparse_solution,
        &complex_indefinite_expected,
    );
    let complex_indefinite_residual = complex_error(
        &complex_indefinite_matvec(&complex_indefinite_sparse_solution),
        &complex_indefinite_rhs,
    );
    for (name, value) in [
        (
            "complex_indefinite_dense_error",
            complex_indefinite_dense_error,
        ),
        (
            "complex_indefinite_sparse_error",
            complex_indefinite_sparse_error,
        ),
        ("complex_indefinite_residual", complex_indefinite_residual),
        (
            "complex_indefinite_sparse_block_error",
            complex_indefinite_block_error,
        ),
    ] {
        if !value.is_finite() || value > 1.0e-12 {
            return Err(format!(
                "{name}={value} exceeds complex indefinite tolerance"
            ));
        }
    }

    let complex32_matrix = vec![
        vec![Complex32::new(2.0, 0.5), Complex32::new(0.5, -0.25)],
        vec![Complex32::new(1.0, -0.75), Complex32::new(3.0, 0.25)],
    ];
    let complex32_expected = [Complex32::new(0.5, -0.25), Complex32::new(-1.0, 0.75)];
    let complex32_matvec = |x: &[Complex32]| {
        complex32_matrix
            .iter()
            .map(|row| {
                row.iter()
                    .zip(x)
                    .map(|(value, entry)| *value * *entry)
                    .sum::<Complex32>()
            })
            .collect::<Vec<_>>()
    };
    let complex32_error = |actual: &[Complex32], expected: &[Complex32]| {
        actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (*actual - *expected).norm() as f64)
            .fold(0.0, f64::max)
    };
    let complex32_rhs = complex32_matvec(&complex32_expected);
    let complex32_dense =
        DenseMatrix::from_row_major_vec(2, 2, complex32_matrix.iter().flatten().copied().collect());
    let complex32_dense_factor = factorize_dense(
        &complex32_dense,
        &DenseDirectConfig {
            matrix_class: MatrixClass::General,
            method: DenseDirectMethod::Lu,
            ..Default::default()
        },
    )
    .map_err(|error| format!("Complex32 dense factorization: {error}"))?;
    let complex32_dense_solution = complex32_dense_factor
        .solve(&complex32_rhs)
        .map_err(|error| format!("Complex32 dense solve: {error}"))?;
    let complex32_triplets = complex32_matrix
        .iter()
        .enumerate()
        .flat_map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .filter_map(move |(column, &value)| {
                    (value.norm() > 0.0).then_some((row, column, value))
                })
        })
        .collect();
    let complex32_csr = CsrMatrix::from_triplets(2, 2, complex32_triplets);
    let complex32_sparse_factor = factorize_sparse(
        &complex32_csr,
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .map_err(|error| format!("Complex32 sparse factorization: {error}"))?;
    let complex32_sparse_solution = complex32_sparse_factor
        .solve(&complex32_rhs)
        .map_err(|error| format!("Complex32 sparse solve: {error}"))?;
    let complex32_dense_error = complex32_error(&complex32_dense_solution, &complex32_expected);
    let complex32_sparse_error = complex32_error(&complex32_sparse_solution, &complex32_expected);
    let complex32_residual = complex32_error(
        &complex32_matvec(&complex32_sparse_solution),
        &complex32_rhs,
    );
    for (name, value) in [
        ("complex32_dense_error", complex32_dense_error),
        ("complex32_sparse_error", complex32_sparse_error),
        ("complex32_residual", complex32_residual),
    ] {
        if !value.is_finite() || value > 1.0e-5 {
            return Err(format!("{name}={value} exceeds Complex32 tolerance"));
        }
    }

    let mut seeded_complex_hpd_solution_error = 0.0_f64;
    let mut seeded_complex_hpd_residual = 0.0_f64;
    for seed in 0..8_u64 {
        let n = 5;
        let mut state = 0xC0DE_0000_u64.wrapping_add(seed);
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0
        };
        let mut lower = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for (row, entries) in lower.iter_mut().enumerate() {
            for entry in entries.iter_mut().take(row + 1) {
                *entry = Complex64::new(0.15 * next(), 0.15 * next());
            }
            entries[row] += Complex64::new(1.5 + 0.25 * next().abs(), 0.0);
        }
        let mut matrix = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for row in 0..n {
            for column in 0..n {
                matrix[row][column] = (0..=row.min(column))
                    .map(|k| lower[row][k] * lower[column][k].conj())
                    .sum();
                if row == column {
                    matrix[row][column] += Complex64::new(0.5, 0.0);
                }
            }
        }
        let expected: Vec<_> = (0..n)
            .map(|index| Complex64::new(0.5 + index as f64, -0.25 * index as f64))
            .collect();
        let rhs: Vec<_> = matrix
            .iter()
            .map(|row| row.iter().zip(&expected).map(|(a, x)| *a * *x).sum())
            .collect();
        let dense =
            DenseMatrix::from_row_major_vec(n, n, matrix.iter().flatten().copied().collect());
        let dense_factor = factorize_dense(
            &dense,
            &DenseDirectConfig {
                matrix_class: MatrixClass::HermitianPositiveDefinite,
                method: DenseDirectMethod::Cholesky,
                ..Default::default()
            },
        )
        .map_err(|error| format!("seeded complex HPD dense factorization: {error}"))?;
        let triplets = matrix
            .iter()
            .enumerate()
            .flat_map(|(row, values)| {
                values
                    .iter()
                    .enumerate()
                    .filter_map(move |(column, &value)| {
                        (value.norm() > 0.0).then_some((row, column, value))
                    })
            })
            .collect();
        let sparse_factor = factorize_sparse(
            &CsrMatrix::from_triplets(n, n, triplets),
            &SparseDirectConfig {
                matrix_class: MatrixClass::HermitianPositiveDefinite,
                method: SparseDirectMethod::SimplicialCholesky,
                ordering: Ordering::Auto,
                ..Default::default()
            },
            None,
        )
        .map_err(|error| format!("seeded complex HPD sparse factorization: {error}"))?;
        let dense_solution = dense_factor
            .solve(&rhs)
            .map_err(|error| format!("seeded complex HPD dense solve: {error}"))?;
        let sparse_solution = sparse_factor
            .solve(&rhs)
            .map_err(|error| format!("seeded complex HPD sparse solve: {error}"))?;
        seeded_complex_hpd_solution_error = seeded_complex_hpd_solution_error.max(
            dense_solution
                .iter()
                .zip(&sparse_solution)
                .map(|(dense, sparse)| (*dense - *sparse).norm())
                .fold(0.0, f64::max),
        );
        seeded_complex_hpd_residual = seeded_complex_hpd_residual.max(
            matrix
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(&sparse_solution)
                        .map(|(a, x)| *a * *x)
                        .sum::<Complex64>()
                })
                .zip(&rhs)
                .map(|(actual, expected)| (actual - expected).norm())
                .fold(0.0, f64::max),
        );
    }
    if !seeded_complex_hpd_solution_error.is_finite()
        || seeded_complex_hpd_solution_error > 1.0e-11
        || !seeded_complex_hpd_residual.is_finite()
        || seeded_complex_hpd_residual > 1.0e-11
    {
        return Err(format!(
            "seeded_complex_hpd_solution_error={seeded_complex_hpd_solution_error}, seeded_complex_hpd_residual={seeded_complex_hpd_residual}"
        ));
    }

    Ok(vec![
        (
            "f32_initial_relative_residual",
            f64::from(refinement.initial_relative_residual),
        ),
        (
            "f32_final_relative_residual",
            f64::from(refinement.relative_residual),
        ),
        ("permuted_solution_error", permutation_error),
        ("refinement_diagnostics_finite", 1.0),
        ("complex_hpd_cases", 1.0),
        ("seeded_complex_hpd_cases", 8.0),
        (
            "seeded_complex_hpd_solution_error",
            seeded_complex_hpd_solution_error,
        ),
        ("seeded_complex_hpd_residual", seeded_complex_hpd_residual),
        ("complex_hpd_dense_error", complex_dense_error),
        ("complex_hpd_sparse_error", complex_sparse_error),
        ("complex_hpd_dense_sparse_error", complex_dense_sparse_error),
        ("complex_hpd_residual", complex_residual),
        ("complex_hpd_block_error", complex_block_error),
        ("complex_indefinite_cases", 1.0),
        (
            "complex_indefinite_dense_error",
            complex_indefinite_dense_error,
        ),
        (
            "complex_indefinite_sparse_error",
            complex_indefinite_sparse_error,
        ),
        ("complex_indefinite_residual", complex_indefinite_residual),
        (
            "complex_indefinite_sparse_block_error",
            complex_indefinite_block_error,
        ),
        ("complex32_cases", 1.0),
        ("complex32_dense_error", complex32_dense_error),
        ("complex32_sparse_error", complex32_sparse_error),
        ("complex32_residual", complex32_residual),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let result = metrics.into_iter().fold(
                QaResult::passed(CASE_ID)
                    .reference("independent known-solution factor/solve property fixture")
                    .reference_checksum(CHECKSUM)
                    .input_checksum(CHECKSUM)
                    .formulation("f32 value-update refinement and f64 sparse permutation invariance")
                    .dimension("3x3 CSR systems with exact same-pattern updates and row/column permutation")
                    .tolerance("f32 final relative residual <= initial + 1e-6 and <= 1e-5; permutation error <= 1e-11")
                    .elapsed_ms(started.elapsed().as_millis() as u64),
                |result, (name, value)| result.metric(name, value),
            );
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result =
                QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
