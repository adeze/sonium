//! Independent execution and workload metrics for the deterministic corpus.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{
    DenseDirectConfig, DenseDirectMethod, DirectFactorization, MatrixClass, Ordering,
    SparseDirectConfig, SparseDirectMethod, factorize_dense, factorize_sparse,
};
use sonium_solvers::preconditioners::IluPreconditioner;
use sonium_solvers::{
    CsrMatrix, GmresConfig, SOLVER_VALIDATION_CORPUS, gmres, gmres_preconditioned,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.validation-corpus-reference.v2";

fn real_matvec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn complex_matvec(matrix: &[Vec<Complex64>], x: &[Complex64]) -> Vec<Complex64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(x).map(|(a, b)| *a * *b).sum())
        .collect()
}

fn complex_max_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max)
}

fn complex_relative_residual(matrix: &[Vec<Complex64>], x: &[Complex64], rhs: &[Complex64]) -> f64 {
    let residual = complex_matvec(matrix, x)
        .into_iter()
        .zip(rhs)
        .map(|(actual, expected)| (actual - expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
    residual / rhs_norm.max(1.0)
}

fn csr(matrix: &[Vec<f64>]) -> CsrMatrix<f64> {
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
    CsrMatrix::from_triplets(matrix.len(), matrix[0].len(), triplets)
}

fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn relative_residual(matrix: &[Vec<f64>], x: &[f64], rhs: &[f64]) -> f64 {
    let residual = real_matvec(matrix, x)
        .into_iter()
        .zip(rhs)
        .map(|(actual, expected)| actual - expected)
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value * value).sum::<f64>().sqrt();
    residual / rhs_norm.max(1.0)
}

fn run() -> Result<Vec<(&'static str, f64, f64)>, String> {
    if SOLVER_VALIDATION_CORPUS.len() < 12 {
        return Err("validation corpus is incomplete".into());
    }

    let general = vec![
        vec![4.0, 1.0, 0.5],
        vec![2.0, 5.0, 1.0],
        vec![0.25, 1.5, 3.0],
    ];
    let x0 = [1.0, -0.5, 2.0];
    let x1 = [-2.0, 0.25, 1.5];
    let rhs0 = real_matvec(&general, &x0);
    let rhs1 = real_matvec(&general, &x1);
    let dense = DenseMatrix::from_row_major_vec(3, 3, general.iter().flatten().copied().collect());
    let dense_factor = factorize_dense(
        &dense,
        &DenseDirectConfig {
            matrix_class: MatrixClass::General,
            method: DenseDirectMethod::Lu,
            ..Default::default()
        },
    )
    .map_err(|error| format!("dense corpus factorization: {error}"))?;
    let dense_solution = dense_factor
        .solve(&rhs0)
        .map_err(|error| format!("dense corpus solve: {error}"))?;
    let block_rhs = [rhs0[0], rhs1[0], rhs0[1], rhs1[1], rhs0[2], rhs1[2]];
    let mut block_solution = vec![0.0; 6];
    dense_factor
        .solve_many_into(&block_rhs, &mut block_solution, 2)
        .map_err(|error| format!("dense corpus block solve: {error}"))?;
    let dense_block_expected = [x0[0], x1[0], x0[1], x1[1], x0[2], x1[2]];

    let general_csr = csr(&general);
    let ilu = IluPreconditioner::try_from_csr(&general_csr)
        .map_err(|error| format!("ILU corpus construction: {error}"))?;
    let ilu_solution = gmres_preconditioned(
        &general_csr,
        &ilu,
        &rhs0,
        &GmresConfig {
            max_iterations: 40,
            tolerance: 1.0e-12,
            print_interval: 0,
            ..Default::default()
        },
    );
    if !ilu_solution.converged {
        return Err(format!(
            "ILU-preconditioned GMRES did not converge: residual={}",
            ilu_solution.residual
        ));
    }
    let sparse_factor = factorize_sparse(
        &general_csr,
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .map_err(|error| format!("sparse corpus factorization: {error}"))?;
    let sparse_solution = sparse_factor
        .solve(&rhs0)
        .map_err(|error| format!("sparse corpus solve: {error}"))?;
    let mut sparse_block_solution = vec![0.0; 6];
    sparse_factor
        .solve_many_into(&block_rhs, &mut sparse_block_solution, 2)
        .map_err(|error| format!("sparse corpus block solve: {error}"))?;

    let helmholtz = vec![
        vec![2.0, -1.0, 0.0],
        vec![-1.0, 2.0, -1.0],
        vec![0.0, -1.0, 2.0],
    ];
    let helmholtz_csr = csr(&helmholtz);
    let helmholtz_rhs = [1.0, 0.0, 1.0];
    let gmres_solution = gmres(
        &helmholtz_csr,
        &helmholtz_rhs,
        &GmresConfig {
            max_iterations: 40,
            tolerance: 1.0e-12,
            print_interval: 0,
            ..Default::default()
        },
    );
    if !gmres_solution.converged {
        return Err(format!(
            "Helmholtz GMRES did not converge: residual={}",
            gmres_solution.residual
        ));
    }

    let hpd = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(4.0, 0.0),
        ],
    );
    let hpd_factor = factorize_dense(
        &hpd,
        &DenseDirectConfig {
            matrix_class: MatrixClass::HermitianPositiveDefinite,
            method: DenseDirectMethod::Cholesky,
            ..Default::default()
        },
    )
    .map_err(|error| format!("HPD corpus factorization: {error}"))?;
    let hpd_expected = [Complex64::new(0.5, -1.0), Complex64::new(2.0, 0.25)];
    let hpd_rhs = [
        hpd[(0, 0)] * hpd_expected[0] + hpd[(0, 1)] * hpd_expected[1],
        hpd[(1, 0)] * hpd_expected[0] + hpd[(1, 1)] * hpd_expected[1],
    ];
    let hpd_solution = hpd_factor
        .solve(&hpd_rhs)
        .map_err(|error| format!("HPD corpus solve: {error}"))?;
    let hpd_error = hpd_solution
        .iter()
        .zip(hpd_expected)
        .map(|(actual, expected)| (*actual - expected).norm())
        .fold(0.0, f64::max);

    let complex_helmholtz = vec![
        vec![
            Complex64::new(2.0, 0.2),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        vec![
            Complex64::new(-1.0, 0.0),
            Complex64::new(2.0, 0.2),
            Complex64::new(-1.0, 0.0),
        ],
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(2.0, 0.2),
        ],
    ];
    let complex_helmholtz_expected = [
        Complex64::new(1.0, 0.5),
        Complex64::new(-0.25, 1.0),
        Complex64::new(2.0, -0.75),
    ];
    let complex_helmholtz_rhs = complex_matvec(&complex_helmholtz, &complex_helmholtz_expected);
    let complex_helmholtz_csr = CsrMatrix::from_triplets(
        3,
        3,
        complex_helmholtz
            .iter()
            .enumerate()
            .flat_map(|(row, values)| {
                values
                    .iter()
                    .enumerate()
                    .filter_map(move |(column, &value)| {
                        (value != Complex64::new(0.0, 0.0)).then_some((row, column, value))
                    })
            })
            .collect(),
    );
    let complex_helmholtz_ilu = IluPreconditioner::try_from_csr(&complex_helmholtz_csr)
        .map_err(|error| format!("complex Helmholtz ILU construction: {error}"))?;
    let complex_helmholtz_solution = gmres_preconditioned(
        &complex_helmholtz_csr,
        &complex_helmholtz_ilu,
        &complex_helmholtz_rhs,
        &GmresConfig {
            max_iterations: 30,
            tolerance: 1.0e-12,
            print_interval: 0,
            ..Default::default()
        },
    );
    if !complex_helmholtz_solution.converged {
        return Err(format!(
            "complex Helmholtz ILU-preconditioned GMRES did not converge: residual={}",
            complex_helmholtz_solution.residual
        ));
    }
    let complex_helmholtz_solution_error =
        complex_max_error(&complex_helmholtz_solution.x, &complex_helmholtz_expected);
    let complex_helmholtz_residual = complex_relative_residual(
        &complex_helmholtz,
        &complex_helmholtz_solution.x,
        &complex_helmholtz_rhs,
    );

    let ilu_error = max_error(&ilu_solution.x, &x0);
    let ilu_residual = relative_residual(&general, &ilu_solution.x, &rhs0);
    let rank_deficient = csr(&[vec![1.0, 1.0], vec![1.0, 1.0]]);
    let rank_deficient_rejected = factorize_sparse(
        &rank_deficient,
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .and_then(|factor| factor.solve(&[2.0, 2.0]))
    .is_err();
    if !rank_deficient_rejected {
        return Err("rank-deficient direct solve was accepted".into());
    }
    let metrics = vec![
        ("fixture_count", SOLVER_VALIDATION_CORPUS.len() as f64, 12.0),
        ("rank_deficient_rejected", 1.0, 1.0),
        (
            "dense_solution_error",
            max_error(&dense_solution, &x0),
            1.0e-12,
        ),
        (
            "sparse_solution_error",
            max_error(&sparse_solution, &x0),
            1.0e-12,
        ),
        ("ilu_gmres_solution_error", ilu_error, 1.0e-10),
        ("ilu_gmres_independent_residual", ilu_residual, 1.0e-10),
        (
            "dense_block_rhs_error",
            max_error(&block_solution, &dense_block_expected),
            1.0e-12,
        ),
        (
            "sparse_block_rhs_error",
            max_error(&sparse_block_solution, &dense_block_expected),
            1.0e-12,
        ),
        (
            "dense_independent_residual",
            relative_residual(&general, &dense_solution, &rhs0),
            1.0e-12,
        ),
        (
            "sparse_independent_residual",
            relative_residual(&general, &sparse_solution, &rhs0),
            1.0e-12,
        ),
        ("helmholtz_gmres_residual", gmres_solution.residual, 1.0e-12),
        ("hpd_solution_error", hpd_error, 1.0e-12),
        (
            "complex_helmholtz_ilu_gmres_solution_error",
            complex_helmholtz_solution_error,
            1.0e-10,
        ),
        (
            "complex_helmholtz_ilu_gmres_independent_residual",
            complex_helmholtz_residual,
            1.0e-10,
        ),
        (
            "dense_factor_nonzeros",
            dense_factor.factor_nonzeros() as f64,
            1.0,
        ),
        (
            "sparse_factor_nonzeros",
            sparse_factor.factor_nonzeros() as f64,
            1.0,
        ),
        (
            "dense_factor_memory_bytes",
            dense_factor.factor_memory_bytes() as f64,
            1.0,
        ),
        (
            "sparse_factor_memory_bytes",
            sparse_factor.factor_memory_bytes() as f64,
            1.0,
        ),
    ];
    if metrics
        .iter()
        .take(8)
        .any(|(_, value, _)| !value.is_finite())
    {
        return Err("corpus produced a non-finite correctness metric".into());
    }
    Ok(metrics)
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent analytic solutions and recomputed residuals")
                .reference_checksum("analytic/deterministic-fixture:solver-corpus-reference-v2")
                .dimension(
                    "3x3 real general, 2x2 complex HPD, 3x3 Helmholtz stencil, and 2x2 rank-deficient rejection",
                )
                .formulation("dense/sparse direct LU, dense HPD Cholesky, block RHS, and GMRES")
                .tolerance(
                    "correctness metrics <= 1e-12; rank-deficient rejection equals one; positive factor workload metrics",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value, tolerance) in metrics {
                result = result.metric(name, value);
                if name == "fixture_count"
                    || name == "rank_deficient_rejected"
                    || name.contains("factor_")
                {
                    result = result.minimum(name, tolerance);
                } else {
                    result = result.maximum(name, tolerance);
                }
            }
            if let Ok(json) = result.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            if let Ok(json) = result.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
