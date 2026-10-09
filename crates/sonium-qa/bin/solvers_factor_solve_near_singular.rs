//! Independent CPU evidence for near-singular sparse-LU factor classes.

use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{
    DenseDirectConfig, DenseDirectMethod, DirectFactorization, MatrixClass, Ordering,
    SparseDirectConfig, SparseDirectMethod, factorize_dense, factorize_sparse,
};
use sonium_solvers::sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.factor-solve-near-singular.v1";
const DELTAS: [f64; 3] = [1.0e-6, 1.0e-8, 1.0e-10];
const VARIANTS: [(SparseDirectMethod, bool); 4] = [
    (SparseDirectMethod::SimplicialLu, false),
    (SparseDirectMethod::SupernodalLu, false),
    (SparseDirectMethod::SimplicialLu, true),
    (SparseDirectMethod::SupernodalLu, true),
];
const TOLERANCE: f64 = 1.0e-8;

fn matrix(delta: f64) -> Vec<Vec<f64>> {
    vec![vec![1.0, 1.0], vec![1.0, 1.0 + delta]]
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

fn matvec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn max_residual(matrix: &[Vec<f64>], solution: &[f64], rhs: &[f64]) -> f64 {
    matvec(matrix, solution)
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let expected_columns = [[2.0_f64, -1.0_f64], [-0.5_f64, 1.25_f64]];
    let mut max_dense_residual = 0.0_f64;
    let mut max_sparse_residual = 0.0_f64;
    let mut max_block_residual = 0.0_f64;
    let mut finite_solution_cases = 0.0_f64;
    let mut nonfinite_rejections = 0.0_f64;
    let mut preserved_destinations = 0.0_f64;

    for &delta in &DELTAS {
        let matrix = matrix(delta);
        let dense = DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0 + delta]);
        let rhs_columns: Vec<Vec<f64>> = expected_columns
            .iter()
            .map(|column| matvec(&matrix, column))
            .collect();
        let block_rhs: Vec<f64> = (0..2)
            .flat_map(|row| [rhs_columns[0][row], rhs_columns[1][row]])
            .collect();

        let dense_factor = factorize_dense(
            &dense,
            &DenseDirectConfig {
                matrix_class: MatrixClass::General,
                method: DenseDirectMethod::Lu,
                ..Default::default()
            },
        )
        .map_err(|error| format!("dense factorization delta={delta:e}: {error}"))?;
        for rhs in &rhs_columns {
            let solution = dense_factor
                .solve(rhs)
                .map_err(|error| format!("dense solve delta={delta:e}: {error}"))?;
            let residual = max_residual(&matrix, &solution, rhs);
            if !solution.iter().all(|value| value.is_finite()) || residual > TOLERANCE {
                return Err(format!(
                    "dense near-singular residual delta={delta:e}: {residual:e}"
                ));
            }
            max_dense_residual = max_dense_residual.max(residual);
        }

        for &(method, equilibrate) in &VARIANTS {
            let factor = factorize_sparse(
                &csr(&matrix),
                &SparseDirectConfig {
                    matrix_class: MatrixClass::General,
                    method,
                    ordering: Ordering::Auto,
                    equilibrate,
                    ..Default::default()
                },
                None,
            )
            .map_err(|error| {
                format!(
                    "sparse factorization delta={delta:e} method={method:?} equilibrate={equilibrate}: {error}"
                )
            })?;
            let mut block_solution = vec![f64::NAN; block_rhs.len()];
            factor
                .solve_many_into(&block_rhs, &mut block_solution, 2)
                .map_err(|error| format!("block solve delta={delta:e}: {error}"))?;
            if !block_solution.iter().all(|value| value.is_finite()) {
                return Err(format!(
                    "non-finite block solution delta={delta:e} method={method:?} equilibrate={equilibrate}"
                ));
            }
            finite_solution_cases += 1.0;
            for column in 0..2 {
                let solution = block_solution
                    .iter()
                    .skip(column)
                    .step_by(2)
                    .copied()
                    .collect::<Vec<_>>();
                let residual = max_residual(&matrix, &solution, &rhs_columns[column]);
                if !residual.is_finite() || residual > TOLERANCE {
                    return Err(format!(
                        "sparse near-singular residual delta={delta:e} method={method:?} equilibrate={equilibrate}: {residual:e}"
                    ));
                }
                max_sparse_residual = max_sparse_residual.max(residual);
                max_block_residual = max_block_residual.max(residual);
            }

            let mut nonfinite_rhs = block_rhs.clone();
            nonfinite_rhs[2] = f64::NAN;
            let destination = vec![17.0_f64; block_rhs.len()];
            let mut rejected = destination.clone();
            if factor
                .solve_many_into(&nonfinite_rhs, &mut rejected, 2)
                .is_err()
                && rejected == destination
            {
                nonfinite_rejections += 1.0;
                preserved_destinations += 1.0;
            }
        }
    }

    let expected_variant_cases = (DELTAS.len() * VARIANTS.len()) as f64;
    if finite_solution_cases != expected_variant_cases
        || nonfinite_rejections != expected_variant_cases
        || preserved_destinations != expected_variant_cases
    {
        return Err(format!(
            "near-singular factor counts finite={finite_solution_cases}, rejections={nonfinite_rejections}, preserved={preserved_destinations}, expected={expected_variant_cases}"
        ));
    }

    Ok(vec![
        ("delta_case_count", DELTAS.len() as f64),
        ("factor_variant_count", VARIANTS.len() as f64),
        ("finite_solution_cases", finite_solution_cases),
        ("nonfinite_rejections", nonfinite_rejections),
        ("preserved_destinations", preserved_destinations),
        ("max_dense_residual", max_dense_residual),
        ("max_sparse_residual", max_sparse_residual),
        ("max_block_residual", max_block_residual),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent near-singular A*x residual oracle")
                .reference_checksum("analytic/deterministic-fixture:factor-solve-near-singular-v1")
                .dimension("2x2 f64 general LU; three deltas; four sparse factor variants; two RHS")
                .formulation("dense/sparse simplicial/supernodal LU with optional equilibration and transactional block admission")
                .tolerance("all residuals <= 1e-8; every variant finite; 12 rejection and destination-preservation checks")
                .seed("factor-solve-near-singular:delta-bank:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.metric(name, value);
            }
            result = result
                .minimum("delta_case_count", DELTAS.len() as f64)
                .minimum("factor_variant_count", VARIANTS.len() as f64)
                .minimum(
                    "finite_solution_cases",
                    (DELTAS.len() * VARIANTS.len()) as f64,
                )
                .minimum(
                    "nonfinite_rejections",
                    (DELTAS.len() * VARIANTS.len()) as f64,
                )
                .minimum(
                    "preserved_destinations",
                    (DELTAS.len() * VARIANTS.len()) as f64,
                )
                .maximum("max_dense_residual", TOLERANCE)
                .maximum("max_sparse_residual", TOLERANCE)
                .maximum("max_block_residual", TOLERANCE);
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result =
                QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
