//! Deterministic CPU contract for bounded FEM--BEM trace solves.

use num_complex::Complex64;
use sonium_bem::{
    ConservativeTraceTransfer, CoupledTraceOperator, CoupledTraceSystem, CouplingError,
};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::BlockGmresConfig;
use sonium_solvers::sparse::{CsrMatrix, DenseBlockOperator};
use sonium_solvers::traits::IdentityPreconditioner;
use std::time::Instant;

const CASE_ID: &str = "bem.fem-bem-trace-solve.v1";
type TraceSolveMetrics = (f64, f64, f64, f64, f64, f64, usize);

fn max_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max)
}

fn system() -> Result<CoupledTraceSystem, String> {
    let transfer = ConservativeTraceTransfer::try_new(
        vec![vec![(0, 0.25), (1, 0.75)], vec![(0, 0.75), (1, 0.25)]],
        vec![1.25, 1.75],
        vec![2.0, 1.0],
    )
    .map_err(|error| error.to_string())?;
    CoupledTraceSystem::try_new(
        DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![
                Complex64::new(2.0, 0.5),
                Complex64::new(-0.2, 0.1),
                Complex64::new(0.4, -0.3),
                Complex64::new(1.5, -0.25),
            ],
        ),
        DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![
                Complex64::new(0.8, -0.1),
                Complex64::new(0.3, 0.2),
                Complex64::new(-0.1, 0.4),
                Complex64::new(1.2, 0.6),
            ],
        ),
        transfer,
        Complex64::new(0.7, -0.2),
    )
    .map_err(|error| error.to_string())
}

fn matrix_free_operator() -> Result<CoupledTraceOperator, String> {
    let transfer = ConservativeTraceTransfer::try_new(
        vec![vec![(0, 1.0)], vec![(1, 1.0)]],
        vec![1.0, 1.0],
        vec![1.0, 1.0],
    )
    .map_err(|error| error.to_string())?;
    let interior = CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, Complex64::new(2.0, 0.15)),
            (1, 1, Complex64::new(2.7, -0.2)),
        ],
    );
    let exterior = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(1.6, -0.1),
            Complex64::new(0.12, 0.04),
            Complex64::new(-0.08, 0.03),
            Complex64::new(2.4, 0.25),
        ],
    );
    CoupledTraceOperator::try_new(
        Box::new(interior),
        Box::new(DenseBlockOperator::new(exterior)),
        transfer,
        Complex64::new(0.2, -0.08),
    )
    .map_err(|error| error.to_string())
}

fn run() -> Result<TraceSolveMetrics, String> {
    let system = system()?;
    let state = vec![
        Complex64::new(0.25, -0.5),
        Complex64::new(-1.0, 0.75),
        Complex64::new(0.5, 0.125),
        Complex64::new(-0.25, 1.0),
    ];
    let forward_rhs = system.apply(&state).map_err(|error| error.to_string())?;
    let transpose_rhs = system
        .apply_transpose(&state)
        .map_err(|error| error.to_string())?;
    let hermitian_rhs = system
        .apply_adjoint(&state)
        .map_err(|error| error.to_string())?;
    let forward = system
        .solve(&forward_rhs)
        .map_err(|error| error.to_string())?;
    let transpose = system
        .solve_transpose(&transpose_rhs)
        .map_err(|error| error.to_string())?;
    let hermitian = system
        .solve_adjoint(&hermitian_rhs)
        .map_err(|error| error.to_string())?;
    let rhs_block = DenseMatrix::from_row_major_vec(
        4,
        3,
        vec![
            forward_rhs[0],
            transpose_rhs[0],
            hermitian_rhs[0],
            forward_rhs[1],
            transpose_rhs[1],
            hermitian_rhs[1],
            forward_rhs[2],
            transpose_rhs[2],
            hermitian_rhs[2],
            forward_rhs[3],
            transpose_rhs[3],
            hermitian_rhs[3],
        ],
    );
    let forward_block = system
        .solve_block(&rhs_block)
        .map_err(|error| error.to_string())?;
    let transpose_block = system
        .solve_transpose_block(&rhs_block)
        .map_err(|error| error.to_string())?;
    let hermitian_block = system
        .solve_adjoint_block(&rhs_block)
        .map_err(|error| error.to_string())?;
    let block_error = max_error(&forward_block.column_to_vec(0), &state)
        .max(max_error(&transpose_block.column_to_vec(1), &state))
        .max(max_error(&hermitian_block.column_to_vec(2), &state));

    let matrix_free = matrix_free_operator()?;
    let matrix_free_state = DenseMatrix::from_row_major_vec(
        4,
        2,
        vec![
            Complex64::new(0.4, -0.6),
            Complex64::new(-0.2, 0.8),
            Complex64::new(0.9, 0.1),
            Complex64::new(-0.7, -0.3),
            Complex64::new(-0.15, 0.25),
            Complex64::new(0.55, -0.4),
            Complex64::new(0.3, 0.75),
            Complex64::new(-0.8, 0.1),
        ],
    );
    let matrix_free_rhs = matrix_free
        .try_apply_block(&matrix_free_state)
        .map_err(|error| error.to_string())?;
    let matrix_free_config = BlockGmresConfig {
        max_iterations: 8,
        restart: 4,
        tolerance: 1.0e-12,
        true_residual_check: true,
        exact_residual_check_interval: 1,
    };
    let matrix_free_solution = matrix_free
        .try_block_gmres(&matrix_free_rhs, &matrix_free_config)
        .map_err(|error| error.to_string())?;
    let matrix_free_preconditioned = matrix_free
        .try_block_gmres_preconditioned(
            &matrix_free_rhs,
            &IdentityPreconditioner,
            &matrix_free_config,
        )
        .map_err(|error| error.to_string())?;
    if !matrix_free_solution.converged || !matrix_free_preconditioned.converged {
        return Err("matrix-free block GMRES did not converge".to_owned());
    }
    let matrix_free_error = (0..matrix_free_state.ncols())
        .map(|column| {
            max_error(
                &matrix_free_solution.x.column_to_vec(column),
                &matrix_free_state.column_to_vec(column),
            )
        })
        .fold(0.0, f64::max);
    let matrix_free_true_residual = matrix_free_solution
        .true_residuals
        .iter()
        .copied()
        .fold(0.0, f64::max)
        .max(
            matrix_free_preconditioned
                .true_residuals
                .iter()
                .copied()
                .fold(0.0, f64::max),
        );

    let mut rejections = 0;
    if !matches!(
        system.solve(&[Complex64::new(1.0, 0.0)]),
        Err(CouplingError::Dimension { .. })
    ) {
        return Err("wrong-size solve was not rejected".to_owned());
    }
    rejections += 1;
    if !matches!(
        system.solve_transpose(&[Complex64::new(f64::NAN, 0.0); 4]),
        Err(CouplingError::NonFinite { .. })
    ) {
        return Err("non-finite transpose rhs was not rejected".to_owned());
    }
    rejections += 1;

    Ok((
        max_error(&forward, &state),
        max_error(&transpose, &state),
        max_error(&hermitian, &state),
        block_error,
        matrix_free_error,
        matrix_free_true_residual,
        rejections,
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((
            forward,
            transpose,
            hermitian,
            block,
            matrix_free_error,
            matrix_free_true_residual,
            rejections,
        ))
            if forward <= 1.0e-10
                && transpose <= 1.0e-10
                && hermitian <= 1.0e-10
                && block <= 1.0e-10
        && matrix_free_error <= 1.0e-10
        && matrix_free_true_residual <= 1.0e-10
        && rejections == 2 => QaResult::passed(CASE_ID)
                .metric("forward_solve_error", forward)
                .metric("transpose_solve_error", transpose)
                .metric("hermitian_solve_error", hermitian)
        .metric("block_solve_error", block)
        .metric("matrix_free_block_solve_error", matrix_free_error)
        .metric(
            "matrix_free_block_true_residual",
            matrix_free_true_residual,
        )
        .metric("invalid_input_rejections", rejections as f64)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent coupled trace action and dense LU solve")
                .input_checksum("analytic/deterministic-fixture:bem-fem-bem-trace-solve-v1")
                .reference_checksum("dense-coupled-trace-system-v1")
                .dimension("4 DOFs; 2 FEM + 2 BEM traces")
                .seed("bem-fem-bem-trace-solve:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64),
        Ok((
            forward,
            transpose,
            hermitian,
            block,
            matrix_free_error,
            matrix_free_true_residual,
            rejections,
        )) => QaResult::failed(
            CASE_ID,
            format!(
                "solve error exceeded gate: forward={forward:e}, transpose={transpose:e}, hermitian={hermitian:e}, block={block:e}, matrix_free_error={matrix_free_error:e}, matrix_free_true_residual={matrix_free_true_residual:e}, rejections={rejections}"
            ),
        )
        .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(error) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
