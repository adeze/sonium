//! Deterministic CPU contract for bounded nested-trace solve orchestration.

use num_complex::Complex64;
use sonium_bem::{
    ConservativeTraceTransfer, CouplingConvention, CouplingError, NestedTraceEdge,
    NestedTraceOperator, NestedTraceRegion,
};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::{BlockGmresConfig, GmresConfig};
use sonium_solvers::sparse::DenseBlockOperator;
use sonium_solvers::traits::{AdjointCapability, Preconditioner};
use std::time::Instant;

const CASE_ID: &str = "bem.nested-trace-solve.v1";

fn convention() -> CouplingConvention {
    CouplingConvention {
        bem: sonium_bem::ACOUSTIC_CONVENTION,
        normal_points_out_of_interior: true,
    }
}

fn transfer() -> ConservativeTraceTransfer {
    ConservativeTraceTransfer::try_new(
        vec![vec![(0, 1.0 / 3.0), (1, 2.0 / 3.0)]],
        vec![1.0, 2.0],
        vec![3.0],
    )
    .expect("conservative transfer")
}

fn operator() -> Result<NestedTraceOperator, String> {
    NestedTraceOperator::try_new(
        vec![
            NestedTraceRegion::new(
                "interior",
                Box::new(DenseBlockOperator::new(DenseMatrix::from_row_major_vec(
                    2,
                    2,
                    vec![
                        Complex64::new(2.0, 0.1),
                        Complex64::new(0.2, -0.1),
                        Complex64::new(-0.3, 0.2),
                        Complex64::new(1.5, -0.2),
                    ],
                ))),
            ),
            NestedTraceRegion::new(
                "exterior",
                Box::new(DenseBlockOperator::new(DenseMatrix::from_row_major_vec(
                    1,
                    1,
                    vec![Complex64::new(1.3, 0.2)],
                ))),
            ),
        ],
        vec![NestedTraceEdge::new(
            "interface",
            0,
            1,
            transfer(),
            Complex64::new(0.15, -0.07),
        )],
        convention(),
    )
    .map_err(|error| error.to_string())
}

fn max_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max)
}

struct IdentityPreconditioner;

impl Preconditioner<Complex64> for IdentityPreconditioner {
    fn apply(&self, rhs: &[Complex64]) -> Vec<Complex64> {
        rhs.to_vec()
    }

    fn apply_transpose(&self, rhs: &[Complex64]) -> Vec<Complex64> {
        rhs.to_vec()
    }

    fn apply_hermitian(&self, rhs: &[Complex64]) -> Vec<Complex64> {
        rhs.to_vec()
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

fn matrix_free_errors(
    operator: &NestedTraceOperator,
    state: &[Complex64],
    rhs: &[Complex64],
    transpose_rhs: &[Complex64],
    hermitian_rhs: &[Complex64],
) -> Result<(f64, f64, f64, f64), String> {
    let preconditioner = IdentityPreconditioner;
    let config = GmresConfig {
        max_iterations: 8,
        restart: 5,
        tolerance: 1.0e-12,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 1,
        min_restart: None,
        max_restart: None,
    };
    let forward = operator
        .try_gmres_solve(rhs, &preconditioner, &config)
        .map_err(|error| error.to_string())?;
    let transpose = operator
        .try_gmres_solve_transpose(transpose_rhs, &preconditioner, &config)
        .map_err(|error| error.to_string())?;
    let hermitian = operator
        .try_gmres_solve_hermitian(hermitian_rhs, &preconditioner, &config)
        .map_err(|error| error.to_string())?;
    let block_config = BlockGmresConfig {
        max_iterations: 8,
        restart: 5,
        tolerance: 1.0e-12,
        true_residual_check: true,
        exact_residual_check_interval: 1,
    };
    let block_rhs = DenseMatrix::from_row_major_vec(
        state.len(),
        2,
        rhs.iter()
            .zip(hermitian_rhs)
            .flat_map(|(&forward, &adjoint)| [forward, adjoint])
            .collect(),
    );
    let block = operator
        .try_block_gmres_solve(&block_rhs, &preconditioner, &block_config)
        .map_err(|error| error.to_string())?;
    let block_error = (0..block_rhs.ncols())
        .map(|column| {
            operator
                .relative_residual(
                    &block.x.column_to_vec(column),
                    &block_rhs.column_to_vec(column),
                )
                .unwrap_or(f64::INFINITY)
        })
        .fold(0.0, f64::max);
    Ok((
        operator
            .relative_residual(&forward.x, rhs)
            .map_err(|error| error.to_string())?,
        operator
            .relative_transpose_residual(&transpose.x, transpose_rhs)
            .map_err(|error| error.to_string())?,
        operator
            .relative_hermitian_residual(&hermitian.x, hermitian_rhs)
            .map_err(|error| error.to_string())?,
        block_error,
    ))
}

fn run() -> Result<(f64, f64, f64, f64, usize), String> {
    let operator = operator()?;
    let state = vec![
        Complex64::new(0.2, -0.5),
        Complex64::new(-0.7, 0.3),
        Complex64::new(0.4, 0.9),
    ];
    let rhs = operator
        .try_apply(&state)
        .map_err(|error| error.to_string())?;
    let solved = operator.solve(&rhs).map_err(|error| error.to_string())?;
    let forward_error = max_error(&solved, &state);
    let transpose_rhs = operator
        .try_apply_transpose(&state)
        .map_err(|error| error.to_string())?;
    let transpose_solved = operator
        .solve_transpose(&transpose_rhs)
        .map_err(|error| error.to_string())?;
    let transpose_error = max_error(&transpose_solved, &state);
    let hermitian_rhs = operator
        .try_apply_hermitian(&state)
        .map_err(|error| error.to_string())?;
    let hermitian_solved = operator
        .solve_hermitian(&hermitian_rhs)
        .map_err(|error| error.to_string())?;
    let hermitian_error = max_error(&hermitian_solved, &state);
    let (
        iterative_forward_error,
        iterative_transpose_error,
        iterative_hermitian_error,
        iterative_block_error,
    ) = matrix_free_errors(&operator, &state, &rhs, &transpose_rhs, &hermitian_rhs)?;
    let forward_error = forward_error.max(iterative_forward_error);
    let transpose_error = transpose_error.max(iterative_transpose_error);
    let hermitian_error = hermitian_error.max(iterative_hermitian_error);
    let rhs_block = DenseMatrix::from_row_major_vec(
        3,
        2,
        vec![
            rhs[0],
            hermitian_rhs[0],
            rhs[1],
            hermitian_rhs[1],
            rhs[2],
            hermitian_rhs[2],
        ],
    );
    let solved_block = operator
        .solve_block(&rhs_block)
        .map_err(|error| error.to_string())?;
    let solved_hermitian_block = operator
        .solve_hermitian_block(&rhs_block)
        .map_err(|error| error.to_string())?;
    let transpose_rhs_block = DenseMatrix::from_row_major_vec(3, 1, transpose_rhs.clone());
    let solved_transpose_block = operator
        .solve_transpose_block(&transpose_rhs_block)
        .map_err(|error| error.to_string())?;
    let block_error = max_error(&solved_block.column_to_vec(0), &state)
        .max(max_error(&solved_hermitian_block.column_to_vec(1), &state))
        .max(max_error(&solved_transpose_block.column_to_vec(0), &state));
    let block_error = block_error.max(iterative_block_error);
    let mut rejection_count = 0;
    if matches!(
        operator.solve(&[Complex64::new(1.0, 0.0)]),
        Err(CouplingError::Dimension { .. })
    ) {
        rejection_count += 1;
    }
    let singular = NestedTraceOperator::try_new(
        vec![NestedTraceRegion::new(
            "singular",
            Box::new(DenseBlockOperator::new(DenseMatrix::from_row_major_vec(
                1,
                1,
                vec![Complex64::new(0.0, 0.0)],
            ))),
        )],
        Vec::new(),
        convention(),
    )
    .map_err(|error| error.to_string())?;
    if matches!(
        singular.solve(&[Complex64::new(1.0, 0.0)]),
        Err(CouplingError::OperatorFailure { .. })
    ) {
        rejection_count += 1;
    }
    Ok((
        forward_error,
        transpose_error,
        hermitian_error,
        block_error,
        rejection_count,
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((forward, transpose, hermitian, block, rejections))
            if forward <= 1.0e-10
                && transpose <= 1.0e-10
                && hermitian <= 1.0e-10
                && block <= 1.0e-10
                && rejections == 2 =>
        {
            QaResult::passed(CASE_ID)
                .metric("forward_solve_error", forward)
                .metric("transpose_solve_error", transpose)
                .metric("hermitian_solve_error", hermitian)
                .metric("block_solve_error", block)
                .metric("invalid_input_rejections", rejections as f64)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent nested trace action and direct LU solve")
                .dimension("2+1 nested trace DOFs")
                .formulation("checked bounded monolithic nested FEM-BEM solve orchestration")
                .seed("bem-nested-trace-solve:cpu:v1")
        }
        Ok((forward, transpose, hermitian, block, rejections)) => QaResult::failed(
            CASE_ID,
            format!(
                "solve tolerance failed: forward={forward:e}, transpose={transpose:e}, Hermitian={hermitian:e}, block={block:e}, rejections={rejections}"
            ),
        ),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    let result = result.elapsed_ms(started.elapsed().as_millis() as u64);
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
