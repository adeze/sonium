//! Independent CPU QA for differentiable block forward/adjoint solves.

use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::{
    AccuracyStatus, AdjointSolve, BlockFgmresConfig, BlockFgmresSolution, CsrMatrix,
    DifferentiableBlockSolveResult, DifferentiableLinearSolve, GmresConfig, IdentityPreconditioner,
    KrylovOpCounts, SolverResultContract,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.differentiable-block-adjoint.v1";
const TOLERANCE: f64 = 1.0e-9;

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 2, 2.0),
        ],
    )
}

fn matvec(x: &[f64; 3]) -> [f64; 3] {
    [
        4.0 * x[0] + x[1],
        x[0] + 3.0 * x[1] + x[2],
        x[1] + 2.0 * x[2],
    ]
}

fn block(columns: &[[f64; 3]; 2]) -> DenseMatrix<f64> {
    let mut result = DenseMatrix::zeros(3, 2);
    for (column, values) in columns.iter().enumerate() {
        for (row, value) in values.iter().enumerate() {
            result[[row, column]] = *value;
        }
    }
    result
}

fn max_error(actual: &DenseMatrix<f64>, expected: &[[f64; 3]; 2]) -> f64 {
    expected
        .iter()
        .enumerate()
        .flat_map(|(column, values)| {
            values
                .iter()
                .enumerate()
                .map(move |(row, expected)| (actual[[row, column]] - expected).abs())
        })
        .fold(0.0, f64::max)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let expected_forward = [[1.0, 2.0, -1.0], [2.0, -1.0, 3.0]];
    let expected_adjoint = [[3.0, -2.0, 1.0], [-1.0, 4.0, 2.0]];
    let forward_rhs = block(&expected_forward.map(|column| matvec(&column)));
    let adjoint_rhs = block(&expected_adjoint.map(|column| matvec(&column)));
    let operator = matrix();
    let preconditioner = IdentityPreconditioner;
    let handle = DifferentiableLinearSolve::new(
        &operator,
        &preconditioner,
        GmresConfig {
            max_iterations: 30,
            restart: 3,
            tolerance: 1.0e-12,
            ..GmresConfig::default()
        },
    )
    .map_err(|error| format!("differentiable handle failed: {error:?}"))?;
    let block_config = BlockFgmresConfig {
        max_iterations: 30,
        restart: 3,
        tolerance: 1.0e-11,
        true_residual_check: true,
        exact_residual_check_interval: 0,
    };
    let result = handle
        .solve_block(
            &forward_rhs,
            Some((&adjoint_rhs, AdjointSolve::Hermitian)),
            &block_config,
        )
        .map_err(|error| format!("differentiable block solve failed: {error:?}"))?;
    result
        .validate_contract()
        .map_err(|error| format!("block result contract failed: {error:?}"))?;
    let malformed: DifferentiableBlockSolveResult<f64> = DifferentiableBlockSolveResult {
        forward: BlockFgmresSolution {
            x: DenseMatrix::zeros(2, 1),
            iterations: 1,
            residuals: vec![0.0],
            true_residuals: vec![0.0],
            backward_errors: vec![0.0],
            attainable_accuracy: vec![AccuracyStatus::Attained],
            converged: true,
            op_counts: KrylovOpCounts::default(),
        },
        adjoint: Some(BlockFgmresSolution {
            x: DenseMatrix::zeros(3, 1),
            iterations: 1,
            residuals: vec![0.0],
            true_residuals: vec![0.0],
            backward_errors: vec![0.0],
            attainable_accuracy: vec![AccuracyStatus::Attained],
            converged: true,
            op_counts: KrylovOpCounts::default(),
        }),
    };
    if !matches!(
        malformed.validate_contract(),
        Err(sonium_solvers_core::traits::SolverError::DimensionMismatch { .. })
    ) {
        return Err("differentiable block dimension contract did not fail closed".into());
    }
    let adjoint = result
        .adjoint
        .as_ref()
        .ok_or_else(|| "adjoint result missing".to_string())?;
    let continuation_forward = handle
        .solve_forward_block_with_guess(&forward_rhs, Some(&result.forward.x), &block_config)
        .map_err(|error| format!("forward differentiable continuation failed: {error:?}"))?;
    let continuation_adjoint = handle
        .solve_adjoint_block_with_guess(
            &adjoint_rhs,
            AdjointSolve::Hermitian,
            Some(&adjoint.x),
            &block_config,
        )
        .map_err(|error| format!("adjoint differentiable continuation failed: {error:?}"))?;
    if !continuation_forward.converged
        || continuation_forward.iterations != 0
        || !continuation_adjoint.converged
        || continuation_adjoint.iterations != 0
        || continuation_forward
            .true_residuals
            .iter()
            .any(|residual| *residual > TOLERANCE)
        || continuation_adjoint
            .true_residuals
            .iter()
            .any(|residual| *residual > TOLERANCE)
    {
        return Err(format!(
            "differentiable continuation did not accept converged iterates: forward iterations={}, adjoint iterations={}, forward residuals={:?}, adjoint residuals={:?}",
            continuation_forward.iterations,
            continuation_adjoint.iterations,
            continuation_forward.true_residuals,
            continuation_adjoint.true_residuals,
        ));
    }
    let forward_error = max_error(&result.forward.x, &expected_forward);
    let adjoint_error = max_error(&adjoint.x, &expected_adjoint);
    let forward_residual = result
        .forward
        .true_residuals
        .iter()
        .copied()
        .fold(0.0, f64::max);
    let adjoint_residual = adjoint.true_residuals.iter().copied().fold(0.0, f64::max);
    if forward_error > TOLERANCE
        || adjoint_error > TOLERANCE
        || forward_residual > TOLERANCE
        || adjoint_residual > TOLERANCE
    {
        return Err(format!(
            "differentiable block errors: forward={forward_error:e}, adjoint={adjoint_error:e}, forward residual={forward_residual:e}, adjoint residual={adjoint_residual:e}"
        ));
    }
    Ok(vec![
        ("forward_solution_error", forward_error),
        ("adjoint_solution_error", adjoint_error),
        ("forward_independent_residual", forward_residual),
        ("adjoint_independent_residual", adjoint_residual),
        ("forward_iterations", result.forward.iterations as f64),
        ("adjoint_iterations", adjoint.iterations as f64),
        (
            "forward_continuation_iterations",
            continuation_forward.iterations as f64,
        ),
        (
            "adjoint_continuation_iterations",
            continuation_adjoint.iterations as f64,
        ),
        ("rhs_columns", result.forward.x.ncols() as f64),
        ("dimension_contract", 1.0),
        ("continuation_contract", 1.0),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("analytic symmetric 3x3 matrix with independent forward/adjoint block solutions")
                .input_checksum("analytic/deterministic-fixture:differentiable-block-v1")
                .reference_checksum("analytic/deterministic-fixture:differentiable-block-v1")
                .dimension("3x3 CSR; two forward and two Hermitian-adjoint RHS columns")
                .tolerance(
                    "solution errors and true residuals <= 1e-9; converged block continuation accepts zero iterations",
                )
                .seed("solvers-differentiable-block:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                if name.contains("error") || name.contains("residual") {
                    result = result.maximum(name, TOLERANCE).metric(name, value);
                } else {
                    result = result.metric(name, value);
                }
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("analytic symmetric 3x3 matrix with independent forward/adjoint block solutions")
                .input_checksum("analytic/deterministic-fixture:differentiable-block-v1")
                .reference_checksum("analytic/deterministic-fixture:differentiable-block-v1")
                .tolerance("solution errors and true residuals <= 1e-9")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
