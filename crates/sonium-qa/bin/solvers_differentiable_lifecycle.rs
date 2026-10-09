//! Independent QA for transient, eigen, and coupled differentiable contracts.

use sonium_qa::QaResult;
use sonium_solvers::iterative::{BlockFgmresSolution, KrylovOpCounts};
use sonium_solvers::time_integration::{NewmarkAdjointResult, NewmarkResult};
use sonium_solvers::{
    AccuracyStatus, AdjointCapability, AdjointSolve, CancellationToken, CsrMatrix,
    DifferentiableLinearSolve, DifferentiableSolveContract, DifferentiableSystemKind, GmresConfig,
    IdentityPreconditioner, LinearOperator, SolverError, SteadyAdjointSeed, block_contract,
    eigen_value_contract, eigen_vector_contract, transient_contract, verify_forward_replay,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.differentiable-lifecycle-contract.v1";
const TOLERANCE: f64 = 1.0e-12;

struct MidIterationCancellationOperator<'a> {
    cancellation: &'a CancellationToken,
}

impl LinearOperator<f64> for MidIterationCancellationOperator<'_> {
    fn num_rows(&self) -> usize {
        2
    }

    fn num_cols(&self) -> usize {
        2
    }

    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
        self.cancellation.cancel();
    }

    fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
        self.cancellation.cancel();
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

fn transient_state(dimension: usize) -> NewmarkResult<f64> {
    NewmarkResult {
        displacement: vec![0.0; dimension],
        velocity: vec![0.0; dimension],
        acceleration: vec![0.0; dimension],
        displacement_history: None,
        velocity_history: None,
        acceleration_history: None,
        steps_completed: 1,
        final_time: 1.0,
    }
}

fn transient_adjoint(dimension: usize) -> NewmarkAdjointResult<f64> {
    NewmarkAdjointResult {
        force_gradient: vec![vec![0.0; dimension]],
        initial_displacement_gradient: vec![0.0; dimension],
        initial_velocity_gradient: vec![0.0; dimension],
        initial_acceleration_gradient: vec![0.0; dimension],
    }
}

fn block_solution(rows: usize) -> BlockFgmresSolution<f64> {
    BlockFgmresSolution {
        x: sonium_solvers::dense::DenseMatrix::zeros(rows, 1),
        iterations: 1,
        residuals: vec![0.0],
        true_residuals: vec![0.0],
        backward_errors: vec![0.0],
        attainable_accuracy: vec![AccuracyStatus::Attained],
        converged: true,
        op_counts: KrylovOpCounts::default(),
    }
}

fn generated_output_rejections() -> usize {
    let transient_forward = transient_contract(
        |_input: &[f64]| {
            let mut state = transient_state(2);
            state.displacement[0] = f64::NAN;
            Ok::<_, SolverError>(state)
        },
        |_forward: &NewmarkResult<f64>, _seed: &[f64]| Ok(transient_adjoint(2)),
    );
    let transient_forward_rejected = transient_forward.solve_forward(&[1.0, 2.0]).is_err();

    let transient_adjoint_contract = transient_contract(
        |_input: &[f64]| Ok::<_, SolverError>(transient_state(2)),
        |_forward: &NewmarkResult<f64>, _seed: &[f64]| {
            let mut adjoint = transient_adjoint(2);
            adjoint.initial_velocity_gradient[0] = f64::INFINITY;
            Ok::<_, SolverError>(adjoint)
        },
    );
    let transient_state_value = transient_adjoint_contract.solve_forward(&[1.0, 2.0]).ok();
    let transient_adjoint_rejected = transient_state_value.is_some_and(|state| {
        transient_adjoint_contract
            .solve_adjoint(&state, &[0.25, -0.5])
            .is_err()
    });

    let eigen_value_forward = eigen_value_contract(
        |_input: &[f64]| Ok::<_, SolverError>(f64::NAN),
        |_forward: &f64, _seed: &[f64]| Ok::<_, SolverError>(0.0),
    );
    let eigen_value_forward_rejected = eigen_value_forward.solve_forward(&[1.0]).is_err();

    let eigen_value_adjoint = eigen_value_contract(
        |_input: &[f64]| Ok::<_, SolverError>(1.0),
        |_forward: &f64, _seed: &[f64]| Ok::<_, SolverError>(f64::INFINITY),
    );
    let eigen_value_state = eigen_value_adjoint.solve_forward(&[1.0]).ok();
    let eigen_value_adjoint_rejected = eigen_value_state
        .is_some_and(|state| eigen_value_adjoint.solve_adjoint(&state, &[0.5]).is_err());

    let eigen_vector_forward = eigen_vector_contract(
        |_input: &[f64]| Ok::<_, SolverError>(vec![f64::NAN, 1.0]),
        |_forward: &Vec<f64>, _seed: &[f64]| Ok::<_, SolverError>(vec![0.0, 0.0]),
    );
    let eigen_vector_forward_rejected = eigen_vector_forward.solve_forward(&[1.0, 2.0]).is_err();

    let eigen_vector_adjoint = eigen_vector_contract(
        |_input: &[f64]| Ok::<_, SolverError>(vec![1.0, 2.0]),
        |_forward: &Vec<f64>, _seed: &[f64]| Ok::<_, SolverError>(vec![0.0, f64::INFINITY]),
    );
    let eigen_vector_state = eigen_vector_adjoint.solve_forward(&[1.0, 2.0]).ok();
    let eigen_vector_adjoint_rejected = eigen_vector_state.is_some_and(|state| {
        eigen_vector_adjoint
            .solve_adjoint(&state, &[0.5, -0.25])
            .is_err()
    });

    let block_forward = block_contract(
        |_input: &[f64]| {
            let mut state = block_solution(2);
            state.x[[0, 0]] = f64::NAN;
            Ok::<_, SolverError>(state)
        },
        |_forward: &BlockFgmresSolution<f64>, _seed: &[f64]| Ok::<_, SolverError>(vec![0.0; 2]),
    );
    let block_forward_rejected = block_forward.solve_forward(&[1.0, 2.0]).is_err();

    let block_adjoint = block_contract(
        |_input: &[f64]| Ok::<_, SolverError>(block_solution(2)),
        |_forward: &BlockFgmresSolution<f64>, _seed: &[f64]| {
            Ok::<_, SolverError>(vec![f64::INFINITY, 0.0])
        },
    );
    let block_state = block_adjoint.solve_forward(&[1.0, 2.0]).ok();
    let block_adjoint_rejected = block_state
        .is_some_and(|state| block_adjoint.solve_adjoint(&state, &[0.25, -0.5]).is_err());

    let malformed_metadata = block_contract(
        |_input: &[f64]| {
            let mut state = block_solution(2);
            state.attainable_accuracy[0] = AccuracyStatus::NotAttained;
            Ok::<_, SolverError>(state)
        },
        |_forward: &BlockFgmresSolution<f64>, _seed: &[f64]| -> Result<Vec<f64>, SolverError> {
            panic!("adjoint callback must not run for malformed block result")
        },
    );
    let malformed_metadata_rejected = malformed_metadata.solve_forward(&[1.0, 2.0]).is_err();

    [
        transient_forward_rejected,
        transient_adjoint_rejected,
        eigen_value_forward_rejected,
        eigen_value_adjoint_rejected,
        eigen_vector_forward_rejected,
        eigen_vector_adjoint_rejected,
        block_forward_rejected,
        block_adjoint_rejected,
        malformed_metadata_rejected,
    ]
    .into_iter()
    .filter(|rejected| *rejected)
    .count()
}

fn steady_retained_state_rejection() -> bool {
    let operator = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 2.0)]);
    let preconditioner = IdentityPreconditioner;
    let config = GmresConfig {
        max_iterations: 4,
        restart: 1,
        tolerance: 1.0e-12,
        ..GmresConfig::default()
    };
    let Ok(handle) = DifferentiableLinearSolve::new(&operator, &preconditioner, config) else {
        return false;
    };
    let Ok(mut forward) = handle.solve_forward(&[2.0]) else {
        return false;
    };
    forward.x[0] = f64::NAN;
    let seed = SteadyAdjointSeed::new(vec![1.0], AdjointSolve::Hermitian);
    matches!(
        DifferentiableSolveContract::solve_adjoint(&handle, &forward, &seed),
        Err(SolverError::OperationFailed { .. })
    )
}

fn steady_mid_iteration_cancellation() -> bool {
    let cancellation = CancellationToken::new();
    let operator = MidIterationCancellationOperator {
        cancellation: &cancellation,
    };
    let config = GmresConfig {
        max_iterations: 4,
        restart: 1,
        tolerance: 1.0e-12,
        ..GmresConfig::default()
    };
    let Ok(handle) = DifferentiableLinearSolve::new(&operator, &IdentityPreconditioner, config)
    else {
        return false;
    };
    matches!(
        handle.solve_forward_cancellable(&[1.0, 0.0], &cancellation),
        Err(SolverError::OperationFailed { ref reason, .. }) if reason.contains("cancelled")
    )
}

fn steady_block_mid_iteration_cancellation() -> bool {
    let cancellation = CancellationToken::new();
    let operator = MidIterationCancellationOperator {
        cancellation: &cancellation,
    };
    let config = sonium_solvers::iterative::BlockFgmresConfig {
        max_iterations: 4,
        restart: 1,
        tolerance: 1.0e-12,
        ..sonium_solvers::iterative::BlockFgmresConfig::default()
    };
    let rhs = sonium_solvers::dense::DenseMatrix::from_row_major_vec(2, 1, vec![1.0, 0.0]);
    let Ok(handle) = DifferentiableLinearSolve::new(
        &operator,
        &IdentityPreconditioner,
        GmresConfig {
            max_iterations: 4,
            restart: 1,
            tolerance: 1.0e-12,
            ..GmresConfig::default()
        },
    ) else {
        return false;
    };
    matches!(
        handle.solve_forward_block_cancellable(&rhs, &config, &cancellation),
        Err(SolverError::OperationFailed { ref reason, .. }) if reason.contains("cancelled")
    )
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let transient = transient_contract(
        |_input: &[f64]| Ok(transient_state(2)),
        |_forward: &NewmarkResult<f64>, _seed: &[f64]| Ok(transient_adjoint(2)),
    );
    let transient_forward = transient
        .solve_forward(&[1.0, 2.0])
        .map_err(|error| format!("transient forward failed: {error:?}"))?;
    transient
        .solve_adjoint(&transient_forward, &[0.25, -0.5])
        .map_err(|error| format!("transient adjoint failed: {error:?}"))?;
    let transient_replay = verify_forward_replay(
        &transient,
        &[1.0, 2.0],
        |left: &NewmarkResult<f64>, right: &NewmarkResult<f64>| {
            left.displacement == right.displacement
                && left.velocity == right.velocity
                && left.acceleration == right.acceleration
                && left.steps_completed == right.steps_completed
                && left.final_time == right.final_time
        },
    )
    .map_err(|error| format!("transient replay failed: {error:?}"))?;
    let transient_dimension_rejection = {
        let malformed = transient_contract(
            |_input: &[f64]| Ok(transient_state(2)),
            |_forward: &NewmarkResult<f64>, _seed: &[f64]| Ok(transient_adjoint(3)),
        );
        let forward = malformed
            .solve_forward(&[1.0, 2.0])
            .map_err(|error| format!("malformed transient forward failed: {error:?}"))?;
        matches!(
            malformed.solve_adjoint(&forward, &[0.25, -0.5]),
            Err(SolverError::OperationFailed { .. })
        )
    };

    let block = block_contract(
        |_input: &[f64]| Ok(block_solution(2)),
        |_forward: &BlockFgmresSolution<f64>, _seed: &[f64]| Ok(vec![0.0; 2]),
    );
    let block_forward = block
        .solve_forward(&[1.0, 2.0])
        .map_err(|error| format!("block forward failed: {error:?}"))?;
    block
        .solve_adjoint(&block_forward, &[0.25, -0.5])
        .map_err(|error| format!("block adjoint failed: {error:?}"))?;
    let block_replay = verify_forward_replay(
        &block,
        &[1.0, 2.0],
        |left: &BlockFgmresSolution<f64>, right: &BlockFgmresSolution<f64>| {
            left.x.as_slice() == right.x.as_slice()
                && left.iterations == right.iterations
                && left.residuals == right.residuals
                && left.true_residuals == right.true_residuals
                && left.backward_errors == right.backward_errors
                && left.converged == right.converged
        },
    )
    .map_err(|error| format!("block replay failed: {error:?}"))?;
    let block_dimension_rejection = {
        let malformed = block_contract(
            |_input: &[f64]| Ok(block_solution(2)),
            |_forward: &BlockFgmresSolution<f64>, _seed: &[f64]| Ok(vec![0.0; 3]),
        );
        let forward = malformed
            .solve_forward(&[1.0, 2.0])
            .map_err(|error| format!("malformed block forward failed: {error:?}"))?;
        matches!(
            malformed.solve_adjoint(&forward, &[0.25, -0.5]),
            Err(SolverError::OperationFailed { .. })
        )
    };

    let eigen_value = eigen_value_contract(
        |input: &[f64]| Ok(input[0] * 2.0),
        |forward: &f64, seed: &[f64]| Ok(*forward + seed[0]),
    );
    let eigen_value_forward = eigen_value
        .solve_forward(&[2.0])
        .map_err(|error| format!("eigenvalue forward failed: {error:?}"))?;
    let eigen_value_adjoint = eigen_value
        .solve_adjoint(&eigen_value_forward, &[0.5])
        .map_err(|error| format!("eigenvalue adjoint failed: {error:?}"))?;
    let eigen_value_replay =
        verify_forward_replay(&eigen_value, &[2.0], |left: &f64, right: &f64| {
            left == right
        })
        .map_err(|error| format!("eigenvalue replay failed: {error:?}"))?;

    let eigen_vector = eigen_vector_contract(
        |input: &[f64]| Ok(vec![input[0], input[0] + 1.0]),
        |forward: &Vec<f64>, seed: &[f64]| {
            Ok(forward
                .iter()
                .zip(seed)
                .map(|(value, direction)| value + direction)
                .collect::<Vec<_>>())
        },
    );
    let eigen_vector_forward = eigen_vector
        .solve_forward(&[2.0, 3.0])
        .map_err(|error| format!("eigenvector forward failed: {error:?}"))?;
    let eigen_vector_adjoint = eigen_vector
        .solve_adjoint(&eigen_vector_forward, &[0.5, -0.25])
        .map_err(|error| format!("eigenvector adjoint failed: {error:?}"))?;
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let cancellation_rejections = [
        transient
            .solve_forward_cancellable(&[1.0, 2.0], &cancellation)
            .is_err(),
        block
            .solve_forward_cancellable(&[1.0, 2.0], &cancellation)
            .is_err(),
        eigen_value
            .solve_forward_cancellable(&[2.0], &cancellation)
            .is_err(),
    ]
    .into_iter()
    .filter(|rejected| *rejected)
    .count();
    let post_cancellation = CancellationToken::new();
    let callback_token = post_cancellation.clone();
    let post_cancellation_contract = eigen_value_contract(
        move |input: &[f64]| {
            callback_token.cancel();
            Ok(input[0] * 2.0)
        },
        |_forward: &f64, seed: &[f64]| Ok(seed[0]),
    );
    let post_cancellation_rejected = post_cancellation_contract
        .solve_forward_cancellable(&[2.0], &post_cancellation)
        .is_err();
    let cancellation_rejections = cancellation_rejections + usize::from(post_cancellation_rejected);
    let generated_output_rejections = generated_output_rejections();
    let steady_retained_state_rejected = steady_retained_state_rejection();
    let mid_iteration_cancellation = steady_mid_iteration_cancellation();
    let block_mid_iteration_cancellation = steady_block_mid_iteration_cancellation();
    if transient_replay.evaluations != 2
        || block_replay.evaluations != 2
        || eigen_value_replay.evaluations != 2
        || cancellation_rejections != 4
        || !mid_iteration_cancellation
        || !block_mid_iteration_cancellation
        || generated_output_rejections != 9
        || !steady_retained_state_rejected
    {
        return Err(
            "differentiable lifecycle contract metrics did not meet their exact counts".to_string(),
        );
    }

    if transient.system_kind() != DifferentiableSystemKind::Transient
        || block.system_kind() != DifferentiableSystemKind::CoupledBlock
        || eigen_value.system_kind() != DifferentiableSystemKind::Eigen
        || eigen_vector.system_kind() != DifferentiableSystemKind::Eigen
    {
        return Err("differentiable family kind contract failed".to_string());
    }

    Ok(vec![
        ("transient_forward_valid", 1.0),
        ("transient_adjoint_valid", 1.0),
        (
            "transient_replay_evaluations",
            transient_replay.evaluations as f64,
        ),
        (
            "transient_dimension_rejection",
            f64::from(transient_dimension_rejection),
        ),
        ("block_forward_valid", 1.0),
        ("cancellation_rejections", cancellation_rejections as f64),
        (
            "mid_iteration_cancellation",
            f64::from(mid_iteration_cancellation),
        ),
        (
            "block_mid_iteration_cancellation",
            f64::from(block_mid_iteration_cancellation),
        ),
        (
            "steady_retained_state_rejection",
            f64::from(steady_retained_state_rejected),
        ),
        (
            "generated_output_rejections",
            generated_output_rejections as f64,
        ),
        ("block_adjoint_valid", 1.0),
        ("block_replay_evaluations", block_replay.evaluations as f64),
        (
            "block_dimension_rejection",
            f64::from(block_dimension_rejection),
        ),
        ("eigen_value_error", (eigen_value_forward - 4.0).abs()),
        ("eigen_adjoint_error", (eigen_value_adjoint - 4.5).abs()),
        (
            "eigen_replay_evaluations",
            eigen_value_replay.evaluations as f64,
        ),
        (
            "eigen_vector_error",
            eigen_vector_forward
                .iter()
                .zip([2.0, 3.0])
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max),
        ),
        (
            "eigen_vector_adjoint_error",
            eigen_vector_adjoint
                .iter()
                .zip([2.5, 2.75])
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max),
        ),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference(
                    "deterministic transient/eigen/coupled differentiable lifecycle callbacks",
                )
                .input_checksum("analytic/deterministic-fixture:differentiable-lifecycle-v1")
                .reference_checksum("analytic/deterministic-fixture:differentiable-lifecycle-v1")
                .dimension("transient state 2; coupled block 2x1; eigen scalar/vector")
                .tolerance(
        "all error metrics <= 1e-12; replay evaluations equal two; cancellation rejections equal four; generated-output/result-contract rejections equal nine; steady retained-state rejection equals one; dimension rejections equal one",
                )
                .seed("solvers-differentiable-lifecycle:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                if name.contains("error") {
                    result = result.maximum(name, TOLERANCE);
                }
                result = result.metric(name, value);
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result =
                QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            std::process::exit(1);
        }
    }
}
