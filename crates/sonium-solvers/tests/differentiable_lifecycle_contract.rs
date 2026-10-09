use num_complex::Complex64;
use sonium_solvers::iterative::{BlockFgmresSolution, KrylovOpCounts};
use sonium_solvers::time_integration::{NewmarkAdjointResult, NewmarkResult};
use sonium_solvers::{
    CancellationToken, DifferentiableSolveContract, DifferentiableSystemKind, SolverError,
    TaylorContractReport, block_contract, check_taylor_remainder, eigen_value_contract,
    eigen_vector_contract, transient_contract, verify_forward_checkpoint_replay,
    verify_forward_replay,
};

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
        attainable_accuracy: vec![sonium_solvers::AccuracyStatus::Attained],
        converged: true,
        op_counts: KrylovOpCounts::default(),
    }
}

#[test]
fn transient_contract_rejects_adjoint_state_dimension_mismatch() {
    let contract = transient_contract(
        |_input: &[f64]| Ok(transient_state(2)),
        |_forward: &NewmarkResult<f64>, _seed: &[f64]| Ok(transient_adjoint(3)),
    );
    let forward = contract
        .solve_forward(&[1.0, 2.0])
        .expect("finite transient forward state");
    let error = contract
        .solve_adjoint(&forward, &[0.0, 0.0])
        .expect_err("adjoint state must match forward state dimension");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
    assert!(
        error
            .to_string()
            .contains("dimension mismatch with forward state")
    );
}

#[test]
fn block_contract_rejects_seed_and_adjoint_dimensions_independently() {
    let seed_contract = block_contract(
        |_input: &[f64]| Ok(block_solution(2)),
        |_forward: &BlockFgmresSolution<f64>, _seed: &[f64]| Ok(vec![0.0; 2]),
    );
    let forward = seed_contract
        .solve_forward(&[1.0, 2.0])
        .expect("finite block forward state");
    let error = seed_contract
        .solve_adjoint(&forward, &[0.0, 0.0, 0.0])
        .expect_err("block adjoint seed must match forward rows");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
    assert!(error.to_string().contains("adjoint seed"));

    let adjoint_contract = block_contract(
        |_input: &[f64]| Ok(block_solution(2)),
        |_forward: &BlockFgmresSolution<f64>, _seed: &[f64]| Ok(vec![0.0; 3]),
    );
    let forward = adjoint_contract
        .solve_forward(&[1.0, 2.0])
        .expect("finite block forward state");
    let error = adjoint_contract
        .solve_adjoint(&forward, &[0.0, 0.0])
        .expect_err("block adjoint state must match forward rows");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
    assert!(error.to_string().contains("adjoint state"));
}

#[test]
fn scalar_eigen_contract_preserves_family_and_valid_lifecycle() {
    let contract = eigen_value_contract(
        |input: &[f64]| Ok(input[0] * 2.0),
        |forward: &f64, seed: &[f64]| Ok(*forward + seed[0]),
    );
    let forward = contract
        .solve_forward(&[2.0])
        .expect("finite eigenvalue state");
    let adjoint = contract
        .solve_adjoint(&forward, &[0.5])
        .expect("finite eigenvalue adjoint state");
    assert_eq!(contract.system_kind(), DifferentiableSystemKind::Eigen);
    assert_eq!(forward, 4.0);
    assert_eq!(adjoint, 4.5);
}

#[test]
fn taylor_contract_reports_second_order_real_and_complex_remainders() {
    let real = check_taylor_remainder(
        &[1.0_f64, -0.5],
        &[0.75_f64, 0.25],
        &[0.2, 0.1, 0.05],
        |point| Ok(vec![point[0] * point[0] + 2.0 * point[1]]),
        |base, direction| Ok(vec![2.0 * base[0] * direction[0] + 2.0 * direction[1]]),
    )
    .expect("real Taylor contract should evaluate");
    assert!(
        real.minimum_observed_order()
            .is_some_and(|order| order > 1.99)
    );
    real.require_order(1.9)
        .expect("real quadratic remainder should be second order");

    let complex = check_taylor_remainder(
        &[Complex64::new(1.0, -0.25)],
        &[Complex64::new(0.5, 0.75)],
        &[0.2, 0.1, 0.05],
        |point| Ok(vec![point[0] * point[0]]),
        |base, direction| Ok(vec![Complex64::new(2.0, 0.0) * base[0] * direction[0]]),
    )
    .expect("complex Taylor contract should evaluate");
    complex
        .require_order(1.9)
        .expect("complex quadratic remainder should be second order");
}

#[test]
fn taylor_contract_accepts_exactly_cancelling_remainders() {
    let report = check_taylor_remainder(
        &[1.0_f64, -0.5],
        &[1.0_f64, 0.0],
        &[0.5, 0.25, 0.125],
        |point| Ok(vec![point[0]]),
        |_base, direction| Ok(vec![direction[0]]),
    )
    .expect("exact affine Taylor contract should evaluate");

    assert!(report.remainders.iter().all(|remainder| *remainder == 0.0));
    assert!(
        report
            .observed_orders
            .iter()
            .all(|order| order.is_infinite() && *order > 0.0)
    );
    report
        .require_order(10.0)
        .expect("positive-infinite order satisfies every finite requirement");

    let malformed = TaylorContractReport {
        steps: vec![0.2, 0.1],
        remainders: vec![0.0, 0.0],
        observed_orders: vec![f64::NAN],
    };
    assert!(malformed.require_order(1.0).is_err());
}

#[test]
fn differentiable_forward_replay_rejects_nondeterministic_state() {
    let calls = std::cell::Cell::new(0_u32);
    let contract = eigen_value_contract(
        move |_input: &[f64]| {
            calls.set(calls.get() + 1);
            Ok(f64::from(calls.get()))
        },
        |forward: &f64, _seed: &[f64]| Ok(*forward),
    );
    let error = verify_forward_replay(&contract, &[0.0], |first, replay| first == replay)
        .expect_err("nondeterministic forward state must fail replay contract");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "differentiable replay",
            ..
        }
    ));
}

#[test]
fn differentiable_forward_replay_accepts_deterministic_complex_state() {
    let contract = eigen_value_contract(
        |input: &[Complex64]| Ok(input[0] * input[0]),
        |forward: &Complex64, _seed: &[Complex64]| Ok(*forward),
    );
    let report =
        verify_forward_replay(&contract, &[Complex64::new(0.5, -0.25)], |first, replay| {
            first == replay
        })
        .expect("deterministic complex forward state should replay");
    assert_eq!(report.evaluations, 2);
    assert!(report.deterministic);
}

#[test]
fn native_transient_and_block_contracts_verify_replay_equivalence() {
    let transient = transient_contract(
        |_input: &[f64]| Ok(transient_state(2)),
        |_forward: &NewmarkResult<f64>, _seed: &[f64]| Ok(transient_adjoint(2)),
    );
    let transient_report = verify_forward_replay(
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
    .expect("transient replay should be deterministic");
    assert_eq!(transient_report.evaluations, 2);
    assert!(transient_report.deterministic);

    let block = block_contract(
        |_input: &[f64]| Ok(block_solution(2)),
        |_forward: &BlockFgmresSolution<f64>, _seed: &[f64]| Ok(vec![0.0; 2]),
    );
    let block_report = verify_forward_replay(
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
    .expect("block replay should be deterministic");
    assert_eq!(block_report.evaluations, 2);
    assert!(block_report.deterministic);
}

#[test]
fn differentiable_taylor_and_replay_boundaries_return_typed_errors() {
    let invalid_steps = check_taylor_remainder(
        &[1.0_f64],
        &[1.0_f64],
        &[0.1, 0.1],
        |point| Ok(point.to_vec()),
        |_base, direction| Ok(direction.to_vec()),
    )
    .expect_err("non-decreasing Taylor steps must be rejected");
    assert!(matches!(
        invalid_steps,
        SolverError::OperationFailed {
            operation: "differentiable Taylor contract",
            ..
        }
    ));

    let malformed = check_taylor_remainder(
        &[1.0_f64],
        &[1.0_f64],
        &[0.1, 0.05],
        |_point| Ok(vec![1.0, 2.0]),
        |_base, _direction| Ok(vec![1.0]),
    )
    .expect_err("malformed Taylor callback output must be rejected");
    assert!(matches!(
        malformed,
        SolverError::OperationFailed {
            operation: "differentiable Taylor contract",
            ..
        }
    ));

    let contract = eigen_value_contract(
        |input: &[f64]| Ok(input[0]),
        |forward: &f64, _seed: &[f64]| Ok(*forward),
    );
    let comparison_panic = verify_forward_replay(&contract, &[1.0], |_first, _second| {
        panic!("comparison failed")
    })
    .expect_err("replay comparator panic must become a typed error");
    assert!(matches!(
        comparison_panic,
        SolverError::OperationFailed {
            operation: "differentiable replay comparison",
            ..
        }
    ));
}

#[test]
fn native_lifecycle_cancellation_rejects_before_forward_and_adjoint_callbacks() {
    let transient = transient_contract(
        |_input: &[f64]| Ok(transient_state(2)),
        |_forward: &NewmarkResult<f64>, _seed: &[f64]| Ok(transient_adjoint(2)),
    );
    let forward = transient.solve_forward(&[1.0, 2.0]).unwrap();
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let error = transient
        .solve_forward_cancellable(&[1.0, 2.0], &cancellation)
        .expect_err("cancelled transient forward should fail before callback");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
    assert!(error.to_string().contains("cancelled"));
    let error = transient
        .solve_adjoint_cancellable(&forward, &[0.0, 0.0], &cancellation)
        .expect_err("cancelled transient adjoint should fail before callback");
    assert!(matches!(error, SolverError::OperationFailed { .. }));

    let block = block_contract(
        |_input: &[f64]| Ok(block_solution(2)),
        |_forward: &BlockFgmresSolution<f64>, _seed: &[f64]| Ok(vec![0.0; 2]),
    );
    let block_forward = block.solve_forward(&[1.0, 2.0]).unwrap();
    let error = block
        .solve_adjoint_cancellable(&block_forward, &[0.0, 0.0], &cancellation)
        .expect_err("cancelled block adjoint should fail before callback");
    assert!(matches!(error, SolverError::OperationFailed { .. }));

    let eigen = eigen_value_contract(
        |input: &[f64]| Ok(input[0] * 2.0),
        |forward: &f64, seed: &[f64]| Ok(*forward + seed[0]),
    );
    let error = eigen
        .solve_forward_cancellable(&[2.0], &cancellation)
        .expect_err("cancelled eigen forward should fail before callback");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn cancellation_is_checked_after_callback_returns() {
    let cancellation = CancellationToken::new();
    let callback_token = cancellation.clone();
    let contract = eigen_value_contract(
        move |input: &[f64]| {
            callback_token.cancel();
            Ok(input[0] * 2.0)
        },
        |_forward: &f64, seed: &[f64]| Ok(seed[0]),
    );
    let error = contract
        .solve_forward_cancellable(&[2.0], &cancellation)
        .expect_err("post-callback cancellation should be reported");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
    assert!(error.to_string().contains("cancelled"));
}

#[test]
fn checkpoint_replay_covers_transient_eigen_and_block_contracts() {
    let token = CancellationToken::new();

    let transient = transient_contract(
        |_input: &[f64]| Ok::<_, SolverError>(transient_state(2)),
        |_forward: &NewmarkResult<f64>, _seed: &[f64]| Ok::<_, SolverError>(transient_adjoint(2)),
    );
    let transient_report = verify_forward_checkpoint_replay(
        &transient,
        &[1.0, 2.0],
        |state| Ok::<_, SolverError>(state.steps_completed.to_le_bytes().to_vec()),
        |payload| {
            if payload == 1_u64.to_le_bytes() {
                Ok(transient_state(2))
            } else {
                Err(SolverError::OperationFailed {
                    operation: "test transient checkpoint",
                    reason: "unexpected payload".into(),
                })
            }
        },
        |left, right| {
            left.displacement == right.displacement
                && left.velocity == right.velocity
                && left.acceleration == right.acceleration
                && left.steps_completed == right.steps_completed
        },
        &token,
    )
    .expect("transient checkpoint replay");
    assert_eq!(transient_report.evaluations, 2);
    assert_eq!(transient_report.checkpoint_bytes, 8);
    assert!(transient_report.deterministic);

    let eigen = eigen_value_contract(
        |_input: &[f64]| Ok::<_, SolverError>(2.5),
        |_forward: &f64, _seed: &[f64]| Ok::<_, SolverError>(1.0),
    );
    let eigen_report = verify_forward_checkpoint_replay(
        &eigen,
        &[0.5],
        |value| Ok::<_, SolverError>(value.to_le_bytes().to_vec()),
        |payload| {
            let bytes: [u8; 8] = payload
                .try_into()
                .map_err(|_| SolverError::OperationFailed {
                    operation: "test eigen checkpoint",
                    reason: "invalid payload length".into(),
                })?;
            Ok(f64::from_le_bytes(bytes))
        },
        |left, right| left == right,
        &token,
    )
    .expect("eigen checkpoint replay");
    assert_eq!(eigen_report.checkpoint_bytes, 8);

    let block = block_contract(
        |_input: &[f64]| Ok::<_, SolverError>(block_solution(2)),
        |_forward: &BlockFgmresSolution<f64>, _seed: &[f64]| Ok::<_, SolverError>(vec![0.0; 2]),
    );
    let block_report = verify_forward_checkpoint_replay(
        &block,
        &[1.0, 2.0],
        |_state| Ok::<_, SolverError>(vec![0x42, 0x17]),
        |_payload| Ok(block_solution(2)),
        |left, right| {
            left.x.as_slice() == right.x.as_slice()
                && left.iterations == right.iterations
                && left.converged == right.converged
        },
        &token,
    )
    .expect("block checkpoint replay");
    assert_eq!(block_report.checkpoint_bytes, 2);
}

#[test]
fn checkpoint_replay_rejects_empty_mismatched_and_cancelled_state() {
    let contract = eigen_value_contract(
        |_input: &[f64]| Ok::<_, SolverError>(1.0),
        |_forward: &f64, _seed: &[f64]| Ok::<_, SolverError>(0.0),
    );
    let active = CancellationToken::new();
    let empty = verify_forward_checkpoint_replay(
        &contract,
        &[0.0],
        |_value| Ok::<_, SolverError>(Vec::new()),
        |_payload| Ok(1.0),
        |left, right| left == right,
        &active,
    )
    .expect_err("empty checkpoint payload must fail");
    assert!(empty.to_string().contains("empty payload"));

    let mismatch = verify_forward_checkpoint_replay(
        &contract,
        &[0.0],
        |_value| Ok::<_, SolverError>(vec![1]),
        |_payload| Ok(2.0),
        |left, right| left == right,
        &active,
    )
    .expect_err("mismatched checkpoint state must fail");
    assert!(mismatch.to_string().contains("differs from original"));

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let error = verify_forward_checkpoint_replay(
        &contract,
        &[0.0],
        |_value| Ok::<_, SolverError>(vec![1]),
        |_payload| Ok(1.0),
        |left, right| left == right,
        &cancelled,
    )
    .expect_err("cancelled checkpoint must fail before callbacks");
    assert!(error.to_string().contains("cancelled"));
}

#[test]
fn differentiable_lifecycle_rejects_nonfinite_generated_outputs() {
    let transient_forward = transient_contract(
        |_input: &[f64]| {
            let mut state = transient_state(2);
            state.displacement[0] = f64::NAN;
            Ok::<_, SolverError>(state)
        },
        |_forward: &NewmarkResult<f64>, _seed: &[f64]| Ok(transient_adjoint(2)),
    );
    assert!(transient_forward.solve_forward(&[1.0, 2.0]).is_err());

    let transient_adjoint_contract = transient_contract(
        |_input: &[f64]| Ok::<_, SolverError>(transient_state(2)),
        |_forward: &NewmarkResult<f64>, _seed: &[f64]| {
            let mut adjoint = transient_adjoint(2);
            adjoint.initial_velocity_gradient[0] = f64::INFINITY;
            Ok::<_, SolverError>(adjoint)
        },
    );
    let transient_state_value = transient_adjoint_contract
        .solve_forward(&[1.0, 2.0])
        .expect("finite transient state");
    assert!(
        transient_adjoint_contract
            .solve_adjoint(&transient_state_value, &[0.25, -0.5])
            .is_err()
    );

    let eigen_value_forward = eigen_value_contract(
        |_input: &[f64]| Ok::<_, SolverError>(f64::NAN),
        |_forward: &f64, _seed: &[f64]| Ok::<_, SolverError>(0.0),
    );
    assert!(eigen_value_forward.solve_forward(&[1.0]).is_err());

    let eigen_value_adjoint = eigen_value_contract(
        |_input: &[f64]| Ok::<_, SolverError>(1.0),
        |_forward: &f64, _seed: &[f64]| Ok::<_, SolverError>(f64::INFINITY),
    );
    let eigen_value_state = eigen_value_adjoint
        .solve_forward(&[1.0])
        .expect("finite eigenvalue state");
    assert!(
        eigen_value_adjoint
            .solve_adjoint(&eigen_value_state, &[0.5])
            .is_err()
    );

    let eigen_vector_forward = eigen_vector_contract(
        |_input: &[f64]| Ok::<_, SolverError>(vec![f64::NAN, 1.0]),
        |_forward: &Vec<f64>, _seed: &[f64]| Ok::<_, SolverError>(vec![0.0, 0.0]),
    );
    assert!(eigen_vector_forward.solve_forward(&[1.0, 2.0]).is_err());

    let eigen_vector_adjoint = eigen_vector_contract(
        |_input: &[f64]| Ok::<_, SolverError>(vec![1.0, 2.0]),
        |_forward: &Vec<f64>, _seed: &[f64]| Ok::<_, SolverError>(vec![0.0, f64::INFINITY]),
    );
    let eigen_vector_state = eigen_vector_adjoint
        .solve_forward(&[1.0, 2.0])
        .expect("finite eigenvector state");
    assert!(
        eigen_vector_adjoint
            .solve_adjoint(&eigen_vector_state, &[0.5, -0.25])
            .is_err()
    );

    let block_forward = block_contract(
        |_input: &[f64]| {
            let mut state = block_solution(2);
            state.x[[0, 0]] = f64::NAN;
            Ok::<_, SolverError>(state)
        },
        |_forward: &BlockFgmresSolution<f64>, _seed: &[f64]| Ok::<_, SolverError>(vec![0.0; 2]),
    );
    assert!(block_forward.solve_forward(&[1.0, 2.0]).is_err());

    let block_adjoint = block_contract(
        |_input: &[f64]| Ok::<_, SolverError>(block_solution(2)),
        |_forward: &BlockFgmresSolution<f64>, _seed: &[f64]| {
            Ok::<_, SolverError>(vec![f64::INFINITY, 0.0])
        },
    );
    let block_state = block_adjoint
        .solve_forward(&[1.0, 2.0])
        .expect("finite block state");
    assert!(
        block_adjoint
            .solve_adjoint(&block_state, &[0.25, -0.5])
            .is_err()
    );
}

#[test]
fn differentiable_lifecycle_rejects_inconsistent_iterative_result_metadata() {
    let contract = block_contract(
        |_input: &[f64]| {
            let mut state = block_solution(2);
            state.attainable_accuracy[0] = sonium_solvers::AccuracyStatus::NotAttained;
            Ok::<_, SolverError>(state)
        },
        |_forward: &BlockFgmresSolution<f64>, _seed: &[f64]| -> Result<Vec<f64>, SolverError> {
            panic!("adjoint callback must not run for malformed block result")
        },
    );
    let error = contract
        .solve_forward(&[1.0, 2.0])
        .expect_err("inconsistent block result metadata must be rejected");
    assert!(error.to_string().contains("iterative result contract"));
}
