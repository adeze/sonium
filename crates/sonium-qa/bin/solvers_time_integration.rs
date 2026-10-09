//! Release-facing contracts for checked time-integration entry points.

use num_traits::Float;
use sonium_qa::QaResult;
use sonium_solvers::time_integration::{
    ExplicitConfig, GeneralizedAlphaConfig, NewmarkConfig, NewmarkInitial, NewmarkResult,
    newmark_adjoint, try_explicit_integrate, try_generalized_alpha_integrate,
    try_newmark_integrate,
};
use sonium_solvers::{
    AccuracyStatus, AdjointCapability, LinearOperator, SolverError, SolverResultContract,
    SolverStatus,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.time-integration-contracts.v1";

#[derive(Debug, Clone, Copy)]
struct ScalarOperator {
    value: f64,
}

impl LinearOperator<f64> for ScalarOperator {
    fn num_rows(&self) -> usize {
        1
    }

    fn num_cols(&self) -> usize {
        1
    }

    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        y[0] = self.value * x[0];
    }

    fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
        y[0] = self.value * x[0];
    }

    fn apply_hermitian(&self, x: &[f64]) -> Vec<f64> {
        vec![self.value * x[0]]
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

fn assert_finite_metadata<T: sonium_solvers::ComplexField>(
    result: &sonium_solvers::NewmarkResult<T>,
) -> Result<(), String> {
    if result.solver_status() != SolverStatus::Converged {
        return Err(format!(
            "time integration did not report Converged: {:?}",
            result.solver_status()
        ));
    }
    if result.accuracy_status() != AccuracyStatus::NotChecked {
        return Err(format!(
            "time integration accuracy status changed unexpectedly: {:?}",
            result.accuracy_status()
        ));
    }
    SolverResultContract::validate_contract(result)
        .map_err(|error| format!("shared result contract rejected trajectory: {error}"))?;
    if !result.final_time.is_finite()
        || result
            .displacement
            .iter()
            .chain(result.velocity.iter())
            .chain(result.acceleration.iter())
            .any(|value| !value.re().is_finite() || !value.im().is_finite())
    {
        return Err("time integration returned a non-finite state".to_string());
    }
    Ok(())
}

fn checkpoint_round_trip_contract() -> Result<(f64, f64), String> {
    let stiffness = ScalarOperator { value: 1.0 };
    let initial = NewmarkInitial {
        displacement: vec![1.0],
        velocity: vec![0.0],
        acceleration: vec![-1.0],
    };
    let generalized = try_generalized_alpha_integrate(
        &stiffness,
        None::<&ScalarOperator>,
        &stiffness,
        |_time| vec![0.0],
        &initial,
        |rhs| rhs.to_vec(),
        &GeneralizedAlphaConfig::from_spectral_radius(1.0, 0.01, 3).with_history(),
    )
    .map_err(|error| format!("generalized-alpha checkpoint fixture failed: {error}"))?;
    let explicit = try_explicit_integrate(
        |_u, _v, _time| vec![0.0],
        |rhs| rhs.to_vec(),
        &initial,
        &ExplicitConfig::new(0.01, 3).with_history(),
    )
    .map_err(|error| format!("explicit checkpoint fixture failed: {error}"))?;
    let mut max_packet_bytes = 0.0_f64;
    for (name, result) in [("generalized-alpha", &generalized), ("explicit", &explicit)] {
        let encoded = result
            .to_checkpoint_bytes()
            .map_err(|error| format!("{name} checkpoint encode failed: {error}"))?;
        let restored = NewmarkResult::from_checkpoint_bytes(&encoded)
            .map_err(|error| format!("{name} checkpoint decode failed: {error}"))?;
        if restored.displacement != result.displacement
            || restored.velocity != result.velocity
            || restored.acceleration != result.acceleration
            || restored.displacement_history != result.displacement_history
            || restored.velocity_history != result.velocity_history
            || restored.acceleration_history != result.acceleration_history
            || restored.steps_completed != result.steps_completed
            || restored.final_time != result.final_time
            || restored.status() != result.status()
        {
            return Err(format!("{name} checkpoint changed trajectory or status"));
        }
        SolverResultContract::validate_contract(&restored)
            .map_err(|error| format!("{name} restored result contract failed: {error}"))?;
        max_packet_bytes = max_packet_bytes.max(encoded.len() as f64);
    }
    Ok((2.0, max_packet_bytes))
}

fn adjoint_status_contract() -> Result<(f64, f64), String> {
    let stiffness = ScalarOperator { value: 1.0 };
    let config = NewmarkConfig::average_acceleration(0.01, 2).with_history();
    let initial = NewmarkInitial {
        displacement: vec![1.0],
        velocity: vec![0.0],
        acceleration: vec![-1.0],
    };
    let forward = try_newmark_integrate(
        None::<&ScalarOperator>,
        &stiffness,
        |_| vec![0.0],
        &initial,
        |rhs| rhs.to_vec(),
        &config,
    )
    .map_err(|error| format!("adjoint forward solve failed: {error}"))?;
    let adjoint = newmark_adjoint(
        None::<&ScalarOperator>,
        &stiffness,
        &forward,
        &[0.0],
        &[0.0],
        &[0.0],
        |rhs| Ok(rhs.to_vec()),
        &config,
    )
    .map_err(|error| format!("adjoint solve failed: {error}"))?;
    if adjoint.status() != SolverStatus::Converged
        || adjoint.accuracy_status() != AccuracyStatus::NotChecked
        || !adjoint.is_finite()
    {
        return Err(format!(
            "finite adjoint result had unexpected metadata: status={:?}, accuracy={:?}",
            adjoint.status(),
            adjoint.accuracy_status()
        ));
    }

    let mut corrupted = adjoint.clone();
    corrupted.force_gradient[0][0] = f64::NAN;
    if corrupted.status() != SolverStatus::NumericalError
        || corrupted.accuracy_status() != AccuracyStatus::NotAttained
    {
        return Err("non-finite adjoint result was not classified as NumericalError".to_string());
    }
    Ok((1.0, 1.0))
}

fn analytical_sdof_contracts() -> Result<(f64, f64), String> {
    let dt = 0.01;
    let steps = 100;
    let stiffness = ScalarOperator { value: 1.0 };
    let damping = ScalarOperator { value: 0.0 };
    let initial = NewmarkInitial {
        displacement: vec![1.0],
        velocity: vec![0.0],
        acceleration: vec![-1.0],
    };

    let newmark_config = NewmarkConfig::average_acceleration(dt, steps).with_history();
    let effective = 1.0 + newmark_config.beta * dt * dt;
    let newmark = try_newmark_integrate(
        Some(&damping),
        &stiffness,
        |_| vec![0.0],
        &initial,
        move |rhs| vec![rhs[0] / effective],
        &newmark_config,
    )
    .map_err(|error| format!("checked Newmark integration failed: {error}"))?;
    assert_finite_metadata(&newmark)?;

    let time = steps as f64 * dt;
    let newmark_error = (newmark.displacement[0] - time.cos()).abs();
    if newmark_error > 2.0e-4 {
        return Err(format!(
            "Newmark SDOF displacement error {newmark_error:.3e} exceeds 2e-4"
        ));
    }
    if newmark.steps_completed != steps
        || newmark.displacement_history.as_ref().map(Vec::len) != Some(steps + 1)
    {
        return Err("Newmark history/step count contract failed".to_string());
    }

    let explicit_config = ExplicitConfig::new(dt, steps).with_history();
    let explicit = try_explicit_integrate(
        |u, _v, _t| vec![-u[0]],
        |rhs| rhs.to_vec(),
        &initial,
        &explicit_config,
    )
    .map_err(|error| format!("checked explicit integration failed: {error}"))?;
    assert_finite_metadata(&explicit)?;

    let explicit_error = (explicit.displacement[0] - time.cos()).abs();
    if explicit_error > 2.0e-4 {
        return Err(format!(
            "explicit SDOF displacement error {explicit_error:.3e} exceeds 2e-4"
        ));
    }
    if explicit.steps_completed != steps
        || explicit.velocity_history.as_ref().map(Vec::len) != Some(steps + 1)
    {
        return Err("explicit history/step count contract failed".to_string());
    }

    Ok((newmark_error, explicit_error))
}

fn checked_failure_contracts() -> Result<(f64, f64, f64, f64, f64), String> {
    let invalid_explicit = ExplicitConfig::new(0.0, 1).validate();
    if !matches!(invalid_explicit, Err(SolverError::OperationFailed { .. })) {
        return Err("invalid explicit timestep was not rejected".to_string());
    }

    let invalid_newmark = NewmarkConfig {
        beta: 0.25,
        gamma: 0.5,
        dt: f64::NAN,
        num_steps: 1,
        store_history: false,
    }
    .validate();
    if !matches!(invalid_newmark, Err(SolverError::OperationFailed { .. })) {
        return Err("invalid Newmark timestep was not rejected".to_string());
    }

    let initial = NewmarkInitial {
        displacement: vec![1.0],
        velocity: vec![0.0, 0.0],
        acceleration: vec![0.0],
    };
    let dimension_error = try_explicit_integrate(
        |_u, _v, _t| vec![0.0],
        |rhs| rhs.to_vec(),
        &initial,
        &ExplicitConfig::new(0.01, 1),
    );
    if !matches!(dimension_error, Err(SolverError::DimensionMismatch { .. })) {
        return Err("explicit displacement/velocity mismatch was not rejected".to_string());
    }

    let panic_initial = NewmarkInitial {
        displacement: vec![1.0],
        velocity: vec![0.0],
        acceleration: vec![-1.0],
    };
    let panic_result = try_explicit_integrate(
        |_u, _v, _t| panic!("deliberate QA callback panic"),
        |rhs| rhs.to_vec(),
        &panic_initial,
        &ExplicitConfig::new(0.01, 1),
    );
    if !matches!(panic_result, Err(SolverError::OperationFailed { .. })) {
        return Err("explicit callback panic was not converted to a typed error".to_string());
    }

    let generated_dimension = try_explicit_integrate(
        |_u: &[f64], _v: &[f64], _t| Vec::new(),
        |rhs: &[f64]| rhs.to_vec(),
        &panic_initial,
        &ExplicitConfig::new(0.01, 1),
    );
    if !matches!(
        generated_dimension,
        Err(SolverError::DimensionMismatch {
            expected: 1,
            got: 0
        })
    ) {
        return Err("generated force dimension mismatch was not typed".to_string());
    }

    let generated_nonfinite = try_explicit_integrate(
        |_u: &[f64], _v: &[f64], _t| vec![0.0],
        |_rhs: &[f64]| vec![f64::NAN],
        &panic_initial,
        &ExplicitConfig::new(0.01, 1),
    );
    if !matches!(
        generated_nonfinite,
        Err(SolverError::OperationFailed { .. })
    ) {
        return Err("generated mass non-finite output was not typed".to_string());
    }

    let stiffness = ScalarOperator { value: 1.0 };
    let implicit_initial = NewmarkInitial {
        displacement: vec![1.0],
        velocity: vec![0.0],
        acceleration: vec![-1.0],
    };
    let newmark_config = NewmarkConfig::average_acceleration(0.01, 1);
    let newmark_force_dimension = try_newmark_integrate(
        None::<&ScalarOperator>,
        &stiffness,
        |_time| Vec::new(),
        &implicit_initial,
        |rhs| rhs.to_vec(),
        &newmark_config,
    );
    if !matches!(
        newmark_force_dimension,
        Err(SolverError::DimensionMismatch {
            expected: 1,
            got: 0
        })
    ) {
        return Err("Newmark force dimension mismatch was not typed".to_string());
    }
    let newmark_solve_nonfinite = try_newmark_integrate(
        None::<&ScalarOperator>,
        &stiffness,
        |_time| vec![0.0],
        &implicit_initial,
        |_rhs| vec![f64::NAN],
        &newmark_config,
    );
    if !matches!(
        newmark_solve_nonfinite,
        Err(SolverError::OperationFailed {
            operation: "Newmark integration linear-solve callback",
            ..
        })
    ) {
        return Err("Newmark linear-solve non-finite output was not typed".to_string());
    }
    let generalized_config = GeneralizedAlphaConfig::from_spectral_radius(1.0, 0.01, 1);
    let generalized_force_nonfinite = try_generalized_alpha_integrate(
        &stiffness,
        None::<&ScalarOperator>,
        &stiffness,
        |_time| vec![f64::INFINITY],
        &implicit_initial,
        |rhs| rhs.to_vec(),
        &generalized_config,
    );
    if !matches!(
        generalized_force_nonfinite,
        Err(SolverError::OperationFailed {
            operation: "generalized-alpha integration force callback",
            ..
        })
    ) {
        return Err("generalized-alpha force non-finite output was not typed".to_string());
    }
    let generalized_solve_dimension = try_generalized_alpha_integrate(
        &stiffness,
        None::<&ScalarOperator>,
        &stiffness,
        |_time| vec![0.0],
        &implicit_initial,
        |_rhs| Vec::new(),
        &generalized_config,
    );
    if !matches!(
        generalized_solve_dimension,
        Err(SolverError::DimensionMismatch {
            expected: 1,
            got: 0
        })
    ) {
        return Err("generalized-alpha linear-solve dimension mismatch was not typed".to_string());
    }
    Ok((1.0, 1.0, 1.0, 2.0, 4.0))
}

fn result_history_contract() -> Result<f64, String> {
    let base = || NewmarkResult {
        displacement: vec![1.0],
        velocity: vec![0.0],
        acceleration: vec![-1.0],
        displacement_history: Some(vec![vec![1.0], vec![0.99]]),
        velocity_history: Some(vec![vec![0.0], vec![-0.1]]),
        acceleration_history: Some(vec![vec![-1.0], vec![-0.99]]),
        steps_completed: 1,
        final_time: 0.01,
    };

    let mut truncated = base();
    truncated
        .displacement_history
        .as_mut()
        .expect("history present")
        .pop();
    let mut partial = base();
    partial.velocity_history = None;
    let rejected = [truncated, partial]
        .into_iter()
        .filter(|result| {
            result.solver_status() == SolverStatus::NumericalError
                && SolverResultContract::validate_contract(result).is_err()
        })
        .count();
    if rejected != 2 {
        return Err(format!(
            "malformed Newmark histories rejected {rejected}/2 cases"
        ));
    }
    Ok(rejected as f64)
}

fn run() -> Result<
    (
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        u64,
    ),
    String,
> {
    let started = Instant::now();
    let (newmark_error, explicit_error) = analytical_sdof_contracts()?;
    let (adjoint_status, adjoint_failure) = adjoint_status_contract()?;
    let (
        invalid_config,
        dimension_rejections,
        panic_rejections,
        generated_rejections,
        implicit_callback_rejections,
    ) = checked_failure_contracts()?;
    let history_rejections = result_history_contract()?;
    let (checkpoint_round_trip_cases, checkpoint_max_packet_bytes) =
        checkpoint_round_trip_contract()?;
    Ok((
        newmark_error,
        explicit_error,
        adjoint_status,
        adjoint_failure,
        invalid_config,
        dimension_rejections,
        panic_rejections,
        generated_rejections,
        implicit_callback_rejections,
        history_rejections,
        checkpoint_round_trip_cases,
        checkpoint_max_packet_bytes,
        started.elapsed().as_millis() as u64,
    ))
}

fn main() {
    match std::panic::catch_unwind(run) {
        Ok(Ok((
            newmark_error,
            explicit_error,
            adjoint_status,
            adjoint_failure,
            invalid_config,
            dimension_rejections,
            panic_rejections,
            generated_rejections,
            implicit_callback_rejections,
            history_rejections,
            checkpoint_round_trip_cases,
            checkpoint_max_packet_bytes,
            elapsed_ms,
        ))) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("newmark_displacement_error", newmark_error)
                .metric("explicit_displacement_error", explicit_error)
                .metric("adjoint_converged_status", adjoint_status)
                .metric("adjoint_numerical_error_status", adjoint_failure)
                .metric("invalid_config_rejections", invalid_config)
                .metric("dimension_rejections", dimension_rejections)
        .metric("panic_rejections", panic_rejections)
        .metric("generated_rejections", generated_rejections)
            .metric("implicit_callback_rejections", implicit_callback_rejections)
            .metric("history_rejections", history_rejections)
            .metric("checkpoint_round_trip_cases", checkpoint_round_trip_cases)
            .metric("checkpoint_max_packet_bytes", checkpoint_max_packet_bytes)
                .maximum("newmark_displacement_error", 2.0e-4)
                .maximum("explicit_displacement_error", 2.0e-4)
                .minimum("adjoint_converged_status", 1.0)
                .minimum("adjoint_numerical_error_status", 1.0)
                .minimum("invalid_config_rejections", 1.0)
                .minimum("dimension_rejections", 1.0)
        .minimum("panic_rejections", 1.0)
        .minimum("generated_rejections", 2.0)
            .minimum("implicit_callback_rejections", 4.0)
            .minimum("history_rejections", 2.0)
            .minimum("checkpoint_round_trip_cases", 2.0)
            .minimum("checkpoint_max_packet_bytes", 1.0)
                .input_checksum("analytic/deterministic-fixture:time-integration-v1")
                .reference("analytic displacement cos(t) and velocity -sin(t)")
                .reference_checksum("cos(t), sin(t)")
                .dimension("1 DOF; 100 steps at dt=0.01")
                .formulation("analytic unit-frequency harmonic oscillator with forward/adjoint finite metadata, invalid configuration, dimension, callback-panic, generated explicit and implicit force/linear-solve outputs, and malformed-history result contracts")
                .tolerance("Newmark and explicit displacement errors <= 2e-4; finite and non-finite adjoint statuses; typed initial, callback-panic, generated-output, four implicit callback-attribution, and malformed-history rejection gates")
                .seed("solvers-time-integration:sdof:v1")
                .timing_ms("time_integration_suite", elapsed_ms)
                .elapsed_ms(elapsed_ms);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("time-integration result must serialize")
            );
            println!("SUCCESS");
        }
        Ok(Err(error)) => {
            let elapsed_ms = 0;
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .elapsed_ms(elapsed_ms)
                    .to_json_line()
                    .expect("time-integration failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
        Err(_) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, "time-integration QA panicked")
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("time-integration panic result must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
