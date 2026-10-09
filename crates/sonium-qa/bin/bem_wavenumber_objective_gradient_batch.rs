//! Release-backed BEM multi-frequency wavenumber objective-gradient contract.

use num_complex::Complex64;
use sonium_bem::core::{AssemblyMethod, BemProblem, BemSolver, PhysicsParams, SolverMethod};
use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.wavenumber-objective-gradient-batch.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    dimension: usize,
    frequency_count: usize,
    maximum_batch_scalar_error: f64,
    maximum_gradient_magnitude: f64,
    malformed_shape_rejected: bool,
}

fn at_frequency(problem: &BemProblem, frequency: f64) -> Result<BemProblem, String> {
    let mut candidate = problem.clone();
    let mut physics = PhysicsParams::try_new(
        frequency,
        problem.physics.speed_of_sound,
        problem.physics.density,
        problem.physics.tau < 0.0,
    )
    .map_err(|error| error.to_string())?;
    physics.harmonic_factor = problem.physics.harmonic_factor;
    physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
    physics.validate().map_err(|error| error.to_string())?;
    candidate.physics = physics;
    Ok(candidate)
}

fn make_rhs(dimension: usize, offset: usize) -> Vec<Complex64> {
    (0..dimension)
        .map(|index| {
            Complex64::new(
                0.2 + (index + offset) as f64 / 17.0,
                -0.1 + (index + 2 * offset) as f64 / 23.0,
            )
        })
        .collect()
}

fn run() -> Result<Metrics, String> {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct);
    let dimension = solver
        .solve(&problem)
        .map_err(|error| error.to_string())?
        .num_dofs();
    let frequencies = [100.0, 120.0];
    let forward_rhs_batches = vec![make_rhs(dimension, 0), make_rhs(dimension, 3)];
    let objective_gradient_batches = vec![make_rhs(dimension, 5), make_rhs(dimension, 7)];

    let gradients = solver
        .solve_wavenumber_objective_gradient_many(
            &problem,
            &frequencies,
            &forward_rhs_batches,
            &objective_gradient_batches,
        )
        .map_err(|error| error.to_string())?;
    if gradients.len() != frequencies.len() {
        return Err(format!(
            "expected {} gradients, received {}",
            frequencies.len(),
            gradients.len()
        ));
    }

    let mut maximum_batch_scalar_error = 0.0_f64;
    for (index, &frequency) in frequencies.iter().enumerate() {
        let candidate = at_frequency(&problem, frequency)?;
        let scalar = solver
            .solve_wavenumber_objective_gradient(
                &candidate,
                &forward_rhs_batches[index],
                &objective_gradient_batches[index],
            )
            .map_err(|error| error.to_string())?;
        maximum_batch_scalar_error =
            maximum_batch_scalar_error.max((gradients[index] - scalar).abs());
    }

    let malformed_shape_rejected = solver
        .solve_wavenumber_objective_gradient_many(
            &problem,
            &frequencies,
            std::slice::from_ref(&forward_rhs_batches[0]),
            &objective_gradient_batches,
        )
        .is_err();
    if !malformed_shape_rejected {
        return Err("mismatched frequency/RHS batches were accepted".to_string());
    }

    let maximum_gradient_magnitude = gradients
        .iter()
        .map(|gradient| gradient.abs())
        .fold(0.0, f64::max);
    if !maximum_batch_scalar_error.is_finite() || !maximum_gradient_magnitude.is_finite() {
        return Err("batch objective-gradient metrics are non-finite".to_string());
    }
    Ok(Metrics {
        dimension,
        frequency_count: frequencies.len(),
        maximum_batch_scalar_error,
        maximum_gradient_magnitude,
        malformed_shape_rejected,
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .metric("dimension", metrics.dimension as f64)
            .metric("frequency_count", metrics.frequency_count as f64)
            .metric(
                "maximum_batch_scalar_error",
                metrics.maximum_batch_scalar_error,
            )
            .metric(
                "maximum_gradient_magnitude",
                metrics.maximum_gradient_magnitude,
            )
            .metric(
                "malformed_shape_rejected",
                if metrics.malformed_shape_rejected {
                    1.0
                } else {
                    0.0
                },
            )
            .minimum("dimension", 1.0)
            .minimum("frequency_count", 2.0)
            .minimum("malformed_shape_rejected", 1.0)
            .maximum("maximum_batch_scalar_error", 1.0e-12)
            .input_checksum(
                "analytic/deterministic-fixture:bem-wavenumber-objective-gradient-batch-v1",
            )
            .reference("independent scalar objective-gradient solves at each frequency")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-wavenumber-objective-gradient-batch-v1",
            )
            .dimension("3-D constant-panel TBEM; two frequencies; paired forward and objective RHS")
            .formulation("production BemSolver multi-frequency wavenumber objective-gradient API")
            .tolerance("batch/scalar gradient error <= 1e-12; malformed batch cardinality rejected")
            .seed("bem-wavenumber-objective-gradient-batch:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(_) => QaResult::failed(
            CASE_ID,
            "panic while executing BEM batch objective-gradient contract",
        )
        .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
    };
    println!(
        "QA_RESULT: {}",
        serde_json::to_string(&result).expect("QA JSON serialization")
    );
    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
