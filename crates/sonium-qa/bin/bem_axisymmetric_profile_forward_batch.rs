//! Release-backed axisymmetric profile forward batch contract.

use num_complex::Complex64;
use sonium_bem::{AxisymmetricPenetrableProfile, PhysicsParams};
use sonium_qa::{QaResult, QaStatus};
use std::f64::consts::PI;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.axisymmetric-profile-forward-batch.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    dimension: usize,
    amplitude_count: usize,
    maximum_coefficient_error: f64,
    maximum_residual: f64,
    empty_batch_accepted: bool,
    nonfinite_amplitude_rejected: bool,
}

fn profile_fixture() -> AxisymmetricPenetrableProfile {
    let radius = 0.2;
    let count = 8;
    let profile = (0..count)
        .map(|index| {
            let theta = PI * (index as f64 + 0.5) / count as f64;
            [radius * theta.sin(), radius * theta.cos()]
        })
        .collect::<Vec<_>>();
    let normals = (0..count)
        .map(|index| {
            let theta = PI * (index as f64 + 0.5) / count as f64;
            [theta.sin(), theta.cos()]
        })
        .collect::<Vec<_>>();
    AxisymmetricPenetrableProfile::try_new(
        profile,
        normals,
        radius * 0.35,
        radius * (1.0 / 0.65 - 1.0),
        32,
        PhysicsParams::new(420.0, 343.0, 1.21, false),
        PhysicsParams::new(420.0, 240.0, 2.4, true),
    )
    .expect("valid axisymmetric profile fixture")
}

fn run() -> Result<Metrics, String> {
    let problem = profile_fixture();
    let amplitudes = [Complex64::new(1.0, 0.0), Complex64::new(-0.35, 0.2)];
    let batch = problem
        .solve_many(&amplitudes, 1.0e-9)
        .map_err(|error| error.to_string())?;
    if batch.len() != amplitudes.len() {
        return Err(format!(
            "expected {} solutions, got {}",
            amplitudes.len(),
            batch.len()
        ));
    }
    let mut maximum_coefficient_error = 0.0_f64;
    let mut maximum_residual = 0.0_f64;
    for (index, &amplitude) in amplitudes.iter().enumerate() {
        let scalar = problem
            .solve(amplitude, 1.0e-9)
            .map_err(|error| error.to_string())?;
        let exterior_error = batch[index]
            .exterior_coefficients()
            .iter()
            .zip(scalar.exterior_coefficients())
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0, f64::max);
        let interior_error = batch[index]
            .interior_coefficients()
            .iter()
            .zip(scalar.interior_coefficients())
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0, f64::max);
        maximum_coefficient_error =
            maximum_coefficient_error.max(exterior_error.max(interior_error));
        maximum_residual = maximum_residual.max(batch[index].relative_residual());
    }
    let empty_batch_accepted = matches!(
        problem.solve_many(&[], 1.0e-9),
        Ok(solutions) if solutions.is_empty()
    );
    let nonfinite_amplitude_rejected = problem
        .solve_many(&[Complex64::new(f64::NAN, 0.0)], 1.0e-9)
        .is_err();
    if !empty_batch_accepted || !nonfinite_amplitude_rejected {
        return Err("profile batch admission contract failed".to_string());
    }
    if !maximum_coefficient_error.is_finite() || !maximum_residual.is_finite() {
        return Err("profile batch metrics are non-finite".to_string());
    }
    Ok(Metrics {
        dimension: problem.num_dofs(),
        amplitude_count: amplitudes.len(),
        maximum_coefficient_error,
        maximum_residual,
        empty_batch_accepted,
        nonfinite_amplitude_rejected,
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .metric("dimension", metrics.dimension as f64)
            .metric("amplitude_count", metrics.amplitude_count as f64)
            .metric(
                "maximum_coefficient_error",
                metrics.maximum_coefficient_error,
            )
            .metric("maximum_residual", metrics.maximum_residual)
            .metric(
                "empty_batch_accepted",
                if metrics.empty_batch_accepted {
                    1.0
                } else {
                    0.0
                },
            )
            .metric(
                "nonfinite_amplitude_rejected",
                if metrics.nonfinite_amplitude_rejected {
                    1.0
                } else {
                    0.0
                },
            )
            .minimum("dimension", 1.0)
            .minimum("amplitude_count", 2.0)
            .minimum("empty_batch_accepted", 1.0)
            .minimum("nonfinite_amplitude_rejected", 1.0)
            .maximum("maximum_coefficient_error", 1.0e-10)
            .maximum("maximum_residual", 1.0e-9)
            .input_checksum(
                "analytic/deterministic-fixture:bem-axisymmetric-profile-forward-batch-v1",
            )
            .reference("independent scalar axisymmetric profile solves with the same operator")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-axisymmetric-profile-forward-batch-v1",
            )
            .dimension("axisymmetric general profile; two complex plane-wave amplitudes")
            .formulation("production AxisymmetricPenetrableProfile shared-LU forward batch")
            .tolerance(
                "coefficient parity <= 1e-10; residual <= 1e-9; empty/non-finite admission checked",
            )
            .seed("bem-axisymmetric-profile-forward-batch:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(_) => QaResult::failed(
            CASE_ID,
            "panic while executing axisymmetric profile forward batch contract",
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
