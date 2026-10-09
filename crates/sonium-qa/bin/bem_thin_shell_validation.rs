//! Release-facing local thin-shell impedance validation.
//!
//! This is a deterministic material-model gate. It exercises the checked
//! mass--spring--damper impedance, analytic frequency derivatives, passivity,
//! and malformed-input admission without launching a simulator workload.

use sonium_bem::core::{BemProblem, BemSolver, BoundaryConditionType, PhysicsParams};
use sonium_bem::materials::{AmbientMedium, ThinShellSurface};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.thin-shell-surface.v1";
const MAX_DERIVATIVE_ERROR: f64 = 2.0e-6;

struct Metrics {
    impedance_derivative_error: f64,
    admittance_derivative_error: f64,
    absorption: f64,
    sweep_error: f64,
    rejection_count: f64,
}

fn relative_error(actual: num_complex::Complex64, expected: num_complex::Complex64) -> f64 {
    (actual - expected).norm() / actual.norm().max(expected.norm()).max(1.0)
}

fn run() -> Result<Metrics, String> {
    let shell = ThinShellSurface::try_new(2.5, 1.8e7, 42.0).map_err(|e| e.to_string())?;
    let frequency = 800.0;
    let step = 0.25;
    let impedance_derivative = shell
        .try_surface_impedance_frequency_derivative(frequency)
        .map_err(|e| e.to_string())?;
    let impedance_centered = (shell
        .try_surface_impedance(frequency + step)
        .map_err(|e| e.to_string())?
        - shell
            .try_surface_impedance(frequency - step)
            .map_err(|e| e.to_string())?)
        / (2.0 * step);
    let impedance_derivative_error = relative_error(impedance_derivative, impedance_centered);

    let admittance_derivative = shell
        .try_admittance_frequency_derivative(frequency)
        .map_err(|e| e.to_string())?;
    let admittance_centered = (shell
        .try_admittance(frequency + step)
        .map_err(|e| e.to_string())?
        - shell
            .try_admittance(frequency - step)
            .map_err(|e| e.to_string())?)
        / (2.0 * step);
    let admittance_derivative_error = relative_error(admittance_derivative, admittance_centered);
    if !impedance_derivative_error.is_finite()
        || impedance_derivative_error > MAX_DERIVATIVE_ERROR
        || !admittance_derivative_error.is_finite()
        || admittance_derivative_error > MAX_DERIVATIVE_ERROR
    {
        return Err(format!(
            "thin-shell derivative errors exceed gate: impedance={impedance_derivative_error:.3e}, admittance={admittance_derivative_error:.3e}"
        ));
    }

    let absorption = shell
        .try_absorption_coefficient(frequency, AmbientMedium::default(), 1.0e-12)
        .map_err(|e| e.to_string())?;
    if !absorption.is_finite() || !(0.0..=1.0).contains(&absorption) {
        return Err(format!("thin-shell absorption outside [0,1]: {absorption}"));
    }

    let mut rejection_count = 0.0;
    if ThinShellSurface::try_new(0.0, 1.0, 1.0).is_err() {
        rejection_count += 1.0;
    }
    if ThinShellSurface::try_new(1.0, -1.0, 1.0).is_err() {
        rejection_count += 1.0;
    }
    if shell.try_admittance(f64::NAN).is_err() {
        rejection_count += 1.0;
    }
    if rejection_count < 3.0 {
        return Err(format!(
            "expected three typed rejections, got {rejection_count}"
        ));
    }

    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 500.0, 343.0, 1.21, 3, 4)
        .with_boundary_condition(BoundaryConditionType::Impedance);
    let solver = BemSolver::new()
        .with_tolerance(1.0e-10)
        .with_thin_shell_surface(shell);
    let sweep = solver
        .solve_frequency_sweep(&problem, &[500.0, 1_000.0])
        .map_err(|error| format!("thin-shell frequency sweep: {error}"))?;
    let expected_admittance = shell
        .try_admittance(1_000.0)
        .map_err(|error| format!("second-frequency admittance: {error}"))?;
    if !sweep[1].elements.iter().all(|element| {
        matches!(
            element.boundary_condition,
            sonium_bem::core::BoundaryCondition::VelocityWithAdmittance { admittance, .. }
                if (admittance - expected_admittance).norm() < 1.0e-12
        )
    }) {
        return Err("thin-shell sweep returned stale element admittance".to_string());
    }
    let mut second_problem = problem.clone();
    second_problem.physics = PhysicsParams::try_new(1_000.0, 343.0, 1.21, false)
        .map_err(|error| format!("second-frequency physics: {error}"))?;
    let reference = solver
        .solve(&second_problem)
        .map_err(|error| format!("thin-shell single-frequency reference: {error}"))?;
    let sweep_error = sweep[1]
        .coefficients()
        .iter()
        .zip(reference.coefficients())
        .map(|(swept, expected)| (*swept - *expected).norm())
        .fold(0.0_f64, f64::max);
    if !sweep_error.is_finite() || sweep_error > 1.0e-6 {
        return Err(format!(
            "thin-shell sweep stale-admittance error {sweep_error:e}"
        ));
    }

    Ok(Metrics {
        impedance_derivative_error,
        admittance_derivative_error,
        absorption,
        sweep_error,
        rejection_count,
    })
}

fn main() {
    let started = Instant::now();
    match std::panic::catch_unwind(run) {
        Ok(Ok(metrics)) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("analytic mass-spring-damper thin-shell impedance")
                .reference_checksum("analytic/deterministic-fixture:bem-thin-shell-v1")
                .input_checksum("analytic/deterministic-fixture:bem-thin-shell-v1")
                .dimension("local normal thin shell; exp(-iωt); Complex64/f64")
                .formulation("checked dimensional impedance/admittance and frequency derivatives")
                .tolerance(
                    "derivative/sweep errors <= 2e-6/1e-6; absorption in [0,1]; three typed rejections",
                )
                .metric(
                    "impedance_derivative_error",
                    metrics.impedance_derivative_error,
                )
                .metric(
                    "admittance_derivative_error",
                    metrics.admittance_derivative_error,
                )
                .metric("absorption", metrics.absorption)
                .metric("sweep_error", metrics.sweep_error)
                .metric("rejection_count", metrics.rejection_count)
                .maximum("impedance_derivative_error", MAX_DERIVATIVE_ERROR)
                .maximum("admittance_derivative_error", MAX_DERIVATIVE_ERROR)
                .maximum("sweep_error", 1.0e-6)
                .minimum("rejection_count", 3.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("thin-shell QA result serializes")
            );
            println!("SUCCESS");
        }
        Ok(Err(error)) => {
            let record = QaResult::failed(CASE_ID, error);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("thin-shell failure serializes")
            );
            println!("FAILED");
            std::process::exit(1);
        }
        Err(_) => {
            let record = QaResult::failed(CASE_ID, "thin-shell QA panicked");
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("thin-shell panic serializes")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
