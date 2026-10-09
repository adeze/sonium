//! Reference-only QA for an axisymmetric rigid circular-cavity acoustic mode.
//!
//! It checks the closed-form `J0(χ_m r/a) cos(nπz/L)` reference, not a FEM or
//! BEM discretization. Radiation, loss, flow, and non-axisymmetric modes are
//! deliberately outside this evidence record.

use sonium_qa::QaResult;
use sonium_wave::analytical::AxisymmetricRigidCavityMode;

const CASE_ID: &str = "wave.axisymmetric-rigid-cavity-mode.v1";
const TOLERANCE: f64 = 5.0e-12;

struct Metrics {
    helmholtz_residual: f64,
    radial_wall_derivative: f64,
    end_wall_derivative: f64,
}

fn run() -> Result<Metrics, String> {
    let mode = AxisymmetricRigidCavityMode::try_new(0.2, 0.8, 343.0, 1, 2)
        .map_err(|error| format!("construct axisymmetric cavity mode: {error}"))?;
    let mut helmholtz_residual = 0.0_f64;
    let mut radial_wall_derivative = 0.0_f64;
    let mut end_wall_derivative = 0.0_f64;
    for radius_m in [0.0, 0.07, 0.19, 0.2] {
        for axial_position_m in [0.0, 0.13, 0.43, 0.8] {
            helmholtz_residual = helmholtz_residual.max(
                mode.try_helmholtz_residual(radius_m, axial_position_m)
                    .map_err(|error| format!("evaluate Helmholtz residual: {error}"))?
                    .abs(),
            );
            let sample = mode
                .try_evaluate(radius_m, axial_position_m)
                .map_err(|error| format!("evaluate cavity mode: {error}"))?;
            if radius_m == 0.2 {
                radial_wall_derivative = radial_wall_derivative.max(sample.radial_gradient.abs());
            }
            if axial_position_m == 0.0 || axial_position_m == 0.8 {
                end_wall_derivative = end_wall_derivative.max(sample.axial_gradient.abs());
            }
        }
    }
    let metrics = Metrics {
        helmholtz_residual,
        radial_wall_derivative,
        end_wall_derivative,
    };
    for (name, value) in [
        ("Helmholtz residual", metrics.helmholtz_residual),
        (
            "radial rigid-wall derivative",
            metrics.radial_wall_derivative,
        ),
        ("end-wall derivative", metrics.end_wall_derivative),
    ] {
        if !value.is_finite() || value > TOLERANCE {
            return Err(format!("{name} failed: {value:.3e} > {TOLERANCE:.3e}"));
        }
    }
    Ok(metrics)
}

fn main() {
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("helmholtz_residual", metrics.helmholtz_residual)
                .metric("radial_wall_derivative", metrics.radial_wall_derivative)
                .metric("end_wall_derivative", metrics.end_wall_derivative)
                .maximum("helmholtz_residual", TOLERANCE)
                .maximum("radial_wall_derivative", TOLERANCE)
                .maximum("end_wall_derivative", TOLERANCE)
                .seed("axisymmetric-rigid-cavity:radius-0.2:length-0.8:radial-1:axial-2:cpu");
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("axisymmetric cavity QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
                .expect("axisymmetric cavity QA failure result must serialize");
            println!("QA_RESULT: {record}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
