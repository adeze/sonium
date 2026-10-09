//! Reference-only QA for a closed-closed, rigid-walled one-dimensional duct.
//!
//! This records analytical eigenmode identities only. It deliberately does
//! not claim validation of a FEM/BEM eigenproblem, duct radiation, damping,
//! flow, transverse modes, or a coupled duct network.

use sonium_qa::QaResult;
use sonium_wave::analytical::{SPEED_OF_SOUND, try_rigid_duct_1d_mode};

const CASE_ID: &str = "wave.rigid-duct-mode-1d.v1";
const EXACT_TOLERANCE: f64 = 5.0e-13;
// The wall derivative is estimated with a one-sided first-order difference;
// its refinement-ratio error is a discretization check, not an exact identity.
const SLOPE_REFINEMENT_TOLERANCE: f64 = 5.0e-6;

struct Metrics {
    frequency_relative_error: f64,
    boundary_amplitude_error: f64,
    boundary_slope_refinement_error: f64,
}

fn one_sided_wall_slopes(sample_count: usize) -> Result<(f64, f64), String> {
    let solution = try_rigid_duct_1d_mode(0.8, 3, sample_count)
        .map_err(|error| format!("evaluate rigid-duct mode: {error}"))?;
    let dx = solution.positions[1].x - solution.positions[0].x;
    let left = (solution.pressure[1].re - solution.pressure[0].re) / dx;
    let last = solution.pressure.len() - 1;
    let right = (solution.pressure[last].re - solution.pressure[last - 1].re) / dx;
    Ok((left, right))
}

fn run() -> Result<Metrics, String> {
    let length_m = 0.8;
    let mode_number = 3;
    let solution = try_rigid_duct_1d_mode(length_m, mode_number, 801)
        .map_err(|error| format!("evaluate rigid-duct mode: {error}"))?;
    let expected_frequency_hz = mode_number as f64 * SPEED_OF_SOUND / (2.0 * length_m);
    let frequency_relative_error =
        (solution.frequency - expected_frequency_hz).abs() / expected_frequency_hz;
    let boundary_amplitude_error = (solution.pressure[0].re.abs() - 1.0).abs().max(
        (solution
            .pressure
            .last()
            .ok_or("rigid-duct grid is unexpectedly empty")?
            .re
            .abs()
            - 1.0)
            .abs(),
    );

    let (coarse_left, coarse_right) = one_sided_wall_slopes(801)?;
    let (fine_left, fine_right) = one_sided_wall_slopes(1601)?;
    let boundary_slope_refinement_error = (fine_left / coarse_left - 0.5)
        .abs()
        .max((fine_right / coarse_right - 0.5).abs());
    let metrics = Metrics {
        frequency_relative_error,
        boundary_amplitude_error,
        boundary_slope_refinement_error,
    };
    for (name, value, tolerance) in [
        (
            "modal frequency",
            metrics.frequency_relative_error,
            EXACT_TOLERANCE,
        ),
        (
            "rigid-wall pressure amplitude",
            metrics.boundary_amplitude_error,
            EXACT_TOLERANCE,
        ),
        (
            "rigid-wall derivative refinement",
            metrics.boundary_slope_refinement_error,
            SLOPE_REFINEMENT_TOLERANCE,
        ),
    ] {
        if !value.is_finite() || value > tolerance {
            return Err(format!("{name} failed: {value:.3e} > {tolerance:.3e}"));
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
                .metric("frequency_relative_error", metrics.frequency_relative_error)
                .metric("boundary_amplitude_error", metrics.boundary_amplitude_error)
                .metric(
                    "boundary_slope_refinement_error",
                    metrics.boundary_slope_refinement_error,
                )
                .maximum("frequency_relative_error", EXACT_TOLERANCE)
                .maximum("boundary_amplitude_error", EXACT_TOLERANCE)
                .maximum(
                    "boundary_slope_refinement_error",
                    SLOPE_REFINEMENT_TOLERANCE,
                )
                .seed("rigid-duct-1d:closed-closed:mode-3:cpu");
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("rigid-duct QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
                .expect("rigid-duct QA failure result must serialize");
            println!("QA_RESULT: {record}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
