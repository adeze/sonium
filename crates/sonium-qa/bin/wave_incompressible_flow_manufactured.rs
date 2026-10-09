//! Reference-only QA for the two-dimensional Taylor-Green vortex.

use sonium_qa::QaResult;
use sonium_wave::analytical::ManufacturedIncompressibleFlow2d;

const CASE_ID: &str = "wave.incompressible-flow-manufactured-2d.v1";
const EXACT_TOLERANCE: f64 = 5.0e-13;
const FINITE_DIFFERENCE_TOLERANCE: f64 = 1.0e-4;

fn main() {
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("max_momentum_residual", metrics.max_momentum_residual)
                .metric("max_divergence", metrics.max_divergence)
                .metric(
                    "max_centered_velocity_gradient_relative_error",
                    metrics.max_centered_velocity_gradient_relative_error,
                )
                .metric("sample_count", metrics.sample_count as f64)
                .maximum("max_momentum_residual", EXACT_TOLERANCE)
                .maximum("max_divergence", EXACT_TOLERANCE)
                .maximum(
                    "max_centered_velocity_gradient_relative_error",
                    FINITE_DIFFERENCE_TOLERANCE,
                )
                .seed("manufactured-incompressible-flow-2d:taylor-green-129x129:cpu");
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("Taylor-Green QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
                .expect("Taylor-Green QA failure result must serialize");
            println!("QA_RESULT: {record}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

struct Metrics {
    max_momentum_residual: f64,
    max_divergence: f64,
    max_centered_velocity_gradient_relative_error: f64,
    sample_count: usize,
}

fn run() -> Result<Metrics, String> {
    let reference = ManufacturedIncompressibleFlow2d::try_new(1.2, 0.03, 1.4, 2.0, [-0.2, 0.3])
        .map_err(|error| error.to_string())?;
    let time_s = 0.37;
    let grid_points = 129;
    let x_min = -1.0;
    let x_max = 1.0;
    let y_min = -0.8;
    let y_max = 1.2;
    let dx = (x_max - x_min) / (grid_points - 1) as f64;
    let dy = (y_max - y_min) / (grid_points - 1) as f64;
    let mut max_momentum_residual = 0.0_f64;
    let mut max_divergence = 0.0_f64;
    let mut max_centered_velocity_gradient_relative_error = 0.0_f64;
    for y_index in 0..grid_points {
        for x_index in 0..grid_points {
            let position = [x_min + x_index as f64 * dx, y_min + y_index as f64 * dy];
            let point = reference
                .try_evaluate(position, time_s)
                .map_err(|error| error.to_string())?;
            max_momentum_residual = max_momentum_residual.max(
                point.momentum_residual_m_per_s2[0]
                    .abs()
                    .max(point.momentum_residual_m_per_s2[1].abs()),
            );
            max_divergence = max_divergence.max(point.divergence_per_s.abs());
            if x_index > 0 && x_index + 1 < grid_points && y_index > 0 && y_index + 1 < grid_points
            {
                let x_left = reference
                    .try_evaluate([position[0] - dx, position[1]], time_s)
                    .map_err(|error| error.to_string())?;
                let x_right = reference
                    .try_evaluate([position[0] + dx, position[1]], time_s)
                    .map_err(|error| error.to_string())?;
                let y_left = reference
                    .try_evaluate([position[0], position[1] - dy], time_s)
                    .map_err(|error| error.to_string())?;
                let y_right = reference
                    .try_evaluate([position[0], position[1] + dy], time_s)
                    .map_err(|error| error.to_string())?;
                for component in 0..2 {
                    let derivative_x = (x_right.velocity_m_per_s[component]
                        - x_left.velocity_m_per_s[component])
                        / (2.0 * dx);
                    let derivative_y = (y_right.velocity_m_per_s[component]
                        - y_left.velocity_m_per_s[component])
                        / (2.0 * dy);
                    max_centered_velocity_gradient_relative_error =
                        max_centered_velocity_gradient_relative_error.max(
                            (derivative_x - point.velocity_gradient_per_s[component][0]).abs()
                                / point.velocity_gradient_per_s[component][0].abs().max(1.0),
                        );
                    max_centered_velocity_gradient_relative_error =
                        max_centered_velocity_gradient_relative_error.max(
                            (derivative_y - point.velocity_gradient_per_s[component][1]).abs()
                                / point.velocity_gradient_per_s[component][1].abs().max(1.0),
                        );
                }
            }
        }
    }
    for (name, value, tolerance) in [
        ("momentum residual", max_momentum_residual, EXACT_TOLERANCE),
        ("divergence", max_divergence, EXACT_TOLERANCE),
        (
            "centered velocity gradient",
            max_centered_velocity_gradient_relative_error,
            FINITE_DIFFERENCE_TOLERANCE,
        ),
    ] {
        if !value.is_finite() || value > tolerance {
            return Err(format!("{name} failed: {value:.3e} > {tolerance:.3e}"));
        }
    }
    Ok(Metrics {
        max_momentum_residual,
        max_divergence,
        max_centered_velocity_gradient_relative_error,
        sample_count: grid_points * grid_points,
    })
}
