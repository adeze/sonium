//! Reference-only QA for the manufactured one-dimensional TE-Maxwell family.
//!
//! This checks the closed-form field/source equations. It is a reusable
//! reference, not a claim that a numerical Maxwell solve has converged.

use sonium_qa::QaResult;
use sonium_wave::analytical::ManufacturedMaxwell1d;
use std::f64::consts::PI;

const CASE_ID: &str = "wave.maxwell-manufactured-1d.v1";
const EXACT_TOLERANCE: f64 = 5.0e-13;
const FINITE_DIFFERENCE_TOLERANCE: f64 = 5.0e-5;

fn main() {
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("max_ampere_residual", metrics.max_ampere_residual)
                .metric("max_faraday_residual", metrics.max_faraday_residual)
                .metric(
                    "max_centered_spatial_relative_error",
                    metrics.max_centered_spatial_relative_error,
                )
                .metric("max_source_free_current", metrics.max_source_free_current)
                .metric(
                    "max_pec_electric_endpoint",
                    metrics.max_pec_electric_endpoint,
                )
                .metric("sample_count", metrics.sample_count as f64)
                .maximum("max_ampere_residual", EXACT_TOLERANCE)
                .maximum("max_faraday_residual", EXACT_TOLERANCE)
                .maximum(
                    "max_centered_spatial_relative_error",
                    FINITE_DIFFERENCE_TOLERANCE,
                )
                .maximum("max_source_free_current", EXACT_TOLERANCE)
                .maximum("max_pec_electric_endpoint", EXACT_TOLERANCE)
                .seed("manufactured-maxwell-1d:te-uniform-257:cpu");
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("manufactured Maxwell QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
                .expect("manufactured Maxwell QA failure result must serialize");
            println!("QA_RESULT: {record}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

struct Metrics {
    max_ampere_residual: f64,
    max_faraday_residual: f64,
    max_centered_spatial_relative_error: f64,
    max_source_free_current: f64,
    max_pec_electric_endpoint: f64,
    sample_count: usize,
}

fn run() -> Result<Metrics, String> {
    let reference = ManufacturedMaxwell1d::try_new(2.0, 3.0, 1.7, 1.3, 2.4, 0.9, -0.2)
        .map_err(|error| error.to_string())?;
    let sample = reference
        .try_sample(-1.0, 1.0, 257, 0.37)
        .map_err(|error| error.to_string())?;
    sample.validate().map_err(|error| error.to_string())?;
    let max_ampere_residual = sample
        .electric_time_derivatives_v_per_m_s
        .iter()
        .zip(&sample.magnetic_spatial_derivatives_a_per_m2)
        .zip(&sample.electric_current_densities_a_per_m2)
        .map(|((electric_time, magnetic_space), current)| {
            (reference.permittivity_f_per_m() * electric_time + magnetic_space + current).abs()
        })
        .fold(0.0_f64, f64::max);
    let max_faraday_residual = sample
        .magnetic_time_derivatives_a_per_m_s
        .iter()
        .zip(&sample.electric_spatial_derivatives_v_per_m2)
        .zip(&sample.magnetic_current_densities_v_per_m2)
        .map(|((magnetic_time, electric_space), current)| {
            (reference.permeability_h_per_m() * magnetic_time + electric_space + current).abs()
        })
        .fold(0.0_f64, f64::max);
    let dx_m = sample.positions_m[1] - sample.positions_m[0];
    let max_centered_spatial_relative_error = sample
        .electric_fields_v_per_m
        .windows(3)
        .zip(sample.electric_spatial_derivatives_v_per_m2.iter().skip(1))
        .chain(
            sample
                .magnetic_fields_a_per_m
                .windows(3)
                .zip(sample.magnetic_spatial_derivatives_a_per_m2.iter().skip(1)),
        )
        .map(|(value, exact_derivative)| {
            (((value[2] - value[0]) / (2.0 * dx_m)) - exact_derivative).abs()
                / exact_derivative.abs().max(1.0)
        })
        .fold(0.0_f64, f64::max);

    let length_m = 2.0;
    let permittivity = 2.0_f64;
    let permeability = 3.0_f64;
    let wavenumber = PI / length_m;
    let angular_frequency = wavenumber / (permittivity * permeability).sqrt();
    let electric_amplitude = 2.3;
    let magnetic_amplitude = permittivity * angular_frequency * electric_amplitude / wavenumber;
    let source_free = ManufacturedMaxwell1d::try_new(
        permittivity,
        permeability,
        angular_frequency,
        wavenumber,
        electric_amplitude,
        magnetic_amplitude,
        0.0,
    )
    .map_err(|error| error.to_string())?;
    let source_free_sample = source_free
        .try_sample(0.0, length_m, 65, PI / (5.0 * angular_frequency))
        .map_err(|error| error.to_string())?;
    let max_source_free_current = source_free_sample
        .electric_current_densities_a_per_m2
        .iter()
        .chain(&source_free_sample.magnetic_current_densities_v_per_m2)
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let max_pec_electric_endpoint = source_free_sample.electric_fields_v_per_m[0].abs().max(
        source_free_sample.electric_fields_v_per_m
            [source_free_sample.electric_fields_v_per_m.len() - 1]
            .abs(),
    );

    for (name, value, tolerance) in [
        ("Ampere residual", max_ampere_residual, EXACT_TOLERANCE),
        ("Faraday residual", max_faraday_residual, EXACT_TOLERANCE),
        (
            "centered spatial derivative",
            max_centered_spatial_relative_error,
            FINITE_DIFFERENCE_TOLERANCE,
        ),
        (
            "source-free current",
            max_source_free_current,
            EXACT_TOLERANCE,
        ),
        (
            "PEC electric endpoint",
            max_pec_electric_endpoint,
            EXACT_TOLERANCE,
        ),
    ] {
        if !value.is_finite() || value > tolerance {
            return Err(format!("{name} failed: {value:.3e} > {tolerance:.3e}"));
        }
    }

    Ok(Metrics {
        max_ampere_residual,
        max_faraday_residual,
        max_centered_spatial_relative_error,
        max_source_free_current,
        max_pec_electric_endpoint,
        sample_count: sample.positions_m.len() + source_free_sample.positions_m.len(),
    })
}
