//! Independent periodic DRP linearized-Euler dispersion reference.
//!
//! This is a small, deterministic reference case for the FEM aeroacoustics
//! roadmap. It does not claim to validate the mixed FEM assembler; it pins the
//! expected characteristic dispersion and time-domain propagation behavior
//! that a future FEM pollution study can compare against.

use sonium_fem::physics::acoustics::aeroacoustics::{
    AcousticBranch, DrpEuler1d, DrpEuler1dConfig, DrpWaveState,
};
use sonium_qa::QaResult;
use std::f64::consts::PI;

const CASE_ID: &str = "fem.aeroacoustics-drp-reference.v1";
const MIN_CONVERGENCE_RATE: f64 = 3.5;
const MAX_FOURIER_L2_ERROR: f64 = 2.0e-5;

fn convergence_rate(errors: &[f64], spacings: &[f64]) -> Result<f64, String> {
    if errors.len() != spacings.len() || errors.len() < 2 {
        return Err("DRP convergence vectors must contain at least two aligned entries".into());
    }
    let mut rates = Vec::with_capacity(errors.len() - 1);
    for pair in errors.windows(2).zip(spacings.windows(2)) {
        let ([coarse, fine], [h_coarse, h_fine]) = pair else {
            unreachable!("windows(2) always yields two entries");
        };
        if !coarse.is_finite()
            || !fine.is_finite()
            || !h_coarse.is_finite()
            || !h_fine.is_finite()
            || *coarse <= *fine
            || *h_coarse <= *h_fine
            || *fine <= 0.0
        {
            return Err(format!(
                "invalid DRP refinement pair: error {coarse:e}->{fine:e}, spacing {h_coarse:e}->{h_fine:e}"
            ));
        }
        rates.push((*coarse / *fine).log2() / (*h_coarse / *h_fine).log2());
    }
    let mean = rates.iter().sum::<f64>() / rates.len() as f64;
    if !mean.is_finite() {
        return Err("DRP convergence rate is non-finite".into());
    }
    Ok(mean)
}

fn run() -> Result<(f64, f64, f64), String> {
    let n_values = [16_usize, 32, 64];
    let mut phase_errors = Vec::with_capacity(n_values.len());
    let mut spacings = Vec::with_capacity(n_values.len());

    for &n_points in &n_values {
        let config = DrpEuler1dConfig::with_cfl(n_points, 2.0 * PI, 1.3, 2.0, 0.3, 0.08)
            .map_err(|error| format!("construct DRP reference config: {error}"))?;
        let dispersion = config.dispersion(1.0, AcousticBranch::Downstream);
        let exact_speed = config.mean_velocity + config.speed_of_sound;
        let phase_error = (dispersion.phase_velocity / exact_speed - 1.0).abs();
        if !phase_error.is_finite() {
            return Err("DRP phase error is non-finite".into());
        }
        phase_errors.push(phase_error);
        spacings.push(config.dx());
    }

    let convergence_rate = convergence_rate(&phase_errors, &spacings)?;
    if convergence_rate < MIN_CONVERGENCE_RATE {
        return Err(format!(
            "DRP dispersion convergence rate {convergence_rate:.3} is below {MIN_CONVERGENCE_RATE:.3}"
        ));
    }

    let config = DrpEuler1dConfig::with_cfl(128, 2.0 * PI, 1.3, 2.0, 0.3, 0.08)
        .map_err(|error| format!("construct Fourier-mode config: {error}"))?;
    let mode = 7.0;
    let mut state = DrpWaveState::zeros(config.n_points);
    for i in 0..config.n_points {
        let x = i as f64 * config.dx();
        state.pressure[i] = (mode * x).cos();
        state.velocity[i] = state.pressure[i] / (config.density * config.speed_of_sound);
    }
    let mut solver = DrpEuler1d::new(config, state)
        .map_err(|error| format!("construct Fourier-mode solver: {error}"))?;
    solver.advance(80);
    let omega = config
        .dispersion(mode, AcousticBranch::Downstream)
        .angular_frequency;
    let relative_l2_error = solver
        .state
        .pressure
        .iter()
        .enumerate()
        .map(|(i, actual)| {
            let x = i as f64 * config.dx();
            let expected = (mode * x - omega * solver.time).cos();
            (actual - expected).powi(2)
        })
        .sum::<f64>()
        .sqrt()
        / (config.n_points as f64 / 2.0).sqrt();
    if !relative_l2_error.is_finite() || relative_l2_error > MAX_FOURIER_L2_ERROR {
        return Err(format!(
            "Fourier-mode relative L2 error {relative_l2_error:e} exceeds {MAX_FOURIER_L2_ERROR:e}"
        ));
    }

    Ok((
        convergence_rate,
        phase_errors.last().copied().unwrap_or_default(),
        relative_l2_error,
    ))
}

fn main() {
    match run() {
        Ok((convergence_rate, fine_phase_error, fourier_l2_error)) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("dispersion_convergence_rate", convergence_rate)
                .metric("fine_phase_error", fine_phase_error)
                .metric("fourier_mode_relative_l2", fourier_l2_error)
                .minimum("dispersion_convergence_rate", MIN_CONVERGENCE_RATE)
                .maximum("fourier_mode_relative_l2", MAX_FOURIER_L2_ERROR)
                .to_json_line()
                .expect("DRP QA result must serialize");
            println!("QA_RESULT: {record}");
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error)
                .to_json_line()
                .expect("DRP QA failure result must serialize");
            println!("QA_RESULT: {record}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
