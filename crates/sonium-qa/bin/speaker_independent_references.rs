//! Independent analytical references for speaker magnetic and enclosure paths.

use sonium_fem::physics::acoustics::pressure::resolved::helmholtz_resonance_frequency_hz;
use sonium_qa::QaResult;
use sonium_sim_speaker_modeling::{ReferenceModel, baseline_reference_models};
use std::time::Instant;

const CASE_ID: &str = "speaker.independent-magnetic-enclosure-references.v1";
const SOUND_SPEED_M_PER_S: f64 = 343.0;
const VACUUM_PERMEABILITY_H_PER_M: f64 = 4.0e-7 * std::f64::consts::PI;
const REFERENCE_TOLERANCE: f64 = 1.0e-12;

fn run() -> Result<(f64, f64, f64, f64, f64), String> {
    let path_length_m = 0.1;
    let cross_section_m2 = 1.0e-4;
    let relative_permeability = 1_000.0;
    let turns: f64 = 100.0;
    let current_rms_a: f64 = 1.0;
    let references = baseline_reference_models();
    let magnetic = references
        .iter()
        .find(|reference| reference.name() == "uniform_magnetic_circuit")
        .ok_or_else(|| "baseline magnetic reference is missing".to_owned())?;
    let (energy_j, force_n) = magnetic
        .uniform_magnetic_circuit_energy_force()
        .ok_or_else(|| "magnetic reference did not expose a circuit contract".to_owned())?
        .map_err(|error| format!("magnetic reference invalid: {error}"))?;
    if !(energy_j.is_finite() && energy_j > 0.0 && force_n.is_finite() && force_n > 0.0) {
        return Err("magnetic energy/force reference is not positive finite".to_owned());
    }
    let inductance_h =
        VACUUM_PERMEABILITY_H_PER_M * relative_permeability * cross_section_m2 * turns.powi(2)
            / path_length_m;
    let expected_energy_j = 0.5 * inductance_h * current_rms_a.powi(2);
    let expected_force_n = 0.5
        * VACUUM_PERMEABILITY_H_PER_M
        * relative_permeability
        * cross_section_m2
        * (turns * current_rms_a / path_length_m).powi(2);
    let relative_error = |actual: f64, expected: f64| {
        (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE)
    };
    let energy_error = relative_error(energy_j, expected_energy_j);
    let force_error = relative_error(force_n, expected_force_n);
    if energy_error > REFERENCE_TOLERANCE || force_error > REFERENCE_TOLERANCE {
        return Err(format!(
            "magnetic reference errors exceed tolerance: energy={energy_error:e}, force={force_error:e}"
        ));
    }

    let cavity = ReferenceModel::RigidRectangularCavity {
        dimensions_m: [0.4, 0.3, 0.25],
    };
    let mode_100 = cavity
        .rigid_rectangular_cavity_mode_hz(SOUND_SPEED_M_PER_S, [1, 0, 0])
        .ok_or_else(|| "cavity reference did not expose modes".to_owned())?
        .map_err(|error| format!("cavity reference invalid: {error}"))?;
    let expected_mode_100 = SOUND_SPEED_M_PER_S / (2.0 * 0.4);
    let mode_error = (mode_100 - expected_mode_100).abs();
    if mode_error > REFERENCE_TOLERANCE {
        return Err(format!("cavity mode reference error {mode_error:e}"));
    }

    let cavity_volume_m3 = 0.4 * 0.3 * 0.25;
    let port_area_m2 = std::f64::consts::PI * 0.035_f64.powi(2);
    let effective_length_m = 0.15 + 1.7 * 0.035;
    let helmholtz_hz = helmholtz_resonance_frequency_hz(
        SOUND_SPEED_M_PER_S,
        cavity_volume_m3,
        port_area_m2,
        effective_length_m,
    );
    let expected_helmholtz_hz = SOUND_SPEED_M_PER_S / (2.0 * std::f64::consts::PI)
        * (port_area_m2 / (cavity_volume_m3 * effective_length_m)).sqrt();
    let helmholtz_error = (helmholtz_hz - expected_helmholtz_hz).abs();
    if helmholtz_error > REFERENCE_TOLERANCE {
        return Err(format!("Helmholtz reference error {helmholtz_error:e}"));
    }

    Ok((energy_j, force_n, helmholtz_hz, energy_error, force_error))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((energy_j, force_n, helmholtz_hz, energy_error, force_error)) => {
            let result = QaResult::passed(CASE_ID)
                .reference("independent uniform magnetic-circuit and enclosure resonance formulas")
                .reference_checksum(
                    "analytic/deterministic-fixture:speaker-independent-references-v1",
                )
                .dimension("uniform magnetic circuit; rigid rectangular cavity; Helmholtz port")
                .formulation("coenergy/reluctance force and closed-form enclosure resonance")
                .tolerance("all independent reference relative errors <= 1e-12")
                .metric("magnetic_energy_j", energy_j)
                .metric("magnetic_force_n", force_n)
                .metric("helmholtz_frequency_hz", helmholtz_hz)
                .metric("magnetic_energy_relative_error", energy_error)
                .metric("magnetic_force_relative_error", force_error)
                .metric("independent_reference_checks", 5.0)
                .minimum("independent_reference_checks", 3.0)
                .maximum("magnetic_energy_relative_error", REFERENCE_TOLERANCE)
                .maximum("magnetic_force_relative_error", REFERENCE_TOLERANCE)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_reference_binary_contract_passes() {
        let (energy, force, helmholtz, energy_error, force_error) =
            run().expect("speaker references should pass");
        assert!(energy > 0.0 && force > 0.0 && helmholtz > 0.0);
        assert!(energy_error <= REFERENCE_TOLERANCE);
        assert!(force_error <= REFERENCE_TOLERANCE);
    }
}
