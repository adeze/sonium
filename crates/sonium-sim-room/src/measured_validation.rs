//! Checked comparison of solved RoomSim responses with measured data.
//!
//! The comparison deliberately lives at the `SimulationOutput` boundary.  It
//! rejects reduced-order and synthetic samples before they can be presented as
//! measured-room validation, checks the complete source/listener/frequency
//! topology, and reports magnitude and wrapped phase errors in physical units.

use crate::SimulationOutput;
use num_complex::Complex64;
use serde::Serialize;
use std::f64::consts::PI;

/// Complex measured pressure samples in peak pascals.
///
/// The tensor uses the same `[source][listener][frequency]` layout as
/// [`SimulationOutput`].  `provenance_label` is intentionally required so a
/// report can identify the instrument, fixture, or imported dataset that was
/// compared without treating the label as proof of calibration.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredResponseSet {
    /// Strictly increasing frequency grid in Hz.
    pub frequencies: Vec<f64>,
    /// Source names matching the first pressure axis.
    pub source_names: Vec<String>,
    /// Complex peak pressure in pascals, indexed `[source][listener][frequency]`.
    pub pressures: Vec<Vec<Vec<Complex64>>>,
    /// Human-readable dataset or instrument provenance label.
    pub provenance_label: String,
}

impl MeasuredResponseSet {
    /// Construct and validate a measured response tensor.
    pub fn new(
        frequencies: Vec<f64>,
        source_names: Vec<String>,
        pressures: Vec<Vec<Vec<Complex64>>>,
        provenance_label: impl Into<String>,
    ) -> Result<Self, String> {
        let measured = Self {
            frequencies,
            source_names,
            pressures,
            provenance_label: provenance_label.into(),
        };
        measured.validate_layout()?;
        Ok(measured)
    }

    /// Validate frequency, source/listener, pressure, and provenance axes.
    pub fn validate_layout(&self) -> Result<(), String> {
        if self.provenance_label.trim().is_empty() {
            return Err("measured response requires a provenance label".into());
        }
        if self.frequencies.is_empty()
            || self
                .frequencies
                .iter()
                .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
            || self
                .frequencies
                .windows(2)
                .any(|window| window[1] <= window[0])
        {
            return Err(
                "measured frequencies must be finite, positive, and strictly increasing".into(),
            );
        }
        if self.source_names.len() != self.pressures.len() || self.pressures.is_empty() {
            return Err("measured source and pressure axes are empty or misaligned".into());
        }
        let mut names = std::collections::HashSet::with_capacity(self.source_names.len());
        for (source_index, name) in self.source_names.iter().enumerate() {
            if name.trim().is_empty() || !names.insert(name.to_ascii_lowercase()) {
                return Err(format!(
                    "measured source name {source_index} is blank or duplicated"
                ));
            }
            let source = &self.pressures[source_index];
            if source.is_empty() {
                return Err(format!("measured source {source_index} has no listeners"));
            }
            for (listener_index, response) in source.iter().enumerate() {
                if response.len() != self.frequencies.len() {
                    return Err(format!(
                        "measured response source {source_index}, listener {listener_index} has {} samples, expected {}",
                        response.len(),
                        self.frequencies.len()
                    ));
                }
                if response
                    .iter()
                    .any(|pressure| !pressure.re.is_finite() || !pressure.im.is_finite())
                {
                    return Err(format!(
                        "measured response source {source_index}, listener {listener_index} contains non-finite pressure"
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Aggregate complex-response errors for one solved-versus-measured comparison.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct MeasuredComparisonReport {
    /// Number of compared source/listener/frequency pressure samples.
    pub sample_count: usize,
    /// Number of samples with a defined phase on both sides.
    pub phase_sample_count: usize,
    /// RMS difference in peak-pressure SPL, in dB.
    pub rms_magnitude_error_db: f64,
    /// Maximum absolute peak-pressure SPL difference, in dB.
    pub max_abs_magnitude_error_db: f64,
    /// RMS wrapped phase difference, in radians.
    pub rms_phase_error_rad: f64,
    /// Maximum absolute wrapped phase difference, in radians.
    pub max_abs_phase_error_rad: f64,
}

impl MeasuredComparisonReport {
    /// Require magnitude and phase errors to stay within explicit budgets.
    pub fn require_within(
        &self,
        max_magnitude_error_db: f64,
        max_phase_error_rad: f64,
    ) -> Result<(), String> {
        if !max_magnitude_error_db.is_finite()
            || max_magnitude_error_db < 0.0
            || !max_phase_error_rad.is_finite()
            || max_phase_error_rad < 0.0
            || self.sample_count == 0
            || self.phase_sample_count > self.sample_count
            || !self.rms_magnitude_error_db.is_finite()
            || self.rms_magnitude_error_db < 0.0
            || !self.max_abs_magnitude_error_db.is_finite()
            || self.max_abs_magnitude_error_db < 0.0
            || !self.rms_phase_error_rad.is_finite()
            || self.rms_phase_error_rad < 0.0
            || !self.max_abs_phase_error_rad.is_finite()
            || self.max_abs_phase_error_rad < 0.0
        {
            return Err("measured comparison report or budgets are invalid".into());
        }
        if self.max_abs_magnitude_error_db > max_magnitude_error_db {
            return Err(format!(
                "measured magnitude error {:.6} dB exceeds {:.6} dB budget",
                self.max_abs_magnitude_error_db, max_magnitude_error_db
            ));
        }
        if self.max_abs_phase_error_rad > max_phase_error_rad {
            return Err(format!(
                "measured phase error {:.6} rad exceeds {:.6} rad budget",
                self.max_abs_phase_error_rad, max_phase_error_rad
            ));
        }
        Ok(())
    }
}

/// Compare solved RoomSim output with a measured complex response tensor.
///
/// The model must be entirely [`crate::FrequencyProvenance::Solved`].  Frequency
/// points are paired by index and may differ by at most
/// `frequency_tolerance_hz`; source names and listener cardinalities must
/// match exactly.  Phase error is wrapped at ±π, so a response crossing the
/// branch cut is not falsely reported as a 2π error.
pub fn compare_solved_output_to_measured(
    output: &SimulationOutput,
    measured: &MeasuredResponseSet,
    frequency_tolerance_hz: f64,
) -> Result<MeasuredComparisonReport, String> {
    output.require_solved("measured response comparison")?;
    measured.validate_layout()?;
    if !frequency_tolerance_hz.is_finite() || frequency_tolerance_hz < 0.0 {
        return Err("frequency comparison tolerance must be finite and non-negative".into());
    }
    if output.source_names != measured.source_names {
        return Err("measured and solved source names differ".into());
    }
    if output.frequencies.len() != measured.frequencies.len() {
        return Err("measured and solved frequency grids differ in length".into());
    }
    for (index, (model_frequency, measured_frequency)) in output
        .frequencies
        .iter()
        .zip(&measured.frequencies)
        .enumerate()
    {
        if (model_frequency - measured_frequency).abs() > frequency_tolerance_hz {
            return Err(format!(
                "measured frequency mismatch at index {index}: model={model_frequency}, measured={measured_frequency}"
            ));
        }
    }
    if output.pressures.len() != measured.pressures.len() {
        return Err("measured and solved source axes differ in length".into());
    }

    let mut magnitude_squared: f64 = 0.0;
    let mut max_magnitude_error_db: f64 = 0.0;
    let mut phase_squared: f64 = 0.0;
    let mut max_phase_error_rad: f64 = 0.0;
    let mut sample_count = 0;
    let mut phase_sample_count = 0;
    for (source_index, (model_source, measured_source)) in
        output.pressures.iter().zip(&measured.pressures).enumerate()
    {
        if model_source.len() != measured_source.len() {
            return Err(format!(
                "measured and solved listener axes differ for source {source_index}"
            ));
        }
        for (listener_index, (model_response, measured_response)) in
            model_source.iter().zip(measured_source).enumerate()
        {
            if model_response.len() != measured_response.len() {
                return Err(format!(
                    "measured and solved response lengths differ for source {source_index}, listener {listener_index}"
                ));
            }
            for (model_pressure, measured_pressure) in model_response.iter().zip(measured_response)
            {
                let magnitude_error_db = crate::room_metrics::peak_pressure_to_spl(*model_pressure)
                    - crate::room_metrics::peak_pressure_to_spl(*measured_pressure);
                magnitude_squared += magnitude_error_db * magnitude_error_db;
                max_magnitude_error_db = max_magnitude_error_db.max(magnitude_error_db.abs());
                sample_count += 1;

                if model_pressure.norm() > 0.0 && measured_pressure.norm() > 0.0 {
                    let phase_error = wrap_phase(model_pressure.arg() - measured_pressure.arg());
                    phase_squared += phase_error * phase_error;
                    max_phase_error_rad = max_phase_error_rad.max(phase_error.abs());
                    phase_sample_count += 1;
                }
            }
        }
    }
    if sample_count == 0 {
        return Err("measured comparison contains no pressure samples".into());
    }
    let report = MeasuredComparisonReport {
        sample_count,
        phase_sample_count,
        rms_magnitude_error_db: (magnitude_squared / sample_count as f64).sqrt(),
        max_abs_magnitude_error_db: max_magnitude_error_db,
        rms_phase_error_rad: if phase_sample_count == 0 {
            0.0
        } else {
            (phase_squared / phase_sample_count as f64).sqrt()
        },
        max_abs_phase_error_rad: max_phase_error_rad,
    };
    if !report.rms_magnitude_error_db.is_finite()
        || !report.max_abs_magnitude_error_db.is_finite()
        || !report.rms_phase_error_rad.is_finite()
        || !report.max_abs_phase_error_rad.is_finite()
    {
        return Err("measured comparison produced non-finite error metrics".into());
    }
    Ok(report)
}

fn wrap_phase(phase: f64) -> f64 {
    (phase + PI).rem_euclid(2.0 * PI) - PI
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrequencyProvenance;

    fn fixture(provenance: FrequencyProvenance) -> (SimulationOutput, MeasuredResponseSet) {
        let frequencies = vec![100.0, 200.0, 400.0];
        let pressures = vec![vec![
            vec![
                Complex64::from_polar(1.0, 0.25),
                Complex64::from_polar(2.0, 1.0),
                Complex64::from_polar(0.5, -2.5),
            ],
            vec![
                Complex64::from_polar(0.75, -0.5),
                Complex64::from_polar(1.5, 2.0),
                Complex64::from_polar(0.25, 2.8),
            ],
        ]];
        (
            SimulationOutput::with_uniform_provenance(
                frequencies.clone(),
                pressures.clone(),
                vec!["left".into()],
                provenance,
            ),
            MeasuredResponseSet::new(
                frequencies,
                vec!["left".into()],
                pressures,
                "fixture/instrument-001",
            )
            .unwrap(),
        )
    }

    #[test]
    fn exact_solved_match_has_zero_magnitude_and_phase_error() {
        let (output, measured) = fixture(FrequencyProvenance::Solved);
        let report = compare_solved_output_to_measured(&output, &measured, 0.0).unwrap();
        assert_eq!(report.sample_count, 6);
        assert_eq!(report.phase_sample_count, 6);
        assert!(report.rms_magnitude_error_db.abs() < 1.0e-12);
        assert!(report.rms_phase_error_rad.abs() < 1.0e-12);
        report.require_within(1.0e-12, 1.0e-12).unwrap();
    }

    #[test]
    fn phase_error_wraps_across_pi_branch_cut() {
        let frequencies = vec![100.0];
        let measured_pressure = Complex64::from_polar(1.0, PI - 1.0e-6);
        let model_pressure = Complex64::from_polar(1.0, -PI + 1.0e-6);
        let output = SimulationOutput::solved(
            frequencies.clone(),
            vec![vec![vec![model_pressure]]],
            vec!["source".into()],
        );
        let measured = MeasuredResponseSet::new(
            frequencies,
            vec!["source".into()],
            vec![vec![vec![measured_pressure]]],
            "branch-cut-fixture",
        )
        .unwrap();
        let report = compare_solved_output_to_measured(&output, &measured, 0.0).unwrap();
        assert!(report.max_abs_phase_error_rad < 3.0e-6);
    }

    #[test]
    fn synthetic_or_reduced_model_is_rejected_before_comparison() {
        for provenance in [
            FrequencyProvenance::ReducedOrder,
            FrequencyProvenance::Interpolated,
            FrequencyProvenance::Geometric,
            FrequencyProvenance::Synthetic,
        ] {
            let (output, measured) = fixture(provenance);
            let error = compare_solved_output_to_measured(&output, &measured, 0.0).unwrap_err();
            assert!(error.contains("requires solved samples"), "{error}");
        }
    }

    #[test]
    fn malformed_measured_layouts_are_rejected() {
        assert!(
            MeasuredResponseSet::new(
                vec![100.0, 100.0],
                vec!["source".into()],
                vec![vec![vec![Complex64::new(1.0, 0.0); 2]]],
                "fixture",
            )
            .is_err()
        );
        assert!(
            MeasuredResponseSet::new(
                vec![100.0],
                vec!["source".into()],
                vec![vec![vec![Complex64::new(f64::NAN, 0.0)]]],
                "fixture",
            )
            .is_err()
        );
        assert!(
            MeasuredResponseSet::new(
                vec![100.0],
                vec![" ".into()],
                vec![vec![vec![Complex64::new(1.0, 0.0)]]],
                "fixture",
            )
            .is_err()
        );
    }

    #[test]
    fn topology_and_frequency_mismatches_are_rejected() {
        let (output, measured) = fixture(FrequencyProvenance::Solved);
        let mut wrong_frequency = measured.clone();
        wrong_frequency.frequencies[1] += 0.1;
        assert!(compare_solved_output_to_measured(&output, &wrong_frequency, 0.01).is_err());

        let mut wrong_source = measured;
        wrong_source.source_names[0] = "right".into();
        assert!(compare_solved_output_to_measured(&output, &wrong_source, 0.0).is_err());
    }

    #[test]
    fn comparison_budgets_reject_invalid_or_excessive_error() {
        let (output, mut measured) = fixture(FrequencyProvenance::Solved);
        measured.pressures[0][0][0] *= 2.0;
        let report = compare_solved_output_to_measured(&output, &measured, 0.0).unwrap();
        assert!(report.require_within(-1.0, 1.0).is_err());
        assert!(report.require_within(0.01, 1.0).is_err());
        assert!(report.require_within(10.0, -1.0).is_err());

        measured.pressures[0][0][0] *= Complex64::from_polar(1.0, 0.1);
        let phase_report = compare_solved_output_to_measured(&output, &measured, 0.0).unwrap();
        assert!(phase_report.require_within(10.0, 0.0).is_err());

        let invalid_report = MeasuredComparisonReport {
            sample_count: 0,
            phase_sample_count: 0,
            rms_magnitude_error_db: f64::NAN,
            max_abs_magnitude_error_db: 0.0,
            rms_phase_error_rad: 0.0,
            max_abs_phase_error_rad: 0.0,
        };
        assert!(invalid_report.require_within(1.0, 1.0).is_err());
    }
}
