//! Reference correlation for the bounded nonlinear distortion path.
//!
//! The correlator deliberately targets the existing vented M2 compression
//! sweep.  It is a report-only boundary for independently supplied drive-level
//! observations; it does not promote the lumped circuit to a distributed
//! nonlinear acoustic or product-scale convergence result.

use serde::{Deserialize, Serialize};

use crate::{LargeSignalSpec, ModelingError, SpeakerDriverStudy};

/// One independently supplied acoustic distortion observation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeasuredDistortionPoint {
    pub frequency_hz: f64,
    pub secondary_frequency_hz: Option<f64>,
    pub voltage_rms_v: f64,
    pub acoustic_pressure_fundamental_peak_pa: f64,
    pub acoustic_thd: f64,
    pub acoustic_imd: f64,
}

/// Separate RMSE and worst-case gates for each distortion observable.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DistortionReferenceTolerances {
    pub pressure_rmse_pa: f64,
    pub pressure_max_pa: f64,
    pub thd_rmse: f64,
    pub thd_max: f64,
    pub imd_rmse: f64,
    pub imd_max: f64,
}

impl Default for DistortionReferenceTolerances {
    fn default() -> Self {
        Self {
            pressure_rmse_pa: 0.05,
            pressure_max_pa: 0.1,
            thd_rmse: 0.01,
            thd_max: 0.02,
            imd_rmse: 0.01,
            imd_max: 0.02,
        }
    }
}

impl DistortionReferenceTolerances {
    fn validate(self) -> Result<(), ModelingError> {
        let values = [
            ("distortion.pressure_rmse_pa", self.pressure_rmse_pa),
            ("distortion.pressure_max_pa", self.pressure_max_pa),
            ("distortion.thd_rmse", self.thd_rmse),
            ("distortion.thd_max", self.thd_max),
            ("distortion.imd_rmse", self.imd_rmse),
            ("distortion.imd_max", self.imd_max),
        ];
        for (field, value) in values {
            if !value.is_finite() || value < 0.0 {
                return Err(ModelingError::InvalidReferenceEvidence(format!(
                    "{field} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }
}

/// Correlation statistics for one distortion observable.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DistortionMetricReport {
    pub samples: usize,
    pub rmse: f64,
    pub maximum_absolute_error: f64,
    pub rmse_tolerance: f64,
    pub maximum_tolerance: f64,
    pub passed: bool,
}

/// Complete distortion/IMD reference-correlation report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistortionReferenceReport {
    pub pressure: DistortionMetricReport,
    pub thd: DistortionMetricReport,
    pub imd: DistortionMetricReport,
    pub passed: bool,
}

/// Correlate the production vented M2 compression sweep with supplied data.
pub fn correlate_distortion_reference(
    study: &SpeakerDriverStudy,
    base_spec: LargeSignalSpec,
    reference: &[MeasuredDistortionPoint],
    tolerances: DistortionReferenceTolerances,
) -> Result<DistortionReferenceReport, ModelingError> {
    if reference.is_empty() {
        return Err(ModelingError::InvalidReferenceEvidence(
            "distortion reference must contain at least one drive level".to_owned(),
        ));
    }
    base_spec.validate()?;
    tolerances.validate()?;

    validate_reference_grid(base_spec, reference)?;
    let voltages: Vec<_> = reference.iter().map(|point| point.voltage_rms_v).collect();
    let sweep = study.try_lumped_nonlinear_compression_sweep(base_spec, &voltages)?;
    if sweep.points.len() != reference.len() {
        return Err(ModelingError::InvalidReferenceEvidence(
            "distortion sweep and reference cardinalities differ".to_owned(),
        ));
    }

    let pressure_actual: Vec<_> = sweep
        .points
        .iter()
        .map(|point| point.box_pressure_amplitude_pa)
        .collect();
    let pressure_expected: Vec<_> = reference
        .iter()
        .map(|point| point.acoustic_pressure_fundamental_peak_pa)
        .collect();
    let thd_actual: Vec<_> = sweep
        .points
        .iter()
        .map(|point| point.acoustic_thd)
        .collect();
    let thd_expected: Vec<_> = reference.iter().map(|point| point.acoustic_thd).collect();
    let imd_actual: Vec<_> = sweep
        .points
        .iter()
        .map(|point| point.acoustic_imd)
        .collect();
    let imd_expected: Vec<_> = reference.iter().map(|point| point.acoustic_imd).collect();

    let pressure = metric_report(
        &pressure_actual,
        &pressure_expected,
        tolerances.pressure_rmse_pa,
        tolerances.pressure_max_pa,
    );
    let thd = metric_report(
        &thd_actual,
        &thd_expected,
        tolerances.thd_rmse,
        tolerances.thd_max,
    );
    let imd = metric_report(
        &imd_actual,
        &imd_expected,
        tolerances.imd_rmse,
        tolerances.imd_max,
    );
    Ok(DistortionReferenceReport {
        passed: pressure.passed && thd.passed && imd.passed,
        pressure,
        thd,
        imd,
    })
}

fn validate_reference_grid(
    base_spec: LargeSignalSpec,
    reference: &[MeasuredDistortionPoint],
) -> Result<(), ModelingError> {
    let mut previous_voltage = None;
    for (index, point) in reference.iter().enumerate() {
        if !point.frequency_hz.is_finite()
            || point.frequency_hz <= 0.0
            || !same_frequency(point.frequency_hz, base_spec.frequency_hz)
        {
            return Err(ModelingError::InvalidReferenceEvidence(format!(
                "distortion reference sample {index} has a frequency different from the base spec"
            )));
        }
        if point.secondary_frequency_hz != base_spec.secondary_frequency_hz {
            return Err(ModelingError::InvalidReferenceEvidence(format!(
                "distortion reference sample {index} has a secondary frequency different from the base spec"
            )));
        }
        if !point.voltage_rms_v.is_finite() || point.voltage_rms_v <= 0.0 {
            return Err(ModelingError::InvalidReferenceEvidence(format!(
                "distortion reference sample {index} has a non-positive voltage"
            )));
        }
        if previous_voltage.is_some_and(|previous| point.voltage_rms_v <= previous) {
            return Err(ModelingError::InvalidReferenceEvidence(
                "distortion reference voltages must be strictly increasing".to_owned(),
            ));
        }
        previous_voltage = Some(point.voltage_rms_v);

        let observations = [
            point.acoustic_pressure_fundamental_peak_pa,
            point.acoustic_thd,
            point.acoustic_imd,
        ];
        if observations
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(ModelingError::InvalidReferenceEvidence(format!(
                "distortion reference sample {index} contains a non-finite or negative observable"
            )));
        }
    }
    Ok(())
}

fn same_frequency(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-9 * left.max(right).max(1.0)
}

fn metric_report(
    actual: &[f64],
    expected: &[f64],
    rmse_tolerance: f64,
    maximum_tolerance: f64,
) -> DistortionMetricReport {
    let errors: Vec<_> = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .collect();
    let rmse = (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt();
    let maximum_absolute_error = errors.iter().copied().fold(0.0, f64::max);
    DistortionMetricReport {
        samples: errors.len(),
        rmse,
        maximum_absolute_error,
        rmse_tolerance,
        maximum_tolerance,
        passed: rmse <= rmse_tolerance && maximum_absolute_error <= maximum_tolerance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EnclosureSpec, LumpedNonlinearCompressionSweep, PortSpec};

    fn vented_study() -> SpeakerDriverStudy {
        let mut study = SpeakerDriverStudy::transient_large_signal_driver();
        study.enclosure = EnclosureSpec::Vented {
            volume_m3: 35.0e-3,
            port: PortSpec {
                radius_m: 35.0e-3,
                length_m: 0.15,
                end_correction_m: 0.055,
                resistance_pa_s_per_m3: 80.0,
            },
            leakage_resistance_pa_s_per_m3: 3.0e6,
            absorption_resistance_pa_s_per_m3: 2.0e6,
        };
        study
    }

    fn base_spec() -> LargeSignalSpec {
        LargeSignalSpec {
            frequency_hz: 80.0,
            duration_s: 0.1,
            samples_per_period: 64,
            secondary_frequency_hz: Some(1_000.0),
            ..LargeSignalSpec::default()
        }
    }

    fn reference_from_sweep(
        sweep: &LumpedNonlinearCompressionSweep,
        frequency_hz: f64,
        secondary_frequency_hz: Option<f64>,
    ) -> Vec<MeasuredDistortionPoint> {
        sweep
            .points
            .iter()
            .map(|point| MeasuredDistortionPoint {
                frequency_hz,
                secondary_frequency_hz,
                voltage_rms_v: point.voltage_rms_v,
                acoustic_pressure_fundamental_peak_pa: point.box_pressure_amplitude_pa,
                acoustic_thd: point.acoustic_thd,
                acoustic_imd: point.acoustic_imd,
            })
            .collect()
    }

    #[test]
    fn distortion_reference_replays_pressure_thd_and_imd_contract() {
        let study = vented_study();
        let spec = base_spec();
        let voltages = [1.0, 2.0];
        let sweep = study
            .try_lumped_nonlinear_compression_sweep(spec, &voltages)
            .expect("valid vented compression sweep");
        let reference =
            reference_from_sweep(&sweep, spec.frequency_hz, spec.secondary_frequency_hz);

        let report = correlate_distortion_reference(
            &study,
            spec,
            &reference,
            DistortionReferenceTolerances {
                pressure_rmse_pa: 1.0e-10,
                pressure_max_pa: 1.0e-10,
                thd_rmse: 1.0e-12,
                thd_max: 1.0e-12,
                imd_rmse: 1.0e-12,
                imd_max: 1.0e-12,
            },
        )
        .expect("reference correlation should succeed");
        assert!(report.passed);
        assert_eq!(report.pressure.samples, voltages.len());

        let mut mismatch = reference.clone();
        mismatch[1].acoustic_imd += 0.1;
        let report = correlate_distortion_reference(
            &study,
            spec,
            &mismatch,
            DistortionReferenceTolerances {
                pressure_rmse_pa: 1.0e-10,
                pressure_max_pa: 1.0e-10,
                thd_rmse: 1.0e-12,
                thd_max: 1.0e-12,
                imd_rmse: 1.0e-12,
                imd_max: 1.0e-12,
            },
        )
        .expect("mismatched reference should still report");
        assert!(!report.passed);
        assert!(!report.imd.passed);
    }

    #[test]
    fn distortion_reference_rejects_bad_grids_and_tolerances() {
        let study = vented_study();
        let spec = base_spec();
        let sweep = study
            .try_lumped_nonlinear_compression_sweep(spec, &[1.0])
            .expect("valid vented compression sweep");
        let mut reference =
            reference_from_sweep(&sweep, spec.frequency_hz, spec.secondary_frequency_hz);
        reference[0].frequency_hz = 81.0;
        assert!(
            correlate_distortion_reference(
                &study,
                spec,
                &reference,
                DistortionReferenceTolerances::default(),
            )
            .is_err()
        );

        reference[0].frequency_hz = spec.frequency_hz;
        reference[0].acoustic_thd = f64::NAN;
        assert!(
            correlate_distortion_reference(
                &study,
                spec,
                &reference,
                DistortionReferenceTolerances::default(),
            )
            .is_err()
        );

        let mut invalid_tolerance = DistortionReferenceTolerances::default();
        invalid_tolerance.imd_max = -1.0;
        reference[0].acoustic_thd = 0.0;
        assert!(
            correlate_distortion_reference(&study, spec, &reference, invalid_tolerance,).is_err()
        );
    }

    #[test]
    fn distortion_reference_report_round_trips_through_serde() {
        let report = DistortionReferenceReport {
            pressure: DistortionMetricReport {
                samples: 2,
                rmse: 0.01,
                maximum_absolute_error: 0.02,
                rmse_tolerance: 0.1,
                maximum_tolerance: 0.2,
                passed: true,
            },
            thd: DistortionMetricReport {
                samples: 2,
                rmse: 0.001,
                maximum_absolute_error: 0.002,
                rmse_tolerance: 0.01,
                maximum_tolerance: 0.02,
                passed: true,
            },
            imd: DistortionMetricReport {
                samples: 2,
                rmse: 0.003,
                maximum_absolute_error: 0.004,
                rmse_tolerance: 0.01,
                maximum_tolerance: 0.02,
                passed: true,
            },
            passed: true,
        };
        let encoded = serde_json::to_string(&report).expect("serialize report");
        let decoded: DistortionReferenceReport =
            serde_json::from_str(&encoded).expect("deserialize report");
        assert_eq!(decoded, report);
    }
}
