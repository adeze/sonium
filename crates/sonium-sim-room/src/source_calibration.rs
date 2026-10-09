//! Independent source calibration equations used by RoomSim references.
//!
//! The FEM right-hand side uses a prescribed volume velocity.  Keeping the
//! corresponding free-field relation in a small, validated module gives QA
//! binaries and tests one physical contract to share without treating a
//! synthetic reference as measured data.

use crate::energy_validation::FemPhysicalPowerEvidence;
use crate::room_metrics::peak_pressure_to_spl;
use std::collections::BTreeSet;

/// Volume velocity represented by a unit RoomSim source amplitude (m³/s).
///
/// This value gives approximately 85 dB SPL at 1 m and 1 kHz in 20 °C air
/// for an omnidirectional monopole.  It is a dataset calibration, not a
/// claim about a particular loudspeaker.
pub const SOURCE_VOLUME_VELOCITY_REFERENCE: f64 = 5.88e-4;

/// One measured free-field calibration observation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationObservation {
    /// Observation frequency in Hz.
    pub frequency_hz: f64,
    /// Source-to-microphone distance in metres.
    pub distance_m: f64,
    /// Measured RMS SPL in dB re 20 µPa.
    pub measured_spl_db: f64,
}

/// Result of fitting one source amplitude to a measured calibration sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceCalibrationFit {
    /// Positive dimensionless RoomSim source amplitude.
    pub source_amplitude: f64,
    /// RMS residual in dB over all observations.
    pub rms_error_db: f64,
    /// Maximum absolute residual in dB over all observations.
    pub max_abs_error_db: f64,
    /// Number of observations used by the fit.
    pub observation_count: usize,
}

/// One FEM source-power observation used to make source amplitude independent
/// of mesh and frequency.
///
/// `source_power_w` must be the physically normalized
/// [`FemPhysicalPowerEvidence::source_power_w`] for the same source amplitude,
/// not a pressure magnitude or `|p|²/(rho*c)` estimate. The calibration uses
/// the linear-acoustics scaling `P(a) = P(a₀) (a/a₀)²`.
#[derive(Debug, Clone, PartialEq)]
pub struct SourcePowerCalibrationObservation {
    /// Positive frequency associated with the FEM solve, in Hz.
    pub frequency_hz: f64,
    /// Stable mesh identifier retained in diagnostics and duplicate checks.
    pub mesh_id: String,
    /// Positive scalar multiplier applied to the source RHS for this solve.
    pub source_amplitude: f64,
    /// Cycle-average source power from FEM physical-power evidence, in watts.
    pub source_power_w: f64,
}

impl SourcePowerCalibrationObservation {
    /// Construct an observation directly from accepted FEM physical-power
    /// evidence.
    pub fn from_physical_power(
        frequency_hz: f64,
        mesh_id: impl Into<String>,
        source_amplitude: f64,
        evidence: &FemPhysicalPowerEvidence,
    ) -> Self {
        Self {
            frequency_hz,
            mesh_id: mesh_id.into(),
            source_amplitude,
            source_power_w: evidence.source_power_w,
        }
    }
}

/// Result of normalizing source amplitudes to one target active power.
#[derive(Debug, Clone, PartialEq)]
pub struct SourcePowerCalibrationReport {
    /// Desired cycle-average source power, in watts.
    pub target_power_w: f64,
    /// Corrected source amplitudes in the same order as the observations.
    pub calibrated_source_amplitudes: Vec<f64>,
    /// RMS relative input-power error against `target_power_w`.
    pub input_relative_power_rms: f64,
    /// Maximum absolute relative input-power error against the target.
    pub input_max_abs_relative_power_error: f64,
    /// RMS relative power error predicted after quadratic amplitude scaling.
    pub calibrated_relative_power_rms: f64,
    /// Maximum absolute predicted post-calibration relative power error.
    pub calibrated_max_abs_relative_power_error: f64,
    /// Number of mesh/frequency observations normalized.
    pub observation_count: usize,
}

impl SourcePowerCalibrationReport {
    /// Require all input observations to lie within a relative power budget.
    pub fn require_input_power_budget(&self, maximum_relative_error: f64) -> Result<(), String> {
        if !maximum_relative_error.is_finite() || maximum_relative_error < 0.0 {
            return Err("source-power relative budget must be finite and non-negative".into());
        }
        if self.input_max_abs_relative_power_error > maximum_relative_error {
            return Err(format!(
                "source-power input spread {} exceeds budget {maximum_relative_error}",
                self.input_max_abs_relative_power_error
            ));
        }
        Ok(())
    }
}

/// Normalize FEM source amplitudes to a mesh/frequency-independent active
/// power target.
///
/// This is valid for a linear, stationary FEM solve with unchanged source
/// shape, boundary/material state, and phasor convention. It is intentionally
/// fail-closed for malformed or duplicate mesh/frequency observations. The
/// report retains the pre-calibration spread so callers cannot mistake the
/// algebraic correction for proof that a mesh-convergence gate passed.
pub fn calibrate_source_power(
    observations: &[SourcePowerCalibrationObservation],
    target_power_w: f64,
) -> Result<SourcePowerCalibrationReport, String> {
    if observations.len() < 2 {
        return Err("source-power calibration requires at least two observations".into());
    }
    if !target_power_w.is_finite() || target_power_w <= 0.0 {
        return Err("source-power calibration target must be finite and positive".into());
    }

    let mut keys = BTreeSet::new();
    let mut input_errors = Vec::with_capacity(observations.len());
    let mut calibrated_source_amplitudes = Vec::with_capacity(observations.len());
    let mut calibrated_errors = Vec::with_capacity(observations.len());

    for observation in observations {
        if !observation.frequency_hz.is_finite() || observation.frequency_hz <= 0.0 {
            return Err("source-power calibration frequencies must be finite and positive".into());
        }
        if observation.mesh_id.trim().is_empty() {
            return Err("source-power calibration mesh identifiers must be non-empty".into());
        }
        if !observation.source_amplitude.is_finite() || observation.source_amplitude <= 0.0 {
            return Err("source-power calibration amplitudes must be finite and positive".into());
        }
        if !observation.source_power_w.is_finite() || observation.source_power_w <= 0.0 {
            return Err("source-power calibration powers must be finite and positive".into());
        }
        if !keys.insert((
            observation.frequency_hz.to_bits(),
            observation.mesh_id.clone(),
        )) {
            return Err(
                "source-power calibration contains duplicate mesh/frequency samples".into(),
            );
        }

        let input_relative_error = observation.source_power_w / target_power_w - 1.0;
        if !input_relative_error.is_finite() {
            return Err("source-power input relative error is non-finite".into());
        }
        input_errors.push(input_relative_error);

        let scale = (target_power_w / observation.source_power_w).sqrt();
        let calibrated_amplitude = observation.source_amplitude * scale;
        if !scale.is_finite() || !calibrated_amplitude.is_finite() || calibrated_amplitude <= 0.0 {
            return Err("source-power calibrated amplitude is non-finite".into());
        }
        calibrated_source_amplitudes.push(calibrated_amplitude);

        let calibrated_power = observation.source_power_w * scale * scale;
        let calibrated_relative_error = calibrated_power / target_power_w - 1.0;
        if !calibrated_relative_error.is_finite() {
            return Err("source-power calibrated relative error is non-finite".into());
        }
        calibrated_errors.push(calibrated_relative_error);
    }

    let rms = |errors: &[f64]| {
        (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt()
    };
    let input_relative_power_rms = rms(&input_errors);
    let input_max_abs_relative_power_error = input_errors
        .iter()
        .map(|error| error.abs())
        .fold(0.0, f64::max);
    let calibrated_relative_power_rms = rms(&calibrated_errors);
    let calibrated_max_abs_relative_power_error = calibrated_errors
        .iter()
        .map(|error| error.abs())
        .fold(0.0, f64::max);
    if [
        input_relative_power_rms,
        input_max_abs_relative_power_error,
        calibrated_relative_power_rms,
        calibrated_max_abs_relative_power_error,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err("source-power calibration report contains non-finite metrics".into());
    }

    Ok(SourcePowerCalibrationReport {
        target_power_w,
        calibrated_source_amplitudes,
        input_relative_power_rms,
        input_max_abs_relative_power_error,
        calibrated_relative_power_rms,
        calibrated_max_abs_relative_power_error,
        observation_count: observations.len(),
    })
}

/// Return the free-field peak pressure of an omnidirectional monopole.
///
/// The convention is the frequency-domain peak pressure used by RoomSim:
/// `|p| = omega * rho * |Q| / (4 pi r)`.  Inputs are checked so callers do
/// not accidentally turn a malformed reference into a passing NaN metric.
pub fn free_field_peak_pressure(
    frequency_hz: f64,
    sound_speed: f64,
    density: f64,
    volume_velocity: f64,
    distance_m: f64,
) -> Result<f64, String> {
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err("frequency must be finite and positive".into());
    }
    if !sound_speed.is_finite() || sound_speed <= 0.0 {
        return Err("sound speed must be finite and positive".into());
    }
    if !density.is_finite() || density <= 0.0 {
        return Err("density must be finite and positive".into());
    }
    if !volume_velocity.is_finite() {
        return Err("volume velocity must be finite".into());
    }
    if !distance_m.is_finite() || distance_m <= 0.0 {
        return Err("distance must be finite and positive".into());
    }

    let pressure = 2.0 * std::f64::consts::PI * frequency_hz * density * volume_velocity.abs()
        / (4.0 * std::f64::consts::PI * distance_m);
    if pressure.is_finite() {
        Ok(pressure)
    } else {
        Err("free-field pressure is non-finite".into())
    }
}

/// Return the free-field pressure for a dimensionless RoomSim source value.
pub fn free_field_peak_pressure_for_amplitude(
    frequency_hz: f64,
    sound_speed: f64,
    density: f64,
    source_amplitude: f64,
    distance_m: f64,
) -> Result<f64, String> {
    if !source_amplitude.is_finite() {
        return Err("source amplitude must be finite".into());
    }
    free_field_peak_pressure(
        frequency_hz,
        sound_speed,
        density,
        source_amplitude * SOURCE_VOLUME_VELOCITY_REFERENCE,
        distance_m,
    )
}

/// Calibrate a unit source amplitude to a target SPL at one reference point.
pub fn source_amplitude_for_spl(
    target_spl_db: f64,
    frequency_hz: f64,
    sound_speed: f64,
    density: f64,
    distance_m: f64,
) -> Result<f64, String> {
    if !target_spl_db.is_finite() {
        return Err("target SPL must be finite".into());
    }
    let unit_pressure = free_field_peak_pressure_for_amplitude(
        frequency_hz,
        sound_speed,
        density,
        1.0,
        distance_m,
    )?;
    let target_peak = crate::room_metrics::spl_to_peak_pressure(target_spl_db, 0.0)
        .map_err(|error| format!("invalid target SPL: {error}"))?
        .norm();
    let amplitude = target_peak / unit_pressure;
    if amplitude.is_finite() {
        Ok(amplitude)
    } else {
        Err("calibrated source amplitude is non-finite".into())
    }
}

/// Evaluate the calibrated source at a reference point and return SPL.
pub fn calibrated_spl_at_distance(
    frequency_hz: f64,
    sound_speed: f64,
    density: f64,
    source_amplitude: f64,
    distance_m: f64,
) -> Result<f64, String> {
    let pressure = free_field_peak_pressure_for_amplitude(
        frequency_hz,
        sound_speed,
        density,
        source_amplitude,
        distance_m,
    )?;
    Ok(peak_pressure_to_spl(num_complex::Complex64::new(
        pressure, 0.0,
    )))
}

/// Fit a source amplitude to a measured free-field SPL sweep.
///
/// The fit is a positive least-squares scale in peak-pressure space.  This
/// keeps the calibration independent of the presentation-unit conversion:
/// each measured RMS SPL is converted to peak pressure, compared with the
/// independently computed unit-amplitude monopole pressure, and the fitted
/// amplitude is then scored back in dB.  The observations must be ordered by
/// a strictly increasing frequency grid so the resulting contract can be
/// joined to a [`SimulationOutput`] without an implicit reordering.
pub fn fit_source_amplitude_from_spl(
    observations: &[CalibrationObservation],
    sound_speed: f64,
    density: f64,
) -> Result<SourceCalibrationFit, String> {
    if observations.len() < 2 {
        return Err("source calibration requires at least two observations".into());
    }
    if !sound_speed.is_finite() || sound_speed <= 0.0 {
        return Err("sound speed must be finite and positive".into());
    }
    if !density.is_finite() || density <= 0.0 {
        return Err("density must be finite and positive".into());
    }
    if observations.windows(2).any(|pair| {
        !pair[0].frequency_hz.is_finite()
            || !pair[1].frequency_hz.is_finite()
            || pair[1].frequency_hz <= pair[0].frequency_hz
    }) {
        return Err("calibration frequencies must be finite and strictly increasing".into());
    }

    let mut unit_pressures = Vec::with_capacity(observations.len());
    let mut measured_pressures = Vec::with_capacity(observations.len());
    for observation in observations {
        if !observation.frequency_hz.is_finite() || observation.frequency_hz <= 0.0 {
            return Err("calibration frequencies must be finite and positive".into());
        }
        if !observation.measured_spl_db.is_finite() {
            return Err("measured SPL must be finite".into());
        }
        let unit_pressure = free_field_peak_pressure_for_amplitude(
            observation.frequency_hz,
            sound_speed,
            density,
            1.0,
            observation.distance_m,
        )?;
        let measured_pressure =
            crate::room_metrics::spl_to_peak_pressure(observation.measured_spl_db, 0.0)?.norm();
        if !measured_pressure.is_finite() || measured_pressure <= 0.0 {
            return Err("measured SPL maps to an invalid pressure".into());
        }
        unit_pressures.push(unit_pressure);
        measured_pressures.push(measured_pressure);
    }

    let denominator = unit_pressures
        .iter()
        .map(|pressure| pressure * pressure)
        .sum::<f64>();
    let numerator = unit_pressures
        .iter()
        .zip(&measured_pressures)
        .map(|(unit, measured)| unit * measured)
        .sum::<f64>();
    if !denominator.is_finite() || denominator <= 0.0 || !numerator.is_finite() {
        return Err("calibration fit has no finite pressure energy".into());
    }
    let source_amplitude = (numerator / denominator).max(0.0);
    if !source_amplitude.is_finite() || source_amplitude <= 0.0 {
        return Err("calibrated source amplitude is not positive and finite".into());
    }

    let errors = observations
        .iter()
        .zip(&measured_pressures)
        .map(|(observation, measured)| {
            let predicted = free_field_peak_pressure_for_amplitude(
                observation.frequency_hz,
                sound_speed,
                density,
                source_amplitude,
                observation.distance_m,
            )?;
            let predicted_spl = peak_pressure_to_spl(num_complex::Complex64::new(predicted, 0.0));
            let measured_spl = peak_pressure_to_spl(num_complex::Complex64::new(*measured, 0.0));
            let error = predicted_spl - measured_spl;
            if error.is_finite() {
                Ok(error)
            } else {
                Err("calibration residual is non-finite".to_string())
            }
        })
        .collect::<Result<Vec<_>, String>>()?;
    let rms_error_db =
        (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt();
    let max_abs_error_db = errors.iter().map(|error| error.abs()).fold(0.0, f64::max);
    if !rms_error_db.is_finite() || !max_abs_error_db.is_finite() {
        return Err("calibration residual metrics are non-finite".into());
    }

    Ok(SourceCalibrationFit {
        source_amplitude,
        rms_error_db,
        max_abs_error_db,
        observation_count: observations.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_xem::constants::AIR_DENSITY_20C;

    #[test]
    fn monopole_reference_scales_with_frequency_and_inverse_distance() {
        let p_1m =
            free_field_peak_pressure_for_amplitude(1_000.0, 343.0, AIR_DENSITY_20C, 1.0, 1.0)
                .unwrap();
        let p_2m =
            free_field_peak_pressure_for_amplitude(2_000.0, 343.0, AIR_DENSITY_20C, 1.0, 2.0)
                .unwrap();
        assert!((p_1m / p_2m - 1.0).abs() < 1.0e-12);
        let p_4m =
            free_field_peak_pressure_for_amplitude(1_000.0, 343.0, AIR_DENSITY_20C, 1.0, 4.0)
                .unwrap();
        assert!((p_1m / p_4m - 4.0).abs() < 1.0e-12);
    }

    #[test]
    fn source_amplitude_calibration_round_trips_target_spl() {
        let amplitude =
            source_amplitude_for_spl(85.0, 1_000.0, 343.0, AIR_DENSITY_20C, 1.0).unwrap();
        let spl =
            calibrated_spl_at_distance(1_000.0, 343.0, AIR_DENSITY_20C, amplitude, 1.0).unwrap();
        assert!((spl - 85.0).abs() < 1.0e-12);
    }

    #[test]
    fn measured_sweep_fit_recovers_source_amplitude() {
        let known_amplitude = 1.75;
        let observations = [(250.0, 0.75), (500.0, 1.0), (1_000.0, 1.5), (2_000.0, 2.0)]
            .into_iter()
            .map(|(frequency_hz, distance_m)| CalibrationObservation {
                frequency_hz,
                distance_m,
                measured_spl_db: calibrated_spl_at_distance(
                    frequency_hz,
                    343.0,
                    AIR_DENSITY_20C,
                    known_amplitude,
                    distance_m,
                )
                .unwrap(),
            })
            .collect::<Vec<_>>();

        let fit = fit_source_amplitude_from_spl(&observations, 343.0, AIR_DENSITY_20C).unwrap();
        assert!((fit.source_amplitude - known_amplitude).abs() < 1.0e-12);
        assert!(fit.rms_error_db < 1.0e-12);
        assert!(fit.max_abs_error_db < 1.0e-12);
        assert_eq!(fit.observation_count, observations.len());
    }

    #[test]
    fn measured_sweep_fit_reports_noise_without_hiding_it() {
        let observations = [
            CalibrationObservation {
                frequency_hz: 500.0,
                distance_m: 1.0,
                measured_spl_db: calibrated_spl_at_distance(
                    500.0,
                    343.0,
                    AIR_DENSITY_20C,
                    1.0,
                    1.0,
                )
                .unwrap()
                    + 1.0,
            },
            CalibrationObservation {
                frequency_hz: 1_000.0,
                distance_m: 1.0,
                measured_spl_db: calibrated_spl_at_distance(
                    1_000.0,
                    343.0,
                    AIR_DENSITY_20C,
                    1.0,
                    1.0,
                )
                .unwrap()
                    - 1.0,
            },
        ];

        let fit = fit_source_amplitude_from_spl(&observations, 343.0, AIR_DENSITY_20C).unwrap();
        assert!(fit.rms_error_db > 0.5);
        assert!(fit.max_abs_error_db > 0.5);
    }

    #[test]
    fn measured_sweep_fit_rejects_malformed_observations() {
        let valid = CalibrationObservation {
            frequency_hz: 500.0,
            distance_m: 1.0,
            measured_spl_db: 85.0,
        };
        assert!(fit_source_amplitude_from_spl(&[valid], 343.0, AIR_DENSITY_20C).is_err());
        assert!(
            fit_source_amplitude_from_spl(
                &[
                    valid,
                    CalibrationObservation {
                        frequency_hz: 400.0,
                        ..valid
                    },
                ],
                343.0,
                AIR_DENSITY_20C,
            )
            .is_err()
        );
        assert!(
            fit_source_amplitude_from_spl(
                &[
                    valid,
                    CalibrationObservation {
                        frequency_hz: 1_000.0,
                        measured_spl_db: f64::NAN,
                        ..valid
                    },
                ],
                343.0,
                AIR_DENSITY_20C,
            )
            .is_err()
        );
    }

    #[test]
    fn calibration_rejects_invalid_physical_inputs() {
        for args in [
            (0.0, 343.0, AIR_DENSITY_20C, 1.0, 1.0),
            (1_000.0, 0.0, AIR_DENSITY_20C, 1.0, 1.0),
            (1_000.0, 343.0, 0.0, 1.0, 1.0),
            (1_000.0, 343.0, AIR_DENSITY_20C, 1.0, 0.0),
        ] {
            assert!(
                free_field_peak_pressure_for_amplitude(args.0, args.1, args.2, args.3, args.4)
                    .is_err()
            );
        }
        assert!(source_amplitude_for_spl(f64::NAN, 1_000.0, 343.0, AIR_DENSITY_20C, 1.0).is_err());
    }

    #[test]
    fn source_power_calibration_corrects_mesh_frequency_power_spread() {
        let observations = vec![
            SourcePowerCalibrationObservation {
                frequency_hz: 100.0,
                mesh_id: "coarse".into(),
                source_amplitude: 1.0,
                source_power_w: 0.25,
            },
            SourcePowerCalibrationObservation {
                frequency_hz: 100.0,
                mesh_id: "fine".into(),
                source_amplitude: 2.0,
                source_power_w: 4.0,
            },
        ];
        let report = calibrate_source_power(&observations, 1.0).unwrap();
        assert_eq!(report.observation_count, 2);
        assert_eq!(report.calibrated_source_amplitudes, vec![2.0, 1.0]);
        let expected_input_rms = ((0.75_f64.powi(2) + 3.0_f64.powi(2)) / 2.0).sqrt();
        assert!((report.input_relative_power_rms - expected_input_rms).abs() < 1.0e-12);
        assert!((report.input_max_abs_relative_power_error - 3.0).abs() < 1.0e-12);
        assert!(report.calibrated_relative_power_rms < 1.0e-15);
        assert!(report.calibrated_max_abs_relative_power_error < 1.0e-15);
        assert!(report.require_input_power_budget(3.0).is_ok());
        assert!(report.require_input_power_budget(0.5).is_err());
    }

    #[test]
    fn source_power_observation_can_use_fem_physical_power_evidence() {
        let evidence = FemPhysicalPowerEvidence {
            source_power_w: 2.5,
            volume_damping_power_w: 1.0,
            boundary_power_w: 1.5,
            balance_residual_w: 0.0,
        };
        let observation =
            SourcePowerCalibrationObservation::from_physical_power(250.0, "mesh-a", 1.0, &evidence);
        assert_eq!(observation.source_power_w, evidence.source_power_w);
        assert_eq!(observation.mesh_id, "mesh-a");
    }

    #[test]
    fn source_power_calibration_rejects_malformed_observations() {
        let valid = SourcePowerCalibrationObservation {
            frequency_hz: 100.0,
            mesh_id: "mesh".into(),
            source_amplitude: 1.0,
            source_power_w: 1.0,
        };
        assert!(calibrate_source_power(&[valid.clone()], 1.0).is_err());
        assert!(calibrate_source_power(&[valid.clone(), valid.clone()], 1.0).is_err());
        assert!(calibrate_source_power(&[valid.clone(), valid.clone()], 0.0).is_err());

        for invalid in [
            SourcePowerCalibrationObservation {
                frequency_hz: f64::NAN,
                ..valid.clone()
            },
            SourcePowerCalibrationObservation {
                mesh_id: " ".into(),
                ..valid.clone()
            },
            SourcePowerCalibrationObservation {
                source_amplitude: 0.0,
                ..valid.clone()
            },
            SourcePowerCalibrationObservation {
                source_power_w: f64::INFINITY,
                ..valid.clone()
            },
        ] {
            assert!(calibrate_source_power(&[invalid, valid.clone()], 1.0).is_err());
        }

        let report = calibrate_source_power(
            &[
                SourcePowerCalibrationObservation {
                    mesh_id: "a".into(),
                    ..valid.clone()
                },
                SourcePowerCalibrationObservation {
                    frequency_hz: 200.0,
                    mesh_id: "b".into(),
                    ..valid
                },
            ],
            1.0,
        )
        .unwrap();
        assert!(report.require_input_power_budget(f64::NAN).is_err());
    }
}
