//! Adapter from speaker-product simulation outputs to the modeling contract.
//!
//! The modeling crate intentionally does not depend on the product simulator.
//! This module owns the reverse-direction handoff: it combines impedance
//! traces, CEA-2034 preference, THD measurements, and signed power-compression
//! points into one [`CoupledProductMetrics`] record.

use std::fmt;

use num_complex::Complex64;
use sonium_sim_speaker_modeling::ProductDesign;
use sonium_sim_speaker_modeling::{CoupledProductMetrics, ModelingError, PowerLedgerTolerance};

use crate::distortion::ThdResult;
use crate::power_compression::CompressionPoint;
use crate::simulation::SimulationResult;

/// Failure while adapting product-side evidence to the modeling contract.
#[derive(Debug, Clone, PartialEq)]
pub enum ProductMetricsError {
    InvalidImpedanceShape {
        driver: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidDynamicImpedanceFrequencyShape {
        expected: usize,
        actual: usize,
    },
    InvalidDynamicImpedanceDriverCount {
        frequency_index: usize,
        expected: usize,
        actual: usize,
    },
    InvalidThermalStateShape {
        expected: usize,
        actual: usize,
    },
    InvalidPowerLedgerShape {
        expected: usize,
        actual: usize,
    },
    InvalidPowerLedger {
        index: usize,
        reason: String,
    },
    InvalidVelocityShape {
        expected: usize,
        actual: usize,
    },
    InvalidVelocityDriverCount {
        frequency_index: usize,
        expected: usize,
        actual: usize,
    },
    MissingExcursionEvidence,
    InvalidReferenceEvidence {
        reason: String,
    },
    InvalidReferenceFrequencyShape {
        expected: usize,
        actual: usize,
    },
    ReferenceFrequencyMismatch {
        index: usize,
        simulation_hz: f64,
        reference_hz: f64,
    },
    InvalidFrequency {
        index: usize,
    },
    NonFinite {
        metric: &'static str,
    },
    Negative {
        metric: &'static str,
    },
    InvalidThermalState {
        metric: &'static str,
    },
    PositiveCompression {
        index: usize,
        value: f64,
    },
    Modeling(ModelingError),
}

impl fmt::Display for ProductMetricsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidImpedanceShape {
                driver,
                expected,
                actual,
            } => write!(
                formatter,
                "{driver} impedance length {actual} does not match frequency count {expected}"
            ),
            Self::InvalidDynamicImpedanceFrequencyShape { expected, actual } => write!(
                formatter,
                "dynamic driver impedance row count {actual} does not match frequency count {expected}"
            ),
            Self::InvalidDynamicImpedanceDriverCount {
                frequency_index,
                expected,
                actual,
            } => write!(
                formatter,
                "dynamic driver impedance count {actual} at frequency index {frequency_index} does not match expected driver count {expected}"
            ),
            Self::InvalidThermalStateShape { expected, actual } => write!(
                formatter,
                "thermal state length {actual} does not match frequency count {expected}"
            ),
            Self::InvalidPowerLedgerShape { expected, actual } => write!(
                formatter,
                "power ledger length {actual} does not match frequency count {expected}"
            ),
            Self::InvalidPowerLedger { index, reason } => {
                write!(
                    formatter,
                    "power ledger at frequency index {index} is invalid: {reason}"
                )
            }
            Self::InvalidVelocityShape { expected, actual } => write!(
                formatter,
                "driver velocity length {actual} does not match frequency count {expected}"
            ),
            Self::InvalidVelocityDriverCount {
                frequency_index,
                expected,
                actual,
            } => write!(
                formatter,
                "driver velocity count {actual} at frequency index {frequency_index} does not match expected driver count {expected}"
            ),
            Self::MissingExcursionEvidence => {
                write!(
                    formatter,
                    "product metrics require finite BEM driver-velocity excursion evidence"
                )
            }
            Self::InvalidReferenceEvidence { reason } => {
                write!(
                    formatter,
                    "distributed FEM-BEM reference evidence is invalid: {reason}"
                )
            }
            Self::InvalidReferenceFrequencyShape { expected, actual } => write!(
                formatter,
                "reference sweep length {actual} does not match frequency count {expected}"
            ),
            Self::ReferenceFrequencyMismatch {
                index,
                simulation_hz,
                reference_hz,
            } => write!(
                formatter,
                "reference frequency at index {index} ({reference_hz}) does not match simulation frequency ({simulation_hz})"
            ),
            Self::InvalidFrequency { index } => {
                write!(
                    formatter,
                    "THD frequency at index {index} must be finite and positive"
                )
            }
            Self::NonFinite { metric } => write!(formatter, "{metric} must be finite"),
            Self::Negative { metric } => {
                write!(formatter, "{metric} must be finite and non-negative")
            }
            Self::InvalidThermalState { metric } => {
                write!(
                    formatter,
                    "thermal state {metric} must be finite and non-negative"
                )
            }
            Self::PositiveCompression { index, value } => write!(
                formatter,
                "compression point {index} has positive signed compression {value}; expected a loss <= 0 dB"
            ),
            Self::Modeling(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ProductMetricsError {}

impl From<ModelingError> for ProductMetricsError {
    fn from(error: ModelingError) -> Self {
        Self::Modeling(error)
    }
}

fn validate_dynamic_impedance_shape(
    simulation: &SimulationResult,
    expected_driver_count: Option<usize>,
) -> Result<(), ProductMetricsError> {
    let rows = &simulation.driver_electrical_impedances_ohm;
    if rows.is_empty() {
        return Ok(());
    }
    if rows.len() != simulation.frequencies.len() {
        return Err(ProductMetricsError::InvalidDynamicImpedanceFrequencyShape {
            expected: simulation.frequencies.len(),
            actual: rows.len(),
        });
    }
    let expected_driver_count = expected_driver_count.unwrap_or_else(|| rows[0].len());
    for (frequency_index, row) in rows.iter().enumerate() {
        if row.len() != expected_driver_count {
            return Err(ProductMetricsError::InvalidDynamicImpedanceDriverCount {
                frequency_index,
                expected: expected_driver_count,
                actual: row.len(),
            });
        }
    }
    Ok(())
}

/// Combine speaker simulation evidence into the modeling crate's typed
/// product record.
///
/// `preference_score` is optional so callers can use this adapter in a
/// non-CLI/native pipeline that has not run CEA-2034 scoring yet.  Empty THD
/// and compression slices likewise leave those optional metrics absent; an
/// active product constraint will then fail safe when it is evaluated.
// Sanctioned reader of the deprecated per-driver vectors: the
// prescribed-result fallback below.
#[allow(deprecated)]
pub fn coupled_product_metrics_from_outputs(
    simulation: &SimulationResult,
    preference_score: Option<f64>,
    thd_results: &[ThdResult],
    compression_points: &[CompressionPoint],
) -> Result<CoupledProductMetrics, ProductMetricsError> {
    let frequency_count = simulation.frequencies.len();
    validate_dynamic_impedance_shape(simulation, None)?;
    let maximum_reference_relative_residual =
        if let Some(reference) = &simulation.product_reference_sweep {
            reference.validate().map_err(|error| {
                ProductMetricsError::InvalidReferenceEvidence {
                    reason: error.to_string(),
                }
            })?;
            if reference.points.len() != frequency_count {
                return Err(ProductMetricsError::InvalidReferenceFrequencyShape {
                    expected: frequency_count,
                    actual: reference.points.len(),
                });
            }
            for (index, (simulation_hz, point)) in simulation
                .frequencies
                .iter()
                .zip(&reference.points)
                .enumerate()
            {
                let scale = simulation_hz.abs().max(point.frequency_hz.abs()).max(1.0);
                if !simulation_hz.is_finite()
                    || (simulation_hz - point.frequency_hz).abs() > 1.0e-9 * scale
                {
                    return Err(ProductMetricsError::ReferenceFrequencyMismatch {
                        index,
                        simulation_hz: *simulation_hz,
                        reference_hz: point.frequency_hz,
                    });
                }
            }
            Some(reference.maximum_relative_residual().ok_or_else(|| {
                ProductMetricsError::InvalidReferenceEvidence {
                    reason: "reference sweep contains no residual evidence".to_owned(),
                }
            })?)
        } else {
            None
        };
    if !simulation.power_ledgers.is_empty() && simulation.power_ledgers.len() != frequency_count {
        return Err(ProductMetricsError::InvalidPowerLedgerShape {
            expected: frequency_count,
            actual: simulation.power_ledgers.len(),
        });
    }
    for (index, ledger) in simulation.power_ledgers.iter().enumerate() {
        if let Some(ledger) = ledger {
            ledger
                .close(PowerLedgerTolerance::default())
                .map_err(|error| ProductMetricsError::InvalidPowerLedger {
                    index,
                    reason: error.to_string(),
                })?;
        }
    }
    // Prefer the dynamic per-frequency/per-driver artifact so products with
    // more than two drivers participate in the same impedance constraint. If
    // no typed driver produced an impedance, retain the legacy two-driver
    // fallback for prescribed or synthetic results (the only sanctioned
    // reader of the deprecated per-driver vectors).
    let mut dynamic_minimum = None;
    for row in &simulation.driver_electrical_impedances_ohm {
        for impedance in row.iter().flatten() {
            if !impedance.re.is_finite() || !impedance.im.is_finite() {
                return Err(ProductMetricsError::NonFinite {
                    metric: "electrical impedance",
                });
            }
            if impedance.re < 0.0 {
                return Err(ProductMetricsError::Negative {
                    metric: "electrical impedance real part",
                });
            }
            dynamic_minimum =
                Some(dynamic_minimum.map_or(impedance.re, |value: f64| value.min(impedance.re)));
        }
    }
    let minimum_impedance_real_ohm = if dynamic_minimum.is_some() {
        dynamic_minimum
    } else {
        let woofer_minimum = minimum_real_impedance(
            "woofer",
            &simulation.woofer_electrical_impedance_ohm,
            frequency_count,
        )?;
        let tweeter_minimum = minimum_real_impedance(
            "tweeter",
            &simulation.tweeter_electrical_impedance_ohm,
            frequency_count,
        )?;
        match (woofer_minimum, tweeter_minimum) {
            (Some(woofer), Some(tweeter)) => Some(woofer.min(tweeter)),
            (Some(woofer), None) => Some(woofer),
            (None, Some(tweeter)) => Some(tweeter),
            (None, None) => None,
        }
    };

    if let Some(score) = preference_score
        && !score.is_finite()
    {
        return Err(ProductMetricsError::NonFinite {
            metric: "CEA-2034 preference score",
        });
    }

    let maximum_distortion_fraction = if thd_results.is_empty() {
        None
    } else {
        let mut maximum = 0.0_f64;
        for (index, result) in thd_results.iter().enumerate() {
            if !result.frequency.is_finite() || result.frequency <= 0.0 {
                return Err(ProductMetricsError::InvalidFrequency { index });
            }
            if !result.thd.is_finite() {
                return Err(ProductMetricsError::NonFinite {
                    metric: "THD fraction",
                });
            }
            if result.thd < 0.0 {
                return Err(ProductMetricsError::Negative {
                    metric: "THD fraction",
                });
            }
            maximum = maximum.max(result.thd);
        }
        Some(maximum)
    };

    let thermal_compression_db = if compression_points.is_empty() {
        if !simulation.thermal_driver_states.is_empty()
            && simulation.thermal_driver_states.len() != frequency_count
        {
            return Err(ProductMetricsError::InvalidThermalStateShape {
                expected: frequency_count,
                actual: simulation.thermal_driver_states.len(),
            });
        }
        let mut maximum = 0.0_f64;
        let mut found = false;
        for states in &simulation.thermal_driver_states {
            for state in states.iter().flatten() {
                if !state.compression_db.is_finite() || state.compression_db < 0.0 {
                    return Err(ProductMetricsError::InvalidThermalState {
                        metric: "compression",
                    });
                }
                if let Some(temperatures) = state.thermal_network_temperatures_k
                    && temperatures
                        .iter()
                        .any(|temperature| !temperature.is_finite() || *temperature <= 0.0)
                {
                    return Err(ProductMetricsError::InvalidThermalState {
                        metric: "network temperatures",
                    });
                }
                if let Some(power) = state.thermal_network_power_w
                    && [
                        power.voice_coil_w,
                        power.magnet_w,
                        power.gap_w,
                        power.cabinet_w,
                    ]
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
                {
                    return Err(ProductMetricsError::InvalidThermalState {
                        metric: "network power",
                    });
                }
                if let Some(residual) = state.thermal_network_residual_w
                    && (!residual.is_finite() || residual < 0.0)
                {
                    return Err(ProductMetricsError::InvalidThermalState {
                        metric: "network residual",
                    });
                }
                maximum = maximum.max(state.compression_db);
                found = true;
            }
        }
        if found { Some(maximum) } else { None }
    } else {
        let mut maximum = 0.0_f64;
        for (index, point) in compression_points.iter().enumerate() {
            if !point.compression_db.is_finite() {
                return Err(ProductMetricsError::NonFinite {
                    metric: "power compression",
                });
            }
            if point.compression_db > 0.0 {
                return Err(ProductMetricsError::PositiveCompression {
                    index,
                    value: point.compression_db,
                });
            }
            // The product-side analysis reports loss as a negative dB value;
            // the modeling contract stores the non-negative loss magnitude.
            maximum = maximum.max(-point.compression_db);
        }
        Some(maximum)
    };

    // An allocated per-frequency container is not evidence by itself: test,
    // legacy, and partially populated results may carry one empty driver row
    // per frequency.  Preserve the optional metric until at least one actual
    // BEM velocity phasor is present.
    let peak_excursion_m = if simulation.driver_velocities_bem.is_empty()
        || simulation
            .driver_velocities_bem
            .iter()
            .all(|velocities| velocities.is_empty())
    {
        None
    } else {
        if simulation.driver_velocities_bem.len() != frequency_count {
            return Err(ProductMetricsError::InvalidVelocityShape {
                expected: frequency_count,
                actual: simulation.driver_velocities_bem.len(),
            });
        }
        let mut maximum = 0.0_f64;
        for (index, (frequency, velocities)) in simulation
            .frequencies
            .iter()
            .zip(&simulation.driver_velocities_bem)
            .enumerate()
        {
            if !frequency.is_finite() || *frequency <= 0.0 {
                return Err(ProductMetricsError::InvalidFrequency { index });
            }
            for velocity in velocities {
                if !velocity.re.is_finite() || !velocity.im.is_finite() {
                    return Err(ProductMetricsError::NonFinite {
                        metric: "driver velocity",
                    });
                }
                // The BEM boundary phasor is peak amplitude; displacement is
                // |v| / omega, also peak amplitude.
                maximum = maximum.max(velocity.norm() / (std::f64::consts::TAU * frequency));
            }
        }
        Some(maximum)
    };

    let metrics = CoupledProductMetrics {
        cea_preference_score: preference_score,
        minimum_impedance_real_ohm,
        maximum_reference_relative_residual,
        peak_excursion_m,
        maximum_distortion_fraction,
        thermal_compression_db,
        ..CoupledProductMetrics::default()
    };
    let metrics = metrics
        .with_structural_responses(&simulation.frequencies, &simulation.structural_responses)?;
    metrics.validate()?;
    Ok(metrics)
}

/// Product-aware adapter that additionally evaluates the linear cabinet
/// response when a distributed FEM–BEM reference sweep is attached.
fn validate_product_excursion_evidence(
    simulation: &SimulationResult,
    expected_driver_count: usize,
) -> Result<(), ProductMetricsError> {
    if simulation.driver_velocities_bem.len() != simulation.frequencies.len() {
        return Err(ProductMetricsError::InvalidVelocityShape {
            expected: simulation.frequencies.len(),
            actual: simulation.driver_velocities_bem.len(),
        });
    }
    if simulation.driver_velocities_bem.is_empty()
        || simulation
            .driver_velocities_bem
            .iter()
            .all(|velocities| velocities.is_empty())
    {
        return Err(ProductMetricsError::MissingExcursionEvidence);
    }
    for (frequency_index, velocities) in simulation.driver_velocities_bem.iter().enumerate() {
        if velocities.len() != expected_driver_count {
            return Err(ProductMetricsError::InvalidVelocityDriverCount {
                frequency_index,
                expected: expected_driver_count,
                actual: velocities.len(),
            });
        }
        if velocities
            .iter()
            .any(|velocity| !velocity.re.is_finite() || !velocity.im.is_finite())
        {
            return Err(ProductMetricsError::NonFinite {
                metric: "driver velocity",
            });
        }
    }
    Ok(())
}

pub fn coupled_product_metrics_from_outputs_with_product(
    simulation: &SimulationResult,
    product: &ProductDesign,
    preference_score: Option<f64>,
    thd_results: &[ThdResult],
    compression_points: &[CompressionPoint],
    max_panel_mode_order: usize,
) -> Result<CoupledProductMetrics, ProductMetricsError> {
    validate_dynamic_impedance_shape(simulation, Some(product.drivers.len()))?;
    validate_product_excursion_evidence(simulation, product.drivers.len())?;
    let mut metrics = coupled_product_metrics_from_outputs(
        simulation,
        preference_score,
        thd_results,
        compression_points,
    )?;
    if let Some(reference) = &simulation.product_reference_sweep {
        let panel_response = product
            .reference_cabinet_panel_response(reference, max_panel_mode_order)
            .map_err(|error| ProductMetricsError::InvalidReferenceEvidence {
                reason: error.to_string(),
            })?;
        metrics = metrics
            .with_reference_sweep(reference)
            .and_then(|metrics| metrics.with_reference_panel_response(&panel_response))?;
    }
    Ok(metrics)
}

/// Adapt a CEA-2034 result without making callers manually unpack its score.
#[cfg(feature = "cli")]
pub fn coupled_product_metrics_from_spinorama(
    simulation: &SimulationResult,
    spinorama: &crate::cea2034_output::SpinoramaMetrics,
    thd_results: &[ThdResult],
    compression_points: &[CompressionPoint],
) -> Result<CoupledProductMetrics, ProductMetricsError> {
    coupled_product_metrics_from_outputs(
        simulation,
        Some(spinorama.preference_score),
        thd_results,
        compression_points,
    )
}

fn minimum_real_impedance(
    driver: &'static str,
    values: &[Option<Complex64>],
    frequency_count: usize,
) -> Result<Option<f64>, ProductMetricsError> {
    if !values.is_empty() && values.len() != frequency_count {
        return Err(ProductMetricsError::InvalidImpedanceShape {
            driver,
            expected: frequency_count,
            actual: values.len(),
        });
    }

    let mut minimum = None;
    for impedance in values.iter().flatten() {
        if !impedance.re.is_finite() || !impedance.im.is_finite() {
            return Err(ProductMetricsError::NonFinite {
                metric: "electrical impedance",
            });
        }
        if impedance.re < 0.0 {
            return Err(ProductMetricsError::Negative {
                metric: "electrical impedance real part",
            });
        }
        minimum = Some(minimum.map_or(impedance.re, |value: f64| value.min(impedance.re)));
    }
    Ok(minimum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_compute::HostMatrix;
    use sonium_sim_speaker_modeling::{
        CabinetDesign, CrossoverBranch, CrossoverNetwork, ProductDesign, ProductDriver,
        ProductReferencePoint, ProductReferenceSweep,
    };

    // Fixture fills the deprecated per-driver vectors (prescribed-result
    // shape for the fallback tests below).
    #[allow(deprecated)]
    fn simulation() -> SimulationResult {
        SimulationResult {
            frequencies: vec![100.0, 200.0],
            spl: HostMatrix::zeros(2, 1),
            phase: HostMatrix::zeros(2, 1),
            complex_pressure: HostMatrix::zeros(2, 1),
            woofer_electrical_impedance_ohm: vec![Some(Complex64::new(4.0, 1.0)), Some(3.0.into())],
            tweeter_electrical_impedance_ohm: vec![None, Some(5.0.into())],
            driver_electrical_impedances_ohm: vec![vec![None; 2]; 2],
            radiation_matrices: vec![None; 2],
            radiation_matrix_diagnostics: vec![None; 2],
            thermal_driver_states: vec![[None, None]; 2],
            driver_thermal_states: vec![vec![None; 2]; 2],
            power_ledgers: vec![None; 2],
            product_reference_sweep: None,
            structural_responses: Vec::new(),
            driver_velocities_bem: vec![vec![]; 2],
            exact_residual_evaluations: 0,
            provenance: Default::default(),
        }
    }

    fn reference_sweep() -> ProductReferenceSweep {
        ProductReferenceSweep {
            speed_of_sound_m_per_s: 343.0,
            density_kg_per_m3: 1.21,
            interior_dofs: 12,
            aperture_elements: 8,
            points: vec![
                ProductReferencePoint {
                    frequency_hz: 100.0,
                    drive_velocity_rms_m_per_s: Complex64::new(0.1, 0.02),
                    driver_velocities_rms_m_per_s: Vec::new(),
                    aperture_volume_velocity_m3_per_s: Complex64::new(0.01, 0.0),
                    mean_pressure_rms_pa: Complex64::new(2.0, 0.5),
                    mean_abs_pressure_pa: 2.0,
                    solver_iterations: 4,
                    relative_residual: 1.0e-8,
                    monolithic_evidence: None,
                    interface_evidence: None,
                    thermal_evidence: None,
                    structural_evidence: None,
                },
                ProductReferencePoint {
                    frequency_hz: 200.0,
                    drive_velocity_rms_m_per_s: Complex64::new(0.08, 0.01),
                    driver_velocities_rms_m_per_s: Vec::new(),
                    aperture_volume_velocity_m3_per_s: Complex64::new(0.008, 0.0),
                    mean_pressure_rms_pa: Complex64::new(2.5, 0.25),
                    mean_abs_pressure_pa: 2.5,
                    solver_iterations: 5,
                    relative_residual: 2.0e-8,
                    monolithic_evidence: None,
                    interface_evidence: None,
                    thermal_evidence: None,
                    structural_evidence: None,
                },
            ],
            driver_count: 1,
            rom_promotion: None,
            provenance: Default::default(),
            mesh_convergence: Vec::new(),
        }
    }

    fn centred_product_design() -> ProductDesign {
        ProductDesign {
            cabinet: {
                let mut cabinet = CabinetDesign::default();
                cabinet.port = None;
                cabinet
            },
            drivers: vec![ProductDriver {
                id: "reference-driver".to_owned(),
                radius_m: 0.025,
                position_m: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                force_factor_tm: 5.0,
                coil_resistance_ohm: 5.0,
                coil_reactance_ohm: 1.0,
                mechanical_resistance_n_s_per_m: 0.5,
                mechanical_reactance_n_s_per_m: 2.0,
                crossover_branch: 0,
            }],
            crossover: CrossoverNetwork {
                branches: vec![CrossoverBranch::default()],
            },
            waveguide: None,
            structural_acoustic: None,
        }
    }

    #[test]
    fn maps_signed_product_outputs_and_keeps_worst_values() {
        let metrics = coupled_product_metrics_from_outputs(
            &simulation(),
            Some(6.5),
            &[
                ThdResult {
                    frequency: 100.0,
                    voltage_amplitude: 1.0,
                    fundamental: 1.0,
                    harmonics: vec![],
                    thd: 0.02,
                    thd_db: -33.9,
                    harmonic_db: vec![],
                },
                ThdResult {
                    frequency: 200.0,
                    voltage_amplitude: 1.0,
                    fundamental: 1.0,
                    harmonics: vec![],
                    thd: 0.04,
                    thd_db: -27.9,
                    harmonic_db: vec![],
                },
            ],
            &[
                CompressionPoint {
                    nominal_power: 10.0,
                    actual_power: 9.0,
                    coil_temperature: 330.0,
                    resistance_hot: 7.0,
                    compression_db: -1.5,
                },
                CompressionPoint {
                    nominal_power: 20.0,
                    actual_power: 16.0,
                    coil_temperature: 350.0,
                    resistance_hot: 8.0,
                    compression_db: -3.0,
                },
            ],
        )
        .unwrap();

        assert_eq!(metrics.cea_preference_score, Some(6.5));
        assert_eq!(metrics.minimum_impedance_real_ohm, Some(3.0));
        assert_eq!(metrics.maximum_distortion_fraction, Some(0.04));
        assert_eq!(metrics.thermal_compression_db, Some(3.0));
    }

    #[test]
    fn dynamic_driver_impedance_artifacts_cover_n_port_products() {
        let mut result = simulation();
        result.driver_electrical_impedances_ohm = vec![
            vec![
                Some(Complex64::new(2.0, 0.0)),
                Some(8.0.into()),
                Some(4.0.into()),
            ],
            vec![Some(3.0.into()), Some(7.0.into()), Some(5.0.into())],
        ];
        let metrics = coupled_product_metrics_from_outputs(&result, None, &[], &[]).unwrap();
        assert_eq!(metrics.minimum_impedance_real_ohm, Some(2.0));
    }

    #[test]
    fn dynamic_driver_impedance_artifacts_must_cover_the_full_grid_and_driver_set() {
        let mut result = simulation();
        result.driver_electrical_impedances_ohm.pop();
        assert!(matches!(
            coupled_product_metrics_from_outputs(&result, None, &[], &[]),
            Err(ProductMetricsError::InvalidDynamicImpedanceFrequencyShape {
                expected: 2,
                actual: 1,
            })
        ));

        let mut result = simulation();
        result.driver_electrical_impedances_ohm[1].pop();
        assert!(matches!(
            coupled_product_metrics_from_outputs(&result, None, &[], &[]),
            Err(ProductMetricsError::InvalidDynamicImpedanceDriverCount {
                frequency_index: 1,
                expected: 2,
                actual: 1,
            })
        ));

        let result = simulation();
        assert!(matches!(
            coupled_product_metrics_from_outputs_with_product(
                &result,
                &centred_product_design(),
                None,
                &[],
                &[],
                2,
            ),
            Err(ProductMetricsError::InvalidDynamicImpedanceDriverCount {
                frequency_index: 0,
                expected: 1,
                actual: 2,
            })
        ));
    }

    #[test]
    fn structural_responses_feed_product_metrics() {
        let mut result = simulation();
        result.structural_responses = vec![
            Some(sonium_sim_speaker_modeling::StructuralAcousticResponse {
                frequency_hz: 100.0,
                displacement_rms_m: Complex64::new(1.0e-3, 0.0),
                stress_rms_pa: Complex64::new(2.0e3, 0.0),
                absorbed_power_w: 0.1,
                ..Default::default()
            }),
            Some(sonium_sim_speaker_modeling::StructuralAcousticResponse {
                frequency_hz: 200.0,
                displacement_rms_m: Complex64::new(2.0e-3, 0.0),
                stress_rms_pa: Complex64::new(1.0e3, 0.0),
                absorbed_power_w: 0.2,
                ..Default::default()
            }),
        ];
        let metrics = coupled_product_metrics_from_outputs(&result, None, &[], &[]).unwrap();
        assert_eq!(
            metrics.maximum_panel_displacement_m,
            Some(2.0e-3 * std::f64::consts::SQRT_2)
        );
        assert_eq!(
            metrics.maximum_panel_bending_stress_pa,
            Some(2.0e3 * std::f64::consts::SQRT_2)
        );
    }

    #[test]
    fn uses_loaded_thermal_states_when_no_legacy_compression_curve_is_supplied() {
        let mut result = simulation();
        result.thermal_driver_states = vec![
            [
                Some(sonium_sim_speaker_modeling::ThermalDriverState {
                    coil_temperature_k: 330.0,
                    hot_resistance_ohm: 7.0,
                    coil_loss_w: 10.0,
                    eddy_loss_w: 1.0,
                    input_power_w: 20.0,
                    mechanical_loss_w: 4.0,
                    radiation_power_w: 5.0,
                    power_balance_residual_w: 0.0,
                    compression_db: 2.25,
                    thermal_iteration_count: 3,
                    thermal_residual_ohm: 1.0e-12,
                    nonlinear_temperature_feedback_applied: false,
                    thermal_network_temperatures_k: None,
                    thermal_network_power_w: None,
                    thermal_network_residual_w: None,
                }),
                None,
            ],
            [None, None],
        ];
        let metrics = coupled_product_metrics_from_outputs(&result, None, &[], &[]).unwrap();
        assert_eq!(metrics.thermal_compression_db, Some(2.25));
    }

    // Pins the legacy-vector fallback contract (deprecated fields).
    #[allow(deprecated)]
    #[test]
    fn missing_optional_evidence_is_preserved_for_fail_safe_constraints() {
        let mut result = simulation();
        result.woofer_electrical_impedance_ohm.clear();
        result.tweeter_electrical_impedance_ohm.clear();
        let metrics = coupled_product_metrics_from_outputs(&result, None, &[], &[]).unwrap();
        assert_eq!(metrics, CoupledProductMetrics::default());
    }

    #[test]
    fn maps_valid_reference_sweep_to_the_product_residual_metric() {
        let mut result = simulation();
        result.product_reference_sweep = Some(reference_sweep());
        let metrics = coupled_product_metrics_from_outputs(&result, None, &[], &[]).unwrap();
        assert_eq!(metrics.maximum_reference_relative_residual, Some(2.0e-8));
    }

    #[test]
    fn rejects_reference_evidence_on_a_different_frequency_grid() {
        let mut result = simulation();
        let mut reference = reference_sweep();
        reference.points[1].frequency_hz = 210.0;
        result.product_reference_sweep = Some(reference);
        assert!(matches!(
            coupled_product_metrics_from_outputs(&result, None, &[], &[]),
            Err(ProductMetricsError::ReferenceFrequencyMismatch { index: 1, .. })
        ));
    }

    #[test]
    fn product_metric_adapter_adds_reference_panel_response() {
        let mut result = simulation();
        result.driver_electrical_impedances_ohm = vec![vec![None]; 2];
        result.driver_velocities_bem = vec![vec![Complex64::new(0.01, 0.0)]; 2];
        result.product_reference_sweep = Some(reference_sweep());
        let metrics = coupled_product_metrics_from_outputs_with_product(
            &result,
            &centred_product_design(),
            None,
            &[],
            &[],
            2,
        )
        .unwrap();
        assert!(metrics.maximum_panel_displacement_m.is_some());
        assert!(metrics.maximum_panel_bending_stress_pa.is_some());
        assert_eq!(metrics.maximum_reference_relative_residual, Some(2.0e-8));
    }

    #[test]
    fn product_metric_admission_rejects_missing_excursion_evidence() {
        let mut result = simulation();
        result.driver_electrical_impedances_ohm = vec![vec![None]; 2];
        let error = coupled_product_metrics_from_outputs_with_product(
            &result,
            &centred_product_design(),
            None,
            &[],
            &[],
            2,
        )
        .expect_err("product metrics must not admit missing excursion evidence");
        assert_eq!(error, ProductMetricsError::MissingExcursionEvidence);
    }

    #[cfg(feature = "cli")]
    #[test]
    fn spinorama_adapter_copies_cea_preference_score() {
        let spinorama = crate::cea2034_output::SpinoramaMetrics {
            on_axis: vec![],
            listening_window: vec![],
            early_reflections: vec![],
            sound_power: vec![],
            pir: vec![],
            preference_score: 4.25,
        };
        let metrics =
            coupled_product_metrics_from_spinorama(&simulation(), &spinorama, &[], &[]).unwrap();
        assert_eq!(metrics.cea_preference_score, Some(4.25));
    }

    // Pins the legacy-vector shape contract (deprecated fields).
    #[allow(deprecated)]
    #[test]
    fn rejects_bad_shapes_and_positive_compression() {
        let mut result = simulation();
        result.woofer_electrical_impedance_ohm.pop();
        assert!(matches!(
            coupled_product_metrics_from_outputs(&result, None, &[], &[]),
            Err(ProductMetricsError::InvalidImpedanceShape { .. })
        ));

        let error = coupled_product_metrics_from_outputs(
            &simulation(),
            None,
            &[],
            &[CompressionPoint {
                nominal_power: 1.0,
                actual_power: 1.0,
                coil_temperature: 300.0,
                resistance_hot: 6.0,
                compression_db: 0.1,
            }],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ProductMetricsError::PositiveCompression { .. }
        ));
    }

    #[test]
    fn rejects_missing_or_unclosed_product_power_ledgers() {
        let mut result = simulation();
        result.power_ledgers = vec![None];
        assert!(matches!(
            coupled_product_metrics_from_outputs(&result, None, &[], &[]),
            Err(ProductMetricsError::InvalidPowerLedgerShape { .. })
        ));

        let mut result = simulation();
        result.power_ledgers[0] = Some(sonium_sim_speaker_modeling::PowerLedger {
            supplied_w: 10.0,
            copper_loss_w: 1.0,
            magnetic_loss_w: 1.0,
            mechanical_loss_w: 1.0,
            structural_loss_w: 1.0,
            internal_acoustic_loss_w: 1.0,
            exterior_acoustic_power_w: 1.0,
            thermal_storage_w: 1.0,
            residual_w: 3.0,
        });
        let error = coupled_product_metrics_from_outputs(&result, None, &[], &[]).unwrap_err();
        assert!(matches!(
            error,
            ProductMetricsError::InvalidPowerLedger { index: 0, .. }
        ));
    }

    #[test]
    fn derives_peak_excursion_from_bem_velocity_evidence() {
        let mut result = simulation();
        result.driver_velocities_bem = vec![
            vec![Complex64::new(1.0, 0.0)],
            vec![Complex64::new(2.0, 0.0)],
        ];
        let metrics = coupled_product_metrics_from_outputs(&result, None, &[], &[]).unwrap();
        let expected = 2.0 / (std::f64::consts::TAU * 200.0);
        assert_eq!(metrics.peak_excursion_m, Some(expected));
    }
}
