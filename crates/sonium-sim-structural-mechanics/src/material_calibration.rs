//! Bounded stress-strain material calibration reference.
//!
//! The fitter accepts caller-supplied monotone uniaxial data, estimates an
//! elastic modulus from the initial samples, locates a conventional offset
//! yield point, and fits isotropic linear hardening. It is deliberately
//! explicit about being a calibration contract; it does not manufacture or
//! authenticate measured data.

use crate::{StructuralExecutionProvenance, StructuralMechanicsError};
use serde::{Deserialize, Serialize};

/// Schema version for the material calibration result.
pub const MATERIAL_CALIBRATION_SCHEMA_VERSION: u32 = 1;

/// One caller-supplied monotone uniaxial stress-strain observation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StressStrainObservation {
    pub strain: f64,
    pub stress_pa: f64,
}

/// Calibration controls and source metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialCalibrationStudy {
    pub observations: Vec<StressStrainObservation>,
    pub elastic_sample_count: usize,
    pub offset_strain: f64,
    pub temperature_c: f64,
    pub source_id: String,
    pub max_fit_rmse_pa: f64,
    pub max_fit_relative_error: f64,
}

impl MaterialCalibrationStudy {
    /// Small deterministic bilinear fixture useful for focused tests.
    pub fn introductory() -> Self {
        let youngs_modulus_pa = 200.0e9;
        let yield_strain = 0.00125;
        let yield_stress_pa = youngs_modulus_pa * yield_strain;
        let hardening_pa = 4.0e9;
        let strains = [0.0, 0.00025, 0.00075, 0.0010, 0.0015, 0.0025, 0.0040];
        let observations = strains
            .into_iter()
            .map(|strain| StressStrainObservation {
                strain,
                stress_pa: if strain <= yield_strain {
                    youngs_modulus_pa * strain
                } else {
                    yield_stress_pa + hardening_pa * (strain - yield_strain)
                },
            })
            .collect();
        Self {
            observations,
            elastic_sample_count: 4,
            offset_strain: 0.002,
            temperature_c: 20.0,
            source_id: "fixture:bilinear-isotropic-hardening".to_string(),
            max_fit_rmse_pa: 20.0e6,
            max_fit_relative_error: 0.10,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        if self.observations.len() < 4 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "material_calibration.observations",
                value: self.observations.len().to_string(),
            });
        }
        if self.elastic_sample_count < 2 || self.elastic_sample_count >= self.observations.len() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "material_calibration.elastic_sample_count",
                value: self.elastic_sample_count.to_string(),
            });
        }
        for (field, value) in [
            ("material_calibration.offset_strain", self.offset_strain),
            ("material_calibration.temperature_c", self.temperature_c),
            ("material_calibration.max_fit_rmse_pa", self.max_fit_rmse_pa),
            (
                "material_calibration.max_fit_relative_error",
                self.max_fit_relative_error,
            ),
        ] {
            if !value.is_finite() || (field != "material_calibration.temperature_c" && value <= 0.0)
            {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if self.source_id.trim().is_empty() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "material_calibration.source_id",
                value: "empty".to_string(),
            });
        }
        let mut previous_strain = -f64::EPSILON;
        let mut previous_stress = -f64::EPSILON;
        for observation in &self.observations {
            if !observation.strain.is_finite()
                || !observation.stress_pa.is_finite()
                || observation.strain < 0.0
                || observation.stress_pa < 0.0
                || observation.strain <= previous_strain
                || observation.stress_pa < previous_stress
            {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "material_calibration.observations",
                    value: "finite, strictly increasing strain and non-decreasing stress required"
                        .to_string(),
                });
            }
            previous_strain = observation.strain;
            previous_stress = observation.stress_pa;
        }
        Ok(())
    }
}

/// Calibrated bilinear isotropic-hardening parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibratedElastoplasticMaterial {
    pub youngs_modulus_pa: f64,
    pub yield_stress_pa: f64,
    pub isotropic_hardening_pa: f64,
    pub temperature_c: f64,
    pub source_id: String,
}

/// Versioned material calibration report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialCalibrationResult {
    pub schema_version: u32,
    pub study: MaterialCalibrationStudy,
    pub material: CalibratedElastoplasticMaterial,
    pub yield_strain_offset: f64,
    pub fit_rmse_pa: f64,
    pub fit_max_abs_error_pa: f64,
    pub fit_max_relative_error: f64,
    pub fit_accepted: bool,
    pub model_assumption: MaterialCalibrationModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Physical idealisation used by the fitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialCalibrationModelAssumption {
    MonotoneBilinearIsotropicHardening,
}

fn interpolate_crossing(
    previous: StressStrainObservation,
    current: StressStrainObservation,
    youngs_modulus_pa: f64,
    offset_strain: f64,
) -> Option<(f64, f64)> {
    let previous_gap = previous.stress_pa - youngs_modulus_pa * (previous.strain - offset_strain);
    let current_gap = current.stress_pa - youngs_modulus_pa * (current.strain - offset_strain);
    if previous_gap > 0.0 && current_gap <= 0.0 {
        let fraction = previous_gap / (previous_gap - current_gap);
        let strain = previous.strain + fraction * (current.strain - previous.strain);
        let stress = previous.stress_pa + fraction * (current.stress_pa - previous.stress_pa);
        Some((strain, stress))
    } else {
        None
    }
}

/// Fit a bilinear isotropic-hardening material from supplied stress-strain data.
pub fn calibrate_material(
    study: &MaterialCalibrationStudy,
) -> Result<MaterialCalibrationResult, StructuralMechanicsError> {
    study.validate()?;
    let elastic = &study.observations[..study.elastic_sample_count];
    let denominator: f64 = elastic.iter().map(|point| point.strain.powi(2)).sum();
    if denominator <= f64::EPSILON {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "material_calibration.elastic_modulus",
            value: denominator,
        });
    }
    let youngs_modulus_pa = elastic
        .iter()
        .map(|point| point.strain * point.stress_pa)
        .sum::<f64>()
        / denominator;
    let mut yield_strain_offset = None;
    let mut yield_stress_pa = None;
    for pair in study.observations[study.elastic_sample_count - 1..].windows(2) {
        if let Some((strain, stress)) =
            interpolate_crossing(pair[0], pair[1], youngs_modulus_pa, study.offset_strain)
        {
            yield_strain_offset = Some(strain);
            yield_stress_pa = Some(stress);
            break;
        }
    }
    let yield_strain_offset =
        yield_strain_offset.ok_or(StructuralMechanicsError::ValidationFailure {
            check: "material_calibration.offset_yield",
            value: 0.0,
        })?;
    let offset_yield_stress_pa =
        yield_stress_pa.ok_or(StructuralMechanicsError::ValidationFailure {
            check: "material_calibration.offset_yield_stress",
            value: 0.0,
        })?;
    let hardening_samples: Vec<(f64, f64)> = study
        .observations
        .iter()
        .filter_map(|point| {
            let plastic_strain = point.strain - point.stress_pa / youngs_modulus_pa;
            (plastic_strain > 0.0).then_some((plastic_strain, point.stress_pa))
        })
        .collect();
    if hardening_samples.len() < 2 {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "material_calibration.isotropic_hardening_samples",
            value: hardening_samples.len() as f64,
        });
    }
    let sample_count = hardening_samples.len() as f64;
    let mean_plastic_strain = hardening_samples
        .iter()
        .map(|(plastic, _)| plastic)
        .sum::<f64>()
        / sample_count;
    let mean_stress_pa = hardening_samples
        .iter()
        .map(|(_, stress)| stress)
        .sum::<f64>()
        / sample_count;
    let hardening_denominator: f64 = hardening_samples
        .iter()
        .map(|(plastic, _)| (plastic - mean_plastic_strain).powi(2))
        .sum();
    if hardening_denominator <= f64::EPSILON {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "material_calibration.isotropic_hardening",
            value: hardening_denominator,
        });
    }
    let apparent_hardening_pa = hardening_samples
        .iter()
        .map(|(plastic, stress)| (plastic - mean_plastic_strain) * (stress - mean_stress_pa))
        .sum::<f64>()
        / hardening_denominator;
    let isotropic_hardening_pa =
        youngs_modulus_pa * apparent_hardening_pa / (youngs_modulus_pa + apparent_hardening_pa);
    let yield_stress_pa = mean_stress_pa - isotropic_hardening_pa * mean_plastic_strain;
    let elastic_yield_strain = yield_stress_pa / youngs_modulus_pa;
    if !youngs_modulus_pa.is_finite()
        || !yield_stress_pa.is_finite()
        || !isotropic_hardening_pa.is_finite()
        || youngs_modulus_pa <= 0.0
        || offset_yield_stress_pa <= 0.0
        || yield_stress_pa <= 0.0
        || isotropic_hardening_pa < 0.0
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "material_calibration.fitted_parameters",
            value: youngs_modulus_pa
                .max(yield_stress_pa)
                .max(isotropic_hardening_pa),
        });
    }

    let mut squared_error = 0.0;
    let mut max_abs_error_pa: f64 = 0.0;
    let mut max_relative_error: f64 = 0.0;
    for point in &study.observations {
        let plastic_strain = (point.strain - elastic_yield_strain).max(0.0);
        let predicted_stress = if point.strain <= elastic_yield_strain {
            youngs_modulus_pa * point.strain
        } else {
            yield_stress_pa + isotropic_hardening_pa * plastic_strain
        };
        let error = predicted_stress - point.stress_pa;
        squared_error += error * error;
        max_abs_error_pa = max_abs_error_pa.max(error.abs());
        max_relative_error = max_relative_error.max(error.abs() / point.stress_pa.max(1.0));
    }
    let fit_rmse_pa = (squared_error / study.observations.len() as f64).sqrt();
    let fit_accepted =
        fit_rmse_pa <= study.max_fit_rmse_pa && max_relative_error <= study.max_fit_relative_error;

    let mut execution = StructuralExecutionProvenance::cpu_reference();
    execution.capability_id =
        "sonium-structural.cpu-reference.f64.material-calibration.v1".to_string();
    execution.limitations.extend([
        "Calibration consumes caller-supplied monotone uniaxial observations and does not authenticate their measurement origin or certificate.".to_string(),
        "The bilinear isotropic-hardening fit is not an anisotropic, rate-dependent, cyclic, temperature-dependent, fracture, or manufacturing-process material law.".to_string(),
    ]);

    Ok(MaterialCalibrationResult {
        schema_version: MATERIAL_CALIBRATION_SCHEMA_VERSION,
        study: study.clone(),
        material: CalibratedElastoplasticMaterial {
            youngs_modulus_pa,
            yield_stress_pa,
            isotropic_hardening_pa,
            temperature_c: study.temperature_c,
            source_id: study.source_id.clone(),
        },
        yield_strain_offset,
        fit_rmse_pa,
        fit_max_abs_error_pa: max_abs_error_pa,
        fit_max_relative_error: max_relative_error,
        fit_accepted,
        model_assumption: MaterialCalibrationModelAssumption::MonotoneBilinearIsotropicHardening,
        execution,
    })
}
