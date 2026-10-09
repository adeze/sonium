//! Deterministic tolerance and calibration scenarios for robust speaker runs.
//!
//! The scenario generator is intentionally solver-independent.  Product and
//! golden-model optimizers can use the same ordered cases for driver,
//! material, temperature, assembly, and measurement uncertainty without
//! making parallel scheduling change the objective.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::driver_model::{DriverModel, ImpedancePoint};

/// Fractional tolerance inputs around a nominal driver/model state.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpeakerToleranceSpec {
    pub force_factor_fraction: f64,
    pub inductance_fraction: f64,
    pub moving_mass_fraction: f64,
    pub compliance_fraction: f64,
    pub resistance_fraction: f64,
    pub material_fraction: f64,
    pub temperature_delta_c: f64,
    pub assembly_offset_m: f64,
    pub measurement_noise_db: f64,
}

impl Default for SpeakerToleranceSpec {
    fn default() -> Self {
        Self {
            force_factor_fraction: 0.05,
            inductance_fraction: 0.05,
            moving_mass_fraction: 0.03,
            compliance_fraction: 0.05,
            resistance_fraction: 0.05,
            material_fraction: 0.05,
            temperature_delta_c: 10.0,
            assembly_offset_m: 0.001,
            measurement_noise_db: 0.25,
        }
    }
}

impl SpeakerToleranceSpec {
    pub fn validate(&self) -> Result<(), UncertaintyError> {
        for (name, value) in [
            ("force_factor_fraction", self.force_factor_fraction),
            ("inductance_fraction", self.inductance_fraction),
            ("moving_mass_fraction", self.moving_mass_fraction),
            ("compliance_fraction", self.compliance_fraction),
            ("resistance_fraction", self.resistance_fraction),
            ("material_fraction", self.material_fraction),
        ] {
            if !value.is_finite() || !(0.0..1.0).contains(&value) {
                return Err(UncertaintyError::InvalidSpec(name));
            }
        }
        for (name, value) in [
            ("temperature_delta_c", self.temperature_delta_c),
            ("assembly_offset_m", self.assembly_offset_m),
            ("measurement_noise_db", self.measurement_noise_db),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(UncertaintyError::InvalidSpec(name));
            }
        }
        Ok(())
    }

    fn dimensions(&self) -> [f64; 9] {
        [
            self.force_factor_fraction,
            self.inductance_fraction,
            self.moving_mass_fraction,
            self.compliance_fraction,
            self.resistance_fraction,
            self.material_fraction,
            self.temperature_delta_c,
            self.assembly_offset_m,
            self.measurement_noise_db,
        ]
    }
}

/// Deterministic robust-sampling policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UncertaintySampling {
    /// One nominal case.
    Nominal,
    /// All 2ⁿ combinations of the configured tolerance bounds.
    Corners,
    /// A deterministic Latin-hypercube set with the supplied seed.
    LatinHypercube { samples: usize, seed: u64 },
}

/// One normalized uncertainty scenario consumed by a solver/evaluator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerUncertaintyCase {
    pub index: usize,
    pub label: String,
    pub force_factor_scale: f64,
    pub inductance_scale: f64,
    pub moving_mass_scale: f64,
    pub compliance_scale: f64,
    pub resistance_scale: f64,
    pub material_scale: f64,
    pub temperature_offset_c: f64,
    pub assembly_offset_m: f64,
    pub measurement_offset_db: f64,
}

impl SpeakerUncertaintyCase {
    /// Validate a case before applying it to a typed driver handoff.
    pub fn validate(&self) -> Result<(), UncertaintyError> {
        for (name, value) in [
            ("force_factor_scale", self.force_factor_scale),
            ("inductance_scale", self.inductance_scale),
            ("moving_mass_scale", self.moving_mass_scale),
            ("compliance_scale", self.compliance_scale),
            ("resistance_scale", self.resistance_scale),
            ("material_scale", self.material_scale),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(UncertaintyError::InvalidCase(name));
            }
        }
        for (name, value) in [
            ("temperature_offset_c", self.temperature_offset_c),
            ("assembly_offset_m", self.assembly_offset_m),
            ("measurement_offset_db", self.measurement_offset_db),
        ] {
            if !value.is_finite() {
                return Err(UncertaintyError::InvalidCase(name));
            }
        }
        Ok(())
    }

    /// Apply the driver-domain part of this case to a typed model.
    ///
    /// The stored velocity and input-impedance tables are recomputed against
    /// the nominal exterior load inferred from the original handoff. This
    /// keeps the product-side radiation boundary unchanged while making the
    /// electrical and mechanical network coefficients physically consistent.
    /// Assembly and measurement offsets intentionally remain product-stage
    /// concerns and are not silently folded into a driver-unit model.
    pub fn apply_to_driver_model(
        &self,
        nominal: &DriverModel,
    ) -> Result<DriverModel, UncertaintyError> {
        self.validate()?;
        nominal
            .validate()
            .map_err(|error| UncertaintyError::DriverModelApplication(error.to_string()))?;

        let copper_temperature_scale = 1.0 + 0.00393 * self.temperature_offset_c;
        if !copper_temperature_scale.is_finite() || copper_temperature_scale <= 0.0 {
            return Err(UncertaintyError::InvalidCase("temperature_offset_c"));
        }
        let electrical_resistance_scale = self.resistance_scale * copper_temperature_scale;
        let mechanical_loss_scale = self.material_scale;
        let mass = nominal.thiele_small.moving_mass_kg * self.moving_mass_scale;
        let compliance = nominal.thiele_small.compliance_m_per_n * self.compliance_scale;
        if nominal.thiele_small.force_factor_tm.abs() <= f64::MIN_POSITIVE {
            return Err(UncertaintyError::DriverModelApplication(
                "nominal force factor cannot recover exterior load".to_owned(),
            ));
        }
        let resistance =
            nominal.thiele_small.mechanical_resistance_kg_per_s * mechanical_loss_scale;
        let resonance_hz =
            (1.0 / (std::f64::consts::TAU * std::f64::consts::TAU * mass * compliance)).sqrt();
        let mechanical_q =
            (resistance > 0.0).then(|| std::f64::consts::TAU * resonance_hz * mass / resistance);
        let electrical_q = (self.force_factor_scale > 0.0).then(|| {
            std::f64::consts::TAU
                * resonance_hz
                * mass
                * (nominal.thiele_small.voice_coil_resistance_ohm * electrical_resistance_scale)
                / (nominal.thiele_small.force_factor_tm * self.force_factor_scale).powi(2)
        });
        let total_q = mechanical_q
            .zip(electrical_q)
            .map(|(mechanical, electrical)| (mechanical.recip() + electrical.recip()).recip());

        let mut model = nominal.clone();
        model.thiele_small.voice_coil_resistance_ohm *= electrical_resistance_scale;
        model.thiele_small.nominal_inductance_h *= self.inductance_scale;
        model.thiele_small.force_factor_tm *= self.force_factor_scale;
        model.thiele_small.moving_mass_kg = mass;
        model.thiele_small.compliance_m_per_n = compliance;
        model.thiele_small.mechanical_resistance_kg_per_s = resistance;
        model.thiele_small.resonance_hz = resonance_hz;
        model.thiele_small.mechanical_q = mechanical_q;
        model.thiele_small.electrical_q = electrical_q;
        model.thiele_small.total_q = total_q;
        model.thiele_small.equivalent_compliance_volume_m3 *= self.compliance_scale;
        for point in &mut model.bl_curve {
            point.force_factor_tm *= self.force_factor_scale;
        }
        for point in &mut model.inductance_curve {
            point.inductance_h *= self.inductance_scale;
            point.eddy_resistance_ohm *= electrical_resistance_scale;
        }
        for point in &mut model.intrinsic_mechanical_impedance {
            let omega = std::f64::consts::TAU * point.frequency_hz;
            let nominal_reactive = omega * nominal.thiele_small.moving_mass_kg
                - 1.0 / (omega * nominal.thiele_small.compliance_m_per_n);
            let non_lumped_reactive = point.imaginary - nominal_reactive;
            point.real *= mechanical_loss_scale;
            point.imaginary = non_lumped_reactive + omega * mass - 1.0 / (omega * compliance);
        }

        model.velocity_transfer = nominal
            .velocity_transfer
            .iter()
            .map(|point| {
                let radiation = infer_radiation_impedance(nominal, point.frequency_hz)?;
                let (velocity, _) = model
                    .response_with_radiation_impedance(point.frequency_hz, radiation)
                    .map_err(|error| UncertaintyError::DriverModelApplication(error.to_string()))?;
                Ok(crate::driver_model::VelocityTransferPoint {
                    frequency_hz: point.frequency_hz,
                    velocity_real_m_per_s: velocity.re,
                    velocity_imag_m_per_s: velocity.im,
                })
            })
            .collect::<Result<Vec<_>, UncertaintyError>>()?;
        model.electrical_impedance = nominal
            .electrical_impedance
            .iter()
            .map(|point| {
                let radiation = infer_radiation_impedance(nominal, point.frequency_hz)?;
                let (_, electrical) = model
                    .response_with_radiation_impedance(point.frequency_hz, radiation)
                    .map_err(|error| UncertaintyError::DriverModelApplication(error.to_string()))?;
                Ok(ImpedancePoint {
                    frequency_hz: point.frequency_hz,
                    real: electrical.re,
                    imaginary: electrical.im,
                })
            })
            .collect::<Result<Vec<_>, UncertaintyError>>()?;
        model
            .validate()
            .map_err(|error| UncertaintyError::DriverModelApplication(error.to_string()))?;
        Ok(model)
    }
}

fn infer_radiation_impedance(
    nominal: &DriverModel,
    frequency_hz: f64,
) -> Result<Complex64, UncertaintyError> {
    let network = nominal
        .network_parameters_at(frequency_hz)
        .map_err(|error| UncertaintyError::DriverModelApplication(error.to_string()))?;
    let electrical = nominal.electrical_impedance_at(frequency_hz);
    let denominator = electrical - network.coil_impedance_ohm;
    if denominator.norm() <= 1.0e-14 {
        return Err(UncertaintyError::DriverModelApplication(
            "nominal electrical impedance cannot recover exterior load".to_owned(),
        ));
    }
    let total_mechanical = Complex64::new(network.force_factor_tm.powi(2), 0.0) / denominator;
    let radiation = total_mechanical - network.intrinsic_mechanical_impedance;
    if !radiation.re.is_finite() || !radiation.im.is_finite() {
        return Err(UncertaintyError::DriverModelApplication(
            "inferred exterior load is non-finite".to_owned(),
        ));
    }
    Ok(radiation)
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UncertaintyError {
    #[error("uncertainty specification is invalid: {0}")]
    InvalidSpec(&'static str),
    #[error("uncertainty case is invalid: {0}")]
    InvalidCase(&'static str),
    #[error("cannot apply uncertainty case to driver model: {0}")]
    DriverModelApplication(String),
    #[error("Latin-hypercube sample count must be positive")]
    EmptySampleSet,
}

/// Generate a stable scenario order.  The ordering is part of the optimizer
/// checkpoint contract: changing thread count must not change the sampled
/// robust objective.
pub fn generate_uncertainty_cases(
    spec: SpeakerToleranceSpec,
    sampling: UncertaintySampling,
) -> Result<Vec<SpeakerUncertaintyCase>, UncertaintyError> {
    spec.validate()?;
    let dimensions = spec.dimensions();
    let points: Vec<[f64; 9]> = match sampling {
        UncertaintySampling::Nominal => vec![[0.0; 9]],
        UncertaintySampling::Corners => {
            let count = 1usize << dimensions.len();
            (0..count)
                .map(|corner| {
                    std::array::from_fn(|dimension| {
                        if corner & (1 << dimension) == 0 {
                            -1.0
                        } else {
                            1.0
                        }
                    })
                })
                .collect()
        }
        UncertaintySampling::LatinHypercube { samples, seed } => {
            if samples == 0 {
                return Err(UncertaintyError::EmptySampleSet);
            }
            let mut permutations = Vec::with_capacity(dimensions.len());
            for dimension in 0..dimensions.len() {
                let mut permutation: Vec<usize> = (0..samples).collect();
                let mut state = seed
                    .wrapping_add(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add((dimension as u64).wrapping_mul(0x517c_c1b7_2722_0a95));
                for index in (1..samples).rev() {
                    state = xorshift64(state);
                    let swap = (state as usize) % (index + 1);
                    permutation.swap(index, swap);
                }
                permutations.push(permutation);
            }
            (0..samples)
                .map(|sample| {
                    std::array::from_fn(|dimension| {
                        let bin = permutations[dimension][sample];
                        let center = (bin as f64 + 0.5) / samples as f64;
                        2.0 * center - 1.0
                    })
                })
                .collect()
        }
    };

    Ok(points
        .into_iter()
        .enumerate()
        .map(|(index, point)| SpeakerUncertaintyCase {
            index,
            label: if point.iter().all(|value| *value == 0.0) {
                "nominal".to_owned()
            } else {
                format!("case-{index:04}")
            },
            force_factor_scale: 1.0 + point[0] * dimensions[0],
            inductance_scale: 1.0 + point[1] * dimensions[1],
            moving_mass_scale: 1.0 + point[2] * dimensions[2],
            compliance_scale: 1.0 + point[3] * dimensions[3],
            resistance_scale: 1.0 + point[4] * dimensions[4],
            material_scale: 1.0 + point[5] * dimensions[5],
            temperature_offset_c: point[6] * dimensions[6],
            assembly_offset_m: point[7] * dimensions[7],
            measurement_offset_db: point[8] * dimensions[8],
        })
        .collect())
}

fn xorshift64(mut state: u64) -> u64 {
    if state == 0 {
        state = 0x9e37_79b9_7f4a_7c15;
    }
    state ^= state << 13;
    state ^= state >> 7;
    state ^ (state << 17)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corner_and_latin_hypercube_cases_are_bounded_and_repeatable() {
        let spec = SpeakerToleranceSpec::default();
        let corners = generate_uncertainty_cases(spec, UncertaintySampling::Corners).unwrap();
        assert_eq!(corners.len(), 512);
        assert!(corners.iter().all(|case| case.force_factor_scale > 0.0));

        let first = generate_uncertainty_cases(
            spec,
            UncertaintySampling::LatinHypercube {
                samples: 17,
                seed: 42,
            },
        )
        .unwrap();
        let second = generate_uncertainty_cases(
            spec,
            UncertaintySampling::LatinHypercube {
                samples: 17,
                seed: 42,
            },
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 17);
        assert!(
            first
                .iter()
                .all(|case| case.measurement_offset_db.abs() <= 0.25)
        );
    }

    #[test]
    fn nominal_case_rebuilds_a_consistent_typed_driver_handoff() {
        let study = crate::SpeakerDriverStudy::detailed_frequency_domain_driver();
        let nominal = DriverModel::from_study(&study).unwrap();
        let case = SpeakerUncertaintyCase {
            index: 0,
            label: "nominal".to_owned(),
            force_factor_scale: 1.0,
            inductance_scale: 1.0,
            moving_mass_scale: 1.0,
            compliance_scale: 1.0,
            resistance_scale: 1.0,
            material_scale: 1.0,
            temperature_offset_c: 0.0,
            assembly_offset_m: 0.0,
            measurement_offset_db: 0.0,
        };
        let rebuilt = case.apply_to_driver_model(&nominal).unwrap();
        rebuilt.validate().unwrap();
        assert_eq!(
            rebuilt.velocity_transfer.len(),
            nominal.velocity_transfer.len()
        );
        assert_eq!(
            rebuilt.electrical_impedance.len(),
            nominal.electrical_impedance.len()
        );
        for (expected, actual) in nominal
            .velocity_transfer
            .iter()
            .zip(&rebuilt.velocity_transfer)
        {
            let expected = expected.velocity();
            let actual = actual.velocity();
            assert!((expected - actual).norm() < 1.0e-9);
        }
    }

    #[test]
    fn physical_case_changes_network_coefficients_and_keeps_passivity() {
        let study = crate::SpeakerDriverStudy::detailed_frequency_domain_driver();
        let nominal = DriverModel::from_study(&study).unwrap();
        let mut cases = generate_uncertainty_cases(
            SpeakerToleranceSpec::default(),
            UncertaintySampling::LatinHypercube {
                samples: 2,
                seed: 7,
            },
        )
        .unwrap();
        let case = cases.pop().unwrap();
        let perturbed = case.apply_to_driver_model(&nominal).unwrap();
        perturbed.validate().unwrap();
        assert_ne!(
            perturbed.network_parameters_at(1_000.0).unwrap(),
            nominal.network_parameters_at(1_000.0).unwrap()
        );
        assert!(
            perturbed
                .electrical_impedance
                .iter()
                .all(|point| point.real >= 0.0)
        );
        assert!(
            perturbed
                .intrinsic_mechanical_impedance
                .iter()
                .all(|point| point.real >= 0.0)
        );
    }
}
