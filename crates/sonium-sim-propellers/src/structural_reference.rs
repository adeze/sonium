//! Checked static centrifugal and aerodynamic bending stress reference.
//!
//! This module provides a transparent structural screening layer over a
//! converged blade-element result. It treats each supplied radial station as a
//! control-volume section with rectangular chord/thickness geometry, integrates
//! outboard centrifugal force, and estimates flapwise bending stress from the
//! station thrust distribution. The result is a static margin contract, not an
//! aeroelastic deformation, modal, fatigue, or nonlinear structural solve.

use std::f64::consts::TAU;
use std::fmt;

use crate::{BladeElementBemResult, BladeElementModel, BladeElementOperatingPoint};

/// Material and section assumptions for the static structural screen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BladeStructuralOptions {
    /// Material density [kg/m³].
    pub material_density_kg_m3: f64,
    /// Allowable material stress before safety factor [Pa].
    pub allowable_stress_pa: f64,
    /// Rectangular section thickness as a fraction of local chord [-].
    pub thickness_fraction: f64,
    /// Required safety factor on the allowable stress [-].
    pub safety_factor: f64,
}

impl Default for BladeStructuralOptions {
    fn default() -> Self {
        Self {
            material_density_kg_m3: 1_600.0,
            allowable_stress_pa: 250.0e6,
            thickness_fraction: 0.12,
            safety_factor: 1.5,
        }
    }
}

/// Static stress and mass values at one radial station.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BladeStructuralStation {
    /// Station radius [m].
    pub radius_m: f64,
    /// Radial control-volume width [m].
    pub width_m: f64,
    /// Local chord [m].
    pub chord_m: f64,
    /// Rectangular section area [m²].
    pub section_area_m2: f64,
    /// Material mass per radial length [kg/m].
    pub mass_per_m_kg_m: f64,
    /// Mass in this control volume [kg].
    pub control_volume_mass_kg: f64,
    /// Centrifugal force carried by this section and all outboard sections [N].
    pub outboard_centrifugal_force_n: f64,
    /// Centrifugal normal stress [Pa].
    pub centrifugal_stress_pa: f64,
    /// Per-blade axial aerodynamic force in this control volume [N].
    pub aerodynamic_force_n: f64,
    /// Flapwise bending moment from this section and all outboard sections [N m].
    pub outboard_bending_moment_nm: f64,
    /// Rectangular-section bending stress [Pa].
    pub bending_stress_pa: f64,
    /// Conservative sum of centrifugal and absolute bending stress [Pa].
    pub combined_stress_pa: f64,
    /// Allowable stress after the requested safety factor [Pa].
    pub design_allowable_stress_pa: f64,
    /// Remaining stress margin, positive when admissible [Pa].
    pub margin_pa: f64,
}

/// Static structural screening result for one converged BEM operating point.
#[derive(Debug, Clone, PartialEq)]
pub struct BladeStructuralResult {
    /// Operating point inherited from BEM.
    pub operating_point: BladeElementOperatingPoint,
    /// Material mass represented by one blade [kg].
    pub blade_mass_kg: f64,
    /// Root centrifugal force carried by one blade [N].
    pub root_centrifugal_force_n: f64,
    /// Root flapwise bending moment carried by one blade [N m].
    pub root_bending_moment_nm: f64,
    /// Maximum combined stress across checked sections [Pa].
    pub max_combined_stress_pa: f64,
    /// Minimum stress margin across checked sections [Pa].
    pub min_margin_pa: f64,
    /// Station results in model order.
    pub stations: Vec<BladeStructuralStation>,
}

impl BladeStructuralResult {
    /// Whether all values are finite and the checked state is structurally admissible.
    pub fn is_finite(&self) -> bool {
        self.operating_point.rpm.is_finite()
            && self.operating_point.rpm > 0.0
            && self.operating_point.freestream_m_s.is_finite()
            && self.operating_point.freestream_m_s > 0.0
            && self.operating_point.density_kg_m3.is_finite()
            && self.operating_point.density_kg_m3 > 0.0
            && self.blade_mass_kg.is_finite()
            && self.blade_mass_kg > 0.0
            && self.root_centrifugal_force_n.is_finite()
            && self.root_centrifugal_force_n >= 0.0
            && self.root_bending_moment_nm.is_finite()
            && self.root_bending_moment_nm >= 0.0
            && self.max_combined_stress_pa.is_finite()
            && self.max_combined_stress_pa >= 0.0
            && self.min_margin_pa.is_finite()
            && self.stations.iter().all(|station| {
                station.radius_m.is_finite()
                    && station.radius_m > 0.0
                    && station.width_m.is_finite()
                    && station.width_m > 0.0
                    && station.chord_m.is_finite()
                    && station.chord_m > 0.0
                    && station.section_area_m2.is_finite()
                    && station.section_area_m2 > 0.0
                    && station.mass_per_m_kg_m.is_finite()
                    && station.mass_per_m_kg_m > 0.0
                    && station.control_volume_mass_kg.is_finite()
                    && station.control_volume_mass_kg > 0.0
                    && station.outboard_centrifugal_force_n.is_finite()
                    && station.outboard_centrifugal_force_n >= 0.0
                    && station.centrifugal_stress_pa.is_finite()
                    && station.centrifugal_stress_pa >= 0.0
                    && station.aerodynamic_force_n.is_finite()
                    && station.aerodynamic_force_n >= 0.0
                    && station.outboard_bending_moment_nm.is_finite()
                    && station.outboard_bending_moment_nm >= 0.0
                    && station.bending_stress_pa.is_finite()
                    && station.bending_stress_pa >= 0.0
                    && station.combined_stress_pa.is_finite()
                    && station.combined_stress_pa >= 0.0
                    && station.design_allowable_stress_pa.is_finite()
                    && station.design_allowable_stress_pa > 0.0
                    && station.margin_pa.is_finite()
            })
    }

    /// Whether the minimum checked stress margin is non-negative.
    pub fn is_feasible(&self) -> bool {
        self.is_finite() && self.min_margin_pa >= 0.0
    }
}

/// Failure-closed structural-reference error.
#[derive(Debug, Clone, PartialEq)]
pub enum BladeStructuralError {
    /// An option is outside its physical domain.
    InvalidOptions {
        /// Option field name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The BEM source does not satisfy the checked result contract.
    InvalidBemResult {
        /// Invalid BEM field.
        field: &'static str,
    },
    /// Model and BEM station counts differ.
    StationCountMismatch {
        /// Model station count.
        model_count: usize,
        /// BEM station count.
        result_count: usize,
    },
    /// A structural calculation became non-finite.
    NonFiniteResult {
        /// Result field name.
        field: &'static str,
    },
}

impl fmt::Display for BladeStructuralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => {
                write!(f, "blade structural option {field} must be {requirement}")
            }
            Self::InvalidBemResult { field } => {
                write!(f, "blade structural BEM result {field} is invalid")
            }
            Self::StationCountMismatch {
                model_count,
                result_count,
            } => write!(
                f,
                "blade structural station count mismatch: model {model_count}, BEM {result_count}"
            ),
            Self::NonFiniteResult { field } => {
                write!(f, "blade structural result {field} is non-finite")
            }
        }
    }
}

impl std::error::Error for BladeStructuralError {}

/// Evaluate static centrifugal and aerodynamic bending stresses from BEM loads.
pub fn evaluate_blade_structural_reference(
    model: &BladeElementModel,
    bem: &BladeElementBemResult,
    options: BladeStructuralOptions,
) -> Result<BladeStructuralResult, BladeStructuralError> {
    validate_options(options)?;
    if !bem.converged || bem.max_iterations_used == 0 {
        return Err(BladeStructuralError::InvalidBemResult { field: "converged" });
    }
    let stations = model.stations();
    if stations.len() != bem.loads.strips.len() || stations.len() != bem.strips.len() {
        return Err(BladeStructuralError::StationCountMismatch {
            model_count: stations.len(),
            result_count: bem.loads.strips.len(),
        });
    }

    let operating_point = bem.loads.operating_point;
    if !operating_point.rpm.is_finite()
        || operating_point.rpm <= 0.0
        || !operating_point.freestream_m_s.is_finite()
        || operating_point.freestream_m_s <= 0.0
        || !operating_point.density_kg_m3.is_finite()
        || operating_point.density_kg_m3 <= 0.0
    {
        return Err(BladeStructuralError::InvalidBemResult {
            field: "operating_point",
        });
    }
    let omega_rad_s = TAU * operating_point.rpm / 60.0;
    if !omega_rad_s.is_finite() || omega_rad_s <= 0.0 {
        return Err(BladeStructuralError::InvalidBemResult {
            field: "omega_rad_s",
        });
    }

    let widths = station_widths(model);
    let blade_count = model.n_blades() as f64;
    let design_allowable_stress_pa = options.allowable_stress_pa / options.safety_factor;
    let mut mass_per_length = Vec::with_capacity(stations.len());
    let mut control_masses = Vec::with_capacity(stations.len());
    let mut aerodynamic_forces = Vec::with_capacity(stations.len());
    let mut blade_mass_kg = 0.0;
    for ((station, width_m), load) in stations
        .iter()
        .zip(widths.iter().copied())
        .zip(&bem.loads.strips)
    {
        if !load.axial_force_per_m_n.is_finite() {
            return Err(BladeStructuralError::InvalidBemResult {
                field: "axial_force_per_m_n",
            });
        }
        let thickness_m = station.chord_m * options.thickness_fraction;
        let section_area_m2 = station.chord_m * thickness_m;
        let mass_per_m_kg_m = options.material_density_kg_m3 * section_area_m2;
        let control_volume_mass_kg = mass_per_m_kg_m * width_m;
        let aerodynamic_force_n = load.axial_force_per_m_n.abs() * width_m / blade_count;
        for (field, value) in [
            ("section_area_m2", section_area_m2),
            ("mass_per_m_kg_m", mass_per_m_kg_m),
            ("control_volume_mass_kg", control_volume_mass_kg),
            ("aerodynamic_force_n", aerodynamic_force_n),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(BladeStructuralError::NonFiniteResult { field });
            }
        }
        blade_mass_kg += control_volume_mass_kg;
        mass_per_length.push(mass_per_m_kg_m);
        control_masses.push(control_volume_mass_kg);
        aerodynamic_forces.push(aerodynamic_force_n);
        if !blade_mass_kg.is_finite() {
            return Err(BladeStructuralError::NonFiniteResult {
                field: "blade_mass_kg",
            });
        }
    }

    let angular_speed_squared = omega_rad_s * omega_rad_s;
    let mut structural_stations = Vec::with_capacity(stations.len());
    let mut outboard_centrifugal_force_n = 0.0;
    let mut max_combined_stress_pa: f64 = 0.0;
    let mut min_margin_pa = f64::INFINITY;
    for index in (0..stations.len()).rev() {
        let station = stations[index];
        outboard_centrifugal_force_n +=
            control_masses[index] * angular_speed_squared * station.radius_m;
        let outboard_bending_moment_nm = aerodynamic_forces
            .iter()
            .enumerate()
            .skip(index)
            .map(|(j, force_n)| force_n * (stations[j].radius_m - station.radius_m).max(0.0))
            .sum::<f64>();
        let thickness_m = station.chord_m * options.thickness_fraction;
        let section_area_m2 = station.chord_m * thickness_m;
        let section_modulus_m3 = station.chord_m * thickness_m * thickness_m / 6.0;
        let centrifugal_stress_pa = outboard_centrifugal_force_n / section_area_m2;
        let bending_stress_pa = outboard_bending_moment_nm / section_modulus_m3;
        let combined_stress_pa = centrifugal_stress_pa + bending_stress_pa.abs();
        let margin_pa = design_allowable_stress_pa - combined_stress_pa;
        for (field, value) in [
            ("outboard_centrifugal_force_n", outboard_centrifugal_force_n),
            ("outboard_bending_moment_nm", outboard_bending_moment_nm),
            ("centrifugal_stress_pa", centrifugal_stress_pa),
            ("bending_stress_pa", bending_stress_pa),
            ("combined_stress_pa", combined_stress_pa),
            ("margin_pa", margin_pa),
        ] {
            if !value.is_finite() {
                return Err(BladeStructuralError::NonFiniteResult { field });
            }
        }
        max_combined_stress_pa = max_combined_stress_pa.max(combined_stress_pa);
        min_margin_pa = min_margin_pa.min(margin_pa);
        structural_stations.push(BladeStructuralStation {
            radius_m: station.radius_m,
            width_m: widths[index],
            chord_m: station.chord_m,
            section_area_m2,
            mass_per_m_kg_m: mass_per_length[index],
            control_volume_mass_kg: control_masses[index],
            outboard_centrifugal_force_n,
            centrifugal_stress_pa,
            aerodynamic_force_n: aerodynamic_forces[index],
            outboard_bending_moment_nm,
            bending_stress_pa,
            combined_stress_pa,
            design_allowable_stress_pa,
            margin_pa,
        });
    }
    structural_stations.reverse();

    let root_centrifugal_force_n = structural_stations
        .first()
        .ok_or(BladeStructuralError::InvalidBemResult { field: "stations" })?
        .outboard_centrifugal_force_n;
    let root_bending_moment_nm = aerodynamic_forces
        .iter()
        .enumerate()
        .map(|(index, force_n)| force_n * (stations[index].radius_m - model.hub_radius_m()))
        .sum::<f64>();
    let result = BladeStructuralResult {
        operating_point,
        blade_mass_kg,
        root_centrifugal_force_n,
        root_bending_moment_nm,
        max_combined_stress_pa,
        min_margin_pa,
        stations: structural_stations,
    };
    if !result.is_finite() {
        return Err(BladeStructuralError::NonFiniteResult {
            field: "structural_result",
        });
    }
    Ok(result)
}

fn station_widths(model: &BladeElementModel) -> Vec<f64> {
    let stations = model.stations();
    stations
        .iter()
        .enumerate()
        .map(|(index, station)| {
            let lower = if index == 0 {
                model.hub_radius_m()
            } else {
                0.5 * (stations[index - 1].radius_m + station.radius_m)
            };
            let upper = if index + 1 == stations.len() {
                model.tip_radius_m()
            } else {
                0.5 * (station.radius_m + stations[index + 1].radius_m)
            };
            upper - lower
        })
        .collect()
}

fn validate_options(options: BladeStructuralOptions) -> Result<(), BladeStructuralError> {
    for (field, value) in [
        ("material_density_kg_m3", options.material_density_kg_m3),
        ("allowable_stress_pa", options.allowable_stress_pa),
        ("thickness_fraction", options.thickness_fraction),
        ("safety_factor", options.safety_factor),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(BladeStructuralError::InvalidOptions {
                field,
                requirement: "finite and positive",
            });
        }
    }
    if options.thickness_fraction > 1.0 {
        return Err(BladeStructuralError::InvalidOptions {
            field: "thickness_fraction",
            requirement: "at most one",
        });
    }
    if options.safety_factor < 1.0 {
        return Err(BladeStructuralError::InvalidOptions {
            field: "safety_factor",
            requirement: "at least one",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AirfoilPolar, AirfoilPolarPoint, BladeElementBemOptions, BladeElementStation};
    use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;

    fn model() -> BladeElementModel {
        let source = PropellerSource::drone_5inch(12_000.0);
        let stations = vec![
            BladeElementStation {
                radius_m: 0.018,
                chord_m: 0.018,
                pitch_deg: 24.0,
            },
            BladeElementStation {
                radius_m: 0.036,
                chord_m: 0.014,
                pitch_deg: 21.0,
            },
            BladeElementStation {
                radius_m: 0.054,
                chord_m: 0.010,
                pitch_deg: 18.0,
            },
        ];
        let polar = AirfoilPolar::try_new(vec![
            AirfoilPolarPoint {
                angle_of_attack_deg: -20.0,
                lift_coefficient: -0.8,
                drag_coefficient: 0.08,
            },
            AirfoilPolarPoint {
                angle_of_attack_deg: 0.0,
                lift_coefficient: 0.0,
                drag_coefficient: 0.02,
            },
            AirfoilPolarPoint {
                angle_of_attack_deg: 20.0,
                lift_coefficient: 0.8,
                drag_coefficient: 0.08,
            },
            AirfoilPolarPoint {
                angle_of_attack_deg: 40.0,
                lift_coefficient: 0.9,
                drag_coefficient: 0.16,
            },
        ])
        .expect("valid structural polar");
        BladeElementModel::try_from_source(&source, stations, polar)
            .expect("valid structural model")
    }

    fn bem(model: &BladeElementModel) -> BladeElementBemResult {
        model
            .evaluate_bem(
                BladeElementOperatingPoint {
                    rpm: 12_000.0,
                    freestream_m_s: 10.0,
                    density_kg_m3: 1.204,
                },
                BladeElementBemOptions::default(),
            )
            .expect("converged structural BEM")
    }

    #[test]
    fn structural_reference_is_finite_and_reports_root_governing_values() {
        let model = model();
        let result = evaluate_blade_structural_reference(
            &model,
            &bem(&model),
            BladeStructuralOptions::default(),
        )
        .expect("structural reference");
        assert!(result.is_finite());
        assert_eq!(result.stations.len(), model.stations().len());
        assert!(
            result.root_centrifugal_force_n
                >= result.stations.last().unwrap().outboard_centrifugal_force_n
        );
        assert!(result.max_combined_stress_pa >= result.stations[0].combined_stress_pa);
        assert!(result.min_margin_pa <= result.stations[0].margin_pa);
    }

    #[test]
    fn structural_reference_matches_lumped_centrifugal_force_sum() {
        let model = model();
        let options = BladeStructuralOptions {
            allowable_stress_pa: 1.0e12,
            ..BladeStructuralOptions::default()
        };
        let bem = bem(&model);
        let result = evaluate_blade_structural_reference(&model, &bem, options)
            .expect("structural reference");
        let omega = TAU * bem.loads.operating_point.rpm / 60.0;
        let expected = result
            .stations
            .iter()
            .map(|station| station.control_volume_mass_kg * omega * omega * station.radius_m)
            .sum::<f64>();
        assert!((result.root_centrifugal_force_n - expected).abs() <= 1.0e-12 * expected.max(1.0));
    }

    #[test]
    fn structural_reference_thinner_sections_raise_stress_and_reduce_margin() {
        let model = model();
        let bem = bem(&model);
        let thick = evaluate_blade_structural_reference(
            &model,
            &bem,
            BladeStructuralOptions {
                thickness_fraction: 0.20,
                ..BladeStructuralOptions::default()
            },
        )
        .unwrap();
        let thin = evaluate_blade_structural_reference(
            &model,
            &bem,
            BladeStructuralOptions {
                thickness_fraction: 0.08,
                ..BladeStructuralOptions::default()
            },
        )
        .unwrap();
        assert!(thin.max_combined_stress_pa > thick.max_combined_stress_pa);
        assert!(thin.min_margin_pa < thick.min_margin_pa);
    }

    #[test]
    fn structural_reference_rejects_invalid_options_and_bem_state() {
        let model = model();
        let bem = bem(&model);
        for options in [
            BladeStructuralOptions {
                thickness_fraction: 0.0,
                ..BladeStructuralOptions::default()
            },
            BladeStructuralOptions {
                thickness_fraction: 1.1,
                ..BladeStructuralOptions::default()
            },
            BladeStructuralOptions {
                safety_factor: 0.5,
                ..BladeStructuralOptions::default()
            },
        ] {
            assert!(matches!(
                evaluate_blade_structural_reference(&model, &bem, options),
                Err(BladeStructuralError::InvalidOptions { .. })
            ));
        }
        let mut unconverged = bem;
        unconverged.converged = false;
        assert!(matches!(
            evaluate_blade_structural_reference(
                &model,
                &unconverged,
                BladeStructuralOptions::default()
            ),
            Err(BladeStructuralError::InvalidBemResult { field: "converged" })
        ));
    }
}
