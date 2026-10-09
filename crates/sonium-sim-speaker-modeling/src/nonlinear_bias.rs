//! Loudspeaker-specific nonlinear magnetic bias workflow.

use crate::axisymmetric_linear::{
    AIR_GAP, AxisymmetricLinearModel, MAGNETIC_AIR, PERMANENT_MAGNET, STEEL_POLE, STEEL_YOKE,
    VOICE_COIL,
};
use sonium_fem::mesh::Mesh;
use sonium_fem::physics::electromagnetics::magnetostatics::{
    AxisymmetricNonlinearMagneticProblem, BhCurve, BhCurveError, MU_0, MagneticConstitutiveLaw,
    MagneticEnergyReport, NonlinearBiasSolveConfig, NonlinearMagneticBiasSolution,
    NonlinearMagneticError, solve_nonlinear_axisymmetric_bias,
};
use std::collections::{BTreeMap, BTreeSet};

use crate::nonlinear_motor::{
    MotorMagneticFluxModulationPoint, MotorMagneticLaw, MotorReluctanceGeometry,
};

#[derive(Debug, Clone)]
pub struct NonlinearSpeakerBiasConfig {
    pub steel_bh_curve: BhCurve,
    pub solve: NonlinearBiasSolveConfig,
    /// Signed voice-coil current used as an additional magnetic source.
    /// Zero preserves the permanent-magnet bias solve.
    pub coil_current_a: f64,
}

impl Default for NonlinearSpeakerBiasConfig {
    fn default() -> Self {
        Self {
            steel_bh_curve: default_low_carbon_steel_bh_curve(),
            solve: NonlinearBiasSolveConfig::default(),
            coil_current_a: 0.0,
        }
    }
}

impl NonlinearSpeakerBiasConfig {
    pub fn validate(&self) -> Result<(), NonlinearSpeakerBiasError> {
        if !self.coil_current_a.is_finite() {
            return Err(NonlinearSpeakerBiasError::InvalidCoilCurrent);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct NonlinearSpeakerBiasResult {
    pub solution: NonlinearMagneticBiasSolution,
    pub distributed_bl_tm: f64,
    pub linear_reference_bl_tm: f64,
    pub energy: MagneticEnergyReport,
    pub peak_flux_density_t: f64,
    pub saturated_quadrature_fraction: f64,
    /// Axisymmetric integration volume of the nonlinear steel regions.
    pub steel_active_volume_m3: f64,
    pub coil_current_a: f64,
    pub coil_mean_flux_density_t: f64,
}

/// Summary of one signed current point in a nonlinear magnetic sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonlinearSpeakerBiasCurrentSample {
    pub current_a: f64,
    pub coil_mean_flux_density_t: f64,
    pub distributed_bl_tm: f64,
    pub steel_active_volume_m3: f64,
}

/// FEM current-sweep identification around the permanent-magnet operating
/// point. The full converged result is retained only for zero current; the
/// other points remain as compact fitted-input evidence.
#[derive(Debug, Clone)]
pub struct NonlinearSpeakerBiasCurrentSweep {
    pub baseline: NonlinearSpeakerBiasResult,
    pub samples: Vec<NonlinearSpeakerBiasCurrentSample>,
    pub flux_modulation_t_per_a: f64,
}

impl NonlinearSpeakerBiasCurrentSweep {
    pub fn to_motor_magnetic_law(
        &self,
        config: &NonlinearSpeakerBiasConfig,
        reluctance: Option<MotorReluctanceGeometry>,
    ) -> Result<MotorMagneticLaw, crate::ModelingError> {
        let baseline_flux_density_t = self.baseline.coil_mean_flux_density_t;
        let modulation_curve = self
            .samples
            .iter()
            .map(|sample| MotorMagneticFluxModulationPoint {
                current_a: sample.current_a,
                delta_flux_density_t: sample.coil_mean_flux_density_t - baseline_flux_density_t,
            })
            .collect();
        MotorMagneticLaw::from_bh_samples_with_flux_modulation_curve(
            config.steel_bh_curve.points(),
            self.baseline.steel_active_volume_m3,
            self.flux_modulation_t_per_a,
            modulation_curve,
            reluctance,
        )
    }
}

impl NonlinearSpeakerBiasResult {
    /// Convert the converged FEM steel law into the serializable driver
    /// handoff.  The active volume comes from the solved steel quadrature
    /// state rather than a guessed cabinet or coil volume.
    pub fn to_motor_magnetic_law(
        &self,
        config: &NonlinearSpeakerBiasConfig,
        flux_modulation_t_per_a: f64,
        reluctance: Option<MotorReluctanceGeometry>,
    ) -> Result<MotorMagneticLaw, crate::ModelingError> {
        MotorMagneticLaw::from_bh_samples(
            config.steel_bh_curve.points(),
            self.steel_active_volume_m3,
            flux_modulation_t_per_a,
            reluctance,
        )
    }
}

impl AxisymmetricLinearModel {
    /// Replace the linear steel bias with a monotone B-H Newton solve.
    pub fn solve_nonlinear_magnetic_bias(
        &self,
        config: &NonlinearSpeakerBiasConfig,
    ) -> Result<NonlinearSpeakerBiasResult, NonlinearSpeakerBiasError> {
        config.validate()?;
        let coil_area_m2 = region_cross_section_area_m2(&self.magnetic_mesh, VOICE_COIL)?;
        let coil_current_density_a_per_m2 =
            self.config.active_turns * config.coil_current_a / coil_area_m2;
        let materials = BTreeMap::from([
            (MAGNETIC_AIR, MagneticConstitutiveLaw::linear(MU_0)?),
            (
                PERMANENT_MAGNET,
                MagneticConstitutiveLaw::linear(1.05 * MU_0)?,
            ),
            (
                STEEL_POLE,
                MagneticConstitutiveLaw::Nonlinear(config.steel_bh_curve.clone()),
            ),
            (
                STEEL_YOKE,
                MagneticConstitutiveLaw::Nonlinear(config.steel_bh_curve.clone()),
            ),
            (VOICE_COIL, MagneticConstitutiveLaw::linear(MU_0)?),
            (AIR_GAP, MagneticConstitutiveLaw::linear(MU_0)?),
        ]);
        let problem = AxisymmetricNonlinearMagneticProblem {
            mesh: self.magnetic_mesh.clone(),
            materials,
            azimuthal_current_density_a_per_m2: {
                let mut sources = BTreeMap::from([(
                    PERMANENT_MAGNET,
                    self.config.remanence_t / (MU_0 * 20.0e-3),
                )]);
                if coil_current_density_a_per_m2 != 0.0 {
                    sources.insert(VOICE_COIL, coil_current_density_a_per_m2);
                }
                sources
            },
            quadrature_order: 2,
        };
        let constraints: Vec<_> = problem
            .mesh
            .boundaries
            .iter()
            .flat_map(|boundary| boundary.nodes.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let solution = solve_nonlinear_axisymmetric_bias(&problem, &constraints, &config.solve)?;
        let distributed_bl_tm = problem.extract_force_factor(
            &solution.quadrature_state,
            VOICE_COIL,
            self.config.active_turns,
        )?;
        let peak_flux_density_t = solution
            .quadrature_state
            .iter()
            .map(|point| point.flux_density_rz_t[0].hypot(point.flux_density_rz_t[1]))
            .fold(0.0, f64::max);
        let steel_points: Vec<_> = solution
            .quadrature_state
            .iter()
            .filter(|point| matches!(point.marker, STEEL_POLE | STEEL_YOKE))
            .collect();
        let saturated = steel_points
            .iter()
            .filter(|point| {
                point.differential_reluctivity_m_per_h
                    > 10.0 * config.steel_bh_curve.reluctivity(0.5)
            })
            .count();
        let saturated_quadrature_fraction = if steel_points.is_empty() {
            0.0
        } else {
            saturated as f64 / steel_points.len() as f64
        };
        let steel_active_volume_m3 = steel_points
            .iter()
            .map(|point| point.integration_weight_m3)
            .sum::<f64>();
        if !steel_active_volume_m3.is_finite() || steel_active_volume_m3 <= 0.0 {
            return Err(NonlinearSpeakerBiasError::InvalidSteelVolume);
        }
        let coil_points: Vec<_> = solution
            .quadrature_state
            .iter()
            .filter(|point| point.marker == VOICE_COIL)
            .collect();
        let coil_volume_m3 = coil_points
            .iter()
            .map(|point| point.integration_weight_m3)
            .sum::<f64>();
        if !coil_volume_m3.is_finite() || coil_volume_m3 <= 0.0 {
            return Err(NonlinearSpeakerBiasError::InvalidCoilRegion);
        }
        let coil_mean_flux_density_t = coil_points
            .iter()
            .map(|point| {
                point.flux_density_rz_t[0].hypot(point.flux_density_rz_t[1])
                    * point.integration_weight_m3
            })
            .sum::<f64>()
            / coil_volume_m3;
        if !coil_mean_flux_density_t.is_finite() {
            return Err(NonlinearSpeakerBiasError::InvalidCoilRegion);
        }
        let energy = solution.energy;
        Ok(NonlinearSpeakerBiasResult {
            solution,
            distributed_bl_tm,
            linear_reference_bl_tm: self.magnetic_bias.distributed_bl_tm,
            energy,
            peak_flux_density_t,
            saturated_quadrature_fraction,
            steel_active_volume_m3,
            coil_current_a: config.coil_current_a,
            coil_mean_flux_density_t,
        })
    }

    /// Solve a bounded signed coil-current sweep and fit the resulting flux
    /// modulation around the permanent-magnet bias point.
    pub fn solve_nonlinear_magnetic_bias_current_sweep(
        &self,
        config: &NonlinearSpeakerBiasConfig,
        coil_currents_a: &[f64],
    ) -> Result<NonlinearSpeakerBiasCurrentSweep, NonlinearSpeakerBiasError> {
        config.validate()?;
        if coil_currents_a.len() < 3
            || coil_currents_a.len() > 33
            || coil_currents_a
                .windows(2)
                .any(|window| !window[0].is_finite() || window[1] <= window[0])
            || coil_currents_a
                .iter()
                .any(|current_a| !current_a.is_finite())
            || !coil_currents_a.iter().any(|current_a| *current_a < 0.0)
            || !coil_currents_a.contains(&0.0)
            || !coil_currents_a.iter().any(|current_a| *current_a > 0.0)
        {
            return Err(NonlinearSpeakerBiasError::InvalidCurrentSweep);
        }

        let mut solved = Vec::with_capacity(coil_currents_a.len());
        for &current_a in coil_currents_a {
            let mut point_config = config.clone();
            point_config.coil_current_a = current_a;
            solved.push(self.solve_nonlinear_magnetic_bias(&point_config)?);
        }
        let baseline_index = solved
            .iter()
            .position(|point| point.coil_current_a == 0.0)
            .ok_or(NonlinearSpeakerBiasError::InvalidCurrentSweep)?;
        let baseline_flux_density_t = solved[baseline_index].coil_mean_flux_density_t;
        let samples: Vec<_> = solved
            .iter()
            .map(|point| NonlinearSpeakerBiasCurrentSample {
                current_a: point.coil_current_a,
                coil_mean_flux_density_t: point.coil_mean_flux_density_t,
                distributed_bl_tm: point.distributed_bl_tm,
                steel_active_volume_m3: point.steel_active_volume_m3,
            })
            .collect();
        let (numerator, denominator) =
            samples
                .iter()
                .fold((0.0, 0.0), |(numerator, denominator), sample| {
                    let delta_flux_density_t =
                        sample.coil_mean_flux_density_t - baseline_flux_density_t;
                    (
                        numerator + sample.current_a * delta_flux_density_t,
                        denominator + sample.current_a * sample.current_a,
                    )
                });
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(NonlinearSpeakerBiasError::InvalidCurrentSweep);
        }
        let flux_modulation_t_per_a = numerator / denominator;
        if !flux_modulation_t_per_a.is_finite() {
            return Err(NonlinearSpeakerBiasError::InvalidCurrentSweep);
        }
        let baseline = solved.swap_remove(baseline_index);
        Ok(NonlinearSpeakerBiasCurrentSweep {
            baseline,
            samples,
            flux_modulation_t_per_a,
        })
    }
}

pub fn default_low_carbon_steel_bh_curve() -> BhCurve {
    BhCurve::new(vec![
        (0.0, 0.0),
        (0.2, 55.0),
        (0.5, 120.0),
        (0.8, 210.0),
        (1.0, 320.0),
        (1.2, 520.0),
        (1.4, 900.0),
        (1.55, 1_600.0),
        (1.65, 3_200.0),
        (1.75, 8_500.0),
        (1.85, 24_000.0),
        (2.0, 85_000.0),
        (2.2, 220_000.0),
    ])
    .expect("built-in steel B-H data are strictly monotone")
}

#[derive(Debug, thiserror::Error)]
pub enum NonlinearSpeakerBiasError {
    #[error(transparent)]
    Curve(#[from] BhCurveError),
    #[error(transparent)]
    Magnetic(#[from] NonlinearMagneticError),
    #[error("nonlinear bias contained no positive steel integration volume")]
    InvalidSteelVolume,
    #[error("nonlinear bias coil current must be finite")]
    InvalidCoilCurrent,
    #[error(
        "nonlinear bias current sweep must contain strictly increasing negative, zero, and positive currents"
    )]
    InvalidCurrentSweep,
    #[error("nonlinear bias contained no positive voice-coil region")]
    InvalidCoilRegion,
}

fn region_cross_section_area_m2(
    mesh: &Mesh,
    marker: i32,
) -> Result<f64, NonlinearSpeakerBiasError> {
    let mut area_m2 = 0.0;
    for element in mesh
        .elements
        .iter()
        .filter(|element| element.marker == marker)
    {
        if element.nodes.len() != 3 {
            return Err(NonlinearSpeakerBiasError::InvalidCoilRegion);
        }
        let p0 = &mesh.nodes[element.nodes[0]];
        let p1 = &mesh.nodes[element.nodes[1]];
        let p2 = &mesh.nodes[element.nodes[2]];
        area_m2 += 0.5 * ((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)).abs();
    }
    if !area_m2.is_finite() || area_m2 <= 0.0 {
        return Err(NonlinearSpeakerBiasError::InvalidCoilRegion);
    }
    Ok(area_m2)
}

/// Marker base for per-design-element nonlinear materials in the map
/// extraction solve (design position `i` gets `MARKER_BASE + i`). Far above
/// the region/boundary markers of the axisymmetric model (10..=113).
const DENSITY_MATERIAL_MARKER_BASE: i32 = 1_000_000;

/// Per-design-element permeability map extracted from a density-aware
/// nonlinear bias solve, used by the motor topology optimizer's
/// `SteelPermeability::DifferentialFromBias` option.
#[derive(Debug, Clone)]
pub struct SteelPermeabilityMap {
    /// Relative differential permeability `1 / (mu_0 * dH/dB)` of the steel
    /// B-H curve at each design element's converged flux density, clamped to
    /// `>= 1` so the density interpolation stays above air.
    pub differential_relative_permeability: Vec<f64>,
    /// Integration-weighted mean flux-density magnitude `|B|` per design
    /// element (T), same ordering as `differential_relative_permeability`.
    pub mean_flux_density_t: Vec<f64>,
}

/// Density-scaled copy of a steel B-H curve: permeability (not reluctivity)
/// is interpolated between air and steel, `mu(B; w) = mu_0 + w*(mu_steel(B)
/// - mu_0)`, mirroring the topology optimizer's RAMP permeability
/// interpolation. In the linear limit (a straight-line `curve`) the scaled
/// curve reproduces `mu_r = 1 + w*(mu_r_steel - 1)` exactly.
fn density_scaled_curve(curve: &BhCurve, weight: f64) -> Result<BhCurve, BhCurveError> {
    let weight = weight.clamp(0.0, 1.0);
    let points = curve
        .points()
        .iter()
        .map(|&(b, h)| {
            if b <= f64::EPSILON {
                (b, h) // the origin (0, 0)
            } else {
                let steel_permeability = b / h;
                (b, b / (MU_0 + weight * (steel_permeability - MU_0)))
            }
        })
        .collect();
    BhCurve::new(points)
}

/// Extract a per-design-element differential permeability map from a
/// nonlinear bias solve of the CURRENT design on the topology mesh itself
/// (same axisymmetric `A_theta` formulation as the linear forward model, so
/// no mesh transfer is needed).
///
/// Each design element is given its own marker and a density-scaled
/// nonlinear constitutive law (see [`density_scaled_curve`]) with
/// `w = interp_weights[position]` — typically `interp(rho_e)` of the current
/// physical densities — so the converged field reflects the gray layout.
/// The map value is the UNscaled steel curve's relative differential
/// permeability at the element's mean `|B|`; the density scaling is
/// re-applied by the optimizer's interpolation on top of it.
pub fn steel_permeability_map_from_bias(
    mesh: &Mesh,
    remanence_t: f64,
    design_elements: &[usize],
    interp_weights: &[f64],
    curve: &BhCurve,
    solve: &NonlinearBiasSolveConfig,
) -> Result<SteelPermeabilityMap, NonlinearSpeakerBiasError> {
    if design_elements.len() != interp_weights.len() {
        return Err(NonlinearMagneticError::DimensionMismatch {
            expected: design_elements.len(),
            actual: interp_weights.len(),
        }
        .into());
    }
    let mut scaled_mesh = mesh.clone();
    let mut materials = BTreeMap::from([
        (MAGNETIC_AIR, MagneticConstitutiveLaw::linear(MU_0)?),
        (
            PERMANENT_MAGNET,
            MagneticConstitutiveLaw::linear(1.05 * MU_0)?,
        ),
        (VOICE_COIL, MagneticConstitutiveLaw::linear(MU_0)?),
        (AIR_GAP, MagneticConstitutiveLaw::linear(MU_0)?),
    ]);
    for (position, (&element, &weight)) in design_elements.iter().zip(interp_weights).enumerate() {
        let marker = DENSITY_MATERIAL_MARKER_BASE + position as i32;
        scaled_mesh.elements[element].marker = marker;
        materials.insert(
            marker,
            MagneticConstitutiveLaw::Nonlinear(density_scaled_curve(curve, weight)?),
        );
    }
    let problem = AxisymmetricNonlinearMagneticProblem {
        mesh: scaled_mesh,
        materials,
        azimuthal_current_density_a_per_m2: BTreeMap::from([(
            PERMANENT_MAGNET,
            remanence_t / (MU_0 * 20.0e-3),
        )]),
        quadrature_order: 2,
    };
    let constraints: Vec<_> = problem
        .mesh
        .boundaries
        .iter()
        .flat_map(|boundary| boundary.nodes.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let solution = solve_nonlinear_axisymmetric_bias(&problem, &constraints, solve)?;

    let mut weighted_flux = vec![0.0; design_elements.len()];
    let mut weights = vec![0.0; design_elements.len()];
    for point in &solution.quadrature_state {
        if point.marker >= DENSITY_MATERIAL_MARKER_BASE {
            let position = (point.marker - DENSITY_MATERIAL_MARKER_BASE) as usize;
            let magnitude = point.flux_density_rz_t[0].hypot(point.flux_density_rz_t[1]);
            weighted_flux[position] += magnitude * point.integration_weight_m3;
            weights[position] += point.integration_weight_m3;
        }
    }
    let mut mean_flux_density_t = Vec::with_capacity(design_elements.len());
    let mut differential_relative_permeability = Vec::with_capacity(design_elements.len());
    for position in 0..design_elements.len() {
        let mean_b = if weights[position] > 0.0 {
            weighted_flux[position] / weights[position]
        } else {
            0.0
        };
        mean_flux_density_t.push(mean_b);
        differential_relative_permeability
            .push((1.0 / (MU_0 * curve.differential_reluctivity(mean_b))).max(1.0));
    }
    Ok(SteelPermeabilityMap {
        differential_relative_permeability,
        mean_flux_density_t,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::axisymmetric_linear::AxisymmetricLinearConfig;

    fn coarse_model() -> AxisymmetricLinearModel {
        AxisymmetricLinearModel::build(AxisymmetricLinearConfig {
            magnetic_radial_cells: 8,
            magnetic_axial_cells: 8,
            structural_radial_cells: 6,
            structural_axial_cells: 1,
            acoustic_axial_cells: 3,
            ..AxisymmetricLinearConfig::default()
        })
        .unwrap()
    }

    #[test]
    fn nonlinear_bias_converges_with_continuation_and_stores_material_state() {
        let model = coarse_model();
        let config = NonlinearSpeakerBiasConfig {
            solve: NonlinearBiasSolveConfig {
                continuation_steps: 4,
                ..NonlinearBiasSolveConfig::default()
            },
            ..NonlinearSpeakerBiasConfig::default()
        };
        let result = model.solve_nonlinear_magnetic_bias(&config).unwrap();
        assert_eq!(result.solution.continuation_iterations.len(), 4);
        assert!(result.solution.final_residual.is_finite());
        assert!(result.distributed_bl_tm > 0.0);
        assert!(result.peak_flux_density_t > 0.0);
        assert!(
            result
                .solution
                .quadrature_state
                .iter()
                .any(|point| point.marker == STEEL_POLE)
        );
        assert!(
            result
                .solution
                .quadrature_state
                .iter()
                .all(|point| point.differential_reluctivity_m_per_h > 0.0)
        );
        assert!(result.steel_active_volume_m3 > 0.0);
        let law = result.to_motor_magnetic_law(&config, 0.0, None).unwrap();
        assert_eq!(law.active_volume_m3, result.steel_active_volume_m3);
        assert_eq!(law.bh_curve.len(), config.steel_bh_curve.points().len());
    }

    #[test]
    fn converged_bias_satisfies_energy_coenergy_identity() {
        let model = coarse_model();
        let result = model
            .solve_nonlinear_magnetic_bias(&NonlinearSpeakerBiasConfig {
                solve: NonlinearBiasSolveConfig {
                    continuation_steps: 4,
                    ..NonlinearBiasSolveConfig::default()
                },
                ..NonlinearSpeakerBiasConfig::default()
            })
            .unwrap();
        let scale = result.energy.source_work_j.abs().max(1.0e-12);
        assert!(result.energy.balance_error_j.abs() / scale < 1.0e-6);
        assert!(result.energy.energy_j > 0.0);
        assert!(result.energy.coenergy_j > 0.0);
    }

    #[test]
    fn coil_current_is_retained_and_drives_the_nonlinear_bias_source() {
        let model = coarse_model();
        let config = NonlinearSpeakerBiasConfig {
            coil_current_a: 0.5,
            solve: NonlinearBiasSolveConfig {
                continuation_steps: 2,
                ..NonlinearBiasSolveConfig::default()
            },
            ..NonlinearSpeakerBiasConfig::default()
        };
        let result = model.solve_nonlinear_magnetic_bias(&config).unwrap();
        assert_eq!(result.coil_current_a, 0.5);
        assert!(result.coil_mean_flux_density_t.is_finite());
        assert!(result.energy.source_work_j.is_finite());
    }

    #[test]
    fn nonlinear_bias_rejects_nonfinite_coil_current_before_mesh_work() {
        let mut config = NonlinearSpeakerBiasConfig::default();
        config.coil_current_a = f64::NAN;
        assert!(matches!(
            config.validate(),
            Err(NonlinearSpeakerBiasError::InvalidCoilCurrent)
        ));
    }

    #[test]
    fn current_sweep_fits_and_retains_flux_modulation_evidence() {
        let model = coarse_model();
        let config = NonlinearSpeakerBiasConfig {
            solve: NonlinearBiasSolveConfig {
                continuation_steps: 2,
                ..NonlinearBiasSolveConfig::default()
            },
            ..NonlinearSpeakerBiasConfig::default()
        };
        let sweep = model
            .solve_nonlinear_magnetic_bias_current_sweep(&config, &[-0.5, 0.0, 0.5])
            .unwrap();

        assert_eq!(sweep.baseline.coil_current_a, 0.0);
        assert_eq!(sweep.samples.len(), 3);
        assert!(sweep.flux_modulation_t_per_a.is_finite());
        assert!(
            sweep
                .samples
                .iter()
                .all(|sample| sample.steel_active_volume_m3 > 0.0)
        );

        let law = sweep.to_motor_magnetic_law(&config, None).unwrap();
        assert_eq!(law.flux_modulation_curve.len(), 3);
        assert_eq!(law.flux_modulation_curve[1].delta_flux_density_t, 0.0);
        law.validate().unwrap();
    }

    #[test]
    fn current_sweep_requires_signed_zero_crossing_grid() {
        let model = coarse_model();
        let error = model.solve_nonlinear_magnetic_bias_current_sweep(
            &NonlinearSpeakerBiasConfig::default(),
            &[0.0, 0.5, 1.0],
        );
        assert!(matches!(
            error,
            Err(NonlinearSpeakerBiasError::InvalidCurrentSweep)
        ));
    }

    #[test]
    fn nonlinear_path_recovers_the_linear_steel_limit() {
        let model = coarse_model();
        let permeability = 1_000.0 * MU_0;
        let linear_curve = BhCurve::new(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
                .into_iter()
                .map(|flux_density| (flux_density, flux_density / permeability))
                .collect(),
        )
        .unwrap();
        let result = model
            .solve_nonlinear_magnetic_bias(&NonlinearSpeakerBiasConfig {
                steel_bh_curve: linear_curve,
                solve: NonlinearBiasSolveConfig {
                    continuation_steps: 2,
                    ..NonlinearBiasSolveConfig::default()
                },
                coil_current_a: 0.0,
            })
            .unwrap();
        let relative_bl_error = (result.distributed_bl_tm - result.linear_reference_bl_tm).abs()
            / result.linear_reference_bl_tm.abs().max(1.0e-15);
        assert!(relative_bl_error < 1.0e-6, "BL error {relative_bl_error:e}");
        assert!(result.saturated_quadrature_fraction == 0.0);
    }
}
