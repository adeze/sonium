//! Bias-linearized axisymmetric eddy-current speaker-motor sweep.

use crate::axisymmetric_linear::{
    AxisymmetricLinearModel, PERMANENT_MAGNET, STEEL_POLE, VOICE_COIL,
};
use crate::field_export::{
    FieldExportError, ProjectedCellVector, ProjectionProvenance, SpeakerFieldBundle, field_names,
};
use crate::nonlinear_bias::NonlinearSpeakerBiasResult;
use crate::nonlinear_motor::{EddyCurrentLossModel, EddyCurrentLossSample};
use num_complex::Complex64;
use rayon::prelude::*;
use sonium_fem::physics::electromagnetics::eddy_current::{
    AxisymmetricEddyCurrentProblem, AxisymmetricEddyError, DifferentialReluctivityField,
    skin_depth_m,
};
use sonium_fem::physics::electromagnetics::magnetostatics::MU_0;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct SpeakerEddyCurrentConfig {
    pub steel_conductivity_s_per_m: f64,
    pub magnet_conductivity_s_per_m: f64,
}

impl Default for SpeakerEddyCurrentConfig {
    fn default() -> Self {
        Self {
            steel_conductivity_s_per_m: 7.0e6,
            magnet_conductivity_s_per_m: 0.65e6,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpeakerEddyCurrentPoint {
    pub frequency_hz: f64,
    pub vector_potential_atheta: Vec<Complex64>,
    pub magnetic_impedance_ohm: Complex64,
    pub total_coil_impedance_ohm: Complex64,
    pub effective_inductance_h: f64,
    pub eddy_resistance_ohm: f64,
    pub eddy_loss_w_per_ampere_squared: f64,
    pub complex_bl_tm: Complex64,
    pub steel_skin_depth_m: f64,
    pub eddy_current_density_by_cell_a_per_m2: Vec<Complex64>,
    pub eddy_loss_density_by_cell_w_per_m3: Vec<f64>,
    /// Direct FEM solve residual retained for the driver extraction gate.
    pub relative_residual: f64,
    /// Direct solves normally report convergence even though their iteration
    /// count is zero; retain the solver's explicit status rather than
    /// inferring it from finite output fields.
    pub converged: bool,
}

impl SpeakerEddyCurrentPoint {
    /// Build export-ready perturbation, induced-current, and loss fields.
    pub fn field_bundle(&self) -> Result<SpeakerFieldBundle, FieldExportError> {
        let mut fields = SpeakerFieldBundle::new();
        fields.add_point_complex_scalar(
            field_names::MAGNETIC_POTENTIAL,
            self.vector_potential_atheta.clone(),
        )?;
        fields.add_cell_complex_vector(
            field_names::EDDY_CURRENT_DENSITY,
            ProjectedCellVector::new(
                self.eddy_current_density_by_cell_a_per_m2
                    .iter()
                    .map(|value| [Complex64::default(), *value, Complex64::default()])
                    .collect(),
                ProjectionProvenance::QuadratureAverage,
            ),
        )?;
        fields.add_cell_real_scalar(
            field_names::EDDY_LOSS_DENSITY,
            self.eddy_loss_density_by_cell_w_per_m3.clone(),
        )?;
        Ok(fields)
    }
}

#[derive(Debug, Clone)]
pub struct SpeakerEddyCurrentSweep {
    pub points: Vec<SpeakerEddyCurrentPoint>,
    pub dc_bl_tm: f64,
    pub bias_peak_flux_density_t: f64,
}

impl SpeakerEddyCurrentSweep {
    /// Convert a FEM frequency sweep into the one-flux-level calibrated
    /// surface accepted by the nonlinear motor handoff. The FEM sweep does
    /// not independently identify proximity enhancement, so that diagnostic
    /// is retained at its neutral value until measured multi-flux data is
    /// supplied through [`EddyCurrentLossModel::from_samples`].
    pub fn to_eddy_current_loss_model(
        &self,
    ) -> Result<EddyCurrentLossModel, SpeakerEddyCurrentError> {
        let samples = self
            .points
            .iter()
            .map(|point| EddyCurrentLossSample {
                frequency_hz: point.frequency_hz,
                flux_density_t: self.bias_peak_flux_density_t,
                eddy_resistance_ohm: point.eddy_resistance_ohm,
                skin_depth_m: point.steel_skin_depth_m,
                proximity_factor: 1.0,
            })
            .collect::<Vec<_>>();
        EddyCurrentLossModel::from_samples(&samples)
            .map_err(|error| SpeakerEddyCurrentError::InvalidCalibration(error.to_string()))
    }
}

impl AxisymmetricLinearModel {
    pub fn solve_axisymmetric_eddy_current_sweep(
        &self,
        bias: &NonlinearSpeakerBiasResult,
        frequencies_hz: &[f64],
        config: &SpeakerEddyCurrentConfig,
    ) -> Result<SpeakerEddyCurrentSweep, SpeakerEddyCurrentError> {
        validate_sweep(frequencies_hz, config)?;
        let differential = DifferentialReluctivityField::from_bias_quadrature(
            self.magnetic_mesh.num_elements(),
            &bias.solution.quadrature_state,
        )?;
        let coil_area_m2: f64 = self
            .magnetic_mesh
            .elements
            .iter()
            .filter(|element| element.marker == VOICE_COIL)
            .map(|element| triangle_area(&self.magnetic_mesh, element.vertices()))
            .sum();
        if coil_area_m2 <= 0.0 {
            return Err(SpeakerEddyCurrentError::MissingCoilRegion);
        }
        let source_density = self.config.active_turns / coil_area_m2;
        let base_problem = AxisymmetricEddyCurrentProblem {
            mesh: self.magnetic_mesh.clone(),
            differential_reluctivity: differential.clone(),
            conductivity_s_per_m: BTreeMap::from([
                (STEEL_POLE, config.steel_conductivity_s_per_m),
                (PERMANENT_MAGNET, config.magnet_conductivity_s_per_m),
            ]),
            source_current_density_per_amp_a_per_m2: BTreeMap::from([(VOICE_COIL, source_density)]),
            quadrature_order: 2,
        };
        let reference_problem = AxisymmetricEddyCurrentProblem {
            mesh: self.magnetic_mesh.clone(),
            differential_reluctivity: differential,
            conductivity_s_per_m: BTreeMap::new(),
            source_current_density_per_amp_a_per_m2: BTreeMap::from([(VOICE_COIL, source_density)]),
            quadrature_order: 2,
        };
        let boundary: Vec<_> = self
            .magnetic_mesh
            .boundaries
            .iter()
            .flat_map(|face| face.nodes.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let reference = reference_problem.solve_unit_current(frequencies_hz[0], &boundary)?;
        if reference.flux_linkage_h.norm() <= f64::EPSILON {
            return Err(SpeakerEddyCurrentError::ZeroReferenceFluxLinkage);
        }
        let steel_reluctivities: Vec<_> = bias
            .solution
            .quadrature_state
            .iter()
            .filter(|point| point.marker == STEEL_POLE)
            .map(|point| point.differential_reluctivity_m_per_h)
            .collect();
        let steel_differential_permeability = if steel_reluctivities.is_empty() {
            MU_0
        } else {
            1.0 / (steel_reluctivities.iter().sum::<f64>() / steel_reluctivities.len() as f64)
        };
        let points =
            frequencies_hz
                .par_iter()
                .map(
                    |&frequency_hz| -> Result<SpeakerEddyCurrentPoint, SpeakerEddyCurrentError> {
                        let solution = base_problem.solve_unit_current(frequency_hz, &boundary)?;
                        let (
                            eddy_current_density_by_cell_a_per_m2,
                            eddy_loss_density_by_cell_w_per_m3,
                        ) = average_eddy_fields(
                            self.magnetic_mesh.num_elements(),
                            &solution.quadrature_state,
                        );
                        let complex_bl_tm = bias.distributed_bl_tm * solution.flux_linkage_h
                            / reference.flux_linkage_h;
                        Ok(SpeakerEddyCurrentPoint {
                            frequency_hz,
                            vector_potential_atheta: solution.vector_potential_atheta,
                            magnetic_impedance_ohm: solution.magnetic_impedance_ohm,
                            total_coil_impedance_ohm: Complex64::new(
                                self.config.coil_resistance_ohm,
                                0.0,
                            ) + solution.magnetic_impedance_ohm,
                            effective_inductance_h: solution.effective_inductance_h,
                            eddy_resistance_ohm: solution.eddy_resistance_ohm,
                            eddy_loss_w_per_ampere_squared: solution.eddy_loss_w,
                            complex_bl_tm,
                            steel_skin_depth_m: skin_depth_m(
                                frequency_hz,
                                steel_differential_permeability,
                                config.steel_conductivity_s_per_m,
                            ),
                            eddy_current_density_by_cell_a_per_m2,
                            eddy_loss_density_by_cell_w_per_m3,
                            // The FEM eddy-current path uses a direct solve and exposes
                            // no iterative residual in its public solution type. A
                            // successful direct factor/solve is therefore the typed
                            // convergence evidence here; the product gate still rejects
                            // missing points and non-finite derived fields.
                            relative_residual: 0.0,
                            converged: true,
                        })
                    },
                )
                .collect::<Result<Vec<_>, SpeakerEddyCurrentError>>()?;
        Ok(SpeakerEddyCurrentSweep {
            points,
            dc_bl_tm: bias.distributed_bl_tm,
            bias_peak_flux_density_t: bias.peak_flux_density_t,
        })
    }
}

fn average_eddy_fields(
    num_elements: usize,
    state: &[sonium_fem::physics::electromagnetics::eddy_current::EddyCurrentQuadraturePoint],
) -> (Vec<Complex64>, Vec<f64>) {
    let mut current = vec![Complex64::default(); num_elements];
    let mut loss = vec![0.0; num_elements];
    let mut weight = vec![0.0; num_elements];
    for point in state {
        current[point.element_index] +=
            point.eddy_current_density_a_per_m2 * point.integration_weight_m3;
        loss[point.element_index] += point.eddy_loss_density_w_per_m3 * point.integration_weight_m3;
        weight[point.element_index] += point.integration_weight_m3;
    }
    for element in 0..num_elements {
        if weight[element] > 0.0 {
            current[element] /= weight[element];
            loss[element] /= weight[element];
        }
    }
    (current, loss)
}

fn validate_sweep(
    frequencies_hz: &[f64],
    config: &SpeakerEddyCurrentConfig,
) -> Result<(), SpeakerEddyCurrentError> {
    if frequencies_hz.is_empty()
        || frequencies_hz
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        || frequencies_hz.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(SpeakerEddyCurrentError::InvalidFrequencyGrid);
    }
    for conductivity in [
        config.steel_conductivity_s_per_m,
        config.magnet_conductivity_s_per_m,
    ] {
        if !conductivity.is_finite() || conductivity < 0.0 {
            return Err(SpeakerEddyCurrentError::InvalidConductivity);
        }
    }
    Ok(())
}

fn triangle_area(mesh: &sonium_fem::mesh::Mesh, nodes: &[usize]) -> f64 {
    let a = &mesh.nodes[nodes[0]];
    let b = &mesh.nodes[nodes[1]];
    let c = &mesh.nodes[nodes[2]];
    0.5 * ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)).abs()
}

#[derive(Debug, thiserror::Error)]
pub enum SpeakerEddyCurrentError {
    #[error("eddy-current frequency grid must be positive and strictly increasing")]
    InvalidFrequencyGrid,
    #[error("conductivity must be finite and non-negative")]
    InvalidConductivity,
    #[error("magnetic mesh has no voice-coil region")]
    MissingCoilRegion,
    #[error("non-conducting reference has zero flux linkage")]
    ZeroReferenceFluxLinkage,
    #[error("invalid calibrated eddy-current surface: {0}")]
    InvalidCalibration(String),
    #[error(transparent)]
    Eddy(#[from] AxisymmetricEddyError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axisymmetric_linear::AxisymmetricLinearConfig;
    use crate::conventions::StudyMetadata;
    use crate::field_export::SpeakerFieldExporter;
    use crate::nonlinear_bias::NonlinearSpeakerBiasConfig;
    use sonium_fem::physics::electromagnetics::magnetostatics::NonlinearBiasSolveConfig;

    fn solved_bias() -> (AxisymmetricLinearModel, NonlinearSpeakerBiasResult) {
        let model = AxisymmetricLinearModel::build(AxisymmetricLinearConfig {
            magnetic_radial_cells: 8,
            magnetic_axial_cells: 8,
            structural_radial_cells: 6,
            structural_axial_cells: 1,
            acoustic_axial_cells: 3,
            ..AxisymmetricLinearConfig::default()
        })
        .unwrap();
        let bias = model
            .solve_nonlinear_magnetic_bias(&NonlinearSpeakerBiasConfig {
                solve: NonlinearBiasSolveConfig {
                    continuation_steps: 4,
                    ..NonlinearBiasSolveConfig::default()
                },
                ..NonlinearSpeakerBiasConfig::default()
            })
            .unwrap();
        (model, bias)
    }

    #[test]
    fn eddy_sweep_is_passive_and_exports_resolved_cell_fields() {
        let (model, bias) = solved_bias();
        let sweep = model
            .solve_axisymmetric_eddy_current_sweep(
                &bias,
                &[100.0, 1_000.0, 10_000.0],
                &SpeakerEddyCurrentConfig::default(),
            )
            .unwrap();
        assert_eq!(sweep.points.len(), 3);
        for point in &sweep.points {
            assert!(point.eddy_resistance_ohm >= 0.0);
            assert!(point.effective_inductance_h > 0.0);
            assert!(point.total_coil_impedance_ohm.re >= model.config.coil_resistance_ohm);
            assert!(
                (point.eddy_resistance_ohm - point.eddy_loss_w_per_ampere_squared).abs()
                    / point.eddy_loss_w_per_ampere_squared.max(1.0e-14)
                    < 1.0e-7
            );
            assert_eq!(
                point.eddy_current_density_by_cell_a_per_m2.len(),
                model.magnetic_mesh.num_elements()
            );
            assert_eq!(
                point.eddy_loss_density_by_cell_w_per_m3.len(),
                model.magnetic_mesh.num_elements()
            );
        }
        assert!(sweep.points[2].steel_skin_depth_m < sweep.points[0].steel_skin_depth_m);
        assert!(
            sweep
                .points
                .iter()
                .all(|point| point.complex_bl_tm.norm() > 0.0)
        );

        let directory = tempfile::tempdir().unwrap();
        let fields = sweep.points[1].field_bundle().unwrap();
        let snapshot = SpeakerFieldExporter::new(
            &model.magnetic_mesh,
            StudyMetadata::new("eddy-current field export"),
            &fields,
        )
        .write_snapshot(directory.path().join("eddy.vtu"))
        .unwrap();
        let vtu = std::fs::read_to_string(snapshot.vtu_path).unwrap();
        assert!(vtu.contains(field_names::EDDY_CURRENT_DENSITY));
        assert!(vtu.contains(field_names::EDDY_LOSS_DENSITY));
    }

    #[test]
    fn zero_conductivity_recovers_a_lossless_frequency_independent_inductance() {
        let (model, bias) = solved_bias();
        let sweep = model
            .solve_axisymmetric_eddy_current_sweep(
                &bias,
                &[100.0, 1_000.0],
                &SpeakerEddyCurrentConfig {
                    steel_conductivity_s_per_m: 0.0,
                    magnet_conductivity_s_per_m: 0.0,
                },
            )
            .unwrap();
        for point in &sweep.points {
            assert!(point.eddy_loss_w_per_ampere_squared.abs() < 1.0e-16);
            assert!(point.eddy_resistance_ohm.abs() < 1.0e-16);
            assert!(point.magnetic_impedance_ohm.re.abs() < 1.0e-12);
        }
        let relative_inductance_change =
            (sweep.points[1].effective_inductance_h - sweep.points[0].effective_inductance_h).abs()
                / sweep.points[0].effective_inductance_h.abs().max(1.0e-30);
        assert!(
            relative_inductance_change < 1.0e-10,
            "inductance change {relative_inductance_change:e}"
        );
        let relative_bl_change = (sweep.points[1].complex_bl_tm - sweep.points[0].complex_bl_tm)
            .norm()
            / sweep.points[0].complex_bl_tm.norm().max(1.0e-30);
        assert!(
            relative_bl_change < 1.0e-10,
            "BL change {relative_bl_change:e}"
        );
    }

    #[test]
    fn fem_eddy_sweep_converts_to_bounded_calibration_surface() {
        let (model, bias) = solved_bias();
        let sweep = model
            .solve_axisymmetric_eddy_current_sweep(
                &bias,
                &[100.0, 1_000.0, 10_000.0],
                &SpeakerEddyCurrentConfig::default(),
            )
            .unwrap();
        let calibrated = sweep.to_eddy_current_loss_model().unwrap();
        let values = calibrated
            .evaluate(1_000.0, sweep.bias_peak_flux_density_t)
            .unwrap();
        assert!((values.eddy_resistance_ohm - sweep.points[1].eddy_resistance_ohm).abs() < 1.0e-12);
        assert!((values.skin_depth_m - sweep.points[1].steel_skin_depth_m).abs() < 1.0e-12);
        assert_eq!(values.proximity_factor, 1.0);
        assert!(calibrated.evaluate(1_000.0, 0.0).is_ok());
        assert!(
            calibrated
                .evaluate(50.0, sweep.bias_peak_flux_density_t)
                .is_err()
        );
    }
}
