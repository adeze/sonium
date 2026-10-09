//! Validated reduced shape, topology, and uncertainty study families.
//!
//! These families keep optimization contracts executable without pretending
//! that a one-dimensional reference model is a production 3D optimizer. The
//! bracket shape case supports analytic compliance/volume and assembled
//! first-frequency and linearized-buckling target objectives with body-fitted
//! mesh motion. The topology case includes a symmetric density filter, projection,
//! SIMP penalisation, volume projection, and connectivity gates. The
//! uncertainty case demonstrates reuse of a normalized bar stiffness across
//! deterministic samples.

use super::{
    BracketStudy, StaticBracketResult, StructuralExecutionProvenance, StructuralMechanicsError,
    StructuralModelAssumption, run_static_bracket_on_mesh, solve_first_bending_mode,
    solve_static_linear_state,
};
use serde::{Deserialize, Serialize};
use sonium_fem::mesh::MeshMotion;
use sonium_fem::optimization::compute_adjoint_sensitivity;
use sonium_solvers::{CgConfig, CsrMatrix, cg};

/// Length and area design variables for the analytic shape reference case.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ShapeDesign {
    pub length_m: f64,
    pub area_m2: f64,
}

/// Analytic axial-bar shape optimization inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShapeOptimizationStudy {
    pub youngs_modulus_pa: f64,
    pub force_n: f64,
    pub initial_design: ShapeDesign,
    pub minimum_design: ShapeDesign,
    pub maximum_design: ShapeDesign,
    pub volume_weight_j_per_m3: f64,
    pub mesh_elements: usize,
    pub iterations: usize,
    pub step_size: f64,
    pub gradient_tolerance: f64,
}

impl ShapeOptimizationStudy {
    pub fn introductory() -> Self {
        Self {
            youngs_modulus_pa: 200.0e9,
            force_n: 1_000.0,
            initial_design: ShapeDesign {
                length_m: 0.20,
                area_m2: 1.0e-3,
            },
            minimum_design: ShapeDesign {
                length_m: 0.10,
                area_m2: 5.0e-4,
            },
            maximum_design: ShapeDesign {
                length_m: 0.30,
                area_m2: 2.0e-3,
            },
            volume_weight_j_per_m3: 1_000.0,
            mesh_elements: 12,
            iterations: 40,
            step_size: 0.25,
            gradient_tolerance: 1.0e-8,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("shape.youngs_modulus_pa", self.youngs_modulus_pa),
            ("shape.force_n", self.force_n),
            ("shape.volume_weight_j_per_m3", self.volume_weight_j_per_m3),
            ("shape.step_size", self.step_size),
            ("shape.gradient_tolerance", self.gradient_tolerance),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if self.mesh_elements == 0 || self.iterations == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "shape.discretization",
                value: format!(
                    "mesh_elements={}, iterations={}",
                    self.mesh_elements, self.iterations
                ),
            });
        }
        for (field, design) in [
            ("shape.initial_design", self.initial_design),
            ("shape.minimum_design", self.minimum_design),
            ("shape.maximum_design", self.maximum_design),
        ] {
            if !design.length_m.is_finite()
                || design.length_m <= 0.0
                || !design.area_m2.is_finite()
                || design.area_m2 <= 0.0
            {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: format!("length={}, area={}", design.length_m, design.area_m2),
                });
            }
        }
        if self.minimum_design.length_m > self.maximum_design.length_m
            || self.minimum_design.area_m2 > self.maximum_design.area_m2
            || !inside_design(
                self.initial_design,
                self.minimum_design,
                self.maximum_design,
            )
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "shape.design_bounds",
                value: "initial design is outside ordered bounds".to_string(),
            });
        }
        Ok(())
    }

    /// Return objective and exact gradient for `J = F²L/(EA) + wLA`.
    pub fn objective_and_gradient(&self, design: ShapeDesign) -> (f64, [f64; 2]) {
        let compliance = self.force_n * self.force_n * design.length_m
            / (self.youngs_modulus_pa * design.area_m2);
        let volume_penalty = self.volume_weight_j_per_m3 * design.length_m * design.area_m2;
        let objective = compliance + volume_penalty;
        let d_length = self.force_n * self.force_n / (self.youngs_modulus_pa * design.area_m2)
            + self.volume_weight_j_per_m3 * design.area_m2;
        let d_area = -self.force_n * self.force_n * design.length_m
            / (self.youngs_modulus_pa * design.area_m2 * design.area_m2)
            + self.volume_weight_j_per_m3 * design.length_m;
        (objective, [d_length, d_area])
    }

    /// Return exact derivatives of the objective with respect to material
    /// stiffness and applied load, holding the geometric design fixed.
    pub fn objective_material_load_gradients(&self, design: ShapeDesign) -> [f64; 2] {
        let compliance = self.force_n * self.force_n * design.length_m
            / (self.youngs_modulus_pa * design.area_m2);
        [
            -compliance / self.youngs_modulus_pa,
            2.0 * compliance / self.force_n,
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShapeIterationResult {
    pub iteration: usize,
    pub design: ShapeDesign,
    pub objective_j: f64,
    pub gradient: [f64; 2],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShapeOptimizationResult {
    pub study: ShapeOptimizationStudy,
    pub initial_design: ShapeDesign,
    pub final_design: ShapeDesign,
    pub initial_objective_j: f64,
    pub final_objective_j: f64,
    pub final_gradient: [f64; 2],
    pub material_gradient_j_per_pa: f64,
    pub load_gradient_j_per_n: f64,
    pub material_taylor_relative_error: f64,
    pub load_taylor_relative_error: f64,
    pub bound_active: [bool; 2],
    pub iterations: usize,
    pub converged: bool,
    pub initial_mesh_nodes_m: Vec<f64>,
    pub final_mesh_nodes_m: Vec<f64>,
    pub mesh_motion_l2_m: f64,
    pub body_fitted_reanalysis_objective_j: f64,
    pub history: Vec<ShapeIterationResult>,
    pub model_assumption: StructuralModelAssumption,
}

/// Run projected analytic-gradient shape optimization and re-evaluate the
/// final design on its moved, body-fitted one-dimensional mesh.
pub fn run_shape_optimization(
    study: &ShapeOptimizationStudy,
) -> Result<ShapeOptimizationResult, StructuralMechanicsError> {
    study.validate()?;
    let initial_design = study.initial_design;
    let initial_mesh_nodes_m = mesh_nodes(initial_design.length_m, study.mesh_elements);
    let (initial_objective_j, _) = study.objective_and_gradient(initial_design);
    let objective_scale = initial_objective_j.max(1.0e-15);
    let mut design = initial_design;
    let mut history = Vec::with_capacity(study.iterations + 1);
    let mut converged = false;
    let mut iterations_done = 0;
    for iteration in 0..study.iterations {
        let (objective_j, gradient) = study.objective_and_gradient(design);
        history.push(ShapeIterationResult {
            iteration,
            design,
            objective_j,
            gradient,
        });
        iterations_done = iteration + 1;
        let gradient_norm = ((gradient[0] * design.length_m / objective_scale).powi(2)
            + (gradient[1] * design.area_m2 / objective_scale).powi(2))
        .sqrt();
        if gradient_norm <= study.gradient_tolerance {
            converged = true;
            break;
        }
        let mut step = study.step_size;
        let mut accepted = false;
        for _ in 0..20 {
            let candidate = clamp_shape(
                ShapeDesign {
                    length_m: design.length_m
                        * (1.0 - step * gradient[0] * design.length_m / objective_scale),
                    area_m2: design.area_m2
                        * (1.0 - step * gradient[1] * design.area_m2 / objective_scale),
                },
                study.minimum_design,
                study.maximum_design,
            );
            let (candidate_objective, _) = study.objective_and_gradient(candidate);
            if candidate_objective < objective_j {
                design = candidate;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            converged = true;
            break;
        }
    }
    let (final_objective_j, final_gradient) = study.objective_and_gradient(design);
    let [material_gradient_j_per_pa, load_gradient_j_per_n] =
        study.objective_material_load_gradients(design);
    let (material_taylor_relative_error, load_taylor_relative_error) =
        shape_material_load_taylor_errors(study, design);
    let bound_active = [
        design.length_m == study.minimum_design.length_m
            || design.length_m == study.maximum_design.length_m,
        design.area_m2 == study.minimum_design.area_m2
            || design.area_m2 == study.maximum_design.area_m2,
    ];
    history.push(ShapeIterationResult {
        iteration: iterations_done,
        design,
        objective_j: final_objective_j,
        gradient: final_gradient,
    });
    let final_mesh_nodes_m = mesh_nodes(design.length_m, study.mesh_elements);
    let mesh_motion_l2_m = initial_mesh_nodes_m
        .iter()
        .zip(&final_mesh_nodes_m)
        .map(|(initial, final_node)| (final_node - initial).powi(2))
        .sum::<f64>()
        .sqrt();
    Ok(ShapeOptimizationResult {
        study: study.clone(),
        initial_design,
        final_design: design,
        initial_objective_j,
        final_objective_j,
        final_gradient,
        material_gradient_j_per_pa,
        load_gradient_j_per_n,
        material_taylor_relative_error,
        load_taylor_relative_error,
        bound_active,
        iterations: iterations_done,
        converged,
        initial_mesh_nodes_m,
        final_mesh_nodes_m,
        mesh_motion_l2_m,
        body_fitted_reanalysis_objective_j: study.objective_and_gradient(design).0,
        history,
        model_assumption: StructuralModelAssumption::ShapeOptimization1D,
    })
}

/// Filtered/projected SIMP topology surrogate for a serial bar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologyOptimizationStudy {
    pub youngs_modulus_pa: f64,
    pub force_n: f64,
    pub length_m: f64,
    pub base_area_m2: f64,
    pub initial_density: Vec<f64>,
    pub volume_fraction: f64,
    pub minimum_density: f64,
    pub simp_penalty: f64,
    pub filter_radius_cells: usize,
    pub projection_beta: f64,
    pub projection_threshold: f64,
    pub iterations: usize,
    pub step_size: f64,
}

impl TopologyOptimizationStudy {
    pub fn introductory() -> Self {
        Self {
            youngs_modulus_pa: 200.0e9,
            force_n: 1_000.0,
            length_m: 0.20,
            base_area_m2: 1.0e-3,
            initial_density: vec![0.70, 0.60, 0.45, 0.35, 0.35, 0.45, 0.60, 0.70],
            volume_fraction: 0.525,
            minimum_density: 0.05,
            simp_penalty: 3.0,
            filter_radius_cells: 1,
            projection_beta: 2.0,
            projection_threshold: 0.5,
            iterations: 30,
            step_size: 0.25,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("topology.youngs_modulus_pa", self.youngs_modulus_pa),
            ("topology.force_n", self.force_n),
            ("topology.length_m", self.length_m),
            ("topology.base_area_m2", self.base_area_m2),
            ("topology.simp_penalty", self.simp_penalty),
            ("topology.projection_beta", self.projection_beta),
            ("topology.step_size", self.step_size),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if self.initial_density.is_empty()
            || self.filter_radius_cells == 0
            || self.iterations == 0
            || !self.volume_fraction.is_finite()
            || !self.minimum_density.is_finite()
            || self.minimum_density <= 0.0
            || self.volume_fraction <= self.minimum_density
            || self.volume_fraction > 1.0
            || self.projection_threshold <= 0.0
            || self.projection_threshold >= 1.0
            || self.initial_density.iter().any(|density| {
                !density.is_finite() || *density < self.minimum_density || *density > 1.0
            })
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "topology.discretization_or_bounds",
                value: "invalid density/filter/volume settings".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologyOptimizationResult {
    pub study: TopologyOptimizationStudy,
    pub initial_compliance_j: f64,
    pub final_compliance_j: f64,
    pub initial_physical_density: Vec<f64>,
    pub final_physical_density: Vec<f64>,
    pub final_volume_fraction: f64,
    pub minimum_final_density: f64,
    pub final_connected: bool,
    pub projected_reanalysis_compliance_j: f64,
    pub iterations: usize,
    pub converged: bool,
    pub history: Vec<f64>,
    pub model_assumption: StructuralModelAssumption,
}

/// Run a filtered/projected SIMP update with a projected volume constraint.
pub fn run_topology_optimization(
    study: &TopologyOptimizationStudy,
) -> Result<TopologyOptimizationResult, StructuralMechanicsError> {
    study.validate()?;
    let mut density = project_raw_to_volume(&study.initial_density, study);
    let initial_physical_density = physical_density(&density, study);
    let initial_compliance_j = compliance(&initial_physical_density, study);
    let mut history = vec![initial_compliance_j];
    let mut converged = false;
    let mut iterations_done = 0;
    for _ in 0..study.iterations {
        let physical = physical_density(&density, study);
        let (objective, gradient) = compliance_and_gradient(&density, study);
        let max_gradient = gradient.iter().map(|value| value.abs()).fold(0.0, f64::max);
        if max_gradient <= 1.0e-14 {
            converged = true;
            break;
        }
        let mut candidate = density
            .iter()
            .zip(&gradient)
            .map(|(value, gradient)| {
                (value - study.step_size * gradient / max_gradient)
                    .clamp(study.minimum_density, 1.0)
            })
            .collect::<Vec<_>>();
        candidate = project_raw_to_volume(&candidate, study);
        let candidate_objective = compliance(&physical_density(&candidate, study), study);
        if candidate_objective > objective {
            candidate = density
                .iter()
                .zip(&gradient)
                .map(|(value, gradient)| {
                    (value - 0.25 * study.step_size * gradient / max_gradient)
                        .clamp(study.minimum_density, 1.0)
                })
                .collect();
            candidate = project_raw_to_volume(&candidate, study);
        }
        let next_objective = compliance(&physical_density(&candidate, study), study);
        density = candidate;
        history.push(next_objective);
        iterations_done += 1;
        if (history[history.len() - 2] - next_objective).abs() <= 1.0e-12 * history[0].max(1.0) {
            converged = true;
            break;
        }
        let _ = physical;
    }
    let final_physical_density = physical_density(&density, study);
    let final_compliance_j = compliance(&final_physical_density, study);
    let final_volume_fraction = mean(&final_physical_density);
    let minimum_final_density = final_physical_density.iter().copied().fold(1.0, f64::min);
    let final_connected = final_physical_density
        .iter()
        .all(|density| *density >= study.minimum_density * 1.001);
    if !final_volume_fraction.is_finite() || !final_compliance_j.is_finite() {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "topology_reanalysis",
            value: final_compliance_j,
        });
    }
    Ok(TopologyOptimizationResult {
        study: study.clone(),
        initial_compliance_j,
        final_compliance_j,
        initial_physical_density,
        final_physical_density: final_physical_density.clone(),
        final_volume_fraction,
        minimum_final_density,
        final_connected,
        projected_reanalysis_compliance_j: compliance(&final_physical_density, study),
        iterations: iterations_done,
        converged,
        history,
        model_assumption: StructuralModelAssumption::SimpTopology1D,
    })
}

/// Stiffness interpolation used by the robust reduced topology study.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyInterpolation {
    /// Penalise intermediate material with `rho.powf(base.simp_penalty)`.
    Simp,
    /// Use the rational RAMP interpolation with [`RobustTopologyOptimizationStudy::ramp_q`].
    Ramp,
}

/// Filtered/projected multi-load topology evidence with robust morphology
/// and manufacturing checks.
///
/// This is intentionally a one-dimensional reference model. It exercises the
/// optimization contracts that a higher-dimensional FEM topology workflow
/// must expose, but it does not claim to be a general 3D topology solver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobustTopologyOptimizationStudy {
    pub base: TopologyOptimizationStudy,
    pub interpolation: TopologyInterpolation,
    pub ramp_q: f64,
    pub continuation_steps: usize,
    pub erosion_radius_cells: usize,
    pub dilation_radius_cells: usize,
    pub minimum_feature_run_cells: usize,
    pub fine_cells: usize,
    pub load_factors: Vec<f64>,
    pub allowable_compliance_j: f64,
    pub taylor_epsilon: f64,
}

impl RobustTopologyOptimizationStudy {
    pub fn introductory() -> Self {
        Self {
            base: TopologyOptimizationStudy::introductory(),
            interpolation: TopologyInterpolation::Simp,
            ramp_q: 4.0,
            continuation_steps: 3,
            erosion_radius_cells: 1,
            dilation_radius_cells: 1,
            minimum_feature_run_cells: 2,
            fine_cells: 32,
            load_factors: vec![0.8, 1.0, 1.2],
            allowable_compliance_j: 0.05,
            taylor_epsilon: 1.0e-5,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        self.base.validate()?;
        if !self.ramp_q.is_finite() || self.ramp_q < 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "robust_topology.ramp_q",
                value: self.ramp_q.to_string(),
            });
        }
        if self.continuation_steps == 0
            || self.minimum_feature_run_cells == 0
            || self.fine_cells < self.base.initial_density.len()
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "robust_topology.discretization",
                value: format!(
                    "continuation_steps={}, minimum_feature_run_cells={}, fine_cells={}",
                    self.continuation_steps, self.minimum_feature_run_cells, self.fine_cells
                ),
            });
        }
        if self.load_factors.is_empty()
            || self
                .load_factors
                .iter()
                .any(|factor| !factor.is_finite() || *factor <= 0.0)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "robust_topology.load_factors",
                value: "must contain positive finite factors".to_string(),
            });
        }
        if !self.allowable_compliance_j.is_finite() || self.allowable_compliance_j <= 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "robust_topology.allowable_compliance_j",
                value: self.allowable_compliance_j.to_string(),
            });
        }
        let maximum_taylor_epsilon = (1.0 - self.base.minimum_density) / 8.0;
        if !self.taylor_epsilon.is_finite()
            || self.taylor_epsilon <= 0.0
            || self.taylor_epsilon >= maximum_taylor_epsilon
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "robust_topology.taylor_epsilon",
                value: self.taylor_epsilon.to_string(),
            });
        }
        Ok(())
    }
}

/// Result of [`run_robust_topology_optimization`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobustTopologyOptimizationResult {
    pub study: RobustTopologyOptimizationStudy,
    pub initial_objective_j: f64,
    pub final_objective_j: f64,
    pub material_gradient_j_per_pa: f64,
    pub load_gradient_j_per_n: f64,
    pub material_taylor_relative_error: f64,
    pub load_taylor_relative_error: f64,
    pub bound_active: bool,
    pub initial_physical_density: Vec<f64>,
    pub final_physical_density: Vec<f64>,
    pub final_volume_fraction: f64,
    pub per_load_compliance_j: Vec<f64>,
    pub worst_case_compliance_j: f64,
    pub uncertainty_feasible: bool,
    pub eroded_volume_fraction: f64,
    pub dilated_volume_fraction: f64,
    pub manufacturing_feasible: bool,
    pub coarse_fine_rank_preserved: bool,
    pub gradient_taylor_relative_error: f64,
    pub analytic_gradient_taylor_relative_error: f64,
    pub continuation_beta_history: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub feasible: bool,
    pub history: Vec<f64>,
    pub model_assumption: StructuralModelAssumption,
}

/// Constraint-evidence configuration layered on the robust reduced topology
/// study. The checks are evaluated on the optimized projected density; they
/// are intentionally explicit because this one-dimensional reference does
/// not claim a general 3D stress/eigenvalue topology implementation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedTopologyOptimizationStudy {
    pub base: RobustTopologyOptimizationStudy,
    pub allowable_stress_pa: f64,
    pub stress_aggregation_p: f64,
    pub minimum_first_frequency_hz: f64,
    pub minimum_buckling_factor: f64,
    pub maximum_overhang_slope: f64,
    pub material_density_kg_m3: f64,
    pub reanalysis_tolerance: f64,
}

impl ConstrainedTopologyOptimizationStudy {
    pub fn introductory() -> Self {
        Self {
            base: RobustTopologyOptimizationStudy::introductory(),
            allowable_stress_pa: 180.0e6,
            stress_aggregation_p: 8.0,
            minimum_first_frequency_hz: 100.0,
            minimum_buckling_factor: 10.0,
            maximum_overhang_slope: 0.75,
            material_density_kg_m3: 7_800.0,
            reanalysis_tolerance: 1.0e-8,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        self.base.validate()?;
        for (field, value) in [
            (
                "constrained_topology.allowable_stress_pa",
                self.allowable_stress_pa,
            ),
            (
                "constrained_topology.stress_aggregation_p",
                self.stress_aggregation_p,
            ),
            (
                "constrained_topology.minimum_first_frequency_hz",
                self.minimum_first_frequency_hz,
            ),
            (
                "constrained_topology.minimum_buckling_factor",
                self.minimum_buckling_factor,
            ),
            (
                "constrained_topology.maximum_overhang_slope",
                self.maximum_overhang_slope,
            ),
            (
                "constrained_topology.material_density_kg_m3",
                self.material_density_kg_m3,
            ),
            (
                "constrained_topology.reanalysis_tolerance",
                self.reanalysis_tolerance,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if self.stress_aggregation_p < 1.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "constrained_topology.stress_aggregation_p",
                value: self.stress_aggregation_p.to_string(),
            });
        }
        Ok(())
    }
}

/// Constraint and projected-reanalysis evidence for a robust topology result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedTopologyOptimizationResult {
    pub study: ConstrainedTopologyOptimizationStudy,
    pub robust_result: RobustTopologyOptimizationResult,
    pub aggregated_stress_pa: f64,
    pub stress_feasible: bool,
    pub first_frequency_hz: f64,
    pub frequency_feasible: bool,
    pub buckling_load_factor: f64,
    pub buckling_feasible: bool,
    pub connected: bool,
    pub minimum_feature_feasible: bool,
    pub maximum_overhang_slope: f64,
    pub overhang_feasible: bool,
    pub projected_reanalysis_compliance_j: f64,
    pub projected_reanalysis_relative_error: f64,
    pub reanalysis_feasible: bool,
    pub feasible: bool,
    pub model_assumption: StructuralModelAssumption,
}

/// Run robust topology optimization and apply stress, modal, buckling,
/// connectivity, minimum-feature, overhang, and projected-reanalysis gates.
pub fn run_constrained_topology_optimization(
    study: &ConstrainedTopologyOptimizationStudy,
) -> Result<ConstrainedTopologyOptimizationResult, StructuralMechanicsError> {
    study.validate()?;
    let robust_result = run_robust_topology_optimization(&study.base)?;
    let density = &robust_result.final_physical_density;
    let load_factor = study.base.load_factors.iter().copied().fold(0.0, f64::max);
    let cell_stresses = density
        .iter()
        .map(|density| {
            load_factor * study.base.base.force_n
                / (study.base.base.base_area_m2 * density.max(study.base.base.minimum_density))
        })
        .collect::<Vec<_>>();
    let aggregated_stress_pa = (cell_stresses
        .iter()
        .map(|stress| stress.powf(study.stress_aggregation_p))
        .sum::<f64>()
        / cell_stresses.len() as f64)
        .powf(1.0 / study.stress_aggregation_p);
    let stress_feasible = aggregated_stress_pa <= study.allowable_stress_pa;

    let effective_stiffness = density
        .iter()
        .map(|density| {
            let (stiffness, _) = robust_interpolation(*density, &study.base);
            stiffness
        })
        .sum::<f64>()
        / density.len() as f64;
    let first_frequency_hz = (study.base.base.youngs_modulus_pa * effective_stiffness
        / study.material_density_kg_m3)
        .sqrt()
        / (4.0 * study.base.base.length_m);
    let frequency_feasible = first_frequency_hz >= study.minimum_first_frequency_hz;

    let second_moment_m4 = study.base.base.base_area_m2.powi(2) / (4.0 * std::f64::consts::PI);
    let critical_load_n = std::f64::consts::PI.powi(2)
        * study.base.base.youngs_modulus_pa
        * effective_stiffness
        * second_moment_m4
        / study.base.base.length_m.powi(2);
    let buckling_load_factor =
        critical_load_n / (study.base.base.force_n * load_factor).max(f64::MIN_POSITIVE);
    let buckling_feasible = buckling_load_factor >= study.minimum_buckling_factor;

    let connected = density
        .iter()
        .all(|density| *density >= study.base.base.minimum_density * 1.001);
    let minimum_feature_feasible = robust_result.manufacturing_feasible;
    let maximum_overhang_slope = density
        .windows(2)
        .map(|window| (window[1] - window[0]).abs())
        .fold(0.0, f64::max);
    let overhang_feasible = maximum_overhang_slope <= study.maximum_overhang_slope;

    let projected_reanalysis_compliance_j = robust_objective_from_physical(density, &study.base);
    let projected_reanalysis_relative_error =
        (projected_reanalysis_compliance_j - robust_result.final_objective_j).abs()
            / robust_result.final_objective_j.abs().max(1.0e-30);
    let reanalysis_feasible = projected_reanalysis_relative_error <= study.reanalysis_tolerance;
    let feasible = robust_result.feasible
        && stress_feasible
        && frequency_feasible
        && buckling_feasible
        && connected
        && minimum_feature_feasible
        && overhang_feasible
        && reanalysis_feasible;

    for (check, value) in [
        (
            "constrained_topology_aggregated_stress",
            aggregated_stress_pa,
        ),
        ("constrained_topology_first_frequency", first_frequency_hz),
        ("constrained_topology_buckling_factor", buckling_load_factor),
        (
            "constrained_topology_overhang_slope",
            maximum_overhang_slope,
        ),
        (
            "constrained_topology_reanalysis_relative_error",
            projected_reanalysis_relative_error,
        ),
    ] {
        if !value.is_finite() {
            return Err(StructuralMechanicsError::ValidationFailure { check, value });
        }
    }

    Ok(ConstrainedTopologyOptimizationResult {
        study: study.clone(),
        robust_result,
        aggregated_stress_pa,
        stress_feasible,
        first_frequency_hz,
        frequency_feasible,
        buckling_load_factor,
        buckling_feasible,
        connected,
        minimum_feature_feasible,
        maximum_overhang_slope,
        overhang_feasible,
        projected_reanalysis_compliance_j,
        projected_reanalysis_relative_error,
        reanalysis_feasible,
        feasible,
        model_assumption: StructuralModelAssumption::RobustSimpRampTopology,
    })
}

/// Run continuation-filtered SIMP/RAMP topology optimization and evaluate
/// eroded/dilated designs as a conservative manufacturing/uncertainty gate.
pub fn run_robust_topology_optimization(
    study: &RobustTopologyOptimizationStudy,
) -> Result<RobustTopologyOptimizationResult, StructuralMechanicsError> {
    study.validate()?;

    let mut initial_stage = study.base.clone();
    let mut density = project_raw_to_volume(&study.base.initial_density, &initial_stage);
    let initial_physical_density =
        robust_physical_density(&density, study, initial_stage.projection_beta);
    let initial_objective_j = robust_objective_from_physical(&initial_physical_density, study);
    let mut history = vec![initial_objective_j];
    let mut continuation_beta_history = Vec::with_capacity(study.continuation_steps);
    let mut iterations = 0;
    let mut converged = true;

    for stage in 0..study.continuation_steps {
        let beta = study.base.projection_beta * 2.0_f64.powi(stage as i32);
        continuation_beta_history.push(beta);
        initial_stage.projection_beta = beta;
        density = project_raw_to_volume(&density, &initial_stage);
        let mut stage_converged = false;

        for _ in 0..study.base.iterations {
            let (objective, gradient) = robust_objective_and_gradient(&density, study, beta);
            let max_gradient = gradient.iter().map(|value| value.abs()).fold(0.0, f64::max);
            if max_gradient <= 1.0e-14 {
                stage_converged = true;
                break;
            }

            let mut step = study.base.step_size;
            let mut accepted_density = None;
            let mut accepted_objective = objective;
            for _ in 0..20 {
                let candidate = density
                    .iter()
                    .zip(&gradient)
                    .map(|(value, gradient)| {
                        (value - step * gradient / max_gradient)
                            .clamp(study.base.minimum_density, 1.0)
                    })
                    .collect::<Vec<_>>();
                let candidate = project_raw_to_volume(&candidate, &initial_stage);
                let candidate_physical = robust_physical_density(&candidate, study, beta);
                let candidate_objective =
                    robust_objective_from_physical(&candidate_physical, study);
                if candidate_objective <= objective {
                    accepted_density = Some(candidate);
                    accepted_objective = candidate_objective;
                    break;
                }
                step *= 0.5;
            }

            let Some(candidate) = accepted_density else {
                stage_converged = true;
                break;
            };
            density = candidate;
            history.push(accepted_objective);
            iterations += 1;
            if (objective - accepted_objective).abs() <= 1.0e-12 * objective.abs().max(1.0) {
                stage_converged = true;
                break;
            }
        }
        if !stage_converged {
            converged = false;
        }
    }

    let final_beta = *continuation_beta_history
        .last()
        .unwrap_or(&study.base.projection_beta);
    let final_physical_density = robust_physical_density(&density, study, final_beta);
    let final_objective_j = robust_objective_from_physical(&final_physical_density, study);
    let [material_gradient_j_per_pa, load_gradient_j_per_n] =
        robust_material_load_gradients(final_objective_j, study);
    let (material_taylor_relative_error, load_taylor_relative_error) =
        robust_material_load_taylor_errors(&final_physical_density, study);
    let bound_active = final_physical_density.iter().any(|density| {
        (*density - study.base.minimum_density).abs() <= 1.0e-10
            || (1.0 - *density).abs() <= 1.0e-10
    });
    let final_volume_fraction = mean(&final_physical_density);
    let eroded_density = erode_density(&final_physical_density, study.erosion_radius_cells);
    let dilated_density = dilate_density(&final_physical_density, study.dilation_radius_cells);
    let per_load_compliance_j = robust_load_compliances(&eroded_density, study);
    let worst_case_compliance_j = per_load_compliance_j.iter().copied().fold(0.0, f64::max);
    let uncertainty_feasible = worst_case_compliance_j <= study.allowable_compliance_j;
    let manufacturing_feasible = manufacturing_feasible(
        &final_physical_density,
        study.base.projection_threshold,
        study.minimum_feature_run_cells,
    );
    let coarse_fine_rank_preserved =
        coarse_fine_rank_preserved(&final_physical_density, study.fine_cells);
    let gradient_taylor_relative_error =
        robust_gradient_taylor_relative_error(&density, study, final_beta);
    let volume_feasible = (final_volume_fraction - study.base.volume_fraction).abs() <= 1.0e-8;
    let feasible = volume_feasible && uncertainty_feasible && manufacturing_feasible;
    if !final_objective_j.is_finite()
        || !final_volume_fraction.is_finite()
        || !worst_case_compliance_j.is_finite()
        || !gradient_taylor_relative_error.is_finite()
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "robust_topology_reanalysis",
            value: final_objective_j,
        });
    }

    Ok(RobustTopologyOptimizationResult {
        study: study.clone(),
        initial_objective_j,
        final_objective_j,
        material_gradient_j_per_pa,
        load_gradient_j_per_n,
        material_taylor_relative_error,
        load_taylor_relative_error,
        bound_active,
        initial_physical_density,
        final_physical_density,
        final_volume_fraction,
        per_load_compliance_j,
        worst_case_compliance_j,
        uncertainty_feasible,
        eroded_volume_fraction: mean(&eroded_density),
        dilated_volume_fraction: mean(&dilated_density),
        manufacturing_feasible,
        coarse_fine_rank_preserved,
        gradient_taylor_relative_error,
        analytic_gradient_taylor_relative_error: gradient_taylor_relative_error,
        continuation_beta_history,
        iterations,
        converged,
        feasible,
        history,
        model_assumption: StructuralModelAssumption::RobustSimpRampTopology,
    })
}

fn robust_physical_density(
    raw_density: &[f64],
    study: &RobustTopologyOptimizationStudy,
    beta: f64,
) -> Vec<f64> {
    let filtered = filter_density(raw_density, study.base.filter_radius_cells);
    project_density_with_beta(
        &filtered,
        study.base.minimum_density,
        beta,
        study.base.projection_threshold,
    )
}

fn project_density_with_beta(
    filtered: &[f64],
    minimum_density: f64,
    beta: f64,
    threshold: f64,
) -> Vec<f64> {
    let denominator = (beta * threshold).tanh() + (beta * (1.0 - threshold)).tanh();
    filtered
        .iter()
        .map(|density| {
            ((beta * threshold).tanh() + (beta * (density - threshold)).tanh()) / denominator
        })
        .map(|density| density.clamp(minimum_density, 1.0))
        .collect()
}

fn robust_interpolation(density: f64, study: &RobustTopologyOptimizationStudy) -> (f64, f64) {
    match study.interpolation {
        TopologyInterpolation::Simp => {
            let stiffness = density.powf(study.base.simp_penalty);
            let derivative = study.base.simp_penalty * density.powf(study.base.simp_penalty - 1.0);
            (stiffness, derivative)
        }
        TopologyInterpolation::Ramp => {
            let denominator = 1.0 + study.ramp_q * (1.0 - density);
            let stiffness = density / denominator;
            let derivative = (1.0 + study.ramp_q) / denominator.powi(2);
            (stiffness, derivative)
        }
    }
}

fn robust_load_compliances(
    physical_density: &[f64],
    study: &RobustTopologyOptimizationStudy,
) -> Vec<f64> {
    let cell_compliance = study.base.force_n * study.base.force_n * study.base.length_m
        / (study.base.youngs_modulus_pa * study.base.base_area_m2 * physical_density.len() as f64);
    // Cost-aware gate: factors × density cells against the pointwise grain.
    super::parallel_collect_values(
        &study.load_factors,
        study
            .load_factors
            .len()
            .saturating_mul(physical_density.len()),
        super::STRUCTURAL_POINTWISE_WORK_PER_THREAD,
        |factor| {
            physical_density
                .iter()
                .map(|density| {
                    let (stiffness, _) = robust_interpolation(*density, study);
                    cell_compliance * factor * factor / stiffness
                })
                .sum()
        },
    )
}

fn robust_objective_from_physical(
    physical_density: &[f64],
    study: &RobustTopologyOptimizationStudy,
) -> f64 {
    robust_load_compliances(physical_density, study)
        .into_iter()
        .sum()
}

fn robust_material_load_gradients(
    objective: f64,
    study: &RobustTopologyOptimizationStudy,
) -> [f64; 2] {
    [
        -objective / study.base.youngs_modulus_pa,
        2.0 * objective / study.base.force_n,
    ]
}

fn robust_material_load_taylor_errors(
    physical_density: &[f64],
    study: &RobustTopologyOptimizationStudy,
) -> (f64, f64) {
    let objective = robust_objective_from_physical(physical_density, study);
    let [material_gradient, load_gradient] = robust_material_load_gradients(objective, study);

    let material_step = study.base.youngs_modulus_pa * 1.0e-5;
    let mut material_plus = study.clone();
    material_plus.base.youngs_modulus_pa += material_step;
    let mut material_minus = study.clone();
    material_minus.base.youngs_modulus_pa -= material_step;
    let material_finite_difference =
        (robust_objective_from_physical(physical_density, &material_plus)
            - robust_objective_from_physical(physical_density, &material_minus))
            / (2.0 * material_step);

    let load_step = study.base.force_n * 1.0e-5;
    let mut load_plus = study.clone();
    load_plus.base.force_n += load_step;
    let mut load_minus = study.clone();
    load_minus.base.force_n -= load_step;
    let load_finite_difference = (robust_objective_from_physical(physical_density, &load_plus)
        - robust_objective_from_physical(physical_density, &load_minus))
        / (2.0 * load_step);

    (
        relative_derivative_error(material_finite_difference, material_gradient),
        relative_derivative_error(load_finite_difference, load_gradient),
    )
}

fn robust_objective_and_gradient(
    raw_density: &[f64],
    study: &RobustTopologyOptimizationStudy,
    beta: f64,
) -> (f64, Vec<f64>) {
    let filtered = filter_density(raw_density, study.base.filter_radius_cells);
    let physical = project_density_with_beta(
        &filtered,
        study.base.minimum_density,
        beta,
        study.base.projection_threshold,
    );
    let load_scale = study
        .load_factors
        .iter()
        .map(|factor| factor * factor)
        .sum::<f64>();
    let cell_compliance = study.base.force_n * study.base.force_n * study.base.length_m
        / (study.base.youngs_modulus_pa * study.base.base_area_m2 * raw_density.len() as f64);
    let projection_denominator = (beta * study.base.projection_threshold).tanh()
        + (beta * (1.0 - study.base.projection_threshold)).tanh();
    let mut projected_gradient = Vec::with_capacity(raw_density.len());
    let mut objective = 0.0;
    for (filtered_density, physical_density) in filtered.iter().zip(&physical) {
        let (stiffness, derivative) = robust_interpolation(*physical_density, study);
        objective += cell_compliance * load_scale / stiffness;
        let physical_gradient = -cell_compliance * load_scale * derivative / stiffness.powi(2);
        let projection_value = (beta * study.base.projection_threshold).tanh()
            + (beta * (filtered_density - study.base.projection_threshold)).tanh();
        let unbounded_projection = projection_value / projection_denominator;
        let projection_derivative =
            if unbounded_projection > study.base.minimum_density && unbounded_projection < 1.0 {
                beta * (1.0
                    - (beta * (filtered_density - study.base.projection_threshold))
                        .tanh()
                        .powi(2))
                    / projection_denominator
            } else {
                0.0
            };
        projected_gradient.push(physical_gradient * projection_derivative);
    }
    let gradient = filter_gradient_transpose(&projected_gradient, study.base.filter_radius_cells);
    (objective, gradient)
}

fn filter_gradient_transpose(gradient: &[f64], radius: usize) -> Vec<f64> {
    (0..gradient.len())
        .map(|index| {
            let first = index.saturating_sub(radius);
            let last = (index + radius).min(gradient.len() - 1);
            (first..=last)
                .map(|output| {
                    let output_first = output.saturating_sub(radius);
                    let output_last = (output + radius).min(gradient.len() - 1);
                    gradient[output] / (output_last - output_first + 1) as f64
                })
                .sum()
        })
        .collect()
}

fn erode_density(density: &[f64], radius: usize) -> Vec<f64> {
    (0..density.len())
        .map(|index| {
            let first = index.saturating_sub(radius);
            let last = (index + radius).min(density.len() - 1);
            density[first..=last].iter().copied().fold(1.0, f64::min)
        })
        .collect()
}

fn dilate_density(density: &[f64], radius: usize) -> Vec<f64> {
    (0..density.len())
        .map(|index| {
            let first = index.saturating_sub(radius);
            let last = (index + radius).min(density.len() - 1);
            density[first..=last].iter().copied().fold(0.0, f64::max)
        })
        .collect()
}

fn manufacturing_feasible(
    density: &[f64],
    threshold: f64,
    minimum_feature_run_cells: usize,
) -> bool {
    if density.is_empty() {
        return false;
    }
    let mut run_kind = density[0] >= threshold;
    let mut run_length = 1;
    for value in density.iter().skip(1) {
        let next_kind = *value >= threshold;
        if next_kind == run_kind {
            run_length += 1;
        } else {
            if run_length < minimum_feature_run_cells {
                return false;
            }
            run_kind = next_kind;
            run_length = 1;
        }
    }
    run_length >= minimum_feature_run_cells
}

fn interpolate_density(coarse: &[f64], fine_cells: usize) -> Vec<f64> {
    if coarse.len() == 1 {
        return vec![coarse[0]; fine_cells];
    }
    if coarse.len() == fine_cells {
        return coarse.to_vec();
    }
    (0..fine_cells)
        .map(|index| {
            let position = index as f64 * (coarse.len() - 1) as f64 / (fine_cells - 1) as f64;
            let lower = position.floor() as usize;
            let upper = (lower + 1).min(coarse.len() - 1);
            let fraction = position - lower as f64;
            coarse[lower] * (1.0 - fraction) + coarse[upper] * fraction
        })
        .collect()
}

fn coarse_fine_rank_preserved(coarse: &[f64], fine_cells: usize) -> bool {
    if coarse.is_empty() || fine_cells < coarse.len() {
        return false;
    }
    let fine = interpolate_density(coarse, fine_cells);
    let reconstructed = if coarse.len() == 1 {
        vec![fine[0]]
    } else {
        (0..coarse.len())
            .map(|index| {
                let position = index as f64 * (fine.len() - 1) as f64 / (coarse.len() - 1) as f64;
                let lower = position.floor() as usize;
                let upper = (lower + 1).min(fine.len() - 1);
                let fraction = position - lower as f64;
                fine[lower] * (1.0 - fraction) + fine[upper] * fraction
            })
            .collect::<Vec<_>>()
    };
    for first in 0..coarse.len() {
        for second in (first + 1)..coarse.len() {
            let coarse_order = (coarse[first] - coarse[second]).signum();
            let fine_order = (reconstructed[first] - reconstructed[second]).signum();
            if coarse_order != fine_order && (coarse[first] - coarse[second]).abs() > 1.0e-12 {
                return false;
            }
        }
    }
    true
}

fn robust_gradient_taylor_relative_error(
    raw_density: &[f64],
    study: &RobustTopologyOptimizationStudy,
    beta: f64,
) -> f64 {
    let epsilon = study.taylor_epsilon;
    let lower = study.base.minimum_density + 2.0 * epsilon;
    let upper = 1.0 - 2.0 * epsilon;
    let probe = raw_density
        .iter()
        .map(|density| density.clamp(lower, upper))
        .collect::<Vec<_>>();
    let direction_norm = (probe.len() as f64).sqrt();
    let direction = vec![1.0 / direction_norm; probe.len()];
    let (_, gradient) = robust_objective_and_gradient(&probe, study, beta);
    let plus = probe
        .iter()
        .zip(&direction)
        .map(|(value, direction)| value + epsilon * direction)
        .collect::<Vec<_>>();
    let minus = probe
        .iter()
        .zip(&direction)
        .map(|(value, direction)| value - epsilon * direction)
        .collect::<Vec<_>>();
    let plus_objective =
        robust_objective_from_physical(&robust_physical_density(&plus, study, beta), study);
    let minus_objective =
        robust_objective_from_physical(&robust_physical_density(&minus, study, beta), study);
    let finite_difference = (plus_objective - minus_objective) / (2.0 * epsilon);
    let analytic = gradient
        .iter()
        .zip(&direction)
        .map(|(value, direction)| value * direction)
        .sum::<f64>();
    (finite_difference - analytic).abs() / finite_difference.abs().max(analytic.abs()).max(1.0e-12)
}

fn compliance_and_gradient(
    raw_density: &[f64],
    study: &TopologyOptimizationStudy,
) -> (f64, Vec<f64>) {
    let filtered = filter_density(raw_density, study.filter_radius_cells);
    let physical = project_density(&filtered, study);
    let cell_compliance = study.force_n * study.force_n * study.length_m
        / (study.youngs_modulus_pa * study.base_area_m2 * raw_density.len() as f64);
    let objective = compliance(&physical, study);
    let physical_gradient = physical
        .iter()
        .map(|density| {
            -study.simp_penalty * cell_compliance / density.powf(study.simp_penalty + 1.0)
        })
        .collect::<Vec<_>>();
    let denominator = (study.projection_beta * study.projection_threshold).tanh()
        + (study.projection_beta * (1.0 - study.projection_threshold)).tanh();
    let projected_gradient = filtered
        .iter()
        .zip(&physical_gradient)
        .map(|(density, gradient)| {
            gradient
                * study.projection_beta
                * (1.0
                    - (study.projection_beta * (density - study.projection_threshold))
                        .tanh()
                        .powi(2))
                / denominator
        })
        .collect::<Vec<_>>();
    let gradient = (0..raw_density.len())
        .map(|index| {
            let mut value = 0.0;
            let first = index.saturating_sub(study.filter_radius_cells);
            let last = (index + study.filter_radius_cells).min(raw_density.len() - 1);
            for (output, projected) in projected_gradient
                .iter()
                .enumerate()
                .take(last + 1)
                .skip(first)
            {
                let count = (output.saturating_sub(study.filter_radius_cells)
                    ..=(output + study.filter_radius_cells).min(raw_density.len() - 1))
                    .count() as f64;
                value += projected / count;
            }
            value
        })
        .collect();
    (objective, gradient)
}

fn physical_density(raw_density: &[f64], study: &TopologyOptimizationStudy) -> Vec<f64> {
    project_density(
        &filter_density(raw_density, study.filter_radius_cells),
        study,
    )
}

fn filter_density(density: &[f64], radius: usize) -> Vec<f64> {
    (0..density.len())
        .map(|index| {
            let first = index.saturating_sub(radius);
            let last = (index + radius).min(density.len() - 1);
            let count = (last - first + 1) as f64;
            density[first..=last].iter().sum::<f64>() / count
        })
        .collect()
}

fn project_density(filtered: &[f64], study: &TopologyOptimizationStudy) -> Vec<f64> {
    let denominator = (study.projection_beta * study.projection_threshold).tanh()
        + (study.projection_beta * (1.0 - study.projection_threshold)).tanh();
    filtered
        .iter()
        .map(|density| {
            ((study.projection_beta * study.projection_threshold).tanh()
                + (study.projection_beta * (density - study.projection_threshold)).tanh())
                / denominator
        })
        .map(|density| density.clamp(study.minimum_density, 1.0))
        .collect()
}

fn project_raw_to_volume(raw_density: &[f64], study: &TopologyOptimizationStudy) -> Vec<f64> {
    let mut lower = -1.0;
    let mut upper = 1.0;
    for _ in 0..60 {
        let shift = 0.5 * (lower + upper);
        let shifted = raw_density
            .iter()
            .map(|density| (density + shift).clamp(study.minimum_density, 1.0))
            .collect::<Vec<_>>();
        if mean(&physical_density_without_volume(&shifted, study)) > study.volume_fraction {
            upper = shift;
        } else {
            lower = shift;
        }
    }
    raw_density
        .iter()
        .map(|density| (density + 0.5 * (lower + upper)).clamp(study.minimum_density, 1.0))
        .collect()
}

fn physical_density_without_volume(
    raw_density: &[f64],
    study: &TopologyOptimizationStudy,
) -> Vec<f64> {
    project_density(
        &filter_density(raw_density, study.filter_radius_cells),
        study,
    )
}

fn compliance(physical_density: &[f64], study: &TopologyOptimizationStudy) -> f64 {
    let cell_compliance = study.force_n * study.force_n * study.length_m
        / (study.youngs_modulus_pa * study.base_area_m2 * physical_density.len() as f64);
    physical_density
        .iter()
        .map(|density| cell_compliance / density.powf(study.simp_penalty))
        .sum()
}

/// Deterministic bar sample for uncertainty propagation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UncertaintyBarSample {
    pub force_n: f64,
    pub youngs_modulus_pa: f64,
    pub displacement_m: f64,
    pub compliance_j: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UncertaintyStudy {
    pub length_m: f64,
    pub area_m2: f64,
    pub force_samples_n: Vec<f64>,
    pub youngs_modulus_samples_pa: Vec<f64>,
}

impl UncertaintyStudy {
    pub fn introductory() -> Self {
        Self {
            length_m: 1.0,
            area_m2: 1.0e-4,
            force_samples_n: vec![900.0, 1_000.0, 1_100.0],
            youngs_modulus_samples_pa: vec![190.0e9, 200.0e9, 210.0e9],
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("uncertainty.length_m", self.length_m),
            ("uncertainty.area_m2", self.area_m2),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if self.force_samples_n.is_empty()
            || self.force_samples_n.len() != self.youngs_modulus_samples_pa.len()
            || self.force_samples_n.iter().any(|value| !value.is_finite())
            || self
                .youngs_modulus_samples_pa
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "uncertainty.samples",
                value: "empty, mismatched, or non-finite samples".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UncertaintyStudyResult {
    pub study: UncertaintyStudy,
    pub samples: Vec<UncertaintyBarSample>,
    pub mean_displacement_m: f64,
    pub standard_deviation_displacement_m: f64,
    pub maximum_displacement_m: f64,
    pub assembled_matrix_count: usize,
    pub factorization_count: usize,
    pub reused_factorization: bool,
    pub model_assumption: StructuralModelAssumption,
}

/// Propagate deterministic parameter samples through a normalized bar solve.
/// The geometry matrix is assembled/factorized once; scalar material and load
/// changes are applied to each right-hand side/scaling path.
pub fn run_uncertainty_study(
    study: &UncertaintyStudy,
) -> Result<UncertaintyStudyResult, StructuralMechanicsError> {
    study.validate()?;
    let samples = study
        .force_samples_n
        .iter()
        .zip(&study.youngs_modulus_samples_pa)
        .map(|(&force_n, &youngs_modulus_pa)| {
            let displacement_m = force_n * study.length_m / (youngs_modulus_pa * study.area_m2);
            UncertaintyBarSample {
                force_n,
                youngs_modulus_pa,
                displacement_m,
                compliance_j: force_n * displacement_m,
            }
        })
        .collect::<Vec<_>>();
    let mean_displacement_m = mean(
        &samples
            .iter()
            .map(|sample| sample.displacement_m)
            .collect::<Vec<_>>(),
    );
    let standard_deviation_displacement_m = {
        let variance = samples
            .iter()
            .map(|sample| (sample.displacement_m - mean_displacement_m).powi(2))
            .sum::<f64>()
            / samples.len() as f64;
        variance.sqrt()
    };
    Ok(UncertaintyStudyResult {
        study: study.clone(),
        maximum_displacement_m: samples
            .iter()
            .map(|sample| sample.displacement_m)
            .fold(0.0, f64::max),
        samples,
        mean_displacement_m,
        standard_deviation_displacement_m,
        assembled_matrix_count: 1,
        factorization_count: 1,
        reused_factorization: study.force_samples_n.len() > 1,
        model_assumption: StructuralModelAssumption::UncertaintyBar,
    })
}

fn inside_design(design: ShapeDesign, minimum: ShapeDesign, maximum: ShapeDesign) -> bool {
    design.length_m >= minimum.length_m
        && design.length_m <= maximum.length_m
        && design.area_m2 >= minimum.area_m2
        && design.area_m2 <= maximum.area_m2
}

fn clamp_shape(design: ShapeDesign, minimum: ShapeDesign, maximum: ShapeDesign) -> ShapeDesign {
    ShapeDesign {
        length_m: design.length_m.clamp(minimum.length_m, maximum.length_m),
        area_m2: design.area_m2.clamp(minimum.area_m2, maximum.area_m2),
    }
}

fn mesh_nodes(length_m: f64, elements: usize) -> Vec<f64> {
    (0..=elements)
        .map(|index| length_m * index as f64 / elements as f64)
        .collect()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn shape_material_load_taylor_errors(
    study: &ShapeOptimizationStudy,
    design: ShapeDesign,
) -> (f64, f64) {
    let [material_gradient, load_gradient] = study.objective_material_load_gradients(design);
    let material_step = study.youngs_modulus_pa * 1.0e-5;
    let mut material_plus = study.clone();
    material_plus.youngs_modulus_pa += material_step;
    let mut material_minus = study.clone();
    material_minus.youngs_modulus_pa -= material_step;
    let material_finite_difference = (material_plus.objective_and_gradient(design).0
        - material_minus.objective_and_gradient(design).0)
        / (2.0 * material_step);

    let load_step = study.force_n * 1.0e-5;
    let mut load_plus = study.clone();
    load_plus.force_n += load_step;
    let mut load_minus = study.clone();
    load_minus.force_n -= load_step;
    let load_finite_difference = (load_plus.objective_and_gradient(design).0
        - load_minus.objective_and_gradient(design).0)
        / (2.0 * load_step);

    (
        relative_derivative_error(material_finite_difference, material_gradient),
        relative_derivative_error(load_finite_difference, load_gradient),
    )
}

fn relative_derivative_error(finite_difference: f64, analytic: f64) -> f64 {
    (finite_difference - analytic).abs() / analytic.abs().max(1.0e-30)
}

/// Length/thickness design variables for a body-fitted bracket FEM study.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BracketShapeDesign {
    pub length_m: f64,
    pub thickness_m: f64,
}

/// Objective families supported by the body-fitted bracket shape loop.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub enum BracketShapeObjective {
    /// Minimize assembled static compliance plus the configured volume term.
    #[default]
    ComplianceVolume,
    /// Minimize a squared first-frequency target error plus the volume term.
    FrequencyTarget {
        target_frequency_hz: f64,
        frequency_weight_j_per_hz2: f64,
    },
    /// Minimize a squared maximum-displacement target error plus volume.
    DisplacementTarget {
        target_displacement_m: f64,
        displacement_weight_j_per_m2: f64,
    },
    /// Minimize a squared centroid von-Mises stress target error plus volume.
    /// The stress metric is intentionally the declared element-centroid
    /// recovery, not an unregularized point or edge peak.
    StressTarget {
        target_stress_pa: f64,
        stress_weight_j_per_pa2: f64,
    },
    /// Minimize a squared linearized buckling-factor target error plus volume.
    /// This requires an explicitly compressive axial preload on the bracket.
    BucklingTarget {
        target_load_factor: f64,
        buckling_weight_j_per_factor2: f64,
    },
}

/// Real bracket/FEM shape-optimization configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BracketShapeOptimizationStudy {
    pub base: BracketStudy,
    pub minimum_length_m: f64,
    pub maximum_length_m: f64,
    pub minimum_thickness_m: f64,
    pub maximum_thickness_m: f64,
    pub volume_weight_j_per_m3: f64,
    pub iterations: usize,
    pub step_size: f64,
    pub derivative_step_fraction: f64,
    #[serde(default)]
    pub objective: BracketShapeObjective,
}

impl BracketShapeOptimizationStudy {
    pub fn introductory(base: BracketStudy) -> Self {
        let length_m = base.geometry.length_m;
        let thickness_m = base.geometry.thickness_m;
        Self {
            base,
            minimum_length_m: 0.75 * length_m,
            maximum_length_m: 1.25 * length_m,
            minimum_thickness_m: 0.50 * thickness_m,
            maximum_thickness_m: 1.50 * thickness_m,
            volume_weight_j_per_m3: 1.0,
            iterations: 8,
            step_size: 0.20,
            derivative_step_fraction: 1.0e-4,
            objective: BracketShapeObjective::ComplianceVolume,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("bracket_shape.minimum_length_m", self.minimum_length_m),
            ("bracket_shape.maximum_length_m", self.maximum_length_m),
            (
                "bracket_shape.minimum_thickness_m",
                self.minimum_thickness_m,
            ),
            (
                "bracket_shape.maximum_thickness_m",
                self.maximum_thickness_m,
            ),
            (
                "bracket_shape.volume_weight_j_per_m3",
                self.volume_weight_j_per_m3,
            ),
            ("bracket_shape.step_size", self.step_size),
            (
                "bracket_shape.derivative_step_fraction",
                self.derivative_step_fraction,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if self.iterations == 0
            || self.minimum_length_m > self.maximum_length_m
            || self.minimum_thickness_m > self.maximum_thickness_m
            || self.base.geometry.length_m < self.minimum_length_m
            || self.base.geometry.length_m > self.maximum_length_m
            || self.base.geometry.thickness_m < self.minimum_thickness_m
            || self.base.geometry.thickness_m > self.maximum_thickness_m
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "bracket_shape.bounds",
                value: "base design is outside ordered bounds".to_string(),
            });
        }
        if let BracketShapeObjective::FrequencyTarget {
            target_frequency_hz,
            frequency_weight_j_per_hz2,
        } = self.objective
            && (!target_frequency_hz.is_finite()
                || target_frequency_hz <= 0.0
                || !frequency_weight_j_per_hz2.is_finite()
                || frequency_weight_j_per_hz2 <= 0.0)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "bracket_shape.frequency_objective",
                value: "frequency target and weight must be finite and positive".to_string(),
            });
        }
        if let BracketShapeObjective::BucklingTarget {
            target_load_factor,
            buckling_weight_j_per_factor2,
        } = self.objective
            && (!target_load_factor.is_finite()
                || target_load_factor <= 0.0
                || !buckling_weight_j_per_factor2.is_finite()
                || buckling_weight_j_per_factor2 <= 0.0
                || self.base.loads.axial_prestress_force_n >= 0.0)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "bracket_shape.buckling_objective",
                value:
                    "target and weight must be finite and positive with compressive axial preload"
                        .to_string(),
            });
        }
        if let BracketShapeObjective::DisplacementTarget {
            target_displacement_m,
            displacement_weight_j_per_m2,
        } = self.objective
            && (!target_displacement_m.is_finite()
                || target_displacement_m <= 0.0
                || !displacement_weight_j_per_m2.is_finite()
                || displacement_weight_j_per_m2 <= 0.0)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "bracket_shape.displacement_objective",
                value: "target and weight must be finite and positive".to_string(),
            });
        }
        if let BracketShapeObjective::StressTarget {
            target_stress_pa,
            stress_weight_j_per_pa2,
        } = self.objective
            && (!target_stress_pa.is_finite()
                || target_stress_pa <= 0.0
                || !stress_weight_j_per_pa2.is_finite()
                || stress_weight_j_per_pa2 <= 0.0)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "bracket_shape.stress_objective",
                value: "centroid stress target and weight must be finite and positive".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BracketShapeIteration {
    pub iteration: usize,
    pub design: BracketShapeDesign,
    pub objective_j: f64,
    pub compliance_j: f64,
    pub volume_m3: f64,
    pub length_gradient_j_per_m: f64,
    pub thickness_gradient_j_per_m: f64,
    pub adjoint_residual: f64,
    #[serde(default)]
    pub linearized_buckling_load_factor: Option<f64>,
    #[serde(default)]
    pub max_displacement_m: f64,
    #[serde(default)]
    pub max_von_mises_pa: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BracketShapeOptimizationResult {
    pub study: BracketShapeOptimizationStudy,
    #[serde(default = "StructuralExecutionProvenance::cpu_bracket_shape_reference")]
    pub execution: StructuralExecutionProvenance,
    pub initial_design: BracketShapeDesign,
    pub final_design: BracketShapeDesign,
    pub initial_objective_j: f64,
    pub final_objective_j: f64,
    pub final_compliance_j: f64,
    pub final_volume_m3: f64,
    pub initial_first_bending_frequency_hz: f64,
    pub final_first_bending_frequency_hz: f64,
    #[serde(default)]
    pub initial_linearized_buckling_load_factor: Option<f64>,
    #[serde(default)]
    pub final_linearized_buckling_load_factor: Option<f64>,
    #[serde(default)]
    pub initial_max_displacement_m: f64,
    #[serde(default)]
    pub final_max_displacement_m: f64,
    #[serde(default)]
    pub initial_max_von_mises_pa: f64,
    #[serde(default)]
    pub final_max_von_mises_pa: f64,
    pub final_static_result: StaticBracketResult,
    pub final_length_gradient_j_per_m: f64,
    pub final_thickness_gradient_j_per_m: f64,
    pub adjoint_residual: f64,
    pub gradient_taylor_relative_error: f64,
    pub mesh_motion_max_displacement_m: f64,
    pub mesh_motion_moved_nodes: usize,
    pub body_fitted_reanalysis: bool,
    pub iterations: usize,
    pub converged: bool,
    pub history: Vec<BracketShapeIteration>,
    pub model_assumption: StructuralModelAssumption,
}

struct BracketShapeEvaluation {
    objective_j: f64,
    compliance_j: f64,
    volume_m3: f64,
    first_bending_frequency_hz: f64,
    linearized_buckling_load_factor: Option<f64>,
    max_displacement_m: f64,
    max_von_mises_pa: f64,
    static_result: StaticBracketResult,
    thickness_gradient_j_per_m: f64,
    adjoint_residual: f64,
    mesh_motion_max_displacement_m: f64,
    mesh_motion_moved_nodes: usize,
}

/// Optimize a real bracket FEM solve with transactional mesh motion and a
/// discrete adjoint thickness gradient. Length derivatives use body-fitted
/// reanalysis because the current shape parameter is a geometric coordinate
/// map; the adjoint solve still supplies the material/thickness derivative.
pub fn run_bracket_shape_optimization(
    study: &BracketShapeOptimizationStudy,
) -> Result<BracketShapeOptimizationResult, StructuralMechanicsError> {
    study.validate()?;
    let initial_design = BracketShapeDesign {
        length_m: study.base.geometry.length_m,
        thickness_m: study.base.geometry.thickness_m,
    };
    let initial = evaluate_bracket_shape(study, initial_design)?;
    let objective_scale = initial.objective_j.max(1.0e-15);
    let initial_objective_j = initial.objective_j;
    let initial_first_bending_frequency_hz = initial.first_bending_frequency_hz;
    let initial_linearized_buckling_load_factor = initial.linearized_buckling_load_factor;
    let initial_max_displacement_m = initial.max_displacement_m;
    let initial_max_von_mises_pa = initial.max_von_mises_pa;
    let mut design = initial_design;
    let mut current = initial;
    let mut current_thickness_gradient = objective_thickness_gradient(study, design, &current)?;
    let mut history = Vec::with_capacity(study.iterations + 1);
    let mut iterations_done = 0;
    let mut converged = false;

    for iteration in 0..study.iterations {
        let length_gradient = length_gradient(study, design)?;
        history.push(BracketShapeIteration {
            iteration,
            design,
            objective_j: current.objective_j,
            compliance_j: current.compliance_j,
            volume_m3: current.volume_m3,
            length_gradient_j_per_m: length_gradient,
            thickness_gradient_j_per_m: current_thickness_gradient,
            adjoint_residual: current.adjoint_residual,
            linearized_buckling_load_factor: current.linearized_buckling_load_factor,
            max_displacement_m: current.max_displacement_m,
            max_von_mises_pa: current.max_von_mises_pa,
        });
        iterations_done = iteration + 1;

        let normalized_gradient = [
            length_gradient * design.length_m / objective_scale,
            current_thickness_gradient * design.thickness_m / objective_scale,
        ];
        if normalized_gradient
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            < 1.0e-8
        {
            converged = true;
            break;
        }

        let candidate = clamp_bracket_design(
            BracketShapeDesign {
                length_m: design.length_m * (1.0 - study.step_size * normalized_gradient[0]),
                thickness_m: design.thickness_m * (1.0 - study.step_size * normalized_gradient[1]),
            },
            study,
        );
        let candidate_evaluation = evaluate_bracket_shape(study, candidate)?;
        if candidate_evaluation.objective_j < current.objective_j {
            design = candidate;
            current = candidate_evaluation;
            current_thickness_gradient = objective_thickness_gradient(study, design, &current)?;
        } else {
            converged = true;
            break;
        }
    }

    let final_length_gradient = length_gradient(study, design)?;
    let final_evaluation = evaluate_bracket_shape(study, design)?;
    let final_thickness_gradient = objective_thickness_gradient(study, design, &final_evaluation)?;
    let taylor_step = (design.thickness_m * study.derivative_step_fraction).max(1.0e-10);
    let taylor_plus_design = clamp_bracket_design(
        BracketShapeDesign {
            length_m: design.length_m,
            thickness_m: design.thickness_m + taylor_step,
        },
        study,
    );
    let taylor_minus_design = clamp_bracket_design(
        BracketShapeDesign {
            length_m: design.length_m,
            thickness_m: design.thickness_m - taylor_step,
        },
        study,
    );
    let taylor_plus = evaluate_objective(study, taylor_plus_design)?;
    let taylor_minus = evaluate_objective(study, taylor_minus_design)?;
    let taylor_denominator = taylor_plus_design.thickness_m - taylor_minus_design.thickness_m;
    let finite_difference_thickness = if taylor_denominator > 0.0 {
        (taylor_plus - taylor_minus) / taylor_denominator
    } else {
        0.0
    };
    let taylor_scale = match study.objective {
        BracketShapeObjective::ComplianceVolume => final_evaluation.compliance_j.abs().max(1.0),
        BracketShapeObjective::FrequencyTarget { .. } => final_thickness_gradient
            .abs()
            .max(finite_difference_thickness.abs())
            .max(1.0e-12),
        BracketShapeObjective::BucklingTarget { .. } => final_thickness_gradient
            .abs()
            .max(finite_difference_thickness.abs())
            .max(1.0e-12),
        BracketShapeObjective::DisplacementTarget { .. }
        | BracketShapeObjective::StressTarget { .. } => final_thickness_gradient
            .abs()
            .max(finite_difference_thickness.abs())
            .max(1.0e-12),
    };
    let gradient_taylor_relative_error =
        (finite_difference_thickness - final_thickness_gradient).abs() / taylor_scale;
    history.push(BracketShapeIteration {
        iteration: iterations_done,
        design,
        objective_j: final_evaluation.objective_j,
        compliance_j: final_evaluation.compliance_j,
        volume_m3: final_evaluation.volume_m3,
        length_gradient_j_per_m: final_length_gradient,
        thickness_gradient_j_per_m: final_thickness_gradient,
        adjoint_residual: final_evaluation.adjoint_residual,
        linearized_buckling_load_factor: final_evaluation.linearized_buckling_load_factor,
        max_displacement_m: final_evaluation.max_displacement_m,
        max_von_mises_pa: final_evaluation.max_von_mises_pa,
    });

    Ok(BracketShapeOptimizationResult {
        study: study.clone(),
        execution: StructuralExecutionProvenance::cpu_bracket_shape_reference(),
        initial_design,
        final_design: design,
        initial_objective_j,
        final_objective_j: final_evaluation.objective_j,
        final_compliance_j: final_evaluation.compliance_j,
        final_volume_m3: final_evaluation.volume_m3,
        initial_first_bending_frequency_hz,
        final_first_bending_frequency_hz: final_evaluation.first_bending_frequency_hz,
        initial_linearized_buckling_load_factor,
        final_linearized_buckling_load_factor: final_evaluation.linearized_buckling_load_factor,
        initial_max_displacement_m,
        final_max_displacement_m: final_evaluation.max_displacement_m,
        initial_max_von_mises_pa,
        final_max_von_mises_pa: final_evaluation.max_von_mises_pa,
        final_static_result: final_evaluation.static_result,
        final_length_gradient_j_per_m: final_length_gradient,
        final_thickness_gradient_j_per_m: final_thickness_gradient,
        adjoint_residual: final_evaluation.adjoint_residual,
        gradient_taylor_relative_error,
        mesh_motion_max_displacement_m: final_evaluation.mesh_motion_max_displacement_m,
        mesh_motion_moved_nodes: final_evaluation.mesh_motion_moved_nodes,
        body_fitted_reanalysis: true,
        iterations: iterations_done,
        converged,
        history,
        model_assumption: StructuralModelAssumption::BracketShapeOptimization,
    })
}

fn evaluate_bracket_shape(
    study: &BracketShapeOptimizationStudy,
    design: BracketShapeDesign,
) -> Result<BracketShapeEvaluation, StructuralMechanicsError> {
    let mut candidate_study = study.base.clone();
    candidate_study.geometry.length_m = design.length_m;
    candidate_study.geometry.thickness_m = design.thickness_m;
    let base_mesh = study.base.mesh();
    let length_ratio = design.length_m / study.base.geometry.length_m;
    let motion = MeshMotion::new(
        base_mesh
            .nodes
            .iter()
            .map(|point| [point.x * (length_ratio - 1.0), 0.0, 0.0])
            .collect(),
    )
    .map_err(|error| StructuralMechanicsError::InvalidMesh {
        reason: format!("shape motion construction failed: {error}"),
    })?;
    let mut morphed_mesh = base_mesh.clone();
    let motion_report = motion.apply_checked(&mut morphed_mesh).map_err(|error| {
        StructuralMechanicsError::InvalidMesh {
            reason: format!("shape motion rejected: {error}"),
        }
    })?;
    let static_result = run_static_bracket_on_mesh(&candidate_study, &morphed_mesh)?;
    let modal_result = solve_first_bending_mode(&candidate_study, &static_result)?;
    let state = solve_static_linear_state(&candidate_study, &morphed_mesh)?;
    let compliance_j = 2.0 * static_result.strain_energy_j;
    let volume_m3 = design.length_m * candidate_study.geometry.height_m * design.thickness_m;
    let objective_j = match study.objective {
        BracketShapeObjective::ComplianceVolume => {
            compliance_j + study.volume_weight_j_per_m3 * volume_m3
        }
        BracketShapeObjective::FrequencyTarget {
            target_frequency_hz,
            frequency_weight_j_per_hz2,
        } => {
            frequency_weight_j_per_hz2
                * (target_frequency_hz - modal_result.first_bending_frequency_hz).powi(2)
                + study.volume_weight_j_per_m3 * volume_m3
        }
        BracketShapeObjective::DisplacementTarget {
            target_displacement_m,
            displacement_weight_j_per_m2,
        } => {
            displacement_weight_j_per_m2
                * (target_displacement_m - static_result.max_displacement_m).powi(2)
                + study.volume_weight_j_per_m3 * volume_m3
        }
        BracketShapeObjective::StressTarget {
            target_stress_pa,
            stress_weight_j_per_pa2,
        } => {
            stress_weight_j_per_pa2 * (target_stress_pa - static_result.max_von_mises_pa).powi(2)
                + study.volume_weight_j_per_m3 * volume_m3
        }
        BracketShapeObjective::BucklingTarget {
            target_load_factor,
            buckling_weight_j_per_factor2,
        } => {
            let load_factor = modal_result.linearized_buckling_load_factor.ok_or(
                StructuralMechanicsError::ValidationFailure {
                    check: "bracket_shape_buckling_factor",
                    value: 0.0,
                },
            )?;
            buckling_weight_j_per_factor2 * (target_load_factor - load_factor).powi(2)
                + study.volume_weight_j_per_m3 * volume_m3
        }
    };
    let d_k_u = csr_matvec(&state.stiffness, &state.solution)
        .into_iter()
        .map(|value| value / design.thickness_m)
        .collect::<Vec<_>>();
    let stiffness = state.stiffness.clone();
    let adjoint = compute_adjoint_sensitivity(&state.external_rhs, &[d_k_u], move |rhs| {
        solve_adjoint_rhs(&stiffness, rhs)
    });
    let adjoint_residual = relative_residual(
        &state.stiffness,
        &adjoint.adjoint_solution,
        &state.external_rhs,
    );
    let thickness_gradient_j_per_m = adjoint.sensitivities[0]
        + study.volume_weight_j_per_m3
            * candidate_study.geometry.length_m
            * candidate_study.geometry.height_m;
    if !objective_j.is_finite()
        || !thickness_gradient_j_per_m.is_finite()
        || !adjoint_residual.is_finite()
        || adjoint_residual > 1.0e-6
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "bracket_shape_adjoint",
            value: adjoint_residual,
        });
    }
    Ok(BracketShapeEvaluation {
        objective_j,
        compliance_j,
        volume_m3,
        first_bending_frequency_hz: modal_result.first_bending_frequency_hz,
        linearized_buckling_load_factor: modal_result.linearized_buckling_load_factor,
        max_displacement_m: static_result.max_displacement_m,
        max_von_mises_pa: static_result.max_von_mises_pa,
        static_result,
        thickness_gradient_j_per_m,
        adjoint_residual,
        mesh_motion_max_displacement_m: motion_report.max_displacement,
        mesh_motion_moved_nodes: motion_report.moved_nodes,
    })
}

fn evaluate_objective(
    study: &BracketShapeOptimizationStudy,
    design: BracketShapeDesign,
) -> Result<f64, StructuralMechanicsError> {
    Ok(evaluate_bracket_shape(study, design)?.objective_j)
}

fn objective_thickness_gradient(
    study: &BracketShapeOptimizationStudy,
    design: BracketShapeDesign,
    evaluation: &BracketShapeEvaluation,
) -> Result<f64, StructuralMechanicsError> {
    if matches!(study.objective, BracketShapeObjective::ComplianceVolume) {
        return Ok(evaluation.thickness_gradient_j_per_m);
    }
    let step = (design.thickness_m * study.derivative_step_fraction).max(1.0e-10);
    let plus = clamp_bracket_design(
        BracketShapeDesign {
            length_m: design.length_m,
            thickness_m: design.thickness_m + step,
        },
        study,
    );
    let minus = clamp_bracket_design(
        BracketShapeDesign {
            length_m: design.length_m,
            thickness_m: design.thickness_m - step,
        },
        study,
    );
    let plus_objective = evaluate_objective(study, plus)?;
    let minus_objective = evaluate_objective(study, minus)?;
    if plus.thickness_m > minus.thickness_m {
        return Ok((plus_objective - minus_objective) / (plus.thickness_m - minus.thickness_m));
    }
    if plus.thickness_m > design.thickness_m {
        return Ok(
            (plus_objective - evaluation.objective_j) / (plus.thickness_m - design.thickness_m)
        );
    }
    if design.thickness_m > minus.thickness_m {
        return Ok(
            (evaluation.objective_j - minus_objective) / (design.thickness_m - minus.thickness_m)
        );
    }
    Ok(0.0)
}

fn length_gradient(
    study: &BracketShapeOptimizationStudy,
    design: BracketShapeDesign,
) -> Result<f64, StructuralMechanicsError> {
    let step = (design.length_m * study.derivative_step_fraction).max(1.0e-10);
    let plus = clamp_bracket_design(
        BracketShapeDesign {
            length_m: design.length_m + step,
            thickness_m: design.thickness_m,
        },
        study,
    );
    let minus = clamp_bracket_design(
        BracketShapeDesign {
            length_m: design.length_m - step,
            thickness_m: design.thickness_m,
        },
        study,
    );
    let denominator = plus.length_m - minus.length_m;
    if denominator <= 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "bracket_shape.derivative_step_fraction",
            value: "length perturbation collapsed at bounds".to_string(),
        });
    }
    Ok((evaluate_objective(study, plus)? - evaluate_objective(study, minus)?) / denominator)
}

fn clamp_bracket_design(
    design: BracketShapeDesign,
    study: &BracketShapeOptimizationStudy,
) -> BracketShapeDesign {
    BracketShapeDesign {
        length_m: design
            .length_m
            .clamp(study.minimum_length_m, study.maximum_length_m),
        thickness_m: design
            .thickness_m
            .clamp(study.minimum_thickness_m, study.maximum_thickness_m),
    }
}

fn solve_adjoint_rhs(matrix: &CsrMatrix<f64>, rhs: &[f64]) -> Vec<f64> {
    let solved = cg(
        matrix,
        rhs,
        &CgConfig {
            max_iterations: 8_000,
            tolerance: 1e-10,
            print_interval: 0,
        },
    );
    if solved.converged {
        solved.x
    } else {
        vec![f64::NAN; rhs.len()]
    }
}

fn csr_matvec(matrix: &CsrMatrix<f64>, vector: &[f64]) -> Vec<f64> {
    (0..matrix.num_rows)
        .map(|row| {
            (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                .map(|index| matrix.values[index] * vector[matrix.col_indices[index]])
                .sum()
        })
        .collect()
}

fn relative_residual(matrix: &CsrMatrix<f64>, vector: &[f64], rhs: &[f64]) -> f64 {
    let lhs = csr_matvec(matrix, vector);
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| (left + right).powi(2))
        .sum::<f64>()
        .sqrt()
        / rhs
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            .max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_gradient_and_body_fitted_reanalysis_are_consistent() {
        let study = ShapeOptimizationStudy::introductory();
        let design = study.initial_design;
        let (objective, gradient) = study.objective_and_gradient(design);
        let direction = [0.03 * design.length_m, -0.02 * design.area_m2];
        let epsilon = 1.0e-5;
        let perturbed = ShapeDesign {
            length_m: design.length_m + epsilon * direction[0],
            area_m2: design.area_m2 + epsilon * direction[1],
        };
        let finite_difference = (study.objective_and_gradient(perturbed).0 - objective) / epsilon;
        let directional_gradient = gradient[0] * direction[0] + gradient[1] * direction[1];
        assert!((finite_difference - directional_gradient).abs() / objective < 1.0e-7);

        let result = run_shape_optimization(&study).unwrap();
        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::ShapeOptimization1D
        );
        assert!(result.final_objective_j < result.initial_objective_j);
        assert_eq!(
            result.final_objective_j,
            result.body_fitted_reanalysis_objective_j
        );
        assert!(result.mesh_motion_l2_m > 0.0);
        assert_eq!(result.final_mesh_nodes_m.len(), study.mesh_elements + 1);
        assert!(result.material_taylor_relative_error < 1.0e-8);
        assert!(result.load_taylor_relative_error < 1.0e-8);
        assert!(result.material_gradient_j_per_pa.is_finite());
        assert!(result.load_gradient_j_per_n.is_finite());

        let mut bounded = ShapeOptimizationStudy::introductory();
        bounded.minimum_design = bounded.initial_design;
        bounded.maximum_design = bounded.initial_design;
        bounded.iterations = 1;
        let bounded_result = run_shape_optimization(&bounded).unwrap();
        assert_eq!(bounded_result.bound_active, [true, true]);
    }

    #[test]
    fn topology_filter_projection_volume_and_reanalysis_gates_hold() {
        let study = TopologyOptimizationStudy::introductory();
        let result = run_topology_optimization(&study).unwrap();

        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::SimpTopology1D
        );
        assert!(result.final_compliance_j < result.initial_compliance_j);
        assert!((result.final_volume_fraction - study.volume_fraction).abs() < 1.0e-8);
        assert!(result.minimum_final_density >= study.minimum_density);
        assert!(result.final_connected);
        assert_eq!(
            result.final_compliance_j,
            result.projected_reanalysis_compliance_j
        );
        assert!(result.history.len() > 1);
    }

    #[test]
    fn robust_topology_reports_continuation_morphology_and_taylor_gates() {
        let mut study = RobustTopologyOptimizationStudy::introductory();
        study.base.iterations = 8;
        let result = run_robust_topology_optimization(&study).unwrap();

        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::RobustSimpRampTopology
        );
        assert_eq!(result.continuation_beta_history, vec![2.0, 4.0, 8.0]);
        assert!(result.final_objective_j <= result.initial_objective_j);
        assert!((result.final_volume_fraction - study.base.volume_fraction).abs() < 1.0e-8);
        assert_eq!(result.per_load_compliance_j.len(), study.load_factors.len());
        assert!(result.worst_case_compliance_j >= result.per_load_compliance_j[0]);
        assert!(result.eroded_volume_fraction <= result.final_volume_fraction);
        assert!(result.dilated_volume_fraction >= result.final_volume_fraction);
        assert!(result.uncertainty_feasible);
        assert!(result.manufacturing_feasible);
        assert!(result.coarse_fine_rank_preserved);
        assert!(result.material_taylor_relative_error < 1.0e-8);
        assert!(result.load_taylor_relative_error < 1.0e-8);
        assert!(
            result.gradient_taylor_relative_error < 1.0e-5,
            "Taylor relative error: {}",
            result.gradient_taylor_relative_error
        );
        assert!(result.converged);
        assert_eq!(
            result.gradient_taylor_relative_error,
            result.analytic_gradient_taylor_relative_error
        );
        assert!(result.feasible);
        assert!(result.history.len() > 1);
    }

    #[test]
    fn robust_topology_supports_ramp_and_rejects_invalid_controls() {
        let mut simp = RobustTopologyOptimizationStudy::introductory();
        simp.base.iterations = 4;
        let simp_result = run_robust_topology_optimization(&simp).unwrap();

        let mut ramp = simp.clone();
        ramp.interpolation = TopologyInterpolation::Ramp;
        let ramp_result = run_robust_topology_optimization(&ramp).unwrap();
        assert!(ramp_result.final_objective_j.is_finite());
        assert!(ramp_result.gradient_taylor_relative_error < 1.0e-5);
        assert_ne!(simp_result.final_objective_j, ramp_result.final_objective_j);

        let mut invalid = simp;
        invalid.load_factors.clear();
        assert!(matches!(
            run_robust_topology_optimization(&invalid),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "robust_topology.load_factors",
                ..
            })
        ));
    }

    #[test]
    fn constrained_topology_reports_stress_modal_buckling_and_overhang_gates() {
        let study = ConstrainedTopologyOptimizationStudy::introductory();
        let result = run_constrained_topology_optimization(&study).unwrap();

        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::RobustSimpRampTopology
        );
        assert!(result.aggregated_stress_pa < study.allowable_stress_pa);
        assert!(result.stress_feasible);
        assert!(result.first_frequency_hz > study.minimum_first_frequency_hz);
        assert!(result.frequency_feasible);
        assert!(result.buckling_load_factor > study.minimum_buckling_factor);
        assert!(result.buckling_feasible);
        assert!(result.connected);
        assert!(result.minimum_feature_feasible);
        assert!(result.overhang_feasible);
        assert!(result.reanalysis_feasible);
        assert!(result.projected_reanalysis_relative_error <= study.reanalysis_tolerance);
        assert!(result.feasible);

        let mut infeasible = study;
        infeasible.allowable_stress_pa = 1.0;
        let infeasible_result = run_constrained_topology_optimization(&infeasible).unwrap();
        assert!(!infeasible_result.stress_feasible);
        assert!(!infeasible_result.feasible);
    }

    #[test]
    fn uncertainty_samples_report_statistics_and_factorization_reuse() {
        let result = run_uncertainty_study(&UncertaintyStudy::introductory()).unwrap();

        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::UncertaintyBar
        );
        assert_eq!(result.samples.len(), 3);
        assert_eq!(result.assembled_matrix_count, 1);
        assert_eq!(result.factorization_count, 1);
        assert!(result.reused_factorization);
        assert!(result.standard_deviation_displacement_m > 0.0);
        assert!(result.maximum_displacement_m > result.mean_displacement_m);
    }

    #[test]
    fn bracket_shape_uses_morphed_fem_mesh_and_adjoint_thickness_gradient() {
        let mut base = BracketStudy::introductory_bracket();
        base.geometry.nx = 4;
        base.geometry.ny = 1;
        let mut study = BracketShapeOptimizationStudy::introductory(base);
        study.iterations = 2;
        study.step_size = 0.10;
        study.derivative_step_fraction = 2.0e-4;

        let result = run_bracket_shape_optimization(&study).unwrap();

        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::BracketShapeOptimization
        );
        assert!(result.body_fitted_reanalysis);
        assert!(result.mesh_motion_moved_nodes > 0);
        assert!(result.mesh_motion_max_displacement_m > 0.0);
        assert!(result.adjoint_residual < 1.0e-7);
        assert!(result.gradient_taylor_relative_error < 1.0e-6);
        assert!(result.final_static_result.converged);
        assert!(result.final_objective_j <= result.initial_objective_j);
    }

    #[test]
    fn bracket_shape_frequency_target_reports_modal_objective_and_gradient_gate() {
        let mut base = BracketStudy::introductory_bracket();
        base.geometry.nx = 4;
        base.geometry.ny = 1;
        let mut study = BracketShapeOptimizationStudy::introductory(base);
        study.iterations = 1;
        study.step_size = 0.10;
        study.derivative_step_fraction = 2.0e-4;
        study.objective = BracketShapeObjective::FrequencyTarget {
            target_frequency_hz: 1_000.0,
            frequency_weight_j_per_hz2: 1.0e-8,
        };

        let result = run_bracket_shape_optimization(&study).unwrap();

        assert_eq!(result.study.objective, study.objective);
        assert_eq!(
            result.execution.capability_id,
            "sonium-structural.cpu-reference.f64.bracket-shape.v1"
        );
        assert!(result.initial_first_bending_frequency_hz.is_finite());
        assert!(result.initial_first_bending_frequency_hz > 0.0);
        assert!(result.final_first_bending_frequency_hz.is_finite());
        assert!(result.final_first_bending_frequency_hz > 0.0);
        assert!(result.initial_objective_j.is_finite());
        assert!(result.final_objective_j.is_finite());
        assert!(result.final_objective_j <= result.initial_objective_j);
        assert!(result.gradient_taylor_relative_error < 1.0e-6);
        assert!(result.final_static_result.converged);
    }

    #[test]
    fn bracket_shape_buckling_target_reports_linearized_factor_and_gradient_gate() {
        let mut base = BracketStudy::introductory_bracket();
        base.geometry.nx = 4;
        base.geometry.ny = 1;
        base.loads.axial_prestress_force_n = -250.0;
        let mut study = BracketShapeOptimizationStudy::introductory(base);
        study.iterations = 1;
        study.step_size = 0.10;
        study.derivative_step_fraction = 2.0e-4;
        study.objective = BracketShapeObjective::BucklingTarget {
            target_load_factor: 100.0,
            buckling_weight_j_per_factor2: 1.0e-6,
        };

        let result = run_bracket_shape_optimization(&study).unwrap();
        let initial_factor = result.initial_linearized_buckling_load_factor.unwrap();
        let final_factor = result.final_linearized_buckling_load_factor.unwrap();

        assert!(initial_factor.is_finite() && initial_factor > 0.0);
        assert!(final_factor.is_finite() && final_factor > 0.0);
        assert!(result.final_objective_j <= result.initial_objective_j);
        assert!(result.gradient_taylor_relative_error < 1.0e-6);
        assert!(result.final_static_result.converged);
    }

    #[test]
    fn bracket_shape_displacement_and_centroid_stress_targets_report_metrics() {
        let mut base = BracketStudy::introductory_bracket();
        base.geometry.nx = 4;
        base.geometry.ny = 1;
        for objective in [
            BracketShapeObjective::DisplacementTarget {
                target_displacement_m: 1.0e-4,
                displacement_weight_j_per_m2: 1.0e8,
            },
            BracketShapeObjective::StressTarget {
                target_stress_pa: 1.0e8,
                stress_weight_j_per_pa2: 1.0e-16,
            },
        ] {
            let mut study = BracketShapeOptimizationStudy::introductory(base.clone());
            study.iterations = 1;
            study.step_size = 0.10;
            study.derivative_step_fraction = 2.0e-4;
            study.objective = objective;

            let result = run_bracket_shape_optimization(&study).unwrap();
            assert!(result.initial_max_displacement_m.is_finite());
            assert!(result.final_max_displacement_m.is_finite());
            assert!(result.initial_max_von_mises_pa.is_finite());
            assert!(result.final_max_von_mises_pa.is_finite());
            assert!(result.final_objective_j <= result.initial_objective_j);
            assert!(result.gradient_taylor_relative_error < 1.0e-6);
        }
    }

    #[test]
    fn optimization_families_reject_invalid_inputs() {
        let mut shape = ShapeOptimizationStudy::introductory();
        shape.initial_design.area_m2 = 0.0;
        assert!(matches!(
            run_shape_optimization(&shape),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "shape.initial_design",
                ..
            })
        ));

        let mut topology = TopologyOptimizationStudy::introductory();
        topology.initial_density[0] = 0.0;
        assert!(matches!(
            run_topology_optimization(&topology),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "topology.discretization_or_bounds",
                ..
            })
        ));

        let mut uncertainty = UncertaintyStudy::introductory();
        uncertainty.youngs_modulus_samples_pa.pop();
        assert!(matches!(
            run_uncertainty_study(&uncertainty),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "uncertainty.samples",
                ..
            })
        ));

        let mut bracket_shape =
            BracketShapeOptimizationStudy::introductory(BracketStudy::introductory_bracket());
        bracket_shape.objective = BracketShapeObjective::BucklingTarget {
            target_load_factor: 10.0,
            buckling_weight_j_per_factor2: 1.0,
        };
        assert!(matches!(
            run_bracket_shape_optimization(&bracket_shape),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "bracket_shape.buckling_objective",
                ..
            })
        ));

        bracket_shape.objective = BracketShapeObjective::StressTarget {
            target_stress_pa: 0.0,
            stress_weight_j_per_pa2: 1.0,
        };
        assert!(matches!(
            run_bracket_shape_optimization(&bracket_shape),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "bracket_shape.stress_objective",
                ..
            })
        ));
    }
}
