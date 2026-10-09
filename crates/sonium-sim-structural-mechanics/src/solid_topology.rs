//! Bounded assembled 3D solid SIMP topology reference workflow.
//!
//! This module intentionally targets a small, auditable P1 tetrahedral path:
//! element-local isotropic elasticity, exact multi-load compliance envelope
//! updates, density filtering/projection continuation, and final static gates
//! for volume, connectivity, stress, equilibrium, energy, and residuals. It
//! includes bounded first-frequency and linearized-buckling objectives. It
//! does not claim nonlinear or post-buckling 3D topology fidelity.

use crate::families::{Solid3DMultiLoadCase, Solid3DStudy};
use crate::{
    StructuralExecutionProvenance, StructuralMechanicsError, StructuralModelAssumption,
    TopologyInterpolation,
};
use serde::{Deserialize, Serialize};
use sonium_fem::mesh::{BoundaryType, ElementType, Mesh};
use sonium_fem::optimization::{DensityFilter, HeavisideProjection};
use sonium_fem::physics::solid::eigenmodes::build_linear_buckling_problem;
use sonium_solvers::{
    LobpcgConfig, MatrixClass, Ordering, SparseDirectConfig, SparseDirectMethod, TripletMatrix,
    factorize_sparse, lobpcg_generalized,
};

const DOFS_PER_NODE: usize = 3;

/// Geometric-stiffness model used by the bounded 3D buckling reanalysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Solid3DBucklingModel {
    /// The recovered compressive stress tensor was positive-semidefinite
    /// after changing sign, so all normal and shear components contributed.
    FullStressTensor,
    /// The recovered stress state had mixed signs. The bounded fallback uses
    /// only compressive `sigma_xx` and `dN/dx`, matching the original model.
    SigmaXxFallbackMixedSign,
}

/// Objective optimized by the assembled 3D SIMP topology workflow.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Solid3DTopologyObjective {
    /// Minimize the worst-case static compliance.
    #[default]
    Compliance,
    /// Maximize the first assembled mass/stiffness eigenfrequency.
    FirstFrequency,
    /// Maximize the bounded linearized initial-stress buckling factor.
    BucklingLoadFactor,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DTopologyStudy {
    pub base: Solid3DStudy,
    pub load_cases: Vec<Solid3DMultiLoadCase>,
    #[serde(default)]
    pub objective: Solid3DTopologyObjective,
    pub volume_fraction: f64,
    pub minimum_density: f64,
    pub interpolation: TopologyInterpolation,
    pub interpolation_parameter: f64,
    pub iterations: usize,
    pub move_limit: f64,
    pub convergence_tolerance: f64,
    /// P-norm exponent used to regularize the element stress constraint.
    /// This avoids treating a single centroid stress as the design objective.
    #[serde(default = "default_solid_stress_aggregation_p")]
    pub stress_aggregation_p: f64,
    pub allowable_stress_pa: f64,
    pub filter_radius_m: f64,
    pub projection_beta: f64,
    pub projection_threshold: f64,
    pub continuation_steps: usize,
    pub minimum_first_frequency_hz: f64,
    pub minimum_buckling_load_factor: f64,
    pub erosion_radius_cells: usize,
    pub dilation_radius_cells: usize,
    pub minimum_feature_cells: usize,
    /// Maximum vertical rise per horizontal neighbor spacing for the selected
    /// build direction. This is a bounded screening constraint, not a
    /// process-calibrated additive-manufacturing model.
    #[serde(default = "default_solid_maximum_overhang_slope")]
    pub maximum_overhang_slope: f64,
    pub require_body_fitted_reanalysis: bool,
}

fn default_solid_maximum_overhang_slope() -> f64 {
    10.0
}

fn default_solid_stress_aggregation_p() -> f64 {
    8.0
}

impl Default for Solid3DTopologyStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl Solid3DTopologyStudy {
    pub fn introductory() -> Self {
        let mut base = Solid3DStudy::introductory();
        base.geometry.nx = 4;
        base.geometry.ny = 2;
        base.geometry.nz = 2;
        let load_cases = vec![
            Solid3DMultiLoadCase {
                name: "axial-service".to_string(),
                force_n: [100.0, 0.0, 0.0],
            },
            Solid3DMultiLoadCase {
                name: "combined-service".to_string(),
                force_n: [70.0, 20.0, -10.0],
            },
        ];
        base.loads.force_n = load_cases[0].force_n;
        Self {
            base,
            load_cases,
            objective: Solid3DTopologyObjective::Compliance,
            volume_fraction: 0.70,
            minimum_density: 0.05,
            interpolation: TopologyInterpolation::Simp,
            interpolation_parameter: 3.0,
            iterations: 8,
            move_limit: 0.10,
            convergence_tolerance: 2.0e-4,
            stress_aggregation_p: default_solid_stress_aggregation_p(),
            allowable_stress_pa: 2.0e8,
            filter_radius_m: 0.03,
            projection_beta: 2.0,
            projection_threshold: 0.5,
            continuation_steps: 2,
            minimum_first_frequency_hz: 0.0,
            minimum_buckling_load_factor: 0.0,
            erosion_radius_cells: 1,
            dilation_radius_cells: 1,
            minimum_feature_cells: 1,
            maximum_overhang_slope: default_solid_maximum_overhang_slope(),
            require_body_fitted_reanalysis: true,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        self.base.mesh()?;
        if self.load_cases.is_empty()
            || !self.volume_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.volume_fraction)
            || !self.minimum_density.is_finite()
            || self.minimum_density <= 0.0
            || self.minimum_density > self.volume_fraction
            || !self.interpolation_parameter.is_finite()
            || self.interpolation_parameter <= 0.0
            || self.iterations == 0
            || !self.move_limit.is_finite()
            || self.move_limit <= 0.0
            || self.move_limit > 1.0
            || !self.convergence_tolerance.is_finite()
            || self.convergence_tolerance <= 0.0
            || !self.stress_aggregation_p.is_finite()
            || self.stress_aggregation_p < 1.0
            || !self.allowable_stress_pa.is_finite()
            || self.allowable_stress_pa <= 0.0
            || !self.filter_radius_m.is_finite()
            || self.filter_radius_m <= 0.0
            || !self.projection_beta.is_finite()
            || self.projection_beta <= 0.0
            || !self.projection_threshold.is_finite()
            || !(0.0..1.0).contains(&self.projection_threshold)
            || self.continuation_steps == 0
            || !self.minimum_first_frequency_hz.is_finite()
            || self.minimum_first_frequency_hz < 0.0
            || !self.minimum_buckling_load_factor.is_finite()
            || self.minimum_buckling_load_factor < 0.0
            || self.minimum_feature_cells == 0
            || !self.maximum_overhang_slope.is_finite()
            || self.maximum_overhang_slope < 0.0
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_topology.controls",
                value:
                    "invalid volume, interpolation, filter, projection, iteration, move, tolerance, or stress control"
                        .to_string(),
            });
        }
        let mut names = std::collections::HashSet::with_capacity(self.load_cases.len());
        for case in &self.load_cases {
            if case.name.trim().is_empty()
                || !names.insert(case.name.clone())
                || case.force_n.iter().any(|value| !value.is_finite())
                || case.force_n.iter().all(|value| value.abs() <= f64::EPSILON)
            {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "solid3d_topology.load_cases",
                    value: case.name.clone(),
                });
            }
        }
        if self.objective == Solid3DTopologyObjective::BucklingLoadFactor
            && !self
                .load_cases
                .iter()
                .any(|case| case.force_n[0] < -f64::EPSILON)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_topology.objective",
                value: "BucklingLoadFactor requires a compressive x-directed load".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DTopologyCaseResult {
    pub name: String,
    pub force_n: [f64; 3],
    pub compliance_j: f64,
    pub max_displacement_m: f64,
    pub max_von_mises_pa: f64,
    #[serde(default)]
    pub aggregated_stress_pa: f64,
    pub strain_energy_j: f64,
    pub solver_residual: f64,
    pub free_equilibrium_residual_n: f64,
    pub force_balance_residual_n: f64,
    pub energy_balance_relative_error: f64,
    pub converged: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DTopologyResult {
    pub study: Solid3DTopologyStudy,
    #[serde(default)]
    pub objective: Solid3DTopologyObjective,
    #[serde(default)]
    pub initial_objective_value: f64,
    #[serde(default)]
    pub final_objective_value: f64,
    #[serde(default)]
    pub objective_history: Vec<f64>,
    #[serde(default)]
    pub objective_improved: bool,
    #[serde(default)]
    pub objective_gradient_taylor_relative_error: f64,
    pub nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    pub iterations: usize,
    pub converged: bool,
    pub initial_compliance_j: f64,
    pub final_compliance_j: f64,
    pub critical_case: String,
    /// Physical density at the initial projected design state.
    #[serde(default)]
    pub initial_density: Vec<f64>,
    /// Unfiltered design variables at the final continuation stage.
    pub final_design_density: Vec<f64>,
    /// Physical densities used by the final assembled reanalysis.
    pub final_density: Vec<f64>,
    pub final_projection_beta: f64,
    pub continuation_beta_history: Vec<f64>,
    pub volume_fraction: f64,
    pub max_von_mises_pa: f64,
    #[serde(default)]
    pub aggregated_stress_pa: f64,
    pub first_frequency_hz: f64,
    pub modal_feasible: bool,
    pub buckling_load_factor: Option<f64>,
    pub buckling_case: Option<String>,
    #[serde(default)]
    pub buckling_model: Option<Solid3DBucklingModel>,
    pub buckling_feasible: bool,
    pub connected: bool,
    pub stress_feasible: bool,
    #[serde(default)]
    pub overhang_slope: f64,
    #[serde(default)]
    pub overhang_feasible: bool,
    pub eroded_density: Vec<f64>,
    pub dilated_density: Vec<f64>,
    pub eroded_volume_fraction: f64,
    pub dilated_volume_fraction: f64,
    pub morphology_feasible: bool,
    pub body_fitted_reanalysis: bool,
    pub body_fitted_feasible: bool,
    pub body_fitted_cases: Vec<Solid3DTopologyCaseResult>,
    pub factorization_reused_across_cases: bool,
    pub factorization_memory_bytes: usize,
    pub cases: Vec<Solid3DTopologyCaseResult>,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Schema version for a body-fitted topology design handoff.
pub const SOLID3D_TOPOLOGY_PROMOTION_SCHEMA_VERSION: u32 = 1;

/// A topology result admitted for downstream design/reporting use.
///
/// Promotion is deliberately stricter than producing a density field: the
/// result must be converged, feasible under all requested design constraints,
/// and have a finite body-fitted reanalysis for every load case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DTopologyPromotion {
    pub schema_version: u32,
    pub model_assumption: StructuralModelAssumption,
    pub objective: Solid3DTopologyObjective,
    pub final_density: Vec<f64>,
    pub volume_fraction: f64,
    pub objective_value: f64,
    pub objective_relative_change: f64,
    pub objective_non_regression_passed: bool,
    pub body_fitted_cases: Vec<Solid3DTopologyCaseResult>,
    pub accepted: bool,
    pub execution: StructuralExecutionProvenance,
}

/// Admit a 3-D topology result only after body-fitted reanalysis and design
/// feasibility gates pass.
pub fn promote_solid3d_topology(
    result: &Solid3DTopologyResult,
    max_objective_relative_increase: f64,
) -> Result<Solid3DTopologyPromotion, StructuralMechanicsError> {
    if !max_objective_relative_increase.is_finite() || max_objective_relative_increase < 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "solid3d_topology.promotion.max_objective_relative_increase",
            value: max_objective_relative_increase.to_string(),
        });
    }
    if !result.converged {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology.promotion.converged",
            value: 0.0,
        });
    }
    if result.final_density.is_empty()
        || result.final_density.iter().any(|density| {
            !density.is_finite()
                || *density < result.study.minimum_density - 1.0e-12
                || *density > 1.0 + 1.0e-12
        })
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology.promotion.final_density",
            value: result.final_density.len() as f64,
        });
    }
    if !result.volume_fraction.is_finite()
        || !result.connected
        || !result.stress_feasible
        || !result.morphology_feasible
        || !result.overhang_feasible
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology.promotion.design_constraints",
            value: result.volume_fraction,
        });
    }
    if !result.body_fitted_reanalysis
        || !result.body_fitted_feasible
        || result.body_fitted_cases.is_empty()
        || result.body_fitted_cases.iter().any(|case| {
            !case.converged
                || !case.solver_residual.is_finite()
                || !case.free_equilibrium_residual_n.is_finite()
                || !case.force_balance_residual_n.is_finite()
                || !case.energy_balance_relative_error.is_finite()
        })
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology.promotion.body_fitted_reanalysis",
            value: result.body_fitted_cases.len() as f64,
        });
    }
    if matches!(result.objective, Solid3DTopologyObjective::FirstFrequency)
        && !result.modal_feasible
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology.promotion.modal_feasibility",
            value: result.first_frequency_hz,
        });
    }
    if matches!(
        result.objective,
        Solid3DTopologyObjective::BucklingLoadFactor
    ) && !result.buckling_feasible
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology.promotion.buckling_feasibility",
            value: result.buckling_load_factor.unwrap_or(f64::NAN),
        });
    }
    if !result.initial_objective_value.is_finite() || !result.final_objective_value.is_finite() {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology.promotion.objective",
            value: result.final_objective_value,
        });
    }
    let scale = result.initial_objective_value.abs().max(1.0e-30);
    let adverse_change = match result.objective {
        Solid3DTopologyObjective::Compliance => {
            (result.final_objective_value - result.initial_objective_value) / scale
        }
        Solid3DTopologyObjective::FirstFrequency | Solid3DTopologyObjective::BucklingLoadFactor => {
            (result.initial_objective_value - result.final_objective_value) / scale
        }
    };
    if !adverse_change.is_finite() || adverse_change > max_objective_relative_increase {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology.promotion.objective_non_regression",
            value: adverse_change,
        });
    }

    let mut execution = result.execution.clone();
    execution.capability_id =
        "sonium-structural.cpu-reference.f64.solid3d-topology-promotion.v1".to_string();
    execution.limitations.push(
        "Promotion admits a checked body-fitted reference design; it is not product-scale manufacturing or external-solver evidence."
            .to_string(),
    );
    Ok(Solid3DTopologyPromotion {
        schema_version: SOLID3D_TOPOLOGY_PROMOTION_SCHEMA_VERSION,
        model_assumption: result.model_assumption,
        objective: result.objective,
        final_density: result.final_density.clone(),
        volume_fraction: result.volume_fraction,
        objective_value: result.final_objective_value,
        objective_relative_change: adverse_change,
        objective_non_regression_passed: true,
        body_fitted_cases: result.body_fitted_cases.clone(),
        accepted: true,
        execution,
    })
}

/// Schema version for the bounded coarse/fine 3-D topology convergence report.
pub const SOLID3D_TOPOLOGY_CONVERGENCE_SCHEMA_VERSION: u32 = 2;

/// Mesh-level diagnostics promoted from one 3-D topology result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DTopologyConvergenceLevel {
    pub nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    pub iterations: usize,
    pub objective_j: f64,
    pub compliance_j: f64,
    pub max_displacement_m: f64,
    pub aggregated_stress_pa: f64,
    pub first_frequency_hz: f64,
    pub volume_fraction: f64,
    pub max_solver_residual: f64,
    pub max_free_equilibrium_residual_n: f64,
    pub max_force_balance_residual_n: f64,
    pub max_energy_balance_relative_error: f64,
    pub objective_gradient_taylor_relative_error: f64,
    pub converged: bool,
    pub connected: bool,
    pub stress_feasible: bool,
    pub morphology_feasible: bool,
    pub body_fitted_feasible: bool,
}

/// Bounded 3-D topology mesh-convergence and projected-rank evidence.
///
/// The refined study doubles all three structured resolutions and re-runs the
/// public topology workflow. Rank preservation compares coarse element density
/// pairs after nearest-centroid projection onto the refined mesh; the report
/// exposes the pair counts and discordance fraction so this is not mistaken
/// for a universal topology-convergence theorem.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DTopologyConvergenceReport {
    pub schema_version: u32,
    /// Structured `(nx, ny, nz)` resolution used for the coarse solve.
    #[serde(default)]
    pub coarse_resolution: [usize; 3],
    /// Structured `(nx, ny, nz)` resolution used for the refined solve.
    #[serde(default)]
    pub refined_resolution: [usize; 3],
    pub coarse: Solid3DTopologyConvergenceLevel,
    pub refined: Solid3DTopologyConvergenceLevel,
    pub tolerance: f64,
    pub objective_relative_change: f64,
    pub stress_relative_change: f64,
    pub volume_fraction_relative_change: f64,
    pub rank_pair_count: usize,
    pub rank_discordant_pair_count: usize,
    pub rank_discordance_fraction: f64,
    pub rank_preserved: bool,
    pub converged: bool,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// A mesh-convergence report admitted for topology-design handoff.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DTopologyConvergencePromotion {
    pub schema_version: u32,
    pub coarse_resolution: [usize; 3],
    pub refined_resolution: [usize; 3],
    pub tolerance: f64,
    pub objective_relative_change: f64,
    pub stress_relative_change: f64,
    pub volume_fraction_relative_change: f64,
    pub rank_discordance_fraction: f64,
    pub rank_preserved: bool,
    pub accepted: bool,
    pub execution: StructuralExecutionProvenance,
}

/// Promote only a converged coarse/refined topology comparison with finite
/// feasibility metrics and preserved density ordering.
pub fn promote_solid3d_topology_mesh_convergence(
    report: &Solid3DTopologyConvergenceReport,
) -> Result<Solid3DTopologyConvergencePromotion, StructuralMechanicsError> {
    if report.schema_version != SOLID3D_TOPOLOGY_CONVERGENCE_SCHEMA_VERSION {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "solid3d_topology_convergence.schema_version",
            value: report.schema_version.to_string(),
        });
    }
    if report.model_assumption != StructuralModelAssumption::SimpTopology3D
        || !report.tolerance.is_finite()
        || report.tolerance <= 0.0
        || report.tolerance >= 1.0
    {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "solid3d_topology_convergence.promotion.controls",
            value: "3-D topology model and tolerance required".to_string(),
        });
    }
    if report
        .refined_resolution
        .iter()
        .zip(report.coarse_resolution)
        .any(|(refined, coarse)| *refined <= coarse)
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology_convergence.promotion.refinement",
            value: 0.0,
        });
    }
    let metrics = [
        report.objective_relative_change,
        report.stress_relative_change,
        report.volume_fraction_relative_change,
        report.rank_discordance_fraction,
    ];
    if metrics.iter().any(|metric| !metric.is_finite()) {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology_convergence.promotion.metrics",
            value: f64::NAN,
        });
    }
    let levels_are_admissible = [&report.coarse, &report.refined].into_iter().all(|level| {
        level.nodes > 0
            && level.elements > 0
            && level.dofs > 0
            && level.converged
            && level.connected
            && level.stress_feasible
            && level.morphology_feasible
            && level.body_fitted_feasible
            && [
                level.objective_j,
                level.compliance_j,
                level.max_displacement_m,
                level.aggregated_stress_pa,
                level.first_frequency_hz,
                level.volume_fraction,
                level.max_solver_residual,
                level.max_free_equilibrium_residual_n,
                level.max_force_balance_residual_n,
                level.max_energy_balance_relative_error,
                level.objective_gradient_taylor_relative_error,
            ]
            .iter()
            .all(|value| value.is_finite())
    });
    if !report.converged
        || !levels_are_admissible
        || report.rank_pair_count == 0
        || !report.rank_preserved
        || report.rank_discordant_pair_count > report.rank_pair_count
        || metrics.iter().any(|metric| *metric > report.tolerance)
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology_convergence.promotion.gates",
            value: report.rank_discordance_fraction,
        });
    }
    let mut execution = report.execution.clone();
    execution.capability_id =
        "sonium-structural.cpu-reference.f64.solid3d-topology-mesh-promotion.v1".to_string();
    execution.limitations.push(
        "Promotion admits a checked coarse/refined topology comparison; it is not a universal mesh-independence or product-scale manufacturing claim."
            .to_string(),
    );
    Ok(Solid3DTopologyConvergencePromotion {
        schema_version: SOLID3D_TOPOLOGY_CONVERGENCE_SCHEMA_VERSION,
        coarse_resolution: report.coarse_resolution,
        refined_resolution: report.refined_resolution,
        tolerance: report.tolerance,
        objective_relative_change: report.objective_relative_change,
        stress_relative_change: report.stress_relative_change,
        volume_fraction_relative_change: report.volume_fraction_relative_change,
        rank_discordance_fraction: report.rank_discordance_fraction,
        rank_preserved: report.rank_preserved,
        accepted: true,
        execution,
    })
}

/// One explicit multiplicative uncertainty scenario for a 3-D static
/// topology reanalysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DTopologyUncertaintyCase {
    pub name: String,
    pub load_scale: f64,
    pub youngs_modulus_scale: f64,
}

/// Robust morphology/uncertainty wrapper around the assembled 3-D topology
/// design path.
///
/// The nominal design is optimized once by [`run_solid3d_topology`]. Its
/// physical density is then reanalysed on nominal, eroded, and dilated
/// morphologies for every explicit load/material scenario. This is a bounded
/// deterministic CPU reference contract, not a calibrated probabilistic
/// optimizer or a nonlinear manufacturing process model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DRobustTopologyStudy {
    pub base: Solid3DTopologyStudy,
    pub uncertainty_cases: Vec<Solid3DTopologyUncertaintyCase>,
    pub erosion_radius_cells: usize,
    pub dilation_radius_cells: usize,
    pub allowable_compliance_j: f64,
}

impl Default for Solid3DRobustTopologyStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl Solid3DRobustTopologyStudy {
    pub fn introductory() -> Self {
        Self {
            base: Solid3DTopologyStudy::introductory(),
            uncertainty_cases: vec![
                Solid3DTopologyUncertaintyCase {
                    name: "nominal".to_string(),
                    load_scale: 1.0,
                    youngs_modulus_scale: 1.0,
                },
                Solid3DTopologyUncertaintyCase {
                    name: "weak-material-overload".to_string(),
                    load_scale: 1.10,
                    youngs_modulus_scale: 0.90,
                },
            ],
            erosion_radius_cells: 1,
            dilation_radius_cells: 1,
            allowable_compliance_j: 1.0,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        self.base.validate()?;
        if self.uncertainty_cases.is_empty()
            || self.erosion_radius_cells == 0
            || self.dilation_radius_cells == 0
            || !self.allowable_compliance_j.is_finite()
            || self.allowable_compliance_j <= 0.0
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_robust_topology.controls",
                value: "uncertainty cases, morphology radii, and compliance limit must be valid"
                    .to_string(),
            });
        }
        let mut names = std::collections::HashSet::with_capacity(self.uncertainty_cases.len());
        for case in &self.uncertainty_cases {
            if case.name.trim().is_empty()
                || !names.insert(case.name.clone())
                || !case.load_scale.is_finite()
                || case.load_scale <= 0.0
                || !case.youngs_modulus_scale.is_finite()
                || case.youngs_modulus_scale <= 0.0
            {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "solid3d_robust_topology.uncertainty_cases",
                    value: case.name.clone(),
                });
            }
        }
        Ok(())
    }
}

/// One static reanalysis result for a morphology, uncertainty case, and load.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DRobustTopologyCaseResult {
    pub morphology: String,
    pub uncertainty_case: String,
    pub load_case: String,
    pub compliance_j: f64,
    pub max_displacement_m: f64,
    #[serde(default)]
    pub aggregated_stress_pa: f64,
    pub solver_residual: f64,
    pub free_equilibrium_residual_n: f64,
    pub force_balance_residual_n: f64,
    pub energy_balance_relative_error: f64,
    pub converged: bool,
    pub feasible: bool,
}

/// Result of the robust 3-D morphology/uncertainty reanalysis packet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DRobustTopologyResult {
    pub study: Solid3DRobustTopologyStudy,
    pub nominal: Solid3DTopologyResult,
    pub nominal_density: Vec<f64>,
    pub eroded_density: Vec<f64>,
    pub dilated_density: Vec<f64>,
    pub nominal_volume_fraction: f64,
    pub eroded_volume_fraction: f64,
    pub dilated_volume_fraction: f64,
    pub cases: Vec<Solid3DRobustTopologyCaseResult>,
    pub worst_case_compliance_j: f64,
    pub worst_case: String,
    pub erosion_connected: bool,
    pub dilation_connected: bool,
    #[serde(default)]
    pub aggregated_stress_pa: f64,
    #[serde(default)]
    pub stress_feasible: bool,
    #[serde(default)]
    pub overhang_feasible: bool,
    pub morphology_feasible: bool,
    pub uncertainty_feasible: bool,
    pub robust_feasible: bool,
    pub factorization_reused_within_scenario: bool,
    pub gradient_taylor_relative_error: f64,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Schema version for a robust topology design handoff.
pub const SOLID3D_ROBUST_TOPOLOGY_PROMOTION_SCHEMA_VERSION: u32 = 1;

/// A robust topology result admitted after morphology and uncertainty gates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DRobustTopologyPromotion {
    pub schema_version: u32,
    pub nominal: Solid3DTopologyPromotion,
    pub scenario_count: usize,
    pub worst_case: String,
    pub worst_case_compliance_j: f64,
    pub aggregated_stress_pa: f64,
    pub gradient_taylor_relative_error: f64,
    pub morphology_feasible: bool,
    pub uncertainty_feasible: bool,
    pub stress_feasible: bool,
    pub overhang_feasible: bool,
    pub accepted: bool,
    pub execution: StructuralExecutionProvenance,
}

/// Admit a robust topology design only after nominal, morphology, multi-load,
/// and explicit material/load-uncertainty evidence passes.
pub fn promote_solid3d_robust_topology(
    result: &Solid3DRobustTopologyResult,
    max_objective_relative_increase: f64,
    max_gradient_taylor_relative_error: f64,
) -> Result<Solid3DRobustTopologyPromotion, StructuralMechanicsError> {
    if !max_gradient_taylor_relative_error.is_finite() || max_gradient_taylor_relative_error < 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "solid3d_robust_topology.promotion.max_gradient_taylor_relative_error",
            value: max_gradient_taylor_relative_error.to_string(),
        });
    }
    let nominal = promote_solid3d_topology(&result.nominal, max_objective_relative_increase)
        .map_err(|error| match error {
            StructuralMechanicsError::InvalidConfiguration { field, value } => {
                StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: format!("nominal: {value}"),
                }
            }
            StructuralMechanicsError::ValidationFailure { check, value } => {
                StructuralMechanicsError::ValidationFailure { check, value }
            }
            other => other,
        })?;
    let load_case_count = result.nominal.study.load_cases.len();
    let uncertainty_case_count = result.study.uncertainty_cases.len();
    let expected_case_count = 3usize
        .checked_mul(uncertainty_case_count)
        .and_then(|count| count.checked_mul(load_case_count))
        .ok_or(StructuralMechanicsError::InvalidConfiguration {
            field: "solid3d_robust_topology.promotion.case_count",
            value: "scenario count overflow".to_string(),
        })?;
    if expected_case_count == 0 || result.cases.len() != expected_case_count {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_robust_topology.promotion.case_coverage",
            value: result.cases.len() as f64,
        });
    }
    let density_lengths_match = result.nominal_density.len() == result.eroded_density.len()
        && result.nominal_density.len() == result.dilated_density.len()
        && result.nominal_density.len() == result.nominal.final_density.len();
    let densities_are_finite = result
        .nominal_density
        .iter()
        .chain(&result.eroded_density)
        .chain(&result.dilated_density)
        .all(|density| {
            density.is_finite()
                && *density >= result.nominal.study.minimum_density - 1.0e-12
                && *density <= 1.0 + 1.0e-12
        });
    let cases_are_feasible = result.cases.iter().all(|case| {
        case.converged
            && case.feasible
            && case.compliance_j.is_finite()
            && case.max_displacement_m.is_finite()
            && case.aggregated_stress_pa.is_finite()
            && case.solver_residual.is_finite()
            && case.free_equilibrium_residual_n.is_finite()
            && case.force_balance_residual_n.is_finite()
            && case.energy_balance_relative_error.is_finite()
    });
    let scalar_gates = result.worst_case_compliance_j.is_finite()
        && !result.worst_case.trim().is_empty()
        && result.aggregated_stress_pa.is_finite()
        && result.aggregated_stress_pa <= result.nominal.study.allowable_stress_pa
        && result.gradient_taylor_relative_error.is_finite()
        && result.gradient_taylor_relative_error <= max_gradient_taylor_relative_error;
    if !result.robust_feasible
        || !result.morphology_feasible
        || !result.uncertainty_feasible
        || !result.stress_feasible
        || !result.overhang_feasible
        || !result.erosion_connected
        || !result.dilation_connected
        || (load_case_count > 1 && !result.factorization_reused_within_scenario)
        || !density_lengths_match
        || !densities_are_finite
        || !cases_are_feasible
        || !scalar_gates
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_robust_topology.promotion.gates",
            value: result.gradient_taylor_relative_error,
        });
    }
    let mut execution = result.execution.clone();
    execution.capability_id =
        "sonium-structural.cpu-reference.f64.solid3d-robust-topology-promotion.v1".to_string();
    execution.limitations.push(
        "Promotion admits checked morphology and explicit uncertainty cases; it is not product-scale reliability, manufacturing, external-solver, or accelerator evidence."
            .to_string(),
    );
    Ok(Solid3DRobustTopologyPromotion {
        schema_version: SOLID3D_ROBUST_TOPOLOGY_PROMOTION_SCHEMA_VERSION,
        nominal,
        scenario_count: result.cases.len(),
        worst_case: result.worst_case.clone(),
        worst_case_compliance_j: result.worst_case_compliance_j,
        aggregated_stress_pa: result.aggregated_stress_pa,
        gradient_taylor_relative_error: result.gradient_taylor_relative_error,
        morphology_feasible: result.morphology_feasible,
        uncertainty_feasible: result.uncertainty_feasible,
        stress_feasible: result.stress_feasible,
        overhang_feasible: result.overhang_feasible,
        accepted: true,
        execution,
    })
}

#[derive(Debug, Clone)]
struct TopologyElement {
    nodes: [usize; 4],
    stiffness: [[f64; 12]; 12],
    protected: bool,
}

#[derive(Debug, Clone)]
struct Evaluation {
    cases: Vec<Solid3DTopologyCaseResult>,
    objective_j: f64,
    gradient: Vec<f64>,
    factorization_memory_bytes: usize,
}

struct BucklingReanalysis {
    load_factor: Option<f64>,
    case_name: Option<String>,
    model: Option<Solid3DBucklingModel>,
}

pub fn run_solid3d_topology(
    study: &Solid3DTopologyStudy,
) -> Result<Solid3DTopologyResult, StructuralMechanicsError> {
    study.validate()?;
    let mesh = study.base.mesh()?;
    let elements = prepare_elements(&mesh, study)?;
    let volumes: Vec<f64> = elements
        .iter()
        .map(|element| tetrahedron_volume(&mesh, &element.nodes))
        .collect();
    let protected_volume_fraction = weighted_mean_bool(
        &elements
            .iter()
            .map(|element| element.protected)
            .collect::<Vec<_>>(),
        &volumes,
    );
    if protected_volume_fraction > study.volume_fraction + 1.0e-12 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "solid3d_topology.volume_fraction",
            value: format!(
                "target {} below protected boundary fraction {}",
                study.volume_fraction, protected_volume_fraction
            ),
        });
    }

    let protected: Vec<bool> = elements.iter().map(|element| element.protected).collect();
    let filter = DensityFilter::build(&mesh, study.filter_radius_m);
    let mut design_density = vec![study.volume_fraction; elements.len()];
    for (rho, &is_protected) in design_density.iter_mut().zip(&protected) {
        if is_protected {
            *rho = 1.0;
        }
    }
    design_density = project_design_to_volume(
        &design_density,
        &filter,
        &protected,
        study,
        &volumes,
        study.projection_beta,
    );
    let initial_density = physical_density(
        &design_density,
        &filter,
        &protected,
        study,
        study.projection_beta,
    );
    let initial = evaluate(&mesh, &elements, study, &initial_density)?;
    let initial_compliance_j = initial.objective_j;
    let initial_objective_value =
        topology_objective_value(&mesh, &elements, study, &initial_density, &initial)?;
    let mut objective_history = vec![initial_objective_value];
    let mut iterations = 0;
    let mut converged = false;
    let continuation_beta_history: Vec<f64> = (0..study.continuation_steps)
        .map(|stage| study.projection_beta * 2.0_f64.powi(stage as i32))
        .collect();
    for &beta in &continuation_beta_history {
        for _ in 0..study.iterations {
            let density = physical_density(&design_density, &filter, &protected, study, beta);
            let evaluation = evaluate(&mesh, &elements, study, &density)?;
            let current_objective_value =
                topology_objective_value(&mesh, &elements, study, &density, &evaluation)?;
            let filtered = filter.filter_densities(&design_density);
            let projection = HeavisideProjection::new(beta, study.projection_threshold);
            let objective_derivative = topology_objective_derivative(
                &mesh,
                &elements,
                study,
                &density,
                &evaluation,
                &protected,
            )?;
            let maximize = matches!(
                study.objective,
                Solid3DTopologyObjective::FirstFrequency
                    | Solid3DTopologyObjective::BucklingLoadFactor
            );
            let filtered_gradient: Vec<f64> = objective_derivative
                .iter()
                .zip(&filtered)
                .zip(&protected)
                .map(|((&gradient, &filtered_density), &is_protected)| {
                    if is_protected {
                        0.0
                    } else {
                        let update_gradient = if maximize { -gradient } else { gradient };
                        update_gradient * projection.project_derivative(filtered_density)
                    }
                })
                .collect();
            let raw_gradient = filter.filter_sensitivities_transpose(&filtered_gradient);
            let max_gradient = raw_gradient
                .iter()
                .map(|value| value.abs())
                .fold(0.0, f64::max);
            if max_gradient <= f64::EPSILON {
                converged = true;
                break;
            }
            let mut candidate = design_density.clone();
            for ((value, &gradient), &is_protected) in
                candidate.iter_mut().zip(&raw_gradient).zip(&protected)
            {
                if !is_protected {
                    *value = (*value - study.move_limit * gradient / max_gradient)
                        .clamp(study.minimum_density, 1.0);
                } else {
                    *value = 1.0;
                }
            }
            let candidate =
                project_design_to_volume(&candidate, &filter, &protected, study, &volumes, beta);
            let change = candidate
                .iter()
                .zip(&design_density)
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0, f64::max);
            let candidate_density = physical_density(&candidate, &filter, &protected, study, beta);
            let candidate_evaluation = evaluate(&mesh, &elements, study, &candidate_density)?;
            let next_objective_value = topology_objective_value(
                &mesh,
                &elements,
                study,
                &candidate_density,
                &candidate_evaluation,
            )?;
            design_density = candidate;
            iterations += 1;
            objective_history.push(next_objective_value);
            let relative_objective_change = (current_objective_value - next_objective_value).abs()
                / current_objective_value.abs().max(1.0);
            if change <= study.convergence_tolerance
                && relative_objective_change <= study.convergence_tolerance
            {
                converged = true;
                break;
            }
        }
    }
    let final_projection_beta = continuation_beta_history.last().copied().ok_or(
        StructuralMechanicsError::InvalidConfiguration {
            field: "solid3d_topology.continuation_steps",
            value: "must be non-zero".to_string(),
        },
    )?;
    let density = physical_density(
        &design_density,
        &filter,
        &protected,
        study,
        final_projection_beta,
    );
    let final_evaluation = evaluate(&mesh, &elements, study, &density)?;
    let final_objective_value =
        topology_objective_value(&mesh, &elements, study, &density, &final_evaluation)?;
    let final_objective_derivative = topology_objective_derivative(
        &mesh,
        &elements,
        study,
        &density,
        &final_evaluation,
        &protected,
    )?;
    let objective_gradient_taylor_relative_error = objective_gradient_taylor_error(
        &mesh,
        &elements,
        study,
        &density,
        &final_objective_derivative,
    )?;
    let objective_improved = match study.objective {
        Solid3DTopologyObjective::Compliance => {
            final_objective_value <= initial_objective_value * (1.0 + 1.0e-6)
        }
        Solid3DTopologyObjective::FirstFrequency | Solid3DTopologyObjective::BucklingLoadFactor => {
            final_objective_value >= initial_objective_value * (1.0 - 1.0e-6)
        }
    };
    let volume_fraction = weighted_mean_f64(&density, &volumes);
    let max_von_mises_pa = final_evaluation
        .cases
        .iter()
        .map(|case| case.max_von_mises_pa)
        .fold(0.0, f64::max);
    let aggregated_stress_pa = final_evaluation
        .cases
        .iter()
        .map(|case| case.aggregated_stress_pa)
        .fold(0.0, f64::max);
    let first_frequency_hz = first_frequency(&mesh, &elements, study, &density)?;
    let modal_feasible = first_frequency_hz >= study.minimum_first_frequency_hz;
    let BucklingReanalysis {
        load_factor: buckling_load_factor,
        case_name: buckling_case,
        model: buckling_model,
    } = linearized_buckling_load_factor(&mesh, &elements, study, &density)?;
    let buckling_feasible = buckling_load_factor
        .map(|factor| factor >= study.minimum_buckling_load_factor)
        .unwrap_or(true);
    let connected = connected_design(&elements, &density);
    let stress_feasible = aggregated_stress_pa <= study.allowable_stress_pa;
    let horizontal_cell_m = (study.base.geometry.length_m / study.base.geometry.nx as f64)
        .min(study.base.geometry.width_m / study.base.geometry.ny as f64);
    let overhang_slope = solid_overhang_slope(&mesh, &elements, &density, horizontal_cell_m);
    let overhang_feasible = overhang_slope <= study.maximum_overhang_slope;
    let eroded_density = morphology_density(
        &elements,
        &density,
        study.minimum_density,
        study.erosion_radius_cells,
        true,
    );
    let dilated_density = morphology_density(
        &elements,
        &density,
        study.minimum_density,
        study.dilation_radius_cells,
        false,
    );
    let erosion_evaluation = evaluate(&mesh, &elements, study, &eroded_density)?;
    let dilation_evaluation = evaluate(&mesh, &elements, study, &dilated_density)?;
    let morphology_feasible = connected_design(&elements, &eroded_density)
        && connected_design(&elements, &dilated_density)
        && minimum_feature_ok(&elements, &eroded_density, study.minimum_feature_cells)
        && minimum_feature_ok(&elements, &dilated_density, study.minimum_feature_cells)
        && erosion_evaluation.cases.iter().all(|case| case.converged)
        && dilation_evaluation.cases.iter().all(|case| case.converged);
    let (body_fitted_reanalysis, body_fitted_cases, body_fitted_feasible) =
        if study.require_body_fitted_reanalysis {
            let body_mesh = body_fitted_mesh(&mesh, &elements, &density, study)?;
            let body_elements = prepare_elements(&body_mesh, study)?;
            let body_density = vec![1.0; body_elements.len()];
            let body_evaluation = evaluate(&body_mesh, &body_elements, study, &body_density)?;
            let feasible = body_evaluation.cases.iter().all(|case| {
                case.converged
                    && case.solver_residual.is_finite()
                    && case.free_equilibrium_residual_n.is_finite()
                    && case.force_balance_residual_n.is_finite()
                    && case.energy_balance_relative_error.is_finite()
            });
            (true, body_evaluation.cases, feasible)
        } else {
            (false, Vec::new(), true)
        };
    let converged = converged
        && final_evaluation.cases.iter().all(|case| case.converged)
        && objective_improved
        && objective_gradient_taylor_relative_error.is_finite()
        && objective_gradient_taylor_relative_error <= 5.0e-2
        && connected
        && stress_feasible
        && modal_feasible
        && buckling_feasible
        && overhang_feasible
        && morphology_feasible
        && body_fitted_feasible;
    let critical_case = final_evaluation
        .cases
        .iter()
        .max_by(|lhs, rhs| lhs.compliance_j.total_cmp(&rhs.compliance_j))
        .map(|case| case.name.clone())
        .ok_or(StructuralMechanicsError::InvalidConfiguration {
            field: "solid3d_topology.load_cases",
            value: "empty".to_string(),
        })?;

    Ok(Solid3DTopologyResult {
        study: study.clone(),
        objective: study.objective,
        initial_objective_value,
        final_objective_value,
        objective_history,
        objective_improved,
        objective_gradient_taylor_relative_error,
        nodes: mesh.num_nodes(),
        elements: mesh.num_elements(),
        dofs: mesh.num_nodes() * DOFS_PER_NODE,
        iterations,
        converged,
        initial_compliance_j,
        final_compliance_j: final_evaluation.objective_j,
        critical_case,
        initial_density,
        final_design_density: design_density,
        final_density: density,
        final_projection_beta,
        continuation_beta_history,
        volume_fraction,
        max_von_mises_pa,
        aggregated_stress_pa,
        first_frequency_hz,
        modal_feasible,
        buckling_load_factor,
        buckling_case,
        buckling_model,
        buckling_feasible,
        connected,
        stress_feasible,
        overhang_slope,
        overhang_feasible,
        eroded_volume_fraction: weighted_mean_f64(&eroded_density, &volumes),
        dilated_volume_fraction: weighted_mean_f64(&dilated_density, &volumes),
        eroded_density,
        dilated_density,
        morphology_feasible,
        body_fitted_reanalysis,
        body_fitted_feasible,
        body_fitted_cases,
        factorization_reused_across_cases: true,
        factorization_memory_bytes: final_evaluation.factorization_memory_bytes,
        cases: final_evaluation.cases,
        model_assumption: StructuralModelAssumption::SimpTopology3D,
        execution: StructuralExecutionProvenance::cpu_solid3d_topology_reference(),
    })
}

/// Run a nominal 3-D topology design followed by robust morphology and
/// explicit material/load uncertainty reanalysis.
pub fn run_solid3d_robust_topology(
    study: &Solid3DRobustTopologyStudy,
) -> Result<Solid3DRobustTopologyResult, StructuralMechanicsError> {
    study.validate()?;
    let nominal = run_solid3d_topology(&study.base)?;
    let mesh = study.base.base.mesh()?;
    let elements = prepare_elements(&mesh, &study.base)?;
    let volumes: Vec<f64> = elements
        .iter()
        .map(|element| tetrahedron_volume(&mesh, &element.nodes))
        .collect();
    let nominal_density = nominal.final_density.clone();
    if nominal_density.len() != elements.len()
        || nominal_density.iter().any(|density| !density.is_finite())
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_robust_topology.nominal_density",
            value: nominal_density.len() as f64,
        });
    }
    let eroded_density = morphology_density(
        &elements,
        &nominal_density,
        study.base.minimum_density,
        study.erosion_radius_cells,
        true,
    );
    let dilated_density = morphology_density(
        &elements,
        &nominal_density,
        study.base.minimum_density,
        study.dilation_radius_cells,
        false,
    );
    let morphologies = [
        ("nominal", nominal_density.clone()),
        ("eroded", eroded_density.clone()),
        ("dilated", dilated_density.clone()),
    ];

    let mut cases = Vec::new();
    for (morphology, density) in morphologies {
        if density.len() != elements.len() || density.iter().any(|value| !value.is_finite()) {
            return Err(StructuralMechanicsError::ValidationFailure {
                check: "solid3d_robust_topology.morphology_density",
                value: density.len() as f64,
            });
        }
        for uncertainty in &study.uncertainty_cases {
            let mut scenario = study.base.clone();
            scenario.base.material.youngs_modulus_pa *= uncertainty.youngs_modulus_scale;
            scenario.load_cases = scenario
                .load_cases
                .iter()
                .map(|load_case| Solid3DMultiLoadCase {
                    name: load_case.name.clone(),
                    force_n: load_case
                        .force_n
                        .map(|value| value * uncertainty.load_scale),
                })
                .collect();
            scenario.base.loads.force_n = scenario
                .load_cases
                .first()
                .map(|case| case.force_n)
                .unwrap_or(scenario.base.loads.force_n);
            let scenario_elements = prepare_elements(&mesh, &scenario)?;
            let evaluation = evaluate(&mesh, &scenario_elements, &scenario, &density)?;
            for case in evaluation.cases {
                let finite = case.compliance_j.is_finite()
                    && case.max_displacement_m.is_finite()
                    && case.solver_residual.is_finite()
                    && case.free_equilibrium_residual_n.is_finite()
                    && case.force_balance_residual_n.is_finite()
                    && case.energy_balance_relative_error.is_finite();
                let feasible = case.converged
                    && finite
                    && case.aggregated_stress_pa.is_finite()
                    && case.aggregated_stress_pa <= study.base.allowable_stress_pa
                    && case.compliance_j <= study.allowable_compliance_j;
                cases.push(Solid3DRobustTopologyCaseResult {
                    morphology: morphology.to_string(),
                    uncertainty_case: uncertainty.name.clone(),
                    load_case: case.name,
                    compliance_j: case.compliance_j,
                    max_displacement_m: case.max_displacement_m,
                    aggregated_stress_pa: case.aggregated_stress_pa,
                    solver_residual: case.solver_residual,
                    free_equilibrium_residual_n: case.free_equilibrium_residual_n,
                    force_balance_residual_n: case.force_balance_residual_n,
                    energy_balance_relative_error: case.energy_balance_relative_error,
                    converged: case.converged && finite,
                    feasible,
                });
            }
        }
    }
    let worst = cases
        .iter()
        .max_by(|lhs, rhs| lhs.compliance_j.total_cmp(&rhs.compliance_j))
        .ok_or(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_robust_topology.cases",
            value: 0.0,
        })?;
    let worst_case_compliance_j = worst.compliance_j;
    let aggregated_stress_pa = cases
        .iter()
        .map(|case| case.aggregated_stress_pa)
        .fold(0.0, f64::max);
    let worst_case = format!(
        "{}::{}::{}",
        worst.morphology, worst.uncertainty_case, worst.load_case
    );
    let erosion_connected = connected_design(&elements, &eroded_density);
    let dilation_connected = connected_design(&elements, &dilated_density);
    let horizontal_cell_m = (study.base.base.geometry.length_m
        / study.base.base.geometry.nx as f64)
        .min(study.base.base.geometry.width_m / study.base.base.geometry.ny as f64);
    let overhang_feasible = nominal.overhang_feasible
        && solid_overhang_slope(&mesh, &elements, &eroded_density, horizontal_cell_m)
            <= study.base.maximum_overhang_slope
        && solid_overhang_slope(&mesh, &elements, &dilated_density, horizontal_cell_m)
            <= study.base.maximum_overhang_slope;
    let morphology_feasible = erosion_connected
        && dilation_connected
        && overhang_feasible
        && minimum_feature_ok(&elements, &eroded_density, study.base.minimum_feature_cells)
        && minimum_feature_ok(
            &elements,
            &dilated_density,
            study.base.minimum_feature_cells,
        )
        && cases.iter().all(|case| case.converged);
    let uncertainty_feasible = cases.iter().all(|case| case.feasible);
    let stress_feasible = aggregated_stress_pa <= study.base.allowable_stress_pa;
    let robust_feasible = morphology_feasible
        && uncertainty_feasible
        && stress_feasible
        && worst_case_compliance_j.is_finite()
        && nominal.objective_gradient_taylor_relative_error.is_finite();
    let eroded_volume_fraction = weighted_mean_f64(&eroded_density, &volumes);
    let dilated_volume_fraction = weighted_mean_f64(&dilated_density, &volumes);

    Ok(Solid3DRobustTopologyResult {
        study: study.clone(),
        nominal: nominal.clone(),
        nominal_density,
        eroded_density,
        dilated_density,
        nominal_volume_fraction: weighted_mean_f64(&nominal.final_density, &volumes),
        eroded_volume_fraction,
        dilated_volume_fraction,
        cases,
        worst_case_compliance_j,
        worst_case,
        erosion_connected,
        dilation_connected,
        aggregated_stress_pa,
        stress_feasible,
        overhang_feasible,
        morphology_feasible,
        uncertainty_feasible,
        robust_feasible,
        factorization_reused_within_scenario: study.base.load_cases.len() > 1,
        gradient_taylor_relative_error: nominal.objective_gradient_taylor_relative_error,
        model_assumption: StructuralModelAssumption::SimpTopology3D,
        execution: StructuralExecutionProvenance::cpu_solid3d_robust_topology_reference(),
    })
}

fn physical_density(
    design_density: &[f64],
    filter: &DensityFilter,
    protected: &[bool],
    study: &Solid3DTopologyStudy,
    beta: f64,
) -> Vec<f64> {
    let projection = HeavisideProjection::new(beta, study.projection_threshold);
    filter
        .filter_densities(design_density)
        .into_iter()
        .zip(protected)
        .map(|(filtered, &is_protected)| {
            if is_protected {
                1.0
            } else {
                projection
                    .project(filtered)
                    .clamp(study.minimum_density, 1.0)
            }
        })
        .collect()
}

fn project_design_to_volume(
    design_density: &[f64],
    filter: &DensityFilter,
    protected: &[bool],
    study: &Solid3DTopologyStudy,
    volumes: &[f64],
    beta: f64,
) -> Vec<f64> {
    let mut lower = -1.0;
    let mut upper = 1.0;
    for _ in 0..60 {
        let shift = 0.5 * (lower + upper);
        let candidate: Vec<f64> = design_density
            .iter()
            .zip(protected)
            .map(|(&rho, &is_protected)| {
                if is_protected {
                    1.0
                } else {
                    (rho + shift).clamp(study.minimum_density, 1.0)
                }
            })
            .collect();
        let projected = physical_density(&candidate, filter, protected, study, beta);
        if weighted_mean_f64(&projected, volumes) > study.volume_fraction {
            upper = shift;
        } else {
            lower = shift;
        }
    }
    let shift = 0.5 * (lower + upper);
    design_density
        .iter()
        .zip(protected)
        .map(|(&rho, &is_protected)| {
            if is_protected {
                1.0
            } else {
                (rho + shift).clamp(study.minimum_density, 1.0)
            }
        })
        .collect()
}

fn topology_objective_value(
    mesh: &Mesh,
    elements: &[TopologyElement],
    study: &Solid3DTopologyStudy,
    density: &[f64],
    evaluation: &Evaluation,
) -> Result<f64, StructuralMechanicsError> {
    match study.objective {
        Solid3DTopologyObjective::Compliance => Ok(evaluation.objective_j),
        Solid3DTopologyObjective::FirstFrequency => first_frequency(mesh, elements, study, density),
        Solid3DTopologyObjective::BucklingLoadFactor => {
            linearized_buckling_load_factor(mesh, elements, study, density)?
                .load_factor
                .ok_or(StructuralMechanicsError::ValidationFailure {
                    check: "solid3d_topology.buckling_objective_compression",
                    value: 0.0,
                })
        }
    }
}

fn topology_objective_derivative(
    mesh: &Mesh,
    elements: &[TopologyElement],
    study: &Solid3DTopologyStudy,
    density: &[f64],
    evaluation: &Evaluation,
    protected: &[bool],
) -> Result<Vec<f64>, StructuralMechanicsError> {
    match study.objective {
        Solid3DTopologyObjective::Compliance => Ok(evaluation.gradient.clone()),
        Solid3DTopologyObjective::FirstFrequency => {
            first_frequency_with_derivative(mesh, elements, study, density)
                .map(|(_, derivative)| derivative)
        }
        Solid3DTopologyObjective::BucklingLoadFactor => {
            buckling_objective_derivative(mesh, elements, study, density, protected)
        }
    }
}

fn first_frequency(
    mesh: &Mesh,
    elements: &[TopologyElement],
    study: &Solid3DTopologyStudy,
    density: &[f64],
) -> Result<f64, StructuralMechanicsError> {
    first_frequency_with_derivative(mesh, elements, study, density).map(|(frequency, _)| frequency)
}

fn first_frequency_with_derivative(
    mesh: &Mesh,
    elements: &[TopologyElement],
    study: &Solid3DTopologyStudy,
    density: &[f64],
) -> Result<(f64, Vec<f64>), StructuralMechanicsError> {
    if density.len() != elements.len() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: format!(
                "3D topology modal density data does not match element count (elements={}, densities={})",
                elements.len(),
                density.len()
            ),
        });
    }
    let dofs = mesh.num_nodes() * DOFS_PER_NODE;
    let mut stiffness = TripletMatrix::with_capacity(dofs, dofs, elements.len() * 144);
    let mut mass = TripletMatrix::with_capacity(dofs, dofs, elements.len() * 144);
    for (element, &rho) in elements.iter().zip(density) {
        let stiffness_ratio = stiffness_ratio(rho, study);
        let element_mass_scale =
            study.base.material.density_kg_m3 * rho * tetrahedron_volume(mesh, &element.nodes)
                / 20.0;
        for local_row in 0..12 {
            let row = element.nodes[local_row / 3] * 3 + local_row % 3;
            for local_col in 0..12 {
                let col = element.nodes[local_col / 3] * 3 + local_col % 3;
                stiffness.push(
                    row,
                    col,
                    stiffness_ratio * element.stiffness[local_row][local_col],
                );
                let row_node = local_row / 3;
                let col_node = local_col / 3;
                if local_row % 3 == local_col % 3 {
                    let coefficient = if row_node == col_node { 2.0 } else { 1.0 };
                    mass.push(row, col, coefficient * element_mass_scale);
                }
            }
        }
    }
    let fixed = fixed_dofs(mesh, study.base.loads.fixed_boundary_marker);
    let (stiffness, free_to_full) = super::reduce_triplets(&stiffness, &fixed);
    let (mass, _) = super::reduce_triplets(&mass, &fixed);
    if stiffness.num_rows == 0 || mass.num_rows == 0 {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "3D topology has no free DOFs for modal reanalysis".to_string(),
        });
    }
    let solved = lobpcg_generalized(
        &stiffness,
        &mass,
        &LobpcgConfig {
            num_eigenvalues: 1,
            max_iterations: 4_000,
            tolerance: 1.0e-9,
            ..LobpcgConfig::default()
        },
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "solid-3d topology modal reanalysis",
        message: error.to_string(),
    })?;
    if !solved.converged {
        return Err(StructuralMechanicsError::ModalNonConvergence {
            iterations: solved.iterations,
            residual: solved.residual_norms.iter().copied().fold(0.0, f64::max),
        });
    }
    let eigenvalue =
        solved
            .eigenvalues
            .first()
            .copied()
            .ok_or(StructuralMechanicsError::InvalidMesh {
                reason: "3D topology modal reanalysis returned no eigenvalue".to_string(),
            })?;
    if !eigenvalue.is_finite() || eigenvalue <= 0.0 {
        return Err(StructuralMechanicsError::UnstableModel { eigenvalue });
    }
    let mut mode = (0..stiffness.num_rows)
        .map(|row| solved.eigenvectors[[row, 0]])
        .collect::<Vec<_>>();
    let mass_norm = super::dot(&mode, &mass.matvec(&mode)).sqrt();
    if !mass_norm.is_finite() || mass_norm <= f64::EPSILON {
        return Err(StructuralMechanicsError::SolverFailure {
            stage: "solid-3d topology modal normalization",
            message: "eigenvector has zero or non-finite mass norm".to_string(),
        });
    }
    for value in &mut mode {
        *value /= mass_norm;
    }
    let mut full_mode = vec![0.0; dofs];
    for (reduced, &full) in free_to_full.iter().enumerate() {
        full_mode[full] = mode[reduced];
    }
    let frequency = eigenvalue.sqrt() / (2.0 * std::f64::consts::PI);
    let frequency_scale = 1.0 / (4.0 * std::f64::consts::PI * eigenvalue.sqrt());
    let derivative = elements
        .iter()
        .zip(density)
        .map(|(element, &rho)| {
            let local = element_displacement(&full_mode, &element.nodes);
            let d_stiffness =
                interpolation_derivative(rho, study) * quadratic_form(&element.stiffness, &local);
            let volume = tetrahedron_volume(mesh, &element.nodes);
            let mass_scale = study.base.material.density_kg_m3 * volume / 20.0;
            let d_mass = (0..3)
                .flat_map(|component| {
                    (0..4).flat_map(move |row| {
                        (0..4).map(move |column| {
                            mass_scale
                                * if row == column { 2.0 } else { 1.0 }
                                * local[row * 3 + component]
                                * local[column * 3 + component]
                        })
                    })
                })
                .sum::<f64>();
            (d_stiffness - eigenvalue * d_mass) * frequency_scale
        })
        .collect();
    Ok((frequency, derivative))
}

fn buckling_objective_derivative(
    mesh: &Mesh,
    elements: &[TopologyElement],
    study: &Solid3DTopologyStudy,
    density: &[f64],
    protected: &[bool],
) -> Result<Vec<f64>, StructuralMechanicsError> {
    let mut derivative = vec![0.0; elements.len()];
    for index in 0..elements.len() {
        if protected.get(index).copied().unwrap_or(false) {
            continue;
        }
        let rho = density[index];
        let plus = (rho + 1.0e-4).min(1.0);
        let minus = (rho - 1.0e-4).max(study.minimum_density);
        if (plus - minus).abs() <= f64::EPSILON {
            continue;
        }
        let mut plus_density = density.to_vec();
        plus_density[index] = plus;
        let plus_factor = linearized_buckling_load_factor(mesh, elements, study, &plus_density)?
            .load_factor
            .ok_or(StructuralMechanicsError::ValidationFailure {
                check: "solid3d_topology.buckling_objective_compression",
                value: 0.0,
            })?;
        let mut minus_density = density.to_vec();
        minus_density[index] = minus;
        let minus_factor = linearized_buckling_load_factor(mesh, elements, study, &minus_density)?
            .load_factor
            .ok_or(StructuralMechanicsError::ValidationFailure {
                check: "solid3d_topology.buckling_objective_compression",
                value: 0.0,
            })?;
        derivative[index] = (plus_factor - minus_factor) / (plus - minus);
    }
    Ok(derivative)
}

fn objective_gradient_taylor_error(
    mesh: &Mesh,
    elements: &[TopologyElement],
    study: &Solid3DTopologyStudy,
    density: &[f64],
    derivative: &[f64],
) -> Result<f64, StructuralMechanicsError> {
    let protected: Vec<bool> = elements.iter().map(|element| element.protected).collect();
    let Some(index) = density.iter().enumerate().find_map(|(index, &rho)| {
        (!protected[index] && rho > study.minimum_density + 0.05 && rho < 1.0 - 0.05)
            .then_some(index)
    }) else {
        return Ok(0.0);
    };
    let mut plus_density = density.to_vec();
    let mut minus_density = density.to_vec();
    plus_density[index] = (plus_density[index] + 1.0e-5).min(1.0);
    minus_density[index] = (minus_density[index] - 1.0e-5).max(study.minimum_density);
    let plus_evaluation = evaluate(mesh, elements, study, &plus_density)?;
    let minus_evaluation = evaluate(mesh, elements, study, &minus_density)?;
    let plus_value =
        topology_objective_value(mesh, elements, study, &plus_density, &plus_evaluation)?;
    let minus_value =
        topology_objective_value(mesh, elements, study, &minus_density, &minus_evaluation)?;
    let numerical = (plus_value - minus_value) / (plus_density[index] - minus_density[index]);
    let analytic = derivative[index];
    Ok((numerical - analytic).abs() / numerical.abs().max(analytic.abs()).max(1.0e-12))
}

/// Compute a bounded linearized buckling factor for the most compressive
/// explicitly x-directed load case.
///
/// The full stress tensor is used when its negative is positive-semidefinite.
/// Mixed-sign stress states use the bounded `sigma_xx`/`dN/dx` fallback. Cases
/// without negative x force return `None`, so transverse or tensile topology
/// studies do not acquire an invented buckling result.
fn linearized_buckling_load_factor(
    mesh: &Mesh,
    elements: &[TopologyElement],
    study: &Solid3DTopologyStudy,
    density: &[f64],
) -> Result<BucklingReanalysis, StructuralMechanicsError> {
    let Some(load_case) = study
        .load_cases
        .iter()
        .filter(|case| case.force_n[0] < -f64::EPSILON)
        .min_by(|lhs, rhs| lhs.force_n[0].total_cmp(&rhs.force_n[0]))
    else {
        return Ok(BucklingReanalysis {
            load_factor: None,
            case_name: None,
            model: None,
        });
    };

    let dofs = mesh.num_nodes() * DOFS_PER_NODE;
    let ratios: Vec<f64> = density
        .iter()
        .map(|&rho| stiffness_ratio(rho, study))
        .collect();
    let mut elastic = TripletMatrix::with_capacity(dofs, dofs, elements.len() * 144);
    for (element, &ratio) in elements.iter().zip(&ratios) {
        for local_row in 0..12 {
            let row = element.nodes[local_row / 3] * 3 + local_row % 3;
            for local_col in 0..12 {
                let col = element.nodes[local_col / 3] * 3 + local_col % 3;
                elastic.push(row, col, ratio * element.stiffness[local_row][local_col]);
            }
        }
    }
    let fixed = fixed_dofs(mesh, study.base.loads.fixed_boundary_marker);
    let (constrained, free_to_full) = super::reduce_triplets(&elastic, &fixed);
    if constrained.num_rows == 0 {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "3D topology has no free DOFs for buckling reanalysis".to_string(),
        });
    }
    let factorization = factorize_sparse(
        &constrained,
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..SparseDirectConfig::default()
        },
        None,
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "solid-3d topology buckling pre-load factorization",
        message: error.to_string(),
    })?;
    let external = traction_rhs(
        mesh,
        study.base.loads.loaded_boundary_marker,
        load_case.force_n,
    )?;
    let reduced_rhs: Vec<f64> = free_to_full.iter().map(|&dof| external[dof]).collect();
    let reduced_solution = factorization.solve(&reduced_rhs).map_err(|error| {
        StructuralMechanicsError::SolverFailure {
            stage: "solid-3d topology buckling pre-load solve",
            message: error.to_string(),
        }
    })?;
    let mut solution = vec![0.0; dofs];
    for (free, &full) in free_to_full.iter().enumerate() {
        solution[full] = reduced_solution[free];
    }

    let mut geometric = TripletMatrix::with_capacity(dofs, dofs, elements.len() * 36);
    let mut has_compression = false;
    let mut buckling_model = None;
    for (index, element) in elements.iter().enumerate() {
        let local = element_displacement(&solution, &element.nodes);
        let stress = element_stress(mesh, &element.nodes, study, &local);
        let gradients = tetrahedron_shape_gradients(mesh, &element.nodes);
        let volume = tetrahedron_volume(mesh, &element.nodes);
        if buckling_model_for_stress(stress) == Solid3DBucklingModel::FullStressTensor {
            let compression = scale_stress_tensor(stress, -ratios[index]);
            if compression
                .iter()
                .enumerate()
                .any(|(row, entries)| entries[row] > f64::EPSILON)
            {
                has_compression = true;
                buckling_model.get_or_insert(Solid3DBucklingModel::FullStressTensor);
                for a in 0..4 {
                    for b in 0..4 {
                        let value =
                            stress_gradient_product(&compression, gradients[a], gradients[b])
                                * volume;
                        for component in 0..3 {
                            let row = element.nodes[a] * 3 + component;
                            let column = element.nodes[b] * 3 + component;
                            geometric.push(row, column, value);
                        }
                    }
                }
            }
        } else {
            let compression_pa = -ratios[index] * stress[0];
            if !compression_pa.is_finite() || compression_pa <= 0.0 {
                continue;
            }
            has_compression = true;
            buckling_model = Some(Solid3DBucklingModel::SigmaXxFallbackMixedSign);
            for a in 0..4 {
                for b in 0..4 {
                    let value = compression_pa * gradients[a][0] * gradients[b][0] * volume;
                    for component in 0..3 {
                        let row = element.nodes[a] * 3 + component;
                        let column = element.nodes[b] * 3 + component;
                        geometric.push(row, column, value);
                    }
                }
            }
        }
    }
    if !has_compression {
        return Ok(BucklingReanalysis {
            load_factor: None,
            case_name: Some(load_case.name.clone()),
            model: None,
        });
    }

    let problem = build_linear_buckling_problem(&elastic, &geometric, &fixed);
    let solved = lobpcg_generalized(
        &problem.stiffness_like,
        &problem.mass_like,
        &LobpcgConfig {
            num_eigenvalues: 1,
            max_iterations: 2_000,
            tolerance: 1.0e-8,
            ..LobpcgConfig::default()
        },
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "solid-3d topology buckling",
        message: error.to_string(),
    })?;
    if !solved.converged {
        return Err(StructuralMechanicsError::ModalNonConvergence {
            iterations: solved.iterations,
            residual: solved.residual_norms.iter().copied().fold(0.0, f64::max),
        });
    }
    let load_factor = solved
        .eigenvalues
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .fold(f64::INFINITY, f64::min);
    if !load_factor.is_finite() {
        return Err(StructuralMechanicsError::UnstableModel {
            eigenvalue: load_factor,
        });
    }
    Ok(BucklingReanalysis {
        load_factor: Some(load_factor),
        case_name: Some(load_case.name.clone()),
        model: buckling_model,
    })
}

fn scale_stress_tensor(stress: [f64; 6], scale: f64) -> [[f64; 3]; 3] {
    [
        [scale * stress[0], scale * stress[5], scale * stress[4]],
        [scale * stress[5], scale * stress[1], scale * stress[3]],
        [scale * stress[4], scale * stress[3], scale * stress[2]],
    ]
}

fn stress_gradient_product(stress: &[[f64; 3]; 3], lhs: [f64; 3], rhs: [f64; 3]) -> f64 {
    (0..3)
        .map(|row| {
            lhs[row]
                * (0..3)
                    .map(|column| stress[row][column] * rhs[column])
                    .sum::<f64>()
        })
        .sum()
}

fn negative_stress_is_positive_semidefinite(stress: [f64; 6]) -> bool {
    if stress.iter().any(|value| !value.is_finite()) {
        return false;
    }
    let compression = scale_stress_tensor(stress, -1.0);
    let scale = compression
        .iter()
        .flat_map(|row| row.iter())
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    if scale <= f64::EPSILON {
        return true;
    }
    let diagonal_tolerance = 1.0e-10 * scale;
    let pair_tolerance = 1.0e-10 * scale * scale;
    let determinant_tolerance = 1.0e-10 * scale * scale * scale;
    let diagonal_ok = (0..3).all(|index| compression[index][index] >= -diagonal_tolerance);
    let pair_ok = [
        compression[0][0] * compression[1][1] - compression[0][1] * compression[1][0],
        compression[0][0] * compression[2][2] - compression[0][2] * compression[2][0],
        compression[1][1] * compression[2][2] - compression[1][2] * compression[2][1],
    ]
    .into_iter()
    .all(|minor| minor >= -pair_tolerance);
    let determinant = compression[0][0]
        * (compression[1][1] * compression[2][2] - compression[1][2] * compression[2][1])
        - compression[0][1]
            * (compression[1][0] * compression[2][2] - compression[1][2] * compression[2][0])
        + compression[0][2]
            * (compression[1][0] * compression[2][1] - compression[1][1] * compression[2][0]);
    diagonal_ok && pair_ok && determinant >= -determinant_tolerance
}

fn buckling_model_for_stress(stress: [f64; 6]) -> Solid3DBucklingModel {
    if negative_stress_is_positive_semidefinite(stress) {
        Solid3DBucklingModel::FullStressTensor
    } else {
        Solid3DBucklingModel::SigmaXxFallbackMixedSign
    }
}

fn prepare_elements(
    mesh: &Mesh,
    study: &Solid3DTopologyStudy,
) -> Result<Vec<TopologyElement>, StructuralMechanicsError> {
    let minimum_x = mesh
        .nodes
        .iter()
        .map(|point| point.x)
        .fold(f64::INFINITY, f64::min);
    let scale = mesh
        .nodes
        .iter()
        .map(|point| point.x.abs().max(point.y.abs()).max(point.z.abs()))
        .fold(1.0, f64::max);
    let tolerance = 1.0e-10 * scale;
    mesh.elements
        .iter()
        .map(|element| {
            let vertices = element.vertices();
            if vertices.len() != 4 || vertices.iter().any(|&node| node >= mesh.nodes.len()) {
                return Err(StructuralMechanicsError::InvalidMesh {
                    reason: "3D topology requires valid tetrahedral elements".to_string(),
                });
            }
            let nodes = [vertices[0], vertices[1], vertices[2], vertices[3]];
            let determinant = tetrahedron_determinant(mesh, &nodes);
            if !determinant.is_finite() || determinant.abs() <= f64::MIN_POSITIVE {
                return Err(StructuralMechanicsError::InvalidMesh {
                    reason: "3D topology contains a degenerate tetrahedron".to_string(),
                });
            }
            let stiffness = element_stiffness(mesh, &nodes, study.base.material);
            let protected = nodes
                .iter()
                .any(|&node| (mesh.nodes[node].x - minimum_x).abs() <= tolerance);
            Ok(TopologyElement {
                nodes,
                stiffness,
                protected,
            })
        })
        .collect()
}

fn morphology_density(
    elements: &[TopologyElement],
    density: &[f64],
    minimum_density: f64,
    radius: usize,
    erosion: bool,
) -> Vec<f64> {
    if radius == 0 {
        return density.to_vec();
    }
    let active: Vec<bool> = density
        .iter()
        .zip(elements)
        .map(|(&rho, element)| element.protected || rho >= 0.5)
        .collect();
    let neighbours = element_neighbours(elements);
    (0..elements.len())
        .map(|index| {
            let mut visited = vec![false; elements.len()];
            let mut frontier = vec![index];
            visited[index] = true;
            for _ in 0..radius {
                let mut next_frontier = Vec::new();
                for current in frontier {
                    for &next in &neighbours[current] {
                        if !visited[next] {
                            visited[next] = true;
                            next_frontier.push(next);
                        }
                    }
                }
                frontier = next_frontier;
                if frontier.is_empty() {
                    break;
                }
            }
            let survives = if erosion {
                visited
                    .iter()
                    .enumerate()
                    .all(|(neighbour, &included)| !included || active[neighbour])
            } else {
                visited
                    .iter()
                    .enumerate()
                    .any(|(neighbour, &included)| included && active[neighbour])
            };
            if elements[index].protected || survives {
                if erosion || active[index] {
                    density[index]
                } else {
                    1.0
                }
            } else {
                minimum_density
            }
        })
        .collect()
}

fn body_fitted_mesh(
    mesh: &Mesh,
    elements: &[TopologyElement],
    density: &[f64],
    study: &Solid3DTopologyStudy,
) -> Result<Mesh, StructuralMechanicsError> {
    let active: Vec<bool> = density
        .iter()
        .zip(elements)
        .map(|(&rho, element)| element.protected || rho >= 0.5)
        .collect();
    if !active.iter().any(|value| *value) {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology.body_fitted_active_domain",
            value: 0.0,
        });
    }
    let mut fitted = Mesh::new(3);
    for point in &mesh.nodes {
        fitted.add_node(*point);
    }
    for (element, &is_active) in elements.iter().zip(&active) {
        if is_active {
            fitted.add_element(ElementType::Tetrahedron, element.nodes.to_vec());
        }
    }
    if fitted.elements.is_empty() {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology.body_fitted_active_domain",
            value: 0.0,
        });
    }
    fitted.detect_boundaries();
    let min_x = fitted
        .nodes
        .iter()
        .map(|point| point.x)
        .fold(f64::INFINITY, f64::min);
    let max_x = fitted
        .nodes
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let scale = fitted
        .nodes
        .iter()
        .map(|point| point.x.abs().max(point.y.abs()).max(point.z.abs()))
        .fold(1.0, f64::max);
    let tolerance = 1.0e-10 * scale;
    fitted.set_boundary_condition(
        BoundaryType::Dirichlet,
        study.base.loads.fixed_boundary_marker,
        move |points| {
            points
                .iter()
                .all(|point| (point.x - min_x).abs() <= tolerance)
        },
    );
    fitted.set_boundary_condition(
        BoundaryType::Neumann,
        study.base.loads.loaded_boundary_marker,
        move |points| {
            points
                .iter()
                .all(|point| (point.x - max_x).abs() <= tolerance)
        },
    );
    Ok(fitted)
}

fn evaluate(
    mesh: &Mesh,
    elements: &[TopologyElement],
    study: &Solid3DTopologyStudy,
    density: &[f64],
) -> Result<Evaluation, StructuralMechanicsError> {
    let dofs = mesh.num_nodes() * DOFS_PER_NODE;
    let mut stiffness = TripletMatrix::with_capacity(dofs, dofs, elements.len() * 144);
    let ratios: Vec<f64> = density
        .iter()
        .map(|&rho| stiffness_ratio(rho, study))
        .collect();
    for (element, &ratio) in elements.iter().zip(&ratios) {
        for local_row in 0..12 {
            let row = element.nodes[local_row / 3] * 3 + local_row % 3;
            for local_col in 0..12 {
                let col = element.nodes[local_col / 3] * 3 + local_col % 3;
                stiffness.push(row, col, ratio * element.stiffness[local_row][local_col]);
            }
        }
    }
    let fixed_dofs = fixed_dofs(mesh, study.base.loads.fixed_boundary_marker);
    let (constrained, free_to_full) = super::reduce_triplets(&stiffness, &fixed_dofs);
    if constrained.num_rows == 0 {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "3D topology has no free DOFs after fixed boundary reduction".to_string(),
        });
    }
    let factorization = factorize_sparse(
        &constrained,
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..SparseDirectConfig::default()
        },
        None,
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "solid-3d topology factorization",
        message: error.to_string(),
    })?;
    let factorization_memory_bytes = factorization.factor_memory_bytes();
    let mut case_data = Vec::with_capacity(study.load_cases.len());
    let mut case_gradients = Vec::with_capacity(study.load_cases.len());

    for load_case in &study.load_cases {
        let external = traction_rhs(
            mesh,
            study.base.loads.loaded_boundary_marker,
            load_case.force_n,
        )?;
        let reduced_rhs = free_to_full
            .iter()
            .map(|&dof| external[dof])
            .collect::<Vec<_>>();
        let reduced_solution = factorization.solve(&reduced_rhs).map_err(|error| {
            StructuralMechanicsError::SolverFailure {
                stage: "solid-3d topology load-case solve",
                message: error.to_string(),
            }
        })?;
        let mut solution = vec![0.0; dofs];
        for (free, &full) in free_to_full.iter().enumerate() {
            solution[full] = reduced_solution[free];
        }
        let internal = super::triplet_matvec(&stiffness, &solution);
        let solver_residual = csr_residual(&constrained, &reduced_solution, &reduced_rhs);
        let free_equilibrium_residual_n = fixed_free_residual(&internal, &external, &fixed_dofs);
        let reaction = fixed_reaction(&internal, &external, &fixed_dofs);
        let force_balance_residual_n = norm3(add3(load_case.force_n, reaction));
        let internal_energy = 0.5 * super::dot(&solution, &internal);
        let external_energy = 0.5 * super::dot(&solution, &external);
        let energy_error =
            (internal_energy - external_energy).abs() / external_energy.abs().max(1.0);
        let compliance_j = super::dot(&solution, &external);
        let mut max_displacement_m: f64 = 0.0;
        for node in 0..mesh.num_nodes() {
            max_displacement_m = max_displacement_m.max(norm3([
                solution[node * 3],
                solution[node * 3 + 1],
                solution[node * 3 + 2],
            ]));
        }
        let mut max_von_mises_pa: f64 = 0.0;
        let mut stress_values = Vec::with_capacity(elements.len());
        let mut gradient: Vec<f64> = vec![0.0; elements.len()];
        for (index, element) in elements.iter().enumerate() {
            let local = element_displacement(&solution, &element.nodes);
            let strain_energy = quadratic_form(&element.stiffness, &local);
            let derivative = interpolation_derivative(density[index], study);
            let von_mises = element_von_mises(mesh, &element.nodes, study, &local);
            let stress_pa = von_mises * ratios[index];
            max_von_mises_pa = max_von_mises_pa.max(stress_pa);
            stress_values.push(stress_pa);
            if !gradient[index].is_finite() {
                gradient[index] = 0.0;
            }
            gradient[index] -= derivative * strain_energy;
        }
        let force_scale = norm3(load_case.force_n).max(1.0);
        let converged = solver_residual.is_finite()
            && free_equilibrium_residual_n.is_finite()
            && force_balance_residual_n.is_finite()
            && energy_error.is_finite()
            && solver_residual <= 1.0e-9
            && free_equilibrium_residual_n <= 1.0e-8 * force_scale
            && force_balance_residual_n <= 1.0e-8 * force_scale
            && energy_error <= 1.0e-9;
        let aggregated_stress_pa = aggregate_stress(&stress_values, study.stress_aggregation_p);
        case_data.push(Solid3DTopologyCaseResult {
            name: load_case.name.clone(),
            force_n: load_case.force_n,
            compliance_j,
            max_displacement_m,
            max_von_mises_pa,
            aggregated_stress_pa,
            strain_energy_j: internal_energy,
            solver_residual,
            free_equilibrium_residual_n,
            force_balance_residual_n,
            energy_balance_relative_error: energy_error,
            converged,
        });
        case_gradients.push(gradient);
    }

    let objective_j = case_data
        .iter()
        .map(|case| case.compliance_j)
        .fold(f64::NEG_INFINITY, f64::max);
    let active_case_indices: Vec<usize> = case_data
        .iter()
        .enumerate()
        .filter(|case| {
            (case.1.compliance_j - objective_j).abs() <= 1.0e-10 * objective_j.abs().max(1.0)
        })
        .map(|case| case.0)
        .collect();
    let active_cases = active_case_indices.len().max(1);
    let gradient = (0..elements.len())
        .map(|element| {
            active_case_indices
                .iter()
                .map(|&case| case_gradients[case][element])
                .sum::<f64>()
                / active_cases as f64
        })
        .collect();
    Ok(Evaluation {
        cases: case_data,
        objective_j,
        gradient,
        factorization_memory_bytes,
    })
}

fn aggregate_stress(stresses: &[f64], p: f64) -> f64 {
    if stresses.is_empty() {
        return f64::INFINITY;
    }
    (stresses.iter().map(|stress| stress.powf(p)).sum::<f64>() / stresses.len() as f64)
        .powf(1.0 / p)
}

fn stiffness_ratio(rho: f64, study: &Solid3DTopologyStudy) -> f64 {
    match study.interpolation {
        TopologyInterpolation::Simp => {
            study.minimum_density
                + (1.0 - study.minimum_density) * rho.powf(study.interpolation_parameter)
        }
        TopologyInterpolation::Ramp => {
            let q = study.interpolation_parameter;
            study.minimum_density + (1.0 - study.minimum_density) * rho / (1.0 + q * (1.0 - rho))
        }
    }
}

fn interpolation_derivative(rho: f64, study: &Solid3DTopologyStudy) -> f64 {
    match study.interpolation {
        TopologyInterpolation::Simp => {
            (1.0 - study.minimum_density)
                * study.interpolation_parameter
                * rho.powf((study.interpolation_parameter - 1.0).max(0.0))
        }
        TopologyInterpolation::Ramp => {
            let q = study.interpolation_parameter;
            (1.0 - study.minimum_density) * (1.0 + q) / (1.0 + q * (1.0 - rho)).powi(2)
        }
    }
}

fn element_stiffness(
    mesh: &Mesh,
    nodes: &[usize; 4],
    material: crate::StructuralMaterial,
) -> [[f64; 12]; 12] {
    let gradients = tetrahedron_shape_gradients(mesh, nodes);
    let volume = tetrahedron_volume(mesh, nodes);
    let c = material.as_fem().voigt_3d();
    let mut b = [[0.0; 12]; 6];
    for (local, gradient) in gradients.iter().enumerate() {
        let dof = local * 3;
        b[0][dof] = gradient[0];
        b[1][dof + 1] = gradient[1];
        b[2][dof + 2] = gradient[2];
        b[3][dof + 1] = gradient[2];
        b[3][dof + 2] = gradient[1];
        b[4][dof] = gradient[2];
        b[4][dof + 2] = gradient[0];
        b[5][dof] = gradient[1];
        b[5][dof + 1] = gradient[0];
    }
    let mut stiffness = [[0.0; 12]; 12];
    for row in 0..12 {
        for column in 0..12 {
            stiffness[row][column] = volume
                * (0..6)
                    .map(|a| {
                        (0..6)
                            .map(|b_index| b[a][row] * c[a * 6 + b_index] * b[b_index][column])
                            .sum::<f64>()
                    })
                    .sum::<f64>();
        }
    }
    stiffness
}

fn tetrahedron_shape_gradients(mesh: &Mesh, nodes: &[usize; 4]) -> [[f64; 3]; 4] {
    let p0 = mesh.nodes[nodes[0]];
    let p1 = mesh.nodes[nodes[1]];
    let p2 = mesh.nodes[nodes[2]];
    let p3 = mesh.nodes[nodes[3]];
    let b = subtract(p1, p0);
    let c = subtract(p2, p0);
    let d = subtract(p3, p0);
    let determinant = dot(b, cross(c, d));
    let g1 = scale(cross(c, d), 1.0 / determinant);
    let g2 = scale(cross(d, b), 1.0 / determinant);
    let g3 = scale(cross(b, c), 1.0 / determinant);
    [scale(add(add(g1, g2), g3), -1.0), g1, g2, g3]
}

fn tetrahedron_determinant(mesh: &Mesh, nodes: &[usize; 4]) -> f64 {
    let p0 = mesh.nodes[nodes[0]];
    let p1 = mesh.nodes[nodes[1]];
    let p2 = mesh.nodes[nodes[2]];
    let p3 = mesh.nodes[nodes[3]];
    dot(subtract(p1, p0), cross(subtract(p2, p0), subtract(p3, p0)))
}

fn tetrahedron_volume(mesh: &Mesh, nodes: &[usize; 4]) -> f64 {
    tetrahedron_determinant(mesh, nodes).abs() / 6.0
}

fn traction_rhs(
    mesh: &Mesh,
    marker: i32,
    force: [f64; 3],
) -> Result<Vec<f64>, StructuralMechanicsError> {
    let mut rhs = vec![0.0; mesh.num_nodes() * DOFS_PER_NODE];
    let mut area = 0.0;
    for boundary in mesh
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == marker)
    {
        if boundary.nodes.len() != 3 {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: "3D topology loaded boundary is not triangular".to_string(),
            });
        }
        let p0 = mesh.nodes[boundary.nodes[0]];
        let p1 = mesh.nodes[boundary.nodes[1]];
        let p2 = mesh.nodes[boundary.nodes[2]];
        area += 0.5 * norm3(cross(subtract(p1, p0), subtract(p2, p0)));
    }
    if !area.is_finite() || area <= 0.0 {
        return Err(StructuralMechanicsError::MissingLoadedBoundary { marker });
    }
    let traction = force.map(|value| value / area);
    for boundary in mesh
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == marker)
    {
        let p0 = mesh.nodes[boundary.nodes[0]];
        let p1 = mesh.nodes[boundary.nodes[1]];
        let p2 = mesh.nodes[boundary.nodes[2]];
        let face_area = 0.5 * norm3(cross(subtract(p1, p0), subtract(p2, p0)));
        for &node in &boundary.nodes {
            for component in 0..3 {
                rhs[node * 3 + component] += traction[component] * face_area / 3.0;
            }
        }
    }
    Ok(rhs)
}

fn fixed_dofs(mesh: &Mesh, marker: i32) -> Vec<usize> {
    let mut dofs = mesh
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == marker)
        .flat_map(|boundary| {
            boundary
                .nodes
                .iter()
                .flat_map(|&node| [node * 3, node * 3 + 1, node * 3 + 2])
        })
        .collect::<Vec<_>>();
    dofs.sort_unstable();
    dofs.dedup();
    dofs
}

fn element_displacement(solution: &[f64], nodes: &[usize; 4]) -> [f64; 12] {
    std::array::from_fn(|index| solution[nodes[index / 3] * 3 + index % 3])
}

fn element_stress(
    mesh: &Mesh,
    nodes: &[usize; 4],
    study: &Solid3DTopologyStudy,
    displacement: &[f64; 12],
) -> [f64; 6] {
    let gradients = tetrahedron_shape_gradients(mesh, nodes);
    let mut strain = [0.0; 6];
    for (local, gradient) in gradients.iter().enumerate() {
        let dof = local * 3;
        let ux = displacement[dof];
        let uy = displacement[dof + 1];
        let uz = displacement[dof + 2];
        strain[0] += ux * gradient[0];
        strain[1] += uy * gradient[1];
        strain[2] += uz * gradient[2];
        strain[3] += uy * gradient[2] + uz * gradient[1];
        strain[4] += ux * gradient[2] + uz * gradient[0];
        strain[5] += ux * gradient[1] + uy * gradient[0];
    }
    let c = study.base.material.as_fem().voigt_3d();
    let stress: [f64; 6] = std::array::from_fn(|row| {
        (0..6)
            .map(|column| c[row * 6 + column] * strain[column])
            .sum::<f64>()
    });
    stress
}

fn element_von_mises(
    mesh: &Mesh,
    nodes: &[usize; 4],
    study: &Solid3DTopologyStudy,
    displacement: &[f64; 12],
) -> f64 {
    let stress = element_stress(mesh, nodes, study, displacement);
    (0.5 * ((stress[0] - stress[1]).powi(2)
        + (stress[1] - stress[2]).powi(2)
        + (stress[2] - stress[0]).powi(2))
        + 3.0 * (stress[3].powi(2) + stress[4].powi(2) + stress[5].powi(2)))
    .sqrt()
}

fn quadratic_form(matrix: &[[f64; 12]; 12], vector: &[f64; 12]) -> f64 {
    (0..12)
        .map(|row| {
            vector[row]
                * (0..12)
                    .map(|column| matrix[row][column] * vector[column])
                    .sum::<f64>()
        })
        .sum()
}

fn csr_residual(matrix: &sonium_solvers::CsrMatrix<f64>, solution: &[f64], rhs: &[f64]) -> f64 {
    let squared = (0..matrix.num_rows)
        .map(|row| {
            let value = (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                .map(|index| matrix.values[index] * solution[matrix.col_indices[index]])
                .sum::<f64>()
                - rhs[row];
            value * value
        })
        .sum::<f64>();
    squared.sqrt()
        / rhs
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            .max(1.0)
}

fn fixed_free_residual(internal: &[f64], external: &[f64], fixed: &[usize]) -> f64 {
    let mut is_fixed = vec![false; internal.len()];
    for &dof in fixed {
        is_fixed[dof] = true;
    }
    internal
        .iter()
        .zip(external)
        .enumerate()
        .filter(|(index, _)| !is_fixed[*index])
        .map(|(_, (&lhs, &rhs))| (lhs - rhs).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn fixed_reaction(internal: &[f64], external: &[f64], fixed: &[usize]) -> [f64; 3] {
    fixed.iter().fold([0.0; 3], |mut result, &dof| {
        result[dof % 3] += internal[dof] - external[dof];
        result
    })
}

fn connected_design(elements: &[TopologyElement], density: &[f64]) -> bool {
    let neighbours = element_neighbours(elements);
    let active: Vec<bool> = density.iter().map(|&rho| rho >= 0.5).collect();
    let Some(start) = elements
        .iter()
        .enumerate()
        .find(|(index, element)| element.protected && active[*index])
        .map(|(index, _)| index)
    else {
        return false;
    };
    let mut visited = vec![false; elements.len()];
    let mut stack = vec![start];
    while let Some(index) = stack.pop() {
        if visited[index] {
            continue;
        }
        visited[index] = true;
        for &next in &neighbours[index] {
            if active[next] && !visited[next] {
                stack.push(next);
            }
        }
    }
    active
        .iter()
        .enumerate()
        .all(|(index, is_active)| !*is_active || visited[index])
}

fn minimum_feature_ok(elements: &[TopologyElement], density: &[f64], minimum_cells: usize) -> bool {
    if minimum_cells <= 1 {
        return true;
    }
    let neighbours = element_neighbours(elements);
    density.iter().enumerate().all(|(index, &rho)| {
        rho < 0.5
            || neighbours[index]
                .iter()
                .filter(|&&neighbour| density[neighbour] >= 0.5)
                .count()
                + 1
                >= minimum_cells
    })
}

/// Return the largest build-direction rise per horizontal neighbor spacing
/// among adjacent active tetrahedra.
///
/// The metric is deliberately geometric and deterministic: it screens the
/// projected design against a declared build direction, while leaving
/// process-specific powder, support, and material calibration outside this
/// reference path.
fn solid_overhang_slope(
    mesh: &Mesh,
    elements: &[TopologyElement],
    density: &[f64],
    minimum_horizontal_cell_m: f64,
) -> f64 {
    if elements.is_empty() || density.len() != elements.len() {
        return f64::INFINITY;
    }
    let neighbours = element_neighbours(elements);
    let active: Vec<bool> = density.iter().map(|&rho| rho >= 0.5).collect();
    let mut maximum: f64 = 0.0;
    for (index, element) in elements.iter().enumerate() {
        if !active[index] {
            continue;
        }
        let centroid = element_centroid(mesh, &element.nodes);
        for &neighbour in &neighbours[index] {
            if !active[neighbour] {
                continue;
            }
            let other = element_centroid(mesh, &elements[neighbour].nodes);
            let horizontal = ((centroid[0] - other[0]).powi(2) + (centroid[1] - other[1]).powi(2))
                .sqrt()
                .max(minimum_horizontal_cell_m);
            let rise = (centroid[2] - other[2]).abs();
            maximum = maximum.max(rise / horizontal);
        }
    }
    maximum
}

fn element_centroid(mesh: &Mesh, nodes: &[usize; 4]) -> [f64; 3] {
    let mut centroid = [0.0; 3];
    for &node in nodes {
        let point = mesh.nodes[node];
        centroid[0] += point.x;
        centroid[1] += point.y;
        centroid[2] += point.z;
    }
    [centroid[0] / 4.0, centroid[1] / 4.0, centroid[2] / 4.0]
}

fn element_neighbours(elements: &[TopologyElement]) -> Vec<Vec<usize>> {
    let mut neighbours = vec![Vec::new(); elements.len()];
    for i in 0..elements.len() {
        for j in (i + 1)..elements.len() {
            let common = elements[i]
                .nodes
                .iter()
                .filter(|node| elements[j].nodes.contains(node))
                .count();
            if common >= 3 {
                neighbours[i].push(j);
                neighbours[j].push(i);
            }
        }
    }
    neighbours
}

fn weighted_mean_bool(values: &[bool], weights: &[f64]) -> f64 {
    weighted_mean_iter(
        values.iter().map(|value| if *value { 1.0 } else { 0.0 }),
        weights,
    )
}

fn weighted_mean_f64(values: &[f64], weights: &[f64]) -> f64 {
    weighted_mean_iter(values.iter().copied(), weights)
}

fn weighted_mean_iter<I>(values: I, weights: &[f64]) -> f64
where
    I: Iterator<Item = f64>,
{
    let (weighted, total) = values
        .zip(weights)
        .fold((0.0, 0.0), |(weighted, total), (value, weight)| {
            (weighted + value * weight, total + weight)
        });
    weighted / total.max(f64::MIN_POSITIVE)
}

fn add3(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

fn norm3(value: [f64; 3]) -> f64 {
    value
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt()
}

fn subtract(lhs: sonium_fem::mesh::Point, rhs: sonium_fem::mesh::Point) -> [f64; 3] {
    [lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z]
}

fn cross(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

fn dot(lhs: [f64; 3], rhs: [f64; 3]) -> f64 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

fn scale(value: [f64; 3], factor: f64) -> [f64; 3] {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

fn add(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solid3d_topology_runs_multi_load_with_shared_factorization() {
        let mut study = Solid3DTopologyStudy::introductory();
        study.base.geometry.nx = 2;
        study.base.geometry.ny = 1;
        study.base.geometry.nz = 1;
        study.volume_fraction = 0.90;
        study.iterations = 2;
        study.allowable_stress_pa = 1.0e14;
        let result = run_solid3d_topology(&study).expect("3D topology");

        assert_eq!(result.cases.len(), 2);
        assert!(result.factorization_reused_across_cases);
        assert!(result.factorization_memory_bytes > 0);
        assert!(result.final_compliance_j > 0.0);
        assert!(result.volume_fraction <= study.volume_fraction + 1.0e-10);
        assert!((result.volume_fraction - study.volume_fraction).abs() < 1.0e-8);
        assert_eq!(
            result.final_design_density.len(),
            result.final_density.len()
        );
        assert_eq!(
            result.continuation_beta_history,
            vec![study.projection_beta, study.projection_beta * 2.0]
        );
        assert_eq!(result.final_projection_beta, study.projection_beta * 2.0);
        assert!(result.first_frequency_hz.is_finite());
        assert!(result.first_frequency_hz > 0.0);
        assert!(result.modal_feasible);
        assert!(result.buckling_load_factor.is_none());
        assert!(result.buckling_feasible);
        assert!(result.connected);
        assert!(result.stress_feasible);
        assert!(result.morphology_feasible);
        assert!(result.body_fitted_reanalysis);
        assert!(result.body_fitted_feasible);
        assert!(result.eroded_volume_fraction <= result.volume_fraction + 1.0e-12);
        assert!(result.dilated_volume_fraction >= result.volume_fraction - 1.0e-12);
        assert!(result.cases.iter().all(|case| {
            case.converged
                && case.solver_residual < 1.0e-8
                && case.free_equilibrium_residual_n < 1.0e-6
                && case.force_balance_residual_n < 1.0e-6
                && case.energy_balance_relative_error < 1.0e-8
        }));
        assert!(result.body_fitted_cases.iter().all(|case| {
            case.converged
                && case.solver_residual < 1.0e-8
                && case.free_equilibrium_residual_n < 1.0e-6
                && case.force_balance_residual_n < 1.0e-6
                && case.energy_balance_relative_error < 1.0e-8
        }));
        assert_eq!(
            result.execution.capability_id,
            "sonium-structural.cpu-reference.f64.solid3d-topology.v1"
        );
    }

    #[test]
    fn solid3d_topology_frequency_objective_uses_assembled_eigen_sensitivity() {
        let mut study = Solid3DTopologyStudy::introductory();
        study.base.geometry.nx = 2;
        study.base.geometry.ny = 1;
        study.base.geometry.nz = 1;
        study.load_cases.truncate(1);
        study.volume_fraction = 0.90;
        study.iterations = 1;
        study.continuation_steps = 1;
        study.require_body_fitted_reanalysis = false;
        study.objective = Solid3DTopologyObjective::FirstFrequency;
        let result = run_solid3d_topology(&study).expect("3D frequency objective");

        assert_eq!(result.objective, Solid3DTopologyObjective::FirstFrequency);
        assert_eq!(result.objective_history.len(), 2);
        assert!(result.initial_objective_value > 0.0);
        assert!(result.final_objective_value > 0.0);
        assert!(result.objective_improved);
        assert!(result.objective_gradient_taylor_relative_error < 1.0e-3);
        assert!(result.first_frequency_hz > 0.0);
    }

    #[test]
    fn solid3d_topology_buckling_objective_reports_resolved_sensitivity() {
        let mut study = Solid3DTopologyStudy::introductory();
        study.base.geometry.nx = 2;
        study.base.geometry.ny = 1;
        study.base.geometry.nz = 1;
        study.load_cases = vec![Solid3DMultiLoadCase {
            name: "compressive-objective".to_string(),
            force_n: [-100.0, 0.0, 0.0],
        }];
        study.base.loads.force_n = [-100.0, 0.0, 0.0];
        study.volume_fraction = 0.90;
        study.iterations = 1;
        study.continuation_steps = 1;
        study.require_body_fitted_reanalysis = false;
        study.objective = Solid3DTopologyObjective::BucklingLoadFactor;
        let result = run_solid3d_topology(&study).expect("3D buckling objective");

        assert_eq!(
            result.objective,
            Solid3DTopologyObjective::BucklingLoadFactor
        );
        assert_eq!(result.objective_history.len(), 2);
        assert!(result.initial_objective_value > 0.0);
        assert!(result.final_objective_value > 0.0);
        assert!(result.objective_improved);
        assert!(result.objective_gradient_taylor_relative_error < 5.0e-2);
        assert!(result.buckling_load_factor.is_some_and(|value| value > 0.0));
    }

    #[test]
    fn solid3d_topology_rejects_buckling_objective_without_compression() {
        let mut study = Solid3DTopologyStudy::introductory();
        study.objective = Solid3DTopologyObjective::BucklingLoadFactor;
        assert!(matches!(
            run_solid3d_topology(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_topology.objective",
                ..
            })
        ));
    }

    #[test]
    fn solid3d_filter_projection_gradient_matches_finite_difference() {
        let mut study = Solid3DTopologyStudy::introductory();
        study.base.geometry.nx = 2;
        study.base.geometry.ny = 1;
        study.base.geometry.nz = 1;
        study.load_cases.truncate(1);
        study.volume_fraction = 0.90;
        study.filter_radius_m = 0.05;
        study.continuation_steps = 1;
        study.iterations = 1;

        let mesh = study.base.mesh().expect("3D topology mesh");
        let elements = prepare_elements(&mesh, &study).expect("topology elements");
        let protected: Vec<bool> = elements.iter().map(|element| element.protected).collect();
        let volumes: Vec<f64> = elements
            .iter()
            .map(|element| tetrahedron_volume(&mesh, &element.nodes))
            .collect();
        let filter = DensityFilter::build(&mesh, study.filter_radius_m);
        let mut raw = vec![0.80; elements.len()];
        for (rho, &is_protected) in raw.iter_mut().zip(&protected) {
            if is_protected {
                *rho = 1.0;
            }
        }
        let raw = project_design_to_volume(
            &raw,
            &filter,
            &protected,
            &study,
            &volumes,
            study.projection_beta,
        );
        let physical = physical_density(&raw, &filter, &protected, &study, study.projection_beta);
        let evaluation = evaluate(&mesh, &elements, &study, &physical).expect("nominal solve");
        let filtered = filter.filter_densities(&raw);
        let projection =
            HeavisideProjection::new(study.projection_beta, study.projection_threshold);
        let filtered_gradient: Vec<f64> = evaluation
            .gradient
            .iter()
            .zip(&filtered)
            .zip(&protected)
            .map(|((&gradient, &filtered_density), &is_protected)| {
                if is_protected {
                    0.0
                } else {
                    gradient * projection.project_derivative(filtered_density)
                }
            })
            .collect();
        let raw_gradient = filter.filter_sensitivities_transpose(&filtered_gradient);
        let direction: Vec<f64> = raw
            .iter()
            .enumerate()
            .zip(&protected)
            .map(|((index, _), &is_protected)| {
                if is_protected {
                    0.0
                } else {
                    0.07 * (index as f64 + 1.0)
                }
            })
            .collect();
        let epsilon = 1.0e-6;
        let plus_raw: Vec<f64> = raw
            .iter()
            .zip(&direction)
            .map(|(&rho, &delta)| rho + epsilon * delta)
            .collect();
        let minus_raw: Vec<f64> = raw
            .iter()
            .zip(&direction)
            .map(|(&rho, &delta)| rho - epsilon * delta)
            .collect();
        let plus = evaluate(
            &mesh,
            &elements,
            &study,
            &physical_density(
                &plus_raw,
                &filter,
                &protected,
                &study,
                study.projection_beta,
            ),
        )
        .expect("plus solve");
        let minus = evaluate(
            &mesh,
            &elements,
            &study,
            &physical_density(
                &minus_raw,
                &filter,
                &protected,
                &study,
                study.projection_beta,
            ),
        )
        .expect("minus solve");
        let numerical = (plus.objective_j - minus.objective_j) / (2.0 * epsilon);
        let analytical = raw_gradient
            .iter()
            .zip(&direction)
            .map(|(&gradient, &delta)| gradient * delta)
            .sum::<f64>();
        let relative_error = (numerical - analytical).abs() / numerical.abs().max(1.0);
        assert!(
            relative_error < 1.0e-5,
            "filtered/projected gradient error: numerical={numerical:.6e}, analytical={analytical:.6e}, relative={relative_error:.6e}"
        );
    }

    #[test]
    fn solid3d_compressive_case_reports_linearized_buckling_factor() {
        let mut study = Solid3DTopologyStudy::introductory();
        study.base.geometry.nx = 2;
        study.base.geometry.ny = 1;
        study.base.geometry.nz = 1;
        study.load_cases = vec![Solid3DMultiLoadCase {
            name: "compressive-service".to_string(),
            force_n: [-100.0, 0.0, 0.0],
        }];
        study.base.loads.force_n = [-100.0, 0.0, 0.0];
        study.volume_fraction = 0.90;
        study.iterations = 1;
        study.allowable_stress_pa = 1.0e14;
        let result = run_solid3d_topology(&study).expect("compressive 3D topology");

        assert_eq!(result.buckling_case.as_deref(), Some("compressive-service"));
        assert_eq!(
            result.buckling_model,
            Some(Solid3DBucklingModel::SigmaXxFallbackMixedSign)
        );
        let load_factor = result
            .buckling_load_factor
            .expect("compressive case should have a buckling factor");
        assert!(load_factor.is_finite());
        assert!(load_factor > 0.0);
        assert!(result.buckling_feasible);
    }

    #[test]
    fn solid3d_buckling_selects_full_or_bounded_fallback_stress_model() {
        assert!(negative_stress_is_positive_semidefinite([
            -10.0, -20.0, -30.0, 0.0, 0.0, 0.0
        ]));
        assert!(negative_stress_is_positive_semidefinite([
            -4.0, -4.0, -4.0, 2.0, 2.0, 2.0
        ]));
        assert!(!negative_stress_is_positive_semidefinite([
            -10.0, 20.0, -30.0, 0.0, 0.0, 0.0
        ]));
        assert!(!negative_stress_is_positive_semidefinite([
            -10.0, -10.0, -10.0, 0.0, 0.0, 20.0
        ]));

        assert_eq!(
            buckling_model_for_stress([-10.0, -20.0, -30.0, 0.0, 0.0, 0.0]),
            Solid3DBucklingModel::FullStressTensor
        );
        assert_eq!(
            buckling_model_for_stress([-10.0, 20.0, -30.0, 0.0, 0.0, 0.0]),
            Solid3DBucklingModel::SigmaXxFallbackMixedSign
        );
    }
}
/// Run a bounded coarse/fine 3-D topology convergence check.
pub fn run_solid3d_topology_mesh_convergence(
    study: &Solid3DTopologyStudy,
    tolerance: f64,
) -> Result<Solid3DTopologyConvergenceReport, StructuralMechanicsError> {
    if !tolerance.is_finite() || tolerance <= 0.0 || tolerance >= 1.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "solid3d_topology_convergence.tolerance",
            value: tolerance.to_string(),
        });
    }
    let coarse_resolution = [
        study.base.geometry.nx,
        study.base.geometry.ny,
        study.base.geometry.nz,
    ];
    let refined_resolution = [
        study.base.geometry.nx.checked_mul(2).ok_or(
            StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_topology_convergence.nx",
                value: "refinement overflow".to_string(),
            },
        )?,
        study.base.geometry.ny.checked_mul(2).ok_or(
            StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_topology_convergence.ny",
                value: "refinement overflow".to_string(),
            },
        )?,
        study.base.geometry.nz.checked_mul(2).ok_or(
            StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_topology_convergence.nz",
                value: "refinement overflow".to_string(),
            },
        )?,
    ];
    let coarse_result = run_solid3d_topology(study)?;
    let mut refined_study = study.clone();
    refined_study.base.geometry.nx = refined_resolution[0];
    refined_study.base.geometry.ny = refined_resolution[1];
    refined_study.base.geometry.nz = refined_resolution[2];
    let refined_result = run_solid3d_topology(&refined_study)?;
    let coarse_mesh = study.base.mesh()?;
    let refined_mesh = refined_study.base.mesh()?;
    let coarse_elements = prepare_elements(&coarse_mesh, study)?;
    let refined_elements = prepare_elements(&refined_mesh, &refined_study)?;
    let (rank_pair_count, rank_discordant_pair_count) = projected_rank_pairs(
        &coarse_mesh,
        &coarse_elements,
        &coarse_result.final_density,
        &refined_mesh,
        &refined_elements,
        &refined_result.final_density,
    );
    let rank_discordance_fraction = if rank_pair_count == 0 {
        f64::INFINITY
    } else {
        rank_discordant_pair_count as f64 / rank_pair_count as f64
    };
    let rank_preserved = rank_pair_count > 0 && rank_discordance_fraction <= tolerance;
    let coarse = convergence_level(&coarse_result);
    let refined = convergence_level(&refined_result);
    let objective_relative_change = relative_change(coarse.objective_j, refined.objective_j);
    let stress_relative_change =
        relative_change(coarse.aggregated_stress_pa, refined.aggregated_stress_pa);
    let volume_fraction_relative_change =
        relative_change(coarse.volume_fraction, refined.volume_fraction);
    let metrics = [
        objective_relative_change,
        stress_relative_change,
        volume_fraction_relative_change,
        rank_discordance_fraction,
    ];
    if metrics.iter().any(|value| !value.is_finite()) {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology_convergence.metrics",
            value: metrics.iter().copied().fold(0.0, f64::max),
        });
    }
    let converged = coarse.converged
        && refined.converged
        && coarse.connected
        && refined.connected
        && coarse.stress_feasible
        && refined.stress_feasible
        && coarse.morphology_feasible
        && refined.morphology_feasible
        && coarse.body_fitted_feasible
        && refined.body_fitted_feasible
        && objective_relative_change <= tolerance
        && stress_relative_change <= tolerance
        && volume_fraction_relative_change <= tolerance
        && rank_preserved;
    Ok(Solid3DTopologyConvergenceReport {
        schema_version: SOLID3D_TOPOLOGY_CONVERGENCE_SCHEMA_VERSION,
        coarse_resolution,
        refined_resolution,
        coarse,
        refined,
        tolerance,
        objective_relative_change,
        stress_relative_change,
        volume_fraction_relative_change,
        rank_pair_count,
        rank_discordant_pair_count,
        rank_discordance_fraction,
        rank_preserved,
        converged,
        model_assumption: StructuralModelAssumption::SimpTopology3D,
        execution: StructuralExecutionProvenance::cpu_solid3d_topology_reference(),
    })
}

fn convergence_level(result: &Solid3DTopologyResult) -> Solid3DTopologyConvergenceLevel {
    let cases = result.cases.iter().chain(result.body_fitted_cases.iter());
    let mut max_displacement_m: f64 = 0.0;
    let mut max_solver_residual: f64 = 0.0;
    let mut max_free_equilibrium_residual_n: f64 = 0.0;
    let mut max_force_balance_residual_n: f64 = 0.0;
    let mut max_energy_balance_relative_error: f64 = 0.0;
    for case in cases {
        max_displacement_m = max_displacement_m.max(case.max_displacement_m);
        max_solver_residual = max_solver_residual.max(case.solver_residual);
        max_free_equilibrium_residual_n =
            max_free_equilibrium_residual_n.max(case.free_equilibrium_residual_n);
        max_force_balance_residual_n =
            max_force_balance_residual_n.max(case.force_balance_residual_n);
        max_energy_balance_relative_error =
            max_energy_balance_relative_error.max(case.energy_balance_relative_error);
    }
    Solid3DTopologyConvergenceLevel {
        nodes: result.nodes,
        elements: result.elements,
        dofs: result.dofs,
        iterations: result.iterations,
        objective_j: result.final_objective_value,
        compliance_j: result.final_compliance_j,
        max_displacement_m,
        aggregated_stress_pa: result.aggregated_stress_pa,
        first_frequency_hz: result.first_frequency_hz,
        volume_fraction: result.volume_fraction,
        max_solver_residual,
        max_free_equilibrium_residual_n,
        max_force_balance_residual_n,
        max_energy_balance_relative_error,
        objective_gradient_taylor_relative_error: result.objective_gradient_taylor_relative_error,
        converged: result.converged,
        connected: result.connected,
        stress_feasible: result.stress_feasible,
        morphology_feasible: result.morphology_feasible,
        body_fitted_feasible: result.body_fitted_feasible,
    }
}

fn relative_change(lhs: f64, rhs: f64) -> f64 {
    (lhs - rhs).abs() / lhs.abs().max(rhs.abs()).max(1.0e-30)
}

fn projected_rank_pairs(
    coarse_mesh: &Mesh,
    coarse_elements: &[TopologyElement],
    coarse_density: &[f64],
    refined_mesh: &Mesh,
    refined_elements: &[TopologyElement],
    refined_density: &[f64],
) -> (usize, usize) {
    if coarse_elements.len() != coarse_density.len()
        || refined_elements.len() != refined_density.len()
        || coarse_elements.is_empty()
        || refined_elements.is_empty()
    {
        return (0, 0);
    }
    let projected: Vec<f64> = coarse_elements
        .iter()
        .map(|element| {
            let centroid = element_centroid(coarse_mesh, &element.nodes);
            refined_elements
                .iter()
                .enumerate()
                .min_by(|(_, lhs), (_, rhs)| {
                    squared_distance(centroid, element_centroid(refined_mesh, &lhs.nodes))
                        .total_cmp(&squared_distance(
                            centroid,
                            element_centroid(refined_mesh, &rhs.nodes),
                        ))
                })
                .map(|(index, _)| refined_density[index])
                .unwrap_or(f64::NAN)
        })
        .collect();
    let mut comparable = 0;
    let mut discordant = 0;
    for first in 0..coarse_density.len() {
        for second in (first + 1)..coarse_density.len() {
            let coarse_delta = coarse_density[first] - coarse_density[second];
            let projected_delta = projected[first] - projected[second];
            if coarse_delta.abs() <= 1.0e-9 || projected_delta.abs() <= 1.0e-9 {
                continue;
            }
            comparable += 1;
            if coarse_delta.signum() != projected_delta.signum() {
                discordant += 1;
            }
        }
    }
    (comparable, discordant)
}

fn squared_distance(lhs: [f64; 3], rhs: [f64; 3]) -> f64 {
    (lhs[0] - rhs[0]).powi(2) + (lhs[1] - rhs[1]).powi(2) + (lhs[2] - rhs[2]).powi(2)
}
