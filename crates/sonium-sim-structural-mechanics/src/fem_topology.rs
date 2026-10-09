//! Deterministic 2D FEM topology optimization reference workflow.
//!
//! This module is intentionally separate from the legacy one-dimensional
//! topology API.  It assembles density-scaled P1 triangular elasticity and
//! consistent mass matrices, solves the constrained displacement problem,
//! computes the compliance adjoint sensitivity from element energies, and
//! performs filtered/projected volume-constrained updates.  The final binary
//! density is reanalysed on an active-element body-fitted submesh.  An
//! explicitly compressive x-load additionally assembles a P1 initial-stress
//! generalized eigenproblem and reports its positive load factor.

use crate::{
    StructuralExecutionProvenance, StructuralMechanicsError, StructuralModelAssumption,
    TopologyInterpolation,
};
use serde::{Deserialize, Serialize};
use sonium_fem::boundary::generic::{FemSystem, try_apply_homogeneous_dirichlet_dofs};
use sonium_fem::mesh::{ElementType, Mesh, rectangular_mesh_triangles};
use sonium_fem::optimization::{DensityFilter, HeavisideProjection};
use sonium_fem::physics::solid::eigenmodes::build_linear_buckling_problem;
use sonium_fem::physics::solid::elasticity::IsotropicMaterial;
use sonium_solvers::{
    LinearSolveConfig, LinearSolverType, LobpcgConfig, MatrixClass, Ordering, SparseDirectConfig,
    SparseDirectMethod, TripletMatrix, factorize_sparse, lobpcg_generalized, solve_real,
};

const DOFS_PER_NODE: usize = 2;
const VALIDATION_TOLERANCE: f64 = 1.0e-7;
const DEFAULT_MINIMUM_STIFFNESS: f64 = 1.0e-3;

/// Structured rectangular design domain for the FEM topology study.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FemTopology2DGeometry {
    pub length_m: f64,
    pub height_m: f64,
    pub thickness_m: f64,
    pub nx: usize,
    pub ny: usize,
}

impl Default for FemTopology2DGeometry {
    fn default() -> Self {
        Self {
            length_m: 0.18,
            height_m: 0.036,
            thickness_m: 0.006,
            nx: 18,
            ny: 6,
        }
    }
}

impl FemTopology2DGeometry {
    fn mesh(self) -> Mesh {
        rectangular_mesh_triangles(
            0.0,
            self.length_m,
            -0.5 * self.height_m,
            0.5 * self.height_m,
            self.nx,
            self.ny,
        )
    }
}

/// Isotropic plane-stress material used by the FEM topology study.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FemTopology2DMaterial {
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
    pub density_kg_m3: f64,
}

/// Objective optimized by the assembled 2D FEM topology workflow.
///
/// Compliance remains the default.  The frequency and buckling objectives
/// maximize the corresponding assembled eigenvalue-derived quantity while
/// retaining the same volume, filtering, projection, connectivity, stress,
/// and body-fitted reanalysis gates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FemTopology2DObjective {
    /// Minimize static strain-energy compliance.
    #[default]
    Compliance,
    /// Maximize the first assembled mass/stiffness eigenfrequency.
    FirstFrequency,
    /// Maximize the assembled linearized initial-stress buckling load factor.
    BucklingLoadFactor,
}

impl Default for FemTopology2DMaterial {
    fn default() -> Self {
        Self {
            youngs_modulus_pa: 70.0e9,
            poisson_ratio: 0.3,
            density_kg_m3: 2_700.0,
        }
    }
}

/// Inputs for a real P1 triangular FEM topology optimization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FemTopology2DStudy {
    pub name: String,
    pub geometry: FemTopology2DGeometry,
    pub material: FemTopology2DMaterial,
    /// Objective used during density updates; defaults to compliance for
    /// backwards-compatible study JSON.
    #[serde(default)]
    pub objective: FemTopology2DObjective,
    /// Total `[Fx, Fy]` traction resultant applied to the right edge.
    pub edge_force_n: [f64; 2],
    pub volume_fraction: f64,
    pub minimum_density: f64,
    pub interpolation: TopologyInterpolation,
    pub interpolation_parameter: f64,
    pub filter_radius_m: f64,
    pub projection_beta: f64,
    pub projection_threshold: f64,
    pub continuation_steps: usize,
    pub iterations_per_stage: usize,
    pub move_limit: f64,
    pub convergence_tolerance: f64,
    pub stress_aggregation_p: f64,
    pub allowable_stress_pa: f64,
    /// Minimum load factor for an optional linearized compression/buckling gate.
    pub minimum_buckling_load_factor: f64,
    pub minimum_feature_cells: usize,
    pub maximum_overhang_slope: f64,
}

impl Default for FemTopology2DStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl FemTopology2DStudy {
    /// Small cantilever-like FEM case suitable for deterministic regression.
    pub fn introductory() -> Self {
        Self {
            name: "fem-2d-simp-topology-reference".to_string(),
            geometry: FemTopology2DGeometry::default(),
            material: FemTopology2DMaterial::default(),
            objective: FemTopology2DObjective::Compliance,
            edge_force_n: [0.0, -120.0],
            volume_fraction: 0.55,
            minimum_density: 0.05,
            interpolation: TopologyInterpolation::Simp,
            interpolation_parameter: 3.0,
            filter_radius_m: 0.006,
            projection_beta: 2.0,
            projection_threshold: 0.5,
            continuation_steps: 2,
            iterations_per_stage: 8,
            move_limit: 0.12,
            convergence_tolerance: 2.0e-4,
            stress_aggregation_p: 8.0,
            allowable_stress_pa: 2.0e8,
            minimum_buckling_load_factor: 1.0,
            minimum_feature_cells: 1,
            maximum_overhang_slope: 10.0,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("fem_topology.length_m", self.geometry.length_m),
            ("fem_topology.height_m", self.geometry.height_m),
            ("fem_topology.thickness_m", self.geometry.thickness_m),
            (
                "fem_topology.youngs_modulus_pa",
                self.material.youngs_modulus_pa,
            ),
            ("fem_topology.density_kg_m3", self.material.density_kg_m3),
            (
                "fem_topology.interpolation_parameter",
                self.interpolation_parameter,
            ),
            ("fem_topology.filter_radius_m", self.filter_radius_m),
            ("fem_topology.projection_beta", self.projection_beta),
            (
                "fem_topology.stress_aggregation_p",
                self.stress_aggregation_p,
            ),
            ("fem_topology.allowable_stress_pa", self.allowable_stress_pa),
            (
                "fem_topology.minimum_buckling_load_factor",
                self.minimum_buckling_load_factor,
            ),
            ("fem_topology.move_limit", self.move_limit),
            (
                "fem_topology.convergence_tolerance",
                self.convergence_tolerance,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if !(-1.0 < self.material.poisson_ratio && self.material.poisson_ratio < 0.5) {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "fem_topology.poisson_ratio",
                value: self.material.poisson_ratio.to_string(),
            });
        }
        if self.edge_force_n.iter().any(|value| !value.is_finite())
            || self.edge_force_n.iter().all(|value| *value == 0.0)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "fem_topology.edge_force_n",
                value: format!("{:?} (must be finite and non-zero)", self.edge_force_n),
            });
        }
        if self.objective == FemTopology2DObjective::BucklingLoadFactor
            && self.edge_force_n[0] >= 0.0
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "fem_topology.objective",
                value: "BucklingLoadFactor requires a compressive x-directed load".to_string(),
            });
        }
        if self.geometry.nx == 0
            || self.geometry.ny == 0
            || self.continuation_steps == 0
            || self.iterations_per_stage == 0
            || !(0.0 < self.volume_fraction && self.volume_fraction <= 1.0)
            || !(0.0 < self.minimum_density && self.minimum_density < self.volume_fraction)
            || !(0.0 < self.projection_threshold && self.projection_threshold < 1.0)
            || self.move_limit > 1.0
            || self.minimum_feature_cells == 0
            || !self.maximum_overhang_slope.is_finite()
            || self.maximum_overhang_slope <= 0.0
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "fem_topology.discretization_or_bounds",
                value: "invalid topology bounds".to_string(),
            });
        }
        Ok(())
    }
}

/// Machine-readable result from the real 2D FEM topology path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FemTopology2DResult {
    pub study: FemTopology2DStudy,
    #[serde(default)]
    pub objective: FemTopology2DObjective,
    #[serde(default)]
    pub initial_objective_value: f64,
    #[serde(default)]
    pub final_objective_value: f64,
    #[serde(default)]
    pub objective_history: Vec<f64>,
    #[serde(default)]
    pub objective_improved: bool,
    pub nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    pub iterations: usize,
    pub converged: bool,
    pub initial_compliance_j: f64,
    pub final_compliance_j: f64,
    pub projected_reanalysis_compliance_j: f64,
    pub body_fitted_reanalysis_compliance_j: f64,
    pub projected_reanalysis_relative_error: f64,
    pub body_fitted_reanalysis_relative_error: f64,
    pub final_volume_fraction: f64,
    pub max_displacement_m: f64,
    pub aggregated_stress_pa: f64,
    pub first_frequency_hz: f64,
    pub solver_residual: f64,
    pub free_equilibrium_residual_n: f64,
    pub force_balance_residual_n: f64,
    pub energy_balance_relative_error: f64,
    pub gradient_taylor_relative_error: f64,
    pub connected: bool,
    pub minimum_feature_feasible: bool,
    pub overhang_feasible: bool,
    pub stress_feasible: bool,
    /// Lowest positive linearized load factor under the recovered compressive
    /// stress field, when the load case contains compression.
    pub buckling_load_factor: Option<f64>,
    pub buckling_feasible: bool,
    pub reanalysis_feasible: bool,
    /// Physical density field at the start of the optimization, retained for
    /// before/after topology evidence and independent design comparisons.
    #[serde(default)]
    pub initial_physical_density: Vec<f64>,
    pub final_physical_density: Vec<f64>,
    pub projected_density: Vec<f64>,
    pub displacement: Vec<[f64; 2]>,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// A named load case verified on one accepted FEM topology design.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FemTopology2DLoadCase {
    pub name: String,
    pub edge_force_n: [f64; 2],
}

/// Multi-load verification configuration for a primary topology design.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FemTopology2DMultiLoadStudy {
    pub topology: FemTopology2DStudy,
    pub load_cases: Vec<FemTopology2DLoadCase>,
}

impl Default for FemTopology2DMultiLoadStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl FemTopology2DMultiLoadStudy {
    pub fn introductory() -> Self {
        let topology = FemTopology2DStudy::introductory();
        Self {
            topology,
            load_cases: vec![
                FemTopology2DLoadCase {
                    name: "service-down".to_string(),
                    edge_force_n: [0.0, -120.0],
                },
                FemTopology2DLoadCase {
                    name: "service-reversal".to_string(),
                    edge_force_n: [0.0, 80.0],
                },
            ],
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        self.topology.validate()?;
        if self.load_cases.is_empty() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "fem_topology_multi_load.load_cases",
                value: "empty".to_string(),
            });
        }
        for case in &self.load_cases {
            if case.name.trim().is_empty()
                || case.edge_force_n.iter().any(|value| !value.is_finite())
                || case
                    .edge_force_n
                    .iter()
                    .all(|value| value.abs() <= f64::EPSILON)
            {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "fem_topology_multi_load.load_cases",
                    value: case.name.clone(),
                });
            }
        }
        Ok(())
    }
}

/// Static response and validation gates for one topology load case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FemTopology2DLoadCaseResult {
    pub name: String,
    pub edge_force_n: [f64; 2],
    pub compliance_j: f64,
    pub max_displacement_m: f64,
    pub max_von_mises_pa: f64,
    /// Density-aware p-norm stress summary for this load case.
    #[serde(default)]
    pub aggregated_stress_pa: f64,
    /// Whether the case-level aggregated stress satisfies the topology limit.
    #[serde(default)]
    pub stress_feasible: bool,
    pub solver_residual: f64,
    pub free_equilibrium_residual_n: f64,
    pub force_balance_residual_n: f64,
    pub energy_balance_relative_error: f64,
    pub converged: bool,
}

/// Shared-factorization multi-load FEM topology verification result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FemTopology2DMultiLoadResult {
    pub study: FemTopology2DMultiLoadStudy,
    pub topology: FemTopology2DResult,
    pub cases: Vec<FemTopology2DLoadCaseResult>,
    pub critical_case: String,
    pub max_compliance_j: f64,
    pub max_displacement_m: f64,
    pub max_von_mises_pa: f64,
    #[serde(default)]
    pub aggregated_stress_pa: f64,
    #[serde(default)]
    pub stress_feasible: bool,
    pub factorization_memory_bytes: usize,
    pub factorization_reused: bool,
    pub converged: bool,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Configuration for topology optimization against a shared worst-case load
/// envelope.  Unlike [`FemTopology2DMultiLoadStudy`], this study includes all
/// load cases in every density update; it does not optimize one primary case
/// and merely verify the others afterwards.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FemTopology2DEnvelopeStudy {
    pub topology: FemTopology2DStudy,
    pub load_cases: Vec<FemTopology2DLoadCase>,
}

impl Default for FemTopology2DEnvelopeStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl FemTopology2DEnvelopeStudy {
    pub fn introductory() -> Self {
        let mut topology = FemTopology2DStudy::introductory();
        let load_cases = vec![
            FemTopology2DLoadCase {
                name: "envelope-down".to_string(),
                edge_force_n: [0.0, -120.0],
            },
            FemTopology2DLoadCase {
                name: "envelope-reversal".to_string(),
                edge_force_n: [0.0, 80.0],
            },
        ];
        topology.edge_force_n = load_cases[0].edge_force_n;
        Self {
            topology,
            load_cases,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        if self.load_cases.is_empty() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "fem_topology_envelope.load_cases",
                value: "empty".to_string(),
            });
        }
        // The topology study historically owns one load for the single-load
        // API.  Validate the common geometry/material controls against the
        // first envelope case without requiring callers to duplicate it.
        let mut topology = self.topology.clone();
        topology.edge_force_n = self.load_cases[0].edge_force_n;
        topology.validate()?;
        let mut names = std::collections::HashSet::with_capacity(self.load_cases.len());
        for case in &self.load_cases {
            if case.name.trim().is_empty()
                || !names.insert(case.name.clone())
                || case.edge_force_n.iter().any(|value| !value.is_finite())
                || case
                    .edge_force_n
                    .iter()
                    .all(|value| value.abs() <= f64::EPSILON)
            {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "fem_topology_envelope.load_cases",
                    value: case.name.clone(),
                });
            }
        }
        Ok(())
    }
}

/// Result of a topology optimization whose objective is the exact maximum
/// compliance over the supplied load cases.  At a tie, the optimizer uses the
/// average of the active-case compliance gradients as a valid subgradient.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FemTopology2DEnvelopeResult {
    pub study: FemTopology2DEnvelopeStudy,
    pub initial_case_compliances_j: Vec<f64>,
    pub final_case_compliances_j: Vec<f64>,
    pub projected_reanalysis_case_compliances_j: Vec<f64>,
    pub body_fitted_reanalysis_case_compliances_j: Vec<f64>,
    pub initial_max_compliance_j: f64,
    pub final_max_compliance_j: f64,
    pub projected_reanalysis_max_compliance_j: f64,
    pub body_fitted_reanalysis_max_compliance_j: f64,
    pub critical_case: String,
    #[serde(default)]
    pub aggregated_stress_pa: f64,
    #[serde(default)]
    pub stress_feasible: bool,
    pub final_physical_density: Vec<f64>,
    pub projected_density: Vec<f64>,
    pub final_volume_fraction: f64,
    pub iterations: usize,
    pub converged: bool,
    pub history: Vec<f64>,
    pub gradient_taylor_relative_error: f64,
    pub connected: bool,
    pub reanalysis_feasible: bool,
    pub factorization_memory_bytes: usize,
    pub factorization_reused_across_cases: bool,
    pub cases: Vec<FemTopology2DLoadCaseResult>,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Multiplicative uncertainty applied to one real FEM topology reanalysis.
///
/// The uncertainty is deliberately explicit and dimensionless: load scales
/// the named edge tractions, Young's-modulus scales the assembled stiffness,
/// and thickness scales the plane-stress section thickness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FemTopology2DUncertaintyCase {
    pub name: String,
    pub load_scale: f64,
    pub youngs_modulus_scale: f64,
    pub thickness_scale: f64,
}

/// Robust morphology and uncertainty study over the assembled 2D FEM path.
///
/// The envelope is optimized first with the real P1 FEM optimizer.  The
/// accepted physical density is then reanalysed after graph-based element
/// erosion and dilation for every named material/load uncertainty case.  The
/// result is evidence for bounded manufacturing robustness, not a claim of
/// general 3D robust topology optimization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FemTopology2DRobustStudy {
    pub envelope: FemTopology2DEnvelopeStudy,
    pub uncertainty_cases: Vec<FemTopology2DUncertaintyCase>,
    pub erosion_radius_cells: usize,
    pub dilation_radius_cells: usize,
    pub allowable_compliance_j: f64,
}

impl Default for FemTopology2DRobustStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl FemTopology2DRobustStudy {
    pub fn introductory() -> Self {
        Self {
            envelope: FemTopology2DEnvelopeStudy::introductory(),
            uncertainty_cases: vec![
                FemTopology2DUncertaintyCase {
                    name: "nominal".to_string(),
                    load_scale: 1.0,
                    youngs_modulus_scale: 1.0,
                    thickness_scale: 1.0,
                },
                FemTopology2DUncertaintyCase {
                    name: "weak-material-overload".to_string(),
                    load_scale: 1.10,
                    youngs_modulus_scale: 0.90,
                    thickness_scale: 0.95,
                },
            ],
            erosion_radius_cells: 1,
            dilation_radius_cells: 1,
            allowable_compliance_j: 1.0,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        self.envelope.validate()?;
        if self.uncertainty_cases.is_empty()
            || !self.allowable_compliance_j.is_finite()
            || self.allowable_compliance_j <= 0.0
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "fem_topology_robust.uncertainty_or_allowable_compliance",
                value: "must contain uncertainty cases and a positive finite compliance limit"
                    .to_string(),
            });
        }
        let mut names = std::collections::HashSet::with_capacity(self.uncertainty_cases.len());
        for case in &self.uncertainty_cases {
            if case.name.trim().is_empty()
                || !names.insert(case.name.clone())
                || [
                    case.load_scale,
                    case.youngs_modulus_scale,
                    case.thickness_scale,
                ]
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "fem_topology_robust.uncertainty_cases",
                    value: case.name.clone(),
                });
            }
        }
        Ok(())
    }
}

/// One real FEM reanalysis in a morphology/uncertainty combination.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FemTopology2DRobustCaseResult {
    pub morphology: String,
    pub uncertainty_case: String,
    pub load_case: String,
    pub compliance_j: f64,
    pub max_displacement_m: f64,
    #[serde(default)]
    pub aggregated_stress_pa: f64,
    #[serde(default)]
    pub stress_feasible: bool,
    pub solver_residual: f64,
    pub free_equilibrium_residual_n: f64,
    pub force_balance_residual_n: f64,
    pub energy_balance_relative_error: f64,
    pub converged: bool,
    pub feasible: bool,
}

/// Result of the real 2D robust morphology/uncertainty topology workflow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FemTopology2DRobustResult {
    pub study: FemTopology2DRobustStudy,
    pub nominal: FemTopology2DEnvelopeResult,
    pub nominal_density: Vec<f64>,
    pub eroded_density: Vec<f64>,
    pub dilated_density: Vec<f64>,
    pub nominal_volume_fraction: f64,
    pub eroded_volume_fraction: f64,
    pub dilated_volume_fraction: f64,
    pub cases: Vec<FemTopology2DRobustCaseResult>,
    pub worst_case_compliance_j: f64,
    pub worst_case: String,
    #[serde(default)]
    pub worst_case_aggregated_stress_pa: f64,
    #[serde(default)]
    pub stress_feasible: bool,
    pub erosion_connected: bool,
    pub dilation_connected: bool,
    pub morphology_feasible: bool,
    pub uncertainty_feasible: bool,
    pub robust_feasible: bool,
    pub factorization_reused_within_scenario: bool,
    pub gradient_taylor_relative_error: f64,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

#[derive(Debug, Clone)]
struct TopologyElement {
    nodes: [usize; 3],
    gradients: [[f64; 2]; 3],
    area_m2: f64,
    local_stiffness: [f64; 36],
    centroid: [f64; 2],
}

#[derive(Debug, Clone)]
struct TopologyEvaluation {
    projected_density: Vec<f64>,
    physical_density: Vec<f64>,
    stiffness_ratios: Vec<f64>,
    displacement: Vec<f64>,
    objective_j: f64,
    element_energy_j: Vec<f64>,
    element_von_mises_pa: Vec<f64>,
    element_stress: Vec<[f64; 3]>,
    solver_residual: f64,
    free_equilibrium_residual_n: f64,
    force_balance_residual_n: f64,
    energy_balance_relative_error: f64,
    max_displacement_m: f64,
}

#[derive(Debug, Clone)]
struct EnvelopeEvaluation {
    projected_density: Vec<f64>,
    cases: Vec<FemTopology2DLoadCaseResult>,
    objective_j: f64,
    gradient: Vec<f64>,
    factorization_memory_bytes: usize,
}

struct EnvelopeContext<'a> {
    mesh: &'a Mesh,
    elements: &'a [TopologyElement],
    topology: &'a FemTopology2DStudy,
    load_cases: &'a [FemTopology2DLoadCase],
    filter: &'a DensityFilter,
    protected: &'a [bool],
}

/// Run the genuine rectangular P1 FEM topology reference path.
pub fn run_fem_topology_2d(
    study: &FemTopology2DStudy,
) -> Result<FemTopology2DResult, StructuralMechanicsError> {
    study.validate()?;
    let mesh = study.geometry.mesh();
    let elements = prepare_elements(&mesh, study)?;
    let volumes: Vec<f64> = elements
        .iter()
        .map(|element| element.area_m2 * study.geometry.thickness_m)
        .collect();
    let protected = protected_elements(&mesh, &elements, study.geometry);
    let protected_volume_fraction = weighted_mean_bool(&protected, &volumes);
    if protected_volume_fraction > study.volume_fraction + 1.0e-12 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "fem_topology.volume_fraction",
            value: format!(
                "target {} is below protected boundary fraction {}",
                study.volume_fraction, protected_volume_fraction
            ),
        });
    }
    let filter = DensityFilter::build(&mesh, study.filter_radius_m);
    let mut raw_density = vec![study.volume_fraction; elements.len()];
    for (rho, &is_protected) in raw_density.iter_mut().zip(&protected) {
        if is_protected {
            *rho = 1.0;
        }
    }
    let initial_projection = project_to_volume(
        &raw_density,
        &filter,
        study,
        &volumes,
        &protected,
        study.projection_beta,
    );
    raw_density = initial_projection.raw_density;
    let initial = evaluate_raw(
        &mesh,
        &elements,
        study,
        &filter,
        &protected,
        &raw_density,
        study.projection_beta,
    )?;
    let initial_compliance_j = initial.objective_j;
    let initial_objective_value = topology_objective_value(&mesh, &elements, study, &initial)?;
    let mut history = vec![initial_objective_value];
    let mut iterations = 0;
    let mut converged = false;

    for stage in 0..study.continuation_steps {
        let beta = study.projection_beta * 2.0_f64.powi(stage as i32);
        for _ in 0..study.iterations_per_stage {
            let current = evaluate_raw(
                &mesh,
                &elements,
                study,
                &filter,
                &protected,
                &raw_density,
                beta,
            )?;
            let current_objective_value =
                topology_objective_value(&mesh, &elements, study, &current)?;
            let projection_derivative = HeavisideProjection::new(beta, study.projection_threshold);
            let objective_derivative =
                topology_objective_derivative(&mesh, &elements, study, &current, &protected)?;
            let maximize = matches!(
                study.objective,
                FemTopology2DObjective::FirstFrequency | FemTopology2DObjective::BucklingLoadFactor
            );
            let projected_gradient: Vec<f64> = objective_derivative
                .into_iter()
                .map(|derivative| if maximize { -derivative } else { derivative })
                .collect();
            let filtered_gradient = projected_gradient
                .iter()
                .zip(filter.filter_densities(&raw_density))
                .map(|(&gradient, filtered)| {
                    gradient * projection_derivative.project_derivative(filtered)
                })
                .collect::<Vec<_>>();
            let raw_gradient = filter.filter_sensitivities_transpose(&filtered_gradient);
            let max_gradient = raw_gradient
                .iter()
                .map(|value| value.abs())
                .fold(0.0, f64::max)
                .max(1.0e-30);
            let candidate: Vec<f64> = raw_density
                .iter()
                .zip(&raw_gradient)
                .zip(&protected)
                .map(|((&rho, &gradient), &is_protected)| {
                    if is_protected {
                        1.0
                    } else {
                        (rho - study.move_limit * gradient / max_gradient)
                            .clamp(study.minimum_density, 1.0)
                    }
                })
                .collect();
            let projected_candidate =
                project_to_volume(&candidate, &filter, study, &volumes, &protected, beta);
            let next = evaluate_raw(
                &mesh,
                &elements,
                study,
                &filter,
                &protected,
                &projected_candidate.raw_density,
                beta,
            )?;
            let next_objective_value = topology_objective_value(&mesh, &elements, study, &next)?;
            let max_change = raw_density
                .iter()
                .zip(&projected_candidate.raw_density)
                .map(|(old, new)| (old - new).abs())
                .fold(0.0, f64::max);
            let relative_objective_change = (current_objective_value - next_objective_value).abs()
                / current_objective_value.abs().max(1.0);
            raw_density = projected_candidate.raw_density;
            history.push(next_objective_value);
            iterations += 1;
            if max_change <= study.convergence_tolerance
                && relative_objective_change <= study.convergence_tolerance
            {
                converged = true;
                break;
            }
        }
    }

    let final_beta =
        study.projection_beta * 2.0_f64.powi((study.continuation_steps.saturating_sub(1)) as i32);
    let final_eval = evaluate_raw(
        &mesh,
        &elements,
        study,
        &filter,
        &protected,
        &raw_density,
        final_beta,
    )?;
    let binary_density: Vec<f64> = final_eval
        .projected_density
        .iter()
        .zip(&protected)
        .map(|(&rho, &is_protected)| {
            if is_protected || rho >= study.projection_threshold {
                1.0
            } else {
                study.minimum_density
            }
        })
        .collect();
    let buckling_load_factor = linearized_buckling_load_factor(
        &mesh,
        &elements,
        study,
        &final_eval.stiffness_ratios,
        &final_eval.element_stress,
    )?;
    let buckling_feasible = buckling_load_factor
        .map(|factor| factor >= study.minimum_buckling_load_factor)
        .unwrap_or(true);
    let connected = connected_design(&elements, &binary_density, &protected);
    if !connected {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "fem_topology.connectivity",
            value: 1.0,
        });
    }
    let projected = evaluate_physical(
        &mesh,
        &elements,
        study,
        &binary_density
            .iter()
            .map(|&rho| stiffness_ratio(rho, study))
            .collect::<Vec<_>>(),
    )?;
    let active_mesh = body_fitted_mesh(&mesh, &elements, &binary_density)?;
    let active_elements = prepare_elements(&active_mesh, study)?;
    let body_fitted = evaluate_physical(
        &active_mesh,
        &active_elements,
        study,
        &vec![1.0; active_elements.len()],
    )?;
    let projected_reanalysis_relative_error =
        relative_error(projected.objective_j, final_eval.objective_j);
    let body_fitted_reanalysis_relative_error =
        relative_error(body_fitted.objective_j, projected.objective_j);
    let final_objective_value = topology_objective_value(&mesh, &elements, study, &final_eval)?;
    let objective_improved = match study.objective {
        FemTopology2DObjective::Compliance => {
            final_objective_value <= initial_objective_value * (1.0 + 1.0e-6)
        }
        FemTopology2DObjective::FirstFrequency | FemTopology2DObjective::BucklingLoadFactor => {
            final_objective_value >= initial_objective_value * (1.0 - 1.0e-6)
        }
    };
    let final_objective_derivative =
        topology_objective_derivative(&mesh, &elements, study, &final_eval, &protected)?;
    let gradient_taylor_relative_error = objective_gradient_taylor_error(
        &mesh,
        &elements,
        study,
        &final_eval,
        &final_objective_derivative,
    )?;
    let first_frequency_hz = first_frequency(
        &mesh,
        &elements,
        study,
        &final_eval.stiffness_ratios,
        &final_eval.physical_density,
    )?;
    let minimum_feature_feasible =
        minimum_feature_ok(&elements, &binary_density, study.minimum_feature_cells);
    let maximum_overhang = overhang_slope(&elements, &binary_density, &protected, study.geometry);
    let aggregated_stress_pa =
        aggregate_stress(&final_eval.element_von_mises_pa, study.stress_aggregation_p);
    let stress_feasible = aggregated_stress_pa <= study.allowable_stress_pa;
    let reanalysis_feasible = projected_reanalysis_relative_error <= 0.25
        && body_fitted_reanalysis_relative_error <= 0.25;
    Ok(FemTopology2DResult {
        study: study.clone(),
        objective: study.objective,
        initial_objective_value,
        final_objective_value,
        objective_history: history,
        objective_improved,
        nodes: mesh.nodes.len(),
        elements: mesh.elements.len(),
        dofs: mesh.nodes.len() * DOFS_PER_NODE,
        iterations,
        converged,
        initial_compliance_j,
        final_compliance_j: final_eval.objective_j,
        projected_reanalysis_compliance_j: projected.objective_j,
        body_fitted_reanalysis_compliance_j: body_fitted.objective_j,
        projected_reanalysis_relative_error,
        body_fitted_reanalysis_relative_error,
        final_volume_fraction: weighted_mean_f64(&final_eval.projected_density, &volumes),
        max_displacement_m: final_eval.max_displacement_m,
        aggregated_stress_pa,
        first_frequency_hz,
        solver_residual: final_eval.solver_residual,
        free_equilibrium_residual_n: final_eval.free_equilibrium_residual_n,
        force_balance_residual_n: final_eval.force_balance_residual_n,
        energy_balance_relative_error: final_eval.energy_balance_relative_error,
        gradient_taylor_relative_error,
        connected,
        minimum_feature_feasible,
        overhang_feasible: maximum_overhang <= study.maximum_overhang_slope,
        stress_feasible,
        buckling_load_factor,
        buckling_feasible,
        reanalysis_feasible,
        initial_physical_density: initial.physical_density,
        final_physical_density: final_eval.physical_density,
        projected_density: final_eval.projected_density,
        displacement: final_eval
            .displacement
            .as_chunks::<DOFS_PER_NODE>()
            .0
            .iter()
            .map(|chunk| [chunk[0], chunk[1]])
            .collect(),
        model_assumption: StructuralModelAssumption::SimpTopology2D,
        execution: StructuralExecutionProvenance::cpu_fem_topology_2d_reference(),
    })
}

/// Optimize one primary load case, then verify named load cases with one
/// constrained sparse factorization of the accepted topology stiffness.
pub fn run_fem_topology_2d_multi_load(
    study: &FemTopology2DMultiLoadStudy,
) -> Result<FemTopology2DMultiLoadResult, StructuralMechanicsError> {
    study.validate()?;
    let topology = run_fem_topology_2d(&study.topology)?;
    let mesh = study.topology.geometry.mesh();
    let elements = prepare_elements(&mesh, &study.topology)?;
    let ratios = topology
        .final_physical_density
        .iter()
        .map(|&density| stiffness_ratio(density, &study.topology))
        .collect::<Vec<_>>();
    let dofs = mesh.nodes.len() * DOFS_PER_NODE;
    let stiffness = assemble_global_stiffness(&elements, &ratios, mesh.nodes.len());
    let unconstrained_csr = stiffness.to_csr();
    let fixed_dofs = boundary_dofs_by_coordinate(&mesh, true);
    let mut zero_system = FemSystem::new(stiffness.clone(), vec![0.0; dofs]);
    try_apply_homogeneous_dirichlet_dofs(&mut zero_system, &fixed_dofs).map_err(|error| {
        StructuralMechanicsError::SolverFailure {
            stage: "fem topology multi-load boundary",
            message: error.to_string(),
        }
    })?;
    let constrained_csr = zero_system.matrix.to_csr();
    let factorization = factorize_sparse(
        &constrained_csr,
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..SparseDirectConfig::default()
        },
        None,
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "fem topology multi-load factorization",
        message: error.to_string(),
    })?;
    let factorization_memory_bytes = factorization.factor_memory_bytes();
    let fixed = fixed_dofs
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    // Cost-aware gate: DOFs × cases against the machine-adaptive grain.
    let cases = crate::parallel_collect_results(
        &study.load_cases,
        dofs.saturating_mul(study.load_cases.len()),
        crate::STRUCTURAL_SOLVE_WORK_PER_THREAD,
        |load_case| -> Result<_, StructuralMechanicsError> {
            let mut rhs = vec![0.0; dofs];
            assemble_right_edge_load(&mesh, load_case.edge_force_n, &mut rhs);
            let mut system = FemSystem::new(stiffness.clone(), rhs.clone());
            try_apply_homogeneous_dirichlet_dofs(&mut system, &fixed_dofs).map_err(|error| {
                StructuralMechanicsError::SolverFailure {
                    stage: "fem topology multi-load case boundary",
                    message: error.to_string(),
                }
            })?;
            let solution = factorization.solve(&system.rhs).map_err(|error| {
                StructuralMechanicsError::SolverFailure {
                    stage: "fem topology multi-load solve",
                    message: error.to_string(),
                }
            })?;
            let full_residual = csr_residual_vector(&unconstrained_csr, &solution, &rhs);
            let free_equilibrium_residual_n = l2_norm(
                &full_residual
                    .iter()
                    .enumerate()
                    .filter_map(|(index, value)| (!fixed.contains(&index)).then_some(*value))
                    .collect::<Vec<_>>(),
            );
            let mut reaction = [0.0; 2];
            for &dof in &fixed_dofs {
                reaction[dof % DOFS_PER_NODE] += full_residual[dof];
            }
            let applied = [
                rhs.iter().step_by(DOFS_PER_NODE).sum::<f64>(),
                rhs.iter().skip(1).step_by(DOFS_PER_NODE).sum::<f64>(),
            ];
            let force_balance_residual_n =
                ((reaction[0] + applied[0]).powi(2) + (reaction[1] + applied[1]).powi(2)).sqrt();
            let internal = csr_matvec(&unconstrained_csr, &solution);
            let compliance_j = dot(&rhs, &solution);
            let internal_energy = 0.5 * dot(&solution, &internal);
            let rhs_energy = 0.5 * compliance_j;
            let energy_balance_relative_error =
                (internal_energy - rhs_energy).abs() / rhs_energy.abs().max(1.0e-30);
            let solver_residual = csr_residual(&system.matrix.to_csr(), &solution, &system.rhs);
            let max_displacement_m = solution
                .as_chunks::<DOFS_PER_NODE>()
                .0
                .iter()
                .map(|chunk| chunk[0].hypot(chunk[1]))
                .fold(0.0, f64::max);
            let element_von_mises_pa = elements
                .iter()
                .map(|element| {
                    von_mises(element_stress(
                        element,
                        &local_displacement(element, &solution),
                        &study.topology,
                    ))
                })
                .collect::<Vec<_>>();
            let max_von_mises_pa = element_von_mises_pa.iter().copied().fold(0.0, f64::max);
            let aggregated_stress_pa =
                aggregate_stress(&element_von_mises_pa, study.topology.stress_aggregation_p);
            let stress_feasible = aggregated_stress_pa <= study.topology.allowable_stress_pa;
            let converged = solver_residual.is_finite()
                && free_equilibrium_residual_n.is_finite()
                && force_balance_residual_n.is_finite()
                && energy_balance_relative_error.is_finite()
                && solver_residual <= VALIDATION_TOLERANCE
                && free_equilibrium_residual_n <= VALIDATION_TOLERANCE
                && force_balance_residual_n <= VALIDATION_TOLERANCE
                && energy_balance_relative_error <= VALIDATION_TOLERANCE;
            Ok(FemTopology2DLoadCaseResult {
                name: load_case.name.clone(),
                edge_force_n: load_case.edge_force_n,
                compliance_j,
                max_displacement_m,
                max_von_mises_pa,
                aggregated_stress_pa,
                stress_feasible,
                solver_residual,
                free_equilibrium_residual_n,
                force_balance_residual_n,
                energy_balance_relative_error,
                converged,
            })
        },
    )?;
    let critical_index = cases
        .iter()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| lhs.compliance_j.total_cmp(&rhs.compliance_j))
        .map(|(index, _)| index)
        .ok_or(StructuralMechanicsError::InvalidConfiguration {
            field: "fem_topology_multi_load.load_cases",
            value: "empty".to_string(),
        })?;
    let max_compliance_j = cases[critical_index].compliance_j;
    Ok(FemTopology2DMultiLoadResult {
        study: study.clone(),
        topology,
        critical_case: cases[critical_index].name.clone(),
        max_compliance_j,
        max_displacement_m: cases
            .iter()
            .map(|case| case.max_displacement_m)
            .fold(0.0, f64::max),
        max_von_mises_pa: cases
            .iter()
            .map(|case| case.max_von_mises_pa)
            .fold(0.0, f64::max),
        aggregated_stress_pa: cases
            .iter()
            .map(|case| case.aggregated_stress_pa)
            .fold(0.0, f64::max),
        stress_feasible: cases.iter().all(|case| case.stress_feasible),
        factorization_memory_bytes,
        factorization_reused: true,
        converged: cases.iter().all(|case| case.converged),
        model_assumption: StructuralModelAssumption::SimpTopology2D,
        execution: StructuralExecutionProvenance::cpu_fem_topology_2d_multi_load_reference(),
        cases,
    })
}

/// Run topology optimization against the exact maximum compliance over all
/// named load cases.  One constrained sparse factorization is built per
/// density evaluation and its solve is reused for every right-hand side.
pub fn run_fem_topology_2d_envelope(
    study: &FemTopology2DEnvelopeStudy,
) -> Result<FemTopology2DEnvelopeResult, StructuralMechanicsError> {
    study.validate()?;
    let mut topology = study.topology.clone();
    topology.edge_force_n = study.load_cases[0].edge_force_n;
    let mesh = topology.geometry.mesh();
    let elements = prepare_elements(&mesh, &topology)?;
    let volumes: Vec<f64> = elements
        .iter()
        .map(|element| element.area_m2 * topology.geometry.thickness_m)
        .collect();
    let protected = protected_elements(&mesh, &elements, topology.geometry);
    let protected_volume_fraction = weighted_mean_bool(&protected, &volumes);
    if protected_volume_fraction > topology.volume_fraction + 1.0e-12 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "fem_topology_envelope.volume_fraction",
            value: format!(
                "target {} is below protected boundary fraction {}",
                topology.volume_fraction, protected_volume_fraction
            ),
        });
    }
    let filter = DensityFilter::build(&mesh, topology.filter_radius_m);
    let context = EnvelopeContext {
        mesh: &mesh,
        elements: &elements,
        topology: &topology,
        load_cases: &study.load_cases,
        filter: &filter,
        protected: &protected,
    };
    let mut raw_density = vec![topology.volume_fraction; elements.len()];
    for (rho, &is_protected) in raw_density.iter_mut().zip(&protected) {
        if is_protected {
            *rho = 1.0;
        }
    }
    raw_density = project_to_volume(
        &raw_density,
        &filter,
        &topology,
        &volumes,
        &protected,
        topology.projection_beta,
    )
    .raw_density;
    let initial = evaluate_envelope_raw(&context, &raw_density, topology.projection_beta)?;
    let initial_case_compliances_j = initial
        .cases
        .iter()
        .map(|case| case.compliance_j)
        .collect::<Vec<_>>();
    let initial_max_compliance_j = initial.objective_j;
    let mut history = vec![initial.objective_j];
    let mut iterations = 0;
    let mut converged = false;

    for stage in 0..topology.continuation_steps {
        let beta = topology.projection_beta * 2.0_f64.powi(stage as i32);
        for _ in 0..topology.iterations_per_stage {
            let current = evaluate_envelope_raw(&context, &raw_density, beta)?;
            let projection = HeavisideProjection::new(beta, topology.projection_threshold);
            let filtered = filter.filter_densities(&raw_density);
            let filtered_gradient = current
                .gradient
                .iter()
                .zip(&filtered)
                .map(|(&gradient, &density)| gradient * projection.project_derivative(density))
                .collect::<Vec<_>>();
            let raw_gradient = filter.filter_sensitivities_transpose(&filtered_gradient);
            let max_gradient = raw_gradient
                .iter()
                .map(|value| value.abs())
                .fold(0.0, f64::max)
                .max(1.0e-30);
            let candidate = raw_density
                .iter()
                .zip(&raw_gradient)
                .zip(&protected)
                .map(|((&rho, &gradient), &is_protected)| {
                    if is_protected {
                        1.0
                    } else {
                        (rho - topology.move_limit * gradient / max_gradient)
                            .clamp(topology.minimum_density, 1.0)
                    }
                })
                .collect::<Vec<_>>();
            let projected_candidate =
                project_to_volume(&candidate, &filter, &topology, &volumes, &protected, beta);
            let next = evaluate_envelope_raw(&context, &projected_candidate.raw_density, beta)?;
            let max_change = raw_density
                .iter()
                .zip(&projected_candidate.raw_density)
                .map(|(old, new)| (old - new).abs())
                .fold(0.0, f64::max);
            let relative_objective_change =
                (current.objective_j - next.objective_j).abs() / current.objective_j.abs().max(1.0);
            raw_density = projected_candidate.raw_density;
            history.push(next.objective_j);
            iterations += 1;
            if max_change <= topology.convergence_tolerance
                && relative_objective_change <= topology.convergence_tolerance
            {
                converged = true;
                break;
            }
        }
    }

    let final_beta = topology.projection_beta
        * 2.0_f64.powi((topology.continuation_steps.saturating_sub(1)) as i32);
    let final_eval = evaluate_envelope_raw(&context, &raw_density, final_beta)?;
    let binary_density = final_eval
        .projected_density
        .iter()
        .zip(&protected)
        .map(|(&rho, &is_protected)| {
            if is_protected || rho >= topology.projection_threshold {
                1.0
            } else {
                topology.minimum_density
            }
        })
        .collect::<Vec<_>>();
    let connected = connected_design(&elements, &binary_density, &protected);
    if !connected {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "fem_topology_envelope.connectivity",
            value: 1.0,
        });
    }
    let projected = evaluate_envelope_physical(
        &mesh,
        &elements,
        &topology,
        &study.load_cases,
        &binary_density
            .iter()
            .map(|&rho| stiffness_ratio(rho, &topology))
            .collect::<Vec<_>>(),
        &binary_density,
    )?;
    let active_mesh = body_fitted_mesh(&mesh, &elements, &binary_density)?;
    let active_elements = prepare_elements(&active_mesh, &topology)?;
    let body_fitted = evaluate_envelope_physical(
        &active_mesh,
        &active_elements,
        &topology,
        &study.load_cases,
        &vec![1.0; active_elements.len()],
        &vec![1.0; active_elements.len()],
    )?;
    let projected_reanalysis_max_compliance_j = projected.objective_j;
    let body_fitted_reanalysis_max_compliance_j = body_fitted.objective_j;
    let reanalysis_feasible = relative_error(
        projected_reanalysis_max_compliance_j,
        final_eval.objective_j,
    ) <= 0.25
        && relative_error(
            body_fitted_reanalysis_max_compliance_j,
            projected_reanalysis_max_compliance_j,
        ) <= 0.25;
    let gradient_taylor_relative_error =
        envelope_gradient_taylor_error(&context, &raw_density, final_beta)?;
    let final_case_compliances_j = final_eval
        .cases
        .iter()
        .map(|case| case.compliance_j)
        .collect::<Vec<_>>();
    let projected_reanalysis_case_compliances_j = projected
        .cases
        .iter()
        .map(|case| case.compliance_j)
        .collect::<Vec<_>>();
    let body_fitted_reanalysis_case_compliances_j = body_fitted
        .cases
        .iter()
        .map(|case| case.compliance_j)
        .collect::<Vec<_>>();
    let critical_case = final_eval
        .cases
        .iter()
        .max_by(|lhs, rhs| lhs.compliance_j.total_cmp(&rhs.compliance_j))
        .map(|case| case.name.clone())
        .ok_or(StructuralMechanicsError::InvalidConfiguration {
            field: "fem_topology_envelope.load_cases",
            value: "empty".to_string(),
        })?;
    let final_physical_density = final_eval.projected_density.clone();
    let final_volume_fraction = weighted_mean_f64(&final_physical_density, &volumes);
    let aggregated_stress_pa = final_eval
        .cases
        .iter()
        .map(|case| case.aggregated_stress_pa)
        .fold(0.0, f64::max);
    let stress_feasible = final_eval.cases.iter().all(|case| case.stress_feasible);

    Ok(FemTopology2DEnvelopeResult {
        study: study.clone(),
        initial_case_compliances_j,
        final_case_compliances_j,
        projected_reanalysis_case_compliances_j,
        body_fitted_reanalysis_case_compliances_j,
        initial_max_compliance_j,
        final_max_compliance_j: final_eval.objective_j,
        projected_reanalysis_max_compliance_j,
        body_fitted_reanalysis_max_compliance_j,
        critical_case,
        aggregated_stress_pa,
        stress_feasible,
        final_physical_density,
        projected_density: final_eval.projected_density,
        final_volume_fraction,
        iterations,
        converged: converged && final_eval.cases.iter().all(|case| case.converged),
        history,
        gradient_taylor_relative_error,
        connected,
        reanalysis_feasible,
        factorization_memory_bytes: final_eval.factorization_memory_bytes,
        factorization_reused_across_cases: true,
        cases: final_eval.cases,
        model_assumption: StructuralModelAssumption::SimpTopology2D,
        execution: StructuralExecutionProvenance::cpu_fem_topology_2d_envelope_reference(),
    })
}

fn evaluate_envelope_raw(
    context: &EnvelopeContext<'_>,
    raw_density: &[f64],
    beta: f64,
) -> Result<EnvelopeEvaluation, StructuralMechanicsError> {
    let projected = project_density(
        raw_density,
        context.filter,
        context.topology,
        context.protected,
        beta,
    );
    let ratios = projected
        .iter()
        .map(|&rho| stiffness_ratio(rho, context.topology))
        .collect::<Vec<_>>();
    let mut result = evaluate_envelope_physical(
        context.mesh,
        context.elements,
        context.topology,
        context.load_cases,
        &ratios,
        &projected,
    )?;
    result.projected_density = projected;
    Ok(result)
}

/// Run the real 2D FEM topology envelope followed by erosion/dilation and
/// material/load uncertainty reanalysis.
pub fn run_fem_topology_2d_robust(
    study: &FemTopology2DRobustStudy,
) -> Result<FemTopology2DRobustResult, StructuralMechanicsError> {
    study.validate()?;
    let nominal = run_fem_topology_2d_envelope(&study.envelope)?;
    let topology = &study.envelope.topology;
    let mesh = topology.geometry.mesh();
    let elements = prepare_elements(&mesh, topology)?;
    let volumes: Vec<f64> = elements
        .iter()
        .map(|element| element.area_m2 * topology.geometry.thickness_m)
        .collect();
    let protected = protected_elements(&mesh, &elements, topology.geometry);
    let nominal_density = nominal.final_physical_density.clone();
    let eroded_density = morphology_density(
        &elements,
        &nominal_density,
        &protected,
        topology.minimum_density,
        topology.projection_threshold,
        study.erosion_radius_cells,
        true,
    );
    let dilated_density = morphology_density(
        &elements,
        &nominal_density,
        &protected,
        topology.minimum_density,
        topology.projection_threshold,
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
        for uncertainty in &study.uncertainty_cases {
            let mut scenario = topology.clone();
            scenario.material.youngs_modulus_pa *= uncertainty.youngs_modulus_scale;
            scenario.geometry.thickness_m *= uncertainty.thickness_scale;
            scenario.edge_force_n = study.envelope.load_cases[0].edge_force_n;
            let scenario_elements = prepare_elements(&mesh, &scenario)?;
            let ratios = density
                .iter()
                .map(|&rho| stiffness_ratio(rho, &scenario))
                .collect::<Vec<_>>();
            let load_cases = study
                .envelope
                .load_cases
                .iter()
                .map(|load_case| FemTopology2DLoadCase {
                    name: load_case.name.clone(),
                    edge_force_n: load_case
                        .edge_force_n
                        .map(|value| value * uncertainty.load_scale),
                })
                .collect::<Vec<_>>();
            let evaluation = evaluate_envelope_physical(
                &mesh,
                &scenario_elements,
                &scenario,
                &load_cases,
                &ratios,
                &density,
            )?;
            for case in evaluation.cases {
                let feasible = case.converged
                    && case.compliance_j <= study.allowable_compliance_j
                    && case.stress_feasible
                    && case.solver_residual.is_finite()
                    && case.free_equilibrium_residual_n.is_finite()
                    && case.force_balance_residual_n.is_finite()
                    && case.energy_balance_relative_error.is_finite();
                cases.push(FemTopology2DRobustCaseResult {
                    morphology: morphology.to_string(),
                    uncertainty_case: uncertainty.name.clone(),
                    load_case: case.name,
                    compliance_j: case.compliance_j,
                    max_displacement_m: case.max_displacement_m,
                    aggregated_stress_pa: case.aggregated_stress_pa,
                    stress_feasible: case.stress_feasible,
                    solver_residual: case.solver_residual,
                    free_equilibrium_residual_n: case.free_equilibrium_residual_n,
                    force_balance_residual_n: case.force_balance_residual_n,
                    energy_balance_relative_error: case.energy_balance_relative_error,
                    converged: case.converged,
                    feasible,
                });
            }
        }
    }

    let worst_case = cases
        .iter()
        .max_by(|lhs, rhs| lhs.compliance_j.total_cmp(&rhs.compliance_j))
        .map(|case| {
            format!(
                "{}/{}/{}",
                case.morphology, case.uncertainty_case, case.load_case
            )
        })
        .ok_or(StructuralMechanicsError::ValidationFailure {
            check: "fem_topology_robust.cases",
            value: 0.0,
        })?;
    let worst_case_compliance_j = cases
        .iter()
        .map(|case| case.compliance_j)
        .fold(f64::NEG_INFINITY, f64::max);
    let worst_case_aggregated_stress_pa = cases
        .iter()
        .map(|case| case.aggregated_stress_pa)
        .fold(0.0, f64::max);
    let stress_feasible = cases.iter().all(|case| case.stress_feasible);
    let erosion_connected = connected_design(&elements, &eroded_density, &protected);
    let dilation_connected = connected_design(&elements, &dilated_density, &protected);
    let morphology_feasible = erosion_connected
        && dilation_connected
        && minimum_feature_ok(&elements, &eroded_density, topology.minimum_feature_cells)
        && minimum_feature_ok(&elements, &dilated_density, topology.minimum_feature_cells);
    let uncertainty_feasible = cases.iter().all(|case| case.feasible);
    let robust_feasible = morphology_feasible
        && uncertainty_feasible
        && stress_feasible
        && worst_case_compliance_j.is_finite()
        && nominal.gradient_taylor_relative_error.is_finite();

    Ok(FemTopology2DRobustResult {
        study: study.clone(),
        nominal_volume_fraction: weighted_mean_f64(&nominal_density, &volumes),
        eroded_volume_fraction: weighted_mean_f64(&eroded_density, &volumes),
        dilated_volume_fraction: weighted_mean_f64(&dilated_density, &volumes),
        nominal_density,
        eroded_density,
        dilated_density,
        cases,
        worst_case_compliance_j,
        worst_case,
        worst_case_aggregated_stress_pa,
        stress_feasible,
        erosion_connected,
        dilation_connected,
        morphology_feasible,
        uncertainty_feasible,
        robust_feasible,
        factorization_reused_within_scenario: true,
        gradient_taylor_relative_error: nominal.gradient_taylor_relative_error,
        model_assumption: StructuralModelAssumption::SimpTopology2D,
        execution: StructuralExecutionProvenance::cpu_fem_topology_2d_robust_reference(),
        nominal,
    })
}

fn evaluate_envelope_physical(
    mesh: &Mesh,
    elements: &[TopologyElement],
    topology: &FemTopology2DStudy,
    load_cases: &[FemTopology2DLoadCase],
    ratios: &[f64],
    physical_density: &[f64],
) -> Result<EnvelopeEvaluation, StructuralMechanicsError> {
    if elements.is_empty()
        || ratios.len() != elements.len()
        || physical_density.len() != elements.len()
        || load_cases.is_empty()
    {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "FEM topology envelope has empty or mismatched inputs".to_string(),
        });
    }
    let node_count = mesh.nodes.len();
    let dofs = node_count * DOFS_PER_NODE;
    let stiffness = assemble_global_stiffness(elements, ratios, node_count);
    let unconstrained_csr = stiffness.to_csr();
    let fixed_dofs = boundary_dofs_by_coordinate(mesh, true);
    let mut zero_system = FemSystem::new(stiffness.clone(), vec![0.0; dofs]);
    try_apply_homogeneous_dirichlet_dofs(&mut zero_system, &fixed_dofs).map_err(|error| {
        StructuralMechanicsError::SolverFailure {
            stage: "fem topology envelope boundary",
            message: error.to_string(),
        }
    })?;
    let constrained_csr = zero_system.matrix.to_csr();
    let factorization = factorize_sparse(
        &constrained_csr,
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..SparseDirectConfig::default()
        },
        None,
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "fem topology envelope factorization",
        message: error.to_string(),
    })?;
    let factorization_memory_bytes = factorization.factor_memory_bytes();
    let fixed = std::collections::HashSet::<usize>::from_iter(fixed_dofs.iter().copied());
    let mut cases = Vec::with_capacity(load_cases.len());
    let mut case_energies = Vec::with_capacity(load_cases.len());
    for load_case in load_cases {
        let mut rhs = vec![0.0; dofs];
        assemble_right_edge_load(mesh, load_case.edge_force_n, &mut rhs);
        let external_rhs = rhs.clone();
        let mut system = FemSystem::new(stiffness.clone(), rhs);
        try_apply_homogeneous_dirichlet_dofs(&mut system, &fixed_dofs).map_err(|error| {
            StructuralMechanicsError::SolverFailure {
                stage: "fem topology envelope case boundary",
                message: error.to_string(),
            }
        })?;
        let solution = factorization.solve(&system.rhs).map_err(|error| {
            StructuralMechanicsError::SolverFailure {
                stage: "fem topology envelope case solve",
                message: error.to_string(),
            }
        })?;
        let full_residual = csr_residual_vector(&unconstrained_csr, &solution, &external_rhs);
        let free_equilibrium_residual_n = l2_norm(
            &full_residual
                .iter()
                .enumerate()
                .filter_map(|(index, value)| (!fixed.contains(&index)).then_some(*value))
                .collect::<Vec<_>>(),
        );
        let mut reaction = [0.0; 2];
        for &dof in &fixed_dofs {
            reaction[dof % DOFS_PER_NODE] += full_residual[dof];
        }
        let applied = [
            external_rhs.iter().step_by(DOFS_PER_NODE).sum::<f64>(),
            external_rhs
                .iter()
                .skip(1)
                .step_by(DOFS_PER_NODE)
                .sum::<f64>(),
        ];
        let force_balance_residual_n =
            ((reaction[0] + applied[0]).powi(2) + (reaction[1] + applied[1]).powi(2)).sqrt();
        let internal = csr_matvec(&unconstrained_csr, &solution);
        let compliance_j = dot(&external_rhs, &solution);
        let internal_energy = 0.5 * dot(&solution, &internal);
        let rhs_energy = 0.5 * compliance_j;
        let energy_balance_relative_error =
            (internal_energy - rhs_energy).abs() / rhs_energy.abs().max(1.0e-30);
        let solver_residual = csr_residual(&system.matrix.to_csr(), &solution, &system.rhs);
        let mut element_energy_j = Vec::with_capacity(elements.len());
        let mut element_von_mises_pa = Vec::with_capacity(elements.len());
        for element in elements {
            let local = local_displacement(element, &solution);
            element_energy_j.push(quadratic_form(&local, &element.local_stiffness));
            element_von_mises_pa.push(von_mises(element_stress(element, &local, topology)));
        }
        let max_von_mises_pa = element_von_mises_pa.iter().copied().fold(0.0, f64::max);
        let aggregated_stress_pa =
            aggregate_stress(&element_von_mises_pa, topology.stress_aggregation_p);
        let stress_feasible = aggregated_stress_pa <= topology.allowable_stress_pa;
        let max_displacement_m = solution
            .as_chunks::<DOFS_PER_NODE>()
            .0
            .iter()
            .map(|chunk| chunk[0].hypot(chunk[1]))
            .fold(0.0, f64::max);
        let converged = solver_residual.is_finite()
            && free_equilibrium_residual_n.is_finite()
            && force_balance_residual_n.is_finite()
            && energy_balance_relative_error.is_finite()
            && solver_residual <= VALIDATION_TOLERANCE
            && free_equilibrium_residual_n <= VALIDATION_TOLERANCE
            && force_balance_residual_n <= VALIDATION_TOLERANCE
            && energy_balance_relative_error <= VALIDATION_TOLERANCE;
        cases.push(FemTopology2DLoadCaseResult {
            name: load_case.name.clone(),
            edge_force_n: load_case.edge_force_n,
            compliance_j,
            max_displacement_m,
            max_von_mises_pa,
            aggregated_stress_pa,
            stress_feasible,
            solver_residual,
            free_equilibrium_residual_n,
            force_balance_residual_n,
            energy_balance_relative_error,
            converged,
        });
        case_energies.push(element_energy_j);
    }
    let objective_j = cases
        .iter()
        .map(|case| case.compliance_j)
        .fold(f64::NEG_INFINITY, f64::max);
    if !objective_j.is_finite() {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "fem_topology_envelope.compliance",
            value: objective_j,
        });
    }
    let active = cases
        .iter()
        .map(|case| (case.compliance_j - objective_j).abs() <= 1.0e-10 * objective_j.abs().max(1.0))
        .collect::<Vec<_>>();
    let active_count = active.iter().filter(|&&is_active| is_active).count().max(1) as f64;
    let gradient = (0..elements.len())
        .map(|element| {
            let active_energy = case_energies
                .iter()
                .zip(&active)
                .filter_map(|(energies, &is_active)| is_active.then_some(energies[element]))
                .sum::<f64>()
                / active_count;
            -interpolation_derivative(physical_density[element], topology) * active_energy
        })
        .collect::<Vec<_>>();
    Ok(EnvelopeEvaluation {
        projected_density: Vec::new(),
        cases,
        objective_j,
        gradient,
        factorization_memory_bytes,
    })
}

fn envelope_gradient_taylor_error(
    context: &EnvelopeContext<'_>,
    raw: &[f64],
    beta: f64,
) -> Result<f64, StructuralMechanicsError> {
    let Some(index) = raw.iter().enumerate().find_map(|(index, &rho)| {
        (!context.protected[index] && rho > context.topology.minimum_density + 0.05)
            .then_some(index)
    }) else {
        return Ok(0.0);
    };
    let base = evaluate_envelope_raw(context, raw, beta)?;
    let h = 1.0e-5;
    let mut plus = raw.to_vec();
    let mut minus = raw.to_vec();
    plus[index] = (plus[index] + h).min(1.0);
    minus[index] = (minus[index] - h).max(context.topology.minimum_density);
    let plus_value = evaluate_envelope_raw(context, &plus, beta)?.objective_j;
    let minus_value = evaluate_envelope_raw(context, &minus, beta)?.objective_j;
    let numerical = (plus_value - minus_value) / (2.0 * h);
    let filtered = context.filter.filter_densities(raw);
    let projection = HeavisideProjection::new(beta, context.topology.projection_threshold);
    let filtered_gradient = base
        .gradient
        .iter()
        .zip(&filtered)
        .map(|(&gradient, &density)| gradient * projection.project_derivative(density))
        .collect::<Vec<_>>();
    let analytic = context
        .filter
        .filter_sensitivities_transpose(&filtered_gradient)[index];
    Ok((numerical - analytic).abs() / numerical.abs().max(analytic.abs()).max(1.0e-12))
}

fn prepare_elements(
    mesh: &Mesh,
    study: &FemTopology2DStudy,
) -> Result<Vec<TopologyElement>, StructuralMechanicsError> {
    let material = IsotropicMaterial::new(
        study.material.youngs_modulus_pa,
        study.material.poisson_ratio,
    );
    let d = material.voigt_2d_plane_stress();
    let mut elements = Vec::with_capacity(mesh.elements.len());
    for element in &mesh.elements {
        if element.element_type != ElementType::Triangle || element.nodes.len() != 3 {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: "FEM topology requires P1 triangular elements".to_string(),
            });
        }
        let nodes = [element.nodes[0], element.nodes[1], element.nodes[2]];
        if nodes.iter().any(|&node| node >= mesh.nodes.len()) {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: "FEM topology element references a missing node".to_string(),
            });
        }
        let p0 = mesh.nodes[nodes[0]];
        let p1 = mesh.nodes[nodes[1]];
        let p2 = mesh.nodes[nodes[2]];
        let determinant = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
        let area_m2 = 0.5 * determinant.abs();
        if !area_m2.is_finite() || area_m2 <= f64::EPSILON {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: "FEM topology contains a degenerate triangle".to_string(),
            });
        }
        let gradients = [
            [(p1.y - p2.y) / determinant, (p2.x - p1.x) / determinant],
            [(p2.y - p0.y) / determinant, (p0.x - p2.x) / determinant],
            [(p0.y - p1.y) / determinant, (p1.x - p0.x) / determinant],
        ];
        let mut local_stiffness = [0.0; 36];
        for a in 0..6 {
            let b_a = strain_column(a, &gradients);
            for b in 0..6 {
                let b_b = strain_column(b, &gradients);
                let value = (0..3)
                    .flat_map(|p| (0..3).map(move |q| (p, q)))
                    .map(|(p, q)| b_a[p] * d[p * 3 + q] * b_b[q])
                    .sum::<f64>()
                    * area_m2
                    * study.geometry.thickness_m;
                local_stiffness[a * 6 + b] = value;
            }
        }
        elements.push(TopologyElement {
            nodes,
            gradients,
            area_m2,
            local_stiffness,
            centroid: [(p0.x + p1.x + p2.x) / 3.0, (p0.y + p1.y + p2.y) / 3.0],
        });
    }
    Ok(elements)
}

fn strain_column(dof: usize, gradients: &[[f64; 2]; 3]) -> [f64; 3] {
    let gradient = gradients[dof / 2];
    if dof.is_multiple_of(2) {
        [gradient[0], 0.0, gradient[1]]
    } else {
        [0.0, gradient[1], gradient[0]]
    }
}

fn assemble_global_stiffness(
    elements: &[TopologyElement],
    ratios: &[f64],
    node_count: usize,
) -> TripletMatrix<f64> {
    let mut matrix = TripletMatrix::with_capacity(
        node_count * DOFS_PER_NODE,
        node_count * DOFS_PER_NODE,
        elements.len() * 36,
    );
    for (element, &ratio) in elements.iter().zip(ratios) {
        let mut dofs = [0usize; 6];
        for (local_node, &node) in element.nodes.iter().enumerate() {
            dofs[local_node * 2] = node * 2;
            dofs[local_node * 2 + 1] = node * 2 + 1;
        }
        for a in 0..6 {
            for b in 0..6 {
                matrix.push(dofs[a], dofs[b], ratio * element.local_stiffness[a * 6 + b]);
            }
        }
    }
    matrix
}

/// Solve the generalized linearized buckling problem for an explicitly
/// compressive x-directed load case.
///
/// The geometric matrix is the P1 initial-stress matrix assembled from the
/// recovered element stress field.  The scalar compressive principal stress
/// is used for the isotropic geometric term; mixed bending stress is not
/// silently presented as a full shell/solid buckling certification.  A
/// transverse-only case therefore reports `None` rather than inventing a
/// buckling factor.
fn linearized_buckling_load_factor(
    mesh: &Mesh,
    elements: &[TopologyElement],
    study: &FemTopology2DStudy,
    ratios: &[f64],
    stresses: &[[f64; 3]],
) -> Result<Option<f64>, StructuralMechanicsError> {
    if study.edge_force_n[0] >= 0.0 {
        return Ok(None);
    }
    if ratios.len() != elements.len() || stresses.len() != elements.len() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "FEM topology buckling data does not match element count".to_string(),
        });
    }
    let mut geometric = TripletMatrix::with_capacity(
        mesh.nodes.len() * DOFS_PER_NODE,
        mesh.nodes.len() * DOFS_PER_NODE,
        elements.len() * 36,
    );
    let mut has_compression = false;
    for ((element, &ratio), stress) in elements.iter().zip(ratios).zip(stresses) {
        let mean = 0.5 * (stress[0] + stress[1]);
        let radius = (0.25 * (stress[0] - stress[1]).powi(2) + stress[2].powi(2)).sqrt();
        let compression_pa = (-mean + radius).max(0.0) * ratio;
        if compression_pa <= 0.0 {
            continue;
        }
        has_compression = true;
        for a in 0..3 {
            for b in 0..3 {
                let gradient_dot = element.gradients[a][0] * element.gradients[b][0]
                    + element.gradients[a][1] * element.gradients[b][1];
                let value =
                    compression_pa * gradient_dot * element.area_m2 * study.geometry.thickness_m;
                let ax = element.nodes[a] * DOFS_PER_NODE;
                let bx = element.nodes[b] * DOFS_PER_NODE;
                geometric.push(ax, bx, value);
                geometric.push(ax + 1, bx + 1, value);
            }
        }
    }
    if !has_compression {
        return Ok(None);
    }

    let elastic = assemble_global_stiffness(elements, ratios, mesh.nodes.len());
    let fixed_dofs = boundary_dofs_by_coordinate(mesh, true);
    let problem = build_linear_buckling_problem(&elastic, &geometric, &fixed_dofs);
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
        stage: "fem topology buckling",
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
    Ok(Some(load_factor))
}

fn assemble_global_mass(
    elements: &[TopologyElement],
    densities: &[f64],
    node_count: usize,
    study: &FemTopology2DStudy,
) -> TripletMatrix<f64> {
    let mut matrix = TripletMatrix::with_capacity(
        node_count * DOFS_PER_NODE,
        node_count * DOFS_PER_NODE,
        elements.len() * 36,
    );
    for (element, &density) in elements.iter().zip(densities) {
        let scale =
            density * study.material.density_kg_m3 * study.geometry.thickness_m * element.area_m2
                / 12.0;
        for i in 0..3 {
            for j in 0..3 {
                let scalar = scale * if i == j { 2.0 } else { 1.0 };
                for component in 0..2 {
                    matrix.push(
                        element.nodes[i] * 2 + component,
                        element.nodes[j] * 2 + component,
                        scalar,
                    );
                }
            }
        }
    }
    matrix
}

fn evaluate_raw(
    mesh: &Mesh,
    elements: &[TopologyElement],
    study: &FemTopology2DStudy,
    filter: &DensityFilter,
    protected: &[bool],
    raw_density: &[f64],
    beta: f64,
) -> Result<TopologyEvaluation, StructuralMechanicsError> {
    let projection = project_density(raw_density, filter, study, protected, beta);
    let ratios: Vec<f64> = projection
        .iter()
        .map(|&rho| stiffness_ratio(rho, study))
        .collect();
    let mut result = evaluate_physical(mesh, elements, study, &ratios)?;
    result.projected_density = projection.clone();
    result.physical_density = projection;
    result.stiffness_ratios = ratios;
    Ok(result)
}

fn evaluate_physical(
    mesh: &Mesh,
    elements: &[TopologyElement],
    study: &FemTopology2DStudy,
    ratios: &[f64],
) -> Result<TopologyEvaluation, StructuralMechanicsError> {
    if elements.is_empty() || ratios.len() != elements.len() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "FEM topology evaluation has no elements or mismatched density".to_string(),
        });
    }
    let node_count = mesh.nodes.len();
    let dofs = node_count * DOFS_PER_NODE;
    let stiffness = assemble_global_stiffness(elements, ratios, node_count);
    let unconstrained_stiffness = stiffness.clone();
    let fixed_dofs = boundary_dofs_by_coordinate(mesh, true);
    let mut rhs = vec![0.0; dofs];
    assemble_right_edge_load(mesh, study.edge_force_n, &mut rhs);
    let external_rhs = rhs.clone();
    let mut system = FemSystem::new(stiffness, rhs);
    try_apply_homogeneous_dirichlet_dofs(&mut system, &fixed_dofs).map_err(|error| {
        StructuralMechanicsError::SolverFailure {
            stage: "fem topology boundary",
            message: error.to_string(),
        }
    })?;
    let solved = solve_real(
        &system.matrix.to_csr(),
        &system.rhs,
        &LinearSolveConfig {
            // The boundary helper keeps the constrained rows explicit, so the
            // assembled CSR system is not guaranteed to remain symmetric.  The
            // small reference system also contains near-void pivots during
            // continuation, so avoid an ILU setup failure and let GMRES work
            // against the assembled operator directly.
            solver: LinearSolverType::GmresIlu,
            max_iterations: 2_000,
            tolerance: 1.0e-9,
            restart: 40,
            ..Default::default()
        },
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "fem topology solve",
        message: format!(
            "{error} (dofs={dofs}, fixed_dofs={}, stiffness_ratio=[{:.3e}, {:.3e}])",
            fixed_dofs.len(),
            ratios.iter().copied().fold(f64::INFINITY, f64::min),
            ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ),
    })?;
    if !solved.converged {
        return Err(StructuralMechanicsError::StaticNonConvergence {
            model: StructuralModelAssumption::SimpTopology2D,
            iterations: solved.iterations,
            residual: solved.residual,
        });
    }
    let solution = solved.x;
    let internal = triplet_matvec(&unconstrained_stiffness, &solution);
    let residual: Vec<f64> = internal
        .iter()
        .zip(&external_rhs)
        .map(|(lhs, rhs)| lhs - rhs)
        .collect();
    let fixed: std::collections::HashSet<usize> = fixed_dofs.iter().copied().collect();
    let free_equilibrium_residual_n = l2_norm(
        &residual
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (!fixed.contains(&index)).then_some(*value))
            .collect::<Vec<_>>(),
    );
    let mut reaction = [0.0; 2];
    for &dof in &fixed_dofs {
        reaction[dof % 2] += residual[dof];
    }
    let applied = [
        external_rhs.iter().step_by(2).sum::<f64>(),
        external_rhs.iter().skip(1).step_by(2).sum::<f64>(),
    ];
    let force_balance_residual_n =
        ((reaction[0] + applied[0]).powi(2) + (reaction[1] + applied[1]).powi(2)).sqrt();
    let constrained_residual = csr_residual(&system.matrix.to_csr(), &solution, &system.rhs);
    let objective_j = dot(&external_rhs, &solution);
    let internal_energy = 0.5 * dot(&solution, &internal);
    let rhs_energy = 0.5 * objective_j;
    let energy_balance_relative_error =
        (internal_energy - rhs_energy).abs() / rhs_energy.abs().max(1.0e-30);
    let force_scale = l2_norm(&applied).max(1.0);
    if !constrained_residual.is_finite()
        || !free_equilibrium_residual_n.is_finite()
        || !force_balance_residual_n.is_finite()
        || constrained_residual > VALIDATION_TOLERANCE * force_scale
        || free_equilibrium_residual_n > VALIDATION_TOLERANCE * force_scale
        || force_balance_residual_n > VALIDATION_TOLERANCE * force_scale
        || energy_balance_relative_error > 1.0e-8
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "fem_topology_equilibrium_energy",
            value: constrained_residual
                .max(free_equilibrium_residual_n / force_scale)
                .max(force_balance_residual_n / force_scale)
                .max(energy_balance_relative_error),
        });
    }
    let mut element_energy_j = Vec::with_capacity(elements.len());
    let mut element_von_mises_pa = Vec::with_capacity(elements.len());
    let mut element_stresses = Vec::with_capacity(elements.len());
    for (element, &ratio) in elements.iter().zip(ratios) {
        let local = local_displacement(element, &solution);
        let energy = quadratic_form(&local, &element.local_stiffness) * ratio;
        let stress = element_stress(element, &local, study);
        element_energy_j.push(energy);
        element_von_mises_pa.push(von_mises(stress));
        element_stresses.push(stress);
    }
    let max_displacement_m = solution
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| chunk[0].hypot(chunk[1]))
        .fold(0.0, f64::max);
    Ok(TopologyEvaluation {
        projected_density: Vec::new(),
        physical_density: Vec::new(),
        stiffness_ratios: Vec::new(),
        displacement: solution,
        objective_j,
        element_energy_j,
        element_von_mises_pa,
        element_stress: element_stresses,
        solver_residual: constrained_residual,
        free_equilibrium_residual_n,
        force_balance_residual_n,
        energy_balance_relative_error,
        max_displacement_m,
    })
}

pub(crate) struct FemStaticStressState {
    pub(crate) element_stress: Vec<[f64; 3]>,
    pub(crate) solver_residual: f64,
    pub(crate) free_equilibrium_residual_n: f64,
    pub(crate) force_balance_residual_n: f64,
}

/// Solve one full-density P1 plane-stress state for another structural family.
pub(crate) fn solve_fem_static_stress_state(
    study: &FemTopology2DStudy,
) -> Result<FemStaticStressState, StructuralMechanicsError> {
    study.validate()?;
    let mesh = study.geometry.mesh();
    let elements = prepare_elements(&mesh, study)?;
    let evaluation = evaluate_physical(&mesh, &elements, study, &vec![1.0; elements.len()])?;
    Ok(FemStaticStressState {
        element_stress: evaluation.element_stress,
        solver_residual: evaluation.solver_residual,
        free_equilibrium_residual_n: evaluation.free_equilibrium_residual_n,
        force_balance_residual_n: evaluation.force_balance_residual_n,
    })
}

fn boundary_dofs_by_coordinate(mesh: &Mesh, minimum: bool) -> Vec<usize> {
    let coordinate = if minimum {
        mesh.nodes
            .iter()
            .map(|point| point.x)
            .fold(f64::INFINITY, f64::min)
    } else {
        mesh.nodes
            .iter()
            .map(|point| point.x)
            .fold(f64::NEG_INFINITY, f64::max)
    };
    let scale = mesh
        .nodes
        .iter()
        .map(|point| point.x.abs().max(point.y.abs()))
        .fold(1.0, f64::max);
    let tolerance = 1.0e-10 * scale;
    mesh.nodes
        .iter()
        .enumerate()
        .filter(|(_, point)| (point.x - coordinate).abs() <= tolerance)
        .flat_map(|(node, _)| [node * 2, node * 2 + 1])
        .collect()
}

fn assemble_right_edge_load(mesh: &Mesh, force: [f64; 2], rhs: &mut [f64]) {
    let coordinate = mesh
        .nodes
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let scale = mesh
        .nodes
        .iter()
        .map(|point| point.x.abs().max(point.y.abs()))
        .fold(1.0, f64::max);
    let tolerance = 1.0e-10 * scale;
    let mut nodes = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, point)| (point.x - coordinate).abs() <= tolerance)
        .map(|(node, point)| (node, point.y))
        .collect::<Vec<_>>();
    nodes.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1));
    let total_length = nodes
        .windows(2)
        .map(|window| (window[1].1 - window[0].1).abs())
        .sum::<f64>();
    if total_length <= f64::EPSILON {
        return;
    }
    for window in nodes.windows(2) {
        let length = (window[1].1 - window[0].1).abs();
        for component in 0..2 {
            let traction = force[component] / total_length;
            rhs[window[0].0 * 2 + component] += 0.5 * traction * length;
            rhs[window[1].0 * 2 + component] += 0.5 * traction * length;
        }
    }
}

fn local_displacement(element: &TopologyElement, solution: &[f64]) -> [f64; 6] {
    let mut local = [0.0; 6];
    for (index, &node) in element.nodes.iter().enumerate() {
        local[index * 2] = solution[node * 2];
        local[index * 2 + 1] = solution[node * 2 + 1];
    }
    local
}

fn element_stress(
    element: &TopologyElement,
    displacement: &[f64; 6],
    study: &FemTopology2DStudy,
) -> [f64; 3] {
    let material = IsotropicMaterial::new(
        study.material.youngs_modulus_pa,
        study.material.poisson_ratio,
    );
    let d = material.voigt_2d_plane_stress();
    let mut strain = [0.0; 3];
    for (dof, &value) in displacement.iter().enumerate().take(6) {
        let column = strain_column(dof, &element.gradients);
        for component in 0..3 {
            strain[component] += column[component] * value;
        }
    }
    [
        d[0] * strain[0] + d[1] * strain[1] + d[2] * strain[2],
        d[3] * strain[0] + d[4] * strain[1] + d[5] * strain[2],
        d[6] * strain[0] + d[7] * strain[1] + d[8] * strain[2],
    ]
}

fn von_mises(stress: [f64; 3]) -> f64 {
    (stress[0] * stress[0] - stress[0] * stress[1]
        + stress[1] * stress[1]
        + 3.0 * stress[2] * stress[2])
        .sqrt()
}

fn stiffness_ratio(rho: f64, study: &FemTopology2DStudy) -> f64 {
    let rho = rho.clamp(study.minimum_density, 1.0);
    let interpolated = match study.interpolation {
        TopologyInterpolation::Simp => rho.powf(study.interpolation_parameter),
        TopologyInterpolation::Ramp => rho / (1.0 + study.interpolation_parameter * (1.0 - rho)),
    };
    DEFAULT_MINIMUM_STIFFNESS + (1.0 - DEFAULT_MINIMUM_STIFFNESS) * interpolated
}

fn interpolation_derivative(rho: f64, study: &FemTopology2DStudy) -> f64 {
    let rho = rho.clamp(study.minimum_density, 1.0);
    let derivative = match study.interpolation {
        TopologyInterpolation::Simp => {
            study.interpolation_parameter * rho.powf(study.interpolation_parameter - 1.0)
        }
        TopologyInterpolation::Ramp => {
            (1.0 + study.interpolation_parameter)
                / (1.0 + study.interpolation_parameter * (1.0 - rho)).powi(2)
        }
    };
    (1.0 - DEFAULT_MINIMUM_STIFFNESS) * derivative
}

fn protected_elements(
    mesh: &Mesh,
    elements: &[TopologyElement],
    geometry: FemTopology2DGeometry,
) -> Vec<bool> {
    let left = 0.0;
    let right = geometry.length_m;
    let tolerance = 1.0e-10 * geometry.length_m.max(geometry.height_m).max(1.0);
    elements
        .iter()
        .map(|element| {
            element.nodes.iter().any(|&node| {
                let x = mesh.nodes[node].x;
                (x - left).abs() <= tolerance || (x - right).abs() <= tolerance
            })
        })
        .collect()
}

fn project_density(
    raw: &[f64],
    filter: &DensityFilter,
    study: &FemTopology2DStudy,
    protected: &[bool],
    beta: f64,
) -> Vec<f64> {
    let projection = HeavisideProjection::new(beta, study.projection_threshold);
    filter
        .filter_densities(raw)
        .into_iter()
        .zip(protected)
        .map(|(rho, &is_protected)| {
            if is_protected {
                1.0
            } else {
                projection.project(rho).clamp(study.minimum_density, 1.0)
            }
        })
        .collect()
}

struct ProjectedVolume {
    raw_density: Vec<f64>,
}

fn project_to_volume(
    raw: &[f64],
    filter: &DensityFilter,
    study: &FemTopology2DStudy,
    volumes: &[f64],
    protected: &[bool],
    beta: f64,
) -> ProjectedVolume {
    let mut lower = -1.0;
    let mut upper = 1.0;
    let mut candidate = raw.to_vec();
    for _ in 0..60 {
        let shift = 0.5 * (lower + upper);
        candidate = raw
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
        let projected = project_density(&candidate, filter, study, protected, beta);
        let volume = weighted_mean_f64(&projected, volumes);
        if volume > study.volume_fraction {
            upper = shift;
        } else {
            lower = shift;
        }
    }
    ProjectedVolume {
        raw_density: candidate,
    }
}

fn weighted_mean_bool(values: &[bool], weights: &[f64]) -> f64 {
    let values = values.iter().map(|value| if *value { 1.0 } else { 0.0 });
    weighted_mean_iter(values, weights)
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

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs).map(|(lhs, rhs)| lhs * rhs).sum()
}

fn l2_norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}

fn triplet_matvec(matrix: &TripletMatrix<f64>, vector: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; matrix.num_rows];
    for ((&row, &column), &value) in matrix.rows.iter().zip(&matrix.cols).zip(&matrix.values) {
        result[row] += value * vector[column];
    }
    result
}

fn csr_residual(matrix: &sonium_solvers::CsrMatrix<f64>, solution: &[f64], rhs: &[f64]) -> f64 {
    l2_norm(&csr_residual_vector(matrix, solution, rhs)) / l2_norm(rhs).max(1.0)
}

fn csr_matvec(matrix: &sonium_solvers::CsrMatrix<f64>, solution: &[f64]) -> Vec<f64> {
    (0..matrix.num_rows)
        .map(|row| {
            (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                .map(|index| matrix.values[index] * solution[matrix.col_indices[index]])
                .sum::<f64>()
        })
        .collect()
}

fn csr_residual_vector(
    matrix: &sonium_solvers::CsrMatrix<f64>,
    solution: &[f64],
    rhs: &[f64],
) -> Vec<f64> {
    csr_matvec(matrix, solution)
        .into_iter()
        .zip(rhs)
        .map(|(value, rhs)| value - rhs)
        .collect()
}

fn quadratic_form(vector: &[f64; 6], matrix: &[f64; 36]) -> f64 {
    (0..6)
        .map(|row| {
            vector[row]
                * (0..6)
                    .map(|column| matrix[row * 6 + column] * vector[column])
                    .sum::<f64>()
        })
        .sum()
}

fn relative_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(1.0e-30)
}

fn aggregate_stress(stresses: &[f64], p: f64) -> f64 {
    if stresses.is_empty() {
        return f64::INFINITY;
    }
    (stresses.iter().map(|stress| stress.powf(p)).sum::<f64>() / stresses.len() as f64)
        .powf(1.0 / p)
}

fn morphology_density(
    elements: &[TopologyElement],
    density: &[f64],
    protected: &[bool],
    minimum_density: f64,
    threshold: f64,
    radius: usize,
    erosion: bool,
) -> Vec<f64> {
    if radius == 0 {
        return density.to_vec();
    }
    let active: Vec<bool> = density
        .iter()
        .zip(protected)
        .map(|(&rho, &is_protected)| is_protected || rho >= threshold)
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
            if protected[index] || survives {
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

fn element_neighbours(elements: &[TopologyElement]) -> Vec<Vec<usize>> {
    let mut neighbours = vec![Vec::new(); elements.len()];
    for i in 0..elements.len() {
        for j in (i + 1)..elements.len() {
            if elements[i]
                .nodes
                .iter()
                .filter(|node| elements[j].nodes.contains(node))
                .count()
                >= 2
            {
                neighbours[i].push(j);
                neighbours[j].push(i);
            }
        }
    }
    neighbours
}

fn connected_design(elements: &[TopologyElement], density: &[f64], protected: &[bool]) -> bool {
    let active: Vec<bool> = density
        .iter()
        .zip(protected)
        .map(|(&rho, &is_protected)| is_protected || rho >= 0.5)
        .collect();
    let mut neighbours = vec![Vec::new(); elements.len()];
    for i in 0..elements.len() {
        for j in (i + 1)..elements.len() {
            if elements[i]
                .nodes
                .iter()
                .filter(|node| elements[j].nodes.contains(node))
                .count()
                >= 2
            {
                neighbours[i].push(j);
                neighbours[j].push(i);
            }
        }
    }
    let Some(start) = active.iter().position(|active| *active) else {
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
        .all(|(index, active)| !*active || visited[index])
}

fn minimum_feature_ok(elements: &[TopologyElement], density: &[f64], minimum_cells: usize) -> bool {
    if minimum_cells <= 1 {
        return true;
    }
    let mut neighbours = vec![Vec::new(); elements.len()];
    for i in 0..elements.len() {
        for j in (i + 1)..elements.len() {
            if elements[i]
                .nodes
                .iter()
                .filter(|node| elements[j].nodes.contains(node))
                .count()
                >= 2
            {
                neighbours[i].push(j);
                neighbours[j].push(i);
            }
        }
    }
    density.iter().enumerate().all(|(i, &rho)| {
        rho < 0.5
            || neighbours[i].iter().filter(|&&j| density[j] >= 0.5).count() + 1 >= minimum_cells
    })
}

fn overhang_slope(
    elements: &[TopologyElement],
    density: &[f64],
    protected: &[bool],
    geometry: FemTopology2DGeometry,
) -> f64 {
    let mut maximum: f64 = 0.0;
    let cell_x = geometry.length_m / geometry.nx.max(1) as f64;
    for i in 0..elements.len() {
        if density[i] < 0.5 && !protected[i] {
            continue;
        }
        for j in (i + 1)..elements.len() {
            if density[j] < 0.5 && !protected[j]
                || elements[i]
                    .nodes
                    .iter()
                    .filter(|node| elements[j].nodes.contains(node))
                    .count()
                    < 2
            {
                continue;
            }
            let dx = (elements[i].centroid[0] - elements[j].centroid[0]).abs();
            let dy = (elements[i].centroid[1] - elements[j].centroid[1]).abs();
            maximum = maximum.max(dy / dx.max(cell_x));
        }
    }
    maximum
}

fn body_fitted_mesh(
    mesh: &Mesh,
    elements: &[TopologyElement],
    density: &[f64],
) -> Result<Mesh, StructuralMechanicsError> {
    let active: Vec<usize> = density
        .iter()
        .enumerate()
        .filter_map(|(index, &rho)| (rho >= 0.5).then_some(index))
        .collect();
    if active.is_empty() {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "fem_topology.body_fitted_active_domain",
            value: 0.0,
        });
    }
    let mut result = Mesh::new(2);
    let mut node_map = vec![usize::MAX; mesh.nodes.len()];
    for &element_index in &active {
        for &node in &elements[element_index].nodes {
            if node_map[node] == usize::MAX {
                node_map[node] = result.add_node(mesh.nodes[node]);
            }
        }
    }
    for &element_index in &active {
        let nodes = elements[element_index]
            .nodes
            .iter()
            .map(|&node| node_map[node])
            .collect::<Vec<_>>();
        result.add_element(ElementType::Triangle, nodes);
    }
    if result.nodes.is_empty() || result.elements.is_empty() {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "fem_topology.body_fitted_active_domain",
            value: 0.0,
        });
    }
    Ok(result)
}

fn topology_objective_value(
    mesh: &Mesh,
    elements: &[TopologyElement],
    study: &FemTopology2DStudy,
    evaluation: &TopologyEvaluation,
) -> Result<f64, StructuralMechanicsError> {
    match study.objective {
        FemTopology2DObjective::Compliance => Ok(evaluation.objective_j),
        FemTopology2DObjective::FirstFrequency => first_frequency(
            mesh,
            elements,
            study,
            &evaluation.stiffness_ratios,
            &evaluation.physical_density,
        ),
        FemTopology2DObjective::BucklingLoadFactor => linearized_buckling_load_factor(
            mesh,
            elements,
            study,
            &evaluation.stiffness_ratios,
            &evaluation.element_stress,
        )?
        .ok_or(StructuralMechanicsError::ValidationFailure {
            check: "fem_topology.buckling_objective_compression",
            value: 0.0,
        }),
    }
}

fn topology_objective_derivative(
    mesh: &Mesh,
    elements: &[TopologyElement],
    study: &FemTopology2DStudy,
    evaluation: &TopologyEvaluation,
    protected: &[bool],
) -> Result<Vec<f64>, StructuralMechanicsError> {
    match study.objective {
        FemTopology2DObjective::Compliance => Ok(evaluation
            .element_energy_j
            .iter()
            .zip(&evaluation.projected_density)
            .map(|(energy, &rho)| -interpolation_derivative(rho, study) * *energy)
            .collect()),
        FemTopology2DObjective::FirstFrequency => first_frequency_with_derivative(
            mesh,
            elements,
            study,
            &evaluation.stiffness_ratios,
            &evaluation.physical_density,
        )
        .map(|(_, derivative)| derivative),
        FemTopology2DObjective::BucklingLoadFactor => {
            buckling_objective_derivative(mesh, elements, study, evaluation, protected)
        }
    }
}

fn first_frequency(
    mesh: &Mesh,
    elements: &[TopologyElement],
    study: &FemTopology2DStudy,
    stiffness_ratios: &[f64],
    densities: &[f64],
) -> Result<f64, StructuralMechanicsError> {
    first_frequency_with_derivative(mesh, elements, study, stiffness_ratios, densities)
        .map(|(frequency, _)| frequency)
}

fn first_frequency_with_derivative(
    mesh: &Mesh,
    elements: &[TopologyElement],
    study: &FemTopology2DStudy,
    stiffness_ratios: &[f64],
    densities: &[f64],
) -> Result<(f64, Vec<f64>), StructuralMechanicsError> {
    if stiffness_ratios.len() != elements.len() || densities.len() != elements.len() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: format!(
                "FEM topology modal density data does not match element count (elements={}, stiffness_ratios={}, densities={})",
                elements.len(),
                stiffness_ratios.len(),
                densities.len()
            ),
        });
    }
    let stiffness = assemble_global_stiffness(elements, stiffness_ratios, mesh.nodes.len());
    let mass = assemble_global_mass(elements, densities, mesh.nodes.len(), study);
    let fixed = boundary_dofs_by_coordinate(mesh, true);
    let (stiffness, free_to_full) = crate::reduce_triplets(&stiffness, &fixed);
    let (mass, _) = crate::reduce_triplets(&mass, &fixed);
    if stiffness.num_rows == 0 {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "FEM topology modal system has no free DOFs".to_string(),
        });
    }
    let solved = lobpcg_generalized(
        &stiffness,
        &mass,
        &LobpcgConfig {
            num_eigenvalues: 1,
            max_iterations: 500,
            tolerance: 1.0e-7,
            block_padding: 1,
        },
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "fem topology modal reanalysis",
        message: error.to_string(),
    })?;
    let Some(&eigenvalue) = solved.eigenvalues.first() else {
        return Err(StructuralMechanicsError::ModalNonConvergence {
            iterations: solved.iterations,
            residual: f64::INFINITY,
        });
    };
    if !solved.converged {
        return Err(StructuralMechanicsError::ModalNonConvergence {
            iterations: solved.iterations,
            residual: solved
                .residual_norms
                .first()
                .copied()
                .unwrap_or(f64::INFINITY),
        });
    }
    if !eigenvalue.is_finite() || eigenvalue <= 0.0 {
        return Err(StructuralMechanicsError::UnstableModel { eigenvalue });
    }
    let mut mode = (0..stiffness.num_rows)
        .map(|row| solved.eigenvectors[[row, 0]])
        .collect::<Vec<_>>();
    let mass_norm = dot(&mode, &mass.matvec(&mode)).sqrt();
    if !mass_norm.is_finite() || mass_norm <= f64::EPSILON {
        return Err(StructuralMechanicsError::SolverFailure {
            stage: "fem topology modal normalization",
            message: "eigenvector has zero or non-finite mass norm".to_string(),
        });
    }
    for value in &mut mode {
        *value /= mass_norm;
    }
    let mut full_mode = vec![0.0; mesh.nodes.len() * DOFS_PER_NODE];
    for (reduced, &full) in free_to_full.iter().enumerate() {
        full_mode[full] = mode[reduced];
    }
    let frequency_hz = eigenvalue.sqrt() / std::f64::consts::TAU;
    let frequency_scale = 1.0 / (2.0 * std::f64::consts::TAU * eigenvalue.sqrt());
    let derivative = elements
        .iter()
        .zip(densities)
        .map(|(element, &density)| {
            let local = local_displacement(element, &full_mode);
            let d_stiffness = interpolation_derivative(density, study)
                * quadratic_form(&local, &element.local_stiffness);
            let d_mass = consistent_mass_quadratic(&local, element, study);
            (d_stiffness - eigenvalue * d_mass) * frequency_scale
        })
        .collect();
    Ok((frequency_hz, derivative))
}

fn consistent_mass_quadratic(
    local: &[f64; 6],
    element: &TopologyElement,
    study: &FemTopology2DStudy,
) -> f64 {
    let scale = study.material.density_kg_m3 * study.geometry.thickness_m * element.area_m2 / 12.0;
    (0..2)
        .flat_map(|component| {
            (0..3).flat_map(move |row| {
                (0..3).map(move |column| {
                    scale
                        * if row == column { 2.0 } else { 1.0 }
                        * local[row * 2 + component]
                        * local[column * 2 + component]
                })
            })
        })
        .sum()
}

fn buckling_objective_derivative(
    mesh: &Mesh,
    elements: &[TopologyElement],
    study: &FemTopology2DStudy,
    evaluation: &TopologyEvaluation,
    protected: &[bool],
) -> Result<Vec<f64>, StructuralMechanicsError> {
    let mut derivative = vec![0.0; elements.len()];
    for index in 0..elements.len() {
        if protected.get(index).copied().unwrap_or(false) {
            continue;
        }
        let density = evaluation.projected_density[index];
        let plus = (density + 1.0e-4).min(1.0);
        let minus = (density - 1.0e-4).max(study.minimum_density);
        if (plus - minus).abs() <= f64::EPSILON {
            continue;
        }
        let mut plus_density = evaluation.projected_density.clone();
        plus_density[index] = plus;
        let plus_ratios = plus_density
            .iter()
            .map(|&rho| stiffness_ratio(rho, study))
            .collect::<Vec<_>>();
        let plus_evaluation = evaluate_physical(mesh, elements, study, &plus_ratios)?;
        let plus_factor = linearized_buckling_load_factor(
            mesh,
            elements,
            study,
            &plus_ratios,
            &plus_evaluation.element_stress,
        )?
        .ok_or(StructuralMechanicsError::ValidationFailure {
            check: "fem_topology.buckling_objective_compression",
            value: 0.0,
        })?;

        let mut minus_density = evaluation.projected_density.clone();
        minus_density[index] = minus;
        let minus_ratios = minus_density
            .iter()
            .map(|&rho| stiffness_ratio(rho, study))
            .collect::<Vec<_>>();
        let minus_evaluation = evaluate_physical(mesh, elements, study, &minus_ratios)?;
        let minus_factor = linearized_buckling_load_factor(
            mesh,
            elements,
            study,
            &minus_ratios,
            &minus_evaluation.element_stress,
        )?
        .ok_or(StructuralMechanicsError::ValidationFailure {
            check: "fem_topology.buckling_objective_compression",
            value: 0.0,
        })?;
        derivative[index] = (plus_factor - minus_factor) / (plus - minus);
    }
    Ok(derivative)
}

fn objective_gradient_taylor_error(
    mesh: &Mesh,
    elements: &[TopologyElement],
    study: &FemTopology2DStudy,
    evaluation: &TopologyEvaluation,
    derivative: &[f64],
) -> Result<f64, StructuralMechanicsError> {
    let protected = protected_elements(mesh, elements, study.geometry);
    let index = evaluation
        .projected_density
        .iter()
        .enumerate()
        .find(|(index, rho)| {
            !protected[*index] && **rho > study.minimum_density + 0.05 && **rho < 1.0 - 0.05
        })
        .map(|(index, _)| index);
    let Some(index) = index else {
        return Ok(0.0);
    };
    let h = 1.0e-5;
    let mut plus = evaluation.projected_density.clone();
    let mut minus = evaluation.projected_density.clone();
    plus[index] = (plus[index] + h).min(1.0);
    minus[index] = (minus[index] - h).max(study.minimum_density);
    let density_delta = plus[index] - minus[index];
    let plus_ratios = plus
        .iter()
        .map(|&rho| stiffness_ratio(rho, study))
        .collect::<Vec<_>>();
    let minus_ratios = minus
        .iter()
        .map(|&rho| stiffness_ratio(rho, study))
        .collect::<Vec<_>>();
    let mut plus_evaluation = evaluate_physical(mesh, elements, study, &plus_ratios)?;
    plus_evaluation.projected_density = plus.clone();
    plus_evaluation.physical_density = plus;
    plus_evaluation.stiffness_ratios = plus_ratios;
    let mut minus_evaluation = evaluate_physical(mesh, elements, study, &minus_ratios)?;
    minus_evaluation.projected_density = minus.clone();
    minus_evaluation.physical_density = minus;
    minus_evaluation.stiffness_ratios = minus_ratios;
    let plus_value = topology_objective_value(mesh, elements, study, &plus_evaluation)?;
    let minus_value = topology_objective_value(mesh, elements, study, &minus_evaluation)?;
    let numerical = (plus_value - minus_value) / density_delta;
    let analytic = derivative[index];
    Ok((numerical - analytic).abs() / numerical.abs().max(analytic.abs()).max(1.0e-12))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_study() -> FemTopology2DStudy {
        let mut study = FemTopology2DStudy::introductory();
        study.geometry.nx = 6;
        study.geometry.ny = 2;
        study.volume_fraction = 0.95;
        study.move_limit = 0.01;
        study.continuation_steps = 1;
        study.iterations_per_stage = 3;
        study.filter_radius_m = 0.02;
        study.minimum_feature_cells = 1;
        study.maximum_overhang_slope = 100.0;
        study
    }

    #[test]
    fn fem_topology_runs_real_static_modal_and_body_fitted_reanalysis() {
        let result = run_fem_topology_2d(&small_study()).unwrap();

        assert_eq!(result.nodes, 21);
        assert_eq!(result.elements, 24);
        assert!(result.final_compliance_j > 0.0);
        assert!(result.projected_reanalysis_compliance_j > 0.0);
        assert!(result.body_fitted_reanalysis_compliance_j > 0.0);
        assert!(result.first_frequency_hz > 0.0);
        assert!(result.solver_residual < 1.0e-7);
        assert!(result.free_equilibrium_residual_n < 1.0e-7);
        assert!(result.force_balance_residual_n < 1.0e-7);
        assert!(result.energy_balance_relative_error < 1.0e-8);
        assert!(result.gradient_taylor_relative_error.is_finite());
        assert!(
            result
                .final_physical_density
                .iter()
                .all(|&density| density >= result.study.minimum_density - 1.0e-12
                    && density <= 1.0 + 1.0e-12)
        );
        assert!(
            result
                .final_physical_density
                .iter()
                .any(|&density| density > result.study.minimum_density + 1.0e-6)
        );
        assert!(
            result
                .final_physical_density
                .iter()
                .zip(&result.projected_density)
                .all(|(density, projected)| (density - projected).abs() < 1.0e-12)
        );
        assert!(result.connected);
        assert!(result.reanalysis_feasible);
        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::SimpTopology2D
        );
        assert_eq!(
            result.execution.capability_id,
            "sonium-structural.cpu-reference.f64.fem-topology-2d.v1"
        );
    }

    #[test]
    fn fem_topology_frequency_objective_uses_assembled_eigen_sensitivity() {
        let mut study = small_study();
        study.objective = FemTopology2DObjective::FirstFrequency;
        study.iterations_per_stage = 1;
        let result = run_fem_topology_2d(&study).expect("frequency objective should converge");

        assert_eq!(result.objective, FemTopology2DObjective::FirstFrequency);
        assert_eq!(result.objective_history.len(), 2);
        assert!(result.initial_objective_value > 0.0);
        assert!(result.final_objective_value > 0.0);
        assert!(result.final_objective_value >= result.initial_objective_value * (1.0 - 1.0e-6));
        assert!(result.gradient_taylor_relative_error < 1.0e-3);
        assert!(result.first_frequency_hz > 0.0);
    }

    #[test]
    fn fem_topology_buckling_objective_reports_finite_difference_sensitivity() {
        let mut study = small_study();
        study.objective = FemTopology2DObjective::BucklingLoadFactor;
        study.edge_force_n = [-120.0, 0.0];
        study.iterations_per_stage = 1;
        let result = run_fem_topology_2d(&study).expect("buckling objective should converge");

        assert_eq!(result.objective, FemTopology2DObjective::BucklingLoadFactor);
        assert_eq!(result.objective_history.len(), 2);
        assert!(result.initial_objective_value > 0.0);
        assert!(result.final_objective_value > 0.0);
        assert!(result.final_objective_value >= result.initial_objective_value * (1.0 - 1.0e-6));
        assert!(result.gradient_taylor_relative_error < 5.0e-2);
        assert!(result.buckling_load_factor.is_some());
    }

    #[test]
    fn fem_topology_rejects_buckling_objective_without_compressive_x_load() {
        let mut study = small_study();
        study.objective = FemTopology2DObjective::BucklingLoadFactor;
        assert!(matches!(
            run_fem_topology_2d(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "fem_topology.objective",
                ..
            })
        ));
    }

    #[test]
    fn fem_topology_objective_result_round_trips_and_accepts_legacy_study_json() {
        let result = run_fem_topology_2d(&small_study()).expect("topology study should converge");
        let encoded = serde_json::to_value(&result).expect("topology result should serialize");
        let decoded: FemTopology2DResult =
            serde_json::from_value(encoded.clone()).expect("topology result should deserialize");
        assert_eq!(decoded.objective, FemTopology2DObjective::Compliance);
        assert_eq!(decoded.objective_history, result.objective_history);
        assert_eq!(decoded.objective_improved, result.objective_improved);

        let mut legacy = encoded;
        let object = legacy.as_object_mut().expect("topology result object");
        object.remove("objective");
        object.remove("initial_objective_value");
        object.remove("final_objective_value");
        object.remove("objective_history");
        object.remove("objective_improved");
        let study = object
            .get_mut("study")
            .and_then(serde_json::Value::as_object_mut)
            .expect("topology study object");
        study.remove("objective");
        let legacy_decoded: FemTopology2DResult =
            serde_json::from_value(legacy).expect("legacy topology result should deserialize");
        assert_eq!(legacy_decoded.objective, FemTopology2DObjective::Compliance);
        assert!(legacy_decoded.objective_history.is_empty());
    }

    #[test]
    fn fem_topology_rejects_zero_load_and_invalid_volume() {
        let mut study = small_study();
        study.edge_force_n = [0.0, 0.0];
        assert!(matches!(
            run_fem_topology_2d(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "fem_topology.edge_force_n",
                ..
            })
        ));
        study = small_study();
        study.volume_fraction = 0.01;
        assert!(matches!(
            run_fem_topology_2d(&study),
            Err(StructuralMechanicsError::InvalidConfiguration { .. })
        ));

        let mut disconnected = FemTopology2DStudy::introductory();
        disconnected.geometry.nx = 6;
        disconnected.geometry.ny = 2;
        disconnected.continuation_steps = 1;
        disconnected.iterations_per_stage = 3;
        disconnected.filter_radius_m = 0.02;
        disconnected.maximum_overhang_slope = 100.0;
        assert!(matches!(
            run_fem_topology_2d(&disconnected),
            Err(StructuralMechanicsError::ValidationFailure {
                check: "fem_topology.connectivity",
                ..
            })
        ));
    }

    #[test]
    fn fem_topology_reports_fem_compressive_buckling_factor() {
        let mut study = small_study();
        study.edge_force_n = [-120.0, 0.0];
        study.minimum_buckling_load_factor = 1.0;
        let result = run_fem_topology_2d(&study).expect("compressive FEM topology case");

        let factor = result
            .buckling_load_factor
            .expect("compressive case should assemble a buckling eigenproblem");
        assert!(factor.is_finite() && factor > 0.0);
        assert!(result.buckling_feasible);
    }

    #[test]
    fn topology_modal_mass_scales_with_physical_density() {
        let study = small_study();
        let mesh = study.geometry.mesh();
        let elements = prepare_elements(&mesh, &study).expect("valid topology mesh");
        let stiffness_ratios = vec![1.0; elements.len()];
        let full_density = vec![1.0; elements.len()];
        let half_density = vec![0.5; elements.len()];
        let full_frequency =
            first_frequency(&mesh, &elements, &study, &stiffness_ratios, &full_density)
                .expect("full-density modal solve");
        let half_frequency =
            first_frequency(&mesh, &elements, &study, &stiffness_ratios, &half_density)
                .expect("half-density modal solve");

        assert!(((half_frequency / full_frequency) - 2.0_f64.sqrt()).abs() < 1.0e-5);
    }

    #[test]
    fn fem_topology_multi_load_reuses_one_constrained_factorization() {
        let mut study = FemTopology2DMultiLoadStudy::introductory();
        study.topology.geometry.nx = 6;
        study.topology.geometry.ny = 2;
        study.topology.volume_fraction = 0.95;
        study.topology.move_limit = 0.01;
        study.topology.continuation_steps = 1;
        study.topology.iterations_per_stage = 3;
        study.topology.filter_radius_m = 0.02;
        study.topology.maximum_overhang_slope = 100.0;
        let result = run_fem_topology_2d_multi_load(&study).expect("multi-load FEM topology");

        assert_eq!(result.cases.len(), 2);
        assert!(result.factorization_reused);
        assert!(result.factorization_memory_bytes > 0);
        assert!(result.converged);
        assert!(result.cases.iter().all(|case| {
            case.solver_residual < 1.0e-7
                && case.free_equilibrium_residual_n < 1.0e-7
                && case.force_balance_residual_n < 1.0e-7
                && case.energy_balance_relative_error < 1.0e-8
        }));
        assert_eq!(
            result.execution.capability_id,
            "sonium-structural.cpu-reference.f64.fem-topology-2d-multi-load.v1"
        );
    }

    #[test]
    fn fem_topology_envelope_optimizes_all_cases_with_shared_factorization() {
        let mut envelope = FemTopology2DEnvelopeStudy::introductory();
        envelope.topology = small_study();
        envelope.topology.edge_force_n = envelope.load_cases[0].edge_force_n;
        let result = run_fem_topology_2d_envelope(&envelope).expect("envelope FEM topology");

        assert_eq!(result.initial_case_compliances_j.len(), 2);
        assert_eq!(result.final_case_compliances_j.len(), 2);
        assert_eq!(result.cases.len(), 2);
        assert!(result.factorization_reused_across_cases);
        assert!(result.factorization_memory_bytes > 0);
        assert!(result.initial_max_compliance_j > 0.0);
        assert!(result.final_max_compliance_j > 0.0);
        assert!(result.final_max_compliance_j <= result.initial_max_compliance_j);
        assert!(result.gradient_taylor_relative_error.is_finite());
        assert!(result.connected);
        assert!(result.reanalysis_feasible);
        assert_eq!(
            result.critical_case,
            result
                .final_case_compliances_j
                .iter()
                .enumerate()
                .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
                .map(|(index, _)| envelope.load_cases[index].name.clone())
                .unwrap()
        );
        assert_eq!(
            result.execution.capability_id,
            "sonium-structural.cpu-reference.f64.fem-topology-2d-envelope.v1"
        );
    }

    #[test]
    fn fem_topology_robust_reanalyses_morphology_and_uncertainty_cases() {
        let mut study = FemTopology2DRobustStudy::introductory();
        study.envelope.topology = small_study();
        study.envelope.topology.edge_force_n = study.envelope.load_cases[0].edge_force_n;
        study.allowable_compliance_j = 1.0e6;

        let result = run_fem_topology_2d_robust(&study).expect("robust FEM topology");

        assert_eq!(result.cases.len(), 12);
        assert!(result.factorization_reused_within_scenario);
        assert!(result.morphology_feasible);
        assert!(result.uncertainty_feasible);
        assert!(result.robust_feasible);
        assert!(result.eroded_volume_fraction <= result.nominal_volume_fraction + 1.0e-12);
        assert!(result.dilated_volume_fraction >= result.nominal_volume_fraction - 1.0e-12);
        assert!(result.worst_case_compliance_j.is_finite());
        assert!(result.gradient_taylor_relative_error.is_finite());
        assert!(result.cases.iter().all(|case| {
            case.converged
                && case.solver_residual < 1.0e-7
                && case.free_equilibrium_residual_n < 1.0e-7
                && case.force_balance_residual_n < 1.0e-7
                && case.energy_balance_relative_error < 1.0e-8
        }));
        assert_eq!(
            result.execution.capability_id,
            "sonium-structural.cpu-reference.f64.fem-topology-2d-robust.v1"
        );
    }
}
