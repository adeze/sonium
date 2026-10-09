//! Executable linear static shell study backed by Sonium FEM's triangular shell.
//!
//! The study is intentionally bounded: it uses flat three-node shell elements
//! (CST membrane + DKT bending), homogeneous clamping on the x-minimum edge,
//! and a pressure load in the element normal direction.  The result records
//! the CPU/f64 reference provenance and equilibrium/energy checks so it cannot
//! be mistaken for a general nonlinear, contact, anisotropic, or viscoelastic
//! shell solver.

use crate::{
    StructuralExecutionProvenance, StructuralMechanicsError, StructuralModelAssumption, dot,
    free_residual_norm, triplet_matvec,
};
use serde::{Deserialize, Serialize};
use sonium_fem::boundary::generic::{FemSystem, try_apply_homogeneous_dirichlet_dofs};
use sonium_fem::mesh::{ElementType, Mesh, rectangular_mesh_triangles};
use sonium_fem::physics::solid::shell::{
    ShellElement, ShellSection, assemble_shell_load, assemble_shell_stiffness,
};
use sonium_solvers::{
    MatrixClass, Ordering, SparseDirectConfig, SparseDirectMethod, factorize_sparse,
};

/// Marker retained for callers that want a stable name for the clamped edge.
pub const SHELL_FIXED_BOUNDARY_MARKER: i32 = 1;

/// Marker retained for callers that want a stable name for the pressure-loaded surface.
pub const SHELL_LOADED_BOUNDARY_MARKER: i32 = 2;

const DOFS_PER_NODE: usize = 6;
const VALIDATION_TOLERANCE: f64 = 1.0e-7;

/// Structured flat shell dimensions and mesh density.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ShellGeometry {
    pub length_m: f64,
    pub width_m: f64,
    pub nx: usize,
    pub ny: usize,
}

impl Default for ShellGeometry {
    fn default() -> Self {
        Self {
            length_m: 1.0,
            width_m: 0.6,
            nx: 12,
            ny: 6,
        }
    }
}

impl ShellGeometry {
    pub fn area_m2(self) -> f64 {
        self.length_m * self.width_m
    }

    pub fn mesh(self) -> Mesh {
        rectangular_mesh_triangles(0.0, self.length_m, 0.0, self.width_m, self.nx, self.ny)
    }
}

/// Isotropic, linear-elastic shell section inputs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ShellMaterial {
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
    pub thickness_m: f64,
    pub density_kg_m3: f64,
}

impl Default for ShellMaterial {
    fn default() -> Self {
        Self {
            youngs_modulus_pa: 200.0e9,
            poisson_ratio: 0.3,
            thickness_m: 0.002,
            density_kg_m3: 7800.0,
        }
    }
}

impl ShellMaterial {
    pub(crate) fn section(self) -> ShellSection {
        ShellSection::new(
            self.youngs_modulus_pa,
            self.poisson_ratio,
            self.thickness_m,
            self.density_kg_m3,
        )
    }
}

/// Linear static flat-shell pressure study.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShellStaticStudy {
    pub name: String,
    pub geometry: ShellGeometry,
    pub material: ShellMaterial,
    /// Pressure in Pa, positive along each triangle's outward normal.
    pub pressure_pa: f64,
}

impl Default for ShellStaticStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl ShellStaticStudy {
    /// Construct the clamped rectangular shell reference study.
    pub fn introductory() -> Self {
        Self {
            name: "linear-clamped-flat-shell-pressure".to_string(),
            geometry: ShellGeometry::default(),
            material: ShellMaterial::default(),
            pressure_pa: -1.0e3,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("shell.length_m", self.geometry.length_m),
            ("shell.width_m", self.geometry.width_m),
            ("shell.youngs_modulus_pa", self.material.youngs_modulus_pa),
            ("shell.thickness_m", self.material.thickness_m),
            ("shell.density_kg_m3", self.material.density_kg_m3),
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
                field: "shell.poisson_ratio",
                value: self.material.poisson_ratio.to_string(),
            });
        }
        if !self.pressure_pa.is_finite() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "shell.pressure_pa",
                value: self.pressure_pa.to_string(),
            });
        }
        if self.geometry.nx == 0 || self.geometry.ny == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "shell.mesh_density",
                value: format!(
                    "nx={}, ny={} (both must be non-zero)",
                    self.geometry.nx, self.geometry.ny
                ),
            });
        }
        Ok(())
    }

    pub(crate) fn shell_elements(
        material: ShellMaterial,
        mesh: &Mesh,
    ) -> Result<Vec<ShellElement>, StructuralMechanicsError> {
        let section = material.section();
        let mut elements = Vec::with_capacity(mesh.elements.len());
        for (element_index, element) in mesh.elements.iter().enumerate() {
            if element.element_type != ElementType::Triangle || element.nodes.len() != 3 {
                return Err(StructuralMechanicsError::InvalidMesh {
                    reason: format!(
                        "shell element {element_index} is {:?} with {} nodes; only P1 triangles are supported",
                        element.element_type,
                        element.nodes.len()
                    ),
                });
            }
            let nodes = [element.nodes[0], element.nodes[1], element.nodes[2]];
            if nodes.iter().any(|&node| node >= mesh.nodes.len()) {
                return Err(StructuralMechanicsError::InvalidMesh {
                    reason: format!("shell element {element_index} references a missing node"),
                });
            }
            let p0 = mesh.nodes[nodes[0]];
            let p1 = mesh.nodes[nodes[1]];
            let p2 = mesh.nodes[nodes[2]];
            let cross = [
                (p1.y - p0.y) * (p2.z - p0.z) - (p1.z - p0.z) * (p2.y - p0.y),
                (p1.z - p0.z) * (p2.x - p0.x) - (p1.x - p0.x) * (p2.z - p0.z),
                (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x),
            ];
            let area =
                0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            if !area.is_finite() || area <= f64::EPSILON {
                return Err(StructuralMechanicsError::InvalidMesh {
                    reason: format!("shell element {element_index} has zero area"),
                });
            }
            elements.push(ShellElement::from_mesh(nodes, mesh, section));
        }
        if elements.is_empty() {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: "shell mesh contains no elements".to_string(),
            });
        }
        Ok(elements)
    }

    fn fixed_nodes(&self, mesh: &Mesh) -> Vec<usize> {
        let tolerance = 1.0e-10 * self.geometry.length_m.max(self.geometry.width_m).max(1.0);
        mesh.nodes
            .iter()
            .enumerate()
            .filter_map(|(node, point)| ((point.x - 0.0).abs() <= tolerance).then_some(node))
            .collect()
    }
}

/// Linear static shell response and physical validation metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShellStaticResult {
    pub nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    pub solver_iterations: usize,
    pub solver_residual: f64,
    pub converged: bool,
    pub max_translation_m: f64,
    pub max_transverse_displacement_m: f64,
    pub max_rotation_rad: f64,
    pub applied_force_n: [f64; 3],
    pub reaction_force_n: [f64; 3],
    pub applied_moment_nm: [f64; 3],
    pub reaction_moment_nm: [f64; 3],
    pub total_applied_load_n: f64,
    pub total_reaction_n: f64,
    pub free_equilibrium_residual_n: f64,
    pub force_balance_residual_n: f64,
    /// Resultant moment diagnostic from translational and generalized
    /// rotational support reactions. The current FEM DKT convention is not
    /// used as a hard validation gate for this quantity.
    pub moment_balance_residual_nm: f64,
    pub load_integration_relative_error: f64,
    pub strain_energy_j: f64,
    pub energy_balance_relative_error: f64,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
    pub coordinates: Vec<[f64; 3]>,
    /// Full `[u, v, w, theta_x, theta_y, theta_z]` shell field per node.
    pub displacement: Vec<[f64; 6]>,
}

/// A named pressure case for the bounded shared-factorization shell path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShellMultiLoadCase {
    pub name: String,
    pub pressure_pa: f64,
}

/// Flat P1 shell study with several pressure cases on one stiffness matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShellMultiLoadStudy {
    pub base: ShellStaticStudy,
    pub load_cases: Vec<ShellMultiLoadCase>,
}

impl Default for ShellMultiLoadStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl ShellMultiLoadStudy {
    pub fn introductory() -> Self {
        let base = ShellStaticStudy::introductory();
        Self {
            base,
            load_cases: vec![
                ShellMultiLoadCase {
                    name: "service-pressure".to_string(),
                    pressure_pa: -1.0e3,
                },
                ShellMultiLoadCase {
                    name: "reversal-pressure".to_string(),
                    pressure_pa: 0.75e3,
                },
            ],
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        if self.load_cases.is_empty() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "shell_multi_load.load_cases",
                value: "empty".to_string(),
            });
        }
        let mut base = self.base.clone();
        base.pressure_pa = self.load_cases[0].pressure_pa;
        base.validate()?;
        let mut names = std::collections::HashSet::with_capacity(self.load_cases.len());
        for case in &self.load_cases {
            if case.name.trim().is_empty()
                || !names.insert(case.name.as_str())
                || !case.pressure_pa.is_finite()
                || case.pressure_pa.abs() <= f64::EPSILON
            {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "shell_multi_load.load_cases",
                    value: case.name.clone(),
                });
            }
        }
        Ok(())
    }
}

/// Per-case shell static diagnostics from a shared factorization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShellMultiLoadCaseResult {
    pub name: String,
    pub pressure_pa: f64,
    pub solver_residual: f64,
    pub max_translation_m: f64,
    pub max_transverse_displacement_m: f64,
    pub max_rotation_rad: f64,
    pub applied_force_n: [f64; 3],
    pub reaction_force_n: [f64; 3],
    pub applied_moment_nm: [f64; 3],
    pub reaction_moment_nm: [f64; 3],
    pub free_equilibrium_residual_n: f64,
    pub force_balance_residual_n: f64,
    pub moment_balance_residual_nm: f64,
    pub load_integration_relative_error: f64,
    pub strain_energy_j: f64,
    pub energy_balance_relative_error: f64,
    pub converged: bool,
}

/// Result of a shared-factorization flat-shell multi-load execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShellMultiLoadResult {
    pub study: ShellMultiLoadStudy,
    pub cases: Vec<ShellMultiLoadCaseResult>,
    pub critical_case: String,
    pub max_transverse_displacement_m: f64,
    pub max_rotation_rad: f64,
    pub factorization_memory_bytes: usize,
    pub factorization_reused: bool,
    pub converged: bool,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Assemble, solve, and validate the bounded flat-shell static study.
pub fn run_shell_static(
    study: &ShellStaticStudy,
) -> Result<ShellStaticResult, StructuralMechanicsError> {
    study.validate()?;
    let mesh = study.geometry.mesh();
    let elements = ShellStaticStudy::shell_elements(study.material, &mesh)?;
    let dofs = mesh.nodes.len() * DOFS_PER_NODE;
    let stiffness = assemble_shell_stiffness(&elements, dofs);
    let unconstrained_stiffness = stiffness.clone();
    let rhs = assemble_shell_load(&elements, dofs, |_| study.pressure_pa);
    let external_rhs = rhs.clone();

    let fixed_nodes = study.fixed_nodes(&mesh);
    if fixed_nodes.is_empty() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "shell fixed edge selected no nodes".to_string(),
        });
    }
    let fixed_dofs = fixed_nodes
        .iter()
        .flat_map(|&node| (0..DOFS_PER_NODE).map(move |component| node * DOFS_PER_NODE + component))
        .collect::<Vec<_>>();
    let mut system = FemSystem::new(stiffness, rhs);
    try_apply_homogeneous_dirichlet_dofs(&mut system, &fixed_dofs).map_err(|error| {
        StructuralMechanicsError::SolverFailure {
            stage: "shell static boundary",
            message: error.to_string(),
        }
    })?;

    // Sparse direct backend (mirroring run_shell_multi_load): these systems
    // are small (<= a few thousand DOFs) and the drilling block defeats
    // ILU(0) on fine meshes — GMRES stalls just under its gate while a
    // direct factorization is milliseconds and exact.
    let constrained_csr = system.matrix.to_csr();
    let solution = factorize_sparse(
        &constrained_csr,
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..SparseDirectConfig::default()
        },
        None,
    )
    .and_then(|factor| factor.solve(&system.rhs))
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "shell static solve",
        message: error.to_string(),
    })?;

    let internal = triplet_matvec(&unconstrained_stiffness, &solution);
    let coordinates = mesh
        .nodes
        .iter()
        .map(|point| [point.x, point.y, point.z])
        .collect::<Vec<_>>();
    let (applied_force_n, applied_moment_nm) = resultants(&coordinates, &external_rhs);
    let (reaction_force_n, reaction_moment_nm) =
        fixed_resultants(&coordinates, &internal, &external_rhs, &fixed_nodes);
    let force_balance = norm3(add3(applied_force_n, reaction_force_n));
    let moment_balance = norm3(add3(applied_moment_nm, reaction_moment_nm));
    let total_applied_load_n = applied_force_n[2];
    let total_reaction_n = reaction_force_n[2];
    let load_integration_relative_error =
        (total_applied_load_n - study.pressure_pa * study.geometry.area_m2()).abs()
            / (study.pressure_pa * study.geometry.area_m2())
                .abs()
                .max(1.0);
    let free_equilibrium_residual_n =
        free_residual_norm(&internal, &external_rhs, &system.dirichlet_dofs);
    let strain_energy_j = 0.5 * dot(&solution, &internal);
    let rhs_energy = 0.5 * dot(&solution, &external_rhs);
    let energy_balance_relative_error =
        (strain_energy_j - rhs_energy).abs() / rhs_energy.abs().max(1.0);
    let applied_force_scale = norm3(applied_force_n).max(1.0);

    if !load_integration_relative_error.is_finite()
        || load_integration_relative_error > VALIDATION_TOLERANCE
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "shell.load_integration",
            value: load_integration_relative_error,
        });
    }
    if !free_equilibrium_residual_n.is_finite()
        || free_equilibrium_residual_n > VALIDATION_TOLERANCE * applied_force_scale
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "shell.free_equilibrium",
            value: free_equilibrium_residual_n,
        });
    }
    if !force_balance.is_finite() || force_balance > VALIDATION_TOLERANCE * applied_force_scale {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "shell.force_balance",
            value: force_balance,
        });
    }
    if !energy_balance_relative_error.is_finite() || energy_balance_relative_error > 1.0e-9 {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "shell.energy_identity",
            value: energy_balance_relative_error,
        });
    }

    let displacement = solution
        .as_chunks::<DOFS_PER_NODE>()
        .0
        .iter()
        .map(|chunk| [chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5]])
        .collect::<Vec<_>>();
    let max_translation_m = displacement
        .iter()
        .map(|value| (value[0].powi(2) + value[1].powi(2) + value[2].powi(2)).sqrt())
        .fold(0.0, f64::max);
    let max_transverse_displacement_m = displacement
        .iter()
        .map(|value| value[2].abs())
        .fold(0.0, f64::max);
    let max_rotation_rad = displacement
        .iter()
        .map(|value| (value[3].powi(2) + value[4].powi(2) + value[5].powi(2)).sqrt())
        .fold(0.0, f64::max);

    let solver_residual = csr_residual_norm(&constrained_csr, &solution, &system.rhs);
    Ok(ShellStaticResult {
        nodes: mesh.nodes.len(),
        elements: elements.len(),
        dofs,
        solver_iterations: 1,
        solver_residual,
        converged: true,
        max_translation_m,
        max_transverse_displacement_m,
        max_rotation_rad,
        applied_force_n,
        reaction_force_n,
        applied_moment_nm,
        reaction_moment_nm,
        total_applied_load_n,
        total_reaction_n,
        free_equilibrium_residual_n,
        force_balance_residual_n: force_balance,
        moment_balance_residual_nm: moment_balance,
        load_integration_relative_error,
        strain_energy_j,
        energy_balance_relative_error,
        model_assumption: StructuralModelAssumption::Shell,
        execution: StructuralExecutionProvenance::cpu_shell_reference(),
        coordinates,
        displacement,
    })
}

/// Solve all named shell pressure cases with one constrained sparse
/// factorization and one RHS solve per case. This is the assembled CPU/f64
/// P1 reference; it intentionally does not claim a nonlinear or curved-shell
/// multi-load solver.
pub fn run_shell_multi_load(
    study: &ShellMultiLoadStudy,
) -> Result<ShellMultiLoadResult, StructuralMechanicsError> {
    study.validate()?;
    let mut base = study.base.clone();
    base.pressure_pa = study.load_cases[0].pressure_pa;
    let mesh = base.geometry.mesh();
    let elements = ShellStaticStudy::shell_elements(base.material, &mesh)?;
    let dofs = mesh.nodes.len() * DOFS_PER_NODE;
    let stiffness = assemble_shell_stiffness(&elements, dofs);
    let unconstrained_stiffness = stiffness.clone();
    let fixed_nodes = base.fixed_nodes(&mesh);
    if fixed_nodes.is_empty() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "shell fixed edge selected no nodes".to_string(),
        });
    }
    let fixed_dofs = fixed_nodes
        .iter()
        .flat_map(|&node| (0..DOFS_PER_NODE).map(move |component| node * DOFS_PER_NODE + component))
        .collect::<Vec<_>>();
    let mut zero_system = FemSystem::new(stiffness.clone(), vec![0.0; dofs]);
    try_apply_homogeneous_dirichlet_dofs(&mut zero_system, &fixed_dofs).map_err(|error| {
        StructuralMechanicsError::SolverFailure {
            stage: "shell multi-load boundary",
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
        stage: "shell multi-load factorization",
        message: error.to_string(),
    })?;
    let factorization_memory_bytes = factorization.factor_memory_bytes();
    let coordinates = mesh
        .nodes
        .iter()
        .map(|point| [point.x, point.y, point.z])
        .collect::<Vec<_>>();
    // Cost-aware gate: DOFs × cases against the machine-adaptive grain.
    let cases = crate::parallel_collect_results(
        &study.load_cases,
        dofs.saturating_mul(study.load_cases.len()),
        crate::STRUCTURAL_SOLVE_WORK_PER_THREAD,
        |load_case| -> Result<_, StructuralMechanicsError> {
            let rhs = assemble_shell_load(&elements, dofs, |_| load_case.pressure_pa);
            let external_rhs = rhs.clone();
            let mut system = FemSystem::new(stiffness.clone(), rhs);
            try_apply_homogeneous_dirichlet_dofs(&mut system, &fixed_dofs).map_err(|error| {
                StructuralMechanicsError::SolverFailure {
                    stage: "shell multi-load case boundary",
                    message: error.to_string(),
                }
            })?;
            let solution = factorization.solve(&system.rhs).map_err(|error| {
                StructuralMechanicsError::SolverFailure {
                    stage: "shell multi-load case solve",
                    message: error.to_string(),
                }
            })?;
            let internal = triplet_matvec(&unconstrained_stiffness, &solution);
            let (applied_force_n, applied_moment_nm) = resultants(&coordinates, &external_rhs);
            let (reaction_force_n, reaction_moment_nm) =
                fixed_resultants(&coordinates, &internal, &external_rhs, &fixed_nodes);
            let force_balance_residual_n = norm3(add3(applied_force_n, reaction_force_n));
            let moment_balance_residual_nm = norm3(add3(applied_moment_nm, reaction_moment_nm));
            let load_integration_relative_error =
                (applied_force_n[2] - load_case.pressure_pa * base.geometry.area_m2()).abs()
                    / (load_case.pressure_pa * base.geometry.area_m2())
                        .abs()
                        .max(1.0);
            let free_equilibrium_residual_n =
                free_residual_norm(&internal, &external_rhs, &system.dirichlet_dofs);
            let strain_energy_j = 0.5 * dot(&solution, &internal);
            let rhs_energy = 0.5 * dot(&solution, &external_rhs);
            let energy_balance_relative_error =
                (strain_energy_j - rhs_energy).abs() / rhs_energy.abs().max(1.0);
            let solver_residual = csr_residual_norm(&constrained_csr, &solution, &system.rhs);
            let displacement = solution
                .as_chunks::<DOFS_PER_NODE>()
                .0
                .iter()
                .map(|chunk| [chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5]])
                .collect::<Vec<_>>();
            let max_translation_m = displacement
                .iter()
                .map(|value| (value[0].powi(2) + value[1].powi(2) + value[2].powi(2)).sqrt())
                .fold(0.0, f64::max);
            let max_transverse_displacement_m = displacement
                .iter()
                .map(|value| value[2].abs())
                .fold(0.0, f64::max);
            let max_rotation_rad = displacement
                .iter()
                .map(|value| (value[3].powi(2) + value[4].powi(2) + value[5].powi(2)).sqrt())
                .fold(0.0, f64::max);
            let applied_force_scale = norm3(applied_force_n).max(1.0);
            let converged = solver_residual.is_finite()
                && free_equilibrium_residual_n.is_finite()
                && force_balance_residual_n.is_finite()
                && moment_balance_residual_nm.is_finite()
                && load_integration_relative_error.is_finite()
                && energy_balance_relative_error.is_finite()
                && solver_residual <= 1.0e-9
                && free_equilibrium_residual_n <= VALIDATION_TOLERANCE * applied_force_scale
                && force_balance_residual_n <= VALIDATION_TOLERANCE * applied_force_scale
                && load_integration_relative_error <= VALIDATION_TOLERANCE
                && energy_balance_relative_error <= 1.0e-9;
            Ok(ShellMultiLoadCaseResult {
                name: load_case.name.clone(),
                pressure_pa: load_case.pressure_pa,
                solver_residual,
                max_translation_m,
                max_transverse_displacement_m,
                max_rotation_rad,
                applied_force_n,
                reaction_force_n,
                applied_moment_nm,
                reaction_moment_nm,
                free_equilibrium_residual_n,
                force_balance_residual_n,
                moment_balance_residual_nm,
                load_integration_relative_error,
                strain_energy_j,
                energy_balance_relative_error,
                converged,
            })
        },
    )?;
    let critical_case = cases
        .iter()
        .max_by(|lhs, rhs| {
            lhs.max_transverse_displacement_m
                .total_cmp(&rhs.max_transverse_displacement_m)
        })
        .map(|case| case.name.clone())
        .ok_or(StructuralMechanicsError::InvalidConfiguration {
            field: "shell_multi_load.load_cases",
            value: "empty".to_string(),
        })?;
    Ok(ShellMultiLoadResult {
        study: study.clone(),
        max_transverse_displacement_m: cases
            .iter()
            .map(|case| case.max_transverse_displacement_m)
            .fold(0.0, f64::max),
        max_rotation_rad: cases
            .iter()
            .map(|case| case.max_rotation_rad)
            .fold(0.0, f64::max),
        factorization_memory_bytes,
        factorization_reused: true,
        converged: cases.iter().all(|case| case.converged),
        model_assumption: StructuralModelAssumption::Shell,
        execution: StructuralExecutionProvenance::cpu_shell_multi_load_reference(),
        cases,
        critical_case,
    })
}

fn csr_residual_norm(
    matrix: &sonium_solvers::CsrMatrix<f64>,
    solution: &[f64],
    rhs: &[f64],
) -> f64 {
    let residual = (0..matrix.num_rows)
        .map(|row| {
            let value = (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                .map(|index| matrix.values[index] * solution[matrix.col_indices[index]])
                .sum::<f64>();
            value - rhs[row]
        })
        .collect::<Vec<_>>();
    dot(&residual, &residual).sqrt() / dot(rhs, rhs).sqrt().max(1.0)
}

fn resultants(coordinates: &[[f64; 3]], vector: &[f64]) -> ([f64; 3], [f64; 3]) {
    let mut force = [0.0; 3];
    let mut moment = [0.0; 3];
    for (node, point) in coordinates.iter().enumerate() {
        let base = node * DOFS_PER_NODE;
        let nodal_force = [vector[base], vector[base + 1], vector[base + 2]];
        force = add3(force, nodal_force);
        moment = add3(moment, cross(*point, nodal_force));
    }
    (force, moment)
}

fn fixed_resultants(
    coordinates: &[[f64; 3]],
    internal: &[f64],
    external: &[f64],
    fixed_nodes: &[usize],
) -> ([f64; 3], [f64; 3]) {
    let mut force = [0.0; 3];
    let mut moment = [0.0; 3];
    for &node in fixed_nodes {
        let base = node * DOFS_PER_NODE;
        let nodal_force = [
            internal[base] - external[base],
            internal[base + 1] - external[base + 1],
            internal[base + 2] - external[base + 2],
        ];
        force = add3(force, nodal_force);
        moment = add3(moment, cross(coordinates[node], nodal_force));
        moment[0] += internal[base + 3] - external[base + 3];
        moment[1] += internal[base + 4] - external[base + 4];
        moment[2] += internal[base + 5] - external[base + 5];
    }
    (force, moment)
}

fn add3(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

fn cross(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

fn norm3(value: [f64; 3]) -> f64 {
    (value[0].powi(2) + value[1].powi(2) + value[2].powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_static_has_equilibrium_and_energy_evidence() {
        let mut study = ShellStaticStudy::introductory();
        study.geometry.nx = 4;
        study.geometry.ny = 2;
        let result = run_shell_static(&study).expect("shell static study should solve");

        assert!(result.converged);
        assert_eq!(result.model_assumption, StructuralModelAssumption::Shell);
        assert_eq!(
            result.execution.capability_id,
            "sonium-structural.cpu-reference.f64.shell-tri3-cst-dkt.v1"
        );
        assert!(result.max_transverse_displacement_m > 0.0);
        assert!(result.force_balance_residual_n < 1.0e-7);
        assert!(result.moment_balance_residual_nm.is_finite());
        assert!(result.energy_balance_relative_error < 1.0e-9);
        assert!(result.load_integration_relative_error < 1.0e-12);
    }

    #[test]
    fn shell_cst_membrane_patch_matches_constitutive_energy() {
        let section = ShellSection::new(2.0e6, 0.25, 0.1, 1.0);
        let element = ShellElement::from_points(
            [0, 1, 2],
            &sonium_fem::mesh::Point::new_3d(0.0, 0.0, 0.0),
            &sonium_fem::mesh::Point::new_3d(1.0, 0.0, 0.0),
            &sonium_fem::mesh::Point::new_3d(0.0, 1.0, 0.0),
            section,
        );
        let strain = [1.0e-3, -2.0e-3, 3.0e-3];
        let displacement = [
            0.0,
            0.0,
            strain[0],
            0.5 * strain[2],
            0.5 * strain[2],
            strain[1],
        ];
        let stiffness = element.membrane_stiffness_local();
        let energy = 0.5
            * displacement
                .iter()
                .enumerate()
                .map(|(i, value)| {
                    value
                        * displacement
                            .iter()
                            .enumerate()
                            .map(|(j, other)| stiffness[i * 6 + j] * other)
                            .sum::<f64>()
                })
                .sum::<f64>();
        let d = section.membrane_constitutive();
        let expected = 0.5
            * element.area
            * strain
                .iter()
                .enumerate()
                .map(|(i, value)| {
                    value
                        * strain
                            .iter()
                            .enumerate()
                            .map(|(j, other)| d[i * 3 + j] * other)
                            .sum::<f64>()
                })
                .sum::<f64>();
        assert!((energy - expected).abs() / expected.abs() < 1.0e-12);
    }

    #[test]
    fn shell_multi_load_reuses_one_factorization_and_preserves_balance() {
        let mut study = ShellMultiLoadStudy::introductory();
        study.base.geometry.nx = 4;
        study.base.geometry.ny = 2;
        let result = run_shell_multi_load(&study).expect("shell multi-load study should solve");

        assert_eq!(result.cases.len(), 2);
        assert!(result.factorization_reused);
        assert!(result.factorization_memory_bytes > 0);
        assert!(result.converged);
        assert!(result.cases.iter().all(|case| {
            case.solver_residual < 1.0e-9
                && case.free_equilibrium_residual_n < 1.0e-6
                && case.force_balance_residual_n < 1.0e-6
                && case.load_integration_relative_error < 1.0e-7
                && case.energy_balance_relative_error < 1.0e-9
                && case.max_transverse_displacement_m > 0.0
        }));
        assert!(!result.critical_case.is_empty());
        assert_eq!(
            result.execution.capability_id,
            "sonium-structural.cpu-reference.f64.shell-tri3-cst-dkt-multi-load.v1"
        );
    }

    #[test]
    fn shell_rejects_invalid_material_and_mesh_density() {
        let mut study = ShellStaticStudy::introductory();
        study.material.poisson_ratio = 0.5;
        assert!(matches!(
            run_shell_static(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "shell.poisson_ratio",
                ..
            })
        ));

        let mut study = ShellStaticStudy::introductory();
        study.geometry.nx = 0;
        assert!(matches!(
            run_shell_static(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "shell.mesh_density",
                ..
            })
        ));
    }
}
