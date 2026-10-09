//! Executable structural study families backed by Sonium FEM.
//!
//! The 3D solid family is intentionally independent from the introductory
//! bracket implementation.  It owns its mesh, boundary-marker policy,
//! traction integration, stress recovery, and modal result provenance so
//! later shell, beam, and composite families can be added without turning the
//! bracket solver into a monolith.

use super::{
    DampingConvention, ModalNormalization, StructuralExecutionProvenance, StructuralMaterial,
    StructuralMechanicsError, StructuralModelAssumption, StructuralPolynomialOrder,
};
use serde::{Deserialize, Serialize};
use sonium_fem::assembly::assemble_bilinear_checked;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_homogeneous_dirichlet};
use sonium_fem::mesh::{BoundaryType, Mesh, MeshTopology, Point, box_mesh_tetrahedra};
use sonium_fem::physics::solid::eigenmodes::{
    StructuralModalConfig, build_structural_modal_problem,
};
use sonium_fem::physics::solid::elasticity::ElasticityForm;
use sonium_fem::solver::MatrixFreeOperator;
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::traits::LinearOperator;
use sonium_solvers::{
    AmgConfig, AmgPreconditioner, CgConfig, DiagonalPreconditioner, FgmresConfig, LobpcgConfig,
    MatrixClass, Ordering, SparseDirectConfig, SparseDirectMethod, cg, factorize_sparse,
    fgmres_preconditioned,
};

/// Marker assigned to the clamped x-minimum face of the reference solid.
pub const SOLID3D_FIXED_BOUNDARY_MARKER: i32 = 1;
/// Marker assigned to the loaded x-maximum face of the reference solid.
pub const SOLID3D_LOADED_BOUNDARY_MARKER: i32 = 2;

/// Dimensions and resolution for a rectangular tetrahedral solid.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Solid3DGeometry {
    pub length_m: f64,
    pub width_m: f64,
    pub height_m: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl Default for Solid3DGeometry {
    fn default() -> Self {
        Self {
            length_m: 0.12,
            width_m: 0.04,
            height_m: 0.04,
            nx: 3,
            ny: 2,
            nz: 2,
        }
    }
}

impl Solid3DGeometry {
    pub fn loaded_face_area_m2(self) -> f64 {
        self.width_m * self.height_m
    }

    pub fn volume_m3(self) -> f64 {
        self.length_m * self.width_m * self.height_m
    }
}

/// Total force applied to the x-maximum face of a solid.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Solid3DLoads {
    pub force_n: [f64; 3],
    pub fixed_boundary_marker: i32,
    pub loaded_boundary_marker: i32,
}

impl Default for Solid3DLoads {
    fn default() -> Self {
        Self {
            force_n: [100.0, 0.0, 0.0],
            fixed_boundary_marker: SOLID3D_FIXED_BOUNDARY_MARKER,
            loaded_boundary_marker: SOLID3D_LOADED_BOUNDARY_MARKER,
        }
    }
}

/// Complete validated 3D solid study configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DStudy {
    pub name: String,
    pub geometry: Solid3DGeometry,
    pub material: StructuralMaterial,
    pub loads: Solid3DLoads,
    pub modal_modes: usize,
}

/// Linear solver/operator representation for a 3D solid static solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Solid3DStaticSolver {
    /// Assemble the sparse stiffness matrix and solve with unpreconditioned CG.
    AssembledCg,
    /// Assemble the sparse stiffness matrix and solve with elasticity AMG-FGMRES.
    AssembledAmgFgmres,
    /// Reassemble element action on every iteration and use diagonal FGMRES.
    MatrixFreeJacobiFgmres,
}

/// Order and solver selection for a 3D solid static solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Solid3DStaticOptions {
    pub polynomial_order: StructuralPolynomialOrder,
    pub solver: Solid3DStaticSolver,
}

impl Default for Solid3DStaticOptions {
    fn default() -> Self {
        Self {
            polynomial_order: StructuralPolynomialOrder::Linear,
            solver: Solid3DStaticSolver::AssembledCg,
        }
    }
}

impl Solid3DStudy {
    pub fn introductory() -> Self {
        Self {
            name: "introductory-3d-solid".to_string(),
            geometry: Solid3DGeometry::default(),
            material: StructuralMaterial::default(),
            loads: Solid3DLoads::default(),
            modal_modes: 3,
        }
    }

    /// Build and mark the reference tetrahedral mesh after validating inputs.
    pub fn mesh(&self) -> Result<Mesh, StructuralMechanicsError> {
        self.validate()?;
        Ok(self.build_mesh())
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        let geometry = self.geometry;
        for (field, value) in [
            ("solid3d.length_m", geometry.length_m),
            ("solid3d.width_m", geometry.width_m),
            ("solid3d.height_m", geometry.height_m),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if geometry.nx == 0 || geometry.ny == 0 || geometry.nz == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d.mesh_density",
                value: format!("nx={}, ny={}, nz={}", geometry.nx, geometry.ny, geometry.nz),
            });
        }
        let material = self.material;
        if !material.youngs_modulus_pa.is_finite() || material.youngs_modulus_pa <= 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d.youngs_modulus_pa",
                value: material.youngs_modulus_pa.to_string(),
            });
        }
        if !material.poisson_ratio.is_finite()
            || material.poisson_ratio <= -1.0
            || material.poisson_ratio >= 0.5
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d.poisson_ratio",
                value: material.poisson_ratio.to_string(),
            });
        }
        if !material.density_kg_m3.is_finite() || material.density_kg_m3 <= 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d.density_kg_m3",
                value: material.density_kg_m3.to_string(),
            });
        }
        if !material.damping_ratio.is_finite() || material.damping_ratio < 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d.damping_ratio",
                value: material.damping_ratio.to_string(),
            });
        }
        if self.modal_modes == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d.modal_modes",
                value: "0".to_string(),
            });
        }
        if self.loads.fixed_boundary_marker == self.loads.loaded_boundary_marker {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d.boundary_markers",
                value: "fixed and loaded markers must differ".to_string(),
            });
        }
        if self.loads.force_n.iter().any(|value| !value.is_finite())
            || self
                .loads
                .force_n
                .iter()
                .all(|value| value.abs() <= f64::EPSILON)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d.force_n",
                value: format!("{:?}", self.loads.force_n),
            });
        }
        Ok(())
    }

    fn build_mesh(&self) -> Mesh {
        let mut mesh = box_mesh_tetrahedra(
            0.0,
            self.geometry.length_m,
            0.0,
            self.geometry.width_m,
            0.0,
            self.geometry.height_m,
            self.geometry.nx,
            self.geometry.ny,
            self.geometry.nz,
        );
        let scale = self
            .geometry
            .length_m
            .max(self.geometry.width_m)
            .max(self.geometry.height_m);
        let tolerance = 1e-12 * scale.max(1.0);
        let x_min = 0.0;
        let x_max = self.geometry.length_m;
        mesh.set_boundary_condition(
            BoundaryType::Dirichlet,
            self.loads.fixed_boundary_marker,
            move |points| {
                points
                    .iter()
                    .all(|point| (point.x - x_min).abs() <= tolerance)
            },
        );
        mesh.set_boundary_condition(
            BoundaryType::Neumann,
            self.loads.loaded_boundary_marker,
            move |points| {
                points
                    .iter()
                    .all(|point| (point.x - x_max).abs() <= tolerance)
            },
        );
        mesh
    }
}

/// Validated static response of a 3D solid.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DStaticResult {
    pub nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    pub solver_iterations: usize,
    pub solver_residual: f64,
    pub converged: bool,
    pub max_displacement_m: f64,
    pub max_von_mises_pa: f64,
    pub strain_energy_j: f64,
    pub applied_force_n: [f64; 3],
    pub reaction_force_n: [f64; 3],
    pub free_equilibrium_residual_n: f64,
    pub force_balance_residual_n: f64,
    pub energy_balance_relative_error: f64,
    pub model_assumption: StructuralModelAssumption,
    pub polynomial_order: StructuralPolynomialOrder,
    pub solver: Solid3DStaticSolver,
    pub execution: StructuralExecutionProvenance,
    pub displacement: Vec<[f64; 3]>,
    pub element_von_mises_pa: Vec<f64>,
}

/// A named total traction resultant for the bounded 3D multi-load reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DMultiLoadCase {
    pub name: String,
    pub force_n: [f64; 3],
}

/// P1 assembled 3D solid study with several load cases on one shared mesh and
/// stiffness operator. The fixed and loaded boundary markers come from the
/// embedded [`Solid3DStudy`] policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DMultiLoadStudy {
    pub base: Solid3DStudy,
    pub load_cases: Vec<Solid3DMultiLoadCase>,
}

impl Default for Solid3DMultiLoadStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl Solid3DMultiLoadStudy {
    pub fn introductory() -> Self {
        let mut base = Solid3DStudy::introductory();
        let load_cases = vec![
            Solid3DMultiLoadCase {
                name: "axial-service".to_string(),
                force_n: [100.0, 0.0, 0.0],
            },
            Solid3DMultiLoadCase {
                name: "axial-reversal".to_string(),
                force_n: [-80.0, 20.0, 0.0],
            },
        ];
        base.loads.force_n = load_cases[0].force_n;
        Self { base, load_cases }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        if self.load_cases.is_empty() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_multi_load.load_cases",
                value: "empty".to_string(),
            });
        }
        let mut base = self.base.clone();
        base.loads.force_n = self.load_cases[0].force_n;
        base.validate()?;
        let mut names = std::collections::HashSet::with_capacity(self.load_cases.len());
        for case in &self.load_cases {
            if case.name.trim().is_empty()
                || !names.insert(case.name.as_str())
                || case.force_n.iter().any(|value| !value.is_finite())
                || case.force_n.iter().all(|value| value.abs() <= f64::EPSILON)
            {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "solid3d_multi_load.load_cases",
                    value: case.name.to_string(),
                });
            }
        }
        Ok(())
    }
}

/// Static diagnostics for one load case in a shared-factorization 3D solve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DMultiLoadCaseResult {
    pub name: String,
    pub force_n: [f64; 3],
    pub solver_residual: f64,
    pub max_displacement_m: f64,
    pub max_von_mises_pa: f64,
    pub strain_energy_j: f64,
    pub applied_force_n: [f64; 3],
    pub reaction_force_n: [f64; 3],
    pub free_equilibrium_residual_n: f64,
    pub force_balance_residual_n: f64,
    pub energy_balance_relative_error: f64,
    pub converged: bool,
}

/// Result of a shared-factorization 3D solid multi-load execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DMultiLoadResult {
    pub study: Solid3DMultiLoadStudy,
    pub cases: Vec<Solid3DMultiLoadCaseResult>,
    pub critical_case: String,
    pub max_displacement_m: f64,
    pub max_von_mises_pa: f64,
    pub factorization_memory_bytes: usize,
    pub factorization_reused: bool,
    pub converged: bool,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Generalized 3D solid modal response with reconstructed full-field modes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DModalResult {
    pub eigenvalues: Vec<f64>,
    pub frequencies_hz: Vec<f64>,
    pub residual_norms: Vec<f64>,
    pub solver_iterations: usize,
    pub converged: bool,
    pub normalization: ModalNormalization,
    pub mass_normalization_error: f64,
    pub orthogonality_error: f64,
    pub damping_ratio: f64,
    pub damping_convention: DampingConvention,
    pub model_assumption: StructuralModelAssumption,
    pub modes: Vec<Vec<[f64; 3]>>,
}

/// Solve the validated 3D solid static problem.
pub fn run_solid3d_static(
    study: &Solid3DStudy,
) -> Result<Solid3DStaticResult, StructuralMechanicsError> {
    run_solid3d_static_with_options(study, Solid3DStaticOptions::default())
}

/// Solve a 3D solid static problem at the requested polynomial order.
///
/// Quadratic order selects the matrix-free CPU reference path. The explicit
/// selection makes it impossible for a caller to infer accelerator execution
/// from the presence of a high-order solve.
pub fn run_solid3d_static_order(
    study: &Solid3DStudy,
    polynomial_order: StructuralPolynomialOrder,
) -> Result<Solid3DStaticResult, StructuralMechanicsError> {
    run_solid3d_static_with_options(
        study,
        Solid3DStaticOptions {
            polynomial_order,
            solver: match polynomial_order {
                StructuralPolynomialOrder::Linear => Solid3DStaticSolver::AssembledCg,
                StructuralPolynomialOrder::Quadratic => Solid3DStaticSolver::MatrixFreeJacobiFgmres,
            },
        },
    )
}

/// Solve a 3D solid static problem using the selected CPU/f64 execution path.
pub fn run_solid3d_static_with_options(
    study: &Solid3DStudy,
    options: Solid3DStaticOptions,
) -> Result<Solid3DStaticResult, StructuralMechanicsError> {
    match options.solver {
        Solid3DStaticSolver::MatrixFreeJacobiFgmres => {
            run_solid3d_static_matrix_free(study, options.polynomial_order)
        }
        Solid3DStaticSolver::AssembledCg | Solid3DStaticSolver::AssembledAmgFgmres => {
            run_solid3d_static_assembled(study, options)
        }
    }
}

/// Solve all named P1 3D solid load cases using one constrained sparse
/// factorization and one right-hand-side solve per case. This is deliberately
/// an assembled CPU/f64 reference path; matrix-free and quadratic execution
/// remain single-case APIs until a block-RHS contract is available.
pub fn run_solid3d_multi_load(
    study: &Solid3DMultiLoadStudy,
) -> Result<Solid3DMultiLoadResult, StructuralMechanicsError> {
    study.validate()?;
    let mut base = study.base.clone();
    base.loads.force_n = study.load_cases[0].force_n;
    let mesh = base.mesh()?;
    let polynomial_order = StructuralPolynomialOrder::Linear;
    let space = VectorH1Space::new(polynomial_order.as_fem(), 3);
    let dofs = space.num_global_dofs(&mesh);
    let form = ElasticityForm::new_3d(base.material.as_fem());
    let stiffness =
        assemble_bilinear_checked(&form, &space, &space, &mesh, polynomial_order.degree() + 1)
            .map_err(|error| StructuralMechanicsError::SolverFailure {
                stage: "solid-3d multi-load assembly",
                message: error.to_string(),
            })?;
    let unconstrained_stiffness = stiffness.clone();
    let fixed_dofs = space.boundary_dofs(&mesh, base.loads.fixed_boundary_marker);
    let mut zero_system = FemSystem::new(stiffness.clone(), vec![0.0; dofs]);
    try_apply_homogeneous_dirichlet(
        &mut zero_system,
        &mesh,
        &space,
        base.loads.fixed_boundary_marker,
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "solid-3d multi-load boundary",
        message: error.to_string(),
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
        stage: "solid-3d multi-load factorization",
        message: error.to_string(),
    })?;
    let factorization_memory_bytes = factorization.factor_memory_bytes();
    // Cost-aware gate: DOFs × cases against the machine-adaptive grain.
    let cases = super::parallel_collect_results(
        &study.load_cases,
        dofs.saturating_mul(study.load_cases.len()),
        super::STRUCTURAL_SOLVE_WORK_PER_THREAD,
        |load_case| -> Result<_, StructuralMechanicsError> {
            let mut system = FemSystem::new(stiffness.clone(), vec![0.0; dofs]);
            apply_solid3d_traction(
                &mut system.rhs,
                &mesh,
                base.loads.loaded_boundary_marker,
                load_case.force_n,
                base.geometry.loaded_face_area_m2(),
                polynomial_order,
            )?;
            let external_rhs = system.rhs.clone();
            try_apply_homogeneous_dirichlet(
                &mut system,
                &mesh,
                &space,
                base.loads.fixed_boundary_marker,
            )
            .map_err(|error| StructuralMechanicsError::SolverFailure {
                stage: "solid-3d multi-load case boundary",
                message: error.to_string(),
            })?;
            let solution = factorization.solve(&system.rhs).map_err(|error| {
                StructuralMechanicsError::SolverFailure {
                    stage: "solid-3d multi-load case solve",
                    message: error.to_string(),
                }
            })?;
            let internal_force = super::triplet_matvec(&unconstrained_stiffness, &solution);
            let applied_force_n = component_force_3(&external_rhs);
            let reaction_force_n = reaction_force_3(&internal_force, &external_rhs, &fixed_dofs);
            let free_equilibrium_residual_n =
                free_residual_norm(&internal_force, &external_rhs, &system.dirichlet_dofs);
            let force_balance_residual_n = norm3(add3(applied_force_n, reaction_force_n));
            let internal_energy = 0.5 * super::dot(&solution, &internal_force);
            let rhs_energy = 0.5 * super::dot(&solution, &external_rhs);
            let energy_balance_relative_error =
                (internal_energy - rhs_energy).abs() / rhs_energy.abs().max(1.0);
            let solver_residual = csr_residual_norm(&constrained_csr, &solution, &system.rhs);
            let displacement = solution
                .as_chunks::<3>()
                .0
                .iter()
                .take(mesh.num_nodes())
                .map(|chunk| [chunk[0], chunk[1], chunk[2]])
                .collect::<Vec<_>>();
            let element_von_mises_pa = element_von_mises(&mesh, &base, &displacement);
            let max_displacement_m = displacement.iter().copied().map(norm3).fold(0.0, f64::max);
            let max_von_mises_pa = element_von_mises_pa.iter().copied().fold(0.0, f64::max);
            let force_scale = norm3(applied_force_n).max(1.0);
            let converged = solver_residual.is_finite()
                && free_equilibrium_residual_n.is_finite()
                && force_balance_residual_n.is_finite()
                && energy_balance_relative_error.is_finite()
                && solver_residual <= 1.0e-9
                && free_equilibrium_residual_n <= 1.0e-6 * force_scale
                && force_balance_residual_n <= 1.0e-6 * force_scale
                && energy_balance_relative_error <= 1.0e-9;
            Ok(Solid3DMultiLoadCaseResult {
                name: load_case.name.clone(),
                force_n: load_case.force_n,
                solver_residual,
                max_displacement_m,
                max_von_mises_pa,
                strain_energy_j: internal_energy,
                applied_force_n,
                reaction_force_n,
                free_equilibrium_residual_n,
                force_balance_residual_n,
                energy_balance_relative_error,
                converged,
            })
        },
    )?;
    let critical_case = cases
        .iter()
        .max_by(|lhs, rhs| lhs.max_von_mises_pa.total_cmp(&rhs.max_von_mises_pa))
        .map(|case| case.name.clone())
        .ok_or(StructuralMechanicsError::InvalidConfiguration {
            field: "solid3d_multi_load.load_cases",
            value: "empty".to_string(),
        })?;
    Ok(Solid3DMultiLoadResult {
        study: study.clone(),
        max_displacement_m: cases
            .iter()
            .map(|case| case.max_displacement_m)
            .fold(0.0, f64::max),
        max_von_mises_pa: cases
            .iter()
            .map(|case| case.max_von_mises_pa)
            .fold(0.0, f64::max),
        factorization_memory_bytes,
        factorization_reused: true,
        converged: cases.iter().all(|case| case.converged),
        model_assumption: StructuralModelAssumption::Solid3D,
        execution: StructuralExecutionProvenance::cpu_solid3d_multi_load_reference(),
        cases,
        critical_case,
    })
}

fn run_solid3d_static_assembled(
    study: &Solid3DStudy,
    options: Solid3DStaticOptions,
) -> Result<Solid3DStaticResult, StructuralMechanicsError> {
    if options.solver == Solid3DStaticSolver::MatrixFreeJacobiFgmres {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "solid3d_static.solver",
            value: "matrix-free solver must use the matrix-free execution path".to_string(),
        });
    }
    let mesh = study.mesh()?;
    let space = VectorH1Space::new(options.polynomial_order.as_fem(), 3);
    let dofs = space.num_global_dofs(&mesh);
    let form = ElasticityForm::new_3d(study.material.as_fem());
    let stiffness = assemble_bilinear_checked(
        &form,
        &space,
        &space,
        &mesh,
        options.polynomial_order.degree() + 1,
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "solid-3d assembly",
        message: error.to_string(),
    })?;
    let unconstrained_stiffness = stiffness.clone();
    let mut system = FemSystem::new(stiffness, vec![0.0; dofs]);
    apply_solid3d_traction(
        &mut system.rhs,
        &mesh,
        study.loads.loaded_boundary_marker,
        study.loads.force_n,
        study.geometry.loaded_face_area_m2(),
        options.polynomial_order,
    )?;
    let external_rhs = system.rhs.clone();
    try_apply_homogeneous_dirichlet(
        &mut system,
        &mesh,
        &space,
        study.loads.fixed_boundary_marker,
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "solid-3d boundary",
        message: error.to_string(),
    })?;

    let csr = system.matrix.to_csr();
    let (solution, solver_iterations, solver_residual, converged) = match options.solver {
        Solid3DStaticSolver::AssembledCg => {
            let solved = cg(
                &csr,
                &system.rhs,
                &CgConfig {
                    max_iterations: 12_000,
                    tolerance: 1e-10,
                    print_interval: 0,
                },
            );
            (
                solved.x,
                solved.iterations,
                solved.residual,
                solved.converged,
            )
        }
        Solid3DStaticSolver::AssembledAmgFgmres => {
            let amg = AmgPreconditioner::from_csr(&csr, AmgConfig::for_elasticity(3));
            let solved = fgmres_preconditioned(
                &csr,
                &amg,
                &system.rhs,
                None,
                &FgmresConfig {
                    max_iterations: 12_000,
                    restart: 50,
                    tolerance: 1e-10,
                    print_interval: 0,
                    convergence_check_interval: 1,
                    exact_residual_check_interval: 1,
                },
            );
            (
                solved.x,
                solved.iterations,
                solved.residual,
                solved.converged,
            )
        }
        Solid3DStaticSolver::MatrixFreeJacobiFgmres => {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_static.solver",
                value: "matrix-free solver must use the matrix-free execution path".to_string(),
            });
        }
    };
    if !converged {
        return Err(StructuralMechanicsError::StaticNonConvergence {
            model: StructuralModelAssumption::Solid3D,
            iterations: solver_iterations,
            residual: solver_residual,
        });
    }

    let displacement = solution
        .as_chunks::<3>()
        .0
        .iter()
        .take(mesh.num_nodes())
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect::<Vec<_>>();
    let internal_force = super::triplet_matvec(&unconstrained_stiffness, &solution);
    let fixed_dofs = space.boundary_dofs(&mesh, study.loads.fixed_boundary_marker);
    let applied_force_n = component_force_3(&external_rhs);
    let reaction_force_n = reaction_force_3(&internal_force, &external_rhs, &fixed_dofs);
    let free_equilibrium_residual_n =
        free_residual_norm(&internal_force, &external_rhs, &system.dirichlet_dofs);
    let force_balance_residual_n = norm3(add3(applied_force_n, reaction_force_n));
    let internal_energy = 0.5 * super::dot(&solution, &internal_force);
    let rhs_energy = 0.5 * super::dot(&solution, &external_rhs);
    let energy_balance_relative_error =
        (internal_energy - rhs_energy).abs() / rhs_energy.abs().max(1.0);
    let force_scale = norm3(applied_force_n).max(1.0);
    if !free_equilibrium_residual_n.is_finite() || free_equilibrium_residual_n > 1e-6 * force_scale
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d.free_equilibrium",
            value: free_equilibrium_residual_n,
        });
    }
    if !force_balance_residual_n.is_finite() || force_balance_residual_n > 1e-6 * force_scale {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d.force_balance",
            value: force_balance_residual_n,
        });
    }
    if !energy_balance_relative_error.is_finite() || energy_balance_relative_error > 1e-9 {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d.energy_identity",
            value: energy_balance_relative_error,
        });
    }

    let element_von_mises_pa = if options.polynomial_order == StructuralPolynomialOrder::Linear {
        element_von_mises(&mesh, study, &displacement)
    } else {
        element_von_mises_order(&mesh, study, &space, &solution)
    };
    Ok(Solid3DStaticResult {
        nodes: mesh.num_nodes(),
        elements: mesh.num_elements(),
        dofs,
        solver_iterations,
        solver_residual,
        converged,
        max_displacement_m: displacement
            .iter()
            .map(|value| norm3(*value))
            .fold(0.0, f64::max),
        max_von_mises_pa: element_von_mises_pa.iter().copied().fold(0.0, f64::max),
        strain_energy_j: internal_energy,
        applied_force_n,
        reaction_force_n,
        free_equilibrium_residual_n,
        force_balance_residual_n,
        energy_balance_relative_error,
        model_assumption: StructuralModelAssumption::Solid3D,
        polynomial_order: options.polynomial_order,
        solver: options.solver,
        execution: StructuralExecutionProvenance::cpu_reference(),
        displacement,
        element_von_mises_pa,
    })
}

fn run_solid3d_static_matrix_free(
    study: &Solid3DStudy,
    polynomial_order: StructuralPolynomialOrder,
) -> Result<Solid3DStaticResult, StructuralMechanicsError> {
    let mesh = study.mesh()?;
    let space = VectorH1Space::new(polynomial_order.as_fem(), 3);
    let dofs = space.num_global_dofs(&mesh);
    let form = ElasticityForm::new_3d(study.material.as_fem());
    let mut external_rhs = vec![0.0; dofs];
    apply_solid3d_traction(
        &mut external_rhs,
        &mesh,
        study.loads.loaded_boundary_marker,
        study.loads.force_n,
        study.geometry.loaded_face_area_m2(),
        polynomial_order,
    )?;

    let fixed_dofs = space.boundary_dofs(&mesh, study.loads.fixed_boundary_marker);
    let dirichlet_dofs = fixed_dofs
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let mut solver_rhs = external_rhs.clone();
    for &dof in &fixed_dofs {
        solver_rhs[dof] = 0.0;
    }

    let base = MatrixFreeOperator::new(&form, &space, &mesh, polynomial_order.degree() + 1);
    let mut diagonal = base.diagonal();
    for &dof in &fixed_dofs {
        diagonal[dof] = 1.0;
    }
    let operator = ConstrainedMatrixFreeOperator {
        base,
        dirichlet_dofs: dirichlet_dofs.clone(),
    };
    let preconditioner = DiagonalPreconditioner::from_diagonal(&diagonal);
    let solved = fgmres_preconditioned(
        &operator,
        &preconditioner,
        &solver_rhs,
        None,
        &FgmresConfig {
            max_iterations: 12_000,
            restart: 50,
            tolerance: 1e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
        },
    );
    if !solved.converged {
        return Err(StructuralMechanicsError::StaticNonConvergence {
            model: StructuralModelAssumption::Solid3D,
            iterations: solved.iterations,
            residual: solved.residual,
        });
    }

    let internal_force = operator.apply_unconstrained(&solved.x);
    let displacement = solved
        .x
        .as_chunks::<3>()
        .0
        .iter()
        .take(mesh.num_nodes())
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect::<Vec<_>>();
    let applied_force_n = component_force_3(&external_rhs);
    let reaction_force_n = reaction_force_3(&internal_force, &external_rhs, &fixed_dofs);
    let free_equilibrium_residual_n =
        free_residual_norm(&internal_force, &external_rhs, &dirichlet_dofs);
    let force_balance_residual_n = norm3(add3(applied_force_n, reaction_force_n));
    let internal_energy = 0.5 * super::dot(&solved.x, &internal_force);
    let rhs_energy = 0.5 * super::dot(&solved.x, &external_rhs);
    let energy_balance_relative_error =
        (internal_energy - rhs_energy).abs() / rhs_energy.abs().max(1.0);
    let force_scale = norm3(applied_force_n).max(1.0);
    if !free_equilibrium_residual_n.is_finite() || free_equilibrium_residual_n > 1e-6 * force_scale
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d.matrix_free.free_equilibrium",
            value: free_equilibrium_residual_n,
        });
    }
    if !force_balance_residual_n.is_finite() || force_balance_residual_n > 1e-6 * force_scale {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d.matrix_free.force_balance",
            value: force_balance_residual_n,
        });
    }
    if !energy_balance_relative_error.is_finite() || energy_balance_relative_error > 1e-9 {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d.matrix_free.energy_identity",
            value: energy_balance_relative_error,
        });
    }

    let element_von_mises_pa = element_von_mises_order(&mesh, study, &space, &solved.x);
    Ok(Solid3DStaticResult {
        nodes: mesh.num_nodes(),
        elements: mesh.num_elements(),
        dofs,
        solver_iterations: solved.iterations,
        solver_residual: solved.residual,
        converged: solved.converged,
        max_displacement_m: solved
            .x
            .as_chunks::<3>()
            .0
            .iter()
            .map(|chunk| norm3([chunk[0], chunk[1], chunk[2]]))
            .fold(0.0, f64::max),
        max_von_mises_pa: element_von_mises_pa.iter().copied().fold(0.0, f64::max),
        strain_energy_j: internal_energy,
        applied_force_n,
        reaction_force_n,
        free_equilibrium_residual_n,
        force_balance_residual_n,
        energy_balance_relative_error,
        model_assumption: StructuralModelAssumption::Solid3D,
        polynomial_order,
        solver: Solid3DStaticSolver::MatrixFreeJacobiFgmres,
        execution: StructuralExecutionProvenance::cpu_matrix_free_p2_reference(),
        displacement,
        element_von_mises_pa,
    })
}

/// Matrix-free elasticity operator with homogeneous Dirichlet rows/columns.
///
/// The underlying FEM operator remains unconstrained. Fixed DOF columns are
/// projected out before the element action and fixed rows are replaced by the
/// identity, matching the assembled boundary-elimination contract.
struct ConstrainedMatrixFreeOperator<'a> {
    base: MatrixFreeOperator<'a, f64, ElasticityForm, VectorH1Space>,
    dirichlet_dofs: std::collections::HashSet<usize>,
}

impl ConstrainedMatrixFreeOperator<'_> {
    fn apply_unconstrained(&self, x: &[f64]) -> Vec<f64> {
        self.base.apply_vec(x)
    }
}

impl LinearOperator<f64> for ConstrainedMatrixFreeOperator<'_> {
    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        let mut projected = x.to_vec();
        for &dof in &self.dirichlet_dofs {
            projected[dof] = 0.0;
        }
        let mut result = self.base.apply_vec(&projected);
        for &dof in &self.dirichlet_dofs {
            result[dof] = x[dof];
        }
        y.copy_from_slice(&result);
    }

    fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
        // Elasticity is symmetric; the same constrained action is its exact
        // transpose for the CPU reference path.
        self.apply_into(x, y);
    }

    fn num_rows(&self) -> usize {
        self.base.num_dofs()
    }

    fn num_cols(&self) -> usize {
        self.base.num_dofs()
    }
}

/// Apply a constant total force to a triangular solid face.
///
/// For P2, the three edge-midpoint DOFs receive the consistent quadratic
/// surface-load contribution. The existing generic FEM traction helper only
/// addresses vertex DOFs, so keeping this policy here prevents a quadratic
/// solve from silently losing one third of its face interpolation space.
pub(crate) fn apply_solid3d_traction(
    rhs: &mut [f64],
    mesh: &Mesh,
    boundary_marker: i32,
    force_n: [f64; 3],
    loaded_face_area_m2: f64,
    polynomial_order: StructuralPolynomialOrder,
) -> Result<(), StructuralMechanicsError> {
    if !loaded_face_area_m2.is_finite() || loaded_face_area_m2 <= 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "solid3d.loaded_face_area_m2",
            value: loaded_face_area_m2.to_string(),
        });
    }
    let traction = force_n.map(|force| force / loaded_face_area_m2);
    let topology = (polynomial_order == StructuralPolynomialOrder::Quadratic)
        .then(|| MeshTopology::build(mesh));
    let node_count = mesh.num_nodes();
    let tri_quadrature = [
        (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),
        (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
    ];

    for boundary in mesh
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == boundary_marker)
    {
        if boundary.nodes.len() != 3 {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: format!(
                    "solid3d loaded boundary element {} has {} nodes; expected a triangle",
                    boundary.element_idx,
                    boundary.nodes.len()
                ),
            });
        }
        let vertices = [boundary.nodes[0], boundary.nodes[1], boundary.nodes[2]];
        if vertices.iter().any(|&node| node >= node_count) {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: "solid3d loaded boundary references a missing node".to_string(),
            });
        }
        let p0 = mesh.nodes[vertices[0]];
        let p1 = mesh.nodes[vertices[1]];
        let p2 = mesh.nodes[vertices[2]];
        let cross = cross3(subtract_point(p1, p0), subtract_point(p2, p0));
        let area = 0.5 * norm3(cross);
        if !area.is_finite() || area <= 0.0 {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: format!(
                    "solid3d loaded boundary element {} has zero or non-finite area",
                    boundary.element_idx
                ),
            });
        }

        let scalar_dofs = match polynomial_order {
            StructuralPolynomialOrder::Linear => vertices.to_vec(),
            StructuralPolynomialOrder::Quadratic => {
                let Some(topology) = topology.as_ref() else {
                    return Err(StructuralMechanicsError::InvalidMesh {
                        reason: "quadratic solid3d load assembly is missing edge topology"
                            .to_string(),
                    });
                };
                let edge_pairs = [
                    (vertices[0], vertices[1]),
                    (vertices[1], vertices[2]),
                    (vertices[2], vertices[0]),
                ];
                let mut dofs = vertices.to_vec();
                for (a, b) in edge_pairs {
                    let key = if a < b { (a, b) } else { (b, a) };
                    let Some(&edge) = topology.edge_map.get(&key) else {
                        return Err(StructuralMechanicsError::InvalidMesh {
                            reason: format!(
                                "solid3d loaded boundary edge ({a}, {b}) is absent from mesh topology"
                            ),
                        });
                    };
                    dofs.push(node_count + edge);
                }
                dofs
            }
        };

        for &(xi, eta, reference_weight) in &tri_quadrature {
            let lambda = [1.0 - xi - eta, xi, eta];
            let values = match polynomial_order {
                StructuralPolynomialOrder::Linear => lambda.to_vec(),
                StructuralPolynomialOrder::Quadratic => vec![
                    lambda[0] * (2.0 * lambda[0] - 1.0),
                    lambda[1] * (2.0 * lambda[1] - 1.0),
                    lambda[2] * (2.0 * lambda[2] - 1.0),
                    4.0 * lambda[0] * lambda[1],
                    4.0 * lambda[1] * lambda[2],
                    4.0 * lambda[2] * lambda[0],
                ],
            };
            let weight = 2.0 * area * reference_weight;
            for (scalar_dof, value) in scalar_dofs.iter().zip(values) {
                for component in 0..3 {
                    rhs[scalar_dof * 3 + component] += traction[component] * value * weight;
                }
            }
        }
    }
    Ok(())
}

fn element_von_mises_order(
    mesh: &Mesh,
    study: &Solid3DStudy,
    space: &VectorH1Space,
    solution: &[f64],
) -> Vec<f64> {
    let c = study.material.as_fem().voigt_3d();
    mesh.elements
        .iter()
        .enumerate()
        .map(|(element_idx, element)| {
            if element.vertices().len() != 4 {
                return f64::NAN;
            }
            let dofs = space.local_to_global(mesh, element_idx);
            let tabulation = space.tabulate(mesh, element_idx, 0.25, 0.25, 0.25);
            let mut grad = [[0.0; 3]; 3];
            for (local_dof, &global_dof) in dofs.iter().enumerate() {
                let component = local_dof % 3;
                for (direction, gradient) in grad[component].iter_mut().enumerate() {
                    *gradient +=
                        solution[global_dof] * tabulation.derivatives[local_dof * 3 + direction];
                }
            }
            von_mises_from_gradient(c, grad)
        })
        .collect()
}

fn von_mises_from_gradient(c: [f64; 36], grad: [[f64; 3]; 3]) -> f64 {
    let strain = [
        grad[0][0],
        grad[1][1],
        grad[2][2],
        grad[1][2] + grad[2][1],
        grad[0][2] + grad[2][0],
        grad[0][1] + grad[1][0],
    ];
    let stress: [f64; 6] = std::array::from_fn(|row| {
        (0..6)
            .map(|column| c[row * 6 + column] * strain[column])
            .sum::<f64>()
    });
    (0.5 * ((stress[0] - stress[1]).powi(2)
        + (stress[1] - stress[2]).powi(2)
        + (stress[2] - stress[0]).powi(2))
        + 3.0 * (stress[3].powi(2) + stress[4].powi(2) + stress[5].powi(2)))
    .sqrt()
}

/// Solve the assembled generalized 3D solid stiffness/mass eigenproblem.
pub fn run_solid3d_modal(
    study: &Solid3DStudy,
) -> Result<Solid3DModalResult, StructuralMechanicsError> {
    let mesh = study.mesh()?;
    let config = StructuralModalConfig {
        degree: PolynomialDegree::P1,
        components: 3,
        density: study.material.density_kg_m3,
        plane: None,
        fixed_boundary_markers: vec![study.loads.fixed_boundary_marker],
    };
    let problem = build_structural_modal_problem(&mesh, study.material.as_fem(), &config);
    let solver_config = LobpcgConfig {
        num_eigenvalues: study.modal_modes,
        max_iterations: 1_000,
        tolerance: 1e-9,
        ..LobpcgConfig::default()
    };
    let solved =
        problem
            .solve(&solver_config)
            .map_err(|error| StructuralMechanicsError::SolverFailure {
                stage: "solid-3d modal",
                message: error.to_string(),
            })?;
    if !solved.converged {
        return Err(StructuralMechanicsError::ModalNonConvergence {
            iterations: solved.iterations,
            residual: solved.residual_norms.iter().copied().fold(0.0, f64::max),
        });
    }
    if solved
        .eigenvalues
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        let eigenvalue = solved
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        return Err(StructuralMechanicsError::UnstableModel { eigenvalue });
    }

    let fixed_dofs = VectorH1Space::new(PolynomialDegree::P1, 3)
        .boundary_dofs(&mesh, study.loads.fixed_boundary_marker);
    let free_dofs = free_dof_map(mesh.num_nodes() * 3, &fixed_dofs);
    let mut modes = Vec::with_capacity(solved.eigenvectors.ncols());
    let mut mass_normalization_error: f64 = 0.0;
    let mut orthogonality_error: f64 = 0.0;
    let vectors = (0..solved.eigenvectors.ncols())
        .map(|column| solved.eigenvectors.column_to_vec(column))
        .collect::<Vec<_>>();
    for vector in &vectors {
        let mass_vector = csr_matvec(&problem.mass_like, vector);
        mass_normalization_error =
            mass_normalization_error.max((super::dot(vector, &mass_vector) - 1.0).abs());
        let mut full = vec![[0.0; 3]; mesh.num_nodes()];
        for (reduced, &full_dof) in free_dofs.iter().enumerate() {
            full[full_dof / 3][full_dof % 3] = vector[reduced];
        }
        modes.push(full);
    }
    for (i, lhs) in vectors.iter().enumerate() {
        for rhs in vectors.iter().skip(i + 1) {
            let mass_rhs = csr_matvec(&problem.mass_like, rhs);
            orthogonality_error = orthogonality_error.max(super::dot(lhs, &mass_rhs).abs());
        }
    }

    Ok(Solid3DModalResult {
        eigenvalues: solved.eigenvalues.clone(),
        frequencies_hz: solved
            .eigenvalues
            .iter()
            .map(|value| value.sqrt() / (2.0 * std::f64::consts::PI))
            .collect(),
        residual_norms: solved.residual_norms,
        solver_iterations: solved.iterations,
        converged: solved.converged,
        normalization: ModalNormalization::MassNormalized,
        mass_normalization_error,
        orthogonality_error,
        damping_ratio: study.material.damping_ratio,
        damping_convention: DampingConvention::MassProportionalAtFirstMode,
        model_assumption: StructuralModelAssumption::Solid3D,
        modes,
    })
}

fn free_dof_map(total_dofs: usize, constrained: &[usize]) -> Vec<usize> {
    let mut is_constrained = vec![false; total_dofs];
    for &dof in constrained {
        if dof < total_dofs {
            is_constrained[dof] = true;
        }
    }
    (0..total_dofs)
        .filter(|dof| !is_constrained[*dof])
        .collect()
}

fn csr_matvec(matrix: &sonium_solvers::CsrMatrix<f64>, vector: &[f64]) -> Vec<f64> {
    (0..matrix.num_rows)
        .map(|row| {
            (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                .map(|index| matrix.values[index] * vector[matrix.col_indices[index]])
                .sum()
        })
        .collect()
}

fn csr_residual_norm(
    matrix: &sonium_solvers::CsrMatrix<f64>,
    solution: &[f64],
    rhs: &[f64],
) -> f64 {
    let residual = csr_matvec(matrix, solution)
        .into_iter()
        .zip(rhs)
        .map(|(lhs, rhs)| lhs - rhs)
        .collect::<Vec<_>>();
    super::dot(&residual, &residual).sqrt() / super::dot(rhs, rhs).sqrt().max(1.0)
}

fn component_force_3(vector: &[f64]) -> [f64; 3] {
    vector
        .as_chunks::<3>()
        .0
        .iter()
        .fold([0.0; 3], |mut total, chunk| {
            total[0] += chunk[0];
            total[1] += chunk[1];
            total[2] += chunk[2];
            total
        })
}

fn reaction_force_3(internal: &[f64], external: &[f64], fixed_dofs: &[usize]) -> [f64; 3] {
    fixed_dofs.iter().fold([0.0; 3], |mut total, &dof| {
        total[dof % 3] += internal[dof] - external[dof];
        total
    })
}

fn free_residual_norm(
    internal: &[f64],
    external: &[f64],
    dirichlet: &std::collections::HashSet<usize>,
) -> f64 {
    internal
        .iter()
        .zip(external)
        .enumerate()
        .filter(|(dof, _)| !dirichlet.contains(dof))
        .map(|(_, (lhs, rhs))| (lhs - rhs).powi(2))
        .sum::<f64>()
        .sqrt()
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

fn element_von_mises(mesh: &Mesh, study: &Solid3DStudy, displacement: &[[f64; 3]]) -> Vec<f64> {
    let c = study.material.as_fem().voigt_3d();
    mesh.elements
        .iter()
        .map(|element| {
            let nodes = element.vertices();
            if nodes.len() != 4 {
                return f64::NAN;
            }
            let gradients = tetrahedron_shape_gradients(mesh, nodes);
            let mut grad = [[0.0; 3]; 3];
            for (local, &node) in nodes.iter().enumerate() {
                for component in 0..3 {
                    for direction in 0..3 {
                        grad[component][direction] +=
                            displacement[node][component] * gradients[local][direction];
                    }
                }
            }
            let strain = [
                grad[0][0],
                grad[1][1],
                grad[2][2],
                grad[1][2] + grad[2][1],
                grad[0][2] + grad[2][0],
                grad[0][1] + grad[1][0],
            ];
            let stress: [f64; 6] = std::array::from_fn(|row| {
                (0..6)
                    .map(|column| c[row * 6 + column] * strain[column])
                    .sum::<f64>()
            });
            (0.5 * ((stress[0] - stress[1]).powi(2)
                + (stress[1] - stress[2]).powi(2)
                + (stress[2] - stress[0]).powi(2))
                + 3.0 * (stress[3].powi(2) + stress[4].powi(2) + stress[5].powi(2)))
            .sqrt()
        })
        .collect()
}

fn tetrahedron_shape_gradients(mesh: &Mesh, nodes: &[usize]) -> [[f64; 3]; 4] {
    let p0 = mesh.nodes[nodes[0]];
    let p1 = mesh.nodes[nodes[1]];
    let p2 = mesh.nodes[nodes[2]];
    let p3 = mesh.nodes[nodes[3]];
    let b = subtract_point(p1, p0);
    let c = subtract_point(p2, p0);
    let d = subtract_point(p3, p0);
    let determinant = dot3(b, cross3(c, d));
    let g1 = scale3(cross3(c, d), 1.0 / determinant);
    let g2 = scale3(cross3(d, b), 1.0 / determinant);
    let g3 = scale3(cross3(b, c), 1.0 / determinant);
    [scale3(add3(add3(g1, g2), g3), -1.0), g1, g2, g3]
}

fn subtract_point(lhs: Point, rhs: Point) -> [f64; 3] {
    [lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z]
}

fn cross3(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

fn dot3(lhs: [f64; 3], rhs: [f64; 3]) -> f64 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

fn scale3(value: [f64; 3], scale: f64) -> [f64; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_study() -> Solid3DStudy {
        let mut study = Solid3DStudy::introductory();
        study.geometry.nx = 4;
        study.geometry.ny = 1;
        study.geometry.nz = 1;
        study.modal_modes = 1;
        study
    }

    #[test]
    fn solid3d_static_recovers_force_energy_and_uniaxial_stress() {
        let study = small_study();
        let result = run_solid3d_static(&study).unwrap();

        assert!(result.converged);
        assert_eq!(result.model_assumption, StructuralModelAssumption::Solid3D);
        assert_eq!(result.nodes, 20);
        assert_eq!(result.elements, 24);
        assert!((result.applied_force_n[0] - 100.0).abs() < 1e-8);
        assert!(result.max_displacement_m > 0.0);
        assert!(result.max_von_mises_pa > 0.0);
        assert!(result.strain_energy_j > 0.0);
        assert!(result.force_balance_residual_n < 1e-6);
        assert!(result.free_equilibrium_residual_n < 1e-6);
        assert!(result.energy_balance_relative_error < 1e-10);
        let expected_stress = 100.0 / study.geometry.loaded_face_area_m2();
        let mean_stress = result.element_von_mises_pa.iter().sum::<f64>()
            / result.element_von_mises_pa.len() as f64;
        assert!(
            (mean_stress - expected_stress).abs() / expected_stress < 0.1,
            "max={}, mean={mean_stress}, expected={expected_stress}",
            result.max_von_mises_pa,
        );
    }

    #[test]
    fn solid3d_quadratic_matrix_free_preserves_load_and_reports_provenance() {
        let result =
            run_solid3d_static_order(&small_study(), StructuralPolynomialOrder::Quadratic).unwrap();

        assert!(result.converged);
        assert_eq!(
            result.polynomial_order,
            StructuralPolynomialOrder::Quadratic
        );
        assert_eq!(result.solver, Solid3DStaticSolver::MatrixFreeJacobiFgmres);
        assert_eq!(
            result.execution.capability_id,
            "sonium-structural.cpu-reference.f64.matrix-free-p2.v1"
        );
        assert!((result.applied_force_n[0] - 100.0).abs() < 1e-8);
        assert!(result.max_displacement_m > 0.0);
        assert!(result.max_von_mises_pa > 0.0);
        assert!(result.free_equilibrium_residual_n < 1e-6);
        assert!(result.energy_balance_relative_error < 1e-9);
        assert_eq!(result.displacement.len(), result.nodes);
        assert_eq!(result.element_von_mises_pa.len(), result.elements);
    }

    #[test]
    fn solid3d_amg_static_path_is_explicit_and_convergent() {
        let result = run_solid3d_static_with_options(
            &small_study(),
            Solid3DStaticOptions {
                polynomial_order: StructuralPolynomialOrder::Linear,
                solver: Solid3DStaticSolver::AssembledAmgFgmres,
            },
        )
        .unwrap();

        assert!(result.converged);
        assert_eq!(result.solver, Solid3DStaticSolver::AssembledAmgFgmres);
        assert_eq!(
            result.execution,
            StructuralExecutionProvenance::cpu_reference()
        );
        assert!(result.force_balance_residual_n < 1e-6);
    }

    #[test]
    fn assembled_solid3d_path_rejects_matrix_free_selector_with_typed_error() {
        let error = run_solid3d_static_assembled(
            &small_study(),
            Solid3DStaticOptions {
                polynomial_order: StructuralPolynomialOrder::Linear,
                solver: Solid3DStaticSolver::MatrixFreeJacobiFgmres,
            },
        )
        .expect_err("assembled path must reject matrix-free selector");

        assert!(matches!(
            error,
            StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_static.solver",
                ..
            }
        ));
    }

    #[test]
    fn solid3d_modal_result_has_positive_frequencies_and_orthogonal_modes() {
        let result = run_solid3d_modal(&small_study()).unwrap();

        assert!(result.converged);
        assert_eq!(result.eigenvalues.len(), 1);
        assert!(result.eigenvalues[0] > 0.0);
        assert!(result.frequencies_hz[0] > 0.0);
        assert!(result.mass_normalization_error < 1e-8);
        assert!(result.orthogonality_error < 1e-8);
        assert_eq!(result.modes.len(), 1);
        assert_eq!(result.modes[0].len(), 20);
    }

    #[test]
    fn solid3d_multi_load_reuses_one_factorization_and_preserves_balance() {
        let study = Solid3DMultiLoadStudy {
            base: small_study(),
            load_cases: vec![
                Solid3DMultiLoadCase {
                    name: "axial".to_string(),
                    force_n: [100.0, 0.0, 0.0],
                },
                Solid3DMultiLoadCase {
                    name: "combined".to_string(),
                    force_n: [60.0, 20.0, -10.0],
                },
            ],
        };
        let result = run_solid3d_multi_load(&study).expect("3D multi-load solve");

        assert_eq!(result.cases.len(), 2);
        assert!(result.factorization_reused);
        assert!(result.factorization_memory_bytes > 0);
        assert!(result.converged);
        assert!(result.cases.iter().all(|case| {
            case.solver_residual < 1.0e-9
                && case.free_equilibrium_residual_n < 1.0e-6
                && case.force_balance_residual_n < 1.0e-6
                && case.energy_balance_relative_error < 1.0e-9
                && case.max_displacement_m > 0.0
                && case.max_von_mises_pa > 0.0
        }));
        assert!(!result.critical_case.is_empty());
        assert_eq!(
            result.execution.capability_id,
            "sonium-structural.cpu-reference.f64.solid3d-multi-load.v1"
        );
    }

    #[test]
    fn solid3d_rejects_invalid_mesh_density_and_markers() {
        let mut study = small_study();
        study.geometry.nz = 0;
        assert_eq!(
            run_solid3d_static(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d.mesh_density",
                value: "nx=4, ny=1, nz=0".to_string(),
            })
        );

        let mut study = small_study();
        study.loads.loaded_boundary_marker = study.loads.fixed_boundary_marker;
        assert!(matches!(
            run_solid3d_modal(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d.boundary_markers",
                ..
            })
        ));
    }
}
