//! Bounded complex-stiffness viscoelastic plane-stress FEM studies.
//!
//! The implementation is deliberately narrow: it solves a rectangular P1
//! plane-stress problem with a homogeneous fixed x-minimum edge and a point
//! load at the opposite corner.  The complex constitutive law is the
//! frequency-independent structural-loss model already implemented by
//! `sonium-fem`: `C* = (1 + i*eta) C`.  This is a reference workflow for
//! complex stiffness, not a time-domain hereditary material model.

use crate::{StructuralExecutionProvenance, StructuralMechanicsError, StructuralModelAssumption};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{
    FemSystem, try_apply_homogeneous_dirichlet_complex, try_apply_homogeneous_dirichlet_dofs,
};
use sonium_fem::mesh::{Mesh, rectangular_mesh_triangles};
use sonium_fem::physics::solid::elasticity::{
    ElasticityForm, IsotropicMaterial, PlaneAssumption, ViscoelasticForm, ViscoelasticMaterial,
};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::{LinearSolveConfig, LinearSolverType, solve_complex, solve_real};

/// Fixed-edge boundary marker emitted by `rectangular_mesh_triangles`.
pub const VISCOELASTIC_FIXED_BOUNDARY_MARKER: i32 = 1;

const VALIDATION_TOLERANCE: f64 = 1.0e-7;
const MAX_CELLS: usize = 16_384;
const DOFS_PER_NODE: usize = 2;

/// Rectangular mesh geometry for the bounded 2D viscoelastic study.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Viscoelastic2DGeometry {
    pub length_m: f64,
    pub width_m: f64,
    pub nx: usize,
    pub ny: usize,
}

impl Default for Viscoelastic2DGeometry {
    fn default() -> Self {
        Self {
            length_m: 1.0,
            width_m: 0.2,
            nx: 12,
            ny: 3,
        }
    }
}

impl Viscoelastic2DGeometry {
    /// Build the structured P1 triangular mesh used by this study family.
    pub fn mesh(self) -> Mesh {
        rectangular_mesh_triangles(0.0, self.length_m, 0.0, self.width_m, self.nx, self.ny)
    }
}

/// Isotropic complex-stiffness material inputs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Viscoelastic2DMaterial {
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
    /// Structural loss factor in `C* = (1 + i*eta) C`.
    pub loss_factor: f64,
}

impl Default for Viscoelastic2DMaterial {
    fn default() -> Self {
        Self {
            youngs_modulus_pa: 2.0e6,
            poisson_ratio: 0.3,
            loss_factor: 0.25,
        }
    }
}

impl Viscoelastic2DMaterial {
    fn elastic(self) -> IsotropicMaterial {
        IsotropicMaterial::new(self.youngs_modulus_pa, self.poisson_ratio)
    }

    fn viscoelastic(self) -> ViscoelasticMaterial {
        ViscoelasticMaterial::new(self.elastic(), self.loss_factor)
    }

    fn complex_factor(self) -> Complex64 {
        self.viscoelastic().complex_factor()
    }
}

/// Bounded rectangular plane-stress study with a corner point load.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Viscoelastic2DStudy {
    pub name: String,
    pub geometry: Viscoelastic2DGeometry,
    pub material: Viscoelastic2DMaterial,
    /// Point load `[Fx, Fy]` in newtons at the right/top mesh corner.
    pub point_load_n: [f64; 2],
}

impl Default for Viscoelastic2DStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl Viscoelastic2DStudy {
    /// Construct the small CPU reference viscoelastic study.
    pub fn introductory() -> Self {
        Self {
            name: "complex-stiffness-viscoelastic-plane-stress".to_string(),
            geometry: Viscoelastic2DGeometry::default(),
            material: Viscoelastic2DMaterial::default(),
            point_load_n: [0.0, -100.0],
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("viscoelastic.length_m", self.geometry.length_m),
            ("viscoelastic.width_m", self.geometry.width_m),
            (
                "viscoelastic.youngs_modulus_pa",
                self.material.youngs_modulus_pa,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if !self.material.loss_factor.is_finite() || self.material.loss_factor < 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "viscoelastic.loss_factor",
                value: self.material.loss_factor.to_string(),
            });
        }
        if !(-1.0 < self.material.poisson_ratio && self.material.poisson_ratio < 0.5) {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "viscoelastic.poisson_ratio",
                value: self.material.poisson_ratio.to_string(),
            });
        }
        if self.geometry.nx == 0 || self.geometry.ny == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "viscoelastic.mesh_density",
                value: format!(
                    "nx={}, ny={} (both must be non-zero)",
                    self.geometry.nx, self.geometry.ny
                ),
            });
        }
        let cells = self
            .geometry
            .nx
            .checked_mul(self.geometry.ny)
            .ok_or_else(|| StructuralMechanicsError::InvalidConfiguration {
                field: "viscoelastic.mesh_density",
                value: "nx * ny overflows usize".to_string(),
            })?;
        if cells > MAX_CELLS {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "viscoelastic.mesh_density",
                value: format!("{cells} cells exceeds the limit {MAX_CELLS}"),
            });
        }
        if self.point_load_n.iter().any(|value| !value.is_finite())
            || self.point_load_n.iter().all(|value| *value == 0.0)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "viscoelastic.point_load_n",
                value: format!("{:?} (must be finite and non-zero)", self.point_load_n),
            });
        }
        Ok(())
    }

    fn load_node(&self, mesh: &Mesh) -> Result<usize, StructuralMechanicsError> {
        mesh.nodes
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| {
                lhs.x
                    .total_cmp(&rhs.x)
                    .then_with(|| lhs.y.total_cmp(&rhs.y))
            })
            .map(|(node, _)| node)
            .ok_or_else(|| StructuralMechanicsError::InvalidMesh {
                reason: "viscoelastic mesh contains no nodes".to_string(),
            })
    }
}

/// Result of a bounded complex-stiffness viscoelastic solve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Viscoelastic2DResult {
    pub nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    pub load_node: usize,
    pub solver_iterations: usize,
    pub solver_residual: f64,
    pub converged: bool,
    pub max_displacement_m: f64,
    /// Loaded-node displacement as `[[ux.re, ux.im], [uy.re, uy.im]]`.
    pub point_displacement_m: [[f64; 2]; 2],
    pub applied_force_n: [f64; 2],
    /// Support reactions as `[[Rx.re, Rx.im], [Ry.re, Ry.im]]`.
    pub reaction_force_n: [[f64; 2]; 2],
    pub free_equilibrium_residual_n: f64,
    pub force_balance_residual_n: f64,
    /// Complex external work/compliance as `[real, imaginary]`.
    pub complex_compliance_j: [f64; 2],
    pub elastic_reference_compliance_j: f64,
    pub expected_compliance_j: [f64; 2],
    pub compliance_scaling_relative_error: f64,
    pub displacement_scaling_relative_error: f64,
    pub phase_lag_rad: f64,
    pub expected_phase_lag_rad: f64,
    pub phase_error_rad: f64,
    pub complex_stiffness_factor: [f64; 2],
    /// Bilinear-form strain energy as `[real, imaginary]`.
    pub complex_strain_energy_j: [f64; 2],
    pub complex_external_work_j: [f64; 2],
    pub energy_balance_relative_error: f64,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
    pub coordinates: Vec<[f64; 3]>,
    /// Full displacement field per node as `[[ux.re, ux.im], [uy.re, uy.im]]`.
    pub displacement: Vec<[[f64; 2]; 2]>,
}

/// Run the bounded complex-stiffness plane-stress reference solve.
pub fn run_viscoelastic_2d(
    study: &Viscoelastic2DStudy,
) -> Result<Viscoelastic2DResult, StructuralMechanicsError> {
    study.validate()?;
    let mesh = study.geometry.mesh();
    let space = VectorH1Space::new(PolynomialDegree::P1, DOFS_PER_NODE);
    let dofs = space.num_global_dofs(&mesh);
    let load_node = study.load_node(&mesh)?;
    let fixed_dofs = space.boundary_dofs(&mesh, VISCOELASTIC_FIXED_BOUNDARY_MARKER);
    if fixed_dofs.is_empty() {
        return Err(StructuralMechanicsError::MissingFixedBoundary {
            marker: VISCOELASTIC_FIXED_BOUNDARY_MARKER,
        });
    }
    if fixed_dofs
        .iter()
        .any(|dof| *dof / DOFS_PER_NODE == load_node)
    {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "viscoelastic point load node lies on the fixed boundary".to_string(),
        });
    }

    let form =
        ViscoelasticForm::new_2d(study.material.viscoelastic(), PlaneAssumption::PlaneStress);
    let stiffness = assemble_bilinear::<Complex64, _, _>(&form, &space, &space, &mesh, 2);
    let unconstrained_stiffness = stiffness.clone();
    let mut rhs = vec![Complex64::new(0.0, 0.0); dofs];
    rhs[load_node * DOFS_PER_NODE] = Complex64::new(study.point_load_n[0], 0.0);
    rhs[load_node * DOFS_PER_NODE + 1] = Complex64::new(study.point_load_n[1], 0.0);
    let external_rhs = rhs.clone();

    let mut system = FemSystem::new(stiffness, rhs);
    try_apply_homogeneous_dirichlet_complex(
        &mut system,
        &mesh,
        &space,
        VISCOELASTIC_FIXED_BOUNDARY_MARKER,
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "viscoelastic complex boundary",
        message: error.to_string(),
    })?;

    let solver_config = LinearSolveConfig {
        solver: LinearSolverType::GmresIlu,
        max_iterations: 8_000,
        tolerance: 1.0e-10,
        restart: 50,
        ..Default::default()
    };
    let solved =
        solve_complex(&system.matrix.to_csr(), &system.rhs, &solver_config).map_err(|error| {
            StructuralMechanicsError::SolverFailure {
                stage: "viscoelastic complex solve",
                message: error.to_string(),
            }
        })?;
    if !solved.converged {
        return Err(StructuralMechanicsError::StaticNonConvergence {
            model: StructuralModelAssumption::ViscoelasticPlaneStress,
            iterations: solved.iterations,
            residual: solved.residual,
        });
    }
    let solution = solved.x;

    // Solve the same mesh and load with the real elastic form.  This is an
    // independent zero-loss reference for the exact constitutive scaling
    // C* = (1 + i*eta) C, not an analytical displacement surrogate.
    let elastic_form =
        ElasticityForm::new_2d(study.material.elastic(), PlaneAssumption::PlaneStress);
    let elastic_stiffness = assemble_bilinear::<f64, _, _>(&elastic_form, &space, &space, &mesh, 2);
    let mut elastic_rhs = vec![0.0; dofs];
    elastic_rhs[load_node * DOFS_PER_NODE] = study.point_load_n[0];
    elastic_rhs[load_node * DOFS_PER_NODE + 1] = study.point_load_n[1];
    let mut elastic_system = FemSystem::new(elastic_stiffness, elastic_rhs.clone());
    try_apply_homogeneous_dirichlet_dofs(&mut elastic_system, &fixed_dofs).map_err(|error| {
        StructuralMechanicsError::SolverFailure {
            stage: "viscoelastic real reference boundary",
            message: error.to_string(),
        }
    })?;
    let elastic_solved = solve_real(
        &elastic_system.matrix.to_csr(),
        &elastic_system.rhs,
        &solver_config,
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "viscoelastic real reference solve",
        message: error.to_string(),
    })?;
    if !elastic_solved.converged {
        return Err(StructuralMechanicsError::StaticNonConvergence {
            model: StructuralModelAssumption::PlaneStress,
            iterations: elastic_solved.iterations,
            residual: elastic_solved.residual,
        });
    }

    let internal = complex_triplet_matvec(&unconstrained_stiffness, &solution);
    let residual: Vec<Complex64> = internal
        .iter()
        .zip(&external_rhs)
        .map(|(lhs, rhs)| *lhs - *rhs)
        .collect();
    let fixed_set: std::collections::HashSet<usize> = fixed_dofs.iter().copied().collect();
    let free_equilibrium_residual_n = complex_norm(
        &residual
            .iter()
            .enumerate()
            .filter_map(|(dof, value)| (!fixed_set.contains(&dof)).then_some(*value))
            .collect::<Vec<_>>(),
    );

    let mut reaction = [Complex64::new(0.0, 0.0); DOFS_PER_NODE];
    for &dof in &fixed_dofs {
        reaction[dof % DOFS_PER_NODE] += residual[dof];
    }
    let applied = [
        Complex64::new(study.point_load_n[0], 0.0),
        Complex64::new(study.point_load_n[1], 0.0),
    ];
    let force_balance_residual_n = complex_norm(
        &applied
            .iter()
            .zip(reaction)
            .map(|(applied, reaction)| *applied + reaction)
            .collect::<Vec<_>>(),
    );
    let load_scale = study
        .point_load_n
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
        .max(1.0);
    if !free_equilibrium_residual_n.is_finite()
        || free_equilibrium_residual_n > VALIDATION_TOLERANCE * load_scale
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "viscoelastic.free_equilibrium",
            value: free_equilibrium_residual_n,
        });
    }
    if !force_balance_residual_n.is_finite()
        || force_balance_residual_n > VALIDATION_TOLERANCE * load_scale
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "viscoelastic.force_balance",
            value: force_balance_residual_n,
        });
    }

    let complex_external_work = complex_dot_noconj(&external_rhs, &solution);
    let complex_strain_energy = Complex64::new(0.5, 0.0) * complex_dot_noconj(&solution, &internal);
    let expected_energy = Complex64::new(0.5, 0.0) * complex_external_work;
    let energy_balance_relative_error =
        relative_complex_error(complex_strain_energy, expected_energy);
    if !energy_balance_relative_error.is_finite() || energy_balance_relative_error > 1.0e-9 {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "viscoelastic.energy_identity",
            value: energy_balance_relative_error,
        });
    }

    let factor = study.material.complex_factor();
    let elastic_reference_work = elastic_solved
        .x
        .iter()
        .zip(&elastic_rhs)
        .map(|(displacement, force)| displacement * force)
        .sum::<f64>();
    let expected_compliance = Complex64::new(elastic_reference_work, 0.0) / factor;
    let compliance_scaling_relative_error =
        relative_complex_error(complex_external_work, expected_compliance);
    let expected_solution: Vec<Complex64> = elastic_solved
        .x
        .iter()
        .map(|value| Complex64::new(*value, 0.0) / factor)
        .collect();
    let displacement_scaling_relative_error =
        relative_complex_vector_error(&solution, &expected_solution);
    let phase_lag_rad = complex_external_work.arg();
    let expected_phase_lag_rad = -factor.arg();
    let phase_error_rad = (phase_lag_rad - expected_phase_lag_rad).abs();
    if !compliance_scaling_relative_error.is_finite() || compliance_scaling_relative_error > 1.0e-8
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "viscoelastic.compliance_scaling",
            value: compliance_scaling_relative_error,
        });
    }
    if !displacement_scaling_relative_error.is_finite()
        || displacement_scaling_relative_error > 1.0e-8
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "viscoelastic.displacement_scaling",
            value: displacement_scaling_relative_error,
        });
    }
    if !phase_error_rad.is_finite() || phase_error_rad > 1.0e-8 {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "viscoelastic.phase_scaling",
            value: phase_error_rad,
        });
    }

    let displacement = solution
        .as_chunks::<DOFS_PER_NODE>()
        .0
        .iter()
        .map(|chunk| [to_pair(chunk[0]), to_pair(chunk[1])])
        .collect::<Vec<_>>();
    let max_displacement_m = solution
        .as_chunks::<DOFS_PER_NODE>()
        .0
        .iter()
        .map(|chunk| (chunk[0].norm_sqr() + chunk[1].norm_sqr()).sqrt())
        .fold(0.0, f64::max);
    let coordinates = mesh
        .nodes
        .iter()
        .map(|point| [point.x, point.y, point.z])
        .collect::<Vec<_>>();

    Ok(Viscoelastic2DResult {
        nodes: mesh.nodes.len(),
        elements: mesh.elements.len(),
        dofs,
        load_node,
        solver_iterations: solved.iterations,
        solver_residual: solved.residual,
        converged: solved.converged,
        max_displacement_m,
        point_displacement_m: displacement[load_node],
        applied_force_n: study.point_load_n,
        reaction_force_n: [to_pair(reaction[0]), to_pair(reaction[1])],
        free_equilibrium_residual_n,
        force_balance_residual_n,
        complex_compliance_j: to_pair(complex_external_work),
        elastic_reference_compliance_j: elastic_reference_work,
        expected_compliance_j: to_pair(expected_compliance),
        compliance_scaling_relative_error,
        displacement_scaling_relative_error,
        phase_lag_rad,
        expected_phase_lag_rad,
        phase_error_rad,
        complex_stiffness_factor: to_pair(factor),
        complex_strain_energy_j: to_pair(complex_strain_energy),
        complex_external_work_j: to_pair(complex_external_work),
        energy_balance_relative_error,
        model_assumption: StructuralModelAssumption::ViscoelasticPlaneStress,
        execution: StructuralExecutionProvenance::cpu_viscoelastic_reference(),
        coordinates,
        displacement,
    })
}

fn complex_triplet_matvec(
    matrix: &sonium_solvers::TripletMatrix<Complex64>,
    x: &[Complex64],
) -> Vec<Complex64> {
    let mut result = vec![Complex64::new(0.0, 0.0); matrix.num_rows];
    for ((&row, &col), &value) in matrix.rows.iter().zip(&matrix.cols).zip(&matrix.values) {
        result[row] += value * x[col];
    }
    result
}

fn complex_dot_noconj(lhs: &[Complex64], rhs: &[Complex64]) -> Complex64 {
    lhs.iter().zip(rhs).map(|(lhs, rhs)| *lhs * *rhs).sum()
}

fn complex_norm(values: &[Complex64]) -> f64 {
    values.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt()
}

fn relative_complex_error(actual: Complex64, expected: Complex64) -> f64 {
    (actual - expected).norm() / expected.norm().max(1.0e-30)
}

fn relative_complex_vector_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    let error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let scale = complex_norm(expected).max(1.0e-30);
    error / scale
}

fn to_pair(value: Complex64) -> [f64; 2] {
    [value.re, value.im]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_study(loss_factor: f64) -> Viscoelastic2DStudy {
        let mut study = Viscoelastic2DStudy::introductory();
        study.geometry.nx = 4;
        study.geometry.ny = 2;
        study.material.loss_factor = loss_factor;
        study
    }

    #[test]
    fn zero_loss_matches_real_reference_and_has_zero_phase() {
        let result = run_viscoelastic_2d(&small_study(0.0)).unwrap();

        assert!(result.converged);
        assert!(result.max_displacement_m > 0.0);
        assert!(result.free_equilibrium_residual_n < 1.0e-7);
        assert!(result.force_balance_residual_n < 1.0e-7);
        assert!(result.energy_balance_relative_error < 1.0e-10);
        assert!(result.compliance_scaling_relative_error < 1.0e-10);
        assert!(result.displacement_scaling_relative_error < 1.0e-10);
        assert!(result.phase_lag_rad.abs() < 1.0e-10);
        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::ViscoelasticPlaneStress
        );
        assert_eq!(
            result.execution.capability_id,
            "sonium-structural.cpu-reference.complex64.viscoelastic-2d.v1"
        );
    }

    #[test]
    fn nonzero_loss_factor_matches_exact_complex_scaling() {
        let loss_factor = 0.35;
        let result = run_viscoelastic_2d(&small_study(loss_factor)).unwrap();
        let expected_phase = -loss_factor.atan();

        assert!(result.converged);
        assert!(result.compliance_scaling_relative_error < 1.0e-8);
        assert!(result.displacement_scaling_relative_error < 1.0e-8);
        assert!((result.phase_lag_rad - expected_phase).abs() < 1.0e-8);
        assert!((result.expected_phase_lag_rad - expected_phase).abs() < 1.0e-12);
        assert!(result.complex_compliance_j[1] < 0.0);
        assert!(result.complex_strain_energy_j[0] > 0.0);
    }

    #[test]
    fn invalid_viscoelastic_inputs_are_rejected_before_material_construction() {
        let mut study = small_study(0.1);
        study.material.loss_factor = -0.1;
        assert!(matches!(
            run_viscoelastic_2d(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "viscoelastic.loss_factor",
                ..
            })
        ));

        study = small_study(0.1);
        study.point_load_n = [0.0, 0.0];
        assert!(matches!(
            run_viscoelastic_2d(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "viscoelastic.point_load_n",
                ..
            })
        ));
    }
}
