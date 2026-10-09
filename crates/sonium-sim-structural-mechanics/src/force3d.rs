//! Bounded force-controlled 3-D hyperelastic reference.
//!
//! The study applies a distributed vertical load to the guided upper face of
//! a P1 tetrahedral block while fixing the lower face. It exercises the
//! production FEM Newton path with an explicit external load vector. This is
//! a small CPU/reference contract, not a post-buckling or calibrated material
//! workflow.

use crate::{StructuralExecutionProvenance, StructuralMechanicsError, StructuralModelAssumption};
use serde::{Deserialize, Serialize};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{Mesh, box_mesh_tetrahedra};
use sonium_fem::physics::solid::nonlinear_solid::{HyperelasticForm, NeoHookean};
use sonium_fem::solver::{FemNewtonConfig, fem_newton_solve_with_bcs_and_load};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};

/// Schema version for [`Hyperelastic3DForceResult`].
pub const HYPERELASTIC_3D_FORCE_SCHEMA_VERSION: u32 = 1;

/// Small guided force-controlled hyperelastic block study.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Hyperelastic3DForceStudy {
    pub length_m: f64,
    pub width_m: f64,
    pub height_m: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub youngs_modulus_pa: f64,
    pub poissons_ratio: f64,
    /// Positive magnitude of the downward force applied to the upper face.
    pub total_force_n: f64,
    pub load_steps: usize,
    pub max_newton_iterations: usize,
    pub newton_residual_tolerance: f64,
}

impl Hyperelastic3DForceStudy {
    /// Compact finite-strain force-controlled reference fixture.
    pub fn introductory() -> Self {
        Self {
            length_m: 0.40,
            width_m: 0.20,
            height_m: 0.40,
            nx: 2,
            ny: 1,
            nz: 2,
            youngs_modulus_pa: 2.0e6,
            poissons_ratio: 0.30,
            total_force_n: 1_600.0,
            load_steps: 8,
            max_newton_iterations: 40,
            newton_residual_tolerance: 1.0e-9,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("hyperelastic_3d_force.length_m", self.length_m),
            ("hyperelastic_3d_force.width_m", self.width_m),
            ("hyperelastic_3d_force.height_m", self.height_m),
            (
                "hyperelastic_3d_force.youngs_modulus_pa",
                self.youngs_modulus_pa,
            ),
            ("hyperelastic_3d_force.total_force_n", self.total_force_n),
            (
                "hyperelastic_3d_force.newton_residual_tolerance",
                self.newton_residual_tolerance,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if !self.poissons_ratio.is_finite() || !(0.0..0.5).contains(&self.poissons_ratio) {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "hyperelastic_3d_force.poissons_ratio",
                value: self.poissons_ratio.to_string(),
            });
        }
        if self.nx == 0 || self.ny == 0 || self.nz == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "hyperelastic_3d_force.mesh_resolution",
                value: format!("({}, {}, {})", self.nx, self.ny, self.nz),
            });
        }
        if self.load_steps == 0 || self.max_newton_iterations == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "hyperelastic_3d_force.iterations",
                value: format!(
                    "load_steps={}, newton={}",
                    self.load_steps, self.max_newton_iterations
                ),
            });
        }
        Ok(())
    }
}

/// Diagnostics for one converged force increment.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Hyperelastic3DForceStepResult {
    pub load_step: usize,
    pub load_factor: f64,
    pub applied_force_n: f64,
    pub newton_iterations: usize,
    pub residual: f64,
    pub average_top_displacement_m: f64,
    pub converged: bool,
}

/// Versioned result for the force-controlled 3-D reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hyperelastic3DForceResult {
    pub schema_version: u32,
    pub study: Hyperelastic3DForceStudy,
    pub nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    pub bottom_nodes: usize,
    pub top_nodes: usize,
    pub steps: Vec<Hyperelastic3DForceStepResult>,
    pub final_force_n: f64,
    pub final_average_top_displacement_m: f64,
    pub final_residual: f64,
    pub converged: bool,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Run the bounded force-controlled 3-D hyperelastic reference.
pub fn run_hyperelastic_3d_force(
    study: &Hyperelastic3DForceStudy,
) -> Result<Hyperelastic3DForceResult, StructuralMechanicsError> {
    study.validate()?;
    let mesh = box_mesh_tetrahedra(
        0.0,
        study.length_m,
        0.0,
        study.width_m,
        0.0,
        study.height_m,
        study.nx,
        study.ny,
        study.nz,
    );
    mesh.validate_connectivity()
        .map_err(|errors| StructuralMechanicsError::InvalidMesh {
            reason: format!("3-D force mesh connectivity: {errors:?}"),
        })?;

    let tolerance = 1.0e-12
        * study
            .length_m
            .max(study.width_m)
            .max(study.height_m)
            .max(1.0);
    let bottom_nodes = nodes_at_z(&mesh, 0.0, tolerance);
    let top_nodes = nodes_at_z(&mesh, study.height_m, tolerance);
    if bottom_nodes.is_empty() || top_nodes.is_empty() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: format!(
                "3-D force reference requires bottom and top nodes (bottom={}, top={})",
                bottom_nodes.len(),
                top_nodes.len()
            ),
        });
    }

    let space = VectorH1Space::new(PolynomialDegree::P1, 3);
    let dofs = space.num_global_dofs(&mesh);
    let material = NeoHookean::from_young(study.youngs_modulus_pa, study.poissons_ratio, 3);
    let form = HyperelasticForm::new(material);
    let mut config = FemNewtonConfig::default();
    config.newton.max_iterations = study.max_newton_iterations;
    config.newton.absolute_tolerance = study.newton_residual_tolerance;
    config.newton.relative_tolerance = study.newton_residual_tolerance;
    config.inner_solver.max_iterations = study.max_newton_iterations.saturating_mul(20);

    let dirichlet_dofs = guided_boundary_conditions(&bottom_nodes, &top_nodes);
    let mut solution = vec![0.0; dofs];
    let mut steps = Vec::with_capacity(study.load_steps);
    let mut final_average_top_displacement_m = 0.0;
    let mut final_residual = f64::INFINITY;

    for load_step in 1..=study.load_steps {
        let load_factor = load_step as f64 / study.load_steps as f64;
        let applied_force_n = study.total_force_n * load_factor;
        let mut external_load = vec![0.0; dofs];
        let nodal_force = -applied_force_n / top_nodes.len() as f64;
        for &node in &top_nodes {
            external_load[3 * node + 2] = nodal_force;
        }

        let newton = fem_newton_solve_with_bcs_and_load(
            &form,
            &space,
            &mesh,
            &dirichlet_dofs,
            &solution,
            &external_load,
            &config,
        )
        .map_err(|error| StructuralMechanicsError::SolverFailure {
            stage: "3-D hyperelastic force-controlled Newton solve",
            message: error.to_string(),
        })?;
        if !newton.converged || !newton.residual.is_finite() {
            return Err(StructuralMechanicsError::NonlinearNonConvergence {
                model: StructuralModelAssumption::HyperelasticSolid3D,
                load_step,
                iterations: newton.iterations,
                residual: newton.residual,
            });
        }

        solution = newton.x;
        let average_top_displacement_m = top_nodes
            .iter()
            .map(|&node| solution[3 * node + 2])
            .sum::<f64>()
            / top_nodes.len() as f64;
        final_average_top_displacement_m = average_top_displacement_m;
        final_residual = newton.residual;
        steps.push(Hyperelastic3DForceStepResult {
            load_step,
            load_factor,
            applied_force_n,
            newton_iterations: newton.iterations,
            residual: newton.residual,
            average_top_displacement_m,
            converged: true,
        });
    }

    Ok(Hyperelastic3DForceResult {
        schema_version: HYPERELASTIC_3D_FORCE_SCHEMA_VERSION,
        study: *study,
        nodes: mesh.nodes.len(),
        elements: mesh.elements.len(),
        dofs,
        bottom_nodes: bottom_nodes.len(),
        top_nodes: top_nodes.len(),
        steps,
        final_force_n: study.total_force_n,
        final_average_top_displacement_m,
        final_residual,
        converged: true,
        model_assumption: StructuralModelAssumption::HyperelasticSolid3D,
        execution: StructuralExecutionProvenance::cpu_hyperelastic_3d_force_reference(),
    })
}

fn nodes_at_z(mesh: &Mesh, z: f64, tolerance: f64) -> Vec<usize> {
    mesh.nodes
        .iter()
        .enumerate()
        .filter_map(|(node, point)| ((point.z - z).abs() <= tolerance).then_some(node))
        .collect()
}

fn guided_boundary_conditions(bottom_nodes: &[usize], top_nodes: &[usize]) -> Vec<(usize, f64)> {
    let mut conditions = Vec::with_capacity(bottom_nodes.len() * 3 + top_nodes.len() * 2);
    for &node in bottom_nodes {
        conditions.extend([(3 * node, 0.0), (3 * node + 1, 0.0), (3 * node + 2, 0.0)]);
    }
    for &node in top_nodes {
        conditions.extend([(3 * node, 0.0), (3 * node + 1, 0.0)]);
    }
    conditions
}
