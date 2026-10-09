//! Bounded force-controlled 3-D hyperelastic contact reference.
//!
//! This module combines the deterministic two-block contact mesh with an
//! explicit distributed load vector. The lower external face is fixed, the
//! upper external face is guided in-plane, and the load is continued in the
//! unconstrained normal direction. It is a small CPU/f64 evidence path for
//! the FEM Newton/Uzawa contract, not a general frictional contact workflow.

use crate::contact3d::{opposing_boundary_faces, two_block_contact_mesh_for_geometry};
use crate::{StructuralExecutionProvenance, StructuralMechanicsError, StructuralModelAssumption};
use serde::{Deserialize, Serialize};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::Mesh;
use sonium_fem::physics::solid::contact::{ContactConfig, ContactManager};
use sonium_fem::physics::solid::nonlinear_solid::{HyperelasticForm, NeoHookean};
use sonium_fem::solver::{
    FemNewtonConfig, fem_newton_solve_contact, fem_newton_solve_contact_with_load,
};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};

/// Schema version for [`Hyperelastic3DForceContactResult`].
pub const HYPERELASTIC_3D_FORCE_CONTACT_SCHEMA_VERSION: u32 = 1;

/// Small force-controlled two-block hyperelastic contact study.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Hyperelastic3DForceContactStudy {
    pub length_m: f64,
    pub width_m: f64,
    pub height_m: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub youngs_modulus_pa: f64,
    pub poissons_ratio: f64,
    /// Positive magnitude of the downward force on the upper external face.
    pub total_force_n: f64,
    pub load_steps: usize,
    pub max_newton_iterations: usize,
    pub newton_residual_tolerance: f64,
    pub contact_epsilon: f64,
    pub contact_search_radius_m: f64,
    pub contact_bvh_margin_m: f64,
    pub contact_uzawa_tolerance: f64,
    pub max_uzawa_iterations: usize,
}

impl Hyperelastic3DForceContactStudy {
    /// Compact fixture whose load continuation reaches the initial contact gap.
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
            load_steps: 12,
            max_newton_iterations: 80,
            newton_residual_tolerance: 1.0e-5,
            // Penalty for area-weighted tractions (same ~0.013 m^2 nodal
            // area as the displacement-controlled study; the top of its
            // probed window, since the force-control tail needs it).
            contact_epsilon: 6.0e6,
            contact_search_radius_m: 0.50,
            contact_bvh_margin_m: 1.0e-6,
            // Gate sits above the inner-solver gap floor (~4e-12 m, i.e.
            // ~2.5e-5 Pa at this penalty); 1e-4 Pa is 1.7e-11 m.
            contact_uzawa_tolerance: 1.0e-4,
            max_uzawa_iterations: 60,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("hyperelastic_3d_force_contact.length_m", self.length_m),
            ("hyperelastic_3d_force_contact.width_m", self.width_m),
            ("hyperelastic_3d_force_contact.height_m", self.height_m),
            (
                "hyperelastic_3d_force_contact.youngs_modulus_pa",
                self.youngs_modulus_pa,
            ),
            (
                "hyperelastic_3d_force_contact.total_force_n",
                self.total_force_n,
            ),
            (
                "hyperelastic_3d_force_contact.newton_residual_tolerance",
                self.newton_residual_tolerance,
            ),
            (
                "hyperelastic_3d_force_contact.contact_epsilon",
                self.contact_epsilon,
            ),
            (
                "hyperelastic_3d_force_contact.contact_search_radius_m",
                self.contact_search_radius_m,
            ),
            (
                "hyperelastic_3d_force_contact.contact_bvh_margin_m",
                self.contact_bvh_margin_m,
            ),
            (
                "hyperelastic_3d_force_contact.contact_uzawa_tolerance",
                self.contact_uzawa_tolerance,
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
                field: "hyperelastic_3d_force_contact.poissons_ratio",
                value: self.poissons_ratio.to_string(),
            });
        }
        if self.nx == 0 || self.ny == 0 || self.nz == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "hyperelastic_3d_force_contact.mesh_resolution",
                value: format!("({}, {}, {})", self.nx, self.ny, self.nz),
            });
        }
        if self.load_steps == 0 || self.max_newton_iterations == 0 || self.max_uzawa_iterations == 0
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "hyperelastic_3d_force_contact.iterations",
                value: format!(
                    "load_steps={}, newton={}, uzawa={}",
                    self.load_steps, self.max_newton_iterations, self.max_uzawa_iterations
                ),
            });
        }
        Ok(())
    }
}

/// Diagnostics for one converged force/contact increment.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Hyperelastic3DForceContactStepResult {
    pub load_step: usize,
    pub load_factor: f64,
    pub applied_force_n: f64,
    pub average_top_displacement_m: f64,
    pub newton_iterations: usize,
    pub uzawa_iterations: usize,
    pub residual: f64,
    pub multiplier_update_norm: f64,
    pub active_pairs: usize,
    pub max_penetration_m: f64,
    pub converged: bool,
}

/// Versioned result for the force-controlled 3-D contact reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hyperelastic3DForceContactResult {
    pub schema_version: u32,
    pub study: Hyperelastic3DForceContactStudy,
    pub nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    pub bottom_external_nodes: usize,
    pub top_external_nodes: usize,
    pub bottom_contact_faces: usize,
    pub top_contact_faces: usize,
    pub steps: Vec<Hyperelastic3DForceContactStepResult>,
    pub contact_engaged_step: Option<usize>,
    pub max_penetration_m: f64,
    pub final_force_n: f64,
    pub final_average_top_displacement_m: f64,
    pub final_residual: f64,
    pub converged: bool,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Run the bounded force-controlled 3-D hyperelastic contact reference.
pub fn run_hyperelastic_3d_force_contact(
    study: &Hyperelastic3DForceContactStudy,
) -> Result<Hyperelastic3DForceContactResult, StructuralMechanicsError> {
    study.validate()?;

    let (mesh, lower_contact_z, upper_contact_z) = two_block_contact_mesh_for_geometry(
        study.length_m,
        study.width_m,
        study.height_m,
        study.nx,
        study.ny,
        study.nz,
    );
    mesh.validate_connectivity()
        .map_err(|errors| StructuralMechanicsError::InvalidMesh {
            reason: format!("3-D force-contact mesh connectivity: {errors:?}"),
        })?;

    let tolerance = 1.0e-12
        * study
            .length_m
            .max(study.width_m)
            .max(study.height_m)
            .max(1.0);
    let (bottom_contact_faces, top_contact_faces) =
        opposing_boundary_faces(&mesh, lower_contact_z, upper_contact_z, tolerance)?;
    let bottom_external_nodes = nodes_at_z(&mesh, 0.0, tolerance);
    let top_external_nodes = nodes_at_z(&mesh, study.height_m, tolerance);
    if bottom_external_nodes.is_empty() || top_external_nodes.is_empty() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: format!(
                "3-D force-contact external faces: bottom_nodes={}, top_nodes={}",
                bottom_external_nodes.len(),
                top_external_nodes.len()
            ),
        });
    }

    let mut contact_faces = bottom_contact_faces.clone();
    contact_faces.extend(top_contact_faces.iter().copied());
    let reference_coords = mesh
        .nodes
        .iter()
        .flat_map(|point| [point.x, point.y, point.z])
        .collect::<Vec<_>>();
    let mut contact = ContactManager::new(
        &contact_faces,
        reference_coords,
        ContactConfig {
            epsilon: study.contact_epsilon,
            uzawa_tolerance: study.contact_uzawa_tolerance,
            max_uzawa_iterations: study.max_uzawa_iterations,
            search_radius: study.contact_search_radius_m,
            bvh_margin: study.contact_bvh_margin_m,
            friction_coefficient: 0.0,
            two_body: false,
        },
    );

    let space = VectorH1Space::new(PolynomialDegree::P1, 3);
    let dofs = space.num_global_dofs(&mesh);
    let material = NeoHookean::from_young(study.youngs_modulus_pa, study.poissons_ratio, 3);
    let form = HyperelasticForm::new(material);
    let mut config = FemNewtonConfig::default();
    config.newton.max_iterations = study.max_newton_iterations;
    config.newton.absolute_tolerance = study.newton_residual_tolerance;
    config.newton.relative_tolerance = study.newton_residual_tolerance;
    config.inner_solver.max_iterations = study.max_newton_iterations.saturating_mul(20);

    let dirichlet_dofs =
        guided_contact_boundary_conditions(&bottom_external_nodes, &top_external_nodes);
    let seating_compression = upper_contact_z - lower_contact_z + 1.0e-4;
    let mut solution = vec![0.0; dofs];
    for seating_step in 1..=study.load_steps {
        let compression = seating_compression * seating_step as f64 / study.load_steps as f64;
        let seating_dofs =
            seating_boundary_conditions(&bottom_external_nodes, &top_external_nodes, compression);
        let seating = fem_newton_solve_contact(
            &form,
            &space,
            &mesh,
            &mut contact,
            &seating_dofs,
            &solution,
            &config,
        )
        .map_err(|error| StructuralMechanicsError::SolverFailure {
            stage: "3-D force-controlled hyperelastic contact seating solve",
            message: error.to_string(),
        })?;
        if !seating.0.converged
            || !seating.1.converged
            || !seating.0.residual.is_finite()
            || !seating.1.max_penetration.is_finite()
        {
            return Err(StructuralMechanicsError::NonlinearNonConvergence {
                model: StructuralModelAssumption::HyperelasticSolid3D,
                load_step: 0,
                iterations: seating.0.iterations.max(seating.1.uzawa_iterations),
                residual: seating.0.residual.max(seating.1.max_penetration),
            });
        }
        solution = seating.0.x;
    }
    let mut steps = Vec::with_capacity(study.load_steps);
    let mut contact_engaged_step = None;
    let mut max_penetration_m: f64 = 0.0;
    let mut final_average_top_displacement_m = 0.0;
    let mut final_residual = f64::INFINITY;

    for load_step in 1..=study.load_steps {
        let load_factor = load_step as f64 / study.load_steps as f64;
        let applied_force_n = study.total_force_n * load_factor;
        let nodal_force = -applied_force_n / top_external_nodes.len() as f64;
        let mut external_load = vec![0.0; dofs];
        for &node in &top_external_nodes {
            external_load[3 * node + 2] = nodal_force;
        }

        let (newton, contact_result) = fem_newton_solve_contact_with_load(
            &form,
            &space,
            &mesh,
            &mut contact,
            &dirichlet_dofs,
            &solution,
            &external_load,
            &config,
        )
        .map_err(|error| StructuralMechanicsError::SolverFailure {
            stage: "3-D force-controlled hyperelastic contact Newton/Uzawa solve",
            message: error.to_string(),
        })?;

        if !newton.converged
            || !contact_result.converged
            || !newton.residual.is_finite()
            || !contact_result.multiplier_update_norm.is_finite()
            || !contact_result.max_penetration.is_finite()
        {
            return Err(StructuralMechanicsError::NonlinearNonConvergence {
                model: StructuralModelAssumption::HyperelasticSolid3D,
                load_step,
                iterations: newton.iterations.max(contact_result.uzawa_iterations),
                residual: newton.residual.max(contact_result.multiplier_update_norm),
            });
        }

        solution = newton.x;
        let average_top_displacement_m = top_external_nodes
            .iter()
            .map(|&node| solution[3 * node + 2])
            .sum::<f64>()
            / top_external_nodes.len() as f64;
        if contact_result.max_penetration > tolerance {
            contact_engaged_step.get_or_insert(load_step);
        }
        max_penetration_m = max_penetration_m.max(contact_result.max_penetration);
        final_average_top_displacement_m = average_top_displacement_m;
        final_residual = newton.residual.max(contact_result.multiplier_update_norm);
        steps.push(Hyperelastic3DForceContactStepResult {
            load_step,
            load_factor,
            applied_force_n,
            average_top_displacement_m,
            newton_iterations: newton.iterations,
            uzawa_iterations: contact_result.uzawa_iterations,
            residual: newton.residual,
            multiplier_update_norm: contact_result.multiplier_update_norm,
            active_pairs: contact_result.active_pairs,
            max_penetration_m: contact_result.max_penetration,
            converged: true,
        });
    }

    Ok(Hyperelastic3DForceContactResult {
        schema_version: HYPERELASTIC_3D_FORCE_CONTACT_SCHEMA_VERSION,
        study: *study,
        nodes: mesh.nodes.len(),
        elements: mesh.elements.len(),
        dofs,
        bottom_external_nodes: bottom_external_nodes.len(),
        top_external_nodes: top_external_nodes.len(),
        bottom_contact_faces: bottom_contact_faces.len(),
        top_contact_faces: top_contact_faces.len(),
        steps,
        contact_engaged_step,
        max_penetration_m,
        final_force_n: study.total_force_n,
        final_average_top_displacement_m,
        final_residual,
        converged: true,
        model_assumption: StructuralModelAssumption::HyperelasticSolid3D,
        execution: StructuralExecutionProvenance::cpu_hyperelastic_3d_force_contact_reference(),
    })
}

fn nodes_at_z(mesh: &Mesh, z: f64, tolerance: f64) -> Vec<usize> {
    mesh.nodes
        .iter()
        .enumerate()
        .filter_map(|(node, point)| ((point.z - z).abs() <= tolerance).then_some(node))
        .collect()
}

fn guided_contact_boundary_conditions(
    bottom_nodes: &[usize],
    top_nodes: &[usize],
) -> Vec<(usize, f64)> {
    let mut conditions = Vec::with_capacity(bottom_nodes.len() * 3 + top_nodes.len() * 2);
    for &node in bottom_nodes {
        conditions.extend([(3 * node, 0.0), (3 * node + 1, 0.0), (3 * node + 2, 0.0)]);
    }
    for &node in top_nodes {
        conditions.extend([(3 * node, 0.0), (3 * node + 1, 0.0)]);
    }
    conditions
}

fn seating_boundary_conditions(
    bottom_nodes: &[usize],
    top_nodes: &[usize],
    prescribed_compression_m: f64,
) -> Vec<(usize, f64)> {
    let mut conditions = Vec::with_capacity(bottom_nodes.len() * 3 + top_nodes.len() * 3);
    for &node in bottom_nodes {
        conditions.extend([(3 * node, 0.0), (3 * node + 1, 0.0), (3 * node + 2, 0.0)]);
    }
    for &node in top_nodes {
        conditions.extend([
            (3 * node, 0.0),
            (3 * node + 1, 0.0),
            (3 * node + 2, -prescribed_compression_m),
        ]);
    }
    conditions
}
