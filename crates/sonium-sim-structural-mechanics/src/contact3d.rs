//! Displacement-controlled 3-D hyperelastic contact reference.
//!
//! This module contains a small but genuine FEM contact path: two P1
//! tetrahedral hyperelastic blocks are compressed across an initial gap
//! between a fixed lower face and a prescribed upper face. Contact is
//! enforced by Sonium FEM's augmented-Lagrangian node-to-surface solver.
//! It is an auditable reference study, not a general self-contact,
//! frictional, or product-scale contact workflow.

use crate::{StructuralExecutionProvenance, StructuralMechanicsError, StructuralModelAssumption};
use serde::{Deserialize, Serialize};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{Mesh, box_mesh_tetrahedra};
use sonium_fem::physics::solid::contact::{ContactConfig, ContactManager};
use sonium_fem::physics::solid::nonlinear_solid::{HyperelasticForm, NeoHookean};
use sonium_fem::solver::{FemNewtonConfig, fem_newton_solve_contact};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};

/// Schema version for [`Hyperelastic3DContactResult`].
pub const HYPERELASTIC_3D_CONTACT_SCHEMA_VERSION: u32 = 1;

/// Small displacement-controlled 3-D contact study.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Hyperelastic3DContactStudy {
    pub length_m: f64,
    pub width_m: f64,
    pub height_m: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub youngs_modulus_pa: f64,
    pub poissons_ratio: f64,
    /// Downward displacement prescribed on the final step.
    pub prescribed_compression_m: f64,
    pub load_steps: usize,
    pub max_newton_iterations: usize,
    pub newton_residual_tolerance: f64,
    pub contact_epsilon: f64,
    pub contact_search_radius_m: f64,
    pub contact_bvh_margin_m: f64,
    pub contact_uzawa_tolerance: f64,
    pub max_uzawa_iterations: usize,
}

impl Hyperelastic3DContactStudy {
    /// Compact two-block case that closes and slightly penetrates the gap.
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
            prescribed_compression_m: 0.201,
            load_steps: 12,
            max_newton_iterations: 80,
            newton_residual_tolerance: 1.0e-9,
            // Penalty for area-weighted tractions: nodal area here is
            // ~0.013 m^2. 5e6 sits mid-window between the slow Uzawa tail
            // (<=2e6 exhausts 60 sweeps) and the GMRES/ILU breakdown
            // cliff (>=1e7, erratic); probed 2026-09-28.
            contact_epsilon: 5.0e6,
            contact_search_radius_m: 0.50,
            contact_bvh_margin_m: 1.0e-6,
            contact_uzawa_tolerance: 1.0e-8,
            max_uzawa_iterations: 60,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("hyperelastic_3d_contact.length_m", self.length_m),
            ("hyperelastic_3d_contact.width_m", self.width_m),
            ("hyperelastic_3d_contact.height_m", self.height_m),
            (
                "hyperelastic_3d_contact.youngs_modulus_pa",
                self.youngs_modulus_pa,
            ),
            (
                "hyperelastic_3d_contact.prescribed_compression_m",
                self.prescribed_compression_m,
            ),
            (
                "hyperelastic_3d_contact.newton_residual_tolerance",
                self.newton_residual_tolerance,
            ),
            (
                "hyperelastic_3d_contact.contact_epsilon",
                self.contact_epsilon,
            ),
            (
                "hyperelastic_3d_contact.contact_search_radius_m",
                self.contact_search_radius_m,
            ),
            (
                "hyperelastic_3d_contact.contact_bvh_margin_m",
                self.contact_bvh_margin_m,
            ),
            (
                "hyperelastic_3d_contact.contact_uzawa_tolerance",
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
        if !(0.0..0.5).contains(&self.poissons_ratio) || !self.poissons_ratio.is_finite() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "hyperelastic_3d_contact.poissons_ratio",
                value: self.poissons_ratio.to_string(),
            });
        }
        if self.prescribed_compression_m >= 2.0 * self.height_m {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "hyperelastic_3d_contact.prescribed_compression_m",
                value: "compression must remain below twice the block height".to_string(),
            });
        }
        if self.nx == 0 || self.ny == 0 || self.nz == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "hyperelastic_3d_contact.mesh_resolution",
                value: format!("({}, {}, {})", self.nx, self.ny, self.nz),
            });
        }
        if self.load_steps == 0 || self.max_newton_iterations == 0 || self.max_uzawa_iterations == 0
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "hyperelastic_3d_contact.iterations",
                value: format!(
                    "load_steps={}, newton={}, uzawa={}",
                    self.load_steps, self.max_newton_iterations, self.max_uzawa_iterations
                ),
            });
        }
        Ok(())
    }
}

/// One converged compression/contact step.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Hyperelastic3DContactStepResult {
    pub load_step: usize,
    pub load_factor: f64,
    pub prescribed_compression_m: f64,
    pub newton_iterations: usize,
    pub uzawa_iterations: usize,
    pub residual: f64,
    pub multiplier_update_norm: f64,
    pub active_pairs: usize,
    pub max_penetration_m: f64,
    pub converged: bool,
}

/// Result and provenance for the 3-D hyperelastic contact reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hyperelastic3DContactResult {
    pub schema_version: u32,
    pub study: Hyperelastic3DContactStudy,
    pub nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    pub bottom_contact_faces: usize,
    pub top_contact_faces: usize,
    pub steps: Vec<Hyperelastic3DContactStepResult>,
    pub contact_engaged_step: Option<usize>,
    pub max_penetration_m: f64,
    pub final_residual: f64,
    pub converged: bool,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Run the bounded 3-D hyperelastic block/contact reference.
pub fn run_hyperelastic_3d_contact(
    study: &Hyperelastic3DContactStudy,
) -> Result<Hyperelastic3DContactResult, StructuralMechanicsError> {
    study.validate()?;
    let (mesh, lower_contact_z, upper_contact_z) = two_block_contact_mesh(study);
    mesh.validate_connectivity()
        .map_err(|errors| StructuralMechanicsError::InvalidMesh {
            reason: format!("3-D contact mesh connectivity: {errors:?}"),
        })?;
    let tolerance = 1.0e-12
        * study
            .length_m
            .max(study.width_m)
            .max(study.height_m)
            .max(1.0);
    let (bottom_faces, top_faces) =
        opposing_boundary_faces(&mesh, lower_contact_z, upper_contact_z, tolerance)?;
    let mut contact_faces = bottom_faces.clone();
    contact_faces.extend(top_faces.iter().copied());
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
    let mut solution = vec![0.0; dofs];
    let mut steps = Vec::with_capacity(study.load_steps);
    let mut contact_engaged_step = None;
    let mut max_penetration_m: f64 = 0.0;
    let mut final_residual = f64::INFINITY;

    for load_step in 1..=study.load_steps {
        let load_factor = load_step as f64 / study.load_steps as f64;
        let prescribed_compression_m = study.prescribed_compression_m * load_factor;
        let dirichlet_dofs = compression_boundary_conditions(
            &mesh,
            study.height_m,
            prescribed_compression_m,
            tolerance,
        );
        let (newton, contact_result) = fem_newton_solve_contact(
            &form,
            &space,
            &mesh,
            &mut contact,
            &dirichlet_dofs,
            &solution,
            &config,
        )
        .map_err(|error| StructuralMechanicsError::SolverFailure {
            stage: "3-D hyperelastic augmented-Lagrangian contact",
            message: error.to_string(),
        })?;
        if !newton.converged || !contact_result.converged {
            return Err(StructuralMechanicsError::NonlinearNonConvergence {
                model: StructuralModelAssumption::HyperelasticSolid3D,
                load_step,
                iterations: newton.iterations.max(contact_result.uzawa_iterations),
                residual: newton.residual.max(contact_result.multiplier_update_norm),
            });
        }
        solution = newton.x;
        if contact_result.active_pairs > 0 || contact_result.max_penetration > tolerance {
            contact_engaged_step.get_or_insert(load_step);
        }
        max_penetration_m = max_penetration_m.max(contact_result.max_penetration);
        final_residual = newton.residual.max(contact_result.multiplier_update_norm);
        steps.push(Hyperelastic3DContactStepResult {
            load_step,
            load_factor,
            prescribed_compression_m,
            newton_iterations: newton.iterations,
            uzawa_iterations: contact_result.uzawa_iterations,
            residual: newton.residual,
            multiplier_update_norm: contact_result.multiplier_update_norm,
            active_pairs: contact_result.active_pairs,
            max_penetration_m: contact_result.max_penetration,
            converged: true,
        });
    }

    Ok(Hyperelastic3DContactResult {
        schema_version: HYPERELASTIC_3D_CONTACT_SCHEMA_VERSION,
        study: *study,
        nodes: mesh.nodes.len(),
        elements: mesh.elements.len(),
        dofs,
        bottom_contact_faces: bottom_faces.len(),
        top_contact_faces: top_faces.len(),
        steps,
        contact_engaged_step,
        max_penetration_m,
        final_residual,
        converged: true,
        model_assumption: StructuralModelAssumption::HyperelasticSolid3D,
        execution: StructuralExecutionProvenance::cpu_hyperelastic_3d_contact_reference(),
    })
}

/// Opposing contact boundary faces: `(bottom_triangles, top_triangles)`.
type OpposingBoundaryFaces = (Vec<[usize; 3]>, Vec<[usize; 3]>);

pub(crate) fn opposing_boundary_faces(
    mesh: &Mesh,
    lower_contact_z: f64,
    upper_contact_z: f64,
    tolerance: f64,
) -> Result<OpposingBoundaryFaces, StructuralMechanicsError> {
    let mut bottom = Vec::new();
    let mut top = Vec::new();
    for boundary in &mesh.boundaries {
        if boundary.nodes.len() != 3 {
            continue;
        }
        let nodes = [boundary.nodes[0], boundary.nodes[1], boundary.nodes[2]];
        if nodes.iter().any(|node| *node >= mesh.nodes.len()) {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: "3-D contact boundary references an unknown node".to_string(),
            });
        }
        let mut z_values = nodes.iter().map(|node| mesh.nodes[*node].z);
        if z_values
            .clone()
            .all(|z| (z - lower_contact_z).abs() <= tolerance)
        {
            bottom.push(orient_face(mesh, nodes, 1.0));
        } else if z_values.all(|z| (z - upper_contact_z).abs() <= tolerance) {
            top.push(orient_face(mesh, nodes, -1.0));
        }
    }
    if bottom.is_empty() || top.is_empty() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: format!(
                "3-D contact requires opposing triangular faces (bottom={}, top={})",
                bottom.len(),
                top.len()
            ),
        });
    }
    Ok((bottom, top))
}

pub(crate) fn two_block_contact_mesh(study: &Hyperelastic3DContactStudy) -> (Mesh, f64, f64) {
    two_block_contact_mesh_for_geometry(
        study.length_m,
        study.width_m,
        study.height_m,
        study.nx,
        study.ny,
        study.nz,
    )
}

/// Build the disconnected lower/upper block mesh used by the contact studies.
///
/// The helper keeps the two contact formulations on the same deterministic
/// topology while allowing their load-control study inputs to remain separate.
pub(crate) fn two_block_contact_mesh_for_geometry(
    length_m: f64,
    width_m: f64,
    height_m: f64,
    nx: usize,
    ny: usize,
    nz: usize,
) -> (Mesh, f64, f64) {
    let block_height = height_m * 0.25;
    let gap = height_m * 0.50;
    let upper_block_bottom = block_height + gap;
    let mut mesh = Mesh::new(3);

    for (z_min, z_max) in [(0.0, block_height), (upper_block_bottom, height_m)] {
        let block = box_mesh_tetrahedra(0.0, length_m, 0.0, width_m, z_min, z_max, nx, ny, nz);
        let node_offset = mesh.nodes.len();
        for point in block.nodes {
            mesh.add_node(point);
        }
        for element in block.elements {
            let nodes = element
                .nodes
                .into_iter()
                .map(|node| node + node_offset)
                .collect();
            mesh.add_element(element.element_type, nodes);
        }
    }
    mesh.detect_boundaries();

    (mesh, block_height, upper_block_bottom)
}

fn orient_face(mesh: &Mesh, nodes: [usize; 3], desired_z_sign: f64) -> [usize; 3] {
    let p0 = mesh.nodes[nodes[0]];
    let p1 = mesh.nodes[nodes[1]];
    let p2 = mesh.nodes[nodes[2]];
    let e1 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
    let e2 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
    let normal_z = e1[0] * e2[1] - e1[1] * e2[0];
    if normal_z * desired_z_sign < 0.0 {
        [nodes[0], nodes[2], nodes[1]]
    } else {
        nodes
    }
}

pub(crate) fn compression_boundary_conditions(
    mesh: &Mesh,
    height_m: f64,
    prescribed_compression_m: f64,
    tolerance: f64,
) -> Vec<(usize, f64)> {
    let mut conditions = Vec::new();
    for (node, point) in mesh.nodes.iter().enumerate() {
        if point.z.abs() <= tolerance {
            conditions.extend([(3 * node, 0.0), (3 * node + 1, 0.0), (3 * node + 2, 0.0)]);
        } else if (point.z - height_m).abs() <= tolerance {
            conditions.extend([
                (3 * node, 0.0),
                (3 * node + 1, 0.0),
                (3 * node + 2, -prescribed_compression_m),
            ]);
        }
    }
    conditions
}
