//! Bounded prescribed 3-D frictional contact reference.
//!
//! This slice exercises the production node-to-surface Coulomb law on the
//! same disconnected P1 tetrahedral two-block geometry as the hyperelastic
//! contact studies. The kinematics are prescribed so the result isolates
//! contact traction, force conservation, and the Coulomb cap; it is not a
//! product-scale frictional solve or a claim of self-contact dynamics.

use crate::contact3d::{opposing_boundary_faces, two_block_contact_mesh_for_geometry};
use crate::{StructuralExecutionProvenance, StructuralMechanicsError, StructuralModelAssumption};
use serde::{Deserialize, Serialize};
use sonium_fem::physics::solid::contact::{ContactConfig, ContactManager};

/// Schema version for [`Frictional3DContactResult`].
pub const FRICTIONAL_3D_CONTACT_SCHEMA_VERSION: u32 = 1;

/// Prescribed two-block frictional-contact study.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Frictional3DContactStudy {
    pub length_m: f64,
    pub width_m: f64,
    pub height_m: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Downward displacement magnitude of the upper block.
    pub normal_compression_m: f64,
    /// Positive x displacement magnitude of the upper block.
    pub tangential_shear_m: f64,
    pub contact_epsilon: f64,
    pub friction_coefficient: f64,
    pub contact_search_radius_m: f64,
    pub contact_bvh_margin_m: f64,
}

impl Frictional3DContactStudy {
    /// Compact fixture with both normal engagement and lateral slip.
    pub fn introductory() -> Self {
        Self {
            length_m: 0.40,
            width_m: 0.20,
            height_m: 0.40,
            nx: 2,
            ny: 1,
            nz: 2,
            normal_compression_m: 0.22,
            tangential_shear_m: 0.01,
            // Penalty for area-weighted tractions (same ~0.013 m^2 nodal
            // area as the displacement-controlled study; see contact3d.rs
            // for the probed window).
            contact_epsilon: 5.0e6,
            friction_coefficient: 0.30,
            contact_search_radius_m: 0.50,
            contact_bvh_margin_m: 1.0e-6,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("frictional_3d_contact.length_m", self.length_m),
            ("frictional_3d_contact.width_m", self.width_m),
            ("frictional_3d_contact.height_m", self.height_m),
            (
                "frictional_3d_contact.normal_compression_m",
                self.normal_compression_m,
            ),
            (
                "frictional_3d_contact.tangential_shear_m",
                self.tangential_shear_m,
            ),
            (
                "frictional_3d_contact.contact_epsilon",
                self.contact_epsilon,
            ),
            (
                "frictional_3d_contact.contact_search_radius_m",
                self.contact_search_radius_m,
            ),
            (
                "frictional_3d_contact.contact_bvh_margin_m",
                self.contact_bvh_margin_m,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if !self.friction_coefficient.is_finite() || self.friction_coefficient <= 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "frictional_3d_contact.friction_coefficient",
                value: self.friction_coefficient.to_string(),
            });
        }
        if self.nx == 0 || self.ny == 0 || self.nz == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "frictional_3d_contact.mesh_resolution",
                value: format!("({}, {}, {})", self.nx, self.ny, self.nz),
            });
        }
        Ok(())
    }
}

/// Versioned result for the prescribed frictional contact reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Frictional3DContactResult {
    pub schema_version: u32,
    pub study: Frictional3DContactStudy,
    pub nodes: usize,
    pub elements: usize,
    pub active_pairs: usize,
    pub max_penetration_m: f64,
    pub total_normal_reaction_n: f64,
    pub total_tangential_reaction_n: f64,
    pub coulomb_utilization: f64,
    pub net_contact_force_n: f64,
    pub friction_active: bool,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Run the bounded prescribed 3-D frictional contact reference.
pub fn run_frictional_3d_contact(
    study: &Frictional3DContactStudy,
) -> Result<Frictional3DContactResult, StructuralMechanicsError> {
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
            reason: format!("3-D frictional contact mesh connectivity: {errors:?}"),
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
            uzawa_tolerance: 1.0e-8,
            max_uzawa_iterations: 1,
            search_radius: study.contact_search_radius_m,
            bvh_margin: study.contact_bvh_margin_m,
            friction_coefficient: study.friction_coefficient,
            two_body: false,
        },
    );

    let mut displacement = vec![0.0; mesh.nodes.len() * 3];
    for (node, point) in mesh.nodes.iter().enumerate() {
        if point.z >= upper_contact_z - tolerance {
            displacement[3 * node] = study.tangential_shear_m;
            displacement[3 * node + 2] = -study.normal_compression_m;
        }
    }
    contact.detect_contact(&displacement);
    let mut residual = vec![0.0; displacement.len()];
    contact.contact_residual(&displacement, &mut residual);

    let total_normal_reaction_n: f64 = residual
        .as_chunks::<3>()
        .0
        .iter()
        .map(|value| value[2].abs())
        .sum();
    let total_tangential_reaction_n: f64 = residual
        .as_chunks::<3>()
        .0
        .iter()
        .map(|value| value[0].abs())
        .sum();
    let net_contact_force_n = {
        let net = residual
            .as_chunks::<3>()
            .0
            .iter()
            .fold([0.0; 3], |mut net, value| {
                for d in 0..3 {
                    net[d] += value[d];
                }
                net
            });
        net.iter().map(|value| value * value).sum::<f64>().sqrt()
    };
    if !total_normal_reaction_n.is_finite()
        || !total_tangential_reaction_n.is_finite()
        || !net_contact_force_n.is_finite()
        || total_normal_reaction_n <= 0.0
    {
        return Err(StructuralMechanicsError::NonlinearNonConvergence {
            model: StructuralModelAssumption::HyperelasticSolid3D,
            load_step: 0,
            iterations: 0,
            residual: total_normal_reaction_n.max(total_tangential_reaction_n),
        });
    }
    let coulomb_limit = study.friction_coefficient * total_normal_reaction_n;
    let coulomb_utilization = total_tangential_reaction_n / coulomb_limit;
    if coulomb_utilization > 1.0 + 1.0e-8 || net_contact_force_n > 1.0e-8 {
        return Err(StructuralMechanicsError::NonlinearNonConvergence {
            model: StructuralModelAssumption::HyperelasticSolid3D,
            load_step: 0,
            iterations: 0,
            residual: coulomb_utilization.max(net_contact_force_n),
        });
    }

    Ok(Frictional3DContactResult {
        schema_version: FRICTIONAL_3D_CONTACT_SCHEMA_VERSION,
        study: *study,
        nodes: mesh.nodes.len(),
        elements: mesh.elements.len(),
        active_pairs: contact.count_active_pairs(&displacement),
        max_penetration_m: contact.max_penetration(&displacement),
        total_normal_reaction_n,
        total_tangential_reaction_n,
        coulomb_utilization,
        net_contact_force_n,
        friction_active: total_tangential_reaction_n > 1.0e-12,
        model_assumption: StructuralModelAssumption::HyperelasticSolid3D,
        execution: StructuralExecutionProvenance::cpu_hyperelastic_3d_friction_contact_reference(),
    })
}
