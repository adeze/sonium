//! Contact mechanics with augmented Lagrangian (Uzawa) enforcement
//!
//! Handles self-contact and surface-to-surface contact in 3D using a
//! node-to-surface formulation with augmented Lagrangian multiplier updates.
//!
//! # Architecture
//!
//! Contact lives outside the standard `BilinearForm`/`NonlinearForm` pipeline
//! because it is a surface-to-surface interaction requiring:
//! - Spatial search (BVH) for contact detection
//! - An outer Uzawa iteration wrapping Newton-Raphson
//! - Active set management (contacts activate/deactivate)
//!
//! The [`ContactManager`] owns all contact state and provides methods to
//! compute residual/tangent contributions that are added to the bulk system
//! during Newton assembly.
//!
//! # Usage
//!
//! ```ignore
//! use sonium_fem::physics::solid::contact::{ContactManager, ContactConfig};
//! use sonium_fem::solver::fem_newton_solve_contact;
//!
//! // Extract boundary triangles from your mesh
//! let boundary_faces: Vec<[usize; 3]> = /* ... */;
//! let ref_coords: Vec<f64> = /* flat [x0,y0,z0, x1,y1,z1, ...] */;
//!
//! let config = ContactConfig {
//!     epsilon: 1e6,
//!     search_radius: 0.05,
//!     ..Default::default()
//! };
//! let contact_mgr = ContactManager::new(&boundary_faces, ref_coords, config);
//!
//! let result = fem_newton_solve_contact(
//!     &hyperelastic_form, &space, &mesh, contact_mgr,
//!     &initial_guess, &fem_config,
//! )?;
//! ```

pub mod augmented_lagrangian;
pub mod detection;
pub mod gap;

pub use augmented_lagrangian::{ContactConfig, ContactManager, ContactResult};
pub use detection::{AabbTree, ContactSurface};
pub use gap::{ContactPair, ProjectionResult};

/// Hertz contact radius for a sphere on an elastic half-space.
pub fn hertz_sphere_contact_radius(force: f64, radius: f64, effective_modulus: f64) -> f64 {
    (3.0 * force * radius / (4.0 * effective_modulus)).powf(1.0 / 3.0)
}

/// Hertz line-contact half-width for cylinders.
pub fn hertz_cylinder_half_width(
    force_per_length: f64,
    radius: f64,
    effective_modulus: f64,
) -> f64 {
    (4.0 * force_per_length * radius / (std::f64::consts::PI * effective_modulus)).sqrt()
}

/// Average pressure under a flat rigid circular punch.
pub fn rigid_punch_average_pressure(force: f64, radius: f64) -> f64 {
    force / (std::f64::consts::PI * radius * radius)
}

/// Winkler foundation pressure for an elastic layer.
pub fn winkler_layer_pressure(displacement: f64, modulus: f64, thickness: f64) -> f64 {
    modulus * displacement / thickness
}

/// Penalty contact pressure for a signed normal gap.
pub fn penalty_contact_pressure(gap: f64, penalty: f64) -> f64 {
    if gap < 0.0 { -penalty * gap } else { 0.0 }
}
