//! Elastoplastic material models
//!
//! Provides rate-independent J2 (von Mises) plasticity with isotropic and
//! kinematic hardening. Uses the radial return mapping algorithm for stress
//! update and computes the consistent (algorithmic) tangent for Newton convergence.
//!
//! Integrates with shell elements via through-thickness integration points
//! and with beam elements via fiber section integration.

mod material;
mod return_mapping;

pub use material::*;
pub use return_mapping::*;
