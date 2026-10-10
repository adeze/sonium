//! Elastoplastic material models
//!
//! Provides rate-independent J2 (von Mises) plasticity with isotropic and
//! kinematic hardening. Uses the radial return mapping algorithm for stress
//! update and computes the consistent (algorithmic) tangent for Newton convergence.
//!
//! Integrates with shell elements via through-thickness integration points
//! and with beam elements via fiber section integration.
//!
//! Contract C1 (shared stress-update shape) lives in [`update`]: the
//! [`StressUpdateModel`](update::StressUpdateModel) trait plus the frozen
//! C1 `StressUpdate` struct. The legacy dynamically sized `StressUpdate`
//! stays at this root for existing shell and beam callers; new fiber and
//! layer code should use the C1 shape in [`update`]. Finite-strain J2
//! plasticity with the Green-Naghdi rate lives in [`finite_strain`].

pub mod crash;
pub mod finite_strain;
mod material;
mod return_mapping;
pub mod update;

pub use crash::{
    CowperSymondsJ2, CrushableFoamMaterial, CrushableFoamState, JohnsonCookMaterial,
    JohnsonCookState, PiecewiseLinearPlasticity, UniaxialPoint, drive_uniaxial_stress,
    drive_uniaxial_stress_rate,
};
pub use finite_strain::{FiniteStrainJ2Material, FiniteStrainState};
pub use material::*;
pub use return_mapping::*;
pub use update::StressUpdateModel;
