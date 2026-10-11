//! Solid physics modules (stage-0 grouping).
//!
//! Each sub-module implements the generic form traits for one physics
//! domain; see `super` for the layer contract.

// Rust guideline compliant 2026-02-21

pub mod axisym_explicit;
pub mod axisymmetric;
pub mod beam;
pub mod biharmonic;
pub mod composites;
pub mod contact;
pub mod eigenmodes;
pub mod elasticity;
pub mod fracture;
pub mod joints;
pub mod mindlin_plate;
pub mod mixed_up;
pub mod nonlinear_solid;
pub mod plasticity;
pub mod shell;
