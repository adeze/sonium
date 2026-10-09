//! Fluid physics modules (stage-0 grouping).
//!
//! Each sub-module implements the generic form traits for one physics
//! domain; see `super` for the layer contract.

// Rust guideline compliant 2026-02-21

pub mod compressible_flow;
pub mod lubrication;
pub mod phase_field_flow;
pub mod porous_flow;
pub mod reacting_flow;
pub mod shallow_water;
pub mod stokes;
pub mod transport;
pub mod turbulence;
