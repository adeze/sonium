//! Coupled physics modules (stage-0 grouping).
//!
//! Each sub-module implements the generic form traits for one physics
//! domain; see `super` for the layer contract.

// Rust guideline compliant 2026-02-21

pub mod battery;
pub mod conjugate_heat_transfer;
pub mod electrochemistry;
pub mod electrothermal;
pub mod fsi;
pub mod lumped_circuit;
pub mod mhd;
pub mod piezoelectric;
pub mod thermoelasticity;
