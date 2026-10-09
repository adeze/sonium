//! Common types and utilities for BEM/FEM room acoustics simulators
//!
//! This crate provides shared functionality between BEM and FEM room acoustics
//! simulators, including:
//!
//! - Room geometry definitions (rectangular, L-shaped rooms)
//! - Sound source configuration (position, directivity, crossover)
//! - JSON configuration loading/saving
//! - Output JSON formatting
//! - Visualization utilities

mod config;
pub mod conventions;
mod geometry;
pub mod io;
pub mod ir;
mod output;
mod parallelism;
pub mod runtime;
pub mod schema;
mod source;
pub mod symmetry;
mod types;

pub use config::*;
pub use conventions::*;
pub use geometry::*;
pub use ir::*;
pub use output::*;
pub use parallelism::{ParallelPolicy, SLICE_CELLS_PER_THREAD, use_parallel};
pub use source::*;
pub use symmetry::{
    ModeParity, SymmetryConfig, SymmetryPlane, detect_symmetry, map_to_reduced_domain,
};
pub use types::*;

/// Library version
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }
}
