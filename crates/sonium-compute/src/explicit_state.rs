//! Frozen explicit-dynamics state for crash analysis.
//!
//! This module is the Phase 1 inter-lane contract (P1C1) and is
//! coordinator-owned: lanes consume it read-only and request changes
//! from the coordinator. It defines the only shared state shape for
//! the explicit path; all Phase 1 kernels take plain slices or this
//! struct, never sparse matrices or FEM types (the explicit path is
//! sparse-matrix-free by construction).
//!
//! Layout rules (frozen):
//!
//! - `f64` everywhere (matches the `dp` engine reference); no generics.
//! - 3-vectors are node-major: `[x0, y0, z0, x1, ...]`, length `3N`.
//! - Nodal scalars (`masses`) have length `N`.
//! - SI units throughout (m, s, kg, Pa); no hidden conversions.
//! - Node ids are 0-based contiguous.
//! - `forces` is scratch, zeroed at the top of every cycle.
//!
//! # Examples
//!
//! ```rust
//! use sonium_compute::explicit_state::ExplicitState;
//!
//! let mut state = ExplicitState::new(4);
//! assert_eq!(state.node_count(), 4);
//! state.velocities[0] = 1.0;
//! assert!(state.is_consistent());
//! ```

// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};

/// Restart/state schema tag. Bump on any layout change.
pub const EXPLICIT_STATE_SCHEMA: u32 = 1;

/// Lumped-mass explicit-dynamics state: positions, velocities,
/// diagonal masses, and force scratch for `N` nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExplicitState {
    /// Node-major positions `[x0, y0, z0, ...]`, length `3N`, meters.
    pub positions: Vec<f64>,
    /// Node-major velocities, length `3N`, meters per second.
    pub velocities: Vec<f64>,
    /// Lumped nodal masses, length `N`, kilograms. Must stay
    /// positive (mass scaling only ever increases entries).
    pub masses: Vec<f64>,
    /// Node-major force scratch, length `3N`, newtons. Zeroed at the
    /// top of every cycle; element kernels scatter-add into it.
    pub forces: Vec<f64>,
    /// Simulation time in seconds.
    pub time: f64,
    /// Completed cycle count.
    pub step: u64,
    /// Step used for (or proposed for) the next cycle, seconds.
    pub dt: f64,
}

impl ExplicitState {
    /// Allocate a zeroed state for `node_count` nodes.
    #[must_use]
    pub fn new(node_count: usize) -> Self {
        Self {
            positions: vec![0.0; 3 * node_count],
            velocities: vec![0.0; 3 * node_count],
            masses: vec![0.0; node_count],
            forces: vec![0.0; 3 * node_count],
            time: 0.0,
            step: 0,
            dt: 0.0,
        }
    }

    /// Node count derived from the position vector.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.positions.len() / 3
    }

    /// Layout validation: vector lengths agree and masses are finite.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        let nodes = self.node_count();
        self.positions.len() == 3 * nodes
            && self.velocities.len() == 3 * nodes
            && self.masses.len() == nodes
            && self.forces.len() == 3 * nodes
            && self.masses.iter().all(|mass| mass.is_finite())
    }

    /// Zero the force scratch (top-of-cycle reset).
    pub fn zero_forces(&mut self) {
        self.forces.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_invariants_hold() {
        let mut state = ExplicitState::new(5);
        assert_eq!(state.node_count(), 5);
        assert!(state.is_consistent());
        state.masses.fill(2.0);
        state.zero_forces();
        assert!(state.forces.iter().all(|force| *force == 0.0));
        assert!(state.is_consistent());
    }

    #[test]
    fn inconsistent_lengths_detected() {
        let mut state = ExplicitState::new(3);
        state.velocities.pop();
        assert!(!state.is_consistent());
    }
}
