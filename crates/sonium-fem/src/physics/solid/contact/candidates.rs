//! Frozen contact-pair handoff for the explicit pass.
//!
//! This module is the Phase 2 inter-lane contract (P2C2) and is
//! coordinator-owned: lanes 2B (producer) and 2D (consumer) import it
//! read-only. It defines the single pair record flowing from penalty
//! evaluation to tiebreak/rupture filtering, so neither lane invents
//! a parallel pair type.
//!
//! Conventions (frozen):
//!
//! - `gap < 0` means penetration (matches
//!   [`penalty_contact_pressure`](super::penalty_contact_pressure)).
//! - `normal` is the unit master-surface outward normal at the
//!   contact point, pointing from master toward slave.
//! - `force` is the penalty force applied to the SLAVE node
//!   (master gets the opposite through the shape-function weights
//!   held by the producer).
//! - SI units throughout.

// Rust guideline compliant 2026-02-21

/// One evaluated penalty contact pair: geometry plus slave force.
///
/// Produced by the lane 2B penalty pass, consumed by the lane 2D
/// tiebreak filter. Plain data: lanes construct it literally in
/// tests, so neither lane blocks on the other.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PenaltyPair {
    /// Slave node id (0-based).
    pub slave: usize,
    /// Signed normal gap in meters (negative = penetration).
    pub gap: f64,
    /// Unit master normal (master toward slave).
    pub normal: [f64; 3],
    /// Penalty force on the slave in newtons.
    pub force: [f64; 3],
}

impl PenaltyPair {
    /// Normal force magnitude (positive in compression).
    ///
    /// Contact push-out acts on the slave along `+normal`, so the raw
    /// projection is already compression-positive; bonded pull-back
    /// (tension) reads negative. Ruling P2-R1: an earlier revision
    /// negated this projection, which made recorded compression
    /// present as tension to rupture criteria.
    #[must_use]
    pub fn normal_force(&self) -> f64 {
        self.force[0] * self.normal[0]
            + self.force[1] * self.normal[1]
            + self.force[2] * self.normal[2]
    }

    /// Tangential force vector (friction response).
    #[must_use]
    pub fn tangential_force(&self) -> [f64; 3] {
        let normal = self.normal_force();
        [
            self.force[0] - normal * self.normal[0],
            self.force[1] - normal * self.normal[1],
            self.force[2] - normal * self.normal[2],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_tangential_split() {
        let pair = PenaltyPair {
            slave: 7,
            gap: -1.0e-5,
            normal: [0.0, 0.0, 1.0],
            force: [3.0, 4.0, 10.0],
        };
        assert!((pair.normal_force() - 10.0).abs() < 1.0e-12);
        let tangent = pair.tangential_force();
        assert!((tangent[0] - 3.0).abs() < 1.0e-12);
        assert!((tangent[1] - 4.0).abs() < 1.0e-12);
        assert!(tangent[2].abs() < 1.0e-12);
    }
}
