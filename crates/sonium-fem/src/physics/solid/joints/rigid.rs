//! Rigid bodies and diaphragms as linear constraints.
//!
//! Small-rotation rigid links express slave motion through master motion,
//! reusing the general [`MultiPointConstraint`](crate::boundary::periodic::MultiPointConstraint)
//! machinery. Large-rotation rigid bodies need corotational updates and stay
//! outside this linear scope.

// Rust guideline compliant 2026-02-21

use crate::boundary::periodic::{MpcTerm, MultiPointConstraint};
use num_complex::Complex64;

/// Real unit coefficient helper for rigid constraints.
fn real_term(dof: usize, coefficient: f64) -> MpcTerm {
    MpcTerm {
        dof,
        coefficient: Complex64::new(coefficient, 0.0),
    }
}

/// Small-rotation rigid link between two 6-DOF nodes.
///
/// The slave follows `u_s = u_m + theta_m cross r` with `r` the master to
/// slave offset, while slave rotations equal master rotations.
#[must_use]
pub fn rigid_link_constraints(
    master_node: usize,
    slave_node: usize,
    offset: [f64; 3],
    dofs_per_node: usize,
) -> Vec<MultiPointConstraint> {
    assert_eq!(dofs_per_node, 6, "rigid links need 6 DOFs per node");
    let m = master_node * 6;
    let s = slave_node * 6;
    let (rx, ry, rz) = (offset[0], offset[1], offset[2]);
    vec![
        // Slave translations pick up the theta cross r terms.
        MultiPointConstraint {
            slave: s,
            masters: vec![
                real_term(m, 1.0),
                real_term(m + 4, rz),
                real_term(m + 5, -ry),
            ],
        },
        MultiPointConstraint {
            slave: s + 1,
            masters: vec![
                real_term(m + 1, 1.0),
                real_term(m + 5, rx),
                real_term(m + 3, -rz),
            ],
        },
        MultiPointConstraint {
            slave: s + 2,
            masters: vec![
                real_term(m + 2, 1.0),
                real_term(m + 3, ry),
                real_term(m + 4, -rx),
            ],
        },
        // Slave rotations equal master rotations.
        MultiPointConstraint {
            slave: s + 3,
            masters: vec![real_term(m + 3, 1.0)],
        },
        MultiPointConstraint {
            slave: s + 4,
            masters: vec![real_term(m + 4, 1.0)],
        },
        MultiPointConstraint {
            slave: s + 5,
            masters: vec![real_term(m + 5, 1.0)],
        },
    ]
}

/// Rigid body with one master node and constrained slaves.
#[derive(Debug, Clone)]
pub struct RigidBody {
    /// Master node carrying the independent motion.
    pub master: usize,
    /// Slave nodes rigidly attached to the master.
    pub slaves: Vec<usize>,
    /// Reference positions indexed by node id.
    pub positions: Vec<[f64; 3]>,
}

impl RigidBody {
    /// Create a rigid body from master, slaves, and positions.
    #[must_use]
    pub fn new(master: usize, slaves: Vec<usize>, positions: Vec<[f64; 3]>) -> Self {
        Self {
            master,
            slaves,
            positions,
        }
    }

    /// Linear rigid-link constraints for every slave.
    #[must_use]
    pub fn constraints(&self) -> Vec<MultiPointConstraint> {
        let mut out = Vec::with_capacity(6 * self.slaves.len());
        for &slave in &self.slaves {
            let offset = [
                self.positions[slave][0] - self.positions[self.master][0],
                self.positions[slave][1] - self.positions[self.master][1],
                self.positions[slave][2] - self.positions[self.master][2],
            ];
            out.extend(rigid_link_constraints(self.master, slave, offset, 6));
        }
        out
    }
}

/// Rigid diaphragm constraining in-plane slave motion.
///
/// Slaves follow the master `ux, uy, rz` motion about the master position;
/// out-of-plane DOFs stay free for floor-slab idealizations.
#[derive(Debug, Clone)]
pub struct RigidDiaphragm {
    /// Master node carrying the diaphragm motion.
    pub master: usize,
    /// Slave nodes attached to the diaphragm.
    pub slaves: Vec<usize>,
    /// Reference positions indexed by node id.
    pub positions: Vec<[f64; 3]>,
}

impl RigidDiaphragm {
    /// Create a diaphragm from master, slaves, and positions.
    #[must_use]
    pub fn new(master: usize, slaves: Vec<usize>, positions: Vec<[f64; 3]>) -> Self {
        Self {
            master,
            slaves,
            positions,
        }
    }

    /// In-plane diaphragm constraints for every slave.
    #[must_use]
    pub fn constraints(&self) -> Vec<MultiPointConstraint> {
        let mut out = Vec::with_capacity(3 * self.slaves.len());
        let m = self.master * 6;
        for &slave in &self.slaves {
            let s = slave * 6;
            let dx = self.positions[slave][0] - self.positions[self.master][0];
            let dy = self.positions[slave][1] - self.positions[self.master][1];
            out.push(MultiPointConstraint {
                slave: s,
                masters: vec![real_term(m, 1.0), real_term(m + 5, -dy)],
            });
            out.push(MultiPointConstraint {
                slave: s + 1,
                masters: vec![real_term(m + 1, 1.0), real_term(m + 5, dx)],
            });
            out.push(MultiPointConstraint {
                slave: s + 5,
                masters: vec![real_term(m + 5, 1.0)],
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rigid_link_carries_rotation_arm() {
        let constraints = rigid_link_constraints(0, 1, [0.0, 2.0, 0.0], 6);
        assert_eq!(constraints.len(), 6);
        // Slave ux picks up -ry * theta_z from the y offset.
        let ux = &constraints[0];
        assert_eq!(ux.slave, 6);
        let arm = ux.masters.iter().find(|term| term.dof == 5).unwrap();
        assert!((arm.coefficient.re + 2.0).abs() < 1.0e-15);
    }

    #[test]
    fn diaphragm_leaves_out_of_plane_free() {
        let body = RigidDiaphragm::new(0, vec![1], vec![[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]);
        let constraints = body.constraints();
        assert_eq!(constraints.len(), 3);
        let slaves: Vec<usize> = constraints.iter().map(|c| c.slave).collect();
        assert_eq!(slaves, vec![6, 7, 11]);
        let uy = &constraints[1];
        let arm = uy.masters.iter().find(|term| term.dof == 5).unwrap();
        assert!((arm.coefficient.re - 3.0).abs() < 1.0e-15);
    }
}
