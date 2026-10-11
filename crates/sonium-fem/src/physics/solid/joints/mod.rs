//! Joints, rigid bodies, and 0-D structural elements.
//!
//! This module owns discrete structural connections: axial and 3-D springs,
//! viscous dampers, point masses, 6x6 bushings, quasi-static spotwelds, and
//! linear rigid-body or diaphragm constraints. General multipoint constraint
//! builders for beam-shell-solid coupling live in
//! [`mpc_general`](crate::boundary::mpc_general).

// Rust guideline compliant 2026-02-21

mod imposed;
mod rbody;
mod rigid;
mod zero_d;

pub use imposed::*;
pub use rbody::*;
pub use rigid::*;
pub use zero_d::*;
