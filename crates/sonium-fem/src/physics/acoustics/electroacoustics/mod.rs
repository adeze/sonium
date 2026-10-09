//! Electroacoustics: coupled fluid-structure interaction
//!
//! Couples the acoustic (Helmholtz) and structural (elasticity) domains via
//! a block system with interface coupling matrices.
//!
//! # Block system
//!
//! ```text
//! | K_f+iωD_f-k²M_f    -ρ_fω²C       0     | | p |   | f_f |
//! |      -C^T        K_s+iωC_s-ω²M_s -G^T  | | u | = | f_s |
//! |       0                 iωG       R+iωL | | i |   | v   |
//! ```
//!
//! where `p` is acoustic pressure (scalar H1) and `u` is structural displacement
//! (vector H1), coupled at the fluid-structure interface, and `i` contains the
//! distributed electrical current unknowns. The electrical block is optional.

pub mod cabinet_radiation;
mod coupling;
mod iterative;
pub mod lumped_extraction;
pub mod speaker;
pub mod thermal_coupling;
pub mod time_domain;
pub mod transducer;

pub use coupling::*;
pub use iterative::*;
pub use speaker::*;
