//! Finite element basis functions
//!
//! - **Lagrange**: H1-conforming nodal basis (P1, P2, P3 for simplices; Q1, Q2 for quads/hexes)
//! - **Nédélec**: H(curl)-conforming edge basis (lowest order, first kind)
//! - **Raviart-Thomas**: H(div)-conforming face basis (lowest order RT0)

mod lagrange;
pub mod nedelec;
pub mod raviart_thomas;
mod shape;

pub use lagrange::*;
pub use shape::*;
