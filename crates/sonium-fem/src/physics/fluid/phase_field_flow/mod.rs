//! Cahn-Hilliard diffuse-interface two-phase flow.
//!
//! The phase convention is `c=-1` and `c=+1` in the pure phases. Chemical
//! potential is `mu = beta(c^3-c) - kappa laplacian(c)`. Capillary force uses
//! the energy-consistent `-c grad(mu)` representation.

mod analytic;
mod assembler;
mod forms;
mod materials;

pub use analytic::*;
pub use assembler::*;
pub use forms::*;
pub use materials::*;
