//! Nonlinear solid mechanics (hyperelasticity)
//!
//! Total Lagrangian formulation for large-deformation solid mechanics.
//! Material models define strain energy `W(F)`, stress `P = dW/dF`, and
//! tangent `A = d²W/dF²`. The Newton-Raphson solver from `sonium-solvers`
//! drives the nonlinear iteration.
//!
//! # Usage
//!
//! ```ignore
//! use sonium_fem::physics::solid::nonlinear_solid::{HyperelasticForm, NeoHookean};
//! use sonium_fem::solver::fem_newton_solve;
//! use sonium_fem::spaces::VectorH1Space;
//!
//! let material = NeoHookean::from_young(1e6, 0.3, 3);
//! let form = HyperelasticForm::new(material);
//! let space = VectorH1Space::new(PolynomialDegree::P1, 3);
//! let result = fem_newton_solve(&form, &space, &mesh, &initial_guess, &config)?;
//! ```

mod eas;
mod fbar;
mod forms;
mod hex;
mod hex_explicit;
mod hex_plastic;
mod materials;
mod tet_penta;

pub use eas::*;
pub use fbar::*;
pub use forms::*;
pub use hex::*;
pub use hex_explicit::*;
pub use hex_plastic::*;
pub use materials::*;
pub use tet_penta::*;
