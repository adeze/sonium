//! Nonlinear finite-rate reacting-flow and combustion forms.
//!
//! Species source terms use the production convention: positive values add a
//! species. Reaction heat is positive for exothermic heat release into the
//! thermal equation.

mod analytic;
mod assembler;
mod forms;
mod materials;

pub use analytic::*;
pub use assembler::*;
pub use forms::*;
pub use materials::*;
