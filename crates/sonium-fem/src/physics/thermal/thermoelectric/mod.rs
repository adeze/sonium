//! Coupled thermoelectric charge and heat transport.
//!
//! Sign convention: electric field `E = -grad(V)` and conventional current
//! `J = sigma * (E - alpha grad(T))`. Positive heat flux is
//! `q = -k grad(T) + alpha T J`.

mod analytic;
mod assembler;
mod forms;
mod materials;

pub use analytic::*;
pub use assembler::*;
pub use forms::*;
pub use materials::*;
