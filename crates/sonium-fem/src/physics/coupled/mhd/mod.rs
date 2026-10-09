//! Inductionless magnetohydrodynamics for incompressible conducting fluids.
//!
//! The electric convention is `J = sigma(-grad(phi) + u cross B)`.
//! Lorentz force is `J cross B`. The velocity block stores `-J_motional cross
//! B` on the left-hand side, giving a positive-semidefinite magnetic damping
//! operator.

mod analytic;
mod assembler;
mod forms;
mod materials;

pub use analytic::*;
pub use assembler::*;
pub use forms::*;
pub use materials::*;
