//! Nonlinear solvers
//!
//! Provides Newton-Raphson and related methods for solving nonlinear systems `F(x) = 0`.

mod anderson;
mod newton;

pub use anderson::*;
pub use newton::*;
