//! General fluid-structure interaction coupling.
//!
//! The module supplies ALE-relative fluid convection, an energy-conserving
//! Lagrange-multiplier interface operator, conservative traction transfer, and
//! robust Aitken relaxation for partitioned time stepping. It composes the
//! existing Navier-Stokes and structural physics rather than replacing them.

mod analytic;
mod coupling;
mod forms;
mod materials;
mod mortar;

pub use analytic::*;
pub use coupling::*;
pub use forms::*;
pub use materials::*;
pub use mortar::*;
