//! Lithium-ion porous-electrode (DFN/P2D) building blocks.
//!
//! The module provides spherical active-particle diffusion, porous-electrode
//! electrolyte/solid operators, Butler-Volmer kinetics, electrochemical heat
//! accounting, and reduced SPM analytical references.

mod analytic;
mod assembler;
mod forms;
mod materials;

pub use analytic::*;
pub use assembler::*;
pub use forms::*;
pub use materials::*;
