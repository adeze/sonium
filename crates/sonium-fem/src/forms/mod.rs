//! Variational form traits for finite element assembly
//!
//! Physics modules implement [`BilinearForm`] and [`LinearForm`] to define
//! their weak formulations. The generic assembler in [`crate::assembly`]
//! evaluates these forms at quadrature points to build system matrices and vectors.

pub mod standard;
pub mod traits;

pub use standard::*;
pub use traits::*;
