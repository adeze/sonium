//! Thin-film hydrodynamic lubrication.
//!
//! The module implements the incompressible Reynolds equation on an H1
//! surface/volume mesh. Positive pressure acts from the lubricating film onto
//! the adjacent solid; `film_rate < 0` therefore represents closing surfaces
//! and produces a positive squeeze pressure.

mod analytic;
mod assembler;
mod forms;
mod materials;

pub use analytic::*;
pub use assembler::*;
pub use forms::*;
pub use materials::*;
