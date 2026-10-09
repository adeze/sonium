//! Grey participating-media radiative transfer with discrete ordinates.
//!
//! Intensity directions point along propagation. On a boundary with outward
//! normal `n`, prescribed inflow applies where `s dot n < 0`. Angular weights
//! integrate over steradians and sum to `4 pi` in three dimensions.

mod analytic;
mod assembler;
mod forms;
mod materials;

pub use analytic::*;
pub use assembler::*;
pub use forms::*;
pub use materials::*;
