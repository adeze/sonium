//! DC and harmonic electrothermal coupling, including induction heating.
//!
//! Harmonic fields use the `exp(+i omega t)` convention. Reported harmonic
//! Joule density is time averaged: `0.5 sigma E dot conj(E)`.

mod analytic;
mod assembler;
mod forms;
mod materials;

pub use analytic::*;
pub use assembler::*;
pub use forms::*;
pub use materials::*;
