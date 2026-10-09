//! Conjugate heat transfer across conforming fluid/solid regions.
//!
//! A shared H1 temperature space enforces temperature continuity strongly.
//! Cell-marker-routed conductivity makes the assembled weak form enforce equal
//! and opposite interface heat flux without a separately tuned interface term.

mod analytic;
mod assembler;
mod forms;
mod materials;

pub use analytic::*;
pub use assembler::*;
pub use forms::*;
pub use materials::*;
