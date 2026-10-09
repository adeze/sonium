//! Low-frequency electromagnetics: eddy currents and magnetoquasistatics
//!
//! Frequency-domain A-formulation for eddy current analysis:
//!
//! ```text
//! curl(1/mu * curl(A)) + j*omega*sigma*A = J_s
//! ```
//!
//! where A is the magnetic vector potential, sigma is electrical conductivity,
//! and J_s is the source current density.
//!
//! Critical for loudspeaker motor design: eddy currents in the pole piece
//! and top plate reduce BL and increase impedance at high frequencies.
//!
//! Uses H(curl) Nédélec elements for A (same as Maxwell).

mod a_phi;
mod axisymmetric;
mod forms;
mod materials;

pub use a_phi::*;
pub use axisymmetric::*;
pub use forms::*;
pub use materials::*;
