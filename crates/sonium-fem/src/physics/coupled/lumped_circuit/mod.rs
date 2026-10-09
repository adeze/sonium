//! Lumped electrical circuit models for electromechanical coupling
//!
//! Models the electrical domain of transducers (loudspeakers, microphones)
//! as a 0D ODE system coupled to the FEM mechanical/acoustic solver.
//!
//! # Voice coil model
//!
//! ```text
//! V(t) = R·i + L(x)·di/dt + BL(x)·v
//! F_coil = BL(x)·i
//! ```
//!
//! The circuit advances in lockstep with the Newmark time integrator:
//! at each time step, `circuit_step` computes the new current given the
//! mechanical velocity, and returns the force to apply to the FEM.

pub mod acoustic_elements;
pub mod network;
pub mod two_port;
mod voice_coil;

pub use voice_coil::*;
