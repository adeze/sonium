//! Loudspeaker simulation and optimization.
//!
//! Parametric model of a sealed loudspeaker cabinet with woofer, tweeter,
//! waveguide, and internal separator. Generates surface meshes, runs BEM
//! acoustic simulation, produces CEA-2034 measurements, and optimises
//! cabinet parameters for flat response and smooth radiation.

pub mod model;
pub mod spline;
pub mod mesh_gen;
pub mod simulation;
pub mod cea2034_output;
pub mod optimization;
