//! Beam elements for frame analysis
//!
//! 2-node beam elements with axial + bending DOFs:
//! - **Euler-Bernoulli**: classical beam theory (no shear deformation)
//! - **Timoshenko**: includes shear deformation (better for thick beams)
//!
//! DOFs per node in 2D: `(u, v, theta)` — axial, transverse, rotation.
//!
//! Uses analytical element matrices (Hermite cubics for bending,
//! linear interpolation for axial) — no numerical quadrature needed.

mod element;
mod section;
pub mod shape_opt;

pub use element::*;
pub use section::*;
pub use shape_opt::{HBeamShapeOpt, ShapeOptResult};
