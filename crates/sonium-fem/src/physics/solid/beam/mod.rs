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

mod corotational3d;
mod curved;
mod element;
mod element3d;
mod fiber;
mod fiber_beam;
mod section;
mod section3d;
pub mod shape_opt;
mod tapered;
mod warping;

pub use corotational3d::*;
pub use curved::*;
pub use element::*;
pub use element3d::*;
pub use fiber::*;
pub use fiber_beam::*;
pub use section::*;
pub use section3d::*;
pub use tapered::*;
pub use warping::*;
pub use shape_opt::{HBeamShapeOpt, ShapeOptResult};
