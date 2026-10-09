//! Flat triangular shell elements
//!
//! Combines CST membrane (in-plane) with plate bending (out-of-plane)
//! for thin shell analysis. 6 DOFs per node: `(u, v, w, theta_x, theta_y, theta_z)`.
//!
//! Used for loudspeaker cone, surround, and dust cap modeling.

pub mod dkt;
pub mod elastoplastic;
mod element;
pub mod nonlinear;
mod section;

pub use elastoplastic::ElastoplasticShellForm;
pub use element::*;
pub use nonlinear::CorotationalShellForm;
pub use section::*;
