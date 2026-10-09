//! Optimization framework for FEM-based design
//!
//! Supports:
//! - **Topology optimization**: density-based (SIMP/RAMP) for structural and magnetic design
//! - **Shape optimization**: adjoint-based sensitivity with mesh morphing
//! - **Frequency optimization**: target frequency response for acoustics
//!
//! # Architecture
//!
//! All optimization types share:
//! - [`Objective`] trait: objective function + adjoint sensitivity
//! - [`Constraint`] trait: inequality/equality constraints
//! - [`DesignVariable`]: density field, shape parameters, or material properties
//! - [`Filter`]: density/sensitivity filtering to regularize topology optimization
//! - [`Optimizer`]: MMA (Method of Moving Asymptotes) for constrained optimization

pub mod adjoint;
pub mod autograd;
pub mod constraints;
pub mod density;
pub mod filter;
pub mod objectives;
pub mod optimizer;
pub mod shroud_opt;
pub mod topology;

pub use adjoint::*;
pub use constraints::*;
pub use density::*;
pub use filter::*;
pub use objectives::*;
pub use optimizer::*;
pub use topology::*;
