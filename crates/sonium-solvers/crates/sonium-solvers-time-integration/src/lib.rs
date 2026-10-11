//! Time integration methods for second-order ODEs
//!
//! Solves `M u'' + C u' + K u = f(t)`. Implicit Newmark-beta / generalized-alpha
//! schemes live in [`newmark`]; the explicit central-difference scheme
//! (conditionally stable, lumped-mass friendly) lives in [`explicit`].

mod central_difference;
mod explicit;
mod newmark;
mod newmark_core;
mod revolve;
mod stiff;
mod transient_adjoint;

pub use central_difference::*;
pub use explicit::*;
pub use newmark::*;
pub use newmark_core::*;
pub use revolve::*;
pub use stiff::*;
pub use transient_adjoint::*;
