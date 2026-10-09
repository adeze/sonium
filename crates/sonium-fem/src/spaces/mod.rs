//! Function space abstractions for finite elements
//!
//! Defines the [`FunctionSpace`] trait that abstracts DOF mapping and basis
//! function evaluation across different element families: H1 (Lagrange nodal),
//! H(curl) (Nedelec edge), and H(div) (Raviart-Thomas face).

pub mod dg;
mod h1;
mod hcurl;
mod hdiv;
mod traits;

pub use dg::DGSpace;
pub mod contract;

pub use contract::*;
pub use h1::*;
pub use hcurl::*;
pub use hdiv::*;
pub use traits::*;
