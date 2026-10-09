//! Mixed displacement-pressure (u-p) linear elasticity.
//!
//! Solves `−div(2μ ε(u)) − ∇p = f` with `div u − c p = 0`, where the
//! pressure `p = λ div u` carries the volumetric response and
//! `c = 1/λ` is the volumetric compliance. As `ν → 1/2` the compliance
//! vanishes and the system tends to a Stokes-like saddle point, which is
//! why Taylor-Hood P2-P1 (or any inf-sup stable pair) stays accurate
//! where P1 displacement formulations lock.
//!
//! # Usage
//!
//! ```ignore
//! use sonium_fem::assembly::assemble_mixed_full_checked;
//! use sonium_fem::physics::solid::mixed_up::{
//!     DeviatoricElasticityForm, MixedElasticMaterial, PressureComplianceForm,
//!     UpCouplingForm, UpCouplingTransposeForm,
//! };
//!
//! let material = MixedElasticMaterial::from_young_poisson(2.0e6, 0.4999)?;
//! let system = assemble_mixed_full_checked(
//!     &DeviatoricElasticityForm::new(material.mu),
//!     &UpCouplingForm,
//!     &UpCouplingTransposeForm,
//!     &PressureComplianceForm::new(material.compliance),
//!     &u_space,
//!     &p_space,
//!     &mesh,
//!     4,
//! )?;
//! ```

mod forms;
mod materials;

pub use forms::*;
pub use materials::*;
