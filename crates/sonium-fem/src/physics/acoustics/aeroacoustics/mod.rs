//! Aeroacoustics physics module
//!
//! Simulates one-way interaction of a background fluid flow with an
//! acoustic field. It includes direct propagation formulations and
//! free-field acoustic analogies.
//!
//! # Physics interfaces
//!
//! ## Acoustic analogies and DRP propagation
//!
//! - [`lighthill`] evaluates compact-quadrupole radiation from the Lighthill
//!   stress tensor, with exact near-field and radiating far-field forms.
//! - [`fwh`] evaluates stationary permeable Ffowcs Williams--Hawkings surfaces
//!   and reports thickness/loading contributions separately.
//! - [`drp_euler`] advances the uniform-flow 1-D linearized Euler equations
//!   with a seven-point Tam--Webb DRP stencil and exposes phase/group velocity.
//!
//! ## Linearized Potential Flow (LPF)
//!
//! Scalar formulation for irrotational mean flows. Solves the convected
//! Helmholtz equation for the velocity potential. Supports efficient
//! frequency sweeps via [`PrebakedAssembler<4>`](crate::assembly::PrebakedAssembler).
//! Modal source decomposition and transmission loss in duct acoustics
//! (e.g. turbofan engines) via port conditions.
//!
//! ## Linearized Euler Equations (LEE)
//!
//! Mixed velocity-pressure formulation for ideal gas flows. Computes
//! acoustic variations in velocity and pressure. Uses Taylor-Hood
//! (P2/P1) elements with block assembly. Captures refraction and
//! convection of acoustic waves by an arbitrary background flow.
//!
//! ## Linearized Navier-Stokes (LNS)
//!
//! The most detailed formulation, adding viscous stress and thermal
//! conduction to the LEE. Three-field system: velocity, pressure,
//! temperature. Accounts for boundary layer losses and damping of
//! acoustic waves in turbulent flow regions.
//!
//! # Boundary conditions
//!
//! - **Acoustic liners**: Ingard-Myers condition (infinitely thin BL)
//!   or Brambley condition (finite BL thickness) for locally-reacting
//!   liners in duct flows.
//! - **Duct modes**: Modal wavenumber computation for rectangular ducts
//!   with uniform mean flow.
//!
//! # Stabilisation
//!
//! GLS/SUPG stabilisation forms for convection-dominated regimes,
//! with frequency-dependent stabilisation parameter.
//!
//! # Which model when (mean-flow acoustics choice)
//!
//! - **No flow (M = 0)**: plain Helmholtz
//!   ([`crate::physics::acoustics::pressure`]); none of the machinery below is needed.
//! - **Irrotational low-Mach flow (M < 0.3), ducts, liners**: LPF
//!   ([`linearized_potential_flow`]) — one scalar unknown, convected
//!   Helmholtz, cheapest per frequency and sweep-friendly. Start here.
//! - **Rotational/sheared flow, refraction matters**: LEE
//!   ([`linearized_euler`]) — mixed velocity–pressure, captures convection
//!   and refraction by arbitrary background flow at ~4× the unknowns.
//! - **Viscous/thermal losses, boundary layers, turbulent damping**: LNS
//!   ([`linearized_navier_stokes`]) — three-field, the most expensive;
//!   use only where LEE demonstrably under-damps.
//! - **Radiated sound from a known unsteady flow (one-way coupling)**:
//!   compute the flow elsewhere, then propagate with [`lighthill`] (compact
//!   quadrupoles) or [`fwh`] (surfaces, thickness/loading split) instead of
//!   meshing the propagation domain.
//! - **Time-domain broadband pulses**: [`drp_euler`] (1-D uniform flow).
//!
//! Deliberately absent: the Galbrun equation and the Acoustic Perturbation
//! Equations (APE). Galbrun's displacement formulation removes the LEE
//! critical-layer singularity but needs H¹-conforming displacement elements
//! the crate does not ship; APE variants add source-filtering complexity
//! without changing the propagation operator. If LEE shows critical-layer
//! trouble in shear flow, add Galbrun before APE.
//! # Example
//!
//! ```ignore
//! use sonium_fem::physics::acoustics::aeroacoustics::{
//!     MeanFlow, build_lpf_assembler, LpfAssemblerExt,
//! };
//! use sonium_fem::basis::PolynomialDegree;
//!
//! let mesh = sonium_fem::mesh::unit_square_triangles(10);
//! let flow = MeanFlow::air_standard([34.3, 0.0, 0.0]); // M = 0.1
//! let assembler = build_lpf_assembler(&mesh, PolynomialDegree::P1, &flow);
//! let system = assembler.assemble_convected_helmholtz(&flow, 1000.0, &Default::default());
//! ```

pub mod axisymmetric;
pub mod boundary;
pub mod drp_euler;
pub mod duct_flow;
pub mod error;
pub mod fwh;
pub mod lighthill;
pub mod linearized_euler;
pub mod linearized_navier_stokes;
pub mod linearized_potential_flow;
pub mod mean_flow;
pub mod propeller_source;
pub mod stabilisation;

pub use axisymmetric::{
    AxiConvectedHelmholtzForm, AxiLpfAssembler, AxiLpfAssemblerExt, AxiScalarConvectionForm,
    AxiStreamingStiffnessForm, build_axi_lpf_assembler, build_axi_lpf_assembler_with_boundaries,
};
pub use boundary::{AcousticLiner, LinerModel};
pub use drp_euler::{
    AcousticBranch, DrpEuler1d, DrpEuler1dConfig, DrpStencil, DrpWaveState, NumericalDispersion,
    TAM_WEBB_DRP_7,
};
pub use duct_flow::DuctFlowField;
pub use error::AeroacousticsError;
pub use fwh::{FwhComponents, FwhSurfacePatch, stationary_fwh_pressure};
pub use lighthill::{LighthillQuadrupole, lighthill_far_field_pressure, lighthill_pressure};
pub use linearized_euler::{
    GradientCouplingForm, LeeSystem, MeanFlowGradientForm, VectorConvectionForm, build_lee_system,
    build_lee_system_uniform,
};
pub use linearized_navier_stokes::{BulkViscosityForm, LnsSystem, build_lns_system};
pub use linearized_potential_flow::{
    ConvectedHelmholtzForm, LpfAssembler, LpfAssemblerExt, ScalarConvectionForm,
    StreamingStiffnessForm, build_lpf_assembler,
};
pub use mean_flow::{MeanFlow, MeanFlowField};
pub use propeller_source::PropellerSource;
pub use stabilisation::aeroacoustic_tau;
