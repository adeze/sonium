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
//! - [`lighthill_2d_axi`] adds planar quadrupoles and spinning ring sources
//!   of any azimuthal order, sharing the 3-D kernel sample by sample.
//! - [`distributed`] sums mesh-cell quadrupole moments for non-compact
//!   sources, including explicit compact-depth 2-D extrusion.
//! - [`fwh`] evaluates stationary permeable Ffowcs Williams--Hawkings surfaces
//!   and reports thickness/loading contributions separately.
//! - [`fwh_2d_axi`] adds planar contours, spinning revolved bands of any
//!   azimuthal order, and uniform-flow convected Kirchhoff surfaces. Note
//!   the convected surfaces carry `(p, ∇p)`: pointwise `(p, u_n)` data
//!   underdetermines the uniform-flow representation.
//! - [`drp_euler`] advances the uniform-flow 1-D linearized Euler equations
//!   with a seven-point Tam--Webb DRP stencil and exposes phase/group velocity.
//! - [`drp_structured`] extends the Tam--Webb advance to 2-D/3-D Cartesian
//!   grids with characteristic outflow plus sponge layers.
//! - [`drp_axi`] advances axisymmetric modal pairs (`cos mφ`, `sin mφ`) on
//!   meridional grids for duct and propeller tones.
//! - [`wave_diagnostics`] tracks packet energy centroids into measured
//!   group velocities shared by every time-domain propagator.
//! - [`dg_lee`] advances the uniform-flow LEE on unstructured triangles
//!   and tetrahedra (P1--P3) with an exact upwind flux, SSPRK(3,3),
//!   outflow/slip-wall/prescribed boundaries, and a structured/DG
//!   equivalence gate.
//! - [`snapshot`] couples time-domain flow snapshots to the analogies:
//!   Hann DFT peak phasors, Lighthill stress tensors, FW-H patches, and
//!   spinning-mode extraction for rings and revolved strips.
//! - [`propeller_source`] evaluates rotor-alone tones with polar directivity
//!   plus Tyler-Sofrin interaction orders for rotor-stator stages.
//! - [`duct_modes`] screens annular-duct cut-on, lined-wall roots, and
//!   rotor-stator interaction propagation at operating conditions.
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
//! [`axisymmetric_lee`] carries the same operator to meridional meshes
//! with spinning ansatz `exp(imφ)` of any azimuthal order.
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
//! - **Time-domain broadband pulses**: [`drp_euler`] (1-D uniform flow),
//!   [`drp_structured`] (Cartesian 2-D/3-D with sponge outflow),
//!   [`drp_axi`] (axisymmetric modal pair of any azimuthal order),
//!   [`dg_lee`] (unstructured DG on simplices), and [`wave_diagnostics`]
//!   (measured group-velocity gates shared by all time-domain propagators).
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
pub mod axisymmetric_lee;
pub mod boundary;
pub mod dg_lee;
pub mod distributed;
pub mod drp_axi;
pub mod drp_euler;
pub mod drp_structured;
pub mod duct_flow;
pub mod duct_modes;
pub mod error;
pub mod fwh;
pub mod fwh_2d_axi;
pub mod lighthill;
pub mod lighthill_2d_axi;
pub mod linearized_euler;
pub mod linearized_navier_stokes;
pub mod linearized_potential_flow;
pub mod mean_flow;
pub mod propeller_source;
pub mod replay;
pub mod snapshot;
pub mod stabilisation;
pub mod wave_diagnostics;

pub use axisymmetric::{
    AxiConvectedHelmholtzForm, AxiLpfAssembler, AxiLpfAssemblerExt, AxiScalarConvectionForm,
    AxiStreamingStiffnessForm, build_axi_lpf_assembler, build_axi_lpf_assembler_with_boundaries,
};
pub use axisymmetric_lee::{
    AxiLeeAxisDofs, AxiLeeSystem, AxiMeridionalDivergenceForm, AxiVectorAxialConvectionForm,
    AxiVectorMassForm, axi_lee_axis_dofs, build_axi_lee_system,
};
pub use boundary::{AcousticLiner, LinerModel};
pub use dg_lee::{
    DgLeeBoundary, DgLeeConfig, DgLeeSolver, DgLeeState, ExteriorStateFn, QuadratureSample,
    acoustic_energy_dg, l2_error, quadrature_samples,
};
pub use distributed::{
    AreaQuadrupoleCell, CellGeometry, VolumeQuadrupoleCell, cell_centroid_and_measure,
    extrude_cells_to_3d, sum_quadrupole_field, sum_quadrupole_field_2d,
};
pub use drp_axi::{
    AxiAxialFace, AxiDrpConfig, AxiDrpEuler, AxiDrpState, AxiModalState, AxiRadialFace,
    DuctModeInit, acoustic_energy_axi, axial_energy_profile, duct_mode_init,
};
pub use drp_euler::{
    AcousticBranch, DrpEuler1d, DrpEuler1dConfig, DrpStencil, DrpWaveState, NumericalDispersion,
    TAM_WEBB_DRP_7,
};
pub use drp_structured::{
    DrpStructuredConfig, DrpStructuredEuler, DrpStructuredState, PlaneWaveDispersion,
    StructuredAxis, StructuredFace, acoustic_energy, transverse_energy_profile,
};
pub use duct_flow::DuctFlowField;
pub use duct_modes::{
    InteractionCutOn, ModeCutOn, annular_mode_cuton, annular_mode_cuton_with_flow, lined_wall_root,
    rotor_stator_cuton,
};
pub use error::AeroacousticsError;
pub use fwh::{FwhComponents, FwhSurfacePatch, stationary_fwh_pressure};
pub use fwh_2d_axi::{
    FwhContourSegment, FwhKirchhoffPatch, FwhKirchhoffSegment, FwhRevolvedPatch,
    RevolvedQuadrature, stationary_fwh_pressure_2d, stationary_fwh_ring_pressure,
    stationary_fwh_ring_pressure_fixed, stationary_kirchhoff_pressure_2d_convected,
    stationary_kirchhoff_pressure_convected,
};
pub use lighthill::{
    LighthillQuadrupole, lighthill_acoustic_power, lighthill_far_field_pressure, lighthill_pressure,
};
pub use lighthill_2d_axi::{
    LighthillQuadrupole2d, LighthillRingSource, lighthill_far_field_pressure_2d,
    lighthill_pressure_2d, lighthill_ring_pressure, lighthill_ring_pressure_fixed,
};
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
pub use propeller_source::{
    PropellerSource, interaction_mode_directivity, interaction_mode_order,
    spinning_mode_directivity,
};
pub use replay::{
    Detrended, PeakTone, ZERO_TOLERANCE, detrend, directivity_factor, fwh_component_gap,
    grid_nonuniformity, lighthill_far_field_gap, lighthill_reciprocity_gap, mean_step,
    observer_at_kr, peak_tone, quadrupole_from_history, revolved_cross_check, ring_point_gap,
    spreading_error, within_band,
};
pub use snapshot::{
    area_cell_from_series, azimuthal_mode_coefficient, dft_peak_phasor, fwh_contour_from_series,
    fwh_patch_from_series, lighthill_stress_phasor, lighthill_stress_phasor_2d,
    revolved_cell_volume_per_rad, revolved_patch_from_modal_data, ring_source_from_modal_stress,
    spinning_phasor, volume_cell_from_series,
};
pub use sonium_wave::analytical::annular_duct_modes::{
    AnnularDuctMode, AxialBranch, DuctModeError, tyler_sofrin_orders,
};
pub use sonium_wave::analytical::azimuthal::AzimuthalMode;
pub use stabilisation::aeroacoustic_tau;
pub use wave_diagnostics::{centroid_1d, group_velocity_error, group_velocity_from_track};
