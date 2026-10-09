//! BEM matrix assembly
//!
//! This module provides four methods for assembling BEM systems:
//!
//! - [`tbem`] - Traditional BEM with O(N²) dense matrix (always available)
//! - [`slfmm`] - Single-Level Fast Multipole Method (requires `native` or `wasm` feature)
//! - [`mlfmm`] - Multi-Level Fast Multipole Method
//! - [`hmatrix`] - H-matrix with Adaptive Cross Approximation (ACA) for far-field compression
//!
//! For small problems (N < 1000), TBEM is usually fastest.
//! For larger problems, FMM methods provide O(N log N) or O(N) scaling.
//! H-matrices with ACA provide kernel-independent compression and are particularly
//! effective for frequency sweeps when combined with Krylov recycling.
//!
//! ## WASM Compatibility
//!
//! With the `wasm` feature, all assembly methods are available including SLFMM.
//! Parallel processing is provided via wasm-bindgen-rayon (Web Workers).

pub mod common;
pub mod galerkin;
pub mod hmatrix;
// Allow needless_range_loop for complex FMM algorithm code ported from NumCalc
#[allow(clippy::needless_range_loop)]
pub mod mlfmm;
pub mod operators;
pub mod slfmm;
pub mod sparse;
pub mod surface_mass;
pub mod tbem;
pub mod transmission;

pub use common::{FmmAccuracyError, FmmAction};

pub use galerkin::{
    GalerkinAssemblyError, GalerkinDoubleLayerOperators, GalerkinHypersingularOperators,
    GalerkinQuad4TransmissionError, GalerkinQuadratureAdmissionError, GalerkinQuadraturePolicy,
    GalerkinTransmissionAssemblyError, GalerkinTransmissionCompressionError,
    GalerkinTransmissionFieldError, GalerkinTransmissionMaterialDerivativeError,
    GalerkinTransmissionOperators, GalerkinTransmissionProjectionError,
    GalerkinTransmissionRhsError, GalerkinTransmissionSolveError,
    assemble_galerkin_double_layer_with_basis, assemble_galerkin_double_layer_with_global_dofs,
    assemble_galerkin_hypersingular_with_global_dofs,
    assemble_galerkin_muller_transmission_material_derivative_with_global_dofs,
    assemble_galerkin_muller_transmission_quad4_pair,
    assemble_galerkin_muller_transmission_quad4_pair_material_derivative,
    assemble_galerkin_muller_transmission_quad4_pair_wavenumber_derivative,
    assemble_galerkin_muller_transmission_quad4_wavenumber_derivative_with_global_dofs,
    assemble_galerkin_muller_transmission_with_global_dofs,
    assemble_galerkin_muller_transmission_with_global_dofs_checked, assemble_galerkin_single_layer,
    assemble_galerkin_single_layer_with_basis, assemble_galerkin_single_layer_with_global_dofs,
    compress_galerkin_muller_transmission_material_derivative_with_global_dofs_checked,
    compress_galerkin_muller_transmission_quad4_material_derivative_with_global_dofs,
    compress_galerkin_muller_transmission_quad4_material_derivative_with_global_dofs_checked,
    compress_galerkin_muller_transmission_quad4_wavenumber_derivative_with_global_dofs,
    compress_galerkin_muller_transmission_quad4_with_global_dofs,
    compress_galerkin_muller_transmission_quad4_with_global_dofs_checked,
    compress_galerkin_muller_transmission_with_global_dofs,
    compress_galerkin_muller_transmission_with_global_dofs_checked,
};
pub use operators::{
    BoundaryOperatorAssemblyError, BoundaryOperators, assemble_boundary_operators,
    assemble_boundary_operators_wavenumber_derivative, assemble_boundary_operators_with_basis,
    assemble_boundary_operators_with_basis_shared_dofs, assemble_hypersingular,
    assemble_single_layer,
};
pub use surface_mass::{
    DualMassPreconditioner, assemble_p1_surface_mass, lift_primal_p0_to_refined_p1,
    restrict_refined_p1_to_primal_p0,
};

pub use hmatrix::{
    AcaVariant, DenseBlock, HMatrix, HMatrixAccuracyError, HMatrixAction, HMatrixOperatorError,
    LowRankBlock, LowRankPrecision, MatrixDerivativeError,
    apply_tbem_stabilized_exact_wavenumber_derivative, build_hmatrix,
    build_hmatrix_density_derivative, build_hmatrix_stabilized_density_derivative,
    build_hmatrix_stabilized_wavenumber_derivative, build_hmatrix_wavenumber_derivative,
    build_hmatrix_with_variant_precision_and_beta, compute_matrix_entry_wavenumber_derivative,
    compute_regular_matrix_entry_wavenumber_derivative,
};
pub use mlfmm::{
    DirectionalMlfmmConfig, MlfmmAssemblyError, MlfmmNearFieldError, MlfmmRowActionError,
    MlfmmSystem, MlfmmTranslationStats, MlfmmWavenumberDerivative, NearFieldBlock,
    build_cluster_tree, build_directional_mlfmm_system_stabilized,
    build_directional_mlfmm_system_stabilized_checked,
    build_directional_mlfmm_system_stabilized_with_basis_checked,
    build_directional_mlfmm_system_stabilized_with_config,
    build_directional_mlfmm_wavenumber_derivative,
    build_directional_mlfmm_wavenumber_derivative_with_basis_checked,
    build_mlfmm_system_stabilized, build_mlfmm_system_stabilized_checked,
    build_mlfmm_system_stabilized_with_basis_checked, build_mlfmm_wavenumber_derivative,
    build_mlfmm_wavenumber_derivative_with_basis_checked, validate_mlfmm_constant_panel_layout,
};
pub use slfmm::{
    SlfmmAssemblyError, SlfmmNearFieldError, SlfmmRowActionError, SlfmmSystem,
    SlfmmWavenumberDerivative, build_slfmm_system_stabilized,
    build_slfmm_system_stabilized_checked, build_slfmm_system_stabilized_with_basis_checked,
    build_slfmm_wavenumber_derivative, build_slfmm_wavenumber_derivative_with_basis_checked,
    validate_slfmm_constant_panel_layout,
};
#[cfg(feature = "cuda")]
pub use sonium_compute::CudaMlfmmC64;
pub use sparse::BlockedCsr;
#[expect(
    deprecated,
    reason = "compatibility re-export of apply_row_sum_correction; new code uses build_tbem_system_corrected"
)]
pub use tbem::{
    HMatrixVelocityOperator, MergeSizeError, MissingSourceDof, StabilizedTbemVelocityOperator,
    TbemDerivativeError, TbemQuadraturePairError, TbemQuadraturePolicy, TbemSystem,
    TbemVelocityOperatorError, apply_row_sum_correction, build_tbem_system,
    build_tbem_system_checked, build_tbem_system_continuous_quadratic_stabilized,
    build_tbem_system_corrected, build_tbem_system_linear, build_tbem_system_linear_stabilized,
    build_tbem_system_quadratic_stabilized, build_tbem_system_scaled, build_tbem_system_stabilized,
    build_tbem_system_stabilized_density_derivative,
    build_tbem_system_stabilized_impedance_density_derivative,
    build_tbem_system_stabilized_impedance_speed_derivative,
    build_tbem_system_stabilized_speed_derivative,
    build_tbem_system_stabilized_wavenumber_derivative,
    build_tbem_system_stabilized_with_beta_simd, build_tbem_system_with_beta,
    tbem_wavenumber_objective_gradient, try_build_tbem_system_with_beta, validate_tbem_quadrature,
    validate_tbem_quadrature_for_basis,
};

pub use tbem::build_tbem_system_parallel;
#[cfg(feature = "metal")]
pub use tbem::build_tbem_system_with_beta_metal;
#[cfg(feature = "cuda")]
pub use tbem::{
    CudaTbemAssemblyError, CudaTbemAssemblyWorkspace, build_quad4_cuda_regular_quadrature,
    build_tbem_system_with_beta_cuda, build_tbem_system_with_beta_cuda_checked,
    build_tri3_cuda_singular_quadrature,
};
pub use transmission::{
    CompressedTransmissionSolution, CompressedTransmissionTbemSystem, TransmissionAssemblyError,
    TransmissionCompressionError, TransmissionMaterialDerivativeError,
    TransmissionMaterialParameter, TransmissionRhsError, TransmissionSide, TransmissionTbemSystem,
    build_transmission_tbem_system, build_transmission_tbem_system_from_xem,
    build_transmission_tbem_system_with_basis, build_transmission_tbem_system_with_basis_checked,
    build_transmission_tbem_system_with_basis_checked_from_xem,
    build_transmission_tbem_system_with_continuous_quadratic,
    build_transmission_tbem_system_with_continuous_quadratic_checked,
    build_transmission_tbem_system_with_continuous_quadratic_checked_from_xem,
    build_transmission_tbem_system_with_continuous_quadratic_from_xem,
};
