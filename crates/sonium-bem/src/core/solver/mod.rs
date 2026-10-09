//! Linear solvers for BEM
//!
//! This module provides various solvers for BEM systems, re-exporting
//! functionality from `sonium-solvers`.
//!
//! - [`direct`] - Direct LU factorization
//! - `cgs` - Conjugate Gradient Squared
//! - `bicgstab` - BiCGSTAB
//! - `gmres` - GMRES (recommended for large BEM problems)
//! - [`fmm_interface`] - Interface to use FMM operators with iterative solvers
//! - `preconditioner` - Preconditioners (ILU, AMG, etc.)
//! - [`batched_blas`] - Batched BLAS operations (native only)
//! - [`frequency_sweep`] - Frequency sweep with solution recycling

pub mod batched_blas;
pub mod fmm_interface;
pub mod frequency_sweep;
pub mod transmission;

// Re-export core solver functionality from sonium-solvers
pub use sonium_solvers::direct;
// Modules are private in sonium-solvers, so we can't re-export them directly.
// We access their contents via the iterative module or specific re-exports.

pub use sonium_solvers::traits::{LinearOperator, Preconditioner};

// Core operator types from fmm_interface
pub use fmm_interface::{
    CsrOperator, DenseOperator, DenseOperatorError, DiagonalPreconditioner, MlfmmOperator,
};

pub use fmm_interface::{BemIrConfig, BemIrSolution, gmres_ir_solve_with_chebyshev_operator};

pub use fmm_interface::SlfmmOperator;

// Helper functions for solving BEM systems
pub use fmm_interface::{
    gmres_solve_tbem_with_ilu, gmres_solve_with_ilu, gmres_solve_with_ilu_operator, solve_bicgstab,
    solve_cgs, solve_gmres, solve_with_ilu, try_gmres_solve_fmm_batched,
    try_gmres_solve_fmm_batched_adjoint_with_hilu, try_gmres_solve_fmm_batched_adjoint_with_ilu,
    try_gmres_solve_fmm_batched_with_hilu, try_gmres_solve_fmm_batched_with_ilu,
    try_gmres_solve_mlfmm_batched_adjoint_with_hilu,
    try_gmres_solve_mlfmm_batched_adjoint_with_ilu, try_gmres_solve_mlfmm_batched_with_hilu,
    try_gmres_solve_mlfmm_batched_with_ilu, try_gmres_solve_with_ilu,
    try_gmres_solve_with_ilu_operator, try_solve_with_ilu, try_solve_with_ilu_operator,
};

// Batched BLAS solvers
pub use fmm_interface::{
    gmres_solve_fmm_batched, gmres_solve_fmm_batched_with_ilu,
    gmres_solve_mlfmm_batched_adjoint_with_ilu, gmres_solve_mlfmm_batched_with_ilu,
};

// Adaptive mesh utilities
pub use fmm_interface::{
    AdaptiveMeshConfig, estimate_element_count, mesh_resolution_for_frequency_range,
    recommended_mesh_resolution,
};

// Re-export specific configuration types for convenience
pub use sonium_solvers::iterative::{BiCgstabConfig, BiCgstabSolution};
pub use sonium_solvers::iterative::{CgsConfig, CgsSolution};
pub use sonium_solvers::iterative::{GmresConfig, GmresSolution};
pub use sonium_solvers::preconditioners::IluPreconditioner;
pub use sonium_solvers::preconditioners::{AmgConfig, AmgPreconditioner};

// Batched BLAS operations
pub use batched_blas::{
    SlfmmMatvecWorkspace, batched_d_matrix_apply, batched_near_field_apply, batched_s_matrix_apply,
    batched_t_matrix_apply, create_batched_matvec, slfmm_matvec_batched,
};

// Frequency sweep with solution recycling
pub use frequency_sweep::{
    FrequencySweepConfig, FrequencySweepPoint, FrequencySweepResult, dispatch_sequential_sweep,
    frequency_sweep, frequency_sweep_with_physics, frequency_sweep_with_physics_and_admittances,
    validate_frequency_grid,
};

// Checked CPU solve for the assembled Müller transmission system.
pub use transmission::{
    TransmissionAdjointSolution, TransmissionSolution, TransmissionSolveError,
    solve_incident_transmission, solve_transmission_adjoint_many,
    solve_transmission_adjoint_system, solve_transmission_system,
};
