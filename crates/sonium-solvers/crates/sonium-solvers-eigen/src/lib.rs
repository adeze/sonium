//! Eigenvalue solvers for sparse matrices
//!
//! - [`lobpcg`]: Locally Optimal Block Preconditioned Conjugate Gradient
//!   for computing a few smallest eigenvalues of large sparse SPD matrices.
//! - [`lobpcg_generalized`]: LOBPCG for generalized SPD pencils `A x = λ B x`.
//!
//! Scope note: LOBPCG entry points are `f64`-only (real symmetric
//! pencils). Complex-shift problems use [`complex_shift_invert_arnoldi_generalized`];
//! eigenpair sensitivity/continuation in [`adjoint`] is generic over
//! [`sonium_solvers_core::traits::ComplexField`].

mod adjoint;
mod checkpoint;
mod complex_arnoldi;
#[cfg(feature = "cuda")]
pub mod cuda;
mod dense_nonsymmetric;
mod divide_conquer;
pub(crate) mod laed;
mod lobpcg;
mod quadratic;
mod result_contract;
mod shift_invert;
mod subspace;
mod x86;

pub use adjoint::{
    EigenContinuationConfig, EigenContinuationEvent, EigenContinuationPoint,
    EigenContinuationPrediction, EigenContinuationStep, EigenPseudoArclengthPrediction,
    EigenPseudoArclengthStep, classify_eigen_continuation_event, continue_generalized_eigenpair,
    continue_generalized_eigenpair_pseudo_arclength, generalized_eigenvalue_sensitivity,
    generalized_eigenvector_sensitivity, predict_generalized_eigenpair,
};
pub use checkpoint::{EIGEN_CHECKPOINT_FORMAT_VERSION, EIGEN_CHECKPOINT_MAGIC};
pub use complex_arnoldi::{
    ComplexArnoldiConfig, ComplexTargetedEigenResult, complex_shift_invert_arnoldi_generalized,
};
pub use dense_nonsymmetric::{nonsymmetric_eigenvalues, nonsymmetric_eigenvalues_f32};

pub use lobpcg::{
    LobpcgConfig, LobpcgResult, lobpcg, lobpcg_generalized, lobpcg_generalized_preconditioned,
    lobpcg_preconditioned,
};
pub use quadratic::{QuadraticEigenConfig, QuadraticEigenResult, quadratic_eigen_damped};
pub use shift_invert::{
    FeastConfig, ShiftInvertConfig, TargetedEigenResult, eigenpairs_in_interval,
    eigenvalue_count_in_interval, feast_generalized, shift_invert_lanczos_generalized,
};
pub use subspace::{orthonormal_basis, subspace_distance};

// Dense symmetric eigensolver (Householder + implicit-shift QL), shared with
// the CMA-ES optimiser for its per-generation covariance decomposition.
pub use lobpcg::symmetric_eigen;
