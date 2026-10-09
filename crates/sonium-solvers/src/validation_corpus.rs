//! Small deterministic matrix/PDE fixtures used by solver property tests.
//!
//! The corpus is intentionally checked in rather than generated from random
//! seeds.  Each case records the model class and admission assumptions so a
//! solver failure can be classified as a structural, numerical, or
//! applicability failure.  Larger Matrix Market and external differential
//! runs remain separate review artifacts.

use sonium_solvers_core::traits::SolverError;
use sonium_solvers_sparse::CsrMatrix;

/// Matrix class represented by a deterministic corpus fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CorpusClass {
    Spd,
    Hpd,
    SymmetricIndefinite,
    Nonsymmetric,
    SaddlePoint,
    IllScaled,
    NearSingular,
    RankDeficient,
    Helmholtz,
    ConvectionDiffusion,
}

/// One fixed corpus admission record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorpusCase {
    pub id: &'static str,
    pub class: CorpusClass,
    pub dimension: usize,
    pub nonzeros: usize,
    pub scalar: &'static str,
    pub assumptions: &'static str,
    pub workload: &'static str,
    pub conditioning: &'static str,
    pub reference_tolerance: &'static str,
    pub adversarial: bool,
}

/// Versioned deterministic corpus metadata.
pub static SOLVER_VALIDATION_CORPUS: &[CorpusCase] = &[
    CorpusCase {
        id: "spd-laplacian-3x3.v1",
        class: CorpusClass::Spd,
        dimension: 3,
        nonzeros: 7,
        scalar: "f64",
        assumptions: "symmetric positive definite",
        workload: "3x3 analytic stencil",
        conditioning: "well-scaled",
        reference_tolerance: "1e-12",
        adversarial: false,
    },
    CorpusCase {
        id: "hpd-complex-2x2.v1",
        class: CorpusClass::Hpd,
        dimension: 2,
        nonzeros: 4,
        scalar: "Complex64",
        assumptions: "Hermitian positive definite",
        workload: "2x2 analytic complex fixture",
        conditioning: "well-scaled",
        reference_tolerance: "1e-12",
        adversarial: false,
    },
    CorpusCase {
        id: "symmetric-indefinite-pivot-2x2.v1",
        class: CorpusClass::SymmetricIndefinite,
        dimension: 2,
        nonzeros: 2,
        scalar: "f64",
        assumptions: "symmetric indefinite with a 2x2 pivot",
        workload: "2x2 pivot admission fixture",
        conditioning: "indefinite zero diagonal",
        reference_tolerance: "1e-12",
        adversarial: true,
    },
    CorpusCase {
        id: "nonsymmetric-dense-3x3.v1",
        class: CorpusClass::Nonsymmetric,
        dimension: 3,
        nonzeros: 9,
        scalar: "f64/Complex64",
        assumptions: "general square matrix with no symmetry claim",
        workload: "3x3 dense analytic fixture",
        conditioning: "well-scaled",
        reference_tolerance: "1e-12",
        adversarial: false,
    },
    CorpusCase {
        id: "saddle-point-3x3.v1",
        class: CorpusClass::SaddlePoint,
        dimension: 3,
        nonzeros: 6,
        scalar: "f64",
        assumptions: "one primal block and one constraint block",
        workload: "3x3 KKT block fixture",
        conditioning: "constraint saddle point",
        reference_tolerance: "1e-10",
        adversarial: true,
    },
    CorpusCase {
        id: "ill-scaled-diagonal-3x3.v1",
        class: CorpusClass::IllScaled,
        dimension: 3,
        nonzeros: 3,
        scalar: "f64",
        assumptions: "finite diagonal with six orders of magnitude spread",
        workload: "3x3 diagonal scaling fixture",
        conditioning: "1e12 diagonal spread",
        reference_tolerance: "1e-10",
        adversarial: true,
    },
    CorpusCase {
        id: "near-singular-2x2.v1",
        class: CorpusClass::NearSingular,
        dimension: 2,
        nonzeros: 4,
        scalar: "f64",
        assumptions: "finite matrix with a small determinant",
        workload: "2x2 near-rank-deficient fixture",
        conditioning: "determinant approximately 1e-8",
        reference_tolerance: "1e-8",
        adversarial: true,
    },
    CorpusCase {
        id: "rank-deficient-2x2.v1",
        class: CorpusClass::RankDeficient,
        dimension: 2,
        nonzeros: 4,
        scalar: "f64",
        assumptions: "finite exactly rank-deficient square matrix",
        workload: "2x2 duplicate-row failure fixture",
        conditioning: "zero determinant; singular system",
        reference_tolerance: "typed rejection",
        adversarial: true,
    },
    CorpusCase {
        id: "helmholtz-1d-3x3.v1",
        class: CorpusClass::Helmholtz,
        dimension: 3,
        nonzeros: 7,
        scalar: "Complex64",
        assumptions: "complex shifted 1-D Helmholtz stencil",
        workload: "3x3 complex Helmholtz stencil",
        conditioning: "complex shift 0.2i",
        reference_tolerance: "1e-10",
        adversarial: true,
    },
    CorpusCase {
        id: "convection-diffusion-1d-3x3.v1",
        class: CorpusClass::ConvectionDiffusion,
        dimension: 3,
        nonzeros: 7,
        scalar: "f64",
        assumptions: "nonsymmetric upwind diffusion stencil",
        workload: "3x3 upwind stencil",
        conditioning: "moderate convection skew",
        reference_tolerance: "1e-10",
        adversarial: true,
    },
    CorpusCase {
        id: "poisson-1d-16x16.v1",
        class: CorpusClass::Spd,
        dimension: 16,
        nonzeros: 46,
        scalar: "f64",
        assumptions: "symmetric positive definite Dirichlet Poisson stencil",
        workload: "generated fixed 1-D Poisson benchmark",
        conditioning: "O(n^2) Laplacian conditioning; deterministic n=16",
        reference_tolerance: "1e-10",
        adversarial: false,
    },
    CorpusCase {
        id: "convection-diffusion-1d-16x16.v1",
        class: CorpusClass::ConvectionDiffusion,
        dimension: 16,
        nonzeros: 46,
        scalar: "f64",
        assumptions: "nonsymmetric upwind convection-diffusion stencil",
        workload: "generated fixed 1-D convection-diffusion benchmark",
        conditioning: "moderate upwind skew; deterministic n=16",
        reference_tolerance: "1e-10",
        adversarial: true,
    },
];

/// Build the checked-in one-dimensional Dirichlet Poisson fixture.
///
/// The matrix is the standard tridiagonal `[-1, 2, -1]` stencil.  Keeping
/// construction here (alongside the metadata) prevents validation callers
/// from silently drifting to a different stencil while retaining a checked
/// zero-size admission boundary.
pub fn poisson_1d_fixture(n: usize) -> Result<CsrMatrix<f64>, SolverError> {
    build_tridiagonal_fixture(n, 2.0, -1.0, -1.0)
}

/// Build the checked-in one-dimensional upwind convection-diffusion fixture.
///
/// The asymmetric lower/upper couplings match the `v1` corpus case and are
/// deliberately kept fixed so independent solver/reference runs share one
/// deterministic operator.
pub fn convection_diffusion_1d_fixture(n: usize) -> Result<CsrMatrix<f64>, SolverError> {
    build_tridiagonal_fixture(n, 2.0, -0.25, -1.0)
}

fn build_tridiagonal_fixture(
    n: usize,
    diagonal: f64,
    lower: f64,
    upper: f64,
) -> Result<CsrMatrix<f64>, SolverError> {
    if n == 0 {
        return Err(SolverError::OperationFailed {
            operation: "validation corpus fixture",
            reason: "dimension must be non-zero".into(),
        });
    }
    let mut triplets = Vec::with_capacity(n.saturating_mul(3).saturating_sub(2));
    for row in 0..n {
        triplets.push((row, row, diagonal));
        if row > 0 {
            triplets.push((row, row - 1, lower));
        }
        if row + 1 < n {
            triplets.push((row, row + 1, upper));
        }
    }
    CsrMatrix::try_from_triplets(n, n, triplets)
}

/// Return the checked-in corpus in stable order.
pub const fn all() -> &'static [CorpusCase] {
    SOLVER_VALIDATION_CORPUS
}

#[cfg(test)]
mod tests {
    use super::{CorpusClass, SOLVER_VALIDATION_CORPUS};
    use std::collections::HashSet;

    #[test]
    fn corpus_is_unique_and_covers_required_classes() {
        let mut ids = HashSet::new();
        assert!(SOLVER_VALIDATION_CORPUS.len() >= 12);
        for case in SOLVER_VALIDATION_CORPUS {
            assert!(ids.insert(case.id));
            assert!(case.dimension > 0);
            assert!(case.nonzeros >= case.dimension);
            assert!(!case.assumptions.is_empty());
            assert!(!case.workload.is_empty());
            assert!(!case.conditioning.is_empty());
            assert!(!case.reference_tolerance.is_empty());
        }
        for class in [
            CorpusClass::Spd,
            CorpusClass::Hpd,
            CorpusClass::SymmetricIndefinite,
            CorpusClass::Nonsymmetric,
            CorpusClass::SaddlePoint,
            CorpusClass::IllScaled,
            CorpusClass::NearSingular,
            CorpusClass::RankDeficient,
            CorpusClass::Helmholtz,
            CorpusClass::ConvectionDiffusion,
        ] {
            assert!(
                SOLVER_VALIDATION_CORPUS
                    .iter()
                    .any(|case| case.class == class)
            );
        }
    }
}
