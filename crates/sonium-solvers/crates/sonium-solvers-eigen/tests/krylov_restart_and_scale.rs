//! Regression tests for the relative Krylov breakdown policy and the
//! deterministic-restart machinery (review items: absolute `1e-28` / `1e-13`
//! / `1e-14` cutoffs, single-vector lucky breakdown).
//!
//! - Scale invariance: uniformly scaling a pencil must not move (or lose)
//!   eigenpairs. An absolute breakdown cutoff is blind to operator scale;
//!   the relative policy keeps the spectrum put at 1e±6.
//! - Analytic cross-check on a 2x2 generalized pencil.
//! - Bit reproducibility across repeated runs (stream-0 preservation).
//! - Config validation of the new `breakdown_tolerance` field.

use num_complex::Complex64;
use sonium_solvers_core::traits::SolverError;
use sonium_solvers_eigen::{
    ComplexArnoldiConfig, ShiftInvertConfig, complex_shift_invert_arnoldi_generalized,
    shift_invert_lanczos_generalized,
};
use sonium_solvers_sparse::CsrMatrix;

fn real_diag(values: &[f64]) -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(
        values.len(),
        values.len(),
        values
            .iter()
            .enumerate()
            .map(|(index, value)| (index, index, *value))
            .collect(),
    )
}

fn real_identity(dimension: usize) -> CsrMatrix<f64> {
    real_diag(&vec![1.0; dimension])
}

fn complex_diag(values: &[Complex64]) -> CsrMatrix<Complex64> {
    CsrMatrix::from_triplets(
        values.len(),
        values.len(),
        values
            .iter()
            .enumerate()
            .map(|(index, value)| (index, index, *value))
            .collect(),
    )
}

fn complex_identity(dimension: usize) -> CsrMatrix<Complex64> {
    complex_diag(&vec![Complex64::new(1.0, 0.0); dimension])
}

fn real_config(target: f64, num: usize, krylov: usize) -> ShiftInvertConfig {
    ShiftInvertConfig {
        target,
        num_eigenvalues: num,
        krylov_dimension: krylov,
        tolerance: 1.0e-10,
        breakdown_tolerance: 1.0e-14,
    }
}

fn complex_config(target: Complex64, num: usize, krylov: usize) -> ComplexArnoldiConfig {
    ComplexArnoldiConfig {
        target,
        num_eigenvalues: num,
        krylov_dimension: krylov,
        tolerance: 1.0e-8,
        max_qr_iterations: 4_000,
        breakdown_tolerance: 1.0e-14,
    }
}

fn sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(|left, right| left.total_cmp(right));
    values
}

#[test]
fn shift_invert_matches_analytic_2x2_pencil() {
    // A = diag(1, 2), B = diag(2, 1): generalized eigenvalues 1/2 and 2.
    let a = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.0), (1, 1, 2.0)]);
    let b = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 2.0), (1, 1, 1.0)]);
    let result =
        shift_invert_lanczos_generalized(&a, &b, &real_config(1.0, 2, 2)).expect("2x2 must solve");
    assert!(result.converged);
    // Analytic comparison with the file's 1e-9 convention (exact float
    // equality is reserved for same-build bit-reproducibility below):
    // the merged BLAS/LAPACK kernels reassociate in the last ulp
    // (observed 2.2e-16), which exact equality cannot survive.
    let got = sorted(result.eigenvalues);
    for (value, expected) in got.iter().zip([0.5, 2.0]) {
        assert!(
            (value - expected).abs() < 1.0e-9,
            "{got:?} drifted from [0.5, 2.0]"
        );
    }
}

#[test]
fn shift_invert_spectrum_is_scale_invariant() {
    // Uniformly scaling (A, B) by `scale` leaves every generalized
    // eigenvalue fixed (the factor cancels), while the operator entries span
    // 1e-6..1e6. The target stays at 3.05: the relative breakdown policy
    // must recover the same {2, 3, 4} spectrum at every entry scale, where
    // an absolute cutoff would misfire.
    let spectrum = [1.0, 2.0, 3.0, 4.0, 5.0];
    for scale in [1.0e-6, 1.0, 1.0e6] {
        let scaled: Vec<f64> = spectrum.iter().map(|value| value * scale).collect();
        let a = real_diag(&scaled);
        let b = real_diag(&[scale; 5]);
        let result = shift_invert_lanczos_generalized(&a, &b, &real_config(3.05, 3, 12))
            .expect("scaled pencil must solve");
        assert!(result.converged, "scale {scale}: {result:?}");
        let got = sorted(result.eigenvalues);
        for (value, expected) in got.iter().zip([2.0, 3.0, 4.0]) {
            assert!(
                (value - expected).abs() < 1.0e-9,
                "scale {scale}: {got:?} drifted from [2, 3, 4]"
            );
        }
    }
}

#[test]
fn shift_invert_repeated_runs_are_bit_reproducible() {
    let a = real_diag(&[1.0, 2.0, 3.0, 4.0]);
    let first =
        shift_invert_lanczos_generalized(&a, &real_identity(4), &real_config(2.5, 2, 8)).unwrap();
    let second =
        shift_invert_lanczos_generalized(&a, &real_identity(4), &real_config(2.5, 2, 8)).unwrap();
    assert_eq!(first.eigenvalues, second.eigenvalues);
}

#[test]
fn shift_invert_rejects_invalid_breakdown_tolerance() {
    let a = real_diag(&[1.0, 2.0]);
    for breakdown_tolerance in [0.0, -1.0e-14, f64::NAN, f64::INFINITY] {
        let config = ShiftInvertConfig {
            breakdown_tolerance,
            ..real_config(1.5, 1, 2)
        };
        let error = shift_invert_lanczos_generalized(&a, &real_identity(2), &config)
            .expect_err("non-positive/non-finite breakdown tolerance must be rejected");
        assert!(
            matches!(error, SolverError::OperationFailed { .. }),
            "unexpected error for {breakdown_tolerance:?}: {error:?}"
        );
    }
}

#[test]
fn complex_arnoldi_spectrum_is_scale_invariant() {
    let spectrum = [
        Complex64::new(1.0, 0.5),
        Complex64::new(2.0, -1.0),
        Complex64::new(3.0, 0.25),
    ];
    let solve = |scale: f64| {
        let scaled: Vec<Complex64> = spectrum.iter().map(|value| value * scale).collect();
        complex_shift_invert_arnoldi_generalized(
            &complex_diag(&scaled),
            &complex_identity(3),
            &complex_config(Complex64::new(2.05, -0.9) * scale, 2, 10),
        )
        .expect("scaled complex pencil must solve")
        .eigenvalues
    };
    let unit = solve(1.0);
    let large = solve(1.0e3);
    assert_eq!(unit.len(), 2);
    assert_eq!(large.len(), 2);
    for (got, expected) in large.iter().zip(unit.iter()) {
        assert!(
            (got / 1.0e3 - expected).norm() < 1.0e-9,
            "scale drift: {got:?} vs {expected:?}"
        );
    }
}

#[test]
fn complex_arnoldi_rejects_invalid_breakdown_tolerance() {
    let a = complex_diag(&[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]);
    let config = ComplexArnoldiConfig {
        breakdown_tolerance: 0.0,
        ..complex_config(Complex64::new(1.5, 0.0), 1, 2)
    };
    let error = complex_shift_invert_arnoldi_generalized(&a, &complex_identity(2), &config)
        .expect_err("zero breakdown tolerance must be rejected");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}
