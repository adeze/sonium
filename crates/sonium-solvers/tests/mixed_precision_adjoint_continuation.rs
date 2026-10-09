use num_complex::{Complex32, Complex64};
use sonium_solvers::mixed_precision::MixedPrecisionPreconditioner;
use sonium_solvers::{
    AdjointCapability, AdjointSolve, CsrMatrix, DiagonalPreconditioner, DifferentiableLinearSolve,
    GmresConfig, LinearOperator, Preconditioner, SolverError,
};

fn operator() -> CsrMatrix<Complex64> {
    CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, Complex64::new(4.0, 0.0)),
            (0, 1, Complex64::new(1.0, 0.5)),
            (1, 0, Complex64::new(1.0, -0.5)),
            (1, 1, Complex64::new(3.0, 0.0)),
            (1, 2, Complex64::new(0.75, 0.25)),
            (2, 1, Complex64::new(0.75, -0.25)),
            (2, 2, Complex64::new(2.5, 0.0)),
        ],
    )
}

fn config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 40,
        restart: 3,
        tolerance: 1.0e-12,
        ..GmresConfig::default()
    }
}

fn preconditioner() -> MixedPrecisionPreconditioner<Complex64, Complex32> {
    MixedPrecisionPreconditioner::new(Box::new(DiagonalPreconditioner::from_diagonal(&[
        Complex32::new(4.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(2.5, 0.0),
    ])))
}

#[test]
fn mixed_precision_transpose_and_hermitian_continuations_accept_exact_iterates() {
    let operator = operator();
    let preconditioner = preconditioner();
    assert_eq!(operator.transpose_capability(), AdjointCapability::Exact);
    assert_eq!(operator.hermitian_capability(), AdjointCapability::Exact);
    assert_eq!(
        preconditioner.transpose_capability(),
        AdjointCapability::Exact
    );
    assert_eq!(
        preconditioner.hermitian_capability(),
        AdjointCapability::Exact
    );
    let handle = DifferentiableLinearSolve::new(&operator, &preconditioner, config()).unwrap();
    let rhs = [
        Complex64::new(1.0, 2.0),
        Complex64::new(-2.0, 0.5),
        Complex64::new(0.5, -1.0),
    ];

    let transpose = handle.solve_adjoint(&rhs, AdjointSolve::Transpose).unwrap();
    let resumed_transpose = handle
        .solve_adjoint_with_guess(&rhs, AdjointSolve::Transpose, Some(&transpose.x))
        .unwrap();
    let hermitian = handle.solve_adjoint(&rhs, AdjointSolve::Hermitian).unwrap();
    let resumed_hermitian = handle
        .solve_adjoint_with_guess(&rhs, AdjointSolve::Hermitian, Some(&hermitian.x))
        .unwrap();

    for result in [
        &transpose,
        &hermitian,
        &resumed_transpose,
        &resumed_hermitian,
    ] {
        assert!(result.converged);
        assert!(result.residual < 1.0e-12);
    }
    assert_eq!(resumed_transpose.iterations, 0);
    assert_eq!(resumed_hermitian.iterations, 0);
    assert!(
        transpose
            .x
            .iter()
            .zip(&resumed_transpose.x)
            .all(|(left, right)| (*left - *right).norm() < 1.0e-12)
    );
    assert!(
        hermitian
            .x
            .iter()
            .zip(&resumed_hermitian.x)
            .all(|(left, right)| (*left - *right).norm() < 1.0e-12)
    );
    assert!(
        handle
            .solve_adjoint_with_guess(
                &rhs,
                AdjointSolve::Hermitian,
                Some(&[Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]),
            )
            .is_err()
    );
}

#[test]
fn mixed_precision_adjoint_continuation_rejects_non_finite_guess() {
    let operator = operator();
    let preconditioner = preconditioner();
    let handle = DifferentiableLinearSolve::new(&operator, &preconditioner, config()).unwrap();
    let error = handle
        .solve_adjoint_with_guess(
            &[Complex64::new(1.0, 0.0); 3],
            AdjointSolve::Hermitian,
            Some(&[
                Complex64::new(0.0, 0.0),
                Complex64::new(f64::NAN, 0.0),
                Complex64::new(0.0, 0.0),
            ]),
        )
        .unwrap_err();
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

fn coupled_hilu_f32() -> sonium_solvers::HiluPreconditioner<Complex32> {
    let n = 20;
    let mut row_ptrs = vec![0usize];
    let mut columns = Vec::new();
    let mut values = Vec::new();
    for row in 0..n {
        if row > 0 {
            columns.push(row - 1);
            values.push(Complex32::new(-1.0, 0.3));
        }
        columns.push(row);
        values.push(Complex32::new(6.0 + 0.1 * (row % 3) as f32, 0.2));
        if row + 1 < n {
            columns.push(row + 1);
            values.push(Complex32::new(-0.7, -0.2));
        }
        row_ptrs.push(columns.len());
    }
    let matrix = CsrMatrix::try_from_raw_parts(n, n, row_ptrs, columns, values)
        .expect("coupled HILU f32 fixture must be canonical");
    sonium_solvers::HiluPreconditioner::try_from_csr_coupled_with_config(
        &matrix,
        &sonium_solvers::HiluConfig {
            min_block_size: 4,
            max_levels: None,
        },
        0.0,
    )
    .expect("coupled HILU f32 fixture should build")
}

fn seeded_complex64_vector(n: usize, mut state: u64) -> Vec<Complex64> {
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let real = ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0;
            state = state.rotate_left(17);
            let imag = ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0;
            Complex64::new(real, imag)
        })
        .collect()
}

fn bilinear_inner(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    a.iter().zip(b).map(|(left, right)| *left * *right).sum()
}

fn hermitian_inner(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    a.iter()
        .zip(b)
        .map(|(left, right)| left.conj() * *right)
        .sum()
}

#[test]
fn mixed_precision_coupled_hilu_preserves_adjoint_identities() {
    let low = coupled_hilu_f32();
    let mixed = MixedPrecisionPreconditioner::<Complex64, Complex32>::new(Box::new(low));
    assert_eq!(mixed.transpose_capability(), AdjointCapability::Exact);
    assert_eq!(mixed.hermitian_capability(), AdjointCapability::Exact);

    let mut max_transpose_error = 0.0_f64;
    let mut max_hermitian_error = 0.0_f64;
    for seed in 0..8_u64 {
        let r = seeded_complex64_vector(20, 0xA5A5_0000 + seed);
        let s = seeded_complex64_vector(20, 0x5A5A_0000 + seed);
        let mr = mixed.apply(&r);
        let mt_s = mixed
            .try_apply_transpose(&s)
            .expect("mixed coupled HILU transpose should be supported");
        let mh_s = mixed
            .try_apply_hermitian(&s)
            .expect("mixed coupled HILU Hermitian should be supported");
        max_transpose_error =
            max_transpose_error.max((bilinear_inner(&mr, &s) - bilinear_inner(&r, &mt_s)).norm());
        max_hermitian_error =
            max_hermitian_error.max((hermitian_inner(&mr, &s) - hermitian_inner(&r, &mh_s)).norm());
    }
    assert!(
        max_transpose_error < 3.0e-5,
        "mixed coupled HILU transpose identity error {max_transpose_error:e}"
    );
    assert!(
        max_hermitian_error < 3.0e-5,
        "mixed coupled HILU Hermitian identity error {max_hermitian_error:e}"
    );
}
