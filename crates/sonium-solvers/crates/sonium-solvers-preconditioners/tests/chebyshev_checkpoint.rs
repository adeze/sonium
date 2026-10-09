use num_complex::Complex64;
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use sonium_solvers_core::traits::Preconditioner;
use sonium_solvers_preconditioners::{
    CHEBYSHEV_CHECKPOINT_FORMAT_VERSION, ChebyshevConfig, ChebyshevPreconditioner,
};
use sonium_solvers_sparse::CsrMatrix;

fn matrix<T: sonium_solvers_core::ComplexField>() -> CsrMatrix<T> {
    CsrMatrix::from_raw_parts(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![
            T::from_re_im(T::Real::from_f64(4.0).unwrap(), T::Real::zero()),
            T::from_re_im(T::Real::from_f64(1.0).unwrap(), T::Real::zero()),
            T::from_re_im(T::Real::from_f64(1.0).unwrap(), T::Real::zero()),
            T::from_re_im(T::Real::from_f64(4.0).unwrap(), T::Real::zero()),
            T::from_re_im(T::Real::from_f64(1.0).unwrap(), T::Real::zero()),
            T::from_re_im(T::Real::from_f64(1.0).unwrap(), T::Real::zero()),
            T::from_re_im(T::Real::from_f64(4.0).unwrap(), T::Real::zero()),
            T::from_re_im(T::Real::from_f64(1.0).unwrap(), T::Real::zero()),
            T::from_re_im(T::Real::from_f64(1.0).unwrap(), T::Real::zero()),
            T::from_re_im(T::Real::from_f64(3.0).unwrap(), T::Real::zero()),
        ],
    )
}

fn assert_close<T: sonium_solvers_core::ComplexField>(left: &[T], right: &[T]) {
    assert_eq!(left.len(), right.len());
    for (actual, expected) in left.iter().zip(right) {
        assert!(((*actual - *expected).norm()).to_f64().unwrap() < 1.0e-12);
    }
}

#[test]
fn real_checkpoint_round_trip_preserves_action_and_parameters() {
    let source = ChebyshevPreconditioner::try_from_csr(
        &matrix::<f64>(),
        ChebyshevConfig {
            degree: 4,
            eigenvalue_ratio: 18.0,
            lambda_max: Some(5.5),
        },
    )
    .expect("Chebyshev setup");
    let rhs = [1.0, -2.0, 0.5, 3.0];
    let expected = source.try_apply(&rhs).expect("source action");
    let bytes = source.to_checkpoint_bytes().expect("serialize checkpoint");
    assert_eq!(
        u16::from_le_bytes([bytes[8], bytes[9]]),
        CHEBYSHEV_CHECKPOINT_FORMAT_VERSION
    );
    let restored =
        ChebyshevPreconditioner::<f64>::from_checkpoint_bytes(&bytes).expect("restore checkpoint");
    assert_eq!(restored.lambda_max(), source.lambda_max());
    assert_close(
        &restored.try_apply(&rhs).expect("restored action"),
        &expected,
    );
}

#[test]
fn complex_checkpoint_round_trip_preserves_exact_adjoint_actions() {
    let source = ChebyshevPreconditioner::try_from_csr(
        &matrix::<Complex64>(),
        ChebyshevConfig {
            degree: 3,
            eigenvalue_ratio: 24.0,
            lambda_max: Some(5.0),
        },
    )
    .expect("complex Chebyshev setup");
    let rhs = [
        Complex64::new(1.0, 0.5),
        Complex64::new(-2.0, 0.25),
        Complex64::new(0.5, -1.0),
        Complex64::new(3.0, 0.75),
    ];
    let bytes = source.to_checkpoint_bytes().expect("serialize checkpoint");
    let restored = ChebyshevPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes)
        .expect("restore checkpoint");
    assert_close(
        &restored.try_apply(&rhs).expect("restored forward"),
        &source.try_apply(&rhs).expect("source forward"),
    );
    assert_close(
        &restored
            .try_apply_transpose(&rhs)
            .expect("restored transpose"),
        &source.try_apply_transpose(&rhs).expect("source transpose"),
    );
    assert_close(
        &restored
            .try_apply_hermitian(&rhs)
            .expect("restored Hermitian"),
        &source.try_apply_hermitian(&rhs).expect("source Hermitian"),
    );
}

#[test]
fn checkpoint_rejects_corruption_truncation_and_scalar_mismatch() {
    let source = ChebyshevPreconditioner::try_from_csr(
        &matrix::<f64>(),
        ChebyshevConfig {
            lambda_max: Some(5.0),
            ..Default::default()
        },
    )
    .expect("Chebyshev setup");
    let bytes = source.to_checkpoint_bytes().expect("serialize checkpoint");
    assert!(
        ChebyshevPreconditioner::<f64>::from_checkpoint_bytes(&bytes[..bytes.len() - 1]).is_err()
    );
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(ChebyshevPreconditioner::<f64>::from_checkpoint_bytes(&trailing).is_err());
    let mut bad_magic = bytes.clone();
    bad_magic[0] ^= 0xff;
    assert!(ChebyshevPreconditioner::<f64>::from_checkpoint_bytes(&bad_magic).is_err());
    assert!(ChebyshevPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes).is_err());
    let mut bad_reserved = bytes;
    bad_reserved[13] = 1;
    assert!(ChebyshevPreconditioner::<f64>::from_checkpoint_bytes(&bad_reserved).is_err());
}
