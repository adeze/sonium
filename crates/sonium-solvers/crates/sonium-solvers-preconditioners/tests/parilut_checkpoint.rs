use num_complex::Complex64;
use num_traits::FromPrimitive;
use sonium_solvers_core::traits::Preconditioner;
use sonium_solvers_preconditioners::{ParIlutConfig, ParIlutPreconditioner};
use sonium_solvers_sparse::CsrMatrix;

fn matrix<T>() -> CsrMatrix<T>
where
    T: sonium_solvers_core::traits::ComplexField,
{
    CsrMatrix::from_raw_parts(
        4,
        4,
        vec![0, 3, 6, 9, 11],
        vec![0, 1, 2, 0, 1, 3, 0, 1, 2, 1, 3],
        vec![
            T::from_real(T::Real::from_f64(4.0).unwrap()),
            T::from_real(T::Real::from_f64(-1.0).unwrap()),
            T::from_real(T::Real::from_f64(0.5).unwrap()),
            T::from_real(T::Real::from_f64(-0.5).unwrap()),
            T::from_real(T::Real::from_f64(3.5).unwrap()),
            T::from_real(T::Real::from_f64(-1.0).unwrap()),
            T::from_real(T::Real::from_f64(0.25).unwrap()),
            T::from_real(T::Real::from_f64(-0.75).unwrap()),
            T::from_real(T::Real::from_f64(4.0).unwrap()),
            T::from_real(T::Real::from_f64(-1.0).unwrap()),
            T::from_real(T::Real::from_f64(3.0).unwrap()),
        ],
    )
}

#[test]
fn real_checkpoint_preserves_action_history_and_factor_metadata() {
    let source = ParIlutPreconditioner::from_csr_with_config(
        &matrix::<f64>(),
        &ParIlutConfig {
            sweeps: 2,
            track_residual: true,
            ..Default::default()
        },
    );
    let bytes = source.to_checkpoint_bytes().expect("serialize ParILUT");
    let restored =
        ParIlutPreconditioner::<f64>::from_checkpoint_bytes(&bytes).expect("restore ParILUT");
    let rhs = [1.0, -2.0, 0.5, 3.0];
    let expected = source.apply(&rhs);
    let actual = restored.apply(&rhs);
    assert!(
        expected
            .iter()
            .zip(&actual)
            .all(|(left, right)| (left - right).abs() < 1e-12)
    );
    assert_eq!(source.residual_history(), restored.residual_history());
    assert_eq!(source.factor_nnz(), restored.factor_nnz());
}

#[test]
fn complex_checkpoint_rejects_corruption_and_trailing_bytes() {
    let source = ParIlutPreconditioner::from_csr_with_config(
        &matrix::<Complex64>(),
        &ParIlutConfig {
            sweeps: 1,
            track_residual: true,
            ..Default::default()
        },
    );
    let bytes = source
        .to_checkpoint_bytes()
        .expect("serialize complex ParILUT");
    let restored = ParIlutPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes)
        .expect("restore complex ParILUT");
    let rhs = vec![
        Complex64::new(1.0, -0.5),
        Complex64::new(-2.0, 0.25),
        Complex64::new(0.5, 1.0),
        Complex64::new(3.0, -1.5),
    ];
    assert!(
        source
            .apply(&rhs)
            .iter()
            .zip(restored.apply(&rhs))
            .all(|(left, right)| (*left - right).norm() < 1e-12)
    );
    let mut bad_magic = bytes.clone();
    bad_magic[0] ^= 0xff;
    assert!(ParIlutPreconditioner::<Complex64>::from_checkpoint_bytes(&bad_magic).is_err());
    assert!(ParIlutPreconditioner::<f64>::from_checkpoint_bytes(&bytes).is_err());
    let mut trailing = bytes;
    trailing.push(0);
    assert!(ParIlutPreconditioner::<Complex64>::from_checkpoint_bytes(&trailing).is_err());
}
