use num_complex::{Complex32, Complex64};
use sonium_solvers_core::traits::{ComplexField, Preconditioner};
use sonium_solvers_preconditioners::{IluPreconditioner, IluReordering};
use sonium_solvers_sparse::CsrMatrix;

fn matrix<T: ComplexField>(values: [T; 7]) -> CsrMatrix<T> {
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        values.to_vec(),
    )
}

#[test]
fn real_and_complex_checkpoint_round_trip() {
    let real = IluPreconditioner::from_csr(&matrix([5.0, 1.0, 2.0, 4.0, 1.0, 1.0, 3.0]));
    let real_restored = IluPreconditioner::<f64>::from_checkpoint_bytes(
        &real.to_checkpoint_bytes().expect("serialize real ILU"),
    )
    .expect("restore real ILU");
    let rhs = [1.0, 2.0, 3.0];
    assert_eq!(real.apply(&rhs), real_restored.apply(&rhs));

    let real32 = IluPreconditioner::from_csr(&matrix([5.0_f32, 1.0, 2.0, 4.0, 1.0, 1.0, 3.0]));
    let real32_restored = IluPreconditioner::<f32>::from_checkpoint_bytes(
        &real32.to_checkpoint_bytes().expect("serialize f32 ILU"),
    )
    .expect("restore f32 ILU");
    let rhs = [1.0_f32, 2.0, 3.0];
    for (actual, expected) in real32.apply(&rhs).iter().zip(real32_restored.apply(&rhs)) {
        assert!((actual - expected).abs() < 1.0e-6);
    }

    let complex = IluPreconditioner::from_csr(&matrix([
        Complex64::new(5.0, 0.5),
        Complex64::new(1.0, -0.25),
        Complex64::new(2.0, 0.1),
        Complex64::new(4.0, -0.5),
        Complex64::new(1.0, 0.2),
        Complex64::new(1.0, -0.1),
        Complex64::new(3.0, 0.3),
    ]));
    let complex_restored = IluPreconditioner::<Complex64>::from_checkpoint_bytes(
        &complex
            .to_checkpoint_bytes()
            .expect("serialize complex ILU"),
    )
    .expect("restore complex ILU");
    let rhs = [
        Complex64::new(1.0, 0.5),
        Complex64::new(2.0, -0.25),
        Complex64::new(3.0, 0.75),
    ];
    for (actual, expected) in complex.apply(&rhs).iter().zip(complex_restored.apply(&rhs)) {
        assert!((*actual - expected).norm() < 1.0e-12);
    }

    let complex32 = IluPreconditioner::from_csr(&matrix([
        Complex32::new(5.0, 0.5),
        Complex32::new(1.0, -0.25),
        Complex32::new(2.0, 0.1),
        Complex32::new(4.0, -0.5),
        Complex32::new(1.0, 0.2),
        Complex32::new(1.0, -0.1),
        Complex32::new(3.0, 0.3),
    ]));
    let complex32_restored = IluPreconditioner::<Complex32>::from_checkpoint_bytes(
        &complex32
            .to_checkpoint_bytes()
            .expect("serialize Complex32 ILU"),
    )
    .expect("restore Complex32 ILU");
    let rhs = [
        Complex32::new(1.0, 0.5),
        Complex32::new(2.0, -0.25),
        Complex32::new(3.0, 0.75),
    ];
    for (actual, expected) in complex32
        .apply(&rhs)
        .iter()
        .zip(complex32_restored.apply(&rhs))
    {
        assert!((*actual - expected).norm() < 1.0e-6);
    }
}

#[test]
fn reordered_checkpoint_preserves_original_numbering() {
    let preconditioner = IluPreconditioner::from_csr_with_reordering(
        &matrix([5.0, 1.0, 2.0, 4.0, 1.0, 1.0, 3.0]),
        IluReordering::Colamd,
    );
    let restored = IluPreconditioner::<f64>::from_checkpoint_bytes(
        &preconditioner
            .to_checkpoint_bytes()
            .expect("serialize reordered ILU"),
    )
    .expect("restore reordered ILU");
    let rhs = [1.0, -2.0, 0.5];
    assert_eq!(preconditioner.apply(&rhs), restored.apply(&rhs));
}

#[test]
fn checkpoint_rejects_corruption_and_scalar_mismatch() {
    let preconditioner = IluPreconditioner::from_csr(&matrix([5.0, 1.0, 2.0, 4.0, 1.0, 1.0, 3.0]));
    let bytes = preconditioner.to_checkpoint_bytes().expect("serialize ILU");

    let mut bad_magic = bytes.clone();
    bad_magic[0] ^= 0xff;
    assert!(IluPreconditioner::<f64>::from_checkpoint_bytes(&bad_magic).is_err());
    assert!(IluPreconditioner::<f64>::from_checkpoint_bytes(&bytes[..bytes.len() - 1]).is_err());

    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(IluPreconditioner::<f64>::from_checkpoint_bytes(&trailing).is_err());
    assert!(IluPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes).is_err());
}

#[test]
fn empty_checkpoint_round_trip() {
    let preconditioner = IluPreconditioner::from_csr(&CsrMatrix::<f64>::from_raw_parts(
        0,
        0,
        vec![0],
        vec![],
        vec![],
    ));
    let restored = IluPreconditioner::<f64>::from_checkpoint_bytes(
        &preconditioner
            .to_checkpoint_bytes()
            .expect("serialize empty ILU"),
    )
    .expect("restore empty ILU");
    assert!(restored.apply(&[]).is_empty());
}
