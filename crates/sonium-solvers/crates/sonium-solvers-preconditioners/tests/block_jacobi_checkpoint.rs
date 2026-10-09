use num_complex::Complex64;
use num_traits::FromPrimitive;
use sonium_solvers_core::traits::{ComplexField, Preconditioner};
use sonium_solvers_preconditioners::{
    BLOCK_JACOBI_CHECKPOINT_FORMAT_VERSION, BlockJacobiPreconditioner, IlutConfig,
};
use sonium_solvers_sparse::CsrMatrix;

fn matrix<T: ComplexField>() -> CsrMatrix<T> {
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
fn checkpoint_rejects_overlapping_and_out_of_bounds_scatter_indices() {
    let source =
        BlockJacobiPreconditioner::try_from_csr(&matrix::<f64>(), 2, IlutConfig::default())
            .unwrap();
    let bytes = source.to_checkpoint_bytes().unwrap();
    assert_eq!(BLOCK_JACOBI_CHECKPOINT_FORMAT_VERSION, 1);
    let read = |offset| u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
    // v1 header (28 bytes), followed by block length, indices, nested length/payload.
    assert_eq!(read(12), 4);
    assert_eq!(read(20), 2);
    assert_eq!(read(28), 2);
    let second_block = 60 + usize::try_from(read(52)).unwrap();
    assert_eq!(read(second_block), 2);
    let first_index = read(36);
    for offset in [44, second_block + 8] {
        let mut corrupt = bytes.clone();
        corrupt[offset..offset + 8].copy_from_slice(&first_index.to_le_bytes());
        let error = BlockJacobiPreconditioner::<f64>::from_checkpoint_bytes(&corrupt)
            .err()
            .expect("duplicate scatter index must be rejected");
        assert!(error.to_string().contains("duplicate global index"));
    }
    let mut corrupt = bytes;
    corrupt[36..44].copy_from_slice(&4_u64.to_le_bytes());
    let error = BlockJacobiPreconditioner::<f64>::from_checkpoint_bytes(&corrupt)
        .err()
        .expect("out-of-bounds scatter index must be rejected");
    assert!(error.to_string().contains("outside matrix dimension"));
}

#[test]
fn real_checkpoint_preserves_action_and_partition_statistics() {
    let source = BlockJacobiPreconditioner::try_from_csr(
        &matrix::<f64>(),
        2,
        IlutConfig {
            fill_factor: 4,
            ..Default::default()
        },
    )
    .expect("construct block Jacobi");
    let bytes = source
        .to_checkpoint_bytes()
        .expect("serialize block Jacobi");
    assert_eq!(
        u16::from_le_bytes([bytes[8], bytes[9]]),
        BLOCK_JACOBI_CHECKPOINT_FORMAT_VERSION
    );
    let restored = BlockJacobiPreconditioner::<f64>::from_checkpoint_bytes(&bytes)
        .expect("restore block Jacobi");
    assert_eq!(source.stats(), restored.stats());

    let rhs = [1.0, -2.0, 0.5, 3.0];
    assert_eq!(
        source.try_apply(&rhs).unwrap(),
        restored.try_apply(&rhs).unwrap()
    );
}

#[test]
fn complex_checkpoint_rejects_corruption_scalar_mismatch_and_trailing_bytes() {
    let source =
        BlockJacobiPreconditioner::try_from_csr(&matrix::<Complex64>(), 2, IlutConfig::default())
            .expect("construct complex block Jacobi");
    let bytes = source
        .to_checkpoint_bytes()
        .expect("serialize block Jacobi");
    let restored = BlockJacobiPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes)
        .expect("restore complex block Jacobi");
    let rhs = [
        Complex64::new(1.0, 0.25),
        Complex64::new(-2.0, 0.5),
        Complex64::new(0.5, -0.75),
        Complex64::new(3.0, 0.125),
    ];
    for (actual, expected) in source
        .try_apply(&rhs)
        .unwrap()
        .iter()
        .zip(restored.try_apply(&rhs).unwrap())
    {
        assert!((*actual - expected).norm() < 1.0e-12);
    }

    let mut bad_magic = bytes.clone();
    bad_magic[0] ^= 0xff;
    assert!(BlockJacobiPreconditioner::<Complex64>::from_checkpoint_bytes(&bad_magic).is_err());

    let mut bad_scalar = bytes.clone();
    bad_scalar[10] = std::mem::size_of::<f64>() as u8;
    assert!(BlockJacobiPreconditioner::<Complex64>::from_checkpoint_bytes(&bad_scalar).is_err());

    let mut trailing = bytes;
    trailing.push(0);
    assert!(BlockJacobiPreconditioner::<Complex64>::from_checkpoint_bytes(&trailing).is_err());
}
