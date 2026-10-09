use num_complex::Complex64;
use sonium_solvers_core::traits::Preconditioner;
use sonium_solvers_preconditioners::{
    BlockPreconditionerConfig, SADDLE_POINT_CHECKPOINT_FORMAT_VERSION, SaddlePointConfig,
    SaddlePointPreconditioner, SchurStrategy,
};
use sonium_solvers_sparse::CsrMatrix;

fn matrix() -> CsrMatrix<Complex64> {
    CsrMatrix::from_raw_parts(
        4,
        4,
        vec![0, 2, 4, 6, 8],
        vec![0, 1, 0, 1, 2, 3, 2, 3],
        vec![
            Complex64::new(4.0, 0.2),
            Complex64::new(-1.0, 0.1),
            Complex64::new(-1.0, -0.1),
            Complex64::new(3.0, 0.3),
            Complex64::new(2.0, 0.4),
            Complex64::new(0.5, -0.2),
            Complex64::new(0.5, 0.2),
            Complex64::new(1.5, 0.1),
        ],
    )
}

fn config() -> SaddlePointConfig {
    SaddlePointConfig {
        block1_size: 2,
        block1_config: BlockPreconditionerConfig::Diagonal,
        block2_config: BlockPreconditionerConfig::None,
        schur_strategy: SchurStrategy::Diagonal,
        upper_triangular: false,
    }
}

fn assert_close(left: &[Complex64], right: &[Complex64]) {
    assert_eq!(left.len(), right.len());
    for (a, b) in left.iter().zip(right) {
        assert!((*a - *b).norm() < 1.0e-12, "{a:?} != {b:?}");
    }
}

#[test]
fn checkpoint_round_trip_preserves_diagonal_actions_and_metadata() {
    let source = SaddlePointPreconditioner::try_from_matrix(&matrix(), config())
        .expect("checked saddle-point setup");
    let rhs = vec![
        Complex64::new(1.0, 0.25),
        Complex64::new(-2.0, 0.5),
        Complex64::new(0.5, -0.75),
        Complex64::new(3.0, 0.125),
    ];
    let forward = source.apply(&rhs);
    let transpose = source.apply_transpose(&rhs);
    let hermitian = source.apply_hermitian(&rhs);
    let bytes = source
        .to_checkpoint_bytes()
        .expect("serialize saddle point");
    assert_eq!(
        u16::from_le_bytes([bytes[8], bytes[9]]),
        SADDLE_POINT_CHECKPOINT_FORMAT_VERSION
    );
    let restored = SaddlePointPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes)
        .expect("restore saddle point");
    assert_close(&forward, &restored.apply(&rhs));
    assert_close(&transpose, &restored.apply_transpose(&rhs));
    assert_close(&hermitian, &restored.apply_hermitian(&rhs));
}

#[test]
fn checkpoint_rejects_unsupported_modes_and_corruption() {
    let source = SaddlePointPreconditioner::try_from_matrix(&matrix(), config())
        .expect("checked saddle-point setup");
    let bytes = source
        .to_checkpoint_bytes()
        .expect("serialize saddle point");
    for length in [0, 1, bytes.len() / 2, bytes.len().saturating_sub(1)] {
        assert!(
            SaddlePointPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes[..length])
                .is_err()
        );
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(SaddlePointPreconditioner::<Complex64>::from_checkpoint_bytes(&trailing).is_err());
    let mut bad_magic = bytes.clone();
    bad_magic[0] ^= 0xff;
    assert!(SaddlePointPreconditioner::<Complex64>::from_checkpoint_bytes(&bad_magic).is_err());
    let mut bad_reserved = bytes.clone();
    bad_reserved[13] = 1;
    assert!(SaddlePointPreconditioner::<Complex64>::from_checkpoint_bytes(&bad_reserved).is_err());
    let mut bad_strategy = bytes.clone();
    bad_strategy[38] = 1;
    assert!(SaddlePointPreconditioner::<Complex64>::from_checkpoint_bytes(&bad_strategy).is_err());
    assert!(SaddlePointPreconditioner::<f64>::from_checkpoint_bytes(&bytes).is_err());

    let mut unsupported = config();
    unsupported.schur_strategy = SchurStrategy::BtDiagB;
    let preconditioner = SaddlePointPreconditioner::from_matrix(&matrix(), unsupported);
    assert!(preconditioner.to_checkpoint_bytes().is_err());
}

#[test]
fn checkpoint_round_trip_supports_real_f64_scalars() {
    let matrix = CsrMatrix::<f64>::identity(4);
    let source = SaddlePointPreconditioner::try_from_matrix(
        &matrix,
        SaddlePointConfig {
            block1_size: 2,
            block1_config: BlockPreconditionerConfig::Diagonal,
            block2_config: BlockPreconditionerConfig::None,
            schur_strategy: SchurStrategy::Diagonal,
            upper_triangular: true,
        },
    )
    .expect("checked real saddle-point setup");
    let bytes = source
        .to_checkpoint_bytes()
        .expect("serialize real saddle point");
    let restored = SaddlePointPreconditioner::<f64>::from_checkpoint_bytes(&bytes)
        .expect("restore real saddle point");
    let rhs = [1.0, -2.0, 0.5, 3.0];
    assert_eq!(source.apply(&rhs), restored.apply(&rhs));
}
