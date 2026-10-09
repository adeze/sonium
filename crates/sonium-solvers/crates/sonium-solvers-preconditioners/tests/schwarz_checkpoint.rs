use num_complex::Complex64;
use sonium_solvers_core::traits::Preconditioner;
use sonium_solvers_preconditioners::{
    AdditiveSchwarzPreconditioner, SCHWARZ_CHECKPOINT_FORMAT_VERSION, SchwarzCoarseSetup,
};
use sonium_solvers_sparse::CsrMatrix;

fn fixture_matrix() -> CsrMatrix<Complex64> {
    CsrMatrix::from_raw_parts(
        6,
        6,
        vec![0, 2, 5, 8, 11, 14, 16],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5],
        vec![
            Complex64::new(4.0, 0.2),
            Complex64::new(-1.0, 0.1),
            Complex64::new(-1.0, -0.1),
            Complex64::new(4.5, 0.3),
            Complex64::new(-1.0, 0.2),
            Complex64::new(-1.0, -0.2),
            Complex64::new(4.5, 0.1),
            Complex64::new(-1.0, 0.1),
            Complex64::new(-1.0, -0.1),
            Complex64::new(4.5, 0.2),
            Complex64::new(-1.0, 0.2),
            Complex64::new(-1.0, -0.2),
            Complex64::new(4.5, 0.1),
            Complex64::new(-1.0, 0.1),
            Complex64::new(-1.0, -0.1),
            Complex64::new(4.0, 0.2),
        ],
    )
}

fn fixture_preconditioner() -> AdditiveSchwarzPreconditioner<Complex64> {
    AdditiveSchwarzPreconditioner::try_from_csr_partitions(
        &fixture_matrix(),
        &[vec![0, 1, 2], vec![3, 4, 5]],
        1,
    )
    .expect("checked Schwarz fixture construction")
}

fn assert_close(lhs: &[Complex64], rhs: &[Complex64]) {
    assert_eq!(lhs.len(), rhs.len());
    for (left, right) in lhs.iter().zip(rhs) {
        assert!((*left - *right).norm() < 1.0e-12, "{left:?} != {right:?}");
    }
}

#[test]
fn checkpoint_round_trip_preserves_actions_and_metadata() {
    let source = fixture_preconditioner();
    let rhs = [
        Complex64::new(1.0, 0.25),
        Complex64::new(-2.0, 0.5),
        Complex64::new(0.5, -0.75),
        Complex64::new(3.0, 0.125),
        Complex64::new(-1.5, 0.25),
        Complex64::new(0.75, -0.5),
    ];
    let forward = source.try_apply(&rhs).expect("forward action");
    let transpose = source.try_apply_transpose(&rhs).expect("transpose action");
    let hermitian = source.try_apply_hermitian(&rhs).expect("Hermitian action");
    let stats = source.stats();
    let diagnostics = source.diagnostics();
    let bytes = source.to_checkpoint_bytes().expect("serialize Schwarz");
    assert_eq!(
        u16::from_le_bytes([bytes[8], bytes[9]]),
        SCHWARZ_CHECKPOINT_FORMAT_VERSION
    );

    // Runtime state is deliberately not part of a portable snapshot.
    assert!(source.runtime_diagnostics().apply_calls > 0);
    let restored =
        AdditiveSchwarzPreconditioner::from_checkpoint_bytes(&bytes).expect("restore Schwarz");
    assert_eq!(restored.stats(), stats);
    assert_eq!(restored.diagnostics(), diagnostics);
    assert_eq!(restored.coarse_setup(), SchwarzCoarseSetup::NotRequested);
    assert_eq!(restored.coarse_dimension(), 0);
    assert_eq!(restored.runtime_diagnostics().apply_calls, 0);
    assert_close(
        &forward,
        &restored.try_apply(&rhs).expect("restored forward"),
    );
    assert_close(
        &transpose,
        &restored
            .try_apply_transpose(&rhs)
            .expect("restored transpose"),
    );
    assert_close(
        &hermitian,
        &restored
            .try_apply_hermitian(&rhs)
            .expect("restored Hermitian"),
    );
}

#[test]
fn checkpoint_rejects_truncation_trailing_and_scalar_mismatch() {
    let bytes = fixture_preconditioner()
        .to_checkpoint_bytes()
        .expect("serialize Schwarz");

    for length in [0, 1, bytes.len() / 2, bytes.len() - 1] {
        assert!(
            AdditiveSchwarzPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes[..length])
                .is_err(),
            "truncated length {length} unexpectedly accepted"
        );
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(AdditiveSchwarzPreconditioner::<Complex64>::from_checkpoint_bytes(&trailing).is_err());

    let mut wrong_scalar = bytes.clone();
    // Header: magic (8), version (2), scalar size (1).
    wrong_scalar[10] = std::mem::size_of::<f64>() as u8;
    assert!(
        AdditiveSchwarzPreconditioner::<Complex64>::from_checkpoint_bytes(&wrong_scalar).is_err()
    );
    assert!(AdditiveSchwarzPreconditioner::<f64>::from_checkpoint_bytes(&bytes).is_err());
}

#[test]
fn checkpoint_round_trip_preserves_coarse_actions() {
    let matrix = fixture_matrix();
    let source = AdditiveSchwarzPreconditioner::try_from_csr_partitions(
        &matrix,
        &[vec![0, 1, 2], vec![3, 4, 5]],
        1,
    )
    .expect("checked Schwarz fixture construction")
    .with_nicolaides_coarse_space(&matrix);
    assert_eq!(source.coarse_setup(), SchwarzCoarseSetup::Attached);
    assert_eq!(source.coarse_dimension(), 2);
    let rhs = [
        Complex64::new(1.0, 0.25),
        Complex64::new(-2.0, 0.5),
        Complex64::new(0.5, -0.75),
        Complex64::new(3.0, 0.125),
        Complex64::new(-1.5, 0.25),
        Complex64::new(0.75, -0.5),
    ];
    let forward = source.try_apply(&rhs).expect("coarse forward action");
    let transpose = source
        .try_apply_transpose(&rhs)
        .expect("coarse transpose action");
    let hermitian = source
        .try_apply_hermitian(&rhs)
        .expect("coarse Hermitian action");
    let bytes = source
        .to_checkpoint_bytes()
        .expect("serialize coarse Schwarz");
    assert_eq!(
        u16::from_le_bytes([bytes[8], bytes[9]]),
        SCHWARZ_CHECKPOINT_FORMAT_VERSION
    );
    let restored = AdditiveSchwarzPreconditioner::from_checkpoint_bytes(&bytes)
        .expect("restore coarse Schwarz");
    assert_eq!(restored.coarse_setup(), SchwarzCoarseSetup::Attached);
    assert_eq!(restored.coarse_dimension(), 2);
    assert_eq!(restored.runtime_diagnostics().apply_calls, 0);
    assert_close(
        &forward,
        &restored.try_apply(&rhs).expect("restored coarse forward"),
    );
    assert_close(
        &transpose,
        &restored
            .try_apply_transpose(&rhs)
            .expect("restored coarse transpose"),
    );
    assert_close(
        &hermitian,
        &restored
            .try_apply_hermitian(&rhs)
            .expect("restored coarse Hermitian"),
    );
}

#[test]
fn checkpoint_v1_one_level_packet_remains_readable() {
    let mut bytes = fixture_preconditioner()
        .to_checkpoint_bytes()
        .expect("serialize Schwarz");
    bytes[8..10].copy_from_slice(&1_u16.to_le_bytes());
    let restored = AdditiveSchwarzPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes)
        .expect("restore legacy v1 Schwarz");
    assert_eq!(restored.coarse_setup(), SchwarzCoarseSetup::NotRequested);
    assert_eq!(restored.coarse_dimension(), 0);
}

#[test]
fn checkpoint_rejects_corrupt_nested_coarse_state() {
    let matrix = fixture_matrix();
    let coarse = AdditiveSchwarzPreconditioner::try_from_csr_partitions(
        &matrix,
        &[vec![0, 1, 2], vec![3, 4, 5]],
        1,
    )
    .expect("checked Schwarz fixture construction")
    .with_nicolaides_coarse_space(&matrix);
    let bytes = coarse
        .to_checkpoint_bytes()
        .expect("serialize coarse Schwarz");
    let factor_magic = b"SNLUCP01";
    let first_factor = bytes
        .windows(factor_magic.len())
        .position(|window| window == factor_magic)
        .expect("nested Sparse-LU factor magic");

    let mut bad_factor = bytes.clone();
    bad_factor[first_factor] ^= 0xff;
    assert!(
        AdditiveSchwarzPreconditioner::<Complex64>::from_checkpoint_bytes(&bad_factor).is_err(),
        "corrupt nested factor must be rejected"
    );

    // The fixture has a 6x2 coarse CSR basis with six stored values.  Its
    // canonical packet occupies 240 bytes before the first factor length.
    let z_start = first_factor
        .checked_sub(8 + 240)
        .expect("coarse CSR prefix before nested factor");
    let mut bad_csr = bytes;
    let first_column = z_start + 8 + 8 + 8 + 7 * 8 + 8;
    bad_csr[first_column..first_column + 8].copy_from_slice(&2_u64.to_le_bytes());
    assert!(
        AdditiveSchwarzPreconditioner::<Complex64>::from_checkpoint_bytes(&bad_csr).is_err(),
        "out-of-range coarse CSR column must be rejected"
    );
}

#[test]
fn checkpoint_rejects_invalid_packet_magic() {
    let mut bytes = fixture_preconditioner()
        .to_checkpoint_bytes()
        .expect("serialize Schwarz");
    bytes[0] ^= 0xff;
    assert!(AdditiveSchwarzPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes).is_err());
}
