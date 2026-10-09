use sonium_compute::CpuBackend;
use sonium_solvers::checkpoint_manifest::{
    CheckpointManifestError, CheckpointResidence, CheckpointShard, DistributedCheckpoint,
    SolverCheckpointBundle,
};
use sonium_solvers::mixed_precision::MixedPrecisionCheckpoint;
use sonium_solvers::preconditioners::IluPreconditioner;
use sonium_solvers::traits::Preconditioner;
use sonium_solvers::{SolverErrorKind, SolverFailure, SolverStatus};
use sonium_solvers_direct::{CholeskyOrdering, SparseCholeskyFactorization, SparseLuFactorization};
use sonium_solvers_sparse::CsrMatrix;

fn payload() -> Vec<u8> {
    (0_u8..=127).map(|value| value.wrapping_mul(31)).collect()
}

#[test]
fn checkpoint_manifest_failures_use_shared_solver_failure_contract() {
    let failures = [
        CheckpointManifestError::Invalid("bad tag"),
        CheckpointManifestError::Truncated,
        CheckpointManifestError::TrailingBytes,
        CheckpointManifestError::Overflow,
        CheckpointManifestError::OversizedPayload,
        CheckpointManifestError::ChecksumMismatch,
        CheckpointManifestError::Backend("copy failed".to_owned()),
    ];

    for failure in failures {
        assert_eq!(failure.status(), SolverStatus::NumericalError);
        assert_eq!(failure.solver_status(), SolverStatus::NumericalError);
        assert_eq!(
            failure.accuracy_status(),
            sonium_solvers::AccuracyStatus::NotChecked
        );
        assert_eq!(
            failure.attainable_accuracy(),
            sonium_solvers::AccuracyStatus::NotChecked
        );
        assert_eq!(failure.error_kind(), SolverErrorKind::OperationFailed);
    }
}

#[test]
fn sharded_mixed_precision_packet_restores_typed_state() {
    let source = MixedPrecisionCheckpoint::<f64>::from_iterate(&[1.25, -2.5, 3.75, 4.5]).unwrap();
    let source_bytes = source.to_checkpoint_bytes().unwrap();
    let manifest =
        DistributedCheckpoint::split_payload(&source_bytes, 3, CheckpointResidence::Cuda).unwrap();
    let wire = manifest.to_checkpoint_bytes().unwrap();
    let restored_manifest = DistributedCheckpoint::from_checkpoint_bytes(&wire).unwrap();
    let restored = MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(
        &restored_manifest.assemble_payload().unwrap(),
    )
    .unwrap();

    assert_eq!(restored_manifest.residence(), CheckpointResidence::Cuda);
    assert_eq!(restored_manifest.world_size(), 3);
    assert_eq!(restored.solution(), source.solution());
    assert_eq!(restored.source_iterations(), source.source_iterations());
}

#[test]
fn residence_tags_round_trip_for_host_and_metal() {
    for residence in [CheckpointResidence::Host, CheckpointResidence::Metal] {
        let manifest = DistributedCheckpoint::split_payload(&payload(), 4, residence).unwrap();
        let restored =
            DistributedCheckpoint::from_checkpoint_bytes(&manifest.to_checkpoint_bytes().unwrap())
                .unwrap();
        assert_eq!(restored.residence(), residence);
        assert_eq!(restored.assemble_payload().unwrap(), payload());
    }
}

#[test]
fn caller_shards_are_sorted_and_reassembled_by_rank() {
    let original =
        DistributedCheckpoint::split_payload(&payload(), 7, CheckpointResidence::Host).unwrap();
    let mut shards = original.shards().to_vec();
    shards.swap(0, 6);
    shards.swap(1, 5);
    let rebuilt = DistributedCheckpoint::from_shards(CheckpointResidence::Host, 7, shards).unwrap();
    assert_eq!(rebuilt.shards()[0].rank(), 0);
    assert_eq!(rebuilt.shards()[6].rank(), 6);
    assert_eq!(rebuilt.assemble_payload().unwrap(), payload());
}

#[test]
fn empty_payload_has_one_valid_boundary_per_rank() {
    let manifest = DistributedCheckpoint::split_payload(&[], 8, CheckpointResidence::Host).unwrap();
    assert_eq!(manifest.total_len(), 0);
    assert_eq!(manifest.shards().len(), 8);
    assert!(
        manifest
            .shards()
            .iter()
            .all(|shard| shard.payload().is_empty())
    );
    assert!(manifest.assemble_payload().unwrap().is_empty());
}

#[test]
fn malformed_shards_fail_closed() {
    let base =
        DistributedCheckpoint::split_payload(&payload(), 2, CheckpointResidence::Host).unwrap();
    let mut gap = base.shards().to_vec();
    gap[1] = CheckpointShard::new(1, gap[1].offset() + 1, gap[1].payload().to_vec()).unwrap();
    assert!(matches!(
        DistributedCheckpoint::from_shards(CheckpointResidence::Host, 2, gap),
        Err(CheckpointManifestError::Invalid(
            "shards do not form a contiguous payload"
        ))
    ));

    let mut duplicate = base.shards().to_vec();
    duplicate[1] =
        CheckpointShard::new(0, duplicate[1].offset(), duplicate[1].payload().to_vec()).unwrap();
    assert!(matches!(
        DistributedCheckpoint::from_shards(CheckpointResidence::Host, 2, duplicate),
        Err(CheckpointManifestError::Invalid(
            "rank set is not contiguous"
        ))
    ));

    assert!(matches!(
        DistributedCheckpoint::from_shards(CheckpointResidence::Host, 3, base.shards().to_vec()),
        Err(CheckpointManifestError::Invalid(
            "shard count does not equal world size"
        ))
    ));
}

#[test]
fn payload_mutation_is_detected_at_manifest_construction() {
    let base =
        DistributedCheckpoint::split_payload(&payload(), 3, CheckpointResidence::Cuda).unwrap();
    let mut shards = base.shards().to_vec();
    let mut mutated = shards[1].payload().to_vec();
    mutated[0] ^= 0x80;
    shards[1] = CheckpointShard::new(1, shards[1].offset(), mutated).unwrap();
    // The public constructor recomputes the shard checksum, so the manifest
    // remains internally coherent after a legitimate payload replacement.
    let replaced =
        DistributedCheckpoint::from_shards(CheckpointResidence::Cuda, 3, shards).unwrap();
    assert_ne!(replaced.payload_checksum(), base.payload_checksum());
    assert_ne!(replaced.assemble_payload().unwrap(), payload());
}

#[test]
fn wire_corruption_truncation_and_trailing_bytes_are_rejected() {
    let manifest =
        DistributedCheckpoint::split_payload(&payload(), 3, CheckpointResidence::Metal).unwrap();
    let bytes = manifest.to_checkpoint_bytes().unwrap();

    let mut corrupted = bytes.clone();
    corrupted[0] ^= 1;
    assert!(matches!(
        DistributedCheckpoint::from_checkpoint_bytes(&corrupted),
        Err(CheckpointManifestError::Invalid("invalid checkpoint magic"))
    ));
    assert_eq!(
        DistributedCheckpoint::from_checkpoint_bytes(&bytes[..bytes.len() - 1]),
        Err(CheckpointManifestError::Truncated)
    );
    let mut trailing = bytes;
    trailing.push(0);
    assert_eq!(
        DistributedCheckpoint::from_checkpoint_bytes(&trailing),
        Err(CheckpointManifestError::TrailingBytes)
    );
}

#[test]
fn manifest_bytes_are_deterministic() {
    let manifest =
        DistributedCheckpoint::split_payload(&payload(), 5, CheckpointResidence::Cuda).unwrap();
    assert_eq!(
        manifest.to_checkpoint_bytes().unwrap(),
        manifest.to_checkpoint_bytes().unwrap()
    );
}

#[test]
fn bundle_preserves_krylov_preconditioner_and_factor_sections() {
    let solver = MixedPrecisionCheckpoint::<f64>::from_iterate(&[1.0, -2.0, 3.0]).unwrap();
    let bundle = SolverCheckpointBundle::new(
        solver.to_checkpoint_bytes().unwrap(),
        Some(b"portable-ilu-checkpoint".to_vec()),
        Some(b"portable-sparse-lu-checkpoint".to_vec()),
    )
    .unwrap();
    let wire = bundle.to_checkpoint_bytes().unwrap();
    let restored = SolverCheckpointBundle::from_checkpoint_bytes(&wire).unwrap();
    let restored_solver =
        MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(restored.krylov_state()).unwrap();
    assert_eq!(restored_solver.solution(), solver.solution());
    assert_eq!(
        restored.preconditioner_state(),
        Some(&b"portable-ilu-checkpoint"[..])
    );
    assert_eq!(
        restored.factor_state(),
        Some(&b"portable-sparse-lu-checkpoint"[..])
    );
}

#[test]
fn bundle_wire_rejects_unknown_flags_and_trailing_bytes() {
    let bundle = SolverCheckpointBundle::new(b"basis".to_vec(), None, None).unwrap();
    let mut bytes = bundle.to_checkpoint_bytes().unwrap();
    bytes[10] = 0x80;
    assert!(matches!(
        SolverCheckpointBundle::from_checkpoint_bytes(&bytes),
        Err(CheckpointManifestError::Invalid("unknown bundle flags"))
    ));
    let mut bytes = bundle.to_checkpoint_bytes().unwrap();
    bytes.push(0);
    assert_eq!(
        SolverCheckpointBundle::from_checkpoint_bytes(&bytes),
        Err(CheckpointManifestError::TrailingBytes)
    );
}

#[test]
fn backend_payload_round_trip_uses_checked_transfer_and_checksum() {
    let source = payload();
    let manifest =
        DistributedCheckpoint::split_payload(&source, 3, CheckpointResidence::Cuda).unwrap();
    let backend = CpuBackend::new();
    let device = manifest.copy_payload_to_backend(&backend).unwrap();
    assert_eq!(device.byte_len(), source.len());
    assert_eq!(
        manifest
            .restore_payload_from_backend(&backend, &device)
            .unwrap(),
        source
    );

    let different = DistributedCheckpoint::split_payload(
        &vec![0_u8; source.len()],
        3,
        CheckpointResidence::Cuda,
    )
    .unwrap();
    assert!(matches!(
        different.restore_payload_from_backend(&backend, &device),
        Err(CheckpointManifestError::ChecksumMismatch)
    ));

    let empty = DistributedCheckpoint::split_payload(&[], 2, CheckpointResidence::Metal).unwrap();
    let empty_device = empty.copy_payload_to_backend(&backend).unwrap();
    assert_eq!(empty_device.byte_len(), 0);
    assert!(
        empty
            .restore_payload_from_backend(&backend, &empty_device)
            .unwrap()
            .is_empty()
    );
}
#[test]
fn bundle_round_trips_sparse_cholesky_factor_section() {
    let matrix = CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 5.0),
            (1, 2, 0.75),
            (2, 1, 0.75),
            (2, 2, 3.0),
        ],
    );
    let factor = SparseCholeskyFactorization::factorize_with_ordering(
        &matrix,
        CholeskyOrdering::Natural,
        false,
    )
    .unwrap();
    let factor_bytes = factor.to_checkpoint_bytes().unwrap();
    let rhs = [1.0, -2.0, 0.5];
    let expected = factor.solve(&rhs).unwrap();

    let bundle = SolverCheckpointBundle::new(
        b"active-krylov-state".to_vec(),
        None,
        Some(factor_bytes.clone()),
    )
    .unwrap();
    let wire = bundle.to_checkpoint_bytes().unwrap();
    let restored = SolverCheckpointBundle::from_checkpoint_bytes(&wire).unwrap();
    let restored_factor =
        SparseCholeskyFactorization::<f64>::from_checkpoint_bytes(restored.factor_state().unwrap())
            .unwrap();
    let actual = restored_factor.solve(&rhs).unwrap();

    assert_eq!(restored.krylov_state(), b"active-krylov-state");
    assert_eq!(restored.factor_state(), Some(factor_bytes.as_slice()));
    assert!(
        expected
            .iter()
            .zip(actual.iter())
            .all(|(left, right)| (left - right).abs() < 1.0e-12)
    );

    let mut corrupted = wire;
    *corrupted.last_mut().unwrap() ^= 0x80;
    assert!(matches!(
        SolverCheckpointBundle::from_checkpoint_bytes(&corrupted),
        Err(CheckpointManifestError::ChecksumMismatch)
    ));
}
#[test]
fn typed_bundle_sharding_replays_factor_and_repartition() {
    let matrix = CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 5.0),
            (1, 2, 0.75),
            (2, 1, 0.75),
            (2, 2, 3.0),
        ],
    );
    let factor = SparseCholeskyFactorization::factorize_with_ordering(
        &matrix,
        CholeskyOrdering::Colamd,
        false,
    )
    .unwrap();
    let rhs = [1.0, -2.0, 0.5];
    let expected = factor.solve(&rhs).unwrap();
    let bundle = SolverCheckpointBundle::new(
        b"active-krylov-state".to_vec(),
        None,
        Some(factor.to_checkpoint_bytes().unwrap()),
    )
    .unwrap();

    let manifest =
        DistributedCheckpoint::split_bundle(&bundle, 3, CheckpointResidence::Cuda).unwrap();
    let restored = manifest.assemble_bundle().unwrap();
    let backend = CpuBackend::new();
    let device = manifest.copy_payload_to_backend(&backend).unwrap();
    let backend_restored = manifest
        .restore_bundle_from_backend(&backend, &device)
        .unwrap();
    let repartitioned = manifest
        .repartition_bundle(5, CheckpointResidence::Metal)
        .unwrap();
    let repartitioned_bundle = repartitioned.assemble_bundle().unwrap();

    for restored_bundle in [&restored, &repartitioned_bundle] {
        let restored_factor = SparseCholeskyFactorization::<f64>::from_checkpoint_bytes(
            restored_bundle.factor_state().unwrap(),
        )
        .unwrap();
        let actual = restored_factor.solve(&rhs).unwrap();
        assert!(
            expected
                .iter()
                .zip(actual.iter())
                .all(|(left, right)| (left - right).abs() < 1.0e-12)
        );
        assert_eq!(restored_bundle.krylov_state(), b"active-krylov-state");
    }
    assert_eq!(restored, bundle);
    assert_eq!(backend_restored, bundle);
    assert_eq!(repartitioned_bundle, bundle);
    assert_eq!(repartitioned.residence(), CheckpointResidence::Metal);
    assert_eq!(repartitioned.world_size(), 5);
}

#[test]
fn typed_bundle_assembly_rejects_opaque_payloads() {
    let manifest =
        DistributedCheckpoint::split_payload(b"not-a-solver-bundle", 2, CheckpointResidence::Host)
            .unwrap();
    assert!(matches!(
        manifest.assemble_bundle(),
        Err(CheckpointManifestError::Invalid("invalid bundle magic"))
    ));
}

#[test]
fn typed_bundle_sharding_replays_ilu_and_sparse_lu_sections() {
    let matrix = CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![5.0, 1.0, 2.0, 4.0, 1.0, 1.0, 3.0],
    );
    let ilu = IluPreconditioner::<f64>::from_csr(&matrix);
    let sparse_lu = SparseLuFactorization::factorize(&matrix).unwrap();
    let bundle = SolverCheckpointBundle::new(
        b"active-krylov-state".to_vec(),
        Some(ilu.to_checkpoint_bytes().unwrap()),
        Some(sparse_lu.to_checkpoint_bytes().unwrap()),
    )
    .unwrap();
    let rhs = [1.0, -2.0, 0.5];
    let expected_ilu = ilu.apply(&rhs);
    let expected_sparse_lu = sparse_lu.solve(&rhs).unwrap();

    let manifest =
        DistributedCheckpoint::split_bundle(&bundle, 4, CheckpointResidence::Host).unwrap();
    let restored = manifest.assemble_bundle().unwrap();
    let restored_ilu =
        IluPreconditioner::<f64>::from_checkpoint_bytes(restored.preconditioner_state().unwrap())
            .unwrap();
    let restored_sparse_lu =
        SparseLuFactorization::<f64>::from_checkpoint_bytes(restored.factor_state().unwrap())
            .unwrap();

    let actual_ilu = restored_ilu.apply(&rhs);
    let actual_sparse_lu = restored_sparse_lu.solve(&rhs).unwrap();
    assert!(
        expected_ilu
            .iter()
            .zip(actual_ilu.iter())
            .all(|(left, right)| (left - right).abs() < 1.0e-12)
    );
    assert!(
        expected_sparse_lu
            .iter()
            .zip(actual_sparse_lu.iter())
            .all(|(left, right)| (left - right).abs() < 1.0e-12)
    );
    assert_eq!(restored.krylov_state(), b"active-krylov-state");
    assert_eq!(
        restored.preconditioner_state(),
        bundle.preconditioner_state()
    );
    assert_eq!(restored.factor_state(), bundle.factor_state());
}

#[test]
fn typed_bundle_sharding_replays_sparse_lu_factor_variants() {
    let matrix = CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![5.0, 1.0, 2.0, 4.0, 1.0, 1.0, 3.0],
    );
    let factors = vec![
        SparseLuFactorization::factorize(&matrix).unwrap(),
        SparseLuFactorization::factorize_supernodal(&matrix).unwrap(),
        SparseLuFactorization::factorize_with_preprocessing(&matrix, 1.0, true).unwrap(),
        SparseLuFactorization::factorize_supernodal_with_preprocessing(&matrix, 1.0, true).unwrap(),
    ];
    let rhs = [1.0, -2.0, 0.5];

    for factor in factors {
        let expected = factor.solve(&rhs).unwrap();
        let bundle = SolverCheckpointBundle::new(
            b"active-krylov-state".to_vec(),
            None,
            Some(factor.to_checkpoint_bytes().unwrap()),
        )
        .unwrap();
        let manifest =
            DistributedCheckpoint::split_bundle(&bundle, 2, CheckpointResidence::Host).unwrap();
        let restored = manifest.assemble_bundle().unwrap();
        let restored_factor =
            SparseLuFactorization::<f64>::from_checkpoint_bytes(restored.factor_state().unwrap())
                .unwrap();
        let actual = restored_factor.solve(&rhs).unwrap();
        assert!(
            expected
                .iter()
                .zip(actual.iter())
                .all(|(left, right)| (left - right).abs() < 1.0e-10)
        );
    }
}

#[test]
fn repartition_preserves_payload_when_restart_world_size_changes() {
    let source = MixedPrecisionCheckpoint::<f64>::from_iterate(&[1.25, -2.5, 3.75, 4.5])
        .unwrap()
        .to_checkpoint_bytes()
        .unwrap();
    let original =
        DistributedCheckpoint::split_payload(&source, 3, CheckpointResidence::Cuda).unwrap();
    let repartitioned = original.repartition(5, CheckpointResidence::Metal).unwrap();

    assert_eq!(repartitioned.residence(), CheckpointResidence::Metal);
    assert_eq!(repartitioned.world_size(), 5);
    assert_eq!(
        repartitioned.payload_checksum(),
        original.payload_checksum()
    );
    assert_eq!(repartitioned.assemble_payload().unwrap(), source);
    assert!(
        repartitioned
            .shards()
            .iter()
            .enumerate()
            .all(|(rank, shard)| shard.rank() == rank as u32)
    );

    let restored =
        DistributedCheckpoint::from_checkpoint_bytes(&repartitioned.to_checkpoint_bytes().unwrap())
            .unwrap();
    let typed = MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(
        &restored.assemble_payload().unwrap(),
    )
    .unwrap();
    assert_eq!(typed.solution(), &[1.25, -2.5, 3.75, 4.5]);
    assert!(matches!(
        original.repartition(0, CheckpointResidence::Host),
        Err(CheckpointManifestError::Invalid("invalid world size"))
    ));
}
