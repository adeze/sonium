//! Independent CPU evidence for rank/device checkpoint sharding and reassembly.

use sonium_compute::CpuBackend;
use sonium_qa::QaResult;
use sonium_solvers::checkpoint_manifest::{
    CheckpointResidence, DistributedCheckpoint, SolverCheckpointBundle,
};
use sonium_solvers::mixed_precision::MixedPrecisionCheckpoint;
use sonium_solvers::preconditioners::IluPreconditioner;
use sonium_solvers::traits::Preconditioner;
use sonium_solvers_direct::SparseLuFactorization;
use sonium_solvers_direct::{CholeskyOrdering, SparseCholeskyFactorization};
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.distributed-checkpoint-manifest.v1";
const TOLERANCE: f64 = 1.0e-12;

fn max_delta(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let source =
        MixedPrecisionCheckpoint::<f64>::from_iterate(&[1.25, -2.5, 3.75, 4.5, -0.75, 2.125])
            .map_err(|error| error.to_string())?;
    let source_bytes = source
        .to_checkpoint_bytes()
        .map_err(|error| error.to_string())?;
    let matrix = CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![5.0, 1.0, 2.0, 4.0, 1.0, 1.0, 3.0],
    );
    let ilu = IluPreconditioner::<f64>::from_csr(&matrix);
    let ilu_bytes = ilu
        .to_checkpoint_bytes()
        .map_err(|error| error.to_string())?;
    IluPreconditioner::<f64>::from_checkpoint_bytes(&ilu_bytes)
        .map_err(|error| error.to_string())?;
    let sparse_lu =
        SparseLuFactorization::<f64>::factorize(&matrix).map_err(|error| error.to_string())?;
    let sparse_lu_bytes = sparse_lu
        .to_checkpoint_bytes()
        .map_err(|error| error.to_string())?;
    SparseLuFactorization::<f64>::from_checkpoint_bytes(&sparse_lu_bytes)
        .map_err(|error| error.to_string())?;
    let cholesky_matrix = CsrMatrix::from_triplets(
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
    let cholesky = SparseCholeskyFactorization::factorize_with_ordering(
        &cholesky_matrix,
        CholeskyOrdering::Natural,
        false,
    )
    .map_err(|error| error.to_string())?;
    let cholesky_bytes = cholesky
        .to_checkpoint_bytes()
        .map_err(|error| error.to_string())?;
    let restored_cholesky =
        SparseCholeskyFactorization::<f64>::from_checkpoint_bytes(&cholesky_bytes)
            .map_err(|error| error.to_string())?;
    let cholesky_rhs = [1.0, -2.0, 0.5];
    let cholesky_expected = cholesky
        .solve(&cholesky_rhs)
        .map_err(|error| error.to_string())?;
    let cholesky_actual = restored_cholesky
        .solve(&cholesky_rhs)
        .map_err(|error| error.to_string())?;
    let cholesky_replay_delta = max_delta(&cholesky_expected, &cholesky_actual);
    let cholesky_bundle = SolverCheckpointBundle::new(
        source_bytes.clone(),
        Some(ilu_bytes.clone()),
        Some(cholesky_bytes.clone()),
    )
    .map_err(|error| error.to_string())?;
    let cholesky_bundle_wire = cholesky_bundle
        .to_checkpoint_bytes()
        .map_err(|error| error.to_string())?;
    let cholesky_bundle_restored =
        SolverCheckpointBundle::from_checkpoint_bytes(&cholesky_bundle_wire)
            .map_err(|error| error.to_string())?;
    let cholesky_bundle_factor = SparseCholeskyFactorization::<f64>::from_checkpoint_bytes(
        cholesky_bundle_restored
            .factor_state()
            .ok_or_else(|| "Sparse-Cholesky bundle factor section missing".to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let cholesky_bundle_solution = cholesky_bundle_factor
        .solve(&cholesky_rhs)
        .map_err(|error| error.to_string())?;
    let cholesky_bundle_delta = max_delta(&cholesky_expected, &cholesky_bundle_solution);
    let bundle = SolverCheckpointBundle::new(
        source_bytes.clone(),
        Some(ilu_bytes.clone()),
        Some(sparse_lu_bytes.clone()),
    )
    .map_err(|error| error.to_string())?;
    let bundle_bytes = bundle
        .to_checkpoint_bytes()
        .map_err(|error| error.to_string())?;
    let manifest =
        DistributedCheckpoint::split_payload(&bundle_bytes, 3, CheckpointResidence::Cuda)
            .map_err(|error| error.to_string())?;
    let typed_manifest = DistributedCheckpoint::split_bundle(&bundle, 3, CheckpointResidence::Cuda)
        .map_err(|error| error.to_string())?;
    let typed_repartitioned = typed_manifest
        .repartition_bundle(5, CheckpointResidence::Metal)
        .map_err(|error| error.to_string())?;
    let typed_repartitioned_bundle = typed_repartitioned
        .assemble_bundle()
        .map_err(|error| error.to_string())?;
    let restored_ilu = IluPreconditioner::<f64>::from_checkpoint_bytes(
        typed_repartitioned_bundle
            .preconditioner_state()
            .ok_or_else(|| "typed ILU section missing".to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let restored_sparse_lu = SparseLuFactorization::<f64>::from_checkpoint_bytes(
        typed_repartitioned_bundle
            .factor_state()
            .ok_or_else(|| "typed Sparse-LU section missing".to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let factor_rhs = [1.0, -2.0, 0.5];
    let expected_ilu = ilu.apply(&factor_rhs);
    let actual_ilu = restored_ilu.apply(&factor_rhs);
    let expected_sparse_lu = sparse_lu
        .solve(&factor_rhs)
        .map_err(|error| error.to_string())?;
    let actual_sparse_lu = restored_sparse_lu
        .solve(&factor_rhs)
        .map_err(|error| error.to_string())?;
    let ilu_bundle_delta = max_delta(&expected_ilu, &actual_ilu);
    let sparse_lu_bundle_delta = max_delta(&expected_sparse_lu, &actual_sparse_lu);
    let sparse_lu_variant_max_delta = {
        let variants = [
            SparseLuFactorization::factorize(&matrix).map_err(|error| error.to_string())?,
            SparseLuFactorization::factorize_supernodal(&matrix)
                .map_err(|error| error.to_string())?,
            SparseLuFactorization::factorize_with_preprocessing(&matrix, 1.0, true)
                .map_err(|error| error.to_string())?,
            SparseLuFactorization::factorize_supernodal_with_preprocessing(&matrix, 1.0, true)
                .map_err(|error| error.to_string())?,
        ];
        let mut maximum = 0.0_f64;
        for variant in variants {
            let expected = variant
                .solve(&factor_rhs)
                .map_err(|error| error.to_string())?;
            let variant_bundle = SolverCheckpointBundle::new(
                source_bytes.clone(),
                Some(ilu_bytes.clone()),
                Some(
                    variant
                        .to_checkpoint_bytes()
                        .map_err(|error| error.to_string())?,
                ),
            )
            .map_err(|error| error.to_string())?;
            let variant_manifest =
                DistributedCheckpoint::split_bundle(&variant_bundle, 2, CheckpointResidence::Host)
                    .map_err(|error| error.to_string())?;
            let restored_variant = variant_manifest
                .assemble_bundle()
                .map_err(|error| error.to_string())?;
            let restored_factor = SparseLuFactorization::<f64>::from_checkpoint_bytes(
                restored_variant
                    .factor_state()
                    .ok_or_else(|| "variant Sparse-LU section missing".to_string())?,
            )
            .map_err(|error| error.to_string())?;
            let actual = restored_factor
                .solve(&factor_rhs)
                .map_err(|error| error.to_string())?;
            maximum = maximum.max(max_delta(&expected, &actual));
        }
        maximum
    };
    let typed_bundle_round_trip = f64::from(
        typed_repartitioned_bundle.krylov_state() == source_bytes.as_slice()
            && typed_repartitioned_bundle.preconditioner_state() == Some(ilu_bytes.as_slice())
            && typed_repartitioned_bundle.factor_state() == Some(sparse_lu_bytes.as_slice()),
    );
    let opaque_bundle_rejected = f64::from(
        DistributedCheckpoint::split_payload(b"not-a-solver-bundle", 2, CheckpointResidence::Host)
            .map_err(|error| error.to_string())?
            .assemble_bundle()
            .is_err(),
    );
    let repartitioned = manifest
        .repartition(5, CheckpointResidence::Metal)
        .map_err(|error| error.to_string())?;
    let repartition_round_trip = f64::from(
        repartitioned.world_size() == 5
            && repartitioned.residence() == CheckpointResidence::Metal
            && repartitioned.payload_checksum() == manifest.payload_checksum()
            && repartitioned
                .assemble_payload()
                .map_err(|error| error.to_string())?
                == bundle_bytes,
    );
    let wire = manifest
        .to_checkpoint_bytes()
        .map_err(|error| error.to_string())?;
    let restored_manifest =
        DistributedCheckpoint::from_checkpoint_bytes(&wire).map_err(|error| error.to_string())?;
    let backend = CpuBackend::new();
    let device_payload = restored_manifest
        .copy_payload_to_backend(&backend)
        .map_err(|error| error.to_string())?;
    let device_bundle = restored_manifest
        .restore_bundle_from_backend(&backend, &device_payload)
        .map_err(|error| error.to_string())?;
    let restored_bundle = SolverCheckpointBundle::from_checkpoint_bytes(
        &restored_manifest
            .assemble_payload()
            .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let restored =
        MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(restored_bundle.krylov_state())
            .map_err(|error| error.to_string())?;

    let mut corrupted = wire.clone();
    corrupted[0] ^= 1;
    let corruption_rejected =
        f64::from(DistributedCheckpoint::from_checkpoint_bytes(&corrupted).is_err());
    let mut trailing = wire.clone();
    trailing.push(0);
    let trailing_rejected =
        f64::from(DistributedCheckpoint::from_checkpoint_bytes(&trailing).is_err());
    let typed_round_trip_delta = max_delta(source.solution(), restored.solution());
    let payload_round_trip_delta = f64::from(source_bytes != restored_bundle.krylov_state());
    let bundle_sections = f64::from(
        restored_bundle.preconditioner_state() == Some(ilu_bytes.as_slice())
            && restored_bundle.factor_state() == Some(sparse_lu_bytes.as_slice()),
    );
    let backend_transfer_round_trip = f64::from(device_bundle.krylov_state() == source_bytes);
    let offsets_contiguous = f64::from(
        manifest
            .shards()
            .iter()
            .enumerate()
            .all(|(rank, shard)| shard.rank() == rank as u32),
    );
    let metadata_match = f64::from(
        restored_manifest.residence() == CheckpointResidence::Cuda
            && restored_manifest.world_size() == 3
            && restored_manifest.payload_checksum() == manifest.payload_checksum()
            && restored_manifest.shards().len() == 3,
    );
    let deterministic_wire = f64::from(wire == manifest.to_checkpoint_bytes().unwrap());

    Ok(vec![
        ("typed_round_trip_delta", typed_round_trip_delta),
        ("payload_round_trip_delta", payload_round_trip_delta),
        ("bundle_sections", bundle_sections),
        ("repartition_round_trip", repartition_round_trip),
        ("typed_bundle_round_trip", typed_bundle_round_trip),
        ("opaque_bundle_rejected", opaque_bundle_rejected),
        ("ilu_bundle_delta", ilu_bundle_delta),
        ("sparse_lu_bundle_delta", sparse_lu_bundle_delta),
        ("sparse_lu_variant_max_delta", sparse_lu_variant_max_delta),
        ("sparse_cholesky_replay_delta", cholesky_replay_delta),
        ("sparse_cholesky_bundle_delta", cholesky_bundle_delta),
        ("backend_transfer_round_trip", backend_transfer_round_trip),
        ("metadata_match", metadata_match),
        ("offsets_contiguous", offsets_contiguous),
        ("deterministic_wire", deterministic_wire),
        ("corruption_rejected", corruption_rejected),
        ("trailing_rejected", trailing_rejected),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
            .reference("portable bundle repartitioning with ILU/Sparse-LU/Sparse-Cholesky action replay")
                .input_checksum("analytic/mixed-precision-iterate-v1")
                .reference_checksum("analytic/mixed-precision-iterate-v1")
            .dimension("6-value Krylov state; ILU/Sparse-LU/Sparse-Cholesky action sections; 3-to-5 rank restart repartition")
                .tolerance("round-trip delta <= 1e-12; metadata and rejection gates = 1")
                .seed("solvers-distributed-checkpoint-manifest:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.metric(name, value);
                if name.contains("delta") {
                    result = result.maximum(name, TOLERANCE);
                } else {
                    result = result.minimum(name, 1.0);
                }
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result =
                QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64);
            eprintln!("FAILED: {result:?}");
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            std::process::exit(1);
        }
    }
}
