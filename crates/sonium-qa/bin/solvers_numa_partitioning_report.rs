//! Deterministic CPU evidence for NUMA partition admission and work balance.

use serde_json::json;
use sonium_qa::{QaResult, host_hardware_provenance, write_qa_artifact};
use sonium_solvers::topology::MachineTopology;
use sonium_solvers::{CsrMatrix, NumaPartitionedCsr, NumaSpmvPolicy};
use std::time::Instant;

const CASE_ID: &str = "solvers.numa-partitioning-report.v1";

fn skewed_matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        7,
        7,
        vec![0, 1, 2, 3, 13, 14, 15, 16],
        (0..16).map(|index| index % 7).collect(),
        vec![1.0; 16],
    )
}

fn runtime_matrix() -> CsrMatrix<f64> {
    const DIM: usize = 4096;
    const NNZ_PER_ROW: usize = 16;
    let mut row_ptrs = Vec::with_capacity(DIM + 1);
    let mut col_indices = Vec::with_capacity(DIM * NNZ_PER_ROW);
    let mut values = Vec::with_capacity(DIM * NNZ_PER_ROW);
    row_ptrs.push(0);
    for row in 0..DIM {
        for offset in 0..NNZ_PER_ROW {
            col_indices.push((row * 13 + offset * 17) % DIM);
            values.push(0.5 + (offset % 5) as f64 * 0.125);
        }
        row_ptrs.push(col_indices.len());
    }
    CsrMatrix::from_raw_parts(DIM, DIM, row_ptrs, col_indices, values)
}

fn max_action_error(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn scalar_adjoint(matrix: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), matrix.num_rows);
    let mut y = vec![0.0; matrix.num_cols];
    for row in 0..matrix.num_rows {
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            y[matrix.col_indices[index]] += matrix.values[index] * x[row];
        }
    }
    y
}

fn timed_policy(
    numa: &NumaPartitionedCsr<f64>,
    x: &[f64],
    policy: NumaSpmvPolicy,
    iterations: usize,
) -> Result<(Vec<f64>, f64, f64), String> {
    let started = Instant::now();
    let mut last = Vec::new();
    let mut checksum = 0.0;
    for _ in 0..iterations {
        last = numa
            .try_matvec_with_policy(x, policy)
            .map_err(|error| format!("NUMA runtime SpMV failed: {error}"))?;
        checksum += last.iter().sum::<f64>();
    }
    let elapsed = started.elapsed().as_secs_f64();
    if !elapsed.is_finite() || elapsed <= 0.0 || !checksum.is_finite() {
        return Err("NUMA runtime measurement was non-finite".into());
    }
    Ok((last, elapsed, checksum))
}

fn run() -> Result<QaResult, String> {
    let matrix = skewed_matrix();
    let topology = MachineTopology::detect();
    let placement = topology.placement_plan(3);
    let numa = NumaPartitionedCsr::try_from_csr_with_placement(&matrix, &placement)
        .map_err(|error| format!("NUMA construction failed: {error}"))?;
    let report = numa.partitioning_report();
    let estimated_storage_bytes = NumaPartitionedCsr::estimate_storage_bytes(&matrix, 3)
        .map_err(|error| format!("NUMA memory estimate failed: {error}"))?;
    let budget_admitted =
        NumaPartitionedCsr::try_from_csr_with_memory_budget(&matrix, 3, estimated_storage_bytes)
            .is_ok();
    let budget_rejected = NumaPartitionedCsr::try_from_csr_with_memory_budget(
        &matrix,
        3,
        estimated_storage_bytes.saturating_sub(1),
    )
    .is_err();
    if !budget_admitted || !budget_rejected {
        return Err("NUMA memory admission did not enforce exact budget".into());
    }
    let replicated_estimate = NumaPartitionedCsr::estimate_storage_bytes_with_policy(
        &matrix,
        placement.workers.len(),
        NumaSpmvPolicy::REPLICATED_PINNED,
    )
    .map_err(|error| format!("NUMA replicated policy estimate failed: {error}"))?;
    let policy_budget_admitted =
        NumaPartitionedCsr::try_from_csr_with_placement_and_policy_memory_budget(
            &matrix,
            &placement,
            NumaSpmvPolicy::REPLICATED_PINNED,
            replicated_estimate.total_bytes,
        )
        .is_ok();
    let policy_budget_rejected =
        NumaPartitionedCsr::try_from_csr_with_placement_and_policy_memory_budget(
            &matrix,
            &placement,
            NumaSpmvPolicy::REPLICATED_PINNED,
            replicated_estimate.total_bytes.saturating_sub(1),
        )
        .is_err();
    if !policy_budget_admitted || !policy_budget_rejected {
        return Err("NUMA replicated policy admission did not enforce exact budget".into());
    }
    if report.requested_partitions != 3
        || report.admitted_partitions != 3
        || report.placement_workers != 3
        || report.num_rows != 7
        || report.total_nnz != 16
    {
        return Err(format!("unexpected NUMA admission report: {report:?}"));
    }
    if report.clamped_to_rows {
        return Err(format!("unexpected row-count clamp: {report:?}"));
    }
    if !report.row_imbalance.is_finite()
        || !report.nnz_imbalance.is_finite()
        || report.row_imbalance < 0.0
        || report.nnz_imbalance < 0.0
        || report.nnz_imbalance > 1.0
    {
        return Err(format!("invalid work-balance metrics: {report:?}"));
    }

    let clamped = NumaPartitionedCsr::try_from_csr(&matrix, 99)
        .map_err(|error| format!("NUMA clamped construction failed: {error}"))?
        .partitioning_report();
    if clamped.admitted_partitions != 7 || !clamped.clamped_to_rows {
        return Err(format!(
            "NUMA row-count clamp was not reported: {clamped:?}"
        ));
    }

    const RUNTIME_ITERATIONS: usize = 8;
    let runtime_csr = runtime_matrix();
    let runtime_numa = NumaPartitionedCsr::try_from_csr(&runtime_csr, 4)
        .map_err(|error| format!("NUMA runtime construction failed: {error}"))?;
    let runtime_x = vec![1.0_f64; runtime_csr.num_cols];
    let runtime_expected_adjoint = scalar_adjoint(&runtime_csr, &runtime_x);
    let (unpin_output, unpinned_seconds, unpinned_checksum) = timed_policy(
        &runtime_numa,
        &runtime_x,
        NumaSpmvPolicy::SHARED_UNPINNED,
        RUNTIME_ITERATIONS,
    )?;
    let (pinned_output, pinned_seconds, pinned_checksum) = timed_policy(
        &runtime_numa,
        &runtime_x,
        NumaSpmvPolicy::SHARED_PINNED,
        RUNTIME_ITERATIONS,
    )?;
    let (replicated_output, replicated_seconds, replicated_checksum) = timed_policy(
        &runtime_numa,
        &runtime_x,
        NumaSpmvPolicy::REPLICATED_PINNED,
        RUNTIME_ITERATIONS,
    )?;
    let pinned_action_error = max_action_error(&unpin_output, &pinned_output);
    let replicated_action_error = max_action_error(&unpin_output, &replicated_output);
    if !pinned_action_error.is_finite()
        || !replicated_action_error.is_finite()
        || pinned_action_error > 1e-10
        || replicated_action_error > 1e-10
    {
        return Err(format!(
            "NUMA runtime policy action mismatch: pinned={pinned_action_error:e}, replicated={replicated_action_error:e}"
        ));
    }
    let runtime_bytes = (runtime_csr.nnz()
        * (std::mem::size_of::<usize>() + std::mem::size_of::<f64>())
        + (runtime_csr.num_rows + 1) * std::mem::size_of::<usize>()
        + (runtime_csr.num_rows + runtime_csr.num_cols) * std::mem::size_of::<f64>())
        as f64;
    let bandwidth = |seconds: f64| runtime_bytes * RUNTIME_ITERATIONS as f64 / seconds / 1.0e9;
    let unpinned_bandwidth = bandwidth(unpinned_seconds);
    let pinned_bandwidth = bandwidth(pinned_seconds);
    let replicated_bandwidth = bandwidth(replicated_seconds);
    if !unpinned_bandwidth.is_finite()
        || !pinned_bandwidth.is_finite()
        || !replicated_bandwidth.is_finite()
        || unpinned_bandwidth <= 0.0
        || pinned_bandwidth <= 0.0
        || replicated_bandwidth <= 0.0
    {
        return Err("NUMA runtime bandwidth measurement was invalid".into());
    }
    let mut action_errors = serde_json::Map::new();
    for (name, policy) in [
        ("shared_unpinned", NumaSpmvPolicy::SHARED_UNPINNED),
        ("shared_pinned", NumaSpmvPolicy::SHARED_PINNED),
        ("replicated_pinned", NumaSpmvPolicy::REPLICATED_PINNED),
    ] {
        let mut max_error = 0.0_f64;
        for _ in 0..2 {
            let transpose = runtime_numa
                .try_matvec_transpose_with_policy(&runtime_x, policy)
                .map_err(|error| format!("NUMA transpose failed for {name}: {error}"))?;
            let hermitian = runtime_numa
                .try_matvec_hermitian_with_policy(&runtime_x, policy)
                .map_err(|error| format!("NUMA Hermitian action failed for {name}: {error}"))?;
            max_error = max_error.max(max_action_error(&runtime_expected_adjoint, &transpose));
            max_error = max_error.max(max_action_error(&runtime_expected_adjoint, &hermitian));
        }
        if !max_error.is_finite() || max_error > 1.0e-10 {
            return Err(format!(
                "NUMA adjoint policy mismatch for {name}: {max_error:e}"
            ));
        }
        action_errors.insert(
            name.to_string(),
            json!({
                "forward": "passed",
                "transpose": "passed",
                "hermitian": "passed",
                "max_abs_error": max_error,
                "repetitions": 2,
            }),
        );
    }

    let runtime_report = runtime_numa.partitioning_report();
    let runtime_stats = runtime_numa.partition_stats();
    let artifact = json!({
        "schema_version": 1,
        "case_id": CASE_ID,
        "evidence_scope": if report.placement_available {
            "multi_socket_placement_candidate"
        } else {
            "functional_policy_parity"
        },
        "topology": {
            "summary": topology.short_summary(),
            "logical_cpus": topology.total_logical_cpus,
            "sockets": topology.sockets.iter().map(|socket| json!({
                "id": socket.id,
                "logical_cpus": socket.logical_cpus,
                "numa_node_ids": socket.numa_node_ids,
            })).collect::<Vec<_>>(),
            "numa_nodes": topology.numa_nodes.iter().map(|node| json!({
                "id": node.id,
                "logical_cpus": node.logical_cpus,
                "memory_bytes": node.memory_bytes,
            })).collect::<Vec<_>>(),
            "notes": topology.notes.clone(),
        },
        "admission": {
            "requested_partitions": report.requested_partitions,
            "admitted_partitions": report.admitted_partitions,
            "estimated_storage_bytes": estimated_storage_bytes,
            "replicated_policy_storage_bytes": replicated_estimate.total_bytes,
            "memory_budget_admitted": budget_admitted,
            "memory_budget_rejected": budget_rejected,
            "policy_memory_budget_admitted": policy_budget_admitted,
            "policy_memory_budget_rejected": policy_budget_rejected,
        },
        "partitioning": {
            "placement_status": report.placement_status.as_str(),
            "placement_available": report.placement_available,
            "row_imbalance": report.row_imbalance,
            "nnz_imbalance": report.nnz_imbalance,
            "stats": numa.partition_stats().iter().map(|stat| json!({
                "row_start": stat.row_start,
                "num_rows": stat.num_rows,
                "nnz": stat.nnz,
                "numa_node_id": stat.numa_node_id,
                "socket_id": stat.socket_id,
                "logical_cpu_count": stat.logical_cpu_count,
                "first_touch_affinity": stat.first_touch_affinity.to_string(),
                "first_touch_affinity_applied": stat.first_touch_affinity_applied,
            })).collect::<Vec<_>>(),
        },
        "runtime": {
            "matrix": {"rows": runtime_csr.num_rows, "cols": runtime_csr.num_cols, "nnz": runtime_csr.nnz()},
            "requested_partitions": runtime_report.requested_partitions,
            "admitted_partitions": runtime_report.admitted_partitions,
            "placement_status": runtime_report.placement_status.as_str(),
            "iterations": RUNTIME_ITERATIONS,
            "effective_gb_per_second": {
                "shared_unpinned": unpinned_bandwidth,
                "shared_pinned": pinned_bandwidth,
                "replicated_pinned": replicated_bandwidth,
            },
            "elapsed_seconds": {
                "shared_unpinned": unpinned_seconds,
                "shared_pinned": pinned_seconds,
                "replicated_pinned": replicated_seconds,
            },
            "action_parity": action_errors,
            "locality_signal": "unavailable_no_per_process_local_remote_counter",
            "partition_stats": runtime_stats.iter().map(|stat| json!({
                "row_start": stat.row_start,
                "num_rows": stat.num_rows,
                "nnz": stat.nnz,
                "numa_node_id": stat.numa_node_id,
                "socket_id": stat.socket_id,
                "logical_cpu_count": stat.logical_cpu_count,
                "first_touch_affinity": stat.first_touch_affinity.to_string(),
                "first_touch_affinity_applied": stat.first_touch_affinity_applied,
            })).collect::<Vec<_>>(),
        },
    });
    let artifact_bytes = serde_json::to_vec_pretty(&artifact)
        .map_err(|error| format!("serialize NUMA evidence artifact: {error}"))?;
    let mut result = QaResult::passed(CASE_ID)
        .metric("requested_partitions", report.requested_partitions as f64)
        .metric("admitted_partitions", report.admitted_partitions as f64)
        .metric("total_nnz", report.total_nnz as f64)
        .metric("row_imbalance", report.row_imbalance)
        .metric("nnz_imbalance", report.nnz_imbalance)
        .metric("placement_available", f64::from(report.placement_available))
        .metric("estimated_storage_bytes", estimated_storage_bytes as f64)
        .metric("memory_budget_admitted", f64::from(budget_admitted))
        .metric("memory_budget_rejected", f64::from(budget_rejected))
        .metric(
            "policy_memory_budget_admitted",
            f64::from(policy_budget_admitted),
        )
        .metric(
            "policy_memory_budget_rejected",
            f64::from(policy_budget_rejected),
        )
        .metric("runtime_rows", runtime_csr.num_rows as f64)
        .metric("runtime_nnz", runtime_csr.nnz() as f64)
        .metric("runtime_iterations", RUNTIME_ITERATIONS as f64)
        .metric("runtime_shared_unpinned_seconds", unpinned_seconds)
        .metric("runtime_shared_pinned_seconds", pinned_seconds)
        .metric("runtime_replicated_pinned_seconds", replicated_seconds)
        .metric("runtime_shared_unpinned_gb_per_second", unpinned_bandwidth)
        .metric("runtime_shared_pinned_gb_per_second", pinned_bandwidth)
        .metric(
            "runtime_replicated_pinned_gb_per_second",
            replicated_bandwidth,
        )
        .metric("runtime_shared_pinned_action_error", pinned_action_error)
        .metric(
            "runtime_replicated_pinned_action_error",
            replicated_action_error,
        )
        .metric("runtime_checksum", unpinned_checksum)
        .metric("runtime_pinned_checksum", pinned_checksum)
        .metric("runtime_replicated_checksum", replicated_checksum)
        .metric(
            "clamped_admitted_partitions",
            clamped.admitted_partitions as f64,
        );
    if std::env::var_os("SONIUM_OUTPUT_DIR").is_some() {
        let (path, checksum) =
            write_qa_artifact("solvers_numa_partitioning_report.json", &artifact_bytes)?;
        result = result.artifact_with_provenance(
            path,
            "crates/sonium-qa/bin/solvers_numa_partitioning_report.rs",
            "runtime NUMA topology, admission, placement, parity, and bandwidth report",
            "v1",
            checksum,
        );
    }
    let mut hardware = host_hardware_provenance();
    hardware.topology = Some(topology.short_summary());
    hardware.numa_policy = Some("shared_unpinned/shared_pinned/replicated_pinned".into());
    Ok(result.hardware_provenance(hardware))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(result) => {
            let result = result.metric("elapsed_seconds", started.elapsed().as_secs_f64());
            println!("QA_RESULT: {}", serde_json::to_string(&result).unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error);
            println!("QA_RESULT: {}", serde_json::to_string(&result).unwrap());
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
