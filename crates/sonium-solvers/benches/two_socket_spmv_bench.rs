//! Two-socket SpMV locality benchmark.
//!
//! This target is meant for Linux servers with at least two NUMA nodes. It
//! still runs on laptops and single-socket hosts, but the perf JSON notes will
//! say that local/remote counters are unavailable or not meaningful there.

#[path = "matrices/mod.rs"]
mod matrices;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use sonium_solvers::{
    CsrMatrix, MachineTopology, NumaPartitioningReport, NumaSpmvPolicy, NumaStorageEstimate,
    PartitionStats, selected_spmv_kernel,
};
use std::process::Command;
use std::time::{Duration, Instant};

#[cfg(feature = "cli")]
use sonium_solvers::perf::{self, PerfRun};

fn spmv_bytes(nrows: usize, ncols: usize, nnz: usize) -> u64 {
    #[cfg(feature = "cli")]
    {
        perf::bytes_real_csr_spmv(nrows, ncols, nnz, std::mem::size_of::<f64>())
    }
    #[cfg(not(feature = "cli"))]
    {
        (8 * nnz + 8 * nnz + 8 * (nrows + 1) + 8 * ncols + 8 * nrows) as u64
    }
}

#[derive(Debug, Clone, Copy)]
struct NumaCounterSnapshot {
    local_node_mb: f64,
    other_node_mb: f64,
}

impl NumaCounterSnapshot {
    fn delta(self, before: Self) -> Self {
        Self {
            local_node_mb: self.local_node_mb - before.local_node_mb,
            other_node_mb: self.other_node_mb - before.other_node_mb,
        }
    }

    fn remote_fraction(self) -> Option<f64> {
        let total = self.local_node_mb + self.other_node_mb;
        (total > 0.0).then_some(self.other_node_mb / total)
    }
}

fn numa_counter_snapshot() -> Option<NumaCounterSnapshot> {
    if !cfg!(target_os = "linux") {
        return None;
    }

    let output = Command::new("numastat").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);

    fn row_sum(text: &str, name: &str) -> Option<f64> {
        text.lines().find_map(|line| {
            let mut parts = line.split_whitespace();
            (parts.next()? == name)
                .then(|| parts.filter_map(|value| value.parse::<f64>().ok()).sum())
        })
    }

    Some(NumaCounterSnapshot {
        local_node_mb: row_sum(&text, "local_node")?,
        other_node_mb: row_sum(&text, "other_node")?,
    })
}

#[cfg(feature = "cli")]
struct TwoSocketPerfMeta<'a> {
    id: &'a str,
    matrix_name: String,
    nrows: usize,
    ncols: usize,
    nnz: usize,
    bytes_per_iter: u64,
    setup_ms: f64,
    correctness_max_abs: f64,
    correctness_tolerance: f64,
    action_parity: &'a str,
    thread_affinity: String,
    numa_policy: &'a str,
    storage: NumaStorageEstimate,
    partitioning_report: NumaPartitioningReport,
    partition_stats: Vec<PartitionStats>,
    evidence_scope: &'a str,
    notes: String,
}

#[cfg(feature = "cli")]
fn record_two_socket_perf<F>(meta: TwoSocketPerfMeta<'_>, mut apply: F)
where
    F: FnMut(),
{
    if perf::perf_json_path().is_none() {
        return;
    }

    let before = numa_counter_snapshot();
    let sample = perf::measure_for(Duration::from_millis(250), 100, || {
        apply();
    });
    let after = numa_counter_snapshot();

    let counter_delta = before.zip(after).map(|(before, after)| after.delta(before));
    let counter_note = match counter_delta {
        Some(delta) => {
            format!(
                "numastat_delta_mb local_node={:.3} other_node={:.3} remote_fraction={:.4}",
                delta.local_node_mb,
                delta.other_node_mb,
                delta.remote_fraction().unwrap_or(0.0)
            )
        }
        None => "numastat local_node/other_node counters unavailable".to_string(),
    };

    let mut run = PerfRun::new("two_socket_spmv_bench", "spmv/two_socket", meta.id)
        .with_bytes(meta.bytes_per_iter)
        .with_timing(sample);
    if let Some(delta) = counter_delta {
        run.numa_local_node_delta_mb = Some(delta.local_node_mb);
        run.numa_other_node_delta_mb = Some(delta.other_node_mb);
        run.numa_local_node_gb_s = Some(mb_to_gb_s(delta.local_node_mb, sample.total_ms));
        run.numa_other_node_gb_s = Some(mb_to_gb_s(delta.other_node_mb, sample.total_ms));
        run.numa_remote_fraction = delta.remote_fraction();
    }
    run.bandwidth_limited_by = Some("numa_spmv_memory_traffic".to_string());
    run.operation = "spmv".to_string();
    run.matrix = Some(meta.matrix_name);
    run.scalar = "f64".to_string();
    run.selected_kernel = Some(selected_spmv_kernel::<f64>().name().to_string());
    run.nrows = Some(meta.nrows);
    run.ncols = Some(meta.ncols);
    run.nnz = Some(meta.nnz);
    run.setup_ms = Some(meta.setup_ms);
    run.correctness_status = Some("passed".to_string());
    run.correctness_max_abs = Some(meta.correctness_max_abs);
    run.correctness_tolerance = Some(meta.correctness_tolerance);
    run.action_parity = Some(meta.action_parity.to_string());
    run.thread_affinity = Some(meta.thread_affinity);
    run.numa_policy = Some(meta.numa_policy.to_string());
    run.numa_memory_estimate_bytes = u64::try_from(meta.storage.total_bytes).ok();
    run.numa_partitioned_csr_bytes = u64::try_from(meta.storage.partitioned_csr_bytes).ok();
    run.numa_input_vector_bytes = u64::try_from(meta.storage.input_vector_bytes).ok();
    run.numa_replicated_input_bytes = u64::try_from(meta.storage.replicated_input_bytes).ok();
    run = run.with_numa_partitioning(&meta.partitioning_report, &meta.partition_stats);
    run.numa_evidence_scope = Some(meta.evidence_scope.to_string());
    run.numa_locality_status = Some(
        if counter_delta.is_some() {
            "observed"
        } else {
            "unavailable"
        }
        .to_string(),
    );
    run.notes = Some(format!("{}; {}", meta.notes, counter_note));

    if let Err(err) = perf::append_run_from_env(run) {
        eprintln!("failed to write {} report: {err}", perf::PERF_JSON_ENV);
    }
}

fn scalar_csr_matvec(matrix: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), matrix.num_cols);
    (0..matrix.num_rows)
        .map(|row| {
            let mut sum = 0.0;
            for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                sum += matrix.values[index] * x[matrix.col_indices[index]];
            }
            sum
        })
        .collect()
}

fn scalar_csr_adjoint_matvec(matrix: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), matrix.num_rows);
    let mut y = vec![0.0; matrix.num_cols];
    for (row, &x_value) in x.iter().enumerate().take(matrix.num_rows) {
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            y[matrix.col_indices[index]] += matrix.values[index] * x_value;
        }
    }
    y
}

fn max_abs_error(expected: &[f64], actual: &[f64]) -> f64 {
    assert_eq!(expected.len(), actual.len());
    expected
        .iter()
        .zip(actual)
        .map(|(expected, actual)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn assert_numa_actions(
    matrix: &CsrMatrix<f64>,
    numa: &sonium_solvers::NumaPartitionedCsr<f64>,
    x: &[f64],
    policy: NumaSpmvPolicy,
    tolerance: f64,
) -> f64 {
    let expected_forward = scalar_csr_matvec(matrix, x);
    let expected_adjoint = scalar_csr_adjoint_matvec(matrix, x);
    let mut error = 0.0;
    for _ in 0..2 {
        let actual_forward = numa
            .try_matvec_with_policy(x, policy)
            .expect("checked NUMA forward action");
        let actual_transpose = numa
            .try_matvec_transpose(x)
            .expect("checked NUMA transpose action");
        let actual_hermitian = numa
            .try_matvec_hermitian(x)
            .expect("checked NUMA Hermitian action");
        error = [
            error,
            max_abs_error(&expected_forward, &actual_forward),
            max_abs_error(&expected_adjoint, &actual_transpose),
            max_abs_error(&expected_adjoint, &actual_hermitian),
        ]
        .into_iter()
        .fold(0.0, f64::max);
    }
    assert!(
        error.is_finite() && error <= tolerance,
        "NUMA action gate failed for policy {:?}: error={error:e} tolerance={tolerance:e}",
        policy
    );
    error
}

fn skewed_matrix(rows: usize) -> CsrMatrix<f64> {
    let mut triplets = Vec::new();
    for row in 0..rows {
        let row_nnz = if row % 32 == 0 { 64 } else { 2 };
        for offset in 0..row_nnz {
            let column = (row * 17 + offset * 31) % rows;
            let value = if row == column {
                row_nnz as f64 + 1.0
            } else {
                -0.125
            };
            triplets.push((row, column, value));
        }
    }
    CsrMatrix::from_triplets(rows, rows, triplets)
}

fn partition_counts(topology: &MachineTopology, rows: usize) -> Vec<usize> {
    let nodes = topology.numa_nodes.len().max(1);
    [1, nodes, nodes.saturating_mul(2)]
        .into_iter()
        .map(|count| count.min(rows.max(1)))
        .fold(Vec::new(), |mut counts, count| {
            if !counts.contains(&count) {
                counts.push(count);
            }
            counts
        })
}

fn partition_evidence(
    topology: &MachineTopology,
    placement: &sonium_solvers::topology::PlacementPlan,
    numa: &sonium_solvers::NumaPartitionedCsr<f64>,
) -> String {
    let stats = numa.partition_stats();
    let balance = numa.partitioning_report();
    let mut next_row = 0;
    let mut contiguous = true;
    for stat in &stats {
        contiguous &= stat.row_start == next_row;
        next_row = stat.row_start + stat.num_rows;
    }
    let topology_matches_plan =
        placement.workers.iter().all(|worker| {
            topology.numa_nodes.iter().any(|node| {
                node.id == worker.numa_node_id && node.logical_cpus == worker.logical_cpus
            }) && topology.sockets.iter().any(|socket| {
                socket.id == worker.socket_id
                    && socket.numa_node_ids.contains(&worker.numa_node_id)
                    && worker
                        .logical_cpus
                        .iter()
                        .all(|cpu| socket.logical_cpus.contains(cpu))
            })
        });
    let partitions_match_plan = stats.len() == placement.workers.len()
        && stats.iter().enumerate().all(|(index, stat)| {
            placement.worker_for_index(index).is_some_and(|worker| {
                stat.numa_node_id == Some(worker.numa_node_id)
                    && stat.socket_id == Some(worker.socket_id)
                    && stat.logical_cpu_count == worker.logical_cpus.len()
            })
        });
    let placement_complete = topology.sockets.len() > 1
        && topology.numa_nodes.len() > 1
        && topology_matches_plan
        && partitions_match_plan
        && stats.iter().all(|stat| {
            stat.numa_node_id.is_some()
                && stat.socket_id.is_some()
                && stat.logical_cpu_count > 0
                && stat.first_touch_affinity_applied
        });
    let placement_status = if placement_complete {
        "affinity_verified"
    } else if balance.placement_status.as_str() == "unavailable" {
        "unavailable"
    } else {
        "affinity_unavailable"
    };
    let stats_note = stats
        .iter()
        .map(|stat| {
            format!(
                "rows={}..{} nnz={} node={:?} socket={:?} cpus={} first_touch_affinity={}",
                stat.row_start,
                stat.row_start + stat.num_rows,
                stat.nnz,
                stat.numa_node_id,
                stat.socket_id,
                stat.logical_cpu_count,
                stat.first_touch_affinity_applied
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        "placement_status={placement_status}; contiguous_nnz_balanced_ranges={contiguous}; topology_matches_plan={topology_matches_plan}; partitions_match_plan={partitions_match_plan}; row_imbalance_max_over_mean={:.6}; nnz_imbalance_max_over_mean={:.6}; topology={}; placement={}; partitions=[{stats_note}]",
        balance.row_imbalance,
        balance.nnz_imbalance,
        topology.short_summary(),
        placement.short_summary()
    )
}

#[cfg(feature = "cli")]
fn mb_to_gb_s(mb: f64, total_ms: f64) -> f64 {
    if total_ms <= 0.0 {
        return 0.0;
    }
    (mb * 1.0e6) / (total_ms / 1.0e3) / 1.0e9
}

fn evidence_gate(topology: &MachineTopology) -> &'static str {
    if !cfg!(target_os = "linux") {
        "non_linux_numastat_unavailable"
    } else if topology.sockets.len() >= 2 && topology.numa_nodes.len() >= 2 {
        "two_socket_local_remote_counters"
    } else {
        "single_socket_or_single_numa_node"
    }
}

fn bench_two_socket_spmv(c: &mut Criterion) {
    let topology = MachineTopology::detect();
    let mut group = c.benchmark_group("spmv/two_socket");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(5));

    let fixtures = [
        (
            "small_poisson",
            matrices::poisson_2d(matrices::grid_size_2d(10_000)),
        ),
        (
            "large_poisson",
            matrices::poisson_2d(matrices::grid_size_2d(500_000)),
        ),
        ("skewed_rows", skewed_matrix(100_000)),
    ];
    let policies = [
        ("shared_unpinned", NumaSpmvPolicy::SHARED_UNPINNED),
        ("shared_pinned", NumaSpmvPolicy::SHARED_PINNED),
        ("replicated_pinned", NumaSpmvPolicy::REPLICATED_PINNED),
    ];

    for (fixture_name, csr) in fixtures {
        let rows = csr.num_rows;
        let base_bytes = spmv_bytes(rows, csr.num_cols, csr.nnz());

        for partitions in partition_counts(&topology, rows) {
            let placement = topology.placement_plan(partitions);
            // This estimate is deliberately computed before construction. The
            // checked constructor repeats it before any first-touch worker is
            // spawned, making the admission ordering observable in the runner.
            let admission_estimate =
                sonium_solvers::NumaPartitionedCsr::<f64>::estimate_storage_bytes_with_policy(
                    &csr,
                    partitions,
                    NumaSpmvPolicy::REPLICATED_PINNED,
                )
                .expect("checked NUMA policy storage estimate");
            let admission_budget = admission_estimate.total_bytes;
            let setup_start = Instant::now();
            let numa =
                sonium_solvers::NumaPartitionedCsr::try_from_csr_with_placement_and_policy_memory_budget(
                    &csr,
                    &placement,
                    NumaSpmvPolicy::REPLICATED_PINNED,
                    admission_budget,
                )
                .expect("exact NUMA placement budget should admit");
            let setup_ms = setup_start.elapsed().as_secs_f64() * 1.0e3;
            let placement_note = partition_evidence(&topology, &placement, &numa);
            let partitioning_report = numa.partitioning_report();
            let partition_stats = numa.partition_stats();
            let evidence_scope = if topology.sockets.len() >= 2 && topology.numa_nodes.len() >= 2 {
                "multi_socket_numastat"
            } else {
                "functional_policy_parity"
            };
            let action_tolerance = 1.0e-10;
            let x = vec![1.0_f64; csr.num_cols];

            for (policy_name, policy) in policies {
                let error = assert_numa_actions(&csr, &numa, &x, policy, action_tolerance);
                let storage = numa
                    .storage_estimate(policy)
                    .expect("checked NUMA policy storage estimate");
                let bytes = base_bytes
                    .checked_add(
                        u64::try_from(storage.replicated_input_bytes)
                            .expect("replicated input byte count fits u64"),
                    )
                    .expect("SpMV byte model overflow");
                group.throughput(Throughput::Bytes(bytes));
                let id = format!("{fixture_name}/p{partitions}/{policy_name}");
                let notes = format!(
                    "evidence_gate={}; matrix={fixture_name}; requested_partitions={partitions}; admitted_partitions={}; action_repetitions=2; {placement_note}; admission_budget_bytes={admission_budget}; policy_storage_bytes={}; replicated_input_accounted=true; vendor_baseline=unavailable_no_pinned_vendor_sparse_library",
                    evidence_gate(&topology),
                    numa.num_partitions(),
                    storage.total_bytes
                );
                #[cfg(feature = "cli")]
                record_two_socket_perf(
                    TwoSocketPerfMeta {
                        id: &id,
                        matrix_name: fixture_name.to_string(),
                        nrows: rows,
                        ncols: csr.num_cols,
                        nnz: numa.nnz(),
                        bytes_per_iter: bytes,
                        setup_ms,
                        correctness_max_abs: error,
                        correctness_tolerance: action_tolerance,
                        action_parity: "forward=passed;transpose=passed;hermitian=passed",
                        thread_affinity: format!("policy={policy_name}; {placement_note}"),
                        numa_policy: policy_name,
                        storage,
                        partitioning_report,
                        partition_stats: partition_stats.clone(),
                        evidence_scope,
                        notes,
                    },
                    || {
                        std::hint::black_box(numa.matvec_with_policy(&x, policy));
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new(&id, rows),
                    &(&numa, &x),
                    |b, &(matrix, x)| {
                        b.iter(|| matrix.matvec_with_policy(x, policy));
                    },
                );
            }
        }
    }

    group.finish();
}

criterion_group!(
    name = two_socket_spmv;
    config = Criterion::default();
    targets = bench_two_socket_spmv
);
criterion_main!(two_socket_spmv);
