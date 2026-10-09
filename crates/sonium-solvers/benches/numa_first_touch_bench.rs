//! NUMA first-touch and input-vector placement experiments.
//!
//! This bench is primarily for two-socket Linux machines. It builds a
//! topology-owned partitioned CSR matrix, records whether affinity was applied
//! before first-touch allocation, and compares shared vs per-partition `x`
//! vector placement. On non-Linux hosts the affinity fields report unsupported
//! and the benchmark still provides a portable baseline.

#[path = "matrices/mod.rs"]
mod matrices;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use sonium_solvers::{MachineTopology, NumaSpmvPolicy};
use std::process::Command;
use std::time::Duration;

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

#[cfg(feature = "cli")]
fn partition_notes(numa: &sonium_solvers::NumaPartitionedCsr<f64>) -> String {
    numa.partition_stats()
        .iter()
        .map(|stats| {
            format!(
                "rows={}..{} nnz={} node={:?} socket={:?} cpus={} first_touch_affinity={}",
                stats.row_start,
                stats.row_start + stats.num_rows,
                stats.nnz,
                stats.numa_node_id,
                stats.socket_id,
                stats.logical_cpu_count,
                stats.first_touch_affinity_applied
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

#[cfg(feature = "cli")]
fn locality_snapshot() -> String {
    if cfg!(target_os = "linux") {
        let pid = std::process::id().to_string();
        match Command::new("numastat").args(["-p", &pid]).output() {
            Ok(output) if output.status.success() => {
                let text = String::from_utf8_lossy(&output.stdout);
                let one_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
                format!("numastat -p {pid}: {one_line}")
            }
            Ok(output) => format!(
                "numastat -p {pid} failed with status {}",
                output.status.code().unwrap_or(-1)
            ),
            Err(err) => format!("numastat unavailable: {err}"),
        }
    } else {
        "numastat unavailable on non-Linux host".to_string()
    }
}

#[cfg(feature = "cli")]
struct NumaPerfMeta<'a> {
    id: &'a str,
    operation: &'a str,
    nrows: usize,
    nnz: usize,
    bytes_per_iter: u64,
    setup_ms: Option<f64>,
    partitioning_report: sonium_solvers::NumaPartitioningReport,
    partition_stats: Vec<sonium_solvers::PartitionStats>,
    evidence_scope: &'a str,
    notes: String,
}

#[cfg(feature = "cli")]
fn record_numa_perf<F>(meta: NumaPerfMeta<'_>, mut apply: F)
where
    F: FnMut(),
{
    if perf::perf_json_path().is_none() {
        return;
    }

    let sample = perf::measure_for(Duration::from_millis(100), 100, || {
        apply();
    });

    let mut run = PerfRun::new("numa_first_touch_bench", "numa/first_touch", meta.id)
        .with_bytes(meta.bytes_per_iter)
        .with_timing(sample);
    run.operation = meta.operation.to_string();
    run.matrix = Some("poisson_2d".to_string());
    run.scalar = "f64".to_string();
    run.nrows = Some(meta.nrows);
    run.ncols = Some(meta.nrows);
    run.nnz = Some(meta.nnz);
    run.setup_ms = meta.setup_ms;
    run = run.with_numa_partitioning(&meta.partitioning_report, &meta.partition_stats);
    run.numa_evidence_scope = Some(meta.evidence_scope.to_string());
    run.numa_locality_status = Some(
        if cfg!(target_os = "linux") {
            "snapshot_if_available"
        } else {
            "unavailable_non_linux"
        }
        .to_string(),
    );
    run.notes = Some(meta.notes);

    if let Err(err) = perf::append_run_from_env(run) {
        eprintln!("failed to write {} report: {err}", perf::PERF_JSON_ENV);
    }
}

fn bench_numa_first_touch(c: &mut Criterion) {
    let mut group = c.benchmark_group("numa/first_touch");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(5));

    let topology = MachineTopology::detect();
    let partitions = topology.numa_nodes.len().max(1);
    let placement = topology.placement_plan(partitions);

    let n = matrices::grid_size_2d(250_000);
    let csr = matrices::poisson_2d(n);
    let rows = csr.num_rows;
    let bytes = spmv_bytes(rows, rows, csr.nnz());
    let (numa, setup_ms) = {
        #[cfg(feature = "cli")]
        {
            perf::time_once(|| csr.to_numa_partitioned_with_placement(&placement))
        }
        #[cfg(not(feature = "cli"))]
        {
            (csr.to_numa_partitioned_with_placement(&placement), 0.0_f64)
        }
    };
    let x = vec![1.0_f64; rows];
    let partitioning_report = numa.partitioning_report();
    let partition_stats = numa.partition_stats();
    let evidence_scope = if topology.sockets.len() >= 2 && topology.numa_nodes.len() >= 2 {
        "multi_socket_numastat"
    } else {
        "functional_policy_parity"
    };

    #[cfg(feature = "cli")]
    {
        let base_notes = format!(
            "topology={}; placement={}; partitions=[{}]; locality={}",
            topology.short_summary(),
            placement.short_summary(),
            partition_notes(&numa),
            locality_snapshot()
        );
        record_numa_perf(
            NumaPerfMeta {
                id: "pinned_shared_x",
                operation: "spmv",
                nrows: rows,
                nnz: numa.nnz(),
                bytes_per_iter: bytes,
                setup_ms: Some(setup_ms),
                partitioning_report,
                partition_stats: partition_stats.clone(),
                evidence_scope,
                notes: format!("{base_notes}; input_vector=shared"),
            },
            || {
                std::hint::black_box(numa.matvec_with_policy(&x, NumaSpmvPolicy::SHARED_PINNED));
            },
        );
        record_numa_perf(
            NumaPerfMeta {
                id: "pinned_replicated_x",
                operation: "spmv",
                nrows: rows,
                nnz: numa.nnz(),
                bytes_per_iter: bytes,
                setup_ms: Some(setup_ms),
                partitioning_report,
                partition_stats: partition_stats.clone(),
                evidence_scope,
                notes: format!(
                    "{base_notes}; input_vector=replicated_per_partition; extra_x_bytes={}",
                    rows * std::mem::size_of::<f64>() * numa.num_partitions()
                ),
            },
            || {
                std::hint::black_box(
                    numa.matvec_with_policy(&x, NumaSpmvPolicy::REPLICATED_PINNED),
                );
            },
        );
    }

    group.throughput(Throughput::Bytes(bytes));
    group.bench_function(BenchmarkId::from_parameter("pinned_shared_x"), |b| {
        b.iter(|| std::hint::black_box(numa.matvec_with_policy(&x, NumaSpmvPolicy::SHARED_PINNED)))
    });
    group.bench_function(BenchmarkId::from_parameter("pinned_replicated_x"), |b| {
        b.iter(|| {
            std::hint::black_box(numa.matvec_with_policy(&x, NumaSpmvPolicy::REPLICATED_PINNED))
        })
    });

    group.finish();
}

criterion_group!(
    name = numa_first_touch;
    config = Criterion::default();
    targets = bench_numa_first_touch
);
criterion_main!(numa_first_touch);
