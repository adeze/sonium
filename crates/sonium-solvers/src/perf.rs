//! Performance reporting helpers for solver and sparse-kernel benchmarks.
//!
//! Set `SONIUM_SOLVERS_PERF_JSON=/path/to/report.json` while running benches
//! to collect a machine-readable report. The helpers are intentionally small:
//! Criterion remains the source of stable timing statistics, while this module
//! records roofline-oriented metadata and single-shot timing samples.

use crate::KrylovOpCounts;
use crate::runtime;
use crate::sparse::detect_numa_partitions;
use crate::topology::MachineTopology;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const PERF_JSON_ENV: &str = "SONIUM_SOLVERS_PERF_JSON";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfReport {
    pub schema_version: u32,
    pub generated_at_unix_ms: u128,
    pub crate_name: String,
    pub crate_version: String,
    pub git_sha: Option<String>,
    pub hardware: HardwareInfo,
    pub runs: Vec<PerfRun>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub os: String,
    pub arch: String,
    pub cpu_model: String,
    #[serde(default)]
    pub compiler: Option<String>,
    #[serde(default)]
    pub target_features: String,
    #[serde(default)]
    pub microcode: Option<String>,
    pub total_threads: usize,
    pub rayon_threads: usize,
    pub runtime_compute_threads: usize,
    pub runtime_bandwidth_threads: usize,
    pub runtime_total_cores: usize,
    pub runtime_perf_cores: usize,
    pub numa_partitions: usize,
    pub socket_count: usize,
    pub numa_node_count: usize,
    pub topology_summary: String,
    pub topology_notes: Vec<String>,
    pub numa_nodes: Vec<NumaNodeInfo>,
    /// CPU sets grouped by physical socket.  Empty on platforms where socket
    /// topology is not exposed; this is intentionally separate from the
    /// NUMA-node list so a UMA run is unambiguous in artifacts.
    #[serde(default)]
    pub socket_cpu_sets: Vec<SocketInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaNodeInfo {
    pub id: usize,
    pub logical_cpus: Vec<usize>,
    pub memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocketInfo {
    pub id: usize,
    pub logical_cpus: Vec<usize>,
    pub numa_node_ids: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaPartitionPerfStat {
    pub row_start: usize,
    pub num_rows: usize,
    pub nnz: usize,
    pub numa_node_id: Option<usize>,
    pub socket_id: Option<usize>,
    pub logical_cpu_count: usize,
    pub first_touch_affinity: String,
    pub first_touch_affinity_applied: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfRun {
    pub bench: String,
    pub group: String,
    pub id: String,
    pub operation: String,
    pub matrix: Option<String>,
    pub scalar: String,
    #[serde(default)]
    pub selected_kernel: Option<String>,
    pub nrows: Option<usize>,
    pub ncols: Option<usize>,
    pub nnz: Option<usize>,
    pub bytes_per_iter: Option<u64>,
    pub csr_spmv_byte_model: Option<CsrSpmvByteModel>,
    #[serde(default)]
    pub correctness_status: Option<String>,
    #[serde(default)]
    pub correctness_max_abs: Option<f64>,
    #[serde(default)]
    pub correctness_tolerance: Option<f64>,
    #[serde(default)]
    pub action_parity: Option<String>,
    #[serde(default)]
    pub thread_affinity: Option<String>,
    #[serde(default)]
    pub numa_policy: Option<String>,
    #[serde(default)]
    pub numa_memory_estimate_bytes: Option<u64>,
    #[serde(default)]
    pub numa_partitioned_csr_bytes: Option<u64>,
    #[serde(default)]
    pub numa_input_vector_bytes: Option<u64>,
    #[serde(default)]
    pub numa_replicated_input_bytes: Option<u64>,
    #[serde(default)]
    pub numa_requested_partitions: Option<usize>,
    #[serde(default)]
    pub numa_admitted_partitions: Option<usize>,
    #[serde(default)]
    pub numa_placement_status: Option<String>,
    #[serde(default)]
    pub numa_evidence_scope: Option<String>,
    #[serde(default)]
    pub numa_locality_status: Option<String>,
    #[serde(default)]
    pub numa_row_imbalance: Option<f64>,
    #[serde(default)]
    pub numa_nnz_imbalance: Option<f64>,
    #[serde(default)]
    pub numa_partition_stats: Vec<NumaPartitionPerfStat>,
    pub measured_iterations: usize,
    pub setup_ms: Option<f64>,
    pub solve_ms: Option<f64>,
    #[serde(default = "default_rhs_count")]
    pub rhs_count: usize,
    #[serde(default)]
    pub mean_solve_ms_per_rhs: Option<f64>,
    #[serde(default)]
    pub mean_solver_iterations_per_rhs: Option<f64>,
    #[serde(default)]
    pub setup_reuse_scope: Option<String>,
    #[serde(default)]
    pub setup_reuse_break_even_rhs: Option<f64>,
    #[serde(default)]
    pub amortized_total_ms_per_rhs_4: Option<f64>,
    #[serde(default)]
    pub amortized_total_ms_per_rhs_16: Option<f64>,
    pub total_ms: f64,
    pub ms_per_iter: f64,
    pub gb_s: Option<f64>,
    pub solver_iterations: Option<usize>,
    pub residual: Option<f64>,
    pub converged: Option<bool>,
    #[serde(default)]
    pub krylov_spmv_count: Option<usize>,
    #[serde(default)]
    pub krylov_preconditioner_count: Option<usize>,
    #[serde(default)]
    pub krylov_dot_reductions: Option<usize>,
    #[serde(default)]
    pub krylov_norm_count: Option<usize>,
    #[serde(default)]
    pub krylov_axpy_passes: Option<usize>,
    #[serde(default)]
    pub krylov_basis_bytes_touched: Option<u64>,
    #[serde(default)]
    pub krylov_synchronization_points: Option<usize>,
    #[serde(default)]
    pub krylov_exact_residual_checks: Option<usize>,
    pub allocation_count: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub bandwidth_limited_by: Option<String>,
    pub numa_local_node_delta_mb: Option<f64>,
    pub numa_other_node_delta_mb: Option<f64>,
    #[serde(default)]
    pub numa_local_node_gb_s: Option<f64>,
    #[serde(default)]
    pub numa_other_node_gb_s: Option<f64>,
    #[serde(default)]
    pub numa_remote_fraction: Option<f64>,
    pub amg_profile: Option<String>,
    pub amg_num_levels: Option<usize>,
    pub amg_grid_complexity: Option<f64>,
    pub amg_operator_complexity: Option<f64>,
    #[serde(default)]
    pub amg_max_grid_complexity: Option<f64>,
    #[serde(default)]
    pub amg_max_operator_complexity: Option<f64>,
    #[serde(default)]
    pub amg_grid_complexity_within_target: Option<bool>,
    #[serde(default)]
    pub amg_operator_complexity_within_target: Option<bool>,
    pub amg_apply_count: Option<u64>,
    pub amg_total_apply_ms: Option<f64>,
    pub amg_mean_apply_ms: Option<f64>,
    pub amg_total_nnz: Option<usize>,
    pub amg_finest_nnz: Option<usize>,
    pub amg_coarsest_dofs: Option<usize>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsrSpmvByteModel {
    pub value_bytes: usize,
    pub col_index_bytes: usize,
    pub row_ptr_index_bytes: usize,
    pub matrix_value_bytes: u64,
    pub matrix_col_index_bytes: u64,
    pub matrix_row_ptr_bytes: u64,
    pub input_vector_bytes: u64,
    pub output_vector_bytes: u64,
    pub total_bytes: u64,
    pub bytes_per_nnz: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct TimedSample {
    pub iterations: usize,
    pub total_ms: f64,
    pub ms_per_iter: f64,
}

impl PerfRun {
    pub fn new(bench: impl Into<String>, group: impl Into<String>, id: impl Into<String>) -> Self {
        Self {
            bench: bench.into(),
            group: group.into(),
            id: id.into(),
            operation: String::new(),
            matrix: None,
            scalar: String::new(),
            selected_kernel: None,
            nrows: None,
            ncols: None,
            nnz: None,
            bytes_per_iter: None,
            csr_spmv_byte_model: None,
            correctness_status: None,
            correctness_max_abs: None,
            correctness_tolerance: None,
            action_parity: None,
            thread_affinity: None,
            numa_policy: None,
            numa_memory_estimate_bytes: None,
            numa_partitioned_csr_bytes: None,
            numa_input_vector_bytes: None,
            numa_replicated_input_bytes: None,
            numa_requested_partitions: None,
            numa_admitted_partitions: None,
            numa_placement_status: None,
            numa_evidence_scope: None,
            numa_locality_status: None,
            numa_row_imbalance: None,
            numa_nnz_imbalance: None,
            numa_partition_stats: Vec::new(),
            measured_iterations: 1,
            setup_ms: None,
            solve_ms: None,
            rhs_count: 1,
            mean_solve_ms_per_rhs: None,
            mean_solver_iterations_per_rhs: None,
            setup_reuse_scope: None,
            setup_reuse_break_even_rhs: None,
            amortized_total_ms_per_rhs_4: None,
            amortized_total_ms_per_rhs_16: None,
            total_ms: 0.0,
            ms_per_iter: 0.0,
            gb_s: None,
            solver_iterations: None,
            residual: None,
            converged: None,
            krylov_spmv_count: None,
            krylov_preconditioner_count: None,
            krylov_dot_reductions: None,
            krylov_norm_count: None,
            krylov_axpy_passes: None,
            krylov_basis_bytes_touched: None,
            krylov_synchronization_points: None,
            krylov_exact_residual_checks: None,
            allocation_count: None,
            peak_rss_bytes: peak_rss_bytes(),
            bandwidth_limited_by: None,
            numa_local_node_delta_mb: None,
            numa_other_node_delta_mb: None,
            numa_local_node_gb_s: None,
            numa_other_node_gb_s: None,
            numa_remote_fraction: None,
            amg_profile: None,
            amg_num_levels: None,
            amg_grid_complexity: None,
            amg_operator_complexity: None,
            amg_max_grid_complexity: None,
            amg_max_operator_complexity: None,
            amg_grid_complexity_within_target: None,
            amg_operator_complexity_within_target: None,
            amg_apply_count: None,
            amg_total_apply_ms: None,
            amg_mean_apply_ms: None,
            amg_total_nnz: None,
            amg_finest_nnz: None,
            amg_coarsest_dofs: None,
            notes: None,
        }
    }

    pub fn with_timing(mut self, sample: TimedSample) -> Self {
        self.measured_iterations = sample.iterations;
        self.total_ms = sample.total_ms;
        self.ms_per_iter = sample.ms_per_iter;
        self.gb_s = self
            .bytes_per_iter
            .map(|bytes| gb_s(bytes, sample.ms_per_iter));
        self
    }

    pub fn with_bytes(mut self, bytes_per_iter: u64) -> Self {
        self.bytes_per_iter = Some(bytes_per_iter);
        if self.ms_per_iter > 0.0 {
            self.gb_s = Some(gb_s(bytes_per_iter, self.ms_per_iter));
        }
        self
    }

    pub fn with_csr_spmv_byte_model(mut self, model: CsrSpmvByteModel) -> Self {
        self.bytes_per_iter = Some(model.total_bytes);
        if self.ms_per_iter > 0.0 {
            self.gb_s = Some(gb_s(model.total_bytes, self.ms_per_iter));
        }
        self.csr_spmv_byte_model = Some(model);
        self
    }

    pub fn with_krylov_counts(mut self, counts: KrylovOpCounts) -> Self {
        self.krylov_spmv_count = Some(counts.spmv_count);
        self.krylov_preconditioner_count = Some(counts.preconditioner_count);
        self.krylov_dot_reductions = Some(counts.dot_reductions);
        self.krylov_norm_count = Some(counts.norm_count);
        self.krylov_axpy_passes = Some(counts.axpy_passes);
        self.krylov_basis_bytes_touched = Some(counts.basis_bytes_touched);
        self.krylov_synchronization_points = Some(counts.synchronization_points);
        self.krylov_exact_residual_checks = Some(counts.exact_residual_checks);
        self
    }

    /// Attach the independent correctness gate that preceded a timed row.
    pub fn with_correctness(
        mut self,
        status: impl Into<String>,
        max_abs: f64,
        tolerance: f64,
        actions: impl Into<String>,
    ) -> Self {
        self.correctness_status = Some(status.into());
        self.correctness_max_abs = Some(max_abs);
        self.correctness_tolerance = Some(tolerance);
        self.action_parity = Some(actions.into());
        self
    }

    /// Attach placement and storage accounting for a NUMA timing row.
    pub fn with_numa_evidence(
        mut self,
        policy: impl Into<String>,
        affinity: impl Into<String>,
        partitioned_csr_bytes: usize,
        input_vector_bytes: usize,
        replicated_input_bytes: usize,
        total_bytes: usize,
    ) -> Self {
        self.numa_policy = Some(policy.into());
        self.thread_affinity = Some(affinity.into());
        self.numa_partitioned_csr_bytes = u64::try_from(partitioned_csr_bytes).ok();
        self.numa_input_vector_bytes = u64::try_from(input_vector_bytes).ok();
        self.numa_replicated_input_bytes = u64::try_from(replicated_input_bytes).ok();
        self.numa_memory_estimate_bytes = u64::try_from(total_bytes).ok();
        self
    }

    /// Attach structured NUMA admission, placement, and partition evidence.
    ///
    /// These fields deliberately remain optional for older performance
    /// artifacts, while new NUMA rows no longer need downstream consumers to
    /// parse the human-readable `notes` string.
    pub fn with_numa_partitioning(
        mut self,
        report: &crate::sparse::NumaPartitioningReport,
        stats: &[crate::sparse::PartitionStats],
    ) -> Self {
        self.numa_requested_partitions = Some(report.requested_partitions);
        self.numa_admitted_partitions = Some(report.admitted_partitions);
        self.numa_placement_status = Some(report.placement_status.as_str().to_string());
        self.numa_row_imbalance = Some(report.row_imbalance);
        self.numa_nnz_imbalance = Some(report.nnz_imbalance);
        self.numa_partition_stats = stats
            .iter()
            .map(|stat| NumaPartitionPerfStat {
                row_start: stat.row_start,
                num_rows: stat.num_rows,
                nnz: stat.nnz,
                numa_node_id: stat.numa_node_id,
                socket_id: stat.socket_id,
                logical_cpu_count: stat.logical_cpu_count,
                first_touch_affinity: stat.first_touch_affinity.to_string(),
                first_touch_affinity_applied: stat.first_touch_affinity_applied,
            })
            .collect();
        self
    }

    pub fn record_setup_reuse_economics(&mut self, rhs_count: usize, scope: impl Into<String>) {
        let rhs_count = rhs_count.max(1);
        self.rhs_count = rhs_count;
        self.setup_reuse_scope = Some(scope.into());

        if let Some(solve_ms) = self.solve_ms {
            let solve_ms_per_rhs = solve_ms / rhs_count as f64;
            self.mean_solve_ms_per_rhs = Some(solve_ms_per_rhs);
            if solve_ms_per_rhs > 0.0
                && let Some(setup_ms) = self.setup_ms
                && setup_ms > 0.0
            {
                self.setup_reuse_break_even_rhs = Some(setup_ms / solve_ms_per_rhs);
            }
            self.amortized_total_ms_per_rhs_4 = Some(amortized_total_ms_per_rhs(
                self.setup_ms,
                solve_ms_per_rhs,
                4,
            ));
            self.amortized_total_ms_per_rhs_16 = Some(amortized_total_ms_per_rhs(
                self.setup_ms,
                solve_ms_per_rhs,
                16,
            ));
        }

        if let Some(iterations) = self.solver_iterations {
            self.mean_solver_iterations_per_rhs = Some(iterations as f64 / rhs_count as f64);
        }
    }
}

pub fn append_run_from_env(run: PerfRun) -> std::io::Result<bool> {
    let Some(path) = perf_json_path() else {
        return Ok(false);
    };
    append_run(&path, run)?;
    Ok(true)
}

pub fn append_run(path: &Path, run: PerfRun) -> std::io::Result<()> {
    let mut report = load_or_new(path);
    report.generated_at_unix_ms = now_unix_ms();
    report.runs.push(run);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&report)
        .map_err(|err| std::io::Error::other(format!("serialize perf report: {err}")))?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn perf_json_path() -> Option<PathBuf> {
    std::env::var_os(PERF_JSON_ENV).map(PathBuf::from)
}

pub fn measure_for<T, F>(min_duration: Duration, max_iterations: usize, mut f: F) -> TimedSample
where
    F: FnMut() -> T,
{
    let max_iterations = max_iterations.max(1);
    std::hint::black_box(f());

    let start = Instant::now();
    let mut iterations = 0usize;
    while iterations < max_iterations {
        std::hint::black_box(f());
        iterations += 1;
        if start.elapsed() >= min_duration {
            break;
        }
    }

    let total_ms = start.elapsed().as_secs_f64() * 1e3;
    TimedSample {
        iterations,
        total_ms,
        ms_per_iter: total_ms / iterations as f64,
    }
}

pub fn time_once<T, F>(f: F) -> (T, f64)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let out = f();
    (out, start.elapsed().as_secs_f64() * 1e3)
}

pub fn bytes_real_csr_spmv(nrows: usize, ncols: usize, nnz: usize, value_bytes: usize) -> u64 {
    csr_spmv_byte_model(nrows, ncols, nnz, value_bytes, std::mem::size_of::<usize>()).total_bytes
}

pub fn bytes_real_csr_spmv_with_index(
    nrows: usize,
    ncols: usize,
    nnz: usize,
    value_bytes: usize,
    index_bytes: usize,
) -> u64 {
    csr_spmv_byte_model(nrows, ncols, nnz, value_bytes, index_bytes).total_bytes
}

pub fn csr_spmv_byte_model(
    nrows: usize,
    ncols: usize,
    nnz: usize,
    value_bytes: usize,
    index_bytes: usize,
) -> CsrSpmvByteModel {
    let matrix_value_bytes = (value_bytes * nnz) as u64;
    let matrix_col_index_bytes = (index_bytes * nnz) as u64;
    let matrix_row_ptr_bytes = (index_bytes * (nrows + 1)) as u64;
    let input_vector_bytes = (value_bytes * ncols) as u64;
    let output_vector_bytes = (value_bytes * nrows) as u64;
    let total_bytes = matrix_value_bytes
        + matrix_col_index_bytes
        + matrix_row_ptr_bytes
        + input_vector_bytes
        + output_vector_bytes;

    CsrSpmvByteModel {
        value_bytes,
        col_index_bytes: index_bytes,
        row_ptr_index_bytes: index_bytes,
        matrix_value_bytes,
        matrix_col_index_bytes,
        matrix_row_ptr_bytes,
        input_vector_bytes,
        output_vector_bytes,
        total_bytes,
        bytes_per_nnz: if nnz == 0 {
            0.0
        } else {
            total_bytes as f64 / nnz as f64
        },
    }
}

pub fn bytes_dense_read(len: usize, elem_bytes: usize) -> u64 {
    (len * elem_bytes) as u64
}

pub fn bytes_dense_write(len: usize, elem_bytes: usize) -> u64 {
    (len * elem_bytes) as u64
}

pub fn bytes_dense_read_write(len: usize, reads: usize, writes: usize, elem_bytes: usize) -> u64 {
    (len * (reads + writes) * elem_bytes) as u64
}

fn load_or_new(path: &Path) -> PerfReport {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str::<PerfReport>(&s).ok())
        .unwrap_or_else(new_report)
}

fn new_report() -> PerfReport {
    PerfReport {
        schema_version: 10,
        generated_at_unix_ms: now_unix_ms(),
        crate_name: env!("CARGO_PKG_NAME").to_string(),
        crate_version: env!("CARGO_PKG_VERSION").to_string(),
        git_sha: detect_git_sha(),
        hardware: detect_hardware(),
        runs: Vec::new(),
    }
}

fn detect_hardware() -> HardwareInfo {
    let rt = runtime::runtime();
    let topology = MachineTopology::detect();
    HardwareInfo {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_model: detect_cpu_model(),
        compiler: command_stdout("rustc", &["-Vv"]),
        target_features: detect_target_features(),
        microcode: detect_microcode(),
        total_threads: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        rayon_threads: rayon::current_num_threads(),
        runtime_compute_threads: rt.compute_threads,
        runtime_bandwidth_threads: rt.bandwidth_threads,
        runtime_total_cores: rt.total_cores,
        runtime_perf_cores: rt.perf_cores,
        numa_partitions: detect_numa_partitions(),
        socket_count: topology.sockets.len().max(1),
        numa_node_count: topology.numa_nodes.len().max(1),
        topology_summary: topology.short_summary(),
        topology_notes: topology.notes.clone(),
        numa_nodes: topology
            .numa_nodes
            .iter()
            .map(|node| NumaNodeInfo {
                id: node.id,
                logical_cpus: node.logical_cpus.clone(),
                memory_bytes: node.memory_bytes,
            })
            .collect(),
        socket_cpu_sets: topology
            .sockets
            .iter()
            .map(|socket| SocketInfo {
                id: socket.id,
                logical_cpus: socket.logical_cpus.clone(),
                numa_node_ids: socket.numa_node_ids.clone(),
            })
            .collect(),
    }
}

fn detect_cpu_model() -> String {
    match std::env::consts::OS {
        "macos" => command_stdout("sysctl", &["-n", "machdep.cpu.brand_string"])
            .unwrap_or_else(|| "unknown".to_string()),
        "linux" => {
            let content = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
            content
                .lines()
                .find_map(|line| {
                    line.strip_prefix("model name")
                        .and_then(|rest| rest.split_once(':'))
                        .map(|(_, value)| value.trim().to_string())
                })
                .unwrap_or_else(|| "unknown".to_string())
        }
        _ => "unknown".to_string(),
    }
}

fn detect_microcode() -> Option<String> {
    if std::env::consts::OS != "linux" {
        return None;
    }
    let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    content.lines().find_map(|line| {
        line.strip_prefix("microcode")
            .and_then(|rest| rest.split_once(':'))
            .map(|(_, value)| value.trim().to_string())
    })
}

fn detect_target_features() -> String {
    let force_scalar = std::env::var_os("SONIUM_SPARSE_FORCE_SCALAR")
        .is_some_and(|value| value == "1" || value == "true");
    #[cfg(target_arch = "x86_64")]
    {
        format!(
            "compile_avx2={};compile_fma={};runtime_avx2={};runtime_fma={};force_scalar={};rustflags={}",
            cfg!(target_feature = "avx2"),
            cfg!(target_feature = "fma"),
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("fma"),
            force_scalar,
            std::env::var("RUSTFLAGS").unwrap_or_default()
        )
    }
    #[cfg(target_arch = "aarch64")]
    {
        format!(
            "compile_neon={};force_scalar={};rustflags={}",
            cfg!(target_feature = "neon"),
            force_scalar,
            std::env::var("RUSTFLAGS").unwrap_or_default()
        )
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        format!(
            "force_scalar={};rustflags={}",
            force_scalar,
            std::env::var("RUSTFLAGS").unwrap_or_default()
        )
    }
}

fn detect_git_sha() -> Option<String> {
    command_stdout("git", &["rev-parse", "--short", "HEAD"])
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn peak_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }

    #[cfg(target_os = "macos")]
    {
        let pid = std::process::id().to_string();
        let output = Command::new("ps")
            .args(["-o", "rss=", "-p", &pid])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let kb = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<u64>()
            .ok()?;
        Some(kb * 1024)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

fn gb_s(bytes: u64, ms: f64) -> f64 {
    if ms <= 0.0 {
        return 0.0;
    }
    bytes as f64 / (ms / 1e3) / 1e9
}

fn default_rhs_count() -> usize {
    1
}

fn amortized_total_ms_per_rhs(
    setup_ms: Option<f64>,
    solve_ms_per_rhs: f64,
    rhs_count: usize,
) -> f64 {
    solve_ms_per_rhs + setup_ms.unwrap_or(0.0) / rhs_count.max(1) as f64
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bandwidth_math_is_reasonable() {
        let bytes = 1_000_000_000;
        let rate = gb_s(bytes, 1_000.0);
        assert!((rate - 1.0).abs() < 1e-12);
    }

    #[test]
    fn csr_byte_model_includes_indices_and_vectors() {
        let bytes = bytes_real_csr_spmv(10, 10, 50, 8);
        assert!(bytes > (50 * 8) as u64);
        let usize_model = csr_spmv_byte_model(10, 10, 50, 8, std::mem::size_of::<usize>());
        let u32_model = csr_spmv_byte_model(10, 10, 50, 8, std::mem::size_of::<u32>());
        assert_eq!(bytes, usize_model.total_bytes);
        assert!(u32_model.total_bytes < usize_model.total_bytes);
        assert!(usize_model.bytes_per_nnz > 0.0);
    }

    #[test]
    fn perf_run_records_exact_residual_checks() {
        let counts = KrylovOpCounts {
            exact_residual_checks: 3,
            ..Default::default()
        };

        let run = PerfRun::new("bench", "group", "id").with_krylov_counts(counts);

        assert_eq!(run.krylov_exact_residual_checks, Some(3));
    }

    #[test]
    fn perf_run_defaults_amg_complexity_targets() {
        let run = PerfRun::new("bench", "group", "id");

        assert_eq!(run.amg_max_grid_complexity, None);
        assert_eq!(run.amg_max_operator_complexity, None);
        assert_eq!(run.amg_grid_complexity_within_target, None);
        assert_eq!(run.amg_operator_complexity_within_target, None);
    }

    #[test]
    fn perf_run_records_setup_reuse_economics() {
        let mut run = PerfRun::new("bench", "group", "id");
        run.setup_ms = Some(12.0);
        run.solve_ms = Some(20.0);
        run.solver_iterations = Some(40);

        run.record_setup_reuse_economics(4, "same_matrix_repeated_rhs");

        assert_eq!(run.rhs_count, 4);
        assert_eq!(
            run.setup_reuse_scope.as_deref(),
            Some("same_matrix_repeated_rhs")
        );
        assert_eq!(run.mean_solve_ms_per_rhs, Some(5.0));
        assert_eq!(run.mean_solver_iterations_per_rhs, Some(10.0));
        assert_eq!(run.setup_reuse_break_even_rhs, Some(2.4));
        assert_eq!(run.amortized_total_ms_per_rhs_4, Some(8.0));
        assert_eq!(run.amortized_total_ms_per_rhs_16, Some(5.75));
    }

    #[test]
    fn perf_run_records_structured_numa_partitioning() {
        let matrix = crate::sparse::CsrMatrix::from_raw_parts(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            vec![1.0_f64, 2.0],
        );
        let numa =
            crate::sparse::NumaPartitionedCsr::try_from_csr(&matrix, 2).expect("valid NUMA matrix");
        let report = numa.partitioning_report();
        let stats = numa.partition_stats();
        let run = PerfRun::new("numa", "spmv", "shared").with_numa_partitioning(&report, &stats);

        assert_eq!(run.numa_requested_partitions, Some(2));
        assert_eq!(run.numa_admitted_partitions, Some(2));
        assert_eq!(run.numa_partition_stats.len(), 2);
        assert!(run.numa_placement_status.is_some());
        let encoded = serde_json::to_string(&run).expect("NUMA perf row serializes");
        assert!(encoded.contains("numa_partition_stats"));
        assert!(encoded.contains("first_touch_affinity"));
    }
}
