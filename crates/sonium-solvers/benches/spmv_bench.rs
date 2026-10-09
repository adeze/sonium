//! SpMV bandwidth benchmarks
//!
//! Measures effective GB/s for sparse matrix-vector multiplication.
//! Compares CSR vs NUMA-partitioned CSR across matrix sizes and patterns.

#[path = "matrices/mod.rs"]
mod matrices;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, ToPrimitive};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::{
    ComplexField, CsrMatrix, LinearOperator, MachineTopology, NumaSpmvPolicy, NumaStorageEstimate,
    SpmvPlan, SpmvPlanConfig, SpmvPlanPreference, detect_numa_partitions, selected_spmv_kernel,
};
use std::time::{Duration, Instant};

#[cfg(feature = "cli")]
use sonium_solvers::perf::{self, CsrSpmvByteModel, PerfRun};

/// SS2's evidence runner can request a bounded sampling schedule.  The
/// default benchmark schedule remains unchanged for normal Criterion use;
/// the fast mode is only for the machine-readable correctness/provenance
/// producer, whose per-row `measure_for` sample already owns a fixed timing
/// interval.
fn ss2_fast_bench() -> bool {
    std::env::var_os("SONIUM_SS2_FAST_BENCH").is_some_and(|value| value == "1" || value == "true")
}

fn ss2_sample_size(default: usize) -> usize {
    if ss2_fast_bench() { 10 } else { default }
}

fn ss2_warm_up_time() -> Duration {
    if ss2_fast_bench() {
        Duration::from_millis(100)
    } else {
        Duration::from_secs(3)
    }
}

fn ss2_measurement_time(default_seconds: u64) -> Duration {
    if ss2_fast_bench() {
        Duration::from_millis(100)
    } else {
        Duration::from_secs(default_seconds)
    }
}

/// Effective bytes transferred during one f64 CSR SpMV.
///
/// Read: values(8*nnz) + col_idx(8*nnz) + row_ptrs(8*(nrows+1)) + x(8*ncols)
/// Write: y(8*nrows)
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
struct SpmvPerfMeta<'a> {
    group: &'a str,
    id: String,
    operation: &'a str,
    matrix_name: &'a str,
    nrows: usize,
    ncols: usize,
    nnz: usize,
    bytes_per_iter: u64,
    csr_byte_model: Option<CsrSpmvByteModel>,
    notes: Option<String>,
}

#[cfg(feature = "cli")]
#[derive(Default)]
struct SpmvPerfEvidence {
    scalar: Option<String>,
    selected_kernel: Option<String>,
    setup_ms: Option<f64>,
    reuse_count: Option<usize>,
    rhs_count: Option<usize>,
    reuse_scope: Option<String>,
    correctness_status: Option<String>,
    correctness_max_abs: Option<f64>,
    correctness_tolerance: Option<f64>,
    action_parity: Option<String>,
    thread_affinity: Option<String>,
    numa_policy: Option<String>,
    numa_storage: Option<NumaStorageEstimate>,
}

#[cfg(feature = "cli")]
fn record_spmv_perf_with_evidence<F>(
    meta: SpmvPerfMeta<'_>,
    evidence: SpmvPerfEvidence,
    mut apply: F,
) where
    F: FnMut(),
{
    if perf::perf_json_path().is_none() {
        return;
    }

    let sample = perf::measure_for(Duration::from_millis(100), 200, || {
        apply();
    });

    let scalar = evidence.scalar.unwrap_or_else(|| "f64".to_string());
    let selected_kernel = evidence
        .selected_kernel
        .unwrap_or_else(|| selected_kernel_label(&scalar).to_string());
    let mut run = PerfRun::new("spmv_bench", meta.group, meta.id)
        .with_bytes(meta.bytes_per_iter)
        .with_timing(sample);
    run.operation = meta.operation.to_string();
    run.matrix = Some(meta.matrix_name.to_string());
    run.scalar = scalar;
    run.selected_kernel = Some(selected_kernel);
    run.nrows = Some(meta.nrows);
    run.ncols = Some(meta.ncols);
    run.nnz = Some(meta.nnz);
    run.setup_ms = evidence.setup_ms;
    if let Some(reuse_count) = evidence.reuse_count {
        run.solve_ms = Some(sample.total_ms);
        run.record_setup_reuse_economics(
            reuse_count,
            evidence
                .reuse_scope
                .unwrap_or_else(|| "same_matrix_repeated_apply".to_string()),
        );
    }
    run.correctness_status = evidence.correctness_status;
    run.correctness_max_abs = evidence.correctness_max_abs;
    run.correctness_tolerance = evidence.correctness_tolerance;
    run.action_parity = evidence.action_parity;
    run.thread_affinity = Some(
        evidence
            .thread_affinity
            .unwrap_or_else(|| "process_default".to_string()),
    );
    run.numa_policy = Some(
        evidence
            .numa_policy
            .unwrap_or_else(|| "not_applicable".to_string()),
    );
    if let Some(rhs_count) = evidence.rhs_count {
        run.rhs_count = rhs_count.max(1);
    }
    if let Some(storage) = evidence.numa_storage {
        run.numa_memory_estimate_bytes = u64::try_from(storage.total_bytes).ok();
        run.numa_partitioned_csr_bytes = u64::try_from(storage.partitioned_csr_bytes).ok();
        run.numa_input_vector_bytes = u64::try_from(storage.input_vector_bytes).ok();
        run.numa_replicated_input_bytes = u64::try_from(storage.replicated_input_bytes).ok();
    }
    if let Some(model) = meta.csr_byte_model {
        run = run.with_csr_spmv_byte_model(model);
        run.bandwidth_limited_by = Some("csr_spmv_memory_traffic".to_string());
    }
    let notes = meta.notes.unwrap_or_default();
    run.notes = Some(format!(
        "{notes}; timing_scope=steady_state_apply_only;setup_separate; vendor_baseline=unavailable_no_pinned_vendor_sparse_library"
    ));

    if let Err(err) = perf::append_run_from_env(run) {
        eprintln!("failed to write {} report: {err}", perf::PERF_JSON_ENV);
    }
}

#[cfg(feature = "cli")]
fn selected_kernel_label(scalar: &str) -> &'static str {
    match scalar {
        "f32" => selected_spmv_kernel::<f32>().name(),
        "complex32" => selected_spmv_kernel::<Complex32>().name(),
        "complex64" => selected_spmv_kernel::<Complex64>().name(),
        _ => selected_spmv_kernel::<f64>().name(),
    }
}

#[cfg(feature = "cli")]
fn topology_note(num_partitions: usize) -> String {
    let topology = MachineTopology::detect();
    let placement = topology.placement_plan(num_partitions);
    let ss3_status = if topology.sockets.len() > 1 && topology.numa_nodes.len() > 1 {
        "multi_socket_evidence"
    } else {
        "uma_functional_only"
    };
    format!(
        "ss3_status={ss3_status}; numa_partitions={num_partitions}; topology={}; placement={}",
        topology.short_summary(),
        placement.short_summary()
    )
}

fn irregular_rows_matrix(rows: usize, cols: usize) -> CsrMatrix<f64> {
    let mut triplets = Vec::new();
    for row in 0..rows {
        let row_nnz = if row % 8 == 0 { cols.min(64) } else { 2 };
        for offset in 0..row_nnz {
            let col = (row + offset * 17) % cols;
            let value = if row == col {
                row_nnz as f64 + 1.0
            } else {
                -0.25
            };
            triplets.push((row, col, value));
        }
    }
    CsrMatrix::from_triplets(rows, cols, triplets)
}

fn mostly_uniform_band_matrix(rows: usize, cols: usize, bandwidth: usize) -> CsrMatrix<f64> {
    let mut triplets = Vec::new();
    for row in 0..rows {
        let start = row.saturating_sub(bandwidth / 2);
        let end = (start + bandwidth).min(cols);
        for col in start..end {
            let value = if row == col {
                bandwidth as f64 + 1.0
            } else {
                -1.0
            };
            triplets.push((row, col, value));
        }
    }
    CsrMatrix::from_triplets(rows, cols, triplets)
}

fn dense_block_diagonal_matrix(blocks: usize, block_size: usize) -> CsrMatrix<f64> {
    let n = blocks * block_size;
    let mut triplets = Vec::with_capacity(blocks * block_size * block_size);
    for block in 0..blocks {
        let base = block * block_size;
        for row in 0..block_size {
            for col in 0..block_size {
                let value = if row == col { 3.0 } else { -0.125 };
                triplets.push((base + row, base + col, value));
            }
        }
    }
    CsrMatrix::from_triplets(n, n, triplets)
}

fn bench_plan_variant(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    label: &str,
    n: usize,
    matrix: &CsrMatrix<f64>,
    plan: &SpmvPlan<f64>,
    x: &[f64],
) {
    let actual = plan.matvec(x);
    assert_forward_gate(matrix, x, &actual, 1.0e-10);
    group.bench_with_input(BenchmarkId::new(label, n), &(plan, x), |b, &(plan, x)| {
        b.iter(|| plan.matvec(x))
    });
}

/// Independent scalar reference used as a correctness gate immediately
/// before a plan enters Criterion timing. The production CSR path may select
/// SIMD, so comparing transformed plans with `matrix.matvec` alone would not
/// detect a common dispatch error.
fn scalar_csr_matvec<T: ComplexField>(matrix: &CsrMatrix<T>, x: &[T]) -> Vec<T> {
    assert_eq!(x.len(), matrix.num_cols);
    (0..matrix.num_rows)
        .map(|row| {
            let mut sum = T::zero();
            for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                sum += matrix.values[index] * x[matrix.col_indices[index]];
            }
            sum
        })
        .collect()
}

fn scalar_csr_adjoint_matvec<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    x: &[T],
    hermitian: bool,
) -> Vec<T> {
    assert_eq!(x.len(), matrix.num_rows);
    let mut y = vec![T::zero(); matrix.num_cols];
    for (row, x_value) in x.iter().enumerate().take(matrix.num_rows) {
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let value = if hermitian {
                matrix.values[index].conj()
            } else {
                matrix.values[index]
            };
            y[matrix.col_indices[index]] += value * *x_value;
        }
    }
    y
}

fn max_abs_error<T: ComplexField>(expected: &[T], actual: &[T]) -> f64 {
    assert_eq!(expected.len(), actual.len());
    expected
        .iter()
        .zip(actual)
        .map(|(expected, actual)| {
            let mut delta = *actual;
            delta -= *expected;
            delta.norm().to_f64().unwrap_or(f64::INFINITY)
        })
        .fold(0.0, f64::max)
}

fn assert_forward_gate<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    x: &[T],
    actual: &[T],
    tolerance: f64,
) -> f64 {
    let expected = scalar_csr_matvec(matrix, x);
    let error = max_abs_error(&expected, actual);
    assert!(
        error.is_finite() && error <= tolerance,
        "independent scalar forward gate failed: error={error:e} tolerance={tolerance:e}"
    );
    error
}

fn assert_direct_gate<T: ComplexField>(matrix: &CsrMatrix<T>, x: &[T], tolerance: f64) -> f64 {
    let actual = matrix
        .try_matvec(x)
        .expect("checked direct CSR correctness gate");
    assert_forward_gate(matrix, x, &actual, tolerance)
}

#[cfg(feature = "cli")]
fn block_fixture<T: ComplexField>(ncols: usize, rhs_count: usize) -> DenseMatrix<T> {
    DenseMatrix::from_fn(ncols, rhs_count, |row, column| {
        T::from_re_im(
            T::Real::from_f64(0.25 + (row % 29) as f64 * 0.03125 + column as f64 * 0.125)
                .expect("block fixture real value is representable"),
            T::Real::from_f64(-0.125 + (row % 11) as f64 * 0.015625)
                .expect("block fixture imaginary value is representable"),
        )
    })
}

#[cfg(feature = "cli")]
fn scalar_block_expected<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    input: &DenseMatrix<T>,
) -> DenseMatrix<T> {
    let mut expected = DenseMatrix::zeros(matrix.num_rows, input.ncols());
    for column in 0..input.ncols() {
        let rhs = input.column_to_vec(column);
        expected.set_column(column, &scalar_csr_matvec(matrix, &rhs));
    }
    expected
}

#[cfg(feature = "cli")]
fn assert_csr_block_gate<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    input: &DenseMatrix<T>,
    tolerance: f64,
) -> f64 {
    let expected = scalar_block_expected(matrix, input);
    let mut actual = DenseMatrix::zeros(matrix.num_rows, input.ncols());
    matrix
        .try_spmm_into(input, &mut actual)
        .expect("checked CSR block correctness gate");
    let error = (0..input.ncols())
        .map(|column| {
            max_abs_error(
                &expected.column_to_vec(column),
                &actual.column_to_vec(column),
            )
        })
        .fold(0.0, f64::max);
    assert!(
        error.is_finite() && error <= tolerance,
        "independent scalar CSR block gate failed: error={error:e} tolerance={tolerance:e}"
    );
    error
}

#[cfg(feature = "cli")]
fn assert_plan_block_gate<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    plan: &SpmvPlan<T>,
    input: &DenseMatrix<T>,
    tolerance: f64,
) -> f64 {
    let expected = scalar_block_expected(matrix, input);
    let mut actual = DenseMatrix::zeros(matrix.num_rows, input.ncols());
    plan.try_apply_block(input, &mut actual)
        .expect("checked planned block correctness gate");
    let error = (0..input.ncols())
        .map(|column| {
            max_abs_error(
                &expected.column_to_vec(column),
                &actual.column_to_vec(column),
            )
        })
        .fold(0.0, f64::max);
    assert!(
        error.is_finite() && error <= tolerance,
        "independent scalar planned block gate failed: error={error:e} tolerance={tolerance:e}"
    );
    error
}

fn assert_numa_gate(
    matrix: &CsrMatrix<f64>,
    numa: &sonium_solvers::NumaPartitionedCsr<f64>,
    x: &[f64],
    policy: NumaSpmvPolicy,
    tolerance: f64,
) -> f64 {
    let expected_forward = scalar_csr_matvec(matrix, x);
    let expected_transpose = scalar_csr_adjoint_matvec(matrix, x, false);
    let expected_hermitian = scalar_csr_adjoint_matvec(matrix, x, true);
    let actual_forward = numa
        .try_matvec_with_policy(x, policy)
        .expect("NUMA forward correctness gate");
    let actual_transpose = numa
        .try_matvec_transpose(x)
        .expect("NUMA transpose correctness gate");
    let actual_hermitian = numa
        .try_matvec_hermitian(x)
        .expect("NUMA Hermitian correctness gate");
    let error = [
        max_abs_error(&expected_forward, &actual_forward),
        max_abs_error(&expected_transpose, &actual_transpose),
        max_abs_error(&expected_hermitian, &actual_hermitian),
    ]
    .into_iter()
    .fold(0.0, f64::max);
    assert!(
        error.is_finite() && error <= tolerance,
        "independent scalar NUMA gate failed: error={error:e} tolerance={tolerance:e}"
    );
    error
}

fn scalar_label<T: ComplexField>() -> &'static str {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        "f32"
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        "f64"
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex32>() {
        "complex32"
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex64>() {
        "complex64"
    } else {
        "unknown"
    }
}

fn scalar_tolerance<T: ComplexField>() -> f64 {
    match scalar_label::<T>() {
        "f32" => 5.0e-5,
        "complex32" => 1.5e-4,
        _ => 1.0e-10,
    }
}

#[cfg(feature = "cli")]
fn assert_plan_actions_gate<T: ComplexField + 'static>(
    matrix: &CsrMatrix<T>,
    plan: &SpmvPlan<T>,
    x: &[T],
    tolerance: f64,
) -> (f64, String) {
    // The scalar CSR oracle is deliberately independent of both the direct
    // dispatch and the transformed storage.  Check every action before the
    // row is handed to the timing harness so a fast but wrong plan cannot be
    // published as a performance result.
    let forward = plan.matvec(x);
    let forward_error = assert_forward_gate(matrix, x, &forward, tolerance);

    let transpose_input = (0..matrix.num_rows)
        .map(|row| {
            x.get(row).copied().unwrap_or_else(|| {
                T::from_re_im(
                    T::Real::from_f64(0.25 + (row % 17) as f64 * 0.03125)
                        .expect("transpose fixture real value is representable"),
                    T::Real::from_f64(-0.125 + (row % 7) as f64 * 0.015625)
                        .expect("transpose fixture imaginary value is representable"),
                )
            })
        })
        .collect::<Vec<_>>();
    let expected_transpose = scalar_csr_adjoint_matvec(matrix, &transpose_input, false);
    let transpose = plan
        .try_apply_transpose(&transpose_input)
        .expect("planned transpose action correctness gate");
    let transpose_error = max_abs_error(&expected_transpose, &transpose);
    assert!(
        transpose_error.is_finite() && transpose_error <= tolerance,
        "independent scalar planned transpose gate failed: error={transpose_error:e} tolerance={tolerance:e}"
    );

    let expected_hermitian = scalar_csr_adjoint_matvec(matrix, &transpose_input, true);
    let hermitian = plan
        .try_apply_hermitian(&transpose_input)
        .expect("planned Hermitian action correctness gate");
    let hermitian_error = max_abs_error(&expected_hermitian, &hermitian);
    assert!(
        hermitian_error.is_finite() && hermitian_error <= tolerance,
        "independent scalar planned Hermitian gate failed: error={hermitian_error:e} tolerance={tolerance:e}"
    );

    let block_input = block_fixture::<T>(matrix.num_cols, 2);
    let block_error = assert_plan_block_gate(matrix, plan, &block_input, tolerance);
    let error = [forward_error, transpose_error, hermitian_error, block_error]
        .into_iter()
        .fold(0.0, f64::max);
    (
        error,
        format!("forward=passed;transpose=passed;hermitian=passed;block=passed;max_abs={error:e}"),
    )
}

#[cfg(feature = "cli")]
fn record_plan_perf(
    label: &str,
    matrix_name: &str,
    csr: &CsrMatrix<f64>,
    plan: &SpmvPlan<f64>,
    x: &[f64],
    setup_ms: f64,
) {
    let (max_abs, action_parity) = assert_plan_actions_gate(csr, plan, x, 1.0e-10);
    record_spmv_perf_with_evidence(
        SpmvPerfMeta {
            group: "spmv/plan",
            id: format!("{label}/{:?}/{}", plan.format(), csr.num_rows),
            operation: "spmv",
            matrix_name,
            nrows: csr.num_rows,
            ncols: csr.num_cols,
            nnz: csr.nnz(),
            bytes_per_iter: plan.report().chosen_bytes_per_apply,
            csr_byte_model: None,
            notes: Some(plan.report().summary()),
        },
        SpmvPerfEvidence {
            selected_kernel: Some(plan.report().selected_kernel.name().to_string()),
            setup_ms: Some(setup_ms),
            reuse_count: Some(plan.report().expected_reuse),
            reuse_scope: Some("same_matrix_repeated_apply".to_string()),
            correctness_status: Some("passed".to_string()),
            correctness_max_abs: Some(max_abs),
            correctness_tolerance: Some(1.0e-10),
            action_parity: Some(action_parity),
            ..Default::default()
        },
        || {
            std::hint::black_box(plan.matvec(x));
        },
    );
}

#[cfg(feature = "cli")]
#[allow(clippy::too_many_arguments)]
fn record_plan_perf_typed<T: ComplexField + 'static>(
    label: &str,
    matrix_name: &str,
    csr: &CsrMatrix<T>,
    plan: &SpmvPlan<T>,
    x: &[T],
    setup_ms: f64,
    reuse_count: usize,
    max_abs: f64,
    tolerance: f64,
) {
    let (action_max_abs, action_parity) = assert_plan_actions_gate(csr, plan, x, tolerance);
    record_spmv_perf_with_evidence(
        SpmvPerfMeta {
            group: "spmv/plan",
            id: format!("{label}/{:?}/{}", plan.format(), csr.num_rows),
            operation: "spmv",
            matrix_name,
            nrows: csr.num_rows,
            ncols: csr.num_cols,
            nnz: csr.nnz(),
            bytes_per_iter: plan.report().chosen_bytes_per_apply,
            csr_byte_model: None,
            notes: Some(plan.report().summary()),
        },
        SpmvPerfEvidence {
            scalar: Some(scalar_label::<T>().to_string()),
            selected_kernel: Some(plan.report().selected_kernel.name().to_string()),
            setup_ms: Some(setup_ms),
            reuse_count: Some(reuse_count),
            reuse_scope: Some("same_matrix_repeated_apply".to_string()),
            correctness_status: Some("passed".to_string()),
            correctness_max_abs: Some(max_abs.max(action_max_abs)),
            correctness_tolerance: Some(tolerance),
            action_parity: Some(action_parity),
            ..Default::default()
        },
        || {
            std::hint::black_box(plan.matvec(x));
        },
    );
}

#[cfg(feature = "cli")]
#[allow(clippy::too_many_arguments)]
fn record_block_perf_typed<T, F>(
    label: &str,
    matrix_name: &str,
    csr: &CsrMatrix<T>,
    input: &DenseMatrix<T>,
    setup_ms: Option<f64>,
    max_abs: f64,
    tolerance: f64,
    selected_kernel: &str,
    notes: String,
    apply: F,
) where
    T: ComplexField + 'static,
    F: FnMut(),
{
    let bytes_per_rhs = perf::bytes_real_csr_spmv(
        csr.num_rows,
        csr.num_cols,
        csr.nnz(),
        std::mem::size_of::<T>(),
    );
    let bytes_per_iter = bytes_per_rhs
        .checked_mul(input.ncols() as u64)
        .expect("block SpMV byte model overflow");
    record_spmv_perf_with_evidence(
        SpmvPerfMeta {
            group: "spmv/block",
            id: format!("{label}/{}/k{}", scalar_label::<T>(), input.ncols()),
            operation: "spmm",
            matrix_name,
            nrows: csr.num_rows,
            ncols: csr.num_cols,
            nnz: csr.nnz(),
            bytes_per_iter,
            csr_byte_model: None,
            notes: Some(notes),
        },
        SpmvPerfEvidence {
            scalar: Some(scalar_label::<T>().to_string()),
            selected_kernel: Some(selected_kernel.to_string()),
            setup_ms,
            rhs_count: Some(input.ncols()),
            correctness_status: Some("passed".to_string()),
            correctness_max_abs: Some(max_abs),
            correctness_tolerance: Some(tolerance),
            action_parity: Some("block=passed".to_string()),
            ..Default::default()
        },
        apply,
    );
}

// ---------------------------------------------------------------------------
// SpMV: CSR vs NUMA
// ---------------------------------------------------------------------------

fn bench_csr_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv/csr");
    group.sample_size(ss2_sample_size(100));
    group.warm_up_time(ss2_warm_up_time());
    group.measurement_time(ss2_measurement_time(10));

    for &target_dofs in &[1_000, 10_000, 100_000] {
        let n = matrices::grid_size_2d(target_dofs);
        let matrix = matrices::poisson_2d(n);
        let nn = matrix.num_rows;
        let x = vec![1.0_f64; nn];
        let bytes = spmv_bytes(nn, nn, matrix.nnz());
        let correctness_error = assert_direct_gate(&matrix, &x, 1.0e-10);

        #[cfg(feature = "cli")]
        let csr_model = perf::csr_spmv_byte_model(
            nn,
            nn,
            matrix.nnz(),
            std::mem::size_of::<f64>(),
            std::mem::size_of::<usize>(),
        );
        #[cfg(feature = "cli")]
        record_spmv_perf_with_evidence(
            SpmvPerfMeta {
                group: "spmv/csr",
                id: format!("poisson_2d/{nn}"),
                operation: "spmv",
                matrix_name: "poisson_2d",
                nrows: nn,
                ncols: nn,
                nnz: matrix.nnz(),
                bytes_per_iter: bytes,
                csr_byte_model: Some(csr_model),
                notes: Some("single-shot perf sample; Criterion owns stable statistics".into()),
            },
            SpmvPerfEvidence {
                correctness_status: Some("passed".to_string()),
                correctness_max_abs: Some(correctness_error),
                correctness_tolerance: Some(1.0e-10),
                action_parity: Some("forward=passed".to_string()),
                ..Default::default()
            },
            || {
                std::hint::black_box(matrix.matvec(&x));
            },
        );

        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(
            BenchmarkId::new("poisson_2d", nn),
            &(matrix, x),
            |b, (matrix, x)| b.iter(|| matrix.matvec(x)),
        );
    }

    // 3D Laplacian (sparser per row, larger)
    for &target_dofs in &[1_000, 10_000, 100_000] {
        let n = matrices::grid_size_3d(target_dofs);
        let matrix = matrices::laplacian_3d(n);
        let nn = matrix.num_rows;
        let x = vec![1.0_f64; nn];
        let bytes = spmv_bytes(nn, nn, matrix.nnz());
        let correctness_error = assert_direct_gate(&matrix, &x, 1.0e-10);

        #[cfg(feature = "cli")]
        let csr_model = perf::csr_spmv_byte_model(
            nn,
            nn,
            matrix.nnz(),
            std::mem::size_of::<f64>(),
            std::mem::size_of::<usize>(),
        );
        #[cfg(feature = "cli")]
        record_spmv_perf_with_evidence(
            SpmvPerfMeta {
                group: "spmv/csr",
                id: format!("laplacian_3d/{nn}"),
                operation: "spmv",
                matrix_name: "laplacian_3d",
                nrows: nn,
                ncols: nn,
                nnz: matrix.nnz(),
                bytes_per_iter: bytes,
                csr_byte_model: Some(csr_model),
                notes: Some("single-shot perf sample; Criterion owns stable statistics".into()),
            },
            SpmvPerfEvidence {
                correctness_status: Some("passed".to_string()),
                correctness_max_abs: Some(correctness_error),
                correctness_tolerance: Some(1.0e-10),
                action_parity: Some("forward=passed".to_string()),
                ..Default::default()
            },
            || {
                std::hint::black_box(matrix.matvec(&x));
            },
        );

        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(
            BenchmarkId::new("laplacian_3d", nn),
            &(matrix, x),
            |b, (matrix, x)| b.iter(|| matrix.matvec(x)),
        );
    }

    // Random banded (irregular access pattern)
    for &n in &[1_000, 10_000, 100_000] {
        let matrix = matrices::random_diag_dominant(n, 10);
        let x = vec![1.0_f64; n];
        let bytes = spmv_bytes(n, n, matrix.nnz());
        let correctness_error = assert_direct_gate(&matrix, &x, 1.0e-10);

        #[cfg(feature = "cli")]
        let csr_model = perf::csr_spmv_byte_model(
            n,
            n,
            matrix.nnz(),
            std::mem::size_of::<f64>(),
            std::mem::size_of::<usize>(),
        );
        #[cfg(feature = "cli")]
        record_spmv_perf_with_evidence(
            SpmvPerfMeta {
                group: "spmv/csr",
                id: format!("random_banded/{n}"),
                operation: "spmv",
                matrix_name: "random_banded",
                nrows: n,
                ncols: n,
                nnz: matrix.nnz(),
                bytes_per_iter: bytes,
                csr_byte_model: Some(csr_model),
                notes: Some("single-shot perf sample; Criterion owns stable statistics".into()),
            },
            SpmvPerfEvidence {
                correctness_status: Some("passed".to_string()),
                correctness_max_abs: Some(correctness_error),
                correctness_tolerance: Some(1.0e-10),
                action_parity: Some("forward=passed".to_string()),
                ..Default::default()
            },
            || {
                std::hint::black_box(matrix.matvec(&x));
            },
        );

        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(
            BenchmarkId::new("random_banded", n),
            &(matrix, x),
            |b, (matrix, x)| b.iter(|| matrix.matvec(x)),
        );
    }

    group.finish();
}

fn bench_numa_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv/numa");
    group.sample_size(ss2_sample_size(100));
    group.warm_up_time(ss2_warm_up_time());
    group.measurement_time(ss2_measurement_time(10));

    let num_partitions = detect_numa_partitions();
    #[cfg(feature = "cli")]
    let topology_note = topology_note(num_partitions);

    for &target_dofs in &[10_000, 100_000, 500_000] {
        let n = matrices::grid_size_2d(target_dofs);
        let csr = matrices::poisson_2d(n);
        let nn = csr.num_rows;
        let numa = csr.to_numa_partitioned(num_partitions);
        let x = vec![1.0_f64; nn];
        let numa_plan_setup_start = Instant::now();
        let numa_plan = SpmvPlan::try_from_csr(
            &csr,
            SpmvPlanConfig {
                preference: SpmvPlanPreference::NumaCsr,
                expected_reuse: 16,
                numa_partitions: Some(num_partitions),
                ..Default::default()
            },
        )
        .expect("checked NUMA plan construction");
        let numa_plan_setup_ms = numa_plan_setup_start.elapsed().as_secs_f64() * 1.0e3;
        let bytes = spmv_bytes(nn, nn, csr.nnz());
        let csr_actual = csr.try_matvec(&x).expect("checked CSR correctness gate");
        let csr_error = assert_forward_gate(&csr, &x, &csr_actual, 1.0e-10);
        let numa_plan_error = assert_forward_gate(&csr, &x, &numa_plan.matvec(&x), 1.0e-10);
        let shared_unpinned_error =
            assert_numa_gate(&csr, &numa, &x, NumaSpmvPolicy::SHARED_UNPINNED, 1.0e-10);
        let shared_pinned_error =
            assert_numa_gate(&csr, &numa, &x, NumaSpmvPolicy::SHARED_PINNED, 1.0e-10);
        let replicated_pinned_error =
            assert_numa_gate(&csr, &numa, &x, NumaSpmvPolicy::REPLICATED_PINNED, 1.0e-10);
        let shared_storage = numa
            .storage_estimate(NumaSpmvPolicy::SHARED_UNPINNED)
            .expect("checked shared NUMA storage estimate");
        let pinned_storage = numa
            .storage_estimate(NumaSpmvPolicy::SHARED_PINNED)
            .expect("checked pinned NUMA storage estimate");
        let replicated_storage = numa
            .storage_estimate(NumaSpmvPolicy::REPLICATED_PINNED)
            .expect("checked replicated NUMA storage estimate");

        #[cfg(feature = "cli")]
        {
            let csr_model = perf::csr_spmv_byte_model(
                nn,
                nn,
                csr.nnz(),
                std::mem::size_of::<f64>(),
                std::mem::size_of::<usize>(),
            );
            let partition_stats = numa
                .partition_stats()
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
                .join("; ");
            record_spmv_perf_with_evidence(
                SpmvPerfMeta {
                    group: "spmv/numa",
                    id: format!("csr/{nn}"),
                    operation: "spmv",
                    matrix_name: "poisson_2d",
                    nrows: nn,
                    ncols: nn,
                    nnz: csr.nnz(),
                    bytes_per_iter: bytes,
                    csr_byte_model: Some(csr_model.clone()),
                    notes: Some(format!(
                        "CSR baseline inside NUMA comparison group; {topology_note}"
                    )),
                },
                SpmvPerfEvidence {
                    correctness_status: Some("passed".to_string()),
                    correctness_max_abs: Some(csr_error),
                    correctness_tolerance: Some(1.0e-10),
                    action_parity: Some("forward=passed".to_string()),
                    ..Default::default()
                },
                || {
                    std::hint::black_box(csr.matvec(&x));
                },
            );
            record_plan_perf(
                "numa_plan",
                "poisson_2d",
                &csr,
                &numa_plan,
                &x,
                numa_plan_setup_ms,
            );
            record_spmv_perf_with_evidence(
                SpmvPerfMeta {
                    group: "spmv/numa",
                    id: format!("numa_p{num_partitions}/{nn}"),
                    operation: "spmv",
                    matrix_name: "poisson_2d",
                    nrows: nn,
                    ncols: nn,
                    nnz: numa.nnz(),
                    bytes_per_iter: bytes,
                    csr_byte_model: Some(csr_model.clone()),
                    notes: Some(format!("{topology_note}; partitions=[{partition_stats}]")),
                },
                SpmvPerfEvidence {
                    correctness_status: Some("passed".to_string()),
                    correctness_max_abs: Some(shared_unpinned_error),
                    correctness_tolerance: Some(1.0e-10),
                    action_parity: Some(
                        "forward=passed;transpose=passed;hermitian=passed".to_string(),
                    ),
                    thread_affinity: Some(format!("apply=unpinned; {topology_note}")),
                    numa_policy: Some("shared_unpinned".to_string()),
                    numa_storage: Some(shared_storage),
                    ..Default::default()
                },
                || {
                    std::hint::black_box(numa.matvec(&x));
                },
            );
            record_spmv_perf_with_evidence(
                SpmvPerfMeta {
                    group: "spmv/numa",
                    id: format!("numa_pinned_shared_p{num_partitions}/{nn}"),
                    operation: "spmv",
                    matrix_name: "poisson_2d",
                    nrows: nn,
                    ncols: nn,
                    nnz: numa.nnz(),
                    bytes_per_iter: bytes,
                    csr_byte_model: Some(csr_model.clone()),
                    notes: Some(format!(
                        "{topology_note}; policy=pinned_shared; partitions=[{partition_stats}]"
                    )),
                },
                SpmvPerfEvidence {
                    correctness_status: Some("passed".to_string()),
                    correctness_max_abs: Some(shared_pinned_error),
                    correctness_tolerance: Some(1.0e-10),
                    action_parity: Some(
                        "forward=passed;transpose=passed;hermitian=passed".to_string(),
                    ),
                    thread_affinity: Some(topology_note.clone()),
                    numa_policy: Some("shared_pinned".to_string()),
                    numa_storage: Some(pinned_storage),
                    ..Default::default()
                },
                || {
                    std::hint::black_box(
                        numa.matvec_with_policy(&x, NumaSpmvPolicy::SHARED_PINNED),
                    );
                },
            );
            record_spmv_perf_with_evidence(
                SpmvPerfMeta {
                    group: "spmv/numa",
                    id: format!("numa_pinned_replicated_p{num_partitions}/{nn}"),
                    operation: "spmv",
                    matrix_name: "poisson_2d",
                    nrows: nn,
                    ncols: nn,
                    nnz: numa.nnz(),
                    bytes_per_iter: bytes,
                    csr_byte_model: Some(csr_model),
                    notes: Some(format!(
                        "{topology_note}; policy=pinned_replicated; partitions=[{partition_stats}]"
                    )),
                },
                SpmvPerfEvidence {
                    correctness_status: Some("passed".to_string()),
                    correctness_max_abs: Some(replicated_pinned_error),
                    correctness_tolerance: Some(1.0e-10),
                    action_parity: Some(
                        "forward=passed;transpose=passed;hermitian=passed".to_string(),
                    ),
                    thread_affinity: Some(topology_note.clone()),
                    numa_policy: Some("replicated_pinned".to_string()),
                    numa_storage: Some(replicated_storage),
                    ..Default::default()
                },
                || {
                    std::hint::black_box(
                        numa.matvec_with_policy(&x, NumaSpmvPolicy::REPLICATED_PINNED),
                    );
                },
            );
        }

        group.throughput(Throughput::Bytes(bytes));

        // CSR baseline (for side-by-side comparison in same group)
        group.bench_with_input(
            BenchmarkId::new("csr", nn),
            &(&csr, &x),
            |b, &(matrix, x)| b.iter(|| matrix.matvec(x)),
        );

        // NUMA-partitioned
        group.bench_with_input(
            BenchmarkId::new(format!("numa_p{num_partitions}"), nn),
            &(&numa, &x),
            |b, &(matrix, x)| b.iter(|| matrix.matvec(x)),
        );

        group.bench_with_input(
            BenchmarkId::new(format!("numa_pinned_shared_p{num_partitions}"), nn),
            &(&numa, &x),
            |b, &(matrix, x)| {
                b.iter(|| matrix.matvec_with_policy(x, NumaSpmvPolicy::SHARED_PINNED))
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("numa_pinned_replicated_p{num_partitions}"), nn),
            &(&numa, &x),
            |b, &(matrix, x)| {
                b.iter(|| matrix.matvec_with_policy(x, NumaSpmvPolicy::REPLICATED_PINNED))
            },
        );
        bench_plan_variant(&mut group, "numa_plan", nn, &csr, &numa_plan, &x);
        std::hint::black_box(numa_plan_error);
    }

    group.finish();
}

fn bench_planned_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv/plan");
    group.sample_size(ss2_sample_size(50));
    group.warm_up_time(ss2_warm_up_time());
    group.measurement_time(ss2_measurement_time(10));

    for &target_dofs in &[10_000, 100_000, 500_000] {
        let n = matrices::grid_size_2d(target_dofs);
        let csr = matrices::poisson_2d(n);
        let nn = csr.num_rows;
        let x = vec![1.0_f64; nn];
        let setup_start = Instant::now();
        let auto_plan = csr.spmv_plan(SpmvPlanConfig {
            expected_reuse: 50,
            ..Default::default()
        });
        let auto_setup_ms = setup_start.elapsed().as_secs_f64() * 1.0e3;
        let setup_start = Instant::now();
        let compact_plan = csr.spmv_plan(SpmvPlanConfig {
            preference: SpmvPlanPreference::CompactCsrU32,
            ..Default::default()
        });
        let compact_setup_ms = setup_start.elapsed().as_secs_f64() * 1.0e3;
        let setup_start = Instant::now();
        let sell_plan = csr.spmv_plan(SpmvPlanConfig {
            preference: SpmvPlanPreference::SellCSigma,
            sell_c_sigma_chunk_size: 16,
            sell_c_sigma_window: 256,
            ..Default::default()
        });
        let sell_setup_ms = setup_start.elapsed().as_secs_f64() * 1.0e3;

        let bytes = spmv_bytes(nn, nn, csr.nnz());
        let csr_error = assert_direct_gate(&csr, &x, 1.0e-10);
        assert_forward_gate(&csr, &x, &auto_plan.matvec(&x), 1.0e-10);
        assert_forward_gate(&csr, &x, &sell_plan.matvec(&x), 1.0e-10);
        group.throughput(Throughput::Bytes(bytes));

        #[cfg(feature = "cli")]
        {
            let usize_model = perf::csr_spmv_byte_model(
                nn,
                nn,
                csr.nnz(),
                std::mem::size_of::<f64>(),
                std::mem::size_of::<usize>(),
            );
            record_spmv_perf_with_evidence(
                SpmvPerfMeta {
                    group: "spmv/plan",
                    id: format!("csr/{nn}"),
                    operation: "spmv",
                    matrix_name: "poisson_2d",
                    nrows: nn,
                    ncols: nn,
                    nnz: csr.nnz(),
                    bytes_per_iter: usize_model.total_bytes,
                    csr_byte_model: Some(usize_model),
                    notes: Some("raw CSR baseline for planned SpMV comparison".into()),
                },
                SpmvPerfEvidence {
                    correctness_status: Some("passed".to_string()),
                    correctness_max_abs: Some(csr_error),
                    correctness_tolerance: Some(1.0e-10),
                    action_parity: Some("forward=passed".to_string()),
                    ..Default::default()
                },
                || {
                    std::hint::black_box(csr.matvec(&x));
                },
            );
            record_plan_perf("auto", "poisson_2d", &csr, &auto_plan, &x, auto_setup_ms);
            record_plan_perf(
                "compact_u32",
                "poisson_2d",
                &csr,
                &compact_plan,
                &x,
                compact_setup_ms,
            );
            record_plan_perf(
                "sell_c_sigma",
                "poisson_2d",
                &csr,
                &sell_plan,
                &x,
                sell_setup_ms,
            );
        }

        group.bench_with_input(BenchmarkId::new("csr", nn), &(&csr, &x), |b, &(m, x)| {
            b.iter(|| m.matvec(x))
        });
        group.bench_with_input(
            BenchmarkId::new(format!("auto_{:?}", auto_plan.format()), nn),
            &(&auto_plan, &x),
            |b, &(plan, x)| b.iter(|| plan.matvec(x)),
        );
        group.bench_with_input(
            BenchmarkId::new("compact_u32", nn),
            &(&compact_plan, &x),
            |b, &(plan, x)| b.iter(|| plan.matvec(x)),
        );
        bench_plan_variant(&mut group, "sell_c_sigma", nn, &csr, &sell_plan, &x);
    }

    for &n in &[10_000, 100_000] {
        let csr = irregular_rows_matrix(n, n);
        let x = vec![1.0_f64; n];
        let setup_start = Instant::now();
        let auto_plan = csr.spmv_plan(SpmvPlanConfig {
            expected_reuse: 50,
            min_reuse_for_row_bucket: 2,
            min_nnz_for_row_bucket: 1,
            min_row_nnz_cv_for_row_bucket: 0.5,
            min_reuse_for_sell_c_sigma: usize::MAX,
            min_nnz_for_numa: usize::MAX,
            ..Default::default()
        });
        let auto_setup_ms = setup_start.elapsed().as_secs_f64() * 1.0e3;
        let setup_start = Instant::now();
        let row_bucket_plan = csr.spmv_plan(SpmvPlanConfig {
            preference: SpmvPlanPreference::RowBucketedCsr,
            ..Default::default()
        });
        let row_bucket_setup_ms = setup_start.elapsed().as_secs_f64() * 1.0e3;
        let bytes = spmv_bytes(n, n, csr.nnz());
        let csr_error = assert_direct_gate(&csr, &x, 1.0e-10);
        group.throughput(Throughput::Bytes(bytes));

        #[cfg(feature = "cli")]
        {
            let usize_model = perf::csr_spmv_byte_model(
                n,
                n,
                csr.nnz(),
                std::mem::size_of::<f64>(),
                std::mem::size_of::<usize>(),
            );
            record_spmv_perf_with_evidence(
                SpmvPerfMeta {
                    group: "spmv/plan",
                    id: format!("irregular_csr/{n}"),
                    operation: "spmv",
                    matrix_name: "irregular_rows",
                    nrows: n,
                    ncols: n,
                    nnz: csr.nnz(),
                    bytes_per_iter: usize_model.total_bytes,
                    csr_byte_model: Some(usize_model),
                    notes: Some("raw CSR baseline for irregular planned SpMV".into()),
                },
                SpmvPerfEvidence {
                    correctness_status: Some("passed".to_string()),
                    correctness_max_abs: Some(csr_error),
                    correctness_tolerance: Some(1.0e-10),
                    action_parity: Some("forward=passed".to_string()),
                    ..Default::default()
                },
                || {
                    std::hint::black_box(csr.matvec(&x));
                },
            );
            record_plan_perf(
                "irregular_auto",
                "irregular_rows",
                &csr,
                &auto_plan,
                &x,
                auto_setup_ms,
            );
            record_plan_perf(
                "row_bucketed",
                "irregular_rows",
                &csr,
                &row_bucket_plan,
                &x,
                row_bucket_setup_ms,
            );
        }

        group.bench_with_input(
            BenchmarkId::new("irregular_csr", n),
            &(&csr, &x),
            |b, &(m, x)| b.iter(|| m.matvec(x)),
        );
        bench_plan_variant(
            &mut group,
            &format!("irregular_auto_{:?}", auto_plan.format()),
            n,
            &csr,
            &auto_plan,
            &x,
        );
        bench_plan_variant(&mut group, "row_bucketed", n, &csr, &row_bucket_plan, &x);
    }

    for &n in &[10_000, 100_000] {
        let csr = mostly_uniform_band_matrix(n, n, 9);
        let x = vec![1.0_f64; n];
        let setup_start = Instant::now();
        let auto_plan = csr.spmv_plan(SpmvPlanConfig {
            expected_reuse: 50,
            min_reuse_for_bsr: usize::MAX,
            min_reuse_for_sell_c_sigma: 2,
            min_nnz_for_sell_c_sigma: 1,
            min_sell_c_sigma_fill_ratio: 0.75,
            min_nnz_for_numa: usize::MAX,
            ..Default::default()
        });
        let auto_setup_ms = setup_start.elapsed().as_secs_f64() * 1.0e3;
        let setup_start = Instant::now();
        let sell_plan = csr.spmv_plan(SpmvPlanConfig {
            preference: SpmvPlanPreference::SellCSigma,
            ..Default::default()
        });
        let sell_setup_ms = setup_start.elapsed().as_secs_f64() * 1.0e3;
        let bytes = spmv_bytes(n, n, csr.nnz());
        let csr_error = assert_direct_gate(&csr, &x, 1.0e-10);
        group.throughput(Throughput::Bytes(bytes));

        #[cfg(feature = "cli")]
        {
            let usize_model = perf::csr_spmv_byte_model(
                n,
                n,
                csr.nnz(),
                std::mem::size_of::<f64>(),
                std::mem::size_of::<usize>(),
            );
            record_spmv_perf_with_evidence(
                SpmvPerfMeta {
                    group: "spmv/plan",
                    id: format!("uniform_band_csr/{n}"),
                    operation: "spmv",
                    matrix_name: "uniform_band",
                    nrows: n,
                    ncols: n,
                    nnz: csr.nnz(),
                    bytes_per_iter: usize_model.total_bytes,
                    csr_byte_model: Some(usize_model),
                    notes: Some("raw CSR baseline for SELL-C-sigma planned SpMV".into()),
                },
                SpmvPerfEvidence {
                    correctness_status: Some("passed".to_string()),
                    correctness_max_abs: Some(csr_error),
                    correctness_tolerance: Some(1.0e-10),
                    action_parity: Some("forward=passed".to_string()),
                    ..Default::default()
                },
                || {
                    std::hint::black_box(csr.matvec(&x));
                },
            );
            record_plan_perf(
                "uniform_band_auto",
                "uniform_band",
                &csr,
                &auto_plan,
                &x,
                auto_setup_ms,
            );
            record_plan_perf(
                "sell_c_sigma",
                "uniform_band",
                &csr,
                &sell_plan,
                &x,
                sell_setup_ms,
            );
        }

        group.bench_with_input(
            BenchmarkId::new("uniform_band_csr", n),
            &(&csr, &x),
            |b, &(m, x)| b.iter(|| m.matvec(x)),
        );
        bench_plan_variant(
            &mut group,
            &format!("uniform_band_auto_{:?}", auto_plan.format()),
            n,
            &csr,
            &auto_plan,
            &x,
        );
        bench_plan_variant(
            &mut group,
            "uniform_band_sell_c_sigma",
            n,
            &csr,
            &sell_plan,
            &x,
        );
    }

    group.finish();
}

/// Rectangular raw CSR fixture with repeated and unsorted column indices.
/// Keeping the duplicate entries in the raw representation makes this a
/// useful plan benchmark in addition to the native action-matrix test.
fn rectangular_duplicate_matrix() -> CsrMatrix<f64> {
    let rows = 4_096;
    let cols = 8_192;
    let entries_per_row = 7;
    let mut row_ptrs = Vec::with_capacity(rows + 1);
    let mut columns = Vec::with_capacity(rows * entries_per_row);
    let mut values = Vec::with_capacity(rows * entries_per_row);
    row_ptrs.push(0);
    for row in 0..rows {
        let base = (row * 37) % cols;
        let row_columns = [
            (base + 17) % cols,
            base,
            base,
            (base + 3_001) % cols,
            (base + 5) % cols,
            (base + 1_003) % cols,
            (base + 29) % cols,
        ];
        for (offset, column) in row_columns.into_iter().enumerate() {
            columns.push(column);
            values.push(0.5 + (row + offset) as f64 * 0.0025);
        }
        row_ptrs.push(columns.len());
    }
    CsrMatrix::from_raw_parts(rows, cols, row_ptrs, columns, values)
}

fn bench_rectangular_duplicate_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv/fixtures");
    group.sample_size(ss2_sample_size(40));
    group.warm_up_time(ss2_warm_up_time());
    group.measurement_time(ss2_measurement_time(3));

    let csr = rectangular_duplicate_matrix();
    let x: Vec<_> = (0..csr.num_cols)
        .map(|column| 0.25 + (column % 31) as f64 * 0.03125)
        .collect();
    let raw_error = assert_direct_gate(&csr, &x, 1.0e-10);
    let compact_setup_start = Instant::now();
    let compact = csr.spmv_plan(SpmvPlanConfig {
        preference: SpmvPlanPreference::CompactCsrU32,
        expected_reuse: 16,
        ..Default::default()
    });
    let compact_setup_ms = compact_setup_start.elapsed().as_secs_f64() * 1.0e3;
    let compact_error = assert_forward_gate(&csr, &x, &compact.matvec(&x), 1.0e-10);
    let sell_setup_start = Instant::now();
    let sell = csr.spmv_plan(SpmvPlanConfig {
        preference: SpmvPlanPreference::SellCSigma,
        expected_reuse: 16,
        ..Default::default()
    });
    let sell_setup_ms = sell_setup_start.elapsed().as_secs_f64() * 1.0e3;
    let sell_error = assert_forward_gate(&csr, &x, &sell.matvec(&x), 1.0e-10);

    #[cfg(feature = "cli")]
    {
        let raw_model = perf::csr_spmv_byte_model(
            csr.num_rows,
            csr.num_cols,
            csr.nnz(),
            std::mem::size_of::<f64>(),
            std::mem::size_of::<usize>(),
        );
        record_spmv_perf_with_evidence(
            SpmvPerfMeta {
                group: "spmv/fixtures",
                id: "rectangular_duplicate_csr".to_string(),
                operation: "spmv",
                matrix_name: "rectangular_duplicate_unsorted",
                nrows: csr.num_rows,
                ncols: csr.num_cols,
                nnz: csr.nnz(),
                bytes_per_iter: raw_model.total_bytes,
                csr_byte_model: Some(raw_model),
                notes: Some(
                    "raw CSR baseline for rectangular duplicate transformed plans".to_string(),
                ),
            },
            SpmvPerfEvidence {
                correctness_status: Some("passed".to_string()),
                correctness_max_abs: Some(raw_error),
                correctness_tolerance: Some(1.0e-10),
                action_parity: Some("forward=passed".to_string()),
                ..Default::default()
            },
            || {
                std::hint::black_box(csr.matvec(&x));
            },
        );
        record_plan_perf(
            "rectangular_duplicate_compact",
            "rectangular_duplicate_unsorted",
            &csr,
            &compact,
            &x,
            compact_setup_ms,
        );
        record_plan_perf(
            "rectangular_duplicate_sell",
            "rectangular_duplicate_unsorted",
            &csr,
            &sell,
            &x,
            sell_setup_ms,
        );
    }

    group.bench_with_input(
        BenchmarkId::new(
            "rectangular_duplicate_csr",
            format!("{}x{}", csr.num_rows, csr.num_cols),
        ),
        &(&csr, &x),
        |b, &(matrix, x)| b.iter(|| matrix.matvec(x)),
    );
    group.bench_with_input(
        BenchmarkId::new("rectangular_duplicate_compact", csr.num_rows),
        &(&compact, &x),
        |b, &(plan, x)| b.iter(|| plan.matvec(x)),
    );
    group.bench_with_input(
        BenchmarkId::new("rectangular_duplicate_sell", csr.num_rows),
        &(&sell, &x),
        |b, &(plan, x)| b.iter(|| plan.matvec(x)),
    );

    // Keep the correctness values live in non-CLI benchmark builds too.
    std::hint::black_box((raw_error, compact_error, sell_error));
    group.finish();
}

fn bench_bsr_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv/bsr");
    group.sample_size(ss2_sample_size(50));
    group.warm_up_time(ss2_warm_up_time());
    group.measurement_time(ss2_measurement_time(3));

    for &(block_size, preference, label) in &[
        (2, SpmvPlanPreference::Bsr2, "bsr2"),
        (3, SpmvPlanPreference::Bsr3, "bsr3"),
        (4, SpmvPlanPreference::Bsr4, "bsr4"),
    ] {
        let csr = dense_block_diagonal_matrix(2048, block_size);
        let x = vec![1.0_f64; csr.num_cols];
        let setup_start = Instant::now();
        let plan = csr.spmv_plan(SpmvPlanConfig {
            preference,
            ..Default::default()
        });
        let setup_ms = setup_start.elapsed().as_secs_f64() * 1.0e3;
        assert_eq!(format!("{:?}", plan.format()).to_lowercase(), label);
        group.throughput(Throughput::Bytes(plan.report().chosen_bytes_per_apply));

        let actual = plan.matvec(&x);
        let max_abs = assert_forward_gate(&csr, &x, &actual, 1.0e-10);

        #[cfg(feature = "cli")]
        {
            let block_input = block_fixture::<f64>(csr.num_cols, 4);
            let raw_block_error = assert_csr_block_gate(&csr, &block_input, 1.0e-10);
            let planned_block_error = assert_plan_block_gate(&csr, &plan, &block_input, 1.0e-10);
            let raw_error = assert_direct_gate(&csr, &x, 1.0e-10);
            let raw_model = perf::csr_spmv_byte_model(
                csr.num_rows,
                csr.num_cols,
                csr.nnz(),
                std::mem::size_of::<f64>(),
                std::mem::size_of::<usize>(),
            );
            record_spmv_perf_with_evidence(
                SpmvPerfMeta {
                    group: "spmv/plan",
                    id: format!("csr_for_{label}/{}", csr.num_rows),
                    operation: "spmv",
                    matrix_name: "dense_block_diagonal",
                    nrows: csr.num_rows,
                    ncols: csr.num_cols,
                    nnz: csr.nnz(),
                    bytes_per_iter: raw_model.total_bytes,
                    csr_byte_model: Some(raw_model),
                    notes: Some(format!("raw CSR baseline for {label} transformed format")),
                },
                SpmvPerfEvidence {
                    selected_kernel: Some(selected_spmv_kernel::<f64>().name().to_string()),
                    correctness_status: Some("passed".to_string()),
                    correctness_max_abs: Some(raw_error),
                    correctness_tolerance: Some(1.0e-10),
                    action_parity: Some("forward=passed".to_string()),
                    ..Default::default()
                },
                || {
                    std::hint::black_box(csr.matvec(&x));
                },
            );
            record_plan_perf_typed(
                label,
                "dense_block_diagonal",
                &csr,
                &plan,
                &x,
                setup_ms,
                16,
                max_abs,
                1.0e-10,
            );

            let mut raw_output = DenseMatrix::zeros(csr.num_rows, block_input.ncols());
            record_block_perf_typed(
                &format!("csr_for_{label}"),
                "dense_block_diagonal",
                &csr,
                &block_input,
                None,
                raw_block_error,
                1.0e-10,
                selected_spmv_kernel::<f64>().name(),
                format!("raw CSR block baseline for {label} transformed format"),
                || {
                    csr.spmm_into(&block_input, &mut raw_output);
                    std::hint::black_box(raw_output.as_slice());
                },
            );
            let mut planned_output = DenseMatrix::zeros(csr.num_rows, block_input.ncols());
            record_block_perf_typed(
                label,
                "dense_block_diagonal",
                &csr,
                &block_input,
                Some(setup_ms),
                planned_block_error,
                1.0e-10,
                plan.report().selected_kernel.name(),
                plan.report().summary(),
                || {
                    plan.apply_block(&block_input, &mut planned_output);
                    std::hint::black_box(planned_output.as_slice());
                },
            );
        }

        bench_plan_variant(&mut group, label, csr.num_rows, &csr, &plan, &x);
    }

    group.finish();
}

fn bench_one_scalar_type<T: sonium_solvers::ComplexField + 'static>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    label: &str,
    matrix: CsrMatrix<T>,
    x: Vec<T>,
) {
    let tolerance = scalar_tolerance::<T>();
    let csr_actual = matrix
        .try_matvec(&x)
        .expect("checked raw CSR correctness gate");
    let csr_error = assert_forward_gate(&matrix, &x, &csr_actual, tolerance);

    let compact_setup_start = Instant::now();
    let compact = matrix.spmv_plan(SpmvPlanConfig {
        preference: SpmvPlanPreference::CompactCsrU32,
        ..Default::default()
    });
    let compact_setup_ms = compact_setup_start.elapsed().as_secs_f64() * 1.0e3;
    let compact_actual = compact.matvec(&x);
    let compact_error = assert_forward_gate(&matrix, &x, &compact_actual, tolerance);

    let sell_setup_start = Instant::now();
    let sell = matrix.spmv_plan(SpmvPlanConfig {
        preference: SpmvPlanPreference::SellCSigma,
        ..Default::default()
    });
    let sell_setup_ms = sell_setup_start.elapsed().as_secs_f64() * 1.0e3;
    let sell_actual = sell.matvec(&x);
    let sell_error = assert_forward_gate(&matrix, &x, &sell_actual, tolerance);

    #[cfg(feature = "cli")]
    {
        let block_input = block_fixture::<T>(matrix.num_cols, 4);
        let csr_block_error = assert_csr_block_gate(&matrix, &block_input, tolerance);
        let compact_block_error =
            assert_plan_block_gate(&matrix, &compact, &block_input, tolerance);
        let sell_block_error = assert_plan_block_gate(&matrix, &sell, &block_input, tolerance);
        let byte_model = perf::csr_spmv_byte_model(
            matrix.num_rows,
            matrix.num_cols,
            matrix.nnz(),
            std::mem::size_of::<T>(),
            std::mem::size_of::<usize>(),
        );
        record_spmv_perf_with_evidence(
            SpmvPerfMeta {
                group: "spmv/types",
                id: format!("csr/{label}"),
                operation: "spmv",
                matrix_name: "typed_regular",
                nrows: matrix.num_rows,
                ncols: matrix.num_cols,
                nnz: matrix.nnz(),
                bytes_per_iter: byte_model.total_bytes,
                csr_byte_model: Some(byte_model),
                notes: Some("raw CSR baseline for typed plan comparison".into()),
            },
            SpmvPerfEvidence {
                scalar: Some(scalar_label::<T>().to_string()),
                reuse_count: Some(16),
                correctness_status: Some("passed".to_string()),
                correctness_max_abs: Some(csr_error),
                correctness_tolerance: Some(tolerance),
                action_parity: Some("forward=passed".to_string()),
                ..Default::default()
            },
            || {
                std::hint::black_box(matrix.matvec(&x));
            },
        );
        record_plan_perf_typed(
            "compact_u32",
            "typed_regular",
            &matrix,
            &compact,
            &x,
            compact_setup_ms,
            16,
            compact_error,
            tolerance,
        );
        record_plan_perf_typed(
            "sell_c_sigma",
            "typed_regular",
            &matrix,
            &sell,
            &x,
            sell_setup_ms,
            16,
            sell_error,
            tolerance,
        );
        let mut csr_block_output = DenseMatrix::zeros(matrix.num_rows, block_input.ncols());
        record_block_perf_typed(
            "csr",
            "typed_regular",
            &matrix,
            &block_input,
            None,
            csr_block_error,
            tolerance,
            selected_spmv_kernel::<T>().name(),
            "raw CSR block baseline for typed plan comparison".to_string(),
            || {
                matrix.spmm_into(&block_input, &mut csr_block_output);
                std::hint::black_box(csr_block_output.as_slice());
            },
        );
        let mut compact_block_output = DenseMatrix::zeros(matrix.num_rows, block_input.ncols());
        record_block_perf_typed(
            "compact_u32",
            "typed_regular",
            &matrix,
            &block_input,
            Some(compact_setup_ms),
            compact_block_error,
            tolerance,
            compact.report().selected_kernel.name(),
            compact.report().summary(),
            || {
                compact.apply_block(&block_input, &mut compact_block_output);
                std::hint::black_box(compact_block_output.as_slice());
            },
        );
        let mut sell_block_output = DenseMatrix::zeros(matrix.num_rows, block_input.ncols());
        record_block_perf_typed(
            "sell_c_sigma",
            "typed_regular",
            &matrix,
            &block_input,
            Some(sell_setup_ms),
            sell_block_error,
            tolerance,
            sell.report().selected_kernel.name(),
            sell.report().summary(),
            || {
                sell.apply_block(&block_input, &mut sell_block_output);
                std::hint::black_box(sell_block_output.as_slice());
            },
        );
    }

    group.bench_with_input(
        BenchmarkId::new("csr", label),
        &(&matrix, &x),
        |b, &(matrix, x)| b.iter(|| matrix.matvec(x)),
    );
    group.bench_with_input(
        BenchmarkId::new("compact_u32", label),
        &(&compact, &x),
        |b, &(plan, x)| b.iter(|| plan.matvec(x)),
    );
    group.bench_with_input(
        BenchmarkId::new("sell_c_sigma", label),
        &(&sell, &x),
        |b, &(plan, x)| b.iter(|| plan.matvec(x)),
    );
}

fn bench_scalar_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv/types");
    group.sample_size(ss2_sample_size(50));
    group.warm_up_time(ss2_warm_up_time());
    group.measurement_time(ss2_measurement_time(3));

    let n = 8_192;
    let triplet_indices = (0..n).flat_map(|row| {
        (0..8).map(move |offset| {
            let col = (row + offset * 257) % n;
            (row, col, offset)
        })
    });
    let real_triplets = triplet_indices
        .clone()
        .map(|(row, col, offset)| (row, col, 1.0_f32 + offset as f32 * 0.125))
        .collect();
    let f64_triplets = triplet_indices
        .clone()
        .map(|(row, col, offset)| (row, col, 1.0_f64 + offset as f64 * 0.125))
        .collect();
    let complex32_triplets = triplet_indices
        .clone()
        .map(|(row, col, offset)| (row, col, Complex32::new(1.0 + offset as f32 * 0.125, -0.25)))
        .collect();
    let complex64_triplets = triplet_indices
        .map(|(row, col, offset)| (row, col, Complex64::new(1.0 + offset as f64 * 0.125, -0.25)))
        .collect();

    bench_one_scalar_type(
        &mut group,
        "f32",
        CsrMatrix::from_triplets(n, n, real_triplets),
        vec![1.0_f32; n],
    );
    bench_one_scalar_type(
        &mut group,
        "f64",
        CsrMatrix::from_triplets(n, n, f64_triplets),
        vec![1.0_f64; n],
    );
    bench_one_scalar_type(
        &mut group,
        "complex32",
        CsrMatrix::from_triplets(n, n, complex32_triplets),
        vec![Complex32::new(1.0, 0.5); n],
    );
    bench_one_scalar_type(
        &mut group,
        "complex64",
        CsrMatrix::from_triplets(n, n, complex64_triplets),
        vec![Complex64::new(1.0, 0.5); n],
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// SpMV size scaling (cache effects)
// ---------------------------------------------------------------------------

fn bench_spmv_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv/scaling");
    group.sample_size(ss2_sample_size(50));
    group.warm_up_time(ss2_warm_up_time());
    group.measurement_time(ss2_measurement_time(10));

    // Sweep from ~100 to ~250K DOFs on 2D Poisson
    for n in [10, 20, 32, 50, 80, 100, 160, 250, 500] {
        let matrix = matrices::poisson_2d(n);
        let nn = matrix.num_rows;
        let x = vec![1.0_f64; nn];
        let bytes = spmv_bytes(nn, nn, matrix.nnz());
        let correctness_error = assert_direct_gate(&matrix, &x, 1.0e-10);

        #[cfg(feature = "cli")]
        let csr_model = perf::csr_spmv_byte_model(
            nn,
            nn,
            matrix.nnz(),
            std::mem::size_of::<f64>(),
            std::mem::size_of::<usize>(),
        );
        #[cfg(feature = "cli")]
        record_spmv_perf_with_evidence(
            SpmvPerfMeta {
                group: "spmv/scaling",
                id: nn.to_string(),
                operation: "spmv",
                matrix_name: "poisson_2d",
                nrows: nn,
                ncols: nn,
                nnz: matrix.nnz(),
                bytes_per_iter: bytes,
                csr_byte_model: Some(csr_model),
                notes: Some("size scaling sweep".into()),
            },
            SpmvPerfEvidence {
                correctness_status: Some("passed".to_string()),
                correctness_max_abs: Some(correctness_error),
                correctness_tolerance: Some(1.0e-10),
                action_parity: Some("forward=passed".to_string()),
                ..Default::default()
            },
            || {
                std::hint::black_box(matrix.matvec(&x));
            },
        );

        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(
            BenchmarkId::from_parameter(nn),
            &(matrix, x),
            |b, (m, x)| b.iter(|| m.matvec(x)),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// SpMV: skewed row lengths (nnz-balanced vs equal-rows partitioning)
// ---------------------------------------------------------------------------

/// Power-law skewed matrix: row degrees follow `deg_i ∝ (i+1)^(-0.7)`
/// normalized to ~`target_nnz` non-zeros, heavy rows clustered at the top —
/// the worst case for an equal-rows parallel split.
fn power_law_skewed_matrix(rows: usize, cols: usize, target_nnz: usize) -> CsrMatrix<f64> {
    let weights: Vec<f64> = (0..rows).map(|i| ((i + 1) as f64).powf(-0.7)).collect();
    let total: f64 = weights.iter().sum();
    let scale = target_nnz as f64 / total;
    let mut triplets = Vec::with_capacity(target_nnz + rows);
    for (i, &w) in weights.iter().enumerate() {
        let deg = ((w * scale).round() as usize).clamp(1, cols);
        for k in 0..deg {
            let col = (i.wrapping_mul(2_654_435_761) + k * 40_503) % cols;
            triplets.push((i, col, 1.0 + (k % 7) as f64));
        }
    }
    CsrMatrix::from_triplets(rows, cols, triplets)
}

fn bench_skewed_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv/skewed");
    group.sample_size(ss2_sample_size(50));
    group.warm_up_time(ss2_warm_up_time());
    group.measurement_time(ss2_measurement_time(3));

    let n = 8_192;
    let matrix = power_law_skewed_matrix(n, n, 200_000);
    let x = vec![1.0_f64; n];
    let bytes = spmv_bytes(n, n, matrix.nnz());
    let expected_forward = scalar_csr_matvec(&matrix, &x);
    let expected_transpose = scalar_csr_adjoint_matvec(&matrix, &x, false);
    let actual_forward = matrix
        .try_matvec(&x)
        .expect("checked skewed CSR correctness gate");
    let actual_transpose = matrix
        .try_matvec_transpose(&x)
        .expect("checked skewed CSR transpose correctness gate");
    let forward_error = max_abs_error(&expected_forward, &actual_forward);
    let transpose_error = max_abs_error(&expected_transpose, &actual_transpose);
    assert!(forward_error <= 1.0e-10);
    assert!(transpose_error <= 1.0e-10);

    #[cfg(feature = "cli")]
    {
        let byte_model = perf::csr_spmv_byte_model(
            n,
            n,
            matrix.nnz(),
            std::mem::size_of::<f64>(),
            std::mem::size_of::<usize>(),
        );
        record_spmv_perf_with_evidence(
            SpmvPerfMeta {
                group: "spmv/skewed",
                id: format!("forward/{n}/nnz{}", matrix.nnz()),
                operation: "spmv",
                matrix_name: "power_law_skewed_rows",
                nrows: n,
                ncols: n,
                nnz: matrix.nnz(),
                bytes_per_iter: byte_model.total_bytes,
                csr_byte_model: Some(byte_model.clone()),
                notes: Some("skewed-row raw CSR bandwidth sample".to_string()),
            },
            SpmvPerfEvidence {
                correctness_status: Some("passed".to_string()),
                correctness_max_abs: Some(forward_error),
                correctness_tolerance: Some(1.0e-10),
                action_parity: Some("forward=passed".to_string()),
                ..Default::default()
            },
            || {
                std::hint::black_box(matrix.matvec(&x));
            },
        );
        record_spmv_perf_with_evidence(
            SpmvPerfMeta {
                group: "spmv/skewed",
                id: format!("transpose/{n}/nnz{}", matrix.nnz()),
                operation: "spmv_transpose",
                matrix_name: "power_law_skewed_rows",
                nrows: n,
                ncols: n,
                nnz: matrix.nnz(),
                bytes_per_iter: byte_model.total_bytes,
                csr_byte_model: Some(byte_model),
                notes: Some("skewed-row raw CSR transpose bandwidth sample".to_string()),
            },
            SpmvPerfEvidence {
                correctness_status: Some("passed".to_string()),
                correctness_max_abs: Some(transpose_error),
                correctness_tolerance: Some(1.0e-10),
                action_parity: Some("transpose=passed".to_string()),
                ..Default::default()
            },
            || {
                std::hint::black_box(matrix.matvec_transpose(&x));
            },
        );
    }
    group.throughput(Throughput::Bytes(bytes));

    group.bench_with_input(
        BenchmarkId::new("matvec", format!("{}x{}/nnz{}", n, n, matrix.nnz())),
        &(&matrix, &x),
        |b, &(m, x)| b.iter(|| m.matvec(x)),
    );
    group.bench_with_input(
        BenchmarkId::new(
            "matvec_transpose",
            format!("{}x{}/nnz{}", n, n, matrix.nnz()),
        ),
        &(&matrix, &x),
        |b, &(m, x)| b.iter(|| m.matvec_transpose(x)),
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// SpMM: one matrix walk for k right-hand sides vs k separate SpMVs
// ---------------------------------------------------------------------------

fn bench_spmm(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv/spmm");
    group.sample_size(ss2_sample_size(50));
    group.warm_up_time(ss2_warm_up_time());
    group.measurement_time(ss2_measurement_time(3));

    // 3D Laplacian at FEM-like sizes; k = 2/4/8 right-hand sides.
    for &target_dofs in &[10_000, 100_000] {
        let n = matrices::grid_size_3d(target_dofs);
        let matrix = matrices::laplacian_3d(n);
        let nn = matrix.num_rows;

        for &k in &[2usize, 4, 8] {
            let x_block = sonium_solvers::dense::DenseMatrix::from_fn(nn, k, |i, j| {
                ((i * 31 + j * 17) % 23) as f64 - 11.0
            });
            let x_cols: Vec<Vec<f64>> = (0..k).map(|j| x_block.column_to_vec(j)).collect();
            let mut y_block = sonium_solvers::dense::DenseMatrix::from_elem(nn, k, 0.0f64);
            let mut y_col = vec![0.0f64; nn];
            for x_col in &x_cols {
                assert_direct_gate(&matrix, x_col, 1.0e-10);
            }
            matrix.spmm_into(&x_block, &mut y_block);
            for (column, x_column) in x_cols.iter().enumerate() {
                let actual = y_block.column_to_vec(column);
                let expected = scalar_csr_matvec(&matrix, x_column);
                assert!(max_abs_error(&expected, &actual) <= 1.0e-10);
            }
            #[cfg(feature = "cli")]
            {
                let block_error = assert_csr_block_gate(&matrix, &x_block, 1.0e-10);
                let mut separate_output = vec![0.0f64; nn];
                record_block_perf_typed(
                    "separate_spmv",
                    "laplacian_3d",
                    &matrix,
                    &x_block,
                    None,
                    block_error,
                    1.0e-10,
                    selected_spmv_kernel::<f64>().name(),
                    "raw CSR block baseline: separate single-RHS applications".to_string(),
                    || {
                        for x_column in &x_cols {
                            matrix.matvec_into_buf(x_column, &mut separate_output);
                        }
                        std::hint::black_box(&separate_output);
                    },
                );
                record_block_perf_typed(
                    "spmm",
                    "laplacian_3d",
                    &matrix,
                    &x_block,
                    None,
                    block_error,
                    1.0e-10,
                    selected_spmv_kernel::<f64>().name(),
                    "raw CSR block workload: fused SpMM".to_string(),
                    || {
                        matrix.spmm_into(&x_block, &mut y_block);
                        std::hint::black_box(y_block.as_slice());
                    },
                );
            }

            group.bench_with_input(
                BenchmarkId::new("separate_spmv", format!("n{nn}/k{k}")),
                &(&matrix, &x_cols),
                |b, &(m, x_cols)| {
                    b.iter(|| {
                        for (j, xj) in x_cols.iter().enumerate() {
                            m.matvec_into_buf(xj, &mut y_col);
                            std::hint::black_box((j, &y_col));
                        }
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new("spmm", format!("n{nn}/k{k}")),
                &(&matrix, &x_block),
                |b, &(m, x_block)| {
                    b.iter(|| m.spmm_into(x_block, &mut y_block));
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    name = spmv_benches;
    config = Criterion::default();
    targets = bench_csr_spmv, bench_numa_spmv, bench_planned_spmv, bench_rectangular_duplicate_spmv, bench_bsr_spmv, bench_scalar_types, bench_spmv_scaling, bench_skewed_spmv, bench_spmm
);
criterion_main!(spmv_benches);
