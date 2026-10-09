//! Standalone threshold evidence for the sparse SpMV planner.
//!
//! The probe sweeps each documented inclusive predicate at below/at/above
//! boundaries. Every row has an independent scalar correctness gate before
//! raw CSR and planned steady-state timings are collected.

use serde_json::{Value, json};
use sonium_solvers_core::traits::LinearOperator;
use sonium_solvers_sparse::{
    CsrMatrix, SpmvPlan, SpmvPlanConfig, SpmvPlanFormat, SpmvPlanPreference,
};
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const SCHEMA: u32 = 2;
const FIXTURE_VERSION: u32 = 1;
const TOLERANCE: f64 = 1.0e-9;
const SMALL_REPEATS: usize = 12;
const LARGE_REPEATS: usize = 4;

fn evidence_path(raw: &str) -> PathBuf {
    let path = Path::new(raw);
    if path.is_absolute() {
        return path.to_path_buf();
    }
    // Cargo runs standalone bench binaries with a package-dependent current
    // directory. Resolve relative output paths from the repository root so a
    // CI artifact cannot silently land under a crate-local `target/`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../")
        .join(path)
}

fn matrix_from_lengths(rows: usize, cols: usize, lengths: &[usize]) -> CsrMatrix<f64> {
    assert_eq!(rows, lengths.len());
    let mut row_ptrs = Vec::with_capacity(rows + 1);
    let mut columns = Vec::new();
    let mut values = Vec::new();
    row_ptrs.push(0);
    for (row, &length) in lengths.iter().enumerate() {
        assert!(length <= cols);
        for offset in 0..length {
            columns.push((row * 17 + offset * 13) % cols);
            values.push(1.0 + ((row + offset) % 19) as f64 * 0.01);
        }
        row_ptrs.push(columns.len());
    }
    CsrMatrix::from_raw_parts(rows, cols, row_ptrs, columns, values)
}

fn uniform_matrix(rows: usize, cols: usize, entries_per_row: usize) -> CsrMatrix<f64> {
    matrix_from_lengths(rows, cols, &vec![entries_per_row; rows])
}

fn block_diagonal_matrix(blocks: usize, block_size: usize) -> CsrMatrix<f64> {
    let rows = blocks * block_size;
    let mut row_ptrs = Vec::with_capacity(rows + 1);
    let mut columns = Vec::with_capacity(rows * block_size);
    let mut values = Vec::with_capacity(rows * block_size);
    row_ptrs.push(0);
    for row in 0..rows {
        let block_start = row / block_size * block_size;
        for offset in 0..block_size {
            columns.push(block_start + offset);
            values.push(1.0 + ((row + offset) % 19) as f64 * 0.01);
        }
        row_ptrs.push(columns.len());
    }
    CsrMatrix::from_raw_parts(rows, rows, row_ptrs, columns, values)
}

fn row_parallel_matrix(nnz: usize, rows: usize, cols: usize) -> CsrMatrix<f64> {
    assert!(nnz >= rows * 12 && nnz <= rows * 13);
    let extra = nnz - rows * 12;
    let lengths: Vec<_> = (0..rows).map(|row| 12 + usize::from(row < extra)).collect();
    matrix_from_lengths(rows, cols, &lengths)
}

fn scalar_forward(matrix: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    (0..matrix.num_rows)
        .map(|row| {
            (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                .map(|index| matrix.values[index] * x[matrix.col_indices[index]])
                .sum()
        })
        .collect()
}

fn checksum(values: &[f64]) -> f64 {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| *value * (index + 1) as f64)
        .sum()
}

fn matrix_checksum(matrix: &CsrMatrix<f64>) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for value in [matrix.num_rows, matrix.num_cols, matrix.values.len()] {
        hash = hash.rotate_left(7) ^ value as u64;
    }
    for &index in &matrix.row_ptrs {
        hash = hash.rotate_left(7) ^ index as u64;
    }
    for &index in &matrix.col_indices {
        hash = hash.rotate_left(7) ^ index as u64;
    }
    for &value in &matrix.values {
        hash = hash.rotate_left(7) ^ value.to_bits();
    }
    format!("{hash:016x}")
}

fn max_abs_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).abs())
        .fold(0.0, f64::max)
}

fn time_repeated<F>(mut operation: F, repeats: usize) -> (u128, f64)
where
    F: FnMut() -> Vec<f64>,
{
    let start = Instant::now();
    let mut final_output = None;
    for _ in 0..repeats {
        // Keep the timed region limited to the actual action.  Reduction of
        // the final output is outside the steady-state timing path and still
        // prevents the optimizer from discarding the measured result.
        final_output = Some(black_box(operation()));
    }
    let observed_checksum = final_output.as_deref().map(checksum).unwrap_or_default();
    (
        start.elapsed().as_nanos() / repeats as u128,
        observed_checksum,
    )
}

fn isolated_config() -> SpmvPlanConfig {
    SpmvPlanConfig {
        preference: SpmvPlanPreference::Auto,
        expected_reuse: 1,
        min_reuse_for_compact_indices: usize::MAX,
        min_saved_index_bytes_for_compact: usize::MAX,
        min_reuse_for_bsr: usize::MAX,
        min_saved_bytes_for_bsr: usize::MAX,
        min_bsr_block_fill_ratio: 2.0,
        min_reuse_for_row_bucket: usize::MAX,
        min_nnz_for_row_bucket: usize::MAX,
        min_row_nnz_cv_for_row_bucket: f64::MAX,
        min_reuse_for_sell_c_sigma: usize::MAX,
        min_nnz_for_sell_c_sigma: usize::MAX,
        min_sell_c_sigma_fill_ratio: 2.0,
        min_nnz_for_numa: usize::MAX,
        numa_partitions: Some(1),
        ..SpmvPlanConfig::default()
    }
}

fn config_for(family: &str, expected_reuse: usize) -> SpmvPlanConfig {
    let mut config = isolated_config();
    config.expected_reuse = expected_reuse;
    match family {
        "compact_index" => {
            config.min_reuse_for_compact_indices = 8;
            config.min_saved_index_bytes_for_compact = 0;
        }
        "bsr" => {
            config.min_reuse_for_bsr = 4;
            config.min_saved_bytes_for_bsr = 0;
            config.min_bsr_block_fill_ratio = 0.0;
        }
        "sell_c_sigma" => {
            config.min_reuse_for_sell_c_sigma = 8;
            config.min_nnz_for_sell_c_sigma = 0;
            config.min_sell_c_sigma_fill_ratio = 0.0;
        }
        "row_bucket" => {
            config.min_reuse_for_row_bucket = 4;
            config.min_nnz_for_row_bucket = 0;
            config.min_row_nnz_cv_for_row_bucket = 0.0;
        }
        "numa" => {
            config.min_nnz_for_numa = 0;
            config.numa_partitions = Some(2);
        }
        "row_parallel" => {}
        _ => panic!("unknown threshold family {family}"),
    }
    config
}

fn expected_format(family: &str, predicate_passes: bool) -> SpmvPlanFormat {
    if !predicate_passes {
        return SpmvPlanFormat::Csr;
    }
    match family {
        "compact_index" => SpmvPlanFormat::CompactCsrU32,
        "bsr" => SpmvPlanFormat::Bsr4,
        "sell_c_sigma" => SpmvPlanFormat::SellCSigma,
        "row_bucket" => SpmvPlanFormat::RowBucketedCsr,
        "numa" => SpmvPlanFormat::NumaCsr,
        "row_parallel" => SpmvPlanFormat::Csr,
        _ => panic!("unknown threshold family {family}"),
    }
}

fn run_case(
    family: &str,
    matrix: &CsrMatrix<f64>,
    threshold_dimension: &str,
    threshold_value: Value,
    threshold_side: &str,
    config: SpmvPlanConfig,
    predicate_passes: bool,
) -> Value {
    let setup_start = Instant::now();
    let plan = SpmvPlan::try_from_csr(matrix, config).expect("valid threshold plan");
    let setup_ns = setup_start.elapsed().as_nanos();
    let x: Vec<_> = (0..matrix.num_cols)
        .map(|column| 1.0 + (column % 23) as f64 * 0.005)
        .collect();
    let expected = scalar_forward(matrix, &x);

    // Correctness is deliberately checked before any timing starts.
    let raw = matrix.try_matvec(&x).expect("checked raw CSR action");
    let planned = plan.try_apply(&x).expect("checked planned action");
    let raw_error = max_abs_error(&raw, &expected);
    let planned_error = max_abs_error(&planned, &expected);
    assert!(raw_error <= TOLERANCE, "raw CSR error {raw_error:e}");
    assert!(
        planned_error <= TOLERANCE,
        "planned error {planned_error:e}"
    );
    assert!(raw.iter().all(|value| value.is_finite()));
    assert!(planned.iter().all(|value| value.is_finite()));

    // Validate planner selection before either timing call.  A benchmark row
    // must never publish a fast but incorrectly selected plan.
    let selected_format = plan.format();
    let expected_format = expected_format(family, predicate_passes);
    assert_eq!(
        selected_format, expected_format,
        "{family} {threshold_dimension}={threshold_value:?}"
    );

    let repeats = if matrix.values.len() >= 200_000 {
        LARGE_REPEATS
    } else {
        SMALL_REPEATS
    };
    let (raw_ns, raw_checksum) = time_repeated(|| matrix.matvec(&x), repeats);
    let (planned_ns, planned_checksum) = time_repeated(|| plan.matvec(&x), repeats);
    let report = plan.report();
    let analysis = &report.analysis;
    let runtime_bandwidth_threads =
        sonium_solvers_sys::runtime::bandwidth_threads_for_nnz(matrix.values.len());
    let runtime_parallel_selected =
        matrix.values.len() >= 50_000 && matrix.num_rows >= 4_000 && runtime_bandwidth_threads > 1;
    let execution_mode = if family == "row_parallel" {
        if runtime_parallel_selected {
            "row_parallel"
        } else {
            "sequential"
        }
    } else {
        "format_specific"
    };
    json!({
        "family": family,
        "dtype": "f64",
        "action": "forward",
        "reference_oracle": "independent_scalar_csr",
        "matrix_checksum": matrix_checksum(matrix),
        "fixture": {
            "rows": matrix.num_rows,
            "cols": matrix.num_cols,
            "nnz": matrix.values.len(),
            "empty_rows": analysis.empty_rows,
            "row_nnz_cv": analysis.row_nnz_cv,
            "conditioning_proxy": analysis.row_nnz_cv,
        },
        "threshold_dimension": threshold_dimension,
        "threshold_value": threshold_value,
        "threshold_side": threshold_side,
        "threshold_side_semantics": "measured metric relative to configured threshold",
        "predicate_passes": predicate_passes,
        "runtime_bandwidth_threads": runtime_bandwidth_threads,
        "runtime_parallel_selected": runtime_parallel_selected,
        "execution_mode": execution_mode,
        "selected_format": format!("{:?}", selected_format),
        "expected_format": format!("{:?}", expected_format),
        "format_match": selected_format == expected_format,
        "selected_kernel": report.selected_kernel.name(),
        "features": report.features.clone(),
        "planner_summary": report.summary(),
        "planner_reasons": report.reasons.clone(),
        "setup_ns": setup_ns,
        "repeats": repeats,
        "raw_csr_ns": raw_ns,
        "planned_ns": planned_ns,
        "speedup_raw_over_plan": raw_ns as f64 / planned_ns.max(1) as f64,
        "correctness_status": "passed",
        "correctness_before_timing": true,
        "raw_max_abs": raw_error,
        "planned_max_abs": planned_error,
        "tolerance": TOLERANCE,
        "raw_checksum": raw_checksum,
        "planned_checksum": planned_checksum,
        "checksum_delta": (raw_checksum - planned_checksum).abs(),
    })
}

fn push_boundaries(
    records: &mut Vec<Value>,
    family: &str,
    matrix: &CsrMatrix<f64>,
    dimension: &str,
    cases: &[(Value, &str, bool, SpmvPlanConfig)],
) {
    for (value, side, passes, config) in cases {
        records.push(run_case(
            family,
            matrix,
            dimension,
            value.clone(),
            side,
            config.clone(),
            *passes,
        ));
    }
}

fn rustc_verbose() -> String {
    Command::new("rustc")
        .arg("-Vv")
        .output()
        .ok()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|| "unavailable".to_string())
}

fn cpu_field(prefix: &str) -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                line.strip_prefix(prefix)
                    .map(|value| value.trim().to_string())
            })
        })
        .unwrap_or_else(|| "unavailable".to_string())
}

#[cfg(target_arch = "x86_64")]
fn runtime_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
fn runtime_avx2() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn runtime_fma() -> bool {
    is_x86_feature_detected!("fma")
}

#[cfg(not(target_arch = "x86_64"))]
fn runtime_fma() -> bool {
    false
}

fn main() {
    let compact = uniform_matrix(4096, 4096, 8);
    let bsr = block_diagonal_matrix(1024, 4);
    let sell = uniform_matrix(8192, 8192, 16);
    let row_bucket = matrix_from_lengths(
        8192,
        8192,
        &(0..8192)
            .map(|row| if row % 2 == 0 { 0 } else { 64 })
            .collect::<Vec<_>>(),
    );
    let numa = uniform_matrix(4096, 4096, 16);

    let compact_baseline = SpmvPlan::try_from_csr(&compact, config_for("compact_index", 16))
        .expect("compact baseline");
    let compact_saved = compact_baseline
        .report()
        .analysis
        .saved_index_bytes_per_apply
        .expect("compact index savings");
    assert!(compact_saved > 0);

    let bsr_baseline = SpmvPlan::try_from_csr(&bsr, config_for("bsr", 16)).expect("BSR baseline");
    assert_eq!(bsr_baseline.format(), SpmvPlanFormat::Bsr4);
    let bsr_analysis = bsr_baseline.report().analysis.bsr4.expect("BSR analysis");
    let bsr_saved = bsr_analysis.saved_bytes_per_apply;
    assert!(bsr_saved > 0);

    let sell_baseline =
        SpmvPlan::try_from_csr(&sell, config_for("sell_c_sigma", 16)).expect("SELL baseline");
    let sell_analysis = sell_baseline
        .report()
        .analysis
        .sell_c_sigma
        .expect("SELL analysis");
    assert_eq!(sell_baseline.format(), SpmvPlanFormat::SellCSigma);
    let row_baseline = SpmvPlan::try_from_csr(&row_bucket, config_for("row_bucket", 16))
        .expect("row bucket baseline");
    let row_cv = row_baseline.report().analysis.row_nnz_cv;
    assert!(row_cv >= 1.0);
    let numa_nnz = numa.values.len();
    let force_scalar = std::env::var_os("SONIUM_SPARSE_FORCE_SCALAR")
        .is_some_and(|value| value == "1" || value == "true");
    let mut records = Vec::new();

    push_boundaries(
        &mut records,
        "compact_index",
        &compact,
        "expected_reuse",
        &[
            (json!(7), "below", false, config_for("compact_index", 7)),
            (json!(8), "at", true, config_for("compact_index", 8)),
            (json!(9), "above", true, config_for("compact_index", 9)),
        ],
    );
    push_boundaries(
        &mut records,
        "compact_index",
        &compact,
        "saved_index_bytes_per_apply",
        &[
            (json!(compact_saved - 1), "above", true, {
                let mut c = config_for("compact_index", 16);
                c.min_saved_index_bytes_for_compact = compact_saved as usize - 1;
                c
            }),
            (json!(compact_saved), "at", true, {
                let mut c = config_for("compact_index", 16);
                c.min_saved_index_bytes_for_compact = compact_saved as usize;
                c
            }),
            (json!(compact_saved + 1), "below", false, {
                let mut c = config_for("compact_index", 16);
                c.min_saved_index_bytes_for_compact = compact_saved as usize + 1;
                c
            }),
        ],
    );

    push_boundaries(
        &mut records,
        "bsr",
        &bsr,
        "expected_reuse",
        &[
            (json!(3), "below", false, config_for("bsr", 3)),
            (json!(4), "at", true, config_for("bsr", 4)),
            (json!(5), "above", true, config_for("bsr", 5)),
        ],
    );
    push_boundaries(
        &mut records,
        "bsr",
        &bsr,
        "saved_bytes_per_apply",
        &[
            (json!(bsr_saved - 1), "above", true, {
                let mut c = config_for("bsr", 16);
                c.min_saved_bytes_for_bsr = bsr_saved as usize - 1;
                c
            }),
            (json!(bsr_saved), "at", true, {
                let mut c = config_for("bsr", 16);
                c.min_saved_bytes_for_bsr = bsr_saved as usize;
                c
            }),
            (json!(bsr_saved + 1), "below", false, {
                let mut c = config_for("bsr", 16);
                c.min_saved_bytes_for_bsr = bsr_saved as usize + 1;
                c
            }),
        ],
    );
    push_boundaries(
        &mut records,
        "bsr",
        &bsr,
        "block_fill_ratio",
        &[
            (
                json!(bsr_analysis.block_fill_ratio - 0.001),
                "above",
                true,
                {
                    let mut c = config_for("bsr", 16);
                    c.min_bsr_block_fill_ratio = bsr_analysis.block_fill_ratio - 0.001;
                    c
                },
            ),
            (json!(bsr_analysis.block_fill_ratio), "at", true, {
                let mut c = config_for("bsr", 16);
                c.min_bsr_block_fill_ratio = bsr_analysis.block_fill_ratio;
                c
            }),
            (
                json!(bsr_analysis.block_fill_ratio + 0.001),
                "below",
                false,
                {
                    let mut c = config_for("bsr", 16);
                    c.min_bsr_block_fill_ratio = bsr_analysis.block_fill_ratio + 0.001;
                    c
                },
            ),
        ],
    );

    push_boundaries(
        &mut records,
        "sell_c_sigma",
        &sell,
        "expected_reuse",
        &[
            (json!(7), "below", false, config_for("sell_c_sigma", 7)),
            (json!(8), "at", true, config_for("sell_c_sigma", 8)),
            (json!(9), "above", true, config_for("sell_c_sigma", 9)),
        ],
    );
    let sell_nnz = sell.values.len();
    push_boundaries(
        &mut records,
        "sell_c_sigma",
        &sell,
        "nnz",
        &[
            (json!(sell_nnz - 1), "above", true, {
                let mut c = config_for("sell_c_sigma", 16);
                c.min_nnz_for_sell_c_sigma = sell_nnz - 1;
                c
            }),
            (json!(sell_nnz), "at", true, {
                let mut c = config_for("sell_c_sigma", 16);
                c.min_nnz_for_sell_c_sigma = sell_nnz;
                c
            }),
            (json!(sell_nnz + 1), "below", false, {
                let mut c = config_for("sell_c_sigma", 16);
                c.min_nnz_for_sell_c_sigma = sell_nnz + 1;
                c
            }),
        ],
    );
    push_boundaries(
        &mut records,
        "sell_c_sigma",
        &sell,
        "fill_ratio",
        &[
            (json!(sell_analysis.fill_ratio - 0.001), "above", true, {
                let mut c = config_for("sell_c_sigma", 16);
                c.min_sell_c_sigma_fill_ratio = sell_analysis.fill_ratio - 0.001;
                c
            }),
            (json!(sell_analysis.fill_ratio), "at", true, {
                let mut c = config_for("sell_c_sigma", 16);
                c.min_sell_c_sigma_fill_ratio = sell_analysis.fill_ratio;
                c
            }),
            (json!(sell_analysis.fill_ratio + 0.001), "below", false, {
                let mut c = config_for("sell_c_sigma", 16);
                c.min_sell_c_sigma_fill_ratio = sell_analysis.fill_ratio + 0.001;
                c
            }),
        ],
    );

    push_boundaries(
        &mut records,
        "row_bucket",
        &row_bucket,
        "expected_reuse",
        &[
            (json!(3), "below", false, config_for("row_bucket", 3)),
            (json!(4), "at", true, config_for("row_bucket", 4)),
            (json!(5), "above", true, config_for("row_bucket", 5)),
        ],
    );
    let row_nnz = row_bucket.values.len();
    push_boundaries(
        &mut records,
        "row_bucket",
        &row_bucket,
        "nnz",
        &[
            (json!(row_nnz - 1), "above", true, {
                let mut c = config_for("row_bucket", 16);
                c.min_nnz_for_row_bucket = row_nnz - 1;
                c
            }),
            (json!(row_nnz), "at", true, {
                let mut c = config_for("row_bucket", 16);
                c.min_nnz_for_row_bucket = row_nnz;
                c
            }),
            (json!(row_nnz + 1), "below", false, {
                let mut c = config_for("row_bucket", 16);
                c.min_nnz_for_row_bucket = row_nnz + 1;
                c
            }),
        ],
    );
    push_boundaries(
        &mut records,
        "row_bucket",
        &row_bucket,
        "row_nnz_cv",
        &[
            (json!(row_cv - 0.001), "above", true, {
                let mut c = config_for("row_bucket", 16);
                c.min_row_nnz_cv_for_row_bucket = row_cv - 0.001;
                c
            }),
            (json!(row_cv), "at", true, {
                let mut c = config_for("row_bucket", 16);
                c.min_row_nnz_cv_for_row_bucket = row_cv;
                c
            }),
            (json!(row_cv + 0.001), "below", false, {
                let mut c = config_for("row_bucket", 16);
                c.min_row_nnz_cv_for_row_bucket = row_cv + 0.001;
                c
            }),
        ],
    );

    push_boundaries(
        &mut records,
        "numa",
        &numa,
        "nnz",
        &[
            (json!(numa_nnz - 1), "above", true, {
                let mut c = config_for("numa", 16);
                c.min_nnz_for_numa = numa_nnz - 1;
                c
            }),
            (json!(numa_nnz), "at", true, {
                let mut c = config_for("numa", 16);
                c.min_nnz_for_numa = numa_nnz;
                c
            }),
            (json!(numa_nnz + 1), "below", false, {
                let mut c = config_for("numa", 16);
                c.min_nnz_for_numa = numa_nnz + 1;
                c
            }),
        ],
    );

    // CSR's row-parallel dispatch has two independent inclusive gates.
    for (nnz, side, passes) in [
        (49_999, "below", false),
        (50_000, "at", true),
        (50_001, "above", true),
    ] {
        let matrix = row_parallel_matrix(nnz, 4_000, 4_096);
        records.push(run_case(
            "row_parallel",
            &matrix,
            "nnz_and_rows",
            json!({
                "nnz": nnz, "rows": 4_000, "nnz_threshold": 50_000, "rows_threshold": 4_000
            }),
            side,
            config_for("row_parallel", 1),
            passes,
        ));
    }
    for (rows, side, passes) in [
        (3_999, "below", false),
        (4_000, "at", true),
        (4_001, "above", true),
    ] {
        let matrix = row_parallel_matrix(50_000, rows, 4_096);
        records.push(run_case(
            "row_parallel",
            &matrix,
            "rows",
            json!({
                "nnz": 50_000, "rows": rows, "nnz_threshold": 50_000, "rows_threshold": 4_000
            }),
            side,
            config_for("row_parallel", 1),
            passes,
        ));
    }

    let output = json!({
        "schema": SCHEMA,
        "evidence": "sonium_sparse_spmv_plan_thresholds",
        "artifact_kind": "sonium-solvers-sparse.ss2.plan-thresholds",
        "fixture_version": FIXTURE_VERSION,
        "status": "passed",
        "mode": if force_scalar { "scalar" } else { "native-avx2-fma" },
        "force_scalar": force_scalar,
        "deterministic_fixture": "ss2-plan-thresholds-v1",
        "target_arch": std::env::consts::ARCH,
        "target_os": std::env::consts::OS,
        "runtime_avx2": runtime_avx2(),
        "runtime_fma": runtime_fma(),
        "rustflags": std::env::var("RUSTFLAGS").unwrap_or_default(),
        "rustc_verbose": rustc_verbose(),
        "cpu_model": cpu_field("model name\t:"),
        "cpu_microcode": cpu_field("microcode\t:"),
        "measurement_contract": {
            "baseline": "raw CSR",
            "reference_oracle": "independent_scalar_csr",
            "correctness_before_timing": true,
            "timing_scope": "steady_state_apply_only;setup_separate",
            "paired_modes": "native-avx2-fma/scalar",
            "deterministic_metadata": true,
            "threshold_predicates": "inclusive below/at/above sweeps",
            "row_parallel_dispatch": "nnz >= 50000 && rows >= 4000 && runtime bandwidth threads > 1",
            "threshold_side_semantics": "below/at/above describe the measured metric relative to the configured threshold",
        },
        "threshold_metadata": {
            "schema": "sonium-solvers-sparse.ss2.threshold-metadata/v1",
            "fixture": "ss2-plan-thresholds-v1",
            "families": ["compact_index", "bsr", "sell_c_sigma", "row_bucket", "numa", "row_parallel"],
            "sides": ["below", "at", "above"],
            "reference_oracle": "independent_scalar_csr",
        },
        "derived_thresholds": {
            "compact_saved_index_bytes_per_apply": compact_saved,
            "bsr_saved_bytes_per_apply": bsr_saved,
            "bsr_block_fill_ratio": bsr_analysis.block_fill_ratio,
            "sell_nnz": sell_nnz,
            "sell_padded_nnz": sell_analysis.padded_nnz,
            "sell_fill_ratio": sell_analysis.fill_ratio,
            "row_bucket_cv": row_cv,
            "numa_nnz": numa_nnz,
        },
        "records": records,
    });
    let encoded = serde_json::to_string_pretty(&output).expect("serialize threshold evidence");
    if let Ok(path) = std::env::var("SONIUM_SPARSE_PLAN_EVIDENCE_OUT") {
        let path = evidence_path(&path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create threshold evidence directory");
        }
        std::fs::write(path, &encoded).expect("write threshold evidence");
    }
    println!("SONIUM_SPARSE_PLAN_EVIDENCE {encoded}");
}
