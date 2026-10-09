//! Reproducible macOS compute evidence bundle.
//!
//! The benchmark deliberately measures only preallocated buffers.  It is a
//! release-evidence producer, not a microbenchmark framework: it writes a
//! versioned JSON record plus human-readable Markdown and SVG companions.
//! `RAYON_NUM_THREADS` sets the production worker count N (machine width
//! when unset); cache/memory regimes additionally run under 1 worker, so
//! one bundle carries rows at both 1 and N threads and the regression gate
//! compares per-thread-count cells. Archive the resulting directory with
//! the release evidence.

#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use std::path::{Path, PathBuf};
use std::time::Instant;

#[cfg(target_os = "macos")]
use rayon::prelude::*;
use serde::Serialize;
use sonium_bench::compute_regression::{
    DEFAULT_MAX_MEDIAN_SLOWDOWN, EvidenceSnapshot, RegressionPolicy, compare_evidence,
};
#[cfg(target_os = "macos")]
use sonium_compute::{Backend as _, CpuBackend, Transpose};
#[cfg(target_os = "macos")]
use std::cell::RefCell;

const WARMUP: usize = 5;
const SAMPLES: usize = 15;

#[derive(Serialize)]
struct Evidence<'a> {
    schema: &'a str,
    git_sha: String,
    timestamp_utc: String,
    hardware: Hardware,
    protocol: Protocol,
    rows: Vec<Row>,
    limitations: Vec<&'a str>,
}

#[derive(Serialize)]
struct Hardware {
    os: String,
    arch: String,
    model: String,
    rustc: String,
    profile: String,
    target_features: String,
    device: String,
    driver: String,
    rayon_threads: String,
    thermal_state: String,
    clocks: String,
}

#[derive(Serialize)]
struct Protocol {
    input_generator: &'static str,
    warmup_samples: usize,
    measured_samples: usize,
    confidence_interval: &'static str,
    correctness: &'static str,
    allocation_measurement: &'static str,
    energy_measurement: &'static str,
    regression_policy: &'static str,
}

#[derive(Serialize)]
struct Row {
    implementation: &'static str,
    comparator: &'static str,
    blas_level: &'static str,
    operation: &'static str,
    regime: &'static str,
    dtype: &'static str,
    shape: String,
    /// Rayon worker count the timed kernels ran under. Cache/memory
    /// regimes sweep 1 and N threads in one bundle; the count is part of
    /// the regression cell identity (see `compute_regression::RowKey`).
    threads: String,
    correctness_passed: bool,
    median_ns: f64,
    ci95_low_ns: f64,
    ci95_high_ns: f64,
    allocations: &'static str,
    energy: &'static str,
}

#[cfg(target_os = "macos")]
mod accelerate {
    use std::os::raw::{c_float, c_int};

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub enum Order {
        RowMajor = 101,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub enum Transpose {
        NoTrans = 111,
    }

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn cblas_saxpy(
            n: c_int,
            alpha: c_float,
            x: *const c_float,
            incx: c_int,
            y: *mut c_float,
            incy: c_int,
        );
        pub fn cblas_sgemv(
            order: Order,
            trans: Transpose,
            m: c_int,
            n: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            x: *const c_float,
            incx: c_int,
            beta: c_float,
            y: *mut c_float,
            incy: c_int,
        );
        pub fn cblas_sgemm(
            order: Order,
            transa: Transpose,
            transb: Transpose,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            b: *const c_float,
            ldb: c_int,
            beta: c_float,
            c: *mut c_float,
            ldc: c_int,
        );
    }
}

fn command_output(program: &str, args: &[&str]) -> String {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned())
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn percentile(sorted: &[f64], numerator: usize, denominator: usize) -> f64 {
    assert!(
        !sorted.is_empty(),
        "percentile requires at least one timing sample"
    );
    sorted[numerator * (sorted.len() - 1) / denominator]
}

/// Rank (0-based) of the lower order statistic of a two-sided nonparametric
/// 95% confidence interval for the median of `n` samples.
///
/// The interval `[x_(lo), x_(n-1-lo)]` is the standard binomial-order-statistic
/// CI: with `X ~ Bin(n, 1/2)`, the largest `lo` with `P(X <= lo) <= 0.025`
/// (plus the symmetric upper rank) covers the population median with at least
/// 95% probability. For `SAMPLES = 15` this is `[x_(3), x_(11)]` — much tighter
/// than the previously reported 2nd/98th sample percentiles (the sample min
/// and second-max), which are an order-statistic spread, not a CI of the
/// median, and made the regression overlap test pass by luck. For tiny `n`
/// where no 95% interval exists the ranks collapse onto the median.
fn median_ci95_lower_rank(n: usize) -> usize {
    assert!(n > 0, "median CI requires at least one timing sample");
    // Largest k with P(X <= k) <= 0.025 for X ~ Bin(n, 1/2); the CI is then
    // [x_(k), x_(n-1-k)] in 0-based ranks.
    let mut cumulative = 0.0;
    let mut lower = 0;
    for k in 0..n {
        cumulative += binomial_point(n, k);
        if cumulative <= 0.025 {
            lower = k;
        } else {
            break;
        }
    }
    lower.min(n / 2)
}

fn binomial_point(n: usize, k: usize) -> f64 {
    // P(X = k) for X ~ Bin(n, 1/2), computed in log space for stability.
    let mut log_prob = -(n as f64) * std::f64::consts::LN_2;
    for i in 1..=k {
        log_prob += ((n - k + i) as f64).ln() - (i as f64).ln();
    }
    log_prob.exp()
}

fn timed(mut reset: impl FnMut(), mut operation: impl FnMut()) -> (f64, f64, f64) {
    for _ in 0..WARMUP {
        reset();
        operation();
    }
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        // Reset outside the timed region so each sample measures the same
        // operation on the same input state.  This matters for in-place
        // BLAS-1 operations such as AXPY, whose output otherwise compounds
        // across warmups and samples.
        reset();
        let started = Instant::now();
        operation();
        samples.push(started.elapsed().as_nanos() as f64);
    }
    samples.sort_by(f64::total_cmp);
    // Median plus a genuine nonparametric 95% CI of the median (stored in the
    // `ci95_*` fields and consumed by the regression overlap test).
    let median = percentile(&samples, 50, 100);
    let lo = median_ci95_lower_rank(samples.len());
    let hi = samples.len() - 1 - lo;
    (median, samples[lo], samples[hi])
}

fn values(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (i as f32 * phase).sin() + (i as f32 * (phase * 0.7)).cos())
        .collect()
}

/// Element-wise closeness with an explicit absolute floor per call site.
///
/// The tolerance is `atol + 2e-3 * |expected|`: the 0.2% relative term is kept
/// because the vendor kernel and the naive reference legitimately differ by
/// summation-order rounding (which can dominate for large cancellation-prone
/// sums), while the absolute floor — previously an implicit `2e-3` via
/// `max(|expected|, 1.0)` — is now set per operation and scaled with GEMM
/// accumulation depth `k`, so near-zero garbage no longer passes.
fn close(got: &[f32], expected: &[f32], atol: f32) -> bool {
    got.iter()
        .zip(expected)
        .all(|(got, expected)| (*got - *expected).abs() <= atol + 2e-3 * expected.abs())
}

fn output_dir() -> PathBuf {
    std::env::var_os("SONIUM_COMPUTE_EVIDENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/sonium-compute-evidence"))
}

fn env_flag(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn controlled_metadata(name: &str, fallback: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| fallback.to_owned())
}

fn regression_policy() -> Result<RegressionPolicy, String> {
    let max_median_slowdown = std::env::var("SONIUM_COMPUTE_MAX_MEDIAN_SLOWDOWN")
        .ok()
        .map(|value| {
            value.parse::<f64>().map_err(|error| {
                format!("invalid SONIUM_COMPUTE_MAX_MEDIAN_SLOWDOWN={value:?}: {error}")
            })
        })
        .transpose()?
        .unwrap_or(DEFAULT_MAX_MEDIAN_SLOWDOWN);
    Ok(RegressionPolicy {
        max_median_slowdown,
        require_non_overlapping_interval: true,
    })
}

fn enforce_controlled_protocol(hardware: &Hardware) -> Result<(), String> {
    if !env_flag("SONIUM_COMPUTE_REQUIRE_CONTROLLED_PROTOCOL") {
        return Ok(());
    }
    if hardware.rayon_threads == "runtime-default" {
        return Err("controlled performance evidence requires explicit RAYON_NUM_THREADS".into());
    }
    if hardware.thermal_state == "not_measured" {
        return Err("controlled performance evidence requires SONIUM_COMPUTE_THERMAL_STATE".into());
    }
    if hardware.clocks == "not_measured" {
        return Err("controlled performance evidence requires SONIUM_COMPUTE_CLOCK_POLICY".into());
    }
    Ok(())
}

fn check_regression(output: &Path, candidate_bytes: &[u8]) -> Result<(), String> {
    let Some(baseline_path) = std::env::var_os("SONIUM_COMPUTE_BASELINE") else {
        if env_flag("SONIUM_COMPUTE_REQUIRE_BASELINE") {
            return Err(
                "SONIUM_COMPUTE_REQUIRE_BASELINE is set but SONIUM_COMPUTE_BASELINE is missing"
                    .into(),
            );
        }
        return Ok(());
    };
    let baseline_bytes = std::fs::read(&baseline_path).map_err(|error| {
        format!(
            "failed to read compute baseline {}: {error}",
            PathBuf::from(&baseline_path).display()
        )
    })?;
    // The comparator retains the full hardware/build environment and timing
    // protocol. Source commit/timestamp may differ between compared runs.
    let baseline = serde_json::from_slice::<EvidenceSnapshot>(&baseline_bytes)
        .map_err(|error| format!("invalid compute baseline JSON: {error}"))?;
    let candidate = serde_json::from_slice::<EvidenceSnapshot>(candidate_bytes)
        .map_err(|error| format!("invalid candidate evidence JSON: {error}"))?;
    let report = compare_evidence(&baseline, &candidate, regression_policy()?)?;
    std::fs::write(
        output.join("regression.json"),
        serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    if !report.passed() {
        return Err(format!(
            "compute performance regression gate failed for {} cells; see {}",
            report.regression_count,
            output.join("regression.json").display()
        ));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn host_axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    for (y, x) in y.iter_mut().zip(x) {
        *y += alpha * x;
    }
}

/// Reference-computation fan-out gate: each output row is a full dot
/// product (microseconds at the measured shapes), so any split across two
/// or more threads wins; the `>= 2` minimum is structural, not a machine
/// threshold — the machine dependence is the thread count.
/// `SONIUM_BENCH_REF_PAR_THRESHOLD=<n>` overrides with an absolute row
/// count; `SONIUM_SERIAL=1` forces the serial reference.
#[cfg(target_os = "macos")]
fn host_reference_parallel(rows: usize) -> bool {
    if std::env::var("SONIUM_SERIAL")
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
    {
        return false;
    }
    if let Some(threshold) = std::env::var("SONIUM_BENCH_REF_PAR_THRESHOLD")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    {
        return rows >= threshold;
    }
    let threads = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    rows >= 2 && threads >= 2
}

#[cfg(target_os = "macos")]
fn host_gemv(m: usize, n: usize, a: &[f32], x: &[f32], y: &mut [f32]) {
    // One output row per worker, accumulated serially within the row:
    // bitwise identical to the serial reference.
    if host_reference_parallel(m) {
        y.par_iter_mut().enumerate().for_each(|(row, y_row)| {
            *y_row = (0..n).map(|col| a[row * n + col] * x[col]).sum();
        });
    } else {
        for row in 0..m {
            y[row] = (0..n).map(|col| a[row * n + col] * x[col]).sum();
        }
    }
}

#[cfg(target_os = "macos")]
fn host_gemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    // One output row per worker, accumulated serially within the row:
    // bitwise identical to the serial reference.
    if host_reference_parallel(m) {
        c.par_chunks_mut(n.max(1))
            .enumerate()
            .for_each(|(row, c_row)| {
                for col in 0..n {
                    c_row[col] = (0..k).map(|p| a[row * k + p] * b[p * n + col]).sum();
                }
            });
    } else {
        for row in 0..m {
            for col in 0..n {
                c[row * n + col] = (0..k).map(|p| a[row * k + p] * b[p * n + col]).sum();
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn benchmark_axpy(
    backend: &CpuBackend,
    regime: &'static str,
    len: usize,
    threads: usize,
) -> Result<Vec<Row>, String> {
    let x = values(len, 0.0013);
    let initial = values(len, 0.0007);
    let mut expected = initial.clone();
    host_axpy(0.75, &x, &mut expected);
    let mut x_d = backend.raw_alloc::<f32>(len).map_err(|e| e.to_string())?;
    let mut initial_d = backend.raw_alloc::<f32>(len).map_err(|e| e.to_string())?;
    let mut y_d = backend.raw_alloc::<f32>(len).map_err(|e| e.to_string())?;
    backend
        .try_copy_h2d(&x, &mut x_d)
        .map_err(|e| e.to_string())?;
    backend
        .try_copy_h2d(&initial, &mut initial_d)
        .map_err(|e| e.to_string())?;
    backend
        .try_copy_d2d(&initial_d, &mut y_d)
        .map_err(|e| e.to_string())?;
    backend
        .try_axpy(0.75, &x_d, &mut y_d)
        .map_err(|e| e.to_string())?;
    let mut got = vec![0.0; len];
    backend
        .try_copy_d2h(&y_d, &mut got)
        .map_err(|e| e.to_string())?;
    let cpu_correct = close(&got, &expected, 1e-6);
    let mut accelerated = initial.clone();
    unsafe {
        accelerate::cblas_saxpy(len as i32, 0.75, x.as_ptr(), 1, accelerated.as_mut_ptr(), 1)
    };
    let accelerate_correct = close(&accelerated, &expected, 1e-6);
    if !cpu_correct || !accelerate_correct {
        return Err(format!("AXPY correctness failed for {regime}"));
    }
    let y_d = RefCell::new(y_d);
    let cpu = timed(
        || {
            backend
                .try_copy_d2d(&initial_d, &mut y_d.borrow_mut())
                .expect("prevalidated AXPY reset");
        },
        || {
            backend
                .try_axpy(0.75, &x_d, &mut y_d.borrow_mut())
                .expect("prevalidated AXPY")
        },
    );
    let accelerated = RefCell::new(accelerated);
    let accelerate = timed(
        || accelerated.borrow_mut().copy_from_slice(&initial),
        || {
            let mut output = accelerated.borrow_mut();
            unsafe {
                accelerate::cblas_saxpy(len as i32, 0.75, x.as_ptr(), 1, output.as_mut_ptr(), 1)
            }
        },
    );
    Ok(rows(
        "blas-1",
        "axpy",
        regime,
        format!("n={len}"),
        threads,
        cpu_correct,
        cpu,
        accelerate_correct,
        accelerate,
    ))
}

#[cfg(target_os = "macos")]
fn benchmark_gemv(
    backend: &CpuBackend,
    regime: &'static str,
    m: usize,
    n: usize,
    threads: usize,
) -> Result<Vec<Row>, String> {
    let a = values(m * n, 0.0013);
    let x = values(n, 0.0007);
    let mut expected = vec![0.0; m];
    host_gemv(m, n, &a, &x, &mut expected);
    let mut a_d = backend
        .raw_alloc::<f32>(a.len())
        .map_err(|e| e.to_string())?;
    let mut x_d = backend.raw_alloc::<f32>(n).map_err(|e| e.to_string())?;
    let mut y_d = backend.raw_alloc::<f32>(m).map_err(|e| e.to_string())?;
    backend
        .try_copy_h2d(&a, &mut a_d)
        .map_err(|e| e.to_string())?;
    backend
        .try_copy_h2d(&x, &mut x_d)
        .map_err(|e| e.to_string())?;
    backend
        .try_gemv(Transpose::NoTrans, m, n, 1.0, &a_d, n, &x_d, 0.0, &mut y_d)
        .map_err(|e| e.to_string())?;
    let mut got = vec![0.0; m];
    backend
        .try_copy_d2h(&y_d, &mut got)
        .map_err(|e| e.to_string())?;
    let cpu_correct = close(&got, &expected, 1e-6);
    let mut accelerated = vec![0.0; m];
    let lda = n.max(1) as i32;
    unsafe {
        accelerate::cblas_sgemv(
            accelerate::Order::RowMajor,
            accelerate::Transpose::NoTrans,
            m as i32,
            n as i32,
            1.0,
            a.as_ptr(),
            lda,
            x.as_ptr(),
            1,
            0.0,
            accelerated.as_mut_ptr(),
            1,
        )
    };
    let accelerate_correct = close(&accelerated, &expected, 1e-6);
    if !cpu_correct || !accelerate_correct {
        return Err(format!("GEMV correctness failed for {regime}"));
    }
    let cpu = timed(
        || {},
        || {
            backend
                .try_gemv(Transpose::NoTrans, m, n, 1.0, &a_d, n, &x_d, 0.0, &mut y_d)
                .expect("prevalidated GEMV")
        },
    );
    let accelerate = timed(
        || {},
        || unsafe {
            accelerate::cblas_sgemv(
                accelerate::Order::RowMajor,
                accelerate::Transpose::NoTrans,
                m as i32,
                n as i32,
                1.0,
                a.as_ptr(),
                lda,
                x.as_ptr(),
                1,
                0.0,
                accelerated.as_mut_ptr(),
                1,
            )
        },
    );
    Ok(rows(
        "blas-2",
        "gemv",
        regime,
        format!("{m}x{n}"),
        threads,
        cpu_correct,
        cpu,
        accelerate_correct,
        accelerate,
    ))
}

#[cfg(target_os = "macos")]
fn benchmark_gemm(
    backend: &CpuBackend,
    regime: &'static str,
    m: usize,
    n: usize,
    k: usize,
    threads: usize,
) -> Result<Vec<Row>, String> {
    let a = values(m * k, 0.0013);
    let b = values(k * n, 0.0007);
    let mut expected = vec![0.0; m * n];
    host_gemm(m, n, k, &a, &b, &mut expected);
    let mut a_d = backend
        .raw_alloc::<f32>(a.len())
        .map_err(|e| e.to_string())?;
    let mut b_d = backend
        .raw_alloc::<f32>(b.len())
        .map_err(|e| e.to_string())?;
    let mut c_d = backend.raw_alloc::<f32>(m * n).map_err(|e| e.to_string())?;
    backend
        .try_copy_h2d(&a, &mut a_d)
        .map_err(|e| e.to_string())?;
    backend
        .try_copy_h2d(&b, &mut b_d)
        .map_err(|e| e.to_string())?;
    backend
        .try_gemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            &a_d,
            k,
            &b_d,
            n,
            0.0,
            &mut c_d,
            n,
        )
        .map_err(|e| e.to_string())?;
    let mut got = vec![0.0; m * n];
    backend
        .try_copy_d2h(&c_d, &mut got)
        .map_err(|e| e.to_string())?;
    let cpu_correct = close(&got, &expected, 1e-6 * k as f32);
    let mut accelerated = vec![0.0; m * n];
    let lda = k.max(1) as i32;
    let ldb = n.max(1) as i32;
    let ldc = n.max(1) as i32;
    unsafe {
        accelerate::cblas_sgemm(
            accelerate::Order::RowMajor,
            accelerate::Transpose::NoTrans,
            accelerate::Transpose::NoTrans,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            0.0,
            accelerated.as_mut_ptr(),
            ldc,
        )
    };
    let accelerate_correct = close(&accelerated, &expected, 1e-6 * k as f32);
    if !cpu_correct || !accelerate_correct {
        return Err(format!("GEMM correctness failed for {regime}"));
    }
    let cpu = timed(
        || {},
        || {
            backend
                .try_gemm(
                    Transpose::NoTrans,
                    Transpose::NoTrans,
                    m,
                    n,
                    k,
                    1.0,
                    &a_d,
                    k,
                    &b_d,
                    n,
                    0.0,
                    &mut c_d,
                    n,
                )
                .expect("prevalidated GEMM")
        },
    );
    let accelerate = timed(
        || {},
        || unsafe {
            accelerate::cblas_sgemm(
                accelerate::Order::RowMajor,
                accelerate::Transpose::NoTrans,
                accelerate::Transpose::NoTrans,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                lda,
                b.as_ptr(),
                ldb,
                0.0,
                accelerated.as_mut_ptr(),
                ldc,
            )
        },
    );
    Ok(rows(
        "blas-3",
        "gemm",
        regime,
        format!("{m}x{n}x{k}"),
        threads,
        cpu_correct,
        cpu,
        accelerate_correct,
        accelerate,
    ))
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn rows(
    blas_level: &'static str,
    operation: &'static str,
    regime: &'static str,
    shape: String,
    threads: usize,
    cpu_correct: bool,
    cpu: (f64, f64, f64),
    accelerate_correct: bool,
    accelerate: (f64, f64, f64),
) -> Vec<Row> {
    [
        ("sonium-cpu", "Accelerate-cBLAS", cpu_correct, cpu),
        (
            "Accelerate-cBLAS",
            "Sonium-CPU",
            accelerate_correct,
            accelerate,
        ),
    ]
    .into_iter()
    .map(
        |(
            implementation,
            comparator,
            correctness_passed,
            (median_ns, ci95_low_ns, ci95_high_ns),
        )| Row {
            implementation,
            comparator,
            blas_level,
            operation,
            regime,
            dtype: "f32",
            shape: shape.clone(),
            threads: threads.to_string(),
            correctness_passed,
            median_ns,
            ci95_low_ns,
            ci95_high_ns,
            allocations: "preallocated buffers; no allocations in timed closure",
            energy: "not_measured",
        },
    )
    .collect()
}

/// Run `run` under an explicit Rayon pool of `threads` workers so the
/// row's `threads` tag always matches the pool the timed kernels actually
/// used — independent of the ambient `RAYON_NUM_THREADS` global pool.
/// The [`CpuBackend`] handle is a unit struct, so each pool gets a fresh
/// one inside the install (the handle is `!Send`, which keeps a shared
/// `&CpuBackend` from crossing the pool boundary).
#[cfg(target_os = "macos")]
fn run_with_threads<T>(
    threads: usize,
    run: impl FnOnce(&CpuBackend) -> T + Send,
) -> Result<T, String>
where
    T: Send,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|error| format!("failed to build {threads}-thread evidence pool: {error}"))?;
    Ok(pool.install(|| run(&CpuBackend::new())))
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), String> {
    // Thread-count dimension: RAYON_NUM_THREADS (or the machine width when
    // unset) is the production count N; cache/memory regimes additionally
    // run under 1 worker so one bundle carries the parallel-speedup
    // evidence. Every row is tagged with its worker count, which is part
    // of the regression cell identity.
    let default_threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&threads| threads >= 1)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(1)
        });
    let sweep: Vec<usize> = if default_threads == 1 {
        vec![1]
    } else {
        vec![1, default_threads]
    };
    let mut rows = Vec::new();
    // No `empty` (n=0 / 0x0) shapes: they time an empty closure (tens of ns of
    // timer overhead), and the gate's 0.0/0.0 special case turns timer noise on
    // those rows into release failures. Correctness of empty inputs is covered
    // by unit tests, not by the timing gate.
    for &(regime, n) in &[
        ("unit", 1),
        ("tail", 31),
        ("cache", 4096),
        ("memory", 1 << 20),
    ] {
        // Only cache/memory regimes sweep the thread count; the small
        // regimes would only time pool dispatch overhead.
        let counts: &[usize] = if regime == "cache" || regime == "memory" {
            &sweep
        } else {
            std::slice::from_ref(&default_threads)
        };
        for &threads in counts {
            rows.extend(run_with_threads(threads, |backend| {
                benchmark_axpy(backend, regime, n, threads)
            })??);
        }
    }
    for &(regime, m, n) in &[
        ("unit", 1, 1),
        ("tail", 31, 29),
        ("square", 256, 256),
        ("tall", 1024, 32),
        ("wide", 32, 1024),
        ("memory", 1024, 1024),
    ] {
        let counts: &[usize] = if regime == "memory" {
            &sweep
        } else {
            std::slice::from_ref(&default_threads)
        };
        for &threads in counts {
            rows.extend(run_with_threads(threads, |backend| {
                benchmark_gemv(backend, regime, m, n, threads)
            })??);
        }
    }
    for &(regime, m, n, k) in &[
        ("unit", 1, 1, 1),
        ("tail", 31, 29, 17),
        ("square", 128, 128, 128),
        ("tall", 512, 32, 128),
        ("wide", 32, 512, 128),
        ("memory", 384, 384, 384),
    ] {
        let counts: &[usize] = if regime == "memory" {
            &sweep
        } else {
            std::slice::from_ref(&default_threads)
        };
        for &threads in counts {
            rows.extend(run_with_threads(threads, |backend| {
                benchmark_gemm(backend, regime, m, n, k, threads)
            })??);
        }
    }
    let hardware = Hardware {
        os: command_output("sw_vers", &["-productVersion"]),
        arch: std::env::consts::ARCH.into(),
        model: command_output("sysctl", &["-n", "hw.model"]),
        rustc: env!("SONIUM_BENCH_RUSTC").into(),
        profile: env!("SONIUM_BENCH_PROFILE").into(),
        target_features: env!("SONIUM_BENCH_TARGET_FEATURES").into(),
        device: command_output("sysctl", &["-n", "machdep.cpu.brand_string"]),
        driver: "not_applicable_cpu".into(),
        rayon_threads: default_threads.to_string(),
        thermal_state: controlled_metadata("SONIUM_COMPUTE_THERMAL_STATE", "not_measured"),
        clocks: controlled_metadata("SONIUM_COMPUTE_CLOCK_POLICY", "not_measured"),
    };
    enforce_controlled_protocol(&hardware)?;
    let evidence = Evidence {
        // v5 binds compiler/build/device and the complete timing protocol.
        // Legacy snapshots remain readable but require regenerated baselines.
        schema: "sonium-compute-macos-evidence/v5",
        git_sha: command_output("git", &["rev-parse", "HEAD"]),
        timestamp_utc: chrono::Utc::now().to_rfc3339(),
        hardware,
        protocol: Protocol {
            input_generator: "compute-macos-evidence-values-v1",
            warmup_samples: WARMUP,
            measured_samples: SAMPLES,
            confidence_interval: "nonparametric 95% CI of the median via binomial order statistics (15 samples)",
            correctness: "independent f32 reference checked before timing; relative tolerance 2e-3",
            allocation_measurement: "timed closures contain no allocation or H2D/D2H transfer",
            energy_measurement: "not_measured; use powermetrics externally when release evidence requires energy",
            regression_policy: "pass only conclusive stable/improved cells; inconclusive slowdowns require more evidence; complete matrix, hardware/build and protocol identity required",
        },
        rows,
        limitations: vec![
            "CPU f32 BLAS-1/2/3 coverage is implemented. Metal/MPS rows require the optional Metal backend and are deliberately not represented as CPU evidence.",
            "Thermal state, clocks, and energy require external macOS measurement tools; the schema records that absence rather than implying measurement.",
        ],
    };
    let output = output_dir();
    std::fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let evidence_bytes = serde_json::to_vec_pretty(&evidence).map_err(|e| e.to_string())?;
    std::fs::write(output.join("evidence.json"), &evidence_bytes).map_err(|e| e.to_string())?;
    check_regression(&output, &evidence_bytes)?;
    let report_rows = evidence
        .rows
        .iter()
        .map(|row| {
            format!(
                "| {} | {} | {} | {} | {} | {:.0} | {:.0}–{:.0} | {} |",
                row.blas_level,
                row.operation,
                row.regime,
                row.implementation,
                row.threads,
                row.median_ns,
                row.ci95_low_ns,
                row.ci95_high_ns,
                row.correctness_passed
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(output.join("report.md"), format!("# Sonium compute macOS evidence\n\nCommit: `{}`  \\nmacOS: `{}`  \\nThreads: `{}`\n\n| BLAS | Operation | Regime | Implementation | Threads | Median ns | 95% interval ns | Correct |\n|---|---|---|---|---|---|---:|---|\n{report_rows}\n", evidence.git_sha, evidence.hardware.os, evidence.hardware.rayon_threads)).map_err(|e| e.to_string())?;
    let max = evidence
        .rows
        .iter()
        .map(|row| row.median_ns)
        .fold(0.0, f64::max)
        .max(1.0);
    let height = 28 * evidence.rows.len() + 24;
    let bars = evidence.rows.iter().enumerate().map(|(i, row)| format!("<text x=\"12\" y=\"{}\" fill=\"white\">{} {} {}</text><rect x=\"270\" y=\"{}\" width=\"{:.1}\" height=\"18\" fill=\"#4cc9f0\"/>", 20 + i * 28, row.operation, row.regime, row.implementation, 6 + i * 28, row.median_ns / max * 500.0)).collect::<String>();
    std::fs::write(output.join("median.svg"), format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"800\" height=\"{height}\" viewBox=\"0 0 800 {height}\"><rect width=\"100%\" height=\"100%\" fill=\"#10131a\"/>{bars}</svg>")).map_err(|e| e.to_string())?;
    println!("evidence bundle written to {}", output.display());
    println!("SUCCESS");
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn run() -> Result<(), String> {
    Err("bench-compute-macos-evidence requires macOS".into())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("COMPUTE EVIDENCE FAILED: {error}");
        println!("FAILED");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_ci_ranks_are_correct() {
        // n = 15 (SAMPLES): P(X<=3) = 0.0176 <= 0.025 < P(X<=4) = 0.0592.
        assert_eq!(median_ci95_lower_rank(15), 3);
        // Degenerate sizes collapse onto the median instead of panicking.
        assert_eq!(median_ci95_lower_rank(1), 0);
        assert_eq!(median_ci95_lower_rank(2), 0);
        // The interval always brackets the reported median rank for any n.
        for n in 1..64 {
            let lo = median_ci95_lower_rank(n);
            let hi = n - 1 - lo;
            let median = 50 * (n - 1) / 100;
            assert!(
                lo <= median && median <= hi,
                "n={n}: [{lo}, {hi}] vs {median}"
            );
        }
    }

    #[test]
    fn percentile_rejects_empty_samples() {
        let result = std::panic::catch_unwind(|| percentile(&[], 50, 100));
        assert!(result.is_err());
    }
}
