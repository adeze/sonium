//! Structured physical-backend parity QA for sonium-compute.
//!
//! Set `SONIUM_REQUIRE_BACKEND=cpu|metal|cuda` to turn an unavailable
//! requested accelerator into a failing gate.  The output is JSONL so CI can
//! archive numerical evidence without scraping human-oriented logs.

use serde::Serialize;
use sonium_compute::{Backend, CpuBackend, Transpose};
use sonium_qa::{QaBackendOutcome, QaResult, QaStatus};
use std::time::Instant;

const CASE_ID: &str = "compute.backend-parity.v1";
const MAX_ABS_ERROR: f64 = 2.0e-5;
const MAX_RELATIVE_L2_ERROR: f64 = 2.0e-6;

#[derive(Serialize)]
struct ParityRecord<'a> {
    backend: &'a str,
    selection: &'a str,
    operation: &'a str,
    status: &'a str,
    max_abs_error: f64,
    relative_l2_error: f64,
    max_ulp_error: u32,
    detail: Option<String>,
}

fn required_backend() -> Result<Option<String>, String> {
    let value = match std::env::var("SONIUM_REQUIRE_BACKEND") {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    if matches!(value.as_str(), "cpu" | "metal" | "cuda") {
        Ok(Some(value))
    } else {
        Err(format!(
            "SONIUM_REQUIRE_BACKEND must be cpu, metal, or cuda; got {value:?}"
        ))
    }
}

fn emit(record: &ParityRecord<'_>) {
    println!(
        "{}",
        serde_json::to_string(record).expect("serializable record")
    );
}

fn ulp_distance(a: f32, b: f32) -> u32 {
    let ordered = |value: f32| {
        let bits = value.to_bits() as i32;
        if bits < 0 { i32::MIN - bits } else { bits }
    };
    ordered(a).abs_diff(ordered(b))
}

fn matrix_value(a: &[f64], lda: usize, op: Transpose, row: usize, column: usize) -> f64 {
    let (stored_row, stored_column) = match op {
        Transpose::NoTrans => (row, column),
        Transpose::Trans | Transpose::ConjTrans => (column, row),
    };
    a[stored_row * lda + stored_column]
}

fn reference_gemv(op: Transpose, m: usize, n: usize, lda: usize, x: &[f64], y: &[f64]) -> Vec<f32> {
    let a = [1.0_f64, -2.0, 0.5, 4.0, -1.0, 3.0];
    (0..m)
        .map(|row| {
            (0.75
                * (0..n)
                    .map(|col| matrix_value(&a, lda, op, row, col) * x[col])
                    .sum::<f64>()
                + 0.25 * y[row]) as f32
        })
        .collect()
}

fn run_gemv<B: Backend>(
    backend: &B,
    selection: &'static str,
    op: Transpose,
) -> Result<ParityRecord<'static>, String> {
    let a = [1.0_f32, -2.0, 0.5, 4.0, -1.0, 3.0];
    let (m, n, x, y, operation) = match op {
        Transpose::NoTrans => (
            2,
            3,
            vec![0.25_f32, -0.5, 2.0],
            vec![1.0_f32, -3.0],
            "gemv_f32_notrans",
        ),
        Transpose::Trans => (
            3,
            2,
            vec![0.25_f32, -0.5],
            vec![1.0_f32, -3.0, 0.5],
            "gemv_f32_trans",
        ),
        Transpose::ConjTrans => (
            3,
            2,
            vec![0.25_f32, -0.5],
            vec![1.0_f32, -3.0, 0.5],
            "gemv_f32_conjtrans",
        ),
    };
    let lda = 3;
    let mut a_device = backend
        .raw_alloc::<f32>(a.len())
        .map_err(|error| error.to_string())?;
    let mut x_device = backend
        .raw_alloc::<f32>(x.len())
        .map_err(|error| error.to_string())?;
    let mut y_device = backend
        .raw_alloc::<f32>(y.len())
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_h2d(&a, &mut a_device)
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_h2d(&x, &mut x_device)
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_h2d(&y, &mut y_device)
        .map_err(|error| error.to_string())?;
    backend
        .try_gemv(
            op,
            m,
            n,
            0.75,
            &a_device,
            lda,
            &x_device,
            0.25,
            &mut y_device,
        )
        .map_err(|error| error.to_string())?;
    let mut actual = vec![0.0_f32; m];
    backend
        .try_copy_d2h(&y_device, &mut actual)
        .map_err(|error| error.to_string())?;
    let expected = reference_gemv(
        op,
        m,
        n,
        lda,
        &x.iter().map(|&value| f64::from(value)).collect::<Vec<_>>(),
        &y.iter().map(|&value| f64::from(value)).collect::<Vec<_>>(),
    );
    let max_abs_error = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| f64::from((*actual - *expected).abs()))
        .fold(0.0, f64::max);
    let relative_l2_error = (actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| f64::from(*actual - *expected).powi(2))
        .sum::<f64>()
        / expected
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>())
    .sqrt();
    let max_ulp_error = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| ulp_distance(*actual, *expected))
        .max()
        .unwrap_or_default();
    let status = if max_abs_error <= MAX_ABS_ERROR && relative_l2_error <= MAX_RELATIVE_L2_ERROR {
        "pass"
    } else {
        "fail"
    };
    Ok(ParityRecord {
        backend: backend.capabilities().backend_name,
        selection,
        operation,
        status,
        max_abs_error,
        relative_l2_error,
        max_ulp_error,
        detail: None,
    })
}

fn run_gemm<B: Backend>(
    backend: &B,
    selection: &'static str,
) -> Result<ParityRecord<'static>, String> {
    let a = [1.0_f32, -2.0, 0.5, 4.0, -1.0, 3.0];
    let b = [0.25_f32, -0.5, 2.0, 1.0, -3.0, 0.5];
    let c = [1.0_f32, -3.0, 0.5, 2.0];
    let mut a_device = backend
        .raw_alloc::<f32>(a.len())
        .map_err(|error| error.to_string())?;
    let mut b_device = backend
        .raw_alloc::<f32>(b.len())
        .map_err(|error| error.to_string())?;
    let mut c_device = backend
        .raw_alloc::<f32>(c.len())
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_h2d(&a, &mut a_device)
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_h2d(&b, &mut b_device)
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_h2d(&c, &mut c_device)
        .map_err(|error| error.to_string())?;
    backend
        .try_gemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            3,
            0.75,
            &a_device,
            3,
            &b_device,
            2,
            0.25,
            &mut c_device,
            2,
        )
        .map_err(|error| error.to_string())?;
    let mut actual = [0.0_f32; 4];
    backend
        .try_copy_d2h(&c_device, &mut actual)
        .map_err(|error| error.to_string())?;
    let mut expected = [0.0_f32; 4];
    for row in 0..2 {
        for column in 0..2 {
            let product = (0..3)
                .map(|inner| f64::from(a[row * 3 + inner]) * f64::from(b[inner * 2 + column]))
                .sum::<f64>();
            expected[row * 2 + column] =
                (0.75 * product + 0.25 * f64::from(c[row * 2 + column])) as f32;
        }
    }
    let max_abs_error = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| f64::from((*actual - *expected).abs()))
        .fold(0.0, f64::max);
    let relative_l2_error = (actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| f64::from(*actual - *expected).powi(2))
        .sum::<f64>()
        / expected
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>())
    .sqrt();
    let max_ulp_error = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| ulp_distance(*actual, *expected))
        .max()
        .unwrap_or_default();
    let status = if max_abs_error <= MAX_ABS_ERROR && relative_l2_error <= MAX_RELATIVE_L2_ERROR {
        "pass"
    } else {
        "fail"
    };
    Ok(ParityRecord {
        backend: backend.capabilities().backend_name,
        selection,
        operation: "gemm_f32_notrans",
        status,
        max_abs_error,
        relative_l2_error,
        max_ulp_error,
        detail: None,
    })
}

fn check_backend<B: Backend>(
    backend: &B,
    selection: &'static str,
) -> Result<Vec<ParityRecord<'static>>, String> {
    let mut records = Vec::new();
    for op in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
        let record = run_gemv(backend, selection, op)?;
        let passed = record.status == "pass";
        emit(&record);
        if !passed {
            return Err(format!(
                "{} {} parity failed",
                record.backend, record.operation
            ));
        }
        records.push(record);
    }
    let record = run_gemm(backend, selection)?;
    let passed = record.status == "pass";
    emit(&record);
    if !passed {
        return Err(format!("{} GEMM parity failed", record.backend));
    }
    records.push(record);
    Ok(records)
}

fn run() -> Result<(Vec<ParityRecord<'static>>, Vec<QaBackendOutcome>), String> {
    let required = required_backend()?;
    let mut records = Vec::new();
    let mut backend_outcomes = Vec::new();
    if required.as_deref().is_none_or(|backend| backend == "cpu") {
        records.extend(check_backend(&CpuBackend::new(), "cpu")?);
        backend_outcomes.push(QaBackendOutcome {
            backend: "cpu".into(),
            status: QaStatus::Passed,
            reason: None,
        });
    }

    #[cfg(feature = "metal")]
    if required.as_deref().is_none_or(|backend| backend == "metal") {
        match sonium_compute::MetalBackend::new() {
            Ok(backend) => {
                records.extend(check_backend(&backend, "metal")?);
                backend_outcomes.push(QaBackendOutcome {
                    backend: "metal".into(),
                    status: QaStatus::Passed,
                    reason: None,
                });
            }
            Err(error) if required.as_deref() == Some("metal") => {
                return Err(format!("required Metal backend unavailable: {error}"));
            }
            Err(error) => backend_outcomes.push(QaBackendOutcome {
                backend: "metal".into(),
                status: QaStatus::Skipped,
                reason: Some(error.to_string()),
            }),
        }
    }
    #[cfg(not(feature = "metal"))]
    if required.as_deref() == Some("metal") {
        return Err("required Metal backend was not compiled; pass --features metal".into());
    }
    #[cfg(not(feature = "metal"))]
    if required.is_none() {
        backend_outcomes.push(QaBackendOutcome {
            backend: "metal".into(),
            status: QaStatus::Skipped,
            reason: Some("backend was not compiled; enable the metal feature".into()),
        });
    }

    #[cfg(feature = "cuda")]
    if required.as_deref().is_none_or(|backend| backend == "cuda") {
        match sonium_compute::CudaBackend::new_default() {
            Ok(backend) => {
                records.extend(check_backend(&backend, "cuda")?);
                backend_outcomes.push(QaBackendOutcome {
                    backend: "cuda".into(),
                    status: QaStatus::Passed,
                    reason: None,
                });
            }
            Err(error) if required.as_deref() == Some("cuda") => {
                return Err(format!("required CUDA backend unavailable: {error}"));
            }
            Err(error) => backend_outcomes.push(QaBackendOutcome {
                backend: "cuda".into(),
                status: QaStatus::Skipped,
                reason: Some(error.to_string()),
            }),
        }
    }
    #[cfg(not(feature = "cuda"))]
    if required.as_deref() == Some("cuda") {
        return Err("required CUDA backend was not compiled; pass --features cuda".into());
    }
    #[cfg(not(feature = "cuda"))]
    if required.is_none() {
        backend_outcomes.push(QaBackendOutcome {
            backend: "cuda".into(),
            status: QaStatus::Skipped,
            reason: Some("backend was not compiled; enable the cuda feature".into()),
        });
    }
    if records.is_empty() {
        return Err("no backend was selected for parity validation".into());
    }
    Ok((records, backend_outcomes))
}

fn success_result(
    records: &[ParityRecord<'_>],
    backend_outcomes: &[QaBackendOutcome],
    required: Option<&str>,
    elapsed_ms: u64,
) -> QaResult {
    let max_abs_error = records
        .iter()
        .map(|record| record.max_abs_error)
        .fold(0.0, f64::max);
    let max_relative_l2_error = records
        .iter()
        .map(|record| record.relative_l2_error)
        .fold(0.0, f64::max);
    let max_ulp_error = records
        .iter()
        .map(|record| record.max_ulp_error)
        .max()
        .unwrap_or_default();
    let actual_backend = required
        .or_else(|| {
            records
                .first()
                .filter(|first| records.iter().all(|record| record.backend == first.backend))
                .map(|record| record.backend)
        })
        .unwrap_or("multiple");
    let backend_count = records
        .iter()
        .map(|record| record.backend)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let mut result = QaResult::passed(CASE_ID)
        .actual_backend(actual_backend)
        .metric("max_abs_error", max_abs_error)
        .metric("relative_l2_error", max_relative_l2_error)
        .metric("max_ulp_error", max_ulp_error as f64)
        .metric("backend_count", backend_count as f64)
        .metric("operation_count", records.len() as f64)
        .maximum("max_abs_error", MAX_ABS_ERROR)
        .maximum("relative_l2_error", MAX_RELATIVE_L2_ERROR)
        .input_checksum("analytic/deterministic-fixture:compute-gemv-gemm-f32-v2")
        .reference("independent f64 GEMV/GEMM")
        .reference_checksum("not-applicable")
        .dimension("dense f32 2x3 GEMV/GEMM; no-transpose, transpose, conjugate-transpose")
        .formulation("CPU/Metal/CUDA parity")
        .tolerance("2e-5 absolute; 2e-6 relative")
        .seed("compute-backend-parity:gemv-gemm-f32-matrix:v2")
        .timing_ms("gemv", elapsed_ms)
        .elapsed_ms(elapsed_ms);
    for outcome in backend_outcomes {
        result = match outcome.status {
            QaStatus::Passed => result.backend_passed(&outcome.backend),
            QaStatus::Skipped => result.backend_skipped(
                &outcome.backend,
                outcome.reason.as_deref().unwrap_or("backend unavailable"),
            ),
            QaStatus::Failed => result,
        };
    }
    if let Some(required) = required {
        result = result.requested_backend(required);
    }
    result
}

fn main() {
    let started = Instant::now();
    let required = required_backend().ok().flatten();
    match run() {
        Ok((records, backend_outcomes)) => {
            let result = success_result(
                &records,
                &backend_outcomes,
                required.as_deref(),
                started.elapsed().as_millis() as u64,
            );
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("compute backend parity result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .elapsed_ms(started.elapsed().as_millis() as u64)
                .to_json_line()
                .expect("compute backend parity failure result must serialize");
            println!("QA_RESULT: {result}");
            eprintln!("COMPUTE BACKEND PARITY FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
