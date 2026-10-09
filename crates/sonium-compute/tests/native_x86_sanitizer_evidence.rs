//! Deterministic native-x86 AddressSanitizer and checked negative-path corpus.
//!
//! This test is intentionally ignored in ordinary test runs.  The C5 runner
//! enables nightly AddressSanitizer and invokes it on a native Linux x86_64
//! AVX2/FMA host.  Every negative case must be rejected by the checked CPU
//! backend before dispatch, preserve its destination when one exists, and be
//! followed by a successful retry on the same backend state.

#![cfg(target_arch = "x86_64")]

use serde::Serialize;
use sha2::{Digest, Sha256};
use sonium_compute::BackendError;
use sonium_compute::backend::{Backend, Transpose};
use sonium_compute::bem::BemRegularPairInF64;
use sonium_compute::cpu::{self, CpuBackend};
use std::process::Command;

const SCHEMA: u32 = 1;
const FIXTURE_VERSION: u32 = 1;
const DETERMINISTIC_FIXTURE: &str = "c5-native-x86-sanitizer-negative-v1";
const SANITIZER: &str = "address";

#[derive(Debug, Serialize)]
struct NegativeProbe {
    id: String,
    operation: String,
    expected_error_kind: String,
    observed_error_kind: String,
    observed_operation: String,
    destination_preserved: bool,
    retry_succeeded: bool,
}

#[derive(Debug, Serialize)]
struct SanitizerArtifact {
    schema: u32,
    artifact_kind: &'static str,
    fixture_version: u32,
    deterministic_fixture: &'static str,
    status: &'static str,
    target_arch: &'static str,
    target_os: &'static str,
    cpu_model: String,
    microcode: String,
    compiler: String,
    thread_policy: String,
    affinity_policy: String,
    numa_policy: String,
    runtime_avx2: bool,
    runtime_fma: bool,
    sanitizer: &'static str,
    sanitizer_options: String,
    leak_detection: &'static str,
    positive_case_count: usize,
    positive_cases: Vec<String>,
    negative_case_count: usize,
    negative_cases: Vec<NegativeProbe>,
    cases_failed: usize,
    corpus_sha256: String,
    unsupported_scope: &'static str,
}

fn error_details(error: &BackendError) -> (&'static str, &'static str) {
    match error {
        BackendError::Unsupported { operation, .. } => ("Unsupported", operation),
        BackendError::HostFallback { operation, .. } => ("HostFallback", operation),
        BackendError::InvalidArgument { operation, .. } => ("InvalidArgument", operation),
        BackendError::OperationFailed { operation, .. } => ("OperationFailed", operation),
        BackendError::DeviceLost { operation, .. } => ("DeviceLost", operation),
        BackendError::CommandBuffer { operation, .. } => ("CommandBuffer", operation),
        BackendError::Launch { operation, .. } => ("Launch", operation),
        BackendError::Synchronization { operation, .. } => ("Synchronization", operation),
        BackendError::Transfer { operation, .. } => ("Transfer", operation),
        BackendError::Allocation { .. } => ("Allocation", "allocation"),
    }
}

fn record_negative(
    id: &str,
    operation: &str,
    expected_kind: &str,
    error: &BackendError,
    destination_preserved: bool,
    retry_succeeded: bool,
) -> NegativeProbe {
    let (observed_kind, observed_operation) = error_details(error);
    assert_eq!(observed_kind, expected_kind, "{id}: typed error kind");
    assert_eq!(observed_operation, operation, "{id}: operation name");
    assert!(destination_preserved, "{id}: destination was mutated");
    assert!(retry_succeeded, "{id}: retry did not succeed");
    NegativeProbe {
        id: id.to_owned(),
        operation: operation.to_owned(),
        expected_error_kind: expected_kind.to_owned(),
        observed_error_kind: observed_kind.to_owned(),
        observed_operation: observed_operation.to_owned(),
        destination_preserved,
        retry_succeeded,
    }
}

fn digest_corpus(positive_cases: &[String], negative_cases: &[NegativeProbe]) -> String {
    let bytes = serde_json::to_vec(&(positive_cases, negative_cases)).expect("serialize corpus");
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn cpuinfo_value(field: &str) -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                let (key, value) = line.split_once(':')?;
                (key.trim() == field).then(|| value.trim().to_owned())
            })
        })
        .unwrap_or_else(|| "unknown".into())
}

fn compiler_version() -> String {
    Command::new("rustc")
        .arg("+nightly")
        .arg("-vV")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|output| output.lines().next().map(str::to_owned))
        .unwrap_or_else(|| "unknown".into())
}

fn probe_spmv_tail_cases() -> Vec<String> {
    let lengths = [0_usize, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33];
    let mut cases = Vec::with_capacity(lengths.len() * 2);
    for &length in &lengths {
        let start = 1;
        let end = start + length;
        let col_idx: Vec<i32> = (0..=length).map(|index| (index % 11) as i32).collect();
        let values_f32: Vec<f32> = (0..=length)
            .map(|index| 0.25 + (index % 7) as f32 * 0.03125)
            .collect();
        let values_f64: Vec<f64> = values_f32.iter().map(|&value| f64::from(value)).collect();
        let x_f32: Vec<f32> = (0..11).map(|index| 0.5 + index as f32 * 0.125).collect();
        let x_f64: Vec<f64> = x_f32.iter().map(|&value| f64::from(value)).collect();
        let actual_f32 = cpu::spmv_simd::row_f32(&col_idx, &values_f32, &x_f32, start, end);
        let expected_f32 =
            cpu::spmv_simd::row_f32_scalar_reference(&col_idx, &values_f32, &x_f32, start, end);
        assert!((actual_f32 - expected_f32).abs() <= 2.0e-5);
        let actual_f64 = cpu::spmv_simd::row_f64(&col_idx, &values_f64, &x_f64, start, end);
        let expected_f64 =
            cpu::spmv_simd::row_f64_scalar_reference(&col_idx, &values_f64, &x_f64, start, end);
        assert!((actual_f64 - expected_f64).abs() <= 2.0e-13);
        cases.push(format!("spmv-row-f32-len-{length}"));
        cases.push(format!("spmv-row-f64-len-{length}"));
    }
    cases
}

fn negative_gemv_shape() -> NegativeProbe {
    let backend = CpuBackend::new();
    let mut a = backend.raw_alloc::<f64>(4).unwrap();
    let mut x = backend.raw_alloc::<f64>(2).unwrap();
    let mut y = backend.raw_alloc::<f64>(2).unwrap();
    backend.copy_h2d(&[1.0, 0.0, 0.0, 1.0], &mut a);
    backend.copy_h2d(&[3.0, 4.0], &mut x);
    backend.copy_h2d(&[7.0, 11.0], &mut y);
    let error = backend
        .try_gemv(Transpose::NoTrans, 2, 2, 1.0, &a, 1, &x, 0.0, &mut y)
        .expect_err("invalid GEMV leading dimension");
    let mut observed = [0.0; 2];
    backend.copy_d2h(&y, &mut observed);
    let preserved = observed == [7.0, 11.0];
    let retry = backend
        .try_gemv(Transpose::NoTrans, 2, 2, 1.0, &a, 2, &x, 0.0, &mut y)
        .is_ok();
    record_negative(
        "gemv-invalid-lda",
        "gemv",
        "InvalidArgument",
        &error,
        preserved,
        retry,
    )
}

fn negative_gemm_shape() -> NegativeProbe {
    let backend = CpuBackend::new();
    let mut a = backend.raw_alloc::<f64>(4).unwrap();
    let mut b = backend.raw_alloc::<f64>(4).unwrap();
    let mut c = backend.raw_alloc::<f64>(4).unwrap();
    backend.copy_h2d(&[1.0, 0.0, 0.0, 1.0], &mut a);
    backend.copy_h2d(&[2.0, 1.0, 1.0, 2.0], &mut b);
    backend.copy_h2d(&[7.0, 11.0, 13.0, 17.0], &mut c);
    let error = backend
        .try_gemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            1.0,
            &a,
            2,
            &b,
            2,
            0.0,
            &mut c,
            1,
        )
        .expect_err("invalid GEMM leading dimension");
    let mut observed = [0.0; 4];
    backend.copy_d2h(&c, &mut observed);
    let preserved = observed == [7.0, 11.0, 13.0, 17.0];
    let retry = backend
        .try_gemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            1.0,
            &a,
            2,
            &b,
            2,
            0.0,
            &mut c,
            2,
        )
        .is_ok();
    record_negative(
        "gemm-invalid-ldc",
        "gemm",
        "InvalidArgument",
        &error,
        preserved,
        retry,
    )
}

fn negative_prebake_and_spmv() -> Vec<NegativeProbe> {
    let backend = CpuBackend::new();
    let mut row_ptr = backend.raw_alloc::<i32>(3).unwrap();
    let mut col_idx = backend.raw_alloc::<i32>(1).unwrap();
    let mut values = backend.raw_alloc::<f64>(1).unwrap();
    backend.copy_h2d(&[0, 1, 0], &mut row_ptr);
    backend.copy_h2d(&[0], &mut col_idx);
    backend.copy_h2d(&[2.0], &mut values);
    let malformed = backend
        .try_prebake_csr(2, 2, &row_ptr, &col_idx, &values)
        .err()
        .expect("non-monotone CSR row pointer");
    backend.copy_h2d(&[0, 1, 1], &mut row_ptr);
    let prebaked = backend
        .try_prebake_csr(2, 2, &row_ptr, &col_idx, &values)
        .unwrap();
    let mut x = backend.raw_alloc::<f64>(2).unwrap();
    let mut y = backend.raw_alloc::<f64>(2).unwrap();
    backend.copy_h2d(&[3.0, 5.0], &mut x);
    backend.copy_h2d(&[7.0, 11.0], &mut y);
    let mut observed = [0.0; 2];
    backend.copy_d2h(&y, &mut observed);
    let mut probes = vec![record_negative(
        "prebake-nonmonotone-row-pointer",
        "csr_row_buckets",
        "InvalidArgument",
        &malformed,
        observed == [7.0, 11.0],
        true,
    )];

    backend.copy_h2d(&[3.0, f64::NAN], &mut x);
    let error = backend
        .try_spmv_prebaked(&prebaked, &x, &mut y)
        .expect_err("non-finite SpMV input");
    backend.copy_d2h(&y, &mut observed);
    backend.copy_h2d(&[3.0, 5.0], &mut x);
    let retry = backend.try_spmv_prebaked(&prebaked, &x, &mut y).is_ok();
    probes.push(record_negative(
        "spmv-nonfinite-input",
        "spmv_prebaked",
        "OperationFailed",
        &error,
        observed == [7.0, 11.0],
        retry,
    ));
    probes
}

fn negative_spmm_shape() -> NegativeProbe {
    let backend = CpuBackend::new();
    let mut row_ptr = backend.raw_alloc::<i32>(3).unwrap();
    let mut col_idx = backend.raw_alloc::<i32>(1).unwrap();
    let mut values = backend.raw_alloc::<f64>(1).unwrap();
    backend.copy_h2d(&[0, 1, 1], &mut row_ptr);
    backend.copy_h2d(&[0], &mut col_idx);
    backend.copy_h2d(&[2.0], &mut values);
    let prebaked = backend
        .try_prebake_csr(2, 2, &row_ptr, &col_idx, &values)
        .unwrap();
    let mut x = backend.raw_alloc::<f64>(4).unwrap();
    let mut y = backend.raw_alloc::<f64>(4).unwrap();
    backend.copy_h2d(&[3.0, 5.0, 7.0, 11.0], &mut x);
    backend.copy_h2d(&[13.0, 17.0, 19.0, 23.0], &mut y);
    let error = backend
        .try_spmm_prebaked(&prebaked, 2, &x, 1, &mut y, 2)
        .expect_err("invalid SpMM input stride");
    let mut observed = [0.0; 4];
    backend.copy_d2h(&y, &mut observed);
    let retry = backend
        .try_spmm_prebaked(&prebaked, 2, &x, 2, &mut y, 2)
        .is_ok();
    record_negative(
        "spmm-invalid-input-stride",
        "spmm_prebaked",
        "InvalidArgument",
        &error,
        observed == [13.0, 17.0, 19.0, 23.0],
        retry,
    )
}

fn negative_gather_index() -> NegativeProbe {
    let backend = CpuBackend::new();
    let mut values = backend.raw_alloc::<f64>(4).unwrap();
    let mut indices = backend.raw_alloc::<i32>(4).unwrap();
    let mut weights = backend.raw_alloc::<f64>(4).unwrap();
    let mut output = backend.raw_alloc::<f64>(1).unwrap();
    backend.copy_h2d(&[1.0, 2.0, 3.0, 4.0], &mut values);
    backend.copy_h2d(&[0, 1, 2, 9], &mut indices);
    backend.copy_h2d(&[1.0, 0.0, 0.0, 0.0], &mut weights);
    backend.copy_h2d(&[13.0], &mut output);
    let error = backend
        .try_gather4_weighted(&values, &indices, &weights, &mut output)
        .expect_err("out-of-range gather index");
    let mut observed = [0.0; 1];
    backend.copy_d2h(&output, &mut observed);
    backend.copy_h2d(&[0, 1, 2, 3], &mut indices);
    let retry = backend
        .try_gather4_weighted(&values, &indices, &weights, &mut output)
        .is_ok();
    record_negative(
        "gather-out-of-range-index",
        "gather4_weighted",
        "InvalidArgument",
        &error,
        observed == [13.0],
        retry,
    )
}

fn negative_cast_length() -> NegativeProbe {
    let backend = CpuBackend::new();
    let mut destination = backend.raw_alloc::<f32>(2).unwrap();
    backend.copy_h2d(&[5.0, 7.0], &mut destination);
    let error = backend
        .try_copy_h2d_cast::<f64>(&[2.0], &mut destination)
        .expect_err("short precision-cast input");
    let mut observed = [0.0; 2];
    backend.copy_d2h(&destination, &mut observed);
    let retry = backend
        .try_copy_h2d_cast::<f64>(&[2.0, 3.0], &mut destination)
        .is_ok();
    record_negative(
        "cast-short-input",
        "copy_h2d_cast",
        "InvalidArgument",
        &error,
        observed == [5.0, 7.0],
        retry,
    )
}

fn negative_bem_pair_inputs() -> Vec<NegativeProbe> {
    let backend = CpuBackend::new();
    let degenerate = BemRegularPairInF64::from_triangle(
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    );
    let error = backend
        .try_bem_regular_pair_c64_batch(&[degenerate], 1.0, 1.0)
        .expect_err("degenerate BEM triangle");
    let valid = BemRegularPairInF64::from_triangle(
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    );
    let retry = backend
        .try_bem_regular_pair_c64_batch(&[valid], 1.0, 1.0)
        .is_ok();
    let mut probes = vec![record_negative(
        "bem-degenerate-triangle",
        "bem_regular_pair_c64_batch",
        "InvalidArgument",
        &error,
        true,
        retry,
    )];
    let error = backend
        .try_bem_regular_pair_c64_batch(&[valid], f64::NAN, 1.0)
        .expect_err("non-finite BEM wave number");
    let retry = backend
        .try_bem_regular_pair_c64_batch(&[valid], 1.0, 1.0)
        .is_ok();
    probes.push(record_negative(
        "bem-nonfinite-wave-number",
        "bem_regular_pair_c64_batch",
        "OperationFailed",
        &error,
        true,
        retry,
    ));
    probes
}

fn negative_view_bounds() -> NegativeProbe {
    let backend = CpuBackend::new();
    let buffer = backend.raw_alloc::<f64>(4).unwrap();
    let error = backend
        .checked_view(&buffer, 3, 2, 1)
        .expect_err("view extends beyond allocation");
    let retry = backend.checked_view(&buffer, 1, 2, 1).is_ok();
    record_negative(
        "view-out-of-bounds",
        "buffer_view",
        "InvalidArgument",
        &error,
        true,
        retry,
    )
}

#[test]
#[ignore = "native Linux x86_64 nightly AddressSanitizer runner only"]
fn native_x86_sanitizer_negative_corpus_v1() {
    assert_eq!(std::env::consts::OS, "linux");
    assert!(std::is_x86_feature_detected!("avx2"));
    assert!(std::is_x86_feature_detected!("fma"));
    assert_eq!(
        std::env::var("SONIUM_C5_SANITIZER").as_deref(),
        Ok(SANITIZER)
    );
    assert_eq!(std::env::var("SONIUM_REQUIRE_AVX2_FMA").as_deref(), Ok("1"));

    let positive_cases = probe_spmv_tail_cases();
    let mut negative_cases = vec![negative_gemv_shape(), negative_gemm_shape()];
    negative_cases.extend(negative_prebake_and_spmv());
    negative_cases.push(negative_spmm_shape());
    negative_cases.push(negative_gather_index());
    negative_cases.push(negative_cast_length());
    negative_cases.extend(negative_bem_pair_inputs());
    negative_cases.push(negative_view_bounds());

    let artifact = SanitizerArtifact {
        schema: SCHEMA,
        artifact_kind: "sonium-compute.c5.native-x86-sanitizer-negative",
        fixture_version: FIXTURE_VERSION,
        deterministic_fixture: DETERMINISTIC_FIXTURE,
        status: "passed",
        target_arch: "x86_64",
        target_os: "linux",
        cpu_model: cpuinfo_value("model name"),
        microcode: cpuinfo_value("microcode"),
        compiler: compiler_version(),
        thread_policy: std::env::var("SONIUM_C5_THREAD_POLICY").expect("C5 thread policy"),
        affinity_policy: std::env::var("SONIUM_C5_AFFINITY_POLICY").expect("C5 affinity policy"),
        numa_policy: std::env::var("SONIUM_C5_NUMA_POLICY").expect("C5 NUMA policy"),
        runtime_avx2: true,
        runtime_fma: true,
        sanitizer: SANITIZER,
        sanitizer_options: format!(
            "ASAN_OPTIONS={};LSAN_OPTIONS={}",
            std::env::var("ASAN_OPTIONS").unwrap_or_else(|_| "unset".into()),
            std::env::var("LSAN_OPTIONS").unwrap_or_else(|_| "unset".into())
        ),
        leak_detection: "disabled_by_runner_in_traced_container",
        positive_case_count: positive_cases.len(),
        positive_cases,
        negative_case_count: negative_cases.len(),
        negative_cases,
        cases_failed: 0,
        corpus_sha256: String::new(),
        unsupported_scope: "no AVX512, Arm64, Windows, external CI, or vendor claim",
    };
    let mut artifact = artifact;
    artifact.corpus_sha256 = digest_corpus(&artifact.positive_cases, &artifact.negative_cases);
    let encoded = serde_json::to_string_pretty(&artifact).expect("serialize C5 sanitizer artifact");
    let output_dir = std::env::var("SONIUM_OUTPUT_DIR").expect("SONIUM_OUTPUT_DIR is required");
    let output_dir = std::path::Path::new(&output_dir);
    std::fs::create_dir_all(output_dir).expect("create C5 sanitizer output directory");
    std::fs::write(
        output_dir.join("native-x86-sanitizer-negative.json"),
        format!("{encoded}\n"),
    )
    .expect("write C5 sanitizer artifact");
    println!(
        "sonium_c5_sanitizer_evidence={{\"status\":\"passed\",\"positive_cases\":{},\"negative_cases\":{},\"corpus_sha256\":\"{}\"}}",
        artifact.positive_case_count, artifact.negative_case_count, artifact.corpus_sha256
    );
}
