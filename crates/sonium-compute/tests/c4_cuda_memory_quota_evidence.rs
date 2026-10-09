//! Genuine one-device CUDA quota/reuse evidence for C4.
//!
//! This test is intentionally separate from the host-only allocator corpus.
//! It requires a real CUDA device when executed, uses the production
//! `MemoryPool<CudaBackend>`, and fails closed when CUDA is unavailable. The
//! test is one-host-thread/context-affine by contract; the host-only corpus
//! covers contention without making an unsafe cross-thread CUDA claim.

#![cfg(feature = "cuda")]

use serde::Serialize;
use sha2::{Digest, Sha256};
use sonium_compute::{Backend, CudaBackend, MemoryPool, MemoryPoolConfig, OomError};
use std::process::Command;

const SCHEMA: u32 = 1;
const FIXTURE_VERSION: u32 = 1;
const FIXTURE: &str = "c4-cuda-memory-quota-reuse-v1";
const DEVICE_ORDINAL: usize = 0;
const CAP_BYTES: usize = 16 * 1024;
const ANCHOR_BYTES: usize = 12 * 1024;
const FRAGMENT_BYTES: [usize; 2] = [2 * 1024, 2 * 1024];
const REQUEST_BYTES: usize = 4 * 1024;
const _: () = assert!(CAP_BYTES < 16 * 1024 * 1024);

#[derive(Debug, Serialize)]
struct CudaProbe {
    rejection_attempts: usize,
    typed_quota_failures: usize,
    pool_live_bytes_before_rejection: usize,
    pool_live_bytes_after_rejection: usize,
    device_live_bytes_before_rejection: usize,
    device_live_bytes_after_rejection: usize,
    retry_successes: usize,
    retry_buffer_bytes: usize,
    destination_preserved: bool,
}

#[derive(Debug, Serialize)]
struct CudaMemoryArtifact {
    schema: u32,
    artifact_kind: &'static str,
    fixture_version: u32,
    deterministic_fixture: &'static str,
    status: &'static str,
    target_os: &'static str,
    target_arch: &'static str,
    backend: &'static str,
    cuda_executed: bool,
    device_ordinal: usize,
    total_device_memory_bytes: usize,
    compiler: String,
    host_thread_count: usize,
    thread_policy: &'static str,
    cap_bytes: usize,
    anchor_bytes: usize,
    fragmentation_blocks_bytes: [usize; 2],
    request_bytes: usize,
    baseline_pool_live_bytes: usize,
    baseline_device_live_bytes: usize,
    probe: CudaProbe,
    rejected_live_bytes_preserved: bool,
    high_water_bytes: usize,
    backend_high_water_bytes: usize,
    retry_successes: usize,
    final_pool_live_bytes: usize,
    final_device_live_bytes: usize,
    restored_cap_bytes: usize,
    corpus_sha256: String,
    unsupported_scope: &'static str,
}

fn compiler_version() -> String {
    Command::new("rustc")
        .arg("-vV")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|output| output.lines().next().map(str::to_owned))
        .unwrap_or_else(|| "unknown".into())
}

fn corpus_digest(probe: &CudaProbe) -> String {
    let bytes = serde_json::to_vec(&(
        DEVICE_ORDINAL,
        CAP_BYTES,
        ANCHOR_BYTES,
        FRAGMENT_BYTES,
        REQUEST_BYTES,
        probe,
    ))
    .expect("serialize CUDA memory corpus");
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn c4_cuda_memory_quota_reuse_evidence_v1() {
    assert_eq!(std::env::consts::OS, "linux");
    assert_eq!(std::env::consts::ARCH, "x86_64");
    assert_eq!(
        ANCHOR_BYTES + FRAGMENT_BYTES[0] + FRAGMENT_BYTES[1],
        CAP_BYTES
    );

    let backend = CudaBackend::try_new(DEVICE_ORDINAL)
        .expect("C4 CUDA allocator evidence requires a real CUDA device");
    let total_device_memory = backend
        .total_device_memory_bytes()
        .expect("CUDA backend must report total device memory");
    assert!(
        total_device_memory >= CAP_BYTES,
        "device is smaller than the bounded CUDA fixture"
    );
    let baseline_device_live_bytes = backend.live_allocation_bytes();
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(CAP_BYTES))
        .expect("fixed-cap CUDA memory pool");
    let baseline_pool_live_bytes = pool.bytes_in_use();
    assert_eq!(baseline_pool_live_bytes, 0);

    let mut anchor = pool
        .try_alloc::<f64>(ANCHOR_BYTES / std::mem::size_of::<f64>())
        .expect("CUDA anchor allocation");
    let anchor_source = vec![3.25; anchor.len()];
    pool.backend()
        .try_copy_h2d(&anchor_source, anchor.raw_mut())
        .expect("CUDA anchor upload");
    pool.backend()
        .try_synchronize()
        .expect("CUDA anchor upload synchronization");
    let fragment_a = pool
        .try_alloc::<f64>(FRAGMENT_BYTES[0] / std::mem::size_of::<f64>())
        .expect("first CUDA fragmentation block");
    let fragment_b = pool
        .try_alloc::<f64>(FRAGMENT_BYTES[1] / std::mem::size_of::<f64>())
        .expect("second CUDA fragmentation block");
    assert_eq!(pool.bytes_in_use(), CAP_BYTES);
    assert_eq!(pool.high_water_bytes(), CAP_BYTES);
    assert_eq!(
        pool.backend().live_allocation_bytes(),
        baseline_device_live_bytes + CAP_BYTES
    );
    drop(fragment_a);
    assert_eq!(pool.bytes_in_use(), ANCHOR_BYTES + FRAGMENT_BYTES[1]);

    let pool_before = pool.bytes_in_use();
    let device_before = pool.backend().live_allocation_bytes();
    let error = pool
        .try_alloc::<f64>(REQUEST_BYTES / std::mem::size_of::<f64>())
        .expect_err("over-cap CUDA allocation must fail before driver allocation");
    let pool_after = pool.bytes_in_use();
    let device_after = pool.backend().live_allocation_bytes();
    assert!(
        matches!(
            &error,
            OomError::QuotaExceeded {
                requested: REQUEST_BYTES,
                in_use: value,
                cap: CAP_BYTES,
            } if *value == pool_before
        ),
        "unexpected CUDA quota error: {error:?}"
    );
    assert_eq!(pool_after, pool_before);
    assert_eq!(device_after, device_before);

    let mut anchor_observed = vec![0.0; anchor.len()];
    pool.backend()
        .try_copy_d2h(anchor.raw(), &mut anchor_observed)
        .expect("CUDA anchor download after rejection");
    pool.backend()
        .try_synchronize()
        .expect("CUDA anchor download synchronization");
    let destination_preserved = anchor_observed == anchor_source;
    assert!(
        destination_preserved,
        "CUDA anchor changed after quota rejection"
    );

    drop(fragment_b);
    assert_eq!(pool.bytes_in_use(), ANCHOR_BYTES);
    let mut retry = pool
        .try_alloc::<f64>(REQUEST_BYTES / std::mem::size_of::<f64>())
        .expect("released CUDA quota must make retry succeed");
    assert_eq!(pool.bytes_in_use(), CAP_BYTES);
    let retry_source = vec![7.5; retry.len()];
    pool.backend()
        .try_copy_h2d(&retry_source, retry.raw_mut())
        .expect("CUDA retry upload");
    pool.backend()
        .try_synchronize()
        .expect("CUDA retry upload synchronization");
    let mut retry_observed = vec![0.0; retry.len()];
    pool.backend()
        .try_copy_d2h(retry.raw(), &mut retry_observed)
        .expect("CUDA retry download");
    pool.backend()
        .try_synchronize()
        .expect("CUDA retry download synchronization");
    assert_eq!(retry_observed, retry_source);
    drop(retry);
    drop(anchor);

    let final_pool_live_bytes = pool.bytes_in_use();
    let final_device_live_bytes = pool.backend().live_allocation_bytes();
    let high_water_bytes = pool.high_water_bytes();
    let backend_high_water_bytes = pool.backend().allocation_high_water_bytes();
    assert_eq!(final_pool_live_bytes, baseline_pool_live_bytes);
    assert_eq!(final_device_live_bytes, baseline_device_live_bytes);
    assert_eq!(high_water_bytes, CAP_BYTES);
    assert!(backend_high_water_bytes >= CAP_BYTES);
    pool.backend().set_allocation_cap_bytes(total_device_memory);
    let restored_cap_bytes = total_device_memory;

    let probe = CudaProbe {
        rejection_attempts: 1,
        typed_quota_failures: 1,
        pool_live_bytes_before_rejection: pool_before,
        pool_live_bytes_after_rejection: pool_after,
        device_live_bytes_before_rejection: device_before,
        device_live_bytes_after_rejection: device_after,
        retry_successes: 1,
        retry_buffer_bytes: REQUEST_BYTES,
        destination_preserved,
    };
    let artifact = CudaMemoryArtifact {
        schema: SCHEMA,
        artifact_kind: "sonium-compute.c4.cuda-memory-quota",
        fixture_version: FIXTURE_VERSION,
        deterministic_fixture: FIXTURE,
        status: "passed",
        target_os: "linux",
        target_arch: "x86_64",
        backend: "cuda",
        cuda_executed: true,
        device_ordinal: DEVICE_ORDINAL,
        total_device_memory_bytes: total_device_memory,
        compiler: compiler_version(),
        host_thread_count: 1,
        thread_policy: "single-host-thread;cuda-context-affine",
        cap_bytes: CAP_BYTES,
        anchor_bytes: ANCHOR_BYTES,
        fragmentation_blocks_bytes: FRAGMENT_BYTES,
        request_bytes: REQUEST_BYTES,
        baseline_pool_live_bytes,
        baseline_device_live_bytes,
        probe,
        rejected_live_bytes_preserved: pool_after == pool_before && device_after == device_before,
        high_water_bytes,
        backend_high_water_bytes,
        retry_successes: 1,
        final_pool_live_bytes,
        final_device_live_bytes,
        restored_cap_bytes,
        corpus_sha256: String::new(),
        unsupported_scope: "no multi-GPU, external CI, energy, or vendor claim",
    };
    let mut artifact = artifact;
    artifact.corpus_sha256 = corpus_digest(&artifact.probe);
    let output_dir = std::env::var("SONIUM_OUTPUT_DIR")
        .unwrap_or_else(|_| "target/sonium-unix-c4-memory/cuda".into());
    let output_dir = std::path::Path::new(&output_dir);
    std::fs::create_dir_all(output_dir).expect("create CUDA memory artifact directory");
    let encoded = serde_json::to_string_pretty(&artifact).expect("serialize CUDA memory artifact");
    std::fs::write(
        output_dir.join("c4-cuda-memory-quota.json"),
        format!("{encoded}\n"),
    )
    .expect("write CUDA memory artifact");
    println!(
        "c4_cuda_memory_quota_evidence={{\"status\":\"passed\",\"device_ordinal\":{},\"rejection_attempts\":{},\"retry_successes\":{},\"high_water_bytes\":{},\"final_device_live_bytes\":{},\"corpus_sha256\":\"{}\"}}",
        artifact.device_ordinal,
        artifact.probe.rejection_attempts,
        artifact.retry_successes,
        artifact.high_water_bytes,
        artifact.final_device_live_bytes,
        artifact.corpus_sha256,
    );
}
