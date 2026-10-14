//! Deterministic host-only C4 memory-quota contention evidence.
//!
//! This producer exercises the production `MemoryPool<B>` allocator through
//! its CPU backend because the C4 GPU lane is serialized with the real-device
//! validation. Four host threads first contend on an intentionally over-cap
//! allocation, then perform turn-gated retries and fragmented-size reuse.
//! The artifact explicitly records that CUDA was not executed.

use serde::Serialize;
use sha2::{Digest, Sha256};
use sonium_compute::{Backend, CpuBackend, MemoryPool, MemoryPoolConfig, OomError};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

const SCHEMA: u32 = 1;
const FIXTURE_VERSION: u32 = 1;
const FIXTURE: &str = "c4-memory-quota-contention-v1";
const THREAD_COUNT: usize = 4;
const CAP_BYTES: usize = 4096;
const ANCHOR_BYTES: usize = 3584;
const FRAGMENT_BYTES: [usize; 2] = [256, 256];
const REQUEST_BYTES: usize = 1024;
const REQUEST_ELEMENTS: usize = REQUEST_BYTES / std::mem::size_of::<f64>();
const REUSE_ELEMENTS: [usize; 4] = [32, 80, 16, 64];

#[derive(Debug, Serialize)]
struct ThreadProbe {
    worker_id: usize,
    accepted_attempts: usize,
    rejected_attempts: usize,
    typed_quota_failures: usize,
    live_bytes_before_rejection: usize,
    live_bytes_after_rejection: usize,
    retry_successes: usize,
    reuse_operations: usize,
    retry_buffer_bytes: usize,
    destination_preserved: bool,
}

#[derive(Debug, Serialize)]
struct MemoryArtifact {
    schema: u32,
    artifact_kind: &'static str,
    fixture_version: u32,
    deterministic_fixture: &'static str,
    status: &'static str,
    target_os: &'static str,
    target_arch: &'static str,
    backend: &'static str,
    cuda_executed: bool,
    cpu_model: String,
    compiler: String,
    thread_count: usize,
    thread_policy: &'static str,
    cap_bytes: usize,
    anchor_bytes: usize,
    fragmentation_blocks_bytes: [usize; 2],
    contended_request_bytes: usize,
    worker_probes: Vec<ThreadProbe>,
    accepted_attempts: usize,
    rejected_attempts: usize,
    typed_quota_failures: usize,
    rejected_live_bytes_preserved: bool,
    anchor_contents_preserved: bool,
    retry_successes: usize,
    reuse_operations: usize,
    high_water_bytes: usize,
    baseline_live_bytes: usize,
    final_live_bytes: usize,
    corpus_sha256: String,
    unsupported_scope: &'static str,
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
        .arg("-vV")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|output| output.lines().next().map(str::to_owned))
        .unwrap_or_else(|| "unknown".into())
}

fn corpus_digest(probes: &[ThreadProbe]) -> String {
    let bytes = serde_json::to_vec(&(
        CAP_BYTES,
        ANCHOR_BYTES,
        FRAGMENT_BYTES,
        REQUEST_BYTES,
        probes,
    ))
    .expect("serialize C4 memory corpus");
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn run_worker(
    worker_id: usize,
    pool: Arc<MemoryPool<CpuBackend>>,
    contention_gate: Arc<Barrier>,
    completion_gate: Arc<Barrier>,
    release_gate: Arc<Barrier>,
    done_gate: Arc<Barrier>,
    retry_turn: Arc<AtomicUsize>,
) -> ThreadProbe {
    // All workers enter the allocator at the same phase. The anchor and two
    // released fragments leave 3584 live bytes, so every 1024-byte request is
    // deterministically rejected before backend allocation.
    contention_gate.wait();
    let before = pool.bytes_in_use();
    let error = pool
        .try_alloc::<f64>(REQUEST_ELEMENTS)
        .expect_err("the contended request must exceed the fixed quota");
    let after = pool.bytes_in_use();
    let typed = matches!(
        &error,
        OomError::QuotaExceeded {
            requested: REQUEST_BYTES,
            in_use: ANCHOR_BYTES,
            cap: CAP_BYTES,
        }
    );
    assert!(
        typed,
        "worker {worker_id}: unexpected quota error {error:?}"
    );
    assert_eq!(before, ANCHOR_BYTES);
    assert_eq!(after, before, "quota rejection changed live bytes");
    completion_gate.wait();

    // Release is performed by the coordinator after it has observed that all
    // four rejection paths completed. A deterministic turn makes every retry
    // succeed without making the artifact depend on scheduler timing.
    release_gate.wait();
    while retry_turn.load(Ordering::Acquire) != worker_id {
        std::hint::spin_loop();
    }
    let mut retry_buffer = pool
        .try_alloc::<f64>(REQUEST_ELEMENTS)
        .expect("released quota must make the retry succeed");
    assert_eq!(pool.bytes_in_use(), REQUEST_BYTES);
    let value = (worker_id + 1) as f64;
    let source = vec![value; REQUEST_ELEMENTS];
    pool.backend().copy_h2d(&source, retry_buffer.raw_mut());
    let mut destination = vec![0.0; REQUEST_ELEMENTS];
    pool.backend()
        .copy_d2h(retry_buffer.raw(), &mut destination);
    let destination_preserved = destination == source;
    assert!(
        destination_preserved,
        "retry buffer contents were not preserved"
    );
    drop(retry_buffer);

    let mut reuse_operations = 0;
    for (index, &elements) in REUSE_ELEMENTS.iter().enumerate() {
        let mut buffer = pool
            .try_alloc::<f64>(elements)
            .expect("fragmented-size reuse must fit after retry release");
        let value = (worker_id * 10 + index + 1) as f64;
        let source = vec![value; elements];
        pool.backend().copy_h2d(&source, buffer.raw_mut());
        let mut observed = vec![0.0; elements];
        pool.backend().copy_d2h(buffer.raw(), &mut observed);
        assert_eq!(observed, source, "reuse buffer contents changed");
        drop(buffer);
        reuse_operations += 1;
    }
    retry_turn.fetch_add(1, Ordering::AcqRel);
    done_gate.wait();

    ThreadProbe {
        worker_id,
        accepted_attempts: 0,
        rejected_attempts: 1,
        typed_quota_failures: usize::from(typed),
        live_bytes_before_rejection: before,
        live_bytes_after_rejection: after,
        retry_successes: 1,
        reuse_operations,
        retry_buffer_bytes: REQUEST_BYTES,
        destination_preserved,
    }
}

#[test]
fn c4_memory_quota_contention_evidence_v1() {
    assert_eq!(std::env::consts::OS, "linux");
    assert_eq!(std::env::consts::ARCH, "x86_64");
    assert_eq!(
        ANCHOR_BYTES + FRAGMENT_BYTES[0] + FRAGMENT_BYTES[1],
        CAP_BYTES
    );
    assert_eq!(REQUEST_BYTES % std::mem::size_of::<f64>(), 0);

    let pool = Arc::new(
        MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(CAP_BYTES))
            .expect("fixed-cap CPU memory pool"),
    );
    assert_eq!(pool.bytes_in_use(), 0);
    let mut anchor = pool
        .try_alloc::<f64>(ANCHOR_BYTES / std::mem::size_of::<f64>())
        .expect("anchor allocation");
    let anchor_source = vec![3.25; anchor.len()];
    pool.backend().copy_h2d(&anchor_source, anchor.raw_mut());
    let fragment_a = pool
        .try_alloc::<f64>(FRAGMENT_BYTES[0] / std::mem::size_of::<f64>())
        .expect("first fragmentation block");
    let fragment_b = pool
        .try_alloc::<f64>(FRAGMENT_BYTES[1] / std::mem::size_of::<f64>())
        .expect("second fragmentation block");
    assert_eq!(pool.bytes_in_use(), CAP_BYTES);
    assert_eq!(pool.high_water_bytes(), CAP_BYTES);
    drop(fragment_a);
    drop(fragment_b);
    assert_eq!(pool.bytes_in_use(), ANCHOR_BYTES);

    let contention_gate = Arc::new(Barrier::new(THREAD_COUNT + 1));
    let completion_gate = Arc::new(Barrier::new(THREAD_COUNT + 1));
    let release_gate = Arc::new(Barrier::new(THREAD_COUNT + 1));
    let done_gate = Arc::new(Barrier::new(THREAD_COUNT + 1));
    let retry_turn = Arc::new(AtomicUsize::new(0));
    let mut workers = Vec::with_capacity(THREAD_COUNT);
    for worker_id in 0..THREAD_COUNT {
        workers.push(thread::spawn({
            let pool = Arc::clone(&pool);
            let contention_gate = Arc::clone(&contention_gate);
            let completion_gate = Arc::clone(&completion_gate);
            let release_gate = Arc::clone(&release_gate);
            let done_gate = Arc::clone(&done_gate);
            let retry_turn = Arc::clone(&retry_turn);
            move || {
                run_worker(
                    worker_id,
                    pool,
                    contention_gate,
                    completion_gate,
                    release_gate,
                    done_gate,
                    retry_turn,
                )
            }
        }));
    }
    contention_gate.wait();
    completion_gate.wait();

    let mut anchor_observed = vec![0.0; anchor.len()];
    pool.backend().copy_d2h(anchor.raw(), &mut anchor_observed);
    let anchor_contents_preserved = anchor_observed == anchor_source;
    assert!(
        anchor_contents_preserved,
        "rejected allocations changed anchor"
    );
    drop(anchor);
    assert_eq!(pool.bytes_in_use(), 0);
    release_gate.wait();
    done_gate.wait();

    let mut probes = workers
        .into_iter()
        .map(|worker| worker.join().expect("C4 worker must not panic"))
        .collect::<Vec<_>>();
    probes.sort_by_key(|probe| probe.worker_id);
    assert_eq!(probes.len(), THREAD_COUNT);
    assert!(probes.iter().all(|probe| probe.worker_id < THREAD_COUNT));
    assert!(probes.iter().all(|probe| probe.accepted_attempts == 0));
    assert!(probes.iter().all(|probe| probe.rejected_attempts == 1));
    assert!(probes.iter().all(|probe| probe.typed_quota_failures == 1));
    assert!(
        probes
            .iter()
            .all(|probe| probe.live_bytes_before_rejection == ANCHOR_BYTES)
    );
    assert!(
        probes
            .iter()
            .all(|probe| probe.live_bytes_after_rejection == ANCHOR_BYTES)
    );
    assert!(probes.iter().all(|probe| probe.retry_successes == 1));
    assert!(
        probes
            .iter()
            .all(|probe| probe.reuse_operations == REUSE_ELEMENTS.len())
    );
    assert!(probes.iter().all(|probe| probe.destination_preserved));

    let accepted_attempts = probes.iter().map(|probe| probe.accepted_attempts).sum();
    let rejected_attempts = probes.iter().map(|probe| probe.rejected_attempts).sum();
    let typed_quota_failures = probes.iter().map(|probe| probe.typed_quota_failures).sum();
    let retry_successes = probes.iter().map(|probe| probe.retry_successes).sum();
    let reuse_operations = probes.iter().map(|probe| probe.reuse_operations).sum();
    let final_live_bytes = pool.bytes_in_use();
    let high_water_bytes = pool.high_water_bytes();
    assert_eq!(accepted_attempts, 0);
    assert_eq!(rejected_attempts, THREAD_COUNT);
    assert_eq!(typed_quota_failures, THREAD_COUNT);
    assert_eq!(retry_successes, THREAD_COUNT);
    assert_eq!(reuse_operations, THREAD_COUNT * REUSE_ELEMENTS.len());
    assert_eq!(final_live_bytes, 0);
    assert_eq!(high_water_bytes, CAP_BYTES);

    let artifact = MemoryArtifact {
        schema: SCHEMA,
        artifact_kind: "sonium-compute.c4.memory-quota-contention",
        fixture_version: FIXTURE_VERSION,
        deterministic_fixture: FIXTURE,
        status: "passed",
        target_os: "linux",
        target_arch: "x86_64",
        backend: "cpu-reference",
        cuda_executed: false,
        cpu_model: cpuinfo_value("model name"),
        compiler: compiler_version(),
        thread_count: THREAD_COUNT,
        thread_policy: "four-host-threads;turn-gated-retry",
        cap_bytes: CAP_BYTES,
        anchor_bytes: ANCHOR_BYTES,
        fragmentation_blocks_bytes: FRAGMENT_BYTES,
        contended_request_bytes: REQUEST_BYTES,
        worker_probes: probes,
        accepted_attempts,
        rejected_attempts,
        typed_quota_failures,
        rejected_live_bytes_preserved: true,
        anchor_contents_preserved,
        retry_successes,
        reuse_operations,
        high_water_bytes,
        baseline_live_bytes: 0,
        final_live_bytes,
        corpus_sha256: String::new(),
        unsupported_scope: "no CUDA execution, multi-GPU, external CI, energy, or vendor claim",
    };
    let mut artifact = artifact;
    artifact.corpus_sha256 = corpus_digest(&artifact.worker_probes);
    let output_dir = std::env::var("SONIUM_OUTPUT_DIR")
        .unwrap_or_else(|_| "target/sonium-unix-c4-memory".into());
    let output_dir = std::path::Path::new(&output_dir);
    std::fs::create_dir_all(output_dir).expect("create C4 memory artifact directory");
    let encoded = serde_json::to_string_pretty(&artifact).expect("serialize C4 memory artifact");
    std::fs::write(
        output_dir.join("c4-memory-quota-contention.json"),
        format!("{encoded}\n"),
    )
    .expect("write C4 memory artifact");
    println!(
        "c4_memory_quota_evidence={{\"status\":\"passed\",\"threads\":{},\"accepted\":{},\"rejected\":{},\"retries\":{},\"high_water_bytes\":{},\"final_live_bytes\":{},\"corpus_sha256\":\"{}\"}}",
        artifact.thread_count,
        artifact.accepted_attempts,
        artifact.rejected_attempts,
        artifact.retry_successes,
        artifact.high_water_bytes,
        artifact.final_live_bytes,
        artifact.corpus_sha256,
    );
}
