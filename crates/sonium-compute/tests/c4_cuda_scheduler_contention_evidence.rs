//! Native single-device scheduler contention evidence for C4.
//!
//! This is deliberately separate from the allocator-only C4 corpus. Two host
//! submitters share one scheduler budget and submit the same capture-safe
//! native CUDA graph. The first submitter holds its lease at a barrier; the
//! second therefore receives a typed quota error, then retries after the
//! first native completion refunds the lease. No multi-device or reset claim
//! is made by this fixture.

#![cfg(feature = "cuda")]

use serde::Serialize;
use sha2::{Digest, Sha256};
use sonium_compute::{
    Backend, BackendError, CudaBackend, CudaGraphReplay, DeviceBudget, DeviceScheduler, MemoryPool,
    MemoryPoolConfig, ScheduleError, WorkRequest,
};
use std::process::Command;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread;

const SCHEMA: &str = "sonium-compute.c4.cuda-scheduler-contention/v1";
const ARTIFACT_KIND: &str = "sonium-compute.c4.cuda-scheduler-contention";
const FIXTURE_VERSION: u32 = 1;
const FIXTURE: &str = "c4-cuda-scheduler-contention-v1";
const DEVICE_ORDINAL: usize = 0;
const BUDGET_BYTES: usize = 4096;
const REQUEST_BYTES: usize = 4096;
const OVER_BUDGET_BYTES: usize = BUDGET_BYTES + 1;
const ACTIVE_SLOTS: usize = 1;
const ELEMENTS: usize = 5;

#[derive(Debug, Serialize, Clone, Copy)]
struct SnapshotRecord {
    bytes_in_use: usize,
    active: usize,
    lost: bool,
    completed: usize,
    cancelled: usize,
    failed: usize,
    device_lost: usize,
}

#[derive(Debug, Serialize, Clone, Copy)]
struct RuntimeRecord {
    kernel_launches: usize,
    library_operations: usize,
    synchronization_calls: usize,
    completion_events: usize,
    completion_waits: usize,
    h2d_operations: usize,
    h2d_bytes: usize,
    d2h_operations: usize,
    d2h_bytes: usize,
    d2d_operations: usize,
    d2d_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct ReplayReport {
    replay_runtime: RuntimeRecord,
    publication_runtime: RuntimeRecord,
    output_matches_reference: bool,
}

#[derive(Debug, Serialize, Clone, Copy)]
struct TypedQuotaError {
    kind: &'static str,
    requested_bytes: usize,
    requested_slots: usize,
}

#[derive(Debug, Serialize)]
struct SchedulerContentionArtifact {
    schema: &'static str,
    artifact_kind: &'static str,
    fixture_version: u32,
    deterministic_fixture: &'static str,
    status: &'static str,
    target_os: &'static str,
    target_arch: &'static str,
    requested_backend: &'static str,
    actual_backend: &'static str,
    cuda_executed: bool,
    device_ordinal: usize,
    device_model: String,
    compute_capability: String,
    compiler: String,
    host_submitter_count: usize,
    thread_policy: &'static str,
    scheduler_device: &'static str,
    budget_bytes: usize,
    budget_active_slots: usize,
    request_bytes: usize,
    request_active_slots: usize,
    over_budget_bytes: usize,
    workload: &'static str,
    workload_elements: usize,
    graph_replays: usize,
    graph_output_matches_reference: bool,
    contention_rejection: TypedQuotaError,
    over_budget_rejection: TypedQuotaError,
    during_contention: SnapshotRecord,
    after_first_refund: SnapshotRecord,
    after_retry: SnapshotRecord,
    replay_runtime: RuntimeRecord,
    publication_runtime: RuntimeRecord,
    high_water_bytes: usize,
    no_stranded_leases: bool,
    evidence_sha256: String,
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

fn device_identity() -> (String, String) {
    let output = Command::new("nvidia-smi")
        .args([
            "--id=0",
            "--query-gpu=name,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .expect("C4 native contention requires nvidia-smi for device identity");
    assert!(
        output.status.success(),
        "nvidia-smi device identity query failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let identity =
        String::from_utf8(output.stdout).expect("nvidia-smi device identity must be UTF-8");
    let line = identity
        .lines()
        .next()
        .expect("nvidia-smi returned no device identity");
    let mut fields = line.split(',').map(str::trim);
    let model = fields
        .next()
        .filter(|value| !value.is_empty())
        .expect("nvidia-smi returned no device model")
        .to_owned();
    let compute_capability = fields
        .next()
        .filter(|value| !value.is_empty())
        .expect("nvidia-smi returned no compute capability")
        .to_owned();
    assert_eq!(
        compute_capability, "12.0",
        "C4 native fixture is pinned to an SM120 device"
    );
    (model, compute_capability)
}

fn snapshot(scheduler: &DeviceScheduler) -> SnapshotRecord {
    let snapshots = scheduler.snapshots().expect("scheduler snapshot");
    assert_eq!(snapshots.len(), 1, "C4 fixture must remain single-device");
    let value = &snapshots[0];
    assert_eq!(value.name, "cuda0");
    assert_eq!(value.max_bytes, BUDGET_BYTES);
    assert_eq!(value.max_active, ACTIVE_SLOTS);
    SnapshotRecord {
        bytes_in_use: value.bytes_in_use,
        active: value.active,
        lost: value.lost,
        completed: value.completed,
        cancelled: value.cancelled,
        failed: value.failed,
        device_lost: value.device_lost,
    }
}

fn runtime(backend: &CudaBackend) -> RuntimeRecord {
    let stats = backend.runtime_stats();
    RuntimeRecord {
        kernel_launches: stats.kernel_launches,
        library_operations: stats.library_operations,
        synchronization_calls: stats.synchronization_calls,
        completion_events: stats.completion_events,
        completion_waits: stats.completion_waits,
        h2d_operations: stats.transfer.h2d_operations,
        h2d_bytes: stats.transfer.h2d_bytes,
        d2h_operations: stats.transfer.d2h_operations,
        d2h_bytes: stats.transfer.d2h_bytes,
        d2d_operations: stats.transfer.d2d_operations,
        d2d_bytes: stats.transfer.d2d_bytes,
    }
}

fn expect_quota(error: ScheduleError, expected_bytes: usize) -> TypedQuotaError {
    match error {
        ScheduleError::QuotaExceeded {
            requested_bytes,
            requested_slots,
        } => {
            assert_eq!(requested_bytes, expected_bytes);
            assert_eq!(requested_slots, ACTIVE_SLOTS);
            TypedQuotaError {
                kind: "quota_exceeded",
                requested_bytes,
                requested_slots,
            }
        }
        other => panic!("expected typed scheduler quota rejection, got {other:?}"),
    }
}

fn graph_submission(
    scheduler: &DeviceScheduler,
    pool: &MemoryPool<CudaBackend>,
    graph: &CudaGraphReplay,
) -> Result<sonium_compute::BackendSubmission<()>, ScheduleError> {
    scheduler.submit_backend(
        WorkRequest::new(REQUEST_BYTES, ACTIVE_SLOTS),
        pool.backend(),
        |_device, _backend| {
            graph.try_launch()?;
            Ok::<_, BackendError>(())
        },
    )
}

fn sum_runtime(first: RuntimeRecord, second: RuntimeRecord) -> RuntimeRecord {
    RuntimeRecord {
        kernel_launches: first.kernel_launches + second.kernel_launches,
        library_operations: first.library_operations + second.library_operations,
        synchronization_calls: first.synchronization_calls + second.synchronization_calls,
        completion_events: first.completion_events + second.completion_events,
        completion_waits: first.completion_waits + second.completion_waits,
        h2d_operations: first.h2d_operations + second.h2d_operations,
        h2d_bytes: first.h2d_bytes + second.h2d_bytes,
        d2h_operations: first.d2h_operations + second.d2h_operations,
        d2h_bytes: first.d2h_bytes + second.d2h_bytes,
        d2d_operations: first.d2d_operations + second.d2d_operations,
        d2d_bytes: first.d2d_bytes + second.d2d_bytes,
    }
}

fn evidence_digest(
    contention: TypedQuotaError,
    over_budget: TypedQuotaError,
    during: SnapshotRecord,
    after_first: SnapshotRecord,
    after_retry: SnapshotRecord,
    replay: RuntimeRecord,
    publication: RuntimeRecord,
) -> String {
    let bytes = serde_json::to_vec(&(
        DEVICE_ORDINAL,
        BUDGET_BYTES,
        REQUEST_BYTES,
        OVER_BUDGET_BYTES,
        contention,
        over_budget,
        during,
        after_first,
        after_retry,
        replay,
        publication,
    ))
    .expect("serialize C4 scheduler contention measurements");
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn write_artifact(artifact: &SchedulerContentionArtifact) {
    let output_dir =
        std::env::var("SONIUM_OUTPUT_DIR").expect("SONIUM_OUTPUT_DIR is required for C4 evidence");
    std::fs::create_dir_all(&output_dir).expect("create C4 evidence directory");
    let path = std::path::Path::new(&output_dir).join("c4-cuda-scheduler-contention.json");
    let json = serde_json::to_string_pretty(artifact).expect("serialize C4 scheduler artifact");
    std::fs::write(path, format!("{json}\n")).expect("write C4 scheduler artifact");
}

fn send_gate(sender: &SyncSender<()>, operation: &'static str) -> Result<(), BackendError> {
    sender.send(()).map_err(|_| BackendError::OperationFailed {
        operation,
        reason: "contention test gate receiver dropped".into(),
    })
}

fn recv_gate(receiver: &Receiver<()>, operation: &'static str) -> Result<(), BackendError> {
    receiver.recv().map_err(|_| BackendError::OperationFailed {
        operation,
        reason: "contention test gate sender dropped".into(),
    })
}

#[test]
fn c4_cuda_scheduler_contention_evidence_v1() {
    assert_eq!(std::env::consts::OS, "linux");
    assert_eq!(std::env::consts::ARCH, "x86_64");
    let (device_model, compute_capability) = device_identity();
    let scheduler = Arc::new(
        DeviceScheduler::new([DeviceBudget::new("cuda0", BUDGET_BYTES, ACTIVE_SLOTS)])
            .expect("single-device scheduler budget"),
    );
    let (admitted_tx, admitted_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let (first_done_tx, first_done_rx) = mpsc::sync_channel(1);
    let (retry_allowed_tx, retry_allowed_rx) = mpsc::sync_channel(1);
    let (first_report_tx, first_report_rx) = mpsc::sync_channel(1);
    let first_scheduler = Arc::clone(&scheduler);
    let first = thread::spawn(move || {
        let backend = CudaBackend::try_new(DEVICE_ORDINAL)
            .expect("C4 native scheduler contention requires a CUDA device");
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20))
            .expect("native CUDA graph fixture pool");
        let mut x_dev = pool
            .try_alloc::<f64>(ELEMENTS)
            .expect("graph input allocation");
        let mut y_dev = pool
            .try_alloc::<f32>(ELEMENTS)
            .expect("graph output allocation");
        pool.backend()
            .try_copy_h2d(&[1.0, 2.0, 4.0, 8.0, 16.0], x_dev.raw_mut())
            .expect("graph input upload");
        pool.backend()
            .try_copy_h2d(&[0.0_f32; ELEMENTS], y_dev.raw_mut())
            .expect("graph output clear");
        pool.backend()
            .try_synchronize()
            .expect("graph fixture upload synchronization");
        let graph = pool
            .backend()
            .try_capture_graph_checked(|backend| {
                backend.try_cast_to_lower(x_dev.raw(), y_dev.raw_mut())
            })
            .expect("capture-safe native cast graph");
        pool.backend().reset_runtime_stats();
        let submission = first_scheduler.submit_backend(
            WorkRequest::new(REQUEST_BYTES, ACTIVE_SLOTS),
            pool.backend(),
            |_device, _backend| {
                send_gate(&admitted_tx, "c4_admission_gate")?;
                recv_gate(&release_rx, "c4_release_gate")?;
                graph.try_launch()?;
                Ok::<_, BackendError>(())
            },
        )?;
        submission.complete()?;
        let replay_runtime = runtime(pool.backend());
        let mut output = [0.0_f32; ELEMENTS];
        pool.backend()
            .try_copy_d2h(y_dev.raw(), &mut output)
            .expect("first native graph result download");
        pool.backend()
            .try_synchronize()
            .expect("first native graph result synchronization");
        let publication_runtime = runtime(pool.backend());
        let output_matches_reference = output == [1.0, 2.0, 4.0, 8.0, 16.0];
        assert!(
            output_matches_reference,
            "first native graph output: {output:?}"
        );
        first_report_tx
            .send(ReplayReport {
                replay_runtime,
                publication_runtime,
                output_matches_reference,
            })
            .map_err(|_| ScheduleError::StatePoisoned)?;
        first_done_tx
            .send(())
            .map_err(|_| ScheduleError::StatePoisoned)?;
        Ok::<_, ScheduleError>(())
    });

    admitted_rx.recv().expect("first submitter admission gate");
    let during_contention = snapshot(&scheduler);
    assert_eq!(during_contention.bytes_in_use, REQUEST_BYTES);
    assert_eq!(during_contention.active, ACTIVE_SLOTS);

    let (second_tx, second_rx) = mpsc::sync_channel(1);
    let (second_report_tx, second_report_rx) = mpsc::sync_channel(1);
    let second_scheduler = Arc::clone(&scheduler);
    let second = thread::spawn(move || {
        let backend = CudaBackend::try_new(DEVICE_ORDINAL)
            .expect("C4 second native submitter requires a CUDA device");
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20))
            .expect("second native CUDA graph fixture pool");
        let mut x_dev = pool
            .try_alloc::<f64>(ELEMENTS)
            .expect("second graph input allocation");
        let mut y_dev = pool
            .try_alloc::<f32>(ELEMENTS)
            .expect("second graph output allocation");
        pool.backend()
            .try_copy_h2d(&[1.0, 2.0, 4.0, 8.0, 16.0], x_dev.raw_mut())
            .expect("second graph input upload");
        pool.backend()
            .try_copy_h2d(&[0.0_f32; ELEMENTS], y_dev.raw_mut())
            .expect("second graph output clear");
        pool.backend()
            .try_synchronize()
            .expect("second graph fixture upload synchronization");
        let graph = pool
            .backend()
            .try_capture_graph_checked(|backend| {
                backend.try_cast_to_lower(x_dev.raw(), y_dev.raw_mut())
            })
            .expect("second capture-safe native cast graph");
        pool.backend().reset_runtime_stats();
        let contention = match second_scheduler.submit_backend(
            WorkRequest::new(REQUEST_BYTES, ACTIVE_SLOTS),
            pool.backend(),
            |_device, _backend| {
                graph.try_launch()?;
                Ok::<_, BackendError>(())
            },
        ) {
            Ok(_) => panic!("held lease must reject the second submitter"),
            Err(error) => error,
        };
        let contention = expect_quota(contention, REQUEST_BYTES);
        second_tx
            .send(contention)
            .map_err(|_| ScheduleError::StatePoisoned)?;
        recv_gate(&first_done_rx, "c4_first_completion_gate")
            .map_err(|_| ScheduleError::StatePoisoned)?;
        recv_gate(&retry_allowed_rx, "c4_retry_release_gate")
            .map_err(|_| ScheduleError::StatePoisoned)?;

        let over_budget = match second_scheduler.submit_backend(
            WorkRequest::new(OVER_BUDGET_BYTES, ACTIVE_SLOTS),
            pool.backend(),
            |_device, _backend| Ok::<_, BackendError>(()),
        ) {
            Ok(_) => panic!("byte-over-budget request must be rejected"),
            Err(error) => error,
        };
        let over_budget = expect_quota(over_budget, OVER_BUDGET_BYTES);
        let retry = graph_submission(&second_scheduler, &pool, &graph)?;
        retry.complete()?;
        let replay_runtime = runtime(pool.backend());
        let mut output = [0.0_f32; ELEMENTS];
        pool.backend()
            .try_copy_d2h(y_dev.raw(), &mut output)
            .expect("second native graph result download");
        pool.backend()
            .try_synchronize()
            .expect("second native graph result synchronization");
        let publication_runtime = runtime(pool.backend());
        let output_matches_reference = output == [1.0, 2.0, 4.0, 8.0, 16.0];
        assert!(
            output_matches_reference,
            "second native graph output: {output:?}"
        );
        second_report_tx
            .send(ReplayReport {
                replay_runtime,
                publication_runtime,
                output_matches_reference,
            })
            .map_err(|_| ScheduleError::StatePoisoned)?;
        second_tx
            .send(over_budget)
            .map_err(|_| ScheduleError::StatePoisoned)?;
        Ok::<_, ScheduleError>(())
    });

    let contention_rejection = second_rx.recv().expect("typed contention rejection");
    release_tx.send(()).expect("release first scheduler lease");
    first
        .join()
        .expect("first submitter thread")
        .expect("first native graph completion");
    let first_report = first_report_rx.recv().expect("first replay evidence");
    let after_first_refund = snapshot(&scheduler);
    assert_eq!(after_first_refund.bytes_in_use, 0);
    assert_eq!(after_first_refund.active, 0);
    assert_eq!(after_first_refund.completed, 1);
    retry_allowed_tx
        .send(())
        .expect("release second submitter retry");
    let over_budget_rejection = second_rx.recv().expect("typed over-budget rejection");
    second
        .join()
        .expect("second submitter thread")
        .expect("second retry graph completion");
    let second_report = second_report_rx.recv().expect("second replay evidence");
    let after_retry = snapshot(&scheduler);
    assert_eq!(after_retry.bytes_in_use, 0);
    assert_eq!(after_retry.active, 0);
    assert_eq!(after_retry.completed, 2);
    assert_eq!(after_retry.failed, 0);
    assert!(!after_retry.lost);

    let replay_runtime = sum_runtime(first_report.replay_runtime, second_report.replay_runtime);
    assert_eq!(replay_runtime.kernel_launches, 2);
    assert_eq!(replay_runtime.completion_events, 2);
    assert_eq!(replay_runtime.completion_waits, 2);
    assert_eq!(replay_runtime.h2d_operations, 0);
    assert_eq!(replay_runtime.d2h_operations, 0);
    assert_eq!(replay_runtime.d2d_operations, 0);

    let publication_runtime = sum_runtime(
        first_report.publication_runtime,
        second_report.publication_runtime,
    );
    let graph_output_matches_reference =
        first_report.output_matches_reference && second_report.output_matches_reference;
    assert!(graph_output_matches_reference);
    assert_eq!(publication_runtime.d2h_operations, 2);
    assert!(publication_runtime.d2h_bytes >= 2 * ELEMENTS * std::mem::size_of::<f32>());

    let high_water_bytes = during_contention
        .bytes_in_use
        .max(after_first_refund.bytes_in_use);
    let high_water_bytes = high_water_bytes.max(after_retry.bytes_in_use);
    let no_stranded_leases = after_retry.bytes_in_use == 0
        && after_retry.active == 0
        && after_retry.cancelled == 0
        && after_retry.failed == 0
        && after_retry.device_lost == 0;
    assert!(no_stranded_leases);
    let evidence_sha256 = evidence_digest(
        contention_rejection,
        over_budget_rejection,
        during_contention,
        after_first_refund,
        after_retry,
        replay_runtime,
        publication_runtime,
    );
    write_artifact(&SchedulerContentionArtifact {
        schema: SCHEMA,
        artifact_kind: ARTIFACT_KIND,
        fixture_version: FIXTURE_VERSION,
        deterministic_fixture: FIXTURE,
        status: "passed",
        target_os: "linux",
        target_arch: "x86_64",
        requested_backend: "cuda",
        actual_backend: "cuda",
        cuda_executed: true,
        device_ordinal: DEVICE_ORDINAL,
        device_model,
        compute_capability,
        compiler: compiler_version(),
        host_submitter_count: 2,
        thread_policy: "two-host-submitters;single-cuda0-budget;barrier-held-lease",
        scheduler_device: "cuda0",
        budget_bytes: BUDGET_BYTES,
        budget_active_slots: ACTIVE_SLOTS,
        request_bytes: REQUEST_BYTES,
        request_active_slots: ACTIVE_SLOTS,
        over_budget_bytes: OVER_BUDGET_BYTES,
        workload: "capture-safe native f64-to-f32 cast graph replay",
        workload_elements: ELEMENTS,
        graph_replays: 2,
        graph_output_matches_reference,
        contention_rejection,
        over_budget_rejection,
        during_contention,
        after_first_refund,
        after_retry,
        replay_runtime,
        publication_runtime,
        high_water_bytes,
        no_stranded_leases,
        evidence_sha256,
        unsupported_scope: "no multi-GPU placement, physical reset, Windows/Arm64, Nsight roofline, or broader application-flow claim",
    });
}
