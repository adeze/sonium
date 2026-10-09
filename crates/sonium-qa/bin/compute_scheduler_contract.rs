//! CPU-only contract for scheduler admission, terminal accounting, and loss recovery.
//!
//! This intentionally exercises only the scheduler.  It does not construct or
//! run a simulation workload, so it is suitable for the fast QA tier when
//! simulator processes are unavailable or deliberately stopped.

use sonium_compute::{DeviceBudget, DeviceScheduler, ScheduleError, WorkRequest};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.scheduler-contract.v1";

#[derive(Debug)]
struct Metrics {
    quota_rejected: bool,
    cancellation_refunded: bool,
    failure_classified: bool,
    success_classified: bool,
    device_loss_blocked: bool,
    restore_reaccepted: bool,
    final_bytes: usize,
    final_active: usize,
}

fn run() -> Result<Metrics, String> {
    let scheduler = DeviceScheduler::new([
        DeviceBudget::new("cpu0", 128, 2),
        DeviceBudget::new("cpu1", 64, 1),
    ])
    .map_err(|error| format!("scheduler construction failed: {error}"))?;

    // A reservation that cannot fit must not be partially committed.
    let held = scheduler
        .acquire_on(0, WorkRequest::new(120, 1))
        .map_err(|error| format!("initial admission failed: {error}"))?;
    let quota_rejected = matches!(
        scheduler.acquire_on(0, WorkRequest::new(16, 1)),
        Err(ScheduleError::QuotaExceeded { .. })
    );
    if !quota_rejected {
        return Err("over-budget admission was not rejected".into());
    }

    held.cancel()
        .map_err(|error| format!("cancellation refund failed: {error}"))?;
    let after_cancel = scheduler
        .snapshots()
        .map_err(|error| format!("snapshot after cancellation failed: {error}"))?;
    let cancellation_refunded = after_cancel[0].bytes_in_use == 0
        && after_cancel[0].active == 0
        && after_cancel[0].cancelled == 1;

    // Synchronous success and backend failure each release their reservation
    // and increment only their corresponding terminal counter.
    scheduler
        .execute(WorkRequest::new(8, 1), |_device| Ok::<_, &'static str>(()))
        .map_err(|error| format!("successful execution failed: {error}"))?;
    let failure = scheduler.execute(WorkRequest::new(8, 1), |_device| {
        Err::<(), _>("injected backend failure")
    });
    let failure_classified = matches!(failure, Err(ScheduleError::BackendFailure { .. }));
    if !failure_classified {
        return Err(format!("backend failure lost its type: {failure:?}"));
    }

    let after_terminal = scheduler
        .snapshots()
        .map_err(|error| format!("terminal snapshot failed: {error}"))?;
    let success_classified = after_terminal[0].completed == 1
        && after_terminal[0].failed == 1
        && after_terminal[0].bytes_in_use == 0
        && after_terminal[0].active == 0;

    // Device loss blocks new work, while an idle lost device can be restored
    // and admitted again without stale quota state.
    scheduler
        .mark_device_lost(0)
        .map_err(|error| format!("device-loss transition failed: {error}"))?;
    let device_loss_blocked = matches!(
        scheduler.acquire_on(0, WorkRequest::new(1, 1)),
        Err(ScheduleError::DeviceUnavailable { .. })
    );
    scheduler
        .restore_device(0)
        .map_err(|error| format!("device restore failed: {error}"))?;
    let restored = scheduler
        .acquire_on(0, WorkRequest::new(1, 1))
        .map_err(|error| format!("restored admission failed: {error}"))?;
    restored
        .complete()
        .map_err(|error| format!("restored completion failed: {error}"))?;
    let final_snapshot = scheduler
        .snapshots()
        .map_err(|error| format!("final snapshot failed: {error}"))?;

    Ok(Metrics {
        quota_rejected,
        cancellation_refunded,
        failure_classified,
        success_classified,
        device_loss_blocked,
        restore_reaccepted: final_snapshot[0].completed == 2 && !final_snapshot[0].lost,
        final_bytes: final_snapshot[0].bytes_in_use,
        final_active: final_snapshot[0].active,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("quota_rejected", metrics.quota_rejected as u8 as f64)
                .metric(
                    "cancellation_refunded",
                    metrics.cancellation_refunded as u8 as f64,
                )
                .metric(
                    "failure_classified",
                    metrics.failure_classified as u8 as f64,
                )
                .metric(
                    "success_classified",
                    metrics.success_classified as u8 as f64,
                )
                .metric(
                    "device_loss_blocked",
                    metrics.device_loss_blocked as u8 as f64,
                )
                .metric(
                    "restore_reaccepted",
                    metrics.restore_reaccepted as u8 as f64,
                )
                .metric("final_bytes", metrics.final_bytes as f64)
                .metric("final_active", metrics.final_active as f64)
                .minimum("quota_rejected", 1.0)
                .minimum("cancellation_refunded", 1.0)
                .minimum("failure_classified", 1.0)
                .minimum("success_classified", 1.0)
                .minimum("device_loss_blocked", 1.0)
                .minimum("restore_reaccepted", 1.0)
                .maximum("final_bytes", 0.0)
                .maximum("final_active", 0.0)
                .reference("independent scheduler terminal-state invariants")
                .reference_checksum("analytic/deterministic-fixture:compute-scheduler-contract-v1")
                .dimension("two CPU scheduler devices; byte and active-slot quotas")
                .seed("compute-scheduler-contract:cpu0-cpu1:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("scheduler QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("scheduler QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
