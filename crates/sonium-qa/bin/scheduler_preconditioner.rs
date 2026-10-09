//! Bounded W2-01 scheduler/preconditioner integration evidence.
//!
//! This case proves the checked hand-off between compute admission and a
//! saddle-point preconditioner on deterministic CPU fixtures. It is a bounded
//! contract case, not a claim that every device backend or saddle strategy is
//! release-complete.

use sonium_compute::{CpuBackend, DeviceBudget, DeviceScheduler, DeviceSnapshot, WorkRequest};
use sonium_qa::QaResult;
use sonium_solvers::preconditioners::{
    BlockPreconditionerConfig, SaddlePointConfig, SaddlePointPreconditioner, SchurStrategy,
};
use sonium_solvers::{CsrMatrix, Preconditioner};
use std::time::Instant;

const CASE_ID: &str = "coupling.scheduler-preconditioner-contracts.v1";

#[derive(Debug)]
struct Metrics {
    scheduler_admission_checks: f64,
    backend_completion_checks: f64,
    scheduler_typed_rejections: f64,
    preconditioner_identity_error: f64,
    preconditioner_typed_rejections: f64,
    elapsed_ms: u64,
}

fn snapshot(scheduler: &DeviceScheduler) -> Result<DeviceSnapshot, String> {
    scheduler
        .snapshots()
        .map_err(|error| format!("scheduler snapshot failed: {error}"))?
        .into_iter()
        .next()
        .ok_or_else(|| "scheduler returned no device snapshot".into())
}

fn rejects_without_panic_output(operation: impl FnOnce() -> bool) -> bool {
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let rejected = operation();
    std::panic::set_hook(previous_hook);
    rejected
}

fn run() -> Result<Metrics, String> {
    let started = Instant::now();
    let scheduler = DeviceScheduler::new([DeviceBudget::new("cpu", 128, 1)])
        .map_err(|error| format!("scheduler setup failed: {error}"))?;

    let lease = scheduler
        .acquire(WorkRequest::new(64, 1))
        .map_err(|error| format!("scheduler admission failed: {error}"))?;
    let admitted = snapshot(&scheduler)?;
    if admitted.bytes_in_use != 64 || admitted.active != 1 {
        return Err(format!("unexpected admitted snapshot: {admitted:?}"));
    }
    lease
        .complete()
        .map_err(|error| format!("scheduler completion failed: {error}"))?;
    let released = snapshot(&scheduler)?;
    if released.bytes_in_use != 0 || released.active != 0 || released.completed != 1 {
        return Err(format!("unexpected released snapshot: {released:?}"));
    }

    let backend = CpuBackend::new();
    let value = scheduler
        .execute_backend(WorkRequest::new(16, 1), &backend, |_device, _backend| {
            Ok::<_, sonium_compute::error::BackendError>(9usize)
        })
        .map_err(|error| format!("backend completion bridge failed: {error}"))?;
    if value != 9 {
        return Err(format!("backend completion returned {value}, expected 9"));
    }
    let after_backend = snapshot(&scheduler)?;
    if after_backend.completed != 2 || after_backend.bytes_in_use != 0 {
        return Err(format!(
            "backend completion was not accounted: {after_backend:?}"
        ));
    }

    let mut scheduler_typed_rejections = 0.0;
    if scheduler.acquire(WorkRequest::new(1, 0)).is_err() {
        scheduler_typed_rejections += 1.0;
    }
    if scheduler.acquire_on(4, WorkRequest::new(1, 1)).is_err() {
        scheduler_typed_rejections += 1.0;
    }
    scheduler
        .mark_device_lost(0)
        .map_err(|error| format!("device-loss transition failed: {error}"))?;
    if scheduler.acquire(WorkRequest::new(1, 1)).is_err() {
        scheduler_typed_rejections += 1.0;
    }
    if scheduler_typed_rejections != 3.0 {
        return Err(format!(
            "expected three typed scheduler rejections, got {scheduler_typed_rejections}"
        ));
    }

    let matrix = CsrMatrix::identity(4);
    let config = SaddlePointConfig {
        block1_size: 2,
        block1_config: BlockPreconditionerConfig::Diagonal,
        block2_config: BlockPreconditionerConfig::Diagonal,
        schur_strategy: SchurStrategy::Diagonal,
        upper_triangular: false,
    };
    let preconditioner = SaddlePointPreconditioner::try_from_matrix(&matrix, config)
        .map_err(|error| format!("checked saddle-point setup failed: {error}"))?;
    let rhs = [1.0_f64, -2.0, 3.0, -4.0];
    let result = preconditioner
        .try_apply(&rhs)
        .map_err(|error| format!("checked saddle-point application failed: {error}"))?;
    let preconditioner_identity_error = result
        .iter()
        .zip(rhs)
        .map(|(got, expected)| (got - expected).abs())
        .fold(0.0, f64::max);
    if preconditioner_identity_error > 1.0e-12 {
        return Err(format!(
            "saddle-point identity application error is {preconditioner_identity_error}"
        ));
    }

    let mut preconditioner_typed_rejections = 0.0;
    if rejects_without_panic_output(|| preconditioner.try_apply(&[1.0, 2.0]).is_err()) {
        preconditioner_typed_rejections += 1.0;
    }
    if preconditioner
        .try_apply(&[1.0, f64::NAN, 3.0, 4.0])
        .is_err()
    {
        preconditioner_typed_rejections += 1.0;
    }
    let non_square = CsrMatrix::try_from_raw_parts(2, 3, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0])
        .map_err(|error| format!("non-square fixture construction failed: {error}"))?;
    let invalid_config = SaddlePointConfig {
        block1_size: 1,
        block1_config: BlockPreconditionerConfig::Diagonal,
        block2_config: BlockPreconditionerConfig::Diagonal,
        schur_strategy: SchurStrategy::Diagonal,
        upper_triangular: false,
    };
    if SaddlePointPreconditioner::try_from_matrix(&non_square, invalid_config).is_err() {
        preconditioner_typed_rejections += 1.0;
    }
    if preconditioner_typed_rejections != 3.0 {
        return Err(format!(
            "expected three typed preconditioner rejections, got {preconditioner_typed_rejections}"
        ));
    }

    Ok(Metrics {
        scheduler_admission_checks: 2.0,
        backend_completion_checks: 1.0,
        scheduler_typed_rejections,
        preconditioner_identity_error,
        preconditioner_typed_rejections,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("scheduler_admission_checks", metrics.scheduler_admission_checks)
                .metric("backend_completion_checks", metrics.backend_completion_checks)
                .metric("scheduler_typed_rejections", metrics.scheduler_typed_rejections)
                .metric(
                    "preconditioner_identity_error",
                    metrics.preconditioner_identity_error,
                )
                .metric(
                    "preconditioner_typed_rejections",
                    metrics.preconditioner_typed_rejections,
                )
                .minimum("scheduler_admission_checks", 2.0)
                .minimum("backend_completion_checks", 1.0)
                .minimum("scheduler_typed_rejections", 3.0)
                .maximum("preconditioner_identity_error", 1.0e-12)
                .minimum("preconditioner_typed_rejections", 3.0)
                .input_checksum("analytic/deterministic-fixture:scheduler-preconditioner-v1")
                .reference("scheduler quota/refund counters, CPU completion bridge, saddle-point identity application, typed rejection paths")
                .reference_checksum("analytic/deterministic-fixture:scheduler-preconditioner-v1")
                .dimension("one CPU device with 128-byte/one-slot budget; 4 DOF 2x2 saddle-point identity")
                .formulation("checked DeviceScheduler admission/completion and diagonal saddle-point preconditioner")
                .tolerance("exact counters and rejection counts; identity error <= 1e-12")
                .seed("coupling-scheduler-preconditioner-v1")
                .timing_ms("scheduler_preconditioner_contracts", metrics.elapsed_ms)
                .elapsed_ms(metrics.elapsed_ms);
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("scheduler/preconditioner result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .elapsed_ms(started.elapsed().as_millis() as u64)
                    .to_json_line()
                    .expect("scheduler/preconditioner failure result must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
