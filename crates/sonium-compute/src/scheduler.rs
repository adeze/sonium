//! Deterministic multi-device admission control.
//!
//! The scheduler owns reservations, not backend command buffers. Callers
//! acquire a [`WorkLease`] before submitting a candidate, frequency, or
//! solver chunk and keep the lease until the command completes. Dropping a
//! lease (including on cancellation or partial command failure) refunds both
//! the byte and active-work reservations. Device loss prevents new work from
//! being admitted while existing leases remain accounted for until released.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};
use thiserror::Error;

use crate::backend::{Backend, BackendCompletion};
use crate::error::{BackendError, BackendResult};

/// Per-device memory and concurrency limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceBudget {
    pub name: String,
    pub max_bytes: usize,
    pub max_active: usize,
}

impl DeviceBudget {
    pub fn new(name: impl Into<String>, max_bytes: usize, max_active: usize) -> Self {
        Self {
            name: name.into(),
            max_bytes,
            max_active,
        }
    }
}

/// Resources requested by one schedulable unit of work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkRequest {
    pub bytes: usize,
    pub active_slots: usize,
}

impl WorkRequest {
    pub const fn new(bytes: usize, active_slots: usize) -> Self {
        Self {
            bytes,
            active_slots,
        }
    }
}

/// Errors raised by admission or device lifecycle operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ScheduleError {
    #[error("at least one device budget is required")]
    NoDevices,
    #[error("invalid budget for device {device:?}: {reason}")]
    InvalidBudget {
        device: String,
        reason: &'static str,
    },
    #[error("invalid work request: {reason}")]
    InvalidRequest { reason: &'static str },
    #[error("all devices are unavailable")]
    AllDevicesUnavailable,
    #[error("no device can admit {requested_bytes} bytes and {requested_slots} active slots")]
    QuotaExceeded {
        requested_bytes: usize,
        requested_slots: usize,
    },
    #[error("device index {index} is out of range")]
    InvalidDevice { index: usize },
    #[error("device {device:?} is unavailable")]
    DeviceUnavailable { device: String },
    #[error("device {device:?} still has active work")]
    DeviceBusy { device: String },
    #[error("backend work failed on device {device:?}: {reason}")]
    BackendFailure { device: String, reason: String },
    #[error("device {device:?} was lost during backend work: {reason}")]
    DeviceLost { device: String, reason: String },
    #[error("asynchronous submission on device {device:?} was already finalized")]
    SubmissionAlreadyFinalized { device: String },
    #[error("scheduler state mutex is poisoned")]
    StatePoisoned,
}

/// Terminal state recorded when an admitted work item releases its lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkCompletion {
    /// The backend operation completed successfully.
    Completed,
    /// The operation was cancelled before successful completion.
    Cancelled,
    /// The backend operation failed after admission.
    Failed,
    /// The backend device was lost before completion.
    DeviceLost,
}

#[derive(Debug, Default)]
struct DeviceState {
    in_use: usize,
    active: usize,
    lost: bool,
    completed: usize,
    cancelled: usize,
    failed: usize,
    device_lost: usize,
}

#[derive(Debug)]
struct DeviceEntry {
    budget: DeviceBudget,
    state: Mutex<DeviceState>,
}

#[derive(Debug)]
struct SchedulerInner {
    devices: Vec<DeviceEntry>,
}

/// A quota-aware scheduler for independent devices.
#[derive(Debug, Clone)]
pub struct DeviceScheduler {
    inner: Arc<SchedulerInner>,
}

/// Snapshot used by admission diagnostics and benchmark artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceSnapshot {
    pub name: String,
    pub max_bytes: usize,
    pub max_active: usize,
    pub bytes_in_use: usize,
    pub active: usize,
    pub lost: bool,
    pub completed: usize,
    pub cancelled: usize,
    pub failed: usize,
    pub device_lost: usize,
}

/// A reservation that refunds resources when dropped.
#[derive(Debug)]
pub struct WorkLease {
    scheduler: Arc<SchedulerInner>,
    device_index: usize,
    bytes: usize,
    active_slots: usize,
    released: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubmissionPhase {
    Pending,
    Finalizing,
    Completed,
    Cancelled,
    Failed,
    DeviceLost,
}

struct BackendSubmissionState<T> {
    phase: SubmissionPhase,
    lease: Option<WorkLease>,
    completion: Option<Box<dyn BackendCompletion>>,
    value: Option<T>,
}

struct BackendSubmissionInner<T> {
    state: Mutex<BackendSubmissionState<T>>,
    device: String,
}

/// A backend operation admitted by the scheduler but not yet completed.
///
/// Cloning a submission creates another race-safe completion handle. Exactly
/// one handle may perform the terminal transition; dropping the last handle
/// waits for the backend token and cancels the work before releasing the lease.
#[derive(Clone)]
pub struct BackendSubmission<T> {
    inner: Arc<BackendSubmissionInner<T>>,
}

impl DeviceScheduler {
    pub fn new<I>(budgets: I) -> Result<Self, ScheduleError>
    where
        I: IntoIterator<Item = DeviceBudget>,
    {
        let budgets: Vec<_> = budgets.into_iter().collect();
        if budgets.is_empty() {
            return Err(ScheduleError::NoDevices);
        }
        for budget in &budgets {
            if budget.max_bytes == 0 {
                return Err(ScheduleError::InvalidBudget {
                    device: budget.name.clone(),
                    reason: "max_bytes must be greater than zero",
                });
            }
            if budget.max_active == 0 {
                return Err(ScheduleError::InvalidBudget {
                    device: budget.name.clone(),
                    reason: "max_active must be greater than zero",
                });
            }
        }
        Ok(Self {
            inner: Arc::new(SchedulerInner {
                devices: budgets
                    .into_iter()
                    .map(|budget| DeviceEntry {
                        budget,
                        state: Mutex::new(DeviceState::default()),
                    })
                    .collect(),
            }),
        })
    }

    pub fn device_count(&self) -> usize {
        self.inner.devices.len()
    }

    /// Admit work on the least-loaded currently healthy device.
    pub fn acquire(&self, request: WorkRequest) -> Result<WorkLease, ScheduleError> {
        self.validate_request(request)?;
        let mut candidates = Vec::new();
        let mut healthy = false;
        for (index, entry) in self.inner.devices.iter().enumerate() {
            let state = entry
                .state
                .lock()
                .map_err(|_| ScheduleError::StatePoisoned)?;
            if state.lost {
                continue;
            }
            healthy = true;
            candidates.push((state.active, state.in_use, index));
        }
        if !healthy {
            return Err(ScheduleError::AllDevicesUnavailable);
        }
        candidates.sort_unstable();
        self.try_acquire_candidates(&candidates, request)
    }

    fn try_acquire_candidates(
        &self,
        candidates: &[(usize, usize, usize)],
        request: WorkRequest,
    ) -> Result<WorkLease, ScheduleError> {
        let mut saw_quota_exhaustion = false;
        let mut saw_device_loss = false;
        for &(_, _, index) in candidates {
            match self.try_acquire_on(index, request) {
                Ok(lease) => return Ok(lease),
                // A device can be marked lost after the initial candidate
                // snapshot but before its admission lock is reacquired. Do
                // not misreport that race as a quota failure.
                Err(ScheduleError::DeviceUnavailable { .. }) => saw_device_loss = true,
                Err(ScheduleError::QuotaExceeded { .. }) => saw_quota_exhaustion = true,
                Err(error) => return Err(error),
            }
        }
        if saw_quota_exhaustion {
            Err(ScheduleError::QuotaExceeded {
                requested_bytes: request.bytes,
                requested_slots: request.active_slots,
            })
        } else if saw_device_loss {
            Err(ScheduleError::AllDevicesUnavailable)
        } else {
            Err(ScheduleError::QuotaExceeded {
                requested_bytes: request.bytes,
                requested_slots: request.active_slots,
            })
        }
    }

    /// Admit work on one explicitly selected device.
    pub fn acquire_on(
        &self,
        device_index: usize,
        request: WorkRequest,
    ) -> Result<WorkLease, ScheduleError> {
        self.validate_request(request)?;
        self.try_acquire_on(device_index, request)
    }

    /// Execute one synchronously submitted work item under a lease.
    ///
    /// The lease is held until the callback returns. Callers with genuinely
    /// asynchronous work can use [`Self::acquire`] and finish the returned
    /// [`WorkLease`] explicitly when the backend signals completion.
    pub fn execute<T, E, F>(&self, request: WorkRequest, operation: F) -> Result<T, ScheduleError>
    where
        E: std::fmt::Display,
        F: FnOnce(usize) -> Result<T, E>,
    {
        let lease = self.acquire(request)?;
        let device_index = lease.device_index();
        let device_name = self
            .inner
            .devices
            .get(device_index)
            .map(|entry| entry.budget.name.clone())
            .ok_or(ScheduleError::InvalidDevice {
                index: device_index,
            })?;
        match catch_unwind(AssertUnwindSafe(|| operation(device_index))) {
            Ok(value) => match value {
                Ok(value) => {
                    lease.complete()?;
                    Ok(value)
                }
                Err(error) => {
                    lease.finish(WorkCompletion::Failed)?;
                    Err(ScheduleError::BackendFailure {
                        device: device_name,
                        reason: error.to_string(),
                    })
                }
            },
            Err(payload) => {
                lease.finish(WorkCompletion::Failed)?;
                Err(ScheduleError::BackendFailure {
                    device: device_name,
                    reason: panic_reason(payload),
                })
            }
        }
    }

    /// Submit backend work and return before a native device completion.
    ///
    /// The callback must enqueue work on `backend` but must not release or
    /// reuse buffers referenced by that work. CUDA records a stream event and
    /// Metal transfers ownership of its pending command buffer into the
    /// returned token. The [`BackendSubmission`] keeps the scheduler lease
    /// live until `complete`, `cancel`, or the last-handle drop finishes that
    /// token.
    pub fn submit_backend<B, T, F>(
        &self,
        request: WorkRequest,
        backend: &B,
        operation: F,
    ) -> Result<BackendSubmission<T>, ScheduleError>
    where
        B: Backend,
        F: FnOnce(usize, &B) -> BackendResult<T>,
    {
        let lease = self.acquire(request)?;
        let device_index = lease.device_index();
        let device_name = self.device_name(device_index)?;
        let value = match catch_unwind(AssertUnwindSafe(|| operation(device_index, backend))) {
            Ok(Ok(value)) => value,
            Ok(Err(error)) => {
                lease.finish(WorkCompletion::Failed)?;
                return Err(backend_error_to_schedule(&device_name, error));
            }
            Err(payload) => {
                lease.finish(WorkCompletion::Failed)?;
                return Err(ScheduleError::BackendFailure {
                    device: device_name,
                    reason: panic_reason(payload),
                });
            }
        };
        let completion = match catch_unwind(AssertUnwindSafe(|| backend.try_submit_completion())) {
            Ok(Ok(completion)) => completion,
            Ok(Err(error)) => {
                let outcome = if matches!(&error, BackendError::DeviceLost { .. }) {
                    WorkCompletion::DeviceLost
                } else {
                    WorkCompletion::Failed
                };
                lease.finish(outcome)?;
                return Err(backend_error_to_schedule(&device_name, error));
            }
            Err(payload) => {
                lease.finish(WorkCompletion::Failed)?;
                return Err(ScheduleError::BackendFailure {
                    device: device_name,
                    reason: panic_reason(payload),
                });
            }
        };
        Ok(BackendSubmission::new(
            lease,
            completion,
            value,
            device_name,
        ))
    }

    /// Execute backend work while keeping the lease until the backend's
    /// checked completion boundary succeeds. This is the synchronous bridge
    /// used by callers that do not own an asynchronous command-buffer handle.
    pub fn execute_backend<B, T, F>(
        &self,
        request: WorkRequest,
        backend: &B,
        operation: F,
    ) -> Result<T, ScheduleError>
    where
        B: Backend,
        F: FnOnce(usize, &B) -> BackendResult<T>,
    {
        self.submit_backend(request, backend, operation)?.complete()
    }

    fn device_name(&self, device_index: usize) -> Result<String, ScheduleError> {
        self.inner
            .devices
            .get(device_index)
            .map(|entry| entry.budget.name.clone())
            .ok_or(ScheduleError::InvalidDevice {
                index: device_index,
            })
    }

    fn try_acquire_on(
        &self,
        device_index: usize,
        request: WorkRequest,
    ) -> Result<WorkLease, ScheduleError> {
        let entry = self
            .inner
            .devices
            .get(device_index)
            .ok_or(ScheduleError::InvalidDevice {
                index: device_index,
            })?;
        let mut state = entry
            .state
            .lock()
            .map_err(|_| ScheduleError::StatePoisoned)?;
        if state.lost {
            return Err(ScheduleError::DeviceUnavailable {
                device: entry.budget.name.clone(),
            });
        }
        if request.bytes > entry.budget.max_bytes.saturating_sub(state.in_use)
            || request.active_slots > entry.budget.max_active.saturating_sub(state.active)
        {
            return Err(ScheduleError::QuotaExceeded {
                requested_bytes: request.bytes,
                requested_slots: request.active_slots,
            });
        }
        state.in_use += request.bytes;
        state.active += request.active_slots;
        Ok(WorkLease {
            scheduler: Arc::clone(&self.inner),
            device_index,
            bytes: request.bytes,
            active_slots: request.active_slots,
            released: false,
        })
    }

    fn validate_request(&self, request: WorkRequest) -> Result<(), ScheduleError> {
        if request.active_slots == 0 {
            return Err(ScheduleError::InvalidRequest {
                reason: "active_slots must be greater than zero",
            });
        }
        Ok(())
    }

    pub fn mark_device_lost(&self, device_index: usize) -> Result<(), ScheduleError> {
        let entry = self
            .inner
            .devices
            .get(device_index)
            .ok_or(ScheduleError::InvalidDevice {
                index: device_index,
            })?;
        let mut state = entry
            .state
            .lock()
            .map_err(|_| ScheduleError::StatePoisoned)?;
        state.lost = true;
        Ok(())
    }

    pub fn restore_device(&self, device_index: usize) -> Result<(), ScheduleError> {
        let entry = self
            .inner
            .devices
            .get(device_index)
            .ok_or(ScheduleError::InvalidDevice {
                index: device_index,
            })?;
        let mut state = entry
            .state
            .lock()
            .map_err(|_| ScheduleError::StatePoisoned)?;
        if state.active != 0 || state.in_use != 0 {
            return Err(ScheduleError::DeviceBusy {
                device: entry.budget.name.clone(),
            });
        }
        state.lost = false;
        Ok(())
    }

    pub fn snapshots(&self) -> Result<Vec<DeviceSnapshot>, ScheduleError> {
        self.inner
            .devices
            .iter()
            .map(|entry| {
                let state = entry
                    .state
                    .lock()
                    .map_err(|_| ScheduleError::StatePoisoned)?;
                Ok(DeviceSnapshot {
                    name: entry.budget.name.clone(),
                    max_bytes: entry.budget.max_bytes,
                    max_active: entry.budget.max_active,
                    bytes_in_use: state.in_use,
                    active: state.active,
                    lost: state.lost,
                    completed: state.completed,
                    cancelled: state.cancelled,
                    failed: state.failed,
                    device_lost: state.device_lost,
                })
            })
            .collect()
    }
}

fn backend_error_to_schedule(device: &str, error: BackendError) -> ScheduleError {
    match error {
        BackendError::DeviceLost { reason, .. } => ScheduleError::DeviceLost {
            device: device.to_owned(),
            reason,
        },
        error => ScheduleError::BackendFailure {
            device: device.to_owned(),
            reason: error.to_string(),
        },
    }
}

fn panic_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "scheduler operation panicked".to_owned()
    }
}

impl<T> BackendSubmission<T> {
    fn new(
        lease: WorkLease,
        completion: Box<dyn BackendCompletion>,
        value: T,
        device: String,
    ) -> Self {
        Self {
            inner: Arc::new(BackendSubmissionInner {
                state: Mutex::new(BackendSubmissionState {
                    phase: SubmissionPhase::Pending,
                    lease: Some(lease),
                    completion: Some(completion),
                    value: Some(value),
                }),
                device,
            }),
        }
    }

    /// Wait for native completion and return the callback's value.
    pub fn complete(&self) -> Result<T, ScheduleError> {
        self.finalize(false)?.ok_or(ScheduleError::StatePoisoned)
    }

    /// Cancel the logical operation. Since already-enqueued device work
    /// cannot be safely abandoned, this waits for the native token first and
    /// only then refunds the lease as cancelled.
    pub fn cancel(&self) -> Result<(), ScheduleError> {
        self.finalize(true).map(|_| ())
    }

    /// Wait for completion on a dedicated thread. The scheduler lease remains
    /// live in the returned join handle until that thread observes the native
    /// completion boundary.
    pub fn spawn(self) -> std::thread::JoinHandle<Result<T, ScheduleError>>
    where
        T: Send + 'static,
    {
        std::thread::spawn(move || self.complete())
    }

    fn finalize(&self, cancelled: bool) -> Result<Option<T>, ScheduleError> {
        let (mut completion, lease) = {
            let mut state = self
                .inner
                .state
                .lock()
                .map_err(|_| ScheduleError::StatePoisoned)?;
            if state.phase != SubmissionPhase::Pending {
                return Err(ScheduleError::SubmissionAlreadyFinalized {
                    device: self.inner.device.clone(),
                });
            }
            state.phase = SubmissionPhase::Finalizing;
            let completion = state
                .completion
                .take()
                .ok_or(ScheduleError::StatePoisoned)?;
            let lease = state.lease.take().ok_or(ScheduleError::StatePoisoned)?;
            (completion, lease)
        };

        let backend_result = match catch_unwind(AssertUnwindSafe(|| {
            if cancelled {
                completion.cancel()
            } else {
                completion.wait()
            }
        })) {
            Ok(result) => result,
            Err(payload) => Err(BackendError::OperationFailed {
                operation: "completion",
                reason: panic_reason(payload),
            }),
        };
        match backend_result {
            Ok(()) => {
                let phase = if cancelled {
                    SubmissionPhase::Cancelled
                } else {
                    SubmissionPhase::Completed
                };
                let release_result = lease.finish(if cancelled {
                    WorkCompletion::Cancelled
                } else {
                    WorkCompletion::Completed
                });
                let value = {
                    let mut state = self
                        .inner
                        .state
                        .lock()
                        .map_err(|_| ScheduleError::StatePoisoned)?;
                    state.phase = phase;
                    if cancelled {
                        state.value.take();
                        None
                    } else {
                        state.value.take()
                    }
                };
                match release_result {
                    Ok(()) => Ok(value),
                    Err(error) => Err(error),
                }
            }
            Err(error) => {
                let device_lost = matches!(&error, BackendError::DeviceLost { .. });
                let phase = if device_lost {
                    SubmissionPhase::DeviceLost
                } else {
                    SubmissionPhase::Failed
                };
                let outcome = if device_lost {
                    WorkCompletion::DeviceLost
                } else {
                    WorkCompletion::Failed
                };
                let release_result = lease.finish(outcome);
                let result = match release_result {
                    Ok(()) => Err(backend_error_to_schedule(&self.inner.device, error)),
                    Err(error) => Err(error),
                };
                if let Ok(mut state) = self.inner.state.lock() {
                    state.phase = phase;
                    state.value.take();
                }
                result
            }
        }
    }
}

impl<T> Drop for BackendSubmissionInner<T> {
    fn drop(&mut self) {
        let state = match self.state.get_mut() {
            Ok(state) => state,
            Err(error) => error.into_inner(),
        };
        if state.phase != SubmissionPhase::Pending {
            return;
        }
        state.phase = SubmissionPhase::Finalizing;
        let (Some(mut completion), Some(lease)) = (state.completion.take(), state.lease.take())
        else {
            return;
        };
        let result = catch_unwind(AssertUnwindSafe(|| completion.cancel()));
        let outcome = match result {
            Ok(Ok(())) => WorkCompletion::Cancelled,
            Ok(Err(BackendError::DeviceLost { .. })) => WorkCompletion::DeviceLost,
            Ok(Err(_)) | Err(_) => WorkCompletion::Failed,
        };
        let _ = lease.finish(outcome);
    }
}

impl WorkLease {
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn active_slots(&self) -> usize {
        self.active_slots
    }

    /// Explicitly release the reservation, returning lifecycle errors.
    pub fn release(mut self) -> Result<(), ScheduleError> {
        let result = self.release_inner(WorkCompletion::Cancelled);
        // Terminal ownership is consumed even when accounting reports an
        // error.  Retrying from Drop could otherwise double-refund a
        // partially completed accounting operation.
        self.released = true;
        result
    }

    /// Mark the admitted operation as successfully completed and refund it.
    pub fn complete(mut self) -> Result<(), ScheduleError> {
        let result = self.release_inner(WorkCompletion::Completed);
        self.released = true;
        result
    }

    /// Mark the admitted operation as cancelled and refund it.
    pub fn cancel(self) -> Result<(), ScheduleError> {
        self.release()
    }

    /// Mark the admitted operation as failed and refund it.
    pub fn fail(self) -> Result<(), ScheduleError> {
        self.finish(WorkCompletion::Failed)
    }

    /// Finish an admitted operation with an explicit terminal outcome.
    pub fn finish(mut self, completion: WorkCompletion) -> Result<(), ScheduleError> {
        let result = self.release_inner(completion);
        self.released = true;
        result
    }

    fn release_inner(&self, completion: WorkCompletion) -> Result<(), ScheduleError> {
        let entry =
            self.scheduler
                .devices
                .get(self.device_index)
                .ok_or(ScheduleError::InvalidDevice {
                    index: self.device_index,
                })?;
        let (mut state, poisoned) = match entry.state.lock() {
            Ok(state) => (state, false),
            Err(error) => (error.into_inner(), true),
        };
        state.in_use = state.in_use.saturating_sub(self.bytes);
        state.active = state.active.saturating_sub(self.active_slots);
        match completion {
            WorkCompletion::Completed => state.completed = state.completed.saturating_add(1),
            WorkCompletion::Cancelled => state.cancelled = state.cancelled.saturating_add(1),
            WorkCompletion::Failed => state.failed = state.failed.saturating_add(1),
            WorkCompletion::DeviceLost => {
                state.device_lost = state.device_lost.saturating_add(1);
                // A completion can be the first boundary that observes device
                // loss. Do not admit further work until the caller explicitly
                // restores this now-idle device.
                state.lost = true;
            }
        }
        if poisoned {
            Err(ScheduleError::StatePoisoned)
        } else {
            Ok(())
        }
    }
}

impl Drop for WorkLease {
    fn drop(&mut self) {
        if !self.released {
            let _ = self.release_inner(WorkCompletion::Cancelled);
            // Drop is the final ownership transition.  Even a poisoned or
            // otherwise unavailable accounting state must not be retried by
            // another implicit path.
            self.released = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::thread;

    struct GateCompletion {
        started: Arc<AtomicBool>,
        release: Arc<AtomicBool>,
    }

    impl BackendCompletion for GateCompletion {
        fn wait(&mut self) -> BackendResult<()> {
            self.started.store(true, Ordering::Release);
            while !self.release.load(Ordering::Acquire) {
                thread::yield_now();
            }
            Ok(())
        }
    }

    struct ErrorCompletion(Option<BackendError>);

    impl BackendCompletion for ErrorCompletion {
        fn wait(&mut self) -> BackendResult<()> {
            Err(self.0.take().expect("completion polled once"))
        }
    }

    struct InjectedCompletion {
        started: Arc<AtomicBool>,
        release: Arc<AtomicBool>,
        calls: Arc<AtomicUsize>,
        outcome: Option<BackendError>,
    }

    impl BackendCompletion for InjectedCompletion {
        fn wait(&mut self) -> BackendResult<()> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            self.started.store(true, Ordering::Release);
            while !self.release.load(Ordering::Acquire) {
                thread::yield_now();
            }
            match self.outcome.take() {
                Some(error) => Err(error),
                None => Ok(()),
            }
        }
    }

    fn scheduler() -> DeviceScheduler {
        DeviceScheduler::new([
            DeviceBudget::new("gpu0", 100, 1),
            DeviceBudget::new("gpu1", 100, 2),
        ])
        .expect("valid scheduler")
    }

    #[test]
    fn admission_prefers_least_loaded_device_and_refunds_on_drop() {
        let scheduler = scheduler();
        let first = scheduler
            .acquire(WorkRequest::new(60, 1))
            .expect("first work item");
        assert_eq!(first.device_index(), 0);
        let second = scheduler
            .acquire(WorkRequest::new(60, 1))
            .expect("second work item should use the other device");
        assert_eq!(second.device_index(), 1);
        drop(first);
        let snapshots = scheduler.snapshots().expect("snapshots");
        assert_eq!(snapshots[0].bytes_in_use, 0);
        assert_eq!(snapshots[0].active, 0);
        assert_eq!(snapshots[1].bytes_in_use, 60);
        drop(second);
        assert!(
            scheduler
                .snapshots()
                .expect("refunded snapshots")
                .iter()
                .all(|snapshot| snapshot.bytes_in_use == 0 && snapshot.active == 0)
        );
    }

    #[test]
    fn quota_and_active_limit_are_typed_and_partial_failure_refunds() {
        let scheduler = scheduler();
        let lease = scheduler
            .acquire_on(0, WorkRequest::new(90, 1))
            .expect("reservation");
        assert!(matches!(
            scheduler.acquire_on(0, WorkRequest::new(20, 1)),
            Err(ScheduleError::QuotaExceeded { .. })
        ));
        drop(lease); // Simulates cancellation/partial command failure.
        scheduler
            .acquire_on(0, WorkRequest::new(100, 1))
            .expect("refund should make the full quota available")
            .release()
            .expect("explicit release");
    }

    #[test]
    fn device_loss_stops_admission_until_empty_device_is_restored() {
        let scheduler = scheduler();
        let lease = scheduler
            .acquire_on(0, WorkRequest::new(10, 1))
            .expect("reservation");
        scheduler.mark_device_lost(0).expect("mark loss");
        assert!(matches!(
            scheduler.acquire_on(0, WorkRequest::new(1, 1)),
            Err(ScheduleError::DeviceUnavailable { .. })
        ));
        assert!(matches!(
            scheduler.restore_device(0),
            Err(ScheduleError::DeviceBusy { .. })
        ));
        drop(lease);
        scheduler.restore_device(0).expect("restore after refund");
        scheduler
            .acquire_on(0, WorkRequest::new(1, 1))
            .expect("restored device accepts work")
            .release()
            .expect("release restored work");
    }

    #[test]
    fn stale_admission_candidates_report_device_loss_not_quota_exhaustion() {
        let scheduler = scheduler();
        scheduler.mark_device_lost(0).expect("mark loss");

        // This is the candidate list captured by `acquire` just before a
        // concurrent device-loss transition. Calling the extracted helper makes
        // that race deterministic without timing-sensitive sleeps.
        let stale_candidates = [(0, 0, 0)];
        let result = scheduler.try_acquire_candidates(&stale_candidates, WorkRequest::new(1, 1));

        assert!(matches!(result, Err(ScheduleError::AllDevicesUnavailable)));
    }

    #[test]
    fn execute_records_completion_and_backend_failure_after_refunding() {
        let scheduler = scheduler();
        let value = scheduler
            .execute(WorkRequest::new(20, 1), |_device| Ok::<_, &str>(42_u32))
            .expect("successful work");
        assert_eq!(value, 42);

        let error = scheduler
            .execute(WorkRequest::new(20, 1), |_device| {
                Err::<(), _>("injected launch failure")
            })
            .expect_err("failed work must be typed");
        assert!(matches!(error, ScheduleError::BackendFailure { .. }));

        let snapshot = scheduler.snapshots().expect("snapshot")[0].clone();
        assert_eq!(snapshot.bytes_in_use, 0);
        assert_eq!(snapshot.active, 0);
        assert_eq!(snapshot.completed, 1);
        assert_eq!(snapshot.failed, 1);
        assert_eq!(snapshot.cancelled, 0);
    }

    #[test]
    fn execute_classifies_panics_as_failed_and_refunds_once() {
        let scheduler = scheduler();
        let error = scheduler
            .execute(WorkRequest::new(20, 1), |_device| -> Result<(), &str> {
                panic!("injected operation panic");
            })
            .expect_err("operation panic must become a typed scheduler failure");

        assert!(matches!(
            error,
            ScheduleError::BackendFailure { ref reason, .. }
                if reason.contains("injected operation panic")
        ));
        let snapshot = scheduler.snapshots().expect("snapshot")[0].clone();
        assert_eq!(snapshot.bytes_in_use, 0);
        assert_eq!(snapshot.active, 0);
        assert_eq!(snapshot.completed, 0);
        assert_eq!(snapshot.failed, 1);
        assert_eq!(snapshot.cancelled, 0);
    }

    #[test]
    fn execute_backend_holds_lease_through_checked_completion() {
        let scheduler = scheduler();
        let backend = crate::cpu::CpuBackend::new();

        let value = scheduler
            .execute_backend(WorkRequest::new(20, 1), &backend, |_device, _backend| {
                Ok::<_, crate::error::BackendError>(42_u32)
            })
            .expect("CPU backend completion");

        assert_eq!(value, 42);
        let snapshot = scheduler.snapshots().expect("snapshot")[0].clone();
        assert_eq!(snapshot.bytes_in_use, 0);
        assert_eq!(snapshot.active, 0);
        assert_eq!(snapshot.completed, 1);
        assert_eq!(snapshot.failed, 0);
    }

    #[test]
    fn async_submission_holds_quota_until_native_completion() {
        let scheduler = scheduler();
        let backend = crate::cpu::CpuBackend::new();
        let submission = scheduler
            .submit_backend(WorkRequest::new(20, 1), &backend, |_device, _backend| {
                Ok::<_, BackendError>(42_u32)
            })
            .expect("submission");

        let pending = scheduler.snapshots().expect("pending snapshot")[0].clone();
        assert_eq!(pending.bytes_in_use, 20);
        assert_eq!(pending.active, 1);

        assert_eq!(submission.complete().expect("completion"), 42);
        let finished = scheduler.snapshots().expect("finished snapshot")[0].clone();
        assert_eq!(finished.bytes_in_use, 0);
        assert_eq!(finished.active, 0);
        assert_eq!(finished.completed, 1);
    }

    #[test]
    fn asynchronous_completion_and_cancellation_race_refunds_once() {
        let scheduler = scheduler();
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let lease = scheduler
            .acquire_on(0, WorkRequest::new(20, 1))
            .expect("reservation");
        let submission = BackendSubmission::new(
            lease,
            Box::new(GateCompletion {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
            }),
            42_u32,
            "gpu0".into(),
        );
        let cancellation = submission.clone();
        let completion_thread = thread::spawn(move || submission.complete());
        while !started.load(Ordering::Acquire) {
            thread::yield_now();
        }
        let cancellation_thread = thread::spawn(move || cancellation.cancel());
        release.store(true, Ordering::Release);

        assert_eq!(completion_thread.join().expect("completion thread"), Ok(42));
        assert!(matches!(
            cancellation_thread.join().expect("cancellation thread"),
            Err(ScheduleError::SubmissionAlreadyFinalized { .. })
        ));
        let snapshot = scheduler.snapshots().expect("snapshot")[0].clone();
        assert_eq!(snapshot.bytes_in_use, 0);
        assert_eq!(snapshot.active, 0);
        assert_eq!(snapshot.completed, 1);
        assert_eq!(snapshot.cancelled, 0);
        assert_eq!(snapshot.failed, 0);
    }

    #[test]
    fn dropped_async_submission_waits_before_cancellation_refund() {
        let scheduler = scheduler();
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let lease = scheduler
            .acquire_on(0, WorkRequest::new(20, 1))
            .expect("reservation");
        let submission = BackendSubmission::new(
            lease,
            Box::new(GateCompletion {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
            }),
            (),
            "gpu0".into(),
        );
        let drop_thread = thread::spawn(move || drop(submission));
        while !started.load(Ordering::Acquire) {
            thread::yield_now();
        }
        assert_eq!(
            scheduler.snapshots().expect("pending snapshot")[0].active,
            1
        );
        release.store(true, Ordering::Release);
        drop_thread.join().expect("drop thread");
        let snapshot = scheduler.snapshots().expect("snapshot")[0].clone();
        assert_eq!(snapshot.active, 0);
        assert_eq!(snapshot.cancelled, 1);
    }

    #[test]
    fn device_loss_completion_is_distinct_from_backend_failure() {
        let scheduler = scheduler();
        let lease = scheduler
            .acquire_on(0, WorkRequest::new(20, 1))
            .expect("reservation");
        let submission = BackendSubmission::new(
            lease,
            Box::new(ErrorCompletion(Some(BackendError::DeviceLost {
                backend: "test",
                operation: "completion",
                reason: "device removed".into(),
            }))),
            42_u32,
            "gpu0".into(),
        );

        assert!(matches!(
            submission.complete(),
            Err(ScheduleError::DeviceLost { ref reason, .. }) if reason == "device removed"
        ));
        let snapshot = scheduler.snapshots().expect("snapshot")[0].clone();
        assert_eq!(snapshot.bytes_in_use, 0);
        assert_eq!(snapshot.active, 0);
        assert_eq!(snapshot.device_lost, 1);
        assert_eq!(snapshot.failed, 0);
        assert!(snapshot.lost);
        assert!(matches!(
            scheduler.acquire_on(0, WorkRequest::new(1, 1)),
            Err(ScheduleError::DeviceUnavailable { .. })
        ));
        scheduler
            .restore_device(0)
            .expect("idle lost device can restore");
        scheduler
            .acquire_on(0, WorkRequest::new(1, 1))
            .expect("restored device accepts work");
    }

    #[test]
    fn device_loss_during_cancellation_blocks_readmission() {
        let scheduler = scheduler();
        let lease = scheduler
            .acquire_on(0, WorkRequest::new(20, 1))
            .expect("reservation");
        let submission = BackendSubmission::new(
            lease,
            Box::new(ErrorCompletion(Some(BackendError::DeviceLost {
                backend: "test",
                operation: "cancel",
                reason: "device removed during cancellation".into(),
            }))),
            (),
            "gpu0".into(),
        );

        assert!(matches!(
            submission.cancel(),
            Err(ScheduleError::DeviceLost { ref reason, .. })
                if reason == "device removed during cancellation"
        ));
        let snapshot = scheduler.snapshots().expect("snapshot")[0].clone();
        assert_eq!(snapshot.bytes_in_use, 0);
        assert_eq!(snapshot.active, 0);
        assert_eq!(snapshot.device_lost, 1);
        assert!(snapshot.lost);
        assert!(matches!(
            scheduler.acquire_on(0, WorkRequest::new(1, 1)),
            Err(ScheduleError::DeviceUnavailable { .. })
        ));
    }

    #[test]
    fn repeated_completion_races_have_one_terminal_refund() {
        let mut seed = 0x9e37_79b9_u32;
        for _ in 0..32 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let scheduler = scheduler();
            let started = Arc::new(AtomicBool::new(false));
            let release = Arc::new(AtomicBool::new(false));
            let lease = scheduler
                .acquire_on(0, WorkRequest::new(20, 1))
                .expect("reservation");
            let submission = BackendSubmission::new(
                lease,
                Box::new(GateCompletion {
                    started: Arc::clone(&started),
                    release: Arc::clone(&release),
                }),
                7_u32,
                "gpu0".into(),
            );
            let complete = submission.clone();
            let cancel = submission.clone();
            let complete_first = seed & 1 == 0;
            let first = thread::spawn(move || {
                if complete_first {
                    complete.complete().map(|_| ())
                } else {
                    cancel.cancel()
                }
            });
            let second_submission = submission.clone();
            let second = thread::spawn(move || second_submission.cancel());
            while !started.load(Ordering::Acquire) {
                thread::yield_now();
            }
            release.store(true, Ordering::Release);
            let first_result = first.join().expect("first race participant");
            let second_result = second.join().expect("second race participant");
            assert!(first_result.is_ok() || second_result.is_ok());
            assert!(first_result.is_err() || second_result.is_err());
            let snapshot = scheduler.snapshots().expect("snapshot")[0].clone();
            assert_eq!(snapshot.bytes_in_use, 0);
            assert_eq!(snapshot.active, 0);
            assert_eq!(snapshot.completed + snapshot.cancelled, 1);
            assert_eq!(snapshot.failed, 0);
        }
    }

    #[test]
    fn randomized_terminal_races_preserve_classification_and_refund_once() {
        let mut seed = 0x6d2b_79f5_u64;
        for iteration in 0..96 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let outcome = (seed >> 32) % 3;
            let mark_lost_concurrently = seed & 0b100 != 0;
            let complete_yields = ((seed >> 8) & 0b111) as usize;
            let cancel_yields = ((seed >> 16) & 0b111) as usize;

            let scheduler =
                DeviceScheduler::new([DeviceBudget::new("gpu0", 64, 1)]).expect("valid scheduler");
            let lease = scheduler
                .acquire_on(0, WorkRequest::new(20, 1))
                .expect("reservation");
            let started = Arc::new(AtomicBool::new(false));
            let release = Arc::new(AtomicBool::new(false));
            let calls = Arc::new(AtomicUsize::new(0));
            let injected_error = match outcome {
                0 => None,
                1 => Some(BackendError::OperationFailed {
                    operation: "injected_completion",
                    reason: format!("backend failure iteration {iteration}"),
                }),
                _ => Some(BackendError::DeviceLost {
                    backend: "test",
                    operation: "injected_completion",
                    reason: format!("device loss iteration {iteration}"),
                }),
            };
            let submission = BackendSubmission::new(
                lease,
                Box::new(InjectedCompletion {
                    started: Arc::clone(&started),
                    release: Arc::clone(&release),
                    calls: Arc::clone(&calls),
                    outcome: injected_error,
                }),
                iteration,
                "gpu0".into(),
            );
            let start = Arc::new(Barrier::new(3));

            let complete = submission.clone();
            let complete_start = Arc::clone(&start);
            let complete_thread = thread::spawn(move || {
                complete_start.wait();
                for _ in 0..complete_yields {
                    thread::yield_now();
                }
                complete.complete().map(|_| ())
            });
            let cancel = submission.clone();
            let cancel_start = Arc::clone(&start);
            let cancel_thread = thread::spawn(move || {
                cancel_start.wait();
                for _ in 0..cancel_yields {
                    thread::yield_now();
                }
                cancel.cancel()
            });

            start.wait();
            while !started.load(Ordering::Acquire) {
                thread::yield_now();
            }
            if mark_lost_concurrently {
                scheduler
                    .mark_device_lost(0)
                    .expect("concurrent device-loss transition");
            }
            release.store(true, Ordering::Release);

            let complete_result = complete_thread.join().expect("complete participant");
            let cancel_result = cancel_thread.join().expect("cancel participant");
            let results = [&complete_result, &cancel_result];
            assert_eq!(
                results
                    .iter()
                    .filter(|result| {
                        matches!(
                            result,
                            Err(ScheduleError::SubmissionAlreadyFinalized { .. })
                        )
                    })
                    .count(),
                1,
                "exactly one racer must lose terminal ownership at iteration {iteration}"
            );
            let terminal_result = results
                .into_iter()
                .find(|result| {
                    !matches!(
                        result,
                        Err(ScheduleError::SubmissionAlreadyFinalized { .. })
                    )
                })
                .expect("one terminal owner");
            match outcome {
                0 => assert_eq!(terminal_result, &Ok(())),
                1 => assert!(matches!(
                    terminal_result,
                    Err(ScheduleError::BackendFailure { reason, .. })
                        if reason.contains("backend failure iteration")
                )),
                _ => assert!(matches!(
                    terminal_result,
                    Err(ScheduleError::DeviceLost { reason, .. })
                        if reason.contains("device loss iteration")
                )),
            }
            assert_eq!(
                calls.load(Ordering::Acquire),
                1,
                "completion token must be polled exactly once"
            );

            let snapshot = scheduler.snapshots().expect("snapshot")[0].clone();
            assert_eq!(snapshot.bytes_in_use, 0);
            assert_eq!(snapshot.active, 0);
            match outcome {
                0 => {
                    assert_eq!(snapshot.completed + snapshot.cancelled, 1);
                    assert_eq!(snapshot.failed, 0);
                    assert_eq!(snapshot.device_lost, 0);
                }
                1 => {
                    assert_eq!(snapshot.failed, 1);
                    assert_eq!(snapshot.completed + snapshot.cancelled, 0);
                    assert_eq!(snapshot.device_lost, 0);
                }
                _ => {
                    assert_eq!(snapshot.device_lost, 1);
                    assert_eq!(snapshot.completed + snapshot.cancelled + snapshot.failed, 0);
                    assert!(snapshot.lost);
                }
            }
            assert_eq!(
                snapshot.lost,
                mark_lost_concurrently || outcome == 2,
                "lost-state classification drift at iteration {iteration}"
            );

            if snapshot.lost {
                scheduler.restore_device(0).expect("restore idle device");
            }
            drop(
                scheduler
                    .acquire_on(0, WorkRequest::new(64, 1))
                    .expect("full-budget readmission after terminal refund"),
            );
            let final_snapshot = scheduler.snapshots().expect("final snapshot")[0].clone();
            assert_eq!(final_snapshot.bytes_in_use, 0);
            assert_eq!(final_snapshot.active, 0);
        }
    }

    #[test]
    fn explicit_lease_outcomes_are_counted_once() {
        let scheduler = scheduler();
        scheduler
            .acquire_on(0, WorkRequest::new(10, 1))
            .expect("completed reservation")
            .complete()
            .expect("completion refund");
        scheduler
            .acquire_on(0, WorkRequest::new(10, 1))
            .expect("failed reservation")
            .fail()
            .expect("failure refund");
        scheduler
            .acquire_on(0, WorkRequest::new(10, 1))
            .expect("cancelled reservation")
            .cancel()
            .expect("cancellation refund");

        let snapshot = scheduler.snapshots().expect("snapshot")[0].clone();
        assert_eq!(snapshot.bytes_in_use, 0);
        assert_eq!(snapshot.active, 0);
        assert_eq!(snapshot.completed, 1);
        assert_eq!(snapshot.failed, 1);
        assert_eq!(snapshot.cancelled, 1);
    }

    #[test]
    fn poisoned_refund_accounts_once_and_lease_is_terminal() {
        let scheduler = scheduler();
        let lease = scheduler
            .acquire_on(0, WorkRequest::new(10, 1))
            .expect("reservation");
        let inner = Arc::clone(&scheduler.inner);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = inner.devices[0].state.lock().unwrap();
            panic!("inject scheduler state poisoning");
        }));

        assert_eq!(lease.complete(), Err(ScheduleError::StatePoisoned));
        let state = match inner.devices[0].state.lock() {
            Ok(_) => panic!("state should remain poisoned"),
            Err(error) => error.into_inner(),
        };
        assert_eq!(state.in_use, 0);
        assert_eq!(state.active, 0);
        assert_eq!(state.completed, 1);
    }
}
