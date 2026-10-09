//! Checkpoint and cooperative-cancellation contracts for FEM run dispatch.
//!
//! The production FEM loop owns mesh assembly and solve execution while using
//! this module as a small, serializable state boundary. A checkpoint is bound
//! to the complete deterministic [`SolveSchedule`], commits work in schedule
//! order, and refuses to continue after an observed memory breach. The generic
//! callback executor remains available for backends whose solve work is
//! already task-shaped.

use super::scheduler::{SolveSchedule, SolveWorkItem, partition_solve_schedule};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sonium_solvers::{CheckpointResidence, DistributedCheckpoint};
use std::collections::BTreeMap;
use std::fmt;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc,
};

/// Versioned JSON schema identifier for [`FemRunCheckpoint`].
pub const FEM_RUN_CHECKPOINT_SCHEMA_V1: &str = "roomsim.fem-run-checkpoint.v1";

/// Lifecycle state recorded by a FEM run checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FemRunStatus {
    Pending,
    Running,
    Completed,
    Cancelled,
}

/// Persisted, schedule-bound FEM run cursor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FemRunCheckpoint {
    /// Schema identifier; kept explicit so future formats fail closed.
    pub schema: String,
    pub status: FemRunStatus,
    /// SHA-256 over the complete lane/task topology, not just summary counts.
    pub schedule_fingerprint: String,
    pub schedule_task_count: usize,
    pub schedule_peak_concurrent_bytes: usize,
    pub memory_budget_bytes: usize,
    /// The next zero-based task in deterministic lane-major order.
    pub next_task_index: usize,
    pub committed_task_count: usize,
    pub observed_peak_bytes: usize,
    pub cancellation_reason: Option<String>,
}

/// Errors raised by checkpoint validation or state transitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FemRunStateError {
    InvalidCheckpoint(String),
    InvalidJson(String),
    Io(String),
    InvalidSchedule(String),
    ScheduleBindingMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
    InvalidTransition {
        status: FemRunStatus,
        action: &'static str,
    },
    CursorMismatch {
        expected: usize,
        actual: usize,
    },
    TaskMismatch {
        task_index: usize,
    },
    IncompleteRun {
        committed: usize,
        task_count: usize,
    },
    ObservedPeakExceedsBudget {
        observed: usize,
        budget: usize,
    },
    EmptyCancellationReason,
    DistributedCheckpoint(String),
}

impl fmt::Display for FemRunStateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCheckpoint(reason) => {
                write!(formatter, "invalid FEM checkpoint: {reason}")
            }
            Self::InvalidJson(reason) => write!(formatter, "invalid FEM checkpoint JSON: {reason}"),
            Self::Io(reason) => write!(formatter, "FEM checkpoint I/O failed: {reason}"),
            Self::InvalidSchedule(reason) => {
                write!(formatter, "invalid FEM solve schedule: {reason}")
            }
            Self::ScheduleBindingMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "FEM checkpoint schedule {field} mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidTransition { status, action } => {
                write!(formatter, "cannot {action} FEM run in {status:?} state")
            }
            Self::CursorMismatch { expected, actual } => {
                write!(
                    formatter,
                    "FEM run cursor mismatch: expected {expected}, got {actual}"
                )
            }
            Self::TaskMismatch { task_index } => {
                write!(
                    formatter,
                    "FEM task {task_index} does not match the bound schedule"
                )
            }
            Self::IncompleteRun {
                committed,
                task_count,
            } => write!(
                formatter,
                "cannot complete FEM run: committed {committed} of {task_count} tasks"
            ),
            Self::ObservedPeakExceedsBudget { observed, budget } => write!(
                formatter,
                "observed FEM peak memory {observed} exceeds budget {budget}"
            ),
            Self::EmptyCancellationReason => {
                write!(formatter, "FEM cancellation reason must not be empty")
            }
            Self::DistributedCheckpoint(reason) => {
                write!(formatter, "distributed FEM checkpoint failed: {reason}")
            }
        }
    }
}

impl std::error::Error for FemRunStateError {}

/// Summary returned after every scheduled task has been dispatched and
/// committed through [`execute_scheduled_fem_run`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FemRunExecutionReport {
    pub status: FemRunStatus,
    pub committed_task_count: usize,
    pub observed_peak_bytes: usize,
}

/// Failure from the checked schedule-dispatch boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FemRunExecutionError {
    State(FemRunStateError),
    Task { task_index: usize, error: String },
    Cancelled { reason: String },
}

impl fmt::Display for FemRunExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::State(error) => error.fmt(formatter),
            Self::Task { task_index, error } => {
                write!(formatter, "scheduled FEM task {task_index} failed: {error}")
            }
            Self::Cancelled { reason } => {
                write!(formatter, "scheduled FEM run cancelled: {reason}")
            }
        }
    }
}

impl std::error::Error for FemRunExecutionError {}

impl From<FemRunStateError> for FemRunExecutionError {
    fn from(error: FemRunStateError) -> Self {
        Self::State(error)
    }
}

impl FemRunCheckpoint {
    /// Create a pending checkpoint for an already-admitted solve schedule.
    pub fn new(
        schedule: &SolveSchedule,
        memory_budget_bytes: usize,
    ) -> Result<Self, FemRunStateError> {
        let task_count = ordered_tasks(schedule)?.len();
        if task_count == 0 {
            return Err(FemRunStateError::InvalidSchedule(
                "schedule must contain at least one task".into(),
            ));
        }
        if memory_budget_bytes == 0 {
            return Err(FemRunStateError::InvalidCheckpoint(
                "memory budget must be positive".into(),
            ));
        }
        if schedule.peak_concurrent_bytes > memory_budget_bytes {
            return Err(FemRunStateError::InvalidCheckpoint(format!(
                "schedule peak {} exceeds memory budget {}",
                schedule.peak_concurrent_bytes, memory_budget_bytes
            )));
        }

        let checkpoint = Self {
            schema: FEM_RUN_CHECKPOINT_SCHEMA_V1.into(),
            status: FemRunStatus::Pending,
            schedule_fingerprint: fem_schedule_fingerprint(schedule),
            schedule_task_count: task_count,
            schedule_peak_concurrent_bytes: schedule.peak_concurrent_bytes,
            memory_budget_bytes,
            next_task_index: 0,
            committed_task_count: 0,
            observed_peak_bytes: 0,
            cancellation_reason: None,
        };
        checkpoint.validate_fields()?;
        Ok(checkpoint)
    }

    /// Validate schema, cursor invariants, and the complete schedule binding.
    pub fn validate_for(&self, schedule: &SolveSchedule) -> Result<(), FemRunStateError> {
        self.validate_fields()?;
        let task_count = ordered_tasks(schedule)?.len();
        check_binding("task_count", self.schedule_task_count, task_count)?;
        check_binding(
            "peak_concurrent_bytes",
            self.schedule_peak_concurrent_bytes,
            schedule.peak_concurrent_bytes,
        )?;
        let fingerprint = fem_schedule_fingerprint(schedule);
        if self.schedule_fingerprint != fingerprint {
            return Err(FemRunStateError::ScheduleBindingMismatch {
                field: "fingerprint",
                expected: self.schedule_fingerprint.clone(),
                actual: fingerprint,
            });
        }
        Ok(())
    }

    /// Move a pending run into execution.
    pub fn start(&mut self, schedule: &SolveSchedule) -> Result<(), FemRunStateError> {
        self.validate_for(schedule)?;
        if self.status != FemRunStatus::Pending {
            return Err(FemRunStateError::InvalidTransition {
                status: self.status,
                action: "start",
            });
        }
        self.status = FemRunStatus::Running;
        Ok(())
    }

    /// Re-open a persisted run at its recorded cursor.
    ///
    /// A cancelled run is an interruption marker rather than a terminal
    /// result. Resuming it clears the old cancellation reason after the
    /// schedule and cursor have been validated; callers must provide a fresh
    /// cancellation token and restore any caller-owned outputs.
    pub fn resume(
        &mut self,
        schedule: &SolveSchedule,
        cursor: usize,
    ) -> Result<(), FemRunStateError> {
        self.validate_for(schedule)?;
        if self.status == FemRunStatus::Completed {
            return Err(FemRunStateError::InvalidTransition {
                status: self.status,
                action: "resume",
            });
        }
        if self.next_task_index != cursor {
            return Err(FemRunStateError::CursorMismatch {
                expected: self.next_task_index,
                actual: cursor,
            });
        }
        self.cancellation_reason = None;
        self.status = FemRunStatus::Running;
        Ok(())
    }

    /// Commit exactly the task at the current deterministic cursor.
    pub fn commit_next_task(
        &mut self,
        schedule: &SolveSchedule,
        task_index: usize,
        task: SolveWorkItem,
        observed_peak_bytes: usize,
    ) -> Result<(), FemRunStateError> {
        self.validate_for(schedule)?;
        if self.status != FemRunStatus::Running {
            return Err(FemRunStateError::InvalidTransition {
                status: self.status,
                action: "commit a task",
            });
        }
        if task_index != self.next_task_index {
            return Err(FemRunStateError::CursorMismatch {
                expected: self.next_task_index,
                actual: task_index,
            });
        }
        if observed_peak_bytes > self.memory_budget_bytes {
            self.cancel_for_memory(observed_peak_bytes);
            return Err(FemRunStateError::ObservedPeakExceedsBudget {
                observed: observed_peak_bytes,
                budget: self.memory_budget_bytes,
            });
        }

        let expected = ordered_tasks(schedule)?
            .get(task_index)
            .copied()
            .ok_or_else(|| {
                FemRunStateError::InvalidCheckpoint(format!(
                    "cursor {task_index} is outside {} scheduled tasks",
                    self.schedule_task_count
                ))
            })?;
        if task != expected {
            return Err(FemRunStateError::TaskMismatch { task_index });
        }

        self.next_task_index += 1;
        self.committed_task_count += 1;
        self.observed_peak_bytes = self.observed_peak_bytes.max(observed_peak_bytes);
        Ok(())
    }

    /// Record a measured peak without committing a task.
    pub fn record_observed_peak(
        &mut self,
        schedule: &SolveSchedule,
        observed_peak_bytes: usize,
    ) -> Result<(), FemRunStateError> {
        self.validate_for(schedule)?;
        if self.status != FemRunStatus::Running {
            return Err(FemRunStateError::InvalidTransition {
                status: self.status,
                action: "record observed memory",
            });
        }
        if observed_peak_bytes > self.memory_budget_bytes {
            self.cancel_for_memory(observed_peak_bytes);
            return Err(FemRunStateError::ObservedPeakExceedsBudget {
                observed: observed_peak_bytes,
                budget: self.memory_budget_bytes,
            });
        }
        self.observed_peak_bytes = self.observed_peak_bytes.max(observed_peak_bytes);
        Ok(())
    }

    /// Mark a fully committed run complete.
    /// Validate and commit one response received from a worker transport.
    ///
    /// Responses may arrive out of order, but only the global next task can
    /// be committed; callers buffer later responses until their predecessors
    /// have been accepted.
    pub fn commit_worker_response(
        &mut self,
        schedule: &SolveSchedule,
        request: &FemWorkerTaskRequest,
        response: &FemWorkerTaskResponse,
    ) -> Result<(), FemRunStateError> {
        request.validate_for(schedule)?;
        response.validate_for(request)?;
        match response.status {
            FemWorkerTaskStatus::Completed => {
                let observed_peak_bytes = response.observed_peak_bytes.ok_or_else(|| {
                    FemRunStateError::InvalidJson(
                        "completed worker response is missing observed peak bytes".into(),
                    )
                })?;
                self.commit_next_task(
                    schedule,
                    request.task_index,
                    request.task,
                    observed_peak_bytes,
                )
            }
            FemWorkerTaskStatus::Failed => {
                let reason = response.error.as_deref().ok_or_else(|| {
                    FemRunStateError::InvalidJson(
                        "failed worker response is missing an error".into(),
                    )
                })?;
                self.cancel(schedule, reason)?;
                Err(FemRunStateError::InvalidCheckpoint(format!(
                    "worker task {} failed: {reason}",
                    request.task_index
                )))
            }
        }
    }

    pub fn complete(&mut self, schedule: &SolveSchedule) -> Result<(), FemRunStateError> {
        self.validate_for(schedule)?;
        if self.status != FemRunStatus::Running {
            return Err(FemRunStateError::InvalidTransition {
                status: self.status,
                action: "complete",
            });
        }
        if self.next_task_index != self.schedule_task_count {
            return Err(FemRunStateError::IncompleteRun {
                committed: self.committed_task_count,
                task_count: self.schedule_task_count,
            });
        }
        self.status = FemRunStatus::Completed;
        Ok(())
    }

    /// Cancel a pending or running run with a durable reason.
    pub fn cancel(
        &mut self,
        schedule: &SolveSchedule,
        reason: impl Into<String>,
    ) -> Result<(), FemRunStateError> {
        self.validate_for(schedule)?;
        if !matches!(self.status, FemRunStatus::Pending | FemRunStatus::Running) {
            return Err(FemRunStateError::InvalidTransition {
                status: self.status,
                action: "cancel",
            });
        }
        let reason = reason.into();
        if reason.trim().is_empty() {
            return Err(FemRunStateError::EmptyCancellationReason);
        }
        self.status = FemRunStatus::Cancelled;
        self.cancellation_reason = Some(reason);
        Ok(())
    }

    /// Import a cancellation request at a cooperative polling point.
    pub fn poll_cancellation(
        &mut self,
        schedule: &SolveSchedule,
        cancellation: &FemRunCancellation,
    ) -> Result<bool, FemRunStateError> {
        self.validate_for(schedule)?;
        if !cancellation.is_cancelled() {
            return Ok(false);
        }
        if self.status == FemRunStatus::Cancelled {
            return Ok(true);
        }
        let reason = cancellation
            .first_error()
            .unwrap_or_else(|| "cancelled cooperatively".into());
        self.cancel(schedule, reason)?;
        Ok(true)
    }

    /// Serialize after checking internal invariants.
    pub fn to_json(&self) -> Result<String, FemRunStateError> {
        self.validate_fields()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| FemRunStateError::InvalidJson(error.to_string()))
    }

    /// Deserialize a strict, versioned checkpoint.
    pub fn from_json(document: &str) -> Result<Self, FemRunStateError> {
        let checkpoint: Self = serde_json::from_str(document)
            .map_err(|error| FemRunStateError::InvalidJson(error.to_string()))?;
        checkpoint.validate_fields()?;
        Ok(checkpoint)
    }

    /// Serialize this run cursor through the portable distributed checkpoint
    /// envelope used for rank or device hand-off.
    ///
    /// The JSON checkpoint remains the typed payload, while the outer envelope
    /// provides deterministic sharding, residence metadata, and checksums.
    /// The recipient must still validate the restored cursor against its local
    /// [`SolveSchedule`] before resuming.
    pub fn to_distributed_checkpoint_bytes(
        &self,
        world_size: u32,
        residence: CheckpointResidence,
    ) -> Result<Vec<u8>, FemRunStateError> {
        let payload = self.to_json()?.into_bytes();
        let envelope = DistributedCheckpoint::split_payload(&payload, world_size, residence)
            .map_err(|error| FemRunStateError::DistributedCheckpoint(error.to_string()))?;
        envelope
            .to_checkpoint_bytes()
            .map_err(|error| FemRunStateError::DistributedCheckpoint(error.to_string()))
    }

    /// Restore a run cursor from a checked distributed checkpoint envelope.
    pub fn from_distributed_checkpoint_bytes(bytes: &[u8]) -> Result<Self, FemRunStateError> {
        let envelope = DistributedCheckpoint::from_checkpoint_bytes(bytes)
            .map_err(|error| FemRunStateError::DistributedCheckpoint(error.to_string()))?;
        let payload = envelope
            .assemble_payload()
            .map_err(|error| FemRunStateError::DistributedCheckpoint(error.to_string()))?;
        let document = String::from_utf8(payload)
            .map_err(|error| FemRunStateError::DistributedCheckpoint(error.to_string()))?;
        Self::from_json(&document)
    }

    /// Write a validated checkpoint atomically from the caller's perspective.
    ///
    /// The helper does not replace an existing file transactionally; callers
    /// that need crash-safe replacement should write to a sibling temporary
    /// path and rename it after this method succeeds.
    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<(), FemRunStateError> {
        std::fs::write(path.as_ref(), self.to_json()?)
            .map_err(|error| FemRunStateError::Io(format!("{}: {error}", path.as_ref().display())))
    }

    /// Read, strictly decode, and validate a checkpoint from disk.
    pub fn read_json(path: impl AsRef<Path>) -> Result<Self, FemRunStateError> {
        let document = std::fs::read_to_string(path.as_ref()).map_err(|error| {
            FemRunStateError::Io(format!("{}: {error}", path.as_ref().display()))
        })?;
        Self::from_json(&document)
    }

    fn validate_fields(&self) -> Result<(), FemRunStateError> {
        if self.schema != FEM_RUN_CHECKPOINT_SCHEMA_V1 {
            return Err(FemRunStateError::InvalidCheckpoint(format!(
                "unsupported schema {}",
                self.schema
            )));
        }
        if self.schedule_task_count == 0 {
            return Err(FemRunStateError::InvalidCheckpoint(
                "schedule task count must be positive".into(),
            ));
        }
        if !is_sha256_hex(&self.schedule_fingerprint) {
            return Err(FemRunStateError::InvalidCheckpoint(
                "schedule fingerprint must be a lowercase SHA-256 digest".into(),
            ));
        }
        if self.memory_budget_bytes == 0 {
            return Err(FemRunStateError::InvalidCheckpoint(
                "memory budget must be positive".into(),
            ));
        }
        if self.schedule_peak_concurrent_bytes > self.memory_budget_bytes {
            return Err(FemRunStateError::InvalidCheckpoint(
                "schedule peak already exceeds memory budget".into(),
            ));
        }
        if self.next_task_index != self.committed_task_count {
            return Err(FemRunStateError::InvalidCheckpoint(
                "next task and committed task cursors differ".into(),
            ));
        }
        if self.next_task_index > self.schedule_task_count {
            return Err(FemRunStateError::InvalidCheckpoint(
                "task cursor exceeds schedule task count".into(),
            ));
        }
        if self.observed_peak_bytes > self.memory_budget_bytes {
            return Err(FemRunStateError::InvalidCheckpoint(
                "observed peak already exceeds memory budget".into(),
            ));
        }
        match self.status {
            FemRunStatus::Pending if self.next_task_index != 0 => {
                return Err(FemRunStateError::InvalidCheckpoint(
                    "pending checkpoint must start at task zero".into(),
                ));
            }
            FemRunStatus::Completed if self.next_task_index != self.schedule_task_count => {
                return Err(FemRunStateError::InvalidCheckpoint(
                    "completed checkpoint has uncommitted tasks".into(),
                ));
            }
            FemRunStatus::Pending | FemRunStatus::Running | FemRunStatus::Completed => {
                if self.cancellation_reason.is_some() {
                    return Err(FemRunStateError::InvalidCheckpoint(
                        "non-cancelled checkpoint has a cancellation reason".into(),
                    ));
                }
            }
            FemRunStatus::Cancelled => {
                if self
                    .cancellation_reason
                    .as_deref()
                    .is_none_or(|reason| reason.trim().is_empty())
                {
                    return Err(FemRunStateError::InvalidCheckpoint(
                        "cancelled checkpoint requires a reason".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn cancel_for_memory(&mut self, observed_peak_bytes: usize) {
        self.status = FemRunStatus::Cancelled;
        self.cancellation_reason = Some(format!(
            "observed peak memory {observed_peak_bytes} exceeds budget {}",
            self.memory_budget_bytes
        ));
    }
}

/// Cloneable cooperative cancellation shared by FEM worker lanes.
#[derive(Clone, Debug, Default)]
pub struct FemRunCancellation {
    cancelled: Arc<AtomicBool>,
    first_error: Arc<Mutex<Option<String>>>,
}

impl FemRunCancellation {
    /// Create a clear cancellation token.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Set cancellation once; later errors cannot replace the first reason.
    pub fn cancel(&self, reason: impl Into<String>) {
        let reason = reason.into();
        if self
            .cancelled
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let reason = if reason.trim().is_empty() {
                "cancelled cooperatively".into()
            } else {
                reason
            };
            *self
                .first_error
                .lock()
                .expect("FEM cancellation mutex poisoned") = Some(reason);
        }
    }

    pub fn first_error(&self) -> Option<String> {
        self.first_error
            .lock()
            .expect("FEM cancellation mutex poisoned")
            .clone()
    }
}

/// Dispatch every task in deterministic lane-major order through a caller's
/// solve callback, committing each successful result to the checkpoint.
///
/// This is the runtime coordination seam used by a FEM backend: the callback
/// performs the actual assembly/solve/extraction and returns its observed peak
/// bytes. A pending checkpoint is started automatically; a running checkpoint
/// resumes at its persisted cursor. The callback is never invoked after a
/// cooperative cancellation request, and callback failures become durable
/// cancellation reasons before being returned.
pub fn execute_scheduled_fem_run<F>(
    schedule: &SolveSchedule,
    checkpoint: &mut FemRunCheckpoint,
    cancellation: &FemRunCancellation,
    mut execute: F,
) -> Result<FemRunExecutionReport, FemRunExecutionError>
where
    F: FnMut(SolveWorkItem) -> Result<usize, String>,
{
    let tasks = ordered_tasks(schedule).map_err(FemRunExecutionError::State)?;
    match checkpoint.status {
        FemRunStatus::Pending => checkpoint.start(schedule)?,
        FemRunStatus::Running | FemRunStatus::Cancelled => {
            let cursor = checkpoint.next_task_index;
            checkpoint.resume(schedule, cursor)?;
        }
        status => {
            return Err(FemRunExecutionError::State(
                FemRunStateError::InvalidTransition {
                    status,
                    action: "execute scheduled tasks",
                },
            ));
        }
    }

    while checkpoint.next_task_index < tasks.len() {
        if cancellation.is_cancelled() {
            let reason = cancellation
                .first_error()
                .unwrap_or_else(|| "cancelled cooperatively".into());
            checkpoint.poll_cancellation(schedule, cancellation)?;
            return Err(FemRunExecutionError::Cancelled { reason });
        }

        let task_index = checkpoint.next_task_index;
        let task = tasks[task_index];
        let observed_peak_bytes = match execute(task) {
            Ok(observed_peak_bytes) => observed_peak_bytes,
            Err(error) => {
                let reason = if error.trim().is_empty() {
                    "scheduled task failed".to_string()
                } else {
                    error.clone()
                };
                cancellation.cancel(reason.clone());
                // The checkpoint is already validated and running, so this
                // poll can only fail if an invariant was concurrently or
                // externally corrupted; preserve the task error otherwise.
                let _ = checkpoint.poll_cancellation(schedule, cancellation);
                return Err(FemRunExecutionError::Task { task_index, error });
            }
        };
        checkpoint.commit_next_task(schedule, task_index, task, observed_peak_bytes)?;
    }

    checkpoint.complete(schedule)?;
    Ok(FemRunExecutionReport {
        status: checkpoint.status,
        committed_task_count: checkpoint.committed_task_count,
        observed_peak_bytes: checkpoint.observed_peak_bytes,
    })
}

enum ConcurrentTaskMessage {
    Completed {
        task_index: usize,
        task: SolveWorkItem,
        observed_peak_bytes: usize,
    },
    Failed {
        task_index: usize,
        error: String,
    },
}

/// Dispatch independent schedule lanes concurrently while committing results
/// in deterministic lane-major order.
///
/// Each lane owns its task order and can therefore retain continuation state
/// inside the callback. The checkpoint remains single-writer: completed work
/// is buffered until the next global cursor is available, so a fast later lane
/// cannot make a restart observe an out-of-order commit. Cancellation is
/// cooperative; an already-running callback may finish before its worker
/// notices the shared token.
pub fn execute_scheduled_fem_run_concurrent<F>(
    schedule: &SolveSchedule,
    checkpoint: &mut FemRunCheckpoint,
    cancellation: &FemRunCancellation,
    execute: F,
) -> Result<FemRunExecutionReport, FemRunExecutionError>
where
    F: Fn(SolveWorkItem) -> Result<usize, String> + Send + Sync,
{
    execute_scheduled_fem_run_concurrent_with_commit(
        schedule,
        checkpoint,
        cancellation,
        execute,
        |_checkpoint, _task| Ok(()),
    )
}

/// Dispatch scheduled tasks concurrently and notify the caller after each
/// deterministic checkpoint commit.
///
/// The callback runs on the single writer thread after the task has been
/// committed, so callers can durably persist or update the checkpoint before
/// the next task is committed. A callback
/// error cancels the run and is reported as a task failure.
pub fn execute_scheduled_fem_run_concurrent_with_commit<F, C>(
    schedule: &SolveSchedule,
    checkpoint: &mut FemRunCheckpoint,
    cancellation: &FemRunCancellation,
    execute: F,
    mut on_commit: C,
) -> Result<FemRunExecutionReport, FemRunExecutionError>
where
    F: Fn(SolveWorkItem) -> Result<usize, String> + Send + Sync,
    C: FnMut(&mut FemRunCheckpoint, SolveWorkItem) -> Result<(), String>,
{
    let tasks = ordered_tasks(schedule).map_err(FemRunExecutionError::State)?;
    match checkpoint.status {
        FemRunStatus::Pending => checkpoint.start(schedule)?,
        FemRunStatus::Running | FemRunStatus::Cancelled => {
            let cursor = checkpoint.next_task_index;
            checkpoint.resume(schedule, cursor)?;
        }
        status => {
            return Err(FemRunExecutionError::State(
                FemRunStateError::InvalidTransition {
                    status,
                    action: "execute concurrent scheduled tasks",
                },
            ));
        }
    }

    if cancellation.is_cancelled() {
        let reason = cancellation
            .first_error()
            .unwrap_or_else(|| "cancelled cooperatively".into());
        checkpoint.poll_cancellation(schedule, cancellation)?;
        return Err(FemRunExecutionError::Cancelled { reason });
    }

    let start_index = checkpoint.next_task_index;
    let mut lane_starts = Vec::with_capacity(schedule.lanes.len());
    let mut lane_start = 0usize;
    for lane in &schedule.lanes {
        lane_starts.push(lane_start);
        lane_start += lane.tasks.len();
    }

    let (sender, receiver) = mpsc::channel();

    std::thread::scope(|scope| {
        for (lane_index, lane) in schedule.lanes.iter().enumerate() {
            let sender = sender.clone();
            let cancellation = cancellation.clone();
            let execute = &execute;
            let lane_start = lane_starts[lane_index];
            scope.spawn(move || {
                for (local_index, task) in lane.tasks.iter().copied().enumerate() {
                    let task_index = lane_start + local_index;
                    if task_index < start_index || cancellation.is_cancelled() {
                        continue;
                    }
                    match execute(task) {
                        Ok(observed_peak_bytes) => {
                            let _ = sender.send(ConcurrentTaskMessage::Completed {
                                task_index,
                                task,
                                observed_peak_bytes,
                            });
                        }
                        Err(error) => {
                            let reason = if error.trim().is_empty() {
                                "scheduled task failed".to_string()
                            } else {
                                error.clone()
                            };
                            cancellation.cancel(reason);
                            let _ =
                                sender.send(ConcurrentTaskMessage::Failed { task_index, error });
                            break;
                        }
                    }
                }
            });
        }
        drop(sender);

        let mut completed = BTreeMap::new();
        let mut next_task_index = start_index;
        while let Ok(message) = receiver.recv() {
            match message {
                ConcurrentTaskMessage::Completed {
                    task_index,
                    task,
                    observed_peak_bytes,
                } => {
                    completed.insert(task_index, (task, observed_peak_bytes));
                }
                ConcurrentTaskMessage::Failed { task_index, error } => {
                    let reason = if error.trim().is_empty() {
                        "scheduled task failed".to_string()
                    } else {
                        error.clone()
                    };
                    cancellation.cancel(reason);
                    let _ = checkpoint.poll_cancellation(schedule, cancellation);
                    return Err(FemRunExecutionError::Task { task_index, error });
                }
            }

            while !cancellation.is_cancelled() {
                let Some((task, observed_peak_bytes)) = completed.remove(&next_task_index) else {
                    break;
                };
                if let Err(error) = checkpoint.commit_next_task(
                    schedule,
                    next_task_index,
                    task,
                    observed_peak_bytes,
                ) {
                    cancellation.cancel(error.to_string());
                    return Err(FemRunExecutionError::State(error));
                }
                if let Err(error) = on_commit(checkpoint, task) {
                    let reason = if error.trim().is_empty() {
                        "checkpoint commit callback failed".to_string()
                    } else {
                        error
                    };
                    cancellation.cancel(reason.clone());
                    let _ = checkpoint.poll_cancellation(schedule, cancellation);
                    return Err(FemRunExecutionError::Task {
                        task_index: next_task_index,
                        error: reason,
                    });
                }
                next_task_index += 1;
            }

            if cancellation.is_cancelled() {
                let reason = cancellation
                    .first_error()
                    .unwrap_or_else(|| "cancelled cooperatively".into());
                let _ = checkpoint.poll_cancellation(schedule, cancellation);
                return Err(FemRunExecutionError::Cancelled { reason });
            }
        }

        if next_task_index != tasks.len() {
            let reason = cancellation
                .first_error()
                .unwrap_or_else(|| "concurrent FEM worker terminated early".into());
            cancellation.cancel(reason.clone());
            let _ = checkpoint.poll_cancellation(schedule, cancellation);
            return Err(FemRunExecutionError::Cancelled { reason });
        }

        checkpoint.complete(schedule)?;
        Ok(FemRunExecutionReport {
            status: checkpoint.status,
            committed_task_count: checkpoint.committed_task_count,
            observed_peak_bytes: checkpoint.observed_peak_bytes,
        })
    })
}

/// Return a deterministic digest of all schedule lanes, candidates, and tasks.
/// Execute a schedule through deterministic worker assignments while keeping
/// the existing single-writer, lane-major checkpoint commit protocol.
///
/// The worker index is supplied to both the task callback and the commit
/// callback. This is the coordinator seam used by a future process or rank
/// transport; the current implementation still uses the local concurrent
/// executor for task execution.
pub fn execute_partitioned_scheduled_fem_run_concurrent_with_commit<F, C>(
    schedule: &SolveSchedule,
    checkpoint: &mut FemRunCheckpoint,
    cancellation: &FemRunCancellation,
    worker_count: usize,
    execute: F,
    mut on_commit: C,
) -> Result<FemRunExecutionReport, FemRunExecutionError>
where
    F: Fn(usize, SolveWorkItem) -> Result<usize, String> + Send + Sync,
    C: FnMut(&mut FemRunCheckpoint, usize, SolveWorkItem) -> Result<(), String>,
{
    let assignments = partition_solve_schedule(schedule, worker_count).map_err(|error| {
        FemRunExecutionError::State(FemRunStateError::InvalidSchedule(error.to_string()))
    })?;
    let tasks = ordered_tasks(schedule).map_err(FemRunExecutionError::State)?;
    let mut worker_by_task = vec![None; tasks.len()];
    for assignment in &assignments {
        assignment.validate_for(schedule).map_err(|error| {
            FemRunExecutionError::State(FemRunStateError::InvalidSchedule(error.to_string()))
        })?;
        for &task_index in &assignment.task_indices {
            let worker = worker_by_task.get_mut(task_index).ok_or_else(|| {
                FemRunExecutionError::State(FemRunStateError::InvalidSchedule(
                    "worker assignment task index is outside schedule".into(),
                ))
            })?;
            if worker.replace(assignment.worker_index).is_some() {
                return Err(FemRunExecutionError::State(
                    FemRunStateError::InvalidSchedule(
                        "worker assignment contains duplicate task index".into(),
                    ),
                ));
            }
        }
    }
    let worker_by_task = worker_by_task
        .into_iter()
        .enumerate()
        .map(|(task_index, worker)| {
            worker.ok_or_else(|| {
                FemRunExecutionError::State(FemRunStateError::InvalidSchedule(format!(
                    "worker assignment omits task {task_index}"
                )))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let execute_tasks = tasks.clone();
    let execute_workers = worker_by_task.clone();
    let execute_for_worker = move |task: SolveWorkItem| {
        let task_index = execute_tasks
            .iter()
            .position(|candidate| *candidate == task)
            .ok_or_else(|| "worker callback received an unknown scheduled task".to_string())?;
        execute(execute_workers[task_index], task)
    };

    let commit_tasks = tasks;
    let commit_workers = worker_by_task;
    let commit_for_worker = move |checkpoint: &mut FemRunCheckpoint, task: SolveWorkItem| {
        let task_index = commit_tasks
            .iter()
            .position(|candidate| *candidate == task)
            .ok_or_else(|| "commit callback received an unknown scheduled task".to_string())?;
        on_commit(checkpoint, commit_workers[task_index], task)
    };

    execute_scheduled_fem_run_concurrent_with_commit(
        schedule,
        checkpoint,
        cancellation,
        execute_for_worker,
        commit_for_worker,
    )
}

/// Versioned request/response schema for process or rank task transport.
pub const FEM_WORKER_TASK_PROTOCOL_SCHEMA_V1: &str = "roomsim.fem-worker-task.v1";

/// Maximum encoded worker frame accepted by the process/rank transport.
///
/// Worker tasks currently contain a compact schedule item, so accepting an
/// unbounded length prefix would turn malformed peer input into an allocation
/// primitive.  The limit is deliberately part of the transport contract and
/// is checked before decoding or allocating a payload buffer.
pub const FEM_WORKER_TASK_MAX_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FemWorkerTaskFrameKind {
    Request,
    Response,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FemWorkerTaskFrame {
    schema: String,
    kind: FemWorkerTaskFrameKind,
    payload: serde_json::Value,
}

/// Strict task request sent from a coordinator to one worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FemWorkerTaskRequest {
    pub schema: String,
    pub schedule_fingerprint: String,
    pub worker_index: usize,
    pub worker_count: usize,
    pub task_index: usize,
    pub task: SolveWorkItem,
}

impl FemWorkerTaskRequest {
    /// Build a request bound to a schedule and its deterministic partition.
    pub fn from_schedule(
        schedule: &SolveSchedule,
        worker_count: usize,
        task_index: usize,
    ) -> Result<Self, FemRunStateError> {
        let assignments = partition_solve_schedule(schedule, worker_count)
            .map_err(|error| FemRunStateError::InvalidSchedule(error.to_string()))?;
        let tasks = ordered_tasks(schedule)?;
        let task = tasks.get(task_index).copied().ok_or_else(|| {
            FemRunStateError::InvalidSchedule(format!(
                "worker task index {task_index} is outside {} scheduled tasks",
                tasks.len()
            ))
        })?;
        let worker_index = assignments
            .iter()
            .find(|assignment| assignment.task_indices.contains(&task_index))
            .map(|assignment| assignment.worker_index)
            .ok_or_else(|| {
                FemRunStateError::InvalidSchedule(format!(
                    "worker task index {task_index} has no assignment"
                ))
            })?;
        let request = Self {
            schema: FEM_WORKER_TASK_PROTOCOL_SCHEMA_V1.into(),
            schedule_fingerprint: fem_schedule_fingerprint(schedule),
            worker_index,
            worker_count,
            task_index,
            task,
        };
        request.validate_for(schedule)?;
        Ok(request)
    }

    fn validate_fields(&self) -> Result<(), FemRunStateError> {
        if self.schema != FEM_WORKER_TASK_PROTOCOL_SCHEMA_V1 {
            return Err(FemRunStateError::InvalidJson(format!(
                "unsupported worker task schema {:?}",
                self.schema
            )));
        }
        if self.worker_count == 0 || self.worker_index >= self.worker_count {
            return Err(FemRunStateError::InvalidSchedule(
                "worker task identity is outside worker count".into(),
            ));
        }
        if self.schedule_fingerprint.len() != 64
            || !self
                .schedule_fingerprint
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(FemRunStateError::InvalidSchedule(
                "worker task schedule fingerprint is not canonical SHA-256".into(),
            ));
        }
        Ok(())
    }

    /// Validate task identity, schedule binding, and worker ownership.
    pub fn validate_for(&self, schedule: &SolveSchedule) -> Result<(), FemRunStateError> {
        self.validate_fields()?;
        let fingerprint = fem_schedule_fingerprint(schedule);
        if self.schedule_fingerprint != fingerprint {
            return Err(FemRunStateError::ScheduleBindingMismatch {
                field: "fingerprint",
                expected: self.schedule_fingerprint.clone(),
                actual: fingerprint,
            });
        }
        let tasks = ordered_tasks(schedule)?;
        let expected_task = tasks.get(self.task_index).copied().ok_or_else(|| {
            FemRunStateError::InvalidSchedule(format!(
                "worker task index {} is outside {} scheduled tasks",
                self.task_index,
                tasks.len()
            ))
        })?;
        if self.task != expected_task {
            return Err(FemRunStateError::TaskMismatch {
                task_index: self.task_index,
            });
        }
        let assignments = partition_solve_schedule(schedule, self.worker_count)
            .map_err(|error| FemRunStateError::InvalidSchedule(error.to_string()))?;
        let expected_worker = assignments
            .iter()
            .find(|assignment| assignment.task_indices.contains(&self.task_index))
            .map(|assignment| assignment.worker_index)
            .ok_or_else(|| {
                FemRunStateError::InvalidSchedule(format!(
                    "worker task index {} has no assignment",
                    self.task_index
                ))
            })?;
        if self.worker_index != expected_worker {
            return Err(FemRunStateError::InvalidSchedule(
                "worker task is assigned to a different worker".into(),
            ));
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<String, FemRunStateError> {
        self.validate_fields()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| FemRunStateError::InvalidJson(error.to_string()))
    }

    pub fn from_json(document: &str) -> Result<Self, FemRunStateError> {
        let request: Self = serde_json::from_str(document)
            .map_err(|error| FemRunStateError::InvalidJson(error.to_string()))?;
        request.validate_fields()?;
        Ok(request)
    }

    /// Encode this request as one bounded, length-prefixed transport frame.
    pub fn to_wire_frame_bytes(&self) -> Result<Vec<u8>, FemRunStateError> {
        self.validate_fields()?;
        encode_worker_task_frame(FemWorkerTaskFrame {
            schema: FEM_WORKER_TASK_PROTOCOL_SCHEMA_V1.into(),
            kind: FemWorkerTaskFrameKind::Request,
            payload: serde_json::to_value(self)
                .map_err(|error| FemRunStateError::InvalidJson(error.to_string()))?,
        })
    }

    /// Decode exactly one request frame, rejecting truncation and trailing data.
    pub fn from_wire_frame_bytes(bytes: &[u8]) -> Result<Self, FemRunStateError> {
        let frame = decode_worker_task_frame(bytes)?;
        validate_worker_task_frame(&frame, FemWorkerTaskFrameKind::Request)?;
        let request: Self = serde_json::from_value(frame.payload)
            .map_err(|error| FemRunStateError::InvalidJson(error.to_string()))?;
        request.validate_fields()?;
        Ok(request)
    }

    /// Write exactly one request frame to a process/rank stream.
    pub fn write_wire_frame<W: Write>(&self, writer: &mut W) -> Result<(), FemRunStateError> {
        writer
            .write_all(&self.to_wire_frame_bytes()?)
            .map_err(|error| {
                FemRunStateError::InvalidJson(format!("worker frame write failed: {error}"))
            })
    }

    /// Read exactly one request frame from a process/rank stream.
    pub fn read_wire_frame<R: Read>(reader: &mut R) -> Result<Self, FemRunStateError> {
        Self::from_wire_frame_bytes(&read_worker_task_frame(reader)?)
    }
}

/// Worker task outcome transported back to the coordinator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FemWorkerTaskStatus {
    Completed,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FemWorkerTaskResponse {
    pub schema: String,
    pub schedule_fingerprint: String,
    pub worker_index: usize,
    pub task_index: usize,
    pub status: FemWorkerTaskStatus,
    pub observed_peak_bytes: Option<usize>,
    pub error: Option<String>,
}

impl FemWorkerTaskResponse {
    pub fn completed(request: &FemWorkerTaskRequest, observed_peak_bytes: usize) -> Self {
        Self {
            schema: FEM_WORKER_TASK_PROTOCOL_SCHEMA_V1.into(),
            schedule_fingerprint: request.schedule_fingerprint.clone(),
            worker_index: request.worker_index,
            task_index: request.task_index,
            status: FemWorkerTaskStatus::Completed,
            observed_peak_bytes: Some(observed_peak_bytes),
            error: None,
        }
    }

    pub fn failed(
        request: &FemWorkerTaskRequest,
        error: impl Into<String>,
    ) -> Result<Self, FemRunStateError> {
        let error = error.into();
        if error.trim().is_empty() {
            return Err(FemRunStateError::InvalidJson(
                "failed worker task response requires an error".into(),
            ));
        }
        Ok(Self {
            schema: FEM_WORKER_TASK_PROTOCOL_SCHEMA_V1.into(),
            schedule_fingerprint: request.schedule_fingerprint.clone(),
            worker_index: request.worker_index,
            task_index: request.task_index,
            status: FemWorkerTaskStatus::Failed,
            observed_peak_bytes: None,
            error: Some(error),
        })
    }

    fn validate_fields(&self) -> Result<(), FemRunStateError> {
        if self.schema != FEM_WORKER_TASK_PROTOCOL_SCHEMA_V1 {
            return Err(FemRunStateError::InvalidJson(format!(
                "unsupported worker task response schema {:?}",
                self.schema
            )));
        }
        if self.schedule_fingerprint.len() != 64
            || !self
                .schedule_fingerprint
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(FemRunStateError::InvalidSchedule(
                "worker response schedule fingerprint is not canonical SHA-256".into(),
            ));
        }
        match self.status {
            FemWorkerTaskStatus::Completed => {
                if self.observed_peak_bytes.is_none() || self.error.is_some() {
                    return Err(FemRunStateError::InvalidJson(
                        "completed worker response has invalid payload".into(),
                    ));
                }
            }
            FemWorkerTaskStatus::Failed => {
                if self.observed_peak_bytes.is_some()
                    || !matches!(self.error.as_deref(), Some(error) if !error.trim().is_empty())
                {
                    return Err(FemRunStateError::InvalidJson(
                        "failed worker response has invalid payload".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn validate_for(&self, request: &FemWorkerTaskRequest) -> Result<(), FemRunStateError> {
        self.validate_fields()?;
        if self.schedule_fingerprint != request.schedule_fingerprint {
            return Err(FemRunStateError::ScheduleBindingMismatch {
                field: "fingerprint",
                expected: request.schedule_fingerprint.clone(),
                actual: self.schedule_fingerprint.clone(),
            });
        }
        if self.worker_index != request.worker_index || self.task_index != request.task_index {
            return Err(FemRunStateError::InvalidSchedule(
                "worker response identity does not match request".into(),
            ));
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<String, FemRunStateError> {
        self.validate_fields()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| FemRunStateError::InvalidJson(error.to_string()))
    }

    pub fn from_json(document: &str) -> Result<Self, FemRunStateError> {
        let response: Self = serde_json::from_str(document)
            .map_err(|error| FemRunStateError::InvalidJson(error.to_string()))?;
        response.validate_fields()?;
        Ok(response)
    }

    /// Encode this response as one bounded, length-prefixed transport frame.
    pub fn to_wire_frame_bytes(&self) -> Result<Vec<u8>, FemRunStateError> {
        self.validate_fields()?;
        encode_worker_task_frame(FemWorkerTaskFrame {
            schema: FEM_WORKER_TASK_PROTOCOL_SCHEMA_V1.into(),
            kind: FemWorkerTaskFrameKind::Response,
            payload: serde_json::to_value(self)
                .map_err(|error| FemRunStateError::InvalidJson(error.to_string()))?,
        })
    }

    /// Decode exactly one response frame, rejecting request/response confusion.
    pub fn from_wire_frame_bytes(bytes: &[u8]) -> Result<Self, FemRunStateError> {
        let frame = decode_worker_task_frame(bytes)?;
        validate_worker_task_frame(&frame, FemWorkerTaskFrameKind::Response)?;
        let response: Self = serde_json::from_value(frame.payload)
            .map_err(|error| FemRunStateError::InvalidJson(error.to_string()))?;
        response.validate_fields()?;
        Ok(response)
    }

    /// Write exactly one response frame to a process/rank stream.
    pub fn write_wire_frame<W: Write>(&self, writer: &mut W) -> Result<(), FemRunStateError> {
        writer
            .write_all(&self.to_wire_frame_bytes()?)
            .map_err(|error| {
                FemRunStateError::InvalidJson(format!("worker frame write failed: {error}"))
            })
    }

    /// Read exactly one response frame from a process/rank stream.
    pub fn read_wire_frame<R: Read>(reader: &mut R) -> Result<Self, FemRunStateError> {
        Self::from_wire_frame_bytes(&read_worker_task_frame(reader)?)
    }
}

/// Coordinator-side response reorder buffer for distributed worker execution.
///
/// Workers may finish in any order, but the durable checkpoint remains
/// lane-major and single-writer.  This boundary validates each response as it
/// arrives, buffers completed responses, and releases only the contiguous
/// prefix that the checkpoint can commit safely.
#[derive(Debug, Clone, PartialEq)]
pub struct FemWorkerTaskCoordinator {
    worker_count: usize,
    schedule_fingerprint: String,
    requests: BTreeMap<usize, FemWorkerTaskRequest>,
    buffered_responses: BTreeMap<usize, FemWorkerTaskResponse>,
    next_task_index: usize,
    failure: Option<String>,
}

impl FemWorkerTaskCoordinator {
    /// Bind a response reorder buffer to one deterministic worker partition.
    pub fn new(schedule: &SolveSchedule, worker_count: usize) -> Result<Self, FemRunStateError> {
        let tasks = ordered_tasks(schedule)?;
        partition_solve_schedule(schedule, worker_count)
            .map_err(|error| FemRunStateError::InvalidSchedule(error.to_string()))?;
        let requests = tasks
            .into_iter()
            .enumerate()
            .map(|(task_index, _)| {
                FemWorkerTaskRequest::from_schedule(schedule, worker_count, task_index)
                    .map(|request| (task_index, request))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        Ok(Self {
            worker_count,
            schedule_fingerprint: fem_schedule_fingerprint(schedule),
            requests,
            buffered_responses: BTreeMap::new(),
            next_task_index: 0,
            failure: None,
        })
    }

    /// Restore worker routing at a persisted checkpoint cursor.
    pub fn from_checkpoint(
        schedule: &SolveSchedule,
        worker_count: usize,
        checkpoint: &FemRunCheckpoint,
    ) -> Result<Self, FemRunStateError> {
        checkpoint.validate_for(schedule)?;
        if checkpoint.status == FemRunStatus::Completed {
            return Err(FemRunStateError::InvalidTransition {
                status: checkpoint.status,
                action: "restore worker coordinator",
            });
        }
        let mut coordinator = Self::new(schedule, worker_count)?;
        coordinator.next_task_index = checkpoint.next_task_index;
        Ok(coordinator)
    }

    /// Verify that this coordinator is still paired with the supplied schedule.
    pub fn validate_for(&self, schedule: &SolveSchedule) -> Result<(), FemRunStateError> {
        let expected = fem_schedule_fingerprint(schedule);
        if expected != self.schedule_fingerprint {
            return Err(FemRunStateError::ScheduleBindingMismatch {
                field: "fingerprint",
                expected,
                actual: self.schedule_fingerprint.clone(),
            });
        }
        let expected_task_count = ordered_tasks(schedule)?.len();
        if expected_task_count != self.requests.len() {
            return Err(FemRunStateError::InvalidSchedule(format!(
                "coordinator has {} requests for schedule with {} tasks",
                self.requests.len(),
                expected_task_count
            )));
        }
        Ok(())
    }

    pub fn worker_count(&self) -> usize {
        self.worker_count
    }

    pub fn next_task_index(&self) -> usize {
        self.next_task_index
    }

    pub fn buffered_response_count(&self) -> usize {
        self.buffered_responses.len()
    }

    pub fn is_complete(&self) -> bool {
        self.next_task_index == self.requests.len()
    }

    /// Return a copy of the task request used to route one task.
    pub fn request_for_task(
        &self,
        task_index: usize,
    ) -> Result<FemWorkerTaskRequest, FemRunStateError> {
        if task_index < self.next_task_index {
            return Err(FemRunStateError::CursorMismatch {
                expected: self.next_task_index,
                actual: task_index,
            });
        }
        self.requests.get(&task_index).cloned().ok_or_else(|| {
            FemRunStateError::InvalidSchedule(format!(
                "worker task index {task_index} outside {} scheduled tasks",
                self.requests.len()
            ))
        })
    }

    /// Return all requests owned by one worker in global lane-major order.
    pub fn requests_for_worker(
        &self,
        worker_index: usize,
    ) -> Result<Vec<FemWorkerTaskRequest>, FemRunStateError> {
        if worker_index >= self.worker_count {
            return Err(FemRunStateError::InvalidSchedule(format!(
                "worker index {worker_index} outside {} workers",
                self.worker_count
            )));
        }
        Ok(self
            .requests
            .values()
            .filter(|request| {
                request.worker_index == worker_index && request.task_index >= self.next_task_index
            })
            .cloned()
            .collect())
    }

    /// Write the currently uncommitted requests for one worker to a stream.
    pub fn write_requests_for_worker<W: Write>(
        &self,
        worker_index: usize,
        writer: &mut W,
    ) -> Result<usize, FemRunStateError> {
        let requests = self.requests_for_worker(worker_index)?;
        for request in &requests {
            request.write_wire_frame(writer)?;
        }
        Ok(requests.len())
    }

    /// Accept one response and release newly contiguous responses in commit order.
    pub fn accept_response(
        &mut self,
        response: FemWorkerTaskResponse,
    ) -> Result<Vec<(FemWorkerTaskRequest, FemWorkerTaskResponse)>, FemRunStateError> {
        if let Some(reason) = &self.failure {
            return Err(FemRunStateError::InvalidCheckpoint(format!(
                "worker coordinator is cancelled: {reason}"
            )));
        }
        let request = self.request_for_task(response.task_index)?;
        response.validate_for(&request)?;
        if response.task_index < self.next_task_index {
            return Err(FemRunStateError::CursorMismatch {
                expected: self.next_task_index,
                actual: response.task_index,
            });
        }
        if self.buffered_responses.contains_key(&response.task_index) {
            return Err(FemRunStateError::InvalidCheckpoint(format!(
                "duplicate worker response for task {}",
                response.task_index
            )));
        }
        self.buffered_responses
            .insert(response.task_index, response.clone());
        if response.status == FemWorkerTaskStatus::Failed {
            let reason = response
                .error
                .clone()
                .expect("validated failed worker response has an error");
            self.failure = Some(reason.clone());
            self.buffered_responses.clear();
            return Err(FemRunStateError::InvalidCheckpoint(format!(
                "worker task {} failed: {reason}",
                response.task_index
            )));
        }

        let mut ready = Vec::new();
        while let Some(response) = self.buffered_responses.remove(&self.next_task_index) {
            let request = self
                .requests
                .get(&self.next_task_index)
                .cloned()
                .expect("buffered response always has a bound request");
            ready.push((request, response));
            self.next_task_index += 1;
        }
        Ok(ready)
    }

    /// Decode and accept one response frame from a process/rank transport.
    pub fn accept_response_frame(
        &mut self,
        bytes: &[u8],
    ) -> Result<Vec<(FemWorkerTaskRequest, FemWorkerTaskResponse)>, FemRunStateError> {
        self.accept_response(FemWorkerTaskResponse::from_wire_frame_bytes(bytes)?)
    }

    /// Atomically reorder and commit one response frame to a checkpoint.
    ///
    /// Both state machines are staged so a memory-budget or checkpoint error
    /// cannot advance the coordinator cursor without advancing durable state.
    pub fn commit_response_frame(
        &mut self,
        schedule: &SolveSchedule,
        checkpoint: &mut FemRunCheckpoint,
        bytes: &[u8],
    ) -> Result<usize, FemRunStateError> {
        self.validate_for(schedule)?;
        checkpoint.validate_for(schedule)?;
        let mut staged_coordinator = self.clone();
        let ready = match staged_coordinator.accept_response_frame(bytes) {
            Ok(ready) => ready,
            Err(error) => {
                if staged_coordinator.failure.is_some() {
                    *self = staged_coordinator;
                    let _ = checkpoint.cancel(schedule, error.to_string());
                }
                return Err(error);
            }
        };
        let committed_count = ready.len();
        let mut staged_checkpoint = checkpoint.clone();
        for (request, response) in ready {
            if let Err(error) =
                staged_checkpoint.commit_worker_response(schedule, &request, &response)
            {
                *checkpoint = staged_checkpoint;
                self.failure = Some(error.to_string());
                return Err(error);
            }
        }
        *checkpoint = staged_checkpoint;
        *self = staged_coordinator;
        Ok(committed_count)
    }
}

/// Serve one worker's exact request stream and emit response frames.
///
/// The caller supplies the task-index set for this stream, which allows a
/// resumed coordinator to serve only its uncommitted suffix. Every received
/// request must belong to that set and to the selected worker; malformed,
/// duplicated, or misrouted requests fail before the callback is invoked.
pub fn serve_worker_task_stream<R, W, F>(
    schedule: &SolveSchedule,
    worker_count: usize,
    worker_index: usize,
    expected_task_indices: &[usize],
    reader: &mut R,
    writer: &mut W,
    mut execute: F,
) -> Result<usize, FemRunStateError>
where
    R: Read,
    W: Write,
    F: FnMut(SolveWorkItem) -> Result<usize, String>,
{
    let assignments = partition_solve_schedule(schedule, worker_count)
        .map_err(|error| FemRunStateError::InvalidSchedule(error.to_string()))?;
    if worker_index >= worker_count {
        return Err(FemRunStateError::InvalidSchedule(format!(
            "worker index {worker_index} outside {worker_count} workers"
        )));
    }
    let assignment = assignments
        .iter()
        .find(|assignment| assignment.worker_index == worker_index)
        .expect("partition creates one assignment per worker");
    let mut expected = BTreeMap::new();
    for &task_index in expected_task_indices {
        if !assignment.task_indices.contains(&task_index) {
            return Err(FemRunStateError::InvalidSchedule(format!(
                "task {task_index} is not assigned to worker {worker_index}"
            )));
        }
        if expected.insert(task_index, ()).is_some() {
            return Err(FemRunStateError::InvalidSchedule(format!(
                "duplicate expected task {task_index}"
            )));
        }
    }

    let expected_count = expected.len();
    for _ in 0..expected_count {
        let request = FemWorkerTaskRequest::read_wire_frame(reader)?;
        request.validate_for(schedule)?;
        if request.worker_index != worker_index {
            return Err(FemRunStateError::InvalidSchedule(format!(
                "worker request {} routed to worker {}, expected {}",
                request.task_index, request.worker_index, worker_index
            )));
        }
        if expected.remove(&request.task_index).is_none() {
            return Err(FemRunStateError::InvalidSchedule(format!(
                "unexpected or duplicate task {} for worker {}",
                request.task_index, worker_index
            )));
        }
        let response = match execute(request.task) {
            Ok(observed_peak_bytes) => {
                FemWorkerTaskResponse::completed(&request, observed_peak_bytes)
            }
            Err(error) => {
                let response = FemWorkerTaskResponse::failed(&request, error.clone())?;
                response.write_wire_frame(writer)?;
                return Err(FemRunStateError::InvalidCheckpoint(format!(
                    "worker task {} failed: {error}",
                    request.task_index
                )));
            }
        };
        response.write_wire_frame(writer)?;
    }
    debug_assert!(expected.is_empty());
    Ok(expected_count)
}

fn encode_worker_task_frame(frame: FemWorkerTaskFrame) -> Result<Vec<u8>, FemRunStateError> {
    if frame.schema != FEM_WORKER_TASK_PROTOCOL_SCHEMA_V1 {
        return Err(FemRunStateError::InvalidJson(format!(
            "unsupported worker frame schema {:?}",
            frame.schema
        )));
    }
    let payload = serde_json::to_vec(&frame)
        .map_err(|error| FemRunStateError::InvalidJson(error.to_string()))?;
    if payload.is_empty() || payload.len() > FEM_WORKER_TASK_MAX_FRAME_BYTES {
        return Err(FemRunStateError::InvalidJson(format!(
            "worker frame size {} outside 1..={}",
            payload.len(),
            FEM_WORKER_TASK_MAX_FRAME_BYTES
        )));
    }
    let length = u32::try_from(payload.len())
        .map_err(|_| FemRunStateError::InvalidJson("worker frame length exceeds u32".into()))?;
    let mut bytes = Vec::with_capacity(std::mem::size_of::<u32>() + payload.len());
    bytes.extend_from_slice(&length.to_be_bytes());
    bytes.extend_from_slice(&payload);
    Ok(bytes)
}

fn decode_worker_task_frame(bytes: &[u8]) -> Result<FemWorkerTaskFrame, FemRunStateError> {
    const HEADER_BYTES: usize = std::mem::size_of::<u32>();
    if bytes.len() < HEADER_BYTES {
        return Err(FemRunStateError::InvalidJson(
            "truncated worker frame length prefix".into(),
        ));
    }
    let length = u32::from_be_bytes(
        bytes[..HEADER_BYTES]
            .try_into()
            .expect("checked worker frame header length"),
    ) as usize;
    if length == 0 || length > FEM_WORKER_TASK_MAX_FRAME_BYTES {
        return Err(FemRunStateError::InvalidJson(format!(
            "worker frame length {} outside 1..={}",
            length, FEM_WORKER_TASK_MAX_FRAME_BYTES
        )));
    }
    let expected_length = HEADER_BYTES + length;
    if bytes.len() != expected_length {
        return Err(FemRunStateError::InvalidJson(format!(
            "worker frame length prefix declares {}, received {} payload bytes",
            length,
            bytes.len().saturating_sub(HEADER_BYTES)
        )));
    }
    serde_json::from_slice(&bytes[HEADER_BYTES..])
        .map_err(|error| FemRunStateError::InvalidJson(error.to_string()))
}

fn read_worker_task_frame<R: Read>(reader: &mut R) -> Result<Vec<u8>, FemRunStateError> {
    const HEADER_BYTES: usize = std::mem::size_of::<u32>();
    let mut header = [0_u8; HEADER_BYTES];
    reader.read_exact(&mut header).map_err(|error| {
        FemRunStateError::InvalidJson(format!("worker frame header read failed: {error}"))
    })?;
    let length = u32::from_be_bytes(header) as usize;
    if length == 0 || length > FEM_WORKER_TASK_MAX_FRAME_BYTES {
        return Err(FemRunStateError::InvalidJson(format!(
            "worker frame length {} outside 1..={}",
            length, FEM_WORKER_TASK_MAX_FRAME_BYTES
        )));
    }
    let mut bytes = Vec::with_capacity(HEADER_BYTES + length);
    bytes.extend_from_slice(&header);
    let mut payload = vec![0_u8; length];
    reader.read_exact(&mut payload).map_err(|error| {
        FemRunStateError::InvalidJson(format!("worker frame payload read failed: {error}"))
    })?;
    bytes.extend_from_slice(&payload);
    Ok(bytes)
}

fn validate_worker_task_frame(
    frame: &FemWorkerTaskFrame,
    expected_kind: FemWorkerTaskFrameKind,
) -> Result<(), FemRunStateError> {
    if frame.schema != FEM_WORKER_TASK_PROTOCOL_SCHEMA_V1 {
        return Err(FemRunStateError::InvalidJson(format!(
            "unsupported worker frame schema {:?}",
            frame.schema
        )));
    }
    if frame.kind != expected_kind {
        return Err(FemRunStateError::InvalidJson(format!(
            "worker frame kind {:?} does not match expected {:?}",
            frame.kind, expected_kind
        )));
    }
    Ok(())
}

pub fn fem_schedule_fingerprint(schedule: &SolveSchedule) -> String {
    let mut hasher = Sha256::new();
    update_usize(&mut hasher, schedule.task_count);
    update_usize(&mut hasher, schedule.peak_concurrent_bytes);
    update_usize(&mut hasher, schedule.lanes.len());
    for lane in &schedule.lanes {
        update_usize(&mut hasher, lane.lane_index);
        update_usize(&mut hasher, lane.peak_bytes);
        update_usize(&mut hasher, lane.candidate_indices.len());
        for &candidate in &lane.candidate_indices {
            update_usize(&mut hasher, candidate);
        }
        update_usize(&mut hasher, lane.tasks.len());
        for task in &lane.tasks {
            update_usize(&mut hasher, task.candidate_index);
            update_usize(&mut hasher, task.frequency_index);
            update_usize(&mut hasher, task.source_start);
            update_usize(&mut hasher, task.source_count);
            update_usize(&mut hasher, task.estimated_bytes);
        }
    }
    let digest = hasher.finalize();
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn ordered_tasks(schedule: &SolveSchedule) -> Result<Vec<SolveWorkItem>, FemRunStateError> {
    let mut tasks = Vec::new();
    for (lane_index, lane) in schedule.lanes.iter().enumerate() {
        if lane.lane_index != lane_index {
            return Err(FemRunStateError::InvalidSchedule(
                "lane indices must be contiguous".into(),
            ));
        }
        tasks.extend(lane.tasks.iter().copied());
    }
    if tasks.len() != schedule.task_count {
        return Err(FemRunStateError::InvalidSchedule(format!(
            "schedule declares {} tasks but contains {}",
            schedule.task_count,
            tasks.len()
        )));
    }
    Ok(tasks)
}

fn check_binding<T>(field: &'static str, expected: T, actual: T) -> Result<(), FemRunStateError>
where
    T: fmt::Display + PartialEq,
{
    if expected != actual {
        return Err(FemRunStateError::ScheduleBindingMismatch {
            field,
            expected: expected.to_string(),
            actual: actual.to_string(),
        });
    }
    Ok(())
}

fn update_usize(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fem_runner::scheduler::{ResourceBudget, WorkloadShape, plan_solve_schedule};
    use std::sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    };

    fn make_schedule() -> (SolveSchedule, ResourceBudget) {
        let shape = WorkloadShape {
            candidate_count: 2,
            frequency_count: 2,
            source_count: 3,
            rhs_block_size: 2,
        };
        let budget = ResourceBudget {
            lane_limit: 2,
            memory_budget_bytes: 1_000,
            shared_bytes: 100,
            per_lane_bytes: 100,
            per_frequency_bytes: 20,
            per_rhs_bytes: 30,
            per_output_bytes: 10,
        };
        (plan_solve_schedule(shape, budget).unwrap(), budget)
    }

    fn task_at(schedule: &SolveSchedule, index: usize) -> SolveWorkItem {
        schedule
            .lanes
            .iter()
            .flat_map(|lane| lane.tasks.iter().copied())
            .nth(index)
            .unwrap()
    }

    #[test]
    fn checkpoint_commits_in_order_and_round_trips_json() {
        let (schedule, budget) = make_schedule();
        let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
        checkpoint.start(&schedule).unwrap();
        for index in 0..schedule.task_count {
            checkpoint
                .commit_next_task(&schedule, index, task_at(&schedule, index), 800)
                .unwrap();
        }
        checkpoint.complete(&schedule).unwrap();
        let restored = FemRunCheckpoint::from_json(&checkpoint.to_json().unwrap()).unwrap();
        assert_eq!(restored, checkpoint);
        assert_eq!(restored.status, FemRunStatus::Completed);
    }

    #[test]
    fn restart_and_schedule_tampering_fail_closed() {
        let (schedule, budget) = make_schedule();
        let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
        checkpoint.start(&schedule).unwrap();
        checkpoint
            .commit_next_task(&schedule, 0, task_at(&schedule, 0), 500)
            .unwrap();
        assert!(matches!(
            checkpoint.resume(&schedule, 0),
            Err(FemRunStateError::CursorMismatch {
                expected: 1,
                actual: 0
            })
        ));

        let mut changed = schedule.clone();
        changed.lanes[0].tasks[0].source_count += 1;
        assert!(matches!(
            checkpoint.validate_for(&changed),
            Err(FemRunStateError::ScheduleBindingMismatch {
                field: "fingerprint",
                ..
            })
        ));
    }

    #[test]
    fn out_of_order_and_observed_memory_are_rejected() {
        let (schedule, budget) = make_schedule();
        let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
        checkpoint.start(&schedule).unwrap();
        assert!(matches!(
            checkpoint.commit_next_task(&schedule, 1, task_at(&schedule, 1), 500),
            Err(FemRunStateError::CursorMismatch {
                expected: 0,
                actual: 1
            })
        ));
        assert!(matches!(
            checkpoint.commit_next_task(&schedule, 0, task_at(&schedule, 0), 1_001),
            Err(FemRunStateError::ObservedPeakExceedsBudget {
                observed: 1_001,
                budget: 1_000
            })
        ));
        assert_eq!(checkpoint.status, FemRunStatus::Cancelled);
        assert!(matches!(
            checkpoint.complete(&schedule),
            Err(FemRunStateError::InvalidTransition {
                status: FemRunStatus::Cancelled,
                action: "complete"
            })
        ));
    }

    #[test]
    fn cancellation_is_cloneable_and_first_error_wins() {
        let cancellation = FemRunCancellation::new();
        let worker = cancellation.clone();
        worker.cancel("worker failed");
        cancellation.cancel("later failure");
        assert!(cancellation.is_cancelled());
        assert_eq!(cancellation.first_error().as_deref(), Some("worker failed"));

        let (schedule, budget) = make_schedule();
        let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
        checkpoint.start(&schedule).unwrap();
        assert!(
            checkpoint
                .poll_cancellation(&schedule, &cancellation)
                .unwrap()
        );
        assert_eq!(checkpoint.status, FemRunStatus::Cancelled);
        assert_eq!(
            checkpoint.cancellation_reason.as_deref(),
            Some("worker failed")
        );
    }

    #[test]
    fn strict_json_rejects_unknown_fields_and_invalid_state() {
        let (schedule, budget) = make_schedule();
        let checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
        let json = checkpoint.to_json().unwrap();
        let with_unknown = format!("{}\n  ,\"unknown\":true}}", json.trim_end_matches('}'));
        assert!(matches!(
            FemRunCheckpoint::from_json(&with_unknown),
            Err(FemRunStateError::InvalidJson(_))
        ));

        let mut invalid = checkpoint;
        invalid.next_task_index = 1;
        invalid.committed_task_count = 0;
        assert!(matches!(
            invalid.to_json(),
            Err(FemRunStateError::InvalidCheckpoint(_))
        ));
    }

    #[test]
    fn checkpoint_file_round_trip_and_corruption_are_checked() {
        let (schedule, budget) = make_schedule();
        let checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("fem-run.json");
        checkpoint.write_json(&path).unwrap();
        assert_eq!(FemRunCheckpoint::read_json(&path).unwrap(), checkpoint);

        std::fs::write(&path, "{\"schema\":\"unknown\"}").unwrap();
        assert!(matches!(
            FemRunCheckpoint::read_json(&path),
            Err(FemRunStateError::InvalidJson(_)) | Err(FemRunStateError::InvalidCheckpoint(_))
        ));
    }

    #[test]
    fn executor_dispatches_in_order_and_resumes_persisted_work() {
        let (schedule, budget) = make_schedule();
        let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
        checkpoint.start(&schedule).unwrap();
        checkpoint
            .commit_next_task(&schedule, 0, task_at(&schedule, 0), 400)
            .unwrap();
        let cancellation = FemRunCancellation::new();
        let mut seen = Vec::new();
        let report = execute_scheduled_fem_run(&schedule, &mut checkpoint, &cancellation, |task| {
            seen.push((
                task.candidate_index,
                task.frequency_index,
                task.source_start,
            ));
            Ok(700)
        })
        .unwrap();
        assert_eq!(seen.len(), schedule.task_count - 1);
        assert_eq!(report.status, FemRunStatus::Completed);
        assert_eq!(report.committed_task_count, schedule.task_count);
        assert_eq!(report.observed_peak_bytes, 700);
    }

    #[test]
    fn executor_durably_cancels_on_callback_failure_or_request() {
        let (schedule, budget) = make_schedule();
        let cancellation = FemRunCancellation::new();
        let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
        let mut calls = 0;
        let error = execute_scheduled_fem_run(&schedule, &mut checkpoint, &cancellation, |_task| {
            calls += 1;
            if calls == 2 {
                Err("solver callback failed".into())
            } else {
                Ok(700)
            }
        })
        .unwrap_err();
        assert!(matches!(
            error,
            FemRunExecutionError::Task { task_index: 1, .. }
        ));
        assert_eq!(checkpoint.status, FemRunStatus::Cancelled);
        assert_eq!(
            checkpoint.cancellation_reason.as_deref(),
            Some("solver callback failed")
        );

        let (schedule, budget) = make_schedule();
        let cancellation = FemRunCancellation::new();
        cancellation.cancel("user stopped run");
        let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
        let error = execute_scheduled_fem_run(&schedule, &mut checkpoint, &cancellation, |_task| {
            panic!("cancelled run must not invoke callback")
        })
        .unwrap_err();
        assert!(matches!(
            error,
            FemRunExecutionError::Cancelled {
                reason
            } if reason == "user stopped run"
        ));
        assert_eq!(checkpoint.status, FemRunStatus::Cancelled);
    }

    #[test]
    fn concurrent_executor_runs_lanes_and_commits_in_schedule_order() {
        let (schedule, budget) = make_schedule();
        let barrier = Arc::new(Barrier::new(2));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
        let cancellation = FemRunCancellation::new();
        let report =
            execute_scheduled_fem_run_concurrent(&schedule, &mut checkpoint, &cancellation, {
                let barrier = barrier.clone();
                let calls = calls.clone();
                move |task| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    if task.frequency_index == 0 && task.source_start == 0 {
                        barrier.wait();
                    }
                    Ok(700)
                }
            })
            .unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), schedule.task_count);
        assert_eq!(report.status, FemRunStatus::Completed);
        assert_eq!(report.committed_task_count, schedule.task_count);
        assert_eq!(checkpoint.next_task_index, schedule.task_count);
    }

    #[test]
    fn concurrent_commit_cancellation_does_not_drain_buffered_completions() {
        let (schedule, budget) = make_schedule();
        let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
        let cancellation = FemRunCancellation::new();
        let callback_calls = Arc::new(AtomicUsize::new(0));
        let callback_cancellation = cancellation.clone();
        let callback_calls_for_worker = callback_calls.clone();

        let error = execute_scheduled_fem_run_concurrent_with_commit(
            &schedule,
            &mut checkpoint,
            &cancellation,
            |_task| Ok(700),
            move |_checkpoint, _task| {
                callback_calls_for_worker.fetch_add(1, Ordering::SeqCst);
                callback_cancellation.cancel("stop after first committed task");
                Ok(())
            },
        )
        .unwrap_err();

        assert!(matches!(
            error,
            FemRunExecutionError::Cancelled { ref reason }
                if reason == "stop after first committed task"
        ));
        assert_eq!(callback_calls.load(Ordering::SeqCst), 1);
        assert_eq!(checkpoint.next_task_index, 1);
        assert_eq!(checkpoint.status, FemRunStatus::Cancelled);
    }

    #[test]
    fn concurrent_executor_cancels_all_lanes_after_first_callback_error() {
        let (schedule, budget) = make_schedule();
        let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
        let cancellation = FemRunCancellation::new();
        let error = execute_scheduled_fem_run_concurrent(
            &schedule,
            &mut checkpoint,
            &cancellation,
            |task| {
                if task.candidate_index == 1 && task.frequency_index == 0 {
                    Err("lane callback failed".into())
                } else {
                    Ok(700)
                }
            },
        )
        .unwrap_err();

        assert!(matches!(
            error,
            FemRunExecutionError::Task { .. } | FemRunExecutionError::Cancelled { .. }
        ));
        assert_eq!(checkpoint.status, FemRunStatus::Cancelled);
        assert_eq!(
            cancellation.first_error().as_deref(),
            Some("lane callback failed")
        );
    }
}
