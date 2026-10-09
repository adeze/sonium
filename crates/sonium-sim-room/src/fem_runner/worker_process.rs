//! Process-level worker lifecycle for the checked FEM task protocol.
//!
//! The coordinator owns checkpoint ordering; this module owns the operating
//! system process boundary. A worker receives the same bounded request frames
//! used by the in-memory stream adapter and returns one response frame per
//! request. Broken pipes, truncated responses, and non-zero exits are surfaced
//! as errors, and dropping an unfinished process terminates it so a failed lane
//! cannot leak a child process.

use super::{
    FemRunCheckpoint, FemRunExecutionError, FemRunExecutionReport, FemRunStateError, FemRunStatus,
    FemWorkerTaskCoordinator, FemWorkerTaskResponse, SolveSchedule,
};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io::{BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

/// Default bound for one worker exchange or clean worker shutdown.
pub const FEM_WORKER_DEFAULT_IO_TIMEOUT: Duration = Duration::from_secs(30);

/// Command and environment used to launch one FEM worker rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FemWorkerProcessSpec {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub environment: Vec<(OsString, OsString)>,
    pub io_timeout: Duration,
}

impl FemWorkerProcessSpec {
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            environment: Vec::new(),
            io_timeout: FEM_WORKER_DEFAULT_IO_TIMEOUT,
        }
    }

    pub fn arg(mut self, argument: impl Into<OsString>) -> Self {
        self.args.push(argument.into());
        self
    }

    pub fn env(mut self, key: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.environment.push((key.into(), value.into()));
        self
    }

    /// Bound one exchange and clean worker shutdown.
    pub fn io_timeout(mut self, timeout: Duration) -> Self {
        self.io_timeout = timeout;
        self
    }

    fn validate(&self) -> Result<(), FemRunStateError> {
        if self.program.as_os_str().is_empty() {
            return Err(FemRunStateError::Io(
                "FEM worker process program must not be empty".into(),
            ));
        }
        if self
            .environment
            .iter()
            .any(|(key, _)| key.is_empty() || key.to_string_lossy().contains('='))
        {
            return Err(FemRunStateError::Io(
                "FEM worker process environment contains an invalid key".into(),
            ));
        }
        if self.io_timeout.is_zero() {
            return Err(FemRunStateError::Io(
                "FEM worker process I/O timeout must be non-zero".into(),
            ));
        }
        Ok(())
    }
}

/// One running worker process attached to the checked task stream.
pub struct FemWorkerProcess {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Option<BufReader<ChildStdout>>,
    worker_index: usize,
    io_timeout: Duration,
    finished: bool,
}

impl std::fmt::Debug for FemWorkerProcess {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FemWorkerProcess")
            .field("worker_index", &self.worker_index)
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

impl FemWorkerProcess {
    /// Spawn a worker with piped request/response streams.
    pub fn spawn(
        spec: &FemWorkerProcessSpec,
        worker_index: usize,
        worker_count: usize,
    ) -> Result<Self, FemRunStateError> {
        spec.validate()?;
        if worker_count == 0 || worker_index >= worker_count {
            return Err(FemRunStateError::InvalidSchedule(format!(
                "worker index {worker_index} outside {worker_count} workers"
            )));
        }

        let mut command = Command::new(&spec.program);
        command
            .args(&spec.args)
            .envs(spec.environment.iter().map(|(key, value)| (key, value)))
            .env("ROOMSIM_WORKER_INDEX", worker_index.to_string())
            .env("ROOMSIM_WORKER_COUNT", worker_count.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = command.spawn().map_err(|error| {
            FemRunStateError::Io(format!(
                "failed to spawn FEM worker {worker_index} {:?}: {error}",
                spec.program
            ))
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| FemRunStateError::Io("FEM worker stdin was not piped".into()));
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| FemRunStateError::Io("FEM worker stdout was not piped".into()));
        let (stdin, stdout) = match (stdin, stdout) {
            (Ok(stdin), Ok(stdout)) => (stdin, stdout),
            (Err(error), _) | (_, Err(error)) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
        };
        Ok(Self {
            child,
            stdin: Some(stdin),
            stdout: Some(BufReader::new(stdout)),
            worker_index,
            io_timeout: spec.io_timeout,
            finished: false,
        })
    }

    /// Send the worker's uncommitted requests and read exactly one response
    /// for each request. The caller remains responsible for passing responses
    /// through the coordinator and checkpoint commit boundary.
    pub fn exchange(
        &mut self,
        coordinator: &FemWorkerTaskCoordinator,
    ) -> Result<Vec<FemWorkerTaskResponse>, FemRunStateError> {
        let requests = coordinator.requests_for_worker(self.worker_index)?;
        let expected_requests = requests
            .into_iter()
            .map(|request| (request.task_index, request))
            .collect::<BTreeMap<_, _>>();
        {
            let stdin = self
                .stdin
                .as_mut()
                .ok_or_else(|| FemRunStateError::Io("FEM worker stdin is already closed".into()))?;
            coordinator.write_requests_for_worker(self.worker_index, stdin)?;
            stdin.flush().map_err(|error| {
                FemRunStateError::Io(format!("failed to flush FEM worker requests: {error}"))
            })?;
        }

        let stdout = self
            .stdout
            .take()
            .ok_or_else(|| FemRunStateError::Io("FEM worker stdout is already closed".into()))?;
        let worker_index = self.worker_index;
        let io_timeout = self.io_timeout;
        let (sender, receiver) = mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let mut stdout = stdout;
            let result = (|| {
                let mut expected_requests = expected_requests;
                let mut responses = Vec::with_capacity(expected_requests.len());
                for _ in 0..expected_requests.len() {
                    let response = FemWorkerTaskResponse::read_wire_frame(&mut stdout)?;
                    let request = expected_requests
                        .remove(&response.task_index)
                        .ok_or_else(|| {
                            FemRunStateError::InvalidCheckpoint(format!(
                                "FEM worker {worker_index} returned duplicate or unexpected task {}",
                                response.task_index
                            ))
                        })?;
                    response.validate_for(&request)?;
                    responses.push(response);
                }
                if !expected_requests.is_empty() {
                    return Err(FemRunStateError::InvalidCheckpoint(format!(
                        "FEM worker {worker_index} omitted {} response(s)",
                        expected_requests.len()
                    )));
                }
                Ok(responses)
            })();
            let _ = sender.send((result, stdout));
        });

        match receiver.recv_timeout(io_timeout) {
            Ok((result, stdout)) => {
                self.stdout = Some(stdout);
                result
            }
            Err(RecvTimeoutError::Timeout) => {
                let _ = self.terminate();
                Err(FemRunStateError::Io(format!(
                    "FEM worker {worker_index} exchange timed out after {io_timeout:?}"
                )))
            }
            Err(RecvTimeoutError::Disconnected) => Err(FemRunStateError::Io(
                "FEM worker response reader terminated unexpectedly".into(),
            )),
        }
    }

    /// Close the request stream and require a successful worker exit.
    pub fn finish(mut self) -> Result<ExitStatus, FemRunStateError> {
        self.close_stdin();
        let status = self.wait_for_exit()?;
        if !status.success() {
            return Err(FemRunStateError::Io(format!(
                "FEM worker {} exited unsuccessfully: {status}",
                self.worker_index
            )));
        }
        Ok(status)
    }

    /// Terminate a worker after a transport, callback, or checkpoint failure.
    pub fn terminate(&mut self) -> Result<ExitStatus, FemRunStateError> {
        self.close_stdin();
        if let Some(status) = self.child.try_wait().map_err(|error| {
            FemRunStateError::Io(format!(
                "failed checking FEM worker {} before termination: {error}",
                self.worker_index
            ))
        })? {
            self.finished = true;
            return Ok(status);
        }
        self.child.kill().map_err(|error| {
            FemRunStateError::Io(format!(
                "failed to terminate FEM worker {}: {error}",
                self.worker_index
            ))
        })?;
        self.wait_for_exit()
    }

    fn close_stdin(&mut self) {
        self.stdin.take();
    }

    fn wait_for_exit(&mut self) -> Result<ExitStatus, FemRunStateError> {
        let deadline = Instant::now() + self.io_timeout;
        loop {
            if let Some(status) = self.child.try_wait().map_err(|error| {
                FemRunStateError::Io(format!(
                    "failed waiting for FEM worker {}: {error}",
                    self.worker_index
                ))
            })? {
                self.finished = true;
                return Ok(status);
            }
            if Instant::now() >= deadline {
                let _ = self.child.kill();
                let _ = self.child.wait();
                self.finished = true;
                return Err(FemRunStateError::Io(format!(
                    "FEM worker {} exit timed out after {:?}",
                    self.worker_index, self.io_timeout
                )));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

impl Drop for FemWorkerProcess {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// Supervisor for a fixed worker-rank set.
///
/// The supervisor deliberately does not mutate checkpoint state. It returns
/// responses to the coordinator and terminates every rank when one process
/// loses its transport, leaving checkpoint rollback/restart decisions with
/// the coordinator's persisted cursor.
pub struct FemWorkerProcessGroup {
    workers: Vec<Option<FemWorkerProcess>>,
}

impl std::fmt::Debug for FemWorkerProcessGroup {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FemWorkerProcessGroup")
            .field("worker_count", &self.workers.len())
            .field(
                "running_workers",
                &self
                    .workers
                    .iter()
                    .filter(|worker| worker.is_some())
                    .count(),
            )
            .finish()
    }
}

impl FemWorkerProcessGroup {
    /// Spawn one process for every rank in deterministic index order.
    pub fn spawn(specs: Vec<FemWorkerProcessSpec>) -> Result<Self, FemRunStateError> {
        if specs.is_empty() {
            return Err(FemRunStateError::InvalidSchedule(
                "FEM worker process group must not be empty".into(),
            ));
        }
        let worker_count = specs.len();
        let mut workers = Vec::with_capacity(worker_count);
        for (worker_index, spec) in specs.iter().enumerate() {
            workers.push(Some(FemWorkerProcess::spawn(
                spec,
                worker_index,
                worker_count,
            )?));
        }
        Ok(Self { workers })
    }

    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Exchange all uncommitted requests, terminating every rank on a
    /// transport failure in any one rank.
    pub fn exchange_all(
        &mut self,
        coordinator: &FemWorkerTaskCoordinator,
    ) -> Result<Vec<FemWorkerTaskResponse>, FemRunStateError> {
        let mut responses = Vec::new();
        for worker in &mut self.workers {
            let worker = worker.as_mut().ok_or_else(|| {
                FemRunStateError::InvalidCheckpoint("FEM worker rank is not running".into())
            })?;
            match worker.exchange(coordinator) {
                Ok(mut worker_responses) => responses.append(&mut worker_responses),
                Err(error) => {
                    let _ = self.terminate_all();
                    return Err(error);
                }
            }
        }
        Ok(responses)
    }

    /// Execute the currently pending checkpoint through the worker ranks.
    ///
    /// Responses are committed through the coordinator's transactional frame
    /// path, so out-of-order rank completion cannot advance the persisted
    /// cursor past a missing predecessor. A transport failure durably marks
    /// the checkpoint cancelled before the error is returned.
    pub fn execute_checkpointed(
        &mut self,
        schedule: &SolveSchedule,
        checkpoint: &mut FemRunCheckpoint,
    ) -> Result<FemRunExecutionReport, FemRunExecutionError> {
        if checkpoint.status == FemRunStatus::Pending {
            checkpoint
                .start(schedule)
                .map_err(FemRunExecutionError::State)?;
        } else if checkpoint.status != FemRunStatus::Running {
            return Err(FemRunExecutionError::State(
                FemRunStateError::InvalidTransition {
                    status: checkpoint.status,
                    action: "execute process worker group",
                },
            ));
        }
        let mut coordinator =
            FemWorkerTaskCoordinator::from_checkpoint(schedule, self.worker_count(), checkpoint)
                .map_err(FemRunExecutionError::State)?;
        let responses = match self.exchange_all(&coordinator) {
            Ok(responses) => responses,
            Err(error) => {
                let reason = error.to_string();
                let _ = checkpoint.cancel(schedule, reason);
                return Err(FemRunExecutionError::State(error));
            }
        };
        for response in responses {
            let frame = response
                .to_wire_frame_bytes()
                .map_err(FemRunExecutionError::State)?;
            coordinator
                .commit_response_frame(schedule, checkpoint, &frame)
                .map_err(FemRunExecutionError::State)?;
        }
        checkpoint
            .complete(schedule)
            .map_err(FemRunExecutionError::State)?;
        Ok(FemRunExecutionReport {
            status: checkpoint.status,
            committed_task_count: checkpoint.committed_task_count,
            observed_peak_bytes: checkpoint.observed_peak_bytes,
        })
    }

    /// Execute a checkpointed worker run, restarting the fixed rank set after
    /// a recoverable worker failure and resuming from the durable cursor.
    ///
    /// A failed exchange or response cancels the checkpoint before returning
    /// from [`Self::execute_checkpointed`]. This wrapper bounds recovery with
    /// `max_restarts`, terminates/replaces every rank before the next epoch,
    /// and reopens the cancelled checkpoint at its committed cursor. Restart
    /// epochs receive `ROOMSIM_WORKER_RESTART_ATTEMPT` in their environment so
    /// worker launchers can distinguish a fresh process from a retry without
    /// changing the wire protocol.
    pub fn execute_checkpointed_with_restart(
        &mut self,
        schedule: &SolveSchedule,
        checkpoint: &mut FemRunCheckpoint,
        specs: &[FemWorkerProcessSpec],
        max_restarts: usize,
    ) -> Result<FemRunExecutionReport, FemRunExecutionError> {
        if specs.len() != self.workers.len() {
            return Err(FemRunExecutionError::State(
                FemRunStateError::InvalidSchedule(format!(
                    "worker restart spec count {} does not match rank count {}",
                    specs.len(),
                    self.workers.len()
                )),
            ));
        }

        let mut restart_count = 0;
        loop {
            match self.execute_checkpointed(schedule, checkpoint) {
                Ok(report) => return Ok(report),
                Err(error) => {
                    if checkpoint.status == FemRunStatus::Running {
                        checkpoint
                            .cancel(schedule, error.to_string())
                            .map_err(FemRunExecutionError::State)?;
                    }
                    if checkpoint.status != FemRunStatus::Cancelled || restart_count >= max_restarts
                    {
                        return Err(error);
                    }

                    restart_count += 1;
                    self.restart_all_for_attempt(specs, restart_count)
                        .map_err(FemRunExecutionError::State)?;
                    let cursor = checkpoint.next_task_index;
                    checkpoint
                        .resume(schedule, cursor)
                        .map_err(FemRunExecutionError::State)?;
                }
            }
        }
    }

    fn restart_all_for_attempt(
        &mut self,
        specs: &[FemWorkerProcessSpec],
        restart_attempt: usize,
    ) -> Result<(), FemRunStateError> {
        for (worker_index, spec) in specs.iter().enumerate() {
            let mut retry_spec = spec.clone();
            retry_spec
                .environment
                .retain(|(key, _)| key != &OsString::from("ROOMSIM_WORKER_RESTART_ATTEMPT"));
            retry_spec.environment.push((
                OsString::from("ROOMSIM_WORKER_RESTART_ATTEMPT"),
                OsString::from(restart_attempt.to_string()),
            ));
            self.restart_worker(worker_index, &retry_spec)?;
        }
        Ok(())
    }

    /// Replace one failed or completed rank before the next dispatch epoch.
    pub fn restart_worker(
        &mut self,
        worker_index: usize,
        spec: &FemWorkerProcessSpec,
    ) -> Result<(), FemRunStateError> {
        let worker_count = self.workers.len();
        let slot = self.workers.get_mut(worker_index).ok_or_else(|| {
            FemRunStateError::InvalidSchedule(format!(
                "worker index {worker_index} outside {} workers",
                worker_count
            ))
        })?;
        if let Some(mut worker) = slot.take() {
            let _ = worker.terminate();
        }
        *slot = Some(FemWorkerProcess::spawn(spec, worker_index, worker_count)?);
        Ok(())
    }

    /// Terminate all ranks, returning the first termination error if any.
    pub fn terminate_all(&mut self) -> Result<(), FemRunStateError> {
        let mut first_error = None;
        for worker in &mut self.workers {
            if let Some(mut worker) = worker.take()
                && let Err(error) = worker.terminate()
            {
                first_error.get_or_insert(error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Close every rank and require all ranks to exit successfully.
    pub fn finish_all(mut self) -> Result<(), FemRunStateError> {
        let mut first_error = None;
        for worker in &mut self.workers {
            if let Some(worker) = worker.take()
                && let Err(error) = worker.finish()
            {
                first_error.get_or_insert(error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}
