//! Bounded subprocess capture for independent reference tools.
//!
//! Both pipes are drained concurrently, each with a byte cap. Deadlines include
//! waiting for EOF, so a launcher exiting while a descendant retains a pipe
//! cannot make the test wait forever. Unix runs use an owned process group for
//! cleanup; other targets kill the direct child (not a general process tree).

use std::{
    io::{self, Read},
    process::{Child, Command, Output, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

struct ChildGuard(Child, bool);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if !self.1 {
            return;
        }
        #[cfg(unix)]
        {
            // SAFETY: this child was spawned into a fresh group whose ID is
            // its PID. The signal is restricted to that owned process group.
            unsafe {
                libc::kill(-(self.0.id() as libc::pid_t), libc::SIGKILL);
            }
        }
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Capture a command with a wall-clock deadline and a per-stream byte limit.
/// Nonzero exit status is retained in Output; launch, timeout and capture
/// failures are errors, never successful numerical results.
pub fn bounded_output(
    command: &mut Command,
    timeout: Duration,
    max_stream_bytes: usize,
) -> io::Result<Output> {
    let limit = max_stream_bytes
        .checked_add(1)
        .filter(|_| max_stream_bytes > 0)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid subprocess output limit",
            )
        })?;
    let deadline = Instant::now()
        .checked_add(timeout)
        .filter(|_| !timeout.is_zero())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid subprocess timeout"))?;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = ChildGuard(command.spawn()?, true);
    let stdout = child.0.stdout.take().expect("piped stdout");
    let stderr = child.0.stderr.take().expect("piped stderr");
    let (tx, rx) = mpsc::channel();
    fn reader(
        stream: impl Read + Send + 'static,
        is_stdout: bool,
        limit: usize,
        tx: mpsc::Sender<(bool, io::Result<Vec<u8>>)>,
    ) -> io::Result<()> {
        thread::Builder::new()
            .name("qa-subprocess-capture".into())
            .spawn(move || {
                let mut bytes = Vec::new();
                let result = stream
                    .take(limit as u64)
                    .read_to_end(&mut bytes)
                    .and_then(|_| {
                        if bytes.len() == limit {
                            Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "subprocess output limit exceeded",
                            ))
                        } else {
                            Ok(bytes)
                        }
                    });
                let _ = tx.send((is_stdout, result));
            })?;
        Ok(())
    }
    reader(stdout, true, limit, tx.clone())?;
    reader(stderr, false, limit, tx)?;
    let mut captured_stdout = None;
    let mut captured_stderr = None;
    let mut status = None;
    loop {
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "reference subprocess timed out",
            ));
        }
        if status.is_none() {
            status = child.0.try_wait()?;
        }
        if let (Some(status), Some(stdout), Some(stderr)) =
            (status, captured_stdout.as_mut(), captured_stderr.as_mut())
        {
            let output = Output {
                status,
                stdout: std::mem::take(stdout),
                stderr: std::mem::take(stderr),
            };
            // The direct child has been reaped and both streams reached EOF.
            // Do not signal a finished process group on the successful path.
            child.1 = false;
            return Ok(output);
        }
        match rx.recv_timeout(
            deadline
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(10)),
        ) {
            Ok((is_stdout, result)) => {
                let bytes = result?;
                if is_stdout {
                    captured_stdout = Some(bytes);
                } else {
                    captured_stderr = Some(bytes);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                if captured_stdout.is_none() || captured_stderr.is_none() {
                    return Err(io::Error::other(
                        "reference subprocess capture ended without output",
                    ));
                }
                // Both readers finished before the process exited.
                thread::sleep(Duration::from_millis(2));
            }
        }
    }
}

/// Extension for live Wolfram reference tests. Default deadline: 15 minutes;
/// default cap: 16 MiB per stream. Explicit overrides must be positive integers.
pub trait WolframCommandExt {
    fn wolfram_output(&mut self) -> io::Result<Output>;
}

impl WolframCommandExt for Command {
    fn wolfram_output(&mut self) -> io::Result<Output> {
        fn setting(name: &str, default: u64) -> io::Result<u64> {
            match std::env::var(name) {
                Ok(value) => value.parse::<u64>().ok().filter(|&n| n > 0).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("{name} must be a positive integer"),
                    )
                }),
                Err(std::env::VarError::NotPresent) => Ok(default),
                Err(error) => Err(io::Error::new(io::ErrorKind::InvalidInput, error)),
            }
        }
        let seconds = setting("SONIUM_WOLFRAM_TIMEOUT_SECONDS", 900)?;
        let bytes = usize::try_from(setting(
            "SONIUM_WOLFRAM_MAX_OUTPUT_BYTES",
            16 * 1024 * 1024,
        )?)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        bounded_output(self, Duration::from_secs(seconds), bytes).map_err(|error| {
            if error.kind() == io::ErrorKind::NotFound {
                // Reaching this method means a live reference was requested.
                // Do not let legacy optional-reference match arms downgrade an
                // explicitly configured missing executable into a passing test.
                io::Error::other(format!("required Wolfram reference unavailable: {error}"))
            } else {
                error
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn child_fixture() {
        let Ok(mode) = std::env::var("SONIUM_QA_SUBPROCESS_FIXTURE") else {
            return;
        };
        match mode.as_str() {
            "timeout" => thread::sleep(Duration::from_secs(5)),
            "stdout" => std::io::stdout().write_all(&vec![b'x'; 65536]).unwrap(),
            "stderr" => std::io::stderr().write_all(&vec![b'x'; 65536]).unwrap(),
            "fail" => std::process::exit(7),
            _ => {
                println!("reference stdout");
                eprintln!("reference stderr");
            }
        }
    }

    fn fixture(mode: &str) -> Command {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "subprocess::tests::child_fixture", "--nocapture"])
            .env("SONIUM_QA_SUBPROCESS_FIXTURE", mode);
        command
    }

    #[test]
    fn captures_both_streams_and_exit_status() {
        let output = bounded_output(&mut fixture("ok"), Duration::from_secs(5), 4096).unwrap();
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).contains("reference stdout"));
        assert!(String::from_utf8_lossy(&output.stderr).contains("reference stderr"));
        assert_eq!(
            bounded_output(&mut fixture("fail"), Duration::from_secs(5), 4096)
                .unwrap()
                .status
                .code(),
            Some(7)
        );
    }

    #[test]
    fn timeout_and_each_output_cap_are_errors() {
        let start = Instant::now();
        let error =
            bounded_output(&mut fixture("timeout"), Duration::from_millis(100), 4096).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert!(start.elapsed() < Duration::from_secs(4));
        for stream in ["stdout", "stderr"] {
            let error =
                bounded_output(&mut fixture(stream), Duration::from_secs(5), 1024).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        }
    }

    #[test]
    fn missing_executable_and_invalid_limits_are_errors() {
        let mut missing = Command::new("/nonexistent-sonium-qa-executable");
        assert_eq!(
            bounded_output(&mut missing, Duration::from_secs(1), 1024)
                .unwrap_err()
                .kind(),
            io::ErrorKind::NotFound
        );
        assert_eq!(
            bounded_output(&mut fixture("ok"), Duration::ZERO, 1024)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }
}
