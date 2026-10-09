//! Test-only protocol worker used by the process-lifecycle integration tests.
//!
//! This binary is deliberately a transport fixture: it validates every
//! request frame and returns a bounded completion response, without pretending
//! to be a physical FEM solver or a QA executable.

use sonium_sim_room::fem_runner::{FemWorkerTaskRequest, FemWorkerTaskResponse};
use std::io::{self, Write};

fn main() {
    match std::env::var("ROOMSIM_WORKER_FIXTURE_MODE").as_deref() {
        Ok("exit_nonzero") => std::process::exit(7),
        Ok("truncate") => {
            let mut stdout = io::stdout().lock();
            stdout.write_all(&[0, 0, 0, 8, 1, 2]).unwrap();
            stdout.flush().unwrap();
            return;
        }
        Ok("hang") => {
            std::thread::sleep(std::time::Duration::from_secs(60));
            return;
        }
        _ => {}
    }

    let task_count: usize = std::env::var("ROOMSIM_WORKER_TASK_COUNT")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .expect("valid ROOMSIM_WORKER_TASK_COUNT");
    let mode = std::env::var("ROOMSIM_WORKER_FIXTURE_MODE").unwrap_or_default();
    let restart_attempt = std::env::var("ROOMSIM_WORKER_RESTART_ATTEMPT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let failed = mode == "failed" || (mode == "fail_first" && restart_attempt == 0);
    let reverse = mode == "reverse";
    let wrong_worker = mode == "wrong_worker";
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdin = stdin.lock();
    let mut stdout = stdout.lock();
    let mut requests = Vec::with_capacity(task_count);
    for _ in 0..task_count {
        requests.push(FemWorkerTaskRequest::read_wire_frame(&mut stdin).unwrap());
    }
    if reverse {
        requests.reverse();
    }
    for request in requests {
        let response = if failed {
            FemWorkerTaskResponse::failed(&request, "fixture worker failure").unwrap()
        } else {
            let mut response =
                FemWorkerTaskResponse::completed(&request, request.task.estimated_bytes);
            if wrong_worker {
                response.worker_index += 1;
            }
            response
        };
        response.write_wire_frame(&mut stdout).unwrap();
        stdout.flush().unwrap();
    }
}
