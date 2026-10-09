use sonium_sim_room::fem_runner::{
    FemRunCheckpoint, FemRunStatus, FemWorkerProcess, FemWorkerProcessGroup, FemWorkerProcessSpec,
    FemWorkerTaskCoordinator, ResourceBudget, SolveSchedule, WorkloadShape, plan_solve_schedule,
};
use std::time::{Duration, Instant};

fn schedule() -> SolveSchedule {
    plan_solve_schedule(
        WorkloadShape {
            candidate_count: 2,
            frequency_count: 1,
            source_count: 1,
            rhs_block_size: 1,
        },
        ResourceBudget {
            lane_limit: 2,
            memory_budget_bytes: 1_024,
            shared_bytes: 1,
            per_lane_bytes: 1,
            per_frequency_bytes: 1,
            per_rhs_bytes: 1,
            per_output_bytes: 1,
        },
    )
    .unwrap()
}

fn one_worker_schedule() -> SolveSchedule {
    plan_solve_schedule(
        WorkloadShape {
            candidate_count: 1,
            frequency_count: 2,
            source_count: 1,
            rhs_block_size: 1,
        },
        ResourceBudget {
            lane_limit: 1,
            memory_budget_bytes: 1_024,
            shared_bytes: 1,
            per_lane_bytes: 1,
            per_frequency_bytes: 1,
            per_rhs_bytes: 1,
            per_output_bytes: 1,
        },
    )
    .unwrap()
}

fn fixture_spec(mode: &str, task_count: usize) -> FemWorkerProcessSpec {
    FemWorkerProcessSpec::new(env!("CARGO_BIN_EXE_roomsim_worker_protocol_fixture"))
        .env("ROOMSIM_WORKER_TASK_COUNT", task_count.to_string())
        .env("ROOMSIM_WORKER_FIXTURE_MODE", mode)
}

#[test]
fn hung_worker_exchange_is_bounded_and_child_is_terminated() {
    let schedule = one_worker_schedule();
    let coordinator = FemWorkerTaskCoordinator::new(&schedule, 1).unwrap();
    let mut worker = FemWorkerProcess::spawn(
        &fixture_spec("hang", 2).io_timeout(Duration::from_millis(50)),
        0,
        1,
    )
    .unwrap();
    let started = Instant::now();

    let error = worker.exchange(&coordinator).unwrap_err();

    assert!(started.elapsed() < Duration::from_secs(2));
    assert!(error.to_string().contains("timed out"));
}

#[test]
fn hung_worker_shutdown_is_bounded_and_zero_timeout_is_rejected() {
    let spec = fixture_spec("hang", 1).io_timeout(Duration::from_millis(50));
    let worker = FemWorkerProcess::spawn(&spec, 0, 1).unwrap();
    let error = worker.finish().unwrap_err();
    assert!(error.to_string().contains("exit timed out"));

    assert!(
        FemWorkerProcess::spawn(&fixture_spec("ok", 1).io_timeout(Duration::ZERO), 0, 1,).is_err()
    );
}

#[test]
fn automatic_restart_replaces_all_ranks_and_resumes_cancelled_checkpoint() {
    let schedule = schedule();
    let specs = vec![fixture_spec("fail_first", 1), fixture_spec("ok", 1)];
    let mut group = FemWorkerProcessGroup::spawn(specs.clone()).unwrap();
    let mut checkpoint = FemRunCheckpoint::new(&schedule, 1_024).unwrap();

    let report = group
        .execute_checkpointed_with_restart(&schedule, &mut checkpoint, &specs, 1)
        .unwrap();

    assert_eq!(report.status, FemRunStatus::Completed);
    assert_eq!(report.committed_task_count, 2);
    assert_eq!(checkpoint.status, FemRunStatus::Completed);
    assert_eq!(checkpoint.next_task_index, 2);
    group.finish_all().unwrap();
}

#[test]
fn automatic_restart_is_bounded_and_leaves_durable_cancellation() {
    let schedule = one_worker_schedule();
    let specs = vec![fixture_spec("failed", 2)];
    let mut group = FemWorkerProcessGroup::spawn(specs.clone()).unwrap();
    let mut checkpoint = FemRunCheckpoint::new(&schedule, 1_024).unwrap();

    let error = group
        .execute_checkpointed_with_restart(&schedule, &mut checkpoint, &specs, 1)
        .unwrap_err();

    assert!(error.to_string().contains("fixture worker failure"));
    assert_eq!(checkpoint.status, FemRunStatus::Cancelled);
    assert_eq!(checkpoint.next_task_index, 0);
    assert!(checkpoint.cancellation_reason.is_some());
    group.finish_all().unwrap();
}

#[test]
fn automatic_restart_rejects_a_mismatched_rank_spec_set_before_dispatch() {
    let schedule = one_worker_schedule();
    let specs = vec![fixture_spec("ok", 2)];
    let mut group = FemWorkerProcessGroup::spawn(specs.clone()).unwrap();
    let mut checkpoint = FemRunCheckpoint::new(&schedule, 1_024).unwrap();

    let error = group
        .execute_checkpointed_with_restart(&schedule, &mut checkpoint, &[], 1)
        .unwrap_err();

    assert!(error.to_string().contains("spec count"));
    assert_eq!(checkpoint.status, FemRunStatus::Pending);
    group.terminate_all().unwrap();
}
