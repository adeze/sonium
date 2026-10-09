use sonium_sim_room::fem_runner::{
    FemMeshConfig, FemRunCancellation, FemRunCheckpoint, FemRunStateError, FemRunStatus,
    FemWorkerTaskCoordinator, FemWorkerTaskRequest, FemWorkerTaskResponse, ResourceBudget,
    SolveSchedule, SolveWorkItem, WorkloadShape,
    execute_partitioned_scheduled_fem_run_concurrent_with_commit, execute_scheduled_fem_run,
    execute_scheduled_fem_run_concurrent, execute_scheduled_fem_run_concurrent_with_commit,
    plan_fem_candidate_schedule, plan_solve_schedule, run_fem_candidates_concurrent,
    run_fem_candidates_concurrent_with_budget, run_fem_candidates_concurrent_with_commit,
    run_fem_with_mesh, run_fem_with_mesh_cancellation, run_fem_with_mesh_checkpoint_cancellation,
    run_fem_with_mesh_checkpoint_cancellation_with_commit, serve_worker_task_stream,
};
use sonium_solvers::{CheckpointResidence, DistributedCheckpoint};
use sonium_xem::{Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source};
use std::sync::{Arc, Mutex};

fn fixture() -> (SolveSchedule, ResourceBudget) {
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

fn cancellation_fixture() -> RoomSimulation {
    RoomSimulation::with_frequencies(
        RoomGeometry::Rectangular(RectangularRoom::new(1.0, 1.0, 1.0)),
        vec![Source::omnidirectional(Point3D::new(0.25, 0.25, 0.25), 1.0)],
        vec![Point3D::new(0.75, 0.75, 0.75)],
        80.0,
        80.0,
        1,
    )
}

fn source_block_fixture() -> RoomSimulation {
    RoomSimulation::with_frequencies(
        RoomGeometry::Rectangular(RectangularRoom::new(1.0, 1.0, 1.0)),
        vec![
            Source::omnidirectional(Point3D::new(0.25, 0.25, 0.25), 1.0),
            Source::omnidirectional(Point3D::new(0.25, 0.75, 0.25), 1.0),
            Source::omnidirectional(Point3D::new(0.75, 0.25, 0.25), 1.0),
        ],
        vec![Point3D::new(0.75, 0.75, 0.75)],
        80.0,
        80.0,
        1,
    )
}

#[test]
fn public_checkpoint_roundtrips_through_distributed_handoff_and_repartition() {
    let (schedule, budget) = fixture();
    let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
    checkpoint.start(&schedule).unwrap();
    checkpoint
        .commit_next_task(&schedule, 0, task_at(&schedule, 0), 700)
        .unwrap();

    let bytes = checkpoint
        .to_distributed_checkpoint_bytes(3, CheckpointResidence::Host)
        .unwrap();
    let envelope = DistributedCheckpoint::from_checkpoint_bytes(&bytes).unwrap();
    assert_eq!(envelope.world_size(), 3);
    assert_eq!(envelope.residence(), CheckpointResidence::Host);

    let repartitioned = envelope.repartition(5, CheckpointResidence::Metal).unwrap();
    let restored = FemRunCheckpoint::from_distributed_checkpoint_bytes(
        &repartitioned.to_checkpoint_bytes().unwrap(),
    )
    .unwrap();
    restored.validate_for(&schedule).unwrap();
    assert_eq!(restored.status, FemRunStatus::Running);
    assert_eq!(restored.next_task_index, 1);
    assert_eq!(restored.committed_task_count, 1);
}

#[test]
fn public_checkpoint_rejects_corrupt_distributed_handoff() {
    let (schedule, budget) = fixture();
    let checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
    let mut bytes = checkpoint
        .to_distributed_checkpoint_bytes(2, CheckpointResidence::Host)
        .unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;

    let error = FemRunCheckpoint::from_distributed_checkpoint_bytes(&bytes).unwrap_err();
    assert!(error.to_string().contains("distributed FEM checkpoint"));
}

#[test]
fn public_worker_task_protocol_roundtrips_and_rejects_tampering() {
    let (schedule, _budget) = fixture();
    let request = FemWorkerTaskRequest::from_schedule(&schedule, 2, 4).unwrap();
    assert_eq!(request.worker_index, 1);
    request.validate_for(&schedule).unwrap();

    let document = request.to_json().unwrap();
    let restored_request = FemWorkerTaskRequest::from_json(&document).unwrap();
    restored_request.validate_for(&schedule).unwrap();
    let malformed = document.replacen("  \"task\":", "  \"unexpected\": true,\n  \"task\":", 1);
    assert!(FemWorkerTaskRequest::from_json(&malformed).is_err());

    let completed = FemWorkerTaskResponse::completed(&request, 700);
    completed.validate_for(&request).unwrap();
    let restored_completed =
        FemWorkerTaskResponse::from_json(&completed.to_json().unwrap()).unwrap();
    restored_completed.validate_for(&request).unwrap();

    let mut commit_checkpoint = FemRunCheckpoint::new(&schedule, 2_000).unwrap();
    commit_checkpoint.start(&schedule).unwrap();
    let first_request = FemWorkerTaskRequest::from_schedule(&schedule, 2, 0).unwrap();
    let first_response = FemWorkerTaskResponse::completed(&first_request, 700);
    commit_checkpoint
        .commit_worker_response(&schedule, &first_request, &first_response)
        .unwrap();
    let later_request = FemWorkerTaskRequest::from_schedule(&schedule, 2, 2).unwrap();
    let later_response = FemWorkerTaskResponse::completed(&later_request, 700);
    let error = commit_checkpoint
        .commit_worker_response(&schedule, &later_request, &later_response)
        .unwrap_err();
    assert!(matches!(
        error,
        FemRunStateError::CursorMismatch {
            expected: 1,
            actual: 2
        }
    ));

    let failed = FemWorkerTaskResponse::failed(&request, "worker process failed").unwrap();
    failed.validate_for(&request).unwrap();
    let mut failed_checkpoint = FemRunCheckpoint::new(&schedule, 2_000).unwrap();
    failed_checkpoint.start(&schedule).unwrap();
    let failure = failed_checkpoint
        .commit_worker_response(&schedule, &request, &failed)
        .unwrap_err();
    assert!(matches!(failure, FemRunStateError::InvalidCheckpoint(_)));
    assert_eq!(failed_checkpoint.status, FemRunStatus::Cancelled);
    assert!(FemWorkerTaskResponse::failed(&request, " ").is_err());

    let mut tampered = completed;
    tampered.worker_index = 0;
    assert!(tampered.validate_for(&request).is_err());
}

#[test]
fn public_worker_task_wire_frames_round_trip_and_reject_transport_corruption() {
    let (schedule, _) = fixture();
    let request = FemWorkerTaskRequest::from_schedule(&schedule, 2, 4).unwrap();
    let response = FemWorkerTaskResponse::completed(&request, 700);

    let request_frame = request.to_wire_frame_bytes().unwrap();
    let restored_request = FemWorkerTaskRequest::from_wire_frame_bytes(&request_frame).unwrap();
    assert_eq!(restored_request, request);
    restored_request.validate_for(&schedule).unwrap();

    let response_frame = response.to_wire_frame_bytes().unwrap();
    let restored_response = FemWorkerTaskResponse::from_wire_frame_bytes(&response_frame).unwrap();
    assert_eq!(restored_response, response);
    restored_response.validate_for(&request).unwrap();

    let mut truncated = request_frame.clone();
    truncated.pop();
    assert!(FemWorkerTaskRequest::from_wire_frame_bytes(&truncated).is_err());

    let mut trailing = request_frame.clone();
    trailing.push(0);
    assert!(FemWorkerTaskRequest::from_wire_frame_bytes(&trailing).is_err());

    let mut wrong_kind = request_frame;
    let request_marker = b"request";
    let response_marker = b"response";
    let marker_start = wrong_kind
        .windows(request_marker.len())
        .position(|window| window == request_marker)
        .unwrap();
    wrong_kind.splice(
        marker_start..marker_start + request_marker.len(),
        response_marker.iter().copied(),
    );
    let wrong_kind_length = (wrong_kind.len() - 4) as u32;
    wrong_kind[..4].copy_from_slice(&wrong_kind_length.to_be_bytes());
    let error = FemWorkerTaskRequest::from_wire_frame_bytes(&wrong_kind).unwrap_err();
    assert!(error.to_string().contains("frame kind"));

    let mut oversized = vec![0; 4];
    oversized[..4].copy_from_slice(&(1_048_577_u32).to_be_bytes());
    assert!(FemWorkerTaskRequest::from_wire_frame_bytes(&oversized).is_err());
}

#[test]
fn public_worker_task_stream_frames_are_exact_and_short_reads_fail_closed() {
    let (schedule, _) = fixture();
    let first = FemWorkerTaskRequest::from_schedule(&schedule, 2, 0).unwrap();
    let second = FemWorkerTaskRequest::from_schedule(&schedule, 2, 1).unwrap();
    let mut stream = Vec::new();
    first.write_wire_frame(&mut stream).unwrap();
    second.write_wire_frame(&mut stream).unwrap();

    let mut reader = std::io::Cursor::new(stream);
    assert_eq!(
        FemWorkerTaskRequest::read_wire_frame(&mut reader).unwrap(),
        first
    );
    assert_eq!(
        FemWorkerTaskRequest::read_wire_frame(&mut reader).unwrap(),
        second
    );
    assert_eq!(reader.position() as usize, reader.get_ref().len());

    let mut truncated = std::io::Cursor::new(first.to_wire_frame_bytes().unwrap());
    let frame_length = truncated.get_ref().len();
    truncated.get_mut().truncate(frame_length - 1);
    assert!(FemWorkerTaskRequest::read_wire_frame(&mut truncated).is_err());

    let response = FemWorkerTaskResponse::completed(&first, 700);
    let mut response_stream = std::io::Cursor::new(response.to_wire_frame_bytes().unwrap());
    assert_eq!(
        FemWorkerTaskResponse::read_wire_frame(&mut response_stream).unwrap(),
        response
    );
}

#[test]
fn public_worker_task_coordinator_reorders_responses_for_single_writer_commits() {
    let (schedule, budget) = fixture();
    let mut coordinator = FemWorkerTaskCoordinator::new(&schedule, 2).unwrap();
    assert_eq!(coordinator.worker_count(), 2);
    assert_eq!(coordinator.requests_for_worker(0).unwrap().len(), 4);
    assert_eq!(coordinator.requests_for_worker(1).unwrap().len(), 4);

    let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
    checkpoint.start(&schedule).unwrap();
    let request_0 = coordinator.request_for_task(0).unwrap();
    let request_1 = coordinator.request_for_task(1).unwrap();
    let request_2 = coordinator.request_for_task(2).unwrap();

    let response_2 = FemWorkerTaskResponse::completed(&request_2, 700);
    assert_eq!(
        coordinator
            .commit_response_frame(
                &schedule,
                &mut checkpoint,
                &response_2.to_wire_frame_bytes().unwrap(),
            )
            .unwrap(),
        0
    );
    assert_eq!(coordinator.buffered_response_count(), 1);

    let response_0 = FemWorkerTaskResponse::completed(&request_0, 800);
    assert_eq!(
        coordinator
            .commit_response_frame(
                &schedule,
                &mut checkpoint,
                &response_0.to_wire_frame_bytes().unwrap(),
            )
            .unwrap(),
        1
    );
    assert_eq!(coordinator.next_task_index(), 1);
    assert_eq!(checkpoint.next_task_index, 1);

    let response_1 = FemWorkerTaskResponse::completed(&request_1, 900);
    assert_eq!(
        coordinator
            .commit_response_frame(
                &schedule,
                &mut checkpoint,
                &response_1.to_wire_frame_bytes().unwrap(),
            )
            .unwrap(),
        2
    );
    assert!(!coordinator.is_complete());
    assert_eq!(coordinator.next_task_index(), 3);
    assert_eq!(checkpoint.next_task_index, 3);
}

#[test]
fn public_worker_task_coordinator_rejects_duplicates_and_cancels_on_failure() {
    let (schedule, _) = fixture();
    let mut coordinator = FemWorkerTaskCoordinator::new(&schedule, 2).unwrap();
    let request = coordinator.request_for_task(0).unwrap();
    let response = FemWorkerTaskResponse::completed(&request, 700);
    coordinator.accept_response(response.clone()).unwrap();
    assert!(matches!(
        coordinator.accept_response(response),
        Err(FemRunStateError::CursorMismatch {
            expected: 1,
            actual: 0
        })
    ));

    let mut failed_coordinator = FemWorkerTaskCoordinator::new(&schedule, 2).unwrap();
    let failed_request = failed_coordinator.request_for_task(2).unwrap();
    let failed = FemWorkerTaskResponse::failed(&failed_request, "worker process failed").unwrap();
    let error = failed_coordinator.accept_response(failed).unwrap_err();
    assert!(error.to_string().contains("worker task 2 failed"));
    assert!(
        failed_coordinator
            .accept_response(FemWorkerTaskResponse::completed(&request, 700))
            .unwrap_err()
            .to_string()
            .contains("cancelled")
    );

    let mut low_budget_checkpoint = FemRunCheckpoint::new(&schedule, 600).unwrap();
    low_budget_checkpoint.start(&schedule).unwrap();
    let mut low_budget_coordinator = FemWorkerTaskCoordinator::new(&schedule, 2).unwrap();
    let low_budget_request = low_budget_coordinator.request_for_task(0).unwrap();
    let low_budget_response = FemWorkerTaskResponse::completed(&low_budget_request, 700);
    let error = low_budget_coordinator
        .commit_response_frame(
            &schedule,
            &mut low_budget_checkpoint,
            &low_budget_response.to_wire_frame_bytes().unwrap(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("exceeds budget"));
    assert_eq!(low_budget_checkpoint.status, FemRunStatus::Cancelled);
    assert_eq!(low_budget_checkpoint.next_task_index, 0);
    assert_eq!(low_budget_coordinator.next_task_index(), 0);
    assert!(
        low_budget_coordinator
            .accept_response(low_budget_response)
            .unwrap_err()
            .to_string()
            .contains("cancelled")
    );
}

#[test]
fn public_worker_task_coordinator_restores_only_uncommitted_checkpoint_suffix() {
    let (schedule, budget) = fixture();
    let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
    checkpoint.start(&schedule).unwrap();
    checkpoint
        .commit_next_task(&schedule, 0, task_at(&schedule, 0), 700)
        .unwrap();

    let coordinator = FemWorkerTaskCoordinator::from_checkpoint(&schedule, 2, &checkpoint).unwrap();
    coordinator.validate_for(&schedule).unwrap();
    assert_eq!(coordinator.next_task_index(), 1);
    assert_eq!(coordinator.requests_for_worker(0).unwrap().len(), 3);
    assert_eq!(coordinator.requests_for_worker(1).unwrap().len(), 4);
    assert!(matches!(
        coordinator.request_for_task(0),
        Err(FemRunStateError::CursorMismatch {
            expected: 1,
            actual: 0
        })
    ));
    assert_eq!(coordinator.request_for_task(1).unwrap().task_index, 1);

    let completed = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
    let error = FemWorkerTaskCoordinator::from_checkpoint(&schedule, 2, &{
        let mut completed = completed;
        completed.start(&schedule).unwrap();
        for index in 0..schedule.task_count {
            completed
                .commit_next_task(&schedule, index, task_at(&schedule, index), 700)
                .unwrap();
        }
        completed.complete(&schedule).unwrap();
        completed
    })
    .unwrap_err();
    assert!(error.to_string().contains("restore worker coordinator"));
}

#[test]
fn public_worker_task_stream_dispatches_each_worker_and_rejoins_lane_major_commits() {
    let (schedule, budget) = fixture();
    let mut coordinator = FemWorkerTaskCoordinator::new(&schedule, 2).unwrap();
    let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
    checkpoint.start(&schedule).unwrap();

    for worker_index in 0..2 {
        let requests = coordinator.requests_for_worker(worker_index).unwrap();
        let expected_task_indices = requests
            .iter()
            .map(|request| request.task_index)
            .collect::<Vec<_>>();
        let mut request_bytes = Vec::new();
        assert_eq!(
            coordinator
                .write_requests_for_worker(worker_index, &mut request_bytes)
                .unwrap(),
            expected_task_indices.len()
        );
        let mut reader = std::io::Cursor::new(request_bytes);
        let mut response_bytes = Vec::new();
        assert_eq!(
            serve_worker_task_stream(
                &schedule,
                2,
                worker_index,
                &expected_task_indices,
                &mut reader,
                &mut response_bytes,
                |_task| Ok(700),
            )
            .unwrap(),
            expected_task_indices.len()
        );
        let mut response_reader = std::io::Cursor::new(response_bytes);
        for _ in &expected_task_indices {
            let response = FemWorkerTaskResponse::read_wire_frame(&mut response_reader).unwrap();
            let response_bytes = response.to_wire_frame_bytes().unwrap();
            coordinator
                .commit_response_frame(&schedule, &mut checkpoint, &response_bytes)
                .unwrap();
        }
        assert_eq!(
            response_reader.position() as usize,
            response_reader.get_ref().len()
        );
    }
    assert!(coordinator.is_complete());
    checkpoint.complete(&schedule).unwrap();
    assert_eq!(checkpoint.status, FemRunStatus::Completed);
}

#[test]
fn public_worker_task_stream_emits_failure_and_rejects_misrouted_requests() {
    let (schedule, _) = fixture();
    let coordinator = FemWorkerTaskCoordinator::new(&schedule, 2).unwrap();
    let requests = coordinator.requests_for_worker(0).unwrap();
    let expected_task_indices = requests
        .iter()
        .map(|request| request.task_index)
        .collect::<Vec<_>>();
    let mut request_bytes = Vec::new();
    coordinator
        .write_requests_for_worker(0, &mut request_bytes)
        .unwrap();
    let mut reader = std::io::Cursor::new(request_bytes);
    let mut response_bytes = Vec::new();
    let error = serve_worker_task_stream(
        &schedule,
        2,
        0,
        &expected_task_indices,
        &mut reader,
        &mut response_bytes,
        |_task| Err("device kernel failed".into()),
    )
    .unwrap_err();
    assert!(error.to_string().contains("worker task"));
    let mut response_reader = std::io::Cursor::new(response_bytes);
    let response = FemWorkerTaskResponse::read_wire_frame(&mut response_reader).unwrap();
    assert_eq!(
        response.status,
        sonium_sim_room::fem_runner::FemWorkerTaskStatus::Failed
    );

    let wrong_request = coordinator.request_for_task(0).unwrap();
    let mut wrong_bytes = Vec::new();
    wrong_request.write_wire_frame(&mut wrong_bytes).unwrap();
    let mut wrong_reader = std::io::Cursor::new(wrong_bytes);
    let mut wrong_output = Vec::new();
    let error = serve_worker_task_stream(
        &schedule,
        2,
        1,
        &[wrong_request.task_index],
        &mut wrong_reader,
        &mut wrong_output,
        |_task| Ok(700),
    )
    .unwrap_err();
    assert!(error.to_string().contains("not assigned"));
    assert!(wrong_output.is_empty());
}

#[test]
fn public_checkpoint_resumes_at_persisted_cursor() {
    let (schedule, budget) = fixture();
    let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
    checkpoint.start(&schedule).unwrap();
    checkpoint
        .commit_next_task(&schedule, 0, task_at(&schedule, 0), 700)
        .unwrap();

    let document = checkpoint.to_json().unwrap();
    let mut resumed = FemRunCheckpoint::from_json(&document).unwrap();
    resumed.resume(&schedule, 1).unwrap();
    for index in 1..schedule.task_count {
        resumed
            .commit_next_task(&schedule, index, task_at(&schedule, index), 900)
            .unwrap();
    }
    resumed.complete(&schedule).unwrap();
    assert_eq!(resumed.status, FemRunStatus::Completed);
    assert_eq!(resumed.observed_peak_bytes, 900);
}

#[test]
fn public_checkpoint_rejects_schedule_change_and_memory_breach() {
    let (schedule, budget) = fixture();
    let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
    checkpoint.start(&schedule).unwrap();

    let mut changed = schedule.clone();
    changed.lanes[0].tasks[0].estimated_bytes += 1;
    assert!(matches!(
        checkpoint.validate_for(&changed),
        Err(FemRunStateError::ScheduleBindingMismatch {
            field: "fingerprint",
            ..
        })
    ));

    let error = checkpoint
        .record_observed_peak(&schedule, budget.memory_budget_bytes + 1)
        .unwrap_err();
    assert!(matches!(
        error,
        FemRunStateError::ObservedPeakExceedsBudget {
            observed: 1_001,
            budget: 1_000
        }
    ));
    assert_eq!(checkpoint.status, FemRunStatus::Cancelled);
}

#[test]
fn public_cancellation_propagates_first_worker_error() {
    let (schedule, budget) = fixture();
    let cancellation = FemRunCancellation::new();
    let worker = cancellation.clone();
    worker.cancel("worker lane 1 failed");
    cancellation.cancel("worker lane 0 failed later");

    let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
    checkpoint.start(&schedule).unwrap();
    assert!(
        checkpoint
            .poll_cancellation(&schedule, &cancellation)
            .unwrap()
    );
    assert_eq!(
        checkpoint.cancellation_reason.as_deref(),
        Some("worker lane 1 failed")
    );
}

#[test]
fn public_executor_and_production_runner_honor_cancellation() {
    let (schedule, budget) = fixture();
    let cancellation = FemRunCancellation::new();
    let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
    let report =
        execute_scheduled_fem_run(&schedule, &mut checkpoint, &cancellation, |_task| Ok(600))
            .unwrap();
    assert_eq!(report.status, FemRunStatus::Completed);

    let mut concurrent_checkpoint =
        FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
    let concurrent_report = execute_scheduled_fem_run_concurrent(
        &schedule,
        &mut concurrent_checkpoint,
        &FemRunCancellation::new(),
        |_task| Ok(600),
    )
    .unwrap();
    assert_eq!(concurrent_report.status, FemRunStatus::Completed);
    assert_eq!(concurrent_report.committed_task_count, schedule.task_count);

    let candidate_budget = ResourceBudget {
        lane_limit: 2,
        memory_budget_bytes: 1_000,
        shared_bytes: 100,
        per_lane_bytes: 100,
        per_frequency_bytes: 20,
        per_rhs_bytes: 30,
        per_output_bytes: 10,
    };
    let candidate_schedule = plan_solve_schedule(
        WorkloadShape {
            candidate_count: 2,
            frequency_count: 1,
            source_count: 1,
            rhs_block_size: 1,
        },
        candidate_budget,
    )
    .unwrap();
    let mut candidate_checkpoint =
        FemRunCheckpoint::new(&candidate_schedule, candidate_budget.memory_budget_bytes).unwrap();
    let candidate_cancellation = FemRunCancellation::new();
    candidate_cancellation.cancel("cancel candidate dispatch before mesh construction");
    let mut candidate_outputs = vec![None, None];
    let error = run_fem_candidates_concurrent(
        &[cancellation_fixture(), cancellation_fixture()],
        &FemMeshConfig::default(),
        &candidate_schedule,
        &mut candidate_checkpoint,
        &candidate_cancellation,
        &mut candidate_outputs,
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cancel candidate dispatch before mesh construction")
    );
    assert_eq!(candidate_checkpoint.status, FemRunStatus::Cancelled);
    assert!(candidate_outputs.iter().all(Option::is_none));

    let cancellation = FemRunCancellation::new();
    cancellation.cancel("cancel before mesh construction");
    let error = run_fem_with_mesh_cancellation(
        &cancellation_fixture(),
        &FemMeshConfig::default(),
        &cancellation,
    )
    .unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("cancel before mesh construction"),
        "{message}"
    );

    let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
    let error = run_fem_with_mesh_checkpoint_cancellation(
        &cancellation_fixture(),
        &FemMeshConfig::default(),
        &schedule,
        &mut checkpoint,
        &cancellation,
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cancel before mesh construction")
    );
    assert_eq!(checkpoint.status, FemRunStatus::Pending);

    let cancellation = FemRunCancellation::new();
    let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
    let error = run_fem_with_mesh_checkpoint_cancellation(
        &cancellation_fixture(),
        &FemMeshConfig::default(),
        &schedule,
        &mut checkpoint,
        &cancellation,
    )
    .unwrap_err();
    assert!(error.to_string().contains("one schedule lane"));
    assert_eq!(checkpoint.status, FemRunStatus::Pending);
}

#[test]
fn partitioned_executor_exposes_worker_identity_and_global_commit_order() {
    let (schedule, budget) = fixture();
    let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
    let cancellation = FemRunCancellation::new();
    let executions = Arc::new(Mutex::new(Vec::new()));
    let commits = Arc::new(Mutex::new(Vec::new()));

    let execution_log = Arc::clone(&executions);
    let commit_log = Arc::clone(&commits);
    let report = execute_partitioned_scheduled_fem_run_concurrent_with_commit(
        &schedule,
        &mut checkpoint,
        &cancellation,
        2,
        move |worker, task| {
            execution_log
                .lock()
                .unwrap()
                .push((worker, task.candidate_index));
            Ok(task.estimated_bytes)
        },
        move |_checkpoint, worker, task| {
            commit_log
                .lock()
                .unwrap()
                .push((worker, task.candidate_index));
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(report.status, FemRunStatus::Completed);
    assert_eq!(report.committed_task_count, schedule.task_count);
    let execution_workers = executions
        .lock()
        .unwrap()
        .iter()
        .map(|(worker, _)| *worker)
        .collect::<Vec<_>>();
    assert!(execution_workers.contains(&0));
    assert!(execution_workers.contains(&1));
    let commit_workers = commits
        .lock()
        .unwrap()
        .iter()
        .map(|(worker, _)| *worker)
        .collect::<Vec<_>>();
    assert_eq!(commit_workers, vec![0, 0, 0, 0, 1, 1, 1, 1]);
}

#[test]
fn concurrent_checkpoint_commit_callback_failure_cancels_run() {
    let (schedule, budget) = fixture();
    let cancellation = FemRunCancellation::new();
    let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();

    let error = execute_scheduled_fem_run_concurrent_with_commit(
        &schedule,
        &mut checkpoint,
        &cancellation,
        |_task| Ok(600),
        |_checkpoint, _task| Err("checkpoint persistence failed".to_string()),
    )
    .expect_err("a commit callback failure must stop the run");

    assert!(error.to_string().contains("checkpoint persistence failed"));
    assert_eq!(checkpoint.status, FemRunStatus::Cancelled);
    assert_eq!(
        cancellation.first_error().as_deref(),
        Some("checkpoint persistence failed")
    );
}

#[test]
fn production_frequency_runner_persists_and_resumes_after_commit_cancellation() {
    let mut simulation = cancellation_fixture();
    simulation.frequencies = vec![80.0, 120.0];
    let budget = ResourceBudget {
        lane_limit: 1,
        memory_budget_bytes: 1_000,
        shared_bytes: 100,
        per_lane_bytes: 100,
        per_frequency_bytes: 20,
        per_rhs_bytes: 30,
        per_output_bytes: 10,
    };
    let schedule = plan_solve_schedule(
        WorkloadShape {
            candidate_count: 1,
            frequency_count: 2,
            source_count: 1,
            rhs_block_size: 1,
        },
        budget,
    )
    .unwrap();
    let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
    let cancellation = FemRunCancellation::new();
    let interruption = cancellation.clone();
    let callback_schedule = schedule.clone();
    let directory = tempfile::tempdir().unwrap();
    let checkpoint_path = directory.path().join("interrupted-frequency-run.json");
    let callback_path = checkpoint_path.clone();

    let error = run_fem_with_mesh_checkpoint_cancellation_with_commit(
        &simulation,
        &FemMeshConfig::default(),
        &schedule,
        &mut checkpoint,
        &cancellation,
        move |checkpoint, frequency_index| {
            if frequency_index == 0 {
                checkpoint.cancel(&callback_schedule, "interrupted after frequency 0")?;
                interruption.cancel("interrupted after frequency 0");
            }
            checkpoint
                .write_json(&callback_path)
                .map_err(|error| anyhow::anyhow!(error.to_string()))
        },
    )
    .expect_err("frequency commit callback should interrupt the physical run");

    assert!(error.to_string().contains("interrupted after frequency 0"));
    assert_eq!(checkpoint.status, FemRunStatus::Cancelled);
    assert_eq!(checkpoint.next_task_index, 1);

    let mut resumed_checkpoint = FemRunCheckpoint::read_json(&checkpoint_path).unwrap();
    assert_eq!(resumed_checkpoint.status, FemRunStatus::Cancelled);
    let resumed = run_fem_with_mesh_checkpoint_cancellation(
        &simulation,
        &FemMeshConfig::default(),
        &schedule,
        &mut resumed_checkpoint,
        &FemRunCancellation::new(),
    )
    .expect("persisted cancelled frequency run should resume");
    let clean = run_fem_with_mesh(&simulation, &FemMeshConfig::default()).unwrap();

    assert_eq!(resumed_checkpoint.status, FemRunStatus::Completed);
    assert_eq!(resumed.frequencies, clean.frequencies);
    for (resumed_source, clean_source) in resumed.pressures.iter().zip(&clean.pressures) {
        for (resumed_listener, clean_listener) in resumed_source.iter().zip(clean_source) {
            for (resumed_value, clean_value) in resumed_listener.iter().zip(clean_listener) {
                let tolerance = 2.0e-5 * (1.0 + clean_value.norm());
                assert!((*resumed_value - *clean_value).norm() <= tolerance);
            }
        }
    }
}

#[test]
fn production_frequency_runner_accepts_source_block_schedule() {
    let simulation = source_block_fixture();
    let budget = ResourceBudget {
        lane_limit: 1,
        memory_budget_bytes: 1_000,
        shared_bytes: 100,
        per_lane_bytes: 100,
        per_frequency_bytes: 20,
        per_rhs_bytes: 30,
        per_output_bytes: 10,
    };
    let schedule = plan_solve_schedule(
        WorkloadShape {
            candidate_count: 1,
            frequency_count: 1,
            source_count: 3,
            rhs_block_size: 2,
        },
        budget,
    )
    .unwrap();
    assert_eq!(schedule.task_count, 2);

    let mut checkpoint = FemRunCheckpoint::new(&schedule, budget.memory_budget_bytes).unwrap();
    let output = run_fem_with_mesh_checkpoint_cancellation(
        &simulation,
        &FemMeshConfig {
            block_source_solve: true,
            ..FemMeshConfig::default()
        },
        &schedule,
        &mut checkpoint,
        &FemRunCancellation::new(),
    )
    .unwrap();

    assert_eq!(checkpoint.status, FemRunStatus::Completed);
    assert_eq!(checkpoint.committed_task_count, 2);
    assert!(
        output
            .pressures
            .iter()
            .flatten()
            .flatten()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}

#[test]
fn production_candidate_runner_persists_and_resumes_after_commit_cancellation() {
    let candidate_budget = ResourceBudget {
        lane_limit: 1,
        memory_budget_bytes: 1_000,
        shared_bytes: 100,
        per_lane_bytes: 100,
        per_frequency_bytes: 20,
        per_rhs_bytes: 30,
        per_output_bytes: 10,
    };
    let candidate_schedule = plan_solve_schedule(
        WorkloadShape {
            candidate_count: 2,
            frequency_count: 1,
            source_count: 1,
            rhs_block_size: 1,
        },
        candidate_budget,
    )
    .unwrap();
    let simulations = [cancellation_fixture(), cancellation_fixture()];
    let mut checkpoint =
        FemRunCheckpoint::new(&candidate_schedule, candidate_budget.memory_budget_bytes).unwrap();
    let cancellation = FemRunCancellation::new();
    let interruption = cancellation.clone();
    let callback_schedule = candidate_schedule.clone();
    let directory = tempfile::tempdir().unwrap();
    let checkpoint_path = directory.path().join("interrupted-fem-run.json");
    let callback_path = checkpoint_path.clone();
    let mut outputs = vec![None, None];

    let error = run_fem_candidates_concurrent_with_commit(
        &simulations,
        &FemMeshConfig::default(),
        &candidate_schedule,
        &mut checkpoint,
        &cancellation,
        &mut outputs,
        move |checkpoint, task| {
            if task.candidate_index == 0 {
                checkpoint
                    .cancel(&callback_schedule, "interrupted after committed candidate")
                    .map_err(|error| error.to_string())?;
                interruption.cancel("interrupted after committed candidate");
            }
            checkpoint
                .write_json(&callback_path)
                .map_err(|error| error.to_string())
        },
    )
    .expect_err("the commit callback should interrupt after candidate 0");

    assert!(
        error
            .to_string()
            .contains("interrupted after committed candidate")
    );
    assert_eq!(checkpoint.status, FemRunStatus::Cancelled);
    assert!(outputs[0].is_some());
    assert!(outputs[1].is_none());

    let mut resumed_checkpoint = FemRunCheckpoint::read_json(&checkpoint_path).unwrap();
    assert_eq!(resumed_checkpoint.status, FemRunStatus::Cancelled);
    assert_eq!(resumed_checkpoint.next_task_index, 1);
    let completed = run_fem_candidates_concurrent(
        &simulations,
        &FemMeshConfig::default(),
        &candidate_schedule,
        &mut resumed_checkpoint,
        &FemRunCancellation::new(),
        &mut outputs,
    )
    .expect("persisted cancelled candidate run should resume");

    assert_eq!(resumed_checkpoint.status, FemRunStatus::Completed);
    assert_eq!(resumed_checkpoint.committed_task_count, 2);
    assert_eq!(completed.len(), 2);
    assert!(outputs.iter().all(Option::is_some));
}

#[test]
fn production_candidate_runner_resumes_cancelled_persisted_checkpoint() {
    let candidate_budget = ResourceBudget {
        lane_limit: 1,
        memory_budget_bytes: 1_000,
        shared_bytes: 100,
        per_lane_bytes: 100,
        per_frequency_bytes: 20,
        per_rhs_bytes: 30,
        per_output_bytes: 10,
    };
    let candidate_schedule = plan_solve_schedule(
        WorkloadShape {
            candidate_count: 2,
            frequency_count: 1,
            source_count: 1,
            rhs_block_size: 1,
        },
        candidate_budget,
    )
    .unwrap();
    let simulations = [cancellation_fixture(), cancellation_fixture()];

    // The first candidate has already completed physically, but its durable
    // checkpoint records an interruption before the next candidate started.
    let first_output = run_fem_with_mesh(&simulations[0], &FemMeshConfig::default()).unwrap();
    let first_task = candidate_schedule.lanes[0].tasks[0];
    let mut checkpoint =
        FemRunCheckpoint::new(&candidate_schedule, candidate_budget.memory_budget_bytes).unwrap();
    checkpoint.start(&candidate_schedule).unwrap();
    checkpoint
        .commit_next_task(
            &candidate_schedule,
            0,
            first_task,
            first_task.estimated_bytes,
        )
        .unwrap();
    checkpoint
        .cancel(&candidate_schedule, "interrupted after candidate 0")
        .unwrap();

    let directory = tempfile::tempdir().unwrap();
    let checkpoint_path = directory.path().join("cancelled-fem-run.json");
    checkpoint.write_json(&checkpoint_path).unwrap();
    let mut resumed_checkpoint = FemRunCheckpoint::read_json(&checkpoint_path).unwrap();
    assert_eq!(resumed_checkpoint.status, FemRunStatus::Cancelled);
    assert_eq!(
        resumed_checkpoint.cancellation_reason.as_deref(),
        Some("interrupted after candidate 0")
    );

    let mut outputs = vec![Some(first_output), None];
    let completed = run_fem_candidates_concurrent(
        &simulations,
        &FemMeshConfig::default(),
        &candidate_schedule,
        &mut resumed_checkpoint,
        &FemRunCancellation::new(),
        &mut outputs,
    )
    .expect("cancelled production candidate run should resume");

    assert_eq!(resumed_checkpoint.status, FemRunStatus::Completed);
    assert_eq!(resumed_checkpoint.cancellation_reason, None);
    assert_eq!(resumed_checkpoint.committed_task_count, 2);
    assert_eq!(completed.len(), 2);
    assert!(outputs.iter().all(Option::is_some));
    assert!(
        completed
            .iter()
            .all(|output| output.pressures[0][0][0].re.is_finite())
    );
}

#[test]
fn planner_backed_candidate_runner_rejects_pre_cancel_before_mesh() {
    let budget = ResourceBudget {
        lane_limit: 2,
        memory_budget_bytes: 1_000,
        shared_bytes: 100,
        per_lane_bytes: 100,
        per_frequency_bytes: 20,
        per_rhs_bytes: 30,
        per_output_bytes: 10,
    };
    let schedule = plan_fem_candidate_schedule(2, budget).unwrap();
    assert_eq!(schedule.task_count, 2);
    assert_eq!(schedule.lanes.len(), 2);

    let cancellation = FemRunCancellation::new();
    cancellation.cancel("cancel planner-backed candidate dispatch before mesh");
    let error = run_fem_candidates_concurrent_with_budget(
        &[cancellation_fixture(), cancellation_fixture()],
        &FemMeshConfig::default(),
        budget,
        &cancellation,
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cancel planner-backed candidate dispatch before mesh")
    );
}

#[test]
fn planner_backed_candidate_runner_executes_small_workload() {
    let budget = ResourceBudget {
        lane_limit: 1,
        memory_budget_bytes: 1_000,
        shared_bytes: 100,
        per_lane_bytes: 100,
        per_frequency_bytes: 20,
        per_rhs_bytes: 30,
        per_output_bytes: 10,
    };
    let outputs = run_fem_candidates_concurrent_with_budget(
        &[cancellation_fixture()],
        &FemMeshConfig::default(),
        budget,
        &FemRunCancellation::new(),
    )
    .expect("planner-backed candidate runner should execute");
    assert_eq!(outputs.len(), 1);
    assert!(outputs[0].pressures[0][0][0].re.is_finite());
}

#[test]
fn production_candidate_runner_populates_each_output_slot() {
    let candidate_budget = ResourceBudget {
        lane_limit: 2,
        memory_budget_bytes: 1_000,
        shared_bytes: 100,
        per_lane_bytes: 100,
        per_frequency_bytes: 20,
        per_rhs_bytes: 30,
        per_output_bytes: 10,
    };
    let candidate_schedule = plan_solve_schedule(
        WorkloadShape {
            candidate_count: 2,
            frequency_count: 1,
            source_count: 1,
            rhs_block_size: 1,
        },
        candidate_budget,
    )
    .unwrap();
    let mut checkpoint =
        FemRunCheckpoint::new(&candidate_schedule, candidate_budget.memory_budget_bytes).unwrap();
    let cancellation = FemRunCancellation::new();
    let mut outputs = vec![None, None];
    let simulations = [cancellation_fixture(), cancellation_fixture()];

    let completed = run_fem_candidates_concurrent(
        &simulations,
        &FemMeshConfig::default(),
        &candidate_schedule,
        &mut checkpoint,
        &cancellation,
        &mut outputs,
    )
    .expect("production candidate dispatch should complete");

    assert_eq!(checkpoint.status, FemRunStatus::Completed);
    assert_eq!(
        checkpoint.committed_task_count,
        candidate_schedule.task_count
    );
    assert_eq!(completed.len(), simulations.len());
    assert!(outputs.iter().all(Option::is_some));
    for output in completed {
        assert_eq!(output.frequencies, vec![80.0]);
        assert_eq!(output.source_names, vec!["Source"]);
        assert_eq!(output.pressures.len(), 1);
        assert_eq!(output.pressures[0].len(), 1);
        assert_eq!(output.pressures[0][0].len(), 1);
        assert!(output.pressures[0][0][0].re.is_finite());
        assert!(output.pressures[0][0][0].im.is_finite());
    }
}
