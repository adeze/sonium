use sonium_sim_room::fem_runner::{
    ResourceBudget, ScheduleError, WorkloadShape, partition_solve_schedule, plan_solve_schedule,
};

fn shape() -> WorkloadShape {
    WorkloadShape {
        candidate_count: 2,
        frequency_count: 3,
        source_count: 4,
        rhs_block_size: 3,
    }
}

fn budget() -> ResourceBudget {
    ResourceBudget {
        lane_limit: 2,
        memory_budget_bytes: 2_000,
        shared_bytes: 100,
        per_lane_bytes: 100,
        per_frequency_bytes: 40,
        per_rhs_bytes: 20,
        per_output_bytes: 10,
    }
}

#[test]
fn public_scheduler_preserves_complete_work_and_lane_affinity() {
    let schedule = plan_solve_schedule(shape(), budget()).expect("plan should be admitted");
    assert_eq!(schedule.task_count, 12);
    assert_eq!(schedule.lanes.len(), 2);
    assert_eq!(schedule.lanes[0].candidate_indices, vec![0]);
    assert_eq!(schedule.lanes[1].candidate_indices, vec![1]);
    schedule
        .validate(shape(), budget())
        .expect("public schedule should revalidate");
}

#[test]
fn public_scheduler_rejects_single_task_and_aggregate_overcommit() {
    let single_task = ResourceBudget {
        memory_budget_bytes: 150,
        shared_bytes: 10,
        per_lane_bytes: 10,
        per_frequency_bytes: 10,
        per_rhs_bytes: 100,
        per_output_bytes: 0,
        ..budget()
    };
    assert!(matches!(
        plan_solve_schedule(shape(), single_task),
        Err(ScheduleError::TaskExceedsMemoryBudget { .. })
    ));

    let aggregate = ResourceBudget {
        memory_budget_bytes: 449,
        shared_bytes: 50,
        ..budget()
    };
    assert!(matches!(
        plan_solve_schedule(shape(), aggregate),
        Err(ScheduleError::ConcurrentMemoryExceedsBudget { .. })
    ));
}

#[test]
fn public_scheduler_revalidation_rejects_tampered_task() {
    let mut schedule = plan_solve_schedule(shape(), budget()).expect("plan should be admitted");
    schedule.lanes[0].tasks[0].source_count += 1;
    assert!(matches!(
        schedule.validate(shape(), budget()),
        Err(ScheduleError::InvalidPlan(_))
    ));
}

#[test]
fn public_scheduler_partitions_complete_lane_major_work_by_worker() {
    let schedule = plan_solve_schedule(shape(), budget()).expect("plan should be admitted");
    let assignments = partition_solve_schedule(&schedule, 2).unwrap();

    assert_eq!(assignments.len(), 2);
    for assignment in &assignments {
        assignment.validate_for(&schedule).unwrap();
    }
    assert_eq!(assignments[0].lane_indices, vec![0]);
    assert_eq!(assignments[1].lane_indices, vec![1]);

    let mut task_indices = assignments
        .iter()
        .flat_map(|assignment| assignment.task_indices.iter().copied())
        .collect::<Vec<_>>();
    task_indices.sort_unstable();
    assert_eq!(task_indices, (0..schedule.task_count).collect::<Vec<_>>());
}

#[test]
fn public_scheduler_rejects_invalid_worker_partition_inputs() {
    let schedule = plan_solve_schedule(shape(), budget()).expect("plan should be admitted");
    assert!(matches!(
        partition_solve_schedule(&schedule, 0),
        Err(ScheduleError::InvalidDimension {
            field: "worker_count"
        })
    ));

    let mut assignment = partition_solve_schedule(&schedule, 2).unwrap().remove(0);
    assignment.lane_indices[0] = 1;
    assert!(assignment.validate_for(&schedule).is_err());
}
