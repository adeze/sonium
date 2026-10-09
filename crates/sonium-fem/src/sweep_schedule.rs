//! Deterministic sweep-schedule planning for multi-problem FEM workloads.
//!
//! Reusable driver contract extracted from the room runner
//! (`fem_runner::scheduler`): workload shapes, resource budgets, lane-major
//! solve plans, deterministic worker partitions, and the revalidation rules
//! that keep a persisted plan honest. Execution, checkpointing, and device
//! residency stay with the callers; this module only plans and admits.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Dimensions of a planned family of independent FEM solves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkloadShape {
    pub candidate_count: usize,
    pub frequency_count: usize,
    pub source_count: usize,
    pub rhs_block_size: usize,
}

impl WorkloadShape {
    fn validate(self) -> Result<(), ScheduleError> {
        for (field, value) in [
            ("candidate_count", self.candidate_count),
            ("frequency_count", self.frequency_count),
            ("source_count", self.source_count),
            ("rhs_block_size", self.rhs_block_size),
        ] {
            if value == 0 {
                return Err(ScheduleError::InvalidDimension { field });
            }
        }
        Ok(())
    }
}

/// Memory and lane limits used by [`plan_solve_schedule`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceBudget {
    /// Maximum number of concurrently active lane states.
    pub lane_limit: usize,
    /// Total admitted bytes, including shared state and every active lane.
    pub memory_budget_bytes: usize,
    /// Mesh/operator state shared by all active lanes.
    pub shared_bytes: usize,
    /// Candidate-local state retained by each lane.
    pub per_lane_bytes: usize,
    /// Frequency-local workspace for one active source block.
    pub per_frequency_bytes: usize,
    /// Bytes required per source RHS in the active block.
    pub per_rhs_bytes: usize,
    /// Bytes required per extracted source result in the active block.
    pub per_output_bytes: usize,
}

impl ResourceBudget {
    fn validate(self) -> Result<(), ScheduleError> {
        if self.lane_limit == 0 {
            return Err(ScheduleError::InvalidDimension {
                field: "lane_limit",
            });
        }
        if self.memory_budget_bytes == 0 {
            return Err(ScheduleError::InvalidMemoryBudget);
        }
        Ok(())
    }
}

/// A single source-RHS transaction in the planned solve family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolveWorkItem {
    pub candidate_index: usize,
    pub frequency_index: usize,
    pub source_start: usize,
    pub source_count: usize,
    /// Estimated candidate-local plus active-block bytes, excluding shared bytes.
    pub estimated_bytes: usize,
}

/// Ordered work assigned to one lane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolveLane {
    pub lane_index: usize,
    /// Candidates are kept in assignment order; each candidate's frequencies
    /// remain contiguous so continuation/recycling state cannot cross designs.
    pub candidate_indices: Vec<usize>,
    pub tasks: Vec<SolveWorkItem>,
    pub peak_bytes: usize,
}

/// A fully admitted deterministic solve plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolveSchedule {
    pub lanes: Vec<SolveLane>,
    pub task_count: usize,
    /// Shared bytes plus the peak state of every concurrently active lane.
    pub peak_concurrent_bytes: usize,
}

/// Deterministic subset of a solve schedule assigned to one distributed
/// worker. Task indices are global lane-major indices, so a coordinator can
/// merge worker completions against the same checkpoint cursor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolveWorkerAssignment {
    pub worker_index: usize,
    pub worker_count: usize,
    pub lane_indices: Vec<usize>,
    pub task_indices: Vec<usize>,
    pub schedule_task_count: usize,
}

impl SolveWorkerAssignment {
    /// Validate this assignment against the schedule it was derived from.
    pub fn validate_for(&self, schedule: &SolveSchedule) -> Result<(), ScheduleError> {
        if self.worker_count == 0 {
            return Err(ScheduleError::InvalidDimension {
                field: "worker_count",
            });
        }
        if self.worker_index >= self.worker_count {
            return Err(ScheduleError::InvalidPlan(
                "worker index is outside worker count".into(),
            ));
        }
        if self.schedule_task_count != schedule.task_count {
            return Err(ScheduleError::InvalidPlan(
                "assignment is bound to a different task count".into(),
            ));
        }
        if self
            .lane_indices
            .windows(2)
            .any(|window| window[0] >= window[1])
        {
            return Err(ScheduleError::InvalidPlan(
                "worker lane indices must be strictly increasing".into(),
            ));
        }
        let mut expected_task_indices = Vec::new();
        let mut task_start: usize = 0;
        for (lane_index, lane) in schedule.lanes.iter().enumerate() {
            if lane.lane_index != lane_index {
                return Err(ScheduleError::InvalidPlan(
                    "schedule lane indices are not contiguous".into(),
                ));
            }
            if lane_index % self.worker_count == self.worker_index
                && !self.lane_indices.contains(&lane_index)
            {
                return Err(ScheduleError::InvalidPlan(
                    "assignment omits an owned lane".into(),
                ));
            }
            let task_end = task_start.checked_add(lane.tasks.len()).ok_or(
                ScheduleError::ArithmeticOverflow {
                    field: "assignment task index",
                },
            )?;
            if lane_index % self.worker_count == self.worker_index {
                expected_task_indices.extend(task_start..task_end);
            }
            task_start = task_end;
        }
        if task_start != schedule.task_count {
            return Err(ScheduleError::InvalidPlan(
                "schedule task count does not match lane topology".into(),
            ));
        }
        for &lane_index in &self.lane_indices {
            if lane_index >= schedule.lanes.len() {
                return Err(ScheduleError::InvalidPlan(
                    "assignment references a lane outside the schedule".into(),
                ));
            }
            if lane_index % self.worker_count != self.worker_index {
                return Err(ScheduleError::InvalidPlan(
                    "assignment references a lane owned by another worker".into(),
                ));
            }
        }
        if self.task_indices != expected_task_indices {
            return Err(ScheduleError::InvalidPlan(
                "assignment task indices do not match owned lanes".into(),
            ));
        }
        Ok(())
    }
}

/// Partition a valid lane-major schedule across deterministic worker ranks.
///
/// Lanes, rather than individual tasks, are assigned so continuation and
/// recycling state remains local to one worker. Empty assignments are retained
/// when the worker count exceeds the lane count, preserving rank identity.
pub fn partition_solve_schedule(
    schedule: &SolveSchedule,
    worker_count: usize,
) -> Result<Vec<SolveWorkerAssignment>, ScheduleError> {
    if worker_count == 0 {
        return Err(ScheduleError::InvalidDimension {
            field: "worker_count",
        });
    }
    if schedule.lanes.is_empty() || schedule.task_count == 0 {
        return Err(ScheduleError::InvalidPlan(
            "cannot partition an empty solve schedule".into(),
        ));
    }

    let mut assignments = (0..worker_count)
        .map(|worker_index| SolveWorkerAssignment {
            worker_index,
            worker_count,
            lane_indices: Vec::new(),
            task_indices: Vec::new(),
            schedule_task_count: schedule.task_count,
        })
        .collect::<Vec<_>>();
    let mut task_start: usize = 0;
    for (lane_index, lane) in schedule.lanes.iter().enumerate() {
        if lane.lane_index != lane_index {
            return Err(ScheduleError::InvalidPlan(
                "schedule lane indices are not contiguous".into(),
            ));
        }
        let worker_index = lane_index % worker_count;
        let assignment = &mut assignments[worker_index];
        assignment.lane_indices.push(lane_index);
        assignment
            .task_indices
            .extend(task_start..task_start + lane.tasks.len());
        task_start =
            task_start
                .checked_add(lane.tasks.len())
                .ok_or(ScheduleError::ArithmeticOverflow {
                    field: "partition task index",
                })?;
    }
    if task_start != schedule.task_count {
        return Err(ScheduleError::InvalidPlan(
            "schedule task count does not match lane topology".into(),
        ));
    }
    for assignment in &assignments {
        assignment.validate_for(schedule)?;
    }
    Ok(assignments)
}

/// Typed planner and validation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleError {
    InvalidDimension {
        field: &'static str,
    },
    InvalidMemoryBudget,
    ArithmeticOverflow {
        field: &'static str,
    },
    TaskExceedsMemoryBudget {
        candidate_index: usize,
        frequency_index: usize,
        source_start: usize,
        required_bytes: usize,
        budget_bytes: usize,
    },
    ConcurrentMemoryExceedsBudget {
        required_bytes: usize,
        budget_bytes: usize,
    },
    InvalidPlan(String),
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension { field } => write!(formatter, "{field} must be positive"),
            Self::InvalidMemoryBudget => write!(formatter, "memory budget must be positive"),
            Self::ArithmeticOverflow { field } => {
                write!(formatter, "resource arithmetic overflowed {field}")
            }
            Self::TaskExceedsMemoryBudget {
                candidate_index,
                frequency_index,
                source_start,
                required_bytes,
                budget_bytes,
            } => write!(
                formatter,
                "task candidate {candidate_index}, frequency {frequency_index}, source {source_start} requires {required_bytes} bytes, budget is {budget_bytes}"
            ),
            Self::ConcurrentMemoryExceedsBudget {
                required_bytes,
                budget_bytes,
            } => write!(
                formatter,
                "concurrent schedule requires {required_bytes} bytes, budget is {budget_bytes}"
            ),
            Self::InvalidPlan(reason) => write!(formatter, "invalid solve schedule: {reason}"),
        }
    }
}

impl std::error::Error for ScheduleError {}

fn task_bytes(
    shape: WorkloadShape,
    budget: ResourceBudget,
    source_count: usize,
) -> Result<usize, ScheduleError> {
    if source_count == 0 || source_count > shape.rhs_block_size {
        return Err(ScheduleError::InvalidPlan(
            "source block has invalid width".into(),
        ));
    }
    let per_source = budget
        .per_rhs_bytes
        .checked_add(budget.per_output_bytes)
        .ok_or(ScheduleError::ArithmeticOverflow {
            field: "per-source bytes",
        })?;
    budget
        .per_lane_bytes
        .checked_add(budget.per_frequency_bytes)
        .and_then(|bytes| bytes.checked_add(per_source.checked_mul(source_count)?))
        .ok_or(ScheduleError::ArithmeticOverflow {
            field: "task bytes",
        })
}

fn expected_task_count(shape: WorkloadShape) -> Result<usize, ScheduleError> {
    let blocks_per_frequency = shape
        .source_count
        .checked_add(shape.rhs_block_size - 1)
        .ok_or(ScheduleError::ArithmeticOverflow {
            field: "source block count",
        })?
        / shape.rhs_block_size;
    shape
        .candidate_count
        .checked_mul(shape.frequency_count)
        .and_then(|count| count.checked_mul(blocks_per_frequency))
        .ok_or(ScheduleError::ArithmeticOverflow {
            field: "task count",
        })
}

/// Plan candidate/frequency/source-block work with deterministic lane affinity.
pub fn plan_solve_schedule(
    shape: WorkloadShape,
    budget: ResourceBudget,
) -> Result<SolveSchedule, ScheduleError> {
    shape.validate()?;
    budget.validate()?;

    let lane_count = shape.candidate_count.min(budget.lane_limit);
    let mut lanes = (0..lane_count)
        .map(|lane_index| SolveLane {
            lane_index,
            candidate_indices: Vec::new(),
            tasks: Vec::new(),
            peak_bytes: 0,
        })
        .collect::<Vec<_>>();

    for candidate_index in 0..shape.candidate_count {
        let lane_index = candidate_index % lane_count;
        lanes[lane_index].candidate_indices.push(candidate_index);
        for frequency_index in 0..shape.frequency_count {
            let mut source_start = 0;
            while source_start < shape.source_count {
                let source_count = (shape.source_count - source_start).min(shape.rhs_block_size);
                let estimated_bytes = task_bytes(shape, budget, source_count)?;
                let required_bytes = budget.shared_bytes.checked_add(estimated_bytes).ok_or(
                    ScheduleError::ArithmeticOverflow {
                        field: "single-task memory",
                    },
                )?;
                if required_bytes > budget.memory_budget_bytes {
                    return Err(ScheduleError::TaskExceedsMemoryBudget {
                        candidate_index,
                        frequency_index,
                        source_start,
                        required_bytes,
                        budget_bytes: budget.memory_budget_bytes,
                    });
                }
                lanes[lane_index].tasks.push(SolveWorkItem {
                    candidate_index,
                    frequency_index,
                    source_start,
                    source_count,
                    estimated_bytes,
                });
                lanes[lane_index].peak_bytes = lanes[lane_index].peak_bytes.max(estimated_bytes);
                source_start += source_count;
            }
        }
    }

    let lane_bytes = lanes.iter().try_fold(0usize, |sum, lane| {
        sum.checked_add(lane.peak_bytes)
            .ok_or(ScheduleError::ArithmeticOverflow {
                field: "concurrent lane memory",
            })
    })?;
    let peak_concurrent_bytes =
        budget
            .shared_bytes
            .checked_add(lane_bytes)
            .ok_or(ScheduleError::ArithmeticOverflow {
                field: "concurrent memory",
            })?;
    if peak_concurrent_bytes > budget.memory_budget_bytes {
        return Err(ScheduleError::ConcurrentMemoryExceedsBudget {
            required_bytes: peak_concurrent_bytes,
            budget_bytes: budget.memory_budget_bytes,
        });
    }

    let task_count = lanes.iter().try_fold(0usize, |sum, lane| {
        sum.checked_add(lane.tasks.len())
            .ok_or(ScheduleError::ArithmeticOverflow {
                field: "planned task count",
            })
    })?;
    let schedule = SolveSchedule {
        task_count,
        lanes,
        peak_concurrent_bytes,
    };
    schedule.validate(shape, budget)?;
    Ok(schedule)
}

impl SolveSchedule {
    /// Re-check topology, lane affinity, ordering, byte estimates, and budget.
    pub fn validate(
        &self,
        shape: WorkloadShape,
        budget: ResourceBudget,
    ) -> Result<(), ScheduleError> {
        shape.validate()?;
        budget.validate()?;
        let expected_lane_count = shape.candidate_count.min(budget.lane_limit);
        if self.lanes.len() != expected_lane_count {
            return Err(ScheduleError::InvalidPlan(format!(
                "expected {expected_lane_count} lanes, got {}",
                self.lanes.len()
            )));
        }

        let mut seen = HashSet::new();
        let mut seen_task_count: usize = 0;
        for (lane_index, lane) in self.lanes.iter().enumerate() {
            if lane.lane_index != lane_index {
                return Err(ScheduleError::InvalidPlan(
                    "lane indices are not contiguous".into(),
                ));
            }
            if lane
                .candidate_indices
                .windows(2)
                .any(|window| window[0] >= window[1])
            {
                return Err(ScheduleError::InvalidPlan(
                    "lane candidate order is not strictly increasing".into(),
                ));
            }
            for &candidate_index in &lane.candidate_indices {
                if candidate_index >= shape.candidate_count
                    || candidate_index % expected_lane_count != lane_index
                {
                    return Err(ScheduleError::InvalidPlan(
                        "candidate is assigned to the wrong lane".into(),
                    ));
                }
            }

            let mut last_by_candidate = HashMap::new();
            let lane_candidates = lane
                .candidate_indices
                .iter()
                .copied()
                .collect::<HashSet<_>>();
            let mut task_candidates = HashSet::new();
            let mut recomputed_peak = 0;
            for task in &lane.tasks {
                if task.candidate_index >= shape.candidate_count
                    || task.frequency_index >= shape.frequency_count
                    || task.source_count == 0
                    || task.source_count > shape.rhs_block_size
                    || task.source_start >= shape.source_count
                    || task.source_count > shape.source_count - task.source_start
                    || task.candidate_index % expected_lane_count != lane_index
                {
                    return Err(ScheduleError::InvalidPlan(
                        "task topology or lane affinity is invalid".into(),
                    ));
                }
                if !lane_candidates.contains(&task.candidate_index) {
                    return Err(ScheduleError::InvalidPlan(
                        "task candidate is missing from lane candidate list".into(),
                    ));
                }
                task_candidates.insert(task.candidate_index);
                let key = (
                    task.candidate_index,
                    task.frequency_index,
                    task.source_start,
                );
                if !seen.insert(key) {
                    return Err(ScheduleError::InvalidPlan("duplicate task".into()));
                }
                let order = (task.frequency_index, task.source_start);
                if let Some(previous) = last_by_candidate.insert(task.candidate_index, order)
                    && order <= previous
                {
                    return Err(ScheduleError::InvalidPlan(
                        "candidate frequency/source order is not increasing".into(),
                    ));
                }
                let expected_bytes = task_bytes(shape, budget, task.source_count)?;
                if task.estimated_bytes != expected_bytes {
                    return Err(ScheduleError::InvalidPlan(
                        "task byte estimate does not match budget model".into(),
                    ));
                }
                let required_bytes = budget
                    .shared_bytes
                    .checked_add(task.estimated_bytes)
                    .ok_or(ScheduleError::ArithmeticOverflow {
                        field: "single-task memory",
                    })?;
                if required_bytes > budget.memory_budget_bytes {
                    return Err(ScheduleError::TaskExceedsMemoryBudget {
                        candidate_index: task.candidate_index,
                        frequency_index: task.frequency_index,
                        source_start: task.source_start,
                        required_bytes,
                        budget_bytes: budget.memory_budget_bytes,
                    });
                }
                recomputed_peak = recomputed_peak.max(task.estimated_bytes);
                seen_task_count =
                    seen_task_count
                        .checked_add(1)
                        .ok_or(ScheduleError::ArithmeticOverflow {
                            field: "validated task count",
                        })?;
            }
            if lane_candidates
                .iter()
                .any(|candidate| !task_candidates.contains(candidate))
            {
                return Err(ScheduleError::InvalidPlan(
                    "lane candidate has no assigned task".into(),
                ));
            }
            if lane.peak_bytes != recomputed_peak {
                return Err(ScheduleError::InvalidPlan(
                    "lane peak does not match task estimates".into(),
                ));
            }
        }

        if seen_task_count != expected_task_count(shape)? || self.task_count != seen_task_count {
            return Err(ScheduleError::InvalidPlan(
                "schedule task count does not match workload shape".into(),
            ));
        }
        let expected_peak = budget
            .shared_bytes
            .checked_add(
                self.lanes
                    .iter()
                    .try_fold(0usize, |sum, lane| sum.checked_add(lane.peak_bytes))
                    .ok_or(ScheduleError::ArithmeticOverflow {
                        field: "concurrent lane memory",
                    })?,
            )
            .ok_or(ScheduleError::ArithmeticOverflow {
                field: "concurrent memory",
            })?;
        if self.peak_concurrent_bytes != expected_peak {
            return Err(ScheduleError::InvalidPlan(
                "schedule peak does not match lane peaks".into(),
            ));
        }
        if expected_peak > budget.memory_budget_bytes {
            return Err(ScheduleError::ConcurrentMemoryExceedsBudget {
                required_bytes: expected_peak,
                budget_bytes: budget.memory_budget_bytes,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> WorkloadShape {
        WorkloadShape {
            candidate_count: 3,
            frequency_count: 2,
            source_count: 5,
            rhs_block_size: 2,
        }
    }

    fn budget() -> ResourceBudget {
        ResourceBudget {
            lane_limit: 2,
            memory_budget_bytes: 1_000,
            shared_bytes: 100,
            per_lane_bytes: 100,
            per_frequency_bytes: 20,
            per_rhs_bytes: 30,
            per_output_bytes: 10,
        }
    }

    #[test]
    fn plan_is_deterministic_and_preserves_lane_local_candidate_order() {
        let first = plan_solve_schedule(shape(), budget()).unwrap();
        let second = plan_solve_schedule(shape(), budget()).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.lanes.len(), 2);
        assert_eq!(first.lanes[0].candidate_indices, vec![0, 2]);
        assert_eq!(first.lanes[1].candidate_indices, vec![1]);
        assert_eq!(first.task_count, 18);
        first.validate(shape(), budget()).unwrap();
    }

    #[test]
    fn source_rhs_tail_blocks_are_explicit_and_costed() {
        let plan = plan_solve_schedule(shape(), budget()).unwrap();
        let tail = plan.lanes[0]
            .tasks
            .iter()
            .find(|task| task.source_start == 4)
            .unwrap();
        assert_eq!(tail.source_count, 1);
        assert_eq!(tail.estimated_bytes, 160);
        assert!(
            plan.lanes
                .iter()
                .flat_map(|lane| lane.tasks.iter())
                .all(|task| task.source_start + task.source_count <= shape().source_count)
        );
    }

    #[test]
    fn candidates_never_share_a_lane_with_interleaved_frequency_state() {
        let plan = plan_solve_schedule(shape(), budget()).unwrap();
        for lane in &plan.lanes {
            let mut previous = None;
            for task in &lane.tasks {
                if previous
                    .map(|candidate| candidate != task.candidate_index)
                    .unwrap_or(false)
                {
                    // A later candidate may start only after all work for the
                    // previous candidate has been emitted.
                    assert!(task.candidate_index > previous.unwrap());
                }
                previous = Some(task.candidate_index);
            }
        }
    }

    #[test]
    fn lane_limit_is_capped_by_candidate_count() {
        let mut budget = budget();
        budget.lane_limit = 99;
        let plan = plan_solve_schedule(
            WorkloadShape {
                candidate_count: 2,
                ..shape()
            },
            budget,
        )
        .unwrap();
        assert_eq!(plan.lanes.len(), 2);
        assert_eq!(plan.lanes[0].candidate_indices, vec![0]);
        assert_eq!(plan.lanes[1].candidate_indices, vec![1]);
    }

    #[test]
    fn invalid_dimensions_and_budget_fail_closed() {
        for shape in [
            WorkloadShape {
                candidate_count: 0,
                ..shape()
            },
            WorkloadShape {
                frequency_count: 0,
                ..shape()
            },
            WorkloadShape {
                source_count: 0,
                ..shape()
            },
            WorkloadShape {
                rhs_block_size: 0,
                ..shape()
            },
        ] {
            assert!(matches!(
                plan_solve_schedule(shape, budget()),
                Err(ScheduleError::InvalidDimension { .. })
            ));
        }
        assert!(matches!(
            plan_solve_schedule(
                shape(),
                ResourceBudget {
                    lane_limit: 0,
                    ..budget()
                }
            ),
            Err(ScheduleError::InvalidDimension { .. })
        ));
        assert!(matches!(
            plan_solve_schedule(
                shape(),
                ResourceBudget {
                    memory_budget_bytes: 0,
                    ..budget()
                }
            ),
            Err(ScheduleError::InvalidMemoryBudget)
        ));
    }

    #[test]
    fn a_single_task_that_exceeds_memory_is_rejected_before_dispatch() {
        let error = plan_solve_schedule(
            shape(),
            ResourceBudget {
                memory_budget_bytes: 150,
                shared_bytes: 10,
                per_lane_bytes: 10,
                per_frequency_bytes: 10,
                per_rhs_bytes: 100,
                per_output_bytes: 0,
                ..budget()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ScheduleError::TaskExceedsMemoryBudget { .. }
        ));
    }

    #[test]
    fn aggregate_lane_memory_is_admitted_separately_from_single_task_memory() {
        let error = plan_solve_schedule(
            shape(),
            ResourceBudget {
                memory_budget_bytes: 449,
                shared_bytes: 50,
                ..budget()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ScheduleError::ConcurrentMemoryExceedsBudget { .. }
        ));
    }

    #[test]
    fn arithmetic_overflow_is_typed() {
        let error = plan_solve_schedule(
            WorkloadShape {
                rhs_block_size: 2,
                ..shape()
            },
            ResourceBudget {
                per_rhs_bytes: usize::MAX,
                ..budget()
            },
        )
        .unwrap_err();
        assert!(matches!(error, ScheduleError::ArithmeticOverflow { .. }));
    }

    #[test]
    fn tampered_plan_is_rejected_by_revalidation() {
        let mut plan = plan_solve_schedule(shape(), budget()).unwrap();
        plan.lanes[0].tasks[0].estimated_bytes += 1;
        assert!(matches!(
            plan.validate(shape(), budget()),
            Err(ScheduleError::InvalidPlan(_))
        ));

        let mut missing_candidate = plan_solve_schedule(shape(), budget()).unwrap();
        missing_candidate.lanes[0].candidate_indices.clear();
        assert!(matches!(
            missing_candidate.validate(shape(), budget()),
            Err(ScheduleError::InvalidPlan(_))
        ));
    }
}
