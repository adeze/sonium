//! Deterministic bounded-memory planning for BEM parameter batches.
//!
//! The planner is intentionally execution-agnostic: it provides a safe local
//! work schedule for candidate × frequency × RHS batches, but it neither
//! distributes operators nor performs MPI communication.

use thiserror::Error;

/// Hard resource limits for a local BEM batch schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BemMemoryBudget {
    /// Maximum RAM reserved by all tasks in a concurrent wave.
    pub max_ram_bytes: usize,
    /// Maximum number of independently assembled operator tasks per wave.
    pub max_concurrent_tasks: usize,
}

/// Cost model for a rectangular candidate × frequency × RHS workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BemBatchWorkload {
    /// Number of candidate geometries/designs.
    pub candidate_count: usize,
    /// Number of frequency points per candidate.
    pub frequency_count: usize,
    /// Number of right-hand sides per candidate/frequency point.
    pub rhs_count: usize,
    /// RAM required for the assembled operator and fixed solve state for one
    /// candidate/frequency point.
    pub operator_bytes_per_point: usize,
    /// Additional RAM required for one simultaneously solved RHS.
    pub rhs_bytes_per_rhs: usize,
}

/// One executable local batch, retaining a shared operator across its RHS span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BemBatchTask {
    /// Candidate/design index.
    pub candidate_index: usize,
    /// Frequency index within that candidate.
    pub frequency_index: usize,
    /// First RHS index in this batch.
    pub rhs_start: usize,
    /// Number of adjacent RHS values in this batch.
    pub rhs_count: usize,
    /// Conservative RAM reservation used when placing this task.
    pub reserved_ram_bytes: usize,
}

/// A deterministic set of waves that never exceeds the declared RAM budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BemBatchSchedule {
    /// Sequential waves; tasks in one wave are safe to execute concurrently.
    pub waves: Vec<Vec<BemBatchTask>>,
    /// Largest sum of task reservations in any wave.
    pub peak_reserved_ram_bytes: usize,
}

/// Failure to construct or validate a bounded BEM batch schedule.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum BemSchedulingError {
    /// A dimension or resource limit was zero.
    #[error("BEM batch scheduling requires non-zero {field}")]
    ZeroValue {
        /// Rejected input field.
        field: &'static str,
    },
    /// One candidate/frequency point cannot fit under the declared RAM limit.
    #[error(
        "BEM batch task needs at least {required_ram_bytes} bytes but the RAM budget is {budget_ram_bytes} bytes"
    )]
    TaskDoesNotFit {
        /// Smallest valid reservation for one task.
        required_ram_bytes: usize,
        /// Caller-provided RAM limit.
        budget_ram_bytes: usize,
    },
    /// An arithmetic overflow would make a memory claim unsound.
    #[error("BEM batch scheduling overflow while computing {field}")]
    Overflow {
        /// Operation that overflowed.
        field: &'static str,
    },
    /// A caller-mutated schedule does not cover the requested workload exactly.
    #[error("BEM batch schedule does not cover workload exactly: {reason}")]
    InvalidSchedule {
        /// Validation diagnostic.
        reason: String,
    },
}

/// Construct a deterministic bounded-memory schedule.
///
/// Each task contains a contiguous RHS span so a solver can share one operator
/// across those RHS values. Tasks are emitted in candidate/frequency/RHS order
/// and packed greedily into sequential waves, under both RAM and task-count
/// limits. This is deliberately conservative: an operator reservation is made
/// per task even when a future executor could share it across separate tasks.
pub fn plan_bem_batch(
    workload: BemBatchWorkload,
    budget: BemMemoryBudget,
) -> Result<BemBatchSchedule, BemSchedulingError> {
    validate_inputs(workload, budget)?;
    let minimum = workload
        .operator_bytes_per_point
        .checked_add(workload.rhs_bytes_per_rhs)
        .ok_or(BemSchedulingError::Overflow {
            field: "minimum task RAM",
        })?;
    if minimum > budget.max_ram_bytes {
        return Err(BemSchedulingError::TaskDoesNotFit {
            required_ram_bytes: minimum,
            budget_ram_bytes: budget.max_ram_bytes,
        });
    }
    let rhs_per_task = ((budget.max_ram_bytes - workload.operator_bytes_per_point)
        / workload.rhs_bytes_per_rhs)
        .max(1)
        .min(workload.rhs_count);

    let mut waves = Vec::new();
    let mut current_wave = Vec::new();
    let mut current_reserved = 0_usize;
    let mut peak_reserved = 0_usize;
    for candidate_index in 0..workload.candidate_count {
        for frequency_index in 0..workload.frequency_count {
            for rhs_start in (0..workload.rhs_count).step_by(rhs_per_task) {
                let rhs_count = (workload.rhs_count - rhs_start).min(rhs_per_task);
                let rhs_bytes = workload.rhs_bytes_per_rhs.checked_mul(rhs_count).ok_or(
                    BemSchedulingError::Overflow {
                        field: "RHS task RAM",
                    },
                )?;
                let reserved_ram_bytes =
                    workload
                        .operator_bytes_per_point
                        .checked_add(rhs_bytes)
                        .ok_or(BemSchedulingError::Overflow { field: "task RAM" })?;
                if !current_wave.is_empty()
                    && (current_wave.len() == budget.max_concurrent_tasks
                        || current_reserved
                            .checked_add(reserved_ram_bytes)
                            .ok_or(BemSchedulingError::Overflow { field: "wave RAM" })?
                            > budget.max_ram_bytes)
                {
                    peak_reserved = peak_reserved.max(current_reserved);
                    waves.push(std::mem::take(&mut current_wave));
                    current_reserved = 0;
                }
                current_reserved = current_reserved
                    .checked_add(reserved_ram_bytes)
                    .ok_or(BemSchedulingError::Overflow { field: "wave RAM" })?;
                current_wave.push(BemBatchTask {
                    candidate_index,
                    frequency_index,
                    rhs_start,
                    rhs_count,
                    reserved_ram_bytes,
                });
            }
        }
    }
    if !current_wave.is_empty() {
        peak_reserved = peak_reserved.max(current_reserved);
        waves.push(current_wave);
    }
    let schedule = BemBatchSchedule {
        waves,
        peak_reserved_ram_bytes: peak_reserved,
    };
    schedule.validate(workload, budget)?;
    Ok(schedule)
}

impl BemBatchSchedule {
    /// Validate wave limits and exact candidate/frequency/RHS coverage.
    pub fn validate(
        &self,
        workload: BemBatchWorkload,
        budget: BemMemoryBudget,
    ) -> Result<(), BemSchedulingError> {
        validate_inputs(workload, budget)?;
        let coverage_len = workload
            .candidate_count
            .checked_mul(workload.frequency_count)
            .and_then(|n| n.checked_mul(workload.rhs_count))
            .ok_or(BemSchedulingError::Overflow { field: "coverage" })?;
        let mut coverage = vec![0_u8; coverage_len];
        let mut observed_peak = 0_usize;
        for wave in &self.waves {
            if wave.is_empty() || wave.len() > budget.max_concurrent_tasks {
                return Err(BemSchedulingError::InvalidSchedule {
                    reason: "wave is empty or exceeds its task limit".into(),
                });
            }
            let mut wave_ram = 0_usize;
            for task in wave {
                if task.candidate_index >= workload.candidate_count
                    || task.frequency_index >= workload.frequency_count
                    || task.rhs_count == 0
                    || task
                        .rhs_start
                        .checked_add(task.rhs_count)
                        .is_none_or(|end| end > workload.rhs_count)
                {
                    return Err(BemSchedulingError::InvalidSchedule {
                        reason: "task index is outside the workload".into(),
                    });
                }
                let expected_ram = workload
                    .operator_bytes_per_point
                    .checked_add(
                        workload
                            .rhs_bytes_per_rhs
                            .checked_mul(task.rhs_count)
                            .ok_or(BemSchedulingError::Overflow { field: "task RAM" })?,
                    )
                    .ok_or(BemSchedulingError::Overflow { field: "task RAM" })?;
                if task.reserved_ram_bytes != expected_ram {
                    return Err(BemSchedulingError::InvalidSchedule {
                        reason: "task RAM reservation disagrees with workload".into(),
                    });
                }
                wave_ram = wave_ram
                    .checked_add(task.reserved_ram_bytes)
                    .ok_or(BemSchedulingError::Overflow { field: "wave RAM" })?;
                for rhs_index in task.rhs_start..task.rhs_start + task.rhs_count {
                    let index = (task.candidate_index * workload.frequency_count
                        + task.frequency_index)
                        * workload.rhs_count
                        + rhs_index;
                    coverage[index] = coverage[index].saturating_add(1);
                }
            }
            if wave_ram > budget.max_ram_bytes {
                return Err(BemSchedulingError::InvalidSchedule {
                    reason: "wave exceeds RAM budget".into(),
                });
            }
            observed_peak = observed_peak.max(wave_ram);
        }
        if coverage.iter().any(|&count| count != 1) {
            return Err(BemSchedulingError::InvalidSchedule {
                reason: "candidate/frequency/RHS coverage is missing or duplicated".into(),
            });
        }
        if observed_peak != self.peak_reserved_ram_bytes {
            return Err(BemSchedulingError::InvalidSchedule {
                reason: "reported peak RAM disagrees with waves".into(),
            });
        }
        Ok(())
    }
}

fn validate_inputs(
    workload: BemBatchWorkload,
    budget: BemMemoryBudget,
) -> Result<(), BemSchedulingError> {
    for (field, value) in [
        ("candidate_count", workload.candidate_count),
        ("frequency_count", workload.frequency_count),
        ("rhs_count", workload.rhs_count),
        (
            "operator_bytes_per_point",
            workload.operator_bytes_per_point,
        ),
        ("rhs_bytes_per_rhs", workload.rhs_bytes_per_rhs),
        ("max_ram_bytes", budget.max_ram_bytes),
        ("max_concurrent_tasks", budget.max_concurrent_tasks),
    ] {
        if value == 0 {
            return Err(BemSchedulingError::ZeroValue { field });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workload() -> BemBatchWorkload {
        BemBatchWorkload {
            candidate_count: 2,
            frequency_count: 3,
            rhs_count: 5,
            operator_bytes_per_point: 100,
            rhs_bytes_per_rhs: 10,
        }
    }

    #[test]
    fn schedule_covers_candidate_frequency_rhs_product_with_hard_ram_bound() {
        let budget = BemMemoryBudget {
            max_ram_bytes: 130,
            max_concurrent_tasks: 2,
        };
        let schedule = plan_bem_batch(workload(), budget).expect("bounded plan");
        assert!(schedule.peak_reserved_ram_bytes <= budget.max_ram_bytes);
        assert!(schedule.waves.iter().all(|wave| wave.len() <= 2));
        assert!(
            schedule
                .waves
                .iter()
                .flatten()
                .all(|task| task.rhs_count <= 3)
        );
        schedule
            .validate(workload(), budget)
            .expect("exact coverage");
    }

    #[test]
    fn schedule_is_deterministic_and_rejects_nonfitting_or_mutated_work() {
        let budget = BemMemoryBudget {
            max_ram_bytes: 120,
            max_concurrent_tasks: 1,
        };
        let first = plan_bem_batch(workload(), budget).expect("bounded plan");
        let second = plan_bem_batch(workload(), budget).expect("bounded plan");
        assert_eq!(first, second);
        assert!(matches!(
            plan_bem_batch(
                workload(),
                BemMemoryBudget {
                    max_ram_bytes: 109,
                    max_concurrent_tasks: 1,
                }
            ),
            Err(BemSchedulingError::TaskDoesNotFit { .. })
        ));

        let mut malformed = first;
        malformed.waves[0][0].rhs_count = 0;
        assert!(matches!(
            malformed.validate(workload(), budget),
            Err(BemSchedulingError::InvalidSchedule { .. })
        ));
    }
}
