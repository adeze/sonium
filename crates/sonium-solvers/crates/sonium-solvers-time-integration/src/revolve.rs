//! Bounded-checkpoint schedules for reverse-mode time integration.
//!
//! The schedule is a deterministic revolve-style binary split. It keeps the
//! initial state as a permanent checkpoint, recursively stores midpoint
//! states, reverses the right interval first, releases its midpoint, and then
//! reverses the left interval. With one checkpoint this degenerates to the
//! expected O(`n²`) recomputation schedule; additional checkpoints reduce the
//! recomputation while keeping the live checkpoint count bounded.

use sonium_solvers_core::traits::{AccuracyStatus, SolverFailure, SolverStatus};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use thiserror::Error;

/// Version of the portable [`RevolveSchedule`] checkpoint packet.
pub const REVOLVE_CHECKPOINT_FORMAT_VERSION: u16 = 1;
/// Magic prefix identifying a portable Revolve schedule packet.
pub const REVOLVE_CHECKPOINT_MAGIC: &[u8; 8] = b"SREVOLV1";
const REVOLVE_CHECKPOINT_HEADER_BYTES: usize = 8 + 2 + 2 + 8 + 8 + 8;
const REVOLVE_CHECKPOINT_ACTION_BYTES: usize = 1 + 8 + 8;
const REVOLVE_CHECKPOINT_FLAGS: u16 = 0;

/// One action in a bounded-checkpoint reverse schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevolveAction {
    /// Retain the state at `step` in a checkpoint slot.
    Store { step: usize },
    /// Restore the current state from a retained checkpoint.
    Restore { step: usize },
    /// Recompute forward from `from` to (exclusive) `to`.
    Advance { from: usize, to: usize },
    /// Release a temporary checkpoint slot.
    Release { step: usize },
    /// Run the reverse recurrence for the transition ending at `step + 1`.
    Reverse { step: usize },
}

/// Errors from invalid checkpoint budgets or malformed schedules.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RevolveError {
    /// A non-empty trajectory needs at least one checkpoint slot.
    #[error("{num_steps}-step trajectory requires at least one checkpoint")]
    NoCheckpoint { num_steps: usize },
    /// A schedule action violates its trajectory bounds or state machine.
    #[error("invalid revolve action {index}: {reason}")]
    InvalidAction { index: usize, reason: String },
    /// A user callback failed while executing a valid schedule.
    #[error("revolve callback failed at action {index} during {operation}: {reason}")]
    CallbackFailed {
        index: usize,
        operation: &'static str,
        reason: String,
    },
}

impl RevolveError {
    /// Return the canonical status represented by this checked failure.
    pub fn status(&self) -> SolverStatus {
        SolverStatus::NumericalError
    }

    pub fn solver_status(&self) -> SolverStatus {
        self.status()
    }

    pub fn accuracy_status(&self) -> AccuracyStatus {
        AccuracyStatus::NotChecked
    }

    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

impl SolverFailure for RevolveError {
    fn error_kind(&self) -> sonium_solvers_core::traits::SolverErrorKind {
        sonium_solvers_core::traits::SolverErrorKind::OperationFailed
    }

    fn status(&self) -> SolverStatus {
        RevolveError::status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        RevolveError::accuracy_status(self)
    }
}

/// Deterministic bounded-checkpoint reverse schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RevolveSchedule {
    num_steps: usize,
    max_checkpoints: usize,
    actions: Vec<RevolveAction>,
}

impl RevolveSchedule {
    /// Construct a schedule for `num_steps` transitions and at most
    /// `max_checkpoints` simultaneously live states.
    pub fn new(num_steps: usize, max_checkpoints: usize) -> Result<Self, RevolveError> {
        if num_steps != 0 && max_checkpoints == 0 {
            return Err(RevolveError::NoCheckpoint { num_steps });
        }

        let mut actions = Vec::new();
        if num_steps != 0 {
            actions.push(RevolveAction::Store { step: 0 });
            emit_interval(&mut actions, 0, num_steps, max_checkpoints);
        }

        let schedule = Self {
            num_steps,
            max_checkpoints,
            actions,
        };
        schedule.validate()?;
        Ok(schedule)
    }

    /// Number of forward transitions represented by this schedule.
    pub const fn num_steps(&self) -> usize {
        self.num_steps
    }

    /// Maximum number of checkpoint slots requested by the caller.
    pub const fn max_checkpoints(&self) -> usize {
        self.max_checkpoints
    }

    /// Borrow the executable action sequence.
    pub fn actions(&self) -> &[RevolveAction] {
        &self.actions
    }

    /// Count forward transitions recomputed by the reverse pass.
    pub fn recomputation_steps(&self) -> usize {
        self.actions
            .iter()
            .filter_map(|action| match action {
                RevolveAction::Advance { from, to } => Some(to.saturating_sub(*from)),
                _ => None,
            })
            .sum()
    }

    /// Count reverse recurrence applications.
    pub fn reverse_steps(&self) -> usize {
        self.actions
            .iter()
            .filter(|action| matches!(action, RevolveAction::Reverse { .. }))
            .count()
    }

    /// Serialize this validated schedule into a portable little-endian packet.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, RevolveError> {
        self.validate()?;

        let num_steps = u64::try_from(self.num_steps)
            .map_err(|_| checkpoint_error("num_steps does not fit in u64"))?;
        let max_checkpoints = u64::try_from(self.max_checkpoints)
            .map_err(|_| checkpoint_error("max_checkpoints does not fit in u64"))?;
        let action_count = u64::try_from(self.actions.len())
            .map_err(|_| checkpoint_error("action count does not fit in u64"))?;
        let action_bytes = self
            .actions
            .len()
            .checked_mul(REVOLVE_CHECKPOINT_ACTION_BYTES)
            .ok_or_else(|| checkpoint_error("action payload length overflows usize"))?;
        let capacity = REVOLVE_CHECKPOINT_HEADER_BYTES
            .checked_add(action_bytes)
            .ok_or_else(|| checkpoint_error("checkpoint length overflows usize"))?;

        let mut bytes = Vec::with_capacity(capacity);
        bytes.extend_from_slice(REVOLVE_CHECKPOINT_MAGIC);
        bytes.extend_from_slice(&REVOLVE_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&REVOLVE_CHECKPOINT_FLAGS.to_le_bytes());
        bytes.extend_from_slice(&num_steps.to_le_bytes());
        bytes.extend_from_slice(&max_checkpoints.to_le_bytes());
        bytes.extend_from_slice(&action_count.to_le_bytes());

        for action in &self.actions {
            let (tag, first, second) = match *action {
                RevolveAction::Store { step } => (1u8, step, 0u64),
                RevolveAction::Restore { step } => (2u8, step, 0u64),
                RevolveAction::Advance { from, to } => (
                    3u8,
                    from,
                    u64::try_from(to)
                        .map_err(|_| checkpoint_error("advance endpoint does not fit in u64"))?,
                ),
                RevolveAction::Release { step } => (4u8, step, 0u64),
                RevolveAction::Reverse { step } => (5u8, step, 0u64),
            };
            bytes.push(tag);
            bytes.extend_from_slice(
                &u64::try_from(first)
                    .map_err(|_| checkpoint_error("action operand does not fit in u64"))?
                    .to_le_bytes(),
            );
            bytes.extend_from_slice(&second.to_le_bytes());
        }
        debug_assert_eq!(bytes.len(), capacity);
        Ok(bytes)
    }

    /// Restore a schedule from a strict portable little-endian packet.
    ///
    /// Unknown versions, flags, action tags, non-zero reserved operands,
    /// truncation, trailing bytes, integer conversion failures, and invalid
    /// state-machine sequences are rejected.
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, RevolveError> {
        if bytes.len() < REVOLVE_CHECKPOINT_HEADER_BYTES {
            return Err(checkpoint_error("checkpoint header is truncated"));
        }
        if &bytes[..REVOLVE_CHECKPOINT_MAGIC.len()] != REVOLVE_CHECKPOINT_MAGIC {
            return Err(checkpoint_error("checkpoint magic does not match"));
        }

        let mut cursor = REVOLVE_CHECKPOINT_MAGIC.len();
        let version = read_u16(bytes, &mut cursor)?;
        if version != REVOLVE_CHECKPOINT_FORMAT_VERSION {
            return Err(checkpoint_error("unsupported checkpoint format version"));
        }
        let flags = read_u16(bytes, &mut cursor)?;
        if flags != REVOLVE_CHECKPOINT_FLAGS {
            return Err(checkpoint_error("unsupported checkpoint flags"));
        }
        let num_steps = usize::try_from(read_u64(bytes, &mut cursor)?)
            .map_err(|_| checkpoint_error("num_steps does not fit in usize"))?;
        let max_checkpoints = usize::try_from(read_u64(bytes, &mut cursor)?)
            .map_err(|_| checkpoint_error("max_checkpoints does not fit in usize"))?;
        let action_count = usize::try_from(read_u64(bytes, &mut cursor)?)
            .map_err(|_| checkpoint_error("action count does not fit in usize"))?;

        let action_bytes = action_count
            .checked_mul(REVOLVE_CHECKPOINT_ACTION_BYTES)
            .ok_or_else(|| checkpoint_error("action payload length overflows usize"))?;
        let expected_len = REVOLVE_CHECKPOINT_HEADER_BYTES
            .checked_add(action_bytes)
            .ok_or_else(|| checkpoint_error("checkpoint length overflows usize"))?;
        if bytes.len() < expected_len {
            return Err(checkpoint_error("checkpoint action payload is truncated"));
        }
        if bytes.len() > expected_len {
            return Err(checkpoint_error("checkpoint contains trailing bytes"));
        }

        let mut actions = Vec::with_capacity(action_count);
        for index in 0..action_count {
            let tag = read_byte(bytes, &mut cursor)?;
            let first = read_u64(bytes, &mut cursor)?;
            let second = read_u64(bytes, &mut cursor)?;
            let first = usize::try_from(first)
                .map_err(|_| checkpoint_error_at(index, "action operand does not fit in usize"))?;
            let action = match tag {
                1 if second == 0 => RevolveAction::Store { step: first },
                2 if second == 0 => RevolveAction::Restore { step: first },
                3 => RevolveAction::Advance {
                    from: first,
                    to: usize::try_from(second).map_err(|_| {
                        checkpoint_error_at(index, "advance endpoint does not fit in usize")
                    })?,
                },
                4 if second == 0 => RevolveAction::Release { step: first },
                5 if second == 0 => RevolveAction::Reverse { step: first },
                1..=5 => {
                    return Err(checkpoint_error_at(
                        index,
                        "reserved action operand must be zero",
                    ));
                }
                _ => return Err(checkpoint_error_at(index, "unknown action tag")),
            };
            actions.push(action);
        }

        let schedule = Self {
            num_steps,
            max_checkpoints,
            actions,
        };
        schedule.validate()?;
        Ok(schedule)
    }

    /// Verify bounds, checkpoint lifetime, and forward/reverse state flow.
    ///
    /// This is public so an executor or QA harness can validate schedules
    /// loaded from a serialized plan before invoking device callbacks.
    pub fn validate(&self) -> Result<(), RevolveError> {
        let mut checkpoints = BTreeSet::new();
        let mut reversed = BTreeSet::new();
        let mut current_step = 0usize;
        let mut max_live = 0usize;

        for (index, action) in self.actions.iter().copied().enumerate() {
            let invalid = |reason: &str| RevolveError::InvalidAction {
                index,
                reason: reason.to_string(),
            };

            match action {
                RevolveAction::Store { step } => {
                    if step > self.num_steps {
                        return Err(invalid("store step exceeds trajectory length"));
                    }
                    if step != current_step {
                        return Err(invalid("stored checkpoint does not match current state"));
                    }
                    if !checkpoints.insert(step) {
                        return Err(invalid("checkpoint is already live"));
                    }
                    max_live = max_live.max(checkpoints.len());
                    if max_live > self.max_checkpoints {
                        return Err(invalid("checkpoint budget exceeded"));
                    }
                }
                RevolveAction::Restore { step } => {
                    if !checkpoints.contains(&step) {
                        return Err(invalid("restored checkpoint is not live"));
                    }
                    current_step = step;
                }
                RevolveAction::Advance { from, to } => {
                    if from != current_step {
                        return Err(invalid("advance source does not match current state"));
                    }
                    if to <= from || to > self.num_steps {
                        return Err(invalid("advance range is outside trajectory"));
                    }
                    current_step = to;
                }
                RevolveAction::Release { step } => {
                    if step == 0 {
                        return Err(invalid("initial checkpoint cannot be released"));
                    }
                    if !checkpoints.remove(&step) {
                        return Err(invalid("released checkpoint is not live"));
                    }
                }
                RevolveAction::Reverse { step } => {
                    if step >= self.num_steps {
                        return Err(invalid("reverse step exceeds trajectory length"));
                    }
                    if step.checked_add(1) != Some(current_step) {
                        return Err(invalid("reverse state is not at step + 1"));
                    }
                    if !reversed.insert(step) {
                        return Err(invalid("reverse step was already processed"));
                    }
                    current_step = step;
                }
            }
        }

        if current_step != 0 {
            return Err(RevolveError::InvalidAction {
                index: self.actions.len(),
                reason: "schedule does not finish at the initial state".to_string(),
            });
        }
        if self.num_steps != 0 && !checkpoints.contains(&0) {
            return Err(RevolveError::InvalidAction {
                index: self.actions.len(),
                reason: "initial checkpoint was not retained".to_string(),
            });
        }
        if self.num_steps == 0 {
            if !self.actions.is_empty() {
                return Err(RevolveError::InvalidAction {
                    index: 0,
                    reason: "zero-step schedule must not contain actions".to_string(),
                });
            }
        } else if reversed.len() != self.num_steps {
            return Err(RevolveError::InvalidAction {
                index: self.actions.len(),
                reason: "schedule does not reverse every transition".to_string(),
            });
        }
        if self.num_steps != 0 && checkpoints.len() != 1 {
            return Err(RevolveError::InvalidAction {
                index: self.actions.len(),
                reason: "schedule leaves temporary checkpoints live".to_string(),
            });
        }
        Ok(())
    }

    /// Execute the schedule with caller-provided state callbacks.
    ///
    /// `advance` receives the current forward state and recomputes it from
    /// `from` to `to`. `reverse` receives the forward state at `step + 1` and
    /// can update an adjoint accumulated by the closure. Forward states are
    /// cloned only at checkpoint stores/restores; the executor does not
    /// allocate a complete trajectory history.
    pub fn execute<State, Error, Advance, Reverse>(
        &self,
        initial_state: State,
        mut advance: Advance,
        mut reverse: Reverse,
    ) -> Result<(), RevolveError>
    where
        State: Clone,
        Error: Display,
        Advance: FnMut(usize, usize, State) -> Result<State, Error>,
        Reverse: FnMut(usize, &State) -> Result<(), Error>,
    {
        self.validate()?;
        let mut checkpoints = BTreeMap::new();
        let mut state = initial_state;
        let mut current_step = 0usize;

        for (index, action) in self.actions.iter().copied().enumerate() {
            match action {
                RevolveAction::Store { step } => {
                    checkpoints.insert(step, state.clone());
                }
                RevolveAction::Restore { step } => {
                    state = checkpoints.get(&step).cloned().ok_or_else(|| {
                        RevolveError::InvalidAction {
                            index,
                            reason: "restored checkpoint is not live".to_string(),
                        }
                    })?;
                    current_step = step;
                }
                RevolveAction::Advance { from, to } => {
                    let previous = state;
                    state = advance(from, to, previous).map_err(|error| {
                        RevolveError::CallbackFailed {
                            index,
                            operation: "advance",
                            reason: error.to_string(),
                        }
                    })?;
                    current_step = to;
                }
                RevolveAction::Release { step } => {
                    checkpoints
                        .remove(&step)
                        .ok_or_else(|| RevolveError::InvalidAction {
                            index,
                            reason: "released checkpoint is not live".to_string(),
                        })?;
                }
                RevolveAction::Reverse { step } => {
                    if current_step != step + 1 {
                        return Err(RevolveError::InvalidAction {
                            index,
                            reason: "reverse state is not at step + 1".to_string(),
                        });
                    }
                    reverse(step, &state).map_err(|error| RevolveError::CallbackFailed {
                        index,
                        operation: "reverse",
                        reason: error.to_string(),
                    })?;
                    current_step = step;
                }
            }
        }
        Ok(())
    }
}

fn checkpoint_error(reason: &str) -> RevolveError {
    RevolveError::InvalidAction {
        index: usize::MAX,
        reason: reason.to_string(),
    }
}

fn checkpoint_error_at(index: usize, reason: &str) -> RevolveError {
    RevolveError::InvalidAction {
        index,
        reason: reason.to_string(),
    }
}

fn read_byte(bytes: &[u8], cursor: &mut usize) -> Result<u8, RevolveError> {
    let byte = *bytes
        .get(*cursor)
        .ok_or_else(|| checkpoint_error("checkpoint action payload is truncated"))?;
    *cursor += 1;
    Ok(byte)
}

fn read_u16(bytes: &[u8], cursor: &mut usize) -> Result<u16, RevolveError> {
    let end = cursor
        .checked_add(2)
        .ok_or_else(|| checkpoint_error("checkpoint cursor overflows usize"))?;
    let value = bytes
        .get(*cursor..end)
        .ok_or_else(|| checkpoint_error("checkpoint header is truncated"))?;
    *cursor = end;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, RevolveError> {
    let end = cursor
        .checked_add(8)
        .ok_or_else(|| checkpoint_error("checkpoint cursor overflows usize"))?;
    let value = bytes
        .get(*cursor..end)
        .ok_or_else(|| checkpoint_error("checkpoint payload is truncated"))?;
    *cursor = end;
    Ok(u64::from_le_bytes(
        value.try_into().expect("eight-byte slice"),
    ))
}

fn emit_interval(actions: &mut Vec<RevolveAction>, start: usize, end: usize, checkpoints: usize) {
    let length = end - start;
    if length == 0 {
        return;
    }
    if length == 1 || checkpoints <= 1 {
        for step in (start..end).rev() {
            actions.push(RevolveAction::Restore { step: start });
            actions.push(RevolveAction::Advance {
                from: start,
                to: step + 1,
            });
            actions.push(RevolveAction::Reverse { step });
        }
        return;
    }

    let midpoint = start + length / 2;
    actions.push(RevolveAction::Restore { step: start });
    actions.push(RevolveAction::Advance {
        from: start,
        to: midpoint,
    });
    actions.push(RevolveAction::Store { step: midpoint });
    emit_interval(actions, midpoint, end, checkpoints - 1);
    actions.push(RevolveAction::Release { step: midpoint });
    emit_interval(actions, start, midpoint, checkpoints);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_schedule_needs_no_checkpoint() {
        let schedule = RevolveSchedule::new(0, 0).expect("empty schedule");
        assert!(schedule.actions().is_empty());
        assert_eq!(schedule.reverse_steps(), 0);
        assert_eq!(schedule.recomputation_steps(), 0);
        schedule.validate().expect("empty schedule validates");
    }

    #[test]
    fn non_empty_schedule_rejects_zero_budget() {
        assert_eq!(
            RevolveSchedule::new(4, 0),
            Err(RevolveError::NoCheckpoint { num_steps: 4 })
        );
    }

    #[test]
    fn schedules_reverse_every_step_within_budget() {
        for num_steps in 1..=12 {
            for checkpoints in 1..=5 {
                let schedule =
                    RevolveSchedule::new(num_steps, checkpoints).expect("valid schedule");
                schedule.validate().expect("generated schedule validates");
                assert_eq!(schedule.reverse_steps(), num_steps);

                let mut seen = vec![0usize; num_steps];
                for action in schedule.actions() {
                    if let RevolveAction::Reverse { step } = action {
                        seen[*step] += 1;
                    }
                }
                assert!(seen.into_iter().all(|count| count == 1));
            }
        }
    }

    #[test]
    fn additional_checkpoints_do_not_increase_binary_recomputation() {
        for num_steps in 2..=32 {
            let one = RevolveSchedule::new(num_steps, 1).expect("one checkpoint");
            let three = RevolveSchedule::new(num_steps, 3).expect("three checkpoints");
            assert!(three.recomputation_steps() <= one.recomputation_steps());
        }
    }

    #[test]
    fn executor_recomputes_states_and_reverses_in_order() {
        let schedule = RevolveSchedule::new(6, 2).expect("schedule");
        let mut reversed = Vec::new();
        schedule
            .execute(
                0usize,
                |from, to, state| {
                    assert_eq!(state, from);
                    Ok::<_, &'static str>(to)
                },
                |step, state| {
                    assert_eq!(*state, step + 1);
                    reversed.push(step);
                    Ok::<_, &'static str>(())
                },
            )
            .expect("executor should complete");
        assert_eq!(reversed, (0..6).rev().collect::<Vec<_>>());
    }

    #[test]
    fn executor_reports_callback_failures() {
        let schedule = RevolveSchedule::new(2, 1).expect("schedule");
        let error = schedule
            .execute(
                0usize,
                |_from, _to, _state| Err::<usize, _>("recompute failed"),
                |_step, _state| Ok::<_, &'static str>(()),
            )
            .expect_err("callback error should be typed");
        assert!(matches!(
            error,
            RevolveError::CallbackFailed {
                operation: "advance",
                ..
            }
        ));
    }

    #[test]
    fn checkpoint_round_trip_preserves_schedule_and_header_contract() {
        for (num_steps, checkpoints) in [(0, 0), (0, 3), (1, 1), (6, 2), (17, 4)] {
            let schedule = RevolveSchedule::new(num_steps, checkpoints).expect("schedule");
            let bytes = schedule.to_checkpoint_bytes().expect("serialize");
            assert_eq!(&bytes[..8], REVOLVE_CHECKPOINT_MAGIC);
            assert_eq!(
                u16::from_le_bytes([bytes[8], bytes[9]]),
                REVOLVE_CHECKPOINT_FORMAT_VERSION
            );
            assert_eq!(
                bytes.len(),
                REVOLVE_CHECKPOINT_HEADER_BYTES
                    + schedule.actions().len() * REVOLVE_CHECKPOINT_ACTION_BYTES
            );
            let restored = RevolveSchedule::from_checkpoint_bytes(&bytes).expect("restore");
            assert_eq!(restored, schedule);
        }
    }

    #[test]
    fn checkpoint_decoder_rejects_header_and_payload_corruption() {
        let schedule = RevolveSchedule::new(4, 2).expect("schedule");
        let bytes = schedule.to_checkpoint_bytes().expect("serialize");

        let mut bad_magic = bytes.clone();
        bad_magic[0] ^= 1;
        assert!(RevolveSchedule::from_checkpoint_bytes(&bad_magic).is_err());

        let mut bad_version = bytes.clone();
        bad_version[8] = 2;
        assert!(RevolveSchedule::from_checkpoint_bytes(&bad_version).is_err());

        let mut bad_flags = bytes.clone();
        bad_flags[10] = 1;
        assert!(RevolveSchedule::from_checkpoint_bytes(&bad_flags).is_err());

        assert!(RevolveSchedule::from_checkpoint_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(RevolveSchedule::from_checkpoint_bytes(&trailing).is_err());

        let mut unknown_tag = bytes.clone();
        unknown_tag[REVOLVE_CHECKPOINT_HEADER_BYTES] = 0xff;
        assert!(RevolveSchedule::from_checkpoint_bytes(&unknown_tag).is_err());

        let mut reserved_operand = bytes.clone();
        reserved_operand[REVOLVE_CHECKPOINT_HEADER_BYTES + 1 + 8] = 1;
        assert!(RevolveSchedule::from_checkpoint_bytes(&reserved_operand).is_err());

        let mut malformed_state = bytes;
        malformed_state[REVOLVE_CHECKPOINT_HEADER_BYTES + 1] = 1;
        assert!(RevolveSchedule::from_checkpoint_bytes(&malformed_state).is_err());
    }

    #[test]
    fn checkpoint_decoder_rejects_truncated_headers_and_overflow_counts() {
        for length in 0..REVOLVE_CHECKPOINT_HEADER_BYTES {
            assert!(RevolveSchedule::from_checkpoint_bytes(&vec![0; length]).is_err());
        }

        let mut overflow = vec![0; REVOLVE_CHECKPOINT_HEADER_BYTES];
        overflow[..8].copy_from_slice(REVOLVE_CHECKPOINT_MAGIC);
        overflow[8..10].copy_from_slice(&REVOLVE_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        overflow[28..36].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(RevolveSchedule::from_checkpoint_bytes(&overflow).is_err());
    }

    #[test]
    fn validator_rejects_incomplete_or_leaked_schedules() {
        let incomplete = RevolveSchedule {
            num_steps: 2,
            max_checkpoints: 1,
            actions: vec![RevolveAction::Store { step: 0 }],
        };
        assert!(incomplete.validate().is_err());

        let leaked = RevolveSchedule {
            num_steps: 1,
            max_checkpoints: 2,
            actions: vec![
                RevolveAction::Store { step: 0 },
                RevolveAction::Store { step: 0 },
            ],
        };
        assert!(leaked.validate().is_err());
    }
}
