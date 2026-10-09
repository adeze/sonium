//! Checked application of a canonical MeshLang state-transfer plan.
//!
//! The canonical IR deliberately does not know how a numerical field is
//! represented.  This module supplies the small consumer boundary shared by
//! remeshing backends: stable values are copied, removed values are dropped,
//! and every action that needs a new value is delegated to a typed callback.
//! A backend can therefore implement interpolation, conservative projection,
//! or an application-specific transfer without reimplementing plan validation.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::ir::{EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan};

/// Failure while validating an ordered multi-stage state-transfer plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateTransferChainError {
    /// A chain must contain at least one executable stage.
    Empty,
    /// One stage did not satisfy the canonical single-plan contract.
    InvalidStage { stage: usize, reason: String },
    /// A later stage asks to preserve an entity that the preceding stage
    /// explicitly dropped.  This is rejected before any callback runs.
    PreserveWithoutPreviousOutput { stage: usize, entity: EntityId },
}

impl fmt::Display for StateTransferChainError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(formatter, "state-transfer chain must contain a stage"),
            Self::InvalidStage { stage, reason } => {
                write!(
                    formatter,
                    "state-transfer stage {stage} is invalid: {reason}"
                )
            }
            Self::PreserveWithoutPreviousOutput { stage, entity } => write!(
                formatter,
                "state-transfer stage {stage} preserves {entity:?} without a previous output"
            ),
        }
    }
}

impl std::error::Error for StateTransferChainError {}

/// Failure while deriving a topology-safe reverse/coarsening plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateTransferReverseError {
    /// The forward plan was not canonical.
    InvalidPlan(String),
    /// The action needs a numerical inverse that the entity-only plan cannot
    /// infer without weights or a field-specific restriction operator.
    NonInvertibleAction {
        entity: EntityId,
        action: IrStateTransferAction,
    },
}

impl fmt::Display for StateTransferReverseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan(reason) => write!(formatter, "invalid forward plan: {reason}"),
            Self::NonInvertibleAction { entity, action } => write!(
                formatter,
                "cannot derive a coarsening inverse for {action:?} on {entity:?}"
            ),
        }
    }
}

impl std::error::Error for StateTransferReverseError {}

/// Failure while applying a validated multi-stage state-transfer plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateTransferChainApplyError<E> {
    /// The chain was malformed before any callback was invoked.
    InvalidChain(StateTransferChainError),
    /// A stage failed; no intermediate map is returned to the caller.
    Stage {
        stage: usize,
        error: StateTransferApplyError<E>,
    },
}

impl IrStateTransferPlan {
    /// Derive the entity-only inverse used by a reverse/coarsening stage.
    ///
    /// Preserve remains Preserve, Initialize and Drop swap roles, and
    /// ApplicationDefined remains ApplicationDefined. Interpolate and
    /// Conservative are rejected because their inverse requires field- and
    /// geometry-specific restriction weights that are not present in an
    /// entity-level plan.
    pub fn reversed_for_coarsening(&self) -> Result<Self, StateTransferReverseError> {
        self.validate()
            .map_err(|reason| StateTransferReverseError::InvalidPlan(reason.to_owned()))?;
        let entries = self
            .entries
            .iter()
            .map(|entry| {
                let action = match entry.action {
                    IrStateTransferAction::Preserve => IrStateTransferAction::Preserve,
                    IrStateTransferAction::Initialize => IrStateTransferAction::Drop,
                    IrStateTransferAction::Drop => IrStateTransferAction::Initialize,
                    IrStateTransferAction::ApplicationDefined => {
                        IrStateTransferAction::ApplicationDefined
                    }
                    action @ (IrStateTransferAction::Interpolate
                    | IrStateTransferAction::Conservative) => {
                        return Err(StateTransferReverseError::NonInvertibleAction {
                            entity: entry.entity,
                            action,
                        });
                    }
                };
                Ok(IrStateTransferEntry {
                    entity: entry.entity,
                    action,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            responsibility: self.responsibility,
            entries,
        })
    }
}

impl<E: fmt::Display> fmt::Display for StateTransferChainApplyError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidChain(error) => error.fmt(formatter),
            Self::Stage { stage, error } => {
                write!(formatter, "state-transfer stage {stage} failed: {error}")
            }
        }
    }
}

impl<E: fmt::Display + fmt::Debug> std::error::Error for StateTransferChainApplyError<E> {}

/// Ordered executable state-transfer stages for a multi-step remesh.
///
/// A chain intentionally retains each canonical plan instead of flattening
/// actions that may require different numerical consumers.  Applying a chain
/// runs stages in order, validates every stage first, and only publishes the
/// final map after all callbacks succeed.  The caller therefore supplies one
/// callback that can select a numerical kernel from the stage index.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IrStateTransferPlanChain {
    pub stages: Vec<IrStateTransferPlan>,
}

impl IrStateTransferPlanChain {
    /// Construct and validate an ordered chain of adjacent transfer stages.
    pub fn new(stages: Vec<IrStateTransferPlan>) -> Result<Self, StateTransferChainError> {
        let chain = Self { stages };
        chain.validate()?;
        Ok(chain)
    }

    /// Number of adjacent transfer stages in this chain.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Derive the topology-only inverse chain used when walking a remesh
    /// lineage from fine to coarse. Stage order is reversed and each plan is
    /// inverted with [`IrStateTransferPlan::reversed_for_coarsening`].
    /// Numerical interpolation/conservative actions remain rejected because
    /// their restriction weights belong to the field-specific consumer.
    pub fn reversed_for_coarsening(&self) -> Result<Self, StateTransferReverseError> {
        self.validate()
            .map_err(|error| StateTransferReverseError::InvalidPlan(error.to_string()))?;
        let stages = self
            .stages
            .iter()
            .rev()
            .map(IrStateTransferPlan::reversed_for_coarsening)
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(stages).map_err(|error| StateTransferReverseError::InvalidPlan(error.to_string()))
    }

    /// Validate all stages and the deterministic hand-off between them.
    pub fn validate(&self) -> Result<(), StateTransferChainError> {
        if self.stages.is_empty() {
            return Err(StateTransferChainError::Empty);
        }

        for (stage, plan) in self.stages.iter().enumerate() {
            plan.validate()
                .map_err(|reason| StateTransferChainError::InvalidStage {
                    stage,
                    reason: reason.to_owned(),
                })?;

            let Some(previous) = stage.checked_sub(1) else {
                continue;
            };
            let previous_outputs = self.stages[previous]
                .entries
                .iter()
                .filter(|entry| entry.action != IrStateTransferAction::Drop)
                .map(|entry| entry.entity)
                .collect::<BTreeSet<_>>();
            if let Some(entry) = plan.entries.iter().find(|entry| {
                entry.action == IrStateTransferAction::Preserve
                    && !previous_outputs.contains(&entry.entity)
            }) {
                return Err(StateTransferChainError::PreserveWithoutPreviousOutput {
                    stage,
                    entity: entry.entity,
                });
            }
        }

        Ok(())
    }

    /// Apply every stage transactionally and return only the final map.
    pub fn apply<T, E, F>(
        &self,
        source: &BTreeMap<EntityId, T>,
        mut transfer: F,
    ) -> Result<BTreeMap<EntityId, T>, StateTransferChainApplyError<E>>
    where
        T: Clone,
        F: FnMut(usize, EntityId, IrStateTransferAction, Option<&T>) -> Result<T, E>,
    {
        self.validate()
            .map_err(StateTransferChainApplyError::InvalidChain)?;

        let mut values = source.clone();
        for (stage, plan) in self.stages.iter().enumerate() {
            values = plan
                .apply(&values, |entity, action, previous| {
                    transfer(stage, entity, action, previous)
                })
                .map_err(|error| StateTransferChainApplyError::Stage { stage, error })?;
        }
        Ok(values)
    }
}

/// Failure returned while applying a state-transfer plan to a keyed field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateTransferApplyError<E> {
    /// The plan was not in its canonical strictly sorted form.
    InvalidPlan(&'static str),
    /// The source contains state for an entity not described by the plan.
    UnexpectedSource(EntityId),
    /// A preserved entity has no source value.
    MissingSource(EntityId),
    /// A weighted action has no valid source-weight set.
    InvalidWeights {
        entity: EntityId,
        reason: &'static str,
    },
    /// A weighted action references a source entity absent from the field.
    MissingWeightedSource { entity: EntityId, source: EntityId },
    /// A weighted action encountered a non-finite source value.
    NonFiniteWeightedSource { entity: EntityId, source: EntityId },
    /// A weighted action produced a non-finite result.
    NonFiniteWeightedResult(EntityId),
    /// The weight map contains an entity not described by the plan.
    UnexpectedWeights(EntityId),
    /// A backend callback rejected a transfer operation.
    Callback {
        entity: EntityId,
        action: IrStateTransferAction,
        error: E,
    },
}

impl<E: fmt::Display> fmt::Display for StateTransferApplyError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan(message) => {
                write!(formatter, "invalid state-transfer plan: {message}")
            }
            Self::UnexpectedSource(entity) => {
                write!(formatter, "source contains unplanned entity {entity:?}")
            }
            Self::MissingSource(entity) => {
                write!(formatter, "preserved entity {entity:?} has no source value")
            }
            Self::InvalidWeights { entity, reason } => {
                write!(formatter, "invalid weights for {entity:?}: {reason}")
            }
            Self::MissingWeightedSource { entity, source } => write!(
                formatter,
                "weighted transfer for {entity:?} references missing source {source:?}"
            ),
            Self::NonFiniteWeightedSource { entity, source } => write!(
                formatter,
                "weighted transfer for {entity:?} references non-finite source {source:?}"
            ),
            Self::NonFiniteWeightedResult(entity) => {
                write!(
                    formatter,
                    "weighted transfer for {entity:?} produced a non-finite result"
                )
            }
            Self::UnexpectedWeights(entity) => {
                write!(formatter, "weights contain unplanned entity {entity:?}")
            }
            Self::Callback {
                entity,
                action,
                error,
            } => write!(
                formatter,
                "{action:?} transfer for {entity:?} failed: {error}"
            ),
        }
    }
}

impl<E: fmt::Display + fmt::Debug> std::error::Error for StateTransferApplyError<E> {}

/// Failure while deriving deterministic geometry-aware interpolation weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpatialInterpolationError {
    /// The state-transfer plan is not in canonical form.
    InvalidPlan(&'static str),
    /// At least one weighted target was requested without a source point.
    MissingSourcePoints,
    /// A source point is not represented by the state-transfer plan.
    UnexpectedSourcePoint(EntityId),
    /// A target point is not represented by the state-transfer plan.
    UnexpectedTargetPoint(EntityId),
    /// A weighted target has no coordinate.
    MissingTargetPoint(EntityId),
    /// A source or target coordinate contains a non-finite component.
    NonFinitePoint(EntityId),
    /// The source point set contains no usable point for a weighted target.
    NoUsableSourcePoint(EntityId),
    /// Squared distance overflowed despite finite input coordinates.
    NonFiniteDistance { target: EntityId, source: EntityId },
    /// A source entity used by conservative transfer has no measure.
    MissingSourceMeasure(EntityId),
    /// A conservative target entity has no measure.
    MissingTargetMeasure(EntityId),
    /// A source or target measure is non-finite or not strictly positive.
    InvalidMeasure(EntityId),
    /// A caller must request at least one nearest neighbour.
    InvalidNeighborCount,
}

impl fmt::Display for SpatialInterpolationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan(message) => {
                write!(formatter, "invalid state-transfer plan: {message}")
            }
            Self::MissingSourcePoints => {
                write!(
                    formatter,
                    "geometry-aware interpolation requires source points"
                )
            }
            Self::UnexpectedSourcePoint(entity) => {
                write!(
                    formatter,
                    "source points contain unplanned entity {entity:?}"
                )
            }
            Self::UnexpectedTargetPoint(entity) => {
                write!(
                    formatter,
                    "target points contain unplanned entity {entity:?}"
                )
            }
            Self::MissingTargetPoint(entity) => {
                write!(formatter, "weighted target {entity:?} has no point")
            }
            Self::NonFinitePoint(entity) => {
                write!(
                    formatter,
                    "point for {entity:?} contains a non-finite coordinate"
                )
            }
            Self::NoUsableSourcePoint(entity) => {
                write!(
                    formatter,
                    "weighted target {entity:?} has no usable source point"
                )
            }
            Self::NonFiniteDistance { target, source } => write!(
                formatter,
                "distance between target {target:?} and source {source:?} is non-finite"
            ),
            Self::MissingSourceMeasure(entity) => {
                write!(formatter, "conservative source {entity:?} has no measure")
            }
            Self::MissingTargetMeasure(entity) => {
                write!(formatter, "conservative target {entity:?} has no measure")
            }
            Self::InvalidMeasure(entity) => {
                write!(
                    formatter,
                    "measure for {entity:?} must be finite and positive"
                )
            }
            Self::InvalidNeighborCount => write!(formatter, "neighbor count must be positive"),
        }
    }
}

impl std::error::Error for SpatialInterpolationError {}

impl IrStateTransferPlan {
    /// Apply this plan to a keyed field/state map.
    ///
    /// `Preserve` is copied without invoking `transfer`; `Drop` produces no
    /// target entry.  `Initialize`, `Interpolate`, `Conservative`, and
    /// `ApplicationDefined` invoke `transfer` exactly once with the optional
    /// source value for that entity.  The callback owns the numerical policy
    /// and returns the value for the candidate entity.
    ///
    /// The source is checked for stale/unplanned entities before any callback
    /// runs, so a failed application cannot expose a partially transferred
    /// map.  The returned `BTreeMap` preserves deterministic entity ordering.
    pub fn apply<T, E, F>(
        &self,
        source: &BTreeMap<EntityId, T>,
        mut transfer: F,
    ) -> Result<BTreeMap<EntityId, T>, StateTransferApplyError<E>>
    where
        T: Clone,
        F: FnMut(EntityId, IrStateTransferAction, Option<&T>) -> Result<T, E>,
    {
        self.validate()
            .map_err(StateTransferApplyError::InvalidPlan)?;

        let planned = self
            .entries
            .iter()
            .map(|entry| entry.entity)
            .collect::<BTreeSet<_>>();
        if let Some(entity) = source
            .keys()
            .find_map(|entity| (!planned.contains(entity)).then_some(*entity))
        {
            return Err(StateTransferApplyError::UnexpectedSource(entity));
        }

        let mut target = BTreeMap::new();
        for entry in &self.entries {
            match entry.action {
                IrStateTransferAction::Drop => {}
                IrStateTransferAction::Preserve => {
                    let value = source
                        .get(&entry.entity)
                        .ok_or(StateTransferApplyError::MissingSource(entry.entity))?;
                    target.insert(entry.entity, value.clone());
                }
                action => {
                    let value = transfer(entry.entity, action, source.get(&entry.entity)).map_err(
                        |error| StateTransferApplyError::Callback {
                            entity: entry.entity,
                            action,
                            error,
                        },
                    )?;
                    target.insert(entry.entity, value);
                }
            }
        }
        Ok(target)
    }

    /// Apply interpolation and conservative actions from explicit numeric
    /// source weights.  The remesher owns the weights; this consumer checks
    /// their finite/source coverage contract and performs the deterministic
    /// weighted sum before delegating initialize/application-defined actions.
    /// Interpolation weights must sum to one (within a small numerical
    /// tolerance); conservative weights must be finite and have positive
    /// total mass, which lets callers encode cell-volume or quadrature mass.
    pub fn apply_weighted_f64<E, F>(
        &self,
        source: &BTreeMap<EntityId, f64>,
        weights: &BTreeMap<EntityId, Vec<(EntityId, f64)>>,
        mut transfer: F,
    ) -> Result<BTreeMap<EntityId, f64>, StateTransferApplyError<E>>
    where
        F: FnMut(EntityId, IrStateTransferAction, Option<&f64>) -> Result<f64, E>,
    {
        self.validate()
            .map_err(StateTransferApplyError::InvalidPlan)?;
        let planned = self
            .entries
            .iter()
            .map(|entry| entry.entity)
            .collect::<BTreeSet<_>>();
        if let Some(entity) = weights
            .keys()
            .find_map(|entity| (!planned.contains(entity)).then_some(*entity))
        {
            return Err(StateTransferApplyError::UnexpectedWeights(entity));
        }

        for entry in &self.entries {
            let weighted = matches!(
                entry.action,
                IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative
            );
            let Some(samples) = weights.get(&entry.entity) else {
                if weighted {
                    return Err(StateTransferApplyError::InvalidWeights {
                        entity: entry.entity,
                        reason: "weighted action requires at least one source",
                    });
                }
                continue;
            };
            if !weighted {
                return Err(StateTransferApplyError::InvalidWeights {
                    entity: entry.entity,
                    reason: "weights are only valid for interpolation or conservative transfer",
                });
            }
            if samples.is_empty() {
                return Err(StateTransferApplyError::InvalidWeights {
                    entity: entry.entity,
                    reason: "weighted action requires at least one source",
                });
            }
            let mut total = 0.0;
            for (source_entity, weight) in samples {
                if !weight.is_finite() {
                    return Err(StateTransferApplyError::InvalidWeights {
                        entity: entry.entity,
                        reason: "weights must be finite",
                    });
                }
                let value = source.get(source_entity).ok_or(
                    StateTransferApplyError::MissingWeightedSource {
                        entity: entry.entity,
                        source: *source_entity,
                    },
                )?;
                if !value.is_finite() {
                    return Err(StateTransferApplyError::NonFiniteWeightedSource {
                        entity: entry.entity,
                        source: *source_entity,
                    });
                }
                total += *weight;
            }
            if !total.is_finite()
                || (entry.action == IrStateTransferAction::Interpolate
                    && (total - 1.0).abs() > 1.0e-9)
                || (entry.action == IrStateTransferAction::Conservative && total <= 0.0)
            {
                return Err(StateTransferApplyError::InvalidWeights {
                    entity: entry.entity,
                    reason: "weights have an invalid total",
                });
            }
            let result = samples
                .iter()
                .map(|(source_entity, weight)| source[source_entity] * *weight)
                .sum::<f64>();
            if !result.is_finite() {
                return Err(StateTransferApplyError::NonFiniteWeightedResult(
                    entry.entity,
                ));
            }
        }

        // Fallible (not `expect`): a caller holding the weights map across
        // validation and `apply` — or a future refactor that splits the two
        // calls — must get an error, not a panic. The generic closure error
        // type `E` cannot carry `StateTransferApplyError`, so a missing entry
        // is recorded out-of-band and reported after `apply` returns.
        let mut missing_weights: Option<EntityId> = None;
        let applied = self.apply(source, |entity, action, previous| match action {
            IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative => {
                let Some(samples) = weights.get(&entity) else {
                    missing_weights = Some(entity);
                    return Ok(previous.copied().unwrap_or(0.0));
                };
                let value = samples
                    .iter()
                    .map(|(source_entity, weight)| source[source_entity] * *weight)
                    .sum::<f64>();
                Ok(value)
            }
            _ => transfer(entity, action, previous),
        })?;
        if let Some(entity) = missing_weights {
            return Err(StateTransferApplyError::InvalidWeights {
                entity,
                reason: "weighted action has no validated weight entry",
            });
        }
        Ok(applied)
    }

    /// Derive deterministic inverse-distance interpolation weights from source
    /// and target entity coordinates.
    ///
    /// `Interpolate` and `Conservative` entries produce weights;
    /// preserve/drop/initialize and application-defined entries are left for
    /// [`Self::apply_weighted_f64`]'s callback. The nearest `max_neighbors`
    /// source points are selected with an
    /// entity-ID tie break. An exact coordinate match receives weight one;
    /// otherwise inverse Euclidean distance weights are normalized to sum one,
    /// making the result directly consumable by `apply_weighted_f64` for
    /// interpolation. Conservative transfer still requires caller-supplied
    /// mass weights because geometry alone does not determine cell measure.
    pub fn derive_interpolation_weights(
        &self,
        source_points: &BTreeMap<EntityId, [f64; 3]>,
        target_points: &BTreeMap<EntityId, [f64; 3]>,
        max_neighbors: usize,
    ) -> Result<BTreeMap<EntityId, Vec<(EntityId, f64)>>, SpatialInterpolationError> {
        if max_neighbors == 0 {
            return Err(SpatialInterpolationError::InvalidNeighborCount);
        }
        self.validate()
            .map_err(SpatialInterpolationError::InvalidPlan)?;

        let planned = self
            .entries
            .iter()
            .map(|entry| entry.entity)
            .collect::<BTreeSet<_>>();
        if let Some(entity) = source_points
            .keys()
            .find(|entity| !planned.contains(entity))
        {
            return Err(SpatialInterpolationError::UnexpectedSourcePoint(*entity));
        }
        if let Some(entity) = target_points
            .keys()
            .find(|entity| !planned.contains(entity))
        {
            return Err(SpatialInterpolationError::UnexpectedTargetPoint(*entity));
        }
        for (entity, point) in source_points.iter().chain(target_points.iter()) {
            if point.iter().any(|value| !value.is_finite()) {
                return Err(SpatialInterpolationError::NonFinitePoint(*entity));
            }
        }

        let weighted_entities = self
            .entries
            .iter()
            .filter(|entry| {
                matches!(
                    entry.action,
                    IrStateTransferAction::Interpolate | IrStateTransferAction::Conservative
                )
            })
            .map(|entry| entry.entity)
            .collect::<Vec<_>>();
        if weighted_entities.is_empty() {
            return Ok(BTreeMap::new());
        }
        if source_points.is_empty() {
            return Err(SpatialInterpolationError::MissingSourcePoints);
        }

        let mut result = BTreeMap::new();
        for target in weighted_entities {
            let target_point = target_points
                .get(&target)
                .ok_or(SpatialInterpolationError::MissingTargetPoint(target))?;
            let mut candidates = Vec::with_capacity(source_points.len());
            for (source, source_point) in source_points {
                let distance_squared = source_point
                    .iter()
                    .zip(target_point)
                    .map(|(source, target)| {
                        let delta = source - target;
                        delta * delta
                    })
                    .sum::<f64>();
                if !distance_squared.is_finite() {
                    return Err(SpatialInterpolationError::NonFiniteDistance {
                        target,
                        source: *source,
                    });
                }
                candidates.push((*source, distance_squared));
            }
            candidates.sort_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            candidates.truncate(max_neighbors.min(candidates.len()));
            if candidates.is_empty() {
                return Err(SpatialInterpolationError::NoUsableSourcePoint(target));
            }
            if candidates[0].1 == 0.0 {
                result.insert(target, vec![(candidates[0].0, 1.0)]);
                continue;
            }
            let mut weights = candidates
                .iter()
                .map(|(source, distance_squared)| (*source, 1.0 / distance_squared.sqrt()))
                .collect::<Vec<_>>();
            let total = weights.iter().map(|(_, weight)| *weight).sum::<f64>();
            if !total.is_finite() || total <= 0.0 {
                return Err(SpatialInterpolationError::NoUsableSourcePoint(target));
            }
            for (_, weight) in &mut weights {
                *weight /= total;
            }
            result.insert(target, weights);
        }
        Ok(result)
    }

    /// Derive positive conservative weights for piecewise-constant extensive
    /// values when source and target entity measures are available.
    ///
    /// The spatial coefficients are the same deterministic normalized
    /// inverse-distance coefficients as interpolation, then scaled by
    /// `target_measure / source_measure`. This makes a constant density map to
    /// the target extensive amount while preserving the explicit mass policy;
    /// callers still own the meaning of the supplied measures.
    pub fn derive_conservative_weights(
        &self,
        source_points: &BTreeMap<EntityId, [f64; 3]>,
        target_points: &BTreeMap<EntityId, [f64; 3]>,
        source_measures: &BTreeMap<EntityId, f64>,
        target_measures: &BTreeMap<EntityId, f64>,
        max_neighbors: usize,
    ) -> Result<BTreeMap<EntityId, Vec<(EntityId, f64)>>, SpatialInterpolationError> {
        let mut weights =
            self.derive_interpolation_weights(source_points, target_points, max_neighbors)?;
        let planned = self
            .entries
            .iter()
            .map(|entry| entry.entity)
            .collect::<BTreeSet<_>>();
        if let Some(entity) = source_measures
            .keys()
            .find(|entity| !planned.contains(entity))
        {
            return Err(SpatialInterpolationError::UnexpectedSourcePoint(*entity));
        }
        if let Some(entity) = target_measures
            .keys()
            .find(|entity| !planned.contains(entity))
        {
            return Err(SpatialInterpolationError::UnexpectedTargetPoint(*entity));
        }
        for (entity, measure) in source_measures.iter().chain(target_measures.iter()) {
            if !measure.is_finite() || *measure <= 0.0 {
                return Err(SpatialInterpolationError::InvalidMeasure(*entity));
            }
        }
        for (target, samples) in &mut weights {
            let target_measure = *target_measures
                .get(target)
                .ok_or(SpatialInterpolationError::MissingTargetMeasure(*target))?;
            for (source, weight) in samples {
                let source_measure = *source_measures
                    .get(source)
                    .ok_or(SpatialInterpolationError::MissingSourceMeasure(*source))?;
                *weight *= target_measure / source_measure;
                if !weight.is_finite() {
                    return Err(SpatialInterpolationError::InvalidMeasure(*target));
                }
            }
        }
        Ok(weights)
    }
}
