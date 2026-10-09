//! A checked reverse-mode tape for matrix-free linear operators.
//!
//! The tape deliberately stores values and callbacks rather than assuming a
//! particular matrix representation. A BEM/FEM operator can therefore record
//! a forward action together with its exact transpose or Hermitian action,
//! while retaining the forward output for the reverse pass. Backends can use
//! the metadata to account for launches, synchronisations, transfers, and
//! floating-point work without changing the differentiation contract.

use crate::DeviceScalar;
use crate::backend::DeviceBufferLike;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use thiserror::Error;

/// Stable handle for a value owned by an [`OperatorTape`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TapeValueId(usize);

impl TapeValueId {
    /// Return the zero-based position used by the tape.
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Cost and synchronisation metadata attached to one tape node.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TapeMetrics {
    /// Estimated floating-point operations.
    pub flops: u64,
    /// Number of backend launches.
    pub launches: u64,
    /// Number of explicit synchronisations.
    pub synchronisations: u64,
    /// Bytes transferred across a host/device boundary.
    pub transferred_bytes: u64,
    /// Allocations attributed to the operation or tape-owned resources.
    pub allocations: u64,
}

impl TapeMetrics {
    /// Construct node metadata without requiring callers to fill every field.
    pub const fn new(
        flops: u64,
        launches: u64,
        synchronisations: u64,
        transferred_bytes: u64,
    ) -> Self {
        Self {
            flops,
            launches,
            synchronisations,
            transferred_bytes,
            allocations: 0,
        }
    }

    /// Construct metadata including an allocation count.
    pub const fn with_allocations(
        flops: u64,
        launches: u64,
        synchronisations: u64,
        transferred_bytes: u64,
        allocations: u64,
    ) -> Self {
        Self {
            flops,
            launches,
            synchronisations,
            transferred_bytes,
            allocations,
        }
    }

    fn accumulate(&mut self, other: Self) {
        self.flops = self.flops.saturating_add(other.flops);
        self.launches = self.launches.saturating_add(other.launches);
        self.synchronisations = self.synchronisations.saturating_add(other.synchronisations);
        self.transferred_bytes = self
            .transferred_bytes
            .saturating_add(other.transferred_bytes);
        self.allocations = self.allocations.saturating_add(other.allocations);
    }

    /// Check that this operation trace stays within an execution budget.
    ///
    /// The check is intentionally metadata-only: backend callbacks remain
    /// responsible for reporting accurate launch and transfer counts. It
    /// gives steady-state solver QA a single, typed assertion for enforcing
    /// policies such as "no host/device traffic in the inner loop".
    pub fn check_budget(self, budget: TapeBudget) -> Result<(), TapeError> {
        for (metric, actual, limit) in [
            ("flops", self.flops, budget.max_flops),
            ("launches", self.launches, budget.max_launches),
            (
                "synchronisations",
                self.synchronisations,
                budget.max_synchronisations,
            ),
            (
                "transferred_bytes",
                self.transferred_bytes,
                budget.max_transferred_bytes,
            ),
            ("allocations", self.allocations, budget.max_allocations),
        ] {
            if actual > limit {
                return Err(TapeError::BudgetExceeded {
                    metric,
                    limit,
                    actual,
                });
            }
        }
        Ok(())
    }

    /// Assert that an execution trace performed no host/device transfers.
    pub fn require_device_resident(self) -> Result<(), TapeError> {
        self.check_budget(TapeBudget::device_resident())
    }
}

/// Upper bounds for the operation metrics recorded on an [`OperatorTape`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TapeBudget {
    /// Maximum estimated floating-point operations.
    pub max_flops: u64,
    /// Maximum backend launches.
    pub max_launches: u64,
    /// Maximum explicit synchronisations.
    pub max_synchronisations: u64,
    /// Maximum bytes transferred across the host/device boundary.
    pub max_transferred_bytes: u64,
    /// Maximum temporary allocations.
    pub max_allocations: u64,
}

impl TapeBudget {
    /// Budget used by steady-state device-resident solver loops.
    pub const fn device_resident() -> Self {
        Self {
            max_flops: u64::MAX,
            max_launches: u64::MAX,
            max_synchronisations: u64::MAX,
            max_transferred_bytes: 0,
            max_allocations: 0,
        }
    }
}

/// Human-readable operation label and cost metadata for one tape node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TapeNodeMetadata {
    /// Stable operation name used in diagnostics and benchmark artifacts.
    pub operation: &'static str,
    /// Estimated work and synchronisation cost.
    pub metrics: TapeMetrics,
}

impl TapeNodeMetadata {
    /// Create metadata for a recorded operation.
    pub const fn new(operation: &'static str, metrics: TapeMetrics) -> Self {
        Self { operation, metrics }
    }
}

/// Errors raised while recording or differentiating a tape.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TapeError {
    #[error("backend operation failed while executing tape: {0}")]
    Backend(#[from] crate::BackendError),
    #[error("tape value {value} does not exist")]
    InvalidValue { value: usize },
    #[error("{operation} dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        operation: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{operation} failed: {reason}")]
    OperationFailed {
        operation: &'static str,
        reason: String,
    },
    #[error("tape metric {metric} exceeds budget: {actual} > {limit}")]
    BudgetExceeded {
        metric: &'static str,
        limit: u64,
        actual: u64,
    },
    #[error("device operator tape is unusable after {operation} failure; clear it before reuse")]
    InvalidState { operation: &'static str },
}

fn tape_reserve<T>(
    values: &mut Vec<T>,
    required: usize,
    operation: &'static str,
) -> Result<(), TapeError> {
    if required > values.len() {
        values
            .try_reserve_exact(required - values.len())
            .map_err(|error| {
                TapeError::Backend(crate::BackendError::Allocation {
                    reason: format!("{operation}: {error}"),
                })
            })?;
    }
    Ok(())
}

fn tape_zeroed<T: Copy + Default>(
    len: usize,
    operation: &'static str,
) -> Result<Vec<T>, TapeError> {
    Ok(crate::backend::try_zeroed_host::<T>(len, operation)?)
}

type ForwardFn<T> = Arc<dyn Fn(&[T], &mut [T]) -> Result<(), TapeError> + Send + Sync>;
type AdjointFn<T> = Arc<dyn Fn(&[T], &[T], &[T], &mut [T]) -> Result<(), TapeError> + Send + Sync>;

struct TapeNode<T: DeviceScalar> {
    input: TapeValueId,
    output: TapeValueId,
    metadata: TapeNodeMetadata,
    forward: ForwardFn<T>,
    adjoint: AdjointFn<T>,
}

/// Reverse-mode gradients for values recorded on an [`OperatorTape`].
#[derive(Debug, Clone)]
pub struct TapeGradients<T: DeviceScalar> {
    values: Vec<Option<Vec<T>>>,
}

impl<T: DeviceScalar> TapeGradients<T> {
    /// Return the gradient for a value, or `None` if it is disconnected from
    /// the output used for the reverse pass.
    pub fn gradient(&self, value: TapeValueId) -> Result<Option<&[T]>, TapeError> {
        self.values
            .get(value.0)
            .map(|gradient| gradient.as_deref())
            .ok_or(TapeError::InvalidValue { value: value.0 })
    }

    /// Consume the gradient set and return an owned gradient for one value.
    pub fn into_gradient(self, value: TapeValueId) -> Result<Option<Vec<T>>, TapeError> {
        self.values
            .into_iter()
            .nth(value.0)
            .ok_or(TapeError::InvalidValue { value: value.0 })
    }
}

/// A matrix-free, reverse-mode tape for linear operator actions.
pub struct OperatorTape<T: DeviceScalar> {
    values: Vec<Vec<T>>,
    nodes: Vec<TapeNode<T>>,
    budget: Option<TapeBudget>,
}

impl<T: DeviceScalar> Default for OperatorTape<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DeviceScalar> OperatorTape<T> {
    /// Create an empty tape.
    pub const fn new() -> Self {
        Self {
            values: Vec::new(),
            nodes: Vec::new(),
            budget: None,
        }
    }

    /// Remove all values and nodes while retaining the allocated storage.
    pub fn clear(&mut self) {
        self.values.clear();
        self.nodes.clear();
    }

    /// Create a tape that rejects recordings or executions exceeding `budget`.
    pub fn with_budget(budget: TapeBudget) -> Self {
        Self {
            values: Vec::new(),
            nodes: Vec::new(),
            budget: Some(budget),
        }
    }

    /// Replace the execution budget after checking the current tape.
    pub fn set_budget(&mut self, budget: TapeBudget) -> Result<(), TapeError> {
        self.metrics().check_budget(budget)?;
        self.budget = Some(budget);
        Ok(())
    }

    /// Check the cumulative metadata against the configured budget.
    pub fn check_budget(&self) -> Result<(), TapeError> {
        if let Some(budget) = self.budget {
            self.metrics().check_budget(budget)
        } else {
            Ok(())
        }
    }

    /// Number of values currently owned by the tape.
    pub fn value_count(&self) -> usize {
        self.values.len()
    }

    /// Number of recorded operator actions.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Record a forward input value through the checked boundary.
    pub fn try_record_input(&mut self, value: &[T]) -> Result<TapeValueId, TapeError> {
        let next_value_count = self.values.len().checked_add(1).ok_or_else(|| {
            TapeError::Backend(crate::BackendError::Allocation {
                reason: "operator tape value count overflow".into(),
            })
        })?;
        tape_reserve(&mut self.values, next_value_count, "operator tape values")?;
        let mut retained = Vec::new();
        tape_reserve(&mut retained, value.len(), "operator tape input")?;
        retained.extend_from_slice(value);
        let id = TapeValueId(self.values.len());
        self.values.push(retained);
        Ok(id)
    }

    /// Record a forward input value using the legacy infallible API.
    ///
    /// New checked callers should use [`Self::try_record_input`].
    ///
    /// Panics when the tape budget is exhausted; production paths must use
    /// [`Self::try_record_input`] and propagate the typed error instead.
    #[deprecated(note = "panics on budget exhaustion; use try_record_input")]
    pub fn record_input(&mut self, value: &[T]) -> TapeValueId {
        self.try_record_input(value)
            .unwrap_or_else(|error| panic!("operator tape input recording failed: {error}"))
    }

    /// Borrow a forward value retained by the tape.
    pub fn value(&self, id: TapeValueId) -> Result<&[T], TapeError> {
        self.values
            .get(id.0)
            .map(Vec::as_slice)
            .ok_or(TapeError::InvalidValue { value: id.0 })
    }

    /// Replace a retained value without changing the tape topology.
    ///
    /// The shape is fixed after recording so captured backend plans remain
    /// valid when a mesh state or coefficient field is updated before
    /// [`Self::replay`].
    pub fn update_value(&mut self, id: TapeValueId, value: &[T]) -> Result<(), TapeError> {
        let retained = self
            .values
            .get_mut(id.0)
            .ok_or(TapeError::InvalidValue { value: id.0 })?;
        if retained.len() != value.len() {
            return Err(TapeError::DimensionMismatch {
                operation: "tape update value",
                expected: retained.len(),
                actual: value.len(),
            });
        }
        retained.copy_from_slice(value);
        Ok(())
    }

    /// Record and execute a matrix-free linear operation.
    ///
    /// `forward` overwrites the supplied output. `adjoint` receives the
    /// retained input, retained output, and output cotangent, and must add or
    /// overwrite the input cotangent. The callbacks are shared so a caller
    /// can capture a reusable device-resident plan or factorisation.
    pub fn record_linear<F, G>(
        &mut self,
        input: TapeValueId,
        output_len: usize,
        metadata: TapeNodeMetadata,
        forward: F,
        adjoint: G,
    ) -> Result<TapeValueId, TapeError>
    where
        F: Fn(&[T], &mut [T]) -> Result<(), TapeError> + Send + Sync + 'static,
        G: Fn(&[T], &[T], &[T], &mut [T]) -> Result<(), TapeError> + Send + Sync + 'static,
    {
        let mut metadata = metadata;
        metadata.metrics.allocations = metadata.metrics.allocations.saturating_add(1);
        let mut projected = self.metrics();
        projected.accumulate(metadata.metrics);
        // This representation retains each output in a newly allocated host
        // vector. Count that allocation before invoking user code so a
        // device-resident budget fails without partially recording a node.
        if let Some(budget) = self.budget {
            projected.check_budget(budget)?;
        }
        let next_value_count = self.values.len().checked_add(1).ok_or_else(|| {
            TapeError::Backend(crate::BackendError::Allocation {
                reason: "operator tape value count overflow".into(),
            })
        })?;
        let next_node_count = self.nodes.len().checked_add(1).ok_or_else(|| {
            TapeError::Backend(crate::BackendError::Allocation {
                reason: "operator tape node count overflow".into(),
            })
        })?;
        tape_reserve(&mut self.values, next_value_count, "operator tape values")?;
        tape_reserve(&mut self.nodes, next_node_count, "operator tape nodes")?;
        let input_value = self.value(input)?;
        let mut output = tape_zeroed::<T>(output_len, "operator_tape forward output")?;
        invoke_callback(metadata.operation, || forward(input_value, &mut output))?;

        let output_id = TapeValueId(self.values.len());
        self.values.push(output);
        self.nodes.push(TapeNode {
            input,
            output: output_id,
            metadata,
            forward: Arc::new(forward),
            adjoint: Arc::new(adjoint),
        });
        Ok(output_id)
    }

    /// Replay all recorded forward callbacks in insertion order.
    ///
    /// This is useful when a mesh state or coefficient field changes but the
    /// operator plan and tape topology remain reusable.
    pub fn replay(&mut self) -> Result<(), TapeError> {
        self.check_budget()?;
        for node_index in 0..self.nodes.len() {
            let (input, output, operation, forward) = {
                let node = &self.nodes[node_index];
                (
                    node.input,
                    node.output,
                    node.metadata.operation,
                    Arc::clone(&node.forward),
                )
            };
            let mut input_value = Vec::new();
            tape_reserve(
                &mut input_value,
                self.values[input.0].len(),
                "operator_tape replay input",
            )?;
            input_value.extend_from_slice(&self.values[input.0]);
            let output_value = &mut self.values[output.0];
            invoke_callback(operation, || forward(&input_value, output_value))?;
        }
        Ok(())
    }

    /// Aggregate operation metrics for the current tape.
    pub fn metrics(&self) -> TapeMetrics {
        let mut total = TapeMetrics::default();
        for node in &self.nodes {
            total.accumulate(node.metadata.metrics);
        }
        total
    }

    /// Run reverse mode from `output` with the supplied output cotangent.
    pub fn backward(&self, output: TapeValueId, seed: &[T]) -> Result<TapeGradients<T>, TapeError> {
        self.check_budget()?;
        let output_value = self.value(output)?;
        if output_value.len() != seed.len() {
            return Err(TapeError::DimensionMismatch {
                operation: "tape backward seed",
                expected: output_value.len(),
                actual: seed.len(),
            });
        }

        let mut gradients: Vec<Option<Vec<T>>> = Vec::new();
        tape_reserve(&mut gradients, self.values.len(), "operator_tape gradients")?;
        gradients.resize_with(self.values.len(), || None);
        let mut output_gradient = Vec::new();
        tape_reserve(
            &mut output_gradient,
            seed.len(),
            "operator_tape output gradient",
        )?;
        output_gradient.extend_from_slice(seed);
        gradients[output.0] = Some(output_gradient);
        for node in self.nodes.iter().rev() {
            let Some(output_gradient) = gradients[node.output.0].as_ref() else {
                continue;
            };
            let input_value = &self.values[node.input.0];
            let output_value = &self.values[node.output.0];
            let mut input_gradient =
                tape_zeroed::<T>(input_value.len(), "operator_tape input gradient")?;
            let adjoint = Arc::clone(&node.adjoint);
            invoke_callback(node.metadata.operation, || {
                adjoint(
                    input_value,
                    output_value,
                    output_gradient,
                    &mut input_gradient,
                )
            })?;
            if let Some(existing) = gradients[node.input.0].as_mut() {
                for (current, contribution) in existing.iter_mut().zip(input_gradient) {
                    *current = *current + contribution;
                }
            } else {
                gradients[node.input.0] = Some(input_gradient);
            }
        }
        Ok(TapeGradients { values: gradients })
    }
}

/// A backend-owned reverse-mode tape.
///
/// Unlike [`OperatorTape`], this variant retains values in backend buffers and
/// executes callbacks against those buffers.  The tape owns every recorded
/// value, so replay and backward do not require host round-trips.  Callback
/// implementations are responsible for selecting native or explicitly
/// documented fallback kernels; failures are preserved as [`TapeError::Backend`]
/// and never publish a partial value.
pub struct DeviceOperatorTape<B: crate::Backend, T: DeviceScalar> {
    values: Vec<B::Buffer<T>>,
    nodes: Vec<DeviceTapeNode<B, T>>,
    metrics: TapeMetrics,
    budget: Option<TapeBudget>,
    poisoned: bool,
}

#[allow(clippy::type_complexity)]
struct DeviceTapeNode<B: crate::Backend, T: DeviceScalar> {
    input: TapeValueId,
    output: TapeValueId,
    metadata: TapeNodeMetadata,
    forward:
        Arc<dyn Fn(&B, &B::Buffer<T>, &mut B::Buffer<T>) -> crate::BackendResult<()> + Send + Sync>,
    adjoint: Arc<
        dyn Fn(
                &B,
                &B::Buffer<T>,
                &B::Buffer<T>,
                &B::Buffer<T>,
                &mut B::Buffer<T>,
            ) -> crate::BackendResult<()>
            + Send
            + Sync,
    >,
}

impl<B: crate::Backend, T: DeviceScalar> DeviceOperatorTape<B, T> {
    /// Create an empty backend-owned tape.
    pub const fn new() -> Self {
        Self {
            values: Vec::new(),
            nodes: Vec::new(),
            metrics: TapeMetrics {
                flops: 0,
                launches: 0,
                synchronisations: 0,
                transferred_bytes: 0,
                allocations: 0,
            },
            budget: None,
            poisoned: false,
        }
    }

    /// Create a tape with a checked aggregate execution budget.
    pub const fn with_budget(budget: TapeBudget) -> Self {
        let mut tape = Self::new();
        tape.budget = Some(budget);
        tape
    }

    /// Remove all recorded values and nodes while retaining vector capacity.
    pub fn clear(&mut self) {
        self.values.clear();
        self.nodes.clear();
        self.metrics = TapeMetrics {
            flops: 0,
            launches: 0,
            synchronisations: 0,
            transferred_bytes: 0,
            allocations: 0,
        };
        self.poisoned = false;
    }

    fn ensure_usable(&self, operation: &'static str) -> Result<(), TapeError> {
        if self.poisoned {
            Err(TapeError::InvalidState { operation })
        } else {
            Ok(())
        }
    }

    /// Number of backend-owned values retained by the tape.
    pub fn value_count(&self) -> usize {
        self.values.len()
    }

    /// Number of recorded linear nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Aggregate metrics, including replay/backward work and tape-owned
    /// allocations already performed by this tape.
    pub fn metrics(&self) -> TapeMetrics {
        self.metrics
    }

    /// Check the currently recorded device-tape metrics against its budget.
    pub fn check_budget(&self) -> Result<(), TapeError> {
        self.validate_budget(self.metrics)
    }

    /// Replace the device-tape budget after validating the current metrics.
    pub fn set_budget(&mut self, budget: TapeBudget) -> Result<(), TapeError> {
        self.metrics.check_budget(budget)?;
        self.budget = Some(budget);
        Ok(())
    }

    fn validate_budget(&self, metrics: TapeMetrics) -> Result<(), TapeError> {
        if let Some(budget) = self.budget {
            metrics.check_budget(budget)?;
        }
        Ok(())
    }

    /// Copy a backend value into tape-owned storage.
    pub fn record_input(
        &mut self,
        backend: &B,
        value: &B::Buffer<T>,
    ) -> Result<TapeValueId, TapeError> {
        self.ensure_usable("record_input")?;
        let mut projected = self.metrics;
        projected.allocations = projected.allocations.saturating_add(1);
        self.validate_budget(projected)?;
        let next_value_count = self.values.len().checked_add(1).ok_or_else(|| {
            TapeError::Backend(crate::BackendError::Allocation {
                reason: "device tape value count overflow".into(),
            })
        })?;
        tape_reserve(&mut self.values, next_value_count, "device tape values")?;
        let mut retained = backend
            .raw_alloc::<T>(value.len())
            .map_err(crate::BackendError::from)?;
        backend.try_copy_d2d(value, &mut retained)?;
        let id = TapeValueId(self.values.len());
        self.values.push(retained);
        self.metrics = projected;
        Ok(id)
    }

    /// Update a recorded input in place without changing its shape.
    pub fn update_value(
        &mut self,
        backend: &B,
        id: TapeValueId,
        value: &B::Buffer<T>,
    ) -> Result<(), TapeError> {
        self.ensure_usable("update_value")?;
        let retained = self
            .values
            .get_mut(id.0)
            .ok_or(TapeError::InvalidValue { value: id.0 })?;
        if retained.len() != value.len() {
            return Err(TapeError::DimensionMismatch {
                operation: "device tape update value",
                expected: retained.len(),
                actual: value.len(),
            });
        }
        if let Err(error) = backend.try_copy_d2d(value, retained) {
            self.poisoned = true;
            return Err(error.into());
        }
        Ok(())
    }

    /// Borrow a recorded backend value.
    pub fn value(&self, id: TapeValueId) -> Result<&B::Buffer<T>, TapeError> {
        self.ensure_usable("value")?;
        self.values
            .get(id.0)
            .ok_or(TapeError::InvalidValue { value: id.0 })
    }

    /// Record and immediately execute one backend linear action.
    #[allow(clippy::type_complexity)]
    pub fn record_linear<F, G>(
        &mut self,
        backend: &B,
        input: TapeValueId,
        output_len: usize,
        metadata: TapeNodeMetadata,
        forward: F,
        adjoint: G,
    ) -> Result<TapeValueId, TapeError>
    where
        F: Fn(&B, &B::Buffer<T>, &mut B::Buffer<T>) -> crate::BackendResult<()>
            + Send
            + Sync
            + 'static,
        G: Fn(
                &B,
                &B::Buffer<T>,
                &B::Buffer<T>,
                &B::Buffer<T>,
                &mut B::Buffer<T>,
            ) -> crate::BackendResult<()>
            + Send
            + Sync
            + 'static,
    {
        self.ensure_usable("record_linear")?;
        let node_metadata = metadata;
        let mut projected = self.metrics;
        projected.accumulate(node_metadata.metrics);
        // The output buffer is a persistent tape allocation, distinct from
        // temporary allocations described by the operation metadata. Keep the
        // original metadata on the node so replay/backward do not charge this
        // retained buffer again.
        projected.allocations = projected.allocations.saturating_add(1);
        self.validate_budget(projected)?;
        let next_value_count = self.values.len().checked_add(1).ok_or_else(|| {
            TapeError::Backend(crate::BackendError::Allocation {
                reason: "device tape value count overflow".into(),
            })
        })?;
        let next_node_count = self.nodes.len().checked_add(1).ok_or_else(|| {
            TapeError::Backend(crate::BackendError::Allocation {
                reason: "device tape node count overflow".into(),
            })
        })?;
        tape_reserve(&mut self.values, next_value_count, "device tape values")?;
        tape_reserve(&mut self.nodes, next_node_count, "device tape nodes")?;
        let mut output = backend
            .raw_alloc::<T>(output_len)
            .map_err(crate::BackendError::from)?;
        let input_value = self.value(input)?;
        invoke_callback(node_metadata.operation, || {
            forward(backend, input_value, &mut output).map_err(TapeError::from)
        })?;
        if output.len() != output_len {
            return Err(TapeError::DimensionMismatch {
                operation: "device tape forward output",
                expected: output_len,
                actual: output.len(),
            });
        }
        let id = TapeValueId(self.values.len());
        self.values.push(output);
        self.nodes.push(DeviceTapeNode {
            input,
            output: id,
            metadata: node_metadata,
            forward: Arc::new(forward),
            adjoint: Arc::new(adjoint),
        });
        self.metrics = projected;
        Ok(id)
    }

    /// Replay all recorded forward actions in order.
    pub fn replay(&mut self, backend: &B) -> Result<(), TapeError> {
        self.ensure_usable("replay")?;
        let mut projected = self.metrics;
        for node in &self.nodes {
            projected.accumulate(node.metadata.metrics);
        }
        self.validate_budget(projected)?;
        let mut replay_error = None;
        for node in &self.nodes {
            let (before, after) = self.values.split_at_mut(node.output.0);
            let input = &before[node.input.0];
            let output = &mut after[0];
            let operation = node.metadata.operation;
            if let Err(error) = invoke_callback(operation, || {
                (node.forward)(backend, input, output).map_err(TapeError::from)
            }) {
                // Replay writes retained values in place. Once a later node
                // fails, rolling back would require extra device allocations,
                // which is incompatible with the device-resident budget.
                // Poison the tape so partial values cannot be observed or
                // reused accidentally; `clear` is the explicit recovery path.
                replay_error = Some(error);
                break;
            }
        }
        if let Some(error) = replay_error {
            self.poisoned = true;
            return Err(error);
        }
        self.metrics = projected;
        Ok(())
    }

    /// Run reverse mode and return backend-owned gradients.
    pub fn backward(
        &mut self,
        backend: &B,
        output: TapeValueId,
        seed: &B::Buffer<T>,
    ) -> Result<DeviceTapeGradients<B, T>, TapeError> {
        self.ensure_usable("backward")?;
        let output_value = self.value(output)?;
        if output_value.len() != seed.len() {
            return Err(TapeError::DimensionMismatch {
                operation: "device tape backward seed",
                expected: output_value.len(),
                actual: seed.len(),
            });
        }

        // A tape may contain independent inputs, branches, or nodes recorded
        // after the requested output. Charge only the reverse path that can
        // actually receive a gradient. Every node has one input and one
        // uniquely-created output, so a reverse scan can track that path
        // without allocating host-side reachability metadata.
        let mut projected = self.metrics;
        let mut reachable_value = output;
        let mut temporary_allocations = 1_u64; // copied seed gradient
        for node in self.nodes.iter().rev() {
            if node.output == reachable_value {
                projected.accumulate(node.metadata.metrics);
                temporary_allocations = temporary_allocations.saturating_add(1);
                reachable_value = node.input;
            }
        }
        projected.allocations = projected.allocations.saturating_add(temporary_allocations);
        self.validate_budget(projected)?;
        let mut gradients: Vec<Option<B::Buffer<T>>> = Vec::new();
        tape_reserve(&mut gradients, self.values.len(), "device tape gradients")?;
        gradients.resize_with(self.values.len(), || None);
        let mut output_gradient = backend
            .raw_alloc::<T>(seed.len())
            .map_err(crate::BackendError::from)?;
        backend.try_copy_d2d(seed, &mut output_gradient)?;
        gradients[output.0] = Some(output_gradient);
        for node in self.nodes.iter().rev() {
            let Some(output_gradient) = gradients[node.output.0].as_ref() else {
                continue;
            };
            let input_value = &self.values[node.input.0];
            let output_value = &self.values[node.output.0];
            let mut input_gradient = backend
                .raw_alloc::<T>(input_value.len())
                .map_err(crate::BackendError::from)?;
            backend.try_scale(T::zero(), &mut input_gradient)?;
            invoke_callback(node.metadata.operation, || {
                (node.adjoint)(
                    backend,
                    input_value,
                    output_value,
                    output_gradient,
                    &mut input_gradient,
                )
                .map_err(TapeError::from)
            })?;
            if let Some(existing) = gradients[node.input.0].as_mut() {
                backend.try_axpy(T::one(), &input_gradient, existing)?;
            } else {
                gradients[node.input.0] = Some(input_gradient);
            }
        }
        self.metrics = projected;
        Ok(DeviceTapeGradients { values: gradients })
    }
}

impl<B: crate::Backend, T: DeviceScalar> Default for DeviceOperatorTape<B, T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Backend-owned reverse-mode gradients returned by [`DeviceOperatorTape`].
pub struct DeviceTapeGradients<B: crate::Backend, T: DeviceScalar> {
    values: Vec<Option<B::Buffer<T>>>,
}

impl<B: crate::Backend, T: DeviceScalar> DeviceTapeGradients<B, T> {
    /// Borrow a gradient buffer, if the value participates in the reverse pass.
    pub fn gradient(&self, value: TapeValueId) -> Result<Option<&B::Buffer<T>>, TapeError> {
        self.values
            .get(value.0)
            .map(Option::as_ref)
            .ok_or(TapeError::InvalidValue { value: value.0 })
    }

    /// Consume the gradient set and return one owned gradient buffer.
    pub fn into_gradient(self, value: TapeValueId) -> Result<Option<B::Buffer<T>>, TapeError> {
        self.values
            .into_iter()
            .nth(value.0)
            .ok_or(TapeError::InvalidValue { value: value.0 })
    }
}

fn invoke_callback<F>(operation: &'static str, callback: F) -> Result<(), TapeError>
where
    F: FnOnce() -> Result<(), TapeError>,
{
    catch_unwind(AssertUnwindSafe(callback)).map_err(|payload| TapeError::OperationFailed {
        operation,
        reason: panic_reason(payload),
    })?
}

fn panic_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "operator callback panicked".to_string()
    }
}

#[cfg(test)]
// Fixtures intentionally use the panicking convenience constructors on
// tiny fixed matrices; production paths must use the `try_*` variants.
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::Backend;
    use num_complex::Complex64;

    #[test]
    fn tape_capacity_failure_is_typed() {
        let mut values = Vec::<u8>::new();
        assert!(matches!(
            tape_reserve(&mut values, usize::MAX, "tape metadata"),
            Err(TapeError::Backend(crate::BackendError::Allocation { .. }))
        ));
    }

    #[test]
    fn records_replays_and_differentiates_real_matrix_free_action() {
        let mut tape = OperatorTape::<f64>::new();
        let input = tape.record_input(&[1.0, 2.0]);
        let matrix = [[2.0, 1.0], [3.0, 4.0]];
        let output = tape
            .record_linear(
                input,
                2,
                TapeNodeMetadata::new("dense_matvec", TapeMetrics::new(8, 1, 0, 0)),
                move |x, y| {
                    y[0] = matrix[0][0] * x[0] + matrix[0][1] * x[1];
                    y[1] = matrix[1][0] * x[0] + matrix[1][1] * x[1];
                    Ok(())
                },
                move |_x, _y, dy, dx| {
                    dx[0] = matrix[0][0] * dy[0] + matrix[1][0] * dy[1];
                    dx[1] = matrix[0][1] * dy[0] + matrix[1][1] * dy[1];
                    Ok(())
                },
            )
            .expect("recording should succeed");
        assert_eq!(tape.value(output).expect("forward value"), &[4.0, 11.0]);
        assert_eq!(tape.metrics(), TapeMetrics::with_allocations(8, 1, 0, 0, 1));

        let gradients = tape.backward(output, &[1.0, 0.0]).expect("backward pass");
        assert_eq!(
            gradients.gradient(input).expect("input gradient"),
            Some(&[2.0, 1.0][..])
        );
        tape.update_value(input, &[2.0, 3.0])
            .expect("same-shaped input update");
        tape.replay().expect("replay should succeed");
        assert_eq!(tape.value(output).expect("replayed value"), &[7.0, 18.0]);
    }

    #[test]
    fn hermitian_callback_supports_complex_operators() {
        let mut tape = OperatorTape::<Complex64>::new();
        let input = tape.record_input(&[Complex64::new(1.0, -1.0), Complex64::new(2.0, 0.5)]);
        let a00 = Complex64::new(2.0, 1.0);
        let a01 = Complex64::new(1.0, -2.0);
        let a10 = Complex64::new(0.0, 3.0);
        let a11 = Complex64::new(4.0, -1.0);
        let output = tape
            .record_linear(
                input,
                2,
                TapeNodeMetadata::new("complex_matvec", TapeMetrics::default()),
                move |x, y| {
                    y[0] = a00 * x[0] + a01 * x[1];
                    y[1] = a10 * x[0] + a11 * x[1];
                    Ok(())
                },
                move |_x, _y, dy, dx| {
                    dx[0] = a00.conj() * dy[0] + a10.conj() * dy[1];
                    dx[1] = a01.conj() * dy[0] + a11.conj() * dy[1];
                    Ok(())
                },
            )
            .expect("recording should succeed");
        let gradients = tape
            .backward(
                output,
                &[Complex64::new(1.0, 0.5), Complex64::new(-2.0, 1.0)],
            )
            .expect("complex backward pass");
        let expected = [
            a00.conj() * Complex64::new(1.0, 0.5) + a10.conj() * Complex64::new(-2.0, 1.0),
            a01.conj() * Complex64::new(1.0, 0.5) + a11.conj() * Complex64::new(-2.0, 1.0),
        ];
        for (actual, expected) in gradients
            .gradient(input)
            .expect("input gradient")
            .expect("connected input")
            .iter()
            .zip(expected)
        {
            assert!((*actual - expected).norm() < 1.0e-12);
        }
    }

    #[test]
    fn tape_metrics_enforce_device_resident_budget() {
        TapeMetrics::new(32, 2, 1, 0)
            .require_device_resident()
            .expect("zero-transfer trace should be accepted");

        let error = TapeMetrics::new(32, 2, 1, 16)
            .require_device_resident()
            .expect_err("host/device traffic must fail the device-resident budget");
        assert!(matches!(
            error,
            TapeError::BudgetExceeded {
                metric: "transferred_bytes",
                limit: 0,
                actual: 16,
            }
        ));

        let budget = TapeBudget {
            max_flops: 8,
            max_launches: 2,
            max_synchronisations: 1,
            max_transferred_bytes: 32,
            max_allocations: 4,
        };
        let error = TapeMetrics::new(9, 2, 1, 0)
            .check_budget(budget)
            .expect_err("an over-budget operation must be rejected");
        assert!(matches!(
            error,
            TapeError::BudgetExceeded {
                metric: "flops",
                limit: 8,
                actual: 9,
            }
        ));

        let allocation_error = TapeMetrics::with_allocations(0, 0, 0, 0, 1)
            .require_device_resident()
            .expect_err("device-resident traces must not allocate");
        assert!(matches!(
            allocation_error,
            TapeError::BudgetExceeded {
                metric: "allocations",
                limit: 0,
                actual: 1,
            }
        ));
    }

    #[test]
    fn tape_budget_rejects_host_backed_recording_for_device_resident_mode() {
        let mut tape = OperatorTape::<f64>::with_budget(TapeBudget::device_resident());
        let input = tape.record_input(&[1.0]);
        let error = tape
            .record_linear(
                input,
                1,
                TapeNodeMetadata::new("host_retained", TapeMetrics::default()),
                |x, y| {
                    y[0] = x[0];
                    Ok(())
                },
                |_x, _y, dy, dx| {
                    dx[0] = dy[0];
                    Ok(())
                },
            )
            .expect_err("host-backed tape recording allocates retained state");
        assert!(matches!(
            error,
            TapeError::BudgetExceeded {
                metric: "allocations",
                limit: 0,
                actual: 1,
            }
        ));
        assert_eq!(tape.node_count(), 0);
    }

    #[test]
    fn callback_panics_and_bad_seeds_are_typed_errors() {
        let mut tape = OperatorTape::<f64>::new();
        let input = tape.record_input(&[1.0]);
        let error = tape
            .record_linear(
                input,
                1,
                TapeNodeMetadata::new("panic_op", TapeMetrics::default()),
                |_x, _y| panic!("simulated operator failure"),
                |_x, _y, _dy, _dx| Ok(()),
            )
            .expect_err("callback panic must be contained");
        assert!(matches!(
            error,
            TapeError::OperationFailed {
                operation: "panic_op",
                ..
            }
        ));

        let output = tape.record_input(&[1.0, 2.0]);
        let error = tape
            .backward(output, &[1.0])
            .expect_err("seed shape must be checked");
        assert!(matches!(error, TapeError::DimensionMismatch { .. }));
    }

    #[test]
    fn device_tape_replays_and_differentiates_without_host_values() {
        let backend = crate::CpuBackend::new();
        let mut input_buffer = backend.raw_alloc::<f64>(2).expect("input allocation");
        backend
            .try_copy_h2d(&[1.5, -2.0], &mut input_buffer)
            .expect("input upload");
        let mut tape = DeviceOperatorTape::<crate::CpuBackend, f64>::with_budget(TapeBudget {
            max_flops: u64::MAX,
            max_launches: u64::MAX,
            max_synchronisations: u64::MAX,
            max_transferred_bytes: 0,
            max_allocations: 4,
        });
        let input = tape
            .record_input(&backend, &input_buffer)
            .expect("record backend input");
        let output = tape
            .record_linear(
                &backend,
                input,
                2,
                TapeNodeMetadata::new("scale_two", TapeMetrics::new(4, 1, 0, 0)),
                |backend, x, y| {
                    backend.try_copy_d2d(x, y)?;
                    backend.try_scale(2.0, y)
                },
                |backend, _x, _y, dy, dx| backend.try_axpy(2.0, dy, dx),
            )
            .expect("record backend action");
        assert_eq!(tape.value_count(), 2);
        assert_eq!(tape.node_count(), 1);
        assert_eq!(tape.metrics().transferred_bytes, 0);
        assert_eq!(tape.metrics().allocations, 2);

        let mut actual = [0.0; 2];
        backend
            .try_copy_d2h(tape.value(output).expect("output value"), &mut actual)
            .expect("output download");
        assert_eq!(actual, [3.0, -4.0]);

        let mut updated = backend.raw_alloc::<f64>(2).expect("updated allocation");
        backend
            .try_copy_h2d(&[2.0, 3.0], &mut updated)
            .expect("updated upload");
        tape.update_value(&backend, input, &updated)
            .expect("same-shaped device update");
        tape.replay(&backend).expect("device replay");
        assert_eq!(tape.metrics().flops, 8);
        assert_eq!(tape.metrics().launches, 2);
        backend
            .try_copy_d2h(tape.value(output).expect("replayed output"), &mut actual)
            .expect("replayed output download");
        assert_eq!(actual, [4.0, 6.0]);

        let mut seed = backend.raw_alloc::<f64>(2).expect("seed allocation");
        backend
            .try_copy_h2d(&[1.0, -0.5], &mut seed)
            .expect("seed upload");
        let gradients = tape
            .backward(&backend, output, &seed)
            .expect("device backward");
        assert_eq!(tape.metrics().flops, 12);
        assert_eq!(tape.metrics().launches, 3);
        assert_eq!(tape.metrics().allocations, 4);
        let mut gradient = [0.0; 2];
        backend
            .try_copy_d2h(
                gradients
                    .gradient(input)
                    .expect("gradient handle")
                    .expect("connected gradient"),
                &mut gradient,
            )
            .expect("gradient download");
        assert_eq!(gradient, [2.0, -1.0]);
    }

    #[test]
    fn device_tape_preserves_backend_failures_without_recording_nodes() {
        let backend = crate::CpuBackend::new();
        let mut input = backend.raw_alloc::<f64>(1).expect("input allocation");
        backend
            .try_copy_h2d(&[1.0], &mut input)
            .expect("input upload");
        let mut tape = DeviceOperatorTape::<crate::CpuBackend, f64>::new();
        let value = tape.record_input(&backend, &input).expect("record input");
        let metrics_before = tape.metrics();
        let error = tape
            .record_linear(
                &backend,
                value,
                1,
                TapeNodeMetadata::new("forced_failure", TapeMetrics::default()),
                |_backend, _input, _output| {
                    Err(crate::BackendError::OperationFailed {
                        operation: "forced_failure",
                        reason: "injected".into(),
                    })
                },
                |_backend, _input, _output, _dy, _dx| Ok(()),
            )
            .expect_err("backend failure must be typed");
        assert!(matches!(error, TapeError::Backend(_)));
        assert_eq!(tape.node_count(), 0);
        assert_eq!(tape.value_count(), 1);
        assert_eq!(tape.metrics(), metrics_before);
    }

    #[test]
    fn device_tape_callback_panics_are_typed_and_transactional() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let backend = crate::CpuBackend::new();
        let mut input = backend.raw_alloc::<f64>(1).expect("input allocation");
        backend
            .try_copy_h2d(&[1.0], &mut input)
            .expect("input upload");

        let mut forward_tape = DeviceOperatorTape::<crate::CpuBackend, f64>::new();
        let value = forward_tape
            .record_input(&backend, &input)
            .expect("record input");
        let metrics_before = forward_tape.metrics();
        let error = forward_tape
            .record_linear(
                &backend,
                value,
                1,
                TapeNodeMetadata::new("device_forward_panic", TapeMetrics::default()),
                |_backend, _input, _output| panic!("forward callback panic"),
                |_backend, _input, _output, _dy, _dx| Ok(()),
            )
            .expect_err("forward callback panic must be contained");
        assert!(matches!(
            error,
            TapeError::OperationFailed {
                operation: "device_forward_panic",
                ..
            }
        ));
        assert_eq!(forward_tape.node_count(), 0);
        assert_eq!(forward_tape.metrics(), metrics_before);

        let calls = Arc::new(AtomicUsize::new(0));
        let replay_calls = Arc::clone(&calls);
        let mut replay_tape = DeviceOperatorTape::<crate::CpuBackend, f64>::new();
        let value = replay_tape
            .record_input(&backend, &input)
            .expect("record replay input");
        let output = replay_tape
            .record_linear(
                &backend,
                value,
                1,
                TapeNodeMetadata::new("device_replay_panic", TapeMetrics::default()),
                move |backend, source, destination| {
                    if replay_calls.fetch_add(1, Ordering::SeqCst) != 0 {
                        panic!("replay callback panic");
                    }
                    backend.try_copy_d2d(source, destination)
                },
                |_backend, _input, _output, _dy, _dx| Ok(()),
            )
            .expect("initial forward callback should succeed");
        let metrics_before = replay_tape.metrics();
        let error = replay_tape
            .replay(&backend)
            .expect_err("replay callback panic must be contained");
        assert!(matches!(
            error,
            TapeError::OperationFailed {
                operation: "device_replay_panic",
                ..
            }
        ));
        assert_eq!(replay_tape.node_count(), 1);
        assert_eq!(replay_tape.value_count(), 2);
        assert_eq!(replay_tape.metrics(), metrics_before);
        assert_eq!(output.index(), 1);

        assert!(matches!(
            replay_tape.value(output),
            Err(TapeError::InvalidState { operation: "value" })
        ));
        replay_tape.clear();
        assert_eq!(replay_tape.value_count(), 0);

        let mut backward_tape = DeviceOperatorTape::<crate::CpuBackend, f64>::new();
        let value = backward_tape
            .record_input(&backend, &input)
            .expect("record backward input");
        let output = backward_tape
            .record_linear(
                &backend,
                value,
                1,
                TapeNodeMetadata::new("device_adjoint_panic", TapeMetrics::default()),
                |backend, source, destination| backend.try_copy_d2d(source, destination),
                |_backend, _input, _output, _dy, _dx| panic!("adjoint callback panic"),
            )
            .expect("forward callback should succeed");
        let mut seed = backend.raw_alloc::<f64>(1).expect("seed allocation");
        backend
            .try_copy_h2d(&[1.0], &mut seed)
            .expect("seed upload");
        let metrics_before = backward_tape.metrics();
        let error = match backward_tape.backward(&backend, output, &seed) {
            Ok(_) => panic!("adjoint callback panic must be contained"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            TapeError::OperationFailed {
                operation: "device_adjoint_panic",
                ..
            }
        ));
        assert_eq!(backward_tape.metrics(), metrics_before);
    }

    #[test]
    fn device_tape_budget_updates_are_checked_and_transactional() {
        let backend = crate::CpuBackend::new();
        let input = backend.raw_alloc::<f64>(1).expect("input allocation");
        let mut tape = DeviceOperatorTape::<crate::CpuBackend, f64>::new();
        tape.record_input(&backend, &input).expect("record input");

        let permissive = TapeBudget {
            max_flops: u64::MAX,
            max_launches: u64::MAX,
            max_synchronisations: u64::MAX,
            max_transferred_bytes: u64::MAX,
            max_allocations: 1,
        };
        tape.set_budget(permissive).expect("budget should fit");
        assert!(tape.check_budget().is_ok());

        let rejected = tape
            .set_budget(TapeBudget {
                max_allocations: 0,
                ..permissive
            })
            .expect_err("tightening below current allocations must fail");
        assert!(matches!(
            rejected,
            TapeError::BudgetExceeded {
                metric: "allocations",
                limit: 0,
                actual: 1,
            }
        ));

        // The failed update must not replace the previous usable budget.
        assert!(tape.check_budget().is_ok());
    }

    #[test]
    fn device_tape_backward_budgets_only_the_reachable_branch() {
        let backend = crate::CpuBackend::new();
        let mut input = backend.raw_alloc::<f64>(1).expect("input allocation");
        backend
            .try_copy_h2d(&[2.0], &mut input)
            .expect("input upload");
        let mut tape = DeviceOperatorTape::<crate::CpuBackend, f64>::with_budget(TapeBudget {
            max_flops: 30,
            max_launches: 3,
            max_synchronisations: 0,
            max_transferred_bytes: 0,
            max_allocations: 5,
        });
        let input = tape.record_input(&backend, &input).expect("record input");
        let metadata = TapeNodeMetadata::new("branch", TapeMetrics::new(10, 1, 0, 0));
        let first = tape
            .record_linear(
                &backend,
                input,
                1,
                metadata,
                |backend, source, destination| backend.try_copy_d2d(source, destination),
                |backend, _input, _output, gradient, input_gradient| {
                    backend.try_axpy(1.0, gradient, input_gradient)
                },
            )
            .expect("record first branch");
        let _second = tape
            .record_linear(
                &backend,
                input,
                1,
                metadata,
                |backend, source, destination| backend.try_copy_d2d(source, destination),
                |backend, _input, _output, gradient, input_gradient| {
                    backend.try_axpy(1.0, gradient, input_gradient)
                },
            )
            .expect("record disconnected branch");
        let mut seed = backend.raw_alloc::<f64>(1).expect("seed allocation");
        backend
            .try_copy_h2d(&[3.0], &mut seed)
            .expect("seed upload");

        let gradients = tape
            .backward(&backend, first, &seed)
            .expect("disconnected branch must not consume reverse budget");

        assert_eq!(tape.metrics().flops, 30);
        assert_eq!(tape.metrics().launches, 3);
        assert_eq!(tape.metrics().allocations, 5);
        let mut gradient = [0.0];
        backend
            .try_copy_d2h(
                gradients
                    .gradient(input)
                    .expect("gradient handle")
                    .expect("reachable input gradient"),
                &mut gradient,
            )
            .expect("gradient download");
        assert_eq!(gradient, [3.0]);
    }
}
