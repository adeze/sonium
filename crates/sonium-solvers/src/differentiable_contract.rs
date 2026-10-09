//! Common forward/adjoint contract for differentiable solver frontends.
//!
//! The solver families intentionally keep their native result types: a
//! transient solve returns a trajectory, an eigen solve returns spectral
//! data, and a block solve returns a dense multi-RHS matrix. This trait
//! standardises the lifecycle and adjoint direction without erasing those
//! useful types. [`ClosureDifferentiableSolve`] is the bridge for callers
//! whose forward state or adjoint seed is family-specific.

use crate::differentiable::{AdjointSolve, DifferentiableLinearSolve};
use crate::eigen::{ComplexTargetedEigenResult, LobpcgResult, TargetedEigenResult};
use crate::iterative::{BlockFgmresSolution, GmresSolution};
use crate::time_integration::{NewmarkAdjointResult, NewmarkResult};
use crate::traits::{
    ComplexField, LinearOperator, Preconditioner, SolverError, SolverResultContract,
};
use half::f16;
use num_complex::{Complex32, Complex64};
use num_traits::{Float, Zero};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Solver family represented by a differentiable contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferentiableSystemKind {
    /// One steady linear system with one or more RHS solves.
    Steady,
    /// A time-discrete transient trajectory and its reverse pass.
    Transient,
    /// An eigenvalue/eigenvector or generalized pencil sensitivity.
    Eigen,
    /// A coupled/block system with a dense multi-RHS state.
    CoupledBlock,
}

/// Cooperative cancellation state shared by differentiable callbacks.
///
/// Cancellation is deliberately cooperative: a callback may poll the token
/// while doing long-running work, and the contract boundary checks it both
/// before dispatch and after the callback returns.  A cancelled operation is
/// reported as a typed `SolverError` before any new callback is entered.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    /// Create a token in the active state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cooperative cancellation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    /// Fail with a typed contract error when cancellation was requested.
    pub fn check(&self, operation: &'static str) -> Result<(), SolverError> {
        if self.is_cancelled() {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "differentiable operation cancelled".into(),
            });
        }
        Ok(())
    }
}

/// Seed for a steady linear adjoint solve in the common contract.
#[derive(Debug, Clone)]
pub struct SteadyAdjointSeed<T> {
    /// Right-hand side of the transpose/Hermitian system.
    pub rhs: Vec<T>,
    /// Adjoint direction required by the objective.
    pub kind: AdjointSolve,
}

impl<T> SteadyAdjointSeed<T> {
    /// Construct a steady adjoint request.
    pub fn new(rhs: Vec<T>, kind: AdjointSolve) -> Self {
        Self { rhs, kind }
    }
}

/// A value that can be checked at a differentiable-contract boundary.
///
/// The generic closure contract deliberately supports family-specific input
/// and seed types, so it cannot infer dimensions from those types. Implement
/// this trait for a value with a meaningful vector shape and use
/// [`ContractValidation::dimension_checked`] when the dimensions are part of
/// the contract. Native solver result types implement it below; callers with
/// opaque state can instead provide explicit callbacks.
pub trait DifferentiableContractValue {
    /// Return the primary vector dimension, when the value has one.
    fn contract_dimension(&self) -> Option<usize>;

    /// Validate dimensions and finite numerical data intrinsic to this value.
    fn validate_contract_value(&self, role: &'static str) -> Result<(), SolverError>;
}

/// Optional exact dimensions for a checked differentiable contract.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ContractDimensions {
    /// Expected forward-input dimension.
    pub input: Option<usize>,
    /// Expected retained forward-state dimension.
    pub forward: Option<usize>,
    /// Expected reverse/terminal seed dimension.
    pub seed: Option<usize>,
    /// Expected reverse/adjoint-state dimension.
    pub adjoint: Option<usize>,
}

impl ContractDimensions {
    /// Set the expected dimensions for all four lifecycle values.
    pub const fn new(
        input: Option<usize>,
        forward: Option<usize>,
        seed: Option<usize>,
        adjoint: Option<usize>,
    ) -> Self {
        Self {
            input,
            forward,
            seed,
            adjoint,
        }
    }
}

impl<T: ComplexField> DifferentiableContractValue for [T] {
    fn contract_dimension(&self) -> Option<usize> {
        Some(self.len())
    }

    fn validate_contract_value(&self, role: &'static str) -> Result<(), SolverError> {
        if self.iter().any(|value| !value.norm().is_finite()) {
            return Err(contract_value_error(role, "contains NaN or infinity"));
        }
        Ok(())
    }
}

impl<T: ComplexField> DifferentiableContractValue for Vec<T> {
    fn contract_dimension(&self) -> Option<usize> {
        self.as_slice().contract_dimension()
    }

    fn validate_contract_value(&self, role: &'static str) -> Result<(), SolverError> {
        self.as_slice().validate_contract_value(role)
    }
}

macro_rules! impl_scalar_contract_value {
    ($($scalar:ty),+ $(,)?) => {
        $(
            impl DifferentiableContractValue for $scalar {
                fn contract_dimension(&self) -> Option<usize> {
                    Some(1)
                }

                fn validate_contract_value(
                    &self,
                    role: &'static str,
                ) -> Result<(), SolverError> {
                    validate_finite_scalar(self, role)
                }
            }
        )+
    };
}

impl_scalar_contract_value!(f16, f32, f64, Complex32, Complex64);

impl<T: ComplexField> DifferentiableContractValue for SteadyAdjointSeed<T> {
    fn contract_dimension(&self) -> Option<usize> {
        self.rhs.contract_dimension()
    }

    fn validate_contract_value(&self, role: &'static str) -> Result<(), SolverError> {
        self.rhs.validate_contract_value(role)
    }
}

impl<T: ComplexField> DifferentiableContractValue for GmresSolution<T> {
    fn contract_dimension(&self) -> Option<usize> {
        Some(self.x.len())
    }

    fn validate_contract_value(&self, role: &'static str) -> Result<(), SolverError> {
        SolverResultContract::validate_contract(self).map_err(|error| {
            contract_value_error(
                role,
                format!("GMRES result violates iterative result contract: {error}"),
            )
        })?;
        self.x.validate_contract_value(role)?;
        if !self.residual.is_finite()
            || !self.estimated_residual.is_finite()
            || !self.backward_error.is_finite()
        {
            return Err(contract_value_error(
                role,
                "residual metadata contains NaN or infinity",
            ));
        }
        Ok(())
    }
}

impl<T: ComplexField> DifferentiableContractValue for BlockFgmresSolution<T> {
    fn contract_dimension(&self) -> Option<usize> {
        Some(self.x.nrows())
    }

    fn validate_contract_value(&self, role: &'static str) -> Result<(), SolverError> {
        SolverResultContract::validate_contract(self).map_err(|error| {
            contract_value_error(
                role,
                format!("block FGMRES result violates iterative result contract: {error}"),
            )
        })?;
        self.x.as_slice().validate_contract_value(role)?;
        let rhs_count = self.x.ncols();
        if self.residuals.len() != rhs_count
            || self.true_residuals.len() != rhs_count
            || self.backward_errors.len() != rhs_count
            || self.attainable_accuracy.len() != rhs_count
        {
            return Err(contract_value_error(
                role,
                "per-RHS metadata length does not match the block width",
            ));
        }
        if self
            .residuals
            .iter()
            .chain(self.true_residuals.iter())
            .chain(self.backward_errors.iter())
            .any(|value| !value.is_finite())
        {
            return Err(contract_value_error(
                role,
                "residual metadata contains NaN or infinity",
            ));
        }
        Ok(())
    }
}

/// Native real shift-invert eigen results participate in the same checked
/// differentiable lifecycle as vector and block solver states.  The result
/// contract already owns the shape/status invariants; this adapter adds the
/// retained eigenvector-row dimension used to relate a reverse seed to the
/// forward state.
impl DifferentiableContractValue for TargetedEigenResult {
    fn contract_dimension(&self) -> Option<usize> {
        Some(self.eigenvectors.nrows())
    }

    fn validate_contract_value(&self, role: &'static str) -> Result<(), SolverError> {
        self.validate_contract().map_err(|error| {
            contract_value_error(
                role,
                format!("native real eigen result violates result contract: {error}"),
            )
        })
    }
}

/// Native complex shift-invert Arnoldi results expose the same typed
/// differentiable-value boundary as their real counterpart.
impl DifferentiableContractValue for ComplexTargetedEigenResult {
    fn contract_dimension(&self) -> Option<usize> {
        Some(self.eigenvectors.nrows())
    }

    fn validate_contract_value(&self, role: &'static str) -> Result<(), SolverError> {
        self.validate_contract().map_err(|error| {
            contract_value_error(
                role,
                format!("native complex eigen result violates result contract: {error}"),
            )
        })
    }
}

/// LOBPCG's native result is also a retained eigenvector state.  Reusing its
/// result contract keeps differentiable validation aligned with solver
/// status/accuracy semantics instead of duplicating those checks here.
impl DifferentiableContractValue for LobpcgResult {
    fn contract_dimension(&self) -> Option<usize> {
        Some(self.eigenvectors.nrows())
    }

    fn validate_contract_value(&self, role: &'static str) -> Result<(), SolverError> {
        self.validate_contract().map_err(|error| {
            contract_value_error(
                role,
                format!("native LOBPCG result violates result contract: {error}"),
            )
        })
    }
}

impl<T: ComplexField> DifferentiableContractValue for NewmarkResult<T> {
    fn contract_dimension(&self) -> Option<usize> {
        Some(self.displacement.len())
    }

    fn validate_contract_value(&self, role: &'static str) -> Result<(), SolverError> {
        let dimension = self.displacement.len();
        if self.velocity.len() != dimension || self.acceleration.len() != dimension {
            return Err(contract_value_error(
                role,
                "displacement, velocity, and acceleration dimensions differ",
            ));
        }
        validate_newmark_history(&self.displacement_history, dimension, role)?;
        validate_newmark_history(&self.velocity_history, dimension, role)?;
        validate_newmark_history(&self.acceleration_history, dimension, role)?;
        self.displacement.validate_contract_value(role)?;
        self.velocity.validate_contract_value(role)?;
        self.acceleration.validate_contract_value(role)?;
        if !self.final_time.is_finite() {
            return Err(contract_value_error(role, "final time is not finite"));
        }
        Ok(())
    }
}

impl<T: ComplexField> DifferentiableContractValue for NewmarkAdjointResult<T> {
    fn contract_dimension(&self) -> Option<usize> {
        Some(self.initial_displacement_gradient.len())
    }

    fn validate_contract_value(&self, role: &'static str) -> Result<(), SolverError> {
        let dimension = self.initial_displacement_gradient.len();
        if self.initial_velocity_gradient.len() != dimension
            || self.initial_acceleration_gradient.len() != dimension
            || self
                .force_gradient
                .iter()
                .any(|gradient| gradient.len() != dimension)
        {
            return Err(contract_value_error(
                role,
                "adjoint gradient dimensions differ",
            ));
        }
        self.initial_displacement_gradient
            .validate_contract_value(role)?;
        self.initial_velocity_gradient
            .validate_contract_value(role)?;
        self.initial_acceleration_gradient
            .validate_contract_value(role)?;
        for gradient in &self.force_gradient {
            gradient.validate_contract_value(role)?;
        }
        Ok(())
    }
}

fn validate_newmark_history<T: ComplexField>(
    history: &Option<Vec<Vec<T>>>,
    dimension: usize,
    role: &'static str,
) -> Result<(), SolverError> {
    if let Some(history) = history {
        if history.iter().any(|state| state.len() != dimension) {
            return Err(contract_value_error(
                role,
                "history state dimension does not match the final state",
            ));
        }
        for state in history {
            state.validate_contract_value(role)?;
        }
    }
    Ok(())
}

fn contract_value_error(role: &'static str, reason: impl Into<String>) -> SolverError {
    SolverError::OperationFailed {
        operation: "differentiable contract validation",
        reason: format!("{role}: {}", reason.into()),
    }
}

fn validate_expected_dimension<V: ?Sized + DifferentiableContractValue>(
    value: &V,
    role: &'static str,
    expected: Option<usize>,
) -> Result<(), SolverError> {
    value.validate_contract_value(role)?;
    if let Some(expected) = expected
        && value.contract_dimension() != Some(expected)
    {
        return Err(contract_value_error(
            role,
            format!(
                "dimension mismatch: expected {expected}, got {:?}",
                value.contract_dimension()
            ),
        ));
    }
    Ok(())
}

fn validate_finite_scalar<T: ComplexField>(
    value: &T,
    role: &'static str,
) -> Result<(), SolverError> {
    if !value.norm().is_finite() {
        return Err(contract_value_error(role, "contains NaN or infinity"));
    }
    Ok(())
}

/// Validation callbacks for a closure-backed differentiable contract.
///
/// Each callback runs inside a panic boundary. [`Self::none`] preserves the
/// historical unchecked closure behaviour; [`Self::dimension_checked`] is
/// available when all four values implement [`DifferentiableContractValue`].
type ContractValidator<T> = Arc<dyn Fn(&T) -> Result<(), SolverError>>;

type ContractRelation<Fwd, Seed, Adj> = Arc<dyn Fn(&Fwd, &Seed, &Adj) -> Result<(), SolverError>>;

pub struct ContractValidation<I: ?Sized, Fwd, Seed: ?Sized, Adj> {
    input: ContractValidator<I>,
    forward: ContractValidator<Fwd>,
    seed: ContractValidator<Seed>,
    adjoint: ContractValidator<Adj>,
    relation: Option<ContractRelation<Fwd, Seed, Adj>>,
}

impl<I: ?Sized, Fwd, Seed: ?Sized, Adj> ContractValidation<I, Fwd, Seed, Adj> {
    /// Construct explicit validation callbacks for family-specific values.
    pub fn callbacks<FI, FF, FS, FA>(input: FI, forward: FF, seed: FS, adjoint: FA) -> Self
    where
        FI: Fn(&I) -> Result<(), SolverError> + 'static,
        FF: Fn(&Fwd) -> Result<(), SolverError> + 'static,
        FS: Fn(&Seed) -> Result<(), SolverError> + 'static,
        FA: Fn(&Adj) -> Result<(), SolverError> + 'static,
    {
        Self {
            input: Arc::new(input),
            forward: Arc::new(forward),
            seed: Arc::new(seed),
            adjoint: Arc::new(adjoint),
            relation: None,
        }
    }
}

impl<I, Fwd, Seed, Adj> ContractValidation<I, Fwd, Seed, Adj>
where
    I: ?Sized + DifferentiableContractValue,
    Fwd: DifferentiableContractValue,
    Seed: ?Sized + DifferentiableContractValue,
    Adj: DifferentiableContractValue,
{
    /// Validate finite data and intrinsic dimensions for all contract values.
    pub fn dimension_checked() -> Self {
        let mut validation = Self::callbacks(
            |value| value.validate_contract_value("forward input"),
            |value| value.validate_contract_value("forward state"),
            |value| value.validate_contract_value("adjoint seed"),
            |value| value.validate_contract_value("adjoint state"),
        );
        validation.relation = Some(Arc::new(|forward, seed, adjoint| {
            let forward_dimension = forward.contract_dimension();
            if let (Some(expected), Some(actual)) = (forward_dimension, seed.contract_dimension())
                && expected != actual
            {
                return Err(contract_value_error(
                    "adjoint seed",
                    format!(
                        "dimension mismatch with forward state: expected {expected}, got {actual}"
                    ),
                ));
            }
            if let (Some(expected), Some(actual)) =
                (forward_dimension, adjoint.contract_dimension())
                && expected != actual
            {
                return Err(contract_value_error(
                    "adjoint state",
                    format!(
                        "dimension mismatch with forward state: expected {expected}, got {actual}"
                    ),
                ));
            }
            Ok(())
        }));
        validation
    }

    /// Validate finite data and enforce caller-declared lifecycle dimensions.
    pub fn exact_dimensions(dimensions: ContractDimensions) -> Self {
        let mut validation = Self::callbacks(
            move |value| validate_expected_dimension(value, "forward input", dimensions.input),
            move |value| validate_expected_dimension(value, "forward state", dimensions.forward),
            move |value| validate_expected_dimension(value, "adjoint seed", dimensions.seed),
            move |value| validate_expected_dimension(value, "adjoint state", dimensions.adjoint),
        );
        validation.relation = Some(Arc::new(|forward, seed, adjoint| {
            let forward_dimension = forward.contract_dimension();
            if let (Some(expected), Some(actual)) = (forward_dimension, seed.contract_dimension())
                && expected != actual
            {
                return Err(contract_value_error(
                    "adjoint seed",
                    format!(
                        "dimension mismatch with forward state: expected {expected}, got {actual}"
                    ),
                ));
            }
            if let (Some(expected), Some(actual)) =
                (forward_dimension, adjoint.contract_dimension())
                && expected != actual
            {
                return Err(contract_value_error(
                    "adjoint state",
                    format!(
                        "dimension mismatch with forward state: expected {expected}, got {actual}"
                    ),
                ));
            }
            Ok(())
        }));
        validation
    }
}

/// Reusable forward/adjoint lifecycle shared by solver families.
pub trait DifferentiableSolveContract {
    /// Input accepted by the forward solve. This may be a slice or a
    /// family-specific configuration object.
    type Input: ?Sized;
    /// Forward state retained for the adjoint pass.
    type Forward;
    /// Terminal/objective seed accepted by the reverse pass.
    type Seed: ?Sized;
    /// Adjoint result returned by the reverse pass.
    type Adjoint;

    /// Identify the solver family for diagnostics and QA dispatch.
    fn system_kind(&self) -> DifferentiableSystemKind;

    /// Execute the reusable forward operation.
    fn solve_forward(&self, input: &Self::Input) -> Result<Self::Forward, SolverError>;

    /// Execute forward work with cooperative cancellation checks.
    fn solve_forward_cancellable(
        &self,
        input: &Self::Input,
        cancellation: &CancellationToken,
    ) -> Result<Self::Forward, SolverError>
    where
        Self: Sized,
    {
        cancellation.check("differentiable forward")?;
        let forward = self.solve_forward(input)?;
        cancellation.check("differentiable forward")?;
        Ok(forward)
    }

    /// Execute the exact transpose/Hermitian or reverse-mode operation for a
    /// retained forward state.
    fn solve_adjoint(
        &self,
        forward: &Self::Forward,
        seed: &Self::Seed,
    ) -> Result<Self::Adjoint, SolverError>;

    /// Execute adjoint work with cooperative cancellation checks.
    fn solve_adjoint_cancellable(
        &self,
        forward: &Self::Forward,
        seed: &Self::Seed,
        cancellation: &CancellationToken,
    ) -> Result<Self::Adjoint, SolverError>
    where
        Self: Sized,
    {
        cancellation.check("differentiable adjoint")?;
        let adjoint = self.solve_adjoint(forward, seed)?;
        cancellation.check("differentiable adjoint")?;
        Ok(adjoint)
    }
}

/// Report from a deterministic forward replay check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayContractReport {
    /// Number of forward evaluations compared.
    pub evaluations: usize,
    /// Whether the replayed state matched the original state.
    pub deterministic: bool,
}

/// Re-run a contract's forward callback and verify deterministic replay.
///
/// Checkpoint/replay systems must not silently accept a different forward
/// state: adjoint calculations would otherwise consume a state that was not
/// the one used by the original primal solve. The equality callback is
/// deliberately family-specific so callers can compare trajectories,
/// eigenvectors, block matrices, or richer state records without lossy
/// serialization.
pub fn verify_forward_replay<C, Equal>(
    contract: &C,
    input: &C::Input,
    equal: Equal,
) -> Result<ReplayContractReport, SolverError>
where
    C: DifferentiableSolveContract,
    Equal: Fn(&C::Forward, &C::Forward) -> bool,
{
    let first = contract.solve_forward(input)?;
    let replay = contract.solve_forward(input)?;
    let deterministic = catch_unwind(AssertUnwindSafe(|| equal(&first, &replay)))
        .map_err(|payload| callback_panic("differentiable replay comparison", payload))?;
    if !deterministic {
        return Err(SolverError::OperationFailed {
            operation: "differentiable replay",
            reason: "forward replay state differs from the original state".into(),
        });
    }
    Ok(ReplayContractReport {
        evaluations: 2,
        deterministic,
    })
}

/// Report from a checked forward checkpoint/replay cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointReplayReport {
    /// Number of forward evaluations compared (original and replay).
    pub evaluations: usize,
    /// Number of bytes emitted by the checkpoint encoder.
    pub checkpoint_bytes: usize,
    /// Whether decoded and replayed states matched the original state.
    pub deterministic: bool,
}

/// Encode a forward state, decode it, and verify deterministic replay.
///
/// This is deliberately generic over the wire representation so transient,
/// eigen, and block solvers can provide lossless state codecs without forcing
/// a serialization dependency into the lifecycle contract. Empty payloads,
/// callback panics, decode failures, state mismatches, and cancellation are
/// all reported as typed solver errors before an adjoint may consume state.
pub fn verify_forward_checkpoint_replay<C, Encode, Decode, Equal>(
    contract: &C,
    input: &C::Input,
    encode: Encode,
    decode: Decode,
    equal: Equal,
    cancellation: &CancellationToken,
) -> Result<CheckpointReplayReport, SolverError>
where
    C: DifferentiableSolveContract,
    Encode: Fn(&C::Forward) -> Result<Vec<u8>, SolverError>,
    Decode: Fn(&[u8]) -> Result<C::Forward, SolverError>,
    Equal: Fn(&C::Forward, &C::Forward) -> bool,
{
    cancellation.check("differentiable checkpoint forward")?;
    let first = contract.solve_forward(input)?;
    cancellation.check("differentiable checkpoint encode")?;
    let payload = catch_unwind(AssertUnwindSafe(|| encode(&first)))
        .map_err(|payload| callback_panic("differentiable checkpoint encode", payload))??;
    if payload.is_empty() {
        return Err(SolverError::OperationFailed {
            operation: "differentiable checkpoint",
            reason: "checkpoint encoder returned an empty payload".into(),
        });
    }
    let restored = catch_unwind(AssertUnwindSafe(|| decode(&payload)))
        .map_err(|payload| callback_panic("differentiable checkpoint decode", payload))??;
    cancellation.check("differentiable checkpoint decode")?;
    let checkpoint_matches = catch_unwind(AssertUnwindSafe(|| equal(&first, &restored)))
        .map_err(|payload| callback_panic("differentiable checkpoint comparison", payload))?;
    if !checkpoint_matches {
        return Err(SolverError::OperationFailed {
            operation: "differentiable checkpoint",
            reason: "decoded forward state differs from original state".into(),
        });
    }
    let replay = contract.solve_forward(input)?;
    cancellation.check("differentiable checkpoint replay")?;
    let deterministic =
        catch_unwind(AssertUnwindSafe(|| equal(&first, &replay))).map_err(|payload| {
            callback_panic("differentiable checkpoint replay comparison", payload)
        })?;
    if !deterministic {
        return Err(SolverError::OperationFailed {
            operation: "differentiable checkpoint replay",
            reason: "forward replay state differs from original state".into(),
        });
    }
    Ok(CheckpointReplayReport {
        evaluations: 2,
        checkpoint_bytes: payload.len(),
        deterministic,
    })
}

/// Measured Taylor remainder for a primal/linearized callback pair.
#[derive(Debug, Clone)]
pub struct TaylorContractReport<R> {
    /// Positive perturbation sizes, in evaluation order.
    pub steps: Vec<R>,
    /// Norm of `F(x + h·δ) - F(x) - h·DF(x)[δ]` for each step.
    pub remainders: Vec<R>,
    /// Observed order between adjacent perturbation sizes.
    pub observed_orders: Vec<R>,
}

impl<R: Float + std::fmt::Display> TaylorContractReport<R> {
    /// Smallest finite observed order, if at least one pair was measured.
    pub fn minimum_observed_order(&self) -> Option<R> {
        self.observed_orders
            .iter()
            .copied()
            .filter(|order| order.is_finite())
            .reduce(R::min)
    }

    /// Require every measured adjacent pair to achieve at least `minimum`.
    pub fn require_order(&self, minimum: R) -> Result<(), SolverError> {
        if !minimum.is_finite() || minimum < R::zero() {
            return Err(SolverError::OperationFailed {
                operation: "differentiable Taylor contract",
                reason: "minimum Taylor order must be finite and non-negative".into(),
            });
        }
        // Exact remainder cancellation is represented by positive infinity
        // and is stronger than any finite requested order. NaN and negative
        // infinity remain invalid observations and fail closed.
        if self
            .observed_orders
            .iter()
            .any(|order| order.is_nan() || *order < minimum)
        {
            return Err(SolverError::OperationFailed {
                operation: "differentiable Taylor contract",
                reason: format!(
                    "observed Taylor remainder order below required threshold {minimum}"
                ),
            });
        }
        Ok(())
    }
}

/// Check first-order Taylor consistency for a differentiable callback pair.
///
/// The linearized callback receives the base point and direction and returns
/// `DF(x)[δ]`. Positive, strictly decreasing steps are required so the
/// measured remainder order is unambiguous. Callback failures and malformed
/// or non-finite values become typed solver errors.
pub fn check_taylor_remainder<T, F, L>(
    base: &[T],
    direction: &[T],
    steps: &[T::Real],
    primal: F,
    linearized: L,
) -> Result<TaylorContractReport<T::Real>, SolverError>
where
    T: ComplexField,
    F: Fn(&[T]) -> Result<Vec<T>, SolverError>,
    L: Fn(&[T], &[T]) -> Result<Vec<T>, SolverError>,
{
    if base.len() != direction.len() {
        return Err(SolverError::DimensionMismatch {
            expected: base.len(),
            got: direction.len(),
        });
    }
    if steps.len() < 2 {
        return Err(SolverError::OperationFailed {
            operation: "differentiable Taylor contract",
            reason: "at least two positive steps are required".into(),
        });
    }
    if base.iter().any(|value| !value.norm().is_finite())
        || direction.iter().any(|value| !value.norm().is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation: "differentiable Taylor contract",
            reason: "base point or direction contains NaN or infinity".into(),
        });
    }
    for pair in steps.windows(2) {
        if !pair[0].is_finite()
            || !pair[1].is_finite()
            || pair[0] <= T::Real::zero()
            || pair[1] <= T::Real::zero()
            || pair[1] >= pair[0]
        {
            return Err(SolverError::OperationFailed {
                operation: "differentiable Taylor contract",
                reason: "steps must be finite, positive, and strictly decreasing".into(),
            });
        }
    }
    if let Some(last) = steps.last()
        && (!last.is_finite() || *last <= T::Real::zero())
    {
        return Err(SolverError::OperationFailed {
            operation: "differentiable Taylor contract",
            reason: "steps must be finite and positive".into(),
        });
    }

    let base_value = catch_unwind(AssertUnwindSafe(|| primal(base)))
        .map_err(|payload| callback_panic("differentiable Taylor primal", payload))??;
    let derivative = catch_unwind(AssertUnwindSafe(|| linearized(base, direction)))
        .map_err(|payload| callback_panic("differentiable Taylor linearized", payload))??;
    if base_value.len() != derivative.len()
        || base_value.iter().any(|value| !value.norm().is_finite())
        || derivative.iter().any(|value| !value.norm().is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation: "differentiable Taylor contract",
            reason: "primal and linearized callback values are malformed or non-finite".into(),
        });
    }

    let mut remainders = Vec::with_capacity(steps.len());
    for &step in steps {
        let scaled = T::from_real(step);
        let point: Vec<T> = base
            .iter()
            .zip(direction)
            .map(|(value, delta)| *value + scaled * *delta)
            .collect();
        let value = catch_unwind(AssertUnwindSafe(|| primal(&point)))
            .map_err(|payload| callback_panic("differentiable Taylor primal", payload))??;
        if value.len() != base_value.len() || value.iter().any(|entry| !entry.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "differentiable Taylor contract",
                reason: "primal callback returned malformed or non-finite value".into(),
            });
        }
        let remainder: Vec<T> = value
            .iter()
            .zip(&base_value)
            .zip(&derivative)
            .map(|((value, base_value), derivative)| *value - *base_value - scaled * *derivative)
            .collect();
        let norm = T::vec_norm(&remainder);
        if !norm.is_finite() {
            return Err(SolverError::OperationFailed {
                operation: "differentiable Taylor contract",
                reason: "Taylor remainder is non-finite".into(),
            });
        }
        remainders.push(norm);
    }
    let observed_orders = steps
        .windows(2)
        .zip(remainders.windows(2))
        .map(|(step_pair, remainder_pair)| {
            if remainder_pair[0] == T::Real::zero() || remainder_pair[1] == T::Real::zero() {
                T::Real::infinity()
            } else {
                (remainder_pair[0] / remainder_pair[1]).ln() / (step_pair[0] / step_pair[1]).ln()
            }
        })
        .collect();
    Ok(TaylorContractReport {
        steps: steps.to_vec(),
        remainders,
        observed_orders,
    })
}

/// Closure-backed contract for transient, eigen, and block frontends.
///
/// The closures are `Fn` rather than `FnMut`, which allows a caller to share
/// immutable plans or put mutable device-resident state behind `Arc<Mutex<_>>`
/// while reusing the same contract across many objective evaluations. Panics
/// at either callback boundary become typed solver errors.
struct ClosureMarker<I: ?Sized, Seed: ?Sized, Fwd, Adj>(
    std::marker::PhantomData<fn(&I, &Seed)>,
    std::marker::PhantomData<(Fwd, Adj)>,
);

pub struct ClosureDifferentiableSolve<I, Fwd, Seed, Adj, ForwardFn, AdjointFn>
where
    I: ?Sized,
    Seed: ?Sized,
    ForwardFn: Fn(&I) -> Result<Fwd, SolverError>,
    AdjointFn: Fn(&Fwd, &Seed) -> Result<Adj, SolverError>,
{
    kind: DifferentiableSystemKind,
    forward: ForwardFn,
    adjoint: AdjointFn,
    validation: ContractValidation<I, Fwd, Seed, Adj>,
    _marker: std::marker::PhantomData<ClosureMarker<I, Seed, Fwd, Adj>>,
}

/// Contract alias for a Newmark or generalized-alpha trajectory.
pub type TransientDifferentiableSolve<I, T, Seed, ForwardFn, AdjointFn> =
    ClosureDifferentiableSolve<
        I,
        NewmarkResult<T>,
        Seed,
        NewmarkAdjointResult<T>,
        ForwardFn,
        AdjointFn,
    >;

/// Contract alias for a scalar generalized-eigenvalue sensitivity.
pub type EigenValueDifferentiableSolve<I, T, Seed, ForwardFn, AdjointFn> =
    ClosureDifferentiableSolve<I, T, Seed, T, ForwardFn, AdjointFn>;

/// Contract alias for a gauge-fixed generalized-eigenvector sensitivity.
pub type EigenVectorDifferentiableSolve<I, T, Seed, ForwardFn, AdjointFn> =
    ClosureDifferentiableSolve<I, Vec<T>, Seed, Vec<T>, ForwardFn, AdjointFn>;

/// Contract alias for a block FGMRES forward state and family-specific
/// reverse result.
pub type BlockDifferentiableSolve<I, T, Seed, Adj, ForwardFn, AdjointFn> =
    ClosureDifferentiableSolve<I, BlockFgmresSolution<T>, Seed, Adj, ForwardFn, AdjointFn>;

/// Build a transient contract with a typed [`NewmarkResult`] forward state.
pub fn transient_contract<I, T, Seed, ForwardFn, AdjointFn>(
    forward: ForwardFn,
    adjoint: AdjointFn,
) -> TransientDifferentiableSolve<I, T, Seed, ForwardFn, AdjointFn>
where
    I: ?Sized + DifferentiableContractValue,
    T: ComplexField,
    Seed: ?Sized + DifferentiableContractValue,
    ForwardFn: Fn(&I) -> Result<NewmarkResult<T>, SolverError>,
    AdjointFn: Fn(&NewmarkResult<T>, &Seed) -> Result<NewmarkAdjointResult<T>, SolverError>,
{
    ClosureDifferentiableSolve::new_checked(
        DifferentiableSystemKind::Transient,
        forward,
        adjoint,
        ContractValidation::dimension_checked(),
    )
}

/// Build a scalar eigenvalue contract with typed scalar forward and reverse
/// states.
pub fn eigen_value_contract<I, T, Seed, ForwardFn, AdjointFn>(
    forward: ForwardFn,
    adjoint: AdjointFn,
) -> EigenValueDifferentiableSolve<I, T, Seed, ForwardFn, AdjointFn>
where
    I: ?Sized + DifferentiableContractValue,
    T: ComplexField + DifferentiableContractValue,
    Seed: ?Sized + DifferentiableContractValue,
    ForwardFn: Fn(&I) -> Result<T, SolverError>,
    AdjointFn: Fn(&T, &Seed) -> Result<T, SolverError>,
{
    ClosureDifferentiableSolve::new_checked(
        DifferentiableSystemKind::Eigen,
        forward,
        adjoint,
        ContractValidation::dimension_checked(),
    )
}

/// Build a gauge-fixed eigenvector contract with typed vector forward and
/// reverse states.
pub fn eigen_vector_contract<I, T, Seed, ForwardFn, AdjointFn>(
    forward: ForwardFn,
    adjoint: AdjointFn,
) -> EigenVectorDifferentiableSolve<I, T, Seed, ForwardFn, AdjointFn>
where
    I: ?Sized + DifferentiableContractValue,
    T: ComplexField,
    Seed: ?Sized + DifferentiableContractValue,
    ForwardFn: Fn(&I) -> Result<Vec<T>, SolverError>,
    AdjointFn: Fn(&Vec<T>, &Seed) -> Result<Vec<T>, SolverError>,
{
    ClosureDifferentiableSolve::new_checked(
        DifferentiableSystemKind::Eigen,
        forward,
        adjoint,
        ContractValidation::dimension_checked(),
    )
}

/// Build a coupled block contract with a typed block-FGMRES forward state.
pub fn block_contract<I, T, Seed, Adj, ForwardFn, AdjointFn>(
    forward: ForwardFn,
    adjoint: AdjointFn,
) -> BlockDifferentiableSolve<I, T, Seed, Adj, ForwardFn, AdjointFn>
where
    I: ?Sized + DifferentiableContractValue,
    T: ComplexField,
    Seed: ?Sized + DifferentiableContractValue,
    Adj: DifferentiableContractValue,
    ForwardFn: Fn(&I) -> Result<BlockFgmresSolution<T>, SolverError>,
    AdjointFn: Fn(&BlockFgmresSolution<T>, &Seed) -> Result<Adj, SolverError>,
{
    ClosureDifferentiableSolve::new_checked(
        DifferentiableSystemKind::CoupledBlock,
        forward,
        adjoint,
        ContractValidation::dimension_checked(),
    )
}

impl<I, Fwd, Seed, Adj, ForwardFn, AdjointFn>
    ClosureDifferentiableSolve<I, Fwd, Seed, Adj, ForwardFn, AdjointFn>
where
    I: ?Sized,
    Seed: ?Sized,
    ForwardFn: Fn(&I) -> Result<Fwd, SolverError>,
    AdjointFn: Fn(&Fwd, &Seed) -> Result<Adj, SolverError>,
{
    /// Construct a contract with explicit boundary validation.
    pub fn new_checked(
        kind: DifferentiableSystemKind,
        forward: ForwardFn,
        adjoint: AdjointFn,
        validation: ContractValidation<I, Fwd, Seed, Adj>,
    ) -> Self {
        Self {
            kind,
            forward,
            adjoint,
            validation,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<I, Fwd, Seed, Adj, ForwardFn, AdjointFn>
    ClosureDifferentiableSolve<I, Fwd, Seed, Adj, ForwardFn, AdjointFn>
where
    I: ?Sized + DifferentiableContractValue,
    Fwd: DifferentiableContractValue,
    Seed: ?Sized + DifferentiableContractValue,
    Adj: DifferentiableContractValue,
    ForwardFn: Fn(&I) -> Result<Fwd, SolverError>,
    AdjointFn: Fn(&Fwd, &Seed) -> Result<Adj, SolverError>,
{
    /// Construct a contract with finite-value and intrinsic-dimension checks
    /// at every forward and adjoint boundary.
    pub fn new(kind: DifferentiableSystemKind, forward: ForwardFn, adjoint: AdjointFn) -> Self {
        Self::new_checked(
            kind,
            forward,
            adjoint,
            ContractValidation::dimension_checked(),
        )
    }
}

impl<I, Fwd, Seed, Adj, ForwardFn, AdjointFn> DifferentiableSolveContract
    for ClosureDifferentiableSolve<I, Fwd, Seed, Adj, ForwardFn, AdjointFn>
where
    I: ?Sized,
    Seed: ?Sized,
    ForwardFn: Fn(&I) -> Result<Fwd, SolverError>,
    AdjointFn: Fn(&Fwd, &Seed) -> Result<Adj, SolverError>,
{
    type Input = I;
    type Forward = Fwd;
    type Seed = Seed;
    type Adjoint = Adj;

    fn system_kind(&self) -> DifferentiableSystemKind {
        self.kind
    }

    fn solve_forward(&self, input: &Self::Input) -> Result<Self::Forward, SolverError> {
        run_contract_check("differentiable input validation", || {
            (self.validation.input)(input)
        })?;
        let forward = catch_unwind(AssertUnwindSafe(|| (self.forward)(input)))
            .map_err(|payload| callback_panic("differentiable forward", payload))??;
        run_contract_check("differentiable forward validation", || {
            (self.validation.forward)(&forward)
        })?;
        Ok(forward)
    }

    fn solve_adjoint(
        &self,
        forward: &Self::Forward,
        seed: &Self::Seed,
    ) -> Result<Self::Adjoint, SolverError> {
        run_contract_check("differentiable forward validation", || {
            (self.validation.forward)(forward)
        })?;
        run_contract_check("differentiable seed validation", || {
            (self.validation.seed)(seed)
        })?;
        let adjoint = catch_unwind(AssertUnwindSafe(|| (self.adjoint)(forward, seed)))
            .map_err(|payload| callback_panic("differentiable adjoint", payload))??;
        run_contract_check("differentiable adjoint validation", || {
            (self.validation.adjoint)(&adjoint)
        })?;
        if let Some(relation) = &self.validation.relation {
            run_contract_check("differentiable lifecycle dimension validation", || {
                relation(forward, seed, &adjoint)
            })?;
        }
        Ok(adjoint)
    }
}

impl<'a, T, A, P> DifferentiableSolveContract for DifferentiableLinearSolve<'a, T, A, P>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    type Input = [T];
    type Forward = GmresSolution<T>;
    type Seed = SteadyAdjointSeed<T>;
    type Adjoint = GmresSolution<T>;

    fn system_kind(&self) -> DifferentiableSystemKind {
        DifferentiableSystemKind::Steady
    }

    fn solve_forward(&self, input: &Self::Input) -> Result<Self::Forward, SolverError> {
        DifferentiableLinearSolve::solve_forward(self, input)
    }
    fn solve_forward_cancellable(
        &self,
        input: &Self::Input,
        cancellation: &CancellationToken,
    ) -> Result<Self::Forward, SolverError> {
        DifferentiableLinearSolve::solve_forward_cancellable(self, input, cancellation)
    }

    fn solve_adjoint(
        &self,
        forward: &Self::Forward,
        seed: &Self::Seed,
    ) -> Result<Self::Adjoint, SolverError> {
        // Keep the native steady handle on the same lifecycle contract as
        // closure-backed transient/eigen/block handles.  In particular, do
        // not silently accept a stale or tampered retained forward state:
        // the adjoint is only meaningful for the primal state that produced
        // it, and the seed must have the corresponding operator dimension.
        forward.validate_contract_value("steady forward state")?;
        seed.validate_contract_value("steady adjoint seed")?;
        if seed.rhs.len() != forward.x.len() {
            return Err(SolverError::DimensionMismatch {
                expected: forward.x.len(),
                got: seed.rhs.len(),
            });
        }
        let adjoint = DifferentiableLinearSolve::solve_adjoint(self, &seed.rhs, seed.kind)?;
        adjoint.validate_contract_value("steady adjoint state")?;
        if adjoint.x.len() != forward.x.len() {
            return Err(SolverError::DimensionMismatch {
                expected: forward.x.len(),
                got: adjoint.x.len(),
            });
        }
        Ok(adjoint)
    }

    fn solve_adjoint_cancellable(
        &self,
        forward: &Self::Forward,
        seed: &Self::Seed,
        cancellation: &CancellationToken,
    ) -> Result<Self::Adjoint, SolverError> {
        forward.validate_contract_value("steady forward state")?;
        seed.validate_contract_value("steady adjoint seed")?;
        if seed.rhs.len() != forward.x.len() {
            return Err(SolverError::DimensionMismatch {
                expected: forward.x.len(),
                got: seed.rhs.len(),
            });
        }
        let adjoint = DifferentiableLinearSolve::solve_adjoint_cancellable(
            self,
            &seed.rhs,
            seed.kind,
            cancellation,
        )?;
        adjoint.validate_contract_value("steady adjoint state")?;
        if adjoint.x.len() != forward.x.len() {
            return Err(SolverError::DimensionMismatch {
                expected: forward.x.len(),
                got: adjoint.x.len(),
            });
        }
        Ok(adjoint)
    }
}

fn callback_panic(operation: &'static str, payload: Box<dyn std::any::Any + Send>) -> SolverError {
    let reason = if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "differentiable callback panicked".to_string()
    };
    SolverError::OperationFailed { operation, reason }
}

fn run_contract_check<F>(operation: &'static str, check: F) -> Result<(), SolverError>
where
    F: FnOnce() -> Result<(), SolverError>,
{
    catch_unwind(AssertUnwindSafe(check)).map_err(|payload| callback_panic(operation, payload))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrMatrix;
    use crate::iterative::GmresConfig;
    use crate::preconditioners::IdentityPreconditioner;
    use num_complex::Complex64;

    #[test]
    fn closure_contract_reuses_family_specific_forward_state() {
        let contract = ClosureDifferentiableSolve::new(
            DifferentiableSystemKind::CoupledBlock,
            |input: &[f64]| Ok(input.iter().map(|value| value * 2.0).collect::<Vec<_>>()),
            |forward: &Vec<f64>, seed: &[f64]| {
                Ok(forward
                    .iter()
                    .zip(seed)
                    .map(|(value, seed)| value + seed)
                    .collect::<Vec<_>>())
            },
        );
        let forward = contract.solve_forward(&[1.0, 2.0]).expect("forward state");
        let adjoint = contract
            .solve_adjoint(&forward, &[0.5, 1.5])
            .expect("adjoint state");
        assert_eq!(
            contract.system_kind(),
            DifferentiableSystemKind::CoupledBlock
        );
        assert_eq!(forward, vec![2.0, 4.0]);
        assert_eq!(adjoint, vec![2.5, 5.5]);
    }

    #[test]
    fn closure_contract_contains_callback_panics() {
        let contract = ClosureDifferentiableSolve::new(
            DifferentiableSystemKind::Transient,
            |_input: &[f64]| -> Result<Vec<f64>, SolverError> { panic!("forward failure") },
            |_forward: &Vec<f64>, _seed: &[f64]| Ok(Vec::<f64>::new()),
        );
        let error = contract
            .solve_forward(&[])
            .expect_err("forward panic should become a solver error");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }

    #[test]
    fn default_closure_constructor_rejects_nonfinite_input_and_seed() {
        let contract = ClosureDifferentiableSolve::new(
            DifferentiableSystemKind::CoupledBlock,
            |input: &[f64]| Ok(input.to_vec()),
            |_forward: &Vec<f64>, _seed: &[f64]| -> Result<Vec<f64>, SolverError> {
                panic!("adjoint callback must not run for an invalid seed")
            },
        );

        let input_error = contract
            .solve_forward(&[f64::NAN])
            .expect_err("non-finite input must be rejected before the callback");
        assert!(input_error.to_string().contains("forward input"));

        let forward = contract
            .solve_forward(&[1.0])
            .expect("finite input should reach the callback");
        let seed_error = contract
            .solve_adjoint(&forward, &[f64::INFINITY])
            .expect_err("non-finite seed must be rejected before the callback");
        assert!(seed_error.to_string().contains("adjoint seed"));
    }

    #[test]
    fn native_family_contract_constructors_preserve_result_types() {
        let transient = transient_contract(
            |input: &[f64]| {
                Ok(NewmarkResult {
                    displacement: input.to_vec(),
                    velocity: vec![0.0; input.len()],
                    acceleration: vec![0.0; input.len()],
                    displacement_history: None,
                    velocity_history: None,
                    acceleration_history: None,
                    steps_completed: 0,
                    final_time: 0.0,
                })
            },
            |forward: &NewmarkResult<f64>, seed: &[f64]| {
                Ok(NewmarkAdjointResult {
                    force_gradient: vec![seed.to_vec()],
                    initial_displacement_gradient: forward.displacement.clone(),
                    initial_velocity_gradient: forward.velocity.clone(),
                    initial_acceleration_gradient: forward.acceleration.clone(),
                })
            },
        );
        let forward = transient
            .solve_forward(&[1.0, 2.0])
            .expect("transient forward");
        let adjoint = transient
            .solve_adjoint(&forward, &[3.0, 4.0])
            .expect("transient adjoint");
        assert_eq!(transient.system_kind(), DifferentiableSystemKind::Transient);
        assert_eq!(adjoint.force_gradient, vec![vec![3.0, 4.0]]);

        let eigen = eigen_value_contract(
            |input: &[f64]| Ok(input[0]),
            |forward: &f64, seed: &[f64]| Ok(*forward + seed[0]),
        );
        let eigen_forward = eigen.solve_forward(&[2.0]).expect("eigen forward");
        let eigen_adjoint = eigen
            .solve_adjoint(&eigen_forward, &[0.5])
            .expect("eigen adjoint");
        assert_eq!(eigen.system_kind(), DifferentiableSystemKind::Eigen);
        assert_eq!(eigen_adjoint, 2.5);
    }

    #[test]
    fn native_transient_contract_rejects_inconsistent_forward_dimensions() {
        let contract = transient_contract(
            |_input: &[f64]| {
                Ok(NewmarkResult {
                    displacement: vec![1.0, 2.0],
                    velocity: vec![0.0],
                    acceleration: vec![0.0, 0.0],
                    displacement_history: None,
                    velocity_history: None,
                    acceleration_history: None,
                    steps_completed: 1,
                    final_time: 1.0,
                })
            },
            |_forward: &NewmarkResult<f64>, _seed: &[f64]| {
                Ok(NewmarkAdjointResult {
                    force_gradient: Vec::new(),
                    initial_displacement_gradient: vec![0.0, 0.0],
                    initial_velocity_gradient: vec![0.0, 0.0],
                    initial_acceleration_gradient: vec![0.0, 0.0],
                })
            },
        );

        let error = contract
            .solve_forward(&[])
            .expect_err("malformed transient state must be rejected");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
        assert!(error.to_string().contains("dimensions differ"));
    }

    #[test]
    fn native_transient_contract_rejects_inconsistent_adjoint_dimensions() {
        let contract = transient_contract(
            |_input: &[f64]| {
                Ok(NewmarkResult {
                    displacement: vec![1.0, 2.0],
                    velocity: vec![0.0, 0.0],
                    acceleration: vec![0.0, 0.0],
                    displacement_history: None,
                    velocity_history: None,
                    acceleration_history: None,
                    steps_completed: 1,
                    final_time: 1.0,
                })
            },
            |_forward: &NewmarkResult<f64>, _seed: &[f64]| {
                Ok(NewmarkAdjointResult {
                    force_gradient: vec![vec![0.0, 0.0]],
                    initial_displacement_gradient: vec![0.0, 0.0],
                    initial_velocity_gradient: vec![0.0],
                    initial_acceleration_gradient: vec![0.0, 0.0],
                })
            },
        );
        let forward = contract.solve_forward(&[]).expect("valid transient state");

        let error = contract
            .solve_adjoint(&forward, &[])
            .expect_err("malformed transient adjoint state must be rejected");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
        assert!(error.to_string().contains("gradient dimensions differ"));
    }

    #[test]
    fn native_contracts_reject_nonfinite_complex_states_before_adjoint() {
        let contract = eigen_value_contract(
            |_input: &[f64]| Ok(Complex64::new(f64::NAN, 0.0)),
            |_forward: &Complex64, _seed: &[f64]| Ok(Complex64::new(1.0, 0.0)),
        );

        let error = contract
            .solve_forward(&[])
            .expect_err("non-finite eigenvalue must be rejected");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
        assert!(error.to_string().contains("NaN or infinity"));
    }

    #[test]
    fn eigen_vector_contract_rejects_nonfinite_generated_state() {
        let contract = eigen_vector_contract(
            |_input: &[f64]| Ok(vec![Complex64::new(1.0, f64::INFINITY)]),
            |_forward: &Vec<Complex64>, _seed: &[f64]| Ok(vec![Complex64::new(0.0, 0.0)]),
        );

        let error = contract
            .solve_forward(&[1.0])
            .expect_err("non-finite eigenvector state must be rejected");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
        assert!(error.to_string().contains("forward state"));
    }

    #[test]
    fn block_contract_rejects_bad_metadata_and_nonfinite_adjoint() {
        let malformed = block_contract(
            |_input: &[f64]| {
                Ok(BlockFgmresSolution {
                    x: sonium_solvers_core::dense::DenseMatrix::from_elem(2, 1, 0.0),
                    iterations: 0,
                    residuals: vec![0.0],
                    true_residuals: Vec::new(),
                    backward_errors: vec![0.0],
                    attainable_accuracy: vec![crate::traits::AccuracyStatus::NotChecked],
                    converged: false,
                    op_counts: Default::default(),
                })
            },
            |_forward: &BlockFgmresSolution<f64>, _seed: &[f64]| Ok(vec![0.0]),
        );
        let error = malformed
            .solve_forward(&[0.0, 0.0])
            .expect_err("per-RHS metadata must match block width");
        assert!(error.to_string().contains("dimension mismatch"));

        let valid = block_contract(
            |_input: &[f64]| {
                Ok(BlockFgmresSolution {
                    x: sonium_solvers_core::dense::DenseMatrix::from_elem(2, 1, 0.0),
                    iterations: 0,
                    residuals: vec![0.0],
                    true_residuals: vec![0.0],
                    backward_errors: vec![0.0],
                    attainable_accuracy: vec![crate::traits::AccuracyStatus::Attained],
                    converged: true,
                    op_counts: Default::default(),
                })
            },
            |_forward: &BlockFgmresSolution<f64>, _seed: &[f64]| Ok(vec![f64::NAN]),
        );
        let forward = valid.solve_forward(&[0.0, 0.0]).expect("valid block state");
        let error = valid
            .solve_adjoint(&forward, &[0.0])
            .expect_err("non-finite block adjoint must be rejected");
        assert!(error.to_string().contains("adjoint state"));
    }

    #[test]
    fn dimension_checked_validation_rejects_nonfinite_input_before_callback() {
        let validation =
            ContractValidation::<[f64], Vec<f64>, [f64], Vec<f64>>::dimension_checked();
        let contract = ClosureDifferentiableSolve::new_checked(
            DifferentiableSystemKind::CoupledBlock,
            |_input: &[f64]| Ok(vec![1.0]),
            |_forward: &Vec<f64>, _seed: &[f64]| Ok(vec![1.0]),
            validation,
        );

        let error = contract
            .solve_forward(&[f64::INFINITY])
            .expect_err("non-finite input must be rejected before the callback");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
        assert!(error.to_string().contains("forward input"));
    }

    #[test]
    fn exact_dimensions_reject_callback_state_with_wrong_declared_shape() {
        let validation = ContractValidation::<[f64], Vec<f64>, [f64], Vec<f64>>::exact_dimensions(
            ContractDimensions::new(Some(2), Some(2), Some(1), Some(1)),
        );
        let contract = ClosureDifferentiableSolve::new_checked(
            DifferentiableSystemKind::CoupledBlock,
            |_input: &[f64]| Ok(vec![1.0]),
            |_forward: &Vec<f64>, _seed: &[f64]| Ok(vec![1.0]),
            validation,
        );

        let error = contract
            .solve_forward(&[0.0, 1.0])
            .expect_err("declared forward dimension must be enforced");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
        assert!(error.to_string().contains("expected 2"));
    }

    #[test]
    fn steady_linear_solve_implements_common_contract() {
        let matrix = CsrMatrix::<f64>::from_triplets(
            2,
            2,
            vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 2.0), (1, 1, 3.0)],
        );
        let config = GmresConfig {
            max_iterations: 32,
            restart: 8,
            tolerance: 1.0e-12,
            ..GmresConfig::default()
        };
        let handle = DifferentiableLinearSolve::new(&matrix, &IdentityPreconditioner, config)
            .expect("square steady system");
        let forward = DifferentiableSolveContract::solve_forward(&handle, &[6.0, 8.0])
            .expect("forward contract solve");
        let seed = SteadyAdjointSeed::new(vec![8.0, 7.0], AdjointSolve::Transpose);
        let adjoint = DifferentiableSolveContract::solve_adjoint(&handle, &forward, &seed)
            .expect("adjoint contract solve");
        assert_eq!(handle.system_kind(), DifferentiableSystemKind::Steady);
        assert!((forward.x[0] - 1.0).abs() < 1.0e-10);
        assert!((adjoint.x[0] - 1.0).abs() < 1.0e-10);
    }
}
