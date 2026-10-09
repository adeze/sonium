//! Backend capability and bounded coupled-solve evidence.
//!
//! The high-level axisymmetric assembly in this crate is still CPU-owned.
//! This module makes that boundary explicit and provides one real backend
//! execution seam: the sparse two-by-two electromechanical driver block.  The
//! CUDA path uploads the live `DriverModel` network coefficients, solves the
//! coupled electrical/mechanical system with device-resident GMRES, and only
//! then returns the observables needed for artifact comparison.

#[cfg(feature = "cuda")]
use crate::axisymmetric_linear::{AxisymmetricLinearConfig, AxisymmetricLinearModel};
#[cfg(feature = "cuda")]
use crate::conventions::ArtifactExecutionProfile;
use crate::conventions::{ArtifactBackend, ArtifactPrecision};
use crate::driver_model::{DriverModel, DriverModelError, DriverModelReferenceArtifact};
#[cfg(feature = "cuda")]
use crate::driver_model::{
    DriverModelConvergenceEvidence, DriverModelPromotionEvidence, DriverModelReferenceExecution,
    DriverModelReferencePoint, DriverModelTransferEvidence,
};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
#[cfg(feature = "cuda")]
use sonium_solvers::sparse::CsrMatrix;

/// Subsystems called out by the CUDA execution contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerSubsystem {
    Magnetic,
    Structural,
    Acoustic,
    Thermal,
    SparseCoupled,
    Derivative,
    NonlinearMaterial,
}

/// Portable counterpart of `sonium_compute::CapabilityState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerCapabilityState {
    Native,
    VendorNative,
    HostSharedFallback,
    HostStagedFallback,
    Unsupported,
}

/// One machine-readable subsystem capability entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeakerSubsystemCapability {
    pub subsystem: SpeakerSubsystem,
    pub state: SpeakerCapabilityState,
    pub operation: String,
    pub reason: String,
}

/// Explicit backend matrix for a speaker modeling run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeakerCapabilityMatrix {
    pub requested_backend: ArtifactBackend,
    pub actual_backend: ArtifactBackend,
    pub precision: ArtifactPrecision,
    pub device: String,
    pub toolchain: String,
    pub subsystems: Vec<SpeakerSubsystemCapability>,
}

impl SpeakerCapabilityMatrix {
    /// Describe the currently supported CPU reference route.
    pub fn cpu() -> Self {
        let entries = [
            (SpeakerSubsystem::Magnetic, "axisymmetric FEM assembly"),
            (
                SpeakerSubsystem::Structural,
                "axisymmetric elasticity assembly",
            ),
            (SpeakerSubsystem::Acoustic, "acoustic FEM/BEM assembly"),
            (SpeakerSubsystem::Thermal, "driver thermal network"),
            (
                SpeakerSubsystem::SparseCoupled,
                "CPU sparse/direct coupled solve",
            ),
            (
                SpeakerSubsystem::Derivative,
                "CPU finite-difference sensitivity",
            ),
            (
                SpeakerSubsystem::NonlinearMaterial,
                "CPU nonlinear material law",
            ),
        ];
        Self {
            requested_backend: ArtifactBackend::Cpu,
            actual_backend: ArtifactBackend::Cpu,
            precision: ArtifactPrecision::F64,
            device: "cpu".to_string(),
            toolchain: env!("SONIUM_BUILD_RUSTC_VERSION").to_string(),
            subsystems: entries
                .into_iter()
                .map(|(subsystem, operation)| SpeakerSubsystemCapability {
                    subsystem,
                    state: SpeakerCapabilityState::Native,
                    operation: operation.to_string(),
                    reason: "portable reference implementation".to_string(),
                })
                .collect(),
        }
    }

    /// Probe the CUDA runtime and return the actual per-subsystem matrix.
    ///
    /// A missing driver, device, or compiled CUDA feature is an error rather
    /// than a silent CPU claim. Callers that explicitly permit fallback should
    /// use [`run_coupled_reference`] and inspect its requested/actual fields.
    #[cfg(feature = "cuda")]
    pub fn probe_cuda() -> Result<Self, SpeakerBackendError> {
        let backend = sonium_compute::CudaBackend::new_default()
            .map_err(|error| SpeakerBackendError::Unavailable(error.to_string()))?;
        Ok(Self::from_cuda_backend(&backend))
    }

    /// Report that this binary cannot provide a CUDA capability matrix.
    #[cfg(not(feature = "cuda"))]
    pub fn probe_cuda() -> Result<Self, SpeakerBackendError> {
        let _ = Self::cuda_without_feature();
        Err(SpeakerBackendError::Unavailable(
            "sonium-sim-speaker-modeling was built without feature `cuda`".to_string(),
        ))
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda_without_feature() -> Self {
        Self {
            requested_backend: ArtifactBackend::Cuda,
            actual_backend: ArtifactBackend::Cuda,
            precision: ArtifactPrecision::F64,
            device: "cuda:unavailable".to_string(),
            toolchain: "sonium-compute/cuda feature disabled".to_string(),
            subsystems: Self::cuda_entries(
                SpeakerCapabilityState::Unsupported,
                "CUDA support is not compiled for this binary",
            ),
        }
    }

    fn cuda_entries(
        sparse_state: SpeakerCapabilityState,
        sparse_reason: &str,
    ) -> Vec<SpeakerSubsystemCapability> {
        [
            (
                SpeakerSubsystem::Magnetic,
                SpeakerCapabilityState::Unsupported,
                "axisymmetric FEM assembly",
                "high-level magnetic assembly is CPU-owned",
            ),
            (
                SpeakerSubsystem::Structural,
                SpeakerCapabilityState::Unsupported,
                "axisymmetric elasticity assembly",
                "high-level structural assembly is CPU-owned",
            ),
            (
                SpeakerSubsystem::Acoustic,
                SpeakerCapabilityState::Unsupported,
                "acoustic FEM/BEM assembly",
                "high-level acoustic assembly is CPU-owned",
            ),
            (
                SpeakerSubsystem::Thermal,
                SpeakerCapabilityState::Unsupported,
                "driver thermal network",
                "thermal material/coupling cells have no CUDA implementation",
            ),
            (
                SpeakerSubsystem::SparseCoupled,
                sparse_state,
                "device CSR GMRES electromechanical block",
                sparse_reason,
            ),
            (
                SpeakerSubsystem::Derivative,
                sparse_state,
                "device CSR voltage-transfer replay",
                sparse_reason,
            ),
            (
                SpeakerSubsystem::NonlinearMaterial,
                SpeakerCapabilityState::Unsupported,
                "nonlinear magnetic/thermal material law",
                "nonlinear material cells must reject before CUDA execution",
            ),
        ]
        .into_iter()
        .map(
            |(subsystem, state, operation, reason)| SpeakerSubsystemCapability {
                subsystem,
                state,
                operation: operation.to_string(),
                reason: reason.to_string(),
            },
        )
        .collect()
    }

    #[cfg(feature = "cuda")]
    fn native_sparse_cuda(
        precision: ArtifactPrecision,
        device: String,
        toolchain: String,
        state: SpeakerCapabilityState,
        reason: &str,
    ) -> Self {
        Self {
            requested_backend: ArtifactBackend::Cuda,
            actual_backend: ArtifactBackend::Cuda,
            precision,
            device,
            toolchain,
            subsystems: Self::cuda_entries(state, reason),
        }
    }

    #[cfg(feature = "cuda")]
    fn from_cuda_backend(backend: &sonium_compute::CudaBackend) -> Self {
        use sonium_compute::Backend;
        let capabilities = backend.capabilities();
        let csr_state = capabilities.csr_spmv.state_for::<Complex64>();
        let blas_state = capabilities.blas1.state_for::<Complex64>();
        let dot_state = capabilities.multi_dot.state_for::<Complex64>();
        let sparse_state =
            if csr_state.is_native() && blas_state.is_native() && dot_state.is_available() {
                if dot_state.uses_host_fallback() {
                    SpeakerCapabilityState::HostStagedFallback
                } else if matches!(csr_state, sonium_compute::CapabilityState::VendorNative)
                    && matches!(blas_state, sonium_compute::CapabilityState::VendorNative)
                {
                    SpeakerCapabilityState::VendorNative
                } else {
                    SpeakerCapabilityState::Native
                }
            } else {
                SpeakerCapabilityState::Unsupported
            };
        let reason = if sparse_state == SpeakerCapabilityState::Unsupported {
            format!(
                "native Complex64 CSR/BLAS/reduction required (csr={csr_state:?}, blas={blas_state:?}, multi_dot={dot_state:?})"
            )
        } else {
            "cuSPARSE/cuBLAS device-resident bounded coupled solve".to_string()
        };
        Self::native_sparse_cuda(
            ArtifactPrecision::F64,
            "cuda:ordinal=0".to_string(),
            "sonium-compute/cudarc + cuBLAS/cuSPARSE".to_string(),
            sparse_state,
            &reason,
        )
    }

    #[cfg(feature = "cuda")]
    fn state_is_available(&self, subsystem: SpeakerSubsystem) -> bool {
        self.subsystems
            .iter()
            .find(|entry| entry.subsystem == subsystem)
            .is_some_and(|entry| entry.state != SpeakerCapabilityState::Unsupported)
    }
}

/// Errors from a requested backend execution.
#[derive(Debug, thiserror::Error)]
pub enum SpeakerBackendError {
    #[error("requested CUDA backend is unavailable: {0}")]
    Unavailable(String),
    #[error("CUDA capability is unsupported for {subsystem:?}: {reason}")]
    Unsupported {
        subsystem: SpeakerSubsystem,
        reason: String,
    },
    #[error("CUDA coupled solve failed: {0}")]
    Compute(String),
    #[error("CUDA backend operation failed: {0}")]
    Backend(#[from] sonium_compute::BackendError),
    #[error(transparent)]
    DriverModel(#[from] DriverModelError),
    #[error("CUDA coupled solve did not converge at {frequency_hz} Hz (residual {residual:e})")]
    Nonconverged { frequency_hz: f64, residual: f64 },
    #[error("CUDA coupled solve produced a non-finite observable")]
    NonFinite,
}

/// Stable stages used by the bounded speaker-modeling CUDA failure contract.
///
/// This vocabulary is intentionally owned by the modeling boundary rather
/// than inferred from a driver error string by a release checker.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerBackendFailureStage {
    Setup,
    Launch,
    Synchronization,
}

#[cfg(feature = "cuda")]
impl SpeakerBackendFailureStage {
    const fn label(self) -> &'static str {
        match self {
            Self::Setup => "setup",
            Self::Launch => "launch",
            Self::Synchronization => "synchronization",
        }
    }
}

/// Structured evidence from one injected native-CUDA failure and its retry.
///
/// The value is deliberately JSON-shaped at the API boundary so an acceptance
/// producer can persist it without exposing CUDA driver types in the artifact
/// schema. The underlying failed operation remains typed in
/// [`SpeakerBackendError::Backend`].
#[cfg(feature = "cuda")]
pub type SpeakerCudaFailureEvidence = serde_json::Value;

/// Backend and parity evidence returned by the bounded coupled path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoupledBackendEvidence {
    pub capability_matrix: SpeakerCapabilityMatrix,
    pub reference: DriverModelReferenceArtifact,
    pub max_relative_residual: f64,
    pub max_power_balance_relative_error: f64,
    pub max_velocity_relative_error: f64,
    pub max_impedance_relative_error: f64,
    pub max_derivative_relative_error: f64,
}

/// Execute a bounded coupled reference using the requested backend.
///
/// CUDA has no implicit fallback here.  If `allow_cpu_fallback` is true and
/// CUDA cannot be initialised or lacks the sparse coupled capability, the
/// returned artifact explicitly records `requested_backend=cuda` and
/// `actual_backend=cpu`; it is never admitted as a CUDA artifact.
pub fn run_coupled_reference(
    model: &DriverModel,
    frequency_hz: &[f64],
    radiation_impedance: &[Complex64],
    requested_backend: ArtifactBackend,
    allow_cpu_fallback: bool,
) -> Result<CoupledBackendEvidence, SpeakerBackendError> {
    match requested_backend {
        ArtifactBackend::Cpu => run_cpu_reference(model, frequency_hz, radiation_impedance),
        ArtifactBackend::Cuda => {
            let result = run_cuda_reference(model, frequency_hz, radiation_impedance);
            match result {
                Ok(evidence) => Ok(evidence),
                Err(error) if allow_cpu_fallback && error.is_fallback_safe() => {
                    run_cpu_fallback(model, frequency_hz, radiation_impedance)
                }
                Err(error) => Err(error),
            }
        }
        other => Err(SpeakerBackendError::Unsupported {
            subsystem: SpeakerSubsystem::SparseCoupled,
            reason: format!("backend {other:?} is not implemented for this bounded path"),
        }),
    }
}

impl SpeakerBackendError {
    fn is_fallback_safe(&self) -> bool {
        matches!(self, Self::Unavailable(_) | Self::Unsupported { .. })
    }
}

#[cfg(feature = "cuda")]
fn backend_operation(error: &sonium_compute::BackendError) -> String {
    use sonium_compute::BackendError;
    match error {
        BackendError::Unsupported { operation, .. }
        | BackendError::HostFallback { operation, .. }
        | BackendError::InvalidArgument { operation, .. }
        | BackendError::OperationFailed { operation, .. }
        | BackendError::DeviceLost { operation, .. }
        | BackendError::CommandBuffer { operation, .. }
        | BackendError::Launch { operation, .. }
        | BackendError::Synchronization { operation, .. }
        | BackendError::Transfer { operation, .. } => (*operation).to_owned(),
        BackendError::Allocation { .. } => "cuda_allocation".to_owned(),
    }
}

#[cfg(feature = "cuda")]
fn failure_stage(error: &sonium_compute::BackendError) -> SpeakerBackendFailureStage {
    use sonium_compute::BackendError;
    match error {
        BackendError::Launch { .. } => SpeakerBackendFailureStage::Launch,
        BackendError::Synchronization { .. } | BackendError::CommandBuffer { .. } => {
            SpeakerBackendFailureStage::Synchronization
        }
        BackendError::DeviceLost { .. }
        | BackendError::Transfer { .. }
        | BackendError::OperationFailed { .. }
        | BackendError::Unsupported { .. }
        | BackendError::HostFallback { .. }
        | BackendError::InvalidArgument { .. }
        | BackendError::Allocation { .. } => SpeakerBackendFailureStage::Setup,
    }
}

#[cfg(feature = "cuda")]
fn context_recovery_label(disposition: sonium_compute::BackendContextDisposition) -> &'static str {
    match disposition {
        sonium_compute::BackendContextDisposition::Reusable => "reusable",
        sonium_compute::BackendContextDisposition::SynchronizeBeforeReuse => {
            "synchronize-before-reuse"
        }
        sonium_compute::BackendContextDisposition::MustRecreate => "recreate",
    }
}

#[cfg(feature = "cuda")]
fn output_disposition_label(disposition: sonium_compute::BackendOutputDisposition) -> &'static str {
    match disposition {
        sonium_compute::BackendOutputDisposition::Preserved => "preserved",
        sonium_compute::BackendOutputDisposition::Unusable => "unusable",
    }
}

/// Exercise the production bounded CUDA coupling seam with one checked native
/// failure, then retry the same fixture on a freshly created context.
///
/// This is an acceptance-only seam for launch and terminal device-loss
/// injection. It intentionally does not claim that the unsupported high-level
/// FEM/BEM physics is resident on the device. The failed attempt owns all
/// buffers locally and therefore cannot publish a response artifact; the
/// returned JSON is constructed only after the fresh-context retry has passed
/// CPU/f64 parity and transfer/residency gates.
#[cfg(feature = "cuda")]
pub fn run_coupled_failure_contract(
    model: &DriverModel,
    frequency_hz: &[f64],
    radiation_impedance: &[Complex64],
    failure_point: sonium_compute::CudaFailureInjectionPoint,
) -> Result<SpeakerCudaFailureEvidence, SpeakerBackendError> {
    use sonium_compute::{
        Backend, CudaBackend, DeviceCsr, GmresBackendConfig, IdentityBackendPreconditioner,
        PersistentDeviceCsr, gmres_backend_preconditioned,
    };

    let expected = match failure_point {
        sonium_compute::CudaFailureInjectionPoint::Launch => ("launch", "cuda_launch"),
        sonium_compute::CudaFailureInjectionPoint::DeviceLoss => {
            ("device-loss", "cuda_device_loss")
        }
        other => {
            return Err(SpeakerBackendError::Compute(format!(
                "SM1 failure contract accepts only launch/device-loss injection, got {other:?}"
            )));
        }
    };
    model.validate()?;
    if model.thermal_compression_spec.is_some() || model.thermal_network.is_some() {
        return Err(SpeakerBackendError::Unsupported {
            subsystem: SpeakerSubsystem::Thermal,
            reason: "bounded CUDA failure contract has no thermal-state kernel".to_string(),
        });
    }
    if model.nonlinear_motor.is_some() || model.nonlinear_operating_domain_evidence.is_some() {
        return Err(SpeakerBackendError::Unsupported {
            subsystem: SpeakerSubsystem::NonlinearMaterial,
            reason: "bounded CUDA failure contract accepts only the validated linear network"
                .to_string(),
        });
    }
    if frequency_hz.len() != radiation_impedance.len() || frequency_hz.is_empty() {
        return Err(SpeakerBackendError::Compute(
            "frequency and radiation grids must be non-empty and equal".to_string(),
        ));
    }

    let cpu_golden = run_cpu_reference(model, frequency_hz, radiation_impedance)?;
    let backend = CudaBackend::new_default()
        .map_err(|error| SpeakerBackendError::Unavailable(error.to_string()))?;
    let matrix = SpeakerCapabilityMatrix::from_cuda_backend(&backend);
    if !matrix.state_is_available(SpeakerSubsystem::SparseCoupled) {
        return Err(SpeakerBackendError::Unsupported {
            subsystem: SpeakerSubsystem::SparseCoupled,
            reason: "native Complex64 CSR/BLAS/reduction is required for failure injection"
                .to_string(),
        });
    }

    let first_network = model.network_parameters_at(frequency_hz[0])?;
    let row_ptr = [0_i32, 2, 4];
    let col_idx = [0_i32, 1, 0, 1];
    let values = [
        first_network.coil_impedance_ohm,
        Complex64::new(first_network.force_factor_tm, 0.0),
        Complex64::new(-first_network.force_factor_tm, 0.0),
        first_network.intrinsic_mechanical_impedance + radiation_impedance[0],
    ];
    let device_matrix = DeviceCsr::<CudaBackend, Complex64>::try_from_host(
        &backend, 2, 2, &row_ptr, &col_idx, &values,
    )
    .map_err(SpeakerBackendError::Backend)?;
    let operator = PersistentDeviceCsr::try_from_device_csr(&backend, device_matrix)
        .map_err(SpeakerBackendError::Backend)?;
    let mut rhs = backend
        .raw_alloc::<Complex64>(2)
        .map_err(|error| SpeakerBackendError::Backend(error.into()))?;
    let mut solution = backend
        .raw_alloc::<Complex64>(2)
        .map_err(|error| SpeakerBackendError::Backend(error.into()))?;
    let voltage = Complex64::new(model.reference_voltage_rms_v, 0.0);
    backend
        .try_copy_h2d(&[voltage, Complex64::new(0.0, 0.0)], &mut rhs)
        .map_err(SpeakerBackendError::Backend)?;

    // Arm the one-shot shim after setup and upload so an observed failure is
    // attributed to the requested execution boundary rather than admission.
    let failure_stats_before = backend.begin_runtime_stats();
    backend.inject_failure(failure_point);
    let failure = gmres_backend_preconditioned(
        &backend,
        &operator,
        &IdentityBackendPreconditioner,
        &rhs,
        None,
        &mut solution,
        &GmresBackendConfig::<Complex64> {
            max_iterations: 8,
            tolerance: 1.0e-10,
        },
    )
    .map(|_| ())
    .map_err(|error| match error {
        sonium_compute::GmresBackendError::Backend { source } => {
            SpeakerBackendError::Backend(source)
        }
        other => SpeakerBackendError::Compute(other.to_string()),
    });
    let failure_error = match failure {
        Err(SpeakerBackendError::Backend(error)) => error,
        Ok(()) => {
            return Err(SpeakerBackendError::Compute(format!(
                "SM1 {expected:?} injection was not consumed by the production coupled solve"
            )));
        }
        Err(other) => return Err(other),
    };
    let operation = backend_operation(&failure_error);
    if operation != expected.1 {
        return Err(SpeakerBackendError::Compute(format!(
            "SM1 {expected:?} injection returned unexpected operation {operation}"
        )));
    }
    let context_disposition = failure_error.context_disposition();
    let output_disposition = failure_error.output_disposition();
    let failure_stats = backend.runtime_stats_since(failure_stats_before);
    let failure_stage = failure_stage(&failure_error);
    let detail = failure_error.to_string();

    // Nothing from the failed attempt is read or serialized. All owned device
    // values are dropped before the backend itself, giving the ledger a direct
    // post-unwind zero-live-allocation observation.
    drop(operator);
    drop(rhs);
    drop(solution);
    let unwind_stats = backend.runtime_stats();
    let resources_released =
        unwind_stats.live_allocation_bytes == 0 && unwind_stats.live_staging_bytes == 0;
    drop(backend);

    // Recreate the context for both lanes. This is mandatory for device loss;
    // it is also a conservative, valid recovery for a launch failure after
    // the failed runner has unwound its local operator and buffers.
    let retry = run_coupled_reference(
        model,
        frequency_hz,
        radiation_impedance,
        ArtifactBackend::Cuda,
        false,
    )?;
    retry
        .reference
        .validate_against_golden(
            &cpu_golden.reference,
            crate::driver_model::DriverModelReferenceTolerance {
                absolute: 1.0e-9,
                relative: 1.0e-7,
            },
        )
        .map_err(SpeakerBackendError::DriverModel)?;
    let retry_transfer = &retry.reference.execution.transfer;
    if retry.reference.execution.actual_backend != ArtifactBackend::Cuda
        || retry.reference.provenance.backend != ArtifactBackend::Cuda
        || retry_transfer.h2d_operations == 0
        || retry_transfer.d2h_operations == 0
        || retry_transfer.allocation_high_water_bytes == 0
        || retry.max_relative_residual > 1.0e-6
        || retry.max_power_balance_relative_error > 5.0e-2
        || retry.max_velocity_relative_error > 1.0e-7
        || retry.max_impedance_relative_error > 1.0e-7
        || retry.max_derivative_relative_error > 1.0e-7
    {
        return Err(SpeakerBackendError::Compute(
            "SM1 recreated-context retry did not satisfy CUDA/parity gates".to_string(),
        ));
    }

    Ok(serde_json::json!({
        "case_id": "speaker.cuda-bounded-coupled-failure-contract.v1",
        "schema_version": 1,
        "status": "passed",
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "injection": {
            "point": expected.0,
            "one_shot": true,
            "native_backend": "cuda",
            "expected_operation": expected.1,
        },
        "capability_matrix": matrix,
        "failure_diagnostics": {
            "backend": "cuda",
            "stage": failure_stage.label(),
            "operation": operation,
            "detail": detail,
            "typed_error": format!("{failure_error:?}"),
            "partial_result_published": false,
            "resources_released_on_unwind": resources_released,
            "context_recovery": context_recovery_label(context_disposition),
            "output_disposition": output_disposition_label(output_disposition),
        },
        "publication": {
            "partial_result_published": false,
            "success_artifact_published_before_retry": false,
            "retry_artifact_published": true,
        },
        "unwind": {
            "owned_resources_drop": "Rust error-unwind Drop",
            "resources_released_on_unwind": resources_released,
            "allocation_ledger": {
                "failure_live_allocation_bytes": failure_stats.live_allocation_bytes,
                "failure_allocation_high_water_bytes": failure_stats.allocation_high_water_bytes,
                "failure_h2d_operations": failure_stats.transfer.h2d_operations,
                "failure_d2h_operations": failure_stats.transfer.d2h_operations,
                "post_unwind_live_allocation_bytes": unwind_stats.live_allocation_bytes,
                "post_unwind_live_staging_bytes": unwind_stats.live_staging_bytes,
            },
        },
        "recovery": {
            "required": context_recovery_label(context_disposition),
            "retry_mode": "recreated-context",
            "context_recreated": true,
            "retry_success": true,
        },
        "cpu_reference": {
            "backend": "cpu",
            "precision": "f64",
            "independent": true,
            "frequency_count": cpu_golden.reference.points.len(),
            "comparison": "recreated CUDA artifact validated against independent CPU/f64 golden",
        },
        "retry_success": {
            "backend": {
                "requested": "cuda",
                "actual": "cuda",
                "fallback_status": "none",
                "precision": "f64",
                "device": retry.reference.execution.device,
            },
            "frequency_count": retry.reference.points.len(),
            "all_frequencies_converged": retry.reference.execution.convergence.converged,
            "max_relative_residual": retry.max_relative_residual,
            "max_power_balance_relative_error": retry.max_power_balance_relative_error,
            "max_velocity_relative_error": retry.max_velocity_relative_error,
            "max_impedance_relative_error": retry.max_impedance_relative_error,
            "max_derivative_relative_error": retry.max_derivative_relative_error,
            "transfer": retry_transfer,
            "residency": {
                "allocation_high_water_bytes": retry_transfer.allocation_high_water_bytes,
                "live_allocation_bytes": retry_transfer.live_allocation_bytes,
                "h2d_operations": retry_transfer.h2d_operations,
                "d2h_operations": retry_transfer.d2h_operations,
            },
        },
        "unsupported_high_level_cells": [
            "magnetic", "structural", "acoustic", "thermal", "nonlinear_material"
        ],
        "scope": "bounded typed electromechanical 2x2 network; no claim of high-level FEM/BEM device residency"
    }))
}

fn run_cpu_reference(
    model: &DriverModel,
    frequency_hz: &[f64],
    radiation_impedance: &[Complex64],
) -> Result<CoupledBackendEvidence, SpeakerBackendError> {
    let reference =
        DriverModelReferenceArtifact::from_model(model, frequency_hz, radiation_impedance)?;
    Ok(CoupledBackendEvidence {
        capability_matrix: SpeakerCapabilityMatrix::cpu(),
        reference,
        max_relative_residual: 0.0,
        max_power_balance_relative_error: 0.0,
        max_velocity_relative_error: 0.0,
        max_impedance_relative_error: 0.0,
        max_derivative_relative_error: 0.0,
    })
}

fn run_cpu_fallback(
    model: &DriverModel,
    frequency_hz: &[f64],
    radiation_impedance: &[Complex64],
) -> Result<CoupledBackendEvidence, SpeakerBackendError> {
    let mut evidence = run_cpu_reference(model, frequency_hz, radiation_impedance)?;
    evidence.capability_matrix.requested_backend = ArtifactBackend::Cuda;
    evidence.capability_matrix.device = "cpu-fallback".to_string();
    let mut execution = evidence.reference.execution.clone();
    execution.requested_backend = ArtifactBackend::Cuda;
    execution.actual_backend = ArtifactBackend::Cpu;
    execution.device = "cpu-fallback".to_string();
    execution.toolchain = evidence.capability_matrix.toolchain.clone();
    evidence.reference.provenance = crate::conventions::ArtifactProvenance::native_preview();
    evidence.reference = evidence.reference.with_execution(execution)?;
    Ok(evidence)
}

#[cfg(not(feature = "cuda"))]
fn run_cuda_reference(
    _model: &DriverModel,
    _frequency_hz: &[f64],
    _radiation_impedance: &[Complex64],
) -> Result<CoupledBackendEvidence, SpeakerBackendError> {
    let _ = SpeakerCapabilityMatrix::cuda_without_feature();
    Err(SpeakerBackendError::Unavailable(
        "sonium-sim-speaker-modeling was built without feature `cuda`".to_string(),
    ))
}

#[cfg(feature = "cuda")]
fn run_cuda_reference(
    model: &DriverModel,
    frequency_hz: &[f64],
    radiation_impedance: &[Complex64],
) -> Result<CoupledBackendEvidence, SpeakerBackendError> {
    use sonium_compute::{
        Backend, CudaBackend, DeviceCsr, GmresBackendConfig, IdentityBackendPreconditioner,
        PersistentDeviceCsr, gmres_backend_preconditioned,
    };

    model.validate()?;
    if model.thermal_compression_spec.is_some() || model.thermal_network.is_some() {
        return Err(SpeakerBackendError::Unsupported {
            subsystem: SpeakerSubsystem::Thermal,
            reason: "bounded CUDA path has no thermal-state or temperature-feedback kernel"
                .to_string(),
        });
    }
    if model.nonlinear_motor.is_some() || model.nonlinear_operating_domain_evidence.is_some() {
        return Err(SpeakerBackendError::Unsupported {
            subsystem: SpeakerSubsystem::NonlinearMaterial,
            reason: "bounded CUDA path accepts only the validated linear DriverModel network"
                .to_string(),
        });
    }
    let golden =
        DriverModelReferenceArtifact::from_model(model, frequency_hz, radiation_impedance)?;
    let backend = CudaBackend::new_default()
        .map_err(|error| SpeakerBackendError::Unavailable(error.to_string()))?;
    let capabilities = backend.capabilities();
    let csr_state = capabilities.csr_spmv.state_for::<Complex64>();
    let blas_state = capabilities.blas1.state_for::<Complex64>();
    let dot_state = capabilities.multi_dot.state_for::<Complex64>();
    let sparse_state =
        if csr_state.is_native() && blas_state.is_native() && dot_state.is_available() {
            if dot_state.uses_host_fallback() {
                SpeakerCapabilityState::HostStagedFallback
            } else if matches!(csr_state, sonium_compute::CapabilityState::VendorNative)
                && matches!(blas_state, sonium_compute::CapabilityState::VendorNative)
            {
                SpeakerCapabilityState::VendorNative
            } else {
                SpeakerCapabilityState::Native
            }
        } else {
            SpeakerCapabilityState::Unsupported
        };
    let reason = if sparse_state == SpeakerCapabilityState::Unsupported {
        format!(
            "native Complex64 CSR/BLAS/reduction required (csr={csr_state:?}, blas={blas_state:?}, multi_dot={dot_state:?})"
        )
    } else {
        "cuSPARSE/cuBLAS device-resident bounded coupled solve".to_string()
    };
    let matrix = SpeakerCapabilityMatrix::native_sparse_cuda(
        ArtifactPrecision::F64,
        "cuda:ordinal=0".to_string(),
        "sonium-compute/cudarc + cuBLAS/cuSPARSE".to_string(),
        sparse_state,
        &reason,
    );
    if !matrix.state_is_available(SpeakerSubsystem::SparseCoupled) {
        return Err(SpeakerBackendError::Unsupported {
            subsystem: SpeakerSubsystem::SparseCoupled,
            reason,
        });
    }
    if frequency_hz.len() != radiation_impedance.len() || frequency_hz.is_empty() {
        return Err(SpeakerBackendError::Compute(
            "frequency and radiation grids must be non-empty and equal".to_string(),
        ));
    }

    let snapshot = backend.begin_runtime_stats();
    // The two-frequency fixture changes only the values of this fixed 2x2
    // coupled network. Upload and prebake its sparsity structure once, then
    // update the resident values between frequencies. This makes the bounded
    // CUDA seam exercise the same setup/steady-state distinction as a real
    // frequency sweep without claiming residency for the unsupported
    // high-level FEM/BEM subsystems.
    let first_network = model.network_parameters_at(frequency_hz[0])?;
    let row_ptr = [0_i32, 2, 4];
    let col_idx = [0_i32, 1, 0, 1];
    // The driver-model convention is V = Ze I + BL v and
    // 0 = -BL I + Zm v. This gives the same +BL² coupling term as
    // `DriverModel::response_with_radiation_impedance`.
    let first_values = [
        first_network.coil_impedance_ohm,
        Complex64::new(first_network.force_factor_tm, 0.0),
        Complex64::new(-first_network.force_factor_tm, 0.0),
        first_network.intrinsic_mechanical_impedance + radiation_impedance[0],
    ];
    let device_matrix = DeviceCsr::<CudaBackend, Complex64>::try_from_host(
        &backend,
        2,
        2,
        &row_ptr,
        &col_idx,
        &first_values,
    )
    .map_err(|error| SpeakerBackendError::Compute(error.to_string()))?;
    let mut operator = PersistentDeviceCsr::try_from_device_csr(&backend, device_matrix)
        .map_err(|error| SpeakerBackendError::Compute(error.to_string()))?;
    let mut rhs = backend
        .raw_alloc::<Complex64>(2)
        .map_err(|error| SpeakerBackendError::Compute(error.to_string()))?;
    let mut solution = backend
        .raw_alloc::<Complex64>(2)
        .map_err(|error| SpeakerBackendError::Compute(error.to_string()))?;
    let mut points = Vec::with_capacity(frequency_hz.len());
    let mut max_residual = 0.0_f64;
    let mut max_power_error = 0.0_f64;
    let mut max_velocity_error = 0.0_f64;
    let mut max_impedance_error = 0.0_f64;
    let mut max_derivative_error = 0.0_f64;
    for (point_index, (&frequency, &radiation)) in
        frequency_hz.iter().zip(radiation_impedance).enumerate()
    {
        let network = model.network_parameters_at(frequency)?;
        let values = [
            network.coil_impedance_ohm,
            Complex64::new(network.force_factor_tm, 0.0),
            Complex64::new(-network.force_factor_tm, 0.0),
            network.intrinsic_mechanical_impedance + radiation,
        ];
        if point_index != 0 {
            operator
                .try_copy_values_from_host(&backend, &values)
                .map_err(|error| SpeakerBackendError::Compute(error.to_string()))?;
        }
        let voltage = Complex64::new(model.reference_voltage_rms_v, 0.0);
        backend
            .try_copy_h2d(&[voltage, Complex64::new(0.0, 0.0)], &mut rhs)
            .map_err(|error| SpeakerBackendError::Compute(error.to_string()))?;
        let status = gmres_backend_preconditioned(
            &backend,
            &operator,
            &IdentityBackendPreconditioner,
            &rhs,
            None,
            &mut solution,
            &GmresBackendConfig::<Complex64> {
                max_iterations: 8,
                tolerance: 1.0e-10,
            },
        )
        .map_err(|error| SpeakerBackendError::Compute(error.to_string()))?;
        let (current, velocity) = if status.converged {
            let mut host = [Complex64::new(0.0, 0.0); 2];
            backend
                .try_copy_d2h(&solution, &mut host)
                .map_err(|error| SpeakerBackendError::Compute(error.to_string()))?;
            (host[0], host[1])
        } else {
            (Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0))
        };
        let residual = status.final_residual;
        if !residual.is_finite()
            || !current.re.is_finite()
            || !current.im.is_finite()
            || !velocity.re.is_finite()
            || !velocity.im.is_finite()
        {
            return Err(SpeakerBackendError::NonFinite);
        }
        if residual > 1.0e-6 {
            return Err(SpeakerBackendError::Nonconverged {
                frequency_hz: frequency,
                residual,
            });
        }
        let electrical = Complex64::new(model.reference_voltage_rms_v, 0.0) / current;
        let (cpu_velocity, cpu_impedance) =
            model.response_with_radiation_impedance(frequency, radiation)?;
        let velocity_error = relative_error(velocity, cpu_velocity);
        let impedance_error = relative_error(electrical, cpu_impedance);
        let derivative_error = relative_error(
            velocity / model.reference_voltage_rms_v,
            cpu_velocity / model.reference_voltage_rms_v,
        );
        let power_error = power_balance_error(
            network.coil_impedance_ohm,
            network.intrinsic_mechanical_impedance,
            radiation,
            voltage,
            current,
            velocity,
        );
        max_residual = max_residual.max(residual);
        max_power_error = max_power_error.max(power_error);
        max_velocity_error = max_velocity_error.max(velocity_error);
        max_impedance_error = max_impedance_error.max(impedance_error);
        max_derivative_error = max_derivative_error.max(derivative_error);
        points.push(DriverModelReferencePoint::from_response(
            frequency, radiation, velocity, electrical,
        ));
    }
    backend
        .try_synchronize()
        .map_err(|error| SpeakerBackendError::Compute(error.to_string()))?;
    let runtime = backend.runtime_stats_since(snapshot);
    let transfer = DriverModelTransferEvidence {
        h2d_operations: runtime.transfer.h2d_operations as u64,
        h2d_bytes: runtime.transfer.h2d_bytes as u64,
        d2h_operations: runtime.transfer.d2h_operations as u64,
        d2h_bytes: runtime.transfer.d2h_bytes as u64,
        live_allocation_bytes: runtime.live_allocation_bytes as u64,
        allocation_high_water_bytes: runtime.allocation_high_water_bytes as u64,
        live_staging_bytes: runtime.live_staging_bytes as u64,
        staging_high_water_bytes: runtime.staging_high_water_bytes as u64,
    };
    if transfer.h2d_operations == 0 || transfer.d2h_operations == 0 {
        return Err(SpeakerBackendError::Compute(
            "CUDA evidence interval observed no host/device transfer".to_string(),
        ));
    }
    let execution = DriverModelReferenceExecution {
        requested_backend: ArtifactBackend::Cuda,
        actual_backend: ArtifactBackend::Cuda,
        precision: ArtifactPrecision::F64,
        device: matrix.device.clone(),
        toolchain: matrix.toolchain.clone(),
        mesh_identity: golden.execution.mesh_identity.clone(),
        frequency_grid_identity: golden.execution.frequency_grid_identity.clone(),
        convergence: DriverModelConvergenceEvidence {
            converged: true,
            max_relative_residual: max_residual,
            power_closure_relative_error: max_power_error,
            coupling_iterations: 1,
        },
        transfer,
        promotion: DriverModelPromotionEvidence {
            promoted: max_residual <= 1.0e-6
                && max_velocity_error <= 1.0e-7
                && max_impedance_error <= 1.0e-7
                && max_derivative_error <= 1.0e-7
                && max_power_error <= 5.0e-2,
            full_order_relative_residual: max_residual,
            projected_relative_residual: max_velocity_error.max(max_impedance_error),
            gradient_relative_error: max_derivative_error,
            maximum_constraint_violation: 0.0,
            rank_agreement: 1.0,
        },
        reference_checksum_sha256: String::new(),
    };
    let candidate = DriverModelReferenceArtifact {
        schema_version: crate::driver_model::DRIVER_MODEL_REFERENCE_ARTIFACT_SCHEMA_VERSION,
        provenance: crate::conventions::ArtifactProvenance {
            schema_version: crate::conventions::ARTIFACT_PROVENANCE_SCHEMA_VERSION,
            backend: ArtifactBackend::Cuda,
            execution_profile: ArtifactExecutionProfile::ProjectedReanalysis,
            precision: ArtifactPrecision::F64,
            deterministic: true,
        },
        execution: execution.clone(),
        points,
    }
    .with_execution(execution)?;
    candidate.validate_against_golden(
        &golden,
        crate::driver_model::DriverModelReferenceTolerance {
            absolute: 1.0e-9,
            relative: 1.0e-7,
        },
    )?;
    candidate.validate_rom_promotion()?;
    Ok(CoupledBackendEvidence {
        capability_matrix: matrix,
        reference: candidate,
        max_relative_residual: max_residual,
        max_power_balance_relative_error: max_power_error,
        max_velocity_relative_error: max_velocity_error,
        max_impedance_relative_error: max_impedance_error,
        max_derivative_relative_error: max_derivative_error,
    })
}

/// Execute the production axisymmetric acoustic/structural/electrical
/// harmonic operator on CUDA while retaining its sparse structure across a
/// bounded frequency step.  FEM assembly remains host-owned; only the
/// assembled production operator and probe vector cross the backend boundary.
/// The output is published only after CPU parity and the injected launch
/// failure/retry contract have passed.
#[cfg(feature = "cuda")]
pub fn run_production_frequency_step_cuda_contract() -> Result<serde_json::Value, String> {
    use sonium_compute::{
        Backend, BackendContextDisposition, BackendError, CudaBackend, CudaFailureInjectionPoint,
        HostCsrRef, PersistentDeviceCsr,
    };
    use std::time::Instant;

    const RESIDUAL_BUDGET: f64 = 1.0e-10;
    let started = Instant::now();
    let config = AxisymmetricLinearConfig::default();
    let model = AxisymmetricLinearModel::build(config)
        .map_err(|error| format!("production axisymmetric model build failed: {error}"))?;
    let frequencies_hz = [250.0_f64, 500.0_f64];
    let points = frequencies_hz.map(|frequency_hz| {
        sonium_fem::physics::acoustics::electroacoustics::HarmonicCoupledPoint {
            omega_rad_per_s: 2.0 * std::f64::consts::PI * frequency_hz,
            sound_speed_m_per_s: model.config.sound_speed_m_per_s,
            fluid_density_kg_per_m3: model.config.fluid_density_kg_per_m3,
        }
    });
    let first = model
        .coupled_system
        .assemble_harmonic(points[0])
        .map_err(|error| format!("assemble first production harmonic operator: {error}"))?
        .try_to_csr()
        .map_err(|error| format!("convert first production operator to CSR: {error}"))?;
    let second = model
        .coupled_system
        .assemble_harmonic(points[1])
        .map_err(|error| format!("assemble second production harmonic operator: {error}"))?
        .try_to_csr()
        .map_err(|error| format!("convert second production operator to CSR: {error}"))?;
    if first.num_rows != first.num_cols
        || second.num_rows != first.num_rows
        || second.num_cols != first.num_cols
        || second.row_ptrs != first.row_ptrs
        || second.col_indices != first.col_indices
    {
        return Err("production frequency step changed CSR structure".to_owned());
    }
    let dimension = first.num_rows;
    let nnz = first.values.len();
    if dimension < 100 || nnz < dimension {
        return Err(format!(
            "production frequency operator is unexpectedly small: dimension={dimension}, nnz={nnz}"
        ));
    }
    let input: Vec<Complex64> = (0..dimension)
        .map(|index| {
            Complex64::new(
                0.125 + 0.0007 * (index % 31) as f64,
                -0.075 + 0.0005 * (index % 17) as f64,
            )
        })
        .collect();
    let cpu_outputs = [&first, &second]
        .into_iter()
        .map(|matrix| sparse_apply_complex(matrix, &input))
        .collect::<Vec<_>>();

    let backend = CudaBackend::new_default()
        .map_err(|error| format!("production frequency CUDA backend unavailable: {error}"))?;
    let host = HostCsrRef::try_new(
        dimension,
        dimension,
        &first.row_ptrs,
        &first.col_indices,
        &first.values,
    )
    .map_err(|error| format!("production frequency CSR admission failed: {error}"))?;
    let mut operator = PersistentDeviceCsr::try_from_host_csr_ref(&backend, host)
        .map_err(|error| format!("production frequency CSR upload failed: {error}"))?;
    let mut input_device = backend
        .raw_alloc::<Complex64>(dimension)
        .map_err(|error| format!("production frequency input allocation failed: {error}"))?;
    let mut output_device = backend
        .raw_alloc::<Complex64>(dimension)
        .map_err(|error| format!("production frequency output allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&input, &mut input_device)
        .map_err(|error| format!("production frequency input upload failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("production frequency setup synchronization failed: {error}"))?;
    backend.reset_runtime_stats();

    let mut observations = Vec::with_capacity(frequencies_hz.len());
    let mut max_residual = 0.0_f64;
    let mut warm_h2d_operations = 0_u64;
    let mut warm_allocation_operations = 0_u64;
    let mut value_update_h2d_operations = 0_u64;
    for (index, (frequency_hz, matrix)) in frequencies_hz.iter().zip([&first, &second]).enumerate()
    {
        let update_start = backend.begin_runtime_stats();
        if index == 1 {
            operator
                .try_copy_values_from_host(&backend, &matrix.values)
                .map_err(|error| {
                    format!("production frequency resident value update failed: {error}")
                })?;
        }
        let action_start = backend.begin_runtime_stats();
        operator
            .try_apply(&backend, &input_device, &mut output_device)
            .map_err(|error| format!("production frequency CUDA action failed: {error}"))?;
        backend.try_synchronize().map_err(|error| {
            format!("production frequency action synchronization failed: {error}")
        })?;
        let mut cuda_output = vec![Complex64::default(); dimension];
        backend
            .try_copy_d2h(&output_device, &mut cuda_output)
            .map_err(|error| format!("production frequency output download failed: {error}"))?;
        let residual = max_abs_relative_complex(&cuda_output, &cpu_outputs[index]);
        if !residual.is_finite() || residual > RESIDUAL_BUDGET {
            return Err(format!(
                "production frequency CUDA/CPU residual exceeded budget at {frequency_hz} Hz: {residual:.3e}"
            ));
        }
        max_residual = max_residual.max(residual);
        let update_stats = backend.runtime_stats_since(update_start);
        let action_stats = backend.runtime_stats_since(action_start);
        if action_stats.allocation_operations != 0 {
            return Err(format!(
                "production frequency action allocated on device at {frequency_hz} Hz: {}",
                action_stats.allocation_operations
            ));
        }
        warm_h2d_operations =
            warm_h2d_operations.saturating_add(action_stats.transfer.h2d_operations as u64);
        warm_allocation_operations =
            warm_allocation_operations.saturating_add(action_stats.allocation_operations as u64);
        if index == 1 {
            value_update_h2d_operations = update_stats.transfer.h2d_operations as u64;
        }
        observations.push(serde_json::json!({
            "frequency_hz": frequency_hz,
            "dimension": dimension,
            "nnz": nnz,
            "cpu_output_l2": l2_norm_complex(&cpu_outputs[index]),
            "cuda_output_l2": l2_norm_complex(&cuda_output),
            "max_relative_residual": residual,
            "value_update": index == 1,
            "action_kernel_operations": action_stats.kernel_launches + action_stats.library_operations,
            "action_h2d_operations": action_stats.transfer.h2d_operations,
            "action_d2h_operations": action_stats.transfer.d2h_operations,
        }));
    }
    if warm_h2d_operations != 0
        || warm_allocation_operations != 0
        || value_update_h2d_operations != 1
    {
        return Err(format!(
            "production frequency residency contract failed: warm_h2d={warm_h2d_operations}, warm_allocations={warm_allocation_operations}, value_update_h2d={value_update_h2d_operations}"
        ));
    }

    // A failed launch cannot publish its output. Discard this attempt after
    // recording the typed disposition, recover the context, then retry the
    // same resident operator and probe vector before constructing the report.
    let failure_start = backend.begin_runtime_stats();
    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = operator.try_apply(&backend, &input_device, &mut output_device);
    let failure_error = match failure {
        Err(error @ BackendError::Launch { .. }) => error,
        Err(error) => {
            return Err(format!(
                "launch injection returned wrong typed error: {error}"
            ));
        }
        Ok(()) => {
            return Err(
                "launch injection was not consumed by production frequency action".to_owned(),
            );
        }
    };
    let failure_stats = backend.runtime_stats_since(failure_start);
    let disposition = failure_error.context_disposition();
    let output_disposition = failure_error.output_disposition();
    if disposition != BackendContextDisposition::SynchronizeBeforeReuse {
        return Err(format!(
            "launch failure required unexpected context disposition: {disposition:?}"
        ));
    }
    backend
        .recover_after_error()
        .map_err(|error| format!("production frequency launch recovery failed: {error}"))?;
    operator
        .try_apply(&backend, &input_device, &mut output_device)
        .map_err(|error| format!("production frequency retry failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("production frequency retry synchronization failed: {error}"))?;
    let mut retry_output = vec![Complex64::default(); dimension];
    backend
        .try_copy_d2h(&output_device, &mut retry_output)
        .map_err(|error| format!("production frequency retry output download failed: {error}"))?;
    let retry_residual = max_abs_relative_complex(&retry_output, &cpu_outputs[1]);
    if !retry_residual.is_finite() || retry_residual > RESIDUAL_BUDGET {
        return Err(format!(
            "production frequency retry residual exceeded budget: {retry_residual:.3e}"
        ));
    }
    let failure_operation = backend_operation(&failure_error);
    let live_device_bytes_before_teardown = backend.runtime_stats().live_allocation_bytes;
    drop(operator);
    drop(input_device);
    drop(output_device);
    let teardown = backend.runtime_stats();
    if teardown.live_allocation_bytes != 0 || teardown.live_staging_bytes != 0 {
        return Err(format!(
            "production frequency teardown leaked device state: live={} staging={}",
            teardown.live_allocation_bytes, teardown.live_staging_bytes
        ));
    }
    Ok(serde_json::json!({
        "case_id": "speaker.cuda-production-frequency-step.v1",
        "schema_version": 1,
        "status": "passed",
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "execution_mode": "DeviceResidentOperator",
        "dispatch": "native-cuda-cusparse",
        "operator": "AxisymmetricLinearModel::CoupledSystem::assemble_harmonic",
        "production_scope": "magnetic-bias-informed acoustic/structural/electrical harmonic FEM operator; host assembly is explicit",
        "dimension": dimension,
        "nnz": nnz,
        "frequencies_hz": frequencies_hz,
        "frequency_actions": observations,
        "production_model": {
            "magnetic_nodes": model.magnetic_mesh.num_nodes(),
            "structural_nodes": model.structural_mesh.num_nodes(),
            "acoustic_nodes": model.acoustic_mesh.num_nodes(),
            "fluid_dofs": model.coupled_system.n_fluid,
            "solid_dofs": model.coupled_system.n_solid,
            "electrical_dofs": model.coupled_system.electrical.as_ref().map_or(0, |block| block.n_dofs),
            "distributed_force_factor_tm": model.magnetic_bias.distributed_bl_tm,
            "operator_provenance": "production axisymmetric FEM assembly with magnetic bias and acoustic/structural coupling",
        },
        "residency": {
            "structure_uploaded_once": true,
            "values_updated_once_for_second_frequency": true,
            "warm_action_h2d_operations": warm_h2d_operations,
            "warm_action_allocation_operations": warm_allocation_operations,
            "frequency_value_update_h2d_operations": value_update_h2d_operations,
            "live_device_bytes_before_teardown": live_device_bytes_before_teardown,
            "live_device_bytes_after_teardown": teardown.live_allocation_bytes,
            "live_staging_bytes_after_teardown": teardown.live_staging_bytes,
        },
        "cpu_reference": {
            "kind": "independent host CSR SpMV from the same checked assembled matrices",
            "precision": "Complex64",
            "max_relative_residual": max_residual,
        },
        "launch_failure": {
            "status": "typed",
            "stage": "launch",
            "operation": failure_operation,
            "typed_error": format!("{failure_error:?}"),
            "context_recovery": "synchronize_before_reuse",
            "output_disposition": format!("{output_disposition:?}"),
            "partial_result_published": false,
            "failure_kernel_operations": failure_stats.kernel_launches + failure_stats.library_operations,
            "recovery_succeeded": true,
            "retry_succeeded": true,
            "retry_max_relative_residual": retry_residual,
        },
        "resources_released_on_teardown": true,
        "elapsed_ms": started.elapsed().as_millis() as u64,
        "scope_boundary": "Host assembly remains explicit: this cell does not claim device-resident FEM assembly, nonlinear magnetic material kernels, full frequency-sweep solve convergence, or high-level acoustic export residency.",
    }))
}

#[cfg(feature = "cuda")]
fn sparse_apply_complex(matrix: &CsrMatrix<Complex64>, input: &[Complex64]) -> Vec<Complex64> {
    let mut output = vec![Complex64::default(); matrix.num_rows];
    for (row, output_value) in output.iter_mut().enumerate() {
        for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            *output_value += matrix.values[entry] * input[matrix.col_indices[entry]];
        }
    }
    output
}

#[cfg(feature = "cuda")]
fn l2_norm_complex(values: &[Complex64]) -> f64 {
    values.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt()
}

#[cfg(feature = "cuda")]
fn max_abs_relative_complex(left: &[Complex64], right: &[Complex64]) -> f64 {
    let scale = l2_norm_complex(right).max(f64::MIN_POSITIVE);
    left.iter()
        .zip(right)
        .map(|(left, right)| (*left - *right).norm())
        .fold(0.0, f64::max)
        / scale
}

#[cfg(feature = "cuda")]
fn relative_error(actual: Complex64, expected: Complex64) -> f64 {
    (actual - expected).norm() / expected.norm().max(1.0e-30)
}

#[cfg(feature = "cuda")]
fn power_balance_error(
    coil_impedance: Complex64,
    intrinsic_mechanical_impedance: Complex64,
    radiation_impedance: Complex64,
    voltage: Complex64,
    current: Complex64,
    velocity: Complex64,
) -> f64 {
    let input = (voltage * current.conj()).re;
    let electrical_loss = coil_impedance.re * current.norm_sqr();
    let mechanical_loss = intrinsic_mechanical_impedance.re * velocity.norm_sqr();
    let radiation_power = radiation_impedance.re * velocity.norm_sqr();
    let scale = input.abs().max(1.0e-30);
    (input - electrical_loss - mechanical_loss - radiation_power).abs() / scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DriverModel, SpeakerDriverStudy};

    #[test]
    fn cpu_matrix_marks_every_subsystem_native() {
        let matrix = SpeakerCapabilityMatrix::cpu();
        assert_eq!(matrix.requested_backend, ArtifactBackend::Cpu);
        assert_eq!(matrix.actual_backend, ArtifactBackend::Cpu);
        assert_eq!(matrix.subsystems.len(), 7);
        assert_eq!(matrix.toolchain, env!("SONIUM_BUILD_RUSTC_VERSION"));
        assert!(matrix.toolchain.starts_with("rustc "));
        assert_ne!(
            matrix.toolchain,
            format!("rustc-{}", env!("CARGO_PKG_VERSION"))
        );
        assert!(
            matrix
                .subsystems
                .iter()
                .all(|entry| entry.state == SpeakerCapabilityState::Native)
        );
    }

    #[test]
    fn cpu_fallback_retains_compiler_provenance() {
        let model =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap();
        let evidence = run_cpu_fallback(&model, &[100.0], &[Complex64::new(0.2, 0.1)]).unwrap();
        assert_eq!(
            evidence.capability_matrix.toolchain,
            env!("SONIUM_BUILD_RUSTC_VERSION")
        );
        assert_eq!(
            evidence.reference.execution.toolchain,
            evidence.capability_matrix.toolchain
        );
        assert_eq!(evidence.capability_matrix.device, "cpu-fallback");
    }

    #[test]
    fn cuda_request_is_explicitly_cpu_fallback_when_allowed() {
        let model =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap();
        let evidence = run_coupled_reference(
            &model,
            &[100.0, 1_000.0],
            &[Complex64::new(0.2, 0.1), Complex64::new(0.4, 0.3)],
            ArtifactBackend::Cuda,
            true,
        )
        .unwrap();
        assert_eq!(
            evidence.reference.execution.requested_backend,
            ArtifactBackend::Cuda
        );
        #[cfg(not(feature = "cuda"))]
        assert_eq!(
            evidence.reference.execution.actual_backend,
            ArtifactBackend::Cpu
        );
        #[cfg(feature = "cuda")]
        {
            if evidence.reference.execution.actual_backend == ArtifactBackend::Cuda {
                assert!(evidence.reference.execution.transfer.h2d_operations > 0);
                assert!(evidence.reference.execution.transfer.d2h_operations > 0);
                assert!(evidence.max_relative_residual <= 1.0e-6);
                assert!(evidence.max_power_balance_relative_error <= 5.0e-2);
                assert!(evidence.max_velocity_relative_error <= 1.0e-7);
                assert!(evidence.max_impedance_relative_error <= 1.0e-7);
                assert!(evidence.max_derivative_relative_error <= 1.0e-7);
                println!(
                    "CUDA_COUPLED_EVIDENCE residual={:.3e} power={:.3e} velocity={:.3e} impedance={:.3e} derivative={:.3e} h2d={} d2h={} high_water_bytes={}",
                    evidence.max_relative_residual,
                    evidence.max_power_balance_relative_error,
                    evidence.max_velocity_relative_error,
                    evidence.max_impedance_relative_error,
                    evidence.max_derivative_relative_error,
                    evidence.reference.execution.transfer.h2d_operations,
                    evidence.reference.execution.transfer.d2h_operations,
                    evidence
                        .reference
                        .execution
                        .transfer
                        .allocation_high_water_bytes,
                );
            } else {
                assert!(std::env::var_os("SONIUM_SIM_SPEAKER_REQUIRE_CUDA").is_none());
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_rejects_unsupported_thermal_state_before_execution() {
        let mut model =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap();
        model.thermal_network = Some(crate::DriverThermalNetworkConfig::default());
        let result = run_coupled_reference(
            &model,
            &[1_000.0],
            &[Complex64::new(0.4, 0.3)],
            ArtifactBackend::Cuda,
            false,
        );
        assert!(matches!(
            result,
            Err(SpeakerBackendError::Unsupported {
                subsystem: SpeakerSubsystem::Thermal,
                ..
            })
        ));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_live_coupled_reference_is_not_a_cpu_fallback() {
        let model =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap();
        let evidence = match run_coupled_reference(
            &model,
            &[100.0, 1_000.0],
            &[Complex64::new(0.2, 0.1), Complex64::new(0.4, 0.3)],
            ArtifactBackend::Cuda,
            false,
        ) {
            Ok(evidence) => evidence,
            Err(
                error @ (SpeakerBackendError::Unavailable(_)
                | SpeakerBackendError::Unsupported { .. }),
            ) if std::env::var_os("SONIUM_SIM_SPEAKER_REQUIRE_CUDA").is_none() => {
                eprintln!("CUDA live evidence skipped: {error}");
                return;
            }
            Err(error) => panic!("CUDA live evidence failed: {error}"),
        };
        assert_eq!(
            evidence.reference.execution.actual_backend,
            ArtifactBackend::Cuda
        );
        assert!(evidence.reference.execution.transfer.h2d_operations > 0);
        assert!(evidence.reference.execution.transfer.d2h_operations > 0);
        assert!(evidence.max_relative_residual <= 1.0e-6);
        assert!(evidence.max_power_balance_relative_error <= 5.0e-2);
        assert!(evidence.max_velocity_relative_error <= 1.0e-7);
        assert!(evidence.max_impedance_relative_error <= 1.0e-7);
        assert!(evidence.max_derivative_relative_error <= 1.0e-7);
        println!(
            "CUDA_COUPLED_NATIVE_EVIDENCE residual={:.3e} power={:.3e} velocity={:.3e} impedance={:.3e} derivative={:.3e} h2d={} d2h={} high_water_bytes={}",
            evidence.max_relative_residual,
            evidence.max_power_balance_relative_error,
            evidence.max_velocity_relative_error,
            evidence.max_impedance_relative_error,
            evidence.max_derivative_relative_error,
            evidence.reference.execution.transfer.h2d_operations,
            evidence.reference.execution.transfer.d2h_operations,
            evidence
                .reference
                .execution
                .transfer
                .allocation_high_water_bytes,
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_failure_contract_is_typed_and_retries_without_partial_publication() {
        let model =
            DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
                .unwrap();
        for point in [
            sonium_compute::CudaFailureInjectionPoint::Launch,
            sonium_compute::CudaFailureInjectionPoint::DeviceLoss,
        ] {
            let evidence = match run_coupled_failure_contract(
                &model,
                &[100.0, 1_000.0],
                &[Complex64::new(0.2, 0.1), Complex64::new(0.4, 0.3)],
                point,
            ) {
                Ok(evidence) => evidence,
                Err(SpeakerBackendError::Unavailable(error))
                    if std::env::var_os("SONIUM_SIM_SPEAKER_REQUIRE_CUDA").is_none() =>
                {
                    eprintln!("CUDA failure contract skipped: {error}");
                    return;
                }
                Err(error) => panic!("CUDA failure contract failed: {error}"),
            };
            assert_eq!(evidence["status"], "passed");
            assert_eq!(evidence["actual_backend"], "cuda");
            assert_eq!(
                evidence["failure_diagnostics"]["partial_result_published"],
                false
            );
            assert_eq!(evidence["publication"]["partial_result_published"], false);
            assert_eq!(evidence["recovery"]["context_recreated"], true);
            assert_eq!(evidence["recovery"]["retry_success"], true);
            assert_eq!(evidence["cpu_reference"]["independent"], true);
        }
    }
}
