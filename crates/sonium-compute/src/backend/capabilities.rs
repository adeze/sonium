//! Machine-readable capability tables for compute backends.
//!
//! Moved verbatim from `backend/mod.rs`; see [`super::Backend`] for the
//! trait surface that reports these tables.

use super::ReductionMode;
use crate::error::{BackendError, BackendResult};
use crate::scalar::DeviceScalar;
use std::any::TypeId;

/// Residency/performance class for an operation on a backend.
///
/// This is intentionally a capability contract rather than an availability
/// probe: if an operation falls back to the host, that fallback is visible in
/// the table and can be asserted by tests or CI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CapabilityState {
    /// Implemented by Sonium kernels on the backend where the buffers live.
    Native,
    /// Implemented by a vendor library without staging through host memory.
    VendorNative,
    /// Implemented on the CPU against host-visible/shared device storage.
    HostSharedFallback,
    /// Implemented by explicitly copying through host memory.
    HostStagedFallback,
    /// Not supported by this backend.
    Unsupported,
}

impl CapabilityState {
    /// Whether the operation executes without a host-residency fallback.
    #[must_use]
    pub const fn is_native(self) -> bool {
        matches!(self, Self::Native | Self::VendorNative)
    }

    /// Whether the operation is implemented, possibly through host memory.
    #[must_use]
    pub const fn is_available(self) -> bool {
        !matches!(self, Self::Unsupported)
    }

    /// Whether the operation may require CPU execution or host staging.
    #[must_use]
    pub const fn uses_host_fallback(self) -> bool {
        matches!(self, Self::HostSharedFallback | Self::HostStagedFallback)
    }
}

/// Capability states for the scalar types used by Sonium kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarCapabilities {
    pub f32: CapabilityState,
    pub f64: CapabilityState,
    pub c32: CapabilityState,
    pub c64: CapabilityState,
}

impl ScalarCapabilities {
    pub fn state_for<T: DeviceScalar>(self) -> CapabilityState {
        let type_id = TypeId::of::<T>();
        if type_id == TypeId::of::<f32>() {
            self.f32
        } else if type_id == TypeId::of::<f64>() {
            self.f64
        } else if type_id == TypeId::of::<num_complex::Complex32>() {
            self.c32
        } else if type_id == TypeId::of::<num_complex::Complex64>() {
            self.c64
        } else {
            CapabilityState::Unsupported
        }
    }

    pub fn is_unsupported(self) -> bool {
        matches!(self.f32, CapabilityState::Unsupported)
            && matches!(self.f64, CapabilityState::Unsupported)
            && matches!(self.c32, CapabilityState::Unsupported)
            && matches!(self.c64, CapabilityState::Unsupported)
    }

    pub fn is_unsupported_for<T: DeviceScalar>(self) -> bool {
        matches!(self.state_for::<T>(), CapabilityState::Unsupported)
    }

    pub const fn splat(state: CapabilityState) -> Self {
        Self {
            f32: state,
            f64: state,
            c32: state,
            c64: state,
        }
    }

    pub const fn new(
        f32: CapabilityState,
        f64: CapabilityState,
        c32: CapabilityState,
        c64: CapabilityState,
    ) -> Self {
        Self { f32, f64, c32, c64 }
    }
}

/// Residency and precision capabilities for the BEM execution phases.
///
/// The pair-kernel fields retain the historical Complex32/Complex64 names
/// because those are the compute ABI's output types. The remaining phases
/// use [`ScalarCapabilities`] so callers can distinguish an implemented
/// Complex64 path from an explicit host fallback or an unsupported cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BemCapabilities {
    pub regular_pair_c32: CapabilityState,
    pub regular_pair_c64: CapabilityState,
    pub near_pair: ScalarCapabilities,
    pub singular_pair: ScalarCapabilities,
    pub hypersingular_pair: ScalarCapabilities,
    pub field_evaluation: ScalarCapabilities,
    /// Tree/admissibility/compression setup for a hierarchical operator.
    pub hmatrix_setup: ScalarCapabilities,
    /// Device-resident hierarchical forward/adjoint actions.
    pub hmatrix: ScalarCapabilities,
    /// Tree/translation setup for an FMM operator.
    pub fmm_setup: ScalarCapabilities,
    /// Device-resident FMM phase and matvec actions.
    pub fmm: ScalarCapabilities,
}

impl BemCapabilities {
    pub const fn unsupported() -> Self {
        let unsupported = CapabilityState::Unsupported;
        Self {
            regular_pair_c32: unsupported,
            regular_pair_c64: unsupported,
            near_pair: ScalarCapabilities::splat(unsupported),
            singular_pair: ScalarCapabilities::splat(unsupported),
            hypersingular_pair: ScalarCapabilities::splat(unsupported),
            field_evaluation: ScalarCapabilities::splat(unsupported),
            hmatrix_setup: ScalarCapabilities::splat(unsupported),
            hmatrix: ScalarCapabilities::splat(unsupported),
            fmm_setup: ScalarCapabilities::splat(unsupported),
            fmm: ScalarCapabilities::splat(unsupported),
        }
    }
}

/// Capability states for precision-conversion operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrecisionCastCapabilities {
    pub f64_to_f32: CapabilityState,
    pub c64_to_c32: CapabilityState,
    pub f32_to_f64: CapabilityState,
    pub c32_to_c64: CapabilityState,
}

/// Capability states for the two reduction policies exposed by [`super::Backend`].
///
/// Keeping the policies separate is important: a CUDA backend can provide a
/// vendor-native fast reduction while deterministic mode intentionally stages
/// through the host to establish a fixed summation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionOrdering {
    /// The backend or vendor library chooses the reduction tree/order.
    BackendDefined,
    /// Elements are accumulated in their logical input order.
    FixedInputOrder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionAccumulation {
    /// Accumulate using the native scalar/kernel precision.
    NativeScalar,
    /// Accumulate through the checked host scalar path in logical order.
    HostScalar,
}

/// Synchronization behavior of a reduction policy's checked implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionSynchronization {
    /// The backend or vendor library controls synchronization.
    BackendDefined,
    /// The checked implementation transfers values to the host in logical
    /// order before accumulating them.
    HostTransfer,
}

/// Failure behavior promised by an explicit, checked reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionFailureGuarantee {
    /// Shape, transfer, device, and non-finite failures are returned as a
    /// typed `BackendError`.
    TypedError,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReductionPolicy {
    pub ordering: ReductionOrdering,
    pub accumulation: ReductionAccumulation,
    pub synchronization: ReductionSynchronization,
    pub failure: ReductionFailureGuarantee,
}

impl ReductionPolicy {
    pub const FAST: Self = Self {
        ordering: ReductionOrdering::BackendDefined,
        accumulation: ReductionAccumulation::NativeScalar,
        synchronization: ReductionSynchronization::BackendDefined,
        failure: ReductionFailureGuarantee::TypedError,
    };

    /// Reproducible reductions currently use the same fixed host order as
    /// deterministic reductions.  Keeping the mode distinct lets callers ask
    /// for repeatability without coupling their API to that stronger current
    /// implementation guarantee.
    pub const REPRODUCIBLE: Self = Self {
        ordering: ReductionOrdering::FixedInputOrder,
        accumulation: ReductionAccumulation::HostScalar,
        synchronization: ReductionSynchronization::HostTransfer,
        failure: ReductionFailureGuarantee::TypedError,
    };

    pub const DETERMINISTIC: Self = Self {
        ordering: ReductionOrdering::FixedInputOrder,
        accumulation: ReductionAccumulation::HostScalar,
        synchronization: ReductionSynchronization::HostTransfer,
        failure: ReductionFailureGuarantee::TypedError,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReductionCapabilities {
    pub fast: ScalarCapabilities,
    pub reproducible: ScalarCapabilities,
    pub deterministic: ScalarCapabilities,
    pub fast_policy: ReductionPolicy,
    pub reproducible_policy: ReductionPolicy,
    pub deterministic_policy: ReductionPolicy,
}

impl ReductionCapabilities {
    pub const fn splat(state: CapabilityState) -> Self {
        Self {
            fast: ScalarCapabilities::splat(state),
            reproducible: ScalarCapabilities::splat(state),
            deterministic: ScalarCapabilities::splat(state),
            fast_policy: ReductionPolicy::FAST,
            reproducible_policy: ReductionPolicy::REPRODUCIBLE,
            deterministic_policy: ReductionPolicy::DETERMINISTIC,
        }
    }

    pub const fn new(fast: ScalarCapabilities, deterministic: ScalarCapabilities) -> Self {
        Self {
            fast,
            reproducible: deterministic,
            deterministic,
            fast_policy: ReductionPolicy::FAST,
            reproducible_policy: ReductionPolicy::REPRODUCIBLE,
            deterministic_policy: ReductionPolicy::DETERMINISTIC,
        }
    }

    pub const fn with_reproducible(
        fast: ScalarCapabilities,
        reproducible: ScalarCapabilities,
        deterministic: ScalarCapabilities,
    ) -> Self {
        Self {
            fast,
            reproducible,
            deterministic,
            fast_policy: ReductionPolicy::FAST,
            reproducible_policy: ReductionPolicy::REPRODUCIBLE,
            deterministic_policy: ReductionPolicy::DETERMINISTIC,
        }
    }

    pub const fn for_mode(self, mode: ReductionMode) -> ScalarCapabilities {
        match mode {
            ReductionMode::Fast => self.fast,
            ReductionMode::Reproducible => self.reproducible,
            ReductionMode::Deterministic => self.deterministic,
        }
    }

    pub const fn policy_for(self, mode: ReductionMode) -> ReductionPolicy {
        match mode {
            ReductionMode::Fast => self.fast_policy,
            ReductionMode::Reproducible => self.reproducible_policy,
            ReductionMode::Deterministic => self.deterministic_policy,
        }
    }
}

impl PrecisionCastCapabilities {
    pub const fn splat(state: CapabilityState) -> Self {
        Self {
            f64_to_f32: state,
            c64_to_c32: state,
            f32_to_f64: state,
            c32_to_c64: state,
        }
    }
}

/// Machine-readable summary of the backend surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub backend_name: &'static str,
    pub blas1: ScalarCapabilities,
    pub gemv: ScalarCapabilities,
    pub gemm: ScalarCapabilities,
    /// Triangular solve with multiple right-hand sides (`trsm`).
    pub trsm: ScalarCapabilities,
    /// Symmetric rank-k update (`syrk`).
    pub syrk: ScalarCapabilities,
    /// Complex FFT execution, including the scalar types used by the
    /// complex buffer (`f32`/`f64` map to Complex32/Complex64).
    pub fft: ScalarCapabilities,
    /// Real-input FFT execution. The scalar fields refer to the real input
    /// precision (`f32`/`f64`); complex fields are unsupported by contract.
    pub real_fft: ScalarCapabilities,
    /// DCT-II/III and their adjoints over real device buffers.
    pub dct: ScalarCapabilities,
    /// DST-II/III and their adjoints over real device buffers.
    pub dst: ScalarCapabilities,
    pub csr_spmv: ScalarCapabilities,
    pub csr_spmm: ScalarCapabilities,
    /// CSR(Aᵀ) construction from a resident CSR matrix.
    pub csr_transpose: ScalarCapabilities,
    /// Fused `y <- y + Σ alpha[i] * x_basis[i]` update.
    pub axpy_many: ScalarCapabilities,
    pub multi_dot: ScalarCapabilities,
    pub multi_dot_strided: ScalarCapabilities,
    pub basis_axpy_strided: ScalarCapabilities,
    pub basis_project_out_strided: ScalarCapabilities,
    pub gather4_weighted: ScalarCapabilities,
    /// Segmented reduction; native kernels or an explicit host fallback.
    pub segmented_reduce: ScalarCapabilities,
    /// Batched local contraction; native kernels or an explicit host fallback.
    pub batched_local_contract: ScalarCapabilities,
    pub reduction: ReductionCapabilities,
    pub precision_cast: PrecisionCastCapabilities,
    pub bem_regular_pair_c32: CapabilityState,
    /// Double-precision regular triangle-pair BEM kernel.
    pub bem_regular_pair_c64: CapabilityState,
    /// Scalar- and phase-specific BEM capability report.
    pub bem: BemCapabilities,
}

impl BackendCapabilities {
    pub const fn unsupported(backend_name: &'static str) -> Self {
        let unsupported = CapabilityState::Unsupported;
        Self {
            backend_name,
            blas1: ScalarCapabilities::splat(unsupported),
            gemv: ScalarCapabilities::splat(unsupported),
            gemm: ScalarCapabilities::splat(unsupported),
            trsm: ScalarCapabilities::splat(unsupported),
            syrk: ScalarCapabilities::splat(unsupported),
            fft: ScalarCapabilities::splat(unsupported),
            real_fft: ScalarCapabilities::splat(unsupported),
            dct: ScalarCapabilities::splat(unsupported),
            dst: ScalarCapabilities::splat(unsupported),
            csr_spmv: ScalarCapabilities::splat(unsupported),
            csr_spmm: ScalarCapabilities::splat(unsupported),
            csr_transpose: ScalarCapabilities::splat(unsupported),
            axpy_many: ScalarCapabilities::splat(unsupported),
            multi_dot: ScalarCapabilities::splat(unsupported),
            multi_dot_strided: ScalarCapabilities::splat(unsupported),
            basis_axpy_strided: ScalarCapabilities::splat(unsupported),
            basis_project_out_strided: ScalarCapabilities::splat(unsupported),
            gather4_weighted: ScalarCapabilities::splat(unsupported),
            segmented_reduce: ScalarCapabilities::splat(unsupported),
            batched_local_contract: ScalarCapabilities::splat(unsupported),
            reduction: ReductionCapabilities::splat(unsupported),
            precision_cast: PrecisionCastCapabilities::splat(unsupported),
            bem_regular_pair_c32: unsupported,
            bem_regular_pair_c64: unsupported,
            bem: BemCapabilities::unsupported(),
        }
    }

    /// Effective residency class for a multi-dot reduction mode.
    ///
    /// The operation table describes the fused vector update, while the
    /// reduction table describes the mode-specific accumulation and
    /// synchronization path. Combining them prevents callers from treating a
    /// fast native kernel as native when reproducible/deterministic mode must
    /// stage through the host (or when either half is unsupported).
    #[must_use]
    pub fn multi_dot_capability_for<T: DeviceScalar>(
        &self,
        mode: ReductionMode,
    ) -> CapabilityState {
        let operation = self.multi_dot.state_for::<T>();
        let reduction = self.reduction.for_mode(mode).state_for::<T>();
        match (operation, reduction) {
            (CapabilityState::Unsupported, _) | (_, CapabilityState::Unsupported) => {
                CapabilityState::Unsupported
            }
            (CapabilityState::HostStagedFallback, _) | (_, CapabilityState::HostStagedFallback) => {
                CapabilityState::HostStagedFallback
            }
            (CapabilityState::HostSharedFallback, _) | (_, CapabilityState::HostSharedFallback) => {
                CapabilityState::HostSharedFallback
            }
            (CapabilityState::VendorNative, _) | (_, CapabilityState::VendorNative) => {
                CapabilityState::VendorNative
            }
            (CapabilityState::Native, CapabilityState::Native) => CapabilityState::Native,
        }
    }

    /// Require the native/vendor-native BEM regular-pair kernel.
    ///
    /// A capability table is intentionally more precise than a boolean
    /// "backend available" flag: a configured CUDA/Metal backend may still
    /// lack a particular kernel and fall back to host execution.  Callers
    /// that requested device residency should use this guard before building
    /// a large batch.
    pub fn require_native_bem_regular_pair(&self) -> BackendResult<()> {
        match self.bem_regular_pair_c32 {
            CapabilityState::Native | CapabilityState::VendorNative => Ok(()),
            CapabilityState::HostSharedFallback | CapabilityState::HostStagedFallback => {
                Err(BackendError::HostFallback {
                    backend: self.backend_name,
                    operation: "bem_regular_pair_c32_batch",
                    scalar: "Complex32",
                })
            }
            CapabilityState::Unsupported => Err(BackendError::Unsupported {
                backend: self.backend_name,
                operation: "bem_regular_pair_c32_batch",
                scalar: "Complex32",
            }),
        }
    }

    /// Require the native/vendor-native double-precision BEM pair kernel.
    pub fn require_native_bem_regular_pair_c64(&self) -> BackendResult<()> {
        match self.bem_regular_pair_c64 {
            CapabilityState::Native | CapabilityState::VendorNative => Ok(()),
            CapabilityState::HostSharedFallback | CapabilityState::HostStagedFallback => {
                Err(BackendError::HostFallback {
                    backend: self.backend_name,
                    operation: "bem_regular_pair_c64_batch",
                    scalar: "Complex64",
                })
            }
            CapabilityState::Unsupported => Err(BackendError::Unsupported {
                backend: self.backend_name,
                operation: "bem_regular_pair_c64_batch",
                scalar: "Complex64",
            }),
        }
    }
}
