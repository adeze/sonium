//! Error types for compute backends.

use thiserror::Error;

/// Whether a backend instance may be used after an operation fails.
///
/// This is deliberately conservative.  A caller may always discard and
/// recreate a backend, but it must not continue submitting work after
/// [`BackendContextDisposition::MustRecreate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendContextDisposition {
    /// Validation, capability, or allocation failed before native execution.
    Reusable,
    /// Native execution failed, but a successful synchronization may restore
    /// a well-defined submission boundary.
    SynchronizeBeforeReuse,
    /// The device, queue, or synchronization boundary is no longer reliable.
    MustRecreate,
}

/// Whether caller-owned output remains usable after an operation fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendOutputDisposition {
    /// The failure is guaranteed to occur before output mutation.
    Preserved,
    /// Output may be partial or otherwise undefined and must not be read.
    /// Drop it, or completely overwrite it after satisfying the context
    /// recovery requirement reported by [`BackendError::context_disposition`].
    Unusable,
}

/// Errors raised while validating or dispatching a backend operation.
///
/// The original [`crate::Backend`] methods predate a fallible dispatch API and
/// therefore retain their small, allocation-free signatures. New code that
/// accepts user-controlled shapes or must distinguish an unavailable device
/// operation should use the corresponding `try_*` method. The infallible
/// methods are still total: their default implementations use a documented
/// host fallback or report the error through the backend logger instead of
/// panicking.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum BackendError {
    #[error("unsupported backend operation {operation} for {scalar} on {backend}")]
    Unsupported {
        backend: &'static str,
        operation: &'static str,
        scalar: &'static str,
    },

    /// The requested operation exists only as a host fallback.
    #[error("backend operation {operation} on {backend} would use a host fallback for {scalar}")]
    HostFallback {
        backend: &'static str,
        operation: &'static str,
        scalar: &'static str,
    },

    #[error("invalid backend argument for {operation}: {reason}")]
    InvalidArgument {
        operation: &'static str,
        reason: String,
    },

    #[error("backend operation {operation} failed: {reason}")]
    OperationFailed {
        operation: &'static str,
        reason: String,
    },

    #[error("backend device lost during {operation} on {backend}: {reason}")]
    DeviceLost {
        backend: &'static str,
        operation: &'static str,
        reason: String,
    },

    #[error("backend command buffer failed during {operation} on {backend}: {reason}")]
    CommandBuffer {
        backend: &'static str,
        operation: &'static str,
        reason: String,
    },

    #[error("backend launch failed during {operation} on {backend}: {reason}")]
    Launch {
        backend: &'static str,
        operation: &'static str,
        reason: String,
    },

    #[error("backend synchronization failed during {operation} on {backend}: {reason}")]
    Synchronization {
        backend: &'static str,
        operation: &'static str,
        reason: String,
    },

    #[error("backend transfer failed during {operation} on {backend}: {reason}")]
    Transfer {
        backend: &'static str,
        operation: &'static str,
        reason: String,
    },

    #[error("backend allocation failed: {reason}")]
    Allocation { reason: String },
}

impl BackendError {
    /// Construct the checked error returned when a caller requests
    /// cancellation after native work has been submitted.  Cancellation is
    /// represented as an `OperationFailed` for compatibility with the
    /// pre-existing public error enum; the dedicated operation name and
    /// [`Self::is_cancelled`] predicate keep it machine-detectable without
    /// forcing downstream solver error enums to add a new variant.
    #[must_use]
    pub fn cancelled(operation: &'static str) -> Self {
        Self::OperationFailed {
            operation,
            reason: "FFT operation cancelled after submission; output is unusable".into(),
        }
    }

    /// Return whether this is the checked post-submission cancellation error.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        matches!(
            self,
            Self::OperationFailed {
                operation: "fft_cancellation",
                ..
            }
        )
    }

    /// Return the conservative backend-recovery requirement for this failure.
    #[must_use]
    pub const fn context_disposition(&self) -> BackendContextDisposition {
        match self {
            Self::Unsupported { .. }
            | Self::HostFallback { .. }
            | Self::InvalidArgument { .. }
            | Self::Allocation { .. } => BackendContextDisposition::Reusable,
            Self::OperationFailed { .. } | Self::Launch { .. } | Self::Transfer { .. } => {
                BackendContextDisposition::SynchronizeBeforeReuse
            }
            Self::DeviceLost { .. } | Self::CommandBuffer { .. } | Self::Synchronization { .. } => {
                BackendContextDisposition::MustRecreate
            }
        }
    }

    /// Return whether an output passed to the failed operation is still valid.
    ///
    /// `OperationFailed` is treated as post-dispatch even when a particular
    /// implementation detects it earlier.  This prevents a caller from
    /// accidentally consuming partially written native output.
    #[must_use]
    pub const fn output_disposition(&self) -> BackendOutputDisposition {
        match self {
            Self::Unsupported { .. }
            | Self::HostFallback { .. }
            | Self::InvalidArgument { .. }
            | Self::Allocation { .. } => BackendOutputDisposition::Preserved,
            Self::OperationFailed { .. }
            | Self::DeviceLost { .. }
            | Self::CommandBuffer { .. }
            | Self::Launch { .. }
            | Self::Synchronization { .. }
            | Self::Transfer { .. } => BackendOutputDisposition::Unusable,
        }
    }
}

/// Result type used by the checked backend contract helpers.
pub type BackendResult<T> = Result<T, BackendError>;

/// Out-of-memory and quota-related allocation errors.
///
/// `QuotaExceeded` is raised by [`crate::MemoryPool`] *before* the backend
/// driver is called. `DriverOom` is raised when the backend itself rejects an
/// allocation that fit within the pool quota (host RAM exhaustion on CPU,
/// `cudaMalloc` failure on CUDA).
#[derive(Debug, Error)]
pub enum OomError {
    #[error(
        "device memory quota exceeded: requested {requested} B, in use {in_use} B, cap {cap} B"
    )]
    QuotaExceeded {
        requested: usize,
        in_use: usize,
        cap: usize,
    },

    #[error("backend driver out of memory: {0}")]
    DriverOom(String),

    #[error("invalid allocation request: {0}")]
    Invalid(String),
}

impl From<OomError> for BackendError {
    fn from(error: OomError) -> Self {
        Self::Allocation {
            reason: error.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendContextDisposition, BackendError, BackendOutputDisposition};

    #[test]
    fn failure_dispositions_distinguish_preflight_execution_and_context_loss() {
        let invalid = BackendError::InvalidArgument {
            operation: "test",
            reason: "bad shape".into(),
        };
        assert_eq!(
            invalid.context_disposition(),
            BackendContextDisposition::Reusable
        );
        assert_eq!(
            invalid.output_disposition(),
            BackendOutputDisposition::Preserved
        );

        let launch = BackendError::Launch {
            backend: "test",
            operation: "kernel",
            reason: "injected failure".into(),
        };
        assert_eq!(
            launch.context_disposition(),
            BackendContextDisposition::SynchronizeBeforeReuse
        );
        assert_eq!(
            launch.output_disposition(),
            BackendOutputDisposition::Unusable
        );

        let lost = BackendError::DeviceLost {
            backend: "test",
            operation: "completion",
            reason: "injected device loss".into(),
        };
        assert_eq!(
            lost.context_disposition(),
            BackendContextDisposition::MustRecreate
        );
        assert_eq!(
            lost.output_disposition(),
            BackendOutputDisposition::Unusable
        );
    }
}
