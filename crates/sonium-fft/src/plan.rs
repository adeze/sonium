//! [`FftPlan`] — the abstract plan trait. One implementation per backend.

use num_complex::Complex;
use num_traits::Float;
use std::fmt;

use crate::FftDirection;

/// Checked failure while admitting a native FFT buffer or scratch arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftPlanError {
    /// The plan length is invalid or the buffer is not a whole number of FFTs.
    InvalidBufferLength {
        /// The length of one transform.
        plan_length: usize,
        /// The supplied buffer length.
        buffer_length: usize,
    },
    /// The caller supplied less scratch than the plan requires.
    ScratchTooShort {
        /// Required scratch length.
        required: usize,
        /// Supplied scratch length.
        actual: usize,
    },
}

impl fmt::Display for FftPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBufferLength {
                plan_length,
                buffer_length,
            } => write!(
                formatter,
                "FFT buffer length {buffer_length} is not a multiple of plan length {plan_length}"
            ),
            Self::ScratchTooShort { required, actual } => {
                write!(
                    formatter,
                    "FFT scratch length {actual} is below required {required}"
                )
            }
        }
    }
}

impl std::error::Error for FftPlanError {}

fn validate_buffer_length<T: Float>(
    plan_length: usize,
    buffer: &[Complex<T>],
) -> Result<(), FftPlanError> {
    if plan_length == 0 || (!buffer.is_empty() && !buffer.len().is_multiple_of(plan_length)) {
        return Err(FftPlanError::InvalidBufferLength {
            plan_length,
            buffer_length: buffer.len(),
        });
    }
    Ok(())
}

/// A precomputed FFT plan. Stateless w.r.t. user data; all state is in
/// twiddle tables / scratch / device kernel handles owned by the plan.
///
/// `T` is the underlying real type (`f32` or `f64`); the buffer always
/// carries `Complex<T>` elements.
#[allow(clippy::len_without_is_empty)]
pub trait FftPlan<T: Float>: Send + Sync {
    /// Single-FFT length in complex elements.
    fn len(&self) -> usize;

    /// `Forward` or `Inverse`. The inverse plan is **unnormalised** — the
    /// caller scales by `1/N` if they want the round-trip identity. This
    /// matches `rustfft` and `numpy`'s convention for `fft` / `ifft`.
    fn direction(&self) -> FftDirection;

    /// In-place transform over `buffer`. `buffer.len()` must be an
    /// integer multiple of `self.len()` — the multiplier is the implicit
    /// batch count. Each `self.len()`-sized chunk is transformed
    /// independently.
    ///
    /// **Empty-buffer behaviour**: an empty `buffer` (batch count = 0) is
    /// a no-op and returns silently. This is a deliberate divergence
    /// from `rustfft::Fft::process`, which panics on empty input. The
    /// rustfft-compatibility shim in [`crate::rustfft_compat`] restores
    /// the panic behaviour at its boundary.
    fn process(&self, buffer: &mut [Complex<T>]);

    /// Checked no-scratch processing for a whole number of batches.
    ///
    /// Empty native buffers remain a successful no-op. Unlike `process`, this
    /// method returns a typed error for malformed batch lengths.
    fn try_process(&self, buffer: &mut [Complex<T>]) -> Result<(), FftPlanError> {
        validate_buffer_length(self.len(), buffer)?;
        self.process(buffer);
        Ok(())
    }

    /// Length of the auxiliary scratch buffer required by [`Self::process_with_scratch`].
    /// `0` for plans that need no scratch.
    #[must_use]
    fn scratch_len(&self) -> usize {
        0
    }

    /// Number of planner-defined cache sections used by this plan.
    ///
    /// Plans that do not expose a stage schedule return zero. A non-zero
    /// value is diagnostic metadata only; it never changes the transform's
    /// normalization or scratch contract.
    #[must_use]
    fn cache_section_count(&self) -> usize {
        0
    }

    /// Cache budget in bytes used when constructing the stage sections.
    /// Returns zero when the backend has no exposed section schedule.
    #[must_use]
    fn cache_section_budget_bytes(&self) -> usize {
        0
    }

    /// Same as [`Self::process`] but uses a caller-provided scratch buffer
    /// of length `>= self.scratch_len()`. Avoids per-call allocation in
    /// hot loops. The buffer and scratch regions are independent mutable
    /// regions as required by Rust's borrowing rules. The default
    /// implementation validates the declared scratch contract before
    /// delegating to [`Self::process`].
    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        assert!(
            scratch.len() >= self.scratch_len(),
            "FFT scratch is too short"
        );
        self.process(buffer);
    }

    /// Checked processing with caller-provided scratch.
    fn try_process_with_scratch(
        &self,
        buffer: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), FftPlanError> {
        validate_buffer_length(self.len(), buffer)?;
        let required = self.scratch_len();
        if scratch.len() < required {
            return Err(FftPlanError::ScratchTooShort {
                required,
                actual: scratch.len(),
            });
        }
        self.process_with_scratch(buffer, scratch);
        Ok(())
    }
}
