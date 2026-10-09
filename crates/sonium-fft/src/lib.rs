//! High-performance batched complex / real FFT on top of [`sonium_compute`].
//!
//! # Layering
//!
//! - [`FftPlanner`] / [`FftPlan`] — native batch-aware API generic over a
//!   [`sonium_compute::Backend`].
//! - [`rustfft_compat`] — drop-in replacement for `rustfft::FftPlanner` /
//!   `rustfft::Fft`, CPU-only. Lets existing call sites migrate by changing
//!   only their `use` statement.
//!
//! # Algorithms
//!
//! - Mixed-radix Stockham auto-sort (no bit-reversal pass) for smooth
//!   lengths (factors of 2, 3, 4, 5). Lengths containing 7 or 8 (e.g. 7,
//!   14, 28, 56) are not smooth and go through Bluestein.
//! - Bluestein chirp-Z for non-smooth lengths (incl. primes outside the
//!   small-radix set).
//! - Real-input FFT via half-length complex FFT packing.
//!
//! # Transform contract
//!
//! Complex transforms use the unnormalised DFT convention
//! `X[k] = Σ_j x[j] exp(sign · 2πi jk/N)`: forward uses `sign = -1` and
//! inverse uses `sign = +1`. Thus an inverse after a forward transform is
//! `N · x`; callers apply `1/N` when they need a normalised round trip.
//! Samples are contiguous `Complex<T>` values. Native plans interpret a
//! buffer whose length is an integer multiple of `N` as independent batches;
//! an empty native buffer is a no-op. The compatibility adapter deliberately
//! restores rustfft's panic for empty, short, or non-multiple buffers.
//!
//! `process_with_scratch` requires at least `scratch_len()` elements and
//! never allocates during steady-state execution. Out-of-place compatibility
//! methods require equal-sized, non-overlapping input/output buffers; the
//! immutable-input variant leaves its input unchanged. No transform promises
//! in-place operation with overlapping mutable regions.
//!
//! # Performance philosophy
//!
//! For typical sizes (≲ 64K) FFT is bandwidth-bound. The implementation
//! minimises twiddle-table footprint, uses Stockham to avoid scratch
//! passes, and vectorises across the batch dimension when batch ≥ SIMD
//! width — see the design plan for the full discussion.

pub mod backend;
pub mod bluestein;
pub mod codelets;
pub mod convolution;
pub mod cpu;
pub mod dct;
pub mod direction;
pub mod factor;
pub mod nd;
pub mod plan;
pub mod planner;
pub mod real;
pub mod rustfft_compat;
pub mod scalar;
pub(crate) mod sincos;
pub mod six_step;
pub mod stft;
pub mod stockham;
pub mod twiddles;

#[cfg(target_arch = "aarch64")]
pub(crate) mod wide_neon;

#[cfg(target_arch = "x86_64")]
pub(crate) mod wide_avx2;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod wasm_simd;

pub use backend::{
    BackendDctPlanner, BackendDstPlanner, BackendFftPlanner, BackendRealFftPlanner, DeviceDctPlan,
    DeviceDstPlan, DeviceFftPlan, DeviceRealFftInversePlan, DeviceRealFftPlan,
    FftCancellationToken, FftExecution,
};
pub use convolution::{
    CircularConvolutionPlan, ConvolutionError, LinearConvolutionPlan, circular_convolution,
    linear_convolution, try_circular_convolution, try_linear_convolution,
};
pub use cpu::planner::{CpuFftDispatch, FftPlanWithAdjoint};
pub use dct::{
    DIRECT_THRESHOLD, DctPlan, DstPlan, dct2, dct2_adjoint, dct3, dct3_adjoint, dst2, dst2_adjoint,
    dst3, dst3_adjoint,
};
pub use direction::FftDirection;
pub use factor::PlanLengthError;
pub use nd::{
    Fft2dPlan, Fft3dPlan, NdFftError, apply_derivative, plan_adjoint, process_strided,
    try_process_strided,
};
pub use plan::{FftPlan, FftPlanError};
pub use planner::FftPlanner;
pub use real::{RealFftPlan, RealFftPlanner};
pub use scalar::FftScalar;
pub use six_step::SixStepFftPlan;
pub use stft::{StftConfig, StftError, StftPlan, WindowFunction};
