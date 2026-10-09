//! Apple Metal backend with native f32 MSL kernels.
//!
//! Built on `objc2-metal` for proper `Retained<T>` ARC ownership and a
//! direct path to `objc2-metal-performance-shaders` (MPSMatrixMultiplication,
//! MPSMatrixFFT) — the older `metal-rs` 0.33 only exposed MPS for ray
//! tracing. We own the Metal device, command queue, and pre-compiled
//! `MTLComputePipelineState`s for the BLAS-1/2 + SpMV primitives that
//! `sonium-solvers` consumes.
//!
//! # Measured perf (M4 Pro, f32 GEMM)
//!
//! Wall-clock per `Backend::gemm<f32>` call with `backend.synchronize()`
//! after each call (so MPS dispatch + GPU completion fully attributed):
//!
//! | Shape           | SME2 CPU | Accelerate | MPS (this backend) |
//! |-----------------|----------|------------|--------------------|
//! | 64³             | 3.8 µs   | 6.0 µs     | 141 µs (overhead-bound) |
//! | 256³            | 96 µs    | 347 µs     | 140 µs             |
//! | 1024³           | 3.44 ms  | 20.9 ms    | **507 µs** (~7× SME2) |
//! | 2048·128·2048   | 734 µs   | 10.9 ms    | 573 µs             |
//!
//! Below ~256³ MPS dispatch + sync overhead dominates (~130 µs per call);
//! `Backend::gemm<f32>` therefore routes shapes at or below that work size
//! to the CPU `cpu_blas3::gemm_f32` SME2 path over shared memory, while
//! keeping MPS for larger f32 matrices.
//!
//! # Precision
//!
//! - **f32**: native MSL kernels for `axpy`, `dot`, `nrm2`, `scale`,
//!   `spmv_csr`. Buffers live in `StorageModeShared` (unified-memory
//!   on Apple Silicon) so `copy_h2d` / `copy_d2h` are direct memcpys
//!   — no staging.
//! - **f64**: Apple Silicon Metal compiler **rejects `double`**, so
//!   f64 ops fall back to host-side compute over the same shared
//!   buffer. Memcpy + CPU compute + memcpy back. Correct but slow.
//!   The mixed-precision GMRES-IR path in `sonium-solvers`
//!   (`iterative::gmres_ir_backend`) is the way to use Metal for
//!   sonium's f64 physics: outer loop in f64 (host-staged), inner
//!   Krylov in f32 (native MSL).
//!
//! # Shared-storage memory model
//!
//! Apple Silicon GPUs use unified physical memory. With
//! `MTLResourceOptions::StorageModeShared` the Metal buffer is the
//! same physical pages as a CPU pointer obtained via
//! `Buffer::contents()`. CPU↔GPU coherence is maintained by command-buffer
//! batching: dispatches accumulate in a pending buffer and
//! `wait_until_completed` runs only when the host must read GPU output
//! or on explicit `synchronize` (see `flush_pending`).

#[cfg(not(target_os = "macos"))]
compile_error!(
    "the `metal` feature requires macOS — no Metal driver on other platforms. \
     Build without `--features metal` or guard the binary with cfg(target_os = \"macos\")."
);

use std::any::TypeId;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, atomic::Ordering};

use num_complex::Complex32;
use num_traits::Zero;

use objc2::AllocAnyThread;
use objc2::rc::{Allocated, Retained};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
    MPSMatrixVectorMultiplication, MPSVector, MPSVectorDescriptor,
};

use crate::backend::{
    Backend, BackendAllocationGuard, BackendAllocationLedger, BackendCapabilities,
    BackendCompletion, BackendMemoryStatsSnapshot, BackendTransferStats, BemCapabilities,
    BufferView, CapabilityState, DeviceBufferLike, PrecisionCastCapabilities,
    ReductionCapabilities, ScalarCapabilities, TransferStatsSnapshot, Transpose, ViewAliasing,
    ViewDescriptor, catch_backend_panic, checked_fft_view_descriptor, checked_strided_span,
    require_finite_scalar, try_collect_backend, try_host_staging_capacity, try_host_staging_vec,
    validate_bem_regular_pair_inputs, validate_bem_regular_pair_outputs,
};
use crate::bem::{BemRegularPairIn, BemRegularPairOut};
use crate::cpu::{blas2 as cpu_blas2, blas3 as cpu_blas3};
use crate::error::{BackendError, BackendResult, OomError};
use crate::scalar::{BufferElement, DeviceScalar, LowerPrecision};
use crate::segmented::{segmented_sum, validate_segmented_offsets};
use crate::tensor_contract::batched_local_contract;

// ---------------------------------------------------------------------------
// Type aliases — keep call sites readable. `Retained<ProtocolObject<dyn ...>>`
// is the canonical objc2 ownership handle for an Objective-C protocol; the
// alias drops the noise.
// ---------------------------------------------------------------------------

type MetalDevice = Retained<ProtocolObject<dyn MTLDevice>>;
type MetalQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type MetalCmdBuf = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type MetalPso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type MetalRawHandle = Retained<ProtocolObject<dyn MTLBuffer>>;
type MpsMatMul = Retained<MPSMatrixMultiplication>;

type MetalAllocationStats = BackendAllocationLedger;
type MetalAllocationGuard = BackendAllocationGuard;

/// Allocate a compatibility result without allowing a legacy infallible
/// wrapper to panic on a caller-controlled result length. Checked callers use
/// the fallible `Backend` methods and retain the typed allocation error.
fn zeroed_compatibility_result<T: DeviceScalar>(len: usize) -> Vec<T> {
    let mut values = Vec::new();
    if values.try_reserve_exact(len).is_err() {
        return Vec::new();
    }
    values.resize(len, T::zero());
    values
}

/// Reference-counted Metal buffer handle with backend allocation accounting.
///
/// Metal command encoders retain buffers asynchronously. The accounting token
/// is therefore shared by every Rust clone of the handle and is released only
/// after the last clone— including deferred scratch references— is dropped.
#[derive(Clone)]
struct MetalRawBuf {
    inner: MetalRawHandle,
    _allocation: Arc<MetalAllocationGuard>,
}

impl MetalRawBuf {
    fn from_guard(inner: MetalRawHandle, allocation: MetalAllocationGuard) -> Self {
        Self {
            inner,
            _allocation: Arc::new(allocation),
        }
    }
}

impl Deref for MetalRawBuf {
    type Target = ProtocolObject<dyn MTLBuffer>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

const F32_GEMM_CPU_SHARED_MAX_FMAS: usize = 256 * 256 * 256;
const C32_GEMM_CPU_SHARED_MAX_FMAS: usize = 256 * 256 * 256;

/// Cache key for `MPSMatrixMultiplication` instances. Apple's kernel binds
/// alpha/beta at construction time (no settable properties), so a complete
/// cache key includes them. For a frequency-sweep workload where alpha/beta
/// are constant across iterations the cache hits 100 %; otherwise we pay
/// one MPS kernel construction per distinct (shape, alpha, beta) combo.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
struct MpsMmKey {
    trans_left: bool,
    trans_right: bool,
    m: u32,
    n: u32,
    k: u32,
    alpha_bits: u64,
    beta_bits: u64,
}

// objc2's ProtocolObject<dyn MTL...> is `!Send + !Sync` by default to be
// conservative — the auto traits don't know that Apple's Metal protocols
// are documented as thread-safe (MTLDevice, MTLCommandQueue, MTLBuffer,
// MTLCommandBuffer, MTLComputePipelineState all permit cross-thread use
// per the Metal Programming Guide). We assert this safely with a wrapper
// newtype that carries `Send + Sync`. Only the bits that cross the Mutex
// boundary in `MetalBackend` need it — buffers held under `MetalBuffer<T>`
// inherit thread-safety via the wrapping struct's own unsafe impl.
#[repr(transparent)]
struct ThreadSafe<T>(T);
// SAFETY: Apple's Metal API documents MTLDevice / MTLCommandQueue /
// MTLBuffer / MTLCommandBuffer / MTLComputePipelineState as safe to
// reference from multiple threads simultaneously. The Retained<...>
// handle is a refcounted pointer; aliasing it across threads is sound
// as long as we don't violate the protocol's own threading rules, which
// we don't (the only mutation point is the pending command buffer,
// already guarded by Mutex).
unsafe impl<T> Send for ThreadSafe<T> {}
unsafe impl<T> Sync for ThreadSafe<T> {}

impl<T> std::ops::Deref for ThreadSafe<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}
impl<T> std::ops::DerefMut for ThreadSafe<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}
impl<T: Clone> Clone for ThreadSafe<T> {
    fn clone(&self) -> Self {
        ThreadSafe(self.0.clone())
    }
}

struct MetalCommandCompletion {
    command_buffer: Option<ThreadSafe<MetalCmdBuf>>,
    deferred_scratch: Vec<(u64, ThreadSafe<MetalRawBuf>)>,
    pending_error: Arc<Mutex<Option<BackendError>>>,
}

fn record_first_pending_error(slot: &Mutex<Option<BackendError>>, error: BackendError) {
    if let Ok(mut pending_error) = slot.lock()
        && pending_error.is_none()
    {
        *pending_error = Some(error);
    }
}

impl BackendCompletion for MetalCommandCompletion {
    fn wait(&mut self) -> BackendResult<()> {
        let Some(command_buffer) = self.command_buffer.take() else {
            return Ok(());
        };

        command_buffer.0.commit();
        command_buffer.0.waitUntilCompleted();
        let result = if command_buffer.0.status() == MTLCommandBufferStatus::Error {
            let reason = command_buffer
                .0
                .error()
                .map(|error| format!("{error:?}"))
                .unwrap_or_else(|| "Metal reported command-buffer error".into());
            let error = BackendError::CommandBuffer {
                backend: "metal",
                operation: "completion_command_buffer",
                reason,
            };
            // A completion can fail after `try_submit_completion` has
            // returned. Preserve that failure on the backend so subsequent
            // checked operations cannot submit work to a device/queue that
            // has already reported device loss or command-buffer failure.
            record_first_pending_error(&self.pending_error, error.clone());
            Err(error)
        } else {
            self.pending_error
                .lock()
                .map_err(|_| BackendError::OperationFailed {
                    operation: "metal_error_state",
                    reason: "pending error mutex poisoned".into(),
                })?
                .as_ref()
                .cloned()
                .map_or(Ok(()), Err)
        };

        // The buffers are dropped only after waitUntilCompleted returns. This
        // is the resource-lifetime boundary required by asynchronous Metal
        // command submission. Recycling can happen on a later submission;
        // correctness does not depend on returning them to the pool here.
        self.deferred_scratch.clear();
        result
    }
}

// ---------------------------------------------------------------------------
// Encoder helpers — fold the most-repeated boilerplate (`set_bytes` with the
// triple cast, NSString construction for kernel names) behind small helpers.
// objc2's encoder protocol methods are `unsafe` because they don't track
// buffer lifetimes; we discharge that safety obligation at the call site
// (every binding outlives the encoder's `endEncoding`).
// ---------------------------------------------------------------------------

/// Bind a scalar `value` at `index` on a compute encoder via `setBytes`.
/// Copies `size_of::<T>()` bytes into the encoder's constant scratch.
#[inline]
fn set_scalar<T: Copy>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: u64,
    value: &T,
) {
    // SAFETY: encoder.setBytes copies `length` bytes from the given pointer
    // into its own constant store before returning, so the borrow of `value`
    // ends with this call.
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::from(value).cast(),
            std::mem::size_of::<T>(),
            index as usize,
        );
    }
}

/// Bind a buffer at `index` on a compute encoder. Offset defaults to 0.
#[inline]
fn set_buffer(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: u64,
    buffer: &ProtocolObject<dyn MTLBuffer>,
) {
    // SAFETY: the bound buffer outlives the encoder (we always end_encoding
    // before dropping any buffer reference).
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(buffer), 0, index as usize);
    }
}

/// Convert a `Retained<NSError>` (from any `*_error` method) to our OomError.
fn ns_error(prefix: &str, e: Retained<objc2_foundation::NSError>) -> OomError {
    OomError::Invalid(format!("{}: {}", prefix, e.localizedDescription()))
}

// ---------------------------------------------------------------------------
// MSL kernels
// ---------------------------------------------------------------------------

/// All f32 MSL kernels in a single library. Compiled once at backend
/// init; each kernel becomes a cached `ComputePipelineState`.
const MSL_LIBRARY_SOURCE: &str = include_str!("kernels.metal");

/// Kernel entry-point names (must match the function names in the MSL).
mod kernel_names {
    pub const AXPY_F32: &str = "sonium_axpy_f32";
    pub const AXPY_MANY_F32: &str = "sonium_axpy_many_f32";
    pub const SCALE_F32: &str = "sonium_scale_f32";
    pub const DOT_F32: &str = "sonium_dot_f32";
    pub const NRM2_SQ_F32: &str = "sonium_nrm2_sq_f32";
    pub const SPMV_CSR_F32: &str = "sonium_spmv_csr_f32";
    pub const MULTI_DOT_F32: &str = "sonium_multi_dot_f32";
    pub const AXPY_C32: &str = "sonium_axpy_c32";
    pub const AXPY_MANY_C32: &str = "sonium_axpy_many_c32";
    pub const SCALE_C32: &str = "sonium_scale_c32";
    pub const DOT_C32: &str = "sonium_dot_c32";
    pub const NRM2_SQ_C32: &str = "sonium_nrm2_sq_c32";
    pub const SPMV_CSR_C32: &str = "sonium_spmv_csr_c32";
    pub const MULTI_DOT_C32: &str = "sonium_multi_dot_c32";
    pub const BEM_REGULAR_PAIR_C32: &str = "sonium_bem_regular_pair_c32";
    pub const GATHER4_F32: &str = "sonium_gather4_f32";
    pub const GATHER4_C32: &str = "sonium_gather4_c32";
    pub const REDUCE_SUM_F32: &str = "sonium_reduce_sum_f32";
    pub const REDUCE_SUM_C32: &str = "sonium_reduce_sum_c32";
    pub const REDUCE_MULTI_SUM_F32: &str = "sonium_reduce_multi_sum_f32";
    pub const REDUCE_MULTI_SUM_C32: &str = "sonium_reduce_multi_sum_c32";
    pub const SEGMENTED_SUM_F32: &str = "sonium_segmented_sum_f32";
    pub const SEGMENTED_SUM_C32: &str = "sonium_segmented_sum_c32";
    pub const BATCHED_LOCAL_CONTRACT_F32: &str = "sonium_batched_local_contract_f32";
    pub const BATCHED_LOCAL_CONTRACT_C32: &str = "sonium_batched_local_contract_c32";
    pub const GEMV_F32_NOTRANS: &str = "sonium_gemv_f32_notrans";
    pub const GEMV_C32_NOTRANS: &str = "sonium_gemv_c32_notrans";
    pub const GEMV_F32_TRANS: &str = "sonium_gemv_f32_trans";
    pub const GEMV_C32_TRANS: &str = "sonium_gemv_c32_trans";
    pub const SPMV_CSR_SHORT_F32: &str = "sonium_spmv_csr_short_f32";
    pub const SPMV_CSR_SIMD_F32: &str = "sonium_spmv_csr_simd_f32";
    pub const SPMV_CSR_CTA_F32: &str = "sonium_spmv_csr_cta_f32";
    pub const SPMV_CSR_SHORT_C32: &str = "sonium_spmv_csr_short_c32";
    pub const SPMV_CSR_SIMD_C32: &str = "sonium_spmv_csr_simd_c32";
    pub const SPMV_CSR_CTA_C32: &str = "sonium_spmv_csr_cta_c32";
    pub const SPMM_CSR_F32: &str = "sonium_spmm_csr_f32";
    pub const SPMM_CSR_C32: &str = "sonium_spmm_csr_c32";
    pub const FMM_PHASE_ACCUM_C32: &str = "sonium_fmm_phase_accum_c32";
    pub const FMM_COMPACT_GATHER_C32: &str = "sonium_fmm_compact_gather_c32";
    pub const FMM_COMPACT_SCATTER_ACCUM_C32: &str = "sonium_fmm_compact_scatter_accum_c32";
    pub const FMM_INDEXED_DENSE_MATVEC_ACCUM_C32: &str =
        "sonium_fmm_indexed_dense_matvec_accum_c32";
    pub const FFT_STOCKHAM_RADIX2_STAGE_C32: &str = "sonium_fft_stockham_radix2_stage_c32";
    pub const FFT_STOCKHAM_RADIX4_STAGE_C32: &str = "sonium_fft_stockham_radix4_stage_c32";
}

/// Threadgroup width for 1-D kernels (axpy, scale, spmv_csr).
const THREADGROUP_SIZE: u64 = 256;

/// Threadgroup width for reductions (dot, nrm2). Must be a power of 2
/// matching the reduction tree in the MSL source.
const REDUCTION_THREADGROUP_SIZE: u64 = 256;

// ---------------------------------------------------------------------------
// Buffer wrapper
// ---------------------------------------------------------------------------

/// CSR descriptor borrowing source buffers. Build cost is `O(m)` because
/// rows are bucketed by nnz at prebake time so the adaptive SpMV (short
/// thread-per-row, simd-per-row, threadgroup-per-row) can dispatch each
/// kernel over only its own row list. The buckets are owned by the
/// prebaked handle so they stay alive across many `spmv_prebaked` calls.
/// f64 / Complex64 prebakings skip the bucketing and use the host SpMV
/// path (Apple Metal compiler rejects `double`).
pub struct MetalCsrPrebaked<'a, T: DeviceScalar> {
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) row_ptr: &'a MetalBuffer<i32>,
    pub(crate) col_idx: &'a MetalBuffer<i32>,
    pub(crate) values: &'a MetalBuffer<T>,
    /// Bucketed row indices for the adaptive SpMV. `None` for f64/c64
    /// (which use host SpMV) or when m == 0.
    pub(crate) buckets: Option<CsrBuckets>,
    /// A compatibility `prebake_csr` call cannot return a `Result`, so
    /// allocation and validation failures are retained here for the checked
    /// `try_prebake_csr` boundary. A handle carrying this field is inert in
    /// the legacy infallible operations and can never dispatch native work.
    pub(crate) initialization_error: Option<BackendError>,
}

impl<'a, T: DeviceScalar> MetalCsrPrebaked<'a, T> {
    fn failed(
        m: usize,
        n: usize,
        row_ptr: &'a MetalBuffer<i32>,
        col_idx: &'a MetalBuffer<i32>,
        values: &'a MetalBuffer<T>,
        error: BackendError,
    ) -> Self {
        Self {
            m,
            n,
            row_ptr,
            col_idx,
            values,
            buckets: None,
            initialization_error: Some(error),
        }
    }
}

/// Owned per-bucket row index buffers for adaptive CSR SpMV. Built once
/// in `prebake_csr` by scanning `row_ptr`.
pub(crate) struct CsrBuckets {
    pub(crate) short: MetalBuffer<i32>, // rows with nnz < 8
    pub(crate) short_count: u32,
    pub(crate) simd: MetalBuffer<i32>, // rows with 8 <= nnz < 64
    pub(crate) simd_count: u32,
    pub(crate) cta: MetalBuffer<i32>, // rows with nnz >= 64
    pub(crate) cta_count: u32,
}

// SAFETY: `Retained<ProtocolObject<dyn MTLBuffer>>` is `Send + Sync` already
// in objc2-metal (the underlying protocol traits don't carry !Send/!Sync
// bounds). These unsafe impls cover only the `PhantomData<T>` and the
// reference fields, which are Send+Sync for our T types — but the compiler
// can't see that through `PhantomData<T>` without help.
unsafe impl<T: DeviceScalar> Send for MetalCsrPrebaked<'_, T> {}
unsafe impl<T: DeviceScalar> Sync for MetalCsrPrebaked<'_, T> {}

/// Shared-storage Metal buffer typed by `T`. The underlying `MTLBuffer`
/// lives in unified memory on Apple Silicon, so `contents()` returns a
/// CPU-accessible pointer without staging.
pub struct MetalBuffer<T: BufferElement> {
    raw: ThreadSafe<MetalRawBuf>,
    len: usize,
    _phantom: PhantomData<T>,
}

// SAFETY: see MetalCsrPrebaked above.
unsafe impl<T: BufferElement> Send for MetalBuffer<T> {}
unsafe impl<T: BufferElement> Sync for MetalBuffer<T> {}

impl<T: BufferElement> MetalBuffer<T> {
    /// View the buffer as a CPU-readable slice. Safe because shared-storage
    /// Metal buffers are always CPU-mappable on Apple Silicon.
    fn host_slice(&self) -> &[T] {
        // SAFETY: shared-storage MTLBuffer's `contents()` always returns a
        // valid CPU pointer to `length` bytes of unified memory; `self.len`
        // is the element count we allocated for.
        let ptr = self.raw.contents().as_ptr() as *const T;
        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }

    fn host_slice_mut(&mut self) -> &mut [T] {
        let ptr = self.raw.contents().as_ptr() as *mut T;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.len) }
    }
}

impl<T: BufferElement> DeviceBufferLike<T> for MetalBuffer<T> {
    fn len(&self) -> usize {
        self.len
    }
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

/// Apple Silicon Metal backend with native f32 MSL kernels.
///
/// **Dispatch batching**: kernel calls accumulate into a single
/// pending command buffer; `wait_until_completed` is only called when
/// the host needs to read GPU output (`dot`, `nrm2`, `copy_d2h`, ...)
/// or on explicit `synchronize`. This collapses the per-kernel ~100 µs
/// sync overhead that otherwise dominates Krylov inner loops with many
/// small ops. See `flush_pending()` for the flush semantics.
pub struct MetalBackend {
    device: ThreadSafe<MetalDevice>,
    queue: ThreadSafe<MetalQueue>,
    allocation_stats: Arc<MetalAllocationStats>,
    /// Temporary host fallback allocations tracked separately from Metal
    /// shared buffers so transient peaks and refunds remain observable.
    staging_stats: Arc<MetalAllocationStats>,
    transfer_stats: Arc<BackendTransferStats>,
    /// Built once at construction and read-only thereafter, so no lock.
    /// Lookup hands out a reference; the underlying pipeline state object
    /// is ObjC-refcounted internally and our `Retained<...>` handle holds
    /// one strong reference for the lifetime of the backend.
    pipelines: HashMap<&'static str, ThreadSafe<MetalPso>>,
    /// Command buffer accumulating queued kernels. `None` = no pending
    /// work. Becomes `Some(cmd_buf)` on first dispatch after a flush;
    /// `flush_pending` commits and waits, returning to `None`.
    pending: Mutex<Option<ThreadSafe<MetalCmdBuf>>>,
    /// Completion error captured by legacy void methods. Once a command
    /// buffer has failed, the context is poisoned for checked operations:
    /// callers must recreate the backend rather than reuse potentially
    /// invalid device state.
    pending_error: Arc<Mutex<Option<BackendError>>>,
    /// Recycled scratch buffers keyed by byte length. Reused for
    /// reduction partials and multi-dot concat staging across many
    /// Krylov iterations; avoids hammering `newBuffer` on every
    /// `dot` / `nrm2` / `multi_dot` call. Bounded per-bucket — see
    /// `take_scratch` / `return_scratch`.
    scratch: Mutex<HashMap<u64, Vec<ThreadSafe<MetalRawBuf>>>>,
    /// Scratch buffers referenced by the currently pending command buffer.
    /// They are recycled only after `flush_pending` waits for completion.
    pending_scratch: Mutex<Vec<(u64, ThreadSafe<MetalRawBuf>)>>,
    /// `MPSMatrixMultiplication` instance cache keyed by shape + alpha + beta.
    /// MPS binds alpha/beta at construction so we cache the whole tuple; a
    /// frequency-sweep / Krylov inner-loop that calls `gemm<f32>` with stable
    /// params hits this cache on every call after the first.
    mps_kernels: Mutex<HashMap<MpsMmKey, ThreadSafe<MpsMatMul>>>,
}

impl MetalBackend {
    /// Initialise Metal and pre-compile the f32 kernel library.
    pub fn new() -> Result<Self, OomError> {
        // Returns None when no GPU is available (sandboxed processes,
        // headless CI, etc.).
        let device: MetalDevice = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| OomError::Invalid("no Metal device".into()))?;
        let queue: MetalQueue = device
            .newCommandQueue()
            .ok_or_else(|| OomError::Invalid("Metal: newCommandQueue returned nil".into()))?;

        // Compile the kernel library eagerly — keeps per-call dispatch fast.
        let opts = MTLCompileOptions::new();
        let source = NSString::from_str(MSL_LIBRARY_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, Some(&opts))
            .map_err(|e| ns_error("MSL compile failed", e))?;

        let mut pipelines: HashMap<&'static str, ThreadSafe<MetalPso>> = HashMap::new();
        for &name in &[
            kernel_names::AXPY_F32,
            kernel_names::AXPY_MANY_F32,
            kernel_names::SCALE_F32,
            kernel_names::DOT_F32,
            kernel_names::NRM2_SQ_F32,
            kernel_names::SPMV_CSR_F32,
            kernel_names::MULTI_DOT_F32,
            kernel_names::AXPY_C32,
            kernel_names::AXPY_MANY_C32,
            kernel_names::SCALE_C32,
            kernel_names::DOT_C32,
            kernel_names::NRM2_SQ_C32,
            kernel_names::SPMV_CSR_C32,
            kernel_names::MULTI_DOT_C32,
            kernel_names::BEM_REGULAR_PAIR_C32,
            kernel_names::GATHER4_F32,
            kernel_names::GATHER4_C32,
            kernel_names::REDUCE_SUM_F32,
            kernel_names::REDUCE_SUM_C32,
            kernel_names::REDUCE_MULTI_SUM_F32,
            kernel_names::REDUCE_MULTI_SUM_C32,
            kernel_names::SEGMENTED_SUM_F32,
            kernel_names::SEGMENTED_SUM_C32,
            kernel_names::BATCHED_LOCAL_CONTRACT_F32,
            kernel_names::BATCHED_LOCAL_CONTRACT_C32,
            kernel_names::GEMV_F32_NOTRANS,
            kernel_names::GEMV_C32_NOTRANS,
            kernel_names::GEMV_F32_TRANS,
            kernel_names::GEMV_C32_TRANS,
            kernel_names::SPMV_CSR_SHORT_F32,
            kernel_names::SPMV_CSR_SIMD_F32,
            kernel_names::SPMV_CSR_CTA_F32,
            kernel_names::SPMV_CSR_SHORT_C32,
            kernel_names::SPMV_CSR_SIMD_C32,
            kernel_names::SPMV_CSR_CTA_C32,
            kernel_names::SPMM_CSR_F32,
            kernel_names::SPMM_CSR_C32,
            kernel_names::FMM_PHASE_ACCUM_C32,
            kernel_names::FMM_COMPACT_GATHER_C32,
            kernel_names::FMM_COMPACT_SCATTER_ACCUM_C32,
            kernel_names::FMM_INDEXED_DENSE_MATVEC_ACCUM_C32,
            kernel_names::FFT_STOCKHAM_RADIX2_STAGE_C32,
            kernel_names::FFT_STOCKHAM_RADIX4_STAGE_C32,
        ] {
            let ns_name = NSString::from_str(name);
            let function = library
                .newFunctionWithName(&ns_name)
                .ok_or_else(|| OomError::Invalid(format!("MSL kernel '{name}' missing")))?;
            let pso = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| ns_error(&format!("MSL pipeline({name})"), e))?;
            pipelines.insert(name, ThreadSafe(pso));
        }

        Ok(Self {
            device: ThreadSafe(device),
            queue: ThreadSafe(queue),
            allocation_stats: Arc::new(MetalAllocationStats::default()),
            staging_stats: Arc::new(MetalAllocationStats::default()),
            transfer_stats: Arc::new(BackendTransferStats::new()),
            pipelines,
            pending: Mutex::new(None),
            pending_error: Arc::new(Mutex::new(None)),
            scratch: Mutex::new(HashMap::new()),
            pending_scratch: Mutex::new(Vec::new()),
            mps_kernels: Mutex::new(HashMap::new()),
        })
    }

    /// Borrow a scratch buffer of at least `bytes` from the recycle pool,
    /// or allocate one if the pool has nothing of that size. Always
    /// `StorageModeShared` — matches every existing scratch use site
    /// (reduction partials, multi-dot concat).
    fn take_scratch(&self, bytes: u64) -> Option<MetalRawBuf> {
        match self.take_scratch_checked(bytes) {
            Ok(buffer) => Some(buffer),
            Err(error) => {
                self.record_pending_error(error);
                None
            }
        }
    }

    fn take_scratch_checked(&self, bytes: u64) -> BackendResult<MetalRawBuf> {
        let bytes_usize = usize::try_from(bytes).map_err(|_| BackendError::InvalidArgument {
            operation: "metal_scratch_pool",
            reason: "scratch size exceeds host address range".into(),
        })?;
        let mut pool = self
            .scratch
            .lock()
            .map_err(|_| BackendError::OperationFailed {
                operation: "metal_scratch_pool",
                reason: "scratch mutex poisoned while taking a buffer".into(),
            })?;
        if let Some(bucket) = pool.get_mut(&bytes)
            && let Some(buf) = bucket.pop()
        {
            return Ok(buf.0);
        }
        drop(pool);
        let allocation = self
            .allocation_stats
            .try_reserve(bytes_usize)
            .map_err(BackendError::from)?;
        let raw = self
            .device
            .newBufferWithLength_options(bytes_usize, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| BackendError::Allocation {
                reason: format!("Metal scratch buffer allocation failed for {bytes} bytes"),
            })?;
        Ok(MetalRawBuf::from_guard(raw, allocation))
    }

    /// Bytes currently owned by Metal buffers created through this backend.
    pub fn live_allocation_bytes(&self) -> usize {
        self.allocation_stats.live_bytes.load(Ordering::Relaxed)
    }

    /// Bytes currently retained by temporary host staging vectors.
    pub fn live_staging_bytes(&self) -> usize {
        self.staging_stats.live_bytes.load(Ordering::Relaxed)
    }

    /// High-water mark for temporary host staging vectors.
    pub fn staging_high_water_bytes(&self) -> usize {
        self.staging_stats.high_water_bytes.load(Ordering::Relaxed)
    }

    /// High-water mark for Metal buffer bytes owned by this backend.
    pub fn allocation_high_water_bytes(&self) -> usize {
        self.allocation_stats
            .high_water_bytes
            .load(Ordering::Relaxed)
    }

    /// Return a scratch buffer to the recycle pool. Bounded so a runaway
    /// solver doesn't accumulate stale per-iteration allocations.
    fn return_scratch(&self, bytes: u64, buf: MetalRawBuf) {
        if let Err(error) = self.return_scratch_checked(bytes, buf) {
            self.record_pending_error(error);
        }
    }

    fn return_scratch_checked(&self, bytes: u64, buf: MetalRawBuf) -> BackendResult<()> {
        const MAX_PER_BUCKET: usize = 8;
        let mut pool = self
            .scratch
            .lock()
            .map_err(|_| BackendError::OperationFailed {
                operation: "metal_scratch_pool",
                reason: "scratch mutex poisoned while returning a completed buffer".into(),
            })?;
        let bucket = pool.entry(bytes).or_default();
        if bucket.len() < MAX_PER_BUCKET {
            bucket.push(ThreadSafe(buf));
        }
        Ok(())
    }

    fn defer_scratch_until_flush(&self, bytes: u64, buf: MetalRawBuf) {
        match self.pending_scratch.lock() {
            Ok(mut pending) => pending.push((bytes, ThreadSafe(buf))),
            Err(_) => self.record_pending_error(BackendError::OperationFailed {
                operation: "metal_pending_scratch",
                reason: "pending scratch mutex poisoned while deferring buffer".into(),
            }),
        }
    }

    fn try_pending_or_new(
        &self,
        pending: &mut Option<ThreadSafe<MetalCmdBuf>>,
    ) -> BackendResult<MetalCmdBuf> {
        if let Some(cmd_buf) = pending.as_ref() {
            return Ok(cmd_buf.0.clone());
        }
        let cmd_buf = self
            .queue
            .commandBuffer()
            .ok_or_else(|| BackendError::CommandBuffer {
                backend: "metal",
                operation: "command_buffer",
                reason: "Metal commandBuffer returned nil".into(),
            })?;
        *pending = Some(ThreadSafe(cmd_buf.clone()));
        Ok(cmd_buf)
    }

    /// Get or create a pending command buffer while retaining failures in the
    /// backend's sticky error state. The returned command buffer is retained,
    /// so callers do not need to hold the pending-queue mutex while encoding.
    fn try_pending_command_buffer(&self) -> BackendResult<MetalCmdBuf> {
        self.check_pending_error()?;
        let mut pending = self
            .pending
            .lock()
            .map_err(|_| BackendError::OperationFailed {
                operation: "metal_pending_queue",
                reason: "pending mutex poisoned".into(),
            })?;
        self.try_pending_or_new(&mut pending)
    }

    fn pending_command_buffer(&self) -> Option<MetalCmdBuf> {
        match self.try_pending_command_buffer() {
            Ok(cmd_buf) => Some(cmd_buf),
            Err(error) => {
                self.record_pending_error(error);
                None
            }
        }
    }

    fn try_compute_encoder(
        &self,
        cmd_buf: &MetalCmdBuf,
    ) -> BackendResult<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
        cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| BackendError::CommandBuffer {
                backend: "metal",
                operation: "compute_encoder",
                reason: "Metal computeCommandEncoder returned nil".into(),
            })
    }

    fn compute_encoder(
        &self,
        cmd_buf: &MetalCmdBuf,
    ) -> Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
        match self.try_compute_encoder(cmd_buf) {
            Ok(encoder) => Some(encoder),
            Err(error) => {
                self.record_pending_error(error);
                None
            }
        }
    }

    fn try_blit_encoder(
        &self,
        cmd_buf: &MetalCmdBuf,
    ) -> BackendResult<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>> {
        cmd_buf
            .blitCommandEncoder()
            .ok_or_else(|| BackendError::CommandBuffer {
                backend: "metal",
                operation: "blit_encoder",
                reason: "Metal blitCommandEncoder returned nil".into(),
            })
    }

    fn blit_encoder(
        &self,
        cmd_buf: &MetalCmdBuf,
    ) -> Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>> {
        match self.try_blit_encoder(cmd_buf) {
            Ok(encoder) => Some(encoder),
            Err(error) => {
                self.record_pending_error(error);
                None
            }
        }
    }

    /// Commit the pending command buffer (if any) and wait for GPU
    /// completion. Called automatically before any host read of a
    /// device buffer; callers can also invoke explicitly.
    fn record_pending_error(&self, error: BackendError) {
        record_first_pending_error(&self.pending_error, error);
    }

    fn record_invalid_argument(
        &self,
        operation: &'static str,
        reason: impl Into<String>,
    ) -> BackendError {
        let error = BackendError::InvalidArgument {
            operation,
            reason: reason.into(),
        };
        self.record_pending_error(error.clone());
        error
    }

    fn check_pending_error(&self) -> BackendResult<()> {
        self.pending_error
            .lock()
            .map_err(|_| BackendError::OperationFailed {
                operation: "metal_error_state",
                reason: "pending error mutex poisoned".into(),
            })?
            .as_ref()
            .cloned()
            .map_or(Ok(()), Err)
    }

    fn flush_pending_checked(&self) -> BackendResult<()> {
        // A sticky encoding error must not short-circuit cleanup: the pending
        // command buffer still owns scratch buffers that are safe to release
        // only after its completion fence has been waited.  Preserve the
        // original error and drain all pending resources before returning it.
        let sticky_error = self.check_pending_error().err();
        let mut pending = self
            .pending
            .lock()
            .map_err(|_| BackendError::OperationFailed {
                operation: "metal_pending_queue",
                reason: "pending mutex poisoned".into(),
            })?;
        let command_error = if let Some(cmd_buf) = pending.take() {
            cmd_buf.0.commit();
            cmd_buf.0.waitUntilCompleted();
            if cmd_buf.0.status() == MTLCommandBufferStatus::Error {
                let reason = cmd_buf
                    .0
                    .error()
                    .map(|error| format!("{error:?}"))
                    .unwrap_or_else(|| "Metal reported command-buffer error".into());
                let error = BackendError::CommandBuffer {
                    backend: "metal",
                    operation: "command_buffer",
                    reason,
                };
                self.record_pending_error(error.clone());
                Some(error)
            } else {
                None
            }
        } else {
            None
        };
        drop(pending);
        let mut deferred =
            self.pending_scratch
                .lock()
                .map_err(|_| BackendError::OperationFailed {
                    operation: "metal_pending_scratch",
                    reason: "pending scratch mutex poisoned".into(),
                })?;
        let completed: Vec<_> = deferred.drain(..).collect();
        drop(deferred);
        let mut release_error = None;
        for (bytes, buf) in completed {
            if let Err(error) = self.return_scratch_checked(bytes, buf.0)
                && release_error.is_none()
            {
                release_error = Some(error);
            }
        }
        if let Some(error) = command_error {
            self.record_pending_error(error.clone());
            return Err(error);
        }
        if let Some(error) = release_error {
            self.record_pending_error(error.clone());
            return Err(error);
        }
        if let Some(error) = sticky_error {
            return Err(error);
        }
        Ok(())
    }

    fn flush_pending(&self) {
        if let Err(error) = self.flush_pending_checked() {
            self.record_pending_error(error);
        }
    }

    /// Look up a cached pipeline state. Lock-free — the map is built
    /// once at construction and immutable thereafter. Missing pipelines are
    /// retained as sticky backend errors and reported as `None` to the
    /// infallible legacy operation surface.
    fn try_pso(
        &self,
        name: &'static str,
    ) -> BackendResult<&ProtocolObject<dyn MTLComputePipelineState>> {
        self.pipelines
            .get(name)
            .map(|pipeline| pipeline.0.as_ref())
            .ok_or_else(|| BackendError::OperationFailed {
                operation: "pipeline_lookup",
                reason: format!("Metal kernel {name} was not compiled"),
            })
    }

    fn pso(&self, name: &'static str) -> Option<&ProtocolObject<dyn MTLComputePipelineState>> {
        match self.try_pso(name) {
            Ok(pipeline) => Some(pipeline),
            Err(error) => {
                self.record_pending_error(error);
                None
            }
        }
    }

    /// Dispatch a 1-D kernel that writes to one buffer of length `n`.
    /// Appends to the pending command buffer — does NOT commit or wait.
    /// Caller must invoke `flush_pending()` before reading any output
    /// buffer back to the host. Generic helper used by `axpy_f32`,
    /// `scale_f32`, `spmv_csr_f32`.
    fn dispatch_1d_lazy(
        &self,
        pso: &ProtocolObject<dyn MTLComputePipelineState>,
        n: u32,
        threadgroup: u64,
        bind: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) {
        if n == 0 {
            return;
        }
        let result = (|| -> BackendResult<()> {
            let cmd_buf = self.try_pending_command_buffer()?;
            let encoder = self.try_compute_encoder(&cmd_buf)?;
            encoder.setComputePipelineState(pso);
            bind(&encoder);
            let threadgroup_count = (n as u64).div_ceil(threadgroup);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: threadgroup_count as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threadgroup as usize,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            Ok(())
        })();
        if let Err(error) = result {
            self.record_pending_error(error);
        }
    }

    /// Execute a batched power-of-two Complex32 FFT entirely in Metal
    /// shared-storage buffers. Radix-4 Stockham stages are used first, with a
    /// residual radix-2 stage for lengths whose exponent is odd. Every stage
    /// ping-pongs through a recycled device buffer; the command buffer is
    /// committed and waited before the scratch buffer is returned to the pool.
    fn fft_stockham_c32(
        &self,
        buffer: &mut MetalBuffer<Complex32>,
        fft_len: usize,
        inverse: bool,
    ) -> BackendResult<()> {
        checked_fft_view_descriptor::<Complex32>(buffer.len, fft_len, "metal")?;
        if fft_len == 0 || !fft_len.is_power_of_two() {
            return Err(BackendError::InvalidArgument {
                operation: "fft",
                reason: format!(
                    "Metal Stockham FFT requires a positive power-of-two length, got {fft_len}"
                ),
            });
        }
        if buffer.len() % fft_len != 0 {
            return Err(BackendError::InvalidArgument {
                operation: "fft",
                reason: format!(
                    "FFT buffer length {} is not a multiple of plan length {fft_len}",
                    buffer.len()
                ),
            });
        }
        let batch = buffer.len() / fft_len;
        let total = buffer.len();
        if total > u32::MAX as usize || batch > u32::MAX as usize {
            return Err(BackendError::InvalidArgument {
                operation: "fft",
                reason: "Metal FFT dimensions exceed uint kernel arguments".into(),
            });
        }
        if total == 0 {
            return Ok(());
        }
        if fft_len == 1 {
            return Ok(());
        }

        let bytes = total
            .checked_mul(std::mem::size_of::<Complex32>())
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "fft",
                reason: "Metal FFT scratch size overflows usize".into(),
            })?;
        let scratch_bytes = bytes as u64;
        let scratch = self.take_scratch_checked(scratch_bytes)?;
        let fft_len_u32 = fft_len as u32;
        let batch_u32 = batch as u32;
        let sign = if inverse { 1.0_f32 } else { -1.0_f32 };

        // Stockham auto-sort: every stage reads one buffer and writes the
        // other, with the stride held at N/2 and output groups packed
        // contiguously. This removes a separate bit-reversal pass while
        // keeping all intermediate data device-resident.
        let mut src_is_buffer = true;
        let mut m = 1usize;
        let mut remaining = fft_len;

        // Match the CPU planner's power-of-two factorisation: radix-4 stages
        // first, followed by one radix-2 stage when the exponent is odd.
        while remaining.is_multiple_of(4) {
            let m_u32 = m as u32;
            let stage_pso = self.try_pso(kernel_names::FFT_STOCKHAM_RADIX4_STAGE_C32)?;
            self.dispatch_1d_lazy(stage_pso, (total / 4) as u32, THREADGROUP_SIZE, |encoder| {
                if src_is_buffer {
                    set_buffer(encoder, 0, &buffer.raw);
                    set_buffer(encoder, 1, &scratch);
                } else {
                    set_buffer(encoder, 0, &scratch);
                    set_buffer(encoder, 1, &buffer.raw);
                }
                set_scalar(encoder, 2, &fft_len_u32);
                set_scalar(encoder, 3, &batch_u32);
                set_scalar(encoder, 4, &m_u32);
                set_scalar(encoder, 5, &sign);
            });
            src_is_buffer = !src_is_buffer;
            m *= 4;
            remaining /= 4;
        }

        if remaining == 2 {
            let m_u32 = m as u32;
            let stage_pso = self.try_pso(kernel_names::FFT_STOCKHAM_RADIX2_STAGE_C32)?;
            self.dispatch_1d_lazy(stage_pso, (total / 2) as u32, THREADGROUP_SIZE, |encoder| {
                if src_is_buffer {
                    set_buffer(encoder, 0, &buffer.raw);
                    set_buffer(encoder, 1, &scratch);
                } else {
                    set_buffer(encoder, 0, &scratch);
                    set_buffer(encoder, 1, &buffer.raw);
                }
                set_scalar(encoder, 2, &fft_len_u32);
                set_scalar(encoder, 3, &batch_u32);
                set_scalar(encoder, 4, &m_u32);
                set_scalar(encoder, 5, &sign);
            });
            src_is_buffer = !src_is_buffer;
        }

        let result = if !src_is_buffer {
            let scratch_buffer = MetalBuffer::<Complex32> {
                raw: ThreadSafe(scratch.clone()),
                len: total,
                _phantom: PhantomData,
            };
            // Keep the ping-pong copy inside the checked transfer boundary;
            // the compatibility wrapper can otherwise log a driver failure
            // and let the FFT appear to have completed.
            self.try_copy_d2d(&scratch_buffer, buffer)
                .and_then(|()| self.flush_pending_checked())
        } else {
            self.flush_pending_checked()
        };
        self.return_scratch(scratch_bytes, scratch);
        result
    }

    /// FMM helper: `dst[p] += phase[p] * src[p]` for Complex32 buffers.
    ///
    /// This is the Metal-native counterpart to
    /// [`crate::fmm_cpu::phase_mul_accumulate_c64`] for mixed-precision FMM
    /// workspaces. Complex64 still uses the shared-storage CPU path because
    /// Apple Silicon Metal rejects `double`.
    pub fn try_fmm_phase_mul_accumulate_c32(
        &self,
        dst: &mut MetalBuffer<Complex32>,
        phase: &MetalBuffer<Complex32>,
        src: &MetalBuffer<Complex32>,
    ) -> BackendResult<()> {
        self.check_pending_error()?;
        if dst.len != phase.len {
            return Err(BackendError::InvalidArgument {
                operation: "fmm_phase_mul_accumulate_c32",
                reason: format!(
                    "phase length {} does not match dst length {}",
                    phase.len, dst.len
                ),
            });
        }
        if dst.len != src.len {
            return Err(BackendError::InvalidArgument {
                operation: "fmm_phase_mul_accumulate_c32",
                reason: format!(
                    "src length {} does not match dst length {}",
                    src.len, dst.len
                ),
            });
        }
        let n = u32::try_from(dst.len).map_err(|_| BackendError::InvalidArgument {
            operation: "fmm_phase_mul_accumulate_c32",
            reason: "buffer length exceeds Metal kernel dimension range".into(),
        })?;
        let pso = self.try_pso(kernel_names::FMM_PHASE_ACCUM_C32)?;
        self.dispatch_1d_lazy(pso, n, THREADGROUP_SIZE, |enc| {
            set_buffer(enc, 0, &dst.raw);
            set_buffer(enc, 1, &phase.raw);
            set_buffer(enc, 2, &src.raw);
            set_scalar(enc, 3, &n);
        });
        self.check_pending_error()
    }

    /// Compatibility wrapper for the historical infallible FMM entry point.
    pub fn fmm_phase_mul_accumulate_c32(
        &self,
        dst: &mut MetalBuffer<Complex32>,
        phase: &MetalBuffer<Complex32>,
        src: &MetalBuffer<Complex32>,
    ) {
        if let Err(error) = self.try_fmm_phase_mul_accumulate_c32(dst, phase, src) {
            self.record_pending_error(error);
        }
    }

    /// FMM helper: gather selected full-row directions into a compact cone row.
    ///
    /// `dst[k] = src[indices[k]]` for every valid index. Invalid indices leave
    /// the destination slot unchanged.
    pub fn fmm_compact_indexed_gather_c32(
        &self,
        src: &MetalBuffer<Complex32>,
        indices: &MetalBuffer<i32>,
        dst: &mut MetalBuffer<Complex32>,
    ) {
        let n = indices.len.min(dst.len) as u32;
        if n == 0 {
            return;
        }
        let src_len = src.len as u32;
        let Some(pso) = self.pso(kernel_names::FMM_COMPACT_GATHER_C32) else {
            return;
        };
        self.dispatch_1d_lazy(pso, n, THREADGROUP_SIZE, |enc| {
            set_buffer(enc, 0, &src.raw);
            set_buffer(enc, 1, &indices.raw);
            set_buffer(enc, 2, &dst.raw);
            set_scalar(enc, 3, &src_len);
            set_scalar(enc, 4, &n);
        });
    }

    /// FMM helper: scatter-add a compact cone row back into selected full-row
    /// directions.
    ///
    /// `dst[indices[k]] += src[k]` for every valid index. Destination indices
    /// must be unique within a dispatch; directional cone active-direction maps
    /// satisfy that contract.
    pub fn fmm_compact_indexed_scatter_accumulate_c32(
        &self,
        src: &MetalBuffer<Complex32>,
        indices: &MetalBuffer<i32>,
        dst: &mut MetalBuffer<Complex32>,
    ) {
        let n = indices.len.min(src.len) as u32;
        if n == 0 {
            return;
        }
        let dst_len = dst.len as u32;
        let Some(pso) = self.pso(kernel_names::FMM_COMPACT_SCATTER_ACCUM_C32) else {
            return;
        };
        self.dispatch_1d_lazy(pso, n, THREADGROUP_SIZE, |enc| {
            set_buffer(enc, 0, &src.raw);
            set_buffer(enc, 1, &indices.raw);
            set_buffer(enc, 2, &dst.raw);
            set_scalar(enc, 3, &dst_len);
            set_scalar(enc, 4, &n);
        });
    }

    /// FMM helper: `dst[dst_indices[i]] += Σ_j A[i,j] * src[src_indices[j]]`.
    ///
    /// One simdgroup computes each dense row. Destination indices must be
    /// unique within a dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn try_fmm_indexed_dense_matvec_accumulate_c32(
        &self,
        matrix: &MetalBuffer<Complex32>,
        rows: usize,
        cols: usize,
        lda: usize,
        src: &MetalBuffer<Complex32>,
        src_indices: &MetalBuffer<i32>,
        dst: &mut MetalBuffer<Complex32>,
        dst_indices: &MetalBuffer<i32>,
    ) -> BackendResult<()> {
        self.check_pending_error()?;
        if lda < cols {
            return Err(BackendError::InvalidArgument {
                operation: "fmm_indexed_dense_matvec_accumulate_c32",
                reason: format!("leading dimension {lda} smaller column count {cols}"),
            });
        }
        let rows = rows.min(dst_indices.len);
        let cols = cols.min(src_indices.len);
        let required = if rows == 0 || cols == 0 {
            0
        } else {
            let Some(required) = (rows - 1)
                .checked_mul(lda)
                .and_then(|offset| offset.checked_add(cols))
            else {
                return Err(BackendError::InvalidArgument {
                    operation: "fmm_indexed_dense_matvec_accumulate_c32",
                    reason: "matrix span overflows usize".into(),
                });
            };
            required
        };
        if matrix.len < required {
            return Err(BackendError::InvalidArgument {
                operation: "fmm_indexed_dense_matvec_accumulate_c32",
                reason: format!(
                    "matrix length {} is smaller than required span {required}",
                    matrix.len
                ),
            });
        }
        if rows > u32::MAX as usize
            || cols > u32::MAX as usize
            || lda > u32::MAX as usize
            || src.len > u32::MAX as usize
            || dst.len > u32::MAX as usize
        {
            return Err(BackendError::InvalidArgument {
                operation: "fmm_indexed_dense_matvec_accumulate_c32",
                reason: "FMM dimensions exceed Metal kernel dimension range".into(),
            });
        }
        if rows == 0 {
            return Ok(());
        }

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let lda_u32 = lda as u32;
        let src_len = src.len as u32;
        let dst_len = dst.len as u32;
        let pso = self.try_pso(kernel_names::FMM_INDEXED_DENSE_MATVEC_ACCUM_C32)?;
        let cmd_buf = self.try_pending_command_buffer()?;
        let encoder = self.try_compute_encoder(&cmd_buf)?;
        encoder.setComputePipelineState(pso);
        set_buffer(&encoder, 0, &matrix.raw);
        set_buffer(&encoder, 1, &src.raw);
        set_buffer(&encoder, 2, &src_indices.raw);
        set_buffer(&encoder, 3, &dst.raw);
        set_buffer(&encoder, 4, &dst_indices.raw);
        set_scalar(&encoder, 5, &rows_u32);
        set_scalar(&encoder, 6, &cols_u32);
        set_scalar(&encoder, 7, &lda_u32);
        set_scalar(&encoder, 8, &src_len);
        set_scalar(&encoder, 9, &dst_len);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: rows,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        self.check_pending_error()
    }

    /// Compatibility wrapper for the historical infallible FMM entry point.
    #[allow(clippy::too_many_arguments)]
    pub fn fmm_indexed_dense_matvec_accumulate_c32(
        &self,
        matrix: &MetalBuffer<Complex32>,
        rows: usize,
        cols: usize,
        lda: usize,
        src: &MetalBuffer<Complex32>,
        src_indices: &MetalBuffer<i32>,
        dst: &mut MetalBuffer<Complex32>,
        dst_indices: &MetalBuffer<i32>,
    ) {
        if let Err(error) = self.try_fmm_indexed_dense_matvec_accumulate_c32(
            matrix,
            rows,
            cols,
            lda,
            src,
            src_indices,
            dst,
            dst_indices,
        ) {
            self.record_pending_error(error);
        }
    }
}

// ---------------------------------------------------------------------------
// Native f32 op implementations
// ---------------------------------------------------------------------------

impl MetalBackend {
    fn axpy_f32_native(&self, alpha: f32, x: &MetalBuffer<f32>, y: &mut MetalBuffer<f32>) {
        if x.len != y.len {
            self.record_invalid_argument(
                "axpy",
                format!("vector lengths {} and {} differ", x.len, y.len),
            );
            return;
        }
        let Ok(n) = u32::try_from(x.len) else {
            self.record_pending_error(BackendError::InvalidArgument {
                operation: "axpy",
                reason: "vector length exceeds Metal's u32 kernel range".into(),
            });
            return;
        };
        let Some(pso) = self.pso(kernel_names::AXPY_F32) else {
            return;
        };
        self.dispatch_1d_lazy(pso, n, THREADGROUP_SIZE, |enc| {
            set_buffer(enc, 0, &x.raw);
            set_buffer(enc, 1, &y.raw);
            set_scalar(enc, 2, &alpha);
            set_scalar(enc, 3, &n);
        });
    }

    fn axpy_many_f32_native(
        &self,
        alphas: &[f32],
        x_basis: &[&MetalBuffer<f32>],
        y: &mut MetalBuffer<f32>,
    ) {
        if alphas.len() != x_basis.len() {
            self.record_invalid_argument(
                "axpy_many",
                format!(
                    "coefficient count {} differs basis count {}",
                    alphas.len(),
                    x_basis.len()
                ),
            );
            return;
        }
        let n = y.len;
        let k = x_basis.len();
        if k == 0 || n == 0 {
            return;
        }
        for x in x_basis {
            if x.len != n {
                self.record_invalid_argument(
                    "axpy_many",
                    format!("basis length {} does not match output length {n}", x.len),
                );
                return;
            }
        }

        let elem = std::mem::size_of::<f32>();
        let (basis_bytes, alpha_bytes) =
            match Self::check_axpy_many_backend_limits::<f32>("axpy_many", n, k) {
                Ok(()) => match Self::checked_axpy_many_byte_sizes("axpy_many", n, k, elem) {
                    Ok(sizes) => sizes,
                    Err(error) => {
                        self.record_pending_error(error);
                        return;
                    }
                },
                Err(error) => {
                    self.record_pending_error(error);
                    return;
                }
            };
        let Some(basis_buf) = self.take_scratch(basis_bytes) else {
            return;
        };
        let Some(alpha_buf) = self.take_scratch(alpha_bytes) else {
            // Return the first scratch allocation when the second one cannot
            // be acquired; otherwise a quota failure leaks the basis buffer.
            self.return_scratch(basis_bytes, basis_buf);
            return;
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                alphas.as_ptr(),
                alpha_buf.contents().as_ptr() as *mut f32,
                k,
            );
        }

        {
            let Some(cmd_buf) = self.pending_command_buffer() else {
                self.defer_scratch_until_flush(basis_bytes, basis_buf);
                self.defer_scratch_until_flush(alpha_bytes, alpha_buf);
                return;
            };
            let Some(encoder) = self.blit_encoder(&cmd_buf) else {
                self.defer_scratch_until_flush(basis_bytes, basis_buf);
                self.defer_scratch_until_flush(alpha_bytes, alpha_buf);
                return;
            };
            for (basis_idx, x) in x_basis.iter().enumerate() {
                let Some(bytes) = n.checked_mul(elem) else {
                    self.record_pending_error(BackendError::InvalidArgument {
                        operation: "axpy_many",
                        reason: "vector byte count overflows usize".into(),
                    });
                    self.defer_scratch_until_flush(basis_bytes, basis_buf);
                    self.defer_scratch_until_flush(alpha_bytes, alpha_buf);
                    return;
                };
                let Some(dst_offset) = basis_idx.checked_mul(bytes) else {
                    self.record_pending_error(BackendError::InvalidArgument {
                        operation: "axpy_many",
                        reason: "packed basis byte offset overflows usize".into(),
                    });
                    self.defer_scratch_until_flush(basis_bytes, basis_buf);
                    self.defer_scratch_until_flush(alpha_bytes, alpha_buf);
                    return;
                };
                unsafe {
                    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &x.raw, 0, &basis_buf, dst_offset, bytes,
                    );
                }
            }
            encoder.endEncoding();

            let Some(pso) = self.pso(kernel_names::AXPY_MANY_F32) else {
                self.defer_scratch_until_flush(basis_bytes, basis_buf);
                self.defer_scratch_until_flush(alpha_bytes, alpha_buf);
                return;
            };
            let Some(encoder) = self.compute_encoder(&cmd_buf) else {
                self.defer_scratch_until_flush(basis_bytes, basis_buf);
                self.defer_scratch_until_flush(alpha_bytes, alpha_buf);
                return;
            };
            encoder.setComputePipelineState(pso);
            set_buffer(&encoder, 0, &basis_buf);
            set_buffer(&encoder, 1, &alpha_buf);
            set_buffer(&encoder, 2, &y.raw);
            let (Ok(n_u32), Ok(k_u32)) = (u32::try_from(n), u32::try_from(k)) else {
                self.record_pending_error(BackendError::InvalidArgument {
                    operation: "axpy_many",
                    reason: "dimensions exceed Metal's u32 kernel range".into(),
                });
                self.defer_scratch_until_flush(basis_bytes, basis_buf);
                self.defer_scratch_until_flush(alpha_bytes, alpha_buf);
                return;
            };
            set_scalar(&encoder, 3, &n_u32);
            set_scalar(&encoder, 4, &k_u32);
            let threadgroup_count = (n as u64).div_ceil(THREADGROUP_SIZE);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: threadgroup_count as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADGROUP_SIZE as usize,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
        }

        self.defer_scratch_until_flush(basis_bytes, basis_buf);
        self.defer_scratch_until_flush(alpha_bytes, alpha_buf);
    }

    fn scale_f32_native(&self, alpha: f32, x: &mut MetalBuffer<f32>) {
        let n = x.len as u32;
        let Some(pso) = self.pso(kernel_names::SCALE_F32) else {
            return;
        };
        self.dispatch_1d_lazy(pso, n, THREADGROUP_SIZE, |enc| {
            set_buffer(enc, 0, &x.raw);
            set_scalar(enc, 1, &alpha);
            set_scalar(enc, 2, &n);
        });
    }

    /// Two-pass GPU reduction: first kernel produces partial sums (one per
    /// threadgroup), second kernel sums those partials to a single scalar
    /// in a one-threadgroup dispatch. The host then reads a single 4-byte
    /// value from a scratch result buffer. Both kernels share the same
    /// pending command buffer so the GPU pipeline never drains between them.
    ///
    /// This replaces the previous "host sums partials" path which required
    /// the host to read `groups` floats and sum them serially, blocking
    /// further GPU work on every dot/nrm2.
    fn reduce_f32_partials(
        &self,
        kernel: &'static str,
        n: u32,
        bind_inputs: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) -> f32 {
        if n == 0 {
            return 0.0;
        }
        let threadgroup = REDUCTION_THREADGROUP_SIZE;
        let groups = (n as u64).div_ceil(threadgroup);
        let partials_bytes = (groups as usize * std::mem::size_of::<f32>()) as u64;
        let Some(partials_buf) = self.take_scratch(partials_bytes) else {
            return 0.0;
        };
        let result_bytes = std::mem::size_of::<f32>() as u64;
        let Some(result_buf) = self.take_scratch(result_bytes) else {
            self.return_scratch(partials_bytes, partials_buf);
            return 0.0;
        };

        let Some(pso_partial) = self.pso(kernel) else {
            self.return_scratch(partials_bytes, partials_buf);
            self.return_scratch(result_bytes, result_buf);
            return 0.0;
        };
        let Some(pso_final) = self.pso(kernel_names::REDUCE_SUM_F32) else {
            self.return_scratch(partials_bytes, partials_buf);
            self.return_scratch(result_bytes, result_buf);
            return 0.0;
        };
        // Queue both reduction passes onto the pending command buffer.
        {
            let Some(cmd_buf) = self.pending_command_buffer() else {
                self.return_scratch(partials_bytes, partials_buf);
                self.return_scratch(result_bytes, result_buf);
                return 0.0;
            };

            // Pass 1: per-threadgroup partials.
            let Some(encoder) = self.compute_encoder(&cmd_buf) else {
                self.return_scratch(partials_bytes, partials_buf);
                self.return_scratch(result_bytes, result_buf);
                return 0.0;
            };
            encoder.setComputePipelineState(pso_partial);
            bind_inputs(&encoder);
            set_buffer(&encoder, 2, &partials_buf);
            set_scalar(&encoder, 3, &n);
            unsafe {
                encoder.setThreadgroupMemoryLength_atIndex(
                    threadgroup as usize * std::mem::size_of::<f32>(),
                    0,
                );
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (groups) as usize,
                    height: 1_usize,
                    depth: 1_usize,
                },
                MTLSize {
                    width: (threadgroup) as usize,
                    height: 1_usize,
                    depth: 1_usize,
                },
            );
            encoder.endEncoding();

            // Pass 2: single-threadgroup final sum.
            let count = groups as u32;
            let Some(encoder2) = self.compute_encoder(&cmd_buf) else {
                self.defer_scratch_until_flush(partials_bytes, partials_buf);
                self.defer_scratch_until_flush(result_bytes, result_buf);
                return 0.0;
            };
            encoder2.setComputePipelineState(pso_final);
            set_buffer(&encoder2, 0, &partials_buf);
            set_buffer(&encoder2, 1, &result_buf);
            set_scalar(&encoder2, 2, &count);
            unsafe {
                encoder2.setThreadgroupMemoryLength_atIndex(
                    threadgroup as usize * std::mem::size_of::<f32>(),
                    0,
                );
            }
            encoder2.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1_usize,
                    height: 1_usize,
                    depth: 1_usize,
                },
                MTLSize {
                    width: (threadgroup) as usize,
                    height: 1_usize,
                    depth: 1_usize,
                },
            );
            encoder2.endEncoding();
        }
        // Need the scalar on host → flush.
        self.flush_pending();

        let ptr = result_buf.contents().as_ptr() as *const f32;
        let acc: f32 = unsafe { *ptr };
        self.return_scratch(partials_bytes, partials_buf);
        self.return_scratch(result_bytes, result_buf);
        acc
    }

    fn dot_f32_native(&self, x: &MetalBuffer<f32>, y: &MetalBuffer<f32>) -> f32 {
        if x.len != y.len {
            self.record_invalid_argument(
                "dot",
                format!("vector lengths {} and {} differ", x.len, y.len),
            );
            return 0.0;
        }
        let Ok(n) = u32::try_from(x.len) else {
            self.record_pending_error(BackendError::InvalidArgument {
                operation: "dot",
                reason: "vector length exceeds Metal's u32 kernel range".into(),
            });
            return 0.0;
        };
        self.reduce_f32_partials(kernel_names::DOT_F32, n, |enc| {
            set_buffer(enc, 0, &x.raw);
            set_buffer(enc, 1, &y.raw);
        })
    }

    fn nrm2_f32_native(&self, x: &MetalBuffer<f32>) -> f32 {
        let n = x.len as u32;
        let sum_sq = self.reduce_f32_partials(kernel_names::NRM2_SQ_F32, n, |enc| {
            set_buffer(enc, 0, &x.raw);
            // Slot 1 is unused for nrm2 — the kernel ignores buffer 1.
            // Bind the same x to keep the layout uniform with `dot`.
            set_buffer(enc, 1, &x.raw);
        });
        sum_sq.sqrt()
    }

    /// Fused multi-dot using the dedicated 2-D MSL kernel.
    ///
    /// Packs basis vectors into a contiguous buffer, dispatches a single
    /// 2-D compute grid `(num_chunks, k)` with threadgroup
    /// `(THREADGROUP_SIZE, 1)`, then host-sums the partials. This collapses
    /// `k` separate encoders into one.
    fn multi_dot_f32_native(
        &self,
        x_basis: &[&MetalBuffer<f32>],
        y: &MetalBuffer<f32>,
        mut output: Option<&mut [f32]>,
    ) -> Vec<f32> {
        let n = y.len;
        let k = x_basis.len();
        if let Some(out) = output.as_ref().filter(|out| out.len() != k) {
            self.record_invalid_argument(
                "multi_dot_into",
                format!("output length {} does not match basis count {k}", out.len()),
            );
            return Vec::new();
        }
        if k == 0 {
            return Vec::new();
        }
        if n == 0 {
            if let Some(out) = output.as_mut() {
                out.fill(0.0);
                return Vec::new();
            }
            return match crate::backend::try_zeroed_host::<f32>(k, "metal_multi_dot") {
                Ok(result) => result,
                Err(error) => {
                    self.record_pending_error(error);
                    Vec::new()
                }
            };
        }
        for x in x_basis {
            if x.len != n {
                self.record_invalid_argument(
                    "multi_dot",
                    format!("basis length {} does not match vector length {n}", x.len),
                );
                return Vec::new();
            }
        }

        let threadgroup = REDUCTION_THREADGROUP_SIZE;
        let n_u64 = match u64::try_from(n) {
            Ok(value) => value,
            Err(_) => {
                self.record_pending_error(BackendError::InvalidArgument {
                    operation: "multi_dot",
                    reason: "vector length exceeds Metal address range".into(),
                });
                return Vec::new();
            }
        };
        let num_chunks = n_u64.div_ceil(threadgroup);
        let num_chunks_usize = match usize::try_from(num_chunks) {
            Ok(chunks) => chunks,
            Err(_) => {
                self.record_pending_error(BackendError::InvalidArgument {
                    operation: "multi_dot",
                    reason: "reduction chunk count exceeds host address range".into(),
                });
                return Vec::new();
            }
        };

        // Pack basis vectors into a contiguous buffer via blit.
        let elem = std::mem::size_of::<f32>();
        let (concat_bytes, result_bytes) =
            match Self::check_axpy_many_backend_limits::<f32>("multi_dot", n, k) {
                Ok(()) => match Self::checked_axpy_many_byte_sizes("multi_dot", n, k, elem) {
                    Ok(sizes) => sizes,
                    Err(error) => {
                        self.record_pending_error(error);
                        return Vec::new();
                    }
                },
                Err(error) => {
                    self.record_pending_error(error);
                    return Vec::new();
                }
            };
        let vector_bytes = match n.checked_mul(elem) {
            Some(bytes) => bytes,
            None => {
                self.record_pending_error(BackendError::InvalidArgument {
                    operation: "multi_dot",
                    reason: "vector byte count overflows usize".into(),
                });
                return Vec::new();
            }
        };
        let Some(concat_buf) = self.take_scratch(concat_bytes) else {
            return Vec::new();
        };
        {
            let Some(cmd_buf) = self.pending_command_buffer() else {
                self.return_scratch(concat_bytes, concat_buf);
                return Vec::new();
            };
            let Some(encoder) = self.blit_encoder(&cmd_buf) else {
                self.return_scratch(concat_bytes, concat_buf);
                return Vec::new();
            };
            for (basis_idx, x) in x_basis.iter().enumerate() {
                let Some(dst_offset) = basis_idx.checked_mul(vector_bytes) else {
                    self.record_pending_error(BackendError::InvalidArgument {
                        operation: "multi_dot",
                        reason: "packed basis byte offset overflows usize".into(),
                    });
                    self.defer_scratch_until_flush(concat_bytes, concat_buf);
                    return Vec::new();
                };
                let bytes = vector_bytes;
                unsafe {
                    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &x.raw,
                        0_usize,
                        &concat_buf,
                        dst_offset,
                        bytes,
                    );
                }
            }
            encoder.endEncoding();
        }

        let partials_bytes = match k
            .checked_mul(num_chunks_usize)
            .and_then(|count| count.checked_mul(elem))
            .and_then(|bytes| u64::try_from(bytes).ok())
        {
            Some(bytes) => bytes,
            None => {
                self.record_pending_error(BackendError::InvalidArgument {
                    operation: "multi_dot",
                    reason: "partial reduction byte count overflows address range".into(),
                });
                self.defer_scratch_until_flush(concat_bytes, concat_buf);
                return Vec::new();
            }
        };
        let Some(partials_buf) = self.take_scratch(partials_bytes) else {
            // The blit that packed `x_basis` into `concat_buf` is already
            // queued on the pending command buffer.  Do not drop the buffer
            // on this allocation-failure path: Metal may still read it.
            // Keep it alive until the command buffer is flushed, just as the
            // successful reduction path does below.
            self.defer_scratch_until_flush(concat_bytes, concat_buf);
            return Vec::new();
        };
        let Some(result_buf) = self.take_scratch(result_bytes) else {
            self.return_scratch(partials_bytes, partials_buf);
            // The basis-pack blit already references `concat_buf`; defer its
            // reuse until the pending command buffer has completed.
            self.defer_scratch_until_flush(concat_bytes, concat_buf);
            return Vec::new();
        };

        let (Ok(n_u32), Ok(_k_u32), Ok(num_chunks_u32)) = (
            u32::try_from(n),
            u32::try_from(k),
            u32::try_from(num_chunks),
        ) else {
            self.defer_scratch_until_flush(concat_bytes, concat_buf);
            self.return_scratch(partials_bytes, partials_buf);
            self.return_scratch(result_bytes, result_buf);
            self.record_pending_error(BackendError::InvalidArgument {
                operation: "multi_dot",
                reason: "dimensions exceed Metal's u32 kernel range".into(),
            });
            return Vec::new();
        };

        let Some(pso) = self.pso(kernel_names::MULTI_DOT_F32) else {
            self.defer_scratch_until_flush(concat_bytes, concat_buf);
            self.return_scratch(partials_bytes, partials_buf);
            self.return_scratch(result_bytes, result_buf);
            return Vec::new();
        };
        let Some(pso_final) = self.pso(kernel_names::REDUCE_MULTI_SUM_F32) else {
            self.defer_scratch_until_flush(concat_bytes, concat_buf);
            self.return_scratch(partials_bytes, partials_buf);
            self.return_scratch(result_bytes, result_buf);
            return Vec::new();
        };
        {
            let Some(cmd_buf) = self.pending_command_buffer() else {
                self.defer_scratch_until_flush(concat_bytes, concat_buf);
                self.return_scratch(partials_bytes, partials_buf);
                self.return_scratch(result_bytes, result_buf);
                return Vec::new();
            };
            let Some(encoder) = self.compute_encoder(&cmd_buf) else {
                self.defer_scratch_until_flush(concat_bytes, concat_buf);
                self.defer_scratch_until_flush(partials_bytes, partials_buf);
                self.defer_scratch_until_flush(result_bytes, result_buf);
                return Vec::new();
            };
            encoder.setComputePipelineState(pso);
            set_buffer(&encoder, 0, &concat_buf);
            set_buffer(&encoder, 1, &y.raw);
            set_buffer(&encoder, 2, &partials_buf);
            set_scalar(&encoder, 3, &n_u32);
            set_scalar(&encoder, 4, &num_chunks_u32);
            unsafe {
                encoder.setThreadgroupMemoryLength_atIndex(
                    threadgroup as usize * std::mem::size_of::<f32>(),
                    0,
                );
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: num_chunks_usize,
                    height: k,
                    depth: 1_usize,
                },
                MTLSize {
                    width: (threadgroup) as usize,
                    height: 1_usize,
                    depth: 1_usize,
                },
            );
            encoder.endEncoding();

            // GPU-side per-basis final reduction: one threadgroup per basis,
            // each strides over `num_chunks` and tree-reduces to one scalar.
            let Some(encoder2) = self.compute_encoder(&cmd_buf) else {
                self.defer_scratch_until_flush(concat_bytes, concat_buf);
                self.defer_scratch_until_flush(partials_bytes, partials_buf);
                self.defer_scratch_until_flush(result_bytes, result_buf);
                return Vec::new();
            };
            encoder2.setComputePipelineState(pso_final);
            set_buffer(&encoder2, 0, &partials_buf);
            set_buffer(&encoder2, 1, &result_buf);
            set_scalar(&encoder2, 2, &num_chunks_u32);
            unsafe {
                encoder2.setThreadgroupMemoryLength_atIndex(
                    threadgroup as usize * std::mem::size_of::<f32>(),
                    0,
                );
            }
            encoder2.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: k,
                    height: 1_usize,
                    depth: 1_usize,
                },
                MTLSize {
                    width: (threadgroup) as usize,
                    height: 1_usize,
                    depth: 1_usize,
                },
            );
            encoder2.endEncoding();
        }

        self.flush_pending();
        if self.check_pending_error().is_err() {
            self.return_scratch(concat_bytes, concat_buf);
            self.return_scratch(partials_bytes, partials_buf);
            self.return_scratch(result_bytes, result_buf);
            return Vec::new();
        }
        let result = if let Some(out) = output {
            let result_slice = unsafe {
                std::slice::from_raw_parts(result_buf.contents().as_ptr() as *const f32, k)
            };
            out.copy_from_slice(result_slice);
            Vec::new()
        } else {
            let result_slice = unsafe {
                std::slice::from_raw_parts(result_buf.contents().as_ptr() as *const f32, k)
            };
            result_slice.to_vec()
        };
        self.return_scratch(concat_bytes, concat_buf);
        self.return_scratch(partials_bytes, partials_buf);
        self.return_scratch(result_bytes, result_buf);
        result
    }

    fn spmv_csr_f32_native(
        &self,
        m: u32,
        row_ptr: &MetalBuffer<i32>,
        col_idx: &MetalBuffer<i32>,
        values: &MetalBuffer<f32>,
        x: &MetalBuffer<f32>,
        y: &mut MetalBuffer<f32>,
    ) {
        let Some(pso) = self.pso(kernel_names::SPMV_CSR_F32) else {
            return;
        };
        self.dispatch_1d_lazy(pso, m, THREADGROUP_SIZE, |enc| {
            set_buffer(enc, 0, &row_ptr.raw);
            set_buffer(enc, 1, &col_idx.raw);
            set_buffer(enc, 2, &values.raw);
            set_buffer(enc, 3, &x.raw);
            set_buffer(enc, 4, &y.raw);
            set_scalar(enc, 5, &m);
        });
    }

    // -----------------------------------------------------------------------
    // Native Complex32 (MSL `float2`) implementations
    // -----------------------------------------------------------------------

    fn axpy_c32_native(
        &self,
        alpha: Complex32,
        x: &MetalBuffer<Complex32>,
        y: &mut MetalBuffer<Complex32>,
    ) {
        if x.len != y.len {
            self.record_invalid_argument(
                "axpy",
                format!("vector lengths {} and {} differ", x.len, y.len),
            );
            return;
        }
        let Ok(n) = u32::try_from(x.len) else {
            self.record_pending_error(BackendError::InvalidArgument {
                operation: "axpy",
                reason: "vector length exceeds Metal's u32 kernel range".into(),
            });
            return;
        };
        let Some(pso) = self.pso(kernel_names::AXPY_C32) else {
            return;
        };
        // alpha is `#[repr(C)] { re: f32, im: f32 }` — 8 bytes, matches MSL `float2`.
        self.dispatch_1d_lazy(pso, n, THREADGROUP_SIZE, |enc| {
            set_buffer(enc, 0, &x.raw);
            set_buffer(enc, 1, &y.raw);
            set_scalar(enc, 2, &alpha);
            set_scalar(enc, 3, &n);
        });
    }

    fn axpy_many_c32_native(
        &self,
        alphas: &[Complex32],
        x_basis: &[&MetalBuffer<Complex32>],
        y: &mut MetalBuffer<Complex32>,
    ) {
        if alphas.len() != x_basis.len() {
            self.record_invalid_argument(
                "axpy_many",
                format!(
                    "coefficient count {} differs basis count {}",
                    alphas.len(),
                    x_basis.len()
                ),
            );
            return;
        }
        let n = y.len;
        let k = x_basis.len();
        if k == 0 || n == 0 {
            return;
        }
        for x in x_basis {
            if x.len != n {
                self.record_invalid_argument(
                    "axpy_many",
                    format!("basis length {} does not match output length {n}", x.len),
                );
                return;
            }
        }

        let elem = std::mem::size_of::<Complex32>();
        let (basis_bytes, alpha_bytes) =
            match Self::check_axpy_many_backend_limits::<Complex32>("axpy_many", n, k) {
                Ok(()) => match Self::checked_axpy_many_byte_sizes("axpy_many", n, k, elem) {
                    Ok(sizes) => sizes,
                    Err(error) => {
                        self.record_pending_error(error);
                        return;
                    }
                },
                Err(error) => {
                    self.record_pending_error(error);
                    return;
                }
            };
        let Some(basis_buf) = self.take_scratch(basis_bytes) else {
            return;
        };
        let Some(alpha_buf) = self.take_scratch(alpha_bytes) else {
            self.return_scratch(basis_bytes, basis_buf);
            return;
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                alphas.as_ptr(),
                alpha_buf.contents().as_ptr() as *mut Complex32,
                k,
            );
        }

        {
            let Some(cmd_buf) = self.pending_command_buffer() else {
                self.defer_scratch_until_flush(basis_bytes, basis_buf);
                self.defer_scratch_until_flush(alpha_bytes, alpha_buf);
                return;
            };
            let Some(encoder) = self.blit_encoder(&cmd_buf) else {
                self.defer_scratch_until_flush(basis_bytes, basis_buf);
                self.defer_scratch_until_flush(alpha_bytes, alpha_buf);
                return;
            };
            for (basis_idx, x) in x_basis.iter().enumerate() {
                let Some(bytes) = n.checked_mul(elem) else {
                    self.record_pending_error(BackendError::InvalidArgument {
                        operation: "axpy_many",
                        reason: "vector byte count overflows usize".into(),
                    });
                    self.defer_scratch_until_flush(basis_bytes, basis_buf);
                    self.defer_scratch_until_flush(alpha_bytes, alpha_buf);
                    return;
                };
                let Some(dst_offset) = basis_idx.checked_mul(bytes) else {
                    self.record_pending_error(BackendError::InvalidArgument {
                        operation: "axpy_many",
                        reason: "packed basis byte offset overflows usize".into(),
                    });
                    self.defer_scratch_until_flush(basis_bytes, basis_buf);
                    self.defer_scratch_until_flush(alpha_bytes, alpha_buf);
                    return;
                };
                unsafe {
                    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &x.raw, 0, &basis_buf, dst_offset, bytes,
                    );
                }
            }
            encoder.endEncoding();

            let Some(pso) = self.pso(kernel_names::AXPY_MANY_C32) else {
                self.defer_scratch_until_flush(basis_bytes, basis_buf);
                self.defer_scratch_until_flush(alpha_bytes, alpha_buf);
                return;
            };
            let Some(encoder) = self.compute_encoder(&cmd_buf) else {
                self.defer_scratch_until_flush(basis_bytes, basis_buf);
                self.defer_scratch_until_flush(alpha_bytes, alpha_buf);
                return;
            };
            encoder.setComputePipelineState(pso);
            set_buffer(&encoder, 0, &basis_buf);
            set_buffer(&encoder, 1, &alpha_buf);
            set_buffer(&encoder, 2, &y.raw);
            let (Ok(n_u32), Ok(k_u32)) = (u32::try_from(n), u32::try_from(k)) else {
                self.record_pending_error(BackendError::InvalidArgument {
                    operation: "axpy_many",
                    reason: "dimensions exceed Metal's u32 kernel range".into(),
                });
                self.defer_scratch_until_flush(basis_bytes, basis_buf);
                self.defer_scratch_until_flush(alpha_bytes, alpha_buf);
                return;
            };
            set_scalar(&encoder, 3, &n_u32);
            set_scalar(&encoder, 4, &k_u32);
            let threadgroup_count = (n as u64).div_ceil(THREADGROUP_SIZE);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: threadgroup_count as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADGROUP_SIZE as usize,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
        }

        self.defer_scratch_until_flush(basis_bytes, basis_buf);
        self.defer_scratch_until_flush(alpha_bytes, alpha_buf);
    }

    fn scale_c32_native(&self, alpha: Complex32, x: &mut MetalBuffer<Complex32>) {
        let n = x.len as u32;
        let Some(pso) = self.pso(kernel_names::SCALE_C32) else {
            return;
        };
        self.dispatch_1d_lazy(pso, n, THREADGROUP_SIZE, |enc| {
            set_buffer(enc, 0, &x.raw);
            set_scalar(enc, 1, &alpha);
            set_scalar(enc, 2, &n);
        });
    }

    /// Reduction returning a Complex32 scalar. Two-pass GPU reduction mirroring
    /// `reduce_f32_partials` but with `float2` partials/output (8-byte stride).
    /// Final pass runs a single-threadgroup reducer; host reads 8 bytes.
    fn reduce_c32_partials(
        &self,
        kernel: &'static str,
        n: u32,
        bind_inputs: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) -> Complex32 {
        if n == 0 {
            return Complex32::new(0.0, 0.0);
        }
        let threadgroup = REDUCTION_THREADGROUP_SIZE;
        let groups = (n as u64).div_ceil(threadgroup);
        let partials_bytes = (groups as usize * std::mem::size_of::<Complex32>()) as u64;
        let Some(partials_buf) = self.take_scratch(partials_bytes) else {
            return Complex32::default();
        };
        let result_bytes = std::mem::size_of::<Complex32>() as u64;
        let Some(result_buf) = self.take_scratch(result_bytes) else {
            self.return_scratch(partials_bytes, partials_buf);
            return Complex32::default();
        };

        let Some(pso_partial) = self.pso(kernel) else {
            self.return_scratch(partials_bytes, partials_buf);
            self.return_scratch(result_bytes, result_buf);
            return Complex32::default();
        };
        let Some(pso_final) = self.pso(kernel_names::REDUCE_SUM_C32) else {
            self.return_scratch(partials_bytes, partials_buf);
            self.return_scratch(result_bytes, result_buf);
            return Complex32::default();
        };
        {
            let Some(cmd_buf) = self.pending_command_buffer() else {
                self.return_scratch(partials_bytes, partials_buf);
                self.return_scratch(result_bytes, result_buf);
                return Complex32::default();
            };

            // Pass 1: per-threadgroup Complex32 partials.
            let Some(encoder) = self.compute_encoder(&cmd_buf) else {
                self.return_scratch(partials_bytes, partials_buf);
                self.return_scratch(result_bytes, result_buf);
                return Complex32::default();
            };
            encoder.setComputePipelineState(pso_partial);
            bind_inputs(&encoder);
            set_buffer(&encoder, 2, &partials_buf);
            set_scalar(&encoder, 3, &n);
            unsafe {
                encoder.setThreadgroupMemoryLength_atIndex(
                    threadgroup as usize * std::mem::size_of::<Complex32>(),
                    0,
                );
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (groups) as usize,
                    height: 1_usize,
                    depth: 1_usize,
                },
                MTLSize {
                    width: (threadgroup) as usize,
                    height: 1_usize,
                    depth: 1_usize,
                },
            );
            encoder.endEncoding();

            // Pass 2: single-threadgroup final reduce.
            let count = groups as u32;
            let Some(encoder2) = self.compute_encoder(&cmd_buf) else {
                self.defer_scratch_until_flush(partials_bytes, partials_buf);
                self.defer_scratch_until_flush(result_bytes, result_buf);
                return Complex32::default();
            };
            encoder2.setComputePipelineState(pso_final);
            set_buffer(&encoder2, 0, &partials_buf);
            set_buffer(&encoder2, 1, &result_buf);
            set_scalar(&encoder2, 2, &count);
            unsafe {
                encoder2.setThreadgroupMemoryLength_atIndex(
                    threadgroup as usize * std::mem::size_of::<Complex32>(),
                    0,
                );
            }
            encoder2.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1_usize,
                    height: 1_usize,
                    depth: 1_usize,
                },
                MTLSize {
                    width: (threadgroup) as usize,
                    height: 1_usize,
                    depth: 1_usize,
                },
            );
            encoder2.endEncoding();
        }
        self.flush_pending();

        let ptr = result_buf.contents().as_ptr() as *const Complex32;
        let acc = unsafe { *ptr };
        self.return_scratch(partials_bytes, partials_buf);
        self.return_scratch(result_bytes, result_buf);
        acc
    }

    fn dot_c32_native(&self, x: &MetalBuffer<Complex32>, y: &MetalBuffer<Complex32>) -> Complex32 {
        if x.len != y.len {
            self.record_invalid_argument(
                "dot",
                format!("vector lengths {} and {} differ", x.len, y.len),
            );
            return Complex32::new(0.0, 0.0);
        }
        let Ok(n) = u32::try_from(x.len) else {
            self.record_pending_error(BackendError::InvalidArgument {
                operation: "dot",
                reason: "vector length exceeds Metal's u32 kernel range".into(),
            });
            return Complex32::new(0.0, 0.0);
        };
        self.reduce_c32_partials(kernel_names::DOT_C32, n, |enc| {
            set_buffer(enc, 0, &x.raw);
            set_buffer(enc, 1, &y.raw);
        })
    }

    fn nrm2_c32_native(&self, x: &MetalBuffer<Complex32>) -> f32 {
        // Real-valued reduction (sum of |x|²) — reuse the f32 partials path.
        let n = x.len as u32;
        let sum_sq = self.reduce_f32_partials(kernel_names::NRM2_SQ_C32, n, |enc| {
            set_buffer(enc, 0, &x.raw);
            set_buffer(enc, 1, &x.raw);
        });
        sum_sq.sqrt()
    }

    fn multi_dot_c32_native(
        &self,
        x_basis: &[&MetalBuffer<Complex32>],
        y: &MetalBuffer<Complex32>,
        mut output: Option<&mut [Complex32]>,
    ) -> Vec<Complex32> {
        let n = y.len;
        let k = x_basis.len();
        if let Some(out) = output.as_ref().filter(|out| out.len() != k) {
            self.record_invalid_argument(
                "multi_dot_into",
                format!("output length {} does not match basis count {k}", out.len()),
            );
            return Vec::new();
        }
        if k == 0 {
            return Vec::new();
        }
        if n == 0 {
            if let Some(out) = output.as_mut() {
                out.fill(Complex32::new(0.0, 0.0));
                return Vec::new();
            }
            return match crate::backend::try_zeroed_host::<Complex32>(k, "metal_multi_dot") {
                Ok(result) => result,
                Err(error) => {
                    self.record_pending_error(error);
                    Vec::new()
                }
            };
        }
        for x in x_basis {
            if x.len != n {
                self.record_invalid_argument(
                    "multi_dot",
                    format!("basis length {} does not match vector length {n}", x.len),
                );
                return Vec::new();
            }
        }

        let threadgroup = REDUCTION_THREADGROUP_SIZE;
        let n_u64 = match u64::try_from(n) {
            Ok(value) => value,
            Err(_) => {
                self.record_pending_error(BackendError::InvalidArgument {
                    operation: "multi_dot",
                    reason: "vector length exceeds Metal address range".into(),
                });
                return Vec::new();
            }
        };
        let num_chunks = n_u64.div_ceil(threadgroup);

        let num_chunks_usize = match usize::try_from(num_chunks) {
            Ok(chunks) => chunks,
            Err(_) => {
                self.record_pending_error(BackendError::InvalidArgument {
                    operation: "multi_dot",
                    reason: "reduction chunk count exceeds host address range".into(),
                });
                return Vec::new();
            }
        };

        // Pack basis vectors into a contiguous buffer via blit.
        let elem = std::mem::size_of::<Complex32>();
        let (concat_bytes, result_bytes) =
            match Self::check_axpy_many_backend_limits::<Complex32>("multi_dot", n, k) {
                Ok(()) => match Self::checked_axpy_many_byte_sizes("multi_dot", n, k, elem) {
                    Ok(sizes) => sizes,
                    Err(error) => {
                        self.record_pending_error(error);
                        return Vec::new();
                    }
                },
                Err(error) => {
                    self.record_pending_error(error);
                    return Vec::new();
                }
            };
        let vector_bytes = match n.checked_mul(elem) {
            Some(bytes) => bytes,
            None => {
                self.record_pending_error(BackendError::InvalidArgument {
                    operation: "multi_dot",
                    reason: "vector byte count overflows usize".into(),
                });
                return Vec::new();
            }
        };
        let Some(concat_buf) = self.take_scratch(concat_bytes) else {
            return Vec::new();
        };
        {
            let Some(cmd_buf) = self.pending_command_buffer() else {
                self.return_scratch(concat_bytes, concat_buf);
                return Vec::new();
            };
            let Some(encoder) = self.blit_encoder(&cmd_buf) else {
                self.return_scratch(concat_bytes, concat_buf);
                return Vec::new();
            };
            for (basis_idx, x) in x_basis.iter().enumerate() {
                let Some(dst_offset) = basis_idx.checked_mul(vector_bytes) else {
                    self.record_pending_error(BackendError::InvalidArgument {
                        operation: "multi_dot",
                        reason: "packed basis byte offset overflows usize".into(),
                    });
                    self.defer_scratch_until_flush(concat_bytes, concat_buf);
                    return Vec::new();
                };
                let bytes = vector_bytes;
                unsafe {
                    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &x.raw,
                        0_usize,
                        &concat_buf,
                        dst_offset,
                        bytes,
                    );
                }
            }
            encoder.endEncoding();
        }

        let partials_bytes = match k
            .checked_mul(num_chunks_usize)
            .and_then(|count| count.checked_mul(elem))
            .and_then(|bytes| u64::try_from(bytes).ok())
        {
            Some(bytes) => bytes,
            None => {
                self.record_pending_error(BackendError::InvalidArgument {
                    operation: "multi_dot",
                    reason: "partial reduction byte count overflows address range".into(),
                });
                self.defer_scratch_until_flush(concat_bytes, concat_buf);
                return Vec::new();
            }
        };
        let Some(partials_buf) = self.take_scratch(partials_bytes) else {
            // The preceding blit still references `concat_buf`; defer its
            // release until completion even though the reduction cannot be
            // submitted.
            self.defer_scratch_until_flush(concat_bytes, concat_buf);
            return Vec::new();
        };
        let Some(result_buf) = self.take_scratch(result_bytes) else {
            self.return_scratch(partials_bytes, partials_buf);
            self.defer_scratch_until_flush(concat_bytes, concat_buf);
            return Vec::new();
        };

        let (Ok(n_u32), Ok(_k_u32), Ok(num_chunks_u32)) = (
            u32::try_from(n),
            u32::try_from(k),
            u32::try_from(num_chunks),
        ) else {
            self.defer_scratch_until_flush(concat_bytes, concat_buf);
            self.return_scratch(partials_bytes, partials_buf);
            self.return_scratch(result_bytes, result_buf);
            self.record_pending_error(BackendError::InvalidArgument {
                operation: "multi_dot",
                reason: "dimensions exceed Metal's u32 kernel range".into(),
            });
            return Vec::new();
        };

        let Some(pso) = self.pso(kernel_names::MULTI_DOT_C32) else {
            self.defer_scratch_until_flush(concat_bytes, concat_buf);
            self.return_scratch(partials_bytes, partials_buf);
            self.return_scratch(result_bytes, result_buf);
            return Vec::new();
        };
        let Some(pso_final) = self.pso(kernel_names::REDUCE_MULTI_SUM_C32) else {
            self.defer_scratch_until_flush(concat_bytes, concat_buf);
            self.return_scratch(partials_bytes, partials_buf);
            self.return_scratch(result_bytes, result_buf);
            return Vec::new();
        };
        {
            let Some(cmd_buf) = self.pending_command_buffer() else {
                self.defer_scratch_until_flush(concat_bytes, concat_buf);
                self.return_scratch(partials_bytes, partials_buf);
                self.return_scratch(result_bytes, result_buf);
                return Vec::new();
            };
            let Some(encoder) = self.compute_encoder(&cmd_buf) else {
                self.defer_scratch_until_flush(concat_bytes, concat_buf);
                self.defer_scratch_until_flush(partials_bytes, partials_buf);
                self.defer_scratch_until_flush(result_bytes, result_buf);
                return Vec::new();
            };
            encoder.setComputePipelineState(pso);
            set_buffer(&encoder, 0, &concat_buf);
            set_buffer(&encoder, 1, &y.raw);
            set_buffer(&encoder, 2, &partials_buf);
            set_scalar(&encoder, 3, &n_u32);
            set_scalar(&encoder, 4, &num_chunks_u32);
            unsafe {
                encoder.setThreadgroupMemoryLength_atIndex(
                    threadgroup as usize * std::mem::size_of::<Complex32>(),
                    0,
                );
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: num_chunks_usize,
                    height: k,
                    depth: 1_usize,
                },
                MTLSize {
                    width: (threadgroup) as usize,
                    height: 1_usize,
                    depth: 1_usize,
                },
            );
            encoder.endEncoding();

            // GPU-side per-basis final reduction: one threadgroup per basis.
            let Some(encoder2) = self.compute_encoder(&cmd_buf) else {
                self.defer_scratch_until_flush(concat_bytes, concat_buf);
                self.defer_scratch_until_flush(partials_bytes, partials_buf);
                self.defer_scratch_until_flush(result_bytes, result_buf);
                return Vec::new();
            };
            encoder2.setComputePipelineState(pso_final);
            set_buffer(&encoder2, 0, &partials_buf);
            set_buffer(&encoder2, 1, &result_buf);
            set_scalar(&encoder2, 2, &num_chunks_u32);
            unsafe {
                encoder2.setThreadgroupMemoryLength_atIndex(
                    threadgroup as usize * std::mem::size_of::<Complex32>(),
                    0,
                );
            }
            encoder2.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: k,
                    height: 1_usize,
                    depth: 1_usize,
                },
                MTLSize {
                    width: (threadgroup) as usize,
                    height: 1_usize,
                    depth: 1_usize,
                },
            );
            encoder2.endEncoding();
        }

        self.flush_pending();
        if self.check_pending_error().is_err() {
            self.return_scratch(concat_bytes, concat_buf);
            self.return_scratch(partials_bytes, partials_buf);
            self.return_scratch(result_bytes, result_buf);
            return Vec::new();
        }
        let result = if let Some(out) = output {
            let result_slice = unsafe {
                std::slice::from_raw_parts(result_buf.contents().as_ptr() as *const Complex32, k)
            };
            out.copy_from_slice(result_slice);
            Vec::new()
        } else {
            let result_slice = unsafe {
                std::slice::from_raw_parts(result_buf.contents().as_ptr() as *const Complex32, k)
            };
            result_slice.to_vec()
        };
        self.return_scratch(concat_bytes, concat_buf);
        self.return_scratch(partials_bytes, partials_buf);
        self.return_scratch(result_bytes, result_buf);
        result
    }

    /// Native flat-basis multi-dot. The caller already owns the basis in the
    /// same contiguous layout consumed by the MSL multi-dot kernels, so this
    /// avoids the per-call blit that `multi_dot_*_native` needs for a slice of
    /// independent buffers.
    fn check_strided_backend_limits(
        operation: &'static str,
        vector_len: usize,
        basis_count: usize,
    ) -> BackendResult<()> {
        let max_u32 = u32::MAX as usize;
        if vector_len > max_u32 {
            return Err(BackendError::InvalidArgument {
                operation,
                reason: format!("vector length {vector_len} exceeds Metal's u32 limit"),
            });
        }
        if basis_count > max_u32 {
            return Err(BackendError::InvalidArgument {
                operation,
                reason: format!("basis count {basis_count} exceeds Metal's u32 limit"),
            });
        }
        let num_chunks = (vector_len as u64).div_ceil(REDUCTION_THREADGROUP_SIZE);
        if num_chunks > u32::MAX as u64 {
            return Err(BackendError::InvalidArgument {
                operation,
                reason: format!("reduction chunk count {num_chunks} exceeds Metal's u32 limit"),
            });
        }
        Ok(())
    }

    fn check_gemv_backend_limits<T: DeviceScalar>(
        operation: &'static str,
        m: usize,
        n: usize,
        lda: usize,
    ) -> BackendResult<()> {
        // Metal's f64/Complex64 GEMV path is host-shared, so it does not use
        // the u32 kernel ABI. f32/Complex32 dispatches all dimensions through
        // the MSL scalar contract and must reject narrowing before dispatch.
        if TypeId::of::<T>() != TypeId::of::<f32>()
            && TypeId::of::<T>() != TypeId::of::<Complex32>()
        {
            return Ok(());
        }
        let max_u32 = u32::MAX as usize;
        for (name, value) in [("m", m), ("n", n), ("lda", lda)] {
            if value > max_u32 {
                return Err(BackendError::InvalidArgument {
                    operation,
                    reason: format!("{name}={value} exceeds Metal's u32 GEMV limit"),
                });
            }
        }
        Ok(())
    }

    fn check_spmv_backend_limits<T: DeviceScalar>(
        operation: &'static str,
        m: usize,
    ) -> BackendResult<()> {
        // f64/Complex64 CSR SpMV is host-shared. Only the native f32 and
        // Complex32 kernels narrow the row count to the MSL u32 ABI.
        if TypeId::of::<T>() != TypeId::of::<f32>()
            && TypeId::of::<T>() != TypeId::of::<Complex32>()
        {
            return Ok(());
        }
        if m > u32::MAX as usize {
            return Err(BackendError::InvalidArgument {
                operation,
                reason: format!("m={m} exceeds Metal's u32 SpMV limit"),
            });
        }
        Ok(())
    }

    fn check_axpy_many_backend_limits<T: DeviceScalar>(
        operation: &'static str,
        vector_len: usize,
        basis_count: usize,
    ) -> BackendResult<()> {
        // f64/Complex64 AXPY-many is host-shared. Native f32/Complex32
        // kernels carry both dimensions through the MSL u32 ABI.
        if TypeId::of::<T>() != TypeId::of::<f32>()
            && TypeId::of::<T>() != TypeId::of::<Complex32>()
        {
            return Ok(());
        }
        let max_u32 = u32::MAX as usize;
        for (name, value) in [("vector length", vector_len), ("basis count", basis_count)] {
            if value > max_u32 {
                return Err(BackendError::InvalidArgument {
                    operation,
                    reason: format!("{name}={value} exceeds Metal's u32 AXPY-many limit"),
                });
            }
        }

        // The native kernels pack all basis vectors into one contiguous
        // scratch buffer and pass byte offsets to Metal's blit encoder.
        // Checking the u32 ABI dimensions alone is insufficient: two valid
        // u32 dimensions can still overflow `usize` or the u64 byte-count
        // expected by the scratch pool.
        let element_size = if TypeId::of::<T>() == TypeId::of::<f32>() {
            std::mem::size_of::<f32>()
        } else {
            std::mem::size_of::<Complex32>()
        };
        Self::checked_axpy_many_byte_sizes(operation, vector_len, basis_count, element_size)?;
        Ok(())
    }

    fn checked_axpy_many_byte_sizes(
        operation: &'static str,
        vector_len: usize,
        basis_count: usize,
        element_size: usize,
    ) -> BackendResult<(u64, u64)> {
        let vector_bytes =
            vector_len
                .checked_mul(element_size)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation,
                    reason: "vector byte count overflows usize".into(),
                })?;
        let basis_bytes =
            basis_count
                .checked_mul(vector_bytes)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation,
                    reason: "packed basis byte count overflows usize".into(),
                })?;
        let alpha_bytes =
            basis_count
                .checked_mul(element_size)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation,
                    reason: "coefficient byte count overflows usize".into(),
                })?;
        let basis_bytes =
            u64::try_from(basis_bytes).map_err(|_| BackendError::InvalidArgument {
                operation,
                reason: "packed basis byte count exceeds Metal address range".into(),
            })?;
        let alpha_bytes =
            u64::try_from(alpha_bytes).map_err(|_| BackendError::InvalidArgument {
                operation,
                reason: "coefficient byte count exceeds Metal address range".into(),
            })?;
        Ok((basis_bytes, alpha_bytes))
    }

    fn check_vector_backend_limits<T: DeviceScalar>(
        operation: &'static str,
        vector_len: usize,
    ) -> BackendResult<()> {
        // f64/Complex64 BLAS-1 and reduction paths are host-shared. Native
        // f32/Complex32 kernels carry the vector length through MSL u32.
        if TypeId::of::<T>() != TypeId::of::<f32>()
            && TypeId::of::<T>() != TypeId::of::<Complex32>()
        {
            return Ok(());
        }
        if vector_len > u32::MAX as usize {
            return Err(BackendError::InvalidArgument {
                operation,
                reason: format!(
                    "vector length {vector_len} exceeds Metal's u32 BLAS-1/reduction limit"
                ),
            });
        }
        Ok(())
    }

    fn check_gather4_backend_limits<T: DeviceScalar>(
        operation: &'static str,
        values_len: usize,
        indices_len: usize,
        output_len: usize,
    ) -> BackendResult<()> {
        // f64/Complex64 gather is host-staged. Native f32/Complex32 kernels
        // use a u32 output gid and compute four-index groups with uint math.
        if TypeId::of::<T>() != TypeId::of::<f32>()
            && TypeId::of::<T>() != TypeId::of::<Complex32>()
        {
            return Ok(());
        }
        let max_u32 = u32::MAX as usize;
        for (name, value) in [
            ("values length", values_len),
            ("indices length", indices_len),
            ("output length", output_len),
        ] {
            if value > max_u32 {
                return Err(BackendError::InvalidArgument {
                    operation,
                    reason: format!("{name}={value} exceeds Metal's u32 gather limit"),
                });
            }
        }
        let index_span =
            output_len
                .checked_mul(4)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation,
                    reason: "gather index span overflows usize".into(),
                })?;
        if index_span > max_u32 {
            return Err(BackendError::InvalidArgument {
                operation,
                reason: format!(
                    "output length {output_len} produces a four-index span beyond Metal's u32 limit"
                ),
            });
        }
        Ok(())
    }

    fn check_segmented_backend_limits<T: DeviceScalar>(
        operation: &'static str,
        _values_len: usize,
        _offsets_len: usize,
        output_len: usize,
    ) -> BackendResult<()> {
        // f64/Complex64 segmented reduction is host-staged. Native
        // f32/Complex32 kernels carry the segment count through MSL u32.
        if TypeId::of::<T>() != TypeId::of::<f32>()
            && TypeId::of::<T>() != TypeId::of::<Complex32>()
        {
            return Ok(());
        }
        if output_len > u32::MAX as usize {
            return Err(BackendError::InvalidArgument {
                operation,
                reason: format!(
                    "output length {output_len} exceeds Metal's u32 segmented-reduction limit"
                ),
            });
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn multi_dot_strided_native<T: BufferElement>(
        &self,
        kernel: &'static str,
        reduce_kernel: &'static str,
        basis: &ProtocolObject<dyn MTLBuffer>,
        y: &ProtocolObject<dyn MTLBuffer>,
        basis_view: &ViewDescriptor,
        y_view: &ViewDescriptor,
        vector_len: usize,
        basis_count: usize,
    ) -> Vec<T> {
        if basis_count == 0 {
            return Vec::new();
        }
        if vector_len == 0 {
            return match crate::backend::try_zeroed_host::<T>(
                basis_count,
                "metal_multi_dot_strided",
            ) {
                Ok(result) => result,
                Err(error) => {
                    self.record_pending_error(error);
                    Vec::new()
                }
            };
        }
        let [basis_row_stride, basis_col_stride] = basis_view.element_strides.as_slice() else {
            let error = self.record_invalid_argument(
                "multi_dot_strided",
                "rank-2 descriptor is missing a stride",
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return Vec::new();
        };
        let [y_stride] = y_view.element_strides.as_slice() else {
            let error = self.record_invalid_argument(
                "multi_dot_strided",
                "rank-1 descriptor is missing a stride",
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return Vec::new();
        };
        let element_size = std::mem::size_of::<T>();
        let Ok(basis_offset) = u32::try_from(basis_view.byte_offset / element_size) else {
            let error = self.record_invalid_argument(
                "multi_dot_strided",
                "basis descriptor offset exceeds Metal's u32 ABI",
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return Vec::new();
        };
        let Ok(y_offset) = u32::try_from(y_view.byte_offset / element_size) else {
            let error = self.record_invalid_argument(
                "multi_dot_strided",
                "input descriptor offset exceeds Metal's u32 ABI",
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return Vec::new();
        };
        let Ok(basis_row_stride) = u32::try_from(*basis_row_stride) else {
            let error = self.record_invalid_argument(
                "multi_dot_strided",
                "basis row stride exceeds Metal's u32 ABI",
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return Vec::new();
        };
        let Ok(basis_col_stride) = u32::try_from(*basis_col_stride) else {
            let error = self.record_invalid_argument(
                "multi_dot_strided",
                "basis column stride exceeds Metal's u32 ABI",
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return Vec::new();
        };
        let Ok(y_stride) = u32::try_from(*y_stride) else {
            let error = self.record_invalid_argument(
                "multi_dot_strided",
                "input stride exceeds Metal's u32 ABI",
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return Vec::new();
        };

        let threadgroup = REDUCTION_THREADGROUP_SIZE;
        let num_chunks = (vector_len as u64).div_ceil(threadgroup);
        let Some(partials_bytes) = basis_count
            .checked_mul(num_chunks as usize)
            .and_then(|count| count.checked_mul(std::mem::size_of::<T>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
        else {
            let error = self.record_invalid_argument(
                "multi_dot_strided",
                format!("partial-buffer size overflows for {basis_count}x{vector_len}"),
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return Vec::new();
        };
        let Some(partials_buf) = self.take_scratch(partials_bytes) else {
            return Vec::new();
        };
        let Some(result_bytes) = basis_count
            .checked_mul(std::mem::size_of::<T>())
            .and_then(|bytes| u64::try_from(bytes).ok())
        else {
            let error = self.record_invalid_argument(
                "multi_dot_strided",
                format!("result-buffer size overflows for {basis_count} vectors"),
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            self.return_scratch(partials_bytes, partials_buf);
            return Vec::new();
        };
        let Some(result_buf) = self.take_scratch(result_bytes) else {
            // `partials_buf` was acquired before the result buffer.  Return
            // it on this allocation-failure path so a transient quota/driver
            // failure does not leak the first scratch reservation.
            self.return_scratch(partials_bytes, partials_buf);
            return Vec::new();
        };
        let Ok(vector_len_u32) = u32::try_from(vector_len) else {
            let error = self.record_invalid_argument(
                "multi_dot_strided",
                "vector length exceeds Metal's u32 limit",
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            self.return_scratch(partials_bytes, partials_buf);
            self.return_scratch(result_bytes, result_buf);
            return Vec::new();
        };
        let Ok(num_chunks_u32) = u32::try_from(num_chunks) else {
            let error = self.record_invalid_argument(
                "multi_dot_strided",
                "chunk count exceeds Metal's u32 limit",
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            self.return_scratch(partials_bytes, partials_buf);
            self.return_scratch(result_bytes, result_buf);
            return Vec::new();
        };

        {
            let Some(pso) = self.pso(kernel) else {
                self.return_scratch(partials_bytes, partials_buf);
                self.return_scratch(result_bytes, result_buf);
                return Vec::new();
            };
            let Some(cmd_buf) = self.pending_command_buffer() else {
                self.return_scratch(partials_bytes, partials_buf);
                self.return_scratch(result_bytes, result_buf);
                return Vec::new();
            };
            let Some(encoder) = self.compute_encoder(&cmd_buf) else {
                self.return_scratch(partials_bytes, partials_buf);
                self.return_scratch(result_bytes, result_buf);
                return Vec::new();
            };
            encoder.setComputePipelineState(pso);
            set_buffer(&encoder, 0, basis);
            set_buffer(&encoder, 1, y);
            set_buffer(&encoder, 2, &partials_buf);
            set_scalar(&encoder, 3, &vector_len_u32);
            set_scalar(&encoder, 4, &num_chunks_u32);
            set_scalar(&encoder, 5, &basis_offset);
            set_scalar(&encoder, 6, &basis_row_stride);
            set_scalar(&encoder, 7, &basis_col_stride);
            set_scalar(&encoder, 8, &y_offset);
            set_scalar(&encoder, 9, &y_stride);
            unsafe {
                encoder.setThreadgroupMemoryLength_atIndex(
                    threadgroup as usize * std::mem::size_of::<T>(),
                    0,
                );
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: num_chunks as usize,
                    height: basis_count,
                    depth: 1,
                },
                MTLSize {
                    width: threadgroup as usize,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();

            let Some(pso_reduce) = self.pso(reduce_kernel) else {
                self.defer_scratch_until_flush(partials_bytes, partials_buf);
                self.defer_scratch_until_flush(result_bytes, result_buf);
                return Vec::new();
            };
            let Some(encoder2) = self.compute_encoder(&cmd_buf) else {
                self.defer_scratch_until_flush(partials_bytes, partials_buf);
                self.defer_scratch_until_flush(result_bytes, result_buf);
                return Vec::new();
            };
            encoder2.setComputePipelineState(pso_reduce);
            set_buffer(&encoder2, 0, &partials_buf);
            set_buffer(&encoder2, 1, &result_buf);
            set_scalar(&encoder2, 2, &num_chunks_u32);
            unsafe {
                encoder2.setThreadgroupMemoryLength_atIndex(
                    threadgroup as usize * std::mem::size_of::<T>(),
                    0,
                );
            }
            encoder2.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: basis_count,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threadgroup as usize,
                    height: 1,
                    depth: 1,
                },
            );
            encoder2.endEncoding();
        }

        if let Err(error) = self.flush_pending_checked() {
            // The checked flush waits for the command buffer before returning
            // an error, so both temporary buffers are safe to recycle here.
            self.return_scratch(partials_bytes, partials_buf);
            self.return_scratch(result_bytes, result_buf);
            self.record_pending_error(error);
            return Vec::new();
        }
        let result = unsafe {
            std::slice::from_raw_parts(result_buf.contents().as_ptr() as *const T, basis_count)
                .to_vec()
        };
        self.return_scratch(partials_bytes, partials_buf);
        self.return_scratch(result_bytes, result_buf);
        result
    }

    fn multi_dot_strided_f32_native(
        &self,
        basis: &MetalBuffer<f32>,
        vector_len: usize,
        basis_count: usize,
        y: &MetalBuffer<f32>,
        basis_view: &ViewDescriptor,
        y_view: &ViewDescriptor,
    ) -> Vec<f32> {
        self.multi_dot_strided_native(
            kernel_names::MULTI_DOT_F32,
            kernel_names::REDUCE_MULTI_SUM_F32,
            &basis.raw,
            &y.raw,
            basis_view,
            y_view,
            vector_len,
            basis_count,
        )
    }

    fn multi_dot_strided_c32_native(
        &self,
        basis: &MetalBuffer<Complex32>,
        vector_len: usize,
        basis_count: usize,
        y: &MetalBuffer<Complex32>,
        basis_view: &ViewDescriptor,
        y_view: &ViewDescriptor,
    ) -> Vec<Complex32> {
        self.multi_dot_strided_native(
            kernel_names::MULTI_DOT_C32,
            kernel_names::REDUCE_MULTI_SUM_C32,
            &basis.raw,
            &y.raw,
            basis_view,
            y_view,
            vector_len,
            basis_count,
        )
    }

    /// Apply a linear combination of vectors packed in a flat basis buffer.
    /// The existing `axpy_many` kernels already consume this exact layout;
    /// only the coefficient buffer needs staging because the coefficients are
    /// supplied as a host slice by the backend trait.
    #[allow(clippy::too_many_arguments)]
    fn basis_axpy_strided_native<T: DeviceScalar + PartialEq>(
        &self,
        kernel: &'static str,
        alpha: T,
        basis: &MetalBuffer<T>,
        vector_len: usize,
        basis_count: usize,
        coeffs: &[T],
        y: &mut MetalBuffer<T>,
        basis_view: &ViewDescriptor,
        y_view: &ViewDescriptor,
    ) {
        if coeffs.len() != basis_count {
            let error = self.record_invalid_argument(
                "basis_axpy_strided",
                format!(
                    "coefficient length {} (expected {basis_count})",
                    coeffs.len()
                ),
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        }
        if y.len != vector_len {
            let error = self.record_invalid_argument(
                "basis_axpy_strided",
                format!("y length {} (expected {vector_len})", y.len),
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        }
        let Some(required) = vector_len.checked_mul(basis_count) else {
            let error = self.record_invalid_argument(
                "basis_axpy_strided",
                format!("basis shape {basis_count}x{vector_len} overflows"),
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        };
        if basis.len < required {
            let error = self.record_invalid_argument(
                "basis_axpy_strided",
                format!("basis length {} (expected at least {required})", basis.len),
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        }
        if basis_count == 0 || vector_len == 0 {
            return;
        }
        let [basis_row_stride, basis_col_stride] = basis_view.element_strides.as_slice() else {
            let error = self.record_invalid_argument(
                "basis_axpy_strided",
                "rank-2 descriptor is missing a stride",
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        };
        let [y_stride] = y_view.element_strides.as_slice() else {
            let error = self.record_invalid_argument(
                "basis_axpy_strided",
                "rank-1 descriptor is missing a stride",
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        };
        let element_size = std::mem::size_of::<T>();
        let Ok(basis_offset) = u32::try_from(basis_view.byte_offset / element_size) else {
            let error = self.record_invalid_argument(
                "basis_axpy_strided",
                "basis descriptor offset exceeds Metal's u32 ABI",
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        };
        let Ok(y_offset) = u32::try_from(y_view.byte_offset / element_size) else {
            let error = self.record_invalid_argument(
                "basis_axpy_strided",
                "output descriptor offset exceeds Metal's u32 ABI",
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        };
        let Ok(basis_row_stride) = u32::try_from(*basis_row_stride) else {
            let error = self.record_invalid_argument(
                "basis_axpy_strided",
                "basis row stride exceeds Metal's u32 ABI",
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        };
        let Ok(basis_col_stride) = u32::try_from(*basis_col_stride) else {
            let error = self.record_invalid_argument(
                "basis_axpy_strided",
                "basis column stride exceeds Metal's u32 ABI",
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        };
        let Ok(y_stride) = u32::try_from(*y_stride) else {
            let error = self.record_invalid_argument(
                "basis_axpy_strided",
                "output stride exceeds Metal's u32 ABI",
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        };

        let Ok(vector_len_u32) = u32::try_from(vector_len) else {
            let error = self.record_invalid_argument(
                "basis_axpy_strided",
                "vector length exceeds Metal's u32 limit",
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        };
        let Ok(basis_count_u32) = u32::try_from(basis_count) else {
            let error = self.record_invalid_argument(
                "basis_axpy_strided",
                "basis count exceeds Metal's u32 limit",
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        };
        // Scaling the coefficients is a temporary host allocation. Keep it
        // under the same staging ledger as the other Metal fallback buffers
        // so quota exhaustion is reported rather than panicking in `Vec`.
        let scaled_coeffs = if alpha != T::one() {
            let (mut values, guard) = match try_host_staging_vec(
                &self.staging_stats,
                coeffs.len(),
                "basis_axpy_strided coefficient staging",
            ) {
                Ok(staging) => staging,
                Err(error) => {
                    self.record_pending_error(error);
                    return;
                }
            };
            for (value, &coefficient) in values.iter_mut().zip(coeffs) {
                *value = alpha * coefficient;
            }
            Some((values, guard))
        } else {
            None
        };
        let upload_coeffs = scaled_coeffs
            .as_ref()
            .map_or(coeffs, |(values, _guard)| values.as_slice());
        let Some(coeff_bytes) = basis_count
            .checked_mul(std::mem::size_of::<T>())
            .and_then(|bytes| u64::try_from(bytes).ok())
        else {
            let error = self.record_invalid_argument(
                "basis_axpy_strided",
                format!("coefficient-buffer size overflows for {basis_count} vectors"),
            );
            log::error!(
                target: "sonium_compute::metal",
                "{error}"
            );
            return;
        };
        let Some(coeff_buf) = self.take_scratch(coeff_bytes) else {
            return;
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                upload_coeffs.as_ptr(),
                coeff_buf.contents().as_ptr() as *mut T,
                basis_count,
            );
        }

        {
            let Some(pso) = self.pso(kernel) else {
                self.return_scratch(coeff_bytes, coeff_buf);
                return;
            };
            let Some(cmd_buf) = self.pending_command_buffer() else {
                self.return_scratch(coeff_bytes, coeff_buf);
                return;
            };
            let Some(encoder) = self.compute_encoder(&cmd_buf) else {
                self.return_scratch(coeff_bytes, coeff_buf);
                return;
            };
            encoder.setComputePipelineState(pso);
            set_buffer(&encoder, 0, &basis.raw);
            set_buffer(&encoder, 1, &coeff_buf);
            set_buffer(&encoder, 2, &y.raw);
            set_scalar(&encoder, 3, &vector_len_u32);
            set_scalar(&encoder, 4, &basis_count_u32);
            set_scalar(&encoder, 5, &basis_offset);
            set_scalar(&encoder, 6, &basis_row_stride);
            set_scalar(&encoder, 7, &basis_col_stride);
            set_scalar(&encoder, 8, &y_offset);
            set_scalar(&encoder, 9, &y_stride);
            let threadgroup_count = (vector_len as u64).div_ceil(THREADGROUP_SIZE);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: threadgroup_count as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADGROUP_SIZE as usize,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
        }
        self.defer_scratch_until_flush(coeff_bytes, coeff_buf);
    }

    /// Native MSL GEMV (NoTrans) for f32. Launches one simdgroup (32 threads)
    /// per output row; each thread strides through `n` and `simd_sum`
    /// reduces. A is column-major with leading dimension `lda`.
    #[allow(clippy::too_many_arguments)]
    fn gemv_f32_native_notrans(
        &self,
        m: u32,
        n: u32,
        alpha: f32,
        a: &MetalBuffer<f32>,
        lda: u32,
        x: &MetalBuffer<f32>,
        beta: f32,
        y: &mut MetalBuffer<f32>,
    ) {
        if m == 0 {
            return;
        }
        let Some(pso) = self.pso(kernel_names::GEMV_F32_NOTRANS) else {
            return;
        };
        let Some(cmd_buf) = self.pending_command_buffer() else {
            return;
        };
        let Some(encoder) = self.compute_encoder(&cmd_buf) else {
            return;
        };
        encoder.setComputePipelineState(pso);
        set_buffer(&encoder, 0, &a.raw);
        set_buffer(&encoder, 1, &x.raw);
        set_buffer(&encoder, 2, &y.raw);
        set_scalar(&encoder, 3, &alpha);
        set_scalar(&encoder, 4, &beta);
        set_scalar(&encoder, 5, &m);
        set_scalar(&encoder, 6, &n);
        set_scalar(&encoder, 7, &lda);
        // One simdgroup (32 threads) per output row.
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (m as u64) as usize,
                height: 1_usize,
                depth: 1_usize,
            },
            MTLSize {
                width: 32_usize,
                height: 1_usize,
                depth: 1_usize,
            },
        );
        encoder.endEncoding();
    }

    #[allow(clippy::too_many_arguments)]
    fn gemv_c32_native_notrans(
        &self,
        m: u32,
        n: u32,
        alpha: Complex32,
        a: &MetalBuffer<Complex32>,
        lda: u32,
        x: &MetalBuffer<Complex32>,
        beta: Complex32,
        y: &mut MetalBuffer<Complex32>,
    ) {
        if m == 0 {
            return;
        }
        let Some(pso) = self.pso(kernel_names::GEMV_C32_NOTRANS) else {
            return;
        };
        let Some(cmd_buf) = self.pending_command_buffer() else {
            return;
        };
        let Some(encoder) = self.compute_encoder(&cmd_buf) else {
            return;
        };
        encoder.setComputePipelineState(pso);
        set_buffer(&encoder, 0, &a.raw);
        set_buffer(&encoder, 1, &x.raw);
        set_buffer(&encoder, 2, &y.raw);
        set_scalar(&encoder, 3, &alpha);
        set_scalar(&encoder, 4, &beta);
        set_scalar(&encoder, 5, &m);
        set_scalar(&encoder, 6, &n);
        set_scalar(&encoder, 7, &lda);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (m as u64) as usize,
                height: 1_usize,
                depth: 1_usize,
            },
            MTLSize {
                width: 32_usize,
                height: 1_usize,
                depth: 1_usize,
            },
        );
        encoder.endEncoding();
    }

    #[allow(clippy::too_many_arguments)]
    fn gemv_f32_native_trans(
        &self,
        m: u32,
        n: u32,
        alpha: f32,
        a: &MetalBuffer<f32>,
        lda: u32,
        x: &MetalBuffer<f32>,
        beta: f32,
        y: &mut MetalBuffer<f32>,
    ) {
        if m == 0 {
            return;
        }
        let Some(pso) = self.pso(kernel_names::GEMV_F32_TRANS) else {
            return;
        };
        let Some(cmd_buf) = self.pending_command_buffer() else {
            return;
        };
        let Some(encoder) = self.compute_encoder(&cmd_buf) else {
            return;
        };
        encoder.setComputePipelineState(pso);
        set_buffer(&encoder, 0, &a.raw);
        set_buffer(&encoder, 1, &x.raw);
        set_buffer(&encoder, 2, &y.raw);
        set_scalar(&encoder, 3, &alpha);
        set_scalar(&encoder, 4, &beta);
        set_scalar(&encoder, 5, &m);
        set_scalar(&encoder, 6, &n);
        set_scalar(&encoder, 7, &lda);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (m as u64) as usize,
                height: 1_usize,
                depth: 1_usize,
            },
            MTLSize {
                width: 32_usize,
                height: 1_usize,
                depth: 1_usize,
            },
        );
        encoder.endEncoding();
    }

    #[allow(clippy::too_many_arguments)]
    fn gemv_c32_native_trans(
        &self,
        m: u32,
        n: u32,
        alpha: Complex32,
        a: &MetalBuffer<Complex32>,
        lda: u32,
        x: &MetalBuffer<Complex32>,
        beta: Complex32,
        y: &mut MetalBuffer<Complex32>,
        conjugate: bool,
    ) {
        if m == 0 {
            return;
        }
        let Some(pso) = self.pso(kernel_names::GEMV_C32_TRANS) else {
            return;
        };
        let Some(cmd_buf) = self.pending_command_buffer() else {
            return;
        };
        let Some(encoder) = self.compute_encoder(&cmd_buf) else {
            return;
        };
        encoder.setComputePipelineState(pso);
        set_buffer(&encoder, 0, &a.raw);
        set_buffer(&encoder, 1, &x.raw);
        set_buffer(&encoder, 2, &y.raw);
        set_scalar(&encoder, 3, &alpha);
        set_scalar(&encoder, 4, &beta);
        set_scalar(&encoder, 5, &m);
        set_scalar(&encoder, 6, &n);
        set_scalar(&encoder, 7, &lda);
        let conj_flag: u32 = if conjugate { 1 } else { 0 };
        set_scalar(&encoder, 8, &conj_flag);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (m as u64) as usize,
                height: 1_usize,
                depth: 1_usize,
            },
            MTLSize {
                width: 32_usize,
                height: 1_usize,
                depth: 1_usize,
            },
        );
        encoder.endEncoding();
    }

    fn spmv_csr_c32_native(
        &self,
        m: u32,
        row_ptr: &MetalBuffer<i32>,
        col_idx: &MetalBuffer<i32>,
        values: &MetalBuffer<Complex32>,
        x: &MetalBuffer<Complex32>,
        y: &mut MetalBuffer<Complex32>,
    ) {
        let Some(pso) = self.pso(kernel_names::SPMV_CSR_C32) else {
            return;
        };
        self.dispatch_1d_lazy(pso, m, THREADGROUP_SIZE, |enc| {
            set_buffer(enc, 0, &row_ptr.raw);
            set_buffer(enc, 1, &col_idx.raw);
            set_buffer(enc, 2, &values.raw);
            set_buffer(enc, 3, &x.raw);
            set_buffer(enc, 4, &y.raw);
            set_scalar(enc, 5, &m);
        });
    }
}

// ---------------------------------------------------------------------------
// Native segmented reduction and local contraction kernels.
impl MetalBackend {
    fn segmented_sum_f32_native(
        &self,
        values: &MetalBuffer<f32>,
        offsets: &MetalBuffer<i32>,
        output: &mut MetalBuffer<f32>,
    ) {
        let segments = output.len as u32;
        let Some(pso) = self.pso(kernel_names::SEGMENTED_SUM_F32) else {
            return;
        };
        self.dispatch_1d_lazy(pso, segments, THREADGROUP_SIZE, |encoder| {
            set_buffer(encoder, 0, &values.raw);
            set_buffer(encoder, 1, &offsets.raw);
            set_buffer(encoder, 2, &output.raw);
            set_scalar(encoder, 3, &segments);
        });
    }

    fn segmented_sum_c32_native(
        &self,
        values: &MetalBuffer<Complex32>,
        offsets: &MetalBuffer<i32>,
        output: &mut MetalBuffer<Complex32>,
    ) {
        let segments = output.len as u32;
        let Some(pso) = self.pso(kernel_names::SEGMENTED_SUM_C32) else {
            return;
        };
        self.dispatch_1d_lazy(pso, segments, THREADGROUP_SIZE, |encoder| {
            set_buffer(encoder, 0, &values.raw);
            set_buffer(encoder, 1, &offsets.raw);
            set_buffer(encoder, 2, &output.raw);
            set_scalar(encoder, 3, &segments);
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn batched_local_contract_f32_native(
        &self,
        lhs: &MetalBuffer<f32>,
        rhs: &MetalBuffer<f32>,
        output: &mut MetalBuffer<f32>,
        batches: u32,
        rows: u32,
        inner: u32,
        cols: u32,
    ) {
        let total = batches * rows * cols;
        if total == 0 {
            return;
        }
        let Some(pso) = self.pso(kernel_names::BATCHED_LOCAL_CONTRACT_F32) else {
            return;
        };
        self.dispatch_1d_lazy(pso, total, THREADGROUP_SIZE, |encoder| {
            set_buffer(encoder, 0, &lhs.raw);
            set_buffer(encoder, 1, &rhs.raw);
            set_buffer(encoder, 2, &output.raw);
            set_scalar(encoder, 3, &batches);
            set_scalar(encoder, 4, &rows);
            set_scalar(encoder, 5, &inner);
            set_scalar(encoder, 6, &cols);
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn batched_local_contract_c32_native(
        &self,
        lhs: &MetalBuffer<Complex32>,
        rhs: &MetalBuffer<Complex32>,
        output: &mut MetalBuffer<Complex32>,
        batches: u32,
        rows: u32,
        inner: u32,
        cols: u32,
    ) {
        let total = batches * rows * cols;
        if total == 0 {
            return;
        }
        let Some(pso) = self.pso(kernel_names::BATCHED_LOCAL_CONTRACT_C32) else {
            return;
        };
        self.dispatch_1d_lazy(pso, total, THREADGROUP_SIZE, |encoder| {
            set_buffer(encoder, 0, &lhs.raw);
            set_buffer(encoder, 1, &rhs.raw);
            set_buffer(encoder, 2, &output.raw);
            set_scalar(encoder, 3, &batches);
            set_scalar(encoder, 4, &rows);
            set_scalar(encoder, 5, &inner);
            set_scalar(encoder, 6, &cols);
        });
    }
}

// MPSMatrixMultiplication f32 GEMM
// ---------------------------------------------------------------------------

impl MetalBackend {
    /// Get-or-create a cached `MPSMatrixMultiplication` kernel. Apple's
    /// kernel binds alpha/beta + transpose flags + shape at construction
    /// time and has no settable properties for those — distinct alpha/beta
    /// require distinct kernel instances, hence the wide cache key.
    fn mps_kernel_for(&self, key: MpsMmKey) -> BackendResult<MpsMatMul> {
        let mut cache = self
            .mps_kernels
            .lock()
            .map_err(|_| BackendError::OperationFailed {
                operation: "metal_mps_kernel_cache",
                reason: "MPS kernel cache mutex poisoned".into(),
            })?;
        if let Some(k) = cache.get(&key) {
            return Ok(k.0.clone());
        }
        // SAFETY: MPSMatrixMultiplication's documented init contract — the
        // device outlives the kernel (it's part of MetalBackend), the
        // dimensions and scalars are plain values.
        let allocated: Allocated<MPSMatrixMultiplication> = MPSMatrixMultiplication::alloc();
        let kernel = unsafe {
            MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                allocated,
                &self.device,
                key.trans_left,
                key.trans_right,
                key.m as usize,
                key.n as usize,
                key.k as usize,
                f64::from_bits(key.alpha_bits),
                f64::from_bits(key.beta_bits),
            )
        };
        cache.insert(key, ThreadSafe(kernel.clone()));
        Ok(kernel)
    }

    /// Build a `MPSMatrix` wrapping `buf` with the given row-major shape.
    /// `lda_elems` is the leading dimension in elements (rowBytes is
    /// `lda_elems * sizeof(f32)`).
    fn mps_matrix_f32(
        &self,
        buf: &MetalBuffer<f32>,
        rows: usize,
        cols: usize,
        lda_elems: usize,
    ) -> Retained<MPSMatrix> {
        // SAFETY: the buffer outlives the MPSMatrix (caller holds &MetalBuffer);
        // descriptor params are plain values.
        let dtype = MPSDataType::Float32;
        let row_bytes = lda_elems * std::mem::size_of::<f32>();
        let desc = unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                rows, cols, row_bytes, dtype,
            )
        };
        let allocated: Allocated<MPSMatrix> = MPSMatrix::alloc();
        unsafe { MPSMatrix::initWithBuffer_descriptor(allocated, &buf.raw, &desc) }
    }

    fn mps_vector_f32(&self, buf: &MetalBuffer<f32>, len: usize) -> Retained<MPSVector> {
        let desc = unsafe {
            MPSVectorDescriptor::vectorDescriptorWithLength_dataType(len, MPSDataType::Float32)
        };
        let allocated: Allocated<MPSVector> = MPSVector::alloc();
        unsafe { MPSVector::initWithBuffer_descriptor(allocated, &buf.raw, &desc) }
    }

    /// Dispatch policy for Metal f32 GEMM.
    ///
    /// On current Apple Silicon measurements, MPSMatrixMultiplication loses
    /// to the SME2 CPU path for tiny/small GEMMs because dispatch overhead
    /// dominates. Shared-storage buffers let us run those cases on CPU
    /// without a host-staging copy.
    pub fn f32_gemm_uses_mps(m: usize, n: usize, k: usize) -> bool {
        m.checked_mul(n)
            .and_then(|mn| mn.checked_mul(k))
            .is_none_or(|fmas| fmas > F32_GEMM_CPU_SHARED_MAX_FMAS)
    }

    /// Dispatch policy for the Complex32 4-real-MPS GEMM prototype.
    ///
    /// The prototype launches four f32 GEMMs plus split/combine work, so the
    /// same small-shape CPU/shared-memory rule is even more important than
    /// for plain f32.
    /// Capability metadata for GEMM before the public descriptor becomes
    /// shape-aware. f32 and Complex32 may use MPS for large shapes, but both
    /// also have a shared-memory CPU path; f64 and Complex64 are always host
    /// shared-memory on Metal. Reporting the conservative state prevents a
    /// caller from treating a shape-dependent operation as vendor-native.
    fn gemm_capabilities() -> ScalarCapabilities {
        ScalarCapabilities::splat(CapabilityState::HostSharedFallback)
    }

    fn native_f32_c32_capabilities() -> ScalarCapabilities {
        ScalarCapabilities::new(
            CapabilityState::Native,
            CapabilityState::HostSharedFallback,
            CapabilityState::Native,
            CapabilityState::HostSharedFallback,
        )
    }

    /// Metal's Stockham FFT kernel operates on interleaved Complex32 values.
    /// Real buffers and Complex64 buffers are rejected by `try_fft_in_place`
    /// before any native resource is touched, so the capability table must
    /// describe only the implemented Complex32 cell.
    fn fft_capabilities() -> ScalarCapabilities {
        ScalarCapabilities::new(
            CapabilityState::Unsupported,
            CapabilityState::Unsupported,
            CapabilityState::Native,
            CapabilityState::Unsupported,
        )
    }

    pub fn c32_gemm_uses_mps(m: usize, n: usize, k: usize) -> bool {
        m.checked_mul(n)
            .and_then(|mn| mn.checked_mul(k))
            .is_none_or(|fmas| fmas > C32_GEMM_CPU_SHARED_MAX_FMAS)
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_gemm_f32_shared(
        &self,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &MetalBuffer<f32>,
        lda: usize,
        b: &MetalBuffer<f32>,
        ldb: usize,
        beta: f32,
        c: &mut MetalBuffer<f32>,
        ldc: usize,
    ) {
        self.flush_pending();
        cpu_blas3::gemm_f32(
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a.host_slice(),
            lda,
            b.host_slice(),
            ldb,
            beta,
            c.host_slice_mut(),
            ldc,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_gemm_c32_shared(
        &self,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex32,
        a: &MetalBuffer<Complex32>,
        lda: usize,
        b: &MetalBuffer<Complex32>,
        ldb: usize,
        beta: Complex32,
        c: &mut MetalBuffer<Complex32>,
        ldc: usize,
    ) {
        self.flush_pending();
        cpu_blas3::gemm_c32(
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a.host_slice(),
            lda,
            b.host_slice(),
            ldb,
            beta,
            c.host_slice_mut(),
            ldc,
        );
    }

    /// Experimental f32 GEMV through `MPSMatrixVectorMultiplication`.
    ///
    /// The default `Backend::gemv<f32>` still uses the native MSL kernel,
    /// which is tuned for Sonium's row-major buffers and pending-command
    /// batching. This hook exists so benches can compare that production
    /// path against Apple's MPS matrix-vector kernel on NoTrans buffers.
    /// Transpose benchmarks must prepack `A^T` as a row-major matrix first.
    #[allow(clippy::too_many_arguments)]
    pub fn try_gemv_f32_mps_experimental(
        &self,
        op: Transpose,
        m: usize,
        n: usize,
        alpha: f32,
        a: &MetalBuffer<f32>,
        lda: usize,
        x: &MetalBuffer<f32>,
        beta: f32,
        y: &mut MetalBuffer<f32>,
    ) -> BackendResult<()> {
        if !matches!(op, Transpose::NoTrans) {
            return Err(BackendError::InvalidArgument {
                operation: "gemv_f32_mps_experimental",
                reason: "MPS f32 GEMV requires a physically prepacked NoTrans matrix".into(),
            });
        }
        let mps_a = self.mps_matrix_f32(a, m, n, lda);
        let mps_x = self.mps_vector_f32(x, n);
        let mps_y = self.mps_vector_f32(y, m);
        let kernel = unsafe {
            MPSMatrixVectorMultiplication::initWithDevice_transpose_rows_columns_alpha_beta(
                MPSMatrixVectorMultiplication::alloc(),
                &self.device,
                false,
                m,
                n,
                alpha as f64,
                beta as f64,
            )
        };

        let cmd_buf = self.try_pending_command_buffer()?;
        unsafe {
            kernel.encodeToCommandBuffer_inputMatrix_inputVector_resultVector(
                &cmd_buf, &mps_a, &mps_x, &mps_y,
            );
        }
        Ok(())
    }

    /// Compatibility wrapper for the experimental MPS GEMV benchmark hook.
    /// Checked callers should use [`Self::try_gemv_f32_mps_experimental`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_f32_mps_experimental(
        &self,
        op: Transpose,
        m: usize,
        n: usize,
        alpha: f32,
        a: &MetalBuffer<f32>,
        lda: usize,
        x: &MetalBuffer<f32>,
        beta: f32,
        y: &mut MetalBuffer<f32>,
    ) {
        if let Err(error) = self.try_gemv_f32_mps_experimental(op, m, n, alpha, a, lda, x, beta, y)
        {
            self.record_pending_error(error);
        }
    }

    /// `C := alpha * op(A) * op(B) + beta * C` on f32 via MPSMatrixMultiplication.
    /// `transa_left` / `transb_right` choose whether each operand is transposed
    /// (ConjTrans collapses to Trans on real types — caller handles that).
    #[allow(clippy::too_many_arguments)]
    fn mps_gemm_f32(
        &self,
        trans_left: bool,
        trans_right: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &MetalBuffer<f32>,
        lda: usize,
        b: &MetalBuffer<f32>,
        ldb: usize,
        beta: f32,
        c: &mut MetalBuffer<f32>,
        ldc: usize,
    ) -> BackendResult<()> {
        // MPSMatrixDescriptor takes the STORED shape (rows × cols) with its
        // own rowBytes; the kernel's `transposeLeft/Right` flags do the
        // operator-side transpose. Sonium's contract for NoTrans-A says
        // stored A is m×k with `lda >= k`; for Trans-A stored A is k×m with
        // `lda >= m`. Same shape convention for B.
        let (a_rows, a_cols) = if trans_left { (k, m) } else { (m, k) };
        let (b_rows, b_cols) = if trans_right { (n, k) } else { (k, n) };
        let mps_a = self.mps_matrix_f32(a, a_rows, a_cols, lda);
        let mps_b = self.mps_matrix_f32(b, b_rows, b_cols, ldb);
        let mps_c = self.mps_matrix_f32(c, m, n, ldc);

        let key = MpsMmKey {
            trans_left,
            trans_right,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            alpha_bits: (alpha as f64).to_bits(),
            beta_bits: (beta as f64).to_bits(),
        };
        let kernel = self.mps_kernel_for(key)?;
        let cmd_buf = self.try_pending_command_buffer()?;
        // SAFETY: all four objects (kernel, A, B, C) outlive the encode call;
        // MPS validates dimensions internally.
        unsafe {
            kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                &cmd_buf, &mps_a, &mps_b, &mps_c,
            );
        }
        Ok(())
    }

    /// Prototype Complex32 GEMM path: split A/B into real and imaginary f32
    /// shared buffers, run four MPS f32 GEMMs, then combine into Complex32.
    #[allow(clippy::too_many_arguments)]
    fn mps_gemm_c32_prototype(
        &self,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex32,
        a: &MetalBuffer<Complex32>,
        lda: usize,
        b: &MetalBuffer<Complex32>,
        ldb: usize,
        beta: Complex32,
        c: &mut MetalBuffer<Complex32>,
        ldc: usize,
    ) -> BackendResult<()> {
        let mut ar =
            unsafe { self.raw_alloc_uninitialized::<f32>(a.len) }.map_err(BackendError::from)?;
        let mut ai =
            unsafe { self.raw_alloc_uninitialized::<f32>(a.len) }.map_err(BackendError::from)?;
        let mut br =
            unsafe { self.raw_alloc_uninitialized::<f32>(b.len) }.map_err(BackendError::from)?;
        let mut bi =
            unsafe { self.raw_alloc_uninitialized::<f32>(b.len) }.map_err(BackendError::from)?;
        let mut rr =
            unsafe { self.raw_alloc_uninitialized::<f32>(c.len) }.map_err(BackendError::from)?;
        let mut ii =
            unsafe { self.raw_alloc_uninitialized::<f32>(c.len) }.map_err(BackendError::from)?;
        let mut ri =
            unsafe { self.raw_alloc_uninitialized::<f32>(c.len) }.map_err(BackendError::from)?;
        let mut ir =
            unsafe { self.raw_alloc_uninitialized::<f32>(c.len) }.map_err(BackendError::from)?;

        let conj_a = matches!(transa, Transpose::ConjTrans);
        let conj_b = matches!(transb, Transpose::ConjTrans);
        {
            let ar = ar.host_slice_mut();
            let ai = ai.host_slice_mut();
            for (idx, src) in a.host_slice().iter().enumerate() {
                ar[idx] = src.re;
                ai[idx] = if conj_a { -src.im } else { src.im };
            }
        }
        {
            let br = br.host_slice_mut();
            let bi = bi.host_slice_mut();
            for (idx, src) in b.host_slice().iter().enumerate() {
                br[idx] = src.re;
                bi[idx] = if conj_b { -src.im } else { src.im };
            }
        }

        let trans_left = !matches!(transa, Transpose::NoTrans);
        let trans_right = !matches!(transb, Transpose::NoTrans);
        self.mps_gemm_f32(
            trans_left,
            trans_right,
            m,
            n,
            k,
            1.0,
            &ar,
            lda,
            &br,
            ldb,
            0.0,
            &mut rr,
            ldc,
        )?;
        self.mps_gemm_f32(
            trans_left,
            trans_right,
            m,
            n,
            k,
            1.0,
            &ai,
            lda,
            &bi,
            ldb,
            0.0,
            &mut ii,
            ldc,
        )?;
        self.mps_gemm_f32(
            trans_left,
            trans_right,
            m,
            n,
            k,
            1.0,
            &ar,
            lda,
            &bi,
            ldb,
            0.0,
            &mut ri,
            ldc,
        )?;
        self.mps_gemm_f32(
            trans_left,
            trans_right,
            m,
            n,
            k,
            1.0,
            &ai,
            lda,
            &br,
            ldb,
            0.0,
            &mut ir,
            ldc,
        )?;
        self.flush_pending_checked()?;

        let rr = rr.host_slice();
        let ii = ii.host_slice();
        let ri = ri.host_slice();
        let ir = ir.host_slice();
        let c = c.host_slice_mut();
        for row in 0..m {
            for col in 0..n {
                let idx = row * ldc + col;
                let prod_re = rr[idx] - ii[idx];
                let prod_im = ri[idx] + ir[idx];
                let old = c[idx];
                c[idx] = Complex32::new(
                    alpha.re * prod_re - alpha.im * prod_im + beta.re * old.re - beta.im * old.im,
                    alpha.re * prod_im + alpha.im * prod_re + beta.re * old.im + beta.im * old.re,
                );
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BEM regular-pair Helmholtz kernel (Phase 1.3 of bem-step1.md)
// ---------------------------------------------------------------------------

/// Constant physics parameters shared by every pair in a batch.
/// `harmonic_factor` is `±1.0` selecting the sign convention for the
/// imaginary part (matches `PhysicsParams::harmonic_factor` on the host).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct BemPhysicsParamsGpu {
    k: f32,
    harmonic_factor: f32,
    n_pairs: u32,
    _pad: u32,
}

impl MetalBackend {
    /// Evaluate the Helmholtz `(G, H, H^T, E)` kernel integrals for a
    /// batch of regular (well-separated) source-field triangle pairs on
    /// the GPU using fixed 4-point Gauss-tri quadrature, no adaptive
    /// subdivision. One thread per pair; all f32 arithmetic.
    ///
    /// Caller is expected to pre-screen pairs with
    /// `regular::optimal_quadrature_order` and route only the
    /// `ratio ≥ 3.0` (4-point regime) cases through this kernel; closer
    /// pairs need higher quadrature order or adaptive subdivision and
    /// stay on the existing CPU path.
    ///
    /// Arguments:
    /// - `pairs`: flat batch of pair descriptors. Empty input returns
    ///   an empty `Vec` without launching the kernel.
    /// - `wave_number`: Helmholtz `k` in rad/m.
    /// - `harmonic_factor`: `+1.0` or `-1.0`, sign of the imaginary part
    ///   (matches `PhysicsParams::harmonic_factor`).
    ///
    /// Returns the per-pair `(G, H, H^T, E)` integrals in input order.
    /// Block-relative error vs the f64 CPU `regular_integration`
    /// implementation is bounded by f32 ULP × condition number; for
    /// `kr ≤ 50` empirically inside `1e-6`. The host-side integration
    /// shim asserts this in `tests/parity.rs`.
    /// Checked Metal BEM regular-pair dispatch. Native failures remain typed
    /// instead of being collapsed into an empty successful result.
    pub fn try_bem_regular_pair_c32_batch(
        &self,
        pairs: &[BemRegularPairIn],
        wave_number: f32,
        harmonic_factor: f32,
    ) -> BackendResult<Vec<BemRegularPairOut>> {
        self.check_pending_error()?;
        validate_bem_regular_pair_inputs(pairs, wave_number, harmonic_factor)?;
        let n = pairs.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let n_pairs = u32::try_from(n).map_err(|_| BackendError::InvalidArgument {
            operation: "bem_regular_pair_c32_batch",
            reason: "pair count exceeds Metal kernel index range".into(),
        })?;
        let in_bytes = u64::try_from(std::mem::size_of_val(pairs)).map_err(|_| {
            BackendError::InvalidArgument {
                operation: "bem_regular_pair_c32_batch",
                reason: "input byte count exceeds Metal address range".into(),
            }
        })?;
        let out_bytes = n
            .checked_mul(std::mem::size_of::<BemRegularPairOut>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "bem_regular_pair_c32_batch",
                reason: "output byte count overflow".into(),
            })?;

        let in_buf = self.take_scratch_checked(in_bytes)?;
        let out_buf = match self.take_scratch_checked(out_bytes) {
            Ok(buffer) => buffer,
            Err(error) => {
                drop(in_buf);
                return Err(error);
            }
        };

        let result = (|| -> BackendResult<Vec<BemRegularPairOut>> {
            // SAFETY: the checked scratch allocation is at least `in_bytes`
            // bytes and `pairs` is a live input slice.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    pairs.as_ptr() as *const u8,
                    in_buf.contents().as_ptr() as *mut u8,
                    usize::try_from(in_bytes).map_err(|_| BackendError::InvalidArgument {
                        operation: "bem_regular_pair_c32_batch",
                        reason: "input byte count does not fit host address range".into(),
                    })?,
                );
            }

            let params = BemPhysicsParamsGpu {
                k: wave_number,
                harmonic_factor,
                n_pairs,
                _pad: 0,
            };
            let pso = self.try_pso(kernel_names::BEM_REGULAR_PAIR_C32)?;
            let cmd_buf = self.try_pending_command_buffer()?;
            let encoder = self.try_compute_encoder(&cmd_buf)?;
            encoder.setComputePipelineState(pso);
            set_buffer(&encoder, 0, &in_buf);
            set_buffer(&encoder, 1, &out_buf);
            set_scalar(&encoder, 2, &params);
            let threadgroup = THREADGROUP_SIZE;
            let groups = (n as u64).div_ceil(threadgroup);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threadgroup as usize,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            self.flush_pending_checked()?;

            // SAFETY: completion above waited for all writes to `out_buf`.
            let out_slice = unsafe {
                std::slice::from_raw_parts(
                    out_buf.contents().as_ptr() as *const BemRegularPairOut,
                    n,
                )
            };
            let output = out_slice.to_vec();
            validate_bem_regular_pair_outputs(&output)?;
            Ok(output)
        })();

        match result {
            Ok(output) => {
                self.return_scratch_checked(in_bytes, in_buf)?;
                self.return_scratch_checked(out_bytes, out_buf)?;
                Ok(output)
            }
            Err(error) => {
                drop(in_buf);
                drop(out_buf);
                Err(error)
            }
        }
    }

    /// Compatibility wrapper for callers that still use the infallible API.
    /// Checked backend callers must use `try_bem_regular_pair_c32_batch`.
    pub fn bem_regular_pair_c32_batch(
        &self,
        pairs: &[BemRegularPairIn],
        wave_number: f32,
        harmonic_factor: f32,
    ) -> Vec<BemRegularPairOut> {
        let n = pairs.len();
        if n == 0 {
            return Vec::new();
        }
        // Allocate the input/output buffers in unified memory. We do NOT
        // flush the pending command buffer here — the BEM kernel touches
        // its own private buffers, so it composes safely with any other
        // queued BLAS-1 / SpMV work. The single synchronization point is
        // the `to_vec()` host readback at the end, which goes through
        // `flush_pending()`.
        let in_bytes = std::mem::size_of_val(pairs) as u64;
        let out_bytes = (n * std::mem::size_of::<BemRegularPairOut>()) as u64;

        // Frequency sweeps call this in a tight loop with the same pair
        // count and the same struct sizes — both input and output buffers
        // can come from the recycle pool keyed on byte length.
        // `take_scratch` allocates on first miss and reuses thereafter.
        let Some(in_buf) = self.take_scratch(in_bytes) else {
            return Vec::new();
        };
        let Some(out_buf) = self.take_scratch(out_bytes) else {
            // The input scratch buffer is not referenced by a command buffer
            // yet, so it is safe to return it immediately when the paired
            // output allocation fails.
            self.return_scratch(in_bytes, in_buf);
            return Vec::new();
        };
        // Shared-storage buffer is CPU-mappable; copy the input bytes in
        // (the previous `newBufferWithBytes` did this implicitly).
        // SAFETY: `pairs` is a live slice and `in_buf` is at least
        // `in_bytes` long; both pointers are non-null and well-aligned.
        unsafe {
            std::ptr::copy_nonoverlapping(
                pairs.as_ptr() as *const u8,
                in_buf.contents().as_ptr() as *mut u8,
                in_bytes as usize,
            );
        }
        let params = BemPhysicsParamsGpu {
            k: wave_number,
            harmonic_factor,
            n_pairs: n as u32,
            _pad: 0,
        };

        let Some(pso) = self.pso(kernel_names::BEM_REGULAR_PAIR_C32) else {
            self.return_scratch(in_bytes, in_buf);
            self.return_scratch(out_bytes, out_buf);
            return Vec::new();
        };
        {
            let Some(cmd_buf) = self.pending_command_buffer() else {
                self.return_scratch(in_bytes, in_buf);
                self.return_scratch(out_bytes, out_buf);
                return Vec::new();
            };
            let Some(encoder) = self.compute_encoder(&cmd_buf) else {
                self.return_scratch(in_bytes, in_buf);
                self.return_scratch(out_bytes, out_buf);
                return Vec::new();
            };
            encoder.setComputePipelineState(pso);
            set_buffer(&encoder, 0, &in_buf);
            set_buffer(&encoder, 1, &out_buf);
            set_scalar(&encoder, 2, &params);
            let threadgroup = THREADGROUP_SIZE;
            let groups = (n as u64).div_ceil(threadgroup);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (groups) as usize,
                    height: 1_usize,
                    depth: 1_usize,
                },
                MTLSize {
                    width: (threadgroup) as usize,
                    height: 1_usize,
                    depth: 1_usize,
                },
            );
            encoder.endEncoding();
        }

        // Host readback is the flush point — drain the queue (this kernel
        // plus any other still-pending BLAS work).
        self.flush_pending();

        // Shared-storage buffer is CPU-mappable.
        let out_slice = unsafe {
            std::slice::from_raw_parts(out_buf.contents().as_ptr() as *const BemRegularPairOut, n)
        };
        let result = out_slice.to_vec();

        // Return both buffers to the pool. Their byte sizes are stable
        // across same-batch-size calls so the next sweep iteration
        // immediately reuses them.
        self.return_scratch(in_bytes, in_buf);
        self.return_scratch(out_bytes, out_buf);
        result
    }
}

// ---------------------------------------------------------------------------
// Host-staged f64 fallback (Apple Silicon Metal compiler rejects `double`)
// ---------------------------------------------------------------------------

/// Generic host-side compute over a shared-storage Metal buffer.
/// Type-erased through `T: DeviceScalar` so the same code services
/// f64 / Complex32 / Complex64.
fn axpy_host<T: DeviceScalar>(alpha: T, x: &[T], y: &mut [T]) {
    for (xi, yi) in x.iter().zip(y.iter_mut()) {
        *yi = *yi + alpha * *xi;
    }
}

fn scale_host<T: DeviceScalar>(alpha: T, x: &mut [T]) {
    for xi in x.iter_mut() {
        *xi = alpha * *xi;
    }
}

fn dot_host<T: DeviceScalar>(x: &[T], y: &[T]) -> T {
    let mut acc = T::zero();
    for (xi, yi) in x.iter().zip(y.iter()) {
        acc = acc + xi.conj_mul(*yi);
    }
    acc
}

fn nrm2_host<T: DeviceScalar>(x: &[T]) -> T::Real {
    crate::scalar::stable_nrm2(x)
}

fn spmv_csr_host<T: DeviceScalar>(
    m: usize,
    row_ptr: &[i32],
    col_idx: &[i32],
    values: &[T],
    x: &[T],
    y: &mut [T],
) {
    for i in 0..m {
        let lo = row_ptr[i] as usize;
        let hi = row_ptr[i + 1] as usize;
        let mut acc = T::zero();
        for k in lo..hi {
            acc = acc + values[k] * x[col_idx[k] as usize];
        }
        y[i] = acc;
    }
}

#[allow(clippy::too_many_arguments)]
fn spmm_csr_host_strided<T: DeviceScalar>(
    m: usize,
    n: usize,
    rhs: usize,
    row_ptr: &[i32],
    col_idx: &[i32],
    values: &[T],
    x: &[T],
    ldx: usize,
    y: &mut [T],
    ldy: usize,
) {
    for col in 0..rhs {
        let x_col = &x[col * ldx..col * ldx + n];
        let y_col = &mut y[col * ldy..col * ldy + m];
        spmv_csr_host(m, row_ptr, col_idx, values, x_col, y_col);
    }
}

// ---------------------------------------------------------------------------
// Backend trait impl — generic over T, branches on TypeId for f32 fast path
// ---------------------------------------------------------------------------

/// Helper: cast `&MetalBuffer<T>` to `&MetalBuffer<f32>` after a TypeId
/// match. SAFETY: caller must have verified `TypeId::of::<T>() ==
/// TypeId::of::<f32>()`.
unsafe fn as_f32_buffer<T: BufferElement>(buf: &MetalBuffer<T>) -> &MetalBuffer<f32> {
    unsafe { &*(buf as *const MetalBuffer<T> as *const MetalBuffer<f32>) }
}

unsafe fn as_f32_buffer_mut<T: BufferElement>(buf: &mut MetalBuffer<T>) -> &mut MetalBuffer<f32> {
    unsafe { &mut *(buf as *mut MetalBuffer<T> as *mut MetalBuffer<f32>) }
}

/// SAFETY: caller must have verified `TypeId::of::<T>() == TypeId::of::<Complex32>()`.
/// `Complex32` is `#[repr(C)] { re: f32, im: f32 }`, identical 8-byte layout
/// across the wrapper, so the buffer reinterpretation is well-defined.
unsafe fn as_c32_buffer<T: BufferElement>(buf: &MetalBuffer<T>) -> &MetalBuffer<Complex32> {
    unsafe { &*(buf as *const MetalBuffer<T> as *const MetalBuffer<Complex32>) }
}

unsafe fn as_c32_buffer_mut<T: BufferElement>(
    buf: &mut MetalBuffer<T>,
) -> &mut MetalBuffer<Complex32> {
    unsafe { &mut *(buf as *mut MetalBuffer<T> as *mut MetalBuffer<Complex32>) }
}

impl MetalBackend {
    fn spmm_csr_f32_native(
        &self,
        prebaked: &MetalCsrPrebaked<'_, f32>,
        nrhs: u32,
        x: &MetalBuffer<f32>,
        ldx: u32,
        y: &mut MetalBuffer<f32>,
        ldy: u32,
    ) {
        let rows = prebaked.m as u32;
        let total = rows * nrhs;
        let Some(pso) = self.pso(kernel_names::SPMM_CSR_F32) else {
            return;
        };
        self.dispatch_1d_lazy(pso, total, THREADGROUP_SIZE, |encoder| {
            set_buffer(encoder, 0, &prebaked.row_ptr.raw);
            set_buffer(encoder, 1, &prebaked.col_idx.raw);
            set_buffer(encoder, 2, &prebaked.values.raw);
            set_buffer(encoder, 3, &x.raw);
            set_buffer(encoder, 4, &y.raw);
            set_scalar(encoder, 5, &rows);
            set_scalar(encoder, 6, &nrhs);
            set_scalar(encoder, 7, &ldx);
            set_scalar(encoder, 8, &ldy);
        });
    }

    fn spmm_csr_c32_native(
        &self,
        prebaked: &MetalCsrPrebaked<'_, Complex32>,
        nrhs: u32,
        x: &MetalBuffer<Complex32>,
        ldx: u32,
        y: &mut MetalBuffer<Complex32>,
        ldy: u32,
    ) {
        let rows = prebaked.m as u32;
        let total = rows * nrhs;
        let Some(pso) = self.pso(kernel_names::SPMM_CSR_C32) else {
            return;
        };
        self.dispatch_1d_lazy(pso, total, THREADGROUP_SIZE, |encoder| {
            set_buffer(encoder, 0, &prebaked.row_ptr.raw);
            set_buffer(encoder, 1, &prebaked.col_idx.raw);
            set_buffer(encoder, 2, &prebaked.values.raw);
            set_buffer(encoder, 3, &x.raw);
            set_buffer(encoder, 4, &y.raw);
            set_scalar(encoder, 5, &rows);
            set_scalar(encoder, 6, &nrhs);
            set_scalar(encoder, 7, &ldx);
            set_scalar(encoder, 8, &ldy);
        });
    }
}

impl Backend for MetalBackend {
    type Buffer<T: BufferElement> = MetalBuffer<T>;
    type CsrPrebaked<'a, T: DeviceScalar>
        = MetalCsrPrebaked<'a, T>
    where
        Self: 'a;

    fn check_pending_error(&self) -> BackendResult<()> {
        MetalBackend::check_pending_error(self)
    }

    fn validate_segmented_backend_limits<T: DeviceScalar>(
        &self,
        operation: &'static str,
        values_len: usize,
        offsets_len: usize,
        output_len: usize,
    ) -> BackendResult<()> {
        Self::check_segmented_backend_limits::<T>(operation, values_len, offsets_len, output_len)
    }

    fn try_segmented_sum<T>(
        &self,
        values: &Self::Buffer<T>,
        offsets: &Self::Buffer<i32>,
        output: &mut Self::Buffer<T>,
    ) -> BackendResult<()>
    where
        T: DeviceScalar,
    {
        let (mut host_offsets, _offsets_staging_guard) =
            try_host_staging_vec(&self.staging_stats, offsets.len, "segmented host offsets")?;
        self.try_copy_d2h(offsets, &mut host_offsets)?;
        validate_segmented_offsets(values.len, &host_offsets, output.len).map_err(|error| {
            BackendError::InvalidArgument {
                operation: "segmented_sum",
                reason: error.to_string(),
            }
        })?;
        self.validate_segmented_backend_limits::<T>(
            "segmented_sum",
            values.len,
            offsets.len,
            output.len,
        )?;
        if output.is_empty() {
            return Ok(());
        }

        let tid = TypeId::of::<T>();
        if tid == TypeId::of::<f32>() {
            let values = unsafe { as_f32_buffer(values) };
            let output = unsafe { as_f32_buffer_mut(output) };
            self.segmented_sum_f32_native(values, offsets, output);
            return Ok(());
        }
        if tid == TypeId::of::<Complex32>() {
            let values = unsafe { as_c32_buffer(values) };
            let output = unsafe { as_c32_buffer_mut(output) };
            self.segmented_sum_c32_native(values, offsets, output);
            return Ok(());
        }

        let (mut host_values, _values_staging_guard) =
            try_host_staging_vec(&self.staging_stats, values.len, "segmented host values")?;
        let (mut host_output, _output_staging_guard) =
            try_host_staging_vec(&self.staging_stats, output.len, "segmented host output")?;
        self.try_copy_d2h(values, &mut host_values)?;
        segmented_sum(&host_values, &host_offsets, &mut host_output).map_err(|error| {
            BackendError::InvalidArgument {
                operation: "segmented_sum",
                reason: error.to_string(),
            }
        })?;
        self.try_copy_h2d(&host_output, output)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn try_batched_local_contract<T>(
        &self,
        lhs: &Self::Buffer<T>,
        rhs: &Self::Buffer<T>,
        output: &mut Self::Buffer<T>,
        batches: usize,
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> BackendResult<()>
    where
        T: DeviceScalar,
    {
        let lhs_len = batches
            .checked_mul(rows)
            .and_then(|value| value.checked_mul(inner))
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "batched_local_contract",
                reason: "lhs shape overflows usize".into(),
            })?;
        let rhs_len = batches
            .checked_mul(inner)
            .and_then(|value| value.checked_mul(cols))
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "batched_local_contract",
                reason: "rhs shape overflows usize".into(),
            })?;
        let output_len = batches
            .checked_mul(rows)
            .and_then(|value| value.checked_mul(cols))
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "batched_local_contract",
                reason: "output shape overflows usize".into(),
            })?;
        for (name, actual, expected) in [
            ("lhs", lhs.len, lhs_len),
            ("rhs", rhs.len, rhs_len),
            ("output", output.len, output_len),
        ] {
            if actual != expected {
                return Err(BackendError::InvalidArgument {
                    operation: "batched_local_contract",
                    reason: format!("{name} length {actual}, expected {expected}"),
                });
            }
        }
        if output_len == 0 {
            return Ok(());
        }

        let batches_u32 = u32::try_from(batches).map_err(|_| BackendError::InvalidArgument {
            operation: "batched_local_contract",
            reason: "batch count exceeds Metal kernel index range".into(),
        })?;
        let rows_u32 = u32::try_from(rows).map_err(|_| BackendError::InvalidArgument {
            operation: "batched_local_contract",
            reason: "row count exceeds Metal kernel index range".into(),
        })?;
        let inner_u32 = u32::try_from(inner).map_err(|_| BackendError::InvalidArgument {
            operation: "batched_local_contract",
            reason: "inner dimension exceeds Metal kernel index range".into(),
        })?;
        let cols_u32 = u32::try_from(cols).map_err(|_| BackendError::InvalidArgument {
            operation: "batched_local_contract",
            reason: "column count exceeds Metal kernel index range".into(),
        })?;
        let tid = TypeId::of::<T>();
        if tid == TypeId::of::<f32>() {
            let lhs = unsafe { as_f32_buffer(lhs) };
            let rhs = unsafe { as_f32_buffer(rhs) };
            let output = unsafe { as_f32_buffer_mut(output) };
            self.batched_local_contract_f32_native(
                lhs,
                rhs,
                output,
                batches_u32,
                rows_u32,
                inner_u32,
                cols_u32,
            );
            return Ok(());
        }
        if tid == TypeId::of::<Complex32>() {
            let lhs = unsafe { as_c32_buffer(lhs) };
            let rhs = unsafe { as_c32_buffer(rhs) };
            let output = unsafe { as_c32_buffer_mut(output) };
            self.batched_local_contract_c32_native(
                lhs,
                rhs,
                output,
                batches_u32,
                rows_u32,
                inner_u32,
                cols_u32,
            );
            return Ok(());
        }

        let (mut host_lhs, _lhs_staging_guard) =
            try_host_staging_vec(&self.staging_stats, lhs.len, "local contract host lhs")?;
        let (mut host_rhs, _rhs_staging_guard) =
            try_host_staging_vec(&self.staging_stats, rhs.len, "local contract host rhs")?;
        let (mut host_output, _output_staging_guard) = try_host_staging_vec(
            &self.staging_stats,
            output.len,
            "local contract host output",
        )?;
        self.try_copy_d2h(lhs, &mut host_lhs)?;
        self.try_copy_d2h(rhs, &mut host_rhs)?;
        batched_local_contract(
            &host_lhs,
            &host_rhs,
            &mut host_output,
            batches,
            rows,
            inner,
            cols,
        )
        .map_err(|error| BackendError::InvalidArgument {
            operation: "batched_local_contract",
            reason: error.to_string(),
        })?;
        self.try_copy_h2d(&host_output, output)?;
        Ok(())
    }

    fn try_fft_in_place<T: BufferElement>(
        &self,
        buffer: &mut Self::Buffer<T>,
        fft_len: usize,
        inverse: bool,
    ) -> BackendResult<()> {
        if TypeId::of::<T>() != TypeId::of::<Complex32>() {
            return Err(BackendError::Unsupported {
                backend: "metal",
                operation: "fft",
                scalar: std::any::type_name::<T>(),
            });
        }

        // SAFETY: TypeId equality proves T is Complex32. MetalBuffer<T> has
        // only the raw buffer, length, and zero-sized PhantomData<T>, so the
        // pointer cast preserves representation and ownership exactly.
        let buffer_c32 =
            unsafe { &mut *(buffer as *mut MetalBuffer<T> as *mut MetalBuffer<Complex32>) };
        self.fft_stockham_c32(buffer_c32, fft_len, inverse)
    }

    fn capabilities(&self) -> BackendCapabilities {
        let native_f32_c32 = Self::native_f32_c32_capabilities();

        BackendCapabilities {
            backend_name: "metal",
            blas1: native_f32_c32,
            gemv: native_f32_c32,
            fft: Self::fft_capabilities(),
            real_fft: ScalarCapabilities::splat(CapabilityState::Unsupported),
            dct: ScalarCapabilities::splat(CapabilityState::Unsupported),
            dst: ScalarCapabilities::splat(CapabilityState::Unsupported),
            // GEMM dispatch is shape-dependent: small worksets use the CPU
            // shared-memory path while larger f32/c32 worksets use MPS.
            // Advertise the conservative state until capabilities become
            // shape-aware; callers must not infer vendor execution here.
            gemm: Self::gemm_capabilities(),
            trsm: ScalarCapabilities::splat(CapabilityState::HostStagedFallback),
            syrk: ScalarCapabilities::splat(CapabilityState::HostStagedFallback),
            csr_spmv: native_f32_c32,
            csr_spmm: native_f32_c32,
            csr_transpose: ScalarCapabilities::splat(CapabilityState::HostStagedFallback),
            axpy_many: native_f32_c32,
            multi_dot: ScalarCapabilities::new(
                CapabilityState::Native,
                CapabilityState::HostSharedFallback,
                CapabilityState::Native,
                CapabilityState::HostSharedFallback,
            ),
            multi_dot_strided: ScalarCapabilities::new(
                CapabilityState::Native,
                CapabilityState::HostSharedFallback,
                CapabilityState::Native,
                CapabilityState::HostSharedFallback,
            ),
            basis_axpy_strided: ScalarCapabilities::new(
                CapabilityState::Native,
                CapabilityState::HostSharedFallback,
                CapabilityState::Native,
                CapabilityState::HostSharedFallback,
            ),
            basis_project_out_strided: ScalarCapabilities::splat(
                CapabilityState::HostStagedFallback,
            ),
            gather4_weighted: ScalarCapabilities::new(
                CapabilityState::Native,
                CapabilityState::HostStagedFallback,
                CapabilityState::Native,
                CapabilityState::HostStagedFallback,
            ),
            segmented_reduce: ScalarCapabilities::new(
                CapabilityState::Native,
                CapabilityState::HostSharedFallback,
                CapabilityState::Native,
                CapabilityState::HostSharedFallback,
            ),
            batched_local_contract: ScalarCapabilities::new(
                CapabilityState::Native,
                CapabilityState::HostSharedFallback,
                CapabilityState::Native,
                CapabilityState::HostSharedFallback,
            ),
            reduction: ReductionCapabilities::new(
                native_f32_c32,
                ScalarCapabilities::splat(CapabilityState::HostSharedFallback),
            ),
            precision_cast: PrecisionCastCapabilities::splat(CapabilityState::HostSharedFallback),
            bem_regular_pair_c32: CapabilityState::Native,
            // Metal's shader compiler rejects double precision on the
            // supported Apple-Silicon devices. Keep this cell explicit so a
            // BEM caller cannot label host f64 work as device execution.
            bem_regular_pair_c64: CapabilityState::Unsupported,
            bem: BemCapabilities {
                regular_pair_c32: CapabilityState::Native,
                regular_pair_c64: CapabilityState::Unsupported,
                near_pair: ScalarCapabilities::new(
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::HostStagedFallback,
                ),
                singular_pair: ScalarCapabilities::new(
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::HostStagedFallback,
                ),
                hypersingular_pair: ScalarCapabilities::new(
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::HostStagedFallback,
                ),
                field_evaluation: ScalarCapabilities::splat(CapabilityState::Unsupported),
                hmatrix_setup: ScalarCapabilities::new(
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::HostSharedFallback,
                ),
                hmatrix: ScalarCapabilities::splat(CapabilityState::Unsupported),
                fmm_setup: ScalarCapabilities::new(
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::HostSharedFallback,
                ),
                fmm: ScalarCapabilities::splat(CapabilityState::Unsupported),
            },
        }
    }

    fn bem_regular_pair_c32_batch(
        &self,
        pairs: &[BemRegularPairIn],
        wave_number: f32,
        harmonic_factor: f32,
    ) -> Vec<BemRegularPairOut> {
        MetalBackend::bem_regular_pair_c32_batch(self, pairs, wave_number, harmonic_factor)
    }

    fn try_bem_regular_pair_c32_batch(
        &self,
        pairs: &[BemRegularPairIn],
        wave_number: f32,
        harmonic_factor: f32,
    ) -> BackendResult<Vec<BemRegularPairOut>> {
        MetalBackend::try_bem_regular_pair_c32_batch(self, pairs, wave_number, harmonic_factor)
    }

    fn transfer_stats(&self) -> TransferStatsSnapshot {
        self.transfer_stats.snapshot()
    }

    fn memory_stats(&self) -> BackendMemoryStatsSnapshot {
        BackendMemoryStatsSnapshot {
            live_allocation_bytes: self.live_allocation_bytes(),
            allocation_high_water_bytes: self.allocation_high_water_bytes(),
            live_staging_bytes: self.live_staging_bytes(),
            staging_high_water_bytes: self.staging_high_water_bytes(),
        }
    }

    fn reset_transfer_stats(&self) {
        self.transfer_stats.reset();
    }

    fn raw_alloc<T: BufferElement>(&self, len: usize) -> Result<Self::Buffer<T>, OomError> {
        // Metal rejects a zero-length `MTLBuffer`, so use a one-byte native
        // sentinel for an empty logical buffer.  The sentinel is not caller-
        // visible storage and must not consume quota: charging it would make
        // `try_alloc(0)` fail under a zero-byte cap and would make the backend
        // ledger disagree with the buffer's logical byte size.
        let logical_bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| OomError::Invalid("metal alloc size overflow".into()))?;
        let bytes = logical_bytes.max(1) as u64;
        let allocation = self.allocation_stats.try_reserve(logical_bytes)?;
        let raw = self
            .device
            .newBufferWithLength_options((bytes) as usize, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| OomError::DriverOom("Metal: newBuffer returned nil".into()))?;
        Ok(MetalBuffer {
            raw: ThreadSafe(MetalRawBuf::from_guard(raw, allocation)),
            len,
            _phantom: PhantomData,
        })
    }

    unsafe fn raw_alloc_uninitialized<T: BufferElement>(
        &self,
        len: usize,
    ) -> Result<Self::Buffer<T>, OomError> {
        if len == 0 {
            return self.raw_alloc::<T>(len);
        }
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| OomError::Invalid("metal alloc size overflow".into()))?
            as u64;
        let allocation = self.allocation_stats.try_reserve(bytes as usize)?;
        let raw = self
            .device
            .newBufferWithLength_options((bytes) as usize, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| OomError::DriverOom("Metal: newBuffer returned nil".into()))?;
        Ok(MetalBuffer {
            raw: ThreadSafe(MetalRawBuf::from_guard(raw, allocation)),
            len,
            _phantom: PhantomData,
        })
    }

    fn set_allocation_cap_bytes(&self, cap_bytes: usize) {
        self.allocation_stats.set_cap_bytes(cap_bytes);
    }

    fn total_device_memory_bytes(&self) -> Option<usize> {
        // Apple Silicon has unified memory; no dedicated VRAM cap.
        // Force callers to set an explicit pool cap.
        None
    }

    fn copy_h2d<T: BufferElement>(&self, host: &[T], dev: &mut Self::Buffer<T>) {
        if host.len() != dev.len {
            let error = self.record_invalid_argument(
                "copy_h2d",
                format!("host length {} != device length {}", host.len(), dev.len),
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        }
        if dev.len == 0 {
            return;
        }
        // Flush any queued GPU writes that may target this buffer
        // before we overwrite from the host.
        self.flush_pending();
        dev.host_slice_mut().copy_from_slice(host);
        self.transfer_stats.record_h2d(std::mem::size_of_val(host));
    }

    fn copy_d2h<T: BufferElement>(&self, dev: &Self::Buffer<T>, host: &mut [T]) {
        if host.len() != dev.len {
            let error = self.record_invalid_argument(
                "copy_d2h",
                format!("host length {} != device length {}", host.len(), dev.len),
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        }
        if dev.len == 0 {
            return;
        }
        // Need GPU writes settled before we read.
        self.flush_pending();
        host.copy_from_slice(dev.host_slice());
        self.transfer_stats.record_d2h(std::mem::size_of_val(host));
    }

    fn try_copy_d2h<T: BufferElement>(
        &self,
        dev: &Self::Buffer<T>,
        host: &mut [T],
    ) -> BackendResult<()> {
        if host.len() != dev.len {
            return Err(BackendError::InvalidArgument {
                operation: "copy_d2h",
                reason: format!("host length {} != device length {}", host.len(), dev.len),
            });
        }
        // Do not read from a failed command buffer.  The infallible
        // compatibility method may only log such a failure; the checked
        // transfer must preserve it as a typed error before touching host
        // memory.
        self.flush_pending_checked()?;
        if dev.len != 0 {
            host.copy_from_slice(dev.host_slice());
            self.transfer_stats.record_d2h(std::mem::size_of_val(host));
        }
        Ok(())
    }

    fn try_copy_views_d2d<T: BufferElement>(
        &self,
        source: &Self::Buffer<T>,
        source_view: BufferView,
        destination: &mut Self::Buffer<T>,
        destination_view: BufferView,
    ) -> BackendResult<()> {
        self.check_pending_error()?;
        self.checked_view_descriptor(
            source,
            source_view.offset,
            source_view.len,
            source_view.stride,
        )?;
        self.checked_view_descriptor_with_aliasing(
            destination,
            destination_view.offset,
            destination_view.len,
            destination_view.stride,
            ViewAliasing::OverlapSafe,
        )?;
        if source_view.len != destination_view.len {
            return Err(BackendError::InvalidArgument {
                operation: "copy_views_d2d",
                reason: format!(
                    "source view length {} != destination view length {}",
                    source_view.len, destination_view.len
                ),
            });
        }
        if source_view.len == 0 {
            return Ok(());
        }
        let element_size = std::mem::size_of::<T>();
        let Some(command_buffer) = self.pending_command_buffer() else {
            return Err(BackendError::OperationFailed {
                operation: "copy_views_d2d",
                reason: "Metal command buffer unavailable".into(),
            });
        };
        let Some(encoder) = self.blit_encoder(&command_buffer) else {
            return Err(BackendError::OperationFailed {
                operation: "copy_views_d2d",
                reason: "Metal blit encoder unavailable".into(),
            });
        };
        for index in 0..source_view.len {
            let source_byte = source_view
                .checked_element_index(index)?
                .checked_mul(element_size)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "copy_views_d2d",
                    reason: "source byte offset overflow".into(),
                })?;
            let destination_byte = destination_view
                .checked_element_index(index)?
                .checked_mul(element_size)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "copy_views_d2d",
                    reason: "destination byte offset overflow".into(),
                })?;
            // SAFETY: both views were checked against their allocation lengths;
            // Metal's blit encoder only receives byte ranges within those
            // shared-storage buffers.
            unsafe {
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &source.raw,
                    source_byte,
                    &destination.raw,
                    destination_byte,
                    element_size,
                );
            }
        }
        encoder.endEncoding();
        self.transfer_stats
            .record_d2d(source_view.len.saturating_mul(element_size));
        Ok(())
    }

    fn try_copy_nd_views_d2d<T: BufferElement>(
        &self,
        source: &Self::Buffer<T>,
        source_descriptor: &ViewDescriptor,
        destination: &mut Self::Buffer<T>,
        destination_descriptor: &ViewDescriptor,
    ) -> BackendResult<()> {
        self.check_pending_error()?;
        let device = self.capabilities().backend_name;
        source_descriptor.validate_for_buffer::<T>(source.len, device)?;
        destination_descriptor.validate_for_buffer::<T>(destination.len, device)?;
        if destination_descriptor.aliasing == ViewAliasing::ReadOnly {
            return Err(BackendError::InvalidArgument {
                operation: "copy_nd_views_d2d",
                reason: "destination descriptor is read-only".into(),
            });
        }
        let source_count = source_descriptor.checked_element_count()?;
        let destination_count = destination_descriptor.checked_element_count()?;
        if source_count != destination_count {
            return Err(BackendError::InvalidArgument {
                operation: "copy_nd_views_d2d",
                reason: format!(
                    "source descriptor count {source_count} != destination descriptor count {destination_count}"
                ),
            });
        }
        if source_count == 0 {
            return Ok(());
        }
        if destination_descriptor.aliasing == ViewAliasing::OverlapSafe {
            // Preserve memmove-like semantics for explicitly overlapping
            // views; the direct blit path is reserved for disjoint buffers.
            let (mut staged, _staging_guard) = try_host_staging_vec(
                &self.staging_stats,
                source_count,
                "copy_nd_views host overlap staging",
            )?;
            self.try_copy_nd_view_d2h(source, source_descriptor, &mut staged)?;
            return self.try_copy_nd_view_h2d(&staged, destination, destination_descriptor);
        }

        let Some(command_buffer) = self.pending_command_buffer() else {
            return Err(BackendError::OperationFailed {
                operation: "copy_nd_views_d2d",
                reason: "Metal command buffer unavailable".into(),
            });
        };
        let Some(encoder) = self.blit_encoder(&command_buffer) else {
            return Err(BackendError::OperationFailed {
                operation: "copy_nd_views_d2d",
                reason: "Metal blit encoder unavailable".into(),
            });
        };
        let element_size = std::mem::size_of::<T>();
        for index in 0..source_count {
            let source_byte = source_descriptor
                .checked_element_index::<T>(index, device)?
                .checked_mul(element_size)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "copy_nd_views_d2d",
                    reason: "source byte offset overflow".into(),
                })?;
            let destination_byte = destination_descriptor
                .checked_element_index::<T>(index, device)?
                .checked_mul(element_size)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "copy_nd_views_d2d",
                    reason: "destination byte offset overflow".into(),
                })?;
            // SAFETY: descriptor validation proves each byte range lies
            // inside its shared-storage allocation and is element-aligned.
            unsafe {
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &source.raw,
                    source_byte,
                    &destination.raw,
                    destination_byte,
                    element_size,
                );
            }
        }
        encoder.endEncoding();
        let bytes = source_count.checked_mul(element_size).ok_or_else(|| {
            BackendError::InvalidArgument {
                operation: "copy_nd_views_d2d",
                reason: "transfer byte count overflow".into(),
            }
        })?;
        self.transfer_stats.record_d2d(bytes);
        Ok(())
    }

    fn copy_d2d<T: BufferElement>(&self, src: &Self::Buffer<T>, dst: &mut Self::Buffer<T>) {
        if src.len != dst.len {
            let error = self.record_invalid_argument(
                "copy_d2d",
                format!(
                    "source length {} != destination length {}",
                    src.len, dst.len
                ),
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        }
        if src.len == 0 {
            return;
        }
        let bytes = (src.len * std::mem::size_of::<T>()) as u64;
        let Some(cmd_buf) = self.pending_command_buffer() else {
            return;
        };
        let Some(encoder) = self.blit_encoder(&cmd_buf) else {
            return;
        };
        unsafe {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &src.raw,
                0_usize,
                &dst.raw,
                0_usize,
                (bytes) as usize,
            );
        }
        encoder.endEncoding();
        self.transfer_stats.record_d2d(bytes as usize);
    }

    fn copy_h2d_cast<Upper>(&self, host: &[Upper], dev: &mut Self::Buffer<Upper::Lower>)
    where
        Upper: LowerPrecision,
    {
        if host.len() != dev.len {
            let error = self.record_invalid_argument(
                "copy_h2d_cast",
                format!("host length {} != device length {}", host.len(), dev.len),
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        }
        if dev.len == 0 {
            return;
        }
        self.flush_pending();
        // Cast on CPU over the shared-storage buffer (unified memory makes
        // dev's bytes directly host-visible). Apple Metal compiler rejects
        // `double`, so a GPU narrowing cast (f64 → f32) would need an MSL
        // kernel that treats f64 buffers as `uint2` and reassembles the
        // narrowed value with manual IEEE-754 bit-twiddling — extracting
        // sign + 11-bit exp + 52-bit mantissa from uint2, re-biasing the
        // exponent (1023 → 127), shifting the mantissa with
        // round-to-nearest-even, and handling NaN / Inf / subnormal /
        // underflow / overflow paths. ~100 lines of error-prone MSL for
        // 50-100 µs per cast at large n — bounded win, deferred behind
        // task #26. The widening direction (f32 → f64) is much simpler
        // but only meaningful if we also move the narrowing side.
        //
        // Today's rayon path is bandwidth-bound and ~4× over serial on M4.
        use rayon::prelude::*;
        let dst = dev.host_slice_mut();
        const CHUNK: usize = 4096;
        host.par_chunks(CHUNK)
            .zip(dst.par_chunks_mut(CHUNK))
            .for_each(|(src_chunk, dst_chunk)| {
                for (s, d) in src_chunk.iter().zip(dst_chunk.iter_mut()) {
                    *d = Upper::cast_to_lower(*s);
                }
            });
        self.transfer_stats.record_h2d(std::mem::size_of_val(host));
    }

    fn cast_to_lower<Upper>(&self, src: &Self::Buffer<Upper>, dst: &mut Self::Buffer<Upper::Lower>)
    where
        Upper: LowerPrecision,
    {
        if src.len != dst.len {
            let error = self.record_invalid_argument(
                "cast_to_lower",
                format!(
                    "source length {} != destination length {}",
                    src.len, dst.len
                ),
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        }
        if src.len == 0 {
            return;
        }
        self.flush_pending();
        use rayon::prelude::*;
        let src_slice = src.host_slice();
        let dst_slice = dst.host_slice_mut();
        const CHUNK: usize = 4096;
        src_slice
            .par_chunks(CHUNK)
            .zip(dst_slice.par_chunks_mut(CHUNK))
            .for_each(|(src_chunk, dst_chunk)| {
                for (s, d) in src_chunk.iter().zip(dst_chunk.iter_mut()) {
                    *d = Upper::cast_to_lower(*s);
                }
            });
    }

    fn cast_from_lower<Upper>(
        &self,
        src: &Self::Buffer<Upper::Lower>,
        dst: &mut Self::Buffer<Upper>,
    ) where
        Upper: LowerPrecision,
    {
        if src.len != dst.len {
            let error = self.record_invalid_argument(
                "cast_from_lower",
                format!(
                    "source length {} != destination length {}",
                    src.len, dst.len
                ),
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        }
        if src.len == 0 {
            return;
        }
        self.flush_pending();
        use rayon::prelude::*;
        let src_slice = src.host_slice();
        let dst_slice = dst.host_slice_mut();
        const CHUNK: usize = 4096;
        src_slice
            .par_chunks(CHUNK)
            .zip(dst_slice.par_chunks_mut(CHUNK))
            .for_each(|(src_chunk, dst_chunk)| {
                for (s, d) in src_chunk.iter().zip(dst_chunk.iter_mut()) {
                    *d = Upper::cast_from_lower(*s);
                }
            });
    }

    fn validate_gather4_backend_limits<T: DeviceScalar>(
        &self,
        operation: &'static str,
        values_len: usize,
        indices_len: usize,
        output_len: usize,
    ) -> BackendResult<()> {
        Self::check_gather4_backend_limits::<T>(operation, values_len, indices_len, output_len)
    }

    fn gather4_weighted<T>(
        &self,
        values: &Self::Buffer<T>,
        indices: &Self::Buffer<i32>,
        weights: &Self::Buffer<T::Real>,
        out: &mut Self::Buffer<T>,
    ) where
        T: DeviceScalar,
        T::Real: BufferElement,
    {
        let Some(expected_indices) = out.len.checked_mul(4) else {
            let error = self.record_invalid_argument(
                "gather4_weighted",
                format!("output length {} overflows four-index shape", out.len),
            );
            log::error!(
                target: "sonium_compute::metal",
                "{error}"
            );
            return;
        };
        if indices.len != expected_indices || weights.len != expected_indices {
            let error = self.record_invalid_argument(
                "gather4_weighted",
                format!(
                    "indices={}, weights={}, expected={expected_indices}",
                    indices.len, weights.len
                ),
            );
            log::error!(
                target: "sonium_compute::metal",
                "{error}"
            );
            return;
        }
        if let Err(error) = Self::check_gather4_backend_limits::<T>(
            "gather4_weighted",
            values.len(),
            indices.len,
            out.len,
        ) {
            self.record_pending_error(error.clone());
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        }
        // Keep the legacy void entry point on the same checked binding
        // contract as the fallible generic boundary.  These canonical
        // descriptors are whole-buffer today, but make allocation, scalar
        // type, byte alignment, and device validation explicit at launch.
        let descriptor_result = (|| -> BackendResult<()> {
            self.checked_nd_view_descriptor(
                values,
                &[values.len],
                0,
                &[1],
                ViewAliasing::ReadOnly,
            )?;
            self.checked_nd_view_descriptor(
                indices,
                &[indices.len],
                0,
                &[1],
                ViewAliasing::ReadOnly,
            )?;
            self.checked_nd_view_descriptor(
                weights,
                &[weights.len],
                0,
                &[1],
                ViewAliasing::ReadOnly,
            )?;
            self.checked_nd_view_descriptor(out, &[out.len], 0, &[1], ViewAliasing::Disjoint)?;
            crate::backend::validate_gather4_finite_inputs(self, values, weights)?;
            Ok(())
        })();
        if let Err(error) = descriptor_result {
            self.record_pending_error(error.clone());
            log::error!(
                target: "sonium_compute::metal",
                "gather4 rejected native view descriptor: {error}"
            );
            return;
        }
        if out.len == 0 {
            return;
        }

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let values_f32 = unsafe { as_f32_buffer(values) };
            let weights_f32 = unsafe { as_f32_buffer(weights) };
            let out_f32 = unsafe { as_f32_buffer_mut(out) };
            let Some(pso) = self.pso(kernel_names::GATHER4_F32) else {
                return;
            };
            self.dispatch_1d_lazy(pso, out_f32.len as u32, THREADGROUP_SIZE, |enc| {
                set_buffer(enc, 0, &values_f32.raw);
                set_buffer(enc, 1, &indices.raw);
                set_buffer(enc, 2, &weights_f32.raw);
                set_buffer(enc, 3, &out_f32.raw);
                let n = out_f32.len as u32;
                set_scalar(enc, 4, &n);
            });
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<Complex32>() {
            let values_c32 = unsafe { as_c32_buffer(values) };
            let weights_f32 = unsafe { as_f32_buffer(weights) };
            let out_c32 = unsafe { as_c32_buffer_mut(out) };
            let Some(pso) = self.pso(kernel_names::GATHER4_C32) else {
                return;
            };
            self.dispatch_1d_lazy(pso, out_c32.len as u32, THREADGROUP_SIZE, |enc| {
                set_buffer(enc, 0, &values_c32.raw);
                set_buffer(enc, 1, &indices.raw);
                set_buffer(enc, 2, &weights_f32.raw);
                set_buffer(enc, 3, &out_c32.raw);
                let n = out_c32.len as u32;
                set_scalar(enc, 4, &n);
            });
            return;
        }
        // Host fallback for f64 / Complex64.
        self.flush_pending();
        let (mut values_host, _values_staging_guard) =
            match try_host_staging_vec(&self.staging_stats, values.len, "gather4 host values") {
                Ok(staging) => staging,
                Err(error) => {
                    self.record_pending_error(error);
                    return;
                }
            };
        let (mut indices_host, _indices_staging_guard) =
            match try_host_staging_vec(&self.staging_stats, indices.len, "gather4 host indices") {
                Ok(staging) => staging,
                Err(error) => {
                    self.record_pending_error(error);
                    return;
                }
            };
        let (mut weights_host, _weights_staging_guard) =
            match try_host_staging_vec(&self.staging_stats, weights.len, "gather4 host weights") {
                Ok(staging) => staging,
                Err(error) => {
                    self.record_pending_error(error);
                    return;
                }
            };
        let (mut out_host, _out_staging_guard) =
            match try_host_staging_vec(&self.staging_stats, out.len, "gather4 host output") {
                Ok(staging) => staging,
                Err(error) => {
                    self.record_pending_error(error);
                    return;
                }
            };
        if let Err(error) = self.try_copy_d2h(values, &mut values_host) {
            self.record_pending_error(error);
            return;
        }
        if let Err(error) = self.try_copy_d2h(indices, &mut indices_host) {
            self.record_pending_error(error);
            return;
        }
        if let Err(error) = self.try_copy_d2h(weights, &mut weights_host) {
            self.record_pending_error(error);
            return;
        }

        for (row, out_value) in out_host.iter_mut().enumerate() {
            let base = 4 * row;
            let mut acc = T::zero();
            for lane in 0..4 {
                let idx = indices_host[base + lane];
                if idx < 0 {
                    let error = self.record_invalid_argument(
                        "gather4_weighted",
                        format!("index {idx} at output {row} is negative"),
                    );
                    log::error!(target: "sonium_compute::metal", "{error}");
                    return;
                }
                let idx = idx as usize;
                if idx >= values_host.len() {
                    let error = self.record_invalid_argument(
                        "gather4_weighted",
                        format!("index {idx} at output {row} is out of bounds"),
                    );
                    log::error!(target: "sonium_compute::metal", "{error}");
                    return;
                }
                acc = acc + values_host[idx] * T::from_real(weights_host[base + lane]);
            }
            *out_value = acc;
        }

        if let Err(error) = self.try_copy_h2d(&out_host, out) {
            self.record_pending_error(error);
        }
    }

    fn synchronize(&self) -> BackendResult<()> {
        self.flush_pending_checked()
    }

    fn try_synchronize(&self) -> BackendResult<()> {
        self.flush_pending_checked()
    }

    fn try_submit_completion(&self) -> BackendResult<Box<dyn BackendCompletion>> {
        self.check_pending_error()?;
        let command_buffer = self
            .pending
            .lock()
            .map_err(|_| BackendError::OperationFailed {
                operation: "metal_pending_queue",
                reason: "pending mutex poisoned".into(),
            })?
            .take();
        let deferred_scratch = self
            .pending_scratch
            .lock()
            .map_err(|_| BackendError::OperationFailed {
                operation: "metal_pending_scratch",
                reason: "pending scratch mutex poisoned".into(),
            })?
            .drain(..)
            .collect();
        Ok(Box::new(MetalCommandCompletion {
            command_buffer,
            deferred_scratch,
            pending_error: self.pending_error.clone(),
        }))
    }

    fn validate_vector_backend_limits<T: DeviceScalar>(
        &self,
        operation: &'static str,
        vector_len: usize,
    ) -> BackendResult<()> {
        Self::check_vector_backend_limits::<T>(operation, vector_len)
    }

    fn axpy<T: DeviceScalar>(&self, alpha: T, x: &Self::Buffer<T>, y: &mut Self::Buffer<T>) {
        if x.len() != y.len() {
            let error = self.record_invalid_argument(
                "axpy",
                format!("vector lengths {} and {} differ", x.len(), y.len()),
            );
            log::error!(
                target: "sonium_compute::metal",
                "{error}"
            );
            return;
        }
        if let Err(error) = Self::check_vector_backend_limits::<T>("axpy", x.len()) {
            self.record_pending_error(error.clone());
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // SAFETY: T == f32 verified by TypeId.
            let alpha_f32 = unsafe { *(&alpha as *const T as *const f32) };
            let x_f32 = unsafe { as_f32_buffer(x) };
            let y_f32 = unsafe { as_f32_buffer_mut(y) };
            self.axpy_f32_native(alpha_f32, x_f32, y_f32);
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<Complex32>() {
            let alpha_c32 = unsafe { *(&alpha as *const T as *const Complex32) };
            let x_c32 = unsafe { as_c32_buffer(x) };
            let y_c32 = unsafe { as_c32_buffer_mut(y) };
            self.axpy_c32_native(alpha_c32, x_c32, y_c32);
            return;
        }
        // Host fallback for f64 / Complex64.
        self.flush_pending();
        let xh = x.host_slice();
        let yh = y.host_slice_mut();
        axpy_host(alpha, xh, yh);
    }

    fn validate_axpy_many_backend_limits<T: DeviceScalar>(
        &self,
        operation: &'static str,
        vector_len: usize,
        basis_count: usize,
    ) -> BackendResult<()> {
        Self::check_axpy_many_backend_limits::<T>(operation, vector_len, basis_count)
    }

    fn axpy_many<T: DeviceScalar>(
        &self,
        alphas: &[T],
        x_basis: &[&Self::Buffer<T>],
        y: &mut Self::Buffer<T>,
    ) {
        if alphas.len() != x_basis.len() {
            let error = self.record_invalid_argument(
                "axpy_many",
                format!(
                    "coefficient count {} differs basis count {}",
                    alphas.len(),
                    x_basis.len()
                ),
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        }
        if let Some((index, basis)) = x_basis
            .iter()
            .enumerate()
            .find(|(_, basis)| basis.len() != y.len())
        {
            let error = self.record_invalid_argument(
                "axpy_many",
                format!(
                    "basis vector {index} has length {}, expected {}",
                    basis.len(),
                    y.len()
                ),
            );
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        }
        if let Err(error) =
            Self::check_axpy_many_backend_limits::<T>("axpy_many", y.len(), x_basis.len())
        {
            self.record_pending_error(error.clone());
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        }
        if x_basis.is_empty() || y.len == 0 {
            return;
        }

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let alphas_f32 =
                unsafe { std::slice::from_raw_parts(alphas.as_ptr() as *const f32, alphas.len()) };
            let x_f32: Vec<&MetalBuffer<f32>> = match try_collect_backend(
                x_basis
                    .iter()
                    .map(|x| Ok::<_, BackendError>(unsafe { as_f32_buffer(*x) })),
                x_basis.len(),
                "axpy_many pointer table",
            ) {
                Ok(values) => values,
                Err(error) => {
                    self.record_pending_error(error.clone());
                    log::error!(target: "sonium_compute::metal", "{error}");
                    return;
                }
            };
            let y_f32 = unsafe { as_f32_buffer_mut(y) };
            self.axpy_many_f32_native(alphas_f32, &x_f32, y_f32);
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<Complex32>() {
            let alphas_c32 = unsafe {
                std::slice::from_raw_parts(alphas.as_ptr() as *const Complex32, alphas.len())
            };
            let x_c32: Vec<&MetalBuffer<Complex32>> = match try_collect_backend(
                x_basis
                    .iter()
                    .map(|x| Ok::<_, BackendError>(unsafe { as_c32_buffer(*x) })),
                x_basis.len(),
                "axpy_many pointer table",
            ) {
                Ok(values) => values,
                Err(error) => {
                    self.record_pending_error(error.clone());
                    log::error!(target: "sonium_compute::metal", "{error}");
                    return;
                }
            };
            let y_c32 = unsafe { as_c32_buffer_mut(y) };
            self.axpy_many_c32_native(alphas_c32, &x_c32, y_c32);
            return;
        }

        for (&alpha, x) in alphas.iter().zip(x_basis.iter()) {
            self.axpy(alpha, *x, y);
        }
    }

    fn validate_multi_dot_backend_limits<T: DeviceScalar>(
        &self,
        operation: &'static str,
        vector_len: usize,
        basis_count: usize,
    ) -> BackendResult<()> {
        Self::check_axpy_many_backend_limits::<T>(operation, vector_len, basis_count)
    }

    fn try_axpy_many<T: DeviceScalar>(
        &self,
        alphas: &[T],
        x_basis: &[&Self::Buffer<T>],
        y: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        let capabilities = self.capabilities();
        crate::backend::require_scalar_capability::<T>(
            capabilities.backend_name,
            "axpy_many",
            capabilities.axpy_many,
        )?;
        if alphas.len() != x_basis.len() {
            return Err(BackendError::InvalidArgument {
                operation: "axpy_many",
                reason: format!(
                    "coefficient count {} differs basis count {}",
                    alphas.len(),
                    x_basis.len()
                ),
            });
        }
        if let Some((index, basis)) = x_basis
            .iter()
            .enumerate()
            .find(|(_, basis)| basis.len() != y.len())
        {
            return Err(BackendError::InvalidArgument {
                operation: "axpy_many",
                reason: format!(
                    "basis vector {index} has length {}, expected {}",
                    basis.len(),
                    y.len()
                ),
            });
        }
        for &alpha in alphas {
            require_finite_scalar("axpy_many", "alpha", alpha)?;
        }
        Self::check_axpy_many_backend_limits::<T>("axpy_many", y.len(), x_basis.len())?;

        // Keep the checked path allocation-fallible. The legacy generic
        // wrapper builds temporary typed reference vectors with `collect()`;
        // native f32/c32 dispatch can bind those references directly here.
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let basis = try_collect_backend(
                x_basis.iter().map(|x| Ok(unsafe { as_f32_buffer(*x) })),
                x_basis.len(),
                "axpy_many",
            )?;
            let alphas =
                unsafe { std::slice::from_raw_parts(alphas.as_ptr().cast::<f32>(), alphas.len()) };
            self.axpy_many_f32_native(alphas, &basis, unsafe { as_f32_buffer_mut(y) });
            return self.check_pending_error();
        }
        if TypeId::of::<T>() == TypeId::of::<Complex32>() {
            let basis = try_collect_backend(
                x_basis.iter().map(|x| Ok(unsafe { as_c32_buffer(*x) })),
                x_basis.len(),
                "axpy_many",
            )?;
            let alphas = unsafe {
                std::slice::from_raw_parts(alphas.as_ptr().cast::<Complex32>(), alphas.len())
            };
            self.axpy_many_c32_native(alphas, &basis, unsafe { as_c32_buffer_mut(y) });
            return self.check_pending_error();
        }
        catch_backend_panic("axpy_many", || self.axpy_many(alphas, x_basis, y))?;
        self.check_pending_error()
    }

    fn dot<T: DeviceScalar>(&self, x: &Self::Buffer<T>, y: &Self::Buffer<T>) -> T {
        if x.len() != y.len() {
            let error = self.record_invalid_argument(
                "dot",
                format!("vector lengths {} and {} differ", x.len(), y.len()),
            );
            log::error!(
                target: "sonium_compute::metal",
                "{error}"
            );
            return T::zero();
        }
        if let Err(error) = Self::check_vector_backend_limits::<T>("dot", x.len()) {
            self.record_pending_error(error.clone());
            log::error!(target: "sonium_compute::metal", "{error}");
            return T::zero();
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let x_f32 = unsafe { as_f32_buffer(x) };
            let y_f32 = unsafe { as_f32_buffer(y) };
            let result = self.dot_f32_native(x_f32, y_f32);
            // SAFETY: T == f32; transmute the f32 back into the generic T.
            return unsafe { *(&result as *const f32 as *const T) };
        }
        if TypeId::of::<T>() == TypeId::of::<Complex32>() {
            let x_c32 = unsafe { as_c32_buffer(x) };
            let y_c32 = unsafe { as_c32_buffer(y) };
            let result = self.dot_c32_native(x_c32, y_c32);
            return unsafe { *(&result as *const Complex32 as *const T) };
        }
        self.flush_pending();
        dot_host(x.host_slice(), y.host_slice())
    }

    fn nrm2<T: DeviceScalar>(&self, x: &Self::Buffer<T>) -> T::Real {
        if let Err(error) = Self::check_vector_backend_limits::<T>("nrm2", x.len()) {
            self.record_pending_error(error.clone());
            log::error!(target: "sonium_compute::metal", "{error}");
            return T::Real::zero();
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let x_f32 = unsafe { as_f32_buffer(x) };
            let result = self.nrm2_f32_native(x_f32);
            if result.is_finite() {
                return unsafe { *(&result as *const f32 as *const T::Real) };
            }
            self.flush_pending();
            return crate::scalar::stable_nrm2(x.host_slice());
        }
        if TypeId::of::<T>() == TypeId::of::<Complex32>() {
            // Complex32::Real == f32, so the same f32-typed return slot
            // holds the result.
            let x_c32 = unsafe { as_c32_buffer(x) };
            let result = self.nrm2_c32_native(x_c32);
            if result.is_finite() {
                return unsafe { *(&result as *const f32 as *const T::Real) };
            }
            self.flush_pending();
            return crate::scalar::stable_nrm2(x.host_slice());
        }
        self.flush_pending();
        nrm2_host::<T>(x.host_slice())
    }

    fn multi_dot<T: DeviceScalar>(
        &self,
        x_basis: &[&Self::Buffer<T>],
        y: &Self::Buffer<T>,
    ) -> Vec<T> {
        if let Some((index, x)) = x_basis.iter().enumerate().find(|(_, x)| x.len() != y.len()) {
            let error = self.record_invalid_argument(
                "multi_dot",
                format!(
                    "basis vector {index} has length {} (expected {})",
                    x.len(),
                    y.len()
                ),
            );
            log::error!(
                target: "sonium_compute::metal",
                "{error}"
            );
            return Vec::new();
        }
        if let Err(error) =
            Self::check_axpy_many_backend_limits::<T>("multi_dot", y.len(), x_basis.len())
        {
            self.record_pending_error(error.clone());
            log::error!(target: "sonium_compute::metal", "{error}");
            return Vec::new();
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // SAFETY: T == f32 verified by TypeId. Cast each basis ref
            // and y to MetalBuffer<f32>, then transmute the result Vec.
            let x_f32: &[&MetalBuffer<f32>] = unsafe {
                std::slice::from_raw_parts(
                    x_basis.as_ptr() as *const &MetalBuffer<f32>,
                    x_basis.len(),
                )
            };
            let y_f32 = unsafe { as_f32_buffer(y) };
            let result_f32 = self.multi_dot_f32_native(x_f32, y_f32, None);
            // Reinterpret Vec<f32> as Vec<T> — both have identical
            // layout when T == f32.
            return unsafe {
                let mut v = std::mem::ManuallyDrop::new(result_f32);
                Vec::from_raw_parts(v.as_mut_ptr() as *mut T, v.len(), v.capacity())
            };
        }
        if TypeId::of::<T>() == TypeId::of::<Complex32>() {
            let x_c32: &[&MetalBuffer<Complex32>] = unsafe {
                std::slice::from_raw_parts(
                    x_basis.as_ptr() as *const &MetalBuffer<Complex32>,
                    x_basis.len(),
                )
            };
            let y_c32 = unsafe { as_c32_buffer(y) };
            let result_c32 = self.multi_dot_c32_native(x_c32, y_c32, None);
            return unsafe {
                let mut v = std::mem::ManuallyDrop::new(result_c32);
                Vec::from_raw_parts(v.as_mut_ptr() as *mut T, v.len(), v.capacity())
            };
        }
        // Fall through to the default impl pattern — serial dot calls.
        x_basis.iter().map(|x| self.dot(*x, y)).collect()
    }

    /// Checked multi-dot that writes directly into caller-owned storage.
    ///
    /// Metal's legacy `multi_dot` entry point predates the checked API and
    /// returns an owned `Vec`.  Keep that compatibility surface intact, but
    /// make the fallible output-buffer contract explicit here so callers do
    /// not have to rely on a best-effort legacy result (especially when a
    /// command-buffer failure has already been recorded).
    fn try_multi_dot_with_mode_into<T: DeviceScalar>(
        &self,
        mode: crate::backend::ReductionMode,
        x_basis: &[&Self::Buffer<T>],
        y: &Self::Buffer<T>,
        out: &mut [T],
    ) -> BackendResult<()> {
        self.check_pending_error()?;
        let capabilities = self.capabilities();
        crate::backend::require_scalar_capability::<T>(
            capabilities.backend_name,
            "multi_dot_into",
            capabilities.multi_dot,
        )?;
        crate::backend::require_reduction_capability::<T>(
            capabilities.backend_name,
            "multi_dot_into",
            capabilities.reduction,
            mode,
        )?;
        if out.len() != x_basis.len() {
            return Err(BackendError::InvalidArgument {
                operation: "multi_dot_into",
                reason: format!(
                    "output length {} must match basis count {}",
                    out.len(),
                    x_basis.len()
                ),
            });
        }
        if let Some((index, basis)) = x_basis
            .iter()
            .enumerate()
            .find(|(_, basis)| basis.len() != y.len())
        {
            return Err(BackendError::InvalidArgument {
                operation: "multi_dot_into",
                reason: format!(
                    "basis vector {index} has length {} (expected {})",
                    basis.len(),
                    y.len()
                ),
            });
        }
        if x_basis.is_empty() {
            return Ok(());
        }

        // Keep the caller-owned output transactional. Native command-buffer
        // failures and late fallback errors must not publish partial results.
        let (mut staged, _staging_guard) =
            try_host_staging_vec(&self.staging_stats, out.len(), "metal_multi_dot_into")?;

        // Fast mode can use Metal's fused native multi-dot kernels.  The
        // compatibility helper still owns the temporary result Vec, but the
        // public checked API writes only the validated result slice into the
        // caller's storage and never exposes that temporary allocation.
        if matches!(mode, crate::backend::ReductionMode::Fast)
            && TypeId::of::<T>() == TypeId::of::<f32>()
        {
            let x_f32: Vec<&MetalBuffer<f32>> = try_collect_backend(
                x_basis
                    .iter()
                    .map(|x| Ok::<_, BackendError>(unsafe { as_f32_buffer(*x) })),
                x_basis.len(),
                "multi_dot_into pointer table",
            )?;
            let y_f32 = unsafe { as_f32_buffer(y) };
            let out_f32 = unsafe {
                std::slice::from_raw_parts_mut(staged.as_mut_ptr() as *mut f32, staged.len())
            };
            self.multi_dot_f32_native(&x_f32, y_f32, Some(out_f32));
            self.check_pending_error()?;
            if out_f32.iter().any(|value| !value.is_finite()) {
                return Err(BackendError::OperationFailed {
                    operation: "multi_dot_into",
                    reason: "reduction result is non-finite".into(),
                });
            }
            out.copy_from_slice(&staged);
            return Ok(());
        }
        if matches!(mode, crate::backend::ReductionMode::Fast)
            && TypeId::of::<T>() == TypeId::of::<Complex32>()
        {
            let x_c32: Vec<&MetalBuffer<Complex32>> = try_collect_backend(
                x_basis
                    .iter()
                    .map(|x| Ok::<_, BackendError>(unsafe { as_c32_buffer(*x) })),
                x_basis.len(),
                "multi_dot_into pointer table",
            )?;
            let y_c32 = unsafe { as_c32_buffer(y) };
            let out_c32 = unsafe {
                std::slice::from_raw_parts_mut(staged.as_mut_ptr() as *mut Complex32, staged.len())
            };
            self.multi_dot_c32_native(&x_c32, y_c32, Some(out_c32));
            self.check_pending_error()?;
            if out_c32.iter().any(|value| !value.is_finite()) {
                return Err(BackendError::OperationFailed {
                    operation: "multi_dot_into",
                    reason: "reduction result is non-finite".into(),
                });
            }
            out.copy_from_slice(&staged);
            return Ok(());
        }
        if matches!(mode, crate::backend::ReductionMode::Fast) {
            let result = self.multi_dot(x_basis, y);
            // Consume any typed error recorded by the legacy compatibility
            // wrapper before inspecting its best-effort Vec result.  Without
            // this check, a native launch/argument failure could be returned
            // as a generic length error while leaking a stale pending error
            // into the caller's next operation.
            self.check_pending_error()?;
            if result.len() != out.len() {
                return Err(BackendError::OperationFailed {
                    operation: "multi_dot_into",
                    reason: format!(
                        "backend returned {} results for {} basis vectors",
                        result.len(),
                        out.len()
                    ),
                });
            }
            if result.iter().any(|value| !value.is_finite()) {
                return Err(BackendError::OperationFailed {
                    operation: "multi_dot_into",
                    reason: "reduction result is non-finite".into(),
                });
            }
            out.copy_from_slice(&result);
            return Ok(());
        }

        // Reproducible and deterministic policies deliberately use the
        // checked scalar dot path so their summation order remains stable.
        for (slot, basis) in staged.iter_mut().zip(x_basis.iter().copied()) {
            *slot = self.try_dot_with_mode(mode, basis, y)?;
        }
        self.check_pending_error()?;
        out.copy_from_slice(&staged);
        Ok(())
    }

    fn validate_strided_backend_limits(
        &self,
        operation: &'static str,
        vector_len: usize,
        basis_count: usize,
    ) -> BackendResult<()> {
        Self::check_strided_backend_limits(operation, vector_len, basis_count)
    }

    fn multi_dot_strided<T: DeviceScalar>(
        &self,
        basis: &Self::Buffer<T>,
        vector_len: usize,
        basis_count: usize,
        y: &Self::Buffer<T>,
    ) -> Vec<T> {
        if let Err(error) =
            self.validate_strided_backend_limits("multi_dot_strided", vector_len, basis_count)
        {
            self.record_pending_error(error.clone());
            log::error!(target: "sonium_compute::metal", "{error}");
            return Vec::new();
        }
        if y.len != vector_len {
            let error = self.record_invalid_argument(
                "multi_dot_strided",
                format!("y length {} (expected {vector_len})", y.len),
            );
            log::error!(
                target: "sonium_compute::metal",
                "{error}"
            );
            return Vec::new();
        }
        let Some(required) = vector_len.checked_mul(basis_count) else {
            let error = self.record_invalid_argument(
                "multi_dot_strided",
                format!("basis shape {basis_count}x{vector_len} overflows"),
            );
            log::error!(
                target: "sonium_compute::metal",
                "{error}"
            );
            return Vec::new();
        };
        if basis.len < required {
            let error = self.record_invalid_argument(
                "multi_dot_strided",
                format!("basis length {} (expected at least {required})", basis.len),
            );
            log::error!(
                target: "sonium_compute::metal",
                "{error}"
            );
            return Vec::new();
        }
        let basis_view = match self.checked_nd_view_descriptor(
            basis,
            &[basis_count, vector_len],
            0,
            &[vector_len.max(1), 1],
            ViewAliasing::ReadOnly,
        ) {
            Ok(view) => view,
            Err(error) => {
                self.record_pending_error(error.clone());
                log::error!(
                    target: "sonium_compute::metal",
                    "multi_dot_strided rejected basis view descriptor: {error}"
                );
                return Vec::new();
            }
        };
        let y_view = match self.checked_nd_view_descriptor(
            y,
            &[vector_len],
            0,
            &[1],
            ViewAliasing::ReadOnly,
        ) {
            Ok(view) => view,
            Err(error) => {
                self.record_pending_error(error.clone());
                log::error!(
                    target: "sonium_compute::metal",
                    "multi_dot_strided rejected input view descriptor: {error}"
                );
                return Vec::new();
            }
        };
        if basis_count == 0 {
            return Vec::new();
        }

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let basis_f32 = unsafe { as_f32_buffer(basis) };
            let y_f32 = unsafe { as_f32_buffer(y) };
            let result = self.multi_dot_strided_f32_native(
                basis_f32,
                vector_len,
                basis_count,
                y_f32,
                &basis_view,
                &y_view,
            );
            return unsafe {
                let mut result = std::mem::ManuallyDrop::new(result);
                Vec::from_raw_parts(
                    result.as_mut_ptr() as *mut T,
                    result.len(),
                    result.capacity(),
                )
            };
        }
        if TypeId::of::<T>() == TypeId::of::<Complex32>() {
            let basis_c32 = unsafe { as_c32_buffer(basis) };
            let y_c32 = unsafe { as_c32_buffer(y) };
            let result = self.multi_dot_strided_c32_native(
                basis_c32,
                vector_len,
                basis_count,
                y_c32,
                &basis_view,
                &y_view,
            );
            return unsafe {
                let mut result = std::mem::ManuallyDrop::new(result);
                Vec::from_raw_parts(
                    result.as_mut_ptr() as *mut T,
                    result.len(),
                    result.capacity(),
                )
            };
        }

        self.flush_pending();
        let mut result = zeroed_compatibility_result::<T>(basis_count);
        if result.len() != basis_count {
            return result;
        }
        for (col, slot) in result.iter_mut().enumerate() {
            let base = col * vector_len;
            let mut acc = T::zero();
            for i in 0..vector_len {
                acc = acc + basis.host_slice()[base + i].conj_mul(y.host_slice()[i]);
            }
            *slot = acc;
        }
        result
    }

    fn basis_axpy_strided<T: DeviceScalar>(
        &self,
        alpha: T,
        basis: &Self::Buffer<T>,
        vector_len: usize,
        basis_count: usize,
        coeffs: &[T],
        y: &mut Self::Buffer<T>,
    ) {
        if let Err(error) =
            self.validate_strided_backend_limits("basis_axpy_strided", vector_len, basis_count)
        {
            self.record_pending_error(error.clone());
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        }
        if coeffs.len() != basis_count {
            let error = self.record_invalid_argument(
                "basis_axpy_strided",
                format!(
                    "coefficient length {} (expected {basis_count})",
                    coeffs.len()
                ),
            );
            log::error!(
                target: "sonium_compute::metal",
                "{error}"
            );
            return;
        }
        if y.len != vector_len {
            let error = self.record_invalid_argument(
                "basis_axpy_strided",
                format!("y length {} (expected {vector_len})", y.len),
            );
            log::error!(
                target: "sonium_compute::metal",
                "{error}"
            );
            return;
        }
        let Some(required) = vector_len.checked_mul(basis_count) else {
            let error = self.record_invalid_argument(
                "basis_axpy_strided",
                format!("basis shape {basis_count}x{vector_len} overflows"),
            );
            log::error!(
                target: "sonium_compute::metal",
                "{error}"
            );
            return;
        };
        if basis.len < required {
            let error = self.record_invalid_argument(
                "basis_axpy_strided",
                format!("basis length {} (expected at least {required})", basis.len),
            );
            log::error!(
                target: "sonium_compute::metal",
                "{error}"
            );
            return;
        }
        let basis_view = match self.checked_nd_view_descriptor(
            basis,
            &[basis_count, vector_len],
            0,
            &[vector_len.max(1), 1],
            ViewAliasing::ReadOnly,
        ) {
            Ok(view) => view,
            Err(error) => {
                self.record_pending_error(error.clone());
                log::error!(
                    target: "sonium_compute::metal",
                    "basis AXPY rejected basis view descriptor: {error}"
                );
                return;
            }
        };
        let y_view = match self.checked_nd_view_descriptor(
            y,
            &[vector_len],
            0,
            &[1],
            ViewAliasing::Disjoint,
        ) {
            Ok(view) => view,
            Err(error) => {
                self.record_pending_error(error.clone());
                log::error!(
                    target: "sonium_compute::metal",
                    "basis AXPY rejected output view descriptor: {error}"
                );
                return;
            }
        };
        if basis_count == 0 || vector_len == 0 {
            return;
        }

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let alpha = unsafe { *(&alpha as *const T as *const f32) };
            let basis = unsafe { as_f32_buffer(basis) };
            let coeffs =
                unsafe { std::slice::from_raw_parts(coeffs.as_ptr() as *const f32, basis_count) };
            let y = unsafe { as_f32_buffer_mut(y) };
            self.basis_axpy_strided_native(
                kernel_names::AXPY_MANY_F32,
                alpha,
                basis,
                vector_len,
                basis_count,
                coeffs,
                y,
                &basis_view,
                &y_view,
            );
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<Complex32>() {
            let alpha = unsafe { *(&alpha as *const T as *const Complex32) };
            let basis = unsafe { as_c32_buffer(basis) };
            let coeffs = unsafe {
                std::slice::from_raw_parts(coeffs.as_ptr() as *const Complex32, basis_count)
            };
            let y = unsafe { as_c32_buffer_mut(y) };
            self.basis_axpy_strided_native(
                kernel_names::AXPY_MANY_C32,
                alpha,
                basis,
                vector_len,
                basis_count,
                coeffs,
                y,
                &basis_view,
                &y_view,
            );
            return;
        }

        self.flush_pending();
        let basis_host = basis.host_slice();
        let y_host = y.host_slice_mut();
        for (i, value) in y_host.iter_mut().enumerate() {
            for basis_index in 0..basis_count {
                *value =
                    *value + alpha * coeffs[basis_index] * basis_host[basis_index * vector_len + i];
            }
        }
    }

    fn scale<T: DeviceScalar>(&self, alpha: T, x: &mut Self::Buffer<T>) {
        if let Err(error) = Self::check_vector_backend_limits::<T>("scale", x.len()) {
            self.record_pending_error(error.clone());
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let alpha_f32 = unsafe { *(&alpha as *const T as *const f32) };
            let x_f32 = unsafe { as_f32_buffer_mut(x) };
            self.scale_f32_native(alpha_f32, x_f32);
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<Complex32>() {
            let alpha_c32 = unsafe { *(&alpha as *const T as *const Complex32) };
            let x_c32 = unsafe { as_c32_buffer_mut(x) };
            self.scale_c32_native(alpha_c32, x_c32);
            return;
        }
        self.flush_pending();
        scale_host(alpha, x.host_slice_mut());
    }

    fn validate_gemv_backend_limits<T: DeviceScalar>(
        &self,
        op: Transpose,
        m: usize,
        n: usize,
        lda: usize,
    ) -> BackendResult<()> {
        let _ = op;
        Self::check_gemv_backend_limits::<T>("gemv", m, n, lda)
    }

    fn gemv<T: DeviceScalar>(
        &self,
        op: Transpose,
        m: usize,
        n: usize,
        alpha: T,
        a: &Self::Buffer<T>,
        lda: usize,
        x: &Self::Buffer<T>,
        beta: T,
        y: &mut Self::Buffer<T>,
    ) {
        if let Err(error) = Self::check_gemv_backend_limits::<T>("gemv", m, n, lda) {
            self.record_pending_error(error.clone());
            log::error!(target: "sonium_compute::metal", "{error}");
            return;
        }
        // Native MSL path for f32 / Complex32. NoTrans dispatches one
        // simdgroup per output row striding cols; Trans / ConjTrans
        // dispatches one simdgroup per output row striding stored rows
        // (column-strided gather). ConjTrans on Complex32 conjugates A
        // inside the kernel via a flag arg; on f32 Trans == ConjTrans.
        // f64 / Complex64 fall through to the CPU BLAS-2 path over the
        // shared-storage buffer (Apple Metal compiler rejects `double`).
        let tid = TypeId::of::<T>();
        let is_no_trans = op == Transpose::NoTrans;
        let is_conj_trans = op == Transpose::ConjTrans;
        if tid == TypeId::of::<f32>() {
            let alpha_f32 = unsafe { *(&alpha as *const T as *const f32) };
            let beta_f32 = unsafe { *(&beta as *const T as *const f32) };
            let a_f32 = unsafe { as_f32_buffer(a) };
            let x_f32 = unsafe { as_f32_buffer(x) };
            let y_f32 = unsafe { as_f32_buffer_mut(y) };
            if is_no_trans {
                self.gemv_f32_native_notrans(
                    m as u32, n as u32, alpha_f32, a_f32, lda as u32, x_f32, beta_f32, y_f32,
                );
            } else {
                // Trans / ConjTrans (== Trans on real f32).
                self.gemv_f32_native_trans(
                    m as u32, n as u32, alpha_f32, a_f32, lda as u32, x_f32, beta_f32, y_f32,
                );
            }
            return;
        }
        if tid == TypeId::of::<Complex32>() {
            let alpha_c32 = unsafe { *(&alpha as *const T as *const Complex32) };
            let beta_c32 = unsafe { *(&beta as *const T as *const Complex32) };
            let a_c32 = unsafe { as_c32_buffer(a) };
            let x_c32 = unsafe { as_c32_buffer(x) };
            let y_c32 = unsafe { as_c32_buffer_mut(y) };
            if is_no_trans {
                self.gemv_c32_native_notrans(
                    m as u32, n as u32, alpha_c32, a_c32, lda as u32, x_c32, beta_c32, y_c32,
                );
            } else {
                self.gemv_c32_native_trans(
                    m as u32,
                    n as u32,
                    alpha_c32,
                    a_c32,
                    lda as u32,
                    x_c32,
                    beta_c32,
                    y_c32,
                    is_conj_trans,
                );
            }
            return;
        }
        // CPU fallback over shared-storage buffer: covers all ops for
        // f64/Complex64 (no `double` on Apple Metal).
        self.flush_pending();
        if tid == TypeId::of::<f32>() {
            // SAFETY: TypeId match guarantees layout-compatible reinterpretation.
            unsafe {
                cpu_blas2::gemv_f32(
                    op,
                    m,
                    n,
                    *(&alpha as *const T as *const f32),
                    std::slice::from_raw_parts(a.host_slice().as_ptr() as *const f32, a.len),
                    lda,
                    std::slice::from_raw_parts(x.host_slice().as_ptr() as *const f32, x.len),
                    *(&beta as *const T as *const f32),
                    std::slice::from_raw_parts_mut(
                        y.host_slice_mut().as_mut_ptr() as *mut f32,
                        y.len,
                    ),
                );
            }
        } else if tid == TypeId::of::<f64>() {
            unsafe {
                cpu_blas2::gemv_f64(
                    op,
                    m,
                    n,
                    *(&alpha as *const T as *const f64),
                    std::slice::from_raw_parts(a.host_slice().as_ptr() as *const f64, a.len),
                    lda,
                    std::slice::from_raw_parts(x.host_slice().as_ptr() as *const f64, x.len),
                    *(&beta as *const T as *const f64),
                    std::slice::from_raw_parts_mut(
                        y.host_slice_mut().as_mut_ptr() as *mut f64,
                        y.len,
                    ),
                );
            }
        } else if tid == TypeId::of::<Complex32>() {
            unsafe {
                cpu_blas2::gemv_c32(
                    op,
                    m,
                    n,
                    *(&alpha as *const T as *const Complex32),
                    std::slice::from_raw_parts(a.host_slice().as_ptr() as *const Complex32, a.len),
                    lda,
                    std::slice::from_raw_parts(x.host_slice().as_ptr() as *const Complex32, x.len),
                    *(&beta as *const T as *const Complex32),
                    std::slice::from_raw_parts_mut(
                        y.host_slice_mut().as_mut_ptr() as *mut Complex32,
                        y.len,
                    ),
                );
            }
        } else if tid == TypeId::of::<num_complex::Complex64>() {
            use num_complex::Complex64;
            unsafe {
                cpu_blas2::gemv_c64(
                    op,
                    m,
                    n,
                    *(&alpha as *const T as *const Complex64),
                    std::slice::from_raw_parts(a.host_slice().as_ptr() as *const Complex64, a.len),
                    lda,
                    std::slice::from_raw_parts(x.host_slice().as_ptr() as *const Complex64, x.len),
                    *(&beta as *const T as *const Complex64),
                    std::slice::from_raw_parts_mut(
                        y.host_slice_mut().as_mut_ptr() as *mut Complex64,
                        y.len,
                    ),
                );
            }
        } else {
            if let Err(error) = crate::backend::host_gemv(self, op, m, n, alpha, a, lda, x, beta, y)
            {
                self.record_pending_error(error.clone());
                log::error!(target: "sonium_compute::metal", "gemv host fallback rejected request: {error}");
            }
        }
    }

    fn gemm<T: DeviceScalar>(
        &self,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        a: &Self::Buffer<T>,
        lda: usize,
        b: &Self::Buffer<T>,
        ldb: usize,
        beta: T,
        c: &mut Self::Buffer<T>,
        ldc: usize,
    ) {
        // f32 routes to MPSMatrixMultiplication — Apple-Silicon tensor-core
        // FMA throughput, no host bounce. ConjTrans on real types collapses
        // to Trans. Complex32 has a prototype 4-real-MPS path; f64/c64 stay
        // on the SME2-accelerated CPU path over the shared-storage buffer
        // (Apple Metal compiler rejects `double`).
        let tid = TypeId::of::<T>();
        if tid == TypeId::of::<f32>() {
            let trans_left = !matches!(transa, Transpose::NoTrans);
            let trans_right = !matches!(transb, Transpose::NoTrans);
            // SAFETY: TypeId match → T = f32; same-size same-layout cast.
            let alpha_f32 = unsafe { *(&alpha as *const T as *const f32) };
            let beta_f32 = unsafe { *(&beta as *const T as *const f32) };
            let a_f32 = unsafe { as_f32_buffer(a) };
            let b_f32 = unsafe { as_f32_buffer(b) };
            let c_f32 = unsafe { as_f32_buffer_mut(c) };
            if Self::f32_gemm_uses_mps(m, n, k) {
                if let Err(error) = self.mps_gemm_f32(
                    trans_left,
                    trans_right,
                    m,
                    n,
                    k,
                    alpha_f32,
                    a_f32,
                    lda,
                    b_f32,
                    ldb,
                    beta_f32,
                    c_f32,
                    ldc,
                ) {
                    self.record_pending_error(error);
                }
            } else {
                self.cpu_gemm_f32_shared(
                    transa, transb, m, n, k, alpha_f32, a_f32, lda, b_f32, ldb, beta_f32, c_f32,
                    ldc,
                );
            }
            return;
        }
        if tid == TypeId::of::<Complex32>() {
            let alpha_c32 = unsafe { *(&alpha as *const T as *const Complex32) };
            let beta_c32 = unsafe { *(&beta as *const T as *const Complex32) };
            let a_c32 = unsafe { as_c32_buffer(a) };
            let b_c32 = unsafe { as_c32_buffer(b) };
            let c_c32 = unsafe { as_c32_buffer_mut(c) };
            if Self::c32_gemm_uses_mps(m, n, k) {
                if let Err(error) = self.mps_gemm_c32_prototype(
                    transa, transb, m, n, k, alpha_c32, a_c32, lda, b_c32, ldb, beta_c32, c_c32,
                    ldc,
                ) {
                    self.record_pending_error(error);
                }
            } else {
                self.cpu_gemm_c32_shared(
                    transa, transb, m, n, k, alpha_c32, a_c32, lda, b_c32, ldb, beta_c32, c_c32,
                    ldc,
                );
            }
            return;
        }
        self.flush_pending();
        if tid == TypeId::of::<f64>() {
            unsafe {
                cpu_blas3::gemm_f64(
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    *(&alpha as *const T as *const f64),
                    std::slice::from_raw_parts(a.host_slice().as_ptr() as *const f64, a.len),
                    lda,
                    std::slice::from_raw_parts(b.host_slice().as_ptr() as *const f64, b.len),
                    ldb,
                    *(&beta as *const T as *const f64),
                    std::slice::from_raw_parts_mut(
                        c.host_slice_mut().as_mut_ptr() as *mut f64,
                        c.len,
                    ),
                    ldc,
                );
            }
        } else if tid == TypeId::of::<num_complex::Complex64>() {
            use num_complex::Complex64;
            unsafe {
                cpu_blas3::gemm_c64(
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    *(&alpha as *const T as *const Complex64),
                    std::slice::from_raw_parts(a.host_slice().as_ptr() as *const Complex64, a.len),
                    lda,
                    std::slice::from_raw_parts(b.host_slice().as_ptr() as *const Complex64, b.len),
                    ldb,
                    *(&beta as *const T as *const Complex64),
                    std::slice::from_raw_parts_mut(
                        c.host_slice_mut().as_mut_ptr() as *mut Complex64,
                        c.len,
                    ),
                    ldc,
                );
            }
        } else {
            if let Err(error) = crate::backend::host_gemm(
                self, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
            ) {
                self.record_pending_error(error.clone());
                log::error!(target: "sonium_compute::metal", "gemm host fallback rejected request: {error}");
            }
        }
    }

    fn prebake_csr<'a, T: DeviceScalar>(
        &'a self,
        m: usize,
        n: usize,
        row_ptr: &'a Self::Buffer<i32>,
        col_idx: &'a Self::Buffer<i32>,
        values: &'a Self::Buffer<T>,
    ) -> Self::CsrPrebaked<'a, T> {
        let invalid = |reason: String| {
            MetalCsrPrebaked::failed(
                m,
                n,
                row_ptr,
                col_idx,
                values,
                BackendError::InvalidArgument {
                    operation: "prebake_csr",
                    reason,
                },
            )
        };
        let Some(expected_row_ptr_len) = m.checked_add(1) else {
            return invalid("row pointer length overflow".into());
        };
        if m > i32::MAX as usize || n > i32::MAX as usize {
            return invalid("CSR dimensions exceed Metal's i32 index range".into());
        }
        if row_ptr.len != expected_row_ptr_len {
            return invalid(format!(
                "row pointer length {} != expected {expected_row_ptr_len}",
                row_ptr.len
            ));
        }
        if col_idx.len != values.len {
            return invalid(format!(
                "column/value lengths {} and {} differ",
                col_idx.len, values.len
            ));
        }
        let row_ptr_host = row_ptr.host_slice();
        if row_ptr_host.first().copied() != Some(0) {
            return invalid("row pointer must start at zero".into());
        }
        for (row, window) in row_ptr_host.windows(2).enumerate() {
            if window[0] < 0 || window[1] < window[0] {
                return invalid(format!("row pointer is invalid at row {row}"));
            }
        }
        let nnz = usize::try_from(*row_ptr_host.last().unwrap_or(&0)).unwrap_or(usize::MAX);
        if nnz != col_idx.len {
            return invalid(format!("terminal row pointer {nnz} != nnz {}", col_idx.len));
        }
        for (position, &column) in col_idx.host_slice().iter().enumerate() {
            if column < 0 || column as usize >= n {
                return invalid(format!(
                    "column {column} at position {position} is out of range"
                ));
            }
        }

        // Bucket rows by nnz for the adaptive SpMV. Only the f32 / Complex32
        // SpMV paths consume the buckets; f64 / Complex64 fall through to
        // host SpMV, so skip the (small) bucketing cost there.
        let buckets = if m == 0 {
            None
        } else {
            let tid = TypeId::of::<T>();
            let gpu_native = tid == TypeId::of::<f32>() || tid == TypeId::of::<Complex32>();
            if !gpu_native {
                None
            } else {
                // Read row_ptr from shared-storage memory. The trait
                // contract is that callers have host-synchronized
                // `row_ptr` before passing it in — every in-tree caller
                // builds row_ptr via `copy_h2d` (which flushes prior
                // pending writes itself) or sets it before any GPU op.
                // We do NOT drain the pending command buffer here:
                // frequency-sweep workloads call `prebake_csr` between
                // every Krylov iteration with the same row_ptr; an
                // implicit flush here would serialise that hot loop.
                let rp = row_ptr_host;
                let (mut short, _short_staging_guard) = match try_host_staging_capacity::<i32>(
                    &self.staging_stats,
                    m,
                    "prebake_csr short-row staging",
                ) {
                    Ok(values) => values,
                    Err(error) => {
                        self.record_pending_error(error.clone());
                        return MetalCsrPrebaked::failed(m, n, row_ptr, col_idx, values, error);
                    }
                };
                let (mut simd_rows, _simd_staging_guard) = match try_host_staging_capacity::<i32>(
                    &self.staging_stats,
                    m,
                    "prebake_csr SIMD-row staging",
                ) {
                    Ok(values) => values,
                    Err(error) => {
                        self.record_pending_error(error.clone());
                        return MetalCsrPrebaked::failed(m, n, row_ptr, col_idx, values, error);
                    }
                };
                let (mut cta_rows, _cta_staging_guard) = match try_host_staging_capacity::<i32>(
                    &self.staging_stats,
                    m,
                    "prebake_csr CTA-row staging",
                ) {
                    Ok(values) => values,
                    Err(error) => {
                        self.record_pending_error(error.clone());
                        return MetalCsrPrebaked::failed(m, n, row_ptr, col_idx, values, error);
                    }
                };
                for i in 0..m {
                    let nnz = rp[i + 1] - rp[i];
                    if nnz < 8 {
                        short.push(i as i32);
                    } else if nnz < 64 {
                        simd_rows.push(i as i32);
                    } else {
                        cta_rows.push(i as i32);
                    }
                }
                let upload_indices = |idx: &[i32]| -> BackendResult<MetalBuffer<i32>> {
                    // Always at least len=1 (raw_alloc with 0 returns
                    // a 1-byte buffer; bound the kernel by count param).
                    let n_alloc = idx.len().max(1);
                    let mut buf = self.raw_alloc::<i32>(n_alloc).map_err(BackendError::from)?;
                    if !idx.is_empty() {
                        buf.host_slice_mut()[..idx.len()].copy_from_slice(idx);
                    }
                    Ok(buf)
                };
                let short_count = short.len() as u32;
                let simd_count = simd_rows.len() as u32;
                let cta_count = cta_rows.len() as u32;
                let buckets = (
                    upload_indices(&short),
                    upload_indices(&simd_rows),
                    upload_indices(&cta_rows),
                );
                match buckets {
                    (Ok(short), Ok(simd), Ok(cta)) => Some(CsrBuckets {
                        short,
                        short_count,
                        simd,
                        simd_count,
                        cta,
                        cta_count,
                    }),
                    (Err(error), _, _) | (_, Err(error), _) | (_, _, Err(error)) => {
                        return MetalCsrPrebaked::failed(m, n, row_ptr, col_idx, values, error);
                    }
                }
            }
        };
        MetalCsrPrebaked {
            m,
            n,
            row_ptr,
            col_idx,
            values,
            buckets,
            initialization_error: None,
        }
    }

    fn try_prebake_csr<'a, T: DeviceScalar>(
        &'a self,
        m: usize,
        n: usize,
        row_ptr: &'a Self::Buffer<i32>,
        col_idx: &'a Self::Buffer<i32>,
        values: &'a Self::Buffer<T>,
    ) -> BackendResult<Self::CsrPrebaked<'a, T>> {
        self.check_pending_error()?;
        let mut prebaked = self.prebake_csr(m, n, row_ptr, col_idx, values);
        prebaked
            .initialization_error
            .take()
            .map_or(Ok(prebaked), Err)
    }

    fn try_spmv_prebaked<T: DeviceScalar>(
        &self,
        prebaked: &Self::CsrPrebaked<'_, T>,
        x: &Self::Buffer<T>,
        y: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        self.check_pending_error()?;
        let capabilities = self.capabilities();
        crate::backend::require_scalar_capability::<T>(
            capabilities.backend_name,
            "spmv_prebaked",
            capabilities.csr_spmv,
        )?;
        if let Some(error) = &prebaked.initialization_error {
            return Err(error.clone());
        }
        if x.len != prebaked.n || y.len != prebaked.m {
            return Err(BackendError::InvalidArgument {
                operation: "spmv_prebaked",
                reason: format!(
                    "buffer lengths ({}, {}) do not match matrix ({}, {})",
                    x.len, y.len, prebaked.n, prebaked.m
                ),
            });
        }
        Self::check_spmv_backend_limits::<T>("spmv_prebaked", prebaked.m)?;
        crate::backend::catch_backend_panic("spmv_prebaked", || {
            self.spmv_prebaked(prebaked, x, y)
        })?;
        self.check_pending_error()
    }

    #[allow(clippy::too_many_arguments)]
    fn try_spmm_prebaked<T: DeviceScalar>(
        &self,
        prebaked: &Self::CsrPrebaked<'_, T>,
        nrhs: usize,
        x: &Self::Buffer<T>,
        ldx: usize,
        y: &mut Self::Buffer<T>,
        ldy: usize,
    ) -> BackendResult<()> {
        self.check_pending_error()?;
        let capabilities = self.capabilities();
        crate::backend::require_scalar_capability::<T>(
            capabilities.backend_name,
            "spmm_prebaked",
            capabilities.csr_spmm,
        )?;
        if let Some(error) = &prebaked.initialization_error {
            return Err(error.clone());
        }
        if nrhs != 0 && (ldx < prebaked.n || ldy < prebaked.m) {
            return Err(BackendError::InvalidArgument {
                operation: "spmm_prebaked",
                reason: format!(
                    "leading dimensions ({ldx}, {ldy}) are smaller than matrix ({}, {})",
                    prebaked.n, prebaked.m
                ),
            });
        }
        let x_required = if nrhs == 0 {
            0
        } else {
            (nrhs - 1)
                .checked_mul(ldx)
                .and_then(|base| base.checked_add(prebaked.n))
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "spmm_prebaked",
                    reason: "x required length overflow".into(),
                })?
        };
        let y_required = if nrhs == 0 {
            0
        } else {
            (nrhs - 1)
                .checked_mul(ldy)
                .and_then(|base| base.checked_add(prebaked.m))
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "spmm_prebaked",
                    reason: "y required length overflow".into(),
                })?
        };
        if x.len < x_required || y.len < y_required {
            return Err(BackendError::InvalidArgument {
                operation: "spmm_prebaked",
                reason: format!("buffers are too short for nrhs={nrhs}, ldx={ldx}, ldy={ldy}"),
            });
        }
        crate::backend::catch_backend_panic("spmm_prebaked", || {
            self.spmm_prebaked(prebaked, nrhs, x, ldx, y, ldy)
        })?;
        self.check_pending_error()
    }

    fn try_spmm_prebaked_colmajor<T: DeviceScalar>(
        &self,
        prebaked: &Self::CsrPrebaked<'_, T>,
        rhs: usize,
        x: &Self::Buffer<T>,
        y: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        self.check_pending_error()?;
        if let Some(error) = &prebaked.initialization_error {
            return Err(error.clone());
        }
        let x_required =
            checked_strided_span("spmm_prebaked_colmajor", rhs, prebaked.n, prebaked.n)?;
        let y_required =
            checked_strided_span("spmm_prebaked_colmajor", rhs, prebaked.m, prebaked.m)?;
        if x.len < x_required || y.len < y_required {
            return Err(BackendError::InvalidArgument {
                operation: "spmm_prebaked_colmajor",
                reason: format!(
                    "buffers are too short for rhs={rhs}: x={}, y={}",
                    x.len, y.len
                ),
            });
        }
        crate::backend::catch_backend_panic("spmm_prebaked_colmajor", || {
            self.spmm_prebaked_colmajor(prebaked, rhs, x, y)
        })?;
        self.check_pending_error()
    }

    fn spmv_prebaked<T: DeviceScalar>(
        &self,
        prebaked: &Self::CsrPrebaked<'_, T>,
        x: &Self::Buffer<T>,
        y: &mut Self::Buffer<T>,
    ) {
        if let Some(error) = &prebaked.initialization_error {
            self.record_pending_error(error.clone());
            return;
        }
        if let Err(error) = Self::check_spmv_backend_limits::<T>("spmv_prebaked", prebaked.m) {
            self.record_pending_error(error);
            return;
        }
        if x.len != prebaked.n || y.len != prebaked.m {
            self.record_pending_error(BackendError::InvalidArgument {
                operation: "spmv_prebaked",
                reason: format!(
                    "buffer lengths ({}, {}) do not match matrix ({}, {})",
                    x.len, y.len, prebaked.n, prebaked.m
                ),
            });
            return;
        }
        let Some(expected_row_ptr_len) = prebaked.m.checked_add(1) else {
            self.record_pending_error(BackendError::InvalidArgument {
                operation: "spmv_prebaked",
                reason: "row pointer length overflows usize".into(),
            });
            return;
        };
        if prebaked.row_ptr.len != expected_row_ptr_len {
            self.record_pending_error(BackendError::InvalidArgument {
                operation: "spmv_prebaked",
                reason: "row pointer length does not match matrix rows".into(),
            });
            return;
        }
        let tid = TypeId::of::<T>();
        if let Some(buckets) = &prebaked.buckets
            && (tid == TypeId::of::<f32>() || tid == TypeId::of::<Complex32>())
        {
            // Choose kernel triples based on T.
            let (short_k, simd_k, cta_k) = if tid == TypeId::of::<f32>() {
                (
                    kernel_names::SPMV_CSR_SHORT_F32,
                    kernel_names::SPMV_CSR_SIMD_F32,
                    kernel_names::SPMV_CSR_CTA_F32,
                )
            } else {
                (
                    kernel_names::SPMV_CSR_SHORT_C32,
                    kernel_names::SPMV_CSR_SIMD_C32,
                    kernel_names::SPMV_CSR_CTA_C32,
                )
            };
            // Dispatch each non-empty bucket onto the pending command buffer.
            let elem_bytes = std::mem::size_of::<T>() as u64;
            let dispatch_bucket = |kernel: &'static str,
                                   row_idx_buf: &MetalBuffer<i32>,
                                   count: u32,
                                   grid_per_row: u64,
                                   tg: u64,
                                   shared_bytes: Option<u64>| {
                if count == 0 {
                    return;
                }
                let Some(pso) = self.pso(kernel) else {
                    return;
                };
                let Some(cmd_buf) = self.pending_command_buffer() else {
                    return;
                };
                let Some(encoder) = self.compute_encoder(&cmd_buf) else {
                    return;
                };
                encoder.setComputePipelineState(pso);
                set_buffer(&encoder, 0, &prebaked.row_ptr.raw);
                set_buffer(&encoder, 1, &prebaked.col_idx.raw);
                set_buffer(&encoder, 2, &prebaked.values.raw);
                set_buffer(&encoder, 3, &x.raw);
                set_buffer(&encoder, 4, &y.raw);
                set_buffer(&encoder, 5, &row_idx_buf.raw);
                set_scalar(&encoder, 6, &count);
                if let Some(bytes) = shared_bytes {
                    unsafe {
                        encoder.setThreadgroupMemoryLength_atIndex((bytes) as usize, 0);
                    }
                }
                // For short kernel: one thread per bucketed row, grouped
                // into THREADGROUP_SIZE threads per group. For simd/cta:
                // one threadgroup per bucketed row.
                let groups = if grid_per_row == 1 {
                    (count as u64).div_ceil(tg)
                } else {
                    count as u64
                };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: (groups) as usize,
                        height: 1_usize,
                        depth: 1_usize,
                    },
                    MTLSize {
                        width: (tg) as usize,
                        height: 1_usize,
                        depth: 1_usize,
                    },
                );
                encoder.endEncoding();
            };
            // Short: thread-per-row, 256 threads/group.
            dispatch_bucket(
                short_k,
                &buckets.short,
                buckets.short_count,
                1,
                THREADGROUP_SIZE,
                None,
            );
            // Simd: one simdgroup (32 threads) per row.
            dispatch_bucket(simd_k, &buckets.simd, buckets.simd_count, 32, 32, None);
            // Cta: one full threadgroup (256 threads) per row, with
            // shared memory for the tree reduction.
            dispatch_bucket(
                cta_k,
                &buckets.cta,
                buckets.cta_count,
                THREADGROUP_SIZE,
                THREADGROUP_SIZE,
                Some(THREADGROUP_SIZE * elem_bytes),
            );
            return;
        }
        // f64 / Complex64 fall through to the un-bucketed path which itself
        // routes to host SpMV.
        self.spmv_csr::<T>(
            prebaked.m,
            prebaked.n,
            prebaked.row_ptr,
            prebaked.col_idx,
            prebaked.values,
            x,
            y,
        );
    }

    fn spmm_prebaked<T: DeviceScalar>(
        &self,
        prebaked: &Self::CsrPrebaked<'_, T>,
        nrhs: usize,
        x: &Self::Buffer<T>,
        ldx: usize,
        y: &mut Self::Buffer<T>,
        ldy: usize,
    ) {
        if let Some(error) = &prebaked.initialization_error {
            self.record_pending_error(error.clone());
            return;
        }
        if nrhs != 0 && (ldx < prebaked.n || ldy < prebaked.m) {
            self.record_pending_error(BackendError::InvalidArgument {
                operation: "spmm_prebaked",
                reason: format!(
                    "leading dimensions ({ldx}, {ldy}) are smaller than matrix ({}, {})",
                    prebaked.n, prebaked.m
                ),
            });
            return;
        }
        let x_required = match checked_strided_span("spmm_prebaked", nrhs, ldx, prebaked.n) {
            Ok(required) => required,
            Err(error) => {
                self.record_pending_error(error);
                return;
            }
        };
        let y_required = match checked_strided_span("spmm_prebaked", nrhs, ldy, prebaked.m) {
            Ok(required) => required,
            Err(error) => {
                self.record_pending_error(error);
                return;
            }
        };
        if x.len < x_required || y.len < y_required {
            self.record_pending_error(BackendError::InvalidArgument {
                operation: "spmm_prebaked",
                reason: format!("buffers are too short for nrhs={nrhs}, ldx={ldx}, ldy={ldy}"),
            });
            return;
        }
        if nrhs == 0 || prebaked.m == 0 {
            return;
        }

        let native_total = prebaked.m.checked_mul(nrhs);
        if let (Ok(_rows), Ok(nrhs_u32), Ok(ldx_u32), Ok(ldy_u32), Some(total)) = (
            u32::try_from(prebaked.m),
            u32::try_from(nrhs),
            u32::try_from(ldx),
            u32::try_from(ldy),
            native_total,
        ) && total <= u32::MAX as usize
        {
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let prebaked = unsafe {
                    &*(prebaked as *const MetalCsrPrebaked<'_, T>
                        as *const MetalCsrPrebaked<'_, f32>)
                };
                let x = unsafe { as_f32_buffer(x) };
                let y = unsafe { as_f32_buffer_mut(y) };
                self.spmm_csr_f32_native(prebaked, nrhs_u32, x, ldx_u32, y, ldy_u32);
                return;
            }
            if TypeId::of::<Complex32>() == TypeId::of::<T>() {
                let prebaked = unsafe {
                    &*(prebaked as *const MetalCsrPrebaked<'_, T>
                        as *const MetalCsrPrebaked<'_, Complex32>)
                };
                let x = unsafe { as_c32_buffer(x) };
                let y = unsafe { as_c32_buffer_mut(y) };
                self.spmm_csr_c32_native(prebaked, nrhs_u32, x, ldx_u32, y, ldy_u32);
                return;
            }
        }

        // Conservative Metal fallback: shared-storage buffers are directly
        // CPU-visible, so support the multi-RHS API without per-column GPU
        // staging. Native MSL/CuSPARSE-like SpMM remains a later perf slice.
        self.flush_pending();
        spmm_csr_host_strided(
            prebaked.m,
            prebaked.n,
            nrhs,
            prebaked.row_ptr.host_slice(),
            prebaked.col_idx.host_slice(),
            prebaked.values.host_slice(),
            x.host_slice(),
            ldx,
            y.host_slice_mut(),
            ldy,
        );
    }

    fn spmm_prebaked_colmajor<T: DeviceScalar>(
        &self,
        prebaked: &Self::CsrPrebaked<'_, T>,
        rhs: usize,
        x: &Self::Buffer<T>,
        y: &mut Self::Buffer<T>,
    ) {
        self.spmm_prebaked::<T>(prebaked, rhs, x, prebaked.n, y, prebaked.m);
    }

    fn spmv_csr<T: DeviceScalar>(
        &self,
        m: usize,
        n: usize,
        row_ptr: &Self::Buffer<i32>,
        col_idx: &Self::Buffer<i32>,
        values: &Self::Buffer<T>,
        x: &Self::Buffer<T>,
        y: &mut Self::Buffer<T>,
    ) {
        if let Err(error) = Self::check_spmv_backend_limits::<T>("spmv_csr", m) {
            self.record_pending_error(error);
            return;
        }
        let Some(expected_row_ptr_len) = m.checked_add(1) else {
            self.record_pending_error(BackendError::InvalidArgument {
                operation: "spmv_csr",
                reason: "row pointer length overflows usize".into(),
            });
            return;
        };
        if row_ptr.len != expected_row_ptr_len || x.len != n || y.len != m {
            self.record_pending_error(BackendError::InvalidArgument {
                operation: "spmv_csr",
                reason: format!(
                    "buffer lengths ({}, {}, {}) do not match matrix ({m}, {n})",
                    row_ptr.len, x.len, y.len
                ),
            });
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let values_f32 = unsafe { as_f32_buffer(values) };
            let x_f32 = unsafe { as_f32_buffer(x) };
            let y_f32 = unsafe { as_f32_buffer_mut(y) };
            self.spmv_csr_f32_native(m as u32, row_ptr, col_idx, values_f32, x_f32, y_f32);
            return;
        }
        if TypeId::of::<T>() == TypeId::of::<Complex32>() {
            let values_c32 = unsafe { as_c32_buffer(values) };
            let x_c32 = unsafe { as_c32_buffer(x) };
            let y_c32 = unsafe { as_c32_buffer_mut(y) };
            self.spmv_csr_c32_native(m as u32, row_ptr, col_idx, values_c32, x_c32, y_c32);
            return;
        }
        self.flush_pending();
        spmv_csr_host(
            m,
            row_ptr.host_slice(),
            col_idx.host_slice(),
            values.host_slice(),
            x.host_slice(),
            y.host_slice_mut(),
        );
        let _ = n;
    }
}

// ---------------------------------------------------------------------------
// Tests (CpuBackend parity)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
