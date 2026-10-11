//! CUDA explicit-dynamics kernels behind NVRTC compilation.
//!
//! Phase 1 lane 1E device counterpart to the CPU kernels in
//! [`crate::explicit_scatter`] (gather, scatter-add, lumped mass),
//! [`crate::explicit_dt`] (critical-dt min-reduce), and
//! [`crate::explicit_hourglass`] (batched HEXA8 hourglass forces). The [`CudaExplicitKernels`]
//! loader compiles `kernels/explicit_kernels.cu` with NVRTC for the device's
//! own architecture at load (mirroring the `cublas.rs` pattern of
//! `include_str!` plus device-arch compile with a default-arch fallback),
//! then launches the `sonium_explicit_*` kernels on the owning backend's
//! stream so device work stays ordered with the backend's own copies.
//!
//! Every kernel assigns one output lane per thread and accumulates that
//! lane sequentially, so device results reproduce the serial CPU kernels
//! bit-exactly: gather is a pure copy, scatter-add folds each node's slots
//! in ascending slot order onto the lane's prior value (the exact serial
//! accumulation order), the min-reduce selects over correctly-rounded
//! quotients (order-independent for the positive finite inputs the API
//! requires), and the hourglass batch replays the serial modal accumulation
//! order. NVRTC compiles with `--fmad=false` so multiply-add chains round
//! exactly like the CPU reference (no FMA contraction). Parity is proven by
//! the in-module tests, which skip gracefully when no CUDA driver is present.
//!
//! Buffer staging always flows through [`MemoryPool`](crate::pool::MemoryPool)
//! quota accounting: pool-backed [`CudaBuffer`] handles for the low-level
//! `try_*` launches, and pool-allocated staging inside the high-level
//! `*_device` convenience calls. Allocation beyond the pool cap fails fast
//! with a quota error; this module never allocates raw driver memory.
//!
//! The low-level `try_*` methods validate buffer shapes and the `u32` kernel
//! index range, but index bounds and value ranges (node ids in range,
//! positive finite lengths/speeds) must be established when the host data is
//! staged. The high-level `*_device` methods perform that validation on the
//! host slices before upload, so they are sound for any caller input; bench
//! hot loops use the low-level methods with pre-staged, pre-validated data.

// Rust guideline compliant 2026-02-21

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};

use crate::backend::{Backend, DeviceBufferLike};
use crate::cuda::{CudaBackend, CudaBuffer};
use crate::error::{BackendError, BackendResult};
use crate::pool::MemoryPool;

/// NVRTC source for the `sonium_explicit_*` kernels.
const EXPLICIT_KERNELS_CU: &str = include_str!("kernels/explicit_kernels.cu");

/// Threads per block for the min-dt block-reduce kernel.
///
/// Must stay a power of two: the kernel halves its shared-memory tile with
/// `stride = blockDim.x >> 1` down to one lane. 256 threads (8 warps) with
/// 2 KiB of shared memory per block keeps small reductions latency-bound
/// while staying far below shared-memory limits on every supported arch.
const MIN_DT_BLOCK_THREADS: u32 = 256;

/// Mode-major Gamma doubles per HEXA8 element (4 modes x 8 nodes).
const HEXA8_GAMMA_PER_ELEMENT: usize = 32;

/// Node-major doubles per HEXA8 element (8 nodes x 3 components).
const HEXA8_NODAL_PER_ELEMENT: usize = 24;

/// NVRTC `--gpu-architecture` literal for a device compute capability.
///
/// Targeting the device's own arch (instead of NVRTC's stale default, with
/// the driver JIT-ing up at load) lets NVRTC codegen select instructions
/// for the arch and skips the load-time JIT. Unknown pairs return `None`:
/// callers fall back to defaults. Mirrors the `cublas.rs` mapping.
fn explicit_nvrtc_arch(major: i32, minor: i32) -> Option<&'static str> {
    match (major, minor) {
        (5, _) => Some("compute_50"),
        (6, _) => Some("compute_60"),
        (7, 0) | (7, 1) | (7, 2) => Some("compute_70"),
        (7, _) => Some("compute_75"),
        (8, 0) => Some("compute_80"),
        (8, 6) | (8, 7) => Some("compute_86"),
        (8, _) => Some("compute_89"),
        (9, _) => Some("compute_90"),
        (10, _) => Some("compute_100"),
        (12, _) => Some("compute_120"),
        _ => None,
    }
}

/// Compile the explicit kernels for the device arch if possible.
///
/// Falls back to NVRTC defaults (plus driver JIT) when the arch is unknown
/// or the arch-specific compile fails, so a bad flag never becomes a load
/// error on its own. Only a total NVRTC failure is reported. Both paths
/// pass `--fmad=false`: without it NVRTC contracts `a * b + c` into a fused
/// multiply-add (single rounding) that the CPU reference never performs,
/// which drifts hourglass lanes by 1 ulp and breaks bit-exact parity.
fn compile_native_ptx(arch: Option<&'static str>, src: &str) -> Result<Ptx, String> {
    if let Some(arch) = arch {
        let opts = CompileOptions {
            arch: Some(arch),
            fmad: Some(false),
            ..Default::default()
        };
        if let Ok(ptx) = compile_ptx_with_opts(src, opts) {
            return Ok(ptx);
        }
    }
    let opts = CompileOptions {
        fmad: Some(false),
        ..Default::default()
    };
    compile_ptx_with_opts(src, opts).map_err(|error| format!("{error:?}"))
}

/// Checked `usize` to `u32` conversion for kernel index arguments.
fn check_u32(operation: &'static str, what: &str, value: usize) -> BackendResult<u32> {
    u32::try_from(value).map_err(|_| BackendError::InvalidArgument {
        operation,
        reason: format!("{what} {value} exceeds the CUDA kernel index range"),
    })
}

/// Borrow the device slice behind a pool-backed buffer for a launch.
fn launch_slice<'a, T: crate::scalar::BufferElement>(
    operation: &'static str,
    what: &'static str,
    buffer: &'a CudaBuffer<T>,
) -> BackendResult<&'a cudarc::driver::CudaSlice<u8>> {
    buffer.raw().ok_or_else(|| BackendError::InvalidArgument {
        operation,
        reason: format!("{what} buffer is unexpectedly empty"),
    })
}

/// Mutably borrow the device slice behind a pool-backed buffer.
fn launch_slice_mut<'a, T: crate::scalar::BufferElement>(
    operation: &'static str,
    what: &'static str,
    buffer: &'a mut CudaBuffer<T>,
) -> BackendResult<&'a mut cudarc::driver::CudaSlice<u8>> {
    buffer
        .raw_mut()
        .ok_or_else(|| BackendError::InvalidArgument {
            operation,
            reason: format!("{what} buffer is unexpectedly empty"),
        })
}

/// Build stable CSR over a scatter contribution list.
///
/// Returns `(offsets, slots)` with `offsets` of length `node_count + 1` and
/// one slot id per index entry, so node `n` owns
/// `slots[offsets[n] as usize .. offsets[n + 1] as usize]`. Slots within a
/// node stay in ascending slot order (a stable counting pass over the index
/// list), which is what makes the device scatter-add reproduce the serial
/// CPU accumulation order bit-exactly. All ids fit `i32` for the kernel.
///
/// # Examples
///
/// ```rust
/// use sonium_compute::cuda::explicit::scatter_csr;
///
/// let (offsets, slots) = scatter_csr(3, &[2, 0, 2, 1, 0]).expect("valid CSR");
/// assert_eq!(offsets, vec![0, 2, 3, 5]);
/// assert_eq!(slots, vec![1, 4, 3, 0, 2]);
/// ```
///
/// # Errors
///
/// Returns [`BackendError::InvalidArgument`] when a node id falls outside
/// `node_count` or the contribution count exceeds the `i32` kernel range.
#[must_use = "CSR build returns the offsets and slots to upload"]
pub fn scatter_csr(node_count: usize, indices: &[usize]) -> BackendResult<(Vec<i32>, Vec<i32>)> {
    const OPERATION: &str = "explicit_scatter_csr";
    if indices.len() > i32::MAX as usize {
        return Err(BackendError::InvalidArgument {
            operation: OPERATION,
            reason: format!(
                "contribution count {} exceeds the i32 kernel index range",
                indices.len()
            ),
        });
    }
    let mut counts = vec![0_i32; node_count];
    for (slot, &node) in indices.iter().enumerate() {
        if node >= node_count {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!("slot {slot} targets node {node} outside {node_count} nodes"),
            });
        }
        counts[node] += 1;
    }
    let mut offsets: Vec<i32> = Vec::with_capacity(node_count + 1);
    offsets.push(0);
    for count in &counts {
        let prev = *offsets.last().expect("offsets holds a running total");
        let next = prev
            .checked_add(*count)
            .expect("contribution count was range-checked above");
        offsets.push(next);
    }
    let mut next = offsets.clone();
    let mut slots = vec![0_i32; indices.len()];
    for (slot, &node) in indices.iter().enumerate() {
        let at = &mut next[node];
        slots[*at as usize] = i32::try_from(slot).expect("slot was range-checked above");
        *at += 1;
    }
    Ok((offsets, slots))
}

/// Cast host node ids to kernel `i32` indices with bounds checks.
///
/// # Examples
///
/// ```rust
/// use sonium_compute::cuda::explicit::cast_indices_i32;
///
/// assert_eq!(cast_indices_i32(&[1, 0], 2).expect("valid ids"), vec![1, 0]);
/// ```
///
/// # Errors
///
/// Returns [`BackendError::InvalidArgument`] when a node id falls outside
/// `node_count` or an id exceeds the `i32` kernel range.
#[must_use = "index cast returns the ids to upload"]
pub fn cast_indices_i32(indices: &[usize], node_count: usize) -> BackendResult<Vec<i32>> {
    const OPERATION: &str = "explicit_cast_indices";
    indices
        .iter()
        .enumerate()
        .map(|(slot, &node)| {
            if node >= node_count {
                return Err(BackendError::InvalidArgument {
                    operation: OPERATION,
                    reason: format!("slot {slot} targets node {node} outside {node_count} nodes"),
                });
            }
            i32::try_from(node).map_err(|_| BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!("node id {node} exceeds the i32 kernel index range"),
            })
        })
        .collect()
}

/// Block count for a min-dt reduction over `elements` quotients.
///
/// One partial per block with [`MIN_DT_BLOCK_THREADS`] threads each; empty
/// input yields zero blocks (callers reject empty reductions before launch).
///
/// # Examples
///
/// ```rust
/// use sonium_compute::cuda::explicit::min_dt_block_count;
///
/// assert_eq!(min_dt_block_count(1), 1);
/// assert_eq!(min_dt_block_count(256), 1);
/// assert_eq!(min_dt_block_count(257), 2);
/// ```
#[must_use]
pub const fn min_dt_block_count(elements: usize) -> usize {
    elements.div_ceil(MIN_DT_BLOCK_THREADS as usize)
}

/// Fold device min-dt partials into the global minimum.
///
/// Every partial holds a real quotient (the kernel reuses element `n - 1`
/// for out-of-range threads), so a plain `min` fold reproduces the serial
/// CPU reduction bit-exactly. Empty input yields `+inf`, the min identity.
///
/// # Examples
///
/// ```rust
/// use sonium_compute::cuda::explicit::finish_min_dt;
///
/// assert_eq!(finish_min_dt(&[3.0, 1.0, 2.0]), 1.0);
/// assert_eq!(finish_min_dt(&[]), f64::INFINITY);
/// ```
#[must_use]
pub fn finish_min_dt(partials: &[f64]) -> f64 {
    partials.iter().copied().fold(f64::INFINITY, f64::min)
}

/// NVRTC-compiled explicit-dynamics kernels for one CUDA backend.
///
/// Owns the loaded `sonium_explicit_*` device functions and launches them
/// on the backend's stream. Loading compiles `kernels/explicit_kernels.cu`
/// for the device's own compute capability (falling back to NVRTC defaults
/// on unknown arches); the low-level `try_*` methods launch single kernels
/// over caller-staged pool buffers, while the high-level `*_device` methods
/// validate host slices, stage them through the pool quota ledger, launch,
/// synchronize, and fetch results.
#[derive(Debug, Clone)]
pub struct CudaExplicitKernels {
    gather_f64: CudaFunction,
    scatter_add_f64: CudaFunction,
    min_dt_block_f64: CudaFunction,
    hourglass_f64: CudaFunction,
    nvrtc_arch: Option<&'static str>,
}

impl CudaExplicitKernels {
    /// Compile and load the explicit kernels for a backend.
    ///
    /// Queries the backend's compute capability, compiles the NVRTC source
    /// for that arch (or NVRTC defaults when the arch is unknown), and loads
    /// the module on the backend's context. The returned functions launch on
    /// the backend's stream via the `try_*` methods below.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use sonium_compute::cuda::explicit::CudaExplicitKernels;
    /// use sonium_compute::{CudaBackend, MemoryPool, MemoryPoolConfig};
    ///
    /// let backend = CudaBackend::new(0).expect("CUDA device");
    /// let pool =
    ///     MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).expect("pool");
    /// let kernels = CudaExplicitKernels::load(pool.backend()).expect("kernels");
    /// println!("compiled for {:?}", kernels.nvrtc_arch());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::OperationFailed`] when NVRTC compilation or
    /// module/function loading fails (driver error details in the reason).
    pub fn load(backend: &CudaBackend) -> BackendResult<Self> {
        fn load_failed(operation: &'static str, error: impl core::fmt::Debug) -> BackendError {
            BackendError::OperationFailed {
                operation,
                reason: format!("{error:?}"),
            }
        }
        let arch = backend
            .context()
            .compute_capability()
            .ok()
            .and_then(|(major, minor)| explicit_nvrtc_arch(major, minor));
        let ptx = compile_native_ptx(arch, EXPLICIT_KERNELS_CU).map_err(|reason| {
            BackendError::OperationFailed {
                operation: "explicit_nvrtc_compile",
                reason,
            }
        })?;
        let module = backend
            .context()
            .load_module(ptx)
            .map_err(|error| load_failed("explicit_module_load", error))?;
        let gather_f64 = module
            .load_function("sonium_explicit_gather_f64")
            .map_err(|error| load_failed("explicit_function_load", error))?;
        let scatter_add_f64 = module
            .load_function("sonium_explicit_scatter_add_f64")
            .map_err(|error| load_failed("explicit_function_load", error))?;
        let min_dt_block_f64 = module
            .load_function("sonium_explicit_min_dt_block_f64")
            .map_err(|error| load_failed("explicit_function_load", error))?;
        let hourglass_f64 = module
            .load_function("sonium_explicit_hourglass_f64")
            .map_err(|error| load_failed("explicit_function_load", error))?;
        Ok(Self {
            gather_f64,
            scatter_add_f64,
            min_dt_block_f64,
            hourglass_f64,
            nvrtc_arch: arch,
        })
    }

    /// NVRTC arch literal the kernels were compiled for, if known.
    ///
    /// `None` means the device arch was unknown to the mapping, so NVRTC
    /// defaults (plus driver JIT) were used instead of a native compile.
    #[must_use]
    pub fn nvrtc_arch(&self) -> Option<&'static str> {
        self.nvrtc_arch
    }

    /// Gather nodal rows into a packed device buffer.
    ///
    /// Launches `sonium_explicit_gather_f64` on the backend's stream: output
    /// slot `m` receives row `indices[m]` of `field`, matching
    /// [`crate::explicit_scatter::gather_nodal`] bit-exactly. Shapes follow
    /// the CPU layout rules (`out.len() == indices.len() * width`, nodal
    /// lengths a multiple of `width`); empty input returns without launching.
    /// Staged indices must be in range — establish that when uploading (the
    /// high-level [`Self::gather_nodal_device`] validates host ids first).
    /// The launch is asynchronous; synchronize before reading `out`.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use sonium_compute::cuda::explicit::CudaExplicitKernels;
    /// use sonium_compute::{Backend, CudaBackend, MemoryPool, MemoryPoolConfig};
    ///
    /// let backend = CudaBackend::new(0).expect("CUDA device");
    /// let pool =
    ///     MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).expect("pool");
    /// let kernels = CudaExplicitKernels::load(pool.backend()).expect("kernels");
    /// let mut field = pool.try_alloc::<f64>(6).expect("field");
    /// let mut indices = pool.try_alloc::<i32>(2).expect("indices");
    /// let mut out = pool.try_alloc::<f64>(6).expect("out");
    /// pool.backend().try_copy_h2d(&[10.0, 11.0, 12.0, 20.0, 21.0, 22.0], field.raw_mut()).expect("upload");
    /// pool.backend().try_copy_h2d(&[1, 0], indices.raw_mut()).expect("upload");
    /// kernels.try_gather(pool.backend(), field.raw(), indices.raw(), out.raw_mut(), 3).expect("gather");
    /// pool.backend().synchronize().expect("sync");
    /// let mut host = vec![0.0; 6];
    /// pool.backend().try_copy_d2h(out.raw(), &mut host).expect("fetch");
    /// assert_eq!(host, vec![20.0, 21.0, 22.0, 10.0, 11.0, 12.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidArgument`] on layout mismatch (packed
    /// length, nodal multiple, zero width, `u32` index overflow) and
    /// [`BackendError::OperationFailed`] when the driver rejects the launch.
    pub fn try_gather(
        &self,
        backend: &CudaBackend,
        field: &CudaBuffer<f64>,
        indices: &CudaBuffer<i32>,
        out: &mut CudaBuffer<f64>,
        width: usize,
    ) -> BackendResult<()> {
        const OPERATION: &str = "explicit_gather";
        if width == 0 {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: "component width must be positive".into(),
            });
        }
        let count = indices.len();
        let want_out = count
            .checked_mul(width)
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!("packed length {count} * {width} overflows"),
            })?;
        if out.len() != want_out {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "output length {} is not indices {count} * width {width}",
                    out.len()
                ),
            });
        }
        if !field.len().is_multiple_of(width) {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "field length {} is not a multiple of width {width}",
                    field.len()
                ),
            });
        }
        if count == 0 {
            return Ok(());
        }
        let width_u32 = check_u32(OPERATION, "width", width)?;
        let count_u32 = check_u32(OPERATION, "index count", count)?;
        let total_u64 = u64::try_from(count).expect("count fits u64") * u64::from(width_u32);
        let total_u32 = u32::try_from(total_u64).map_err(|_| BackendError::InvalidArgument {
            operation: OPERATION,
            reason: format!("packed length {total_u64} exceeds the CUDA kernel index range"),
        })?;
        let field_raw = launch_slice(OPERATION, "field", field)?;
        let indices_raw = launch_slice(OPERATION, "indices", indices)?;
        let out_raw = launch_slice_mut(OPERATION, "out", out)?;
        // SAFETY: the five args match the `sonium_explicit_gather_f64`
        // signature (field, indices, out, width, count); the device pointers
        // are backend-allocated and borrowed live from the caller buffers;
        // validated lengths keep every thread's computed index in bounds
        // provided staged indices are in range (established at staging
        // time); the stream and function belong to the backend's live
        // context, and the caller synchronizes before reading `out`.
        unsafe {
            backend
                .stream()
                .launch_builder(&self.gather_f64)
                .arg(field_raw)
                .arg(indices_raw)
                .arg(out_raw)
                .arg(&width_u32)
                .arg(&count_u32)
                .launch(LaunchConfig::for_num_elems(total_u32))
                .map_err(|error| BackendError::OperationFailed {
                    operation: OPERATION,
                    reason: format!("{error:?}"),
                })?;
        }
        Ok(())
    }

    /// Scatter-add packed rows into a nodal device buffer.
    ///
    /// Launches `sonium_explicit_scatter_add_f64` on the backend's stream:
    /// thread (node, c) folds that node's slots — the host-built CSR from
    /// [`scatter_csr`] — in ascending slot order onto the lane's prior
    /// value, matching [`crate::explicit_scatter::scatter_add_nodal`]
    /// bit-exactly. Requires `offsets.len() == nodes + 1` with
    /// `nodes = target.len() / width`, and
    /// `slots.len() == values.len() / width`. Empty targets return without
    /// launching. CSR validity (monotone offsets, in-range slots) must be
    /// established at staging time. The launch is asynchronous; synchronize
    /// before reading `target`.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidArgument`] on layout mismatch (CSR
    /// lengths, packed lengths, zero width, `u32` index overflow) and
    /// [`BackendError::OperationFailed`] when the driver rejects the launch.
    pub fn try_scatter_add(
        &self,
        backend: &CudaBackend,
        values: &CudaBuffer<f64>,
        slots: &CudaBuffer<i32>,
        offsets: &CudaBuffer<i32>,
        target: &mut CudaBuffer<f64>,
        width: usize,
    ) -> BackendResult<()> {
        const OPERATION: &str = "explicit_scatter_add";
        if width == 0 {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: "component width must be positive".into(),
            });
        }
        if !target.len().is_multiple_of(width) {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "target length {} is not a multiple of width {width}",
                    target.len()
                ),
            });
        }
        if !values.len().is_multiple_of(width) {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "values length {} is not a multiple of width {width}",
                    values.len()
                ),
            });
        }
        let nodes = target.len() / width;
        if offsets.len() != nodes + 1 {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "offset length {} is not node count {nodes} + 1",
                    offsets.len()
                ),
            });
        }
        if slots.len() != values.len() / width {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "slot count {} is not values {} / width {width}",
                    slots.len(),
                    values.len()
                ),
            });
        }
        if nodes == 0 {
            return Ok(());
        }
        let width_u32 = check_u32(OPERATION, "width", width)?;
        let nodes_u32 = check_u32(OPERATION, "node count", nodes)?;
        let total_u64 = u64::try_from(nodes).expect("count fits u64") * u64::from(width_u32);
        let total_u32 = u32::try_from(total_u64).map_err(|_| BackendError::InvalidArgument {
            operation: OPERATION,
            reason: format!("target length {total_u64} exceeds the CUDA kernel index range"),
        })?;
        let values_raw = launch_slice(OPERATION, "values", values)?;
        let slots_raw = launch_slice(OPERATION, "slots", slots)?;
        let offsets_raw = launch_slice(OPERATION, "offsets", offsets)?;
        let target_raw = launch_slice_mut(OPERATION, "target", target)?;
        // SAFETY: the six args match the `sonium_explicit_scatter_add_f64`
        // signature (values, slots, offsets, target, width, nodes); the
        // device pointers are backend-allocated and borrowed live from the
        // caller buffers; validated lengths keep every thread's computed
        // index in bounds provided the staged CSR is valid (established at
        // staging time); the stream and function belong to the backend's
        // live context, and the caller synchronizes before reading `target`.
        unsafe {
            backend
                .stream()
                .launch_builder(&self.scatter_add_f64)
                .arg(values_raw)
                .arg(slots_raw)
                .arg(offsets_raw)
                .arg(target_raw)
                .arg(&width_u32)
                .arg(&nodes_u32)
                .launch(LaunchConfig::for_num_elems(total_u32))
                .map_err(|error| BackendError::OperationFailed {
                    operation: OPERATION,
                    reason: format!("{error:?}"),
                })?;
        }
        Ok(())
    }

    /// Reduce per-element quotients to min-dt block partials.
    ///
    /// Launches `sonium_explicit_min_dt_block_f64` on the backend's stream:
    /// each block reduces its chunk of `lengths[i] / speeds[i]` quotients to
    /// one partial with a shared-memory min tree. Requires
    /// `partials.len() == min_dt_block_count(lengths.len())` with
    /// `lengths.len() == speeds.len() > 0`; the caller fetches the partials
    /// and finishes with [`finish_min_dt`], matching
    /// [`crate::explicit_dt::critical_dt`] bit-exactly for positive finite
    /// inputs. Positivity must be established at staging time (division by
    /// zero or NaN inputs would silently poison the minimum). The launch is
    /// asynchronous; synchronize before reading `partials`.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidArgument`] on empty input, length
    /// mismatch, a wrongly sized partials buffer, or `u32` index overflow,
    /// and [`BackendError::OperationFailed`] when the driver rejects the
    /// launch.
    pub fn try_min_dt_partials(
        &self,
        backend: &CudaBackend,
        lengths: &CudaBuffer<f64>,
        speeds: &CudaBuffer<f64>,
        partials: &mut CudaBuffer<f64>,
    ) -> BackendResult<()> {
        const OPERATION: &str = "explicit_min_dt";
        let elements = lengths.len();
        if elements == 0 {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: "min-dt reduction needs at least one element".into(),
            });
        }
        if speeds.len() != elements {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "length/speed slice mismatch: {} vs {}",
                    elements,
                    speeds.len()
                ),
            });
        }
        let blocks = min_dt_block_count(elements);
        if partials.len() != blocks {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "partials length {} is not the block count {blocks}",
                    partials.len()
                ),
            });
        }
        let n = check_u32(OPERATION, "element count", elements)?;
        let blocks_u32 = check_u32(OPERATION, "block count", blocks)?;
        let lengths_raw = launch_slice(OPERATION, "lengths", lengths)?;
        let speeds_raw = launch_slice(OPERATION, "speeds", speeds)?;
        let partials_raw = launch_slice_mut(OPERATION, "partials", partials)?;
        let config = LaunchConfig {
            grid_dim: (blocks_u32, 1, 1),
            block_dim: (MIN_DT_BLOCK_THREADS, 1, 1),
            shared_mem_bytes: MIN_DT_BLOCK_THREADS * 8,
        };
        // SAFETY: the four args match the `sonium_explicit_min_dt_block_f64`
        // signature (lengths, speeds, partials, n); the device pointers are
        // backend-allocated and borrowed live from the caller buffers; the
        // grid covers exactly `n` elements with in-range threads (out-of-
        // range threads reuse element `n - 1`, and `n > 0` was checked);
        // shared memory holds one double per thread of the power-of-two
        // block; the stream and function belong to the backend's live
        // context, and the caller synchronizes before reading `partials`.
        unsafe {
            backend
                .stream()
                .launch_builder(&self.min_dt_block_f64)
                .arg(lengths_raw)
                .arg(speeds_raw)
                .arg(partials_raw)
                .arg(&n)
                .launch(config)
                .map_err(|error| BackendError::OperationFailed {
                    operation: OPERATION,
                    reason: format!("{error:?}"),
                })?;
        }
        Ok(())
    }

    /// Evaluate batched HEXA8 hourglass forces on the device.
    ///
    /// Launches `sonium_explicit_hourglass_f64` on the backend's stream:
    /// thread `(e, i)` evaluates element `e`'s `i`-th force component lane
    /// with the exact serial accumulation order of
    /// [`crate::explicit_hourglass::hexa8_hourglass_forces`], matching it
    /// bit-exactly elementwise. Requires `gamma.len() == 32 * elements`
    /// with `rates`/`forces` at `24 * elements`; empty batches return
    /// without launching. The launch is asynchronous; synchronize before
    /// reading `forces`.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidArgument`] on layout mismatch or `u32`
    /// index overflow, and [`BackendError::OperationFailed`] when the driver
    /// rejects the launch.
    pub fn try_hourglass_forces(
        &self,
        backend: &CudaBackend,
        gamma: &CudaBuffer<f64>,
        rates: &CudaBuffer<f64>,
        forces: &mut CudaBuffer<f64>,
        coeff: f64,
        elements: usize,
    ) -> BackendResult<()> {
        const OPERATION: &str = "explicit_hourglass";
        if gamma.len() != HEXA8_GAMMA_PER_ELEMENT * elements {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "gamma length {} is not 32 * elements {elements}",
                    gamma.len()
                ),
            });
        }
        if rates.len() != HEXA8_NODAL_PER_ELEMENT * elements {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "rates length {} is not 24 * elements {elements}",
                    rates.len()
                ),
            });
        }
        if forces.len() != HEXA8_NODAL_PER_ELEMENT * elements {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "forces length {} is not 24 * elements {elements}",
                    forces.len()
                ),
            });
        }
        if elements == 0 {
            return Ok(());
        }
        let elements_u32 = check_u32(OPERATION, "element count", elements)?;
        let total_u64 = u64::from(elements_u32) * 3;
        let total_u32 = u32::try_from(total_u64).map_err(|_| BackendError::InvalidArgument {
            operation: OPERATION,
            reason: format!("lane count {total_u64} exceeds the CUDA kernel index range"),
        })?;
        let gamma_raw = launch_slice(OPERATION, "gamma", gamma)?;
        let rates_raw = launch_slice(OPERATION, "rates", rates)?;
        let forces_raw = launch_slice_mut(OPERATION, "forces", forces)?;
        // SAFETY: the five args match the `sonium_explicit_hourglass_f64`
        // signature (gamma, rates, forces, coeff, elements); the device
        // pointers are backend-allocated and borrowed live from the caller
        // buffers; validated lengths keep every thread's computed index in
        // bounds; the stream and function belong to the backend's live
        // context, and the caller synchronizes before reading `forces`.
        unsafe {
            backend
                .stream()
                .launch_builder(&self.hourglass_f64)
                .arg(gamma_raw)
                .arg(rates_raw)
                .arg(forces_raw)
                .arg(&coeff)
                .arg(&elements_u32)
                .launch(LaunchConfig::for_num_elems(total_u32))
                .map_err(|error| BackendError::OperationFailed {
                    operation: OPERATION,
                    reason: format!("{error:?}"),
                })?;
        }
        Ok(())
    }

    /// Gather nodal rows on the device from host slices.
    ///
    /// Validates the host layout (width, nodal multiple, index bounds),
    /// stages the inputs through the pool quota ledger, launches
    /// [`Self::try_gather`], synchronizes, and returns the packed output.
    /// Matches [`crate::explicit_scatter::gather_nodal`] bit-exactly.
    /// Prefer [`Self::try_gather`] with pre-staged buffers in hot loops.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use sonium_compute::cuda::explicit::CudaExplicitKernels;
    /// use sonium_compute::{CudaBackend, MemoryPool, MemoryPoolConfig};
    ///
    /// let backend = CudaBackend::new(0).expect("CUDA device");
    /// let pool =
    ///     MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).expect("pool");
    /// let kernels = CudaExplicitKernels::load(pool.backend()).expect("kernels");
    /// let out = kernels
    ///     .gather_nodal_device(&pool, &[10.0, 11.0, 12.0, 20.0, 21.0, 22.0], &[1, 0], 3)
    ///     .expect("gather");
    /// assert_eq!(out, vec![20.0, 21.0, 22.0, 10.0, 11.0, 12.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidArgument`] on layout mismatch or
    /// out-of-range indices, [`BackendError::Allocation`] when pool staging
    /// exceeds the quota, and [`BackendError::OperationFailed`] on driver
    /// launch or synchronization failure.
    pub fn gather_nodal_device(
        &self,
        pool: &MemoryPool<CudaBackend>,
        field: &[f64],
        indices: &[usize],
        width: usize,
    ) -> BackendResult<Vec<f64>> {
        const OPERATION: &str = "explicit_gather_device";
        if width == 0 {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: "component width must be positive".into(),
            });
        }
        if !field.len().is_multiple_of(width) {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "field length {} is not a multiple of width {width}",
                    field.len()
                ),
            });
        }
        let nodes = field.len() / width;
        let device_ids = cast_indices_i32(indices, nodes)?;
        let want_out = indices
            .len()
            .checked_mul(width)
            .filter(|&total| total <= u32::MAX as usize)
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "packed length {} * {width} exceeds the kernel index range",
                    indices.len()
                ),
            })?;
        if indices.is_empty() {
            return Ok(Vec::new());
        }
        let mut field_dev = pool.try_alloc::<f64>(field.len())?;
        let mut indices_dev = pool.try_alloc::<i32>(device_ids.len())?;
        let mut out_dev = pool.try_alloc::<f64>(want_out)?;
        let backend = pool.backend();
        backend.try_copy_h2d(field, field_dev.raw_mut())?;
        backend.try_copy_h2d(&device_ids, indices_dev.raw_mut())?;
        self.try_gather(
            backend,
            field_dev.raw(),
            indices_dev.raw(),
            out_dev.raw_mut(),
            width,
        )?;
        backend.synchronize()?;
        let mut out = vec![0.0; want_out];
        backend.try_copy_d2h(out_dev.raw(), &mut out)?;
        Ok(out)
    }

    /// Scatter-add packed rows on the device from host slices.
    ///
    /// Validates the host layout (width, nodal and packed multiples, index
    /// bounds), builds the stable node CSR, stages everything through the
    /// pool quota ledger, launches [`Self::try_scatter_add`], synchronizes,
    /// and writes the accumulated target back. Matches
    /// [`crate::explicit_scatter::scatter_add_nodal`] bit-exactly. Prefer
    /// [`Self::try_scatter_add`] with pre-staged buffers in hot loops.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use sonium_compute::cuda::explicit::CudaExplicitKernels;
    /// use sonium_compute::{CudaBackend, MemoryPool, MemoryPoolConfig};
    ///
    /// let backend = CudaBackend::new(0).expect("CUDA device");
    /// let pool =
    ///     MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).expect("pool");
    /// let kernels = CudaExplicitKernels::load(pool.backend()).expect("kernels");
    /// let mut target = vec![0.0; 6];
    /// kernels
    ///     .scatter_add_nodal_device(&pool, &mut target, &[0, 1, 0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2)
    ///     .expect("scatter");
    /// assert_eq!(target, vec![6.0, 8.0, 3.0, 4.0, 0.0, 0.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidArgument`] on layout mismatch or
    /// out-of-range indices, [`BackendError::Allocation`] when pool staging
    /// exceeds the quota, and [`BackendError::OperationFailed`] on driver
    /// launch or synchronization failure.
    pub fn scatter_add_nodal_device(
        &self,
        pool: &MemoryPool<CudaBackend>,
        target: &mut [f64],
        indices: &[usize],
        values: &[f64],
        width: usize,
    ) -> BackendResult<()> {
        const OPERATION: &str = "explicit_scatter_add_device";
        if width == 0 {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: "component width must be positive".into(),
            });
        }
        if !target.len().is_multiple_of(width) {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "target length {} is not a multiple of width {width}",
                    target.len()
                ),
            });
        }
        let want_values = indices
            .len()
            .checked_mul(width)
            .filter(|&total| total <= u32::MAX as usize)
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "packed length {} * {width} exceeds the kernel index range",
                    indices.len()
                ),
            })?;
        if values.len() != want_values {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "values length {} is not indices {} * width {width}",
                    values.len(),
                    indices.len()
                ),
            });
        }
        if indices.is_empty() {
            return Ok(());
        }
        let nodes = target.len() / width;
        let (offsets, slots) = scatter_csr(nodes, indices)?;
        let mut values_dev = pool.try_alloc::<f64>(values.len())?;
        let mut slots_dev = pool.try_alloc::<i32>(slots.len())?;
        let mut offsets_dev = pool.try_alloc::<i32>(offsets.len())?;
        let mut target_dev = pool.try_alloc::<f64>(target.len())?;
        let backend = pool.backend();
        backend.try_copy_h2d(values, values_dev.raw_mut())?;
        backend.try_copy_h2d(&slots, slots_dev.raw_mut())?;
        backend.try_copy_h2d(&offsets, offsets_dev.raw_mut())?;
        backend.try_copy_h2d(target, target_dev.raw_mut())?;
        self.try_scatter_add(
            backend,
            values_dev.raw(),
            slots_dev.raw(),
            offsets_dev.raw(),
            target_dev.raw_mut(),
            width,
        )?;
        backend.synchronize()?;
        backend.try_copy_d2h(target_dev.raw(), target)?;
        Ok(())
    }

    /// Compute the global critical time step on the device.
    ///
    /// Validates the host slices (non-empty, equal lengths, positive finite
    /// entries — the [`crate::explicit_dt::critical_dt`] contract, reported
    /// as errors instead of panics), stages them through the pool quota
    /// ledger, launches [`Self::try_min_dt_partials`], synchronizes, and
    /// folds the partials with [`finish_min_dt`]. Matches
    /// [`crate::explicit_dt::critical_dt`] bit-exactly.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use sonium_compute::cuda::explicit::CudaExplicitKernels;
    /// use sonium_compute::{CudaBackend, MemoryPool, MemoryPoolConfig};
    ///
    /// let backend = CudaBackend::new(0).expect("CUDA device");
    /// let pool =
    ///     MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).expect("pool");
    /// let kernels = CudaExplicitKernels::load(pool.backend()).expect("kernels");
    /// let dt = kernels
    ///     .critical_dt_device(&pool, &[0.01, 0.1, 1.0], &[5000.0, 5000.0, 5000.0])
    ///     .expect("critical dt");
    /// assert!((dt - 2.0e-6).abs() < 1.0e-18);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidArgument`] on empty or mismatched
    /// slices, non-positive or non-finite entries, or `u32` index overflow,
    /// [`BackendError::Allocation`] when pool staging exceeds the quota, and
    /// [`BackendError::OperationFailed`] on driver launch or sync failure.
    pub fn critical_dt_device(
        &self,
        pool: &MemoryPool<CudaBackend>,
        char_lengths: &[f64],
        sound_speeds: &[f64],
    ) -> BackendResult<f64> {
        const OPERATION: &str = "explicit_critical_dt_device";
        if char_lengths.is_empty() {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: "critical_dt needs at least one element".into(),
            });
        }
        if char_lengths.len() != sound_speeds.len() {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "length/sound-speed slice mismatch: {} vs {}",
                    char_lengths.len(),
                    sound_speeds.len()
                ),
            });
        }
        for (element, (&length, &speed)) in char_lengths.iter().zip(sound_speeds.iter()).enumerate()
        {
            if !length.is_finite() || length <= 0.0 {
                return Err(BackendError::InvalidArgument {
                    operation: OPERATION,
                    reason: format!(
                        "element {element} length must be positive and finite, got {length}"
                    ),
                });
            }
            if !speed.is_finite() || speed <= 0.0 {
                return Err(BackendError::InvalidArgument {
                    operation: OPERATION,
                    reason: format!(
                        "element {element} sound speed must be positive and finite, got {speed}"
                    ),
                });
            }
        }
        check_u32(OPERATION, "element count", char_lengths.len())?;
        let blocks = min_dt_block_count(char_lengths.len());
        let mut lengths_dev = pool.try_alloc::<f64>(char_lengths.len())?;
        let mut speeds_dev = pool.try_alloc::<f64>(sound_speeds.len())?;
        let mut partials_dev = pool.try_alloc::<f64>(blocks)?;
        let backend = pool.backend();
        backend.try_copy_h2d(char_lengths, lengths_dev.raw_mut())?;
        backend.try_copy_h2d(sound_speeds, speeds_dev.raw_mut())?;
        self.try_min_dt_partials(
            backend,
            lengths_dev.raw(),
            speeds_dev.raw(),
            partials_dev.raw_mut(),
        )?;
        backend.synchronize()?;
        let mut partials = vec![0.0; blocks];
        backend.try_copy_d2h(partials_dev.raw(), &mut partials)?;
        Ok(finish_min_dt(&partials))
    }

    /// Assemble lumped masses on the device from host slices.
    ///
    /// Stretch counterpart to [`crate::explicit_scatter::assemble_lumped_mass`]:
    /// validates the element topology (offsets non-empty, starting at zero,
    /// strictly increasing, inside the connectivity) and per-element inputs,
    /// expands each element mass share over its node slots, and reuses the
    /// deterministic scatter-add path at width 1 — matching the CPU kernel
    /// bit-exactly, since per-node slots preserve element order. Staging
    /// flows through the pool quota ledger with a synchronizing round trip.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use sonium_compute::cuda::explicit::CudaExplicitKernels;
    /// use sonium_compute::{CudaBackend, MemoryPool, MemoryPoolConfig};
    ///
    /// let backend = CudaBackend::new(0).expect("CUDA device");
    /// let pool =
    ///     MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).expect("pool");
    /// let kernels = CudaExplicitKernels::load(pool.backend()).expect("kernels");
    /// let mut masses = vec![0.0; 4];
    /// kernels
    ///     .assemble_lumped_mass_device(&pool, &mut masses, &[0, 1, 2, 3], &[0, 4], &[8.0], &[2.0])
    ///     .expect("lumped mass");
    /// assert_eq!(masses, vec![4.0; 4]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidArgument`] on topology or length
    /// mismatch, [`BackendError::Allocation`] when pool staging exceeds the
    /// quota, and [`BackendError::OperationFailed`] on driver launch or
    /// synchronization failure.
    pub fn assemble_lumped_mass_device(
        &self,
        pool: &MemoryPool<CudaBackend>,
        masses: &mut [f64],
        connectivity: &[usize],
        element_offsets: &[usize],
        volumes: &[f64],
        densities: &[f64],
    ) -> BackendResult<()> {
        const OPERATION: &str = "explicit_lumped_mass_device";
        if element_offsets.is_empty() {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: "element offsets must hold E + 1 entries, got an empty slice".into(),
            });
        }
        if element_offsets[0] != 0 {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "element offsets must start at 0, got {}",
                    element_offsets[0]
                ),
            });
        }
        for (element, window) in element_offsets.windows(2).enumerate() {
            let (start, end) = (window[0], window[1]);
            if end <= start {
                return Err(BackendError::InvalidArgument {
                    operation: OPERATION,
                    reason: format!("element {element} is empty"),
                });
            }
            if end > connectivity.len() {
                return Err(BackendError::InvalidArgument {
                    operation: OPERATION,
                    reason: format!(
                        "element {element} end offset {end} exceeds connectivity length {}",
                        connectivity.len()
                    ),
                });
            }
        }
        let elements = element_offsets.len() - 1;
        if volumes.len() != elements {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "volumes length {} does not match element count {elements}",
                    volumes.len()
                ),
            });
        }
        if densities.len() != elements {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "densities length {} does not match element count {elements}",
                    densities.len()
                ),
            });
        }
        let mut shares = Vec::with_capacity(connectivity.len());
        for element in 0..elements {
            let start = element_offsets[element];
            let end = element_offsets[element + 1];
            let share = densities[element] * volumes[element] / (end - start) as f64;
            shares.extend(std::iter::repeat_n(share, end - start));
        }
        self.scatter_add_nodal_device(pool, masses, connectivity, &shares, 1)
    }

    /// Evaluate batched HEXA8 hourglass forces from host slices.
    ///
    /// Stretch counterpart to [`crate::explicit_hourglass::hexa8_hourglass_forces`]:
    /// validates the batched layout (`gamma` at 32 doubles per element,
    /// `rates`/`forces` at 24, finite entries, non-negative finite
    /// coefficient — the CPU contract, reported as errors instead of panics),
    /// stages the batch through the pool quota ledger, launches
    /// [`Self::try_hourglass_forces`], synchronizes, and writes the forces
    /// back. Matches the CPU kernel bit-exactly elementwise.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use sonium_compute::cuda::explicit::CudaExplicitKernels;
    /// use sonium_compute::explicit_hourglass::HEXA8_HOURGLASS_BASE;
    /// use sonium_compute::{CudaBackend, MemoryPool, MemoryPoolConfig};
    ///
    /// let backend = CudaBackend::new(0).expect("CUDA device");
    /// let pool =
    ///     MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).expect("pool");
    /// let kernels = CudaExplicitKernels::load(pool.backend()).expect("kernels");
    /// let gamma: Vec<f64> = HEXA8_HOURGLASS_BASE.iter().flatten().copied().collect();
    /// let mut forces = vec![1.0; 24];
    /// kernels
    ///     .hexa8_hourglass_forces_device(&pool, &gamma, &[2.0; 24], 0.1, &mut forces)
    ///     .expect("hourglass");
    /// assert!(forces.iter().all(|force| *force == 0.0));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidArgument`] on layout mismatch,
    /// non-finite entries, or a negative/non-finite coefficient,
    /// [`BackendError::Allocation`] when pool staging exceeds the quota, and
    /// [`BackendError::OperationFailed`] on driver launch or sync failure.
    pub fn hexa8_hourglass_forces_device(
        &self,
        pool: &MemoryPool<CudaBackend>,
        gamma: &[f64],
        rates: &[f64],
        coeff: f64,
        forces: &mut [f64],
    ) -> BackendResult<()> {
        const OPERATION: &str = "explicit_hourglass_device";
        if !gamma.len().is_multiple_of(HEXA8_GAMMA_PER_ELEMENT) {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!("gamma length {} is not a multiple of 32", gamma.len()),
            });
        }
        let elements = gamma.len() / HEXA8_GAMMA_PER_ELEMENT;
        if rates.len() != HEXA8_NODAL_PER_ELEMENT * elements {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "rates length {} is not 24 * elements {elements}",
                    rates.len()
                ),
            });
        }
        if forces.len() != HEXA8_NODAL_PER_ELEMENT * elements {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "forces length {} is not 24 * elements {elements}",
                    forces.len()
                ),
            });
        }
        if gamma.iter().any(|value| !value.is_finite()) {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: "hourglass forces need finite Gamma".into(),
            });
        }
        if rates.iter().any(|value| !value.is_finite()) {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: "hourglass forces need finite rates".into(),
            });
        }
        if !coeff.is_finite() || coeff < 0.0 {
            return Err(BackendError::InvalidArgument {
                operation: OPERATION,
                reason: format!(
                    "hourglass forces need a non-negative finite coefficient, got {coeff}"
                ),
            });
        }
        if elements == 0 {
            return Ok(());
        }
        let mut gamma_dev = pool.try_alloc::<f64>(gamma.len())?;
        let mut rates_dev = pool.try_alloc::<f64>(rates.len())?;
        let mut forces_dev = pool.try_alloc::<f64>(forces.len())?;
        let backend = pool.backend();
        backend.try_copy_h2d(gamma, gamma_dev.raw_mut())?;
        backend.try_copy_h2d(rates, rates_dev.raw_mut())?;
        self.try_hourglass_forces(
            backend,
            gamma_dev.raw(),
            rates_dev.raw(),
            forces_dev.raw_mut(),
            coeff,
            elements,
        )?;
        backend.synchronize()?;
        backend.try_copy_d2h(forces_dev.raw(), forces)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit_dt::critical_dt;
    use crate::explicit_hourglass::{HEXA8_HOURGLASS_BASE, hexa8_hourglass_forces};
    use crate::explicit_scatter::{
        ElementTopology, assemble_lumped_mass, gather_nodal, scatter_add_nodal,
    };
    use crate::pool::MemoryPoolConfig;

    /// Serialize device tests onto one driver/NVRTC user at a time.
    ///
    /// Each test builds its own backend, but all share the primary context
    /// and the NVRTC compiler; serializing keeps parity runs deterministic
    /// and avoids loader races under the test thread pool. Poisoning is
    /// recovered (a failed test must not cascade into the others, since no
    /// shared state hides behind this lock beyond serialization itself).
    fn device_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Build a pool plus loaded kernels, or `None` without a driver.
    fn try_test_stack() -> Option<(MemoryPool<CudaBackend>, CudaExplicitKernels)> {
        let backend = CudaBackend::new(0).ok()?;
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).ok()?;
        let kernels = CudaExplicitKernels::load(pool.backend()).ok()?;
        Some((pool, kernels))
    }

    /// Deterministic fractional ramp (no RNG needed for parity checks).
    fn ramp(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| ((i as f64) * 0.0123).sin() + ((i as f64) * 0.0456).cos())
            .collect()
    }

    /// Deterministic strictly positive ramp (lengths, speeds, volumes).
    fn positive_ramp(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| ((i as f64) * 0.0177).sin() + 2.0)
            .collect()
    }

    /// Hex8 chain: element `e` owns nodes `(e + k) % nodes`, so every node
    /// is shared by 8 elements (the race case for scatter).
    fn chain_hex(nodes: usize, elements: usize) -> (Vec<usize>, Vec<usize>) {
        let mut connectivity = Vec::with_capacity(8 * elements);
        let mut offsets = Vec::with_capacity(elements + 1);
        for element in 0..elements {
            offsets.push(connectivity.len());
            for k in 0..8 {
                connectivity.push((element + k) % nodes);
            }
        }
        offsets.push(connectivity.len());
        (connectivity, offsets)
    }

    #[test]
    fn scatter_csr_is_stable_and_checked() {
        let (offsets, slots) = scatter_csr(3, &[2, 0, 2, 1, 0]).expect("valid CSR");
        assert_eq!(offsets, vec![0, 2, 3, 5]);
        assert_eq!(slots, vec![1, 4, 3, 0, 2]);
        // Empty contributions still produce a valid (all-zero) offset table.
        let (offsets, slots) = scatter_csr(4, &[]).expect("empty CSR");
        assert_eq!(offsets, vec![0; 5]);
        assert!(slots.is_empty());
        // Out-of-range node ids fail instead of poisoning device memory.
        let error = scatter_csr(2, &[0, 7]).expect_err("node 7 is out of range");
        assert!(matches!(error, BackendError::InvalidArgument { .. }));
    }

    #[test]
    fn cast_indices_rejects_out_of_range() {
        assert_eq!(cast_indices_i32(&[1, 0], 2).expect("valid ids"), vec![1, 0]);
        let error = cast_indices_i32(&[0, 2], 2).expect_err("node 2 is out of range");
        assert!(matches!(error, BackendError::InvalidArgument { .. }));
    }

    #[test]
    fn min_dt_helpers_cover_edges() {
        assert_eq!(min_dt_block_count(0), 0);
        assert_eq!(min_dt_block_count(256), 1);
        assert_eq!(min_dt_block_count(257), 2);
        assert_eq!(finish_min_dt(&[3.0, 1.0, 2.0]), 1.0);
        assert_eq!(finish_min_dt(&[]), f64::INFINITY);
    }

    #[test]
    fn loader_reports_native_arch() {
        let _guard = device_lock();
        let Some((pool, kernels)) = try_test_stack() else {
            eprintln!("no CUDA device — skipping");
            return;
        };
        let capability = pool.backend().context().compute_capability();
        eprintln!("explicit kernels NVRTC arch: {:?}", kernels.nvrtc_arch());
        eprintln!("explicit kernels compute capability: {capability:?}");
        assert!(capability.is_ok(), "compute capability must be queryable");
    }

    #[test]
    fn gather_matches_cpu_bit_exact() {
        let _guard = device_lock();
        let Some((pool, kernels)) = try_test_stack() else {
            eprintln!("no CUDA device — skipping");
            return;
        };
        let nodes = 37;
        let (connectivity, _) = chain_hex(nodes, 64);
        for width in [1, 3] {
            let field = ramp(width * nodes);
            let mut expected = vec![0.0; width * connectivity.len()];
            gather_nodal(&field, &connectivity, &mut expected);
            let got = kernels
                .gather_nodal_device(&pool, &field, &connectivity, width)
                .expect("device gather");
            assert_eq!(got, expected, "width {width}: device differs from CPU");
            // A second launch proves run-to-run determinism, not luck.
            let again = kernels
                .gather_nodal_device(&pool, &field, &connectivity, width)
                .expect("device gather repeats");
            assert_eq!(
                again, expected,
                "width {width}: device run is not deterministic"
            );
        }
    }

    #[test]
    fn scatter_add_matches_cpu_bit_exact() {
        let _guard = device_lock();
        let Some((pool, kernels)) = try_test_stack() else {
            eprintln!("no CUDA device — skipping");
            return;
        };
        let nodes = 37;
        let (connectivity, _) = chain_hex(nodes, 64);
        for width in [1, 3] {
            let values = ramp(width * connectivity.len());
            // Nonzero initial target: the kernel must fold onto it exactly.
            let mut expected = ramp(width * nodes);
            scatter_add_nodal(&mut expected, &connectivity, &values);
            let mut got = ramp(width * nodes);
            kernels
                .scatter_add_nodal_device(&pool, &mut got, &connectivity, &values, width)
                .expect("device scatter");
            assert_eq!(got, expected, "width {width}: device differs from CPU");
        }
    }

    #[test]
    fn empty_scatter_and_gather_are_noops() {
        let _guard = device_lock();
        let Some((pool, kernels)) = try_test_stack() else {
            eprintln!("no CUDA device — skipping");
            return;
        };
        let mut target = vec![1.0; 6];
        kernels
            .scatter_add_nodal_device(&pool, &mut target, &[], &[], 3)
            .expect("empty scatter is a no-op");
        assert_eq!(target, vec![1.0; 6]);
        let gathered = kernels
            .gather_nodal_device(&pool, &[1.0; 6], &[], 3)
            .expect("empty gather is a no-op");
        assert!(gathered.is_empty());
    }

    #[test]
    fn scatter_rejects_bad_layout() {
        let _guard = device_lock();
        let Some((pool, kernels)) = try_test_stack() else {
            eprintln!("no CUDA device — skipping");
            return;
        };
        let mut target = vec![0.0; 6];
        let error = kernels
            .scatter_add_nodal_device(&pool, &mut target, &[0, 1], &[1.0], 1)
            .expect_err("packed length must match the index count");
        assert!(matches!(error, BackendError::InvalidArgument { .. }));
        let error = kernels
            .scatter_add_nodal_device(&pool, &mut target, &[0, 9], &[1.0, 2.0], 1)
            .expect_err("node 9 is out of range");
        assert!(matches!(error, BackendError::InvalidArgument { .. }));
    }

    #[test]
    fn critical_dt_matches_cpu_bit_exact() {
        let _guard = device_lock();
        let Some((pool, kernels)) = try_test_stack() else {
            eprintln!("no CUDA device — skipping");
            return;
        };
        // Single element, exact block, and partial tail (17 blocks, so the
        // last block's out-of-range threads exercise the n - 1 clamp).
        for elements in [1, 256, 4097] {
            let lengths = positive_ramp(elements);
            let speeds: Vec<f64> = positive_ramp(elements)
                .iter()
                .map(|speed| speed * 1000.0)
                .collect();
            let expected = critical_dt(&lengths, &speeds);
            let got = kernels
                .critical_dt_device(&pool, &lengths, &speeds)
                .expect("device critical dt");
            assert_eq!(
                got, expected,
                "elements {elements}: device differs from CPU"
            );
        }
    }

    #[test]
    fn critical_dt_rejects_invalid_input() {
        let _guard = device_lock();
        let Some((pool, kernels)) = try_test_stack() else {
            eprintln!("no CUDA device — skipping");
            return;
        };
        for (lengths, speeds) in [
            (vec![], vec![]),
            (vec![0.1, 0.2], vec![5000.0]),
            (vec![0.1, 0.0], vec![5000.0, 5000.0]),
            (vec![0.1, 0.2], vec![5000.0, f64::NAN]),
        ] {
            let error = kernels
                .critical_dt_device(&pool, &lengths, &speeds)
                .expect_err("invalid dt input must fail");
            assert!(
                matches!(error, BackendError::InvalidArgument { .. }),
                "unexpected error for {lengths:?}: {error:?}"
            );
        }
    }

    #[test]
    fn lumped_mass_matches_cpu_bit_exact() {
        let _guard = device_lock();
        let Some((pool, kernels)) = try_test_stack() else {
            eprintln!("no CUDA device — skipping");
            return;
        };
        let nodes = 37;
        let (connectivity, offsets) = chain_hex(nodes, 64);
        let topology = ElementTopology::new(&connectivity, &offsets);
        let volumes = positive_ramp(64);
        let densities = positive_ramp(64);
        // Nonzero initial masses: assembly must accumulate onto them exactly.
        let mut expected = positive_ramp(nodes);
        assemble_lumped_mass(&mut expected, topology, &volumes, &densities);
        let mut got = positive_ramp(nodes);
        kernels
            .assemble_lumped_mass_device(
                &pool,
                &mut got,
                &connectivity,
                &offsets,
                &volumes,
                &densities,
            )
            .expect("device lumped mass");
        assert_eq!(got, expected);
    }

    #[test]
    fn staging_fails_fast_on_quota() {
        let _guard = device_lock();
        let backend = match CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(_) => {
                eprintln!("no CUDA device — skipping");
                return;
            }
        };
        let pool =
            MemoryPool::new(backend, MemoryPoolConfig::with_cap(8)).expect("tiny pool constructs");
        let error = pool
            .try_alloc::<f64>(16)
            .expect_err("16 doubles exceed an 8-byte cap");
        assert!(
            matches!(error, crate::error::OomError::QuotaExceeded { .. }),
            "unexpected quota error: {error:?}"
        );
    }

    #[test]
    fn low_level_launches_match_cpu() {
        let _guard = device_lock();
        let Some((pool, kernels)) = try_test_stack() else {
            eprintln!("no CUDA device — skipping");
            return;
        };
        let backend = pool.backend();
        // Gather over pre-staged buffers (the bench hot-loop path).
        let field = ramp(3 * 37);
        let (connectivity, _) = chain_hex(37, 64);
        let device_ids = cast_indices_i32(&connectivity, 37).expect("valid ids");
        let mut field_dev = pool.try_alloc::<f64>(field.len()).expect("field");
        let mut ids_dev = pool.try_alloc::<i32>(device_ids.len()).expect("ids");
        let mut out_dev = pool.try_alloc::<f64>(3 * connectivity.len()).expect("out");
        backend
            .try_copy_h2d(&field, field_dev.raw_mut())
            .expect("upload");
        backend
            .try_copy_h2d(&device_ids, ids_dev.raw_mut())
            .expect("upload");
        kernels
            .try_gather(
                backend,
                field_dev.raw(),
                ids_dev.raw(),
                out_dev.raw_mut(),
                3,
            )
            .expect("gather launch");
        backend.synchronize().expect("sync");
        let mut got = vec![0.0; 3 * connectivity.len()];
        backend
            .try_copy_d2h(out_dev.raw(), &mut got)
            .expect("fetch");
        let mut expected = vec![0.0; 3 * connectivity.len()];
        gather_nodal(&field, &connectivity, &mut expected);
        assert_eq!(got, expected);

        // Scatter-add over pre-staged buffers.
        let values = ramp(3 * connectivity.len());
        let (offsets, slots) = scatter_csr(37, &connectivity).expect("valid CSR");
        let mut values_dev = pool.try_alloc::<f64>(values.len()).expect("values");
        let mut slots_dev = pool.try_alloc::<i32>(slots.len()).expect("slots");
        let mut offsets_dev = pool.try_alloc::<i32>(offsets.len()).expect("offsets");
        let mut target_dev = pool.try_alloc::<f64>(3 * 37).expect("target");
        let initial = ramp(3 * 37);
        backend
            .try_copy_h2d(&values, values_dev.raw_mut())
            .expect("upload");
        backend
            .try_copy_h2d(&slots, slots_dev.raw_mut())
            .expect("upload");
        backend
            .try_copy_h2d(&offsets, offsets_dev.raw_mut())
            .expect("upload");
        backend
            .try_copy_h2d(&initial, target_dev.raw_mut())
            .expect("upload");
        kernels
            .try_scatter_add(
                backend,
                values_dev.raw(),
                slots_dev.raw(),
                offsets_dev.raw(),
                target_dev.raw_mut(),
                3,
            )
            .expect("scatter launch");
        backend.synchronize().expect("sync");
        let mut got = vec![0.0; 3 * 37];
        backend
            .try_copy_d2h(target_dev.raw(), &mut got)
            .expect("fetch");
        let mut expected = initial;
        scatter_add_nodal(&mut expected, &connectivity, &values);
        assert_eq!(got, expected);

        // Min-dt partials over pre-staged buffers.
        let lengths = positive_ramp(1000);
        let speeds: Vec<f64> = positive_ramp(1000)
            .iter()
            .map(|speed| speed * 1000.0)
            .collect();
        let blocks = min_dt_block_count(lengths.len());
        let mut lengths_dev = pool.try_alloc::<f64>(lengths.len()).expect("lengths");
        let mut speeds_dev = pool.try_alloc::<f64>(speeds.len()).expect("speeds");
        let mut partials_dev = pool.try_alloc::<f64>(blocks).expect("partials");
        backend
            .try_copy_h2d(&lengths, lengths_dev.raw_mut())
            .expect("upload");
        backend
            .try_copy_h2d(&speeds, speeds_dev.raw_mut())
            .expect("upload");
        kernels
            .try_min_dt_partials(
                backend,
                lengths_dev.raw(),
                speeds_dev.raw(),
                partials_dev.raw_mut(),
            )
            .expect("min-dt launch");
        backend.synchronize().expect("sync");
        let mut partials = vec![0.0; blocks];
        backend
            .try_copy_d2h(partials_dev.raw(), &mut partials)
            .expect("fetch");
        assert_eq!(finish_min_dt(&partials), critical_dt(&lengths, &speeds));
    }

    #[test]
    fn hourglass_matches_cpu_bit_exact() {
        let _guard = device_lock();
        let Some((pool, kernels)) = try_test_stack() else {
            eprintln!("no CUDA device — skipping");
            return;
        };
        let base: Vec<f64> = HEXA8_HOURGLASS_BASE.iter().flatten().copied().collect();
        assert_eq!(base.len(), 32);
        let elements = 65;
        let mut gamma = Vec::with_capacity(32 * elements);
        for element in 0..elements {
            let scale = 1.0 + 0.01 * element as f64;
            gamma.extend(base.iter().map(|value| value * scale));
        }
        let rates = ramp(24 * elements);
        let coeff = 0.1;
        let mut expected = vec![0.0; 24 * elements];
        for element in 0..elements {
            hexa8_hourglass_forces(
                &gamma[32 * element..32 * element + 32],
                &rates[24 * element..24 * element + 24],
                coeff,
                &mut expected[24 * element..24 * element + 24],
            );
        }
        let mut got = vec![0.0; 24 * elements];
        kernels
            .hexa8_hourglass_forces_device(&pool, &gamma, &rates, coeff, &mut got)
            .expect("device hourglass");
        assert_eq!(got, expected);
        // Rigid translation carries no hourglass force, exactly.
        let mut zeros = vec![1.0; 24];
        kernels
            .hexa8_hourglass_forces_device(&pool, &base, &[2.0; 24], coeff, &mut zeros)
            .expect("device hourglass on rigid rates");
        assert!(zeros.iter().all(|force| *force == 0.0));
    }

    #[test]
    fn hourglass_rejects_invalid_input() {
        let _guard = device_lock();
        let Some((pool, kernels)) = try_test_stack() else {
            eprintln!("no CUDA device — skipping");
            return;
        };
        let base: Vec<f64> = HEXA8_HOURGLASS_BASE.iter().flatten().copied().collect();
        let mut forces = vec![0.0; 24];
        // Bad gamma length.
        let error = kernels
            .hexa8_hourglass_forces_device(&pool, &base[..31], &[0.0; 24], 0.1, &mut forces)
            .expect_err("gamma must come in 32s");
        assert!(matches!(error, BackendError::InvalidArgument { .. }));
        // Mismatched rates.
        let error = kernels
            .hexa8_hourglass_forces_device(&pool, &base, &[0.0; 23], 0.1, &mut forces)
            .expect_err("rates must come in 24s");
        assert!(matches!(error, BackendError::InvalidArgument { .. }));
        // Non-finite Gamma.
        let mut bad_gamma = base.clone();
        bad_gamma[7] = f64::INFINITY;
        let error = kernels
            .hexa8_hourglass_forces_device(&pool, &bad_gamma, &[0.0; 24], 0.1, &mut forces)
            .expect_err("Gamma must be finite");
        assert!(matches!(error, BackendError::InvalidArgument { .. }));
        // Negative coefficient.
        let error = kernels
            .hexa8_hourglass_forces_device(&pool, &base, &[0.0; 24], -0.1, &mut forces)
            .expect_err("coefficient must be non-negative");
        assert!(matches!(error, BackendError::InvalidArgument { .. }));
    }
}
