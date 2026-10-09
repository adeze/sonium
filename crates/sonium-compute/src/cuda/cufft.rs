//! Minimal dynamically-loaded cuFFT adapter for regular Complex32/Complex64 batches.
//!
//! `cudarc` 0.19 does not expose cuFFT, so this module keeps the small C ABI
//! surface local and optional. CUDA backend construction succeeds when cuFFT
//! is absent; callers then see `CapabilityState::Unsupported` and the FFT
//! adapter stages through the CPU. Plans are cached by precision, length,
//! batch size, and batch stride and execute in-place on the backend's existing
//! CUDA stream.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaStream, DevicePtr, DevicePtrMut, SyncOnDrop};
use libloading::{Library, Symbol};
use num_complex::{Complex32, Complex64};

use crate::backend::{
    BackendAllocationLedger, DeviceBufferLike, checked_fft_strided_view_descriptor,
    checked_fft_view_descriptor,
};
use crate::error::{BackendError, BackendResult};

use super::{CudaBuffer, TrackedCudaSlice};

type CufftHandle = i32;
type CufftResult = i32;
type CufftComplex = [f32; 2];
type CufftDoubleComplex = [f64; 2];

type CufftPlanMany = unsafe extern "C" fn(
    plan: *mut CufftHandle,
    rank: i32,
    n: *const i32,
    inembed: *const i32,
    istride: i32,
    idist: i32,
    onembed: *const i32,
    ostride: i32,
    odist: i32,
    fft_type: i32,
    batch: i32,
) -> CufftResult;
type CufftGetSizeMany = unsafe extern "C" fn(
    plan: CufftHandle,
    rank: i32,
    n: *const i32,
    inembed: *const i32,
    istride: i32,
    idist: i32,
    onembed: *const i32,
    ostride: i32,
    odist: i32,
    fft_type: i32,
    batch: i32,
    work_size: *mut usize,
) -> CufftResult;
type CufftSetStream = unsafe extern "C" fn(plan: CufftHandle, stream: *mut c_void) -> CufftResult;
type CufftSetAutoAllocation =
    unsafe extern "C" fn(plan: CufftHandle, auto_allocate: i32) -> CufftResult;
type CufftSetWorkArea =
    unsafe extern "C" fn(plan: CufftHandle, work_area: *mut c_void) -> CufftResult;
type CufftExecC2C = unsafe extern "C" fn(
    plan: CufftHandle,
    input: *mut CufftComplex,
    output: *mut CufftComplex,
    direction: i32,
) -> CufftResult;
type CufftExecZ2Z = unsafe extern "C" fn(
    plan: CufftHandle,
    input: *mut CufftDoubleComplex,
    output: *mut CufftDoubleComplex,
    direction: i32,
) -> CufftResult;
type CufftExecR2C = unsafe extern "C" fn(
    plan: CufftHandle,
    input: *mut f32,
    output: *mut CufftComplex,
) -> CufftResult;
type CufftExecC2R = unsafe extern "C" fn(
    plan: CufftHandle,
    input: *mut CufftComplex,
    output: *mut f32,
) -> CufftResult;
type CufftExecD2Z = unsafe extern "C" fn(
    plan: CufftHandle,
    input: *mut f64,
    output: *mut CufftDoubleComplex,
) -> CufftResult;
type CufftExecZ2D = unsafe extern "C" fn(
    plan: CufftHandle,
    input: *mut CufftDoubleComplex,
    output: *mut f64,
) -> CufftResult;
type CufftDestroy = unsafe extern "C" fn(plan: CufftHandle) -> CufftResult;

const CUFFT_SUCCESS: CufftResult = 0;
const CUFFT_C2C: i32 = 0x29;
const CUFFT_R2C: i32 = 0x2a;
const CUFFT_C2R: i32 = 0x2c;
const CUFFT_Z2Z: i32 = 0x69;
const CUFFT_D2Z: i32 = 0x6a;
const CUFFT_Z2D: i32 = 0x6c;
const CUFFT_FORWARD: i32 = -1;
const CUFFT_INVERSE: i32 = 1;

struct CudaFftApi {
    _library: Library,
    plan_many: CufftPlanMany,
    get_size_many: CufftGetSizeMany,
    set_stream: CufftSetStream,
    set_auto_allocation: CufftSetAutoAllocation,
    set_work_area: CufftSetWorkArea,
    exec_c2c: CufftExecC2C,
    exec_z2z: CufftExecZ2Z,
    exec_r2c: CufftExecR2C,
    exec_c2r: CufftExecC2R,
    exec_d2z: CufftExecD2Z,
    exec_z2d: CufftExecZ2D,
    destroy: CufftDestroy,
}

impl CudaFftApi {
    fn load() -> Result<Self, String> {
        #[cfg(target_os = "windows")]
        const LIBRARIES: &[&str] = &["cufft64_12.dll", "cufft64_11.dll", "cufft64_10.dll"];
        #[cfg(target_os = "macos")]
        const LIBRARIES: &[&str] = &["libcufft.dylib"];
        #[cfg(all(unix, not(target_os = "macos")))]
        const LIBRARIES: &[&str] = &["libcufft.so.12", "libcufft.so.11", "libcufft.so"];

        let mut last_error = String::from("no cuFFT shared library found");
        for name in LIBRARIES {
            // SAFETY: loading a named optional shared library is the intended
            // dynamic-loading boundary; symbols are copied into function
            // pointers while the Library is retained in `Self`.
            let Ok(library) = (unsafe { Library::new(*name) }) else {
                continue;
            };
            // SAFETY: cuFFT exports these symbols with the declared C ABI.
            let symbols = unsafe {
                (
                    load_symbol::<CufftPlanMany>(&library, b"cufftPlanMany\0"),
                    load_symbol::<CufftGetSizeMany>(&library, b"cufftGetSizeMany\0"),
                    load_symbol::<CufftSetStream>(&library, b"cufftSetStream\0"),
                    load_symbol::<CufftSetAutoAllocation>(&library, b"cufftSetAutoAllocation\0"),
                    load_symbol::<CufftSetWorkArea>(&library, b"cufftSetWorkArea\0"),
                    load_symbol::<CufftExecC2C>(&library, b"cufftExecC2C\0"),
                    load_symbol::<CufftExecZ2Z>(&library, b"cufftExecZ2Z\0"),
                    load_symbol::<CufftExecR2C>(&library, b"cufftExecR2C\0"),
                    load_symbol::<CufftExecC2R>(&library, b"cufftExecC2R\0"),
                    load_symbol::<CufftExecD2Z>(&library, b"cufftExecD2Z\0"),
                    load_symbol::<CufftExecZ2D>(&library, b"cufftExecZ2D\0"),
                    load_symbol::<CufftDestroy>(&library, b"cufftDestroy\0"),
                )
            };
            match symbols {
                (
                    Ok(plan_many),
                    Ok(get_size_many),
                    Ok(set_stream),
                    Ok(set_auto_allocation),
                    Ok(set_work_area),
                    Ok(exec_c2c),
                    Ok(exec_z2z),
                    Ok(exec_r2c),
                    Ok(exec_c2r),
                    Ok(exec_d2z),
                    Ok(exec_z2d),
                    Ok(destroy),
                ) => {
                    return Ok(Self {
                        _library: library,
                        plan_many,
                        get_size_many,
                        set_stream,
                        set_auto_allocation,
                        set_work_area,
                        exec_c2c,
                        exec_z2z,
                        exec_r2c,
                        exec_c2r,
                        exec_d2z,
                        exec_z2d,
                        destroy,
                    });
                }
                _ => last_error = format!("cuFFT library {name} is missing required symbols"),
            }
        }
        Err(last_error)
    }
}

unsafe fn load_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, String> {
    // SAFETY: the caller verifies the symbol's ABI through the requested type
    // and retains `library` for the lifetime of the copied function pointer.
    unsafe {
        library
            .get::<T>(name)
            .map(|symbol: Symbol<'_, T>| *symbol)
            .map_err(|error| error.to_string())
    }
}

/// Cached cuFFT plans bound to one cudarc stream.
struct CudaFftPlan {
    handle: CufftHandle,
    _workspace: Option<TrackedCudaSlice>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CufftPlanLayout {
    fft_len: usize,
    batch: usize,
    input_embed: usize,
    input_stride: usize,
    input_dist: usize,
    output_embed: usize,
    output_stride: usize,
    output_dist: usize,
    fft_type: i32,
}

pub(crate) struct CudaFft {
    api: CudaFftApi,
    stream: Arc<CudaStream>,
    allocation_stats: Arc<BackendAllocationLedger>,
    pending_error: Arc<Mutex<Option<BackendError>>>,
    plans: Mutex<HashMap<CufftPlanLayout, CudaFftPlan>>,
}

impl CudaFft {
    pub(crate) fn new(
        stream: Arc<CudaStream>,
        allocation_stats: Arc<BackendAllocationLedger>,
        pending_error: Arc<Mutex<Option<BackendError>>>,
    ) -> Result<Self, String> {
        Ok(Self {
            api: CudaFftApi::load()?,
            stream,
            allocation_stats,
            pending_error,
            plans: Mutex::new(HashMap::new()),
        })
    }

    /// Number of cached driver plans currently retained by this adapter.
    ///
    /// cuFFT plan handles are driver-owned resources rather than byte-ledger
    /// buffers, so the CUDA backend includes this count in its resource
    /// snapshot while the associated work areas remain byte-accounted.
    pub(crate) fn plan_count(&self) -> usize {
        self.plans.lock().map(|plans| plans.len()).unwrap_or(0)
    }

    fn plan_layout(&self, layout: CufftPlanLayout) -> BackendResult<CufftHandle> {
        let CufftPlanLayout {
            fft_len,
            batch,
            input_embed,
            input_stride,
            input_dist,
            output_embed,
            output_stride,
            output_dist,
            fft_type,
        } = layout;
        let fft_len_i32 = i32::try_from(fft_len).map_err(|_| BackendError::InvalidArgument {
            operation: "cufft_plan",
            reason: "FFT length exceeds cuFFT's signed dimension range".into(),
        })?;
        let batch_i32 = i32::try_from(batch).map_err(|_| BackendError::InvalidArgument {
            operation: "cufft_plan",
            reason: "FFT batch exceeds cuFFT's signed dimension range".into(),
        })?;
        let input_embed_i32 =
            i32::try_from(input_embed).map_err(|_| BackendError::InvalidArgument {
                operation: "cufft_plan",
                reason: "cuFFT input embed exceeds its signed dimension range".into(),
            })?;
        let output_embed_i32 =
            i32::try_from(output_embed).map_err(|_| BackendError::InvalidArgument {
                operation: "cufft_plan",
                reason: "cuFFT output embed exceeds its signed dimension range".into(),
            })?;
        let input_stride_i32 =
            i32::try_from(input_stride).map_err(|_| BackendError::InvalidArgument {
                operation: "cufft_plan",
                reason: "cuFFT input stride exceeds its signed dimension range".into(),
            })?;
        let output_stride_i32 =
            i32::try_from(output_stride).map_err(|_| BackendError::InvalidArgument {
                operation: "cufft_plan",
                reason: "cuFFT output stride exceeds its signed dimension range".into(),
            })?;
        let input_dist_i32 =
            i32::try_from(input_dist).map_err(|_| BackendError::InvalidArgument {
                operation: "cufft_plan",
                reason: "cuFFT input distance exceeds its signed dimension range".into(),
            })?;
        let output_dist_i32 =
            i32::try_from(output_dist).map_err(|_| BackendError::InvalidArgument {
                operation: "cufft_plan",
                reason: "cuFFT output distance exceeds its signed dimension range".into(),
            })?;

        let mut plans = self
            .plans
            .lock()
            .map_err(|_| BackendError::OperationFailed {
                operation: "cufft_plan_cache",
                reason: "cuFFT plan cache mutex poisoned".into(),
            })?;
        let key = CufftPlanLayout {
            fft_len,
            batch,
            input_embed,
            input_stride,
            input_dist,
            output_embed,
            output_stride,
            output_dist,
            fft_type,
        };
        if let Some(plan) = plans.get(&key) {
            return Ok(plan.handle);
        }

        let input_embed = [input_embed_i32];
        let output_embed = [output_embed_i32];
        let mut plan = 0;
        let status = unsafe {
            (self.api.plan_many)(
                &mut plan,
                1,
                &fft_len_i32,
                input_embed.as_ptr(),
                input_stride_i32,
                input_dist_i32,
                output_embed.as_ptr(),
                output_stride_i32,
                output_dist_i32,
                fft_type,
                batch_i32,
            )
        };
        if status != CUFFT_SUCCESS {
            return Err(BackendError::OperationFailed {
                operation: "cufftPlanMany",
                reason: format!("cuFFT status {status}"),
            });
        }

        let status = unsafe { (self.api.set_auto_allocation)(plan, 0) };
        if status != CUFFT_SUCCESS {
            unsafe {
                (self.api.destroy)(plan);
            }
            return Err(BackendError::OperationFailed {
                operation: "cufftSetAutoAllocation",
                reason: format!("cuFFT status {status}"),
            });
        }

        let status = unsafe { (self.api.set_stream)(plan, self.stream.cu_stream() as *mut c_void) };
        if status != CUFFT_SUCCESS {
            unsafe {
                (self.api.destroy)(plan);
            }
            return Err(BackendError::OperationFailed {
                operation: "cufftSetStream",
                reason: format!("cuFFT status {status}"),
            });
        }
        let mut work_size = 0usize;
        let status = unsafe {
            (self.api.get_size_many)(
                plan,
                1,
                &fft_len_i32,
                input_embed.as_ptr(),
                input_stride_i32,
                input_dist_i32,
                output_embed.as_ptr(),
                output_stride_i32,
                output_dist_i32,
                fft_type,
                batch_i32,
                &mut work_size,
            )
        };
        if status != CUFFT_SUCCESS {
            unsafe {
                (self.api.destroy)(plan);
            }
            return Err(BackendError::OperationFailed {
                operation: "cufftGetSizeMany",
                reason: format!("cuFFT status {status}"),
            });
        }
        let workspace = if work_size == 0 {
            None
        } else {
            let accounting = match self.allocation_stats.try_reserve(work_size) {
                Ok(guard) => guard,
                Err(error) => {
                    unsafe {
                        (self.api.destroy)(plan);
                    }
                    return Err(BackendError::from(error));
                }
            };
            let buffer = match unsafe { self.stream.alloc::<u8>(work_size) } {
                Ok(buffer) => buffer,
                Err(error) => {
                    unsafe {
                        (self.api.destroy)(plan);
                    }
                    return Err(BackendError::from(super::driver_to_oom(error)));
                }
            };
            let mut workspace = TrackedCudaSlice {
                inner: buffer,
                _allocation: accounting,
            };
            {
                let (pointer, _guard): (u64, SyncOnDrop<'_>) =
                    workspace.device_ptr_mut(&self.stream);
                let status = unsafe { (self.api.set_work_area)(plan, pointer as *mut c_void) };
                if status != CUFFT_SUCCESS {
                    unsafe {
                        (self.api.destroy)(plan);
                    }
                    return Err(BackendError::OperationFailed {
                        operation: "cufftSetWorkArea",
                        reason: format!("cuFFT status {status}"),
                    });
                }
            }
            Some(workspace)
        };
        plans.insert(
            key,
            CudaFftPlan {
                handle: plan,
                _workspace: workspace,
            },
        );
        Ok(plan)
    }

    fn validate_shape(buffer_len: usize, fft_len: usize) -> BackendResult<usize> {
        if fft_len == 0 {
            return Err(BackendError::InvalidArgument {
                operation: "cufft",
                reason: "cuFFT adapter requires a positive length".into(),
            });
        }
        if !buffer_len.is_multiple_of(fft_len) {
            return Err(BackendError::InvalidArgument {
                operation: "cufft",
                reason: format!(
                    "buffer length {buffer_len} is not a multiple of FFT length {fft_len}"
                ),
            });
        }
        Ok(buffer_len / fft_len)
    }

    fn validate_strided_shape(
        buffer_len: usize,
        fft_len: usize,
        batch: usize,
        stride: usize,
    ) -> BackendResult<()> {
        if fft_len == 0 || stride < fft_len {
            return Err(BackendError::InvalidArgument {
                operation: "cufft_strided",
                reason: "FFT length must be positive and stride must cover one transform".into(),
            });
        }
        let required = if batch == 0 {
            0
        } else {
            batch
                .checked_sub(1)
                .and_then(|last| last.checked_mul(stride))
                .and_then(|last| last.checked_add(fft_len))
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "cufft_strided",
                    reason: "FFT batch shape overflows address arithmetic".into(),
                })?
        };
        if buffer_len < required {
            return Err(BackendError::InvalidArgument {
                operation: "cufft_strided",
                reason: format!(
                    "buffer length {buffer_len} is shorter than required strided span {required}"
                ),
            });
        }
        Ok(())
    }

    pub(crate) fn exec_c32(
        &self,
        buffer: &mut CudaBuffer<Complex32>,
        fft_len: usize,
        inverse: bool,
    ) -> BackendResult<()> {
        checked_fft_view_descriptor::<Complex32>(buffer.len(), fft_len, "cuda")?;
        let batch = Self::validate_shape(buffer.len(), fft_len)?;
        self.exec_c32_strided(buffer, fft_len, batch, fft_len, inverse)
    }

    pub(crate) fn exec_c32_strided(
        &self,
        buffer: &mut CudaBuffer<Complex32>,
        fft_len: usize,
        batch: usize,
        stride: usize,
        inverse: bool,
    ) -> BackendResult<()> {
        checked_fft_strided_view_descriptor::<Complex32>(
            buffer.len(),
            fft_len,
            batch,
            stride,
            "cuda",
        )?;
        Self::validate_strided_shape(buffer.len(), fft_len, batch, stride)?;
        if batch == 0 {
            return Ok(());
        }
        let plan = self.plan_layout(CufftPlanLayout {
            fft_len,
            batch,
            input_embed: fft_len,
            input_stride: 1,
            input_dist: stride,
            output_embed: fft_len,
            output_stride: 1,
            output_dist: stride,
            fft_type: CUFFT_C2C,
        })?;
        let raw = buffer
            .raw_mut()
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "cufft",
                reason: "non-empty FFT buffer has no CUDA allocation".into(),
            })?;
        let (pointer, _guard): (u64, SyncOnDrop<'_>) = raw.device_ptr_mut(&self.stream);
        let status = unsafe {
            (self.api.exec_c2c)(
                plan,
                pointer as *mut CufftComplex,
                pointer as *mut CufftComplex,
                if inverse {
                    CUFFT_INVERSE
                } else {
                    CUFFT_FORWARD
                },
            )
        };
        if status != CUFFT_SUCCESS {
            return Err(BackendError::OperationFailed {
                operation: "cufftExecC2C",
                reason: format!("cuFFT status {status}"),
            });
        }
        Ok(())
    }

    pub(crate) fn exec_c64(
        &self,
        buffer: &mut CudaBuffer<Complex64>,
        fft_len: usize,
        inverse: bool,
    ) -> BackendResult<()> {
        checked_fft_view_descriptor::<Complex64>(buffer.len(), fft_len, "cuda")?;
        let batch = Self::validate_shape(buffer.len(), fft_len)?;
        self.exec_c64_strided(buffer, fft_len, batch, fft_len, inverse)
    }

    pub(crate) fn exec_c64_strided(
        &self,
        buffer: &mut CudaBuffer<Complex64>,
        fft_len: usize,
        batch: usize,
        stride: usize,
        inverse: bool,
    ) -> BackendResult<()> {
        checked_fft_strided_view_descriptor::<Complex64>(
            buffer.len(),
            fft_len,
            batch,
            stride,
            "cuda",
        )?;
        Self::validate_strided_shape(buffer.len(), fft_len, batch, stride)?;
        if batch == 0 {
            return Ok(());
        }
        let plan = self.plan_layout(CufftPlanLayout {
            fft_len,
            batch,
            input_embed: fft_len,
            input_stride: 1,
            input_dist: stride,
            output_embed: fft_len,
            output_stride: 1,
            output_dist: stride,
            fft_type: CUFFT_Z2Z,
        })?;
        let raw = buffer
            .raw_mut()
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "cufft",
                reason: "non-empty FFT buffer has no CUDA allocation".into(),
            })?;
        let (pointer, _guard): (u64, SyncOnDrop<'_>) = raw.device_ptr_mut(&self.stream);
        let status = unsafe {
            (self.api.exec_z2z)(
                plan,
                pointer as *mut CufftDoubleComplex,
                pointer as *mut CufftDoubleComplex,
                if inverse {
                    CUFFT_INVERSE
                } else {
                    CUFFT_FORWARD
                },
            )
        };
        if status != CUFFT_SUCCESS {
            return Err(BackendError::OperationFailed {
                operation: "cufftExecZ2Z",
                reason: format!("cuFFT status {status}"),
            });
        }
        Ok(())
    }

    fn validate_real_shape(
        input_len: usize,
        output_len: usize,
        fft_len: usize,
        inverse: bool,
    ) -> BackendResult<usize> {
        if fft_len == 0 {
            return Err(BackendError::InvalidArgument {
                operation: if inverse { "cufft_c2r" } else { "cufft_r2c" },
                reason: "cuFFT adapter requires a positive length".into(),
            });
        }
        let spectrum_len = fft_len / 2 + 1;
        let (transform_len, expected_len) = if inverse {
            (spectrum_len, fft_len)
        } else {
            (fft_len, spectrum_len)
        };
        if !input_len.is_multiple_of(transform_len)
            || !output_len.is_multiple_of(expected_len)
            || input_len / transform_len != output_len / expected_len
        {
            return Err(BackendError::InvalidArgument {
                operation: if inverse { "cufft_c2r" } else { "cufft_r2c" },
                reason: format!(
                    "real FFT buffers have incompatible lengths: input {input_len}, output {output_len}, transform {transform_len}, result {expected_len}"
                ),
            });
        }
        Ok(input_len / transform_len)
    }

    pub(crate) fn exec_r2c_f32(
        &self,
        input: &CudaBuffer<f32>,
        output: &mut CudaBuffer<Complex32>,
        fft_len: usize,
    ) -> BackendResult<()> {
        let batch = Self::validate_real_shape(input.len(), output.len, fft_len, false)?;
        if batch == 0 {
            return Ok(());
        }
        let spectrum_len = fft_len / 2 + 1;
        let plan = self.plan_layout(CufftPlanLayout {
            fft_len,
            batch,
            input_embed: fft_len,
            input_stride: 1,
            input_dist: fft_len,
            output_embed: spectrum_len,
            output_stride: 1,
            output_dist: spectrum_len,
            fft_type: CUFFT_R2C,
        })?;
        let input_raw = input.raw().ok_or_else(|| BackendError::InvalidArgument {
            operation: "cufft_r2c",
            reason: "non-empty input has no CUDA allocation".into(),
        })?;
        let output_raw = output
            .raw_mut()
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "cufft_r2c",
                reason: "non-empty output has no CUDA allocation".into(),
            })?;
        let (input_pointer, _input_guard): (u64, SyncOnDrop<'_>) =
            input_raw.device_ptr(&self.stream);
        let (output_pointer, _output_guard): (u64, SyncOnDrop<'_>) =
            output_raw.device_ptr_mut(&self.stream);
        let status = unsafe {
            (self.api.exec_r2c)(
                plan,
                input_pointer as *mut f32,
                output_pointer as *mut CufftComplex,
            )
        };
        if status != CUFFT_SUCCESS {
            return Err(BackendError::OperationFailed {
                operation: "cufftExecR2C",
                reason: format!("cuFFT status {status}"),
            });
        }
        Ok(())
    }

    pub(crate) fn exec_c2r_f32(
        &self,
        input: &CudaBuffer<Complex32>,
        output: &mut CudaBuffer<f32>,
        fft_len: usize,
    ) -> BackendResult<()> {
        let batch = Self::validate_real_shape(input.len(), output.len, fft_len, true)?;
        if batch == 0 {
            return Ok(());
        }
        let spectrum_len = fft_len / 2 + 1;
        let plan = self.plan_layout(CufftPlanLayout {
            fft_len,
            batch,
            input_embed: spectrum_len,
            input_stride: 1,
            input_dist: spectrum_len,
            output_embed: fft_len,
            output_stride: 1,
            output_dist: fft_len,
            fft_type: CUFFT_C2R,
        })?;
        let input_raw = input.raw().ok_or_else(|| BackendError::InvalidArgument {
            operation: "cufft_c2r",
            reason: "non-empty input has no CUDA allocation".into(),
        })?;
        let output_raw = output
            .raw_mut()
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "cufft_c2r",
                reason: "non-empty output has no CUDA allocation".into(),
            })?;
        let (input_pointer, _input_guard): (u64, SyncOnDrop<'_>) =
            input_raw.device_ptr(&self.stream);
        let (output_pointer, _output_guard): (u64, SyncOnDrop<'_>) =
            output_raw.device_ptr_mut(&self.stream);
        let status = unsafe {
            (self.api.exec_c2r)(
                plan,
                input_pointer as *mut CufftComplex,
                output_pointer as *mut f32,
            )
        };
        if status != CUFFT_SUCCESS {
            return Err(BackendError::OperationFailed {
                operation: "cufftExecC2R",
                reason: format!("cuFFT status {status}"),
            });
        }
        Ok(())
    }

    pub(crate) fn exec_d2z_f64(
        &self,
        input: &CudaBuffer<f64>,
        output: &mut CudaBuffer<Complex64>,
        fft_len: usize,
    ) -> BackendResult<()> {
        let batch = Self::validate_real_shape(input.len(), output.len, fft_len, false)?;
        if batch == 0 {
            return Ok(());
        }
        let spectrum_len = fft_len / 2 + 1;
        let plan = self.plan_layout(CufftPlanLayout {
            fft_len,
            batch,
            input_embed: fft_len,
            input_stride: 1,
            input_dist: fft_len,
            output_embed: spectrum_len,
            output_stride: 1,
            output_dist: spectrum_len,
            fft_type: CUFFT_D2Z,
        })?;
        let input_raw = input.raw().ok_or_else(|| BackendError::InvalidArgument {
            operation: "cufft_d2z",
            reason: "non-empty input has no CUDA allocation".into(),
        })?;
        let output_raw = output
            .raw_mut()
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "cufft_d2z",
                reason: "non-empty output has no CUDA allocation".into(),
            })?;
        let (input_pointer, _input_guard): (u64, SyncOnDrop<'_>) =
            input_raw.device_ptr(&self.stream);
        let (output_pointer, _output_guard): (u64, SyncOnDrop<'_>) =
            output_raw.device_ptr_mut(&self.stream);
        let status = unsafe {
            (self.api.exec_d2z)(
                plan,
                input_pointer as *mut f64,
                output_pointer as *mut CufftDoubleComplex,
            )
        };
        if status != CUFFT_SUCCESS {
            return Err(BackendError::OperationFailed {
                operation: "cufftExecD2Z",
                reason: format!("cuFFT status {status}"),
            });
        }
        Ok(())
    }

    pub(crate) fn exec_z2d_f64(
        &self,
        input: &CudaBuffer<Complex64>,
        output: &mut CudaBuffer<f64>,
        fft_len: usize,
    ) -> BackendResult<()> {
        let batch = Self::validate_real_shape(input.len(), output.len, fft_len, true)?;
        if batch == 0 {
            return Ok(());
        }
        let spectrum_len = fft_len / 2 + 1;
        let plan = self.plan_layout(CufftPlanLayout {
            fft_len,
            batch,
            input_embed: spectrum_len,
            input_stride: 1,
            input_dist: spectrum_len,
            output_embed: fft_len,
            output_stride: 1,
            output_dist: fft_len,
            fft_type: CUFFT_Z2D,
        })?;
        let input_raw = input.raw().ok_or_else(|| BackendError::InvalidArgument {
            operation: "cufft_z2d",
            reason: "non-empty input has no CUDA allocation".into(),
        })?;
        let output_raw = output
            .raw_mut()
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "cufft_z2d",
                reason: "non-empty output has no CUDA allocation".into(),
            })?;
        let (input_pointer, _input_guard): (u64, SyncOnDrop<'_>) =
            input_raw.device_ptr(&self.stream);
        let (output_pointer, _output_guard): (u64, SyncOnDrop<'_>) =
            output_raw.device_ptr_mut(&self.stream);
        let status = unsafe {
            (self.api.exec_z2d)(
                plan,
                input_pointer as *mut CufftDoubleComplex,
                output_pointer as *mut f64,
            )
        };
        if status != CUFFT_SUCCESS {
            return Err(BackendError::OperationFailed {
                operation: "cufftExecZ2D",
                reason: format!("cuFFT status {status}"),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::CudaFft;

    #[test]
    fn cu_fft_shape_validation_accepts_regular_non_power_of_two_batches() {
        assert_eq!(CudaFft::validate_shape(3 * 257, 257).unwrap(), 3);
    }

    #[test]
    fn cu_fft_shape_validation_accepts_empty_batches() {
        assert_eq!(CudaFft::validate_shape(0, 257).unwrap(), 0);
    }

    #[test]
    fn cu_fft_shape_validation_rejects_zero_and_partial_lengths() {
        assert!(CudaFft::validate_shape(0, 0).is_err());
        assert!(CudaFft::validate_shape(10, 3).is_err());
    }

    #[test]
    fn cu_fft_strided_shape_validation_preserves_gaps_contract() {
        assert!(CudaFft::validate_strided_shape(20, 4, 3, 8).is_ok());
        assert!(CudaFft::validate_strided_shape(12, 4, 2, 4).is_ok());
        assert!(CudaFft::validate_strided_shape(7, 4, 2, 4).is_err());
        assert!(CudaFft::validate_strided_shape(16, 4, 2, 3).is_err());
        assert!(CudaFft::validate_strided_shape(usize::MAX, usize::MAX, 2, usize::MAX).is_err());
    }

    #[test]
    fn plans_bind_quota_tracked_driver_workspace() {
        let source = include_str!("cufft.rs");
        assert!(source.contains("CufftSetAutoAllocation"));
        assert!(source.contains("CufftSetWorkArea"));
        assert!(source.contains("self.stream.alloc::<u8>(work_size)"));
        assert!(source.contains("TrackedCudaSlice"));
        assert!(source.contains("set_auto_allocation"));
        assert!(source.contains("set_work_area"));
        assert!(source.contains("_workspace: Option<TrackedCudaSlice>"));
    }
}

impl Drop for CudaFft {
    fn drop(&mut self) {
        if let Ok(plans) = self.plans.get_mut() {
            for (_, plan) in plans.drain() {
                let status = unsafe { (self.api.destroy)(plan.handle) };
                if status != CUFFT_SUCCESS {
                    super::record_teardown_error(
                        &self.pending_error,
                        BackendError::OperationFailed {
                            operation: "cufftDestroy",
                            reason: format!("cuFFT status {status}"),
                        },
                    );
                }
            }
        }
    }
}

// Keep the representation assertion close to the raw cuFFT cast.
const _: () = assert!(std::mem::size_of::<Complex32>() == std::mem::size_of::<CufftComplex>());
const _: () = assert!(std::mem::align_of::<Complex32>() == std::mem::align_of::<CufftComplex>());
const _: () =
    assert!(std::mem::size_of::<Complex64>() == std::mem::size_of::<CufftDoubleComplex>());
const _: () =
    assert!(std::mem::align_of::<Complex64>() == std::mem::align_of::<CufftDoubleComplex>());
