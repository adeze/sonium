//! Backend-aware FFT adapters.
//!
//! `sonium-compute` already owns device buffers and quota accounting. This
//! module selects a backend-native path when the backend capability table
//! proves one exists and otherwise stages through the CPU plan. The execution
//! class is retained on each plan so callers and benchmarks can distinguish
//! device-resident work from an explicit host fallback.

use num_complex::Complex;
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use sonium_compute::{
    Backend, BackendError, BackendResult, BufferElement, CapabilityState, DeviceBufferLike,
    RealTransformKind,
};

use crate::cpu::planner::CpuFftPlanner;
use crate::dct::{dct2, dct2_adjoint, dct3, dct3_adjoint, dst2, dst2_adjoint, dst3, dst3_adjoint};
use crate::real::RealFftPlan;
use crate::{FftDirection, FftPlan, FftScalar};

/// Execution class of a backend FFT plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FftExecution {
    CpuNative,
    /// Executed by Sonium's backend-native kernel without host staging.
    BackendNative,
    /// Executed by explicitly copying through host memory.
    HostStagedFallback,
    /// Executed by a vendor FFT library without host staging.
    VendorNative,
}

/// Cooperative cancellation state for a device FFT submission.
///
/// CUDA/cuFFT cannot safely interrupt a kernel that has already been queued.
/// A cancelled native submission therefore drains its completion fence before
/// returning and marks the in-place destination unusable.  The plan, stream,
/// and cached workspace remain valid and may be reused for a subsequent
/// submission.  Callers that need to retain the input must keep a separate
/// copy before starting an in-place operation.
#[derive(Clone, Debug, Default)]
pub struct FftCancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl FftCancellationToken {
    /// Create a clear cancellation token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation.  The request is monotonic for this token.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Return whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// A backend FFT plan with an explicit execution-class report.
pub struct DeviceFftPlan<B, T>
where
    B: Backend,
    T: FftScalar,
    Complex<T>: BufferElement,
{
    backend: Arc<B>,
    cpu_plan: Arc<dyn FftPlan<T>>,
    execution: FftExecution,
    host_staging: Mutex<Vec<Complex<T>>>,
}

type DevicePlanCache<B, T> = RwLock<HashMap<(usize, FftDirection), Arc<DeviceFftPlan<B, T>>>>;

fn select_execution(n: usize, fft_state: CapabilityState) -> FftExecution {
    match fft_state {
        CapabilityState::Native if n > 0 && n.is_power_of_two() => FftExecution::BackendNative,
        // cuFFT supports regular non-power-of-two 1-D transforms too; keep
        // those device-resident rather than unnecessarily staging through
        // the CPU. Backend-native Metal kernels remain power-of-two-only.
        CapabilityState::VendorNative if n > 0 => FftExecution::VendorNative,
        CapabilityState::HostSharedFallback
        | CapabilityState::HostStagedFallback
        | CapabilityState::Unsupported
        | CapabilityState::Native
        | CapabilityState::VendorNative => FftExecution::HostStagedFallback,
    }
}

fn fft_capability_for<T: FftScalar>(
    capabilities: sonium_compute::ScalarCapabilities,
) -> CapabilityState {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        capabilities.c32
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        capabilities.c64
    } else {
        CapabilityState::Unsupported
    }
}

impl<B, T> DeviceFftPlan<B, T>
where
    B: Backend,
    T: FftScalar,
    Complex<T>: BufferElement,
{
    pub fn len(&self) -> usize {
        self.cpu_plan.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn direction(&self) -> FftDirection {
        self.cpu_plan.direction()
    }

    pub fn scratch_len(&self) -> usize {
        self.cpu_plan.scratch_len()
    }

    pub fn execution(&self) -> FftExecution {
        self.execution
    }

    fn finish_cancellable(&self, cancellation: &FftCancellationToken) -> BackendResult<()> {
        // A native backend returns after enqueueing work.  Always wait on its
        // checked completion fence before deciding whether a request arrived
        // in flight; otherwise dropping/reusing the caller's buffer could
        // race the cuFFT stream.  The CPU fallback gets the same contract via
        // an already-completed fence.
        let mut completion = self.backend.try_submit_completion()?;
        completion.wait()?;
        if cancellation.is_cancelled() {
            return Err(BackendError::cancelled("fft_cancellation"));
        }
        Ok(())
    }

    /// Execute an FFT with cooperative cancellation and drain its completion
    /// fence before returning.
    ///
    /// Cancellation requested before submission is rejected without touching
    /// the destination.  Cancellation requested after submission is reported
    /// only after the backend work is complete; cuFFT has no safe kernel
    /// interruption primitive, so the destination must be discarded.  The
    /// plan and backend remain reusable after this checked cancellation path.
    pub fn process_cancellable(
        &self,
        buffer: &mut B::Buffer<Complex<T>>,
        cancellation: &FftCancellationToken,
    ) -> BackendResult<()> {
        self.process_cancellable_with_hook(buffer, cancellation, || {})
    }

    /// Variant of [`Self::process_cancellable`] with a callback invoked after
    /// the transform has been submitted and before its completion fence is
    /// waited.  It is useful for deterministic integration tests and for
    /// cancellation controllers that receive an explicit submission event.
    /// The callback is not invoked when preflight or validation rejects the
    /// request.
    pub fn process_cancellable_with_hook<F>(
        &self,
        buffer: &mut B::Buffer<Complex<T>>,
        cancellation: &FftCancellationToken,
        after_submit: F,
    ) -> BackendResult<()>
    where
        F: FnOnce(),
    {
        if cancellation.is_cancelled() {
            return Err(BackendError::cancelled("fft_cancellation"));
        }
        self.process(buffer)?;
        if !buffer.is_empty() {
            after_submit();
            self.finish_cancellable(cancellation)?;
        }
        Ok(())
    }

    /// Transform a device buffer through the explicitly reported host-staged
    /// fallback. The device buffer remains owned by the caller and is copied
    /// back only after the CPU transform completes. Host staging grows on the
    /// first call that needs a larger batch and is then reused, so steady-state
    /// calls with a stable buffer shape do not allocate.
    pub fn process(&self, buffer: &mut B::Buffer<Complex<T>>) -> BackendResult<()> {
        let fft_len = self.len();
        if fft_len == 0 {
            if !buffer.is_empty() {
                return Err(BackendError::InvalidArgument {
                    operation: "fft",
                    reason: "a zero-length FFT plan can only process an empty buffer".into(),
                });
            }
            return Ok(());
        }
        // A zero-length batch is the backend equivalent of the native CPU
        // plan's empty-buffer no-op. In particular, do not hand a zero batch
        // to cuFFT/Metal, where a vendor or device API may reject it even
        // though the public FFT contract accepts it.
        if buffer.is_empty() {
            return Ok(());
        }
        if !buffer.len().is_multiple_of(fft_len) {
            return Err(BackendError::InvalidArgument {
                operation: "fft",
                reason: format!(
                    "FFT buffer length {} is not a multiple of plan length {fft_len}",
                    buffer.len()
                ),
            });
        }

        if matches!(
            self.execution,
            FftExecution::BackendNative | FftExecution::VendorNative
        ) {
            let inverse = self.direction() == FftDirection::Inverse;
            match self.backend.try_fft_in_place(buffer, fft_len, inverse) {
                Ok(()) => return Ok(()),
                // The execution class was selected from the backend's
                // capability table. Falling back here would make a failed
                // native operation look like a successful vendor run to the
                // caller and would also hide a capability-table regression.
                Err(BackendError::Unsupported { .. }) => {
                    return Err(BackendError::Unsupported {
                        backend: std::any::type_name::<B>(),
                        operation: "fft",
                        scalar: std::any::type_name::<Complex<T>>(),
                    });
                }
                Err(error) => return Err(error),
            }
        }

        let mut host = self
            .host_staging
            .lock()
            .expect("FFT host staging mutex poisoned");
        if host.len() < buffer.len() {
            host.resize(buffer.len(), Complex::<T>::default());
        }
        let host = &mut host[..buffer.len()];
        self.backend.try_copy_d2h(buffer, host)?;
        self.cpu_plan.process(host);
        self.backend.try_copy_h2d(host, buffer)
    }

    /// Transform `batch` contiguous FFTs whose starts are `stride` elements
    /// apart. Padding between transforms is preserved. Native backends receive
    /// the strided view directly; unsupported backends use the same explicit
    /// host-staged fallback as [`Self::process`].
    pub fn process_strided(
        &self,
        buffer: &mut B::Buffer<Complex<T>>,
        batch: usize,
        stride: usize,
    ) -> BackendResult<()> {
        let fft_len = self.len();
        if fft_len == 0 || stride < fft_len {
            return Err(BackendError::InvalidArgument {
                operation: "fft_strided",
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
                    operation: "fft_strided",
                    reason: "FFT batch shape overflows address arithmetic".into(),
                })?
        };
        if buffer.len() < required {
            return Err(BackendError::InvalidArgument {
                operation: "fft_strided",
                reason: format!(
                    "buffer length {} is shorter than required strided span {required}",
                    buffer.len()
                ),
            });
        }
        if batch == 0 {
            return Ok(());
        }

        if matches!(
            self.execution,
            FftExecution::BackendNative | FftExecution::VendorNative
        ) {
            let inverse = self.direction() == FftDirection::Inverse;
            match self
                .backend
                .try_fft_strided_in_place(buffer, fft_len, batch, stride, inverse)
            {
                Ok(()) => return Ok(()),
                Err(BackendError::Unsupported { .. }) => {
                    return Err(BackendError::Unsupported {
                        backend: std::any::type_name::<B>(),
                        operation: "fft_strided",
                        scalar: std::any::type_name::<Complex<T>>(),
                    });
                }
                Err(error) => return Err(error),
            }
        }

        let mut host = self
            .host_staging
            .lock()
            .expect("FFT host staging mutex poisoned");
        if host.len() < buffer.len() {
            host.resize(buffer.len(), Complex::<T>::default());
        }
        let host = &mut host[..buffer.len()];
        self.backend.try_copy_d2h(buffer, host)?;
        for batch_index in 0..batch {
            let start = batch_index * stride;
            self.cpu_plan.process(&mut host[start..start + fft_len]);
        }
        self.backend.try_copy_h2d(host, buffer)
    }

    /// Strided counterpart of [`Self::process_cancellable`].  Padding remains
    /// untouched on successful native execution, while a cancelled in-place
    /// destination is unusable until the caller overwrites it.
    pub fn process_strided_cancellable(
        &self,
        buffer: &mut B::Buffer<Complex<T>>,
        batch: usize,
        stride: usize,
        cancellation: &FftCancellationToken,
    ) -> BackendResult<()> {
        self.process_strided_cancellable_with_hook(buffer, batch, stride, cancellation, || {})
    }

    /// Strided cancellable FFT with a deterministic post-submission hook.
    pub fn process_strided_cancellable_with_hook<F>(
        &self,
        buffer: &mut B::Buffer<Complex<T>>,
        batch: usize,
        stride: usize,
        cancellation: &FftCancellationToken,
        after_submit: F,
    ) -> BackendResult<()>
    where
        F: FnOnce(),
    {
        if cancellation.is_cancelled() {
            return Err(BackendError::cancelled("fft_cancellation"));
        }
        self.process_strided(buffer, batch, stride)?;
        if batch != 0 {
            after_submit();
            self.finish_cancellable(cancellation)?;
        }
        Ok(())
    }
}

fn real_scalar_capability<T: FftScalar>(
    capabilities: sonium_compute::ScalarCapabilities,
) -> CapabilityState {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        capabilities.f32
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        capabilities.f64
    } else {
        CapabilityState::Unsupported
    }
}

fn select_real_execution(capability: CapabilityState) -> FftExecution {
    match capability {
        CapabilityState::Native => FftExecution::BackendNative,
        CapabilityState::VendorNative => FftExecution::VendorNative,
        CapabilityState::HostSharedFallback
        | CapabilityState::HostStagedFallback
        | CapabilityState::Unsupported => FftExecution::HostStagedFallback,
    }
}

/// Backend-aware real-input FFT plan. Forward plans map contiguous batches of
/// `N` real samples to `N/2+1` complex bins; inverse plans use the reverse
/// shape. CUDA uses cuFFT R2C/C2R directly and applies the inverse scale on
/// device, so no host staging occurs on the native path.
pub struct DeviceRealFftPlan<B, T>
where
    B: Backend,
    T: FftScalar + BufferElement,
    Complex<T>: BufferElement,
{
    backend: Arc<B>,
    cpu_plan: Arc<RealFftPlan<T>>,
    execution: FftExecution,
    host_real: Mutex<Vec<T>>,
    host_spectrum: Mutex<Vec<Complex<T>>>,
    host_scratch: Mutex<Vec<Complex<T>>>,
}

impl<B, T> DeviceRealFftPlan<B, T>
where
    B: Backend,
    T: FftScalar + BufferElement,
    Complex<T>: BufferElement,
{
    pub fn len(&self) -> usize {
        self.cpu_plan.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn spectrum_len(&self) -> usize {
        self.cpu_plan.spectrum_len()
    }

    pub fn execution(&self) -> FftExecution {
        self.execution
    }

    /// Execute every contiguous batch represented by `input` and `output`.
    pub fn process(
        &self,
        input: &B::Buffer<T>,
        output: &mut B::Buffer<Complex<T>>,
    ) -> BackendResult<()> {
        let n = self.len();
        let spectrum_len = self.spectrum_len();
        if n == 0 {
            if !input.is_empty() || !output.is_empty() {
                return Err(BackendError::InvalidArgument {
                    operation: "real_fft",
                    reason: "a zero-length real FFT plan can only process empty buffers".into(),
                });
            }
            return Ok(());
        }
        if !input.len().is_multiple_of(n) || !output.len().is_multiple_of(spectrum_len) {
            return Err(BackendError::InvalidArgument {
                operation: "real_fft",
                reason: format!(
                    "real FFT buffers are not batched by input N={n} and spectrum N/2+1={spectrum_len}"
                ),
            });
        }
        if input.len() / n != output.len() / spectrum_len {
            return Err(BackendError::InvalidArgument {
                operation: "real_fft",
                reason: format!(
                    "real FFT batch counts differ: input {}, output {}",
                    input.len() / n,
                    output.len() / spectrum_len
                ),
            });
        }
        if input.is_empty() {
            return Ok(());
        }
        if matches!(
            self.execution,
            FftExecution::BackendNative | FftExecution::VendorNative
        ) {
            return self.backend.try_real_fft_forward(input, output, n);
        }

        let mut host_real = self
            .host_real
            .lock()
            .expect("real FFT host buffer poisoned");
        let mut host_spectrum = self
            .host_spectrum
            .lock()
            .expect("real FFT host buffer poisoned");
        let mut scratch = self.host_scratch.lock().expect("real FFT scratch poisoned");
        host_real.resize(input.len(), T::default());
        host_spectrum.resize(output.len(), Complex::new(T::zero(), T::zero()));
        scratch.resize(
            self.cpu_plan.scratch_len(),
            Complex::new(T::zero(), T::zero()),
        );
        self.backend.try_copy_d2h(input, &mut host_real)?;
        self.cpu_plan
            .forward_batched_with_scratch(&host_real, &mut host_spectrum, &mut scratch);
        self.backend.try_copy_h2d(&host_spectrum, output)
    }
}

/// Inverse half-spectrum plan. It is separate from `DeviceRealFftPlan`'s
/// forward input/output shape so Rust's type system cannot accidentally pass
/// a real sample buffer where a spectrum is required.
pub struct DeviceRealFftInversePlan<B, T>
where
    B: Backend,
    T: FftScalar + BufferElement,
    Complex<T>: BufferElement,
{
    backend: Arc<B>,
    cpu_plan: Arc<RealFftPlan<T>>,
    execution: FftExecution,
    host_real: Mutex<Vec<T>>,
    host_spectrum: Mutex<Vec<Complex<T>>>,
    host_scratch: Mutex<Vec<Complex<T>>>,
}

impl<B, T> DeviceRealFftInversePlan<B, T>
where
    B: Backend,
    T: FftScalar + BufferElement,
    Complex<T>: BufferElement,
{
    pub fn len(&self) -> usize {
        self.cpu_plan.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn spectrum_len(&self) -> usize {
        self.cpu_plan.spectrum_len()
    }

    pub fn execution(&self) -> FftExecution {
        self.execution
    }

    pub fn process(
        &self,
        input: &B::Buffer<Complex<T>>,
        output: &mut B::Buffer<T>,
    ) -> BackendResult<()> {
        let n = self.len();
        let spectrum_len = self.spectrum_len();
        if n == 0 {
            if !input.is_empty() || !output.is_empty() {
                return Err(BackendError::InvalidArgument {
                    operation: "real_fft_inverse",
                    reason: "a zero-length real FFT plan can only process empty buffers".into(),
                });
            }
            return Ok(());
        }
        if !input.len().is_multiple_of(spectrum_len) || !output.len().is_multiple_of(n) {
            return Err(BackendError::InvalidArgument {
                operation: "real_fft_inverse",
                reason: format!(
                    "real inverse buffers are not batched by spectrum N/2+1={spectrum_len} and N={n}"
                ),
            });
        }
        if input.len() / spectrum_len != output.len() / n {
            return Err(BackendError::InvalidArgument {
                operation: "real_fft_inverse",
                reason: "real inverse FFT batch counts differ".into(),
            });
        }
        if input.is_empty() {
            return Ok(());
        }
        if matches!(
            self.execution,
            FftExecution::BackendNative | FftExecution::VendorNative
        ) {
            return self.backend.try_real_fft_inverse(input, output, n);
        }

        let mut host_real = self
            .host_real
            .lock()
            .expect("real FFT host buffer poisoned");
        let mut host_spectrum = self
            .host_spectrum
            .lock()
            .expect("real FFT host buffer poisoned");
        let mut scratch = self.host_scratch.lock().expect("real FFT scratch poisoned");
        host_real.resize(output.len(), T::default());
        host_spectrum.resize(input.len(), Complex::new(T::zero(), T::zero()));
        scratch.resize(
            self.cpu_plan.scratch_len(),
            Complex::new(T::zero(), T::zero()),
        );
        self.backend.try_copy_d2h(input, &mut host_spectrum)?;
        self.cpu_plan
            .inverse_batched_with_scratch(&host_spectrum, &mut host_real, &mut scratch);
        self.backend.try_copy_h2d(&host_real, output)
    }
}

/// Planner for backend-aware real FFT forward and inverse plans.
pub struct BackendRealFftPlanner<B, T>
where
    B: Backend,
    T: FftScalar + BufferElement,
    Complex<T>: BufferElement,
{
    backend: Arc<B>,
    cpu: CpuFftPlanner<T>,
}

impl<B, T> BackendRealFftPlanner<B, T>
where
    B: Backend,
    T: FftScalar + BufferElement,
    Complex<T>: BufferElement,
{
    pub fn new(backend: Arc<B>) -> Self {
        Self {
            backend,
            cpu: CpuFftPlanner::new(),
        }
    }

    pub fn plan_fft_forward(&self, n: usize) -> Arc<DeviceRealFftPlan<B, T>> {
        let execution = select_real_execution(real_scalar_capability::<T>(
            self.backend.capabilities().real_fft,
        ));
        Arc::new(DeviceRealFftPlan {
            backend: self.backend.clone(),
            cpu_plan: Arc::new(RealFftPlan::with_planner(n, &self.cpu)),
            execution,
            host_real: Mutex::new(Vec::new()),
            host_spectrum: Mutex::new(Vec::new()),
            host_scratch: Mutex::new(Vec::new()),
        })
    }

    pub fn plan_fft_inverse(&self, n: usize) -> Arc<DeviceRealFftInversePlan<B, T>> {
        let execution = select_real_execution(real_scalar_capability::<T>(
            self.backend.capabilities().real_fft,
        ));
        Arc::new(DeviceRealFftInversePlan {
            backend: self.backend.clone(),
            cpu_plan: Arc::new(RealFftPlan::with_planner(n, &self.cpu)),
            execution,
            host_real: Mutex::new(Vec::new()),
            host_spectrum: Mutex::new(Vec::new()),
            host_scratch: Mutex::new(Vec::new()),
        })
    }
}

struct DeviceRealTransformPlan<B, T>
where
    B: Backend,
    T: FftScalar + BufferElement,
{
    backend: Arc<B>,
    n: usize,
    execution: FftExecution,
    host_input: Mutex<Vec<T>>,
    host_output: Mutex<Vec<T>>,
}

impl<B, T> DeviceRealTransformPlan<B, T>
where
    B: Backend,
    T: FftScalar + BufferElement,
{
    fn process(
        &self,
        kind: RealTransformKind,
        input: &B::Buffer<T>,
        output: &mut B::Buffer<T>,
        operation: &'static str,
    ) -> BackendResult<()> {
        if input.len() != output.len() || (self.n != 0 && !input.len().is_multiple_of(self.n)) {
            return Err(BackendError::InvalidArgument {
                operation,
                reason: format!(
                    "{operation} buffers must have equal lengths batched by N={}",
                    self.n
                ),
            });
        }
        if self.n == 0 || input.is_empty() {
            return Ok(());
        }
        if matches!(
            self.execution,
            FftExecution::BackendNative | FftExecution::VendorNative
        ) {
            return self.backend.try_real_transform(kind, input, output, self.n);
        }
        let mut host_input = self
            .host_input
            .lock()
            .expect("real transform host buffer poisoned");
        let mut host_output = self
            .host_output
            .lock()
            .expect("real transform host buffer poisoned");
        host_input.resize(input.len(), T::default());
        host_output.resize(output.len(), T::default());
        self.backend.try_copy_d2h(input, &mut host_input)?;
        for (source, destination) in host_input
            .chunks_exact(self.n)
            .zip(host_output.chunks_exact_mut(self.n))
        {
            match kind {
                RealTransformKind::Dct2 => dct2(source, destination),
                RealTransformKind::Dct3 => dct3(source, destination),
                RealTransformKind::Dct2Adjoint => dct2_adjoint(source, destination),
                RealTransformKind::Dct3Adjoint => dct3_adjoint(source, destination),
                RealTransformKind::Dst2 => dst2(source, destination),
                RealTransformKind::Dst3 => dst3(source, destination),
                RealTransformKind::Dst2Adjoint => dst2_adjoint(source, destination),
                RealTransformKind::Dst3Adjoint => dst3_adjoint(source, destination),
            }
        }
        self.backend.try_copy_h2d(&host_output, output)
    }
}

/// Backend-aware DCT-II/III plan, including both Euclidean adjoints.
pub struct DeviceDctPlan<B, T>(DeviceRealTransformPlan<B, T>)
where
    B: Backend,
    T: FftScalar + BufferElement;

impl<B, T> DeviceDctPlan<B, T>
where
    B: Backend,
    T: FftScalar + BufferElement,
{
    pub fn len(&self) -> usize {
        self.0.n
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn execution(&self) -> FftExecution {
        self.0.execution
    }
    pub fn forward(&self, input: &B::Buffer<T>, output: &mut B::Buffer<T>) -> BackendResult<()> {
        self.0
            .process(RealTransformKind::Dct2, input, output, "dct2")
    }
    pub fn inverse(&self, input: &B::Buffer<T>, output: &mut B::Buffer<T>) -> BackendResult<()> {
        self.0
            .process(RealTransformKind::Dct3, input, output, "dct3")
    }
    pub fn forward_adjoint(
        &self,
        input: &B::Buffer<T>,
        output: &mut B::Buffer<T>,
    ) -> BackendResult<()> {
        self.0.process(
            RealTransformKind::Dct2Adjoint,
            input,
            output,
            "dct2_adjoint",
        )
    }
    pub fn inverse_adjoint(
        &self,
        input: &B::Buffer<T>,
        output: &mut B::Buffer<T>,
    ) -> BackendResult<()> {
        self.0.process(
            RealTransformKind::Dct3Adjoint,
            input,
            output,
            "dct3_adjoint",
        )
    }
}

/// Backend-aware DST-II/III plan, including both Euclidean adjoints.
pub struct DeviceDstPlan<B, T>(DeviceRealTransformPlan<B, T>)
where
    B: Backend,
    T: FftScalar + BufferElement;

impl<B, T> DeviceDstPlan<B, T>
where
    B: Backend,
    T: FftScalar + BufferElement,
{
    pub fn len(&self) -> usize {
        self.0.n
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn execution(&self) -> FftExecution {
        self.0.execution
    }
    pub fn forward(&self, input: &B::Buffer<T>, output: &mut B::Buffer<T>) -> BackendResult<()> {
        self.0
            .process(RealTransformKind::Dst2, input, output, "dst2")
    }
    pub fn inverse(&self, input: &B::Buffer<T>, output: &mut B::Buffer<T>) -> BackendResult<()> {
        self.0
            .process(RealTransformKind::Dst3, input, output, "dst3")
    }
    pub fn forward_adjoint(
        &self,
        input: &B::Buffer<T>,
        output: &mut B::Buffer<T>,
    ) -> BackendResult<()> {
        self.0.process(
            RealTransformKind::Dst2Adjoint,
            input,
            output,
            "dst2_adjoint",
        )
    }
    pub fn inverse_adjoint(
        &self,
        input: &B::Buffer<T>,
        output: &mut B::Buffer<T>,
    ) -> BackendResult<()> {
        self.0.process(
            RealTransformKind::Dst3Adjoint,
            input,
            output,
            "dst3_adjoint",
        )
    }
}

pub struct BackendDctPlanner<B, T>
where
    B: Backend,
    T: FftScalar + BufferElement,
{
    backend: Arc<B>,
    _marker: std::marker::PhantomData<T>,
}

impl<B, T> BackendDctPlanner<B, T>
where
    B: Backend,
    T: FftScalar + BufferElement,
{
    pub fn new(backend: Arc<B>) -> Self {
        Self {
            backend,
            _marker: std::marker::PhantomData,
        }
    }
    pub fn plan(&self, n: usize) -> Arc<DeviceDctPlan<B, T>> {
        Arc::new(DeviceDctPlan(DeviceRealTransformPlan {
            backend: self.backend.clone(),
            n,
            execution: select_real_execution(real_scalar_capability::<T>(
                self.backend.capabilities().dct,
            )),
            host_input: Mutex::new(Vec::new()),
            host_output: Mutex::new(Vec::new()),
        }))
    }
}

pub struct BackendDstPlanner<B, T>
where
    B: Backend,
    T: FftScalar + BufferElement,
{
    backend: Arc<B>,
    _marker: std::marker::PhantomData<T>,
}

impl<B, T> BackendDstPlanner<B, T>
where
    B: Backend,
    T: FftScalar + BufferElement,
{
    pub fn new(backend: Arc<B>) -> Self {
        Self {
            backend,
            _marker: std::marker::PhantomData,
        }
    }
    pub fn plan(&self, n: usize) -> Arc<DeviceDstPlan<B, T>> {
        Arc::new(DeviceDstPlan(DeviceRealTransformPlan {
            backend: self.backend.clone(),
            n,
            execution: select_real_execution(real_scalar_capability::<T>(
                self.backend.capabilities().dst,
            )),
            host_input: Mutex::new(Vec::new()),
            host_output: Mutex::new(Vec::new()),
        }))
    }
}

/// Planner for a concrete `sonium-compute` backend.
pub struct BackendFftPlanner<B, T>
where
    B: Backend,
    T: FftScalar,
    Complex<T>: BufferElement,
{
    backend: Arc<B>,
    cpu: CpuFftPlanner<T>,
    cache: DevicePlanCache<B, T>,
}

impl<B, T> BackendFftPlanner<B, T>
where
    B: Backend,
    T: FftScalar,
    Complex<T>: BufferElement,
{
    pub fn new(backend: Arc<B>) -> Self {
        Self {
            backend,
            cpu: CpuFftPlanner::new(),
            cache: RwLock::new(HashMap::new()),
        }
    }

    pub fn plan(&self, n: usize, direction: FftDirection) -> Arc<DeviceFftPlan<B, T>> {
        if let Some(plan) = self
            .cache
            .read()
            .expect("backend FFT planner cache poisoned")
            .get(&(n, direction))
        {
            return plan.clone();
        }

        let fft_state = {
            let capabilities = self.backend.capabilities();
            fft_capability_for::<T>(capabilities.fft)
        };
        let execution = select_execution(n, fft_state);

        let plan = Arc::new(DeviceFftPlan {
            backend: self.backend.clone(),
            cpu_plan: self.cpu.plan(n, direction),
            execution,
            host_staging: Mutex::new(Vec::new()),
        });
        let mut cache = self
            .cache
            .write()
            .expect("backend FFT planner cache poisoned");
        if let Some(existing) = cache.get(&(n, direction)) {
            return existing.clone();
        }
        cache.insert((n, direction), plan.clone());
        plan
    }
}

#[cfg(test)]
mod tests {
    use super::FftExecution;
    use crate::FftDirection;
    use num_complex::{Complex32, Complex64};
    use sonium_compute::Backend;
    use sonium_compute::CapabilityState;
    use std::sync::{Arc, Barrier};
    use std::thread;

    #[cfg(feature = "cuda")]
    fn cuda_backend_or_skip(label: &str) -> Option<Arc<sonium_compute::CudaBackend>> {
        match sonium_compute::CudaBackend::new_default() {
            Ok(backend) => Some(Arc::new(backend)),
            Err(error) => {
                let required = std::env::var("SONIUM_REQUIRE_CUDA")
                    .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
                    .unwrap_or(false);
                if required {
                    panic!(
                        "CUDA required by SONIUM_REQUIRE_CUDA=1 but unavailable for {label}: {error}"
                    );
                }
                eprintln!("CUDA unavailable, skipping {label}: {error}");
                None
            }
        }
    }

    #[cfg(feature = "cuda")]
    type DctCudaOp = fn(
        &super::DeviceDctPlan<sonium_compute::CudaBackend, f64>,
        &sonium_compute::CudaBuffer<f64>,
        &mut sonium_compute::CudaBuffer<f64>,
    ) -> sonium_compute::BackendResult<()>;
    #[cfg(feature = "cuda")]
    type DctCudaCase = (&'static str, DctCudaOp, fn(&[f64], &mut [f64]));

    #[cfg(feature = "cuda")]
    type DstCudaOp = fn(
        &super::DeviceDstPlan<sonium_compute::CudaBackend, f64>,
        &sonium_compute::CudaBuffer<f64>,
        &mut sonium_compute::CudaBuffer<f64>,
    ) -> sonium_compute::BackendResult<()>;
    #[cfg(feature = "cuda")]
    type DstCudaCase = (&'static str, DstCudaOp, fn(&[f64], &mut [f64]));

    // The adapter's backend-specific execution is exercised by the compute
    // crate's CPU backend tests. Keep this module intentionally small here so
    // enabling CUDA or Metal is never required merely to compile FFT.
    #[test]
    fn host_staging_is_explicit_in_the_public_execution_enum() {
        assert_eq!(
            super::FftExecution::HostStagedFallback,
            super::FftExecution::HostStagedFallback
        );
    }

    #[test]
    fn backend_planner_reuses_device_plan_and_cpu_plan_cache_entry() {
        let backend = Arc::new(sonium_compute::cpu::CpuBackend::new());
        let planner = super::BackendFftPlanner::<_, f64>::new(backend);
        let first = planner.plan(32, FftDirection::Forward);
        let second = planner.plan(32, FftDirection::Forward);
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.execution(), FftExecution::HostStagedFallback);
    }

    #[test]
    fn concurrent_planner_requests_share_the_canonical_cached_plan() {
        const THREADS: usize = 8;

        let backend = Arc::new(sonium_compute::cpu::CpuBackend::new());
        let planner = Arc::new(super::BackendFftPlanner::<_, f64>::new(backend));
        let barrier = Arc::new(Barrier::new(THREADS));
        let mut handles = Vec::with_capacity(THREADS);

        for _ in 0..THREADS {
            let planner = Arc::clone(&planner);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                planner.plan(257, FftDirection::Forward)
            }));
        }

        let plans: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().expect("planner worker panicked"))
            .collect();
        let first = &plans[0];
        assert!(plans.iter().all(|plan| Arc::ptr_eq(first, plan)));
    }

    #[test]
    fn cpu_backend_reports_host_fallback_for_f32_without_native_fft() {
        let backend = Arc::new(sonium_compute::cpu::CpuBackend::new());
        let planner = super::BackendFftPlanner::<_, f32>::new(backend);
        assert_eq!(
            planner.plan(32, FftDirection::Forward).execution(),
            FftExecution::HostStagedFallback
        );
    }

    #[test]
    fn cpu_backend_real_transform_plans_match_reference_contracts() {
        let backend = Arc::new(sonium_compute::cpu::CpuBackend::new());
        let n = 7;
        let input: Vec<f64> = (0..n * 2)
            .map(|index| (0.17 * index as f64).sin() + 0.03 * index as f64)
            .collect();

        let real = super::BackendRealFftPlanner::<_, f64>::new(backend.clone());
        let forward = real.plan_fft_forward(n);
        let inverse = real.plan_fft_inverse(n);
        assert_eq!(forward.execution(), FftExecution::HostStagedFallback);
        let mut input_device = backend.raw_alloc::<f64>(input.len()).unwrap();
        backend.copy_h2d(&input, &mut input_device);
        let mut spectrum_device = backend.raw_alloc::<Complex64>((n / 2 + 1) * 2).unwrap();
        forward
            .process(&input_device, &mut spectrum_device)
            .unwrap();
        let mut spectrum = vec![Complex64::default(); (n / 2 + 1) * 2];
        backend.copy_d2h(&spectrum_device, &mut spectrum);
        let reference: Vec<_> = input
            .chunks_exact(n)
            .flat_map(|chunk| crate::RealFftPlan::<f64>::new(n).forward(chunk))
            .collect();
        for (actual, expected) in spectrum.iter().zip(reference.iter().cycle()) {
            assert!((*actual - *expected).norm() < 1.0e-10);
        }

        let mut recovered_device = backend.raw_alloc::<f64>(input.len()).unwrap();
        inverse
            .process(&spectrum_device, &mut recovered_device)
            .unwrap();
        let mut recovered = vec![0.0; input.len()];
        backend.copy_d2h(&recovered_device, &mut recovered);
        for (actual, expected) in recovered.iter().zip(&input) {
            assert!((actual - expected).abs() < 1.0e-10);
        }

        let dct = super::BackendDctPlanner::<_, f64>::new(backend.clone()).plan(n);
        let dst = super::BackendDstPlanner::<_, f64>::new(backend.clone()).plan(n);
        let mut one_device = backend.raw_alloc::<f64>(n).unwrap();
        backend.copy_h2d(&input[..n], &mut one_device);
        let mut transformed_device = backend.raw_alloc::<f64>(n).unwrap();
        let mut transformed = vec![0.0; n];
        dct.forward(&one_device, &mut transformed_device).unwrap();
        backend.copy_d2h(&transformed_device, &mut transformed);
        let mut expected = vec![0.0; n];
        crate::dct2(&input[..n], &mut expected);
        for (actual, expected) in transformed.iter().zip(&expected) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
        dst.forward(&one_device, &mut transformed_device).unwrap();
        backend.copy_d2h(&transformed_device, &mut transformed);
        crate::dst2(&input[..n], &mut expected);
        for (actual, expected) in transformed.iter().zip(&expected) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
    }

    #[test]
    fn native_execution_requires_power_of_two_and_native_capability() {
        assert_eq!(
            super::select_execution(64, sonium_compute::CapabilityState::Native),
            FftExecution::BackendNative
        );
        assert_eq!(
            super::select_execution(64, sonium_compute::CapabilityState::VendorNative),
            FftExecution::VendorNative
        );
        assert_eq!(
            super::select_execution(63, sonium_compute::CapabilityState::VendorNative),
            FftExecution::VendorNative
        );
        assert_eq!(
            super::select_execution(63, sonium_compute::CapabilityState::Native),
            FftExecution::HostStagedFallback
        );
        assert_eq!(
            super::select_execution(64, sonium_compute::CapabilityState::HostStagedFallback),
            FftExecution::HostStagedFallback
        );
    }

    #[test]
    fn fft_capability_uses_complex_buffer_precision() {
        let capabilities = sonium_compute::ScalarCapabilities::new(
            CapabilityState::Unsupported,
            CapabilityState::Unsupported,
            CapabilityState::Native,
            CapabilityState::VendorNative,
        );
        assert_eq!(
            super::fft_capability_for::<f32>(capabilities),
            CapabilityState::Native
        );
        assert_eq!(
            super::fft_capability_for::<f64>(capabilities),
            CapabilityState::VendorNative
        );
    }

    #[test]
    fn backend_plan_rejects_partial_batch_without_panicking() {
        let backend = Arc::new(sonium_compute::cpu::CpuBackend::new());
        let mut buffer = backend
            .raw_alloc::<Complex32>(9)
            .expect("CPU FFT test allocation");
        let planner = super::BackendFftPlanner::<_, f32>::new(backend);
        let plan = planner.plan(8, FftDirection::Forward);
        let error = plan
            .process(&mut buffer)
            .expect_err("partial FFT batches must be rejected");
        assert!(matches!(
            error,
            sonium_compute::BackendError::InvalidArgument { .. }
        ));
    }

    #[test]
    fn backend_plan_accepts_empty_batch_without_dispatching() {
        let backend = Arc::new(sonium_compute::cpu::CpuBackend::new());
        let planner = super::BackendFftPlanner::<_, f32>::new(backend.clone());
        let plan = planner.plan(8, FftDirection::Forward);
        let mut buffer = backend
            .raw_alloc::<Complex32>(0)
            .expect("empty CPU FFT allocation");
        plan.process(&mut buffer)
            .expect("empty FFT batches are documented no-ops");
    }

    #[test]
    fn cpu_backend_host_staging_executes_strided_batch_and_preserves_padding() {
        let backend = Arc::new(sonium_compute::cpu::CpuBackend::new());
        let planner = super::BackendFftPlanner::<_, f64>::new(backend.clone());
        let plan = planner.plan(4, FftDirection::Forward);
        let stride = 6;
        let mut input = vec![Complex64::new(-9.0, 2.0); stride + 4 + 2];
        input[..4].copy_from_slice(&[
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ]);
        input[stride..stride + 4].copy_from_slice(&[
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(8.0, 0.0),
        ]);
        let mut buffer = backend
            .raw_alloc::<Complex64>(input.len())
            .expect("CPU strided FFT allocation");
        backend.copy_h2d(&input, &mut buffer);

        plan.process_strided(&mut buffer, 2, stride)
            .expect("CPU strided FFT should execute");

        let mut actual = vec![Complex64::default(); input.len()];
        backend.copy_d2h(&buffer, &mut actual);
        assert_eq!(actual[4], input[4]);
        assert_eq!(actual[5], input[5]);
        assert_eq!(actual[10], input[10]);
        assert_eq!(actual[11], input[11]);
        assert_eq!(
            actual[..4],
            [
                Complex64::new(10.0, 0.0),
                Complex64::new(-2.0, 2.0),
                Complex64::new(-2.0, 0.0),
                Complex64::new(-2.0, -2.0),
            ]
        );
        assert_eq!(
            actual[stride..stride + 4],
            [
                Complex64::new(20.0, 0.0),
                Complex64::new(-4.0, 4.0),
                Complex64::new(-4.0, 0.0),
                Complex64::new(-4.0, -4.0),
            ]
        );
    }

    #[test]
    fn backend_plan_rejects_invalid_strided_shape_without_touching_buffer() {
        let backend = Arc::new(sonium_compute::cpu::CpuBackend::new());
        let planner = super::BackendFftPlanner::<_, f32>::new(backend.clone());
        let plan = planner.plan(4, FftDirection::Forward);
        let input = vec![Complex32::new(1.0, -2.0); 8];
        let mut buffer = backend
            .raw_alloc::<Complex32>(input.len())
            .expect("CPU invalid-stride FFT allocation");
        backend.copy_h2d(&input, &mut buffer);

        let error = plan
            .process_strided(&mut buffer, 2, 3)
            .expect_err("stride shorter than one FFT must be rejected");
        assert!(matches!(
            error,
            sonium_compute::BackendError::InvalidArgument {
                operation: "fft_strided",
                ..
            }
        ));

        let mut actual = vec![Complex32::default(); input.len()];
        backend.copy_d2h(&buffer, &mut actual);
        assert_eq!(actual, input);
    }

    #[test]
    fn cpu_backend_host_staging_executes_complex64_batch() {
        let backend = Arc::new(sonium_compute::cpu::CpuBackend::new());
        let planner = super::BackendFftPlanner::<_, f64>::new(backend.clone());
        let plan = planner.plan(4, FftDirection::Forward);
        assert_eq!(plan.execution(), FftExecution::HostStagedFallback);

        let input = [
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let mut buffer = backend
            .raw_alloc::<Complex64>(input.len())
            .expect("CPU FFT fallback allocation");
        backend.copy_h2d(&input, &mut buffer);
        plan.process(&mut buffer)
            .expect("CPU host-staged FFT should execute");

        let mut actual = [Complex64::default(); 4];
        backend.copy_d2h(&buffer, &mut actual);
        let expected = [
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-2.0, -2.0),
        ];
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).norm() < 1.0e-12);
        }
    }

    #[test]
    fn cancellable_fft_rejects_pre_cancel_and_reuses_plan_after_in_flight_cancel() {
        let backend = Arc::new(sonium_compute::cpu::CpuBackend::new());
        let planner = super::BackendFftPlanner::<_, f64>::new(backend.clone());
        let plan = planner.plan(4, FftDirection::Forward);
        let input = [
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(4.0, 2.0),
        ];
        let mut buffer = backend
            .raw_alloc::<Complex64>(input.len())
            .expect("CPU cancellable FFT allocation");
        backend.copy_h2d(&input, &mut buffer);

        let pre_cancelled = super::FftCancellationToken::new();
        pre_cancelled.cancel();
        let error = plan
            .process_cancellable(&mut buffer, &pre_cancelled)
            .expect_err("pre-cancelled FFT must not submit work");
        assert!(error.is_cancelled());
        let mut unchanged = [Complex64::default(); 4];
        backend.copy_d2h(&buffer, &mut unchanged);
        assert_eq!(unchanged, input);

        let in_flight = super::FftCancellationToken::new();
        let request = in_flight.clone();
        let error = plan
            .process_cancellable_with_hook(&mut buffer, &in_flight, move || request.cancel())
            .expect_err("post-submission cancellation must be reported");
        assert!(error.is_cancelled());

        // The cancelled destination is intentionally discarded. Overwriting
        // it and submitting through the same plan proves the completion wait
        // drained all cuFFT/CPU work and left cached resources reusable.
        backend.copy_h2d(&input, &mut buffer);
        plan.process(&mut buffer)
            .expect("plan must be reusable after cancellation");
        backend.try_synchronize().expect("reused FFT completion");
        let mut actual = [Complex64::default(); 4];
        backend.copy_d2h(&buffer, &mut actual);
        let expected = [
            Complex64::new(10.0, 1.5),
            Complex64::new(-5.0, 1.5),
            Complex64::new(-2.0, -0.5),
            Complex64::new(1.0, -2.5),
        ];
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).norm() < 1.0e-12);
        }
    }

    #[test]
    fn cancellable_strided_fft_drains_and_reuses_after_cancellation() {
        let backend = Arc::new(sonium_compute::cpu::CpuBackend::new());
        let planner = super::BackendFftPlanner::<_, f32>::new(backend.clone());
        let plan = planner.plan(4, FftDirection::Forward);
        let stride = 6;
        let input = vec![Complex32::new(-7.0, 3.0); stride * 2];
        let mut buffer = backend
            .raw_alloc::<Complex32>(input.len())
            .expect("CPU strided cancellable FFT allocation");
        backend.copy_h2d(&input, &mut buffer);

        let cancellation = super::FftCancellationToken::new();
        let request = cancellation.clone();
        let error = plan
            .process_strided_cancellable_with_hook(
                &mut buffer,
                2,
                stride,
                &cancellation,
                move || request.cancel(),
            )
            .expect_err("strided post-submission cancellation must be reported");
        assert!(error.is_cancelled());

        backend.copy_h2d(&input, &mut buffer);
        let clear = super::FftCancellationToken::new();
        plan.process_strided_cancellable(&mut buffer, 2, stride, &clear)
            .expect("strided plan must be reusable after cancellation");
        let mut actual = vec![Complex32::default(); input.len()];
        backend.copy_d2h(&buffer, &mut actual);
        assert_eq!(&actual[4..6], &input[4..6]);
        assert_eq!(&actual[10..12], &input[10..12]);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_backend_planner_executes_native_batched_fft() {
        let backend = match sonium_compute::MetalBackend::new() {
            Ok(backend) => Arc::new(backend),
            Err(error) => {
                eprintln!("Metal unavailable, skipping FFT planner parity: {error}");
                return;
            }
        };
        if backend.capabilities().fft.c32 != sonium_compute::CapabilityState::Native {
            eprintln!("Metal Complex32 FFT unavailable, skipping planner parity");
            return;
        }

        fn direct_dft(input: &[Complex32], inverse: bool) -> Vec<Complex32> {
            let n = input.len();
            let sign = if inverse { 1.0_f32 } else { -1.0_f32 };
            (0..n)
                .map(|k| {
                    input
                        .iter()
                        .enumerate()
                        .map(|(j, &value)| {
                            let angle = sign * std::f32::consts::TAU * (j * k) as f32 / n as f32;
                            value * Complex32::new(angle.cos(), angle.sin())
                        })
                        .sum()
                })
                .collect()
        }

        let fft_len = 8;
        let input: Vec<_> = (0..16)
            .map(|i| {
                let phase = i as f32 * 0.23;
                Complex32::new(phase.sin(), (phase * 0.61).cos())
            })
            .collect();
        let expected_forward: Vec<_> = input
            .chunks_exact(fft_len)
            .flat_map(|chunk| direct_dft(chunk, false))
            .collect();
        let expected_inverse: Vec<_> = expected_forward
            .chunks_exact(fft_len)
            .flat_map(|chunk| direct_dft(chunk, true))
            .collect();

        let planner = super::BackendFftPlanner::<_, f32>::new(backend.clone());
        let forward = planner.plan(fft_len, FftDirection::Forward);
        let inverse = planner.plan(fft_len, FftDirection::Inverse);
        assert_eq!(forward.execution(), FftExecution::BackendNative);
        assert_eq!(inverse.execution(), FftExecution::BackendNative);

        let mut buffer = backend
            .raw_alloc::<Complex32>(input.len())
            .expect("Metal FFT planner allocation");
        backend.copy_h2d(&input, &mut buffer);
        forward
            .process(&mut buffer)
            .expect("native Metal forward FFT");
        let mut actual_forward = vec![Complex32::default(); input.len()];
        backend.copy_d2h(&buffer, &mut actual_forward);
        for (actual, expected) in actual_forward.iter().zip(&expected_forward) {
            assert!((actual - expected).norm() < 4.0e-4);
        }

        inverse
            .process(&mut buffer)
            .expect("native Metal inverse FFT");
        let mut actual_inverse = vec![Complex32::default(); input.len()];
        backend.copy_d2h(&buffer, &mut actual_inverse);
        for (actual, expected) in actual_inverse.iter().zip(&expected_inverse) {
            assert!((actual - expected).norm() < 5.0e-4);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_backend_planner_executes_vendor_native_batched_fft() {
        let Some(backend) = cuda_backend_or_skip("FFT planner parity") else {
            return;
        };
        if backend.capabilities().fft.c64 != sonium_compute::CapabilityState::VendorNative {
            if std::env::var("SONIUM_REQUIRE_CUDA")
                .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
            {
                panic!("CUDA required by SONIUM_REQUIRE_CUDA=1 but cuFFT is unavailable");
            }
            eprintln!("cuFFT unavailable, skipping FFT planner parity");
            return;
        }

        fn direct_dft(input: &[Complex64], inverse: bool) -> Vec<Complex64> {
            let n = input.len();
            let sign = if inverse { 1.0 } else { -1.0 };
            (0..n)
                .map(|k| {
                    input
                        .iter()
                        .enumerate()
                        .map(|(j, &value)| {
                            let angle = sign * std::f64::consts::TAU * (j * k) as f64 / n as f64;
                            value * Complex64::new(angle.cos(), angle.sin())
                        })
                        .sum()
                })
                .collect()
        }

        // cuFFT supports regular non-power-of-two 1-D transforms, so keep
        // this planner-level regression on an irregular batched length.
        let fft_len = 7;
        let input: Vec<_> = (0..21)
            .map(|i| {
                let phase = i as f64 * 0.23;
                Complex64::new(phase.sin(), (phase * 0.61).cos())
            })
            .collect();
        let expected_forward: Vec<_> = input
            .chunks_exact(fft_len)
            .flat_map(|chunk| direct_dft(chunk, false))
            .collect();
        let expected_inverse: Vec<_> = expected_forward
            .chunks_exact(fft_len)
            .flat_map(|chunk| direct_dft(chunk, true))
            .collect();

        let planner = super::BackendFftPlanner::<_, f64>::new(backend.clone());
        let forward = planner.plan(fft_len, FftDirection::Forward);
        let inverse = planner.plan(fft_len, FftDirection::Inverse);
        assert_eq!(forward.execution(), FftExecution::VendorNative);
        assert_eq!(inverse.execution(), FftExecution::VendorNative);

        let mut buffer = backend
            .raw_alloc::<Complex64>(input.len())
            .expect("CUDA FFT planner allocation");
        backend.copy_h2d(&input, &mut buffer);
        forward
            .process(&mut buffer)
            .expect("vendor-native CUDA forward FFT");
        let mut actual_forward = vec![Complex64::default(); input.len()];
        backend.copy_d2h(&buffer, &mut actual_forward);
        for (actual, expected) in actual_forward.iter().zip(&expected_forward) {
            assert!((actual - expected).norm() < 1.0e-11);
        }

        inverse
            .process(&mut buffer)
            .expect("vendor-native CUDA inverse FFT");
        let mut actual_inverse = vec![Complex64::default(); input.len()];
        backend.copy_d2h(&buffer, &mut actual_inverse);
        for (actual, expected) in actual_inverse.iter().zip(&expected_inverse) {
            assert!((actual - expected).norm() < 1.0e-10);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fft_c32_c64_forward_inverse_and_strided_contract() {
        let Some(backend) = cuda_backend_or_skip("Complex32/Complex64 FFT parity") else {
            return;
        };
        let capabilities = backend.capabilities();
        if capabilities.fft.c32 != sonium_compute::CapabilityState::VendorNative
            || capabilities.fft.c64 != sonium_compute::CapabilityState::VendorNative
        {
            if std::env::var("SONIUM_REQUIRE_CUDA")
                .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
            {
                panic!(
                    "CUDA required by SONIUM_REQUIRE_CUDA=1 but both cuFFT complex precisions are not available"
                );
            }
            eprintln!("cuFFT complex precisions unavailable, skipping parity matrix");
            return;
        }

        let planner = super::BackendFftPlanner::<_, f32>::new(backend.clone());
        for &direction in &[FftDirection::Forward, FftDirection::Inverse] {
            let plan = planner.plan(7, direction);
            assert_eq!(plan.execution(), FftExecution::VendorNative);
            let input: Vec<Complex32> = (0..21)
                .map(|index| {
                    let phase = index as f32 * 0.31;
                    Complex32::new(phase.sin(), (phase * 0.47).cos())
                })
                .collect();
            let mut buffer = backend
                .raw_alloc::<Complex32>(input.len())
                .expect("CUDA Complex32 FFT allocation");
            backend.copy_h2d(&input, &mut buffer);
            plan.process(&mut buffer).expect("CUDA Complex32 FFT");
            let mut actual = vec![Complex32::default(); input.len()];
            backend.copy_d2h(&buffer, &mut actual);
            for (batch, chunk) in input.chunks_exact(7).enumerate() {
                let expected = direct_dft_f32(chunk, direction);
                for (index, (actual, expected)) in actual[batch * 7..(batch + 1) * 7]
                    .iter()
                    .zip(expected)
                    .enumerate()
                {
                    assert!(
                        (*actual - expected).norm() < 4.0e-4,
                        "CUDA Complex32 {direction:?} mismatch at batch {batch}, index {index}"
                    );
                }
            }

            let stride = 10;
            let required = 2 * stride + 7;
            let sentinel = Complex32::new(-17.0, 4.0);
            let mut strided_input = vec![sentinel; required + 2];
            for batch in 0..3 {
                for index in 0..7 {
                    let phase = (batch * 7 + index) as f32 * 0.19;
                    strided_input[batch * stride + index] =
                        Complex32::new(phase.sin(), (phase * 0.73).cos());
                }
            }
            let original_strided = strided_input.clone();
            let mut strided = backend
                .raw_alloc::<Complex32>(strided_input.len())
                .expect("CUDA strided Complex32 FFT allocation");
            backend.copy_h2d(&strided_input, &mut strided);
            plan.process_strided(&mut strided, 3, stride)
                .expect("CUDA strided Complex32 FFT");
            backend.copy_d2h(&strided, &mut strided_input);
            for batch in 0..3 {
                let expected = direct_dft_f32(
                    &original_strided[batch * stride..batch * stride + 7],
                    direction,
                );
                for (actual, expected) in strided_input[batch * stride..batch * stride + 7]
                    .iter()
                    .zip(expected)
                {
                    assert!((*actual - expected).norm() < 4.0e-4);
                }
            }
            assert_eq!(strided_input[7], sentinel);
            assert_eq!(strided_input[8], sentinel);
            assert_eq!(strided_input[9], sentinel);
            assert_eq!(strided_input[17], sentinel);
            assert_eq!(strided_input[18], sentinel);
            assert_eq!(strided_input[19], sentinel);
            assert_eq!(strided_input[required], sentinel);
            assert_eq!(strided_input[required + 1], sentinel);
        }

        let planner = super::BackendFftPlanner::<_, f64>::new(backend.clone());
        for &direction in &[FftDirection::Forward, FftDirection::Inverse] {
            let plan = planner.plan(7, direction);
            assert_eq!(plan.execution(), FftExecution::VendorNative);
            let input: Vec<Complex64> = (0..21)
                .map(|index| {
                    let phase = index as f64 * 0.31;
                    Complex64::new(phase.sin(), (phase * 0.47).cos())
                })
                .collect();
            let expected: Vec<_> = input
                .chunks_exact(7)
                .flat_map(|chunk| direct_dft_f64(chunk, matches!(direction, FftDirection::Inverse)))
                .collect();
            let mut buffer = backend
                .raw_alloc::<Complex64>(input.len())
                .expect("CUDA Complex64 FFT allocation");
            backend.copy_h2d(&input, &mut buffer);
            plan.process(&mut buffer).expect("CUDA Complex64 FFT");
            let mut actual = vec![Complex64::default(); input.len()];
            backend.copy_d2h(&buffer, &mut actual);
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((*actual - expected).norm() < 1.0e-11);
            }

            let stride = 10;
            let required = 2 * stride + 7;
            let sentinel = Complex64::new(-17.0, 4.0);
            let mut strided_input = vec![sentinel; required + 2];
            for batch in 0..3 {
                for index in 0..7 {
                    let phase = (batch * 7 + index) as f64 * 0.19;
                    strided_input[batch * stride + index] =
                        Complex64::new(phase.sin(), (phase * 0.73).cos());
                }
            }
            let original_strided = strided_input.clone();
            let mut strided = backend
                .raw_alloc::<Complex64>(strided_input.len())
                .expect("CUDA strided Complex64 FFT allocation");
            backend.copy_h2d(&strided_input, &mut strided);
            plan.process_strided(&mut strided, 3, stride)
                .expect("CUDA strided Complex64 FFT");
            backend.copy_d2h(&strided, &mut strided_input);
            for batch in 0..3 {
                let expected = direct_dft_f64(
                    &original_strided[batch * stride..batch * stride + 7],
                    matches!(direction, FftDirection::Inverse),
                );
                for (actual, expected) in strided_input[batch * stride..batch * stride + 7]
                    .iter()
                    .zip(expected)
                {
                    assert!((*actual - expected).norm() < 1.0e-11);
                }
            }
            assert_eq!(strided_input[7], sentinel);
            assert_eq!(strided_input[8], sentinel);
            assert_eq!(strided_input[9], sentinel);
            assert_eq!(strided_input[17], sentinel);
            assert_eq!(strided_input[18], sentinel);
            assert_eq!(strided_input[19], sentinel);
            assert_eq!(strided_input[required], sentinel);
            assert_eq!(strided_input[required + 1], sentinel);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fft_roundtrip_parseval_and_hermitian_adjoint_contract() {
        let Some(backend) = cuda_backend_or_skip("CUDA FFT identity checks") else {
            return;
        };
        let n = 15;
        let planner = super::BackendFftPlanner::<_, f64>::new(backend.clone());
        let forward = planner.plan(n, FftDirection::Forward);
        let inverse = planner.plan(n, FftDirection::Inverse);
        assert_eq!(forward.execution(), FftExecution::VendorNative);
        assert_eq!(inverse.execution(), FftExecution::VendorNative);

        let x: Vec<Complex64> = (0..n)
            .map(|index| {
                let phase = index as f64 * 0.17;
                Complex64::new(phase.sin(), (phase * 0.61).cos())
            })
            .collect();
        let y: Vec<Complex64> = (0..n)
            .map(|index| {
                let phase = index as f64 * 0.23;
                Complex64::new((phase * 0.7).cos(), phase.sin())
            })
            .collect();

        let mut x_device = backend
            .raw_alloc::<Complex64>(n)
            .expect("CUDA x allocation");
        backend.copy_h2d(&x, &mut x_device);
        forward
            .process(&mut x_device)
            .expect("CUDA forward identity transform");
        backend.try_synchronize().expect("CUDA forward completion");
        let mut fx = vec![Complex64::default(); n];
        backend.copy_d2h(&x_device, &mut fx);

        let input_energy: f64 = x.iter().map(Complex64::norm_sqr).sum();
        let transformed_energy: f64 = fx.iter().map(Complex64::norm_sqr).sum::<f64>() / n as f64;
        assert!(
            (input_energy - transformed_energy).abs() < 1.0e-10,
            "CUDA Parseval mismatch: input={input_energy}, transformed={transformed_energy}"
        );

        inverse
            .process(&mut x_device)
            .expect("CUDA inverse identity transform");
        backend
            .try_synchronize()
            .expect("CUDA roundtrip completion");
        let mut roundtrip = vec![Complex64::default(); n];
        backend.copy_d2h(&x_device, &mut roundtrip);
        for (actual, expected) in roundtrip.iter().zip(&x) {
            assert!((*actual / n as f64 - *expected).norm() < 1.0e-10);
        }

        let mut y_device = backend
            .raw_alloc::<Complex64>(n)
            .expect("CUDA y allocation");
        backend.copy_h2d(&y, &mut y_device);
        inverse
            .process(&mut y_device)
            .expect("CUDA inverse adjoint transform");
        backend.try_synchronize().expect("CUDA adjoint completion");
        let mut inverse_y = vec![Complex64::default(); n];
        backend.copy_d2h(&y_device, &mut inverse_y);
        let lhs: Complex64 = fx.iter().zip(&y).map(|(a, b)| a.conj() * b).sum();
        let rhs: Complex64 = x.iter().zip(&inverse_y).map(|(a, b)| a.conj() * b).sum();
        assert!(
            (lhs - rhs).norm() < 1.0e-9,
            "CUDA Hermitian-adjoint mismatch: lhs={lhs:?}, rhs={rhs:?}"
        );

        let empty = backend
            .raw_alloc::<Complex64>(0)
            .expect("CUDA empty allocation");
        let mut empty = empty;
        forward
            .process(&mut empty)
            .expect("empty CUDA FFT batch is a no-op");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fft_c32_roundtrip_parseval_and_hermitian_adjoint_contract() {
        let Some(backend) = cuda_backend_or_skip("CUDA Complex32 identity checks") else {
            return;
        };
        if backend.capabilities().fft.c32 != sonium_compute::CapabilityState::VendorNative {
            if std::env::var("SONIUM_REQUIRE_CUDA")
                .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
            {
                panic!("CUDA required by SONIUM_REQUIRE_CUDA=1 but Complex32 cuFFT is unavailable");
            }
            eprintln!("Complex32 cuFFT unavailable, skipping identity checks");
            return;
        }

        let n = 15;
        let planner = super::BackendFftPlanner::<_, f32>::new(backend.clone());
        let forward = planner.plan(n, FftDirection::Forward);
        let inverse = planner.plan(n, FftDirection::Inverse);
        assert_eq!(forward.execution(), FftExecution::VendorNative);
        assert_eq!(inverse.execution(), FftExecution::VendorNative);

        let x: Vec<Complex32> = (0..n)
            .map(|index| {
                let phase = index as f32 * 0.17;
                Complex32::new(phase.sin(), (phase * 0.61).cos())
            })
            .collect();
        let y: Vec<Complex32> = (0..n)
            .map(|index| {
                let phase = index as f32 * 0.23;
                Complex32::new((phase * 0.7).cos(), phase.sin())
            })
            .collect();

        let mut x_device = backend
            .raw_alloc::<Complex32>(n)
            .expect("CUDA x32 allocation");
        backend.copy_h2d(&x, &mut x_device);
        forward
            .process(&mut x_device)
            .expect("CUDA Complex32 forward identity transform");
        backend
            .try_synchronize()
            .expect("CUDA Complex32 forward completion");
        let mut fx = vec![Complex32::default(); n];
        backend.copy_d2h(&x_device, &mut fx);

        let input_energy: f32 = x.iter().map(Complex32::norm_sqr).sum();
        let transformed_energy: f32 = fx.iter().map(Complex32::norm_sqr).sum::<f32>() / n as f32;
        assert!(
            (input_energy - transformed_energy).abs() < 2.0e-3,
            "CUDA Complex32 Parseval mismatch: input={input_energy}, transformed={transformed_energy}"
        );

        inverse
            .process(&mut x_device)
            .expect("CUDA Complex32 inverse identity transform");
        backend
            .try_synchronize()
            .expect("CUDA Complex32 roundtrip completion");
        let mut roundtrip = vec![Complex32::default(); n];
        backend.copy_d2h(&x_device, &mut roundtrip);
        for (actual, expected) in roundtrip.iter().zip(&x) {
            assert!((*actual / n as f32 - *expected).norm() < 2.0e-3);
        }

        let mut y_device = backend
            .raw_alloc::<Complex32>(n)
            .expect("CUDA y32 allocation");
        backend.copy_h2d(&y, &mut y_device);
        inverse
            .process(&mut y_device)
            .expect("CUDA Complex32 inverse adjoint transform");
        backend
            .try_synchronize()
            .expect("CUDA Complex32 adjoint completion");
        let mut inverse_y = vec![Complex32::default(); n];
        backend.copy_d2h(&y_device, &mut inverse_y);
        let lhs: Complex32 = fx.iter().zip(&y).map(|(a, b)| a.conj() * b).sum();
        let rhs: Complex32 = x.iter().zip(&inverse_y).map(|(a, b)| a.conj() * b).sum();
        assert!(
            (lhs - rhs).norm() < 2.0e-2,
            "CUDA Complex32 Hermitian-adjoint mismatch: lhs={lhs:?}, rhs={rhs:?}"
        );

        let mut empty = backend
            .raw_alloc::<Complex32>(0)
            .expect("CUDA Complex32 empty allocation");
        forward
            .process(&mut empty)
            .expect("empty CUDA Complex32 batch is a no-op");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fft_injected_launch_and_allocation_failures_preserve_destination() {
        let Some(backend) = cuda_backend_or_skip("CUDA FFT failure injection") else {
            return;
        };
        if backend.capabilities().fft.c64 != sonium_compute::CapabilityState::VendorNative {
            eprintln!("cuFFT unavailable, skipping CUDA FFT failure injection");
            return;
        }
        let input: Vec<Complex64> = (0..8)
            .map(|index| Complex64::new(index as f64 + 1.0, -(index as f64) * 0.25))
            .collect();
        let planner = super::BackendFftPlanner::<_, f64>::new(backend.clone());
        let plan = planner.plan(8, FftDirection::Forward);
        let mut buffer = backend
            .raw_alloc::<Complex64>(input.len())
            .expect("CUDA FFT failure-injection allocation");
        backend.copy_h2d(&input, &mut buffer);
        backend.try_synchronize().expect("CUDA upload completion");

        for point in [
            sonium_compute::CudaFailureInjectionPoint::DescriptorCreation,
            sonium_compute::CudaFailureInjectionPoint::Allocation,
            sonium_compute::CudaFailureInjectionPoint::Launch,
        ] {
            backend.inject_failure(point);
            let error = plan
                .process(&mut buffer)
                .expect_err("injected CUDA FFT failure must be returned");
            assert!(matches!(
                (point, error),
                (
                    sonium_compute::CudaFailureInjectionPoint::DescriptorCreation,
                    sonium_compute::BackendError::OperationFailed { .. }
                ) | (
                    sonium_compute::CudaFailureInjectionPoint::Allocation,
                    sonium_compute::BackendError::Allocation { .. }
                ) | (
                    sonium_compute::CudaFailureInjectionPoint::Launch,
                    sonium_compute::BackendError::Launch { .. }
                )
            ));
            backend
                .recover_after_error()
                .expect("recoverable injected CUDA FFT failure");

            let mut actual = vec![Complex64::default(); input.len()];
            backend
                .try_copy_d2h(&buffer, &mut actual)
                .expect("CUDA destination inspection after rejection");
            assert_eq!(actual, input, "failure point {point:?} mutated destination");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fft_c32_injected_failures_preserve_destination() {
        let Some(backend) = cuda_backend_or_skip("CUDA Complex32 failure injection") else {
            return;
        };
        if backend.capabilities().fft.c32 != sonium_compute::CapabilityState::VendorNative {
            if std::env::var("SONIUM_REQUIRE_CUDA")
                .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
            {
                panic!("CUDA required by SONIUM_REQUIRE_CUDA=1 but Complex32 cuFFT is unavailable");
            }
            eprintln!("Complex32 cuFFT unavailable, skipping failure injection");
            return;
        }
        let input: Vec<Complex32> = (0..8)
            .map(|index| Complex32::new(index as f32 + 1.0, -(index as f32) * 0.25))
            .collect();
        let planner = super::BackendFftPlanner::<_, f32>::new(backend.clone());
        let plan = planner.plan(8, FftDirection::Forward);
        let mut buffer = backend
            .raw_alloc::<Complex32>(input.len())
            .expect("CUDA Complex32 FFT failure-injection allocation");
        backend.copy_h2d(&input, &mut buffer);
        backend
            .try_synchronize()
            .expect("CUDA Complex32 upload completion");

        for point in [
            sonium_compute::CudaFailureInjectionPoint::DescriptorCreation,
            sonium_compute::CudaFailureInjectionPoint::Allocation,
            sonium_compute::CudaFailureInjectionPoint::Launch,
        ] {
            backend.inject_failure(point);
            let error = plan
                .process(&mut buffer)
                .expect_err("injected CUDA Complex32 failure must be returned");
            assert!(matches!(
                (point, error),
                (
                    sonium_compute::CudaFailureInjectionPoint::DescriptorCreation,
                    sonium_compute::BackendError::OperationFailed { .. }
                ) | (
                    sonium_compute::CudaFailureInjectionPoint::Allocation,
                    sonium_compute::BackendError::Allocation { .. }
                ) | (
                    sonium_compute::CudaFailureInjectionPoint::Launch,
                    sonium_compute::BackendError::Launch { .. }
                )
            ));
            backend
                .recover_after_error()
                .expect("recoverable injected CUDA Complex32 failure");

            let mut actual = vec![Complex32::default(); input.len()];
            backend
                .try_copy_d2h(&buffer, &mut actual)
                .expect("CUDA Complex32 destination inspection after rejection");
            assert_eq!(actual, input, "failure point {point:?} mutated destination");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fft_malformed_buffers_are_rejected_without_touching_destination() {
        let Some(backend) = cuda_backend_or_skip("CUDA FFT malformed-buffer contract") else {
            return;
        };
        if backend.capabilities().fft.c64 != sonium_compute::CapabilityState::VendorNative {
            eprintln!("cuFFT unavailable, skipping CUDA malformed-buffer contract");
            return;
        }
        let planner = super::BackendFftPlanner::<_, f64>::new(backend.clone());
        let plan = planner.plan(8, FftDirection::Forward);

        let input: Vec<Complex64> = (0..7)
            .map(|index| Complex64::new(index as f64 + 0.5, -(index as f64)))
            .collect();
        let mut buffer = backend
            .raw_alloc::<Complex64>(input.len())
            .expect("CUDA malformed-buffer allocation");
        backend.copy_h2d(&input, &mut buffer);
        let error = plan
            .process(&mut buffer)
            .expect_err("short non-multiple CUDA FFT buffer must be rejected");
        assert!(matches!(
            error,
            sonium_compute::BackendError::InvalidArgument { .. }
        ));
        let mut actual = vec![Complex64::default(); input.len()];
        backend
            .try_copy_d2h(&buffer, &mut actual)
            .expect("CUDA malformed-buffer destination inspection");
        assert_eq!(actual, input);

        let mut strided = backend
            .raw_alloc::<Complex64>(18)
            .expect("CUDA malformed-stride allocation");
        let strided_input = vec![Complex64::new(-3.0, 2.0); 18];
        backend.copy_h2d(&strided_input, &mut strided);
        let error = plan
            .process_strided(&mut strided, 2, 7)
            .expect_err("stride shorter than the FFT length must be rejected");
        assert!(matches!(
            error,
            sonium_compute::BackendError::InvalidArgument { .. }
        ));
        let mut actual = vec![Complex64::default(); strided_input.len()];
        backend
            .try_copy_d2h(&strided, &mut actual)
            .expect("CUDA malformed-stride destination inspection");
        assert_eq!(actual, strided_input);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fft_c32_malformed_buffers_are_rejected_without_touching_destination() {
        let Some(backend) = cuda_backend_or_skip("CUDA Complex32 malformed-buffer contract") else {
            return;
        };
        if backend.capabilities().fft.c32 != sonium_compute::CapabilityState::VendorNative {
            if std::env::var("SONIUM_REQUIRE_CUDA")
                .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
            {
                panic!("CUDA required by SONIUM_REQUIRE_CUDA=1 but Complex32 cuFFT is unavailable");
            }
            eprintln!("Complex32 cuFFT unavailable, skipping malformed-buffer contract");
            return;
        }
        let planner = super::BackendFftPlanner::<_, f32>::new(backend.clone());
        let plan = planner.plan(8, FftDirection::Forward);

        let input: Vec<Complex32> = (0..7)
            .map(|index| Complex32::new(index as f32 + 0.5, -(index as f32)))
            .collect();
        let mut buffer = backend
            .raw_alloc::<Complex32>(input.len())
            .expect("CUDA Complex32 malformed-buffer allocation");
        backend.copy_h2d(&input, &mut buffer);
        let error = plan
            .process(&mut buffer)
            .expect_err("short non-multiple CUDA Complex32 FFT buffer must be rejected");
        assert!(matches!(
            error,
            sonium_compute::BackendError::InvalidArgument { .. }
        ));
        let mut actual = vec![Complex32::default(); input.len()];
        backend
            .try_copy_d2h(&buffer, &mut actual)
            .expect("CUDA Complex32 malformed-buffer destination inspection");
        assert_eq!(actual, input);

        let mut strided = backend
            .raw_alloc::<Complex32>(18)
            .expect("CUDA Complex32 malformed-stride allocation");
        let strided_input = vec![Complex32::new(-3.0, 2.0); 18];
        backend.copy_h2d(&strided_input, &mut strided);
        let error = plan
            .process_strided(&mut strided, 2, 7)
            .expect_err("Complex32 stride shorter than the FFT length must be rejected");
        assert!(matches!(
            error,
            sonium_compute::BackendError::InvalidArgument { .. }
        ));
        let mut actual = vec![Complex32::default(); strided_input.len()];
        backend
            .try_copy_d2h(&strided, &mut actual)
            .expect("CUDA Complex32 malformed-stride destination inspection");
        assert_eq!(actual, strided_input);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fft_device_loss_is_a_terminal_checked_error() {
        let Some(backend) = cuda_backend_or_skip("CUDA FFT device-loss injection") else {
            return;
        };
        if backend.capabilities().fft.c64 != sonium_compute::CapabilityState::VendorNative {
            eprintln!("cuFFT unavailable, skipping CUDA FFT device-loss injection");
            return;
        }
        let planner = super::BackendFftPlanner::<_, f64>::new(backend.clone());
        let plan = planner.plan(8, FftDirection::Forward);
        let mut buffer = backend
            .raw_alloc::<Complex64>(8)
            .expect("CUDA device-loss test allocation");
        backend.inject_failure(sonium_compute::CudaFailureInjectionPoint::DeviceLoss);
        let error = plan
            .process(&mut buffer)
            .expect_err("injected CUDA device loss must be returned");
        assert!(matches!(
            error,
            sonium_compute::BackendError::DeviceLost { .. }
        ));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fft_c32_device_loss_is_a_terminal_checked_error() {
        let Some(backend) = cuda_backend_or_skip("CUDA Complex32 device-loss injection") else {
            return;
        };
        if backend.capabilities().fft.c32 != sonium_compute::CapabilityState::VendorNative {
            if std::env::var("SONIUM_REQUIRE_CUDA")
                .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
            {
                panic!("CUDA required by SONIUM_REQUIRE_CUDA=1 but Complex32 cuFFT is unavailable");
            }
            eprintln!("Complex32 cuFFT unavailable, skipping device-loss injection");
            return;
        }
        let planner = super::BackendFftPlanner::<_, f32>::new(backend.clone());
        let plan = planner.plan(8, FftDirection::Forward);
        let mut buffer = backend
            .raw_alloc::<Complex32>(8)
            .expect("CUDA Complex32 device-loss test allocation");
        backend.inject_failure(sonium_compute::CudaFailureInjectionPoint::DeviceLoss);
        let error = plan
            .process(&mut buffer)
            .expect_err("injected CUDA Complex32 device loss must be returned");
        assert!(matches!(
            error,
            sonium_compute::BackendError::DeviceLost { .. }
        ));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_real_fft_dct_dst_are_vendor_native_and_match_cpu() {
        let Some(backend) = cuda_backend_or_skip("CUDA real FFT/DCT/DST parity") else {
            return;
        };
        let capabilities = backend.capabilities();
        let native = sonium_compute::CapabilityState::VendorNative;
        if capabilities.real_fft.f64 != native
            || capabilities.real_fft.f32 != native
            || capabilities.dct.f64 != native
            || capabilities.dct.f32 != native
            || capabilities.dst.f64 != native
            || capabilities.dst.f32 != native
        {
            if std::env::var("SONIUM_REQUIRE_CUDA")
                .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
            {
                panic!("CUDA real FFT/DCT/DST capability set is incomplete: {capabilities:?}");
            }
            eprintln!("CUDA real FFT/DCT/DST unavailable, skipping parity");
            return;
        }

        let n = 15;
        let batch = 2;
        let input: Vec<f64> = (0..n * batch)
            .map(|index| (0.13 * index as f64).sin() + 0.07 * (0.31 * index as f64).cos())
            .collect();
        let mut input_device = backend.raw_alloc::<f64>(input.len()).unwrap();
        backend.copy_h2d(&input, &mut input_device);

        let real_planner = super::BackendRealFftPlanner::<_, f64>::new(backend.clone());
        let forward = real_planner.plan_fft_forward(n);
        let inverse = real_planner.plan_fft_inverse(n);
        assert_eq!(forward.execution(), FftExecution::VendorNative);
        assert_eq!(inverse.execution(), FftExecution::VendorNative);
        let spectrum_len = n / 2 + 1;
        let mut spectrum_device = backend
            .raw_alloc::<Complex64>(spectrum_len * batch)
            .unwrap();
        forward
            .process(&input_device, &mut spectrum_device)
            .expect("CUDA real forward FFT");
        let mut spectrum = vec![Complex64::default(); spectrum_len * batch];
        backend.copy_d2h(&spectrum_device, &mut spectrum);
        let cpu_real = crate::RealFftPlan::<f64>::new(n);
        let expected_spectrum: Vec<_> = input
            .chunks_exact(n)
            .flat_map(|chunk| cpu_real.forward(chunk))
            .collect();
        for (actual, expected) in spectrum.iter().zip(expected_spectrum) {
            assert!((*actual - expected).norm() < 1.0e-10);
        }

        let mut recovered_device = backend.raw_alloc::<f64>(input.len()).unwrap();
        inverse
            .process(&spectrum_device, &mut recovered_device)
            .expect("CUDA real inverse FFT");
        let mut recovered = vec![0.0; input.len()];
        backend.copy_d2h(&recovered_device, &mut recovered);
        for (actual, expected) in recovered.iter().zip(&input) {
            assert!((actual - expected).abs() < 1.0e-10);
        }

        let dct = super::BackendDctPlanner::<_, f64>::new(backend.clone()).plan(n);
        let dst = super::BackendDstPlanner::<_, f64>::new(backend.clone()).plan(n);
        assert_eq!(dct.execution(), FftExecution::VendorNative);
        assert_eq!(dst.execution(), FftExecution::VendorNative);
        let one_input = &input[..n];
        let mut one_input_device = backend.raw_alloc::<f64>(n).unwrap();
        let mut output_device = backend.raw_alloc::<f64>(n).unwrap();
        backend.copy_h2d(one_input, &mut one_input_device);
        let mut actual = vec![0.0; n];
        let mut expected = vec![0.0; n];

        let dct_ops: &[DctCudaCase] = &[
            ("dct2", super::DeviceDctPlan::forward, crate::dct2),
            ("dct3", super::DeviceDctPlan::inverse, crate::dct3),
            (
                "dct2_adjoint",
                super::DeviceDctPlan::forward_adjoint,
                crate::dct2_adjoint,
            ),
            (
                "dct3_adjoint",
                super::DeviceDctPlan::inverse_adjoint,
                crate::dct3_adjoint,
            ),
        ];
        for &(label, native_op, reference_op) in dct_ops {
            reference_op(one_input, &mut expected);
            native_op(&dct, &one_input_device, &mut output_device)
                .unwrap_or_else(|error| panic!("CUDA {label}: {error}"));
            backend.copy_d2h(&output_device, &mut actual);
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() < 1.0e-10,
                    "CUDA {label} mismatch at {index}: actual={actual:?}, expected={expected:?}"
                );
            }
        }

        let dst_ops: &[DstCudaCase] = &[
            ("dst2", super::DeviceDstPlan::forward, crate::dst2),
            ("dst3", super::DeviceDstPlan::inverse, crate::dst3),
            (
                "dst2_adjoint",
                super::DeviceDstPlan::forward_adjoint,
                crate::dst2_adjoint,
            ),
            (
                "dst3_adjoint",
                super::DeviceDstPlan::inverse_adjoint,
                crate::dst3_adjoint,
            ),
        ];
        for &(label, native_op, reference_op) in dst_ops {
            reference_op(one_input, &mut expected);
            native_op(&dst, &one_input_device, &mut output_device)
                .unwrap_or_else(|error| panic!("CUDA {label}: {error}"));
            backend.copy_d2h(&output_device, &mut actual);
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() < 1.0e-10,
                    "CUDA {label} mismatch at {index}: actual={actual:?}, expected={expected:?}"
                );
            }
        }

        let input32: Vec<f32> = input.iter().map(|&value| value as f32).collect();
        let mut input32_device = backend.raw_alloc::<f32>(input32.len()).unwrap();
        backend.copy_h2d(&input32, &mut input32_device);
        let real32 = super::BackendRealFftPlanner::<_, f32>::new(backend.clone());
        let forward32 = real32.plan_fft_forward(n);
        let inverse32 = real32.plan_fft_inverse(n);
        assert_eq!(forward32.execution(), FftExecution::VendorNative);
        let mut spectrum32_device = backend
            .raw_alloc::<Complex32>(spectrum_len * batch)
            .unwrap();
        forward32
            .process(&input32_device, &mut spectrum32_device)
            .expect("CUDA f32 real forward FFT");
        let mut spectrum32 = vec![Complex32::default(); spectrum_len * batch];
        backend.copy_d2h(&spectrum32_device, &mut spectrum32);
        let cpu_real32 = crate::RealFftPlan::<f32>::new(n);
        let expected_spectrum32: Vec<_> = input32
            .chunks_exact(n)
            .flat_map(|chunk| cpu_real32.forward(chunk))
            .collect();
        for (actual, expected) in spectrum32.iter().zip(expected_spectrum32) {
            assert!((*actual - expected).norm() < 3.0e-4);
        }
        let mut recovered32_device = backend.raw_alloc::<f32>(input32.len()).unwrap();
        inverse32
            .process(&spectrum32_device, &mut recovered32_device)
            .expect("CUDA f32 real inverse FFT");
        let mut recovered32 = vec![0.0f32; input32.len()];
        backend.copy_d2h(&recovered32_device, &mut recovered32);
        for (actual, expected) in recovered32.iter().zip(&input32) {
            assert!((actual - expected).abs() < 3.0e-5);
        }

        let dct32 = super::BackendDctPlanner::<_, f32>::new(backend.clone()).plan(n);
        let dst32 = super::BackendDstPlanner::<_, f32>::new(backend.clone()).plan(n);
        let mut input32_one_device = backend.raw_alloc::<f32>(n).unwrap();
        let mut output32_device = backend.raw_alloc::<f32>(n).unwrap();
        backend.copy_h2d(&input32[..n], &mut input32_one_device);
        let mut actual32 = vec![0.0f32; n];
        let mut expected32 = vec![0.0f32; n];
        dct32
            .forward(&input32_one_device, &mut output32_device)
            .unwrap();
        backend.copy_d2h(&output32_device, &mut actual32);
        crate::dct2(&input32[..n], &mut expected32);
        for (actual, expected) in actual32.iter().zip(&expected32) {
            assert!((actual - expected).abs() < 3.0e-4);
        }
        dst32
            .forward(&input32_one_device, &mut output32_device)
            .unwrap();
        backend.copy_d2h(&output32_device, &mut actual32);
        crate::dst2(&input32[..n], &mut expected32);
        for (actual, expected) in actual32.iter().zip(&expected32) {
            assert!((actual - expected).abs() < 3.0e-4);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_real_transform_edge_lengths_match_cpu() {
        let Some(backend) = cuda_backend_or_skip("CUDA real transform edge lengths") else {
            return;
        };
        let capabilities = backend.capabilities();
        let native = sonium_compute::CapabilityState::VendorNative;
        if capabilities.real_fft.f64 != native
            || capabilities.dct.f64 != native
            || capabilities.dst.f64 != native
        {
            if std::env::var("SONIUM_REQUIRE_CUDA")
                .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
            {
                panic!("CUDA real transform edge-length capabilities are incomplete");
            }
            return;
        }

        let input_host = |n: usize| {
            (0..n)
                .map(|index| (0.17 * index as f64).sin() - 0.11 * (0.29 * index as f64).cos())
                .collect::<Vec<_>>()
        };
        for &n in &[1, 2, 3, 4, 16] {
            let input = input_host(n);
            let mut input_device = backend.raw_alloc::<f64>(n).unwrap();
            backend.copy_h2d(&input, &mut input_device);

            let real = super::BackendRealFftPlanner::<_, f64>::new(backend.clone());
            let forward = real.plan_fft_forward(n);
            let inverse = real.plan_fft_inverse(n);
            let spectrum_len = n / 2 + 1;
            let mut spectrum_device = backend.raw_alloc::<Complex64>(spectrum_len).unwrap();
            forward
                .process(&input_device, &mut spectrum_device)
                .unwrap_or_else(|error| panic!("CUDA real FFT length {n}: {error}"));
            let mut spectrum = vec![Complex64::default(); spectrum_len];
            backend.copy_d2h(&spectrum_device, &mut spectrum);
            let expected_spectrum = crate::RealFftPlan::<f64>::new(n).forward(&input);
            for (actual, expected) in spectrum.iter().zip(expected_spectrum) {
                assert!((actual - expected).norm() < 1.0e-10, "real FFT length {n}");
            }
            let mut recovered_device = backend.raw_alloc::<f64>(n).unwrap();
            inverse
                .process(&spectrum_device, &mut recovered_device)
                .unwrap_or_else(|error| panic!("CUDA real inverse FFT length {n}: {error}"));
            let mut recovered = vec![0.0; n];
            backend.copy_d2h(&recovered_device, &mut recovered);
            for (actual, expected) in recovered.iter().zip(&input) {
                assert!(
                    (actual - expected).abs() < 1.0e-10,
                    "real inverse FFT length {n}"
                );
            }

            let dct = super::BackendDctPlanner::<_, f64>::new(backend.clone()).plan(n);
            let dst = super::BackendDstPlanner::<_, f64>::new(backend.clone()).plan(n);
            let mut output_device = backend.raw_alloc::<f64>(n).unwrap();
            let mut actual = vec![0.0; n];
            let mut expected = vec![0.0; n];
            let dct_ops: &[DctCudaCase] = &[
                ("dct2", super::DeviceDctPlan::forward, crate::dct2),
                ("dct3", super::DeviceDctPlan::inverse, crate::dct3),
                (
                    "dct2_adjoint",
                    super::DeviceDctPlan::forward_adjoint,
                    crate::dct2_adjoint,
                ),
                (
                    "dct3_adjoint",
                    super::DeviceDctPlan::inverse_adjoint,
                    crate::dct3_adjoint,
                ),
            ];
            for &(label, native_op, reference_op) in dct_ops {
                reference_op(&input, &mut expected);
                native_op(&dct, &input_device, &mut output_device)
                    .unwrap_or_else(|error| panic!("CUDA {label} length {n}: {error}"));
                backend.copy_d2h(&output_device, &mut actual);
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert!(
                        (actual - expected).abs() < 1.0e-10,
                        "CUDA {label} length {n}"
                    );
                }
            }
            let dst_ops: &[DstCudaCase] = &[
                ("dst2", super::DeviceDstPlan::forward, crate::dst2),
                ("dst3", super::DeviceDstPlan::inverse, crate::dst3),
                (
                    "dst2_adjoint",
                    super::DeviceDstPlan::forward_adjoint,
                    crate::dst2_adjoint,
                ),
                (
                    "dst3_adjoint",
                    super::DeviceDstPlan::inverse_adjoint,
                    crate::dst3_adjoint,
                ),
            ];
            for &(label, native_op, reference_op) in dst_ops {
                reference_op(&input, &mut expected);
                native_op(&dst, &input_device, &mut output_device)
                    .unwrap_or_else(|error| panic!("CUDA {label} length {n}: {error}"));
                backend.copy_d2h(&output_device, &mut actual);
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert!(
                        (actual - expected).abs() < 1.0e-10,
                        "CUDA {label} length {n}"
                    );
                }
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn direct_dft_f64(input: &[Complex64], inverse: bool) -> Vec<Complex64> {
        let n = input.len();
        let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
        (0..n)
            .map(|k| {
                input
                    .iter()
                    .enumerate()
                    .map(|(j, value)| {
                        let angle = sign * std::f64::consts::TAU * (j * k) as f64 / n as f64;
                        *value * Complex64::new(angle.cos(), angle.sin())
                    })
                    .sum()
            })
            .collect()
    }

    #[cfg(feature = "cuda")]
    fn direct_dft_f32(input: &[Complex32], direction: FftDirection) -> Vec<Complex32> {
        let sign = if direction == FftDirection::Forward {
            -1.0_f32
        } else {
            1.0_f32
        };
        let n = input.len() as f32;
        (0..input.len())
            .map(|k| {
                input
                    .iter()
                    .enumerate()
                    .map(|(j, value)| {
                        let angle = sign * std::f32::consts::TAU * (j * k) as f32 / n;
                        *value * Complex32::new(angle.cos(), angle.sin())
                    })
                    .sum()
            })
            .collect()
    }
}
