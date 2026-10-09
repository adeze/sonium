use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use num_complex::Complex32;
use num_complex::Complex64;
use sonium_fft::rustfft_compat::FftPlanner as CompatPlanner;
use sonium_fft::{
    CircularConvolutionPlan, Fft2dPlan, Fft3dPlan, FftDirection, FftPlan, LinearConvolutionPlan,
    SixStepFftPlan, process_strided,
};

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static TEST_LOCK: Mutex<()> = Mutex::new(());

struct CountingAlloc;

thread_local! {
    static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
}

struct AllocationTrackingGuard;

impl AllocationTrackingGuard {
    fn new() -> Self {
        TRACK_ALLOCATIONS.with(|tracking| {
            assert!(!tracking.replace(true), "nested allocation tracking");
        });
        Self
    }
}

impl Drop for AllocationTrackingGuard {
    fn drop(&mut self) {
        TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
    }
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        TRACK_ALLOCATIONS.with(|tracking| {
            if tracking.get() && layout.size() != 0 && !ptr.is_null() {
                ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            }
        });
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        TRACK_ALLOCATIONS.with(|tracking| {
            if tracking.get() && new_size != 0 && !new_ptr.is_null() {
                ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            }
        });
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn input(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| Complex64::new((i as f64 * 0.013).sin(), (i as f64 * 0.019).cos()))
        .collect()
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn input_f32(n: usize) -> Vec<Complex32> {
    (0..n)
        .map(|i| Complex32::new((i as f32 * 0.013).sin(), (i as f32 * 0.019).cos()))
        .collect()
}

fn assert_no_allocs(label: &str, f: impl FnOnce()) {
    ALLOCATIONS.store(0, Ordering::SeqCst);
    let tracking = AllocationTrackingGuard::new();
    f();
    drop(tracking);
    assert_eq!(ALLOCATIONS.load(Ordering::SeqCst), 0, "{label}");
}

#[test]
fn fft_execution_allocates_zero_heap_blocks_after_planning() {
    let _guard = TEST_LOCK.lock().expect("allocation test lock poisoned");
    let native = sonium_fft::FftPlanner::<f64>::new();

    let smooth = native.plan(1024, FftDirection::Forward);
    let mut smooth_buf = input(1024);
    assert_no_allocs("native smooth process allocated", || {
        smooth.process(&mut smooth_buf);
    });

    let mut smooth_scratch = vec![Complex64::new(0.0, 0.0); smooth.scratch_len()];
    assert_no_allocs("native smooth process_with_scratch allocated", || {
        smooth.process_with_scratch(&mut smooth_buf, &mut smooth_scratch);
    });

    let bluestein = native.plan(1031, FftDirection::Forward);
    let mut bluestein_buf = input(1031);
    assert_no_allocs("native Bluestein process allocated", || {
        bluestein.process(&mut bluestein_buf);
    });

    let mut bluestein_scratch = vec![Complex64::new(0.0, 0.0); bluestein.scratch_len()];
    assert_no_allocs("native Bluestein process_with_scratch allocated", || {
        bluestein.process_with_scratch(&mut bluestein_buf, &mut bluestein_scratch);
    });

    let mut compat_planner = CompatPlanner::<f64>::new();
    let compat = compat_planner.plan_fft_forward(1024);
    let mut compat_buf = input(1024);
    assert_no_allocs("rustfft compat process allocated", || {
        compat.process(&mut compat_buf);
    });
}

#[test]
fn multidimensional_and_strided_scratch_execution_allocates_zero_heap_blocks() {
    let _guard = TEST_LOCK.lock().expect("allocation test lock poisoned");

    let two_d = Fft2dPlan::<f64>::new(7, 5, FftDirection::Forward);
    let mut two_d_buffer = input(35);
    let mut two_d_scratch = vec![Complex64::new(0.0, 0.0); two_d.scratch_len()];
    // Optional compute backends may lazily provision runtime state on the
    // first execution. Measure the reusable steady-state contract below.
    two_d.process_with_scratch(&mut two_d_buffer, &mut two_d_scratch);
    assert_no_allocs("2D Bluestein scratch process allocated", || {
        two_d.process_with_scratch(&mut two_d_buffer, &mut two_d_scratch);
    });

    let three_d = Fft3dPlan::<f64>::new(7, 2, 2, FftDirection::Forward);
    let mut three_d_buffer = input(28);
    let mut three_d_scratch = vec![Complex64::new(0.0, 0.0); three_d.scratch_len()];
    three_d.process_with_scratch(&mut three_d_buffer, &mut three_d_scratch);
    assert_no_allocs("3D Bluestein scratch process allocated", || {
        three_d.process_with_scratch(&mut three_d_buffer, &mut three_d_scratch);
    });

    let planner = sonium_fft::FftPlanner::<f64>::new();
    let strided = planner.plan(1031, FftDirection::Forward);
    let batch = 2;
    let stride = 1040;
    let mut strided_buffer = vec![Complex64::new(0.0, 0.0); (batch - 1) * stride + 1031];
    let mut strided_scratch = vec![Complex64::new(0.0, 0.0); 1031 + strided.scratch_len()];
    process_strided(
        &*strided,
        &mut strided_buffer,
        batch,
        stride,
        &mut strided_scratch,
    );
    assert_no_allocs("strided Bluestein scratch process allocated", || {
        process_strided(
            &*strided,
            &mut strided_buffer,
            batch,
            stride,
            &mut strided_scratch,
        );
    });
}

#[test]
fn six_step_bluestein_axes_use_caller_scratch() {
    let _guard = TEST_LOCK.lock().expect("allocation test lock poisoned");
    let plan = SixStepFftPlan::<f64>::new(77, FftDirection::Forward);
    let mut buffer = input(plan.len());
    let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len()];
    assert_no_allocs("six-step Bluestein axis process allocated", || {
        plan.process_with_scratch(&mut buffer, &mut scratch);
    });
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_batched_process_reuses_pair_buffers_after_warmup() {
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        return;
    }
    let _guard = TEST_LOCK.lock().expect("allocation test lock poisoned");
    let native = sonium_fft::FftPlanner::<f64>::new();
    let plan = native.plan(32, FftDirection::Forward);

    // The first batched call provisions the thread-local interleaving
    // buffers. Keep it outside the measured region.
    let mut warmup = input(32 * 3);
    plan.process(&mut warmup);

    let mut steady_state = input(32 * 3);
    assert_no_allocs("AVX2 batched process allocated after warmup", || {
        plan.process(&mut steady_state);
    });
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn batched_f32_process_reuses_pair_buffers_after_warmup() {
    #[cfg(target_arch = "x86_64")]
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        return;
    }
    let _guard = TEST_LOCK.lock().expect("allocation test lock poisoned");
    let native = sonium_fft::FftPlanner::<f32>::new();
    let plan = native.plan(32, FftDirection::Forward);

    let mut warmup = input_f32(32 * 3);
    plan.process(&mut warmup);

    let mut steady_state = input_f32(32 * 3);
    assert_no_allocs("AVX2 f32 batched process allocated after warmup", || {
        plan.process(&mut steady_state);
    });
}

#[test]
fn convolution_process_with_scratch_allocates_zero_after_warmup() {
    let _guard = TEST_LOCK.lock().expect("allocation test lock poisoned");
    let plan = LinearConvolutionPlan::<f64>::new(31, 17);
    let lhs = input(plan.lhs_len());
    let rhs = input(plan.rhs_len());
    let mut output = vec![Complex64::new(0.0, 0.0); plan.output_len()];
    let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len()];

    plan.process_with_scratch(&lhs, &rhs, &mut output, &mut scratch);
    assert_no_allocs(
        "convolution process_with_scratch allocated after warmup",
        || {
            plan.process_with_scratch(&lhs, &rhs, &mut output, &mut scratch);
        },
    );

    let circular = CircularConvolutionPlan::<f64>::new(31);
    let mut circular_output = vec![Complex64::new(0.0, 0.0); circular.len()];
    let mut circular_scratch = vec![Complex64::new(0.0, 0.0); circular.scratch_len()];
    circular.process_with_scratch(&lhs, &lhs, &mut circular_output, &mut circular_scratch);
    assert_no_allocs("circular convolution allocated after warmup", || {
        circular.process_with_scratch(&lhs, &lhs, &mut circular_output, &mut circular_scratch);
    });
}
