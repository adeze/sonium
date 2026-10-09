//! Emit a deterministic machine-readable CPU FFT benchmark artifact.
//!
//! The benchmark separates plan construction from warmed steady-state
//! execution and records the shape of every measured case. It intentionally
//! reports allocation instrumentation and energy as unavailable rather than
//! inferring them from timing.

use num_complex::{Complex32, Complex64};
use sonium_fft::{
    CpuFftDispatch, Fft2dPlan, Fft3dPlan, FftDirection, FftPlan, FftPlanner, LinearConvolutionPlan,
    process_strided,
};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

const BATCHES: usize = 24;

#[derive(Clone, Copy)]
enum CaseKind {
    Complex,
    Strided {
        stride: usize,
    },
    TwoD {
        rows: usize,
        cols: usize,
    },
    ThreeD {
        depth: usize,
        rows: usize,
        cols: usize,
    },
}

#[derive(Clone, Copy)]
struct Case {
    n: usize,
    batch: usize,
    label: &'static str,
    kind: CaseKind,
}

enum BenchPlan {
    Complex(Arc<dyn FftPlan<f64>>),
    Strided(Arc<dyn FftPlan<f64>>),
    TwoD(Fft2dPlan<f64>),
    ThreeD(Fft3dPlan<f64>),
}

#[derive(Clone, Copy)]
struct Timing {
    plan_ns: u128,
    ns_per_batch: f64,
    p95_batch_ns: u128,
    p99_batch_ns: u128,
    scratch_bytes: usize,
    bytes_per_second: f64,
    estimated_gflops: f64,
}

fn percentile(values: &mut [u128], percentile: usize) -> u128 {
    values.sort_unstable();
    values[((values.len() - 1) * percentile) / 100]
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn cpu_description() -> String {
    #[cfg(target_os = "macos")]
    {
        for key in ["machdep.cpu.brand_string", "hw.model"] {
            if let Ok(output) = std::process::Command::new("sysctl")
                .args(["-n", key])
                .output()
                && output.status.success()
            {
                let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
                if !value.is_empty() {
                    return value;
                }
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/cpuinfo")
            && let Some(model) = contents
                .lines()
                .find_map(|line| line.strip_prefix("model name\t: "))
        {
            return model.to_owned();
        }
    }
    "unknown".to_owned()
}

fn target_features() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return "avx2,fma";
        }
        return "scalar";
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "neon";
    }
    #[allow(unreachable_code)]
    "scalar"
}

fn dispatch() -> &'static str {
    requested_dispatch().name()
}

fn requested_dispatch() -> CpuFftDispatch {
    match std::env::var("SONIUM_FFT_DISPATCH").as_deref() {
        Ok("scalar") => CpuFftDispatch::Scalar,
        Ok("avx2") => CpuFftDispatch::Avx2,
        _ => CpuFftDispatch::Auto,
    }
}

fn make_input(len: usize, seed: u64) -> Vec<Complex64> {
    let mut state = 0xC0FFEE_u64 ^ len as u64 ^ seed.rotate_left(17);
    (0..len)
        .map(|i| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let re = (state as f64 / u64::MAX as f64) - 0.5;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let im = (state as f64 / u64::MAX as f64) - 0.5;
            Complex64::new(re + (i as f64 * 1.0e-7), im)
        })
        .collect()
}

fn make_input_f32(len: usize, seed: u64) -> Vec<Complex32> {
    let mut state = 0xF32C_0FFE_u64 ^ len as u64 ^ seed.rotate_left(11);
    (0..len)
        .map(|i| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let re = (state as f32 / u64::MAX as f32) - 0.5;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let im = (state as f32 / u64::MAX as f32) - 0.5;
            Complex32::new(re + (i as f32 * 1.0e-5), im)
        })
        .collect()
}

fn measure_case(case: Case) -> Timing {
    let planner = FftPlanner::<f64>::with_dispatch(requested_dispatch());
    let plan_start = Instant::now();
    let plan = match case.kind {
        CaseKind::Complex => BenchPlan::Complex(planner.plan(case.n, FftDirection::Forward)),
        CaseKind::Strided { .. } => BenchPlan::Strided(planner.plan(case.n, FftDirection::Forward)),
        CaseKind::TwoD { rows, cols } => BenchPlan::TwoD(Fft2dPlan::with_planner(
            rows,
            cols,
            FftDirection::Forward,
            &planner,
        )),
        CaseKind::ThreeD { depth, rows, cols } => BenchPlan::ThreeD(Fft3dPlan::with_planner(
            depth,
            rows,
            cols,
            FftDirection::Forward,
            &planner,
        )),
    };
    let plan_ns = plan_start.elapsed().as_nanos();

    let buffer_len = match case.kind {
        CaseKind::Complex => case.n * case.batch,
        CaseKind::Strided { stride } => (case.batch - 1) * stride + case.n,
        CaseKind::TwoD { rows, cols } => rows * cols,
        CaseKind::ThreeD { depth, rows, cols } => depth * rows * cols,
    };
    let input = make_input(buffer_len, case.n as u64 ^ case.batch as u64);
    let mut buffer = input.clone();
    let scratch_len = match &plan {
        BenchPlan::Complex(plan) => plan.scratch_len(),
        // `process_strided` reserves the first N elements as its gather
        // buffer and forwards the remainder to the one-dimensional plan.
        BenchPlan::Strided(plan) => case.n + plan.scratch_len(),
        BenchPlan::TwoD(plan) => plan.scratch_len(),
        BenchPlan::ThreeD(plan) => plan.scratch_len(),
    };
    let mut scratch = vec![Complex64::new(0.0, 0.0); scratch_len];

    let mut process = |buffer: &mut [Complex64]| match &plan {
        BenchPlan::Complex(plan) => plan.process(buffer),
        BenchPlan::Strided(plan) => {
            let CaseKind::Strided { stride } = case.kind else {
                unreachable!()
            };
            process_strided(plan.as_ref(), buffer, case.batch, stride, &mut scratch);
        }
        BenchPlan::TwoD(plan) => plan.process_with_scratch(buffer, &mut scratch),
        BenchPlan::ThreeD(plan) => plan.process_with_scratch(buffer, &mut scratch),
    };

    // Prime the reusable plan scratch and any architecture-specific caches
    // before collecting steady-state samples.
    process(&mut buffer);
    buffer.copy_from_slice(&input);

    let mut samples = Vec::with_capacity(BATCHES);
    for _ in 0..BATCHES {
        buffer.copy_from_slice(&input);
        let start = Instant::now();
        process(&mut buffer);
        samples.push(start.elapsed().as_nanos());
        black_box(&buffer);
    }

    let total_ns: u128 = samples.iter().sum();
    let ns_per_batch = total_ns as f64 / BATCHES as f64;
    let bytes = (buffer_len * std::mem::size_of::<Complex64>()) as f64;
    let bytes_per_second = bytes * 1.0e9 / ns_per_batch;
    // A radix-2 complex FFT is conventionally estimated as 5 N log2(N)
    // floating-point operations. This is a shape estimate, not a hardware
    // counter, and remains useful for comparing cases in one artifact.
    let estimated_ops = match case.kind {
        CaseKind::Complex | CaseKind::Strided { .. } => {
            5.0 * case.n as f64 * (case.n as f64).log2() * case.batch as f64
        }
        CaseKind::TwoD { rows, cols } => {
            5.0 * rows as f64 * cols as f64 * ((rows as f64).log2() + (cols as f64).log2())
        }
        CaseKind::ThreeD { depth, rows, cols } => {
            5.0 * depth as f64
                * rows as f64
                * cols as f64
                * ((depth as f64).log2() + (rows as f64).log2() + (cols as f64).log2())
        }
    };
    // One GFLOP/s is one floating-point operation per nanosecond, so the
    // conversion from the nanosecond timing cancels the GFLOP scale factor.
    let estimated_gflops = estimated_ops / ns_per_batch;

    Timing {
        plan_ns,
        ns_per_batch,
        p95_batch_ns: percentile(&mut samples, 95),
        p99_batch_ns: percentile(&mut samples, 99),
        scratch_bytes: scratch_len * std::mem::size_of::<Complex64>(),
        bytes_per_second,
        estimated_gflops,
    }
}

fn measure_f32_batched_case(n: usize, batch: usize) -> Timing {
    let planner = FftPlanner::<f32>::with_dispatch(requested_dispatch());
    let plan_start = Instant::now();
    let plan = planner.plan(n, FftDirection::Forward);
    let plan_ns = plan_start.elapsed().as_nanos();

    let buffer_len = n * batch;
    let input = make_input_f32(buffer_len, n as u64 ^ batch as u64);
    let mut buffer = input.clone();
    let scratch_len = plan.scratch_len();
    let mut scratch = vec![Complex32::new(0.0, 0.0); scratch_len];

    let mut process = |buffer: &mut [Complex32]| {
        plan.process_with_scratch(buffer, &mut scratch);
    };
    process(&mut buffer);
    buffer.copy_from_slice(&input);

    let mut samples = Vec::with_capacity(BATCHES);
    for _ in 0..BATCHES {
        buffer.copy_from_slice(&input);
        let start = Instant::now();
        process(&mut buffer);
        samples.push(start.elapsed().as_nanos());
        black_box(&buffer);
    }

    let total_ns: u128 = samples.iter().sum();
    let ns_per_batch = total_ns as f64 / BATCHES as f64;
    let bytes = (buffer_len * std::mem::size_of::<Complex32>()) as f64;
    let bytes_per_second = bytes * 1.0e9 / ns_per_batch;
    let estimated_ops = 5.0 * n as f64 * (n as f64).log2() * batch as f64;
    let estimated_gflops = estimated_ops / ns_per_batch;

    Timing {
        plan_ns,
        ns_per_batch,
        p95_batch_ns: percentile(&mut samples, 95),
        p99_batch_ns: percentile(&mut samples, 99),
        scratch_bytes: scratch_len * std::mem::size_of::<Complex32>(),
        bytes_per_second,
        estimated_gflops,
    }
}

fn measure_convolution_case(lhs_len: usize, rhs_len: usize) -> Timing {
    let plan_start = Instant::now();
    let plan =
        LinearConvolutionPlan::<f64>::new_with_dispatch(lhs_len, rhs_len, requested_dispatch());
    let plan_ns = plan_start.elapsed().as_nanos();
    let lhs = make_input(lhs_len, lhs_len as u64 ^ 0x11);
    let rhs = make_input(rhs_len, rhs_len as u64 ^ 0x22);
    let mut output = vec![Complex64::new(0.0, 0.0); plan.output_len()];
    let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len()];

    let mut process = || {
        plan.process_with_scratch(&lhs, &rhs, &mut output, &mut scratch);
        black_box(&output);
    };
    process();

    let mut samples = Vec::with_capacity(BATCHES);
    for _ in 0..BATCHES {
        let start = Instant::now();
        process();
        samples.push(start.elapsed().as_nanos());
    }
    let total_ns: u128 = samples.iter().sum();
    let ns_per_batch = total_ns as f64 / BATCHES as f64;
    let bytes = ((lhs_len + rhs_len + plan.output_len()) * std::mem::size_of::<Complex64>()) as f64;
    let bytes_per_second = bytes * 1.0e9 / ns_per_batch;
    // Three complex FFTs plus pointwise multiplication, reported as a
    // conservative radix-2-equivalent operation estimate.
    let estimated_ops = 15.0 * plan.fft_len() as f64 * (plan.fft_len() as f64).log2();
    let estimated_gflops = estimated_ops / ns_per_batch;

    Timing {
        plan_ns,
        ns_per_batch,
        p95_batch_ns: percentile(&mut samples, 95),
        p99_batch_ns: percentile(&mut samples, 99),
        scratch_bytes: plan.scratch_len() * std::mem::size_of::<Complex64>(),
        bytes_per_second,
        estimated_gflops,
    }
}

fn main() {
    let cases = [
        Case {
            n: 64,
            batch: 1,
            label: "smooth_single",
            kind: CaseKind::Complex,
        },
        Case {
            n: 256,
            batch: 16,
            label: "smooth_batched",
            kind: CaseKind::Complex,
        },
        Case {
            n: 257,
            batch: 4,
            label: "prime_bluestein",
            kind: CaseKind::Complex,
        },
        Case {
            n: 1024,
            batch: 16,
            label: "audio_batched",
            kind: CaseKind::Complex,
        },
        Case {
            n: 4096,
            batch: 4,
            label: "large_batched",
            kind: CaseKind::Complex,
        },
        Case {
            n: 16384,
            batch: 1,
            label: "large_single",
            kind: CaseKind::Complex,
        },
        Case {
            n: 128,
            batch: 8,
            label: "strided_batched",
            kind: CaseKind::Strided { stride: 160 },
        },
        Case {
            n: 256,
            batch: 1,
            label: "multidim_16x16",
            kind: CaseKind::TwoD { rows: 16, cols: 16 },
        },
        Case {
            n: 512,
            batch: 1,
            label: "multidim_8x8x8",
            kind: CaseKind::ThreeD {
                depth: 8,
                rows: 8,
                cols: 8,
            },
        },
    ];
    let compiler = json_escape(option_env!("SONIUM_RUSTC_VERSION").unwrap_or("unknown"));
    let cpu = json_escape(&cpu_description());
    let reports = cases
        .iter()
        .map(|&case| {
            let timing = measure_case(case);
            format!(
                "{{\"label\":\"{}\",\"n\":{},\"batch\":{},\"plan_ns\":{},\"ns_per_batch\":{:.6},\"p95_batch_ns\":{},\"p99_batch_ns\":{},\"scratch_bytes\":{},\"bytes_per_second\":{:.6e},\"estimated_gflops\":{:.6e}}}",
                case.label,
                case.n,
                case.batch,
                timing.plan_ns,
                timing.ns_per_batch,
                timing.p95_batch_ns,
                timing.p99_batch_ns,
                timing.scratch_bytes,
                timing.bytes_per_second,
                timing.estimated_gflops,
            )
        })
        .collect::<Vec<_>>();
    let f32_cases = [(256usize, 16usize), (1024, 16), (4096, 4)];
    let reports_f32 = f32_cases
        .iter()
        .map(|&(n, batch)| {
            let timing = measure_f32_batched_case(n, batch);
            format!(
                "{{\"label\":\"f32_batched_{}x{}\",\"n\":{},\"batch\":{},\"plan_ns\":{},\"ns_per_batch\":{:.6},\"p95_batch_ns\":{},\"p99_batch_ns\":{},\"scratch_bytes\":{},\"bytes_per_second\":{:.6e},\"estimated_gflops\":{:.6e}}}",
                n,
                batch,
                n,
                batch,
                timing.plan_ns,
                timing.ns_per_batch,
                timing.p95_batch_ns,
                timing.p99_batch_ns,
                timing.scratch_bytes,
                timing.bytes_per_second,
                timing.estimated_gflops,
            )
        })
        .collect::<Vec<_>>();
    let convolution_cases = [(256usize, 129usize), (1024, 513)];
    let reports_convolution = convolution_cases
        .iter()
        .map(|&(lhs_len, rhs_len)| {
            let timing = measure_convolution_case(lhs_len, rhs_len);
            format!(
                "{{\"label\":\"linear_convolution_{}x{}\",\"lhs_len\":{},\"rhs_len\":{},\"fft_len\":{},\"plan_ns\":{},\"ns_per_batch\":{:.6},\"p95_batch_ns\":{},\"p99_batch_ns\":{},\"scratch_bytes\":{},\"bytes_per_second\":{:.6e},\"estimated_gflops\":{:.6e}}}",
                lhs_len,
                rhs_len,
                lhs_len,
                rhs_len,
                sonium_fft::factor::next_pow2_pub(lhs_len + rhs_len - 1),
                timing.plan_ns,
                timing.ns_per_batch,
                timing.p95_batch_ns,
                timing.p99_batch_ns,
                timing.scratch_bytes,
                timing.bytes_per_second,
                timing.estimated_gflops,
            )
        })
        .collect::<Vec<_>>();

    // Profile is inferred from debug assertions: release binaries are built
    // without them. Source revision comes from build-time best-effort git
    // capture; "unknown" means the provenance is missing, never a clean tree.
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let package_version = json_escape(env!("CARGO_PKG_VERSION"));
    let git_commit = json_escape(option_env!("SONIUM_GIT_COMMIT").unwrap_or("unknown"));
    let git_dirty = json_escape(option_env!("SONIUM_GIT_DIRTY").unwrap_or("unknown"));
    println!(
        "{{\"schema_version\":1,\"artifact\":\"sonium-fft-cpu-benchmark\",\"architecture\":\"{}\",\"target_features\":\"{}\",\"dispatch\":\"{}\",\"compiler\":\"{}\",\"cpu\":\"{}\",\"package_version\":\"{}\",\"profile\":\"{}\",\"git_commit\":\"{}\",\"git_dirty\":\"{}\",\"threads\":1,\"threading\":\"single-threaded caller\",\"batches_per_case\":{},\"transform\":\"complex_forward_unnormalized\",\"correctness_gate\":\"independent_dft_and_fftw_reference_tests\",\"allocation_blocks_after_warmup\":null,\"energy_joules\":null,\"allocation_contract\":\"covered_by_no_alloc_execution_test\",\"reference_benchmark\":\"criterion_fft_vs_rustfft\",\"cases\":[{}],\"f32_cases\":[{}],\"convolution_cases\":[{}]}}",
        std::env::consts::ARCH,
        target_features(),
        dispatch(),
        compiler,
        cpu,
        package_version,
        profile,
        git_commit,
        git_dirty,
        BATCHES,
        reports.join(","),
        reports_f32.join(","),
        reports_convolution.join(",")
    );
}
