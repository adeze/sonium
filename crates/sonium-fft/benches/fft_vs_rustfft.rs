//! Baseline benchmark: sonium-fft (CPU) vs rustfft, single-FFT and batched.
//!
//! Run: `cargo bench -p sonium-fft --bench fft_vs_rustfft`.

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
#[cfg(not(target_arch = "wasm32"))]
use num_complex::{Complex32, Complex64};
#[cfg(not(target_arch = "wasm32"))]
use sonium_fft::rustfft_compat::FftPlanner as OurFftPlanner;

#[cfg(not(target_arch = "wasm32"))]
fn sizes() -> &'static [usize] {
    &[64, 256, 512, 1024, 2048, 4096, 16384]
}

#[cfg(not(target_arch = "wasm32"))]
fn make_input(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| Complex64::new((i as f64 * 0.0123).sin(), (i as f64 * 0.0177).cos()))
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn make_input_f32(n: usize) -> Vec<Complex32> {
    (0..n)
        .map(|i| Complex32::new((i as f32 * 0.0123).sin(), (i as f32 * 0.0177).cos()))
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_single_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("single_f64");
    for &n in sizes() {
        let input = make_input(n);

        let mut ours = OurFftPlanner::<f64>::new();
        let p_ours = ours.plan_fft_forward(n);
        g.bench_with_input(BenchmarkId::new("sonium-fft", n), &n, |b, _| {
            let mut buf = input.clone();
            b.iter(|| {
                buf.copy_from_slice(&input);
                p_ours.process(&mut buf);
                std::hint::black_box(&buf);
            });
        });

        let mut theirs = rustfft::FftPlanner::<f64>::new();
        let p_theirs = theirs.plan_fft_forward(n);
        g.bench_with_input(BenchmarkId::new("rustfft", n), &n, |b, _| {
            let mut buf = input.clone();
            b.iter(|| {
                buf.copy_from_slice(&input);
                p_theirs.process(&mut buf);
                std::hint::black_box(&buf);
            });
        });
    }
    g.finish();
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_batched_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("batched_f64_x16");
    let batch = 16usize;
    for &n in &[256usize, 512, 1024, 2048, 4096] {
        let input = make_input(n * batch);

        let mut ours = OurFftPlanner::<f64>::new();
        let p_ours = ours.plan_fft_forward(n);
        g.bench_with_input(BenchmarkId::new("sonium-fft", n), &n, |b, _| {
            let mut buf = input.clone();
            b.iter(|| {
                buf.copy_from_slice(&input);
                p_ours.process(&mut buf);
                std::hint::black_box(&buf);
            });
        });

        let mut theirs = rustfft::FftPlanner::<f64>::new();
        let p_theirs = theirs.plan_fft_forward(n);
        g.bench_with_input(BenchmarkId::new("rustfft", n), &n, |b, _| {
            let mut buf = input.clone();
            b.iter(|| {
                buf.copy_from_slice(&input);
                for chunk in buf.chunks_exact_mut(n) {
                    p_theirs.process(chunk);
                }
                std::hint::black_box(&buf);
            });
        });
    }
    g.finish();
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_batched_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("batched_f32");
    for &(n, batch) in &[(256usize, 16usize), (1024, 16), (4096, 4)] {
        let input = make_input_f32(n * batch);

        let mut ours = OurFftPlanner::<f32>::new();
        let p_ours = ours.plan_fft_forward(n);
        g.bench_with_input(BenchmarkId::new("sonium-fft", n), &n, |b, _| {
            let mut buf = input.clone();
            b.iter(|| {
                buf.copy_from_slice(&input);
                p_ours.process(&mut buf);
                std::hint::black_box(&buf);
            });
        });

        let mut theirs = rustfft::FftPlanner::<f32>::new();
        let p_theirs = theirs.plan_fft_forward(n);
        g.bench_with_input(BenchmarkId::new("rustfft", n), &n, |b, _| {
            let mut buf = input.clone();
            b.iter(|| {
                buf.copy_from_slice(&input);
                for chunk in buf.chunks_exact_mut(n) {
                    p_theirs.process(chunk);
                }
                std::hint::black_box(&buf);
            });
        });
    }
    g.finish();
}

/// Measure planner construction cost. Includes twiddle-table generation
/// (sin/cos batch via sonium-fastmath) and any Bluestein chirp/B-kernel
/// setup if `n` isn't decomposable into the radix-2-3-5-7 mixed-radix
/// strategy. Plan execution is NOT measured here.
#[cfg(not(target_arch = "wasm32"))]
fn bench_plan_build_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("plan_build_f64");
    for &n in sizes() {
        g.bench_with_input(BenchmarkId::new("sonium-fft", n), &n, |b, &n| {
            b.iter(|| {
                let mut planner = OurFftPlanner::<f64>::new();
                let p = planner.plan_fft_forward(n);
                std::hint::black_box(p);
            });
        });
        g.bench_with_input(BenchmarkId::new("rustfft", n), &n, |b, &n| {
            b.iter(|| {
                let mut planner = rustfft::FftPlanner::<f64>::new();
                let p = planner.plan_fft_forward(n);
                std::hint::black_box(p);
            });
        });
    }
    g.finish();
}

#[cfg(not(target_arch = "wasm32"))]
criterion_group!(
    benches,
    bench_single_f64,
    bench_batched_f64,
    bench_batched_f32,
    bench_plan_build_f64
);
#[cfg(not(target_arch = "wasm32"))]
criterion_main!(benches);
