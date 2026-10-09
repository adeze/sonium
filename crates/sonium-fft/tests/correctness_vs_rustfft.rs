//! Cross-check every supported length and direction against `rustfft`.
//!
//! Tolerance scales with `n` because the error of `O(n log n)` floating
//! ops grows roughly with `√(n log n)` in expectation.

use approx::assert_relative_eq;
use num_complex::{Complex32, Complex64};
use sonium_fft::FftDirection;
use sonium_fft::rustfft_compat::FftPlanner as OursFftPlanner;

fn random_c64(seed: u64, n: usize) -> Vec<Complex64> {
    let mut s = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let re = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let im = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
            Complex64::new(re, im)
        })
        .collect()
}

fn random_c32(seed: u64, n: usize) -> Vec<Complex32> {
    random_c64(seed, n)
        .into_iter()
        .map(|c| Complex32::new(c.re as f32, c.im as f32))
        .collect()
}

fn check_f64(n: usize, dir: FftDirection) {
    let mut ours = OursFftPlanner::<f64>::new();
    let mut theirs = rustfft::FftPlanner::<f64>::new();
    let (op_o, op_t) = match dir {
        FftDirection::Forward => (ours.plan_fft_forward(n), theirs.plan_fft_forward(n)),
        FftDirection::Inverse => (ours.plan_fft_inverse(n), theirs.plan_fft_inverse(n)),
    };
    let inp = random_c64(0xDEAD_BEEF + n as u64, n);
    let mut a = inp.clone();
    let mut b = inp.clone();
    op_o.process(&mut a);
    op_t.process(&mut b);
    let tol = (n as f64).log2() * 1e-13;
    for (ai, bi) in a.iter().zip(b.iter()) {
        assert_relative_eq!(ai.re, bi.re, max_relative = tol, epsilon = tol);
        assert_relative_eq!(ai.im, bi.im, max_relative = tol, epsilon = tol);
    }
}

fn check_f32(n: usize, dir: FftDirection) {
    let mut ours = OursFftPlanner::<f32>::new();
    let mut theirs = rustfft::FftPlanner::<f32>::new();
    let (op_o, op_t) = match dir {
        FftDirection::Forward => (ours.plan_fft_forward(n), theirs.plan_fft_forward(n)),
        FftDirection::Inverse => (ours.plan_fft_inverse(n), theirs.plan_fft_inverse(n)),
    };
    let inp = random_c32(0xBEEF + n as u64, n);
    let mut a = inp.clone();
    let mut b = inp.clone();
    op_o.process(&mut a);
    op_t.process(&mut b);
    let tol = ((n as f32).log2() * 5e-5_f32).max(1e-5);
    for (ai, bi) in a.iter().zip(b.iter()) {
        assert_relative_eq!(ai.re, bi.re, max_relative = tol, epsilon = tol);
        assert_relative_eq!(ai.im, bi.im, max_relative = tol, epsilon = tol);
    }
}

#[test]
fn all_small_lengths_f64() {
    for n in 1..=64usize {
        check_f64(n, FftDirection::Forward);
        check_f64(n, FftDirection::Inverse);
    }
}

#[test]
fn smooth_medium_lengths_f64() {
    for n in [
        100usize, 120, 128, 144, 240, 256, 360, 480, 512, 720, 1024, 1200, 1536, 2048, 2880,
    ] {
        check_f64(n, FftDirection::Forward);
        check_f64(n, FftDirection::Inverse);
    }
}

#[test]
fn primes_via_bluestein_f64() {
    for n in [
        7usize, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 97, 211, 401,
    ] {
        check_f64(n, FftDirection::Forward);
        check_f64(n, FftDirection::Inverse);
    }
}

#[test]
fn all_small_lengths_f32() {
    for n in 1..=32usize {
        check_f32(n, FftDirection::Forward);
        check_f32(n, FftDirection::Inverse);
    }
}

#[test]
fn smooth_medium_lengths_f32() {
    for n in [
        100usize, 120, 128, 144, 240, 256, 360, 480, 512, 720, 1024, 1200, 1536, 2048,
    ] {
        check_f32(n, FftDirection::Forward);
        check_f32(n, FftDirection::Inverse);
    }
}

#[test]
fn primes_via_bluestein_f32() {
    for n in [7usize, 11, 13, 17, 19, 23, 29, 31, 37, 41, 53, 97, 211] {
        check_f32(n, FftDirection::Forward);
        check_f32(n, FftDirection::Inverse);
    }
}

#[test]
fn bluestein_batch_processes_each_chunk() {
    // Prime length through Bluestein with batch > 1: every chunk must match
    // an independent single-FFT reference.
    let mut ours = OursFftPlanner::<f64>::new();
    let mut theirs = rustfft::FftPlanner::<f64>::new();
    let n = 97;
    let batch = 4;
    let op_o = ours.plan_fft_forward(n);
    let op_t = theirs.plan_fft_forward(n);
    let inp = random_c64(0xB1E57, n * batch);
    let mut a = inp.clone();
    let mut b = inp.clone();
    op_o.process(&mut a);
    for chunk in b.chunks_exact_mut(n) {
        op_t.process(chunk);
    }
    let tol = (n as f64).log2() * 1e-13;
    for (ai, bi) in a.iter().zip(b.iter()) {
        assert_relative_eq!(ai.re, bi.re, max_relative = tol, epsilon = tol);
        assert_relative_eq!(ai.im, bi.im, max_relative = tol, epsilon = tol);
    }
}

#[test]
fn batched_call_processes_each_chunk() {
    let mut ours = OursFftPlanner::<f64>::new();
    let mut theirs = rustfft::FftPlanner::<f64>::new();
    let n = 256;
    let batch = 5;
    let op_o = ours.plan_fft_forward(n);
    let op_t = theirs.plan_fft_forward(n);
    let inp = random_c64(0xC0FFEE, n * batch);
    let mut a = inp.clone();
    let mut b = inp.clone();
    op_o.process(&mut a);
    for chunk in b.chunks_exact_mut(n) {
        op_t.process(chunk);
    }
    let tol = (n as f64).log2() * 1e-13;
    for (ai, bi) in a.iter().zip(b.iter()) {
        assert_relative_eq!(ai.re, bi.re, max_relative = tol, epsilon = tol);
        assert_relative_eq!(ai.im, bi.im, max_relative = tol, epsilon = tol);
    }
}
