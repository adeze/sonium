//! Independent correctness properties for the CPU FFT.
//!
//! These tests intentionally do not use rustfft or Sonium's twiddle tables as
//! a reference. The O(N²) DFT below computes each phase directly and uses
//! Neumaier compensation for the real and imaginary accumulators. It is not
//! a performance test; it is a stable, independently structured oracle for
//! Stockham, Bluestein, codelet, and twiddle-boundary regressions.

use num_complex::Complex64;
use sonium_fft::FftDirection;
use sonium_fft::rustfft_compat::FftPlanner;
use sonium_fft::twiddles::{TwiddleTable, build_stage_twiddles};
use std::f64::consts::TAU;

fn compensated_add(sum: &mut f64, correction: &mut f64, value: f64) {
    let next = *sum + value;
    if sum.abs() >= value.abs() {
        *correction += (*sum - next) + value;
    } else {
        *correction += (value - next) + *sum;
    }
    *sum = next;
}

#[derive(Clone, Copy, Debug, Default)]
struct DoubleDouble {
    hi: f64,
    lo: f64,
}

fn two_sum(a: f64, b: f64) -> DoubleDouble {
    let hi = a + b;
    let b_virtual = hi - a;
    let lo = (a - (hi - b_virtual)) + (b - b_virtual);
    DoubleDouble { hi, lo }
}

fn two_product(a: f64, b: f64) -> DoubleDouble {
    let hi = a * b;
    DoubleDouble {
        hi,
        lo: a.mul_add(b, -hi),
    }
}

fn dd_add(a: DoubleDouble, b: DoubleDouble) -> DoubleDouble {
    let head = two_sum(a.hi, b.hi);
    let tail = two_sum(a.lo, b.lo);
    two_sum(head.hi, head.lo + tail.hi + tail.lo)
}

fn dd_sub(a: DoubleDouble, b: DoubleDouble) -> DoubleDouble {
    dd_add(
        a,
        DoubleDouble {
            hi: -b.hi,
            lo: -b.lo,
        },
    )
}

/// Independent double-double accumulation for a small high-precision DFT
/// oracle. It does not reuse Sonium twiddles or FFT code; the phase is formed
/// directly for every `(j,k)` pair and products retain their FMA residuals.
fn direct_dft_double_double(input: &[Complex64], direction: FftDirection) -> Vec<Complex64> {
    let n = input.len();
    let sign = match direction {
        FftDirection::Forward => -1.0,
        FftDirection::Inverse => 1.0,
    };
    (0..n)
        .map(|k| {
            let mut sum_re = DoubleDouble::default();
            let mut sum_im = DoubleDouble::default();
            for (j, sample) in input.iter().enumerate() {
                let theta = sign * TAU * (j as f64) * (k as f64) / (n as f64);
                let (sin_theta, cos_theta) = theta.sin_cos();
                sum_re = dd_add(
                    sum_re,
                    dd_sub(
                        two_product(sample.re, cos_theta),
                        two_product(sample.im, sin_theta),
                    ),
                );
                sum_im = dd_add(
                    sum_im,
                    dd_add(
                        two_product(sample.re, sin_theta),
                        two_product(sample.im, cos_theta),
                    ),
                );
            }
            Complex64::new(sum_re.hi + sum_re.lo, sum_im.hi + sum_im.lo)
        })
        .collect()
}

fn direct_dft(input: &[Complex64], direction: FftDirection) -> Vec<Complex64> {
    let n = input.len();
    let sign = match direction {
        FftDirection::Forward => -1.0,
        FftDirection::Inverse => 1.0,
    };
    (0..n)
        .map(|k| {
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;
            let mut correction_re = 0.0;
            let mut correction_im = 0.0;
            for (j, sample) in input.iter().enumerate() {
                let theta = sign * TAU * (j as f64) * (k as f64) / (n as f64);
                let (sin_theta, cos_theta) = theta.sin_cos();
                compensated_add(
                    &mut sum_re,
                    &mut correction_re,
                    sample.re * cos_theta - sample.im * sin_theta,
                );
                compensated_add(
                    &mut sum_im,
                    &mut correction_im,
                    sample.re * sin_theta + sample.im * cos_theta,
                );
            }
            Complex64::new(sum_re + correction_re, sum_im + correction_im)
        })
        .collect()
}

fn input(seed: u64, n: usize) -> Vec<Complex64> {
    let mut state = seed;
    (0..n)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let re = ((state >> 11) as f64) / ((1_u64 << 53) as f64) - 0.5;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let im = ((state >> 11) as f64) / ((1_u64 << 53) as f64) - 0.5;
            // Include coherent phase ramps as well as pseudo-random values.
            Complex64::new(re + 0.1 * (i as f64 * 0.17).sin(), im)
        })
        .collect()
}

fn assert_close(got: &[Complex64], want: &[Complex64], n: usize) {
    // The direct oracle is compensated, but the transform itself performs
    // O(log N) rounded butterfly operations (and fused SIMD paths may use
    // FMA). Keep a strict, explicit budget while allowing that accumulation.
    let tolerance = ((n as f64).log2() * 1.5e-12).max(1.0e-13);
    for (index, (actual, expected)) in got.iter().zip(want).enumerate() {
        let scale = expected.re.abs().max(expected.im.abs()).max(1.0);
        assert!(
            (actual.re - expected.re).abs() <= tolerance * scale
                && (actual.im - expected.im).abs() <= tolerance * scale,
            "DFT mismatch at n={n}, k={index}: actual={actual:?}, expected={expected:?}, tol={tolerance:e}"
        );
    }
}

fn check_against_direct_dft(n: usize, direction: FftDirection) {
    let samples = input(0x51A7_0000 + n as u64, n);
    let expected = direct_dft(&samples, direction);
    let mut actual = samples.clone();
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft(n, direction);
    plan.process(&mut actual);
    assert_close(&actual, &expected, n);
}

#[test]
fn smooth_and_prime_lengths_match_compensated_direct_dft() {
    for &n in &[
        1usize, 2, 3, 4, 5, 6, 8, 9, 12, 15, 16, 20, 25, 32, 60, 64, 128, 256,
    ] {
        check_against_direct_dft(n, FftDirection::Forward);
        check_against_direct_dft(n, FftDirection::Inverse);
    }
    for &n in &[7usize, 11, 13, 17, 23, 31, 47, 97] {
        check_against_direct_dft(n, FftDirection::Forward);
        check_against_direct_dft(n, FftDirection::Inverse);
    }
}

#[test]
fn small_lengths_match_double_double_direct_dft() {
    for &n in &[8usize, 13, 32, 47] {
        let samples = input(0xDD00_0000 + n as u64, n);
        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            let expected = direct_dft_double_double(&samples, direction);
            let mut actual = samples.clone();
            let mut planner = FftPlanner::<f64>::new();
            planner.plan_fft(n, direction).process(&mut actual);
            assert_close(&actual, &expected, n);
        }
    }
}

fn inner_product(lhs: &[Complex64], rhs: &[Complex64]) -> Complex64 {
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| a.conj() * b)
        .fold(Complex64::new(0.0, 0.0), |sum, value| sum + value)
}

#[test]
fn forward_parseval_and_adjoint_identities_hold() {
    for &n in &[15usize, 60, 97] {
        let x = input(0xA11D + n as u64, n);
        let y = input(0xBEEF + n as u64, n);
        let mut planner = FftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(n);
        let inverse = planner.plan_fft_inverse(n);

        let mut fx = x.clone();
        let mut inverse_y = y.clone();
        forward.process(&mut fx);
        inverse.process(&mut inverse_y);

        let lhs = inner_product(&fx, &y);
        let rhs = inner_product(&x, &inverse_y);
        assert!(
            (lhs - rhs).norm() < n as f64 * 2.0e-10,
            "adjoint mismatch n={n}"
        );

        let input_energy: f64 = x.iter().map(|z| z.norm_sqr()).sum();
        let output_energy: f64 = fx.iter().map(|z| z.norm_sqr()).sum::<f64>() / n as f64;
        assert!(
            (input_energy - output_energy).abs() < n as f64 * 2.0e-10,
            "Parseval mismatch n={n}"
        );
    }
}

#[test]
fn optimized_batch_shapes_match_independent_reference() {
    // On AArch64 these sizes exercise the pair-transpose and fused final
    // stages; on other CPUs they still verify the batched API contract.
    for &(n, batch) in &[(256usize, 2usize), (256, 3), (1024, 2), (1024, 3)] {
        let samples = input(0xC0DE + n as u64 + batch as u64, n * batch);
        let mut actual = samples.clone();
        let mut planner = FftPlanner::<f64>::new();
        let plan = planner.plan_fft_forward(n);
        plan.process(&mut actual);
        for (batch_index, chunk) in actual.chunks_exact(n).enumerate() {
            let expected = direct_dft(
                &samples[batch_index * n..(batch_index + 1) * n],
                FftDirection::Forward,
            );
            assert_close(chunk, &expected, n);
        }
    }
}

#[test]
fn twiddle_slice_boundaries_are_complete() {
    let table = TwiddleTable::<f64>::new(60, FftDirection::Forward);
    assert_eq!(table.at(0), Complex64::new(1.0, 0.0));
    assert!(table.at(59).re.is_finite() && table.at(59).im.is_finite());

    let stages = build_stage_twiddles::<f64>(60, &[5, 4, 3], FftDirection::Forward);
    assert_eq!(stages.len(), 3);
    for stage in stages {
        for k in 1..stage.radix {
            let twiddles = stage.k_slice(k);
            assert_eq!(twiddles.len(), stage.m);
            assert!(twiddles[0].re.is_finite());
            assert!(twiddles[stage.m - 1].im.is_finite());
        }
    }
}

#[test]
fn zero_padded_fft_convolution_matches_direct_linear_convolution() {
    let x: Vec<Complex64> = (0..11)
        .map(|i| Complex64::new((i as f64 * 0.19).sin(), (i as f64 * 0.07).cos() * 0.1))
        .collect();
    let h: Vec<Complex64> = (0..7)
        .map(|i| Complex64::new((i as f64 * 0.31).cos() * 0.2, (i as f64 * 0.11).sin()))
        .collect();
    let output_len = x.len() + h.len() - 1;
    let fft_len = output_len.next_power_of_two();

    let mut x_padded = vec![Complex64::new(0.0, 0.0); fft_len];
    let mut h_padded = vec![Complex64::new(0.0, 0.0); fft_len];
    x_padded[..x.len()].copy_from_slice(&x);
    h_padded[..h.len()].copy_from_slice(&h);

    let mut planner = FftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(fft_len);
    let inverse = planner.plan_fft_inverse(fft_len);
    forward.process(&mut x_padded);
    forward.process(&mut h_padded);
    for (x_bin, h_bin) in x_padded.iter_mut().zip(h_padded.iter()) {
        *x_bin *= *h_bin;
    }
    inverse.process(&mut x_padded);

    for (n, actual) in x_padded.iter().enumerate().take(output_len) {
        let mut expected = Complex64::new(0.0, 0.0);
        for (i, x_value) in x.iter().enumerate() {
            if let Some(h_value) = h.get(n.checked_sub(i).unwrap_or(h.len())) {
                expected += *x_value * *h_value;
            }
        }
        assert!((*actual / fft_len as f64 - expected).norm() < 2e-9, "n={n}");
    }
}
