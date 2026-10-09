//! Probe tests for edge cases and API contract guarantees.

use num_complex::Complex64;
use sonium_fft::FftDirection;
use sonium_fft::rustfft_compat::FftPlanner;

#[test]
fn length_one_is_identity() {
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(1);
    assert_eq!(plan.len(), 1);
    let mut buf = vec![Complex64::new(1.5, -2.5)];
    plan.process(&mut buf);
    assert_eq!(buf[0], Complex64::new(1.5, -2.5));
}

#[test]
fn batched_length_one_passes_through() {
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(1);
    let mut buf = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 1.0),
        Complex64::new(-1.0, 4.0),
    ];
    let expected = buf.clone();
    plan.process(&mut buf);
    assert_eq!(buf, expected);
}

/// The native [`sonium_fft::FftPlan::process`] accepts an empty buffer as
/// a no-op (zero batches). This is a deliberate divergence from
/// `rustfft::Fft::process`, which panics. See
/// `tests/rustfft_compat_contract.rs::empty_buffer_panics_in_rustfft_compat`
/// for the matching rustfft-shape behaviour.
#[test]
fn native_plan_empty_buffer_is_noop() {
    use sonium_fft::FftDirection;
    let planner = sonium_fft::FftPlanner::<f64>::new();
    let plan = planner.plan(4, FftDirection::Forward);
    let mut empty: Vec<Complex64> = Vec::new();
    plan.process(&mut empty);
    assert_eq!(empty.len(), 0);
}

#[test]
#[should_panic(expected = "multiple of plan length")]
fn mismatched_buffer_length_panics_native() {
    use sonium_fft::FftDirection;
    let planner = sonium_fft::FftPlanner::<f64>::new();
    let plan = planner.plan(4, FftDirection::Forward);
    let mut buf = vec![Complex64::new(0.0, 0.0); 5];
    plan.process(&mut buf);
}

#[test]
#[should_panic(expected = "not a multiple")]
fn rustfft_compat_mismatched_buffer_length_panics() {
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(4);
    let mut buf = vec![Complex64::new(0.0, 0.0); 5];
    plan.process(&mut buf);
}

#[test]
fn planner_caches_plans() {
    let mut planner = FftPlanner::<f64>::new();
    let p1 = planner.plan_fft_forward(64);
    let p2 = planner.plan_fft_forward(64);
    // The adapter wraps the inner plan in a fresh Arc per call, but both
    // adapters should report the same length / direction.
    assert_eq!(p1.len(), p2.len());
    assert_eq!(p1.fft_direction(), p2.fft_direction());
}

#[test]
fn forward_then_inverse_round_trip_smooth() {
    let mut planner = FftPlanner::<f64>::new();
    let n = 360;
    let fwd = planner.plan_fft_forward(n);
    let inv = planner.plan_fft_inverse(n);
    let input: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64).sin(), (i as f64 * 0.3).cos()))
        .collect();
    let mut buf = input.clone();
    fwd.process(&mut buf);
    inv.process(&mut buf);
    let inv_n = 1.0 / n as f64;
    for (b, x) in buf.iter().zip(input.iter()) {
        approx::assert_relative_eq!(b.re * inv_n, x.re, max_relative = 1e-12, epsilon = 1e-12);
        approx::assert_relative_eq!(b.im * inv_n, x.im, max_relative = 1e-12, epsilon = 1e-12);
    }
}

#[test]
fn forward_then_inverse_round_trip_bluestein() {
    let mut planner = FftPlanner::<f64>::new();
    let n = 97; // prime → Bluestein
    let fwd = planner.plan_fft_forward(n);
    let inv = planner.plan_fft_inverse(n);
    let input: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64).sin(), (i as f64 * 0.7).cos()))
        .collect();
    let mut buf = input.clone();
    fwd.process(&mut buf);
    inv.process(&mut buf);
    let inv_n = 1.0 / n as f64;
    for (b, x) in buf.iter().zip(input.iter()) {
        approx::assert_relative_eq!(b.re * inv_n, x.re, max_relative = 1e-10, epsilon = 1e-10);
        approx::assert_relative_eq!(b.im * inv_n, x.im, max_relative = 1e-10, epsilon = 1e-10);
    }
}

#[test]
fn process_with_scratch_matches_process_smooth() {
    let mut planner = FftPlanner::<f64>::new();
    let n = 256;
    let plan = planner.plan_fft_forward(n);
    let input: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(i as f64 * 0.1, i as f64 * -0.2))
        .collect();

    let mut buf_a = input.clone();
    plan.process(&mut buf_a);

    let mut buf_b = input.clone();
    let mut scratch = vec![Complex64::new(0.0, 0.0); plan.get_inplace_scratch_len()];
    plan.process_with_scratch(&mut buf_b, &mut scratch);

    assert_eq!(buf_a, buf_b);
}

#[test]
fn process_with_scratch_matches_process_bluestein() {
    let mut planner = FftPlanner::<f64>::new();
    let n = 41; // prime
    let plan = planner.plan_fft_forward(n);
    let input: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(i as f64 * 0.1, i as f64 * -0.2))
        .collect();

    let mut buf_a = input.clone();
    plan.process(&mut buf_a);

    let mut buf_b = input.clone();
    let mut scratch = vec![Complex64::new(0.0, 0.0); plan.get_inplace_scratch_len()];
    plan.process_with_scratch(&mut buf_b, &mut scratch);

    assert_eq!(buf_a, buf_b);
}

#[test]
fn process_outofplace_matches_inplace() {
    let mut planner = FftPlanner::<f64>::new();
    let n = 1024;
    let plan = planner.plan_fft_forward(n);
    let input: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(i as f64 * 0.013, (i as f64).cos()))
        .collect();

    let mut buf_in_inplace = input.clone();
    plan.process(&mut buf_in_inplace);

    let mut buf_in = input.clone();
    let mut buf_out = vec![Complex64::new(0.0, 0.0); n];
    let mut scratch = vec![Complex64::new(0.0, 0.0); plan.get_outofplace_scratch_len()];
    plan.process_outofplace_with_scratch(&mut buf_in, &mut buf_out, &mut scratch);

    assert_eq!(buf_in_inplace, buf_out);
}

#[test]
fn batched_processes_each_chunk_independently_bluestein() {
    let mut planner = FftPlanner::<f64>::new();
    let n = 23;
    let batch = 4;
    let plan = planner.plan_fft_forward(n);

    // Build a deliberately different per-chunk input to catch any
    // cross-chunk state leakage.
    let mut input = Vec::with_capacity(n * batch);
    for b in 0..batch {
        for j in 0..n {
            input.push(Complex64::new(
                (b * 100 + j) as f64 * 0.05,
                (b as f64 + j as f64).sin(),
            ));
        }
    }

    // Reference: per-chunk
    let mut reference = input.clone();
    for chunk in reference.chunks_exact_mut(n) {
        plan.process(chunk);
    }

    // Batched call
    let mut buf = input.clone();
    plan.process(&mut buf);

    for (a, b) in reference.iter().zip(buf.iter()) {
        approx::assert_relative_eq!(a.re, b.re, max_relative = 1e-12, epsilon = 1e-12);
        approx::assert_relative_eq!(a.im, b.im, max_relative = 1e-12, epsilon = 1e-12);
    }
}

#[test]
fn fft_direction_reported_correctly() {
    let mut planner = FftPlanner::<f64>::new();
    let fwd = planner.plan_fft_forward(16);
    let inv = planner.plan_fft_inverse(16);
    assert_eq!(fwd.fft_direction(), FftDirection::Forward);
    assert_eq!(inv.fft_direction(), FftDirection::Inverse);
}

#[test]
fn scratch_len_consistent_with_in_use_size() {
    let mut planner = FftPlanner::<f64>::new();

    // Smooth case
    let p_smooth = planner.plan_fft_forward(256);
    assert!(p_smooth.get_inplace_scratch_len() >= 256);

    // Bluestein case (prime)
    let p_blue = planner.plan_fft_forward(41);
    // Bluestein scratch is 2*m where m = next_pow2(2n-1). For n=41, m=128.
    assert!(p_blue.get_inplace_scratch_len() >= 256);
}
