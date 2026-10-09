//! Differential coverage for threshold-gated batch parallelism.
//!
//! Above the batch/size thresholds in `cpu::plan`, the owned-scratch
//! `process` path splits batches across `std::thread::scope` workers with
//! per-thread scratch instead of serialising on the shared plan scratch.
//! These tests pin the parallel path against the serial caller-scratch path
//! (same plans, same inputs) plus independent single-chunk references, and
//! pin concurrent shared-plan use against serial execution.

use num_complex::Complex64;
use sonium_fft::{FftDirection, FftPlan, FftPlanner, SixStepFftPlan};
use std::f64::consts::TAU;
use std::sync::Arc;

fn pseudo_random(seed: u64, n: usize) -> Vec<Complex64> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let re = ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let im = ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
            Complex64::new(re, im)
        })
        .collect()
}

fn naive_dft(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    (0..n)
        .map(|k| {
            input
                .iter()
                .enumerate()
                .map(|(j, sample)| {
                    let theta = -TAU * (j as f64) * (k as f64) / (n as f64);
                    sample * Complex64::new(theta.cos(), theta.sin())
                })
                .sum()
        })
        .collect()
}

/// Large smooth batch: `process` (parallel above threshold) must equal the
/// serial caller-scratch path, and the first chunk must match a naive DFT.
#[test]
fn stockham_parallel_batch_matches_serial_and_naive_reference() {
    let planner = FftPlanner::<f64>::new();
    let plan = planner.plan(1024, FftDirection::Forward);
    let batch = 64;
    let input = pseudo_random(0xBE57, 1024 * batch);

    let mut via_owned = input.clone();
    plan.process(&mut via_owned);

    let mut via_scratch = input.clone();
    let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len()];
    plan.process_with_scratch(&mut via_scratch, &mut scratch);

    assert_eq!(via_owned.len(), via_scratch.len());
    // Re-splitting the batch regroups SIMD pairs across different chunk
    // neighbours, so summation order (and the last ulp) may differ between
    // the two paths. Agreement must be tight, not bitwise.
    let sifatida = 1e-12;
    for (index, (a, b)) in via_owned.iter().zip(via_scratch.iter()).enumerate() {
        assert!(
            (a - b).norm() <= sifatida * b.norm().max(1.0),
            "parallel/serial divergence at element {index}: {a} vs {b}"
        );
    }

    let expected = naive_dft(&input[..1024]);
    for (got, want) in via_owned[..1024].iter().zip(expected.iter()) {
        assert!(
            (got - want).norm() <= 1e-11 * want.norm().max(1.0),
            "parallel chunk mismatch vs naive DFT: {got} vs {want}"
        );
    }
}

/// Large prime batch through Bluestein: same differential contract.
#[test]
fn bluestein_parallel_batch_matches_serial_and_naive_reference() {
    let planner = FftPlanner::<f64>::new();
    let plan = planner.plan(127, FftDirection::Forward);
    let batch = 520;
    let input = pseudo_random(0xB1E57, 127 * batch);

    let mut via_owned = input.clone();
    plan.process(&mut via_owned);

    let mut via_scratch = input.clone();
    let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len()];
    plan.process_with_scratch(&mut via_scratch, &mut scratch);

    let scale = via_scratch
        .iter()
        .map(|value| value.norm())
        .fold(1.0_f64, f64::max);
    for (index, (a, b)) in via_owned.iter().zip(via_scratch.iter()).enumerate() {
        assert!(
            (a - b).norm() <= 1e-12 * scale,
            "parallel/serial divergence at element {index}"
        );
    }

    let expected = naive_dft(&input[..127]);
    let tol = (127.0_f64 * 1e-12).max(1e-12);
    for (got, want) in via_owned[..127].iter().zip(expected.iter()) {
        assert!(
            (got - want).norm() <= tol * scale,
            "parallel chunk mismatch vs naive DFT: {got} vs {want}"
        );
    }
}

/// Large six-step batch: same differential contract.
#[test]
fn six_step_parallel_batch_matches_serial_path() {
    let plan = SixStepFftPlan::<f64>::new(1024, FftDirection::Forward);
    assert_eq!(plan.factors(), (32, 32));
    let batch = 64;
    let input = pseudo_random(0x5157, 1024 * batch);

    let mut via_owned = input.clone();
    plan.process(&mut via_owned);

    let mut via_scratch = input.clone();
    let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len()];
    plan.process_with_scratch(&mut via_scratch, &mut scratch);

    let scale = via_scratch
        .iter()
        .map(|value| value.norm())
        .fold(1.0_f64, f64::max);
    for (index, (a, b)) in via_owned.iter().zip(via_scratch.iter()).enumerate() {
        assert!(
            (a - b).norm() <= 1e-12 * scale,
            "parallel/serial divergence at element {index}"
        );
    }
}

/// Threads sharing one plan with caller-owned scratch must observe the same
/// transforms as serial execution (no shared-scratch races).
#[test]
fn concurrent_shared_plan_matches_serial_execution() {
    let planner = Arc::new(FftPlanner::<f64>::new());
    let smooth = planner.plan(256, FftDirection::Forward);
    let prime = planner.plan(97, FftDirection::Forward);

    let inputs: Vec<Vec<Complex64>> = (0..8)
        .map(|thread| pseudo_random(1000 + thread, 256 * 2))
        .collect();
    let prime_inputs: Vec<Vec<Complex64>> = (0..8)
        .map(|thread| pseudo_random(5000 + thread, 97 * 2))
        .collect();

    let mut handles = Vec::new();
    for thread in 0..8 {
        let smooth = Arc::clone(&smooth);
        let prime = Arc::clone(&prime);
        let mut input = inputs[thread].clone();
        let mut prime_input = prime_inputs[thread].clone();
        handles.push(std::thread::spawn(move || {
            let mut scratch = vec![Complex64::new(0.0, 0.0); smooth.scratch_len()];
            smooth.process_with_scratch(&mut input, &mut scratch);
            let mut prime_scratch = vec![Complex64::new(0.0, 0.0); prime.scratch_len()];
            prime.process_with_scratch(&mut prime_input, &mut prime_scratch);
            (input, prime_input)
        }));
    }

    for (thread, handle) in handles.into_iter().enumerate() {
        let (concurrent, concurrent_prime) = handle.join().expect("worker panicked");
        let mut expected = inputs[thread].clone();
        let mut scratch = vec![Complex64::new(0.0, 0.0); smooth.scratch_len()];
        smooth.process_with_scratch(&mut expected, &mut scratch);
        assert_eq!(concurrent, expected);
        let mut expected_prime = prime_inputs[thread].clone();
        let mut prime_scratch = vec![Complex64::new(0.0, 0.0); prime.scratch_len()];
        prime.process_with_scratch(&mut expected_prime, &mut prime_scratch);
        assert_eq!(concurrent_prime, expected_prime);
    }
}
