#![no_main]

use libfuzzer_sys::fuzz_target;
use num_complex::Complex64;
use sonium_fft::rustfft_compat::{Fft as CompatFft, FftPlanner as CompatFftPlanner};
use sonium_fft::{FftDirection, FftPlanner};

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    // Bound lengths and batches while retaining smooth, mixed-radix, prime,
    // and Bluestein-backed cases. The plan's advertised scratch size is used
    // verbatim so this also fuzzes the execution contract.
    let n = 1 + usize::from(u16::from_le_bytes([data[0], data[1]]) % 513);
    let batch = 1 + usize::from(data[2] % 4);
    let mut state = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state
    };

    let planner = FftPlanner::<f64>::new();
    let forward = planner.plan(n, FftDirection::Forward);
    let inverse = planner.plan(n, FftDirection::Inverse);
    let mut buffer = Vec::with_capacity(n * batch);
    for _ in 0..n * batch {
        let re = f64::from(next()) / f64::from(u32::MAX) - 0.5;
        let im = f64::from(next()) / f64::from(u32::MAX) - 0.5;
        buffer.push(Complex64::new(re, im));
    }
    let expected = buffer.clone();
    // Exercise both the exact lower bound and accepted oversized scratch
    // buffers. The extra capacity is deliberately small so fuzzing remains
    // focused on the planner/execution contract rather than allocation size.
    let scratch_extra = usize::from(data.get(4).copied().unwrap_or_default() % 17);
    let required_scratch = forward.scratch_len().max(inverse.scratch_len());
    let mut scratch = vec![Complex64::new(0.0, 0.0); required_scratch + scratch_extra];

    forward.process_with_scratch(&mut buffer, &mut scratch);
    assert!(buffer
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite()));

    scratch.fill(Complex64::new(0.0, 0.0));
    inverse.process_with_scratch(&mut buffer, &mut scratch);
    let tolerance = 1.0e-6 * (n as f64).sqrt().max(1.0);
    for (actual, expected) in buffer.iter().zip(expected.iter()) {
        assert!((actual / n as f64 - expected).norm() <= tolerance);
    }

    // The compatibility adapter has a stricter public contract than the
    // native plan. Exercise each scratch-bearing entry point with valid
    // buffers and verify that the immutable variant does not modify input.
    let mut compat_planner = CompatFftPlanner::<f64>::new();
    let compat_forward = compat_planner.plan_fft_forward(n);
    let compat_inverse = compat_planner.plan_fft_inverse(n);
    let forward_scratch_len = compat_forward.get_inplace_scratch_len();
    let inverse_scratch_len = compat_inverse.get_inplace_scratch_len();

    let mut compat_buffer = expected.clone();
    let mut compat_scratch = vec![Complex64::new(0.0, 0.0); forward_scratch_len];
    compat_forward.process_with_scratch(&mut compat_buffer, &mut compat_scratch);
    let compat_forward_output = compat_buffer.clone();

    compat_scratch.resize(inverse_scratch_len, Complex64::new(0.0, 0.0));
    compat_inverse.process_with_scratch(&mut compat_buffer, &mut compat_scratch);
    for (actual, original) in compat_buffer.iter().zip(expected.iter()) {
        assert!((actual / n as f64 - original).norm() <= tolerance);
    }

    let mut outofplace_input = expected.clone();
    let input_snapshot = outofplace_input.clone();
    let mut outofplace_output = vec![Complex64::new(0.0, 0.0); expected.len()];
    let mut outofplace_scratch =
        vec![Complex64::new(0.0, 0.0); compat_forward.get_outofplace_scratch_len()];
    compat_forward.process_outofplace_with_scratch(
        &mut outofplace_input,
        &mut outofplace_output,
        &mut outofplace_scratch,
    );
    assert_eq!(outofplace_input, input_snapshot);
    assert_eq!(outofplace_output, compat_forward_output);

    let mut immutable_output = vec![Complex64::new(0.0, 0.0); expected.len()];
    let mut immutable_scratch =
        vec![Complex64::new(0.0, 0.0); compat_forward.get_immutable_scratch_len()];
    compat_forward.process_immutable_with_scratch(
        &expected,
        &mut immutable_output,
        &mut immutable_scratch,
    );
    assert_eq!(immutable_output, compat_forward_output);
});
