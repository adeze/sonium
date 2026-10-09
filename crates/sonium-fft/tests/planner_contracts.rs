//! Deterministic property-style coverage for planner and scratch contracts.

use num_complex::Complex64;
use sonium_fft::cpu::planner::CpuFftPlanner;
use sonium_fft::{Fft2dPlan, Fft3dPlan, FftDirection, FftPlanError, FftPlanner, NdFftError};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn varied_lengths_and_precisions_round_trip() {
    for &n in &[1usize, 2, 3, 5, 7, 11, 16, 31, 64, 97, 128, 257] {
        let planner = FftPlanner::<f64>::new();
        let forward = planner.plan(n, FftDirection::Forward);
        let inverse = planner.plan(n, FftDirection::Inverse);
        let mut buffer: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.23).sin(), (i as f64 * 0.17).cos()))
            .collect();
        forward.process(&mut buffer);
        inverse.process(&mut buffer);
        for (i, value) in buffer.iter().enumerate() {
            let expected = Complex64::new((i as f64 * 0.23).sin(), (i as f64 * 0.17).cos());
            assert!((value / n as f64 - expected).norm() < 2e-9, "n={n}, i={i}");
        }
    }
}

#[test]
fn smooth_plans_publish_cache_section_schedule_metadata() {
    let planner = FftPlanner::<f64>::new();
    let plan = planner.plan_forward(16_384);
    assert!(plan.cache_section_count() > 0);
    assert!(plan.cache_section_budget_bytes() > 0);
}

#[test]
fn native_empty_is_noop_and_invalid_shapes_panic() {
    let planner = FftPlanner::<f64>::new();
    let plan = planner.plan_forward(16);
    let mut empty = Vec::<Complex64>::new();
    plan.process(&mut empty);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            plan.process(&mut vec![Complex64::new(0.0, 0.0); 15]);
        }))
        .is_err()
    );
    let mut buffer = vec![Complex64::new(0.0, 0.0); 16];
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            plan.process_with_scratch(&mut buffer, &mut vec![Complex64::new(0.0, 0.0); 15]);
        }))
        .is_err()
    );
}

#[test]
fn factorization_rejects_overflowing_bluestein_lengths() {
    assert_eq!(
        sonium_fft::factor::try_plan_length(usize::MAX),
        Err(sonium_fft::factor::PlanLengthError::LengthOverflow { length: usize::MAX })
    );

    // Keep the panic-compatible API covered as a compatibility boundary, but
    // make the checked result the source of truth for the typed failure.
    assert!(catch_unwind(|| sonium_fft::factor::plan_length(usize::MAX)).is_err());
}

#[test]
fn multidimensional_plans_reject_overflowing_shapes() {
    assert!(matches!(
        Fft2dPlan::<f64>::try_new(usize::MAX, 2, FftDirection::Forward),
        Err(NdFftError::LengthOverflow)
    ));
    assert!(matches!(
        Fft3dPlan::<f64>::try_new(usize::MAX, 2, 2, FftDirection::Forward),
        Err(NdFftError::LengthOverflow)
    ));

    // Compatibility constructors intentionally panic after the checked
    // constructors reject the impossible shape.
    assert!(catch_unwind(|| Fft2dPlan::<f64>::new(usize::MAX, 2, FftDirection::Forward)).is_err());
    assert!(
        catch_unwind(|| {
            Fft3dPlan::<f64>::new(usize::MAX, 2, 2, FftDirection::Forward);
        })
        .is_err()
    );
}

#[test]
fn deterministic_fuzz_planner_and_scratch_contracts() {
    // Keep this property-style sweep dependency-free and reproducible. It
    // exercises smooth, mixed-radix, prime, and Bluestein-backed lengths with
    // both single and batched buffers while using the exact advertised
    // scratch size.
    let mut state = 0x4D595DF4D0F33173_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state
    };
    let planner = FftPlanner::<f64>::new();

    for _ in 0..64 {
        let n = 1 + (next() as usize % 192);
        let batch = 1 + (next() as usize % 3);
        let forward = planner.plan(n, FftDirection::Forward);
        let inverse = planner.plan(n, FftDirection::Inverse);
        let mut input = Vec::with_capacity(n * batch);
        for _ in 0..n * batch {
            let re = (next() as f64 / u64::MAX as f64) - 0.5;
            let im = (next() as f64 / u64::MAX as f64) - 0.5;
            input.push(Complex64::new(re, im));
        }
        let expected = input.clone();
        let mut scratch = vec![Complex64::new(0.0, 0.0); forward.scratch_len()];
        forward.process_with_scratch(&mut input, &mut scratch);
        assert!(
            input
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );

        scratch.fill(Complex64::new(0.0, 0.0));
        inverse.process_with_scratch(&mut input, &mut scratch);
        for (actual, expected) in input.iter().zip(expected.iter()) {
            assert!(
                (actual / n as f64 - expected).norm() < 3e-8,
                "n={n}, batch={batch}"
            );
        }
    }
}

#[test]
fn native_checked_process_rejects_bad_shape_and_scratch_without_panicking() {
    let planner = FftPlanner::<f64>::new();
    let plan = planner.plan(16, FftDirection::Forward);

    let mut malformed = vec![Complex64::new(0.0, 0.0); 17];
    assert_eq!(
        plan.try_process(&mut malformed),
        Err(FftPlanError::InvalidBufferLength {
            plan_length: 16,
            buffer_length: 17,
        })
    );

    if plan.scratch_len() > 0 {
        let mut buffer = vec![Complex64::new(0.0, 0.0); 16];
        let mut short_scratch = Vec::new();
        assert_eq!(
            plan.try_process_with_scratch(&mut buffer, &mut short_scratch),
            Err(FftPlanError::ScratchTooShort {
                required: plan.scratch_len(),
                actual: 0,
            })
        );
    }

    let mut buffer = vec![Complex64::new(0.0, 0.0); 16];
    let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len()];
    plan.try_process_with_scratch(&mut buffer, &mut scratch)
        .expect("valid checked native FFT execution");
}

#[test]
fn checked_planner_construction_reports_typed_length_errors() {
    use sonium_fft::factor::PlanLengthError;

    let planner = FftPlanner::<f64>::new();
    assert!(matches!(
        planner.try_plan(0, FftDirection::Forward),
        Err(PlanLengthError::ZeroLength)
    ));
    assert!(matches!(
        planner.try_plan(usize::MAX, FftDirection::Forward),
        Err(PlanLengthError::LengthOverflow { .. })
    ));
    // The Bluestein fallible entry point reports the same overflow type.
    assert!(matches!(
        sonium_fft::bluestein::BluesteinPlan::<f64>::try_new(usize::MAX, FftDirection::Forward),
        Err(PlanLengthError::LengthOverflow { .. })
    ));
    // Representable lengths still build through the checked path.
    assert!(planner.try_plan(97, FftDirection::Forward).is_ok());
}

#[test]
fn checked_empty_buffer_is_a_successful_noop() {
    let planner = FftPlanner::<f64>::new();
    for n in [16usize, 97] {
        let plan = planner.plan(n, FftDirection::Forward);
        let mut empty = Vec::<Complex64>::new();
        plan.try_process(&mut empty)
            .expect("empty native buffer must be a no-op");
        let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len()];
        plan.try_process_with_scratch(&mut empty, &mut scratch)
            .expect("empty native buffer with scratch must be a no-op");
    }
}

#[test]
fn checked_adjoint_pair_preserves_shared_scratch_contract() {
    let planner = CpuFftPlanner::<f64>::new();
    let pair = planner.plan_with_adjoint(16, FftDirection::Forward);
    let mut buffer = vec![Complex64::new(0.0, 0.0); 16];
    let mut scratch = vec![Complex64::new(0.0, 0.0); pair.scratch_len()];

    pair.try_process_with_scratch(&mut buffer, &mut scratch)
        .expect("checked primary transform");
    pair.try_process_adjoint_with_scratch(&mut buffer, &mut scratch)
        .expect("checked adjoint transform");

    let mut malformed = vec![Complex64::new(0.0, 0.0); 15];
    assert!(matches!(
        pair.try_process(&mut malformed),
        Err(FftPlanError::InvalidBufferLength { .. })
    ));
}
