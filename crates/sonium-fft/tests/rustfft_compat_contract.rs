//! Contract pins for the `rustfft_compat` shim — these test the parts of
//! the rustfft 6.x API surface that the shim claims to provide. Failures
//! here mean migrating users will see different behaviour than they get
//! from upstream rustfft.

use num_complex::Complex64;
use sonium_fft::rustfft_compat::{Direction, FftPlanner, Length};

#[test]
fn fft_impls_length_supertrait() {
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(64);
    fn takes_length<L: Length + ?Sized>(l: &L) -> usize {
        l.len()
    }
    assert_eq!(takes_length(&*plan), 64);
}

#[test]
fn fft_impls_direction_supertrait() {
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(64);
    fn takes_direction<D: Direction + ?Sized>(d: &D) -> sonium_fft::FftDirection {
        d.fft_direction()
    }
    assert_eq!(takes_direction(&*plan), sonium_fft::FftDirection::Forward);
}

#[test]
#[should_panic(expected = "buffer")]
fn empty_buffer_panics_in_rustfft_compat() {
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(4);
    let mut empty: Vec<Complex64> = Vec::new();
    plan.process(&mut empty);
}

#[test]
#[should_panic(expected = "buffer")]
fn short_buffer_panics_in_rustfft_compat() {
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(8);
    let mut buf = vec![Complex64::new(0.0, 0.0); 4]; // less than plan length
    plan.process(&mut buf);
}

#[test]
#[should_panic(expected = "scratch length")]
fn short_scratch_panics_in_rustfft_compat() {
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(8);
    let mut buffer = vec![Complex64::new(0.0, 0.0); 8];
    let mut scratch = vec![Complex64::new(0.0, 0.0); plan.get_inplace_scratch_len() - 1];
    plan.process_with_scratch(&mut buffer, &mut scratch);
}

#[test]
fn process_immutable_with_scratch_matches_in_place() {
    let mut planner = FftPlanner::<f64>::new();
    let n = 128;
    let plan = planner.plan_fft_forward(n);
    let input: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64).sin(), (i as f64).cos()))
        .collect();

    // Reference: in-place processing
    let mut buf_ref = input.clone();
    plan.process(&mut buf_ref);

    // Immutable scratch path
    let mut output = vec![Complex64::new(0.0, 0.0); n];
    let mut scratch = vec![Complex64::new(0.0, 0.0); plan.get_immutable_scratch_len()];
    plan.process_immutable_with_scratch(&input, &mut output, &mut scratch);

    // Input is unchanged
    for (got, want) in input.iter().zip(
        (0..n)
            .map(|i| Complex64::new((i as f64).sin(), (i as f64).cos()))
            .collect::<Vec<_>>()
            .iter(),
    ) {
        assert_eq!(got, want, "immutable variant must not modify input");
    }
    // Output equals in-place result
    assert_eq!(buf_ref, output);
}
