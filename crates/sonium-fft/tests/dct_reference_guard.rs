//! Guard against accidental use of the quadratic DCT/DST reference path.
//!
//! The free functions ([`dct2`], [`dct3`], [`dst2`], [`dst3`] and their
//! Euclidean adjoints) are direct O(N^2) summations at every length, while
//! [`DctPlan`] / [`DstPlan`] switch to an O(N log N) FFT kernel above
//! [`DIRECT_THRESHOLD`]. These tests pin that public contract: the threshold
//! value, the plan dispatch boundary, and numerical agreement between the
//! plan (FFT) path and the free (reference) path above the threshold, so a
//! caller choosing the reference path for a large workload fails review
//! against a checked number rather than silently paying O(N^2).

use sonium_fft::{
    DIRECT_THRESHOLD, DctPlan, DstPlan, dct2, dct2_adjoint, dct3, dct3_adjoint, dst2, dst2_adjoint,
    dst3, dst3_adjoint,
};

fn pseudo_random(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        })
        .collect()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        let allowed = tol * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= allowed,
            "{label} index {index}: got={got:e}, want={want:e}, tol={allowed:e}"
        );
    }
}

#[test]
fn direct_threshold_contract_is_published_and_stable() {
    // Pin the value so a threshold change is a deliberate, reviewed edit.
    assert_eq!(DIRECT_THRESHOLD, 64);
    for n in [0usize, 1, 63, 64] {
        assert!(
            !DctPlan::<f64>::new(n).uses_fft(),
            "DctPlan({n}) should use the direct path"
        );
        assert!(
            !DstPlan::<f64>::new(n).uses_fft(),
            "DstPlan({n}) should use the direct path"
        );
    }
    for n in [65usize, 100, 127, 200, 256] {
        assert!(
            DctPlan::<f64>::new(n).uses_fft(),
            "DctPlan({n}) should use the FFT kernel"
        );
        assert!(
            DstPlan::<f64>::new(n).uses_fft(),
            "DstPlan({n}) should use the FFT kernel"
        );
    }
}

#[test]
fn plans_agree_with_reference_above_threshold() {
    // Above DIRECT_THRESHOLD the plans run the FFT kernel while the free
    // functions stay O(N^2); both must agree so the plan is a safe
    // drop-in for the reference at large lengths.
    for &n in &[65usize, 100, 128, 200, 256] {
        let input = pseudo_random(n, n as u64 * 31 + 7);
        let tol = (n as f64 * 1e-13).max(1e-12);
        let mut reference = vec![0.0; n];
        let mut planned = vec![0.0; n];
        let dct = DctPlan::<f64>::new(n);
        let dst = DstPlan::<f64>::new(n);

        dct2(&input, &mut reference);
        dct.forward(&input, &mut planned);
        assert_close(&planned, &reference, tol, "dct2 plan vs reference");

        dct3(&input, &mut reference);
        dct.inverse(&input, &mut planned);
        assert_close(&planned, &reference, tol, "dct3 plan vs reference");

        dct2_adjoint(&input, &mut reference);
        dct.forward_adjoint(&input, &mut planned);
        assert_close(&planned, &reference, tol, "dct2_adjoint plan vs reference");

        dct3_adjoint(&input, &mut reference);
        dct.inverse_adjoint(&input, &mut planned);
        assert_close(&planned, &reference, tol, "dct3_adjoint plan vs reference");

        dst2(&input, &mut reference);
        dst.forward(&input, &mut planned);
        assert_close(&planned, &reference, tol, "dst2 plan vs reference");

        dst3(&input, &mut reference);
        dst.inverse(&input, &mut planned);
        assert_close(&planned, &reference, tol, "dst3 plan vs reference");

        dst2_adjoint(&input, &mut reference);
        dst.forward_adjoint(&input, &mut planned);
        assert_close(&planned, &reference, tol, "dst2_adjoint plan vs reference");

        dst3_adjoint(&input, &mut reference);
        dst.inverse_adjoint(&input, &mut planned);
        assert_close(&planned, &reference, tol, "dst3_adjoint plan vs reference");
    }
}
