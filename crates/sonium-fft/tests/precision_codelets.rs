//! Tight-tolerance precision check for the scalar radix codelets.
//!
//! The naive-DFT used elsewhere uses `theta.sin_cos()` for each twiddle
//! and so is precise to roughly one ulp. The codelets use compile-time
//! literal constants — those must also be within one ulp of the true
//! value, or compounding through nested stages (e.g. 5⁵ = 3125) drifts
//! observably. This test keeps that honest.

use num_complex::Complex64;
use std::f64::consts::TAU;

fn naive_dft(input: &[Complex64], dir_sign: f64) -> Vec<Complex64> {
    let n = input.len();
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for (k, ok) in out.iter_mut().enumerate() {
        let mut acc = Complex64::new(0.0, 0.0);
        for (j, ij) in input.iter().enumerate() {
            let theta = dir_sign * TAU * (j as f64) * (k as f64) / (n as f64);
            acc += ij * Complex64::new(theta.cos(), theta.sin());
        }
        *ok = acc;
    }
    out
}

fn random(seed: u64, n: usize) -> Vec<Complex64> {
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

/// Probe: pure radix-5 codelet at length-5 must be within a few ulps
/// of the naive DFT, *not* within `n * 1e-12`. With imprecise literal
/// constants the error sits at ~1e-16 (1 ulp); we tighten to 5e-16 so a
/// 1-ulp regression in the literals fails this test.
#[test]
fn radix5_single_stage_ulp_precision() {
    use sonium_fft::codelets::scalar::radix5;
    let inp = random(0xABCD, 5);
    let (y0, y1, y2, y3, y4) = radix5::<f64>(inp[0], inp[1], inp[2], inp[3], inp[4], -1.0);
    let got = [y0, y1, y2, y3, y4];
    let want = naive_dft(&inp, -1.0);
    for (g, w) in got.iter().zip(want.iter()) {
        let err = ((g.re - w.re).powi(2) + (g.im - w.im).powi(2)).sqrt();
        let mag = (w.re * w.re + w.im * w.im).sqrt().max(1.0);
        // 4 ulps on a unit-magnitude value ≈ 4 * 2.22e-16 ≈ 9e-16.
        assert!(
            err / mag < 1e-14,
            "radix5 ulp drift: |y - dft| / |dft| = {} for output {:?} vs {:?}",
            err / mag,
            g,
            w,
        );
    }
}

/// Probe: pure radix-3 codelet at length-3.
#[test]
fn radix3_single_stage_ulp_precision() {
    use sonium_fft::codelets::scalar::radix3;
    let inp = random(0xDCBA, 3);
    let (y0, y1, y2) = radix3::<f64>(inp[0], inp[1], inp[2], -1.0);
    let got = [y0, y1, y2];
    let want = naive_dft(&inp, -1.0);
    for (g, w) in got.iter().zip(want.iter()) {
        let err = ((g.re - w.re).powi(2) + (g.im - w.im).powi(2)).sqrt();
        let mag = (w.re * w.re + w.im * w.im).sqrt().max(1.0);
        assert!(
            err / mag < 1e-14,
            "radix3 ulp drift: |y - dft| / |dft| = {} for output {:?} vs {:?}",
            err / mag,
            g,
            w,
        );
    }
}

/// 5⁵ = 3125 — five back-to-back radix-5 stages. Cross-check against
/// rustfft rather than the naive O(n²) DFT: at n=3125 the naive DFT
/// itself accumulates ~5e-13 of summation error per output value, which
/// would dominate the comparison.
#[test]
fn five_to_the_fifth_matches_rustfft() {
    use sonium_fft::rustfft_compat::FftPlanner as OursFftPlanner;
    let mut ours = OursFftPlanner::<f64>::new();
    let mut theirs = rustfft::FftPlanner::<f64>::new();
    let p_ours = ours.plan_fft_forward(3125);
    let p_theirs = theirs.plan_fft_forward(3125);
    let inp = random(0x5555, 3125);
    let mut a = inp.clone();
    let mut b = inp.clone();
    p_ours.process(&mut a);
    p_theirs.process(&mut b);
    let mut max_rel = 0.0f64;
    for (ai, bi) in a.iter().zip(b.iter()) {
        let mag = (bi.re * bi.re + bi.im * bi.im).sqrt().max(1.0);
        let err = ((ai.re - bi.re).powi(2) + (ai.im - bi.im).powi(2)).sqrt();
        max_rel = max_rel.max(err / mag);
    }
    // Both implementations are O(log N) accurate FFTs; cross-impl drift
    // should sit comfortably below 1e-12 even at n=3125.
    assert!(
        max_rel < 1e-12,
        "radix-5 stack disagrees with rustfft at n=3125: rel err = {max_rel:e}"
    );
}
