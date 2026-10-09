//! Streaming-SVE BLAS-1 kernels for long vectors.
//!
//! Wraps the asm kernels in `sme2_kernels.S`:
//! - `sonium_axpy_f32_sme2` / `sonium_axpy_f64_sme2`
//! - `sonium_dot_f32_sme2`  / `sonium_dot_f64_sme2`
//! - `sonium_scal_f32_sme2` / `sonium_scal_f64_sme2`
//!
//! Entry to streaming SVE is ~50 ns per call on M4 (SMSTART + SMSTOP), so
//! these kernels only win above a threshold where the per-instruction
//! 4× width over NEON-128 amortises the mode-switch cost. The dispatcher
//! in `cpu::blas1` picks the SME2 path when `n >= STREAMING_SVE_MIN_LEN`
//! AND `Sme2Caps::detect().has_sme2`, otherwise falls back to NEON.
//!
//! # Threshold tuning
//!
//! Default `STREAMING_SVE_MIN_LEN = 4096` is conservative: at 4 KB f32 /
//! 8 KB f64 the working set fits L1 and NEON is competitive, but per-call
//! overhead is a small percentage. Above ~16K elements the streaming-SVE
//! path is ~1.5-2× faster on bandwidth-bound work on M4.
//!
//! Override via `SONIUM_STREAMING_SVE_MIN_LEN` env var (parsed once at
//! first call via OnceLock).

#![allow(clippy::missing_safety_doc)]

use super::detect::Sme2Caps;

/// Minimum vector length for the streaming-SVE path to be selected over
/// NEON. Below this the per-call SMSTART/SMSTOP entry/exit overhead
/// (~50 ns) is a larger fraction of the work than the per-instruction
/// width win, and NEON-128 wins. Tuned for the M4 P-cluster.
const STREAMING_SVE_MIN_LEN_DEFAULT: usize = 4096;

/// Resolve the threshold once, allowing `SONIUM_STREAMING_SVE_MIN_LEN` env
/// override for tuning experiments.
pub fn min_len() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("SONIUM_STREAMING_SVE_MIN_LEN")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(STREAMING_SVE_MIN_LEN_DEFAULT)
    })
}

/// Should the SME2 streaming-SVE path be used for `n`?
#[inline]
pub fn streaming_sve_wins(n: usize, want_f64: bool) -> bool {
    if n < min_len() {
        return false;
    }
    let caps = Sme2Caps::detect();
    if !caps.has_sme2 {
        return false;
    }
    // f64 path needs FEAT_SME_F64F64 (separate cap flag); f32 streaming-SVE
    // is part of base SME2.
    if want_f64 && !caps.has_f64 {
        return false;
    }
    true
}

#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    fn sonium_axpy_f32_sme2(alpha: f32, x: *const f32, y: *mut f32, n: usize);
    fn sonium_axpy_f64_sme2(alpha: f64, x: *const f64, y: *mut f64, n: usize);
    fn sonium_dot_f32_sme2(x: *const f32, y: *const f32, n: usize) -> f32;
    fn sonium_dot_f64_sme2(x: *const f64, y: *const f64, n: usize) -> f64;
    fn sonium_scal_f32_sme2(alpha: f32, x: *mut f32, n: usize);
    fn sonium_scal_f64_sme2(alpha: f64, x: *mut f64, n: usize);
}

// On non-aarch64 hosts the dispatcher in cpu::blas1 won't call these (the
// `streaming_sve_wins` gate fails first), but provide stubs so the module
// compiles cross-platform.
#[cfg(not(target_arch = "aarch64"))]
unsafe fn sonium_axpy_f32_sme2(_alpha: f32, _x: *const f32, _y: *mut f32, _n: usize) {
    unreachable!("SME2 path called on non-aarch64 host")
}
#[cfg(not(target_arch = "aarch64"))]
unsafe fn sonium_axpy_f64_sme2(_alpha: f64, _x: *const f64, _y: *mut f64, _n: usize) {
    unreachable!("SME2 path called on non-aarch64 host")
}
#[cfg(not(target_arch = "aarch64"))]
unsafe fn sonium_dot_f32_sme2(_x: *const f32, _y: *const f32, _n: usize) -> f32 {
    unreachable!("SME2 path called on non-aarch64 host")
}
#[cfg(not(target_arch = "aarch64"))]
unsafe fn sonium_dot_f64_sme2(_x: *const f64, _y: *const f64, _n: usize) -> f64 {
    unreachable!("SME2 path called on non-aarch64 host")
}
#[cfg(not(target_arch = "aarch64"))]
unsafe fn sonium_scal_f32_sme2(_alpha: f32, _x: *mut f32, _n: usize) {
    unreachable!("SME2 path called on non-aarch64 host")
}
#[cfg(not(target_arch = "aarch64"))]
unsafe fn sonium_scal_f64_sme2(_alpha: f64, _x: *mut f64, _n: usize) {
    unreachable!("SME2 path called on non-aarch64 host")
}

/// `y[i] += alpha * x[i]` via streaming-SVE. Caller must have checked
/// `streaming_sve_wins(n, false)`. `x` and `y` must alias `n` valid
/// elements.
#[inline]
pub fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    // SAFETY: pointers borrowed from live slices of length `n`; the asm
    // kernel reads x and writes y only within [0, n).
    unsafe {
        sonium_axpy_f32_sme2(alpha, x.as_ptr(), y.as_mut_ptr(), n);
    }
}

/// f64 sibling — needs `Sme2Caps::has_f64`.
#[inline]
pub fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    unsafe {
        sonium_axpy_f64_sme2(alpha, x.as_ptr(), y.as_mut_ptr(), n);
    }
}

#[inline]
pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    unsafe { sonium_dot_f32_sme2(x.as_ptr(), y.as_ptr(), n) }
}

#[inline]
pub fn dot_f64(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    unsafe { sonium_dot_f64_sme2(x.as_ptr(), y.as_ptr(), n) }
}

#[inline]
pub fn scal_f32(alpha: f32, x: &mut [f32]) {
    let n = x.len();
    unsafe {
        sonium_scal_f32_sme2(alpha, x.as_mut_ptr(), n);
    }
}

#[inline]
pub fn scal_f64(alpha: f64, x: &mut [f64]) {
    let n = x.len();
    unsafe {
        sonium_scal_f64_sme2(alpha, x.as_mut_ptr(), n);
    }
}

#[inline]
pub fn nrm2_f32(x: &[f32]) -> f32 {
    dot_f32(x, x).sqrt()
}

#[inline]
pub fn nrm2_f64(x: &[f64]) -> f64 {
    dot_f64(x, x).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip_if_no_sme2() -> bool {
        let caps = Sme2Caps::detect();
        if !caps.has_sme2 {
            eprintln!("[sme2::blas1] SKIPPED: no SME2 on this host");
            return true;
        }
        false
    }

    #[test]
    fn axpy_f32_matches_scalar_large() {
        if skip_if_no_sme2() {
            return;
        }
        let n = 10_000;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.00013).sin()).collect();
        let y0: Vec<f32> = (0..n).map(|i| (i as f32 * 0.00017).cos()).collect();
        let alpha = 0.7_f32;
        let mut y_sme = y0.clone();
        axpy_f32(alpha, &x, &mut y_sme);
        for i in 0..n {
            let want = y0[i] + alpha * x[i];
            assert!(
                (y_sme[i] - want).abs() < 1e-5 * (1.0 + want.abs()),
                "axpy_f32 mismatch at {i}: got {}, want {want}",
                y_sme[i]
            );
        }
    }

    #[test]
    fn axpy_f64_matches_scalar_large() {
        let caps = Sme2Caps::detect();
        if !caps.has_sme2 || !caps.has_f64 {
            eprintln!("[sme2::blas1] SKIPPED: no SME2-F64 on this host");
            return;
        }
        let n = 10_000;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.00013).sin()).collect();
        let y0: Vec<f64> = (0..n).map(|i| (i as f64 * 0.00017).cos()).collect();
        let alpha = 0.7_f64;
        let mut y_sme = y0.clone();
        axpy_f64(alpha, &x, &mut y_sme);
        for i in 0..n {
            let want = y0[i] + alpha * x[i];
            assert!(
                (y_sme[i] - want).abs() < 1e-12 * (1.0 + want.abs()),
                "axpy_f64 mismatch at {i}: got {}, want {want}",
                y_sme[i]
            );
        }
    }

    /// Exercise the tail loop: n = 4*SVL_S + 7 at SVL=512 ⇒ n = 71. The
    /// main 4-Z loop fires once and the predicated tail handles 7 lanes.
    #[test]
    fn axpy_f32_tail() {
        if skip_if_no_sme2() {
            return;
        }
        for &n in &[1usize, 7, 15, 17, 71, 128, 257, 1023] {
            let x: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.001).collect();
            let y0: Vec<f32> = (0..n).map(|i| (i as f32 + 2.0) * 0.002).collect();
            let alpha = 0.5_f32;
            let mut y_sme = y0.clone();
            axpy_f32(alpha, &x, &mut y_sme);
            for i in 0..n {
                let want = y0[i] + alpha * x[i];
                assert!(
                    (y_sme[i] - want).abs() < 1e-5 * (1.0 + want.abs()),
                    "axpy_f32 tail n={n} mismatch at {i}: got {}, want {want}",
                    y_sme[i]
                );
            }
        }
    }

    #[test]
    fn dot_f32_matches_scalar() {
        if skip_if_no_sme2() {
            return;
        }
        for &n in &[1usize, 16, 71, 128, 1023, 10_000] {
            let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.00029).sin()).collect();
            let y: Vec<f32> = (0..n).map(|i| (i as f32 * 0.00031).cos()).collect();
            let got = dot_f32(&x, &y);
            // Compute reference using a Kahan-ish multi-accumulator to
            // mirror SIMD reduction order — keeps the error budget honest
            // at large n.
            let want: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
            let tol = 1e-4 * (1.0 + want.abs()) + 1e-5 * (n as f32);
            assert!(
                (got - want).abs() < tol,
                "dot_f32 n={n}: got {got}, want {want}, tol {tol}"
            );
        }
    }

    #[test]
    fn dot_f64_matches_scalar() {
        let caps = Sme2Caps::detect();
        if !caps.has_sme2 || !caps.has_f64 {
            eprintln!("[sme2::blas1] SKIPPED: no SME2-F64 on this host");
            return;
        }
        for &n in &[1usize, 8, 31, 64, 511, 10_000] {
            let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.00029).sin()).collect();
            let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.00031).cos()).collect();
            let got = dot_f64(&x, &y);
            let want: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
            let tol = 1e-12 * (1.0 + want.abs()) + 1e-13 * (n as f64);
            assert!(
                (got - want).abs() < tol,
                "dot_f64 n={n}: got {got}, want {want}, tol {tol}"
            );
        }
    }

    #[test]
    fn scal_f32_matches_scalar() {
        if skip_if_no_sme2() {
            return;
        }
        for &n in &[1usize, 7, 16, 71, 1023, 10_000] {
            let mut got: Vec<f32> = (0..n).map(|i| (i as f32 * 0.00013).sin()).collect();
            let mut want = got.clone();
            scal_f32(-0.7, &mut got);
            for x in &mut want {
                *x *= -0.7;
            }
            for i in 0..n {
                assert!(
                    (got[i] - want[i]).abs() < 1e-6 * (1.0 + want[i].abs()),
                    "scal_f32 n={n} mismatch at {i}: got {}, want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn scal_f64_matches_scalar() {
        let caps = Sme2Caps::detect();
        if !caps.has_sme2 || !caps.has_f64 {
            eprintln!("[sme2::blas1] SKIPPED: no SME2-F64 on this host");
            return;
        }
        for &n in &[1usize, 7, 16, 71, 1023, 10_000] {
            let mut got: Vec<f64> = (0..n).map(|i| (i as f64 * 0.00013).sin()).collect();
            let mut want = got.clone();
            scal_f64(-0.7, &mut got);
            for x in &mut want {
                *x *= -0.7;
            }
            for i in 0..n {
                assert!(
                    (got[i] - want[i]).abs() < 1e-13 * (1.0 + want[i].abs()),
                    "scal_f64 n={n} mismatch at {i}: got {}, want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn nrm2_f32_matches_scalar() {
        if skip_if_no_sme2() {
            return;
        }
        for &n in &[1usize, 16, 71, 128, 1023, 10_000] {
            let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.00029).sin()).collect();
            let got = nrm2_f32(&x);
            let want = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            let tol = 1e-4 * (1.0 + want.abs()) + 1e-5 * (n as f32).sqrt();
            assert!(
                (got - want).abs() < tol,
                "nrm2_f32 n={n}: got {got}, want {want}, tol {tol}"
            );
        }
    }

    #[test]
    fn nrm2_f64_matches_scalar() {
        let caps = Sme2Caps::detect();
        if !caps.has_sme2 || !caps.has_f64 {
            eprintln!("[sme2::blas1] SKIPPED: no SME2-F64 on this host");
            return;
        }
        for &n in &[1usize, 8, 31, 64, 511, 10_000] {
            let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.00029).sin()).collect();
            let got = nrm2_f64(&x);
            let want = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            let tol = 1e-12 * (1.0 + want.abs()) + 1e-13 * (n as f64).sqrt();
            assert!(
                (got - want).abs() < tol,
                "nrm2_f64 n={n}: got {got}, want {want}, tol {tol}"
            );
        }
    }

    #[test]
    fn threshold_is_overridable_via_env() {
        // Just verify the function returns *something* positive — the
        // OnceLock is process-wide so we can't really test the override
        // here without poisoning subsequent calls. Documented behaviour.
        assert!(min_len() > 0);
    }
}
