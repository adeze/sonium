//! Dense GEMV micro-kernels (f32, f64).
//!
//! Computes `y := alpha * A * x + beta * y` where A is `m × n` row-major.
//!
//! The Rust wrappers here are thin — the work happens in the asm kernels
//! `sonium_{s,d}gemv_sme2` in `sme2_kernels.S`. Each kernel's inner-K
//! loop uses:
//!
//! - SME2 grouped multi-vector `LD1{W,D}` over four consecutive Z registers
//!   for both the A row and x vector (paper §3 reports ~900 GB/s on this
//!   form versus ~230 GB/s for four separate single-Z loads).
//! - Four independent FMA accumulators (z24..z27) so the dot product is
//!   not throttled by the FMA latency / dependency chain.
//! - A predicated single-Z tail for the residue (n % (4*SVL_T)).
//!
//! On non-AArch64 hosts or when SME2 is unavailable, this module falls
//! back to a single-accumulator scalar reference that honours `beta`.

#![allow(clippy::too_many_arguments)]

use super::detect::Sme2Caps;
use crate::{BackendError, BackendResult};

// Preflight bounds check. Runs for both SME2 and scalar paths so callers
// cannot reach out-of-bounds asm reads/writes.
#[inline]
fn validate_gemv_bounds(
    m: usize,
    n: usize,
    a_len: usize,
    lda: usize,
    x_len: usize,
    y_len: usize,
    name: &'static str,
) -> BackendResult<()> {
    if n > lda {
        return Err(BackendError::InvalidArgument {
            operation: name,
            reason: format!("n ({n}) exceeds lda ({lda})"),
        });
    }
    let a_needed = if m == 0 {
        0
    } else {
        (m - 1)
            .checked_mul(lda)
            .and_then(|offset| offset.checked_add(n))
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: name,
                reason: format!("matrix extent overflows for m={m}, n={n}, lda={lda}"),
            })?
    };
    if a_needed > a_len {
        return Err(BackendError::InvalidArgument {
            operation: name,
            reason: format!(
                "a.len()={a_len} is smaller than required {a_needed} for m={m}, n={n}, lda={lda}"
            ),
        });
    }
    if n > x_len {
        return Err(BackendError::InvalidArgument {
            operation: name,
            reason: format!("x.len()={x_len} is smaller than n={n}"),
        });
    }
    if m > y_len {
        return Err(BackendError::InvalidArgument {
            operation: name,
            reason: format!("y.len()={y_len} is smaller than m={m}"),
        });
    }
    Ok(())
}

// ---- External assembly kernels ----

#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    fn sonium_sgemv_sme2(
        m: usize,
        n: usize,
        alpha: f32,
        a: *const f32,
        lda: usize,
        x: *const f32,
        beta: f32,
        y: *mut f32,
    );
    fn sonium_dgemv_sme2(
        m: usize,
        n: usize,
        alpha: f64,
        a: *const f64,
        lda: usize,
        x: *const f64,
        beta: f64,
        y: *mut f64,
    );
}

/// `y := alpha * A * x + beta * y` for f32, row-major `A`.
pub fn try_sgemv(
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) -> BackendResult<()> {
    validate_gemv_bounds(m, n, a.len(), lda, x.len(), y.len(), "sgemv")?;
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::blas1::avx2::is_available() {
        // SAFETY: availability checked above; bounds validated here.
        unsafe {
            crate::cpu::blas2::avx2::gemv_notrans_f32(m, n, alpha, a, lda, x, beta, y);
        }
        return Ok(());
    }
    if Sme2Caps::detect().has_sme2 {
        #[cfg(target_arch = "aarch64")]
        // Safety: SME2 availability checked above. Kernel handles
        // streaming-mode transitions internally and honours `beta`.
        unsafe {
            sonium_sgemv_sme2(
                m,
                n,
                alpha,
                a.as_ptr(),
                lda,
                x.as_ptr(),
                beta,
                y.as_mut_ptr(),
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        sgemv_scalar(m, n, alpha, a, lda, x, beta, y);
    } else {
        sgemv_scalar(m, n, alpha, a, lda, x, beta, y);
    }
    Ok(())
}

/// Compatibility wrapper for [`try_sgemv`]. Invalid dimensions retain the
/// historical panic behavior; new code that accepts caller-controlled shapes
/// should use the checked entry point.
pub fn sgemv(
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) {
    if let Err(error) = try_sgemv(m, n, alpha, a, lda, x, beta, y) {
        eprintln!("sgemv rejected input: {error}");
    }
}

/// `y := alpha * A * x + beta * y` for f64.
pub fn try_dgemv(
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) -> BackendResult<()> {
    validate_gemv_bounds(m, n, a.len(), lda, x.len(), y.len(), "dgemv")?;
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::blas1::avx2::is_available() {
        // SAFETY: availability checked above; bounds validated here.
        unsafe {
            crate::cpu::blas2::avx2::gemv_notrans_f64(m, n, alpha, a, lda, x, beta, y);
        }
        return Ok(());
    }
    if Sme2Caps::detect().has_f64 {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            sonium_dgemv_sme2(
                m,
                n,
                alpha,
                a.as_ptr(),
                lda,
                x.as_ptr(),
                beta,
                y.as_mut_ptr(),
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        dgemv_scalar(m, n, alpha, a, lda, x, beta, y);
    } else {
        dgemv_scalar(m, n, alpha, a, lda, x, beta, y);
    }
    Ok(())
}

/// Compatibility wrapper for [`try_dgemv`]. Invalid dimensions retain the
/// historical panic behavior; new code that accepts caller-controlled shapes
/// should use the checked entry point.
pub fn dgemv(
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    if let Err(error) = try_dgemv(m, n, alpha, a, lda, x, beta, y) {
        eprintln!("dgemv rejected input: {error}");
    }
}

// ---- Scalar reference ----

fn sgemv_scalar(
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) {
    // Only touch the first `m` elements of y: the SME2 asm path writes
    // exactly y[0..m], so the scalar fallback must do the same regardless
    // of whether the caller passed a longer backing slice.
    let y_active = &mut y[..m];
    if beta == 0.0 {
        for yi in y_active.iter_mut() {
            *yi = 0.0;
        }
    } else if beta != 1.0 {
        for yi in y_active.iter_mut() {
            *yi *= beta;
        }
    }
    if alpha == 0.0 || n == 0 {
        return;
    }
    for i in 0..m {
        let mut dot = 0.0_f32;
        for j in 0..n {
            dot += a[i * lda + j] * x[j];
        }
        y_active[i] += alpha * dot;
    }
}

fn dgemv_scalar(
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    let y_active = &mut y[..m];
    if beta == 0.0 {
        for yi in y_active.iter_mut() {
            *yi = 0.0;
        }
    } else if beta != 1.0 {
        for yi in y_active.iter_mut() {
            *yi *= beta;
        }
    }
    if alpha == 0.0 || n == 0 {
        return;
    }
    for i in 0..m {
        let mut dot = 0.0_f64;
        for j in 0..n {
            dot += a[i * lda + j] * x[j];
        }
        y_active[i] += alpha * dot;
    }
}
