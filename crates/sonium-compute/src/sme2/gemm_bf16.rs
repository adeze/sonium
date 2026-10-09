//! Mixed-precision GEMM: bf16 × bf16 → f32.
//!
//! Computes `C := alpha * A * B + beta * C` with bf16 A, bf16 B, and f32 C
//! (with f32 alpha/beta). Accumulation is in f32 — accuracy matches f32 GEMM
//! save for the bf16-precision loss on the inputs themselves.
//!
//! # Implementation
//!
//! The intended SME2 path was `BFMOPA` (widening bf16-pair → f32 accumulator).
//! Empirically Apple M4's SME unit emits `BFMOPA` but executes it as a
//! single-lane outer product — the K-second element of each pair is ignored
//! — so we cannot use it to get the theoretical 2× bf16 throughput. See
//! `sme2_kernels.S::sonium_bfgemm_panel_sme2` for the preserved-but-unused
//! kernel. If a future silicon revision fixes pair-widening, only the
//! dispatch in this module needs to change.
//!
//! Today we instead widen bf16 → f32 at pack time (a free shift-left-16
//! conversion) and route the widened buffers through the full f32 pipeline
//! in [`super::gemm::sgemm`]. That buys:
//!
//!   • The `gemm_bf16` API surface for bf16-input call sites.
//!   • Identical numerical behaviour to a single-pass f32 widening per inner-k.
//!   • Full f32 SME2 throughput: multi-tile FMOPA, grouped 4-Z loads, Kc/Nc
//!     blocking, and thread-local pack buffers — all reused, not duplicated.
//!
//! The bf16 → f32 conversion is done into a thread-local growth buffer once
//! per call (M·K + K·N elements). The conversion itself is O(M·K + K·N) but
//! the FMOPA work that follows is O(M·N·K), so for any non-degenerate K the
//! conversion is a small fraction of total cost.

#![allow(clippy::too_many_arguments)]

use super::detect::Sme2Caps;
use super::gemm::{sgemm, validate_gemm_bounds};
use crate::BackendResult;
use half::bf16;
use std::cell::RefCell;

thread_local! {
    static BFGEMM_A_F32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static BFGEMM_B_F32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

// Borrow a thread-local widening buffer, growing it to `need` on first use.
// The buffer is written by this call and read by `sgemm` while the closure
// runs; its address stays stable across sgemm's internal `rayon::scope`s.
//
// Reentrancy: a rayon worker waiting in one of those scopes may steal
// another `gemm_bf16` call onto the same thread. `try_borrow_mut` detects
// that case and falls back to a freshly allocated buffer owned by the nested
// call — one extra allocation on the nested path instead of panicking with
// "RefCell already borrowed". The hot non-nested path is unchanged.
fn with_widen_buf<F>(cell: &'static std::thread::LocalKey<RefCell<Vec<f32>>>, need: usize, f: F)
where
    F: FnOnce(&mut [f32]),
{
    cell.with(|c| match c.try_borrow_mut() {
        Ok(mut buf) => {
            if buf.len() < need {
                buf.resize(need, 0.0);
            }
            f(&mut buf[..need]);
        }
        Err(_) => {
            let mut buf = vec![0.0; need];
            f(&mut buf);
        }
    });
}

/// Checked `C := alpha * A * B + beta * C` with bf16 inputs and f32 output.
///
/// The checked entry point rejects malformed row-major spans before widening
/// or touching the destination. The compatibility [`gemm_bf16`] wrapper keeps
/// its historical panic-on-invalid-input behavior.
#[allow(clippy::too_many_arguments)]
pub fn try_gemm_bf16(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[bf16],
    lda: usize,
    b: &[bf16],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) -> BackendResult<()> {
    validate_gemm_bounds(
        "gemm_bf16",
        false,
        m,
        n,
        k,
        a.len(),
        lda,
        b.len(),
        ldb,
        c.len(),
        ldc,
    )?;
    gemm_bf16(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    Ok(())
}

/// `C := alpha * A * B + beta * C` with bf16 inputs and f32 output.
///
/// A is m×k row-major (stride `lda`), B is k×n row-major (stride `ldb`), C
/// is m×n row-major (stride `ldc`). Falls back to a scalar reference when
/// SME2 is unavailable.
pub fn gemm_bf16(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[bf16],
    lda: usize,
    b: &[bf16],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    // Bounds checks: required because the widening loops below would
    // otherwise read past A/B and we need parity with the f32 path.
    check_bf16_bounds(a.len(), m, k, lda, "gemm_bf16 A");
    check_bf16_bounds(b.len(), k, n, ldb, "gemm_bf16 B");
    check_bf16_bounds(c.len(), m, n, ldc, "gemm_bf16 C");

    if !Sme2Caps::detect().has_sme2 {
        return gemm_bf16_scalar(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    // Early-return on the degenerate cases BEFORE the bf16 → f32 widening:
    // there is no point spending O(M·K + K·N) widening A and B just to
    // discover sgemm would only run `C := beta * C`.
    if k == 0 || alpha == 0.0 {
        scale_c(m, n, beta, c, ldc);
        return;
    }
    let a_need = m * k;
    let b_need = k * n;
    with_widen_buf(&BFGEMM_A_F32, a_need, |a_f32| {
        with_widen_buf(&BFGEMM_B_F32, b_need, |b_f32| {
            widen_bf16_to_f32(a, lda, m, k, a_f32);
            widen_bf16_to_f32(b, ldb, k, n, b_f32);
            sgemm(
                m, n, k, alpha, a_f32, k, // widened stride is tight
                b_f32, n, beta, c, ldc, false,
            );
        });
    });
}

#[inline]
fn check_bf16_bounds(slice_len: usize, rows: usize, cols: usize, ld: usize, name: &str) {
    if rows == 0 || cols == 0 {
        return;
    }
    assert!(cols <= ld, "{name}: cols ({cols}) exceeds ld ({ld})");
    let last = (rows - 1).saturating_mul(ld).saturating_add(cols);
    assert!(
        last <= slice_len,
        "{name}: slice len {slice_len} < required {last}"
    );
}

// Widen a row-major bf16 sub-matrix into a tightly-strided f32 buffer.
fn widen_bf16_to_f32(src: &[bf16], src_stride: usize, rows: usize, cols: usize, dst: &mut [f32]) {
    for i in 0..rows {
        for j in 0..cols {
            dst[i * cols + j] = src[i * src_stride + j].to_f32();
        }
    }
}

fn scale_c(m: usize, n: usize, beta: f32, c: &mut [f32], ldc: usize) {
    if beta == 1.0 {
        return;
    }
    if beta == 0.0 {
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] = 0.0;
            }
        }
    } else {
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] *= beta;
            }
        }
    }
}

fn gemm_bf16_scalar(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[bf16],
    lda: usize,
    b: &[bf16],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    scale_c(m, n, beta, c, ldc);
    if alpha == 0.0 || k == 0 {
        return;
    }
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * lda + p].to_f32() * b[p * ldb + j].to_f32();
            }
            c[i * ldc + j] += alpha * sum;
        }
    }
}
