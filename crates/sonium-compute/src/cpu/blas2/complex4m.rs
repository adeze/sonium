//! Complex `gemv` via 4 real `gemv` calls.
//!
//! For row-major complex `A` (m × n), `x` (length n), `y` (length m),
//! `α, β ∈ ℂ`:
//!
//! ```text
//! Re(A·x) = Ar·xr - Ai·xi
//! Im(A·x) = Ar·xi + Ai·xr
//! y := α·(A·x) + β·y
//! ```
//!
//! We deinterleave `op(A)` and `x` into split real/imag arrays, run four
//! SME2-accelerated real `gemv` calls into two scratch vectors, then fold
//! `alpha` and `beta` element-wise into the result. Scratch arrays are
//! thread-local growth buffers, so repeated Krylov calls avoid allocator
//! traffic.

#![allow(clippy::too_many_arguments)]

use crate::backend::Transpose;
use crate::sme2;
use num_complex::{Complex32, Complex64};
use std::cell::RefCell;

thread_local! {
    static C32_AR: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static C32_AI: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static C32_XR: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static C32_XI: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static C32_R:  RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static C32_I:  RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static C64_AR: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static C64_AI: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static C64_XR: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static C64_XI: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static C64_R:  RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static C64_I:  RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Borrow a thread-local scratch buffer, growing to `need` on first call.
///
/// Reentrancy: the borrow is held while `f` runs, and a rayon worker
/// waiting in a nested `rayon::scope` may steal another GEMV onto the same
/// thread, re-entering this pool slot. `try_borrow_mut` detects that case
/// and falls back to a freshly allocated buffer owned by the nested call —
/// one allocation on the nested path instead of panicking with
/// "RefCell already borrowed". The hot non-nested path is unchanged.
fn with_buf<T, F, R>(cell: &'static std::thread::LocalKey<RefCell<Vec<T>>>, need: usize, f: F) -> R
where
    T: Copy + Default + 'static,
    F: FnOnce(&mut [T]) -> R,
{
    cell.with(|c| match c.try_borrow_mut() {
        Ok(mut buf) => {
            if buf.len() < need {
                buf.resize(need, T::default());
            }
            f(&mut buf[..need])
        }
        Err(_) => {
            let mut buf = vec![T::default(); need];
            f(&mut buf)
        }
    })
}

#[inline]
fn get_c32(z: &[Complex32], ld: usize, i: usize, j: usize, op: Transpose) -> Complex32 {
    match op {
        Transpose::NoTrans => z[i * ld + j],
        Transpose::Trans => z[j * ld + i],
        Transpose::ConjTrans => z[j * ld + i].conj(),
    }
}

#[inline]
fn get_c64(z: &[Complex64], ld: usize, i: usize, j: usize, op: Transpose) -> Complex64 {
    match op {
        Transpose::NoTrans => z[i * ld + j],
        Transpose::Trans => z[j * ld + i],
        Transpose::ConjTrans => z[j * ld + i].conj(),
    }
}

#[inline]
fn deinterleave_matrix_f32(
    z: &[Complex32],
    rows: usize,
    cols: usize,
    ld: usize,
    op: Transpose,
    re: &mut [f32],
    im: &mut [f32],
) {
    for i in 0..rows {
        for j in 0..cols {
            let zij = get_c32(z, ld, i, j, op);
            re[i * cols + j] = zij.re;
            im[i * cols + j] = zij.im;
        }
    }
}

#[inline]
fn deinterleave_matrix_f64(
    z: &[Complex64],
    rows: usize,
    cols: usize,
    ld: usize,
    op: Transpose,
    re: &mut [f64],
    im: &mut [f64],
) {
    for i in 0..rows {
        for j in 0..cols {
            let zij = get_c64(z, ld, i, j, op);
            re[i * cols + j] = zij.re;
            im[i * cols + j] = zij.im;
        }
    }
}

#[inline]
fn deinterleave_vector_f32(z: &[Complex32], re: &mut [f32], im: &mut [f32]) {
    for (i, zi) in z.iter().enumerate() {
        re[i] = zi.re;
        im[i] = zi.im;
    }
}

#[inline]
fn deinterleave_vector_f64(z: &[Complex64], re: &mut [f64], im: &mut [f64]) {
    for (i, zi) in z.iter().enumerate() {
        re[i] = zi.re;
        im[i] = zi.im;
    }
}

pub fn gemv_c32(
    op: Transpose,
    m: usize,
    n: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    x: &[Complex32],
    beta: Complex32,
    y: &mut [Complex32],
) {
    if m == 0 {
        return;
    }
    with_buf(&C32_AR, m * n, |ar| {
        with_buf(&C32_AI, m * n, |ai| {
            with_buf(&C32_XR, n, |xr| {
                with_buf(&C32_XI, n, |xi| {
                    with_buf(&C32_R, m, |r| {
                        with_buf(&C32_I, m, |imag| {
                            deinterleave_matrix_f32(a, m, n, lda, op, ar, ai);
                            deinterleave_vector_f32(x, xr, xi);

                            // R := Ar*xr - Ai*xi
                            sme2::gemv::sgemv(m, n, 1.0, ar, n, xr, 0.0, r);
                            sme2::gemv::sgemv(m, n, -1.0, ai, n, xi, 1.0, r);

                            // I := Ar*xi + Ai*xr
                            sme2::gemv::sgemv(m, n, 1.0, ar, n, xi, 0.0, imag);
                            sme2::gemv::sgemv(m, n, 1.0, ai, n, xr, 1.0, imag);

                            apply_c32_result(alpha, beta, r, imag, y);
                        });
                    });
                });
            });
        });
    });
}

pub fn gemv_c32_n(
    m: usize,
    n: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    x: &[Complex32],
    beta: Complex32,
    y: &mut [Complex32],
) {
    gemv_c32(Transpose::NoTrans, m, n, alpha, a, lda, x, beta, y);
}

pub fn gemv_c64(
    op: Transpose,
    m: usize,
    n: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    x: &[Complex64],
    beta: Complex64,
    y: &mut [Complex64],
) {
    if m == 0 {
        return;
    }
    with_buf(&C64_AR, m * n, |ar| {
        with_buf(&C64_AI, m * n, |ai| {
            with_buf(&C64_XR, n, |xr| {
                with_buf(&C64_XI, n, |xi| {
                    with_buf(&C64_R, m, |r| {
                        with_buf(&C64_I, m, |imag| {
                            deinterleave_matrix_f64(a, m, n, lda, op, ar, ai);
                            deinterleave_vector_f64(x, xr, xi);

                            sme2::gemv::dgemv(m, n, 1.0, ar, n, xr, 0.0, r);
                            sme2::gemv::dgemv(m, n, -1.0, ai, n, xi, 1.0, r);

                            sme2::gemv::dgemv(m, n, 1.0, ar, n, xi, 0.0, imag);
                            sme2::gemv::dgemv(m, n, 1.0, ai, n, xr, 1.0, imag);

                            apply_c64_result(alpha, beta, r, imag, y);
                        });
                    });
                });
            });
        });
    });
}

pub fn gemv_c64_n(
    m: usize,
    n: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    x: &[Complex64],
    beta: Complex64,
    y: &mut [Complex64],
) {
    gemv_c64(Transpose::NoTrans, m, n, alpha, a, lda, x, beta, y);
}

fn apply_c32_result(
    alpha: Complex32,
    beta: Complex32,
    r: &[f32],
    imag: &[f32],
    y: &mut [Complex32],
) {
    let ar_s = alpha.re;
    let ai_s = alpha.im;
    let br_s = beta.re;
    let bi_s = beta.im;
    let beta_zero = br_s == 0.0 && bi_s == 0.0;
    for i in 0..y.len() {
        let ri = r[i];
        let ii = imag[i];
        let ab_re = ar_s * ri - ai_s * ii;
        let ab_im = ar_s * ii + ai_s * ri;
        y[i] = if beta_zero {
            Complex32::new(ab_re, ab_im)
        } else {
            let prev = y[i];
            Complex32::new(
                ab_re + br_s * prev.re - bi_s * prev.im,
                ab_im + br_s * prev.im + bi_s * prev.re,
            )
        };
    }
}

fn apply_c64_result(
    alpha: Complex64,
    beta: Complex64,
    r: &[f64],
    imag: &[f64],
    y: &mut [Complex64],
) {
    let ar_s = alpha.re;
    let ai_s = alpha.im;
    let br_s = beta.re;
    let bi_s = beta.im;
    let beta_zero = br_s == 0.0 && bi_s == 0.0;
    for i in 0..y.len() {
        let ri = r[i];
        let ii = imag[i];
        let ab_re = ar_s * ri - ai_s * ii;
        let ab_im = ar_s * ii + ai_s * ri;
        y[i] = if beta_zero {
            Complex64::new(ab_re, ab_im)
        } else {
            let prev = y[i];
            Complex64::new(
                ab_re + br_s * prev.re - bi_s * prev.im,
                ab_im + br_s * prev.im + bi_s * prev.re,
            )
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::blas2::scalar;
    use approx::assert_relative_eq;

    #[test]
    fn gemv_c64_n_matches_reference() {
        let m = 7;
        let n = 5;
        let alpha = Complex64::new(0.4, -0.6);
        let beta = Complex64::new(0.3, 0.8);
        let a: Vec<Complex64> = (0..m * n)
            .map(|i| Complex64::new(0.1 * i as f64 - 0.3, 1.0 - 0.15 * i as f64))
            .collect();
        let x: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(0.7 + 0.21 * i as f64, -0.5 + 0.13 * i as f64))
            .collect();
        let y0: Vec<Complex64> = (0..m)
            .map(|i| Complex64::new(1.0 - 0.05 * i as f64, 0.4 + 0.07 * i as f64))
            .collect();
        let mut y = y0.clone();
        gemv_c64_n(m, n, alpha, &a, n, &x, beta, &mut y);
        // Reference loop.
        let mut want = vec![Complex64::new(0.0, 0.0); m];
        for i in 0..m {
            let mut s = Complex64::new(0.0, 0.0);
            for j in 0..n {
                s += a[i * n + j] * x[j];
            }
            want[i] = alpha * s + beta * y0[i];
        }
        for (a, b) in y.iter().zip(want.iter()) {
            assert_relative_eq!(a.re, b.re, max_relative = 1e-12);
            assert_relative_eq!(a.im, b.im, max_relative = 1e-12);
        }
    }

    #[test]
    fn gemv_c32_conjtrans_matches_scalar() {
        let m = 4;
        let n = 6;
        let alpha = Complex32::new(0.4, -0.6);
        let beta = Complex32::new(0.3, 0.8);
        let a: Vec<Complex32> = (0..n * m)
            .map(|i| Complex32::new(0.1 * i as f32 - 0.3, 1.0 - 0.15 * i as f32))
            .collect();
        let x: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(0.7 + 0.21 * i as f32, -0.5 + 0.13 * i as f32))
            .collect();
        let y0: Vec<Complex32> = (0..m)
            .map(|i| Complex32::new(1.0 - 0.05 * i as f32, 0.4 + 0.07 * i as f32))
            .collect();
        let mut got = y0.clone();
        let mut want = y0;
        gemv_c32(Transpose::ConjTrans, m, n, alpha, &a, m, &x, beta, &mut got);
        scalar::gemv_c32(
            Transpose::ConjTrans,
            m,
            n,
            alpha,
            &a,
            m,
            &x,
            beta,
            &mut want,
        );
        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(g.re, w.re, max_relative = 1e-4, epsilon = 1e-5);
            assert_relative_eq!(g.im, w.im, max_relative = 1e-4, epsilon = 1e-5);
        }
    }

    #[test]
    fn gemv_c64_trans_matches_scalar() {
        let m = 4;
        let n = 6;
        let alpha = Complex64::new(0.4, -0.6);
        let beta = Complex64::new(0.3, 0.8);
        let a: Vec<Complex64> = (0..n * m)
            .map(|i| Complex64::new(0.1 * i as f64 - 0.3, 1.0 - 0.15 * i as f64))
            .collect();
        let x: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(0.7 + 0.21 * i as f64, -0.5 + 0.13 * i as f64))
            .collect();
        let y0: Vec<Complex64> = (0..m)
            .map(|i| Complex64::new(1.0 - 0.05 * i as f64, 0.4 + 0.07 * i as f64))
            .collect();
        let mut got = y0.clone();
        let mut want = y0;
        gemv_c64(Transpose::Trans, m, n, alpha, &a, m, &x, beta, &mut got);
        scalar::gemv_c64(Transpose::Trans, m, n, alpha, &a, m, &x, beta, &mut want);
        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(g.re, w.re, max_relative = 1e-12);
            assert_relative_eq!(g.im, w.im, max_relative = 1e-12);
        }
    }
}
