//! Scalar reference implementations for in-house BLAS-2 (`gemv`).
//!
//! All kernels are row-major:
//!
//! - `op(A)` is the matrix as observed by the caller; it has `m` rows and `n`
//!   columns. The *stored* matrix is `m × n` for `NoTrans`, `n × m` for
//!   `Trans`/`ConjTrans`. `lda` is the leading dimension of the stored matrix
//!   (number of columns in row-major), and must be `≥ stored_cols`.
//! - `x` has length `n`, `y` has length `m`.
//! - The operation is `y := alpha * op(A) * x + beta * y`.
//!
//! Multi-accumulator inner loops let LLVM auto-vectorise these on
//! AVX2/AVX-512 (and on AArch64 when SME2 is unavailable). For tuned
//! AArch64, [`crate::sme2`] provides the FMOPA-based real-valued kernels;
//! the dispatcher in [`super`] forwards to those when `op = NoTrans` and
//! the type is real.

#![allow(clippy::too_many_arguments)]

use crate::backend::Transpose;
use num_complex::{Complex32, Complex64};

// ----- f32 -----------------------------------------------------------------

pub fn gemv_f32(
    op: Transpose,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) {
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(y.len(), m);
    if m == 0 {
        return;
    }
    // `y := beta * y` first so the inner loop only adds.
    if beta == 0.0 {
        for yi in y.iter_mut() {
            *yi = 0.0;
        }
    } else if beta != 1.0 {
        for yi in y.iter_mut() {
            *yi *= beta;
        }
    }
    if alpha == 0.0 || n == 0 {
        return;
    }
    match op {
        Transpose::NoTrans => {
            debug_assert!(lda >= n);
            for i in 0..m {
                let row = &a[i * lda..i * lda + n];
                let mut s0 = 0.0_f32;
                let mut s1 = 0.0_f32;
                let mut s2 = 0.0_f32;
                let mut s3 = 0.0_f32;
                let mut j = 0;
                while j + 4 <= n {
                    s0 += row[j] * x[j];
                    s1 += row[j + 1] * x[j + 1];
                    s2 += row[j + 2] * x[j + 2];
                    s3 += row[j + 3] * x[j + 3];
                    j += 4;
                }
                let mut acc = (s0 + s1) + (s2 + s3);
                while j < n {
                    acc += row[j] * x[j];
                    j += 1;
                }
                y[i] += alpha * acc;
            }
        }
        Transpose::Trans | Transpose::ConjTrans => {
            // Conjugation is a no-op on f32; both cases are the same.
            debug_assert!(lda >= m);
            for j in 0..n {
                let xj = alpha * x[j];
                let row = &a[j * lda..j * lda + m];
                for (i, &aij) in row.iter().enumerate() {
                    y[i] += aij * xj;
                }
            }
        }
    }
}

// ----- f64 -----------------------------------------------------------------

pub fn gemv_f64(
    op: Transpose,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(y.len(), m);
    if m == 0 {
        return;
    }
    if beta == 0.0 {
        for yi in y.iter_mut() {
            *yi = 0.0;
        }
    } else if beta != 1.0 {
        for yi in y.iter_mut() {
            *yi *= beta;
        }
    }
    if alpha == 0.0 || n == 0 {
        return;
    }
    match op {
        Transpose::NoTrans => {
            debug_assert!(lda >= n);
            for i in 0..m {
                let row = &a[i * lda..i * lda + n];
                let mut s0 = 0.0_f64;
                let mut s1 = 0.0_f64;
                let mut s2 = 0.0_f64;
                let mut s3 = 0.0_f64;
                let mut j = 0;
                while j + 4 <= n {
                    s0 += row[j] * x[j];
                    s1 += row[j + 1] * x[j + 1];
                    s2 += row[j + 2] * x[j + 2];
                    s3 += row[j + 3] * x[j + 3];
                    j += 4;
                }
                let mut acc = (s0 + s1) + (s2 + s3);
                while j < n {
                    acc += row[j] * x[j];
                    j += 1;
                }
                y[i] += alpha * acc;
            }
        }
        Transpose::Trans | Transpose::ConjTrans => {
            debug_assert!(lda >= m);
            for j in 0..n {
                let xj = alpha * x[j];
                let row = &a[j * lda..j * lda + m];
                for (i, &aij) in row.iter().enumerate() {
                    y[i] += aij * xj;
                }
            }
        }
    }
}

// ----- Complex32 -----------------------------------------------------------

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
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(y.len(), m);
    if m == 0 {
        return;
    }
    if beta == Complex32::new(0.0, 0.0) {
        for yi in y.iter_mut() {
            *yi = Complex32::new(0.0, 0.0);
        }
    } else if beta != Complex32::new(1.0, 0.0) {
        for yi in y.iter_mut() {
            *yi *= beta;
        }
    }
    if alpha == Complex32::new(0.0, 0.0) || n == 0 {
        return;
    }
    match op {
        Transpose::NoTrans => {
            debug_assert!(lda >= n);
            for i in 0..m {
                let row = &a[i * lda..i * lda + n];
                let mut acc = Complex32::new(0.0, 0.0);
                for (j, &aij) in row.iter().enumerate() {
                    acc += aij * x[j];
                }
                y[i] += alpha * acc;
            }
        }
        Transpose::Trans => {
            debug_assert!(lda >= m);
            for j in 0..n {
                let xj = alpha * x[j];
                let row = &a[j * lda..j * lda + m];
                for (i, &aji) in row.iter().enumerate() {
                    y[i] += aji * xj;
                }
            }
        }
        Transpose::ConjTrans => {
            debug_assert!(lda >= m);
            for j in 0..n {
                let xj = alpha * x[j];
                let row = &a[j * lda..j * lda + m];
                for (i, &aji) in row.iter().enumerate() {
                    y[i] += aji.conj() * xj;
                }
            }
        }
    }
}

// ----- Complex64 -----------------------------------------------------------

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
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(y.len(), m);
    if m == 0 {
        return;
    }
    if beta == Complex64::new(0.0, 0.0) {
        for yi in y.iter_mut() {
            *yi = Complex64::new(0.0, 0.0);
        }
    } else if beta != Complex64::new(1.0, 0.0) {
        for yi in y.iter_mut() {
            *yi *= beta;
        }
    }
    if alpha == Complex64::new(0.0, 0.0) || n == 0 {
        return;
    }
    match op {
        Transpose::NoTrans => {
            debug_assert!(lda >= n);
            for i in 0..m {
                let row = &a[i * lda..i * lda + n];
                let mut acc = Complex64::new(0.0, 0.0);
                for (j, &aij) in row.iter().enumerate() {
                    acc += aij * x[j];
                }
                y[i] += alpha * acc;
            }
        }
        Transpose::Trans => {
            debug_assert!(lda >= m);
            for j in 0..n {
                let xj = alpha * x[j];
                let row = &a[j * lda..j * lda + m];
                for (i, &aji) in row.iter().enumerate() {
                    y[i] += aji * xj;
                }
            }
        }
        Transpose::ConjTrans => {
            debug_assert!(lda >= m);
            for j in 0..n {
                let xj = alpha * x[j];
                let row = &a[j * lda..j * lda + m];
                for (i, &aji) in row.iter().enumerate() {
                    y[i] += aji.conj() * xj;
                }
            }
        }
    }
}
