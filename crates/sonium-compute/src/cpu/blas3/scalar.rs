//! Scalar reference implementations for in-house BLAS-3 (`gemm`).
//!
//! All kernels are row-major.
//!
//! `C := alpha * op(A) * op(B) + beta * C` where:
//! - `op(A)` is `m × k`, `op(B)` is `k × n`, `C` is `m × n`.
//! - Stored `A` is `m × k` for `NoTrans`, `k × m` for `Trans`/`ConjTrans`;
//!   `lda` is the row stride of the stored matrix (≥ stored_cols).
//! - Same for `B`. `C` is always row-major `m × n` with stride `ldc ≥ n`.
//!
//! Inner kernel is a simple triple loop. The auto-vectoriser handles
//! AVX2/AVX-512 reasonably; tuned dispatchers in [`super`] forward real
//! `NoTrans` cases to the SME2 micro-kernels on AArch64.

#![allow(clippy::too_many_arguments)]

use crate::backend::Transpose;
use num_complex::{Complex32, Complex64};

#[inline]
fn aget_f32(a: &[f32], lda: usize, i: usize, p: usize, op: Transpose) -> f32 {
    match op {
        Transpose::NoTrans => a[i * lda + p],
        Transpose::Trans | Transpose::ConjTrans => a[p * lda + i],
    }
}

#[inline]
fn bget_f32(b: &[f32], ldb: usize, p: usize, j: usize, op: Transpose) -> f32 {
    match op {
        Transpose::NoTrans => b[p * ldb + j],
        Transpose::Trans | Transpose::ConjTrans => b[j * ldb + p],
    }
}

#[inline]
fn aget_f64(a: &[f64], lda: usize, i: usize, p: usize, op: Transpose) -> f64 {
    match op {
        Transpose::NoTrans => a[i * lda + p],
        Transpose::Trans | Transpose::ConjTrans => a[p * lda + i],
    }
}

#[inline]
fn bget_f64(b: &[f64], ldb: usize, p: usize, j: usize, op: Transpose) -> f64 {
    match op {
        Transpose::NoTrans => b[p * ldb + j],
        Transpose::Trans | Transpose::ConjTrans => b[j * ldb + p],
    }
}

#[inline]
fn aget_c32(a: &[Complex32], lda: usize, i: usize, p: usize, op: Transpose) -> Complex32 {
    match op {
        Transpose::NoTrans => a[i * lda + p],
        Transpose::Trans => a[p * lda + i],
        Transpose::ConjTrans => a[p * lda + i].conj(),
    }
}

#[inline]
fn bget_c32(b: &[Complex32], ldb: usize, p: usize, j: usize, op: Transpose) -> Complex32 {
    match op {
        Transpose::NoTrans => b[p * ldb + j],
        Transpose::Trans => b[j * ldb + p],
        Transpose::ConjTrans => b[j * ldb + p].conj(),
    }
}

#[inline]
fn aget_c64(a: &[Complex64], lda: usize, i: usize, p: usize, op: Transpose) -> Complex64 {
    match op {
        Transpose::NoTrans => a[i * lda + p],
        Transpose::Trans => a[p * lda + i],
        Transpose::ConjTrans => a[p * lda + i].conj(),
    }
}

#[inline]
fn bget_c64(b: &[Complex64], ldb: usize, p: usize, j: usize, op: Transpose) -> Complex64 {
    match op {
        Transpose::NoTrans => b[p * ldb + j],
        Transpose::Trans => b[j * ldb + p],
        Transpose::ConjTrans => b[j * ldb + p].conj(),
    }
}

pub fn gemm_f32(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    debug_assert!(ldc >= n);
    if m == 0 || n == 0 {
        return;
    }
    if beta == 0.0 {
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] = 0.0;
            }
        }
    } else if beta != 1.0 {
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] *= beta;
            }
        }
    }
    if alpha == 0.0 || k == 0 {
        return;
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += aget_f32(a, lda, i, p, transa) * bget_f32(b, ldb, p, j, transb);
            }
            c[i * ldc + j] += alpha * acc;
        }
    }
}

pub fn gemm_f64(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    debug_assert!(ldc >= n);
    if m == 0 || n == 0 {
        return;
    }
    if beta == 0.0 {
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] = 0.0;
            }
        }
    } else if beta != 1.0 {
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] *= beta;
            }
        }
    }
    if alpha == 0.0 || k == 0 {
        return;
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for p in 0..k {
                acc += aget_f64(a, lda, i, p, transa) * bget_f64(b, ldb, p, j, transb);
            }
            c[i * ldc + j] += alpha * acc;
        }
    }
}

pub fn gemm_c32(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    b: &[Complex32],
    ldb: usize,
    beta: Complex32,
    c: &mut [Complex32],
    ldc: usize,
) {
    debug_assert!(ldc >= n);
    if m == 0 || n == 0 {
        return;
    }
    let zero = Complex32::new(0.0, 0.0);
    let one = Complex32::new(1.0, 0.0);
    if beta == zero {
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] = zero;
            }
        }
    } else if beta != one {
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] *= beta;
            }
        }
    }
    if alpha == zero || k == 0 {
        return;
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = zero;
            for p in 0..k {
                acc += aget_c32(a, lda, i, p, transa) * bget_c32(b, ldb, p, j, transb);
            }
            c[i * ldc + j] += alpha * acc;
        }
    }
}

pub fn gemm_c64(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    b: &[Complex64],
    ldb: usize,
    beta: Complex64,
    c: &mut [Complex64],
    ldc: usize,
) {
    debug_assert!(ldc >= n);
    if m == 0 || n == 0 {
        return;
    }
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    if beta == zero {
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] = zero;
            }
        }
    } else if beta != one {
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] *= beta;
            }
        }
    }
    if alpha == zero || k == 0 {
        return;
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = zero;
            for p in 0..k {
                acc += aget_c64(a, lda, i, p, transa) * bget_c64(b, ldb, p, j, transb);
            }
            c[i * ldc + j] += alpha * acc;
        }
    }
}
