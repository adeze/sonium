//! Host fallback kernels shared by all backends.
//!
//! Moved verbatim from `backend/mod.rs`.

use super::Backend;
use super::DeviceBufferLike;
use super::validate::{try_zeroed_host, validate_gemm_args, validate_gemv_args};
#[cfg(feature = "cpu")]
use super::validate::{validate_syrk_args, validate_trsm_args};
use super::views::Transpose;
#[cfg(feature = "cpu")]
use crate::cpu::blas3::trsm::{TrsmSide, TrsmUplo};
use crate::error::{BackendError, BackendResult};
use crate::scalar::{BufferElement, DeviceScalar, LowerPrecision};
use num_traits::Float;

#[allow(clippy::too_many_arguments)]
pub(crate) fn host_gemv<B: Backend + ?Sized, T: DeviceScalar>(
    backend: &B,
    op: Transpose,
    m: usize,
    n: usize,
    alpha: T,
    a: &B::Buffer<T>,
    lda: usize,
    x: &B::Buffer<T>,
    beta: T,
    y: &mut B::Buffer<T>,
) -> BackendResult<()> {
    validate_gemv_args(op, m, n, a.len(), lda, x.len(), y.len())?;
    let mut a_host = try_zeroed_host(a.len(), "gemv")?;
    let mut x_host = try_zeroed_host(x.len(), "gemv")?;
    let mut y_host = try_zeroed_host(y.len(), "gemv")?;
    backend.try_copy_d2h(a, &mut a_host)?;
    backend.try_copy_d2h(x, &mut x_host)?;
    backend.try_copy_d2h(y, &mut y_host)?;
    for row in 0..m {
        let mut sum = T::zero();
        for col in 0..n {
            let stored = match op {
                Transpose::NoTrans => a_host[row * lda + col],
                Transpose::Trans | Transpose::ConjTrans => a_host[col * lda + row],
            };
            let term = if matches!(op, Transpose::ConjTrans) {
                stored.conj_mul(x_host[col])
            } else {
                stored * x_host[col]
            };
            sum = sum + term;
        }
        y_host[row] = alpha * sum + beta * y_host[row];
    }
    backend.try_copy_h2d(&y_host, y)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn host_gemm<B: Backend + ?Sized, T: DeviceScalar>(
    backend: &B,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: &B::Buffer<T>,
    lda: usize,
    b: &B::Buffer<T>,
    ldb: usize,
    beta: T,
    c: &mut B::Buffer<T>,
    ldc: usize,
) -> BackendResult<()> {
    validate_gemm_args(
        transa,
        transb,
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
    let mut a_host = try_zeroed_host(a.len(), "gemm")?;
    let mut b_host = try_zeroed_host(b.len(), "gemm")?;
    let mut c_host = try_zeroed_host(c.len(), "gemm")?;
    backend.try_copy_d2h(a, &mut a_host)?;
    backend.try_copy_d2h(b, &mut b_host)?;
    backend.try_copy_d2h(c, &mut c_host)?;
    for row in 0..m {
        for col in 0..n {
            let mut sum = T::zero();
            for inner in 0..k {
                let av = match transa {
                    Transpose::NoTrans => a_host[row * lda + inner],
                    Transpose::Trans | Transpose::ConjTrans => a_host[inner * lda + row],
                };
                let bv = match transb {
                    Transpose::NoTrans => b_host[inner * ldb + col],
                    Transpose::Trans | Transpose::ConjTrans => b_host[col * ldb + inner],
                };
                let av = if matches!(transa, Transpose::ConjTrans) {
                    av.conj_mul(T::one())
                } else {
                    av
                };
                let bv = if matches!(transb, Transpose::ConjTrans) {
                    bv.conj_mul(T::one())
                } else {
                    bv
                };
                sum = sum + av * bv;
            }
            c_host[row * ldc + col] = alpha * sum + beta * c_host[row * ldc + col];
        }
    }
    backend.try_copy_h2d(&c_host, c)?;
    Ok(())
}

#[cfg(feature = "cpu")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn host_trsm<B: Backend + ?Sized, T: DeviceScalar>(
    backend: &B,
    side: TrsmSide,
    uplo: TrsmUplo,
    trans: Transpose,
    unit: bool,
    m: usize,
    n: usize,
    alpha: T,
    a: &B::Buffer<T>,
    lda: usize,
    b: &mut B::Buffer<T>,
    ldb: usize,
) -> BackendResult<()> {
    validate_trsm_args(side, uplo, m, n, a.len(), lda, b.len(), ldb)?;
    let mut a_host = try_zeroed_host(a.len(), "trsm")?;
    let mut b_host = try_zeroed_host(b.len(), "trsm")?;
    backend.try_copy_d2h(a, &mut a_host)?;
    backend.try_copy_d2h(b, &mut b_host)?;
    host_trsm_compute(
        side,
        uplo,
        trans,
        unit,
        m,
        n,
        alpha,
        &a_host,
        lda,
        &mut b_host,
        ldb,
    );
    backend.try_copy_h2d(&b_host, b)?;
    Ok(())
}

/// Scalar triangular solve over plain host slices (the compute core of
/// [`host_trsm`): split out so unit tests can pin every variant without a
/// device backend.
#[cfg(feature = "cpu")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn host_trsm_compute<T: DeviceScalar>(
    side: TrsmSide,
    uplo: TrsmUplo,
    trans: Transpose,
    unit: bool,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
) {
    // Element of op(A); conjugation folds in through `conj_mul` (identity
    // for real types, matching the CPU kernel's `TrsmScalar::conj`).
    let ae = |r: usize, c: usize| -> T {
        let v = match trans {
            Transpose::NoTrans => a[r * lda + c],
            Transpose::Trans | Transpose::ConjTrans => a[c * lda + r],
        };
        if matches!(trans, Transpose::ConjTrans) {
            v.conj_mul(T::one())
        } else {
            v
        }
    };
    // Substitution order follows op(A), not the stored triangle: a
    // transposed lower factor solves backward, and vice versa. A left
    // solve advances with op(A)'s triangle (forward for lower); a right
    // solve advances against it (forward only for upper), because each
    // equation couples the unknown to the far side of the triangle.
    let eff_upper = match (uplo, trans) {
        (TrsmUplo::Upper, Transpose::NoTrans) => true,
        (TrsmUplo::Upper, _) => false,
        (TrsmUplo::Lower, Transpose::NoTrans) => false,
        (TrsmUplo::Lower, _) => true,
    };
    let forward = match side {
        TrsmSide::Left => !eff_upper,
        TrsmSide::Right => eff_upper,
    };
    for v in b.iter_mut() {
        *v = alpha * *v;
    }
    match side {
        TrsmSide::Left => {
            for j in 0..n {
                if forward {
                    for i in 0..m {
                        let mut s = b[i * ldb + j];
                        for p in 0..i {
                            s = s - ae(i, p) * b[p * ldb + j];
                        }
                        b[i * ldb + j] = if unit { s } else { s / ae(i, i) };
                    }
                } else {
                    for i in (0..m).rev() {
                        let mut s = b[i * ldb + j];
                        for p in (i + 1)..m {
                            s = s - ae(i, p) * b[p * ldb + j];
                        }
                        b[i * ldb + j] = if unit { s } else { s / ae(i, i) };
                    }
                }
            }
        }
        TrsmSide::Right => {
            for i in 0..m {
                if forward {
                    for j in 0..n {
                        let mut s = b[i * ldb + j];
                        for p in 0..j {
                            s = s - b[i * ldb + p] * ae(p, j);
                        }
                        b[i * ldb + j] = if unit { s } else { s / ae(j, j) };
                    }
                } else {
                    for j in (0..n).rev() {
                        let mut s = b[i * ldb + j];
                        for p in (j + 1)..n {
                            s = s - b[i * ldb + p] * ae(p, j);
                        }
                        b[i * ldb + j] = if unit { s } else { s / ae(j, j) };
                    }
                }
            }
        }
    }
}

#[cfg(feature = "cpu")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn host_syrk<B: Backend + ?Sized, T: DeviceScalar>(
    backend: &B,
    uplo: TrsmUplo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: T,
    a: &B::Buffer<T>,
    lda: usize,
    beta: T,
    c: &mut B::Buffer<T>,
    ldc: usize,
) -> BackendResult<()> {
    validate_syrk_args(trans, n, k, a.len(), lda, c.len(), ldc)?;
    let mut a_host = try_zeroed_host(a.len(), "syrk")?;
    let mut c_host = try_zeroed_host(c.len(), "syrk")?;
    backend.try_copy_d2h(a, &mut a_host)?;
    backend.try_copy_d2h(c, &mut c_host)?;
    host_syrk_compute(
        uplo,
        trans,
        n,
        k,
        alpha,
        &a_host,
        lda,
        beta,
        &mut c_host,
        ldc,
    );
    backend.try_copy_h2d(&c_host, c)?;
    Ok(())
}

/// Scalar symmetric rank-k update over plain host slices (the compute core
/// of [`host_syrk`): only the `uplo` triangle is read and written, matching
/// the CPU kernel. Complex `ConjTrans` conjugates the first factor (the
/// Hermitian update); real `ConjTrans` behaves as `Trans`.
#[cfg(feature = "cpu")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn host_syrk_compute<T: DeviceScalar>(
    uplo: TrsmUplo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) {
    for i in 0..n {
        for j in 0..n {
            let in_tri = match uplo {
                TrsmUplo::Lower => j <= i,
                TrsmUplo::Upper => j >= i,
            };
            if !in_tri {
                continue;
            }
            let mut sum = T::zero();
            for p in 0..k {
                let (f, g) = match trans {
                    Transpose::NoTrans => (a[i * lda + p], a[j * lda + p]),
                    Transpose::Trans => (a[p * lda + i], a[p * lda + j]),
                    Transpose::ConjTrans => (a[p * lda + i].conj_mul(T::one()), a[p * lda + j]),
                };
                // Real `ConjTrans` must behave as `Trans`: `conj_mul` is
                // the identity for real types, so no special case needed.
                sum = sum + f * g;
            }
            c[i * ldc + j] = alpha * sum + beta * c[i * ldc + j];
        }
    }
}

pub(crate) fn host_gather4<B: Backend + ?Sized, T: DeviceScalar>(
    backend: &B,
    values: &B::Buffer<T>,
    indices: &B::Buffer<i32>,
    weights: &B::Buffer<T::Real>,
    out: &mut B::Buffer<T>,
) -> BackendResult<()>
where
    T::Real: BufferElement,
{
    let expected = out
        .len()
        .checked_mul(4)
        .ok_or_else(|| BackendError::InvalidArgument {
            operation: "gather4_weighted",
            reason: "output length overflow".into(),
        })?;
    if indices.len() != expected || weights.len() != expected {
        return Err(BackendError::InvalidArgument {
            operation: "gather4_weighted",
            reason: format!("expected {expected} indices and weights"),
        });
    }
    let mut values_host = try_zeroed_host(values.len(), "gather4_weighted")?;
    let mut indices_host = try_zeroed_host(indices.len(), "gather4_weighted")?;
    let mut weights_host = try_zeroed_host(weights.len(), "gather4_weighted")?;
    let mut out_host = try_zeroed_host(out.len(), "gather4_weighted")?;
    backend.try_copy_d2h(values, &mut values_host)?;
    backend.try_copy_d2h(indices, &mut indices_host)?;
    backend.try_copy_d2h(weights, &mut weights_host)?;
    if values_host.iter().any(|value| !value.is_finite()) {
        return Err(BackendError::OperationFailed {
            operation: "gather4_weighted",
            reason: "values contain NaN or infinity".into(),
        });
    }
    if weights_host.iter().any(|weight| !weight.is_finite()) {
        return Err(BackendError::OperationFailed {
            operation: "gather4_weighted",
            reason: "weights contain NaN or infinity".into(),
        });
    }
    for (row, out_value) in out_host.iter_mut().enumerate() {
        let base = row * 4;
        let mut acc = T::zero();
        for lane in 0..4 {
            let index = usize::try_from(indices_host[base + lane]).map_err(|_| {
                BackendError::InvalidArgument {
                    operation: "gather4_weighted",
                    reason: format!("negative index at lane {}", base + lane),
                }
            })?;
            let value = values_host
                .get(index)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "gather4_weighted",
                    reason: format!("index {index} exceeds values length {}", values_host.len()),
                })?;
            acc = acc + *value * T::from_real(weights_host[base + lane]);
        }
        *out_value = acc;
    }
    backend.try_copy_h2d(&out_host, out)?;
    Ok(())
}

#[allow(dead_code)]
pub(crate) fn host_axpy<B: Backend + ?Sized, T: DeviceScalar>(
    backend: &B,
    alpha: T,
    x: &B::Buffer<T>,
    y: &mut B::Buffer<T>,
) -> BackendResult<()> {
    if x.len() != y.len() {
        return Err(BackendError::InvalidArgument {
            operation: "axpy",
            reason: format!("x length {} != y length {}", x.len(), y.len()),
        });
    }
    let mut x_host = try_zeroed_host(x.len(), "axpy")?;
    let mut y_host = try_zeroed_host(y.len(), "axpy")?;
    backend.try_copy_d2h(x, &mut x_host)?;
    backend.try_copy_d2h(y, &mut y_host)?;
    for (xv, yv) in x_host.into_iter().zip(y_host.iter_mut()) {
        *yv = alpha * xv + *yv;
    }
    backend.try_copy_h2d(&y_host, y)?;
    Ok(())
}

#[allow(dead_code)]
pub(crate) fn host_scale<B: Backend + ?Sized, T: DeviceScalar>(
    backend: &B,
    alpha: T,
    x: &mut B::Buffer<T>,
) -> BackendResult<()> {
    let mut host = try_zeroed_host(x.len(), "scale")?;
    backend.try_copy_d2h(x, &mut host)?;
    for value in &mut host {
        *value = alpha * *value;
    }
    backend.try_copy_h2d(&host, x)?;
    Ok(())
}

#[allow(dead_code)]
pub(crate) fn host_copy_h2d_cast<B: Backend + ?Sized, Upper: LowerPrecision>(
    backend: &B,
    host: &[Upper],
    dev: &mut B::Buffer<Upper::Lower>,
) -> BackendResult<()> {
    if host.len() != dev.len() {
        return Err(BackendError::InvalidArgument {
            operation: "copy_h2d_cast",
            reason: format!("host length {} != device length {}", host.len(), dev.len()),
        });
    }
    let mut values = try_zeroed_host(host.len(), "copy_h2d_cast")?;
    for (value, lower) in host.iter().copied().zip(values.iter_mut()) {
        *lower = Upper::cast_to_lower(value);
    }
    backend.try_copy_h2d(&values, dev)
}

#[allow(dead_code)]
pub(crate) fn host_cast_to_lower<B: Backend + ?Sized, Upper: LowerPrecision>(
    backend: &B,
    src: &B::Buffer<Upper>,
    dst: &mut B::Buffer<Upper::Lower>,
) -> BackendResult<()> {
    if src.len() != dst.len() {
        return Err(BackendError::InvalidArgument {
            operation: "cast_to_lower",
            reason: format!(
                "source length {} != destination length {}",
                src.len(),
                dst.len()
            ),
        });
    }
    let mut src_host = try_zeroed_host(src.len(), "cast_to_lower")?;
    backend.try_copy_d2h(src, &mut src_host)?;
    let mut dst_host = Vec::new();
    dst_host
        .try_reserve_exact(src_host.len())
        .map_err(|_| BackendError::Allocation {
            reason: format!(
                "cast_to_lower: host staging allocation failed for {} elements",
                src_host.len()
            ),
        })?;
    dst_host.extend(src_host.into_iter().map(Upper::cast_to_lower));
    backend.try_copy_h2d(&dst_host, dst)?;
    Ok(())
}

#[allow(dead_code)]
pub(crate) fn host_cast_from_lower<B: Backend + ?Sized, Upper: LowerPrecision>(
    backend: &B,
    src: &B::Buffer<Upper::Lower>,
    dst: &mut B::Buffer<Upper>,
) -> BackendResult<()> {
    if src.len() != dst.len() {
        return Err(BackendError::InvalidArgument {
            operation: "cast_from_lower",
            reason: format!(
                "source length {} != destination length {}",
                src.len(),
                dst.len()
            ),
        });
    }
    let mut src_host = try_zeroed_host(src.len(), "cast_from_lower")?;
    backend.try_copy_d2h(src, &mut src_host)?;
    let mut dst_host = Vec::new();
    dst_host
        .try_reserve_exact(src_host.len())
        .map_err(|_| BackendError::Allocation {
            reason: format!(
                "cast_from_lower: host staging allocation failed for {} elements",
                src_host.len()
            ),
        })?;
    dst_host.extend(src_host.into_iter().map(Upper::cast_from_lower));
    backend.try_copy_h2d(&dst_host, dst)?;
    Ok(())
}
