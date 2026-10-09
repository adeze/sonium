//! Checked argument validation for backend entry points.
//!
//! Moved verbatim from `backend/mod.rs`.

use super::Backend;
use super::DeviceBufferLike;
use super::ReductionMode;
use super::capabilities::{
    CapabilityState, PrecisionCastCapabilities, ReductionCapabilities, ScalarCapabilities,
};
use super::views::Transpose;
use crate::bem::{BemRegularPairIn, BemRegularPairInF64, BemRegularPairOut, BemRegularPairOutF64};
#[cfg(feature = "cpu")]
use crate::cpu::blas3::trsm::{TrsmSide, TrsmUplo};
use crate::error::{BackendError, BackendResult};
use crate::scalar::{BufferElement, DeviceScalar, LowerPrecision};
use num_traits::{Float, Zero};
use std::any::TypeId;

pub(crate) fn require_scalar_capability<T: DeviceScalar>(
    backend: &'static str,
    operation: &'static str,
    capability: ScalarCapabilities,
) -> BackendResult<()> {
    if capability.is_unsupported_for::<T>() {
        return Err(BackendError::Unsupported {
            backend,
            operation,
            scalar: std::any::type_name::<T>(),
        });
    }
    Ok(())
}

pub(crate) fn require_finite_scalar<T: DeviceScalar>(
    operation: &'static str,
    name: &'static str,
    value: T,
) -> BackendResult<()> {
    if !value.is_finite() {
        return Err(BackendError::OperationFailed {
            operation,
            reason: format!("{name} contains NaN or infinity"),
        });
    }
    Ok(())
}

pub(crate) fn validate_bem_regular_pair_inputs(
    pairs: &[BemRegularPairIn],
    wave_number: f32,
    harmonic_factor: f32,
) -> BackendResult<()> {
    if !wave_number.is_finite() || !harmonic_factor.is_finite() {
        return Err(BackendError::OperationFailed {
            operation: "bem_regular_pair_c32_batch",
            reason: "wave number and harmonic factor must be finite".into(),
        });
    }
    for (index, pair) in pairs.iter().enumerate() {
        let finite = pair
            .source_point
            .iter()
            .chain(pair.source_normal.iter())
            .chain(pair.v0.iter())
            .chain(pair.v1.iter())
            .chain(pair.v2.iter())
            .all(|value| value.is_finite());
        if !finite {
            return Err(BackendError::OperationFailed {
                operation: "bem_regular_pair_c32_batch",
                reason: format!("pair {index} contains NaN or infinity"),
            });
        }
        let edge1 = [
            pair.v1[0] - pair.v0[0],
            pair.v1[1] - pair.v0[1],
            pair.v1[2] - pair.v0[2],
        ];
        let edge2 = [
            pair.v2[0] - pair.v0[0],
            pair.v2[1] - pair.v0[1],
            pair.v2[2] - pair.v0[2],
        ];
        let cross = [
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0],
        ];
        let jacobian_squared = cross.iter().map(|value| value * value).sum::<f32>();
        let source_normal_squared = pair
            .source_normal
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        if !jacobian_squared.is_finite()
            || jacobian_squared.sqrt() < 1.0e-15
            || !source_normal_squared.is_finite()
            || source_normal_squared.sqrt() < 1.0e-15
        {
            return Err(BackendError::InvalidArgument {
                operation: "bem_regular_pair_c32_batch",
                reason: format!("pair {index} has degenerate triangle or source normal"),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_bem_regular_pair_outputs(
    outputs: &[BemRegularPairOut],
) -> BackendResult<()> {
    for (index, output) in outputs.iter().enumerate() {
        let finite = [output.g, output.h, output.ht, output.e]
            .iter()
            .flat_map(|value| [value.re, value.im])
            .all(|value| value.is_finite());
        if !finite {
            return Err(BackendError::OperationFailed {
                operation: "bem_regular_pair_c32_batch",
                reason: format!("pair {index} produced NaN or infinity"),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_bem_regular_pair_f64_inputs(
    pairs: &[BemRegularPairInF64],
    wave_number: f64,
    harmonic_factor: f64,
) -> BackendResult<()> {
    if !wave_number.is_finite() || !harmonic_factor.is_finite() {
        return Err(BackendError::OperationFailed {
            operation: "bem_regular_pair_c64_batch",
            reason: "wave number and harmonic factor must be finite".into(),
        });
    }
    for (index, pair) in pairs.iter().enumerate() {
        let finite = pair
            .source_point
            .iter()
            .chain(pair.source_normal.iter())
            .chain(pair.v0.iter())
            .chain(pair.v1.iter())
            .chain(pair.v2.iter())
            .all(|value| value.is_finite());
        if !finite {
            return Err(BackendError::OperationFailed {
                operation: "bem_regular_pair_c64_batch",
                reason: format!("pair {index} contains NaN or infinity"),
            });
        }
        let edge1 = [
            pair.v1[0] - pair.v0[0],
            pair.v1[1] - pair.v0[1],
            pair.v1[2] - pair.v0[2],
        ];
        let edge2 = [
            pair.v2[0] - pair.v0[0],
            pair.v2[1] - pair.v0[1],
            pair.v2[2] - pair.v0[2],
        ];
        let cross = [
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0],
        ];
        let jacobian_squared = cross.iter().map(|value| value * value).sum::<f64>();
        let source_normal_squared = pair
            .source_normal
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        if !jacobian_squared.is_finite()
            || jacobian_squared.sqrt() < 1.0e-15
            || !source_normal_squared.is_finite()
            || source_normal_squared.sqrt() < 1.0e-15
        {
            return Err(BackendError::InvalidArgument {
                operation: "bem_regular_pair_c64_batch",
                reason: format!("pair {index} has degenerate triangle or source normal"),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_bem_regular_pair_f64_outputs(
    outputs: &[BemRegularPairOutF64],
) -> BackendResult<()> {
    for (index, output) in outputs.iter().enumerate() {
        let finite = [output.g, output.h, output.ht, output.e]
            .iter()
            .flat_map(|value| [value.re, value.im])
            .all(|value| value.is_finite());
        if !finite {
            return Err(BackendError::OperationFailed {
                operation: "bem_regular_pair_c64_batch",
                reason: format!("pair {index} produced NaN or infinity"),
            });
        }
    }
    Ok(())
}

pub(crate) fn require_reduction_capability<T: DeviceScalar>(
    backend: &'static str,
    operation: &'static str,
    capabilities: ReductionCapabilities,
    mode: ReductionMode,
) -> BackendResult<()> {
    require_scalar_capability::<T>(backend, operation, capabilities.for_mode(mode))
}

pub(crate) fn require_precision_capability<Upper: LowerPrecision>(
    backend: &'static str,
    operation: &'static str,
    capabilities: PrecisionCastCapabilities,
    from_lower: bool,
) -> BackendResult<()> {
    let type_id = TypeId::of::<Upper>();
    let state = if type_id == TypeId::of::<f64>() {
        if from_lower {
            capabilities.f32_to_f64
        } else {
            capabilities.f64_to_f32
        }
    } else if type_id == TypeId::of::<num_complex::Complex64>() {
        if from_lower {
            capabilities.c32_to_c64
        } else {
            capabilities.c64_to_c32
        }
    } else {
        CapabilityState::Unsupported
    };
    if matches!(state, CapabilityState::Unsupported) {
        return Err(BackendError::Unsupported {
            backend,
            operation,
            scalar: std::any::type_name::<Upper>(),
        });
    }
    Ok(())
}

fn required_matrix_len(rows: usize, cols: usize, ld: usize) -> Option<usize> {
    if rows == 0 || cols == 0 {
        Some(0)
    } else {
        rows.checked_sub(1)?.checked_mul(ld)?.checked_add(cols)
    }
}

/// Return the minimum element count for `count` logical vectors stored with
/// `stride` elements between starts and `extent` elements used per vector.
///
/// This is the common padded-layout contract for CSR SpMM and similar batched
/// operations.  The final vector may end before the final stride, so callers
/// must not replace this with `count * stride`.
pub(crate) fn checked_strided_span(
    operation: &'static str,
    count: usize,
    stride: usize,
    extent: usize,
) -> BackendResult<usize> {
    if count == 0 {
        return Ok(0);
    }
    if stride < extent {
        return Err(BackendError::InvalidArgument {
            operation,
            reason: format!("stride {stride} is smaller than logical extent {extent}"),
        });
    }
    (count - 1)
        .checked_mul(stride)
        .and_then(|base| base.checked_add(extent))
        .ok_or_else(|| BackendError::InvalidArgument {
            operation,
            reason: "strided layout span overflows usize".into(),
        })
}

pub(crate) fn validate_gemv_args(
    op: Transpose,
    m: usize,
    n: usize,
    a_len: usize,
    lda: usize,
    x_len: usize,
    y_len: usize,
) -> BackendResult<()> {
    let (stored_rows, stored_cols) = match op {
        Transpose::NoTrans => (m, n),
        Transpose::Trans | Transpose::ConjTrans => (n, m),
    };
    if lda < stored_cols {
        return Err(BackendError::InvalidArgument {
            operation: "gemv",
            reason: format!("lda {lda} is smaller than stored column count {stored_cols}"),
        });
    }
    let required = required_matrix_len(stored_rows, stored_cols, lda).ok_or_else(|| {
        BackendError::InvalidArgument {
            operation: "gemv",
            reason: "matrix span overflows usize".into(),
        }
    })?;
    if a_len < required {
        return Err(BackendError::InvalidArgument {
            operation: "gemv",
            reason: format!("matrix buffer length {a_len} is too small"),
        });
    }
    if x_len != n || y_len != m {
        return Err(BackendError::InvalidArgument {
            operation: "gemv",
            reason: format!("expected x/y lengths {n}/{m}, got {x_len}/{y_len}"),
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_gemm_args(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
    c_len: usize,
    ldc: usize,
) -> BackendResult<()> {
    let (a_rows, a_cols) = match transa {
        Transpose::NoTrans => (m, k),
        Transpose::Trans | Transpose::ConjTrans => (k, m),
    };
    let (b_rows, b_cols) = match transb {
        Transpose::NoTrans => (k, n),
        Transpose::Trans | Transpose::ConjTrans => (n, k),
    };
    for (operation, rows, cols, len, ld) in [
        ("gemm(A)", a_rows, a_cols, a_len, lda),
        ("gemm(B)", b_rows, b_cols, b_len, ldb),
        ("gemm(C)", m, n, c_len, ldc),
    ] {
        if ld < cols {
            return Err(BackendError::InvalidArgument {
                operation: "gemm",
                reason: format!("{operation} leading dimension {ld} < columns {cols}"),
            });
        }
        let required =
            required_matrix_len(rows, cols, ld).ok_or_else(|| BackendError::InvalidArgument {
                operation: "gemm",
                reason: format!("{operation} matrix span overflows usize"),
            })?;
        if len < required {
            return Err(BackendError::InvalidArgument {
                operation: "gemm",
                reason: format!("{operation} buffer length {len} is too small"),
            });
        }
    }
    Ok(())
}

#[cfg(feature = "cpu")]
pub(crate) fn validate_trsm_args(
    side: TrsmSide,
    _uplo: TrsmUplo,
    m: usize,
    n: usize,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
) -> BackendResult<()> {
    // `_uplo` only selects which triangle is read; dimensions are set by
    // `side` (kept as a parameter so call sites read symmetrically with
    // the solve call itself).
    let adim = match side {
        TrsmSide::Left => m,
        TrsmSide::Right => n,
    };
    for (operation, rows, cols, len, ld) in [
        ("trsm(A)", adim, adim, a_len, lda),
        ("trsm(B)", m, n, b_len, ldb),
    ] {
        if ld < cols {
            return Err(BackendError::InvalidArgument {
                operation: "trsm",
                reason: format!("{operation} leading dimension {ld} < columns {cols}"),
            });
        }
        let required =
            required_matrix_len(rows, cols, ld).ok_or_else(|| BackendError::InvalidArgument {
                operation: "trsm",
                reason: format!("{operation} matrix span overflows usize"),
            })?;
        if len < required {
            return Err(BackendError::InvalidArgument {
                operation: "trsm",
                reason: format!("{operation} buffer length {len} is too small"),
            });
        }
    }
    Ok(())
}

pub(crate) fn try_zeroed_host<T: Copy + Default>(
    len: usize,
    operation: &'static str,
) -> BackendResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| BackendError::Allocation {
            reason: format!("{operation}: host staging allocation failed for {len} elements"),
        })?;
    values.resize(len, T::default());
    Ok(values)
}

/// Validate CSR structure and transpose it into CSR(Aᵀ) using stable source
/// row order within each destination column. The caller owns all allocations;
/// this helper only writes outputs after the complete input contract passes.
pub(crate) fn validate_host_csr_structure(
    m: usize,
    n: usize,
    row_ptr: &[i32],
    col_idx: &[i32],
) -> BackendResult<()> {
    let expected_row_ptr = m
        .checked_add(1)
        .ok_or_else(|| BackendError::InvalidArgument {
            operation: "csr_transpose",
            reason: "row-pointer length overflows m + 1".into(),
        })?;
    if row_ptr.len() != expected_row_ptr {
        return Err(BackendError::InvalidArgument {
            operation: "csr_transpose",
            reason: "row-pointer buffer does not match matrix dimensions".into(),
        });
    }
    if row_ptr.first().copied() != Some(0) {
        return Err(BackendError::InvalidArgument {
            operation: "csr_transpose",
            reason: "row pointer must start at zero".into(),
        });
    }
    let nnz = col_idx.len();
    for (row, window) in row_ptr.windows(2).enumerate() {
        if window[0] < 0
            || window[1] < window[0]
            || usize::try_from(window[1]).map_or(true, |end| end > nnz)
        {
            return Err(BackendError::InvalidArgument {
                operation: "csr_transpose",
                reason: format!("row pointer is invalid at row {row}"),
            });
        }
    }
    if usize::try_from(*row_ptr.last().unwrap_or(&-1)).ok() != Some(nnz) {
        return Err(BackendError::InvalidArgument {
            operation: "csr_transpose",
            reason: "row pointer terminal value does not equal nnz".into(),
        });
    }
    for (position, &column) in col_idx.iter().enumerate() {
        if column < 0 || usize::try_from(column).map_or(true, |column| column >= n) {
            return Err(BackendError::InvalidArgument {
                operation: "csr_transpose",
                reason: format!("column {column} at position {position} is out of range"),
            });
        }
    }
    Ok(())
}

/// Host reference for [`super::Backend::try_gs_color_sweep`]: SOR Gauss-Seidel over
/// one color subset. Mirrors the host `smooth_multicolor_gauss_seidel`
/// semantics (missing-diagonal default of 1, `|a_ii| <= 1e-15` skip).
/// Reused by the default staged implementation and by GPU fallbacks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gs_color_sweep_host<T: DeviceScalar>(
    row_ptr: &[i32],
    col_idx: &[i32],
    values: &[T],
    color_rows: &[i32],
    omega: T::Real,
    b: &[T],
    x: &mut [T],
) -> BackendResult<()> {
    const OP: &str = "gs_color_sweep";
    let n = x.len();
    if b.len() != n {
        return Err(BackendError::InvalidArgument {
            operation: OP,
            reason: format!("b length {} != x length {n}", b.len()),
        });
    }
    validate_host_csr_structure(n, n, row_ptr, col_idx).map_err(|_| {
        BackendError::InvalidArgument {
            operation: OP,
            reason: "CSR structure is invalid for the vector length".into(),
        }
    })?;
    if values.len() != col_idx.len() {
        return Err(BackendError::InvalidArgument {
            operation: OP,
            reason: format!(
                "values length {} != column-index length {}",
                values.len(),
                col_idx.len()
            ),
        });
    }
    let tol = <T::Real as num_traits::FromPrimitive>::from_f64(1e-15).unwrap_or_else(T::Real::zero);
    let one_minus_omega = <T::Real as num_traits::One>::one() - omega;
    for &ri in color_rows {
        let i = usize::try_from(ri).map_err(|_| BackendError::InvalidArgument {
            operation: OP,
            reason: format!("color row {ri} is negative"),
        })?;
        if i >= n {
            return Err(BackendError::InvalidArgument {
                operation: OP,
                reason: format!("color row {i} exceeds dimension {n}"),
            });
        }
        // Validated above: row_ptr has n+1 entries, is monotone, and its
        // terminal value equals nnz == col_idx.len() == values.len().
        let start = row_ptr[i] as usize;
        let end = row_ptr[i + 1] as usize;
        let mut sum = b[i];
        let mut diag = T::one();
        for k in start..end {
            let j = col_idx[k] as usize;
            if j == i {
                diag = values[k];
            } else {
                sum = sum - values[k] * x[j];
            }
        }
        if diag.magnitude() > tol {
            let new_val = sum / diag;
            x[i] = T::from_real(one_minus_omega) * x[i] + T::from_real(omega) * new_val;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn host_csr_transpose<T: DeviceScalar>(
    m: usize,
    n: usize,
    row_ptr: &[i32],
    col_idx: &[i32],
    values: &[T],
    out_row_ptr: &mut [i32],
    out_col_idx: &mut [i32],
    out_values: &mut [T],
) -> BackendResult<()> {
    let expected_out_row_ptr = n
        .checked_add(1)
        .ok_or_else(|| BackendError::InvalidArgument {
            operation: "csr_transpose",
            reason: "output row-pointer length overflows n + 1".into(),
        })?;
    if out_row_ptr.len() != expected_out_row_ptr {
        return Err(BackendError::InvalidArgument {
            operation: "csr_transpose",
            reason: "output row-pointer buffer does not match matrix dimensions".into(),
        });
    }
    if col_idx.len() != values.len()
        || out_col_idx.len() != col_idx.len()
        || out_values.len() != values.len()
    {
        return Err(BackendError::InvalidArgument {
            operation: "csr_transpose",
            reason: "CSR input/output nonzero buffers have inconsistent lengths".into(),
        });
    }
    validate_host_csr_structure(m, n, row_ptr, col_idx)?;
    out_row_ptr.fill(0);
    for &column in col_idx {
        let destination = usize::try_from(column).map_err(|_| BackendError::InvalidArgument {
            operation: "csr_transpose",
            reason: "column index is negative".into(),
        })?;
        out_row_ptr[destination + 1] =
            out_row_ptr[destination + 1].checked_add(1).ok_or_else(|| {
                BackendError::InvalidArgument {
                    operation: "csr_transpose",
                    reason: "transposed row-pointer count exceeds i32 range".into(),
                }
            })?;
    }
    for column in 0..n {
        out_row_ptr[column + 1] = out_row_ptr[column + 1]
            .checked_add(out_row_ptr[column])
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "csr_transpose",
                reason: "transposed row-pointer prefix sum exceeds i32 range".into(),
            })?;
    }
    let mut next = try_zeroed_host(n, "csr_transpose")?;
    next.copy_from_slice(&out_row_ptr[..n]);
    for row in 0..m {
        let start = usize::try_from(row_ptr[row]).map_err(|_| BackendError::InvalidArgument {
            operation: "csr_transpose",
            reason: format!("row pointer is negative at row {row}"),
        })?;
        let end = usize::try_from(row_ptr[row + 1]).map_err(|_| BackendError::InvalidArgument {
            operation: "csr_transpose",
            reason: format!("row pointer is negative at row {row}"),
        })?;
        for position in start..end {
            let column =
                usize::try_from(col_idx[position]).map_err(|_| BackendError::InvalidArgument {
                    operation: "csr_transpose",
                    reason: format!("column index is negative at position {position}"),
                })?;
            let destination =
                usize::try_from(next[column]).map_err(|_| BackendError::InvalidArgument {
                    operation: "csr_transpose",
                    reason: "transposed row-pointer cursor is negative".into(),
                })?;
            out_col_idx[destination] =
                i32::try_from(row).map_err(|_| BackendError::InvalidArgument {
                    operation: "csr_transpose",
                    reason: "row index exceeds i32 range".into(),
                })?;
            out_values[destination] = values[position];
            next[column] =
                next[column]
                    .checked_add(1)
                    .ok_or_else(|| BackendError::InvalidArgument {
                        operation: "csr_transpose",
                        reason: "transposed row-pointer cursor exceeds i32 range".into(),
                    })?;
        }
    }
    Ok(())
}

/// Collect checked results without allowing allocator growth to panic.
///
/// The iterator length is supplied by the validated operation shape. A
/// mismatch is treated as a typed contract failure rather than allowing an
/// unexpected extra item to trigger `Vec` reallocation.
pub(crate) fn try_collect_backend<T, I>(
    iter: I,
    capacity: usize,
    operation: &'static str,
) -> BackendResult<Vec<T>>
where
    I: IntoIterator<Item = BackendResult<T>>,
{
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| BackendError::Allocation {
            reason: format!("{operation}: result allocation failed for {capacity} elements"),
        })?;
    for item in iter {
        if values.len() == capacity {
            return Err(BackendError::OperationFailed {
                operation,
                reason: "checked result iterator exceeded its declared capacity".into(),
            });
        }
        values.push(item?);
    }
    if values.len() != capacity {
        return Err(BackendError::OperationFailed {
            operation,
            reason: format!(
                "checked result iterator returned {} elements, expected {capacity}",
                values.len()
            ),
        });
    }
    Ok(values)
}

#[cfg(feature = "cpu")]
pub(crate) fn validate_syrk_args(
    trans: Transpose,
    n: usize,
    k: usize,
    a_len: usize,
    lda: usize,
    c_len: usize,
    ldc: usize,
) -> BackendResult<()> {
    let (a_rows, a_cols) = match trans {
        Transpose::NoTrans => (n, k),
        Transpose::Trans | Transpose::ConjTrans => (k, n),
    };
    for (operation, rows, cols, len, ld) in [
        ("syrk(A)", a_rows, a_cols, a_len, lda),
        ("syrk(C)", n, n, c_len, ldc),
    ] {
        if ld < cols {
            return Err(BackendError::InvalidArgument {
                operation: "syrk",
                reason: format!("{operation} leading dimension {ld} < columns {cols}"),
            });
        }
        let required =
            required_matrix_len(rows, cols, ld).ok_or_else(|| BackendError::InvalidArgument {
                operation: "syrk",
                reason: format!("{operation} matrix span overflows usize"),
            })?;
        if len < required {
            return Err(BackendError::InvalidArgument {
                operation: "syrk",
                reason: format!("{operation} buffer length {len} is too small"),
            });
        }
    }
    Ok(())
}

/// Validate finite gather inputs before a native backend launch.
///
/// Native gather kernels cannot portably report a non-finite input through
/// their ordinary output buffer.  Checked CUDA/Metal paths therefore perform
/// this small setup-time admission check; legacy void methods retain their
/// historical behavior.
#[allow(dead_code)]
pub(crate) fn validate_gather4_finite_inputs<B: Backend + ?Sized, T: DeviceScalar>(
    backend: &B,
    values: &B::Buffer<T>,
    weights: &B::Buffer<T::Real>,
) -> BackendResult<()>
where
    T::Real: BufferElement,
{
    if values.len() > 0 {
        let mut values_host = try_zeroed_host(values.len(), "gather4_weighted")?;
        backend.try_copy_d2h(values, &mut values_host)?;
        if values_host.iter().any(|value| !value.is_finite()) {
            return Err(BackendError::OperationFailed {
                operation: "gather4_weighted",
                reason: "values contain NaN or infinity".into(),
            });
        }
    }
    if weights.len() > 0 {
        let mut weights_host = try_zeroed_host(weights.len(), "gather4_weighted")?;
        backend.try_copy_d2h(weights, &mut weights_host)?;
        if weights_host.iter().any(|weight| !weight.is_finite()) {
            return Err(BackendError::OperationFailed {
                operation: "gather4_weighted",
                reason: "weights contain NaN or infinity".into(),
            });
        }
    }
    Ok(())
}
