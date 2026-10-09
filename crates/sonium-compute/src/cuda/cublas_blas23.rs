use super::*;
use crate::backend::Transpose;

/// Translate row-major [`Transpose`] to the matching cuBLAS column-major
/// op (used in the gemm swap-trick).
#[inline]
fn cu_op(t: Transpose) -> cublas_sys::cublasOperation_t {
    match t {
        Transpose::NoTrans => cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        Transpose::Trans => cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        Transpose::ConjTrans => cublas_sys::cublasOperation_t::CUBLAS_OP_C,
    }
}

/// Real-only gemv: op_c = N corresponds to row-major Trans / ConjTrans;
/// op_c = T corresponds to row-major NoTrans. (See module header.)
#[inline]
fn cu_op_gemv_real(t: Transpose) -> cublas_sys::cublasOperation_t {
    match t {
        Transpose::NoTrans => cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        Transpose::Trans | Transpose::ConjTrans => cublas_sys::cublasOperation_t::CUBLAS_OP_N,
    }
}

// --- gemv f32 -----------------------------------------------------------

pub unsafe fn gemv_f32(
    h: cublas_sys::cublasHandle_t,
    op: Transpose,
    m: i32,
    n: i32,
    alpha: f32,
    a: u64,
    lda: i32,
    x: u64,
    beta: f32,
    y: u64,
) -> cublas_sys::cublasStatus_t {
    // Col-major view of row-major A is n_stored × m_stored (where
    // m_stored × n_stored is the stored shape). cuBLAS rows/cols
    // refer to the col-major matrix. For NoTrans (stored m×n), the
    // col-major matrix is n×m. For Trans/ConjTrans (stored n×m), it's
    // m×n.
    let (cu_m, cu_n) = match op {
        Transpose::NoTrans => (n, m),
        Transpose::Trans | Transpose::ConjTrans => (m, n),
    };
    unsafe {
        cublas_sys::cublasSgemv_v2(
            h,
            cu_op_gemv_real(op),
            cu_m,
            cu_n,
            &alpha as *const f32,
            a as *const f32,
            lda,
            x as *const f32,
            1,
            &beta as *const f32,
            y as *mut f32,
            1,
        )
    }
}

// --- gemv f64 -----------------------------------------------------------

pub unsafe fn gemv_f64(
    h: cublas_sys::cublasHandle_t,
    op: Transpose,
    m: i32,
    n: i32,
    alpha: f64,
    a: u64,
    lda: i32,
    x: u64,
    beta: f64,
    y: u64,
) -> cublas_sys::cublasStatus_t {
    let (cu_m, cu_n) = match op {
        Transpose::NoTrans => (n, m),
        Transpose::Trans | Transpose::ConjTrans => (m, n),
    };
    unsafe {
        cublas_sys::cublasDgemv_v2(
            h,
            cu_op_gemv_real(op),
            cu_m,
            cu_n,
            &alpha as *const f64,
            a as *const f64,
            lda,
            x as *const f64,
            1,
            &beta as *const f64,
            y as *mut f64,
            1,
        )
    }
}

// --- gemv complex (N / T only; ConjTrans handled by caller via gemm) ----

pub unsafe fn gemv_c32_nt(
    h: cublas_sys::cublasHandle_t,
    op: Transpose,
    m: i32,
    n: i32,
    alpha: Complex32,
    a: u64,
    lda: i32,
    x: u64,
    beta: Complex32,
    y: u64,
) -> cublas_sys::cublasStatus_t {
    if matches!(op, Transpose::ConjTrans) {
        return cublas_sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    }
    let (cu_m, cu_n) = match op {
        Transpose::NoTrans => (n, m),
        Transpose::Trans | Transpose::ConjTrans => (m, n),
    };
    let alpha = cu_complex32(alpha);
    let beta = cu_complex32(beta);
    unsafe {
        cublas_sys::cublasCgemv_v2(
            h,
            cu_op_gemv_real(op),
            cu_m,
            cu_n,
            &alpha as *const cublas_sys::cuComplex,
            a as *const cublas_sys::cuComplex,
            lda,
            x as *const cublas_sys::cuComplex,
            1,
            &beta as *const cublas_sys::cuComplex,
            y as *mut cublas_sys::cuComplex,
            1,
        )
    }
}

pub unsafe fn gemv_c64_nt(
    h: cublas_sys::cublasHandle_t,
    op: Transpose,
    m: i32,
    n: i32,
    alpha: Complex64,
    a: u64,
    lda: i32,
    x: u64,
    beta: Complex64,
    y: u64,
) -> cublas_sys::cublasStatus_t {
    if matches!(op, Transpose::ConjTrans) {
        return cublas_sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    }
    let (cu_m, cu_n) = match op {
        Transpose::NoTrans => (n, m),
        Transpose::Trans | Transpose::ConjTrans => (m, n),
    };
    let alpha = cu_complex64(alpha);
    let beta = cu_complex64(beta);
    unsafe {
        cublas_sys::cublasZgemv_v2(
            h,
            cu_op_gemv_real(op),
            cu_m,
            cu_n,
            &alpha as *const cublas_sys::cuDoubleComplex,
            a as *const cublas_sys::cuDoubleComplex,
            lda,
            x as *const cublas_sys::cuDoubleComplex,
            1,
            &beta as *const cublas_sys::cuDoubleComplex,
            y as *mut cublas_sys::cuDoubleComplex,
            1,
        )
    }
}

// --- batched gemv -------------------------------------------------------

pub unsafe fn gemv_batched_f32(
    h: cublas_sys::cublasHandle_t,
    op: Transpose,
    m: i32,
    n: i32,
    alpha: f32,
    a: &[u64],
    lda: i32,
    x: &[u64],
    beta: f32,
    y: &[u64],
) -> cublas_sys::cublasStatus_t {
    let batch_count = match i32::try_from(a.len()) {
        Ok(count) => count,
        Err(_) => return cublas_sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE,
    };
    let (cu_m, cu_n) = match op {
        Transpose::NoTrans => (n, m),
        Transpose::Trans | Transpose::ConjTrans => (m, n),
    };
    unsafe {
        cublas_sys::cublasSgemvBatched(
            h,
            cu_op_gemv_real(op),
            cu_m,
            cu_n,
            &alpha as *const f32,
            a.as_ptr() as *const *const f32,
            lda,
            x.as_ptr() as *const *const f32,
            1,
            &beta as *const f32,
            y.as_ptr() as *const *mut f32,
            1,
            batch_count,
        )
    }
}

pub unsafe fn gemv_batched_f64(
    h: cublas_sys::cublasHandle_t,
    op: Transpose,
    m: i32,
    n: i32,
    alpha: f64,
    a: &[u64],
    lda: i32,
    x: &[u64],
    beta: f64,
    y: &[u64],
) -> cublas_sys::cublasStatus_t {
    let batch_count = match i32::try_from(a.len()) {
        Ok(count) => count,
        Err(_) => return cublas_sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE,
    };
    let (cu_m, cu_n) = match op {
        Transpose::NoTrans => (n, m),
        Transpose::Trans | Transpose::ConjTrans => (m, n),
    };
    unsafe {
        cublas_sys::cublasDgemvBatched(
            h,
            cu_op_gemv_real(op),
            cu_m,
            cu_n,
            &alpha as *const f64,
            a.as_ptr() as *const *const f64,
            lda,
            x.as_ptr() as *const *const f64,
            1,
            &beta as *const f64,
            y.as_ptr() as *const *mut f64,
            1,
            batch_count,
        )
    }
}

pub unsafe fn gemv_batched_c32_nt(
    h: cublas_sys::cublasHandle_t,
    op: Transpose,
    m: i32,
    n: i32,
    alpha: Complex32,
    a: &[u64],
    lda: i32,
    x: &[u64],
    beta: Complex32,
    y: &[u64],
) -> cublas_sys::cublasStatus_t {
    let batch_count = match i32::try_from(a.len()) {
        Ok(count) => count,
        Err(_) => return cublas_sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE,
    };
    if matches!(op, Transpose::ConjTrans) {
        return cublas_sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    }
    let (cu_m, cu_n) = match op {
        Transpose::NoTrans => (n, m),
        Transpose::Trans | Transpose::ConjTrans => (m, n),
    };
    let alpha = cu_complex32(alpha);
    let beta = cu_complex32(beta);
    unsafe {
        cublas_sys::cublasCgemvBatched(
            h,
            cu_op_gemv_real(op),
            cu_m,
            cu_n,
            &alpha as *const cublas_sys::cuComplex,
            a.as_ptr() as *const *const cublas_sys::cuComplex,
            lda,
            x.as_ptr() as *const *const cublas_sys::cuComplex,
            1,
            &beta as *const cublas_sys::cuComplex,
            y.as_ptr() as *const *mut cublas_sys::cuComplex,
            1,
            batch_count,
        )
    }
}

pub unsafe fn gemv_batched_c64_nt(
    h: cublas_sys::cublasHandle_t,
    op: Transpose,
    m: i32,
    n: i32,
    alpha: Complex64,
    a: &[u64],
    lda: i32,
    x: &[u64],
    beta: Complex64,
    y: &[u64],
) -> cublas_sys::cublasStatus_t {
    let batch_count = match i32::try_from(a.len()) {
        Ok(count) => count,
        Err(_) => return cublas_sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE,
    };
    if matches!(op, Transpose::ConjTrans) {
        return cublas_sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    }
    let (cu_m, cu_n) = match op {
        Transpose::NoTrans => (n, m),
        Transpose::Trans | Transpose::ConjTrans => (m, n),
    };
    let alpha = cu_complex64(alpha);
    let beta = cu_complex64(beta);
    unsafe {
        cublas_sys::cublasZgemvBatched(
            h,
            cu_op_gemv_real(op),
            cu_m,
            cu_n,
            &alpha as *const cublas_sys::cuDoubleComplex,
            a.as_ptr() as *const *const cublas_sys::cuDoubleComplex,
            lda,
            x.as_ptr() as *const *const cublas_sys::cuDoubleComplex,
            1,
            &beta as *const cublas_sys::cuDoubleComplex,
            y.as_ptr() as *const *mut cublas_sys::cuDoubleComplex,
            1,
            batch_count,
        )
    }
}

// --- gemm ---------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_f32(
    h: cublas_sys::cublasHandle_t,
    transa: Transpose,
    transb: Transpose,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: u64,
    lda: i32,
    b: u64,
    ldb: i32,
    beta: f32,
    c: u64,
    ldc: i32,
) -> cublas_sys::cublasStatus_t {
    // Row-major swap trick: call cuBLAS as gemm(transb, transa, n, m, k,
    // B, ldb, A, lda, C, ldc).
    unsafe {
        cublas_sys::cublasSgemm_v2(
            h,
            cu_op(transb),
            cu_op(transa),
            n,
            m,
            k,
            &alpha as *const f32,
            b as *const f32,
            ldb,
            a as *const f32,
            lda,
            &beta as *const f32,
            c as *mut f32,
            ldc,
        )
    }
}

/// Row-major strided-batched GEMM for the f32
/// [`try_batched_local_contract`](Backend::try_batched_local_contract)
/// fast path: per-batch `C = A@B` via the swap trick
/// (`C^T = B^T@A^T`, both `NoTrans`). Strides are in elements.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_strided_batched_f32(
    h: cublas_sys::cublasHandle_t,
    batches: i32,
    rows: i32,
    inner: i32,
    cols: i32,
    a: u64,
    b: u64,
    c: u64,
) -> cublas_sys::cublasStatus_t {
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let stride_a = i64::from(rows) * i64::from(inner);
    let stride_b = i64::from(inner) * i64::from(cols);
    let stride_c = i64::from(rows) * i64::from(cols);
    unsafe {
        cublas_sys::cublasSgemmStridedBatched(
            h,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            cols,
            rows,
            inner,
            &alpha as *const f32,
            b as *const f32,
            cols,
            stride_b,
            a as *const f32,
            inner,
            stride_a,
            &beta as *const f32,
            c as *mut f32,
            cols,
            stride_c,
            batches,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_f64(
    h: cublas_sys::cublasHandle_t,
    transa: Transpose,
    transb: Transpose,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a: u64,
    lda: i32,
    b: u64,
    ldb: i32,
    beta: f64,
    c: u64,
    ldc: i32,
) -> cublas_sys::cublasStatus_t {
    unsafe {
        cublas_sys::cublasDgemm_v2(
            h,
            cu_op(transb),
            cu_op(transa),
            n,
            m,
            k,
            &alpha as *const f64,
            b as *const f64,
            ldb,
            a as *const f64,
            lda,
            &beta as *const f64,
            c as *mut f64,
            ldc,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_c32(
    h: cublas_sys::cublasHandle_t,
    transa: Transpose,
    transb: Transpose,
    m: i32,
    n: i32,
    k: i32,
    alpha: Complex32,
    a: u64,
    lda: i32,
    b: u64,
    ldb: i32,
    beta: Complex32,
    c: u64,
    ldc: i32,
) -> cublas_sys::cublasStatus_t {
    let alpha = cu_complex32(alpha);
    let beta = cu_complex32(beta);
    unsafe {
        cublas_sys::cublasCgemm_v2(
            h,
            cu_op(transb),
            cu_op(transa),
            n,
            m,
            k,
            &alpha as *const cublas_sys::cuComplex,
            b as *const cublas_sys::cuComplex,
            ldb,
            a as *const cublas_sys::cuComplex,
            lda,
            &beta as *const cublas_sys::cuComplex,
            c as *mut cublas_sys::cuComplex,
            ldc,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_c64(
    h: cublas_sys::cublasHandle_t,
    transa: Transpose,
    transb: Transpose,
    m: i32,
    n: i32,
    k: i32,
    alpha: Complex64,
    a: u64,
    lda: i32,
    b: u64,
    ldb: i32,
    beta: Complex64,
    c: u64,
    ldc: i32,
) -> cublas_sys::cublasStatus_t {
    let alpha = cu_complex64(alpha);
    let beta = cu_complex64(beta);
    unsafe {
        cublas_sys::cublasZgemm_v2(
            h,
            cu_op(transb),
            cu_op(transa),
            n,
            m,
            k,
            &alpha as *const cublas_sys::cuDoubleComplex,
            b as *const cublas_sys::cuDoubleComplex,
            ldb,
            a as *const cublas_sys::cuDoubleComplex,
            lda,
            &beta as *const cublas_sys::cuDoubleComplex,
            c as *mut cublas_sys::cuDoubleComplex,
            ldc,
        )
    }
}

// --- trsm ---------------------------------------------------------------
//
// Row-major swap trick (same shape as the GEMM one: operands exchange
// roles, ops pass through). A row-major left solve `op(A) * X = B` is
// the column-major right solve `X^T * op(A^T) = B^T` over the same
// memory with `op(A^T) = op(A)^T`, so the cuBLAS call flips side
// (L<->R) and triangle (L<->U, the stored triangle transposes), swaps
// `m`/`n`, and keeps the transpose op unchanged. Same for right solves
// by symmetry. `lda`/`ldb` are unchanged: leading dimensions coincide
// between the two views.

/// Row-major [`TrsmSide`] to the matching column-major cuBLAS side.
#[inline]
fn cu_side_trsm(s: TrsmSide) -> cublas_sys::cublasSideMode_t {
    match s {
        TrsmSide::Left => cublas_sys::cublasSideMode_t::CUBLAS_SIDE_RIGHT,
        TrsmSide::Right => cublas_sys::cublasSideMode_t::CUBLAS_SIDE_LEFT,
    }
}

/// Row-major [`TrsmUplo`] to the matching column-major cuBLAS triangle.
#[inline]
fn cu_uplo_trsm(u: TrsmUplo) -> cublas_sys::cublasFillMode_t {
    match u {
        TrsmUplo::Lower => cublas_sys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
        TrsmUplo::Upper => cublas_sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
    }
}

/// Row-major [`Transpose`] to the matching column-major cuBLAS op:
/// identity — `op(A^T) = op(A)^T` elementwise, so the op passes through
/// (like the GEMM swap trick, which reorders operands but not ops).
#[inline]
fn cu_trans_trsm(t: Transpose) -> cublas_sys::cublasOperation_t {
    cu_op(t)
}

/// Unit-diagonal flag to the cuBLAS diagonal type.
#[inline]
fn cu_diag_trsm(unit: bool) -> cublas_sys::cublasDiagType_t {
    if unit {
        cublas_sys::cublasDiagType_t::CUBLAS_DIAG_UNIT
    } else {
        cublas_sys::cublasDiagType_t::CUBLAS_DIAG_NON_UNIT
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn trsm_f32(
    h: cublas_sys::cublasHandle_t,
    side: TrsmSide,
    uplo: TrsmUplo,
    trans: Transpose,
    unit: bool,
    m: i32,
    n: i32,
    alpha: f32,
    a: u64,
    lda: i32,
    b: u64,
    ldb: i32,
) -> cublas_sys::cublasStatus_t {
    unsafe {
        cublas_sys::cublasStrsm_v2(
            h,
            cu_side_trsm(side),
            cu_uplo_trsm(uplo),
            cu_trans_trsm(trans),
            cu_diag_trsm(unit),
            n,
            m,
            &alpha as *const f32,
            a as *const f32,
            lda,
            b as *mut f32,
            ldb,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn trsm_f64(
    h: cublas_sys::cublasHandle_t,
    side: TrsmSide,
    uplo: TrsmUplo,
    trans: Transpose,
    unit: bool,
    m: i32,
    n: i32,
    alpha: f64,
    a: u64,
    lda: i32,
    b: u64,
    ldb: i32,
) -> cublas_sys::cublasStatus_t {
    unsafe {
        cublas_sys::cublasDtrsm_v2(
            h,
            cu_side_trsm(side),
            cu_uplo_trsm(uplo),
            cu_trans_trsm(trans),
            cu_diag_trsm(unit),
            n,
            m,
            &alpha as *const f64,
            a as *const f64,
            lda,
            b as *mut f64,
            ldb,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn trsm_c32(
    h: cublas_sys::cublasHandle_t,
    side: TrsmSide,
    uplo: TrsmUplo,
    trans: Transpose,
    unit: bool,
    m: i32,
    n: i32,
    alpha: Complex32,
    a: u64,
    lda: i32,
    b: u64,
    ldb: i32,
) -> cublas_sys::cublasStatus_t {
    let alpha = cu_complex32(alpha);
    unsafe {
        cublas_sys::cublasCtrsm_v2(
            h,
            cu_side_trsm(side),
            cu_uplo_trsm(uplo),
            cu_trans_trsm(trans),
            cu_diag_trsm(unit),
            n,
            m,
            &alpha as *const cublas_sys::cuComplex,
            a as *const cublas_sys::cuComplex,
            lda,
            b as *mut cublas_sys::cuComplex,
            ldb,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn trsm_c64(
    h: cublas_sys::cublasHandle_t,
    side: TrsmSide,
    uplo: TrsmUplo,
    trans: Transpose,
    unit: bool,
    m: i32,
    n: i32,
    alpha: Complex64,
    a: u64,
    lda: i32,
    b: u64,
    ldb: i32,
) -> cublas_sys::cublasStatus_t {
    let alpha = cu_complex64(alpha);
    unsafe {
        cublas_sys::cublasZtrsm_v2(
            h,
            cu_side_trsm(side),
            cu_uplo_trsm(uplo),
            cu_trans_trsm(trans),
            cu_diag_trsm(unit),
            n,
            m,
            &alpha as *const cublas_sys::cuDoubleComplex,
            a as *const cublas_sys::cuDoubleComplex,
            lda,
            b as *mut cublas_sys::cuDoubleComplex,
            ldb,
        )
    }
}

// --- syrk ---------------------------------------------------------------
//
// Row-major symmetric update `C := alpha * op(A) * op(A)^T + beta * C`
// is the column-major update over the same memory with the transpose
// flipped (N<->T): the column-major view of row-major `A` is `A^T`, so
// `op_col(A^T) = op_row(A)` needs the opposite op. The triangle flips
// too (L<->U): cuBLAS triangle positions address the transposed view,
// so its lower triangle lands on our upper one and vice versa; the
// symmetric product's values coincide on either triangle, which is
// what makes the flip value-correct. `n`, `k`, and all leading
// dimensions are unchanged. Real `ConjTrans` behaves as `Trans`
// (normalized by the real wrappers); complex `ConjTrans` (the
// Hermitian update) falls back to the host core in `try_syrk`, never
// reaching these wrappers from checked dispatch.

/// Row-major [`TrsmUplo`] to cuBLAS for symmetric updates: flipped,
/// like the triangular solve.
#[inline]
fn cu_uplo_sym(u: TrsmUplo) -> cublas_sys::cublasFillMode_t {
    match u {
        TrsmUplo::Lower => cublas_sys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
        TrsmUplo::Upper => cublas_sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
    }
}

/// Row-major transpose to cuBLAS for symmetric updates: flipped. Real
/// `ConjTrans` folds to `Trans` (same flip target as `Trans`).
#[inline]
fn cu_trans_syrk_real(t: Transpose) -> cublas_sys::cublasOperation_t {
    match t {
        Transpose::NoTrans => cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        Transpose::Trans | Transpose::ConjTrans => cublas_sys::cublasOperation_t::CUBLAS_OP_N,
    }
}

/// Row-major transpose to cuBLAS for complex symmetric updates:
/// flipped; `ConjTrans` passes through (checked dispatch rejects the
/// Hermitian update before it can reach here).
#[inline]
fn cu_trans_syrk_complex(t: Transpose) -> cublas_sys::cublasOperation_t {
    match t {
        Transpose::NoTrans => cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        Transpose::Trans => cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        Transpose::ConjTrans => cublas_sys::cublasOperation_t::CUBLAS_OP_C,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn syrk_f32(
    h: cublas_sys::cublasHandle_t,
    uplo: TrsmUplo,
    trans: Transpose,
    n: i32,
    k: i32,
    alpha: f32,
    a: u64,
    lda: i32,
    beta: f32,
    c: u64,
    ldc: i32,
) -> cublas_sys::cublasStatus_t {
    unsafe {
        cublas_sys::cublasSsyrk_v2(
            h,
            cu_uplo_sym(uplo),
            cu_trans_syrk_real(trans),
            n,
            k,
            &alpha as *const f32,
            a as *const f32,
            lda,
            &beta as *const f32,
            c as *mut f32,
            ldc,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn syrk_f64(
    h: cublas_sys::cublasHandle_t,
    uplo: TrsmUplo,
    trans: Transpose,
    n: i32,
    k: i32,
    alpha: f64,
    a: u64,
    lda: i32,
    beta: f64,
    c: u64,
    ldc: i32,
) -> cublas_sys::cublasStatus_t {
    unsafe {
        cublas_sys::cublasDsyrk_v2(
            h,
            cu_uplo_sym(uplo),
            cu_trans_syrk_real(trans),
            n,
            k,
            &alpha as *const f64,
            a as *const f64,
            lda,
            &beta as *const f64,
            c as *mut f64,
            ldc,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn syrk_c32(
    h: cublas_sys::cublasHandle_t,
    uplo: TrsmUplo,
    trans: Transpose,
    n: i32,
    k: i32,
    alpha: Complex32,
    a: u64,
    lda: i32,
    beta: Complex32,
    c: u64,
    ldc: i32,
) -> cublas_sys::cublasStatus_t {
    let alpha = cu_complex32(alpha);
    let beta = cu_complex32(beta);
    unsafe {
        cublas_sys::cublasCsyrk_v2(
            h,
            cu_uplo_sym(uplo),
            cu_trans_syrk_complex(trans),
            n,
            k,
            &alpha as *const cublas_sys::cuComplex,
            a as *const cublas_sys::cuComplex,
            lda,
            &beta as *const cublas_sys::cuComplex,
            c as *mut cublas_sys::cuComplex,
            ldc,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn syrk_c64(
    h: cublas_sys::cublasHandle_t,
    uplo: TrsmUplo,
    trans: Transpose,
    n: i32,
    k: i32,
    alpha: Complex64,
    a: u64,
    lda: i32,
    beta: Complex64,
    c: u64,
    ldc: i32,
) -> cublas_sys::cublasStatus_t {
    let alpha = cu_complex64(alpha);
    let beta = cu_complex64(beta);
    unsafe {
        cublas_sys::cublasZsyrk_v2(
            h,
            cu_uplo_sym(uplo),
            cu_trans_syrk_complex(trans),
            n,
            k,
            &alpha as *const cublas_sys::cuDoubleComplex,
            a as *const cublas_sys::cuDoubleComplex,
            lda,
            &beta as *const cublas_sys::cuDoubleComplex,
            c as *mut cublas_sys::cuDoubleComplex,
            ldc,
        )
    }
}

/// Row-major Hermitian rank-k update via cuBLAS `Cherk`: the row-major
/// `C = alpha*A^H*A + beta*C` with A (k×n) is the col-major
/// `C' = alpha*A'*A'^H + beta*C'` with `A' = A^T` (n×k), i.e. herk with
/// `CUBLAS_OP_N` and a flipped fill mode. `alpha`/`beta` are real;
/// callers must route complex scalars through the host fallback (the
/// result is not Hermitian then, so no herk spelling applies).
#[allow(clippy::too_many_arguments)]
pub unsafe fn herk_c32(
    h: cublas_sys::cublasHandle_t,
    uplo: TrsmUplo,
    n: i32,
    k: i32,
    alpha: f32,
    a: u64,
    lda: i32,
    beta: f32,
    c: u64,
    ldc: i32,
) -> cublas_sys::cublasStatus_t {
    unsafe {
        cublas_sys::cublasCherk_v2(
            h,
            cu_uplo_sym(uplo),
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            n,
            k,
            &alpha as *const f32,
            a as *const cublas_sys::cuComplex,
            lda,
            &beta as *const f32,
            c as *mut cublas_sys::cuComplex,
            ldc,
        )
    }
}

/// Row-major Hermitian rank-k update via cuBLAS `Zherk`: same
/// col-major spelling as [`herk_c32`] in double precision.
#[allow(clippy::too_many_arguments)]
pub unsafe fn herk_c64(
    h: cublas_sys::cublasHandle_t,
    uplo: TrsmUplo,
    n: i32,
    k: i32,
    alpha: f64,
    a: u64,
    lda: i32,
    beta: f64,
    c: u64,
    ldc: i32,
) -> cublas_sys::cublasStatus_t {
    unsafe {
        cublas_sys::cublasZherk_v2(
            h,
            cu_uplo_sym(uplo),
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            n,
            k,
            &alpha as *const f64,
            a as *const cublas_sys::cuDoubleComplex,
            lda,
            &beta as *const f64,
            c as *mut cublas_sys::cuDoubleComplex,
            ldc,
        )
    }
}

// --- batched gemm -------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_batched_f32(
    h: cublas_sys::cublasHandle_t,
    transa: Transpose,
    transb: Transpose,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: &[u64],
    lda: i32,
    b: &[u64],
    ldb: i32,
    beta: f32,
    c: &[u64],
    ldc: i32,
) -> cublas_sys::cublasStatus_t {
    let batch_count = match i32::try_from(a.len()) {
        Ok(count) => count,
        Err(_) => return cublas_sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE,
    };
    unsafe {
        cublas_sys::cublasSgemmBatched(
            h,
            cu_op(transb),
            cu_op(transa),
            n,
            m,
            k,
            &alpha as *const f32,
            b.as_ptr() as *const *const f32,
            ldb,
            a.as_ptr() as *const *const f32,
            lda,
            &beta as *const f32,
            c.as_ptr() as *const *mut f32,
            ldc,
            batch_count,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_batched_f64(
    h: cublas_sys::cublasHandle_t,
    transa: Transpose,
    transb: Transpose,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a: &[u64],
    lda: i32,
    b: &[u64],
    ldb: i32,
    beta: f64,
    c: &[u64],
    ldc: i32,
) -> cublas_sys::cublasStatus_t {
    let batch_count = match i32::try_from(a.len()) {
        Ok(count) => count,
        Err(_) => return cublas_sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE,
    };
    unsafe {
        cublas_sys::cublasDgemmBatched(
            h,
            cu_op(transb),
            cu_op(transa),
            n,
            m,
            k,
            &alpha as *const f64,
            b.as_ptr() as *const *const f64,
            ldb,
            a.as_ptr() as *const *const f64,
            lda,
            &beta as *const f64,
            c.as_ptr() as *const *mut f64,
            ldc,
            batch_count,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_batched_c32(
    h: cublas_sys::cublasHandle_t,
    transa: Transpose,
    transb: Transpose,
    m: i32,
    n: i32,
    k: i32,
    alpha: Complex32,
    a: &[u64],
    lda: i32,
    b: &[u64],
    ldb: i32,
    beta: Complex32,
    c: &[u64],
    ldc: i32,
) -> cublas_sys::cublasStatus_t {
    let batch_count = match i32::try_from(a.len()) {
        Ok(count) => count,
        Err(_) => return cublas_sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE,
    };
    let alpha = cu_complex32(alpha);
    let beta = cu_complex32(beta);
    unsafe {
        cublas_sys::cublasCgemmBatched(
            h,
            cu_op(transb),
            cu_op(transa),
            n,
            m,
            k,
            &alpha as *const cublas_sys::cuComplex,
            b.as_ptr() as *const *const cublas_sys::cuComplex,
            ldb,
            a.as_ptr() as *const *const cublas_sys::cuComplex,
            lda,
            &beta as *const cublas_sys::cuComplex,
            c.as_ptr() as *const *mut cublas_sys::cuComplex,
            ldc,
            batch_count,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_batched_c64(
    h: cublas_sys::cublasHandle_t,
    transa: Transpose,
    transb: Transpose,
    m: i32,
    n: i32,
    k: i32,
    alpha: Complex64,
    a: &[u64],
    lda: i32,
    b: &[u64],
    ldb: i32,
    beta: Complex64,
    c: &[u64],
    ldc: i32,
) -> cublas_sys::cublasStatus_t {
    let batch_count = match i32::try_from(a.len()) {
        Ok(count) => count,
        Err(_) => return cublas_sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE,
    };
    let alpha = cu_complex64(alpha);
    let beta = cu_complex64(beta);
    unsafe {
        cublas_sys::cublasZgemmBatched(
            h,
            cu_op(transb),
            cu_op(transa),
            n,
            m,
            k,
            &alpha as *const cublas_sys::cuDoubleComplex,
            b.as_ptr() as *const *const cublas_sys::cuDoubleComplex,
            ldb,
            a.as_ptr() as *const *const cublas_sys::cuDoubleComplex,
            lda,
            &beta as *const cublas_sys::cuDoubleComplex,
            c.as_ptr() as *const *mut cublas_sys::cuDoubleComplex,
            ldc,
            batch_count,
        )
    }
}
