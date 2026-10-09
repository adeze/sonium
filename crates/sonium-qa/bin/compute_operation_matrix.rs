//! Operation/type/layout matrix for the checked compute contract.
//!
//! This is deliberately independent of vendor BLAS.  It exercises the public
//! checked backend boundaries for every `DeviceScalar` type, including GEMV/
//! GEMM, precision casts, gather, CSR SpMV/SpMM/transpose, reductions, and
//! local contractions.  It covers all transpose combinations, padded leading
//! dimensions, and the typed failure contract.  `SONIUM_BACKEND=metal` or
//! `SONIUM_BACKEND=cuda` selects a
//! feature-enabled native backend for the same independent fixture; an
//! unrequested backend is represented as a typed skip in the result.

use num_complex::{Complex32, Complex64};
use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};
use sonium_compute::{
    Backend, BackendError, BufferElement, CapabilityState, CpuBackend, DeviceBufferLike,
    DeviceScalar, LowerPrecision, ReductionMode, Transpose, ViewAliasing,
};
use sonium_qa::QaResult;
#[cfg(feature = "cuda")]
use sonium_qa::required_cuda_backend;
use std::time::Instant;

const CASE_ID: &str = "compute.operation-matrix.v1";
const MAX_RELATIVE_ERROR: f64 = 4.0e-5;

fn stored_shape(op: Transpose, m: usize, n: usize) -> (usize, usize) {
    match op {
        Transpose::NoTrans => (m, n),
        Transpose::Trans | Transpose::ConjTrans => (n, m),
    }
}

fn matrix_value<T: DeviceScalar>(
    data: &[T],
    leading_dimension: usize,
    op: Transpose,
    row: usize,
    column: usize,
) -> T {
    let (stored_row, stored_column) = match op {
        Transpose::NoTrans => (row, column),
        Transpose::Trans | Transpose::ConjTrans => (column, row),
    };
    let value = data[stored_row * leading_dimension + stored_column];
    if matches!(op, Transpose::ConjTrans) {
        value.conj_mul(T::one())
    } else {
        value
    }
}

fn relative_error<T: DeviceScalar>(actual: T, expected: T) -> f64
where
    T::Real: ToPrimitive,
{
    let numerator = actual
        .sub(expected)
        .norm_sqr()
        .to_f64()
        .unwrap_or(f64::INFINITY)
        .sqrt();
    let denominator = expected
        .norm_sqr()
        .to_f64()
        .unwrap_or(f64::INFINITY)
        .sqrt()
        .max(1.0);
    numerator / denominator
}

#[derive(Default)]
struct MatrixSummary {
    scalar_types: usize,
    layout_cases: usize,
    operation_cases: usize,
    max_relative_error: f64,
}

fn run_scalar_matrix<B, T, F>(
    backend: &B,
    make_scalar: F,
    label: &str,
) -> Result<MatrixSummary, String>
where
    B: Backend,
    T: DeviceScalar + PartialEq,
    T::Real: ToPrimitive,
    F: Fn(usize) -> T + Copy,
{
    let mut summary = MatrixSummary {
        scalar_types: 1,
        ..MatrixSummary::default()
    };
    for case in 0..18 {
        let m = 1 + (case * 5 % 8);
        let n = 1 + (case * 7 % 7);
        let k = 1 + (case * 3 % 6);
        let transa = match case % 3 {
            0 => Transpose::NoTrans,
            1 => Transpose::Trans,
            _ => Transpose::ConjTrans,
        };
        let transb = match (case / 3) % 3 {
            0 => Transpose::NoTrans,
            1 => Transpose::Trans,
            _ => Transpose::ConjTrans,
        };

        let (a_rows, a_columns) = stored_shape(transa, m, k);
        let (b_rows, b_columns) = stored_shape(transb, k, n);
        let lda = a_columns + case % 2;
        let ldb = b_columns + (case + 1) % 2;
        let ldc = n + case % 3;
        let a_host: Vec<_> = (0..a_rows * lda).map(make_scalar).collect();
        let b_host: Vec<_> = (0..b_rows * ldb)
            .map(|index| make_scalar(index + 2000))
            .collect();
        let c_initial: Vec<_> = (0..m * ldc)
            .map(|index| make_scalar(index + 4000))
            .collect();
        let alpha = make_scalar(6000 + case);
        let beta = make_scalar(7000 + case);

        let mut a = backend
            .raw_alloc::<T>(a_host.len())
            .map_err(|error| format!("{label} GEMM case {case}: allocate A: {error}"))?;
        let mut b = backend
            .raw_alloc::<T>(b_host.len())
            .map_err(|error| format!("{label} GEMM case {case}: allocate B: {error}"))?;
        let mut c = backend
            .raw_alloc::<T>(c_initial.len())
            .map_err(|error| format!("{label} GEMM case {case}: allocate C: {error}"))?;
        backend
            .try_copy_h2d(&a_host, &mut a)
            .map_err(|error| format!("{label} GEMM case {case}: copy A: {error}"))?;
        backend
            .try_copy_h2d(&b_host, &mut b)
            .map_err(|error| format!("{label} GEMM case {case}: copy B: {error}"))?;
        backend
            .try_copy_h2d(&c_initial, &mut c)
            .map_err(|error| format!("{label} GEMM case {case}: copy C: {error}"))?;
        backend
            .try_gemm(
                transa, transb, m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c, ldc,
            )
            .map_err(|error| format!("{label} GEMM case {case}: {error}"))?;
        let mut actual = vec![T::zero(); c_initial.len()];
        backend
            .try_copy_d2h(&c, &mut actual)
            .map_err(|error| format!("{label} GEMM case {case}: copy result: {error}"))?;
        let mut expected = c_initial;
        for row in 0..m {
            for column in 0..n {
                let product = (0..k).fold(T::zero(), |sum, inner| {
                    sum + matrix_value(&a_host, lda, transa, row, inner)
                        * matrix_value(&b_host, ldb, transb, inner, column)
                });
                expected[row * ldc + column] =
                    alpha * product + beta * expected[row * ldc + column];
                summary.max_relative_error = summary.max_relative_error.max(relative_error(
                    actual[row * ldc + column],
                    expected[row * ldc + column],
                ));
            }
        }
        summary.layout_cases += 1;
        summary.operation_cases += 1;

        let (stored_rows, stored_columns) = stored_shape(transa, m, n);
        let lda = stored_columns + case % 2;
        let a_host: Vec<_> = (0..stored_rows * lda).map(make_scalar).collect();
        let x_host: Vec<_> = (0..n).map(|index| make_scalar(index + 8000)).collect();
        let y_initial: Vec<_> = (0..m).map(|index| make_scalar(index + 9000)).collect();
        let alpha = make_scalar(10_000 + case);
        let beta = make_scalar(11_000 + case);
        let mut a = backend
            .raw_alloc::<T>(a_host.len())
            .map_err(|error| format!("{label} GEMV case {case}: allocate A: {error}"))?;
        let mut x = backend
            .raw_alloc::<T>(x_host.len())
            .map_err(|error| format!("{label} GEMV case {case}: allocate x: {error}"))?;
        let mut y = backend
            .raw_alloc::<T>(y_initial.len())
            .map_err(|error| format!("{label} GEMV case {case}: allocate y: {error}"))?;
        backend
            .try_copy_h2d(&a_host, &mut a)
            .map_err(|error| format!("{label} GEMV case {case}: copy A: {error}"))?;
        backend
            .try_copy_h2d(&x_host, &mut x)
            .map_err(|error| format!("{label} GEMV case {case}: copy x: {error}"))?;
        backend
            .try_copy_h2d(&y_initial, &mut y)
            .map_err(|error| format!("{label} GEMV case {case}: copy y: {error}"))?;
        backend
            .try_gemv(transa, m, n, alpha, &a, lda, &x, beta, &mut y)
            .map_err(|error| format!("{label} GEMV case {case}: {error}"))?;
        let mut actual = vec![T::zero(); m];
        backend
            .try_copy_d2h(&y, &mut actual)
            .map_err(|error| format!("{label} GEMV case {case}: copy result: {error}"))?;
        for row in 0..m {
            let product = (0..n).fold(T::zero(), |sum, column| {
                sum + matrix_value(&a_host, lda, transa, row, column) * x_host[column]
            });
            let expected = alpha * product + beta * y_initial[row];
            summary.max_relative_error = summary
                .max_relative_error
                .max(relative_error(actual[row], expected));
        }
        summary.layout_cases += 1;
        summary.operation_cases += 1;
    }
    Ok(summary)
}

/// Exercise the local tensor-contraction primitive through the checked
/// backend boundary for every public device scalar family.  The operation
/// matrix previously counted this capability without executing it, which
/// allowed a backend to advertise the contract while only GEMV/GEMM were
/// actually covered by the release case.
fn run_local_contract_matrix<B, T, F>(
    backend: &B,
    make_scalar: F,
    label: &str,
) -> Result<usize, String>
where
    B: Backend,
    T: DeviceScalar,
    T::Real: ToPrimitive,
    F: Fn(usize) -> T + Copy,
{
    let mut cases = 0;
    for case in 0..8 {
        let batches = 1 + case % 3;
        let rows = 1 + (case * 3 % 4);
        let inner = 1 + (case * 5 % 3);
        let cols = 1 + (case * 7 % 4);
        let lhs_len = batches * rows * inner;
        let rhs_len = batches * inner * cols;
        let output_len = batches * rows * cols;
        let lhs_host: Vec<_> = (0..lhs_len).map(|i| make_scalar(i + 12_000)).collect();
        let rhs_host: Vec<_> = (0..rhs_len).map(|i| make_scalar(i + 13_000)).collect();
        let mut lhs = backend
            .raw_alloc::<T>(lhs_len)
            .map_err(|error| format!("{label} local contract {case}: lhs allocation: {error}"))?;
        let mut rhs = backend
            .raw_alloc::<T>(rhs_len)
            .map_err(|error| format!("{label} local contract {case}: rhs allocation: {error}"))?;
        let mut output = backend.raw_alloc::<T>(output_len).map_err(|error| {
            format!("{label} local contract {case}: output allocation: {error}")
        })?;
        backend
            .try_copy_h2d(&lhs_host, &mut lhs)
            .map_err(|error| format!("{label} local contract {case}: copy lhs: {error}"))?;
        backend
            .try_copy_h2d(&rhs_host, &mut rhs)
            .map_err(|error| format!("{label} local contract {case}: copy rhs: {error}"))?;
        backend
            .try_batched_local_contract(&lhs, &rhs, &mut output, batches, rows, inner, cols)
            .map_err(|error| format!("{label} local contract {case}: dispatch: {error}"))?;
        let mut actual = vec![T::zero(); output_len];
        backend
            .try_copy_d2h(&output, &mut actual)
            .map_err(|error| format!("{label} local contract {case}: copy output: {error}"))?;
        for batch in 0..batches {
            for row in 0..rows {
                for col in 0..cols {
                    let expected = (0..inner).fold(T::zero(), |sum, k| {
                        sum + lhs_host[(batch * rows + row) * inner + k]
                            * rhs_host[(batch * inner + k) * cols + col]
                    });
                    let index = (batch * rows + row) * cols + col;
                    let error = relative_error(actual[index], expected);
                    if error > MAX_RELATIVE_ERROR {
                        return Err(format!(
                            "{label} local contract {case} element {index}: relative error {error} exceeds {MAX_RELATIVE_ERROR}"
                        ));
                    }
                }
            }
        }
        cases += 1;
    }
    Ok(cases)
}

/// Exercise both directions of every supported mixed-precision cast pair.
/// The reference is formed with the scalar trait itself, independently of the
/// backend's implementation and without assuming a particular fallback mode.
fn run_cast_matrix<B, Upper, F>(backend: &B, make_scalar: F, label: &str) -> Result<usize, String>
where
    B: Backend,
    Upper: LowerPrecision,
    Upper::Lower: DeviceScalar,
    Upper::Real: ToPrimitive,
    F: Fn(usize) -> Upper + Copy,
{
    let mut cases = 0;
    for case in 0..12 {
        let length = 1 + case * 3;
        let host: Vec<_> = (0..length)
            .map(|index| make_scalar(index + case * 1000))
            .collect();
        let mut source = backend
            .raw_alloc::<Upper>(length)
            .map_err(|error| format!("{label} cast case {case}: source allocation: {error}"))?;
        let mut lower = backend
            .raw_alloc::<Upper::Lower>(length)
            .map_err(|error| format!("{label} cast case {case}: lower allocation: {error}"))?;
        let mut round_trip = backend
            .raw_alloc::<Upper>(length)
            .map_err(|error| format!("{label} cast case {case}: output allocation: {error}"))?;
        backend
            .try_copy_h2d(&host, &mut source)
            .map_err(|error| format!("{label} cast case {case}: source upload: {error}"))?;
        backend
            .try_cast_to_lower(&source, &mut lower)
            .map_err(|error| format!("{label} cast case {case}: downcast: {error}"))?;
        backend
            .try_cast_from_lower(&lower, &mut round_trip)
            .map_err(|error| format!("{label} cast case {case}: upcast: {error}"))?;

        let mut actual = vec![Upper::zero(); length];
        backend
            .try_copy_d2h(&round_trip, &mut actual)
            .map_err(|error| format!("{label} cast case {case}: output download: {error}"))?;
        for (index, value) in actual.into_iter().enumerate() {
            let expected = Upper::cast_from_lower(Upper::cast_to_lower(host[index]));
            let error = relative_error(value, expected);
            if error > MAX_RELATIVE_ERROR {
                return Err(format!(
                    "{label} cast case {case} element {index}: relative error {error} exceeds {MAX_RELATIVE_ERROR}"
                ));
            }
        }
        cases += 1;
    }
    Ok(cases)
}

/// Exercise successful fixed-width gather cells for each device scalar.  The
/// fixture includes repeated and non-monotone indices so native kernels cannot
/// rely on a contiguous or unique access pattern.
fn run_gather_matrix<B, T>(backend: &B, label: &str) -> Result<usize, String>
where
    B: Backend,
    T: DeviceScalar,
    T::Real: BufferElement + FromPrimitive + ToPrimitive,
{
    let indices_host = [0_i32, 7, 3, 11, 5, 5, 1, 9, 10, 2, 10, 4];
    let raw_weights = [
        0.05_f64, 0.15, 0.25, 0.55, 0.4, 0.1, 0.2, 0.3, 0.7, 0.1, 0.05, 0.15,
    ];
    let weights_host: Vec<T::Real> = raw_weights
        .into_iter()
        .map(|value| {
            T::Real::from_f64(value)
                .ok_or_else(|| format!("{label} gather fixture cannot represent weight {value}"))
        })
        .collect::<Result<_, _>>()?;
    let mut cases = 0;
    for case in 0..3 {
        let values_host: Vec<_> = (0..12)
            .map(|index| {
                T::from_real(
                    T::Real::from_f64((index + case * 13) as f64 * 0.17)
                        .unwrap_or_else(|| T::Real::zero()),
                )
            })
            .collect();
        let mut values = backend
            .raw_alloc::<T>(values_host.len())
            .map_err(|error| format!("{label} gather case {case}: values allocation: {error}"))?;
        let mut indices = backend
            .raw_alloc::<i32>(indices_host.len())
            .map_err(|error| format!("{label} gather case {case}: indices allocation: {error}"))?;
        let mut weights = backend
            .raw_alloc::<T::Real>(weights_host.len())
            .map_err(|error| format!("{label} gather case {case}: weights allocation: {error}"))?;
        let mut output = backend
            .raw_alloc::<T>(indices_host.len() / 4)
            .map_err(|error| format!("{label} gather case {case}: output allocation: {error}"))?;
        backend
            .try_copy_h2d(&values_host, &mut values)
            .map_err(|error| format!("{label} gather case {case}: values upload: {error}"))?;
        backend
            .try_copy_h2d(&indices_host, &mut indices)
            .map_err(|error| format!("{label} gather case {case}: indices upload: {error}"))?;
        backend
            .try_copy_h2d(&weights_host, &mut weights)
            .map_err(|error| format!("{label} gather case {case}: weights upload: {error}"))?;
        backend
            .try_gather4_weighted(&values, &indices, &weights, &mut output)
            .map_err(|error| format!("{label} gather case {case}: dispatch: {error}"))?;
        let mut actual = vec![T::zero(); output.len()];
        backend
            .try_copy_d2h(&output, &mut actual)
            .map_err(|error| format!("{label} gather case {case}: output download: {error}"))?;
        for output_index in 0..actual.len() {
            let expected = (0..4).fold(T::zero(), |sum, lane| {
                let position = output_index * 4 + lane;
                sum + values_host[indices_host[position] as usize]
                    * T::from_real(weights_host[position])
            });
            let error = relative_error(actual[output_index], expected);
            if error > MAX_RELATIVE_ERROR {
                return Err(format!(
                    "{label} gather case {case} output {output_index}: relative error {error} exceeds {MAX_RELATIVE_ERROR}"
                ));
            }
        }
        cases += 1;
    }
    Ok(cases)
}

/// Exercise CSR SpMV, both checked SpMM layouts, and CSR(Aᵀ) construction
/// against an independent stable transpose reference.
fn run_sparse_matrix<B, T, F>(backend: &B, make_scalar: F, label: &str) -> Result<usize, String>
where
    B: Backend,
    T: DeviceScalar,
    T::Real: ToPrimitive,
    F: Fn(usize) -> T + Copy,
{
    let m = 4usize;
    let n = 5usize;
    let rhs = 3usize;
    let row_host = [0_i32, 2, 4, 5, 7];
    let col_host = [0_i32, 2, 1, 4, 3, 0, 4];
    let values_host: Vec<_> = (0..col_host.len())
        .map(|index| make_scalar(index + 30_000))
        .collect();
    let mut row = backend
        .raw_alloc::<i32>(row_host.len())
        .map_err(|error| format!("{label} sparse: row allocation: {error}"))?;
    let mut col = backend
        .raw_alloc::<i32>(col_host.len())
        .map_err(|error| format!("{label} sparse: column allocation: {error}"))?;
    let mut values = backend
        .raw_alloc::<T>(values_host.len())
        .map_err(|error| format!("{label} sparse: values allocation: {error}"))?;
    backend
        .try_copy_h2d(&row_host, &mut row)
        .map_err(|error| format!("{label} sparse: row upload: {error}"))?;
    backend
        .try_copy_h2d(&col_host, &mut col)
        .map_err(|error| format!("{label} sparse: column upload: {error}"))?;
    backend
        .try_copy_h2d(&values_host, &mut values)
        .map_err(|error| format!("{label} sparse: values upload: {error}"))?;
    let prebaked = backend
        .try_prebake_csr(m, n, &row, &col, &values)
        .map_err(|error| format!("{label} sparse: prebake: {error}"))?;

    let x_host: Vec<_> = (0..n).map(|index| make_scalar(index + 31_000)).collect();
    let mut x = backend
        .raw_alloc::<T>(x_host.len())
        .map_err(|error| format!("{label} sparse: SpMV x allocation: {error}"))?;
    let mut y = backend
        .raw_alloc::<T>(m)
        .map_err(|error| format!("{label} sparse: SpMV y allocation: {error}"))?;
    backend
        .try_copy_h2d(&x_host, &mut x)
        .map_err(|error| format!("{label} sparse: SpMV x upload: {error}"))?;
    backend
        .try_spmv_prebaked(&prebaked, &x, &mut y)
        .map_err(|error| format!("{label} sparse: SpMV: {error}"))?;
    let mut actual = vec![T::zero(); m];
    backend
        .try_copy_d2h(&y, &mut actual)
        .map_err(|error| format!("{label} sparse: SpMV download: {error}"))?;
    let expected_spmv: Vec<_> = (0..m)
        .map(|row_index| {
            (row_host[row_index] as usize..row_host[row_index + 1] as usize)
                .fold(T::zero(), |sum, position| {
                    sum + values_host[position] * x_host[col_host[position] as usize]
                })
        })
        .collect();
    for (index, (&value, &expected)) in actual.iter().zip(&expected_spmv).enumerate() {
        let error = relative_error(value, expected);
        if error > MAX_RELATIVE_ERROR {
            return Err(format!(
                "{label} sparse SpMV element {index}: relative error {error} exceeds {MAX_RELATIVE_ERROR}"
            ));
        }
    }

    let x_multi_host: Vec<_> = (0..n * rhs)
        .map(|index| make_scalar(index + 32_000))
        .collect();
    let mut x_multi = backend
        .raw_alloc::<T>(x_multi_host.len())
        .map_err(|error| format!("{label} sparse: SpMM x allocation: {error}"))?;
    let mut y_multi = backend
        .raw_alloc::<T>(m * rhs)
        .map_err(|error| format!("{label} sparse: SpMM y allocation: {error}"))?;
    backend
        .try_copy_h2d(&x_multi_host, &mut x_multi)
        .map_err(|error| format!("{label} sparse: SpMM x upload: {error}"))?;
    backend
        .try_spmm_prebaked_colmajor(&prebaked, rhs, &x_multi, &mut y_multi)
        .map_err(|error| format!("{label} sparse: column-major SpMM: {error}"))?;
    let mut actual_multi = vec![T::zero(); m * rhs];
    backend
        .try_copy_d2h(&y_multi, &mut actual_multi)
        .map_err(|error| format!("{label} sparse: column-major SpMM download: {error}"))?;
    for right_hand_side in 0..rhs {
        for row_index in 0..m {
            let expected = (row_host[row_index] as usize..row_host[row_index + 1] as usize).fold(
                T::zero(),
                |sum, position| {
                    sum + values_host[position]
                        * x_multi_host[right_hand_side * n + col_host[position] as usize]
                },
            );
            let index = right_hand_side * m + row_index;
            let error = relative_error(actual_multi[index], expected);
            if error > MAX_RELATIVE_ERROR {
                return Err(format!(
                    "{label} sparse column-major SpMM element {index}: relative error {error} exceeds {MAX_RELATIVE_ERROR}"
                ));
            }
        }
    }

    let ldx = n + 2;
    let ldy = m + 1;
    let mut x_padded_host = vec![T::zero(); ldx * rhs];
    for right_hand_side in 0..rhs {
        x_padded_host[right_hand_side * ldx..right_hand_side * ldx + n]
            .copy_from_slice(&x_multi_host[right_hand_side * n..(right_hand_side + 1) * n]);
    }
    let mut x_padded = backend
        .raw_alloc::<T>(x_padded_host.len())
        .map_err(|error| format!("{label} sparse: padded SpMM x allocation: {error}"))?;
    let mut y_padded = backend
        .raw_alloc::<T>(ldy * rhs)
        .map_err(|error| format!("{label} sparse: padded SpMM y allocation: {error}"))?;
    backend
        .try_copy_h2d(&x_padded_host, &mut x_padded)
        .map_err(|error| format!("{label} sparse: padded SpMM x upload: {error}"))?;
    backend
        .try_spmm_prebaked(&prebaked, rhs, &x_padded, ldx, &mut y_padded, ldy)
        .map_err(|error| format!("{label} sparse: padded SpMM: {error}"))?;
    let mut actual_padded = vec![T::zero(); ldy * rhs];
    backend
        .try_copy_d2h(&y_padded, &mut actual_padded)
        .map_err(|error| format!("{label} sparse: padded SpMM download: {error}"))?;
    for right_hand_side in 0..rhs {
        for row_index in 0..m {
            let index = right_hand_side * ldy + row_index;
            let expected = actual_multi[right_hand_side * m + row_index];
            let error = relative_error(actual_padded[index], expected);
            if error > MAX_RELATIVE_ERROR {
                return Err(format!(
                    "{label} sparse padded SpMM element {index}: relative error {error} exceeds {MAX_RELATIVE_ERROR}"
                ));
            }
        }
    }

    let mut trans_row = backend
        .raw_alloc::<i32>(n + 1)
        .map_err(|error| format!("{label} sparse: transpose row allocation: {error}"))?;
    let mut trans_col = backend
        .raw_alloc::<i32>(col_host.len())
        .map_err(|error| format!("{label} sparse: transpose column allocation: {error}"))?;
    let mut trans_values = backend
        .raw_alloc::<T>(values_host.len())
        .map_err(|error| format!("{label} sparse: transpose values allocation: {error}"))?;
    backend
        .try_csr_transpose(
            m,
            n,
            &row,
            &col,
            &values,
            &mut trans_row,
            &mut trans_col,
            &mut trans_values,
        )
        .map_err(|error| format!("{label} sparse transpose: {error}"))?;
    let mut expected_trans_row = vec![0_i32; n + 1];
    for &column in &col_host {
        expected_trans_row[column as usize + 1] += 1;
    }
    for column in 0..n {
        expected_trans_row[column + 1] += expected_trans_row[column];
    }
    let mut next = expected_trans_row[..n].to_vec();
    let mut expected_trans_col = vec![0_i32; col_host.len()];
    let mut expected_trans_values = vec![T::zero(); values_host.len()];
    for row_index in 0..m {
        for position in row_host[row_index] as usize..row_host[row_index + 1] as usize {
            let column = col_host[position] as usize;
            let destination = next[column] as usize;
            expected_trans_col[destination] = row_index as i32;
            expected_trans_values[destination] = values_host[position];
            next[column] += 1;
        }
    }
    let mut actual_trans_row = vec![0_i32; n + 1];
    let mut actual_trans_col = vec![0_i32; col_host.len()];
    let mut actual_trans_values = vec![T::zero(); values_host.len()];
    backend
        .try_copy_d2h(&trans_row, &mut actual_trans_row)
        .map_err(|error| format!("{label} sparse transpose row download: {error}"))?;
    backend
        .try_copy_d2h(&trans_col, &mut actual_trans_col)
        .map_err(|error| format!("{label} sparse transpose column download: {error}"))?;
    backend
        .try_copy_d2h(&trans_values, &mut actual_trans_values)
        .map_err(|error| format!("{label} sparse transpose values download: {error}"))?;
    if actual_trans_row != expected_trans_row || actual_trans_col != expected_trans_col {
        return Err(format!("{label} sparse transpose structure mismatch"));
    }
    for (index, (&value, &expected)) in actual_trans_values
        .iter()
        .zip(&expected_trans_values)
        .enumerate()
    {
        let error = relative_error(value, expected);
        if error > MAX_RELATIVE_ERROR {
            return Err(format!(
                "{label} sparse transpose value {index}: relative error {error} exceeds {MAX_RELATIVE_ERROR}"
            ));
        }
    }
    Ok(4)
}

/// Exercise one-dimensional and rank-N checked view contracts, including
/// strided/offset/empty views, overlap-safe copies, and malformed descriptors.
fn run_view_contract_matrix<B, T, F>(
    backend: &B,
    make_scalar: F,
    label: &str,
) -> Result<usize, String>
where
    B: Backend,
    T: DeviceScalar + PartialEq,
    F: Fn(usize) -> T + Copy,
{
    let mut cases = 0;
    for case in 0..6 {
        let buffer_len = 32 + case;
        let host: Vec<_> = (0..buffer_len).map(make_scalar).collect();
        let mut source = backend
            .raw_alloc::<T>(buffer_len)
            .map_err(|error| format!("{label} view case {case}: source allocation: {error}"))?;
        backend
            .try_copy_h2d(&host, &mut source)
            .map_err(|error| format!("{label} view case {case}: source upload: {error}"))?;

        let offset = 1 + case % 3;
        let stride = 1 + case % 3;
        let len = 5 + case % 4;
        let view = backend
            .checked_view(&source, offset, len, stride)
            .map_err(|error| format!("{label} view case {case}: view admission: {error}"))?;
        let mut selected = vec![T::zero(); len];
        backend
            .try_copy_view_d2h(&source, view, &mut selected)
            .map_err(|error| format!("{label} view case {case}: strided download: {error}"))?;
        let expected: Vec<_> = (0..len)
            .map(|index| host[offset + index * stride])
            .collect();
        if selected != expected {
            return Err(format!(
                "{label} view case {case}: strided reference mismatch"
            ));
        }

        let empty = backend
            .checked_view(&source, buffer_len, 0, 2)
            .map_err(|error| format!("{label} view case {case}: empty admission: {error}"))?;
        backend
            .try_copy_view_d2h(&source, empty, &mut [])
            .map_err(|error| format!("{label} view case {case}: empty copy: {error}"))?;

        let shape = [2, 3];
        let element_strides = [4 + case % 2, 1];
        let byte_offset = (1 + case % 2) * std::mem::size_of::<T>();
        let descriptor = backend
            .checked_nd_view_descriptor(
                &source,
                &shape,
                byte_offset,
                &element_strides,
                ViewAliasing::ReadOnly,
            )
            .map_err(|error| format!("{label} view case {case}: rank-N admission: {error}"))?;
        let mut rank_values = vec![T::zero(); 6];
        backend
            .try_copy_nd_view_d2h(&source, &descriptor, &mut rank_values)
            .map_err(|error| format!("{label} view case {case}: rank-N download: {error}"))?;
        let host_ref = &host;
        let rank_expected: Vec<_> = (0..2)
            .flat_map(|row| {
                let base = 1 + case % 2 + row * element_strides[0];
                (0..3).map(move |column| host_ref[base + column])
            })
            .collect();
        if rank_values != rank_expected {
            return Err(format!(
                "{label} view case {case}: rank-N reference mismatch"
            ));
        }

        let mut destination = backend.raw_alloc::<T>(buffer_len).map_err(|error| {
            format!("{label} view case {case}: destination allocation: {error}")
        })?;
        let destination_descriptor = backend
            .checked_nd_view_descriptor(
                &destination,
                &shape,
                byte_offset,
                &element_strides,
                ViewAliasing::Disjoint,
            )
            .map_err(|error| format!("{label} view case {case}: destination admission: {error}"))?;
        backend
            .try_copy_nd_view_h2d(&rank_expected, &mut destination, &destination_descriptor)
            .map_err(|error| format!("{label} view case {case}: rank-N upload: {error}"))?;
        let mut round_trip = vec![T::zero(); 6];
        backend
            .try_copy_nd_view_d2h(&destination, &destination_descriptor, &mut round_trip)
            .map_err(|error| format!("{label} view case {case}: rank-N round trip: {error}"))?;
        if round_trip != rank_expected {
            return Err(format!(
                "{label} view case {case}: rank-N round-trip mismatch"
            ));
        }

        // Exercise the explicit overlap-safe aliasing policy as a real
        // memmove operation.  The source and destination descriptors share
        // one allocation and overlap by three elements; the checked backend
        // must stage the source before publishing the shifted view.
        let mut before_overlap = vec![T::zero(); buffer_len];
        backend
            .try_copy_d2h(&destination, &mut before_overlap)
            .map_err(|error| format!("{label} view case {case}: overlap snapshot: {error}"))?;
        let overlap_source = backend
            .checked_nd_view_descriptor(&destination, &[4], 0, &[1], ViewAliasing::ReadOnly)
            .map_err(|error| format!("{label} view case {case}: overlap source: {error}"))?;
        let overlap_destination = backend
            .checked_nd_view_descriptor(
                &destination,
                &[4],
                std::mem::size_of::<T>(),
                &[1],
                ViewAliasing::OverlapSafe,
            )
            .map_err(|error| format!("{label} view case {case}: overlap destination: {error}"))?;
        backend
            .try_copy_nd_view_in_place(&mut destination, &overlap_source, &overlap_destination)
            .map_err(|error| format!("{label} view case {case}: overlap copy: {error}"))?;
        let mut after_overlap = vec![T::zero(); buffer_len];
        backend
            .try_copy_d2h(&destination, &mut after_overlap)
            .map_err(|error| format!("{label} view case {case}: overlap result: {error}"))?;
        let mut expected_overlap = before_overlap;
        let shifted = expected_overlap[0..4].to_vec();
        expected_overlap[1..5].copy_from_slice(&shifted);
        if after_overlap != expected_overlap {
            return Err(format!(
                "{label} view case {case}: overlap aliasing reference mismatch"
            ));
        }

        let malformed = backend.checked_nd_view_descriptor(
            &source,
            &[2, 3],
            byte_offset,
            &[0, 1],
            ViewAliasing::ReadOnly,
        );
        if malformed.is_ok() {
            return Err(format!(
                "{label} view case {case}: zero stride was accepted"
            ));
        }
        cases += 1;
    }
    Ok(cases)
}

/// Exercise all explicit reduction policies against independent host sums.
fn run_reduction_matrix<B, T, F>(backend: &B, make_scalar: F, label: &str) -> Result<usize, String>
where
    B: Backend,
    T: DeviceScalar + PartialEq,
    T::Real: ToPrimitive,
    F: Fn(usize) -> T + Copy,
{
    let modes = [
        ReductionMode::Fast,
        ReductionMode::Reproducible,
        ReductionMode::Deterministic,
    ];
    let lengths = [0usize, 1, 7, 10];
    let mut cases = 0;
    for (fixture, &length) in lengths.iter().enumerate() {
        let x_host: Vec<_> = (0..length)
            .map(|index| make_scalar(index + fixture * 101))
            .collect();
        let y_host: Vec<_> = (0..length)
            .map(|index| make_scalar(index + fixture * 211 + 10_000))
            .collect();
        let basis_count = 3;
        let basis_host: Vec<_> = (0..basis_count * length)
            .map(|index| make_scalar(index + fixture * 307 + 20_000))
            .collect();
        let mut x = backend.raw_alloc::<T>(length).map_err(|error| {
            format!("{label} reduction fixture {fixture}: x allocation: {error}")
        })?;
        let mut y = backend.raw_alloc::<T>(length).map_err(|error| {
            format!("{label} reduction fixture {fixture}: y allocation: {error}")
        })?;
        let mut basis = backend.raw_alloc::<T>(basis_host.len()).map_err(|error| {
            format!("{label} reduction fixture {fixture}: basis allocation: {error}")
        })?;
        backend
            .try_copy_h2d(&x_host, &mut x)
            .map_err(|error| format!("{label} reduction fixture {fixture}: x upload: {error}"))?;
        backend
            .try_copy_h2d(&y_host, &mut y)
            .map_err(|error| format!("{label} reduction fixture {fixture}: y upload: {error}"))?;
        backend
            .try_copy_h2d(&basis_host, &mut basis)
            .map_err(|error| {
                format!("{label} reduction fixture {fixture}: basis upload: {error}")
            })?;

        let dot_expected = x_host
            .iter()
            .zip(&y_host)
            .fold(T::zero(), |sum, (&x_value, &y_value)| {
                sum + x_value.conj_mul(y_value)
            });
        let norm_expected = x_host
            .iter()
            .map(|value| value.norm_sqr().to_f64().unwrap_or(f64::INFINITY))
            .sum::<f64>()
            .sqrt();
        for &mode in &modes {
            let dot = backend.try_dot_with_mode(mode, &x, &y).map_err(|error| {
                format!("{label} reduction fixture {fixture} {mode:?} dot: {error}")
            })?;
            if relative_error(dot, dot_expected) > MAX_RELATIVE_ERROR {
                return Err(format!(
                    "{label} reduction fixture {fixture} {mode:?}: dot mismatch"
                ));
            }
            let norm = backend.try_nrm2_with_mode(mode, &x).map_err(|error| {
                format!("{label} reduction fixture {fixture} {mode:?} norm: {error}")
            })?;
            let norm_error = (norm.to_f64().unwrap_or(f64::INFINITY) - norm_expected).abs()
                / norm_expected.max(1.0);
            if norm_error > MAX_RELATIVE_ERROR {
                return Err(format!(
                    "{label} reduction fixture {fixture} {mode:?}: norm mismatch"
                ));
            }
            let multi = backend
                .try_multi_dot_strided_with_mode(mode, &basis, length, basis_count, &y)
                .map_err(|error| {
                    format!("{label} reduction fixture {fixture} {mode:?} multi-dot: {error}")
                })?;
            if multi.len() != basis_count {
                return Err(format!(
                    "{label} reduction fixture {fixture} {mode:?}: multi-dot length"
                ));
            }
            if !matches!(mode, ReductionMode::Fast) {
                let repeated_dot = backend.try_dot_with_mode(mode, &x, &y).map_err(|error| {
                    format!("{label} reduction fixture {fixture} {mode:?} repeated dot: {error}")
                })?;
                let repeated_norm = backend.try_nrm2_with_mode(mode, &x).map_err(|error| {
                    format!("{label} reduction fixture {fixture} {mode:?} repeated norm: {error}")
                })?;
                let repeated_multi = backend
                    .try_multi_dot_strided_with_mode(mode, &basis, length, basis_count, &y)
                    .map_err(|error| {
                        format!(
                            "{label} reduction fixture {fixture} {mode:?} repeated multi-dot: \
                         {error}"
                        )
                    })?;
                if repeated_dot != dot || repeated_norm != norm || repeated_multi != multi {
                    return Err(format!(
                        "{label} reduction fixture {fixture} {mode:?}: repeated result changed"
                    ));
                }
            }
            for basis_index in 0..basis_count {
                let expected = (0..length).fold(T::zero(), |sum, column| {
                    sum + basis_host[basis_index * length + column].conj_mul(y_host[column])
                });
                if relative_error(multi[basis_index], expected) > MAX_RELATIVE_ERROR {
                    return Err(format!(
                        "{label} reduction fixture {fixture} {mode:?}: multi-dot mismatch"
                    ));
                }
            }
            cases += 3;
        }
    }
    Ok(cases)
}

/// Exercise the checked reduction failure contract for every public scalar.
///
/// A non-finite input must be reported as a typed `OperationFailed` error for
/// every explicit reduction policy. Keeping this matrix separate from the
/// finite-value matrix makes it impossible for a backend to silently accept a
/// NaN while still passing the numerical parity checks.
fn run_reduction_failure_matrix<B, T>(backend: &B, label: &str) -> Result<usize, String>
where
    B: Backend,
    T: DeviceScalar,
    T::Real: BufferElement + Float + ToPrimitive,
{
    let modes = [
        ReductionMode::Fast,
        ReductionMode::Reproducible,
        ReductionMode::Deterministic,
    ];
    let vector_len = 3usize;
    let basis_count = 2usize;
    let nan = T::from_real(T::Real::nan());
    let x_host = vec![nan, T::one(), T::zero()];
    let y_host = vec![T::one(); vector_len];
    let basis_host = vec![nan, T::one(), T::zero(), T::one(), T::zero(), T::one()];

    let mut x = backend
        .raw_alloc::<T>(vector_len)
        .map_err(|error| format!("{label} reduction failure x allocation: {error}"))?;
    let mut y = backend
        .raw_alloc::<T>(vector_len)
        .map_err(|error| format!("{label} reduction failure y allocation: {error}"))?;
    let mut basis = backend
        .raw_alloc::<T>(basis_host.len())
        .map_err(|error| format!("{label} reduction failure basis allocation: {error}"))?;
    backend
        .try_copy_h2d(&x_host, &mut x)
        .map_err(|error| format!("{label} reduction failure x upload: {error}"))?;
    backend
        .try_copy_h2d(&y_host, &mut y)
        .map_err(|error| format!("{label} reduction failure y upload: {error}"))?;
    backend
        .try_copy_h2d(&basis_host, &mut basis)
        .map_err(|error| format!("{label} reduction failure basis upload: {error}"))?;

    let mut cases = 0usize;
    for mode in modes {
        let dot = backend.try_dot_with_mode(mode, &x, &y);
        if !matches!(dot, Err(BackendError::OperationFailed { .. })) {
            return Err(format!(
                "{label} reduction failure {mode:?} dot did not return OperationFailed: {dot:?}"
            ));
        }
        cases += 1;

        let norm = backend.try_nrm2_with_mode(mode, &x);
        if !matches!(norm, Err(BackendError::OperationFailed { .. })) {
            return Err(format!(
                "{label} reduction failure {mode:?} nrm2 did not return OperationFailed: {norm:?}"
            ));
        }
        cases += 1;

        let multi_dot =
            backend.try_multi_dot_strided_with_mode(mode, &basis, vector_len, basis_count, &y);
        if !matches!(multi_dot, Err(BackendError::OperationFailed { .. })) {
            return Err(format!(
                "{label} reduction failure {mode:?} multi-dot did not return OperationFailed: \
                 {multi_dot:?}"
            ));
        }
        cases += 1;

        let mut multi_dot_output = [T::zero(); 1];
        let multi_dot_into =
            backend.try_multi_dot_with_mode_into(mode, &[&x], &y, &mut multi_dot_output);
        if !matches!(multi_dot_into, Err(BackendError::OperationFailed { .. })) {
            return Err(format!(
                "{label} reduction failure {mode:?} multi-dot-into did not return \
                 OperationFailed: {multi_dot_into:?}"
            ));
        }
        cases += 1;

        let mut projection = backend
            .raw_alloc::<T>(vector_len)
            .map_err(|error| format!("{label} reduction failure projection allocation: {error}"))?;
        backend
            .try_copy_h2d(&y_host, &mut projection)
            .map_err(|error| format!("{label} reduction failure projection upload: {error}"))?;
        let projected = backend.try_basis_project_out_strided_with_mode(
            mode,
            &basis,
            vector_len,
            basis_count,
            &mut projection,
        );
        if !matches!(projected, Err(BackendError::OperationFailed { .. })) {
            return Err(format!(
                "{label} reduction failure {mode:?} projection did not return \
                 OperationFailed: {projected:?}"
            ));
        }
        cases += 1;
    }

    Ok(cases)
}

/// Exercise the checked gather admission contract for non-finite values,
/// non-finite weights, and invalid indices. Each failure leaves output intact.
fn run_gather_failure_matrix<B, T>(backend: &B, label: &str) -> Result<usize, String>
where
    B: Backend,
    T: DeviceScalar,
    T::Real: BufferElement + Float + ToPrimitive,
{
    let mut values = backend
        .raw_alloc::<T>(2)
        .map_err(|error| format!("{label} gather values allocation: {error}"))?;
    let mut indices = backend
        .raw_alloc::<i32>(4)
        .map_err(|error| format!("{label} gather indices allocation: {error}"))?;
    let mut weights = backend
        .raw_alloc::<T::Real>(4)
        .map_err(|error| format!("{label} gather weights allocation: {error}"))?;
    let mut output = backend
        .raw_alloc::<T>(1)
        .map_err(|error| format!("{label} gather output allocation: {error}"))?;
    let finite_values = [T::one(), T::zero()];
    let finite_indices = [0_i32, 1, 0, 1];
    let finite_weights = [T::Real::zero(); 4];
    let sentinel = T::one();
    let mut output_host = [T::zero(); 1];

    backend
        .try_copy_h2d(&finite_indices, &mut indices)
        .map_err(|error| format!("{label} gather indices upload: {error}"))?;
    backend
        .try_copy_h2d(&finite_weights, &mut weights)
        .map_err(|error| format!("{label} gather weights upload: {error}"))?;

    backend
        .try_copy_h2d(&[T::from_real(T::Real::nan()), T::zero()], &mut values)
        .map_err(|error| format!("{label} gather NaN values upload: {error}"))?;
    backend
        .try_copy_h2d(&[sentinel], &mut output)
        .map_err(|error| format!("{label} gather sentinel upload: {error}"))?;
    let result = backend.try_gather4_weighted(&values, &indices, &weights, &mut output);
    if !matches!(
        result,
        Err(BackendError::OperationFailed {
            operation: "gather4_weighted",
            ..
        })
    ) {
        return Err(format!(
            "{label} gather NaN values did not fail closed: {result:?}"
        ));
    }
    backend
        .try_copy_d2h(&output, &mut output_host)
        .map_err(|error| format!("{label} gather NaN values output download: {error}"))?;
    if relative_error(output_host[0], sentinel) != 0.0 {
        return Err(format!("{label} gather NaN values mutated output"));
    }

    backend
        .try_copy_h2d(&finite_values, &mut values)
        .map_err(|error| format!("{label} gather values upload: {error}"))?;
    backend
        .try_copy_h2d(
            &[
                T::Real::nan(),
                T::Real::zero(),
                T::Real::zero(),
                T::Real::zero(),
            ],
            &mut weights,
        )
        .map_err(|error| format!("{label} gather NaN weights upload: {error}"))?;
    backend
        .try_copy_h2d(&[sentinel], &mut output)
        .map_err(|error| format!("{label} gather sentinel upload: {error}"))?;
    let result = backend.try_gather4_weighted(&values, &indices, &weights, &mut output);
    if !matches!(
        result,
        Err(BackendError::OperationFailed {
            operation: "gather4_weighted",
            ..
        })
    ) {
        return Err(format!(
            "{label} gather NaN weights did not fail closed: {result:?}"
        ));
    }
    backend
        .try_copy_d2h(&output, &mut output_host)
        .map_err(|error| format!("{label} gather NaN weights output download: {error}"))?;
    if relative_error(output_host[0], sentinel) != 0.0 {
        return Err(format!("{label} gather NaN weights mutated output"));
    }

    backend
        .try_copy_h2d(&finite_weights, &mut weights)
        .map_err(|error| format!("{label} gather finite weights upload: {error}"))?;
    backend
        .try_copy_h2d(&[-1_i32, 1, 0, 1], &mut indices)
        .map_err(|error| format!("{label} gather invalid index upload: {error}"))?;
    backend
        .try_copy_h2d(&[sentinel], &mut output)
        .map_err(|error| format!("{label} gather sentinel upload: {error}"))?;
    let result = backend.try_gather4_weighted(&values, &indices, &weights, &mut output);
    if !matches!(result, Err(BackendError::InvalidArgument { .. })) {
        return Err(format!(
            "{label} gather invalid index did not fail closed: {result:?}"
        ));
    }
    backend
        .try_copy_d2h(&output, &mut output_host)
        .map_err(|error| format!("{label} gather invalid index output download: {error}"))?;
    if relative_error(output_host[0], sentinel) != 0.0 {
        return Err(format!("{label} gather invalid index mutated output"));
    }

    Ok(3)
}

/// Exercise the caller-owned flat multi-dot output contract for every scalar
/// family, reduction policy, and empty/odd vector shape.
fn run_multi_dot_into_matrix<B, T, F>(
    backend: &B,
    make_scalar: F,
    label: &str,
) -> Result<usize, String>
where
    B: Backend,
    T: DeviceScalar,
    T::Real: ToPrimitive,
    F: Fn(usize) -> T + Copy,
{
    let modes = [
        ReductionMode::Fast,
        ReductionMode::Reproducible,
        ReductionMode::Deterministic,
    ];
    let lengths = [0usize, 1, 7, 10];
    let basis_count = 3usize;
    let mut cases = 0usize;
    for (fixture, &length) in lengths.iter().enumerate() {
        let y_host: Vec<_> = (0..length)
            .map(|index| make_scalar(index + fixture * 211 + 10_000))
            .collect();
        let basis_host: Vec<Vec<_>> = (0..basis_count)
            .map(|basis| {
                (0..length)
                    .map(|index| make_scalar(index + fixture * 307 + basis * 101 + 20_000))
                    .collect()
            })
            .collect();
        let mut y = backend
            .raw_alloc::<T>(length)
            .map_err(|error| format!("{label} multi-dot-into fixture {fixture}: y: {error}"))?;
        backend.try_copy_h2d(&y_host, &mut y).map_err(|error| {
            format!("{label} multi-dot-into fixture {fixture}: y upload: {error}")
        })?;
        let mut basis_buffers = Vec::with_capacity(basis_count);
        for (basis_index, host) in basis_host.iter().enumerate() {
            let mut buffer = backend.raw_alloc::<T>(length).map_err(|error| {
                format!(
                    "{label} multi-dot-into fixture {fixture} basis {basis_index}: allocation: {error}"
                )
            })?;
            backend.try_copy_h2d(host, &mut buffer).map_err(|error| {
                format!(
                    "{label} multi-dot-into fixture {fixture} basis {basis_index}: upload: {error}"
                )
            })?;
            basis_buffers.push(buffer);
        }
        let basis_refs: Vec<&B::Buffer<T>> = basis_buffers.iter().collect();
        for mode in modes {
            let mut actual = vec![T::zero(); basis_count];
            backend
                .try_multi_dot_with_mode_into(mode, &basis_refs, &y, &mut actual)
                .map_err(|error| {
                    format!("{label} multi-dot-into fixture {fixture} {mode:?}: {error}")
                })?;
            for (basis_index, actual_value) in actual.into_iter().enumerate() {
                let expected = basis_host[basis_index]
                    .iter()
                    .zip(&y_host)
                    .fold(T::zero(), |sum, (basis_value, y_value)| {
                        sum + basis_value.conj_mul(*y_value)
                    });
                if relative_error(actual_value, expected) > MAX_RELATIVE_ERROR {
                    return Err(format!(
                        "{label} multi-dot-into fixture {fixture} {mode:?} basis {basis_index}: mismatch"
                    ));
                }
            }
            cases += 1;
        }
    }
    Ok(cases)
}

fn check_failure_contract<B: Backend>(backend: &B) -> Result<(), String> {
    let a = backend
        .raw_alloc::<f64>(6)
        .map_err(|error| error.to_string())?;
    let x = backend
        .raw_alloc::<f64>(3)
        .map_err(|error| error.to_string())?;
    let mut y = backend
        .raw_alloc::<f64>(2)
        .map_err(|error| error.to_string())?;
    let invalid = backend.try_gemv(Transpose::NoTrans, 2, 3, 1.0, &a, 2, &x, 0.0, &mut y);
    if !matches!(invalid, Err(BackendError::InvalidArgument { .. })) {
        return Err(format!(
            "invalid GEMV layout did not produce InvalidArgument: {invalid:?}"
        ));
    }
    let nonfinite = backend.try_gemv(Transpose::NoTrans, 2, 3, f64::NAN, &a, 3, &x, 0.0, &mut y);
    if !matches!(nonfinite, Err(BackendError::OperationFailed { .. })) {
        return Err(format!(
            "non-finite GEMV scalar did not produce OperationFailed: {nonfinite:?}"
        ));
    }
    Ok(())
}

fn capability_count(capabilities: sonium_compute::BackendCapabilities) -> (usize, usize) {
    let states = [
        capabilities.blas1,
        capabilities.gemv,
        capabilities.gemm,
        capabilities.fft,
        capabilities.csr_spmv,
        capabilities.csr_spmm,
        capabilities.csr_transpose,
        capabilities.multi_dot,
        capabilities.multi_dot_strided,
        capabilities.basis_axpy_strided,
        capabilities.basis_project_out_strided,
        capabilities.gather4_weighted,
        capabilities.segmented_reduce,
        capabilities.batched_local_contract,
        capabilities.reduction.fast,
        capabilities.reduction.reproducible,
        capabilities.reduction.deterministic,
    ];
    let mut supported = 0;
    let mut unsupported = 0;
    for scalar in states {
        for state in [scalar.f32, scalar.f64, scalar.c32, scalar.c64] {
            if matches!(state, CapabilityState::Unsupported) {
                unsupported += 1;
            } else {
                supported += 1;
            }
        }
    }
    for state in [
        capabilities.precision_cast.f64_to_f32,
        capabilities.precision_cast.c64_to_c32,
        capabilities.precision_cast.f32_to_f64,
        capabilities.precision_cast.c32_to_c64,
    ] {
        if matches!(state, CapabilityState::Unsupported) {
            unsupported += 1;
        } else {
            supported += 1;
        }
    }
    (supported, unsupported)
}

/// Exercise overflow admission before any dense buffer access or native launch.
fn run_dense_overflow_contract<B, T>(backend: &B, label: &str) -> Result<usize, String>
where
    B: Backend,
    T: DeviceScalar,
{
    let empty = backend
        .raw_alloc::<T>(0)
        .map_err(|error| format!("{label} overflow A allocation: {error}"))?;
    let x = backend
        .raw_alloc::<T>(1)
        .map_err(|error| format!("{label} overflow x allocation: {error}"))?;
    let mut y = backend
        .raw_alloc::<T>(2)
        .map_err(|error| format!("{label} overflow y allocation: {error}"))?;
    let gemv = backend.try_gemv(
        Transpose::NoTrans,
        2,
        1,
        T::one(),
        &empty,
        usize::MAX,
        &x,
        T::zero(),
        &mut y,
    );
    if !matches!(
        &gemv,
        Err(BackendError::InvalidArgument { operation: "gemv", reason })
            if reason.contains("span overflows")
    ) {
        return Err(format!(
            "{label} overflowing GEMV was not rejected: {gemv:?}"
        ));
    }

    let b = backend
        .raw_alloc::<T>(1)
        .map_err(|error| format!("{label} overflow B allocation: {error}"))?;
    let mut c = backend
        .raw_alloc::<T>(2)
        .map_err(|error| format!("{label} overflow C allocation: {error}"))?;
    let gemm = backend.try_gemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        2,
        1,
        1,
        T::one(),
        &empty,
        usize::MAX,
        &b,
        1,
        T::zero(),
        &mut c,
        1,
    );
    if !matches!(
        &gemm,
        Err(BackendError::InvalidArgument { operation: "gemm", reason })
            if reason.contains("span overflows")
    ) {
        return Err(format!(
            "{label} overflowing GEMM was not rejected: {gemm:?}"
        ));
    }
    Ok(2)
}

fn requested_backend() -> Result<Option<String>, String> {
    match std::env::var("SONIUM_BACKEND") {
        Ok(value) => {
            let value = value.trim().to_ascii_lowercase();
            if matches!(value.as_str(), "cpu" | "metal" | "cuda") {
                Ok(Some(value))
            } else {
                Err(format!(
                    "SONIUM_BACKEND must be cpu, metal, or cuda; got {value:?}"
                ))
            }
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("could not read SONIUM_BACKEND: {error}")),
    }
}

fn run_for_backend<B: Backend>(backend: &B, requested: Option<&str>) -> Result<QaResult, String> {
    let started = Instant::now();
    let (capability_supported, capability_unsupported) = capability_count(backend.capabilities());
    check_failure_contract(backend)?;
    let dense_overflow_cases = run_dense_overflow_contract::<_, f64>(backend, "f64")?;
    let summaries = [
        run_scalar_matrix::<_, f32, _>(
            backend,
            |index| ((index as f32 * 0.17).sin() * 0.75) as f32,
            "f32",
        )?,
        run_scalar_matrix::<_, f64, _>(backend, |index| (index as f64 * 0.17).sin() * 0.75, "f64")?,
        run_scalar_matrix::<_, Complex32, _>(
            backend,
            |index| {
                Complex32::new(
                    (index as f32 * 0.17).sin() * 0.75,
                    (index as f32 * 0.11).cos() * 0.5,
                )
            },
            "complex32",
        )?,
        run_scalar_matrix::<_, Complex64, _>(
            backend,
            |index| {
                Complex64::new(
                    (index as f64 * 0.17).sin() * 0.75,
                    (index as f64 * 0.11).cos() * 0.5,
                )
            },
            "complex64",
        )?,
    ];
    let local_contract_cases = run_local_contract_matrix::<_, f32, _>(
        backend,
        |index| ((index as f32 * 0.13).sin() * 0.5) as f32,
        "f32",
    )? + run_local_contract_matrix::<_, f64, _>(
        backend,
        |index| (index as f64 * 0.13).sin() * 0.5,
        "f64",
    )? + run_local_contract_matrix::<_, Complex32, _>(
        backend,
        |index| {
            Complex32::new(
                (index as f32 * 0.13).sin() * 0.5,
                (index as f32 * 0.07).cos() * 0.25,
            )
        },
        "complex32",
    )? + run_local_contract_matrix::<_, Complex64, _>(
        backend,
        |index| {
            Complex64::new(
                (index as f64 * 0.13).sin() * 0.5,
                (index as f64 * 0.07).cos() * 0.25,
            )
        },
        "complex64",
    )?;
    let cast_cases = run_cast_matrix::<_, f64, _>(
        backend,
        |index| (index as f64 * 0.17).sin() * 0.75,
        "f64/f32",
    )? + run_cast_matrix::<_, Complex64, _>(
        backend,
        |index| {
            Complex64::new(
                (index as f64 * 0.17).sin() * 0.75,
                (index as f64 * 0.11).cos() * 0.5,
            )
        },
        "complex64/complex32",
    )?;
    let gather_cases = run_gather_matrix::<_, f32>(backend, "f32")?
        + run_gather_matrix::<_, f64>(backend, "f64")?
        + run_gather_matrix::<_, Complex32>(backend, "complex32")?
        + run_gather_matrix::<_, Complex64>(backend, "complex64")?;
    let sparse_cases = run_sparse_matrix::<_, f32, _>(
        backend,
        |index| ((index as f32 * 0.13).sin() * 0.5) as f32,
        "f32",
    )? + run_sparse_matrix::<_, f64, _>(
        backend,
        |index| (index as f64 * 0.13).sin() * 0.5,
        "f64",
    )? + run_sparse_matrix::<_, Complex32, _>(
        backend,
        |index| {
            Complex32::new(
                (index as f32 * 0.13).sin() * 0.5,
                (index as f32 * 0.07).cos() * 0.25,
            )
        },
        "complex32",
    )? + run_sparse_matrix::<_, Complex64, _>(
        backend,
        |index| {
            Complex64::new(
                (index as f64 * 0.13).sin() * 0.5,
                (index as f64 * 0.07).cos() * 0.25,
            )
        },
        "complex64",
    )?;
    let view_cases = run_view_contract_matrix::<_, f32, _>(
        backend,
        |index| ((index as f32 * 0.19).sin() * 0.4) as f32,
        "f32",
    )? + run_view_contract_matrix::<_, f64, _>(
        backend,
        |index| (index as f64 * 0.19).sin() * 0.4,
        "f64",
    )? + run_view_contract_matrix::<_, Complex32, _>(
        backend,
        |index| {
            Complex32::new(
                (index as f32 * 0.19).sin() * 0.4,
                (index as f32 * 0.09).cos() * 0.2,
            )
        },
        "complex32",
    )? + run_view_contract_matrix::<_, Complex64, _>(
        backend,
        |index| {
            Complex64::new(
                (index as f64 * 0.19).sin() * 0.4,
                (index as f64 * 0.09).cos() * 0.2,
            )
        },
        "complex64",
    )?;
    let reduction_cases = run_reduction_matrix::<_, f32, _>(
        backend,
        |index| ((index as f32 * 0.23).sin() * 0.35) as f32,
        "f32",
    )? + run_reduction_matrix::<_, f64, _>(
        backend,
        |index| (index as f64 * 0.23).sin() * 0.35,
        "f64",
    )? + run_reduction_matrix::<_, Complex32, _>(
        backend,
        |index| {
            Complex32::new(
                (index as f32 * 0.23).sin() * 0.35,
                (index as f32 * 0.17).cos() * 0.21,
            )
        },
        "complex32",
    )? + run_reduction_matrix::<_, Complex64, _>(
        backend,
        |index| {
            Complex64::new(
                (index as f64 * 0.23).sin() * 0.35,
                (index as f64 * 0.17).cos() * 0.21,
            )
        },
        "complex64",
    )?;
    let reduction_failure_cases = run_reduction_failure_matrix::<_, f32>(backend, "f32")?
        + run_reduction_failure_matrix::<_, f64>(backend, "f64")?
        + run_reduction_failure_matrix::<_, Complex32>(backend, "complex32")?
        + run_reduction_failure_matrix::<_, Complex64>(backend, "complex64")?;
    let gather_failure_cases = run_gather_failure_matrix::<_, f32>(backend, "f32")?
        + run_gather_failure_matrix::<_, f64>(backend, "f64")?
        + run_gather_failure_matrix::<_, Complex32>(backend, "complex32")?
        + run_gather_failure_matrix::<_, Complex64>(backend, "complex64")?;
    let multi_dot_into_cases = run_multi_dot_into_matrix::<_, f32, _>(
        backend,
        |index| ((index as f32 * 0.29).sin() * 0.31) as f32,
        "f32",
    )? + run_multi_dot_into_matrix::<_, f64, _>(
        backend,
        |index| (index as f64 * 0.29).sin() * 0.31,
        "f64",
    )? + run_multi_dot_into_matrix::<_, Complex32, _>(
        backend,
        |index| {
            Complex32::new(
                (index as f32 * 0.29).sin() * 0.31,
                (index as f32 * 0.13).cos() * 0.19,
            )
        },
        "complex32",
    )? + run_multi_dot_into_matrix::<_, Complex64, _>(
        backend,
        |index| {
            Complex64::new(
                (index as f64 * 0.29).sin() * 0.31,
                (index as f64 * 0.13).cos() * 0.19,
            )
        },
        "complex64",
    )?;

    let max_relative_error = summaries
        .iter()
        .map(|summary| summary.max_relative_error)
        .fold(0.0, f64::max);
    let scalar_type_count = summaries
        .iter()
        .map(|summary| summary.scalar_types)
        .sum::<usize>();
    let layout_case_count = summaries
        .iter()
        .map(|summary| summary.layout_cases)
        .sum::<usize>();
    let operation_case_count = summaries
        .iter()
        .map(|summary| summary.operation_cases)
        .sum::<usize>();
    let elapsed_ms = started.elapsed().as_millis() as u64;
    if max_relative_error > MAX_RELATIVE_ERROR {
        return Err(format!(
            "maximum relative error {max_relative_error} exceeds {MAX_RELATIVE_ERROR}"
        ));
    }

    let actual_backend = backend.capabilities().backend_name;
    let mut result = QaResult::passed(CASE_ID)
        .actual_backend(actual_backend)
        .metric("max_relative_error", max_relative_error)
        .metric("scalar_type_count", scalar_type_count as f64)
        .metric("layout_case_count", layout_case_count as f64)
        .metric("operation_case_count", operation_case_count as f64)
        .metric("multi_dot_into_case_count", multi_dot_into_cases as f64)
        .metric("local_contract_case_count", local_contract_cases as f64)
        .metric("cast_case_count", cast_cases as f64)
        .metric("gather_case_count", gather_cases as f64)
        .metric("sparse_case_count", sparse_cases as f64)
        .metric("view_case_count", view_cases as f64)
        .metric("alias_view_case_count", view_cases as f64)
        .metric("reduction_case_count", reduction_cases as f64)
        .metric(
            "reduction_failure_case_count",
            reduction_failure_cases as f64,
        )
        .metric("gather_failure_case_count", gather_failure_cases as f64)
        .metric("capability_supported_count", capability_supported as f64)
        .metric("dense_overflow_case_count", dense_overflow_cases as f64)
        .metric(
            "capability_unsupported_count",
            capability_unsupported as f64,
        )
        .metric("failure_contract_count", 2.0)
        .maximum("max_relative_error", MAX_RELATIVE_ERROR)
        .minimum("scalar_type_count", 4.0)
        .minimum("layout_case_count", 144.0)
        .minimum("operation_case_count", 144.0)
        .minimum("local_contract_case_count", 32.0)
        .minimum("cast_case_count", 24.0)
        .minimum("gather_case_count", 12.0)
        .minimum("sparse_case_count", 16.0)
        .minimum("view_case_count", 24.0)
        .minimum("alias_view_case_count", 24.0)
        .minimum("reduction_case_count", 144.0)
        .minimum("reduction_failure_case_count", 60.0)
        .minimum("gather_failure_case_count", 12.0)
        .minimum("multi_dot_into_case_count", 48.0)
        .minimum("dense_overflow_case_count", 2.0)
        .input_checksum("analytic/deterministic-fixture:compute-operation-matrix-v1")
        .reference(
            "independent scalar row-major GEMV/GEMM/reduction/local-contraction accumulation, precision cast round trips, gather index/weight accumulation, CSR SpMV/SpMM and stable transpose references, and view-index/memmove alias references",
        )
        .reference_checksum("analytic/deterministic-fixture")
        .dimension(
            "f32/f64/Complex32/Complex64; transpose matrix; padded leading dimensions; precision cast round trips; successful weighted four-point gather with repeated/non-monotone indices; CSR SpMV/SpMM with padded leading dimensions and stable CSR(Aᵀ); explicit reduction modes; local contractions; strided/rank-N views with offset, empty, malformed, and overlap-safe alias cases",
        )
        .formulation(format!(
            "checked {actual_backend} Backend operation/type/layout/failure matrix"
        ))
        .tolerance("maximum relative element error <= 4e-5; at least 144 finite reduction, 60 reduction-failure, 12 successful gather, 12 gather-failure, 32 local-contract, 24 cast, 16 sparse, and 24 view cases")
        .seed("compute-operation-matrix:18-layout-cases-per-operation-and-scalar:v4-coverage")
        .timing_ms("matrix", elapsed_ms)
        .elapsed_ms(elapsed_ms)
        .backend_passed(actual_backend);
    if let Some(requested) = requested {
        result = result.requested_backend(requested);
    }
    for optional_backend in ["cpu", "metal", "cuda"] {
        if optional_backend != actual_backend {
            result = result.backend_skipped(
                optional_backend,
                "backend was not selected for this operation-matrix invocation",
            );
        }
    }
    result
        .validate()
        .map_err(|error| format!("operation-matrix result failed validation: {error}"))?;
    Ok(result)
}

fn run() -> Result<QaResult, String> {
    let requested = requested_backend()?;
    match requested.as_deref() {
        None | Some("cpu") => {
            let backend = CpuBackend::new();
            run_for_backend(&backend, requested.as_deref())
        }
        Some("metal") => {
            #[cfg(feature = "metal")]
            {
                let backend = sonium_compute::MetalBackend::new()
                    .map_err(|error| format!("Metal backend initialization failed: {error}"))?;
                run_for_backend(&backend, requested.as_deref())
            }
            #[cfg(not(feature = "metal"))]
            {
                Err("SONIUM_BACKEND=metal requires the sonium-qa `metal` feature".into())
            }
        }
        Some("cuda") => {
            #[cfg(feature = "cuda")]
            {
                let backend = required_cuda_backend()?;
                run_for_backend(&backend, requested.as_deref())
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err("SONIUM_BACKEND=cuda requires the sonium-qa `cuda` feature".into())
            }
        }
        Some(_) => unreachable!("requested_backend validates the value"),
    }
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("compute operation matrix result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend(
                        std::env::var("SONIUM_BACKEND").unwrap_or_else(|_| "cpu".into()),
                    )
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("compute operation matrix failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
