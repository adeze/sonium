//! Contract tests for the validate-once (`_unchecked` + `into_buf`) hot path.
//!
//! Every unchecked kernel must agree bit-for-bit with its checked sibling on
//! valid input (same operation order, only validation differs), the CSR block
//! adjoint overrides must match the per-column reference, and the BSR/NUMA
//! infallible paths must propagate non-finite values arithmetically (mirroring
//! the CSR hot-path precedent) while the `try_*` boundaries still reject them.

use num_complex::Complex64;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::{LinearOperator, SolverError};
use sonium_solvers_sparse::{BsrMatrix, CsrMatrix, NumaPartitionedCsr, NumaSpmvError};

fn nonsymmetric_4x3() -> CsrMatrix<f64> {
    // Deliberately nonsymmetric and rectangular so transpose != forward.
    CsrMatrix::from_raw_parts(
        4,
        3,
        vec![0, 2, 3, 5, 6],
        vec![0, 2, 1, 0, 2, 1],
        vec![4.0, 1.0, 3.0, 0.5, 2.0, 1.5],
    )
}

fn complex_3x3() -> CsrMatrix<Complex64> {
    // Non-Hermitian: transpose and Hermitian actions differ by conjugation.
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 4, 6],
        vec![0, 1, 1, 2, 0, 2],
        vec![
            Complex64::new(2.0, 1.0),
            Complex64::new(0.5, -1.5),
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, 2.0),
            Complex64::new(-1.0, 0.5),
            Complex64::new(4.0, -2.0),
        ],
    )
}

fn assert_vec_eq(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() <= 1e-12, "mismatch at {index}: {a} != {e}");
    }
}

fn assert_complex_vec_eq(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).norm() <= 1e-12, "mismatch at {index}: {a} != {e}");
    }
}

#[test]
fn unchecked_transpose_matches_checked_real_and_complex() {
    let real = nonsymmetric_4x3();
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut unchecked = vec![0.0; 3];
    unsafe { real.matvec_transpose_into_buf_unchecked(&x, &mut unchecked) };
    let mut checked = vec![0.0; 3];
    real.matvec_transpose_into_buf(&x, &mut checked);
    assert_vec_eq(&unchecked, &checked);
    assert_vec_eq(&checked, &real.matvec_transpose(&x));

    let complex = complex_3x3();
    let xc = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(1.0, -1.0),
    ];
    let mut unchecked_c = vec![Complex64::new(0.0, 0.0); 3];
    unsafe { complex.matvec_transpose_into_buf_unchecked(&xc, &mut unchecked_c) };
    let mut checked_c = vec![Complex64::new(0.0, 0.0); 3];
    complex.matvec_transpose_into_buf(&xc, &mut checked_c);
    assert_complex_vec_eq(&unchecked_c, &checked_c);
}

#[test]
fn unchecked_hermitian_matches_checked_and_differs_from_transpose() {
    let complex = complex_3x3();
    let x = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(-1.0, 1.0),
        Complex64::new(0.5, -0.5),
    ];
    let mut unchecked = vec![Complex64::new(0.0, 0.0); 3];
    unsafe { complex.matvec_hermitian_into_buf_unchecked(&x, &mut unchecked) };
    let mut checked = vec![Complex64::new(0.0, 0.0); 3];
    complex.matvec_hermitian_into_buf(&x, &mut checked);
    assert_complex_vec_eq(&unchecked, &checked);

    // The fixture is non-Hermitian, so this also proves the unchecked path
    // conjugates rather than aliasing the transpose kernel.
    let transpose = complex.matvec_transpose(&x);
    assert!(
        unchecked
            .iter()
            .zip(transpose.iter())
            .any(|(h, t)| (h - t).norm() > 1e-6),
        "hermitian and transpose actions unexpectedly agree"
    );
}

#[test]
fn unchecked_spmm_matches_checked_f64_and_f32() {
    let real = nonsymmetric_4x3();
    let x = DenseMatrix::from_row_major_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut unchecked = DenseMatrix::zeros(4, 2);
    unsafe { real.spmm_into_unchecked(&x, &mut unchecked) };
    let mut checked = DenseMatrix::zeros(4, 2);
    real.spmm_into(&x, &mut checked);
    assert_eq!(unchecked.as_slice(), checked.as_slice());

    // f32 exercises the generic runtime-k SpMM kernel (no const-K dispatch).
    let f32_matrix = CsrMatrix::from_raw_parts(
        4,
        3,
        vec![0, 2, 3, 5, 6],
        vec![0, 2, 1, 0, 2, 1],
        vec![4.0f32, 1.0, 3.0, 0.5, 2.0, 1.5],
    );
    let xf = DenseMatrix::from_row_major_vec(3, 5, (1..=15).map(|v| v as f32).collect());
    let mut unchecked_f = DenseMatrix::zeros(4, 5);
    unsafe { f32_matrix.spmm_into_unchecked(&xf, &mut unchecked_f) };
    let mut checked_f = DenseMatrix::zeros(4, 5);
    f32_matrix.spmm_into(&xf, &mut checked_f);
    assert_eq!(unchecked_f.as_slice(), checked_f.as_slice());
}

#[test]
fn unchecked_add_matches_checked() {
    let real = nonsymmetric_4x3();
    let x = vec![1.0, 2.0, 3.0];
    let mut unchecked = vec![10.0, 20.0, 30.0, 40.0];
    unsafe { real.matvec_add_unchecked(&x, &mut unchecked) };
    let mut checked = vec![10.0, 20.0, 30.0, 40.0];
    real.matvec_add(&x, &mut checked);
    assert_vec_eq(&unchecked, &checked);
}

#[test]
fn unchecked_entries_still_assert_vector_lengths() {
    let real = nonsymmetric_4x3();
    let short = vec![1.0; 2];
    let mut y = vec![0.0; 4];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        real.matvec_add_unchecked(&short, &mut y);
    }));
    assert!(result.is_err(), "length mismatch must panic, not wrap");

    let x = vec![1.0; 4];
    let mut short_y = vec![0.0; 2];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        real.matvec_transpose_into_buf_unchecked(&x, &mut short_y);
    }));
    assert!(result.is_err(), "length mismatch must panic, not wrap");
}

#[test]
fn block_transpose_override_matches_column_reference() {
    let real = nonsymmetric_4x3();
    let x = DenseMatrix::from_row_major_vec(
        4,
        3,
        vec![
            1.0, 0.5, -1.0, 2.0, 1.5, 0.0, 3.0, -2.0, 1.0, 4.0, 2.5, -0.5,
        ],
    );
    let mut y = DenseMatrix::zeros(3, 3);
    LinearOperator::apply_transpose_block(&real, &x, &mut y);

    for j in 0..3 {
        let xj: Vec<f64> = (0..4).map(|i| x[[i, j]]).collect();
        let expected = real.try_matvec_transpose(&xj).expect("reference column");
        let got: Vec<f64> = (0..3).map(|i| y[[i, j]]).collect();
        assert_vec_eq(&got, &expected);
    }
}

#[test]
fn block_hermitian_override_matches_column_reference() {
    let complex = complex_3x3();
    let x = DenseMatrix::from_row_major_vec(
        3,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(-1.0, 0.5),
            Complex64::new(0.5, 2.0),
        ],
    );
    let mut y = DenseMatrix::zeros(3, 2);
    LinearOperator::apply_hermitian_block(&complex, &x, &mut y);

    for j in 0..2 {
        let xj: Vec<Complex64> = (0..3).map(|i| x[[i, j]]).collect();
        let expected = complex.try_matvec_hermitian(&xj).expect("reference column");
        let got: Vec<Complex64> = (0..3).map(|i| y[[i, j]]).collect();
        assert_complex_vec_eq(&got, &expected);
    }
}

fn stokes_like_4x4() -> CsrMatrix<f64> {
    let dense = DenseMatrix::from_row_major_vec(
        4,
        4,
        vec![
            4.0_f64, 1.0, 0.5, 0.0, 1.0, 4.0, 0.0, 0.5, 0.5, 0.0, 3.0, 1.0, 0.0, 0.5, 1.0, 3.0,
        ],
    );
    CsrMatrix::from_dense(&dense, 1e-15)
}

#[test]
fn bsr_into_buf_and_unchecked_match_checked_actions() {
    let bsr: BsrMatrix<f64, 2> = BsrMatrix::from_csr(&stokes_like_4x4());
    let x = vec![1.0, 2.0, 3.0, 4.0];

    let mut buf = vec![0.0; 4];
    bsr.matvec_into_buf(&x, &mut buf);
    assert_vec_eq(&buf, &bsr.matvec(&x));
    assert_vec_eq(&buf, &bsr.try_matvec(&x).expect("checked forward"));

    let mut unchecked = vec![0.0; 4];
    unsafe { bsr.matvec_into_buf_unchecked(&x, &mut unchecked) };
    assert_vec_eq(&unchecked, &buf);

    let mut transpose = vec![0.0; 4];
    bsr.matvec_transpose_into_buf(&x, &mut transpose);
    assert_vec_eq(
        &transpose,
        &bsr.try_matvec_transpose(&x).expect("checked transpose"),
    );
    let mut transpose_unchecked = vec![0.0; 4];
    unsafe { bsr.matvec_transpose_into_buf_unchecked(&x, &mut transpose_unchecked) };
    assert_vec_eq(&transpose_unchecked, &transpose);

    let mut hermitian = vec![0.0; 4];
    bsr.matvec_hermitian_into_buf(&x, &mut hermitian);
    assert_vec_eq(
        &hermitian,
        &bsr.try_matvec_hermitian(&x).expect("checked hermitian"),
    );
    let mut hermitian_unchecked = vec![0.0; 4];
    unsafe { bsr.matvec_hermitian_into_buf_unchecked(&x, &mut hermitian_unchecked) };
    assert_vec_eq(&hermitian_unchecked, &hermitian);

    // The operator surface routes through the same kernels without copying.
    let mut applied = vec![0.0; 4];
    LinearOperator::apply_into(&bsr, &x, &mut applied);
    assert_vec_eq(&applied, &buf);
    let mut applied_t = vec![0.0; 4];
    LinearOperator::apply_transpose_into(&bsr, &x, &mut applied_t);
    assert_vec_eq(&applied_t, &transpose);
}

#[test]
fn bsr_hot_path_propagates_non_finite_while_checked_rejects() {
    let mut bsr: BsrMatrix<f64, 2> = BsrMatrix::from_csr(&stokes_like_4x4());
    bsr.block_values[0] = f64::NAN;
    let x = vec![1.0, 2.0, 3.0, 4.0];

    // Infallible hot path: no panic, NaN propagates arithmetically (pinned,
    // mirroring the CSR `hot_validation_accepts_non_finite_values` contract).
    let y = bsr.matvec(&x);
    assert!(y.iter().any(|value| !value.is_finite()));

    let mut buf = vec![0.0; 4];
    bsr.matvec_into_buf(&x, &mut buf);
    assert!(buf.iter().any(|value| !value.is_finite()));

    // Checked boundary still rejects at both structure and output gates.
    assert!(matches!(
        bsr.try_matvec(&x),
        Err(SolverError::OperationFailed { .. })
    ));
}

#[test]
fn bsr_hot_path_still_rejects_corrupt_structure() {
    let mut bsr: BsrMatrix<f64, 2> = BsrMatrix::from_csr(&stokes_like_4x4());
    bsr.block_col_indices[0] = bsr.num_block_cols + 5;
    let x = vec![1.0, 2.0, 3.0, 4.0];

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| bsr.matvec(&x)));
    assert!(result.is_err(), "out-of-range block column must panic");

    let mut y = vec![0.0; 4];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        bsr.matvec_transpose_into_buf(&x, &mut y);
    }));
    assert!(result.is_err(), "out-of-range block column must panic");
}

#[test]
fn numa_hot_path_propagates_non_finite_while_checked_rejects() {
    let csr = CsrMatrix::from_raw_parts(
        6,
        6,
        vec![0, 2, 4, 6, 8, 10, 12],
        vec![0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
        vec![1.0; 12],
    );
    let mut numa = NumaPartitionedCsr::from_csr(&csr, 2);
    numa.partitions[0].values[0] = f64::INFINITY;
    let x = vec![1.0; 6];

    // Infallible hot path: no panic, non-finite propagates arithmetically.
    let y = numa.matvec(&x);
    assert!(y.iter().any(|value| !value.is_finite()));

    // Checked boundary still rejects the corrupted partition values.
    assert!(matches!(
        numa.try_matvec(&x),
        Err(NumaSpmvError::NonFinite { .. })
    ));
}

#[test]
fn numa_hot_path_still_rejects_corrupt_partitions() {
    let csr = CsrMatrix::from_raw_parts(
        6,
        6,
        vec![0, 2, 4, 6, 8, 10, 12],
        vec![0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
        vec![1.0; 12],
    );
    let mut numa = NumaPartitionedCsr::from_csr(&csr, 2);
    numa.partitions[1].row_start = 999;
    let x = vec![1.0; 6];

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| numa.matvec(&x)));
    assert!(result.is_err(), "corrupt row range must panic");
}
