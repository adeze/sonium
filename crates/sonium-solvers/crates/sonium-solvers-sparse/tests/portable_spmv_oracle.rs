//! Portable SpMV oracle test (runs on every architecture).
//!
//! The x86-only `native_x86_sparse_evidence` test pins AVX2 dispatch; this
//! test pins *numerical* correctness portably: for all four scalar types and
//! every [`SpmvPlanPreference`], forward SpMV through the selected plan must
//! match an independent scalar COO oracle, and CSR transpose/Hermitian must
//! match the transposed oracle. On aarch64 this exercises the NEON CSR
//! kernel; with `SONIUM_SPARSE_FORCE_SCALAR=1` it exercises the scalar
//! fallback on any machine.

use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, ToPrimitive};
use sonium_solvers_core::traits::{ComplexField, LinearOperator};
use sonium_solvers_sparse::{
    CsrMatrix, SpmvPlan, SpmvPlanConfig, SpmvPlanPreference, TripletMatrix,
};

const ROWS: usize = 7;
const COLS: usize = 11;

/// (row, col, re, im) fixture: empty row 3, duplicate accumulation at
/// (0,0), unsorted insertion order, irregular row lengths.
const ENTRIES: [(usize, usize, f64, f64); 14] = [
    (5, 10, 2.5, -1.0),
    (0, 0, 1.0, 0.5),
    (1, 3, -2.0, 1.5),
    (0, 0, 0.25, -0.75),
    (2, 1, 3.0, 0.0),
    (1, 9, 0.5, 2.0),
    (4, 4, -1.25, -0.5),
    (6, 0, 1.75, 1.25),
    (2, 10, -0.5, -2.5),
    (5, 2, 4.0, 0.25),
    (6, 10, 1.0, -1.0),
    (4, 0, 0.0, 3.0),
    (1, 3, 1.0, 1.0),
    (5, 5, -3.5, 0.0),
];

const PREFERENCES: [SpmvPlanPreference; 9] = [
    SpmvPlanPreference::Auto,
    SpmvPlanPreference::Csr,
    SpmvPlanPreference::CompactCsrU32,
    SpmvPlanPreference::NumaCsr,
    SpmvPlanPreference::RowBucketedCsr,
    SpmvPlanPreference::SellCSigma,
    SpmvPlanPreference::Bsr2,
    SpmvPlanPreference::Bsr3,
    SpmvPlanPreference::Bsr4,
];

fn value<T: ComplexField>(re: f64, im: f64) -> T {
    T::from_re_im(
        T::Real::from_f64(re).expect("fixture value is representable"),
        T::Real::from_f64(im).expect("fixture value is representable"),
    )
}

fn fixture<T: ComplexField + Default>() -> (CsrMatrix<T>, Vec<(usize, usize, T)>) {
    let mut triplets = TripletMatrix::new(ROWS, COLS);
    let mut entries = Vec::with_capacity(ENTRIES.len());
    for (row, col, re, im) in ENTRIES {
        let v = value(re, im);
        triplets.push(row, col, v);
        entries.push((row, col, v));
    }
    (triplets.to_csr(), entries)
}

fn input<T: ComplexField>(len: usize) -> Vec<T> {
    (0..len)
        .map(|i| value((i as f64) * 0.5 - 1.0, (i as f64) * 0.25 + 0.5))
        .collect()
}

/// Independent scalar oracle: y = op(A)·x from the raw entry list with
/// duplicate coalescing (mirrors `to_csr` accumulation semantics).
fn oracle<T: ComplexField>(
    entries: &[(usize, usize, T)],
    x: &[T],
    adjoint: bool,
    conj: bool,
) -> Vec<T> {
    let (out_len, in_len) = if adjoint { (COLS, ROWS) } else { (ROWS, COLS) };
    assert_eq!(x.len(), in_len);
    let mut y = vec![T::zero(); out_len];
    for (row, col, v) in entries {
        let (o, i) = if adjoint { (col, row) } else { (row, col) };
        let a = if conj { v.conj() } else { *v };
        y[*o] += a * x[*i];
    }
    y
}

fn max_abs_diff<T: ComplexField>(a: &[T], b: &[T]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).norm().to_f64().unwrap_or(f64::INFINITY))
        .fold(0.0, f64::max)
}

fn check_type<T: ComplexField + Default>(tol: f64) {
    let (csr, entries) = fixture::<T>();
    let x = input::<T>(COLS);
    let xt = input::<T>(ROWS);
    let expected_fwd = oracle(&entries, &x, false, false);
    let expected_trans = oracle(&entries, &xt, true, false);
    let expected_herm = oracle(&entries, &xt, true, true);

    // CSR reference paths (SIMD-dispatched on the host arch).
    assert!(
        max_abs_diff(&csr.matvec(&x), &expected_fwd) < tol,
        "CSR forward mismatch"
    );
    assert!(
        max_abs_diff(&csr.matvec_transpose(&xt), &expected_trans) < tol,
        "CSR transpose mismatch"
    );
    assert!(
        max_abs_diff(&csr.matvec_hermitian(&xt), &expected_herm) < tol,
        "CSR Hermitian mismatch"
    );

    // Every plan layout must reproduce the same forward action.
    for preference in PREFERENCES {
        let config = SpmvPlanConfig {
            preference,
            ..SpmvPlanConfig::default()
        };
        let plan = SpmvPlan::try_from_csr(&csr, config).expect("plan must build");
        let got = plan.apply(&x);
        assert!(
            max_abs_diff(&got, &expected_fwd) < tol,
            "plan {preference:?} (format {:?}) forward mismatch",
            plan.format()
        );
    }
}

#[test]
fn portable_spmv_oracle_f32() {
    check_type::<f32>(1e-5);
}

#[test]
fn portable_spmv_oracle_f64() {
    check_type::<f64>(1e-12);
}

#[test]
fn portable_spmv_oracle_complex32() {
    check_type::<Complex32>(1e-4);
}

#[test]
fn portable_spmv_oracle_complex64() {
    check_type::<Complex64>(1e-12);
}
