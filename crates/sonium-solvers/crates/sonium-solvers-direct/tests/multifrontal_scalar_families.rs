//! Scalar-family coverage for the multifrontal BLR and LDLT paths.
//!
//! The existing parity tests exercise the complex-double path on larger
//! fixtures.  This test keeps a small, well-conditioned symmetric fixture and
//! runs both checked multifrontal implementations through every scalar family
//! supported by the direct-solver API.  The residual is recomputed from the
//! CSR entries rather than trusting solver-reported diagnostics.

use num_complex::{Complex32, Complex64};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_direct::{
    BlrConfig, LdltBlrConfig, LdltConfig, MultifrontalBlrFactorization,
    MultifrontalLdltFactorization, PanelKind,
};
use sonium_solvers_sparse::CsrMatrix;

fn scalar<T: ComplexField>(value: f64) -> T {
    T::from_real(T::Real::from_f64(value).expect("test value representable"))
}

fn matrix<T: ComplexField>() -> CsrMatrix<T> {
    // Symmetric positive definite 3x3 matrix.  Keeping all entries real makes
    // the same fixture valid for the Hermitian LDLT mode while still testing
    // complex storage and arithmetic for Complex32/Complex64.
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![
            scalar(6.0),
            scalar(1.0),
            scalar(0.5),
            scalar(1.0),
            scalar(5.0),
            scalar(0.75),
            scalar(0.5),
            scalar(0.75),
            scalar(4.0),
        ],
    )
}

fn matvec<T: ComplexField>(matrix: &CsrMatrix<T>, x: &[T]) -> Vec<T> {
    (0..matrix.num_rows)
        .map(|row| {
            (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1]).fold(T::zero(), |sum, index| {
                sum + matrix.values[index] * x[matrix.col_indices[index]]
            })
        })
        .collect()
}

fn residual<T: ComplexField>(matrix: &CsrMatrix<T>, x: &[T], rhs: &[T]) -> f64 {
    let ax = matvec(matrix, x);
    let residual_norm = ax
        .iter()
        .zip(rhs)
        .fold(T::Real::zero(), |sum, (actual, expected)| {
            sum + (*actual - *expected).norm_sqr()
        })
        .sqrt();
    let rhs_norm = rhs
        .iter()
        .fold(T::Real::zero(), |sum, value| sum + value.norm_sqr())
        .sqrt();
    (residual_norm / rhs_norm.max(T::Real::one()))
        .to_f64()
        .expect("test residual representable")
}

fn check_matrix<T: ComplexField + FromPrimitive>(matrix: CsrMatrix<T>, tolerance: f64) {
    let expected: Vec<_> = (0..matrix.num_rows)
        .map(|index| match index % 3 {
            0 => scalar(1.0),
            1 => scalar(-2.0),
            _ => scalar(0.5),
        })
        .collect();
    let rhs = matvec(&matrix, &expected);

    let blr = MultifrontalBlrFactorization::<T>::factorize_with_config(
        &matrix,
        &BlrConfig {
            epsilon: 0.0,
            block_size: 2,
            min_block_size: 1,
            ..BlrConfig::default()
        },
    )
    .expect("BLR factorization");
    let blr_solution = blr.try_solve(&rhs).expect("BLR solve");
    assert!(
        residual(&matrix, &blr_solution, &rhs) <= tolerance,
        "BLR residual exceeded tolerance: {}",
        residual(&matrix, &blr_solution, &rhs)
    );

    let ldlt = MultifrontalLdltFactorization::<T>::factorize_with_config(
        &matrix,
        &LdltConfig {
            min_block_size: 1,
            blr: Some(LdltBlrConfig {
                epsilon: 0.0,
                block_size: 2,
                panel_kind: PanelKind::Blr,
            }),
            ..LdltConfig::default()
        },
    )
    .expect("LDLT factorization");
    let ldlt_solution = ldlt.try_solve(&rhs).expect("LDLT solve");
    assert!(
        residual(&matrix, &ldlt_solution, &rhs) <= tolerance,
        "LDLT residual exceeded tolerance: {}",
        residual(&matrix, &ldlt_solution, &rhs)
    );
}

fn complex32_fixture() -> CsrMatrix<Complex32> {
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![
            Complex32::new(6.0, 0.0),
            Complex32::new(1.0, 0.5),
            Complex32::new(0.5, -0.25),
            Complex32::new(1.0, -0.5),
            Complex32::new(5.0, 0.0),
            Complex32::new(0.75, 0.125),
            Complex32::new(0.5, 0.25),
            Complex32::new(0.75, -0.125),
            Complex32::new(4.0, 0.0),
        ],
    )
}

fn complex64_fixture() -> CsrMatrix<Complex64> {
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![
            Complex64::new(6.0, 0.0),
            Complex64::new(1.0, 0.5),
            Complex64::new(0.5, -0.25),
            Complex64::new(1.0, -0.5),
            Complex64::new(5.0, 0.0),
            Complex64::new(0.75, 0.125),
            Complex64::new(0.5, 0.25),
            Complex64::new(0.75, -0.125),
            Complex64::new(4.0, 0.0),
        ],
    )
}

fn seeded_spd<T: ComplexField>(seed: u64) -> CsrMatrix<T> {
    let n = 6;
    let mut state = seed.wrapping_add(1);
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let unit = ((state >> 11) as f64) / ((1_u64 << 53) as f64);
        0.18 * (2.0 * unit - 1.0)
    };
    let mut values = vec![vec![0.0_f64; n]; n];
    for row in 0..n {
        for col in (row + 1)..n {
            let value = next();
            values[row][col] = value;
            values[col][row] = value;
        }
    }
    for row in 0..n {
        values[row][row] = 2.0 + values[row].iter().map(|value| value.abs()).sum::<f64>();
    }
    let mut triplets = Vec::with_capacity(n * n);
    for row in 0..n {
        for col in 0..n {
            triplets.push((row, col, scalar::<T>(values[row][col])));
        }
    }
    CsrMatrix::from_triplets(n, n, triplets)
}

#[test]
fn multifrontal_real_scalar_families() {
    check_matrix(matrix::<f32>(), 3.0e-5);
    check_matrix(matrix::<f64>(), 3.0e-13);
}

#[test]
fn multifrontal_complex_scalar_families() {
    check_matrix(complex32_fixture(), 3.0e-5);
    check_matrix(complex64_fixture(), 3.0e-13);
}

#[test]
fn multifrontal_deterministic_spd_matrix_bank() {
    for seed in 0..8 {
        check_matrix(seeded_spd::<f64>(seed), 3.0e-13);
        check_matrix(seeded_spd::<Complex64>(0xC0DE_0000 + seed), 3.0e-13);
    }
}
