use num_complex::Complex64;
use sonium_solvers_core::traits::{AdjointCapability, Preconditioner};
use sonium_solvers_preconditioners::{HiluConfig, HiluPreconditioner};
use sonium_solvers_sparse::CsrMatrix;

fn matrix(n: usize) -> CsrMatrix<Complex64> {
    let mut row_ptrs = vec![0];
    let mut columns = Vec::new();
    let mut values = Vec::new();
    for row in 0..n {
        if row > 0 {
            columns.push(row - 1);
            values.push(Complex64::new(-1.0, 0.3));
        }
        columns.push(row);
        values.push(Complex64::new(6.0 + 0.1 * (row % 3) as f64, 0.2));
        if row + 1 < n {
            columns.push(row + 1);
            values.push(Complex64::new(-0.7, -0.2));
        }
        row_ptrs.push(columns.len());
    }
    CsrMatrix::try_from_raw_parts(n, n, row_ptrs, columns, values)
        .expect("tridiagonal HILU fixture must be canonical")
}

fn vector(n: usize, seed: u64) -> Vec<Complex64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let real = ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0;
            state = state.rotate_left(17);
            let imag = ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0;
            Complex64::new(real, imag)
        })
        .collect()
}

fn transpose_inner(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    a.iter().zip(b).map(|(left, right)| *left * *right).sum()
}

fn hermitian_inner(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    a.iter()
        .zip(b)
        .map(|(left, right)| left.conj() * *right)
        .sum()
}

#[test]
fn coupled_hilu_exposes_exact_transpose_and_hermitian_actions() {
    let matrix = matrix(20);
    let config = HiluConfig {
        min_block_size: 4,
        max_levels: None,
    };
    let preconditioner =
        HiluPreconditioner::try_from_csr_coupled_with_config(&matrix, &config, 0.0)
            .expect("coupled HILU fixture should build");
    assert!(preconditioner.is_coupled());
    assert_eq!(
        preconditioner.transpose_capability(),
        AdjointCapability::Exact
    );
    assert_eq!(
        preconditioner.hermitian_capability(),
        AdjointCapability::Exact
    );

    for seed in 0..16_u64 {
        let r = vector(matrix.num_rows, 0xA5A5_0000 + seed);
        let s = vector(matrix.num_rows, 0x5A5A_0000 + seed);
        let mr = preconditioner.apply(&r);
        let mt_s = preconditioner
            .try_apply_transpose(&s)
            .expect("coupled HILU transpose action should be supported");
        let mh_s = preconditioner
            .try_apply_hermitian(&s)
            .expect("coupled HILU Hermitian action should be supported");
        let mut mh_s_into = vec![Complex64::new(0.0, 0.0); matrix.num_rows];
        preconditioner.apply_hermitian_into(&s, &mut mh_s_into);
        let in_place_error = mh_s
            .iter()
            .zip(&mh_s_into)
            .map(|(expected, actual)| (*expected - *actual).norm())
            .fold(0.0, f64::max);
        let transpose_error = (transpose_inner(&mr, &s) - transpose_inner(&r, &mt_s)).norm();
        let hermitian_error = (hermitian_inner(&mr, &s) - hermitian_inner(&r, &mh_s)).norm();
        assert!(
            transpose_error < 1.0e-10,
            "coupled HILU transpose mismatch for seed {seed}: {transpose_error:e}"
        );
        assert!(
            hermitian_error < 1.0e-10,
            "coupled HILU Hermitian mismatch for seed {seed}: {hermitian_error:e}"
        );
        assert!(
            in_place_error < 1.0e-12,
            "coupled HILU in-place Hermitian mismatch for seed {seed}: {in_place_error:e}"
        );
    }
}
