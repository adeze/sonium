use num_complex::Complex32;
use sonium_solvers_core::traits::{AdjointCapability, Preconditioner};
use sonium_solvers_preconditioners::{DiagonalPreconditioner, IluPreconditioner};
use sonium_solvers_sparse::CsrMatrix;

fn inner_conjugate(left: &[Complex32], right: &[Complex32]) -> Complex32 {
    left.iter().zip(right).map(|(a, b)| a.conj() * *b).sum()
}

fn inner_plain(left: &[Complex32], right: &[Complex32]) -> Complex32 {
    left.iter().zip(right).map(|(a, b)| *a * *b).sum()
}

fn assert_hermitian_contract<P>(preconditioner: &P, r: &[Complex32], s: &[Complex32])
where
    P: Preconditioner<Complex32>,
{
    assert_eq!(
        preconditioner.transpose_capability(),
        AdjointCapability::Exact
    );
    assert_eq!(
        preconditioner.hermitian_capability(),
        AdjointCapability::Exact
    );
    let forward = preconditioner.apply(r);
    let transpose = preconditioner
        .try_apply_transpose(s)
        .expect("Complex32 transpose action should be supported");
    let hermitian = preconditioner
        .try_apply_hermitian(s)
        .expect("Complex32 Hermitian action should be supported");
    let transpose_lhs = inner_plain(&forward, s);
    let transpose_rhs = inner_plain(r, &transpose);
    assert!(
        (transpose_lhs - transpose_rhs).norm() <= 3.0e-5,
        "{transpose_lhs:?} != {transpose_rhs:?}"
    );
    let lhs = inner_conjugate(&forward, s);
    let rhs = inner_conjugate(r, &hermitian);
    assert!((lhs - rhs).norm() <= 3.0e-5, "{lhs:?} != {rhs:?}");
}

fn matrix() -> CsrMatrix<Complex32> {
    CsrMatrix::try_from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            Complex32::new(4.0, 0.5),
            Complex32::new(0.8, -0.2),
            Complex32::new(0.3, 0.1),
            Complex32::new(3.0, -0.75),
            Complex32::new(0.7, 0.2),
            Complex32::new(0.4, -0.15),
            Complex32::new(2.5, 0.3),
        ],
    )
    .expect("canonical Complex32 matrix")
}

fn vector(seed: u32) -> Vec<Complex32> {
    let mut state = seed;
    (0..3)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let real = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            state = state.rotate_left(11);
            let imag = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            Complex32::new(real, imag)
        })
        .collect()
}

#[test]
fn complex32_diagonal_and_ilu_preserve_hermitian_adjoint_identity() {
    let diagonal = DiagonalPreconditioner::try_from_diagonal(&[
        Complex32::new(4.0, 0.5),
        Complex32::new(3.0, -0.75),
        Complex32::new(2.5, 0.3),
    ])
    .expect("Complex32 diagonal preconditioner");
    let ilu = IluPreconditioner::try_from_csr(&matrix()).expect("Complex32 ILU preconditioner");

    for seed in 0..32_u32 {
        let r = vector(0xA5A5_0000_u32.wrapping_add(seed));
        let s = vector(0x5A5A_0000_u32.wrapping_add(seed));
        assert_hermitian_contract(&diagonal, &r, &s);
        assert_hermitian_contract(&ilu, &r, &s);
    }
}
