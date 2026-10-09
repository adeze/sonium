use num_complex::Complex64;
use sonium_solvers_core::{Preconditioner, traits::AdjointCapability};
use sonium_solvers_preconditioners::{
    AmgConfig, AmgCycle, AmgPreconditioner, AmgRestriction, AmgSmoother,
};
use sonium_solvers_sparse::CsrMatrix;

fn diagonal_matrix() -> CsrMatrix<Complex64> {
    CsrMatrix::from_raw_parts(
        8,
        8,
        (0..=8).collect(),
        (0..8).collect(),
        (0..8)
            .map(|index| Complex64::new(2.0 + index as f64, 0.0))
            .collect(),
    )
}

fn dotc(lhs: &[Complex64], rhs: &[Complex64]) -> Complex64 {
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| left.conj() * right)
        .fold(Complex64::new(0.0, 0.0), |sum, value| sum + value)
}

fn diagonal_smoother_v_cycle_exposes_exact_adjoint_actions(smoother: AmgSmoother) {
    let config = AmgConfig {
        cycle: AmgCycle::VCycle,
        restriction: AmgRestriction::ConjugateTranspose,
        smoother,
        num_pre_smooth: 2,
        num_post_smooth: 2,
        ..AmgConfig::default()
    };
    let amg = AmgPreconditioner::try_from_csr(&diagonal_matrix(), config)
        .expect("diagonal Jacobi AMG setup");

    assert_eq!(amg.transpose_capability(), AdjointCapability::Exact);
    assert_eq!(amg.hermitian_capability(), AdjointCapability::Exact);

    let r: Vec<_> = (0..8)
        .map(|index| Complex64::new(index as f64 + 0.25, -0.5 * index as f64))
        .collect();
    let s: Vec<_> = (0..8)
        .map(|index| Complex64::new(0.5 * index as f64, index as f64 + 1.0))
        .collect();
    let forward_r = amg.try_apply(&r).expect("forward AMG action");
    let forward_s = amg.try_apply(&s).expect("forward AMG action");
    let transpose_s = amg.try_apply_transpose(&s).expect("transpose AMG action");
    let hermitian_s = amg.try_apply_hermitian(&s).expect("Hermitian AMG action");

    assert!(
        transpose_s
            .iter()
            .zip(&forward_s)
            .all(|(lhs, rhs)| (*lhs - *rhs).norm() < 1.0e-12)
    );
    assert!(
        hermitian_s
            .iter()
            .zip(&forward_s)
            .all(|(lhs, rhs)| (*lhs - *rhs).norm() < 1.0e-12)
    );
    assert!((dotc(&r, &hermitian_s) - dotc(&forward_r, &s)).norm() < 1.0e-12);
}

#[test]
fn diagonal_jacobi_v_cycle_exposes_exact_adjoint_actions() {
    diagonal_smoother_v_cycle_exposes_exact_adjoint_actions(AmgSmoother::Jacobi);
}

#[test]
fn diagonal_l1_jacobi_v_cycle_exposes_exact_adjoint_actions() {
    diagonal_smoother_v_cycle_exposes_exact_adjoint_actions(AmgSmoother::L1Jacobi);
}
