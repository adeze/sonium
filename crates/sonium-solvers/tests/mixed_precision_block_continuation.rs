use half::f16;
use sonium_solvers::BlockFgmresConfig;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::mixed_precision::try_mixed_precision_block_solve;
use sonium_solvers::preconditioners::DiagonalPreconditioner;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::traits::IdentityPreconditioner;

fn config(true_residual_check: bool) -> BlockFgmresConfig<f64> {
    BlockFgmresConfig {
        max_iterations: 80,
        restart: 30,
        tolerance: 1.0e-12,
        true_residual_check,
        exact_residual_check_interval: 0,
    }
}

fn spd_with_spectrum(n: usize, kappa: f64) -> CsrMatrix<f64> {
    let q = |i: usize, k: usize| {
        if k == 0 {
            1.0 / (n as f64).sqrt()
        } else {
            (2.0 / n as f64).sqrt()
                * (std::f64::consts::PI * k as f64 * (2 * i + 1) as f64 / (2 * n) as f64).cos()
        }
    };
    let lambda = |k: usize| kappa.powf(k as f64 / (n - 1) as f64);
    let mut dense = DenseMatrix::from_elem(n, n, 0.0_f64);
    for i in 0..n {
        for j in 0..n {
            dense[[i, j]] = (0..n).map(|k| q(i, k) * lambda(k) * q(j, k)).sum();
        }
    }
    CsrMatrix::from_dense(&dense, 1.0e-300)
}

#[test]
fn mixed_precision_block_forces_true_residual_gate_after_low_cast_rounding() {
    let operator = spd_with_spectrum(128, 1.0e5);
    let x_exact: Vec<f64> = (0..128).map(|i| (i as f64).sin()).collect();
    let values = operator.matvec(&x_exact);
    let mut rhs = DenseMatrix::from_elem(128, 1, 0.0_f64);
    for (row, value) in values.into_iter().enumerate() {
        rhs[[row, 0]] = value;
    }

    // f16 rounds the small component while preserving the dominant one.
    // Without the mixed-precision block policy, the projected residual is
    // zero and disabling the block gate incorrectly publishes convergence
    // for the original f64 operator.
    let low_matrix = operator.cast::<f16>();
    let result = try_mixed_precision_block_solve::<f64, f16, _>(
        &operator,
        &rhs,
        Box::new(DiagonalPreconditioner::from_csr(&low_matrix)),
        &config(false),
    )
    .expect("finite underflow is not a low-precision failure");
    assert!(!result.converged);
    assert!(result.true_residuals[0] > result.residuals[0]);
    assert!(result.residuals[0] <= config(false).tolerance);
    assert!(result.true_residuals[0] > config(false).tolerance);
    assert!(result.op_counts.exact_residual_checks > 0);
}

#[test]
fn mixed_precision_block_still_accepts_a_valid_exact_candidate() {
    let operator = CsrMatrix::identity(1);
    let rhs = DenseMatrix::from_elem(1, 1, 1.0_f64);
    let result = try_mixed_precision_block_solve::<f64, f16, _>(
        &operator,
        &rhs,
        Box::new(IdentityPreconditioner),
        &config(false),
    )
    .expect("identity block solve");

    assert!(result.converged);
    assert!(result.true_residuals[0] <= 1.0e-12);
    assert!(result.op_counts.exact_residual_checks > 0);
}
