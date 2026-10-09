use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::sparse::{BsrMatrix, CsrMatrix, FnOperator};
use sonium_solvers::{FgmresConfig, GmresConfig, SolverError, gmres, try_fgmres};

#[test]
fn test_fn_operator_with_krylov() {
    let n = 20;
    let op = FnOperator::new(n, n, move |x: &[f64]| {
        let mut y = vec![0.0; n];
        for i in 0..n {
            y[i] = 2.0 * x[i];
            if i > 0 {
                y[i] -= x[i - 1];
            }
            if i < n - 1 {
                y[i] -= x[i + 1];
            }
        }
        y
    });

    let b = vec![1.0_f64; n];
    let config = GmresConfig {
        max_iterations: 100,
        tolerance: 1e-10,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        ..Default::default()
    };
    let result = gmres(&op, &b, &config);
    assert!(
        result.converged,
        "Matrix-free GMRES should converge: {} iters",
        result.iterations
    );
}

#[test]
fn test_checked_fgmres_rejects_non_finite_matrix_free_callback() {
    let op = FnOperator::new(2, 2, |_x: &[f64]| vec![f64::NAN; 2]);
    let config = FgmresConfig {
        max_iterations: 4,
        restart: 2,
        tolerance: 1e-8,
        ..Default::default()
    };

    let error = try_fgmres(&op, &[1.0, 0.0], &config)
        .expect_err("checked FGMRES must reject non-finite matrix-free output");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn test_bsr_with_krylov_solver() {
    let dense = DenseMatrix::from_row_major_vec(
        4,
        4,
        vec![
            4.0_f64, 1.0, 0.5, 0.0, 1.0, 4.0, 0.0, 0.5, 0.5, 0.0, 4.0, 1.0, 0.0, 0.5, 1.0, 4.0,
        ],
    );
    let csr = CsrMatrix::from_dense(&dense, 1e-15);
    let bsr: BsrMatrix<f64, 2> = BsrMatrix::from_csr(&csr);

    let b = vec![1.0_f64, 2.0, 3.0, 4.0];
    let config = GmresConfig {
        max_iterations: 100,
        tolerance: 1e-12,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        ..Default::default()
    };

    let result = gmres(&bsr, &b, &config);
    assert!(
        result.converged,
        "GMRES with BSR should converge: {} iters, res {:.2e}",
        result.iterations, result.residual
    );

    let bx = bsr.matvec(&result.x);
    let r_norm: f64 = b
        .iter()
        .zip(bx.iter())
        .map(|(bi, bxi)| (bi - bxi) * (bi - bxi))
        .sum::<f64>()
        .sqrt();
    assert!(r_norm < 1e-10, "Residual too large: {r_norm:.2e}");
}
