use sonium_solvers::CsrMatrix;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::differentiable::DifferentiableLinearSolve;
use sonium_solvers::direct::{
    DenseDirectConfig, DenseDirectMethod, MatrixClass, Ordering, SparseDirectConfig,
    SparseDirectMethod, factorize_dense, factorize_sparse,
};
use sonium_solvers::eigen::{LobpcgConfig, lobpcg};
use sonium_solvers::nonlinear::{NewtonConfig, newton_raphson};
use sonium_solvers::optimization::{Bound, SqpConfig, sqp};
use sonium_solvers::preconditioners::IluPreconditioner;
use sonium_solvers::time_integration::{NewmarkConfig, NewmarkInitial, try_newmark_integrate};
use sonium_solvers::{GmresConfig, IdentityPreconditioner};

#[test]
fn direct_and_preconditioner_boundaries_reject_bad_inputs() {
    assert!(
        factorize_dense(
            &DenseMatrix::from_row_major_vec(1, 1, vec![f64::NAN]),
            &DenseDirectConfig {
                matrix_class: MatrixClass::General,
                method: DenseDirectMethod::Lu,
                ..Default::default()
            },
        )
        .is_err()
    );

    let mut malformed = CsrMatrix::<f64>::identity(2);
    malformed.row_ptrs[2] = 1;
    assert!(
        factorize_sparse(
            &malformed,
            &SparseDirectConfig {
                matrix_class: MatrixClass::General,
                method: SparseDirectMethod::SimplicialLu,
                ordering: Ordering::Auto,
                ..Default::default()
            },
            None,
        )
        .is_err()
    );

    let zero_pivot = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0, 1.0, 1.0, 1.0],
    );
    assert!(IluPreconditioner::try_from_csr(&zero_pivot).is_err());
}

#[test]
fn nonlinear_optimization_eigen_and_transient_boundaries_reject_bad_inputs() {
    assert!(
        newton_raphson(
            &[0.0_f64],
            |_| panic!("failure-exit matrix residual callback"),
            |_x, rhs| Ok(rhs.to_vec()),
            &NewtonConfig::default(),
        )
        .is_err()
    );

    assert!(
        sqp(
            &|x: &[f64]| x[0] * x[0],
            &|x: &[f64]| vec![2.0 * x[0]],
            &[0.0],
            &[],
            &SqpConfig {
                bounds: vec![Bound::both(1.0, 0.0)],
                ..SqpConfig::default()
            },
        )
        .is_err()
    );

    let mut nan_operator = CsrMatrix::<f64>::identity(2);
    nan_operator.values[0] = f64::NAN;
    assert!(
        lobpcg(
            &nan_operator,
            &LobpcgConfig {
                num_eigenvalues: 1,
                ..LobpcgConfig::default()
            },
        )
        .is_err()
    );

    assert!(
        try_newmark_integrate(
            None::<&CsrMatrix<f64>>,
            &CsrMatrix::identity(2),
            |_| vec![0.0],
            &NewmarkInitial {
                displacement: vec![1.0],
                velocity: vec![0.0],
                acceleration: vec![-1.0],
            },
            |_rhs| vec![0.0],
            &NewmarkConfig::average_acceleration(0.1, 1),
        )
        .is_err()
    );

    let matrix = CsrMatrix::<f64>::identity(1);
    let preconditioner = IdentityPreconditioner;
    assert!(
        DifferentiableLinearSolve::new(
            &matrix,
            &preconditioner,
            GmresConfig {
                tolerance: f64::NAN,
                ..GmresConfig::default()
            },
        )
        .is_err()
    );
}
