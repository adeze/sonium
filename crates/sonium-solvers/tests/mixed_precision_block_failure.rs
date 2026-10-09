use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::mixed_precision::try_mixed_precision_block_solve;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::traits::{LinearOperator, Preconditioner};
use sonium_solvers::{BlockFgmresConfig, SolverError};
use std::sync::atomic::{AtomicUsize, Ordering};

struct NonFinitePreconditioner;

impl Preconditioner<f32> for NonFinitePreconditioner {
    fn apply(&self, r: &[f32]) -> Vec<f32> {
        vec![f32::NAN; r.len()]
    }
}

#[test]
fn mixed_precision_block_solver_reports_low_precision_failure_provenance() {
    let operator = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![4.0_f64, -1.0, -1.0, 4.0],
    );
    let rhs = DenseMatrix::from_row_major_vec(2, 2, vec![3.0, 1.0, 1.0, 3.0]);
    let config = BlockFgmresConfig {
        max_iterations: 8,
        restart: 2,
        tolerance: 1.0e-10,
        true_residual_check: true,
        exact_residual_check_interval: 1,
    };

    let error = try_mixed_precision_block_solve(
        &operator,
        &rhs,
        Box::new(NonFinitePreconditioner),
        &config,
    )
    .expect_err("non-finite low-precision block application must fail closed");
    match error {
        SolverError::OperationFailed { operation, reason } => {
            assert_eq!(operation, "mixed-precision block FGMRES preconditioner");
            assert!(reason.contains("non-finite intermediate value"));
            assert!(reason.contains("1 failures"));
        }
        other => panic!("unexpected mixed-precision block error: {other:?}"),
    }
}

struct CountingOperator {
    inner: CsrMatrix<f64>,
    calls: AtomicUsize,
}

impl LinearOperator<f64> for CountingOperator {
    fn num_rows(&self) -> usize {
        self.inner.num_rows()
    }

    fn num_cols(&self) -> usize {
        self.inner.num_cols()
    }

    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.apply_into(x, y);
    }

    fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.apply_transpose_into(x, y);
    }

    fn apply_block(&self, x: &DenseMatrix<f64>, y: &mut DenseMatrix<f64>) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.apply_block(x, y);
    }
}

#[test]
fn mixed_precision_block_failure_stops_before_failed_correction_operator() {
    let operator = CountingOperator {
        inner: CsrMatrix::from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![4.0_f64, -1.0, -1.0, 4.0],
        ),
        calls: AtomicUsize::new(0),
    };
    let rhs = DenseMatrix::from_row_major_vec(2, 2, vec![3.0, 1.0, 1.0, 3.0]);
    let config = BlockFgmresConfig {
        max_iterations: 8,
        restart: 2,
        tolerance: 1.0e-10,
        true_residual_check: true,
        exact_residual_check_interval: 1,
    };

    let error = try_mixed_precision_block_solve(
        &operator,
        &rhs,
        Box::new(NonFinitePreconditioner),
        &config,
    )
    .expect_err("non-finite low-precision block application must fail closed");
    assert!(matches!(
        error,
        SolverError::OperationFailed { operation, .. }
            if operation == "mixed-precision block FGMRES preconditioner"
    ));
    // The initial residual uses one forward action per RHS. The failed
    // preconditioned correction is rejected before it reaches A.
    assert_eq!(operator.calls.load(Ordering::Relaxed), 2);
}
