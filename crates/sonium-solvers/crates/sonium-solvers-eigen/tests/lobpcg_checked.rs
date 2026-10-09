use sonium_solvers_core::{
    dense::DenseMatrix,
    traits::{AccuracyStatus, LinearOperator, Preconditioner, SolverError, SolverStatus},
};
use sonium_solvers_eigen::{
    LobpcgConfig, LobpcgResult, lobpcg, lobpcg_generalized, lobpcg_preconditioned,
};
use sonium_solvers_sparse::CsrMatrix;
use std::sync::atomic::{AtomicUsize, Ordering};

fn diagonal(values: &[f64]) -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(
        values.len(),
        values.len(),
        values
            .iter()
            .enumerate()
            .map(|(index, &value)| (index, index, value))
            .collect(),
    )
}

#[test]
fn lobpcg_rejects_invalid_configuration_before_iteration() {
    let matrix = CsrMatrix::identity(2);
    let configs = [
        LobpcgConfig {
            num_eigenvalues: 0,
            ..Default::default()
        },
        LobpcgConfig {
            max_iterations: 0,
            ..Default::default()
        },
        LobpcgConfig {
            tolerance: f64::NAN,
            ..Default::default()
        },
        LobpcgConfig {
            num_eigenvalues: 3,
            ..Default::default()
        },
    ];

    for config in configs {
        let error = lobpcg(&matrix, &config).expect_err("invalid LOBPCG input must be typed");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }
}

struct NonFiniteOperator;

impl LinearOperator<f64> for NonFiniteOperator {
    fn num_rows(&self) -> usize {
        2
    }

    fn num_cols(&self) -> usize {
        2
    }

    fn apply_into(&self, _x: &[f64], y: &mut [f64]) {
        y.fill(f64::NAN);
    }

    fn apply_transpose_into(&self, _x: &[f64], y: &mut [f64]) {
        y.fill(f64::NAN);
    }
}

#[test]
fn lobpcg_rejects_non_finite_operator_output_before_iteration() {
    let config = LobpcgConfig {
        num_eigenvalues: 1,
        ..Default::default()
    };
    let error = lobpcg(&NonFiniteOperator, &config)
        .expect_err("non-finite operator output must be a typed failure");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[derive(Clone, Copy)]
enum LaterFailure {
    Panic,
    WrongLength,
    NonFinite,
}

struct LaterFailingOperator {
    calls: AtomicUsize,
    fail_at: usize,
    failure: LaterFailure,
    diagonal: Vec<f64>,
}

impl LaterFailingOperator {
    fn new(diagonal: Vec<f64>, fail_at: usize, failure: LaterFailure) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            fail_at,
            failure,
            diagonal,
        }
    }
}

impl LinearOperator<f64> for LaterFailingOperator {
    fn num_rows(&self) -> usize {
        self.diagonal.len()
    }

    fn num_cols(&self) -> usize {
        self.diagonal.len()
    }

    fn apply(&self, x: &[f64]) -> Vec<f64> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if call == self.fail_at {
            match self.failure {
                LaterFailure::Panic => panic!("later matrix-free failure"),
                LaterFailure::WrongLength => return vec![0.0; x.len() - 1],
                LaterFailure::NonFinite => return vec![f64::NAN; x.len()],
            }
        }
        self.diagonal
            .iter()
            .zip(x)
            .map(|(diagonal, value)| diagonal * value)
            .collect()
    }

    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(&self.apply(x));
    }

    fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
        for ((output, diagonal), input) in y.iter_mut().zip(&self.diagonal).zip(x) {
            *output = diagonal * input;
        }
    }
}

fn single_pair_config() -> LobpcgConfig {
    LobpcgConfig {
        num_eigenvalues: 1,
        max_iterations: 4,
        tolerance: 1e-14,
        block_padding: 0,
    }
}

#[test]
fn lobpcg_propagates_later_operator_callback_failures() {
    for failure in [
        LaterFailure::Panic,
        LaterFailure::WrongLength,
        LaterFailure::NonFinite,
    ] {
        // Call 1 is the validation probe, call 2 forms the initial A*X block,
        // and call 3 is the first iterative A*W application.
        let operator = LaterFailingOperator::new(vec![1.0, 2.0, 3.0, 4.0], 3, failure);
        let error = lobpcg(&operator, &single_pair_config())
            .expect_err("a later matrix-free failure must stay inside Result");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
    }
}

struct LaterPanicPreconditioner {
    calls: AtomicUsize,
}

impl Preconditioner<f64> for LaterPanicPreconditioner {
    fn apply_into(&self, r: &[f64], z: &mut [f64]) {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        assert_ne!(call, 2, "later preconditioner failure");
        z.copy_from_slice(r);
    }
}

#[test]
fn lobpcg_propagates_later_preconditioner_panic() {
    let operator =
        LaterFailingOperator::new(vec![1.0, 2.0, 3.0, 4.0], usize::MAX, LaterFailure::Panic);
    let preconditioner = LaterPanicPreconditioner {
        calls: AtomicUsize::new(0),
    };

    let error = lobpcg_preconditioned(&operator, &preconditioner, &single_pair_config())
        .expect_err("a later preconditioner panic must stay inside Result");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn lobpcg_propagates_later_mass_operator_panic() {
    let operator =
        LaterFailingOperator::new(vec![1.0, 2.0, 3.0, 4.0], usize::MAX, LaterFailure::Panic);
    // Call 1 is the validation probe, call 2 forms the initial B*X block,
    // and call 3 is the first iterative B*W application.
    let mass = LaterFailingOperator::new(vec![1.0; 4], 3, LaterFailure::Panic);

    let error = lobpcg_generalized(&operator, &mass, &single_pair_config())
        .expect_err("a later mass-operator panic must stay inside Result");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

struct OverflowingFiniteOperator {
    dimension: usize,
}

impl LinearOperator<f64> for OverflowingFiniteOperator {
    fn num_rows(&self) -> usize {
        self.dimension
    }

    fn num_cols(&self) -> usize {
        self.dimension
    }

    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        for (output, input) in y.iter_mut().zip(x) {
            *output = f64::MAX * input;
        }
    }

    fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
        self.apply_into(x, y);
    }
}

#[test]
fn lobpcg_rejects_non_finite_generated_rayleigh_quotient() {
    let operator = OverflowingFiniteOperator { dimension: 128 };
    let error = lobpcg(&operator, &single_pair_config())
        .expect_err("finite callback output must not hide an overflowing quotient");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}

#[test]
fn lobpcg_result_contract_reports_convergence_and_thresholded_accuracy() {
    let result = lobpcg(
        &diagonal(&[1.0, 3.0, 5.0]),
        &LobpcgConfig {
            num_eigenvalues: 1,
            max_iterations: 80,
            tolerance: 1.0e-8,
            block_padding: 0,
        },
    )
    .expect("LOBPCG should converge on a diagonal SPD fixture");

    assert_eq!(result.status(), SolverStatus::Converged);
    assert_eq!(result.accuracy_status(), AccuracyStatus::Attained);
    assert!(result.residual_norms[0] <= 1.0e-8);
}

#[test]
fn lobpcg_result_contract_reports_max_iterations_without_accuracy() {
    let result = lobpcg(
        &diagonal(&[1.0, 3.0, 5.0]),
        &LobpcgConfig {
            num_eigenvalues: 1,
            max_iterations: 1,
            tolerance: 1.0e-14,
            block_padding: 0,
        },
    )
    .expect("LOBPCG should return its best finite iterate at the iteration cap");

    assert_eq!(result.status(), SolverStatus::MaxIterationsReached);
    assert_eq!(result.accuracy_status(), AccuracyStatus::NotAttained);
    assert!(result.residual_norms[0] > 1.0e-14);
}

struct ZeroPreconditioner;

impl Preconditioner<f64> for ZeroPreconditioner {
    fn apply_into(&self, r: &[f64], z: &mut [f64]) {
        assert_eq!(r.len(), z.len());
        z.fill(0.0);
    }
}

#[test]
fn lobpcg_maps_stagnant_search_space_to_typed_breakdown() {
    let error = lobpcg_preconditioned(
        &diagonal(&[1.0, 3.0, 5.0]),
        &ZeroPreconditioner,
        &LobpcgConfig {
            num_eigenvalues: 1,
            max_iterations: 8,
            tolerance: 1.0e-14,
            block_padding: 0,
        },
    )
    .expect_err("a zero preconditioner must produce an explicit LOBPCG breakdown");

    assert!(matches!(
        error,
        SolverError::Breakdown {
            operation: "LOBPCG iteration",
            ..
        }
    ));
    assert_eq!(error.status(), SolverStatus::Breakdown);
    assert_eq!(error.accuracy_status(), AccuracyStatus::NotChecked);
}

struct BreakdownOperator;

impl LinearOperator<f64> for BreakdownOperator {
    fn num_rows(&self) -> usize {
        2
    }

    fn num_cols(&self) -> usize {
        2
    }

    fn try_apply(&self, _x: &[f64]) -> Result<Vec<f64>, SolverError> {
        Err(SolverError::Breakdown {
            operation: "inner operator",
            reason: "synthetic factor failure".into(),
        })
    }

    fn apply_into(&self, _x: &[f64], _y: &mut [f64]) {
        panic!("the checked callback must fail before the unchecked callback");
    }

    fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
        panic!("transpose is not used by LOBPCG validation");
    }
}

#[test]
fn lobpcg_preserves_nested_breakdown_at_the_operator_boundary() {
    let error = lobpcg(
        &BreakdownOperator,
        &LobpcgConfig {
            num_eigenvalues: 1,
            ..Default::default()
        },
    )
    .expect_err("nested operator breakdown must remain typed");

    assert!(matches!(
        error,
        SolverError::Breakdown {
            operation: "operator",
            ..
        }
    ));
}

struct RectangularOperator;

impl LinearOperator<f64> for RectangularOperator {
    fn num_rows(&self) -> usize {
        2
    }

    fn num_cols(&self) -> usize {
        3
    }

    fn apply_into(&self, _x: &[f64], y: &mut [f64]) {
        y.fill(0.0);
    }

    fn apply_transpose_into(&self, _x: &[f64], y: &mut [f64]) {
        y.fill(0.0);
    }
}

#[test]
fn lobpcg_reports_dimension_failures_with_canonical_status() {
    let error = lobpcg(
        &RectangularOperator,
        &LobpcgConfig {
            num_eigenvalues: 1,
            ..Default::default()
        },
    )
    .expect_err("non-square operators must be rejected as a dimension failure");

    assert!(matches!(
        error,
        SolverError::DimensionMismatch {
            expected: 2,
            got: 3
        }
    ));
    assert_eq!(error.status(), SolverStatus::NumericalError);
    assert_eq!(error.accuracy_status(), AccuracyStatus::NotChecked);
}

#[test]
fn lobpcg_result_contract_rejects_non_finite_metadata_as_numerical() {
    let result = LobpcgResult {
        eigenvalues: vec![1.0],
        eigenvectors: DenseMatrix::from_row_major_vec(2, 1, vec![1.0, f64::NAN]),
        iterations: 1,
        residual_norms: vec![0.0],
        converged: true,
    };

    assert_eq!(result.status(), SolverStatus::NumericalError);
    assert_eq!(result.accuracy_status(), AccuracyStatus::NotChecked);
}
