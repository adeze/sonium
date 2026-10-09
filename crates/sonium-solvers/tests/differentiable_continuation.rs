use sonium_solvers::{
    AdjointCapability, AdjointSolve, CsrMatrix, DifferentiableLinearSolve, GmresConfig,
    IdentityPreconditioner, LinearOperator, SolverError,
};
use std::sync::atomic::{AtomicUsize, Ordering};

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 2, 2.0),
        ],
    )
}

fn config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 30,
        restart: 3,
        tolerance: 1.0e-12,
        ..GmresConfig::default()
    }
}

#[test]
fn scalar_forward_and_adjoint_continuation_accept_converged_iterates() {
    let operator = matrix();
    let handle = DifferentiableLinearSolve::new(&operator, &IdentityPreconditioner, config())
        .expect("square differentiable system should be accepted");
    let rhs = [1.0, 2.0, -1.0];
    let adjoint_rhs = [3.0, -2.0, 1.0];

    let forward = handle
        .solve_forward(&rhs)
        .expect("cold forward solve should converge");
    let resumed_forward = handle
        .solve_forward_with_guess(&rhs, Some(&forward.x))
        .expect("converged forward iterate should be reusable");
    assert!(resumed_forward.converged);
    assert_eq!(resumed_forward.iterations, 0);
    assert!(resumed_forward.residual <= 1.0e-12);

    let transpose = handle
        .solve_adjoint(&adjoint_rhs, AdjointSolve::Transpose)
        .expect("cold transpose solve should converge");
    let resumed_transpose = handle
        .solve_adjoint_with_guess(&adjoint_rhs, AdjointSolve::Transpose, Some(&transpose.x))
        .expect("converged transpose iterate should be reusable");
    assert!(resumed_transpose.converged);
    assert_eq!(resumed_transpose.iterations, 0);

    let hermitian = handle
        .solve_adjoint(&adjoint_rhs, AdjointSolve::Hermitian)
        .expect("cold Hermitian solve should converge");
    let resumed_hermitian = handle
        .solve_adjoint_with_guess(&adjoint_rhs, AdjointSolve::Hermitian, Some(&hermitian.x))
        .expect("converged Hermitian iterate should be reusable");
    assert!(resumed_hermitian.converged);
    assert_eq!(resumed_hermitian.iterations, 0);
}

#[test]
fn scalar_continuation_rejects_malformed_and_nonfinite_guesses() {
    let operator = matrix();
    let handle = DifferentiableLinearSolve::new(&operator, &IdentityPreconditioner, config())
        .expect("square differentiable system should be accepted");

    assert!(matches!(
        handle.solve_forward_with_guess(&[1.0, 2.0, -1.0], Some(&[0.0, 0.0])),
        Err(SolverError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        handle.solve_adjoint_with_guess(
            &[1.0, 2.0, -1.0],
            AdjointSolve::Hermitian,
            Some(&[0.0, f64::NAN, 0.0]),
        ),
        Err(SolverError::OperationFailed { .. })
    ));
}

/// The first operator action describes `A = 1`, while all later actions use
/// `A = 2`.  With projected-only GMRES the first candidate appears converged,
/// but the checked differentiable boundary rejects it when it recomputes the
/// original residual.  Promoting exact checks into the Krylov config lets the
/// solver restart from that candidate and discover the stable operator value.
struct PromotedGateOperator {
    calls: AtomicUsize,
}

impl PromotedGateOperator {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }

    fn apply_with_changing_scale(&self, x: &[f64], y: &mut [f64]) {
        let scale = if self.calls.fetch_add(1, Ordering::Relaxed) == 0 {
            1.0
        } else {
            2.0
        };
        y[0] = scale * x[0];
    }
}

impl LinearOperator<f64> for PromotedGateOperator {
    fn num_rows(&self) -> usize {
        1
    }

    fn num_cols(&self) -> usize {
        1
    }

    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        self.apply_with_changing_scale(x, y);
    }

    fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
        self.apply_with_changing_scale(x, y);
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

#[test]
fn differentiable_forward_restarts_after_promoted_residual_rejection() {
    let operator = PromotedGateOperator::new();
    let config = GmresConfig {
        max_iterations: 4,
        restart: 1,
        tolerance: 1.0e-12,
        exact_residual_check_interval: 0,
        ..GmresConfig::default()
    };
    let handle = DifferentiableLinearSolve::new(&operator, &IdentityPreconditioner, config)
        .expect("changing-scale operator is square");

    let result = handle
        .solve_forward(&[1.0])
        .expect("promoted residual rejection should restart the solve");
    assert!(result.converged);
    assert!(result.residual <= 1.0e-12);
    assert!(result.iterations >= 1);
    assert!(operator.calls.load(Ordering::Relaxed) >= 2);

    let adjoint_operator = PromotedGateOperator::new();
    let adjoint_handle = DifferentiableLinearSolve::new(
        &adjoint_operator,
        &IdentityPreconditioner,
        GmresConfig {
            max_iterations: 4,
            restart: 1,
            tolerance: 1.0e-12,
            exact_residual_check_interval: 0,
            ..GmresConfig::default()
        },
    )
    .expect("changing-scale operator is square");
    let adjoint = adjoint_handle
        .solve_adjoint(&[1.0], AdjointSolve::Transpose)
        .expect("adjoint promoted residual rejection should restart");
    assert!(adjoint.converged);
    assert!(adjoint.residual <= 1.0e-12);
}

struct ForwardOnly;

impl LinearOperator<f64> for ForwardOnly {
    fn num_rows(&self) -> usize {
        1
    }

    fn num_cols(&self) -> usize {
        1
    }

    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }

    fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
        panic!("transpose intentionally unavailable");
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Unsupported
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Unsupported
    }
}

#[test]
fn scalar_unsupported_adjoint_is_rejected_before_iteration() {
    let handle = DifferentiableLinearSolve::new(&ForwardOnly, &IdentityPreconditioner, config())
        .expect("forward-only square system should be accepted");
    let error = handle
        .solve_adjoint_with_guess(&[1.0], AdjointSolve::Hermitian, Some(&[1.0]))
        .expect_err("unsupported Hermitian adjoint should fail closed");
    assert!(matches!(
        error,
        SolverError::UnsupportedOperatorOperation { .. }
    ));
}
