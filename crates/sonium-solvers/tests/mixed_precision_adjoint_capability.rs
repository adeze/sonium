use sonium_solvers::mixed_precision::MixedPrecisionPreconditioner;
use sonium_solvers::{
    AdjointCapability, AdjointSolve, DifferentiableLinearSolve, GmresConfig, LinearOperator,
    Preconditioner, SolverError,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingIdentity {
    applications: AtomicUsize,
}

impl CountingIdentity {
    fn new() -> Self {
        Self {
            applications: AtomicUsize::new(0),
        }
    }
}

impl LinearOperator<f64> for CountingIdentity {
    fn num_rows(&self) -> usize {
        1
    }

    fn num_cols(&self) -> usize {
        1
    }

    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        self.applications.fetch_add(1, Ordering::Relaxed);
        y.copy_from_slice(x);
    }

    fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
        self.applications.fetch_add(1, Ordering::Relaxed);
        y.copy_from_slice(x);
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

struct ForwardOnlyLow {
    applications: AtomicUsize,
}

impl ForwardOnlyLow {
    fn new() -> Self {
        Self {
            applications: AtomicUsize::new(0),
        }
    }
}

impl Preconditioner<f32> for ForwardOnlyLow {
    fn apply(&self, r: &[f32]) -> Vec<f32> {
        self.applications.fetch_add(1, Ordering::Relaxed);
        r.to_vec()
    }
}

fn config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 4,
        restart: 1,
        tolerance: 1.0e-12,
        ..GmresConfig::default()
    }
}

#[test]
fn mixed_precision_unsupported_adjoint_rejects_before_iteration() {
    let operator = CountingIdentity::new();
    let low = ForwardOnlyLow::new();
    let mixed = MixedPrecisionPreconditioner::<f64, f32>::new(Box::new(low));

    assert_eq!(mixed.transpose_capability(), AdjointCapability::Unsupported);
    assert_eq!(mixed.hermitian_capability(), AdjointCapability::Unsupported);

    let handle = DifferentiableLinearSolve::new(&operator, &mixed, config())
        .expect("forward-only mixed preconditioner should still construct a handle");

    for kind in [AdjointSolve::Transpose, AdjointSolve::Hermitian] {
        let error = handle
            .solve_adjoint(&[1.0], kind)
            .expect_err("unsupported mixed adjoint must fail before Krylov iteration");
        assert!(matches!(error, SolverError::UnsupportedOperation { .. }));
    }

    assert_eq!(operator.applications.load(Ordering::Relaxed), 0);
    assert_eq!(mixed.low_precision_failure_count(), 0);
}
