use sonium_solvers_core::traits::{
    AccuracyStatus, AdjointCapability, LinearOperator, SolverError, SolverResultContract,
    SolverStatus,
};
use sonium_solvers_sparse::CsrMatrix;
use sonium_solvers_time_integration::{
    NewmarkAdjointResult, NewmarkConfig, NewmarkInitial, newmark_adjoint, newmark_integrate,
};

#[test]
fn newmark_adjoint_implements_shared_result_contract() {
    let result = NewmarkAdjointResult {
        force_gradient: vec![vec![0.0_f64]],
        initial_displacement_gradient: vec![0.0],
        initial_velocity_gradient: vec![0.0],
        initial_acceleration_gradient: vec![0.0],
    };
    assert_eq!(result.solver_status(), SolverStatus::Converged);
    assert_eq!(result.accuracy_status(), AccuracyStatus::NotChecked);
    result.validate_contract().expect("finite adjoint is valid");
}

#[test]
fn newmark_adjoint_contract_rejects_nonfinite_gradient() {
    let result = NewmarkAdjointResult {
        force_gradient: vec![vec![f64::NAN]],
        initial_displacement_gradient: vec![0.0],
        initial_velocity_gradient: vec![0.0],
        initial_acceleration_gradient: vec![0.0],
    };
    assert!(result.validate_contract().is_err());
}

/// Stiffness stand-in with controllable Hermitian capability/action for the
/// validate-once adjoint gates.
struct AdjointProbe {
    inner: CsrMatrix<f64>,
    capability: AdjointCapability,
    hermitian_fill: Option<f64>,
}

impl LinearOperator<f64> for AdjointProbe {
    fn num_rows(&self) -> usize {
        self.inner.num_rows
    }

    fn num_cols(&self) -> usize {
        self.inner.num_cols
    }

    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        self.inner.apply_into(x, y);
    }

    fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
        self.inner.apply_transpose_into(x, y);
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        self.capability
    }

    fn apply_hermitian(&self, x: &[f64]) -> Vec<f64> {
        match self.hermitian_fill {
            Some(value) => vec![value; x.len()],
            None => self.inner.apply_hermitian(x),
        }
    }
}

fn stored_forward_trajectory() -> (
    NewmarkConfig,
    sonium_solvers_time_integration::NewmarkResult<f64>,
) {
    let stiffness = CsrMatrix::identity(1);
    let initial = NewmarkInitial {
        displacement: vec![1.0],
        velocity: vec![0.0],
        acceleration: vec![-1.0],
    };
    let config = NewmarkConfig::average_acceleration(0.1, 2).with_history();
    let forward = newmark_integrate(
        None::<&CsrMatrix<f64>>,
        &stiffness,
        |_time| vec![0.0],
        &initial,
        |rhs: &[f64]| rhs.to_vec(),
        &config,
    );
    (config, forward)
}

#[test]
fn newmark_adjoint_admits_hermitian_capability_once() {
    let (config, forward) = stored_forward_trajectory();
    let probe = AdjointProbe {
        inner: CsrMatrix::identity(1),
        capability: AdjointCapability::Unknown,
        hermitian_fill: None,
    };
    let error = newmark_adjoint(
        None::<&CsrMatrix<f64>>,
        &probe,
        &forward,
        &[1.0],
        &[0.0],
        &[0.0],
        |rhs: &[f64]| Ok(rhs.to_vec()),
        &config,
    )
    .expect_err("unknown Hermitian capability must be rejected");
    assert!(
        matches!(error, SolverError::UnsupportedOperatorOperation { .. }),
        "unexpected error: {error:?}"
    );
}

#[test]
fn newmark_adjoint_detects_nonfinite_operator_output_at_trajectory_end() {
    let (config, forward) = stored_forward_trajectory();
    let probe = AdjointProbe {
        inner: CsrMatrix::identity(1),
        capability: AdjointCapability::Exact,
        hermitian_fill: Some(f64::NAN),
    };
    let error = newmark_adjoint(
        None::<&CsrMatrix<f64>>,
        &probe,
        &forward,
        &[1.0],
        &[0.0],
        &[0.0],
        |rhs: &[f64]| Ok(rhs.to_vec()),
        &config,
    )
    .expect_err("non-finite Hermitian action must be rejected");
    assert!(
        matches!(error, SolverError::OperationFailed { .. }),
        "unexpected error: {error:?}"
    );
}
