use sonium_solvers_core::traits::{AccuracyStatus, SolverResultContract, SolverStatus};
use sonium_solvers_nonlinear::{AndersonSolution, NewtonSolution};

#[test]
fn nonlinear_results_implement_shared_result_contract() {
    let anderson = AndersonSolution {
        x: vec![1.0_f64],
        iterations: 1,
        residual: 0.0,
        converged: true,
        status: SolverStatus::Converged,
        attainable_accuracy: AccuracyStatus::Attained,
    };
    assert_eq!(anderson.solver_status(), SolverStatus::Converged);
    anderson
        .validate_contract()
        .expect("Anderson result is valid");

    let newton = NewtonSolution {
        x: vec![1.0_f64],
        iterations: 1,
        residual: 0.0,
        converged: true,
        status: SolverStatus::Converged,
        attainable_accuracy: AccuracyStatus::Attained,
        residual_evaluations: 2,
        jacobian_evaluations: 1,
        inner_iterations: 0,
    };
    assert_eq!(newton.accuracy_status(), AccuracyStatus::Attained);
    newton.validate_contract().expect("Newton result is valid");

    let invalid = NewtonSolution {
        residual: f64::NAN,
        ..newton
    };
    assert!(invalid.validate_contract().is_err());
}
