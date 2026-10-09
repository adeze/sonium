use sonium_solvers_core::traits::{AccuracyStatus, SolverResultContract, SolverStatus};
use sonium_solvers_optimization::{
    BrentConfig, BrentStatus, LineSearchResult, OptimResult, OptimStatus, ParetoResult, QpSolution,
    QpStatus, brent_minimize, solve_qp,
};

fn finite_optimizer_result(status: OptimStatus, converged: bool) -> OptimResult<f64> {
    OptimResult {
        x: vec![0.0],
        fun: 0.0,
        grad: Some(vec![0.0]),
        iterations: 1,
        fun_evals: 1,
        grad_evals: 1,
        hess_evals: 0,
        grad_norm: 0.0,
        converged,
        status,
        kkt: None,
    }
}

#[test]
fn line_search_result_implements_shared_result_contract() {
    let result = LineSearchResult {
        alpha: 0.5,
        fun: 1.0,
        grad: Some(vec![0.0]),
        fun_evals: 1,
        grad_evals: 1,
        success: true,
    };
    assert_eq!(result.solver_status(), SolverStatus::Converged);
    assert_eq!(result.accuracy_status(), AccuracyStatus::Attained);
    result
        .validate_contract()
        .expect("finite line search is valid");

    let invalid = LineSearchResult {
        alpha: f64::NAN,
        ..result
    };
    assert!(invalid.validate_contract().is_err());
}

#[test]
fn pareto_result_validates_all_subproblem_contracts() {
    let result = ParetoResult {
        solutions: vec![vec![0.0]],
        objective_values: vec![vec![0.0]],
        subproblem_results: vec![finite_optimizer_result(OptimStatus::Converged, true)],
    };
    assert_eq!(result.solver_status(), SolverStatus::Converged);
    assert_eq!(result.accuracy_status(), AccuracyStatus::Attained);
    result
        .validate_contract()
        .expect("finite Pareto result is valid");

    let invalid = ParetoResult {
        objective_values: Vec::new(),
        ..result
    };
    assert!(invalid.validate_contract().is_err());
}

#[test]
fn qp_result_contract_maps_statuses_and_rejects_malformed_vectors() {
    let h =
        sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    let g = sonium_solvers_core::array![-1.0, -2.0];
    let constraints = sonium_solvers_core::dense::DenseMatrix::from_row_major_vec(0, 2, vec![]);
    let optimal = solve_qp(&h, &g, &constraints, &[], &[]);
    assert_eq!(optimal.status, QpStatus::Optimal);
    assert_eq!(optimal.solver_status(), SolverStatus::Converged);
    assert_eq!(optimal.accuracy_status(), AccuracyStatus::Attained);
    optimal
        .validate_contract()
        .expect("finite optimal QP result should satisfy the contract");

    let capped = QpSolution {
        d: vec![0.0, 0.0],
        multipliers: vec![0.0],
        active_set: vec![],
        status: QpStatus::MaxIterations,
    };
    assert_eq!(capped.solver_status(), SolverStatus::MaxIterationsReached);
    assert_eq!(capped.accuracy_status(), AccuracyStatus::NotAttained);
    capped
        .validate_contract()
        .expect("finite capped QP result should satisfy the contract");

    let infeasible = QpSolution {
        d: vec![0.0, 0.0],
        multipliers: vec![0.0, 0.0],
        active_set: vec![],
        status: QpStatus::Infeasible,
    };
    assert_eq!(infeasible.solver_status(), SolverStatus::Stagnated);
    assert_eq!(infeasible.accuracy_status(), AccuracyStatus::NotAttained);
    infeasible
        .validate_contract()
        .expect("finite infeasible QP result should satisfy the contract");

    let non_finite = QpSolution {
        d: vec![f64::NAN],
        multipliers: vec![],
        active_set: vec![],
        status: QpStatus::Optimal,
    };
    assert_eq!(non_finite.solver_status(), SolverStatus::Converged);
    assert!(non_finite.validate_contract().is_err());

    let malformed_active_set = QpSolution {
        d: vec![0.0],
        multipliers: vec![0.0],
        active_set: vec![1],
        status: QpStatus::Optimal,
    };
    assert!(malformed_active_set.validate_contract().is_err());

    let duplicate_active_set = QpSolution {
        d: vec![0.0],
        multipliers: vec![0.0, 0.0],
        active_set: vec![0, 0],
        status: QpStatus::Optimal,
    };
    assert!(duplicate_active_set.validate_contract().is_err());
}

#[test]
fn brent_result_contract_distinguishes_convergence_budget_and_numerical_failure() {
    let converged = brent_minimize(
        &|x: f64| (x - 1.5) * (x - 1.5),
        -2.0,
        4.0,
        &BrentConfig::default(),
    );
    assert_eq!(converged.status, BrentStatus::Converged);
    assert_eq!(converged.solver_status(), SolverStatus::Converged);
    assert_eq!(converged.accuracy_status(), AccuracyStatus::Attained);
    converged
        .validate_contract()
        .expect("converged Brent result should satisfy the contract");

    let capped = brent_minimize(
        &|x: f64| (x - 1.5) * (x - 1.5),
        -2.0,
        4.0,
        &BrentConfig {
            tol: 1.0e-12,
            max_iterations: 0,
        },
    );
    assert_eq!(capped.status, BrentStatus::MaxIterations);
    assert_eq!(capped.solver_status(), SolverStatus::MaxIterationsReached);
    assert_eq!(capped.accuracy_status(), AccuracyStatus::NotAttained);
    capped
        .validate_contract()
        .expect("finite capped Brent result should satisfy the contract");

    let numerical = brent_minimize(&|_: f64| f64::NAN, -2.0, 4.0, &BrentConfig::default());
    assert_eq!(numerical.status, BrentStatus::NumericalError);
    assert_eq!(numerical.solver_status(), SolverStatus::NumericalError);
    assert_eq!(numerical.accuracy_status(), AccuracyStatus::NotChecked);
    assert!(numerical.validate_contract().is_err());
}
