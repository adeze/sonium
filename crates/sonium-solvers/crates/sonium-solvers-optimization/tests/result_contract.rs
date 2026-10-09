use sonium_solvers_core::traits::{AccuracyStatus, SolverResultContract, SolverStatus};
use sonium_solvers_optimization::{
    IpmDiagnostics, KktDiagnostics, KktMultipliers, KktReport, OptimResult, OptimStatus,
};

fn finite_result(status: OptimStatus, converged: bool) -> OptimResult<f64> {
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
fn optimization_results_implement_shared_result_contract() {
    for status in [
        OptimStatus::Converged,
        OptimStatus::FunctionTolerance,
        OptimStatus::StepTolerance,
    ] {
        let result = finite_result(status, true);
        assert_eq!(result.solver_status(), SolverStatus::Converged);
        assert_eq!(result.accuracy_status(), AccuracyStatus::Attained);
        result.validate_contract().expect("finite result is valid");
    }
}

#[test]
fn optimization_result_contract_rejects_nonfinite_metadata() {
    let result = OptimResult {
        grad_norm: f64::NAN,
        ..finite_result(OptimStatus::Converged, true)
    };
    assert!(result.validate_contract().is_err());
}

#[test]
fn optimization_result_contract_rejects_malformed_kkt_active_set() {
    let mut result = finite_result(OptimStatus::Converged, true);
    result.kkt = Some(KktReport {
        multipliers: KktMultipliers {
            constraints: Vec::new(),
            lower_bounds: vec![0.0],
            upper_bounds: vec![0.0],
        },
        diagnostics: KktDiagnostics {
            primal_feasibility: 0.0,
            stationarity: 0.0,
            dual_feasibility: 0.0,
            complementarity: 0.0,
            active_lower_bounds: vec![1],
            active_upper_bounds: Vec::new(),
        },
        norm: 0.0,
        ipm: None,
    });

    let error = result
        .validate_contract()
        .expect_err("out-of-range active bound must be rejected");
    assert!(error.to_string().contains("active lower-bound"));
}

#[test]
fn optimization_result_contract_rejects_inconsistent_ipm_metadata() {
    let mut result = finite_result(OptimStatus::Converged, true);
    result.kkt = Some(KktReport {
        multipliers: KktMultipliers {
            constraints: Vec::new(),
            lower_bounds: Vec::new(),
            upper_bounds: Vec::new(),
        },
        diagnostics: KktDiagnostics {
            primal_feasibility: 0.0,
            stationarity: 0.0,
            dual_feasibility: 0.0,
            complementarity: 0.0,
            active_lower_bounds: Vec::new(),
            active_upper_bounds: Vec::new(),
        },
        norm: 0.0,
        ipm: Some(IpmDiagnostics {
            iterations: 2,
            barrier_parameter: 1.0,
            dual_residual: 0.0,
            primal_residual: 0.0,
            complementarity: 0.0,
            restoration_attempts: 1,
            restoration_successes: 1,
            active_inequalities: 1,
            active_bounds: 0,
            active_inequality_rows: Vec::new(),
            active_bound_rows: Vec::new(),
            line_search_backtracks: 0,
            restoration_failures: 1,
        }),
    });

    let error = result
        .validate_contract()
        .expect_err("inconsistent IPM metadata must be rejected");
    assert!(error.to_string().contains("IPM active inequality count"));

    result
        .kkt
        .as_mut()
        .expect("KKT report")
        .ipm
        .as_mut()
        .expect("IPM diagnostics")
        .active_inequalities = 0;
    let error = result
        .validate_contract()
        .expect_err("impossible restoration accounting must be rejected");
    assert!(error.to_string().contains("restoration outcome"));
}
