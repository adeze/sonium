//! Release-backed result-contract checks for line search, Pareto, and QP results.

use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::optimization::{
    BrentConfig, BrentStatus, LineSearchResult, ParetoResult, QpStatus, brent_minimize, solve_qp,
};
use sonium_solvers::{
    AccuracyStatus, OptimResult, OptimStatus, SolverResultContract, SolverStatus,
};
use std::time::Instant;

fn finite_optimizer_result() -> OptimResult<f64> {
    OptimResult {
        x: vec![0.0],
        fun: 0.0,
        grad: Some(vec![0.0]),
        iterations: 1,
        fun_evals: 1,
        grad_evals: 1,
        hess_evals: 0,
        grad_norm: 0.0,
        converged: true,
        status: OptimStatus::Converged,
        kkt: None,
    }
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let line_search = LineSearchResult {
        alpha: 0.5,
        fun: 1.0,
        grad: Some(vec![0.0]),
        fun_evals: 1,
        grad_evals: 1,
        success: true,
    };
    line_search
        .validate_contract()
        .map_err(|error| error.to_string())?;
    if line_search.solver_status() != SolverStatus::Converged
        || line_search.accuracy_status() != AccuracyStatus::Attained
    {
        return Err("line-search canonical metadata mismatch".into());
    }

    let pareto = ParetoResult {
        solutions: vec![vec![0.0]],
        objective_values: vec![vec![0.0]],
        subproblem_results: vec![finite_optimizer_result()],
    };
    pareto
        .validate_contract()
        .map_err(|error| error.to_string())?;
    if pareto.solver_status() != SolverStatus::Converged
        || pareto.accuracy_status() != AccuracyStatus::Attained
    {
        return Err("Pareto canonical metadata mismatch".into());
    }

    let h = DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    let g = vec![-1.0, -2.0];
    let constraints = DenseMatrix::from_row_major_vec(0, 2, vec![]);
    let qp = solve_qp(&h, &g, &constraints, &[], &[]);
    qp.validate_contract().map_err(|error| error.to_string())?;
    if qp.status != QpStatus::Optimal
        || qp.solver_status() != SolverStatus::Converged
        || qp.accuracy_status() != AccuracyStatus::Attained
    {
        return Err("QP canonical metadata mismatch".into());
    }

    let brent = brent_minimize(
        &|x: f64| (x - 1.5) * (x - 1.5),
        -2.0,
        4.0,
        &BrentConfig::default(),
    );
    brent
        .validate_contract()
        .map_err(|error| error.to_string())?;
    if brent.status != BrentStatus::Converged
        || brent.solver_status() != SolverStatus::Converged
        || brent.accuracy_status() != AccuracyStatus::Attained
    {
        return Err("Brent canonical metadata mismatch".into());
    }

    let capped = brent_minimize(
        &|x: f64| (x - 1.5) * (x - 1.5),
        -2.0,
        4.0,
        &BrentConfig {
            tol: 1.0e-12,
            max_iterations: 0,
        },
    );
    if capped.status != BrentStatus::MaxIterations
        || capped.solver_status() != SolverStatus::MaxIterationsReached
        || capped.accuracy_status() != AccuracyStatus::NotAttained
    {
        return Err("Brent max-iteration mapping mismatch".into());
    }

    let numerical = brent_minimize(&|_: f64| f64::NAN, -2.0, 4.0, &BrentConfig::default());
    if numerical.status != BrentStatus::NumericalError
        || numerical.solver_status() != SolverStatus::NumericalError
        || numerical.accuracy_status() != AccuracyStatus::NotChecked
        || numerical.validate_contract().is_ok()
    {
        return Err("Brent numerical-error mapping mismatch".into());
    }

    Ok(vec![
        ("line_search_contract", 1.0),
        ("pareto_contract", 1.0),
        ("pareto_subproblem_contract", 1.0),
        ("qp_contract", 1.0),
        ("brent_contract", 1.0),
        ("brent_max_iterations_contract", 1.0),
        ("brent_numerical_contract", 1.0),
        ("canonical_accuracy_contract", 1.0),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("checked line-search, Pareto, and QP result contracts")
                .reference_checksum(
                    "analytic/deterministic-fixture:optimization-result-contract-v1",
                )
                .input_checksum("analytic/deterministic-fixture:optimization-result-contract-v1")
                .tolerance("all optimization result-contract metrics equal one")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.metric(name, value);
            }
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error).to_json_line().unwrap()
            );
            eprintln!("OPTIMIZATION RESULT CONTRACT FAILED");
            std::process::exit(1);
        }
    }
}

const CASE_ID: &str = "solvers.optimization-result-contract.v1";
