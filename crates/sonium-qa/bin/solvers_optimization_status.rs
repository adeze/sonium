//! Release-facing canonical status/accuracy contract for optimization results.

use sonium_qa::QaResult;
use sonium_solvers::optimization::{
    BrentConfig, DifferentialEvolutionConfig, LineSearchResult, ParetoResult, RandomSearchConfig,
    SimulatedAnnealingConfig, brent_minimize, differential_evolution, random_search,
    simulated_annealing,
};
use sonium_solvers::{
    AccuracyStatus, BfgsConfig, NewtonCgConfig, OptimError, OptimResult, OptimStatus, SolverStatus,
    bfgs, newton_cg,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.optimization-status.v1";
const CONTRACT_CASES: f64 = 16.0;

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

fn line_search_contracts() {
    let converged = LineSearchResult {
        alpha: 0.5_f64,
        fun: 1.0,
        grad: Some(vec![0.0]),
        fun_evals: 1,
        grad_evals: 1,
        success: true,
    };
    assert_eq!(converged.status(), SolverStatus::Converged);
    assert_eq!(converged.accuracy_status(), AccuracyStatus::Attained);

    let failed = LineSearchResult {
        success: false,
        ..converged.clone()
    };
    assert_eq!(failed.status(), SolverStatus::MaxIterationsReached);
    assert_eq!(failed.accuracy_status(), AccuracyStatus::NotAttained);

    let non_finite = LineSearchResult {
        alpha: f64::NAN,
        ..converged
    };
    assert_eq!(non_finite.status(), SolverStatus::NumericalError);
    assert_eq!(non_finite.accuracy_status(), AccuracyStatus::NotChecked);

    let exhausted = brent_minimize(
        &|x: f64| (x - 3.0).powi(2),
        0.0,
        10.0,
        &BrentConfig {
            max_iterations: 3,
            ..BrentConfig::default()
        },
    );
    assert_eq!(exhausted.iterations, 3);
}

fn pareto_status_contract() {
    let converged = ParetoResult {
        solutions: vec![vec![0.0_f64]],
        objective_values: vec![vec![0.0]],
        subproblem_results: vec![finite_result(OptimStatus::Converged, true)],
    };
    assert!(converged.is_finite());
    assert_eq!(converged.status(), SolverStatus::Converged);
    assert_eq!(converged.accuracy_status(), AccuracyStatus::Attained);

    let capped = ParetoResult {
        subproblem_results: vec![finite_result(OptimStatus::MaxIterationsReached, false)],
        ..converged
    };
    assert_eq!(capped.status(), SolverStatus::MaxIterationsReached);
    assert_eq!(capped.accuracy_status(), AccuracyStatus::NotAttained);

    let mut non_finite = capped;
    non_finite.objective_values[0][0] = f64::NAN;
    assert_eq!(non_finite.status(), SolverStatus::NumericalError);
    assert_eq!(non_finite.accuracy_status(), AccuracyStatus::NotChecked);
}

fn main() {
    let started = Instant::now();
    let result = std::panic::catch_unwind(|| {
        let bfgs_result = bfgs(
            &|x: &[f64]| 0.5 * x[0] * x[0],
            &|x: &[f64]| vec![x[0]],
            &[2.0],
            &BfgsConfig::default(),
        )
        .expect("quadratic BFGS fixture must return a result");
        assert_eq!(bfgs_result.status(), SolverStatus::Converged);
        assert_eq!(bfgs_result.attainable_accuracy(), AccuracyStatus::Attained);

        let objective_calls = std::cell::Cell::new(0usize);
        let generated_error = newton_cg(
            &|x: &[f64]| {
                let call = objective_calls.get();
                objective_calls.set(call + 1);
                if call >= 2 {
                    f64::NAN
                } else {
                    x.iter().map(|value| value * value).sum()
                }
            },
            &|x: &[f64]| x.iter().map(|value| 2.0 * value).collect::<Vec<_>>(),
            &|_: &[f64], vector: &[f64]| vector.iter().map(|value| 2.0 * value).collect::<Vec<_>>(),
            &[1.0, 1.0],
            &NewtonCgConfig::default(),
        )
        .expect_err("generated Newton-CG objective value must be rejected");
        assert!(matches!(generated_error, OptimError::NumericalError { .. }));

        let bounds = vec![sonium_solvers::Bound::both(-1.0, 1.0)];
        let objective = |x: &[f64]| x[0] * x[0];
        let invalid_differential = differential_evolution(
            &objective,
            &DifferentialEvolutionConfig {
                bounds: bounds.clone(),
                target_f: f64::INFINITY,
                ..DifferentialEvolutionConfig::default()
            },
        );
        let invalid_annealing = simulated_annealing(
            &objective,
            &SimulatedAnnealingConfig {
                bounds: bounds.clone(),
                target_f: f64::NAN,
                ..SimulatedAnnealingConfig::default()
            },
        );
        let invalid_random = random_search(
            &objective,
            &RandomSearchConfig {
                bounds,
                target_f: f64::INFINITY,
                ..RandomSearchConfig::default()
            },
        );
        for result in [invalid_differential, invalid_annealing, invalid_random] {
            assert!(matches!(
                result,
                Err(OptimError::NumericalError { iteration: 0, .. })
            ));
        }

        line_search_contracts();
        pareto_status_contract();

        let max_iterations = finite_result(OptimStatus::MaxIterationsReached, false);
        assert_eq!(max_iterations.status(), SolverStatus::MaxIterationsReached);
        assert_eq!(
            max_iterations.attainable_accuracy(),
            AccuracyStatus::NotAttained
        );

        let stagnated = finite_result(OptimStatus::LineSearchFailed, false);
        assert_eq!(stagnated.status(), SolverStatus::Stagnated);
        assert_eq!(stagnated.attainable_accuracy(), AccuracyStatus::NotAttained);

        let mut numerical = finite_result(OptimStatus::Converged, true);
        numerical.fun = f64::NAN;
        assert_eq!(numerical.status(), SolverStatus::NumericalError);
        assert_eq!(numerical.attainable_accuracy(), AccuracyStatus::NotChecked);
    });

    match result {
        Ok(()) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("status_contract_case_count", CONTRACT_CASES)
                .minimum("status_contract_case_count", CONTRACT_CASES)
                .metric("canonical_success", 1.0)
                .minimum("canonical_success", 1.0)
                .input_checksum("analytic/deterministic-fixture:solvers-optimization-status-v1")
                .reference("shared SolverStatus and AccuracyStatus vocabulary")
                .reference_checksum("analytic")
                .dimension("f64; optimization result and typed failure metadata")
                .formulation(
                    "optimization, line-search, Brent iteration-accounting, bounded-global target-threshold, and non-finite result paths",
                )
                .tolerance("16/16 canonical status/accuracy/generated-callback contract cases pass")
                .seed("solvers-optimization-status:5-contracts:v2")
                .timing_ms(
                    "optimization_status_contract",
                    started.elapsed().as_millis() as u64,
                );
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("optimization status result must serialize")
            );
            println!("SUCCESS");
        }
        Err(payload) => {
            let message = if let Some(message) = payload.downcast_ref::<&str>() {
                (*message).to_owned()
            } else if let Some(message) = payload.downcast_ref::<String>() {
                message.clone()
            } else {
                "optimization status contract panicked".to_owned()
            };
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, message)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .elapsed_ms(started.elapsed().as_millis() as u64)
                    .to_json_line()
                    .expect("optimization status failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
