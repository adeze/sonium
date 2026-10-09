use crate::evaluation::{
    attach_operating_point_spread, evaluate_pc_fan, evaluate_pc_fan_fan_curve_objective,
};
use crate::model::*;
use rayon::prelude::*;
use sonium_solvers_optimization::unconstrained::de::{
    DEConfigBuilder, LShadeConfig, Strategy, differential_evolution,
};
use sonium_solvers_optimization::{
    Bound, LbfgsbConfig, OptimError, OptimResult, OptimStatus, lbfgsb,
};

/// Maximum number of warm-restart outer iterations used to drive the
/// converged design back inside the airflow band when L-BFGS-B has
/// settled outside it.
const MAX_WARM_RESTARTS: usize = 3;

/// Multiplicative escalation of `airflow_penalty_weight` per warm-restart
/// outer iteration when the airflow constraint is violated.
const WARM_RESTART_PENALTY_FACTOR: f64 = 10.0;

/// Deterministically restore a reduced-order candidate before promotion.
///
/// The optimizer is still allowed to use a smooth penalty while searching,
/// but this pass performs an explicit bounded coordinate restoration against
/// the hard feasibility contract. It never runs a CFD solve: CFD-backed
/// optimization remains fail-closed until a converged high-fidelity objective
/// is available.
pub fn restore_pc_fan_feasibility(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    max_iterations: usize,
) -> PcFanFeasibilityRestoration {
    let bounds = design_bounds(config);
    let original_x = design.to_vector().to_vec();
    let mut x = original_x.clone();
    project_to_bounds(&bounds, &mut x);
    let was_projected = x != original_x;

    let mut evaluated = EvaluatedDesign::from_design(config, vector_to_design(config, &x));
    let initial_feasible = evaluated.evaluation.is_feasible(config);
    let mut function_evaluations = 1usize;
    let mut iterations = 0usize;

    if initial_feasible || max_iterations == 0 {
        return PcFanFeasibilityRestoration {
            design_variables: evaluated.design_variables,
            evaluation: evaluated.evaluation,
            iterations,
            function_evaluations,
            restored: initial_feasible,
        };
    }

    let mut steps = bounds
        .iter()
        .map(|bound| {
            let lower = bound.lower.unwrap_or(-1.0);
            let upper = bound.upper.unwrap_or(1.0);
            ((upper - lower).abs() * 0.10).max(1.0e-8)
        })
        .collect::<Vec<_>>();

    for iteration in 0..max_iterations {
        iterations = iteration + 1;
        let current_score = restoration_score(config, &evaluated.evaluation);
        let mut best_x = x.clone();
        let mut best_evaluated = evaluated.clone();
        let mut best_score = current_score;
        let mut improved = false;

        for index in 0..x.len() {
            for direction in [-1.0, 1.0] {
                let mut trial_x = x.clone();
                trial_x[index] += direction * steps[index];
                project_to_bounds(&bounds, &mut trial_x);
                if trial_x[index] == x[index] {
                    continue;
                }

                let trial =
                    EvaluatedDesign::from_design(config, vector_to_design(config, &trial_x));
                function_evaluations = function_evaluations.saturating_add(1);
                let trial_score = restoration_score(config, &trial.evaluation);
                if restoration_score_is_better(trial_score, best_score) {
                    best_x = trial_x;
                    best_evaluated = trial;
                    best_score = trial_score;
                    improved = true;
                }
            }
        }

        x = best_x;
        evaluated = best_evaluated;
        if evaluated.evaluation.is_feasible(config) {
            break;
        }
        if !improved {
            for step in &mut steps {
                *step *= 0.5;
            }
            if steps.iter().all(|step| *step <= 1.0e-9) {
                break;
            }
        }
    }

    let restored = evaluated.evaluation.is_feasible(config) && (was_projected || !initial_feasible);
    PcFanFeasibilityRestoration {
        design_variables: evaluated.design_variables,
        evaluation: evaluated.evaluation,
        iterations,
        function_evaluations,
        restored,
    }
}

fn project_to_bounds(bounds: &[Bound<f64>], x: &mut [f64]) {
    for (value, bound) in x.iter_mut().zip(bounds) {
        if let Some(lower) = bound.lower {
            *value = value.max(lower);
        }
        if let Some(upper) = bound.upper {
            *value = value.min(upper);
        }
        if !value.is_finite() {
            *value = bound.lower.or(bound.upper).unwrap_or(0.0);
        }
    }
}

fn restoration_score(config: &PcFanConfig, evaluation: &PcFanEvaluation) -> (u8, f64, f64) {
    let margins = evaluation.constraint_margins;
    let mut violation = evaluation.feasibility_violation(config);
    violation += (-margins.minimum_frame_clearance_m).max(0.0);
    violation += margins.root_tangential_attachment_error_m.abs();
    if !margins.minimum_frame_clearance_m.is_finite()
        || !margins.root_tangential_attachment_error_m.is_finite()
    {
        violation += 1.0;
    }
    if !violation.is_finite() {
        return (1, f64::INFINITY, f64::INFINITY);
    }
    (
        u8::from(!evaluation.is_feasible(config)),
        violation,
        evaluation.objective,
    )
}

fn restoration_score_is_better(candidate: (u8, f64, f64), incumbent: (u8, f64, f64)) -> bool {
    candidate.0 < incumbent.0
        || (candidate.0 == incumbent.0
            && (candidate.1 < incumbent.1 - 1.0e-12
                || ((candidate.1 - incumbent.1).abs() <= 1.0e-12 && candidate.2 < incumbent.2)))
}

/// Optimise a PC fan using **L-SHADE** (Tanabe & Fukunaga 2014) — the
/// SHADE variant of differential evolution with linear population-size
/// reduction. L-SHADE is gradient-free → robust against noisy CFD-FD
/// gradients that stall L-BFGS-B at local minima, and converges faster
/// than vanilla DE because it adapts mutation/crossover parameters per
/// generation and shrinks the population from `np_init` down to
/// `np_final = 4` over the budget.
///
/// Costs roughly `np_init · maxiter / 2` objective evaluations (the
/// average population over the linear-reduction schedule is `(np_init +
/// np_final) / 2`). For the default 30-DOF blade vector with
/// `popsize_factor = 15`, that's `15·30 = 450` initial individuals
/// shrinking to 4, mean ≈ 227, × `maxiter = 30` ≈ 6800 evaluations.
///
/// Returns the best design found across the run. The seed is the L-SHADE
/// random seed and is the only source of stochasticity, so runs with the
/// same `(config, maxiter, popsize_factor, seed)` triple are exactly
/// reproducible.
pub fn optimize_pc_fan_de(
    config: PcFanConfig,
    maxiter: usize,
    popsize_factor: usize,
    seed: u64,
) -> OptimizedPcFan {
    // The public optimization APIs return an infallible compatibility result,
    // whereas CFD admission is fallible: a Picard field must be checked before
    // it can contribute to an objective. Do not silently turn a failed CFD
    // solve into a DE fitness value. A checked, CFD-backed optimizer belongs
    // behind a fallible API once its convergence/conservation contract exists.
    if config.cfd.flow_model.is_cfd() {
        return cfd_optimization_rejected(config);
    }
    let baseline = EvaluatedDesign::from_design(&config, config.default_design_variables());
    let bounds_box = design_bounds(&config);
    let de_bounds: Vec<(f64, f64)> = bounds_box
        .iter()
        .map(|b| {
            (
                b.lower.unwrap_or(f64::NEG_INFINITY),
                b.upper.unwrap_or(f64::INFINITY),
            )
        })
        .collect();

    let lshade = LShadeConfig {
        // np_init in `LShadeConfig` is the *multiplier*: actual initial
        // population size is `np_init · n_dim`. We expose the multiplier
        // through `popsize_factor` so the CLI knob remains familiar.
        np_init: popsize_factor.max(5),
        np_final: 4,
        p: 0.11,
        arc_rate: 2.1,
        memory_size: 6,
    };

    let de_config = match DEConfigBuilder::new()
        .maxiter(maxiter)
        .popsize(popsize_factor.max(5))
        .seed(seed)
        .strategy(Strategy::LShadeBin)
        .lshade(lshade)
        .build()
    {
        Ok(c) => c,
        Err(err) => {
            eprintln!("DE config build failed: {err:?} — falling back to L-BFGS-B");
            return optimize_pc_fan(config);
        }
    };

    let result = differential_evolution(
        &|x: &[f64]| objective_for_vector(&config, x),
        &de_bounds,
        de_config,
    );
    let report = match result {
        Ok(r) => r,
        Err(err) => {
            eprintln!("DE failed: {err:?} — returning baseline");
            return OptimizedPcFan {
                config: config.clone(),
                design_variables: baseline.design_variables.clone(),
                geometry: baseline.geometry.clone(),
                evaluation: baseline.evaluation.clone(),
                optimization: PcFanOptimizationSummary {
                    baseline_objective: baseline.evaluation.objective,
                    best_objective: baseline.evaluation.objective,
                    iterations: 0,
                    function_evaluations: 1,
                    gradient_evaluations: 0,
                    feasible: baseline.evaluation.is_feasible(&config),
                    fan_curve_objective: None,
                    fan_curve_feasible: true,
                    converged: false,
                    status: OptimStatus::NumericalError,
                },
            };
        }
    };

    let report_x = report.x.clone();
    let candidate = EvaluatedDesign::from_design(&config, vector_to_design(&config, &report_x));
    let candidate = if candidate.evaluation.is_feasible(&config) {
        candidate
    } else {
        let restored = restore_pc_fan_feasibility(&config, &candidate.design_variables, 24);
        EvaluatedDesign::from_design(&config, restored.design_variables)
    };
    let selected = select_for_promotion(&config, baseline.clone(), candidate);
    let mut summary = PcFanOptimizationSummary {
        baseline_objective: baseline.evaluation.objective,
        best_objective: selected.evaluation.objective,
        iterations: report.nit,
        function_evaluations: report.nfev,
        gradient_evaluations: 0,
        feasible: selected.evaluation.is_feasible(&config),
        fan_curve_objective: None,
        fan_curve_feasible: true,
        converged: report.success,
        // DE doesn't map cleanly onto OptimStatus; flag whether the
        // population settled within the configured iteration budget.
        status: if report.success {
            OptimStatus::Converged
        } else {
            OptimStatus::MaxIterationsReached
        },
    };
    if !summary.feasible {
        summary.converged = false;
        summary.status = OptimStatus::RestorationFailed;
    }

    OptimizedPcFan {
        config,
        design_variables: selected.design_variables,
        geometry: selected.geometry,
        evaluation: selected.evaluation,
        optimization: summary,
    }
}

/// Optimize a reduced-order design against an explicit multi-point fan curve.
///
/// This is the first-class multi-point objective path: the ordinary optimizer
/// remains available for one operating point, while this API scores every
/// requested flow sample against flow/pressure and optional acoustic or
/// efficiency targets. It is deliberately reduced-order and rejects CFD
/// configurations; a resolved-flow fan-curve optimizer needs a pressure/load
/// boundary sweep and a separate high-fidelity admission contract.
pub fn optimize_pc_fan_fan_curve(
    config: PcFanConfig,
    objective_config: PcFanFanCurveObjectiveConfig,
    maxiter: usize,
    popsize_factor: usize,
    seed: u64,
) -> Result<OptimizedPcFan, crate::evaluation::PcFanFanCurveError> {
    if config.cfd.flow_model != PcFanFlowModel::ReducedOrder {
        return Err(crate::evaluation::PcFanFanCurveError::UnsupportedFlowModel(
            config.cfd.flow_model,
        ));
    }

    // Validate targets and establish a baseline before creating the DE
    // closure. This prevents malformed target vectors from being converted to
    // an all-infinite population objective.
    let baseline = EvaluatedDesign::from_design(&config, config.default_design_variables());
    let baseline_curve = evaluate_pc_fan_fan_curve_objective(
        &config,
        &baseline.design_variables,
        &objective_config,
    )?;
    let bounds = design_bounds(&config);
    let de_bounds: Vec<(f64, f64)> = bounds
        .iter()
        .map(|bound| {
            (
                bound.lower.unwrap_or(f64::NEG_INFINITY),
                bound.upper.unwrap_or(f64::INFINITY),
            )
        })
        .collect();
    let lshade = LShadeConfig {
        np_init: popsize_factor.max(5),
        np_final: 4,
        p: 0.11,
        arc_rate: 2.1,
        memory_size: 6,
    };
    let de_config = DEConfigBuilder::new()
        .maxiter(maxiter)
        .popsize(popsize_factor.max(5))
        .seed(seed)
        .strategy(Strategy::LShadeBin)
        .lshade(lshade)
        .build()
        .map_err(|error| {
            crate::evaluation::PcFanFanCurveError::OptimizerFailure(format!("{error:?}"))
        })?;
    let report = differential_evolution(
        &|x: &[f64]| {
            let design = vector_to_design(&config, x);
            evaluate_pc_fan_fan_curve_objective(&config, &design, &objective_config)
                .map(|result| result.score)
                .unwrap_or(f64::INFINITY)
        },
        &de_bounds,
        de_config,
    )
    .map_err(|error| {
        crate::evaluation::PcFanFanCurveError::OptimizerFailure(format!("{error:?}"))
    })?;
    let report_x = report.x.as_slice();
    let candidate = EvaluatedDesign::from_design(&config, vector_to_design(&config, report_x));
    let candidate_curve = evaluate_pc_fan_fan_curve_objective(
        &config,
        &candidate.design_variables,
        &objective_config,
    )?;
    let baseline_eligible = baseline.evaluation.is_feasible(&config) && baseline_curve.feasible;
    let candidate_eligible = candidate.evaluation.is_feasible(&config) && candidate_curve.feasible;
    let baseline_objective = baseline.evaluation.objective;
    let (selected, selected_curve) = if candidate_eligible
        && (!baseline_eligible || candidate_curve.score <= baseline_curve.score)
    {
        (candidate, candidate_curve)
    } else {
        (baseline, baseline_curve)
    };
    let center_feasible = selected.evaluation.is_feasible(&config);
    let fan_curve_feasible = selected_curve.feasible;
    let feasible = center_feasible && fan_curve_feasible;
    let best_objective = selected.evaluation.objective;
    let status = if !feasible {
        OptimStatus::RestorationFailed
    } else if report.success {
        OptimStatus::Converged
    } else {
        OptimStatus::MaxIterationsReached
    };
    Ok(OptimizedPcFan {
        config,
        design_variables: selected.design_variables,
        geometry: selected.geometry,
        evaluation: selected.evaluation,
        optimization: PcFanOptimizationSummary {
            baseline_objective,
            best_objective,
            iterations: report.nit,
            function_evaluations: report.nfev,
            gradient_evaluations: 0,
            feasible,
            fan_curve_objective: Some(selected_curve.score),
            fan_curve_feasible,
            converged: report.success && feasible,
            status,
        },
    })
}

/// Optimize with an explicit derivative policy.
///
/// The finite-difference policy selects the existing L-BFGS-B path. The
/// derivative-free policy selects L-SHADE differential evolution, keeping
/// objectives with unsupported or nonsmooth sensitivities away from a
/// fabricated gradient contract.
pub fn optimize_pc_fan_with_derivative_policy(
    config: PcFanConfig,
    policy: PcFanDerivativePolicy,
    maxiter: usize,
    popsize_factor: usize,
    seed: u64,
) -> OptimizedPcFan {
    match policy {
        PcFanDerivativePolicy::FiniteDifferenceFullObjective => optimize_pc_fan(config),
        PcFanDerivativePolicy::ReducedOrderDiscreteAdjoint => {
            optimize_pc_fan_reduced_adjoint(config)
        }
        PcFanDerivativePolicy::DerivativeFree => {
            optimize_pc_fan_de(config, maxiter, popsize_factor, seed)
        }
    }
}

/// One DE experiment within a multistart run.
#[derive(Debug, Clone)]
pub struct MultiDEExperiment {
    pub seed: u64,
    pub result: OptimizedPcFan,
}

/// Multistart-DE result: a vector of `(seed, OptimizedPcFan)` pairs
/// plus the index of the best-objective experiment.
#[derive(Debug, Clone)]
pub struct MultiDEResult {
    pub experiments: Vec<MultiDEExperiment>,
    pub best_index: usize,
}

impl MultiDEResult {
    /// The best-objective experiment.
    pub fn best(&self) -> &MultiDEExperiment {
        &self.experiments[self.best_index]
    }
}

/// Run `experiments` independent L-SHADE DE optimisations in parallel,
/// each pinned to its own rayon thread pool of `cores_per_experiment`
/// threads, and return all results plus the index of the best.
///
/// Seeds are derived deterministically from `base_seed`: experiment `i`
/// uses `base_seed.wrapping_add(i as u64)`. This makes the whole
/// multistart run reproducible from a single seed.
///
/// Each experiment runs in its own `std::thread`, with a dedicated
/// rayon pool installed via `pool.install(...)`. Inside the DE, the
/// `math-optimisation` population evaluation uses `par_iter()`, which
/// inherits the ambient pool — so per-experiment fan-out stays bounded
/// to `cores_per_experiment`. Between experiments there is no shared
/// state and no rayon contention.
///
/// # Panics
///
/// Fallible multi-experiment differential evolution: `experiments == 0`, a
/// pool-construction failure, or a worker panic returns a
/// `PcFanFanCurveError::OptimizerFailure` instead of aborting the process.
pub fn try_optimize_pc_fan_multi_de(
    config: PcFanConfig,
    maxiter: usize,
    popsize_factor: usize,
    base_seed: u64,
    experiments: usize,
    cores_per_experiment: usize,
) -> Result<MultiDEResult, crate::evaluation::PcFanFanCurveError> {
    use crate::evaluation::PcFanFanCurveError::OptimizerFailure;
    if experiments == 0 {
        return Err(OptimizerFailure(
            "multi-DE requires at least one experiment".to_string(),
        ));
    }
    let cores = cores_per_experiment.max(1);

    // Shared experiment body used by both paths below so the serial
    // fast-path result is identical to the parallel one (seed base + i,
    // per-experiment pool of `cores` threads).
    let run_experiment =
        |i: usize| -> Result<MultiDEExperiment, crate::evaluation::PcFanFanCurveError> {
            let cfg = config.clone();
            let seed = base_seed.wrapping_add(i as u64);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(cores)
                .thread_name(move |idx| format!("pcfan-de-{i}-{idx}"))
                .build()
                .map_err(|error| {
                    OptimizerFailure(format!(
                        "failed to build per-experiment rayon pool: {error}"
                    ))
                })?;
            let result = pool.install(|| optimize_pc_fan_de(cfg, maxiter, popsize_factor, seed));
            Ok(MultiDEExperiment { seed, result })
        };

    // A single experiment pays only pool-construction and thread-spawn
    // overhead for the parallel path, so it runs in-process instead.
    let experiments: Vec<MultiDEExperiment> = if crate::parallelism::use_pcfan_parallel(
        experiments,
        crate::parallelism::PCFAN_EXPERIMENTS_PER_THREAD,
    ) {
        // Scoped threads so the shared experiment body can borrow
        // `config` from this frame; results stay in index order.
        std::thread::scope(
            |scope| -> Result<Vec<MultiDEExperiment>, crate::evaluation::PcFanFanCurveError> {
                let handles: Vec<_> = (0..experiments)
                    .map(|i| scope.spawn(move || run_experiment(i)))
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| match handle.join() {
                        Ok(outcome) => outcome,
                        Err(error) => Err(OptimizerFailure(format!(
                            "multi-DE worker thread panicked: {error:?}"
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()
            },
        )?
    } else {
        (0..experiments)
            .map(run_experiment)
            .collect::<Result<Vec<_>, _>>()?
    };

    let best_index = experiments
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            b.result
                .is_feasible()
                .cmp(&a.result.is_feasible())
                .then_with(|| {
                    a.result
                        .evaluation
                        .objective
                        .partial_cmp(&b.result.evaluation.objective)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        })
        .map(|(i, _)| i)
        .ok_or_else(|| OptimizerFailure("at least one experiment must run".to_string()))?;

    Ok(MultiDEResult {
        experiments,
        best_index,
    })
}

/// Compatibility multi-experiment differential evolution.
///
/// # Panics
///
/// Panics on the same conditions [`try_optimize_pc_fan_multi_de`] reports as
/// `Err` (`experiments == 0`, pool/worker failure); prefer the `try_` variant
/// in library code.
pub fn optimize_pc_fan_multi_de(
    config: PcFanConfig,
    maxiter: usize,
    popsize_factor: usize,
    base_seed: u64,
    experiments: usize,
    cores_per_experiment: usize,
) -> MultiDEResult {
    try_optimize_pc_fan_multi_de(
        config,
        maxiter,
        popsize_factor,
        base_seed,
        experiments,
        cores_per_experiment,
    )
    .expect("multi-DE run failed")
}

pub fn optimize_pc_fan(config: PcFanConfig) -> OptimizedPcFan {
    optimize_pc_fan_with_gradient(config, false)
}

/// Optimize the reduced-order model with its declared scalar discrete
/// adjoint rather than the full-objective finite-difference gradient.
pub fn optimize_pc_fan_reduced_adjoint(config: PcFanConfig) -> OptimizedPcFan {
    optimize_pc_fan_with_gradient(config, true)
}

fn optimize_pc_fan_with_gradient(config: PcFanConfig, use_reduced_adjoint: bool) -> OptimizedPcFan {
    // See the corresponding L-SHADE guard above. L-BFGS-B's local finite
    // differences would otherwise evaluate unconverged CFD states as if they
    // were legitimate objective values.
    if config.cfd.flow_model.is_cfd() {
        return cfd_optimization_rejected(config);
    }
    let baseline = EvaluatedDesign::from_design(&config, config.default_design_variables());

    let x0 = baseline.design_variables.to_vector().to_vec();
    let bounds = design_bounds(&config);

    let mut active_config = config.clone();
    let mut current_x = x0.clone();
    let mut last_result: Option<Result<OptimResult<f64>, OptimError>> = None;
    let mut total_iterations = 0usize;
    let mut total_fun_evals = 0usize;
    let mut total_grad_evals = 0usize;

    for outer in 0..=MAX_WARM_RESTARTS {
        let lbfgsb_config = LbfgsbConfig {
            max_iterations: active_config.optimizer.max_iterations,
            grad_tolerance: active_config.optimizer.gradient_tolerance,
            fun_tolerance: active_config.optimizer.objective_tolerance,
            memory: active_config.optimizer.memory,
            bounds: bounds.clone(),
        };
        let objective = |x: &[f64]| objective_for_vector(&active_config, x);
        let gradient = |x: &[f64]| {
            if use_reduced_adjoint {
                reduced_adjoint_gradient_for_vector(&active_config, x)
            } else {
                gradient_for_vector(&active_config, x)
            }
        };
        let result = lbfgsb(&objective, &gradient, &current_x, &lbfgsb_config);

        if let Ok(ok) = &result {
            current_x = ok.x.clone();
            total_iterations = total_iterations.saturating_add(ok.iterations);
            total_fun_evals = total_fun_evals.saturating_add(ok.fun_evals);
            total_grad_evals = total_grad_evals.saturating_add(ok.grad_evals);
        }

        let inside_band = airflow_inside_band(&active_config, &current_x);
        last_result = Some(result);
        if inside_band || outer == MAX_WARM_RESTARTS {
            break;
        }
        // Escalate the airflow penalty and warm-restart from the current
        // design. The new run sees a stiffer landscape that pushes toward
        // the band.
        active_config.weights.airflow_penalty_weight *= WARM_RESTART_PENALTY_FACTOR;
    }

    let result = last_result.expect("warm-restart loop must produce at least one result");
    let aggregated = AggregatedSummary {
        total_iterations,
        total_fun_evals,
        total_grad_evals,
    };
    let candidate = OptimizationCandidate::from_result(&config, &baseline, result, aggregated);
    let OptimizationCandidate {
        evaluated: mut candidate_evaluated,
        mut summary,
    } = candidate;
    if !candidate_evaluated.evaluation.is_feasible(&config) {
        let restored =
            restore_pc_fan_feasibility(&config, &candidate_evaluated.design_variables, 24);
        candidate_evaluated = EvaluatedDesign::from_design(&config, restored.design_variables);
    }
    let selected = select_for_promotion(&config, baseline, candidate_evaluated);
    summary.best_objective = selected.evaluation.objective;
    summary.feasible = selected.evaluation.is_feasible(&config);
    if !summary.feasible {
        summary.converged = false;
        summary.status = OptimStatus::RestorationFailed;
    }

    OptimizedPcFan {
        config,
        design_variables: selected.design_variables,
        geometry: selected.geometry,
        evaluation: selected.evaluation,
        optimization: summary,
    }
}

/// Construct an explicitly non-promotable result when a caller asks the
/// infallible reduced-order optimizers to operate on CFD. This never invokes a
/// CFD solve or uses its fields as an objective: callers can inspect the
/// geometry for diagnostics, but `is_feasible()` is necessarily false and the
/// numerical-error status must not be treated as a completed optimization.
fn cfd_optimization_rejected(config: PcFanConfig) -> OptimizedPcFan {
    let design_variables = config.default_design_variables();
    let geometry = config.build_geometry(&design_variables);
    let evaluation = PcFanEvaluation::geometry_only(&config, &geometry);
    let baseline_objective = evaluation.objective;
    OptimizedPcFan {
        config,
        design_variables,
        geometry,
        evaluation,
        optimization: PcFanOptimizationSummary {
            baseline_objective,
            best_objective: baseline_objective,
            iterations: 0,
            function_evaluations: 0,
            gradient_evaluations: 0,
            feasible: false,
            fan_curve_objective: None,
            fan_curve_feasible: true,
            converged: false,
            status: OptimStatus::NumericalError,
        },
    }
}

#[derive(Clone, Copy)]
struct AggregatedSummary {
    total_iterations: usize,
    total_fun_evals: usize,
    total_grad_evals: usize,
}

fn airflow_inside_band(config: &PcFanConfig, x: &[f64]) -> bool {
    let design = vector_to_design(config, x);
    let evaluation = evaluate_pc_fan(config, &design);
    let target = config.operating_point.target_airflow_cfm;
    let band = config.weights.airflow_band_fraction.max(0.0) * target.abs();
    (evaluation.airflow_cfm - target).abs() <= band
}

#[derive(Clone)]
struct EvaluatedDesign {
    design_variables: PcFanDesignVariables,
    geometry: PcFanGeometry,
    evaluation: PcFanEvaluation,
}

impl EvaluatedDesign {
    fn from_design(config: &PcFanConfig, design_variables: PcFanDesignVariables) -> Self {
        let geometry = config.build_geometry(&design_variables);
        let evaluation = PcFanEvaluation::from_design(config, &design_variables, &geometry);
        let evaluation =
            attach_operating_point_spread(config, &design_variables, &geometry, evaluation);

        Self {
            design_variables,
            geometry,
            evaluation,
        }
    }
}

struct OptimizationCandidate {
    evaluated: EvaluatedDesign,
    summary: PcFanOptimizationSummary,
}

impl OptimizationCandidate {
    fn from_result(
        config: &PcFanConfig,
        baseline: &EvaluatedDesign,
        result: Result<OptimResult<f64>, OptimError>,
        aggregated: AggregatedSummary,
    ) -> Self {
        match result {
            Ok(result) => Self::from_success(config, baseline, result, aggregated),
            Err(_) => Self::from_failure(config, baseline, aggregated),
        }
    }

    fn from_success(
        config: &PcFanConfig,
        baseline: &EvaluatedDesign,
        result: OptimResult<f64>,
        aggregated: AggregatedSummary,
    ) -> Self {
        let evaluated = EvaluatedDesign::from_design(config, vector_to_design(config, &result.x));
        let best_objective = baseline
            .evaluation
            .objective
            .min(evaluated.evaluation.objective);
        let summary = PcFanOptimizationSummary {
            baseline_objective: baseline.evaluation.objective,
            best_objective,
            iterations: aggregated.total_iterations,
            function_evaluations: aggregated.total_fun_evals,
            gradient_evaluations: aggregated.total_grad_evals,
            feasible: evaluated.evaluation.is_feasible(config),
            fan_curve_objective: None,
            fan_curve_feasible: true,
            converged: result.converged,
            status: result.status,
        };

        Self { evaluated, summary }
    }

    fn from_failure(
        config: &PcFanConfig,
        baseline: &EvaluatedDesign,
        aggregated: AggregatedSummary,
    ) -> Self {
        let evaluated = EvaluatedDesign {
            design_variables: baseline.design_variables.clone(),
            geometry: baseline.geometry.clone(),
            evaluation: baseline.evaluation.clone(),
        };
        let summary = PcFanOptimizationSummary {
            baseline_objective: baseline.evaluation.objective,
            best_objective: baseline.evaluation.objective,
            iterations: aggregated.total_iterations,
            function_evaluations: aggregated.total_fun_evals.max(1),
            gradient_evaluations: aggregated.total_grad_evals,
            feasible: baseline.evaluation.is_feasible(config),
            fan_curve_objective: None,
            fan_curve_feasible: true,
            converged: false,
            status: OptimStatus::NumericalError,
        };

        Self { evaluated, summary }
    }
}

fn select_for_promotion(
    config: &PcFanConfig,
    baseline: EvaluatedDesign,
    candidate: EvaluatedDesign,
) -> EvaluatedDesign {
    let baseline_feasible = baseline.evaluation.is_feasible(config);
    let candidate_feasible = candidate.evaluation.is_feasible(config);
    match (baseline_feasible, candidate_feasible) {
        (false, true) => candidate,
        (true, false) => baseline,
        (true, true) if candidate.evaluation.objective <= baseline.evaluation.objective => {
            candidate
        }
        (true, true) => baseline,
        (false, false) if candidate.evaluation.objective <= baseline.evaluation.objective => {
            candidate
        }
        (false, false) => baseline,
    }
}

fn objective_for_vector(config: &PcFanConfig, x: &[f64]) -> f64 {
    let design = vector_to_design(config, x);
    let geometry = config.build_geometry(&design);
    let evaluation = PcFanEvaluation::from_design(config, &design, &geometry);
    let evaluation = attach_operating_point_spread(config, &design, &geometry, evaluation);
    evaluation.objective
}

fn gradient_for_vector(config: &PcFanConfig, x: &[f64]) -> Vec<f64> {
    finite_difference_gradient_for_vector(config, x)
}

fn reduced_adjoint_gradient_for_vector(config: &PcFanConfig, x: &[f64]) -> Vec<f64> {
    let design = vector_to_design(config, x);
    crate::evaluation::solve_pc_fan_shape_gradient_reduced_adjoint(config, &design)
        .expect("reduced adjoint optimizer must use a reduced-order flow model")
        .objective_gradient
        .to_vec()
}

fn finite_difference_gradient_for_vector(config: &PcFanConfig, x: &[f64]) -> Vec<f64> {
    let bounds = design_bounds(config);
    if crate::parallelism::use_pcfan_parallel(
        x.len(),
        crate::parallelism::PCFAN_FD_COMPONENTS_PER_THREAD,
    ) {
        (0..x.len())
            .into_par_iter()
            .map(|index| finite_difference_gradient_component(config, x, index, &bounds))
            .collect()
    } else {
        (0..x.len())
            .map(|index| finite_difference_gradient_component(config, x, index, &bounds))
            .collect()
    }
}

fn finite_difference_gradient_component(
    config: &PcFanConfig,
    x: &[f64],
    index: usize,
    bounds: &[Bound<f64>],
) -> f64 {
    let value = x[index];
    let step = finite_difference_step_for_variable(config, index, value);
    let bound = bounds[index];
    let lower = bound.lower.unwrap_or(f64::NEG_INFINITY);
    let upper = bound.upper.unwrap_or(f64::INFINITY);
    let plus_value = (value + step).min(upper);
    let minus_value = (value - step).max(lower);
    let can_step_plus = plus_value > value;
    let can_step_minus = minus_value < value;

    if can_step_plus && can_step_minus {
        let mut plus = x.to_vec();
        let mut minus = x.to_vec();
        plus[index] = plus_value;
        minus[index] = minus_value;
        let (plus_objective, minus_objective) = rayon::join(
            || objective_for_vector(config, &plus),
            || objective_for_vector(config, &minus),
        );
        return (plus_objective - minus_objective) / (plus_value - minus_value);
    }

    let baseline = objective_for_vector(config, x);
    if can_step_plus {
        let mut plus = x.to_vec();
        plus[index] = plus_value;
        return (objective_for_vector(config, &plus) - baseline) / (plus_value - value);
    }
    if can_step_minus {
        let mut minus = x.to_vec();
        minus[index] = minus_value;
        return (baseline - objective_for_vector(config, &minus)) / (value - minus_value);
    }

    0.0
}

fn finite_difference_step_for_variable(config: &PcFanConfig, index: usize, value: f64) -> f64 {
    let scale = if index < BLADE_PATH_CONTROL_COUNT * 2 {
        0.01
    } else if index < BLADE_PATH_CONTROL_COUNT * 2 + BLADE_SECTION_STATION_COUNT * 4 {
        match (index - BLADE_PATH_CONTROL_COUNT * 2) % 4 {
            0 => config.dimensions.depth_m,
            1 => 0.004,
            2 => 0.001,
            _ => 10.0,
        }
    } else {
        0.003
    };

    let relative_step = config.optimizer.finite_difference_step.abs().max(1.0e-8);
    (relative_step * value.abs().max(scale)).max(1.0e-10)
}

pub(crate) fn vector_to_design(config: &PcFanConfig, x: &[f64]) -> PcFanDesignVariables {
    debug_assert_eq!(x.len(), DESIGN_VARIABLE_COUNT);
    let bounds = design_bounds(config);
    let mut i = 0;
    let next = |index: &mut usize| {
        let bound = bounds[*index];
        let mut value = x[*index];
        if let Some(lower) = bound.lower {
            value = value.max(lower);
        }
        if let Some(upper) = bound.upper {
            value = value.min(upper);
        }
        *index += 1;
        value
    };

    let mut values = [0.0; DESIGN_VARIABLE_COUNT];
    for value in &mut values {
        *value = next(&mut i);
    }

    let mut design = PcFanDesignVariables::from_vector(values);
    // Hard-pin the hub-attached root's tangential lean. The bounds make
    // the optimiser see a flat landscape; this assignment ensures the
    // geometry is exactly anchored even under accumulated floating-point
    // noise. Axial offset is left under optimiser control (it doesn't
    // affect radial attachment to the hub).
    design.blade_path[0].tangential_offset_m = 0.0;
    // Hard-pin frozen DOFs back to their baseline so the optimiser can't
    // accidentally drift into invalid territory through accumulated
    // floating-point noise. The bounds make the optimiser see a flat
    // landscape; this assignment makes the geometry truly identical to
    // the baseline.
    if config.optimizer.freeze_shroud {
        let baseline = config.default_design_variables();
        design.shroud_radius_offsets_m = baseline.shroud_radius_offsets_m;
    }
    design
}

pub(crate) fn design_bounds(config: &PcFanConfig) -> Vec<Bound<f64>> {
    let mut bounds = Vec::with_capacity(DESIGN_VARIABLE_COUNT);
    // Path bounds — generous so the optimiser can produce strongly-swept
    // blades. The actual feasible region is shaped by the admissibility
    // penalties (`shroud_clearance_penalty`, `frame_containment_penalty`)
    // not by these box bounds: backward sweep beyond ~15 mm pushes the
    // tip past the shroud at the default geometry, and the penalty pulls
    // the optimiser back. The wide box here is what gives L-BFGS-B
    // enough room to *find* the curved optimum in the first place.
    for i in 0..BLADE_PATH_CONTROL_COUNT {
        if i == 0 {
            // Hub-attached root. The blade is mechanically welded to the
            // hub at span_fraction = 0, so its in-plane tangential lean
            // there must be zero — `polar_point` places vertices at
            // `√(R_hub² + tang_offset²)` from the rotation axis, so any
            // tangential lean at the root *increases* the apparent
            // radius and visually disconnects the blade from the rotor.
            // The axial offset is *not* pinned here — the baseline
            // anchors the blade's axial centre on the hub centre (which
            // is offset from z=0 by `hub_center_z`), and that knob is a
            // legitimate design degree of freedom.
            bounds.push(Bound::both(0.0, 0.0));
            bounds.push(Bound::both(-0.012, 0.012));
            continue;
        }
        // Tangential lean (plan-view sweep). Negative = backward (trails
        // behind rotation), positive = forward. ±30 mm — 2.5× the old
        // ±12 mm bound so the classic backward-skewed fan profile is
        // reachable from the baseline.
        bounds.push(Bound::both(-0.030, 0.030));
        // Axial sweep (forward/backward bow along the rotation axis).
        // ±12 mm matches the depth of the frame; further than that the
        // blade leaves the cavity.
        bounds.push(Bound::both(-0.012, 0.012));
    }
    let camber_lower = if config.optimizer.require_positive_camber {
        0.0
    } else {
        -0.008
    };
    for _ in 0..BLADE_SECTION_STATION_COUNT {
        // Chord upper bound widened to 2× depth so the optimiser can
        // grow the blade surface well beyond the chord-taper baseline.
        // Lower bound 0.20× depth keeps blades non-degenerate.
        bounds.push(Bound::both(
            0.20 * config.dimensions.depth_m,
            2.00 * config.dimensions.depth_m,
        ));
        // Camber upper bound 12 mm — about 50 % of chord at the longest
        // station. Real axial fans rarely use more than ~8 % of chord,
        // but the wider bound lets the optimiser explore aggressive
        // cambered profiles before the admissibility penalties pull it
        // back. The objective itself penalises excess camber via
        // `mean_section_noise_factor`.
        bounds.push(Bound::both(camber_lower, 0.012));
        bounds.push(Bound::both(0.0004, 0.0035));
        // Twist bounds widened to `[−30°, +60°]` (was `[−15°, +35°]`)
        // so the optimiser can produce a steep root pitch and an almost
        // flat tip — the classic axial-fan profile in its full range.
        bounds.push(Bound::both(-30.0, 60.0));
    }

    // Radial-envelope controls are bounded symmetrically around the nominal
    // hub/tip radii. Keep enough margin that every combination of the two
    // independent controls preserves a positive hub-to-tip gap, while the
    // tip control also remains inside the nominal shroud opening.
    let dimensions = config.dimensions;
    const RADIAL_ENVELOPE_BOUND_SAFETY_M: f64 = 1.0e-6;
    let radial_gap_limit = ((dimensions.blade_tip_radius_m - dimensions.hub_radius_m)
        - 2.0 * (RADIAL_ENVELOPE_MIN_GAP_M + RADIAL_ENVELOPE_BOUND_SAFETY_M))
        .max(0.0)
        * 0.5;
    let hub_offset_limit = RADIAL_ENVELOPE_OFFSET_LIMIT_M.min(radial_gap_limit).min(
        (dimensions.hub_radius_m - RADIAL_ENVELOPE_MIN_GAP_M - RADIAL_ENVELOPE_BOUND_SAFETY_M)
            .max(0.0),
    );
    let tip_offset_limit = RADIAL_ENVELOPE_OFFSET_LIMIT_M.min(radial_gap_limit).min(
        (dimensions.shroud_inner_radius_m
            - dimensions.blade_tip_radius_m
            - RADIAL_ENVELOPE_MIN_GAP_M
            - RADIAL_ENVELOPE_BOUND_SAFETY_M)
            .max(0.0),
    );
    bounds.push(Bound::both(-hub_offset_limit, hub_offset_limit));
    bounds.push(Bound::both(-tip_offset_limit, tip_offset_limit));

    if config.optimizer.freeze_shroud {
        // Pin every shroud DOF at the baseline (zero offset) by giving
        // L-BFGS-B a degenerate single-point box.
        for _ in 0..SHROUD_CONTROL_COUNT {
            bounds.push(Bound::both(0.0, 0.0));
        }
    } else {
        let max_shroud_offset_m = (0.5 * config.dimensions.frame_side_m
            - config.dimensions.frame_wall_thickness_m
            - config.dimensions.shroud_wall_thickness_m
            - config.dimensions.shroud_inner_radius_m)
            .max(0.0);
        for _ in 0..SHROUD_CONTROL_COUNT {
            bounds.push(Bound::both(-0.003, max_shroud_offset_m));
        }
    }
    bounds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blade_root_tangential_is_pinned_to_zero() {
        // The first design-vector entry controls
        // `blade_path[0].tangential_offset_m` — the hub-attached root's
        // in-plane lean. Must be a degenerate single-point box at 0 so
        // the optimiser can never detach the blade root by chasing dB.
        // Entry 1 (axial offset) keeps a normal range — it's a
        // legitimate design knob that doesn't affect hub attachment.
        let config = PcFanConfig::default_120mm();
        let bounds = design_bounds(&config);
        assert_eq!(
            bounds[0].lower,
            Some(0.0),
            "blade_path[0] tangential lower bound must be 0"
        );
        assert_eq!(
            bounds[0].upper,
            Some(0.0),
            "blade_path[0] tangential upper bound must be 0"
        );
        let axial = bounds[1];
        assert!(
            axial.lower.unwrap() < axial.upper.unwrap(),
            "blade_path[0] axial offset should keep its full range"
        );
    }

    #[test]
    fn vector_to_design_pins_blade_root_tangential_under_garbage_input() {
        // Even when L-SHADE injects garbage into the first entry,
        // vector_to_design must hard-pin tangential back to zero (axial
        // is left intact).
        let config = PcFanConfig::default_120mm();
        let mut x = config.default_design_variables().to_vector().to_vec();
        x[0] = 0.025;
        let design = vector_to_design(&config, &x);
        assert_eq!(
            design.blade_path[0].tangential_offset_m, 0.0,
            "blade_path[0] tangential must be hard-pinned to 0"
        );
    }

    #[test]
    fn frozen_shroud_bounds_are_degenerate() {
        let mut config = PcFanConfig::default_120mm();
        config.optimizer.freeze_shroud = true;
        let bounds = design_bounds(&config);
        // Last SHROUD_CONTROL_COUNT bounds must be a single point at zero.
        let n = bounds.len();
        for b in &bounds[n - SHROUD_CONTROL_COUNT..] {
            assert_eq!(b.lower, Some(0.0), "frozen shroud lower != 0");
            assert_eq!(b.upper, Some(0.0), "frozen shroud upper != 0");
        }
    }

    #[test]
    fn unfrozen_shroud_bounds_have_range() {
        let mut config = PcFanConfig::default_120mm();
        config.optimizer.freeze_shroud = false;
        let bounds = design_bounds(&config);
        let n = bounds.len();
        // First (deepest) shroud entry must allow some movement.
        let first_shroud = &bounds[n - SHROUD_CONTROL_COUNT];
        let lower = first_shroud.lower.expect("lower bound");
        let upper = first_shroud.upper.expect("upper bound");
        assert!(
            upper > lower,
            "shroud bound has no range: [{lower},{upper}]"
        );
    }

    #[test]
    fn radial_envelope_bounds_are_bounded_before_shroud_controls() {
        let config = PcFanConfig::default_120mm();
        let bounds = design_bounds(&config);
        let shroud_start = bounds.len() - SHROUD_CONTROL_COUNT;
        let radial_start = shroud_start - RADIAL_ENVELOPE_CONTROL_COUNT;

        for bound in &bounds[radial_start..shroud_start] {
            let lower = bound.lower.expect("radial lower bound");
            let upper = bound.upper.expect("radial upper bound");
            assert!(lower < upper);
            assert!(lower.abs() <= RADIAL_ENVELOPE_OFFSET_LIMIT_M);
            assert!(upper.abs() <= RADIAL_ENVELOPE_OFFSET_LIMIT_M);
        }
        assert_eq!(
            bounds[shroud_start].lower,
            Some(0.0),
            "shroud controls must remain the trailing vector block"
        );
    }

    #[test]
    fn vector_to_design_clamps_radial_envelope_controls() {
        let config = PcFanConfig::default_120mm();
        let bounds = design_bounds(&config);
        let shroud_start = bounds.len() - SHROUD_CONTROL_COUNT;
        let radial_start = shroud_start - RADIAL_ENVELOPE_CONTROL_COUNT;
        let mut values = config.default_design_variables().to_vector();
        values[radial_start] = 1.0;
        values[radial_start + 1] = -1.0;

        let design = vector_to_design(&config, &values);
        assert!(design.hub_radius_offset_m <= RADIAL_ENVELOPE_OFFSET_LIMIT_M);
        assert!(design.tip_radius_offset_m >= -RADIAL_ENVELOPE_OFFSET_LIMIT_M);
        assert!(config.try_build_geometry(&design).is_ok());
    }

    #[test]
    fn vector_to_design_pins_shroud_when_frozen() {
        let config = PcFanConfig::default_120mm();
        // Build an x vector with garbage shroud values; vector_to_design
        // must replace them with the baseline (zero offsets).
        let mut x = config.default_design_variables().to_vector().to_vec();
        let n = x.len();
        for value in x.iter_mut().skip(n - SHROUD_CONTROL_COUNT) {
            *value = 0.123;
        }
        let design = vector_to_design(&config, &x);
        for offset in &design.shroud_radius_offsets_m {
            assert_eq!(*offset, 0.0, "shroud not pinned: {offset}");
        }
    }

    #[test]
    fn positive_camber_bound_is_zero() {
        let mut config = PcFanConfig::default_120mm();
        config.optimizer.require_positive_camber = true;
        let bounds = design_bounds(&config);
        // Camber is field index 1 within each blade-section block.
        let path_count = BLADE_PATH_CONTROL_COUNT * 2;
        for s in 0..BLADE_SECTION_STATION_COUNT {
            let camber_idx = path_count + s * 4 + 1;
            let lower = bounds[camber_idx].lower.expect("camber lower");
            assert!(
                lower >= 0.0,
                "camber lower bound at station {s} is negative: {lower}"
            );
        }
    }

    #[test]
    fn shroud_clearance_penalty_zero_at_baseline() {
        // With the default geometry, blades sit inside the shroud with
        // ~2 mm clearance — the penalty should be exactly zero.
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let penalty = crate::evaluation::shroud_clearance_penalty_for_test(&config, &geometry);
        assert_eq!(penalty, 0.0);
    }

    #[test]
    fn shroud_clearance_penalty_positive_when_violated() {
        // Pick a clearance requirement that exceeds the natural tip-to-
        // shroud gap on the default geometry. The default tip radius is
        // 50 mm, shroud inner radius is 57 mm — a 12 mm clearance ask
        // is impossible to satisfy and must trigger a positive penalty.
        let mut config = PcFanConfig::default_120mm();
        config.weights.shroud_clearance_min_m = 0.012;
        config.weights.shroud_clearance_penalty_weight = 1.0;
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let penalty = crate::evaluation::shroud_clearance_penalty_for_test(&config, &geometry);
        assert!(penalty > 0.0, "expected positive penalty, got {penalty}");
    }

    #[test]
    fn promotion_rejects_infeasible_candidate_even_when_its_penalty_is_zero() {
        let mut config = PcFanConfig::default_120mm();
        config.weights.collision_penalty_weight = 0.0;
        let reference = EvaluatedDesign::from_design(&config, config.default_design_variables());
        config.operating_point.target_airflow_cfm = reference.evaluation.airflow_cfm;
        let baseline = EvaluatedDesign::from_design(&config, config.default_design_variables());
        assert!(
            baseline.evaluation.is_feasible(&config),
            "baseline must be feasible: {:?}",
            baseline.evaluation.constraint_margins
        );

        let mut candidate = baseline.clone();
        candidate.evaluation.objective = baseline.evaluation.objective - 1.0;
        candidate.evaluation.collision_status.is_valid = false;
        candidate
            .evaluation
            .constraint_margins
            .invalid_intersection_count = 1;
        candidate.evaluation.collision_penalty = 0.0;
        assert!(!candidate.evaluation.is_feasible(&config));

        let selected = select_for_promotion(&config, baseline.clone(), candidate);
        assert_eq!(selected.evaluation.objective, baseline.evaluation.objective);
        assert!(selected.evaluation.is_feasible(&config));
    }

    #[test]
    fn de_optimiser_runs_and_improves_on_baseline() {
        // Tiny DE budget so the test is fast (≈ 1 s on the reduced-order
        // path). The point isn't to find the optimum — it's to verify
        // the DE wiring returns a valid `OptimizedPcFan` whose objective
        // is at most the baseline.
        let mut config = PcFanConfig::default_120mm();
        config.mesh = PcFanMeshResolution {
            radial_segments: 3,
            chord_segments: 2,
            circumferential_segments: 8,
            harmonic_count: 2,
        };
        let baseline = evaluate_pc_fan(&config, &config.default_design_variables());
        let result = optimize_pc_fan_de(config.clone(), 5, 6, 0);
        assert!(
            result.evaluation.objective <= baseline.objective + 1.0e-9,
            "DE made the objective worse: {} > {}",
            result.evaluation.objective,
            baseline.objective
        );
        assert!(result.optimization.function_evaluations > 0);
        assert_eq!(result.optimization.gradient_evaluations, 0);
    }

    #[test]
    fn multi_de_runs_three_experiments_and_picks_best() {
        // Three tiny DE runs with distinct seeds. Verifies that the
        // multistart wiring (a) produces one `OptimizedPcFan` per
        // experiment, (b) reports a `best_index` whose objective is the
        // minimum of the three, and (c) doesn't make the objective
        // worse than the baseline for *any* seed.
        let mut config = PcFanConfig::default_120mm();
        config.mesh = PcFanMeshResolution {
            radial_segments: 3,
            chord_segments: 2,
            circumferential_segments: 8,
            harmonic_count: 2,
        };
        let baseline = evaluate_pc_fan(&config, &config.default_design_variables());

        let multi = optimize_pc_fan_multi_de(config.clone(), 5, 6, 0, 3, 2);

        assert_eq!(multi.experiments.len(), 3);

        let error = try_optimize_pc_fan_multi_de(config.clone(), 5, 6, 0, 0, 2).unwrap_err();
        assert!(
            format!("{error:?}").contains("at least one experiment"),
            "zero experiments must fail closed, got: {error:?}"
        );
        let seeds: Vec<u64> = multi.experiments.iter().map(|e| e.seed).collect();
        assert_eq!(seeds, vec![0, 1, 2], "seeds must be base + i");
        assert!(multi.best_index < 3);

        for exp in &multi.experiments {
            assert!(
                exp.result.evaluation.objective <= baseline.objective + 1.0e-9,
                "seed {} made the objective worse: {} > {}",
                exp.seed,
                exp.result.evaluation.objective,
                baseline.objective
            );
        }

        let min_obj = multi
            .experiments
            .iter()
            .map(|e| e.result.evaluation.objective)
            .fold(f64::INFINITY, f64::min);
        assert_eq!(
            multi.best().result.evaluation.objective,
            min_obj,
            "best_index does not point at the minimum-objective experiment",
        );
    }

    #[test]
    fn cfd_optimizers_refuse_to_promote_an_unchecked_objective() {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 2,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 2,
            },
            cfd: PcFanCfdConfig {
                flow_model: PcFanFlowModel::IncompressibleCfd,
                cells_x: 2,
                cells_y: 2,
                cells_z: 2,
                max_picard_iterations: 1,
                nonlinear_tolerance: 1.0e-4,
                ..PcFanCfdConfig::default()
            },
            ..PcFanConfig::default_120mm()
        };
        let lbfgsb = optimize_pc_fan(config.clone());
        let de = optimize_pc_fan_de(config, 1, 5, 7);

        for result in [&lbfgsb, &de] {
            assert_eq!(result.optimization.status, OptimStatus::NumericalError);
            assert_eq!(result.optimization.function_evaluations, 0);
            assert_eq!(result.optimization.gradient_evaluations, 0);
            assert!(!result.is_feasible());
        }
    }

    #[test]
    fn feasibility_restoration_projects_design_back_into_bounds() {
        let mut config = PcFanConfig::bench_120mm();
        config.weights.airflow_band_fraction = 1.0;
        let mut values = config.default_design_variables().to_vector();
        values[0] = 0.010;
        let invalid = PcFanDesignVariables::from_vector(values);

        let restored = restore_pc_fan_feasibility(&config, &invalid, 8);

        assert_eq!(
            restored.design_variables.blade_path[0].tangential_offset_m,
            0.0
        );
        assert!(restored.evaluation.is_feasible(&config));
        assert!(restored.restored);
        assert_eq!(restored.function_evaluations, 1);
    }

    #[test]
    fn feasibility_restoration_reports_unreachable_hard_constraint() {
        let mut config = PcFanConfig::default_120mm();
        config.weights.shroud_clearance_min_m = 0.012;
        let baseline = evaluate_pc_fan(&config, &config.default_design_variables());
        assert!(!baseline.is_feasible(&config));

        let restored = restore_pc_fan_feasibility(&config, &config.default_design_variables(), 4);

        assert!(!restored.restored);
        assert!(!restored.evaluation.is_feasible(&config));
        assert!(restored.function_evaluations > 1);
        assert!(
            restored
                .evaluation
                .feasibility_violation(&config)
                .is_finite()
        );
    }

    #[test]
    fn reduced_adjoint_policy_routes_optimization_with_a_gradient() {
        let mut config = PcFanConfig::bench_120mm();
        config.weights.airflow_band_fraction = 1.0;
        config.mesh = PcFanMeshResolution {
            radial_segments: 3,
            chord_segments: 2,
            circumferential_segments: 8,
            harmonic_count: 2,
        };
        config.optimizer.max_iterations = 1;

        let result = optimize_pc_fan_with_derivative_policy(
            config,
            PcFanDerivativePolicy::ReducedOrderDiscreteAdjoint,
            1,
            5,
            17,
        );

        assert!(result.optimization.function_evaluations > 0);
        assert!(result.optimization.gradient_evaluations > 0);
    }

    #[test]
    fn derivative_free_policy_routes_optimization_without_a_gradient() {
        let mut config = PcFanConfig::default_120mm();
        config.mesh = PcFanMeshResolution {
            radial_segments: 3,
            chord_segments: 2,
            circumferential_segments: 8,
            harmonic_count: 2,
        };
        let result = optimize_pc_fan_with_derivative_policy(
            config,
            PcFanDerivativePolicy::DerivativeFree,
            1,
            5,
            17,
        );
        assert_eq!(result.optimization.gradient_evaluations, 0);
        assert!(result.optimization.function_evaluations > 0);
    }

    #[test]
    fn derivative_free_policy_rejects_shape_gradient_requests() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        assert!(matches!(
            crate::solve_pc_fan_shape_gradient_with_policy(
                &config,
                &design,
                PcFanDerivativePolicy::DerivativeFree,
            ),
            Err(
                crate::PcFanDerivativePolicyError::ShapeGradientUnavailable {
                    policy: PcFanDerivativePolicy::DerivativeFree,
                }
            )
        ));
    }

    #[test]
    fn fan_curve_optimizer_routes_multi_point_objective_and_preserves_admission() {
        let mut config = PcFanConfig::bench_120mm();
        config.mesh = PcFanMeshResolution {
            radial_segments: 3,
            chord_segments: 2,
            circumferential_segments: 8,
            harmonic_count: 2,
        };
        config.weights.airflow_band_fraction = 1.0;
        let design = config.default_design_variables();
        let fractions = vec![0.0, 0.5, 1.0];
        let curve = crate::evaluate_pc_fan_fan_curve(&config, &design, 600.0, &fractions).unwrap();
        let target = PcFanFanCurveObjectiveConfig {
            rpm: 600.0,
            flow_fractions: fractions,
            target_airflow_cfm: curve.points.iter().map(|point| point.airflow_cfm).collect(),
            target_pressure_rise_pa: curve
                .points
                .iter()
                .map(|point| point.fan_pressure_rise_pa)
                .collect(),
            target_acoustic_level_db: None,
            target_efficiency: None,
            airflow_weight: 1.0,
            pressure_weight: 1.0,
            acoustic_weight: 0.0,
            efficiency_weight: 0.0,
        };

        let result = optimize_pc_fan_fan_curve(config, target, 1, 5, 19)
            .expect("reduced multi-point optimizer should run");
        assert!(result.optimization.function_evaluations > 0);
        assert_eq!(result.optimization.gradient_evaluations, 0);
        assert!(result.optimization.fan_curve_objective.is_some());
        assert!(result.optimization.fan_curve_feasible);
        assert!(result.is_feasible());
    }
}
