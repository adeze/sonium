//! Optimisation algorithms: unconstrained and constrained minimisation.
//!
//! # Unconstrained
//! - [`nelder_mead`] — derivative-free simplex method
//! - [`bfgs`] — full BFGS quasi-Newton
//! - [`lbfgs`] — limited-memory BFGS
//! - [`newton_cg`] — Hessian-free truncated Newton
//! - [`powell`] — derivative-free direction-set method
//! - [`bobyqa`] — model-based trust-region derivative-free (bound-constrained)
//! - [`cma_es`] — bounded full-covariance CMA-ES (global, derivative-free)
//! - [`differential_evolution`] — bounded population-based global search
//! - [`simulated_annealing`] — bounded stochastic annealing
//! - [`random_search`] — deterministic-seed multistart uniform search

//! - [`nonlinear_cg`] — nonlinear conjugate gradient (FR/PR/HS/PR+/Hager–Zhang)
//! - [`quasi_newton`] — SR1 and DFP quasi-Newton updates
//!
//! # Constrained
//! - [`lbfgsb`] — L-BFGS with bound constraints
//! - [`slsqp`] — sequential least-squares programming
//! - [`sqp`] — sequential quadratic programming
//! - [`sqp_advanced`] — SQP with exact Hessian and regularisation
//! - [`trust_region_constrained`] — trust-region constrained
//! - [`ipm`] — primal-dual interior-point with filter line search
//! - [`mma`] — method of moving asymptotes (topology optimisation)
//! - [`augmented_lagrangian`] — augmented Lagrangian with adaptive penalty
//! - [`penalty_method`] — quadratic/barrier/log-barrier penalty
//! - [`epsilon_constraint`] — epsilon-constraint for Pareto fronts

/// Gaussian-process Bayesian optimization (EI, q-EI, qEHVI, constrained).
pub mod bayesian;
pub mod constrained;
pub mod line_search;
/// Pareto multi-objective optimizers (NSGA-II, NSGA-III).
pub mod multiobjective;
pub mod risk;
pub mod types;
pub mod unconstrained;
pub mod util;

pub mod test_functions;

// Re-export core types
pub use risk::{
    CvarResult, KsResult, RiskAggregation, RiskAggregationError, RiskObjective,
    RiskObjectiveEvaluation, ScenarioEvaluation, bfgs_risk, chance_constraint_satisfied,
    empirical_cvar, empirical_cvar_with_gradients, ks_aggregate, ks_aggregate_with_gradients,
};
pub use types::*;

// Re-export line search
pub use line_search::{
    BacktrackingConfig, BrentConfig, BrentResult, BrentStatus, HagerZhangConfig, LineSearchResult,
    StrongWolfeConfig, backtracking, brent_minimize, hager_zhang_line_search, strong_wolfe,
};

// Re-export unconstrained solvers
#[allow(deprecated)]
pub use unconstrained::{
    BfgsConfig, BobyqaConfig, CgVariant, CmaEsConfig, DifferentialEvolutionConfig, LbfgsConfig,
    NelderMeadConfig, NewtonCgConfig, NonlinearCgConfig, PowellConfig, QuasiNewtonConfig,
    QuasiNewtonUpdate, RandomSearchConfig, SimulatedAnnealingConfig, bfgs, bobyqa, cma_es,
    differential_evolution, lbfgs, nelder_mead, newton_cg, nonlinear_cg, powell, quasi_newton,
    random_search, simulated_annealing,
};

// Re-export strategy-rich Differential Evolution API (merged from
// `math-optimisation`). The bare `differential_evolution` free function is
// intentionally NOT re-exported: that name keeps referring to the simple
// `OptimResult`-based solver above, while the rich solver lives at
// `unconstrained::de::differential_evolution`.
pub use unconstrained::de::external_archive::ExternalArchive;
pub use unconstrained::de::{
    AdaptiveConfig, CallbackAction, CallbackFn, Crossover, DEConfig, DEConfigBuilder, DEError,
    DEIntermediate, DEReport, DifferentialEvolution, Init, LinearConstraintHelper, LinearPenalty,
    Mutation, NonlinearConstraintHelper, PenaltyTuple, PolishConfig, Result, ScalarConstraintFn,
    Strategy, Updating, VectorConstraintFn,
};
pub use util::parallel_eval::ParallelConfig;
pub use util::recorder::{OptimizationRecord, OptimizationRecorder};
pub use util::run_recorded::run_recorded_differential_evolution;

// Re-export Levenberg-Marquardt API (merged from `math-optimisation`).
pub use unconstrained::levenberg_marquardt::{
    JacobianFn, LMCallback, LMCallbackAction, LMConfig, LMConfigBuilder, LMError, LMIntermediate,
    LMReport, LMResult, levenberg_marquardt,
};

// Re-export COBYLA API (merged from `math-optimisation`).
pub use constrained::cobyla::{CobylaConfig, CobylaConstraint, CobylaReport, CobylaStopTols};

// Re-export ISRES API (merged from `math-optimisation`).
pub use constrained::isres::{IsresConfig, IsresConstraint, IsresConstraintFn, IsresReport, isres};

// Re-export COBRA + RBF surrogate API (merged from `math-optimisation`).
pub use constrained::cobra::{
    CobraCallback, CobraConfig, CobraConstraint, CobraIntermediate, CobraReport, cobra,
};
pub use util::surrogate::{RbfKind, RbfSurrogate};

// Re-export Bayesian optimization API (merged from `math-optimisation`).
pub use bayesian::{
    BayesAcquisition, BayesOptCallback, BayesOptConfig, BayesOptConstraint, BayesOptConstraintFn,
    BayesOptIntermediate, BayesOptParetoReport, BayesOptReport, BayesParetoSolution,
    ConstrainedBayesOptCallback, ConstrainedBayesOptConfig, ConstrainedBayesOptIntermediate,
    ConstrainedBayesOptReport, bayesian_multi_objective, bayesian_optimization,
    constrained_bayesian_optimization,
};

// Re-export NSGA Pareto API (merged from `math-optimisation`).
pub use multiobjective::{NsgaConfig, NsgaReport, NsgaVariant, ParetoSolution, nsga, nsga2, nsga3};

// Re-export continuous-area loss API (merged from `math-optimisation`).
pub use unconstrained::de::lshade::LShadeConfig;
pub use util::continuous_area::{
    AreaError, AreaScalarisation, Prior, Quadrature, evaluate_area_loss,
};

// Re-export constrained solvers
pub use constrained::{
    AdvancedSqpConfig, AugmentedLagrangianConfig, EpsilonConstraintConfig, HessianRegularisation,
    InnerSolver, IpmConfig, LbfgsbConfig, MmaConfig, MmaVariant, ParetoResult, PenaltyConfig,
    PenaltyType, QpSense, QpSolution, QpStatus, SlsqpConfig, SqpAcceptance, SqpConfig,
    TrustRegionConstrainedConfig, augmented_lagrangian, epsilon_constraint, ipm, lbfgsb, mma,
    penalty_method, slsqp, solve_qp, sqp, sqp_advanced, trust_region_constrained,
};
