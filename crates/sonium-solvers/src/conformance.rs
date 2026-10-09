//! Algorithm conformance records for the public solver surface.
//!
//! The solver crate deliberately contains more algorithms than the automatic
//! compatibility dispatcher uses.  Keeping the assumptions, stopping rule,
//! breakdown policy, and reference source in one machine-readable table makes
//! that distinction explicit and gives QA a stable inventory to audit.  The
//! records are descriptive: they do not replace the checked execution
//! contracts of the individual algorithms.

/// Maturity of an algorithm in the public API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverMaturity {
    /// Safe for the normal production dispatch when its assumptions hold.
    Production,
    /// Public and supported, but requires an explicit method/configuration
    /// choice and is not selected by automatic dispatch.
    Experimental,
    /// Exposed for research/replay and not a production dispatch candidate.
    Research,
}

/// Kind of evidence backing a conformance record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceKind {
    /// An independent implementation or analytic solution is exercised by a
    /// checked test or release QA case.
    Independent,
    /// The implementation is checked against a sibling implementation or
    /// equivalent production path, but not an independent numerical solver.
    CrossChecked,
    /// The algorithm reference is documented, while an independent executable
    /// reference is still a roadmap item.
    DocumentedOnly,
}

/// One conformance row for a public algorithm or solver family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolverConformance {
    /// Stable family name used by QA and release reports.
    pub family: &'static str,
    /// Public algorithm/function name.
    pub algorithm: &'static str,
    /// Maturity and automatic-dispatch policy.
    pub maturity: SolverMaturity,
    /// Matrix/model assumptions that callers must satisfy.
    pub assumptions: &'static str,
    /// Stopping and accuracy contract.
    pub stopping_rule: &'static str,
    /// Behaviour on breakdown, stagnation, or unattainable tolerance.
    pub breakdown_policy: &'static str,
    /// Primary paper, standard, or independent fixture used as reference.
    pub reference: &'static str,
    /// Strength of the currently checked evidence.
    pub reference_kind: ReferenceKind,
    /// Scalar families covered by the checked boundary.
    pub scalar_types: &'static str,
}

macro_rules! row {
    ($family:literal, $algorithm:literal, $maturity:ident, $assumptions:literal,
     $stopping:literal, $breakdown:literal, $reference:literal, $kind:ident,
     $scalars:literal) => {
        SolverConformance {
            family: $family,
            algorithm: $algorithm,
            maturity: SolverMaturity::$maturity,
            assumptions: $assumptions,
            stopping_rule: $stopping,
            breakdown_policy: $breakdown,
            reference: $reference,
            reference_kind: ReferenceKind::$kind,
            scalar_types: $scalars,
        }
    };
}

/// The complete public algorithm inventory.
///
/// Keep this list synchronized with new public solver entry points.  The
/// conformance tests reject duplicate `(family, algorithm)` keys and empty
/// contract fields, so adding an entry without a usable reference is visible
/// immediately in CI.
pub static SOLVER_CONFORMANCE: &[SolverConformance] = &[
    row!(
        "facade",
        "solve",
        Production,
        "square CSR; dispatch assumption is explicit",
        "configured true residual/backward error",
        "typed error or non-converged result",
        "compatibility_cg_execution + compatibility_scalar_execution",
        Independent,
        "f32/f64/Complex32/Complex64"
    ),
    row!(
        "facade",
        "solve_saddle_point",
        Production,
        "two-block operator and Schur assumptions",
        "true residual and KKT-compatible block contract",
        "typed dimension/dispatch/Schur error",
        "saddle_point_contract.v1",
        Independent,
        "f64"
    ),
    row!(
        "iterative",
        "cg/pcg",
        Production,
        "SPD/HPD operator and compatible preconditioner",
        "true residual with backward error",
        "breakdown and unattainable accuracy are non-success",
        "iterative_family_reference.v1 + cg_accuracy_contract",
        Independent,
        "f32/f64/Complex32/Complex64"
    ),
    row!(
        "iterative",
        "ca-cg/ca-pcg",
        Production,
        "SPD/HPD operator; reduction policy is explicit",
        "true residual after block replacement",
        "typed breakdown or max-iteration result",
        "iterative_extended_reference.v1",
        Independent,
        "f64/Complex64"
    ),
    row!(
        "iterative",
        "gmres",
        Production,
        "general linear operator",
        "restart-aware true residual",
        "typed breakdown/max-iteration result",
        "iterative_independent_reference.v1",
        Independent,
        "f32/f64/Complex32/Complex64"
    ),
    row!(
        "iterative",
        "fgmres",
        Production,
        "general operator and variable preconditioner",
        "true residual with exact candidate gate",
        "typed breakdown/max-iteration result",
        "iterative_family_reference.v1",
        Independent,
        "f64/Complex64"
    ),
    row!(
        "iterative",
        "block-gmres/block-fgmres",
        Production,
        "general block operator and compatible RHS columns",
        "per-RHS true residual",
        "typed per-RHS breakdown/non-convergence",
        "iterative_block_workload_reference.v1",
        Independent,
        "f64"
    ),
    row!(
        "iterative",
        "pipelined-gmres/pipelined-bicgstab",
        Production,
        "operator and reduction synchronization contract",
        "promoted true residual",
        "typed synchronization/breakdown result",
        "iterative_result_contract.v1",
        CrossChecked,
        "f32/f64/Complex32/Complex64"
    ),
    row!(
        "iterative",
        "bicgstab",
        Production,
        "general nonsymmetric operator",
        "true residual and backward error",
        "breakdown/stagnation never claims convergence",
        "iterative_scalar_execution.v1",
        Independent,
        "f32/f64/Complex32/Complex64"
    ),
    row!(
        "iterative",
        "cgs",
        Production,
        "general nonsymmetric operator",
        "true residual",
        "typed breakdown or max-iteration result",
        "iterative_family_reference.v1",
        Independent,
        "f64/Complex64"
    ),
    row!(
        "iterative",
        "idrs/idrs-stab",
        Production,
        "general nonsymmetric operator; shadow-space size valid",
        "true residual",
        "typed breakdown/stagnation result",
        "iterative_family_reference.v1",
        Independent,
        "f64/Complex64"
    ),
    row!(
        "iterative",
        "minres",
        Production,
        "real symmetric or complex Hermitian operator",
        "true residual",
        "rejects non-Hermitian input or reports breakdown",
        "iterative_family_reference.v1",
        Independent,
        "f64/Complex64"
    ),
    row!(
        "iterative",
        "tfqmr",
        Production,
        "general nonsymmetric operator",
        "true residual and attainable accuracy",
        "typed breakdown/max-iteration result",
        "iterative_extended_reference.v1",
        Independent,
        "f64/Complex64"
    ),
    row!(
        "iterative",
        "recycled/deflated/gmres-dr/gcro-dr",
        Experimental,
        "general operator plus valid recycle/deflation subspace",
        "true residual after recycle restart",
        "invalid subspace or breakdown is non-success",
        "gmres_family_contract + deterministic robustness fixture",
        CrossChecked,
        "f64/Complex64"
    ),
    row!(
        "direct",
        "dense-lu",
        Production,
        "square finite matrix",
        "factor solve residual",
        "typed singular/pivot breakdown",
        "direct_scalar_method_reference.v1",
        Independent,
        "f32/f64/Complex32/Complex64"
    ),
    row!(
        "direct",
        "dense-qr",
        Production,
        "full-rank rectangular or square finite matrix",
        "least-squares residual",
        "typed rank/dimension failure",
        "direct_scalar_method_reference.v1",
        Independent,
        "f64/Complex64"
    ),
    row!(
        "direct",
        "dense-cholesky",
        Production,
        "SPD/HPD finite matrix",
        "factor solve residual",
        "typed non-positive-definite failure",
        "direct_contracts.v1",
        CrossChecked,
        "f32/f64/Complex32/Complex64"
    ),
    row!(
        "direct",
        "sparse-lu",
        Production,
        "square structurally valid CSR",
        "factor solve and refinement residual",
        "typed pattern/singularity failure",
        "sparse_lu_pattern_contract + direct_contracts.v1",
        Independent,
        "f32/f64/Complex32/Complex64"
    ),
    row!(
        "direct",
        "sparse-cholesky",
        Production,
        "SPD/HPD structurally valid CSR",
        "factor solve residual",
        "typed ordering/definiteness failure",
        "sparse_cholesky_checked + direct_contracts.v1",
        CrossChecked,
        "f32/f64/Complex32/Complex64"
    ),
    row!(
        "direct",
        "multifrontal-blr/ldlt",
        Production,
        "finite square CSR and valid elimination tree",
        "factor solve residual",
        "typed malformed-front or pivot failure",
        "multifrontal-ldlt-reference.v1",
        Independent,
        "f64/Complex64"
    ),
    row!(
        "direct",
        "iterative-refinement",
        Production,
        "factor and matrix share structural fingerprint",
        "promoted true residual",
        "typed compatibility or numerical failure",
        "refinement_checked + sparse_lu_refinement",
        Independent,
        "f32/f64/Complex32/Complex64"
    ),
    row!(
        "preconditioner",
        "diagonal/jacobi",
        Production,
        "finite nonzero diagonal",
        "preconditioner action contract",
        "typed dimension/non-finite setup error",
        "preconditioner_contracts.v1",
        CrossChecked,
        "f32/f64/Complex32/Complex64"
    ),
    row!(
        "preconditioner",
        "ilu/ilut/hilu",
        Production,
        "structurally valid finite CSR",
        "action residual/diagnostics",
        "typed pivot and generated-value diagnostics",
        "preconditioner_failure_contract.v1 + adjoint_preconditioner_contract",
        Independent,
        "f32/f64/Complex32/Complex64"
    ),
    row!(
        "preconditioner",
        "block-jacobi/schwarz",
        Production,
        "square finite CSR and valid partitions",
        "weighted local action",
        "aggregate local setup diagnostics",
        "canonical_preconditioner_boundaries + adjoint_preconditioner_contract",
        Independent,
        "f64/Complex64"
    ),
    row!(
        "preconditioner",
        "amg/chebyshev",
        Production,
        "SPD/HPD-compatible operator and valid hierarchy",
        "true residual through outer solver",
        "typed hierarchy/setup failure",
        "preconditioner_contracts.v1 + convergence-spectrum.v1",
        CrossChecked,
        "f32/f64/Complex32/Complex64"
    ),
    row!(
        "preconditioner",
        "saddle-point-block",
        Production,
        "declared block and Schur approximation",
        "block residual/KKT contract",
        "typed assumption or Schur failure",
        "saddle_point_contract.v1",
        Independent,
        "f64/Complex64"
    ),
    row!(
        "eigen",
        "lobpcg",
        Production,
        "Hermitian generalized eigenproblem with SPD mass",
        "eigen residual and subspace checks",
        "typed rank/breakdown result",
        "eigen-laplacian-reference.v1",
        Independent,
        "f64/Complex64"
    ),
    row!(
        "eigen",
        "symmetric-eigen",
        Production,
        "finite real symmetric dense matrix",
        "orthogonality and eigen residual",
        "typed malformed-input or convergence failure",
        "eigen-laplacian-reference.v1",
        Independent,
        "f64"
    ),
    row!(
        "eigen",
        "arnoldi/shift-invert",
        Production,
        "finite operator and valid factor/shift",
        "eigen residual",
        "typed factor, rank, or convergence failure",
        "eigen-generalized-reference.v1 + eigen-complex-generalized-reference.v1",
        Independent,
        "f64/Complex64"
    ),
    row!(
        "nonlinear",
        "newton/anderson",
        Production,
        "finite callback values and compatible Jacobian/action",
        "nonlinear residual and attainable accuracy",
        "typed callback/breakdown/max-iteration result",
        "extended-result-contract.v1 + checked_contract",
        CrossChecked,
        "f64/Complex64"
    ),
    row!(
        "optimization",
        "bfgs",
        Production,
        "finite objective/gradient",
        "objective plus gradient/KKT where constrained",
        "typed line-search or max-iteration result",
        "optimization-differential-reference.v1",
        Independent,
        "f64"
    ),
    row!(
        "optimization",
        "lbfgs",
        Production,
        "finite objective/gradient",
        "objective plus gradient/KKT where constrained",
        "typed line-search or max-iteration result",
        "optimization-differential-reference.v1",
        Independent,
        "f64"
    ),
    row!(
        "optimization",
        "lbfgsb",
        Production,
        "finite objective/gradient and valid bound constraints",
        "objective, projected gradient, bound feasibility, and KKT gate",
        "infeasible/stagnated candidate is non-success",
        "optimization-kkt-reference.v1 + optimization-kkt-matrix.v1",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "slsqp",
        Production,
        "finite constraints, Jacobian, and valid bounds",
        "feasibility, stationarity, complementarity, and KKT gate",
        "infeasible/stagnated candidate is non-success",
        "optimization-kkt-matrix.v1 + optimizer_kkt_contract",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "sqp",
        Production,
        "finite constraints, Jacobian, and valid bounds",
        "feasibility, stationarity, complementarity, and KKT gate",
        "infeasible/stagnated candidate is non-success",
        "optimization-kkt-matrix.v1 + optimizer_kkt_contract",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "sqp_advanced",
        Production,
        "finite constraints, Jacobian, Hessian, and valid bounds",
        "feasibility, stationarity, complementarity, and KKT gate",
        "infeasible/stagnated candidate is non-success",
        "optimization-kkt-matrix.v1 + optimizer_kkt_contract",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "ipm",
        Production,
        "finite constraints, Jacobian, and valid bounds",
        "feasibility, stationarity, complementarity, and KKT gate",
        "infeasible/stagnated candidate is non-success",
        "optimization-ipm-state.v1 + optimizer_kkt_contract",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "penalty_method",
        Production,
        "finite objective, constraints, and valid bounds",
        "feasibility, stationarity, complementarity, and KKT gate",
        "infeasible/stagnated candidate is non-success",
        "optimization-kkt-gate-family.v1 + optimizer_kkt_contract",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "augmented_lagrangian",
        Production,
        "finite objective, constraints, and valid bounds",
        "feasibility, stationarity, complementarity, and KKT gate",
        "infeasible/stagnated candidate is non-success",
        "optimization-kkt-gate-family.v1 + optimizer_kkt_contract",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "trust_region_constrained",
        Production,
        "finite objective, constraints, Jacobian, and valid bounds",
        "feasibility, stationarity, complementarity, and KKT gate",
        "infeasible/stagnated candidate is non-success",
        "optimization-kkt-gate-family.v1 + optimizer_kkt_contract",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "mma",
        Production,
        "finite objective, constraints, gradients, and valid bounds",
        "feasibility, stationarity, complementarity, and KKT gate",
        "infeasible/stagnated candidate is non-success",
        "optimization-kkt-gate-family.v1 + optimizer_kkt_contract",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "gmma",
        Production,
        "finite objective, constraints, gradients, and valid bounds",
        "feasibility, stationarity, complementarity, and KKT gate",
        "infeasible/stagnated candidate is non-success",
        "optimization-gcmma.v1 + optimizer_kkt_contract",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "epsilon_constraint",
        Experimental,
        "finite objective, constraints, and deterministic Pareto configuration",
        "feasibility and objective termination for each Pareto point",
        "invalid epsilon schedule or unattainable point is non-success",
        "optimization-contracts.v1",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "solve_qp",
        Experimental,
        "finite convex or explicitly indefinite quadratic program with valid bounds",
        "primal/dual feasibility and active-set KKT residual",
        "infeasible or rank-deficient active set is a typed non-success",
        "optimization-kkt-matrix.v1 + optimizer_kkt_contract",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "nonlinear_cg",
        Experimental,
        "finite objective/gradient and compatible line-search configuration",
        "objective/gradient termination with attainable-accuracy reporting",
        "typed line-search, breakdown, or max-iteration result",
        "optimization-contracts.v1 + result_contract_extended",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "newton_cg",
        Experimental,
        "finite objective/gradient and Hessian-vector callback",
        "objective/gradient termination with attainable-accuracy reporting",
        "typed callback, line-search, or max-iteration result",
        "optimization-contracts.v1 + result_contract_extended",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "quasi_newton",
        Experimental,
        "finite objective/gradient and compatible update policy",
        "objective/gradient termination with attainable-accuracy reporting",
        "typed update, line-search, or max-iteration result",
        "optimization-contracts.v1 + result_contract_extended",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "bobyqa",
        Experimental,
        "finite objective and deterministic bound/model configuration",
        "objective termination with trust-region radius metadata",
        "typed evaluation, geometry, or max-iteration result",
        "optimization-contracts.v1 + result_contract_extended",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "nelder_mead",
        Experimental,
        "finite objective and deterministic simplex configuration",
        "objective and simplex-size termination policy",
        "typed evaluation or max-iteration result",
        "optimization-contracts.v1 + result_contract_extended",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "powell",
        Experimental,
        "finite objective and deterministic direction-set configuration",
        "objective and direction-set termination policy",
        "typed evaluation or max-iteration result",
        "optimization-contracts.v1 + result_contract_extended",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "cma_es",
        Research,
        "finite objective and deterministic population/seed configuration",
        "objective and population-step termination policy",
        "typed evaluation, covariance, or max-generation result",
        "optimization-contracts.v1 + result_contract_extended",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "differential_evolution",
        Research,
        "finite objective and deterministic population/seed configuration",
        "objective and generation termination policy",
        "typed evaluation or max-generation result",
        "optimization-contracts.v1 + result_contract_extended",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "simulated_annealing",
        Research,
        "finite objective and deterministic temperature/seed configuration",
        "objective and temperature termination policy",
        "typed evaluation or cooling-schedule result",
        "optimization-contracts.v1 + result_contract_extended",
        CrossChecked,
        "f64"
    ),
    row!(
        "optimization",
        "random_search",
        Research,
        "finite objective and deterministic sampling/seed configuration",
        "objective and sample-budget termination policy",
        "typed evaluation or max-sample result",
        "optimization-contracts.v1 + result_contract_extended",
        CrossChecked,
        "f64"
    ),
    row!(
        "time-integration",
        "newmark/generalized-alpha",
        Production,
        "finite mass/damping/stiffness and compatible state",
        "trajectory residual and energy metadata",
        "typed dimension/configuration/callback failure",
        "time-integration-contracts.v1 + adjoint_contract",
        Independent,
        "f64/Complex64"
    ),
    row!(
        "time-integration",
        "revolve",
        Production,
        "valid checkpoint budget and replay schedule",
        "deterministic replay contract",
        "typed checkpoint/schedule failure",
        "time-integration-contracts.v1",
        CrossChecked,
        "f64"
    ),
];

/// Public optimization entry points represented as distinct conformance rows.
///
/// Keeping this list explicit prevents a broad family alias from hiding an
/// unreviewed algorithm when the optimization crate grows another export.
pub const OPTIMIZATION_CONFORMANCE_ALGORITHMS: &[&str] = &[
    "bfgs",
    "lbfgs",
    "lbfgsb",
    "slsqp",
    "sqp",
    "sqp_advanced",
    "ipm",
    "penalty_method",
    "augmented_lagrangian",
    "trust_region_constrained",
    "mma",
    "gmma",
    "epsilon_constraint",
    "solve_qp",
    "nonlinear_cg",
    "newton_cg",
    "quasi_newton",
    "bobyqa",
    "nelder_mead",
    "powell",
    "cma_es",
    "differential_evolution",
    "simulated_annealing",
    "random_search",
];

/// Return the complete conformance inventory.
pub const fn all() -> &'static [SolverConformance] {
    SOLVER_CONFORMANCE
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn every_public_record_has_a_complete_contract() {
        assert!(SOLVER_CONFORMANCE.len() >= 35);
        let mut keys = HashSet::new();
        for row in SOLVER_CONFORMANCE {
            assert!(!row.family.is_empty());
            assert!(!row.algorithm.is_empty());
            assert!(!row.assumptions.is_empty());
            assert!(!row.stopping_rule.is_empty());
            assert!(!row.breakdown_policy.is_empty());
            assert!(!row.reference.is_empty());
            assert!(!row.scalar_types.is_empty());
            assert!(keys.insert((row.family, row.algorithm)));
        }
    }

    #[test]
    fn experimental_algorithms_are_explicitly_opt_in() {
        for row in SOLVER_CONFORMANCE {
            if matches!(
                row.maturity,
                SolverMaturity::Experimental | SolverMaturity::Research
            ) {
                assert_ne!(row.family, "facade");
            }
        }
    }

    #[test]
    fn production_rows_have_executable_reference_evidence() {
        for row in SOLVER_CONFORMANCE {
            if row.maturity == SolverMaturity::Production {
                assert_ne!(
                    row.reference_kind,
                    ReferenceKind::DocumentedOnly,
                    "production row lacks executable evidence: {}/{}",
                    row.family,
                    row.algorithm
                );
            }
        }
    }

    #[test]
    fn optimization_inventory_is_algorithm_granular() {
        let rows: HashSet<_> = SOLVER_CONFORMANCE
            .iter()
            .filter(|row| row.family == "optimization")
            .map(|row| row.algorithm)
            .collect();
        let expected: HashSet<_> = OPTIMIZATION_CONFORMANCE_ALGORITHMS
            .iter()
            .copied()
            .collect();
        assert_eq!(rows, expected);
        assert!(
            OPTIMIZATION_CONFORMANCE_ALGORITHMS
                .iter()
                .all(|algorithm| !algorithm.contains('/')),
            "optimization conformance rows must not combine public algorithms"
        );
    }
}
