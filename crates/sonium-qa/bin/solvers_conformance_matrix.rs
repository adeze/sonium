//! Release-facing inventory of solver assumptions and conformance evidence.

use sonium_qa::QaResult;
use sonium_solvers::{
    OPTIMIZATION_CONFORMANCE_ALGORITHMS, ReferenceKind, SOLVER_CONFORMANCE, SolverMaturity,
};
use std::collections::HashSet;
use std::time::Instant;

const CASE_ID: &str = "solvers.conformance-matrix.v1";

fn main() {
    let started = Instant::now();
    let mut keys = HashSet::new();
    let mut incomplete = 0.0;
    let mut independent = 0.0;
    let mut cross_checked = 0.0;
    let mut documented_only = 0.0;
    let mut experimental = 0.0;
    let mut optimization_rows = HashSet::new();

    for row in SOLVER_CONFORMANCE {
        if row.family == "optimization" {
            optimization_rows.insert(row.algorithm);
        }
        if !keys.insert((row.family, row.algorithm))
            || row.family.is_empty()
            || row.algorithm.is_empty()
            || row.assumptions.is_empty()
            || row.stopping_rule.is_empty()
            || row.breakdown_policy.is_empty()
            || row.reference.is_empty()
            || row.scalar_types.is_empty()
        {
            incomplete += 1.0;
        }
        match row.reference_kind {
            ReferenceKind::Independent => independent += 1.0,
            ReferenceKind::CrossChecked => cross_checked += 1.0,
            ReferenceKind::DocumentedOnly => documented_only += 1.0,
        }
        if matches!(
            row.maturity,
            SolverMaturity::Experimental | SolverMaturity::Research
        ) {
            experimental += 1.0;
        }
    }

    let count = SOLVER_CONFORMANCE.len() as f64;
    let optimization_algorithm_count = optimization_rows.len() as f64;
    let missing_optimization_rows = OPTIMIZATION_CONFORMANCE_ALGORITHMS
        .iter()
        .filter(|algorithm| !optimization_rows.contains(**algorithm))
        .count() as f64;
    let result = if count >= 56.0
        && incomplete == 0.0
        && documented_only == 0.0
        && optimization_algorithm_count == OPTIMIZATION_CONFORMANCE_ALGORITHMS.len() as f64
        && missing_optimization_rows == 0.0
    {
        QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("algorithm_count", count)
            .metric("independent_reference_count", independent)
            .metric("cross_checked_count", cross_checked)
            .metric("documented_only_count", documented_only)
            .metric("experimental_or_research_count", experimental)
            .metric("optimization_algorithm_count", optimization_algorithm_count)
            .metric("missing_optimization_rows", missing_optimization_rows)
            .metric("incomplete_rows", incomplete)
            .minimum("algorithm_count", 56.0)
            .minimum(
                "optimization_algorithm_count",
                OPTIMIZATION_CONFORMANCE_ALGORITHMS.len() as f64,
            )
            .maximum("missing_optimization_rows", 0.0)
            .maximum("incomplete_rows", 0.0)
            .maximum("documented_only_count", 0.0)
            .elapsed_ms(started.elapsed().as_millis() as u64)
    } else {
        QaResult::failed(
            CASE_ID,
            format!(
                "conformance inventory invalid: count={count}, incomplete={incomplete}, documented_only={documented_only}"
            ),
        )
        .requested_backend("cpu")
        .actual_backend("cpu")
        .elapsed_ms(started.elapsed().as_millis() as u64)
    };

    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => eprintln!("failed to serialize QA result: {error}"),
    }
    if count >= 35.0 && incomplete == 0.0 && documented_only == 0.0 {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
