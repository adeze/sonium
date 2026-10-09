//! Fail-closed statistical regression policy contract for compute evidence.
//!
//! The timing producer is platform-specific, but the release gate itself must
//! be exercised on every CPU.  This case loads the checked-in matrix fixture,
//! proves an identical replay is accepted, proves a separated slowdown is
//! rejected, and proves missing cells and hardware drift are rejected before
//! any timing comparison can be reported.

use serde_json::{Value, from_str};
use sonium_bench::compute_regression::{
    EvidenceSnapshot, RegressionPolicy, RegressionStatus, compare_evidence,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.performance-regression-contract.v1";
const BASELINE: &str = include_str!("../schema/compute-performance-baseline-v1.json");

#[derive(Debug, Clone, Copy)]
struct Metrics {
    baseline_cells: usize,
    accepted_cells: usize,
    regression_cases: usize,
    rejection_cases: usize,
    protocol_checks: usize,
}

fn baseline() -> Result<EvidenceSnapshot, String> {
    from_str(BASELINE).map_err(|error| format!("baseline fixture is invalid JSON: {error}"))
}

fn validate_protocol_metadata() -> Result<usize, String> {
    let raw: Value = from_str(BASELINE)
        .map_err(|error| format!("baseline protocol JSON is invalid: {error}"))?;
    let protocol = raw
        .get("protocol")
        .ok_or_else(|| "baseline is missing protocol metadata".to_owned())?;
    let warmup = protocol
        .get("warmup_samples")
        .and_then(Value::as_u64)
        .ok_or_else(|| "baseline protocol warmup_samples is missing".to_owned())?;
    let measured = protocol
        .get("measured_samples")
        .and_then(Value::as_u64)
        .ok_or_else(|| "baseline protocol measured_samples is missing".to_owned())?;
    let confidence = protocol
        .get("confidence_interval")
        .and_then(Value::as_str)
        .ok_or_else(|| "baseline protocol confidence interval is missing".to_owned())?;
    let thread_policy = protocol
        .get("thread_policy")
        .and_then(Value::as_str)
        .ok_or_else(|| "baseline protocol thread policy is missing".to_owned())?;
    if warmup != 5 || measured != 15 {
        return Err(format!(
            "unexpected baseline sample protocol: warmup={warmup}, measured={measured}"
        ));
    }
    if !confidence.contains("2nd-98th percentile") {
        return Err("baseline confidence interval is not the declared 2nd-98th percentile".into());
    }
    if !thread_policy.contains("RAYON_NUM_THREADS=1")
        || !thread_policy.contains("VECLIB_MAXIMUM_THREADS=1")
    {
        return Err(
            "baseline thread policy does not cap Rayon and Accelerate to one thread".into(),
        );
    }
    Ok(4)
}

fn run() -> Result<Metrics, String> {
    let protocol_checks = validate_protocol_metadata()?;
    let fixture = baseline()?;
    if fixture.rows.is_empty() {
        return Err("baseline fixture must contain at least one timing cell".into());
    }

    let accepted = compare_evidence(&fixture, &fixture, RegressionPolicy::default())
        .map_err(|error| format!("identical fixture was rejected: {error}"))?;
    if !accepted.passed() || accepted.comparisons.len() != fixture.rows.len() {
        return Err(format!(
            "identical fixture did not produce an accepted full matrix: passed={}, cells={}",
            accepted.passed(),
            accepted.comparisons.len()
        ));
    }

    let mut regressed_fixture = fixture.clone();
    let first = regressed_fixture
        .rows
        .first_mut()
        .ok_or_else(|| "baseline fixture unexpectedly empty".to_owned())?;
    first.median_ns = 200.0;
    first.ci95_low_ns = 190.0;
    first.ci95_high_ns = 210.0;
    let regression = compare_evidence(&fixture, &regressed_fixture, RegressionPolicy::default())
        .map_err(|error| format!("synthetic slowdown comparison failed: {error}"))?;
    if regression.passed()
        || regression.regression_count != 1
        || !regression
            .comparisons
            .iter()
            .any(|comparison| comparison.status == RegressionStatus::Regressed)
    {
        return Err(format!(
            "separated slowdown was not classified as one regression: {:?}",
            regression
        ));
    }

    let mut missing_cell_fixture = fixture.clone();
    missing_cell_fixture.rows.pop();
    let missing_cell_rejected =
        compare_evidence(&fixture, &missing_cell_fixture, RegressionPolicy::default()).is_err();

    let mut hardware_drift_fixture = fixture.clone();
    hardware_drift_fixture.hardware.rayon_threads = "2".into();
    let hardware_drift_rejected = compare_evidence(
        &fixture,
        &hardware_drift_fixture,
        RegressionPolicy::default(),
    )
    .is_err();
    let rejection_cases = usize::from(missing_cell_rejected) + usize::from(hardware_drift_rejected);
    if rejection_cases != 2 {
        return Err(format!(
            "fail-closed admission rejected {rejection_cases}/2 malformed comparisons"
        ));
    }

    Ok(Metrics {
        baseline_cells: fixture.rows.len(),
        accepted_cells: accepted.comparisons.len(),
        regression_cases: regression.regression_count,
        rejection_cases,
        protocol_checks,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("baseline_cells", metrics.baseline_cells as f64)
                .metric("accepted_cells", metrics.accepted_cells as f64)
                .metric("regression_cases", metrics.regression_cases as f64)
                .metric("rejection_cases", metrics.rejection_cases as f64)
                .metric("protocol_checks", metrics.protocol_checks as f64)
                .minimum("baseline_cells", 3.0)
                .minimum("accepted_cells", 3.0)
                .minimum("regression_cases", 1.0)
                .minimum("rejection_cases", 2.0)
                .minimum("protocol_checks", 4.0)
                .input_checksum("checked-in:compute-performance-baseline-v1")
                .reference("sonium-bench fail-closed confidence-interval comparison")
                .reference_checksum(
                    "analytic/deterministic-fixture:compute-performance-contract-v1",
                )
                .dimension("CPU BLAS-1 AXPY empty/unit/tail timing matrix; three cells")
                .formulation("same-hardware median slowdown and empirical CI policy")
                .tolerance("10% maximum median slowdown; malformed matrix rejected")
                .seed("compute-performance-contract:policy-fixture:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("compute performance contract result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("compute performance failure result must serialize")
            );
            eprintln!("COMPUTE PERFORMANCE CONTRACT FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_fixture_replays_and_rejects_drift() {
        let metrics = run().expect("performance contract fixture");
        assert_eq!(metrics.baseline_cells, 3);
        assert_eq!(metrics.accepted_cells, 3);
        assert_eq!(metrics.regression_cases, 1);
        assert_eq!(metrics.rejection_cases, 2);
        assert_eq!(metrics.protocol_checks, 4);
    }
}
