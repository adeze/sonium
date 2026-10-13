//! Admission and reporting for native parity gates using external references.

// Rust guideline compliant 2026-02-21

use crate::{QaFailureStage, QaResult, QaStatus};

/// Build the structured result for an unavailable external reference engine.
///
/// Optional references yield a skip. Required references yield an admission
/// failure. Neither result claims native execution or release eligibility.
pub fn missing_engine_result(case_id: &str, required: bool) -> QaResult {
    let reason = "external reference unavailable: set OPENCOURANT_EXE (or legacy RADISS_EXE)";
    let result = if required {
        QaResult::failed(case_id, reason).failure_stage(QaFailureStage::Environment)
    } else {
        QaResult::skipped(case_id, reason)
    };
    result
        .requested_backend("opencourant")
        .actual_backend("not-executed")
        .metric("external_reference_executed", 0.0)
        .metric("native_parity_executed", 0.0)
}

/// Report a missing engine using the configured required-reference policy.
///
/// `SONIUM_QA_REQUIRE_OPENCOURANT=1` makes absence fail with exit code one.
/// Otherwise absence is an optional structured skip with exit code zero.
/// This function always exits the process after publishing the result.
pub fn report_missing_engine(case_id: &str) -> ! {
    let required = std::env::var("SONIUM_QA_REQUIRE_OPENCOURANT").as_deref() == Ok("1");
    publish_and_exit(missing_engine_result(case_id, required))
}

/// Record external reference health without claiming native numerical parity.
///
/// Until the matching native problem has been solved and compared, even a
/// healthy external engine cannot pass the original native parity gate.
/// `reference_passed` records the existing external analytical probe checks.
pub fn reference_health_result(case_id: &str, engine: &str, reference_passed: bool) -> QaResult {
    let reason = if reference_passed {
        "external reference probes passed; matching native parity has not been executed"
    } else {
        "external reference probes failed; matching native parity has not been executed"
    };
    QaResult::failed(case_id, reason)
        .requested_backend("cpu+opencourant")
        .actual_backend("opencourant")
        .metric("external_reference_executed", 1.0)
        .metric(
            "external_reference_health_passed",
            if reference_passed { 1.0 } else { 0.0 },
        )
        .minimum("external_reference_health_passed", 1.0)
        .metric("native_parity_executed", 0.0)
        .minimum("native_parity_executed", 1.0)
        .reference(engine)
        .formulation(
            "external reference health only; native parity and full-phenomenon proof are pending",
        )
}

/// Record an external runner error without claiming a completed reference solve.
pub fn execution_failure_result(case_id: &str, error: &str) -> QaResult {
    QaResult::failed(case_id, error)
        .requested_backend("opencourant")
        .actual_backend("not-completed")
        .metric("external_reference_attempted", 1.0)
        .metric("native_parity_executed", 0.0)
}

fn exit_code(result: &QaResult) -> i32 {
    match result.status {
        QaStatus::Passed | QaStatus::Skipped => 0,
        QaStatus::Failed => 1,
    }
}

/// Publish a validated QA result and exit with its corresponding status.
///
/// A claimed pass also needs release eligibility. Optional skips print
/// `SKIPPED` and never print `SUCCESS`; failures exit with code one.
/// Invalid records fail publication and also exit with code one.
pub fn publish_and_exit(mut result: QaResult) -> ! {
    if result.status == QaStatus::Passed
        && let Err(error) = result.release_eligible()
    {
        result.status = QaStatus::Failed;
        result.message = Some(error.to_string());
    }
    match result.to_json_line() {
        Ok(line) => println!("QA_RESULT: {line}"),
        Err(error) => {
            eprintln!("FAILED: external parity result publication: {error}");
            std::process::exit(1);
        }
    }
    println!(
        "{}",
        match result.status {
            QaStatus::Passed => "SUCCESS",
            QaStatus::Failed => "FAILED",
            QaStatus::Skipped => "SKIPPED",
        }
    );
    std::process::exit(exit_code(&result));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_missing_reference_is_a_serializable_ineligible_skip() {
        let result = missing_engine_result("fem.opencourant.cantilever.v1", false);
        assert_eq!(result.status, QaStatus::Skipped);
        assert_eq!(exit_code(&result), 0);
        assert!(result.to_json_line().is_ok());
        assert!(result.release_eligible().is_err());
    }

    #[test]
    fn required_missing_reference_fails_admission() {
        let result = missing_engine_result("fem.opencourant.cantilever.v1", true);
        assert_eq!(result.status, QaStatus::Failed);
        assert_eq!(exit_code(&result), 1);
        assert!(result.to_json_line().is_ok());
        assert!(result.release_eligible().is_err());
    }

    #[test]
    fn external_reference_success_cannot_certify_native_parity() {
        let result =
            reference_health_result("fem.opencourant.cantilever.v1", "test-reference", true);
        assert_eq!(result.status, QaStatus::Failed);
        assert!(result.to_json_line().is_ok());
        assert!(result.release_eligible().is_err());
        // Even changing only the status cannot bypass the native-execution budget.
        let mut forged = result;
        forged.status = QaStatus::Passed;
        forged.message = None;
        assert!(forged.release_eligible().is_err());
    }
}
