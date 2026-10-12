//! Structured results for explicitly reference-only analytical fixtures.

use sonium_qa::QaResult;

pub(super) fn emit(case_id: &str, passed: bool) {
    let result = if passed {
        QaResult::passed(case_id)
    } else {
        QaResult::failed(case_id, "analytical reference assertion failed")
    }
    .metric("numerical_solve_executed", 0.0)
    .maximum("numerical_solve_executed", 0.0)
    .formulation(
        "reference-only: closed-form/helper assertions; no production PDE assembly or solve",
    )
    .dimension("analytical fixture; no discrete field")
    .reference("closed-form identities and limiting/scaling checks in this binary")
    .reference_checksum(format!("analytic-fixture:{case_id}"));
    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("FAILED: cannot serialize analytical reference: {error}");
            std::process::exit(1);
        }
    }
}
