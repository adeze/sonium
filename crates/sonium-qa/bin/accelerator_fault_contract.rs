//! Schema/runner fault-injection contract.
//!
//! These probes exercise the release policy without requiring a GPU: fallback
//! provenance, skipped-as-passed records, stale artifact checksums, and
//! non-finite metrics must all be rejected before a result can be promoted.

use sonium_qa::{
    QaContextReuse, QaExecutionCounts, QaMemoryEvidence, QaResult, QaRuntimeCounters,
    QaRuntimeEvidence, host_hardware_provenance, write_qa_artifact,
};
use std::time::Instant;

const CASE_ID: &str = "qa.accelerator-fault-contract.v1";

fn run() -> Result<QaResult, String> {
    let fallback = QaResult::passed("fault.fallback")
        .requested_backend("cuda")
        .actual_backend("cpu");
    if fallback
        .release_eligible_for(Some("cuda"), false, false, false, false)
        .is_ok()
    {
        return Err("fallback-labelled-as-CUDA record was accepted".into());
    }

    if QaResult::skipped("fault.skipped", "injected unavailable backend")
        .release_eligible()
        .is_ok()
    {
        return Err("skipped-as-passed record was accepted".into());
    }

    let mut stale = QaResult::passed("fault.stale").artifact_with_provenance(
        "stale.json",
        "fault-contract",
        "test",
        "v1",
        "sha256:old",
    );
    stale
        .evidence
        .artifact_checksums
        .insert("stale.json".into(), "sha256:new".into());
    if stale.validate().is_ok() {
        return Err("stale artifact provenance was accepted".into());
    }

    if QaResult::passed("fault.nonfinite")
        .metric("error", f64::NAN)
        .validate()
        .is_ok()
    {
        return Err("non-finite metric was accepted".into());
    }

    let mut hardware = host_hardware_provenance();
    hardware
        .cpu_model
        .get_or_insert_with(|| "qa-contract-host".into());
    hardware
        .compiler
        .get_or_insert_with(|| "qa-contract-compiler".into());
    let artifact = serde_json::to_vec_pretty(&serde_json::json!({
        "case_id": CASE_ID,
        "faults_rejected": 4,
        "contract": "fallback,skip,stale-checksum,non-finite",
    }))
    .map_err(|error| format!("serialize fault contract artifact: {error}"))?;
    let (path, checksum) = write_qa_artifact("accelerator_fault_contract.json", &artifact)?;
    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .backend_passed("cpu")
        .reference("typed release policy fault-injection probes")
        .reference_checksum("analytic/deterministic-fixture:accelerator-fault-contract-v1")
        .dimension("schema and runner policy")
        .formulation("fallback/skip/stale-checksum/non-finite fail-closed probes")
        .tolerance("all four injected invalid records must be rejected")
        .metric("faults_rejected", 4.0)
        .minimum("faults_rejected", 4.0)
        .artifact_with_provenance(
            path,
            "crates/sonium-qa/bin/accelerator_fault_contract.rs",
            "deterministic fault-injection contract report",
            "v1",
            checksum,
        )
        .runtime_evidence(QaRuntimeEvidence {
            context_reuse: QaContextReuse::NotApplicable,
            counts: QaExecutionCounts {
                executed: 1,
                passed: 1,
                ..QaExecutionCounts::default()
            },
            counters: QaRuntimeCounters::default(),
            memory: QaMemoryEvidence::default(),
            hardware,
            ..QaRuntimeEvidence::default()
        })
        .elapsed_ms(Instant::now().elapsed().as_millis() as u64))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("fault contract QA result must serialize")
    );
    if result.status != sonium_qa::QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
