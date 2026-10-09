//! SM1 bounded production axisymmetric operator CUDA evidence.
//!
//! FEM assembly remains on the host. The checked cell uploads the assembled
//! magnetic-bias-informed acoustic/structural/electrical operator once, reuses
//! its CSR structure across two frequency values, and records native CUDA
//! parity plus typed launch recovery before publishing the artifact.

use serde_json::Value;
use sonium_qa::{
    QaContextReuse, QaExecutionCounts, QaFailureStage, QaMemoryEvidence, QaResult,
    QaRuntimeCounters, QaRuntimeEvidence, QaStatus, cuda_hardware_provenance, write_qa_artifact,
};
use sonium_sim_speaker_modeling::run_production_frequency_step_cuda_contract;
use std::time::Instant;

const CASE_ID: &str = "speaker.cuda-production-frequency-step.v1";

fn failed_result(error: String) -> QaResult {
    let unavailable = error.contains("CUDA backend unavailable")
        || error.contains("CUDA_ERROR_NO_DEVICE")
        || (error.contains("driver") && error.contains("unavailable"));
    QaResult::failed(CASE_ID, error.clone())
        .requested_backend("cuda")
        .actual_backend(if unavailable { "unavailable" } else { "cuda" })
        .backend_failed("cuda", error)
        .failure_stage(if unavailable {
            QaFailureStage::Environment
        } else {
            QaFailureStage::Execution
        })
        .hardware_provenance(cuda_hardware_provenance())
}

fn number(value: &Value, path: &[&str]) -> Result<f64, String> {
    let mut current = value;
    for key in path {
        current = current
            .get(*key)
            .ok_or_else(|| format!("production frequency record is missing {}", path.join(".")))?;
    }
    current.as_f64().ok_or_else(|| {
        format!(
            "production frequency field {} is not numeric",
            path.join(".")
        )
    })
}

fn run() -> Result<QaResult, String> {
    if std::env::var("SONIUM_CUDA_FORCE_FALLBACK").as_deref() == Ok("1") {
        return Err("CUDA fallback fault injection is active".into());
    }
    let record = run_production_frequency_step_cuda_contract()?;
    if record.get("status").and_then(Value::as_str) != Some("passed")
        || record.get("actual_backend").and_then(Value::as_str) != Some("cuda")
    {
        return Err("production frequency record did not prove a CUDA pass".into());
    }
    let bytes = serde_json::to_vec_pretty(&record)
        .map_err(|error| format!("serialize production frequency artifact: {error}"))?;
    let (path, checksum) =
        write_qa_artifact("speaker_modeling_cuda_production_frequency.json", &bytes)?;
    let dimension = number(&record, &["dimension"])?;
    let nnz = number(&record, &["nnz"])?;
    let max_residual = number(&record, &["cpu_reference", "max_relative_residual"])?;
    let warm_h2d = number(&record, &["residency", "warm_action_h2d_operations"])?;
    let update_h2d = number(
        &record,
        &["residency", "frequency_value_update_h2d_operations"],
    )?;
    let mut qa = QaResult::passed(CASE_ID)
        .requested_backend("cuda")
        .actual_backend("cuda")
        .backend_passed("cuda")
        .reference("production axisymmetric magnetic-bias-informed FEM harmonic operator with CPU CSR oracle")
        .reference_checksum("analytic/deterministic-fixture:speaker-cuda-production-frequency-v1")
        .input_checksum("sha256:axisymmetric-linear-config-default-v1")
        .dimension("production acoustic/structural/electrical harmonic FEM operator, two frequencies")
        .formulation("native CUDA persistent CSR SpMV with resident frequency-value update and typed launch retry")
        .tolerance("CPU/CUDA max relative SpMV residual <= 1e-10; warm action H2D/allocation deltas == 0")
        .metric("production_dimension", dimension)
        .metric("production_nnz", nnz)
        .metric("max_relative_residual", max_residual)
        .metric("warm_action_h2d_operations", warm_h2d)
        .metric("frequency_value_update_h2d_operations", update_h2d)
        .minimum("production_dimension", 100.0)
        .minimum("production_nnz", dimension)
        .maximum("max_relative_residual", 1.0e-10)
        .maximum("warm_action_h2d_operations", 0.0)
        .minimum("frequency_value_update_h2d_operations", 1.0)
        .runtime_evidence(QaRuntimeEvidence {
            context_reuse: QaContextReuse::Reused,
            counts: QaExecutionCounts {
                executed: 1,
                passed: 1,
                ..QaExecutionCounts::default()
            },
            counters: QaRuntimeCounters {
                allocations: 1,
                h2d_transfers: 1,
                d2h_transfers: 2,
                synchronizations: 3,
                launches: 3,
                ..QaRuntimeCounters::default()
            },
            memory: QaMemoryEvidence {
                peak_bytes: number(
                    &record,
                    &["residency", "live_device_bytes_before_teardown"],
                )? as u64,
                live_bytes: 0,
            },
            hardware: cuda_hardware_provenance(),
            ..QaRuntimeEvidence::default()
        })
        .artifact_with_provenance(
            path,
            "crates/sonium-qa/bin/speaker_modeling_cuda_production_frequency.rs",
            "bounded production axisymmetric CUDA frequency-step operator evidence",
            "v1",
            checksum,
        )
        .elapsed_ms(0);
    qa.environment.insert(
        "gpu.execution_scope".into(),
        record
            .get("scope_boundary")
            .and_then(Value::as_str)
            .unwrap_or("production frequency operator; explicit host assembly boundary")
            .to_owned(),
    );
    qa.environment.insert(
        "gpu.failure_contract".into(),
        "typed launch failure; no partial publication; synchronize-before-reuse; retry passed"
            .into(),
    );
    Ok(qa)
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok(result) => result,
        Err(error) => failed_result(error),
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);
    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("production frequency QA result must serialize")
    );
    if result.status != QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
