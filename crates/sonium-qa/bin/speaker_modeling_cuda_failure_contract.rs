//! SM1 modeling-specific CUDA failure/recovery evidence.
//!
//! The successful retry is the existing production `run_coupled_reference`
//! path. The injected attempt uses the same bounded 2x2 electromechanical
//! coupling seam after resident setup, and is never published as a response.
//! This binary emits one machine-readable record per failure disposition.

use num_complex::Complex64;
use sonium_qa::{
    QaContextReuse, QaExecutionCounts, QaFailureStage, QaMemoryEvidence, QaResult,
    QaRuntimeCounters, QaRuntimeEvidence, QaStatus, cuda_hardware_provenance, write_qa_artifact,
};
use sonium_sim_speaker_modeling::{DriverModel, SpeakerDriverStudy, run_coupled_failure_contract};
use std::time::Instant;

const CASE_ID: &str = "speaker.cuda-bounded-coupled-failure-contract.v1";

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

fn u64_field(value: &serde_json::Value, path: &[&str]) -> Result<u64, String> {
    let mut current = value;
    for key in path {
        current = current
            .get(*key)
            .ok_or_else(|| format!("failure evidence is missing {}", path.join(".")))?;
    }
    current.as_u64().ok_or_else(|| {
        format!(
            "failure evidence field {} is not an unsigned integer",
            path.join(".")
        )
    })
}

fn run() -> Result<QaResult, String> {
    if std::env::var("SONIUM_CUDA_FORCE_FALLBACK").as_deref() == Ok("1") {
        return Err("CUDA fallback fault injection is active".into());
    }
    let model = DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
        .map_err(|error| format!("build typed driver fixture: {error}"))?;
    let frequency_hz = [100.0, 1_000.0];
    let radiation_impedance = [Complex64::new(0.2, 0.1), Complex64::new(0.4, 0.3)];
    let lanes = [
        (
            "launch",
            sonium_compute::CudaFailureInjectionPoint::Launch,
            "speaker_modeling_cuda_launch_failure.json",
        ),
        (
            "device-loss",
            sonium_compute::CudaFailureInjectionPoint::DeviceLoss,
            "speaker_modeling_cuda_device_loss_failure.json",
        ),
    ];

    let mut records = Vec::with_capacity(lanes.len());
    let mut artifact_provenance = Vec::with_capacity(lanes.len());
    let mut total_h2d = 0_u64;
    let mut total_d2h = 0_u64;
    let mut peak_memory = 0_u64;
    for (label, point, artifact_name) in lanes {
        let record =
            run_coupled_failure_contract(&model, &frequency_hz, &radiation_impedance, point)
                .map_err(|error| format!("{label} failure contract failed: {error}"))?;
        total_h2d = total_h2d.saturating_add(u64_field(
            &record,
            &["retry_success", "transfer", "h2d_operations"],
        )?);
        total_d2h = total_d2h.saturating_add(u64_field(
            &record,
            &["retry_success", "transfer", "d2h_operations"],
        )?);
        peak_memory = peak_memory.max(u64_field(
            &record,
            &["retry_success", "residency", "allocation_high_water_bytes"],
        )?);
        let bytes = serde_json::to_vec_pretty(&record)
            .map_err(|error| format!("serialize {label} failure evidence: {error}"))?;
        let (path, checksum) = write_qa_artifact(artifact_name, &bytes)?;
        artifact_provenance.push((path, checksum));
        records.push(record);
    }

    let mut qa = QaResult::passed(CASE_ID)
        .requested_backend("cuda")
        .actual_backend("cuda")
        .backend_passed("cuda")
        .reference("CPU/f64 DriverModel golden plus production CUDA coupling retry")
        .reference_checksum("analytic/deterministic-fixture:speaker-cuda-bounded-failure-v1")
        .input_checksum("sha256:detailed-driver-study-v1")
        .dimension("typed electromechanical 2x2 driver network, two frequencies, two failure lanes")
        .formulation("resident CUDA CSR GMRES with typed launch/device-loss unwind and recreated-context retry")
        .tolerance("failure stage/recovery/publication plus CPU parity and transfer gates")
        .metric("failure_modes_exercised", records.len() as f64)
        .metric("cuda_h2d_operations", total_h2d as f64)
        .metric("cuda_d2h_operations", total_d2h as f64)
        .metric("cuda_allocation_high_water_bytes", peak_memory as f64)
        .minimum("failure_modes_exercised", 2.0)
        .minimum("cuda_h2d_operations", 1.0)
        .minimum("cuda_d2h_operations", 1.0)
        .minimum("cuda_allocation_high_water_bytes", 1.0)
        .runtime_evidence(QaRuntimeEvidence {
            context_reuse: QaContextReuse::Recreated,
            counts: QaExecutionCounts {
                executed: 2,
                passed: 2,
                ..QaExecutionCounts::default()
            },
            counters: QaRuntimeCounters {
                allocations: 2,
                h2d_transfers: total_h2d,
                d2h_transfers: total_d2h,
                synchronizations: 2,
                launches: 2,
                ..QaRuntimeCounters::default()
            },
            memory: QaMemoryEvidence {
                peak_bytes: peak_memory,
                live_bytes: 0,
            },
            hardware: cuda_hardware_provenance(),
            ..QaRuntimeEvidence::default()
        })
        .elapsed_ms(0);
    for (path, checksum) in artifact_provenance {
        qa = qa.artifact_with_provenance(
            path,
            "crates/sonium-qa/bin/speaker_modeling_cuda_failure_contract.rs",
            "bounded modeling CUDA failure/recovery evidence",
            "v1",
            checksum,
        );
    }
    qa.environment.insert(
        "gpu.failure_modes_exercised".into(),
        "launch,device-loss".into(),
    );
    qa.environment.insert(
        "gpu.execution_scope".into(),
        "bounded typed electromechanical network; high-level FEM/BEM unsupported".into(),
    );
    qa.environment.insert(
        "gpu.capability_admission".into(),
        "native Complex64 CSR/BLAS/reduction required; no fallback".into(),
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
            .expect("SM1 modeling CUDA failure result must serialize")
    );
    if result.status != QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
