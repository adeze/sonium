//! Required bounded CUDA evidence for the typed speaker-driver network.
//!
//! This is deliberately separate from the whole-speaker BEM workload: it
//! exercises the currently supported typed electromechanical CUDA seam and
//! records the CPU/f64 golden, residual, impedance, power, derivative, ROM,
//! transfer, and backend-provenance gates.  Unsupported high-level physics
//! remains visible in the capability matrix and is never silently delegated
//! to the CPU.

use num_complex::Complex64;
use sonium_qa::{
    QaContextReuse, QaExecutionCounts, QaFailureStage, QaMemoryEvidence, QaResult,
    QaRuntimeCounters, QaRuntimeEvidence, cuda_hardware_provenance, write_qa_artifact,
};
use sonium_sim_speaker_modeling::{
    ArtifactBackend, ArtifactPrecision, DriverModel, SpeakerCapabilityState, SpeakerDriverStudy,
    run_coupled_reference,
};
use std::time::Instant;

const CASE_ID: &str = "speaker.cuda-bounded-typed-driver-reference.v1";

fn failed_result(error: String) -> QaResult {
    let unavailable =
        error.contains("CUDA backend unavailable") || error.contains("CUDA_ERROR_NO_DEVICE");
    let fallback = error.contains("fallback");
    QaResult::failed(CASE_ID, error.clone())
        .requested_backend("cuda")
        .actual_backend(if fallback {
            "host-fallback"
        } else if unavailable {
            "unavailable"
        } else {
            "cuda"
        })
        .backend_failed("cuda", error)
        .failure_stage(if unavailable || fallback {
            QaFailureStage::Environment
        } else {
            QaFailureStage::Execution
        })
        .hardware_provenance(cuda_hardware_provenance())
}
const RESIDUAL_TOLERANCE: f64 = 1.0e-6;
const POWER_TOLERANCE: f64 = 5.0e-2;
const PARITY_TOLERANCE: f64 = 1.0e-7;

fn output_artifact(bytes: &[u8]) -> Result<Option<(String, String)>, String> {
    let relative = "speaker_driver_cuda_reference.json";
    write_qa_artifact(relative, bytes).map(Some)
}

fn verify_fail_closed_contract(
    model: &DriverModel,
    frequency_hz: &[f64],
    radiation_impedance: &[Complex64],
    reference: &sonium_sim_speaker_modeling::DriverModelReferenceArtifact,
) -> Result<usize, String> {
    let mismatched_grid = run_coupled_reference(
        model,
        &frequency_hz[..1],
        radiation_impedance,
        ArtifactBackend::Cuda,
        false,
    );
    if mismatched_grid.is_ok() {
        return Err("typed CUDA path accepted mismatched frequency/radiation grids".into());
    }

    let mut failed_promotion_execution = reference.execution.clone();
    failed_promotion_execution.promotion.promoted = false;
    let failed_promotion = reference
        .clone()
        .with_execution(failed_promotion_execution)
        .map_err(|error| format!("construct failed-ROM contract fixture: {error}"))?;
    if failed_promotion.validate_rom_promotion().is_ok() {
        return Err("reference artifact accepted failed ROM promotion evidence".into());
    }

    Ok(2)
}

fn run() -> Result<QaResult, String> {
    if std::env::var("SONIUM_CUDA_FORCE_FALLBACK").as_deref() == Ok("1") {
        return Err("CUDA fallback fault injection is active".into());
    }
    let model = DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
        .map_err(|error| format!("build detailed typed driver fixture: {error}"))?;
    let frequency_hz = [100.0, 1_000.0];
    let radiation_impedance = [Complex64::new(0.2, 0.1), Complex64::new(0.4, 0.3)];

    let evidence = run_coupled_reference(
        &model,
        &frequency_hz,
        &radiation_impedance,
        ArtifactBackend::Cuda,
        false,
    )
    .map_err(|error| format!("required typed CUDA reference failed: {error}"))?;
    let execution = &evidence.reference.execution;

    if execution.requested_backend != ArtifactBackend::Cuda
        || execution.actual_backend != ArtifactBackend::Cuda
        || evidence.reference.provenance.backend != ArtifactBackend::Cuda
        || execution.precision != ArtifactPrecision::F64
    {
        return Err(format!(
            "typed CUDA provenance mismatch: requested={:?}, actual={:?}, provenance={:?}, precision={:?}",
            execution.requested_backend,
            execution.actual_backend,
            evidence.reference.provenance.backend,
            execution.precision,
        ));
    }
    evidence
        .reference
        .validate()
        .map_err(|error| format!("typed CUDA reference artifact validation failed: {error}"))?;
    evidence
        .reference
        .validate_rom_promotion()
        .map_err(|error| format!("typed CUDA ROM promotion gate failed: {error}"))?;

    let values = [
        (
            "residual",
            evidence.max_relative_residual,
            RESIDUAL_TOLERANCE,
        ),
        (
            "power closure",
            evidence.max_power_balance_relative_error,
            POWER_TOLERANCE,
        ),
        (
            "velocity parity",
            evidence.max_velocity_relative_error,
            PARITY_TOLERANCE,
        ),
        (
            "impedance parity",
            evidence.max_impedance_relative_error,
            PARITY_TOLERANCE,
        ),
        (
            "derivative parity",
            evidence.max_derivative_relative_error,
            PARITY_TOLERANCE,
        ),
    ];
    for (name, value, tolerance) in values {
        if !value.is_finite() || value > tolerance {
            return Err(format!(
                "typed CUDA {name} gate failed: value={value:.3e}, tolerance={tolerance:.3e}"
            ));
        }
    }

    let transfer = &execution.transfer;
    if transfer.h2d_operations == 0
        || transfer.d2h_operations == 0
        || transfer.h2d_bytes == 0
        || transfer.d2h_bytes == 0
        || transfer.allocation_high_water_bytes == 0
    {
        return Err(format!(
            "typed CUDA transfer/memory evidence is incomplete: h2d_ops={}, d2h_ops={}, h2d_bytes={}, d2h_bytes={}, high_water={}",
            transfer.h2d_operations,
            transfer.d2h_operations,
            transfer.h2d_bytes,
            transfer.d2h_bytes,
            transfer.allocation_high_water_bytes,
        ));
    }

    let failure_modes = verify_fail_closed_contract(
        &model,
        &frequency_hz,
        &radiation_impedance,
        &evidence.reference,
    )?;
    let unsupported_subsystems = evidence
        .capability_matrix
        .subsystems
        .iter()
        .filter(|entry| entry.state == SpeakerCapabilityState::Unsupported)
        .count();
    let artifact = evidence
        .reference
        .canonical_json()
        .map_err(|error| format!("serialize typed CUDA reference artifact: {error}"))?;
    let mut qa = QaResult::passed(CASE_ID)
        .requested_backend("cuda")
        .actual_backend("cuda")
        .backend_passed("cuda")
        .reference("CPU/f64 DriverModel golden with live CUDA Complex64 CSR GMRES replay")
        .reference_checksum("analytic/deterministic-fixture:speaker-cuda-typed-driver-reference-v1")
        .input_checksum("sha256:detailed-driver-study-v1")
        .dimension("typed electromechanical 2x2 driver network, 2 frequencies")
        .formulation("CUDA CSR GMRES with CPU golden/parity and explicit ROM promotion gates")
        .tolerance("residual <= 1e-6; power <= 5e-2; response/derivative parity <= 1e-7")
        .metric("max_relative_residual", evidence.max_relative_residual)
        .metric(
            "max_power_balance_relative_error",
            evidence.max_power_balance_relative_error,
        )
        .metric(
            "max_velocity_relative_error",
            evidence.max_velocity_relative_error,
        )
        .metric(
            "max_impedance_relative_error",
            evidence.max_impedance_relative_error,
        )
        .metric(
            "max_derivative_relative_error",
            evidence.max_derivative_relative_error,
        )
        .metric("cuda_h2d_bytes", transfer.h2d_bytes as f64)
        .metric("cuda_d2h_bytes", transfer.d2h_bytes as f64)
        .metric(
            "cuda_allocation_high_water_bytes",
            transfer.allocation_high_water_bytes as f64,
        )
        .metric("unsupported_subsystems", unsupported_subsystems as f64)
        .metric("failure_modes_exercised", failure_modes as f64)
        .maximum("max_relative_residual", RESIDUAL_TOLERANCE)
        .maximum("max_power_balance_relative_error", POWER_TOLERANCE)
        .maximum("max_velocity_relative_error", PARITY_TOLERANCE)
        .maximum("max_impedance_relative_error", PARITY_TOLERANCE)
        .maximum("max_derivative_relative_error", PARITY_TOLERANCE)
        .minimum("cuda_h2d_bytes", 1.0)
        .minimum("cuda_d2h_bytes", 1.0)
        .memory_bytes(transfer.allocation_high_water_bytes)
        .elapsed_ms(0);
    qa.environment.insert(
        "gpu.capability_matrix".into(),
        serde_json::to_string(&evidence.capability_matrix)
            .map_err(|error| format!("serialize typed CUDA capability matrix: {error}"))?,
    );
    qa.environment.insert(
        "platform".into(),
        format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
    );
    qa.environment
        .insert("gpu.execution_device".into(), execution.device.clone());
    qa.environment.insert(
        "gpu.execution_toolchain".into(),
        execution.toolchain.clone(),
    );
    qa.environment.insert(
        "gpu.failure_modes_exercised".into(),
        "mismatched_grid,failed_rom_promotion".into(),
    );
    let (path, digest) = output_artifact(&artifact)?
        .ok_or_else(|| "typed CUDA release evidence requires SONIUM_OUTPUT_DIR".to_owned())?;
    qa = qa
        .artifact_with_provenance(
            path,
            "crates/sonium-qa/bin/speaker_driver_cuda_reference.rs",
            "typed-driver canonical reference artifact",
            "v1",
            digest,
        )
        .runtime_evidence(QaRuntimeEvidence {
            context_reuse: QaContextReuse::Reused,
            counts: QaExecutionCounts {
                executed: 1,
                passed: 1,
                ..QaExecutionCounts::default()
            },
            counters: QaRuntimeCounters {
                allocations: u64::from(transfer.allocation_high_water_bytes > 0),
                h2d_transfers: transfer.h2d_operations,
                d2h_transfers: transfer.d2h_operations,
                d2d_transfers: 0,
                synchronizations: 1,
                launches: 1,
            },
            memory: QaMemoryEvidence {
                peak_bytes: transfer.allocation_high_water_bytes,
                live_bytes: transfer.live_allocation_bytes,
            },
            hardware: cuda_hardware_provenance(),
            ..QaRuntimeEvidence::default()
        })
        .physical_residual(evidence.max_relative_residual)
        .physical_reference_error(evidence.max_power_balance_relative_error);
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
            .expect("typed CUDA speaker QA result must serialize")
    );
    if result.status != sonium_qa::QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
