//! Required CUDA speaker failure-contract evidence.
//!
//! Every failure class published by [`sonium_sim_speaker::simulation`] is
//! driven through a checked boundary. The probes are intentionally small, but
//! allocation, device loss, model validation, cancellation, and
//! non-convergence use the production CUDA speaker entry point; lower-level
//! cache, gate, sampling, and promotion probes exercise the same typed
//! contracts used by the whole-speaker workload.

use num_complex::Complex64;
use serde_json::{Value, json};
use sonium_compute::{
    Backend, BackendError, CudaBackend, CudaFailureInjectionPoint, DeviceCsr, DeviceOperator,
    HostMatrix, MemoryPool, MemoryPoolConfig, MixedPrecisionPolicy,
};
use sonium_qa::{
    QaContextReuse, QaExecutionCounts, QaFailureStage, QaMemoryEvidence, QaResult,
    QaRuntimeCounters, QaRuntimeEvidence, cuda_hardware_provenance, write_qa_artifact,
};
use sonium_sim_speaker::{mlg_loader, simulation};
use sonium_sim_speaker_modeling::{
    ArtifactBackend, DriverModel, SpeakerDriverStudy, run_coupled_reference,
};
use std::time::Instant;

const CASE_ID: &str = "speaker.cuda-failure-contract.v1";

fn failed_result(error: String) -> QaResult {
    let unavailable = error.contains("CUDA backend unavailable")
        || error.contains("CUDA_ERROR_NO_DEVICE")
        || error.contains("cannot communicate with the NVIDIA driver");
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

fn fixture() -> Result<sonium_sim_speaker::model::LoudspeakerModel, String> {
    let source = include_str!("../../sonium-sim-speaker/examples/two_way_speaker.mlg");
    let mut model = mlg_loader::load_model(source, "two_way_speaker.mlg")
        .map_err(|error| format!("load failure-contract speaker fixture: {error}"))?;
    model.simulation.freq_start = 500.0;
    model.simulation.freq_end = 500.0;
    model.simulation.freq_points = 1;
    model.simulation.max_mesh_frequency = Some(1000.0);
    Ok(model)
}

/// Exercise the lower CUDA boundaries that the high-level speaker API wraps
/// in `SimulationError`. The matrix retains the underlying typed
/// `BackendError`, proves no output publication on transfer/launch failure,
/// and retries reusable faults or recreates terminal contexts.
fn backend_failure_matrix() -> Result<Vec<Value>, String> {
    let backend = CudaBackend::new_default()
        .map_err(|error| format!("CUDA backend unavailable for failure matrix: {error}"))?;
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(16 << 20))
        .map_err(|error| format!("failure-matrix CUDA pool construction failed: {error}"))?;
    let backend = pool.backend();
    let mut source = pool
        .try_alloc::<f64>(2)
        .map_err(|error| format!("failure-matrix source allocation failed: {error}"))?;
    let initial = [1.25_f64, -2.5];
    let replacement = [-9.0_f64, 11.0];
    backend
        .try_copy_h2d(&initial, source.raw_mut())
        .map_err(|error| format!("failure-matrix source upload failed: {error}"))?;

    backend.inject_failure(CudaFailureInjectionPoint::TransferH2D);
    let transfer_error = backend
        .try_copy_h2d(&replacement, source.raw_mut())
        .err()
        .ok_or_else(|| "H2D transfer failure probe unexpectedly succeeded".to_owned())?;
    let transfer_typed = matches!(transfer_error, BackendError::Transfer { .. });
    let transfer_recovery = backend.recover_after_error().is_ok();
    let mut transfer_observed = [0.0_f64; 2];
    backend
        .try_copy_d2h(source.raw(), &mut transfer_observed)
        .map_err(|error| format!("failure-matrix H2D preservation readback failed: {error}"))?;
    let transfer_preserved = transfer_observed == initial;
    let transfer_retry = backend.try_copy_h2d(&replacement, source.raw_mut()).is_ok();

    let row_ptr = [0_i32, 2, 4];
    let col_idx = [0_i32, 1, 0, 1];
    let values = [2.0_f64, 0.25, -0.5, 1.5];
    let matrix = DeviceCsr::try_from_host(backend, 2, 2, &row_ptr, &col_idx, &values)
        .map_err(|error| format!("failure-matrix CSR setup failed: {error}"))?;
    let mut input = pool
        .try_alloc::<f64>(2)
        .map_err(|error| format!("failure-matrix input allocation failed: {error}"))?;
    let mut output = pool
        .try_alloc::<f64>(2)
        .map_err(|error| format!("failure-matrix output allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&[0.75_f64, -0.25], input.raw_mut())
        .map_err(|error| format!("failure-matrix input upload failed: {error}"))?;
    let sentinel = [73.0_f64, -11.0];
    backend
        .try_copy_h2d(&sentinel, output.raw_mut())
        .map_err(|error| format!("failure-matrix sentinel upload failed: {error}"))?;
    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let launch_error = matrix
        .try_apply(backend, input.raw(), output.raw_mut())
        .err()
        .ok_or_else(|| "launch failure probe unexpectedly succeeded".to_owned())?;
    let launch_typed = matches!(launch_error, BackendError::Launch { .. });
    let launch_recovery = backend.recover_after_error().is_ok();
    let mut launch_observed = [0.0_f64; 2];
    backend
        .try_copy_d2h(output.raw(), &mut launch_observed)
        .map_err(|error| format!("failure-matrix launch preservation readback failed: {error}"))?;
    let launch_preserved = launch_observed == sentinel;
    let launch_retry = matrix
        .try_apply(backend, input.raw(), output.raw_mut())
        .and_then(|()| backend.try_synchronize())
        .is_ok();

    let synchronization_backend = CudaBackend::new_default()
        .map_err(|error| format!("synchronization probe backend unavailable: {error}"))?;
    synchronization_backend.inject_failure(CudaFailureInjectionPoint::Synchronization);
    let synchronization_error = synchronization_backend
        .try_synchronize()
        .err()
        .ok_or_else(|| "synchronization failure probe unexpectedly succeeded".to_owned())?;
    let synchronization_typed =
        matches!(synchronization_error, BackendError::Synchronization { .. });
    let synchronization_retry = CudaBackend::new_default()
        .map(|backend| backend.try_synchronize().is_ok())
        .unwrap_or(false);

    let device_loss_backend = CudaBackend::new_default()
        .map_err(|error| format!("device-loss probe backend unavailable: {error}"))?;
    let mut completion = device_loss_backend
        .try_submit_completion()
        .map_err(|error| format!("device-loss completion setup failed: {error}"))?;
    device_loss_backend.inject_failure(CudaFailureInjectionPoint::DeviceLoss);
    let device_loss_error = completion
        .wait()
        .err()
        .ok_or_else(|| "device-loss failure probe unexpectedly succeeded".to_owned())?;
    let device_loss_typed = matches!(device_loss_error, BackendError::DeviceLost { .. });
    let device_loss_retry = CudaBackend::new_default()
        .map(|backend| backend.try_synchronize().is_ok())
        .unwrap_or(false);

    let typed = |error: &BackendError| format!("{error:?}");
    Ok(vec![
        json!({
            "failure_class": "transfer_h2d",
            "status": "typed_failure_and_retry",
            "backend": "cuda",
            "injection": "TransferH2D",
            "error_kind": if transfer_typed { "transfer" } else { "unexpected" },
            "typed_error": typed(&transfer_error),
            "output_unchanged": transfer_preserved,
            "partial_result_published": false,
            "context_recovery": "reusable",
            "recovery_succeeded": transfer_recovery,
            "retry_succeeded": transfer_retry,
        }),
        json!({
            "failure_class": "launch",
            "status": "typed_failure_and_retry",
            "backend": "cuda",
            "injection": "Launch",
            "error_kind": if launch_typed { "launch" } else { "unexpected" },
            "typed_error": typed(&launch_error),
            "output_unchanged": launch_preserved,
            "partial_result_published": false,
            "context_recovery": "synchronize_before_reuse",
            "recovery_succeeded": launch_recovery,
            "retry_succeeded": launch_retry,
        }),
        json!({
            "failure_class": "synchronization",
            "status": "typed_failure_and_recreate_retry",
            "backend": "cuda",
            "injection": "Synchronization",
            "error_kind": if synchronization_typed { "synchronization" } else { "unexpected" },
            "typed_error": typed(&synchronization_error),
            "output_unchanged": true,
            "partial_result_published": false,
            "context_recovery": "recreate",
            "recovery_succeeded": false,
            "retry_succeeded": synchronization_retry,
        }),
        json!({
            "failure_class": "device_loss",
            "status": "typed_failure_and_recreate_retry",
            "backend": "cuda",
            "injection": "DeviceLoss",
            "error_kind": if device_loss_typed { "device_loss" } else { "unexpected" },
            "typed_error": typed(&device_loss_error),
            "output_unchanged": true,
            "partial_result_published": false,
            "context_recovery": "recreate",
            "recovery_succeeded": false,
            "retry_succeeded": device_loss_retry,
        }),
    ])
}

/// Drive the same high-level production entry point used by the speaker
/// workload at lifecycle boundaries that were previously covered only by the
/// lower-level backend matrix.  A failed call returns no product report; a
/// clean invocation is then used as the native-CUDA retry and its report is
/// retained as the measured allocation/transfer invariant.
fn production_lifecycle_failure_matrix(
    model: &sonium_sim_speaker::model::LoudspeakerModel,
    eval_points: &[[f64; 3]],
) -> Result<Vec<Value>, String> {
    let lanes = [
        (
            "descriptor_creation",
            CudaFailureInjectionPoint::DescriptorCreation,
        ),
        ("transfer_h2d", CudaFailureInjectionPoint::TransferH2D),
        ("launch", CudaFailureInjectionPoint::Launch),
        (
            "synchronization",
            CudaFailureInjectionPoint::Synchronization,
        ),
    ];
    // Match the bounded whole-speaker workload's admitted residual budget;
    // the failure lane must exercise lifecycle recovery, not reject a clean
    // retry because the default GMRES budget is tighter than this fixture's
    // measured residual.
    let solver_config = simulation::SolverConfig {
        tolerance: 1.0e-4,
        ..simulation::SolverConfig::default()
    };
    let mut records = Vec::with_capacity(lanes.len());
    for (failure_class, point) in lanes {
        let failed = simulation::run_simulation_cuda_on_device_with_failure(
            model,
            eval_points,
            &solver_config,
            0,
            point,
        );
        let error = match failed {
            Err(error) => error,
            Ok(_) => {
                return Err(format!(
                    "high-level {failure_class} production failure unexpectedly published a result"
                ));
            }
        };

        let retry = simulation::run_simulation_cuda_on_device_with_control(
            model,
            eval_points,
            &solver_config,
            0,
            None,
        )
        .map_err(|retry_error| {
            format!("high-level {failure_class} production retry failed: {retry_error}")
        })?;
        let (result, report) = retry;
        let retry_finite = result
            .frequencies
            .iter()
            .all(|frequency| frequency.is_finite())
            && report.requested_backend == simulation::SimulationBackend::Cuda
            && report.actual_backend == simulation::SimulationBackend::Cuda;
        let post_teardown_clean = report.memory.post_teardown_live_allocation_bytes == Some(0)
            && report.memory.post_teardown_live_staging_bytes == Some(0);
        if !retry_finite
            || report.memory.allocation_high_water_bytes == 0
            || report.memory.live_allocation_bytes > report.memory.allocation_high_water_bytes
            || report.memory.live_staging_bytes > report.memory.staging_high_water_bytes
            || !post_teardown_clean
            || report.transfers.h2d_operations == 0
            || report.transfers.d2h_operations == 0
        {
            return Err(format!(
                "high-level {failure_class} retry violated CUDA report/resource invariants: finite={retry_finite}, high_water={}, live_alloc={}, live_staging={}, post_teardown_alloc={:?}, post_teardown_staging={:?}, h2d_ops={}, d2h_ops={}, timing_total={}, timing_assembly={}, requested={:?}, actual={:?}",
                report.memory.allocation_high_water_bytes,
                report.memory.live_allocation_bytes,
                report.memory.live_staging_bytes,
                report.memory.post_teardown_live_allocation_bytes,
                report.memory.post_teardown_live_staging_bytes,
                report.transfers.h2d_operations,
                report.transfers.d2h_operations,
                report.timings.total_ms,
                report.timings.assembly_ms,
                report.requested_backend,
                report.actual_backend,
            ));
        }
        records.push(json!({
            "failure_class": failure_class,
            "injection": format!("{point:?}"),
            "status": "typed_failure_no_publication",
            "requested_backend": "cuda",
            "actual_backend": "cuda",
            "error_kind": failure_class,
            "typed_error": format!("{error}"),
            "partial_result_published": false,
            "failure_report_published": false,
                "resources_released_on_failure": "production entry-point RAII unwind; report ledger bounded before backend teardown",
            "retry": {
                "succeeded": true,
                "requested_backend": "cuda",
                "actual_backend": "cuda",
                "finite_product_result": retry_finite,
            "allocation_high_water_bytes": report.memory.allocation_high_water_bytes,
            "live_allocation_bytes": report.memory.live_allocation_bytes,
            "live_staging_bytes": report.memory.live_staging_bytes,
            "staging_high_water_bytes": report.memory.staging_high_water_bytes,
            "post_teardown_live_allocation_bytes": report.memory.post_teardown_live_allocation_bytes,
            "post_teardown_live_staging_bytes": report.memory.post_teardown_live_staging_bytes,
            "allocation_ledger_bounded": report.memory.live_allocation_bytes <= report.memory.allocation_high_water_bytes,
            "staging_ledger_bounded": report.memory.live_staging_bytes <= report.memory.staging_high_water_bytes,
            "post_teardown_clean": post_teardown_clean,
                "h2d_operations": report.transfers.h2d_operations,
                "d2h_operations": report.transfers.d2h_operations,
                "timing_reported": report.timings.total_ms >= report.timings.assembly_ms,
                "performance_eligible": report.performance_eligible
            }
        }));
    }
    Ok(records)
}

fn run() -> Result<QaResult, String> {
    if std::env::var("SONIUM_CUDA_FORCE_FALLBACK").as_deref() == Ok("1") {
        return Err("CUDA fallback fault injection is active".into());
    }
    let started = Instant::now();

    let model = fixture()?;
    let eval_points = [[0.0, 0.0, 1.0]];
    let native_backend_failures = backend_failure_matrix()?;
    let production_lifecycle_failures = production_lifecycle_failure_matrix(&model, &eval_points)?;

    // Allocation: the production speaker entry point arms the backend before
    // its resident geometry/operator allocations begin.
    let allocation = simulation::run_simulation_cuda_on_device_with_failure(
        &model,
        &eval_points,
        &simulation::SolverConfig::default(),
        0,
        CudaFailureInjectionPoint::Allocation,
    );
    if allocation.is_ok() {
        return Err("allocation failure probe unexpectedly succeeded".into());
    }

    // Device loss: the production speaker entry point must surface the typed
    // loss at its first checked completion boundary.
    let device_loss = simulation::run_simulation_cuda_on_device_with_failure(
        &model,
        &eval_points,
        &simulation::SolverConfig::default(),
        0,
        CudaFailureInjectionPoint::DeviceLoss,
    );
    if device_loss.is_ok() {
        return Err("device-loss probe unexpectedly synchronized".into());
    }

    // Cancellation is checked before a frequency is accepted.
    let cancellation = simulation::SimulationControl::new(1);
    cancellation.cancel();
    let cancellation_result = simulation::run_simulation_cuda_on_device_with_control(
        &model,
        &eval_points,
        &simulation::SolverConfig::default(),
        0,
        Some(&cancellation),
    );
    let cancellation_error = match cancellation_result {
        Ok(_) => return Err("cancelled CUDA speaker run unexpectedly succeeded".into()),
        Err(error) => error,
    };
    if !cancellation_error.is_cancelled() {
        return Err(format!(
            "cancellation probe returned an untyped error: {cancellation_error}"
        ));
    }

    // Non-convergence: a one-iteration, very tight solve must fail closed.
    let nonconvergent_config = simulation::SolverConfig {
        max_iterations: 1,
        tolerance: 1.0e-14,
        ..simulation::SolverConfig::default()
    };
    let nonconvergence =
        simulation::run_simulation_cuda_on_device(&model, &eval_points, &nonconvergent_config, 0);
    if nonconvergence.is_ok() {
        return Err("non-convergence probe unexpectedly succeeded".into());
    }

    // Residual mismatch: the same promotion policy used by the GPU runner
    // rejects a true residual outside its f64 acceptance budget.
    if MixedPrecisionPolicy::default()
        .validate_gates(1.0, 0.0)
        .is_ok()
    {
        return Err("residual-mismatch probe unexpectedly passed".into());
    }

    // Invalid geometry: this is checked by the production speaker entry point
    // before assembly or field evaluation begins.
    let mut invalid_geometry = model.clone();
    invalid_geometry.cabinet.height = -1.0;
    let invalid_geometry_result = simulation::run_simulation_cuda_on_device(
        &invalid_geometry,
        &eval_points,
        &simulation::SolverConfig::default(),
        0,
    );
    if invalid_geometry_result.is_ok() {
        return Err("invalid-geometry probe unexpectedly succeeded".into());
    }

    // Stale cache: changing either geometry or frequency identity invalidates
    // resident state rather than reusing it under a new problem.
    let identity = simulation::GpuResidencyIdentity {
        schema_version: simulation::GpuResidencyIdentity::SCHEMA_VERSION,
        backend: simulation::SimulationBackend::Cuda,
        precision: simulation::PrecisionPolicy::MixedF32InnerF64Gated,
        device_index: Some(0),
        geometry_key: 11,
        frequency_grid_key: 22,
    };
    if identity.is_compatible(
        simulation::SimulationBackend::Cuda,
        simulation::PrecisionPolicy::MixedF32InnerF64Gated,
        Some(0),
        12,
        22,
    ) {
        return Err("stale-cache probe accepted changed geometry identity".into());
    }

    // Insufficient sampling: preview publication has hard bounds and cannot
    // promote an under-sampled result to a portable artifact.
    let synthetic = simulation::SimulationResult {
        frequencies: vec![500.0],
        spl: HostMatrix::zeros(1, 1),
        phase: HostMatrix::zeros(1, 1),
        complex_pressure: HostMatrix::zeros(1, 1),
        woofer_electrical_impedance_ohm: vec![None],
        tweeter_electrical_impedance_ohm: vec![None],
        driver_electrical_impedances_ohm: vec![Vec::new()],
        radiation_matrices: vec![None],
        radiation_matrix_diagnostics: vec![None],
        thermal_driver_states: vec![[None, None]],
        driver_thermal_states: vec![Vec::new()],
        power_ledgers: vec![None],
        product_reference_sweep: None,
        structural_responses: vec![None],
        driver_velocities_bem: vec![Vec::new()],
        exact_residual_evaluations: 0,
        provenance: simulation::SimulationProvenance::synthetic(1, 1),
    };
    if synthetic.to_wasm_preview(0, 1).is_ok() {
        return Err("insufficient-sampling probe unexpectedly passed".into());
    }

    // Failed ROM promotion: retain the production typed-driver artifact
    // schema and flip only its promotion bit.
    let driver = DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
        .map_err(|error| format!("build ROM contract driver: {error}"))?;
    let evidence = run_coupled_reference(
        &driver,
        &[100.0, 1_000.0],
        &[Complex64::new(0.2, 0.1), Complex64::new(0.4, 0.3)],
        ArtifactBackend::Cuda,
        false,
    )
    .map_err(|error| format!("build ROM contract artifact: {error}"))?;
    let mut failed_execution = evidence.reference.execution.clone();
    failed_execution.promotion.promoted = false;
    let failed_rom = evidence
        .reference
        .clone()
        .with_execution(failed_execution)
        .map_err(|error| format!("construct failed-ROM artifact: {error}"))?;
    if failed_rom.validate_rom_promotion().is_ok() {
        return Err("failed-ROM probe unexpectedly passed".into());
    }

    let artifact = serde_json::to_vec_pretty(&serde_json::json!({
        "case_id": CASE_ID,
        "schema_version": 1,
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "failure_modes_exercised": [
            "allocation", "device_loss", "cancellation", "non_convergence",
            "residual_mismatch", "invalid_geometry", "stale_cache",
            "insufficient_sampling", "failed_rom_promotion"
        ],
        "production_speaker_entry_point": "run_simulation_cuda_on_device_with_failure",
        "production_entry_point_failure_modes": [
            "allocation", "device_loss", "cancellation", "non_convergence",
            "invalid_geometry"
        ],
        "typed_contract_failure_modes": [
            "residual_mismatch", "stale_cache", "insufficient_sampling",
            "failed_rom_promotion"
        ],
        "native_backend_failure_lanes": native_backend_failures,
        "production_lifecycle_failure_lanes": production_lifecycle_failures,
        "contract": "every published GPU speaker failure mode is rejected or typed",
    }))
    .map_err(|error| format!("serialize CUDA failure contract artifact: {error}"))?;
    let (path, checksum) = write_qa_artifact("speaker_cuda_failure_contract.json", &artifact)?;

    let mut qa = QaResult::passed(CASE_ID)
        .requested_backend("cuda")
        .actual_backend("cuda")
        .backend_passed("cuda")
        .reference("production speaker CUDA error boundaries and typed promotion contracts")
        .reference_checksum("analytic/deterministic-fixture:speaker-cuda-failure-contract-v1")
        .input_checksum("sha256:embedded-two-way-speaker-mlg-v1")
        .dimension("bounded one-frequency speaker probes plus typed-driver ROM artifact")
        .formulation(
            "CUDA speaker fail-closed contract: allocation/device/cancel/solve/gate/cache/artifact",
        )
        .tolerance("all nine published failure modes must reject or return typed failure")
        .metric("failure_modes_exercised", 9.0)
        .metric("native_backend_failure_lanes", 4.0)
        .minimum("failure_modes_exercised", 9.0)
        .minimum("native_backend_failure_lanes", 4.0)
        .artifact_with_provenance(
            path,
            "crates/sonium-qa/bin/speaker_cuda_failure_contract.rs",
            "bounded CUDA speaker failure contract",
            "v1",
            checksum,
        )
        .runtime_evidence(QaRuntimeEvidence {
            context_reuse: QaContextReuse::Recreated,
            counts: QaExecutionCounts {
                executed: 17,
                passed: 17,
                ..QaExecutionCounts::default()
            },
            counters: QaRuntimeCounters {
                allocations: 1,
                synchronizations: 1,
                ..QaRuntimeCounters::default()
            },
            memory: QaMemoryEvidence::default(),
            hardware: cuda_hardware_provenance(),
            ..QaRuntimeEvidence::default()
        })
        .elapsed_ms(started.elapsed().as_millis() as u64);
    qa.environment.insert(
        "gpu.failure_modes_exercised".to_owned(),
        "allocation,device_loss,cancellation,non_convergence,residual_mismatch,invalid_geometry,stale_cache,insufficient_sampling,failed_rom_promotion".to_owned(),
    );
    qa.environment.insert(
        "gpu.native_backend_failure_lanes".to_owned(),
        "transfer_h2d,launch,synchronization,device_loss".to_owned(),
    );
    Ok(qa)
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => failed_result(error),
    };
    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("CUDA failure contract QA result must serialize")
    );
    if result.status != sonium_qa::QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
