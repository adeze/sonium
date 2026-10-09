//! PP2 bounded native CUDA adapter evidence.
//!
//! The test drives the propeller-owned panel mesh through the checked
//! sonium-compute Complex64 regular-pair kernel. It is intentionally not a
//! claim that source-model harmonic evaluation, resolved flow, wake, or the
//! full observer pipeline are CUDA-native.

#![cfg(feature = "cuda")]

use serde_json::json;
use sonium_compute::{
    BackendContextDisposition, BackendError, BackendOutputDisposition, CudaBackend,
    CudaFailureInjectionPoint, CudaRuntimeStatsSnapshot,
};
use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    Blade3dConfig, PropellerCudaAdapterError, PropellerCudaPanelResult, PropellerCudaPanelSession,
    PropellerDesign, evaluate_cuda_panel_bem,
};
use std::fs;
use std::path::PathBuf;

const PARITY_TOLERANCE: f64 = 1.0e-10;
const LIFECYCLE_SCHEMA: &str = "sonium.propellers.cuda-native-adapter-failures.v1";
const ACOUSTIC_SWEEP_SCHEMA: &str = "sonium.propellers.cuda-native-acoustic-sweep.v1";

#[test]
fn pp2_adapter_rejects_zero_harmonic_before_device_work() {
    let error = evaluate_cuda_panel_bem(&test_design(), &test_mesh_config(), 0, 0)
        .expect_err("zero harmonic must be rejected before CUDA initialization");
    assert!(matches!(
        error,
        PropellerCudaAdapterError::InvalidWaveNumber
    ));
}

fn test_design() -> PropellerDesign {
    PropellerDesign::conventional(PropellerSource::drone_5inch(12_000.0))
}

fn test_mesh_config() -> Blade3dConfig {
    Blade3dConfig {
        radial_segments: 4,
        chord_segments: 2,
        loop_segments: 2,
        ..Blade3dConfig::default()
    }
}

fn inventory() -> String {
    std::env::var_os("SONIUM_PP2_DEVICE_INVENTORY_FILE")
        .and_then(|path| fs::read_to_string(path).ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unavailable (inventory not supplied)".into())
}

fn runtime_json(runtime: CudaRuntimeStatsSnapshot) -> serde_json::Value {
    let transfer = runtime.transfer;
    json!({
        "allocation_operations": runtime.allocation_operations,
        "allocation_high_water_bytes": runtime.allocation_high_water_bytes,
        "live_allocation_bytes_at_sample": runtime.live_allocation_bytes,
        "staging_allocation_operations": runtime.staging_allocation_operations,
        "staging_high_water_bytes": runtime.staging_high_water_bytes,
        "kernel_launches": runtime.kernel_launches,
        "library_operations": runtime.library_operations,
        "synchronization_calls": runtime.synchronization_calls,
        "transfers": {
            "h2d_operations": transfer.h2d_operations,
            "h2d_bytes": transfer.h2d_bytes,
            "d2h_operations": transfer.d2h_operations,
            "d2h_bytes": transfer.d2h_bytes,
            "d2d_operations": transfer.d2d_operations,
            "d2d_bytes": transfer.d2d_bytes,
        },
        "driver_resources": {
            "live_contexts": runtime.driver_resources.live_contexts,
            "live_streams": runtime.driver_resources.live_streams,
        },
    })
}

fn outputs_equal(
    left: &[sonium_compute::BemRegularPairOutF64],
    right: &[sonium_compute::BemRegularPairOutF64],
) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.g == right.g && left.h == right.h && left.ht == right.ht && left.e == right.e
        })
}

fn context_disposition_name(disposition: BackendContextDisposition) -> &'static str {
    match disposition {
        BackendContextDisposition::Reusable => "reusable",
        BackendContextDisposition::SynchronizeBeforeReuse => "synchronize_before_reuse",
        BackendContextDisposition::MustRecreate => "must_recreate",
    }
}

fn output_disposition_name(disposition: BackendOutputDisposition) -> &'static str {
    match disposition {
        BackendOutputDisposition::Preserved => "preserved",
        BackendOutputDisposition::Unusable => "unusable",
    }
}

fn backend_error_json(error: &PropellerCudaAdapterError) -> serde_json::Value {
    let backend_error = match error {
        PropellerCudaAdapterError::Backend(error) => error,
        other => panic!("driver-shim lane returned non-backend error: {other}"),
    };
    let (kind, operation) = match backend_error {
        BackendError::Allocation { .. } => ("allocation", "cuda_allocation"),
        BackendError::Launch { operation, .. } => ("launch", *operation),
        BackendError::Synchronization { operation, .. } => ("synchronization", *operation),
        BackendError::DeviceLost { operation, .. } => ("device_loss", *operation),
        BackendError::Transfer { operation, .. } => ("transfer", *operation),
        BackendError::OperationFailed { operation, .. } => ("operation", *operation),
        BackendError::Unsupported { operation, .. } => ("unsupported", *operation),
        BackendError::HostFallback { operation, .. } => ("host_fallback", *operation),
        BackendError::InvalidArgument { operation, .. } => ("invalid_argument", *operation),
        BackendError::CommandBuffer { operation, .. } => ("command_buffer", *operation),
    };
    json!({
        "kind": kind,
        "operation": operation,
        "context_disposition": context_disposition_name(backend_error.context_disposition()),
        "output_disposition": output_disposition_name(backend_error.output_disposition()),
        "typed_debug": format!("{backend_error:?}"),
    })
}

fn fresh_context_ledger() -> serde_json::Value {
    let backend = CudaBackend::new(0).expect("fresh CUDA context after PP2 lifecycle lane");
    let runtime = backend.runtime_stats();
    json!({
        "live_allocation_bytes": runtime.live_allocation_bytes,
        "live_staging_bytes": runtime.live_staging_bytes,
        "allocation_high_water_bytes": runtime.allocation_high_water_bytes,
        "driver_resources": {
            "live_contexts": runtime.driver_resources.live_contexts,
            "live_streams": runtime.driver_resources.live_streams,
            "live_modules": runtime.driver_resources.live_modules,
            "live_library_handles": runtime.driver_resources.live_library_handles,
        },
    })
}

fn session_or_skip() -> Option<PropellerCudaPanelSession> {
    match PropellerCudaPanelSession::new(&test_design(), &test_mesh_config(), 1, 0) {
        Ok(session) => Some(session),
        Err(PropellerCudaAdapterError::Device(error)) => {
            if std::env::var_os("SONIUM_PP2_REQUIRE_CUDA").is_some() {
                panic!("CUDA is required for PP2 lifecycle evidence: {error}");
            }
            eprintln!("no CUDA device — skipping PP2 lifecycle evidence: {error}");
            None
        }
        Err(error) => panic!("PP2 lifecycle session creation failed: {error}"),
    }
}

fn write_lifecycle_artifact(path: PathBuf, lanes: Vec<serde_json::Value>) {
    let artifact = json!({
        "schema": LIFECYCLE_SCHEMA,
        "status": "pass",
        "result_kind": "native-production-connected-bem-panel-lifecycle-failures",
        "scope": "native Linux x86_64; CUDA SM120; bounded resident propeller panel-BEM adapter only",
        "backend": {
            "requested": "cuda",
            "actual": "cuda",
            "fallback_reason": null,
            "precision": "f64 / Complex64",
            "fidelity": "panel_geometry_surrogate",
            "device_inventory": inventory(),
        },
        "lanes": lanes,
        "unsupported_or_unexercised": [
            "source-model harmonic evaluation remains host-side",
            "resolved flow/wake/observer/structural CUDA phases are unsupported",
            "cancellation is explicitly unsupported because PropellerCudaPanelSession has no cancellation-token API",
            "Metal and non-panel-BEM propeller cells remain typed unsupported",
        ],
    });
    fs::create_dir_all(path.parent().unwrap_or_else(|| std::path::Path::new(".")))
        .expect("create PP2 CUDA lifecycle output directory");
    fs::write(
        path,
        serde_json::to_vec_pretty(&artifact).expect("serialize PP2 CUDA lifecycle artifact"),
    )
    .expect("write PP2 CUDA lifecycle artifact");
}

fn write_artifact(
    path: PathBuf,
    result: &PropellerCudaPanelResult,
    session_initialization: CudaRuntimeStatsSnapshot,
    warm_first: CudaRuntimeStatsSnapshot,
    warm_second: CudaRuntimeStatsSnapshot,
    warm_resident_bytes: usize,
) {
    let runtime = result.runtime;
    let transfer = runtime.transfer;
    let artifact = json!({
        "schema": result.schema,
        "status": "pass",
        "result_kind": "native-production-connected-bem-panel-pair",
        "scope": "native Linux x86_64; CUDA SM120 cell; Metal/Windows/Arm64 waived or unsupported",
        "backend": {
            "requested": result.requested_backend,
            "actual": result.actual_backend,
            "fallback_reason": null,
            "precision": result.precision,
            "fidelity": result.fidelity,
            "device_inventory": inventory(),
        },
        "source_model": {
            "stage": "host-side production PropellerSource and BladeMesh panelization",
            "harmonic": result.harmonic,
            "wave_number": result.wave_number,
            "harmonic_factor": result.harmonic_factor,
        },
        "geometry": {
            "panel_count": result.panel_count,
            "pair_count": result.pair_count,
            "pair_input": "propeller-owned BladeMesh triangles",
        },
        "parity": {
            "reference": "sonium_compute::bem_cpu::regular_pair_f64_batch",
            "sample_count": result.cuda.len() * 4,
            "max_relative_error": result.max_relative_error,
            "tolerance": PARITY_TOLERANCE,
        },
        "residency": {
            "panel_input": "device buffer during checked BEM launch",
            "pair_output": "device buffer during checked BEM launch",
            "source_model_and_panelization": "host-side; not claimed device-resident",
            "steady_state_policy": "reusable session retains pair geometry and output workspace; parameter-only harmonic updates do not structurally reallocate",
        },
        "warm_replay": {
            "replays": 2,
            "parameter_update": "harmonic 1 -> 2; kernel scalar values updated without geometry re-upload",
            "resident_device_bytes_at_sample": warm_resident_bytes,
            "session_initialization": runtime_json(session_initialization),
            "first_replay": runtime_json(warm_first),
            "second_replay": runtime_json(warm_second),
        },
        "counters": {
            "allocation_operations": runtime.allocation_operations,
            "allocation_high_water_bytes": runtime.allocation_high_water_bytes,
            "live_allocation_bytes_at_sample": runtime.live_allocation_bytes,
            "staging_allocation_operations": runtime.staging_allocation_operations,
            "staging_high_water_bytes": runtime.staging_high_water_bytes,
            "kernel_launches": runtime.kernel_launches,
            "library_operations": runtime.library_operations,
            "synchronization_calls": runtime.synchronization_calls,
            "transfers": {
                "h2d_operations": transfer.h2d_operations,
                "h2d_bytes": transfer.h2d_bytes,
                "d2h_operations": transfer.d2h_operations,
                "d2h_bytes": transfer.d2h_bytes,
                "d2d_operations": transfer.d2d_operations,
                "d2d_bytes": transfer.d2d_bytes,
            },
            "driver_resources": {
                "live_contexts": runtime.driver_resources.live_contexts,
                "live_streams": runtime.driver_resources.live_streams,
                "live_modules": runtime.driver_resources.live_modules,
                "live_dynamic_libraries": runtime.driver_resources.live_dynamic_libraries,
                "live_library_handles": runtime.driver_resources.live_library_handles,
                "live_fft_plans": runtime.driver_resources.live_fft_plans,
                "live_completion_events": runtime.driver_resources.live_completion_events,
                "initialization_baseline_free_bytes": runtime.driver_resources.initialization_baseline_free_bytes,
                "initialization_free_bytes": runtime.driver_resources.initialization_free_bytes,
                "measured_driver_bytes": runtime.driver_resources.measured_driver_bytes,
            },
        },
        "unsupported_or_unexercised": [
            "source-model harmonic evaluation remains host-side",
            "resolved flow/wake/observer/structural CUDA phases are unsupported",
            "Metal adapter is unsupported",
            "cancellation and device-loss probes remain unimplemented for this adapter",
        ],
    });
    fs::create_dir_all(path.parent().unwrap_or_else(|| std::path::Path::new(".")))
        .expect("create PP2 CUDA adapter output directory");
    fs::write(
        path,
        serde_json::to_vec_pretty(&artifact).expect("serialize PP2 CUDA adapter artifact"),
    )
    .expect("write PP2 CUDA adapter artifact");
}

fn complex_json(value: num_complex::Complex64) -> serde_json::Value {
    json!({"re": value.re, "im": value.im})
}

fn write_acoustic_sweep_artifact(
    path: PathBuf,
    result: &sonium_sim_propellers::PropellerCudaAcousticSweepResult,
    initialization: CudaRuntimeStatsSnapshot,
    failure: serde_json::Value,
    malformed: serde_json::Value,
    retry: CudaRuntimeStatsSnapshot,
    resident_bytes: usize,
) {
    let harmonics: Vec<serde_json::Value> = result
        .harmonics
        .iter()
        .map(|harmonic| {
            json!({
                "harmonic": harmonic.harmonic,
                "frequency_hz": harmonic.frequency_hz,
                "source_pressure_pa": harmonic.source_pressure_pa,
                "cpu_panel_pressure": complex_json(harmonic.cpu_panel_pressure),
                "cuda_panel_pressure": complex_json(harmonic.cuda_panel_pressure),
                "max_relative_error": harmonic.max_relative_error,
            })
        })
        .collect();
    let artifact = json!({
        "schema": result.schema,
        "status": "pass",
        "result_kind": "native-production-connected-panel-acoustic-multipoint",
        "scope": "native Linux x86_64; CUDA SM120; bounded production panel-integrated acoustic sweep",
        "backend": {
            "requested": result.requested_backend,
            "actual": result.actual_backend,
            "fallback_reason": null,
            "precision": result.precision,
            "fidelity": "panel_geometry_surrogate",
            "device_inventory": inventory(),
        },
        "production_state": {
            "source": "production PropellerSource harmonic amplitudes",
            "geometry": "production BladeMesh panel triangles",
            "state": "area-weighted Complex64 regular-BEM panel transfer times phase-zero source pressure",
            "phase_convention": "source magnitude used as phase-zero proxy; no observer directivity",
            "panel_count": result.panel_count,
            "pair_count": result.pair_count,
        },
        "multipoint": {
            "batch": "multipoint",
            "harmonics": harmonics,
            "count": result.harmonics.len(),
            "cpu_reference": "independent f64 regular_pair_f64_batch plus host scalar area-weighted reduction",
            "parity_tolerance": PARITY_TOLERANCE,
        },
        "residency": {
            "geometry": "resident CUDA pair buffer across all harmonic points",
            "workspace": "resident CUDA operator scratch/output workspace",
            "source_update": "host-side harmonic parameter update only",
            "publication": "transactional host publication; failed batch restores prior output",
            "resident_bytes_at_sample": resident_bytes,
        },
        "counters": runtime_json(result.runtime),
        "initialization": runtime_json(initialization),
        "failure_and_retry": failure,
        "malformed_input": malformed,
        "retry_counters": runtime_json(retry),
        "unsupported_or_unexercised": [
            "source-model harmonic evaluation and panelization remain host-side",
            "panel-integrated result is a phase-zero transfer proxy, not resolved observer pressure",
            "resolved wake, flow, FW-H observer, structural, transpose, transient, and Metal phases remain unsupported",
        ],
    });
    fs::create_dir_all(path.parent().unwrap_or_else(|| std::path::Path::new(".")))
        .expect("create PP2 acoustic sweep output directory");
    fs::write(
        path,
        serde_json::to_vec_pretty(&artifact).expect("serialize PP2 acoustic sweep artifact"),
    )
    .expect("write PP2 acoustic sweep artifact");
}

#[test]
fn pp2_native_cuda_multipoint_panel_acoustic_state_reuses_residency() {
    assert_eq!(std::env::consts::OS, "linux", "PP2 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "PP2 requires native x86_64"
    );
    let mut session = match session_or_skip() {
        Some(session) => session,
        None => return,
    };
    let initialization = session.runtime_stats();
    let output_before_malformed = session.last_published_output().to_vec();
    let malformed_before = session.runtime_stats();
    let empty_error = session
        .try_solve_multipoint(&test_design(), &[])
        .expect_err("empty multipoint request must be rejected before device work");
    assert!(matches!(
        empty_error,
        PropellerCudaAdapterError::EmptyHarmonicBatch
    ));
    assert!(outputs_equal(
        &output_before_malformed,
        session.last_published_output()
    ));
    let malformed_after_empty = session.runtime_stats();
    assert_eq!(
        malformed_before.allocation_operations,
        malformed_after_empty.allocation_operations
    );
    assert_eq!(
        malformed_before.kernel_launches,
        malformed_after_empty.kernel_launches
    );
    let zero_error = session
        .try_solve_multipoint(&test_design(), &[1, 0, 2])
        .expect_err("zero harmonic in a multipoint request must be rejected");
    assert!(matches!(
        zero_error,
        PropellerCudaAdapterError::InvalidWaveNumber
    ));
    assert!(outputs_equal(
        &output_before_malformed,
        session.last_published_output()
    ));
    let malformed_after_zero = session.runtime_stats();
    assert_eq!(
        malformed_after_empty.kernel_launches,
        malformed_after_zero.kernel_launches
    );

    let result = session
        .try_solve_multipoint(&test_design(), &[1, 2, 3])
        .expect("native CUDA multipoint panel acoustic state");
    assert_eq!(result.schema, ACOUSTIC_SWEEP_SCHEMA);
    assert_eq!(result.requested_backend, "cuda");
    assert_eq!(result.actual_backend, "cuda");
    assert_eq!(result.panel_count, result.pair_count);
    assert_eq!(result.harmonics.len(), 3);
    for harmonic in &result.harmonics {
        assert!(harmonic.frequency_hz.is_finite() && harmonic.frequency_hz > 0.0);
        assert!(harmonic.source_pressure_pa.is_finite() && harmonic.source_pressure_pa > 0.0);
        assert!(harmonic.max_relative_error <= PARITY_TOLERANCE);
        assert!(
            (harmonic.cuda_panel_pressure - harmonic.cpu_panel_pressure).norm()
                <= PARITY_TOLERANCE * harmonic.cpu_panel_pressure.norm().max(1.0)
        );
    }
    assert_eq!(result.runtime.kernel_launches, 3);
    assert_eq!(result.runtime.synchronization_calls, 3);
    assert_eq!(result.runtime.transfer.d2h_operations, 3);
    assert!(result.runtime.transfer.d2h_bytes > 0);
    assert_eq!(result.runtime.transfer.h2d_operations, 0);
    assert_eq!(result.runtime.transfer.h2d_bytes, 0);
    assert!(result.runtime.allocation_operations > 0);
    assert!(result.runtime.live_allocation_bytes > 0);
    let resident_bytes = session.runtime_stats().live_allocation_bytes;
    assert!(resident_bytes > 0);

    // A failure in any point is transactional: a prior published harmonic
    // remains visible, no partial sweep is returned, and a checked launch
    // failure can be recovered before retrying the same multipoint request.
    let output_before_failure = session.last_published_output().to_vec();
    let ledger_before_failure = session.runtime_stats();
    session.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure_error = session
        .try_solve_multipoint(&test_design(), &[1, 2])
        .expect_err("injected multipoint launch failure must reject the sweep");
    assert!(matches!(
        failure_error,
        PropellerCudaAdapterError::Backend(BackendError::Launch { .. })
    ));
    let output_unchanged = outputs_equal(&output_before_failure, session.last_published_output());
    assert!(output_unchanged);
    let ledger_after_failure = session.runtime_stats();
    session
        .recover_after_error()
        .expect("multipoint launch failure should be recoverable");
    let retry = session
        .try_solve_multipoint(&test_design(), &[1, 2])
        .expect("multipoint launch retry");
    assert_eq!(retry.harmonics.len(), 2);
    assert!(
        retry
            .harmonics
            .iter()
            .all(|harmonic| harmonic.max_relative_error <= PARITY_TOLERANCE)
    );
    assert_eq!(retry.runtime.kernel_launches, 2);
    assert_eq!(retry.runtime.synchronization_calls, 2);
    assert_eq!(retry.runtime.transfer.h2d_operations, 0);
    assert_eq!(retry.runtime.allocation_operations, 0);

    let malformed = json!({
        "status": "typed_rejected",
        "device_work": false,
        "output_unchanged": true,
        "lanes": [
            {"input": "empty_harmonics", "error_kind": "empty_harmonic_batch"},
            {"input": "contains_zero_harmonic", "error_kind": "invalid_wave_number"}
        ]
    });
    let failure = json!({
        "status": "typed_rejected",
        "error": backend_error_json(&failure_error),
        "output": {"unchanged": output_unchanged, "published": false},
        "recovery": {
            "strategy": "recover_after_error_then_retry_same_session",
            "context_disposition": "synchronize_before_reuse",
            "retry_status": "passed",
            "retry_points": retry.harmonics.len()
        },
        "ledger": {
            "before": runtime_json(ledger_before_failure),
            "after_failure": runtime_json(ledger_after_failure),
            "after_retry": runtime_json(retry.runtime)
        }
    });
    if let Some(path) = std::env::var_os("SONIUM_PP2_CUDA_ACOUSTIC_SWEEP_ARTIFACT") {
        write_acoustic_sweep_artifact(
            PathBuf::from(path),
            &result,
            initialization,
            failure,
            malformed,
            retry.runtime,
            resident_bytes,
        );
    }
}

#[test]
fn pp2_native_cuda_panel_adapter_matches_cpu_and_publishes_runtime_counters() {
    assert_eq!(std::env::consts::OS, "linux", "PP2 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "PP2 requires native x86_64"
    );

    let result = match evaluate_cuda_panel_bem(&test_design(), &test_mesh_config(), 1, 0) {
        Ok(result) => result,
        Err(PropellerCudaAdapterError::Device(error)) => {
            if std::env::var_os("SONIUM_PP2_REQUIRE_CUDA").is_some() {
                panic!("CUDA is required for PP2 adapter evidence: {error}");
            }
            eprintln!("no CUDA device — skipping PP2 adapter evidence: {error}");
            return;
        }
        Err(error) => panic!("PP2 CUDA adapter failed with typed error: {error}"),
    };

    assert_eq!(result.schema, "sonium.propellers.cuda-native-adapter.v1");
    assert_eq!(result.requested_backend, "cuda");
    assert_eq!(result.actual_backend, "cuda");
    assert_eq!(result.precision, "f64 / Complex64");
    assert!(result.panel_count > 0);
    assert_eq!(result.pair_count, result.cuda.len());
    assert_eq!(result.cpu.len(), result.cuda.len());
    assert!(result.max_relative_error.is_finite());
    assert!(
        result.max_relative_error <= PARITY_TOLERANCE,
        "native CUDA BEM panel parity exceeded budget: {}",
        result.max_relative_error
    );
    assert!(result.runtime.kernel_launches > 0);
    assert!(result.runtime.synchronization_calls > 0);
    assert!(result.runtime.allocation_operations > 0);
    assert!(result.runtime.allocation_high_water_bytes > 0);
    assert!(result.runtime.transfer.h2d_bytes > 0);
    assert!(result.runtime.transfer.d2h_bytes > 0);
    assert_eq!(result.runtime.live_allocation_bytes, 0);
    assert_eq!(result.runtime.live_staging_bytes, 0);

    let mut session =
        match PropellerCudaPanelSession::new(&test_design(), &test_mesh_config(), 1, 0) {
            Ok(session) => session,
            Err(PropellerCudaAdapterError::Device(error)) => {
                panic!("session creation lost the already available CUDA device: {error}");
            }
            Err(error) => panic!("PP2 CUDA warm session failed with typed error: {error}"),
        };
    let session_initialization = session.runtime_stats();
    assert!(session_initialization.allocation_operations > 0);
    assert!(session_initialization.transfer.h2d_operations > 0);
    let warm_first = session
        .try_solve(&test_design(), 1)
        .expect("first warm replay");
    let warm_second = session
        .try_solve(&test_design(), 2)
        .expect("second warm replay with updated harmonic");
    assert!(warm_first.max_relative_error <= PARITY_TOLERANCE);
    assert!(warm_second.max_relative_error <= PARITY_TOLERANCE);
    assert!(
        warm_first
            .cuda
            .iter()
            .zip(&warm_second.cuda)
            .any(|(first, second)| first.g != second.g || first.h != second.h)
    );
    assert!(warm_first.runtime.allocation_operations > 0);
    assert_eq!(warm_second.runtime.allocation_operations, 0);
    assert_eq!(warm_second.runtime.transfer.h2d_operations, 0);
    assert_eq!(warm_second.runtime.transfer.h2d_bytes, 0);
    assert_eq!(warm_second.runtime.transfer.d2h_operations, 1);
    assert!(warm_second.runtime.transfer.d2h_bytes > 0);
    assert_eq!(warm_second.runtime.kernel_launches, 1);
    assert_eq!(warm_second.runtime.synchronization_calls, 1);
    assert!(warm_second.runtime.live_allocation_bytes > 0);
    let warm_resident_bytes = session.runtime_stats().live_allocation_bytes;
    assert!(warm_resident_bytes > 0);

    if let Some(path) = std::env::var_os("SONIUM_PP2_CUDA_ADAPTER_ARTIFACT") {
        write_artifact(
            PathBuf::from(path),
            &result,
            session_initialization,
            warm_first.runtime,
            warm_second.runtime,
            warm_resident_bytes,
        );
    }
}

#[test]
fn pp2_warm_session_rejects_failure_without_partial_result_and_recovers() {
    assert_eq!(std::env::consts::OS, "linux", "PP2 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "PP2 requires native x86_64"
    );
    let mut session =
        match PropellerCudaPanelSession::new(&test_design(), &test_mesh_config(), 1, 0) {
            Ok(session) => session,
            Err(PropellerCudaAdapterError::Device(error)) => {
                eprintln!("no CUDA device — skipping PP2 warm failure evidence: {error}");
                return;
            }
            Err(error) => panic!("session creation failed with typed error: {error}"),
        };
    session.inject_failure(CudaFailureInjectionPoint::TransferD2H);
    let error = session
        .try_solve(&test_design(), 1)
        .expect_err("injected D2H failure must reject the solve");
    assert!(matches!(
        error,
        PropellerCudaAdapterError::Backend(BackendError::Transfer { .. })
    ));
    session
        .recover_after_error()
        .expect("recoverable transfer failure should clear the session error");
    let recovered = session
        .try_solve(&test_design(), 1)
        .expect("session should remain usable after recoverable failure");
    assert!(recovered.max_relative_error <= PARITY_TOLERANCE);
}

#[test]
fn pp2_panel_session_driver_shim_lifecycle_evidence() {
    assert_eq!(std::env::consts::OS, "linux", "PP2 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "PP2 requires native x86_64"
    );
    if session_or_skip().is_none() {
        return;
    }

    let mut lanes = Vec::new();

    // Allocation failures happen on the first warm solve when the operator's
    // scratch workspace is materialized. The checked pool rejects before
    // charging the ledger, and the session remains reusable.
    let mut session = PropellerCudaPanelSession::new_with_backend(
        &test_design(),
        &test_mesh_config(),
        1,
        CudaBackend::new(0).expect("CUDA backend for allocation lane"),
    )
    .expect("allocation lane session");
    let output_before = session.last_published_output().to_vec();
    let ledger_before = session.runtime_stats();
    session.inject_failure(CudaFailureInjectionPoint::Allocation);
    let error = session
        .try_solve(&test_design(), 1)
        .expect_err("injected allocation failure must reject the solve");
    assert!(matches!(
        error,
        PropellerCudaAdapterError::Backend(BackendError::Allocation { .. })
    ));
    let output_unchanged = outputs_equal(&output_before, session.last_published_output());
    let ledger_after_failure = session.runtime_stats();
    assert!(output_unchanged);
    assert_eq!(
        ledger_after_failure.live_allocation_bytes,
        ledger_before.live_allocation_bytes
    );
    assert_eq!(
        ledger_after_failure.allocation_operations,
        ledger_before.allocation_operations
    );
    session
        .recover_after_error()
        .expect("allocation failure is reusable without synchronization");
    let retry = session
        .try_solve(&test_design(), 1)
        .expect("allocation lane retry");
    assert!(retry.max_relative_error <= PARITY_TOLERANCE);
    let ledger_after_recovery = session.runtime_stats();
    drop(session);
    let cleanup = fresh_context_ledger();
    lanes.push(json!({
        "name": "allocation",
        "status": "typed_rejected",
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "fallback_reason": null,
        "failure": backend_error_json(&error),
        "output": {"unchanged": output_unchanged, "published": false},
        "recovery": {
            "context_disposition": "reusable",
            "strategy": "recover_after_error_then_retry_same_session",
            "recovery_status": "passed",
            "retry_status": "passed",
        },
        "ledger": {
            "before": runtime_json(ledger_before),
            "after_failure": runtime_json(ledger_after_failure),
            "after_recovery": runtime_json(ledger_after_recovery),
            "fresh_context_after_drop": cleanup,
        },
    }));

    // Launch injection is a recoverable native execution failure. The output
    // destination still contains the previous publication, then the same
    // session is synchronized and retried successfully.
    let mut session = session_or_skip().expect("CUDA launch lane session");
    let baseline = session
        .try_solve(&test_design(), 1)
        .expect("launch lane baseline");
    let output_before = baseline.cuda.clone();
    let ledger_before = session.runtime_stats();
    session.inject_failure(CudaFailureInjectionPoint::Launch);
    let error = session
        .try_solve(&test_design(), 1)
        .expect_err("injected launch failure must reject the solve");
    assert!(matches!(
        error,
        PropellerCudaAdapterError::Backend(BackendError::Launch { .. })
    ));
    let output_unchanged = outputs_equal(&output_before, session.last_published_output());
    assert!(output_unchanged);
    let ledger_after_failure = session.runtime_stats();
    session
        .recover_after_error()
        .expect("launch failure should recover after synchronization");
    let retry = session
        .try_solve(&test_design(), 1)
        .expect("launch lane retry");
    assert!(retry.max_relative_error <= PARITY_TOLERANCE);
    let ledger_after_recovery = session.runtime_stats();
    drop(session);
    let cleanup = fresh_context_ledger();
    lanes.push(json!({
        "name": "launch",
        "status": "typed_rejected",
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "fallback_reason": null,
        "failure": backend_error_json(&error),
        "output": {"unchanged": output_unchanged, "published": false},
        "recovery": {
            "context_disposition": "synchronize_before_reuse",
            "strategy": "recover_after_error_then_retry_same_session",
            "recovery_status": "passed",
            "retry_status": "passed",
        },
        "ledger": {
            "before": runtime_json(ledger_before),
            "after_failure": runtime_json(ledger_after_failure),
            "after_recovery": runtime_json(ledger_after_recovery),
            "fresh_context_after_drop": cleanup,
        },
    }));

    // Synchronization failure poisons the context. No host publication is
    // attempted; recovery reports the original typed error and a fresh
    // context is required for the successful retry.
    let mut session = session_or_skip().expect("CUDA synchronization lane session");
    let baseline = session
        .try_solve(&test_design(), 1)
        .expect("synchronization lane baseline");
    let output_before = baseline.cuda.clone();
    let ledger_before = session.runtime_stats();
    session.inject_failure(CudaFailureInjectionPoint::Synchronization);
    let error = session
        .try_solve(&test_design(), 1)
        .expect_err("injected synchronization failure must reject the solve");
    assert!(matches!(
        error,
        PropellerCudaAdapterError::Backend(BackendError::Synchronization { .. })
    ));
    let output_unchanged = outputs_equal(&output_before, session.last_published_output());
    assert!(output_unchanged);
    let ledger_after_failure = session.runtime_stats();
    let recovery_error = session
        .recover_after_error()
        .expect_err("synchronization failure requires fresh context");
    assert_eq!(recovery_error.to_string(), error.to_string());
    drop(session);
    let cleanup_before_retry = fresh_context_ledger();
    let mut fresh = session_or_skip().expect("fresh CUDA synchronization retry session");
    let retry = fresh
        .try_solve(&test_design(), 1)
        .expect("synchronization lane fresh-context retry");
    assert!(retry.max_relative_error <= PARITY_TOLERANCE);
    let ledger_after_recovery = fresh.runtime_stats();
    drop(fresh);
    let cleanup = fresh_context_ledger();
    lanes.push(json!({
        "name": "synchronization",
        "status": "typed_rejected",
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "fallback_reason": null,
        "failure": backend_error_json(&error),
        "output": {"unchanged": output_unchanged, "published": false},
        "recovery": {
            "context_disposition": "must_recreate",
            "strategy": "drop_poisoned_session_and_recreate_context",
            "recovery_status": "original_error_retained",
            "retry_status": "passed_on_fresh_context",
        },
        "ledger": {
            "before": runtime_json(ledger_before),
            "after_failure": runtime_json(ledger_after_failure),
            "fresh_context_before_retry": cleanup_before_retry,
            "after_recovery": runtime_json(ledger_after_recovery),
            "fresh_context_after_drop": cleanup,
        },
    }));

    // Device-loss injection has the same terminal disposition as a real
    // lost CUDA context: the adapter drops all resident resources and retries
    // only through a newly initialized context.
    let mut session = session_or_skip().expect("CUDA device-loss lane session");
    let baseline = session
        .try_solve(&test_design(), 1)
        .expect("device-loss lane baseline");
    let output_before = baseline.cuda.clone();
    let ledger_before = session.runtime_stats();
    session.inject_failure(CudaFailureInjectionPoint::DeviceLoss);
    let error = session
        .try_solve(&test_design(), 1)
        .expect_err("injected device loss must reject the solve");
    assert!(matches!(
        error,
        PropellerCudaAdapterError::Backend(BackendError::DeviceLost { .. })
    ));
    let output_unchanged = outputs_equal(&output_before, session.last_published_output());
    assert!(output_unchanged);
    let ledger_after_failure = session.runtime_stats();
    let recovery_error = session
        .recover_after_error()
        .expect_err("device loss requires fresh context");
    assert_eq!(recovery_error.to_string(), error.to_string());
    drop(session);
    let cleanup_before_retry = fresh_context_ledger();
    let mut fresh = session_or_skip().expect("fresh CUDA device-loss retry session");
    let retry = fresh
        .try_solve(&test_design(), 1)
        .expect("device-loss lane fresh-context retry");
    assert!(retry.max_relative_error <= PARITY_TOLERANCE);
    let ledger_after_recovery = fresh.runtime_stats();
    drop(fresh);
    let cleanup = fresh_context_ledger();
    lanes.push(json!({
        "name": "device_loss",
        "status": "typed_rejected",
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "fallback_reason": null,
        "failure": backend_error_json(&error),
        "output": {"unchanged": output_unchanged, "published": false},
        "recovery": {
            "context_disposition": "must_recreate",
            "strategy": "drop_poisoned_session_and_recreate_context",
            "recovery_status": "original_error_retained",
            "retry_status": "passed_on_fresh_context",
        },
        "ledger": {
            "before": runtime_json(ledger_before),
            "after_failure": runtime_json(ledger_after_failure),
            "fresh_context_before_retry": cleanup_before_retry,
            "after_recovery": runtime_json(ledger_after_recovery),
            "fresh_context_after_drop": cleanup,
        },
    }));

    // There is deliberately no cancellation token in this bounded adapter.
    // Keep the requested backend provenance explicit while refusing to turn
    // an unsupported cancellation request into a passing execution row.
    lanes.push(json!({
        "name": "cancellation",
        "status": "unsupported",
        "requested_backend": "cuda",
        "actual_backend": null,
        "fallback_reason": "cancellation-token API is unsupported by PropellerCudaPanelSession",
        "failure": null,
        "output": {"unchanged": true, "published": false},
        "recovery": {
            "context_disposition": "not_applicable",
            "strategy": "not_supported",
            "recovery_status": "not_applicable",
            "retry_status": "not_applicable",
        },
        "reason": "PropellerCudaPanelSession has no cancellation-token API",
    }));

    if let Some(path) = std::env::var_os("SONIUM_PP2_CUDA_LIFECYCLE_ARTIFACT") {
        write_lifecycle_artifact(PathBuf::from(path), lanes);
    }
}

#[test]
fn pp2_warm_session_teardown_releases_resident_buffers() {
    assert_eq!(std::env::consts::OS, "linux", "PP2 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "PP2 requires native x86_64"
    );
    let mut session =
        match PropellerCudaPanelSession::new(&test_design(), &test_mesh_config(), 1, 0) {
            Ok(session) => session,
            Err(PropellerCudaAdapterError::Device(error)) => {
                eprintln!("no CUDA device — skipping PP2 warm teardown evidence: {error}");
                return;
            }
            Err(error) => panic!("session creation failed with typed error: {error}"),
        };
    session
        .try_solve(&test_design(), 1)
        .expect("warm setup solve");
    assert!(session.runtime_stats().live_allocation_bytes > 0);
    drop(session);
    // Recreating the same backend after the session drops checks that the
    // resident pair/workspace buffers and their CUDA context teardown are
    // not left in a poisoned state.
    let backend =
        sonium_compute::CudaBackend::new(0).expect("CUDA backend can be recreated after teardown");
    assert_eq!(backend.runtime_stats().live_allocation_bytes, 0);
}
