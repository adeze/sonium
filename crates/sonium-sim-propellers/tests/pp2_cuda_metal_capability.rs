//! PP2 capability-boundary evidence for the propeller application.
//!
//! The propeller crate owns one bounded CUDA adapter cell and no Metal adapter.
//! This test proves the capability boundary over the complete matrix: the
//! supported CUDA panel-BEM cell is separated from the 431 typed unsupported
//! device cells, which have no actual backend and cannot silently fall back to
//! the CPU surrogate. The separate adapter test exercises that one cell; this
//! producer records policy only. A GPU inventory is recorded when the runner
//! supplies one, but hardware presence alone is not treated as execution.

use serde::Serialize;
use sonium_sim_propellers::{
    EXECUTION_EVIDENCE_SCHEMA_V1, PropellerBackend, PropellerCapabilityMatrix,
    PropellerCapabilityRequest, PropellerCapabilityResult, PropellerExecutionBatch,
    PropellerExecutionDirection, PropellerHostMetadata, PropellerPrecision, PropellerWorkload,
};
use std::fs;
use std::path::PathBuf;

const SCHEMA: &str = "sonium.propellers.cuda-metal-capability.v1";

#[derive(Debug, Serialize)]
struct DeviceExecutionEvidence {
    status: &'static str,
    adapter: &'static str,
    requested_backend: Vec<&'static str>,
    actual_backend: Option<&'static str>,
    fallback_reason: Option<&'static str>,
    allocations: u64,
    transfers: u64,
    synchronizations: u64,
    launches: u64,
    peak_live_bytes: u64,
    residency_policy: &'static str,
    invalidation_policy: &'static str,
    warm_replays: &'static str,
}

#[derive(Debug, Serialize)]
struct FaultEvidence {
    id: &'static str,
    status: &'static str,
    reason: &'static str,
    partial_publish: bool,
}

#[derive(Debug, Serialize)]
struct Pp2Artifact {
    schema: &'static str,
    status: &'static str,
    result_kind: &'static str,
    scope: &'static str,
    host: PropellerHostMetadata,
    device_inventory: String,
    matrix: PropellerCapabilityMatrix,
    device_cells: usize,
    supported_device_cells: usize,
    typed_unsupported_cells: usize,
    requests_with_no_actual_backend: usize,
    requests_with_no_fallback: usize,
    cpu_reference: &'static str,
    device_execution: DeviceExecutionEvidence,
    fault_probes: Vec<FaultEvidence>,
}

fn device_inventory() -> String {
    std::env::var_os("SONIUM_PP2_DEVICE_INVENTORY_FILE")
        .and_then(|path| fs::read_to_string(path).ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unavailable (inventory not supplied)".into())
}

fn matrix_evidence() -> (PropellerCapabilityMatrix, usize, usize, usize, usize, usize) {
    let matrix = PropellerCapabilityMatrix::current();
    matrix.validate().expect("complete matrix policy");
    let mut device_cells = 0;
    let mut supported_device_cells = 0;
    let mut typed_unsupported_cells = 0;
    let mut no_actual_backend = 0;
    let mut no_fallback = 0;
    for cell in &matrix.cells {
        if !matches!(
            cell.request.backend,
            PropellerBackend::Cuda | PropellerBackend::Metal
        ) {
            continue;
        }
        device_cells += 1;
        match matrix.resolve(cell.request) {
            PropellerCapabilityResult::Supported(request) => {
                assert_eq!(request.backend, PropellerBackend::Cuda);
                assert_eq!(request.workload, PropellerWorkload::BladeElementBem);
                assert_eq!(request.precision, PropellerPrecision::F64);
                assert_eq!(request.mesh_order, 1);
                assert_eq!(request.direction, PropellerExecutionDirection::Forward);
                assert!(matches!(
                    request.batch,
                    PropellerExecutionBatch::Single | PropellerExecutionBatch::Multipoint
                ));
                supported_device_cells += 1;
            }
            PropellerCapabilityResult::Unsupported(result) => {
                assert_eq!(result.requested_backend, cell.request.backend);
                assert_eq!(result.actual_backend, None);
                assert_eq!(result.fallback_reason, None);
                assert!(!result.reason.trim().is_empty());
                typed_unsupported_cells += 1;
                if result.actual_backend.is_none() {
                    no_actual_backend += 1;
                }
                if result.fallback_reason.is_none() {
                    no_fallback += 1;
                }
            }
        }
    }
    assert_eq!(matrix.schema, EXECUTION_EVIDENCE_SCHEMA_V1);
    assert_eq!(matrix.cells.len(), 648);
    assert_eq!(device_cells, 432);
    let expected_supported = usize::from(cfg!(feature = "cuda"));
    assert_eq!(supported_device_cells, expected_supported * 2);
    assert_eq!(
        typed_unsupported_cells,
        device_cells - expected_supported * 2
    );
    assert_eq!(no_actual_backend, typed_unsupported_cells);
    assert_eq!(no_fallback, typed_unsupported_cells);
    (
        matrix,
        device_cells,
        supported_device_cells,
        typed_unsupported_cells,
        no_actual_backend,
        no_fallback,
    )
}

#[test]
fn pp2_cuda_metal_matrix_is_complete_and_fail_closed() {
    assert_eq!(std::env::consts::OS, "linux", "PP2 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "PP2 requires native x86_64"
    );

    let (
        matrix,
        device_cells,
        supported_device_cells,
        typed_unsupported_cells,
        no_actual_backend,
        no_fallback,
    ) = matrix_evidence();
    let artifact = Pp2Artifact {
        schema: SCHEMA,
        status: "pass",
        result_kind: "capability-boundary-plus-native-cuda-cell",
        scope: "native Linux x86_64; Windows/Arm64 waived; one bounded CUDA panel-BEM cell; Metal adapter unavailable",
        host: PropellerHostMetadata::collect(),
        device_inventory: device_inventory(),
        matrix,
        device_cells,
        supported_device_cells,
        typed_unsupported_cells,
        requests_with_no_actual_backend: no_actual_backend,
        requests_with_no_fallback: no_fallback,
        cpu_reference: "available; no device result promoted and no CPU/device parity claim",
        device_execution: DeviceExecutionEvidence {
            status: "bounded-native-cell-separate-artifact",
            adapter: "one CUDA Complex64 regular-panel-BEM cell; see cuda-native-adapter-v1.json",
            requested_backend: vec!["cuda", "metal"],
            actual_backend: None,
            fallback_reason: None,
            allocations: 0,
            transfers: 0,
            synchronizations: 0,
            launches: 0,
            peak_live_bytes: 0,
            residency_policy: "see cuda-native-adapter-v1.json; panel input/output buffers are device-resident during the checked kernel",
            invalidation_policy: "bounded call owns and drops panel buffers after result publication",
            warm_replays: "two bounded production-connected harmonic replays with resident panel/workspace counters in separate adapter artifact",
        },
        fault_probes: vec![
            FaultEvidence {
                id: "allocation",
                status: "not-applicable",
                reason: "matrix producer does not execute the separate native adapter",
                partial_publish: false,
            },
            FaultEvidence {
                id: "launch-synchronization-device-loss-cancellation-nonconvergence",
                status: "not-applicable",
                reason: "matrix producer does not execute the separate native adapter",
                partial_publish: false,
            },
        ],
    };

    if let Some(output_dir) = std::env::var_os("SONIUM_OUTPUT_DIR") {
        let output_dir = PathBuf::from(output_dir);
        fs::create_dir_all(&output_dir).expect("create PP2 output directory");
        let bytes = serde_json::to_vec_pretty(&artifact).expect("serialize PP2 artifact");
        fs::write(output_dir.join("cuda-metal-capability-v1.json"), bytes)
            .expect("write PP2 artifact");
    }
}

#[test]
fn pp2_device_resolution_rejects_every_declared_backend_cell() {
    let matrix = PropellerCapabilityMatrix::current();
    for backend in [PropellerBackend::Cuda, PropellerBackend::Metal] {
        for workload in [
            PropellerWorkload::SourceSurrogate,
            PropellerWorkload::BladeElementBem,
            PropellerWorkload::VortexLattice,
            PropellerWorkload::FreeWake,
            PropellerWorkload::BodyFittedMeridional,
            PropellerWorkload::AxisymmetricFlow,
            PropellerWorkload::RotatingWallSwirl,
            PropellerWorkload::ObserverFwh,
            PropellerWorkload::Structural,
        ] {
            for precision in [PropellerPrecision::F32, PropellerPrecision::F64] {
                for mesh_order in 1..=2 {
                    for direction in [
                        PropellerExecutionDirection::Forward,
                        PropellerExecutionDirection::Transpose,
                    ] {
                        for batch in [
                            PropellerExecutionBatch::Single,
                            PropellerExecutionBatch::Multipoint,
                            PropellerExecutionBatch::Transient,
                        ] {
                            let request = PropellerCapabilityRequest {
                                backend,
                                workload,
                                precision,
                                mesh_order,
                                direction,
                                batch,
                            };
                            match matrix.resolve(request) {
                                PropellerCapabilityResult::Supported(actual) => {
                                    assert_eq!(actual.backend, PropellerBackend::Cuda);
                                    assert_eq!(actual.workload, PropellerWorkload::BladeElementBem);
                                    assert_eq!(actual.precision, PropellerPrecision::F64);
                                    assert_eq!(actual.mesh_order, 1);
                                    assert_eq!(
                                        actual.direction,
                                        PropellerExecutionDirection::Forward
                                    );
                                    assert!(matches!(
                                        actual.batch,
                                        PropellerExecutionBatch::Single
                                            | PropellerExecutionBatch::Multipoint
                                    ));
                                }
                                PropellerCapabilityResult::Unsupported(result) => {
                                    assert_eq!(result.actual_backend, None);
                                    assert_eq!(result.fallback_reason, None);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
