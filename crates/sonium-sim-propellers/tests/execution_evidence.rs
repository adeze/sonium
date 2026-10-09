use sonium_sim_propellers::{
    EXECUTION_EVIDENCE_SCHEMA_V1, PropellerBackend, PropellerCapabilityMatrix,
    PropellerCapabilityRequest, PropellerCapabilityResult, PropellerExecutionBatch,
    PropellerExecutionDirection, PropellerPrecision, PropellerWorkload,
};

#[test]
fn capability_matrix_covers_the_declared_application_dimensions() {
    let matrix = PropellerCapabilityMatrix::current();
    assert_eq!(matrix.schema, EXECUTION_EVIDENCE_SCHEMA_V1);
    assert_eq!(matrix.cells.len(), 648);
    assert!(
        matrix
            .cells
            .iter()
            .all(|cell| { cell.request.backend == PropellerBackend::Cpu || !cell.supported })
    );
    assert!(matrix.cells.iter().any(|cell| {
        cell.request.workload == PropellerWorkload::ObserverFwh
            && cell.request.batch == PropellerExecutionBatch::Single
    }));
    assert!(matrix.cells.iter().any(|cell| {
        cell.request.workload == PropellerWorkload::Structural
            && cell.request.direction == PropellerExecutionDirection::Transpose
    }));
}

#[test]
fn device_requests_are_typed_unsupported_without_a_cpu_fallback() {
    let matrix = PropellerCapabilityMatrix::current();
    for backend in [PropellerBackend::Cuda, PropellerBackend::Metal] {
        let request = PropellerCapabilityRequest {
            backend,
            workload: PropellerWorkload::BodyFittedMeridional,
            precision: PropellerPrecision::F64,
            mesh_order: 1,
            direction: PropellerExecutionDirection::Forward,
            batch: PropellerExecutionBatch::Single,
        };
        let PropellerCapabilityResult::Unsupported(result) = matrix.resolve(request) else {
            panic!("{backend} must not be reported as supported");
        };
        assert_eq!(result.requested_backend, backend);
        assert_eq!(result.actual_backend, None);
        assert_eq!(result.fallback_reason, None);
        assert!(!result.reason.is_empty());
    }
}
