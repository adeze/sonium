//! Capability-matrix QA for the application-level propeller lanes.

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    PropellerBackend, PropellerCapabilityMatrix, PropellerCapabilityRequest,
    PropellerCapabilityResult, PropellerExecutionBatch, PropellerExecutionDirection,
    PropellerPrecision, PropellerWorkload,
};
use std::time::Instant;

const CASE_ID: &str = "propeller.capability-matrix.v1";

fn run() -> Result<(usize, usize), String> {
    let matrix = PropellerCapabilityMatrix::current();
    let cpu_request = PropellerCapabilityRequest {
        backend: PropellerBackend::Cpu,
        workload: PropellerWorkload::SourceSurrogate,
        precision: PropellerPrecision::F64,
        mesh_order: 1,
        direction: PropellerExecutionDirection::Forward,
        batch: PropellerExecutionBatch::Single,
    };
    if !matches!(
        matrix.resolve(cpu_request),
        PropellerCapabilityResult::Supported(_)
    ) {
        return Err("the declared CPU/f64 source-surrogate cell is not supported".into());
    }

    let device_cells = matrix
        .cells
        .iter()
        .filter(|cell| {
            matches!(
                cell.request.backend,
                PropellerBackend::Cuda | PropellerBackend::Metal
            )
        })
        .collect::<Vec<_>>();
    if device_cells.is_empty() || device_cells.iter().any(|cell| cell.supported) {
        return Err(
            "CUDA/Metal cells must be explicit unsupported results until a native adapter exists"
                .into(),
        );
    }
    for cell in device_cells {
        let PropellerCapabilityResult::Unsupported(unsupported) = matrix.resolve(cell.request)
        else {
            return Err("device capability resolution unexpectedly accepted a cell".into());
        };
        if unsupported.actual_backend.is_some()
            || unsupported.fallback_reason.is_some()
            || unsupported.reason.trim().is_empty()
        {
            return Err("unsupported device capability carried fallback or missing reason".into());
        }
    }
    Ok((matrix.cells.len(), matrix.cells.len() / 3 * 2))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((total_cells, device_cells)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("matrix_cells", total_cells as f64)
                .metric("typed_unsupported_device_cells", device_cells as f64)
                .minimum("matrix_cells", 1.0)
                .minimum("typed_unsupported_device_cells", 1.0)
                .reference("explicit application capability matrix; no implicit CUDA/Metal fallback")
                .dimension("CPU/f64 application references; CUDA and Metal typed unsupported cells")
                .tolerance("supported CPU cells resolve; every device cell has no actual backend and a reason")
                .seed("propeller-capability-matrix-v1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Ok(json) = result.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX))
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("PROPELLER CAPABILITY MATRIX QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
