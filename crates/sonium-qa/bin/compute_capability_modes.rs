//! CPU capability contract for mode-sensitive fused reductions.
//!
//! This keeps the operation capability and reduction-mode capability joined:
//! deterministic/reproducible multi-dot must not be reported as native when
//! the mode requires host staging.

use num_complex::{Complex32, Complex64};
use sonium_compute::{Backend, CapabilityState, CpuBackend, DeviceScalar, ReductionMode};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.capability-modes.v1";

fn check_scalar<T: DeviceScalar>(backend: &CpuBackend) -> Result<usize, String> {
    let capabilities = backend.capabilities();
    let mut cells = 0;
    for mode in [
        ReductionMode::Fast,
        ReductionMode::Reproducible,
        ReductionMode::Deterministic,
    ] {
        let state = capabilities.multi_dot_capability_for::<T>(mode);
        if state != CapabilityState::Native {
            return Err(format!(
                "CPU multi-dot mode {:?} for {} advertised as {:?}",
                mode,
                std::any::type_name::<T>(),
                state
            ));
        }
        cells += 1;
    }
    Ok(cells)
}

fn run() -> Result<usize, String> {
    let backend = CpuBackend::new();
    Ok(check_scalar::<f32>(&backend)?
        + check_scalar::<f64>(&backend)?
        + check_scalar::<Complex32>(&backend)?
        + check_scalar::<Complex64>(&backend)?)
}

fn main() {
    match run() {
        Ok(mode_cells) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("mode_cells", mode_cells as f64)
                .minimum("mode_cells", 12.0)
                .reference("CPU capability table joined with reduction-mode residency")
                .reference_checksum("analytic/deterministic-fixture:compute-capability-modes-v1")
                .dimension("four scalar families; fast/reproducible/deterministic modes")
                .seed("compute-capability-modes:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("capability-mode QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("capability-mode QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
