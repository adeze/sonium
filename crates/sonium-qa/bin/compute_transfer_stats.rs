//! CPU reference for backend transfer accounting.
//!
//! The same checked copy boundaries are used by CUDA and Metal. This case
//! provides a deterministic baseline for the counters without requiring a
//! physical accelerator or a simulator workload.

use sonium_compute::{Backend, CpuBackend};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.transfer-stats.v1";

#[derive(Debug)]
struct Metrics {
    h2d_operations: usize,
    h2d_bytes: usize,
    d2h_operations: usize,
    d2h_bytes: usize,
    d2d_operations: usize,
    d2d_bytes: usize,
    invalid_did_not_count: bool,
}

fn run() -> Result<Metrics, String> {
    let backend = CpuBackend::new();
    backend.reset_transfer_stats();
    let mut source = backend
        .raw_alloc::<f64>(4)
        .map_err(|error| format!("source allocation failed: {error}"))?;
    let mut destination = backend
        .raw_alloc::<f64>(4)
        .map_err(|error| format!("destination allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&[1.0, 2.0, 3.0, 4.0], &mut source)
        .map_err(|error| format!("host-to-device copy failed: {error}"))?;
    backend
        .try_copy_d2d(&source, &mut destination)
        .map_err(|error| format!("device-to-device copy failed: {error}"))?;
    let mut host = [0.0_f64; 4];
    backend
        .try_copy_d2h(&destination, &mut host)
        .map_err(|error| format!("device-to-host copy failed: {error}"))?;
    let before_invalid = backend.transfer_stats();
    let invalid = backend.try_copy_h2d(&[9.0_f64], &mut source);
    if invalid.is_ok() {
        return Err("malformed transfer was accepted".into());
    }
    let after_invalid = backend.transfer_stats();
    if host != [1.0, 2.0, 3.0, 4.0] {
        return Err(format!("round-trip data mismatch: {host:?}"));
    }
    Ok(Metrics {
        h2d_operations: after_invalid.h2d_operations,
        h2d_bytes: after_invalid.h2d_bytes,
        d2h_operations: after_invalid.d2h_operations,
        d2h_bytes: after_invalid.d2h_bytes,
        d2d_operations: after_invalid.d2d_operations,
        d2d_bytes: after_invalid.d2d_bytes,
        invalid_did_not_count: before_invalid == after_invalid,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("h2d_operations", metrics.h2d_operations as f64)
                .metric("h2d_bytes", metrics.h2d_bytes as f64)
                .metric("d2h_operations", metrics.d2h_operations as f64)
                .metric("d2h_bytes", metrics.d2h_bytes as f64)
                .metric("d2d_operations", metrics.d2d_operations as f64)
                .metric("d2d_bytes", metrics.d2d_bytes as f64)
                .metric(
                    "invalid_did_not_count",
                    metrics.invalid_did_not_count as u8 as f64,
                )
                .minimum("h2d_operations", 1.0)
                .minimum("h2d_bytes", 32.0)
                .minimum("d2h_operations", 1.0)
                .minimum("d2h_bytes", 32.0)
                .minimum("d2d_operations", 1.0)
                .minimum("d2d_bytes", 32.0)
                .minimum("invalid_did_not_count", 1.0)
                .reference("checked CPU copy boundaries and independent round-trip data")
                .reference_checksum("analytic/deterministic-fixture:compute-transfer-stats-v1")
                .dimension("four-element f64 host/device buffers")
                .seed("compute-transfer-stats:4xf64:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("transfer stats QA result must serialize")
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
                    .expect("transfer stats QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
