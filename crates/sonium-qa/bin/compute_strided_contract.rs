//! Checked flat-basis projection contract for the CPU backend.
//!
//! This case exercises the failure boundary that used to grow its
//! coefficient vector through an infallible `Vec` path.  Shape, non-finite,
//! and allocation failures must be typed; successful projection must preserve
//! the fixed-order reduction result.

use sonium_compute::{Backend, BackendError, CpuBackend, ReductionMode};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.strided-projection-contract.v1";

#[derive(Debug)]
struct Metrics {
    successful_coefficients: usize,
    shape_rejected: bool,
    non_finite_rejected: bool,
    allocation_rejected: bool,
    destination_preserved: bool,
}

fn run() -> Result<Metrics, String> {
    let backend = CpuBackend::new();
    let mut basis = backend
        .raw_alloc::<f64>(6)
        .map_err(|error| format!("basis allocation failed: {error}"))?;
    let mut y = backend
        .raw_alloc::<f64>(3)
        .map_err(|error| format!("projection vector allocation failed: {error}"))?;
    backend.copy_h2d(&[1.0, 2.0, 3.0, -1.0, 0.5, 4.0], &mut basis);
    backend.copy_h2d(&[2.0, -1.0, 0.5], &mut y);

    let coefficients = backend
        .try_basis_project_out_strided_with_mode(ReductionMode::Deterministic, &basis, 3, 2, &mut y)
        .map_err(|error| format!("valid projection failed: {error}"))?;
    if coefficients.len() != 2
        || (coefficients[0] - 1.5).abs() > 1.0e-12
        || (coefficients[1] + 0.5).abs() > 1.0e-12
    {
        return Err(format!(
            "unexpected projection coefficients: {coefficients:?}"
        ));
    }

    let mut short_basis = backend
        .raw_alloc::<f64>(5)
        .map_err(|error| format!("short basis allocation failed: {error}"))?;
    let mut sentinel = backend
        .raw_alloc::<f64>(3)
        .map_err(|error| format!("sentinel allocation failed: {error}"))?;
    backend.copy_h2d(&[9.0, 8.0, 7.0, 6.0, 5.0], &mut short_basis);
    backend.copy_h2d(&[11.0, 12.0, 13.0], &mut sentinel);
    let shape_error = backend.try_basis_project_out_strided_with_mode(
        ReductionMode::Fast,
        &short_basis,
        3,
        2,
        &mut sentinel,
    );
    let shape_rejected = matches!(shape_error, Err(BackendError::InvalidArgument { .. }));
    if !shape_rejected {
        return Err(format!("short basis was not rejected: {shape_error:?}"));
    }

    let mut non_finite_basis = backend
        .raw_alloc::<f64>(3)
        .map_err(|error| format!("non-finite basis allocation failed: {error}"))?;
    backend.copy_h2d(&[f64::NAN, 1.0, 2.0], &mut non_finite_basis);
    let non_finite_error = backend.try_basis_project_out_strided_with_mode(
        ReductionMode::Deterministic,
        &non_finite_basis,
        3,
        1,
        &mut sentinel,
    );
    let non_finite_rejected = matches!(
        non_finite_error,
        Err(BackendError::OperationFailed {
            operation: "basis_project_out_strided",
            ..
        })
    );
    if !non_finite_rejected {
        return Err(format!(
            "non-finite basis was not rejected: {non_finite_error:?}"
        ));
    }

    let empty_basis = backend
        .raw_alloc::<f64>(0)
        .map_err(|error| format!("empty basis allocation failed: {error}"))?;
    let mut empty_y = backend
        .raw_alloc::<f64>(0)
        .map_err(|error| format!("empty output allocation failed: {error}"))?;
    let allocation_error = backend.try_basis_project_out_strided_with_mode(
        ReductionMode::Deterministic,
        &empty_basis,
        0,
        usize::MAX,
        &mut empty_y,
    );
    let allocation_rejected = matches!(allocation_error, Err(BackendError::Allocation { .. }));
    if !allocation_rejected {
        return Err(format!(
            "coefficient allocation was not rejected: {allocation_error:?}"
        ));
    }

    let mut sentinel_host = [0.0; 3];
    backend.copy_d2h(&sentinel, &mut sentinel_host);
    let destination_preserved = sentinel_host == [11.0, 12.0, 13.0];
    if !destination_preserved {
        return Err(format!(
            "rejected projection changed destination: {sentinel_host:?}"
        ));
    }

    Ok(Metrics {
        successful_coefficients: coefficients.len(),
        shape_rejected,
        non_finite_rejected,
        allocation_rejected,
        destination_preserved,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .backend_passed("cpu")
                .metric(
                    "successful_coefficients",
                    metrics.successful_coefficients as f64,
                )
                .metric("shape_rejected", metrics.shape_rejected as u8 as f64)
                .metric(
                    "non_finite_rejected",
                    metrics.non_finite_rejected as u8 as f64,
                )
                .metric(
                    "allocation_rejected",
                    metrics.allocation_rejected as u8 as f64,
                )
                .metric(
                    "destination_preserved",
                    metrics.destination_preserved as u8 as f64,
                )
                .minimum("successful_coefficients", 2.0)
                .minimum("shape_rejected", 1.0)
                .minimum("non_finite_rejected", 1.0)
                .minimum("allocation_rejected", 1.0)
                .minimum("destination_preserved", 1.0)
                .reference("CPU checked flat-basis projection allocation and failure contract")
                .reference_checksum("analytic/deterministic-fixture:compute-strided-projection-v1")
                .dimension(
                    "f64; deterministic projection; typed shape/non-finite/allocation failures",
                )
                .seed("compute-strided-projection-contract:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("strided projection QA result must serialize")
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
                    .expect("strided projection QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn cpu_strided_projection_contract_passes() {
        super::run().expect("CPU strided projection QA contract should pass");
    }
}
