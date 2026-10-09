//! CPU QA for checked rank-N device-view copies.
//!
//! The same descriptor contract is consumed by CUDA and Metal native copy
//! implementations.  This case exercises row-major and column-major
//! metadata, arbitrary logical strides, and destination-preserving rejection
//! paths without starting a simulator workload.

use std::mem::size_of;

use num_complex::{Complex32, Complex64};
use sonium_compute::{Backend, BackendError, CpuBackend, DeviceScalar, ViewAliasing, ViewLayout};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.nd-view-copy-contract.v1";

fn check_scalar<T: DeviceScalar + PartialEq + Copy>(
    backend: &CpuBackend,
    values: &[T],
    sentinel: T,
) -> Result<(usize, usize), String> {
    let mut source = backend
        .raw_alloc::<T>(values.len())
        .map_err(|error| format!("source allocation failed: {error}"))?;
    let mut destination = backend
        .raw_alloc::<T>(values.len())
        .map_err(|error| format!("destination allocation failed: {error}"))?;
    backend
        .try_copy_h2d(values, &mut source)
        .map_err(|error| format!("source upload failed: {error}"))?;
    backend
        .try_copy_h2d(&vec![sentinel; values.len()], &mut destination)
        .map_err(|error| format!("destination upload failed: {error}"))?;

    let element_size = size_of::<T>();
    let source_descriptor = backend
        .checked_nd_view_descriptor_with_layout(
            &source,
            &[2, 3],
            element_size,
            &[3, 1],
            ViewLayout::RowMajor,
            ViewAliasing::ReadOnly,
        )
        .map_err(|error| format!("source descriptor failed: {error}"))?;
    let destination_descriptor = backend
        .checked_nd_view_descriptor_with_layout(
            &destination,
            &[3, 2],
            element_size,
            &[1, 3],
            ViewLayout::ColumnMajor,
            ViewAliasing::Disjoint,
        )
        .map_err(|error| format!("destination descriptor failed: {error}"))?;
    backend
        .try_copy_nd_views_d2d(
            &source,
            &source_descriptor,
            &mut destination,
            &destination_descriptor,
        )
        .map_err(|error| format!("rank-N copy failed: {error}"))?;

    let mut copied = vec![T::zero(); values.len()];
    backend
        .try_copy_d2h(&destination, &mut copied)
        .map_err(|error| format!("rank-N copy download failed: {error}"))?;
    let mut expected = vec![sentinel; values.len()];
    expected[1] = values[1];
    expected[2] = values[3];
    expected[3] = values[5];
    expected[4] = values[2];
    expected[5] = values[4];
    expected[6] = values[6];
    if copied != expected {
        return Err("rank-N copy produced unexpected logical layout".into());
    }

    let before_rejection = copied.clone();
    let read_only_destination = backend
        .checked_nd_view_descriptor_with_layout(
            &destination,
            &[2, 3],
            element_size,
            &[3, 1],
            ViewLayout::RowMajor,
            ViewAliasing::ReadOnly,
        )
        .map_err(|error| format!("read-only descriptor failed: {error}"))?;
    let error = backend
        .try_copy_nd_views_d2d(
            &source,
            &source_descriptor,
            &mut destination,
            &read_only_destination,
        )
        .expect_err("read-only rank-N destination must be rejected");
    if !matches!(error, BackendError::InvalidArgument { .. }) {
        return Err(format!("unexpected read-only rejection: {error}"));
    }
    let mut after_rejection = vec![T::zero(); values.len()];
    backend
        .try_copy_d2h(&destination, &mut after_rejection)
        .map_err(|error| format!("rejection download failed: {error}"))?;
    if after_rejection != before_rejection {
        return Err("rank-N rejection mutated destination".into());
    }

    Ok((1, 1))
}

fn run() -> Result<(usize, usize, usize), String> {
    let backend = CpuBackend::new();
    backend.reset_transfer_stats();
    let mut copy_cases = 0;
    let mut rejection_cases = 0;
    for (copies, rejections) in [
        check_scalar(
            &backend,
            &[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            -1.0,
        )?,
        check_scalar(
            &backend,
            &[0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            -1.0,
        )?,
        check_scalar(
            &backend,
            &[
                Complex32::new(0.0, 0.0),
                Complex32::new(1.0, -1.0),
                Complex32::new(2.0, -2.0),
                Complex32::new(3.0, -3.0),
                Complex32::new(4.0, -4.0),
                Complex32::new(5.0, -5.0),
                Complex32::new(6.0, -6.0),
                Complex32::new(7.0, -7.0),
            ],
            Complex32::new(-1.0, 2.0),
        )?,
        check_scalar(
            &backend,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(2.0, -2.0),
                Complex64::new(3.0, -3.0),
                Complex64::new(4.0, -4.0),
                Complex64::new(5.0, -5.0),
                Complex64::new(6.0, -6.0),
                Complex64::new(7.0, -7.0),
            ],
            Complex64::new(-1.0, 2.0),
        )?,
    ] {
        copy_cases += copies;
        rejection_cases += rejections;
    }
    Ok((
        copy_cases,
        rejection_cases,
        backend.transfer_stats().d2d_operations,
    ))
}

fn main() {
    match run() {
        Ok((copy_cases, rejection_cases, d2d_operations)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("copy_cases", copy_cases as f64)
                .metric("rejection_cases", rejection_cases as f64)
                .metric("d2d_operations", d2d_operations as f64)
                .minimum("copy_cases", 4.0)
                .minimum("rejection_cases", 4.0)
                // CPU intentionally uses the checked host-staged default;
                // CUDA/Metal source-contract tests require their native
                // implementations separately.
                .minimum("d2d_operations", 0.0)
                .input_checksum("analytic:rank-n-view-layout-fixture-v1")
                .reference("independent logical-index row/column-major mapping")
                .reference_checksum(
                    "analytic/deterministic-fixture:compute-nd-view-copy-contract-v1",
                )
                .dimension("four scalar families; rank-2 row/column-major views")
                .formulation("checked descriptor metadata, bounds, dtype/device, aliasing")
                .tolerance("exact selected elements; rejected destination unchanged")
                .seed("compute-nd-view-copy-contract:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("rank-N view-copy QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("rank-N view-copy QA failure must serialize")
            );
            eprintln!("RANK-N VIEW-COPY CONTRACT FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_n_cpu_contract_is_reproducible() {
        let (copy_cases, rejection_cases, d2d_operations) = run().expect("rank-N QA fixture");
        assert_eq!(copy_cases, 4);
        assert_eq!(rejection_cases, 4);
        assert_eq!(d2d_operations, 0);
    }
}
