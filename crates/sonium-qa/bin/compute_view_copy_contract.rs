//! CPU checked strided-view copy contract for `sonium-compute`.
//!
//! The backend trait permits accelerators to replace the default host-staged
//! implementation with a device-native copy.  This contract keeps the shared
//! bounds, stride, aliasing, and failure semantics executable on every CPU
//! runner without starting a simulator workload.

use num_complex::{Complex32, Complex64};
use sonium_compute::{Backend, BackendError, BufferView, CpuBackend, DeviceScalar};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.view-copy-contract.v1";

fn check_scalar<T: DeviceScalar + PartialEq>(
    backend: &CpuBackend,
    values: [T; 8],
    sentinel: T,
) -> Result<(usize, usize), String> {
    let mut source = backend
        .raw_alloc::<T>(values.len())
        .map_err(|error| format!("source allocation failed: {error}"))?;
    let mut destination = backend
        .raw_alloc::<T>(values.len())
        .map_err(|error| format!("destination allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&values, &mut source)
        .map_err(|error| format!("source upload failed: {error}"))?;
    backend
        .try_copy_h2d(&[sentinel; 8], &mut destination)
        .map_err(|error| format!("destination upload failed: {error}"))?;

    let source_view = BufferView::new(8, 1, 3, 2)
        .map_err(|error| format!("source view construction failed: {error}"))?;
    let destination_view = BufferView::new(8, 2, 3, 1)
        .map_err(|error| format!("destination view construction failed: {error}"))?;
    backend
        .try_copy_views_d2d(&source, source_view, &mut destination, destination_view)
        .map_err(|error| format!("strided device copy failed: {error}"))?;

    let mut copied = [T::zero(); 8];
    backend
        .try_copy_d2h(&destination, &mut copied)
        .map_err(|error| format!("destination download failed: {error}"))?;
    let expected = [
        sentinel, sentinel, values[1], values[3], values[5], sentinel, sentinel, sentinel,
    ];
    if copied != expected {
        return Err(format!(
            "view copy mismatch: got {copied:?}, expected {expected:?}"
        ));
    }

    let mut selected = [T::zero(); 3];
    backend
        .try_copy_view_d2h(&source, source_view, &mut selected)
        .map_err(|error| format!("strided download failed: {error}"))?;
    if selected != [values[1], values[3], values[5]] {
        return Err(format!("strided download mismatch: got {selected:?}"));
    }

    let before = copied;
    let error = backend
        .try_copy_views_d2d(
            &source,
            BufferView::new(8, 0, 2, 1).expect("valid source mismatch view"),
            &mut destination,
            BufferView::new(8, 0, 1, 1).expect("valid destination mismatch view"),
        )
        .expect_err("mismatched view lengths must fail");
    if !matches!(error, BackendError::InvalidArgument { .. }) {
        return Err(format!(
            "mismatched views returned unexpected error: {error}"
        ));
    }
    let mut after_rejection = [T::zero(); 8];
    backend
        .try_copy_d2h(&destination, &mut after_rejection)
        .map_err(|error| format!("rejection download failed: {error}"))?;
    if after_rejection != before {
        return Err("mismatched view rejection mutated destination".into());
    }

    Ok((3, 1))
}

fn run() -> Result<(usize, usize, usize), String> {
    let backend = CpuBackend::new();
    backend.reset_transfer_stats();
    let mut view_cases = 0;
    let mut rejection_cases = 0;
    for (cases, rejections) in [
        check_scalar(
            &backend,
            [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            -10.0,
        )?,
        check_scalar(
            &backend,
            [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            -10.0,
        )?,
        check_scalar(
            &backend,
            [
                Complex32::new(0.0, 0.0),
                Complex32::new(1.0, -1.0),
                Complex32::new(2.0, -2.0),
                Complex32::new(3.0, -3.0),
                Complex32::new(4.0, -4.0),
                Complex32::new(5.0, -5.0),
                Complex32::new(6.0, -6.0),
                Complex32::new(7.0, -7.0),
            ],
            Complex32::new(-10.0, 3.0),
        )?,
        check_scalar(
            &backend,
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(2.0, -2.0),
                Complex64::new(3.0, -3.0),
                Complex64::new(4.0, -4.0),
                Complex64::new(5.0, -5.0),
                Complex64::new(6.0, -6.0),
                Complex64::new(7.0, -7.0),
            ],
            Complex64::new(-10.0, 3.0),
        )?,
    ] {
        view_cases += cases;
        rejection_cases += rejections;
    }
    let stats = backend.transfer_stats();
    Ok((view_cases, rejection_cases, stats.d2d_operations))
}

fn main() {
    match run() {
        Ok((view_cases, rejection_cases, d2d_operations)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("view_copy_cases", view_cases as f64)
                .metric("rejection_cases", rejection_cases as f64)
                .metric("native_d2d_operations", d2d_operations as f64)
                .minimum("view_copy_cases", 12.0)
                .minimum("rejection_cases", 4.0)
                .minimum("native_d2d_operations", 4.0)
                .reference("independent strided-view copy and rejection invariants")
                .reference_checksum("analytic/deterministic-fixture:compute-view-copy-contract-v1")
                .dimension("four scalar families; offset/stride views; mismatch rejection")
                .seed("compute-view-copy-contract:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("view-copy QA result must serialize")
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
                    .expect("view-copy QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
