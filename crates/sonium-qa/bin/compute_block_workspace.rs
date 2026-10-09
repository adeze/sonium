//! Reusable block-operator staging contract.
//!
//! This is a bounded CPU QA case for the checked `DeviceBlockWorkspace` API.
//! It verifies independent scalar results, single- and multi-RHS dispatch,
//! warm-up capacity stability, explicit clear/rebuild semantics, and typed
//! rejection of a workspace created for another layout.

use sonium_compute::{Backend, BlockLayout, CpuBackend, DeviceBlockOperator, DeviceCsr};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.block-workspace.v1";
const MAX_ERROR: f64 = 1.0e-12;

#[derive(Debug)]
struct Metrics {
    single_error: f64,
    multi_error: f64,
    warm_capacity_stable: bool,
    warm_buffers_stable: bool,
    cleared_buffers: bool,
    layout_rejected: bool,
}

fn max_abs(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<Metrics, String> {
    let backend = CpuBackend::new();
    let layout = BlockLayout::try_new(vec![2, 1], vec![2, 1])
        .map_err(|error| format!("layout admission failed: {error}"))?;
    let a00 = DeviceCsr::try_from_host(
        &backend,
        2,
        2,
        &[0, 2, 4],
        &[0, 1, 0, 1],
        &[1.0, 2.0, 3.0, 4.0],
    )
    .map_err(|error| format!("A00 upload failed: {error}"))?;
    let a01 = DeviceCsr::try_from_host(&backend, 2, 1, &[0, 1, 2], &[0, 0], &[5.0, 6.0])
        .map_err(|error| format!("A01 upload failed: {error}"))?;
    let a10 = DeviceCsr::try_from_host(&backend, 1, 2, &[0, 2], &[0, 1], &[7.0, 8.0])
        .map_err(|error| format!("A10 upload failed: {error}"))?;
    let operator = DeviceBlockOperator::try_new(layout.clone())
        .map_err(|error| format!("operator admission failed: {error}"))?
        .try_with_block(0, 0, Box::new(a00))
        .map_err(|error| format!("A00 block insertion failed: {error}"))?
        .try_with_block(0, 1, Box::new(a01))
        .map_err(|error| format!("A01 block insertion failed: {error}"))?
        .try_with_block(1, 0, Box::new(a10))
        .map_err(|error| format!("A10 block insertion failed: {error}"))?;

    let mut x = backend
        .raw_alloc::<f64>(3)
        .map_err(|error| format!("input allocation failed: {error}"))?;
    let mut y = backend
        .raw_alloc::<f64>(3)
        .map_err(|error| format!("output allocation failed: {error}"))?;
    backend.copy_h2d(&[2.0, 3.0, 4.0], &mut x);
    let mut workspace = operator.new_workspace();
    operator
        .try_apply_with_workspace(&backend, &x, &mut y, &mut workspace)
        .map_err(|error| format!("first checked workspace apply failed: {error}"))?;
    let mut actual = [0.0; 3];
    backend.copy_d2h(&y, &mut actual);
    let single_error = max_abs(&actual, &[28.0, 42.0, 38.0]);
    let first_capacity = workspace.host_capacity();
    let first_buffers = workspace.device_buffer_count();

    backend.copy_h2d(&[1.0, -1.0, 2.0], &mut x);
    operator
        .try_apply_with_workspace(&backend, &x, &mut y, &mut workspace)
        .map_err(|error| format!("warm checked workspace apply failed: {error}"))?;
    backend.copy_d2h(&y, &mut actual);
    let warm_capacity_stable = workspace.host_capacity() == first_capacity;
    let warm_buffers_stable = workspace.device_buffer_count() == first_buffers;
    let single_error = single_error.max(max_abs(&actual, &[9.0, 11.0, -1.0]));

    let mut x_multi = backend
        .raw_alloc::<f64>(6)
        .map_err(|error| format!("multi-RHS input allocation failed: {error}"))?;
    let mut y_multi = backend
        .raw_alloc::<f64>(6)
        .map_err(|error| format!("multi-RHS output allocation failed: {error}"))?;
    backend.copy_h2d(&[2.0, 3.0, 4.0, 1.0, -1.0, 2.0], &mut x_multi);
    operator
        .try_apply_colmajor_with_workspace(&backend, 2, &x_multi, &mut y_multi, &mut workspace)
        .map_err(|error| format!("multi-RHS workspace apply failed: {error}"))?;
    let mut actual_multi = [0.0; 6];
    backend.copy_d2h(&y_multi, &mut actual_multi);
    let multi_error = max_abs(&actual_multi, &[28.0, 42.0, 38.0, 9.0, 11.0, -1.0]);

    workspace.clear();
    let cleared_buffers = workspace.device_buffer_count() == 0;
    operator
        .try_apply_with_workspace(&backend, &x, &mut y, &mut workspace)
        .map_err(|error| format!("post-clear workspace apply failed: {error}"))?;

    let other_layout = BlockLayout::try_new(vec![1], vec![1])
        .map_err(|error| format!("other layout admission failed: {error}"))?;
    let mut other_workspace = sonium_compute::DeviceBlockWorkspace::new(&other_layout);
    let layout_rejected = matches!(
        operator.try_apply_with_workspace(&backend, &x, &mut y, &mut other_workspace),
        Err(sonium_compute::BackendError::InvalidArgument {
            operation: "device_block_workspace_layout",
            ..
        })
    );

    if single_error > MAX_ERROR || multi_error > MAX_ERROR {
        return Err(format!(
            "workspace result error exceeds tolerance: single={single_error:.3e}, multi={multi_error:.3e}"
        ));
    }
    if !warm_capacity_stable || !warm_buffers_stable || !cleared_buffers || !layout_rejected {
        return Err(format!(
            "workspace lifecycle contract failed: capacity={warm_capacity_stable}, buffers={warm_buffers_stable}, clear={cleared_buffers}, layout={layout_rejected}"
        ));
    }
    Ok(Metrics {
        single_error,
        multi_error,
        warm_capacity_stable,
        warm_buffers_stable,
        cleared_buffers,
        layout_rejected,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("single_error", metrics.single_error)
                .metric("multi_error", metrics.multi_error)
                .metric(
                    "warm_capacity_stable",
                    metrics.warm_capacity_stable as u8 as f64,
                )
                .metric(
                    "warm_buffers_stable",
                    metrics.warm_buffers_stable as u8 as f64,
                )
                .metric("cleared_buffers", metrics.cleared_buffers as u8 as f64)
                .metric("layout_rejected", metrics.layout_rejected as u8 as f64)
                .maximum("single_error", MAX_ERROR)
                .maximum("multi_error", MAX_ERROR)
                .minimum("warm_capacity_stable", 1.0)
                .minimum("warm_buffers_stable", 1.0)
                .minimum("cleared_buffers", 1.0)
                .minimum("layout_rejected", 1.0)
                .reference("independent scalar block-matrix multiplication")
                .reference_checksum("analytic/deterministic-fixture:compute-block-workspace-v1")
                .dimension("2x2 block layout; f64; single and two-RHS column-major paths")
                .seed("compute-block-workspace:2x2-block-matrix:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("block workspace QA result must serialize")
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
                    .expect("block workspace QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
