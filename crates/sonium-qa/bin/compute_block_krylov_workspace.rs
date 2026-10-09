//! CPU QA for caller-owned block-GMRES device scratch reuse.

use sonium_compute::{
    Backend, BlockGmresConfig, BlockGmresWorkspace, CpuBackend, DeviceCsr,
    IdentityDevicePreconditioner, gmres_backend_block_preconditioned_with_workspace,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.block-krylov-workspace.v1";
const MAX_ERROR: f64 = 1.0e-8;

#[derive(Debug)]
struct Metrics {
    first_error: f64,
    second_error: f64,
    stable_buffers: bool,
    stable_capacity: bool,
    stable_host_capacity: bool,
    cleared: bool,
    mismatch_rejected: bool,
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
    let matrix = DeviceCsr::try_from_host(
        &backend,
        2,
        2,
        &[0, 2, 4],
        &[0, 1, 0, 1],
        &[4.0_f64, 1.0, 1.0, 3.0],
    )
    .map_err(|error| format!("fixture upload failed: {error}"))?;
    let mut rhs = backend
        .raw_alloc::<f64>(4)
        .map_err(|error| format!("RHS allocation failed: {error}"))?;
    backend.copy_h2d(&[9.0, 8.0, 3.0, 7.0], &mut rhs);
    let mut output = backend
        .raw_alloc::<f64>(4)
        .map_err(|error| format!("output allocation failed: {error}"))?;
    let config = BlockGmresConfig {
        max_block_iterations: 4,
        tolerance: 1.0e-10,
    };
    let mut workspace = BlockGmresWorkspace::new(2);

    let expected = [19.0 / 11.0, 23.0 / 11.0, 2.0 / 11.0, 25.0 / 11.0];
    let solve = |workspace: &mut BlockGmresWorkspace<CpuBackend, f64>,
                 output: &mut <CpuBackend as Backend>::Buffer<f64>|
     -> Result<f64, String> {
        let status = gmres_backend_block_preconditioned_with_workspace(
            &backend,
            &matrix,
            &IdentityDevicePreconditioner,
            &rhs,
            None,
            output,
            2,
            &config,
            workspace,
        )
        .map_err(|error| format!("block-GMRES failed: {error}"))?;
        if !status.converged {
            return Err(format!("block-GMRES did not converge: {status:?}"));
        }
        let mut actual = [0.0; 4];
        backend.copy_d2h(output, &mut actual);
        Ok(max_abs(&actual, &expected))
    };

    let first_error = solve(&mut workspace, &mut output)?;
    let first_buffers = workspace.device_buffer_count();
    let first_capacity = workspace.device_capacity();
    let first_host_capacity = workspace.host_capacity();
    let second_error = solve(&mut workspace, &mut output)?;
    let stable_buffers = workspace.device_buffer_count() == first_buffers;
    let stable_capacity = workspace.device_capacity() == first_capacity;
    let stable_host_capacity = workspace.host_capacity() == first_host_capacity;

    workspace.clear();
    let cleared = workspace.device_buffer_count() == 0 && workspace.device_capacity() == 0;

    let mut mismatch = BlockGmresWorkspace::new(3);
    let mismatch_rejected = matches!(
        gmres_backend_block_preconditioned_with_workspace(
            &backend,
            &matrix,
            &IdentityDevicePreconditioner,
            &rhs,
            None,
            &mut output,
            2,
            &config,
            &mut mismatch,
        ),
        Err(sonium_compute::BlockGmresError::InvalidArgument(message))
            if message.contains("workspace dimension mismatch")
    );

    if first_error > MAX_ERROR || second_error > MAX_ERROR {
        return Err(format!(
            "solution error exceeds tolerance: first={first_error:.3e}, second={second_error:.3e}"
        ));
    }
    if !stable_buffers || !stable_capacity || !cleared || !mismatch_rejected {
        return Err(format!(
            "workspace lifecycle failed: buffers={stable_buffers}, capacity={stable_capacity}, \
             cleared={cleared}, mismatch={mismatch_rejected}"
        ));
    }

    Ok(Metrics {
        first_error,
        second_error,
        stable_buffers,
        stable_capacity,
        stable_host_capacity,
        cleared,
        mismatch_rejected,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("first_error", metrics.first_error)
                .metric("second_error", metrics.second_error)
                .metric("stable_buffers", metrics.stable_buffers as u8 as f64)
                .metric("stable_capacity", metrics.stable_capacity as u8 as f64)
                .metric(
                    "stable_host_capacity",
                    metrics.stable_host_capacity as u8 as f64,
                )
                .metric("cleared", metrics.cleared as u8 as f64)
                .metric("mismatch_rejected", metrics.mismatch_rejected as u8 as f64)
                .maximum("first_error", MAX_ERROR)
                .maximum("second_error", MAX_ERROR)
                .minimum("stable_buffers", 1.0)
                .minimum("stable_capacity", 1.0)
                .minimum("stable_host_capacity", 1.0)
                .minimum("cleared", 1.0)
                .minimum("mismatch_rejected", 1.0)
                .reference("independent dense 2x2 block-GMRES solution")
                .reference_checksum(
                    "analytic/deterministic-fixture:compute-block-krylov-workspace-v1",
                )
                .dimension("2x2 f64 operator; two RHS; caller-owned workspace")
                .seed("compute-block-krylov-workspace:2x2:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("block-Krylov workspace QA result must serialize")
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
                    .expect("block-Krylov workspace QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
