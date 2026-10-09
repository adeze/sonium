//! CPU QA for pooled scalar backend-GMRES workspace reuse.

use sonium_compute::{
    Backend, BackendCsr, CpuBackend, GmresBackendConfig, GmresWorkspace,
    IdentityBackendPreconditioner, MemoryPool, MemoryPoolConfig,
    gmres_backend_preconditioned_pooled_with_workspace,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.gmres-workspace.v1";
const MAX_ERROR: f64 = 1.0e-11;

#[derive(Debug)]
struct Metrics {
    first_error: f64,
    second_error: f64,
    stable_device_buffers: bool,
    stable_host_capacity: bool,
    cleared: bool,
    dimension_rejected: bool,
    transfer_stable: bool,
    steady_h2d_operations: usize,
    steady_d2h_operations: usize,
    steady_d2d_operations: usize,
}

fn max_abs(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<Metrics, String> {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(1 << 20))
        .map_err(|error| format!("pool creation failed: {error}"))?;
    let backend = pool.backend();
    let matrix = BackendCsr::try_from_host(
        backend,
        2,
        2,
        &[0, 2, 4],
        &[0, 1, 0, 1],
        &[4.0, 1.0, 1.0, 3.0],
    )
    .map_err(|error| format!("fixture upload failed: {error}"))?;
    let mut rhs = pool
        .try_alloc::<f64>(2)
        .map_err(|error| format!("RHS allocation failed: {error}"))?;
    let mut output = pool
        .try_alloc::<f64>(2)
        .map_err(|error| format!("output allocation failed: {error}"))?;
    backend.copy_h2d(&[1.0, 2.0], rhs.raw_mut());
    // Exclude one-time fixture setup from the steady-state measurement.
    backend.reset_transfer_stats();

    let expected = [1.0 / 11.0, 7.0 / 11.0];
    let config = GmresBackendConfig {
        max_iterations: 8,
        tolerance: 1.0e-12,
    };
    let mut workspace = GmresWorkspace::new(2);
    let first = gmres_backend_preconditioned_pooled_with_workspace(
        &pool,
        &mut workspace,
        &matrix,
        &IdentityBackendPreconditioner,
        rhs.raw(),
        None,
        output.raw_mut(),
        &config,
    )
    .map_err(|error| format!("first solve failed: {error}"))?;
    if !first.converged {
        return Err(format!("first solve did not converge: {first:?}"));
    }
    let mut first_solution = [0.0; 2];
    backend.copy_d2h(output.raw(), &mut first_solution);
    let first_transfer_stats = backend.transfer_stats();
    backend.reset_transfer_stats();
    let first_error = max_abs(&first_solution, &expected);
    let first_device_buffers = workspace
        .cached_buffer_count()
        .map_err(|error| format!("device-buffer inspection failed: {error}"))?;
    let first_host_capacity = workspace
        .cached_host_capacity()
        .map_err(|error| format!("host-capacity inspection failed: {error}"))?;

    let second = gmres_backend_preconditioned_pooled_with_workspace(
        &pool,
        &mut workspace,
        &matrix,
        &IdentityBackendPreconditioner,
        rhs.raw(),
        None,
        output.raw_mut(),
        &config,
    )
    .map_err(|error| format!("second solve failed: {error}"))?;
    if !second.converged {
        return Err(format!("second solve did not converge: {second:?}"));
    }
    let mut second_solution = [0.0; 2];
    backend.copy_d2h(output.raw(), &mut second_solution);
    let second_transfer_stats = backend.transfer_stats();
    let second_error = max_abs(&second_solution, &expected);
    let stable_device_buffers = first_device_buffers
        == workspace
            .cached_buffer_count()
            .map_err(|error| format!("device-buffer inspection failed: {error}"))?;
    let stable_host_capacity = first_host_capacity
        == workspace
            .cached_host_capacity()
            .map_err(|error| format!("host-capacity inspection failed: {error}"))?;

    workspace
        .clear()
        .map_err(|error| format!("workspace clear failed: {error}"))?;
    let cleared = workspace
        .cached_buffer_count()
        .map_err(|error| format!("device-buffer inspection failed: {error}"))?
        == 0
        && workspace
            .cached_host_capacity()
            .map_err(|error| format!("host-capacity inspection failed: {error}"))?
            == 0;

    let mut wrong_workspace = GmresWorkspace::new(3);
    let dimension_rejected = matches!(
        gmres_backend_preconditioned_pooled_with_workspace(
            &pool,
            &mut wrong_workspace,
            &matrix,
            &IdentityBackendPreconditioner,
            rhs.raw(),
            None,
            output.raw_mut(),
            &config,
        ),
        Err(sonium_compute::GmresBackendError::Workspace { .. })
    );

    Ok(Metrics {
        first_error,
        second_error,
        stable_device_buffers,
        stable_host_capacity,
        cleared,
        dimension_rejected,
        transfer_stable: first_transfer_stats == second_transfer_stats,
        steady_h2d_operations: second_transfer_stats.h2d_operations,
        steady_d2h_operations: second_transfer_stats.d2h_operations,
        steady_d2d_operations: second_transfer_stats.d2d_operations,
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
                .metric(
                    "stable_device_buffers",
                    metrics.stable_device_buffers as u8 as f64,
                )
                .metric(
                    "stable_host_capacity",
                    metrics.stable_host_capacity as u8 as f64,
                )
                .metric("cleared", metrics.cleared as u8 as f64)
                .metric(
                    "dimension_rejected",
                    metrics.dimension_rejected as u8 as f64,
                )
                .metric("transfer_stable", metrics.transfer_stable as u8 as f64)
                .metric(
                    "steady_h2d_operations",
                    metrics.steady_h2d_operations as f64,
                )
                .metric(
                    "steady_d2h_operations",
                    metrics.steady_d2h_operations as f64,
                )
                .metric(
                    "steady_d2d_operations",
                    metrics.steady_d2d_operations as f64,
                )
                .maximum("first_error", MAX_ERROR)
                .maximum("second_error", MAX_ERROR)
                .minimum("stable_device_buffers", 1.0)
                .minimum("stable_host_capacity", 1.0)
                .minimum("cleared", 1.0)
                .minimum("dimension_rejected", 1.0)
                .minimum("transfer_stable", 1.0)
                .reference("independent analytic solution of a 2x2 SPD system")
                .reference_checksum("analytic/deterministic-fixture:compute-gmres-workspace-v1")
                .dimension("2x2 f64 CSR; pooled scalar GMRES; caller-owned workspace")
                .seed("compute-gmres-workspace:2x2:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("GMRES workspace QA result must serialize")
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
                    .expect("GMRES workspace QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
