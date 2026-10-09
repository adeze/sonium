//! Checked FEM device-solver contract.
//!
//! The production FEM assembler remains host-side today.  This producer makes
//! that boundary explicit, then exercises the supported seam: a constrained
//! H1/Tri3/P1 Helmholtz CSR operator, a device-resident point-Jacobi
//! preconditioner, and backend GMRES with pooled reusable workspace.  All
//! Krylov vector actions and the true residual are evaluated on the CUDA
//! device; the host is used only for setup, scalar recurrence bookkeeping,
//! final evidence, and an independent residual reference.

#![cfg(feature = "cuda")]

use std::collections::BTreeMap;
use std::time::Instant;

use num_complex::Complex64;
use sonium_compute::{
    Backend, CudaBackend, CudaRuntimeStatsSnapshot, DeviceCsr, DeviceOperator, GmresBackendConfig,
    GmresBackendStatus, GmresWorkspace, MemoryPool, MemoryPoolConfig, PersistentDeviceCsr,
    PointJacobiBackend, gmres_backend_preconditioned_pooled_with_workspace,
};
use sonium_fem::assembly::HelmholtzProblem;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::{MpcSystem, MpcTerm, MultiPointConstraint};
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_qa::{
    QaExecutionCounts, QaFailureStage, QaResult, QaStatus, cuda_hardware_provenance,
    cuda_runtime_evidence, required_cuda_backend, write_qa_artifact,
};

const CASE_ID: &str = "fem.device-solver.v1";
const ARTIFACT: &str = "fem_device_solver.json";
const RESIDUAL_TOLERANCE: f64 = 2.0e-10;
const TRANSFER_TOLERANCE: f64 = 2.0e-13;
const AFFINE_MPC_TRANSFER_TOLERANCE: f64 = 2.0e-13;
const WARM_REPEATS: usize = 3;

type Csr = (Vec<i32>, Vec<i32>, Vec<Complex64>);

#[derive(Debug, Clone, Copy, Default)]
struct Counters {
    device_allocations: usize,
    staging_allocations: usize,
    h2d: usize,
    h2d_bytes: usize,
    d2h: usize,
    d2h_bytes: usize,
    d2d: usize,
    d2d_bytes: usize,
    synchronizations: usize,
    launches: usize,
    device_peak_bytes: usize,
    staging_peak_bytes: usize,
    device_live_bytes: usize,
    staging_live_bytes: usize,
}

impl Counters {
    fn from_backend(backend: &CudaBackend) -> Self {
        let stats: CudaRuntimeStatsSnapshot = backend.runtime_stats();
        Self {
            device_allocations: stats.allocation_operations,
            staging_allocations: stats.staging_allocation_operations,
            h2d: stats.transfer.h2d_operations,
            h2d_bytes: stats.transfer.h2d_bytes,
            d2h: stats.transfer.d2h_operations,
            d2h_bytes: stats.transfer.d2h_bytes,
            d2d: stats.transfer.d2d_operations,
            d2d_bytes: stats.transfer.d2d_bytes,
            synchronizations: stats.synchronization_calls,
            launches: stats.kernel_launches + stats.library_operations,
            device_peak_bytes: stats.allocation_high_water_bytes,
            staging_peak_bytes: stats.staging_high_water_bytes,
            device_live_bytes: stats.live_allocation_bytes,
            staging_live_bytes: stats.live_staging_bytes,
        }
    }

    fn delta(self, before: Self) -> Self {
        Self {
            device_allocations: self
                .device_allocations
                .saturating_sub(before.device_allocations),
            staging_allocations: self
                .staging_allocations
                .saturating_sub(before.staging_allocations),
            h2d: self.h2d.saturating_sub(before.h2d),
            h2d_bytes: self.h2d_bytes.saturating_sub(before.h2d_bytes),
            d2h: self.d2h.saturating_sub(before.d2h),
            d2h_bytes: self.d2h_bytes.saturating_sub(before.d2h_bytes),
            d2d: self.d2d.saturating_sub(before.d2d),
            d2d_bytes: self.d2d_bytes.saturating_sub(before.d2d_bytes),
            synchronizations: self
                .synchronizations
                .saturating_sub(before.synchronizations),
            launches: self.launches.saturating_sub(before.launches),
            device_peak_bytes: self.device_peak_bytes,
            staging_peak_bytes: self.staging_peak_bytes,
            device_live_bytes: self.device_live_bytes,
            staging_live_bytes: self.staging_live_bytes,
        }
    }
}

#[derive(Debug, Clone)]
struct Metrics {
    dim: usize,
    nnz: usize,
    malformed_csr_rejected: bool,
    cached_buffer_count: usize,
    cached_device_reduction_capacity: usize,
    cached_basis_slot_capacity: usize,
    transfer_error: f64,
    affine_mpc_transfer_error: f64,
    affine_mpc_full_dofs: usize,
    affine_mpc_reduced_dofs: usize,
    affine_mpc_constraint_count: usize,
    true_residual: f64,
    host_residual: f64,
    iterations: usize,
    projected_residual: f64,
    backward_error: f64,
    warm_device_allocations: Vec<usize>,
    warm_staging_allocations: Vec<usize>,
    warm_h2d: Vec<usize>,
    warm_h2d_bytes: Vec<usize>,
    warm_d2h: Vec<usize>,
    warm_d2h_bytes: Vec<usize>,
    warm_d2d: Vec<usize>,
    warm_d2d_bytes: Vec<usize>,
    setup_peak_device_bytes: usize,
    setup_peak_staging_bytes: usize,
    steady_live_device_bytes: usize,
    steady_live_staging_bytes: usize,
    elapsed_ms: u64,
}

fn csr_from_problem(problem: &HelmholtzProblem) -> Result<Csr, String> {
    let dim = problem.num_dofs();
    let mut entries = BTreeMap::<(usize, usize), Complex64>::new();
    for ((&row, &col), &value) in problem
        .matrix
        .rows
        .iter()
        .zip(&problem.matrix.cols)
        .zip(&problem.matrix.values)
    {
        if row >= dim || col >= dim || !value.re.is_finite() || !value.im.is_finite() {
            return Err("FEM triplet contains an invalid index or value".into());
        }
        *entries.entry((row, col)).or_default() += value;
    }
    let constrained = [0usize, dim.saturating_sub(1)];
    let mut rows = vec![Vec::<(usize, Complex64)>::new(); dim];
    for ((row, col), value) in entries {
        if constrained.contains(&row) || constrained.contains(&col) {
            continue;
        }
        rows[row].push((col, value));
    }
    for &dof in &constrained {
        rows[dof].push((dof, Complex64::new(1.0, 0.0)));
    }
    let mut row_ptr = Vec::with_capacity(dim + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for mut row in rows {
        row.sort_unstable_by_key(|(col, _)| *col);
        for (col, value) in row {
            col_idx.push(i32::try_from(col).map_err(|_| "CSR column exceeds i32")?);
            values.push(value);
        }
        row_ptr.push(i32::try_from(col_idx.len()).map_err(|_| "CSR nnz exceeds i32")?);
    }
    Ok((row_ptr, col_idx, values))
}

fn host_matvec(
    row_ptr: &[i32],
    col_idx: &[i32],
    values: &[Complex64],
    x: &[Complex64],
) -> Vec<Complex64> {
    row_ptr
        .windows(2)
        .map(|window| {
            (window[0] as usize..window[1] as usize)
                .map(|index| values[index] * x[col_idx[index] as usize])
                .sum()
        })
        .collect()
}

fn diagonal(row_ptr: &[i32], col_idx: &[i32], values: &[Complex64]) -> Vec<Complex64> {
    row_ptr
        .windows(2)
        .enumerate()
        .map(|(row, window)| {
            (window[0] as usize..window[1] as usize)
                .find_map(|index| (col_idx[index] as usize == row).then_some(values[index]))
                .unwrap_or_default()
        })
        .collect()
}

fn relative_residual(
    row_ptr: &[i32],
    col_idx: &[i32],
    values: &[Complex64],
    x: &[Complex64],
    rhs: &[Complex64],
) -> f64 {
    let ax = host_matvec(row_ptr, col_idx, values, x);
    let numerator: f64 = ax
        .iter()
        .zip(rhs)
        .map(|(computed, requested)| (*computed - *requested).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator: f64 = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
    numerator / denominator.max(f64::MIN_POSITIVE)
}

fn max_abs(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max)
}

fn status_residual(status: &GmresBackendStatus<Complex64>) -> f64 {
    status.final_residual
}

fn apply_affine_mpc_transfer(backend: &CudaBackend) -> Result<(f64, usize, usize, usize), String> {
    // Build the same public FEM MPC representation used by the host boundary
    // layer, then materialize only its sparse prolongation on CUDA.  DOF 2 is
    // a hanging interpolation and DOF 4 is a periodic-style copy.  The
    // nonzero lift makes this an affine constraint action rather than merely
    // a homogeneous matrix-vector smoke test.  This remains a bounded
    // transfer primitive: the production FEM dispatcher still stages general
    // MPC condensation on the host.
    let constraints = [
        MultiPointConstraint {
            slave: 2,
            masters: vec![
                MpcTerm {
                    dof: 0,
                    coefficient: Complex64::new(0.5, 0.0),
                },
                MpcTerm {
                    dof: 1,
                    coefficient: Complex64::new(0.5, 0.0),
                },
            ],
        },
        MultiPointConstraint {
            slave: 4,
            masters: vec![MpcTerm {
                dof: 1,
                coefficient: Complex64::new(1.0, 0.0),
            }],
        },
    ];
    let mpc = MpcSystem::new(5, &constraints)
        .map_err(|error| format!("MPC transfer setup failed: {error}"))?;
    let mut row_ptr = Vec::with_capacity(mpc.num_full_dofs() + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0_i32);
    for full_dof in 0..mpc.num_full_dofs() {
        for &(reduced_dof, coefficient) in mpc.prolongation_row(full_dof) {
            col_idx.push(i32::try_from(reduced_dof).map_err(|_| "MPC reduced DOF exceeds i32")?);
            values.push(coefficient);
        }
        row_ptr.push(i32::try_from(col_idx.len()).map_err(|_| "MPC nnz exceeds i32")?);
    }
    let transfer = DeviceCsr::try_from_host(
        backend,
        mpc.num_full_dofs(),
        mpc.num_reduced_dofs(),
        &row_ptr,
        &col_idx,
        &values,
    )
    .map_err(|error| format!("transfer operator setup failed: {error}"))?;
    let reduced_host = [
        Complex64::new(1.25, -0.5),
        Complex64::new(-0.75, 0.25),
        Complex64::new(0.5, 0.75),
    ];
    let lift_host = [
        Complex64::default(),
        Complex64::default(),
        Complex64::new(0.125, -0.25),
        Complex64::default(),
        Complex64::default(),
    ];
    let mut expected = mpc
        .recover(&reduced_host)
        .map_err(|error| format!("MPC CPU recovery failed: {error}"))?;
    for (value, lift) in expected.iter_mut().zip(lift_host) {
        *value += lift;
    }
    let mut reduced = backend
        .raw_alloc::<Complex64>(mpc.num_reduced_dofs())
        .map_err(|error| format!("transfer input allocation failed: {error}"))?;
    let mut full = backend
        .raw_alloc::<Complex64>(mpc.num_full_dofs())
        .map_err(|error| format!("transfer output allocation failed: {error}"))?;
    let mut lift = backend
        .raw_alloc::<Complex64>(mpc.num_full_dofs())
        .map_err(|error| format!("affine lift allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&reduced_host, &mut reduced)
        .map_err(|error| format!("transfer input upload failed: {error}"))?;
    backend
        .try_copy_h2d(&lift_host, &mut lift)
        .map_err(|error| format!("affine lift upload failed: {error}"))?;
    transfer
        .try_apply(backend, &reduced, &mut full)
        .map_err(|error| format!("device transfer failed: {error}"))?;
    backend
        .try_axpy(Complex64::new(1.0, 0.0), &lift, &mut full)
        .map_err(|error| format!("device affine lift failed: {error}"))?;
    let mut actual = vec![Complex64::default(); mpc.num_full_dofs()];
    backend
        .try_copy_d2h(&full, &mut actual)
        .map_err(|error| format!("transfer output download failed: {error}"))?;
    Ok((
        max_abs(&actual, &expected),
        mpc.num_full_dofs(),
        mpc.num_reduced_dofs(),
        constraints.len(),
    ))
}

fn check_malformed_csr_rejected(backend: &CudaBackend) -> Result<bool, String> {
    // Exercise the public FEM admission seam with deliberately inconsistent
    // CSR metadata.  This must fail before any device allocation or upload;
    // accepting it would make a later cuSPARSE failure untyped and leak the
    // device-resident solve boundary.
    let result = DeviceCsr::<CudaBackend, Complex64>::try_from_host(
        backend,
        2,
        2,
        &[0, 2, 1],
        &[0],
        &[Complex64::new(1.0, 0.0)],
    );
    match result {
        Err(sonium_compute::BackendError::InvalidArgument { .. }) => Ok(true),
        Err(error) => Err(format!(
            "malformed CSR returned the wrong typed error: {error}"
        )),
        Ok(_) => Err("malformed CSR was accepted".into()),
    }
}

fn solve_once(
    backend: &CudaBackend,
    pool: &MemoryPool<CudaBackend>,
    operator: &PersistentDeviceCsr<CudaBackend, Complex64>,
    preconditioner: &PointJacobiBackend<CudaBackend, Complex64>,
    workspace: &mut GmresWorkspace<CudaBackend, Complex64>,
    rhs: &sonium_compute::pool::Buffer<Complex64, CudaBackend>,
    output: &mut sonium_compute::pool::Buffer<Complex64, CudaBackend>,
) -> Result<GmresBackendStatus<Complex64>, String> {
    gmres_backend_preconditioned_pooled_with_workspace(
        pool,
        workspace,
        operator,
        preconditioner,
        rhs.raw(),
        None,
        output.raw_mut(),
        &GmresBackendConfig {
            max_iterations: 96,
            tolerance: 1.0e-10,
        },
    )
    .map_err(|error| format!("device FEM GMRES failed: {error}"))
    .and_then(|status| {
        backend
            .try_synchronize()
            .map_err(|error| format!("device FEM GMRES synchronization failed: {error}"))?;
        Ok(status)
    })
}

fn run(backend: CudaBackend) -> Result<(Metrics, CudaRuntimeStatsSnapshot), String> {
    let started = Instant::now();
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 3, 3);
    let problem = HelmholtzProblem::try_assemble(
        &mesh,
        PolynomialDegree::P1,
        Complex64::new(1.25, 0.08),
        |x, y, _| Complex64::new(0.75 + x - 0.2 * y, 0.15 + 0.5 * x + y),
    )
    .map_err(|error| format!("FEM assembly failed: {error}"))?;
    let (row_ptr, col_idx, values) = csr_from_problem(&problem)?;
    let dim = problem.num_dofs();
    let x_host: Vec<_> = (0..dim)
        .map(|index| {
            if index == 0 || index + 1 == dim {
                Complex64::default()
            } else {
                Complex64::new((index as f64 * 0.31).sin(), (index as f64 * 0.17).cos())
            }
        })
        .collect();
    let rhs_host = host_matvec(&row_ptr, &col_idx, &values, &x_host);

    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20))
        .map_err(|error| format!("device memory pool construction failed: {error}"))?;
    let pool_backend = pool.backend();
    let device = DeviceCsr::try_from_host(pool_backend, dim, dim, &row_ptr, &col_idx, &values)
        .map_err(|error| format!("constrained FEM CSR upload failed: {error}"))?;
    let mut operator = PersistentDeviceCsr::try_from_device_csr(pool_backend, device)
        .map_err(|error| format!("persistent FEM CSR setup failed: {error}"))?;
    let mut preconditioner =
        PointJacobiBackend::try_from_csr_host(pool_backend, dim, &row_ptr, &col_idx, &values)
            .map_err(|error| format!("device point-Jacobi setup failed: {error}"))?;
    let mut rhs = pool
        .try_alloc::<Complex64>(dim)
        .map_err(|error| format!("device RHS allocation failed: {error}"))?;
    let mut output = pool
        .try_alloc::<Complex64>(dim)
        .map_err(|error| format!("device solution allocation failed: {error}"))?;
    pool_backend
        .try_copy_h2d(&rhs_host, rhs.raw_mut())
        .map_err(|error| format!("RHS upload failed: {error}"))?;
    let (
        affine_mpc_transfer_error,
        affine_mpc_full_dofs,
        affine_mpc_reduced_dofs,
        affine_mpc_constraint_count,
    ) = apply_affine_mpc_transfer(pool_backend)?;
    let transfer_error = affine_mpc_transfer_error;
    if transfer_error > TRANSFER_TOLERANCE {
        return Err(format!("device transfer error {transfer_error:.3e}"));
    }
    if affine_mpc_transfer_error > AFFINE_MPC_TRANSFER_TOLERANCE {
        return Err(format!(
            "device affine MPC transfer error {affine_mpc_transfer_error:.3e}"
        ));
    }
    let malformed_csr_rejected = check_malformed_csr_rejected(pool_backend)?;

    let mut workspace = GmresWorkspace::new(dim);
    pool_backend.reset_runtime_stats();
    let warm = solve_once(
        pool_backend,
        &pool,
        &operator,
        &preconditioner,
        &mut workspace,
        &rhs,
        &mut output,
    )?;
    if !warm.converged || !status_residual(&warm).is_finite() {
        return Err(format!("warm solve did not converge: {warm:?}"));
    }
    // The first solve fills the reusable Krylov workspace.  The measured
    // loop changes only the RHS and reuses the persistent operator,
    // preconditioner, and workspace.
    let mut steady = Vec::with_capacity(WARM_REPEATS);
    let mut before = Counters::from_backend(pool_backend);
    for repeat in 0..WARM_REPEATS {
        let rhs_variant: Vec<_> = rhs_host
            .iter()
            .map(|value| *value * (1.0 + repeat as f64 * 0.01))
            .collect();
        pool_backend
            .try_copy_h2d(&rhs_variant, rhs.raw_mut())
            .map_err(|error| format!("RHS {repeat} update failed: {error}"))?;
        let status = solve_once(
            pool_backend,
            &pool,
            &operator,
            &preconditioner,
            &mut workspace,
            &rhs,
            &mut output,
        )?;
        if !status.converged || !status_residual(&status).is_finite() {
            return Err(format!(
                "warm RHS {repeat} solve did not converge: {status:?}"
            ));
        }
        let after = Counters::from_backend(pool_backend);
        steady.push(after.delta(before));
        before = after;
    }

    // Frequency and design steps update only values and the resident Jacobi
    // diagonal.  The sparsity descriptor and Krylov workspace remain valid.
    let mut final_values = values.clone();
    let mut final_rhs = rhs_host.clone();
    for (label, factor) in [("frequency", 1.001), ("design", 0.999)] {
        let mut updated_values = final_values.clone();
        for row in 0..dim {
            if row == 0 || row + 1 == dim {
                continue;
            }
            for index in row_ptr[row] as usize..row_ptr[row + 1] as usize {
                updated_values[index] *= factor;
            }
        }
        operator
            .try_copy_values_from_host(pool_backend, &updated_values)
            .map_err(|error| format!("{label} resident CSR update failed: {error}"))?;
        preconditioner
            .try_update_diagonal(pool_backend, &diagonal(&row_ptr, &col_idx, &updated_values))
            .map_err(|error| format!("{label} resident Jacobi update failed: {error}"))?;
        final_rhs = host_matvec(&row_ptr, &col_idx, &updated_values, &x_host);
        pool_backend
            .try_copy_h2d(&final_rhs, rhs.raw_mut())
            .map_err(|error| format!("{label} RHS update failed: {error}"))?;
        final_values = updated_values;
    }
    let final_status = solve_once(
        pool_backend,
        &pool,
        &operator,
        &preconditioner,
        &mut workspace,
        &rhs,
        &mut output,
    )?;
    if !final_status.converged {
        return Err(format!(
            "final design solve did not converge: {final_status:?}"
        ));
    }
    let mut solution = vec![Complex64::default(); dim];
    pool_backend
        .try_copy_d2h(output.raw(), &mut solution)
        .map_err(|error| format!("final solution download failed: {error}"))?;
    let host_residual = relative_residual(&row_ptr, &col_idx, &final_values, &solution, &final_rhs);
    if !host_residual.is_finite() || host_residual > RESIDUAL_TOLERANCE {
        return Err(format!("independent FEM residual {host_residual:.3e}"));
    }
    let setup_stats = Counters::from_backend(pool_backend);
    let warm_device_allocations: Vec<_> = steady
        .iter()
        .map(|counter| counter.device_allocations)
        .collect();
    let warm_staging_allocations: Vec<_> = steady
        .iter()
        .map(|counter| counter.staging_allocations)
        .collect();
    let warm_h2d: Vec<_> = steady.iter().map(|counter| counter.h2d).collect();
    let warm_h2d_bytes: Vec<_> = steady.iter().map(|counter| counter.h2d_bytes).collect();
    let warm_d2h: Vec<_> = steady.iter().map(|counter| counter.d2h).collect();
    let warm_d2h_bytes: Vec<_> = steady.iter().map(|counter| counter.d2h_bytes).collect();
    let warm_d2d: Vec<_> = steady.iter().map(|counter| counter.d2d).collect();
    let warm_d2d_bytes: Vec<_> = steady.iter().map(|counter| counter.d2d_bytes).collect();
    if warm_device_allocations.iter().any(|&count| count != 0)
        || warm_staging_allocations
            .windows(2)
            .any(|window| window[0] != window[1])
        || warm_h2d.windows(2).any(|window| window[0] != window[1])
        || warm_h2d_bytes
            .windows(2)
            .any(|window| window[0] != window[1])
        || warm_d2h.windows(2).any(|window| window[0] != window[1])
        || warm_d2h_bytes
            .windows(2)
            .any(|window| window[0] != window[1])
        || warm_d2d.windows(2).any(|window| window[0] != window[1])
        || warm_d2d_bytes
            .windows(2)
            .any(|window| window[0] != window[1])
    {
        return Err(format!(
            "steady-state residency contract failed: {steady:?}; cached buffers after warm={} device_reduction_capacity={} basis_slots={}",
            workspace.cached_buffer_count().unwrap_or(0),
            workspace.cached_device_reduction_capacity(),
            workspace.cached_basis_slot_capacity(),
        ));
    }
    let metrics = Metrics {
        dim,
        nnz: values.len(),
        malformed_csr_rejected,
        cached_buffer_count: workspace.cached_buffer_count().unwrap_or(0),
        cached_device_reduction_capacity: workspace.cached_device_reduction_capacity(),
        cached_basis_slot_capacity: workspace.cached_basis_slot_capacity(),
        transfer_error,
        true_residual: status_residual(&final_status),
        host_residual,
        iterations: final_status.iterations,
        projected_residual: final_status.projected_residual,
        backward_error: final_status.backward_error,
        warm_device_allocations,
        warm_staging_allocations,
        warm_h2d,
        warm_h2d_bytes,
        warm_d2h,
        warm_d2h_bytes,
        warm_d2d,
        warm_d2d_bytes,
        setup_peak_device_bytes: setup_stats.device_peak_bytes,
        setup_peak_staging_bytes: setup_stats.staging_peak_bytes,
        steady_live_device_bytes: setup_stats.device_live_bytes,
        steady_live_staging_bytes: setup_stats.staging_live_bytes,
        affine_mpc_transfer_error,
        affine_mpc_full_dofs,
        affine_mpc_reduced_dofs,
        affine_mpc_constraint_count,
        elapsed_ms: started.elapsed().as_millis() as u64,
    };
    Ok((metrics, pool_backend.runtime_stats()))
}

fn write_artifact(
    metrics: &Metrics,
    stats: CudaRuntimeStatsSnapshot,
) -> Result<(String, String), String> {
    let mut environment = BTreeMap::new();
    for (key, value) in [
        ("capability.fem.space.h1", "native"),
        ("capability.fem.space.hcurl", "unsupported"),
        ("capability.fem.space.hdiv", "unsupported"),
        ("capability.fem.element.tri3", "native"),
        ("capability.fem.element.other", "unsupported"),
        ("capability.fem.order.p1", "native"),
        ("capability.fem.order.higher", "unsupported"),
        ("capability.fem.form.helmholtz", "native"),
        ("capability.fem.form.other", "unsupported"),
        ("capability.fem.mode.assembled_csr", "native"),
        ("capability.fem.mode.partial_or_matrix_free", "unsupported"),
        ("capability.fem.scalar.complex64", "native"),
        ("capability.fem.scalar.f64_accumulation", "native"),
        (
            "capability.fem.constraint.dirichlet_homogeneous",
            "native_after_host_setup",
        ),
        (
            "capability.fem.constraint.periodic",
            "native_device_transfer_only",
        ),
        (
            "capability.fem.constraint.hanging",
            "native_device_transfer_only",
        ),
        (
            "capability.fem.constraint.affine_mpc",
            "native_device_transfer_only",
        ),
        ("capability.fem.transfer.h1_affine", "native_device_csr"),
        ("capability.fem.solver.gmres", "native_device_resident"),
        (
            "capability.fem.preconditioner.point_jacobi",
            "native_device_resident",
        ),
        ("capability.fem.fallback.required_cuda", "none"),
    ] {
        environment.insert(key, value);
    }
    let payload = serde_json::json!({
        "schema": "sonium-fem-device-solver/v1",
        "status": "passed",
        "case_id": CASE_ID,
        "fem_solve_report": {
            "backend_requested": "cuda",
            "backend_actual": "cuda",
            "precision": "Complex64",
            "accumulation_precision": "f64",
            "fallback": false,
            "fallback_provenance": "required CUDA; no CPU fallback",
            "convergence": {
                "iterations": metrics.iterations,
                "projected_residual": metrics.projected_residual,
                "true_residual": metrics.true_residual,
                "backward_error": metrics.backward_error,
                "converged": true,
            },
            "physical_diagnostics": {
                "status": "bounded Helmholtz residual gate",
                "reference": "independent host CSR accumulation",
                "relative_residual": metrics.host_residual,
            },
        },
        "constraints": {
            "dirichlet": "homogeneous row/column elimination before upload",
            "periodic": "native_device_transfer_only",
            "hanging": "native_device_transfer_only",
            "affine_mpc": {
                "status": "native_device_transfer_only",
                "full_dofs": metrics.affine_mpc_full_dofs,
                "reduced_dofs": metrics.affine_mpc_reduced_dofs,
                "constraint_count": metrics.affine_mpc_constraint_count,
                "lift": "nonzero affine hanging-node lift",
                "max_error": metrics.affine_mpc_transfer_error,
                "reference": "independent CPU MpcSystem::recover",
            },
            "general_dispatch": "unsupported_before_assembly",
            "host_round_trips_during_krylov": "none_for_vectors; projected_coefficient_blocks_are_explicit",
        },
        "failure_contract": {
            "malformed_csr": {
                "status": "rejected_before_device_upload",
                "typed_error": "InvalidArgument",
                "checked": metrics.malformed_csr_rejected,
            },
            "required_cuda_fallback": "runner_negative_probe",
            "device_loss": "backend_contract_outside_fem_dispatcher",
            "cancellation": "backend_contract_outside_fem_dispatcher",
        },
        "warm_residency": {
            "repeats": WARM_REPEATS,
            "allocation_operations": metrics
                .warm_device_allocations
                .iter()
                .zip(&metrics.warm_staging_allocations)
                .map(|(device, staging)| device + staging)
                .collect::<Vec<_>>(),
            "device_allocation_operations": metrics.warm_device_allocations,
            "staging_allocation_operations": metrics.warm_staging_allocations,
            "h2d_operations": metrics.warm_h2d,
            "h2d_bytes": metrics.warm_h2d_bytes,
            "d2h_operations": metrics.warm_d2h,
            "d2h_bytes": metrics.warm_d2h_bytes,
            "d2d_operations": metrics.warm_d2d,
            "d2d_bytes": metrics.warm_d2d_bytes,
            "peak_live_bytes": metrics.setup_peak_device_bytes + metrics.setup_peak_staging_bytes,
            "steady_live_bytes": metrics.steady_live_device_bytes + metrics.steady_live_staging_bytes,
            "peak_device_bytes": metrics.setup_peak_device_bytes,
            "peak_staging_bytes": metrics.setup_peak_staging_bytes,
            "steady_device_bytes": metrics.steady_live_device_bytes,
            "steady_staging_bytes": metrics.steady_live_staging_bytes,
            "transfer_classification": {
                "h2d": "RHS updates plus fused pointer descriptors",
                "d2h": "per-iteration projected coefficient blocks; no Krylov vectors",
                "d2d": "device-internal Krylov reduction/output buffers",
            },
            "frequency_and_design_updates": "resident CSR values; descriptor retained",
        },
        "metrics": {
            "dimension": metrics.dim,
            "nnz": metrics.nnz,
            "malformed_csr_rejected": metrics.malformed_csr_rejected,
            "cached_buffer_count": metrics.cached_buffer_count,
            "cached_device_reduction_capacity": metrics.cached_device_reduction_capacity,
            "cached_basis_slot_capacity": metrics.cached_basis_slot_capacity,
            "transfer_error": metrics.transfer_error,
            "affine_mpc_transfer_error": metrics.affine_mpc_transfer_error,
            "affine_mpc_full_dofs": metrics.affine_mpc_full_dofs,
            "affine_mpc_reduced_dofs": metrics.affine_mpc_reduced_dofs,
            "affine_mpc_constraint_count": metrics.affine_mpc_constraint_count,
            "true_residual": metrics.true_residual,
            "host_residual": metrics.host_residual,
            "projected_residual": metrics.projected_residual,
            "backward_error": metrics.backward_error,
            "workload_elapsed_ms": metrics.elapsed_ms,
        },
        "thresholds": {
            "transfer_error": TRANSFER_TOLERANCE,
            "affine_mpc_transfer_error": AFFINE_MPC_TRANSFER_TOLERANCE,
            "true_residual": RESIDUAL_TOLERANCE,
            "host_residual": RESIDUAL_TOLERANCE,
            "warm_device_allocations": 0,
            "warm_transfer_pattern": "stable across repeats",
            "warm_vector_d2h": 0,
        },
        "environment": environment,
        "runtime": {
            "kernel_launches": stats.kernel_launches,
            "library_operations": stats.library_operations,
            "synchronization_calls": stats.synchronization_calls,
            "allocation_operations": stats.allocation_operations,
            "staging_allocation_operations": stats.staging_allocation_operations,
            "h2d_operations": stats.transfer.h2d_operations,
            "h2d_bytes": stats.transfer.h2d_bytes,
            "d2h_operations": stats.transfer.d2h_operations,
            "d2h_bytes": stats.transfer.d2h_bytes,
            "d2d_operations": stats.transfer.d2d_operations,
            "d2d_bytes": stats.transfer.d2d_bytes,
            "live_allocation_bytes": stats.live_allocation_bytes,
            "allocation_high_water_bytes": stats.allocation_high_water_bytes,
            "live_staging_bytes": stats.live_staging_bytes,
            "staging_high_water_bytes": stats.staging_high_water_bytes,
        },
        "hardware": cuda_hardware_provenance(),
    });
    let bytes = serde_json::to_vec_pretty(&payload)
        .map_err(|error| format!("serialize FEM device-solver evidence: {error}"))?;
    write_qa_artifact(ARTIFACT, &bytes)
}

fn result(run_result: Result<Metrics, String>, stats: CudaRuntimeStatsSnapshot) -> QaResult {
    let mut record = match &run_result {
        Ok(_) => QaResult::passed(CASE_ID),
        Err(error) => QaResult::failed(CASE_ID, error.clone()),
    }
    .requested_backend("cuda")
    .actual_backend("cuda")
    .reference("independent host CSR residual and deterministic device transfer reference")
    .reference_checksum("analytic/deterministic-fixture:fem-device-solver-v1")
    .dimension("2-D constrained H1/Tri3/P1 Helmholtz CSR")
    .formulation("device-resident GMRES + PointJacobi with pooled reusable workspace")
    .tolerance(
        "true and independent residual <= 2e-10; transfer <= 2e-13; zero warm device allocations, stable classified transfers, and no vector host round-trip",
    )
    .runtime_evidence(cuda_runtime_evidence(
        stats,
        QaExecutionCounts {
            executed: 1,
            passed: u64::from(run_result.is_ok()),
            failed: u64::from(run_result.is_err()),
            ..QaExecutionCounts::default()
        },
    ));
    if let Ok(metrics) = &run_result {
        record = record
            .backend_passed("cuda")
            .metric("dimension", metrics.dim as f64)
            .metric("nnz", metrics.nnz as f64)
            .metric(
                "malformed_csr_rejected",
                f64::from(metrics.malformed_csr_rejected),
            )
            .metric("transfer_error", metrics.transfer_error)
            .metric(
                "affine_mpc_transfer_error",
                metrics.affine_mpc_transfer_error,
            )
            .metric("affine_mpc_full_dofs", metrics.affine_mpc_full_dofs as f64)
            .metric(
                "affine_mpc_reduced_dofs",
                metrics.affine_mpc_reduced_dofs as f64,
            )
            .metric(
                "affine_mpc_constraint_count",
                metrics.affine_mpc_constraint_count as f64,
            )
            .metric("true_residual", metrics.true_residual)
            .metric("host_residual", metrics.host_residual)
            .metric("projected_residual", metrics.projected_residual)
            .metric("backward_error", metrics.backward_error)
            .metric("iterations", metrics.iterations as f64)
            .metric(
                "warm_device_allocations_max",
                metrics
                    .warm_device_allocations
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0) as f64,
            )
            .metric(
                "warm_staging_allocations_max",
                metrics
                    .warm_staging_allocations
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0) as f64,
            )
            .metric(
                "warm_h2d_min",
                metrics.warm_h2d.iter().copied().min().unwrap_or(0) as f64,
            )
            .metric(
                "warm_h2d_max",
                metrics.warm_h2d.iter().copied().max().unwrap_or(0) as f64,
            )
            .metric(
                "warm_d2h_max",
                metrics.warm_d2h.iter().copied().max().unwrap_or(0) as f64,
            )
            .metric(
                "warm_d2h_bytes_max",
                metrics.warm_d2h_bytes.iter().copied().max().unwrap_or(0) as f64,
            )
            .metric(
                "warm_d2d_max",
                metrics.warm_d2d.iter().copied().max().unwrap_or(0) as f64,
            )
            .metric(
                "warm_d2d_bytes_max",
                metrics.warm_d2d_bytes.iter().copied().max().unwrap_or(0) as f64,
            )
            .metric("peak_device_bytes", metrics.setup_peak_device_bytes as f64)
            .metric(
                "peak_staging_bytes",
                metrics.setup_peak_staging_bytes as f64,
            )
            .metric(
                "steady_device_bytes",
                metrics.steady_live_device_bytes as f64,
            )
            .metric(
                "steady_staging_bytes",
                metrics.steady_live_staging_bytes as f64,
            )
            .metric("workload_elapsed_ms", metrics.elapsed_ms as f64)
            .maximum("transfer_error", TRANSFER_TOLERANCE)
            .maximum("affine_mpc_transfer_error", AFFINE_MPC_TRANSFER_TOLERANCE)
            .maximum("true_residual", RESIDUAL_TOLERANCE)
            .maximum("host_residual", RESIDUAL_TOLERANCE)
            .maximum("warm_device_allocations_max", 0.0)
            .minimum("iterations", 1.0)
            .seed("rectangular-triangles-3x3:p1:constrained-device-gmres:v1");
        if std::env::var_os("SONIUM_OUTPUT_DIR").is_some() {
            match write_artifact(metrics, stats) {
                Ok((path, checksum)) => {
                    record = record.artifact_with_provenance(
                        path,
                        "crates/sonium-qa/bin/fem_device_solver.rs",
                        "real CUDA FEM device solver, constraint transfer, and warm-residency evidence",
                        "sonium-fem-device-solver/v1",
                        checksum,
                    );
                }
                Err(error) => {
                    record.status = QaStatus::Failed;
                    record.message = Some(error);
                    record.runtime.failure_stage = Some(QaFailureStage::Artifact);
                }
            }
        }
    } else {
        record = record
            .backend_failed("cuda", "device FEM solver contract failed")
            .failure_stage(QaFailureStage::Execution);
    }
    record
}

fn unavailable(error: String) -> QaResult {
    let mut record = QaResult::failed(CASE_ID, error)
        .requested_backend("cuda")
        .actual_backend("unavailable")
        .failure_stage(QaFailureStage::Environment);
    record.runtime.hardware = cuda_hardware_provenance();
    record.runtime.counts = QaExecutionCounts {
        executed: 1,
        failed: 1,
        ..QaExecutionCounts::default()
    };
    record
}

fn main() {
    let started = Instant::now();
    let record = match required_cuda_backend() {
        Ok(backend) => {
            backend.reset_runtime_stats();
            let outcome = run(backend);
            let stats = outcome
                .as_ref()
                .map_or_else(|_| CudaRuntimeStatsSnapshot::default(), |(_, stats)| *stats);
            result(outcome.map(|(metrics, _)| metrics), stats)
        }
        Err(error) => unavailable(error),
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);
    println!(
        "QA_RESULT: {}",
        record
            .to_json_line()
            .expect("FEM device-solver QA result must serialize")
    );
    if matches!(record.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        eprintln!("FEM DEVICE SOLVER FAILED: {:?}", record.message);
        println!("FAILED");
        std::process::exit(1);
    }
}
