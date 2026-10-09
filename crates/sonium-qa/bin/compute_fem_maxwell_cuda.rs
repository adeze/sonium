//! Native CUDA parity for a bounded H(curl) Maxwell CSR action cell.
//!
//! `MaxwellForm` and the checked FEM assembler produce the matrix on the host;
//! an independent direct edge-basis mass quadrature is the CPU oracle.  CUDA
//! owns the forward, transpose, and Hermitian CSR descriptors and vectors.
//! This is a production action/gradient-reference cell, not CUDA quadrature,
//! a general Maxwell solve, or a device-resident FEM gradient implementation.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use num_complex::Complex64;
use serde_json::{Value, json};
use sonium_compute::{
    AdjointDeviceCsr, Backend, BackendError, CudaBackend, CudaFailureInjectionPoint,
    CudaRuntimeStatsSnapshot, DeviceAdjointOperator, DeviceCsr, DeviceOperator, HostCsrRef,
    PrebakedSpmv,
};
use sonium_fem::assembly::{TripletMatrix, assemble_bilinear_checked};
use sonium_fem::mesh::{Mesh, unit_square_triangles};
use sonium_fem::physics::electromagnetics::maxwell::MaxwellForm;
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_fem::{
    FemAction, FemActualBackend, FemBackend, FemBackendRequest, FemScalar, FemWorkload,
    admit_fem_workload, fem_cuda_maxwell_capability_rows,
};
use sonium_qa::{
    QaExecutionCounts, QaFailureStage, QaResult, cuda_hardware_provenance, cuda_runtime_evidence,
    required_cuda_backend, write_qa_artifact,
};

const CASE_ID: &str = "compute.fem-maxwell-cuda.v1";
const QUADRATURE_ORDER: usize = 7;
const MASS_COEFFICIENT: Complex64 = Complex64::new(1.75, -0.5);
const ACTION_TOLERANCE: f64 = 5.0e-12;
const FD_TOLERANCE: f64 = 5.0e-9;
const ENERGY_TOLERANCE: f64 = 5.0e-12;

type EntryMap = BTreeMap<(usize, usize), Complex64>;
type CsrParts = (Vec<usize>, Vec<usize>, Vec<Complex64>);

#[derive(Debug)]
struct Metrics {
    dofs: usize,
    nnz: usize,
    assembly_relative_error: f64,
    forward_residual: f64,
    transpose_residual: f64,
    hermitian_residual: f64,
    inner_product_error: f64,
    symmetry_error: f64,
    energy_real: f64,
    energy_error: f64,
    fd_directional_error: f64,
    warm_action_count: usize,
    warm_h2d_operations: usize,
    warm_allocation_operations: usize,
    failure_typed: bool,
    output_preserved: bool,
    retry_residual: f64,
    malformed_csr_rejected: bool,
    shape_rejected: bool,
    tail_elements: usize,
    elapsed_ms: u64,
}

fn aggregate(matrix: &TripletMatrix<Complex64>) -> EntryMap {
    let mut entries = EntryMap::new();
    for ((&row, &column), &value) in matrix.rows.iter().zip(&matrix.cols).zip(&matrix.values) {
        *entries.entry((row, column)).or_default() += value;
    }
    entries
}

/// Integrate the public H(curl) edge basis directly, independently of
/// MaxwellForm.  The production cell deliberately sets curl-curl to zero so
/// this oracle exercises the complex Maxwell mass coefficient without sharing
/// its form implementation.
fn independent_edge_mass(mesh: &Mesh) -> BTreeMap<(usize, usize), f64> {
    let space = HcurlSpace::new();
    let mut reference = BTreeMap::new();
    for element in 0..mesh.num_elements() {
        let quadrature = QuadratureRule::new(mesh.elements[element].element_type, QUADRATURE_ORDER);
        let dofs = space.local_to_global(mesh, element);
        for point in quadrature.iter() {
            let tabulation = space.tabulate(mesh, element, point.xi(), point.eta(), point.zeta());
            let weight = point.weight * tabulation.det_j;
            for (local_row, &row) in dofs.iter().enumerate() {
                for (local_column, &column) in dofs.iter().enumerate() {
                    let value = (0..tabulation.spatial_dim)
                        .map(|component| {
                            tabulation.vector_value(local_row, component)
                                * tabulation.vector_value(local_column, component)
                        })
                        .sum::<f64>()
                        * weight;
                    *reference.entry((row, column)).or_default() += value;
                }
            }
        }
    }
    reference
}

fn csr_from_entries(dim: usize, entries: &EntryMap) -> CsrParts {
    let mut row_ptrs = vec![0usize; dim + 1];
    for &(row, _) in entries.keys() {
        row_ptrs[row + 1] += 1;
    }
    for row in 0..dim {
        row_ptrs[row + 1] += row_ptrs[row];
    }
    let mut columns = Vec::with_capacity(entries.len());
    let mut values = Vec::with_capacity(entries.len());
    for row in 0..dim {
        for (&(entry_row, column), &value) in entries {
            if entry_row == row {
                columns.push(column);
                values.push(value);
            }
        }
    }
    (row_ptrs, columns, values)
}

fn csr_apply(
    row_ptrs: &[usize],
    columns: &[usize],
    values: &[Complex64],
    x: &[Complex64],
    transpose: bool,
    conjugate: bool,
) -> Vec<Complex64> {
    let dim = row_ptrs.len() - 1;
    let mut output = vec![Complex64::new(0.0, 0.0); dim];
    for row in 0..dim {
        for index in row_ptrs[row]..row_ptrs[row + 1] {
            let value = if conjugate {
                values[index].conj()
            } else {
                values[index]
            };
            if transpose {
                output[columns[index]] += value * x[row];
            } else {
                output[row] += value * x[columns[index]];
            }
        }
    }
    output
}

fn relative_l2(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    let numerator = actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| (actual - expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator = expected
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    numerator / denominator
}

fn inner(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    x.iter().zip(y).map(|(&x, &y)| x.conj() * y).sum()
}

fn stats_value(stats: CudaRuntimeStatsSnapshot) -> Value {
    json!({
        "kernel_launches": stats.kernel_launches,
        "library_operations": stats.library_operations,
        "synchronization_calls": stats.synchronization_calls,
        "failed_synchronizations": stats.failed_synchronizations,
        "completion_events": stats.completion_events,
        "completion_waits": stats.completion_waits,
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
    })
}

fn malformed_csr_rejected(backend: &CudaBackend) -> bool {
    let non_monotone_values = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
    let non_monotone =
        HostCsrRef::try_new(2, 2, &[0usize, 2, 1], &[0usize, 1], &non_monotone_values);
    if non_monotone.is_ok() {
        return false;
    }
    let rows = [0usize, 1, 2];
    let columns = [0usize, 2];
    let values = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
    let Ok(host) = HostCsrRef::try_new(2, 2, &rows, &columns, &values) else {
        return true;
    };
    matches!(
        DeviceCsr::<CudaBackend, Complex64>::try_from_host_csr_ref(backend, host),
        Err(BackendError::InvalidArgument { .. })
    )
}

fn run(backend: &CudaBackend) -> Result<(Metrics, CudaRuntimeStatsSnapshot), String> {
    for action in [
        FemAction::Forward,
        FemAction::Transpose,
        FemAction::Hermitian,
    ] {
        let admission = admit_fem_workload(
            FemWorkload::native_cuda_maxwell(FemScalar::Complex64, action),
            FemBackendRequest::required(FemBackend::Cuda),
        )
        .map_err(|error| format!("Maxwell CUDA admission failed before assembly: {error}"))?;
        if admission.provenance.actual_backend != FemActualBackend::Cuda
            || admission.provenance.fallback != "none"
        {
            return Err(format!(
                "Maxwell admission did not prove native CUDA: {admission:?}"
            ));
        }
    }

    let started = Instant::now();
    let mesh = unit_square_triangles(1);
    let space = HcurlSpace::new();
    let form = MaxwellForm::from_coefficients(0.0, MASS_COEFFICIENT);
    let assembled = assemble_bilinear_checked(&form, &space, &space, &mesh, QUADRATURE_ORDER)
        .map_err(|error| format!("production Maxwell assembly failed: {error:?}"))?;
    let actual = aggregate(&assembled);
    let mass = independent_edge_mass(&mesh);
    let dim = space.num_global_dofs(&mesh);
    if dim == 0 || dim % 2 == 0 {
        return Err(format!(
            "fixture must exercise a non-empty odd tail, got {dim} DOFs"
        ));
    }
    let expected: EntryMap = mass
        .iter()
        .map(|(&key, &value)| (key, MASS_COEFFICIENT * value))
        .collect();
    let keys: BTreeSet<_> = actual.keys().chain(expected.keys()).copied().collect();
    let scale = expected
        .values()
        .map(|value| value.norm())
        .sum::<f64>()
        .max(1.0e-12);
    let assembly_error = keys
        .iter()
        .map(|key| {
            (actual.get(key).copied().unwrap_or_default()
                - expected.get(key).copied().unwrap_or_default())
            .norm()
        })
        .fold(0.0, f64::max)
        / scale;
    if !assembly_error.is_finite() || assembly_error > ACTION_TOLERANCE {
        return Err(format!(
            "independent Maxwell mass oracle failed: {assembly_error:e}"
        ));
    }
    let (row_ptrs, columns, values) = csr_from_entries(dim, &expected);
    let host = HostCsrRef::try_new(dim, dim, &row_ptrs, &columns, &values)
        .map_err(|error| format!("Maxwell CSR admission failed: {error}"))?;
    let transposed = host
        .try_transpose()
        .map_err(|error| format!("Maxwell transpose CSR construction failed: {error}"))?;
    let adjoint = AdjointDeviceCsr::try_from_host_csr_ref(backend, host)
        .map_err(|error| format!("Maxwell forward/Hermitian upload failed: {error}"))?;
    let transpose = DeviceCsr::try_from_host_csr_ref(backend, transposed.as_ref())
        .map_err(|error| format!("Maxwell transpose upload failed: {error}"))?;
    let forward = adjoint
        .try_prebake(backend)
        .map_err(|error| format!("Maxwell forward/Hermitian prebake failed: {error}"))?;
    let transpose = PrebakedSpmv::try_from_csr(backend, &transpose)
        .map_err(|error| format!("Maxwell transpose prebake failed: {error}"))?;

    let x_host: Vec<_> = (0..dim)
        .map(|index| {
            Complex64::new(
                (index as f64 * 0.37 + 0.11).sin(),
                (index as f64 * 0.19).cos(),
            )
        })
        .collect();
    let y_host: Vec<_> = (0..dim)
        .map(|index| {
            Complex64::new(
                (index as f64 * 0.23 + 0.41).cos(),
                (index as f64 * 0.17).sin(),
            )
        })
        .collect();
    let forward_reference = csr_apply(&row_ptrs, &columns, &values, &x_host, false, false);
    let transpose_reference = csr_apply(&row_ptrs, &columns, &values, &y_host, true, false);
    let hermitian_reference = csr_apply(&row_ptrs, &columns, &values, &y_host, true, true);
    let energy_real = inner(&x_host, &forward_reference).re;
    if !energy_real.is_finite() || energy_real <= 0.0 {
        return Err(format!(
            "Maxwell mass energy is not positive: {energy_real}"
        ));
    }
    let symmetry_error = relative_l2(
        &csr_apply(&row_ptrs, &columns, &values, &x_host, true, false),
        &forward_reference,
    );
    let epsilon = 1.0e-5;
    let plus: Vec<_> = x_host
        .iter()
        .zip(&y_host)
        .map(|(&x, &y)| x + epsilon * y)
        .collect();
    let minus: Vec<_> = x_host
        .iter()
        .zip(&y_host)
        .map(|(&x, &y)| x - epsilon * y)
        .collect();
    let energy = |vector: &[Complex64]| {
        inner(
            vector,
            &csr_apply(&row_ptrs, &columns, &values, vector, false, false),
        )
        .re
    };
    let fd_derivative = (energy(&plus) - energy(&minus)) / (2.0 * epsilon);
    let analytic_derivative = (inner(&y_host, &forward_reference)
        + inner(
            &x_host,
            &csr_apply(&row_ptrs, &columns, &values, &y_host, false, false),
        ))
    .re;
    let fd_directional_error = (fd_derivative - analytic_derivative).abs()
        / analytic_derivative.abs().max(f64::MIN_POSITIVE);

    let mut x = backend
        .raw_alloc::<Complex64>(dim)
        .map_err(|error| format!("Maxwell x allocation failed: {error}"))?;
    let mut y = backend
        .raw_alloc::<Complex64>(dim)
        .map_err(|error| format!("Maxwell y allocation failed: {error}"))?;
    let mut ax = backend
        .raw_alloc::<Complex64>(dim)
        .map_err(|error| format!("Maxwell Ax allocation failed: {error}"))?;
    let mut aty = backend
        .raw_alloc::<Complex64>(dim)
        .map_err(|error| format!("Maxwell A^H y allocation failed: {error}"))?;
    let mut at_y = backend
        .raw_alloc::<Complex64>(dim)
        .map_err(|error| format!("Maxwell A^T y allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&x_host, &mut x)
        .map_err(|error| format!("Maxwell x upload failed: {error}"))?;
    backend
        .try_copy_h2d(&y_host, &mut y)
        .map_err(|error| format!("Maxwell y upload failed: {error}"))?;

    let initial_baseline = backend.begin_runtime_stats();
    forward
        .try_apply(backend, &x, &mut ax)
        .map_err(|error| format!("Maxwell forward action failed: {error}"))?;
    transpose
        .try_apply(backend, &y, &mut at_y)
        .map_err(|error| format!("Maxwell transpose action failed: {error}"))?;
    forward
        .try_apply_hermitian(backend, &y, &mut aty)
        .map_err(|error| format!("Maxwell Hermitian action failed: {error}"))?;
    let initial_stats = backend.runtime_stats_since(initial_baseline);
    let mut ax_host = vec![Complex64::new(0.0, 0.0); dim];
    let mut at_y_host = vec![Complex64::new(0.0, 0.0); dim];
    let mut aty_host = vec![Complex64::new(0.0, 0.0); dim];
    backend
        .try_copy_d2h(&ax, &mut ax_host)
        .map_err(|error| format!("Maxwell Ax download failed: {error}"))?;
    backend
        .try_copy_d2h(&at_y, &mut at_y_host)
        .map_err(|error| format!("Maxwell A^T y download failed: {error}"))?;
    backend
        .try_copy_d2h(&aty, &mut aty_host)
        .map_err(|error| format!("Maxwell A^H y download failed: {error}"))?;
    let forward_residual = relative_l2(&ax_host, &forward_reference);
    let transpose_residual = relative_l2(&at_y_host, &transpose_reference);
    let hermitian_residual = relative_l2(&aty_host, &hermitian_reference);
    let inner_product_error = (inner(&ax_host, &y_host) - inner(&x_host, &aty_host)).norm()
        / inner(&ax_host, &y_host)
            .norm()
            .max(inner(&x_host, &aty_host).norm())
            .max(f64::MIN_POSITIVE);
    let energy_error = (inner(&x_host, &ax_host).re - energy_real).abs() / energy_real;

    let warm_baseline = backend.begin_runtime_stats();
    forward
        .try_apply(backend, &x, &mut ax)
        .map_err(|error| format!("Maxwell warm forward action failed: {error}"))?;
    transpose
        .try_apply(backend, &y, &mut at_y)
        .map_err(|error| format!("Maxwell warm transpose action failed: {error}"))?;
    forward
        .try_apply_hermitian(backend, &y, &mut aty)
        .map_err(|error| format!("Maxwell warm Hermitian action failed: {error}"))?;
    let warm_stats = backend.runtime_stats_since(warm_baseline);
    let mut warm_ax_host = vec![Complex64::new(0.0, 0.0); dim];
    backend
        .try_copy_d2h(&ax, &mut warm_ax_host)
        .map_err(|error| format!("Maxwell warm Ax download failed: {error}"))?;

    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = forward.try_apply(backend, &x, &mut ax);
    let failure_typed = matches!(&failure, Err(BackendError::Launch { .. }));
    backend.clear_failure();
    if failure_typed {
        backend
            .recover_after_error()
            .map_err(|error| format!("Maxwell CUDA recovery failed: {error}"))?;
    }
    let mut after_failure = vec![Complex64::new(0.0, 0.0); dim];
    backend
        .try_copy_d2h(&ax, &mut after_failure)
        .map_err(|error| format!("Maxwell failure-preservation download failed: {error}"))?;
    let output_preserved = failure_typed
        && relative_l2(&after_failure, &warm_ax_host) == 0.0
        && backend
            .pending_error()
            .map_err(|error| error.to_string())?
            .is_none();
    forward
        .try_apply(backend, &x, &mut ax)
        .map_err(|error| format!("Maxwell retry action failed: {error}"))?;
    let mut retry_host = vec![Complex64::new(0.0, 0.0); dim];
    backend
        .try_copy_d2h(&ax, &mut retry_host)
        .map_err(|error| format!("Maxwell retry download failed: {error}"))?;
    let retry_residual = relative_l2(&retry_host, &forward_reference);
    let malformed = malformed_csr_rejected(backend);
    let mut short = backend
        .raw_alloc::<Complex64>(dim.saturating_sub(1))
        .map_err(|error| format!("Maxwell short-vector allocation failed: {error}"))?;
    let shape_rejected = matches!(
        forward.try_apply(backend, &short, &mut ax),
        Err(BackendError::InvalidArgument { .. })
    );
    // Keep the short buffer live until the typed shape check has completed.
    let _ = &mut short;

    let metrics = Metrics {
        dofs: dim,
        nnz: values.len(),
        assembly_relative_error: assembly_error,
        forward_residual,
        transpose_residual,
        hermitian_residual,
        inner_product_error,
        symmetry_error,
        energy_real,
        energy_error,
        fd_directional_error,
        warm_action_count: 3,
        warm_h2d_operations: warm_stats.transfer.h2d_operations,
        warm_allocation_operations: warm_stats.allocation_operations,
        failure_typed,
        output_preserved,
        retry_residual,
        malformed_csr_rejected: malformed,
        shape_rejected,
        tail_elements: dim % 32,
        elapsed_ms: started.elapsed().as_millis() as u64,
    };
    if metrics.assembly_relative_error > ACTION_TOLERANCE
        || metrics.forward_residual > ACTION_TOLERANCE
        || metrics.transpose_residual > ACTION_TOLERANCE
        || metrics.hermitian_residual > ACTION_TOLERANCE
        || metrics.inner_product_error > ACTION_TOLERANCE
        || metrics.symmetry_error > ACTION_TOLERANCE
        || metrics.energy_error > ENERGY_TOLERANCE
        || metrics.fd_directional_error > FD_TOLERANCE
        || metrics.retry_residual > ACTION_TOLERANCE
        || !metrics.failure_typed
        || !metrics.output_preserved
        || !metrics.malformed_csr_rejected
        || !metrics.shape_rejected
        || metrics.warm_h2d_operations != 0
        || metrics.warm_allocation_operations != 0
        || initial_stats.kernel_launches + initial_stats.library_operations == 0
    {
        return Err(format!("Maxwell CUDA parity gates failed: {metrics:?}"));
    }
    Ok((metrics, backend.runtime_stats()))
}

fn write_artifact(
    metrics: &Metrics,
    stats: CudaRuntimeStatsSnapshot,
) -> Result<(String, String), String> {
    let capability_rows: Vec<_> = fem_cuda_maxwell_capability_rows()
        .into_iter()
        .map(|row| serde_json::to_value(row).expect("Maxwell capability row serializes"))
        .collect();
    let payload = json!({
        "schema": "sonium-compute-cuda-fem-maxwell/v1",
        "status": "passed",
        "case_id": CASE_ID,
        "cell": {
            "space": "Hcurl",
            "element": "tri3",
            "order": "P1",
            "physics": "maxwell",
            "mode": "assembled",
            "scalar": "Complex64",
            "actions": ["forward", "transpose", "hermitian"],
        },
        "matrix": {
            "dofs": metrics.dofs,
            "nnz": metrics.nnz,
            "fixture": "unit-square-triangles-1:hcurl:complex-mass",
            "assembly": "production MaxwellForm + assemble_bilinear_checked",
            "cpu_oracle": "independent direct Hcurl edge-basis mass quadrature",
        },
        "metrics": {
            "assembly_relative_error": metrics.assembly_relative_error,
            "forward_residual": metrics.forward_residual,
            "transpose_residual": metrics.transpose_residual,
            "hermitian_residual": metrics.hermitian_residual,
            "inner_product_error": metrics.inner_product_error,
            "symmetry_error": metrics.symmetry_error,
            "energy_real": metrics.energy_real,
            "energy_error": metrics.energy_error,
            "fd_directional_error": metrics.fd_directional_error,
            "warm_action_count": metrics.warm_action_count,
            "warm_h2d_operations": metrics.warm_h2d_operations,
            "warm_allocation_operations": metrics.warm_allocation_operations,
            "failure_typed": metrics.failure_typed,
            "output_preserved": metrics.output_preserved,
            "retry_residual": metrics.retry_residual,
            "malformed_csr_rejected": metrics.malformed_csr_rejected,
            "shape_rejected": metrics.shape_rejected,
            "tail_elements": metrics.tail_elements,
        },
        "thresholds": {
            "assembly_relative_error": ACTION_TOLERANCE,
            "forward_residual": ACTION_TOLERANCE,
            "transpose_residual": ACTION_TOLERANCE,
            "hermitian_residual": ACTION_TOLERANCE,
            "inner_product_error": ACTION_TOLERANCE,
            "symmetry_error": ACTION_TOLERANCE,
            "energy_error": ENERGY_TOLERANCE,
            "fd_directional_error": FD_TOLERANCE,
            "retry_residual": ACTION_TOLERANCE,
            "warm_action_count": 3,
            "warm_h2d_operations": 0,
            "warm_allocation_operations": 0,
            "failure_typed": 1,
            "output_preserved": 1,
            "malformed_csr_rejected": 1,
            "shape_rejected": 1,
            "tail_elements": 1,
        },
        "threshold_directions": {
            "assembly_relative_error": "maximum",
            "forward_residual": "maximum",
            "transpose_residual": "maximum",
            "hermitian_residual": "maximum",
            "inner_product_error": "maximum",
            "symmetry_error": "maximum",
            "energy_error": "maximum",
            "fd_directional_error": "maximum",
            "retry_residual": "maximum",
            "warm_action_count": "minimum",
            "warm_h2d_operations": "maximum",
            "warm_allocation_operations": "maximum",
            "failure_typed": "minimum",
            "output_preserved": "minimum",
            "malformed_csr_rejected": "minimum",
            "shape_rejected": "minimum",
            "tail_elements": "minimum",
        },
        "workload_elapsed_ms": metrics.elapsed_ms.max(1),
        "runtime": stats_value(stats),
        "warm_residency": {
            "operator_storage": "device-resident",
            "input_vectors": "device-resident",
            "forward_transpose_hermitian": "prebaked-device-resident",
            "warm_action_count": metrics.warm_action_count,
            "h2d_operations": metrics.warm_h2d_operations,
            "allocation_operations": metrics.warm_allocation_operations,
        },
        "dispatch": {
            "requested_backend": "cuda",
            "actual_backend": "cuda",
            "fallback": "none",
            "admission_before_assembly": true,
            "native_device_csr": true,
        },
        "failure_contracts": [
            {
                "id": "launch.output_preservation_and_retry",
                "status": "typed_failure_recovered",
                "stage": "launch",
                "error": "BackendError::Launch",
                "output_preserved": metrics.output_preserved,
                "retry_succeeded": metrics.retry_residual <= ACTION_TOLERANCE,
                "fallback": "none",
                "evidence": "native CUDA launch injection plus D2H comparison",
            },
            {
                "id": "malformed_csr.admission",
                "status": "typed_rejection",
                "stage": "admission",
                "error": "HostCsrError/BackendError::InvalidArgument",
                "rejected": metrics.malformed_csr_rejected,
                "fallback": "none",
                "evidence": "non-monotone row pointer and out-of-range column checked",
            },
            {
                "id": "vector_shape.admission",
                "status": "typed_rejection",
                "stage": "action_preflight",
                "error": "BackendError::InvalidArgument",
                "rejected": metrics.shape_rejected,
                "fallback": "none",
                "evidence": "short device input rejected before CUDA action",
            },
        ],
        "gradient_reference": {
            "status": "cpu_finite_difference_only",
            "directional_energy_error": metrics.fd_directional_error,
            "device_gradient": "not_implemented",
        },
        "capability_rows": capability_rows,
        "hardware": cuda_hardware_provenance(),
    });
    let bytes = serde_json::to_vec_pretty(&payload)
        .map_err(|error| format!("serialize Maxwell artifact: {error}"))?;
    write_qa_artifact("compute_fem_maxwell_cuda.json", &bytes)
}

fn qa_result(
    run_result: Result<(Metrics, CudaRuntimeStatsSnapshot), String>,
    started: Instant,
) -> QaResult {
    let mut result = match &run_result {
        Ok(_) => QaResult::passed(CASE_ID),
        Err(error) => QaResult::failed(CASE_ID, error.clone()),
    }
    .requested_backend("cuda")
    .actual_backend("cuda")
    .reference("independent H(curl) edge-basis mass quadrature plus CPU CSR and finite-difference oracle")
    .reference_checksum("analytic/deterministic-fixture:compute-fem-maxwell-cuda-v1")
    .dimension("2-D unit-square Tri3, H(curl) P1, Complex64 Maxwell mass action")
    .formulation("production MaxwellForm assembly plus native CUDA prebaked CSR forward/transpose/Hermitian action")
    .tolerance("action/energy/symmetry <= 5e-12; CPU FD <= 5e-9; warm resident; typed failure recovery")
    .seed("unit-square-triangles-1:hcurl:mass=1.75-0.5i")
    .elapsed_ms(started.elapsed().as_millis() as u64)
    .runtime_evidence(cuda_runtime_evidence(
        run_result.as_ref().ok().map_or_else(CudaRuntimeStatsSnapshot::default, |(_, stats)| *stats),
        QaExecutionCounts {
            executed: 1,
            passed: u64::from(run_result.is_ok()),
            failed: u64::from(run_result.is_err()),
            ..QaExecutionCounts::default()
        },
    ));
    result = match &run_result {
        Ok((metrics, stats)) => match write_artifact(metrics, *stats) {
            Ok((path, checksum)) => result
                .backend_passed("cuda")
                .artifact_with_provenance(
                    path,
                    "crates/sonium-qa/bin/compute_fem_maxwell_cuda.rs",
                    "native SM120 CUDA Maxwell H(curl) CSR action parity",
                    "sonium-compute-cuda-fem-maxwell/v1",
                    checksum,
                )
                .metric("dofs", metrics.dofs as f64)
                .metric("nnz", metrics.nnz as f64)
                .metric("forward_residual", metrics.forward_residual)
                .metric("transpose_residual", metrics.transpose_residual)
                .metric("hermitian_residual", metrics.hermitian_residual)
                .metric("fd_directional_error", metrics.fd_directional_error),
            Err(error) => {
                result = result
                    .backend_failed("cuda", error.clone())
                    .failure_stage(QaFailureStage::Artifact);
                result.message = Some(error);
                result
            }
        },
        Err(error) => result
            .backend_failed("cuda", error.clone())
            .failure_stage(QaFailureStage::Execution),
    };
    result
}

fn main() {
    let started = Instant::now();
    let run_result = match required_cuda_backend() {
        Ok(backend) => run(&backend),
        Err(error) => Err(error),
    };
    let result = qa_result(run_result, started);
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("Maxwell QA result serializes")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        eprintln!("MAXWELL CUDA FEM FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
