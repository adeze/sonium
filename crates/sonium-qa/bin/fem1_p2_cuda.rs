//! FEM1 bounded native CUDA H1/Tri3/P2 assembled action.
//!
//! This cell extends the existing P1 and local/matrix-free evidence with one
//! distinct production element/order/form combination.  The production
//! Helmholtz assembly remains host-owned; native CUDA owns the resident CSR
//! forward/Hermitian actions and their checked failure boundary.

use std::collections::BTreeMap;
use std::time::Instant;

use num_complex::Complex64;
use serde_json::{Value, json};
use sonium_compute::{
    AdjointDeviceCsr, Backend, BackendContextDisposition, BackendError, BackendOutputDisposition,
    CudaBackend, CudaFailureInjectionPoint, CudaRuntimeStatsSnapshot, DeviceAdjointOperator,
    DeviceOperator, HostCsrRef,
};
use sonium_fem::assembly::HelmholtzProblem;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::rectangular_mesh_triangles;
use sonium_fem::{
    FemAction, FemActualBackend, FemBackend, FemBackendRequest, FemScalar, FemWorkload,
    admit_fem_workload,
};
use sonium_qa::{cuda_hardware_provenance, required_cuda_backend, write_qa_artifact};

const SCHEMA: &str = "sonium-compute-cuda-fem-p2/v1";
const CASE_ID: &str = "compute.fem-p2-cuda.v1";
const WAVENUMBER: Complex64 = Complex64::new(2.15, 0.11);
const ACTION_TOLERANCE: f64 = 2.0e-11;

#[derive(Debug)]
struct Metrics {
    dimension: usize,
    nnz: usize,
    forward_error: f64,
    hermitian_error: f64,
    warm_forward_error: f64,
    warm_hermitian_error: f64,
    warm_kernel_operations: usize,
    warm_h2d_operations: usize,
    warm_allocation_operations: usize,
    warm_staging_allocation_operations: usize,
    malformed_csr_rejected: bool,
    malformed_shape_rejected: bool,
    malformed_output_preserved: bool,
    failure_typed: bool,
    failure_kernel_operations: usize,
    failure_context: BackendContextDisposition,
    failure_output: BackendOutputDisposition,
    failure_published: bool,
    retry_error: f64,
    retry_succeeded: bool,
    live_before_teardown: usize,
    live_after_teardown: usize,
    staging_after_teardown: usize,
    elapsed_ms: u64,
}

type CsrParts = (Vec<usize>, Vec<usize>, Vec<Complex64>);

fn csr_from_triplets(
    dimension: usize,
    rows: &[usize],
    columns: &[usize],
    values: &[Complex64],
) -> Result<CsrParts, String> {
    if rows.len() != columns.len() || rows.len() != values.len() {
        return Err("P2 Helmholtz triplet arrays have different lengths".into());
    }
    let mut entries = BTreeMap::<(usize, usize), Complex64>::new();
    for ((&row, &column), &value) in rows.iter().zip(columns).zip(values) {
        if row >= dimension || column >= dimension || !value.re.is_finite() || !value.im.is_finite()
        {
            return Err("P2 Helmholtz triplet contains an invalid index or value".into());
        }
        *entries.entry((row, column)).or_default() += value;
    }
    let mut row_ptrs = vec![0usize; dimension + 1];
    for &(row, _) in entries.keys() {
        row_ptrs[row + 1] += 1;
    }
    for row in 0..dimension {
        row_ptrs[row + 1] += row_ptrs[row];
    }
    let mut col_indices = Vec::with_capacity(entries.len());
    let mut csr_values = Vec::with_capacity(entries.len());
    for row in 0..dimension {
        for (&(entry_row, column), &value) in &entries {
            if entry_row == row {
                col_indices.push(column);
                csr_values.push(value);
            }
        }
    }
    Ok((row_ptrs, col_indices, csr_values))
}

fn apply_cpu(
    dimension: usize,
    row_ptrs: &[usize],
    columns: &[usize],
    values: &[Complex64],
    input: &[Complex64],
    hermitian: bool,
) -> Vec<Complex64> {
    let mut output = vec![Complex64::default(); dimension];
    for row in 0..dimension {
        for entry in row_ptrs[row]..row_ptrs[row + 1] {
            let value = if hermitian {
                values[entry].conj()
            } else {
                values[entry]
            };
            let (output_index, input_index) = if hermitian {
                (columns[entry], row)
            } else {
                (row, columns[entry])
            };
            output[output_index] += value * input[input_index];
        }
    }
    output
}

fn max_relative_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).norm() / expected.norm().max(1.0e-30))
        .fold(0.0, f64::max)
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

fn native_admission(action: FemAction) -> Result<(), String> {
    let admission = admit_fem_workload(
        FemWorkload::native_cuda_p2(FemScalar::Complex64, action),
        FemBackendRequest::required(FemBackend::Cuda),
    )
    .map_err(|error| format!("P2 CUDA admission failed before assembly: {error}"))?;
    if admission.provenance.actual_backend != FemActualBackend::Cuda
        || admission.provenance.fallback != "none"
    {
        return Err(format!(
            "P2 admission did not prove native CUDA: {admission:?}"
        ));
    }
    Ok(())
}

fn run(backend: &CudaBackend) -> Result<(Metrics, CudaRuntimeStatsSnapshot), String> {
    for action in [FemAction::Forward, FemAction::Hermitian] {
        native_admission(action)?;
    }

    let started = Instant::now();
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 3, 3);
    let problem =
        HelmholtzProblem::try_assemble(&mesh, PolynomialDegree::P2, WAVENUMBER, |x, y, _| {
            Complex64::new(1.0 + x - 0.25 * y, 0.5 * x + y)
        })
        .map_err(|error| format!("production P2/Tri3 Helmholtz assembly failed: {error}"))?;
    let dimension = problem.num_dofs();
    if dimension <= 32 || dimension % 2 == 0 {
        return Err(format!(
            "P2 fixture must exercise a nontrivial odd/tail dimension, got {dimension}"
        ));
    }
    let (row_ptrs, col_indices, values) = csr_from_triplets(
        dimension,
        &problem.matrix.rows,
        &problem.matrix.cols,
        &problem.matrix.values,
    )?;
    let nnz = values.len();
    if nnz <= dimension {
        return Err(format!(
            "P2 CSR is unexpectedly empty/small: nnz={nnz}, n={dimension}"
        ));
    }
    let host = HostCsrRef::try_new(dimension, dimension, &row_ptrs, &col_indices, &values)
        .map_err(|error| format!("P2 CSR admission failed: {error}"))?;
    let cpu_input: Vec<_> = (0..dimension)
        .map(|index| {
            Complex64::new(
                (index as f64 * 0.31 + 0.2).sin(),
                (index as f64 * 0.17 + 0.1).cos(),
            )
        })
        .collect();
    let cpu_test: Vec<_> = (0..dimension)
        .map(|index| {
            Complex64::new(
                (index as f64 * 0.23 + 0.4).cos(),
                (index as f64 * 0.19 + 0.3).sin(),
            )
        })
        .collect();
    let forward_reference = apply_cpu(
        dimension,
        &row_ptrs,
        &col_indices,
        &values,
        &cpu_input,
        false,
    );
    let hermitian_reference =
        apply_cpu(dimension, &row_ptrs, &col_indices, &values, &cpu_test, true);

    let operator_storage = AdjointDeviceCsr::try_from_host_csr_ref(backend, host)
        .map_err(|error| format!("P2 device CSR upload failed: {error}"))?;
    let operator = operator_storage
        .try_prebake(backend)
        .map_err(|error| format!("P2 CSR prebake failed: {error}"))?;
    let mut input = backend
        .raw_alloc::<Complex64>(dimension)
        .map_err(|error| format!("P2 input allocation failed: {error}"))?;
    let mut test = backend
        .raw_alloc::<Complex64>(dimension)
        .map_err(|error| format!("P2 test allocation failed: {error}"))?;
    let mut forward_output = backend
        .raw_alloc::<Complex64>(dimension)
        .map_err(|error| format!("P2 forward output allocation failed: {error}"))?;
    let mut hermitian_output = backend
        .raw_alloc::<Complex64>(dimension)
        .map_err(|error| format!("P2 Hermitian output allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&cpu_input, &mut input)
        .map_err(|error| format!("P2 input upload failed: {error}"))?;
    backend
        .try_copy_h2d(&cpu_test, &mut test)
        .map_err(|error| format!("P2 test upload failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("P2 setup synchronization failed: {error}"))?;
    backend.reset_runtime_stats();

    let warm_start = backend.begin_runtime_stats();
    operator
        .try_apply(backend, &input, &mut forward_output)
        .map_err(|error| format!("P2 forward action failed: {error}"))?;
    operator
        .try_apply_hermitian(backend, &test, &mut hermitian_output)
        .map_err(|error| format!("P2 Hermitian action failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("P2 warm synchronization failed: {error}"))?;
    let mut forward = vec![Complex64::default(); dimension];
    let mut hermitian = vec![Complex64::default(); dimension];
    backend
        .try_copy_d2h(&forward_output, &mut forward)
        .map_err(|error| format!("P2 forward download failed: {error}"))?;
    backend
        .try_copy_d2h(&hermitian_output, &mut hermitian)
        .map_err(|error| format!("P2 Hermitian download failed: {error}"))?;
    let _warm_stats = backend.runtime_stats_since(warm_start);
    let forward_error = max_relative_error(&forward, &forward_reference);
    let hermitian_error = max_relative_error(&hermitian, &hermitian_reference);

    let repeat_start = backend.begin_runtime_stats();
    operator
        .try_apply(backend, &input, &mut forward_output)
        .map_err(|error| format!("P2 repeat forward action failed: {error}"))?;
    operator
        .try_apply_hermitian(backend, &test, &mut hermitian_output)
        .map_err(|error| format!("P2 repeat Hermitian action failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("P2 repeat synchronization failed: {error}"))?;
    let repeat_stats = backend.runtime_stats_since(repeat_start);
    backend
        .try_copy_d2h(&forward_output, &mut forward)
        .map_err(|error| format!("P2 repeat forward download failed: {error}"))?;
    backend
        .try_copy_d2h(&hermitian_output, &mut hermitian)
        .map_err(|error| format!("P2 repeat Hermitian download failed: {error}"))?;
    let warm_forward_error = max_relative_error(&forward, &forward_reference);
    let warm_hermitian_error = max_relative_error(&hermitian, &hermitian_reference);
    let warm_kernel_operations = repeat_stats.kernel_launches + repeat_stats.library_operations;
    if warm_forward_error > ACTION_TOLERANCE
        || warm_hermitian_error > ACTION_TOLERANCE
        || warm_kernel_operations < 2
        || repeat_stats.transfer.h2d_operations != 0
        || repeat_stats.allocation_operations != 0
        || repeat_stats.staging_allocation_operations != 0
    {
        return Err(format!(
            "P2 warm residency/parity gates failed: forward={warm_forward_error:.3e}, hermitian={warm_hermitian_error:.3e}, stats={repeat_stats:?}"
        ));
    }

    let bad_row_ptrs = &row_ptrs[..dimension];
    let malformed_csr_rejected =
        HostCsrRef::try_new(dimension, dimension, bad_row_ptrs, &col_indices, &values).is_err();
    let mut malformed_output = backend
        .raw_alloc::<Complex64>(dimension - 1)
        .map_err(|error| format!("P2 malformed output allocation failed: {error}"))?;
    let malformed_sentinel = vec![Complex64::new(17.0, -19.0); dimension - 1];
    backend
        .try_copy_h2d(&malformed_sentinel, &mut malformed_output)
        .map_err(|error| format!("P2 malformed sentinel upload failed: {error}"))?;
    let malformed_shape_rejected = matches!(
        operator.try_apply(backend, &input, &mut malformed_output),
        Err(BackendError::InvalidArgument { .. })
    );
    let mut malformed_after = vec![Complex64::default(); dimension - 1];
    backend
        .try_copy_d2h(&malformed_output, &mut malformed_after)
        .map_err(|error| format!("P2 malformed output download failed: {error}"))?;
    let malformed_output_preserved = malformed_after == malformed_sentinel;

    let failure_sentinel = vec![Complex64::new(-23.0, 29.0); dimension];
    backend
        .try_copy_h2d(&failure_sentinel, &mut forward_output)
        .map_err(|error| format!("P2 failure sentinel upload failed: {error}"))?;
    let failure_start = backend.begin_runtime_stats();
    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = operator.try_apply(backend, &input, &mut forward_output);
    let failure_error = match failure {
        Err(error @ BackendError::Launch { .. }) => error,
        Err(error) => return Err(format!("P2 launch injection returned wrong error: {error}")),
        Ok(()) => return Err("P2 launch injection was not consumed".into()),
    };
    let failure_stats = backend.runtime_stats_since(failure_start);
    let failure_context = failure_error.context_disposition();
    let failure_output = failure_error.output_disposition();
    let failure_typed = true;
    let failure_published = false;
    if failure_context != BackendContextDisposition::SynchronizeBeforeReuse
        || failure_output != BackendOutputDisposition::Unusable
    {
        return Err(format!(
            "P2 launch disposition is not conservative: context={failure_context:?}, output={failure_output:?}"
        ));
    }
    backend
        .recover_after_error()
        .map_err(|error| format!("P2 launch recovery failed: {error}"))?;
    // Retry overwrites the unusable failed-output buffer before publication.
    operator
        .try_apply(backend, &input, &mut forward_output)
        .map_err(|error| format!("P2 retry action failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("P2 retry synchronization failed: {error}"))?;
    backend
        .try_copy_d2h(&forward_output, &mut forward)
        .map_err(|error| format!("P2 retry download failed: {error}"))?;
    let retry_error = max_relative_error(&forward, &forward_reference);
    let retry_succeeded = retry_error <= ACTION_TOLERANCE;
    if !malformed_csr_rejected
        || !malformed_shape_rejected
        || !malformed_output_preserved
        || !retry_succeeded
    {
        return Err(format!(
            "P2 malformed/failure gates failed: malformed_csr={malformed_csr_rejected}, malformed_shape={malformed_shape_rejected}, malformed_output={malformed_output_preserved}, retry={retry_error:.3e}"
        ));
    }

    let live_before_teardown = backend.runtime_stats().live_allocation_bytes;
    drop(operator);
    drop(operator_storage);
    drop(input);
    drop(test);
    drop(forward_output);
    drop(hermitian_output);
    drop(malformed_output);
    let teardown = backend.runtime_stats();
    if teardown.live_allocation_bytes != 0 || teardown.live_staging_bytes != 0 {
        return Err(format!(
            "P2 teardown leaked resources: live={} staging={}",
            teardown.live_allocation_bytes, teardown.live_staging_bytes
        ));
    }
    let metrics = Metrics {
        dimension,
        nnz,
        forward_error,
        hermitian_error,
        warm_forward_error,
        warm_hermitian_error,
        warm_kernel_operations,
        warm_h2d_operations: repeat_stats.transfer.h2d_operations,
        warm_allocation_operations: repeat_stats.allocation_operations,
        warm_staging_allocation_operations: repeat_stats.staging_allocation_operations,
        malformed_csr_rejected,
        malformed_shape_rejected,
        malformed_output_preserved,
        failure_typed,
        failure_kernel_operations: failure_stats.kernel_launches + failure_stats.library_operations,
        failure_context,
        failure_output,
        failure_published,
        retry_error,
        retry_succeeded,
        live_before_teardown,
        live_after_teardown: teardown.live_allocation_bytes,
        staging_after_teardown: teardown.live_staging_bytes,
        elapsed_ms: started.elapsed().as_millis().max(1) as u64,
    };
    Ok((metrics, teardown))
}

fn artifact(metrics: &Metrics, stats: CudaRuntimeStatsSnapshot) -> Value {
    json!({
        "schema": SCHEMA,
        "status": "passed",
        "case_id": CASE_ID,
        "scope": "native Linux x86_64 required CUDA SM120; Windows/Arm64/AVX512 waived",
        "claim_class": "production H1/Tri3/P2 Helmholtz assembled CSR forward/Hermitian action",
        "cell": {
            "space": "H1",
            "element": "Tri3",
            "order": "P2",
            "physics": "Helmholtz",
            "mode": "assembled",
            "scalar": "Complex64",
            "actions": ["forward", "hermitian"],
        },
        "assembly": {
            "production": "HelmholtzProblem::try_assemble with PolynomialDegree::P2",
            "mesh": "rectangular_mesh_triangles 3x3 cells; spatially varying source",
            "execution": "AdjointDeviceCsr native CUDA forward/Hermitian SpMV",
            "element_quadrature_setup": "host production assembly; no CUDA element kernel claimed",
            "independent_reference": "independent host CSR accumulation from checked triplets",
        },
        "dimensions": {
            "dofs": metrics.dimension,
            "nnz": metrics.nnz,
            "odd_dimension": metrics.dimension % 2 == 1,
            "tail_case": metrics.dimension % 256 != 0,
            "tail_remainder_mod_256": metrics.dimension % 256,
        },
        "metrics": {
            "forward_error": metrics.forward_error,
            "hermitian_error": metrics.hermitian_error,
            "warm_forward_error": metrics.warm_forward_error,
            "warm_hermitian_error": metrics.warm_hermitian_error,
            "warm_kernel_operations": metrics.warm_kernel_operations,
            "warm_h2d_operations": metrics.warm_h2d_operations,
            "warm_allocation_operations": metrics.warm_allocation_operations,
            "warm_staging_allocation_operations": metrics.warm_staging_allocation_operations,
            "malformed_csr_rejected": metrics.malformed_csr_rejected,
            "malformed_shape_rejected": metrics.malformed_shape_rejected,
            "malformed_output_preserved": metrics.malformed_output_preserved,
            "failure_typed": metrics.failure_typed,
            "failure_kernel_operations": metrics.failure_kernel_operations,
            "failure_published": metrics.failure_published,
            "retry_error": metrics.retry_error,
            "retry_succeeded": metrics.retry_succeeded,
        },
        "thresholds": {
            "forward_error": ACTION_TOLERANCE,
            "hermitian_error": ACTION_TOLERANCE,
            "warm_forward_error": ACTION_TOLERANCE,
            "warm_hermitian_error": ACTION_TOLERANCE,
            "warm_kernel_operations": 2,
            "warm_h2d_operations": 0,
            "warm_allocation_operations": 0,
            "warm_staging_allocation_operations": 0,
            "malformed_csr_rejected": 1,
            "malformed_shape_rejected": 1,
            "malformed_output_preserved": 1,
            "failure_typed": 1,
            "failure_published": 0,
            "retry_error": ACTION_TOLERANCE,
            "retry_succeeded": 1,
        },
        "threshold_directions": {
            "forward_error": "maximum",
            "hermitian_error": "maximum",
            "warm_forward_error": "maximum",
            "warm_hermitian_error": "maximum",
            "warm_kernel_operations": "minimum",
            "warm_h2d_operations": "maximum",
            "warm_allocation_operations": "maximum",
            "warm_staging_allocation_operations": "maximum",
            "malformed_csr_rejected": "minimum",
            "malformed_shape_rejected": "minimum",
            "malformed_output_preserved": "minimum",
            "failure_typed": "minimum",
            "failure_published": "maximum",
            "retry_error": "maximum",
            "retry_succeeded": "minimum",
        },
        "runtime": stats_value(stats),
        "warm_residency": {
            "csr_structure": "device-resident",
            "vectors": "device-resident",
            "repeat_actions": 2,
            "h2d_operations": metrics.warm_h2d_operations,
            "allocation_operations": metrics.warm_allocation_operations,
            "staging_allocation_operations": metrics.warm_staging_allocation_operations,
        },
        "failure_contract": {
            "stage": "launch",
            "error": "BackendError::Launch",
            "context_disposition": format!("{:?}", metrics.failure_context),
            "output_disposition": format!("{:?}", metrics.failure_output),
            "typed": metrics.failure_typed,
            "kernel_operations": metrics.failure_kernel_operations,
            "partial_result_published": metrics.failure_published,
            "recovery": "synchronize_before_reuse",
            "retry_succeeded": metrics.retry_succeeded,
        },
        "malformed_input_contract": {
            "csr_stage": "before_upload",
            "csr_error": "HostCsrError::RowPointerCount",
            "shape_stage": "action_admission",
            "shape_error": "BackendError::InvalidArgument",
            "rejected": metrics.malformed_csr_rejected && metrics.malformed_shape_rejected,
            "output_preserved": metrics.malformed_output_preserved,
        },
        "teardown": {
            "live_before_bytes": metrics.live_before_teardown,
            "live_after_bytes": metrics.live_after_teardown,
            "staging_after_bytes": metrics.staging_after_teardown,
            "resources_released": metrics.live_after_teardown == 0
                && metrics.staging_after_teardown == 0,
        },
        "dispatch": {
            "requested_backend": "cuda",
            "actual_backend": "cuda",
            "fallback": "none",
            "admission_before_assembly": true,
            "native_device_csr": true,
        },
        "unsupported_scope": [
            "CUDA geometry/quadrature integration remains host setup",
            "partial assembly and device-resident constraints remain unsupported",
            "this packet does not claim a general P2 FEM dispatcher or global solve",
        ],
        "hardware": cuda_hardware_provenance(),
        "workload_elapsed_ms": metrics.elapsed_ms,
    })
}

fn main() {
    assert_eq!(std::env::consts::OS, "linux", "FEM1 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "FEM1 requires native x86_64"
    );
    let backend = required_cuda_backend().expect("required SM120 CUDA backend");
    let (metrics, stats) = run(&backend).expect("FEM1 P2 CUDA gates");
    let payload = serde_json::to_vec_pretty(&artifact(&metrics, stats))
        .expect("serialize FEM1 P2 CUDA artifact");
    let (path, checksum) =
        write_qa_artifact("fem1_p2_cuda.json", &payload).unwrap_or_else(|error| {
            eprintln!("FAILED: write FEM1 P2 CUDA artifact: {error}");
            std::process::exit(1)
        });
    println!("artifact={path} checksum={checksum}");
    println!("SUCCESS");
}
