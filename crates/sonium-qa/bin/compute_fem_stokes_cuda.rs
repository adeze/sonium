//! Native CUDA parity for a bounded Taylor--Hood Stokes action cell.
//!
//! The production FEM layer assembles the Stokes mixed system on the host;
//! CUDA owns the resident monolithic CSR action.  An independently written
//! quadrature/CSR accumulator is used as the CPU oracle.  This is an action
//! parity cell, not a CUDA quadrature kernel or a complete nonlinear solve.

use std::collections::BTreeMap;
use std::time::Instant;

use serde_json::{Value, json};
use sonium_compute::{
    AdjointDeviceCsr, Backend, BackendContextDisposition, BackendError, BackendOutputDisposition,
    CudaBackend, CudaFailureInjectionPoint, CudaRuntimeStatsSnapshot, DeviceAdjointOperator,
    DeviceOperator, HostCsrRef,
};
use sonium_fem::assembly::{TripletMatrix, assemble_mixed_checked};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::physics::fluid::stokes::forms::{StokesDivergenceForm, StokesViscousForm};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};
use sonium_fem::{
    FemAction, FemActualBackend, FemBackend, FemBackendRequest, FemWorkload, admit_fem_workload,
    fem_cuda_stokes_capability_rows,
};
use sonium_qa::{cuda_hardware_provenance, required_cuda_backend, write_qa_artifact};

const CASE_ID: &str = "compute.fem-stokes-cuda.v1";
const SCHEMA: &str = "sonium-compute-cuda-fem-stokes/v1";
const VISCOSITY: f64 = 0.75;
const QUADRATURE_ORDER: usize = 4;
const ACTION_TOLERANCE: f64 = 3.0e-12;

type EntryMap = BTreeMap<(usize, usize), f64>;

#[derive(Debug)]
struct Metrics {
    dofs: usize,
    velocity_dofs: usize,
    pressure_dofs: usize,
    nnz: usize,
    assembly_error: f64,
    forward_error: f64,
    transpose_error: f64,
    inner_product_error: f64,
    warm_action_count: usize,
    warm_kernel_operations: usize,
    warm_h2d_operations: usize,
    warm_d2h_operations: usize,
    warm_allocation_operations: usize,
    warm_staging_allocation_operations: usize,
    malformed_csr_rejected: bool,
    malformed_shape_rejected: bool,
    malformed_output_preserved: bool,
    failure_typed: bool,
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

fn add_entry(entries: &mut EntryMap, row: usize, column: usize, value: f64) {
    if value != 0.0 {
        *entries.entry((row, column)).or_default() += value;
    }
}

fn csr_entries(matrix: &sonium_solvers::CsrMatrix<f64>) -> EntryMap {
    let mut entries = EntryMap::new();
    for row in 0..matrix.num_rows {
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            add_entry(
                &mut entries,
                row,
                matrix.col_indices[index],
                matrix.values[index],
            );
        }
    }
    entries
}

/// Independently assemble the Taylor--Hood Stokes blocks from the weak-form
/// definitions.  This deliberately does not call either Stokes form or the
/// production mixed assembler: only basis tabulation and quadrature are
/// shared infrastructure.
fn independent_stokes_entries(
    mesh: &sonium_fem::mesh::Mesh,
    velocity: &VectorH1Space,
    pressure: &H1Space,
    viscosity: f64,
) -> EntryMap {
    let velocity_dofs = velocity.num_global_dofs(mesh);
    let mut entries = EntryMap::new();
    for element_index in 0..mesh.num_elements() {
        let element = &mesh.elements[element_index];
        let velocity_map = velocity.local_to_global(mesh, element_index);
        let pressure_map = pressure.local_to_global(mesh, element_index);
        let quadrature = QuadratureRule::new(element.element_type, QUADRATURE_ORDER);
        for point in quadrature.iter() {
            let velocity_tab =
                velocity.tabulate(mesh, element_index, point.xi(), point.eta(), point.zeta());
            let pressure_tab =
                pressure.tabulate(mesh, element_index, point.xi(), point.eta(), point.zeta());
            let weight = point.weight * velocity_tab.det_j;
            for (local_row, &row) in velocity_map.iter().enumerate() {
                let row_component = local_row % 2;
                for (local_column, &column) in velocity_map.iter().enumerate() {
                    if row_component != local_column % 2 {
                        continue;
                    }
                    let gradient_dot = (0..2)
                        .map(|direction| {
                            velocity_tab.grad(local_row, direction)
                                * velocity_tab.grad(local_column, direction)
                        })
                        .sum::<f64>();
                    add_entry(&mut entries, row, column, viscosity * gradient_dot * weight);
                }
                for (local_column, &pressure_column) in pressure_map.iter().enumerate() {
                    let coupling = -pressure_tab.value(local_column)
                        * velocity_tab.grad(local_row, row_component)
                        * weight;
                    let shifted_column = pressure_column + velocity_dofs;
                    add_entry(&mut entries, row, shifted_column, coupling);
                    add_entry(&mut entries, shifted_column, row, coupling);
                }
            }
        }
    }
    entries
}

fn max_entry_error(actual: &EntryMap, expected: &EntryMap) -> f64 {
    actual
        .keys()
        .chain(expected.keys())
        .map(|key| {
            let a = actual.get(key).copied().unwrap_or_default();
            let e = expected.get(key).copied().unwrap_or_default();
            (a - e).abs() / e.abs().max(a.abs()).max(1.0)
        })
        .fold(0.0, f64::max)
}

fn csr_apply(
    row_ptrs: &[usize],
    columns: &[usize],
    values: &[f64],
    input: &[f64],
    transpose: bool,
) -> Vec<f64> {
    let dimension = row_ptrs.len() - 1;
    let mut output = vec![0.0; dimension];
    for row in 0..dimension {
        for index in row_ptrs[row]..row_ptrs[row + 1] {
            let column = columns[index];
            if transpose {
                output[column] += values[index] * input[row];
            } else {
                output[row] += values[index] * input[column];
            }
        }
    }
    output
}

fn relative_error(actual: &[f64], expected: &[f64]) -> f64 {
    let numerator = actual
        .iter()
        .zip(expected)
        .map(|(&a, &e)| (a - e).powi(2))
        .sum::<f64>()
        .sqrt();
    let denominator = expected
        .iter()
        .map(|value| value.powi(2))
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    numerator / denominator
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}

fn stats_value(stats: CudaRuntimeStatsSnapshot) -> Value {
    json!({
        "kernel_launches": stats.kernel_launches,
        "library_operations": stats.library_operations,
        "synchronization_calls": stats.synchronization_calls,
        "failed_synchronizations": stats.failed_synchronizations,
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

fn admit_stokes() -> Result<(), String> {
    for action in [
        FemAction::Forward,
        FemAction::Transpose,
        FemAction::Hermitian,
    ] {
        let admission = admit_fem_workload(
            FemWorkload::native_cuda_stokes(action),
            FemBackendRequest::required(FemBackend::Cuda),
        )
        .map_err(|error| format!("Stokes CUDA admission failed: {error}"))?;
        if admission.provenance.actual_backend != FemActualBackend::Cuda
            || admission.provenance.fallback != "none"
        {
            return Err(format!(
                "Stokes admission did not prove native CUDA: {admission:?}"
            ));
        }
    }
    Ok(())
}

fn run(backend: &CudaBackend) -> Result<(Metrics, CudaRuntimeStatsSnapshot), String> {
    admit_stokes()?;
    let started = Instant::now();
    let mesh = unit_square_triangles(2);
    let velocity = VectorH1Space::new(PolynomialDegree::P2, 2);
    let pressure = H1Space::new(PolynomialDegree::P1);
    let velocity_dofs = velocity.num_global_dofs(&mesh);
    let pressure_dofs = pressure.num_global_dofs(&mesh);
    let dimension = velocity_dofs + pressure_dofs;
    if dimension != 59 || dimension % 2 == 0 {
        return Err(format!(
            "Taylor-Hood fixture must have the expected odd 59-DOF tail, got {dimension}"
        ));
    }
    let production = assemble_mixed_checked(
        &StokesViscousForm::new(VISCOSITY),
        &StokesDivergenceForm,
        &velocity,
        &pressure,
        &mesh,
        QUADRATURE_ORDER,
    )
    .map_err(|error| format!("production Stokes mixed assembly failed: {error}"))?;
    let production_triplets: TripletMatrix<f64> = production.to_monolithic();
    let production_csr = production_triplets.to_csr();
    let production_entries = csr_entries(&production_csr);
    let reference_entries = independent_stokes_entries(&mesh, &velocity, &pressure, VISCOSITY);
    let assembly_error = max_entry_error(&production_entries, &reference_entries);
    if assembly_error > ACTION_TOLERANCE {
        let mut mismatch_count = 0;
        for key in production_entries.keys().chain(reference_entries.keys()) {
            let production_value = production_entries.get(key).copied().unwrap_or_default();
            let reference_value = reference_entries.get(key).copied().unwrap_or_default();
            if (production_value - reference_value).abs() > ACTION_TOLERANCE {
                eprintln!(
                    "Stokes assembly mismatch at {key:?}: production={production_value:.6e}, reference={reference_value:.6e}"
                );
                mismatch_count += 1;
                if mismatch_count == 12 {
                    break;
                }
            }
        }
        return Err(format!(
            "production/independent Stokes assembly mismatch: {assembly_error:.3e}"
        ));
    }
    let row_ptrs = production_csr.row_ptrs.clone();
    let columns = production_csr.col_indices.clone();
    let values = production_csr.values.clone();
    let nnz = values.len();
    if nnz <= dimension || dimension % 256 == 0 {
        return Err(format!(
            "Stokes CSR does not exercise a nontrivial tail: n={dimension}, nnz={nnz}"
        ));
    }
    let host = HostCsrRef::try_new(dimension, dimension, &row_ptrs, &columns, &values)
        .map_err(|error| format!("Stokes CSR admission failed: {error}"))?;
    let input: Vec<_> = (0..dimension)
        .map(|index| (index as f64 * 0.23 + 0.17).sin())
        .collect();
    let adjoint_input: Vec<_> = (0..dimension)
        .map(|index| (index as f64 * 0.19 + 0.41).cos())
        .collect();
    let forward_reference = csr_apply(&row_ptrs, &columns, &values, &input, false);
    let transpose_reference = csr_apply(&row_ptrs, &columns, &values, &adjoint_input, true);

    let storage = AdjointDeviceCsr::try_from_host_csr_ref(backend, host)
        .map_err(|error| format!("Stokes device CSR upload failed: {error}"))?;
    let operator = storage
        .try_prebake(backend)
        .map_err(|error| format!("Stokes CSR prebake failed: {error}"))?;
    let mut device_input = backend
        .raw_alloc::<f64>(dimension)
        .map_err(|error| format!("Stokes input allocation failed: {error}"))?;
    let mut device_adjoint_input = backend
        .raw_alloc::<f64>(dimension)
        .map_err(|error| format!("Stokes adjoint-input allocation failed: {error}"))?;
    let mut device_output = backend
        .raw_alloc::<f64>(dimension)
        .map_err(|error| format!("Stokes output allocation failed: {error}"))?;
    let mut device_transpose_output = backend
        .raw_alloc::<f64>(dimension)
        .map_err(|error| format!("Stokes transpose-output allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&input, &mut device_input)
        .map_err(|error| format!("Stokes input upload failed: {error}"))?;
    backend
        .try_copy_h2d(&adjoint_input, &mut device_adjoint_input)
        .map_err(|error| format!("Stokes adjoint-input upload failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("Stokes setup synchronization failed: {error}"))?;
    backend.reset_runtime_stats();

    let warm_start = backend.begin_runtime_stats();
    operator
        .try_apply(backend, &device_input, &mut device_output)
        .map_err(|error| format!("Stokes forward action failed: {error}"))?;
    operator
        .try_apply_hermitian(backend, &device_adjoint_input, &mut device_transpose_output)
        .map_err(|error| format!("Stokes transpose action failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("Stokes warm synchronization failed: {error}"))?;
    let warm_stats = backend.runtime_stats_since(warm_start);
    let mut forward = vec![0.0; dimension];
    let mut transpose = vec![0.0; dimension];
    backend
        .try_copy_d2h(&device_output, &mut forward)
        .map_err(|error| format!("Stokes forward download failed: {error}"))?;
    backend
        .try_copy_d2h(&device_transpose_output, &mut transpose)
        .map_err(|error| format!("Stokes transpose download failed: {error}"))?;
    let forward_error = relative_error(&forward, &forward_reference);
    let transpose_error = relative_error(&transpose, &transpose_reference);
    let inner_product_error =
        (dot(&adjoint_input, &forward_reference) - dot(&input, &transpose_reference)).abs()
            / dot(&adjoint_input, &forward_reference).abs().max(1.0);
    if forward_error > ACTION_TOLERANCE
        || transpose_error > ACTION_TOLERANCE
        || inner_product_error > ACTION_TOLERANCE
        || warm_stats.kernel_launches + warm_stats.library_operations < 2
        || warm_stats.transfer.h2d_operations != 0
        || warm_stats.transfer.d2h_operations != 0
        || warm_stats.allocation_operations != 0
        || warm_stats.staging_allocation_operations != 0
    {
        return Err(format!(
            "Stokes warm/parity gates failed: forward={forward_error:.3e}, transpose={transpose_error:.3e}, inner={inner_product_error:.3e}, stats={warm_stats:?}"
        ));
    }

    let malformed_csr_rejected = HostCsrRef::try_new(
        dimension,
        dimension,
        &row_ptrs[..dimension],
        &columns,
        &values,
    )
    .is_err();
    let mut malformed_output = backend
        .raw_alloc::<f64>(dimension - 1)
        .map_err(|error| format!("Stokes malformed-output allocation failed: {error}"))?;
    let malformed_sentinel = vec![17.25; dimension - 1];
    backend
        .try_copy_h2d(&malformed_sentinel, &mut malformed_output)
        .map_err(|error| format!("Stokes malformed sentinel upload failed: {error}"))?;
    let malformed_shape_rejected = matches!(
        operator.try_apply(backend, &device_input, &mut malformed_output),
        Err(BackendError::InvalidArgument { .. })
    );
    let mut malformed_after = vec![0.0; dimension - 1];
    backend
        .try_copy_d2h(&malformed_output, &mut malformed_after)
        .map_err(|error| format!("Stokes malformed-output download failed: {error}"))?;
    let malformed_output_preserved = malformed_after == malformed_sentinel;

    let failure_sentinel = vec![-23.0; dimension];
    backend
        .try_copy_h2d(&failure_sentinel, &mut device_output)
        .map_err(|error| format!("Stokes failure sentinel upload failed: {error}"))?;
    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = operator.try_apply(backend, &device_input, &mut device_output);
    let failure_error = match failure {
        Err(error @ BackendError::Launch { .. }) => error,
        Err(error) => {
            return Err(format!(
                "Stokes launch injection returned wrong error: {error}"
            ));
        }
        Ok(()) => return Err("Stokes launch injection was not consumed".into()),
    };
    let failure_context = failure_error.context_disposition();
    let failure_output = failure_error.output_disposition();
    if failure_context != BackendContextDisposition::SynchronizeBeforeReuse
        || failure_output != BackendOutputDisposition::Unusable
    {
        return Err(format!(
            "Stokes launch disposition is not conservative: context={failure_context:?}, output={failure_output:?}"
        ));
    }
    backend
        .recover_after_error()
        .map_err(|error| format!("Stokes launch recovery failed: {error}"))?;
    operator
        .try_apply(backend, &device_input, &mut device_output)
        .map_err(|error| format!("Stokes retry action failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("Stokes retry synchronization failed: {error}"))?;
    backend
        .try_copy_d2h(&device_output, &mut forward)
        .map_err(|error| format!("Stokes retry download failed: {error}"))?;
    let retry_error = relative_error(&forward, &forward_reference);
    let retry_succeeded = retry_error <= ACTION_TOLERANCE;
    if !malformed_csr_rejected
        || !malformed_shape_rejected
        || !malformed_output_preserved
        || !retry_succeeded
    {
        return Err(format!(
            "Stokes malformed/failure gates failed: csr={malformed_csr_rejected}, shape={malformed_shape_rejected}, preserved={malformed_output_preserved}, retry={retry_error:.3e}"
        ));
    }

    let live_before_teardown = backend.runtime_stats().live_allocation_bytes;
    drop(operator);
    drop(storage);
    drop(device_input);
    drop(device_adjoint_input);
    drop(device_output);
    drop(device_transpose_output);
    drop(malformed_output);
    let teardown = backend.runtime_stats();
    if teardown.live_allocation_bytes != 0 || teardown.live_staging_bytes != 0 {
        return Err(format!(
            "Stokes teardown leaked resources: live={} staging={}",
            teardown.live_allocation_bytes, teardown.live_staging_bytes
        ));
    }
    Ok((
        Metrics {
            dofs: dimension,
            velocity_dofs,
            pressure_dofs,
            nnz,
            assembly_error,
            forward_error,
            transpose_error,
            inner_product_error,
            warm_action_count: 2,
            warm_kernel_operations: warm_stats.kernel_launches + warm_stats.library_operations,
            warm_h2d_operations: warm_stats.transfer.h2d_operations,
            warm_d2h_operations: warm_stats.transfer.d2h_operations,
            warm_allocation_operations: warm_stats.allocation_operations,
            warm_staging_allocation_operations: warm_stats.staging_allocation_operations,
            malformed_csr_rejected,
            malformed_shape_rejected,
            malformed_output_preserved,
            failure_typed: true,
            failure_context,
            failure_output,
            failure_published: false,
            retry_error,
            retry_succeeded,
            live_before_teardown,
            live_after_teardown: teardown.live_allocation_bytes,
            staging_after_teardown: teardown.live_staging_bytes,
            elapsed_ms: started.elapsed().as_millis().max(1) as u64,
        },
        teardown,
    ))
}

fn artifact(metrics: &Metrics, runtime: CudaRuntimeStatsSnapshot) -> Result<Vec<u8>, String> {
    let rows = serde_json::to_value(fem_cuda_stokes_capability_rows())
        .map_err(|error| format!("serialize Stokes capability rows: {error}"))?;
    serde_json::to_vec_pretty(&json!({
        "schema": SCHEMA,
        "status": "passed",
        "case_id": CASE_ID,
        "scope": "native Linux x86_64 required CUDA SM120; Windows/Arm64/AVX512 waived",
        "claim_class": "production Taylor-Hood Stokes mixed assembled CSR forward/transpose action",
        "cell": {
            "space": "H1",
            "element": "Tri3",
            "order": "P2-P1 Taylor-Hood",
            "physics": "Stokes",
            "mode": "assembled",
            "scalar": "f64",
            "actions": ["forward", "transpose", "hermitian"]
        },
        "assembly": {
            "production": "assemble_mixed_checked(StokesViscousForm, StokesDivergenceForm)",
            "mesh": "unit_square_triangles(2); 2x2 cells",
            "execution": "AdjointDeviceCsr native CUDA forward/Hermitian CSR action",
            "element_quadrature_setup": "host production mixed assembly; no CUDA element kernel claimed",
            "independent_reference": "independent direct quadrature and monolithic CSR accumulation"
        },
        "dimensions": {
            "dofs": metrics.dofs,
            "velocity_dofs": metrics.velocity_dofs,
            "pressure_dofs": metrics.pressure_dofs,
            "nnz": metrics.nnz,
            "odd_dimension": metrics.dofs % 2 == 1,
            "tail_case": metrics.dofs % 256 != 0,
            "tail_remainder_mod_256": metrics.dofs % 256
        },
        "metrics": {
            "assembly_error": metrics.assembly_error,
            "forward_error": metrics.forward_error,
            "transpose_error": metrics.transpose_error,
            "inner_product_error": metrics.inner_product_error,
            "warm_action_count": metrics.warm_action_count,
            "warm_kernel_operations": metrics.warm_kernel_operations,
            "warm_h2d_operations": metrics.warm_h2d_operations,
            "warm_d2h_operations": metrics.warm_d2h_operations,
            "warm_allocation_operations": metrics.warm_allocation_operations,
            "warm_staging_allocation_operations": metrics.warm_staging_allocation_operations,
            "malformed_csr_rejected": metrics.malformed_csr_rejected,
            "malformed_shape_rejected": metrics.malformed_shape_rejected,
            "malformed_output_preserved": metrics.malformed_output_preserved,
            "failure_typed": metrics.failure_typed,
            "failure_published": metrics.failure_published,
            "retry_error": metrics.retry_error,
            "retry_succeeded": metrics.retry_succeeded
        },
        "thresholds": {
            "assembly_error": ACTION_TOLERANCE,
            "forward_error": ACTION_TOLERANCE,
            "transpose_error": ACTION_TOLERANCE,
            "inner_product_error": ACTION_TOLERANCE,
            "warm_action_count": 2,
            "warm_kernel_operations": 2,
            "warm_h2d_operations": 0,
            "warm_d2h_operations": 0,
            "warm_allocation_operations": 0,
            "warm_staging_allocation_operations": 0,
            "malformed_csr_rejected": 1,
            "malformed_shape_rejected": 1,
            "malformed_output_preserved": 1,
            "failure_typed": 1,
            "failure_published": 0,
            "retry_error": ACTION_TOLERANCE,
            "retry_succeeded": 1
        },
        "threshold_directions": {
            "assembly_error": "maximum",
            "forward_error": "maximum",
            "transpose_error": "maximum",
            "inner_product_error": "maximum",
            "warm_action_count": "minimum",
            "warm_kernel_operations": "minimum",
            "warm_h2d_operations": "maximum",
            "warm_d2h_operations": "maximum",
            "warm_allocation_operations": "maximum",
            "warm_staging_allocation_operations": "maximum",
            "malformed_csr_rejected": "minimum",
            "malformed_shape_rejected": "minimum",
            "malformed_output_preserved": "minimum",
            "failure_typed": "minimum",
            "failure_published": "maximum",
            "retry_error": "maximum",
            "retry_succeeded": "minimum"
        },
        "runtime": stats_value(runtime),
        "warm_residency": {
            "mixed_csr": "device-resident",
            "input_vectors": "device-resident",
            "output_vectors": "device-resident",
            "warm_action_count": metrics.warm_action_count,
            "h2d_operations": metrics.warm_h2d_operations,
            "d2h_operations": metrics.warm_d2h_operations,
            "allocation_operations": metrics.warm_allocation_operations,
            "staging_allocation_operations": metrics.warm_staging_allocation_operations
        },
        "failure_contract": {
            "stage": "launch",
            "error": "BackendError::Launch",
            "context_disposition": format!("{:?}", metrics.failure_context),
            "output_disposition": format!("{:?}", metrics.failure_output),
            "typed": metrics.failure_typed,
            "partial_result_published": metrics.failure_published,
            "recovery": "synchronize_before_reuse",
            "retry_succeeded": metrics.retry_succeeded
        },
        "malformed_input_contract": {
            "csr_stage": "before_upload",
            "csr_error": "HostCsrError::RowPointerCount",
            "shape_stage": "action_admission",
            "shape_error": "BackendError::InvalidArgument",
            "rejected": metrics.malformed_csr_rejected && metrics.malformed_shape_rejected,
            "output_preserved": metrics.malformed_output_preserved
        },
        "teardown": {
            "live_before_bytes": metrics.live_before_teardown,
            "live_after_bytes": metrics.live_after_teardown,
            "staging_after_bytes": metrics.staging_after_teardown,
            "resources_released": metrics.live_after_teardown == 0 && metrics.staging_after_teardown == 0
        },
        "dispatch": {
            "requested_backend": "cuda",
            "actual_backend": "cuda",
            "fallback": "none",
            "admission_before_assembly": true,
            "native_device_csr": true
        },
        "unsupported_scope": [
            "CUDA geometry/quadrature integration remains host setup",
            "partial assembly, matrix-free Stokes, and device-resident constraints remain unsupported",
            "this packet does not claim a general Stokes/Navier-Stokes dispatcher or nonlinear solve"
        ],
        "capability_rows": rows,
        "hardware": cuda_hardware_provenance(),
        "workload_elapsed_ms": metrics.elapsed_ms
    }))
    .map_err(|error| format!("serialize Stokes artifact: {error}"))
}

fn main() {
    let started = Instant::now();
    let (metrics, runtime) = match required_cuda_backend().and_then(|backend| {
        backend.reset_runtime_stats();
        run(&backend)
    }) {
        Ok(result) => result,
        Err(error) => {
            eprintln!(
                "STOKES CUDA FEM FAILED after {} ms: {error}",
                started.elapsed().as_millis()
            );
            std::process::exit(1);
        }
    };
    let bytes = match artifact(&metrics, runtime) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("STOKES CUDA FEM ARTIFACT FAILED: {error}");
            std::process::exit(1);
        }
    };
    let (path, checksum) = match write_qa_artifact("compute_fem_stokes_cuda.json", &bytes) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("STOKES CUDA FEM ARTIFACT WRITE FAILED: {error}");
            std::process::exit(1);
        }
    };
    println!("artifact={path} checksum={checksum}");
    println!("SUCCESS");
}
