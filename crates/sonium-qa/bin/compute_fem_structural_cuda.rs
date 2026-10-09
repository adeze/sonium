//! Native CUDA parity for the bounded 3-D linear-elasticity action cell.
//!
//! Production `ElasticityForm` assembly supplies a constrained P1 tetrahedral
//! stiffness CSR.  CUDA owns the resident forward/Hermitian operator and
//! vectors; an independent host CSR accumulation is used only as the strict
//! reference.  This is an action-parity case, not a claim of CUDA quadrature
//! or a general structural solver.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use serde_json::{Value, json};
use sonium_compute::{
    AdjointDeviceCsr, Backend, BackendError, CudaBackend, CudaFailureInjectionPoint,
    CudaRuntimeStatsSnapshot, DeviceAdjointOperator, DeviceCsr, DeviceOperator, HostCsrRef,
};
use sonium_fem::assembly::assemble_bilinear_checked;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::unit_cube_tetrahedra;
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_fem::{
    FemAction, FemBackend, FemBackendRequest, FemScalar, FemWorkload, admit_fem_workload,
    fem_cuda_structural_capability_rows,
};
use sonium_qa::{
    QaExecutionCounts, QaFailureStage, QaResult, QaStatus, cuda_hardware_provenance,
    cuda_runtime_evidence, required_cuda_backend, write_qa_artifact,
};

const CASE_ID: &str = "compute.fem-structural-cuda.v1";
const ACTION_TOLERANCE: f64 = 2.0e-12;
const ENERGY_TOLERANCE: f64 = 2.0e-12;
const SYMMETRY_TOLERANCE: f64 = 2.0e-12;

#[derive(Debug)]
struct Metrics {
    dofs: usize,
    nnz: usize,
    forward_residual: f64,
    hermitian_residual: f64,
    energy_error: f64,
    energy: f64,
    symmetry_error: f64,
    device_inner_product_error: f64,
    warm_forward_residual: f64,
    warm_hermitian_residual: f64,
    warm_action_count: usize,
    warm_h2d_operations: usize,
    warm_allocation_operations: usize,
    failure_typed: bool,
    output_preserved: bool,
    retry_residual: f64,
    malformed_csr_rejected: bool,
    elapsed_ms: u64,
}

type CsrParts = (Vec<usize>, Vec<usize>, Vec<f64>);

fn csr_from_constrained_elasticity(
    dim: usize,
    rows: &[usize],
    cols: &[usize],
    values: &[f64],
    boundary_dofs: &BTreeSet<usize>,
) -> Result<CsrParts, String> {
    if rows.len() != cols.len() || rows.len() != values.len() {
        return Err("elasticity triplet arrays have different lengths".into());
    }
    let mut entries = BTreeMap::<(usize, usize), f64>::new();
    for ((&row, &col), &value) in rows.iter().zip(cols).zip(values) {
        if row >= dim || col >= dim || !value.is_finite() {
            return Err("elasticity triplet contains an invalid index or value".into());
        }
        // Homogeneous Dirichlet elimination: retain the free-free stiffness
        // block and replace every constrained row by an identity row.
        if !boundary_dofs.contains(&row) && !boundary_dofs.contains(&col) {
            *entries.entry((row, col)).or_default() += value;
        }
    }
    for &dof in boundary_dofs {
        entries.insert((dof, dof), 1.0);
    }
    let mut row_ptrs = vec![0usize; dim + 1];
    for &(row, _) in entries.keys() {
        row_ptrs[row + 1] += 1;
    }
    for row in 0..dim {
        row_ptrs[row + 1] += row_ptrs[row];
    }
    let mut col_indices = Vec::with_capacity(entries.len());
    let mut csr_values = Vec::with_capacity(entries.len());
    for row in 0..dim {
        for (&(entry_row, col), &value) in &entries {
            if entry_row == row {
                col_indices.push(col);
                csr_values.push(value);
            }
        }
    }
    Ok((row_ptrs, col_indices, csr_values))
}

fn csr_apply(row_ptrs: &[usize], col_indices: &[usize], values: &[f64], x: &[f64]) -> Vec<f64> {
    let mut output = vec![0.0; row_ptrs.len() - 1];
    for row in 0..output.len() {
        for index in row_ptrs[row]..row_ptrs[row + 1] {
            output[row] += values[index] * x[col_indices[index]];
        }
    }
    output
}

fn relative_l2(actual: &[f64], expected: &[f64]) -> f64 {
    let numerator = actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| (actual - expected).powi(2))
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

fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(&x, &y)| x * y).sum()
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
    // This is deliberately malformed before any device allocation: the
    // checked host-CSR boundary must reject the non-monotone row pointer.
    let row_ptrs = [0usize, 2, 1];
    let cols = [0usize, 1];
    let values = [1.0f64, 2.0];
    let host = HostCsrRef::try_new(2, 2, &row_ptrs, &cols, &values);
    if host.is_ok() {
        return false;
    }
    // Also exercise the typed device admission path on a valid shape with an
    // invalid column, without allowing a CPU fallback to mask the failure.
    let valid_rows = [0usize, 1, 2];
    let bad_cols = [0usize, 2];
    let valid_values = [1.0f64, 2.0];
    let Ok(host) = HostCsrRef::try_new(2, 2, &valid_rows, &bad_cols, &valid_values) else {
        return true;
    };
    matches!(
        DeviceCsr::<CudaBackend, f64>::try_from_host_csr_ref(backend, host),
        Err(BackendError::InvalidArgument { .. })
    )
}

fn run(backend: &CudaBackend) -> Result<(Metrics, CudaRuntimeStatsSnapshot), String> {
    // Admission is intentionally before mesh creation and production
    // assembly.  It is the dispatch proof for exactly this structural cell.
    for action in [
        FemAction::Forward,
        FemAction::Transpose,
        FemAction::Hermitian,
    ] {
        let admission = admit_fem_workload(
            FemWorkload::native_cuda_structural(FemScalar::F64, action),
            FemBackendRequest::required(FemBackend::Cuda),
        )
        .map_err(|error| format!("structural CUDA admission failed before assembly: {error}"))?;
        if admission.provenance.actual_backend != sonium_fem::FemActualBackend::Cuda
            || admission.provenance.fallback != "none"
        {
            return Err(format!(
                "structural admission did not prove native CUDA: {admission:?}"
            ));
        }
    }

    let started = Instant::now();
    let mut mesh = unit_cube_tetrahedra(2);
    mesh.detect_boundaries();
    let space = VectorH1Space::new(PolynomialDegree::P1, 3);
    let form = ElasticityForm::new_3d(IsotropicMaterial::new(210.0e9, 0.30));
    let stiffness = assemble_bilinear_checked(&form, &space, &space, &mesh, 1)
        .map_err(|error| format!("production linear-elasticity assembly failed: {error}"))?;
    let dim = space.num_global_dofs(&mesh);
    let boundary_dofs: BTreeSet<usize> = space.boundary_dofs(&mesh, 1).into_iter().collect();
    if boundary_dofs.len() >= dim {
        return Err("structural fixture has no free displacement degrees of freedom".into());
    }
    let (row_ptrs, col_indices, values) = csr_from_constrained_elasticity(
        dim,
        &stiffness.rows,
        &stiffness.cols,
        &stiffness.values,
        &boundary_dofs,
    )?;
    let host = HostCsrRef::try_new(dim, dim, &row_ptrs, &col_indices, &values)
        .map_err(|error| format!("structural CSR admission failed: {error}"))?;
    let operator = AdjointDeviceCsr::try_from_host_csr_ref(backend, host)
        .map_err(|error| format!("structural device CSR upload failed: {error}"))?;

    let x_host: Vec<_> = (0..dim)
        .map(|index| (index as f64 * 0.37 + 0.11).sin())
        .collect();
    let y_host: Vec<_> = (0..dim)
        .map(|index| (index as f64 * 0.23 + 0.41).cos())
        .collect();
    let forward_reference = csr_apply(&row_ptrs, &col_indices, &values, &x_host);
    let hermitian_reference = csr_apply(&row_ptrs, &col_indices, &values, &y_host);
    let energy = dot(&x_host, &forward_reference);
    if !energy.is_finite() || energy <= 0.0 {
        return Err(format!(
            "elasticity energy is not strictly positive: {energy}"
        ));
    }
    let symmetry_error = (dot(&forward_reference, &y_host) - dot(&x_host, &hermitian_reference))
        .abs()
        / dot(&forward_reference, &y_host)
            .abs()
            .max(dot(&x_host, &hermitian_reference).abs())
            .max(f64::MIN_POSITIVE);

    let mut x = backend
        .raw_alloc::<f64>(dim)
        .map_err(|error| format!("structural x allocation failed: {error}"))?;
    let mut y = backend
        .raw_alloc::<f64>(dim)
        .map_err(|error| format!("structural y allocation failed: {error}"))?;
    let mut ax = backend
        .raw_alloc::<f64>(dim)
        .map_err(|error| format!("structural Ax allocation failed: {error}"))?;
    let mut aty = backend
        .raw_alloc::<f64>(dim)
        .map_err(|error| format!("structural A^T y allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&x_host, &mut x)
        .map_err(|error| format!("structural x upload failed: {error}"))?;
    backend
        .try_copy_h2d(&y_host, &mut y)
        .map_err(|error| format!("structural y upload failed: {error}"))?;

    let initial_baseline = backend.begin_runtime_stats();
    operator
        .try_apply(backend, &x, &mut ax)
        .map_err(|error| format!("structural forward action failed: {error}"))?;
    operator
        .try_apply_hermitian(backend, &y, &mut aty)
        .map_err(|error| format!("structural Hermitian action failed: {error}"))?;
    let initial_stats = backend.runtime_stats_since(initial_baseline);
    let mut ax_host = vec![0.0; dim];
    let mut aty_host = vec![0.0; dim];
    backend
        .try_copy_d2h(&ax, &mut ax_host)
        .map_err(|error| format!("structural Ax download failed: {error}"))?;
    backend
        .try_copy_d2h(&aty, &mut aty_host)
        .map_err(|error| format!("structural A^T y download failed: {error}"))?;
    let forward_residual = relative_l2(&ax_host, &forward_reference);
    let hermitian_residual = relative_l2(&aty_host, &hermitian_reference);
    let energy_error = (dot(&x_host, &ax_host) - energy).abs() / energy.abs();
    let device_inner_product_error = (dot(&ax_host, &y_host) - dot(&x_host, &aty_host)).abs()
        / dot(&ax_host, &y_host)
            .abs()
            .max(dot(&x_host, &aty_host).abs())
            .max(f64::MIN_POSITIVE);

    // The same resident vectors and CSR descriptors are used for the warm
    // pair.  Only result publication is downloaded; no H2D upload is allowed.
    let warm_baseline = backend.begin_runtime_stats();
    operator
        .try_apply(backend, &x, &mut ax)
        .map_err(|error| format!("structural warm forward action failed: {error}"))?;
    operator
        .try_apply_hermitian(backend, &y, &mut aty)
        .map_err(|error| format!("structural warm Hermitian action failed: {error}"))?;
    let warm_stats = backend.runtime_stats_since(warm_baseline);
    let mut warm_ax_host = vec![0.0; dim];
    let mut warm_aty_host = vec![0.0; dim];
    backend
        .try_copy_d2h(&ax, &mut warm_ax_host)
        .map_err(|error| format!("structural warm Ax download failed: {error}"))?;
    backend
        .try_copy_d2h(&aty, &mut warm_aty_host)
        .map_err(|error| format!("structural warm A^T y download failed: {error}"))?;
    let warm_forward_residual = relative_l2(&warm_ax_host, &forward_reference);
    let warm_hermitian_residual = relative_l2(&warm_aty_host, &hermitian_reference);

    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = operator.try_apply(backend, &x, &mut ax);
    let failure_typed = matches!(&failure, Err(BackendError::Launch { .. }));
    if failure_typed {
        backend.clear_failure();
        backend
            .recover_after_error()
            .map_err(|error| format!("structural CUDA recovery failed: {error}"))?;
    } else {
        backend.clear_failure();
    }
    let mut after_failure = vec![0.0; dim];
    backend
        .try_copy_d2h(&ax, &mut after_failure)
        .map_err(|error| format!("structural output-preservation download failed: {error}"))?;
    let output_preserved = failure_typed
        && relative_l2(&after_failure, &warm_ax_host) == 0.0
        && backend
            .pending_error()
            .map_err(|error| error.to_string())?
            .is_none();
    operator
        .try_apply(backend, &x, &mut ax)
        .map_err(|error| format!("structural retry action failed: {error}"))?;
    let mut retry_host = vec![0.0; dim];
    backend
        .try_copy_d2h(&ax, &mut retry_host)
        .map_err(|error| format!("structural retry download failed: {error}"))?;
    let retry_residual = relative_l2(&retry_host, &forward_reference);
    let malformed = malformed_csr_rejected(backend);

    let metrics = Metrics {
        dofs: dim,
        nnz: values.len(),
        forward_residual,
        hermitian_residual,
        energy_error,
        energy,
        symmetry_error,
        device_inner_product_error,
        warm_forward_residual,
        warm_hermitian_residual,
        warm_action_count: 2,
        warm_h2d_operations: warm_stats.transfer.h2d_operations,
        warm_allocation_operations: warm_stats.allocation_operations,
        failure_typed,
        output_preserved,
        retry_residual,
        malformed_csr_rejected: malformed,
        elapsed_ms: started.elapsed().as_millis() as u64,
    };
    if metrics.forward_residual > ACTION_TOLERANCE
        || metrics.hermitian_residual > ACTION_TOLERANCE
        || metrics.warm_forward_residual > ACTION_TOLERANCE
        || metrics.warm_hermitian_residual > ACTION_TOLERANCE
        || metrics.retry_residual > ACTION_TOLERANCE
        || metrics.energy_error > ENERGY_TOLERANCE
        || metrics.symmetry_error > SYMMETRY_TOLERANCE
        || metrics.device_inner_product_error > SYMMETRY_TOLERANCE
        || !metrics.failure_typed
        || !metrics.output_preserved
        || !metrics.malformed_csr_rejected
        || metrics.warm_h2d_operations != 0
        || metrics.warm_action_count != 2
        || initial_stats.kernel_launches + initial_stats.library_operations == 0
    {
        return Err(format!("structural CUDA parity gates failed: {metrics:?}"));
    }
    Ok((metrics, backend.runtime_stats()))
}

fn write_artifact(
    metrics: &Metrics,
    stats: CudaRuntimeStatsSnapshot,
) -> Result<(String, String), String> {
    let capability_rows: Vec<_> = fem_cuda_structural_capability_rows()
        .into_iter()
        .map(|row| serde_json::to_value(row).expect("structural capability row serializes"))
        .collect();
    let payload = json!({
        "schema": "sonium-compute-cuda-fem-structural/v1",
        "status": "passed",
        "case_id": CASE_ID,
        "cell": {
            "space": "H1",
            "element": "tetrahedron",
            "order": "P1",
            "physics": "linear-elasticity",
            "mode": "assembled",
            "scalar": "f64",
            "actions": ["forward", "transpose", "hermitian"],
        },
        "matrix": {
            "dofs": metrics.dofs,
            "nnz": metrics.nnz,
            "fixture": "unit-cube-tetrahedra-2:p1-vector-h1:homogeneous-dirichlet-boundary",
            "assembly": "production ElasticityForm + assemble_bilinear_checked",
        },
        "metrics": {
            "forward_residual": metrics.forward_residual,
            "hermitian_residual": metrics.hermitian_residual,
            "energy_error": metrics.energy_error,
            "energy": metrics.energy,
            "symmetry_error": metrics.symmetry_error,
            "device_inner_product_error": metrics.device_inner_product_error,
            "warm_forward_residual": metrics.warm_forward_residual,
            "warm_hermitian_residual": metrics.warm_hermitian_residual,
            "warm_action_count": metrics.warm_action_count,
            "warm_h2d_operations": metrics.warm_h2d_operations,
            "warm_allocation_operations": metrics.warm_allocation_operations,
            "failure_typed": metrics.failure_typed,
            "output_preserved": metrics.output_preserved,
            "retry_residual": metrics.retry_residual,
            "malformed_csr_rejected": metrics.malformed_csr_rejected,
        },
        "thresholds": {
            "forward_residual": ACTION_TOLERANCE,
            "hermitian_residual": ACTION_TOLERANCE,
            "warm_forward_residual": ACTION_TOLERANCE,
            "warm_hermitian_residual": ACTION_TOLERANCE,
            "retry_residual": ACTION_TOLERANCE,
            "energy_error": ENERGY_TOLERANCE,
            "symmetry_error": SYMMETRY_TOLERANCE,
            "device_inner_product_error": SYMMETRY_TOLERANCE,
            "warm_action_count": 2,
            "warm_h2d_operations": 0,
            "failure_typed": 1,
            "output_preserved": 1,
            "malformed_csr_rejected": 1,
        },
        "threshold_directions": {
            "forward_residual": "maximum",
            "hermitian_residual": "maximum",
            "warm_forward_residual": "maximum",
            "warm_hermitian_residual": "maximum",
            "retry_residual": "maximum",
            "energy_error": "maximum",
            "symmetry_error": "maximum",
            "device_inner_product_error": "maximum",
            "warm_action_count": "minimum",
            "warm_h2d_operations": "maximum",
            "failure_typed": "minimum",
            "output_preserved": "minimum",
            "malformed_csr_rejected": "minimum",
        },
        "workload_elapsed_ms": metrics.elapsed_ms.max(1),
        "runtime": stats_value(stats),
        "warm_residency": {
            "operator_storage": "device-resident",
            "input_vectors": "device-resident",
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
                "evidence": "one-shot native CUDA launch injection plus D2H comparison",
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
        ],
        "capability_rows": capability_rows,
        "hardware": cuda_hardware_provenance(),
    });
    let bytes = serde_json::to_vec_pretty(&payload)
        .map_err(|error| format!("serialize structural CUDA artifact: {error}"))?;
    write_qa_artifact("compute_fem_structural_cuda.json", &bytes)
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
    .reference("independent host CSR accumulation from production ElasticityForm matrix")
    .reference_checksum("analytic/deterministic-fixture:compute-fem-structural-cuda-v1")
    .dimension("3-D unit cube tetrahedra(2), vector H1 P1, f64 linear elasticity")
    .formulation("production ElasticityForm assembly plus native CUDA DeviceCsr forward/Hermitian SpMV")
    .tolerance("action residual and energy/symmetry gates <= 2e-12; warm actions resident; typed launch recovery")
    .seed("unit-cube-tetrahedra-2:p1-vector-h1:E=210e9:nu=0.30")
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
        Ok((metrics, stats)) => {
            let artifact = write_artifact(metrics, *stats);
            match artifact {
                Ok((path, checksum)) => result
                    .backend_passed("cuda")
                    .artifact_with_provenance(
                        path,
                        "crates/sonium-qa/bin/compute_fem_structural_cuda.rs",
                        "native SM120 CUDA structural CSR action parity",
                        "sonium-compute-cuda-fem-structural/v1",
                        checksum,
                    )
                    .metric("dofs", metrics.dofs as f64)
                    .metric("nnz", metrics.nnz as f64)
                    .metric("forward_residual", metrics.forward_residual)
                    .metric("hermitian_residual", metrics.hermitian_residual)
                    .metric("energy_error", metrics.energy_error)
                    .metric("symmetry_error", metrics.symmetry_error),
                Err(error) => {
                    result = result
                        .backend_failed("cuda", error.clone())
                        .failure_stage(QaFailureStage::Artifact);
                    result.message = Some(error);
                    result
                }
            }
        }
        Err(error) => result
            .backend_failed("cuda", error.clone())
            .failure_stage(QaFailureStage::Execution),
    };
    result
}

fn main() {
    let started = Instant::now();
    let result = match required_cuda_backend() {
        Ok(backend) => {
            backend.reset_runtime_stats();
            qa_result(run(&backend), started)
        }
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cuda")
            .actual_backend("unavailable")
            .failure_stage(QaFailureStage::Environment)
            .runtime_evidence(cuda_runtime_evidence(
                CudaRuntimeStatsSnapshot::default(),
                QaExecutionCounts {
                    executed: 1,
                    failed: 1,
                    ..QaExecutionCounts::default()
                },
            )),
    };
    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .unwrap_or_else(|error| format!("{{\"error\":\"{error}\"}}"))
    );
    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        eprintln!("COMPUTE FEM STRUCTURAL CUDA FAILED: {:?}", result.message);
        println!("FAILED");
        std::process::exit(1);
    }
}
