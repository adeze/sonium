//! End-to-end device-resident FEM forward/adjoint workload.
//!
//! This case assembles the production legacy Helmholtz problem, uploads one
//! deterministic CSR operator and its Hermitian pair to the compute backend,
//! and checks both the inner-product identity and column-major block parity.
//! The host is used only for setup and final evidence; applications stay on
//! device buffers.

use std::collections::BTreeMap;
use std::time::Instant;

use num_complex::Complex64;
use sonium_compute::{
    AdjointDeviceCsr, Backend, CpuBackend, DeviceAdjointOperator, DeviceOperator, HostCsrRef,
};
use sonium_fem::assembly::HelmholtzProblem;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::{
    FemAction, FemBackend, FemBackendRequest, FemScalar, FemWorkload, admit_fem_workload,
    fem_cuda_capability_rows,
};
use sonium_qa::QaResult;

#[cfg(feature = "cuda")]
use sonium_compute::CudaRuntimeStatsSnapshot;
#[cfg(feature = "cuda")]
use sonium_qa::{
    QaExecutionCounts, QaFailureStage, QaStatus, cuda_hardware_provenance, cuda_runtime_evidence,
    required_cuda_backend, write_qa_artifact,
};

const CPU_CASE_ID: &str = "compute.fem-adjoint.v1";
#[cfg(feature = "cuda")]
const CUDA_CASE_ID: &str = "compute.fem-adjoint-cuda.v1";
const INNER_PRODUCT_TOLERANCE: f64 = 2.0e-12;
const BLOCK_TOLERANCE: f64 = 2.0e-13;
const RESIDUAL_TOLERANCE: f64 = 2.0e-12;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    dofs: usize,
    nnz: usize,
    inner_product_error: f64,
    block_error: f64,
    forward_residual: f64,
    adjoint_residual: f64,
    f64_inner_product_error: f64,
    f64_forward_residual: f64,
    f64_adjoint_residual: f64,
    elapsed_ms: u64,
}

type CsrParts = (Vec<usize>, Vec<usize>, Vec<Complex64>);

fn case_id() -> &'static str {
    #[cfg(feature = "cuda")]
    if std::env::var("SONIUM_REQUIRE_BACKEND").as_deref() == Ok("cuda") {
        return CUDA_CASE_ID;
    }
    CPU_CASE_ID
}

fn csr_from_triplets(
    dim: usize,
    rows: &[usize],
    cols: &[usize],
    values: &[Complex64],
) -> Result<CsrParts, String> {
    if rows.len() != cols.len() || rows.len() != values.len() {
        return Err("Helmholtz triplet arrays have different lengths".into());
    }
    let mut entries = BTreeMap::<(usize, usize), Complex64>::new();
    for ((&row, &col), &value) in rows.iter().zip(cols).zip(values) {
        if row >= dim || col >= dim || !value.re.is_finite() || !value.im.is_finite() {
            return Err("Helmholtz triplet contains an invalid index or value".into());
        }
        *entries.entry((row, col)).or_default() += value;
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

fn max_relative_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| (actual - expected).norm() / expected.norm().max(1.0e-30))
        .fold(0.0, f64::max)
}

fn max_relative_error_f64(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| (actual - expected).abs() / expected.abs().max(1.0e-30))
        .fold(0.0, f64::max)
}

fn dotc(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    x.iter().zip(y).map(|(&x, &y)| x.conj() * y).sum()
}

fn run_real_forward_adjoint<B: Backend>(
    backend: &B,
    dim: usize,
    row_ptrs: &[usize],
    col_indices: &[usize],
    values: &[Complex64],
) -> Result<(f64, f64, f64), String> {
    let real_values: Vec<_> = values.iter().map(|value| value.re).collect();
    let host = HostCsrRef::try_new(dim, dim, row_ptrs, col_indices, &real_values)
        .map_err(|error| format!("real CSR admission failed: {error}"))?;
    let operator = AdjointDeviceCsr::<B, f64>::try_from_host_csr_ref(backend, host)
        .map_err(|error| format!("real device CSR upload failed: {error}"))?;
    let x_host: Vec<_> = (0..dim).map(|index| (index as f64 + 1.0).sin()).collect();
    let y_host: Vec<_> = (0..dim).map(|index| (index as f64 * 0.23).cos()).collect();
    let mut x = backend
        .raw_alloc::<f64>(dim)
        .map_err(|error| format!("real x allocation failed: {error}"))?;
    let mut y = backend
        .raw_alloc::<f64>(dim)
        .map_err(|error| format!("real y allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&x_host, &mut x)
        .map_err(|error| format!("real x upload failed: {error}"))?;
    backend
        .try_copy_h2d(&y_host, &mut y)
        .map_err(|error| format!("real y upload failed: {error}"))?;
    let mut ax = backend
        .raw_alloc::<f64>(dim)
        .map_err(|error| format!("real Ax allocation failed: {error}"))?;
    let mut aty = backend
        .raw_alloc::<f64>(dim)
        .map_err(|error| format!("real Aty allocation failed: {error}"))?;
    operator
        .try_apply(backend, &x, &mut ax)
        .map_err(|error| format!("real forward apply failed: {error}"))?;
    operator
        .try_apply_hermitian(backend, &y, &mut aty)
        .map_err(|error| format!("real transpose apply failed: {error}"))?;
    let mut ax_host = vec![0.0; dim];
    let mut aty_host = vec![0.0; dim];
    backend
        .try_copy_d2h(&ax, &mut ax_host)
        .map_err(|error| format!("real Ax download failed: {error}"))?;
    backend
        .try_copy_d2h(&aty, &mut aty_host)
        .map_err(|error| format!("real Aty download failed: {error}"))?;
    let mut forward_reference = vec![0.0; dim];
    let mut adjoint_reference = vec![0.0; dim];
    for row in 0..dim {
        for index in row_ptrs[row]..row_ptrs[row + 1] {
            forward_reference[row] += real_values[index] * x_host[col_indices[index]];
            adjoint_reference[col_indices[index]] += real_values[index] * y_host[row];
        }
    }
    let forward_residual = max_relative_error_f64(&ax_host, &forward_reference);
    let adjoint_residual = max_relative_error_f64(&aty_host, &adjoint_reference);
    let lhs: f64 = ax_host.iter().zip(&y_host).map(|(&ax, &y)| ax * y).sum();
    let rhs: f64 = x_host.iter().zip(&aty_host).map(|(&x, &aty)| x * aty).sum();
    let denominator = lhs.abs().max(rhs.abs()).max(f64::MIN_POSITIVE);
    let inner_product_error = (lhs - rhs).abs() / denominator;
    if !inner_product_error.is_finite() {
        return Err("real f64 inner product is non-finite".into());
    }
    Ok((forward_residual, adjoint_residual, inner_product_error))
}

fn run_with_backend<B: Backend>(
    backend: &B,
    requested_backend: FemBackend,
) -> Result<Metrics, String> {
    // This is intentionally before mesh creation and production FEM
    // assembly. Unsupported scalar/space/order/form/action requests therefore
    // cannot become a CPU matrix that is later mislabeled as CUDA evidence.
    for (scalar, action) in [
        (FemScalar::Complex64, FemAction::Forward),
        (FemScalar::Complex64, FemAction::Hermitian),
        (FemScalar::F64, FemAction::Forward),
        (FemScalar::F64, FemAction::Hermitian),
        (FemScalar::F64, FemAction::Transpose),
    ] {
        admit_fem_workload(
            FemWorkload::native_cuda(scalar, action),
            FemBackendRequest::required(requested_backend),
        )
        .map_err(|error| format!("FEM backend admission failed before assembly: {error}"))?;
    }
    let started = Instant::now();
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 3, 3);
    let problem = HelmholtzProblem::try_assemble(
        &mesh,
        PolynomialDegree::P1,
        Complex64::new(2.25, 0.07),
        |x, y, _| Complex64::new(1.0 + x - 0.25 * y, 0.5 * x + y),
    )
    .map_err(|error| format!("FEM assembly failed: {error}"))?;
    let dim = problem.num_dofs();
    let (row_ptrs, col_indices, values) = csr_from_triplets(
        dim,
        &problem.matrix.rows,
        &problem.matrix.cols,
        &problem.matrix.values,
    )?;
    let host = HostCsrRef::try_new(dim, dim, &row_ptrs, &col_indices, &values)
        .map_err(|error| format!("CSR admission failed: {error}"))?;

    let operator = AdjointDeviceCsr::try_from_host_csr_ref(backend, host)
        .map_err(|error| format!("device CSR upload failed: {error}"))?;

    let x_host: Vec<_> = (0..dim)
        .map(|index| Complex64::new((index as f64 + 1.0).sin(), (index as f64 * 0.17).cos()))
        .collect();
    let y_host: Vec<_> = (0..dim)
        .map(|index| Complex64::new((index as f64 * 0.23).cos(), (index as f64 + 0.5).sin()))
        .collect();
    let mut x = backend
        .raw_alloc::<Complex64>(dim)
        .map_err(|error| format!("x allocation failed: {error}"))?;
    let mut y = backend
        .raw_alloc::<Complex64>(dim)
        .map_err(|error| format!("y allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&x_host, &mut x)
        .map_err(|error| format!("x upload failed: {error}"))?;
    backend
        .try_copy_h2d(&y_host, &mut y)
        .map_err(|error| format!("y upload failed: {error}"))?;
    let mut ax = backend
        .raw_alloc::<Complex64>(dim)
        .map_err(|error| format!("Ax allocation failed: {error}"))?;
    let mut ahy = backend
        .raw_alloc::<Complex64>(dim)
        .map_err(|error| format!("Ahy allocation failed: {error}"))?;
    let mut ahx = backend
        .raw_alloc::<Complex64>(dim)
        .map_err(|error| format!("Ahx allocation failed: {error}"))?;
    operator
        .try_apply(backend, &x, &mut ax)
        .map_err(|error| format!("forward apply failed: {error}"))?;
    operator
        .try_apply_hermitian(backend, &y, &mut ahy)
        .map_err(|error| format!("Hermitian apply failed: {error}"))?;
    operator
        .try_apply_hermitian(backend, &x, &mut ahx)
        .map_err(|error| format!("Hermitian x apply failed: {error}"))?;
    let mut ax_host = vec![Complex64::default(); dim];
    let mut ahy_host = vec![Complex64::default(); dim];
    let mut ahx_host = vec![Complex64::default(); dim];
    backend
        .try_copy_d2h(&ax, &mut ax_host)
        .map_err(|error| format!("Ax download failed: {error}"))?;
    backend
        .try_copy_d2h(&ahy, &mut ahy_host)
        .map_err(|error| format!("Ahy download failed: {error}"))?;
    backend
        .try_copy_d2h(&ahx, &mut ahx_host)
        .map_err(|error| format!("Ahx download failed: {error}"))?;

    let inner_product_error = (dotc(&ax_host, &y_host) - dotc(&x_host, &ahy_host)).norm()
        / dotc(&ax_host, &y_host).norm().max(1.0e-30);

    let columns = [x_host.as_slice(), y_host.as_slice()].concat();
    let mut block_input = backend
        .raw_alloc::<Complex64>(2 * dim)
        .map_err(|error| format!("block input allocation failed: {error}"))?;
    let mut block_output = backend
        .raw_alloc::<Complex64>(2 * dim)
        .map_err(|error| format!("block output allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&columns, &mut block_input)
        .map_err(|error| format!("block input upload failed: {error}"))?;
    operator
        .try_apply_hermitian_colmajor(backend, 2, &block_input, &mut block_output)
        .map_err(|error| format!("block Hermitian apply failed: {error}"))?;
    let mut block_host = vec![Complex64::default(); 2 * dim];
    backend
        .try_copy_d2h(&block_output, &mut block_host)
        .map_err(|error| format!("block output download failed: {error}"))?;
    let mut expected_block = vec![Complex64::default(); 2 * dim];
    expected_block[..dim].copy_from_slice(&ahx_host);
    let mut second_input = backend
        .raw_alloc::<Complex64>(dim)
        .map_err(|error| format!("second RHS allocation failed: {error}"))?;
    let mut second_output = backend
        .raw_alloc::<Complex64>(dim)
        .map_err(|error| format!("second output allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&y_host, &mut second_input)
        .map_err(|error| format!("second RHS upload failed: {error}"))?;
    operator
        .try_apply_hermitian(backend, &second_input, &mut second_output)
        .map_err(|error| format!("second Hermitian apply failed: {error}"))?;
    let mut second_host = vec![Complex64::default(); dim];
    backend
        .try_copy_d2h(&second_output, &mut second_host)
        .map_err(|error| format!("second output download failed: {error}"))?;
    expected_block[dim..].copy_from_slice(&second_host);
    let block_error = max_relative_error(&block_host, &expected_block);

    let mut forward_reference = vec![Complex64::default(); dim];
    let mut adjoint_reference = vec![Complex64::default(); dim];
    for row in 0..dim {
        for index in row_ptrs[row]..row_ptrs[row + 1] {
            forward_reference[row] += values[index] * x_host[col_indices[index]];
            adjoint_reference[col_indices[index]] += values[index].conj() * y_host[row];
        }
    }
    let forward_residual = max_relative_error(&ax_host, &forward_reference);
    let adjoint_residual = max_relative_error(&ahy_host, &adjoint_reference);

    if !inner_product_error.is_finite()
        || !block_error.is_finite()
        || !forward_residual.is_finite()
        || !adjoint_residual.is_finite()
    {
        return Err("device FEM workload produced a non-finite metric".into());
    }
    if inner_product_error > INNER_PRODUCT_TOLERANCE {
        return Err(format!(
            "Hermitian inner-product identity failed: {inner_product_error:.3e}"
        ));
    }
    if block_error > BLOCK_TOLERANCE {
        return Err(format!("Hermitian block parity failed: {block_error:.3e}"));
    }
    if forward_residual > RESIDUAL_TOLERANCE || adjoint_residual > RESIDUAL_TOLERANCE {
        return Err(format!(
            "device CSR reference residual failed: forward {forward_residual:.3e}, adjoint {adjoint_residual:.3e}"
        ));
    }
    let (f64_forward_residual, f64_adjoint_residual, f64_inner_product_error) =
        run_real_forward_adjoint(backend, dim, &row_ptrs, &col_indices, &values)?;
    if f64_forward_residual > RESIDUAL_TOLERANCE
        || f64_adjoint_residual > RESIDUAL_TOLERANCE
        || f64_inner_product_error > INNER_PRODUCT_TOLERANCE
    {
        return Err(format!(
            "real CSR reference contract failed: forward {f64_forward_residual:.3e}, adjoint {f64_adjoint_residual:.3e}, inner {f64_inner_product_error:.3e}"
        ));
    }
    Ok(Metrics {
        dofs: dim,
        nnz: values.len(),
        inner_product_error,
        block_error,
        forward_residual,
        adjoint_residual,
        f64_inner_product_error,
        f64_forward_residual,
        f64_adjoint_residual,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

fn add_metrics(mut record: QaResult, metrics: &Metrics) -> QaResult {
    let action_count = metrics.nnz.saturating_mul(4);
    let elapsed_seconds = (metrics.elapsed_ms.max(1) as f64) / 1_000.0;
    record = record
        .metric("dofs", metrics.dofs as f64)
        .metric("nnz", metrics.nnz as f64)
        .metric("operation_count", action_count as f64)
        .metric(
            "complex_csr_flop_count",
            action_count.saturating_mul(8) as f64,
        )
        .metric(
            "operation_throughput_per_second",
            action_count as f64 / elapsed_seconds,
        )
        .metric("inner_product_error", metrics.inner_product_error)
        .metric("block_error", metrics.block_error)
        .metric("forward_residual", metrics.forward_residual)
        .metric("adjoint_residual", metrics.adjoint_residual)
        .metric("f64_forward_residual", metrics.f64_forward_residual)
        .metric("f64_adjoint_residual", metrics.f64_adjoint_residual)
        .metric("f64_inner_product_error", metrics.f64_inner_product_error)
        .metric("workload_elapsed_ms", metrics.elapsed_ms as f64)
        .maximum("inner_product_error", INNER_PRODUCT_TOLERANCE)
        .maximum("block_error", BLOCK_TOLERANCE)
        .maximum("forward_residual", RESIDUAL_TOLERANCE)
        .maximum("adjoint_residual", RESIDUAL_TOLERANCE)
        .maximum("f64_forward_residual", RESIDUAL_TOLERANCE)
        .maximum("f64_adjoint_residual", RESIDUAL_TOLERANCE)
        .maximum("f64_inner_product_error", INNER_PRODUCT_TOLERANCE)
        .minimum("operation_count", 1.0)
        .seed("rectangular-triangles-3x3:p1:k=2.25+0.07i");
    record
}

#[cfg(feature = "cuda")]
fn cuda_capability_environment() -> BTreeMap<String, String> {
    let mut capabilities = BTreeMap::new();
    for (key, value) in [
        ("capability.fem.space.h1", "native"),
        ("capability.fem.space.hcurl", "unsupported"),
        ("capability.fem.space.hdiv", "unsupported"),
        ("capability.fem.space.dg", "unsupported"),
        ("capability.fem.element.tri3", "native"),
        ("capability.fem.element.quad", "unsupported"),
        ("capability.fem.element.tetrahedron", "unsupported"),
        ("capability.fem.element.hexahedron", "unsupported"),
        ("capability.fem.order.p1", "native"),
        ("capability.fem.order.p2", "native"),
        ("capability.fem.order.p2_or_higher", "unsupported"),
        ("capability.fem.form.helmholtz", "native"),
        ("capability.fem.form.other", "unsupported"),
        ("capability.fem.mode.assembled", "native"),
        ("capability.fem.mode.partial_assembly", "unsupported"),
        ("capability.fem.mode.matrix_free", "unsupported"),
        ("capability.fem.scalar.f64", "native"),
        ("capability.fem.scalar.c64", "native"),
        ("capability.fem.scalar.f32", "unsupported"),
        ("capability.fem.scalar.c32", "unsupported"),
        ("capability.fem.action.forward", "native"),
        // The device-adjoint API exposes Hermitian application.  For real
        // f64 this is exactly transpose; Complex64 transpose is not exposed
        // and is represented explicitly in the structured rows below.
        ("capability.fem.action.transpose", "native_real_equivalent"),
        ("capability.fem.action.hermitian", "native"),
        ("capability.fem.batching.element_quadrature", "unsupported"),
        ("capability.fem.residency.vectors", "device"),
        ("capability.fem.residency.coefficients", "device"),
        ("capability.fem.residency.constraints", "host_staged"),
        ("capability.fem.residency.quadrature", "host_setup"),
        ("capability.fem.fallback.required_cuda", "none"),
        ("capability.fem.unsupported_admission", "before_assembly"),
    ] {
        capabilities.insert(key.to_owned(), value.to_owned());
    }
    capabilities
}

#[cfg(feature = "cuda")]
fn cuda_capability_rows() -> Vec<serde_json::Value> {
    fem_cuda_capability_rows()
        .into_iter()
        .map(|row| serde_json::to_value(row).expect("FEM capability row is serializable"))
        .collect()
}

#[cfg(feature = "cuda")]
fn write_cuda_evidence(
    metrics: &Metrics,
    stats: CudaRuntimeStatsSnapshot,
) -> Result<(String, String), String> {
    let action_count = metrics.nnz.saturating_mul(4);
    let complex_flop_count = action_count.saturating_mul(8);
    let workload_elapsed_ms = metrics.elapsed_ms.max(1);
    let throughput_flops_per_second =
        complex_flop_count as f64 / (workload_elapsed_ms as f64 / 1_000.0);
    let peak_flops = std::env::var("SONIUM_QA_CUDA_PEAK_FLOPS")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0);
    let roofline_fraction = peak_flops
        .map(|peak| throughput_flops_per_second / peak)
        .filter(|value| value.is_finite() && *value >= 0.0);
    let mut environment = serde_json::Map::new();
    for (key, value) in cuda_capability_environment() {
        environment.insert(key, serde_json::Value::String(value));
    }
    environment.insert(
        "performance.roofline_fraction".into(),
        serde_json::json!(
            roofline_fraction.map_or_else(|| "unavailable".into(), |value| value.to_string())
        ),
    );
    environment.insert(
        "vendor_baseline.cusparse".into(),
        serde_json::json!("exercised for device CSR forward, transpose, and Hermitian products"),
    );
    environment.insert(
        "vendor_baseline.cufft".into(),
        serde_json::json!("not_applicable_to_fem_workload"),
    );
    let payload = serde_json::json!({
        "schema": "sonium-compute-cuda-fem-adjoint/v1",
        "status": "passed",
        "case_id": CUDA_CASE_ID,
        "scalar_types": ["f64", "Complex64"],
        "operation_count": action_count,
        "complex_csr_flop_count": complex_flop_count,
        "throughput_flops_per_second": throughput_flops_per_second,
        "workload_elapsed_ms": workload_elapsed_ms,
        "metrics": {
            "operation_count": action_count,
            "inner_product_error": metrics.inner_product_error,
            "block_error": metrics.block_error,
            "complex64_forward_residual": metrics.forward_residual,
            "complex64_adjoint_residual": metrics.adjoint_residual,
            "f64_forward_residual": metrics.f64_forward_residual,
            "f64_adjoint_residual": metrics.f64_adjoint_residual,
            "f64_inner_product_error": metrics.f64_inner_product_error,
            "throughput_flops_per_second": throughput_flops_per_second,
        },
        "thresholds": {
            "inner_product_error": INNER_PRODUCT_TOLERANCE,
            "block_error": BLOCK_TOLERANCE,
            "complex64_forward_residual": RESIDUAL_TOLERANCE,
            "complex64_adjoint_residual": RESIDUAL_TOLERANCE,
            "f64_forward_residual": RESIDUAL_TOLERANCE,
            "f64_adjoint_residual": RESIDUAL_TOLERANCE,
            "f64_inner_product_error": INNER_PRODUCT_TOLERANCE,
            "operation_count": 1.0,
        },
        "threshold_directions": {
            "inner_product_error": "maximum",
            "block_error": "maximum",
            "complex64_forward_residual": "maximum",
            "complex64_adjoint_residual": "maximum",
            "f64_forward_residual": "maximum",
            "f64_adjoint_residual": "maximum",
            "f64_inner_product_error": "maximum",
            "operation_count": "minimum",
        },
        "runtime": {
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
        },
        "environment": environment,
        "capability_rows": cuda_capability_rows(),
        "failure_contracts": [
            {
                "id": "required_cuda.initialization",
                "status": "fail_closed",
                "stage": "initialization",
                "requested_backend": "cuda",
                "actual_backend": "unavailable",
                "actual_backend_on_failure": "unavailable",
                "fallback": "none",
                "fallback_reason": "required CUDA initialization failed; no fallback requested",
                "dispatch": false,
                "evidence": "required_cuda_backend and unavailable_cuda_result",
            },
            {
                "id": "unsupported_cell.admission",
                "status": "reject_before_assembly",
                "stage": "admission",
                "actual_backend": "none",
                "actual_backend_on_failure": "none",
                "requested_backend": "cuda",
                "fallback": "none",
                "fallback_reason": "no fallback requested; rejected before assembly",
                "dispatch": false,
                "evidence": "structured capability rows are not promoted to assembly",
            },
        ],
        "hardware": cuda_hardware_provenance(),
    });
    let bytes = serde_json::to_vec_pretty(&payload)
        .map_err(|error| format!("serialize CUDA FEM evidence: {error}"))?;
    write_qa_artifact("compute_fem_cuda_adjoint.json", &bytes)
}

#[cfg(feature = "cuda")]
fn cuda_result(run_result: Result<Metrics, String>, stats: CudaRuntimeStatsSnapshot) -> QaResult {
    let mut record = match &run_result {
        Ok(_) => QaResult::passed(case_id()),
        Err(error) => QaResult::failed(case_id(), error.clone()),
    }
    .requested_backend("cuda")
    .actual_backend("cuda")
    .reference("independent CPU CSR accumulation and f64/Complex64 adjoint reference")
    .reference_checksum("analytic/deterministic-fixture:compute-fem-adjoint-cuda-v1")
    .dimension("2-D rectangular triangles P1, deterministic complex wavenumber")
    .formulation("production HelmholtzProblem assembly plus CUDA DeviceCsr forward and Hermitian SpMV/SpMM for f64 and Complex64")
    .tolerance("inner-product 2e-12; block 2e-13; f64/Complex64 residuals 2e-12; nonzero native CUDA work and artifact provenance")
    .runtime_evidence(cuda_runtime_evidence(
        stats,
        QaExecutionCounts {
            executed: 1,
            passed: u64::from(run_result.is_ok()),
            failed: u64::from(run_result.is_err()),
            ..QaExecutionCounts::default()
        },
    ));
    record = match &run_result {
        Ok(_) => record.backend_passed("cuda"),
        Err(error) => record
            .backend_failed("cuda", error.clone())
            .failure_stage(QaFailureStage::Execution),
    };
    if let Ok(metrics) = &run_result {
        record = add_metrics(record, metrics);
        for (key, value) in cuda_capability_environment() {
            record.environment.insert(key, value);
        }
        if std::env::var_os("SONIUM_OUTPUT_DIR").is_some() {
            match write_cuda_evidence(metrics, stats) {
                Ok((path, checksum)) => {
                    record = record.artifact_with_provenance(
                        path,
                        "crates/sonium-qa/bin/compute_fem_adjoint.rs",
                        "runtime CUDA FEM forward/adjoint evidence",
                        "sonium-compute-cuda-fem-adjoint/v1",
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
    }
    record
}

fn cpu_result(run_result: Result<Metrics, String>) -> QaResult {
    let mut record = match &run_result {
        Ok(_) => QaResult::passed(case_id()),
        Err(error) => QaResult::failed(case_id(), error.clone()),
    }
    .requested_backend("cpu")
    .actual_backend("cpu");
    if let Ok(metrics) = &run_result {
        record = add_metrics(record, metrics);
    }
    record
}

#[cfg(feature = "cuda")]
fn unavailable_cuda_result(error: String) -> QaResult {
    let mut record = QaResult::failed(CUDA_CASE_ID, error)
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
    let result = {
        #[cfg(feature = "cuda")]
        {
            if std::env::var("SONIUM_REQUIRE_BACKEND").as_deref() == Ok("cuda") {
                match required_cuda_backend() {
                    Ok(backend) => {
                        backend.reset_runtime_stats();
                        let run_result = run_with_backend(&backend, FemBackend::Cuda);
                        cuda_result(run_result, backend.runtime_stats())
                    }
                    Err(error) => unavailable_cuda_result(error),
                }
            } else {
                cpu_result(run_with_backend(&CpuBackend::new(), FemBackend::Cpu))
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            cpu_result(run_with_backend(&CpuBackend::new(), FemBackend::Cpu))
        }
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);
    if let Ok(json) = result.to_json_line() {
        println!("QA_RESULT: {json}");
    }
    if matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        eprintln!("COMPUTE FEM ADJOINT FAILED: {:?}", result.message);
        println!("FAILED");
        std::process::exit(1);
    }
}
