//! CUDA device-residency gate for the compressed BEM operator.
//!
//! The fixture contains both dense near-field and rank-one far-field blocks.
//! It compares the CUDA forward, algebraic-transpose, and Hermitian actions
//! with the checked CPU H-matrix actions and asserts that the steady-state
//! actions perform no host transfers.

use num_complex::Complex64;
use sonium_bem::compute_scattered_field_checked;
use sonium_bem::compute_scattered_field_cuda_checked;
use sonium_bem::core::assembly::tbem::build_tbem_system_stabilized_with_beta;
use sonium_bem::core::assembly::{
    HMatrix, HMatrixAction, build_cluster_tree, build_hmatrix,
    build_mlfmm_system_stabilized_checked,
};
use sonium_bem::core::types::{
    BoundaryCondition, Cluster, Element, ElementProperty, ElementType, PhysicsParams,
};
use sonium_compute::{
    Backend, BackendError, CudaBackend, CudaFailureInjectionPoint, MemoryPool, MemoryPoolConfig,
    Transpose,
};
use sonium_qa::{
    QaExecutionCounts, QaFailureStage, QaResult, cuda_hardware_provenance, cuda_runtime_evidence,
    required_cuda_backend, write_qa_artifact,
};
use std::time::Instant;

const CASE_ID: &str = "bem.cuda-hmatrix-fmm-device-residency.v1";
const TOLERANCE: f64 = 1.0e-10;
const DENSE_RESIDUAL_TOLERANCE: f64 = 1.0e-8;

fn failed_cuda(error: String, actual_backend: &str, stage: QaFailureStage) -> QaResult {
    QaResult::failed(CASE_ID, error.clone())
        .requested_backend("cuda")
        .actual_backend(actual_backend)
        .backend_failed("cuda", error)
        .failure_stage(stage)
        .hardware_provenance(cuda_hardware_provenance())
}

fn fixture() -> (Vec<Element>, Vec<[f64; 3]>, Vec<Cluster>, PhysicsParams) {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.5, 1.0, 0.0],
        [3.0, 0.0, 1.0],
    ];
    let element = |connectivity: Vec<usize>, center: [f64; 3], dof| Element {
        connectivity,
        element_type: ElementType::Tri3,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0; 3]; 3],
        center,
        area: 0.5,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses: vec![dof],
    };
    let elements = vec![
        element(vec![0, 1, 2], [0.5, 0.33, 0.0], 0),
        element(vec![1, 3, 2], [1.17, 0.33, 0.0], 1),
        element(vec![3, 4, 2], [1.83, 0.33, 0.0], 2),
        element(vec![3, 5, 4], [2.17, 0.33, 0.33], 3),
    ];
    let mut left = Cluster::new([0.83, 0.33, 0.0]);
    left.element_indices = vec![0, 1];
    left.num_elements = 2;
    left.radius = 1.0;
    left.near_clusters = vec![0];
    left.far_clusters = vec![1];
    let mut right = Cluster::new([2.0, 0.33, 0.17]);
    right.element_indices = vec![2, 3];
    right.num_elements = 2;
    right.radius = 1.0;
    right.near_clusters = vec![1];
    right.far_clusters = vec![0];
    let physics = PhysicsParams::new(250.0, 343.0, 1.21, false);
    (elements, nodes, vec![left, right], physics)
}

fn relative_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    if actual.len() != expected.len()
        || actual
            .iter()
            .chain(expected)
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return f64::INFINITY;
    }
    let numerator = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator = expected.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
    let error = numerator / denominator.max(f64::MIN_POSITIVE);
    if error.is_finite() {
        error
    } else {
        f64::INFINITY
    }
}

fn complex_absolute_error(actual: Complex64, expected: Complex64) -> f64 {
    if !actual.re.is_finite()
        || !actual.im.is_finite()
        || !expected.re.is_finite()
        || !expected.im.is_finite()
    {
        return f64::INFINITY;
    }
    let error = (actual - expected).norm();
    if error.is_finite() {
        error
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_metrics_reject_non_finite_values() {
        assert!(
            relative_error(
                &[Complex64::new(f64::NAN, 0.0)],
                &[Complex64::new(1.0, 0.0)]
            )
            .is_infinite(),
            "non-finite vector parity must fail closed"
        );
        assert!(
            complex_absolute_error(Complex64::new(0.0, f64::INFINITY), Complex64::default())
                .is_infinite(),
            "non-finite inner-product parity must fail closed"
        );
    }
}

fn cpu_action(hmatrix: &HMatrix, probe: &[Complex64], action: HMatrixAction) -> Vec<Complex64> {
    match action {
        HMatrixAction::Forward => hmatrix
            .matvec_checked(probe)
            .expect("valid forward H-matrix"),
        HMatrixAction::Transpose => hmatrix
            .matvec_transpose_checked(probe)
            .expect("valid transpose H-matrix"),
        HMatrixAction::Hermitian => hmatrix
            .matvec_hermitian_checked(probe)
            .expect("valid Hermitian H-matrix"),
    }
}

fn dense_action(
    matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
    probe: &[Complex64],
    action: HMatrixAction,
) -> Vec<Complex64> {
    let size = matrix.nrows();
    (0..size)
        .map(|row| {
            (0..size)
                .map(|column| {
                    let entry = match action {
                        HMatrixAction::Forward => matrix[[row, column]],
                        HMatrixAction::Transpose => matrix[[column, row]],
                        HMatrixAction::Hermitian => matrix[[column, row]].conj(),
                    };
                    entry * probe[column]
                })
                .sum()
        })
        .collect()
}

fn run(backend: CudaBackend) -> Result<QaResult, String> {
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(512 << 20))
        .map_err(|error| format!("CUDA H-matrix pool: {error}"))?;
    let backend = pool.backend();
    if !backend.has_fmm_c64() {
        return Err("CUDA Complex64 FMM module is unavailable".into());
    }

    let (elements, nodes, clusters, physics) = fixture();
    let started = Instant::now();
    let hmatrix = build_hmatrix(&elements, &nodes, &clusters, &physics, 1.0e-10, 16);
    let mlfmm_levels = build_cluster_tree(&elements, 1, &physics);
    let mlfmm =
        build_mlfmm_system_stabilized_checked(&elements, &nodes, mlfmm_levels, &physics, 1.0, 1.0)
            .map_err(|error| format!("build CPU MLFMM reference: {error}"))?;
    let cuda_mlfmm = mlfmm
        .try_to_cuda(pool.clone())
        .map_err(|error| format!("upload MLFMM hierarchy to CUDA: {error}"))?;
    let dense = build_tbem_system_stabilized_with_beta(
        &elements,
        &nodes,
        &physics,
        sonium_bem::core::types::PhysicsParams::optimal_formulation_sign(physics.wave_number),
        physics.burton_miller_beta(),
    );
    let cuda_hmatrix = hmatrix
        .try_to_cuda(pool.clone())
        .map_err(|error| format!("upload H-matrix to CUDA: {error}"))?;
    let setup_ms = started.elapsed().as_millis() as u64;

    let probe = vec![
        Complex64::new(0.2, -0.1),
        Complex64::new(-0.4, 0.3),
        Complex64::new(0.7, 0.2),
        Complex64::new(0.1, -0.5),
    ];
    let mut input = pool
        .try_alloc::<Complex64>(probe.len())
        .map_err(|error| error.to_string())?;
    let mut output = pool
        .try_alloc::<Complex64>(probe.len())
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_h2d(&probe, input.raw_mut())
        .map_err(|error| format!("upload probe: {error}"))?;

    let mlfmm_cpu = [
        mlfmm
            .matvec_checked(&probe)
            .map_err(|error| format!("CPU MLFMM forward action: {error}"))?,
        mlfmm
            .matvec_transpose_checked(&probe)
            .map_err(|error| format!("CPU MLFMM transpose action: {error}"))?,
        mlfmm
            .apply_hermitian_checked(&probe)
            .map_err(|error| format!("CPU MLFMM Hermitian action: {error}"))?,
    ];
    backend.reset_runtime_stats();
    let mlfmm_started = Instant::now();
    let mut mlfmm_errors = Vec::with_capacity(3);
    for (action, expected) in [
        (Transpose::NoTrans, &mlfmm_cpu[0]),
        (Transpose::Trans, &mlfmm_cpu[1]),
        (Transpose::ConjTrans, &mlfmm_cpu[2]),
    ] {
        cuda_mlfmm
            .try_apply_device(&input, &mut output, action)
            .map_err(|error| format!("CUDA MLFMM {action:?} action: {error}"))?;
        let mut actual = vec![Complex64::new(0.0, 0.0); probe.len()];
        backend
            .try_copy_d2h(output.raw(), &mut actual)
            .map_err(|error| format!("read back CUDA MLFMM {action:?}: {error}"))?;
        mlfmm_errors.push(relative_error(&actual, expected));
    }
    let mlfmm_error = mlfmm_errors.iter().copied().fold(0.0, f64::max);
    let mlfmm_elapsed_ms = mlfmm_started.elapsed().as_millis() as u64;
    let mlfmm_stats = backend.runtime_stats();
    if mlfmm_error > TOLERANCE {
        return Err(format!(
            "CUDA/CPU MLFMM action error {mlfmm_error:.6e} exceeds {TOLERANCE:.6e}"
        ));
    }
    if mlfmm_stats.transfer.h2d_operations != 0 || mlfmm_stats.transfer.d2h_operations != 3 {
        return Err(format!(
            "unexpected MLFMM steady-state transfers: h2d={}, d2h={}",
            mlfmm_stats.transfer.h2d_operations, mlfmm_stats.transfer.d2h_operations
        ));
    }
    if mlfmm_stats.kernel_launches == 0 {
        return Err("CUDA MLFMM action launched no device kernels".into());
    }
    let dual_probe = vec![
        Complex64::new(-0.3, 0.4),
        Complex64::new(0.6, -0.2),
        Complex64::new(0.15, 0.8),
        Complex64::new(-0.5, -0.1),
    ];
    let mut dual_input = pool
        .try_alloc::<Complex64>(dual_probe.len())
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_h2d(&dual_probe, dual_input.raw_mut())
        .map_err(|error| format!("upload reciprocity probe: {error}"))?;

    // A warm replay updates only a caller-owned resident RHS slot.  The
    // structure, transfer coefficients, near blocks, and workspaces remain
    // owned by the operator; the slot is refreshed device-to-device so the
    // replay has no structural allocation or host-to-device transfer.
    let mut mlfmm_warm_rhs = pool
        .try_alloc::<Complex64>(probe.len())
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_d2d(input.raw(), mlfmm_warm_rhs.raw_mut())
        .map_err(|error| format!("seed resident MLFMM RHS slot: {error}"))?;
    let sentinel = vec![Complex64::new(73.0, -11.0); probe.len()];
    backend
        .try_copy_h2d(&sentinel, output.raw_mut())
        .map_err(|error| format!("seed MLFMM failure sentinel: {error}"))?;
    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = cuda_mlfmm
        .try_apply_device(&mlfmm_warm_rhs, &mut output, Transpose::NoTrans)
        .expect_err("injected MLFMM warm launch failure must be typed");
    if !matches!(failure, BackendError::Launch { .. }) {
        return Err(format!(
            "MLFMM warm failure was not a typed launch error: {failure}"
        ));
    }
    // The operator rejects during launch preflight, before the backend's
    // injected fault is consumed by a CUDA call. Clear that one-shot fault so
    // the preservation readback observes the caller buffer rather than
    // failing at the diagnostic boundary itself.
    backend.clear_failure();
    backend
        .recover_after_error()
        .map_err(|error| format!("recover MLFMM warm launch failure: {error}"))?;
    let mut failed_output = vec![Complex64::new(0.0, 0.0); probe.len()];
    backend
        .try_copy_d2h(output.raw(), &mut failed_output)
        .map_err(|error| format!("read back MLFMM failure sentinel: {error}"))?;
    if failed_output != sentinel {
        return Err("MLFMM warm launch failure modified the caller output".into());
    }
    let mlfmm_dual_cpu = mlfmm
        .matvec_checked(&dual_probe)
        .map_err(|error| format!("CPU MLFMM warm RHS reference: {error}"))?;
    backend.reset_runtime_stats();
    let mlfmm_warm_start = backend.begin_runtime_stats();
    let mut mlfmm_warm_errors = Vec::with_capacity(2);
    for (rhs, expected) in [(&input, &mlfmm_cpu[0]), (&dual_input, &mlfmm_dual_cpu)] {
        backend
            .try_copy_d2d(rhs.raw(), mlfmm_warm_rhs.raw_mut())
            .map_err(|error| format!("update resident MLFMM RHS: {error}"))?;
        cuda_mlfmm
            .try_apply_device(&mlfmm_warm_rhs, &mut output, Transpose::NoTrans)
            .map_err(|error| format!("CUDA MLFMM warm replay: {error}"))?;
        backend
            .try_synchronize()
            .map_err(|error| format!("synchronize MLFMM warm replay: {error}"))?;
        let mut actual = vec![Complex64::new(0.0, 0.0); probe.len()];
        backend
            .try_copy_d2h(output.raw(), &mut actual)
            .map_err(|error| format!("read back MLFMM warm replay: {error}"))?;
        mlfmm_warm_errors.push(relative_error(&actual, expected));
    }
    let mlfmm_warm_stats = backend.runtime_stats_since(mlfmm_warm_start);
    let mlfmm_warm_error = mlfmm_warm_errors.iter().copied().fold(0.0, f64::max);
    if !mlfmm_warm_error.is_finite()
        || mlfmm_warm_error > TOLERANCE
        || mlfmm_warm_stats.allocation_operations != 0
        || mlfmm_warm_stats.transfer.h2d_operations != 0
        || mlfmm_warm_stats.transfer.d2d_operations < 2
        || mlfmm_warm_stats.transfer.d2h_operations != 2
    {
        return Err(format!(
            "MLFMM warm residency contract failed: error={mlfmm_warm_error:.6e}, stats={mlfmm_warm_stats:?}"
        ));
    }

    let field_points = [[0.25_f64, 0.33, 1.0], [2.25, 0.4, 1.1]];
    let field_velocity = vec![Complex64::new(0.0, 0.0); elements.len()];
    let cpu_field = compute_scattered_field_checked(
        &field_points,
        &elements,
        &nodes,
        &probe,
        Some(&field_velocity),
        &physics,
        sonium_bem::BasisOrder::Constant,
    )
    .map_err(|error| format!("CPU H-matrix field reference failed: {error}"))?;
    backend.reset_runtime_stats();
    let cuda_field = compute_scattered_field_cuda_checked(
        &pool,
        &field_points,
        &elements,
        &nodes,
        &probe,
        Some(&field_velocity),
        &physics,
        sonium_bem::BasisOrder::Constant,
    )
    .map_err(|error| format!("CUDA H-matrix field evaluation failed: {error}"))?;
    let field_error = relative_error(&cuda_field, &cpu_field);
    let field_stats = backend.runtime_stats();
    if field_error > TOLERANCE {
        return Err(format!(
            "CUDA H-matrix field error {field_error:.6e} exceeds {TOLERANCE:.6e}"
        ));
    }
    if field_stats.kernel_launches == 0
        || field_stats.transfer.h2d_operations == 0
        || field_stats.transfer.d2h_operations == 0
    {
        return Err(format!(
            "CUDA H-matrix field did not prove device work: kernels={}, h2d={}, d2h={}",
            field_stats.kernel_launches,
            field_stats.transfer.h2d_operations,
            field_stats.transfer.d2h_operations
        ));
    }

    backend.reset_runtime_stats();
    let action_high_water_before = backend.runtime_stats().allocation_high_water_bytes;
    let product_started = Instant::now();
    let mut actual = Vec::new();
    let mut errors = Vec::new();
    let mut dense_errors = Vec::new();
    let mut forward_result = None;
    let mut transpose_result = None;
    let mut hermitian_result = None;
    for action in [
        HMatrixAction::Forward,
        HMatrixAction::Transpose,
        HMatrixAction::Hermitian,
    ] {
        match action {
            HMatrixAction::Forward => cuda_hmatrix
                .try_apply(&input, &mut output)
                .map_err(|error| error.to_string())?,
            HMatrixAction::Transpose => cuda_hmatrix
                .try_apply_transpose(&input, &mut output)
                .map_err(|error| error.to_string())?,
            HMatrixAction::Hermitian => cuda_hmatrix
                .try_apply_hermitian(&input, &mut output)
                .map_err(|error| error.to_string())?,
        }
        backend
            .try_synchronize()
            .map_err(|error| format!("synchronize {action:?}: {error}"))?;
        let mut result = vec![Complex64::new(0.0, 0.0); probe.len()];
        backend
            .try_copy_d2h(output.raw(), &mut result)
            .map_err(|error| format!("read back {action:?}: {error}"))?;
        let expected = cpu_action(&hmatrix, &probe, action);
        errors.push(relative_error(&result, &expected));
        let dense_expected = dense_action(&dense.matrix, &probe, action);
        dense_errors.push(relative_error(&result, &dense_expected));
        match action {
            HMatrixAction::Forward => forward_result = Some(result.clone()),
            HMatrixAction::Hermitian => hermitian_result = Some(result.clone()),
            HMatrixAction::Transpose => transpose_result = Some(result.clone()),
        }
        actual.extend(result);
    }
    cuda_hmatrix
        .try_apply(&dual_input, &mut output)
        .map_err(|error| format!("CUDA reciprocity forward action failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("synchronize reciprocity action: {error}"))?;
    let mut dual_forward = vec![Complex64::new(0.0, 0.0); dual_probe.len()];
    backend
        .try_copy_d2h(output.raw(), &mut dual_forward)
        .map_err(|error| format!("read back reciprocity action: {error}"))?;
    let elapsed_ms = product_started.elapsed().as_millis() as u64;
    let stats = backend.runtime_stats();
    let max_error = errors.iter().copied().fold(0.0, f64::max);
    let dense_max_error = dense_errors.iter().copied().fold(0.0, f64::max);
    let dense_forward = dense_action(&dense.matrix, &probe, HMatrixAction::Forward);
    let cpu_forward = cpu_action(&hmatrix, &probe, HMatrixAction::Forward);
    let compressed_residual = relative_error(&cpu_forward, &dense_forward);
    let forward_result = forward_result.expect("forward H-matrix result");
    let transpose_result = transpose_result.expect("transpose H-matrix result");
    let hermitian_result = hermitian_result.expect("Hermitian H-matrix result");
    let forward_inner_product: Complex64 = probe
        .iter()
        .zip(&forward_result)
        .map(|(x, value)| x.conj() * value)
        .sum();
    let adjoint_inner_product: Complex64 = hermitian_result
        .iter()
        .zip(&probe)
        .map(|(value, x)| value.conj() * x)
        .sum();
    let adjoint_error = complex_absolute_error(forward_inner_product, adjoint_inner_product);
    let reciprocity_left: Complex64 = probe
        .iter()
        .zip(&dual_forward)
        .map(|(x, value)| *x * *value)
        .sum();
    let reciprocity_right: Complex64 = dual_probe
        .iter()
        .zip(&transpose_result)
        .map(|(y, value)| *y * *value)
        .sum();
    let reciprocity_error = complex_absolute_error(reciprocity_left, reciprocity_right);
    if !max_error.is_finite() || max_error > TOLERANCE {
        return Err(format!(
            "CUDA H-matrix action error {max_error:.6e} exceeds {TOLERANCE:.6e}"
        ));
    }
    if !adjoint_error.is_finite() || adjoint_error > TOLERANCE {
        return Err(format!(
            "CUDA H-matrix adjoint identity error {adjoint_error:.6e} exceeds {TOLERANCE:.6e}"
        ));
    }
    if !dense_max_error.is_finite()
        || !compressed_residual.is_finite()
        || dense_max_error > DENSE_RESIDUAL_TOLERANCE
        || compressed_residual > DENSE_RESIDUAL_TOLERANCE
    {
        return Err(format!(
            "CUDA/CPU H-matrix dense residual {dense_max_error:.6e}/{compressed_residual:.6e} exceeds {DENSE_RESIDUAL_TOLERANCE:.6e}"
        ));
    }
    if !reciprocity_error.is_finite() || reciprocity_error > TOLERANCE {
        return Err(format!(
            "CUDA H-matrix reciprocity identity error {reciprocity_error:.6e} exceeds {TOLERANCE:.6e}"
        ));
    }
    // The readbacks above are publication boundaries.  No H-matrix action is
    // allowed to perform an intermediate host transfer or rebuild its blocks.
    if stats.transfer.h2d_operations != 0 || stats.transfer.d2h_operations != 4 {
        return Err(format!(
            "unexpected steady-state transfers: h2d={}, d2h={}",
            stats.transfer.h2d_operations, stats.transfer.d2h_operations
        ));
    }
    if stats.kernel_launches == 0 {
        return Err("CUDA H-matrix action launched no device kernels".into());
    }
    if stats.allocation_high_water_bytes != action_high_water_before {
        return Err(format!(
            "CUDA H-matrix actions allocated after setup: high-water {} -> {} bytes",
            action_high_water_before, stats.allocation_high_water_bytes
        ));
    }

    let mut hmatrix_warm_rhs = pool
        .try_alloc::<Complex64>(probe.len())
        .map_err(|error| error.to_string())?;
    backend
        .try_copy_d2d(input.raw(), hmatrix_warm_rhs.raw_mut())
        .map_err(|error| format!("seed resident H-matrix RHS slot: {error}"))?;
    backend
        .try_copy_h2d(&sentinel, output.raw_mut())
        .map_err(|error| format!("seed H-matrix failure sentinel: {error}"))?;
    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = cuda_hmatrix
        .try_apply(&hmatrix_warm_rhs, &mut output)
        .expect_err("injected H-matrix warm launch failure must be typed");
    if !matches!(failure, BackendError::Launch { .. }) {
        return Err(format!(
            "H-matrix warm failure was not a typed launch error: {failure}"
        ));
    }
    backend.clear_failure();
    backend
        .recover_after_error()
        .map_err(|error| format!("recover H-matrix warm launch failure: {error}"))?;
    let mut failed_output = vec![Complex64::new(0.0, 0.0); probe.len()];
    backend
        .try_copy_d2h(output.raw(), &mut failed_output)
        .map_err(|error| format!("read back H-matrix failure sentinel: {error}"))?;
    if failed_output != sentinel {
        return Err("H-matrix warm launch failure modified the caller output".into());
    }
    let hmatrix_dual_cpu = cpu_action(&hmatrix, &dual_probe, HMatrixAction::Forward);
    backend.reset_runtime_stats();
    let hmatrix_warm_start = backend.begin_runtime_stats();
    let mut hmatrix_warm_errors = Vec::with_capacity(2);
    for (rhs, expected) in [(&input, &cpu_forward), (&dual_input, &hmatrix_dual_cpu)] {
        backend
            .try_copy_d2d(rhs.raw(), hmatrix_warm_rhs.raw_mut())
            .map_err(|error| format!("update resident H-matrix RHS: {error}"))?;
        cuda_hmatrix
            .try_apply(&hmatrix_warm_rhs, &mut output)
            .map_err(|error| format!("CUDA H-matrix warm replay: {error}"))?;
        backend
            .try_synchronize()
            .map_err(|error| format!("synchronize H-matrix warm replay: {error}"))?;
        let mut actual = vec![Complex64::new(0.0, 0.0); probe.len()];
        backend
            .try_copy_d2h(output.raw(), &mut actual)
            .map_err(|error| format!("read back H-matrix warm replay: {error}"))?;
        hmatrix_warm_errors.push(relative_error(&actual, expected));
    }
    let hmatrix_warm_stats = backend.runtime_stats_since(hmatrix_warm_start);
    let hmatrix_warm_error = hmatrix_warm_errors.iter().copied().fold(0.0, f64::max);
    if !hmatrix_warm_error.is_finite()
        || hmatrix_warm_error > TOLERANCE
        || hmatrix_warm_stats.allocation_operations != 0
        || hmatrix_warm_stats.transfer.h2d_operations != 0
        || hmatrix_warm_stats.transfer.d2d_operations < 2
        || hmatrix_warm_stats.transfer.d2h_operations != 2
    {
        return Err(format!(
            "H-matrix warm residency contract failed: error={hmatrix_warm_error:.6e}, stats={hmatrix_warm_stats:?}"
        ));
    }

    // Drop the resident operators only after the replay and retry have
    // completed.  The pool remains alive so the checked live-byte counter can
    // prove that teardown released the hierarchy and workspace allocations.
    let live_bytes_before_teardown = backend.runtime_stats().live_allocation_bytes;
    drop(cuda_hmatrix);
    drop(cuda_mlfmm);
    let live_bytes_after_teardown = backend.runtime_stats().live_allocation_bytes;
    if live_bytes_after_teardown >= live_bytes_before_teardown {
        return Err(format!(
            "resident H-matrix/MLFMM teardown released no device bytes: before={live_bytes_before_teardown}, after={live_bytes_after_teardown}"
        ));
    }

    let mut result = QaResult::passed(CASE_ID)
        .requested_backend("cuda")
        .actual_backend("cuda")
        .backend_passed("cuda")
        .reference("checked CPU H-matrix actions plus exact dense TBEM residual and field reference")
        .reference_checksum("analytic/deterministic-fixture:bem-cuda-hmatrix-fmm-device-residency-v1")
        .input_checksum("analytic/deterministic-fixture:bem-cuda-hmatrix-fmm-v1")
        .dimension("3D Tri3 constant-panel H-matrix/FMM action")
        .formulation("host-built Complex64 dense near-field plus low-rank far-field blocks with device-resident actions")
        .tolerance("action/field/adjoint/reciprocity errors <= 1e-10; dense residual <= 1e-8")
        .metric("mlfmm_max_action_relative_error", mlfmm_error)
        .metric("max_action_relative_error", max_error)
        .metric("dense_residual_relative_error", dense_max_error)
        .metric("compressed_residual_relative_error", compressed_residual)
        .metric("adjoint_identity_error", adjoint_error)
        .metric("reciprocity_identity_error", reciprocity_error)
        .metric("field_relative_error", field_error)
        .metric(
            "mlfmm_cuda_kernel_launches",
            mlfmm_stats.kernel_launches as f64,
        )
        .metric(
            "mlfmm_cuda_h2d_operations_steady_state",
            mlfmm_stats.transfer.h2d_operations as f64,
        )
        .metric(
            "mlfmm_cuda_d2h_publications",
            mlfmm_stats.transfer.d2h_operations as f64,
        )
        .metric("cuda_kernel_launches", stats.kernel_launches as f64)
        .metric(
            "cuda_h2d_operations_steady_state",
            stats.transfer.h2d_operations as f64,
        )
        .metric(
            "cuda_d2h_publications",
            stats.transfer.d2h_operations as f64,
        )
        .metric(
            "field_cuda_kernel_launches",
            field_stats.kernel_launches as f64,
        )
        .metric(
            "field_cuda_h2d_operations",
            field_stats.transfer.h2d_operations as f64,
        )
        .metric(
            "field_cuda_d2h_operations",
            field_stats.transfer.d2h_operations as f64,
        )
        .metric(
            "cuda_allocation_high_water_bytes",
            stats.allocation_high_water_bytes as f64,
        )
        .metric(
            "cuda_action_allocation_high_water_bytes",
            stats.allocation_high_water_bytes as f64,
        )
        .metric("mlfmm_warm_replays", 2.0)
        .metric("mlfmm_warm_rhs_updates", 2.0)
        .metric(
            "mlfmm_warm_structural_allocations",
            mlfmm_warm_stats.allocation_operations as f64,
        )
        .metric(
            "mlfmm_warm_structural_h2d_operations",
            mlfmm_warm_stats.transfer.h2d_operations as f64,
        )
        .metric(
            "mlfmm_warm_rhs_d2d_operations",
            2.0,
        )
        .metric(
            "mlfmm_warm_d2h_publications",
            mlfmm_warm_stats.transfer.d2h_operations as f64,
        )
        .metric("mlfmm_warm_max_action_relative_error", mlfmm_warm_error)
        .metric("mlfmm_warm_failure_recovery", 1.0)
        .metric("mlfmm_warm_failure_output_preserved", 1.0)
        .metric("hmatrix_warm_replays", 2.0)
        .metric("hmatrix_warm_rhs_updates", 2.0)
        .metric(
            "hmatrix_warm_structural_allocations",
            hmatrix_warm_stats.allocation_operations as f64,
        )
        .metric(
            "hmatrix_warm_structural_h2d_operations",
            hmatrix_warm_stats.transfer.h2d_operations as f64,
        )
        .metric(
            "hmatrix_warm_rhs_d2d_operations",
            2.0,
        )
        .metric(
            "hmatrix_warm_d2h_publications",
            hmatrix_warm_stats.transfer.d2h_operations as f64,
        )
        .metric("hmatrix_warm_max_action_relative_error", hmatrix_warm_error)
        .metric("hmatrix_warm_failure_recovery", 1.0)
        .metric("hmatrix_warm_failure_output_preserved", 1.0)
        .metric(
            "cuda_teardown_live_bytes_before",
            live_bytes_before_teardown as f64,
        )
        .metric(
            "cuda_teardown_live_bytes_after",
            live_bytes_after_teardown as f64,
        )
        .metric(
            "cuda_teardown_released_bytes",
            live_bytes_before_teardown.saturating_sub(live_bytes_after_teardown) as f64,
        )
        .metric(
            "cuda_action_output_l2",
            actual.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt(),
        )
        .maximum("mlfmm_max_action_relative_error", TOLERANCE)
        .maximum("max_action_relative_error", TOLERANCE)
        .maximum("dense_residual_relative_error", DENSE_RESIDUAL_TOLERANCE)
        .maximum(
            "compressed_residual_relative_error",
            DENSE_RESIDUAL_TOLERANCE,
        )
        .maximum("adjoint_identity_error", TOLERANCE)
        .maximum("reciprocity_identity_error", TOLERANCE)
        .maximum("field_relative_error", TOLERANCE)
        .timing_ms("cuda_hmatrix_setup", setup_ms)
        .timing_ms("cuda_mlfmm_actions", mlfmm_elapsed_ms)
        .timing_ms("cuda_hmatrix_actions", elapsed_ms)
        .memory_bytes(stats.allocation_high_water_bytes as u64)
        .elapsed_ms(setup_ms.saturating_add(elapsed_ms));
    result
        .environment
        .insert("capability.bem.hmatrix.c64".into(), "native".into());
    result.environment.insert(
        "capability.bem.hmatrix_setup.c64".into(),
        "host_shared_fallback".into(),
    );
    result
        .environment
        .insert("capability.bem.fmm.c64".into(), "native".into());
    result.environment.insert(
        "capability.bem.fmm_setup.c64".into(),
        "host_shared_fallback".into(),
    );
    result
        .environment
        .insert("capability.bem.mlfmm.c64".into(), "native".into());
    result.environment.insert(
        "capability.bem.mlfmm_setup.c64".into(),
        "host_shared_fallback".into(),
    );
    result.environment.insert(
        "capability.bem.field_evaluation.c64".into(),
        "native".into(),
    );
    let report = serde_json::to_vec_pretty(&serde_json::json!({
        "case_id": CASE_ID,
        "metrics": result.metrics,
        "environment": result.environment,
    }))
    .map_err(|error| format!("serialize CUDA H-matrix QA artifact: {error}"))?;
    let (artifact, checksum) = write_qa_artifact("bem_cuda_hmatrix_fmm_report.json", &report)?;
    result = result
        .artifact_with_provenance(
            artifact,
            "crates/sonium-qa/bin/bem_cuda_hmatrix_fmm.rs",
            "runtime H-matrix/FMM residency report",
            "v1",
            checksum,
        )
        .runtime_evidence(cuda_runtime_evidence(
            stats,
            QaExecutionCounts {
                executed: 1,
                passed: 1,
                ..QaExecutionCounts::default()
            },
        ))
        .physical_residual(dense_max_error)
        .physical_reference_error(field_error);
    Ok(result)
}

fn main() {
    let started = Instant::now();
    let result = match required_cuda_backend() {
        Ok(backend) => match run(backend) {
            Ok(result) => result,
            Err(error) => failed_cuda(error, "cuda", QaFailureStage::Execution),
        },
        Err(error) => failed_cuda(
            format!("required NVIDIA CUDA backend unavailable: {error}"),
            "unavailable",
            QaFailureStage::Initialization,
        ),
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);
    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("CUDA H-matrix QA result must serialize")
    );
    if result.status != sonium_qa::QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
