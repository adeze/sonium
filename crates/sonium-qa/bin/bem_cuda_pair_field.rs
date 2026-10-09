#![recursion_limit = "256"]

//! Required native-CUDA B1 pair/field coverage and failure-contract matrix.
//!
//! The producer intentionally reports only capabilities implemented by the
//! checked CUDA seams: regular Tri3 pairs in Complex32/Complex64, a native
//! Complex64 source-collocation-x derivative,
//! chart-driven Complex64 quadrature (bulk/singular/edge), and constant Tri3
//! Complex64 exterior fields.  The production BEM adapter is additionally
//! exercised with resident far-regular, near-singular, and self-singular
//! batches across a warm frequency replay, plus a genuine scalar-f64 Laplace
//! regular-pair channel.  A bounded two-medium constant-panel Muller cell
//! runs the same resident pair and field adapters for distinct exterior and
//! interior media and forms the four production transmission blocks from
//! those native channels.  Unsupported Quad4, higher-order, and on-surface
//! field cells are rows in the artifact rather than silent CPU substitutions.

use num_complex::{Complex32, Complex64};
use serde_json::{Value, json};
use sonium_bem::core::assembly::{
    CudaTbemAssemblyError, CudaTbemAssemblyWorkspace, TbemSystem, build_tbem_system_parallel,
};
use sonium_bem::core::types::{
    BoundaryCondition, Element, ElementProperty, ElementType, PhysicsParams,
};
use sonium_compute::{
    Backend, BackendError, BemFieldElementC64, BemFieldPointC64, BemQuadraturePairC64,
    BemQuadratureSampleC64, BemRegularPairIn, BemRegularPairInF64, BemRegularPairOutRealF64,
    CpuBackend, CudaBackend, CudaBemFieldOperatorC64, CudaBemQuadratureOperatorC64,
    CudaBemRegularPairOperatorC64, CudaFailureInjectionPoint, MemoryPool, MemoryPoolConfig,
};
use sonium_qa::{
    QaExecutionCounts, QaFailureStage, QaResult, cuda_hardware_provenance, cuda_runtime_evidence,
    required_cuda_backend, write_qa_artifact,
};
use std::time::Instant;

const CASE_ID: &str = "bem.cuda-pair-field-matrix.v1";
const C32_TOLERANCE: f64 = 1.0e-4;
const C64_TOLERANCE: f64 = 3.0e-12;
const SOURCE_X_DERIVATIVE_TOLERANCE: f64 = 5.0e-6;
const SOURCE_X_DERIVATIVE_STEP: f64 = 1.0e-6;
const TBEM_CUDA_TOLERANCE: f64 = 1.0e-3;
const TRANSMISSION_CUDA_TOLERANCE: f64 = 3.0e-12;

#[derive(Clone, Copy)]
struct PairGeometry {
    source_point: [f64; 3],
    source_normal: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
}

type ProductionTbemFixture = (Vec<Element>, Vec<[f64; 3]>, PhysicsParams);

fn pair_geometry() -> [PairGeometry; 3] {
    [
        PairGeometry {
            source_point: [0.0, 0.0, 0.0],
            source_normal: [0.0, 0.0, 1.0],
            v0: [-0.7, -0.3, 1.0],
            v1: [0.8, -0.2, 1.1],
            v2: [-0.1, 0.9, 1.2],
        },
        PairGeometry {
            source_point: [0.2, -0.1, 0.3],
            source_normal: [0.0, 1.0, 0.0],
            v0: [1.2, -0.4, 1.4],
            v1: [1.8, 0.1, 1.6],
            v2: [1.1, 0.8, 1.5],
        },
        PairGeometry {
            source_point: [-0.3, 0.4, 0.2],
            source_normal: [1.0, 0.0, 0.0],
            v0: [2.1, -0.5, 0.2],
            v1: [2.7, -0.4, 0.3],
            v2: [2.1, 0.4, 0.5],
        },
    ]
}

fn pairs_c32() -> Vec<BemRegularPairIn> {
    pair_geometry()
        .into_iter()
        .map(|pair| {
            BemRegularPairIn::from_triangle(
                pair.source_point.map(|value| value as f32),
                pair.source_normal.map(|value| value as f32),
                pair.v0.map(|value| value as f32),
                pair.v1.map(|value| value as f32),
                pair.v2.map(|value| value as f32),
            )
        })
        .collect()
}

fn pairs_c64() -> Vec<BemRegularPairInF64> {
    pair_geometry()
        .into_iter()
        .map(|pair| {
            BemRegularPairInF64::from_triangle(
                pair.source_point,
                pair.source_normal,
                pair.v0,
                pair.v1,
                pair.v2,
            )
        })
        .collect()
}

fn component_error(actual: Complex64, expected: Complex64) -> f64 {
    if [actual.re, actual.im, expected.re, expected.im]
        .into_iter()
        .any(|value| !value.is_finite())
    {
        return f64::INFINITY;
    }
    (actual - expected).norm() / expected.norm().max(1.0)
}

fn c32_as_c64(value: Complex32) -> Complex64 {
    Complex64::new(value.re as f64, value.im as f64)
}

fn pair32_error(
    actual: &sonium_compute::BemRegularPairOut,
    expected: &sonium_compute::BemRegularPairOut,
) -> f64 {
    [
        (actual.g, expected.g),
        (actual.h, expected.h),
        (actual.ht, expected.ht),
        (actual.e, expected.e),
    ]
    .into_iter()
    .map(|(got, want)| component_error(c32_as_c64(got), c32_as_c64(want)))
    .fold(0.0, f64::max)
}

fn pair64_error(
    actual: &sonium_compute::BemRegularPairOutF64,
    expected: &sonium_compute::BemRegularPairOutF64,
) -> f64 {
    [
        (actual.g, expected.g),
        (actual.h, expected.h),
        (actual.ht, expected.ht),
        (actual.e, expected.e),
    ]
    .into_iter()
    .map(|(got, want)| component_error(got, want))
    .fold(0.0, f64::max)
}

fn relative_vector_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    if actual.len() != expected.len() || actual.is_empty() {
        return f64::INFINITY;
    }
    let numerator = actual
        .iter()
        .zip(expected)
        .map(|(got, want)| (*got - *want).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator = expected.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
    if numerator.is_finite() && denominator.is_finite() {
        numerator / denominator.max(f64::MIN_POSITIVE)
    } else {
        f64::INFINITY
    }
}

fn relative_tbem_error(actual: &TbemSystem, expected: &TbemSystem) -> f64 {
    if actual.matrix.shape() != expected.matrix.shape() || actual.rhs.len() != expected.rhs.len() {
        return f64::INFINITY;
    }
    relative_vector_error(actual.matrix.as_slice(), expected.matrix.as_slice())
        .max(relative_vector_error(&actual.rhs, &expected.rhs))
}

/// Three independent constant Tri3 panels deliberately exercise every route
/// owned by the production CUDA TBEM adapter: self-singular, near-singular,
/// and far regular quadrature charts.
fn production_tbem_fixture() -> Result<ProductionTbemFixture, String> {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 10.0],
        [1.0, 0.0, 10.0],
        [0.0, 1.0, 10.0],
        [1.5, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [1.5, 1.0, 0.0],
    ];
    let element = |connectivity: Vec<usize>, center_z: f64, dof: usize| Element {
        connectivity,
        element_type: ElementType::Tri3,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 3],
        center: [1.0 / 3.0, 1.0 / 3.0, center_z],
        area: 0.5,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::default()]),
        group: 0,
        dof_addresses: vec![dof],
    };
    let elements = vec![
        element(vec![0, 1, 2], 0.0, 0),
        // Identical collocation height makes this pair intentionally enter
        // the production near-singular quadrature route (< 3.0).
        element(vec![6, 7, 8], 0.0, 1),
        element(vec![3, 4, 5], 10.0, 2),
    ];
    let physics = PhysicsParams::try_new(600.0, 343.0, 1.21, false)
        .map_err(|error| format!("production TBEM fixture physics: {error:?}"))?;
    Ok((elements, nodes, physics))
}

fn production_far_pairs(elements: &[Element], nodes: &[[f64; 3]]) -> Vec<BemRegularPairInF64> {
    let mut pairs = Vec::new();
    for (source_index, source) in elements.iter().enumerate() {
        for (field_index, field) in elements.iter().enumerate() {
            if source_index == field_index {
                continue;
            }
            let distance = [
                field.center[0] - source.center[0],
                field.center[1] - source.center[1],
                field.center[2] - source.center[2],
            ]
            .into_iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
            if distance / field.area.sqrt().max(1.0e-30) < 3.0 {
                continue;
            }
            let [i0, i1, i2] = field.connectivity.as_slice() else {
                unreachable!("production fixture uses Tri3 panels")
            };
            pairs.push(BemRegularPairInF64::from_triangle(
                source.center,
                source.normal,
                nodes[*i0],
                nodes[*i1],
                nodes[*i2],
            ));
        }
    }
    pairs
}

fn laplace_pair_reference(pair: &BemRegularPairInF64) -> BemRegularPairOutRealF64 {
    const XI: [f64; 4] = [0.333333333333333, 0.6, 0.2, 0.2];
    const ETA: [f64; 4] = [0.333333333333333, 0.2, 0.6, 0.2];
    const WEIGHTS: [f64; 4] = [
        -0.28125,
        0.260416666666667,
        0.260416666666667,
        0.260416666666667,
    ];
    let edge1 = [
        pair.v1[0] - pair.v0[0],
        pair.v1[1] - pair.v0[1],
        pair.v1[2] - pair.v0[2],
    ];
    let edge2 = [
        pair.v2[0] - pair.v0[0],
        pair.v2[1] - pair.v0[1],
        pair.v2[2] - pair.v0[2],
    ];
    let raw_normal = [
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
        edge1[2] * edge2[0] - edge1[0] * edge2[2],
        edge1[0] * edge2[1] - edge1[1] * edge2[0],
    ];
    let jacobian = raw_normal
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let field_normal = raw_normal.map(|value| value / jacobian);
    let mut output = BemRegularPairOutRealF64::default();
    for q in 0..4 {
        let n0 = 1.0 - XI[q] - ETA[q];
        let position = [
            n0 * pair.v0[0] + XI[q] * pair.v1[0] + ETA[q] * pair.v2[0],
            n0 * pair.v0[1] + XI[q] * pair.v1[1] + ETA[q] * pair.v2[1],
            n0 * pair.v0[2] + XI[q] * pair.v1[2] + ETA[q] * pair.v2[2],
        ];
        let difference = [
            position[0] - pair.source_point[0],
            position[1] - pair.source_point[1],
            position[2] - pair.source_point[2],
        ];
        let radius = difference
            .into_iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if radius < 1.0e-15 {
            continue;
        }
        let unit = difference.map(|value| value / radius);
        let r_dot_ny = unit
            .into_iter()
            .zip(field_normal)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let r_dot_nx = -unit
            .into_iter()
            .zip(pair.source_normal)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let normal_dot = pair
            .source_normal
            .into_iter()
            .zip(field_normal)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let green = WEIGHTS[q] * jacobian / (4.0 * std::f64::consts::PI * radius);
        let derivative = -1.0 / radius;
        let inverse_radius_squared = 1.0 / (radius * radius);
        output.g += green;
        output.h += green * derivative * r_dot_ny;
        output.ht += green * derivative * r_dot_nx;
        output.e += green * inverse_radius_squared * (3.0 * r_dot_ny * r_dot_nx + normal_dot);
    }
    output
}

fn real_f64_relative_error(
    actual: &[BemRegularPairOutRealF64],
    expected: &[BemRegularPairOutRealF64],
) -> f64 {
    if actual.len() != expected.len() {
        return f64::INFINITY;
    }
    actual
        .iter()
        .zip(expected)
        .flat_map(|(actual, expected)| {
            [
                (actual.g, expected.g),
                (actual.h, expected.h),
                (actual.ht, expected.ht),
                (actual.e, expected.e),
            ]
        })
        .map(|(actual, expected)| {
            if !actual.is_finite() || !expected.is_finite() {
                f64::INFINITY
            } else {
                (actual - expected).abs() / expected.abs().max(1.0)
            }
        })
        .fold(0.0, f64::max)
}

fn production_real_f64_row(backend: &CudaBackend) -> Result<Value, String> {
    let (elements, nodes, physics) = production_tbem_fixture()?;
    let pairs = production_far_pairs(&elements, &nodes);
    let expected: Vec<_> = pairs.iter().map(laplace_pair_reference).collect();
    let baseline_live_bytes = backend.live_allocation_bytes();
    let mut workspace = CudaTbemAssemblyWorkspace::new_with_regular_pair_channels(
        &elements, &nodes, &physics, backend,
    )
    .map_err(|error| format!("construct real-f64 production workspace: {error}"))?;
    let construction_live_bytes = backend.live_allocation_bytes();

    let first_baseline = backend.runtime_stats();
    let first = workspace
        .assemble_real_f64(&elements, &nodes, backend)
        .map_err(|error| format!("real-f64 first assembly: {error}"))?;
    let first_error = real_f64_relative_error(&first, &expected);
    let first_stats = backend.runtime_stats_since(first_baseline);

    let warm_baseline = backend.runtime_stats();
    let warm = workspace
        .assemble_real_f64(&elements, &nodes, backend)
        .map_err(|error| format!("real-f64 warm assembly: {error}"))?;
    let warm_error = real_f64_relative_error(&warm, &expected);
    let warm_stats = backend.runtime_stats_since(warm_baseline);

    let failure_baseline = backend.runtime_stats();
    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = workspace.assemble_real_f64(&elements, &nodes, backend);
    let failure_typed = matches!(
        failure,
        Err(CudaTbemAssemblyError::Backend(BackendError::Launch {
            operation: "cuda_launch",
            ..
        }))
    );
    backend
        .recover_after_error()
        .map_err(|error| format!("recover real-f64 launch failure: {error}"))?;
    let retry = workspace
        .assemble_real_f64(&elements, &nodes, backend)
        .map_err(|error| format!("real-f64 retry after launch failure: {error}"))?;
    let retry_error = real_f64_relative_error(&retry, &expected);
    let failure_stats = backend.runtime_stats_since(failure_baseline);

    let live_before_drop = backend.live_allocation_bytes();
    drop(workspace);
    let live_after_drop = backend.live_allocation_bytes();
    let released = live_before_drop.saturating_sub(live_after_drop);
    let resident = first_stats.allocation_operations == 0
        && first_stats.transfer.h2d_operations == 0
        && first_stats.kernel_launches >= 1
        && first_stats.transfer.d2h_operations >= 1
        && warm_stats.allocation_operations == 0
        && warm_stats.transfer.h2d_operations == 0
        && warm_stats.kernel_launches >= 1
        && warm_stats.transfer.d2h_operations >= 1;
    if first_error > TBEM_CUDA_TOLERANCE
        || warm_error > TBEM_CUDA_TOLERANCE
        || retry_error > TBEM_CUDA_TOLERANCE
        || !resident
        || !failure_typed
        || released == 0
        || construction_live_bytes <= baseline_live_bytes
        || live_after_drop > baseline_live_bytes
    {
        return Err(format!(
            "production real-f64 contract failed: first={first_error:.6e}, warm={warm_error:.6e}, retry={retry_error:.6e}, first_allocs={}, warm_allocs={}, first_h2d={}, warm_h2d={}, warm_d2h={}, warm_launches={}, failure_typed={failure_typed}, released={released}, baseline={}, after_drop={}",
            first_stats.allocation_operations,
            warm_stats.allocation_operations,
            first_stats.transfer.h2d_operations,
            warm_stats.transfer.h2d_operations,
            warm_stats.transfer.d2h_operations,
            warm_stats.kernel_launches,
            baseline_live_bytes,
            live_after_drop,
        ));
    }
    Ok(json!({
        "id": "production_tbem.real-f64-laplace-regular",
        "status": "passed",
        "scalar": "real f64 Laplace channels",
        "reference": "independent CPU f64 four-point Laplace pair integration",
        "active_pairs": pairs.len(),
        "initial_relative_error": first_error,
        "warm_relative_error": warm_error,
        "retry_relative_error": retry_error,
        "tolerance": TBEM_CUDA_TOLERANCE,
        "initial_kernel_launches": first_stats.kernel_launches,
        "initial_allocations": first_stats.allocation_operations,
        "initial_h2d_operations": first_stats.transfer.h2d_operations,
        "initial_d2h_operations": first_stats.transfer.d2h_operations,
        "warm_kernel_launches": warm_stats.kernel_launches,
        "warm_structural_allocations": warm_stats.allocation_operations,
        "warm_steady_state_h2d_operations": warm_stats.transfer.h2d_operations,
        "warm_publication_d2h_operations": warm_stats.transfer.d2h_operations,
        "failure_typed": failure_typed,
        "failure_retry_kernel_launches": failure_stats.kernel_launches,
        "failure_retry_h2d_operations": failure_stats.transfer.h2d_operations,
        "teardown_live_bytes_before": live_before_drop,
        "teardown_live_bytes_after": live_after_drop,
        "teardown_released_bytes": released,
    }))
}

fn production_tbem_row(backend: &CudaBackend) -> Result<Value, String> {
    let (elements, nodes, physics) = production_tbem_fixture()?;
    let beta = physics.burton_miller_beta();
    let warm_physics = PhysicsParams::try_new(720.0, 343.0, 1.21, false)
        .map_err(|error| format!("production TBEM warm physics: {error:?}"))?;
    let warm_beta = warm_physics.burton_miller_beta();
    let cpu = build_tbem_system_parallel(&elements, &nodes, &physics);
    let warm_cpu = build_tbem_system_parallel(&elements, &nodes, &warm_physics);

    let baseline_live_bytes = backend.live_allocation_bytes();
    let mut workspace = CudaTbemAssemblyWorkspace::new(&elements, &nodes, &physics, backend)
        .map_err(|error| format!("construct production TBEM workspace: {error}"))?;
    let construction_live_bytes = backend.live_allocation_bytes();

    // A normal Helmholtz workspace must reject optional primitive channels,
    // rather than silently publishing an empty derivative for nonempty geometry.
    let admission_baseline = backend.runtime_stats();
    let unavailable = workspace.assemble_source_x_derivative(&elements, &nodes, &physics, backend);
    let admission_stats = backend.runtime_stats_since(admission_baseline);
    if !matches!(unavailable, Err(CudaTbemAssemblyError::InvalidInput { .. }))
        || admission_stats.kernel_launches != 0
        || admission_stats.transfer.h2d_operations != 0
        || admission_stats.transfer.d2h_operations != 0
        || admission_stats.allocation_operations != 0
    {
        return Err("optional far-pair channel admission did not fail before device work".into());
    }

    // The workspace constructor owns the only geometry uploads.  Snapshot
    // rather than resetting here: allocation operations are a lifetime
    // ledger (unlike launch/transfer counters), so `runtime_stats_since`
    // is the only way to isolate action-time residency from setup and prior
    // QA rows.
    let first_baseline = backend.runtime_stats();
    let first = workspace
        .assemble(&elements, &nodes, &physics, beta, backend)
        .map_err(|error| format!("production TBEM first assembly: {error}"))?;
    let first_error = relative_tbem_error(&first, &cpu);
    let first_stats = backend.runtime_stats_since(first_baseline);

    let warm_baseline = backend.runtime_stats();
    let warm = workspace
        .assemble(&elements, &nodes, &warm_physics, warm_beta, backend)
        .map_err(|error| format!("production TBEM warm assembly: {error}"))?;
    let warm_error = relative_tbem_error(&warm, &warm_cpu);
    let warm_stats = backend.runtime_stats_since(warm_baseline);

    // Launch injection is consumed before either resident output buffer is
    // touched.  A retry on the same workspace proves typed recovery without
    // rebuilding or re-uploading any pair geometry.
    let failure_baseline = backend.runtime_stats();
    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = workspace.assemble(&elements, &nodes, &warm_physics, warm_beta, backend);
    let failure_typed = matches!(
        failure,
        Err(CudaTbemAssemblyError::Backend(BackendError::Launch {
            operation: "cuda_launch",
            ..
        }))
    );
    backend
        .recover_after_error()
        .map_err(|error| format!("recover production TBEM launch failure: {error}"))?;
    let retry = workspace
        .assemble(&elements, &nodes, &warm_physics, warm_beta, backend)
        .map_err(|error| format!("production TBEM retry after launch failure: {error}"))?;
    let retry_error = relative_tbem_error(&retry, &warm_cpu);
    let failure_stats = backend.runtime_stats_since(failure_baseline);

    let live_before_drop = backend.live_allocation_bytes();
    drop(workspace);
    let live_after_drop = backend.live_allocation_bytes();
    let released = live_before_drop.saturating_sub(live_after_drop);
    // Far, near, and self pairs now share the higher-order quadrature kernel.
    // Separate regular-pair rows still validate the four-point primitive.
    let first_routes_native =
        first_stats.kernel_launches == 1 && first_stats.transfer.d2h_operations == 1;
    let warm_resident = warm_stats.allocation_operations == 0
        && warm_stats.transfer.h2d_operations == 0
        && warm_stats.transfer.d2h_operations == 1
        && warm_stats.kernel_launches == 1;
    if first_error > TBEM_CUDA_TOLERANCE
        || warm_error > TBEM_CUDA_TOLERANCE
        || retry_error > TBEM_CUDA_TOLERANCE
        || first_stats.transfer.h2d_operations != 0
        || !first_routes_native
        || !warm_resident
        || !failure_typed
        || released == 0
        || construction_live_bytes <= baseline_live_bytes
        || live_after_drop > baseline_live_bytes
    {
        return Err(format!(
            "production TBEM contract failed: first={first_error:.6e}, warm={warm_error:.6e}, retry={retry_error:.6e}, first_h2d={}, warm_allocs={}, warm_h2d={}, warm_d2h={}, warm_launches={}, failure_typed={failure_typed}, released={released}, baseline={}, after_drop={}",
            first_stats.transfer.h2d_operations,
            warm_stats.allocation_operations,
            warm_stats.transfer.h2d_operations,
            warm_stats.transfer.d2h_operations,
            warm_stats.kernel_launches,
            baseline_live_bytes,
            live_after_drop,
        ));
    }
    Ok(json!({
        "id": "production_tbem.resident-tri3-regular-near-singular",
        "status": "passed",
        "scalar": "f64 geometry / Complex64 channels",
        "reference": "independent CPU canonical TBEM assembly",
        "routes": ["far-regular", "near-singular-quadrature", "self-singular-quadrature"],
        "active_elements": elements.len(),
        "initial_relative_error": first_error,
        "warm_relative_error": warm_error,
        "retry_relative_error": retry_error,
        "tolerance": TBEM_CUDA_TOLERANCE,
        "initial_kernel_launches": first_stats.kernel_launches,
        "initial_h2d_operations": first_stats.transfer.h2d_operations,
        "initial_d2h_operations": first_stats.transfer.d2h_operations,
        "warm_kernel_launches": warm_stats.kernel_launches,
        "warm_structural_allocations": warm_stats.allocation_operations,
        "warm_steady_state_h2d_operations": warm_stats.transfer.h2d_operations,
        "warm_publication_d2h_operations": warm_stats.transfer.d2h_operations,
        "failure_typed": failure_typed,
        "failure_retry_kernel_launches": failure_stats.kernel_launches,
        "failure_retry_h2d_operations": failure_stats.transfer.h2d_operations,
        "teardown_live_bytes_before": live_before_drop,
        "teardown_live_bytes_after": live_after_drop,
        "teardown_released_bytes": released,
    }))
}

fn reference_quadrature(
    pair: BemQuadraturePairC64,
    samples: &[BemQuadratureSampleC64],
    wave_number: f64,
    harmonic_factor: f64,
) -> sonium_compute::BemRegularPairOutF64 {
    let mut result = sonium_compute::BemRegularPairOutF64::default();
    let wavruim = wave_number * harmonic_factor;
    let start = pair.sample_offset as usize;
    let end = (pair.sample_offset + pair.sample_count) as usize;
    for sample in &samples[start..end] {
        let diff = [
            sample.position[0] - pair.source_point[0],
            sample.position[1] - pair.source_point[1],
            sample.position[2] - pair.source_point[2],
        ];
        let radius = diff.iter().map(|value| value * value).sum::<f64>().sqrt();
        let unit = diff.map(|value| value / radius);
        let scaled_weight = sample.weight * sample.basis_weight;
        let green = Complex64::from_polar(
            scaled_weight / (4.0 * std::f64::consts::PI * radius),
            wavruim * radius,
        );
        if sample.kind == BemQuadratureSampleC64::KIND_HYPERSINGULAR_EDGE {
            let factor = Complex64::from_polar(
                1.0 / (4.0 * std::f64::consts::PI * radius),
                wavruim * radius,
            ) * Complex64::new(-1.0 / radius, wavruim);
            let cross = [
                factor * (unit[1] * sample.edge_direction[2] - unit[2] * sample.edge_direction[1]),
                factor * (unit[2] * sample.edge_direction[0] - unit[0] * sample.edge_direction[2]),
                factor * (unit[0] * sample.edge_direction[1] - unit[1] * sample.edge_direction[0]),
            ];
            result.e += cross
                .into_iter()
                .zip(pair.source_normal)
                .map(|(value, normal)| value * normal)
                .sum::<Complex64>()
                * scaled_weight;
            continue;
        }
        let r_dot_ny = unit
            .iter()
            .zip(sample.field_normal)
            .map(|(value, normal)| value * normal)
            .sum::<f64>();
        let r_dot_nx = -unit
            .iter()
            .zip(pair.source_normal)
            .map(|(value, normal)| value * normal)
            .sum::<f64>();
        let normal_dot = pair
            .source_normal
            .iter()
            .zip(sample.field_normal)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let factor = Complex64::new(-1.0 / radius, wavruim);
        result.g += green;
        result.h += green * factor * r_dot_ny;
        result.ht += green * factor * r_dot_nx;
        match sample.kind {
            BemQuadratureSampleC64::KIND_FULL => {
                let rq = r_dot_ny * r_dot_nx;
                let radius_squared = radius * radius;
                result.e += green
                    * Complex64::new(
                        (3.0 / radius_squared - wave_number * wave_number) * rq
                            + normal_dot / radius_squared,
                        -wavruim / radius * (3.0 * rq + normal_dot),
                    );
            }
            BemQuadratureSampleC64::KIND_SINGULAR_BULK => {
                result.e += green * (wave_number * wave_number * normal_dot);
            }
            BemQuadratureSampleC64::KIND_GH_ONLY => {}
            _ => unreachable!(),
        }
    }
    result
}

fn field_reference(
    point: [f64; 3],
    element: BemFieldElementC64,
    wave_number: f64,
    harmonic_factor: f64,
    pressure_factor: f64,
) -> Complex64 {
    let edge1 = [
        element.v1[0] - element.v0[0],
        element.v1[1] - element.v0[1],
        element.v1[2] - element.v0[2],
    ];
    let edge2 = [
        element.v2[0] - element.v0[0],
        element.v2[1] - element.v0[1],
        element.v2[2] - element.v0[2],
    ];
    let normal_raw = [
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
        edge1[2] * edge2[0] - edge1[0] * edge2[2],
        edge1[0] * edge2[1] - edge1[1] * edge2[0],
    ];
    let jacobian = normal_raw
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let normal = normal_raw.map(|value| value / jacobian);
    let xis = [
        0.333333333333333,
        0.797426985353087,
        0.101286507323456,
        0.101286507323456,
        0.470142064105115,
        0.059715871789770,
        0.470142064105115,
    ];
    let etas = [
        0.333333333333333,
        0.101286507323456,
        0.797426985353087,
        0.101286507323456,
        0.059715871789770,
        0.470142064105115,
        0.470142064105115,
    ];
    let weights = [
        0.1125,
        0.0629695902724135,
        0.0629695902724135,
        0.0629695902724135,
        0.066197076394253,
        0.066197076394253,
        0.066197076394253,
    ];
    let wavruim = wave_number * harmonic_factor;
    let mut result = Complex64::default();
    for ((&xi, &eta), &weight) in xis.iter().zip(&etas).zip(&weights) {
        let n0 = 1.0 - xi - eta;
        let position = [
            n0 * element.v0[0] + xi * element.v1[0] + eta * element.v2[0],
            n0 * element.v0[1] + xi * element.v1[1] + eta * element.v2[1],
            n0 * element.v0[2] + xi * element.v1[2] + eta * element.v2[2],
        ];
        let diff = [
            position[0] - point[0],
            position[1] - point[1],
            position[2] - point[2],
        ];
        let radius = diff.iter().map(|value| value * value).sum::<f64>().sqrt();
        let inv_radius = 1.0 / radius;
        let drdn_y = diff
            .iter()
            .zip(normal)
            .map(|(distance, normal)| distance * normal)
            .sum::<f64>()
            * inv_radius;
        let green = Complex64::from_polar(
            jacobian * weight / (4.0 * std::f64::consts::PI * radius),
            wavruim * radius,
        );
        let derivative = green * Complex64::new(-inv_radius, wavruim);
        result += element.pressure * derivative * drdn_y;
        if element.velocity.norm_sqr() > 1.0e-30 {
            result -= element.velocity * Complex64::new(0.0, pressure_factor) * green;
        }
    }
    result
}

fn regular_pair_rows(backend: &CudaBackend) -> Result<Vec<Value>, String> {
    let c32 = pairs_c32();
    let c64 = pairs_c64();
    let cpu = CpuBackend::new();
    let cpu32 = cpu
        .try_bem_regular_pair_c32_batch(&c32, 1.0, -1.0)
        .map_err(|error| format!("CPU Complex32 BEM reference: {error}"))?;
    let cpu64 = cpu
        .try_bem_regular_pair_c64_batch(&c64, 1.0, -1.0)
        .map_err(|error| format!("CPU Complex64 BEM reference: {error}"))?;

    backend.reset_runtime_stats();
    let gpu32 = backend
        .try_bem_regular_pair_c32_batch(&c32, 1.0, -1.0)
        .map_err(|error| format!("CUDA Complex32 regular pair: {error}"))?;
    let stats32 = backend.runtime_stats();
    let c32_error = gpu32
        .iter()
        .zip(&cpu32)
        .map(|(got, want)| pair32_error(got, want))
        .fold(0.0, f64::max);
    if c32_error > C32_TOLERANCE {
        return Err(format!(
            "Complex32 pair parity {c32_error:.6e} exceeds {C32_TOLERANCE:.6e}"
        ));
    }

    backend.reset_runtime_stats();
    let gpu64 = backend
        .try_bem_regular_pair_c64_batch(&c64, 1.0, -1.0)
        .map_err(|error| format!("CUDA Complex64 regular pair: {error}"))?;
    let stats64 = backend.runtime_stats();
    let c64_error = gpu64
        .iter()
        .zip(&cpu64)
        .map(|(got, want)| pair64_error(got, want))
        .fold(0.0, f64::max);
    if c64_error > C64_TOLERANCE {
        return Err(format!(
            "Complex64 pair parity {c64_error:.6e} exceeds {C64_TOLERANCE:.6e}"
        ));
    }
    Ok(vec![
        json!({
            "id": "regular_pair.c32",
            "status": "passed",
            "scalar": "Complex32",
            "reference": "independent CPU f32 ABI",
            "channels": ["G", "H", "H_transpose", "E"],
            "batch_count": c32.len(),
            "relative_error": c32_error,
            "tolerance": C32_TOLERANCE,
            "kernel_launches": stats32.kernel_launches,
            "h2d_operations": stats32.transfer.h2d_operations,
            "d2h_operations": stats32.transfer.d2h_operations,
        }),
        json!({
            "id": "regular_pair.c64",
            "status": "passed",
            "scalar": "Complex64",
            "reference": "independent CPU f64 ABI",
            "channels": ["G", "H", "H_transpose", "E"],
            "batch_count": c64.len(),
            "relative_error": c64_error,
            "tolerance": C64_TOLERANCE,
            "kernel_launches": stats64.kernel_launches,
            "h2d_operations": stats64.transfer.h2d_operations,
            "d2h_operations": stats64.transfer.d2h_operations,
        }),
    ])
}

/// Exercise the native source-collocation x derivative on resident C64 pair
/// descriptors.  The reference differentiates the independent CPU f64 pair
/// formula with a centered source-point perturbation; it does not call the
/// CUDA derivative seam or stage geometry through a host-side convenience
/// wrapper.
fn regular_pair_source_x_derivative_row(pool: &MemoryPool<CudaBackend>) -> Result<Value, String> {
    let pairs = pairs_c64();
    let cpu = CpuBackend::new();
    let shifted = |delta: f64| {
        pairs
            .iter()
            .map(|pair| {
                let mut shifted = *pair;
                shifted.source_point[0] += delta;
                shifted
            })
            .collect::<Vec<_>>()
    };
    let plus = cpu
        .try_bem_regular_pair_c64_batch(&shifted(SOURCE_X_DERIVATIVE_STEP), 1.7, -1.0)
        .map_err(|error| format!("CPU source-x derivative plus reference: {error}"))?;
    let minus = cpu
        .try_bem_regular_pair_c64_batch(&shifted(-SOURCE_X_DERIVATIVE_STEP), 1.7, -1.0)
        .map_err(|error| format!("CPU source-x derivative minus reference: {error}"))?;
    let expected = plus
        .iter()
        .zip(&minus)
        .map(|(plus, minus)| sonium_compute::BemRegularPairOutF64 {
            g: (plus.g - minus.g) / (2.0 * SOURCE_X_DERIVATIVE_STEP),
            h: (plus.h - minus.h) / (2.0 * SOURCE_X_DERIVATIVE_STEP),
            ht: (plus.ht - minus.ht) / (2.0 * SOURCE_X_DERIVATIVE_STEP),
            e: (plus.e - minus.e) / (2.0 * SOURCE_X_DERIVATIVE_STEP),
        })
        .collect::<Vec<_>>();

    let mut pairs_device = pool
        .backend()
        .raw_alloc::<BemRegularPairInF64>(pairs.len())
        .map_err(|error| format!("source-x derivative pair allocation: {error}"))?;
    pool.backend()
        .try_copy_h2d(&pairs, &mut pairs_device)
        .map_err(|error| format!("source-x derivative pair upload: {error}"))?;
    let mut output = pool
        .backend()
        .raw_alloc::<sonium_compute::BemRegularPairOutF64>(pairs.len())
        .map_err(|error| format!("source-x derivative output allocation: {error}"))?;

    // Allocation and the one-time geometry upload occur before this baseline.
    // The two actions below therefore prove a resident replay rather than a
    // hidden host-staged convenience path.
    pool.backend().reset_runtime_stats();
    for replay in 0..2 {
        pool.backend()
            .try_bem_regular_pair_c64_source_x_derivative_device(
                &pairs_device,
                &mut output,
                1.7,
                -1.0,
            )
            .map_err(|error| format!("source-x derivative CUDA replay {replay}: {error}"))?;
        pool.backend()
            .try_synchronize()
            .map_err(|error| format!("source-x derivative CUDA replay {replay} sync: {error}"))?;
    }
    let mut warm_actual = vec![sonium_compute::BemRegularPairOutF64::default(); pairs.len()];
    pool.backend()
        .try_copy_d2h(&output, &mut warm_actual)
        .map_err(|error| format!("source-x derivative warm publication: {error}"))?;
    pool.backend()
        .inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = pool
        .backend()
        .try_bem_regular_pair_c64_source_x_derivative_device(&pairs_device, &mut output, 1.7, -1.0);
    let failure_typed = matches!(failure, Err(BackendError::Launch { .. }));
    if !failure_typed {
        return Err(format!(
            "source-x derivative launch failure was not typed: {failure:?}"
        ));
    }
    pool.backend().clear_failure();
    pool.backend()
        .recover_after_error()
        .map_err(|error| format!("source-x derivative recovery: {error}"))?;
    let mut after_failure = vec![sonium_compute::BemRegularPairOutF64::default(); pairs.len()];
    pool.backend()
        .try_copy_d2h(&output, &mut after_failure)
        .map_err(|error| format!("source-x derivative failure publication: {error}"))?;
    let output_preserved = after_failure
        .iter()
        .zip(&warm_actual)
        .map(|(after, warm)| pair64_error(after, warm))
        .fold(0.0, f64::max)
        == 0.0;
    if !output_preserved {
        return Err("source-x derivative launch failure changed the resident output".into());
    }
    let stats = pool.backend().runtime_stats();
    let error = warm_actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| pair64_error(actual, expected))
        .fold(0.0, f64::max);
    if error > SOURCE_X_DERIVATIVE_TOLERANCE {
        return Err(format!(
            "source-x derivative parity {error:.6e} exceeds {SOURCE_X_DERIVATIVE_TOLERANCE:.6e}"
        ));
    }
    if stats.kernel_launches != 2
        || stats.transfer.h2d_operations != 0
        || stats.transfer.d2h_operations != 2
    {
        return Err(format!(
            "source-x derivative residency counters are invalid: launches={}, h2d={}, d2h={}",
            stats.kernel_launches, stats.transfer.h2d_operations, stats.transfer.d2h_operations
        ));
    }
    Ok(json!({
        "id": "regular_pair.c64.source_x_derivative",
        "status": "passed",
        "scalar": "Complex64",
        "reference": "independent CPU f64 centered source-point-x difference",
        "channels": ["dG/dx_source", "dH/dx_source", "dH_transpose/dx_source", "dE/dx_source"],
        "batch_count": pairs.len(),
        "relative_error": error,
        "tolerance": SOURCE_X_DERIVATIVE_TOLERANCE,
        "centered_step": SOURCE_X_DERIVATIVE_STEP,
        "repeated_actions": 2,
        "failure_typed": failure_typed,
        "output_preserved": output_preserved,
        "kernel_launches": stats.kernel_launches,
        "steady_state_h2d_operations": stats.transfer.h2d_operations,
        "publication_d2h_operations": stats.transfer.d2h_operations,
    }))
}

fn quadrature_row(pool: &MemoryPool<CudaBackend>) -> Result<Value, String> {
    let samples = vec![
        BemQuadratureSampleC64::full([1.0, 0.2, 0.7], [0.0, 0.0, 1.0], 0.7, 1.2),
        BemQuadratureSampleC64::singular_bulk([0.3, -0.4, 0.9], [0.0, 0.0, 1.0], 0.4, -0.8),
        BemQuadratureSampleC64::hypersingular_edge([0.4, 0.8, 0.6], [1.0, 0.0, 0.0], 0.25, 1.0),
        BemQuadratureSampleC64::gh_only([-0.2, 0.6, 1.1], [0.0, 0.0, 1.0], 0.35, 0.5),
    ];
    let pairs = vec![BemQuadraturePairC64::new(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        0,
        samples.len() as u64,
    )];
    let operator =
        CudaBemQuadratureOperatorC64::new(pool.clone(), pairs.clone(), samples.clone(), 2.1, -1.0)
            .map_err(|error| format!("quadrature operator: {error}"))?;
    let mut output = pool
        .try_alloc::<sonium_compute::BemRegularPairOutF64>(1)
        .map_err(|error| format!("quadrature output: {error}"))?;
    pool.backend().reset_runtime_stats();
    operator
        .try_apply_device(&mut output)
        .map_err(|error| format!("quadrature first action: {error}"))?;
    operator
        .try_apply_device(&mut output)
        .map_err(|error| format!("quadrature second action: {error}"))?;
    let mut actual = vec![sonium_compute::BemRegularPairOutF64::default()];
    pool.backend()
        .try_copy_d2h(output.raw(), &mut actual)
        .map_err(|error| format!("quadrature publication: {error}"))?;
    // Capture counters after the explicit host publication.  The two device
    // actions above only write the resident workspace and copy it to the
    // device output; neither increments D2H.  Treating a pre-publication
    // snapshot as a publication count would reject a valid resident action.
    let stats = pool.backend().runtime_stats();
    let expected = reference_quadrature(pairs[0], &samples, 2.1, -1.0);
    let error = pair64_error(&actual[0], &expected);
    if error > C64_TOLERANCE {
        return Err(format!(
            "quadrature parity {error:.6e} exceeds {C64_TOLERANCE:.6e}"
        ));
    }
    if stats.transfer.h2d_operations != 0 || stats.transfer.d2h_operations != 1 {
        return Err(format!(
            "quadrature residency counters are invalid: h2d={}, d2h={}",
            stats.transfer.h2d_operations, stats.transfer.d2h_operations
        ));
    }
    Ok(json!({
        "id": "quadrature.c64.bulk-singular-edge",
        "status": "passed",
        "scalar": "Complex64",
        "channels": ["G", "H", "H_transpose", "E"],
        "sample_kinds": ["full", "singular_bulk", "hypersingular_edge", "gh_only"],
        "relative_error": error,
        "tolerance": C64_TOLERANCE,
        "repeated_actions": 2,
        "kernel_launches": stats.kernel_launches,
        "steady_state_h2d_operations": stats.transfer.h2d_operations,
        "publication_d2h_operations": stats.transfer.d2h_operations,
    }))
}

/// Exercise the production chart-driven quadrature adapter with a real batch
/// of near/singular-corrected pair descriptors.  The flattened samples are
/// kept resident by the operator across both device actions; only the final
/// explicit D2H publication crosses the API boundary.
fn quadrature_batch_row(pool: &MemoryPool<CudaBackend>) -> Result<Value, String> {
    let mut pairs = Vec::new();
    let mut samples = Vec::new();
    for index in 0..5 {
        let offset = samples.len() as u64;
        let source = [0.17 * index as f64, -0.11 * index as f64, 0.0];
        let normal = [0.0, 0.0, 1.0];
        samples.push(BemQuadratureSampleC64::full(
            [
                source[0] + 0.35,
                source[1] + 0.10,
                0.55 + 0.03 * index as f64,
            ],
            normal,
            0.7 + 0.01 * index as f64,
            1.2,
        ));
        samples.push(BemQuadratureSampleC64::singular_bulk(
            [
                source[0] - 0.21,
                source[1] + 0.18,
                0.42 + 0.02 * index as f64,
            ],
            normal,
            0.4,
            -0.8,
        ));
        samples.push(BemQuadratureSampleC64::hypersingular_edge(
            [
                source[0] + 0.12,
                source[1] - 0.26,
                0.63 + 0.01 * index as f64,
            ],
            [1.0, 0.0, 0.0],
            0.25,
            1.0,
        ));
        samples.push(BemQuadratureSampleC64::gh_only(
            [
                source[0] - 0.16,
                source[1] - 0.09,
                0.48 + 0.02 * index as f64,
            ],
            normal,
            0.35,
            0.5,
        ));
        pairs.push(BemQuadraturePairC64::new(source, normal, offset, 4));
    }
    let wave_number = 2.1;
    let harmonic_factor = -1.0;
    let operator = CudaBemQuadratureOperatorC64::new(
        pool.clone(),
        pairs.clone(),
        samples.clone(),
        wave_number,
        harmonic_factor,
    )
    .map_err(|error| format!("quadrature batch operator: {error}"))?;
    let mut output = pool
        .try_alloc::<sonium_compute::BemRegularPairOutF64>(pairs.len())
        .map_err(|error| format!("quadrature batch output: {error}"))?;
    pool.backend().reset_runtime_stats();
    operator
        .try_apply_device(&mut output)
        .map_err(|error| format!("quadrature batch first action: {error}"))?;
    operator
        .try_apply_device(&mut output)
        .map_err(|error| format!("quadrature batch warm action: {error}"))?;
    let mut actual = vec![sonium_compute::BemRegularPairOutF64::default(); pairs.len()];
    pool.backend()
        .try_copy_d2h(output.raw(), &mut actual)
        .map_err(|error| format!("quadrature batch publication: {error}"))?;
    let expected = pairs
        .iter()
        .map(|pair| reference_quadrature(*pair, &samples, wave_number, harmonic_factor))
        .collect::<Vec<_>>();
    let error = actual
        .iter()
        .zip(&expected)
        .map(|(got, want)| pair64_error(got, want))
        .fold(0.0, f64::max);
    let stats = pool.backend().runtime_stats();
    let resident = stats.transfer.h2d_operations == 0
        && stats.transfer.d2h_operations == 1
        && stats.kernel_launches == 2
        && stats.transfer.d2d_operations == 2;
    if error > C64_TOLERANCE || !resident {
        return Err(format!(
            "quadrature batch contract failed: error={error:.6e}, h2d={}, d2h={}, d2d={}, launches={}",
            stats.transfer.h2d_operations,
            stats.transfer.d2h_operations,
            stats.transfer.d2d_operations,
            stats.kernel_launches,
        ));
    }
    Ok(json!({
        "id": "quadrature.c64.near-singular-batch",
        "status": "passed",
        "scalar": "Complex64",
        "formulation": "production chart-driven near/singular-corrected quadrature batch",
        "reference": "independent CPU quadrature channel per pair descriptor",
        "sample_kinds": ["full", "singular_bulk", "hypersingular_edge", "gh_only"],
        "pair_count": pairs.len(),
        "pair_tail": pairs.len(),
        "samples_per_pair": 4,
        "sample_count": samples.len(),
        "relative_error": error,
        "tolerance": C64_TOLERANCE,
        "repeated_actions": 2,
        "kernel_launches": stats.kernel_launches,
        "steady_state_h2d_operations": stats.transfer.h2d_operations,
        "publication_d2h_operations": stats.transfer.d2h_operations,
        "warm_d2d_operations": stats.transfer.d2d_operations,
    }))
}

fn field_row(pool: &MemoryPool<CudaBackend>) -> Result<Value, String> {
    let element = BemFieldElementC64::from_triangle(
        [-0.7, -0.3, 0.0],
        [0.8, -0.2, 0.1],
        [-0.1, 0.9, -0.15],
        Complex64::new(1.2, -0.4),
        Complex64::new(-0.15, 0.25),
    );
    let points_host = vec![
        BemFieldPointC64::new([0.3, -0.4, 1.8]),
        BemFieldPointC64::new([-0.8, 0.2, 1.3]),
        BemFieldPointC64::new([0.1, 0.6, 2.1]),
    ];
    let operator = CudaBemFieldOperatorC64::new(pool.clone(), vec![element], 2.3, -1.0, 3.7)
        .map_err(|error| format!("field operator: {error}"))?;
    let mut points = pool
        .try_alloc::<BemFieldPointC64>(points_host.len())
        .map_err(|error| format!("field points: {error}"))?;
    let mut output = pool
        .try_alloc::<Complex64>(points_host.len())
        .map_err(|error| format!("field output: {error}"))?;
    pool.backend()
        .try_copy_h2d(&points_host, points.raw_mut())
        .map_err(|error| format!("field point upload: {error}"))?;
    pool.backend().reset_runtime_stats();
    operator
        .try_apply_device(&points, &mut output)
        .map_err(|error| format!("field first action: {error}"))?;
    operator
        .try_apply_device(&points, &mut output)
        .map_err(|error| format!("field second action: {error}"))?;
    let mut actual = vec![Complex64::default(); points_host.len()];
    pool.backend()
        .try_copy_d2h(output.raw(), &mut actual)
        .map_err(|error| format!("field publication: {error}"))?;
    let stats = pool.backend().runtime_stats();
    let expected = points_host
        .iter()
        .map(|point| field_reference(point.position, element, 2.3, -1.0, 3.7))
        .collect::<Vec<_>>();
    let error = relative_vector_error(&actual, &expected);
    if error > C64_TOLERANCE {
        return Err(format!(
            "field parity {error:.6e} exceeds {C64_TOLERANCE:.6e}"
        ));
    }
    if stats.transfer.h2d_operations != 0 || stats.transfer.d2h_operations != 1 {
        return Err(format!(
            "field residency counters are invalid: h2d={}, d2h={}",
            stats.transfer.h2d_operations, stats.transfer.d2h_operations
        ));
    }
    Ok(json!({
        "id": "field.c64.constant-tri3",
        "status": "passed",
        "scalar": "Complex64",
        "element": "Tri3-constant",
        "relative_error": error,
        "tolerance": C64_TOLERANCE,
        "points": points_host.len(),
        "repeated_actions": 2,
        "kernel_launches": stats.kernel_launches,
        "steady_state_h2d_operations": stats.transfer.h2d_operations,
        "publication_d2h_operations": stats.transfer.d2h_operations,
    }))
}

/// Exercise the constant-panel Müller transmission blocks with the production
/// resident pair and field adapters.  The CUDA seam currently exposes the
/// scalar Green-function channels rather than a fused transmission kernel, so
/// the two media are evaluated natively and the four block combinations are
/// formed from those channel outputs using the same coefficients as the CPU
/// constant-panel transmission assembler.  This is deliberately a pair/field
/// cell, not evidence for a complete CUDA transmission solve.
fn muller_transmission_pair_channels(
    exterior: &[sonium_compute::BemRegularPairOutF64],
    interior: &[sonium_compute::BemRegularPairOutF64],
    exterior_physics: &PhysicsParams,
    interior_physics: &PhysicsParams,
) -> Vec<Complex64> {
    let i_omega = Complex64::new(
        0.0,
        exterior_physics.harmonic_factor * exterior_physics.omega,
    );
    exterior
        .iter()
        .zip(interior)
        .flat_map(|(exterior, interior)| {
            [
                interior.h + exterior.h,
                -i_omega
                    * (interior_physics.density * interior.g
                        + exterior_physics.density * exterior.g),
                interior.e / interior_physics.density + exterior.e / exterior_physics.density,
                -i_omega * (interior.ht + exterior.ht),
            ]
        })
        .collect()
}

fn muller_transmission_row(pool: &MemoryPool<CudaBackend>) -> Result<Value, String> {
    let exterior = PhysicsParams::try_new(540.0, 343.0, 1.21, false)
        .map_err(|error| format!("transmission exterior physics: {error:?}"))?;
    let interior = PhysicsParams::try_new(540.0, 277.0, 2.40, true)
        .map_err(|error| format!("transmission interior physics: {error:?}"))?;
    let pairs = pairs_c64();
    let cpu = CpuBackend::new();
    let cpu_exterior = cpu
        .try_bem_regular_pair_c64_batch(&pairs, exterior.wave_number, exterior.harmonic_factor)
        .map_err(|error| format!("CPU transmission exterior pair reference: {error}"))?;
    let cpu_interior = cpu
        .try_bem_regular_pair_c64_batch(&pairs, interior.wave_number, interior.harmonic_factor)
        .map_err(|error| format!("CPU transmission interior pair reference: {error}"))?;
    let expected_pair =
        muller_transmission_pair_channels(&cpu_exterior, &cpu_interior, &exterior, &interior);

    let baseline_live_bytes = pool.bytes_in_use();
    let exterior_operator = CudaBemRegularPairOperatorC64::new(
        pool.clone(),
        pairs.clone(),
        exterior.wave_number,
        exterior.harmonic_factor,
    )
    .map_err(|error| format!("transmission exterior pair operator: {error}"))?;
    let interior_operator = CudaBemRegularPairOperatorC64::new(
        pool.clone(),
        pairs.clone(),
        interior.wave_number,
        interior.harmonic_factor,
    )
    .map_err(|error| format!("transmission interior pair operator: {error}"))?;
    let mut exterior_output = pool
        .try_alloc::<sonium_compute::BemRegularPairOutF64>(pairs.len())
        .map_err(|error| format!("transmission exterior pair output: {error}"))?;
    let mut interior_output = pool
        .try_alloc::<sonium_compute::BemRegularPairOutF64>(pairs.len())
        .map_err(|error| format!("transmission interior pair output: {error}"))?;

    let pair_first_baseline = pool.backend().runtime_stats();
    exterior_operator
        .try_apply_device(&mut exterior_output)
        .map_err(|error| format!("transmission exterior pair first action: {error}"))?;
    let mut exterior_first = vec![sonium_compute::BemRegularPairOutF64::default(); pairs.len()];
    pool.backend()
        .try_copy_d2h(exterior_output.raw(), &mut exterior_first)
        .map_err(|error| format!("transmission exterior pair first publication: {error}"))?;
    interior_operator
        .try_apply_device(&mut interior_output)
        .map_err(|error| format!("transmission interior pair first action: {error}"))?;
    let mut interior_first = vec![sonium_compute::BemRegularPairOutF64::default(); pairs.len()];
    pool.backend()
        .try_copy_d2h(interior_output.raw(), &mut interior_first)
        .map_err(|error| format!("transmission interior pair first publication: {error}"))?;
    let pair_first_stats = pool.backend().runtime_stats_since(pair_first_baseline);
    let pair_first =
        muller_transmission_pair_channels(&exterior_first, &interior_first, &exterior, &interior);
    let pair_initial_error = relative_vector_error(&pair_first, &expected_pair);

    let pair_warm_baseline = pool.backend().runtime_stats();
    exterior_operator
        .try_apply_device(&mut exterior_output)
        .map_err(|error| format!("transmission exterior pair warm action: {error}"))?;
    let mut exterior_warm = vec![sonium_compute::BemRegularPairOutF64::default(); pairs.len()];
    pool.backend()
        .try_copy_d2h(exterior_output.raw(), &mut exterior_warm)
        .map_err(|error| format!("transmission exterior pair warm publication: {error}"))?;
    interior_operator
        .try_apply_device(&mut interior_output)
        .map_err(|error| format!("transmission interior pair warm action: {error}"))?;
    let mut interior_warm = vec![sonium_compute::BemRegularPairOutF64::default(); pairs.len()];
    pool.backend()
        .try_copy_d2h(interior_output.raw(), &mut interior_warm)
        .map_err(|error| format!("transmission interior pair warm publication: {error}"))?;
    let pair_warm_stats = pool.backend().runtime_stats_since(pair_warm_baseline);
    let pair_warm =
        muller_transmission_pair_channels(&exterior_warm, &interior_warm, &exterior, &interior);
    let pair_warm_error = relative_vector_error(&pair_warm, &expected_pair);

    // The launch injector fires before the native kernel touches its scratch
    // output.  A caller-owned sentinel therefore proves no publication; the
    // recoverable launch disposition permits retry on the same resident
    // operator without another geometry upload.
    let sentinel = sonium_compute::BemRegularPairOutF64 {
        g: Complex64::new(13.0, -7.0),
        h: Complex64::new(11.0, -5.0),
        ht: Complex64::new(9.0, -3.0),
        e: Complex64::new(7.0, -1.0),
    };
    pool.backend()
        .try_copy_h2d(&vec![sentinel; pairs.len()], exterior_output.raw_mut())
        .map_err(|error| format!("transmission pair failure sentinel upload: {error}"))?;
    let failure_baseline = pool.backend().runtime_stats();
    pool.backend()
        .inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = exterior_operator.try_apply_device(&mut exterior_output);
    let failure_stats = pool.backend().runtime_stats_since(failure_baseline);
    let failure_typed = matches!(
        failure,
        Err(BackendError::Launch {
            operation: "cuda_launch",
            ..
        })
    );
    pool.backend()
        .recover_after_error()
        .map_err(|error| format!("recover transmission pair launch failure: {error}"))?;
    let mut failure_actual = vec![sonium_compute::BemRegularPairOutF64::default(); pairs.len()];
    pool.backend()
        .try_copy_d2h(exterior_output.raw(), &mut failure_actual)
        .map_err(|error| format!("transmission pair failure publication: {error}"))?;
    let failure_no_publication = failure_actual.iter().all(|actual| {
        actual.g == sentinel.g
            && actual.h == sentinel.h
            && actual.ht == sentinel.ht
            && actual.e == sentinel.e
    });
    exterior_operator
        .try_apply_device(&mut exterior_output)
        .map_err(|error| format!("transmission pair retry action: {error}"))?;
    let mut exterior_retry = vec![sonium_compute::BemRegularPairOutF64::default(); pairs.len()];
    pool.backend()
        .try_copy_d2h(exterior_output.raw(), &mut exterior_retry)
        .map_err(|error| format!("transmission pair retry publication: {error}"))?;
    let retry_pair =
        muller_transmission_pair_channels(&exterior_retry, &interior_warm, &exterior, &interior);
    let retry_error = relative_vector_error(&retry_pair, &expected_pair);

    let element = BemFieldElementC64::from_triangle(
        [-0.7, -0.3, 0.0],
        [0.8, -0.2, 0.1],
        [-0.1, 0.9, -0.15],
        Complex64::new(1.2, -0.4),
        Complex64::new(-0.15, 0.25),
    );
    let points_host = vec![
        BemFieldPointC64::new([0.3, -0.4, 1.8]),
        BemFieldPointC64::new([-0.8, 0.2, 1.3]),
        BemFieldPointC64::new([0.1, 0.6, 2.1]),
        BemFieldPointC64::new([1.1, -0.5, 1.7]),
        BemFieldPointC64::new([-0.4, 0.9, 2.4]),
    ];
    let field_exterior_operator = CudaBemFieldOperatorC64::new(
        pool.clone(),
        vec![element],
        exterior.wave_number,
        exterior.harmonic_factor,
        exterior.pressure_factor,
    )
    .map_err(|error| format!("transmission exterior field operator: {error}"))?;
    let field_interior_operator = CudaBemFieldOperatorC64::new(
        pool.clone(),
        vec![element],
        interior.wave_number,
        interior.harmonic_factor,
        interior.pressure_factor,
    )
    .map_err(|error| format!("transmission interior field operator: {error}"))?;
    let mut points = pool
        .try_alloc::<BemFieldPointC64>(points_host.len())
        .map_err(|error| format!("transmission field points: {error}"))?;
    pool.backend()
        .try_copy_h2d(&points_host, points.raw_mut())
        .map_err(|error| format!("transmission field point upload: {error}"))?;
    let mut field_exterior_output = pool
        .try_alloc::<Complex64>(points_host.len())
        .map_err(|error| format!("transmission exterior field output: {error}"))?;
    let mut field_interior_output = pool
        .try_alloc::<Complex64>(points_host.len())
        .map_err(|error| format!("transmission interior field output: {error}"))?;
    let expected_exterior = points_host
        .iter()
        .map(|point| {
            field_reference(
                point.position,
                element,
                exterior.wave_number,
                exterior.harmonic_factor,
                exterior.pressure_factor,
            )
        })
        .collect::<Vec<_>>();
    let expected_interior = points_host
        .iter()
        .map(|point| {
            field_reference(
                point.position,
                element,
                interior.wave_number,
                interior.harmonic_factor,
                interior.pressure_factor,
            )
        })
        .collect::<Vec<_>>();

    let field_first_baseline = pool.backend().runtime_stats();
    field_exterior_operator
        .try_apply_device(&points, &mut field_exterior_output)
        .map_err(|error| format!("transmission exterior field first action: {error}"))?;
    let mut field_exterior_first = vec![Complex64::default(); points_host.len()];
    pool.backend()
        .try_copy_d2h(field_exterior_output.raw(), &mut field_exterior_first)
        .map_err(|error| format!("transmission exterior field first publication: {error}"))?;
    field_interior_operator
        .try_apply_device(&points, &mut field_interior_output)
        .map_err(|error| format!("transmission interior field first action: {error}"))?;
    let mut field_interior_first = vec![Complex64::default(); points_host.len()];
    pool.backend()
        .try_copy_d2h(field_interior_output.raw(), &mut field_interior_first)
        .map_err(|error| format!("transmission interior field first publication: {error}"))?;
    let field_first_stats = pool.backend().runtime_stats_since(field_first_baseline);
    let field_exterior_initial_error =
        relative_vector_error(&field_exterior_first, &expected_exterior);
    let field_interior_initial_error =
        relative_vector_error(&field_interior_first, &expected_interior);

    let field_warm_baseline = pool.backend().runtime_stats();
    field_exterior_operator
        .try_apply_device(&points, &mut field_exterior_output)
        .map_err(|error| format!("transmission exterior field warm action: {error}"))?;
    let mut field_exterior_warm = vec![Complex64::default(); points_host.len()];
    pool.backend()
        .try_copy_d2h(field_exterior_output.raw(), &mut field_exterior_warm)
        .map_err(|error| format!("transmission exterior field warm publication: {error}"))?;
    field_interior_operator
        .try_apply_device(&points, &mut field_interior_output)
        .map_err(|error| format!("transmission interior field warm action: {error}"))?;
    let mut field_interior_warm = vec![Complex64::default(); points_host.len()];
    pool.backend()
        .try_copy_d2h(field_interior_output.raw(), &mut field_interior_warm)
        .map_err(|error| format!("transmission interior field warm publication: {error}"))?;
    let field_warm_stats = pool.backend().runtime_stats_since(field_warm_baseline);
    let field_exterior_warm_error = relative_vector_error(&field_exterior_warm, &expected_exterior);
    let field_interior_warm_error = relative_vector_error(&field_interior_warm, &expected_interior);

    let malformed_pair = CudaBemRegularPairOperatorC64::new(
        pool.clone(),
        vec![BemRegularPairInF64::default()],
        exterior.wave_number,
        exterior.harmonic_factor,
    );
    let malformed_pair_typed = matches!(
        malformed_pair,
        Err(BackendError::InvalidArgument {
            operation: "bem_regular_pair_c64_batch",
            ..
        })
    );
    let malformed_field = CudaBemFieldOperatorC64::new(
        pool.clone(),
        Vec::new(),
        exterior.wave_number,
        exterior.harmonic_factor,
        exterior.pressure_factor,
    );
    let malformed_field_typed = matches!(
        malformed_field,
        Err(BackendError::InvalidArgument {
            operation: "cuda_bem_field_operator",
            ..
        })
    );
    let empty_operator = CudaBemRegularPairOperatorC64::new(
        pool.clone(),
        Vec::new(),
        exterior.wave_number,
        exterior.harmonic_factor,
    )
    .map_err(|error| format!("transmission empty pair operator: {error}"))?;
    let mut empty_output = pool
        .try_alloc::<sonium_compute::BemRegularPairOutF64>(0)
        .map_err(|error| format!("transmission empty pair output: {error}"))?;
    pool.backend().reset_runtime_stats();
    empty_operator
        .try_apply_device(&mut empty_output)
        .map_err(|error| format!("transmission empty pair action: {error}"))?;
    let empty_stats = pool.backend().runtime_stats();
    let empty_succeeds = empty_operator.is_empty()
        && empty_output.is_empty()
        && empty_stats.kernel_launches == 0
        && empty_stats.transfer.h2d_operations == 0
        && empty_stats.transfer.d2h_operations == 0;

    let pair_resident = pair_first_stats.allocation_operations > 0
        && pair_first_stats.transfer.h2d_operations == 0
        && pair_first_stats.kernel_launches >= 2
        && pair_first_stats.transfer.d2h_operations >= 2
        && pair_warm_stats.allocation_operations == 0
        && pair_warm_stats.transfer.h2d_operations == 0
        && pair_warm_stats.kernel_launches >= 2
        && pair_warm_stats.transfer.d2h_operations >= 2
        && pair_warm_stats.transfer.d2d_operations >= 2;
    let field_resident = field_first_stats.allocation_operations > 0
        && field_first_stats.transfer.h2d_operations == 0
        && field_first_stats.kernel_launches >= 2
        && field_first_stats.transfer.d2h_operations >= 2
        && field_warm_stats.allocation_operations == 0
        && field_warm_stats.transfer.h2d_operations == 0
        && field_warm_stats.kernel_launches >= 2
        && field_warm_stats.transfer.d2h_operations >= 2
        && field_warm_stats.transfer.d2d_operations >= 2;
    let pair_parity = pair_initial_error <= TRANSMISSION_CUDA_TOLERANCE
        && pair_warm_error <= TRANSMISSION_CUDA_TOLERANCE
        && retry_error <= TRANSMISSION_CUDA_TOLERANCE;
    let field_parity = field_exterior_initial_error <= TRANSMISSION_CUDA_TOLERANCE
        && field_interior_initial_error <= TRANSMISSION_CUDA_TOLERANCE
        && field_exterior_warm_error <= TRANSMISSION_CUDA_TOLERANCE
        && field_interior_warm_error <= TRANSMISSION_CUDA_TOLERANCE;

    let live_before_drop = pool.bytes_in_use();
    drop(empty_output);
    drop(empty_operator);
    drop(field_interior_output);
    drop(field_exterior_output);
    drop(points);
    drop(field_interior_operator);
    drop(field_exterior_operator);
    drop(interior_output);
    drop(exterior_output);
    drop(interior_operator);
    drop(exterior_operator);
    let live_after_drop = pool.bytes_in_use();
    let released = live_before_drop.saturating_sub(live_after_drop);
    let teardown_clean = live_after_drop == baseline_live_bytes;
    if !pair_parity
        || !field_parity
        || !pair_resident
        || !field_resident
        || !failure_typed
        || !failure_no_publication
        || failure_stats.kernel_launches != 0
        || failure_stats.transfer.d2d_operations != 0
        || !malformed_pair_typed
        || !malformed_field_typed
        || !empty_succeeds
        || !teardown_clean
    {
        return Err(format!(
            "transmission pair/field contract failed: pair_initial={pair_initial_error:.6e}, pair_warm={pair_warm_error:.6e}, retry={retry_error:.6e}, field_ext_initial={field_exterior_initial_error:.6e}, field_int_initial={field_interior_initial_error:.6e}, field_ext_warm={field_exterior_warm_error:.6e}, field_int_warm={field_interior_warm_error:.6e}, pair_resident={pair_resident}, field_resident={field_resident}, failure_typed={failure_typed}, no_publication={failure_no_publication}, failure_launches={}, failure_d2d={}, malformed_pair={malformed_pair_typed}, malformed_field={malformed_field_typed}, empty={empty_succeeds}, teardown={teardown_clean}, baseline={baseline_live_bytes}, after={live_after_drop}",
            failure_stats.kernel_launches, failure_stats.transfer.d2d_operations,
        ));
    }
    Ok(json!({
        "id": "production_transmission.c64.constant-tri3",
        "status": "passed",
        "formulation": "constant-panel Muller two-medium transmission pair/field cell",
        "scalar": "Complex64",
        "element": "Tri3-constant",
        "reference": "CPU constant-panel Muller block coefficients over independent CPU medium pair/field channels",
        "media": {"exterior": {"frequency_hz": exterior.frequency, "speed_of_sound": exterior.speed_of_sound, "density": exterior.density}, "interior": {"frequency_hz": interior.frequency, "speed_of_sound": interior.speed_of_sound, "density": interior.density}},
        "pair_count": pairs.len(),
        "pair_tail": pairs.len(),
        "field_points": points_host.len(),
        "field_tail": points_host.len(),
        "pair_initial_relative_error": pair_initial_error,
        "pair_warm_relative_error": pair_warm_error,
        "pair_retry_relative_error": retry_error,
        "field_exterior_initial_relative_error": field_exterior_initial_error,
        "field_interior_initial_relative_error": field_interior_initial_error,
        "field_exterior_warm_relative_error": field_exterior_warm_error,
        "field_interior_warm_relative_error": field_interior_warm_error,
        "tolerance": TRANSMISSION_CUDA_TOLERANCE,
        "pair_initial_kernel_launches": pair_first_stats.kernel_launches,
        "pair_initial_allocations": pair_first_stats.allocation_operations,
        "pair_initial_h2d_operations": pair_first_stats.transfer.h2d_operations,
        "pair_initial_d2h_operations": pair_first_stats.transfer.d2h_operations,
        "pair_warm_kernel_launches": pair_warm_stats.kernel_launches,
        "pair_warm_structural_allocations": pair_warm_stats.allocation_operations,
        "pair_warm_steady_state_h2d_operations": pair_warm_stats.transfer.h2d_operations,
        "pair_warm_publication_d2h_operations": pair_warm_stats.transfer.d2h_operations,
        "pair_warm_d2d_operations": pair_warm_stats.transfer.d2d_operations,
        "field_initial_kernel_launches": field_first_stats.kernel_launches,
        "field_initial_allocations": field_first_stats.allocation_operations,
        "field_initial_h2d_operations": field_first_stats.transfer.h2d_operations,
        "field_initial_d2h_operations": field_first_stats.transfer.d2h_operations,
        "field_warm_kernel_launches": field_warm_stats.kernel_launches,
        "field_warm_structural_allocations": field_warm_stats.allocation_operations,
        "field_warm_steady_state_h2d_operations": field_warm_stats.transfer.h2d_operations,
        "field_warm_publication_d2h_operations": field_warm_stats.transfer.d2h_operations,
        "field_warm_d2d_operations": field_warm_stats.transfer.d2d_operations,
        "failure_typed": failure_typed,
        "failure_no_publication": failure_no_publication,
        "failure_kernel_launches": failure_stats.kernel_launches,
        "failure_d2d_operations": failure_stats.transfer.d2d_operations,
        "malformed_pair_typed": malformed_pair_typed,
        "malformed_field_typed": malformed_field_typed,
        "empty_succeeds": empty_succeeds,
        "empty_kernel_launches": empty_stats.kernel_launches,
        "empty_h2d_operations": empty_stats.transfer.h2d_operations,
        "empty_d2h_operations": empty_stats.transfer.d2h_operations,
        "teardown_live_bytes_before": live_before_drop,
        "teardown_live_bytes_after": live_after_drop,
        "teardown_released_bytes": released,
    }))
}

fn failure_rows(pool: &MemoryPool<CudaBackend>) -> Result<Vec<Value>, String> {
    let baseline = pool.bytes_in_use();
    let pair = pairs_c64().remove(0);
    let operator = CudaBemRegularPairOperatorC64::new(pool.clone(), vec![pair], 1.7, -1.0)
        .map_err(|error| format!("failure regular operator: {error}"))?;
    let sentinel = sonium_compute::BemRegularPairOutF64 {
        g: Complex64::new(7.0, -3.0),
        h: Complex64::new(5.0, -2.0),
        ht: Complex64::new(4.0, -1.0),
        e: Complex64::new(2.0, -6.0),
    };
    let mut output = pool
        .try_alloc::<sonium_compute::BemRegularPairOutF64>(1)
        .map_err(|error| format!("failure output: {error}"))?;
    pool.backend()
        .try_copy_h2d(&[sentinel], output.raw_mut())
        .map_err(|error| format!("failure sentinel upload: {error}"))?;
    pool.backend()
        .inject_failure(CudaFailureInjectionPoint::Launch);
    let error = operator
        .try_apply_device(&mut output)
        .expect_err("injected pair launch failure must be typed");
    let pair_typed = matches!(
        error,
        BackendError::Launch {
            operation: "cuda_launch",
            ..
        }
    );
    pool.backend()
        .recover_after_error()
        .map_err(|error| format!("recover pair launch failure: {error}"))?;
    let mut actual = vec![sonium_compute::BemRegularPairOutF64::default()];
    pool.backend()
        .try_copy_d2h(output.raw(), &mut actual)
        .map_err(|error| format!("pair failure publication: {error}"))?;
    if !pair_typed
        || actual[0].g != sentinel.g
        || actual[0].h != sentinel.h
        || actual[0].ht != sentinel.ht
        || actual[0].e != sentinel.e
    {
        return Err("regular-pair failure was not typed and transactional".into());
    }
    drop(output);
    drop(operator);
    let pair_cleanup = pool.bytes_in_use() == baseline;

    let field_operator = CudaBemFieldOperatorC64::new(
        pool.clone(),
        vec![BemFieldElementC64::from_triangle(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            Complex64::new(1.0, -0.25),
            Complex64::new(0.1, 0.05),
        )],
        1.7,
        -1.0,
        2.3,
    )
    .map_err(|error| format!("failure field operator: {error}"))?;
    let mut points = pool
        .try_alloc::<BemFieldPointC64>(1)
        .map_err(|error| format!("failure field points: {error}"))?;
    let mut field_output = pool
        .try_alloc::<Complex64>(1)
        .map_err(|error| format!("failure field output: {error}"))?;
    pool.backend()
        .try_copy_h2d(&[BemFieldPointC64::new([0.2, 0.3, 1.5])], points.raw_mut())
        .map_err(|error| format!("failure field point upload: {error}"))?;
    let field_sentinel = Complex64::new(7.0, -3.0);
    pool.backend()
        .try_copy_h2d(&[field_sentinel], field_output.raw_mut())
        .map_err(|error| format!("failure field sentinel upload: {error}"))?;
    pool.backend()
        .inject_failure(CudaFailureInjectionPoint::Launch);
    let field_error = field_operator
        .try_apply_device(&points, &mut field_output)
        .expect_err("injected field launch failure must be typed");
    let field_typed = matches!(
        field_error,
        BackendError::Launch {
            operation: "cuda_launch",
            ..
        }
    );
    pool.backend()
        .recover_after_error()
        .map_err(|error| format!("recover field launch failure: {error}"))?;
    let mut field_actual = vec![Complex64::default()];
    pool.backend()
        .try_copy_d2h(field_output.raw(), &mut field_actual)
        .map_err(|error| format!("field failure publication: {error}"))?;
    let field_transactional = field_actual[0] == field_sentinel;
    drop(field_output);
    drop(points);
    drop(field_operator);
    let field_cleanup = pool.bytes_in_use() == baseline;

    let invalid = CudaBemQuadratureOperatorC64::new(
        pool.clone(),
        vec![BemQuadraturePairC64::new([0.0; 3], [0.0, 0.0, 1.0], 1, 1)],
        vec![BemQuadratureSampleC64::default()],
        1.0,
        -1.0,
    );
    let invalid_typed = matches!(
        invalid,
        Err(BackendError::InvalidArgument {
            operation: "cuda_bem_quadrature_operator",
            ..
        })
    );
    if !pair_cleanup || !field_cleanup || !invalid_typed {
        return Err(format!(
            "failure contracts incomplete: pair_cleanup={pair_cleanup}, field_cleanup={field_cleanup}, invalid_typed={invalid_typed}"
        ));
    }
    Ok(vec![
        json!({
            "id": "failure.regular_pair.launch",
            "status": "passed",
            "typed_error": pair_typed,
            "output_preserved": true,
            "cleanup": pair_cleanup,
        }),
        json!({
            "id": "failure.field.launch",
            "status": "passed",
            "typed_error": field_typed,
            "output_preserved": field_transactional,
            "cleanup": field_cleanup,
        }),
        json!({
            "id": "admission.quadrature.invalid_range",
            "status": "passed",
            "typed_error": invalid_typed,
            "dispatch": false,
        }),
        json!({
            "id": "admission.field.quad4-and-on-surface",
            "status": "unsupported",
            "reason": "checked CUDA field seam accepts constant Tri3 off-surface cells only",
        }),
    ])
}

fn run(backend: CudaBackend) -> Result<QaResult, String> {
    if !matches!(
        backend.capabilities().bem.regular_pair_c32,
        sonium_compute::CapabilityState::Native | sonium_compute::CapabilityState::VendorNative
    ) {
        return Err("required CUDA Complex32 regular-pair capability is unavailable".into());
    }
    if !backend.has_bem_regular_pair_c64() {
        return Err("required CUDA Complex64 regular-pair capability is unavailable".into());
    }
    if !backend.has_bem_regular_pair_real_f64() {
        return Err("required CUDA real-f64 Laplace regular-pair capability is unavailable".into());
    }
    if !backend.has_bem_source_x_derivative_c64() {
        return Err("required CUDA Complex64 source-x derivative capability is unavailable".into());
    }
    let started = Instant::now();
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(8 << 20))
        .map_err(|error| format!("B1 CUDA memory pool: {error}"))?;
    let mut rows = regular_pair_rows(pool.backend())?;
    rows.push(regular_pair_source_x_derivative_row(&pool)?);
    rows.push(quadrature_row(&pool)?);
    rows.push(quadrature_batch_row(&pool)?);
    rows.push(field_row(&pool)?);
    rows.push(muller_transmission_row(&pool)?);
    rows.push(production_tbem_row(pool.backend())?);
    rows.push(production_real_f64_row(pool.backend())?);
    rows.extend(failure_rows(&pool)?);
    let passed = rows.iter().filter(|row| row["status"] == "passed").count();
    let unsupported = rows
        .iter()
        .filter(|row| row["status"] == "unsupported")
        .count();
    if passed != 12 || unsupported != 1 {
        return Err(format!(
            "unexpected B1 row counts: passed={passed}, unsupported={unsupported}"
        ));
    }
    // The failure rows intentionally leave the checked backend with no
    // successful launch in its final one-shot stats.  Re-run one checked
    // Complex64 batch for the aggregate runtime evidence after all failures
    // have recovered; the detailed row counters above retain each contract.
    pool.backend().reset_runtime_stats();
    pool.backend()
        .try_bem_regular_pair_c64_batch(&pairs_c64(), 1.0, -1.0)
        .map_err(|error| format!("aggregate B1 CUDA pair action: {error}"))?;
    let stats = pool.backend().runtime_stats();
    if stats.kernel_launches == 0
        || stats.transfer.h2d_operations == 0
        || stats.transfer.d2h_operations == 0
    {
        return Err("B1 producer did not observe native CUDA work".into());
    }
    let artifact_value = json!({
        "schema": "sonium-compute-cuda-bem-b1-evidence/v1",
        "status": "passed",
        "case_id": CASE_ID,
        "hardware": cuda_hardware_provenance(),
        "capabilities": {
            "regular_pair.c32": "native",
            "regular_pair.c64": "native",
            "regular_pair.c64.source_x_derivative": "native",
            "regular_pair.real_f64_laplace": "native",
            "quadrature.c64": "native",
            "quadrature.c64.near-singular-batch": "native",
            "field.c64.constant_tri3": "native",
            "transmission.c64.constant_tri3": "native",
            "field.c64.quad4": "unsupported",
            "field.c64.on_surface": "unsupported",
            "higher_order": "unsupported",
        },
        "rows": rows,
        "runtime": {
            "kernel_launches": stats.kernel_launches,
            "library_operations": stats.library_operations,
            "synchronization_calls": stats.synchronization_calls,
            "h2d_operations": stats.transfer.h2d_operations,
            "d2h_operations": stats.transfer.d2h_operations,
            "allocation_high_water_bytes": stats.allocation_high_water_bytes,
            "live_allocation_bytes": stats.live_allocation_bytes,
        },
        "row_counts": {"passed": passed, "unsupported": unsupported},
        "elapsed_ms": started.elapsed().as_millis(),
    });
    let bytes = serde_json::to_vec_pretty(&artifact_value)
        .map_err(|error| format!("serialize B1 CUDA artifact: {error}"))?;
    let (artifact, checksum) = write_qa_artifact("bem_cuda_pair_field_report.json", &bytes)?;
    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cuda")
        .actual_backend("cuda")
        .backend_passed("cuda")
        .reference(
            "CPU Complex32/Complex64 pair formulas, centered source-x derivative, independent real-f64 Laplace, quadrature/field references, and CPU constant-panel Muller block coefficients",
        )
        .dimension("3-D constant Tri3 regular/source-x derivative, five-pair near/singular-corrected chart quadrature batch, off-surface field, and bounded two-medium transmission pair/field cells")
        .formulation("native CUDA BEM pair, field, and bounded transmission capability/failure matrix")
        .tolerance("Complex32 <= 1e-4; Complex64 <= 3e-12; failure outputs transactional")
        .metric("passed_rows", passed as f64)
        .metric("unsupported_rows", unsupported as f64)
        .metric("cuda_kernel_launches", stats.kernel_launches as f64)
        .metric("cuda_h2d_operations", stats.transfer.h2d_operations as f64)
        .metric("cuda_d2h_operations", stats.transfer.d2h_operations as f64)
        .artifact_with_provenance(
            artifact,
            "crates/sonium-qa/bin/bem_cuda_pair_field.rs",
            "native CUDA B1 pair/field parity and failure matrix",
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
        )))
}

fn failed(error: String, stage: QaFailureStage) -> QaResult {
    QaResult::failed(CASE_ID, error.clone())
        .requested_backend("cuda")
        .actual_backend(if stage == QaFailureStage::Initialization {
            "unavailable"
        } else {
            "cuda"
        })
        .backend_failed("cuda", error)
        .failure_stage(stage)
        .hardware_provenance(cuda_hardware_provenance())
}

fn main() {
    let started = Instant::now();
    let result = match required_cuda_backend() {
        Ok(backend) => match run(backend) {
            Ok(result) => result,
            Err(error) => failed(error, QaFailureStage::Execution),
        },
        Err(error) => failed(
            format!("required NVIDIA CUDA backend unavailable: {error}"),
            QaFailureStage::Initialization,
        ),
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("B1 QA result serialization")
    );
    if result.status != sonium_qa::QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
