//! Bounded native CUDA Quad4/higher-order BEM evidence.
//!
//! The production TBEM matrix adapter remains intentionally Tri3/constant
//! only.  This producer exercises the real production CUDA quadrature seam
//! with a Quad4 isoparametric panel and its nine-function quadratic basis,
//! alongside the existing production Duffy/edge chart for a corrected Tri3
//! self pair.  Every action is resident after construction; no CPU result is
//! accepted as a CUDA fallback.

use num_complex::Complex64;
use serde_json::{Value, json};
use sonium_bem::core::assembly::tbem::{
    build_quad4_cuda_regular_quadrature, build_tri3_cuda_singular_quadrature,
};
use sonium_bem::core::integration::regular::regular_integration_linear;
use sonium_bem::core::integration::singular::{QuadratureParams, singular_integration_with_params};
use sonium_bem::core::types::{
    BoundaryCondition, Element, ElementProperty, ElementType, IntegrationResult, PhysicsParams,
};
use sonium_compute::{
    Backend, BackendError, BemQuadraturePairC64, BemQuadratureSampleC64, BemRegularPairOutF64,
    CudaBackend, CudaBemQuadratureOperatorC64, CudaFailureInjectionPoint, MemoryPool,
    MemoryPoolConfig,
};
use sonium_qa::{
    QaExecutionCounts, QaFailureStage, QaResult, cuda_hardware_provenance, cuda_runtime_evidence,
    required_cuda_backend, write_qa_artifact,
};
use sonium_solvers::dense::DenseMatrix;
use std::time::Instant;

const CASE_ID: &str = "bem.cuda-quad4-real-breadth.v1";
const QUAD_TOLERANCE: f64 = 1.0e-6;
const NEAR_TOLERANCE: f64 = 2.0e-5;
const SINGULAR_TOLERANCE: f64 = 4.0e-5;

fn coords(element: &Element, nodes: &[[f64; 3]]) -> DenseMatrix<f64> {
    DenseMatrix::from_row_major_vec(
        element.connectivity.len(),
        3,
        element
            .connectivity
            .iter()
            .flat_map(|&index| nodes[index])
            .collect(),
    )
}

fn quad4_element(nodes: &[[f64; 3]]) -> Element {
    Element {
        connectivity: (0..4).collect(),
        element_type: ElementType::Quad4,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 4],
        center: [
            nodes.iter().map(|node| node[0]).sum::<f64>() / 4.0,
            nodes.iter().map(|node| node[1]).sum::<f64>() / 4.0,
            nodes.iter().map(|node| node[2]).sum::<f64>() / 4.0,
        ],
        area: 4.01,
        boundary_condition: BoundaryCondition::Pressure(vec![Complex64::new(0.0, 0.0); 9]),
        group: 0,
        dof_addresses: (0..9).collect(),
    }
}

fn tri3_element() -> (Element, Vec<[f64; 3]>) {
    let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let element = Element {
        connectivity: vec![0, 1, 2],
        element_type: ElementType::Tri3,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 3],
        center: [1.0 / 3.0, 1.0 / 3.0, 0.0],
        area: 0.5,
        boundary_condition: BoundaryCondition::Pressure(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses: vec![0],
    };
    (element, nodes)
}

fn output(result: &IntegrationResult) -> BemRegularPairOutF64 {
    BemRegularPairOutF64 {
        g: result.g_integral,
        h: result.dg_dn_integral,
        ht: result.dg_dnx_integral,
        e: result.d2g_dnxdny_integral,
    }
}

fn finite_output(value: &BemRegularPairOutF64) -> bool {
    [
        value.g.re,
        value.g.im,
        value.h.re,
        value.h.im,
        value.ht.re,
        value.ht.im,
        value.e.re,
        value.e.im,
    ]
    .into_iter()
    .all(f64::is_finite)
}

fn output_error(actual: &BemRegularPairOutF64, expected: &BemRegularPairOutF64) -> f64 {
    if !finite_output(actual) || !finite_output(expected) {
        return f64::INFINITY;
    }
    [
        (actual.g, expected.g),
        (actual.h, expected.h),
        (actual.ht, expected.ht),
        (actual.e, expected.e),
    ]
    .into_iter()
    .map(|(got, want)| (got - want).norm() / want.norm().max(1.0))
    .fold(0.0, f64::max)
}

fn vector_error(actual: &[BemRegularPairOutF64], expected: &[BemRegularPairOutF64]) -> f64 {
    if actual.len() != expected.len() || actual.is_empty() {
        return f64::INFINITY;
    }
    actual
        .iter()
        .zip(expected)
        .map(|(got, want)| output_error(got, want))
        .fold(0.0, f64::max)
}

/// Independent scalar CPU evaluation of the descriptor contract.  This is
/// deliberately separate from the production adaptive integrator: it checks
/// the uploaded positions, normals, weights, basis multipliers, and every CUDA
/// channel with the same fixed descriptor set.
fn descriptor_reference(
    pair: BemQuadraturePairC64,
    samples: &[BemQuadratureSampleC64],
    wave_number: f64,
    harmonic_factor: f64,
) -> BemRegularPairOutF64 {
    let mut result = BemRegularPairOutF64::default();
    let start = pair.sample_offset as usize;
    let end = start + pair.sample_count as usize;
    let phase_factor = wave_number * harmonic_factor;
    for sample in &samples[start..end] {
        let difference = [
            sample.position[0] - pair.source_point[0],
            sample.position[1] - pair.source_point[1],
            sample.position[2] - pair.source_point[2],
        ];
        let radius = difference
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if radius < 1.0e-15 {
            continue;
        }
        let unit = difference.map(|value| value / radius);
        let scaled_weight = sample.weight * sample.basis_weight;
        let green = Complex64::from_polar(
            scaled_weight / (4.0 * std::f64::consts::PI * radius),
            phase_factor * radius,
        );
        if sample.kind == BemQuadratureSampleC64::KIND_HYPERSINGULAR_EDGE {
            let gradient = green * Complex64::new(-1.0 / radius, phase_factor);
            let cross = [
                gradient
                    * (unit[1] * sample.edge_direction[2] - unit[2] * sample.edge_direction[1]),
                gradient
                    * (unit[2] * sample.edge_direction[0] - unit[0] * sample.edge_direction[2]),
                gradient
                    * (unit[0] * sample.edge_direction[1] - unit[1] * sample.edge_direction[0]),
            ];
            result.e += cross
                .into_iter()
                .zip(pair.source_normal)
                .map(|(value, normal)| value * normal)
                .sum::<Complex64>();
            continue;
        }
        let r_dot_ny = unit
            .iter()
            .zip(sample.field_normal)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let r_dot_nx = -unit
            .iter()
            .zip(pair.source_normal)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let normal_dot = sample
            .field_normal
            .into_iter()
            .zip(pair.source_normal)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let kernel_factor = Complex64::new(-1.0 / radius, phase_factor);
        result.g += green;
        result.h += green * kernel_factor * r_dot_ny;
        result.ht += green * kernel_factor * r_dot_nx;
        if sample.kind == BemQuadratureSampleC64::KIND_FULL {
            let radius_squared = radius * radius;
            let e_factor = Complex64::new(
                (3.0 / radius_squared - wave_number * wave_number) * r_dot_ny * r_dot_nx
                    + normal_dot / radius_squared,
                -phase_factor / radius * (3.0 * r_dot_ny * r_dot_nx + normal_dot),
            );
            result.e += green * e_factor;
        } else if sample.kind == BemQuadratureSampleC64::KIND_SINGULAR_BULK {
            result.e += green * (wave_number * wave_number * normal_dot);
        }
    }
    result
}

fn stats_value(stats: sonium_compute::CudaRuntimeStatsSnapshot) -> Value {
    json!({
        "kernel_launches": stats.kernel_launches,
        "allocations": stats.allocation_operations,
        "h2d_operations": stats.transfer.h2d_operations,
        "d2h_operations": stats.transfer.d2h_operations,
        "d2d_operations": stats.transfer.d2d_operations,
        "synchronizations": stats.synchronization_calls,
    })
}

struct ActionReport {
    actual: Vec<BemRegularPairOutF64>,
    first: sonium_compute::CudaRuntimeStatsSnapshot,
    warm: sonium_compute::CudaRuntimeStatsSnapshot,
    publication: sonium_compute::CudaRuntimeStatsSnapshot,
    retry: sonium_compute::CudaRuntimeStatsSnapshot,
    failure: sonium_compute::CudaRuntimeStatsSnapshot,
    failure_typed: bool,
    output_preserved: bool,
    retry_error: f64,
}

fn resident_action(
    pool: &MemoryPool<CudaBackend>,
    operator: &CudaBemQuadratureOperatorC64,
    expected: &[BemRegularPairOutF64],
) -> Result<ActionReport, String> {
    let mut output = pool
        .try_alloc::<BemRegularPairOutF64>(expected.len())
        .map_err(|error| format!("allocate resident Quad4 output: {error}"))?;
    pool.backend().reset_runtime_stats();
    let first_baseline = pool.backend().runtime_stats();
    operator
        .try_apply_device(&mut output)
        .map_err(|error| format!("first resident CUDA action: {error}"))?;
    let first = pool.backend().runtime_stats_since(first_baseline);

    let warm_baseline = pool.backend().runtime_stats();
    operator
        .try_apply_device(&mut output)
        .map_err(|error| format!("warm resident CUDA action: {error}"))?;
    let warm = pool.backend().runtime_stats_since(warm_baseline);
    let mut actual = vec![BemRegularPairOutF64::default(); expected.len()];
    pool.backend()
        .try_copy_d2h(output.raw(), &mut actual)
        .map_err(|error| format!("publish resident CUDA output: {error}"))?;
    let publication = pool.backend().runtime_stats_since(warm_baseline);
    let warm_error = vector_error(&actual, expected);

    let failure_baseline = pool.backend().runtime_stats();
    pool.backend()
        .inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = operator.try_apply_device(&mut output);
    let failure_typed = matches!(
        failure,
        Err(BackendError::Launch { .. }) | Err(BackendError::OperationFailed { .. })
    );
    pool.backend()
        .recover_after_error()
        .map_err(|error| format!("recover after injected CUDA launch failure: {error}"))?;
    let mut preserved = vec![BemRegularPairOutF64::default(); expected.len()];
    pool.backend()
        .try_copy_d2h(output.raw(), &mut preserved)
        .map_err(|error| format!("read preserved CUDA output: {error}"))?;
    let output_preserved = vector_error(&preserved, expected) <= 2.0e-12;
    let retry_baseline = pool.backend().runtime_stats();
    operator
        .try_apply_device(&mut output)
        .map_err(|error| format!("retry resident CUDA action: {error}"))?;
    let mut retried = vec![BemRegularPairOutF64::default(); expected.len()];
    pool.backend()
        .try_copy_d2h(output.raw(), &mut retried)
        .map_err(|error| format!("publish retried CUDA output: {error}"))?;
    let retry_error = vector_error(&retried, expected);
    let failure_stats = pool.backend().runtime_stats_since(failure_baseline);
    let retry = pool.backend().runtime_stats_since(retry_baseline);
    let _ = warm_error;
    Ok(ActionReport {
        actual,
        first,
        warm,
        publication,
        retry,
        failure: failure_stats,
        failure_typed,
        output_preserved,
        retry_error,
    })
}

fn check_action(
    report: &ActionReport,
    expected: &[BemRegularPairOutF64],
    tolerance: f64,
) -> Result<(), String> {
    let error = vector_error(&report.actual, expected);
    if error > tolerance || report.retry_error > tolerance {
        return Err(format!(
            "CUDA parity error {error:.6e}, retry {retry:.6e} exceeds {tolerance:.6e}",
            retry = report.retry_error
        ));
    }
    if report.first.kernel_launches == 0
        || report.warm.kernel_launches == 0
        || report.publication.transfer.d2h_operations == 0
        || report.warm.allocation_operations != 0
        || report.warm.transfer.h2d_operations != 0
        || !report.failure_typed
        || !report.output_preserved
    {
        return Err(format!(
            "resident/failure contract failed: first={:?}, warm={:?}, publication={:?}, failure_typed={}, preserved={}",
            stats_value(report.first),
            stats_value(report.warm),
            stats_value(report.publication),
            report.failure_typed,
            report.output_preserved
        ));
    }
    Ok(())
}

fn quad4_case(
    pool: &MemoryPool<CudaBackend>,
    nodes: &[[f64; 3]],
    element: &Element,
    source_point: [f64; 3],
    gauss_order: usize,
    id: &'static str,
    tolerance: f64,
) -> Result<Value, String> {
    let physics = PhysicsParams::try_new(180.0, 343.0, 1.21, false)
        .map_err(|error| format!("Quad4 physics: {error:?}"))?;
    let (pairs, samples) = build_quad4_cuda_regular_quadrature(
        element,
        nodes,
        source_point,
        [0.0, 0.0, 1.0],
        gauss_order,
    )
    .map_err(|error| format!("build production Quad4 CUDA descriptors: {error}"))?;
    let element_coords = coords(element, nodes);
    let production = regular_integration_linear(
        &source_point,
        &[0.0, 0.0, 1.0],
        &element_coords,
        ElementType::Quad4,
        element.area,
        &physics,
        9,
    );
    let production_expected: Vec<_> = production.basis_results.iter().map(output).collect();
    let expected: Vec<_> = pairs
        .iter()
        .map(|pair| {
            descriptor_reference(
                *pair,
                &samples,
                physics.wave_number,
                physics.harmonic_factor,
            )
        })
        .collect();
    let cpu_crosscheck_error = vector_error(&expected, &production_expected);
    if cpu_crosscheck_error > tolerance {
        return Err(format!(
            "independent descriptor/production Quad4 CPU cross-check {cpu_crosscheck_error:.6e} exceeds {tolerance:.6e}; descriptor={expected:?}; production={production_expected:?}"
        ));
    }
    let malformed = pairs[0].sample_count.checked_add(samples.len() as u64);
    let mut malformed_pair = pairs[0];
    malformed_pair.sample_offset = malformed.unwrap_or(u64::MAX);
    let malformed_rejected = CudaBemQuadratureOperatorC64::new(
        pool.clone(),
        vec![malformed_pair],
        samples.clone(),
        physics.wave_number,
        physics.harmonic_factor,
    )
    .is_err();
    if !malformed_rejected {
        return Err("malformed Quad4 sample range was accepted".into());
    }

    let pair_count = pairs.len();
    let samples_per_pair = pairs[0].sample_count;
    let operator = CudaBemQuadratureOperatorC64::new(
        pool.clone(),
        pairs,
        samples,
        physics.wave_number,
        physics.harmonic_factor,
    )
    .map_err(|error| format!("construct resident Quad4 CUDA operator: {error}"))?;
    let report = resident_action(pool, &operator, &expected)?;
    check_action(&report, &expected, tolerance)?;
    Ok(json!({
        "id": id,
        "status": "passed",
        "scalar": "Complex64",
        "element": "Quad4-isoparametric/quadratic-9",
        "reference": "independent fixed-descriptor CPU channels cross-checked with production regular_integration_linear",
        "pair_count": pair_count,
        "samples_per_pair": samples_per_pair,
        "tail_samples": samples_per_pair % 2 == 1,
        "relative_error": vector_error(&report.actual, &expected),
        "cpu_production_crosscheck_error": cpu_crosscheck_error,
        "tolerance": tolerance,
        "initial": stats_value(report.first),
        "warm": stats_value(report.warm),
        "publication": stats_value(report.publication),
        "failure": stats_value(report.failure),
        "retry": stats_value(report.retry),
        "failure_typed": report.failure_typed,
        "output_preserved": report.output_preserved,
        "retry_relative_error": report.retry_error,
        "malformed_rejected": malformed_rejected,
    }))
}

fn singular_case(pool: &MemoryPool<CudaBackend>) -> Result<Value, String> {
    let (element, nodes) = tri3_element();
    let physics = PhysicsParams::try_new(180.0, 343.0, 1.21, false)
        .map_err(|error| format!("singular physics: {error:?}"))?;
    let (pair, samples) = build_tri3_cuda_singular_quadrature(&element, &nodes, &physics)
        .map_err(|error| format!("build production Duffy/edge chart: {error}"))?;
    let element_coords = coords(&element, &nodes);
    let quadrature = QuadratureParams::for_ka(physics.wave_number * element.area.sqrt());
    let production = singular_integration_with_params(
        &element.center,
        &element.normal,
        &element_coords,
        ElementType::Tri3,
        &physics,
        None,
        0,
        false,
        &quadrature,
        None,
    );
    let production_expected = output(&production);
    let expected =
        descriptor_reference(pair, &samples, physics.wave_number, physics.harmonic_factor);
    let cpu_crosscheck_error = output_error(&expected, &production_expected);
    if cpu_crosscheck_error > SINGULAR_TOLERANCE {
        return Err(format!(
            "independent singular descriptor/production cross-check {cpu_crosscheck_error:.6e} exceeds {SINGULAR_TOLERANCE:.6e}"
        ));
    }
    let mut malformed_pair = pair;
    malformed_pair.sample_offset = samples.len() as u64 + 1;
    let malformed_rejected = CudaBemQuadratureOperatorC64::new(
        pool.clone(),
        vec![malformed_pair],
        samples.clone(),
        physics.wave_number,
        physics.harmonic_factor,
    )
    .is_err();
    if !malformed_rejected {
        return Err("malformed singular sample range was accepted".into());
    }
    let operator = CudaBemQuadratureOperatorC64::new(
        pool.clone(),
        vec![pair],
        samples.clone(),
        physics.wave_number,
        physics.harmonic_factor,
    )
    .map_err(|error| format!("construct resident singular CUDA operator: {error}"))?;
    let report = resident_action(pool, &operator, &[expected])?;
    check_action(&report, &[expected], SINGULAR_TOLERANCE)?;
    Ok(json!({
        "id": "production.quadrature.tri3-singular-corrected",
        "status": "passed",
        "scalar": "Complex64",
        "element": "Tri3-constant/Duffy-edge-corrected",
        "reference": "independent descriptor CPU channels cross-checked with production singular_integration_with_params",
        "pair_count": 1,
        "samples_per_pair": pair.sample_count,
        "relative_error": vector_error(&report.actual, &[expected]),
        "cpu_production_crosscheck_error": cpu_crosscheck_error,
        "tolerance": SINGULAR_TOLERANCE,
        "initial": stats_value(report.first),
        "warm": stats_value(report.warm),
        "publication": stats_value(report.publication),
        "failure": stats_value(report.failure),
        "retry": stats_value(report.retry),
        "failure_typed": report.failure_typed,
        "output_preserved": report.output_preserved,
        "retry_relative_error": report.retry_error,
        "malformed_rejected": malformed_rejected,
    }))
}

fn run(backend: CudaBackend) -> Result<QaResult, String> {
    let started = Instant::now();
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28))
        .map_err(|error| format!("CUDA Quad4 evidence memory pool: {error}"))?;
    let nodes = vec![
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.12],
        [1.0, 1.0, -0.08],
        [-1.0, 1.0, 0.05],
    ];
    let element = quad4_element(&nodes);
    let regular = quad4_case(
        &pool,
        &nodes,
        &element,
        [0.27, -0.19, 6.0],
        13,
        "production.quad4.quadratic-regular",
        QUAD_TOLERANCE,
    )?;
    let near = quad4_case(
        &pool,
        &nodes,
        &element,
        [0.27, -0.19, 1.0],
        32,
        "production.quad4.quadratic-near-regular",
        NEAR_TOLERANCE,
    )?;
    let singular = singular_case(&pool)?;
    let aggregate = pool.backend().runtime_stats();
    let rows = vec![regular, near, singular];
    let artifact_value = json!({
        "schema": "sonium-bem-cuda-b1-quad4-real-evidence/v1",
        "status": "passed",
        "case_id": CASE_ID,
        "hardware": cuda_hardware_provenance(),
        "dispatch": {
            "backend": "cuda",
            "native": true,
            "production_adapter": "CudaBemQuadratureOperatorC64",
            "cuda_sm": "SM120",
        },
        "rows": rows,
        "unsupported_boundaries": [
            "full Quad4 TBEM matrix assembly",
            "Quad4 CUDA field post-processing",
            "general higher-order singular field matrix",
        ],
        "runtime": stats_value(aggregate),
        "elapsed_ms": started.elapsed().as_millis(),
    });
    let bytes = serde_json::to_vec_pretty(&artifact_value)
        .map_err(|error| format!("serialize Quad4 CUDA artifact: {error}"))?;
    let (artifact, checksum) = write_qa_artifact("bem_cuda_quad4_real_report.json", &bytes)?;
    let stats = pool.backend().runtime_stats();
    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cuda")
        .actual_backend("cuda")
        .backend_passed("cuda")
        .reference(
            "independent fixed-descriptor CPU Green-function channels cross-checked with production Quad4 regular and Tri3 singular integration",
        )
        .dimension("3-D production Quad4 isoparametric quadratic-9 regular/near pair actions plus corrected Tri3 Duffy/edge pair")
        .formulation("bounded native CUDA BEM quadrature breadth; full Quad4 TBEM matrix and field paths remain unsupported")
        .tolerance("regular <= 1e-6; near <= 2e-5; corrected singular <= 4e-5; typed failure preserves publication")
        .metric("passed_rows", 3.0)
        .metric("cuda_kernel_launches", stats.kernel_launches as f64)
        .metric("cuda_h2d_operations", stats.transfer.h2d_operations as f64)
        .metric("cuda_d2h_operations", stats.transfer.d2h_operations as f64)
        .elapsed_ms(started.elapsed().as_millis() as u64)
        .artifact_with_provenance(
            artifact,
            "crates/sonium-bem/src/core/assembly/tbem.rs + crates/sonium-qa/bin/bem_cuda_quad4_real.rs",
            "native CUDA Quad4/higher-order production quadrature and corrected Tri3 evidence",
            "v1",
            checksum,
        )
        .runtime_evidence(cuda_runtime_evidence(
            stats,
            QaExecutionCounts {
                executed: 3,
                passed: 3,
                ..QaExecutionCounts::default()
            },
        )))
}

fn failed(error: String, stage: QaFailureStage) -> QaResult {
    QaResult::failed(CASE_ID, error.clone())
        .requested_backend("cuda")
        .actual_backend("unavailable")
        .backend_failed("cuda", error)
        .failure_stage(stage)
        .hardware_provenance(cuda_hardware_provenance())
}

fn main() {
    let started = Instant::now();
    let result = match required_cuda_backend() {
        Ok(backend) => {
            run(backend).unwrap_or_else(|error| failed(error, QaFailureStage::Execution))
        }
        Err(error) => failed(
            format!("required NVIDIA CUDA backend unavailable: {error}"),
            QaFailureStage::Initialization,
        ),
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);
    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("Quad4 CUDA QA result serialization")
    );
    if result.status != sonium_qa::QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("PASSED");
}
