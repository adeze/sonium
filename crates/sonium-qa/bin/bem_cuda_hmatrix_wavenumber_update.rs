//! Native CUDA evidence for a resident production H-matrix wavenumber update.
//!
//! The BEM crate builds the value and analytic fixed-geometry derivative with
//! its production HCA path.  The CUDA adapter uploads both compressed layouts
//! once and combines `A(k)x + delta_k A'(k)x` on device for repeated frequency
//! updates.  The dense TBEM finite-difference action is an independent CPU
//! oracle; this packet does not claim device-side tree construction.

use num_complex::Complex64;
use serde_json::json;
use sonium_bem::core::assembly::build_tbem_system_stabilized;
use sonium_bem::core::assembly::hmatrix::{
    AcaVariant, LowRankPrecision, apply_tbem_stabilized_exact_wavenumber_derivative,
    build_hmatrix_stabilized, build_hmatrix_stabilized_wavenumber_derivative,
};
use sonium_bem::core::types::{BoundaryCondition, Element, ElementProperty, ElementType};
use sonium_bem::core::{Cluster, PhysicsParams};
use sonium_compute::{
    Backend, BackendError, CudaBackend, CudaFailureInjectionPoint, MemoryPool, MemoryPoolConfig,
};
use sonium_qa::{
    QaExecutionCounts, QaFailureStage, QaResult, cuda_hardware_provenance, cuda_runtime_evidence,
    required_cuda_backend, write_qa_artifact,
};
use std::time::Instant;

const CASE_ID: &str = "bem.cuda-hmatrix-wavenumber-update.v1";
const ARTIFACT: &str = "bem_cuda_hmatrix_wavenumber_update_report.json";
const FREQUENCY_HZ: f64 = 42.0;
const SPEED_OF_SOUND: f64 = 343.0;
const DENSITY: f64 = 1.21;
const PATCH_SEPARATION: f64 = 12.0;
const ELEMENT_COUNT: usize = 16;
const ACA_TOLERANCE: f64 = 1.0e-4;
const MAX_RANK: usize = 16;
const GAUSS_ORDER: usize = 6;
const WAVENUMBER_STEP: f64 = 1.0e-4;
const ACTION_TOLERANCE: f64 = 1.0e-7;
const DERIVATIVE_TOLERANCE: f64 = 3.0e-5;
const UPDATE_TOLERANCE: f64 = 3.0e-5;

fn failed_cuda(error: String, actual_backend: &str, stage: QaFailureStage) -> QaResult {
    QaResult::failed(CASE_ID, error.clone())
        .requested_backend("cuda")
        .actual_backend(actual_backend)
        .backend_failed("cuda", error)
        .failure_stage(stage)
        .hardware_provenance(cuda_hardware_provenance())
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
    let denominator = expected
        .iter()
        .map(Complex64::norm_sqr)
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    numerator / denominator
}

fn physics_at_wavenumber(wavenumber: f64) -> PhysicsParams {
    PhysicsParams::try_new(
        wavenumber * SPEED_OF_SOUND / std::f64::consts::TAU,
        SPEED_OF_SOUND,
        DENSITY,
        false,
    )
    .expect("finite positive B2 wavenumber")
}

fn fixture() -> (Vec<Element>, Vec<[f64; 3]>, Vec<Cluster>, PhysicsParams) {
    let mut nodes = Vec::new();
    let mut elements = Vec::new();
    for patch in 0..2 {
        let x_offset = patch as f64 * PATCH_SEPARATION;
        for local in 0..8 {
            let x = x_offset + local as f64 * 1.1;
            let base = nodes.len();
            nodes.extend_from_slice(&[[x, 0.0, 0.0], [x + 0.8, 0.0, 0.0], [x, 0.8, 0.0]]);
            elements.push(Element {
                connectivity: vec![base, base + 1, base + 2],
                element_type: ElementType::Tri3,
                property: ElementProperty::Surface,
                normal: [0.0, 0.0, 1.0],
                node_normals: vec![[0.0, 0.0, 1.0]; 3],
                center: [x + 0.8 / 3.0, 0.8 / 3.0, 0.0],
                area: 0.32,
                boundary_condition: BoundaryCondition::Velocity(vec![Complex64::default()]),
                group: 0,
                dof_addresses: vec![patch * 8 + local],
            });
        }
    }
    let mut left = Cluster::new([4.12, 0.27, 0.0]);
    left.element_indices = (0..8).collect();
    left.num_elements = 8;
    left.radius = 4.5;
    left.near_clusters = vec![0];
    left.far_clusters = vec![1];
    let mut right = Cluster::new([16.12, 0.27, 0.0]);
    right.element_indices = (8..16).collect();
    right.num_elements = 8;
    right.radius = 4.5;
    right.near_clusters = vec![1];
    right.far_clusters = vec![0];
    let physics = PhysicsParams::try_new(FREQUENCY_HZ, SPEED_OF_SOUND, DENSITY, false)
        .expect("valid B2 production physics");
    (elements, nodes, vec![left, right], physics)
}

fn probe_vector(length: usize) -> Vec<Complex64> {
    (0..length)
        .map(|index| {
            Complex64::new(
                (0.37 * index as f64 + 0.1).sin(),
                (0.19 * index as f64).cos(),
            )
        })
        .collect()
}

fn centered_difference(plus: &[Complex64], minus: &[Complex64], step: f64) -> Vec<Complex64> {
    plus.iter()
        .zip(minus)
        .map(|(plus, minus)| (*plus - *minus) / (2.0 * step))
        .collect()
}

fn read_device(
    backend: &CudaBackend,
    buffer: &sonium_compute::pool::Buffer<Complex64, CudaBackend>,
) -> Result<Vec<Complex64>, String> {
    let mut values = vec![Complex64::new(0.0, 0.0); buffer.len()];
    backend
        .try_copy_d2h(buffer.raw(), &mut values)
        .map_err(|error| format!("copy device output: {error}"))?;
    Ok(values)
}

fn taylor(base: &[Complex64], derivative: &[Complex64], delta: f64) -> Vec<Complex64> {
    base.iter()
        .zip(derivative)
        .map(|(base, derivative)| *base + *derivative * delta)
        .collect()
}

fn run(backend: CudaBackend) -> Result<QaResult, String> {
    if !backend.has_fmm_c64() {
        return Err("CUDA FMM module is not loaded; refusing a fallback result".into());
    }
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(512 << 20))
        .map_err(|error| format!("create CUDA memory pool: {error}"))?;
    let backend = pool.backend();
    let (elements, nodes, clusters, physics) = fixture();
    let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);
    let value_hmatrix = build_hmatrix_stabilized(
        &elements,
        &nodes,
        &clusters,
        &physics,
        ACA_TOLERANCE,
        MAX_RANK,
        sign,
        1.0,
    );
    let derivative_hmatrix = build_hmatrix_stabilized_wavenumber_derivative(
        &elements,
        &nodes,
        &clusters,
        &physics,
        ACA_TOLERANCE,
        MAX_RANK,
        AcaVariant::Hca,
        LowRankPrecision::F64,
        sign,
        1.0,
        GAUSS_ORDER,
    )
    .map_err(|error| format!("build production H-matrix derivative: {error}"))?;
    if value_hmatrix.low_rank_blocks.is_empty() || derivative_hmatrix.low_rank_blocks.is_empty() {
        return Err("production value and derivative must both retain a low-rank block".into());
    }

    let probe = probe_vector(elements.len());
    let value_cpu = value_hmatrix
        .matvec_checked(&probe)
        .map_err(|error| format!("value CPU action: {error}"))?;
    let derivative_cpu = derivative_hmatrix
        .matvec_checked(&probe)
        .map_err(|error| format!("derivative CPU action: {error}"))?;
    let dense_base = build_tbem_system_stabilized(&elements, &nodes, &physics, sign, 1.0)
        .matrix
        .matvec(&probe);
    let plus_physics = physics_at_wavenumber(physics.wave_number + WAVENUMBER_STEP);
    let minus_physics = physics_at_wavenumber(physics.wave_number - WAVENUMBER_STEP);
    let dense_plus = build_tbem_system_stabilized(&elements, &nodes, &plus_physics, sign, 1.0)
        .matrix
        .matvec(&probe);
    let dense_minus = build_tbem_system_stabilized(&elements, &nodes, &minus_physics, sign, 1.0)
        .matrix
        .matvec(&probe);
    let centered_dense_derivative = centered_difference(&dense_plus, &dense_minus, WAVENUMBER_STEP);
    let exact_derivative = apply_tbem_stabilized_exact_wavenumber_derivative(
        &elements, &nodes, &physics, sign, 1.0, &probe,
    )
    .map_err(|error| format!("exact CPU derivative oracle: {error}"))?;
    let value_action_error = relative_error(&value_cpu, &dense_base);
    let derivative_action_error = relative_error(&derivative_cpu, &exact_derivative);
    let derivative_fd_error = relative_error(&exact_derivative, &centered_dense_derivative);
    if !value_action_error.is_finite() || value_action_error > ACTION_TOLERANCE {
        return Err(format!("value H-matrix residual={value_action_error:.6e}"));
    }
    if !derivative_action_error.is_finite() || derivative_action_error > DERIVATIVE_TOLERANCE {
        return Err(format!(
            "derivative H-matrix residual={derivative_action_error:.6e}"
        ));
    }
    if !derivative_fd_error.is_finite() || derivative_fd_error > DERIVATIVE_TOLERANCE {
        return Err(format!(
            "derivative finite-difference residual={derivative_fd_error:.6e}"
        ));
    }

    backend.reset_runtime_stats();
    let setup_start = backend.begin_runtime_stats();
    let mut update = value_hmatrix
        .try_to_cuda_wavenumber_update(&derivative_hmatrix, pool.clone(), physics.wave_number)
        .map_err(|error| format!("upload resident production update: {error}"))?;
    let setup_stats = backend.runtime_stats_since(setup_start);
    if setup_stats.allocation_operations == 0 || setup_stats.transfer.h2d_operations == 0 {
        return Err(format!(
            "resident setup did not upload/allocate: {setup_stats:?}"
        ));
    }

    let mut input = pool
        .try_alloc::<Complex64>(probe.len())
        .map_err(|error| format!("allocate input: {error}"))?;
    let mut input_seed = pool
        .try_alloc::<Complex64>(probe.len())
        .map_err(|error| format!("allocate input seed: {error}"))?;
    let sentinel = Complex64::new(73.0, -11.0);
    let output_seed = vec![sentinel; probe.len()];
    let mut output = pool
        .try_alloc::<Complex64>(probe.len())
        .map_err(|error| format!("allocate output: {error}"))?;
    let mut reset = pool
        .try_alloc::<Complex64>(probe.len())
        .map_err(|error| format!("allocate output reset: {error}"))?;
    backend
        .try_copy_h2d(&probe, input.raw_mut())
        .map_err(|error| format!("upload input: {error}"))?;
    backend
        .try_copy_h2d(&probe, input_seed.raw_mut())
        .map_err(|error| format!("upload input seed: {error}"))?;
    backend
        .try_copy_h2d(&output_seed, output.raw_mut())
        .map_err(|error| format!("upload output: {error}"))?;
    backend
        .try_copy_h2d(&output_seed, reset.raw_mut())
        .map_err(|error| format!("upload output reset: {error}"))?;

    backend.reset_runtime_stats();
    let initial_start = backend.begin_runtime_stats();
    update
        .try_apply(&input, &mut output, physics.wave_number, 0.0)
        .map_err(|error| format!("initial resident value action: {error}"))?;
    let initial_stats = backend.runtime_stats_since(initial_start);
    let initial_actual = read_device(backend, &output)?;
    let initial_error = relative_error(&initial_actual, &value_cpu);
    if initial_error > UPDATE_TOLERANCE {
        return Err(format!("initial CUDA value error={initial_error:.6e}"));
    }

    backend
        .try_copy_h2d(&output_seed, output.raw_mut())
        .map_err(|error| format!("reset stale metadata output: {error}"))?;
    let stale = update
        .try_apply(
            &input,
            &mut output,
            physics.wave_number + 0.01,
            WAVENUMBER_STEP,
        )
        .expect_err("stale wavenumber metadata must be rejected");
    if !matches!(stale, BackendError::InvalidArgument { .. }) {
        return Err(format!("stale metadata error was not typed: {stale}"));
    }
    let stale_output = read_device(backend, &output)?;
    let stale_preserved = stale_output == output_seed;

    let mut malformed_input = pool
        .try_alloc::<Complex64>(probe.len() - 1)
        .map_err(|error| format!("allocate malformed input: {error}"))?;
    backend
        .try_copy_h2d(&probe[..probe.len() - 1], malformed_input.raw_mut())
        .map_err(|error| format!("upload malformed input: {error}"))?;
    backend
        .try_copy_h2d(&output_seed, output.raw_mut())
        .map_err(|error| format!("reset malformed output: {error}"))?;
    let malformed = update
        .try_apply(
            &malformed_input,
            &mut output,
            physics.wave_number,
            WAVENUMBER_STEP,
        )
        .expect_err("malformed vector length must be rejected");
    if !matches!(malformed, BackendError::InvalidArgument { .. }) {
        return Err(format!("malformed vector error was not typed: {malformed}"));
    }
    let malformed_output = read_device(backend, &output)?;
    let malformed_preserved = malformed_output == output_seed;

    backend
        .try_copy_h2d(&output_seed, output.raw_mut())
        .map_err(|error| format!("seed launch-failure output: {error}"))?;
    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let launch_failure = update
        .try_apply(&input, &mut output, physics.wave_number, WAVENUMBER_STEP)
        .expect_err("injected resident update launch failure must be typed");
    if !matches!(launch_failure, BackendError::Launch { .. }) {
        return Err(format!("launch failure was not typed: {launch_failure}"));
    }
    backend.clear_failure();
    backend
        .recover_after_error()
        .map_err(|error| format!("recover after launch failure: {error}"))?;
    let failure_output = read_device(backend, &output)?;
    let failure_preserved = failure_output == output_seed;
    if !failure_preserved || !stale_preserved || !malformed_preserved {
        return Err("invalid or failed update modified caller output".into());
    }
    update
        .try_apply(&input, &mut output, physics.wave_number, WAVENUMBER_STEP)
        .map_err(|error| format!("resident update retry: {error}"))?;
    let retry_actual = read_device(backend, &output)?;
    let retry_expected = taylor(&dense_base, &exact_derivative, WAVENUMBER_STEP);
    let retry_error = relative_error(&retry_actual, &retry_expected);
    if retry_error > UPDATE_TOLERANCE {
        return Err(format!("retry Taylor error={retry_error:.6e}"));
    }

    backend.reset_runtime_stats();
    let warm_start = backend.begin_runtime_stats();
    let mut warm_errors = Vec::new();
    for delta in [-WAVENUMBER_STEP, WAVENUMBER_STEP] {
        backend
            .try_copy_d2d(input_seed.raw(), input.raw_mut())
            .map_err(|error| format!("warm input D2D refresh: {error}"))?;
        backend
            .try_copy_d2d(reset.raw(), output.raw_mut())
            .map_err(|error| format!("warm output D2D reset: {error}"))?;
        update
            .try_apply(&input, &mut output, physics.wave_number, delta)
            .map_err(|error| format!("warm resident update: {error}"))?;
        let actual = read_device(backend, &output)?;
        warm_errors.push(relative_error(
            &actual,
            &taylor(&dense_base, &exact_derivative, delta),
        ));
    }
    let warm_stats = backend.runtime_stats_since(warm_start);
    let warm_error = warm_errors.iter().copied().fold(0.0, f64::max);
    if !warm_error.is_finite()
        || warm_error > UPDATE_TOLERANCE
        || warm_stats.allocation_operations != 0
        || warm_stats.transfer.h2d_operations != 0
        || warm_stats.transfer.d2d_operations < 4
        || warm_stats.transfer.d2h_operations != 2
        || warm_stats.kernel_launches == 0
    {
        return Err(format!(
            "warm resident update contract failed: {warm_stats:?}"
        ));
    }

    let runtime_stats = backend.runtime_stats();
    let teardown_before = runtime_stats.live_allocation_bytes;
    let teardown_high_water = runtime_stats.allocation_high_water_bytes;
    let value_low_rank_blocks = value_hmatrix.low_rank_blocks.len();
    let derivative_low_rank_blocks = derivative_hmatrix.low_rank_blocks.len();
    let value_compression_ratio = value_hmatrix.compression_ratio;
    let derivative_compression_ratio = derivative_hmatrix.compression_ratio;
    drop(update);
    drop(input);
    drop(input_seed);
    drop(output);
    drop(reset);
    drop(malformed_input);
    let teardown_after = backend.runtime_stats().live_allocation_bytes;
    if teardown_before == 0 || teardown_after >= teardown_before {
        return Err(format!(
            "resident teardown did not release bytes: before={teardown_before}, after={teardown_after}"
        ));
    }

    let warm_replays = 2.0;
    let mut result = QaResult::passed(CASE_ID)
        .requested_backend("cuda")
        .actual_backend("cuda")
        .backend_passed("cuda")
        .hardware_provenance(cuda_hardware_provenance())
        .metric("value_action_relative_error", value_action_error)
        .metric("derivative_action_relative_error", derivative_action_error)
        .metric(
            "derivative_finite_difference_relative_error",
            derivative_fd_error,
        )
        .metric("initial_update_relative_error", initial_error)
        .metric("retry_update_relative_error", retry_error)
        .metric("warm_update_relative_error", warm_error)
        .metric("warm_replays", warm_replays)
        .metric(
            "warm_structural_allocations",
            warm_stats.allocation_operations as f64,
        )
        .metric(
            "warm_h2d_operations",
            warm_stats.transfer.h2d_operations as f64,
        )
        .metric(
            "warm_d2d_operations",
            warm_stats.transfer.d2d_operations as f64,
        )
        .metric(
            "warm_d2h_publications",
            warm_stats.transfer.d2h_operations as f64,
        )
        .metric(
            "warm_native_kernel_launches",
            warm_stats.kernel_launches as f64,
        )
        .metric(
            "setup_allocations",
            setup_stats.allocation_operations as f64,
        )
        .metric(
            "setup_h2d_operations",
            setup_stats.transfer.h2d_operations as f64,
        )
        .metric(
            "initial_native_kernel_launches",
            initial_stats.kernel_launches as f64,
        )
        .metric("value_low_rank_blocks", value_low_rank_blocks as f64)
        .metric(
            "derivative_low_rank_blocks",
            derivative_low_rank_blocks as f64,
        )
        .metric("value_compression_ratio", value_compression_ratio)
        .metric("derivative_compression_ratio", derivative_compression_ratio)
        .metric("malformed_rejected", 1.0)
        .metric("malformed_output_preserved", f64::from(malformed_preserved))
        .metric("stale_metadata_rejected", 1.0)
        .metric("stale_output_preserved", f64::from(stale_preserved))
        .metric("typed_launch_failure", 1.0)
        .metric(
            "launch_failure_output_preserved",
            f64::from(failure_preserved),
        )
        .metric("retry_passed", 1.0)
        .metric("cuda_kernel_launches", runtime_stats.kernel_launches as f64)
        .metric(
            "cuda_allocation_high_water_bytes",
            teardown_high_water as f64,
        )
        .metric("cuda_teardown_live_bytes_before", teardown_before as f64)
        .metric("cuda_teardown_live_bytes_after", teardown_after as f64)
        .metric(
            "cuda_teardown_released_bytes",
            teardown_before.saturating_sub(teardown_after) as f64,
        )
        .maximum("value_action_relative_error", ACTION_TOLERANCE)
        .maximum("derivative_action_relative_error", DERIVATIVE_TOLERANCE)
        .maximum(
            "derivative_finite_difference_relative_error",
            DERIVATIVE_TOLERANCE,
        )
        .maximum("initial_update_relative_error", UPDATE_TOLERANCE)
        .maximum("retry_update_relative_error", UPDATE_TOLERANCE)
        .maximum("warm_update_relative_error", UPDATE_TOLERANCE)
        .minimum("value_low_rank_blocks", 1.0)
        .minimum("derivative_low_rank_blocks", 1.0)
        .reference("independent dense TBEM action and centered CPU wavenumber derivative")
        .tolerance(format!(
            "value <= {ACTION_TOLERANCE:.1e}; derivative/update <= {UPDATE_TOLERANCE:.1e}"
        ));
    result
        .environment
        .insert("capability.bem.hmatrix.c64".into(), "native".into());
    result.environment.insert(
        "capability.bem.hmatrix_wavenumber_update.c64".into(),
        "native".into(),
    );
    result.environment.insert(
        "capability.bem.hmatrix_setup.c64".into(),
        "host_shared_fallback".into(),
    );
    result.environment.insert(
        "claim.scope".into(),
        "resident Complex64 production H-matrix first-order wavenumber update".into(),
    );
    let artifact_bytes = serde_json::to_vec_pretty(&json!({
        "schema": "sonium-bem-cuda-hmatrix-wavenumber-update/v1",
        "case_id": CASE_ID,
        "claim_scope": "resident Complex64 production H-matrix first-order wavenumber update; host tree/compression setup remains shared",
        "base_wavenumber": physics.wave_number,
        "delta_wavenumbers": [-WAVENUMBER_STEP, WAVENUMBER_STEP],
        "matrix": {
            "dofs": ELEMENT_COUNT,
            "value_low_rank_blocks": value_low_rank_blocks,
            "derivative_low_rank_blocks": derivative_low_rank_blocks,
            "value_compression_ratio": value_compression_ratio,
            "derivative_compression_ratio": derivative_compression_ratio,
            "gauss_order": GAUSS_ORDER,
        },
        "accuracy": {
            "value_action_relative_error": value_action_error,
            "derivative_action_relative_error": derivative_action_error,
            "derivative_finite_difference_relative_error": derivative_fd_error,
            "initial_update_relative_error": initial_error,
            "retry_update_relative_error": retry_error,
            "warm_update_relative_error": warm_error,
        },
        "metadata_contract": {
            "stale_base_rejected": stale_preserved,
            "malformed_dimension_rejected": malformed_preserved,
        },
        "failure_contract": {
            "typed_launch_failure": true,
            "output_preserved": failure_preserved,
            "retry_passed": true,
        },
        "residency": {
            "setup_allocations": setup_stats.allocation_operations,
            "setup_h2d_operations": setup_stats.transfer.h2d_operations,
            "warm_replays": 2,
            "warm_allocations": warm_stats.allocation_operations,
            "warm_h2d_operations": warm_stats.transfer.h2d_operations,
            "warm_d2d_operations": warm_stats.transfer.d2d_operations,
            "warm_d2h_publications": warm_stats.transfer.d2h_operations,
            "warm_native_kernel_launches": warm_stats.kernel_launches,
        },
        "metrics": result.metrics,
        "environment": result.environment,
    }))
    .map_err(|error| format!("serialize H-matrix update artifact: {error}"))?;
    let (artifact, checksum) = write_qa_artifact(ARTIFACT, &artifact_bytes)?;
    result = result
        .artifact_with_provenance(
            artifact,
            "crates/sonium-compute/src/cuda/hmatrix.rs",
            "native CUDA resident production H-matrix wavenumber update evidence",
            "v1",
            checksum,
        )
        .runtime_evidence(cuda_runtime_evidence(
            runtime_stats,
            QaExecutionCounts {
                executed: 1,
                passed: 1,
                ..QaExecutionCounts::default()
            },
        ));
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
            .expect("H-matrix wavenumber update result serializes")
    );
    if result.status != sonium_qa::QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
