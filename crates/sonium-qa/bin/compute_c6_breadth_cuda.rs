//! Measured native CUDA breadth evidence for the production BEM quadrature
//! adapter.
//!
//! This is deliberately a workload evidence producer, not a roofline
//! simulator.  It exercises the device-resident `CudaBemQuadratureOperatorC64`
//! over odd small/medium/large pair tiers, checks each result against an
//! independent CPU/f64 accumulation, and records repeated wall-clock samples
//! plus the checked CUDA runtime counters.  There is no vendor implementation
//! of this chart-driven quadrature path in this repository, so the artifact
//! records that comparison as unavailable rather than inventing one.

use num_complex::Complex64;
use serde_json::{Value, json};
use sonium_compute::{
    Backend, BackendError, BemQuadraturePairC64, BemQuadratureSampleC64, CudaBackend,
    CudaBemQuadratureOperatorC64, CudaFailureInjectionPoint, MemoryPool, MemoryPoolConfig,
};
use sonium_qa::{cuda_hardware_provenance, required_cuda_backend, write_qa_artifact};
use std::hint::black_box;
use std::time::Instant;

const CASE_ID: &str = "compute.cuda-bem-quadrature-c6-breadth.v1";
const SCHEMA: &str = "sonium-compute-cuda-bem-quadrature-c6-breadth/v1";
const WAVE_NUMBER: f64 = 2.1;
const HARMONIC_FACTOR: f64 = -1.0;
const SAMPLES_PER_PAIR: usize = 5;
const WARMUP_ACTIONS: usize = 3;
const TIMING_SAMPLES: usize = 9;
const ACTIONS_PER_SAMPLE: usize = 4;
const PARITY_TOLERANCE: f64 = 8.0e-12;

const TIERS: [(&str, usize); 3] = [("small_odd", 7), ("medium_odd", 31), ("large_odd", 127)];

struct Batch {
    pairs: Vec<BemQuadraturePairC64>,
    samples: Vec<BemQuadratureSampleC64>,
}

struct Timing {
    samples_ms: Vec<f64>,
    median_ms: f64,
    mad_ms: f64,
    minimum_ms: f64,
    maximum_ms: f64,
    checksum: f64,
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn timing_json(timing: &Timing, work_units_per_action: usize) -> Value {
    let seconds = timing.median_ms / 1_000.0;
    json!({
        "samples_ms": timing.samples_ms,
        "sample_count": timing.samples_ms.len(),
        "median_ms": timing.median_ms,
        "mad_ms": timing.mad_ms,
        "minimum_ms": timing.minimum_ms,
        "maximum_ms": timing.maximum_ms,
        "actions_per_sample": ACTIONS_PER_SAMPLE,
        "work_units_per_action": work_units_per_action,
        "work_units_per_second_at_median": work_units_per_action as f64 * ACTIONS_PER_SAMPLE as f64 / seconds,
        "checksum": timing.checksum,
    })
}

fn make_batch(pair_count: usize) -> Batch {
    let mut pairs = Vec::with_capacity(pair_count);
    let mut samples = Vec::with_capacity(pair_count * SAMPLES_PER_PAIR);
    for pair_index in 0..pair_count {
        let index = pair_index as f64;
        let source = [0.013 * index, -0.009 * index, 0.0];
        let sample_offset = samples.len() as u64;
        for sample_index in 0..SAMPLES_PER_PAIR {
            let sample = sample_index as f64;
            let position = [
                source[0] + 0.23 + 0.017 * sample,
                source[1] - 0.19 + 0.011 * sample,
                0.47 + 0.003 * index + 0.021 * sample,
            ];
            let weight = 0.19 + 0.013 * sample;
            let basis_weight = 0.71 - 0.031 * sample + 0.0007 * index;
            let normal = [0.0, 0.0, 1.0];
            samples.push(match sample_index % 4 {
                0 => BemQuadratureSampleC64::full(position, normal, weight, basis_weight),
                1 => BemQuadratureSampleC64::singular_bulk(
                    position,
                    normal,
                    weight * 0.83,
                    basis_weight,
                ),
                2 => BemQuadratureSampleC64::hypersingular_edge(
                    position,
                    [1.0, 0.0, 0.0],
                    weight * 0.67,
                    basis_weight,
                ),
                _ => BemQuadratureSampleC64::gh_only(position, normal, weight * 0.91, basis_weight),
            });
        }
        pairs.push(BemQuadraturePairC64::new(
            source,
            [0.0, 0.0, 1.0],
            sample_offset,
            SAMPLES_PER_PAIR as u64,
        ));
    }
    Batch { pairs, samples }
}

fn reference_into(batch: &Batch, output: &mut [sonium_compute::BemRegularPairOutF64]) {
    output.fill(sonium_compute::BemRegularPairOutF64::default());
    let phase_factor = WAVE_NUMBER * HARMONIC_FACTOR;
    for (pair, result) in batch.pairs.iter().zip(output.iter_mut()) {
        let start = pair.sample_offset as usize;
        let end = start + pair.sample_count as usize;
        for sample in &batch.samples[start..end] {
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
            let unit = difference.map(|value| value / radius);
            let scaled_weight = sample.weight * sample.basis_weight;
            let green = Complex64::from_polar(
                scaled_weight / (4.0 * std::f64::consts::PI * radius),
                phase_factor * radius,
            );
            if sample.kind == BemQuadratureSampleC64::KIND_HYPERSINGULAR_EDGE {
                let factor = Complex64::from_polar(
                    1.0 / (4.0 * std::f64::consts::PI * radius),
                    phase_factor * radius,
                ) * Complex64::new(-1.0 / radius, phase_factor);
                let cross = [
                    factor
                        * (unit[1] * sample.edge_direction[2] - unit[2] * sample.edge_direction[1]),
                    factor
                        * (unit[2] * sample.edge_direction[0] - unit[0] * sample.edge_direction[2]),
                    factor
                        * (unit[0] * sample.edge_direction[1] - unit[1] * sample.edge_direction[0]),
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
            let factor = Complex64::new(-1.0 / radius, phase_factor);
            result.g += green;
            result.h += green * factor * r_dot_ny;
            result.ht += green * factor * r_dot_nx;
            match sample.kind {
                BemQuadratureSampleC64::KIND_FULL => {
                    let radius_squared = radius * radius;
                    let product = r_dot_ny * r_dot_nx;
                    result.e += green
                        * Complex64::new(
                            (3.0 / radius_squared - WAVE_NUMBER * WAVE_NUMBER) * product
                                + normal_dot / radius_squared,
                            -phase_factor / radius * (3.0 * product + normal_dot),
                        );
                }
                BemQuadratureSampleC64::KIND_SINGULAR_BULK => {
                    result.e += green * (WAVE_NUMBER * WAVE_NUMBER * normal_dot);
                }
                BemQuadratureSampleC64::KIND_GH_ONLY => {}
                _ => unreachable!("batch generator only emits known sample kinds"),
            }
        }
    }
}

fn pair_error(
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
    .map(|(got, want)| {
        if [got.re, got.im, want.re, want.im]
            .into_iter()
            .any(|value| !value.is_finite())
        {
            f64::INFINITY
        } else {
            (got - want).norm() / want.norm().max(1.0e-30)
        }
    })
    .fold(0.0, f64::max)
}

fn vector_error(
    actual: &[sonium_compute::BemRegularPairOutF64],
    expected: &[sonium_compute::BemRegularPairOutF64],
) -> f64 {
    if actual.len() != expected.len() || actual.is_empty() {
        return f64::INFINITY;
    }
    actual
        .iter()
        .zip(expected)
        .map(|(got, want)| pair_error(got, want))
        .fold(0.0, f64::max)
}

fn checksum(values: &[sonium_compute::BemRegularPairOutF64]) -> f64 {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let weight = (index + 1) as f64;
            weight * (value.g.re + value.h.im - value.ht.re + value.e.im)
        })
        .sum()
}

fn cpu_timing(batch: &Batch, expected: &[sonium_compute::BemRegularPairOutF64]) -> Timing {
    let mut output = vec![sonium_compute::BemRegularPairOutF64::default(); batch.pairs.len()];
    let mut samples_ms = Vec::with_capacity(TIMING_SAMPLES);
    let mut sink = 0.0;
    for _ in 0..TIMING_SAMPLES {
        let started = Instant::now();
        for _ in 0..ACTIONS_PER_SAMPLE {
            reference_into(batch, &mut output);
        }
        samples_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        sink += checksum(&output);
    }
    let median_ms = median(&samples_ms);
    let deviations: Vec<_> = samples_ms
        .iter()
        .map(|sample| (sample - median_ms).abs())
        .collect();
    let timing = Timing {
        minimum_ms: samples_ms.iter().copied().fold(f64::INFINITY, f64::min),
        maximum_ms: samples_ms.iter().copied().fold(0.0, f64::max),
        mad_ms: median(&deviations),
        median_ms,
        samples_ms,
        checksum: checksum(&output),
    };
    assert!(vector_error(&output, expected) <= PARITY_TOLERANCE);
    black_box(sink);
    timing
}

fn stats_json(stats: sonium_compute::CudaRuntimeStatsSnapshot) -> Value {
    json!({
        "kernel_launches": stats.kernel_launches,
        "library_operations": stats.library_operations,
        "synchronization_calls": stats.synchronization_calls,
        "allocation_operations": stats.allocation_operations,
        "staging_allocation_operations": stats.staging_allocation_operations,
        "h2d_operations": stats.transfer.h2d_operations,
        "d2h_operations": stats.transfer.d2h_operations,
        "d2d_operations": stats.transfer.d2d_operations,
        "live_allocation_bytes": stats.live_allocation_bytes,
        "allocation_high_water_bytes": stats.allocation_high_water_bytes,
        "live_staging_bytes": stats.live_staging_bytes,
        "staging_high_water_bytes": stats.staging_high_water_bytes,
    })
}

fn failure_lane(
    pool: &MemoryPool<CudaBackend>,
    operator: &CudaBemQuadratureOperatorC64,
    output: &mut sonium_compute::pool::Buffer<sonium_compute::BemRegularPairOutF64, CudaBackend>,
    expected: &[sonium_compute::BemRegularPairOutF64],
) -> Result<Value, String> {
    let failure_baseline = pool.backend().begin_runtime_stats();
    pool.backend()
        .inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = operator.try_apply_device(output);
    let failure_typed = matches!(
        &failure,
        Err(BackendError::Launch { .. }) | Err(BackendError::OperationFailed { .. })
    );
    if failure.is_ok() {
        return Err("injected launch failure unexpectedly succeeded".into());
    }
    pool.backend()
        .recover_after_error()
        .map_err(|error| format!("recover after injected launch failure: {error}"))?;
    let mut preserved = vec![sonium_compute::BemRegularPairOutF64::default(); expected.len()];
    pool.backend()
        .try_copy_d2h(output.raw(), &mut preserved)
        .map_err(|error| format!("read preserved output after launch failure: {error}"))?;
    let output_preserved = vector_error(&preserved, expected) <= PARITY_TOLERANCE;
    let retry_baseline = pool.backend().begin_runtime_stats();
    operator
        .try_apply_device(output)
        .map_err(|error| format!("retry after launch failure: {error}"))?;
    let mut retried = vec![sonium_compute::BemRegularPairOutF64::default(); expected.len()];
    pool.backend()
        .try_copy_d2h(output.raw(), &mut retried)
        .map_err(|error| format!("publish retried output: {error}"))?;
    let retry_error = vector_error(&retried, expected);
    let failure_stats = pool.backend().runtime_stats_since(failure_baseline);
    let retry_stats = pool.backend().runtime_stats_since(retry_baseline);
    Ok(json!({
        "failure_typed": failure_typed,
        "output_preserved": output_preserved,
        "retry_succeeded": retry_error <= PARITY_TOLERANCE,
        "retry_relative_error": retry_error,
        "failure_runtime": stats_json(failure_stats),
        "retry_runtime": stats_json(retry_stats),
    }))
}

fn run_tier(
    pool: &MemoryPool<CudaBackend>,
    tier_id: &str,
    pair_count: usize,
) -> Result<Value, String> {
    let batch = make_batch(pair_count);
    let mut expected = vec![sonium_compute::BemRegularPairOutF64::default(); pair_count];
    reference_into(&batch, &mut expected);
    if expected.iter().any(|value| {
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
        .any(|component| !component.is_finite())
    }) {
        return Err(format!("{tier_id} CPU oracle produced a non-finite result"));
    }
    let mut malformed = batch.pairs[0];
    malformed.sample_offset = batch.samples.len() as u64 + 1;
    let malformed_rejected = CudaBemQuadratureOperatorC64::new(
        pool.clone(),
        vec![malformed],
        batch.samples.clone(),
        WAVE_NUMBER,
        HARMONIC_FACTOR,
    )
    .is_err();
    if !malformed_rejected {
        return Err(format!("{tier_id} malformed sample range was accepted"));
    }

    let setup_baseline = pool.backend().begin_runtime_stats();
    let operator = CudaBemQuadratureOperatorC64::new(
        pool.clone(),
        batch.pairs.clone(),
        batch.samples.clone(),
        WAVE_NUMBER,
        HARMONIC_FACTOR,
    )
    .map_err(|error| format!("construct {tier_id} production quadrature operator: {error}"))?;
    let mut output = pool
        .try_alloc::<sonium_compute::BemRegularPairOutF64>(pair_count)
        .map_err(|error| format!("allocate {tier_id} CUDA output: {error}"))?;
    let setup = pool.backend().runtime_stats_since(setup_baseline);

    for _ in 0..WARMUP_ACTIONS {
        operator
            .try_apply_device(&mut output)
            .map_err(|error| format!("{tier_id} warmup CUDA action: {error}"))?;
    }
    let mut warmup_actual = vec![sonium_compute::BemRegularPairOutF64::default(); pair_count];
    pool.backend()
        .try_copy_d2h(output.raw(), &mut warmup_actual)
        .map_err(|error| format!("{tier_id} warmup correctness publication: {error}"))?;
    let warmup_error = vector_error(&warmup_actual, &expected);
    if warmup_error > PARITY_TOLERANCE {
        return Err(format!(
            "{tier_id} pre-timing CUDA parity {warmup_error:.6e} exceeds tolerance"
        ));
    }

    let cpu = cpu_timing(&batch, &expected);
    pool.backend().reset_runtime_stats();
    let timed_baseline = pool.backend().begin_runtime_stats();
    let mut cuda_samples_ms = Vec::with_capacity(TIMING_SAMPLES);
    let mut cuda_sink = 0.0;
    for _ in 0..TIMING_SAMPLES {
        let started = Instant::now();
        for _ in 0..ACTIONS_PER_SAMPLE {
            operator
                .try_apply_device(&mut output)
                .map_err(|error| format!("{tier_id} timed CUDA action: {error}"))?;
        }
        cuda_samples_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        cuda_sink += pair_count as f64;
    }
    let timed_stats = pool.backend().runtime_stats_since(timed_baseline);
    let after_timing = pool.backend().begin_runtime_stats();
    let mut actual = vec![sonium_compute::BemRegularPairOutF64::default(); pair_count];
    pool.backend()
        .try_copy_d2h(output.raw(), &mut actual)
        .map_err(|error| format!("{tier_id} timed correctness publication: {error}"))?;
    let publication_stats = pool.backend().runtime_stats_since(after_timing);
    let cuda_error = vector_error(&actual, &expected);
    if cuda_error > PARITY_TOLERANCE {
        return Err(format!(
            "{tier_id} post-timing CUDA parity {cuda_error:.6e} exceeds tolerance"
        ));
    }
    let cuda_median = median(&cuda_samples_ms);
    let cuda_deviations: Vec<_> = cuda_samples_ms
        .iter()
        .map(|sample| (sample - cuda_median).abs())
        .collect();
    let cuda = Timing {
        minimum_ms: cuda_samples_ms
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
        maximum_ms: cuda_samples_ms.iter().copied().fold(0.0, f64::max),
        mad_ms: median(&cuda_deviations),
        median_ms: cuda_median,
        samples_ms: cuda_samples_ms,
        checksum: checksum(&actual),
    };
    let failure = failure_lane(pool, &operator, &mut output, &expected)?;
    if !failure["failure_typed"].as_bool().unwrap_or(false)
        || !failure["output_preserved"].as_bool().unwrap_or(false)
        || !failure["retry_succeeded"].as_bool().unwrap_or(false)
    {
        return Err(format!(
            "{tier_id} launch failure/retry contract failed: {failure}"
        ));
    }
    let failure_stats = pool.backend().runtime_stats();
    drop(operator);
    drop(output);
    let teardown = pool.backend().runtime_stats();
    black_box(cuda_sink);
    Ok(json!({
        "id": tier_id,
        "status": "passed",
        "pair_count": pair_count,
        "samples_per_pair": SAMPLES_PER_PAIR,
        "sample_count": batch.samples.len(),
        "odd_pair_tail": pair_count % 2 == 1,
        "sample_kind_mix": ["full", "singular_bulk", "hypersingular_edge", "gh_only"],
        "correctness": {
            "reference": "independent CPU/f64 Green-function accumulation over the same descriptors",
            "pre_timing_relative_error": warmup_error,
            "post_timing_relative_error": cuda_error,
            "tolerance": PARITY_TOLERANCE,
            "correctness_before_timing": true,
        },
        "cpu": timing_json(&cpu, pair_count * SAMPLES_PER_PAIR),
        "cuda": timing_json(&cuda, pair_count * SAMPLES_PER_PAIR),
        "runtime": {
            "setup": stats_json(setup),
            "timed_resident": stats_json(timed_stats),
            "publication": stats_json(publication_stats),
            "after_failure_and_retry": stats_json(failure_stats),
            "teardown": stats_json(teardown),
        },
        "failure_recovery": failure,
        "malformed_input_rejected": malformed_rejected,
    }))
}

fn run(backend: CudaBackend) -> Result<Value, String> {
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28))
        .map_err(|error| format!("construct C6 breadth CUDA pool: {error}"))?;
    let mut rows = Vec::with_capacity(TIERS.len());
    for (tier_id, pair_count) in TIERS {
        rows.push(run_tier(&pool, tier_id, pair_count)?);
    }
    let hardware = cuda_hardware_provenance();
    Ok(json!({
        "schema": SCHEMA,
        "version": 1,
        "status": "passed",
        "case_id": CASE_ID,
        "scope": "single native NVIDIA SM120; production BEM chart-driven quadrature; odd size tiers",
        "claims": {
            "native_cuda": true,
            "cpu_comparator": "measured_independent_f64_oracle",
            "vendor_comparator": "not_available_for_chart_driven_quadrature",
            "roofline": "not_claimed_no_profiler_or_instruction_count",
            "energy": "not_claimed_no_energy_counter",
        },
        "hardware": hardware,
        "workload": {
            "production_adapter": "CudaBemQuadratureOperatorC64",
            "scalar": "Complex64",
            "channels": ["G", "H", "H_transpose", "E"],
            "wave_number": WAVE_NUMBER,
            "harmonic_factor": HARMONIC_FACTOR,
            "warmup_actions": WARMUP_ACTIONS,
            "timing_samples": TIMING_SAMPLES,
            "actions_per_sample": ACTIONS_PER_SAMPLE,
        },
        "rows": rows,
        "unsupported_boundaries": [
            "no Nsight or instruction-level roofline attribution",
            "no energy or power-counter claim",
            "no vendor quadrature comparator exists locally",
            "no full-order device-resident BEM assembly claim",
            "no multi-GPU, physical-reset, Arm64, Windows, or AVX512 claim",
        ],
    }))
}

fn main() {
    let result = required_cuda_backend()
        .map_err(|error| format!("required native CUDA backend: {error}"))
        .and_then(run)
        .and_then(|artifact| {
            let bytes = serde_json::to_vec_pretty(&artifact)
                .map_err(|error| format!("serialize C6 breadth artifact: {error}"))?;
            write_qa_artifact("cuda_c6_breadth_bem_quadrature.json", &bytes)
                .map(|(_, checksum)| (artifact, checksum))
        });
    match result {
        Ok((artifact, checksum)) => {
            println!(
                "{}",
                json!({"status":"passed", "artifact": artifact, "artifact_checksum": checksum})
            );
            println!("SUCCESS");
        }
        Err(error) => {
            eprintln!("FAILED: C6 breadth CUDA evidence: {error}");
            std::process::exit(1);
        }
    }
}
