//! Real-CUDA C2 rows for specialised BEM/FMM device operations.
//!
//! The generic operation matrix cannot describe the ABI-specific buffers used
//! by BEM and FMM. This producer keeps those rows separate, exercises checked
//! offset/stride entry points on resident allocations, and records a per-cell
//! runtime census. Host copies seed and publish the fixture only; every
//! measured action must launch CUDA work without a transfer or staging
//! allocation.

use num_complex::Complex64;
use serde_json::{Value, json};
use sonium_compute::{
    Backend, BackendError, BemFieldElementC64, BemFieldPointC64, BemQuadraturePairC64,
    BemQuadratureSampleC64, BemRegularPairInF64, BemRegularPairOutF64, BufferView, CudaBackend,
    CudaFailureInjectionPoint, CudaRuntimeStatsSnapshot, MemoryPool, MemoryPoolConfig,
};
use sonium_qa::{
    QaExecutionCounts, QaFailureStage, QaResult, cuda_hardware_provenance, cuda_runtime_evidence,
    required_cuda_backend, write_qa_artifact,
};
use std::time::Instant;

const CASE_ID: &str = "compute.cuda-specialized-operation-matrix.v1";

fn transfer_json(stats: &CudaRuntimeStatsSnapshot) -> Value {
    json!({
        "h2d_operations": stats.transfer.h2d_operations,
        "h2d_bytes": stats.transfer.h2d_bytes,
        "d2h_operations": stats.transfer.d2h_operations,
        "d2h_bytes": stats.transfer.d2h_bytes,
        "d2d_operations": stats.transfer.d2d_operations,
        "d2d_bytes": stats.transfer.d2d_bytes,
        "staging_allocation_operations": stats.staging_allocation_operations,
        "live_staging_bytes": stats.live_staging_bytes,
    })
}

fn runtime_json(stats: &CudaRuntimeStatsSnapshot) -> Value {
    json!({
        "kernel_launches": stats.kernel_launches,
        "library_operations": stats.library_operations,
        "synchronization_calls": stats.synchronization_calls,
        "allocations": stats.allocation_operations,
        "transfers": transfer_json(stats),
        "live_allocation_bytes": stats.live_allocation_bytes,
        "allocation_high_water_bytes": stats.allocation_high_water_bytes,
    })
}

fn action<F>(
    backend: &CudaBackend,
    phase: &'static str,
    scalar: &'static str,
    layout: &'static str,
    run: F,
) -> Result<Value, String>
where
    F: FnOnce() -> sonium_compute::BackendResult<()>,
{
    let before = backend.begin_runtime_stats();
    run().map_err(|error| format!("{phase}: {error}"))?;
    let stats = backend.runtime_stats_since(before);
    if stats.kernel_launches == 0 && stats.library_operations == 0 {
        return Err(format!("{phase}: no native CUDA work was submitted"));
    }
    if stats.transfer.h2d_operations != 0
        || stats.transfer.d2h_operations != 0
        || stats.transfer.d2d_operations != 0
        || stats.staging_allocation_operations != 0
        || stats.allocation_operations != 0
    {
        return Err(format!(
            "{phase}: resident action used an undocumented transfer/staging/allocation path: {}",
            transfer_json(&stats)
        ));
    }
    Ok(json!({
        "phase": phase,
        "scalar": scalar,
        "layout": layout,
        "status": "native",
        "residency": {
            "host_transfers": false,
            "device_staging": false,
            "runtime": runtime_json(&stats),
        },
    }))
}

/// Independent scalar CPU reference for the chart-driven Complex64 pair
/// kernel.  Keeping this arithmetic in the QA producer prevents a CUDA
/// implementation from validating itself through the same backend helper.
fn quadrature_pair_reference(
    pair: BemQuadraturePairC64,
    samples: &[BemQuadratureSampleC64],
    wave_number: f64,
    harmonic_factor: f64,
) -> BemRegularPairOutF64 {
    let mut result = BemRegularPairOutF64::default();
    let source = pair.source_point;
    let source_normal = pair.source_normal;
    let wavruim = wave_number * harmonic_factor;
    let start = pair.sample_offset as usize;
    let end = (pair.sample_offset + pair.sample_count) as usize;
    for sample in &samples[start..end] {
        let diff = [
            sample.position[0] - source[0],
            sample.position[1] - source[1],
            sample.position[2] - source[2],
        ];
        let r = diff.iter().map(|value| value * value).sum::<f64>().sqrt();
        if r < 1.0e-15 {
            continue;
        }
        let unit = diff.map(|value| value / r);
        let scaled_weight = sample.weight * sample.basis_weight;
        let green = Complex64::from_polar(
            scaled_weight / (4.0 * std::f64::consts::PI * r),
            wavruim * r,
        );
        if sample.kind == BemQuadratureSampleC64::KIND_HYPERSINGULAR_EDGE {
            let factor = Complex64::from_polar(1.0 / (4.0 * std::f64::consts::PI * r), wavruim * r)
                * Complex64::new(-1.0 / r, wavruim);
            let cross = [
                factor * (unit[1] * sample.edge_direction[2] - unit[2] * sample.edge_direction[1]),
                factor * (unit[2] * sample.edge_direction[0] - unit[0] * sample.edge_direction[2]),
                factor * (unit[0] * sample.edge_direction[1] - unit[1] * sample.edge_direction[0]),
            ];
            result.e += cross
                .into_iter()
                .zip(source_normal)
                .map(|(value, normal)| value * normal)
                .sum::<Complex64>()
                * scaled_weight;
            continue;
        }

        let field_normal = sample.field_normal;
        let r_dot_ny = unit
            .iter()
            .zip(field_normal)
            .map(|(value, normal)| value * normal)
            .sum::<f64>();
        let r_dot_nx = -unit
            .iter()
            .zip(source_normal)
            .map(|(value, normal)| value * normal)
            .sum::<f64>();
        let normal_dot = source_normal
            .iter()
            .zip(field_normal)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let factor = Complex64::new(-1.0 / r, wavruim);
        result.g += green;
        result.h += green * factor * r_dot_ny;
        result.ht += green * factor * r_dot_nx;
        match sample.kind {
            BemQuadratureSampleC64::KIND_FULL => {
                let rq = r_dot_ny * r_dot_nx;
                let r2 = r * r;
                result.e += green
                    * Complex64::new(
                        (3.0 / r2 - wave_number * wave_number) * rq + normal_dot / r2,
                        -wavruim / r * (3.0 * rq + normal_dot),
                    );
            }
            BemQuadratureSampleC64::KIND_SINGULAR_BULK => {
                result.e += green * (wave_number * wave_number * normal_dot);
            }
            BemQuadratureSampleC64::KIND_GH_ONLY => {}
            _ => unreachable!("validated quadrature sample kind"),
        }
    }
    result
}

fn quadrature_relative_error(
    actual: &[BemRegularPairOutF64],
    expected: &[BemRegularPairOutF64],
) -> Result<f64, String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "quadrature output length mismatch: {} != {}",
            actual.len(),
            expected.len()
        ));
    }
    let mut error = 0.0_f64;
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        for (name, actual, reference) in [
            ("g", got.g, want.g),
            ("h", got.h, want.h),
            ("ht", got.ht, want.ht),
            ("e", got.e, want.e),
        ] {
            let scale = reference.norm().max(1.0);
            let relative = (actual - reference).norm() / scale;
            if !relative.is_finite() {
                return Err(format!("quadrature pair {index} {name} is non-finite"));
            }
            error = error.max(relative);
        }
    }
    Ok(error)
}

/// Independent scalar CPU reference for the analytic wavenumber derivative of
/// the chart-driven Complex64 pair kernel.  This deliberately mirrors the
/// kernel arithmetic without calling the CUDA value or derivative entry point;
/// the centered check below therefore tests the analytic derivative against a
/// separately evaluated CPU value reference as well.
fn quadrature_pair_wavenumber_derivative_reference(
    pair: BemQuadraturePairC64,
    samples: &[BemQuadratureSampleC64],
    wave_number: f64,
    harmonic_factor: f64,
) -> BemRegularPairOutF64 {
    let mut result = BemRegularPairOutF64::default();
    let source = pair.source_point;
    let source_normal = pair.source_normal;
    let wavruim = wave_number * harmonic_factor;
    let start = pair.sample_offset as usize;
    let end = (pair.sample_offset + pair.sample_count) as usize;
    for sample in &samples[start..end] {
        let diff = [
            sample.position[0] - source[0],
            sample.position[1] - source[1],
            sample.position[2] - source[2],
        ];
        let r = diff.iter().map(|value| value * value).sum::<f64>().sqrt();
        if r < 1.0e-15 {
            continue;
        }
        let unit = diff.map(|value| value / r);
        let scaled_weight = sample.weight * sample.basis_weight;
        let green = Complex64::from_polar(
            scaled_weight / (4.0 * std::f64::consts::PI * r),
            wavruim * r,
        );
        let dgreen = green * Complex64::new(0.0, harmonic_factor * r);
        let z1 = Complex64::new(-1.0 / r, wavruim);
        let dz1 = Complex64::new(0.0, harmonic_factor);
        let dz_kernel = dgreen * z1 + green * dz1;

        if sample.kind == BemQuadratureSampleC64::KIND_HYPERSINGULAR_EDGE {
            let edge = sample.edge_direction;
            let edge_scale = [
                unit[1] * edge[2] - unit[2] * edge[1],
                unit[2] * edge[0] - unit[0] * edge[2],
                unit[0] * edge[1] - unit[1] * edge[0],
            ]
            .into_iter()
            .zip(source_normal)
            .map(|(value, normal)| value * normal)
            .sum::<f64>();
            result.e += dz_kernel * edge_scale;
            continue;
        }

        let field_normal = sample.field_normal;
        let r_dot_ny = unit
            .iter()
            .zip(field_normal)
            .map(|(value, normal)| value * normal)
            .sum::<f64>();
        let r_dot_nx = -unit
            .iter()
            .zip(source_normal)
            .map(|(value, normal)| value * normal)
            .sum::<f64>();
        let normal_dot = source_normal
            .iter()
            .zip(field_normal)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        result.g += dgreen;
        result.h += dz_kernel * r_dot_ny;
        result.ht += dz_kernel * r_dot_nx;
        match sample.kind {
            BemQuadratureSampleC64::KIND_FULL => {
                let rq = r_dot_ny * r_dot_nx;
                let r2 = r * r;
                let e_factor = Complex64::new(
                    (3.0 / r2 - wave_number * wave_number) * rq + normal_dot / r2,
                    -wavruim / r * (3.0 * rq + normal_dot),
                );
                let de_factor = Complex64::new(
                    -2.0 * wave_number * rq,
                    -harmonic_factor / r * (3.0 * rq + normal_dot),
                );
                result.e += dgreen * e_factor + green * de_factor;
            }
            BemQuadratureSampleC64::KIND_SINGULAR_BULK => {
                result.e += (dgreen * (wave_number * wave_number) + green * (2.0 * wave_number))
                    * normal_dot;
            }
            BemQuadratureSampleC64::KIND_GH_ONLY => {}
            _ => unreachable!("validated quadrature sample kind"),
        }
    }
    result
}

const TRI4_XI: [f64; 4] = [0.333333333333333, 0.6, 0.2, 0.2];
const TRI4_ETA: [f64; 4] = [0.333333333333333, 0.2, 0.6, 0.2];
const TRI4_WEIGHTS: [f64; 4] = [
    -0.281250,
    0.260416666666667,
    0.260416666666667,
    0.260416666666667,
];

/// Independent scalar CPU value reference for the regular constant-panel
/// pair.  The derivative packet below uses this value path for a centered
/// finite-difference check, while its analytic reference is implemented
/// separately so neither check calls the CUDA backend.
fn regular_pair_reference(
    pair: BemRegularPairInF64,
    wave_number: f64,
    harmonic_factor: f64,
) -> BemRegularPairOutF64 {
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
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if jacobian <= 1.0e-15 {
        return BemRegularPairOutF64::default();
    }
    let field_normal = raw_normal.map(|value| value / jacobian);
    let wavruim = harmonic_factor * wave_number;
    let mut result = BemRegularPairOutF64::default();
    for ((&xi, &eta), &weight) in TRI4_XI.iter().zip(&TRI4_ETA).zip(&TRI4_WEIGHTS) {
        let n0 = 1.0 - xi - eta;
        let point = [
            n0 * pair.v0[0] + xi * pair.v1[0] + eta * pair.v2[0],
            n0 * pair.v0[1] + xi * pair.v1[1] + eta * pair.v2[1],
            n0 * pair.v0[2] + xi * pair.v1[2] + eta * pair.v2[2],
        ];
        let difference = [
            point[0] - pair.source_point[0],
            point[1] - pair.source_point[1],
            point[2] - pair.source_point[2],
        ];
        let radius = difference
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if radius < 1.0e-15 {
            continue;
        }
        let inverse_radius = 1.0 / radius;
        let unit = difference.map(|value| value * inverse_radius);
        let green = Complex64::from_polar(
            weight * jacobian / (4.0 * std::f64::consts::PI * radius),
            wavruim * radius,
        );
        let green_derivative = green * Complex64::new(-inverse_radius, wavruim);
        let source_normal = pair.source_normal;
        let r1h = unit
            .iter()
            .zip(field_normal)
            .map(|(distance, normal)| distance * normal)
            .sum::<f64>();
        let r2h = -unit
            .iter()
            .zip(source_normal)
            .map(|(distance, normal)| distance * normal)
            .sum::<f64>();
        let rq = r1h * r2h;
        let normal_dot = source_normal
            .iter()
            .zip(field_normal)
            .map(|(source, field)| source * field)
            .sum::<f64>();
        let radius_squared = radius * radius;
        let e_factor = Complex64::new(
            (3.0 / radius_squared - wave_number * wave_number) * rq + normal_dot / radius_squared,
            -wavruim / radius * (3.0 * rq + normal_dot),
        );
        result.g += green;
        result.h += green_derivative * r1h;
        result.ht += green_derivative * r2h;
        result.e += green * e_factor;
    }
    result
}

/// Independent analytic d/dk reference for all four regular-pair channels.
/// This mirrors the mathematical derivative, not the CUDA launch or ABI.
fn regular_pair_wavenumber_derivative_reference(
    pair: BemRegularPairInF64,
    wave_number: f64,
    harmonic_factor: f64,
) -> BemRegularPairOutF64 {
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
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if jacobian <= 1.0e-15 {
        return BemRegularPairOutF64::default();
    }
    let field_normal = raw_normal.map(|value| value / jacobian);
    let wavruim = harmonic_factor * wave_number;
    let mut result = BemRegularPairOutF64::default();
    for ((&xi, &eta), &weight) in TRI4_XI.iter().zip(&TRI4_ETA).zip(&TRI4_WEIGHTS) {
        let n0 = 1.0 - xi - eta;
        let point = [
            n0 * pair.v0[0] + xi * pair.v1[0] + eta * pair.v2[0],
            n0 * pair.v0[1] + xi * pair.v1[1] + eta * pair.v2[1],
            n0 * pair.v0[2] + xi * pair.v1[2] + eta * pair.v2[2],
        ];
        let difference = [
            point[0] - pair.source_point[0],
            point[1] - pair.source_point[1],
            point[2] - pair.source_point[2],
        ];
        let radius = difference
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if radius < 1.0e-15 {
            continue;
        }
        let inverse_radius = 1.0 / radius;
        let unit = difference.map(|value| value * inverse_radius);
        let green = Complex64::from_polar(
            weight * jacobian / (4.0 * std::f64::consts::PI * radius),
            wavruim * radius,
        );
        let green_k = green * Complex64::new(0.0, harmonic_factor * radius);
        let z1 = Complex64::new(-inverse_radius, wavruim);
        let z1_k = Complex64::new(0.0, harmonic_factor);
        let h_base_k = green_k * z1 + green * z1_k;
        let source_normal = pair.source_normal;
        let r1h = unit
            .iter()
            .zip(field_normal)
            .map(|(distance, normal)| distance * normal)
            .sum::<f64>();
        let r2h = -unit
            .iter()
            .zip(source_normal)
            .map(|(distance, normal)| distance * normal)
            .sum::<f64>();
        let rq = r1h * r2h;
        let normal_dot = source_normal
            .iter()
            .zip(field_normal)
            .map(|(source, field)| source * field)
            .sum::<f64>();
        let radius_squared = radius * radius;
        let e_factor = Complex64::new(
            (3.0 / radius_squared - wave_number * wave_number) * rq + normal_dot / radius_squared,
            -wavruim / radius * (3.0 * rq + normal_dot),
        );
        let e_factor_k = Complex64::new(
            -2.0 * wave_number * rq,
            -harmonic_factor / radius * (3.0 * rq + normal_dot),
        );
        result.g += green_k;
        result.h += h_base_k * r1h;
        result.ht += h_base_k * r2h;
        result.e += green_k * e_factor + green * e_factor_k;
    }
    result
}

fn regular_pair_relative_error(
    actual: &[BemRegularPairOutF64],
    expected: &[BemRegularPairOutF64],
) -> Result<f64, String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "regular-pair output length mismatch: {} != {}",
            actual.len(),
            expected.len()
        ));
    }
    let mut error = 0.0_f64;
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        for (name, actual, reference) in [
            ("g", got.g, want.g),
            ("h", got.h, want.h),
            ("ht", got.ht, want.ht),
            ("e", got.e, want.e),
        ] {
            let relative = (actual - reference).norm() / reference.norm().max(1.0);
            if !relative.is_finite() {
                return Err(format!("regular pair {index} {name} is non-finite"));
            }
            error = error.max(relative);
        }
    }
    Ok(error)
}

/// Independent centered CPU reference for the source-collocation x
/// derivative.  The scalar regular-pair value path is evaluated at displaced
/// source points; this deliberately does not call the CUDA derivative entry
/// point or reuse its analytic geometry terms.
fn regular_pair_source_x_centered_reference(
    pair: BemRegularPairInF64,
    wave_number: f64,
    harmonic_factor: f64,
    step: f64,
) -> BemRegularPairOutF64 {
    let mut plus_pair = pair;
    plus_pair.source_point[0] += step;
    let mut minus_pair = pair;
    minus_pair.source_point[0] -= step;
    let plus = regular_pair_reference(plus_pair, wave_number, harmonic_factor);
    let minus = regular_pair_reference(minus_pair, wave_number, harmonic_factor);
    BemRegularPairOutF64 {
        g: (plus.g - minus.g) / (2.0 * step),
        h: (plus.h - minus.h) / (2.0 * step),
        ht: (plus.ht - minus.ht) / (2.0 * step),
        e: (plus.e - minus.e) / (2.0 * step),
    }
}

/// Independent centered CPU reference for the source-collocation y
/// derivative. The value reference is evaluated at displaced source points;
/// it deliberately does not call any CUDA derivative entry point.
fn regular_pair_source_y_centered_reference(
    pair: BemRegularPairInF64,
    wave_number: f64,
    harmonic_factor: f64,
    step: f64,
) -> BemRegularPairOutF64 {
    let mut plus_pair = pair;
    plus_pair.source_point[1] += step;
    let mut minus_pair = pair;
    minus_pair.source_point[1] -= step;
    let plus = regular_pair_reference(plus_pair, wave_number, harmonic_factor);
    let minus = regular_pair_reference(minus_pair, wave_number, harmonic_factor);
    BemRegularPairOutF64 {
        g: (plus.g - minus.g) / (2.0 * step),
        h: (plus.h - minus.h) / (2.0 * step),
        ht: (plus.ht - minus.ht) / (2.0 * step),
        e: (plus.e - minus.e) / (2.0 * step),
    }
}

/// Independent scalar CPU reference for the constant-triangle field view.
/// This intentionally duplicates the arithmetic contract rather than calling
/// the CUDA backend, so a native kernel cannot validate itself.
fn field_reference(
    point: BemFieldPointC64,
    element: BemFieldElementC64,
    pressure: Complex64,
    velocity: Complex64,
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
        let y = [
            n0 * element.v0[0] + xi * element.v1[0] + eta * element.v2[0],
            n0 * element.v0[1] + xi * element.v1[1] + eta * element.v2[1],
            n0 * element.v0[2] + xi * element.v1[2] + eta * element.v2[2],
        ];
        let delta = [
            y[0] - point.position[0],
            y[1] - point.position[1],
            y[2] - point.position[2],
        ];
        let radius = delta.iter().map(|value| value * value).sum::<f64>().sqrt();
        if radius < 1.0e-15 {
            continue;
        }
        let inv_radius = 1.0 / radius;
        let drdn_y = delta
            .iter()
            .zip(normal)
            .map(|(distance, component)| distance * component)
            .sum::<f64>()
            * inv_radius;
        let green = Complex64::from_polar(
            jacobian * weight / (4.0 * std::f64::consts::PI * radius),
            wavruim * radius,
        );
        let derivative = green * Complex64::new(-inv_radius, wavruim);
        result += pressure * derivative * drdn_y;
        if velocity.norm_sqr() > 1.0e-30 {
            result -= velocity * Complex64::new(0.0, pressure_factor) * green;
        }
    }
    result
}

fn field_relative_error(actual: &[Complex64], expected: &[Complex64]) -> Result<f64, String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "field output length mismatch: {} != {}",
            actual.len(),
            expected.len()
        ));
    }
    let mut error = 0.0_f64;
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        let scale = want.norm().max(1.0);
        let relative = (*got - *want).norm() / scale;
        if !relative.is_finite() {
            return Err(format!("field output {index} is non-finite"));
        }
        error = error.max(relative);
    }
    Ok(error)
}

fn run(backend: CudaBackend) -> Result<QaResult, String> {
    if !backend.has_fmm_c64() {
        return Err("CUDA FMM helper module is unavailable".into());
    }
    if !backend.has_bem_regular_pair_c64() {
        return Err("CUDA BEM regular-pair module is unavailable".into());
    }

    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20))
        .map_err(|error| format!("specialized CUDA pool: {error}"))?;
    let backend = pool.backend();
    let zero = Complex64::new(0.0, 0.0);
    let phase_host = (0..8)
        .map(|index| Complex64::new(index as f64 + 1.0, 0.25 * index as f64))
        .collect::<Vec<_>>();
    let src_host = (0..8)
        .map(|index| Complex64::new(0.5 + index as f64, -0.125 * index as f64))
        .collect::<Vec<_>>();

    let mut phase = pool
        .try_alloc::<Complex64>(8)
        .map_err(|error| format!("phase allocation: {error}"))?;
    let mut src = pool
        .try_alloc::<Complex64>(8)
        .map_err(|error| format!("source allocation: {error}"))?;
    let mut phase_dst = pool
        .try_alloc::<Complex64>(8)
        .map_err(|error| format!("phase output allocation: {error}"))?;
    let mut gather_dst = pool
        .try_alloc::<Complex64>(8)
        .map_err(|error| format!("gather output allocation: {error}"))?;
    let mut scatter_dst = pool
        .try_alloc::<Complex64>(8)
        .map_err(|error| format!("scatter output allocation: {error}"))?;
    let mut dense_dst = pool
        .try_alloc::<Complex64>(8)
        .map_err(|error| format!("indexed dense output allocation: {error}"))?;
    let mut scatter_dense_dst = pool
        .try_alloc::<Complex64>(8)
        .map_err(|error| format!("dense scatter output allocation: {error}"))?;
    backend
        .try_copy_h2d(&phase_host, phase.raw_mut())
        .map_err(|error| format!("phase upload: {error}"))?;
    backend
        .try_copy_h2d(&src_host, src.raw_mut())
        .map_err(|error| format!("source upload: {error}"))?;
    for (name, buffer) in [
        ("phase output", &mut phase_dst),
        ("gather output", &mut gather_dst),
        ("scatter output", &mut scatter_dst),
        ("indexed dense output", &mut dense_dst),
        ("dense scatter output", &mut scatter_dense_dst),
    ] {
        backend
            .try_copy_h2d(&[zero; 8], buffer.raw_mut())
            .map_err(|error| format!("{name} upload: {error}"))?;
    }

    let values = BufferView::new(8, 1, 3, 2).map_err(|error| format!("value view: {error}"))?;
    let indices_view =
        BufferView::new(8, 1, 3, 2).map_err(|error| format!("index view: {error}"))?;
    let dense_indices_view =
        BufferView::new(8, 1, 2, 2).map_err(|error| format!("dense index view: {error}"))?;
    let mut gather_indices = pool
        .try_alloc::<i32>(8)
        .map_err(|error| format!("gather-index allocation: {error}"))?;
    let mut dense_src_indices = pool
        .try_alloc::<i32>(8)
        .map_err(|error| format!("source-index allocation: {error}"))?;
    let mut dense_dst_indices = pool
        .try_alloc::<i32>(8)
        .map_err(|error| format!("destination-index allocation: {error}"))?;
    let mut index_host = vec![0_i32; 8];
    index_host[1] = 2;
    index_host[3] = 0;
    index_host[5] = 1;
    backend
        .try_copy_h2d(&index_host, gather_indices.raw_mut())
        .map_err(|error| format!("gather-index upload: {error}"))?;
    let mut source_index_host = vec![0_i32; 8];
    source_index_host[1] = 2;
    source_index_host[3] = 0;
    backend
        .try_copy_h2d(&source_index_host, dense_src_indices.raw_mut())
        .map_err(|error| format!("source-index upload: {error}"))?;
    let mut destination_index_host = vec![0_i32; 8];
    destination_index_host[1] = 0;
    destination_index_host[3] = 2;
    backend
        .try_copy_h2d(&destination_index_host, dense_dst_indices.raw_mut())
        .map_err(|error| format!("destination-index upload: {error}"))?;

    let mut matrix_host = vec![zero; 12];
    matrix_host[2] = Complex64::new(1.0, 0.5);
    matrix_host[3] = Complex64::new(2.0, -0.5);
    matrix_host[5] = Complex64::new(-1.0, 0.25);
    matrix_host[6] = Complex64::new(0.75, 1.0);
    let mut matrix = pool
        .try_alloc::<Complex64>(matrix_host.len())
        .map_err(|error| format!("matrix allocation: {error}"))?;
    backend
        .try_copy_h2d(&matrix_host, matrix.raw_mut())
        .map_err(|error| format!("matrix upload: {error}"))?;

    let mut rows = Vec::new();
    rows.push(action(
        backend,
        "fmm.phase_accumulate.c64",
        "Complex64",
        "strided(offset=1,stride=2)",
        || {
            backend.try_fmm_phase_mul_accumulate_c64_view(
                phase_dst.raw_mut(),
                values,
                phase.raw(),
                values,
                src.raw(),
                values,
            )
        },
    )?);
    rows.push(action(
        backend,
        "fmm.compact_gather.c64",
        "Complex64",
        "strided(offset=1,stride=2)",
        || {
            backend.try_fmm_compact_indexed_gather_c64_view(
                src.raw(),
                values,
                gather_indices.raw(),
                indices_view,
                gather_dst.raw_mut(),
                values,
            )
        },
    )?);
    rows.push(action(
        backend,
        "fmm.compact_scatter.c64",
        "Complex64",
        "strided(offset=1,stride=2)",
        || {
            backend.try_fmm_compact_indexed_scatter_accumulate_c64_view(
                gather_dst.raw(),
                values,
                gather_indices.raw(),
                indices_view,
                scatter_dst.raw_mut(),
                values,
            )
        },
    )?);
    rows.push(action(
        backend,
        "fmm.indexed_dense_matvec.c64",
        "Complex64",
        "matrix_offset=2,values_strided(offset=1,stride=2)",
        || {
            backend.try_fmm_indexed_dense_matvec_accumulate_c64_view(
                matrix.raw(),
                2,
                2,
                2,
                3,
                src.raw(),
                values,
                dense_src_indices.raw(),
                dense_indices_view,
                dense_dst.raw_mut(),
                values,
                dense_dst_indices.raw(),
                dense_indices_view,
            )
        },
    )?);
    rows.push(action(
        backend,
        "fmm.dense_scatter_matvec.c64",
        "Complex64",
        "matrix_offset=2,values_strided(offset=1,stride=2)",
        || {
            backend.try_fmm_dense_matvec_scatter_accumulate_c64_view(
                matrix.raw(),
                2,
                2,
                2,
                3,
                src.raw(),
                values,
                scatter_dense_dst.raw_mut(),
                values,
                dense_dst_indices.raw(),
                dense_indices_view,
            )
        },
    )?);

    let pair_view = BufferView::new(9, 1, 4, 2).map_err(|error| format!("pair view: {error}"))?;
    let output_view =
        BufferView::new(9, 1, 4, 2).map_err(|error| format!("pair output view: {error}"))?;
    let pair_values = [
        BemRegularPairInF64::from_triangle(
            [5.0, 5.0, 5.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ),
        BemRegularPairInF64::from_triangle(
            [0.0, 0.0, 4.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.5],
            [2.0, 1.0, 0.5],
        ),
        BemRegularPairInF64::from_triangle(
            [-3.0, 4.0, 2.0],
            [0.5_f64.sqrt(), 0.5_f64.sqrt(), 0.0],
            [-1.5, -1.0, 1.0],
            [-0.5, -1.0, 1.2],
            [-1.0, 0.0, 1.5],
        ),
        BemRegularPairInF64::from_triangle(
            [5.0, 5.0, 5.0],
            [0.0, 0.0, 1.0],
            [0.5, 2.0, -1.0],
            [1.5, 2.5, -1.0],
            [0.7, 3.0, -0.6],
        ),
    ];
    let mut pair_backing = vec![BemRegularPairInF64::default(); 9];
    for (index, pair) in pair_values.iter().enumerate() {
        pair_backing[pair_view
            .checked_element_index(index)
            .map_err(|error| format!("pair view index: {error}"))?] = *pair;
    }
    let mut pairs = pool
        .try_alloc::<BemRegularPairInF64>(9)
        .map_err(|error| format!("pair allocation: {error}"))?;
    let mut pair_output = pool
        .try_alloc::<BemRegularPairOutF64>(9)
        .map_err(|error| format!("pair output allocation: {error}"))?;
    backend
        .try_copy_h2d(&pair_backing, pairs.raw_mut())
        .map_err(|error| format!("pair upload: {error}"))?;
    rows.push(action(
        backend,
        "bem.regular_pair.c64",
        "Complex64",
        "records_strided(offset=1,stride=2)",
        || {
            backend.try_bem_regular_pair_c64_device_view(
                pairs.raw(),
                pair_view,
                pair_output.raw_mut(),
                output_view,
                1.0,
                -1.0,
            )
        },
    )?);

    // Chart-driven quadrature uses a distinct ABI from regular triangle
    // pairs: pair descriptors index a resident sample table.  Keep both the
    // descriptor input and publication output strided so this cell proves
    // the checked view contract rather than merely re-running the contiguous
    // operator.  Three descriptors cover full, singular-bulk, edge, and
    // G/H-only sample kinds through an independently evaluated CPU reference.
    let quadrature_wave_number = 2.1;
    let quadrature_harmonic_factor = -1.0;
    let quadrature_samples_host = vec![
        BemQuadratureSampleC64::full([1.0, 0.2, 0.7], [0.0, 0.0, 1.0], 0.7, 1.2),
        BemQuadratureSampleC64::singular_bulk([0.3, -0.4, 0.9], [0.0, 0.0, 1.0], 0.4, -0.8),
        BemQuadratureSampleC64::hypersingular_edge([0.4, 0.8, 0.6], [1.0, 0.0, 0.0], 0.25, 1.0),
        BemQuadratureSampleC64::gh_only([-0.2, 0.6, 1.1], [0.0, 0.0, 1.0], 0.35, 0.5),
        BemQuadratureSampleC64::full([1.5, -0.1, 0.8], [0.0, 1.0, 0.0], 0.45, 0.9),
        BemQuadratureSampleC64::singular_bulk([0.8, 0.7, 0.2], [0.0, 1.0, 0.0], 0.3, 0.75),
        BemQuadratureSampleC64::hypersingular_edge([1.1, 0.4, 0.3], [0.0, 0.0, 1.0], 0.2, -0.6),
        BemQuadratureSampleC64::gh_only([0.6, -0.8, 0.9], [0.0, 1.0, 0.0], 0.5, 0.4),
        BemQuadratureSampleC64::full([-0.4, 0.9, 0.3], [1.0, 0.0, 0.0], 0.6, 0.8),
        BemQuadratureSampleC64::singular_bulk([-0.7, 0.2, 1.4], [1.0, 0.0, 0.0], 0.35, 0.7),
        BemQuadratureSampleC64::hypersingular_edge([-0.2, 1.1, 0.5], [0.0, 1.0, 0.0], 0.3, 0.65),
        BemQuadratureSampleC64::gh_only([0.9, 0.3, -0.5], [1.0, 0.0, 0.0], 0.4, -0.55),
    ];
    let quadrature_pairs_host = [
        BemQuadraturePairC64::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0, 4),
        BemQuadraturePairC64::new([0.2, -0.3, 0.4], [0.0, 1.0, 0.0], 4, 4),
        BemQuadraturePairC64::new([-0.5, 0.3, -0.2], [1.0, 0.0, 0.0], 8, 4),
    ];
    let quadrature_expected = quadrature_pairs_host
        .iter()
        .copied()
        .map(|pair| {
            quadrature_pair_reference(
                pair,
                &quadrature_samples_host,
                quadrature_wave_number,
                quadrature_harmonic_factor,
            )
        })
        .collect::<Vec<_>>();
    let quadrature_pair_view = BufferView::new(7, 1, quadrature_pairs_host.len(), 2)
        .map_err(|error| format!("quadrature pair view: {error}"))?;
    let quadrature_output_view = BufferView::new(7, 1, quadrature_pairs_host.len(), 2)
        .map_err(|error| format!("quadrature output view: {error}"))?;
    let mut quadrature_pair_backing = vec![BemQuadraturePairC64::default(); 7];
    for (index, pair) in quadrature_pairs_host.iter().enumerate() {
        quadrature_pair_backing[quadrature_pair_view
            .checked_element_index(index)
            .map_err(|error| format!("quadrature pair view index: {error}"))?] = *pair;
    }
    let mut quadrature_pairs = pool
        .try_alloc::<BemQuadraturePairC64>(quadrature_pair_backing.len())
        .map_err(|error| format!("quadrature pair allocation: {error}"))?;
    let mut quadrature_samples = pool
        .try_alloc::<BemQuadratureSampleC64>(quadrature_samples_host.len())
        .map_err(|error| format!("quadrature sample allocation: {error}"))?;
    let quadrature_sentinel = BemRegularPairOutF64 {
        g: Complex64::new(7.0, -3.0),
        h: Complex64::new(5.0, -2.0),
        ht: Complex64::new(4.0, -1.0),
        e: Complex64::new(2.0, -6.0),
    };
    let mut quadrature_output = pool
        .try_alloc::<BemRegularPairOutF64>(7)
        .map_err(|error| format!("quadrature output allocation: {error}"))?;
    backend
        .try_copy_h2d(&quadrature_pair_backing, quadrature_pairs.raw_mut())
        .map_err(|error| format!("quadrature pair upload: {error}"))?;
    backend
        .try_copy_h2d(&quadrature_samples_host, quadrature_samples.raw_mut())
        .map_err(|error| format!("quadrature sample upload: {error}"))?;
    backend
        .try_copy_h2d(&vec![quadrature_sentinel; 7], quadrature_output.raw_mut())
        .map_err(|error| format!("quadrature output seed: {error}"))?;

    let empty_pairs = pool
        .try_alloc::<BemQuadraturePairC64>(0)
        .map_err(|error| format!("empty quadrature pair allocation: {error}"))?;
    let empty_samples = pool
        .try_alloc::<BemQuadratureSampleC64>(0)
        .map_err(|error| format!("empty quadrature sample allocation: {error}"))?;
    let mut empty_output = pool
        .try_alloc::<BemRegularPairOutF64>(0)
        .map_err(|error| format!("empty quadrature output allocation: {error}"))?;
    let empty_view = BufferView::contiguous(0, 0);
    let empty_noop = backend
        .try_bem_quadrature_pair_c64_device_view(
            empty_pairs.raw(),
            empty_view,
            empty_samples.raw(),
            empty_output.raw_mut(),
            empty_view,
            quadrature_wave_number,
            quadrature_harmonic_factor,
        )
        .is_ok();
    let malformed_view = BufferView {
        offset: 1,
        len: 1,
        stride: 0,
    };
    let malformed_error = backend
        .try_bem_quadrature_pair_c64_device_view(
            quadrature_pairs.raw(),
            malformed_view,
            quadrature_samples.raw(),
            quadrature_output.raw_mut(),
            BufferView::contiguous(0, 1),
            quadrature_wave_number,
            quadrature_harmonic_factor,
        )
        .expect_err("malformed quadrature view must be rejected before launch");
    let malformed_typed = matches!(
        malformed_error,
        BackendError::InvalidArgument {
            operation: "bem_quadrature_pair_c64_device_view",
            ..
        }
    );
    let empty_samples_error = backend
        .try_bem_quadrature_pair_c64_device_view(
            quadrature_pairs.raw(),
            BufferView::contiguous(0, 1),
            empty_samples.raw(),
            quadrature_output.raw_mut(),
            BufferView::contiguous(0, 1),
            quadrature_wave_number,
            quadrature_harmonic_factor,
        )
        .expect_err("non-empty quadrature view with empty samples must be rejected");
    let empty_typed = matches!(
        empty_samples_error,
        BackendError::InvalidArgument {
            operation: "bem_quadrature_pair_c64_device_view",
            ..
        }
    );
    if !empty_noop || !malformed_typed || !empty_typed {
        return Err(format!(
            "quadrature checked admission failed (empty_noop={empty_noop}, malformed_typed={malformed_typed}, empty_typed={empty_typed})"
        ));
    }

    let initial_action = action(
        backend,
        "bem.quadrature_pair.c64",
        "Complex64",
        "records_strided(offset=1,stride=2),samples_resident",
        || {
            backend.try_bem_quadrature_pair_c64_device_view(
                quadrature_pairs.raw(),
                quadrature_pair_view,
                quadrature_samples.raw(),
                quadrature_output.raw_mut(),
                quadrature_output_view,
                quadrature_wave_number,
                quadrature_harmonic_factor,
            )
        },
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("quadrature initial synchronization: {error}"))?;
    let mut initial_backing = vec![BemRegularPairOutF64::default(); 7];
    backend
        .try_copy_d2h(quadrature_output.raw(), &mut initial_backing)
        .map_err(|error| format!("quadrature initial readback: {error}"))?;
    let initial_logical = (0..quadrature_output_view.len)
        .map(|logical| {
            quadrature_output_view
                .checked_element_index(logical)
                .map(|index| initial_backing[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("quadrature initial output view: {error}"))?;
    let initial_error = quadrature_relative_error(&initial_logical, &quadrature_expected)?;

    let warm_action = action(
        backend,
        "bem.quadrature_pair.c64.warm_replay",
        "Complex64",
        "records_strided(offset=1,stride=2),samples_resident",
        || {
            backend.try_bem_quadrature_pair_c64_device_view(
                quadrature_pairs.raw(),
                quadrature_pair_view,
                quadrature_samples.raw(),
                quadrature_output.raw_mut(),
                quadrature_output_view,
                quadrature_wave_number,
                quadrature_harmonic_factor,
            )
        },
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("quadrature warm synchronization: {error}"))?;
    let mut warm_backing = vec![BemRegularPairOutF64::default(); 7];
    backend
        .try_copy_d2h(quadrature_output.raw(), &mut warm_backing)
        .map_err(|error| format!("quadrature warm readback: {error}"))?;
    let warm_logical = (0..quadrature_output_view.len)
        .map(|logical| {
            quadrature_output_view
                .checked_element_index(logical)
                .map(|index| warm_backing[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("quadrature warm output view: {error}"))?;
    let warm_error = quadrature_relative_error(&warm_logical, &quadrature_expected)?;

    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure_error = backend
        .try_bem_quadrature_pair_c64_device_view(
            quadrature_pairs.raw(),
            quadrature_pair_view,
            quadrature_samples.raw(),
            quadrature_output.raw_mut(),
            quadrature_output_view,
            quadrature_wave_number,
            quadrature_harmonic_factor,
        )
        .expect_err("injected quadrature launch failure must be typed");
    let failure_typed = matches!(
        failure_error,
        BackendError::Launch {
            operation: "cuda_launch",
            ..
        }
    );
    backend
        .recover_after_error()
        .map_err(|error| format!("quadrature failure recovery: {error}"))?;
    let mut after_failure_backing = vec![BemRegularPairOutF64::default(); 7];
    backend
        .try_copy_d2h(quadrature_output.raw(), &mut after_failure_backing)
        .map_err(|error| format!("quadrature failed-output readback: {error}"))?;
    let failed_logical = (0..quadrature_output_view.len)
        .map(|logical| {
            quadrature_output_view
                .checked_element_index(logical)
                .map(|index| after_failure_backing[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("quadrature failed-output view: {error}"))?;
    let failure_preserved = quadrature_relative_error(&failed_logical, &warm_logical)
        .map(|error| error == 0.0)
        .unwrap_or(false);
    if !failure_typed || !failure_preserved {
        return Err(format!(
            "quadrature failure contract failed (typed={failure_typed}, output_preserved={failure_preserved})"
        ));
    }

    let retry_action = action(
        backend,
        "bem.quadrature_pair.c64.retry",
        "Complex64",
        "records_strided(offset=1,stride=2),samples_resident",
        || {
            backend.try_bem_quadrature_pair_c64_device_view(
                quadrature_pairs.raw(),
                quadrature_pair_view,
                quadrature_samples.raw(),
                quadrature_output.raw_mut(),
                quadrature_output_view,
                quadrature_wave_number,
                quadrature_harmonic_factor,
            )
        },
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("quadrature retry synchronization: {error}"))?;
    let mut retry_backing = vec![BemRegularPairOutF64::default(); 7];
    backend
        .try_copy_d2h(quadrature_output.raw(), &mut retry_backing)
        .map_err(|error| format!("quadrature retry readback: {error}"))?;
    let retry_logical = (0..quadrature_output_view.len)
        .map(|logical| {
            quadrature_output_view
                .checked_element_index(logical)
                .map(|index| retry_backing[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("quadrature retry output view: {error}"))?;
    let retry_error = quadrature_relative_error(&retry_logical, &quadrature_expected)?;
    let quadrature_residency = initial_action
        .get("residency")
        .cloned()
        .ok_or_else(|| "quadrature initial action missing residency evidence".to_string())?;
    rows.push(json!({
        "phase": "bem.quadrature_pair.c64",
        "scalar": "Complex64",
        "layout": "records_strided(offset=1,stride=2),samples_resident",
        "status": "native",
        "residency": quadrature_residency,
        "relative_error": initial_error,
        "warm_relative_error": warm_error,
        "retry_relative_error": retry_error,
        "tolerance": 2.0e-11,
        "typed_rejections": {
            "malformed_view": malformed_typed,
            "empty_samples": empty_typed,
            "empty_view_noop": empty_noop,
        },
        "failure_recovery": {
            "typed": failure_typed,
            "output_preserved": failure_preserved,
        },
        "warm_replay": {
            "initial": initial_action,
            "warm": warm_action,
            "retry": retry_action,
        },
    }));

    // The chart-driven wavenumber derivative has its own checked-view ABI.
    // Keep the descriptor and output storage independently strided, and use
    // both an analytic scalar CPU reference and a centered finite difference
    // of the independent CPU value reference.  The action itself only binds
    // resident device allocations and therefore must remain transfer-free.
    let quadrature_derivative_expected = quadrature_pairs_host
        .iter()
        .copied()
        .map(|pair| {
            quadrature_pair_wavenumber_derivative_reference(
                pair,
                &quadrature_samples_host,
                quadrature_wave_number,
                quadrature_harmonic_factor,
            )
        })
        .collect::<Vec<_>>();
    let quadrature_derivative_step = 1.0e-5;
    let quadrature_derivative_centered_expected = quadrature_pairs_host
        .iter()
        .copied()
        .map(|pair| {
            let plus = quadrature_pair_reference(
                pair,
                &quadrature_samples_host,
                quadrature_wave_number + quadrature_derivative_step,
                quadrature_harmonic_factor,
            );
            let minus = quadrature_pair_reference(
                pair,
                &quadrature_samples_host,
                quadrature_wave_number - quadrature_derivative_step,
                quadrature_harmonic_factor,
            );
            BemRegularPairOutF64 {
                g: (plus.g - minus.g) / (2.0 * quadrature_derivative_step),
                h: (plus.h - minus.h) / (2.0 * quadrature_derivative_step),
                ht: (plus.ht - minus.ht) / (2.0 * quadrature_derivative_step),
                e: (plus.e - minus.e) / (2.0 * quadrature_derivative_step),
            }
        })
        .collect::<Vec<_>>();
    let quadrature_derivative_centered_reference_error = quadrature_relative_error(
        &quadrature_derivative_centered_expected,
        &quadrature_derivative_expected,
    )?;
    let quadrature_derivative_sentinel = BemRegularPairOutF64 {
        g: Complex64::new(-37.0, 31.0),
        h: Complex64::new(-29.0, 23.0),
        ht: Complex64::new(-19.0, 17.0),
        e: Complex64::new(-13.0, 11.0),
    };
    let mut quadrature_derivative_output = pool
        .try_alloc::<BemRegularPairOutF64>(7)
        .map_err(|error| format!("quadrature derivative output allocation: {error}"))?;
    backend
        .try_copy_h2d(
            &[quadrature_derivative_sentinel; 7],
            quadrature_derivative_output.raw_mut(),
        )
        .map_err(|error| format!("quadrature derivative output seed: {error}"))?;

    let empty_quadrature_derivative_noop = backend
        .try_bem_quadrature_pair_c64_derivative_device_view(
            empty_pairs.raw(),
            empty_view,
            empty_samples.raw(),
            empty_output.raw_mut(),
            empty_view,
            quadrature_wave_number,
            quadrature_harmonic_factor,
        )
        .is_ok();
    let malformed_quadrature_derivative_error = backend
        .try_bem_quadrature_pair_c64_derivative_device_view(
            quadrature_pairs.raw(),
            malformed_view,
            quadrature_samples.raw(),
            quadrature_derivative_output.raw_mut(),
            BufferView::contiguous(0, 1),
            quadrature_wave_number,
            quadrature_harmonic_factor,
        )
        .expect_err("malformed quadrature derivative view must be rejected before launch");
    let malformed_quadrature_derivative_typed = matches!(
        malformed_quadrature_derivative_error,
        BackendError::InvalidArgument {
            operation: "bem_quadrature_pair_c64_derivative_device_view",
            ..
        }
    );
    let empty_quadrature_derivative_samples_error = backend
        .try_bem_quadrature_pair_c64_derivative_device_view(
            quadrature_pairs.raw(),
            BufferView::contiguous(0, 1),
            empty_samples.raw(),
            quadrature_derivative_output.raw_mut(),
            BufferView::contiguous(0, 1),
            quadrature_wave_number,
            quadrature_harmonic_factor,
        )
        .expect_err("non-empty quadrature derivative view with empty samples must be rejected");
    let empty_quadrature_derivative_samples_typed = matches!(
        empty_quadrature_derivative_samples_error,
        BackendError::InvalidArgument {
            operation: "bem_quadrature_pair_c64_derivative_device_view",
            ..
        }
    );
    if !empty_quadrature_derivative_noop
        || !malformed_quadrature_derivative_typed
        || !empty_quadrature_derivative_samples_typed
    {
        return Err(format!(
            "quadrature derivative checked admission failed (empty_noop={empty_quadrature_derivative_noop}, malformed_typed={malformed_quadrature_derivative_typed}, empty_samples_typed={empty_quadrature_derivative_samples_typed})"
        ));
    }

    let quadrature_derivative_initial_action = action(
        backend,
        "bem.quadrature_pair_derivative.view.c64",
        "Complex64",
        "records_strided(offset=1,stride=2),samples_resident,analytic_dk",
        || {
            backend.try_bem_quadrature_pair_c64_derivative_device_view(
                quadrature_pairs.raw(),
                quadrature_pair_view,
                quadrature_samples.raw(),
                quadrature_derivative_output.raw_mut(),
                quadrature_output_view,
                quadrature_wave_number,
                quadrature_harmonic_factor,
            )
        },
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("quadrature derivative initial synchronization: {error}"))?;
    let mut quadrature_derivative_initial_backing = vec![BemRegularPairOutF64::default(); 7];
    backend
        .try_copy_d2h(
            quadrature_derivative_output.raw(),
            &mut quadrature_derivative_initial_backing,
        )
        .map_err(|error| format!("quadrature derivative initial readback: {error}"))?;
    let quadrature_derivative_initial_logical = (0..quadrature_output_view.len)
        .map(|logical| {
            quadrature_output_view
                .checked_element_index(logical)
                .map(|index| quadrature_derivative_initial_backing[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("quadrature derivative initial output: {error}"))?;
    let quadrature_derivative_initial_error = quadrature_relative_error(
        &quadrature_derivative_initial_logical,
        &quadrature_derivative_expected,
    )?;

    let quadrature_derivative_warm_action = action(
        backend,
        "bem.quadrature_pair_derivative.view.c64.warm_replay",
        "Complex64",
        "records_strided(offset=1,stride=2),samples_resident,analytic_dk",
        || {
            backend.try_bem_quadrature_pair_c64_derivative_device_view(
                quadrature_pairs.raw(),
                quadrature_pair_view,
                quadrature_samples.raw(),
                quadrature_derivative_output.raw_mut(),
                quadrature_output_view,
                quadrature_wave_number,
                quadrature_harmonic_factor,
            )
        },
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("quadrature derivative warm synchronization: {error}"))?;
    let mut quadrature_derivative_warm_backing = vec![BemRegularPairOutF64::default(); 7];
    backend
        .try_copy_d2h(
            quadrature_derivative_output.raw(),
            &mut quadrature_derivative_warm_backing,
        )
        .map_err(|error| format!("quadrature derivative warm readback: {error}"))?;
    let quadrature_derivative_warm_logical = (0..quadrature_output_view.len)
        .map(|logical| {
            quadrature_output_view
                .checked_element_index(logical)
                .map(|index| quadrature_derivative_warm_backing[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("quadrature derivative warm output: {error}"))?;
    let quadrature_derivative_warm_error = quadrature_relative_error(
        &quadrature_derivative_warm_logical,
        &quadrature_derivative_expected,
    )?;

    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let quadrature_derivative_failure_error = backend
        .try_bem_quadrature_pair_c64_derivative_device_view(
            quadrature_pairs.raw(),
            quadrature_pair_view,
            quadrature_samples.raw(),
            quadrature_derivative_output.raw_mut(),
            quadrature_output_view,
            quadrature_wave_number,
            quadrature_harmonic_factor,
        )
        .expect_err("injected quadrature derivative launch failure must be typed");
    let quadrature_derivative_failure_typed = matches!(
        quadrature_derivative_failure_error,
        BackendError::Launch {
            operation: "cuda_launch",
            ..
        }
    );
    backend
        .recover_after_error()
        .map_err(|error| format!("quadrature derivative failure recovery: {error}"))?;
    let mut quadrature_derivative_failed_backing = vec![BemRegularPairOutF64::default(); 7];
    backend
        .try_copy_d2h(
            quadrature_derivative_output.raw(),
            &mut quadrature_derivative_failed_backing,
        )
        .map_err(|error| format!("quadrature derivative failed-output readback: {error}"))?;
    let quadrature_derivative_failure_preserved = quadrature_derivative_failed_backing
        .iter()
        .zip(&quadrature_derivative_warm_backing)
        .all(|(failed, warm)| {
            failed.g == warm.g && failed.h == warm.h && failed.ht == warm.ht && failed.e == warm.e
        });
    if !quadrature_derivative_failure_typed || !quadrature_derivative_failure_preserved {
        return Err(format!(
            "quadrature derivative failure contract failed (typed={quadrature_derivative_failure_typed}, output_preserved={quadrature_derivative_failure_preserved})"
        ));
    }

    let quadrature_derivative_retry_action = action(
        backend,
        "bem.quadrature_pair_derivative.view.c64.retry",
        "Complex64",
        "records_strided(offset=1,stride=2),samples_resident,analytic_dk",
        || {
            backend.try_bem_quadrature_pair_c64_derivative_device_view(
                quadrature_pairs.raw(),
                quadrature_pair_view,
                quadrature_samples.raw(),
                quadrature_derivative_output.raw_mut(),
                quadrature_output_view,
                quadrature_wave_number,
                quadrature_harmonic_factor,
            )
        },
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("quadrature derivative retry synchronization: {error}"))?;
    let mut quadrature_derivative_retry_backing = vec![BemRegularPairOutF64::default(); 7];
    backend
        .try_copy_d2h(
            quadrature_derivative_output.raw(),
            &mut quadrature_derivative_retry_backing,
        )
        .map_err(|error| format!("quadrature derivative retry readback: {error}"))?;
    let quadrature_derivative_retry_logical = (0..quadrature_output_view.len)
        .map(|logical| {
            quadrature_output_view
                .checked_element_index(logical)
                .map(|index| quadrature_derivative_retry_backing[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("quadrature derivative retry output: {error}"))?;
    let quadrature_derivative_retry_error = quadrature_relative_error(
        &quadrature_derivative_retry_logical,
        &quadrature_derivative_expected,
    )?;
    let quadrature_derivative_residency = quadrature_derivative_initial_action
        .get("residency")
        .cloned()
        .ok_or_else(|| {
            "quadrature derivative initial action missing residency evidence".to_string()
        })?;
    rows.push(json!({
        "phase": "bem.quadrature_pair_derivative.view.c64",
        "scalar": "Complex64",
        "layout": "records_strided(offset=1,stride=2),samples_resident,analytic_dk",
        "status": "native",
        "residency": quadrature_derivative_residency,
        "relative_error": quadrature_derivative_initial_error,
        "warm_relative_error": quadrature_derivative_warm_error,
        "retry_relative_error": quadrature_derivative_retry_error,
        "centered_reference_relative_error": quadrature_derivative_centered_reference_error,
        "tolerance": 1.0e-7,
        "typed_rejections": {
            "malformed_view": malformed_quadrature_derivative_typed,
            "empty_samples": empty_quadrature_derivative_samples_typed,
            "empty_view_noop": empty_quadrature_derivative_noop,
        },
        "failure_recovery": {
            "typed": quadrature_derivative_failure_typed,
            "output_preserved": quadrature_derivative_failure_preserved,
        },
        "warm_replay": {
            "initial": quadrature_derivative_initial_action,
            "warm": quadrature_derivative_warm_action,
            "retry": quadrature_derivative_retry_action,
        },
    }));

    // Constant-triangle field evaluation has its own ABI: geometry and traces
    // are contiguous while points and output are independently strided views.
    // Keep this packet separate from the regular/quadrature pair descriptors
    // so the checked field-view contract is exercised directly.
    let field_wave_number = 2.3;
    let field_harmonic_factor = -1.0;
    let field_pressure_factor = 3.7;
    let field_element_host = BemFieldElementC64::from_triangle(
        [-0.7, -0.3, 0.0],
        [0.8, -0.2, 0.1],
        [-0.1, 0.9, -0.15],
        Complex64::new(1.2, -0.4),
        Complex64::new(-0.15, 0.25),
    );
    let field_pressure_host = field_element_host.pressure;
    let field_velocity_host = field_element_host.velocity;
    let field_points_host = [
        BemFieldPointC64::new([0.3, -0.4, 1.8]),
        BemFieldPointC64::new([-0.8, 0.2, 1.3]),
        BemFieldPointC64::new([0.1, 0.6, 2.1]),
    ];
    let field_expected = field_points_host
        .iter()
        .copied()
        .map(|point| {
            field_reference(
                point,
                field_element_host,
                field_pressure_host,
                field_velocity_host,
                field_wave_number,
                field_harmonic_factor,
                field_pressure_factor,
            )
        })
        .collect::<Vec<_>>();
    let field_points_view = BufferView::new(7, 1, field_points_host.len(), 2)
        .map_err(|error| format!("field point view: {error}"))?;
    let field_output_view = BufferView::new(7, 1, field_points_host.len(), 2)
        .map_err(|error| format!("field output view: {error}"))?;
    let mut field_elements = pool
        .try_alloc::<BemFieldElementC64>(1)
        .map_err(|error| format!("field element allocation: {error}"))?;
    let mut field_pressure = pool
        .try_alloc::<Complex64>(1)
        .map_err(|error| format!("field pressure allocation: {error}"))?;
    let mut field_velocity = pool
        .try_alloc::<Complex64>(1)
        .map_err(|error| format!("field velocity allocation: {error}"))?;
    let mut field_points = pool
        .try_alloc::<BemFieldPointC64>(7)
        .map_err(|error| format!("field point allocation: {error}"))?;
    let field_sentinel = Complex64::new(8.0, -4.0);
    let mut field_output = pool
        .try_alloc::<Complex64>(7)
        .map_err(|error| format!("field output allocation: {error}"))?;
    let mut field_points_backing = vec![BemFieldPointC64::default(); 7];
    for (index, point) in field_points_host.iter().enumerate() {
        let storage_index = field_points_view
            .checked_element_index(index)
            .map_err(|error| format!("field point view index: {error}"))?;
        field_points_backing[storage_index] = *point;
    }
    backend
        .try_copy_h2d(
            std::slice::from_ref(&field_element_host),
            field_elements.raw_mut(),
        )
        .map_err(|error| format!("field element upload: {error}"))?;
    backend
        .try_copy_h2d(
            std::slice::from_ref(&field_pressure_host),
            field_pressure.raw_mut(),
        )
        .map_err(|error| format!("field pressure upload: {error}"))?;
    backend
        .try_copy_h2d(
            std::slice::from_ref(&field_velocity_host),
            field_velocity.raw_mut(),
        )
        .map_err(|error| format!("field velocity upload: {error}"))?;
    backend
        .try_copy_h2d(&field_points_backing, field_points.raw_mut())
        .map_err(|error| format!("field point upload: {error}"))?;
    backend
        .try_copy_h2d(&[field_sentinel; 7], field_output.raw_mut())
        .map_err(|error| format!("field output seed: {error}"))?;

    let empty_field_points = pool
        .try_alloc::<BemFieldPointC64>(0)
        .map_err(|error| format!("empty field point allocation: {error}"))?;
    let mut empty_field_output = pool
        .try_alloc::<Complex64>(0)
        .map_err(|error| format!("empty field output allocation: {error}"))?;
    let empty_field_view = BufferView::contiguous(0, 0);
    let empty_points_noop = backend
        .try_bem_field_c64_device_with_traces_view(
            field_elements.raw(),
            field_pressure.raw(),
            field_velocity.raw(),
            empty_field_points.raw(),
            empty_field_view,
            empty_field_output.raw_mut(),
            empty_field_view,
            field_wave_number,
            field_harmonic_factor,
            field_pressure_factor,
        )
        .is_ok();
    let malformed_field_view = BufferView {
        offset: 1,
        len: 1,
        stride: 0,
    };
    let malformed_field_error = backend
        .try_bem_field_c64_device_with_traces_view(
            field_elements.raw(),
            field_pressure.raw(),
            field_velocity.raw(),
            field_points.raw(),
            malformed_field_view,
            field_output.raw_mut(),
            BufferView::contiguous(0, 1),
            field_wave_number,
            field_harmonic_factor,
            field_pressure_factor,
        )
        .expect_err("malformed field point view must be rejected before launch");
    let malformed_field_typed = matches!(
        malformed_field_error,
        BackendError::InvalidArgument {
            operation: "bem_field_c64_device_view",
            ..
        }
    );
    let empty_field_elements = pool
        .try_alloc::<BemFieldElementC64>(0)
        .map_err(|error| format!("empty field element allocation: {error}"))?;
    let empty_field_pressure = pool
        .try_alloc::<Complex64>(0)
        .map_err(|error| format!("empty field pressure allocation: {error}"))?;
    let empty_field_velocity = pool
        .try_alloc::<Complex64>(0)
        .map_err(|error| format!("empty field velocity allocation: {error}"))?;
    let empty_field_error = backend
        .try_bem_field_c64_device_with_traces_view(
            empty_field_elements.raw(),
            empty_field_pressure.raw(),
            empty_field_velocity.raw(),
            field_points.raw(),
            BufferView::contiguous(0, 1),
            field_output.raw_mut(),
            BufferView::contiguous(0, 1),
            field_wave_number,
            field_harmonic_factor,
            field_pressure_factor,
        )
        .expect_err("empty field element buffer must be rejected");
    let empty_field_typed = matches!(
        empty_field_error,
        BackendError::InvalidArgument {
            operation: "bem_field_c64_device_view",
            ..
        }
    );
    if !empty_points_noop || !malformed_field_typed || !empty_field_typed {
        return Err(format!(
            "field checked admission failed (empty_noop={empty_points_noop}, malformed_typed={malformed_field_typed}, empty_elements_typed={empty_field_typed})"
        ));
    }

    let field_initial_action = action(
        backend,
        "bem.field.c64",
        "Complex64",
        "points/output_strided(offset=1,stride=2),geometry_traces_resident",
        || {
            backend.try_bem_field_c64_device_with_traces_view(
                field_elements.raw(),
                field_pressure.raw(),
                field_velocity.raw(),
                field_points.raw(),
                field_points_view,
                field_output.raw_mut(),
                field_output_view,
                field_wave_number,
                field_harmonic_factor,
                field_pressure_factor,
            )
        },
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("field initial synchronization: {error}"))?;
    let mut field_initial_backing = vec![Complex64::default(); 7];
    backend
        .try_copy_d2h(field_output.raw(), &mut field_initial_backing)
        .map_err(|error| format!("field initial readback: {error}"))?;
    let field_initial_logical = (0..field_output_view.len)
        .map(|logical| {
            field_output_view
                .checked_element_index(logical)
                .map(|index| field_initial_backing[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("field initial output view: {error}"))?;
    let field_initial_error = field_relative_error(&field_initial_logical, &field_expected)?;

    let field_warm_action = action(
        backend,
        "bem.field.c64.warm_replay",
        "Complex64",
        "points/output_strided(offset=1,stride=2),geometry_traces_resident",
        || {
            backend.try_bem_field_c64_device_with_traces_view(
                field_elements.raw(),
                field_pressure.raw(),
                field_velocity.raw(),
                field_points.raw(),
                field_points_view,
                field_output.raw_mut(),
                field_output_view,
                field_wave_number,
                field_harmonic_factor,
                field_pressure_factor,
            )
        },
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("field warm synchronization: {error}"))?;
    let mut field_warm_backing = vec![Complex64::default(); 7];
    backend
        .try_copy_d2h(field_output.raw(), &mut field_warm_backing)
        .map_err(|error| format!("field warm readback: {error}"))?;
    let field_warm_logical = (0..field_output_view.len)
        .map(|logical| {
            field_output_view
                .checked_element_index(logical)
                .map(|index| field_warm_backing[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("field warm output view: {error}"))?;
    let field_warm_error = field_relative_error(&field_warm_logical, &field_expected)?;

    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let field_failure_error = backend
        .try_bem_field_c64_device_with_traces_view(
            field_elements.raw(),
            field_pressure.raw(),
            field_velocity.raw(),
            field_points.raw(),
            field_points_view,
            field_output.raw_mut(),
            field_output_view,
            field_wave_number,
            field_harmonic_factor,
            field_pressure_factor,
        )
        .expect_err("injected field launch failure must be typed");
    let field_failure_typed = matches!(
        field_failure_error,
        BackendError::Launch {
            operation: "cuda_launch",
            ..
        }
    );
    backend
        .recover_after_error()
        .map_err(|error| format!("field failure recovery: {error}"))?;
    let mut field_failed_backing = vec![Complex64::default(); 7];
    backend
        .try_copy_d2h(field_output.raw(), &mut field_failed_backing)
        .map_err(|error| format!("field failed-output readback: {error}"))?;
    let field_failure_preserved = field_failed_backing == field_warm_backing;
    if !field_failure_typed || !field_failure_preserved {
        return Err(format!(
            "field failure contract failed (typed={field_failure_typed}, output_preserved={field_failure_preserved})"
        ));
    }

    let field_retry_action = action(
        backend,
        "bem.field.c64.retry",
        "Complex64",
        "points/output_strided(offset=1,stride=2),geometry_traces_resident",
        || {
            backend.try_bem_field_c64_device_with_traces_view(
                field_elements.raw(),
                field_pressure.raw(),
                field_velocity.raw(),
                field_points.raw(),
                field_points_view,
                field_output.raw_mut(),
                field_output_view,
                field_wave_number,
                field_harmonic_factor,
                field_pressure_factor,
            )
        },
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("field retry synchronization: {error}"))?;
    let mut field_retry_backing = vec![Complex64::default(); 7];
    backend
        .try_copy_d2h(field_output.raw(), &mut field_retry_backing)
        .map_err(|error| format!("field retry readback: {error}"))?;
    let field_retry_logical = (0..field_output_view.len)
        .map(|logical| {
            field_output_view
                .checked_element_index(logical)
                .map(|index| field_retry_backing[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("field retry output view: {error}"))?;
    let field_retry_error = field_relative_error(&field_retry_logical, &field_expected)?;
    let field_residency = field_initial_action
        .get("residency")
        .cloned()
        .ok_or_else(|| "field initial action missing residency evidence".to_string())?;
    rows.push(json!({
        "phase": "bem.field.c64",
        "scalar": "Complex64",
        "layout": "points/output_strided(offset=1,stride=2),geometry_traces_resident",
        "status": "native",
        "residency": field_residency,
        "relative_error": field_initial_error,
        "warm_relative_error": field_warm_error,
        "retry_relative_error": field_retry_error,
        "tolerance": 3.0e-12,
        "typed_rejections": {
            "malformed_view": malformed_field_typed,
            "empty_points_noop": empty_points_noop,
            "empty_elements": empty_field_typed,
        },
        "failure_recovery": {
            "typed": field_failure_typed,
            "output_preserved": field_failure_preserved,
        },
        "warm_replay": {
            "initial": field_initial_action,
            "warm": field_warm_action,
            "retry": field_retry_action,
        },
    }));

    let mut derivative_pairs = pool
        .try_alloc::<BemRegularPairInF64>(pair_values.len())
        .map_err(|error| format!("derivative pair allocation: {error}"))?;
    let mut derivative_output = pool
        .try_alloc::<BemRegularPairOutF64>(pair_values.len())
        .map_err(|error| format!("derivative output allocation: {error}"))?;
    backend
        .try_copy_h2d(&pair_values, derivative_pairs.raw_mut())
        .map_err(|error| format!("derivative pair upload: {error}"))?;
    rows.push(action(
        backend,
        "bem.regular_pair_derivative.c64",
        "Complex64",
        "contiguous_device",
        || {
            backend.try_bem_regular_pair_c64_derivative_device(
                derivative_pairs.raw(),
                derivative_output.raw_mut(),
                1.0,
                -1.0,
            )
        },
    )?);

    // The checked derivative view is a distinct native cell from the
    // contiguous compatibility call above.  Reuse the already resident pair
    // descriptors, place the four logical records at an independent offset /
    // stride, and seed every output slot so a failed launch can be checked for
    // exact preservation without another action-time upload.
    let derivative_wave_number = 1.0;
    let derivative_harmonic_factor = -1.0;
    let derivative_expected = pair_values
        .iter()
        .copied()
        .map(|pair| {
            regular_pair_wavenumber_derivative_reference(
                pair,
                derivative_wave_number,
                derivative_harmonic_factor,
            )
        })
        .collect::<Vec<_>>();
    let centered_step = 1.0e-5;
    let centered_expected = pair_values
        .iter()
        .copied()
        .map(|pair| {
            let plus = regular_pair_reference(
                pair,
                derivative_wave_number + centered_step,
                derivative_harmonic_factor,
            );
            let minus = regular_pair_reference(
                pair,
                derivative_wave_number - centered_step,
                derivative_harmonic_factor,
            );
            BemRegularPairOutF64 {
                g: (plus.g - minus.g) / (2.0 * centered_step),
                h: (plus.h - minus.h) / (2.0 * centered_step),
                ht: (plus.ht - minus.ht) / (2.0 * centered_step),
                e: (plus.e - minus.e) / (2.0 * centered_step),
            }
        })
        .collect::<Vec<_>>();
    let centered_reference_error =
        regular_pair_relative_error(&centered_expected, &derivative_expected)?;
    let derivative_pair_view = BufferView::new(9, 1, pair_values.len(), 2)
        .map_err(|error| format!("derivative pair view: {error}"))?;
    let derivative_output_view = BufferView::new(9, 1, pair_values.len(), 2)
        .map_err(|error| format!("derivative output view: {error}"))?;
    let derivative_sentinel = BemRegularPairOutF64 {
        g: Complex64::new(17.0, -11.0),
        h: Complex64::new(13.0, -7.0),
        ht: Complex64::new(5.0, -19.0),
        e: Complex64::new(23.0, -3.0),
    };
    let mut derivative_view_output = pool
        .try_alloc::<BemRegularPairOutF64>(9)
        .map_err(|error| format!("derivative view output allocation: {error}"))?;
    backend
        .try_copy_h2d(&[derivative_sentinel; 9], derivative_view_output.raw_mut())
        .map_err(|error| format!("derivative view output seed: {error}"))?;

    let empty_derivative_pairs = pool
        .try_alloc::<BemRegularPairInF64>(0)
        .map_err(|error| format!("empty derivative pair allocation: {error}"))?;
    let mut empty_derivative_output = pool
        .try_alloc::<BemRegularPairOutF64>(0)
        .map_err(|error| format!("empty derivative output allocation: {error}"))?;
    let empty_derivative_view = BufferView::contiguous(0, 0);
    let empty_derivative_noop = backend
        .try_bem_regular_pair_c64_derivative_device_view(
            empty_derivative_pairs.raw(),
            empty_derivative_view,
            empty_derivative_output.raw_mut(),
            empty_derivative_view,
            derivative_wave_number,
            derivative_harmonic_factor,
        )
        .is_ok();
    let malformed_derivative_view = BufferView {
        offset: 1,
        len: 1,
        stride: 0,
    };
    let malformed_derivative_error = backend
        .try_bem_regular_pair_c64_derivative_device_view(
            pairs.raw(),
            malformed_derivative_view,
            derivative_view_output.raw_mut(),
            BufferView::contiguous(0, 1),
            derivative_wave_number,
            derivative_harmonic_factor,
        )
        .expect_err("malformed derivative view must be rejected before launch");
    let malformed_derivative_typed = matches!(
        malformed_derivative_error,
        BackendError::InvalidArgument {
            operation: "bem_regular_pair_c64_derivative_device_view",
            ..
        }
    );
    let empty_derivative_output_error = backend
        .try_bem_regular_pair_c64_derivative_device_view(
            pairs.raw(),
            BufferView::contiguous(0, 1),
            empty_derivative_output.raw_mut(),
            BufferView::contiguous(0, 1),
            derivative_wave_number,
            derivative_harmonic_factor,
        )
        .expect_err("empty derivative output must be rejected for a non-empty view");
    let empty_derivative_typed = matches!(
        empty_derivative_output_error,
        BackendError::InvalidArgument {
            operation: "bem_regular_pair_c64_derivative_device_view",
            ..
        }
    );
    if !empty_derivative_noop || !malformed_derivative_typed || !empty_derivative_typed {
        return Err(format!(
            "derivative checked admission failed (empty_noop={empty_derivative_noop}, malformed_typed={malformed_derivative_typed}, empty_output_typed={empty_derivative_typed})"
        ));
    }

    let derivative_initial_action = action(
        backend,
        "bem.regular_pair_derivative.view.c64",
        "Complex64",
        "records_strided(offset=1,stride=2),analytic_dk",
        || {
            backend.try_bem_regular_pair_c64_derivative_device_view(
                pairs.raw(),
                derivative_pair_view,
                derivative_view_output.raw_mut(),
                derivative_output_view,
                derivative_wave_number,
                derivative_harmonic_factor,
            )
        },
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("derivative view initial synchronization: {error}"))?;
    let mut derivative_initial_backing = vec![BemRegularPairOutF64::default(); 9];
    backend
        .try_copy_d2h(
            derivative_view_output.raw(),
            &mut derivative_initial_backing,
        )
        .map_err(|error| format!("derivative view initial readback: {error}"))?;
    let derivative_initial_logical = (0..derivative_output_view.len)
        .map(|logical| {
            derivative_output_view
                .checked_element_index(logical)
                .map(|index| derivative_initial_backing[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("derivative view initial output: {error}"))?;
    let derivative_initial_error =
        regular_pair_relative_error(&derivative_initial_logical, &derivative_expected)?;

    let derivative_warm_action = action(
        backend,
        "bem.regular_pair_derivative.view.c64.warm_replay",
        "Complex64",
        "records_strided(offset=1,stride=2),analytic_dk",
        || {
            backend.try_bem_regular_pair_c64_derivative_device_view(
                pairs.raw(),
                derivative_pair_view,
                derivative_view_output.raw_mut(),
                derivative_output_view,
                derivative_wave_number,
                derivative_harmonic_factor,
            )
        },
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("derivative view warm synchronization: {error}"))?;
    let mut derivative_warm_backing = vec![BemRegularPairOutF64::default(); 9];
    backend
        .try_copy_d2h(derivative_view_output.raw(), &mut derivative_warm_backing)
        .map_err(|error| format!("derivative view warm readback: {error}"))?;
    let derivative_warm_logical = (0..derivative_output_view.len)
        .map(|logical| {
            derivative_output_view
                .checked_element_index(logical)
                .map(|index| derivative_warm_backing[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("derivative view warm output: {error}"))?;
    let derivative_warm_error =
        regular_pair_relative_error(&derivative_warm_logical, &derivative_expected)?;

    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let derivative_failure_error = backend
        .try_bem_regular_pair_c64_derivative_device_view(
            pairs.raw(),
            derivative_pair_view,
            derivative_view_output.raw_mut(),
            derivative_output_view,
            derivative_wave_number,
            derivative_harmonic_factor,
        )
        .expect_err("injected derivative view launch failure must be typed");
    let derivative_failure_typed = matches!(
        derivative_failure_error,
        BackendError::Launch {
            operation: "cuda_launch",
            ..
        }
    );
    backend
        .recover_after_error()
        .map_err(|error| format!("derivative view failure recovery: {error}"))?;
    let mut derivative_failed_backing = vec![BemRegularPairOutF64::default(); 9];
    backend
        .try_copy_d2h(derivative_view_output.raw(), &mut derivative_failed_backing)
        .map_err(|error| format!("derivative view failed-output readback: {error}"))?;
    let derivative_failure_preserved = derivative_failed_backing
        .iter()
        .zip(&derivative_warm_backing)
        .all(|(failed, warm)| {
            failed.g == warm.g && failed.h == warm.h && failed.ht == warm.ht && failed.e == warm.e
        });
    if !derivative_failure_typed || !derivative_failure_preserved {
        return Err(format!(
            "derivative view failure contract failed (typed={derivative_failure_typed}, output_preserved={derivative_failure_preserved})"
        ));
    }

    let derivative_retry_action = action(
        backend,
        "bem.regular_pair_derivative.view.c64.retry",
        "Complex64",
        "records_strided(offset=1,stride=2),analytic_dk",
        || {
            backend.try_bem_regular_pair_c64_derivative_device_view(
                pairs.raw(),
                derivative_pair_view,
                derivative_view_output.raw_mut(),
                derivative_output_view,
                derivative_wave_number,
                derivative_harmonic_factor,
            )
        },
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("derivative view retry synchronization: {error}"))?;
    let mut derivative_retry_backing = vec![BemRegularPairOutF64::default(); 9];
    backend
        .try_copy_d2h(derivative_view_output.raw(), &mut derivative_retry_backing)
        .map_err(|error| format!("derivative view retry readback: {error}"))?;
    let derivative_retry_logical = (0..derivative_output_view.len)
        .map(|logical| {
            derivative_output_view
                .checked_element_index(logical)
                .map(|index| derivative_retry_backing[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("derivative view retry output: {error}"))?;
    let derivative_retry_error =
        regular_pair_relative_error(&derivative_retry_logical, &derivative_expected)?;
    let derivative_residency = derivative_initial_action
        .get("residency")
        .cloned()
        .ok_or_else(|| "derivative view initial action missing residency evidence".to_string())?;
    rows.push(json!({
        "phase": "bem.regular_pair_derivative.view.c64",
        "scalar": "Complex64",
        "layout": "records_strided(offset=1,stride=2),analytic_dk",
        "status": "native",
        "residency": derivative_residency,
        "relative_error": derivative_initial_error,
        "warm_relative_error": derivative_warm_error,
        "retry_relative_error": derivative_retry_error,
        "centered_reference_relative_error": centered_reference_error,
        "tolerance": 1.0e-7,
        "typed_rejections": {
            "malformed_view": malformed_derivative_typed,
            "empty_output": empty_derivative_typed,
            "empty_view_noop": empty_derivative_noop,
        },
        "failure_recovery": {
            "typed": derivative_failure_typed,
            "output_preserved": derivative_failure_preserved,
        },
        "warm_replay": {
            "initial": derivative_initial_action,
            "warm": derivative_warm_action,
            "retry": derivative_retry_action,
        },
    }));

    // The source-collocation x derivative has a separate native kernel and
    // checked-view ABI.  Reuse the resident strided pair records, publish to
    // a separately seeded strided output, and compare against a centered CPU
    // displacement reference so this cell proves geometry differentiation,
    // not just another wavenumber derivative launch.
    let source_x_wave_number = 1.0;
    let source_x_harmonic_factor = -1.0;
    let source_x_step = 1.0e-5;
    let source_x_expected = pair_values
        .iter()
        .copied()
        .map(|pair| {
            regular_pair_source_x_centered_reference(
                pair,
                source_x_wave_number,
                source_x_harmonic_factor,
                source_x_step,
            )
        })
        .collect::<Vec<_>>();
    let source_x_pair_view = pair_view;
    let source_x_output_view = BufferView::new(9, 1, pair_values.len(), 2)
        .map_err(|error| format!("source-x derivative output view: {error}"))?;
    let source_x_sentinel = BemRegularPairOutF64 {
        g: Complex64::new(-31.0, 29.0),
        h: Complex64::new(-23.0, 19.0),
        ht: Complex64::new(-17.0, 13.0),
        e: Complex64::new(-11.0, 7.0),
    };
    let mut source_x_output = pool
        .try_alloc::<BemRegularPairOutF64>(9)
        .map_err(|error| format!("source-x derivative output allocation: {error}"))?;
    backend
        .try_copy_h2d(&[source_x_sentinel; 9], source_x_output.raw_mut())
        .map_err(|error| format!("source-x derivative output seed: {error}"))?;

    let empty_source_x_pairs = pool
        .try_alloc::<BemRegularPairInF64>(0)
        .map_err(|error| format!("empty source-x pair allocation: {error}"))?;
    let mut empty_source_x_output = pool
        .try_alloc::<BemRegularPairOutF64>(0)
        .map_err(|error| format!("empty source-x output allocation: {error}"))?;
    let empty_source_x_view = BufferView::contiguous(0, 0);
    let empty_source_x_noop = backend
        .try_bem_regular_pair_c64_source_x_derivative_device_view(
            empty_source_x_pairs.raw(),
            empty_source_x_view,
            empty_source_x_output.raw_mut(),
            empty_source_x_view,
            source_x_wave_number,
            source_x_harmonic_factor,
        )
        .is_ok();
    let malformed_source_x_view = BufferView {
        offset: 1,
        len: 1,
        stride: 0,
    };
    let malformed_source_x_error = backend
        .try_bem_regular_pair_c64_source_x_derivative_device_view(
            pairs.raw(),
            malformed_source_x_view,
            source_x_output.raw_mut(),
            BufferView::contiguous(0, 1),
            source_x_wave_number,
            source_x_harmonic_factor,
        )
        .expect_err("malformed source-x derivative view must be rejected before launch");
    let malformed_source_x_typed = matches!(
        malformed_source_x_error,
        BackendError::InvalidArgument {
            operation: "bem_regular_pair_c64_source_x_derivative_device_view",
            ..
        }
    );
    let empty_source_x_output_error = backend
        .try_bem_regular_pair_c64_source_x_derivative_device_view(
            pairs.raw(),
            BufferView::contiguous(0, 1),
            empty_source_x_output.raw_mut(),
            BufferView::contiguous(0, 1),
            source_x_wave_number,
            source_x_harmonic_factor,
        )
        .expect_err("empty source-x output must reject a non-empty view");
    let empty_source_x_typed = matches!(
        empty_source_x_output_error,
        BackendError::InvalidArgument {
            operation: "bem_regular_pair_c64_source_x_derivative_device_view",
            ..
        }
    );
    if !empty_source_x_noop || !malformed_source_x_typed || !empty_source_x_typed {
        return Err(format!(
            "source-x derivative checked admission failed (empty_noop={empty_source_x_noop}, malformed_typed={malformed_source_x_typed}, empty_output_typed={empty_source_x_typed})"
        ));
    }

    let source_x_initial_action = action(
        backend,
        "bem.regular_pair_source_x_derivative.view.c64",
        "Complex64",
        "records_strided(offset=1,stride=2),centered_cpu_dx",
        || {
            backend.try_bem_regular_pair_c64_source_x_derivative_device_view(
                pairs.raw(),
                source_x_pair_view,
                source_x_output.raw_mut(),
                source_x_output_view,
                source_x_wave_number,
                source_x_harmonic_factor,
            )
        },
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("source-x derivative initial synchronization: {error}"))?;
    let mut source_x_initial_backing = vec![BemRegularPairOutF64::default(); 9];
    backend
        .try_copy_d2h(source_x_output.raw(), &mut source_x_initial_backing)
        .map_err(|error| format!("source-x derivative initial readback: {error}"))?;
    let source_x_initial_logical = (0..source_x_output_view.len)
        .map(|logical| {
            source_x_output_view
                .checked_element_index(logical)
                .map(|index| source_x_initial_backing[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("source-x derivative initial output: {error}"))?;
    let source_x_initial_error =
        regular_pair_relative_error(&source_x_initial_logical, &source_x_expected)?;

    let source_x_warm_action = action(
        backend,
        "bem.regular_pair_source_x_derivative.view.c64.warm_replay",
        "Complex64",
        "records_strided(offset=1,stride=2),centered_cpu_dx",
        || {
            backend.try_bem_regular_pair_c64_source_x_derivative_device_view(
                pairs.raw(),
                source_x_pair_view,
                source_x_output.raw_mut(),
                source_x_output_view,
                source_x_wave_number,
                source_x_harmonic_factor,
            )
        },
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("source-x derivative warm synchronization: {error}"))?;
    let mut source_x_warm_backing = vec![BemRegularPairOutF64::default(); 9];
    backend
        .try_copy_d2h(source_x_output.raw(), &mut source_x_warm_backing)
        .map_err(|error| format!("source-x derivative warm readback: {error}"))?;
    let source_x_warm_logical = (0..source_x_output_view.len)
        .map(|logical| {
            source_x_output_view
                .checked_element_index(logical)
                .map(|index| source_x_warm_backing[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("source-x derivative warm output: {error}"))?;
    let source_x_warm_error =
        regular_pair_relative_error(&source_x_warm_logical, &source_x_expected)?;

    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let source_x_failure_error = backend
        .try_bem_regular_pair_c64_source_x_derivative_device_view(
            pairs.raw(),
            source_x_pair_view,
            source_x_output.raw_mut(),
            source_x_output_view,
            source_x_wave_number,
            source_x_harmonic_factor,
        )
        .expect_err("injected source-x derivative launch failure must be typed");
    let source_x_failure_typed = matches!(
        source_x_failure_error,
        BackendError::Launch {
            operation: "cuda_launch",
            ..
        }
    );
    backend
        .recover_after_error()
        .map_err(|error| format!("source-x derivative failure recovery: {error}"))?;
    let mut source_x_failed_backing = vec![BemRegularPairOutF64::default(); 9];
    backend
        .try_copy_d2h(source_x_output.raw(), &mut source_x_failed_backing)
        .map_err(|error| format!("source-x derivative failed-output readback: {error}"))?;
    let source_x_failure_preserved = source_x_failed_backing
        .iter()
        .zip(&source_x_warm_backing)
        .all(|(failed, warm)| {
            failed.g == warm.g && failed.h == warm.h && failed.ht == warm.ht && failed.e == warm.e
        });
    if !source_x_failure_typed || !source_x_failure_preserved {
        return Err(format!(
            "source-x derivative failure contract failed (typed={source_x_failure_typed}, output_preserved={source_x_failure_preserved})"
        ));
    }

    let source_x_retry_action = action(
        backend,
        "bem.regular_pair_source_x_derivative.view.c64.retry",
        "Complex64",
        "records_strided(offset=1,stride=2),centered_cpu_dx",
        || {
            backend.try_bem_regular_pair_c64_source_x_derivative_device_view(
                pairs.raw(),
                source_x_pair_view,
                source_x_output.raw_mut(),
                source_x_output_view,
                source_x_wave_number,
                source_x_harmonic_factor,
            )
        },
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("source-x derivative retry synchronization: {error}"))?;
    let mut source_x_retry_backing = vec![BemRegularPairOutF64::default(); 9];
    backend
        .try_copy_d2h(source_x_output.raw(), &mut source_x_retry_backing)
        .map_err(|error| format!("source-x derivative retry readback: {error}"))?;
    let source_x_retry_logical = (0..source_x_output_view.len)
        .map(|logical| {
            source_x_output_view
                .checked_element_index(logical)
                .map(|index| source_x_retry_backing[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("source-x derivative retry output: {error}"))?;
    let source_x_retry_error =
        regular_pair_relative_error(&source_x_retry_logical, &source_x_expected)?;
    let source_x_residency = source_x_initial_action
        .get("residency")
        .cloned()
        .ok_or_else(|| {
            "source-x derivative initial action missing residency evidence".to_string()
        })?;
    rows.push(json!({
        "phase": "bem.regular_pair_source_x_derivative.view.c64",
        "scalar": "Complex64",
        "layout": "records_strided(offset=1,stride=2),centered_cpu_dx",
        "status": "native",
        "residency": source_x_residency,
        "relative_error": source_x_initial_error,
        "warm_relative_error": source_x_warm_error,
        "retry_relative_error": source_x_retry_error,
        "centered_reference_relative_error": source_x_initial_error,
        "tolerance": 1.0e-7,
        "typed_rejections": {
            "malformed_view": malformed_source_x_typed,
            "empty_output": empty_source_x_typed,
            "empty_view_noop": empty_source_x_noop,
        },
        "failure_recovery": {
            "typed": source_x_failure_typed,
            "output_preserved": source_x_failure_preserved,
        },
        "warm_replay": {
            "initial": source_x_initial_action,
            "warm": source_x_warm_action,
            "retry": source_x_retry_action,
        },
    }));

    // The source-collocation y derivative uses the same checked descriptor
    // contract as x, but is a separate production CUDA kernel. Use an odd
    // three-record view so this cell also exercises the non-block-aligned
    // tail rather than only the contiguous whole-buffer wrapper.
    let source_y_wave_number = 1.0;
    let source_y_harmonic_factor = -1.0;
    let source_y_step = 1.0e-5;
    let source_y_pair_view = BufferView::new(9, 1, 3, 2)
        .map_err(|error| format!("source-y derivative pair view: {error}"))?;
    let source_y_output_view = source_y_pair_view;
    let source_y_expected = pair_values[..source_y_pair_view.len]
        .iter()
        .copied()
        .map(|pair| {
            regular_pair_source_y_centered_reference(
                pair,
                source_y_wave_number,
                source_y_harmonic_factor,
                source_y_step,
            )
        })
        .collect::<Vec<_>>();
    let source_y_sentinel = BemRegularPairOutF64 {
        g: Complex64::new(37.0, -31.0),
        h: Complex64::new(29.0, -23.0),
        ht: Complex64::new(19.0, -17.0),
        e: Complex64::new(13.0, -11.0),
    };
    let mut source_y_output = pool
        .try_alloc::<BemRegularPairOutF64>(9)
        .map_err(|error| format!("source-y derivative output allocation: {error}"))?;
    backend
        .try_copy_h2d(&[source_y_sentinel; 9], source_y_output.raw_mut())
        .map_err(|error| format!("source-y derivative output seed: {error}"))?;

    let empty_source_y_pairs = pool
        .try_alloc::<BemRegularPairInF64>(0)
        .map_err(|error| format!("empty source-y pair allocation: {error}"))?;
    let mut empty_source_y_output = pool
        .try_alloc::<BemRegularPairOutF64>(0)
        .map_err(|error| format!("empty source-y output allocation: {error}"))?;
    let empty_source_y_view = BufferView::contiguous(0, 0);
    let empty_source_y_noop = backend
        .try_bem_regular_pair_c64_source_y_derivative_device_view(
            empty_source_y_pairs.raw(),
            empty_source_y_view,
            empty_source_y_output.raw_mut(),
            empty_source_y_view,
            source_y_wave_number,
            source_y_harmonic_factor,
        )
        .is_ok();
    let malformed_source_y_view = BufferView {
        offset: 1,
        len: 1,
        stride: 0,
    };
    let malformed_source_y_error = backend
        .try_bem_regular_pair_c64_source_y_derivative_device_view(
            pairs.raw(),
            malformed_source_y_view,
            source_y_output.raw_mut(),
            BufferView::contiguous(0, 1),
            source_y_wave_number,
            source_y_harmonic_factor,
        )
        .expect_err("malformed source-y derivative view must be rejected before launch");
    let malformed_source_y_typed = matches!(
        malformed_source_y_error,
        BackendError::InvalidArgument {
            operation: "bem_regular_pair_c64_source_y_derivative_device_view",
            ..
        }
    );
    let empty_source_y_output_error = backend
        .try_bem_regular_pair_c64_source_y_derivative_device_view(
            pairs.raw(),
            BufferView::contiguous(0, 1),
            empty_source_y_output.raw_mut(),
            BufferView::contiguous(0, 1),
            source_y_wave_number,
            source_y_harmonic_factor,
        )
        .expect_err("empty source-y output must reject a non-empty view");
    let empty_source_y_typed = matches!(
        empty_source_y_output_error,
        BackendError::InvalidArgument {
            operation: "bem_regular_pair_c64_source_y_derivative_device_view",
            ..
        }
    );
    if !empty_source_y_noop || !malformed_source_y_typed || !empty_source_y_typed {
        return Err(format!(
            "source-y derivative checked admission failed (empty_noop={empty_source_y_noop}, malformed_typed={malformed_source_y_typed}, empty_output_typed={empty_source_y_typed})"
        ));
    }

    let source_y_initial_action = action(
        backend,
        "bem.regular_pair_source_y_derivative.view.c64",
        "Complex64",
        "records_strided(offset=1,stride=2),tail=3,centered_cpu_dy",
        || {
            backend.try_bem_regular_pair_c64_source_y_derivative_device_view(
                pairs.raw(),
                source_y_pair_view,
                source_y_output.raw_mut(),
                source_y_output_view,
                source_y_wave_number,
                source_y_harmonic_factor,
            )
        },
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("source-y derivative initial synchronization: {error}"))?;
    let mut source_y_initial_backing = vec![BemRegularPairOutF64::default(); 9];
    backend
        .try_copy_d2h(source_y_output.raw(), &mut source_y_initial_backing)
        .map_err(|error| format!("source-y derivative initial readback: {error}"))?;
    let source_y_initial_logical = (0..source_y_output_view.len)
        .map(|logical| {
            source_y_output_view
                .checked_element_index(logical)
                .map(|index| source_y_initial_backing[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("source-y derivative initial output: {error}"))?;
    let source_y_initial_error =
        regular_pair_relative_error(&source_y_initial_logical, &source_y_expected)?;

    let source_y_warm_action = action(
        backend,
        "bem.regular_pair_source_y_derivative.view.c64.warm_replay",
        "Complex64",
        "records_strided(offset=1,stride=2),tail=3,centered_cpu_dy",
        || {
            backend.try_bem_regular_pair_c64_source_y_derivative_device_view(
                pairs.raw(),
                source_y_pair_view,
                source_y_output.raw_mut(),
                source_y_output_view,
                source_y_wave_number,
                source_y_harmonic_factor,
            )
        },
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("source-y derivative warm synchronization: {error}"))?;
    let mut source_y_warm_backing = vec![BemRegularPairOutF64::default(); 9];
    backend
        .try_copy_d2h(source_y_output.raw(), &mut source_y_warm_backing)
        .map_err(|error| format!("source-y derivative warm readback: {error}"))?;
    let source_y_warm_logical = (0..source_y_output_view.len)
        .map(|logical| {
            source_y_output_view
                .checked_element_index(logical)
                .map(|index| source_y_warm_backing[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("source-y derivative warm output: {error}"))?;
    let source_y_warm_error =
        regular_pair_relative_error(&source_y_warm_logical, &source_y_expected)?;

    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let source_y_failure_error = backend
        .try_bem_regular_pair_c64_source_y_derivative_device_view(
            pairs.raw(),
            source_y_pair_view,
            source_y_output.raw_mut(),
            source_y_output_view,
            source_y_wave_number,
            source_y_harmonic_factor,
        )
        .expect_err("injected source-y derivative launch failure must be typed");
    let source_y_failure_typed = matches!(
        source_y_failure_error,
        BackendError::Launch {
            operation: "cuda_launch",
            ..
        }
    );
    backend
        .recover_after_error()
        .map_err(|error| format!("source-y derivative failure recovery: {error}"))?;
    let mut source_y_failed_backing = vec![BemRegularPairOutF64::default(); 9];
    backend
        .try_copy_d2h(source_y_output.raw(), &mut source_y_failed_backing)
        .map_err(|error| format!("source-y derivative failed-output readback: {error}"))?;
    let source_y_failure_preserved = source_y_failed_backing
        .iter()
        .zip(&source_y_warm_backing)
        .all(|(failed, warm)| {
            failed.g == warm.g && failed.h == warm.h && failed.ht == warm.ht && failed.e == warm.e
        });
    if !source_y_failure_typed || !source_y_failure_preserved {
        return Err(format!(
            "source-y derivative failure contract failed (typed={source_y_failure_typed}, output_preserved={source_y_failure_preserved})"
        ));
    }

    let source_y_retry_action = action(
        backend,
        "bem.regular_pair_source_y_derivative.view.c64.retry",
        "Complex64",
        "records_strided(offset=1,stride=2),tail=3,centered_cpu_dy",
        || {
            backend.try_bem_regular_pair_c64_source_y_derivative_device_view(
                pairs.raw(),
                source_y_pair_view,
                source_y_output.raw_mut(),
                source_y_output_view,
                source_y_wave_number,
                source_y_harmonic_factor,
            )
        },
    )?;
    backend
        .try_synchronize()
        .map_err(|error| format!("source-y derivative retry synchronization: {error}"))?;
    let mut source_y_retry_backing = vec![BemRegularPairOutF64::default(); 9];
    backend
        .try_copy_d2h(source_y_output.raw(), &mut source_y_retry_backing)
        .map_err(|error| format!("source-y derivative retry readback: {error}"))?;
    let source_y_retry_logical = (0..source_y_output_view.len)
        .map(|logical| {
            source_y_output_view
                .checked_element_index(logical)
                .map(|index| source_y_retry_backing[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("source-y derivative retry output: {error}"))?;
    let source_y_retry_error =
        regular_pair_relative_error(&source_y_retry_logical, &source_y_expected)?;
    let source_y_residency = source_y_initial_action
        .get("residency")
        .cloned()
        .ok_or_else(|| {
            "source-y derivative initial action missing residency evidence".to_string()
        })?;
    rows.push(json!({
        "phase": "bem.regular_pair_source_y_derivative.view.c64",
        "scalar": "Complex64",
        "layout": "records_strided(offset=1,stride=2),tail=3,centered_cpu_dy",
        "status": "native",
        "residency": source_y_residency,
        "relative_error": source_y_initial_error,
        "warm_relative_error": source_y_warm_error,
        "retry_relative_error": source_y_retry_error,
        "centered_reference_relative_error": source_y_initial_error,
        "tolerance": 1.0e-7,
        "tail_case": { "logical_len": source_y_pair_view.len, "non_block_aligned": true },
        "typed_rejections": {
            "malformed_view": malformed_source_y_typed,
            "empty_output": empty_source_y_typed,
            "empty_view_noop": empty_source_y_noop,
        },
        "failure_recovery": {
            "typed": source_y_failure_typed,
            "output_preserved": source_y_failure_preserved,
        },
        "warm_replay": {
            "initial": source_y_initial_action,
            "warm": source_y_warm_action,
            "retry": source_y_retry_action,
        },
    }));

    backend
        .try_synchronize()
        .map_err(|error| format!("specialized CUDA synchronization: {error}"))?;
    let stats = backend.runtime_stats();
    let report = json!({
        "schema": "sonium-compute-cuda-specialized-evidence/v1",
        "case_id": CASE_ID,
        "status": "passed",
        "reference": "checked native CUDA kernels with deterministic Complex64 fixtures and independent scalar CPU regular-pair, quadrature, field, centered/analytic wavenumber derivative, and centered source-coordinate derivative references",
        "dimension": "Complex64 specialised FMM phase/gather/scatter/dense translations plus BEM regular, quadrature, field, wavenumber-derivative, and source-coordinate derivative phases",
        "rows": rows,
        "runtime": runtime_json(&stats),
        "hardware": cuda_hardware_provenance(),
        "counts": { "executed": 13, "passed": 13, "failed": 0, "unsupported": 0 },
    });
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("serialize specialized CUDA report: {error}"))?;
    let (artifact, checksum) = write_qa_artifact("c2-specialized-residency.json", &bytes)?;
    let result = QaResult::passed(CASE_ID)
        .requested_backend("cuda")
        .actual_backend("cuda")
        .backend_passed("cuda")
        .reference("checked native CUDA specialised BEM/FMM views, independent CPU parity, and resident action phases")
        .reference_checksum("analytic/deterministic-fixture:c2-specialized-cuda-v1")
        .input_checksum("analytic/deterministic-fixture:c2-specialized-cuda-v1")
        .dimension("Complex64 FMM/BEM specialised operations")
        .formulation("native CUDA specialised operations over checked offset/stride descriptors")
        .tolerance("all thirteen specialised cells execute native CUDA work with zero measured transfers or allocations during action intervals")
        .metric("specialized_cell_count", 13.0)
        .metric("specialized_native_cell_count", 13.0)
        .metric("specialized_unsupported_cell_count", 0.0)
        .metric("specialized_h2d_action_count", 0.0)
        .metric("specialized_d2h_action_count", 0.0)
        .metric("specialized_d2d_action_count", 0.0)
        .minimum("specialized_cell_count", 13.0)
        .minimum("specialized_native_cell_count", 13.0)
        .maximum("specialized_unsupported_cell_count", 0.0)
        .maximum("specialized_h2d_action_count", 0.0)
        .maximum("specialized_d2h_action_count", 0.0)
        .maximum("specialized_d2d_action_count", 0.0)
        .artifact_with_provenance(
            artifact,
            "crates/sonium-qa/bin/compute_specialized_cuda_matrix.rs",
            "real-SM120 specialised BEM/FMM operation and residency matrix",
            "v1",
            checksum,
        )
        .runtime_evidence(cuda_runtime_evidence(
            stats,
            QaExecutionCounts {
                executed: 13,
                passed: 13,
                ..QaExecutionCounts::default()
            },
        ))
        .hardware_provenance(cuda_hardware_provenance());
    Ok(result)
}

fn failed(error: String, actual_backend: &str, stage: QaFailureStage) -> QaResult {
    QaResult::failed(CASE_ID, error.clone())
        .requested_backend("cuda")
        .actual_backend(actual_backend)
        .backend_failed("cuda", error)
        .failure_stage(stage)
        .hardware_provenance(cuda_hardware_provenance())
}

fn main() {
    let started = Instant::now();
    let result = match required_cuda_backend() {
        Ok(backend) => match run(backend) {
            Ok(result) => result,
            Err(error) => failed(error, "cuda", QaFailureStage::Execution),
        },
        Err(error) => failed(
            format!("required CUDA backend unavailable: {error}"),
            "unavailable",
            QaFailureStage::Initialization,
        ),
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);

    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("specialized CUDA QA result serialization failed: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
    if matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
