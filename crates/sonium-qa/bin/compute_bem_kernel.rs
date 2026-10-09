//! CPU BEM-kernel QA against an analytical point-source value and the
//! canonical scalar regular-pair reference.

use num_complex::Complex64;
use sonium_compute::bem_cpu::{RegularPairInF64, regular_pair_batch, regular_pair_scalar};
use sonium_compute::{Backend, BemRegularPairIn, BemRegularPairInF64, CpuBackend};
use sonium_qa::QaResult;
use sonium_wave::special::try_greens_function_3d;
use std::f64::consts::PI;

const CASE_ID: &str = "compute.bem-kernel.v1";
const POINTWISE_TOLERANCE: f64 = 2.0e-14;
const BATCH_SCALAR_TOLERANCE: f64 = 2.0e-12;
const CHECKED_BATCH_TOLERANCE: f64 = 5.0e-5;

#[derive(Debug, Default)]
struct Metrics {
    pointwise_relative_error: f64,
    batch_scalar_relative_error: f64,
    checked_batch_relative_error: f64,
    checked_c64_batch_relative_error: f64,
    checked_nonfinite_rejection: bool,
    pair_count: usize,
}

fn relative_error(actual: Complex64, expected: Complex64) -> f64 {
    (actual - expected).norm() / expected.norm().max(1.0e-30)
}

fn run() -> Result<Metrics, String> {
    let source = [0.15, -0.25, 0.35];
    let field = [1.05, 0.45, -0.55];
    let delta = [
        field[0] - source[0],
        field[1] - source[1],
        field[2] - source[2],
    ];
    let radius = delta.iter().map(|value| value * value).sum::<f64>().sqrt();
    let wave_number = 3.7;
    let analytical = Complex64::from_polar(1.0 / (4.0 * PI * radius), wave_number * radius);
    let sonium_wave = try_greens_function_3d(radius, wave_number)
        .map_err(|error| format!("sonium-wave Green function failed: {error}"))?;
    let pointwise_relative_error = relative_error(sonium_wave, analytical);
    if pointwise_relative_error > POINTWISE_TOLERANCE {
        return Err(format!(
            "point-source Green function relative error {pointwise_relative_error:.3e} exceeds {POINTWISE_TOLERANCE:.3e}"
        ));
    }

    let pairs = vec![
        RegularPairInF64 {
            source_point: [0.0, 0.0, 0.0],
            source_normal: [0.0, 0.0, 1.0],
            v0: [0.7, -0.2, 0.15],
            v1: [1.1, 0.25, 0.2],
            v2: [0.55, 0.45, 0.8],
        },
        RegularPairInF64 {
            source_point: [-0.35, 0.2, -0.1],
            source_normal: [0.0, 1.0, 0.0],
            v0: [0.25, -0.4, 0.6],
            v1: [0.9, -0.1, 0.45],
            v2: [0.4, 0.35, 1.1],
        },
        RegularPairInF64 {
            source_point: [0.2, -0.45, 0.3],
            source_normal: [1.0, 0.0, 0.0],
            v0: [-0.3, 0.1, 0.2],
            v1: [0.15, 0.55, 0.25],
            v2: [-0.15, 0.2, 0.85],
        },
    ];
    let batch = regular_pair_batch(&pairs, wave_number, 1.0);
    let mut batch_scalar_relative_error: f64 = 0.0;
    for (actual, pair) in batch.iter().zip(&pairs) {
        let expected = regular_pair_scalar(pair, wave_number, 1.0);
        for (actual, expected) in [
            (actual.0, expected.0),
            (actual.1, expected.1),
            (actual.2, expected.2),
            (actual.3, expected.3),
        ] {
            batch_scalar_relative_error =
                batch_scalar_relative_error.max(relative_error(actual, expected));
        }
    }
    let backend = CpuBackend::new();
    let abi_pairs: Vec<_> = pairs
        .iter()
        .map(|pair| {
            BemRegularPairIn::from_triangle(
                pair.source_point.map(|value| value as f32),
                pair.source_normal.map(|value| value as f32),
                pair.v0.map(|value| value as f32),
                pair.v1.map(|value| value as f32),
                pair.v2.map(|value| value as f32),
            )
        })
        .collect();
    let checked = backend
        .try_bem_regular_pair_c32_batch(&abi_pairs, wave_number as f32, 1.0)
        .map_err(|error| format!("checked CPU BEM batch failed: {error}"))?;
    let mut checked_batch_relative_error = 0.0_f64;
    for (actual, expected) in checked.iter().zip(&batch) {
        let values = [
            (
                Complex64::new(f64::from(actual.g.re), f64::from(actual.g.im)),
                expected.0,
            ),
            (
                Complex64::new(f64::from(actual.h.re), f64::from(actual.h.im)),
                expected.1,
            ),
            (
                Complex64::new(f64::from(actual.ht.re), f64::from(actual.ht.im)),
                expected.2,
            ),
            (
                Complex64::new(f64::from(actual.e.re), f64::from(actual.e.im)),
                expected.3,
            ),
        ];
        for (actual, expected) in values {
            checked_batch_relative_error =
                checked_batch_relative_error.max(relative_error(actual, expected));
        }
    }
    let c64_pairs: Vec<BemRegularPairInF64> = pairs
        .iter()
        .map(|pair| {
            BemRegularPairInF64::from_triangle(
                pair.source_point,
                pair.source_normal,
                pair.v0,
                pair.v1,
                pair.v2,
            )
        })
        .collect();
    let checked_c64 = backend
        .try_bem_regular_pair_c64_batch(&c64_pairs, wave_number, 1.0)
        .map_err(|error| format!("checked CPU Complex64 BEM batch failed: {error}"))?;
    let mut checked_c64_batch_relative_error = 0.0_f64;
    for (actual, expected) in checked_c64.iter().zip(&batch) {
        for (actual, expected) in [
            (actual.g, expected.0),
            (actual.h, expected.1),
            (actual.ht, expected.2),
            (actual.e, expected.3),
        ] {
            checked_c64_batch_relative_error =
                checked_c64_batch_relative_error.max(relative_error(actual, expected));
        }
    }
    let mut invalid_pair = abi_pairs[0];
    invalid_pair.v1[0] = f32::NAN;
    let checked_nonfinite_rejection = backend
        .try_bem_regular_pair_c32_batch(&[invalid_pair], wave_number as f32, 1.0)
        .is_err();

    if batch_scalar_relative_error > BATCH_SCALAR_TOLERANCE {
        return Err(format!(
            "regular-pair batch/scalar relative error {batch_scalar_relative_error:.3e} exceeds {BATCH_SCALAR_TOLERANCE:.3e}"
        ));
    }
    if checked_c64_batch_relative_error > BATCH_SCALAR_TOLERANCE {
        return Err(format!(
            "checked Complex64 batch/scalar relative error {checked_c64_batch_relative_error:.3e} exceeds {BATCH_SCALAR_TOLERANCE:.3e}"
        ));
    }

    Ok(Metrics {
        pointwise_relative_error,
        batch_scalar_relative_error,
        checked_batch_relative_error,
        checked_c64_batch_relative_error,
        checked_nonfinite_rejection,
        pair_count: pairs.len(),
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("pointwise_relative_error", metrics.pointwise_relative_error)
                .metric(
                    "batch_scalar_relative_error",
                    metrics.batch_scalar_relative_error,
                )
                .metric(
                    "checked_batch_relative_error",
                    metrics.checked_batch_relative_error,
                )
                .metric(
                    "checked_c64_batch_relative_error",
                    metrics.checked_c64_batch_relative_error,
                )
                .metric(
                    "checked_nonfinite_rejection",
                    metrics.checked_nonfinite_rejection as u8 as f64,
                )
                .metric("pair_count", metrics.pair_count as f64)
                .maximum("pointwise_relative_error", POINTWISE_TOLERANCE)
                .maximum("batch_scalar_relative_error", BATCH_SCALAR_TOLERANCE)
                .maximum("checked_batch_relative_error", CHECKED_BATCH_TOLERANCE)
                .maximum(
                    "checked_c64_batch_relative_error",
                    BATCH_SCALAR_TOLERANCE,
                )
                .minimum("checked_nonfinite_rejection", 1.0)
                .input_checksum("analytic/deterministic-fixture:compute-bem-kernel-v1")
                .reference("sonium-wave 3-D Green function and regular_pair_scalar")
                .reference_checksum("analytic/deterministic-fixture:greens-function-3d-v1")
                .dimension("3-D regular Helmholtz pair kernel")
                .formulation("CPU regular-pair BEM kernel")
                .tolerance(
                    "point-source 2e-14; batch/scalar 2e-12; checked f32 ABI 5e-5; checked f64 ABI 2e-12",
                )
                .seed("compute-bem-kernel:source-field-pairs:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("BEM-kernel QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("BEM-kernel QA failure result must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
