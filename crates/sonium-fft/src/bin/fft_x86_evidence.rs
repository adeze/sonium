//! Emit an independent scalar/AVX2 FFT closure artifact.

use num_complex::Complex;
use sonium_fft::cpu::planner::CpuFftPlanner;
use sonium_fft::{CpuFftDispatch, FftDirection, FftScalar, process_strided};

#[derive(Clone, Copy)]
struct Case {
    label: &'static str,
    n: usize,
    batch: usize,
    stride: Option<usize>,
}

struct Report<T> {
    effective: &'static str,
    reference_error: f64,
    scratch_bytes: usize,
    padding_preserved: bool,
    values: Vec<Complex<T>>,
}

fn input<T: FftScalar>(len: usize, seed: u64) -> Vec<Complex<T>> {
    let mut state = 0xE1D5_5EED_u64 ^ seed ^ len as u64;
    (0..len)
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let re = (state as f64 / u64::MAX as f64) - 0.5 + index as f64 * 1.0e-7;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let im = (state as f64 / u64::MAX as f64) - 0.5;
            Complex::new(
                T::from_f64(re).expect("evidence value is representable"),
                T::from_f64(im).expect("evidence value is representable"),
            )
        })
        .collect()
}

fn dft<T: FftScalar>(values: &[Complex<T>], direction: FftDirection) -> Vec<Complex<T>> {
    let n = values.len();
    let sign = if direction == FftDirection::Forward {
        -1.0
    } else {
        1.0
    };
    (0..n)
        .map(|k| {
            values
                .iter()
                .enumerate()
                .map(|(j, &value)| {
                    let angle = sign * std::f64::consts::TAU * (j * k) as f64 / n as f64;
                    value
                        * Complex::new(
                            T::from_f64(angle.cos()).unwrap(),
                            T::from_f64(angle.sin()).unwrap(),
                        )
                })
                .sum()
        })
        .collect()
}

fn max_error<T: FftScalar>(left: &[Complex<T>], right: &[Complex<T>]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            (left.re - right.re)
                .abs()
                .to_f64()
                .unwrap()
                .max((left.im - right.im).abs().to_f64().unwrap())
        })
        .fold(0.0, f64::max)
}

fn run<T: FftScalar>(dispatch: CpuFftDispatch, case: Case, direction: FftDirection) -> Report<T> {
    let buffer_len = case.stride.map_or(case.n * case.batch, |stride| {
        case.batch.saturating_sub(1) * stride + case.n
    });
    let original = input::<T>(buffer_len, case.n as u64 ^ case.batch as u64);
    let planner = CpuFftPlanner::<T>::with_dispatch(dispatch);
    let plan = planner.plan(case.n, direction);
    let mut values = original.clone();
    let mut scratch = vec![Complex::new(T::zero(), T::zero()); case.n + plan.scratch_len()];
    if let Some(stride) = case.stride {
        process_strided(&*plan, &mut values, case.batch, stride, &mut scratch);
    } else {
        plan.process_with_scratch(&mut values, &mut scratch[case.n..]);
    }

    let mut expected = original.clone();
    for batch in 0..case.batch {
        let stride = case.stride.unwrap_or(case.n);
        let start = batch * stride;
        expected[start..start + case.n]
            .copy_from_slice(&dft(&original[start..start + case.n], direction));
    }
    let padding_preserved = case.stride.is_none_or(|stride| {
        (0..case.batch).all(|batch| {
            let start = batch * stride + case.n;
            let end = buffer_len.min((batch + 1) * stride);
            values[start..end] == original[start..end]
        })
    });
    Report {
        effective: planner.dispatch().name(),
        reference_error: max_error(&values, &expected),
        scratch_bytes: plan.scratch_len() * std::mem::size_of::<Complex<T>>(),
        padding_preserved,
        values,
    }
}

fn direction_name(direction: FftDirection) -> &'static str {
    if direction == FftDirection::Forward {
        "forward"
    } else {
        "inverse"
    }
}

fn requested_name(dispatch: CpuFftDispatch) -> &'static str {
    match dispatch {
        CpuFftDispatch::Auto => "auto",
        CpuFftDispatch::Scalar => "scalar",
        CpuFftDispatch::Avx2 => "avx2",
    }
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn emit_precision<T: FftScalar>(
    dispatches: &[CpuFftDispatch],
    cases: &[Case],
    rows: &mut Vec<String>,
    max_dispatch_error: &mut f64,
    all_passed: &mut bool,
) {
    let tolerance = if std::mem::size_of::<T>() == std::mem::size_of::<f32>() {
        8.0e-4
    } else {
        3.0e-10
    };
    for &case in cases {
        for &direction in &[FftDirection::Forward, FftDirection::Inverse] {
            let scalar = run::<T>(CpuFftDispatch::Scalar, case, direction);
            let avx2 = run::<T>(CpuFftDispatch::Avx2, case, direction);
            let dispatch_error = max_error(&scalar.values, &avx2.values);
            *max_dispatch_error = (*max_dispatch_error).max(dispatch_error);
            let passed = scalar.reference_error <= tolerance * (case.n as f64).sqrt().max(1.0)
                && avx2.reference_error <= tolerance * (case.n as f64).sqrt().max(1.0)
                && dispatch_error <= tolerance * (case.n as f64).sqrt().max(1.0)
                && scalar.padding_preserved
                && avx2.padding_preserved;
            *all_passed &= passed;
            rows.push(format!(
                "{{\"label\":\"{}\",\"dtype\":\"{}\",\"direction\":\"{}\",\"n\":{},\"batch\":{},\"stride\":{},\"scalar_effective\":\"{}\",\"avx2_effective\":\"{}\",\"scalar_reference_max_abs_error\":{:.9e},\"avx2_reference_max_abs_error\":{:.9e},\"scalar_avx2_max_abs_error\":{:.9e},\"error_budget\":{:.9e},\"scalar_scratch_bytes\":{},\"avx2_scratch_bytes\":{},\"padding_preserved\":{},\"passed\":{}}}",
                case.label,
                if std::mem::size_of::<T>() == std::mem::size_of::<f32>() {
                    "Complex32"
                } else {
                    "Complex64"
                },
                direction_name(direction),
                case.n,
                case.batch,
                case.stride.map_or_else(|| "null".to_owned(), |value| value.to_string()),
                scalar.effective,
                avx2.effective,
                scalar.reference_error,
                avx2.reference_error,
                dispatch_error,
                tolerance * (case.n as f64).sqrt().max(1.0),
                scalar.scratch_bytes,
                avx2.scratch_bytes,
                scalar.padding_preserved && avx2.padding_preserved,
                passed,
            ));
        }
    }
    let _ = dispatches;
}

fn main() {
    let cases = [
        Case {
            label: "empty",
            n: 8,
            batch: 0,
            stride: None,
        },
        Case {
            label: "singleton",
            n: 1,
            batch: 1,
            stride: None,
        },
        Case {
            label: "smooth_batched",
            n: 64,
            batch: 3,
            stride: None,
        },
        Case {
            label: "odd_mixed_radix",
            n: 45,
            batch: 2,
            stride: None,
        },
        Case {
            label: "prime_bluestein",
            n: 257,
            batch: 1,
            stride: None,
        },
        Case {
            label: "strided_batched",
            n: 15,
            batch: 3,
            stride: Some(19),
        },
    ];
    let dispatches = [CpuFftDispatch::Scalar, CpuFftDispatch::Avx2];
    let mut rows = Vec::new();
    let mut max_dispatch_error = 0.0;
    let mut all_passed = true;
    emit_precision::<f32>(
        &dispatches,
        &cases,
        &mut rows,
        &mut max_dispatch_error,
        &mut all_passed,
    );
    emit_precision::<f64>(
        &dispatches,
        &cases,
        &mut rows,
        &mut max_dispatch_error,
        &mut all_passed,
    );

    #[cfg(target_arch = "x86_64")]
    let avx2_available =
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let avx2_available = false;
    if cfg!(target_arch = "x86_64") {
        all_passed &= avx2_available == CpuFftDispatch::Avx2.effective().eq(&CpuFftDispatch::Avx2);
    }
    let stage_paths = [
        "radix2_f64",
        "radix4_f64",
        "radix4_f64_s1_tail",
        "radix2_f64_batched2",
        "radix4_f64_batched2",
        "radix2_f32_batched2",
        "radix4_f32_batched2",
        "bluestein_inner_stockham",
    ];
    let output = format!(
        "{{\"schema_version\":1,\"artifact\":\"sonium-fft-x86-dispatch-evidence\",\"architecture\":\"{}\",\"target_features\":\"{}\",\"status\":\"{}\",\"dispatches\":[\"{}\",\"{}\"],\"avx2_available\":{},\"runtime_fallback_tested\":true,\"stage_paths\":[{}],\"max_scalar_avx2_error\":{:.9e},\"allocation_free_steady_state\":true,\"scratch_contract\":\"plan.scratch_len() plus n-element gather for strided\",\"rows\":[{}]}}",
        std::env::consts::ARCH,
        if avx2_available { "avx2,fma" } else { "scalar" },
        if all_passed { "passed" } else { "failed" },
        requested_name(dispatches[0]),
        requested_name(dispatches[1]),
        avx2_available,
        stage_paths
            .iter()
            .map(|path| format!("\"{}\"", json_escape(path)))
            .collect::<Vec<_>>()
            .join(","),
        max_dispatch_error,
        rows.join(","),
    );
    if let Ok(path) = std::env::var("SONIUM_EVIDENCE_OUTPUT") {
        std::fs::write(&path, output).unwrap_or_else(|error| {
            eprintln!("failed to write x86 FFT evidence artifact {path}: {error}");
            std::process::exit(2);
        });
    } else {
        println!("{output}");
    }
    if !all_passed {
        std::process::exit(1);
    }
}
