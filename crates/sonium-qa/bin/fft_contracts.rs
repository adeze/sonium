//! CPU FFT family contract QA with independent scalar references.

use num_complex::Complex64;
use sonium_fft::{
    Fft2dPlan, FftDirection, FftPlanner, LinearConvolutionPlan, StftConfig, StftPlan,
    WindowFunction,
};
use sonium_qa::QaResult;
use std::f64::consts::TAU;
use std::time::Instant;

const CASE_ID: &str = "fft.family-contracts.v1";
const TOLERANCE: f64 = 2.0e-9;

fn input(seed: f64, length: usize) -> Vec<Complex64> {
    (0..length)
        .map(|index| {
            let phase = seed + index as f64 * 0.173;
            Complex64::new(phase.sin() * 0.7, (phase * 0.61).cos() * 0.4)
        })
        .collect()
}

fn inner_product(lhs: &[Complex64], rhs: &[Complex64]) -> Complex64 {
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| left.conj() * right)
        .sum()
}

fn max_relative_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(got, want)| (got - want).norm() / want.norm().max(1.0))
        .fold(0.0, f64::max)
}

fn direct_linear_convolution(lhs: &[Complex64], rhs: &[Complex64]) -> Vec<Complex64> {
    let mut result = vec![Complex64::new(0.0, 0.0); lhs.len() + rhs.len() - 1];
    for (left_index, &left) in lhs.iter().enumerate() {
        for (right_index, &right) in rhs.iter().enumerate() {
            result[left_index + right_index] += left * right;
        }
    }
    result
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let mut transform_cases = 0usize;
    let mut max_round_trip_error: f64 = 0.0;
    let mut max_parseval_error: f64 = 0.0;
    let mut max_adjoint_error: f64 = 0.0;
    for &length in &[3usize, 8, 15, 31] {
        let planner = FftPlanner::<f64>::new();
        let forward = planner.plan(length, FftDirection::Forward);
        let inverse = planner.plan(length, FftDirection::Inverse);
        let x = input(0.31 + length as f64, length);
        let y = input(0.77 + length as f64, length);
        let mut transformed = x.clone();
        forward.process(&mut transformed);
        let mut recovered = transformed.clone();
        inverse.process(&mut recovered);
        for (got, want) in recovered.iter().zip(&x) {
            max_round_trip_error = max_round_trip_error.max((got / length as f64 - want).norm());
        }
        let input_energy: f64 = x.iter().map(|value| value.norm_sqr()).sum();
        let output_energy: f64 = transformed.iter().map(|value| value.norm_sqr()).sum();
        max_parseval_error = max_parseval_error
            .max((output_energy / length as f64 - input_energy).abs() / input_energy.max(1.0));
        let mut fy = y.clone();
        inverse.process(&mut fy);
        let lhs = inner_product(&transformed, &y);
        let rhs = inner_product(&x, &fy);
        max_adjoint_error = max_adjoint_error.max((lhs - rhs).norm() / lhs.norm().max(1.0));
        transform_cases += 1;
    }

    let lhs = input(0.19, 7);
    let rhs = input(0.41, 5);
    let convolution = LinearConvolutionPlan::<f64>::try_new(lhs.len(), rhs.len())
        .map_err(|error| format!("linear convolution plan: {error:?}"))?
        .process(&lhs, &rhs);
    let reference = direct_linear_convolution(&lhs, &rhs);
    let convolution_error = max_relative_error(&convolution, &reference);

    let rows = 5;
    let cols = 7;
    let two_d = Fft2dPlan::<f64>::new(rows, cols, FftDirection::Forward);
    let mut field = input(0.53, rows * cols);
    let mut scratch = vec![Complex64::new(0.0, 0.0); two_d.scratch_len()];
    two_d
        .try_process_with_scratch(&mut field, &mut scratch)
        .map_err(|error| format!("2-D FFT execution: {error:?}"))?;
    let nd_energy: f64 = field.iter().map(|value| value.norm_sqr()).sum();
    let nd_input_energy: f64 = input(0.53, rows * cols)
        .iter()
        .map(|value| value.norm_sqr())
        .sum();
    let nd_error =
        (nd_energy / (rows * cols) as f64 - nd_input_energy).abs() / nd_input_energy.max(1.0);

    let stft = StftPlan::<f64>::try_new(StftConfig::new(16, 4, WindowFunction::Hamming))
        .map_err(|error| format!("STFT plan: {error:?}"))?;
    let samples: Vec<f64> = (0..48)
        .map(|index| (TAU * index as f64 / 16.0).sin())
        .collect();
    let spectra = stft.analyze(&samples);
    let reconstructed = stft.synthesize(&spectra, samples.len());
    let stft_error = samples
        .iter()
        .zip(&reconstructed)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0, f64::max);

    let elapsed_ms = started.elapsed().as_millis();
    let mut result = QaResult::passed(CASE_ID)
        .actual_backend("cpu")
        .metric("transform_case_count", transform_cases as f64)
        .metric("max_round_trip_error", max_round_trip_error)
        .metric("max_parseval_error", max_parseval_error)
        .metric("max_adjoint_error", max_adjoint_error)
        .metric("linear_convolution_error", convolution_error)
        .metric("nd_parseval_error", nd_error)
        .metric("stft_reconstruction_error", stft_error)
        .minimum("transform_case_count", 4.0)
        .maximum("max_round_trip_error", TOLERANCE)
        .maximum("max_parseval_error", TOLERANCE)
        .maximum("max_adjoint_error", TOLERANCE)
        .maximum("linear_convolution_error", TOLERANCE)
        .maximum("nd_parseval_error", TOLERANCE)
        .maximum("stft_reconstruction_error", TOLERANCE)
        .input_checksum("analytic/deterministic-fixture:fft-family-contracts-v1")
        .reference("independent scalar inner products, direct convolution, and energy identities")
        .reference_checksum("analytic/deterministic-fixture:fft-family-contracts-v1")
        .dimension("complex 1-D FFT; 2-D FFT; linear convolution; Hann STFT")
        .tolerance("all reported relative/absolute errors <= 2e-9")
        .seed("fft-family-contracts:3,8,15,31:v1")
        .timing_ms("cpu", elapsed_ms as u64)
        .elapsed_ms(elapsed_ms as u64)
        .backend_passed("cpu");
    result.validate().map_err(|error| error.to_string())?;
    Ok(result)
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result serialization")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone());
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result serialization")
            );
            eprintln!("FFT FAMILY CONTRACTS FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
