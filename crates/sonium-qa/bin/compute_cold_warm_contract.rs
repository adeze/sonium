//! Bounded cold/warm compute evidence with allocation and correctness gates.
//!
//! This deliberately uses a tiny CPU GEMM so it cannot turn into a simulator
//! workload.  Input allocation and transfers are outside the timing window;
//! the gate records first-use (cold) and repeated (warm) compute timings,
//! checks numerical output against an independent scalar reference, and
//! verifies that the caller-owned pool accounting remains stable.

use std::hint::black_box;
use std::time::Instant;

use sonium_compute::{Backend, CpuBackend, MemoryPool, MemoryPoolConfig, Transpose};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.cold-warm-contract.v1";
const M: usize = 8;
const N: usize = 7;
const K: usize = 6;
const WARM_SAMPLES: usize = 5;
const MEMORY_CAP_BYTES: usize = 16 * 1024;
const MAX_ERROR: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    cold_ns: u128,
    warm_min_ns: u128,
    warm_median_ns: u128,
    cold_to_warm_ratio: f64,
    warm_samples: usize,
    max_abs_error: f64,
    bytes_in_use: usize,
    high_water_bytes: usize,
    expected_bytes: usize,
}

fn fixture(len: usize, salt: usize) -> Vec<f64> {
    (0..len)
        .map(|index| {
            let phase = ((index * 17 + salt * 11 + 3) as f64) * 0.03125;
            phase.sin() * 0.75 + phase.cos() * 0.125
        })
        .collect()
}

fn reference(a: &[f64], b: &[f64]) -> Vec<f64> {
    (0..M * N)
        .map(|index| {
            let row = index / N;
            let column = index % N;
            (0..K)
                .map(|inner| a[row * K + inner] * b[inner * N + column])
                .sum()
        })
        .collect()
}

fn run() -> Result<Metrics, String> {
    let a = fixture(M * K, 1);
    let b = fixture(K * N, 2);
    let expected = reference(&a, &b);
    let pool = MemoryPool::new(
        CpuBackend::new(),
        MemoryPoolConfig::with_cap(MEMORY_CAP_BYTES),
    )
    .map_err(|error| format!("pool construction failed: {error}"))?;
    let mut a_buffer = pool
        .try_alloc::<f64>(a.len())
        .map_err(|error| format!("A allocation failed: {error}"))?;
    let mut b_buffer = pool
        .try_alloc::<f64>(b.len())
        .map_err(|error| format!("B allocation failed: {error}"))?;
    let mut c_buffer = pool
        .try_alloc::<f64>(expected.len())
        .map_err(|error| format!("C allocation failed: {error}"))?;
    pool.backend().copy_h2d(&a, a_buffer.raw_mut());
    pool.backend().copy_h2d(&b, b_buffer.raw_mut());
    pool.backend()
        .copy_h2d(&vec![0.0; expected.len()], c_buffer.raw_mut());

    let cold_start = Instant::now();
    pool.backend()
        .try_gemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            M,
            N,
            K,
            1.0,
            a_buffer.raw(),
            K,
            b_buffer.raw(),
            N,
            0.0,
            c_buffer.raw_mut(),
            N,
        )
        .map_err(|error| format!("cold GEMM failed: {error}"))?;
    pool.backend()
        .try_synchronize()
        .map_err(|error| format!("cold synchronization failed: {error}"))?;
    let cold_ns = cold_start.elapsed().as_nanos();

    let mut warm_timings = Vec::with_capacity(WARM_SAMPLES);
    for _ in 0..WARM_SAMPLES {
        let start = Instant::now();
        pool.backend()
            .try_gemm(
                Transpose::NoTrans,
                Transpose::NoTrans,
                M,
                N,
                K,
                1.0,
                a_buffer.raw(),
                K,
                b_buffer.raw(),
                N,
                0.0,
                c_buffer.raw_mut(),
                N,
            )
            .map_err(|error| format!("warm GEMM failed: {error}"))?;
        pool.backend()
            .try_synchronize()
            .map_err(|error| format!("warm synchronization failed: {error}"))?;
        warm_timings.push(start.elapsed().as_nanos());
    }
    warm_timings.sort_unstable();

    let mut actual = vec![0.0; expected.len()];
    pool.backend().copy_d2h(c_buffer.raw(), &mut actual);
    let max_abs_error = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    if !max_abs_error.is_finite() || max_abs_error > MAX_ERROR {
        return Err(format!(
            "cold/warm GEMM reference error {max_abs_error:.3e} exceeds {MAX_ERROR:.3e}"
        ));
    }

    let expected_bytes = (a.len() + b.len() + expected.len()) * std::mem::size_of::<f64>();
    let warm_median_ns = warm_timings[WARM_SAMPLES / 2];
    let cold_to_warm_ratio = cold_ns as f64 / warm_median_ns as f64;
    let metrics = Metrics {
        cold_ns,
        warm_min_ns: *warm_timings
            .first()
            .ok_or_else(|| "warm timing sample set is empty".to_owned())?,
        warm_median_ns,
        cold_to_warm_ratio,
        warm_samples: warm_timings.len(),
        max_abs_error,
        bytes_in_use: pool.bytes_in_use(),
        high_water_bytes: pool.high_water_bytes(),
        expected_bytes,
    };
    if metrics.cold_ns == 0
        || metrics.warm_min_ns == 0
        || metrics.warm_median_ns == 0
        || !metrics.cold_to_warm_ratio.is_finite()
        || metrics.cold_to_warm_ratio <= 0.0
        || metrics.warm_samples != WARM_SAMPLES
        || metrics.bytes_in_use != expected_bytes
        || metrics.high_water_bytes != expected_bytes
    {
        return Err(format!("cold/warm accounting contract failed: {metrics:?}"));
    }
    black_box(actual);
    Ok(metrics)
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("cold_ns", metrics.cold_ns as f64)
                .metric("warm_min_ns", metrics.warm_min_ns as f64)
                .metric("warm_median_ns", metrics.warm_median_ns as f64)
                .metric("cold_to_warm_ratio", metrics.cold_to_warm_ratio)
                .metric("warm_samples", metrics.warm_samples as f64)
                .metric("max_abs_error", metrics.max_abs_error)
                .metric("bytes_in_use", metrics.bytes_in_use as f64)
                .metric("high_water_bytes", metrics.high_water_bytes as f64)
                .metric("expected_bytes", metrics.expected_bytes as f64)
                .minimum("warm_samples", WARM_SAMPLES as f64)
                .maximum("max_abs_error", MAX_ERROR)
                .input_checksum("analytic/deterministic-fixture:compute-cold-warm-v1")
                .reference("independent scalar row-major GEMM reference")
                .reference_checksum("analytic/deterministic-fixture")
                .dimension("CPU f64 8x7x6 GEMM; one cold and five warm samples")
                .formulation("caller-owned MemoryPool cold/warm compute contract")
                .tolerance("finite non-zero timings; exact pool accounting; error <= 1e-12")
                .seed("compute-cold-warm:8x7x6:f64:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("cold/warm result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("cold/warm failure result must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_warm_contract_checks_reference_and_pool_accounting() {
        let metrics = run().expect("tiny cold/warm contract should pass");
        assert_eq!(metrics.warm_samples, WARM_SAMPLES);
        assert_eq!(metrics.bytes_in_use, metrics.expected_bytes);
        assert_eq!(metrics.high_water_bytes, metrics.expected_bytes);
        assert!(metrics.max_abs_error <= MAX_ERROR);
        assert!(metrics.cold_to_warm_ratio.is_finite());
        assert!(metrics.cold_to_warm_ratio > 0.0);
    }
}
