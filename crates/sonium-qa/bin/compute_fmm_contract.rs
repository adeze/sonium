//! CPU-only checked FMM gather/matvec/scatter contract.

use num_complex::Complex64;
use rand::{Rng, SeedableRng, rngs::StdRng};
use sonium_compute::fmm_cpu::{
    try_indexed_dense_matvec_accumulate_c64, try_indexed_dense_matvec_c64,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.fmm-contract.v1";
const CASES: usize = 48;
const MAX_ERROR: f64 = 2.0e-12;

struct Metrics {
    cases: usize,
    max_error: f64,
}

fn max_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max)
}

fn run() -> Result<Metrics, String> {
    let mut rng = StdRng::seed_from_u64(0xF5AA_0001);
    let mut maximum_error = 0.0_f64;

    for case in 0..CASES {
        let rows = 1 + (case * 5 % 7);
        let cols = 1 + (case * 7 % 8);
        let leading_dimension = cols + case % 3;
        let source_len = cols + 3 + case % 5;
        let matrix: Vec<Complex64> = (0..rows * leading_dimension)
            .map(|_| Complex64::new(rng.random_range(-2.0..=2.0), rng.random_range(-2.0..=2.0)))
            .collect();
        let source: Vec<Complex64> = (0..source_len)
            .map(|_| Complex64::new(rng.random_range(-2.0..=2.0), rng.random_range(-2.0..=2.0)))
            .collect();
        let src_indices: Vec<usize> = (0..cols).map(|_| rng.random_range(0..source_len)).collect();
        let dst_indices: Vec<usize> = (0..rows).map(|_| rng.random_range(0..source_len)).collect();

        let mut gathered = vec![Complex64::default(); rows];
        try_indexed_dense_matvec_c64(
            &matrix,
            rows,
            cols,
            leading_dimension,
            &source,
            &src_indices,
            &mut gathered,
        )
        .map_err(|error| format!("gather case {case} failed: {error}"))?;
        let expected_gather: Vec<Complex64> = (0..rows)
            .map(|row| {
                (0..cols).fold(Complex64::default(), |sum, col| {
                    sum + matrix[row * leading_dimension + col] * source[src_indices[col]]
                })
            })
            .collect();

        let initial: Vec<Complex64> = (0..source_len)
            .map(|_| Complex64::new(rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0)))
            .collect();
        let mut scattered = initial.clone();
        try_indexed_dense_matvec_accumulate_c64(
            &matrix,
            rows,
            cols,
            leading_dimension,
            &source,
            &src_indices,
            &mut scattered,
            &dst_indices,
        )
        .map_err(|error| format!("scatter case {case} failed: {error}"))?;
        let mut expected_scatter = initial;
        for row in 0..rows {
            let value = expected_gather[row];
            expected_scatter[dst_indices[row]] += value;
        }

        let error =
            max_error(&gathered, &expected_gather).max(max_error(&scattered, &expected_scatter));
        maximum_error = maximum_error.max(error);
        if error > MAX_ERROR {
            return Err(format!("case {case} exceeded error gate: {error}"));
        }
    }

    Ok(Metrics {
        cases: CASES,
        max_error: maximum_error,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("case_count", metrics.cases as f64)
                .metric("max_error", metrics.max_error)
                .minimum("case_count", CASES as f64)
                .maximum("max_error", MAX_ERROR)
                .reference("independent scalar complex row accumulation")
                .reference_checksum("analytic/deterministic-fixture:compute-fmm-contract-v1")
                .dimension(
                    "48 seeded rectangular complex FMM cases; repeated source/destination indices",
                )
                .seed("compute-fmm-contract:rectangular-repeated-indices-v1");
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("FMM QA result must serialize")
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
                    .expect("FMM QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
