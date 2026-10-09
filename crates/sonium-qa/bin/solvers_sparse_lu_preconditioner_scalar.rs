//! Independent CPU scalar-family coverage for relaxed sparse-LU pivots.

use num_complex::{Complex32, Complex64};
use num_traits::{Float, FromPrimitive, One, ToPrimitive};
use sonium_qa::QaResult;
use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_direct::SparseLuFactorization;
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.sparse-lu-preconditioner-scalar.v1";

fn real<T: ComplexField>(value: f64) -> T {
    T::from_real(T::Real::from_f64(value).expect("fixture value representable"))
}

fn check<T: ComplexField + FromPrimitive>(pivot: f64, tolerance: f64) -> Result<f64, String> {
    let matrix = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![real::<T>(pivot)]);
    let rhs = [real::<T>(1.0)];
    let expected = rhs[0] * real::<T>(pivot).inv();
    let factor = SparseLuFactorization::factorize_for_preconditioner(&matrix)
        .map_err(|error| format!("factorization rejected finite pivot: {error}"))?;
    let solution = factor
        .solve(&rhs)
        .map_err(|error| format!("solve rejected admitted pivot: {error}"))?;
    let relative_error = (solution[0] - expected).norm() / expected.norm().max(T::Real::one());
    let error = relative_error
        .to_f64()
        .ok_or_else(|| "relative error did not convert to f64".to_string())?;
    if !error.is_finite() || error > tolerance {
        return Err(format!("relative error {error:e} exceeds {tolerance:e}"));
    }
    Ok(error)
}

fn run() -> Result<(usize, f64), String> {
    let mut checks = 0;
    let mut maximum_error = 0.0_f64;
    maximum_error = maximum_error.max(check::<f32>(1.0e-31, 3.0e-6)?);
    checks += 1;
    maximum_error = maximum_error.max(check::<f64>(1.0e-31, 3.0e-13)?);
    checks += 1;
    maximum_error = maximum_error.max(check::<Complex32>(1.0e-31, 3.0e-6)?);
    checks += 1;
    maximum_error = maximum_error.max(check::<Complex64>(1.0e-31, 3.0e-13)?);
    checks += 1;
    Ok((checks, maximum_error))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((checks, maximum_error)) => {
            let result = QaResult::passed(CASE_ID)
                .reference(
                    "independent f32/f64/Complex32/Complex64 sparse-LU preconditioner pivot solve",
                )
                .input_checksum(
                    "analytic/deterministic-fixture:solvers-sparse-lu-preconditioner-scalar-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:solvers-sparse-lu-preconditioner-scalar-v1",
                )
                .dimension("1x1 CSR with finite 1e-31 pivot across four scalar families")
                .metric("scalar_checks", checks as f64)
                .metric("maximum_relative_error", maximum_error)
                .minimum("scalar_checks", 4.0)
                .maximum("maximum_relative_error", 3.0e-6)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference(
                    "independent f32/f64/Complex32/Complex64 sparse-LU preconditioner pivot solve",
                )
                .input_checksum(
                    "analytic/deterministic-fixture:solvers-sparse-lu-preconditioner-scalar-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:solvers-sparse-lu-preconditioner-scalar-v1",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
