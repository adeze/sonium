//! Independent checked setup evidence for Chebyshev finite-value admission.

use sonium_qa::QaResult;
use sonium_solvers::CsrMatrix;
use sonium_solvers::preconditioners::{ChebyshevConfig, ChebyshevError, ChebyshevPreconditioner};
use std::time::Instant;

const CASE_ID: &str = "solvers.chebyshev-finite-setup.v1";

fn run() -> Result<(f64, f64, f64), String> {
    let mut nan_matrix = CsrMatrix::identity(1);
    nan_matrix.values[0] = f64::NAN;
    let nan_rejections =
        match ChebyshevPreconditioner::try_from_csr(&nan_matrix, ChebyshevConfig::default()) {
            Err(ChebyshevError::NonFiniteEntries { count: 1 }) => 1.0,
            Ok(_) => return Err("NaN matrix entry was admitted".into()),
            Err(error) => return Err(format!("unexpected NaN admission error: {error}")),
        };

    let mut infinity_matrix = CsrMatrix::identity(1);
    infinity_matrix.values[0] = f64::INFINITY;
    let infinity_rejections =
        match ChebyshevPreconditioner::try_from_csr(&infinity_matrix, ChebyshevConfig::default()) {
            Err(ChebyshevError::NonFiniteEntries { count: 1 }) => 1.0,
            Ok(_) => return Err("infinite matrix entry was admitted".into()),
            Err(error) => return Err(format!("unexpected infinity admission error: {error}")),
        };

    let finite = ChebyshevPreconditioner::try_from_csr(
        &CsrMatrix::<f64>::identity(2),
        ChebyshevConfig::default(),
    )
    .map_err(|error| format!("finite Chebyshev setup failed: {error}"))?;
    if !finite.lambda_max().is_finite() {
        return Err("finite Chebyshev setup produced a non-finite lambda_max".into());
    }

    Ok((nan_rejections, infinity_rejections, 1.0))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((nan_rejections, infinity_rejections, finite_setup)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .metric("nan_rejections", nan_rejections)
            .metric("infinity_rejections", infinity_rejections)
            .metric("finite_setup", finite_setup)
            .minimum("nan_rejections", 1.0)
            .minimum("infinity_rejections", 1.0)
            .minimum("finite_setup", 1.0),
        Err(error) => QaResult::failed(CASE_ID, error),
    }
    .elapsed_ms(started.elapsed().as_millis().try_into().unwrap());
    println!("{}", result.to_json_line().unwrap());
}
