//! Independent CPU evidence for coarse-space Schwarz adjoints and fail-closed
//! fixed-point ILU adjoint capability.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers::preconditioners::{AdditiveSchwarzPreconditioner, IluFixedPointPreconditioner};
use sonium_solvers::{AdjointCapability, CsrMatrix, Preconditioner, SolverError};
use std::time::Instant;

const CASE_ID: &str = "solvers.preconditioner-coarse-adjoint.v1";
const TOLERANCE: f64 = 1.0e-10;

fn matrix() -> CsrMatrix<Complex64> {
    CsrMatrix::from_raw_parts(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -0.5),
            Complex64::new(2.0, 0.25),
            Complex64::new(5.0, -1.0),
            Complex64::new(1.0, 0.75),
            Complex64::new(0.5, -0.25),
            Complex64::new(3.0, 0.5),
            Complex64::new(0.75, -0.2),
            Complex64::new(1.0, 0.1),
            Complex64::new(2.5, -0.5),
        ],
    )
}

fn vectors() -> (Vec<Complex64>, Vec<Complex64>) {
    (
        vec![
            Complex64::new(0.7, -0.2),
            Complex64::new(-1.1, 0.4),
            Complex64::new(0.3, 0.9),
            Complex64::new(1.4, -0.6),
        ],
        vec![
            Complex64::new(-0.5, 0.8),
            Complex64::new(0.2, -1.3),
            Complex64::new(1.1, 0.1),
            Complex64::new(-0.4, 0.6),
        ],
    )
}

fn hermitian_inner(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    a.iter().zip(b).map(|(a, b)| a.conj() * b).sum()
}

fn transpose_inner(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    a.iter().zip(b).map(|(a, b)| *a * *b).sum()
}

fn check_coarse_adjoint(
    preconditioner: &AdditiveSchwarzPreconditioner<Complex64>,
    r: &[Complex64],
    s: &[Complex64],
) -> (f64, f64) {
    let forward = preconditioner.apply(r);
    let transpose = preconditioner
        .try_apply_transpose(s)
        .expect("coarse Schwarz transpose capability is exact");
    let hermitian = preconditioner
        .try_apply_hermitian(s)
        .expect("coarse Schwarz Hermitian capability is exact");

    let transpose_error = (transpose_inner(s, &forward) - transpose_inner(&transpose, r)).norm();
    let hermitian_error = (hermitian_inner(s, &forward) - hermitian_inner(&hermitian, r)).norm();
    (transpose_error, hermitian_error)
}

fn run() -> Result<QaResult, String> {
    let matrix = matrix();
    let coarse = AdditiveSchwarzPreconditioner::try_from_csr(&matrix, 2, 1)
        .map_err(|error| format!("coarse Schwarz setup failed: {error}"))?
        .with_nicolaides_coarse_space(&matrix);
    if !coarse.has_coarse_space() {
        return Err("coarse Schwarz setup unexpectedly fell back to one level".into());
    }
    if coarse.transpose_capability() != AdjointCapability::Exact
        || coarse.hermitian_capability() != AdjointCapability::Exact
    {
        return Err("coarse Schwarz did not advertise exact adjoint actions".into());
    }

    let (r, s) = vectors();
    let (transpose_error, hermitian_error) = check_coarse_adjoint(&coarse, &r, &s);
    if transpose_error > TOLERANCE || hermitian_error > TOLERANCE {
        return Err(format!(
            "coarse Schwarz adjoint parity exceeded tolerance: transpose={transpose_error:e}, hermitian={hermitian_error:e}"
        ));
    }

    let fixed_point = IluFixedPointPreconditioner::try_from_csr(&matrix, 3)
        .map_err(|error| format!("fixed-point ILU setup failed: {error}"))?;
    if fixed_point.transpose_capability() != AdjointCapability::Unsupported
        || fixed_point.hermitian_capability() != AdjointCapability::Unsupported
    {
        return Err("fixed-point ILU advertised an unimplemented adjoint action".into());
    }
    if !matches!(
        fixed_point.try_apply_transpose(&r),
        Err(SolverError::UnsupportedOperation { .. })
    ) || !matches!(
        fixed_point.try_apply_hermitian(&r),
        Err(SolverError::UnsupportedOperation { .. })
    ) {
        return Err("fixed-point ILU adjoint calls did not fail closed".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("coarse_transpose_error", transpose_error)
        .metric("coarse_hermitian_error", hermitian_error)
        .metric("coarse_adjoint_tolerance", TOLERANCE)
        .metric("fixed_point_adjoint_unsupported", 1.0))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(result) => {
            let result = result.metric("elapsed_seconds", started.elapsed().as_secs_f64());
            println!("QA_RESULT: {}", serde_json::to_string(&result).unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error);
            println!("QA_RESULT: {}", serde_json::to_string(&result).unwrap());
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
