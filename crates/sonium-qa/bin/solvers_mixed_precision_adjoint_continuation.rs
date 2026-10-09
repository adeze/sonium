//! Independent CPU evidence for mixed-precision transpose/Hermitian continuation.

use num_complex::{Complex32, Complex64};
use sonium_qa::QaResult;
use sonium_solvers::mixed_precision::MixedPrecisionPreconditioner;
use sonium_solvers::{
    AdjointCapability, AdjointSolve, CsrMatrix, DiagonalPreconditioner, DifferentiableLinearSolve,
    GmresConfig, LinearOperator, Preconditioner,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.mixed-precision-adjoint-continuation.v1";
const TOLERANCE: f64 = 1.0e-10;

fn operator() -> CsrMatrix<Complex64> {
    CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, Complex64::new(4.0, 0.0)),
            (0, 1, Complex64::new(1.0, 0.5)),
            (1, 0, Complex64::new(1.0, -0.5)),
            (1, 1, Complex64::new(3.0, 0.0)),
            (1, 2, Complex64::new(0.75, 0.25)),
            (2, 1, Complex64::new(0.75, -0.25)),
            (2, 2, Complex64::new(2.5, 0.0)),
        ],
    )
}

fn preconditioner() -> MixedPrecisionPreconditioner<Complex64, Complex32> {
    MixedPrecisionPreconditioner::new(Box::new(DiagonalPreconditioner::from_diagonal(&[
        Complex32::new(4.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(2.5, 0.0),
    ])))
}

fn config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 40,
        restart: 3,
        tolerance: 1.0e-12,
        ..GmresConfig::default()
    }
}

fn relative_residual(
    operator: &CsrMatrix<Complex64>,
    kind: AdjointSolve,
    solution: &[Complex64],
    rhs: &[Complex64],
) -> f64 {
    let action = match kind {
        AdjointSolve::Transpose => operator.apply_transpose(solution),
        AdjointSolve::Hermitian => operator.apply_hermitian(solution),
    };
    let residual = action
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
    residual / rhs_norm.max(f64::MIN_POSITIVE)
}

fn max_delta(left: &[Complex64], right: &[Complex64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (*left - *right).norm())
        .fold(0.0, f64::max)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let operator = operator();
    let preconditioner = preconditioner();
    if operator.transpose_capability() != AdjointCapability::Exact
        || operator.hermitian_capability() != AdjointCapability::Exact
        || preconditioner.transpose_capability() != AdjointCapability::Exact
        || preconditioner.hermitian_capability() != AdjointCapability::Exact
    {
        return Err("mixed-precision adjoint capabilities were not exact".into());
    }
    let handle = DifferentiableLinearSolve::new(&operator, &preconditioner, config())
        .map_err(|error| format!("mixed-precision handle setup failed: {error}"))?;
    let rhs = [
        Complex64::new(1.0, 2.0),
        Complex64::new(-2.0, 0.5),
        Complex64::new(0.5, -1.0),
    ];
    let transpose = handle
        .solve_adjoint(&rhs, AdjointSolve::Transpose)
        .map_err(|error| format!("mixed-precision transpose solve failed: {error}"))?;
    let resumed_transpose = handle
        .solve_adjoint_with_guess(&rhs, AdjointSolve::Transpose, Some(&transpose.x))
        .map_err(|error| format!("mixed-precision transpose continuation failed: {error}"))?;
    let hermitian = handle
        .solve_adjoint(&rhs, AdjointSolve::Hermitian)
        .map_err(|error| format!("mixed-precision Hermitian solve failed: {error}"))?;
    let resumed_hermitian = handle
        .solve_adjoint_with_guess(&rhs, AdjointSolve::Hermitian, Some(&hermitian.x))
        .map_err(|error| format!("mixed-precision Hermitian continuation failed: {error}"))?;
    if !transpose.converged
        || !resumed_transpose.converged
        || !hermitian.converged
        || !resumed_hermitian.converged
        || resumed_transpose.iterations != 0
        || resumed_hermitian.iterations != 0
    {
        return Err(format!(
            "mixed-precision adjoint continuation failed: transpose={}/{} Hermitian={}/{} iterations={}/{}",
            transpose.converged,
            resumed_transpose.converged,
            hermitian.converged,
            resumed_hermitian.converged,
            resumed_transpose.iterations,
            resumed_hermitian.iterations,
        ));
    }
    let malformed_guess_rejected = handle
        .solve_adjoint_with_guess(
            &rhs,
            AdjointSolve::Hermitian,
            Some(&[Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]),
        )
        .is_err();
    let non_finite_guess_rejected = handle
        .solve_adjoint_with_guess(
            &rhs,
            AdjointSolve::Hermitian,
            Some(&[
                Complex64::new(0.0, 0.0),
                Complex64::new(f64::NAN, 0.0),
                Complex64::new(0.0, 0.0),
            ]),
        )
        .is_err();
    if !malformed_guess_rejected || !non_finite_guess_rejected {
        return Err("mixed-precision malformed adjoint guess was accepted".into());
    }
    Ok(vec![
        (
            "transpose_residual",
            relative_residual(
                &operator,
                AdjointSolve::Transpose,
                &resumed_transpose.x,
                &rhs,
            ),
        ),
        (
            "hermitian_residual",
            relative_residual(
                &operator,
                AdjointSolve::Hermitian,
                &resumed_hermitian.x,
                &rhs,
            ),
        ),
        (
            "transpose_continuation_delta",
            max_delta(&transpose.x, &resumed_transpose.x),
        ),
        (
            "hermitian_continuation_delta",
            max_delta(&hermitian.x, &resumed_hermitian.x),
        ),
        (
            "transpose_resumed_iterations",
            resumed_transpose.iterations as f64,
        ),
        (
            "hermitian_resumed_iterations",
            resumed_hermitian.iterations as f64,
        ),
        ("continuation_contract", 1.0),
        (
            "malformed_guess_rejection",
            if malformed_guess_rejected { 1.0 } else { 0.0 },
        ),
        (
            "non_finite_guess_rejection",
            if non_finite_guess_rejected { 1.0 } else { 0.0 },
        ),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("analytic complex Hermitian CSR mixed-precision adjoint continuation")
                .input_checksum(
                    "analytic/deterministic-fixture:mixed-precision-adjoint-continuation-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:mixed-precision-adjoint-continuation-v1",
                )
                .dimension("3x3 Complex64 CSR; Complex32 diagonal preconditioner")
                .tolerance("adjoint residual/delta <= 1e-10; continuation/rejection contracts = 1")
                .seed("solvers-mixed-precision-adjoint-continuation:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = if name == "continuation_contract" || name.ends_with("_rejection") {
                    result.minimum(name, 1.0).metric(name, value)
                } else {
                    result.maximum(name, TOLERANCE).metric(name, value)
                };
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            eprintln!("FAILED: {error}");
            let result = QaResult::failed(CASE_ID, error)
                .input_checksum(
                    "analytic/deterministic-fixture:mixed-precision-adjoint-continuation-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:mixed-precision-adjoint-continuation-v1",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            std::process::exit(1);
        }
    }
}
