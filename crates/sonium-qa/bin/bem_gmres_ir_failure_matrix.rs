//! Deterministic checked-failure matrix for the CPU BEM mixed-precision solver.

use num_complex::Complex64;
use sonium_bem::core::solver::{
    BemIrConfig, DenseOperator, gmres_ir_solve_with_chebyshev_operator,
};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use std::time::Instant;

const CASE_ID: &str = "bem.gmres-ir-failure-matrix.v1";

fn c(value: f64) -> Complex64 {
    Complex64::new(value, 0.0)
}

fn require_rejected(
    label: &str,
    solution: &sonium_bem::core::solver::BemIrSolution,
    expected_size: usize,
) -> Result<(), String> {
    if solution.converged {
        return Err(format!("{label}: invalid input reported convergence"));
    }
    if solution.solution.len() != expected_size {
        return Err(format!(
            "{label}: solution length {} != {expected_size}",
            solution.solution.len()
        ));
    }
    Ok(())
}

fn require_invalid_input(
    label: &str,
    solution: &sonium_bem::core::solver::BemIrSolution,
    expected_size: usize,
) -> Result<(), String> {
    require_rejected(label, solution, expected_size)?;
    if !solution.final_residual.is_infinite() {
        return Err(format!(
            "{label}: invalid input residual was {:?}, expected infinity",
            solution.final_residual
        ));
    }
    Ok(())
}

fn run() -> Result<(usize, usize), String> {
    let operator = DenseOperator::new(DenseMatrix::eye(2));
    let nearfield = DenseMatrix::eye(2);
    let valid_rhs = [c(1.0), c(0.0)];

    let nonfinite_rhs = [Complex64::new(f64::NAN, 0.0), c(0.0)];
    require_invalid_input(
        "nonfinite_rhs",
        &gmres_ir_solve_with_chebyshev_operator(
            &operator,
            &nearfield,
            &nonfinite_rhs,
            &BemIrConfig::default(),
        ),
        2,
    )?;

    let mut invalid_configs = Vec::new();
    let mut config = BemIrConfig::default();
    config.max_outer_iterations = 0;
    invalid_configs.push(config);
    let mut config = BemIrConfig::default();
    config.max_inner_iterations = 0;
    invalid_configs.push(config);
    let mut config = BemIrConfig::default();
    config.chebyshev_degree = 0;
    invalid_configs.push(config);
    let mut config = BemIrConfig::default();
    config.outer_tolerance = f64::NAN;
    invalid_configs.push(config);
    let mut config = BemIrConfig::default();
    config.inner_tolerance = f32::NAN;
    invalid_configs.push(config);
    let mut config = BemIrConfig::default();
    config.chebyshev_eigenvalue_ratio = 1.0;
    invalid_configs.push(config);

    for (index, config) in invalid_configs.into_iter().enumerate() {
        require_invalid_input(
            &format!("invalid_config_{index}"),
            &gmres_ir_solve_with_chebyshev_operator(&operator, &nearfield, &valid_rhs, &config),
            2,
        )?;
    }

    let zero_nearfield = DenseMatrix::zeros(2, 2);
    require_rejected(
        "inner_preconditioner_failure",
        &gmres_ir_solve_with_chebyshev_operator(
            &operator,
            &zero_nearfield,
            &valid_rhs,
            &BemIrConfig::default(),
        ),
        2,
    )?;

    Ok((1 + 6, 1))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((invalid_input_cases, inner_failure_cases)) => {
            let result = QaResult::passed(CASE_ID)
                .reference("independent checked failure/status matrix")
                .reference_checksum("analytic/deterministic-fixture:bem-gmres-ir-failure-matrix-v1")
                .input_checksum("analytic/deterministic-fixture:bem-gmres-ir-failure-matrix-v1")
                .metric("invalid_input_cases", invalid_input_cases as f64)
                .metric("inner_failure_cases", inner_failure_cases as f64)
                .metric("panic_cases", 0.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            let result =
                QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
