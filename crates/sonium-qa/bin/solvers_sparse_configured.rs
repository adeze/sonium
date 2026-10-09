//! Independent CPU QA for operator-retaining sparse direct diagnostics.

use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{
    DirectFactorization, MatrixClass, SparseDirectConfig, SparseDirectError, SparseDirectMethod,
    factorize_sparse, factorize_sparse_configured,
};
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers_core::traits::SolverResultContract;
use std::time::Instant;

const CASE_ID: &str = "solvers.sparse-configured-diagnostics.v1";
const TOLERANCE: f64 = 1.0e-12;

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_dense(
        &DenseMatrix::from_row_major_vec(2, 2, vec![2.0, 0.3, 0.2, 1.1]),
        0.0,
    )
}

fn config(right_hand_sides: usize) -> SparseDirectConfig {
    SparseDirectConfig {
        matrix_class: MatrixClass::General,
        method: SparseDirectMethod::SimplicialLu,
        right_hand_sides,
        backward_error_tolerance: TOLERANCE,
        ..SparseDirectConfig::default()
    }
}

fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn main() {
    let started = Instant::now();
    let outcome = (|| -> Result<QaResult, String> {
        let source = matrix();
        let factor = factorize_sparse_configured(&source, &config(2), None)
            .map_err(|error| format!("configured sparse setup: {error}"))?;

        let block_rhs = [1.4, -0.775, -2.0, 0.725];
        let expected_block = [1.0, -0.5, -2.0, 0.75];
        let mut block_solution = vec![0.0; block_rhs.len()];
        factor
            .solve_many_into(&block_rhs, &mut block_solution, 2)
            .map_err(|error| format!("configured sparse block solve: {error}"))?;
        let block_solution_error = max_error(&block_solution, &expected_block);
        let block_refined = factor
            .solve_refined_many(&block_rhs, 2, 2)
            .map_err(|error| format!("configured sparse block refinement: {error}"))?;
        if block_refined.len() != 2 {
            return Err("configured sparse block refinement returned wrong column count".into());
        }
        for result in &block_refined {
            result
                .validate_contract()
                .map_err(|error| format!("configured sparse block refinement contract: {error}"))?;
        }

        let checked_result = factor
            .solve_with_result(&[1.4, -2.0])
            .map_err(|error| format!("configured sparse result contract: {error}"))?;
        checked_result
            .validate_contract()
            .map_err(|error| format!("configured sparse result validation: {error}"))?;
        let checked_result_residual = checked_result
            .residual
            .ok_or_else(|| "configured sparse result omitted residual".to_string())?;
        let checked_result_backward_error = checked_result
            .backward_error
            .ok_or_else(|| "configured sparse result omitted backward error".to_string())?;

        let refined = factor
            .solve_refined(&[1.4, -2.0], 2)
            .map_err(|error| format!("configured sparse refinement: {error}"))?;
        if !refined.initial_backward_error.is_finite()
            || !refined.backward_error.is_finite()
            || refined.backward_error < 0.0
            || refined.iterations > 2
            || checked_result_residual > TOLERANCE
            || checked_result_backward_error > TOLERANCE
        {
            return Err("invalid sparse refinement diagnostics".into());
        }

        let mut invalid_rhs_count = config(0);
        invalid_rhs_count.right_hand_sides = 0;
        let invalid_rhs_count_rejected = matches!(
            factorize_sparse_configured(&source, &invalid_rhs_count, None),
            Err(SparseDirectError::InvalidConfiguration {
                field: "right_hand_sides",
                ..
            })
        );

        let mut invalid_tolerance = config(1);
        invalid_tolerance.backward_error_tolerance = f64::NAN;
        let invalid_tolerance_rejected = matches!(
            factorize_sparse_configured(&source, &invalid_tolerance, None),
            Err(SparseDirectError::InvalidConfiguration {
                field: "backward_error_tolerance",
                ..
            })
        );

        let mut budget = config(1);
        budget.memory_budget_bytes = Some(factor.factor_memory_bytes().saturating_sub(1));
        let memory_budget_rejected = matches!(
            factorize_sparse_configured(&source, &budget, None),
            Err(SparseDirectError::MemoryBudgetExceeded { .. })
        );

        let legacy_options_rejected = matches!(
            factorize_sparse(&source, &config(2), None),
            Err(SparseDirectError::OptionUnavailable { .. })
        );

        if block_solution_error > TOLERANCE
            || !refined.converged
            || !invalid_rhs_count_rejected
            || !invalid_tolerance_rejected
            || !memory_budget_rejected
            || !legacy_options_rejected
        {
            return Err(format!(
                "sparse configured gates failed: block_error={block_solution_error:e}, refined_converged={}, invalid_rhs_count_rejected={invalid_rhs_count_rejected}, invalid_tolerance_rejected={invalid_tolerance_rejected}, memory_budget_rejected={memory_budget_rejected}, legacy_options_rejected={legacy_options_rejected}",
                refined.converged
            ));
        }

        Ok(QaResult::passed(CASE_ID)
            .input_checksum("analytic/deterministic-fixture:solvers-sparse-configured-diagnostics-v1")
            .reference("known 2x2 sparse LU solution with retained-operator backward-error diagnostics")
            .reference_checksum("analytic/deterministic-fixture:solvers-sparse-configured-diagnostics-v1")
            .dimension("2x2 general CSR Simplicial LU; row-major two-RHS block")
            .tolerance("solution and backward-error diagnostics <= 1e-12; invalid options and budget reject")
            .metric("block_solution_error", block_solution_error)
            .metric("block_refinement_columns", block_refined.len() as f64)
            .metric("checked_result_residual", checked_result_residual)
            .metric("checked_result_backward_error", checked_result_backward_error)
            .metric("initial_backward_error", refined.initial_backward_error)
            .metric("backward_error", refined.backward_error)
            .metric("refinement_iterations", refined.iterations as f64)
            .metric("invalid_rhs_count_rejected", invalid_rhs_count_rejected as u8 as f64)
            .metric("invalid_tolerance_rejected", invalid_tolerance_rejected as u8 as f64)
            .metric("memory_budget_rejected", memory_budget_rejected as u8 as f64)
            .metric("legacy_options_rejected", legacy_options_rejected as u8 as f64)
            .maximum("block_solution_error", TOLERANCE)
            .maximum("checked_result_residual", TOLERANCE)
            .maximum("checked_result_backward_error", TOLERANCE)
            .maximum("initial_backward_error", TOLERANCE)
            .maximum("backward_error", TOLERANCE)
            .elapsed_ms(started.elapsed().as_millis() as u64))
    })();

    match outcome {
        Ok(result) => {
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("known 2x2 sparse LU solution with retained-operator backward-error diagnostics")
                .input_checksum("analytic/deterministic-fixture:solvers-sparse-configured-diagnostics-v1")
                .reference_checksum("analytic/deterministic-fixture:solvers-sparse-configured-diagnostics-v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("SPARSE CONFIGURED DIAGNOSTICS FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
