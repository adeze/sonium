//! Direct-solver release contract: accuracy, backward error, and rejection.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{
    DenseDirectConfig, DenseDirectMethod, DirectFactorization, MatrixClass, dense_backward_error,
    factorize_dense,
};
use sonium_solvers::{CsrMatrix, IluConfig, IluPreconditioner, Preconditioner};
use std::time::Instant;

fn main() {
    let started = Instant::now();
    match run() {
        Ok((
            backward,
            solution_error,
            indefinite_backward,
            indefinite_error,
            complex_error,
            direct_dimension_rejections,
            direct_non_finite_rejections,
            qr_multi_rhs_checks,
            preconditioner_converged,
            preconditioner_dimension_rejections,
            preconditioner_non_finite_rejections,
            preconditioner_config_rejections,
        )) => {
            let record = QaResult::passed("solvers.direct-contracts.v1")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("spd_backward_error", backward)
                .metric("spd_solution_error", solution_error)
                .metric("indefinite_backward_error", indefinite_backward)
                .metric("indefinite_solution_error", indefinite_error)
                .metric("complex_solution_error", complex_error)
                .metric("direct_dimension_rejections", direct_dimension_rejections)
                .metric("direct_non_finite_rejections", direct_non_finite_rejections)
                .metric("qr_multi_rhs_checks", qr_multi_rhs_checks)
                .metric("preconditioner_converged", preconditioner_converged)
                .metric(
                    "preconditioner_dimension_rejections",
                    preconditioner_dimension_rejections,
                )
                .metric(
                    "preconditioner_non_finite_rejections",
                    preconditioner_non_finite_rejections,
                )
                .metric(
                    "preconditioner_config_rejections",
                    preconditioner_config_rejections,
                )
                .maximum("spd_backward_error", 1.0e-12)
                .maximum("spd_solution_error", 1.0e-12)
                .maximum("indefinite_backward_error", 1.0e-12)
                .maximum("indefinite_solution_error", 1.0e-12)
                .maximum("complex_solution_error", 1.0e-12)
                .minimum("direct_dimension_rejections", 1.0)
                .minimum("direct_non_finite_rejections", 1.0)
                .minimum("qr_multi_rhs_checks", 1.0)
                .minimum("preconditioner_converged", 1.0)
                .minimum("preconditioner_dimension_rejections", 1.0)
                .minimum("preconditioner_non_finite_rejections", 1.0)
                .minimum("preconditioner_config_rejections", 1.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed("solvers.direct-contracts.v1", error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis() as u64)
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("DIRECT SOLVER FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64), String> {
    let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![4.0_f64, 1.0, 1.0, 3.0]);
    let rhs = [9.0, 8.0];
    let config = DenseDirectConfig {
        matrix_class: MatrixClass::SymmetricPositiveDefinite,
        method: DenseDirectMethod::Auto,
        ..Default::default()
    };
    let factor = factorize_dense(&matrix, &config).map_err(|e| e.to_string())?;
    let solution = factor.solve(&rhs).map_err(|e| e.to_string())?;
    let backward = dense_backward_error(&matrix, &solution, &rhs).map_err(|e| e.to_string())?;
    if backward > 1.0e-12
        || (solution[0] - 19.0 / 11.0).abs() > 1.0e-12
        || (solution[1] - 23.0 / 11.0).abs() > 1.0e-12
    {
        return Err(format!(
            "SPD solve accuracy/backward-error gate failed: {backward:e}, {solution:?}"
        ));
    }
    let direct_dimension_rejections = if factor.solve(&[1.0]).is_err() {
        1.0
    } else {
        return Err("dense direct factor accepted a dimension-mismatched RHS".into());
    };
    let direct_non_finite_rejections = if factor.solve(&[f64::NAN, 8.0]).is_err() {
        1.0
    } else {
        return Err("dense direct factor accepted a non-finite RHS".into());
    };

    let rectangular =
        DenseMatrix::from_row_major_vec(4, 2, vec![1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0]);
    let rectangular_factor = factorize_dense(
        &rectangular,
        &DenseDirectConfig {
            matrix_class: MatrixClass::RectangularFullRank,
            ..Default::default()
        },
    )
    .map_err(|error| format!("rectangular QR factorization: {error}"))?;
    let mut qr_solution = vec![0.0; 4];
    rectangular_factor
        .solve_many_into(
            &[2.0, 1.0, -3.0, 4.0, -1.0, 5.0, 7.0, -2.0],
            &mut qr_solution,
            2,
        )
        .map_err(|error| format!("rectangular QR multi-RHS solve: {error}"))?;
    let qr_multi_rhs_checks = if qr_solution.iter().all(|value| value.is_finite()) {
        1.0
    } else {
        return Err("rectangular QR multi-RHS produced non-finite output".into());
    };

    let singular = DenseMatrix::from_row_major_vec(2, 2, vec![1.0_f64, 2.0, 2.0, 4.0]);
    if factorize_dense(&singular, &DenseDirectConfig::default()).is_ok() {
        return Err("singular dense matrix was accepted".into());
    }
    let indefinite = DenseMatrix::from_row_major_vec(2, 2, vec![0.0_f64, 1.0, 1.0, 0.0]);
    let config = DenseDirectConfig {
        matrix_class: MatrixClass::SymmetricIndefinite,
        method: DenseDirectMethod::Auto,
        ..Default::default()
    };
    let indefinite_factor = factorize_dense(&indefinite, &config)
        .map_err(|error| format!("symmetric-indefinite LDLT factorization failed: {error}"))?;
    let indefinite_rhs = [1.0, 2.0];
    let indefinite_solution = indefinite_factor
        .solve(&indefinite_rhs)
        .map_err(|error| format!("symmetric-indefinite solve failed: {error}"))?;
    let indefinite_backward =
        dense_backward_error(&indefinite, &indefinite_solution, &indefinite_rhs)
            .map_err(|error| error.to_string())?;
    let indefinite_error = (indefinite_solution[0] - 2.0)
        .abs()
        .max((indefinite_solution[1] - 1.0).abs());
    if indefinite_backward > 1.0e-12 || indefinite_error > 1.0e-12 {
        return Err(format!(
            "symmetric-indefinite solve accuracy gate failed: backward={indefinite_backward:e}, solution={indefinite_solution:?}"
        ));
    }
    let direct_non_finite_rejections = direct_non_finite_rejections
        + if indefinite_factor.solve(&[f64::NAN, 2.0]).is_err() {
            1.0
        } else {
            return Err("dense LDLT factor accepted a non-finite RHS".into());
        };
    let complex = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, -1.0),
        ],
    );
    let complex_rhs = [Complex64::new(4.0, 0.0), Complex64::new(10.0, -3.0)];
    let complex_solution = factorize_dense(&complex, &DenseDirectConfig::default())
        .map_err(|e| e.to_string())?
        .solve(&complex_rhs)
        .map_err(|e| e.to_string())?;
    let complex_expected = [Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)];
    let complex_error = complex_solution
        .iter()
        .zip(complex_expected)
        .map(|(got, expected)| (*got - expected).norm())
        .fold(0.0_f64, f64::max);
    if complex_error > 1.0e-12 {
        return Err(format!("complex direct solve failed: {complex_solution:?}"));
    }

    let sparse_matrix: CsrMatrix<f64> = CsrMatrix::identity(2);
    let ilu = IluPreconditioner::try_from_csr_with_config(&sparse_matrix, &IluConfig::default())
        .map_err(|error| format!("checked ILU setup failed: {error}"))?;
    let ilu_solution = ilu
        .try_apply(&[1.0, -2.0])
        .map_err(|error| format!("checked ILU application failed: {error}"))?;
    if ilu_solution
        .iter()
        .zip([1.0, -2.0])
        .any(|(got, expected)| (*got - expected).abs() > 1.0e-12)
    {
        return Err(format!(
            "checked ILU application returned the wrong identity result: {ilu_solution:?}"
        ));
    }
    let preconditioner_converged = 1.0;
    let preconditioner_dimension_rejections = if ilu.try_apply(&[1.0]).is_err() {
        1.0
    } else {
        return Err("checked ILU accepted a dimension-mismatched input".into());
    };
    let preconditioner_non_finite_rejections = if ilu.try_apply(&[f64::NAN, -2.0]).is_err() {
        1.0
    } else {
        return Err("checked ILU accepted a non-finite input".into());
    };
    let mut invalid_ilu_config = IluConfig::default();
    invalid_ilu_config.pivot_tolerance = f64::NAN;
    let preconditioner_config_rejections = if IluPreconditioner::try_from_csr_with_config(
        &sparse_matrix,
        &invalid_ilu_config,
    )
    .is_err()
    {
        1.0
    } else {
        return Err("checked ILU accepted a non-finite pivot tolerance".into());
    };
    println!(
        "direct=auto spd_backward_error={backward:e} singular=rejected indefinite=ldlt-solved complex=solved"
    );
    Ok((
        backward,
        (solution[0] - 19.0 / 11.0)
            .abs()
            .max((solution[1] - 23.0 / 11.0).abs()),
        indefinite_backward,
        indefinite_error,
        complex_error,
        direct_dimension_rejections,
        direct_non_finite_rejections,
        qr_multi_rhs_checks,
        preconditioner_converged,
        preconditioner_dimension_rejections,
        preconditioner_non_finite_rejections,
        preconditioner_config_rejections,
    ))
}
