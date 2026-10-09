//! Release-facing reference for checked multifrontal LDLT factorization.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers::CsrMatrix;
use sonium_solvers::direct::MultifrontalLdltFactorization;
use std::time::Instant;

const CASE_ID: &str = "solvers.multifrontal-ldlt-reference.v1";

fn run_real() -> Result<(f64, f64, f64), String> {
    // The exchange matrix has an exact 2x2 Bunch--Kaufman pivot.  Its
    // independent analytic inverse is itself: A [2, -1] = [-1, 2].
    let matrix = CsrMatrix::from_triplets(2, 2, vec![(0, 1, 1.0_f64), (1, 0, 1.0_f64)]);
    let rhs = [2.0_f64, -1.0];
    let expected = [-1.0_f64, 2.0];
    let factor = MultifrontalLdltFactorization::factorize(&matrix)
        .map_err(|error| format!("real multifrontal LDLT factorization: {error}"))?;
    let solution = factor
        .try_solve(&rhs)
        .map_err(|error| format!("real multifrontal LDLT solve: {error}"))?;
    let solution_error = solution
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    let residual_error = (0..2)
        .map(|row| {
            let value = if row == 0 { solution[1] } else { solution[0] };
            (value - rhs[row]).abs()
        })
        .fold(0.0, f64::max);
    Ok((
        solution_error,
        residual_error,
        if factor.two_by_two_pivot_count() > 0 {
            1.0
        } else {
            0.0
        },
    ))
}

fn run_complex() -> Result<(f64, f64), String> {
    let a01 = Complex64::new(1.0, 2.0);
    let a10 = a01.conj();
    let expected = [Complex64::new(1.0, -0.5), Complex64::new(2.0, 0.75)];
    let matrix = CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, Complex64::new(3.0, 0.0)),
            (0, 1, a01),
            (1, 0, a10),
            (1, 1, Complex64::new(6.0, 0.0)),
        ],
    );
    let rhs = [
        Complex64::new(3.0, 0.0) * expected[0] + a01 * expected[1],
        a10 * expected[0] + Complex64::new(6.0, 0.0) * expected[1],
    ];
    let factor = MultifrontalLdltFactorization::factorize(&matrix)
        .map_err(|error| format!("complex multifrontal LDLH factorization: {error}"))?;
    let solution = factor
        .try_solve(&rhs)
        .map_err(|error| format!("complex multifrontal LDLH solve: {error}"))?;
    let solution_error = solution
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - expected).norm())
        .fold(0.0, f64::max);
    let residual_error = (0..2)
        .map(|row| {
            let value = if row == 0 {
                Complex64::new(3.0, 0.0) * solution[0] + a01 * solution[1]
            } else {
                a10 * solution[0] + Complex64::new(6.0, 0.0) * solution[1]
            };
            (value - rhs[row]).norm()
        })
        .fold(0.0, f64::max);
    Ok((solution_error, residual_error))
}

fn run_checked_rejections() -> Result<f64, String> {
    let mut malformed =
        CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 1.0]);
    malformed.col_indices[0] = 2;
    if MultifrontalLdltFactorization::factorize(&malformed).is_ok() {
        return Err("multifrontal LDLT accepted an out-of-range CSR index".into());
    }

    let identity = CsrMatrix::<f64>::identity(2);
    let factor = MultifrontalLdltFactorization::factorize(&identity)
        .map_err(|error| format!("identity factorization: {error}"))?;
    if factor.try_solve(&[f64::NAN, 1.0]).is_ok() {
        return Err("multifrontal LDLT accepted a non-finite RHS".into());
    }
    Ok(2.0)
}

fn main() {
    let started = Instant::now();
    let result = (|| {
        let (real_solution_error, real_residual_error, two_by_two_check) = run_real()?;
        let (complex_solution_error, complex_residual_error) = run_complex()?;
        let rejection_checks = run_checked_rejections()?;
        Ok::<_, String>((
            real_solution_error,
            real_residual_error,
            two_by_two_check,
            complex_solution_error,
            complex_residual_error,
            rejection_checks,
        ))
    })();

    match result {
        Ok((
            real_solution_error,
            real_residual_error,
            two_by_two_check,
            complex_solution_error,
            complex_residual_error,
            rejection_checks,
        )) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("real_solution_error", real_solution_error)
                .metric("real_residual_error", real_residual_error)
                .metric("two_by_two_pivot_check", two_by_two_check)
                .metric("complex_solution_error", complex_solution_error)
                .metric("complex_residual_error", complex_residual_error)
                .metric("typed_rejection_checks", rejection_checks)
                .maximum("real_solution_error", 1.0e-12)
                .maximum("real_residual_error", 1.0e-12)
                .minimum("two_by_two_pivot_check", 1.0)
                .maximum("complex_solution_error", 1.0e-12)
                .maximum("complex_residual_error", 1.0e-12)
                .minimum("typed_rejection_checks", 2.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            eprintln!("MULTIFRONTAL LDLT REFERENCE FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
