//! Public compatibility-facade method/initial-guess contract matrix.

use num_complex::Complex64;
use sonium_solvers::{
    AccuracyStatus, CsrMatrix, LinearSolveConfig, LinearSolverType, LinearSystemAssumption,
    SolverStatus, solve_with_assumption, solve_with_guess,
};

fn real_matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(
        4,
        4,
        [
            (0, 0, 6.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 5.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 2, 4.0),
            (2, 3, 1.0),
            (3, 2, 1.0),
            (3, 3, 3.0),
        ]
        .to_vec(),
    )
}

fn complex_identity() -> CsrMatrix<Complex64> {
    CsrMatrix::from_triplets(
        2,
        2,
        [
            (0, 0, Complex64::new(1.0, 0.0)),
            (1, 1, Complex64::new(1.0, 0.0)),
        ]
        .to_vec(),
    )
}

fn config(solver: LinearSolverType) -> LinearSolveConfig {
    LinearSolveConfig {
        solver,
        max_iterations: 64,
        restart: 8,
        tolerance: 1.0e-12,
        ..LinearSolveConfig::default()
    }
}

fn assert_contract<T>(result: &sonium_solvers::LinearSolveResult<T>, tolerance: f64)
where
    T: sonium_solvers::ComplexField,
{
    assert_eq!(result.status, SolverStatus::Converged);
    assert_eq!(result.attainable_accuracy, AccuracyStatus::Attained);
    assert!(result.converged);
    assert!(result.residual.is_finite() && result.residual <= tolerance);
    assert!(result.backward_error.is_finite() && result.backward_error <= tolerance);
}

fn run_real_method(
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
    expected: &[f64],
    solver: LinearSolverType,
) -> Result<f64, String> {
    let result = solve_with_assumption(
        matrix,
        rhs,
        &config(solver),
        LinearSystemAssumption::PositiveDefinite,
    )
    .map_err(|error| format!("{solver:?} compatibility solve failed: {error}"))?;
    assert_contract(&result, 1.0e-10);
    Ok(result
        .x
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max))
}

fn run_scalar_method<T>(
    matrix: &CsrMatrix<T>,
    rhs: &[T],
    solver: LinearSolverType,
    tolerance: f64,
) -> Result<f64, String>
where
    T: sonium_solvers::ComplexField,
    T::Real: num_traits::FromPrimitive + Send + Sync,
{
    let result = sonium_solvers::solve_with_assumption(
        matrix,
        rhs,
        &config(solver),
        LinearSystemAssumption::General,
    )
    .map_err(|error| format!("{solver:?} scalar compatibility solve failed: {error}"))?;
    assert_contract(&result, tolerance);
    Ok(result.residual)
}

pub fn run_case() -> Result<Vec<(&'static str, f64)>, String> {
    let matrix = real_matrix();
    let expected = [1.0, -0.5, 0.75, -1.25];
    let rhs = matrix.matvec(&expected);
    let guess = [0.25, 0.25, 0.25, 0.25];
    let iterative = solve_with_guess(&matrix, &rhs, &guess, &config(LinearSolverType::Gmres))
        .map_err(|error| format!("real solve_with_guess failed: {error}"))?;
    assert_contract(&iterative, 1.0e-10);
    let real_error = iterative
        .x
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);

    let assumed = solve_with_assumption(
        &matrix,
        &rhs,
        &config(LinearSolverType::Cg),
        LinearSystemAssumption::PositiveDefinite,
    )
    .map_err(|error| format!("positive-definite compatibility solve failed: {error}"))?;
    assert_contract(&assumed, 1.0e-10);

    let complex = complex_identity();
    let complex_rhs = [Complex64::new(2.0, -1.0), Complex64::new(-0.5, 3.0)];
    let complex_guess = [Complex64::new(0.1, 0.2), Complex64::new(0.0, 0.0)];
    let complex_result = solve_with_guess(
        &complex,
        &complex_rhs,
        &complex_guess,
        &config(LinearSolverType::Gmres),
    )
    .map_err(|error| format!("complex solve_with_guess failed: {error}"))?;
    assert_contract(&complex_result, 1.0e-10);
    let complex_error = complex_result
        .x
        .iter()
        .zip(complex_rhs)
        .map(|(actual, expected)| (*actual - expected).norm())
        .fold(0.0, f64::max);

    let invalid = solve_with_guess(&matrix, &rhs, &[0.0, 0.0], &config(LinearSolverType::Gmres));
    if !matches!(
        invalid,
        Err(sonium_solvers::LinearSolveError::DimensionMismatch { .. })
    ) {
        return Err(format!(
            "wrong-size initial guess was not rejected: {invalid:?}"
        ));
    }

    let methods = [
        LinearSolverType::Gmres,
        LinearSolverType::GmresJacobi,
        LinearSolverType::GmresIlu,
        LinearSolverType::GmresAmg,
        LinearSolverType::GmresChebyshev,
        LinearSolverType::GmresSchwarz,
        LinearSolverType::FgmresSchwarz,
        LinearSolverType::Cg,
        LinearSolverType::CgAmg,
    ];
    let mut method_error: f64 = 0.0;
    for solver in methods {
        method_error = method_error.max(run_real_method(&matrix, &rhs, &expected, solver)?);
    }

    let f32_matrix = CsrMatrix::from_triplets(
        2,
        2,
        [
            (0, 0, 4.0_f32),
            (0, 1, 1.0_f32),
            (1, 0, 1.0_f32),
            (1, 1, 3.0_f32),
        ]
        .to_vec(),
    );
    let f32_expected = [1.0_f32, -0.5_f32];
    let f32_rhs = f32_matrix.matvec(&f32_expected);
    let f32_residual = run_scalar_method(&f32_matrix, &f32_rhs, LinearSolverType::Gmres, 1.0e-5)?;

    let complex32_matrix = CsrMatrix::from_triplets(
        2,
        2,
        [
            (0, 0, num_complex::Complex32::new(1.0, 0.0)),
            (1, 1, num_complex::Complex32::new(1.0, 0.0)),
        ]
        .to_vec(),
    );
    let complex32_rhs = [
        num_complex::Complex32::new(0.75, -0.5),
        num_complex::Complex32::new(-1.25, 0.25),
    ];
    let complex32_residual = run_scalar_method(
        &complex32_matrix,
        &complex32_rhs,
        LinearSolverType::Gmres,
        1.0e-5,
    )?;

    Ok(vec![
        ("real_guess_solution_error", real_error),
        ("real_guess_residual", iterative.residual),
        ("real_assumption_residual", assumed.residual),
        ("complex_guess_solution_error", complex_error),
        ("complex_guess_residual", complex_result.residual),
        ("facade_method_error", method_error),
        ("facade_method_checks", methods.len() as f64),
        ("f32_residual", f32_residual),
        ("complex32_residual", complex32_residual),
        ("contract_checks", 4.0 + methods.len() as f64 + 2.0),
    ])
}

#[test]
fn compatibility_method_matrix_contracts() {
    let metrics = run_case().expect("compatibility method matrix");
    assert_eq!(metrics.last().map(|(_, value)| *value), Some(15.0));
}
