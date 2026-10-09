//! Release-backed compressed forward driver/unit-load BEM contract.

use num_complex::Complex64;
use sonium_bem::core::{AssemblyMethod, BemProblem, BemSolver, SolverMethod};
use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.driver-loads-compressed.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    backend_count: usize,
    rhs_count: usize,
    matrix_dimension: usize,
    maximum_residual: f64,
    maximum_exact_reference_residual: f64,
    maximum_coefficient_error: f64,
    maximum_field_error: f64,
}

fn run() -> Result<Metrics, String> {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let direct = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct);
    let dimension = direct
        .solve(&problem)
        .map_err(|error| error.to_string())?
        .surface_pressure
        .len();
    let rhs_columns = (0..2)
        .map(|column| {
            (0..dimension)
                .map(|row| {
                    Complex64::new(
                        0.15 + (row + column) as f64 / 29.0,
                        -(((row + 2 * column) % 6) as f64) / 31.0,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let reference = direct
        .solve_rhs_many(&problem, &rhs_columns)
        .map_err(|error| error.to_string())?;

    let mut maximum_residual = 0.0_f64;
    let mut maximum_exact_reference_residual = 0.0_f64;
    let mut maximum_coefficient_error = 0.0_f64;
    let mut maximum_field_error = 0.0_f64;
    for assembly_method in [
        AssemblyMethod::Hmatrix,
        AssemblyMethod::Slfmm,
        AssemblyMethod::Mlfmm,
    ] {
        let solver = BemSolver::new()
            .with_assembly_method(assembly_method)
            .with_solver_method(SolverMethod::Gmres)
            .with_max_iterations(500)
            .with_tolerance(1.0e-7)
            .with_compressed_exact_residual_certification(128, 1.0e-5)
            .with_compressed_retries(0);
        let solutions = solver
            .solve_rhs_many(&problem, &rhs_columns)
            .map_err(|error| format!("{assembly_method:?} forward loads: {error}"))?;
        if solutions.len() != rhs_columns.len() {
            return Err(format!(
                "{assembly_method:?} returned {} solutions, expected {}",
                solutions.len(),
                rhs_columns.len()
            ));
        }
        for (solution, reference_solution) in solutions.iter().zip(reference.iter()) {
            if solution.num_dofs() != dimension
                || !solution.relative_residual().is_finite()
                || solution.relative_residual() > 1.0e-6
            {
                return Err(format!(
                    "{assembly_method:?} returned invalid residual or dimension"
                ));
            }
            if solution
                .coefficients()
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                return Err(format!(
                    "{assembly_method:?} returned non-finite coefficients"
                ));
            }
            maximum_residual = maximum_residual.max(solution.relative_residual());
            let exact_residual = solution.exact_reference_residual().ok_or_else(|| {
                format!("{assembly_method:?} missing exact-reference residual certificate")
            })?;
            if !exact_residual.is_finite() {
                return Err(format!(
                    "{assembly_method:?} exact-reference residual was not finite"
                ));
            }
            maximum_exact_reference_residual = maximum_exact_reference_residual.max(exact_residual);
            let coefficient_error = solution
                .coefficients()
                .iter()
                .zip(reference_solution.coefficients())
                .map(|(actual, expected)| (*actual - *expected).norm())
                .fold(0.0, f64::max);
            maximum_coefficient_error = maximum_coefficient_error.max(coefficient_error);
            let points = [[0.0, 0.0, 0.3], [0.2, -0.1, 0.4]];
            let field = solution.evaluate_scattered_pressure_field(&points);
            let reference_field = reference_solution.evaluate_scattered_pressure_field(&points);
            if field.len() != points.len()
                || field
                    .iter()
                    .any(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                return Err(format!(
                    "{assembly_method:?} field evaluation was not finite"
                ));
            }
            let field_error = field
                .iter()
                .zip(reference_field.iter())
                .map(|(actual, expected)| (*actual - *expected).norm())
                .fold(0.0, f64::max);
            maximum_field_error = maximum_field_error.max(field_error);
            if field_error > 5.0e-5 {
                return Err(format!(
                    "{assembly_method:?} field error {field_error:e} exceeds bound"
                ));
            }
        }
    }

    Ok(Metrics {
        backend_count: 3,
        rhs_count: rhs_columns.len(),
        matrix_dimension: dimension,
        maximum_residual,
        maximum_exact_reference_residual,
        maximum_coefficient_error,
        maximum_field_error,
    })
}

fn main() {
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(run));
    let result = match outcome {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("backend_count", metrics.backend_count as f64)
            .metric("rhs_count", metrics.rhs_count as f64)
            .metric("matrix_dimension", metrics.matrix_dimension as f64)
            .metric("maximum_residual", metrics.maximum_residual)
            .metric(
                "maximum_exact_reference_residual",
                metrics.maximum_exact_reference_residual,
            )
            .metric("maximum_coefficient_error", metrics.maximum_coefficient_error)
            .metric("maximum_field_error", metrics.maximum_field_error)
            .minimum("backend_count", 3.0)
            .minimum("rhs_count", 2.0)
            .minimum("matrix_dimension", 1.0)
            .maximum("maximum_residual", 1.0e-6)
            .maximum("maximum_exact_reference_residual", 1.0e-5)
            .maximum("maximum_coefficient_error", 5.0e-5)
            .maximum("maximum_field_error", 5.0e-5)
            .input_checksum("analytic/deterministic-fixture:bem-driver-loads-compressed-v1")
            .reference(
                "independent dense forward residual and coefficient certification for H-matrix, SLFMM, and MLFMM",
            )
            .reference_checksum("analytic/deterministic-fixture:bem-driver-loads-compressed-v1")
            .dimension("3-D rigid sphere; two arbitrary driver/unit-load RHS columns; H-matrix, SLFMM, and MLFMM")
            .formulation("production high-level BemSolver compressed forward driver-load dispatch")
            .tolerance(
                "three compressed backends; residual <= 1e-6; exact-reference residual <= 1e-5; coefficient and field error <= 5e-5",
            )
            .seed("bem-driver-loads-compressed:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
        Err(_) => QaResult::failed(CASE_ID, "panic while running compressed driver loads")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
    };

    println!(
        "QA_RESULT: {}",
        serde_json::to_string(&result).expect("QA JSON serialization")
    );
    if result.status == sonium_qa::QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
