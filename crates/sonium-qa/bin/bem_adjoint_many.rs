//! Release-backed high-level BEM multi-RHS Hermitian-adjoint contract.

use num_complex::Complex64;
use sonium_bem::core::{AssemblyMethod, BemProblem, BemSolver, SolverMethod};
use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.adjoint-many.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    matrix_dimension: usize,
    rhs_count: usize,
    maximum_dense_residual: f64,
    maximum_hmatrix_residual: f64,
}

fn run() -> Result<Metrics, String> {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct);
    let forward = solver.solve(&problem).map_err(|error| error.to_string())?;
    let matrix_dimension = forward.surface_pressure.len();
    let rhs_columns = (0..3)
        .map(|column| {
            (0..matrix_dimension)
                .map(|row| {
                    Complex64::new(
                        0.2 + (row + column) as f64 / 17.0,
                        -(((row + 2 * column) % 5) as f64) / 23.0,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let solutions = solver
        .solve_adjoint_many(&problem, &rhs_columns)
        .map_err(|error| error.to_string())?;
    let named_solutions = solver
        .solve_adjoint_driver_loads(&problem, &rhs_columns)
        .map_err(|error| error.to_string())?;
    if named_solutions.len() != solutions.len()
        || named_solutions
            .iter()
            .zip(&solutions)
            .any(|(named, generic)| {
                named
                    .coefficients()
                    .iter()
                    .zip(generic.coefficients())
                    .any(|(actual, expected)| (*actual - *expected).norm() > 1.0e-12)
            })
    {
        return Err("adjoint driver-load alias diverged from generic batch".to_string());
    }
    if solutions.len() != rhs_columns.len()
        || solutions.iter().any(|solution| {
            solution.num_dofs() != matrix_dimension || !solution.relative_residual().is_finite()
        })
    {
        return Err("adjoint batch returned invalid shape or residual".to_string());
    }

    let hmatrix_solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres)
        .with_max_iterations(500)
        .with_tolerance(1.0e-7)
        .with_compressed_retries(0);
    let hmatrix_solutions = hmatrix_solver
        .solve_adjoint_many(&problem, &rhs_columns)
        .map_err(|error| error.to_string())?;
    if hmatrix_solutions.len() != rhs_columns.len()
        || hmatrix_solutions.iter().any(|solution| {
            solution.num_dofs() != matrix_dimension || !solution.relative_residual().is_finite()
        })
    {
        return Err("H-matrix adjoint batch returned invalid shape or residual".to_string());
    }

    Ok(Metrics {
        matrix_dimension,
        rhs_count: solutions.len(),
        maximum_dense_residual: solutions
            .iter()
            .map(|solution| solution.relative_residual())
            .fold(0.0, f64::max),
        maximum_hmatrix_residual: hmatrix_solutions
            .iter()
            .map(|solution| solution.relative_residual())
            .fold(0.0, f64::max),
    })
}

fn main() {
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(run));
    let result = match outcome {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("matrix_dimension", metrics.matrix_dimension as f64)
            .metric("rhs_count", metrics.rhs_count as f64)
            .metric("maximum_dense_residual", metrics.maximum_dense_residual)
            .metric("maximum_hmatrix_residual", metrics.maximum_hmatrix_residual)
            .minimum("matrix_dimension", 1.0)
            .minimum("rhs_count", 3.0)
            .maximum("maximum_dense_residual", 1.0e-10)
            .maximum("maximum_hmatrix_residual", 1.0e-6)
            .input_checksum("analytic/deterministic-fixture:bem-adjoint-many-v1")
            .reference("independent dense and compressed Hermitian-adjoint residual certification")
            .reference_checksum("analytic/deterministic-fixture:bem-adjoint-many-v1")
            .dimension("3-D rigid sphere; small TBEM/H-matrix operators; three adjoint RHS columns")
            .formulation("production high-level BemSolver dense direct and H-matrix Hermitian-adjoint batches plus named driver/unit-load alias")
            .tolerance("three RHS columns; dense residual <= 1e-10; H-matrix residual <= 1e-6")
            .seed("bem-adjoint-many:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
        Err(_) => QaResult::failed(CASE_ID, "panic while running high-level adjoint batch")
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
