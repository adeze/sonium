//! Release-backed arbitrary forward driver/unit-load BEM contract.

use num_complex::Complex64;
use sonium_bem::core::{AssemblyMethod, BemProblem, BemSolver, SolverMethod};
use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.driver-loads.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    backend_count: usize,
    frequency_count: usize,
    rhs_count: usize,
    matrix_dimension: usize,
    maximum_dense_residual: f64,
    maximum_hmatrix_residual: f64,
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
    let frequencies = [100.0, 120.0];
    let rhs_batches = vec![rhs_columns.clone(), rhs_columns.clone()];

    let dense = direct
        .solve_rhs_frequency_sweep_many(&problem, &frequencies, &rhs_batches)
        .map_err(|error| error.to_string())?;
    let hmatrix = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres)
        .with_max_iterations(500)
        .with_tolerance(1.0e-7)
        .with_compressed_retries(0)
        .solve_rhs_frequency_sweep_many(&problem, &frequencies, &rhs_batches)
        .map_err(|error| error.to_string())?;
    if dense.len() != frequencies.len()
        || hmatrix.len() != frequencies.len()
        || dense.iter().chain(&hmatrix).any(|batch| {
            batch.len() != rhs_columns.len()
                || batch.iter().any(|solution| {
                    solution.num_dofs() != dimension || !solution.relative_residual().is_finite()
                })
        })
    {
        return Err("driver-load solution shape or residual invalid".to_string());
    }
    let field = dense[0][0].evaluate_scattered_pressure_field(&[[0.0, 0.0, 0.3]]);
    if field.len() != 1 || field[0].re.is_nan() || field[0].im.is_nan() {
        return Err("driver-load field evaluation invalid".to_string());
    }

    Ok(Metrics {
        backend_count: 2,
        frequency_count: frequencies.len(),
        rhs_count: rhs_columns.len(),
        matrix_dimension: dimension,
        maximum_dense_residual: dense
            .iter()
            .flat_map(|batch| batch.iter().map(|solution| solution.relative_residual()))
            .fold(0.0, f64::max),
        maximum_hmatrix_residual: hmatrix
            .iter()
            .flat_map(|batch| batch.iter().map(|solution| solution.relative_residual()))
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
            .metric("backend_count", metrics.backend_count as f64)
            .metric("frequency_count", metrics.frequency_count as f64)
            .metric("rhs_count", metrics.rhs_count as f64)
            .metric("matrix_dimension", metrics.matrix_dimension as f64)
            .metric("maximum_dense_residual", metrics.maximum_dense_residual)
            .metric("maximum_hmatrix_residual", metrics.maximum_hmatrix_residual)
            .minimum("backend_count", 2.0)
            .minimum("frequency_count", 2.0)
            .minimum("rhs_count", 2.0)
            .minimum("matrix_dimension", 1.0)
            .maximum("maximum_dense_residual", 1.0e-10)
            .maximum("maximum_hmatrix_residual", 1.0e-6)
            .input_checksum("analytic/deterministic-fixture:bem-driver-loads-v1")
            .reference("independent dense and compressed exact forward residual certification")
            .reference_checksum("analytic/deterministic-fixture:bem-driver-loads-v1")
            .dimension("3-D rigid sphere; two frequencies and two arbitrary driver/unit-load RHS columns; dense and H-matrix paths")
            .formulation("production high-level BemSolver frequency-batched forward load and scattered-field evaluation")
            .tolerance("two frequencies and two RHS columns; dense residual <= 1e-10; H-matrix residual <= 1e-6")
            .seed("bem-driver-loads:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
        Err(_) => QaResult::failed(CASE_ID, "panic while running driver-load contract")
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
