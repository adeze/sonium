//! Release-backed high-level compressed forward frequency-batch BEM contract.

use num_complex::Complex64;
use sonium_bem::core::{AssemblyMethod, BemProblem, BemSolver, SolverMethod};
use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.forward-frequency-batch-compressed.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    backend_count: usize,
    frequency_count: usize,
    rhs_count: usize,
    matrix_dimension: usize,
    maximum_residual: f64,
    maximum_coefficient_error: f64,
}

fn run() -> Result<Metrics, String> {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let frequencies = [100.0, 110.0];
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
    let rhs_batches = vec![rhs_columns.clone(), rhs_columns.clone()];
    let reference = direct
        .solve_rhs_frequency_sweep_many(&problem, &frequencies, &rhs_batches)
        .map_err(|error| error.to_string())?;

    let mut maximum_residual = 0.0_f64;
    let mut maximum_coefficient_error = 0.0_f64;
    for assembly_method in [
        AssemblyMethod::Hmatrix,
        AssemblyMethod::Slfmm,
        AssemblyMethod::Mlfmm,
        AssemblyMethod::DirectionalMlfmm,
    ] {
        let solver = BemSolver::new()
            .with_assembly_method(assembly_method)
            .with_solver_method(SolverMethod::Gmres)
            .with_max_iterations(500)
            .with_tolerance(1.0e-7)
            .with_compressed_retries(0);
        let batches = solver
            .solve_rhs_frequency_sweep_many(&problem, &frequencies, &rhs_batches)
            .map_err(|error| format!("{assembly_method:?}: {error}"))?;
        if batches.len() != frequencies.len() {
            return Err(format!(
                "{assembly_method:?}: expected {} frequency batches, got {}",
                frequencies.len(),
                batches.len()
            ));
        }
        for (batch, reference_batch) in batches.iter().zip(reference.iter()) {
            if batch.len() != rhs_columns.len() {
                return Err(format!(
                    "{assembly_method:?}: expected {} RHS columns, got {}",
                    rhs_columns.len(),
                    batch.len()
                ));
            }
            for (solution, reference_solution) in batch.iter().zip(reference_batch.iter()) {
                let residual = solution.relative_residual();
                if !residual.is_finite() || residual > 1.0e-6 {
                    return Err(format!(
                        "{assembly_method:?}: residual {residual:e} exceeds 1e-6"
                    ));
                }
                maximum_residual = maximum_residual.max(residual);
                let coefficient_error = solution
                    .coefficients()
                    .iter()
                    .zip(reference_solution.coefficients())
                    .map(|(actual, expected)| (*actual - *expected).norm())
                    .fold(0.0, f64::max);
                if !coefficient_error.is_finite() || coefficient_error > 5.0e-5 {
                    return Err(format!(
                        "{assembly_method:?}: coefficient error {coefficient_error:e} exceeds 5e-5"
                    ));
                }
                maximum_coefficient_error = maximum_coefficient_error.max(coefficient_error);
            }
        }
    }

    Ok(Metrics {
        backend_count: 4,
        frequency_count: frequencies.len(),
        rhs_count: rhs_columns.len(),
        matrix_dimension: dimension,
        maximum_residual,
        maximum_coefficient_error,
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .metric("backend_count", metrics.backend_count as f64)
            .metric("frequency_count", metrics.frequency_count as f64)
            .metric("rhs_count", metrics.rhs_count as f64)
            .metric("matrix_dimension", metrics.matrix_dimension as f64)
            .metric("maximum_residual", metrics.maximum_residual)
            .metric(
                "maximum_coefficient_error",
                metrics.maximum_coefficient_error,
            )
        .minimum("backend_count", 4.0)
            .minimum("frequency_count", 2.0)
            .minimum("rhs_count", 2.0)
            .minimum("matrix_dimension", 1.0)
            .maximum("maximum_residual", 1.0e-6)
            .maximum("maximum_coefficient_error", 5.0e-5)
            .input_checksum(
                "analytic/deterministic-fixture:bem-forward-frequency-batch-compressed-v1",
            )
            .reference(
            "independent dense forward frequency-batch solve versus H-matrix, SLFMM, MLFMM, and directional-MLFMM",
            )
            .reference_checksum(
                "analytic/deterministic-fixture:bem-forward-frequency-batch-compressed-v1",
            )
            .dimension("3-D rigid sphere; two frequencies; two arbitrary forward RHS columns")
            .formulation("production high-level BemSolver::solve_rhs_frequency_sweep_many")
        .tolerance("four compressed backends including directional-MLFMM; residual <= 1e-6; coefficient error <= 5e-5")
            .seed("bem-forward-frequency-batch-compressed:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
        Err(_) => QaResult::failed(
            CASE_ID,
            "panic while running compressed forward frequency batch",
        )
        .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
    };

    println!(
        "QA_RESULT: {}",
        serde_json::to_string(&result).expect("QA JSON serialization")
    );
    if matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
