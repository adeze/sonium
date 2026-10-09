//! Release-backed high-level BEM frequency-batch Hermitian-adjoint contract.

use num_complex::Complex64;
use sonium_bem::core::{AssemblyMethod, BemProblem, BemSolver, SolverMethod};
use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.adjoint-frequency-batch.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    backend_count: usize,
    frequency_count: usize,
    rhs_count: usize,
    matrix_dimension: usize,
    maximum_dense_residual: f64,
    maximum_compressed_residual: f64,
    maximum_exact_reference_residual: f64,
    maximum_coefficient_error: f64,
}

fn run() -> Result<Metrics, String> {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct);
    let forward = solver.solve(&problem).map_err(|error| error.to_string())?;
    let matrix_dimension = forward.surface_pressure.len();
    let frequencies = [100.0, 120.0];
    let rhs_batches = frequencies
        .iter()
        .enumerate()
        .map(|(frequency_index, _)| {
            (0..2)
                .map(|column| {
                    (0..matrix_dimension)
                        .map(|row| {
                            Complex64::new(
                                0.1 + (row + column + 3 * frequency_index) as f64 / 23.0,
                                -(((row + 2 * column + frequency_index) % 7) as f64) / 29.0,
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let dense_batches = solver
        .solve_adjoint_frequency_sweep_many(&problem, &frequencies, &rhs_batches)
        .map_err(|error| error.to_string())?;
    if dense_batches.len() != frequencies.len()
        || dense_batches.iter().any(|batch| {
            batch.len() != 2
                || batch.iter().any(|solution| {
                    solution.num_dofs() != matrix_dimension
                        || !solution.relative_residual().is_finite()
                })
        })
    {
        return Err("frequency adjoint batch returned invalid shape or residual".to_string());
    }

    let backends = [
        AssemblyMethod::Hmatrix,
        AssemblyMethod::Slfmm,
        AssemblyMethod::Mlfmm,
        AssemblyMethod::DirectionalMlfmm,
    ];
    let mut maximum_compressed_residual = 0.0_f64;
    let mut maximum_exact_reference_residual = 0.0_f64;
    let mut maximum_coefficient_error = 0.0_f64;
    for assembly_method in backends {
        let compressed_solver = BemSolver::new()
            .with_assembly_method(assembly_method)
            .with_solver_method(SolverMethod::Gmres)
            .with_max_iterations(500)
            .with_tolerance(1.0e-7)
            .with_compressed_exact_residual_certification(64, 1.0e-4)
            .with_compressed_retries(0);
        let compressed_batches = compressed_solver
            .solve_adjoint_frequency_sweep_many(&problem, &frequencies, &rhs_batches)
            .map_err(|error| format!("{assembly_method:?}: {error}"))?;
        if compressed_batches.len() != frequencies.len()
            || compressed_batches.iter().any(|batch| {
                batch.len() != 2
                    || batch.iter().any(|solution| {
                        solution.num_dofs() != matrix_dimension
                            || !solution.relative_residual().is_finite()
                    })
            })
        {
            return Err(format!(
                "{assembly_method:?} frequency adjoint batch returned invalid shape or residual"
            ));
        }

        for (frequency_index, (compressed_batch, dense_batch)) in compressed_batches
            .iter()
            .zip(dense_batches.iter())
            .enumerate()
        {
            for (rhs_index, (compressed, dense)) in
                compressed_batch.iter().zip(dense_batch.iter()).enumerate()
            {
                let residual = compressed.relative_residual();
                maximum_compressed_residual = maximum_compressed_residual.max(residual);
                if !residual.is_finite() || residual > 1.0e-6 {
                    return Err(format!(
                        "{assembly_method:?} frequency {frequency_index} RHS {rhs_index} residual {residual:e} exceeds 1e-6"
                    ));
                }
                let exact_residual = compressed.exact_reference_residual().ok_or_else(|| {
                    format!(
                        "{assembly_method:?} frequency {frequency_index} RHS {rhs_index} exact-reference residual certificate missing"
                    )
                })?;
                maximum_exact_reference_residual =
                    maximum_exact_reference_residual.max(exact_residual);
                if !exact_residual.is_finite() || exact_residual > 1.0e-4 {
                    return Err(format!(
                        "{assembly_method:?} frequency {frequency_index} RHS {rhs_index} exact-reference residual {exact_residual:e} exceeds 1e-4"
                    ));
                }
                let coefficient_error = compressed
                    .coefficients()
                    .iter()
                    .zip(dense.coefficients())
                    .map(|(actual, expected)| (*actual - *expected).norm())
                    .fold(0.0, f64::max);
                maximum_coefficient_error = maximum_coefficient_error.max(coefficient_error);
                if !coefficient_error.is_finite() || coefficient_error > 1.0e-4 {
                    return Err(format!(
                        "{assembly_method:?} frequency {frequency_index} RHS {rhs_index} coefficient error {coefficient_error:e} exceeds 1e-4"
                    ));
                }
            }
        }
    }

    Ok(Metrics {
        backend_count: backends.len(),
        frequency_count: dense_batches.len(),
        rhs_count: dense_batches.iter().map(Vec::len).sum::<usize>() / dense_batches.len(),
        matrix_dimension,
        maximum_dense_residual: dense_batches
            .iter()
            .flat_map(|batch| batch.iter().map(|solution| solution.relative_residual()))
            .fold(0.0, f64::max),
        maximum_compressed_residual,
        maximum_exact_reference_residual,
        maximum_coefficient_error,
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
            .metric("maximum_compressed_residual", metrics.maximum_compressed_residual)
            .metric(
                "maximum_exact_reference_residual",
                metrics.maximum_exact_reference_residual,
            )
            .metric("maximum_coefficient_error", metrics.maximum_coefficient_error)
            .minimum("backend_count", 4.0)
            .minimum("frequency_count", 2.0)
            .minimum("rhs_count", 2.0)
            .minimum("matrix_dimension", 1.0)
            .maximum("maximum_dense_residual", 1.0e-10)
            .maximum("maximum_compressed_residual", 1.0e-6)
            .maximum("maximum_exact_reference_residual", 1.0e-4)
            .maximum("maximum_coefficient_error", 1.0e-4)
            .input_checksum("analytic/deterministic-fixture:bem-adjoint-frequency-batch-v1")
            .reference("independent dense and compressed exact-TBEM Hermitian-adjoint residual certification for every frequency/RHS column")
            .reference_checksum("analytic/deterministic-fixture:bem-adjoint-frequency-batch-v1")
            .dimension("3-D rigid sphere; two frequencies; two adjoint RHS columns per frequency; four compressed backends")
            .formulation("production high-level BemSolver dense direct and compressed H-matrix/SLFMM/MLFMM/directional-MLFMM frequency-batched Hermitian-adjoint solves")
            .tolerance("two frequencies and two RHS columns per frequency; dense/compressed residual <= 1e-6; independent exact-reference residual <= 1e-4; coefficient error <= 1e-4")
            .seed("bem-adjoint-frequency-batch:cpu:v3")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap()),
        Err(_) => QaResult::failed(CASE_ID, "panic while running frequency adjoint batch")
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
