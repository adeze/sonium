//! Sustained FEM/BEM solver consumer workload.
//!
//! This packet deliberately exercises the production assembly-to-solver
//! path rather than a synthetic matrix fixture.  It repeats a small FEM
//! Helmholtz frequency sweep and a small BEM TBEM frequency sweep, recording
//! assembly/solve timings, iteration counts, and independently recomputed
//! true residuals.  It is a CPU workload gate, not an external-solver or
//! performance-competitiveness claim.

use num_complex::Complex64;
use num_traits::ToPrimitive;
use sonium_bem::core::assembly::build_tbem_system;
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::solver::{DenseOperator, GmresConfig, solve_gmres};
use sonium_bem::core::types::{BoundaryCondition, PhysicsParams};
use sonium_fem::assembly::HelmholtzProblem;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::{DirichletBC, apply_dirichlet};
use sonium_fem::mesh::unit_square_triangles;
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::iterative::gmres;
use sonium_solvers::{CsrMatrix, GmresConfig as SoniumGmresConfig};
use std::time::Instant;

const CASE_ID: &str = "solvers.fem-bem-workload.v1";
// Keep a tenfold margin over the observed cross-frequency residuals while
// still requiring a strict true-residual solve on every consumer path.
const TOLERANCE: f64 = 1.0e-7;

#[derive(Debug)]
struct WorkloadMetrics {
    values: Vec<(String, f64)>,
}

fn relative_residual<T>(matrix: &CsrMatrix<T>, x: &[T], rhs: &[T]) -> f64
where
    T: sonium_solvers::ComplexField,
{
    if x.len() != matrix.num_cols || rhs.len() != matrix.num_rows || rhs.is_empty() {
        return f64::INFINITY;
    }
    let ax = matrix.matvec(x);
    let mut residual_sq = 0.0;
    let mut rhs_sq = 0.0;
    for (computed, expected) in ax.iter().zip(rhs) {
        residual_sq += (*computed - *expected)
            .norm_sqr()
            .to_f64()
            .unwrap_or(f64::INFINITY);
        rhs_sq += expected.norm_sqr().to_f64().unwrap_or(f64::INFINITY);
    }
    residual_sq.sqrt() / rhs_sq.sqrt().max(1.0e-30)
}

fn assemble_fem_matrix(problem: &HelmholtzProblem) -> CsrMatrix<Complex64> {
    problem.matrix.to_csr()
}

fn run_fem_sweep(values: &mut Vec<(String, f64)>) -> Result<(), String> {
    let mesh = unit_square_triangles(8);
    let frequencies = [1.8_f64, 2.0, 2.2];
    let plane_wave = |k: f64| {
        move |x: f64, y: f64, _z: f64| {
            let phase = k * (x + y) / 2.0_f64.sqrt();
            Complex64::new(phase.cos(), phase.sin())
        }
    };
    let config = SoniumGmresConfig {
        max_iterations: 400,
        restart: 32,
        tolerance: TOLERANCE,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 1,
        min_restart: None,
        max_restart: None,
    };

    let mut total_assembly_ms = 0.0_f64;
    let mut total_solve_ms = 0.0_f64;
    let mut max_residual = 0.0_f64;
    let mut max_error = 0.0_f64;
    let mut total_iterations = 0_usize;

    for (index, wavenumber) in frequencies.into_iter().enumerate() {
        let source = |_x: f64, _y: f64, _z: f64| Complex64::new(0.0, 0.0);
        let assembly_started = Instant::now();
        let mut problem = HelmholtzProblem::assemble(
            &mesh,
            PolynomialDegree::P1,
            Complex64::new(wavenumber, 0.0),
            source,
        );
        let exact = plane_wave(wavenumber);
        let bcs: Vec<_> = (1..=4).map(|tag| DirichletBC::new(tag, exact)).collect();
        apply_dirichlet(&mut problem, &mesh, &bcs);
        let matrix = assemble_fem_matrix(&problem);
        total_assembly_ms += assembly_started.elapsed().as_secs_f64() * 1.0e3;

        let solve_started = Instant::now();
        let solution = gmres(&matrix, &problem.rhs, &config);
        total_solve_ms += solve_started.elapsed().as_secs_f64() * 1.0e3;
        let residual = relative_residual(&matrix, &solution.x, &problem.rhs);
        if !solution.converged || !residual.is_finite() || residual > TOLERANCE {
            return Err(format!(
                "FEM frequency {wavenumber} failed: converged={}, reported={:.3e}, true={:.3e}",
                solution.converged, solution.residual, residual
            ));
        }

        let mut error_sq = 0.0;
        let mut exact_sq = 0.0;
        for (node, value) in mesh.nodes.iter().zip(&solution.x) {
            let reference = exact(node.x, node.y, node.z);
            error_sq += (*value - reference).norm_sqr();
            exact_sq += reference.norm_sqr();
        }
        let error = (error_sq / exact_sq.max(1.0e-30)).sqrt();
        max_residual = max_residual.max(residual);
        max_error = max_error.max(error);
        total_iterations += solution.iterations;
        values.push((format!("fem_frequency_{index}_residual"), residual));
        values.push((
            format!("fem_frequency_{index}_iterations"),
            solution.iterations as f64,
        ));
        values.push((format!("fem_frequency_{index}_error"), error));
    }

    values.push(("fem_max_residual".into(), max_residual));
    values.push(("fem_max_solution_error".into(), max_error));
    values.push(("fem_total_iterations".into(), total_iterations as f64));
    values.push(("fem_assembly_ms".into(), total_assembly_ms));
    values.push(("fem_solve_ms".into(), total_solve_ms));
    Ok(())
}

fn run_bem_sweep(values: &mut Vec<(String, f64)>) -> Result<(), String> {
    let mesh = generate_icosphere_mesh(0.1, 1);
    let mut elements = mesh.elements.clone();
    for (index, element) in elements.iter_mut().enumerate() {
        let drive = 1.0e-3 * (1.0 + (index as f64 * 0.37).sin());
        element.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(drive, 0.0)]);
        element.dof_addresses = vec![index];
    }

    let frequencies = [350.0_f64, 500.0, 650.0];
    let config = GmresConfig {
        max_iterations: 300,
        restart: 32,
        tolerance: TOLERANCE,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 1,
        min_restart: None,
        max_restart: None,
    };
    let mut total_assembly_ms = 0.0_f64;
    let mut total_solve_ms = 0.0_f64;
    let mut max_residual = 0.0_f64;
    let mut total_iterations = 0_usize;

    for (index, frequency) in frequencies.into_iter().enumerate() {
        let physics = PhysicsParams::new(frequency, 343.0, 1.21, false);
        let assembly_started = Instant::now();
        let system = build_tbem_system(&elements, &mesh.nodes, &physics);
        total_assembly_ms += assembly_started.elapsed().as_secs_f64() * 1.0e3;
        let operator = DenseOperator::new(system.matrix.clone());

        let solve_started = Instant::now();
        let solution = solve_gmres(&operator, &system.rhs, &config);
        total_solve_ms += solve_started.elapsed().as_secs_f64() * 1.0e3;
        let ax = system.matrix.matvec(&solution.x);
        let mut residual_sq: f64 = 0.0;
        let mut rhs_sq: f64 = 0.0;
        for (computed, expected) in ax.iter().zip(&system.rhs) {
            residual_sq += (*computed - *expected).norm_sqr();
            rhs_sq += expected.norm_sqr();
        }
        let residual = residual_sq.sqrt() / rhs_sq.sqrt().max(1.0e-30);
        if !solution.converged || !residual.is_finite() || residual > TOLERANCE {
            return Err(format!(
                "BEM frequency {frequency} failed: converged={}, reported={:.3e}, true={:.3e}",
                solution.converged, solution.residual, residual
            ));
        }
        max_residual = max_residual.max(residual);
        total_iterations += solution.iterations;
        values.push((format!("bem_frequency_{index}_residual"), residual));
        values.push((
            format!("bem_frequency_{index}_iterations"),
            solution.iterations as f64,
        ));
    }

    values.push(("bem_max_residual".into(), max_residual));
    values.push(("bem_total_iterations".into(), total_iterations as f64));
    values.push(("bem_assembly_ms".into(), total_assembly_ms));
    values.push(("bem_solve_ms".into(), total_solve_ms));
    Ok(())
}

fn run() -> Result<WorkloadMetrics, String> {
    let mut values = Vec::new();
    run_fem_sweep(&mut values)?;
    run_bem_sweep(&mut values)?;
    Ok(WorkloadMetrics { values })
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok(metrics) => metrics.values.into_iter().fold(
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("production FEM Helmholtz and BEM TBEM consumers; independently recomputed true residuals")
                .reference_checksum("analytic/deterministic-fixture:solvers-fem-bem-workload-v1")
                .input_checksum("analytic/deterministic-fixture:solvers-fem-bem-workload-v1")
                .dimension("FEM 8x8 unit-square P1 mesh, 3 frequencies; BEM 80-panel icosphere, 3 frequencies")
                .formulation("assembled FEM CSR GMRES and dense TBEM GMRES frequency sweeps")
                .tolerance("all six solves converge with independently recomputed relative residual <= 1e-7")
                .seed("solvers-fem-bem-workload:cpu:v1"),
            |result, (name, value)| {
                let result = result.metric(name.clone(), value);
                if name.contains("residual") {
                    result.maximum(name, TOLERANCE)
                } else if name == "fem_max_solution_error" {
                    result.maximum(name, 1.0e-2)
                } else if name.ends_with("_ms") {
                    result.minimum(name, 0.0)
                } else {
                    result
                }
            },
        ),
        Err(error) => QaResult::failed(CASE_ID, error),
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    match result.status {
        QaStatus::Passed => println!("SUCCESS"),
        QaStatus::Skipped => println!("SKIPPED"),
        QaStatus::Failed => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
