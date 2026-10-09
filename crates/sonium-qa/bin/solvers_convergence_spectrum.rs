//! Deterministic convergence-spectrum and failure-mode contracts.
//!
//! The fixtures are deliberately small, but each one exercises a different
//! operator class.  The expected solution is generated before the solve so
//! the QA gate checks both the reported convergence metadata and an
//! independently recomputed error.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers::iterative::{
    CgConfig, FgmresConfig, GmresConfig, MinresConfig, cg, fgmres_preconditioned, gmres, minres,
};
use sonium_solvers::preconditioners::DiagonalPreconditioner;
use sonium_solvers::{CsrBuilder, CsrMatrix};
use std::time::Instant;

const TOLERANCE: f64 = 1.0e-10;

fn real_matrix(rows: &[Vec<(usize, f64)>], dimension: usize) -> CsrMatrix<f64> {
    let mut builder = CsrBuilder::new(dimension, dimension);
    for row in rows {
        let mut entries = row.clone();
        entries.sort_unstable_by_key(|(column, _)| *column);
        builder.add_row_entries(entries.into_iter());
    }
    builder.finish()
}

fn complex_matrix(rows: &[Vec<(usize, Complex64)>], dimension: usize) -> CsrMatrix<Complex64> {
    let mut builder = CsrBuilder::new(dimension, dimension);
    for row in rows {
        let mut entries = row.clone();
        entries.sort_unstable_by_key(|(column, _)| *column);
        builder.add_row_entries(entries.into_iter());
    }
    builder.finish()
}

fn real_relative_error(actual: &[f64], expected: &[f64]) -> f64 {
    let (error, reference) = actual.iter().zip(expected).fold(
        (0.0_f64, 0.0_f64),
        |(error, reference), (&actual, &expected)| {
            (
                error.max((actual - expected).abs()),
                reference.max(expected.abs()),
            )
        },
    );
    error / reference.max(1.0)
}

fn complex_relative_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    let (error, reference) = actual.iter().zip(expected).fold(
        (0.0_f64, 0.0_f64),
        |(error, reference), (&actual, &expected)| {
            (
                error.max((actual - expected).norm()),
                reference.max(expected.norm()),
            )
        },
    );
    error / reference.max(1.0)
}

fn run() -> Result<Vec<(String, f64, f64, usize)>, String> {
    let mut records = Vec::new();

    // SPD path: the 1-D Laplacian has a known positive spectrum and is a
    // useful regression for CG's applicability domain and residual contract.
    let n = 32;
    let spd_rows = (0..n)
        .map(|row| {
            let mut entries = vec![(row, 2.0)];
            if row > 0 {
                entries.push((row - 1, -1.0));
            }
            if row + 1 < n {
                entries.push((row + 1, -1.0));
            }
            entries
        })
        .collect::<Vec<_>>();
    let spd = real_matrix(&spd_rows, n);
    let expected = (0..n)
        .map(|index| 1.0 + (index as f64 * 0.17).sin())
        .collect::<Vec<_>>();
    let rhs = spd.matvec(&expected);
    let cg_result = cg(
        &spd,
        &rhs,
        &CgConfig {
            max_iterations: 2 * n,
            tolerance: TOLERANCE,
            print_interval: 0,
        },
    );
    let cg_error = real_relative_error(&cg_result.x, &expected);
    if !cg_result.converged || cg_result.residual > TOLERANCE || cg_error > 10.0 * TOLERANCE {
        return Err(format!(
            "CG SPD spectrum contract failed: converged={}, residual={:.3e}, error={:.3e}",
            cg_result.converged, cg_result.residual, cg_error
        ));
    }
    records.push((
        "cg_spd_spectrum".into(),
        cg_result.residual,
        cg_error,
        cg_result.iterations,
    ));

    // Symmetric-indefinite path: a diagonal matrix with four distinct signed
    // eigenvalues should terminate in at most four MINRES steps.
    let indefinite_diagonal = [-3.0, -1.0, 1.0, 3.0, -3.0, -1.0, 1.0, 3.0];
    let indefinite = real_matrix(
        &indefinite_diagonal
            .iter()
            .enumerate()
            .map(|(index, &value)| vec![(index, value)])
            .collect::<Vec<_>>(),
        indefinite_diagonal.len(),
    );
    let expected = vec![1.0, -0.5, 0.75, -1.25, 1.5, -1.75, 2.0, -2.25];
    let rhs = indefinite.matvec(&expected);
    let minres_result = minres(
        &indefinite,
        &rhs,
        &MinresConfig {
            max_iterations: indefinite_diagonal.len(),
            tolerance: TOLERANCE,
            print_interval: 0,
            condition_limit: 1.0e14,
        },
    );
    let minres_error = real_relative_error(&minres_result.x, &expected);
    if !minres_result.converged
        || minres_result.residual > TOLERANCE
        || minres_error > 10.0 * TOLERANCE
    {
        return Err(format!(
            "MINRES symmetric-indefinite contract failed: converged={}, residual={:.3e}, error={:.3e}, status={:?}",
            minres_result.converged, minres_result.residual, minres_error, minres_result.status
        ));
    }
    records.push((
        "minres_symmetric_indefinite".into(),
        minres_result.residual,
        minres_error,
        minres_result.iterations,
    ));

    // Nonnormal convection-diffusion path.  The asymmetric stencil makes a
    // CG/MINRES assumption invalid while remaining easy to solve accurately.
    let n = 24;
    let convection_rows = (0..n)
        .map(|row| {
            let mut entries = vec![(row, 2.4 + row as f64 * 0.002)];
            if row > 0 {
                entries.push((row - 1, -1.0));
            }
            if row + 1 < n {
                entries.push((row + 1, -0.22));
            }
            entries
        })
        .collect::<Vec<_>>();
    let convection = real_matrix(&convection_rows, n);
    let expected = (0..n)
        .map(|index| 0.5 + (index as f64 * 0.31).cos())
        .collect::<Vec<_>>();
    let rhs = convection.matvec(&expected);
    let gmres_result = gmres(
        &convection,
        &rhs,
        &GmresConfig {
            max_iterations: 160,
            restart: 12,
            tolerance: TOLERANCE,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
            min_restart: None,
            max_restart: None,
        },
    );
    let gmres_error = real_relative_error(&gmres_result.x, &expected);
    if !gmres_result.converged
        || gmres_result.residual > TOLERANCE
        || gmres_error > 10.0 * TOLERANCE
    {
        return Err(format!(
            "GMRES convection-diffusion contract failed: converged={}, residual={:.3e}, error={:.3e}, status={:?}",
            gmres_result.converged, gmres_result.residual, gmres_error, gmres_result.status
        ));
    }
    records.push((
        "gmres_nonnormal_convection".into(),
        gmres_result.residual,
        gmres_error,
        gmres_result.iterations,
    ));

    // Complex nonnormal Helmholtz-like path.  FGMRES uses the diagonal
    // preconditioner but still finalizes against the original operator.
    let n = 20;
    let helmholtz_rows = (0..n)
        .map(|row| {
            let mut entries = vec![(row, Complex64::new(-1.35 + row as f64 * 0.003, 0.32))];
            if row > 0 {
                entries.push((row - 1, Complex64::new(-1.0, 0.08)));
            }
            if row + 1 < n {
                entries.push((row + 1, Complex64::new(-0.38, -0.04)));
            }
            entries
        })
        .collect::<Vec<_>>();
    let helmholtz = complex_matrix(&helmholtz_rows, n);
    let expected = (0..n)
        .map(|index| Complex64::new((index as f64 * 0.23).sin(), (index as f64 * 0.19).cos()))
        .collect::<Vec<_>>();
    let rhs = helmholtz.matvec(&expected);
    let preconditioner = DiagonalPreconditioner::try_from_csr(&helmholtz)
        .map_err(|error| format!("Helmholtz preconditioner setup failed: {error}"))?;
    let fgmres_result = fgmres_preconditioned(
        &helmholtz,
        &preconditioner,
        &rhs,
        None,
        &FgmresConfig {
            max_iterations: 160,
            restart: 12,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
        },
    );
    let fgmres_error = complex_relative_error(&fgmres_result.x, &expected);
    if !fgmres_result.converged
        || fgmres_result.residual > TOLERANCE
        || fgmres_error > 10.0 * TOLERANCE
    {
        return Err(format!(
            "FGMRES Helmholtz contract failed: converged={}, residual={:.3e}, error={:.3e}, status={:?}",
            fgmres_result.converged, fgmres_result.residual, fgmres_error, fgmres_result.status
        ));
    }
    records.push((
        "fgmres_complex_helmholtz".into(),
        fgmres_result.residual,
        fgmres_error,
        fgmres_result.iterations,
    ));

    // Happy breakdown should be reported as a clean one-step convergence.
    let identity = real_matrix(
        &(0..4).map(|index| vec![(index, 1.0)]).collect::<Vec<_>>(),
        4,
    );
    let ones = vec![1.0; 4];
    let happy = gmres(&identity, &ones, &GmresConfig::for_small_problems());
    if !happy.converged || happy.iterations > 2 || happy.residual > TOLERANCE {
        return Err(format!(
            "GMRES happy-breakdown contract failed: converged={}, iterations={}, residual={:.3e}, status={:?}",
            happy.converged, happy.iterations, happy.residual, happy.status
        ));
    }
    records.push((
        "gmres_happy_breakdown".into(),
        happy.residual,
        0.0,
        happy.iterations,
    ));

    // A zero operator with a non-zero RHS must stop as a typed non-converged
    // breakdown/stagnation, never panic or claim a partial solution converged.
    let zero = real_matrix(&(0..4).map(|_| Vec::new()).collect::<Vec<_>>(), 4);
    let stagnated = gmres(&zero, &ones, &GmresConfig::for_small_problems());
    if stagnated.converged
        || !matches!(
            stagnated.status,
            sonium_solvers::traits::SolverStatus::Breakdown
                | sonium_solvers::traits::SolverStatus::Stagnated
                | sonium_solvers::traits::SolverStatus::MaxIterationsReached
                | sonium_solvers::traits::SolverStatus::NumericalError
        )
    {
        return Err(format!(
            "GMRES stagnation contract failed: converged={}, residual={:.3e}, status={:?}",
            stagnated.converged, stagnated.residual, stagnated.status
        ));
    }
    records.push((
        "gmres_stagnation".into(),
        stagnated.residual,
        0.0,
        stagnated.iterations,
    ));

    Ok(records)
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(records) => {
            let max_residual = records
                .iter()
                .filter(|(name, _, _, _)| name != "gmres_stagnation")
                .map(|(_, residual, _, _)| *residual)
                .filter(|value| value.is_finite())
                .fold(0.0_f64, f64::max);
            let max_error = records
                .iter()
                .map(|(_, _, error, _)| *error)
                .fold(0.0, f64::max);
            let total_iterations = records
                .iter()
                .map(|(_, _, _, iterations)| *iterations)
                .sum::<usize>();
            let mut result = QaResult::passed("solvers.convergence-spectrum.v1")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed("fixed-spectrum-fixtures")
                .metric("max_true_residual", max_residual)
                .metric("max_solution_error", max_error)
                .metric("total_iterations", total_iterations as f64)
                .metric("case_count", records.len() as f64)
                .maximum("max_true_residual", TOLERANCE)
                .maximum("max_solution_error", 10.0 * TOLERANCE)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, residual, error, iterations) in records {
                result = result
                    .metric(format!("{name}.residual"), residual)
                    .metric(format!("{name}.error"), error)
                    .metric(format!("{name}.iterations"), iterations as f64);
            }
            match result.to_json_line() {
                Ok(json) => println!("QA_RESULT: {json}"),
                Err(error) => {
                    eprintln!("QA result serialization failed: {error}");
                    std::process::exit(1);
                }
            }
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed("solvers.convergence-spectrum.v1", error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed("fixed-spectrum-fixtures")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            if let Ok(json) = result.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            eprintln!("SOLVER CONVERGENCE SPECTRUM FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
