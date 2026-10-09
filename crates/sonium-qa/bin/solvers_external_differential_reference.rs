//! Release-facing differential solver reference against a live PETSc binary.
//!
//! The fixture is deliberately small and deterministic so the QA record can
//! compare the same MatrixMarket input, tolerance, and stopping policy across
//! Sonium and PETSc.  A missing or unusable PETSc binary is reported as an
//! explicit skip (or a failure when `SONIUM_REQUIRE_PETSC=1`), never as a
//! passing correctness result.

use serde_json::Value;
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::{CsrMatrix, GmresConfig, IluPreconditioner, gmres, try_gmres_preconditioned};
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::Instant;
use tempfile::NamedTempFile;

const CASE_ID: &str = "solvers.external-differential-reference.v1";
const TOLERANCE: f64 = 1.0e-10;
const MAX_ITERATIONS: usize = 500;

#[derive(Debug)]
enum RunError {
    Unavailable(String),
    Failed(String),
}

struct Fixture {
    name: &'static str,
    size: usize,
    entries: Vec<(usize, usize, f64)>,
}

impl Fixture {
    fn rhs(&self) -> Vec<f64> {
        vec![1.0; self.size]
    }

    fn csr(&self) -> CsrMatrix<f64> {
        CsrMatrix::from_triplets(self.size, self.size, self.entries.clone())
    }

    fn write_matrix_market(&self) -> Result<NamedTempFile, RunError> {
        let mut file = NamedTempFile::new()
            .map_err(|error| RunError::Failed(format!("create MatrixMarket fixture: {error}")))?;
        writeln!(file, "%%MatrixMarket matrix coordinate real general")
            .map_err(|error| RunError::Failed(format!("write MatrixMarket header: {error}")))?;
        writeln!(file, "% Sonium deterministic PETSc differential fixture")
            .map_err(|error| RunError::Failed(format!("write MatrixMarket comment: {error}")))?;
        writeln!(file, "{} {} {}", self.size, self.size, self.entries.len())
            .map_err(|error| RunError::Failed(format!("write MatrixMarket dimensions: {error}")))?;
        for &(row, column, value) in &self.entries {
            writeln!(file, "{} {} {:.17e}", row + 1, column + 1, value)
                .map_err(|error| RunError::Failed(format!("write MatrixMarket entry: {error}")))?;
        }
        file.flush()
            .map_err(|error| RunError::Failed(format!("flush MatrixMarket fixture: {error}")))?;
        Ok(file)
    }
}

fn poisson_fixture(size: usize) -> Fixture {
    let mut entries = Vec::with_capacity(3 * size - 2);
    for row in 0..size {
        entries.push((row, row, 2.0));
        if row > 0 {
            entries.push((row, row - 1, -1.0));
        }
        if row + 1 < size {
            entries.push((row, row + 1, -1.0));
        }
    }
    Fixture {
        name: "poisson_spd_48",
        size,
        entries,
    }
}

fn convection_diffusion_fixture(size: usize) -> Fixture {
    let mut entries = Vec::with_capacity(3 * size - 2);
    for row in 0..size {
        entries.push((row, row, 2.4 + row as f64 * 0.002));
        if row > 0 {
            entries.push((row, row - 1, -1.0));
        }
        if row + 1 < size {
            entries.push((row, row + 1, -0.22));
        }
    }
    Fixture {
        name: "convection_diffusion_48",
        size,
        entries,
    }
}

fn helmholtz_fixture(size: usize) -> Fixture {
    let mut entries = Vec::with_capacity(3 * size - 2);
    for row in 0..size {
        entries.push((row, row, -1.4 + row as f64 * 0.001));
        if row > 0 {
            entries.push((row, row - 1, 1.0));
        }
        if row + 1 < size {
            entries.push((row, row + 1, 1.0));
        }
    }
    Fixture {
        name: "helmholtz_indefinite_48",
        size,
        entries,
    }
}

fn saddle_point_fixture() -> Fixture {
    Fixture {
        name: "saddle_point_3",
        size: 3,
        entries: vec![
            (0, 0, 4.0),
            (0, 2, 1.0),
            (1, 1, 3.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
            (2, 1, 1.0),
        ],
    }
}

fn petsc_binary() -> Result<String, RunError> {
    let path = std::env::var("SONIUM_PETSC_REFERENCE")
        .unwrap_or_else(|_| "crates/sonium-solvers/benches/ffi/petsc_reference".to_string());
    if !Path::new(&path).is_file() {
        return Err(RunError::Unavailable(format!(
            "PETSc reference binary is not a file: {path}"
        )));
    }
    Ok(path)
}

fn run_petsc(
    fixture: &Fixture,
    matrix_path: &Path,
    binary: &str,
    preconditioner: &str,
) -> Result<(f64, usize, f64, f64, f64), RunError> {
    let output = Command::new(binary)
        .arg(matrix_path)
        .arg("gmres")
        .arg(preconditioner)
        .arg(format!("{TOLERANCE:.17e}"))
        .arg(MAX_ITERATIONS.to_string())
        // PETSc's default TCP probe is not needed for a single-process QA
        // comparison and is rejected by restricted CI sandboxes.
        .env("OMPI_MCA_btl", "^tcp")
        .output()
        .map_err(|error| {
            RunError::Unavailable(format!("execute PETSc for {}: {error}", fixture.name))
        })?;
    if !output.status.success() {
        return Err(RunError::Unavailable(format!(
            "PETSc exited {:?} for {}: {}",
            output.status.code(),
            fixture.name,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json_line = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| {
            RunError::Failed(format!(
                "PETSc emitted no JSON result for {}: {}",
                fixture.name,
                String::from_utf8_lossy(&output.stdout).trim()
            ))
        })?;
    let json: Value = serde_json::from_str(json_line).map_err(|error| {
        RunError::Failed(format!("parse PETSc JSON for {}: {error}", fixture.name))
    })?;
    if !json
        .get("converged")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Err(RunError::Failed(format!(
            "PETSc did not converge on {}",
            fixture.name
        )));
    }
    let residual = json
        .get("residual")
        .and_then(Value::as_f64)
        .ok_or_else(|| {
            RunError::Failed(format!("PETSc result has no residual for {}", fixture.name))
        })?;
    let iterations = json
        .get("iterations")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            RunError::Failed(format!(
                "PETSc result has no iterations for {}",
                fixture.name
            ))
        })? as usize;
    if !residual.is_finite() {
        return Err(RunError::Failed(format!(
            "PETSc residual is non-finite for {}",
            fixture.name
        )));
    }
    let timing = |name: &str| {
        json.get(name)
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .ok_or_else(|| {
                RunError::Failed(format!(
                    "PETSc result has no finite {name} for {}",
                    fixture.name
                ))
            })
    };
    Ok((
        residual,
        iterations,
        timing("setup_ms")?,
        timing("solve_ms")?,
        timing("total_ms")?,
    ))
}

fn run_fixture(fixture: &Fixture, binary: &str) -> Result<Vec<(String, f64)>, RunError> {
    run_fixture_with_preconditioner(fixture, binary, "none")
}

fn run_fixture_with_preconditioner(
    fixture: &Fixture,
    binary: &str,
    preconditioner: &str,
) -> Result<Vec<(String, f64)>, RunError> {
    let matrix = fixture.csr();
    let rhs = fixture.rhs();
    let config = GmresConfig {
        max_iterations: MAX_ITERATIONS,
        restart: 24,
        tolerance: TOLERANCE,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 1,
        min_restart: None,
        max_restart: None,
    };
    let sonium_started = Instant::now();
    let sonium = if preconditioner == "ilu" {
        let ilu = IluPreconditioner::try_from_csr(&matrix)
            .map_err(|error| RunError::Failed(format!("ILU setup on {}: {error}", fixture.name)))?;
        try_gmres_preconditioned(&matrix, &ilu, &rhs, &config).map_err(|error| {
            RunError::Failed(format!(
                "checked preconditioned GMRES on {}: {error}",
                fixture.name
            ))
        })?
    } else {
        gmres(&matrix, &rhs, &config)
    };
    let sonium_ms = sonium_started.elapsed().as_secs_f64() * 1.0e3;
    if !sonium.converged || !sonium.residual.is_finite() || sonium.residual > 1.0e-8 {
        return Err(RunError::Failed(format!(
            "Sonium GMRES failed on {}: converged={}, residual={:e}, status={:?}",
            fixture.name, sonium.converged, sonium.residual, sonium.status
        )));
    }
    let matrix_file = fixture.write_matrix_market()?;
    let (petsc_residual, petsc_iterations, petsc_setup_ms, petsc_solve_ms, petsc_total_ms) =
        run_petsc(fixture, matrix_file.path(), binary, preconditioner)?;
    if petsc_solve_ms <= 0.0 || petsc_total_ms <= 0.0 {
        return Err(RunError::Failed(format!(
            "PETSc reported non-positive timing for {}",
            fixture.name
        )));
    }
    let residual_gap = (sonium.residual - petsc_residual).abs();
    let iteration_delta = sonium.iterations.abs_diff(petsc_iterations) as f64;
    let metric_prefix = if preconditioner == "none" {
        fixture.name.to_string()
    } else {
        format!("{}_{}", fixture.name, preconditioner)
    };
    Ok(vec![
        (format!("{metric_prefix}_sonium_residual"), sonium.residual),
        (format!("{metric_prefix}_petsc_residual"), petsc_residual),
        (format!("{metric_prefix}_residual_gap"), residual_gap),
        (
            format!("{metric_prefix}_sonium_iterations"),
            sonium.iterations as f64,
        ),
        (
            format!("{metric_prefix}_petsc_iterations"),
            petsc_iterations as f64,
        ),
        (format!("{metric_prefix}_iteration_delta"), iteration_delta),
        (format!("{metric_prefix}_sonium_ms"), sonium_ms),
        (format!("{metric_prefix}_petsc_setup_ms"), petsc_setup_ms),
        (format!("{metric_prefix}_petsc_solve_ms"), petsc_solve_ms),
        (format!("{metric_prefix}_petsc_total_ms"), petsc_total_ms),
        (
            format!("{metric_prefix}_sonium_to_petsc_solve_ratio"),
            sonium_ms / petsc_solve_ms,
        ),
        (format!("{}_size", fixture.name), fixture.size as f64),
    ])
}

fn run() -> Result<Vec<(String, f64)>, RunError> {
    let binary = petsc_binary()?;
    let mut metrics = Vec::new();
    for fixture in [
        poisson_fixture(48),
        convection_diffusion_fixture(48),
        helmholtz_fixture(48),
        saddle_point_fixture(),
    ] {
        metrics.extend(run_fixture(&fixture, &binary)?);
        if fixture.name != "saddle_point_3" {
            metrics.extend(run_fixture_with_preconditioner(&fixture, &binary, "ilu")?);
        }
    }
    Ok(metrics)
}

fn main() {
    let started = Instant::now();
    let require_petsc = std::env::var("SONIUM_REQUIRE_PETSC").as_deref() == Ok("1");
    let result = match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .requested_backend("cpu+petsc")
                .actual_backend("cpu+petsc")
                .reference("live PETSc GMRES on identical MatrixMarket fixtures (none + ILU lanes)")
                .reference_checksum("analytic/deterministic-fixture:solver-petsc-v1")
                .input_checksum("analytic/deterministic-fixture:solver-petsc-v1")
                .dimension(
                    "48x48 SPD Poisson, 48x48 nonsymmetric convection-diffusion, 48x48 indefinite Helmholtz, 3x3 saddle point",
                )
                .formulation("Sonium GMRES versus PETSc GMRES: unpreconditioned and ILU lanes")
                .tolerance("both residuals <= 1e-8; residual gap <= 1e-8; iteration deltas are diagnostic only")
                .seed("solvers-external-differential-reference:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.metric(name.clone(), value);
                if name.ends_with("residual") || name.ends_with("residual_gap") {
                    result = result.maximum(name, 1.0e-8);
                } else if name.ends_with("_ms") {
                    result = result.minimum(name, 0.0);
                }
            }
            result
        }
        Err(RunError::Unavailable(reason)) if !require_petsc => {
            QaResult::skipped(CASE_ID, reason).requested_backend("cpu+petsc")
        }
        Err(RunError::Unavailable(reason)) => QaResult::failed(CASE_ID, reason),
        Err(RunError::Failed(reason)) => QaResult::failed(CASE_ID, reason),
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if result.status == QaStatus::Passed {
        println!("SUCCESS");
    } else if result.status == QaStatus::Skipped {
        println!("SKIPPED");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
