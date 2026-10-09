//! `numcalc` — minimal sonium-bem replacement for the Mesh2HRTF NumCalc CLI.
//!
//! Reads a Mesh2HRTF project (path to its `NC.inp`), runs the BEM frequency
//! sweep, and writes a JSON file (`be.out.json`) with the per-frequency
//! surface and evaluation-grid pressure.
//!
//! ```text
//! cargo run -p sonium-bem --features cli --bin numcalc -- path/to/Project/NC.inp
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use sonium_bem::numcalc::{load_project, run_project};

#[derive(Parser, Debug)]
#[command(
    name = "numcalc",
    about = "sonium-bem replacement for the Mesh2HRTF NumCalc CLI"
)]
struct Args {
    /// Path to the project's NC.inp file.
    nc_inp: PathBuf,

    /// First frequency-curve index to solve (0-based, inclusive).
    #[arg(long, default_value_t = 0)]
    istart: usize,

    /// One past the last frequency index to solve (defaults to all).
    #[arg(long)]
    iend: Option<usize>,

    /// Maximum iterations per frequency.
    #[arg(long, default_value_t = 250)]
    nitermax: usize,

    /// Solver tolerance.
    #[arg(long, default_value_t = 1e-6)]
    tolerance: f64,

    /// Output JSON path (defaults to `<NC.inp parent>/be.out.json`).
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let project = match load_project(&args.nc_inp) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("failed to load project: {e}");
            return ExitCode::FAILURE;
        }
    };
    let n_surface = project
        .elements
        .iter()
        .filter(|e| !e.property.is_evaluation())
        .count();
    log::info!(
        "Loaded project: {} surface elements, {} eval points, {} frequencies",
        n_surface,
        project.evaluation_indices.len(),
        project.frequencies.len()
    );

    let out_path = args.out.unwrap_or_else(|| {
        args.nc_inp
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join("be.out.json")
    });

    let output = match run_project(
        &project,
        Some(args.istart),
        args.iend,
        args.nitermax,
        args.tolerance,
    ) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("sweep failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    if let Err(e) = output.write_json(&out_path) {
        eprintln!("failed to write {}: {e}", out_path.display());
        return ExitCode::FAILURE;
    }
    log::info!(
        "Wrote {} ({} frequencies, {} surface DOFs)",
        out_path.display(),
        output.frequencies.len(),
        output.num_surface_dofs
    );
    ExitCode::SUCCESS
}
