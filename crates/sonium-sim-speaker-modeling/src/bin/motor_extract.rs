//! `motor-extract`: extract BL(x) and Le(f) curves from a motor geometry definition.
//!
//! Reads a JSON motor geometry file, runs magnetostatic FEM for BL(x), and
//! outputs BL plus passive semi-analytical Le(f)/Re(f) curves as JSON.

use std::path::PathBuf;

use clap::Parser;

use sonium_sim_speaker_modeling::motor_extraction::{
    ExtractionConfig, MotorGeometry, extract_motor_parameters,
};

#[derive(Parser)]
#[command(
    name = "motor-extract",
    about = "Extract FEM BL(x) and passive approximate Le(f)/Re(f) from motor geometry"
)]
struct Cli {
    /// Path to motor geometry JSON file.
    #[arg(short, long)]
    geometry: PathBuf,

    /// Output JSON file for extracted parameters.
    #[arg(short, long, default_value = "motor_parameters.json")]
    output: PathBuf,

    /// Maximum coil displacement for BL(x) sweep (meters).
    #[arg(long, default_value = "0.010")]
    x_max: f64,

    /// Number of displacement steps per side (total points = 2N+1).
    #[arg(long, default_value = "20")]
    n_steps: usize,

    /// Mesh resolution (elements per region edge).
    #[arg(long, default_value = "4")]
    mesh_resolution: usize,

    /// Print progress to stderr.
    #[arg(long, short)]
    verbose: bool,

    /// Total Rayon worker budget. Zero uses Rayon's automatic default,
    /// including `RAYON_NUM_THREADS` when set.
    #[arg(long, visible_alias = "cores", default_value_t = 0)]
    threads: usize,
    /// Print a bounded CPU execution contract as JSON and exit.
    #[arg(long)]
    execution_contract: bool,
}

fn main() {
    env_logger::init();
    if let Err(error) = run(Cli::parse()) {
        eprintln!("motor-extract failed: {error}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    sonium_xem::runtime::install_global_rayon_pool(cli.threads, None)?;

    if cli.execution_contract {
        let evidence = sonium_xem::runtime::observe_execution_contract(
            "motor-extract",
            "cpu",
            "cpu",
            std::collections::BTreeMap::from([
                ("displacement_cases".to_string(), 2 * cli.n_steps + 1),
                ("frequency_budget".to_string(), rayon::current_num_threads()),
            ]),
            false,
            "host-cpu",
        );
        println!("{}", serde_json::to_string_pretty(&evidence)?);
        return Ok(());
    }

    // Load geometry
    let json = std::fs::read_to_string(&cli.geometry)?;
    let geom: MotorGeometry = serde_json::from_str(&json)?;

    if cli.verbose {
        eprintln!("Motor geometry loaded:");
        eprintln!("  Coil radius:  {:.1} mm", geom.coil.radius * 1e3);
        eprintln!("  Coil height:  {:.1} mm", geom.coil.height * 1e3);
        eprintln!("  Coil turns:   {}", geom.coil.n_turns);
        eprintln!("  Air gap:      {:.2} mm", geom.air_gap * 1e3);
        eprintln!(
            "  Pole piece:   r={:.1} mm, h={:.1} mm, mu_r={}",
            geom.pole_piece.radius * 1e3,
            geom.pole_piece.height * 1e3,
            geom.pole_piece.mu_r,
        );
        eprintln!("Extraction config:");
        eprintln!("  x_max:        ±{:.1} mm", cli.x_max * 1e3);
        eprintln!(
            "  steps:        {} ({}  points)",
            cli.n_steps,
            2 * cli.n_steps + 1
        );
        eprintln!("  mesh res:     {}", cli.mesh_resolution);
        eprintln!("  threads:      {}", rayon::current_num_threads());
    }

    let config = ExtractionConfig {
        x_max: cli.x_max,
        n_displacement_steps: cli.n_steps,
        mesh_resolution: cli.mesh_resolution,
        ..Default::default()
    };

    if cli.verbose {
        eprintln!(
            "Running magnetostatic BL extraction plus passive semi-analytical Le(f)/Re(f)..."
        );
    }

    let params = extract_motor_parameters(&geom, &config)?;

    if cli.verbose {
        eprintln!("Results:");
        eprintln!("  BL(0):  {:.3} T·m", params.bl_nominal);
        eprintln!("  Le(DC): {:.3} mH", params.le_nominal * 1e3);
        eprintln!("  BL curve: {} points", params.bl_curve.len());
        eprintln!("  Le(f):    {} points", params.le_vs_frequency.len());
    }

    // Write output
    let output_json = serde_json::to_string_pretty(&params)?;
    std::fs::write(&cli.output, &output_json)?;

    if cli.verbose {
        eprintln!("Written to {}", cli.output.display());
    } else {
        println!("{}", cli.output.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threads_default_to_automatic_and_accept_explicit_override() {
        let automatic = Cli::parse_from(["motor-extract", "--geometry", "motor.json"]);
        assert_eq!(automatic.threads, 0);
        assert!(!automatic.execution_contract);
        let explicit = Cli::parse_from([
            "motor-extract",
            "--geometry",
            "motor.json",
            "--threads",
            "8",
        ]);
        assert_eq!(explicit.threads, 8);
    }
}
