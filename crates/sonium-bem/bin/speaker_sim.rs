//! `speaker-sim`: loudspeaker BEM simulation and optimisation CLI.

use std::path::PathBuf;

use clap::Parser;

use sonium_bem::speaker::{cea2034_output, mesh_gen, model::LoudspeakerModel, optimization, simulation};
use sonium_xem::io::binary_output_dir;

fn default_output_dir() -> PathBuf {
    binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"))
}

#[derive(Parser)]
#[command(name = "speaker-sim", about = "Loudspeaker BEM simulation and optimisation")]
struct Cli {
    /// Path to loudspeaker model JSON file.
    #[arg(short, long)]
    model: PathBuf,

    /// Output directory for results and CEA-2034 files.
    /// Defaults to `<sonium-out-root>/sonium-bem/speaker-sim/`.
    #[arg(short, long, default_value_os_t = default_output_dir())]
    output: PathBuf,

    /// Run parameter optimisation before simulation.
    #[arg(long)]
    optimize: bool,

    /// Maximum optimisation iterations.
    #[arg(long, default_value = "20")]
    max_iter: usize,

    /// Export mesh to VTK and exit (no simulation).
    #[arg(long)]
    export_mesh: bool,

    /// Print progress to stderr.
    #[arg(long, short)]
    verbose: bool,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    // Load model
    let json = std::fs::read_to_string(&cli.model)
        .unwrap_or_else(|e| panic!("Cannot read {}: {e}", cli.model.display()));
    let model: LoudspeakerModel = serde_json::from_str(&json)
        .unwrap_or_else(|e| panic!("Invalid JSON in {}: {e}", cli.model.display()));

    if let Err(e) = model.validate() {
        eprintln!("Model validation failed: {e}");
        std::process::exit(1);
    }

    if cli.verbose {
        eprintln!(
            "Cabinet: {:.0}x{:.0}x{:.0} mm",
            model.cabinet.height * 1000.0,
            model.cabinet.width * 1000.0,
            model.cabinet.depth * 1000.0,
        );
        eprintln!(
            "Woofer: r={:.1} mm, y={:.1} mm",
            model.woofer.radius * 1000.0,
            model.woofer.y_offset * 1000.0,
        );
        eprintln!(
            "Tweeter: r={:.1} mm, y={:.1} mm",
            model.tweeter.radius * 1000.0,
            model.tweeter.y_offset * 1000.0,
        );
    }

    // --- Mesh-only mode ---
    if cli.export_mesh {
        std::fs::create_dir_all(&cli.output).unwrap();
        let mesh = mesh_gen::generate_loudspeaker_mesh(&model);
        let vtk_path = cli.output.join("speaker_mesh.vtu");
        mesh_gen::export_vtu(&mesh, vtk_path.to_str().unwrap());
        if cli.verbose {
            eprintln!(
                "Mesh: {} nodes, {} triangles",
                mesh.nodes.len(),
                mesh.triangles.len()
            );
        }
        eprintln!("Mesh exported to {}", vtk_path.display());
        return;
    }

    // --- Optimisation ---
    let model = if cli.optimize {
        let config = optimization::OptimizationConfig {
            max_iterations: cli.max_iter,
            ..Default::default()
        };
        if cli.verbose {
            eprintln!("Starting optimisation (max {} iterations)...", cli.max_iter);
        }
        let result = optimization::optimize(&model, &config);
        if cli.verbose {
            eprintln!(
                "Optimisation: {} iterations, objective={:.4}, converged={}",
                result.iterations, result.objective, result.converged
            );
        }
        // Write optimised model JSON
        std::fs::create_dir_all(&cli.output).unwrap();
        let opt_json = serde_json::to_string_pretty(&result.model).unwrap();
        let opt_path = cli.output.join("optimised_model.json");
        std::fs::write(&opt_path, &opt_json).unwrap();
        if cli.verbose {
            eprintln!("Optimised model written to {}", opt_path.display());
        }
        result.model
    } else {
        model
    };

    // --- Simulation ---
    let eval_points = cea2034_output::generate_eval_points(model.baffle_center());

    if cli.verbose {
        eprintln!(
            "Running BEM simulation: {:.0}-{:.0} Hz, {} points...",
            model.simulation.freq_start,
            model.simulation.freq_end,
            model.simulation.freq_points
        );
    }

    let result = simulation::run_simulation(&model, &eval_points);

    // --- CEA-2034 output ---
    std::fs::create_dir_all(&cli.output).unwrap();
    cea2034_output::write_cea2034_files(&cli.output, &result).unwrap();

    // --- Spinorama scoring ---
    let metrics = cea2034_output::compute_spinorama_metrics(&result);
    eprintln!("\n--- Spinorama Metrics ---");
    eprintln!("Preference score: {:.2}", metrics.preference_score);

    // Write mesh VTK alongside
    let mesh = mesh_gen::generate_loudspeaker_mesh(&model);
    let vtk_path = cli.output.join("speaker_mesh.vtu");
    mesh_gen::export_vtu(&mesh, vtk_path.to_str().unwrap());

    if cli.verbose {
        eprintln!(
            "Mesh: {} nodes, {} triangles",
            mesh.nodes.len(),
            mesh.triangles.len()
        );
    }
    eprintln!("Results written to {}", cli.output.display());
}
