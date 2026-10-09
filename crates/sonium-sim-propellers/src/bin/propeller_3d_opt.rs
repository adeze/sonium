//! 3D propeller blade shape optimization.
//!
//! Run with:
//! `cargo run -p sonium-sim-propellers --release --bin propeller-3d-opt`

use clap::Parser;
use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    COMBINED_REDUCTION_PRIOR_DB, Propeller3dOptConfig, SERRATION_AMPLITUDE_CHORD_FRACTION_RANGE,
    SERRATION_SPAN_FRACTION_RANGE, SERRATION_WAVELENGTH_CHORD_FRACTION_RANGE,
    try_optimize_blade_shape,
};
use sonium_xem::io::binary_output_dir;

#[derive(Debug, Parser)]
#[command(about = "Optimize a calibrated 5-inch low-noise propeller prior")]
struct Cli {
    /// Total Rayon worker budget. Zero uses Rayon's automatic default.
    #[arg(long, visible_alias = "cores", default_value_t = 0)]
    threads: usize,
    /// Backend for the bounded panel-integral cell: cpu, cuda, or metal.
    /// Optimization/source evaluation remains host-side.
    #[arg(long, default_value = "cpu")]
    backend: String,
    /// CUDA device ordinal for the bounded panel-integral cell.
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,
    /// Print the binary capability matrix and exit.
    #[arg(long)]
    capabilities: bool,
    /// Print a bounded CPU execution contract as JSON and exit.
    #[arg(long)]
    execution_contract: bool,
    /// Reference speed for the two-blade, 3 N source operating point.
    #[arg(long, default_value_t = 18_000.0, value_parser = parse_rpm)]
    rpm: f64,
    /// Coordinate-search passes.
    #[arg(long, default_value_t = 8, value_parser = parse_iterations)]
    iterations: usize,
    /// Maximum fractional thrust loss. The calibrated default is 0.05;
    /// values through 0.20 are allowed only for explicit trade studies.
    #[arg(long, default_value_t = 0.05, value_parser = parse_thrust_loss)]
    max_thrust_loss: f64,
    /// Initial serration amplitude as a local-chord fraction (0.006–0.035).
    #[arg(long, default_value_t = 0.018, value_parser = parse_serration_amplitude)]
    serration_amplitude: f64,
    /// Initial serration wavelength as a local-chord fraction (0.035–0.120).
    #[arg(long, default_value_t = 0.055, value_parser = parse_serration_wavelength)]
    serration_wavelength: f64,
    /// Initial serrated fraction of blade span (0.40–0.95).
    #[arg(long, default_value_t = 0.72, value_parser = parse_serration_span)]
    serration_span: f64,
    /// Write a deterministic, headless website gallery and provenance manifest.
    #[arg(long, value_name = "DIR")]
    gallery: Option<std::path::PathBuf>,
}

fn parse_bounded_f64(value: &str, name: &str, min: f64, max: f64) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|error| format!("invalid {name} '{value}': {error}"))?;
    if !parsed.is_finite() || !(min..=max).contains(&parsed) {
        return Err(format!("{name} must be finite and in [{min}, {max}]"));
    }
    Ok(parsed)
}

fn parse_rpm(value: &str) -> Result<f64, String> {
    parse_bounded_f64(value, "RPM", 3_000.0, 35_000.0)
}

fn parse_iterations(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid iteration count '{value}': {error}"))?;
    if parsed > 100 {
        return Err("iteration count must be in [0, 100]".to_string());
    }
    Ok(parsed)
}

fn parse_thrust_loss(value: &str) -> Result<f64, String> {
    parse_bounded_f64(value, "maximum thrust loss", 0.0, 0.20)
}

fn parse_serration_amplitude(value: &str) -> Result<f64, String> {
    parse_bounded_f64(
        value,
        "serration amplitude",
        SERRATION_AMPLITUDE_CHORD_FRACTION_RANGE[0],
        SERRATION_AMPLITUDE_CHORD_FRACTION_RANGE[1],
    )
}

fn parse_serration_wavelength(value: &str) -> Result<f64, String> {
    parse_bounded_f64(
        value,
        "serration wavelength",
        SERRATION_WAVELENGTH_CHORD_FRACTION_RANGE[0],
        SERRATION_WAVELENGTH_CHORD_FRACTION_RANGE[1],
    )
}

fn parse_serration_span(value: &str) -> Result<f64, String> {
    parse_bounded_f64(
        value,
        "serration span",
        SERRATION_SPAN_FRACTION_RANGE[0],
        SERRATION_SPAN_FRACTION_RANGE[1],
    )
}

fn build_config(cli: &Cli) -> Propeller3dOptConfig {
    let mut config = Propeller3dOptConfig {
        source: PropellerSource::drone_5inch(cli.rpm),
        max_iterations: cli.iterations,
        max_thrust_loss: cli.max_thrust_loss,
        ..Propeller3dOptConfig::default()
    };
    config.initial_variables.serration_amplitude = cli.serration_amplitude;
    config.initial_variables.serration_wavelength = cli.serration_wavelength;
    config.initial_variables.serration_span = cli.serration_span;
    config
}

fn main() {
    let cli = Cli::parse();
    if let Err(error) = run(cli) {
        eprintln!("propeller optimization failed: {error}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    if cli.capabilities {
        println!("cell,cpu,cuda,metal");
        println!(
            "optimization,yes,no,no\npanel-integral,yes,{},no",
            if cfg!(feature = "cuda") {
                "yes"
            } else {
                "not-built"
            }
        );
        return Ok(());
    }
    sonium_xem::runtime::install_global_rayon_pool(cli.threads, None)?;
    if cli.execution_contract {
        if cli.backend != "cpu" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "--execution-contract is a host contract; run the CUDA panel cell to record its device identity",
            )
            .into());
        }
        let evidence = sonium_xem::runtime::observe_execution_contract(
            "propeller-3d-opt",
            "cpu",
            "cpu",
            std::collections::BTreeMap::from([(
                "harmonic_and_radial_budget".to_string(),
                rayon::current_num_threads(),
            )]),
            false,
            "host-cpu",
        );
        println!("{}", serde_json::to_string_pretty(&evidence)?);
        return Ok(());
    }
    match cli.backend.as_str() {
        "cpu" | "cuda" => {}
        "metal" => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "Metal is not implemented for propeller-3d-opt",
            )
            .into());
        }
        backend => {
            return Err(format!("unknown backend '{backend}'; expected cpu|cuda|metal").into());
        }
    }
    let config = build_config(&cli);
    let result = try_optimize_blade_shape(&config)?;

    let out_dir = cli
        .gallery
        .clone()
        .unwrap_or_else(|| binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME")));
    if cli.gallery.is_some() {
        if let Err(error) = write_gallery_bundle(&out_dir, &config, &result) {
            return Err(format!("failed to write gallery: {error}").into());
        }
        println!(
            "Gallery manifest written to {}",
            out_dir.join("gallery.json").display()
        );
    } else {
        std::fs::create_dir_all(&out_dir)?;
        let vtu_path = out_dir.join("optimized_blade.vtu");
        result.mesh.write_vtu(&vtu_path)?;
        println!("Blade mesh written to {}", vtu_path.display());
    }

    if cli.backend == "cuda" {
        run_cuda_panel_cell(&out_dir, &result, cli.cuda_device)?;
    }

    println!("=== 3D Propeller Blade Shape Optimization ===\n");
    println!(
        "Reference operating point: {:.0} RPM, {:.2} N thrust",
        config.source.rpm, config.source.thrust
    );
    println!(
        "Constraints: radius scale <= 1.0, thrust loss <= {:.0}%, constant mass via thickness scale",
        config.max_thrust_loss * 100.0
    );
    println!(
        "Calibration domain: modeled 1-5 kHz reduction <= {:.1} dB",
        config.max_calibrated_reduction_db
    );
    println!(
        "Optimized after {} coordinate-search passes\n",
        result.iterations
    );

    println!("Shape:");
    println!("  radius scale      = {:.3}", result.variables.radius_scale);
    println!(
        "  pitch root/tip    = {:+.2} / {:+.2} deg delta",
        result.variables.pitch_root_delta_deg, result.variables.pitch_tip_delta_deg
    );
    println!(
        "  chord root/tip    = {:.3} / {:.3} scale",
        result.variables.chord_root_scale, result.variables.chord_tip_scale
    );
    println!(
        "  loop close/lift   = {:.3} / {:.3} lift scale",
        result.variables.loop_closure_fraction, result.variables.loop_axial_lift_scale
    );
    println!(
        "  serrations amp/wl/span = {:.3} / {:.3} / {:.3}",
        result.variables.serration_amplitude,
        result.variables.serration_wavelength,
        result.variables.serration_span
    );

    println!("\nConstraints:");
    println!("  thrust loss       = {:.1}%", result.thrust_loss * 100.0);
    println!(
        "  thickness scale   = {:.3} (keeps material weight constant)",
        result.thickness_scale_for_constant_weight
    );
    println!("  panel area        = {:.5} m2", result.mesh.total_area());
    println!("\nLoads and power closure:");
    println!(
        "  thrust / torque   = {:.3} N / {:.5} N m",
        result.loads.thrust_n, result.loads.torque_nm
    );
    println!(
        "  shaft / ideal     = {:.3} / {:.3} W (ideal hover η {:.3})",
        result.loads.shaft_power_w,
        result.loads.ideal_hover_power_w,
        result.loads.ideal_hover_efficiency
    );
    println!(
        "  acoustic power    = {:.3e} W ({:.3e} of shaft; closure {:.3e})",
        result.loads.acoustic_power_w,
        result.loads.acoustic_to_shaft_power_ratio,
        result.loads.acoustic_power_closure_error
    );

    println!("\nNoise:");
    println!(
        "  1-5 kHz reduction = {:+.2} dB",
        result.panel_summary.sensitive_band_reduction_db
    );
    println!(
        "  empirical combined prior = {:.1} dB ({:.1}–{:.1} dB)",
        COMBINED_REDUCTION_PRIOR_DB.center_db,
        COMBINED_REDUCTION_PRIOR_DB.min_db,
        COMBINED_REDUCTION_PRIOR_DB.max_db
    );
    println!(
        "  peak spread delta = {:+.2} dB",
        result.peak_spread_delta_db
    );
    println!(
        "  vortex index      = {:.3}",
        result.panel_summary.vortex_source_index
    );
    println!(
        "  edge coherence    = {:.3}",
        result.panel_summary.edge_coherence_index
    );
    println!(
        "  rigidity index    = {:.3}",
        result.panel_summary.structural_rigidity_index
    );
    println!("  objective         = {:.3}", result.objective);
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_cuda_panel_cell(
    output_dir: &std::path::Path,
    result: &sonium_sim_propellers::OptimizedBlade,
    cuda_device: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let panel = sonium_sim_propellers::evaluate_cuda_panel_bem(
        &result.design,
        &result.mesh_config,
        1,
        cuda_device,
    )?;
    let evidence = serde_json::json!({
        "schema": panel.schema,
        "requested_backend": panel.requested_backend,
        "actual_backend": panel.actual_backend,
        "device_identity": format!("cuda:ordinal={cuda_device}"),
        "precision": panel.precision,
        "panel_count": panel.panel_count,
        "pair_count": panel.pair_count,
        "harmonic": panel.harmonic,
        "wave_number": panel.wave_number,
        "max_relative_error": panel.max_relative_error,
        "fallback_used": false,
        "scope": "bounded panel-integral CUDA cell; optimization and source model remain CPU"
    });
    std::fs::write(
        output_dir.join("cuda-panel-evidence.json"),
        serde_json::to_vec_pretty(&evidence)?,
    )?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn run_cuda_panel_cell(
    _output_dir: &std::path::Path,
    _result: &sonium_sim_propellers::OptimizedBlade,
    _cuda_device: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "CUDA requested but propeller-3d-opt was built without the `cuda` feature",
    )
    .into())
}

fn write_gallery_bundle(
    output_dir: &std::path::Path,
    config: &sonium_sim_propellers::Propeller3dOptConfig,
    result: &sonium_sim_propellers::OptimizedBlade,
) -> Result<(), Box<dyn std::error::Error>> {
    sonium_sim_propellers::gallery::write_gallery_bundle(output_dir, config, result)
}

#[allow(dead_code)]
fn write_gallery(
    output_dir: &std::path::Path,
    config: &sonium_sim_propellers::Propeller3dOptConfig,
    result: &sonium_sim_propellers::OptimizedBlade,
) -> Result<(), Box<dyn std::error::Error>> {
    use sonium_xem::io::GalleryRenderer;
    if !result.is_exportable() {
        return Err("propeller gallery refuses a non-admissible production result".into());
    }
    let panel_area = result
        .mesh
        .panels
        .iter()
        .map(|panel| panel.area)
        .collect::<Vec<_>>();
    let frequency = result
        .spectrum
        .harmonics
        .iter()
        .map(|harmonic| harmonic.frequency_hz)
        .collect::<Vec<_>>();
    let pressure = result
        .spectrum
        .harmonics
        .iter()
        .map(|harmonic| harmonic.pressure_pa)
        .collect::<Vec<_>>();
    let baseline = sonium_sim_propellers::baseline_blade_shape(config);
    let baseline_pressure = baseline
        .spectrum
        .harmonics
        .iter()
        .map(|harmonic| harmonic.pressure_pa)
        .collect::<Vec<_>>();
    let mut blade_load = result
        .mesh
        .panels
        .iter()
        .map(|panel| {
            (
                panel.radius_fraction,
                panel.area * panel.radius_fraction.powi(2),
            )
        })
        .collect::<Vec<_>>();
    blade_load.sort_by(|left, right| left.0.total_cmp(&right.0));
    let blade_load_radius = blade_load
        .iter()
        .map(|(radius, _)| *radius)
        .collect::<Vec<_>>();
    let blade_load_value = blade_load.iter().map(|(_, load)| *load).collect::<Vec<_>>();
    if panel_area.is_empty()
        || frequency.len() < 2
        || frequency.len() != pressure.len()
        || frequency.len() != baseline_pressure.len()
        || panel_area
            .iter()
            .chain(frequency.iter())
            .chain(pressure.iter())
            .chain(baseline_pressure.iter())
            .any(|value| !value.is_finite())
    {
        return Err("propeller production result contains invalid gallery arrays".into());
    }
    let mut gallery = GalleryRenderer::new(output_dir)?;
    let spectrum_csv = output_dir.join("source_spectrum.csv");
    let mut spectrum_csv_contents = String::from("frequency_hz,pressure_pa,baseline_pressure_pa\n");
    for ((frequency_hz, pressure_pa), baseline_pressure_pa) in
        frequency.iter().zip(&pressure).zip(&baseline_pressure)
    {
        spectrum_csv_contents.push_str(&format!(
            "{frequency_hz:.17e},{pressure_pa:.17e},{baseline_pressure_pa:.17e}\n"
        ));
    }
    std::fs::write(&spectrum_csv, spectrum_csv_contents)?;
    let blade_load_csv = output_dir.join("blade_load_distribution.csv");
    let mut blade_load_csv_contents = String::from("radius_fraction,loading_proxy_m2\n");
    for (radius, load) in &blade_load {
        blade_load_csv_contents.push_str(&format!("{radius:.17e},{load:.17e}\n"));
    }
    std::fs::write(&blade_load_csv, blade_load_csv_contents)?;
    gallery.existing_figure(
        "source_spectrum.csv",
        "csv",
        "optimized_blade.vtu",
        "Finite production harmonic source spectrum; columns are Hz, Pa, and baseline Pa",
    )?;
    gallery.existing_figure(
        "blade_load_distribution.csv",
        "csv",
        "optimized_blade.vtu",
        "Finite production spanwise loading proxy; columns are radius fraction and m²",
    )?;
    gallery.heatmap_png(
        "optimized-blade-loading.png",
        &panel_area,
        "optimized_blade.vtu",
        "Fixed camera blade-panel loading proxy; panel areas in m²",
    )?;
    gallery.heatmap_png(
        "axial-wake-proxy.png",
        &result
            .mesh
            .panels
            .iter()
            .map(|panel| panel.radius_fraction * panel.chord_fraction)
            .collect::<Vec<_>>(),
        "optimized_blade.vtu",
        "Pinned axial wake/loading proxy sampled from production panel geometry",
    )?;
    gallery.line_svg(
        "spectrum.svg",
        &frequency,
        &pressure,
        "Propeller harmonic pressure spectrum",
        "optimized_blade.vtu",
        "Harmonic peak pressure (Pa) versus frequency (Hz)",
    )?;
    gallery.line_svg(
        "thrust-versus-operating-point.svg",
        &[baseline.design.source.rpm, result.design.source.rpm],
        &[baseline.design.source.thrust, result.design.source.thrust],
        "Baseline and optimized thrust",
        "baseline_blade.vtu",
        "Thrust (N) at the pinned operating point (RPM)",
    )?;
    gallery.line_svg(
        "torque-versus-operating-point.svg",
        &[baseline.design.source.rpm, result.design.source.rpm],
        &[baseline.design.source.torque, result.design.source.torque],
        "Baseline and optimized torque",
        "baseline_blade.vtu",
        "Torque (N·m) at the pinned operating point (RPM)",
    )?;
    gallery.line_svg(
        "blade-load-distribution.svg",
        &blade_load_radius,
        &blade_load_value,
        "Spanwise blade loading proxy",
        "optimized_blade.vtu",
        "Panel area weighted by radial loading factor (m²), ordered by pinned panel generation",
    )?;
    gallery.line_svg(
        "conventional-toroidal-comparison.svg",
        &frequency,
        &baseline_pressure
            .iter()
            .zip(&pressure)
            .map(|(reference, candidate)| 20.0 * (reference.max(1.0e-30) / candidate.max(1.0e-30)).log10())
            .collect::<Vec<_>>(),
        "Conventional-to-toroidal harmonic reduction",
        "baseline_blade.vtu",
        "Peak-pressure reduction (dB) by harmonic; baseline and candidate are generated by the same pinned solver configuration",
    )?;
    let rotation_frames = (0..12)
        .map(|phase| {
            panel_area
                .iter()
                .enumerate()
                .map(|(index, value)| value * (1.0 + 0.2 * ((index + phase) as f64 * 0.31).sin()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    gallery.animation_gif(
        "rotation.gif",
        &rotation_frames,
        "optimized_blade.vtu",
        "Twelve-frame deterministic rotation/loading animation derived from production blade panels",
    )?;
    let manifest_path = gallery.finish(
        "Optimized propeller gallery",
        vec![
            "Pinned 18,000 RPM two-blade 3 N operating point; CPU surrogate model".into(),
            "1920x1080 fixed raster canvas, deterministic colour mapping and SVG axes".into(),
            format!(
                "thrust loss {:.6}; calibrated 1–5 kHz reduction {:.6} dB",
                result.thrust_loss, result.panel_summary.sensitive_band_reduction_db
            ),
        ],
    )?;
    sonium_xem::io::validate_gallery_manifest(&manifest_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_sim_propellers::{Blade3dConfig, optimize_blade_shape};

    #[test]
    fn cli_defaults_use_calibrated_operating_prior() {
        let cli = Cli::parse_from(["propeller-3d-opt"]);
        let config = build_config(&cli);
        assert_eq!(config.source.rpm, 18_000.0);
        assert_eq!(config.max_iterations, 8);
        assert_eq!(config.max_thrust_loss, 0.05);
        assert_eq!(config.max_calibrated_reduction_db, 10.0);
        assert_eq!(config.initial_variables.serration_amplitude, 0.018);
        assert_eq!(config.initial_variables.serration_wavelength, 0.055);
        assert_eq!(config.initial_variables.serration_span, 0.72);
        assert_eq!(cli.gallery, None);
    }

    #[test]
    fn cli_rejects_values_outside_calibration_ranges() {
        assert!(Cli::try_parse_from(["propeller-3d-opt", "--max-thrust-loss", "0.25"]).is_err());
        assert!(
            Cli::try_parse_from(["propeller-3d-opt", "--serration-amplitude", "0.05"]).is_err()
        );
        assert!(Cli::try_parse_from(["propeller-3d-opt", "--rpm", "1000"]).is_err());
    }

    #[test]
    fn gallery_is_headless_manifested_and_has_a_fixed_rotation_frame_count() {
        let dir = tempfile::tempdir().unwrap();
        let config = Propeller3dOptConfig {
            mesh: Blade3dConfig {
                radial_segments: 8,
                chord_segments: 4,
                loop_segments: 4,
                ..Blade3dConfig::default()
            },
            n_harmonics: 4,
            max_iterations: 0,
            ..Propeller3dOptConfig::default()
        };
        let result = optimize_blade_shape(&config);
        write_gallery_bundle(dir.path(), &config, &result).unwrap();
        assert_eq!(
            sonium_xem::io::validate_gif(dir.path().join("rotation.gif")).unwrap(),
            (320, 180, 12)
        );
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("gallery.json")).unwrap())
                .unwrap();
        let figures = manifest["figures"].as_array().unwrap();
        assert_eq!(figures.len(), 10);
        let checked_manifest =
            sonium_xem::io::validate_gallery_manifest(dir.path().join("gallery.json")).unwrap();
        assert_eq!(checked_manifest.width_px, 1920);
        assert_eq!(checked_manifest.height_px, 1080);
        let mut rasters = 0;
        let mut vectors = 0;
        let mut animations = 0;
        let mut csvs = 0;
        for figure in figures {
            let filename = figure["filename"].as_str().unwrap();
            assert!(
                figure["source_checksum_fnv1a64"].is_string(),
                "figure {filename} must retain source provenance"
            );
            let artifact = dir.path().join(filename);
            assert!(artifact.is_file(), "missing gallery figure {filename}");
            match figure["kind"].as_str().unwrap() {
                "png" => {
                    assert_eq!(
                        sonium_xem::io::validate_png_nonblank(&artifact).unwrap(),
                        (1920, 1080),
                        "raster figure {filename} must be nonblank and fixed-size"
                    );
                    rasters += 1;
                }
                "svg" => {
                    let svg = std::fs::read_to_string(&artifact).unwrap();
                    assert!(
                        svg.contains("<svg"),
                        "SVG figure {filename} is not structured"
                    );
                    assert!(
                        svg.contains("width=\"1920\""),
                        "SVG figure {filename} is not fixed-size"
                    );
                    vectors += 1;
                }
                "gif" => {
                    assert_eq!(
                        sonium_xem::io::validate_gif(&artifact).unwrap(),
                        (320, 180, 12)
                    );
                    animations += 1;
                }
                "csv" => {
                    let csv = std::fs::read_to_string(&artifact).unwrap();
                    assert!(
                        !csv.ends_with("\n\n"),
                        "CSV {filename} has a blank trailing row"
                    );
                    let rows = csv.lines().collect::<Vec<_>>();
                    match filename {
                        "source_spectrum.csv" => {
                            assert_eq!(rows[0], "frequency_hz,pressure_pa,baseline_pressure_pa");
                            assert_eq!(rows.len(), result.spectrum.harmonics.len() + 1);
                            for row in rows.iter().skip(1) {
                                assert_eq!(row.split(',').count(), 3);
                                assert!(
                                    row.split(',')
                                        .all(|value| value.parse::<f64>().unwrap().is_finite())
                                );
                            }
                        }
                        "blade_load_distribution.csv" => {
                            assert_eq!(rows[0], "radius_fraction,loading_proxy_m2");
                            assert_eq!(rows.len(), result.mesh.panels.len() + 1);
                            for row in rows.iter().skip(1) {
                                assert_eq!(row.split(',').count(), 2);
                                assert!(
                                    row.split(',')
                                        .all(|value| value.parse::<f64>().unwrap().is_finite())
                                );
                            }
                        }
                        other => panic!("unexpected CSV figure {other}"),
                    }
                    csvs += 1;
                }
                other => panic!("unexpected gallery figure kind {other}"),
            }
        }
        assert_eq!(rasters, 2);
        assert_eq!(vectors, 5);
        assert_eq!(animations, 1);
        assert_eq!(csvs, 2);
    }

    #[test]
    fn gallery_refuses_corrupted_load_power_ledger() {
        let dir = tempfile::tempdir().unwrap();
        let config = Propeller3dOptConfig {
            mesh: Blade3dConfig {
                radial_segments: 8,
                chord_segments: 4,
                loop_segments: 4,
                ..Blade3dConfig::default()
            },
            n_harmonics: 4,
            max_iterations: 0,
            ..Propeller3dOptConfig::default()
        };
        let mut result = optimize_blade_shape(&config);
        result.loads.shaft_power_w = f64::NAN;
        let error = write_gallery_bundle(dir.path(), &config, &result)
            .expect_err("corrupt load ledger must not reach gallery rendering");
        assert!(
            error
                .to_string()
                .contains("non-admissible production result")
        );
        assert!(!dir.path().join("optimized_blade.vtu").exists());
        assert!(!dir.path().join("baseline_blade.vtu").exists());
        assert!(!dir.path().join("source_spectrum.csv").exists());
        assert!(!dir.path().join("gallery.json").exists());
    }
}
