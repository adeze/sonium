fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let options = DemoOptions::parse(&args)?;
    if options.capabilities {
        println!("backend,cpu,cuda,metal");
        println!(
            "structural-bracket-demo,yes,{},no",
            if cfg!(feature = "cuda") {
                "yes"
            } else {
                "not-built"
            }
        );
        return Ok(());
    }
    sonium_xem::runtime::install_global_rayon_pool(options.threads, None)?;

    if options.execution_contract {
        if options.backend != "cpu" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "--execution-contract is a host contract; run the CUDA static cell to record its device identity",
            )
            .into());
        }
        let evidence = sonium_xem::runtime::observe_execution_contract(
            "structural-bracket-demo",
            "cpu",
            "cpu",
            std::collections::BTreeMap::from([(
                "harmonic_frequency_budget".to_string(),
                rayon::current_num_threads(),
            )]),
            false,
            "host-cpu",
        );
        println!("{}", serde_json::to_string_pretty(&evidence)?);
        return Ok(());
    }

    let gallery_dir = options.gallery;
    let output_dir = gallery_dir.clone().unwrap_or(options.output_dir);
    std::fs::create_dir_all(&output_dir)?;
    let mut study = sonium_sim_structural_mechanics::BracketStudy::introductory_bracket();
    match options.scaling_size.as_deref() {
        None | Some("small") => {}
        Some("medium") => {
            study.geometry.nx *= 2;
            study.geometry.ny *= 2;
        }
        Some("large") => {
            study.geometry.nx *= 2;
            study.geometry.ny *= 2;
            let count = 120usize;
            study.frequency_hz = (0..count)
                .map(|index| 20.0 * (2_000.0_f64 / 20.0).powf(index as f64 / (count - 1) as f64))
                .collect();
        }
        Some(size) => {
            return Err(
                format!("unknown --scaling-size '{size}'; expected small|medium|large").into(),
            );
        }
    }

    match options.backend.as_str() {
        "cpu" => {}
        "cuda" => run_cuda_static_cell(&output_dir, &study)?,
        "metal" => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "Metal is not implemented for structural-bracket-demo",
            )
            .into());
        }
        backend => {
            return Err(format!("unknown backend '{backend}'; expected cpu|cuda|metal").into());
        }
    }
    let report = sonium_sim_structural_mechanics::run_bracket_example(study)?;
    let artifacts =
        sonium_sim_structural_mechanics::write_bracket_visualization(report, &output_dir)?;

    if let Some(gallery_dir) = gallery_dir {
        write_gallery(&gallery_dir, &artifacts)?;
        println!("  gallery: {}", gallery_dir.join("gallery.json").display());
    }

    println!("Wrote structural bracket visualization:");
    println!("  report: {}", artifacts.report_json.display());
    println!("  result: {}", artifacts.result_vtu.display());
    println!(
        "  deformed preview: {}",
        artifacts.deformed_preview_vtu.display()
    );
    println!("  preview warp scale: {:.1}x", artifacts.warp_scale);
    Ok(())
}

struct DemoOptions {
    output_dir: std::path::PathBuf,
    gallery: Option<std::path::PathBuf>,
    threads: usize,
    backend: String,
    capabilities: bool,
    execution_contract: bool,
    scaling_size: Option<String>,
}

impl DemoOptions {
    fn parse(args: &[String]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut output_dir = None;
        let mut gallery = None;
        let mut threads = 0usize;
        let mut backend = "cpu".to_string();
        let mut capabilities = false;
        let mut execution_contract = false;
        let mut scaling_size = None;
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--gallery" => {
                    index += 1;
                    gallery = Some(
                        args.get(index)
                            .ok_or("--gallery requires a directory")?
                            .into(),
                    );
                }
                "--threads" | "--cores" => {
                    index += 1;
                    threads = args
                        .get(index)
                        .ok_or("--threads requires a non-negative integer")?
                        .parse()?;
                }
                "--backend" => {
                    index += 1;
                    backend = args.get(index).ok_or("--backend requires a value")?.clone();
                }
                "--capabilities" => capabilities = true,
                "--execution-contract" => execution_contract = true,
                "--scaling-size" => {
                    index += 1;
                    scaling_size = Some(
                        args.get(index)
                            .ok_or("--scaling-size requires small, medium, or large")?
                            .clone(),
                    );
                }
                value if value.starts_with('-') => {
                    return Err(format!("unknown option '{value}'").into());
                }
                value if output_dir.is_none() => output_dir = Some(value.into()),
                value => return Err(format!("unexpected positional argument '{value}'").into()),
            }
            index += 1;
        }
        Ok(Self {
            output_dir: output_dir.unwrap_or_else(|| "target/structural-bracket-demo".into()),
            gallery,
            threads,
            backend,
            capabilities,
            execution_contract,
            scaling_size,
        })
    }
}

#[cfg(feature = "cuda")]
fn run_cuda_static_cell(
    output_dir: &std::path::Path,
    study: &sonium_sim_structural_mechanics::BracketStudy,
) -> Result<(), Box<dyn std::error::Error>> {
    let result = sonium_sim_structural_mechanics::run_static_bracket_cuda(
        study,
        &sonium_sim_structural_mechanics::CudaStaticBracketConfig::default(),
    )?;
    std::fs::write(
        output_dir.join("cuda_static_bracket.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn run_cuda_static_cell(
    _output_dir: &std::path::Path,
    _study: &sonium_sim_structural_mechanics::BracketStudy,
) -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "CUDA requested but structural-bracket-demo was built without the `cuda` feature",
    )
    .into())
}

fn write_gallery(
    output_dir: &std::path::Path,
    artifacts: &sonium_sim_structural_mechanics::VisualizationArtifacts,
) -> Result<(), Box<dyn std::error::Error>> {
    use sonium_xem::io::GalleryRenderer;
    let report = &artifacts.report;
    let displacement = report
        .static_result
        .displacement
        .iter()
        .map(|value| value[0].hypot(value[1]))
        .collect::<Vec<_>>();
    if displacement.iter().any(|value| !value.is_finite())
        || report
            .frequency_response
            .iter()
            .any(|point| !point.frequency_hz.is_finite() || !point.displacement_abs_m.is_finite())
    {
        return Err("structural gallery input contains non-finite production values".into());
    }
    let mut gallery = GalleryRenderer::new(output_dir)?;
    gallery.heatmap_png("bracket-displacement.png", &displacement, "bracket_result.vtu", "Fixed camera; displacement magnitude (m), deformed display scale recorded in bracket_report.json")?;
    let frequency = report
        .frequency_response
        .iter()
        .map(|point| point.frequency_hz)
        .collect::<Vec<_>>();
    let response = report
        .frequency_response
        .iter()
        .map(|point| point.displacement_abs_m)
        .collect::<Vec<_>>();
    let phase = report
        .frequency_response
        .iter()
        .map(|point| point.phase_lag_rad)
        .collect::<Vec<_>>();
    let magnification = report
        .frequency_response
        .iter()
        .map(|point| point.dynamic_magnification)
        .collect::<Vec<_>>();
    gallery.line_svg(
        "harmonic-response.svg",
        &frequency,
        &response,
        "Bracket harmonic tip response",
        "bracket_report.json",
        "Tip displacement magnitude (m) versus frequency (Hz)",
    )?;
    gallery.line_svg(
        "harmonic-phase.svg",
        &frequency,
        &phase,
        "Bracket harmonic phase lag",
        "bracket_report.json",
        "Tip response phase lag (rad) versus frequency (Hz)",
    )?;
    gallery.line_svg(
        "dynamic-magnification.svg",
        &frequency,
        &magnification,
        "Bracket dynamic magnification",
        "bracket_report.json",
        "Dynamic magnification versus frequency (Hz)",
    )?;
    gallery.line_svg("modal-frequency.svg", &[0.0, 1.0], &[report.modal.first_bending_frequency_hz; 2], "First bending-mode frequency", "bracket_report.json", "First bending frequency (Hz); modal normalization and orthogonality are recorded in the report")?;
    let convergence_study = sonium_sim_structural_mechanics::BracketStudy::introductory_bracket();
    let convergence =
        sonium_sim_structural_mechanics::run_static_mesh_convergence(&convergence_study, 0.2)?;
    if !convergence.converged {
        return Err(format!(
            "structural gallery mesh-convergence gate did not converge (displacement={:.3e}, energy={:.3e}, tolerance={:.3e})",
            convergence.displacement_relative_change,
            convergence.energy_relative_change,
            convergence.tolerance
        )
        .into());
    }
    let convergence_scalars = [
        convergence.coarse.average_tip_deflection_m,
        convergence.refined.average_tip_deflection_m,
        convergence.coarse.strain_energy_j,
        convergence.refined.strain_energy_j,
        convergence.coarse.max_von_mises_pa,
        convergence.refined.max_von_mises_pa,
        convergence.displacement_relative_change,
        convergence.energy_relative_change,
        convergence.stress_relative_change,
    ];
    if convergence.coarse.nodes == 0
        || convergence.refined.nodes <= convergence.coarse.nodes
        || convergence.coarse.dofs == 0
        || convergence.refined.dofs <= convergence.coarse.dofs
        || convergence_scalars.iter().any(|value| !value.is_finite())
    {
        return Err(
            "structural gallery convergence contains invalid topology or scalar values".into(),
        );
    }
    let convergence_json = serde_json::json!({
        "schema": "sonium-structural.mesh-convergence.v1",
        "study": convergence_study.name,
        "tolerance": convergence.tolerance,
        "converged": convergence.converged,
        "levels": [
            {
                "label": "coarse",
                "nodes": convergence.coarse.nodes,
                "elements": convergence.coarse.elements,
                "dofs": convergence.coarse.dofs,
                "average_tip_deflection_m": convergence.coarse.average_tip_deflection_m,
                "strain_energy_j": convergence.coarse.strain_energy_j,
                "max_von_mises_pa": convergence.coarse.max_von_mises_pa,
            },
            {
                "label": "refined",
                "nodes": convergence.refined.nodes,
                "elements": convergence.refined.elements,
                "dofs": convergence.refined.dofs,
                "average_tip_deflection_m": convergence.refined.average_tip_deflection_m,
                "strain_energy_j": convergence.refined.strain_energy_j,
                "max_von_mises_pa": convergence.refined.max_von_mises_pa,
            },
        ],
        "relative_change": {
            "displacement": convergence.displacement_relative_change,
            "energy": convergence.energy_relative_change,
            "stress": convergence.stress_relative_change,
            "stress_converged": convergence.stress_converged,
        },
        "provenance": {
            "solver": "run_static_mesh_convergence",
            "source_report": "bracket_report.json",
            "mesh_levels": "36x8 and 72x16 production triangular meshes",
        },
    });
    std::fs::write(
        output_dir.join("mesh-convergence.json"),
        serde_json::to_vec_pretty(&convergence_json)?,
    )?;
    gallery.line_svg(
        "mesh-convergence.svg",
        &[convergence.coarse.dofs as f64, convergence.refined.dofs as f64],
        &[
            convergence.coarse.average_tip_deflection_m,
            convergence.refined.average_tip_deflection_m,
        ],
        "Bracket mesh convergence",
        "mesh-convergence.json",
        "Production average tip deflection (m) versus DOFs; coarse/refined levels and relative changes are recorded in mesh-convergence.json",
    )?;
    let mut topology_study = sonium_sim_structural_mechanics::FemTopology2DStudy::introductory();
    topology_study.geometry.nx = 6;
    topology_study.geometry.ny = 2;
    topology_study.volume_fraction = 0.95;
    topology_study.move_limit = 0.01;
    topology_study.continuation_steps = 1;
    topology_study.iterations_per_stage = 3;
    topology_study.filter_radius_m = 0.02;
    topology_study.maximum_overhang_slope = 100.0;
    let topology = sonium_sim_structural_mechanics::run_fem_topology_2d(&topology_study)?;
    if !topology.connected
        || !topology.reanalysis_feasible
        || topology.final_physical_density.is_empty()
        || topology
            .final_physical_density
            .iter()
            .chain(&topology.projected_density)
            .any(|value| !value.is_finite())
        || topology
            .objective_history
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err("structural gallery topology result failed finite/connectivity gates".into());
    }
    let topology_density = topology.final_physical_density.clone();
    let topology_json = serde_json::json!({
        "schema": "sonium-structural.topology-design.v1",
        "study": topology.study.name,
        "nodes": topology.nodes,
        "elements": topology.elements,
        "dofs": topology.dofs,
        "iterations": topology.iterations,
        "converged": topology.converged,
        "objective": format!("{:?}", topology.objective),
        "objective_history": topology.objective_history,
        "initial_compliance_j": topology.initial_compliance_j,
        "final_compliance_j": topology.final_compliance_j,
        "body_fitted_reanalysis_compliance_j": topology.body_fitted_reanalysis_compliance_j,
        "body_fitted_reanalysis_relative_error": topology.body_fitted_reanalysis_relative_error,
        "final_volume_fraction": topology.final_volume_fraction,
        "connected": topology.connected,
        "reanalysis_feasible": topology.reanalysis_feasible,
        "density_range": {
            "min": topology_density.iter().copied().fold(f64::INFINITY, f64::min),
            "max": topology_density.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            "mean": topology_density.iter().sum::<f64>() / topology_density.len() as f64,
        },
        "provenance": {
            "solver": "run_fem_topology_2d",
            "mesh": "6x2 P1 triangular cantilever",
            "reanalysis": "body-fitted active-element production reanalysis",
        },
    });
    std::fs::write(
        output_dir.join("topology-reference.json"),
        serde_json::to_vec_pretty(&topology_json)?,
    )?;
    gallery.heatmap_png(
        "topology-density.png",
        &topology_density,
        "topology-reference.json",
        "Production FEM topology density; protected boundary, volume fraction, and body-fitted reanalysis are recorded in topology-reference.json",
    )?;
    gallery.line_svg(
        "topology-objective.svg",
        &(0..topology.objective_history.len())
            .map(|iteration| iteration as f64)
            .collect::<Vec<_>>(),
        &topology.objective_history,
        "Topology objective history",
        "topology-reference.json",
        "Production objective history over filtered/projected density updates",
    )?;
    sonium_sim_structural_mechanics::append_fem_topology_before_after_gallery(
        &mut gallery,
        output_dir,
        &topology,
    )?;
    let manifest_path = gallery.finish("Structural bracket gallery", vec![
        "1920x1080 fixed raster canvas; deterministic false-colour map".into(),
        format!("display deformation scale: {:.3}x; physical displacements remain metres", artifacts.warp_scale),
        "CPU reference solve; figures map directly to bracket_result.vtu, bracket_report.json, mesh-convergence.json, and topology-reference.json".into(),
    ])?;
    sonium_xem::io::validate_gallery_manifest(&manifest_path)?;
    Ok(())
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn automatic_and_explicit_thread_budgets_parse() {
        let automatic = DemoOptions::parse(&[]).unwrap();
        assert_eq!(automatic.threads, 0);
        assert_eq!(automatic.backend, "cpu");

        let explicit = DemoOptions::parse(&[
            "--threads".into(),
            "12".into(),
            "--backend".into(),
            "cuda".into(),
            "out".into(),
        ])
        .unwrap();
        assert_eq!(explicit.threads, 12);
        assert_eq!(explicit.backend, "cuda");
        assert_eq!(explicit.output_dir, std::path::PathBuf::from("out"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gallery_is_headless_manifested_and_nonblank() {
        let directory = tempfile::tempdir().unwrap();
        let artifacts = sonium_sim_structural_mechanics::write_introductory_bracket_visualization(
            directory.path(),
        )
        .unwrap();
        write_gallery(directory.path(), &artifacts).unwrap();
        assert_eq!(
            sonium_xem::io::validate_png_nonblank(
                directory.path().join("bracket-displacement.png")
            )
            .unwrap(),
            (1920, 1080)
        );
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.path().join("gallery.json")).unwrap())
                .unwrap();
        assert!(manifest["figures"].as_array().unwrap().len() >= 12);
        assert!(
            manifest["figures"]
                .as_array()
                .unwrap()
                .iter()
                .all(|figure| figure["source_checksum_fnv1a64"].is_string())
        );
        let convergence: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.path().join("mesh-convergence.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            convergence["schema"],
            "sonium-structural.mesh-convergence.v1"
        );
        assert!(convergence["converged"].as_bool().unwrap());
        assert!(directory.path().join("mesh-convergence.svg").is_file());
        let topology: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.path().join("topology-reference.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(topology["schema"], "sonium-structural.topology-design.v1");
        assert!(topology["connected"].as_bool().unwrap());
        assert!(topology["reanalysis_feasible"].as_bool().unwrap());
        assert!(directory.path().join("topology-density.png").is_file());
        assert!(
            directory
                .path()
                .join("topology-density-initial.png")
                .is_file()
        );
        assert!(
            directory
                .path()
                .join("topology-density-final.png")
                .is_file()
        );
        assert!(
            directory
                .path()
                .join("topology-density-delta.svg")
                .is_file()
        );
        assert!(
            directory
                .path()
                .join("topology-before-after.json")
                .is_file()
        );
        assert!(directory.path().join("topology-objective.svg").is_file());
    }

    #[test]
    fn gallery_renderer_is_byte_deterministic_for_fixed_fixture() {
        fn render_fixture(directory: &std::path::Path) {
            std::fs::write(directory.join("fixture.json"), b"{\"schema\":1}").unwrap();
            let mut gallery = sonium_xem::io::GalleryRenderer::new(directory).unwrap();
            gallery
                .heatmap_png(
                    "fixture.png",
                    &[0.0, 0.25, 0.75, 1.0],
                    "fixture.json",
                    "fixed fixture heatmap",
                )
                .unwrap();
            gallery
                .line_svg(
                    "fixture.svg",
                    &[0.0, 1.0, 2.0],
                    &[0.0, 0.5, 1.0],
                    "fixed fixture curve",
                    "fixture.json",
                    "fixed fixture curve",
                )
                .unwrap();
            gallery
                .finish(
                    "Structural gallery renderer fixture",
                    vec!["fixed camera".into()],
                )
                .unwrap();
        }

        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        render_fixture(first.path());
        render_fixture(second.path());
        for filename in ["fixture.png", "fixture.svg", "gallery.json"] {
            assert_eq!(
                std::fs::read(first.path().join(filename)).unwrap(),
                std::fs::read(second.path().join(filename)).unwrap(),
                "gallery artifact {filename} is not byte-deterministic"
            );
        }
    }
}
