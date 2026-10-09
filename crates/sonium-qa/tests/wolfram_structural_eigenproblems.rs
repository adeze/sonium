//! Sonium counterparts of published Wolfram structural NDEigensystem examples.

use serde::Deserialize;
use sonium_fem::{
    mesh::generators::rectangular_mesh_triangles,
    physics::{
        solid::eigenmodes::{StructuralModalConfig, build_structural_modal_problem},
        solid::elasticity::IsotropicMaterial,
    },
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_sim_structural_mechanics::{
    StructuralMaterial,
    beam::{Beam2DGeometry, Beam2DStudy, run_beam2d_modal},
};
use sonium_solvers::LobpcgConfig;
use std::{env, path::PathBuf, process::Command};

#[derive(Deserialize)]
struct WolframStructuralReference {
    schema_version: u32,
    structural: Vec<f64>,
    beam_frequencies_hz: Vec<f64>,
}

fn solve_plane_stress_cantilever() -> Vec<f64> {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.2, 48, 10);
    let problem = build_structural_modal_problem(
        &mesh,
        IsotropicMaterial::new(10.0, 0.25),
        &StructuralModalConfig::plane_stress(2.0, vec![1, 2, 3, 4]),
    );
    let mut eigenvalues = problem
        .solve(&LobpcgConfig {
            num_eigenvalues: 3,
            max_iterations: 1_000,
            tolerance: 1.0e-8,
            block_padding: 5,
        })
        .expect("Sonium should solve the constrained plane-stress modes")
        .eigenvalues;
    eigenvalues.sort_by(f64::total_cmp);
    eigenvalues
}

fn solve_cantilever_beam() -> Vec<f64> {
    run_beam2d_modal(&Beam2DStudy {
        name: "wolfram-cantilever-beam".into(),
        geometry: Beam2DGeometry {
            length_m: 1.0,
            width_m: 0.1,
            height_m: 0.05,
            elements: 40,
        },
        material: StructuralMaterial {
            youngs_modulus_pa: 2.0e7,
            poisson_ratio: 0.3,
            density_kg_m3: 1000.0,
            damping_ratio: 0.0,
        },
        vertical_tip_force_n: -1.0,
        timoshenko: false,
        modal_modes: 3,
    })
    .expect("Sonium should solve the cantilever beam modes")
    .frequencies_hz
}

fn wolfram_reference() -> Option<WolframStructuralReference> {
    let Some(executable) = env::var_os("WOLFRAMSCRIPT") else {
        return Some(
            serde_json::from_str(include_str!(
                "../wolfram/goldens/structural_eigenproblems.json"
            ))
            .expect("parse checked-in Wolfram structural reference"),
        );
    };
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/structural_eigenproblems.wls");
    let mut command = Command::new(&executable);
    command.arg("-file").arg(script);
    #[cfg(target_os = "macos")]
    if env::var_os("WolframKernel").is_none() {
        let kernel = PathBuf::from(
            "/Applications/Wolfram Engine.app/Contents/Resources/Wolfram Player.app/Contents/MacOS/WolframKernel",
        );
        if kernel.is_file() {
            command.env("WolframKernel", kernel);
        }
    }
    let output = match command.wolfram_output() {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Wolfram Engine not available; skipping live structural comparisons");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram structural examples failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram structural output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

fn assert_spectrum_close(name: &str, sonium: &[f64], wolfram: &[f64], tolerance: f64) {
    assert_eq!(sonium.len(), wolfram.len());
    for (mode, (&actual, &reference)) in sonium.iter().zip(wolfram).enumerate() {
        let relative_error = (actual - reference).abs() / reference.abs().max(1.0);
        assert!(
            relative_error < tolerance,
            "{name} mode {}: Sonium={actual:.9}, Wolfram={reference:.9}, error={:.3}%",
            mode + 1,
            100.0 * relative_error
        );
    }
}

#[test]
fn sonium_reproduces_wolfram_structural_eigenproblem_families() {
    let structural = solve_plane_stress_cantilever();
    let beam = solve_cantilever_beam();
    assert!(
        structural
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
    );
    assert!(beam.windows(2).all(|pair| pair[0] < pair[1]));

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_spectrum_close(
            "Constrained plane-stress rectangle",
            &structural,
            &reference.structural,
            0.06,
        );
        assert_spectrum_close(
            "Euler-Bernoulli cantilever",
            &beam,
            &reference.beam_frequencies_hz,
            0.015,
        );
    }
}
