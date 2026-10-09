//! Sonium counterpart of Wolfram's published clamped circular-membrane
//! `NDEigensystem` example.
//!
//! Source: <https://reference.wolfram.com/language/ref/NDEigensystem.html>

use serde::Deserialize;
use sonium_fem::{
    basis::PolynomialDegree,
    mesh::circular_mesh_triangles,
    physics::solid::eigenmodes::{AcousticEigenmodeConfig, build_acoustic_eigenmode_problem},
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::LobpcgConfig;
use std::{env, path::PathBuf, process::Command};

const BESSEL_REFERENCE: [f64; 6] = [
    5.783_185_962_946_785,
    14.681_970_642_123_895,
    14.681_970_642_123_895,
    26.374_616_427_163_39,
    26.374_616_427_163_39,
    30.471_262_343_662_087,
];

fn sonium_disk_eigenvalues() -> Vec<f64> {
    let mesh = circular_mesh_triangles(0.0, 0.0, 1.0, 12, 72);
    let problem = build_acoustic_eigenmode_problem(
        &mesh,
        &AcousticEigenmodeConfig {
            degree: PolynomialDegree::P1,
            sound_speed: 1.0,
            pressure_release_boundary_markers: vec![1],
        },
    );
    let mut result = problem
        .solve(&LobpcgConfig {
            num_eigenvalues: 6,
            max_iterations: 400,
            tolerance: 1.0e-8,
            block_padding: 4,
        })
        .expect("Sonium LOBPCG should solve the disk membrane eigenproblem");
    result.eigenvalues.sort_by(f64::total_cmp);
    result.eigenvalues
}

#[derive(Deserialize)]
struct WolframDiskReference {
    schema_version: u32,
    ndeigensystem: Vec<f64>,
    exact: Vec<f64>,
}

fn wolfram_disk_reference() -> Option<WolframDiskReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/disk_membrane.wls");
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
            eprintln!("Wolfram Engine not available; using exact Bessel references only");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram disk example failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram disk output was not JSON: {error}\nstdout:\n{stdout}\n")
    }))
}

#[test]
fn sonium_reproduces_wolfram_clamped_circular_membrane() {
    let actual = sonium_disk_eigenvalues();
    assert_eq!(actual.len(), BESSEL_REFERENCE.len());
    for (mode, (&got, &want)) in actual.iter().zip(&BESSEL_REFERENCE).enumerate() {
        let relative_error = (got - want).abs() / want;
        assert!(
            relative_error < 0.035,
            "disk mode {}: Sonium={got:.9}, exact/Wolfram={want:.9}, relative error={:.3}%",
            mode + 1,
            100.0 * relative_error,
        );
    }

    // Rotational symmetry makes modes 2/3 and 4/5 double eigenvalues. This
    // catches asymmetric mesh or eigensolver regressions independently of the
    // absolute discretization error.
    for (left, right) in [(1, 2), (3, 4)] {
        let split = (actual[right] - actual[left]).abs() / actual[left];
        assert!(
            split < 2.0e-3,
            "degenerate pair split by {:.3}%",
            100.0 * split
        );
    }

    if let Some(reference) = wolfram_disk_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.exact.len(), BESSEL_REFERENCE.len());
        assert_eq!(reference.ndeigensystem.len(), BESSEL_REFERENCE.len());
        for (mode, ((&sonium, &wolfram), &exact)) in actual
            .iter()
            .zip(&reference.ndeigensystem)
            .zip(&reference.exact)
            .enumerate()
        {
            assert!(
                (wolfram - exact).abs() / exact < 0.01,
                "Wolfram mode {} is not mesh-converged",
                mode + 1
            );
            assert!(
                (sonium - wolfram).abs() / wolfram < 0.035,
                "Sonium/Wolfram mode {} mismatch",
                mode + 1
            );
        }
    }
}
