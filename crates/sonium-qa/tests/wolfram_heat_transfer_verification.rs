//! Sonium counterpart of the sinusoidal-plate case in Wolfram's published
//! heat-transfer verification suite.
//!
//! Source:
//! <https://reference.wolfram.com/language/PDEModels/tutorial/HeatTransfer/HeatTransferVerificationTests.html?view=all>

use serde::Deserialize;
use sonium_fem::{
    assembly::{assemble_bilinear, assemble_linear},
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_dirichlet_values},
    mesh::generators::unit_square_triangles,
    physics::thermal::heat_conduction::{ConductionForm, HeatSourceForm, ThermalMaterial},
    spaces::H1Space,
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{GmresConfig, gmres};
use std::{env, path::PathBuf, process::Command};

const CONDUCTIVITY_W_M_K: f64 = 3.0;
const SAMPLE_POINTS: [[f64; 2]; 5] = [
    [0.25, 0.25],
    [0.50, 0.25],
    [0.50, 0.50],
    [0.75, 0.50],
    [0.75, 0.75],
];

fn exact_temperature(x: f64, y: f64) -> f64 {
    (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin()
}

fn sonium_temperature() -> Vec<f64> {
    const CELLS_PER_SIDE: usize = 24;
    let mesh = unit_square_triangles(CELLS_PER_SIDE);
    let space = H1Space::new(PolynomialDegree::P1);
    let material = ThermalMaterial::new(CONDUCTIVITY_W_M_K, 1.0, 1.0);
    let stiffness = assemble_bilinear(&ConductionForm::new(&material), &space, &space, &mesh, 2);
    let source = HeatSourceForm {
        source_fn: |x, y, _| {
            2.0 * CONDUCTIVITY_W_M_K * std::f64::consts::PI.powi(2) * exact_temperature(x, y)
        },
    };
    let rhs = assemble_linear(&source, &space, &mesh, 2);
    let mut system = FemSystem::new(stiffness, rhs);
    for marker in 1..=4 {
        apply_dirichlet_values(&mut system, &mesh, &space, marker, &|_, _, _| 0.0, None);
    }

    let result = gmres(
        &system.matrix.to_csr(),
        &system.rhs,
        &GmresConfig {
            max_iterations: 2_000,
            restart: 50,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(
        result.converged,
        "sinusoidal-plate heat solve did not converge"
    );

    SAMPLE_POINTS
        .iter()
        .map(|&[x, y]| {
            let ix = (x * CELLS_PER_SIDE as f64).round() as usize;
            let iy = (y * CELLS_PER_SIDE as f64).round() as usize;
            result.x[iy * (CELLS_PER_SIDE + 1) + ix]
        })
        .collect()
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    conductivity_w_m_k: f64,
    sample_points: Vec<[f64; 2]>,
    temperature: Vec<f64>,
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("wolfram/heat_transfer_sinusoidal_plate.wls");
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
            eprintln!("Wolfram Engine not available; using analytical heat reference only");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram heat-transfer verification failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram heat output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

fn assert_temperature_close(name: &str, actual: &[f64], reference: &[f64]) {
    assert_eq!(actual.len(), reference.len());
    for (sample, (&got, &want)) in actual.iter().zip(reference).enumerate() {
        assert!(
            (got - want).abs() < 0.003,
            "{name} sample {sample}: Sonium={got:.9}, reference={want:.9}"
        );
    }
}

#[test]
fn sonium_reproduces_wolfram_sinusoidal_plate_temperature() {
    let sonium = sonium_temperature();
    let exact = SAMPLE_POINTS.map(|[x, y]| exact_temperature(x, y));
    assert_temperature_close("Sonium/analytical", &sonium, &exact);

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.conductivity_w_m_k, CONDUCTIVITY_W_M_K);
        assert_eq!(reference.sample_points, SAMPLE_POINTS);
        assert_temperature_close("Wolfram/analytical", &reference.temperature, &exact);
        assert_temperature_close("Sonium/Wolfram", &sonium, &reference.temperature);
    }
}
