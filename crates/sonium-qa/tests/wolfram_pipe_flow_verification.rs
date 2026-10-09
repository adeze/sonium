//! Sonium counterpart of the stationary circular-pipe case in Wolfram's
//! published fluid-dynamics verification suite.
//!
//! Source:
//! <https://reference.wolfram.com/language/PDEModels/tutorial/FluidDynamics/FluidDynamicsVerificationTests.html?view=all>
//!
//! Fully developed axial flow reduces the Stokes equations to a scalar
//! Poisson problem on the pipe cross-section. This validates Sonium's scalar
//! FEM assembly for that reduction; it is not a mixed velocity-pressure test.

use serde::Deserialize;
use sonium_fem::{
    assembly::{assemble_bilinear, assemble_linear},
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_dirichlet_values},
    mesh::generators::circular_mesh_triangles,
    physics::thermal::heat_conduction::{ConductionForm, HeatSourceForm},
    spaces::H1Space,
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{GmresConfig, gmres};
use std::{env, path::PathBuf, process::Command};

const PIPE_RADIUS_M: f64 = 0.05;
const DYNAMIC_VISCOSITY_PA_S: f64 = 0.001;
const PRESSURE_GRADIENT_PA_M: f64 = -2.0;
const SAMPLE_RADII_M: [f64; 6] = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05];

fn exact_velocity(radius: f64) -> f64 {
    -PRESSURE_GRADIENT_PA_M * (PIPE_RADIUS_M.powi(2) - radius.powi(2))
        / (4.0 * DYNAMIC_VISCOSITY_PA_S)
}

fn sonium_velocity() -> Vec<f64> {
    const RADIAL_LAYERS: usize = 10;
    const ANGULAR_DIVISIONS: usize = 48;
    let mesh = circular_mesh_triangles(0.0, 0.0, PIPE_RADIUS_M, RADIAL_LAYERS, ANGULAR_DIVISIONS);
    let space = H1Space::new(PolynomialDegree::P1);
    let stiffness = assemble_bilinear(
        &ConductionForm::from_conductivity(DYNAMIC_VISCOSITY_PA_S),
        &space,
        &space,
        &mesh,
        2,
    );
    let source = HeatSourceForm {
        source_fn: |_, _, _| -PRESSURE_GRADIENT_PA_M,
    };
    let rhs = assemble_linear(&source, &space, &mesh, 2);
    let mut system = FemSystem::new(stiffness, rhs);
    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| 0.0, None);

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
        "stationary pipe-flow solve did not converge"
    );

    SAMPLE_RADII_M
        .iter()
        .map(|&radius| {
            if radius == 0.0 {
                return result.x[0];
            }
            let layer = (radius / PIPE_RADIUS_M * RADIAL_LAYERS as f64).round() as usize;
            result.x[1 + (layer - 1) * ANGULAR_DIVISIONS]
        })
        .collect()
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    pipe_radius_m: f64,
    dynamic_viscosity_pa_s: f64,
    pressure_gradient_pa_m: f64,
    sample_radii_m: Vec<f64>,
    axial_velocity_m_s: Vec<f64>,
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/pipe_flow_verification.wls");
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
            eprintln!("Wolfram Engine not available; using analytical pipe-flow reference only");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram pipe-flow verification failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram pipe output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

fn assert_velocity_close(name: &str, actual: &[f64], reference: &[f64]) {
    assert_eq!(actual.len(), reference.len());
    for (sample, (&got, &want)) in actual.iter().zip(reference).enumerate() {
        assert!(
            (got - want).abs() < 0.0125,
            "{name} sample {sample}: velocity={got:.9} m/s, reference={want:.9} m/s"
        );
    }
}

#[test]
fn sonium_reproduces_wolfram_stationary_pipe_velocity_profile() {
    let sonium = sonium_velocity();
    let exact = SAMPLE_RADII_M.map(exact_velocity);
    assert_velocity_close("Sonium/analytical", &sonium, &exact);

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.pipe_radius_m, PIPE_RADIUS_M);
        assert_eq!(reference.dynamic_viscosity_pa_s, DYNAMIC_VISCOSITY_PA_S);
        assert_eq!(reference.pressure_gradient_pa_m, PRESSURE_GRADIENT_PA_M);
        assert_eq!(reference.sample_radii_m, SAMPLE_RADII_M);
        assert_velocity_close("Wolfram/analytical", &reference.axial_velocity_m_s, &exact);
        assert_velocity_close("Sonium/Wolfram", &sonium, &reference.axial_velocity_m_s);
    }
}
