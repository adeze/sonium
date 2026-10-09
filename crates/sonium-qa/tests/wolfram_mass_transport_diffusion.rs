//! Gaussian benchmark derived from the diffusion-only special case published
//! in Wolfram's mass-transport monograph; not its literal reactor geometry.

use serde::Deserialize;
use sonium_fem::{
    assembly::assemble_bilinear,
    basis::PolynomialDegree,
    mesh::rectangular_mesh_triangles,
    physics::{
        fluid::transport::{TransportCapacityForm, TransportDiffusionForm, TransportMaterial},
        thermal::heat_conduction::ThermalTimeStepper,
    },
    spaces::{FunctionSpace, H1Space},
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{LinearSolveConfig, LinearSolverType};
use std::{env, path::PathBuf, process::Command};

const DIFFUSIVITY_M2_S: f64 = 1.0;
const INITIAL_SIGMA_M: f64 = 0.5;
const FINAL_TIME_S: f64 = 0.01;
const SAMPLE_X_M: [f64; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];

fn exact_concentration(x: f64, time: f64) -> f64 {
    let variance = INITIAL_SIGMA_M.powi(2) + 2.0 * DIFFUSIVITY_M2_S * time;
    (-x.powi(2) / (2.0 * variance)).exp() / (2.0 * std::f64::consts::PI * variance).sqrt()
}

fn sonium_solution() -> (Vec<f64>, f64) {
    let mesh = rectangular_mesh_triangles(-5.0, 5.0, 0.0, 0.1, 200, 2);
    let space = H1Space::new(PolynomialDegree::P1);
    let material = TransportMaterial::isotropic(DIFFUSIVITY_M2_S, [0.0; 3], 0.0);
    let mass = assemble_bilinear(
        &TransportCapacityForm::new(&material),
        &space,
        &space,
        &mesh,
        2,
    )
    .to_csr();
    let stiffness = assemble_bilinear(
        &TransportDiffusionForm::new(&material),
        &space,
        &space,
        &mesh,
        2,
    )
    .to_csr();
    let mut concentration: Vec<f64> = mesh
        .nodes
        .iter()
        .map(|node| exact_concentration(node.x, 0.0))
        .collect();
    assert_eq!(concentration.len(), space.num_global_dofs(&mesh));
    let initial_mass: f64 = mass.matvec(&concentration).iter().sum();
    let zero_source = vec![0.0; concentration.len()];
    let stepper = ThermalTimeStepper::crank_nicolson(0.001);
    let solve_config = LinearSolveConfig {
        solver: LinearSolverType::GmresIlu,
        max_iterations: 1_000,
        tolerance: 1.0e-12,
        restart: 50,
        exact_residual_check_interval: 0,
        verbosity: 0,
    };
    for _ in 0..10 {
        let result = stepper
            .step(
                &concentration,
                &zero_source,
                Some(&zero_source),
                &mass,
                &stiffness,
                &solve_config,
            )
            .expect("transport time step should solve");
        assert!(result.converged);
        concentration = result.x;
    }
    let final_mass: f64 = mass.matvec(&concentration).iter().sum();
    let mass_drift = (final_mass - initial_mass).abs() / initial_mass.abs();
    let samples = SAMPLE_X_M
        .iter()
        .map(|&sample_x| {
            let index = mesh
                .nodes
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| (a.x - sample_x).abs().total_cmp(&(b.x - sample_x).abs()))
                .expect("transport mesh should have nodes")
                .0;
            concentration[index]
        })
        .collect();
    (samples, mass_drift)
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    diffusivity_m2_s: f64,
    initial_sigma_m: f64,
    final_time_s: f64,
    sample_x_m: Vec<f64>,
    concentration: Vec<f64>,
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/mass_transport_diffusion.wls");
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
            eprintln!("Wolfram Engine not available; using analytical transport reference only");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram mass-transport reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram transport output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

fn assert_concentration_close(actual: &[f64], reference: &[f64]) {
    assert_eq!(actual.len(), reference.len());
    for (sample, (&got, &want)) in actual.iter().zip(reference).enumerate() {
        assert!(
            (got - want).abs() < 0.005,
            "sample {sample}: concentration={got:.9}, reference={want:.9}"
        );
    }
}

#[test]
fn sonium_reproduces_wolfram_diffusion_only_mass_transport() {
    let (sonium, mass_drift) = sonium_solution();
    let exact = SAMPLE_X_M.map(|x| exact_concentration(x, FINAL_TIME_S));
    assert_concentration_close(&sonium, &exact);
    assert!(
        mass_drift < 1.0e-9,
        "discrete total-mass drift {mass_drift:.3e} exceeds 1e-9"
    );
    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.diffusivity_m2_s, DIFFUSIVITY_M2_S);
        assert_eq!(reference.initial_sigma_m, INITIAL_SIGMA_M);
        assert_eq!(reference.final_time_s, FINAL_TIME_S);
        assert_eq!(reference.sample_x_m, SAMPLE_X_M);
        assert_concentration_close(&reference.concentration, &exact);
        assert_concentration_close(&sonium, &reference.concentration);
    }
}
