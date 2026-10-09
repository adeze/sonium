//! Literal Sonium counterpart of Wolfram/NAFEMS verification case
//! `FEM-Stationary-2D-Single-HeatTransfer-0001-A`.
//!
//! The 0.6 m by 1 m plate has a 100 K lower edge, insulated left edge, and
//! convection to 0 K on the top and right edges. Wolfram publishes 18.25 K at
//! `(0.6 m, 0.2 m)` and requires relative error below 0.1%.

use serde::Deserialize;
use sonium_fem::{
    assembly::{MassMatrix, TripletMatrix, assemble_bilinear, assemble_boundary_mass},
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_dirichlet_values},
    mesh::{interpolate_scalar_at_point, rectangular_mesh_triangles},
    physics::thermal::heat_conduction::ConductionForm,
    spaces::H1Space,
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{GmresConfig, gmres};
use std::{env, path::PathBuf, process::Command};

const WIDTH_M: f64 = 0.6;
const HEIGHT_M: f64 = 1.0;
const CONDUCTIVITY_W_M_K: f64 = 52.0;
const FIXED_TEMPERATURE_K: f64 = 100.0;
const CONVECTION_W_M2_K: f64 = 750.0;
const AMBIENT_TEMPERATURE_K: f64 = 0.0;
const PUBLISHED_TEMPERATURE_K: f64 = 18.25;
const SAMPLE_POINT_M: [f64; 2] = [0.6, 0.2];
const NX: usize = 60;
const NY: usize = 100;

fn add_scaled_mass(matrix: &mut TripletMatrix<f64>, mass: &MassMatrix, scale: f64) {
    for entry in 0..mass.nnz() {
        matrix.push(
            mass.rows[entry],
            mass.cols[entry],
            scale * mass.values[entry],
        );
    }
}

struct SoniumSolution {
    sample_temperature_k: f64,
    heat_balance_relative_error: f64,
}

fn sonium_solution() -> SoniumSolution {
    let mesh = rectangular_mesh_triangles(0.0, WIDTH_M, 0.0, HEIGHT_M, NX, NY);
    let space = H1Space::new(PolynomialDegree::P1);
    let mut matrix = assemble_bilinear(
        &ConductionForm::from_conductivity(CONDUCTIVITY_W_M_K),
        &space,
        &space,
        &mesh,
        2,
    );
    let right_mass = assemble_boundary_mass(&mesh, PolynomialDegree::P1, 2);
    let top_mass = assemble_boundary_mass(&mesh, PolynomialDegree::P1, 4);
    add_scaled_mass(&mut matrix, &right_mass, CONVECTION_W_M2_K);
    add_scaled_mass(&mut matrix, &top_mass, CONVECTION_W_M2_K);
    let unconstrained = matrix.to_csr();

    // The official ambient temperature is zero, so the Robin load vanishes.
    let mut system = FemSystem::new(matrix, vec![0.0; mesh.num_nodes()]);
    apply_dirichlet_values(
        &mut system,
        &mesh,
        &space,
        3,
        &|_, _, _| FIXED_TEMPERATURE_K,
        None,
    );
    let result = gmres(
        &system.matrix.to_csr(),
        &system.rhs,
        &GmresConfig {
            max_iterations: 8_000,
            restart: 100,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(result.converged, "NAFEMS convection solve did not converge");

    let sample_temperature_k = interpolate_scalar_at_point(
        &mesh,
        &result.x,
        [SAMPLE_POINT_M[0], SAMPLE_POINT_M[1], 0.0],
    )
    .expect("published NAFEMS probe must lie on the plate boundary");

    // Sum the unconstrained residual on the fixed lower edge and compare it
    // with convection leaving the right and top edges.
    let residual = unconstrained.matvec(&result.x);
    let bottom_input = (0..mesh.num_nodes())
        .filter(|&node| mesh.nodes[node].y.abs() < 1.0e-12)
        .map(|node| residual[node])
        .sum::<f64>();
    let convection_power = [&right_mass, &top_mass]
        .into_iter()
        .flat_map(|mass| {
            mass.rows.iter().zip(&mass.values).map(|(&row, &value)| {
                CONVECTION_W_M2_K * value * (result.x[row] - AMBIENT_TEMPERATURE_K)
            })
        })
        .sum::<f64>();
    let heat_balance_relative_error =
        (bottom_input - convection_power).abs() / convection_power.abs();

    SoniumSolution {
        sample_temperature_k,
        heat_balance_relative_error,
    }
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    sample_point_m: [f64; 2],
    temperature_k: f64,
    published_temperature_k: f64,
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/heat_convection.wls");
    let output = Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
        .unwrap_or_else(|error| panic!("failed to launch {executable:?}: {error}"));
    assert!(
        output.status.success(),
        "Wolfram convection solve failed:\n{}",
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

#[test]
fn sonium_reproduces_wolfram_nafems_convection_plate() {
    let sonium = sonium_solution();
    let relative_error =
        ((sonium.sample_temperature_k - PUBLISHED_TEMPERATURE_K) / PUBLISHED_TEMPERATURE_K).abs();
    eprintln!(
        "Sonium T(0.6,0.2)={:.8} K relative_error={relative_error:.6e} heat_balance={:.3e}",
        sonium.sample_temperature_k, sonium.heat_balance_relative_error
    );
    assert!(
        relative_error < 1.0e-3,
        "Sonium relative error {relative_error:.6e} exceeds Wolfram's 0.1% gate"
    );
    assert!(sonium.heat_balance_relative_error < 1.0e-9);

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.sample_point_m, SAMPLE_POINT_M);
        assert_eq!(reference.published_temperature_k, PUBLISHED_TEMPERATURE_K);
        assert!(
            ((reference.temperature_k - PUBLISHED_TEMPERATURE_K) / PUBLISHED_TEMPERATURE_K).abs()
                < 1.0e-3
        );
        assert!(
            ((sonium.sample_temperature_k - reference.temperature_k) / reference.temperature_k)
                .abs()
                < 1.0e-3
        );
    }
}
