//! Literal Sonium counterpart of Wolfram verification case
//! `FEM-Stationary-2D-Single-HeatTransfer-0002-A`.
//!
//! A 20 mm by 10 mm ceramic strip has both vertical sides fixed at 1173 K,
//! an insulated bottom, and convection plus Stefan--Boltzmann radiation from
//! the top into a 323 K environment. Wolfram publishes three temperatures and
//! accepts the Euclidean norm of their relative errors below 1%.

use serde::Deserialize;
use sonium_fem::{
    assembly::{TripletMatrix, assemble_bilinear, assemble_boundary_mass},
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_dirichlet_values},
    mesh::{Mesh, interpolate_scalar_at_point, rectangular_mesh_triangles},
    physics::thermal::heat_conduction::{ConductionForm, RadiationResidual},
    spaces::H1Space,
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{GmresConfig, gmres};
use std::{env, path::PathBuf, process::Command};

const WIDTH_M: f64 = 0.02;
const HEIGHT_M: f64 = 0.01;
const CONDUCTIVITY_W_M_K: f64 = 3.0;
const SIDE_TEMPERATURE_K: f64 = 1173.0;
const AMBIENT_TEMPERATURE_K: f64 = 323.0;
const CONVECTION_W_M2_K: f64 = 50.0;
const EMISSIVITY: f64 = 0.7;
const TOP_MARKER: i32 = 4;
const NX: usize = 64;
const NY: usize = 32;

const SAMPLE_POINTS_M: [[f64; 2]; 3] = [[0.005, 0.0], [0.005, 0.005], [0.01, 0.005]];
const PUBLISHED_TEMPERATURE_K: [f64; 3] = [1111.38, 1092.37, 1064.21];

struct SoniumSolution {
    temperature_k: Vec<f64>,
    samples_k: [f64; 3],
    iterations: usize,
    heat_balance_relative_error: f64,
}

fn add_radiation_terms(
    mesh: &Mesh,
    temperature: &[f64],
    radiation: &RadiationResidual,
    residual: &mut [f64],
    jacobian: &mut TripletMatrix<f64>,
) -> f64 {
    // Three-point Gauss--Legendre exactly integrates the P1-temperature
    // fourth-power flux along each straight boundary edge.
    let xi = [-(3.0_f64 / 5.0).sqrt(), 0.0, (3.0_f64 / 5.0).sqrt()];
    let weights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0];
    let mut outward_power = 0.0;

    for face in &mesh.boundaries {
        if face.marker != TOP_MARKER || face.nodes.len() != 2 {
            continue;
        }
        let nodes = [face.nodes[0], face.nodes[1]];
        let p0 = mesh.nodes[nodes[0]];
        let p1 = mesh.nodes[nodes[1]];
        let edge_length = ((p1.x - p0.x).powi(2) + (p1.y - p0.y).powi(2)).sqrt();

        for (&gauss_xi, &gauss_weight) in xi.iter().zip(&weights) {
            let s = 0.5 * (1.0 + gauss_xi);
            let shape = [1.0 - s, s];
            let weight = 0.5 * edge_length * gauss_weight;
            let surface_temperature =
                shape[0] * temperature[nodes[0]] + shape[1] * temperature[nodes[1]];
            let flux = radiation.heat_flux(surface_temperature);
            let tangent = radiation.linearized_h(surface_temperature);
            outward_power += weight * flux;

            for local_i in 0..2 {
                let global_i = nodes[local_i];
                residual[global_i] += weight * shape[local_i] * flux;
                for local_j in 0..2 {
                    jacobian.push(
                        global_i,
                        nodes[local_j],
                        weight * shape[local_i] * shape[local_j] * tangent,
                    );
                }
            }
        }
    }
    outward_power
}

fn solve_increment(system: &FemSystem<f64>) -> Vec<f64> {
    let result = gmres(
        &system.matrix.to_csr(),
        &system.rhs,
        &GmresConfig {
            max_iterations: 4_000,
            restart: 80,
            tolerance: 1.0e-11,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(
        result.converged,
        "radiative heat Newton increment did not converge"
    );
    result.x
}

fn sonium_solution() -> SoniumSolution {
    let mesh = rectangular_mesh_triangles(0.0, WIDTH_M, 0.0, HEIGHT_M, NX, NY);
    let space = H1Space::new(PolynomialDegree::P1);
    let stiffness = assemble_bilinear(
        &ConductionForm::from_conductivity(CONDUCTIVITY_W_M_K),
        &space,
        &space,
        &mesh,
        2,
    );
    let top_mass = assemble_boundary_mass(&mesh, PolynomialDegree::P1, TOP_MARKER);

    // Linear conduction and convection Jacobian.
    let mut linear_jacobian = stiffness.clone();
    for entry in 0..top_mass.nnz() {
        linear_jacobian.push(
            top_mass.rows[entry],
            top_mass.cols[entry],
            CONVECTION_W_M2_K * top_mass.values[entry],
        );
    }
    let linear_csr = linear_jacobian.to_csr();

    // h T_inf integral(N_i) on the right-hand side.
    let mut ambient_load = vec![0.0; mesh.num_nodes()];
    for entry in 0..top_mass.nnz() {
        ambient_load[top_mass.rows[entry]] +=
            CONVECTION_W_M2_K * AMBIENT_TEMPERATURE_K * top_mass.values[entry];
    }

    let radiation = RadiationResidual::new(EMISSIVITY, AMBIENT_TEMPERATURE_K);
    let mut temperature = vec![SIDE_TEMPERATURE_K; mesh.num_nodes()];
    let mut iterations = 0;
    for iteration in 1..=20 {
        let mut residual = linear_csr.matvec(&temperature);
        for (value, load) in residual.iter_mut().zip(&ambient_load) {
            *value -= load;
        }
        let mut jacobian = linear_jacobian.clone();
        add_radiation_terms(
            &mesh,
            &temperature,
            &radiation,
            &mut residual,
            &mut jacobian,
        );

        let rhs = residual.into_iter().map(|value| -value).collect();
        let mut system = FemSystem::new(jacobian, rhs);
        for marker in [1, 2] {
            apply_dirichlet_values(&mut system, &mesh, &space, marker, &|_, _, _| 0.0, None);
        }
        let increment = solve_increment(&system);
        let max_increment = increment
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        for (value, delta) in temperature.iter_mut().zip(increment) {
            *value += delta;
        }
        iterations = iteration;
        if max_increment < 1.0e-8 {
            break;
        }
    }
    assert!(
        iterations < 20,
        "radiative heat Newton solve did not converge"
    );

    let samples_k = SAMPLE_POINTS_M.map(|[x, y]| {
        interpolate_scalar_at_point(&mesh, &temperature, [x, y, 0.0])
            .expect("published Wolfram sample must lie inside the strip")
    });

    // The complete unconstrained weak residual at the fixed side nodes is the
    // heat supplied by those boundaries. Include the top-flux share assigned
    // to the two corner nodes, then compare it with the integrated top loss.
    let mut final_residual = linear_csr.matvec(&temperature);
    for (value, load) in final_residual.iter_mut().zip(&ambient_load) {
        *value -= load;
    }
    let mut unused_jacobian = TripletMatrix::new(mesh.num_nodes(), mesh.num_nodes());
    let final_radiation_power = add_radiation_terms(
        &mesh,
        &temperature,
        &radiation,
        &mut final_residual,
        &mut unused_jacobian,
    );
    let side_input = (0..mesh.num_nodes())
        .filter(|&node| {
            let x = mesh.nodes[node].x;
            x.abs() < 1.0e-12 || (x - WIDTH_M).abs() < 1.0e-12
        })
        .map(|node| final_residual[node])
        .sum::<f64>();
    let convection_power = top_mass
        .rows
        .iter()
        .zip(&top_mass.values)
        .map(|(&row, &value)| {
            CONVECTION_W_M2_K * value * (temperature[row] - AMBIENT_TEMPERATURE_K)
        })
        .sum::<f64>();
    let top_loss = convection_power + final_radiation_power;
    let heat_balance_relative_error = (side_input - top_loss).abs() / top_loss.abs();

    SoniumSolution {
        temperature_k: temperature,
        samples_k,
        iterations,
        heat_balance_relative_error,
    }
}

fn relative_error_norm(actual: &[f64; 3], reference: &[f64; 3]) -> f64 {
    actual
        .iter()
        .zip(reference)
        .map(|(&value, &expected)| ((value - expected) / expected).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    sample_points_m: Vec<[f64; 2]>,
    temperature_k: Vec<f64>,
    published_temperature_k: Vec<f64>,
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/heat_convection_radiation.wls");
    let output = Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
        .unwrap_or_else(|error| panic!("failed to launch {executable:?}: {error}"));
    assert!(
        output.status.success(),
        "Wolfram convection/radiation solve failed:\n{}",
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
fn sonium_reproduces_wolfram_convection_and_radiation_strip() {
    let sonium = sonium_solution();
    let error = relative_error_norm(&sonium.samples_k, &PUBLISHED_TEMPERATURE_K);
    eprintln!(
        "Sonium samples={:?} relative_norm={error:.6e} Newton_iterations={} heat_balance={:.3e}",
        sonium.samples_k, sonium.iterations, sonium.heat_balance_relative_error
    );
    assert!(
        error < 1.0e-2,
        "Sonium relative error norm {error:.6e} exceeds Wolfram's 1% gate"
    );
    assert!(
        sonium.heat_balance_relative_error < 1.0e-8,
        "heat-balance relative error={:.6e}",
        sonium.heat_balance_relative_error
    );
    assert!(sonium.temperature_k.iter().all(|value| value.is_finite()));

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.sample_points_m, SAMPLE_POINTS_M);
        assert_eq!(reference.published_temperature_k, PUBLISHED_TEMPERATURE_K);
        let temperatures: [f64; 3] = reference
            .temperature_k
            .try_into()
            .expect("Wolfram must return three published probes");
        assert!(relative_error_norm(&temperatures, &PUBLISHED_TEMPERATURE_K) < 1.0e-2);
        assert!(relative_error_norm(&sonium.samples_k, &temperatures) < 1.0e-2);
    }
}
