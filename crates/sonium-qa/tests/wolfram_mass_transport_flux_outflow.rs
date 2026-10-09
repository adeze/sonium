//! Literal transient conservative mass-flux example from Wolfram's mass-
//! transport monograph.
//!
//! Wolfram applies a flux of 10 at the left boundary through t=50 s, with
//! D=0.01, v=0.02, an outflow boundary at x=1 m, and solves through t=100 s.
//! Sonium currently has no high-level MassFluxValue/MassOutflowValue adapter,
//! so this test assembles the actual transport volume forms and adds the P1
//! boundary load locally.

use serde::Deserialize;
use sonium_fem::{
    assembly::assemble_bilinear,
    basis::PolynomialDegree,
    mesh::{Mesh, rectangular_mesh_triangles},
    physics::fluid::transport::{
        TransportAdvectionForm, TransportCapacityForm, TransportDiffusionForm, TransportMaterial,
    },
    spaces::{FunctionSpace, H1Space},
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{CsrMatrix, SparseLuFactorization, TripletMatrix};
use std::{env, path::PathBuf, process::Command};

const LEFT_M: f64 = 0.0;
const RIGHT_M: f64 = 1.0;
const STRIP_WIDTH_M: f64 = 0.01;
const DIFFUSIVITY_M2_S: f64 = 0.01;
const VELOCITY_M_S: f64 = 0.02;
const FLUX: f64 = 10.0;
const FLUX_END_TIME_S: f64 = 50.0;
const FINAL_TIME_S: f64 = 100.0;
const TIME_STEP_S: f64 = 0.1;
const SAMPLE_X_M: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
const SAMPLE_TIME_S: [f64; 4] = [25.0, 50.0, 75.0, 100.0];

fn linear_combination(
    first: &CsrMatrix<f64>,
    first_scale: f64,
    second: &CsrMatrix<f64>,
    second_scale: f64,
) -> CsrMatrix<f64> {
    let mut result = TripletMatrix::new(first.num_rows, first.num_cols);
    for row in 0..first.num_rows {
        for entry in first.row_ptrs[row]..first.row_ptrs[row + 1] {
            result.push(
                row,
                first.col_indices[entry],
                first_scale * first.values[entry],
            );
        }
        for entry in second.row_ptrs[row]..second.row_ptrs[row + 1] {
            result.push(
                row,
                second.col_indices[entry],
                second_scale * second.values[entry],
            );
        }
    }
    result.to_csr()
}

/// P1 line-integral weights on a vertical boundary, reconstructed from its
/// ordered nodes. Their sum is the physical boundary length.
fn vertical_boundary_weights(mesh: &Mesh, x: f64) -> Vec<f64> {
    let mut boundary_nodes: Vec<usize> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| ((node.x - x).abs() < 1.0e-12).then_some(index))
        .collect();
    boundary_nodes.sort_by(|&a, &b| mesh.nodes[a].y.total_cmp(&mesh.nodes[b].y));
    let mut weights = vec![0.0; mesh.nodes.len()];
    for pair in boundary_nodes.windows(2) {
        let length = (mesh.nodes[pair[1]].y - mesh.nodes[pair[0]].y).abs();
        weights[pair[0]] += 0.5 * length;
        weights[pair[1]] += 0.5 * length;
    }
    assert!((weights.iter().sum::<f64>() - STRIP_WIDTH_M).abs() < 1.0e-14);
    weights
}

/// Boundary mass matrix used by conservative inflow. With Sonium's strong
/// `v . grad(c)` volume form, `MassFluxValue` requires `-v.n c` on the inflow
/// side so the prescribed load represents total, rather than diffusive, flux.
fn vertical_boundary_mass(mesh: &Mesh, x: f64) -> CsrMatrix<f64> {
    let mut boundary_nodes: Vec<usize> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| ((node.x - x).abs() < 1.0e-12).then_some(index))
        .collect();
    boundary_nodes.sort_by(|&a, &b| mesh.nodes[a].y.total_cmp(&mesh.nodes[b].y));
    let mut matrix = TripletMatrix::new(mesh.nodes.len(), mesh.nodes.len());
    for pair in boundary_nodes.windows(2) {
        let length = (mesh.nodes[pair[1]].y - mesh.nodes[pair[0]].y).abs();
        matrix.push(pair[0], pair[0], length / 3.0);
        matrix.push(pair[0], pair[1], length / 6.0);
        matrix.push(pair[1], pair[0], length / 6.0);
        matrix.push(pair[1], pair[1], length / 3.0);
    }
    matrix.to_csr()
}

fn dot(weights: &[f64], values: &[f64]) -> f64 {
    weights.iter().zip(values).map(|(a, b)| a * b).sum()
}

fn profile_samples(mesh: &Mesh, concentration: &[f64]) -> Vec<f64> {
    SAMPLE_X_M
        .iter()
        .map(|&sample_x| {
            let mut count = 0_usize;
            let sum: f64 = mesh
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| (node.x - sample_x).abs() < 1.0e-12)
                .map(|(index, _)| {
                    count += 1;
                    concentration[index]
                })
                .sum();
            assert!(count > 0, "sample x={sample_x} must be a mesh line");
            sum / count as f64
        })
        .collect()
}

#[derive(Debug)]
struct SoniumResult {
    final_profile: Vec<f64>,
    outlet_history: Vec<f64>,
    relative_balance_error: f64,
    minimum_concentration: f64,
}

fn sonium_solution(nx: usize) -> SoniumResult {
    let mesh = rectangular_mesh_triangles(LEFT_M, RIGHT_M, 0.0, STRIP_WIDTH_M, nx, 2);
    let space = H1Space::new(PolynomialDegree::P1);
    let material =
        TransportMaterial::with_capacity(DIFFUSIVITY_M2_S, [VELOCITY_M_S, 0.0, 0.0], 0.0, 1.0);
    let mass = assemble_bilinear(
        &TransportCapacityForm::new(&material),
        &space,
        &space,
        &mesh,
        2,
    )
    .to_csr();
    let diffusion = assemble_bilinear(
        &TransportDiffusionForm::new(&material),
        &space,
        &space,
        &mesh,
        2,
    )
    .to_csr();
    let advection = assemble_bilinear(
        &TransportAdvectionForm::new(&material),
        &space,
        &space,
        &mesh,
        2,
    )
    .to_csr();
    assert_eq!(space.num_global_dofs(&mesh), mesh.nodes.len());

    let volume_operator = linear_combination(&diffusion, 1.0, &advection, 1.0);
    let inlet_advection = vertical_boundary_mass(&mesh, LEFT_M);
    let operator = linear_combination(&volume_operator, 1.0, &inlet_advection, VELOCITY_M_S);
    let left = linear_combination(&mass, 1.0, &operator, 0.5 * TIME_STEP_S);
    let right = linear_combination(&mass, 1.0, &operator, -0.5 * TIME_STEP_S);
    let factorization = SparseLuFactorization::factorize(&left)
        .expect("mass-flux Crank-Nicolson operator must factorize");
    let inlet_weights = vertical_boundary_weights(&mesh, LEFT_M);
    let outlet_weights = vertical_boundary_weights(&mesh, RIGHT_M);

    let mut concentration = vec![0.0; mesh.nodes.len()];
    let mut outlet_history = Vec::new();
    let mut cumulative_outflow = 0.0;
    let steps = (FINAL_TIME_S / TIME_STEP_S).round() as usize;
    for step in 0..steps {
        let time = step as f64 * TIME_STEP_S;
        let midpoint = time + 0.5 * TIME_STEP_S;
        let old_outlet = VELOCITY_M_S * dot(&outlet_weights, &concentration);
        let mut rhs = right.matvec(&concentration);
        if midpoint <= FLUX_END_TIME_S {
            for (entry, weight) in rhs.iter_mut().zip(&inlet_weights) {
                *entry += TIME_STEP_S * FLUX * weight;
            }
        }
        let result = factorization
            .solve_refined(&left, &rhs, 1.0e-12, 2)
            .expect("reused sparse solve must succeed");
        assert!(result.converged, "sparse refinement failed: {result:?}");
        concentration = result.x;
        let new_outlet = VELOCITY_M_S * dot(&outlet_weights, &concentration);
        cumulative_outflow += 0.5 * TIME_STEP_S * (old_outlet + new_outlet);

        let new_time = (step + 1) as f64 * TIME_STEP_S;
        if SAMPLE_TIME_S
            .iter()
            .any(|sample| (new_time - sample).abs() < 0.5 * TIME_STEP_S)
        {
            outlet_history.push(dot(&outlet_weights, &concentration) / STRIP_WIDTH_M);
        }
    }

    let stored_mass: f64 = mass.matvec(&concentration).iter().sum();
    let injected_mass = FLUX * STRIP_WIDTH_M * FLUX_END_TIME_S;
    let relative_balance_error =
        (stored_mass + cumulative_outflow - injected_mass).abs() / injected_mass;
    let minimum_concentration = concentration.iter().copied().fold(f64::INFINITY, f64::min);
    SoniumResult {
        final_profile: profile_samples(&mesh, &concentration),
        outlet_history,
        relative_balance_error,
        minimum_concentration,
    }
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    diffusivity_m2_s: f64,
    velocity_m_s: f64,
    flux: f64,
    flux_end_time_s: f64,
    final_time_s: f64,
    sample_x_m: Vec<f64>,
    sample_time_s: Vec<f64>,
    final_profile: Vec<f64>,
    outlet_history: Vec<f64>,
}

fn wolfram_reference() -> Option<WolframReference> {
    let Some(executable) = env::var_os("WOLFRAMSCRIPT") else {
        return Some(
            serde_json::from_str(include_str!(
                "../wolfram/goldens/mass_transport_flux_outflow.json"
            ))
            .expect("parse checked-in Wolfram mass-flux reference"),
        );
    };
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/mass_transport_flux_outflow.wls");
    let output = match Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Wolfram Engine not available; using conservation and refinement oracles");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram mass-flux reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram mass-flux output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

#[test]
fn sonium_reproduces_wolfram_transient_mass_flux_and_outflow() {
    let coarse = sonium_solution(100);
    let fine = sonium_solution(200);
    eprintln!("coarse mass-flux result: {coarse:?}");
    eprintln!("fine mass-flux result: {fine:?}");

    assert!(
        fine.relative_balance_error < 1.0e-10,
        "relative balance error {:.3e}",
        fine.relative_balance_error
    );
    assert!(
        fine.minimum_concentration > -1.0e-8,
        "minimum concentration {:.9}",
        fine.minimum_concentration
    );
    assert_eq!(
        fine.outlet_history.len(),
        SAMPLE_TIME_S.len(),
        "outlet history sample count"
    );
    for (sample, (&got, &coarse_value)) in fine
        .final_profile
        .iter()
        .zip(&coarse.final_profile)
        .enumerate()
    {
        assert!(
            (got - coarse_value).abs() < 0.15,
            "final profile sample {sample}: fine={got:.9}, coarse={coarse_value:.9}"
        );
    }
    for (sample, (&got, &coarse_value)) in fine
        .outlet_history
        .iter()
        .zip(&coarse.outlet_history)
        .enumerate()
    {
        assert!(
            (got - coarse_value).abs() < 0.15,
            "outlet sample {sample}: fine={got:.9}, coarse={coarse_value:.9}"
        );
    }

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.diffusivity_m2_s, DIFFUSIVITY_M2_S);
        assert_eq!(reference.velocity_m_s, VELOCITY_M_S);
        assert_eq!(reference.flux, FLUX);
        assert_eq!(reference.flux_end_time_s, FLUX_END_TIME_S);
        assert_eq!(reference.final_time_s, FINAL_TIME_S);
        assert_eq!(reference.sample_x_m, SAMPLE_X_M);
        assert_eq!(reference.sample_time_s, SAMPLE_TIME_S);
        for (sample, (&got, &want)) in fine
            .final_profile
            .iter()
            .zip(&reference.final_profile)
            .enumerate()
        {
            assert!(
                (got - want).abs() < 0.5,
                "final sample {sample}: Sonium={got:.9}, Wolfram={want:.9}"
            );
        }
        for (sample, (&got, &want)) in fine
            .outlet_history
            .iter()
            .zip(&reference.outlet_history)
            .enumerate()
        {
            assert!(
                (got - want).abs() < 0.5,
                "outlet sample {sample}: Sonium={got:.9}, Wolfram={want:.9}"
            );
        }
    }
}
