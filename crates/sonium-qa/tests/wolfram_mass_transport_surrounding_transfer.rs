//! Literal CO2 surrounding-transfer example from Wolfram's mass-transport
//! monograph. Wolfram publishes the initial/final line integrals
//! `{354.346, 81.1071}` at t={0,10080} s.
//!
//! Sonium has the required diffusion and capacity forms, but no high-level
//! `MassTransferValue` boundary adapter. This test assembles the equivalent
//! Robin boundary mass matrix locally on a thin 2-D extrusion of the 1-D line.

use serde::Deserialize;
use sonium_fem::{
    assembly::assemble_bilinear,
    basis::PolynomialDegree,
    mesh::{Mesh, rectangular_mesh_triangles},
    physics::fluid::transport::{TransportCapacityForm, TransportDiffusionForm, TransportMaterial},
    spaces::{FunctionSpace, H1Space},
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{CsrMatrix, SparseLuFactorization, TripletMatrix};
use std::{env, path::PathBuf, process::Command};

const LEFT_M: f64 = 0.0;
const RIGHT_M: f64 = 1.0;
const STRIP_WIDTH_M: f64 = 0.01;
const DIFFUSIVITY_M2_S: f64 = 1.64e-5;
const TRANSFER_COEFFICIENT_M_S: f64 = 8.2e-3;
const AMBIENT_CONCENTRATION: f64 = 0.0;
const FINAL_TIME_S: f64 = 10080.0;
const TIME_STEP_S: f64 = 10.0;
const PUBLISHED_INITIAL_INTEGRAL: f64 = 354.346;
const PUBLISHED_FINAL_INTEGRAL: f64 = 81.1071;
const SAMPLE_X_M: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

fn initial_concentration(x: f64) -> f64 {
    1000.0 * (-((x - 0.5) / 0.2).powi(2)).exp()
}

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

fn vertical_boundary_nodes(mesh: &Mesh, x: f64) -> Vec<usize> {
    let mut nodes: Vec<usize> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| ((node.x - x).abs() < 1.0e-12).then_some(index))
        .collect();
    nodes.sort_by(|&a, &b| mesh.nodes[a].y.total_cmp(&mesh.nodes[b].y));
    nodes
}

fn add_vertical_boundary_mass(mesh: &Mesh, x: f64, matrix: &mut TripletMatrix<f64>) {
    for pair in vertical_boundary_nodes(mesh, x).windows(2) {
        let length = (mesh.nodes[pair[1]].y - mesh.nodes[pair[0]].y).abs();
        matrix.push(pair[0], pair[0], length / 3.0);
        matrix.push(pair[0], pair[1], length / 6.0);
        matrix.push(pair[1], pair[0], length / 6.0);
        matrix.push(pair[1], pair[1], length / 3.0);
    }
}

fn transfer_boundary_mass(mesh: &Mesh) -> CsrMatrix<f64> {
    let mut matrix = TripletMatrix::new(mesh.nodes.len(), mesh.nodes.len());
    add_vertical_boundary_mass(mesh, LEFT_M, &mut matrix);
    add_vertical_boundary_mass(mesh, RIGHT_M, &mut matrix);
    matrix.to_csr()
}

fn boundary_integral(mesh: &Mesh, x: f64, concentration: &[f64]) -> f64 {
    vertical_boundary_nodes(mesh, x)
        .windows(2)
        .map(|pair| {
            let length = (mesh.nodes[pair[1]].y - mesh.nodes[pair[0]].y).abs();
            0.5 * length * (concentration[pair[0]] + concentration[pair[1]])
        })
        .sum()
}

fn profile_samples(mesh: &Mesh, concentration: &[f64]) -> Vec<f64> {
    SAMPLE_X_M
        .iter()
        .map(|&sample_x| {
            let values: Vec<f64> = mesh
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| (node.x - sample_x).abs() < 1.0e-12)
                .map(|(index, _)| concentration[index])
                .collect();
            assert!(
                !values.is_empty(),
                "sample x={sample_x} must be a mesh line"
            );
            values.iter().sum::<f64>() / values.len() as f64
        })
        .collect()
}

#[derive(Debug)]
struct SoniumResult {
    initial_integral: f64,
    final_integral: f64,
    final_profile: Vec<f64>,
    relative_balance_error: f64,
    symmetry_error: f64,
}

fn sonium_solution(nx: usize) -> SoniumResult {
    let mesh = rectangular_mesh_triangles(LEFT_M, RIGHT_M, 0.0, STRIP_WIDTH_M, nx, 1);
    let space = H1Space::new(PolynomialDegree::P1);
    let material = TransportMaterial::with_capacity(DIFFUSIVITY_M2_S, [0.0; 3], 0.0, 1.0);
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
    assert_eq!(space.num_global_dofs(&mesh), mesh.nodes.len());

    let transfer_mass = transfer_boundary_mass(&mesh);
    let operator = linear_combination(&diffusion, 1.0, &transfer_mass, TRANSFER_COEFFICIENT_M_S);
    let left = linear_combination(&mass, 1.0, &operator, 0.5 * TIME_STEP_S);
    let right = linear_combination(&mass, 1.0, &operator, -0.5 * TIME_STEP_S);
    let factorization = SparseLuFactorization::factorize(&left)
        .expect("surrounding-transfer Crank-Nicolson operator must factorize");

    let mut concentration: Vec<f64> = mesh
        .nodes
        .iter()
        .map(|node| initial_concentration(node.x))
        .collect();
    let initial_mass: f64 = mass.matvec(&concentration).iter().sum();
    let mut cumulative_outflow = 0.0;
    let steps = (FINAL_TIME_S / TIME_STEP_S).round() as usize;
    for _ in 0..steps {
        let old_outflow = TRANSFER_COEFFICIENT_M_S
            * (boundary_integral(&mesh, LEFT_M, &concentration)
                + boundary_integral(&mesh, RIGHT_M, &concentration));
        let rhs = right.matvec(&concentration);
        let result = factorization
            .solve_refined(&left, &rhs, 1.0e-12, 2)
            .expect("reused sparse solve must succeed");
        assert!(result.converged, "sparse refinement failed: {result:?}");
        concentration = result.x;
        let new_outflow = TRANSFER_COEFFICIENT_M_S
            * (boundary_integral(&mesh, LEFT_M, &concentration)
                + boundary_integral(&mesh, RIGHT_M, &concentration));
        cumulative_outflow += 0.5 * TIME_STEP_S * (old_outflow + new_outflow);
    }

    let final_mass: f64 = mass.matvec(&concentration).iter().sum();
    let final_profile = profile_samples(&mesh, &concentration);
    let symmetry_error = final_profile
        .iter()
        .zip(final_profile.iter().rev())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    SoniumResult {
        initial_integral: initial_mass / STRIP_WIDTH_M,
        final_integral: final_mass / STRIP_WIDTH_M,
        final_profile,
        relative_balance_error: (final_mass + cumulative_outflow - initial_mass).abs()
            / initial_mass,
        symmetry_error,
    }
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    diffusivity_m2_s: f64,
    transfer_coefficient_m_s: f64,
    ambient_concentration: f64,
    final_time_s: f64,
    sample_x_m: Vec<f64>,
    initial_integral: f64,
    final_integral: f64,
    final_profile: Vec<f64>,
}

fn wolfram_reference() -> Option<WolframReference> {
    let Some(executable) = env::var_os("WOLFRAMSCRIPT") else {
        return Some(
            serde_json::from_str(include_str!(
                "../wolfram/goldens/mass_transport_surrounding_transfer.json"
            ))
            .expect("parse checked-in Wolfram surrounding-transfer reference"),
        );
    };
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("wolfram/mass_transport_surrounding_transfer.wls");
    let output = match Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Wolfram Engine not available; using published integral and balance oracles");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram surrounding-transfer reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram surrounding-transfer output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

#[test]
fn sonium_reproduces_wolfram_surrounding_mass_transfer() {
    let coarse = sonium_solution(400);
    let fine = sonium_solution(800);
    eprintln!("coarse surrounding-transfer result: {coarse:?}");
    eprintln!("fine surrounding-transfer result: {fine:?}");

    assert!((fine.initial_integral - PUBLISHED_INITIAL_INTEGRAL).abs() < 0.01);
    assert!((fine.final_integral - PUBLISHED_FINAL_INTEGRAL).abs() < 0.2);
    assert!((fine.final_integral - coarse.final_integral).abs() < 0.1);
    assert!(fine.relative_balance_error < 1.0e-10);
    assert!(fine.symmetry_error < 1.0e-9);
    for (sample, (&got, &coarse_value)) in fine
        .final_profile
        .iter()
        .zip(&coarse.final_profile)
        .enumerate()
    {
        assert!(
            (got - coarse_value).abs() < 0.2,
            "profile sample {sample}: fine={got:.9}, coarse={coarse_value:.9}"
        );
    }

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.diffusivity_m2_s, DIFFUSIVITY_M2_S);
        assert_eq!(reference.transfer_coefficient_m_s, TRANSFER_COEFFICIENT_M_S);
        assert_eq!(reference.ambient_concentration, AMBIENT_CONCENTRATION);
        assert_eq!(reference.final_time_s, FINAL_TIME_S);
        assert_eq!(reference.sample_x_m, SAMPLE_X_M);
        assert!((reference.initial_integral - PUBLISHED_INITIAL_INTEGRAL).abs() < 0.01);
        assert!((reference.final_integral - PUBLISHED_FINAL_INTEGRAL).abs() < 0.01);
        for (sample, (&got, &want)) in fine
            .final_profile
            .iter()
            .zip(&reference.final_profile)
            .enumerate()
        {
            assert!(
                (got - want).abs() < 0.3,
                "profile sample {sample}: Sonium={got:.9}, Wolfram={want:.9}"
            );
        }
    }
}
