//! Literal transient Catalytic Converter model from Wolfram's PDEModels
//! collection. Wolfram publishes an outlet-average concentration of 1357.2 at
//! 150 s, or a reduction of 142.797 from the 1500 inlet concentration.
//!
//! Sonium supplies the diffusion/advection volume forms but no nonlinear mass-
//! flux boundary object. This test integrates the published catalytic wall law
//! locally while retaining Sonium's assembled transport operators.

use serde::Deserialize;
use sonium_fem::{
    assembly::{assemble_bilinear, assemble_bilinear_per_element},
    basis::PolynomialDegree,
    mesh::{Mesh, rectangular_mesh_triangles},
    physics::fluid::transport::{
        TransportAdvectionForm, TransportCapacityForm, TransportDiffusionForm, TransportMaterial,
    },
    spaces::H1Space,
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{CsrMatrix, SparseLuFactorization, TripletMatrix, traits::Preconditioner};
use std::{env, path::PathBuf, process::Command};

const WIDTH_M: f64 = 0.2;
const LENGTH_M: f64 = 0.5;
const DIFFUSIVITY_M2_S: f64 = 1.6e-5;
const AVERAGE_VELOCITY_M_S: f64 = 5.0e-3;
const INLET_CONCENTRATION: f64 = 1500.0;
const SURFACE_CAPACITY: f64 = 1000.0;
const CATALYTIC_RATE: f64 = 1.0e-6;
const FINAL_TIME_S: f64 = 150.0;
const PUBLISHED_OUTLET_CONCENTRATION: f64 = 1357.2;
const PUBLISHED_REDUCTION: f64 = 142.797;
const SAMPLE_POINTS: [[f64; 2]; 4] = [[0.1, 0.25], [0.2, 0.25], [0.1, 0.5], [0.1, 0.0]];

fn axial_velocity(x: f64) -> f64 {
    2.0 * AVERAGE_VELOCITY_M_S * (1.0 - ((x - WIDTH_M / 2.0) / (WIDTH_M / 2.0)).powi(2))
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

fn right_wall_nodes(mesh: &Mesh) -> Vec<usize> {
    let mut nodes: Vec<usize> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| ((node.x - WIDTH_M).abs() < 1.0e-12).then_some(index))
        .collect();
    nodes.sort_by(|&a, &b| mesh.nodes[a].y.total_cmp(&mesh.nodes[b].y));
    nodes
}

fn catalytic_flux(concentration: f64) -> f64 {
    -CATALYTIC_RATE * concentration * (SURFACE_CAPACITY - (INLET_CONCENTRATION - concentration))
}

/// Consistent P1 boundary load using three-point Gauss integration. The
/// integrand is cubic because the catalytic flux is quadratic in concentration.
fn add_catalytic_wall_load(mesh: &Mesh, concentration: &[f64], rhs: &mut [f64], dt: f64) {
    let gauss = [
        (-0.7745966692414834, 5.0 / 18.0),
        (0.0, 4.0 / 9.0),
        (0.7745966692414834, 5.0 / 18.0),
    ];
    for pair in right_wall_nodes(mesh).windows(2) {
        let length = mesh.nodes[pair[1]].y - mesh.nodes[pair[0]].y;
        for (xi, weight) in gauss {
            let s = 0.5 * (1.0 + xi);
            let shape = [1.0 - s, s];
            let value = shape[0] * concentration[pair[0]] + shape[1] * concentration[pair[1]];
            let flux = catalytic_flux(value);
            rhs[pair[0]] += dt * length * weight * shape[0] * flux;
            rhs[pair[1]] += dt * length * weight * shape[1] * flux;
        }
    }
}

fn constrain_inlet_matrix(
    matrix: &CsrMatrix<f64>,
    inlet_dofs: &[usize],
) -> (CsrMatrix<f64>, Vec<f64>) {
    let mut constrained = vec![false; matrix.num_rows];
    for &dof in inlet_dofs {
        constrained[dof] = true;
    }
    let mut result = matrix.clone();
    let mut correction = vec![0.0; matrix.num_rows];
    for row in 0..matrix.num_rows {
        if constrained[row] {
            continue;
        }
        for entry in result.row_ptrs[row]..result.row_ptrs[row + 1] {
            if constrained[result.col_indices[entry]] {
                correction[row] -= result.values[entry] * INLET_CONCENTRATION;
                result.values[entry] = 0.0;
            }
        }
    }
    for &dof in inlet_dofs {
        let mut found_diagonal = false;
        for entry in result.row_ptrs[dof]..result.row_ptrs[dof + 1] {
            if result.col_indices[entry] == dof {
                result.values[entry] = 1.0;
                found_diagonal = true;
            } else {
                result.values[entry] = 0.0;
            }
        }
        assert!(found_diagonal);
    }
    (result, correction)
}

fn outlet_average(mesh: &Mesh, concentration: &[f64]) -> f64 {
    let mut nodes: Vec<usize> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| ((node.y - LENGTH_M).abs() < 1.0e-12).then_some(index))
        .collect();
    nodes.sort_by(|&a, &b| mesh.nodes[a].x.total_cmp(&mesh.nodes[b].x));
    let integral: f64 = nodes
        .windows(2)
        .map(|pair| {
            let length = mesh.nodes[pair[1]].x - mesh.nodes[pair[0]].x;
            0.5 * length * (concentration[pair[0]] + concentration[pair[1]])
        })
        .sum();
    integral / WIDTH_M
}

fn sample_field(mesh: &Mesh, concentration: &[f64]) -> Vec<f64> {
    SAMPLE_POINTS
        .iter()
        .map(|point| {
            let index = mesh
                .nodes
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (a.x - point[0]).powi(2) + (a.y - point[1]).powi(2);
                    let db = (b.x - point[0]).powi(2) + (b.y - point[1]).powi(2);
                    da.total_cmp(&db)
                })
                .map(|(index, _)| index)
                .expect("converter mesh is nonempty");
            concentration[index]
        })
        .collect()
}

#[derive(Debug)]
struct SoniumResult {
    outlet_concentration: f64,
    reduction: f64,
    samples: Vec<f64>,
    minimum_concentration: f64,
    maximum_concentration: f64,
    relative_linear_residual: f64,
}

fn sonium_solution(nx: usize, ny: usize, dt: f64) -> SoniumResult {
    let mesh = rectangular_mesh_triangles(0.0, WIDTH_M, 0.0, LENGTH_M, nx, ny);
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
    let advection_forms: Vec<_> = mesh
        .elements
        .iter()
        .map(|element| {
            let x = element
                .nodes
                .iter()
                .map(|&node| mesh.nodes[node].x)
                .sum::<f64>()
                / element.nodes.len() as f64;
            TransportAdvectionForm {
                velocity: [0.0, axial_velocity(x), 0.0],
            }
        })
        .collect();
    let advection =
        assemble_bilinear_per_element(&advection_forms, &space, &space, &mesh, 2).to_csr();
    let operator = linear_combination(&diffusion, 1.0, &advection, 1.0);
    let implicit = linear_combination(&mass, 1.0, &operator, dt);
    let inlet_dofs: Vec<usize> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| (node.y.abs() < 1.0e-12).then_some(index))
        .collect();
    let (constrained, inlet_correction) = constrain_inlet_matrix(&implicit, &inlet_dofs);
    let factorization = SparseLuFactorization::factorize(&constrained)
        .expect("catalytic-converter implicit operator must factorize");

    let mut concentration = vec![INLET_CONCENTRATION; mesh.nodes.len()];
    let mut last_rhs = vec![0.0; mesh.nodes.len()];
    let steps = (FINAL_TIME_S / dt).round() as usize;
    assert!((steps as f64 * dt - FINAL_TIME_S).abs() < 1.0e-12);
    for _ in 0..steps {
        let mut rhs = mass.matvec(&concentration);
        add_catalytic_wall_load(&mesh, &concentration, &mut rhs, dt);
        for (entry, correction) in rhs.iter_mut().zip(&inlet_correction) {
            *entry += correction;
        }
        for &dof in &inlet_dofs {
            rhs[dof] = INLET_CONCENTRATION;
        }
        concentration = factorization.apply(&rhs);
        last_rhs = rhs;
    }

    let outlet_concentration = outlet_average(&mesh, &concentration);
    let residual = constrained
        .matvec(&concentration)
        .iter()
        .zip(&last_rhs)
        .map(|(lhs, rhs)| (lhs - rhs).powi(2))
        .sum::<f64>()
        .sqrt();
    let rhs_norm = last_rhs
        .iter()
        .map(|value| value.powi(2))
        .sum::<f64>()
        .sqrt();
    SoniumResult {
        outlet_concentration,
        reduction: INLET_CONCENTRATION - outlet_concentration,
        samples: sample_field(&mesh, &concentration),
        minimum_concentration: concentration.iter().copied().fold(f64::INFINITY, f64::min),
        maximum_concentration: concentration
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        relative_linear_residual: residual / rhs_norm,
    }
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    final_time_s: f64,
    sample_points: Vec<Vec<f64>>,
    samples: Vec<f64>,
    outlet_concentration: f64,
    reduction: f64,
}

fn wolfram_reference() -> Option<WolframReference> {
    let Some(executable) = env::var_os("WOLFRAMSCRIPT") else {
        return Some(
            serde_json::from_str(include_str!("../wolfram/goldens/catalytic_converter.json"))
                .expect("parse checked-in Wolfram catalytic-converter reference"),
        );
    };
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/catalytic_converter.wls");
    let output = match Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!(
                "Wolfram Engine not available; using published outlet and refinement oracles"
            );
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram catalytic-converter reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram catalytic-converter output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

#[test]
fn sonium_reproduces_wolfram_catalytic_converter() {
    let coarse = sonium_solution(20, 50, 0.4);
    let fine = sonium_solution(30, 75, 0.2);
    eprintln!("coarse catalytic-converter result: {coarse:?}");
    eprintln!("fine catalytic-converter result: {fine:?}");

    assert!((fine.outlet_concentration - PUBLISHED_OUTLET_CONCENTRATION).abs() < 2.0);
    assert!((fine.reduction - PUBLISHED_REDUCTION).abs() < 2.0);
    assert!((fine.outlet_concentration - coarse.outlet_concentration).abs() < 1.0);
    assert!(fine.minimum_concentration >= 0.0);
    assert!(fine.maximum_concentration <= INLET_CONCENTRATION + 0.25);
    assert!(fine.relative_linear_residual < 1.0e-10);
    assert!((fine.samples[3] - INLET_CONCENTRATION).abs() < 1.0e-10);
    for (sample, (&got, &coarse_value)) in fine.samples.iter().zip(&coarse.samples).enumerate() {
        assert!(
            (got - coarse_value).abs() / got.abs().max(1.0) < 0.01,
            "sample {sample}: fine={got:.9}, coarse={coarse_value:.9}"
        );
    }

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.final_time_s, FINAL_TIME_S);
        assert_eq!(reference.sample_points, SAMPLE_POINTS);
        assert!((fine.outlet_concentration - reference.outlet_concentration).abs() < 2.0);
        assert!((fine.reduction - reference.reduction).abs() < 2.0);
        for (sample, (&got, &want)) in fine.samples.iter().zip(&reference.samples).enumerate() {
            assert!(
                (got - want).abs() / want.abs().max(1.0) < 0.02,
                "sample {sample}: Sonium={got:.9}, Wolfram={want:.9}"
            );
        }
    }
}
