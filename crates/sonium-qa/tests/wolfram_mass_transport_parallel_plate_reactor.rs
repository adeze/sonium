//! Literal steady parallel-plate reactor from Wolfram's mass-transport
//! monograph. The flow is parabolic across the 0.2 m plate spacing, species
//! enters at concentration 1000, and the right plate imposes mass flux -0.5.
//!
//! Sonium's transport material stores one constant velocity. This test uses
//! the public per-element assembly route and evaluates Wolfram's velocity at
//! each triangle centroid, making the approximation explicit and convergent.

use serde::Deserialize;
use sonium_fem::{
    assembly::{assemble_bilinear, assemble_bilinear_per_element},
    basis::PolynomialDegree,
    mesh::{Mesh, rectangular_mesh_triangles},
    physics::fluid::transport::{
        TransportAdvectionForm, TransportDiffusionForm, TransportMaterial,
    },
    spaces::H1Space,
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{CsrMatrix, SparseLuFactorization, TripletMatrix};
use std::{env, path::PathBuf, process::Command};

const WIDTH_M: f64 = 0.2;
const LENGTH_M: f64 = 0.5;
const DIFFUSIVITY_M2_S: f64 = 1.0e-4;
const AVERAGE_VELOCITY_M_S: f64 = 5.0e-3;
const INLET_CONCENTRATION: f64 = 1000.0;
const ACTIVE_WALL_FLUX: f64 = -0.5;
const SAMPLE_POINTS: [[f64; 2]; 5] = [
    [0.0, 0.25],
    [0.1, 0.25],
    [0.2, 0.25],
    [0.1, 0.5],
    [0.1, 0.0],
];

fn axial_velocity(x: f64) -> f64 {
    2.0 * AVERAGE_VELOCITY_M_S * (1.0 - ((x - 0.1) / 0.1).powi(2))
}

fn add_right_wall_flux(mesh: &Mesh, rhs: &mut [f64]) {
    let mut nodes: Vec<usize> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| ((node.x - WIDTH_M).abs() < 1.0e-12).then_some(index))
        .collect();
    nodes.sort_by(|&a, &b| mesh.nodes[a].y.total_cmp(&mesh.nodes[b].y));
    for pair in nodes.windows(2) {
        let length = mesh.nodes[pair[1]].y - mesh.nodes[pair[0]].y;
        rhs[pair[0]] += 0.5 * length * ACTIVE_WALL_FLUX;
        rhs[pair[1]] += 0.5 * length * ACTIVE_WALL_FLUX;
    }
}

fn apply_inlet_concentration(matrix: &mut CsrMatrix<f64>, rhs: &mut [f64], inlet_dofs: &[usize]) {
    let mut constrained = vec![false; matrix.num_rows];
    for &dof in inlet_dofs {
        constrained[dof] = true;
    }
    for row in 0..matrix.num_rows {
        if constrained[row] {
            continue;
        }
        for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            if constrained[matrix.col_indices[entry]] {
                rhs[row] -= matrix.values[entry] * INLET_CONCENTRATION;
                matrix.values[entry] = 0.0;
            }
        }
    }
    for &dof in inlet_dofs {
        let mut found_diagonal = false;
        for entry in matrix.row_ptrs[dof]..matrix.row_ptrs[dof + 1] {
            if matrix.col_indices[entry] == dof {
                matrix.values[entry] = 1.0;
                found_diagonal = true;
            } else {
                matrix.values[entry] = 0.0;
            }
        }
        assert!(found_diagonal);
        rhs[dof] = INLET_CONCENTRATION;
    }
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
                .expect("reactor mesh is nonempty");
            concentration[index]
        })
        .collect()
}

fn top_advective_outflow(mesh: &Mesh, concentration: &[f64]) -> f64 {
    let mut nodes: Vec<usize> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| ((node.y - LENGTH_M).abs() < 1.0e-12).then_some(index))
        .collect();
    nodes.sort_by(|&a, &b| mesh.nodes[a].x.total_cmp(&mesh.nodes[b].x));
    nodes
        .windows(2)
        .map(|pair| {
            let length = mesh.nodes[pair[1]].x - mesh.nodes[pair[0]].x;
            0.5 * length
                * (axial_velocity(mesh.nodes[pair[0]].x) * concentration[pair[0]]
                    + axial_velocity(mesh.nodes[pair[1]].x) * concentration[pair[1]])
        })
        .sum()
}

#[derive(Debug)]
struct SoniumResult {
    samples: Vec<f64>,
    area_integral: f64,
    top_advective_outflow: f64,
    inlet_reaction: f64,
    relative_flux_balance_error: f64,
    relative_interior_residual: f64,
}

fn sonium_solution(nx: usize, ny: usize) -> SoniumResult {
    let mesh = rectangular_mesh_triangles(0.0, WIDTH_M, 0.0, LENGTH_M, nx, ny);
    let space = H1Space::new(PolynomialDegree::P1);
    let material = TransportMaterial::isotropic(DIFFUSIVITY_M2_S, [0.0; 3], 0.0);
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
    let mut operator_triplet = TripletMatrix::new(mesh.nodes.len(), mesh.nodes.len());
    for matrix in [&diffusion, &advection] {
        for row in 0..matrix.num_rows {
            for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                operator_triplet.push(row, matrix.col_indices[entry], matrix.values[entry]);
            }
        }
    }
    let original_operator = operator_triplet.to_csr();
    let mut original_rhs = vec![0.0; mesh.nodes.len()];
    add_right_wall_flux(&mesh, &mut original_rhs);
    let inlet_dofs: Vec<usize> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| (node.y.abs() < 1.0e-12).then_some(index))
        .collect();
    let mut constrained_operator = original_operator.clone();
    let mut constrained_rhs = original_rhs.clone();
    apply_inlet_concentration(&mut constrained_operator, &mut constrained_rhs, &inlet_dofs);
    let factorization = SparseLuFactorization::factorize(&constrained_operator)
        .expect("parallel-plate reactor operator must factorize");
    let solve = factorization
        .solve_refined(&constrained_operator, &constrained_rhs, 1.0e-11, 3)
        .expect("parallel-plate reactor solve must succeed");
    assert!(solve.converged, "sparse refinement failed: {solve:?}");
    let concentration = solve.x;

    let residual: Vec<f64> = original_operator
        .matvec(&concentration)
        .iter()
        .zip(&original_rhs)
        .map(|(lhs, rhs)| lhs - rhs)
        .collect();
    let mut constrained_mask = vec![false; mesh.nodes.len()];
    for &dof in &inlet_dofs {
        constrained_mask[dof] = true;
    }
    let interior_norm = residual
        .iter()
        .enumerate()
        .filter(|(index, _)| !constrained_mask[*index])
        .map(|(_, value)| value * value)
        .sum::<f64>()
        .sqrt();
    let rhs_scale = constrained_rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mass = assemble_bilinear(
        &sonium_fem::physics::fluid::transport::TransportCapacityForm::new(
            &TransportMaterial::with_capacity(0.0, [0.0; 3], 0.0, 1.0),
        ),
        &space,
        &space,
        &mesh,
        2,
    )
    .to_csr();
    let inlet_advective_flux = WIDTH_M * (4.0 / 3.0) * AVERAGE_VELOCITY_M_S * INLET_CONCENTRATION;
    let total_inlet =
        inlet_advective_flux + inlet_dofs.iter().map(|&dof| residual[dof]).sum::<f64>();
    let total_outlet = top_advective_outflow(&mesh, &concentration) - ACTIVE_WALL_FLUX * LENGTH_M;
    SoniumResult {
        samples: sample_field(&mesh, &concentration),
        area_integral: mass.matvec(&concentration).iter().sum(),
        top_advective_outflow: top_advective_outflow(&mesh, &concentration),
        inlet_reaction: inlet_dofs.iter().map(|&dof| residual[dof]).sum(),
        relative_flux_balance_error: (total_inlet - total_outlet).abs() / total_inlet.abs(),
        relative_interior_residual: interior_norm / rhs_scale,
    }
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    diffusivity_m2_s: f64,
    average_velocity_m_s: f64,
    inlet_concentration: f64,
    active_wall_flux: f64,
    sample_points: Vec<Vec<f64>>,
    samples: Vec<f64>,
    area_integral: f64,
}

fn wolfram_reference() -> Option<WolframReference> {
    let Some(executable) = env::var_os("WOLFRAMSCRIPT") else {
        return Some(
            serde_json::from_str(include_str!(
                "../wolfram/goldens/mass_transport_parallel_plate_reactor.json"
            ))
            .expect("parse checked-in Wolfram parallel-plate reference"),
        );
    };
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("wolfram/mass_transport_parallel_plate_reactor.wls");
    let output = match Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Wolfram Engine not available; using refinement and residual oracles");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram parallel-plate reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram parallel-plate output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

#[test]
fn sonium_reproduces_wolfram_parallel_plate_reactor() {
    let coarse = sonium_solution(40, 100);
    let fine = sonium_solution(60, 150);
    eprintln!("coarse parallel-plate result: {coarse:?}");
    eprintln!("fine parallel-plate result: {fine:?}");

    assert!(fine.relative_interior_residual < 1.0e-9);
    assert!(fine.relative_flux_balance_error < 5.0e-4);
    assert!(fine.top_advective_outflow > 0.0);
    assert!(fine.inlet_reaction > 0.0);
    assert!((fine.samples[4] - INLET_CONCENTRATION).abs() < 1.0e-10);
    assert!(
        fine.samples
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
    );
    assert!((fine.area_integral - coarse.area_integral).abs() / fine.area_integral < 0.005);
    for (sample, (&got, &coarse_value)) in fine.samples.iter().zip(&coarse.samples).enumerate() {
        assert!(
            (got - coarse_value).abs() / got.abs().max(1.0) < 0.02,
            "sample {sample}: fine={got:.9}, coarse={coarse_value:.9}"
        );
    }

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.diffusivity_m2_s, DIFFUSIVITY_M2_S);
        assert_eq!(reference.average_velocity_m_s, AVERAGE_VELOCITY_M_S);
        assert_eq!(reference.inlet_concentration, INLET_CONCENTRATION);
        assert_eq!(reference.active_wall_flux, ACTIVE_WALL_FLUX);
        assert_eq!(reference.sample_points, SAMPLE_POINTS);
        assert!(
            (fine.area_integral - reference.area_integral).abs() / reference.area_integral < 0.01
        );
        for (sample, (&got, &want)) in fine.samples.iter().zip(&reference.samples).enumerate() {
            assert!(
                (got - want).abs() / want.abs().max(1.0) < 0.03,
                "sample {sample}: Sonium={got:.9}, Wolfram={want:.9}"
            );
        }
    }
}
