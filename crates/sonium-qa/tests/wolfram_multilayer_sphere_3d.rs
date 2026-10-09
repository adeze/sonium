//! Literal three-dimensional Sonium port of Wolfram's multilayer-sphere heat
//! transfer example, including three marked materials and its one-sided
//! angular outer heat flux.

use serde::Deserialize;
use sonium_fem::{
    assembly::{TripletMatrix, assemble_bilinear_per_element},
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_neumann_real},
    mesh::{Mesh, layered_spherical_shell_mesh_tetrahedra},
    physics::thermal::heat_conduction::{ConductionForm, HeatCapacityForm, ThermalMaterial},
    spaces::{FunctionSpace, H1Space},
};
use sonium_solvers::{CsrMatrix, SparseLuFactorization, traits::Preconditioner};
use std::{collections::BTreeSet, f64::consts::PI};

const RADII_M: [f64; 4] = [1.0, 2.0, 4.0, 6.0];
const CONDUCTIVITY: [f64; 3] = [1.0, 2.0, 4.0];
const VOLUMETRIC_HEAT_CAPACITY: [f64; 3] = [1.0, 0.5, 4.0 / 9.0];
const RADIAL_SUBDIVISIONS: [usize; 3] = [2, 4, 4];
const SAMPLE_RADII_M: [f64; 3] = [1.5, 3.0, 5.0];
const FINAL_TIME_S: f64 = 50.0;
const TIME_STEP_S: f64 = 0.25;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WolframReference {
    spherical_average_flux: f64,
    spherical_average_samples: [[f64; 3]; 6],
}

#[derive(Debug)]
struct LiteralResult {
    spherical_average: [f64; 3],
    angular_spread: f64,
    relative_flux_quadrature_error: f64,
    relative_energy_balance: f64,
    relative_free_residual: f64,
}

fn add_scaled(
    first: &CsrMatrix<f64>,
    first_scale: f64,
    second: &CsrMatrix<f64>,
    second_scale: f64,
) -> CsrMatrix<f64> {
    let mut triplets = TripletMatrix::new(first.num_rows, first.num_cols);
    for row in 0..first.num_rows {
        for entry in first.row_ptrs[row]..first.row_ptrs[row + 1] {
            triplets.push(
                row,
                first.col_indices[entry],
                first_scale * first.values[entry],
            );
        }
        for entry in second.row_ptrs[row]..second.row_ptrs[row + 1] {
            triplets.push(
                row,
                second.col_indices[entry],
                second_scale * second.values[entry],
            );
        }
    }
    triplets.to_csr()
}

fn impose_zero_rows(matrix: &mut CsrMatrix<f64>, dofs: &[usize]) {
    for &dof in dofs {
        for entry in matrix.row_ptrs[dof]..matrix.row_ptrs[dof + 1] {
            matrix.values[entry] = if matrix.col_indices[entry] == dof {
                1.0
            } else {
                0.0
            };
        }
    }
}

fn angular_flux(x: f64, y: f64, z: f64) -> f64 {
    let radius = x.hypot(y).hypot(z);
    let theta = (z / radius).clamp(-1.0, 1.0).acos();
    let phi = y.atan2(x);
    if !(0.0..=PI).contains(&phi) {
        return 0.0;
    }
    theta.powi(2) * (PI - theta).powi(2) * phi.powi(2) * (PI - phi).powi(2)
}

fn exact_average_flux() -> f64 {
    (12.0 - PI * PI) * PI.powi(4) / 30.0
}

fn surface_triangles(mesh: &Mesh, radius: f64) -> Vec<[usize; 3]> {
    let mut triangles = BTreeSet::new();
    for element in &mesh.elements {
        let nodes = &element.nodes;
        for mut face in [
            [nodes[0], nodes[1], nodes[2]],
            [nodes[0], nodes[1], nodes[3]],
            [nodes[0], nodes[2], nodes[3]],
            [nodes[1], nodes[2], nodes[3]],
        ] {
            if face.iter().all(|&node| {
                (mesh.nodes[node]
                    .x
                    .hypot(mesh.nodes[node].y)
                    .hypot(mesh.nodes[node].z)
                    - radius)
                    .abs()
                    < 1.0e-10
            }) {
                face.sort_unstable();
                triangles.insert(face);
            }
        }
    }
    triangles.into_iter().collect()
}

fn spherical_surface_average(mesh: &Mesh, values: &[f64], radius: f64) -> f64 {
    let mut integral = 0.0;
    let mut area = 0.0;
    for triangle in surface_triangles(mesh, radius) {
        let first = mesh.nodes[triangle[0]];
        let second = mesh.nodes[triangle[1]];
        let third = mesh.nodes[triangle[2]];
        let first_edge = [second.x - first.x, second.y - first.y, second.z - first.z];
        let second_edge = [third.x - first.x, third.y - first.y, third.z - first.z];
        let cross = [
            first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
            first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
            first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0],
        ];
        let triangle_area = 0.5 * cross[0].hypot(cross[1]).hypot(cross[2]);
        integral +=
            triangle_area * (values[triangle[0]] + values[triangle[1]] + values[triangle[2]]) / 3.0;
        area += triangle_area;
    }
    assert!(area > 0.0, "missing spherical interface at radius {radius}");
    integral / area
}

fn solve_literal(subdivisions: usize) -> LiteralResult {
    let mesh = layered_spherical_shell_mesh_tetrahedra(
        0.0,
        0.0,
        0.0,
        &RADII_M,
        subdivisions,
        &RADIAL_SUBDIVISIONS,
    );
    let space = H1Space::new(PolynomialDegree::P1);
    let materials: Vec<_> = mesh
        .elements
        .iter()
        .map(|element| {
            let layer = usize::try_from(element.marker - 1).expect("positive layer marker");
            ThermalMaterial::new(CONDUCTIVITY[layer], 1.0, VOLUMETRIC_HEAT_CAPACITY[layer])
        })
        .collect();
    let conduction_forms: Vec<_> = materials.iter().map(ConductionForm::new).collect();
    let capacity_forms: Vec<_> = materials.iter().map(HeatCapacityForm::new).collect();
    let conduction =
        assemble_bilinear_per_element(&conduction_forms, &space, &space, &mesh, 2).to_csr();
    let capacity =
        assemble_bilinear_per_element(&capacity_forms, &space, &space, &mesh, 2).to_csr();

    let mut load_system = FemSystem::new(
        TripletMatrix::new(space.num_global_dofs(&mesh), space.num_global_dofs(&mesh)),
        vec![0.0; space.num_global_dofs(&mesh)],
    );
    apply_neumann_real(&mut load_system, &mesh, 2, &angular_flux);
    let boundary_load = load_system.rhs;
    let input_power = boundary_load.iter().sum::<f64>();
    let exact_power = exact_average_flux() * 4.0 * PI * RADII_M[3].powi(2);

    let inner_dofs: Vec<_> = mesh
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == 1)
        .flat_map(|boundary| boundary.nodes.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut effective = add_scaled(&capacity, 1.0, &conduction, TIME_STEP_S);
    impose_zero_rows(&mut effective, &inner_dofs);
    let factorization = SparseLuFactorization::factorize(&effective)
        .expect("literal multilayer-sphere heat operator must factorize");

    let mut temperature = vec![0.0; mesh.num_nodes()];
    let mut removed_heat = 0.0;
    let mut relative_free_residual = 0.0;
    for _ in 0..(FINAL_TIME_S / TIME_STEP_S).round() as usize {
        let previous = temperature.clone();
        let mut rhs = capacity.matvec(&previous);
        for (value, load) in rhs.iter_mut().zip(&boundary_load) {
            *value += TIME_STEP_S * load;
        }
        for &dof in &inner_dofs {
            rhs[dof] = 0.0;
        }
        temperature = factorization.apply(&rhs);

        let increment: Vec<_> = temperature
            .iter()
            .zip(&previous)
            .map(|(current, old)| current - old)
            .collect();
        let capacity_increment = capacity.matvec(&increment);
        let conductive_step = conduction.matvec(&temperature);
        let residual: Vec<_> = capacity_increment
            .iter()
            .zip(&conductive_step)
            .zip(&boundary_load)
            .map(|((mass, stiffness), load)| mass + TIME_STEP_S * (stiffness - load))
            .collect();
        removed_heat -= inner_dofs.iter().map(|&dof| residual[dof]).sum::<f64>();
        let free_norm = residual
            .iter()
            .enumerate()
            .filter(|(index, _)| inner_dofs.binary_search(index).is_err())
            .map(|(_, value)| value * value)
            .sum::<f64>()
            .sqrt();
        relative_free_residual = free_norm / (TIME_STEP_S * input_power);
    }

    let spherical_average =
        SAMPLE_RADII_M.map(|radius| spherical_surface_average(&mesh, &temperature, radius));
    let outer_values: Vec<_> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| {
            ((node.x.hypot(node.y).hypot(node.z) - SAMPLE_RADII_M[2]).abs() < 1.0e-10)
                .then_some(temperature[index])
        })
        .collect();
    let angular_spread = outer_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        - outer_values.iter().copied().fold(f64::INFINITY, f64::min);
    let stored_heat = capacity.matvec(&temperature).iter().sum::<f64>();
    LiteralResult {
        spherical_average,
        angular_spread,
        relative_flux_quadrature_error: (input_power - exact_power).abs() / exact_power,
        relative_energy_balance: (stored_heat + removed_heat - input_power * FINAL_TIME_S).abs()
            / (input_power * FINAL_TIME_S),
        relative_free_residual,
    }
}

#[test]
fn literal_three_dimensional_multilayer_sphere_matches_wolfram_average() {
    let coarse = solve_literal(1);
    let fine = solve_literal(2);
    eprintln!("literal Wolfram multilayer sphere: coarse={coarse:?}, fine={fine:?}");

    assert!(fine.relative_flux_quadrature_error < 0.03);
    assert!(fine.relative_energy_balance < 1.0e-10);
    assert!(fine.relative_free_residual < 1.0e-10);
    assert!(
        fine.angular_spread > 1.0,
        "angular heat flux must produce a 3D field"
    );
    for (&coarse_value, &fine_value) in coarse.spherical_average.iter().zip(&fine.spherical_average)
    {
        assert!((fine_value - coarse_value).abs() / fine_value < 0.08);
    }

    let reference: WolframReference = serde_json::from_str(include_str!(
        "../wolfram/goldens/multilayer_sphere_radial_average.json"
    ))
    .expect("checked-in exact Wolfram degree-zero reference");
    assert!((reference.spherical_average_flux - exact_average_flux()).abs() < 1.0e-12);
    for (&actual, &expected) in fine
        .spherical_average
        .iter()
        .zip(reference.spherical_average_samples[5].iter())
    {
        assert!((actual - expected).abs() / expected < 0.08);
    }
}
