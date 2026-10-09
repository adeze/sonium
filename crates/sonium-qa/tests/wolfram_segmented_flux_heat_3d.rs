//! Three-dimensional rotational-sector check for Wolfram/NAFEMS heat case
//! `HeatTransfer-FEM-Stationary-2DAxisym-Single-HeatTransfer-0002-A`.
//!
//! The published problem is axisymmetric. A finite 30-degree 3D sector with
//! insulated cut faces is an exact symmetry reduction of the full annular
//! cylinder and independently exercises Sonium's tetrahedral thermal form.

use sonium_fem::{
    assembly::assemble_bilinear,
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_dirichlet_values},
    mesh::{BoundaryType, Mesh, Point, box_mesh_tetrahedra, interpolate_scalar_at_point},
    physics::thermal::heat_conduction::ConductionForm,
    spaces::H1Space,
};
use sonium_solvers::{SparseLuFactorization, traits::Preconditioner};
use std::f64::consts::PI;

const INNER_RADIUS_M: f64 = 0.02;
const OUTER_RADIUS_M: f64 = 0.1;
const HEIGHT_M: f64 = 0.14;
const FLUX_Z_MIN_M: f64 = 0.04;
const FLUX_Z_MAX_M: f64 = 0.1;
const SECTOR_ANGLE_RAD: f64 = PI / 6.0;
const CONDUCTIVITY_W_M_K: f64 = 52.0;
const HEAT_FLUX_W_M2: f64 = 5.0e5;
const FIXED_TEMPERATURE_K: f64 = 273.15;
const PUBLISHED_TEMPERATURE_K: f64 = 332.97;
const SAMPLE_POINT_M: [f64; 3] = [0.04, 0.0, 0.04];

#[derive(Debug)]
struct SolutionSummary {
    temperature_k: f64,
    input_power_w: f64,
    heat_balance_relative_error: f64,
    nodes: usize,
    elements: usize,
}

fn sector_mesh(radial_cells: usize, angular_cells: usize, axial_cells: usize) -> Mesh {
    let mut mesh = box_mesh_tetrahedra(
        INNER_RADIUS_M,
        OUTER_RADIUS_M,
        0.0,
        SECTOR_ANGLE_RAD,
        0.0,
        HEIGHT_M,
        radial_cells,
        angular_cells,
        axial_cells,
    );
    for point in &mut mesh.nodes {
        let radius = point.x;
        let angle = point.y;
        *point = Point::new_3d(radius * angle.cos(), radius * angle.sin(), point.z);
    }
    let tolerance = 1.0e-10;
    mesh.set_boundary_condition(BoundaryType::Neumann, 1, |points| {
        points.iter().all(|point| {
            ((point.x * point.x + point.y * point.y).sqrt() - INNER_RADIUS_M).abs() < tolerance
        })
    });
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 2, |points| {
        points.iter().all(|point| {
            ((point.x * point.x + point.y * point.y).sqrt() - OUTER_RADIUS_M).abs() < tolerance
        })
    });
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 3, |points| {
        points.iter().all(|point| point.z.abs() < tolerance)
    });
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 4, |points| {
        points
            .iter()
            .all(|point| (point.z - HEIGHT_M).abs() < tolerance)
    });
    mesh
}

fn triangle_area(points: [Point; 3]) -> f64 {
    let first = [
        points[1].x - points[0].x,
        points[1].y - points[0].y,
        points[1].z - points[0].z,
    ];
    let second = [
        points[2].x - points[0].x,
        points[2].y - points[0].y,
        points[2].z - points[0].z,
    ];
    let cross = [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ];
    0.5 * cross.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn assemble_segmented_flux(mesh: &Mesh, rhs: &mut [f64]) -> f64 {
    let mut input_power = 0.0;
    for face in &mesh.boundaries {
        if face.marker != 1 || face.nodes.len() != 3 {
            continue;
        }
        let points = [
            mesh.nodes[face.nodes[0]],
            mesh.nodes[face.nodes[1]],
            mesh.nodes[face.nodes[2]],
        ];
        let centroid_z = points.iter().map(|point| point.z).sum::<f64>() / 3.0;
        if !(FLUX_Z_MIN_M - 1.0e-12..=FLUX_Z_MAX_M + 1.0e-12).contains(&centroid_z) {
            continue;
        }
        let load = HEAT_FLUX_W_M2 * triangle_area(points);
        input_power += load;
        for &node in &face.nodes {
            rhs[node] += load / 3.0;
        }
    }
    input_power
}

fn solve(radial_cells: usize, angular_cells: usize, axial_cells: usize) -> SolutionSummary {
    let mesh = sector_mesh(radial_cells, angular_cells, axial_cells);
    let space = H1Space::new(PolynomialDegree::P1);
    let stiffness = assemble_bilinear(
        &ConductionForm::from_conductivity(CONDUCTIVITY_W_M_K),
        &space,
        &space,
        &mesh,
        2,
    );
    let unconstrained = stiffness.to_csr();
    let mut rhs = vec![0.0; mesh.num_nodes()];
    let input_power = assemble_segmented_flux(&mesh, &mut rhs);
    let unconstrained_rhs = rhs.clone();
    let mut system = FemSystem::new(stiffness, rhs);
    for marker in [2, 3, 4] {
        apply_dirichlet_values(
            &mut system,
            &mesh,
            &space,
            marker,
            &|_, _, _| FIXED_TEMPERATURE_K,
            None,
        );
    }
    let matrix = system.matrix.to_csr();
    let factor = SparseLuFactorization::factorize(&matrix).expect("factor 3D heat matrix");
    let temperature = factor.apply(&system.rhs);
    let temperature_k = interpolate_scalar_at_point(&mesh, &temperature, SAMPLE_POINT_M)
        .expect("interpolate 3D heat sample");

    let residual = unconstrained.matvec(&temperature);
    let reaction = -(0..mesh.num_nodes())
        .filter(|&node| {
            let point = mesh.nodes[node];
            let radius = (point.x * point.x + point.y * point.y).sqrt();
            (radius - OUTER_RADIUS_M).abs() < 1.0e-10
                || point.z.abs() < 1.0e-10
                || (point.z - HEIGHT_M).abs() < 1.0e-10
        })
        .map(|node| residual[node] - unconstrained_rhs[node])
        .sum::<f64>();

    SolutionSummary {
        temperature_k,
        input_power_w: input_power,
        heat_balance_relative_error: (reaction - input_power).abs() / input_power,
        nodes: mesh.num_nodes(),
        elements: mesh.elements.len(),
    }
}

#[test]
fn sonium_3d_sector_reproduces_wolfram_segmented_flux_heat_case() {
    let coarse = solve(8, 2, 14);
    let refined = solve(16, 4, 28);
    let exact_sector_power =
        HEAT_FLUX_W_M2 * INNER_RADIUS_M * SECTOR_ANGLE_RAD * (FLUX_Z_MAX_M - FLUX_Z_MIN_M);
    eprintln!(
        "Wolfram segmented-flux heat 3D sector: coarse={coarse:?}, refined={refined:?}, exact_power={exact_sector_power:.9} W"
    );

    assert!((coarse.input_power_w / exact_sector_power - 1.0).abs() < 0.01);
    assert!((refined.input_power_w / exact_sector_power - 1.0).abs() < 0.003);
    assert!(coarse.heat_balance_relative_error < 1.0e-9);
    assert!(refined.heat_balance_relative_error < 1.0e-9);
    assert!(refined.nodes > coarse.nodes && refined.elements > coarse.elements);
    assert!((refined.temperature_k / coarse.temperature_k - 1.0).abs() < 0.01);
    assert!((refined.temperature_k / PUBLISHED_TEMPERATURE_K - 1.0).abs() < 0.003);
}
