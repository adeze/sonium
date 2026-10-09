//! Sonium reproduction of Wolfram verification case
//! `SolidMechanics-FEM-Stationary-2D-PlaneStress-0002`.
//!
//! The official case is a unit-square quarter plate with a 0.1 m quarter-hole,
//! symmetry rollers on `x=0` and `y=0`, and a 1000 Pa tensile traction on
//! `x=1`. Wolfram checks the recovered `sigma_xx` at the crown of the hole
//! against the Kirsch value `3 sigma` with a 4-5% tolerance.

use serde::Deserialize;
use sonium_fem::{
    assembly::assemble_bilinear,
    basis::PolynomialDegree,
    mesh::{BoundaryType, ElementType, Mesh, Point},
    physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption},
    spaces::{FunctionSpace, VectorH1Space},
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{CsrMatrix, GmresConfig, IluPreconditioner, gmres_preconditioned};
use std::{collections::BTreeSet, env, f64::consts::FRAC_PI_2, path::PathBuf, process::Command};

const WIDTH_M: f64 = 1.0;
const HEIGHT_M: f64 = 1.0;
const THICKNESS_M: f64 = 0.1;
const HOLE_RADIUS_M: f64 = 0.1;
const YOUNG_MODULUS_PA: f64 = 200.0e9;
const POISSON_RATIO: f64 = 0.3;
const TRACTION_PA: f64 = 1_000.0;
const ANGULAR_CELLS: usize = 64;
const RADIAL_CELLS: usize = 36;

fn quarter_plate_mesh() -> Mesh {
    let mut mesh = Mesh::new(2);
    for radial in 0..=RADIAL_CELLS {
        let s = radial as f64 / RADIAL_CELLS as f64;
        let graded = s.powf(1.5);
        for angular in 0..=ANGULAR_CELLS {
            let theta = FRAC_PI_2 * angular as f64 / ANGULAR_CELLS as f64;
            let outer_radius = if theta <= std::f64::consts::FRAC_PI_4 {
                WIDTH_M / theta.cos()
            } else {
                HEIGHT_M / theta.sin()
            };
            let radius = HOLE_RADIUS_M + graded * (outer_radius - HOLE_RADIUS_M);
            mesh.add_node(Point::new_2d(radius * theta.cos(), radius * theta.sin()));
        }
    }

    let stride = ANGULAR_CELLS + 1;
    for radial in 0..RADIAL_CELLS {
        for angular in 0..ANGULAR_CELLS {
            let inner = radial * stride + angular;
            let inner_next = inner + 1;
            let outer = inner + stride;
            let outer_next = outer + 1;
            mesh.add_element(ElementType::Triangle, vec![inner, outer, outer_next]);
            mesh.add_element(ElementType::Triangle, vec![inner, outer_next, inner_next]);
        }
    }

    mesh.detect_boundaries();
    let tolerance = 1.0e-10;
    mesh.set_boundary_condition(BoundaryType::Neumann, 1, |points| {
        points.iter().all(|point| {
            ((point.x * point.x + point.y * point.y).sqrt() - HOLE_RADIUS_M).abs() < tolerance
        })
    });
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 2, |points| {
        points.iter().all(|point| point.y.abs() < tolerance)
    });
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 3, |points| {
        points.iter().all(|point| point.x.abs() < tolerance)
    });
    mesh.set_boundary_condition(BoundaryType::Neumann, 4, |points| {
        points
            .iter()
            .all(|point| (point.x - WIDTH_M).abs() < tolerance)
    });
    mesh.set_boundary_condition(BoundaryType::Neumann, 5, |points| {
        points
            .iter()
            .all(|point| (point.y - HEIGHT_M).abs() < tolerance)
    });
    mesh
}

fn component_dofs(mesh: &Mesh, marker: i32, component: usize) -> Vec<usize> {
    mesh.boundaries
        .iter()
        .filter(|boundary| boundary.marker == marker)
        .flat_map(|boundary| boundary.nodes.iter().copied())
        .map(|node| 2 * node + component)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn apply_zero_rows(matrix: &mut CsrMatrix<f64>, rhs: &mut [f64], dofs: &[usize]) {
    for &dof in dofs {
        let row_start = matrix.row_ptrs[dof];
        let row_end = matrix.row_ptrs[dof + 1];
        let mut diagonal = None;
        for index in row_start..row_end {
            if matrix.col_indices[index] == dof {
                diagonal = Some(index);
            }
            matrix.values[index] = 0.0;
        }
        matrix.values[diagonal.expect("elasticity row must contain its diagonal")] = 1.0;
        rhs[dof] = 0.0;
    }
}

fn assemble_right_traction(mesh: &Mesh, rhs: &mut [f64]) -> f64 {
    let mut total = 0.0;
    for boundary in mesh
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == 4)
    {
        assert_eq!(boundary.nodes.len(), 2);
        let first = mesh.nodes[boundary.nodes[0]];
        let second = mesh.nodes[boundary.nodes[1]];
        let length = ((second.x - first.x).powi(2) + (second.y - first.y).powi(2)).sqrt();
        let force = TRACTION_PA * length;
        rhs[2 * boundary.nodes[0]] += 0.5 * force;
        rhs[2 * boundary.nodes[1]] += 0.5 * force;
        total += force;
    }
    total
}

fn element_stress_xx(mesh: &Mesh, element_index: usize, displacement: &[f64]) -> f64 {
    let nodes = &mesh.elements[element_index].nodes;
    assert_eq!(nodes.len(), 3);
    let [first, second, third] = [
        mesh.nodes[nodes[0]],
        mesh.nodes[nodes[1]],
        mesh.nodes[nodes[2]],
    ];
    let twice_area =
        (second.x - first.x) * (third.y - first.y) - (third.x - first.x) * (second.y - first.y);
    assert!(
        twice_area > 0.0,
        "quarter-plate triangles must be counter-clockwise"
    );

    let dndx = [
        (second.y - third.y) / twice_area,
        (third.y - first.y) / twice_area,
        (first.y - second.y) / twice_area,
    ];
    let dndy = [
        (third.x - second.x) / twice_area,
        (first.x - third.x) / twice_area,
        (second.x - first.x) / twice_area,
    ];
    let mut epsilon_xx = 0.0;
    let mut epsilon_yy = 0.0;
    for local in 0..3 {
        epsilon_xx += dndx[local] * displacement[2 * nodes[local]];
        epsilon_yy += dndy[local] * displacement[2 * nodes[local] + 1];
    }
    YOUNG_MODULUS_PA / (1.0 - POISSON_RATIO.powi(2)) * (epsilon_xx + POISSON_RATIO * epsilon_yy)
}

fn solve_dense_system(mut matrix: [[f64; 6]; 6], mut rhs: [f64; 6]) -> [f64; 6] {
    for column in 0..6 {
        let pivot = (column..6)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))
            .expect("nonempty pivot range");
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let diagonal = matrix[column][column];
        assert!(diagonal.abs() > 1.0e-12, "stress-recovery fit is singular");
        let pivot_row = matrix[column];
        for row in column + 1..6 {
            let factor = matrix[row][column] / diagonal;
            for (target, &pivot_value) in matrix[row][column..].iter_mut().zip(&pivot_row[column..])
            {
                *target -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    let mut solution = [0.0; 6];
    for row in (0..6).rev() {
        let known = (row + 1..6)
            .map(|column| matrix[row][column] * solution[column])
            .sum::<f64>();
        solution[row] = (rhs[row] - known) / matrix[row][row];
    }
    solution
}

/// Test-local quadratic patch recovery at the hole crown. A reusable version
/// belongs in Sonium's public elasticity postprocessing API.
fn recovered_hole_crown_stress(mesh: &Mesh, displacement: &[f64]) -> f64 {
    const PATCH_RADIUS_M: f64 = 0.09;
    let mut normal = [[0.0; 6]; 6];
    let mut rhs = [0.0; 6];
    let mut samples = 0;
    for (element_index, element) in mesh.elements.iter().enumerate() {
        let centroid_x = element
            .nodes
            .iter()
            .map(|&node| mesh.nodes[node].x)
            .sum::<f64>()
            / 3.0;
        let centroid_y = element
            .nodes
            .iter()
            .map(|&node| mesh.nodes[node].y)
            .sum::<f64>()
            / 3.0;
        let dx = centroid_x / PATCH_RADIUS_M;
        let dy = (centroid_y - HOLE_RADIUS_M) / PATCH_RADIUS_M;
        let distance = (dx * dx + dy * dy).sqrt();
        if distance > 1.0 {
            continue;
        }
        let basis = [1.0, dx, dy, dx * dx, dx * dy, dy * dy];
        let weight = 1.0 / (0.15 + distance);
        let stress = element_stress_xx(mesh, element_index, displacement);
        for row in 0..6 {
            rhs[row] += weight * basis[row] * stress;
            for column in 0..6 {
                normal[row][column] += weight * basis[row] * basis[column];
            }
        }
        samples += 1;
    }
    assert!(
        samples >= 12,
        "stress-recovery patch has only {samples} samples"
    );
    solve_dense_system(normal, rhs)[0]
}

struct SoniumResult {
    crown_stress_pa: f64,
    applied_force_n_per_m: f64,
    support_reaction_n_per_m: f64,
}

fn sonium_result() -> SoniumResult {
    let mesh = quarter_plate_mesh();
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let stiffness = assemble_bilinear(
        &ElasticityForm::new_2d(
            IsotropicMaterial::new(YOUNG_MODULUS_PA, POISSON_RATIO),
            PlaneAssumption::PlaneStress,
        ),
        &space,
        &space,
        &mesh,
        2,
    );
    let original = stiffness.to_csr();
    let mut constrained = stiffness.to_csr();
    let mut rhs = vec![0.0; space.num_global_dofs(&mesh)];
    let applied_force = assemble_right_traction(&mesh, &mut rhs);
    let unconstrained_rhs = rhs.clone();

    // y=0: u_y=0. x=0: u_x=0.
    apply_zero_rows(&mut constrained, &mut rhs, &component_dofs(&mesh, 2, 1));
    apply_zero_rows(&mut constrained, &mut rhs, &component_dofs(&mesh, 3, 0));
    let result = gmres_preconditioned(
        &constrained,
        &IluPreconditioner::from_csr(&constrained),
        &rhs,
        &GmresConfig {
            max_iterations: 5_000,
            restart: 200,
            tolerance: 1.0e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(result.converged, "quarter plate solve did not converge");

    let internal = original.matvec(&result.x);
    let reaction_x = component_dofs(&mesh, 3, 0)
        .into_iter()
        .map(|dof| internal[dof] - unconstrained_rhs[dof])
        .sum::<f64>();
    SoniumResult {
        crown_stress_pa: recovered_hole_crown_stress(&mesh, &result.x),
        applied_force_n_per_m: applied_force,
        support_reaction_n_per_m: reaction_x,
    }
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    case_id: String,
    dimensions_m: [f64; 3],
    hole_radius_m: f64,
    young_modulus_pa: f64,
    poisson_ratio: f64,
    traction_pa: f64,
    analytical_crown_stress_pa: f64,
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/plate_with_hole_plane_stress.wls");
    let output = match Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Wolfram Engine not available; using the published Kirsch reference");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram plate-with-hole reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram plate-with-hole output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

fn assert_stress_close(actual: f64, reference: f64) {
    let relative_error = (actual - reference).abs() / reference.abs();
    assert!(
        relative_error < 0.05,
        "recovered crown stress {actual:.9e} Pa differs from {reference:.9e} Pa by {:.3}%",
        100.0 * relative_error
    );
}

#[test]
fn sonium_reproduces_wolfram_quarter_plate_kirsch_stress() {
    let sonium = sonium_result();
    let exact_stress = 3.0 * TRACTION_PA;
    assert_stress_close(sonium.crown_stress_pa, exact_stress);
    assert!((sonium.applied_force_n_per_m - TRACTION_PA * HEIGHT_M).abs() < 1.0e-10);
    assert!(
        (sonium.support_reaction_n_per_m + sonium.applied_force_n_per_m).abs()
            / sonium.applied_force_n_per_m
            < 1.0e-8,
        "support reaction {} N/m does not balance applied force {} N/m",
        sonium.support_reaction_n_per_m,
        sonium.applied_force_n_per_m
    );

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(
            reference.case_id,
            "SolidMechanics-FEM-Stationary-2D-PlaneStress-0002"
        );
        assert_eq!(reference.dimensions_m, [WIDTH_M, HEIGHT_M, THICKNESS_M]);
        assert_eq!(reference.hole_radius_m, HOLE_RADIUS_M);
        assert_eq!(reference.young_modulus_pa, YOUNG_MODULUS_PA);
        assert_eq!(reference.poisson_ratio, POISSON_RATIO);
        assert_eq!(reference.traction_pa, TRACTION_PA);
        assert_eq!(reference.analytical_crown_stress_pa, exact_stress);
        assert_stress_close(sonium.crown_stress_pa, reference.analytical_crown_stress_pa);
    }
}
