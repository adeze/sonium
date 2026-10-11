//! Shell validation tests
//!
//! Tests for flat triangular shell elements (CST membrane + DKT bending).
//!
//! - Shell section properties
//! - Element area calculation
//! - Rotation matrix orthogonality
//! - Membrane stiffness matrix symmetry
//! - Bending stiffness matrix symmetry
//! - Solved cantilever bending against beam theory (deflection, rotation)
//! - Solved axial tension against uniform stress (extension, stress)
//! - Global force and moment equilibrium of the solved reactions

use sonium_fem::mesh::{Point, rectangular_mesh_triangles};
use sonium_fem::physics::solid::shell::{ShellElement, ShellSection};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "fem.shell-validation.v1";

// Cantilever strip geometry and material (aluminum).
const STRIP_LENGTH: f64 = 1.0;
const STRIP_WIDTH: f64 = 0.1;
const STRIP_THICKNESS: f64 = 0.01;
const YOUNGS_MODULUS: f64 = 70e9;
const POISSON_RATIO: f64 = 0.33;
const CELLS_X: usize = 12;
const CELLS_Y: usize = 2;

// Applied loads.
const TIP_LOAD: f64 = 50.0;
const AXIAL_LOAD: f64 = 1e6;

// Acceptance tolerances.
const TOL_SECTION: f64 = 1e-6;
const TOL_GEOMETRY: f64 = 1e-10;
const TOL_DEFLECTION: f64 = 0.05;
const TOL_ROTATION: f64 = 0.08;
const TOL_EXTENSION: f64 = 0.02;
const TOL_STRESS: f64 = 0.05;
const TOL_EQUILIBRIUM: f64 = 1e-6;
const TOL_RESIDUAL: f64 = 1e-9;

fn check(condition: bool, message: String) -> Result<(), String> {
    if condition { Ok(()) } else { Err(message) }
}

/// Dense Gaussian elimination with partial pivoting.
fn dense_solve(matrix: &mut [f64], rhs: &mut [f64], n: usize) -> Result<(), String> {
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_value = matrix[col * n + col].abs();
        for row in (col + 1)..n {
            let value = matrix[row * n + col].abs();
            if value > pivot_value {
                pivot_value = value;
                pivot = row;
            }
        }
        if pivot_value == 0.0 {
            return Err(format!("singular matrix at column {col}"));
        }
        if pivot != col {
            for k in col..n {
                matrix.swap(col * n + k, pivot * n + k);
            }
            rhs.swap(col, pivot);
        }
        let diagonal = matrix[col * n + col];
        for row in (col + 1)..n {
            let factor = matrix[row * n + col] / diagonal;
            matrix[row * n + col] = 0.0;
            for k in (col + 1)..n {
                matrix[row * n + k] -= factor * matrix[col * n + k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    for row in (0..n).rev() {
        let mut sum = rhs[row];
        for k in (row + 1)..n {
            sum -= matrix[row * n + k] * rhs[k];
        }
        rhs[row] = sum / matrix[row * n + row];
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct Metrics {
    component_checks: usize,
    tip_deflection: f64,
    deflection_error: f64,
    tip_rotation: f64,
    rotation_error: f64,
    bending_residual: f64,
    bending_force_error: f64,
    bending_moment_error: f64,
    tip_extension: f64,
    extension_error: f64,
    membrane_stress: f64,
    membrane_stress_error: f64,
    tension_force_error: f64,
}

fn run() -> Result<Metrics, String> {
    let mut component_checks = 0usize;

    // Section creation.
    let section = ShellSection::new(YOUNGS_MODULUS, POISSON_RATIO, STRIP_THICKNESS, 2700.0);
    check(
        (section.thickness - STRIP_THICKNESS).abs() < 1e-10,
        "section thickness mismatch".to_string(),
    )?;
    component_checks += 1;
    check(
        (section.youngs_modulus - YOUNGS_MODULUS).abs() < 1e6,
        "section modulus mismatch".to_string(),
    )?;
    component_checks += 1;
    check(
        (section.poisson_ratio - POISSON_RATIO).abs() < 1e-6,
        "section Poisson mismatch".to_string(),
    )?;
    component_checks += 1;

    // Bending and membrane stiffness formulas.
    let bending = section.bending_stiffness();
    let expected_bending =
        YOUNGS_MODULUS * STRIP_THICKNESS.powi(3) / (12.0 * (1.0 - POISSON_RATIO * POISSON_RATIO));
    check(
        (bending - expected_bending).abs() / expected_bending < TOL_SECTION,
        "bending stiffness mismatch".to_string(),
    )?;
    component_checks += 1;
    let membrane = section.membrane_stiffness();
    let expected_membrane =
        YOUNGS_MODULUS * STRIP_THICKNESS / (1.0 - POISSON_RATIO * POISSON_RATIO);
    check(
        (membrane - expected_membrane).abs() / expected_membrane < TOL_SECTION,
        "membrane stiffness mismatch".to_string(),
    )?;
    component_checks += 1;

    // Element geometry on the reference triangle.
    let p0 = Point::new_3d(0.0, 0.0, 0.0);
    let p1 = Point::new_3d(1.0, 0.0, 0.0);
    let p2 = Point::new_3d(0.0, 1.0, 0.0);
    let element = ShellElement::from_points([0, 1, 2], &p0, &p1, &p2, section);
    check(
        (element.area - 0.5).abs() < TOL_GEOMETRY,
        "element area mismatch".to_string(),
    )?;
    component_checks += 1;

    // Rotation orthogonality.
    let rotation = &element.rotation;
    let mut max_err: f64 = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let dot: f64 = (0..3)
                .map(|k| rotation[i * 3 + k] * rotation[j * 3 + k])
                .sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            max_err = max_err.max((dot - expected).abs());
        }
    }
    check(
        max_err < TOL_GEOMETRY,
        format!("rotation not orthogonal: {max_err:.2e}"),
    )?;
    component_checks += 1;

    // Local coordinates.
    check(
        element.local_coords[0].0.abs() < TOL_GEOMETRY
            && element.local_coords[0].1.abs() < TOL_GEOMETRY
            && (element.local_coords[1].0 - 1.0).abs() < TOL_GEOMETRY,
        "local coordinates mismatch".to_string(),
    )?;
    component_checks += 1;

    // Alloy spot checks.
    let aluminum = ShellSection::new(70e9, 0.33, 1e-3, 2700.0);
    check(
        (aluminum.youngs_modulus - 70e9).abs() < 1e9 && (aluminum.density - 2700.0).abs() < 10.0,
        "aluminum section mismatch".to_string(),
    )?;
    component_checks += 1;
    let steel = ShellSection::new(200e9, 0.3, 2e-3, 7800.0);
    check(
        (steel.youngs_modulus - 200e9).abs() < 1e9 && (steel.density - 7800.0).abs() < 50.0,
        "steel section mismatch".to_string(),
    )?;
    component_checks += 1;

    // Solved cantilever bending: clamp the root edge, load the tip edge.
    let mesh = rectangular_mesh_triangles(0.0, STRIP_LENGTH, 0.0, STRIP_WIDTH, CELLS_X, CELLS_Y);
    let section = ShellSection::new(YOUNGS_MODULUS, POISSON_RATIO, STRIP_THICKNESS, 2700.0);
    let elements: Vec<ShellElement> = mesh
        .elements
        .iter()
        .map(|elem| {
            ShellElement::from_mesh(
                [elem.nodes[0], elem.nodes[1], elem.nodes[2]],
                &mesh,
                section,
            )
        })
        .collect();
    let n_dofs = mesh.num_nodes() * 6;
    let triplet = sonium_fem::physics::solid::shell::assemble_shell_stiffness(&elements, n_dofs);
    let mut stiffness = vec![0.0; n_dofs * n_dofs];
    for k in 0..triplet.values.len() {
        stiffness[triplet.rows[k] * n_dofs + triplet.cols[k]] += triplet.values[k];
    }

    let coord_tol = 1e-10;
    let is_root = |node: usize| mesh.nodes[node].x.abs() < coord_tol;
    let is_tip = |node: usize| (mesh.nodes[node].x - STRIP_LENGTH).abs() < coord_tol;
    let tip_center = (0..mesh.num_nodes())
        .find(|&node| is_tip(node) && (mesh.nodes[node].y - 0.5 * STRIP_WIDTH).abs() < coord_tol)
        .ok_or_else(|| "tip center node not found".to_string())?;
    let tip_nodes: Vec<usize> = (0..mesh.num_nodes()).filter(|&node| is_tip(node)).collect();
    check(!tip_nodes.is_empty(), "no tip nodes found".to_string())?;

    let mut constrained = Vec::new();
    for node in 0..mesh.num_nodes() {
        if is_root(node) {
            for dof in 0..6 {
                constrained.push(node * 6 + dof);
            }
        }
    }
    check(!constrained.is_empty(), "no root nodes found".to_string())?;

    // Bending solve with the tip load distributed over the tip edge.
    let mut load = vec![0.0; n_dofs];
    for &node in &tip_nodes {
        sonium_fem::physics::solid::shell::add_shell_point_load(
            &mut load,
            node,
            [0.0, 0.0, -TIP_LOAD / tip_nodes.len() as f64, 0.0, 0.0, 0.0],
        );
    }
    let mut bending_matrix = stiffness.clone();
    let mut bending_rhs = load.clone();
    for &dof in &constrained {
        for col in 0..n_dofs {
            bending_matrix[dof * n_dofs + col] = 0.0;
            bending_matrix[col * n_dofs + dof] = 0.0;
        }
        bending_matrix[dof * n_dofs + dof] = 1.0;
        bending_rhs[dof] = 0.0;
    }
    dense_solve(&mut bending_matrix, &mut bending_rhs, n_dofs)?;
    let bending_solution = bending_rhs;

    // Free-DOF equilibrium residual with the unmodified stiffness.
    let mut bending_residual = 0.0f64;
    let mut load_norm = 0.0f64;
    let is_constrained = |dof: usize| constrained.contains(&dof);
    for row in 0..n_dofs {
        load_norm += load[row] * load[row];
        if is_constrained(row) {
            continue;
        }
        let mut row_sum = 0.0;
        for col in 0..n_dofs {
            row_sum += stiffness[row * n_dofs + col] * bending_solution[col];
        }
        let residual = row_sum - load[row];
        bending_residual += residual * residual;
    }
    bending_residual = (bending_residual / load_norm).sqrt();
    check(
        bending_residual <= TOL_RESIDUAL,
        format!("bending residual {bending_residual:.3e} exceeds {TOL_RESIDUAL:.1e}"),
    )?;

    // Beam theory references for tip deflection and rotation.
    let second_moment = STRIP_WIDTH * STRIP_THICKNESS.powi(3) / 12.0;
    let reference_deflection =
        TIP_LOAD * STRIP_LENGTH.powi(3) / (3.0 * YOUNGS_MODULUS * second_moment);
    let reference_rotation =
        TIP_LOAD * STRIP_LENGTH.powi(2) / (2.0 * YOUNGS_MODULUS * second_moment);
    let tip_deflection = bending_solution[tip_center * 6 + 2].abs();
    let deflection_error = (tip_deflection - reference_deflection).abs() / reference_deflection;
    check(
        deflection_error <= TOL_DEFLECTION,
        format!("deflection error {deflection_error:.3e} exceeds {TOL_DEFLECTION:.1e}"),
    )?;
    let tip_rotation = bending_solution[tip_center * 6 + 4].abs();
    let rotation_error = (tip_rotation - reference_rotation).abs() / reference_rotation;
    check(
        rotation_error <= TOL_ROTATION,
        format!("rotation error {rotation_error:.3e} exceeds {TOL_ROTATION:.1e}"),
    )?;

    // Global equilibrium: reactions balance the applied tip load.
    let mut reaction_force = [0.0; 3];
    let mut reaction_moment = [0.0; 3];
    for &dof in &constrained {
        let node = dof / 6;
        let component = dof % 6;
        let mut row_sum = 0.0;
        for col in 0..n_dofs {
            row_sum += stiffness[dof * n_dofs + col] * bending_solution[col];
        }
        let reaction = row_sum - load[dof];
        let point = &mesh.nodes[node];
        let position = [point.x, point.y, point.z];
        if component < 3 {
            reaction_force[component] += reaction;
            let moment = [
                position[1] * reaction * f64::from(component == 2)
                    - position[2] * reaction * f64::from(component == 1),
                position[2] * reaction * f64::from(component == 0)
                    - position[0] * reaction * f64::from(component == 2),
                position[0] * reaction * f64::from(component == 1)
                    - position[1] * reaction * f64::from(component == 0),
            ];
            for axis in 0..3 {
                reaction_moment[axis] += moment[axis];
            }
        } else {
            reaction_moment[component - 3] += reaction;
        }
    }
    let mut external_force = [0.0; 3];
    let mut external_moment = [0.0; 3];
    for node in 0..mesh.num_nodes() {
        let point = &mesh.nodes[node];
        let force = [load[node * 6], load[node * 6 + 1], load[node * 6 + 2]];
        for axis in 0..3 {
            external_force[axis] += force[axis];
        }
        external_moment[0] += point.y * force[2] - point.z * force[1];
        external_moment[1] += point.z * force[0] - point.x * force[2];
        external_moment[2] += point.x * force[1] - point.y * force[0];
    }
    let force_norm: f64 = external_force
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let force_imbalance: f64 = reaction_force
        .iter()
        .zip(external_force.iter())
        .map(|(reaction, external)| (reaction + external).powi(2))
        .sum::<f64>()
        .sqrt();
    let bending_force_error = force_imbalance / force_norm;
    check(
        bending_force_error <= TOL_EQUILIBRIUM,
        format!("bending force imbalance {bending_force_error:.3e} exceeds {TOL_EQUILIBRIUM:.1e}"),
    )?;
    let moment_norm: f64 = external_moment
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let moment_imbalance: f64 = reaction_moment
        .iter()
        .zip(external_moment.iter())
        .map(|(reaction, external)| (reaction + external).powi(2))
        .sum::<f64>()
        .sqrt();
    let bending_moment_error = moment_imbalance / moment_norm;
    check(
        bending_moment_error <= TOL_EQUILIBRIUM,
        format!(
            "bending moment imbalance {bending_moment_error:.3e} exceeds {TOL_EQUILIBRIUM:.1e}"
        ),
    )?;

    // Membrane tension solve with axial tip loading.
    let mut axial_load = vec![0.0; n_dofs];
    for &node in &tip_nodes {
        sonium_fem::physics::solid::shell::add_shell_point_load(
            &mut axial_load,
            node,
            [AXIAL_LOAD / tip_nodes.len() as f64, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
    }
    let mut tension_matrix = stiffness.clone();
    let mut tension_rhs = axial_load.clone();
    for &dof in &constrained {
        for col in 0..n_dofs {
            tension_matrix[dof * n_dofs + col] = 0.0;
            tension_matrix[col * n_dofs + dof] = 0.0;
        }
        tension_matrix[dof * n_dofs + dof] = 1.0;
        tension_rhs[dof] = 0.0;
    }
    dense_solve(&mut tension_matrix, &mut tension_rhs, n_dofs)?;
    let tension_solution = tension_rhs;

    let cross_section = STRIP_WIDTH * STRIP_THICKNESS;
    let reference_extension = AXIAL_LOAD * STRIP_LENGTH / (YOUNGS_MODULUS * cross_section);
    let tip_extension = tension_solution[tip_center * 6];
    let extension_error = (tip_extension - reference_extension).abs() / reference_extension;
    check(
        extension_error <= TOL_EXTENSION,
        format!("extension error {extension_error:.3e} exceeds {TOL_EXTENSION:.1e}"),
    )?;

    // Membrane stress recovery on a mid-strip element with local x along the
    // strip axis, compared against the uniform tension reference.
    let mid_element = elements
        .iter()
        .find(|element| {
            let centroid_x = element
                .nodes
                .iter()
                .map(|&node| mesh.nodes[node].x)
                .sum::<f64>()
                / 3.0;
            let edge_x = mesh.nodes[element.nodes[1]].x - mesh.nodes[element.nodes[0]].x;
            let edge_y = mesh.nodes[element.nodes[1]].y - mesh.nodes[element.nodes[0]].y;
            (centroid_x - 0.5 * STRIP_LENGTH).abs() < 0.5 * STRIP_LENGTH / CELLS_X as f64
                && edge_x > 0.0
                && edge_y.abs() < coord_tol
        })
        .ok_or_else(|| "mid-strip membrane element not found".to_string())?;
    let rotation = &mid_element.rotation;
    let local_axis_x = [rotation[0], rotation[3], rotation[6]];
    let local_axis_y = [rotation[1], rotation[4], rotation[7]];
    let mut local_displacements = [0.0; 6];
    for (local, &node) in mid_element.nodes.iter().enumerate() {
        let global = [
            tension_solution[node * 6],
            tension_solution[node * 6 + 1],
            tension_solution[node * 6 + 2],
        ];
        local_displacements[local * 2] =
            local_axis_x[0] * global[0] + local_axis_x[1] * global[1] + local_axis_x[2] * global[2];
        local_displacements[local * 2 + 1] =
            local_axis_y[0] * global[0] + local_axis_y[1] * global[1] + local_axis_y[2] * global[2];
    }
    let b_matrix = mid_element.cst_b_matrix();
    let mut strain = [0.0; 3];
    for row in 0..3 {
        for col in 0..6 {
            strain[row] += b_matrix[row * 6 + col] * local_displacements[col];
        }
    }
    let constitutive = section.membrane_constitutive();
    let mut membrane_force = 0.0;
    for col in 0..3 {
        membrane_force += constitutive[col] * strain[col];
    }
    // The membrane constitutive matrix yields forces per unit length.
    let membrane_stress = membrane_force / STRIP_THICKNESS;
    let reference_stress = AXIAL_LOAD / cross_section;
    let membrane_stress_error = (membrane_stress - reference_stress).abs() / reference_stress;
    check(
        membrane_stress_error <= TOL_STRESS,
        format!("membrane stress error {membrane_stress_error:.3e} exceeds {TOL_STRESS:.1e}"),
    )?;

    // Axial force equilibrium of the tension reactions.
    let mut axial_reaction = 0.0;
    for &dof in &constrained {
        if dof % 6 != 0 {
            continue;
        }
        let mut row_sum = 0.0;
        for col in 0..n_dofs {
            row_sum += stiffness[dof * n_dofs + col] * tension_solution[col];
        }
        axial_reaction += row_sum - axial_load[dof];
    }
    let tension_force_error = (axial_reaction + AXIAL_LOAD).abs() / AXIAL_LOAD;
    check(
        tension_force_error <= TOL_EQUILIBRIUM,
        format!("tension force imbalance {tension_force_error:.3e} exceeds {TOL_EQUILIBRIUM:.1e}"),
    )?;

    Ok(Metrics {
        component_checks,
        tip_deflection,
        deflection_error,
        tip_rotation,
        rotation_error,
        bending_residual,
        bending_force_error,
        bending_moment_error,
        tip_extension,
        extension_error,
        membrane_stress,
        membrane_stress_error,
        tension_force_error,
    })
}

fn main() {
    let result = match run() {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("component_checks", metrics.component_checks as f64)
            .minimum("component_checks", 10.0)
            .metric("tip_deflection", metrics.tip_deflection)
            .minimum("tip_deflection", 1e-6)
            .metric("deflection_error", metrics.deflection_error)
            .maximum("deflection_error", TOL_DEFLECTION)
            .metric("tip_rotation", metrics.tip_rotation)
            .minimum("tip_rotation", 1e-6)
            .metric("rotation_error", metrics.rotation_error)
            .maximum("rotation_error", TOL_ROTATION)
            .metric("bending_residual", metrics.bending_residual)
            .maximum("bending_residual", TOL_RESIDUAL)
            .metric("bending_force_error", metrics.bending_force_error)
            .maximum("bending_force_error", TOL_EQUILIBRIUM)
            .metric("bending_moment_error", metrics.bending_moment_error)
            .maximum("bending_moment_error", TOL_EQUILIBRIUM)
            .metric("tip_extension", metrics.tip_extension)
            .minimum("tip_extension", 1e-6)
            .metric("extension_error", metrics.extension_error)
            .maximum("extension_error", TOL_EXTENSION)
            .metric("membrane_stress", metrics.membrane_stress)
            .minimum("membrane_stress", 1e3)
            .metric("membrane_stress_error", metrics.membrane_stress_error)
            .maximum("membrane_stress_error", TOL_STRESS)
            .metric("tension_force_error", metrics.tension_force_error)
            .maximum("tension_force_error", TOL_EQUILIBRIUM)
            .reference("cantilever beam theory and uniform axial tension")
            .reference_checksum("analytic/deterministic-fixture:fem-shell-validation-v1")
            .dimension("1.0 m by 0.1 m strip, 12 by 2 cells, CST membrane plus DKT bending")
            .formulation("w = P L^3 / (3 E I); theta = P L^2 / (2 E I); sigma = P / A")
            .tolerance("deflection 5%; rotation 8%; extension 2%; stress 5%; equilibrium 1e-6")
            .input_checksum("mesh:rect-strip-12x2:shell-cantilever:v1"),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };

    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("shell QA serialization failed: {error}");
            std::process::exit(1);
        }
    }

    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
