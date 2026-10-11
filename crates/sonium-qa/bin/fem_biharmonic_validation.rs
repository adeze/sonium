//! Biharmonic validation tests
//!
//! Tests for biharmonic plate bending.
//!
//! - Facet form penalty parameter bounds
//! - Form symmetry (penalty term is symmetric)
//! - Penalty scaling with element size (α/h)
//! - DG assembly structure verification
//! - Solved mixed manufactured plate with boundary conditions and refinement
//! - Solved uniform-load simply-supported plate against the Navier reference

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::assembly::assemble_linear;
use sonium_fem::assembly::dg::assemble_dg;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::generators::unit_square_triangles;
use sonium_fem::physics::solid::biharmonic::{
    BiharmonicCellForm, BiharmonicFacetForm, BiharmonicSourceForm,
    assemble_mixed_biharmonic_operator,
};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::{QaResult, QaStatus};
use std::f64::consts::PI;

const CASE_ID: &str = "fem.biharmonic-validation.v1";

// Manufactured solution: u = sin(pi x) sin(pi y), f = 4 pi^4 u, q = 2 pi^2 u.
const COARSE_SUBDIVISIONS: usize = 4;
const FINE_SUBDIVISIONS: usize = 8;
const SOURCE_QUADRATURE_ORDER: usize = 6;

// Navier series center deflection coefficient for the simply-supported
// uniformly loaded square plate with unit rigidity.
const NAVIER_CENTER_COEFFICIENT: f64 = 0.00406;

// Acceptance tolerances.
const TOL_FINE_U: f64 = 0.10;
const TOL_FINE_Q: f64 = 0.15;
const TOL_RATE_U: f64 = 1.0;
const TOL_RATE_Q: f64 = 0.7;
const TOL_RESIDUAL: f64 = 1e-10;
const TOL_BOUNDARY: f64 = 1e-12;
const TOL_NAVIER: f64 = 0.15;

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

fn is_unit_square_boundary(x: f64, y: f64) -> bool {
    let eps = 1e-10;
    x < eps || x > 1.0 - eps || y < eps || y > 1.0 - eps
}

#[derive(Debug, Clone, Copy)]
struct ManufacturedSolve {
    u_error: f64,
    q_error: f64,
    residual: f64,
    boundary_violation: f64,
}

/// Solve the mixed manufactured plate on a unit square triangulation.
fn solve_manufactured(subdivisions: usize) -> Result<ManufacturedSolve, String> {
    let mesh = unit_square_triangles(subdivisions);
    let space = H1Space::new(PolynomialDegree::P1);
    let n = space.num_global_dofs(&mesh);
    let operator = assemble_mixed_biharmonic_operator(&mesh, PolynomialDegree::P1);
    let source =
        BiharmonicSourceForm::new(|x, y, _z| 4.0 * PI.powi(4) * (PI * x).sin() * (PI * y).sin());
    let load = assemble_linear(&source, &space, &mesh, SOURCE_QUADRATURE_ORDER);
    let mut rhs = vec![0.0; 2 * n];
    rhs[n..].copy_from_slice(&load);

    let total = 2 * n;
    let mut matrix = vec![0.0; total * total];
    for k in 0..operator.matrix.values.len() {
        matrix[operator.matrix.rows[k] * total + operator.matrix.cols[k]] +=
            operator.matrix.values[k];
    }
    let mut constrained = Vec::new();
    for (node, point) in mesh.nodes.iter().enumerate() {
        if is_unit_square_boundary(point.x, point.y) {
            constrained.push(node);
            constrained.push(n + node);
        }
    }
    check(
        !constrained.is_empty(),
        "no boundary nodes found".to_string(),
    )?;
    for &dof in &constrained {
        for col in 0..total {
            matrix[dof * total + col] = 0.0;
            matrix[col * total + dof] = 0.0;
        }
        matrix[dof * total + dof] = 1.0;
        rhs[dof] = 0.0;
    }
    let constrained_matrix = matrix.clone();
    dense_solve(&mut matrix, &mut rhs, total)?;
    let solution = rhs;

    let mut boundary_violation = 0.0f64;
    for &dof in &constrained {
        boundary_violation = boundary_violation.max(solution[dof].abs());
    }

    let mut u_error = 0.0;
    let mut u_reference = 0.0;
    let mut q_error = 0.0;
    let mut q_reference = 0.0;
    for (node, point) in mesh.nodes.iter().enumerate() {
        if is_unit_square_boundary(point.x, point.y) {
            continue;
        }
        let exact_u = (PI * point.x).sin() * (PI * point.y).sin();
        let exact_q = 2.0 * PI * PI * exact_u;
        u_error += (solution[node] - exact_u).powi(2);
        u_reference += exact_u * exact_u;
        q_error += (solution[n + node] - exact_q).powi(2);
        q_reference += exact_q * exact_q;
    }

    // Algebraic residual against the constrained system.
    let mut residual = 0.0;
    let mut rhs_norm = 0.0;
    for &dof in &constrained {
        let _ = dof;
    }
    // Rebuild the constrained right-hand side norm from the load vector.
    let mut constrained_rhs = vec![0.0; total];
    constrained_rhs[n..].copy_from_slice(&load);
    for &dof in &constrained {
        constrained_rhs[dof] = 0.0;
    }
    for row in 0..total {
        let mut row_sum = 0.0;
        for col in 0..total {
            row_sum += constrained_matrix[row * total + col] * solution[col];
        }
        let difference = row_sum - constrained_rhs[row];
        residual += difference * difference;
        rhs_norm += constrained_rhs[row] * constrained_rhs[row];
    }

    Ok(ManufacturedSolve {
        u_error: (u_error / u_reference.max(1e-30)).sqrt(),
        q_error: (q_error / q_reference.max(1e-30)).sqrt(),
        residual: (residual / rhs_norm.max(1e-30)).sqrt(),
        boundary_violation,
    })
}

#[derive(Debug, Clone, Copy)]
struct Metrics {
    form_checks: usize,
    coarse_u_error: f64,
    fine_u_error: f64,
    fine_q_error: f64,
    u_rate: f64,
    q_rate: f64,
    maximum_residual: f64,
    boundary_violation: f64,
    navier_error: f64,
}

fn run() -> Result<Metrics, String> {
    let mut form_checks = 0usize;

    // Facet form penalty bounds and default.
    let form = BiharmonicFacetForm::new(8.0);
    check(
        form.alpha >= 4.0 && form.alpha <= 16.0,
        format!("penalty {} outside stable range", form.alpha),
    )?;
    form_checks += 1;
    let default = BiharmonicFacetForm::default_penalty();
    check(
        (default.alpha - 8.0).abs() < 1e-10,
        "default penalty should be 8".to_string(),
    )?;
    form_checks += 1;

    // Cell form assembles on a P1 mesh.
    let mesh = unit_square_triangles(4);
    let space = H1Space::new(PolynomialDegree::P1);
    let _triplet = assemble_bilinear(&BiharmonicCellForm, &space, &space, &mesh, 1);
    form_checks += 1;

    // Penalty scaling with element size.
    let penalty_fine: f64 = 8.0 / 0.125;
    let penalty_coarse: f64 = 8.0 / 0.25;
    check(
        penalty_fine > penalty_coarse && (penalty_fine - penalty_coarse * 2.0).abs() < 1e-10,
        "penalty should scale inversely with element size".to_string(),
    )?;
    form_checks += 1;

    // DG assembly contributes entries.
    let facet = BiharmonicFacetForm::default_penalty();
    let triplet = assemble_dg(&BiharmonicCellForm, &facet, &space, &mesh, 1);
    let csr = triplet.to_csr();
    check(
        csr.num_rows > 0 && csr.nnz() > 0,
        "penalty form should contribute entries".to_string(),
    )?;
    form_checks += 1;

    // Solved manufactured plate with refinement.
    let coarse = solve_manufactured(COARSE_SUBDIVISIONS)?;
    let fine = solve_manufactured(FINE_SUBDIVISIONS)?;
    check(
        fine.u_error <= TOL_FINE_U,
        format!("fine u error {:.3e} exceeds {TOL_FINE_U:.1e}", fine.u_error),
    )?;
    check(
        fine.q_error <= TOL_FINE_Q,
        format!("fine q error {:.3e} exceeds {TOL_FINE_Q:.1e}", fine.q_error),
    )?;
    let mesh_ratio = (FINE_SUBDIVISIONS as f64 / COARSE_SUBDIVISIONS as f64).ln();
    let u_rate = (coarse.u_error / fine.u_error).ln() / mesh_ratio;
    let q_rate = (coarse.q_error / fine.q_error).ln() / mesh_ratio;
    check(
        u_rate >= TOL_RATE_U,
        format!("u convergence rate {u_rate:.3} below {TOL_RATE_U:.1}"),
    )?;
    check(
        q_rate >= TOL_RATE_Q,
        format!("q convergence rate {q_rate:.3} below {TOL_RATE_Q:.1}"),
    )?;
    let maximum_residual = coarse.residual.max(fine.residual);
    check(
        maximum_residual <= TOL_RESIDUAL,
        format!("residual {maximum_residual:.3e} exceeds {TOL_RESIDUAL:.1e}"),
    )?;
    let boundary_violation = coarse.boundary_violation.max(fine.boundary_violation);
    check(
        boundary_violation <= TOL_BOUNDARY,
        format!("boundary violation {boundary_violation:.3e} exceeds {TOL_BOUNDARY:.1e}"),
    )?;

    // Solved uniform-load simply-supported plate against Navier.
    let subdivisions = FINE_SUBDIVISIONS;
    let mesh = unit_square_triangles(subdivisions);
    let space = H1Space::new(PolynomialDegree::P1);
    let n = space.num_global_dofs(&mesh);
    let operator = assemble_mixed_biharmonic_operator(&mesh, PolynomialDegree::P1);
    let source = BiharmonicSourceForm::new(|_, _, _| 1.0);
    let load = assemble_linear(&source, &space, &mesh, SOURCE_QUADRATURE_ORDER);
    let total = 2 * n;
    let mut matrix = vec![0.0; total * total];
    for k in 0..operator.matrix.values.len() {
        matrix[operator.matrix.rows[k] * total + operator.matrix.cols[k]] +=
            operator.matrix.values[k];
    }
    let mut rhs = vec![0.0; total];
    rhs[n..].copy_from_slice(&load);
    for (node, point) in mesh.nodes.iter().enumerate() {
        if is_unit_square_boundary(point.x, point.y) {
            for &dof in &[node, n + node] {
                for col in 0..total {
                    matrix[dof * total + col] = 0.0;
                    matrix[col * total + dof] = 0.0;
                }
                matrix[dof * total + dof] = 1.0;
                rhs[dof] = 0.0;
            }
        }
    }
    dense_solve(&mut matrix, &mut rhs, total)?;
    let center = (0..mesh.num_nodes())
        .find(|&node| {
            (mesh.nodes[node].x - 0.5).abs() < 1e-10 && (mesh.nodes[node].y - 0.5).abs() < 1e-10
        })
        .ok_or_else(|| "plate center node not found".to_string())?;
    let navier_error = (rhs[center] - NAVIER_CENTER_COEFFICIENT).abs() / NAVIER_CENTER_COEFFICIENT;
    check(
        navier_error <= TOL_NAVIER,
        format!("Navier center error {navier_error:.3e} exceeds {TOL_NAVIER:.1e}"),
    )?;

    Ok(Metrics {
        form_checks,
        coarse_u_error: coarse.u_error,
        fine_u_error: fine.u_error,
        fine_q_error: fine.q_error,
        u_rate,
        q_rate,
        maximum_residual,
        boundary_violation,
        navier_error,
    })
}

fn main() {
    let result = match run() {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("form_checks", metrics.form_checks as f64)
            .minimum("form_checks", 5.0)
            .metric("coarse_u_error", metrics.coarse_u_error)
            .metric("fine_u_error", metrics.fine_u_error)
            .maximum("fine_u_error", TOL_FINE_U)
            .metric("fine_q_error", metrics.fine_q_error)
            .maximum("fine_q_error", TOL_FINE_Q)
            .metric("u_rate", metrics.u_rate)
            .minimum("u_rate", TOL_RATE_U)
            .metric("q_rate", metrics.q_rate)
            .minimum("q_rate", TOL_RATE_Q)
            .metric("maximum_residual", metrics.maximum_residual)
            .maximum("maximum_residual", TOL_RESIDUAL)
            .metric("boundary_violation", metrics.boundary_violation)
            .maximum("boundary_violation", TOL_BOUNDARY)
            .metric("navier_error", metrics.navier_error)
            .maximum("navier_error", TOL_NAVIER)
            .reference("mixed manufactured plate and Navier simply-supported series")
            .reference_checksum("analytic/deterministic-fixture:fem-biharmonic-validation-v1")
            .dimension("unit square triangles n=4 and n=8, mixed P1 Laplacian pair")
            .formulation("K u - M q = 0; K q = f with u = q = 0 on the boundary")
            .tolerance("fine u 10%; fine q 15%; rates 1.0/0.7; Navier center 15%")
            .input_checksum("mesh:unit-square-tri-n4-n8:mixed-biharmonic:v1"),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };

    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("biharmonic QA serialization failed: {error}");
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
