//! Elasticity validation tests
//!
//! Patch test: a linear displacement field produces constant strain.
//! The FEM solution must reproduce it exactly (to machine precision) with P1 elements.

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::FemSystem;
use sonium_fem::mesh::{BoundaryType, ElementType, Mesh, Point};
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::{GmresConfig, gmres};

/// Patch test: prescribe u_x = x, u_y = 0 on all boundaries, verify interior
/// nodes reproduce the exact linear field after solving.
///
/// This produces constant strain eps_xx = 1, all other strains = 0.
/// Any correct P1 FEM implementation MUST pass this test exactly.
fn test_patch_test_constant_strain() {
    let material = IsotropicMaterial::new(1.0, 0.3);

    // Build a 5x5 structured mesh with interior nodes
    let n = 5;
    let mut mesh = Mesh::new(2);
    for j in 0..n {
        for i in 0..n {
            mesh.add_node(Point::new_2d(
                i as f64 / (n - 1) as f64,
                j as f64 / (n - 1) as f64,
            ));
        }
    }
    for j in 0..(n - 1) {
        for i in 0..(n - 1) {
            let v0 = j * n + i;
            let v1 = v0 + 1;
            let v2 = v0 + n;
            let v3 = v2 + 1;
            mesh.add_element(ElementType::Triangle, vec![v0, v1, v2]);
            mesh.add_element(ElementType::Triangle, vec![v1, v3, v2]);
        }
    }
    mesh.detect_boundaries();
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 1, |_| true);

    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStrain);

    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 1);
    let n_dofs = space.num_global_dofs(&mesh);
    let rhs = vec![0.0f64; n_dofs];

    // Exact solution: u_x = x, u_y = 0
    let exact_u = |node: usize, comp: usize| -> f64 {
        match comp {
            0 => mesh.nodes[node].x,
            1 => 0.0,
            _ => unreachable!(),
        }
    };

    // Find boundary DOFs
    let boundary_dofs: std::collections::HashSet<usize> =
        space.boundary_dofs(&mesh, 1).into_iter().collect();

    let mut system = FemSystem::new(stiffness, rhs);

    // RHS correction for constrained columns
    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !boundary_dofs.contains(&row) && boundary_dofs.contains(&col) {
            let prescribed = exact_u(col / 2, col % 2);
            system.rhs[row] -= system.matrix.values[i] * prescribed;
        }
    }

    // Filter constrained rows/columns, add diagonal identity
    let mut new_rows = Vec::new();
    let mut new_cols = Vec::new();
    let mut new_vals = Vec::new();
    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !boundary_dofs.contains(&row) && !boundary_dofs.contains(&col) {
            new_rows.push(row);
            new_cols.push(col);
            new_vals.push(system.matrix.values[i]);
        }
    }
    for &dof in &boundary_dofs {
        new_rows.push(dof);
        new_cols.push(dof);
        new_vals.push(1.0);
        system.rhs[dof] = exact_u(dof / 2, dof % 2);
    }
    system.matrix.rows = new_rows;
    system.matrix.cols = new_cols;
    system.matrix.values = new_vals;

    // Solve
    let csr = system.matrix.to_csr();
    let result = gmres(
        &csr,
        &system.rhs,
        &GmresConfig {
            max_iterations: 500,
            tolerance: 1e-12,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        },
    );
    assert!(
        result.converged,
        "GMRES did not converge: {:.2e}",
        result.residual
    );

    // Verify all DOFs match the exact linear field
    for node in 0..mesh.num_nodes() {
        let x = mesh.nodes[node].x;
        let y = mesh.nodes[node].y;
        let ux = result.x[node * 2];
        let uy = result.x[node * 2 + 1];

        assert!(
            (ux - x).abs() < 1e-8,
            "u_x at ({x:.2}, {y:.2}): got {ux:.6e}, expected {x:.6e}"
        );
        assert!(
            uy.abs() < 1e-8,
            "u_y at ({x:.2}, {y:.2}): got {uy:.6e}, expected 0"
        );
    }
}

/// Patch test with shear: u_x = y, u_y = x (constant shear strain gamma_xy = 2).
fn test_patch_test_pure_shear() {
    let material = IsotropicMaterial::new(1.0, 0.3);

    let n = 4;
    let mut mesh = Mesh::new(2);
    for j in 0..n {
        for i in 0..n {
            mesh.add_node(Point::new_2d(
                i as f64 / (n - 1) as f64,
                j as f64 / (n - 1) as f64,
            ));
        }
    }
    for j in 0..(n - 1) {
        for i in 0..(n - 1) {
            let v0 = j * n + i;
            let v1 = v0 + 1;
            let v2 = v0 + n;
            let v3 = v2 + 1;
            mesh.add_element(ElementType::Triangle, vec![v0, v1, v2]);
            mesh.add_element(ElementType::Triangle, vec![v1, v3, v2]);
        }
    }
    mesh.detect_boundaries();
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 1, |_| true);

    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStrain);

    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 1);
    let n_dofs = space.num_global_dofs(&mesh);

    // Exact solution: u_x = y, u_y = x
    let exact_u = |node: usize, comp: usize| -> f64 {
        match comp {
            0 => mesh.nodes[node].y, // u_x = y
            1 => mesh.nodes[node].x, // u_y = x
            _ => unreachable!(),
        }
    };

    let boundary_dofs: std::collections::HashSet<usize> =
        space.boundary_dofs(&mesh, 1).into_iter().collect();

    let mut system = FemSystem::new(stiffness, vec![0.0; n_dofs]);

    // RHS correction
    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !boundary_dofs.contains(&row) && boundary_dofs.contains(&col) {
            system.rhs[row] -= system.matrix.values[i] * exact_u(col / 2, col % 2);
        }
    }

    // Filter + diagonal
    let mut new_rows = Vec::new();
    let mut new_cols = Vec::new();
    let mut new_vals = Vec::new();
    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !boundary_dofs.contains(&row) && !boundary_dofs.contains(&col) {
            new_rows.push(row);
            new_cols.push(col);
            new_vals.push(system.matrix.values[i]);
        }
    }
    for &dof in &boundary_dofs {
        new_rows.push(dof);
        new_cols.push(dof);
        new_vals.push(1.0);
        system.rhs[dof] = exact_u(dof / 2, dof % 2);
    }
    system.matrix.rows = new_rows;
    system.matrix.cols = new_cols;
    system.matrix.values = new_vals;

    let csr = system.matrix.to_csr();
    let result = gmres(
        &csr,
        &system.rhs,
        &GmresConfig {
            max_iterations: 500,
            tolerance: 1e-12,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        },
    );
    assert!(result.converged);

    for node in 0..mesh.num_nodes() {
        let x = mesh.nodes[node].x;
        let y = mesh.nodes[node].y;
        let ux = result.x[node * 2];
        let uy = result.x[node * 2 + 1];

        assert!(
            (ux - y).abs() < 1e-8,
            "u_x at ({x:.2}, {y:.2}): got {ux:.6e}, expected {y:.6e}"
        );
        assert!(
            (uy - x).abs() < 1e-8,
            "u_y at ({x:.2}, {y:.2}): got {uy:.6e}, expected {x:.6e}"
        );
    }
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_patch_test_constant_strain();
        test_patch_test_pure_shear();
    });
    match result {
        Ok(()) => {
            println!("SUCCESS");
        }
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
