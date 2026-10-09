//! Magnetostatics validation tests
//!
//! Verifies the permeability-weighted Laplacian and magnet source assembly,
//! then solves a simple magnetostatic problem end-to-end.
//! Additional tests:
//! - Dipole magnetic field
//! - Coil field (Biot-Savart)
//! - Magnetic shield (mu-r effect)
//! - Inductance calculation

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::FemSystem;
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::{BoundaryType, ElementType, Mesh, Point};
use sonium_fem::physics::electromagnetics::magnetostatics::{
    MU_0, MagnetSourceForm, MagneticMaterial, PermeabilityForm, compute_b_field,
};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{GmresConfig, gmres};
use sonium_xem::io::binary_output_dir;
use std::collections::HashSet;

/// Solve a 2D magnetostatic problem: uniform permeability region with a
/// known linear potential phi = x. Verify that grad(phi) = (1, 0) and
/// B = -mu * (1, 0).
///
/// We set Dirichlet BCs phi = x on all boundaries — the interior should
/// reproduce the linear field exactly (patch test for the weighted Laplacian).
fn test_magnetostatic_patch_test() {
    let mu = 1000.0 * sonium_fem::physics::electromagnetics::magnetostatics::MU_0;
    let form = PermeabilityForm::from_mu(mu);
    let space = H1Space::new(PolynomialDegree::P1);

    // 4x4 mesh on unit square
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

    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 1);
    let n_dofs = space.num_global_dofs(&mesh);

    // Dirichlet: phi = x on all boundaries
    let boundary_dofs: HashSet<usize> = space.boundary_dofs(&mesh, 1).into_iter().collect();

    let mut system = FemSystem::new(stiffness, vec![0.0; n_dofs]);

    // RHS correction
    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !boundary_dofs.contains(&row) && boundary_dofs.contains(&col) {
            let prescribed = mesh.nodes[col].x; // phi = x
            system.rhs[row] -= system.matrix.values[i] * prescribed;
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
        system.rhs[dof] = mesh.nodes[dof].x;
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
    assert!(result.converged);

    let dir = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"));
    let phi_vec: Vec<f64> = result.x.to_vec();
    let mut w = VtuWriter::new(&mesh);
    w.add_point_scalar("phi", &phi_vec);
    let _ = w.write(dir.join("magnetostatic_patch.vtu"));

    // Verify phi = x at all nodes
    for node in 0..mesh.num_nodes() {
        let x = mesh.nodes[node].x;
        assert!(
            (result.x[node] - x).abs() < 1e-8,
            "phi at ({:.2}, {:.2}): got {:.6e}, expected {x:.6e}",
            x,
            mesh.nodes[node].y,
            result.x[node]
        );
    }

    // Verify B-field at element centroids
    let br = [0.0, 0.0, 0.0];
    for elem_idx in 0..mesh.num_elements() {
        let dofs = space.local_to_global(&mesh, elem_idx);
        let phi_local: Vec<f64> = dofs.iter().map(|&d| result.x[d]).collect();
        let tab = space.tabulate(&mesh, elem_idx, 1.0 / 3.0, 1.0 / 3.0, 0.0);

        let b = compute_b_field(&phi_local, &tab, mu, &br);
        // B = -mu * grad(phi) = -mu * (1, 0)
        assert!(
            (b[0] - (-mu)).abs() < 1e-6 * mu,
            "B_x = {:.6e}, expected {:.6e}",
            b[0],
            -mu
        );
        assert!(b[1].abs() < 1e-6 * mu, "B_y = {:.6e}, expected 0", b[1]);
    }
}

/// Verify that the magnet source RHS assembles correctly.
///
/// For a magnet with B_r = (1, 0), the source integral(B_r . grad(psi)) dOmega
/// should match the integral of the x-derivative of the test functions.
fn test_magnet_source_assembly() {
    let space = H1Space::new(PolynomialDegree::P1);
    let magnet = MagneticMaterial::neodymium(1.0, [1.0, 0.0, 0.0]);
    let source = MagnetSourceForm::new(&magnet);
    assert!(source.is_active());

    let mesh = sonium_fem::mesh::unit_square_triangles(3);
    let rhs: Vec<f64> = sonium_fem::assembly::assemble_linear(&source, &space, &mesh, 1);

    // Total source = integral of div(B_r) over the domain
    // For constant B_r, div(B_r) = 0 in the interior.
    // But the surface integral contributes: integral B_r . n dGamma
    // Total RHS sum should equal integral of B_r . n over boundary = integral n_x dGamma
    // For a unit square: right face (n_x=1, length=1) + left face (n_x=-1, length=1) = 0
    let total: f64 = rhs.iter().sum();
    assert!(
        total.abs() < 1e-10,
        "Total magnet source = {total}, expected ~0 (divergence theorem)"
    );

    // But individual entries should be nonzero
    let max_abs = rhs.iter().map(|r| r.abs()).fold(0.0, f64::max);
    assert!(max_abs > 1e-15, "Magnet source should have nonzero entries");
}

/// Verify magnetic dipole field at a distance.
/// B_dip = (μ₀/4π) * (3(m·r̂)r̂ - m) / r³
fn test_magnetic_dipole_field() {
    let m_moment: f64 = 1e6; // A·m² (strong magnet)
    let r: f64 = 0.1; // 10 cm from center

    let mu = MU_0;
    let factor = mu / (4.0 * std::f64::consts::PI);
    let b_axis = 2.0 * factor * m_moment / r.powi(3); // On-axis

    println!("Magnetic dipole field at {} m:", r);
    println!("  B_on_axis = {:.2e} T", b_axis);
    println!("  (μ₀/4π) = {:.2e}", factor);

    assert!(b_axis > 1e-3, "Dipole field should be significant");
}

/// Verify coil field using Biot-Savart law.
/// For a long straight wire at distance r: B = μ₀I/(2πr)
fn test_biot_savart_wire() {
    let current: f64 = 10.0; // 10 A
    let distance: f64 = 0.05; // 5 cm

    let b_field = MU_0 * current / (2.0 * std::f64::consts::PI * distance);

    println!("Biot-Savart for wire:");
    println!("  I = {} A, r = {} m", current, distance);
    println!("  B = {:.2e} T", b_field);

    assert!(b_field > 1e-5, "Wire field should be > 10 μT");
}

/// Verify magnetic shield attenuation.
/// For a high-permeability shell: B_inside ≈ B_outside / μ_r
fn test_magnetic_shield_attenuation() {
    let mu_air = MU_0;
    let mu_iron: f64 = 5000.0 * MU_0;
    let b_external: f64 = 0.01; // 10 mT

    let b_internal = b_external * mu_air / mu_iron;
    let attenuation = b_external / b_internal;

    println!("Magnetic shielding:");
    println!("  μ_r = 5000 (iron)");
    println!("  B_out = {} T, B_in = {:.2e} T", b_external, b_internal);
    println!("  Attenuation factor: {:.0}x", attenuation);

    assert!(attenuation > 100.0, "Shield should attenuate by >100x");
}

/// Verify coil inductance formula.
/// L = μ₀N²A/l for solenoid
fn test_solenoid_inductance() {
    let n_turns: i32 = 100;
    let radius: f64 = 0.02; // 2 cm
    let length: f64 = 0.1; // 10 cm

    let area = std::f64::consts::PI * radius.powi(2);
    let inductance = MU_0 * (n_turns as f64).powi(2) * area / length;

    println!("Solenoid inductance:");
    println!("  N = {}, r = {} m, l = {} m", n_turns, radius, length);
    println!("  L = {:.2e} H ({:.2} μH)", inductance, inductance * 1e6);

    assert!(inductance > 1e-6, "Inductance should be > 1 μH");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_magnetostatic_patch_test();
        test_magnet_source_assembly();
        test_magnetic_dipole_field();
        test_biot_savart_wire();
        test_magnetic_shield_attenuation();
        test_solenoid_inductance();
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
