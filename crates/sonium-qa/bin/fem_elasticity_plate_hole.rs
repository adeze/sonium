//! QA: FEM elasticity — infinite plate with circular hole under uniform tension.
//!
//! Analytical (Kirsch): sigma_theta at r=a, theta=pi/2 equals 3*sigma0.
//! We apply far-field displacement BCs on the outer boundary and check
//! the stress concentration factor K_t on the hole boundary.

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::generators::annular_mesh_triangles;
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::{CsrMatrix, GmresConfig, IluPreconditioner, gmres_preconditioned};

fn apply_dirichlet_csr(
    csr: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    prescribed_dofs: &[usize],
    prescribed_values: &[f64],
) {
    for (&dof, &value) in prescribed_dofs.iter().zip(prescribed_values) {
        for row in 0..csr.num_rows {
            if row == dof {
                continue;
            }
            for index in csr.row_ptrs[row]..csr.row_ptrs[row + 1] {
                if csr.col_indices[index] == dof {
                    rhs[row] -= csr.values[index] * value;
                    csr.values[index] = 0.0;
                }
            }
        }
        for index in csr.row_ptrs[dof]..csr.row_ptrs[dof + 1] {
            csr.values[index] = if csr.col_indices[index] == dof {
                1.0
            } else {
                0.0
            };
        }
        rhs[dof] = value;
    }
}

/// Compute sigma_xx for a P1 triangle given nodal displacements.
fn element_stress_xx(
    nodes: &[sonium_fem::mesh::Point],
    u: &[f64], // [u1x, u1y, u2x, u2y, u3x, u3y]
    e_mod: f64,
    nu: f64,
) -> f64 {
    let x1 = nodes[0].x;
    let y1 = nodes[0].y;
    let x2 = nodes[1].x;
    let y2 = nodes[1].y;
    let x3 = nodes[2].x;
    let y3 = nodes[2].y;

    let area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)).abs();
    if area < 1e-30 {
        return 0.0;
    }

    let inv_2a = 1.0 / (2.0 * area);

    // B-matrix (3 x 6) for plane stress
    let b11 = (y2 - y3) * inv_2a;
    let _b12 = 0.0;
    let b13 = (y3 - y1) * inv_2a;
    let _b14 = 0.0;
    let b15 = (y1 - y2) * inv_2a;
    let _b16 = 0.0;

    let _b21 = 0.0;
    let b22 = (x3 - x2) * inv_2a;
    let _b23 = 0.0;
    let b24 = (x1 - x3) * inv_2a;
    let _b25 = 0.0;
    let b26 = (x2 - x1) * inv_2a;

    let b31 = (x3 - x2) * inv_2a;
    let b32 = (y2 - y3) * inv_2a;
    let b33 = (x1 - x3) * inv_2a;
    let b34 = (y3 - y1) * inv_2a;
    let b35 = (x2 - x1) * inv_2a;
    let b36 = (y1 - y2) * inv_2a;

    let ex = b11 * u[0] + b13 * u[2] + b15 * u[4];
    let ey = b22 * u[1] + b24 * u[3] + b26 * u[5];
    let _gxy = b31 * u[0] + b32 * u[1] + b33 * u[2] + b34 * u[3] + b35 * u[4] + b36 * u[5];

    let factor = e_mod / (1.0 - nu * nu);
    factor * (ex + nu * ey)
}

fn test_elasticity_plate_hole() {
    let a = 0.5; // hole radius
    let b = 5.0; // outer radius (large enough to approximate infinity)
    let sigma0 = 1.0e6; // applied tension at infinity
    let e_mod = 200e9;
    let nu = 0.3;

    let mesh = annular_mesh_triangles(0.0, 0.0, a, b, 12, 40);
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let n_dofs = space.num_global_dofs(&mesh);

    let material = IsotropicMaterial::new(e_mod, nu);
    let form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStress);
    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 1);

    let mut csr = stiffness.to_csr();
    let mut rhs = vec![0.0; n_dofs];

    // Outer boundary (marker 2): apply far-field displacement
    // For plane stress uniaxial tension in x: u_x = sigma0*x/E, u_y = -nu*sigma0*y/E
    let tol = b * 0.02;
    let mut prescribed = Vec::new();
    let mut prescribed_vals = Vec::new();
    for (i, node) in mesh.nodes.iter().enumerate() {
        let r = (node.x * node.x + node.y * node.y).sqrt();
        if (r - b).abs() < tol {
            let ux = sigma0 * node.x / e_mod;
            let uy = -nu * sigma0 * node.y / e_mod;
            prescribed.push(i * 2);
            prescribed.push(i * 2 + 1);
            prescribed_vals.push(ux);
            prescribed_vals.push(uy);
        }
    }
    apply_dirichlet_csr(&mut csr, &mut rhs, &prescribed, &prescribed_vals);

    // Inner boundary (marker 1) is traction-free — no action needed

    let precond = IluPreconditioner::from_csr(&csr);
    let result = gmres_preconditioned(
        &csr,
        &precond,
        &rhs,
        &GmresConfig {
            max_iterations: 5000,
            restart: 200,
            tolerance: 1e-8,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(
        result.converged,
        "GMRES+ILU should converge for plate-hole: residual={:.3e}, status={:?}",
        result.residual, result.status
    );

    // Recover stress concentration factor K_t = sigma_xx / sigma0 near theta=PI/2, r=a
    let mut max_sigma_xx: f64 = 0.0;
    for elem in mesh.elements.iter() {
        let nodes_idx = &elem.nodes;
        if nodes_idx.len() != 3 {
            continue;
        }

        // Check if this element is near the hole boundary at theta≈PI/2
        let mut near_hole = false;
        for &ni in nodes_idx {
            let node = &mesh.nodes[ni];
            let r = (node.x * node.x + node.y * node.y).sqrt();
            let theta = node.y.atan2(node.x);
            if r < a * 1.3 && (theta - std::f64::consts::PI / 2.0).abs() < 0.3 {
                near_hole = true;
            }
        }
        if !near_hole {
            continue;
        }

        let node_pts: Vec<_> = nodes_idx.iter().map(|&i| mesh.nodes[i]).collect();
        let u_elem: Vec<f64> = nodes_idx
            .iter()
            .flat_map(|&i| vec![result.x[i * 2], result.x[i * 2 + 1]])
            .collect();

        let sxx = element_stress_xx(&node_pts, &u_elem, e_mod, nu);
        max_sigma_xx = max_sigma_xx.max(sxx.abs());
    }

    let kt = max_sigma_xx / sigma0;
    println!("Elasticity plate-hole: K_t = {:.4} (exact = 3.0)", kt);
    assert!(
        kt > 2.0 && kt < 4.5,
        "Stress concentration factor out of range: {:.4}",
        kt
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_elasticity_plate_hole();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
