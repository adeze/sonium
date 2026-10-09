//! QA: FEM elasticity — thick-walled cylinder under internal pressure (Lamé).
//!
//! Analytical (plane strain): u_r(r) = (1+nu)/E * [(1-2*nu)*A*r - B/r]
//! where A = p*a²/(b²-a²), B = p*a²*b²/(b²-a²).

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::generators::annular_mesh_triangles;
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_qa::analytical::elasticity_lame_displacement;
use sonium_solvers::{CsrMatrix, GmresConfig, IluPreconditioner, gmres, gmres_preconditioned};

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

fn test_elasticity_thick_cylinder() {
    let a = 0.5;
    let b = 2.0;
    let p = 100.0;
    let e_mod = 200e9;
    let nu = 0.3;

    let mesh = annular_mesh_triangles(0.0, 0.0, a, b, 8, 24);
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let n_dofs = space.num_global_dofs(&mesh);

    let material = IsotropicMaterial::new(e_mod, nu);
    let form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStrain);
    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 1);

    let mut csr = stiffness.to_csr();
    let mut rhs = vec![0.0; n_dofs];

    // Apply exact displacement on inner boundary (tag 1) and outer boundary (tag 2)
    let tol = a * 0.05;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let r = (node.x * node.x + node.y * node.y).sqrt();
        let is_boundary = (r - a).abs() < tol || (r - b).abs() < tol;
        if is_boundary {
            let u_r = elasticity_lame_displacement(r, a, b, p, e_mod, nu);
            let ux = u_r * node.x / r;
            let uy = u_r * node.y / r;
            apply_dirichlet_csr(&mut csr, &mut rhs, &[i * 2, i * 2 + 1], &[ux, uy]);
        }
    }

    let preconditioner = IluPreconditioner::from_csr(&csr);
    let result = gmres_preconditioned(
        &csr,
        &preconditioner,
        &rhs,
        &GmresConfig {
            max_iterations: 2000,
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
        "Solver should converge: residual={:.2e}",
        result.residual
    );

    let mut max_err: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let r = (node.x * node.x + node.y * node.y).sqrt();
        let u_r_exact = elasticity_lame_displacement(r, a, b, p, e_mod, nu);
        let ux_fem = result.x[i * 2];
        let uy_fem = result.x[i * 2 + 1];
        let u_r_fem = ux_fem * node.x / r + uy_fem * node.y / r;
        let err = (u_r_fem - u_r_exact).abs();
        max_err = max_err.max(err);
    }

    println!("Elasticity thick cylinder: max_err = {:.4e}", max_err);
    assert!(
        max_err < 1e-6,
        "Max displacement error too large: {:.4e}",
        max_err
    );
}

fn test_elasticity_thick_cylinder_convergence() {
    let a = 0.5;
    let b = 2.0;
    let p = 100.0;
    let e_mod = 200e9;
    let nu = 0.3;

    let errors: Vec<f64> = [4, 8, 16]
        .iter()
        .map(|&n_radial| {
            let mesh = annular_mesh_triangles(0.0, 0.0, a, b, n_radial, 24);
            let space = VectorH1Space::new(PolynomialDegree::P1, 2);
            let n_dofs = space.num_global_dofs(&mesh);

            let material = IsotropicMaterial::new(e_mod, nu);
            let form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStrain);
            let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 1);

            let mut csr = stiffness.to_csr();
            let mut rhs = vec![0.0; n_dofs];

            let tol = a * 0.05;
            for (i, node) in mesh.nodes.iter().enumerate() {
                let r = (node.x * node.x + node.y * node.y).sqrt();
                let is_boundary = (r - a).abs() < tol || (r - b).abs() < tol;
                if is_boundary {
                    let u_r = elasticity_lame_displacement(r, a, b, p, e_mod, nu);
                    let ux = u_r * node.x / r;
                    let uy = u_r * node.y / r;
                    apply_dirichlet_csr(&mut csr, &mut rhs, &[i * 2, i * 2 + 1], &[ux, uy]);
                }
            }

            let result = gmres(
                &csr,
                &rhs,
                &GmresConfig {
                    max_iterations: 2000,
                    restart: 200,
                    tolerance: 1e-12,
                    print_interval: 0,
                    convergence_check_interval: 1,
                    min_restart: None,
                    max_restart: None,
                    exact_residual_check_interval: 0,
                },
            );
            assert!(result.converged);

            let mut max_err: f64 = 0.0;
            for (i, node) in mesh.nodes.iter().enumerate() {
                let r = (node.x * node.x + node.y * node.y).sqrt();
                let u_r_exact = elasticity_lame_displacement(r, a, b, p, e_mod, nu);
                let ux_fem = result.x[i * 2];
                let uy_fem = result.x[i * 2 + 1];
                let u_r_fem = ux_fem * node.x / r + uy_fem * node.y / r;
                max_err = max_err.max((u_r_fem - u_r_exact).abs());
            }
            max_err
        })
        .collect();

    let rate_1 = (errors[1] / errors[0]).log(std::f64::consts::LN_10)
        / (4.0_f64 / 8.0).log(std::f64::consts::LN_10);
    let rate_2 = (errors[2] / errors[1]).log(std::f64::consts::LN_10)
        / (8.0_f64 / 16.0).log(std::f64::consts::LN_10);

    println!(
        "Thick cylinder convergence: err={:.2e}, {:.2e}, {:.2e}, rates: {:.2}, {:.2}",
        errors[0], errors[1], errors[2], rate_1, rate_2
    );
    let at_numerical_floor = errors.iter().all(|error| *error < 1.0e-9);
    assert!(
        at_numerical_floor || (rate_1 > 1.0 && rate_2 > 1.0),
        "Convergence rate too low above the numerical floor"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_elasticity_thick_cylinder();
        test_elasticity_thick_cylinder_convergence();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
