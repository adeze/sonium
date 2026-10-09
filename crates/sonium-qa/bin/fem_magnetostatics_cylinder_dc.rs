//! QA: FEM magnetostatics — cylindrical conductor with DC current.
//!
//! Solves the 2D magnetic-vector-potential Poisson equation:
//!   -(1/μ) ∇² A_z = J_z   on a disk
//! with A_z = 0 on the outer boundary.
//!
//! Analytical B-field:
//!   B_θ(r) = μ·J·r/2          (r ≤ R)
//!   B_θ(r) = μ·J·R²/(2r)      (r ≥ R)

use sonium_fem::assembly::{assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_homogeneous_dirichlet};
use sonium_fem::forms::{LinearForm, LocalVector, WeightedLaplacianForm};
use sonium_fem::mesh::generators::circular_mesh_triangles;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{GmresConfig, gmres};

/// Uniform current density source: l(φ) = ∫ J_z φ dΩ.
struct CurrentDensityForm {
    jz: f64,
    conductor_radius: f64,
    cx: f64,
    cy: f64,
}

impl LinearForm<f64> for CurrentDensityForm {
    fn integrate_point(
        &self,
        test: &sonium_fem::spaces::BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let x = test.physical_coords[0];
        let y = test.physical_coords[1];
        let r = ((x - self.cx).powi(2) + (y - self.cy).powi(2)).sqrt();
        let j = if r <= self.conductor_radius {
            self.jz
        } else {
            0.0
        };
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            local_vector.add(i, j * test.value(i) * w);
        }
    }
}

fn test_magnetostatics_cylinder_dc() {
    let mu = 4.0 * std::f64::consts::PI * 1e-7; // Vacuum permeability
    let i = 10.0; // Current (A)
    let r_conductor = 0.05; // Conductor radius (m)
    let r_domain = 0.15; // Domain radius (m)

    let jz = i / (std::f64::consts::PI * r_conductor * r_conductor);

    let mesh = circular_mesh_triangles(0.0, 0.0, r_domain, 20, 60);
    let space = H1Space::new(PolynomialDegree::P1);

    let form = WeightedLaplacianForm {
        coefficient: 1.0 / mu,
    };
    let source = CurrentDensityForm {
        jz,
        conductor_radius: r_conductor,
        cx: 0.0,
        cy: 0.0,
    };

    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 1);
    let rhs = assemble_linear(&source, &space, &mesh, 1);

    let _n_dofs = space.num_global_dofs(&mesh);
    let mut system = FemSystem::new(stiffness, rhs);
    apply_homogeneous_dirichlet(&mut system, &mesh, &space, 1);

    let csr = system.matrix.to_csr();
    let result = gmres(
        &csr,
        &system.rhs,
        &GmresConfig {
            max_iterations: 2000,
            restart: 50,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(
        result.converged,
        "GMRES should converge for Poisson equation"
    );

    // Compute B-field at element centroids and compare to analytical
    let mut max_err: f64 = 0.0;
    for elem_idx in 0..mesh.num_elements() {
        let dofs = space.local_to_global(&mesh, elem_idx);
        let phi_local: Vec<f64> = dofs.iter().map(|&d| result.x[d]).collect();
        let tab = space.tabulate(&mesh, elem_idx, 1.0 / 3.0, 1.0 / 3.0, 0.0);

        // B = (∂A_z/∂y, -∂A_z/∂x)
        let mut daz_dx = 0.0;
        let mut daz_dy = 0.0;
        for (i, &phi_i) in phi_local.iter().enumerate() {
            daz_dx += phi_i * tab.grad(i, 0);
            daz_dy += phi_i * tab.grad(i, 1);
        }
        let bx = daz_dy;
        let by = -daz_dx;

        let px = tab.physical_coords[0];
        let py = tab.physical_coords[1];
        let r = (px * px + py * py).sqrt();

        // B_θ = B · e_θ = Bx*(-y/r) + By*(x/r)
        let b_theta = if r > 1e-12 {
            bx * (-py / r) + by * (px / r)
        } else {
            0.0
        };

        let b_exact = if r <= r_conductor {
            mu * jz * r / 2.0
        } else {
            mu * jz * r_conductor * r_conductor / (2.0 * r)
        };

        let err = (b_theta - b_exact).abs();
        max_err = max_err.max(err);
    }

    println!(
        "Magnetostatics cylinder DC: max B-field error = {:.4e} T",
        max_err
    );
    assert!(max_err < 5e-4, "B-field error too large: {:.4e}", max_err);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_magnetostatics_cylinder_dc();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
