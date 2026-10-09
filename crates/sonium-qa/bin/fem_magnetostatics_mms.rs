//! QA: FEM magnetostatics — scalar-potential manufactured solution (MMS).
//!
//! Manufactured solution: φ(x,y) = sin(πx) sin(πy) on unit square.
//! Source: f = -∇²φ = 2π² sin(πx) sin(πy).
//! Dirichlet BC φ = 0 on all boundaries.

use sonium_fem::assembly::{assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_homogeneous_dirichlet};
use sonium_fem::forms::{LinearForm, LocalVector, WeightedLaplacianForm};
use sonium_fem::mesh::generators::unit_square_triangles;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::analytical::{mms_laplacian_sin_sin, mms_laplacian_sin_sin_source};
use sonium_solvers::{GmresConfig, gmres};

struct MMSSourceForm;

impl LinearForm<f64> for MMSSourceForm {
    fn integrate_point(
        &self,
        test: &sonium_fem::spaces::BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let x = test.physical_coords[0];
        let y = test.physical_coords[1];
        let f = mms_laplacian_sin_sin_source(x, y);
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            local_vector.add(i, f * test.value(i) * w);
        }
    }
}

fn test_magnetostatics_mms() {
    let mesh = unit_square_triangles(40);
    let space = H1Space::new(PolynomialDegree::P1);

    let form = WeightedLaplacianForm { coefficient: 1.0 };
    let source = MMSSourceForm;

    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 1);
    let rhs = assemble_linear(&source, &space, &mesh, 1);

    let _n_dofs = space.num_global_dofs(&mesh);
    let mut system = FemSystem::new(stiffness, rhs);
    // Unit square boundaries: marker depends on mesh generator
    // unit_square_triangles uses markers 1..4 for the four sides
    for marker in 1..=4 {
        apply_homogeneous_dirichlet(&mut system, &mesh, &space, marker);
    }

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
        "GMRES should converge for MMS Poisson problem"
    );

    let mut max_err: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let exact = mms_laplacian_sin_sin(node.x, node.y);
        let err = (result.x[i] - exact).abs();
        max_err = max_err.max(err);
    }

    println!("Magnetostatics MMS: max potential error = {:.4e}", max_err);
    assert!(max_err < 1e-2, "MMS error too large: {:.4e}", max_err);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_magnetostatics_mms();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
