//! QA: FEM convection-diffusion — 1D steady with SUPG stabilization.
//!
//! Validates stabilized FEM against the exact exponential solution.
//!
//! Equation: a·du/dx − k·d²u/dx² = 0,  u(0)=0, u(L)=1
//! Exact:    u(x) = (1 − e^{Pe·x/L}) / (1 − e^{Pe}),  Pe = a·L/k
//!
//! For convection-dominated flow (Pe ≫ 1), Galerkin FEM produces
//! node-to-node oscillations. SUPG with optimal τ gives the exact
//! nodal solution on a uniform mesh.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::forms::{BilinearForm, LocalMatrix};
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, H1Space};
use sonium_solvers::{GmresConfig, gmres};

/// SUPG-stabilized convection-diffusion bilinear form.
///
/// Weak form (including SUPG for linear elements):
///   ∫ [k·∇v·∇u + v·(a·∇u) + τ·(a·∇v)·(a·∇u)] dΩ
///
/// For linear elements the diffusion residual ∇²u vanishes, so the
/// SUPG term reduces to τ·∫ (a·∇v)·(a·∇u) dΩ.
#[derive(Debug, Clone, Copy)]
struct ConvectionDiffusionSupgForm {
    diffusivity: f64,
    velocity: [f64; 2],
    tau: f64,
}

impl BilinearForm<f64> for ConvectionDiffusionSupgForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j;

        for i in 0..n {
            for j in 0..n {
                let mut val = 0.0;

                // Diffusion: k * ∇v · ∇u
                for d in 0..dim {
                    val += self.diffusivity * test.grad(i, d) * trial.grad(j, d);
                }

                // Convection: v * (a · ∇u)
                let mut conv = 0.0;
                for d in 0..dim.min(2) {
                    conv += self.velocity[d] * trial.grad(j, d);
                }
                val += test.value(i) * conv;

                // SUPG stabilization: τ * (a · ∇v) * (a · ∇u)
                let mut supg_test = 0.0;
                let mut supg_trial = 0.0;
                for d in 0..dim.min(2) {
                    supg_test += self.velocity[d] * test.grad(i, d);
                    supg_trial += self.velocity[d] * trial.grad(j, d);
                }
                val += self.tau * supg_test * supg_trial;

                local_matrix.add(i, j, val * w);
            }
        }
    }
}

/// Pure Galerkin convection-diffusion (no stabilization).
#[derive(Debug, Clone, Copy)]
struct ConvectionDiffusionGalerkinForm {
    diffusivity: f64,
    velocity: [f64; 2],
}

impl BilinearForm<f64> for ConvectionDiffusionGalerkinForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j;

        for i in 0..n {
            for j in 0..n {
                let mut val = 0.0;

                for d in 0..dim {
                    val += self.diffusivity * test.grad(i, d) * trial.grad(j, d);
                }

                let mut conv = 0.0;
                for d in 0..dim.min(2) {
                    conv += self.velocity[d] * trial.grad(j, d);
                }
                val += test.value(i) * conv;

                local_matrix.add(i, j, val * w);
            }
        }
    }
}

fn exact_solution(x: f64, l: f64, pe: f64) -> f64 {
    if pe.abs() < 1e-10 {
        return x / l;
    }
    (1.0 - (pe * x / l).exp()) / (1.0 - pe.exp())
}

fn solve_and_error(l: f64, a: f64, k: f64, nx: usize, use_supg: bool) -> (f64, f64) {
    let ny = 2;
    let width = 0.01 * l;
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, width, nx, ny);

    let space = H1Space::new(PolynomialDegree::P1);
    let n_dofs = space.num_global_dofs(&mesh);

    let h = l / nx as f64;
    let pe_elem = a * h / (2.0 * k);

    let tau = if use_supg {
        // Optimal 1D SUPG parameter
        let coth_pe = if pe_elem.abs() > 20.0 {
            pe_elem.signum()
        } else {
            pe_elem.cosh() / pe_elem.sinh()
        };
        (h / (2.0 * a.abs())) * (coth_pe - 1.0 / pe_elem)
    } else {
        0.0
    };

    let matrix = if use_supg {
        sonium_fem::assembly::assemble_bilinear(
            &ConvectionDiffusionSupgForm {
                diffusivity: k,
                velocity: [a, 0.0],
                tau,
            },
            &space,
            &space,
            &mesh,
            1,
        )
    } else {
        sonium_fem::assembly::assemble_bilinear(
            &ConvectionDiffusionGalerkinForm {
                diffusivity: k,
                velocity: [a, 0.0],
            },
            &space,
            &space,
            &mesh,
            1,
        )
    };

    let mut system = FemSystem::new(matrix, vec![0.0; n_dofs]);

    // u(0) = 0
    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| 0.0, None);
    // u(L) = 1
    apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| 1.0, None);

    let csr = system.matrix.to_csr();

    let result = gmres(
        &csr,
        &system.rhs,
        &GmresConfig {
            max_iterations: 1000,
            restart: 50,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(result.converged, "GMRES should converge");

    let pe = a * l / k;

    // Max error over all nodes (focus on x-variation)
    let mut max_err = 0.0f64;
    let mut l2_err = 0.0f64;
    let mut l2_norm = 0.0f64;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let u_exact = exact_solution(node.x, l, pe);
        let u_fem = result.x[i];
        let diff = (u_fem - u_exact).abs();
        max_err = max_err.max(diff);
        l2_err += diff * diff;
        l2_norm += u_exact * u_exact;
    }

    let rel_l2 = if l2_norm > 0.0 {
        (l2_err / l2_norm).sqrt()
    } else {
        0.0
    };

    (max_err, rel_l2)
}

fn test_convection_diffusion_diffusion_dominated() {
    let l = 1.0;
    let a = 1.0;
    let k = 2.0; // Pe = 0.5

    println!("Convection-diffusion (Pe = {:.1}):", a * l / k);

    let (max_err_gal, rel_l2_gal) = solve_and_error(l, a, k, 20, false);
    println!(
        "  Galerkin:   max_err = {:.4e}, rel_L2 = {:.4e}",
        max_err_gal, rel_l2_gal
    );

    let (max_err_supg, rel_l2_supg) = solve_and_error(l, a, k, 20, true);
    println!(
        "  SUPG:       max_err = {:.4e}, rel_L2 = {:.4e}",
        max_err_supg, rel_l2_supg
    );

    // For diffusion-dominated, both should be accurate
    assert!(
        max_err_gal < 0.01,
        "Galerkin error too large: {:.4e}",
        max_err_gal
    );
    assert!(
        max_err_supg < 0.01,
        "SUPG error too large: {:.4e}",
        max_err_supg
    );
}

fn test_convection_diffusion_convection_dominated() {
    let l = 1.0;
    let a = 1.0;
    let k = 0.005; // Pe = 200

    println!("Convection-diffusion (Pe = {:.0}):", a * l / k);

    let (max_err_gal, rel_l2_gal) = solve_and_error(l, a, k, 40, false);
    println!(
        "  Galerkin:   max_err = {:.4e}, rel_L2 = {:.4e}",
        max_err_gal, rel_l2_gal
    );

    let (max_err_supg, rel_l2_supg) = solve_and_error(l, a, k, 40, true);
    println!(
        "  SUPG:       max_err = {:.4e}, rel_L2 = {:.4e}",
        max_err_supg, rel_l2_supg
    );

    // Galerkin should oscillate badly
    assert!(
        max_err_gal > 0.1,
        "Galerkin should show oscillations (err={:.4e})",
        max_err_gal
    );

    // SUPG should be much better (near-exact at nodes for 1D)
    assert!(
        max_err_supg < 0.02,
        "SUPG error too large: {:.4e}",
        max_err_supg
    );
    assert!(
        max_err_supg < max_err_gal / 10.0,
        "SUPG should beat Galerkin by >10x"
    );
}

fn test_convection_diffusion_supg_vs_galerkin_refinement() {
    let l = 1.0;
    let a = 1.0;
    let k = 0.02; // Pe = 50

    println!(
        "Convection-diffusion SUPG vs Galerkin refinement (Pe = {:.0}):",
        a * l / k
    );

    let mesh_sizes = [10, 20, 40, 80];

    for &nx in &mesh_sizes {
        let (max_err_gal, _) = solve_and_error(l, a, k, nx, false);
        let (max_err_supg, _) = solve_and_error(l, a, k, nx, true);
        println!(
            "  nx={}: Galerkin err = {:.4e}, SUPG err = {:.4e}, ratio = {:.1}x",
            nx,
            max_err_gal,
            max_err_supg,
            max_err_gal / max_err_supg.max(1e-12)
        );
        // SUPG should be consistently better than Galerkin
        assert!(
            max_err_supg < max_err_gal,
            "SUPG should beat Galerkin at nx={}",
            nx
        );
    }
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_convection_diffusion_diffusion_dominated();
        test_convection_diffusion_convection_dominated();
        test_convection_diffusion_supg_vs_galerkin_refinement();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
