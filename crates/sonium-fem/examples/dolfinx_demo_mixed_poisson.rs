//! Mixed Poisson equation — Raviart-Thomas + DG formulation
//!
//! Translated from DOLFINx cpp/demo/mixed_poisson/main.cpp
//!
//! Solves the mixed formulation of the Poisson equation:
//!   σ + ∇u = 0   (flux equation)
//!   div(σ) = f    (conservation)
//!
//! Domain: Ω = [0,1]²
//! Source: f(x,y) = sin(5x) + 1
//!
//! Function spaces:
//!   σ ∈ RT0 (Raviart-Thomas, H(div)-conforming)  — normal continuity
//!   u ∈ DG P1 (piecewise linear, L²)             — no continuity
//!
//! Weak form:
//!   a((σ,u),(τ,v)) = ∫ σ·τ dx + ∫ u div(τ) dx + ∫ div(σ) v dx
//!   L((τ,v))       = -∫ f v dx
//!
//! This produces a saddle-point system:
//!   | M   B | | σ |   | 0  |
//!   | B^T 0 | | u | = | -f |

use sonium_fem::assembly::mixed::assemble_mixed;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::{BilinearForm, LocalMatrix, SourceForm};
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::generators::unit_square_triangles;
use sonium_fem::spaces::dg::DGSpace;
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, HdivSpace};

use sonium_solvers::direct::SparseLuFactorization;

/// H(div) mass form: a(σ,τ) = ∫ σ·τ dx
struct HdivMassForm;

impl BilinearForm<f64> for HdivMassForm {
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
                let mut dot = 0.0;
                for d in 0..dim {
                    dot += test.vector_value(i, d) * trial.vector_value(j, d);
                }
                local_matrix.add(i, j, dot * w);
            }
        }
    }
}

/// Divergence coupling form: b(τ,u) = ∫ u div(τ) dx
/// test = H(div) (τ), trial = DG (u)
struct DivCouplingForm;

impl BilinearForm<f64> for DivCouplingForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n_test = test.n_dofs;
        let n_trial = trial.n_dofs;
        let w = weight * test.det_j;
        for i in 0..n_test {
            let div_tau = test.derivatives[i];
            for j in 0..n_trial {
                local_matrix.add(i, j, div_tau * trial.value(j) * w);
            }
        }
    }
}

fn main() {
    println!("=== DOLFINx Demo: Mixed Poisson (RT0 + DG P1) ===\n");

    // --- 1. Mesh ---
    let n = 16;
    let mesh = unit_square_triangles(n);
    println!(
        "Mesh: {} nodes, {} elements",
        mesh.num_nodes(),
        mesh.num_elements()
    );

    // --- 2. Function spaces ---
    let sigma_space = HdivSpace::new();
    let u_space = DGSpace::new(PolynomialDegree::P1);
    let n_sigma = sigma_space.num_global_dofs(&mesh);
    let n_u = u_space.num_global_dofs(&mesh);
    println!("RT0 (σ): {} DOFs", n_sigma);
    println!("DG1 (u): {} DOFs", n_u);
    println!("Total:   {} DOFs\n", n_sigma + n_u);

    // --- 3. Assemble block system ---
    println!("Assembling mixed system...");
    let system = assemble_mixed(
        &HdivMassForm,
        &DivCouplingForm,
        &sigma_space,
        &u_space,
        &mesh,
        2,
    );

    let mono_triplet = system.to_monolithic();
    let a = mono_triplet.to_csr();
    println!("Monolithic: {}×{}, {} nnz", a.num_rows, a.num_cols, a.nnz());

    // --- 4. RHS: [0; -f] where f = sin(5x) + 1 ---
    let f_source = SourceForm::new(|x, _y, _z| -((5.0 * x).sin() + 1.0));
    let rhs_u = sonium_fem::assembly::assemble_linear(&f_source, &u_space, &mesh, 2);

    let n_total = n_sigma + n_u;
    let mut b: Vec<f64> = vec![0.0; n_total];
    for (i, &val) in rhs_u.iter().enumerate() {
        b[n_sigma + i] = val;
    }

    // --- 5. Solve ---
    // Saddle-point system (zero (2,2) block) → ILU breaks down.
    // Use sparse LU (small problem, 2336 DOFs).
    println!("\nSolving with sparse LU...");

    match SparseLuFactorization::factorize(&a) {
        Ok(lu) => {
            let x = lu.solve(&b).expect("sparse LU solve");
            println!("  Direct solve complete");

            let sigma_sol: Vec<f64> = x.iter().take(n_sigma).copied().collect();
            let u_sol: Vec<f64> = x.iter().skip(n_sigma).copied().collect();

            let max_s = sigma_sol.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_s = sigma_sol.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_u = u_sol.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_u = u_sol.iter().cloned().fold(f64::INFINITY, f64::min);

            println!("\n--- Results ---");
            println!("σ range: [{:.6}, {:.6}]", min_s, max_s);
            println!("u range: [{:.6}, {:.6}]", min_u, max_u);

            // Project DG u to nodes for VTK
            let mut node_u = vec![0.0; mesh.num_nodes()];
            let mut node_count = vec![0u32; mesh.num_nodes()];
            for elem_idx in 0..mesh.num_elements() {
                let verts = mesh.elements[elem_idx].vertices();
                let dofs = u_space.local_to_global(&mesh, elem_idx);
                for (li, &v) in verts.iter().enumerate() {
                    if li < dofs.len() {
                        node_u[v] += u_sol[dofs[li]];
                        node_count[v] += 1;
                    }
                }
            }
            for i in 0..mesh.num_nodes() {
                if node_count[i] > 0 {
                    node_u[i] /= node_count[i] as f64;
                }
            }

            let mut writer = VtuWriter::new(&mesh);
            writer.add_point_scalar("u", &node_u);
            match writer.write("mixed_poisson.vtu") {
                Ok(_) => println!("\nWrote mixed_poisson.vtu"),
                Err(e) => println!("\nFailed to write VTK: {e}"),
            }
        }
        Err(e) => println!("  LU factorization failed: {e}"),
    }

    println!("\n=== Demo Complete ===");
}
