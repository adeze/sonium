//! Poisson equation
//!
//! Translated from DOLFINx cpp/demo/poisson/main.cpp
//!
//! Solves: -κ∇²u = f  in Ω = [0,2] × [0,1]
//! BCs:    u = 0  on x=0 and x=2 (Dirichlet)
//!         κ∂u/∂n = g  on y=0 and y=1 (Neumann)
//!
//! Parameters:
//!   κ = 2.0  (diffusion coefficient)
//!   f(x,y) = 10 exp(-((x-0.5)² + (y-0.5)²) / 0.02)  (Gaussian source)
//!   g(x,y) = sin(5x)  (Neumann flux on top/bottom)
//!
//! DOLFINx form: a = κ inner(grad(u), grad(v)) dx
//!               L = inner(f, v) dx + inner(g, v) ds

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::locate_dofs_geometrical;
use sonium_fem::forms::{BilinearForm, LocalMatrix, SourceForm};
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, H1Space};

use sonium_solvers::{LinearSolveConfig, LinearSolverType, solve};

/// Scaled Laplacian form: a(u,v) = κ ∫ ∇u·∇v dx
struct ScaledLaplacianForm {
    kappa: f64,
}

impl BilinearForm<f64> for ScaledLaplacianForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j * self.kappa;
        for i in 0..n {
            for j in 0..n {
                let grad_dot: f64 = (0..dim).map(|d| test.grad(i, d) * trial.grad(j, d)).sum();
                local_matrix.add(i, j, grad_dot * w);
            }
        }
    }
}

fn main() {
    println!("=== DOLFINx Demo: Poisson Equation ===\n");

    // --- 1. Mesh: [0,2] × [0,1] rectangle ---
    let nx = 32;
    let ny = 16;
    let mesh = rectangular_mesh_triangles(0.0, 2.0, 0.0, 1.0, nx, ny);
    let space = H1Space::new(PolynomialDegree::P1);
    let n_dofs = space.num_global_dofs(&mesh);
    println!(
        "Mesh: {} nodes, {} elements",
        mesh.num_nodes(),
        mesh.num_elements()
    );
    println!("P1 space: {} DOFs", n_dofs);

    // --- 2. Assemble stiffness: a(u,v) = κ ∫ ∇u·∇v dx, κ = 2.0 ---
    let kappa = 2.0;
    let form = ScaledLaplacianForm { kappa };
    let triplet = assemble_bilinear(&form, &space, &space, &mesh, 1);
    let mut a = triplet.to_csr();
    println!("Stiffness: {} nnz", a.nnz());

    // --- 3. RHS: L(v) = ∫ f·v dx + ∫ g·v ds ---
    // Source: f(x,y) = 10 exp(-((x-0.5)² + (y-0.5)²) / 0.02)
    let source = SourceForm::new(|x, y, _z| {
        let dx = x - 0.5;
        let dy = y - 0.5;
        10.0 * (-(dx * dx + dy * dy) / 0.02).exp()
    });
    let rhs_vol = sonium_fem::assembly::assemble_linear(&source, &space, &mesh, 2);
    let mut b: Vec<f64> = rhs_vol;

    // Neumann: g(x) = sin(5x) on top/bottom boundaries
    // Add ∫ g·v ds to the RHS using boundary mass approach
    // For detected boundaries, marker 0 = all boundary edges
    let mut mesh_bc = mesh.clone();
    mesh_bc.detect_boundaries();

    // Compute ∫_∂Ω sin(5x) φ_i dΓ on the Neumann boundaries (y=0 and y=1)
    let tol = 1e-10;
    for (node_idx, node) in mesh.nodes.iter().enumerate() {
        if node.y < tol || (1.0 - node.y) < tol {
            // Approximate: g × boundary node contribution
            // (Proper integration via boundary mass, simplified here)
            let h = 2.0 / nx as f64;
            b[node_idx] += (5.0_f64 * node.x).sin() * h * 0.5; // midpoint rule per edge
        }
    }

    // --- 4. Dirichlet BC: u = 0 on x=0 and x=2 ---
    let dirichlet_dofs =
        locate_dofs_geometrical(&mesh, &space, |x, _y, _z| x < tol || (2.0 - x) < tol);
    println!("Dirichlet DOFs: {}", dirichlet_dofs.len());

    for &dof in &dirichlet_dofs {
        let row_start = a.row_ptrs[dof];
        let row_end = a.row_ptrs[dof + 1];
        for j in row_start..row_end {
            a.values[j] = if a.col_indices[j] == dof { 1.0 } else { 0.0 };
        }
        b[dof] = 0.0;
    }

    // --- 5. Solve (DOLFINx uses LU; we use GMRES+ILU for scalability) ---
    println!("\nSolving...");
    let config = LinearSolveConfig {
        solver: LinearSolverType::GmresIlu,
        max_iterations: 1000,
        tolerance: 1e-12,
        ..Default::default()
    };

    match solve(&a, &b, &config) {
        Ok(result) => {
            println!(
                "  Converged in {} iterations (residual: {:.2e})",
                result.iterations, result.residual
            );

            let u: Vec<f64> = result.x.to_vec();
            let max_u = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_u = u.iter().cloned().fold(f64::INFINITY, f64::min);
            println!("\n--- Results ---");
            println!("u range: [{:.6}, {:.6}]", min_u, max_u);

            // --- 6. VTK output ---
            let mut writer = VtuWriter::new(&mesh);
            writer.add_point_scalar("u", &u);

            // Also output the source term for visualization
            let f_vals: Vec<f64> = mesh
                .nodes
                .iter()
                .map(|p| {
                    let dx = p.x - 0.5;
                    let dy = p.y - 0.5;
                    10.0 * (-(dx * dx + dy * dy) / 0.02).exp()
                })
                .collect();
            writer.add_point_scalar("f", &f_vals);

            match writer.write("poisson.vtu") {
                Ok(_) => println!("\nWrote poisson.vtu"),
                Err(e) => println!("\nFailed to write VTK: {e}"),
            }
        }
        Err(e) => println!("  Solver failed: {e}"),
    }

    println!("\n=== Demo Complete ===");
}
