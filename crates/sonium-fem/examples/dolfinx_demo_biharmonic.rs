//! Biharmonic equation — C0 interior penalty method
//!
//! Translated from DOLFINx cpp/demo/biharmonic/main.cpp
//!
//! Solves: ∇⁴u = f  in Ω = [0,1]²
//! BCs:    u = 0 on ∂Ω,  ∇²u = 0 on ∂Ω (simply supported)
//!
//! Source: f(x,y) = 4π⁴ sin(πx) sin(πy)
//! Exact:  u(x,y) = sin(πx) sin(πy)
//!
//! Method: C0-IP on continuous P1 Lagrange space.  The weak form is:
//!
//!   a(u,v) = Σ_K ∫_K Δu Δv dx                           (cell integral — zero for P1)
//!          - Σ_F ∫_F {{Δu}} [[∇v·n]] dS                 (consistency — zero for P1)
//!          - Σ_F ∫_F [[∇u·n]] {{Δv}} dS                 (symmetry — zero for P1)
//!          + Σ_F α/h ∫_F [[∇u·n]] [[∇v·n]] dS           (penalty)
//!
//! For P1 elements, div(grad(u)) = 0 (constant gradients), so the cell
//! integral and consistency/symmetry terms all vanish.  The biharmonic is
//! captured entirely by the normal-derivative jump penalty.
//!
//! DOLFINx uses P2 elements where the cell integral is nonzero; with P1
//! we get a lower-order but structurally identical method.

use sonium_fem::assembly::dg::assemble_dg;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::locate_dofs_geometrical;
use sonium_fem::forms::SourceForm;
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::generators::unit_square_triangles;
use sonium_fem::physics::solid::biharmonic::{BiharmonicCellForm, BiharmonicFacetForm};
use sonium_fem::spaces::{FunctionSpace, H1Space};

use sonium_solvers::{LinearSolveConfig, LinearSolverType, solve};

fn main() {
    println!("=== DOLFINx Demo: Biharmonic Equation (C0-IP) ===\n");

    // --- 1. Create mesh (matching DOLFINx: 32×32 unit square, triangles) ---
    let n = 32;
    let mesh = unit_square_triangles(n);
    println!(
        "Mesh: {} nodes, {} triangles",
        mesh.num_nodes(),
        mesh.num_elements()
    );

    // --- 2. Function space: continuous P1 Lagrange ---
    // DOLFINx uses P2; we use P1 (cell Hessian vanishes, penalty-only method)
    let space = H1Space::new(PolynomialDegree::P1);
    let n_dofs = space.num_global_dofs(&mesh);
    println!("H1 P1 space: {} DOFs", n_dofs);

    // --- 3. Assemble biharmonic operator ---
    // α = 8.0 (matching DOLFINx)
    let alpha = 8.0;
    println!("Penalty α = {alpha}");

    let cell_form = BiharmonicCellForm;
    let facet_form = BiharmonicFacetForm::new(alpha);

    // assemble_dg works on any FunctionSpace (H1 or DG).
    // For H1, interior facets still connect adjacent elements with shared DOFs.
    let triplet = assemble_dg(&cell_form, &facet_form, &space, &mesh, 2);
    let mut a = triplet.to_csr();
    println!("System: {}×{}, {} nnz", a.num_rows, a.num_cols, a.nnz());

    // --- 4. RHS: l(v) = ∫ f·v dx,  f = 4π⁴ sin(πx)sin(πy) ---
    let pi = std::f64::consts::PI;
    let source = SourceForm::new(|x, y, _z| 4.0 * pi.powi(4) * (pi * x).sin() * (pi * y).sin());
    let mut b = sonium_fem::assembly::assemble_linear(&source, &space, &mesh, 2);

    // --- 5. Dirichlet BC: u = 0 on ∂Ω ---
    // Locate boundary DOFs geometrically (matching DOLFINx locate_dofs_topological)
    let tol = 1e-10;
    let boundary_dofs = locate_dofs_geometrical(&mesh, &space, |x, y, _z| {
        x < tol || (1.0 - x) < tol || y < tol || (1.0 - y) < tol
    });
    println!("Boundary DOFs: {}", boundary_dofs.len());

    // Strong enforcement: zero row, set diagonal to 1, set RHS to 0
    for &dof in &boundary_dofs {
        let row_start = a.row_ptrs[dof];
        let row_end = a.row_ptrs[dof + 1];
        for j in row_start..row_end {
            a.values[j] = if a.col_indices[j] == dof { 1.0 } else { 0.0 };
        }
        b[dof] = 0.0;
    }

    // --- 6. Solve ---
    // DOLFINx uses LU; we use GMRES+ILU for large systems
    println!("\nSolving...");
    let config = LinearSolveConfig {
        solver: LinearSolverType::GmresIlu,
        max_iterations: 2000,
        tolerance: 1e-10,
        ..Default::default()
    };

    match solve(&a, &b, &config) {
        Ok(result) => {
            println!(
                "  Converged in {} iterations (residual: {:.2e})",
                result.iterations, result.residual
            );

            let u: Vec<f64> = result.x.to_vec();

            // --- 7. Error analysis ---
            let mut l2_error_sq = 0.0;
            let mut l2_exact_sq = 0.0;
            let h_elem = 1.0 / n as f64;
            let area = 0.5 * h_elem * h_elem;

            for elem_idx in 0..mesh.num_elements() {
                let verts = mesh.elements[elem_idx].vertices();
                let cx: f64 = verts.iter().map(|&v| mesh.nodes[v].x).sum::<f64>() / 3.0;
                let cy: f64 = verts.iter().map(|&v| mesh.nodes[v].y).sum::<f64>() / 3.0;

                // Interpolate u_h at centroid using P1 shape functions (1/3, 1/3, 1/3)
                let u_h: f64 = verts.iter().map(|&v| u[v]).sum::<f64>() / 3.0;
                let u_ex = (pi * cx).sin() * (pi * cy).sin();

                l2_error_sq += (u_h - u_ex).powi(2) * area;
                l2_exact_sq += u_ex.powi(2) * area;
            }

            let l2_error = l2_error_sq.sqrt();
            let l2_norm = l2_exact_sq.sqrt();

            println!("\n--- Results ---");
            println!("L2 error:     {:.6e}", l2_error);
            println!("L2 norm(u):   {:.6e}", l2_norm);
            println!("Relative err: {:.4}%", 100.0 * l2_error / l2_norm);

            let max_u = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_u = u.iter().cloned().fold(f64::INFINITY, f64::min);
            println!("u range:      [{:.6}, {:.6}]", min_u, max_u);
            println!("Exact max:    {:.6}", 1.0);

            // --- 8. VTK output (matching DOLFINx: file.write({u}, 0)) ---
            let exact: Vec<f64> = mesh
                .nodes
                .iter()
                .map(|p| (pi * p.x).sin() * (pi * p.y).sin())
                .collect();
            let error: Vec<f64> = u.iter().zip(&exact).map(|(a, b)| (a - b).abs()).collect();

            let mut writer = VtuWriter::new(&mesh);
            writer.add_point_scalar("u", &u);
            writer.add_point_scalar("u_exact", &exact);
            writer.add_point_scalar("error", &error);

            match writer.write("biharmonic.vtu") {
                Ok(_) => println!("\nWrote biharmonic.vtu"),
                Err(e) => println!("\nFailed to write VTK: {e}"),
            }
        }
        Err(e) => println!("  Solver failed: {e}"),
    }

    println!("\n=== Demo Complete ===");
}
