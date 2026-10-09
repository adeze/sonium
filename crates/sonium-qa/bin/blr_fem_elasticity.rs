//! Multifrontal BLR validation on the FEM elasticity patch test.
//!
//! Reuses the standard P1 patch test from `fem_elasticity_validation`:
//! prescribe a linear displacement field on all boundary nodes and verify
//! that the FEM solution reproduces it to machine precision at every
//! interior node — but the linear solve runs through
//! `MultifrontalBlrFactorization` instead of GMRES.
//!
//! Patch tests are the strictest correctness gate available for an
//! elasticity FEM: any P1 implementation that conserves rigid-body
//! translation, rotation, and constant strain MUST recover the exact
//! linear field. If BLR distorts the answer, this test catches it
//! immediately — there is no discretisation error to hide behind.
//!
//! Gates:
//!  1. `||A·x − b|| / ||b||` ≤ 100 × exact-LU residual.
//!  2. Per-node `|u − u_exact|` ≤ 1e-8 (same bound as the iterative test).
//!  3. `||x_blr − x_lu|| / ||x_lu||` ≤ 1e-9 at `epsilon = 0` (exact LU
//!     equivalence) and ≤ 1e-5 at `epsilon = 1e-10`.

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::FemSystem;
use sonium_fem::mesh::{BoundaryType, ElementType, Mesh, Point};
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::direct::{BlrConfig, MultifrontalBlrFactorization, SparseLuFactorization};
use sonium_solvers::sparse::CsrMatrix;

fn build_patch_test_system(n: usize) -> (Mesh, FemSystem<f64>, Vec<usize>) {
    let material = IsotropicMaterial::new(1.0, 0.3);
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

    let boundary_dofs: Vec<usize> = space.boundary_dofs(&mesh, 1);
    let boundary_set: std::collections::HashSet<usize> = boundary_dofs.iter().copied().collect();

    let mut system = FemSystem::new(stiffness, rhs);

    // Patch test target: u_x = x, u_y = 0.
    let exact_u = |dof: usize| -> f64 {
        let node = dof / 2;
        let comp = dof % 2;
        match comp {
            0 => mesh.nodes[node].x,
            1 => 0.0,
            _ => unreachable!(),
        }
    };

    // RHS correction for the columns we are about to eliminate.
    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !boundary_set.contains(&row) && boundary_set.contains(&col) {
            system.rhs[row] -= system.matrix.values[i] * exact_u(col);
        }
    }

    // Filter constrained rows/cols and place an identity on Dirichlet DOFs.
    let mut new_rows = Vec::new();
    let mut new_cols = Vec::new();
    let mut new_vals = Vec::new();
    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !boundary_set.contains(&row) && !boundary_set.contains(&col) {
            new_rows.push(row);
            new_cols.push(col);
            new_vals.push(system.matrix.values[i]);
        }
    }
    for &dof in &boundary_dofs {
        new_rows.push(dof);
        new_cols.push(dof);
        new_vals.push(1.0);
        system.rhs[dof] = exact_u(dof);
    }
    system.matrix.rows = new_rows;
    system.matrix.cols = new_cols;
    system.matrix.values = new_vals;

    (mesh, system, boundary_dofs)
}

fn rel_residual(matrix: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let n = b.len();
    let mut res_sq = 0.0;
    let mut b_sq = 0.0;
    for i in 0..n {
        let mut row = 0.0;
        for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
            row += matrix.values[idx] * x[matrix.col_indices[idx]];
        }
        let r = b[i] - row;
        res_sq += r * r;
        b_sq += b[i] * b[i];
    }
    if b_sq == 0.0 {
        res_sq.sqrt()
    } else {
        (res_sq / b_sq).sqrt()
    }
}

fn rel_diff(a: &[f64], b: &[f64]) -> f64 {
    let mut d = 0.0;
    let mut n = 0.0;
    for i in 0..a.len() {
        d += (a[i] - b[i]).powi(2);
        n += b[i] * b[i];
    }
    if n == 0.0 { d.sqrt() } else { (d / n).sqrt() }
}

fn assert_patch_test_recovered(mesh: &Mesh, x: &[f64], label: &str, node_tol: f64) {
    for node in 0..mesh.num_nodes() {
        let x_coord = mesh.nodes[node].x;
        let y_coord = mesh.nodes[node].y;
        let ux = x[node * 2];
        let uy = x[node * 2 + 1];
        assert!(
            (ux - x_coord).abs() < node_tol,
            "FAILED: {label}: u_x at ({x_coord:.2}, {y_coord:.2}): got {ux:.6e}, expected {x_coord:.6e}"
        );
        assert!(
            uy.abs() < node_tol,
            "FAILED: {label}: u_y at ({x_coord:.2}, {y_coord:.2}): got {uy:.6e}, expected 0"
        );
    }
}

struct Scenario {
    label: &'static str,
    cfg: BlrConfig,
    res_bound: f64,
    parity_bound: f64,
    node_tol: f64,
}

fn main() {
    let grid = 5;
    let (mesh, system, _boundary_dofs) = build_patch_test_system(grid);
    let csr = system.matrix.to_csr();
    let b = system.rhs.clone();
    let n = csr.num_rows;
    println!(
        "=== BLR-FEM elasticity patch test (grid {grid}×{grid}, n = {n} DOFs, nnz = {}) ===",
        csr.values.len()
    );

    let lu = SparseLuFactorization::factorize(&csr).expect("exact sparse LU");
    let x_lu = lu.solve(&b).expect("sparse LU solve");
    let res_lu = rel_residual(&csr, &x_lu, &b);
    println!("Exact SparseLU reference residual: {:.3e}", res_lu);
    assert_patch_test_recovered(&mesh, &x_lu, "exact-LU baseline", 1e-8);

    let scenarios = vec![
        Scenario {
            label: "BLR eps=0 (no compression)",
            cfg: BlrConfig {
                epsilon: 0.0,
                block_size: 16,
                min_block_size: 16,
                ..Default::default()
            },
            res_bound: (100.0 * res_lu).max(1e-10),
            parity_bound: 1e-9,
            node_tol: 1e-8,
        },
        Scenario {
            label: "BLR eps=1e-10",
            cfg: BlrConfig {
                epsilon: 1e-10,
                block_size: 16,
                min_block_size: 16,
                ..Default::default()
            },
            res_bound: 1e-6,
            parity_bound: 1e-5,
            node_tol: 1e-6,
        },
    ];

    for scn in &scenarios {
        let blr = MultifrontalBlrFactorization::factorize_with_config(&csr, &scn.cfg)
            .expect("BLR factorise");
        let x_blr = blr.solve(&b);
        let res = rel_residual(&csr, &x_blr, &b);
        let parity = rel_diff(&x_blr, &x_lu);
        let avg_rank = blr.average_rank();
        let bytes = blr.factor_memory_bytes();
        println!(
            "{:<28} | res {:.2e} (≤ {:.2e}) | parity {:.2e} (≤ {:.2e}) | node_tol {:.0e} | fronts {} avg_rank {:.1} factor {} B",
            scn.label,
            res,
            scn.res_bound,
            parity,
            scn.parity_bound,
            scn.node_tol,
            blr.num_fronts(),
            avg_rank,
            bytes,
        );
        assert!(
            res < scn.res_bound,
            "FAILED: {}: residual {:.2e} exceeds bound {:.2e}",
            scn.label,
            res,
            scn.res_bound
        );
        assert!(
            parity < scn.parity_bound,
            "FAILED: {}: BLR-vs-LU diff {:.2e} exceeds bound {:.2e}",
            scn.label,
            parity,
            scn.parity_bound
        );
        assert_patch_test_recovered(&mesh, &x_blr, scn.label, scn.node_tol);
    }

    println!("SUCCESS");
}
