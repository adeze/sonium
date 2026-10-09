//! Multifrontal BLR validation on the BEM rigid-sphere scattering problem.
//!
//! Assembles the dense Burton-Miller BEM system on a sphere mesh, converts it
//! to CSR, factorises with `MultifrontalBlrFactorization` at several BLR
//! configurations, and gates:
//!
//! 1. The relative residual `||A·x − b|| / ||b||` of the BLR solve.
//! 2. Parity against the exact `SparseLuFactorization` solution
//!    (`||x_blr − x_lu|| / ||x_lu||`), with a tolerance that scales with
//!    the BLR `epsilon`.
//! 3. The triangle-area-weighted complex BLR surface pressure against the
//!    analytical Mie trace at every collocation direction.
//!
//! Notes on the test problem:
//!
//! - The BEM operator is fully dense, so `from_dense` produces a CSR with
//!   `n²` nonzeros. Multifrontal BLR with the default block_size 128 will
//!   typically end up with a single front of size `n` (no useful
//!   dissection on a complete pattern), but the BLR compression of the
//!   dense `F11` and its panels is what we want to exercise — that path
//!   is otherwise only covered by the synthetic 2D Laplacian unit tests
//!   in `multifrontal_blr_parity.rs`.
//! - We stay at subdiv = 2 (320 elements) so the dense factorise runs in
//!   seconds. Heavier loads should go to a benchmark, not a CI gate.

use num_complex::Complex64;
use sonium_bem::core::assembly::tbem::build_tbem_system_with_beta;
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::types::{BoundaryCondition, PhysicsParams};
use sonium_qa::rigid_sphere_surface_relative_l2_error;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{
    BlrConfig, MultifrontalBlrFactorization, PanelKind, SparseLuFactorization,
};
use sonium_solvers::sparse::CsrMatrix;
use std::f64::consts::PI;

const SPEED_OF_SOUND: f64 = 343.0;
const DENSITY: f64 = 1.21;

fn array2_to_csr(matrix: &DenseMatrix<Complex64>) -> CsrMatrix<Complex64> {
    CsrMatrix::from_dense(matrix, 0.0)
}

fn rel_residual(matrix: &CsrMatrix<Complex64>, x: &[Complex64], b: &[Complex64]) -> f64 {
    let n = b.len();
    let mut res_sq = 0.0;
    let mut b_sq = 0.0;
    for i in 0..n {
        let mut row = Complex64::new(0.0, 0.0);
        for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
            row += matrix.values[idx] * x[matrix.col_indices[idx]];
        }
        let r = b[i] - row;
        res_sq += r.norm_sqr();
        b_sq += b[i].norm_sqr();
    }
    if b_sq == 0.0 {
        res_sq.sqrt()
    } else {
        (res_sq / b_sq).sqrt()
    }
}

fn rel_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut diff_sq = 0.0;
    let mut norm_sq = 0.0;
    for i in 0..a.len() {
        diff_sq += (a[i] - b[i]).norm_sqr();
        norm_sq += b[i].norm_sqr();
    }
    if norm_sq == 0.0 {
        diff_sq.sqrt()
    } else {
        (diff_sq / norm_sq).sqrt()
    }
}

struct Scenario {
    label: &'static str,
    cfg: BlrConfig,
    /// Upper bound on `||A·x − b|| / ||b||` for this scenario.
    res_bound: f64,
    /// Upper bound on `||x_blr − x_lu|| / ||x_lu||`.
    parity_bound: f64,
}

fn main() {
    let radius = 1.0;
    let ka = 1.0;
    let k = ka / radius;
    let frequency = k * SPEED_OF_SOUND / (2.0 * PI);
    let physics = PhysicsParams::new(frequency, SPEED_OF_SOUND, DENSITY, false);

    // subdiv = 2 → 320 elements, n × n dense matrix factorises in < 5 s.
    let subdiv = 2;
    let mesh = generate_icosphere_mesh(radius, subdiv);
    let mut elements = mesh.elements.clone();
    for (i, elem) in elements.iter_mut().enumerate() {
        elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        elem.dof_addresses = vec![i];
    }

    let (beta, beta_scale) = physics.burton_miller_beta_adaptive(radius);
    println!(
        "=== BLR-BEM sphere scattering (subdiv={subdiv}, ka={ka}, β_scale={beta_scale:.1}) ==="
    );

    let system = build_tbem_system_with_beta(&elements, &mesh.nodes, &physics, beta);
    let n = system.num_dofs;
    println!("Assembled dense Burton-Miller system: {n} × {n}");

    let centres: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
    let normals: Vec<[f64; 3]> = elements.iter().map(|e| e.normal).collect();
    let incident = IncidentField::plane_wave_z();
    let rhs_inc = incident.compute_rhs_with_beta(&centres, &normals, &physics, beta);
    let b: Vec<Complex64> = system
        .rhs
        .iter()
        .zip(rhs_inc.iter())
        .map(|(a, c)| a + c)
        .collect();

    let csr = array2_to_csr(&system.matrix);
    println!(
        "Converted to CSR: nnz = {} (dense ratio = {:.2})",
        csr.values.len(),
        csr.values.len() as f64 / (n * n) as f64
    );

    let lu = SparseLuFactorization::factorize(&csr).expect("exact sparse LU");
    let x_lu = lu.solve(&b).expect("sparse LU solve");
    let res_lu = rel_residual(&csr, &x_lu, &b);
    println!("Exact SparseLU reference residual: {:.3e}", res_lu);

    // Mie validation of the exact solve so we know the analytical bar is
    // achievable on this mesh — otherwise the BLR-vs-Mie check would
    // measure mesh discretisation error, not BLR error.
    let mie_error_lu = rigid_sphere_surface_relative_l2_error(&x_lu, &elements, k, radius, 50);
    println!(
        "Exact-LU area-weighted complex Mie error: {:.3}%",
        100.0 * mie_error_lu
    );
    assert!(
        mie_error_lu < 0.02,
        "FAILED: exact-LU complex Mie error {:.3}% exceeds 2% — mesh is too coarse",
        100.0 * mie_error_lu
    );

    let scenarios = vec![
        Scenario {
            label: "BLR eps=0 (no compression)",
            cfg: BlrConfig {
                epsilon: 0.0,
                block_size: 64,
                min_block_size: 64,
                ..Default::default()
            },
            res_bound: (100.0 * res_lu).max(1e-10),
            parity_bound: 1e-8,
        },
        Scenario {
            label: "BLR eps=1e-10",
            cfg: BlrConfig {
                epsilon: 1e-10,
                block_size: 64,
                min_block_size: 64,
                ..Default::default()
            },
            res_bound: 1e-6,
            parity_bound: 1e-4,
        },
        Scenario {
            label: "HODLR eps=1e-10",
            cfg: BlrConfig {
                epsilon: 1e-10,
                block_size: 64,
                min_block_size: 64,
                panel_kind: PanelKind::Hodlr,
                ..Default::default()
            },
            res_bound: 1e-6,
            parity_bound: 1e-4,
        },
    ];

    for scn in &scenarios {
        let blr = MultifrontalBlrFactorization::factorize_with_config(&csr, &scn.cfg)
            .expect("BLR factorise");
        let x_blr = blr.solve(&b);
        let res = rel_residual(&csr, &x_blr, &b);
        let parity = rel_diff(&x_blr, &x_lu);
        let mie_error = rigid_sphere_surface_relative_l2_error(&x_blr, &elements, k, radius, 50);
        let avg_rank = blr.average_rank();
        let bytes = blr.factor_memory_bytes();
        println!(
            "{:<28} | res {:.2e} (≤ {:.2e}) | parity {:.2e} (≤ {:.2e}) | complex Mie {:5.2}% | fronts {} avg_rank {:.1} factor {} KB",
            scn.label,
            res,
            scn.res_bound,
            parity,
            scn.parity_bound,
            100.0 * mie_error,
            blr.num_fronts(),
            avg_rank,
            bytes / 1024,
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
        // BLR must retain the physically accurate complex solution, in
        // addition to its algebraic parity with the exact factorization.
        assert!(
            mie_error < 0.02 && mie_error <= mie_error_lu + 10.0 * scn.parity_bound,
            "FAILED: {}: complex Mie error {:.3}% regressed from LU baseline {:.3}%",
            scn.label,
            100.0 * mie_error,
            100.0 * mie_error_lu
        );
    }

    println!("SUCCESS");
}
