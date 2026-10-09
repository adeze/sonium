//! Sonium side of the wide-MFEM harness, slice 1: H1 Poisson stiffness
//! operator action on the unit cube with Cartesian hexahedra.
//!
//! Assembles the same Galerkin discretization the MFEM driver
//! (`crates/sonium-qa/mfem/poisson_action.cpp`, MFEM 4.5.2, Qp H1,
//! `DiffusionIntegrator`, explicit exact quadrature) builds: `LaplacianForm`
//! on `unit_cube_hexahedra(nx)` with `H1Space`, quadrature exactness `2p`.
//! Compares only basis-independent evidence: DOF count, significance-gated
//! sparsity count, matrix sum (= a(1,1)), and the ones-action nullspace
//! signature. Trace and Frobenius norm are reported but NOT gated — they
//! depend on the nodal basis (uniform sonium nodes vs Lobatto MFEM nodes,
//! which coincide only at Q1/Q2). With `--solve`, additionally runs the
//! manufactured-solve leg (homogeneous Dirichlet, CG, true L2 error), whose
//! Galerkin solution function is basis-independent: the equal-error anchor.
//! Timings compare time-to-action and time-to-solve (assembly once,
//! best-of-21 actions).
//!
//! Run e.g.:
//!   ./mfem/poisson_action 4 2 > /tmp/mfem.json
//!   ./mfem/poisson_solve 4 2 > /tmp/mfem_solve.json
//!   cargo run -p sonium-qa --features qa-binaries --bin qa-mfem-poisson-action -- \
//!     --nx 4 --order 2 --mfem-json /tmp/mfem.json \
//!     --solve --mfem-solve-json /tmp/mfem_solve.json

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::Serialize;
use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::LaplacianForm;
use sonium_fem::mesh::unit_cube_hexahedra;
use sonium_fem::solver::MatrixFreeOperator;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use std::fs;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "qa-mfem-poisson-action",
    about = "Compare sonium-fem Poisson stiffness action against MFEM."
)]
struct Args {
    /// Elements per side of the unit cube.
    #[arg(long, default_value_t = 4)]
    nx: usize,
    /// Polynomial order (1, 2, or 3).
    #[arg(long, default_value_t = 2)]
    order: usize,
    /// MFEM driver JSON to compare against (prints sonium evidence only if absent).
    #[arg(long, default_value = "")]
    mfem_json: String,
    /// Also run the manufactured-solve leg (Dirichlet + CG + L2 error).
    #[arg(long, default_value_t = false)]
    solve: bool,
    /// MFEM solve-driver JSON for the solve leg.
    #[arg(long, default_value = "")]
    mfem_solve_json: String,
}

#[derive(Debug, Serialize)]
struct Evidence {
    nx: usize,
    order: usize,
    ndofs: usize,
    nnz: usize,
    nnz_nonzero: usize,
    nnz_significant: usize,
    matrix_sum: f64,
    matrix_trace: f64,
    matrix_frob2: f64,
    action_sum: f64,
    action_sumsq: f64,
    assemble_s: f64,
    action_s_best: f64,
    csr_action_s_best: f64,
}

#[derive(Debug, serde::Deserialize)]
struct MfemSolveEvidence {
    l2_error: f64,
    cg_iters: usize,
    solve_s: f64,
}

#[derive(Debug, serde::Deserialize)]
struct MfemEvidence {
    nx: usize,
    order: usize,
    ndofs: usize,
    nnz: usize,
    nnz_nonzero: usize,
    nnz_significant: usize,
    matrix_sum: f64,
    matrix_trace: f64,
    matrix_frob2: f64,
    action_sum: f64,
    action_sumsq: f64,
    assemble_s: f64,
    action_s_best: f64,
}

fn rel_diff(a: f64, b: f64) -> f64 {
    let scale = a.abs().max(b.abs()).max(1e-300);
    (a - b).abs() / scale
}

fn main() {
    // Machine-checkable output contract: SUCCESS on stdout for a pass,
    // FAILED plus a non-zero exit for any failure.
    match run() {
        Ok(()) => println!("SUCCESS"),
        Err(error) => {
            println!("FAILED: {error:?}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    let degree = match args.order {
        1 => PolynomialDegree::P1,
        2 => PolynomialDegree::P2,
        3 => PolynomialDegree::P3,
        _ => bail!("order must be 1, 2, or 3"),
    };

    let mesh = unit_cube_hexahedra(args.nx);
    let space = H1Space::new(degree);
    let ndofs = space.num_global_dofs(&mesh);
    // Exactness 2p over-integrates the degree-2(p-1) stiffness integrand;
    // MFEM's default diffusion rule is likewise exact on affine hexes.
    let quad_order = 2 * args.order;

    let form = LaplacianForm;
    let start = Instant::now();
    let triplet = assemble_bilinear(&form, &space, &space, &mesh, quad_order);
    let csr = triplet.to_csr();
    let assemble_s = start.elapsed().as_secs_f64();

    let mut matrix_sum = 0.0;
    let mut matrix_trace = 0.0;
    let mut matrix_frob2 = 0.0;
    for row in 0..ndofs {
        for (col, value) in csr.row_entries(row) {
            matrix_sum += value;
            matrix_frob2 += value * value;
            if col == row {
                matrix_trace += value;
            }
        }
    }

    let operator = MatrixFreeOperator::new(&form, &space, &mesh, quad_order);
    let x = vec![1.0; ndofs];
    let mut action_s_best = f64::INFINITY;
    let mut y = vec![0.0; ndofs];
    for _ in 0..21 {
        let start = Instant::now();
        y = operator.apply_vec(&x);
        action_s_best = action_s_best.min(start.elapsed().as_secs_f64());
    }
    let action_sum: f64 = y.iter().sum();
    let action_sumsq: f64 = y.iter().map(|v| v * v).sum();

    // Assembled-path action for a like-for-like SpMV comparison: MFEM's
    // timed action above is CSR SpMV, while `action_s_best` is the
    // matrix-free element-rebuild path.
    let mut csr_action_s_best = f64::INFINITY;
    for _ in 0..21 {
        let start = Instant::now();
        let _ = csr.matvec(&x);
        csr_action_s_best = csr_action_s_best.min(start.elapsed().as_secs_f64());
    }

    let mut nnz_nonzero = 0;
    let mut maxabs = 0.0f64;
    for row in 0..ndofs {
        for (_, value) in csr.row_entries(row) {
            maxabs = maxabs.max(value.abs());
            if value != 0.0 {
                nnz_nonzero += 1;
            }
        }
    }
    let mut nnz_significant = 0;
    for row in 0..ndofs {
        for (_, value) in csr.row_entries(row) {
            if value.abs() > 1e-12 * maxabs {
                nnz_significant += 1;
            }
        }
    }

    let evidence = Evidence {
        nx: args.nx,
        order: args.order,
        ndofs,
        nnz: csr.nnz(),
        nnz_nonzero,
        nnz_significant,
        matrix_sum,
        matrix_trace,
        matrix_frob2,
        action_sum,
        action_sumsq,
        assemble_s,
        action_s_best,
        csr_action_s_best,
    };
    println!("{}", serde_json::to_string_pretty(&evidence)?);

    if args.mfem_json.is_empty() {
        return Ok(());
    }
    let document =
        fs::read_to_string(&args.mfem_json).with_context(|| "read MFEM JSON".to_string())?;
    let mfem: MfemEvidence = serde_json::from_str(&document)?;
    if mfem.nx != args.nx || mfem.order != args.order {
        bail!(
            "MFEM grid ({}, {}) does not match requested grid",
            mfem.nx,
            mfem.order
        );
    }
    if mfem.ndofs != ndofs {
        bail!("DOF mismatch: MFEM {} vs sonium {ndofs}", mfem.ndofs);
    }
    // Entries below 1e-12 of max are analytic-zero roundoff residue whose
    // storage differs (exact zeros kept vs dropped); the significant count
    // is the structural gate.
    if mfem.nnz_significant != nnz_significant {
        bail!(
            "significant-nonzero mismatch: MFEM {} vs sonium {nnz_significant} (nonzero {} vs {}, stored {} vs {})",
            mfem.nnz_significant,
            mfem.nnz_nonzero,
            nnz_nonzero,
            mfem.nnz,
            csr.nnz()
        );
    }
    // Basis-independent gates only. Trace and Frobenius norm depend on the
    // nodal basis (uniform sonium nodes vs Lobatto MFEM nodes agree only at
    // Q1/Q2 where the node sets coincide), so they are reported, not gated.
    // matrix_sum (= a(1,1)) and the ones-action nullspace signature are
    // basis-independent and gate absolutely.
    for (label, got, want) in [
        ("matrix_sum", matrix_sum, mfem.matrix_sum),
        ("action_sum", action_sum, mfem.action_sum),
        ("action_sumsq", action_sumsq, mfem.action_sumsq),
    ] {
        if (got - want).abs() > 1e-9 {
            bail!("{label} mismatch: sonium {got:.6e} vs MFEM {want:.6e}");
        }
    }
    println!(
        "AGREE ndofs={ndofs} nnz_significant={nnz_significant} trace(sonium {matrix_trace:.6e}/{:.6e} mfem) frob2(sonium {matrix_frob2:.6e}/{:.6e} mfem) assemble_s(sonium {:.4}/{:.4} mfem) action_s_best(sonium {:.6}/{:.6} mfem, csr {:.6})",
        mfem.matrix_trace,
        mfem.matrix_frob2,
        assemble_s,
        mfem.assemble_s,
        action_s_best,
        mfem.action_s_best,
        csr_action_s_best,
    );
    if !args.solve {
        return Ok(());
    }
    run_solve_leg(&args, &mesh, &space, ndofs, quad_order)?;
    Ok(())
}

/// Manufactured-solve leg: -Δu = f with u = sin(πx)sin(πy)sin(πz), zero on
/// the whole boundary, so homogeneous Dirichlet applies. The Galerkin
/// solution function is basis-independent (same Qp space on the same mesh),
/// hence both L2 errors must agree to solver tolerance — the equal-error
/// anchor of the harness.
fn run_solve_leg(
    args: &Args,
    mesh: &sonium_fem::mesh::Mesh,
    space: &H1Space,
    ndofs: usize,
    quad_order: usize,
) -> Result<()> {
    use sonium_fem::assembly::assemble_linear;
    use sonium_fem::boundary::{FemSystem, generic::apply_homogeneous_dirichlet_dofs};
    use sonium_fem::forms::SourceForm;
    use sonium_fem::spaces::FunctionSpace;
    use sonium_solvers::{CgConfig, cg};

    let pi = std::f64::consts::PI;
    let load = SourceForm::new(move |x, y, z| {
        3.0 * pi * pi * (pi * x).sin() * (pi * y).sin() * (pi * z).sin()
    });
    // Over-integrated load (exactness 8), matching the MFEM side, so the
    // transcendental source resolves identically on both sides.
    let rhs = assemble_linear(&load, space, mesh, 8);
    let triplets = assemble_bilinear(&LaplacianForm, space, space, mesh, quad_order);
    let mut system = FemSystem::new(triplets, rhs);

    // Boundary DOFs identified geometrically (u vanishes on all six faces).
    let coords = space
        .dof_coordinates(mesh)
        .with_context(|| "dof coordinates unavailable".to_string())?;
    let on_boundary = |p: &[f64; 3]| p.iter().any(|c| c.abs() < 1e-12 || (c - 1.0).abs() < 1e-12);
    let boundary_dofs: Vec<usize> = coords
        .iter()
        .enumerate()
        .filter(|(_, p)| on_boundary(p))
        .map(|(dof, _)| dof)
        .collect();
    apply_homogeneous_dirichlet_dofs(&mut system, &boundary_dofs);
    let csr = system.matrix.to_csr();

    let config = CgConfig {
        tolerance: 1e-12,
        max_iterations: 10000,
        ..Default::default()
    };
    let start = Instant::now();
    let solution = cg(&csr, &system.rhs, &config);
    let solve_s = start.elapsed().as_secs_f64();

    // True L2 error ||u_h - u|| by high-order element quadrature. (A
    // mass-weighted coefficient error would measure against the nodal
    // interpolant instead — a different, smaller quantity.)
    use sonium_fem::mesh::ElementType;
    use sonium_fem::quadrature::QuadratureRule;
    let error_rule = QuadratureRule::new(ElementType::Hexahedron, 8);
    let mut err2: f64 = 0.0;
    for (elem_idx, _) in mesh.elements.iter().enumerate() {
        let dofs: Vec<usize> = space.local_to_global(mesh, elem_idx);
        for qp in &error_rule.points {
            let tab = space.tabulate(mesh, elem_idx, qp.coords[0], qp.coords[1], qp.coords[2]);
            let uh: f64 = dofs
                .iter()
                .enumerate()
                .map(|(local, &global)| solution.x[global] * tab.values[local])
                .sum();
            let [x, y, z] = tab.physical_coords;
            let u = (pi * x).sin() * (pi * y).sin() * (pi * z).sin();
            err2 += (uh - u) * (uh - u) * qp.weight * tab.det_j;
        }
    }
    let l2_error = err2.max(0.0).sqrt();

    println!(
        "SOLVE ndofs={ndofs} l2_error={l2_error:.6e} cg_iters={} cg_res={:.2e} solve_s={solve_s:.4}",
        solution.iterations, solution.residual,
    );
    if args.mfem_solve_json.is_empty() {
        return Ok(());
    }
    let document = fs::read_to_string(&args.mfem_solve_json)
        .with_context(|| "read MFEM solve JSON".to_string())?;
    let mfem: MfemSolveEvidence = serde_json::from_str(&document)?;
    if (l2_error - mfem.l2_error).abs() > 1e-9 + 1e-6 * mfem.l2_error {
        bail!(
            "L2 error mismatch: sonium {l2_error:.6e} vs MFEM {:.6e}",
            mfem.l2_error
        );
    }
    println!(
        "SOLVE-AGREE l2_rel={:.2e} iters(sonium {}/{} mfem) solve_s(sonium {:.4}/{:.4} mfem)",
        rel_diff(l2_error, mfem.l2_error),
        solution.iterations,
        mfem.cg_iters,
        solve_s,
        mfem.solve_s,
    );
    Ok(())
}
