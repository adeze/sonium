//! Sonium side of the wide-MFEM harness, slice 3: complex Helmholtz
//! operator action + manufactured solve at fixed kh on the unit cube.
//!
//! Assembles the same Galerkin discretization the MFEM driver
//! (`crates/sonium-qa/mfem/helmholtz.cpp`, MFEM 4.5.2, Qp H1,
//! `DiffusionIntegrator` + `MassIntegrator`, explicit exact quadrature)
//! builds: `LaplacianForm` + `MassForm` on `unit_cube_hexahedra(nx)` with
//! `H1Space`, quadrature exactness `2p`, combined as A = S - k²M with complex
//! k. Compares only basis-independent evidence: DOF count,
//! significance-gated sparsity count, complex matrix sum (= -k²·volume), and
//! the ones-action signature. The solve leg (homogeneous Dirichlet, complex
//! GMRES restart 50 to 1e-12, true complex L2 error) gates L2 agreement with
//! the MFEM real-block GMRES leg — the equal-error anchor. kh = Re(k)/nx is
//! matched by construction; run (nx, k) pairs at the same kh.
//!
//! Run e.g.:
//!   ./mfem/helmholtz 8 1 8.0 0.5 > /tmp/mfem_h.json
//!   cargo run -p sonium-qa --features qa-binaries --bin qa-mfem-helmholtz -- \
//!     --nx 8 --order 1 --kr 8.0 --ki 0.5 --mfem-json /tmp/mfem_h.json

use anyhow::{Context, Result, bail};
use clap::Parser;
use num_complex::Complex64;
use serde::Serialize;
use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::FemSystem;
use sonium_fem::forms::{LaplacianForm, MassForm};
use sonium_fem::mesh::unit_cube_hexahedra;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::sparse::TripletMatrix;
use std::fs;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "qa-mfem-helmholtz",
    about = "Compare sonium-fem complex Helmholtz action+solve against MFEM."
)]
struct Args {
    /// Elements per side of the unit cube.
    #[arg(long, default_value_t = 8)]
    nx: usize,
    /// Polynomial order (1, 2, or 3).
    #[arg(long, default_value_t = 1)]
    order: usize,
    /// Real part of the wavenumber.
    #[arg(long, default_value_t = 8.0)]
    kr: f64,
    /// Imaginary part of the wavenumber.
    #[arg(long, default_value_t = 0.5)]
    ki: f64,
    /// MFEM driver JSON to compare against (prints sonium evidence only if absent).
    #[arg(long, default_value = "")]
    mfem_json: String,
}

#[derive(Debug, Serialize)]
struct Evidence {
    nx: usize,
    order: usize,
    kr: f64,
    ki: f64,
    kh: f64,
    ndofs: usize,
    nnz: usize,
    nnz_nonzero: usize,
    nnz_significant: usize,
    matrix_sum_re: f64,
    matrix_sum_im: f64,
    action_sum_re: f64,
    action_sum_im: f64,
    action_sumsq: f64,
    assemble_s: f64,
    action_s_best: f64,
    l2_error: f64,
    gmres_iters: usize,
    gmres_res: f64,
    solve_s: f64,
}

#[derive(Debug, serde::Deserialize)]
struct MfemEvidence {
    nx: usize,
    order: usize,
    kr: f64,
    ki: f64,
    ndofs: usize,
    matrix_sum_re: f64,
    matrix_sum_im: f64,
    nnz_nonzero: usize,
    nnz_significant: usize,
    action_sum_re: f64,
    action_sum_im: f64,
    action_sumsq: f64,
    assemble_s: f64,
    action_s_best: f64,
    l2_error: f64,
    gmres_iters: usize,
    solve_s: f64,
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
    let wavenumber = Complex64::new(args.kr, args.ki);
    let kh = args.kr / args.nx as f64;

    let mesh = unit_cube_hexahedra(args.nx);
    let space = H1Space::new(degree);
    let ndofs = space.num_global_dofs(&mesh);
    let quad_order = 2 * args.order;

    let start = Instant::now();
    let stiffness: TripletMatrix<f64> =
        assemble_bilinear(&LaplacianForm, &space, &space, &mesh, quad_order);
    let mass: TripletMatrix<f64> = assemble_bilinear(&MassForm, &space, &space, &mesh, quad_order);
    // A = S - k²M in triplet form; duplicates sum in to_csr().
    let neg_k2 = -(wavenumber * wavenumber);
    let mut rows = Vec::with_capacity(stiffness.nnz() + mass.nnz());
    let mut cols = Vec::with_capacity(stiffness.nnz() + mass.nnz());
    let mut values = Vec::with_capacity(stiffness.nnz() + mass.nnz());
    for t in 0..stiffness.nnz() {
        rows.push(stiffness.rows[t]);
        cols.push(stiffness.cols[t]);
        values.push(Complex64::new(stiffness.values[t], 0.0));
    }
    for t in 0..mass.nnz() {
        rows.push(mass.rows[t]);
        cols.push(mass.cols[t]);
        values.push(neg_k2 * mass.values[t]);
    }
    let triplet = TripletMatrix {
        rows,
        cols,
        values,
        num_rows: ndofs,
        num_cols: ndofs,
    };
    let csr = triplet.to_csr();
    let assemble_s = start.elapsed().as_secs_f64();

    let mut matrix_sum = Complex64::new(0.0, 0.0);
    let mut maxabs = 0.0f64;
    let mut nnz_nonzero = 0usize;
    for row in 0..ndofs {
        for (_, value) in csr.row_entries(row) {
            matrix_sum += value;
            maxabs = maxabs.max(value.norm());
            if value.norm() != 0.0 {
                nnz_nonzero += 1;
            }
        }
    }
    let mut nnz_significant = 0usize;
    for row in 0..ndofs {
        for (_, value) in csr.row_entries(row) {
            if value.norm() > 1e-12 * maxabs {
                nnz_significant += 1;
            }
        }
    }

    let ones = vec![Complex64::new(1.0, 0.0); ndofs];
    let mut action_s_best = f64::INFINITY;
    let mut y = vec![Complex64::new(0.0, 0.0); ndofs];
    for _ in 0..21 {
        let start = Instant::now();
        y = csr.matvec(&ones);
        action_s_best = action_s_best.min(start.elapsed().as_secs_f64());
    }
    let action_sum: Complex64 = y.iter().sum();
    let action_sumsq: f64 = y.iter().map(|v| v.norm_sqr()).sum();

    let (l2_error, gmres_iters, gmres_res, solve_s) =
        run_solve_leg(&mesh, &space, ndofs, wavenumber, quad_order)?;

    let evidence = Evidence {
        nx: args.nx,
        order: args.order,
        kr: args.kr,
        ki: args.ki,
        kh,
        ndofs,
        nnz: csr.nnz(),
        nnz_nonzero,
        nnz_significant,
        matrix_sum_re: matrix_sum.re,
        matrix_sum_im: matrix_sum.im,
        action_sum_re: action_sum.re,
        action_sum_im: action_sum.im,
        action_sumsq,
        assemble_s,
        action_s_best,
        l2_error,
        gmres_iters,
        gmres_res,
        solve_s,
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
    if (mfem.kr - args.kr).abs() + (mfem.ki - args.ki).abs() > 0.0 {
        bail!(
            "MFEM k ({}+{}i) does not match requested k",
            mfem.kr,
            mfem.ki
        );
    }
    if mfem.ndofs != ndofs {
        bail!("DOF mismatch: MFEM {} vs sonium {ndofs}", mfem.ndofs);
    }
    if mfem.nnz_significant != nnz_significant {
        bail!(
            "significant-nonzero mismatch: MFEM {} vs sonium {nnz_significant} (nonzero {} vs {})",
            mfem.nnz_significant,
            mfem.nnz_nonzero,
            nnz_nonzero,
        );
    }
    // Basis-independent gates: matrix_sum = -k²·volume and the ones-action
    // signature. action_sumsq is reported, not gated (nodal-basis dependent).
    for (label, got, want) in [
        ("matrix_sum_re", matrix_sum.re, mfem.matrix_sum_re),
        ("matrix_sum_im", matrix_sum.im, mfem.matrix_sum_im),
        ("action_sum_re", action_sum.re, mfem.action_sum_re),
        ("action_sum_im", action_sum.im, mfem.action_sum_im),
    ] {
        if (got - want).abs() > 1e-9 {
            bail!("{label} mismatch: sonium {got:.6e} vs MFEM {want:.6e}");
        }
    }
    println!(
        "AGREE kh={kh:.4} ndofs={ndofs} nnz_significant={nnz_significant} action_sumsq(sonium {action_sumsq:.6e}/{:.6e} mfem, report-only) assemble_s(sonium {:.4}/{:.4} mfem) action_s_best(sonium {:.6}/{:.6} mfem)",
        mfem.action_sumsq, assemble_s, mfem.assemble_s, action_s_best, mfem.action_s_best,
    );
    if (l2_error - mfem.l2_error).abs() > 1e-9 + 1e-6 * mfem.l2_error {
        bail!(
            "L2 error mismatch: sonium {l2_error:.6e} vs MFEM {:.6e}",
            mfem.l2_error
        );
    }
    println!(
        "SOLVE-AGREE l2_rel={:.2e} iters(sonium {gmres_iters}/{} mfem) solve_s(sonium {:.4}/{:.4} mfem)",
        rel_diff(l2_error, mfem.l2_error),
        mfem.gmres_iters,
        solve_s,
        mfem.solve_s,
    );
    Ok(())
}

/// Manufactured-solve leg: -Δu - k²u = f with u = (1+i)·s,
/// s = sin(πx)sin(πy)sin(πz), zero on the whole boundary. f = (3π²-k²)u is
/// assembled in real/imaginary parts (the load forms are real) and combined;
/// homogeneous Dirichlet is eliminated on the complex triplets before CSR
/// conversion. Complex GMRES (restart 50, tol 1e-12) mirrors the MFEM
/// real-block GMRES leg.
fn run_solve_leg(
    mesh: &sonium_fem::mesh::Mesh,
    space: &H1Space,
    ndofs: usize,
    wavenumber: Complex64,
    quad_order: usize,
) -> Result<(f64, usize, f64, f64)> {
    use sonium_fem::assembly::assemble_linear;
    use sonium_fem::forms::SourceForm;
    use sonium_solvers::{GmresConfig, gmres};

    let pi = std::f64::consts::PI;
    let factor = 3.0 * pi * pi - wavenumber * wavenumber;
    // f = factor·(1+i)·s; split into real/imag envelopes for the real forms.
    let f_scale = factor * Complex64::new(1.0, 1.0);
    let load_re = SourceForm::new(move |x, y, z| {
        f_scale.re * (pi * x).sin() * (pi * y).sin() * (pi * z).sin()
    });
    let load_im = SourceForm::new(move |x, y, z| {
        f_scale.im * (pi * x).sin() * (pi * y).sin() * (pi * z).sin()
    });
    let rhs_re = assemble_linear(&load_re, space, mesh, 8);
    let rhs_im = assemble_linear(&load_im, space, mesh, 8);
    let mut rhs: Vec<Complex64> = rhs_re
        .iter()
        .zip(rhs_im.iter())
        .map(|(re, im)| Complex64::new(*re, *im))
        .collect();

    // Rebuild the complex triplets (same combination as the action leg) so
    // Dirichlet elimination edits COO entries before CSR conversion.
    let stiffness: TripletMatrix<f64> =
        assemble_bilinear(&LaplacianForm, space, space, mesh, quad_order);
    let mass: TripletMatrix<f64> = assemble_bilinear(&MassForm, space, space, mesh, quad_order);
    let neg_k2 = -(wavenumber * wavenumber);
    let mut triplet = TripletMatrix {
        rows: Vec::with_capacity(stiffness.nnz() + mass.nnz()),
        cols: Vec::with_capacity(stiffness.nnz() + mass.nnz()),
        values: Vec::with_capacity(stiffness.nnz() + mass.nnz()),
        num_rows: ndofs,
        num_cols: ndofs,
    };
    for t in 0..stiffness.nnz() {
        triplet.rows.push(stiffness.rows[t]);
        triplet.cols.push(stiffness.cols[t]);
        triplet
            .values
            .push(Complex64::new(stiffness.values[t], 0.0));
    }
    for t in 0..mass.nnz() {
        triplet.rows.push(mass.rows[t]);
        triplet.cols.push(mass.cols[t]);
        triplet.values.push(neg_k2 * mass.values[t]);
    }

    let coords = space
        .dof_coordinates(mesh)
        .with_context(|| "dof coordinates unavailable".to_string())?;
    let on_boundary = |p: &[f64; 3]| p.iter().any(|c| c.abs() < 1e-12 || (c - 1.0).abs() < 1e-12);
    let boundary: std::collections::HashSet<usize> = coords
        .iter()
        .enumerate()
        .filter(|(_, p)| on_boundary(p))
        .map(|(dof, _)| dof)
        .collect();
    let mut kept_rows = Vec::with_capacity(triplet.rows.len());
    let mut kept_cols = Vec::with_capacity(triplet.cols.len());
    let mut kept_values = Vec::with_capacity(triplet.values.len());
    for t in 0..triplet.rows.len() {
        if boundary.contains(&triplet.rows[t]) || boundary.contains(&triplet.cols[t]) {
            continue;
        }
        kept_rows.push(triplet.rows[t]);
        kept_cols.push(triplet.cols[t]);
        kept_values.push(triplet.values[t]);
    }
    for &dof in &boundary {
        kept_rows.push(dof);
        kept_cols.push(dof);
        kept_values.push(Complex64::new(1.0, 0.0));
        rhs[dof] = Complex64::new(0.0, 0.0);
    }
    let eliminated = TripletMatrix {
        rows: kept_rows,
        cols: kept_cols,
        values: kept_values,
        num_rows: ndofs,
        num_cols: ndofs,
    };
    let system = FemSystem::new(eliminated, rhs);
    let csr = system.matrix.to_csr();

    let config = GmresConfig {
        tolerance: 1e-12,
        max_iterations: 200,
        restart: 50,
        ..Default::default()
    };
    let start = Instant::now();
    let solution = gmres(&csr, &system.rhs, &config);
    let solve_s = start.elapsed().as_secs_f64();

    // True complex L2 error by high-order element quadrature.
    use sonium_fem::mesh::ElementType;
    use sonium_fem::quadrature::QuadratureRule;
    let error_rule = QuadratureRule::new(ElementType::Hexahedron, 8);
    let mut err2: f64 = 0.0;
    for (elem_idx, _) in mesh.elements.iter().enumerate() {
        let dofs: Vec<usize> = space.local_to_global(mesh, elem_idx);
        for qp in &error_rule.points {
            let tab = space.tabulate(mesh, elem_idx, qp.coords[0], qp.coords[1], qp.coords[2]);
            let uh: Complex64 = dofs
                .iter()
                .enumerate()
                .map(|(local, &global)| solution.x[global] * tab.values[local])
                .sum();
            let [x, y, z] = tab.physical_coords;
            let s = (pi * x).sin() * (pi * y).sin() * (pi * z).sin();
            let diff = uh - Complex64::new(s, s);
            err2 += diff.norm_sqr() * qp.weight * tab.det_j;
        }
    }
    let l2_error = err2.max(0.0).sqrt();
    println!(
        "SOLVE ndofs={ndofs} l2_error={l2_error:.6e} gmres_iters={} gmres_res={:.2e} solve_s={solve_s:.4}",
        solution.iterations, solution.residual,
    );
    Ok((l2_error, solution.iterations, solution.residual, solve_s))
}
