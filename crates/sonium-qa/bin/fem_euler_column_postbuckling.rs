//! QA: Euler column buckling and post-buckling via arc length.
//!
//! Compresses a slender hyperelastic cantilever column (K = 2, no point
//! pins) with a tiny lateral imperfection using the production
//! `fem_riks_solve_with_bcs_and_load` driver, then checks the knee load
//! against Euler (plane-strain adjusted) and the post-buckling slope
//! against the cantilever elastica `P/Pcr - 1 ~ (pi^2/32) (tip/L)^2`.
//! Point-pinned ends are avoided deliberately: a continuum pin funnels the
//! full axial reaction through one node and its NeoHookean hot spot
//! pollutes the path (see the worker report).
//!
//! P1-B-W1-1 adds a pinned-pinned (K = 1) path that replaces the point
//! pins with rigid-face MPCs built from Worker 2's general constraint
//! machinery (`interpolate_slave` over `MpcSystem`, read-only): each end
//! face interpolates axial motion between its center and top masters,
//! which is exact rigid kinematics at finite rotation and spreads the
//! axial reaction over the face, while lateral motion stays free (see
//! `rigid_face_constraints`). The pinned path runs MPC-condensed Newton
//! load stepping (the arc-length driver takes Dirichlet-only BCs) and
//! gates the knee plus the pinned elastica
//! `P/Pcr - 1 ~ (pi^2/8) (mid/L)^2`.

// Rust guideline compliant 2026-02-21

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::{MpcSystem, MultiPointConstraint, interpolate_slave};
use sonium_fem::forms::{LocalMatrix, LocalVector, NonlinearForm};
use sonium_fem::mesh::{Mesh, rectangular_mesh_quads};
use sonium_fem::physics::solid::joints::RigidBody;
use sonium_fem::physics::solid::nonlinear_solid::{HyperelasticForm, NeoHookean};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::solver::nonlinear::FemNewtonConfig;
use sonium_fem::solver::nonlinear::arc_length::{
    ArcLengthConfig, fem_riks_solve_with_bcs_and_load,
};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_qa::column_euler_critical_load;
use sonium_solvers::{CsrMatrix, LinearSolveConfig, TripletMatrix, solve_real};
use std::f64::consts::PI;

/// Solve a 3x3 system by Gaussian elimination with partial pivoting.
fn solve_3x3(mut ata: [[f64; 3]; 3], mut aty: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let pivot = (col..3)
            .max_by(|a, b| ata[*a][col].abs().total_cmp(&ata[*b][col].abs()))
            .unwrap_or(col);
        if ata[pivot][col].abs() < 1e-18 {
            return None;
        }
        ata.swap(col, pivot);
        aty.swap(col, pivot);
        for row in col + 1..3 {
            let factor = ata[row][col] / ata[col][col];
            for k in col..3 {
                ata[row][k] -= factor * ata[col][k];
            }
            aty[row] -= factor * aty[col];
        }
    }
    let mut solution = [0.0; 3];
    for i in (0..3).rev() {
        let mut residual = aty[i];
        for k in i + 1..3 {
            residual -= ata[i][k] * solution[k];
        }
        solution[i] = residual / ata[i][i];
    }
    Some(solution)
}

fn run_cantilever_k2() {
    let (length, height): (f64, f64) = (1.0, 0.02);
    let (young, poisson): (f64, f64) = (200.0e9, 0.3);
    let second_moment = height.powi(3) / 12.0;
    let pcr_euler = column_euler_critical_load(young, second_moment, length, 2.0);
    // Wide-strip plane strain stiffens bending by 1/(1-nu^2).
    let pcr_plane_strain = pcr_euler / (1.0 - poisson.powi(2));
    println!("euler Pcr cantilever (beam):        {pcr_euler:.6e} N");
    println!("euler Pcr cantilever (plane strain): {pcr_plane_strain:.6e} N");

    // P1 quads: P2 slender tangents stall every available inner solver
    // (ILU/Schwarz) and trip the AMG symmetry gate on assembly roundoff,
    // so the column runs P1 with a fine mesh and a locking-aware gate.
    let mesh = rectangular_mesh_quads(0.0, length, 0.0, height, 160, 8);
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let form = HyperelasticForm::new(NeoHookean::from_young(young, poisson, 2));
    let n_dofs = space.num_global_dofs(&mesh);
    println!("dofs: {n_dofs}");

    // Clamped base (full face): a physical built-in end with no point pins.
    let mut dirichlet = Vec::new();
    let mut top_x_dofs = Vec::new();
    for (node, point) in mesh.nodes.iter().enumerate() {
        if point.x.abs() < 1e-12 {
            dirichlet.push((2 * node, 0.0));
            dirichlet.push((2 * node + 1, 0.0));
        }
        if (point.x - length).abs() < 1e-12 {
            top_x_dofs.push(2 * node);
        }
    }
    // Reference load: Pcr in axial compression plus a 1e-3 lateral bias.
    // The bias transcritically unfolds the pitchfork (rounding the knee for
    // the iterative solver); the Koiter fit below separates its 1/delta tail
    // from the elastica curvature instead of forcing a through-origin slope.
    let mut reference = vec![0.0; n_dofs];
    for dof in &top_x_dofs {
        reference[*dof] = -pcr_plane_strain / top_x_dofs.len() as f64;
    }
    let lateral_total = 1e-3 * pcr_plane_strain;
    let y_count = mesh.nodes.len() as f64;
    for node in 0..mesh.nodes.len() {
        reference[2 * node + 1] += lateral_total / y_count;
    }

    // The slender tangent trips the AMG symmetry gate on assembly roundoff
    // and stagnates restarted GMRES(30); wide-restart Schwarz FGMRES with
    // inexact-Newton tolerances converges through the rounded knee.
    let mut fem_config = FemNewtonConfig::default();
    fem_config.inner_solver.solver = sonium_solvers::LinearSolverType::FgmresSchwarz;
    fem_config.inner_solver.tolerance = 1e-7;
    fem_config.inner_solver.restart = 500;
    fem_config.inner_solver.max_iterations = 3000;
    let arc = ArcLengthConfig {
        initial_step: 2e-3,
        min_step: 1e-7,
        max_step: 3e-3,
        max_steps: 320,
        corrector_tolerance: 1e-8,
        psi: 0.03,
        ..Default::default()
    };
    let path = fem_riks_solve_with_bcs_and_load(
        &form,
        &space,
        &mesh,
        &dirichlet,
        &reference,
        &vec![0.0; n_dofs],
        &fem_config,
        &arc,
    )
    .expect("column trace must advance");
    println!(
        "steps accepted/rejected: {}/{} ({:?})",
        path.steps_accepted, path.steps_rejected, path.stopped
    );

    // Lateral amplitude per point (max |uy|, attained at midspan).
    let amplitude = |u: &[f64]| {
        u.iter()
            .skip(1)
            .step_by(2)
            .map(|value| value.abs())
            .fold(0.0, f64::max)
    };
    println!("step lambda delta/L iters residual");
    for (index, point) in path.points.iter().enumerate() {
        if index % 10 == 0 || index + 1 == path.points.len() {
            println!(
                "{} {:.6} {:.6} {} {:.2e}",
                index,
                point.lambda,
                amplitude(&point.u) / length,
                point.newton_iterations,
                point.residual
            );
        }
    }
    // The imperfect path rises monotonically in lambda, so the buckling
    // load is read where the knee develops (delta/L = 0.005), not as a peak.
    let knee = path
        .points
        .iter()
        .find(|point| amplitude(&point.u) / length >= 0.005)
        .map(|point| point.lambda);
    match knee {
        Some(lambda) => {
            println!("traced knee lambda at delta/L=0.005: {lambda:.6}");
            assert!(
                (lambda - 1.0).abs() <= 0.15,
                "knee load misses plane-strain Euler by more than 15%"
            );
        }
        None => panic!("trace never developed a buckling knee"),
    }

    let tip = path.points.last().unwrap();
    let tip_delta = amplitude(&tip.u) / length;
    println!("tip delta/L: {tip_delta:.6}, tip lambda: {:.6}", tip.lambda);
    println!("centerline x uy tip-shape:");
    let mid_row = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, p)| (p.y - height / 2.0).abs() < 1e-12)
        .map(|(n, _)| n)
        .collect::<Vec<_>>();
    let mut row: Vec<(f64, f64)> = mid_row
        .iter()
        .map(|n| (mesh.nodes[*n].x, tip.u[2 * n + 1]))
        .collect();
    row.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (x, uy) in row.iter().step_by(16) {
        println!("{x:.4} {uy:.6e}");
    }
    let top_ux: f64 = top_x_dofs.iter().map(|d| tip.u[*d]).sum::<f64>() / top_x_dofs.len() as f64;
    println!("mean top-face ux (shortening): {top_ux:.6e}");
    println!(
        "bowing shortening estimate: {:.6e}",
        tip_delta * tip_delta * PI.powi(2) / 16.0 * length
    );
    assert!(tip_delta > 0.030, "trace never reached post-buckling");

    // Koiter unfolding fit past the knee: lambda - 1 = A + B/delta + C delta^2
    // with delta = tip/L. A captures the discrete bifurcation offset
    // (locking), B < 0 the decaying imperfection tail (~ -b*eps), and C the
    // elastica curvature (cantilever: pi^2/32, the pinned pi^2/8 in (tip/2L)^2
    // since the cantilever is the symmetric half of a pinned column of 2L).
    let rows: Vec<(f64, f64)> = path
        .points
        .iter()
        .map(|point| (amplitude(&point.u) / length, point.lambda - 1.0))
        .filter(|(delta, _)| *delta > 0.008)
        .collect();
    assert!(
        rows.len() >= 10,
        "too few post-buckling points for a Koiter fit"
    );
    // Normal equations for [1, 1/delta, delta^2].
    let mut ata = [[0.0; 3]; 3];
    let mut aty = [0.0; 3];
    for (delta, value) in &rows {
        let basis = [1.0, 1.0 / delta, delta * delta];
        for i in 0..3 {
            aty[i] += basis[i] * value;
            for j in 0..3 {
                ata[i][j] += basis[i] * basis[j];
            }
        }
    }
    let koiter = solve_3x3(ata, aty).expect("koiter fit must be solvable");
    let elastica = PI.powi(2) / 32.0;
    println!(
        "koiter fit: A={:.4e} B={:.4e} C={:.4} (elastica C={elastica:.4})",
        koiter[0], koiter[1], koiter[2]
    );
    assert!(
        koiter[0].abs() <= 0.08,
        "bifurcation offset exceeds 8% (locking out of hand)"
    );
    assert!(
        koiter[1] < 0.0 && koiter[1].abs() <= 5e-3,
        "imperfection tail has the wrong sign or scale"
    );
    assert!(
        (koiter[2] - elastica).abs() <= 3.0 * elastica,
        "post-buckling curvature misses the cantilever elastica by more than 3x"
    );
    println!("PASS: cantilever K=2 path");
}

// ---------------------------------------------------------------------------
// P1-B-W1-1: pinned-pinned path with rigid-face MPC.
// ---------------------------------------------------------------------------

/// Assemble hyperelastic residual and tangent through the production form.
///
/// Mirrors the solver's element loop (`NonlinearForm` kernels with the
/// same quadrature) so the MPC Newton below solves production physics.
fn assemble_hyperelastic<F: NonlinearForm<f64>>(
    form: &F,
    space: &VectorH1Space,
    mesh: &Mesh,
    solution: &[f64],
    quad_order: usize,
) -> (Vec<f64>, TripletMatrix<f64>) {
    let n_dofs = space.num_global_dofs(mesh);
    let mut residual = vec![0.0; n_dofs];
    let mut tangent = TripletMatrix::new(n_dofs, n_dofs);
    for elem_idx in 0..mesh.elements.len() {
        let elem = &mesh.elements[elem_idx];
        let dofs = space.local_to_global(mesh, elem_idx);
        let local_sol: Vec<f64> = dofs.iter().map(|&d| solution[d]).collect();
        let quad = QuadratureRule::new(elem.element_type, quad_order);
        let mut local_vec = LocalVector::zeros(dofs.len());
        let mut local_mat = LocalMatrix::zeros(dofs.len());
        for qp in quad.iter() {
            let tab = space.tabulate(mesh, elem_idx, qp.xi(), qp.eta(), qp.zeta());
            form.element_residual(&tab, &local_sol, qp.weight, &mut local_vec);
            form.element_tangent(&tab, &tab, &local_sol, qp.weight, &mut local_mat);
        }
        for (i, &dof) in dofs.iter().enumerate() {
            residual[dof] += local_vec.values[i];
            for (j, &col) in dofs.iter().enumerate() {
                tangent.push(dof, col, local_mat.get(i, j));
            }
        }
    }
    (residual, tangent)
}

/// Rigid-face constraints for one column end (2-DOF continuum nodes).
///
/// The face interpolates axial motion between its center and top masters
/// (`ux = (1-w) ux_c + w ux_t` with the span fraction `w`). Axial
/// interpolation is exact rigid kinematics at finite rotation (rotation
/// about z keeps `ux` linear in `y`, carried by `ux_t - ux_c`), so the
/// face rotates freely and spreads the axial load over its nodes.
///
/// Lateral motion stays deliberately free: sharing `uy` (`uy = uy_c`)
/// is only infinitesimally rigid — finite rotation needs
/// `uy = (y-yc)(cos t - 1)` — and forcing `uy = 0` locks the face
/// against rotation with spurious energy past the knee (measured
/// post-buckling slope 119x the pinned elastica before this fix).
fn rigid_face_constraints(
    face: &[(usize, f64)],
    center: usize,
    center_y: f64,
    top: usize,
    top_y: f64,
) -> Vec<MultiPointConstraint> {
    let mut out = Vec::new();
    for &(node, y) in face {
        if node == center || node == top {
            continue;
        }
        // Axial motion interpolates between the center and top masters.
        let w = (y - center_y) / (top_y - center_y);
        out.push(interpolate_slave(
            2 * node,
            &[(2 * center, 1.0 - w), (2 * top, w)],
        ));
    }
    out
}

/// Cross-check the 2-DOF face weights against rigid-body kinematics.
///
/// Builds Worker 2's 6-DOF `RigidBody` on the same face, prolongs a
/// master motion `(ux, rz)`, and requires the slave axial translations
/// to match the interpolation formula to roundoff. Lateral motion is
/// covered only at small rotation: the production constraint frees
/// `uy` for finite-rotation exactness (see above), so the oracle
/// checks `ux` exactly and `uy` in the linear regime.
fn check_rigid_face_oracle(face: &[(usize, f64)], center_y: f64, face_x: f64) {
    let mut local: Vec<(usize, f64)> = face.to_vec();
    local.sort_by(|a, b| a.1.total_cmp(&b.1));
    let center_local = local
        .iter()
        .position(|(_, y)| (y - center_y).abs() < 1e-12)
        .expect("face center must exist");
    let positions: Vec<[f64; 3]> = local.iter().map(|(_, y)| [face_x, *y, 0.0]).collect();
    let slaves: Vec<usize> = (0..local.len()).filter(|&i| i != center_local).collect();
    let body = RigidBody::new(center_local, slaves, positions);
    let mpc = MpcSystem::new(6 * local.len(), &body.constraints()).expect("oracle MPC builds");
    // Master motion ux = 0.05, rz = 0.03; everything else zero.
    let order = mpc.reduced_full_dofs().to_vec();
    let mut reduced = vec![0.0; mpc.num_reduced_dofs()];
    for (i, &dof) in order.iter().enumerate() {
        if dof == 6 * center_local {
            reduced[i] = 0.05;
        } else if dof == 6 * center_local + 5 {
            reduced[i] = 0.03;
        }
    }
    let full = mpc.recover_real(&reduced).expect("oracle recovers");
    for (i, &(_, y)) in local.iter().enumerate() {
        if i == center_local {
            continue;
        }
        // RigidBody linearizes rotation: ux matches the interpolation
        // formula exactly, uy stays zero (linear regime).
        let expected_ux = 0.05 - (y - center_y) * 0.03;
        assert!(
            (full[6 * i] - expected_ux).abs() < 1e-12,
            "oracle slave {i} ux mismatch"
        );
        assert!(
            full[6 * i + 1].abs() < 1e-12,
            "oracle slave {i} uy mismatch"
        );
    }
    println!(
        "rigid-face oracle: {} slaves match rigid kinematics",
        local.len() - 1
    );
}

/// Extract the free-reduced submatrix of a condensed tangent.
fn free_submatrix(csr: &CsrMatrix<f64>, free: &[usize]) -> CsrMatrix<f64> {
    let mut triplet = TripletMatrix::new(free.len(), free.len());
    for (i, &row) in free.iter().enumerate() {
        for (col, value) in csr.row_entries(row) {
            if let Some(j) = free.iter().position(|&d| d == col) {
                triplet.push(i, j, value);
            }
        }
    }
    triplet.to_csr()
}

/// Fixed inputs of the displacement-controlled MPC Newton solve.
struct DispNewtonContext<'a, F: NonlinearForm<f64>> {
    form: &'a F,
    space: &'a VectorH1Space,
    mesh: &'a Mesh,
    mpc: &'a MpcSystem,
    free_reduced: &'a [usize],
    constraint_free: usize,
    reference: &'a [f64],
    inner: &'a LinearSolveConfig,
}

/// MPC-condensed Newton solve at one prescribed midspan deflection.
///
/// Solves the bordered system for `(u, lambda)` with `uy_mid = target`:
/// `K a = -(R - l F)`, `K b = F`, `dl = -(c + a_j)/b_j`,
/// `du = a + dl b`. Returns the load factor, or `None` (no panic) when
/// the step fails so the caller can halve the deflection increment.
fn mpc_newton_solve_disp<F: NonlinearForm<f64>>(
    ctx: &DispNewtonContext<F>,
    target: f64,
    solution: &mut [f64],
    lambda: f64,
) -> Option<f64> {
    let f_norm: f64 = ctx.reference.iter().map(|f| f * f).sum::<f64>().sqrt();
    let mut load = lambda;
    for _ in 0..25 {
        let (mut residual, tangent) =
            assemble_hyperelastic(ctx.form, ctx.space, ctx.mesh, solution, 2);
        for (r, f) in residual.iter_mut().zip(ctx.reference.iter()) {
            *r -= load * f;
        }
        let condensed = ctx.mpc.condense_real_rhs(&residual).ok()?;
        let norm: f64 = ctx
            .free_reduced
            .iter()
            .map(|&i| condensed[i] * condensed[i])
            .sum::<f64>()
            .sqrt();
        let gap = solution[mid_dof(ctx.mesh, 1.0, 0.02)] - target;
        if !norm.is_finite() || !gap.is_finite() {
            return None;
        }
        if norm <= 1e-8 * (1.0 + load * f_norm).max(1.0) && gap.abs() <= 1e-10 {
            return Some(load);
        }
        let matrix = ctx.mpc.condense_real_matrix(&tangent.to_csr()).ok()?;
        let free_matrix = free_submatrix(&matrix, ctx.free_reduced);
        let rhs_a: Vec<f64> = ctx.free_reduced.iter().map(|&i| -condensed[i]).collect();
        let condensed_f = ctx.mpc.condense_real_rhs(ctx.reference).ok()?;
        let rhs_b: Vec<f64> = ctx.free_reduced.iter().map(|&i| condensed_f[i]).collect();
        let a = match solve_real(&free_matrix, &rhs_a, ctx.inner) {
            Ok(result) if result.converged => result.x,
            _ => return None,
        };
        let b = match solve_real(&free_matrix, &rhs_b, ctx.inner) {
            Ok(result) if result.converged => result.x,
            _ => return None,
        };
        if b[ctx.constraint_free].abs() < 1e-300 {
            return None;
        }
        let dlambda = (-gap - a[ctx.constraint_free]) / b[ctx.constraint_free];
        let mut reduced = vec![0.0; ctx.mpc.num_reduced_dofs()];
        for (k, &i) in ctx.free_reduced.iter().enumerate() {
            reduced[i] = a[k] + dlambda * b[k];
        }
        let delta = ctx.mpc.recover_real(&reduced).ok()?;
        for (u, d) in solution.iter_mut().zip(delta.iter()) {
            *u += d;
        }
        load += dlambda;
    }
    None
}

/// Full lateral DOF of the node nearest the strip center.
fn mid_dof(mesh: &Mesh, length: f64, height: f64) -> usize {
    let (node, _) = mesh
        .nodes
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (a.x - length / 2.0).abs() + (a.y - height / 2.0).abs();
            let db = (b.x - length / 2.0).abs() + (b.y - height / 2.0).abs();
            da.total_cmp(&db)
        })
        .expect("mesh must have a center node");
    2 * node + 1
}

fn run_pinned_pinned_k1() {
    let (length, height): (f64, f64) = (1.0, 0.02);
    let (young, poisson): (f64, f64) = (200.0e9, 0.3);
    let second_moment = height.powi(3) / 12.0;
    let pcr_euler = column_euler_critical_load(young, second_moment, length, 1.0);
    let pcr_plane_strain = pcr_euler / (1.0 - poisson.powi(2));
    println!("euler Pcr pinned (beam):        {pcr_euler:.6e} N");
    println!("euler Pcr pinned (plane strain): {pcr_plane_strain:.6e} N");

    let mesh = rectangular_mesh_quads(0.0, length, 0.0, height, 160, 8);
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let form = HyperelasticForm::new(NeoHookean::from_young(young, poisson, 2));
    let n_dofs = space.num_global_dofs(&mesh);
    println!("pinned dofs: {n_dofs}");

    // End faces with (node, y) rows; center and top masters per face.
    let mut bottom = Vec::new();
    let mut top = Vec::new();
    for (node, point) in mesh.nodes.iter().enumerate() {
        if point.x.abs() < 1e-12 {
            bottom.push((node, point.y));
        }
        if (point.x - length).abs() < 1e-12 {
            top.push((node, point.y));
        }
    }
    assert!(!bottom.is_empty() && !top.is_empty(), "faces must exist");
    let pick = |face: &[(usize, f64)]| {
        let center = *face
            .iter()
            .min_by(|a, b| {
                (a.1 - height / 2.0)
                    .abs()
                    .total_cmp(&(b.1 - height / 2.0).abs())
            })
            .unwrap();
        let top = *face.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        (center, top)
    };
    let ((bc, bcy), (bt, bty)) = pick(&bottom);
    let ((tc, tcy), (tt, tty)) = pick(&top);
    check_rigid_face_oracle(&bottom, bcy, 0.0);

    let mut constraints = rigid_face_constraints(&bottom, bc, bcy, bt, bty);
    constraints.extend(rigid_face_constraints(&top, tc, tcy, tt, tty));
    let mpc = MpcSystem::new(n_dofs, &constraints).expect("face MPC builds");
    println!(
        "face MPC: {} full -> {} reduced DOFs",
        mpc.num_full_dofs(),
        mpc.num_reduced_dofs()
    );
    // Prolongation spot check: a mid-face slave interpolates exactly.
    let mid_slave = bottom
        .iter()
        .find(|(node, _)| *node != bc && *node != bt)
        .expect("face needs an interior slave");
    let w = (mid_slave.1 - bcy) / (bty - bcy);
    let row = mpc.prolongation_row(2 * mid_slave.0);
    assert_eq!(row.len(), 2, "axial slave needs two masters");
    let order = mpc.reduced_full_dofs();
    let mut weights = row
        .iter()
        .map(|&(reduced, coeff)| (order[reduced], coeff.re))
        .collect::<Vec<_>>();
    weights.sort_by_key(|&(dof, _)| dof);
    assert_eq!(weights[0].0, 2 * bc);
    assert!((weights[0].1 - (1.0 - w)).abs() < 1e-12);
    assert_eq!(weights[1].0, 2 * bt);
    assert!((weights[1].1 - w).abs() < 1e-12);

    // Pin the bottom center (ux, uy) and the top center lateral (uy);
    // the top center axial master stays free to shorten.
    let pinned_full = [2 * bc, 2 * bc + 1, 2 * tc + 1];
    let mut pinned_reduced = Vec::new();
    for &full in &pinned_full {
        let reduced = order
            .iter()
            .position(|&dof| dof == full)
            .expect("pinned master must be reduced");
        pinned_reduced.push(reduced);
    }
    let free_reduced: Vec<usize> = (0..mpc.num_reduced_dofs())
        .filter(|i| !pinned_reduced.contains(i))
        .collect();

    // Reference load: Pcr axial over the top face plus the 1e-3 bias.
    let mut reference = vec![0.0; n_dofs];
    for &(node, _) in &top {
        reference[2 * node] = -pcr_plane_strain / top.len() as f64;
    }
    let lateral_total = 1e-3 * pcr_plane_strain;
    for node in 0..mesh.nodes.len() {
        reference[2 * node + 1] += lateral_total / mesh.nodes.len() as f64;
    }
    let f_norm: f64 = reference.iter().map(|f| f * f).sum::<f64>().sqrt();
    // Condensed axial resultant must equal Pcr (virtual work check).
    let condensed = mpc.condense_real_rhs(&reference).expect("rhs condenses");
    let top_ux = order
        .iter()
        .position(|&dof| dof == 2 * tc)
        .expect("top axial master must be reduced");
    assert!(
        (condensed[top_ux] + pcr_plane_strain).abs() / pcr_plane_strain < 1e-9,
        "axial resultant must condense to Pcr"
    );
    let _ = f_norm;

    let inner = LinearSolveConfig {
        solver: sonium_solvers::LinearSolverType::FgmresSchwarz,
        max_iterations: 3000,
        tolerance: 1e-7,
        restart: 500,
        exact_residual_check_interval: 0,
        verbosity: 0,
    };
    // Load stepping with halving; the imperfect path has no limit point.
    // The 0.01 cap plus the monotonicity guard keep Newton on the flat
    // buckled branch: coarse steps jump the rounded knee and converge
    // onto the unstable straight branch (amplitude collapses).
    let amplitude = |u: &[f64]| {
        u.iter()
            .skip(1)
            .step_by(2)
            .map(|value| value.abs())
            .fold(0.0, f64::max)
    };
    let mut solution = vec![0.0; n_dofs];
    let mut lambda = 0.0_f64;
    // Displacement control on the center node: the post-buckling
    // branch is nearly flat in lambda, so the deflection targets below
    // sample it uniformly where load control would jump or stall.
    let mid = mid_dof(&mesh, length, height);
    let mid_reduced = order
        .iter()
        .position(|&dof| dof == mid)
        .expect("center node must be reduced");
    let constraint_free = free_reduced
        .iter()
        .position(|&i| i == mid_reduced)
        .expect("center node must be free");
    let ctx = DispNewtonContext {
        form: &form,
        space: &space,
        mesh: &mesh,
        mpc: &mpc,
        free_reduced: &free_reduced,
        constraint_free,
        reference: &reference,
        inner: &inner,
    };
    let mut points = vec![(0.0, solution.clone())];
    let mut guard = 0;
    let mut target = 0.0_f64;
    let mut step = 0.002_f64;
    while target < 0.05 {
        guard += 1;
        assert!(guard <= 400, "pinned trace stalled at delta={target:.4}");
        let next = (target + step).min(0.05);
        let mut trial = solution.clone();
        match mpc_newton_solve_disp(&ctx, next, &mut trial, lambda) {
            Some(value) => {
                lambda = value;
                target = next;
                solution = trial;
                points.push((lambda, solution.clone()));
                step = (step * 1.25).min(0.004);
                if points.len() % 10 == 0 {
                    println!(
                        "pinned step {}: delta={target:.4} lambda={lambda:.6}",
                        points.len()
                    );
                }
            }
            None => {
                step *= 0.5;
                assert!(step >= 1e-5, "pinned Newton fails at delta={target:.4}");
            }
        }
    }
    println!("pinned points: {}", points.len());
    let knee = points
        .iter()
        .find(|(_, u)| amplitude(u) / length >= 0.005)
        .map(|(lambda, _)| *lambda);
    match knee {
        Some(value) => {
            println!("pinned knee lambda at delta/L=0.005: {value:.6}");
            assert!(
                (value - 1.0).abs() <= 0.15,
                "pinned knee misses plane-strain Euler by more than 15%"
            );
        }
        None => panic!("pinned trace never developed a buckling knee"),
    }
    let (last_lambda, last) = points.last().unwrap();
    let mid_delta = amplitude(last) / length;
    println!("pinned mid delta/L: {mid_delta:.6}");
    // Axial diagnostics: free shortening vs the inextensible estimate.
    let inextensible = PI.powi(2) * mid_delta.powi(2) / 4.0 * length
        + last_lambda * pcr_plane_strain * length / (young * height);
    println!(
        "pinned top ux_c={:.6e} ux_t={:.6e} bottom ux_t={:.6e} inextensible={:.6e}",
        last[2 * tc],
        last[2 * tt],
        last[2 * bt],
        -inextensible
    );
    let (final_res, _) = assemble_hyperelastic(&form, &space, &mesh, last, 2);
    // Global equilibrium: pin reactions balance the applied resultants.
    let mut pin_r = [0.0; 2];
    for node in 0..mesh.nodes.len() {
        pin_r[0] -= final_res[2 * node] - last_lambda * reference[2 * node];
        pin_r[1] -= final_res[2 * node + 1] - last_lambda * reference[2 * node + 1];
    }
    println!(
        "pinned pin reactions: Rx={:.6e} Ry={:.6e} (P={:.6e}, Q={:.6e})",
        pin_r[0],
        pin_r[1],
        last_lambda * pcr_plane_strain,
        last_lambda * 1e-3 * pcr_plane_strain
    );
    println!("pinned trace (lambda, delta/L):");
    for (lambda, u) in points.iter().step_by(5) {
        println!("{lambda:.6} {:.6e}", amplitude(u) / length);
    }
    println!("pinned centerline x uy final-shape:");
    let mut row: Vec<(f64, f64)> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, p)| (p.y - height / 2.0).abs() < 1e-12)
        .map(|(n, p)| (p.x, last[2 * n + 1]))
        .collect();
    row.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (x, uy) in row.iter().step_by(16) {
        println!("{x:.4} {uy:.6e}");
    }
    assert!(
        mid_delta > 0.030,
        "pinned trace never reached post-buckling"
    );

    // Koiter fit against the pinned elastica pi^2/8 in (mid/L)^2.
    let rows: Vec<(f64, f64)> = points
        .iter()
        .map(|(lambda, u)| (amplitude(u) / length, lambda - 1.0))
        .filter(|(delta, _)| *delta > 0.008)
        .collect();
    assert!(rows.len() >= 10, "too few pinned points for a Koiter fit");
    let mut ata = [[0.0; 3]; 3];
    let mut aty = [0.0; 3];
    for (delta, value) in &rows {
        let basis = [1.0, 1.0 / delta, delta * delta];
        for i in 0..3 {
            aty[i] += basis[i] * value;
            for j in 0..3 {
                ata[i][j] += basis[i] * basis[j];
            }
        }
    }
    let koiter = solve_3x3(ata, aty).expect("pinned koiter fit must be solvable");
    let elastica = PI.powi(2) / 8.0;
    println!(
        "pinned koiter: A={:.4e} B={:.4e} C={:.4} (elastica C={elastica:.4})",
        koiter[0], koiter[1], koiter[2]
    );
    assert!(
        koiter[0].abs() <= 0.08,
        "pinned bifurcation offset exceeds 8%"
    );
    assert!(
        koiter[1] < 0.0 && koiter[1].abs() <= 5e-3,
        "pinned imperfection tail has the wrong sign or scale"
    );
    assert!(
        (koiter[2] - elastica).abs() <= 3.0 * elastica,
        "pinned curvature misses the elastica by more than 3x"
    );
    println!("PASS: pinned-pinned K=1 path");
}

fn main() {
    match run() {
        Ok(()) => println!("SUCCESS: qa-fem-euler-column-postbuckling"),
        Err(error) => {
            eprintln!("FAILED: qa-fem-euler-column-postbuckling: {error}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(), String> {
    // EULER_PINNED_ONLY=1 skips the cantilever arc trace to iterate fast
    // on the pinned path; the default full run gates both paths.
    if std::env::var("EULER_PINNED_ONLY").as_deref() != Ok("1") {
        run_cantilever_k2();
    }
    run_pinned_pinned_k1();
    println!("PASS: qa-fem-euler-column-postbuckling");
    Ok(())
}
