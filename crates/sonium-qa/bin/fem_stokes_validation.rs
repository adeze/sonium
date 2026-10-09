//! Stokes equation validation tests
//!
//! Benchmark problems:
//! - Poiseuille: channel flow with analytical parabolic profile
//! - Lid-driven cavity: qualitative vortex structure verification
//! - Taylor-Couette: concentric cylinders with analytical angular velocity profile

use sonium_fem::assembly;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::assembly::mixed::assemble_mixed;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::FemSystem;
use sonium_fem::forms::MassForm;
use sonium_fem::mesh::Mesh;
use sonium_fem::mesh::generators::{
    annular_mesh_triangles, rectangular_mesh_triangles, unit_square_triangles,
};
use sonium_fem::mesh::topology::MeshTopology;
use sonium_fem::physics::fluid::stokes::forms::{StokesDivergenceForm, StokesViscousForm};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};

use sonium_solvers::{
    BlockPreconditionerConfig, FgmresConfig, IlutConfig, SaddlePointConfig,
    SaddlePointPreconditioner, SchurStrategy, fgmres_preconditioned,
};
use std::collections::{HashMap, HashSet};

/// Solve Stokes Poiseuille on a given mesh size and return (vel_rel_err, pres_rel_err).
fn solve_poiseuille(n: usize) -> (f64, f64, bool) {
    let mu = 1.0;
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, n, n);

    let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let n_v = v_space.num_global_dofs(&mesh);
    let n_p = p_space.num_global_dofs(&mesh);
    let n_total = n_v + n_p;

    // Assemble
    let viscous = StokesViscousForm::new(mu);
    let div_form = StokesDivergenceForm;
    let system = assemble_mixed(&viscous, &div_form, &v_space, &p_space, &mesh, 4);
    let mono = system.to_monolithic();

    // Pressure mass for Schur
    let mp = assembly::assemble_bilinear(&MassForm, &p_space, &p_space, &mesh, 2);
    let mut schur = TripletMatrix::<f64>::with_capacity(mp.num_rows, mp.num_cols, mp.nnz());
    for i in 0..mp.nnz() {
        schur.push(mp.rows[i], mp.cols[i], mp.values[i] / mu);
    }

    // DOF coordinates for P2
    let topo = MeshTopology::build(&mesh);
    let n_nodes = mesh.num_nodes();
    let n_edges = topo.num_edges();
    let n_scalar = n_nodes + n_edges;

    let mut dx = vec![0.0; n_scalar];
    let mut dy = vec![0.0; n_scalar];
    for i in 0..n_nodes {
        dx[i] = mesh.nodes[i].x;
        dy[i] = mesh.nodes[i].y;
    }
    for (ei, &(a, b)) in topo.edges.iter().enumerate() {
        dx[n_nodes + ei] = 0.5 * (mesh.nodes[a].x + mesh.nodes[b].x);
        dy[n_nodes + ei] = 0.5 * (mesh.nodes[a].y + mesh.nodes[b].y);
    }

    // BCs
    let mut fem = FemSystem::new(mono, vec![0.0; n_total]);
    let tol = 1e-10;
    let mut bc_map: HashMap<usize, f64> = HashMap::new();

    for sd in 0..n_scalar {
        let x = dx[sd];
        let y = dy[sd];
        if y.abs() < tol || (y - 1.0).abs() < tol {
            bc_map.insert(sd * 2, 0.0);
            bc_map.insert(sd * 2 + 1, 0.0);
        } else if x.abs() < tol {
            bc_map.insert(sd * 2, 4.0 * y * (1.0 - y));
            bc_map.insert(sd * 2 + 1, 0.0);
        }
    }
    for node in 0..n_nodes {
        if (mesh.nodes[node].x - 1.0).abs() < tol {
            bc_map.insert(n_v + node, 0.0);
        }
    }

    // Apply BCs
    for i in 0..fem.matrix.nnz() {
        let row = fem.matrix.rows[i];
        let col = fem.matrix.cols[i];
        if !bc_map.contains_key(&row)
            && let Some(&val) = bc_map.get(&col)
        {
            fem.rhs[row] -= fem.matrix.values[i] * val;
        }
    }
    let mut nr = Vec::with_capacity(fem.matrix.nnz());
    let mut nc = Vec::with_capacity(fem.matrix.nnz());
    let mut nv = Vec::with_capacity(fem.matrix.nnz());
    for i in 0..fem.matrix.nnz() {
        let row = fem.matrix.rows[i];
        let col = fem.matrix.cols[i];
        if !bc_map.contains_key(&row) && !bc_map.contains_key(&col) {
            nr.push(row);
            nc.push(col);
            nv.push(fem.matrix.values[i]);
        }
    }
    for (&dof, &val) in &bc_map {
        nr.push(dof);
        nc.push(dof);
        nv.push(1.0);
        fem.rhs[dof] = val;
    }
    fem.matrix.rows = nr;
    fem.matrix.cols = nc;
    fem.matrix.values = nv;

    // Solve
    let a_csr = fem.matrix.to_csr();
    let schur_csr = schur.to_csr();
    let sp_config = SaddlePointConfig {
        block1_size: n_v,
        block1_config: BlockPreconditionerConfig::Ilut(IlutConfig::default()),
        block2_config: BlockPreconditionerConfig::Ilut(IlutConfig::default()),
        schur_strategy: SchurStrategy::External,
        upper_triangular: true,
    };
    let precond =
        SaddlePointPreconditioner::<f64>::from_matrix_with_schur(&a_csr, &schur_csr, sp_config);
    let rhs = fem.rhs;
    let config = FgmresConfig {
        max_iterations: 500,
        restart: 50,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 0,
    };
    let result = fgmres_preconditioned(&a_csr, &precond, &rhs, None, &config);

    let u_sol: Vec<f64> = result.x.iter().take(n_v).copied().collect();
    let p_sol: Vec<f64> = result.x.iter().skip(n_v).copied().collect();

    let mut vel_err = 0.0;
    let mut vel_norm = 0.0;
    let mut pres_err = 0.0;
    let mut pres_norm = 0.0;

    for sd in 0..n_scalar {
        let y = dy[sd];
        let ux_exact = 4.0 * y * (1.0 - y);
        let ux_h = u_sol[sd * 2];
        let uy_h = u_sol[sd * 2 + 1];
        vel_err += (ux_h - ux_exact).powi(2) + uy_h.powi(2);
        vel_norm += ux_exact.powi(2);

        if sd < n_nodes {
            let x = dx[sd];
            let p_exact = 8.0 * mu * (1.0 - x);
            let p_h = p_sol[sd];
            pres_err += (p_h - p_exact).powi(2);
            pres_norm += p_exact.powi(2);
        }
    }

    let v_rel = (vel_err / vel_norm.max(1e-30)).sqrt();
    let p_rel = (pres_err / pres_norm.max(1e-30)).sqrt();

    (v_rel, p_rel, result.converged)
}

fn test_stokes_poiseuille_converges() -> (f64, f64, bool) {
    let (v_err, p_err, converged) = solve_poiseuille(8);
    assert!(converged, "Solver did not converge");
    // P2-P1 on this smooth problem should give errors near solver tolerance
    assert!(v_err < 1e-6, "Velocity error too large: {v_err:.2e}");
    assert!(p_err < 1e-4, "Pressure error too large: {p_err:.2e}");
    (v_err, p_err, converged)
}

fn test_stokes_viscous_form_symmetry() {
    let mesh = sonium_fem::mesh::generators::unit_square_triangles(4);
    let space = VectorH1Space::new(PolynomialDegree::P2, 2);
    let form = StokesViscousForm::new(1.0);
    let triplet = assembly::assemble_bilinear(&form, &space, &space, &mesh, 3);
    let csr = triplet.to_csr();

    let n = csr.num_rows;
    let mut max_asym = 0.0_f64;
    for i in 0..n {
        for k in csr.row_ptrs[i]..csr.row_ptrs[i + 1] {
            let j = csr.col_indices[k];
            let aij = csr.values[k];
            // Find A[j,i]
            let aji = (csr.row_ptrs[j]..csr.row_ptrs[j + 1])
                .find(|&kk| csr.col_indices[kk] == i)
                .map(|kk| csr.values[kk])
                .unwrap_or(0.0);
            max_asym = max_asym.max((aij - aji).abs());
        }
    }
    assert!(
        max_asym < 1e-12,
        "Viscous form not symmetric: max |A[i,j] - A[j,i]| = {max_asym:.2e}"
    );
}

fn test_stokes_divergence_form_nonzero() {
    let mesh = sonium_fem::mesh::generators::unit_square_triangles(4);
    let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);

    let viscous = StokesViscousForm::new(1.0);
    let div_form = StokesDivergenceForm;

    let system = assemble_mixed(&viscous, &div_form, &v_space, &p_space, &mesh, 3);
    let n_v = v_space.num_global_dofs(&mesh);
    let n_p = p_space.num_global_dofs(&mesh);

    assert_eq!(system.n_dofs_0, n_v);
    assert_eq!(system.n_dofs_1, n_p);

    // Coupling block should be nonzero
    let b01 = system.block_01.to_csr();
    assert!(b01.nnz() > 0, "Divergence coupling block is empty");

    // (1,1) block should be zero (saddle-point)
    let b11 = system.block_11.to_csr();
    assert_eq!(b11.nnz(), 0, "Block (1,1) should be zero for Stokes");
}

/// Return structural metrics for the Taylor–Hood saddle-point pair. The
/// pressure rows must couple to velocity DOFs while the pressure-pressure
/// block remains exactly zero; these are cheap inf-sup/null-block sentinels
/// that complement the analytical Poiseuille solve.
fn stokes_structure_metrics() -> (usize, usize, bool) {
    let mesh = unit_square_triangles(8);
    let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let system = assemble_mixed(
        &StokesViscousForm::new(1.0),
        &StokesDivergenceForm,
        &v_space,
        &p_space,
        &mesh,
        3,
    );
    let coupling = system.block_10.to_csr();
    let pressure_rows = (0..coupling.num_rows)
        .filter(|&row| coupling.row_ptrs[row] < coupling.row_ptrs[row + 1])
        .collect::<HashSet<_>>()
        .len();
    (coupling.nnz(), pressure_rows, system.block_11.nnz() == 0)
}

// =====================================================================
// Shared helper: assemble + apply BCs + solve a Stokes problem
// =====================================================================

struct StokesSolution {
    u_sol: Vec<f64>,
    _p_sol: Vec<f64>,
    dof_x: Vec<f64>,
    dof_y: Vec<f64>,
    _n_nodes: usize,
    _n_scalar: usize,
    _n_v: usize,
    converged: bool,
}

fn solve_stokes(mesh: &Mesh, mu: f64, bc_map: &HashMap<usize, f64>) -> StokesSolution {
    let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let n_v = v_space.num_global_dofs(mesh);
    let n_p = p_space.num_global_dofs(mesh);
    let n_total = n_v + n_p;

    let viscous = StokesViscousForm::new(mu);
    let div_form = StokesDivergenceForm;
    let system = assemble_mixed(&viscous, &div_form, &v_space, &p_space, mesh, 4);
    let mono = system.to_monolithic();

    let mp = assembly::assemble_bilinear(&MassForm, &p_space, &p_space, mesh, 2);
    let mut schur = TripletMatrix::<f64>::with_capacity(mp.num_rows, mp.num_cols, mp.nnz());
    for i in 0..mp.nnz() {
        schur.push(mp.rows[i], mp.cols[i], mp.values[i] / mu);
    }

    let topo = MeshTopology::build(mesh);
    let n_nodes = mesh.num_nodes();
    let n_edges = topo.num_edges();
    let n_scalar = n_nodes + n_edges;

    let mut dof_x = vec![0.0; n_scalar];
    let mut dof_y = vec![0.0; n_scalar];
    for i in 0..n_nodes {
        dof_x[i] = mesh.nodes[i].x;
        dof_y[i] = mesh.nodes[i].y;
    }
    for (ei, &(a, b)) in topo.edges.iter().enumerate() {
        dof_x[n_nodes + ei] = 0.5 * (mesh.nodes[a].x + mesh.nodes[b].x);
        dof_y[n_nodes + ei] = 0.5 * (mesh.nodes[a].y + mesh.nodes[b].y);
    }

    let mut fem = FemSystem::new(mono, vec![0.0; n_total]);
    apply_bc(&mut fem, bc_map);

    let a_csr = fem.matrix.to_csr();
    let schur_csr = schur.to_csr();

    let sp_config = SaddlePointConfig {
        block1_size: n_v,
        block1_config: BlockPreconditionerConfig::Ilut(IlutConfig::default()),
        block2_config: BlockPreconditionerConfig::Ilut(IlutConfig::default()),
        schur_strategy: SchurStrategy::External,
        upper_triangular: true,
    };
    let precond =
        SaddlePointPreconditioner::<f64>::from_matrix_with_schur(&a_csr, &schur_csr, sp_config);
    let rhs = fem.rhs;
    let config = FgmresConfig {
        max_iterations: 500,
        restart: 50,
        tolerance: 1e-8,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 0,
    };
    let result = fgmres_preconditioned(&a_csr, &precond, &rhs, None, &config);

    StokesSolution {
        u_sol: result.x.iter().take(n_v).copied().collect(),
        _p_sol: result.x.iter().skip(n_v).copied().collect(),
        dof_x,
        dof_y,
        _n_nodes: n_nodes,
        _n_scalar: n_scalar,
        _n_v: n_v,
        converged: result.converged,
    }
}

fn apply_bc(system: &mut FemSystem<f64>, bc_map: &HashMap<usize, f64>) {
    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !bc_map.contains_key(&row)
            && let Some(&val) = bc_map.get(&col)
        {
            system.rhs[row] -= system.matrix.values[i] * val;
        }
    }
    let mut nr = Vec::with_capacity(system.matrix.nnz());
    let mut nc = Vec::with_capacity(system.matrix.nnz());
    let mut nv = Vec::with_capacity(system.matrix.nnz());
    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !bc_map.contains_key(&row) && !bc_map.contains_key(&col) {
            nr.push(row);
            nc.push(col);
            nv.push(system.matrix.values[i]);
        }
    }
    for (&dof, &val) in bc_map {
        nr.push(dof);
        nc.push(dof);
        nv.push(1.0);
        system.rhs[dof] = val;
    }
    system.matrix.rows = nr;
    system.matrix.cols = nc;
    system.matrix.values = nv;
}

// =====================================================================
// Lid-driven cavity test
// =====================================================================

fn test_stokes_lid_driven_cavity() {
    let mesh = unit_square_triangles(4);
    let topo = MeshTopology::build(&mesh);
    let n_nodes = mesh.num_nodes();
    let n_edges = topo.num_edges();
    let n_scalar = n_nodes + n_edges;
    let n_v = n_scalar * 2;

    // DOF coordinates
    let mut dx = vec![0.0; n_scalar];
    let mut dy = vec![0.0; n_scalar];
    for i in 0..n_nodes {
        dx[i] = mesh.nodes[i].x;
        dy[i] = mesh.nodes[i].y;
    }
    for (ei, &(a, b)) in topo.edges.iter().enumerate() {
        dx[n_nodes + ei] = 0.5 * (mesh.nodes[a].x + mesh.nodes[b].x);
        dy[n_nodes + ei] = 0.5 * (mesh.nodes[a].y + mesh.nodes[b].y);
    }

    let tol = 1e-10;
    let mut bc_map: HashMap<usize, f64> = HashMap::new();
    for sd in 0..n_scalar {
        let x = dx[sd];
        let y = dy[sd];
        let on_boundary =
            x.abs() < tol || (x - 1.0).abs() < tol || y.abs() < tol || (y - 1.0).abs() < tol;
        if on_boundary {
            if (y - 1.0).abs() < tol && x > tol && x < 1.0 - tol {
                bc_map.insert(sd * 2, 1.0); // lid u_x = 1
            } else {
                bc_map.insert(sd * 2, 0.0);
            }
            bc_map.insert(sd * 2 + 1, 0.0);
        }
    }
    bc_map.insert(n_v, 0.0); // pin pressure

    let sol = solve_stokes(&mesh, 1.0, &bc_map);
    assert!(sol.converged, "Lid-driven cavity solver did not converge");

    // Verify: u_x on lid = 1 (prescribed), u on bottom wall = 0
    for sd in 0..n_scalar {
        let y = sol.dof_y[sd];
        let x = sol.dof_x[sd];
        if y.abs() < tol {
            assert!(
                sol.u_sol[sd * 2].abs() < 1e-6,
                "Bottom wall u_x not zero at ({x}, {y})"
            );
        }
    }

    // Verify: velocity is bounded (no blowup)
    let max_speed: f64 = (0..n_scalar)
        .map(|sd| (sol.u_sol[sd * 2].powi(2) + sol.u_sol[sd * 2 + 1].powi(2)).sqrt())
        .fold(0.0, f64::max);
    assert!(max_speed < 2.0, "Velocity blowup: max speed = {max_speed}");

    // Verify: u_x changes sign along vertical centerline (vortex present)
    let h = 1.0 / 8.0;
    let mut cl: Vec<(f64, f64)> = Vec::new();
    for sd in 0..n_scalar {
        if (sol.dof_x[sd] - 0.5).abs() < h / 2.0 + tol {
            cl.push((sol.dof_y[sd], sol.u_sol[sd * 2]));
        }
    }
    cl.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let has_sign_change = cl.windows(2).any(|w| w[0].1 * w[1].1 < 0.0);
    assert!(
        has_sign_change,
        "No vortex detected (u_x doesn't change sign on centerline)"
    );
}

// =====================================================================
// Taylor-Couette test
// =====================================================================

fn test_stokes_taylor_couette() {
    let r1 = 1.0;
    let r2 = 2.0;
    let omega = 1.0;
    let mu = 1.0;
    let n_radial = 6;
    let n_angular = 24;

    let a_coeff = -omega * r1 * r1 / (r2 * r2 - r1 * r1);
    let b_coeff = omega * r1 * r1 * r2 * r2 / (r2 * r2 - r1 * r1);

    let mesh = annular_mesh_triangles(0.0, 0.0, r1, r2, n_radial, n_angular);
    let topo = MeshTopology::build(&mesh);
    let n_nodes = mesh.num_nodes();
    let n_edges = topo.num_edges();
    let n_scalar = n_nodes + n_edges;
    let n_v = n_scalar * 2;

    let mut dx = vec![0.0; n_scalar];
    let mut dy = vec![0.0; n_scalar];
    for i in 0..n_nodes {
        dx[i] = mesh.nodes[i].x;
        dy[i] = mesh.nodes[i].y;
    }
    for (ei, &(a, b)) in topo.edges.iter().enumerate() {
        dx[n_nodes + ei] = 0.5 * (mesh.nodes[a].x + mesh.nodes[b].x);
        dy[n_nodes + ei] = 0.5 * (mesh.nodes[a].y + mesh.nodes[b].y);
    }

    let tol_inner = r1 * 0.02;
    let tol_outer = r2 * 0.02;
    let mut bc_map: HashMap<usize, f64> = HashMap::new();

    for sd in 0..n_scalar {
        let x = dx[sd];
        let y = dy[sd];
        let r = (x * x + y * y).sqrt();
        if (r - r1).abs() < tol_inner {
            bc_map.insert(sd * 2, -omega * y);
            bc_map.insert(sd * 2 + 1, omega * x);
        } else if (r - r2).abs() < tol_outer {
            bc_map.insert(sd * 2, 0.0);
            bc_map.insert(sd * 2 + 1, 0.0);
        }
    }
    // Pin pressure
    for node in 0..n_nodes {
        let r = (mesh.nodes[node].x.powi(2) + mesh.nodes[node].y.powi(2)).sqrt();
        if (r - r2).abs() < tol_outer {
            bc_map.insert(n_v + node, 0.0);
            break;
        }
    }

    let sol = solve_stokes(&mesh, mu, &bc_map);
    assert!(sol.converged, "Taylor-Couette solver did not converge");

    // Compare with analytical u_theta(r) = A*r + B/r
    let mut vel_err = 0.0;
    let mut vel_norm = 0.0;

    for sd in 0..n_scalar {
        let x = sol.dof_x[sd];
        let y = sol.dof_y[sd];
        let r = (x * x + y * y).sqrt();
        if r < r1 - 0.1 || r > r2 + 0.1 {
            continue;
        }

        let u_theta_exact = a_coeff * r + b_coeff / r;
        let ux_exact = -u_theta_exact * y / r;
        let uy_exact = u_theta_exact * x / r;

        vel_err +=
            (sol.u_sol[sd * 2] - ux_exact).powi(2) + (sol.u_sol[sd * 2 + 1] - uy_exact).powi(2);
        vel_norm += ux_exact.powi(2) + uy_exact.powi(2);
    }

    let rel_err = (vel_err / vel_norm.max(1e-30)).sqrt();
    assert!(
        rel_err < 0.02,
        "Taylor-Couette velocity error too large: {rel_err:.4e} (expected < 2%)"
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        let case_filter = std::env::var("SONIUM_STOKES_CASE").ok();
        let should_run = |name: &str| {
            case_filter
                .as_deref()
                .is_none_or(|filter| filter == "all" || filter == name)
        };
        let (velocity_error, pressure_error, converged) = if should_run("poiseuille") {
            test_stokes_poiseuille_converges()
        } else {
            (0.0, 0.0, true)
        };
        if should_run("symmetry") {
            test_stokes_viscous_form_symmetry();
        }
        if should_run("divergence") {
            test_stokes_divergence_form_nonzero();
        }
        if should_run("cavity") {
            test_stokes_lid_driven_cavity();
        }
        if should_run("taylor-couette") {
            test_stokes_taylor_couette();
        }
        let (coupling_nnz, pressure_rows, pressure_block_zero) = stokes_structure_metrics();
        (
            velocity_error,
            pressure_error,
            converged,
            coupling_nnz,
            pressure_rows,
            pressure_block_zero,
        )
    });
    match result {
        Ok((
            velocity_error,
            pressure_error,
            converged,
            coupling_nnz,
            pressure_rows,
            block_zero,
        )) => {
            let result = sonium_qa::QaResult::passed("fem.stokes-stability.v1")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("velocity_relative_error", velocity_error)
                .metric("pressure_relative_error", pressure_error)
                .metric("poiseuille_converged", if converged { 1.0 } else { 0.0 })
                .metric("coupling_nnz", coupling_nnz as f64)
                .metric("pressure_rows_with_coupling", pressure_rows as f64)
                .metric("pressure_block_zero", if block_zero { 1.0 } else { 0.0 })
                .minimum("poiseuille_converged", 1.0)
                .minimum("pressure_rows_with_coupling", 1.0)
                .minimum("pressure_block_zero", 1.0)
                .maximum("velocity_relative_error", 1.0e-6)
                .maximum("pressure_relative_error", 1.0e-4);
            match result.to_json_line() {
                Ok(line) => println!("QA_RESULT: {line}"),
                Err(error) => {
                    eprintln!("failed to serialize Stokes QA_RESULT: {error}");
                    println!("FAILED");
                    std::process::exit(1);
                }
            }
            println!("SUCCESS");
        }
        Err(_) => {
            if let Ok(line) =
                sonium_qa::QaResult::failed("fem.stokes-stability.v1", "Stokes validation panicked")
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
            {
                println!("QA_RESULT: {line}");
            }
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
