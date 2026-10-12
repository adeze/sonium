//! Scordelis-Lo roof benchmark with MITC4 quad shells.
//!
//! Cylindrical roof (R = 25, L = 50, t = 0.25) under self weight q = 90 per
//! unit area, diaphragm-supported curved ends, free longitudinal edges.
//! Quarter model with apex and midspan symmetry. Target: vertical deflection
//! at the free-edge midpoint w = -0.3024.
//!
//! Reference: Scordelis, A. C. and Lo, K. S. (1969), "Computer analysis of
//! cylindrical shells", J. ACI 61; standard target value 0.3024 after
//! MacNeal, R. H. and Harder, R. L. (1985), "A proposed standard set of
//! problems to test finite element accuracy", FEAD 1.
//!
//! The C3 OpenCourant case uses the shared `sonium_qa::rad` extended API
//! (`shell_scordelis_lo_case`); engine-gated runs live in
//! `qa-fem-opencourant-shell-scordelis-lo`.

// Rust guideline compliant 2026-02-21

use sonium_fem::mesh::Point;
use sonium_fem::physics::solid::shell::{
    QuadShellElement, ShellSection, assemble_quad_shell_stiffness,
};
use sonium_qa::rad::{assert_probe_close, shell_scordelis_lo_case, write_extended_starter};
use sonium_solvers::direct::SparseLuFactorization;

const RADIUS: f64 = 25.0;
const LENGTH: f64 = 50.0;
const THICK: f64 = 0.25;
const E_MOD: f64 = 4.32e8;
const LOAD: f64 = 90.0;
const HALF_ANGLE_DEG: f64 = 40.0;
const TARGET_W: f64 = -0.3024;

fn build_quarter_model(n_theta: usize, n_z: usize) -> (Vec<Point>, Vec<[usize; 4]>) {
    // Quarter: theta in [0, 40 deg] from apex, z in [0, 25] from diaphragm.
    let half_angle = HALF_ANGLE_DEG.to_radians();
    let mut points = Vec::with_capacity((n_theta + 1) * (n_z + 1));
    for iz in 0..=n_z {
        let z = LENGTH / 2.0 * iz as f64 / n_z as f64;
        for it in 0..=n_theta {
            let theta = half_angle * it as f64 / n_theta as f64;
            points.push(Point::new_3d(RADIUS * theta.sin(), RADIUS * theta.cos(), z));
        }
    }
    let node = |iz: usize, it: usize| -> usize { iz * (n_theta + 1) + it };
    let mut quads = Vec::with_capacity(n_theta * n_z);
    for iz in 0..n_z {
        for it in 0..n_theta {
            quads.push([
                node(iz, it),
                node(iz, it + 1),
                node(iz + 1, it + 1),
                node(iz + 1, it),
            ]);
        }
    }
    (points, quads)
}

fn solve_quarter(n_theta: usize, n_z: usize) -> f64 {
    let (points, quads) = build_quarter_model(n_theta, n_z);
    let section = ShellSection::new(E_MOD, 0.0, THICK, 0.0);
    let elements: Vec<QuadShellElement> = quads
        .iter()
        .map(|q| {
            QuadShellElement::from_points_with_section(
                *q,
                &points[q[0]],
                &points[q[1]],
                &points[q[2]],
                &points[q[3]],
                section,
            )
        })
        .collect();
    let n_dofs = points.len() * 6;
    let mut csr = assemble_quad_shell_stiffness(&elements, n_dofs).to_csr();
    // Self weight in -Y, uniform per unit area.
    let mut rhs = vec![0.0; n_dofs];
    for elem in &elements {
        let nodal = LOAD * elem.area / 4.0;
        for &node in &elem.nodes {
            rhs[node * 6 + 1] -= nodal;
        }
    }
    // Dirichlet: diaphragm at z = 0 (ux, uy); midspan symmetry (uz, tx, ty);
    // apex symmetry on x = 0 (ux, ty, tz).
    let node = |iz: usize, it: usize| -> usize { iz * (n_theta + 1) + it };
    let mut fixed = Vec::new();
    for it in 0..=n_theta {
        let n = node(0, it);
        fixed.push(n * 6);
        fixed.push(n * 6 + 1);
    }
    for it in 0..=n_theta {
        let n = node(n_z, it);
        fixed.push(n * 6 + 2);
        fixed.push(n * 6 + 3);
        fixed.push(n * 6 + 4);
    }
    for iz in 0..=n_z {
        let n = node(iz, 0);
        fixed.push(n * 6);
        fixed.push(n * 6 + 4);
        fixed.push(n * 6 + 5);
    }
    apply_dirichlet(&mut csr, &mut rhs, &fixed);
    let lu = SparseLuFactorization::factorize(&csr).expect("sparse LU must factorize");
    let x = lu.solve(&rhs).expect("sparse LU must solve");
    // Free-edge midpoint: (theta = 40 deg, z = L/2), vertical deflection.
    x[node(n_z, n_theta) * 6 + 1]
}

fn apply_dirichlet(csr: &mut sonium_solvers::CsrMatrix<f64>, rhs: &mut [f64], fixed: &[usize]) {
    use std::collections::BTreeSet;
    for dof in BTreeSet::from_iter(fixed.iter().copied()) {
        let range = csr.row_range(dof);
        for i in range {
            csr.values[i] = 0.0;
        }
        let diag = csr
            .row_entries(dof)
            .position(|(j, _)| j == dof)
            .expect("shell CSR rows must carry a diagonal");
        csr.values[csr.row_ptrs[dof] + diag] = 1.0;
        rhs[dof] = 0.0;
    }
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        let w_coarse = solve_quarter(8, 8);
        let w_fine = solve_quarter(20, 20);
        let err_coarse = (w_coarse - TARGET_W).abs() / TARGET_W.abs();
        let err_fine = (w_fine - TARGET_W).abs() / TARGET_W.abs();
        println!(
            "Scordelis-Lo MITC4: 8x8 w = {w_coarse:.6e} (err {err_coarse:.3e}), \
             20x20 w = {w_fine:.6e} (err {err_fine:.3e}), target {TARGET_W}"
        );
        assert!(
            err_fine < 0.03,
            "fine-mesh deflection {w_fine:.5} misses target {TARGET_W}"
        );
        assert!(
            err_fine < err_coarse,
            "mesh refinement must reduce the error"
        );
        // C3: static deck verification through the real API (no file writes;
        // engine-gated runs live in qa-fem-opencourant-shell-scordelis-lo).
        let case = shell_scordelis_lo_case();
        case.validate().expect("shell case must validate");
        let starter = write_extended_starter(&case);
        assert!(
            starter.contains("/PROP/SHELL/1/1") && starter.contains("/GRAV/1/1"),
            "starter deck must carry shell section and gravity"
        );
        println!("C3 shell_scordelis_lo: static deck check passes");
        let probe = case
            .base
            .probes
            .iter()
            .find(|probe| probe.label == "free edge mid DY")
            .expect("displacement probe must exist");
        assert_probe_close(
            w_fine,
            probe.expected,
            probe.abs_tol,
            probe.rel_tol,
            &probe.label,
        )
        .expect("FEM deflection must match the C3 probe");
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
