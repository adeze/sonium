//! QA: 3-D Timoshenko cantilever vs Euler-Bernoulli and shear closed forms.
//!
//! Covers the axial mode, both bending planes, and an oblique mesh
//! orientation. The C3 comparison case lives in
//! `fem_radioss_beam_3d_cantilever.rs`.

// Rust guideline compliant 2026-02-21

use sonium_fem::physics::solid::beam::{BeamElement3D, Section3D, assemble_beam3d_stiffness};
use sonium_solvers::{CsrMatrix, GmresConfig, gmres};

fn apply_dirichlet_csr(
    csr: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    prescribed_dofs: &[usize],
    prescribed_values: &[f64],
) {
    for dof in prescribed_dofs {
        let range = csr.row_range(*dof);
        for i in range {
            csr.values[i] = 0.0;
        }
        let diag_pos = csr.row_entries(*dof).position(|(j, _)| j == *dof);
        if let Some(idx) = diag_pos {
            let global_idx = csr.row_ptrs[*dof] + idx;
            csr.values[global_idx] = 1.0;
        }
    }
    for (dof, &val) in prescribed_dofs.iter().zip(prescribed_values.iter()) {
        rhs[*dof] = val;
    }
}

fn gmres_config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 5000,
        restart: 200,
        tolerance: 8e-11,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    }
}

/// Solve a tip-loaded cantilever and return tip `(uy, uz)` at the free end.
fn cantilever_tip(
    p0: [f64; 3],
    direction: [f64; 3],
    up: [f64; 3],
    length: f64,
    section: Section3D,
    n_elements: usize,
    force: [f64; 3],
) -> [f64; 3] {
    let dx = length / n_elements as f64;
    let mut norm = 0.0;
    for component in direction {
        norm += component * component;
    }
    let norm = norm.sqrt();
    let axis = [
        direction[0] / norm,
        direction[1] / norm,
        direction[2] / norm,
    ];
    let elements: Vec<BeamElement3D> = (0..n_elements)
        .map(|i| {
            let a = i as f64 * dx;
            let b = (i + 1) as f64 * dx;
            BeamElement3D::new(
                [i, i + 1],
                [
                    p0[0] + axis[0] * a,
                    p0[1] + axis[1] * a,
                    p0[2] + axis[2] * a,
                ],
                [
                    p0[0] + axis[0] * b,
                    p0[1] + axis[1] * b,
                    p0[2] + axis[2] * b,
                ],
                section,
                up,
            )
        })
        .collect();
    let n_dofs = (n_elements + 1) * 6;
    let mut csr = assemble_beam3d_stiffness(&elements, n_dofs).to_csr();
    let mut rhs = vec![0.0; n_dofs];
    for component in 0..3 {
        rhs[n_elements * 6 + component] = force[component];
    }
    apply_dirichlet_csr(&mut csr, &mut rhs, &[0, 1, 2, 3, 4, 5], &[0.0; 6]);
    let result = gmres(&csr, &rhs, &gmres_config());
    assert!(
        result.converged,
        "cantilever solve must converge: residual={:.3e} iterations={}",
        result.residual, result.iterations
    );
    [
        result.x[n_elements * 6],
        result.x[n_elements * 6 + 1],
        result.x[n_elements * 6 + 2],
    ]
}

fn test_cantilever_planes() {
    let length = 1.0;
    let force = 1000.0;
    let section = Section3D::rectangular(200.0e9, 0.3, 0.05, 0.05, 7800.0, false);
    // Bending in the x-y plane (about local z).
    let tip_y_plane = cantilever_tip(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        length,
        section,
        10,
        [0.0, force, 0.0],
    );
    let tip = tip_y_plane;
    let exact_y = force * length.powi(3) / (3.0 * section.eiz());
    let error_y = (tip[1] - exact_y).abs() / exact_y;
    println!(
        "cantilever y-plane: FEM={:.12e} exact={:.12e} rel={:.2e}",
        tip[1], exact_y, error_y
    );
    assert!(error_y < 1.0e-9, "y-plane tip error {error_y:.3e}");
    // Bending in the x-z plane (about local y).
    let tip = cantilever_tip(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        length,
        section,
        10,
        [0.0, 0.0, force],
    );
    let exact_z = force * length.powi(3) / (3.0 * section.eiy());
    let error_z = (tip[2] - exact_z).abs() / exact_z;
    println!(
        "cantilever z-plane: FEM={:.12e} exact={:.12e} rel={:.2e}",
        tip[2], exact_z, error_z
    );
    assert!(error_z < 1.0e-9, "z-plane tip error {error_z:.3e}");
    // Axial response along the beam.
    let tip = cantilever_tip(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        length,
        section,
        10,
        [force, 0.0, 0.0],
    );
    let exact_x = force * length / section.ea();
    let error_x = (tip[0] - exact_x).abs() / exact_x;
    println!(
        "cantilever axial: FEM={:.12e} exact={:.12e} rel={:.2e}",
        tip[0], exact_x, error_x
    );
    assert!(error_x < 1.0e-9, "axial tip error {error_x:.3e}");
    println!(
        "GOLDEN beam_3d_cantilever_tip fem={:.17e} exact={exact_y:.17e}",
        tip_y_plane[1]
    );
}

fn test_cantilever_timoshenko() {
    let length = 1.0;
    let force = 1000.0;
    let section = Section3D::rectangular(200.0e9, 0.3, 0.05, 0.05, 7800.0, true);
    let tip = cantilever_tip(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        length,
        section,
        10,
        [0.0, force, 0.0],
    );
    let exact = force * length.powi(3) / (3.0 * section.eiz())
        + force * length / section.shear_stiffness_y();
    let error = (tip[1] - exact).abs() / exact;
    println!(
        "cantilever timoshenko: FEM={:.12e} exact={:.12e} rel={:.2e}",
        tip[1], exact, error
    );
    assert!(error < 1.0e-9, "timoshenko tip error {error:.3e}");
}

fn test_cantilever_oblique() {
    let length = 1.0;
    let force = 1000.0;
    let section = Section3D::circular(70.0e9, 0.33, 0.025, 2700.0, false);
    // Beam along (1,1,0) with transverse load along (-1,1,0).
    let tip = cantilever_tip(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        length,
        section,
        8,
        [-force / 2.0_f64.sqrt(), force / 2.0_f64.sqrt(), 0.0],
    );
    let tip_norm = (tip[0] * tip[0] + tip[1] * tip[1] + tip[2] * tip[2]).sqrt();
    let exact = force * length.powi(3) / (3.0 * section.eiz());
    let error = (tip_norm - exact).abs() / exact;
    println!("cantilever oblique: FEM={tip_norm:.12e} exact={exact:.12e} rel={error:.2e}");
    assert!(error < 1.0e-9, "oblique tip error {error:.3e}");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_cantilever_planes();
        test_cantilever_timoshenko();
        test_cantilever_oblique();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
