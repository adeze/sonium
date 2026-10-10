//! QA: 0-D joints, rigid diaphragms, and tied interfaces.
//!
//! A spring-beam series matches its closed form, a rigid diaphragm reproduces
//! rigid in-plane kinematics through the MPC prolongation, and a tied
//! two-element cantilever matches the shared-node reference. A bushing
//! assembly smoke test exercises the triplet path.

// Rust guideline compliant 2026-02-21

use sonium_fem::boundary::{MpcSystem, condense_dense, tie_node_vectors};
use sonium_fem::physics::solid::beam::{BeamElement3D, Section3D};
use sonium_fem::physics::solid::joints::{AxialSpring, Bushing, RigidDiaphragm, Spotweld};

/// Solve a dense system in place with partial pivoting.
fn dense_solve(matrix: &mut [f64], rhs: &mut [f64], n: usize) {
    for col in 0..n {
        let mut pivot = col;
        let mut best = matrix[col * n + col].abs();
        for row in col + 1..n {
            let candidate = matrix[row * n + col].abs();
            if candidate > best {
                best = candidate;
                pivot = row;
            }
        }
        assert!(best > 1.0e-300, "singular system");
        if pivot != col {
            for k in col..n {
                matrix.swap(col * n + k, pivot * n + k);
            }
            rhs.swap(col, pivot);
        }
        let diag = matrix[col * n + col];
        for row in col + 1..n {
            let factor = matrix[row * n + col] / diag;
            matrix[row * n + col] = 0.0;
            for k in col + 1..n {
                matrix[row * n + k] -= factor * matrix[col * n + k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    for row in (0..n).rev() {
        let mut sum = rhs[row];
        for k in row + 1..n {
            sum -= matrix[row * n + k] * rhs[k];
        }
        rhs[row] = sum / matrix[row * n + row];
    }
}

fn test_spring_beam_series() {
    let section = Section3D::rectangular(200.0e9, 0.3, 0.05, 0.05, 7800.0, false);
    let beam = BeamElement3D::new(
        [0, 1],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        section,
        [0.0, 1.0, 0.0],
    );
    // Axial spring from the beam tip to a loaded node.
    let spring = AxialSpring::new(6, 12, 1.0e8).element();
    let n = 18;
    let mut k = vec![0.0; n * n];
    let kb = beam.stiffness_global();
    let dofs = beam.global_dofs();
    for (i, &gi) in dofs.iter().enumerate() {
        for (j, &gj) in dofs.iter().enumerate() {
            k[gi * n + gj] += kb[i * 12 + j];
        }
    }
    for (i, &gi) in spring.dofs.iter().enumerate() {
        for (j, &gj) in spring.dofs.iter().enumerate() {
            k[gi * n + gj] += spring.matrix[i * spring.dofs.len() + j];
        }
    }
    // Penalty clamp on node 0.
    for dof in 0..6 {
        k[dof * n + dof] += 1.0e14;
    }
    // The load node moves axially only; constrain its unused DOFs.
    for dof in 13..18 {
        k[dof * n + dof] += 1.0e14;
    }
    let force = 10000.0;
    let mut rhs = vec![0.0; n];
    rhs[12] = force;
    dense_solve(&mut k, &mut rhs, n);
    let k_beam = section.ea() / 1.0;
    let exact = force * (1.0 / 1.0e8 + 1.0 / k_beam);
    let error = (rhs[12] - exact).abs() / exact;
    println!(
        "spring-beam series: FEM={:.12e} exact={exact:.12e} rel={error:.2e}",
        rhs[12]
    );
    assert!(error < 1.0e-6, "series displacement error {error:.3e}");
}

fn test_rigid_diaphragm_patch() {
    let diaphragm = RigidDiaphragm::new(
        0,
        vec![1, 2, 3],
        vec![
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [3.0, 4.0, 0.0],
        ],
    );
    let constraints = diaphragm.constraints();
    assert_eq!(constraints.len(), 9);
    let mpc = MpcSystem::new(24, &constraints).expect("diaphragm MPC builds");
    // Master motion: ux=0.1, uy=-0.05, rz=0.02; everything else zero.
    let mut reduced = vec![0.0; mpc.num_reduced_dofs()];
    let order = mpc.reduced_full_dofs().to_vec();
    for (i, &dof) in order.iter().enumerate() {
        if dof == 0 {
            reduced[i] = 0.1;
        } else if dof == 1 {
            reduced[i] = -0.05;
        } else if dof == 5 {
            reduced[i] = 0.02;
        }
    }
    let full = mpc.recover_real(&reduced).expect("diaphragm recovers");
    // Slave 3 at (3,4): ux = 0.1 - 0.02*4, uy = -0.05 + 0.02*3.
    assert!((full[18] - 0.02).abs() < 1.0e-12);
    assert!((full[19] - 0.01).abs() < 1.0e-12);
    assert!((full[23] - 0.02).abs() < 1.0e-12);
    // Slave 1 at (3,0): ux = 0.1, uy = -0.05 + 0.06.
    assert!((full[6] - 0.1).abs() < 1.0e-12);
    assert!((full[7] - 0.01).abs() < 1.0e-12);
    println!("rigid diaphragm: slave motion matches rigid kinematics");
}

fn cantilever_tip_shared() -> f64 {
    let section = Section3D::rectangular(200.0e9, 0.3, 0.05, 0.05, 7800.0, false);
    let elements = [
        BeamElement3D::new(
            [0, 1],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            section,
            [0.0, 1.0, 0.0],
        ),
        BeamElement3D::new(
            [1, 2],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            section,
            [0.0, 1.0, 0.0],
        ),
    ];
    let n = 18;
    let mut k = vec![0.0; n * n];
    for elem in &elements {
        let ke = elem.stiffness_global();
        let dofs = elem.global_dofs();
        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                k[gi * n + gj] += ke[i * 12 + j];
            }
        }
    }
    for dof in 0..6 {
        k[dof * n + dof] += 1.0e14;
    }
    let mut rhs = vec![0.0; n];
    rhs[13] = 1000.0;
    dense_solve(&mut k, &mut rhs, n);
    rhs[13]
}

fn test_tied_interface_patch() {
    let section = Section3D::rectangular(200.0e9, 0.3, 0.05, 0.05, 7800.0, false);
    // Two elements with separate interface nodes 1 and 2.
    let elements = [
        BeamElement3D::new(
            [0, 1],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            section,
            [0.0, 1.0, 0.0],
        ),
        BeamElement3D::new(
            [2, 3],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            section,
            [0.0, 1.0, 0.0],
        ),
    ];
    let n = 24;
    let mut k = vec![0.0; n * n];
    for elem in &elements {
        let ke = elem.stiffness_global();
        let dofs = elem.global_dofs();
        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                k[gi * n + gj] += ke[i * 12 + j];
            }
        }
    }
    for dof in 0..6 {
        k[dof * n + dof] += 1.0e14;
    }
    let mut rhs = vec![0.0; n];
    rhs[19] = 1000.0;
    let constraints = tie_node_vectors(2, 1, 6);
    let mpc = MpcSystem::new(n, &constraints).expect("tied MPC builds");
    let k_red = condense_dense(&mpc, &k, n).expect("dense condense works");
    let rhs_red = mpc.condense_real_rhs(&rhs).expect("rhs condense works");
    let m = mpc.num_reduced_dofs();
    let mut matrix = k_red;
    let mut load = rhs_red;
    dense_solve(&mut matrix, &mut load, m);
    let full = mpc.recover_real(&load).expect("tied recovers");
    let shared = cantilever_tip_shared();
    let error = (full[19] - shared).abs() / shared.abs();
    println!(
        "tied interface: tied={:.12e} shared={shared:.12e} rel={error:.2e}",
        full[19]
    );
    assert!(error < 1.0e-9, "tied interface error {error:.3e}");
    // Absolute check against Euler-Bernoulli (penalty clamping limits accuracy).
    let exact = 1000.0 / (3.0 * section.eiz());
    let absolute = (full[19] - exact).abs() / exact;
    assert!(absolute < 1.0e-4, "absolute tip error {absolute:.3e}");
}

fn test_bushing_assembly_smoke() {
    let bushing = Bushing::diagonal(
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [1.0e6, 2.0e6, 3.0e6, 4.0e3, 5.0e3, 6.0e3],
        [10.0, 20.0, 30.0, 4.0, 5.0, 6.0],
    );
    let mut triplet = sonium_fem::assembly::TripletMatrix::new(12, 12);
    bushing.stiffness_element().assemble_into(&mut triplet);
    let csr = triplet.to_csr();
    // Diagonal 6x6 block contributes 4 entries per diagonal: 24 total.
    assert_eq!(csr.values.len(), 24);
    let weld = Spotweld::new(
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        1.0e9,
        1.0e6,
        5000.0,
    );
    let (_, utilization) = weld.check([2.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert!((utilization - 0.4).abs() < 1.0e-12);
    println!("bushing assembly: 24 nonzeros; spotweld utilization {utilization:.3}");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_spring_beam_series();
        test_rigid_diaphragm_patch();
        test_tied_interface_patch();
        test_bushing_assembly_smoke();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
