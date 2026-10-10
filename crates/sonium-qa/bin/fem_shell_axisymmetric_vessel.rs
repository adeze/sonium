//! Axisymmetric pressure vessels with shells of revolution.
//!
//! Spherical vessel (R = 2, t = 0.02) under internal pressure p = 1e6:
//! radial expansion vs u = p R^2 (1 - nu) / (2 E t). Open cylindrical vessel
//! (R = 1, t = 0.01, L = 10) under p = 1e6: mid-length expansion vs
//! u = p R^2 / (E t).
//!
//! Reference: Timoshenko, S. and Woinowsky-Krieger, S., "Theory of Plates
//! and Shells", 2nd ed., membrane theory of shells of revolution.

use sonium_fem::physics::solid::shell::{
    RevolutionShellElement, ShellSection, assemble_revolution_pressure,
    assemble_revolution_stiffness, cylinder_membrane_displacement, sphere_membrane_displacement,
};
use sonium_solvers::direct::SparseLuFactorization;

const E_MOD: f64 = 200e9;
const NU: f64 = 0.3;

fn solve_meridian(
    coords: &[(f64, f64)],
    pressure: f64,
    thickness: f64,
    fixed: &[usize],
) -> Vec<f64> {
    let section = ShellSection::new(E_MOD, NU, thickness, 7800.0);
    let elements: Vec<RevolutionShellElement> = coords
        .windows(2)
        .enumerate()
        .map(|(i, w)| {
            RevolutionShellElement::new([i, i + 1], [w[0].0, w[1].0], [w[0].1, w[1].1], section)
        })
        .collect();
    let n_dofs = coords.len() * 3;
    let mut csr = assemble_revolution_stiffness(&elements, n_dofs).to_csr();
    let mut rhs = assemble_revolution_pressure(&elements, n_dofs, pressure);
    for &dof in fixed {
        let range = csr.row_range(dof);
        for i in range {
            csr.values[i] = 0.0;
        }
        let diag = csr
            .row_entries(dof)
            .position(|(j, _)| j == dof)
            .expect("revolution CSR rows must carry a diagonal");
        csr.values[csr.row_ptrs[dof] + diag] = 1.0;
        rhs[dof] = 0.0;
    }
    let lu = SparseLuFactorization::factorize(&csr).expect("sparse LU must factorize");
    lu.solve(&rhs).expect("sparse LU must solve")
}

fn sphere_case() -> (f64, f64) {
    // Hemisphere meridian from the equator to 5 deg off the pole.
    // Traversal keeps the interior on the left, which is the element
    // convention for an outward pressure normal.
    let radius = 2.0;
    let thick = 0.02;
    let pressure = 1e6;
    let n = 40;
    let coords: Vec<(f64, f64)> = (0..=n)
        .map(|i| {
            let phi = (90.0 - 85.0 * i as f64 / n as f64).to_radians();
            (radius * phi.sin(), radius * phi.cos())
        })
        .collect();
    // Equator symmetry: uz = 0, beta = 0 on the first node.
    let fixed = vec![1, 2];
    let x = solve_meridian(&coords, pressure, thick, &fixed);
    // Radial expansion at the equator.
    let u_r = x[0];
    let exact = sphere_membrane_displacement(pressure, radius, thick, E_MOD, NU);
    (u_r, exact)
}

fn cylinder_case() -> (f64, f64) {
    // Open cylinder, half model z in [0, 5] with mid symmetry at z = 5.
    let radius = 1.0;
    let thick = 0.01;
    let pressure = 1e6;
    let n = 40;
    let coords: Vec<(f64, f64)> = (0..=n)
        .map(|i| (radius, 5.0 * i as f64 / n as f64))
        .collect();
    // Mid symmetry: uz = 0, beta = 0 on the last node.
    let fixed = vec![n * 3 + 1, n * 3 + 2];
    let x = solve_meridian(&coords, pressure, thick, &fixed);
    let u_r = x[n * 3];
    let exact = cylinder_membrane_displacement(pressure, radius, thick, E_MOD);
    (u_r, exact)
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        let (u_sphere, exact_sphere) = sphere_case();
        let err_sphere = (u_sphere - exact_sphere).abs() / exact_sphere;
        println!(
            "Sphere vessel: u_r = {u_sphere:.6e}, membrane {exact_sphere:.6e} \
             (err {err_sphere:.3e})"
        );
        assert!(err_sphere < 0.02, "sphere expansion misses membrane theory");

        let (u_cyl, exact_cyl) = cylinder_case();
        let err_cyl = (u_cyl - exact_cyl).abs() / exact_cyl;
        println!(
            "Cylinder vessel: u_r = {u_cyl:.6e}, membrane {exact_cyl:.6e} (err {err_cyl:.3e})"
        );
        assert!(err_cyl < 0.02, "cylinder expansion misses membrane theory");
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
