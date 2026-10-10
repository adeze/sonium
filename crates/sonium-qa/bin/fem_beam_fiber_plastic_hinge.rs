//! QA: elastoplastic fiber beam vs the analytic plastic hinge.
//!
//! Covers rectangular-section moment-curvature (first yield My, plastic
//! moment Mp, and the M(kappa) curve), a cantilever push toward the
//! collapse load Mp/L with Newton load stepping, and the elastic limit.
//! Closed forms for the elastic-perfectly-plastic rectangular section
//! (width b in y, height h in z, bent about z): My = s_y h b^2/6,
//! Mp = s_y h b^2/4, first-yield curvature ky = 2 s_y/(E b), and
//! M = Mp (1 - ky^2/(3 k^2)) for k > ky. See e.g. Chakrabarty, Theory
//! of Plasticity, or Lubliner, Plasticity Theory (rectangular hinge).

// Rust guideline compliant 2026-02-21

use sonium_fem::forms::NonlinearElementForm;
use sonium_fem::physics::solid::beam::{
    BeamElement3D, Fiber, FiberBeamElement, FiberBeamForm3D, FiberSection, Section3D,
    UniaxialStressJ2Adapter,
};
use sonium_fem::physics::solid::plasticity::{HardeningLaw, J2Material, PlasticState};

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
        assert!(best > 1.0e-300, "singular QA system");
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

fn perfect_steel() -> UniaxialStressJ2Adapter {
    UniaxialStressJ2Adapter::new(J2Material::new(
        200.0e9,
        0.3,
        250.0e6,
        HardeningLaw::Perfect,
    ))
}

/// Moment-curvature point: fresh section driven to target curvature.
fn moment_at_curvature(
    section: &FiberSection<UniaxialStressJ2Adapter>,
    kappa_z: f64,
    steps: usize,
) -> f64 {
    let mut states = section.initial_states(PlasticState::new_3d);
    let mut moment = 0.0;
    for _ in 0..steps {
        let result = section.integrate(&states, 0.0, 0.0, kappa_z / steps as f64);
        moment = result.mz;
        section.commit(&mut states, &result);
    }
    moment
}

fn test_moment_curvature() {
    // Rectangular section bent about z: b = 0.02 in y, h = 0.01 in z.
    let b = 0.02;
    let h = 0.01;
    let e = 200.0e9;
    let sy = 250.0e6;
    let section = FiberSection::new(perfect_steel(), Fiber::grid(b, h, 32, 8));
    let my = sy * h * b * b / 6.0;
    let mp = sy * h * b * b / 4.0;
    let ky = 2.0 * sy / (e * b);
    for (kappa, expected, label) in [
        (ky, my, "first yield"),
        (3.0 * ky, mp * (1.0 - 1.0 / 27.0), "3 ky"),
        (10.0 * ky, mp * (1.0 - 1.0 / 300.0), "10 ky"),
    ] {
        let fem = moment_at_curvature(&section, kappa, 40);
        let error = (fem - expected).abs() / expected;
        println!("M-kappa {label}: FEM={fem:.12e} exact={expected:.12e} rel={error:.2e}");
        assert!(error < 0.03, "{label} error {error:.3e}");
    }
    println!("GOLDEN beam_fiber_mp my={my:.17e} mp={mp:.17e}");
}

/// Newton solve of the fiber form with load stepping and commit.
fn push_cantilever(
    form: &mut FiberBeamForm3D<UniaxialStressJ2Adapter>,
    tip_dof: usize,
    target: f64,
    steps: usize,
) -> Vec<f64> {
    let n = form.num_global_dofs();
    let free: Vec<usize> = (6..n).collect();
    let m = free.len();
    let mut u = vec![0.0; n];
    for step in 1..=steps {
        let load = target * step as f64 / steps as f64;
        for _ in 0..50 {
            let mut residual = vec![0.0; n];
            let mut tangent = vec![0.0; n * n];
            for elem in 0..form.num_elements() {
                let map = form.local_to_global(elem);
                let lr = form.element_residual(elem, &u);
                let lt = form.element_tangent(elem, &u);
                for (i, &gi) in map.iter().enumerate() {
                    residual[gi] += lr[i];
                    for (j, &gj) in map.iter().enumerate() {
                        tangent[gi * n + gj] += lt[i * map.len() + j];
                    }
                }
            }
            residual[tip_dof] -= load;
            let norm: f64 = free
                .iter()
                .map(|&d| residual[d] * residual[d])
                .sum::<f64>()
                .sqrt();
            if norm / load.max(1.0e-30) < 1.0e-9 {
                break;
            }
            let mut matrix = vec![0.0; m * m];
            let mut rhs = vec![0.0; m];
            for (i, &gi) in free.iter().enumerate() {
                rhs[i] = -residual[gi];
                for (j, &gj) in free.iter().enumerate() {
                    matrix[i * m + j] = tangent[gi * n + gj];
                }
            }
            dense_solve(&mut matrix, &mut rhs, m);
            for (i, &gi) in free.iter().enumerate() {
                u[gi] += rhs[i];
            }
        }
        // Verify the step converged before committing its history.
        let mut residual = vec![0.0; n];
        for elem in 0..form.num_elements() {
            let map = form.local_to_global(elem);
            let lr = form.element_residual(elem, &u);
            for (i, &gi) in map.iter().enumerate() {
                residual[gi] += lr[i];
            }
        }
        residual[tip_dof] -= load;
        let norm: f64 = free
            .iter()
            .map(|&d| residual[d] * residual[d])
            .sum::<f64>()
            .sqrt();
        assert!(
            norm / load.max(1.0e-30) < 1.0e-8,
            "cantilever push stalled at load {load:.3e}"
        );
        form.commit(&u);
    }
    u
}

fn fiber_cantilever() -> FiberBeamForm3D<UniaxialStressJ2Adapter> {
    let b = 0.02;
    let h = 0.01;
    let l = 1.0;
    let n = 4;
    let dx = l / n as f64;
    let elements: Vec<FiberBeamElement<UniaxialStressJ2Adapter>> = (0..n)
        .map(|i| {
            let beam = BeamElement3D::new(
                [i, i + 1],
                [i as f64 * dx, 0.0, 0.0],
                [(i + 1) as f64 * dx, 0.0, 0.0],
                Section3D::rectangular(200.0e9, 0.3, b, h, 7800.0, false),
                [0.0, 1.0, 0.0],
            );
            let section = FiberSection::new(perfect_steel(), Fiber::grid(b, h, 16, 8));
            FiberBeamElement::new(beam, section, 3)
        })
        .collect();
    FiberBeamForm3D::new(elements, PlasticState::new_3d, (n + 1) * 6)
}

fn test_plastic_push() {
    // Collapse load of the tip-loaded cantilever: Pp = Mp / L.
    let b = 0.02;
    let h = 0.01;
    let l = 1.0;
    let mp = 250.0e6 * h * b * b / 4.0;
    let mut form = fiber_cantilever();
    let tip_dof = 4 * 6 + 1;
    let u = push_cantilever(&mut form, tip_dof, 0.9 * mp / l, 12);
    // Equilibrium: moment at the first Gauss point of the root element
    // balances P (L - x_g) with x_g = 0.1127 * 0.25 for 3-point Gauss.
    let root = form.resultants_for(0, &u);
    let root_moment = root[0][2].abs();
    let expected = 0.9 * mp * (l - 0.112_701_665_379_258_3 * 0.25) / l;
    let err_m = (root_moment - expected).abs() / expected;
    println!("push: root M={root_moment:.12e} P*L={expected:.12e} rel={err_m:.2e}");
    assert!(err_m < 0.02, "root moment error {err_m:.3e}");
    // Plastic softening: tip deflects past the elastic prediction.
    let eiz = 200.0e9 * h * b.powi(3) / 12.0;
    let elastic = 0.9 * mp / l * l.powi(3) / (3.0 * eiz);
    println!("push: tip={:.12e} elastic={elastic:.12e}", u[tip_dof]);
    assert!(u[tip_dof] > elastic, "no plastic softening observed");
    // Root fibers yield.
    assert!(
        form.states[0].fiber_states[0]
            .iter()
            .any(|s| s.equiv_plastic_strain > 0.0),
        "root fibers stayed elastic"
    );
}

fn test_elastic_limit() {
    // Small load: fiber beam reproduces Euler-Bernoulli deflection.
    let b = 0.02;
    let h = 0.01;
    let l = 1.0;
    let my = 250.0e6 * h * b * b / 6.0;
    let mut form = fiber_cantilever();
    let tip_dof = 4 * 6 + 1;
    let u = push_cantilever(&mut form, tip_dof, 0.1 * my / l, 2);
    let eiz = 200.0e9 * h * b.powi(3) / 12.0;
    let expected = 0.1 * my / l * l.powi(3) / (3.0 * eiz);
    let error = (u[tip_dof] - expected).abs() / expected;
    println!("elastic: FEM={:.12e} exact={expected:.12e} rel={error:.2e}", u[tip_dof]);
    assert!(error < 0.02, "elastic error {error:.3e}");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_moment_curvature();
        test_plastic_push();
        test_elastic_limit();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
