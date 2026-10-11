//! QA: curved arch beam vs Castigliano closed forms.
//!
//! Covers the quarter-circular cantilever (Euler-Bernoulli and Timoshenko,
//! single exact element), the pinned semicircular arch under crown load,
//! the membrane quarter-ring pressure state, and a faceted straight-beam
//! cross-check. Closed forms: Castigliano derivations for the circular arch
//! (bending plus axial, shear added for Timoshenko); see also Roark's
//! Formulas for Stress and Strain, Table 8.2 (circular arches), and
//! Timoshenko & Gere, Mechanics of Materials (curved-beam deflection).

// Rust guideline compliant 2026-02-21

use sonium_fem::physics::solid::beam::{BeamElement3D, CurvedBeamElement3D, Section3D};

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

fn test_quarter_cantilever() {
    // Quarter-circular cantilever (R = 1), tip tangential load P = 1000.
    // Castigliano: bend P R^3 (3pi/4 - 2)/EI + axial P R pi/(4 EA)
    // + shear P R pi/(4 GA kappa) for Timoshenko.
    let r = 1.0;
    let p = 1000.0;
    let pi = std::f64::consts::PI;
    for timoshenko in [false, true] {
        let section = Section3D::rectangular(200.0e9, 0.3, 0.05, 0.05, 7800.0, timoshenko);
        let elem = CurvedBeamElement3D::from_arc(
            [0, 1],
            [r, 0.0, 0.0],
            [0.0, r, 0.0],
            [0.0, 0.0, 0.0],
            section,
        );
        let k = elem.stiffness_global();
        let free: Vec<usize> = (6..12).collect();
        let mut matrix = [0.0; 36];
        for (i, &gi) in free.iter().enumerate() {
            for (j, &gj) in free.iter().enumerate() {
                matrix[i * 6 + j] = k[gi * 12 + gj];
            }
        }
        // Tip tangent at (0, R) runs along -x.
        let mut rhs = [0.0; 6];
        rhs[0] = -p;
        dense_solve(&mut matrix, &mut rhs, 6);
        let mut expected = p * r.powi(3) / section.eiz() * (3.0 * pi / 4.0 - 2.0)
            + p * r * pi / (4.0 * section.ea());
        if timoshenko {
            expected += p * r * pi / (4.0 * section.shear_stiffness_y());
        }
        let error = (rhs[0].abs() - expected).abs() / expected;
        println!(
            "quarter cantilever (timoshenko={timoshenko}): FEM={:.12e} exact={expected:.12e} rel={error:.2e}",
            rhs[0].abs()
        );
        assert!(error < 1.0e-9, "quarter cantilever error {error:.3e}");
    }
}

fn solve_pinned_crown(n: usize, timoshenko: bool) -> (f64, f64) {
    // Pinned semicircle, crown load P downward. Returns (crown uy, thrust).
    let r = 1.0;
    let p = 1000.0;
    let section = Section3D::rectangular(200.0e9, 0.3, 0.05, 0.05, 7800.0, timoshenko);
    let center = [0.0, 0.0, 0.0];
    let elements: Vec<CurvedBeamElement3D> = (0..n)
        .map(|i| {
            let a0 = std::f64::consts::PI * i as f64 / n as f64;
            let a1 = std::f64::consts::PI * (i + 1) as f64 / n as f64;
            CurvedBeamElement3D::from_arc(
                [i, i + 1],
                [r * a0.cos(), r * a0.sin(), 0.0],
                [r * a1.cos(), r * a1.sin(), 0.0],
                center,
                section,
            )
        })
        .collect();
    let n_dofs = (n + 1) * 6;
    let mut k = vec![0.0; n_dofs * n_dofs];
    for elem in &elements {
        let ke = elem.stiffness_global();
        let dofs = elem.global_dofs();
        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                k[gi * n_dofs + gj] += ke[i * 12 + j];
            }
        }
    }
    let mut fixed = vec![false; n_dofs];
    for node in [0, n] {
        for c in [0, 1, 2, 3, 4] {
            fixed[node * 6 + c] = true;
        }
    }
    for node in 1..n {
        fixed[node * 6 + 2] = true;
    }
    let free: Vec<usize> = (0..n_dofs).filter(|d| !fixed[*d]).collect();
    let m = free.len();
    let mut matrix = vec![0.0; m * m];
    let mut rhs = vec![0.0; m];
    for (i, &gi) in free.iter().enumerate() {
        for (j, &gj) in free.iter().enumerate() {
            matrix[i * m + j] = k[gi * n_dofs + gj];
        }
    }
    let crown = n / 2;
    rhs[free.iter().position(|&d| d == crown * 6 + 1).unwrap()] = -p;
    dense_solve(&mut matrix, &mut rhs, m);
    let mut u = vec![0.0; n_dofs];
    for (i, &gi) in free.iter().enumerate() {
        u[gi] = rhs[i];
    }
    // Thrust from the node-0 (right pin) x-reaction, inward positive.
    let mut reaction = 0.0;
    for j in 0..n_dofs {
        reaction += k[j] * u[j];
    }
    (u[crown * 6 + 1], -reaction)
}

fn test_pinned_arch_crown() {
    // Hand closed forms: thrust H = (P/pi)(R^2 - r^2)/(R^2 + r^2) is exact
    // within bending+axial theory; crown deflection uses the bending-only
    // value 0.0189424 P R^3/EI (axial adds ~1% for this section).
    let p = 1000.0;
    let r = 1.0;
    let section = Section3D::rectangular(200.0e9, 0.3, 0.05, 0.05, 7800.0, false);
    let gyro2 = section.iz / section.area;
    let expected_h = p / std::f64::consts::PI * (r * r - gyro2) / (r * r + gyro2);
    let expected_d = 0.018_942_4 * p * r.powi(3) / section.eiz();
    let (uy, h) = solve_pinned_crown(16, false);
    let err_h = (h - expected_h).abs() / expected_h;
    let err_d = (uy.abs() - expected_d).abs() / expected_d;
    println!("pinned arch: H FEM={h:.12e} exact={expected_h:.12e} rel={err_h:.2e}");
    println!(
        "pinned arch: crown FEM={:.12e} exact={expected_d:.12e} rel={err_d:.2e}",
        uy.abs()
    );
    assert!(err_h < 1.0e-6, "thrust error {err_h:.3e}");
    assert!(err_d < 0.02, "crown error {err_d:.3e}");
    // Self-convergence: exact elements agree across mesh sizes.
    let (uy32, h32) = solve_pinned_crown(32, false);
    assert!((uy32 - uy).abs() / uy.abs() < 1.0e-6);
    assert!((h32 - h).abs() / h.abs() < 1.0e-6);
    println!(
        "GOLDEN beam_curved_arch_crown fem={:.17e} hand={expected_d:.17e}",
        uy.abs()
    );
}

fn test_membrane_ring() {
    // Quarter ring with symmetry BCs under uniform external pressure:
    // pure membrane, uniform radial contraction w = -q R^2/(E A).
    let r = 1.0;
    let q = 1000.0;
    let section = Section3D::rectangular(200.0e9, 0.3, 0.05, 0.05, 7800.0, false);
    let n = 32;
    let center = [0.0, 0.0, 0.0];
    let elements: Vec<CurvedBeamElement3D> = (0..n)
        .map(|i| {
            let a0 = std::f64::consts::FRAC_PI_2 * i as f64 / n as f64;
            let a1 = std::f64::consts::FRAC_PI_2 * (i + 1) as f64 / n as f64;
            CurvedBeamElement3D::from_arc(
                [i, i + 1],
                [r * a0.cos(), r * a0.sin(), 0.0],
                [r * a1.cos(), r * a1.sin(), 0.0],
                center,
                section,
            )
        })
        .collect();
    let n_dofs = (n + 1) * 6;
    let mut k = vec![0.0; n_dofs * n_dofs];
    for elem in &elements {
        let ke = elem.stiffness_global();
        let dofs = elem.global_dofs();
        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                k[gi * n_dofs + gj] += ke[i * 12 + j];
            }
        }
    }
    let mut fixed = vec![false; n_dofs];
    for node in 0..=n {
        fixed[node * 6 + 2] = true;
    }
    for c in [1, 2, 3, 4, 5] {
        fixed[c] = true;
    }
    for c in [0, 2, 3, 4, 5] {
        fixed[n * 6 + c] = true;
    }
    let free: Vec<usize> = (0..n_dofs).filter(|d| !fixed[*d]).collect();
    let m = free.len();
    let mut matrix = vec![0.0; m * m];
    let mut rhs = vec![0.0; m];
    for (i, &gi) in free.iter().enumerate() {
        for (j, &gj) in free.iter().enumerate() {
            matrix[i * m + j] = k[gi * n_dofs + gj];
        }
    }
    let dtheta = std::f64::consts::FRAC_PI_2 / n as f64;
    for i in 0..=n {
        let theta = std::f64::consts::FRAC_PI_2 * i as f64 / n as f64;
        let tributary = if i == 0 || i == n { 0.5 } else { 1.0 } * q * r * dtheta;
        for (component, value) in [(0, -tributary * theta.cos()), (1, -tributary * theta.sin())] {
            let dof = i * 6 + component;
            if let Some(pos) = free.iter().position(|&d| d == dof) {
                rhs[pos] += value;
            }
        }
    }
    dense_solve(&mut matrix, &mut rhs, m);
    let mut u = vec![0.0; n_dofs];
    for (i, &gi) in free.iter().enumerate() {
        u[gi] = rhs[i];
    }
    let expected = -q * r * r / section.ea();
    let mut worst: f64 = 0.0;
    for i in 0..=n {
        let theta = std::f64::consts::FRAC_PI_2 * i as f64 / n as f64;
        let radial = u[i * 6] * theta.cos() + u[i * 6 + 1] * theta.sin();
        worst = worst.max((radial - expected).abs() / expected.abs());
    }
    println!("membrane ring: expected={expected:.12e} worst radial rel={worst:.2e}");
    assert!(worst < 1.0e-3, "membrane error {worst:.3e}");
}

fn test_faceted_cross_check() {
    // Independent discretization: 64 straight prismatic elements faceting
    // the semicircle must reproduce the 8-element curved result.
    let r = 1.0;
    let p = 1000.0;
    let section = Section3D::rectangular(200.0e9, 0.3, 0.05, 0.05, 7800.0, false);
    let n = 64;
    let elements: Vec<BeamElement3D> = (0..n)
        .map(|i| {
            let a0 = std::f64::consts::PI * i as f64 / n as f64;
            let a1 = std::f64::consts::PI * (i + 1) as f64 / n as f64;
            BeamElement3D::new(
                [i, i + 1],
                [r * a0.cos(), r * a0.sin(), 0.0],
                [r * a1.cos(), r * a1.sin(), 0.0],
                section,
                [0.0, 0.0, 1.0],
            )
        })
        .collect();
    let n_dofs = (n + 1) * 6;
    let mut k = vec![0.0; n_dofs * n_dofs];
    for elem in &elements {
        let ke = elem.stiffness_global();
        let dofs = elem.global_dofs();
        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                k[gi * n_dofs + gj] += ke[i * 12 + j];
            }
        }
    }
    let mut fixed = vec![false; n_dofs];
    for node in [0, n] {
        for c in [0, 1, 2, 3, 4] {
            fixed[node * 6 + c] = true;
        }
    }
    for node in 1..n {
        fixed[node * 6 + 2] = true;
    }
    let free: Vec<usize> = (0..n_dofs).filter(|d| !fixed[*d]).collect();
    let m = free.len();
    let mut matrix = vec![0.0; m * m];
    let mut rhs = vec![0.0; m];
    for (i, &gi) in free.iter().enumerate() {
        for (j, &gj) in free.iter().enumerate() {
            matrix[i * m + j] = k[gi * n_dofs + gj];
        }
    }
    let crown = n / 2;
    rhs[free.iter().position(|&d| d == crown * 6 + 1).unwrap()] = -p;
    dense_solve(&mut matrix, &mut rhs, m);
    let faceted = rhs[free.iter().position(|&d| d == crown * 6 + 1).unwrap()].abs();
    let (curved, _) = solve_pinned_crown(8, false);
    let error = (faceted - curved.abs()).abs() / curved.abs();
    println!(
        "faceted cross-check: faceted={faceted:.12e} curved={:.12e} rel={error:.2e}",
        curved.abs()
    );
    assert!(error < 5.0e-3, "faceted error {error:.3e}");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_quarter_cantilever();
        test_pinned_arch_crown();
        test_membrane_ring();
        test_faceted_cross_check();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
