//! Wolfram golden replay: pinned circular arch.
//!
//! Replays `fixtures/solids_beam_curved_arch_golden.json` (derived by
//! `fixtures/solids_beam_curved_arch.wls`) through the production
//! `CurvedBeamElement3D` arch element: crown-load thrust and deflection in
//! both bending regimes, plus the membrane pressure state. Regenerate with:
//! `wolframscript -script fixtures/solids_beam_curved_arch.wls > fixtures/solids_beam_curved_arch_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::beam::{CurvedBeamElement3D, Section3D};

const GOLDEN: &str = include_str!("../fixtures/solids_beam_curved_arch_golden.json");

fn golden_value(doc: &Value, name: &str) -> f64 {
    doc["values"][name]["values"][0]
        .as_str()
        .unwrap_or_else(|| panic!("golden lacks {name}"))
        .parse()
        .unwrap()
}

fn check(doc: &Value, name: &str, actual: f64) {
    let expected = golden_value(doc, name);
    let abs: f64 = doc["tolerances"][format!("{name}_abs")].as_f64().unwrap();
    let rel: f64 = doc["tolerances"][format!("{name}_rel")].as_f64().unwrap();
    assert!(
        (actual - expected).abs() <= abs + rel * expected.abs(),
        "{name}: production {actual:.6e} vs golden {expected:.6e}"
    );
}

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
        assert!(best > 1.0e-300, "singular golden system");
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

/// Pinned semicircle mesh with `n` arch elements and `n + 1` nodes.
fn arch_elements(n: usize, radius: f64, section: Section3D) -> Vec<CurvedBeamElement3D> {
    let center = [0.0, 0.0, 0.0];
    (0..n)
        .map(|i| {
            let a0 = std::f64::consts::PI * i as f64 / n as f64;
            let a1 = std::f64::consts::PI * (i + 1) as f64 / n as f64;
            CurvedBeamElement3D::from_arc(
                [i, i + 1],
                [radius * a0.cos(), radius * a0.sin(), 0.0],
                [radius * a1.cos(), radius * a1.sin(), 0.0],
                center,
                section,
            )
        })
        .collect()
}

/// Solve the pinned arch and return the full displacement vector.
///
/// Pins fix all translations plus out-of-plane rotations (`rz` stays free);
/// interior nodes fix `uz` against out-of-plane rigid motion.
fn solve_arch(elements: &[CurvedBeamElement3D], loads: &[(usize, usize, f64)]) -> Vec<f64> {
    let n_nodes = elements.len() + 1;
    let n = n_nodes * 6;
    let mut k = vec![0.0; n * n];
    for elem in elements {
        let ke = elem.stiffness_global();
        let dofs = elem.global_dofs();
        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                k[gi * n + gj] += ke[i * 12 + j];
            }
        }
    }
    let mut fixed = vec![false; n];
    for node in [0, n_nodes - 1] {
        for c in [0, 1, 2, 3, 4] {
            fixed[node * 6 + c] = true;
        }
    }
    for node in 1..n_nodes - 1 {
        fixed[node * 6 + 2] = true;
    }
    let free: Vec<usize> = (0..n).filter(|d| !fixed[*d]).collect();
    let m = free.len();
    let mut matrix = vec![0.0; m * m];
    let mut rhs = vec![0.0; m];
    for (i, &gi) in free.iter().enumerate() {
        for (j, &gj) in free.iter().enumerate() {
            matrix[i * m + j] = k[gi * n + gj];
        }
    }
    for &(node, component, value) in loads {
        let dof = node * 6 + component;
        // Loads on fixed DOFs skip the free system; they still enter the
        // reaction recovery below.
        if let Some(pos) = free.iter().position(|&d| d == dof) {
            rhs[pos] += value;
        }
    }
    dense_solve(&mut matrix, &mut rhs, m);
    let mut u = vec![0.0; n];
    for (i, &gi) in free.iter().enumerate() {
        u[gi] = rhs[i];
    }
    // Reactions at fixed DOFs from the full stiffness action.
    let mut reactions = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += k[i * n + j] * u[j];
        }
        reactions[i] = sum;
    }
    for &(node, component, value) in loads {
        reactions[node * 6 + component] -= value;
    }
    // Stash reactions after the displacements for the caller.
    u.extend(reactions);
    u
}

#[test]
fn curved_arch_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["case"], "beam_curved_arch");
    assert_eq!(doc["meta"]["units"], "SI");

    let r = doc["inputs"]["radius_m"].as_f64().unwrap();
    let p = doc["inputs"]["crown_load_n"].as_f64().unwrap();
    let q = doc["inputs"]["pressure_pa"].as_f64().unwrap();
    let e = doc["inputs"]["youngs_modulus_pa"].as_f64().unwrap();
    let nu = doc["inputs"]["poisson_ratio"].as_f64().unwrap();
    let b = doc["inputs"]["width_m"].as_f64().unwrap();
    let h = doc["inputs"]["height_m"].as_f64().unwrap();

    for (timoshenko, thrust_name, crown_name) in [
        (false, "thrust", "crown_deflection"),
        (true, "thrust_timoshenko", "crown_deflection_timoshenko"),
    ] {
        let section = Section3D::rectangular(e, nu, b, h, 7800.0, timoshenko);
        let n = 16;
        let elements = arch_elements(n, r, section);
        let crown = n / 2;
        let u = solve_arch(&elements, &[(crown, 1, -p)]);
        let n_dofs = (n + 1) * 6;
        // Crown deflects downward; the golden stores the magnitude.
        check(&doc, crown_name, -u[crown * 6 + 1]);
        // Node 0 is the right pin at (R, 0); its thrust points inward (-x)
        // while the golden stores H > 0.
        check(&doc, thrust_name, -u[n_dofs]);
    }

    // Membrane state: quarter ring with symmetry BCs under uniform external
    // pressure. Symmetry (uy = rz = 0 at theta = 0, ux = rz = 0 at
    // theta = pi/2) kills all rigid modes while the uniform membrane state
    // satisfies every constraint naturally, so the contraction is exact.
    let section = Section3D::rectangular(e, nu, b, h, 7800.0, false);
    let n = 64;
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
    let n_nodes = n + 1;
    let n_dofs = n_nodes * 6;
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
    for node in 0..n_nodes {
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
    // Radial displacement is uniform in the membrane state; check mid-arc
    // against the golden value and the ends for uniformity.
    let mid = n / 2;
    let theta = std::f64::consts::FRAC_PI_4;
    let radial = u[mid * 6] * theta.cos() + u[mid * 6 + 1] * theta.sin();
    check(&doc, "membrane_radial", radial);
    let expected = golden_value(&doc, "membrane_radial");
    for node in [0, n] {
        let theta = std::f64::consts::FRAC_PI_2 * node as f64 / n as f64;
        let end_radial = u[node * 6] * theta.cos() + u[node * 6 + 1] * theta.sin();
        assert!(
            (end_radial - expected).abs() <= 1.0e-15 + 1.0e-3 * expected.abs(),
            "end {node} radial {end_radial:.6e} vs golden {expected:.6e}"
        );
    }
}
