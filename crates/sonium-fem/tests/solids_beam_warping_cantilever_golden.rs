//! Wolfram golden replay: warping-restrained cantilever.
//!
//! Replays `fixtures/solids_beam_warping_cantilever_golden.json` (derived by
//! `fixtures/solids_beam_warping_cantilever.wls`) through the production
//! `WarpingBeamElement3D` 7-DOF element. Regenerate with:
//! `wolframscript -script fixtures/solids_beam_warping_cantilever.wls > fixtures/solids_beam_warping_cantilever_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::beam::{Section3D, WarpingBeamElement3D};

const GOLDEN: &str = include_str!("../fixtures/solids_beam_warping_cantilever_golden.json");

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

#[test]
fn warping_cantilever_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["case"], "beam_warping_cantilever");
    assert_eq!(doc["meta"]["units"], "SI");

    let l = doc["inputs"]["length_m"].as_f64().unwrap();
    let t = doc["inputs"]["tip_torque_n_m"].as_f64().unwrap();
    let e = doc["inputs"]["youngs_modulus_pa"].as_f64().unwrap();
    let nu = doc["inputs"]["poisson_ratio"].as_f64().unwrap();
    let b = doc["inputs"]["flange_width_m"].as_f64().unwrap();
    let h = doc["inputs"]["web_height_m"].as_f64().unwrap();
    let th = doc["inputs"]["thickness_m"].as_f64().unwrap();
    let section = Section3D::i_beam(e, nu, b, h, th, 7800.0, false);

    // Golden section constants must match the production thin-wall formulas.
    let golden_j: f64 = doc["inputs"]["torsion_constant_m4"]
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    let golden_iw: f64 = doc["inputs"]["warping_constant_m6"]
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    assert!((section.torsion_constant - golden_j).abs() / golden_j < 1.0e-12);
    assert!((section.warping_constant - golden_iw).abs() / golden_iw < 1.0e-12);

    let n_elements = 64;
    let dx = l / n_elements as f64;
    let elements: Vec<WarpingBeamElement3D> = (0..n_elements)
        .map(|i| {
            WarpingBeamElement3D::new(
                [i, i + 1],
                [i as f64 * dx, 0.0, 0.0],
                [(i + 1) as f64 * dx, 0.0, 0.0],
                section,
                [0.0, 1.0, 0.0],
            )
        })
        .collect();
    let n = (n_elements + 1) * 7;
    let mut k = vec![0.0; n * n];
    for elem in &elements {
        let ke = elem.stiffness_global();
        let dofs = elem.global_dofs();
        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                k[gi * n + gj] += ke[i * 14 + j];
            }
        }
    }
    // Clamp the root (all 7 DOFs: twist and warping fixed).
    let free: Vec<usize> = (7..n).collect();
    let m = free.len();
    let mut matrix = vec![0.0; m * m];
    let mut rhs = vec![0.0; m];
    for (i, &gi) in free.iter().enumerate() {
        for (j, &gj) in free.iter().enumerate() {
            matrix[i * m + j] = k[gi * n + gj];
        }
    }
    // Tip torque on the tip rx DOF.
    let tip_rx = n - 4;
    rhs[free.iter().position(|&d| d == tip_rx).unwrap()] = t;
    dense_solve(&mut matrix, &mut rhs, m);
    let mut u = vec![0.0; n];
    for (i, &gi) in free.iter().enumerate() {
        u[gi] = rhs[i];
    }
    check(&doc, "tip_twist", u[tip_rx]);
    check(&doc, "midspan_twist", u[(n_elements / 2) * 7 + 3]);
    // Root bimoment from equilibrium: reaction at the root warping DOF.
    let mut reaction = 0.0;
    for j in 0..n {
        reaction += k[6 * n + j] * u[j];
    }
    check(&doc, "root_bimoment", reaction);
}
