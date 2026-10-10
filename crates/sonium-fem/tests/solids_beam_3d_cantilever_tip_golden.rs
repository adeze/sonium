//! Wolfram golden replay: 3-D cantilever tip.
//!
//! Replays `fixtures/solids_beam_3d_cantilever_tip_golden.json` (derived by
//! `fixtures/solids_beam_3d_cantilever_tip.wls`) through the production
//! `BeamElement3D` space-frame element. Regenerate with:
//! `wolframscript -script fixtures/solids_beam_3d_cantilever_tip.wls > fixtures/solids_beam_3d_cantilever_tip_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::beam::{BeamElement3D, Section3D};

const GOLDEN: &str = include_str!("../fixtures/solids_beam_3d_cantilever_tip_golden.json");

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
fn beam_3d_cantilever_tip_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["case"], "beam_3d_cantilever_tip");
    assert_eq!(doc["meta"]["units"], "SI");

    let f = doc["inputs"]["load_n"].as_f64().unwrap();
    let l = doc["inputs"]["length_m"].as_f64().unwrap();
    let e = doc["inputs"]["youngs_modulus_pa"].as_f64().unwrap();
    let b = doc["inputs"]["width_m"].as_f64().unwrap();
    let h = doc["inputs"]["height_m"].as_f64().unwrap();
    let section = Section3D::rectangular(e, 0.3, b, h, 7800.0, false);

    let n_elements = 10;
    let dx = l / n_elements as f64;
    let elements: Vec<BeamElement3D> = (0..n_elements)
        .map(|i| {
            BeamElement3D::new(
                [i, i + 1],
                [i as f64 * dx, 0.0, 0.0],
                [(i + 1) as f64 * dx, 0.0, 0.0],
                section,
                [0.0, 1.0, 0.0],
            )
        })
        .collect();
    let n = (n_elements + 1) * 6;
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
    // Eliminate the clamped root DOFs directly.
    let free: Vec<usize> = (6..n).collect();
    let m = free.len();
    let mut matrix = vec![0.0; m * m];
    let mut rhs = vec![0.0; m];
    for (i, &gi) in free.iter().enumerate() {
        for (j, &gj) in free.iter().enumerate() {
            matrix[i * m + j] = k[gi * n + gj];
        }
    }
    rhs[m - 5] = f;
    dense_solve(&mut matrix, &mut rhs, m);
    check(&doc, "tip_deflection", rhs[m - 5]);
    check(&doc, "tip_slope", rhs[m - 1]);
}
