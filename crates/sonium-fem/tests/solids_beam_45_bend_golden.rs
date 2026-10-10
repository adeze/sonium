//! Wolfram golden replay: 45-degree elastica bend.
//!
//! Replays `fixtures/solids_beam_45_bend_golden.json` (derived by
//! `fixtures/solids_beam_45_bend.wls`) through the production 3-D
//! corotational beam form with Newton load stepping. Regenerate with:
//! `wolframscript -script fixtures/solids_beam_45_bend.wls > fixtures/solids_beam_45_bend_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::forms::NonlinearElementForm;
use sonium_fem::physics::solid::beam::{BeamElement3D, CorotationalBeamForm3D, Section3D};

const GOLDEN: &str = include_str!("../fixtures/solids_beam_45_bend_golden.json");

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
fn beam_45_bend_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["case"], "beam_45_bend");
    assert_eq!(doc["meta"]["units"], "SI");

    let l = doc["inputs"]["length_m"].as_f64().unwrap();
    let theta = std::f64::consts::FRAC_PI_4;
    let n_elements = 20;
    let section = Section3D::rectangular(1.2e6, 0.0, 1.0, 0.1, 1.0, false);
    let dx = l / n_elements as f64;
    let mut positions = Vec::with_capacity(n_elements + 1);
    for i in 0..=n_elements {
        positions.push([i as f64 * dx, 0.0, 0.0]);
    }
    let elements: Vec<BeamElement3D> = (0..n_elements)
        .map(|i| {
            BeamElement3D::new(
                [i, i + 1],
                positions[i],
                positions[i + 1],
                section,
                [0.0, 1.0, 0.0],
            )
        })
        .collect();
    let form = CorotationalBeamForm3D::new(elements, positions, (n_elements + 1) * 6);
    let n = form.num_global_dofs();
    let moment = section.eiz() * theta / l;
    let mut external = vec![0.0; n];
    external[n_elements * 6 + 5] = moment;
    let free: Vec<usize> = (6..n).collect();

    let mut u = vec![0.0; n];
    for step in 1..=10 {
        let scale = step as f64 / 10.0;
        let scaled: Vec<f64> = external.iter().map(|f| f * scale).collect();
        for _ in 0..25 {
            let mut residual = vec![0.0; n];
            let mut tangent = vec![0.0; n * n];
            for elem in 0..form.num_elements() {
                let map = form.local_to_global(elem);
                let local_r = form.element_residual(elem, &u);
                let local_k = form.element_tangent(elem, &u);
                for (i, &gi) in map.iter().enumerate() {
                    residual[gi] += local_r[i];
                    for (j, &gj) in map.iter().enumerate() {
                        tangent[gi * n + gj] += local_k[i * map.len() + j];
                    }
                }
            }
            for i in 0..n {
                residual[i] -= scaled[i];
            }
            let norm: f64 = free
                .iter()
                .map(|&d| residual[d] * residual[d])
                .sum::<f64>()
                .sqrt();
            let load: f64 = scaled
                .iter()
                .map(|f| f * f)
                .sum::<f64>()
                .sqrt()
                .max(1.0e-30);
            if norm / load < 1.0e-10 {
                break;
            }
            let m = free.len();
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
    }
    check(&doc, "tip_x", l + u[n_elements * 6]);
    check(&doc, "tip_y", u[n_elements * 6 + 1]);
}
