//! Wolfram golden replay: tapered cantilever.
//!
//! Replays `fixtures/solids_beam_tapered_cantilever_golden.json` (derived by
//! `fixtures/solids_beam_tapered_cantilever.wls`) through the production
//! `TaperedBeamElement3D` element. Regenerate with:
//! `wolframscript -script fixtures/solids_beam_tapered_cantilever.wls > fixtures/solids_beam_tapered_cantilever_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::beam::{Section3D, TaperedBeamElement3D};

const GOLDEN: &str = include_str!("../fixtures/solids_beam_tapered_cantilever_golden.json");

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

fn dense_solve(matrix: &[f64], rhs: &[f64], n: usize) -> Vec<f64> {
    // Jacobi equilibration plus one refinement step. Fine cantilever chains
    // carry bending entries near 1e16 whose products cancel ten orders in
    // every residual, stalling plain elimination near 1e-5; scaling the
    // diagonal to one restores a clean residual and refinement finishes it.
    let mut scale = vec![0.0; n];
    for i in 0..n {
        scale[i] = matrix[i * n + i].abs().sqrt().max(1.0e-300);
    }
    let mut balanced = vec![0.0; n * n];
    let mut scaled_rhs = vec![0.0; n];
    for i in 0..n {
        scaled_rhs[i] = rhs[i] / scale[i];
        for j in 0..n {
            balanced[i * n + j] = matrix[i * n + j] / (scale[i] * scale[j]);
        }
    }
    let mut y = eliminate(&balanced, &scaled_rhs, n);
    let mut residual = vec![0.0; n];
    for i in 0..n {
        let mut sum = scaled_rhs[i];
        for j in 0..n {
            sum -= balanced[i * n + j] * y[j];
        }
        residual[i] = sum;
    }
    let correction = eliminate(&balanced, &residual, n);
    for i in 0..n {
        y[i] += correction[i];
    }
    y.iter().zip(scale.iter()).map(|(v, s)| v / s).collect()
}

fn eliminate(matrix: &[f64], rhs: &[f64], n: usize) -> Vec<f64> {
    let mut m = matrix.to_vec();
    let mut x = rhs.to_vec();
    for col in 0..n {
        let mut pivot = col;
        let mut best = m[col * n + col].abs();
        for row in col + 1..n {
            let candidate = m[row * n + col].abs();
            if candidate > best {
                best = candidate;
                pivot = row;
            }
        }
        assert!(best > 1.0e-300, "singular golden system");
        if pivot != col {
            for k in col..n {
                m.swap(col * n + k, pivot * n + k);
            }
            x.swap(col, pivot);
        }
        let diag = m[col * n + col];
        for row in col + 1..n {
            let factor = m[row * n + col] / diag;
            m[row * n + col] = 0.0;
            for k in col + 1..n {
                m[row * n + k] -= factor * m[col * n + k];
            }
            x[row] -= factor * x[col];
        }
    }
    for row in (0..n).rev() {
        let mut sum = x[row];
        for k in row + 1..n {
            sum -= m[row * n + k] * x[k];
        }
        x[row] = sum / m[row * n + row];
    }
    x
}

#[test]
fn tapered_cantilever_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["case"], "beam_tapered_cantilever");
    assert_eq!(doc["meta"]["units"], "SI");

    let l = doc["inputs"]["length_m"].as_f64().unwrap();
    let p = doc["inputs"]["tip_load_n"].as_f64().unwrap();
    let p_ax = doc["inputs"]["axial_load_n"].as_f64().unwrap();
    let e = doc["inputs"]["youngs_modulus_pa"].as_f64().unwrap();
    let h = doc["inputs"]["height_m"].as_f64().unwrap();
    let b0 = doc["inputs"]["root_width_m"].as_f64().unwrap();
    let b1 = doc["inputs"]["tip_width_m"].as_f64().unwrap();

    // Taper follows b(x) linearly; each element spans its own tapered range.
    // Quadrature is exact for the linear-Iz element interpolation, so the
    // remaining error is the linear-vs-cubic Iz model error converging
    // quadratically in element size (~3e-6 at 256 elements).
    let n_elements = 256;
    let dx = l / n_elements as f64;
    let width_at = |i: usize| b0 + (b1 - b0) * i as f64 / n_elements as f64;
    let elements: Vec<TaperedBeamElement3D> = (0..n_elements)
        .map(|i| {
            TaperedBeamElement3D::new(
                [i, i + 1],
                [i as f64 * dx, 0.0, 0.0],
                [(i + 1) as f64 * dx, 0.0, 0.0],
                Section3D::rectangular(e, 0.3, width_at(i), h, 7800.0, false),
                Section3D::rectangular(e, 0.3, width_at(i + 1), h, 7800.0, false),
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
    for (i, &gi) in free.iter().enumerate() {
        for (j, &gj) in free.iter().enumerate() {
            matrix[i * m + j] = k[gi * n + gj];
        }
    }
    // Transverse tip load: deflection plus slope.
    let mut rhs = vec![0.0; m];
    rhs[m - 5] = p;
    let u = dense_solve(&matrix, &rhs, m);
    check(&doc, "tip_deflection", u[m - 5]);
    check(&doc, "tip_slope", u[m - 1]);
    // Axial tip load: tapered-bar extension.
    let mut rhs_ax = vec![0.0; m];
    rhs_ax[m - 6] = p_ax;
    let u_ax = dense_solve(&matrix, &rhs_ax, m);
    check(&doc, "axial_tip", u_ax[m - 6]);
}
