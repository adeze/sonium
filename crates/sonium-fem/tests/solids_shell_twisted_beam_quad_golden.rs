//! Wolfram golden replay: twisted beam tip deflections with MITC4 quads.
//!
//! Replays `fixtures/solids_shell_twisted_beam_golden.json` (emitted by
//! `fixtures/solids_shell_twisted_beam.wls`) through the production warp-aware
//! MITC4 quad assembly at the QA fine mesh (32x8). Regenerate with:
//! `wolframscript -script fixtures/solids_shell_twisted_beam.wls > fixtures/solids_shell_twisted_beam_golden.json`.
//! The CST+DKT triangle replay lives in `solids_shell_twisted_beam_golden.rs`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::mesh::Point;
use sonium_fem::physics::solid::shell::{
    QuadShellElement, ShellSection, add_quad_shell_point_load, assemble_quad_shell_stiffness,
};
use sonium_solvers::direct::SparseLuFactorization;

const GOLDEN: &str = include_str!("../fixtures/solids_shell_twisted_beam_golden.json");

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

fn solve_tip(n_x: usize, n_y: usize, load: [f64; 6]) -> [f64; 6] {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    let length = doc["inputs"]["length_m"].as_f64().unwrap();
    let width = doc["inputs"]["width_m"].as_f64().unwrap();
    let thick = doc["inputs"]["thickness_m"].as_f64().unwrap();
    let ymod = doc["inputs"]["youngs_modulus_pa"].as_f64().unwrap();
    let nu = doc["inputs"]["poisson_ratio"].as_f64().unwrap();
    let total = doc["inputs"]["twist_deg"].as_f64().unwrap().to_radians();
    let mut points = Vec::with_capacity((n_x + 1) * (n_y + 1));
    for ix in 0..=n_x {
        let x = length * ix as f64 / n_x as f64;
        let twist = total * ix as f64 / n_x as f64;
        for iy in 0..=n_y {
            let s = width * (iy as f64 / n_y as f64 - 0.5);
            points.push(Point::new_3d(x, s * twist.cos(), s * twist.sin()));
        }
    }
    let node = |ix: usize, iy: usize| -> usize { ix * (n_y + 1) + iy };
    let section = ShellSection::new(ymod, nu, thick, 0.0);
    let mut elements = Vec::with_capacity(n_x * n_y);
    for ix in 0..n_x {
        for iy in 0..n_y {
            let q = [
                node(ix, iy),
                node(ix + 1, iy),
                node(ix + 1, iy + 1),
                node(ix, iy + 1),
            ];
            elements.push(QuadShellElement::from_points_with_section(
                q,
                &points[q[0]],
                &points[q[1]],
                &points[q[2]],
                &points[q[3]],
                section,
            ));
        }
    }
    let n_dofs = points.len() * 6;
    let mut csr = assemble_quad_shell_stiffness(&elements, n_dofs).to_csr();
    let mut rhs = vec![0.0; n_dofs];
    let tip = n_x * (n_y + 1) + n_y / 2;
    add_quad_shell_point_load(&mut rhs, tip, load);
    let mut fixed = Vec::new();
    for iy in 0..=n_y {
        for d in 0..6 {
            fixed.push(iy * 6 + d);
        }
    }
    for &dof in fixed.iter() {
        let range = csr.row_range(dof);
        for i in range {
            csr.values[i] = 0.0;
        }
        let diag = csr
            .row_entries(dof)
            .position(|(j, _)| j == dof)
            .expect("shell CSR rows must carry a diagonal");
        csr.values[csr.row_ptrs[dof] + diag] = 1.0;
        rhs[dof] = 0.0;
    }
    let lu = SparseLuFactorization::factorize(&csr).expect("sparse LU must factorize");
    let x = lu.solve(&rhs).expect("sparse LU must solve");
    [
        x[tip * 6],
        x[tip * 6 + 1],
        x[tip * 6 + 2],
        x[tip * 6 + 3],
        x[tip * 6 + 4],
        x[tip * 6 + 5],
    ]
}

#[test]
fn twisted_beam_quad_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let load = doc["inputs"]["tip_load_n"].as_f64().unwrap();
    let tip_in = solve_tip(32, 8, [0.0, 0.0, load, 0.0, 0.0, 0.0]);
    let tip_out = solve_tip(32, 8, [0.0, load, 0.0, 0.0, 0.0, 0.0]);
    check(&doc, "tip_in_plane", tip_in[2]);
    check(&doc, "tip_out_of_plane", tip_out[1]);
}
