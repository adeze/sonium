//! Wolfram golden replay: pinched hemisphere deflections.
//!
//! Replays `fixtures/solids_shell_pinched_hemisphere_golden.json` (emitted by
//! `fixtures/solids_shell_pinched_hemisphere.wls`) through the production
//! CST+DKT shell assembly at the QA fine mesh (48x48). Regenerate with:
//! `wolframscript -script fixtures/solids_shell_pinched_hemisphere.wls > fixtures/solids_shell_pinched_hemisphere_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::mesh::Point;
use sonium_fem::physics::solid::shell::{
    ShellElement, ShellSection, add_shell_point_load, assemble_shell_stiffness,
};
use sonium_solvers::direct::SparseLuFactorization;

const GOLDEN: &str = include_str!("../fixtures/solids_shell_pinched_hemisphere_golden.json");

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

fn solve_quarter(n_circ: usize, n_mer: usize) -> (f64, f64) {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    let radius = doc["inputs"]["radius_m"].as_f64().unwrap();
    let thick = doc["inputs"]["thickness_m"].as_f64().unwrap();
    let hole_deg = doc["inputs"]["hole_deg"].as_f64().unwrap();
    let ymod = doc["inputs"]["youngs_modulus_pa"].as_f64().unwrap();
    let nu = doc["inputs"]["poisson_ratio"].as_f64().unwrap();
    let load = doc["inputs"]["pinch_load_n"].as_f64().unwrap();
    let mut points = Vec::with_capacity((n_circ + 1) * (n_mer + 1));
    for im in 0..=n_mer {
        let phi = (hole_deg + (90.0 - hole_deg) * im as f64 / n_mer as f64).to_radians();
        for ic in 0..=n_circ {
            let theta = (90.0 * ic as f64 / n_circ as f64).to_radians();
            points.push(Point::new_3d(
                radius * phi.sin() * theta.cos(),
                radius * phi.cos(),
                radius * phi.sin() * theta.sin(),
            ));
        }
    }
    let node = |im: usize, ic: usize| -> usize { im * (n_circ + 1) + ic };
    let mut quads = Vec::with_capacity(n_circ * n_mer);
    for im in 0..n_mer {
        for ic in 0..n_circ {
            quads.push([
                node(im, ic),
                node(im, ic + 1),
                node(im + 1, ic + 1),
                node(im + 1, ic),
            ]);
        }
    }
    let section = ShellSection::new(ymod, nu, thick, 0.0);
    let mut elements = Vec::with_capacity(quads.len() * 2);
    for q in &quads {
        elements.push(ShellElement::from_points(
            [q[0], q[1], q[2]],
            &points[q[0]],
            &points[q[1]],
            &points[q[2]],
            section,
        ));
        elements.push(ShellElement::from_points(
            [q[0], q[2], q[3]],
            &points[q[0]],
            &points[q[2]],
            &points[q[3]],
            section,
        ));
    }
    let n_dofs = points.len() * 6;
    let mut csr = assemble_shell_stiffness(&elements, n_dofs).to_csr();
    let mut rhs = vec![0.0; n_dofs];
    add_shell_point_load(
        &mut rhs,
        node(n_mer, 0),
        [-load / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    );
    add_shell_point_load(
        &mut rhs,
        node(n_mer, n_circ),
        [0.0, 0.0, load / 2.0, 0.0, 0.0, 0.0],
    );
    let mut fixed = Vec::new();
    for im in 0..=n_mer {
        let a = node(im, 0);
        fixed.extend([a * 6 + 2, a * 6 + 3, a * 6 + 4]);
        let b = node(im, n_circ);
        fixed.extend([b * 6, b * 6 + 4, b * 6 + 5]);
    }
    fixed.push(node(n_mer, n_circ / 2) * 6 + 1);
    for dof in std::collections::BTreeSet::from_iter(fixed.iter().copied()) {
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
    let u_in = -x[node(n_mer, 0) * 6];
    let u_out = x[node(n_mer, n_circ) * 6 + 2];
    (u_in, u_out)
}

#[test]
fn pinched_hemisphere_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let (u_in, u_out) = solve_quarter(48, 48);
    check(&doc, "radial_inward", u_in);
    check(&doc, "radial_outward", u_out);
}
