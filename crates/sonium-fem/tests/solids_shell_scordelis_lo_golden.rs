//! Wolfram golden replay: Scordelis-Lo roof free-edge deflection.
//!
//! Replays `fixtures/solids_shell_scordelis_lo_golden.json` (emitted by
//! `fixtures/solids_shell_scordelis_lo.wls`) through the production MITC4
//! shell assembly at the QA fine mesh (20x20). Regenerate with:
//! `wolframscript -script fixtures/solids_shell_scordelis_lo.wls > fixtures/solids_shell_scordelis_lo_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::mesh::Point;
use sonium_fem::physics::solid::shell::{
    QuadShellElement, ShellSection, assemble_quad_shell_stiffness,
};
use sonium_solvers::direct::SparseLuFactorization;

const GOLDEN: &str = include_str!("../fixtures/solids_shell_scordelis_lo_golden.json");

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

fn solve_quarter(n_theta: usize, n_z: usize) -> f64 {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    let radius = doc["inputs"]["radius_m"].as_f64().unwrap();
    let length = doc["inputs"]["length_m"].as_f64().unwrap();
    let thick = doc["inputs"]["thickness_m"].as_f64().unwrap();
    let ymod = doc["inputs"]["youngs_modulus_pa"].as_f64().unwrap();
    let nu = doc["inputs"]["poisson_ratio"].as_f64().unwrap();
    let load = doc["inputs"]["pressure_load_pa"].as_f64().unwrap();
    let half_angle = doc["inputs"]["half_angle_deg"]
        .as_f64()
        .unwrap()
        .to_radians();
    let mut points = Vec::with_capacity((n_theta + 1) * (n_z + 1));
    for iz in 0..=n_z {
        let z = length / 2.0 * iz as f64 / n_z as f64;
        for it in 0..=n_theta {
            let theta = half_angle * it as f64 / n_theta as f64;
            points.push(Point::new_3d(radius * theta.sin(), radius * theta.cos(), z));
        }
    }
    let node = |iz: usize, it: usize| -> usize { iz * (n_theta + 1) + it };
    let mut quads = Vec::with_capacity(n_theta * n_z);
    for iz in 0..n_z {
        for it in 0..n_theta {
            quads.push([
                node(iz, it),
                node(iz, it + 1),
                node(iz + 1, it + 1),
                node(iz + 1, it),
            ]);
        }
    }
    let section = ShellSection::new(ymod, nu, thick, 0.0);
    let elements: Vec<QuadShellElement> = quads
        .iter()
        .map(|q| {
            QuadShellElement::from_points_with_section(
                *q,
                &points[q[0]],
                &points[q[1]],
                &points[q[2]],
                &points[q[3]],
                section,
            )
        })
        .collect();
    let n_dofs = points.len() * 6;
    let mut csr = assemble_quad_shell_stiffness(&elements, n_dofs).to_csr();
    let mut rhs = vec![0.0; n_dofs];
    for elem in &elements {
        let nodal = load * elem.area / 4.0;
        for &node in &elem.nodes {
            rhs[node * 6 + 1] -= nodal;
        }
    }
    let mut fixed = Vec::new();
    for it in 0..=n_theta {
        let n = node(0, it);
        fixed.push(n * 6);
        fixed.push(n * 6 + 1);
    }
    for it in 0..=n_theta {
        let n = node(n_z, it);
        fixed.push(n * 6 + 2);
        fixed.push(n * 6 + 3);
        fixed.push(n * 6 + 4);
    }
    for iz in 0..=n_z {
        let n = node(iz, 0);
        fixed.push(n * 6);
        fixed.push(n * 6 + 4);
        fixed.push(n * 6 + 5);
    }
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
    x[node(n_z, n_theta) * 6 + 1]
}

#[test]
fn scordelis_lo_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    check(&doc, "free_edge_deflection", solve_quarter(20, 20));
}
