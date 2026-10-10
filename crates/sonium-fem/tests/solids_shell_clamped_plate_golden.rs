//! Wolfram golden replay: clamped circular plate bending.
//!
//! Replays `fixtures/solids_shell_clamped_plate_golden.json` (emitted by
//! `fixtures/solids_shell_clamped_plate.wls`) through the production
//! `sonium_wave` plate closed forms and the DKT shell assembly at the QA
//! fine mesh (12x48). Regenerate with:
//! `wolframscript -script fixtures/solids_shell_clamped_plate.wls > fixtures/solids_shell_clamped_plate_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::mesh::{ElementType, circular_shell_mesh_triangles};
use sonium_fem::physics::solid::shell::{
    ShellElement, ShellSection, assemble_shell_load, assemble_shell_stiffness,
};
use sonium_solvers::direct::SparseLuFactorization;
use sonium_wave::analytical::manufactured_elasticity::shells::{
    plate_bending_stiffness, plate_clamped_circular_center_uniform,
};

const GOLDEN: &str = include_str!("../fixtures/solids_shell_clamped_plate_golden.json");

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

fn input(doc: &Value, name: &str) -> f64 {
    doc["inputs"][name].as_f64().unwrap()
}

fn dkt_center_deflection(radius: f64, thick: f64, young: f64, nu: f64, pressure: f64) -> f64 {
    let mesh = circular_shell_mesh_triangles(radius, 12, 48);
    let section = ShellSection::new(young, nu, thick, 7800.0);
    let elements: Vec<ShellElement> = mesh
        .elements
        .iter()
        .filter(|e| e.element_type == ElementType::Triangle)
        .map(|e| ShellElement::from_mesh([e.nodes[0], e.nodes[1], e.nodes[2]], &mesh, section))
        .collect();
    let n_dofs = mesh.nodes.len() * 6;
    let mut csr = assemble_shell_stiffness(&elements, n_dofs).to_csr();
    let mut rhs = assemble_shell_load(&elements, n_dofs, |_| -pressure);
    let mut fixed = Vec::new();
    for (i, p) in mesh.nodes.iter().enumerate() {
        if (p.x.hypot(p.y) - radius).abs() < 1e-9 {
            for d in 0..6 {
                fixed.push(i * 6 + d);
            }
        }
    }
    assert!(!fixed.is_empty(), "rim nodes must be found");
    for &dof in &fixed {
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
    let center = mesh
        .nodes
        .iter()
        .position(|p| p.x.hypot(p.y) < 1e-12)
        .expect("center node must exist");
    x[center * 6 + 2]
}

#[test]
fn clamped_plate_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let a = input(&doc, "radius_m");
    let t = input(&doc, "thickness_m");
    let e = input(&doc, "youngs_modulus_pa");
    let nu = input(&doc, "poisson_ratio");
    let q = input(&doc, "pressure_load_pa");
    let r = input(&doc, "probe_radius_m");
    let rigidity = plate_bending_stiffness(e, nu, t);
    check(&doc, "flexural_rigidity", rigidity);
    check(
        &doc,
        "center_deflection",
        -plate_clamped_circular_center_uniform(q, a, rigidity),
    );
    // Hoop moment closed form (Timoshenko plates, axisymmetric bending).
    let hoop = q / 16.0 * ((1.0 + nu) * a.powi(2) - (1.0 + 3.0 * nu) * r.powi(2));
    check(&doc, "hoop_moment", hoop);
    check(&doc, "probe_m1", hoop / t.powi(2));
    check(&doc, "center_deflection", dkt_center_deflection(a, t, e, nu, q));
}
