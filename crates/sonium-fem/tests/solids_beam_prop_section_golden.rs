//! Wolfram golden replay: spring property mapping.
//!
//! Replays `fixtures/solids_beam_prop_section_golden.json` (derived by
//! `fixtures/solids_beam_prop_section.wls` at 50-digit precision)
//! through the production property-mapping API. Regenerate with:
//! `wolframscript -script fixtures/solids_beam_prop_section.wls > fixtures/solids_beam_prop_section_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::beam::{
    PreLock, SpringAxiProp, SpringGeneProp, SpringMode, SpringPreProp,
};

const GOLDEN: &str = include_str!("../fixtures/solids_beam_prop_section_golden.json");

fn golden_values(doc: &Value, name: &str) -> Vec<f64> {
    doc["values"][name]["values"]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks {name}"))
        .iter()
        .map(|v| {
            v.as_str()
                .unwrap_or_else(|| panic!("golden lacks {name}"))
                .parse()
                .unwrap()
        })
        .collect()
}

fn check(doc: &Value, name: &str, actual: f64) {
    let expected = golden_values(doc, name);
    assert_eq!(expected.len(), 1, "{name}: length mismatch");
    let want = expected[0];
    let abs: f64 = doc["tolerances"][format!("{name}_abs")].as_f64().unwrap();
    let rel: f64 = doc["tolerances"][format!("{name}_rel")].as_f64().unwrap();
    assert!(
        (actual - want).abs() <= abs + rel * want.abs(),
        "{name}: production {actual:.6e} vs golden {want:.6e}"
    );
}

#[test]
fn beam_prop_section_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");

    let stiff = SpringMode::new(3.0e6, 0.0);
    let shear = SpringMode::new(1.0e6, 0.0);
    let torsion = SpringMode::new(2.0e4, 0.0);
    let bend = SpringMode::new(5.0e4, 0.0);
    let gene = SpringGeneProp::new(7.8, 0.0, stiff, shear, shear, torsion, bend, bend);
    let section = gene.to_section(2.0, 200.0e9, 0.3, 7800.0);
    check(&doc, "gene_area", section.area);
    check(&doc, "gene_modulus", section.youngs_modulus);
    check(&doc, "gene_shear_modulus", section.shear_modulus);
    check(&doc, "gene_torsion", section.torsion_constant);
    check(&doc, "gene_iy", section.iy);
    check(&doc, "gene_kappa", section.kappa_y);

    let tension = SpringMode::new(2.0e6, 0.0);
    let axi_shear = SpringMode::new(8.0e5, 0.0);
    let axi_torsion = SpringMode::new(1.0e4, 0.0);
    let axi_bend = SpringMode::new(3.0e4, 0.0);
    let axi = SpringAxiProp::new(3.9, 0.0, tension, axi_shear, axi_torsion, axi_bend);
    let axi_section = axi.to_section(1.0, 200.0e9, 0.3, 7800.0);
    check(&doc, "axi_area", axi_section.area);
    check(&doc, "axi_torsion", axi_section.torsion_constant);
    check(&doc, "axi_inertia", axi_section.iy);

    let pre = SpringPreProp::new(
        0.05,
        PreLock::OnSlideOrZeroForce,
        1.0e5,
        100.0,
        0.25,
        12.5,
        400.0,
    );
    let planar = pre.to_beam_section(0.25, 200.0e9, 7800.0);
    check(&doc, "pre_ea", planar.ea());
    check(&doc, "pre_area", planar.area);
    check(&doc, "pre_modulus", planar.youngs_modulus);
}
