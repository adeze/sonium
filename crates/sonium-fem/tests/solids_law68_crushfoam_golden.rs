//! Wolfram golden replay: LAW68 Cosserat crushable foam.
//!
//! Replays `fixtures/law68_crushfoam_golden.json` (derived by
//! `fixtures/law68_crushfoam.wls` at 50-digit precision) through the
//! production LAW68 API. Regenerate with:
//! `wolframscript -script fixtures/law68_crushfoam.wls > fixtures/law68_crushfoam_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::law_foams::{Cosserat68, CosseratState};

const GOLDEN: &str = include_str!("../fixtures/law68_crushfoam_golden.json");

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

fn check_all(doc: &Value, name: &str, actual: &[f64]) {
    let expected = golden_values(doc, name);
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    let abs: f64 = doc["tolerances"][format!("{name}_abs")].as_f64().unwrap();
    let rel: f64 = doc["tolerances"][format!("{name}_rel")].as_f64().unwrap();
    for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= abs + rel * want.abs(),
            "{name}[{i}]: production {got:.6e} vs golden {want:.6e}"
        );
    }
}

#[test]
fn law68_crushfoam_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let law = Cosserat68::new(2.0e9, 0.1, 1.5, 5.0e6, 50.0e6, 1.1e9, 0.1).unwrap();
    check_all(&doc, "couple_modulus", &[law.couple_modulus]);
    check_all(&doc, "crush_virgin", &[law.foam.crush_strength(0.0)]);
    check_all(&doc, "crush_tenth", &[law.foam.crush_strength(0.1)]);
    let mut elastic = CosseratState::new();
    law.step_curve(&mut elastic, &[0.0; 6], &[1.0e-4, 0.0, 0.0], 1.0);
    check_all(&doc, "couple_elastic", &[elastic.couple[0]]);
    let mut capped = CosseratState::new();
    let update = law.step_curve(&mut capped, &[0.0; 6], &[0.1, 0.0, 0.0], 1.0);
    check_all(&doc, "couple_capped", &[capped.couple[0]]);
    check_all(&doc, "couple_flow", &[update.delta_equiv_plastic_strain]);
}
