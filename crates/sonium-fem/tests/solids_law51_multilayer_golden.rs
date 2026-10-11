//! Wolfram golden replay: LAW51 JWL mixture.
//!
//! Replays `fixtures/law51_multilayer_golden.json` (derived by
//! `fixtures/law51_multilayer.wls` at 50-digit precision) through the
//! production JWL API. Regenerate with:
//! `wolframscript -script fixtures/law51_multilayer.wls > fixtures/law51_multilayer_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::law_foams::{JwlEos, Multimat51};

const GOLDEN: &str = include_str!("../fixtures/law51_multilayer_golden.json");

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
fn law51_jwl_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let eos = JwlEos::new(2.0e9, 5.0e8, 4.5, 1.2, 0.25, 1500.0, 1.0e5).unwrap();
    let reacted = JwlEos::new(1.0e9, 5.0e8, 4.5, 1.2, 0.25, 1500.0, 1.0e5).unwrap();
    check_all(&doc, "pressure", &[eos.pressure(0.3, 2.0e5)]);
    check_all(&doc, "dp_dmu", &[eos.dp_dmu(0.3, 2.0e5)]);
    check_all(&doc, "dp_de", &[eos.dp_de(0.3)]);
    let ramp = Multimat51::new(eos, reacted, 0.0, 2.0, 2.0).unwrap();
    check_all(
        &doc,
        "mixture",
        &[ramp.mixture_pressure(0.3, 2.0e5, 2.0e5, 0.25)],
    );
    check_all(&doc, "burn_ramp", &[ramp.burn_at(3.0)]);
    check_all(&doc, "burn_before", &[ramp.burn_at(1.0)]);
    let instant = Multimat51::new(eos, reacted, 0.0, 2.0, 0.0).unwrap();
    check_all(&doc, "burn_instant_on", &[instant.burn_at(2.0)]);
    check_all(&doc, "burn_instant_off", &[instant.burn_at(1.99)]);
}
