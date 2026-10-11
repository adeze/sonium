//! Wolfram golden replay: LAW50 rated yield tables.
//!
//! Replays `fixtures/law50_soc_golden.json` (derived by
//! `fixtures/law50_soc.wls` at 50-digit precision) through the production
//! LAW50 API. Regenerate with:
//! `wolframscript -script fixtures/law50_soc.wls > fixtures/law50_soc_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::funct::Funct1D;
use sonium_fem::physics::solid::plasticity::law_foams::HoneycombLaw;

const GOLDEN: &str = include_str!("../fixtures/law50_soc_golden.json");

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

fn rated_law() -> HoneycombLaw {
    let lo = Funct1D::new(vec![0.0, 1.0], vec![1.0, 3.0]).unwrap();
    let hi = Funct1D::new(vec![0.0, 1.0], vec![2.0, 4.0]).unwrap();
    let rated = core::array::from_fn(|_| vec![(lo.clone(), 0.0), (hi.clone(), 10.0)]);
    HoneycombLaw::law50([100.0; 6], rated).unwrap()
}

#[test]
fn law50_rated_tables_match_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let law = rated_law();
    check_all(&doc, "yield_mid_lo", &[law.yield_stress(0, 0.5, 0.0)]);
    check_all(&doc, "yield_mid_hi", &[law.yield_stress(0, 0.5, 10.0)]);
    check_all(&doc, "yield_mid_mid", &[law.yield_stress(0, 0.5, 5.0)]);
    check_all(&doc, "yield_qtr_mid", &[law.yield_stress(0, 0.25, 5.0)]);
    check_all(&doc, "yield_clamp_lo", &[law.yield_stress(0, 0.5, -3.0)]);
    check_all(&doc, "yield_clamp_hi", &[law.yield_stress(0, 0.5, 99.0)]);
    check_all(&doc, "yield_x0_r7", &[law.yield_stress(0, 0.0, 7.0)]);
    check_all(&doc, "yield_x1_r3", &[law.yield_stress(0, 1.0, 3.0)]);
}
