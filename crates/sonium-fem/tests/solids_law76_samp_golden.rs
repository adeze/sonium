//! Wolfram golden replay: LAW76 SAMP triaxiality yield.
//!
//! Replays `fixtures/law76_samp_golden.json` (derived by
//! `fixtures/law76_samp.wls` at 50-digit precision) through the
//! production SAMP API. Regenerate with:
//! `wolframscript -script fixtures/law76_samp.wls > fixtures/law76_samp_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::funct::Funct1D;
use sonium_fem::physics::solid::plasticity::law_foams::{Samp76, SampState};
use sonium_fem::physics::solid::plasticity::update::von_mises_c1;

const GOLDEN: &str = include_str!("../fixtures/law76_samp_golden.json");

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

fn flat(value: f64) -> Funct1D {
    Funct1D::new(vec![0.0, 1.0], vec![value, value]).unwrap()
}

#[test]
fn law76_samp_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let law = Samp76::new(1000.0, 0.3, flat(10.0), flat(20.0), flat(6.0)).unwrap();
    check_all(&doc, "station_tension", &[law.yield_at(0.0, -1.0 / 3.0)]);
    check_all(&doc, "station_shear", &[law.yield_at(0.0, 0.0)]);
    check_all(&doc, "station_compression", &[law.yield_at(0.0, 1.0 / 3.0)]);
    check_all(&doc, "station_mid", &[law.yield_at(0.0, -1.0 / 6.0)]);
    check_all(
        &doc,
        "station_biaxial_end",
        &[law.yield_at(0.0, -2.0 / 3.0)],
    );
    check_all(&doc, "station_clamp", &[law.yield_at(0.0, 1.0)]);
    let elastic = Samp76::new(1000.0, 0.3, flat(1.0e18), flat(1.0e18), flat(1.0e18)).unwrap();
    let mut estate = SampState::new();
    let trial = elastic.step(&mut estate, &[0.02, -0.006, -0.006, 0.0, 0.0, 0.0], 1.0);
    assert!(!trial.yielding);
    check_all(&doc, "trial_q", &[von_mises_c1(&trial.stress)]);
    let mut state = SampState::new();
    let update = law.step(&mut state, &[0.02, -0.006, -0.006, 0.0, 0.0, 0.0], 1.0);
    assert!(update.yielding);
    check_all(&doc, "return_dep", &[update.delta_equiv_plastic_strain]);
    check_all(&doc, "return_sxx", &[update.stress[0]]);
}
