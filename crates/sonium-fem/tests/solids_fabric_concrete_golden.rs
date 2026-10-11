//! Wolfram golden replay: fabric membranes and concrete.
//!
//! Replays `fixtures/fabric_concrete_golden.json` (derived by
//! `fixtures/fabric_concrete.wls` at 50-digit precision) through the
//! production fabric and concrete APIs. Regenerate with:
//! `wolframscript -script fixtures/fabric_concrete.wls > fixtures/fabric_concrete_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::funct::Funct1D;
use sonium_fem::physics::solid::plasticity::law_foams::{
    Concrete24, ConcreteState, FabricDirection, FabricLaw, FabricState, MazarsDamage, pressure_c1,
};
use sonium_fem::physics::solid::plasticity::update::StressUpdateModel;

const GOLDEN: &str = include_str!("../fixtures/fabric_concrete_golden.json");

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

fn fabric19() -> FabricLaw {
    FabricLaw::law19(
        [10.0e6, 10.0e6],
        [1.0e6, 1.0e6],
        2.0e6,
        [0.1, 0.1],
        [0.2, 0.2],
        0.3,
    )
    .unwrap()
}

fn mazars() -> MazarsDamage {
    MazarsDamage {
        strain0: 1.0e-4,
        at: 0.8,
        bt: 1.5e4,
        ac: 1.0,
        bc: 2.0e3,
    }
}

#[test]
fn fabric_concrete_matches_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    let law19 = fabric19();
    let mut tension = FabricState::new();
    let pulled = law19.step(&mut tension, &[0.01, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
    check_all(&doc, "fabric_tension", &[pulled.stress[0]]);
    let mut compression = FabricState::new();
    let pushed = law19.step(&mut compression, &[-0.01, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
    check_all(&doc, "fabric_compression", &[pushed.stress[0]]);
    let mut shear = FabricState::new();
    let sheared = law19.step(&mut shear, &[0.0, 0.0, 0.0, 0.01, 0.0, 0.0], 1.0);
    check_all(&doc, "fabric_shear", &[sheared.stress[3]]);
    let warp = FabricDirection::new(
        Funct1D::new(vec![0.0, 0.1], vec![0.0, 1.0e6]).unwrap(),
        20.0e6,
    )
    .unwrap();
    let law58 = FabricLaw::law58(
        [10.0e6, 10.0e6],
        [1.0e6, 1.0e6],
        2.0e6,
        [0.1, 0.1],
        [0.2, 0.2],
        0.3,
        warp.clone(),
        warp,
    )
    .unwrap();
    let mut cloth = FabricState::new();
    let loaded = law58.step(&mut cloth, &[0.05, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
    check_all(&doc, "fabric_loaded", &[loaded.stress[0]]);
    let unloaded = law58.step(&mut cloth, &[-0.03, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
    check_all(&doc, "fabric_unloaded", &[unloaded.stress[0]]);
    let mazars = mazars();
    check_all(
        &doc,
        "mazars_uniaxial",
        &[mazars.damage(&[2.0e-4, 0.0, 0.0])],
    );
    check_all(
        &doc,
        "mazars_below_threshold",
        &[mazars.damage(&[5.0e-5, 0.0, 0.0])],
    );
    let concrete = Concrete24::new(30.0e9, 0.2, mazars, 0.0, 30.0e6, 0.0).unwrap();
    let mut block = ConcreteState::new();
    let capped = concrete.step(&mut block, &[-0.01, -0.01, -0.01, 0.0, 0.0, 0.0], 1.0);
    check_all(&doc, "concrete_cap", &[pressure_c1(&capped.stress)]);
    check_all(&doc, "concrete_trial_pressure", &[concrete.bulk() * 0.03]);
    let slow = Concrete24::new(30.0e9, 0.2, mazars, 1.0, 30.0e6, 0.0).unwrap();
    let vstate = ConcreteState::new();
    let dstrain = [1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0];
    let inviscid = slow.update(&vstate, &dstrain);
    let blended = slow.update_with_rate(&vstate, &dstrain, 1.0);
    check_all(
        &doc,
        "concrete_visco_ratio",
        &[blended.stress[0] / inviscid.stress[0]],
    );
}
