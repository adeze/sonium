//! Wolfram golden replay: LAW51 multimat strain paths.
//!
//! Replays `fixtures/solids_law51_paths_golden.json` (derived by
//! `fixtures/solids_law51_paths.wls` at 100-digit working precision,
//! exported at 50 digits) through the production LAW51 API. Regenerate
//! with:
//! `wolframscript -script fixtures/solids_law51_paths.wls > fixtures/solids_law51_paths_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::law_foams::{JwlEos, Multimat51, MultimatState};
use sonium_fem::physics::solid::plasticity::update::StressUpdateModel;

const GOLDEN: &str = include_str!("../fixtures/solids_law51_paths_golden.json");

/// Steps per leg, matching the derivation script.
const STEPS_PER_LEG: usize = 100;
/// Step time in s; burn ramps linearly over 200 steps.
const DT: f64 = 1.0e-6;
/// Burn duration in s, matching the derivation script.
const BURN_TIME: f64 = 200.0 * DT;
/// Uniaxial tension increment: mu reaches -0.02 over the leg.
const DE_TENSION: f64 = 2.0e-4;
/// Shear increment: volume frozen, burn mixing only.
const DG_SHEAR: f64 = 2.0e-3;
/// Volumetric increment per axis: mu reaches +0.3 over the leg.
const DV_VOL: f64 = 1.0e-3;
/// Initial specific energy in J/kg of both materials.
const E0: f64 = 1.0e5;

/// Accumulated result of one driven strain path.
struct PathResult {
    sigma: [f64; 6],
    pressure: f64,
    mu: f64,
    burn: f64,
    energy_u: f64,
    energy_r: f64,
    yielding: Vec<bool>,
    epbar_stations: Vec<f64>,
}

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

fn check_flags(doc: &Value, name: &str, actual: &[bool]) {
    let numeric: Vec<f64> = actual.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    check_all(doc, name, &numeric);
}

/// Builds the programmed-burn mixture of the derivation script.
fn burn_mixture() -> Multimat51 {
    let eos = JwlEos::new(2.0e9, 5.0e8, 4.5, 1.2, 0.25, 1500.0, E0).unwrap();
    let reacted = JwlEos::new(1.0e9, 5.0e8, 4.5, 1.2, 0.25, 1500.0, E0).unwrap();
    Multimat51::new(eos, reacted, 0.0, 0.0, BURN_TIME).unwrap()
}

/// Drives fixed strain increments, recording stations (1-based steps).
///
/// Each step runs the full production triple (`update`, `commit`,
/// `advance_totals`), exactly what `step` bundles. The law is rate
/// independent, so `update` (not `update_with_rate`) drives.
fn drive_fixed(
    law: &Multimat51,
    state: &mut MultimatState,
    steps: &[[f64; 6]],
    stations: &[usize],
) -> PathResult {
    let mut yielding = Vec::new();
    let mut epbar_stations = Vec::new();
    for (i, step) in steps.iter().enumerate() {
        let update = law.update(state, step);
        law.commit(state, &update);
        law.advance_totals(state, step, DT);
        if stations.contains(&(i + 1)) {
            yielding.push(update.yielding);
            epbar_stations.push(state.equiv_plastic);
        }
    }
    PathResult {
        sigma: update_stress(law, state),
        pressure: state.pressure,
        mu: state.mu,
        burn: state.burn,
        energy_u: state.energy_u,
        energy_r: state.energy_r,
        yielding,
        epbar_stations,
    }
}

/// Re-evaluates the committed end stress from the end state.
///
/// The production commit stores only the pressure; the end stress is
/// the hydrostatic stress of that pressure (shear-free by construction).
fn update_stress(law: &Multimat51, state: &MultimatState) -> [f64; 6] {
    let net = state.pressure - law.pressure_ext;
    [-net, -net, -net, 0.0, 0.0, 0.0]
}

fn check_leg(doc: &Value, prefix: &str, result: &PathResult) {
    check_all(doc, &format!("{prefix}_sigma"), &result.sigma);
    check_all(doc, &format!("{prefix}_pressure"), &[result.pressure]);
    check_all(doc, &format!("{prefix}_mu"), &[result.mu]);
    check_all(doc, &format!("{prefix}_burn"), &[result.burn]);
    check_all(doc, &format!("{prefix}_eu"), &[result.energy_u]);
    check_all(doc, &format!("{prefix}_er"), &[result.energy_r]);
    check_flags(doc, &format!("{prefix}_yielding"), &result.yielding);
    check_all(doc, &format!("{prefix}_epbar_st"), &result.epbar_stations);
}

#[test]
fn law51_paths_match_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["units"], "SI");
    let law = burn_mixture();
    assert!(!law.is_rate_dependent());
    let stations_100 = vec![20, 40, 60, 80, 100];
    let stations_cycle = vec![80, 160, 240, 320, 400];

    // Uniaxial-strain tension to mu = -0.02.
    let mut state = MultimatState::new(E0, E0);
    let uni = drive_fixed(
        &law,
        &mut state,
        &vec![[DE_TENSION, 0.0, 0.0, 0.0, 0.0, 0.0]; STEPS_PER_LEG],
        &stations_100,
    );
    check_leg(&doc, "uni", &uni);
    assert!(uni.yielding.iter().all(|&y| !y), "the mixture never yields");
    assert!(
        uni.pressure < law.unreacted.pressure(0.0, E0),
        "tension must depressurize below virgin"
    );

    // Pure shear: volume frozen, deviatoric response exactly zero.
    let mut state = MultimatState::new(E0, E0);
    let shear = drive_fixed(
        &law,
        &mut state,
        &vec![[0.0, 0.0, 0.0, DG_SHEAR, 0.0, 0.0]; STEPS_PER_LEG],
        &stations_100,
    );
    check_leg(&doc, "shear", &shear);
    assert_eq!(&shear.sigma[3..], &[0.0; 3], "shear must stay zero");
    assert_eq!(shear.mu, 0.0);
    assert_eq!(shear.energy_u, E0);
    assert_eq!(shear.energy_r, E0);

    // Volumetric compression to mu = +0.3: the mixture pressurizes.
    let mut state = MultimatState::new(E0, E0);
    let vol = drive_fixed(
        &law,
        &mut state,
        &vec![[-DV_VOL, -DV_VOL, -DV_VOL, 0.0, 0.0, 0.0]; STEPS_PER_LEG],
        &stations_100,
    );
    check_leg(&doc, "vol", &vol);
    assert!(
        vol.pressure > law.unreacted.pressure(0.0, E0),
        "compression must pressurize above virgin"
    );
    assert!(vol.energy_u > E0 && vol.energy_r > E0);

    // Symmetric volumetric cycle 0 -> +0.3 -> -0.3 -> 0.
    let mut dvs = vec![-DV_VOL; STEPS_PER_LEG];
    dvs.extend(vec![DV_VOL; 2 * STEPS_PER_LEG]);
    dvs.extend(vec![-DV_VOL; STEPS_PER_LEG]);
    let steps: Vec<[f64; 6]> = dvs.iter().map(|&d| [d, d, d, 0.0, 0.0, 0.0]).collect();
    let mut state = MultimatState::new(E0, E0);
    let cyc = drive_fixed(&law, &mut state, &steps, &stations_cycle);
    check_leg(&doc, "cycle", &cyc);
    assert_eq!(cyc.burn, 1.0, "the cycle must end fully burned");
    let closure = cyc.pressure - law.reacted.pressure(0.0, E0);
    check_all(&doc, "cycle_closure", &[closure]);
}
