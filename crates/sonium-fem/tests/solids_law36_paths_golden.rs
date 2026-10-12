//! Wolfram golden replay: LAW36 tabulated-metal strain paths.
//!
//! Replays `fixtures/solids_law36_paths_golden.json` (derived by
//! `fixtures/solids_law36_paths.wls` at 200-digit working precision,
//! exported at 50 digits) through the production LAW36 API. Regenerate
//! with:
//! `wolframscript -script fixtures/solids_law36_paths.wls > fixtures/solids_law36_paths_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::law_metals::{Law36, Law36State};
use sonium_fem::physics::solid::plasticity::update::{StressUpdateModel, von_mises_c1};

const GOLDEN: &str = include_str!("../fixtures/solids_law36_paths_golden.json");

/// Steps per leg, matching the derivation script.
const STEPS_PER_LEG: usize = 100;
/// Lateral bisection passes per axial step, matching the derivation.
const LATERAL_ITERATIONS: usize = 50;
/// Axial extent of every path: 0.05 drives the card-aluminum curve to
/// 4.7% plastic strain on the monotonic legs (inside the tabulated
/// 7.5% range) and into flat end-slope extrapolation on the cycle.
const EMAX: f64 = 0.05;

/// Accumulated result of one driven strain path.
struct PathResult {
    sigma: [f64; 6],
    epbar: f64,
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

/// Solves the lateral strain of one axial step for zero lateral stress.
///
/// Bisects the pure production `update` (50 passes over a widening
/// bracket), mirroring the derivation script exactly.
fn solve_lateral(law: &Law36, state: &Law36State, de: f64) -> f64 {
    let (mut lo, mut hi) = if de >= 0.0 { (-de, 0.0) } else { (0.0, -de) };
    let probe = |dl: f64| law.update(state, &[de, dl, dl, 0.0, 0.0, 0.0]).stress[1];
    let mut blo = probe(lo);
    let mut bhi = probe(hi);
    for _ in 0..20 {
        if blo < 0.0 && bhi > 0.0 {
            break;
        }
        let width = (hi - lo).abs();
        lo -= width;
        hi += width;
        blo = probe(lo);
        bhi = probe(hi);
    }
    assert!(blo < 0.0 && bhi > 0.0, "lateral bracket failed for de={de}");
    for _ in 0..LATERAL_ITERATIONS {
        let mid = 0.5 * (lo + hi);
        if probe(mid) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Drives fixed strain increments, recording stations (1-based steps).
fn drive_fixed(
    law: &Law36,
    state: &mut Law36State,
    steps: &[[f64; 6]],
    stations: &[usize],
) -> PathResult {
    let mut yielding = Vec::new();
    let mut epbar_stations = Vec::new();
    for (i, step) in steps.iter().enumerate() {
        let update = law.update(state, step);
        law.commit(state, &update);
        if stations.contains(&(i + 1)) {
            yielding.push(update.yielding);
            epbar_stations.push(state.equiv_plastic_strain);
        }
    }
    PathResult {
        sigma: state.stress,
        epbar: state.equiv_plastic_strain,
        yielding,
        epbar_stations,
    }
}

/// Drives axial increments with the lateral solve, recording stations.
fn drive_axial(law: &Law36, state: &mut Law36State, des: &[f64], stations: &[usize]) -> PathResult {
    let mut yielding = Vec::new();
    let mut epbar_stations = Vec::new();
    for (i, &de) in des.iter().enumerate() {
        let dl = solve_lateral(law, state, de);
        let update = law.update(state, &[de, dl, dl, 0.0, 0.0, 0.0]);
        law.commit(state, &update);
        if stations.contains(&(i + 1)) {
            yielding.push(update.yielding);
            epbar_stations.push(state.equiv_plastic_strain);
        }
    }
    PathResult {
        sigma: state.stress,
        epbar: state.equiv_plastic_strain,
        yielding,
        epbar_stations,
    }
}

#[test]
fn law36_paths_match_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["units"], "SI");
    let law = Law36::card_aluminum();
    let stations_100 = vec![20, 40, 60, 80, 100];
    let stations_cycle = vec![80, 160, 240, 320, 400];

    // Uniaxial extension with solved lateral strain.
    let de = EMAX / STEPS_PER_LEG as f64;
    let mut state = Law36State::new();
    let uni = drive_axial(&law, &mut state, &vec![de; STEPS_PER_LEG], &stations_100);
    check_all(&doc, "uni_sigma", &uni.sigma);
    check_all(&doc, "uni_epbar", &[uni.epbar]);
    check_flags(&doc, "uni_yielding", &uni.yielding);
    check_all(&doc, "uni_epbar_st", &uni.epbar_stations);
    assert!(uni.yielding.iter().any(|&y| y), "uniaxial must yield");

    // Pure shear ramp.
    let dg = EMAX / STEPS_PER_LEG as f64;
    let mut state = Law36State::new();
    let shear = drive_fixed(
        &law,
        &mut state,
        &vec![[0.0, 0.0, 0.0, dg, 0.0, 0.0]; STEPS_PER_LEG],
        &stations_100,
    );
    check_all(&doc, "shear_sigma", &shear.sigma);
    check_all(&doc, "shear_epbar", &[shear.epbar]);
    check_flags(&doc, "shear_yielding", &shear.yielding);
    check_all(&doc, "shear_epbar_st", &shear.epbar_stations);
    assert!(shear.yielding.iter().any(|&y| y), "shear must yield");

    // Volumetric ramp: J2 never yields hydrostatically.
    let dv = EMAX / STEPS_PER_LEG as f64;
    let mut state = Law36State::new();
    let vol = drive_fixed(
        &law,
        &mut state,
        &vec![[dv, dv, dv, 0.0, 0.0, 0.0]; STEPS_PER_LEG],
        &stations_100,
    );
    check_all(&doc, "vol_sigma", &vol.sigma);
    check_all(&doc, "vol_epbar", &[vol.epbar]);
    check_flags(&doc, "vol_yielding", &vol.yielding);
    check_all(&doc, "vol_epbar_st", &vol.epbar_stations);
    assert!(
        vol.yielding.iter().all(|&y| !y),
        "volumetric must stay elastic"
    );
    assert_eq!(vol.epbar, 0.0);

    // Symmetric strain cycle with solved lateral strain.
    let mut des = vec![de; STEPS_PER_LEG];
    des.extend(vec![-de; 2 * STEPS_PER_LEG]);
    des.extend(vec![de; STEPS_PER_LEG]);
    let mut state = Law36State::new();
    let cyc = drive_axial(&law, &mut state, &des, &stations_cycle);
    check_all(&doc, "cycle_sigma", &cyc.sigma);
    check_all(&doc, "cycle_epbar", &[cyc.epbar]);
    check_flags(&doc, "cycle_yielding", &cyc.yielding);
    check_all(&doc, "cycle_epbar_st", &cyc.epbar_stations);
    assert!(cyc.yielding.iter().any(|&y| y), "cycle must yield");
    let closure = von_mises_c1(&cyc.sigma) - law.flow_stress(cyc.epbar, 0.0, 0.0);
    check_all(&doc, "cycle_closure", &[closure]);
    assert!(
        closure.abs() <= 1.0,
        "cycle must close on the surface, got {closure:.3e}"
    );

    // Failure-flag station at the cycle end: open limits stay shut.
    check_flags(&doc, "failure_ruptured", &[state.ruptured]);
    check_flags(&doc, "failure_deleted", &[state.deleted]);
    check_all(&doc, "failure_tscale", &[state.tensile_scale]);
}
