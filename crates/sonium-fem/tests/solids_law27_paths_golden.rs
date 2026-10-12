//! Wolfram golden replay: LAW27 brittle-damage strain paths.
//!
//! Replays `fixtures/solids_law27_paths_golden.json` (derived by
//! `fixtures/solids_law27_paths.wls` at 400-digit working precision,
//! exported at 50 digits) through the production LAW27 API. Regenerate
//! with:
//! `wolframscript -script fixtures/solids_law27_paths.wls > fixtures/solids_law27_paths_golden.json`.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::law_metals::{Law27, Law27State};
use sonium_fem::physics::solid::plasticity::update::{StressUpdateModel, von_mises_c1};

const GOLDEN: &str = include_str!("../fixtures/solids_law27_paths_golden.json");

/// Steps per leg, matching the derivation script.
const STEPS_PER_LEG: usize = 100;
/// Lateral bisection passes per axial step, matching the derivation.
const LATERAL_ITERATIONS: usize = 50;
/// Axial extent of the uniaxial, volumetric, and cycle paths: 0.10
/// opens brittle damage (onset 0.08) without reaching deletion (0.16).
const EMAX: f64 = 0.10;
/// Shear extent: 0.43 drives the rebuilt shear principals to +-0.11,
/// damaging the non-coaxial path without deleting it.
const EMAX_SHEAR: f64 = 0.43;
/// Tension extent: 0.18 carries the axial totals past deletion (0.16)
/// with margin for the damage-softening lag.
const EMAX_TENSION: f64 = 0.18;
/// Reference temperature in K: the virgin state of every path (the
/// card sets Cv = 0, so no leg heats).
const T_REF: f64 = 298.0;

/// Accumulated result of one driven strain path.
struct PathResult {
    sigma: [f64; 6],
    epbar: f64,
    yielding: Vec<bool>,
    epbar_stations: Vec<f64>,
    damage: [f64; 2],
    dmg_stations: Vec<f64>,
    ruptured: Vec<bool>,
    temperature: f64,
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
/// bracket), mirroring the derivation script exactly, including the
/// post-deletion convention: ruptured (or rupture-swath) probes read
/// identically zero, so the unconstrained lateral returns 0.
fn solve_lateral(law: &Law27, state: &Law27State, de: f64) -> f64 {
    if state.ruptured {
        return 0.0;
    }
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
    if !(blo < 0.0 && bhi > 0.0) {
        if blo == 0.0 || bhi == 0.0 {
            return 0.0;
        }
        panic!("lateral bracket failed for de={de}");
    }
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
    law: &Law27,
    state: &mut Law27State,
    steps: &[[f64; 6]],
    stations: &[usize],
) -> PathResult {
    let mut yielding = Vec::new();
    let mut epbar_stations = Vec::new();
    let mut dmg_stations = Vec::new();
    let mut ruptured = Vec::new();
    for (i, step) in steps.iter().enumerate() {
        let update = law.update(state, step);
        law.commit(state, &update);
        if stations.contains(&(i + 1)) {
            yielding.push(update.yielding);
            epbar_stations.push(state.inner.equiv_plastic_strain);
            dmg_stations.push(state.damage[0]);
            ruptured.push(state.ruptured);
        }
    }
    PathResult {
        sigma: state.inner.stress,
        epbar: state.inner.equiv_plastic_strain,
        yielding,
        epbar_stations,
        damage: state.damage,
        dmg_stations,
        ruptured,
        temperature: state.inner.temperature,
    }
}

/// Drives axial increments with the lateral solve, recording stations.
fn drive_axial(law: &Law27, state: &mut Law27State, des: &[f64], stations: &[usize]) -> PathResult {
    let mut yielding = Vec::new();
    let mut epbar_stations = Vec::new();
    let mut dmg_stations = Vec::new();
    let mut ruptured = Vec::new();
    for (i, &de) in des.iter().enumerate() {
        let dl = solve_lateral(law, state, de);
        let update = law.update(state, &[de, dl, dl, 0.0, 0.0, 0.0]);
        law.commit(state, &update);
        if stations.contains(&(i + 1)) {
            yielding.push(update.yielding);
            epbar_stations.push(state.inner.equiv_plastic_strain);
            dmg_stations.push(state.damage[0]);
            ruptured.push(state.ruptured);
        }
    }
    PathResult {
        sigma: state.inner.stress,
        epbar: state.inner.equiv_plastic_strain,
        yielding,
        epbar_stations,
        damage: state.damage,
        dmg_stations,
        ruptured,
        temperature: state.inner.temperature,
    }
}

#[test]
fn law27_paths_match_wolfram() {
    let doc: Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(doc["meta"]["protocol"], "solids-golden-v1");
    assert_eq!(doc["meta"]["units"], "SI");
    let law = Law27::steel_brittle();
    let stations_100 = vec![20, 40, 60, 80, 100];
    let stations_cycle = vec![80, 160, 240, 320, 400];

    // Uniaxial extension with solved lateral strain: damaged, intact.
    let de = EMAX / STEPS_PER_LEG as f64;
    let mut state = Law27State::new(T_REF);
    let uni = drive_axial(&law, &mut state, &vec![de; STEPS_PER_LEG], &stations_100);
    check_all(&doc, "uni_sigma", &uni.sigma);
    check_all(&doc, "uni_epbar", &[uni.epbar]);
    check_flags(&doc, "uni_yielding", &uni.yielding);
    check_all(&doc, "uni_epbar_st", &uni.epbar_stations);
    check_all(&doc, "uni_temp", &[uni.temperature]);
    check_all(&doc, "uni_damage", &uni.damage);
    check_all(&doc, "uni_dmg_st", &uni.dmg_stations);
    assert!(uni.yielding.iter().any(|&y| y), "uniaxial must yield");
    assert!(
        uni.damage[0] > 0.0 && uni.damage[0] < 1.0,
        "uniaxial must partially damage, got {}",
        uni.damage[0]
    );
    assert_eq!(uni.damage[1], 0.0);
    assert!(
        !uni.ruptured.iter().any(|&r| r),
        "uniaxial must stay intact"
    );
    assert_eq!(uni.temperature, T_REF);

    // Pure shear ramp: non-coaxial damage, intact.
    let dg = EMAX_SHEAR / STEPS_PER_LEG as f64;
    let mut state = Law27State::new(T_REF);
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
    check_all(&doc, "shear_temp", &[shear.temperature]);
    check_all(&doc, "shear_damage", &shear.damage);
    check_all(&doc, "shear_dmg_st", &shear.dmg_stations);
    assert!(shear.yielding.iter().any(|&y| y), "shear must yield");
    assert!(
        shear.damage[0] > 0.0 && shear.damage[0] < 1.0,
        "shear must partially damage, got {}",
        shear.damage[0]
    );
    assert_eq!(shear.damage[1], 0.0);
    assert!(!shear.ruptured.iter().any(|&r| r), "shear must stay intact");

    // Volumetric compression: elastic, damage is tension-only.
    let dv = -EMAX / STEPS_PER_LEG as f64;
    let mut state = Law27State::new(T_REF);
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
    check_all(&doc, "vol_temp", &[vol.temperature]);
    check_all(&doc, "vol_damage", &vol.damage);
    check_all(&doc, "vol_dmg_st", &vol.dmg_stations);
    assert!(
        vol.yielding.iter().all(|&y| !y),
        "volumetric must stay elastic"
    );
    assert_eq!(vol.epbar, 0.0);
    assert_eq!(vol.damage, [0.0, 0.0]);
    assert_eq!(vol.temperature, T_REF);
    assert!(
        !vol.ruptured.iter().any(|&r| r),
        "volumetric must stay intact"
    );

    // Symmetric strain cycle with solved lateral strain: damage follows
    // the totals back to zero; the reloading end sits on the surface.
    let mut des = vec![de; STEPS_PER_LEG];
    des.extend(vec![-de; 2 * STEPS_PER_LEG]);
    des.extend(vec![de; STEPS_PER_LEG]);
    let mut state = Law27State::new(T_REF);
    let cyc = drive_axial(&law, &mut state, &des, &stations_cycle);
    check_all(&doc, "cycle_sigma", &cyc.sigma);
    check_all(&doc, "cycle_epbar", &[cyc.epbar]);
    check_flags(&doc, "cycle_yielding", &cyc.yielding);
    check_all(&doc, "cycle_epbar_st", &cyc.epbar_stations);
    check_all(&doc, "cycle_temp", &[cyc.temperature]);
    check_all(&doc, "cycle_damage", &cyc.damage);
    check_all(&doc, "cycle_dmg_st", &cyc.dmg_stations);
    assert!(cyc.yielding.iter().any(|&y| y), "cycle must yield");
    assert!(!cyc.ruptured.iter().any(|&r| r), "cycle must stay intact");
    let closure = von_mises_c1(&cyc.sigma)
        - law
            .base
            .flow_stress(cyc.epbar, law.base.reference_strain_rate, cyc.temperature);
    check_all(&doc, "cycle_closure", &[closure]);
    assert!(
        closure.abs() < 1.0,
        "no thermal lag: cycle must end on the surface, got {closure:.3e}"
    );

    // Solved-lateral tension past deletion: stress zeroed, latch set.
    let dten = EMAX_TENSION / STEPS_PER_LEG as f64;
    let mut state = Law27State::new(T_REF);
    let tension = drive_axial(&law, &mut state, &vec![dten; STEPS_PER_LEG], &stations_100);
    check_all(&doc, "tension_sigma", &tension.sigma);
    check_all(&doc, "tension_epbar", &[tension.epbar]);
    check_flags(&doc, "tension_yielding", &tension.yielding);
    check_all(&doc, "tension_epbar_st", &tension.epbar_stations);
    check_all(&doc, "tension_temp", &[tension.temperature]);
    check_all(&doc, "tension_damage", &tension.damage);
    check_all(&doc, "tension_dmg_st", &tension.dmg_stations);
    check_flags(&doc, "tension_ruptured", &tension.ruptured);
    assert_eq!(tension.sigma, [0.0; 6]);
    assert_eq!(tension.damage, [0.999, 0.0]);
    assert!(
        tension.ruptured.last() == Some(&true),
        "tension must delete by the final station"
    );
    assert!(
        tension.ruptured.first() == Some(&false),
        "tension must start intact"
    );
}
